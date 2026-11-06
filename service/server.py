"""HTTP API of the Predškoláčik animator (Kresli a tancuj).

  POST /animate?motion=dab&format=webp[&scene=luka]   multipart field `image` (PNG)
       -> image/gif (default) or image/webp
       401 bad key · 422 {"code":"no_figure"} · 503 {"code":"busy"}
  GET  /health

Only the backend calls it (header X-Animator-Key = ANIMATOR_KEY).

Work per drawing: detection + pose (in this process, one at a time, ~9 s on
the VM) produce the rig, cached by image hash; each dance is then a render
subprocess (render_job.py, ~10-15 s), cached next to the rig. On top:

- A request whose client went away (the child picked another dance, or
  closed the stage) stops: its render is killed and the slot freed at once.
  Detection already running finishes in its thread and stays cached, so the
  next request for the same drawing skips it.
- After a drawing's first dance, the other dances render in the background,
  one at a time at the lowest CPU priority and only while no child is
  waiting, so switching dances is usually instant. A child's request
  pre-empts it (the background render is killed and re-queued), unless it is
  the very dance being pre-rendered, which the request then simply awaits.
- Only one render subprocess runs at a time (memory on the 6 GB VM).
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from annotate import NoFigure, image_to_annotations
from models import Models

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('animator')

KEY = os.environ.get('ANIMATOR_KEY', '')
WORKERS = max(1, int(os.environ.get('ANIMATOR_WORKERS', '1')))
# Requests allowed to wait for a free slot; the rest get "busy" at once.
MAX_WAITING = int(os.environ.get('ANIMATOR_MAX_WAITING', '3'))
# Cloudflare cuts a request at 100 s, so a request that cannot start within
# this is better answered "busy" than left to time out.
SLOT_WAIT_S = float(os.environ.get('ANIMATOR_SLOT_WAIT', '40'))
RENDER_TIMEOUT_S = float(os.environ.get('ANIMATOR_RENDER_TIMEOUT', '90'))
PRERENDER = os.environ.get('ANIMATOR_PRERENDER', '1') != '0'
MAX_IMAGE_BYTES = 5 * 1024 * 1024
CACHE_TTL_S = 3600

WORK = Path(os.environ.get('ANIMATOR_WORK', tempfile.gettempdir())) / 'animator'
CACHE = WORK / 'cache'
HERE = Path(__file__).resolve().parent

# motion -> retarget config that fits its BVH skeleton (as in the upstream
# examples). Order = the app's dance picker, which is also the background
# pre-render order.
MOTIONS = {
    'dab': 'fair1_ppf',
    'jesse_dance': 'mixamo_fff',
    'jumping': 'fair1_ppf',
    'jumping_jacks': 'cmu1_pfp',
    'wave_hello': 'fair1_ppf',
    'zombie': 'fair1_ppf',
}
DEFAULT_MOTION = 'dab'
FORMATS = {'gif': 'image/gif', 'webp': 'image/webp'}
# Backdrops for downloads (exported from the app's CSS by
# scripts/export-dance-scenes.mjs in the predskolacik repo).
SCENES_DIR = HERE / 'scenes'
SCENES = {p.stem for p in SCENES_DIR.glob('*.png')}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
models: Models | None = None
model_lock = threading.Lock()
slots = asyncio.Semaphore(WORKERS)
waiting = 0
# Requests holding a slot (detecting or rendering for a child).
active = 0


class ClientGone(Exception):
    """The caller disconnected; nobody wants the result any more."""


def error(status: int, code: str, **headers: str) -> JSONResponse:
    return JSONResponse({'code': code}, status_code=status, headers=headers or None)


@app.on_event('startup')
def load_models() -> None:
    global models
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    models = Models.load()
    log.info('models loaded in %.1f s', time.monotonic() - started)
    if PRERENDER:
        asyncio.get_event_loop().create_task(prerender_worker())


@app.get('/health')
def health() -> dict:
    return {'ok': models is not None}


def prune_cache() -> None:
    cutoff = time.time() - CACHE_TTL_S
    for d in CACHE.iterdir():
        try:
            if d.stat().st_mtime < cutoff and not d.name.startswith('.tmp-'):
                shutil.rmtree(d, ignore_errors=True)
        except FileNotFoundError:
            pass


def anno_dir(data: bytes) -> Path:
    return CACHE / hashlib.sha256(data).hexdigest()[:32]


def annotate(data: bytes) -> Path:
    """The annotation dir of this drawing (cached). Raises NoFigure."""
    anno = anno_dir(data)

    def cached() -> bool:
        if (anno / 'char_cfg.yaml').exists():
            os.utime(anno)
            return True
        if (anno / 'no_figure').exists():
            raise NoFigure('cached')
        return False

    if cached():
        return anno
    with model_lock:
        # A request for this drawing that was abandoned mid-detection may
        # have finished it while we waited for the lock.
        if cached():
            return anno
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        tmp = Path(tempfile.mkdtemp(dir=CACHE, prefix='.tmp-'))
        try:
            assert models is not None
            image_to_annotations(img, tmp, models)
        except NoFigure:
            shutil.rmtree(tmp, ignore_errors=True)
            anno.mkdir(exist_ok=True)
            (anno / 'no_figure').touch()
            raise
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        shutil.rmtree(anno, ignore_errors=True)
        tmp.rename(anno)
    return anno


async def until_gone(request: Request | None) -> None:
    """Returns when the client disconnects (never without a request)."""
    if request is None:
        await asyncio.Event().wait()
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)


async def unless_gone(awaitable, request: Request | None):
    """Await `awaitable`, or raise ClientGone if the client leaves first.

    The awaitable itself is not cancelled: an abandoned detection still
    completes and lands in the cache.
    """
    work = asyncio.ensure_future(awaitable)
    gone = asyncio.ensure_future(until_gone(request))
    done, _ = await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
    gone.cancel()
    if work in done:
        return work.result()
    raise ClientGone


# --- Rendering ----------------------------------------------------------------

class RenderJob:
    def __init__(self, anno: Path, motion: str, fmt: str, background: bool):
        self.anno, self.motion, self.fmt = anno, motion, fmt
        self.background = background
        self.proc: asyncio.subprocess.Process | None = None
        self.result: asyncio.Future[bytes] = asyncio.get_event_loop().create_future()

    @property
    def key(self) -> tuple[str, str, str]:
        return (str(self.anno), self.motion, self.fmt)

    @property
    def output(self) -> Path:
        return self.anno / f'{self.motion}.{self.fmt}'

    def kill(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.kill()


# The one render subprocess that may run at a time.
running: RenderJob | None = None
render_lock = asyncio.Lock()


async def run_render(job: RenderJob) -> bytes:
    """Render `job` (render_lock held by the caller); cache and return it."""
    global running
    with tempfile.TemporaryDirectory(dir=WORK) as out_dir:
        out = Path(out_dir) / f'video.{job.fmt}'
        job.proc = await asyncio.create_subprocess_exec(
            sys.executable, str(HERE / 'render_job.py'),
            str(job.anno), job.motion, MOTIONS[job.motion], str(out),
            # The motion configs name their BVH files relative to the repo root.
            cwd=str(HERE.parent), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            preexec_fn=(lambda: os.nice(19)) if job.background else None,
        )
        running = job
        try:
            output, _ = await asyncio.wait_for(job.proc.communicate(), RENDER_TIMEOUT_S)
        except asyncio.TimeoutError:
            job.kill()
            await job.proc.wait()
            raise RuntimeError('render timed out')
        finally:
            running = None
        if job.proc.returncode != 0 or not out.exists():
            if job.proc.returncode is not None and job.proc.returncode < 0:
                raise asyncio.CancelledError  # killed: pre-empted or abandoned
            raise RuntimeError(f'render failed ({job.proc.returncode}): '
                               f'{output.decode(errors="replace")[-2000:]}')
        data = out.read_bytes()
    tmp = job.output.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, job.output)
    return data


async def render_for_child(anno: Path, motion: str, fmt: str, request: Request | None) -> bytes:
    cached = anno / f'{motion}.{fmt}'
    if cached.exists():
        return cached.read_bytes()

    # The dance is being pre-rendered right now: wait for it rather than
    # start it again.
    bg = running
    if bg is not None and bg.key == (str(anno), motion, fmt):
        try:
            return await unless_gone(asyncio.shield(bg.result), request)
        except asyncio.CancelledError:
            pass  # it was pre-empted after all; render it ourselves

    # Otherwise any background render gives way.
    if running is not None and running.background:
        running.kill()

    job = RenderJob(anno, motion, fmt, background=False)
    async with render_lock:
        if cached.exists():
            return cached.read_bytes()
        task = asyncio.ensure_future(run_render(job))
        try:
            data = await unless_gone(asyncio.shield(task), request)
        except ClientGone:
            job.kill()
            await asyncio.gather(task, return_exceptions=True)
            raise
        job.result.set_result(data)
        return data


def with_scene(dance: Path, scene: str) -> bytes:
    """The (transparent) dance composited onto a scene, cached next to it.

    For the download: the stage shows the dance over a CSS backdrop, a saved
    file should look the same. Opaque frames, so lossy WebP is the smaller
    choice here.
    """
    from PIL import Image, ImageSequence

    out = dance.with_name(f'{dance.stem}.{scene}.webp')
    if out.exists():
        return out.read_bytes()
    anim = Image.open(dance)
    backdrop = Image.open(SCENES_DIR / f'{scene}.png').convert('RGBA').resize(anim.size, Image.LANCZOS)
    frames, durations = [], []
    for frame in ImageSequence.Iterator(anim):
        composed = backdrop.copy()
        composed.alpha_composite(frame.convert('RGBA'))
        frames.append(composed.convert('RGB'))
        durations.append(frame.info.get('duration', anim.info.get('duration', 66)))
    tmp = out.with_suffix(f'.{os.getpid()}.tmp')
    frames[0].save(tmp, format='WEBP', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, quality=80, method=4)
    os.replace(tmp, out)
    return out.read_bytes()


# --- Background pre-rendering ---------------------------------------------------------

prerender_queue: collections.deque[tuple[Path, str, str]] = collections.deque()
prerender_wakeup = asyncio.Event()


def queue_prerender(anno: Path, fmt: str, done: str) -> None:
    if not PRERENDER:
        return
    for motion in MOTIONS:
        item = (anno, motion, fmt)
        if motion != done and item not in prerender_queue and not (anno / f'{motion}.{fmt}').exists():
            prerender_queue.append(item)
    prerender_wakeup.set()


async def prerender_worker() -> None:
    while True:
        if not prerender_queue:
            prerender_wakeup.clear()
            await prerender_wakeup.wait()
            continue
        # Children first: wait while anyone is waiting or being served.
        if active or waiting or render_lock.locked():
            await asyncio.sleep(1)
            continue
        anno, motion, fmt = prerender_queue[0]
        if not anno.exists() or (anno / f'{motion}.{fmt}').exists():
            prerender_queue.popleft()
            continue
        job = RenderJob(anno, motion, fmt, background=True)
        started = time.monotonic()
        async with render_lock:
            try:
                data = await run_render(job)
            except asyncio.CancelledError:
                job.result.cancel()
                log.info('pre-render of %s pre-empted', motion)
                await asyncio.sleep(1)
                continue  # stays first in the queue
            except Exception:
                log.exception('pre-render of %s failed', motion)
                job.result.cancel()
                prerender_queue.popleft()
                continue
        prerender_queue.popleft()
        job.result.set_result(data)
        log.info('pre-rendered %s in %.1f s', motion, time.monotonic() - started)


# --- API --------------------------------------------------------------------------------

@app.post('/animate')
async def animate(
    request: Request,
    image: UploadFile = File(...),
    motion: str = Query(DEFAULT_MOTION),
    format: str = Query('gif'),
    scene: str | None = Query(None),
    x_animator_key: str = Header(''),
):
    global waiting, active
    if KEY and not hmac.compare_digest(x_animator_key.encode(), KEY.encode()):
        return error(401, 'unauthorized')
    if motion not in MOTIONS:
        return error(400, 'bad_motion')
    if format not in FORMATS:
        return error(400, 'bad_format')
    if scene is not None and (scene not in SCENES or format != 'webp'):
        return error(400, 'bad_scene')
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data or len(data) > MAX_IMAGE_BYTES:
        return error(400, 'bad_image')

    async def answer(dance: Path, video: bytes) -> Response:
        if scene is not None:
            video = await asyncio.to_thread(with_scene, dance, scene)
        return Response(video, media_type=FORMATS[format])

    # Already rendered (by an earlier request or in the background): no
    # queue, no slot.
    ready = anno_dir(data) / f'{motion}.{format}'
    if ready.exists():
        os.utime(ready.parent)
        return await answer(ready, ready.read_bytes())

    if waiting >= MAX_WAITING:
        return error(503, 'busy', **{'Retry-After': '20'})
    waiting += 1
    try:
        got = await acquire_slot(request)
    finally:
        waiting -= 1
    if got is None:
        return Response(status_code=499)  # left while queued
    if not got:
        return error(503, 'busy', **{'Retry-After': '20'})

    active += 1
    started = time.monotonic()
    try:
        prune_cache()
        anno = await unless_gone(asyncio.to_thread(annotate, data), request)
        t_anno = time.monotonic() - started
        video = await render_for_child(anno, motion, format, request)
        log.info('animated (%s, %s) in %.1f s (annotate %.1f s), %d kB',
                 motion, format, time.monotonic() - started, t_anno, len(video) // 1024)
        queue_prerender(anno, format, motion)
        return await answer(anno / f'{motion}.{format}', video)
    except ClientGone:
        log.info('client left during %s after %.1f s', motion, time.monotonic() - started)
        return Response(status_code=499)
    except NoFigure as e:
        log.info('no figure: %s', e)
        return error(422, 'no_figure')
    except Exception:
        log.exception('animation failed')
        return error(500, 'failed')
    finally:
        active -= 1
        slots.release()



async def acquire_slot(request: Request) -> bool | None:
    """True with a slot held, False after SLOT_WAIT_S, None if the client left."""
    acquire = asyncio.ensure_future(asyncio.wait_for(slots.acquire(), SLOT_WAIT_S))
    gone = asyncio.ensure_future(until_gone(request))
    done, _ = await asyncio.wait({acquire, gone}, return_when=asyncio.FIRST_COMPLETED)
    gone.cancel()
    if acquire in done:
        try:
            return acquire.result()
        except asyncio.TimeoutError:
            return False
    acquire.cancel()
    try:
        if await acquire:
            slots.release()  # granted just as the client left
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    return None
