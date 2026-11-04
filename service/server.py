"""HTTP API of the Predškoláčik animator (Kresli a tancuj).

  POST /animate?motion=dab   multipart field `image` (PNG)  -> image/gif
       401 bad key · 422 {"code":"no_figure"} · 503 {"code":"busy"}
  GET  /health

Only the backend calls it (header X-Animator-Key = ANIMATOR_KEY). Detection
and pose run in this process, one at a time; each render is a subprocess
(render_job.py). Annotations are cached by image hash for an hour, so asking
for a second dance of the same drawing skips the models.
"""

from __future__ import annotations

import asyncio
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
from fastapi import FastAPI, File, Header, Query, UploadFile
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
MAX_IMAGE_BYTES = 5 * 1024 * 1024
CACHE_TTL_S = 3600

WORK = Path(os.environ.get('ANIMATOR_WORK', tempfile.gettempdir())) / 'animator'
CACHE = WORK / 'cache'
HERE = Path(__file__).resolve().parent

# motion -> retarget config that fits its BVH skeleton (as in the upstream examples)
MOTIONS = {
    'dab': 'fair1_ppf',
    'jumping': 'fair1_ppf',
    'wave_hello': 'fair1_ppf',
    'zombie': 'fair1_ppf',
    'jumping_jacks': 'cmu1_pfp',
    'jesse_dance': 'mixamo_fff',
}
DEFAULT_MOTION = 'dab'
FORMATS = {'gif': 'image/gif', 'webp': 'image/webp'}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
models: Models | None = None
model_lock = threading.Lock()
slots = asyncio.Semaphore(WORKERS)
waiting = 0


def error(status: int, code: str, **headers: str) -> JSONResponse:
    return JSONResponse({'code': code}, status_code=status, headers=headers or None)


@app.on_event('startup')
def load_models() -> None:
    global models
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    models = Models.load()
    log.info('models loaded in %.1f s', time.monotonic() - started)


@app.get('/health')
def health() -> dict:
    return {'ok': models is not None}


def prune_cache() -> None:
    cutoff = time.time() - CACHE_TTL_S
    for d in CACHE.iterdir():
        try:
            if d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except FileNotFoundError:
            pass


def annotate(data: bytes) -> Path:
    """The annotation dir of this drawing (cached). Raises NoFigure."""
    digest = hashlib.sha256(data).hexdigest()[:32]
    anno = CACHE / digest
    if (anno / 'char_cfg.yaml').exists():
        os.utime(anno)
        return anno
    if (anno / 'no_figure').exists():
        raise NoFigure('cached')

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    tmp = Path(tempfile.mkdtemp(dir=CACHE, prefix='.tmp-'))
    try:
        with model_lock:
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


async def render(anno: Path, motion: str, fmt: str) -> bytes:
    cached = anno / f'{motion}.{fmt}'
    if cached.exists():
        return cached.read_bytes()
    with tempfile.TemporaryDirectory(dir=WORK) as out_dir:
        out = Path(out_dir) / f'video.{fmt}'
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(HERE / 'render_job.py'), str(anno), motion, MOTIONS[motion], str(out),
            # The motion configs name their BVH files relative to the repo root.
            cwd=str(HERE.parent), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), RENDER_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError('render timed out')
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f'render failed ({proc.returncode}): {output.decode(errors="replace")[-2000:]}')
        data = out.read_bytes()
    cached.write_bytes(data)
    return data


@app.post('/animate')
async def animate(
    image: UploadFile = File(...),
    motion: str = Query(DEFAULT_MOTION),
    format: str = Query('gif'),
    x_animator_key: str = Header(''),
):
    global waiting
    if KEY and not hmac.compare_digest(x_animator_key.encode(), KEY.encode()):
        return error(401, 'unauthorized')
    if motion not in MOTIONS:
        return error(400, 'bad_motion')
    if format not in FORMATS:
        return error(400, 'bad_format')
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data or len(data) > MAX_IMAGE_BYTES:
        return error(400, 'bad_image')

    if waiting >= MAX_WAITING:
        return error(503, 'busy', **{'Retry-After': '20'})
    waiting += 1
    try:
        await asyncio.wait_for(slots.acquire(), SLOT_WAIT_S)
    except asyncio.TimeoutError:
        return error(503, 'busy', **{'Retry-After': '20'})
    finally:
        waiting -= 1

    started = time.monotonic()
    try:
        prune_cache()
        anno = await asyncio.to_thread(annotate, data)
        t_anno = time.monotonic() - started
        video = await render(anno, motion, format)
        log.info('animated (%s, %s) in %.1f s (annotate %.1f s), %d kB',
                 motion, format, time.monotonic() - started, t_anno, len(video) // 1024)
        return Response(video, media_type=FORMATS[format])
    except NoFigure as e:
        log.info('no figure: %s', e)
        return error(422, 'no_figure')
    except Exception:
        log.exception('animation failed')
        return error(500, 'failed')
    finally:
        slots.release()
