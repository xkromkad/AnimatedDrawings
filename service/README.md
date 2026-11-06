# Predškoláčik animator

The backend of "Kresli a tancuj" (predskolacik.app, `/hry/kreslenie-postavy/hrat`): a child's drawing goes in and a dancing animation comes out.

```
app ──► predskolacik_backend  POST api/drawing/animate  (auth, size limit)
         └─► https://animator.predskolacik.app/animate   (X-Animator-Key)
              Caddy on the Oracle VM ─► this container :8000
```

## Timings

On the Oracle VM (1 Ampere core):

| | |
|---|---|
| New drawing, first dance | ~21 s (detection + pose ~9 s, render ~10 s) |
| Another dance, not rendered yet | 6–10 s |
| Another dance, already pre-rendered | instant |

## Files

**`server.py`** is the FastAPI app.

- `POST /animate?motion=<name>&format=gif|webp` takes a multipart `image`. It returns `image/gif` (the default) or an animated `image/webp`. The app asks for WebP: it has real alpha around the figure and is much smaller.
- `GET /health`.
- **Errors:** `401` for a bad key, `422 {"code":"no_figure"}`, and `503 {"code":"busy"}` with `Retry-After`.
- **Caching:** the rig is cached by image hash for an hour, and every rendered dance sits next to it. A dance that's already there is answered at once, without queueing.
- **Abandoned requests:** a request whose client leaves (the child picked another dance) stops waiting, and its render is killed. Detection that's already running still finishes and stays cached, so the next request for that drawing skips it.
- **Pre-rendering:** after a drawing's first dance, the other dances render in the background. They run one at a time, at nice 19, and only while nobody is waiting. A child's request pre-empts them, unless it asks for the very dance being pre-rendered, in which case it waits for it. Set `ANIMATOR_PRERENDER=0` to switch this off.
- **Concurrency:** one child is served at a time (`ANIMATOR_WORKERS`). Up to `ANIMATOR_MAX_WAITING` (3) more wait for at most `ANIMATOR_SLOT_WAIT` s (40), so no request outlives Cloudflare's 100 s limit. Only one render subprocess runs at a time.

**`annotate.py`** is `examples/image_to_annotations.py` with the models called in-process and `NoFigure` instead of asserts.

- A drawing also counts as `NoFigure` when its mean keypoint score is below 0.2.
- The detector's box gets an 8% margin.

**`models.py`** holds the detector (mmdet Mask R-CNN) and the pose estimator (mmpose top-down ResNet-50).

- Both load from the upstream v0.0.1 `.mar` files, as their TorchServe handlers did. **There's no TorchServe**: it was archived in Aug 2025.
- Detection runs at 800×480 (`ANIMATOR_DET_SCALE`) instead of 1333×800. On drawings the scores and boxes match, and it takes 7 s instead of 18 s on the VM.

**`render_job.py`** runs one OSMesa render per subprocess.

- It keeps every n-th frame of the motion clip, for ~15 fps and at most 10 s that loop. The clips themselves run at 30 or 100 fps for up to 28 s.
- It caches the joint→triangle map per drawing.
- It writes lossless WebP quantised to 256 colours.

**Motions:** `dab` (the default), `jesse_dance`, `jumping`, `jumping_jacks`, `wave_hello` and `zombie`.

## Changes to upstream code

`animated_drawings/model/arap.py`:

- **Factorised once:** the two fixed system matrices are factorised once instead of in every `spsolve` (three LU factorisations per frame).
- **No determinant check:** the `np.linalg.det(...) == 0.0` singularity test is gone. It took 14 s per render on the VM, and on a ~5000² matrix it underflows anyway. The LU factorisation reports real singularity, and the same 1e-8 nudge applies only then.
- **Vectorised:** the edge rotation runs on whole arrays instead of a per-edge loop.

Renders come out pixel-identical, and `tests/test_arap.py` passes.

## Build and run

```bash
docker build -f service/Dockerfile -t predskolacik-animator .   # from the repo root
docker run --rm -p 8000:8000 -e ANIMATOR_KEY=dev predskolacik-animator
curl -H "X-Animator-Key: dev" -F image=@examples/drawings/postava.png \
     "localhost:8000/animate?motion=dab&format=webp" -o out.webp
```

On the VM, this is the `animator` service of `predskolacik-cms/deploy/docker-compose.yml`: profile `animator`, capped at 3 GB and `ANIMATOR_CPUS` (0.85) next to Strapi. The image is built for arm64 elsewhere and `docker load`ed. A code-only change can be layered on top of the running image instead: `FROM predskolacik-animator:prev` plus `COPY` of the changed files.

The first build compiles `mmcv-full` 1.7.0, the newest mmpose 0.29 accepts, because no aarch64 wheel exists. That takes roughly 15–30 min on Ampere (much longer under QEMU), and later builds reuse the cached layer.
