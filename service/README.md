# Predškoláčik animator

The backend of "Kresli a tancuj" (predskolacik.app, `/hry/kreslenie-postavy/hrat`): a child's drawing in, a dancing GIF out.

```
app ──► predskolacik_backend  POST api/drawing/animate  (auth, size limit)
         └─► https://animator.predskolacik.app/animate   (X-Animator-Key)
              Caddy on the Oracle VM ─► this container :8000
```

- `server.py`: FastAPI. `POST /animate?motion=<name>&format=gif|webp` takes multipart `image` and returns `image/gif` (default) or an animated `image/webp`. The app asks for WebP: real alpha around the figure, and much smaller files. `GET /health`.
  - Each rendered dance is cached next to the drawing's annotations, so asking again is instant.
  - Errors: `401` bad key, `422 {"code":"no_figure"}`, `503 {"code":"busy"}` with `Retry-After`.
  - Concurrency: `ANIMATOR_WORKERS` renders at once (default 1), up to `ANIMATOR_MAX_WAITING` (3) waiting at most `ANIMATOR_SLOT_WAIT` s (40). Everything else gets "busy", so no request outlives Cloudflare's 100 s limit.
  - Annotations are cached by image hash for an hour, so a second motion of the same drawing only re-renders.
- `annotate.py`: `examples/image_to_annotations.py` with the models called in-process and `NoFigure` instead of asserts. A drawing also counts as `NoFigure` when the mean keypoint score is below 0.2.
- `models.py`: the detector (mmdet Mask R-CNN) and pose estimator (mmpose top-down ResNet-50). Both are loaded from the upstream v0.0.1 `.mar` files, as their TorchServe handlers did. **No TorchServe**: it was archived in Aug 2025.
- `render_job.py`: one OSMesa render per subprocess.
- Motions: `dab` (default), `jumping`, `wave_hello`, `zombie`, `jumping_jacks`, `jesse_dance`.

## Build and run

```bash
docker build -f service/Dockerfile -t predskolacik-animator .   # from the repo root
docker run --rm -p 8000:8000 -e ANIMATOR_KEY=dev predskolacik-animator
curl -H "X-Animator-Key: dev" -F image=@examples/drawings/postava.png \
     "localhost:8000/animate?motion=dab" -o out.gif
```

On the VM the service is the `animator` service of `predskolacik-cms/deploy/docker-compose.yml` (profile `animator`, capped at 3 GB / 2 CPUs next to Strapi).

The first build compiles `mmcv-full` 1.7.0 (the newest mmpose 0.29 accepts), because no aarch64 wheel exists. That takes roughly 15–30 min on the Ampere VM, and later builds reuse the cached layer.
