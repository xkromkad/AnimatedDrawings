# Predškoláčik animator: AnimatedDrawings + its two models in one CPU-only
# container, for the Oracle Always Free Arm VM (also builds on amd64).
# Build context is the repo root:
#
#   docker build -f service/Dockerfile -t predskolacik-animator .
#
# No TorchServe (archived upstream in 2025): service/models.py loads the
# detector and pose estimator from the release .mar files directly.
#
# The OpenMMLab 1.x stack the models were trained with has no aarch64 wheels
# for mmcv-full, so the builder stage compiles 1.7.0, the newest mmpose 0.29 accepts (CPU ops only, ~15-30 min
# on Ampere; Docker caches the layer afterwards).

ARG PYTHON=3.10

FROM python:${PYTHON}-slim-bookworm AS base
ARG TARGETARCH
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# torch 2.0 needs numpy<2; mmcv 1.x needs pkg_resources and yapf<=0.40.1
RUN pip install "setuptools<70" wheel "numpy==1.24.4" \
 && if [ "$TARGETARCH" = "amd64" ]; then \
      pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install torch==2.0.1 torchvision==0.15.2; \
    fi

FROM base AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ninja-build git \
 && rm -rf /var/lib/apt/lists/*
# MMCV_WITH_OPS=1: the detector (Mask R-CNN) needs RoIAlign and NMS.
RUN pip install Cython \
 && MMCV_WITH_OPS=1 FORCE_CUDA=0 MAX_JOBS=4 \
    pip wheel --no-build-isolation --no-deps -w /wheels mmcv-full==1.7.0 \
 && pip wheel --no-build-isolation --no-deps -w /wheels xtcocotools==1.14.3 pycocotools==2.0.7

# The release models (MIT): a .mar is a zip of config + checkpoint + handler.
ADD https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_detector.mar /tmp/det.mar
ADD https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_pose_estimator.mar /tmp/pose.mar
RUN python -c "import zipfile; zipfile.ZipFile('/tmp/det.mar').extractall('/models/detector'); zipfile.ZipFile('/tmp/pose.mar').extractall('/models/pose')" \
 && rm /tmp/*.mar

FROM base
RUN apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /wheels /wheels
# The wheels' own dependencies are unpinned (mmcv-full would pull numpy 2,
# which torch 2.0 cannot load); the constraints hold every install to the
# versions the models were built with.
RUN printf '%s\n' "numpy==1.24.4" "matplotlib<3.9" "opencv-python-headless==4.8.1.78" \
      "Pillow==10.1.0" "scipy==1.10.1" "torch==2.0.1" > /constraints.txt
ENV PIP_CONSTRAINT=/constraints.txt
RUN pip install /wheels/*.whl \
 # OpenMMLab without their pinned extras (chumpy, which mmpose only needs
 # for SMPL, no longer builds).
 && pip install --no-deps mmdet==2.28.2 mmpose==0.29.0 \
 && pip install addict "yapf==0.40.1" packaging terminaltables six json_tricks munkres \
      "matplotlib<3.9" \
 # AnimatedDrawings' own pins (setup.py), minus torchserve, glfw and Flask
      "scipy==1.10.1" "scikit-image==0.19.3" "scikit-learn==1.1.3" "shapely==1.8.5.post1" \
      "opencv-python-headless==4.8.1.78" "Pillow==10.1.0" "PyOpenGL==3.1.6" "PyYAML==6.0.1" "tqdm==4.66.3" \
 # the API
      "fastapi==0.110.3" "uvicorn==0.29.0" "python-multipart==0.0.9" \
 && rm -rf /wheels

COPY --from=build /models /models
WORKDIR /app
COPY animated_drawings ./animated_drawings
COPY examples/config ./examples/config
COPY examples/bvh ./examples/bvh
COPY service ./service

ENV PYTHONPATH=/app:/app/service \
    PYOPENGL_PLATFORM=osmesa \
    ANIMATOR_MODELS=/models \
    ANIMATOR_WORK=/tmp \
    OMP_NUM_THREADS=2

RUN useradd --uid 10001 --create-home animator \
 && mkdir -p /tmp/animator && chown animator /tmp/animator
USER animator

WORKDIR /app/service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]
