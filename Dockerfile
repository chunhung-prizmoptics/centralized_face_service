# ---------------------------------------------------------------------------
# Centralized GPU Face Recognition Service
# ---------------------------------------------------------------------------
# Base: CUDA 12.4 + cuDNN runtime + Ubuntu 22.04
#       matches onnxruntime-gpu 1.23 (requires CUDA 12.x)
#       Host driver must be >= 550 to support CUDA 12.4
# Build:  docker build -t face-recognition-service .
# Run:    see docker-compose.yml
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# -- system deps ----------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-distutils \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# -- Python deps (layer-cached unless requirements.txt changes) ----------
COPY requirements.txt .

# onnxruntime-gpu 1.23 requires CUDA 12 wheels from the official index
RUN pip3 install --no-cache-dir -r requirements.txt

# -- application code -----------------------------------------------------
COPY . .

# Pre-download InsightFace buffalo_l model at build time so the container
# starts immediately without a network download on first run.
# The model is stored in ~/.insightface/models/ inside the image.
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1, det_size=(640,640)); \
print('buffalo_l downloaded.')"

# -- runtime defaults ------------------------------------------------------
# All of these can be overridden via environment variables or docker-compose.
ENV PYTHONNOUSERSITE=1 \
    REDIS_URL=redis://redis:6379 \
    GALLERY_DIR=/data/gallery \
    CACHE_PATH=/data/face_db_cache.npz \
    WORKERS=2 \
    BATCH_SIZE=16 \
    MATCH_THRESHOLD=0.40 \
    GPU_ID=0 \
    NO_REID=1 \
    API_ENABLED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    TRACK_CACHE_TTL=30 \
    TRACK_CACHE_RECHECK=10 \
    TRACK_CACHE_UNKNOWN_RECHECK=3 \
    FACE_CROP_TTL=60

EXPOSE 8000

# /data is the single mutable mount:
#   /data/gallery/   → identity folders (source of truth)
#   /data/face_db_cache.npz     → embedding cache
#   /data/face_db_cache.manifest.json → manifest
VOLUME ["/data"]

ENTRYPOINT ["python3", "main.py"]
CMD ["--api"]
