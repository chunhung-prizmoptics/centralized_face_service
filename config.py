"""
config.py – All service settings with sensible defaults and env-var overrides.

Edit the defaults below to avoid passing CLI arguments every run.
All values can also be overridden at runtime via environment variables.
"""

import os

# -----------------------------------------------------------------------
# Redis
# -----------------------------------------------------------------------
REDIS_URL       = os.getenv("REDIS_URL",        "redis://localhost:6379")
REQUEST_STREAM  = os.getenv("REQUEST_STREAM",   "vision:track_events")
RESULT_STREAM   = os.getenv("RESULT_STREAM",    "face_inference:results")
CONSUMER_GROUP  = os.getenv("CONSUMER_GROUP",   "face_inference_workers")
CONSUMER_NAME   = os.getenv("CONSUMER_NAME",    "worker_0")

# -----------------------------------------------------------------------
# Batching / workers
# -----------------------------------------------------------------------
BATCH_SIZE       = int(os.getenv("BATCH_SIZE",       "16"))
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", "50"))   # ms to wait before flushing a partial batch
WORKERS          = int(os.getenv("WORKERS",          "2"))    # parallel BatchWorker threads
DECODE_WORKERS   = int(os.getenv("DECODE_WORKERS",   "4"))    # CPU threads for JPEG decode per worker

# -----------------------------------------------------------------------
# Stream size limits
# -----------------------------------------------------------------------
RESULT_MAXLEN   = int(os.getenv("RESULT_MAXLEN",  "10000"))  # max entries in result stream
REQUEST_MAXLEN  = int(os.getenv("REQUEST_MAXLEN", "1000"))   # max JPEG crops kept in request stream

# -----------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.35"))

# -----------------------------------------------------------------------
# Gallery / cache
# -----------------------------------------------------------------------
GALLERY_DIR = os.getenv("GALLERY_DIR", "gallery")
CACHE_PATH  = os.getenv("CACHE_PATH",  "face_db_cache.npz")

# -----------------------------------------------------------------------
# InsightFace
# -----------------------------------------------------------------------
DET_SIZE = int(os.getenv("DET_SIZE", "640"))
GPU_ID   = int(os.getenv("GPU_ID",   "0"))

# When True: run full detection pipeline (SCRFD + alignment + ArcFace).
# When False (default): skip detection — use pre-cropped inputs with landmark_5_xy alignment.
RUN_DETECTION = os.getenv("RUN_DETECTION", "0") == "1"

# -----------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------
NO_REID = os.getenv("NO_REID", "1") == "1"   # set "0" to enable ReID embeddings
