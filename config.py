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
REQUEST_STREAM  = os.getenv("REQUEST_STREAM",   "face_inference:requests")
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
RESULT_MAXLEN   = int(os.getenv("RESULT_MAXLEN",  "100000"))  # max entries in result stream (no embeddings → small msgs)
REQUEST_MAXLEN  = int(os.getenv("REQUEST_MAXLEN", "1000"))    # max JPEG crops kept in request stream

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

# Enrollment API (runtime add/delete identities)
API_ENABLED = os.getenv("API_ENABLED", "0") == "1"
API_HOST    = os.getenv("API_HOST", "127.0.0.1")
API_PORT    = int(os.getenv("API_PORT", "8000"))

# -----------------------------------------------------------------------
# Deadline filtering
# -----------------------------------------------------------------------
# Maximum age (ms) past deadline_ts_ms before a message is dropped.
# Set to 0 to disable deadline filtering entirely.
# This must be large enough to cover clock skew between producer and worker.
DEADLINE_GRACE_MS = int(os.getenv("DEADLINE_GRACE_MS", "0"))  # 0 = disabled

# -----------------------------------------------------------------------
# Face crop key store
# -----------------------------------------------------------------------
# When > 0, the aligned 112×112 face crop is stored as a Redis key
# "face_crop:{event_id}" with this TTL (seconds). Downstream consumers
# can fetch it on demand without embedding binary in the result stream.
# Set to 0 to disable.
FACE_CROP_TTL = int(os.getenv("FACE_CROP_TTL", "60"))  # seconds
