"""
config.py – All service settings with sensible defaults and env-var overrides.
"""

import os

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REQUEST_STREAM  = os.getenv("REQUEST_STREAM",  "face_inference:requests")
RESULT_STREAM   = os.getenv("RESULT_STREAM",   "face_inference:results")
CONSUMER_GROUP  = os.getenv("CONSUMER_GROUP",  "face_inference_workers")
CONSUMER_NAME   = os.getenv("CONSUMER_NAME",   "worker_0")

# Batching
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",      "8"))
BATCH_TIMEOUT_MS= int(os.getenv("BATCH_TIMEOUT_MS","50"))  # max wait to fill a batch
RESULT_MAXLEN   = int(os.getenv("RESULT_MAXLEN",   "10000"))  # max entries in result stream

# Matching
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.35"))

# Gallery / cache
GALLERY_DIR = os.getenv("GALLERY_DIR", "../gallery")
CACHE_PATH = os.getenv("CACHE_PATH", "face_db_cache.npz")

# InsightFace
DET_SIZE = int(os.getenv("DET_SIZE", "640"))
GPU_ID = int(os.getenv("GPU_ID", "0"))

# When True: run full detection pipeline (SCRFD + alignment + ArcFace).
# When False (default for pre-cropped inputs): skip detection, run
# landmark alignment + ArcFace directly on the supplied crop.
RUN_DETECTION = os.getenv("RUN_DETECTION", "0") == "1"
