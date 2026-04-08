"""
main.py – Entry point for the centralized face inference service.

Loads models once, starts the BatchWorker thread, and blocks until
interrupted (Ctrl+C or SIGTERM).

Usage:
    python main.py --gallery ../gallery

Full options:
    python main.py --help
"""

import os
# Ensure the conda env's own site-packages take priority over user site-packages
# so onnxruntime-gpu finds its cuDNN DLLs.
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

from loguru import logger

import config as cfg


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Centralized GPU Face Recognition Service"
    )
    p.add_argument("--gallery",    default=cfg.GALLERY_DIR,   help=f"Gallery directory  [{cfg.GALLERY_DIR}]")
    p.add_argument("--cache",      default=cfg.CACHE_PATH,    help=f"Embedding cache path  [{cfg.CACHE_PATH}]")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help=f"Inference batch size  [{cfg.BATCH_SIZE}]")
    p.add_argument("--det-size",   type=int, default=cfg.DET_SIZE,   help=f"InsightFace detection size  [{cfg.DET_SIZE}]")
    p.add_argument("--threshold",  type=float, default=cfg.MATCH_THRESHOLD, help=f"Match cosine threshold  [{cfg.MATCH_THRESHOLD}]")
    p.add_argument("--redis-url",       default=cfg.REDIS_URL,        help=f"Redis URL  [{cfg.REDIS_URL}]")
    p.add_argument("--request-stream",  default=cfg.REQUEST_STREAM,   help=f"Redis input stream name  [{cfg.REQUEST_STREAM}]")
    p.add_argument("--result-stream",   default=cfg.RESULT_STREAM,    help=f"Redis output stream name  [{cfg.RESULT_STREAM}]")
    p.add_argument("--consumer-group",  default=cfg.CONSUMER_GROUP,   help=f"Redis consumer group name  [{cfg.CONSUMER_GROUP}]")
    p.add_argument("--result-maxlen",   type=int, default=cfg.RESULT_MAXLEN, help=f"Max entries kept in result stream (0=unlimited)  [{cfg.RESULT_MAXLEN}]")
    p.add_argument("--no-reid",    action="store_true",       help="Skip ReID embedding (faster, uses zero vectors)")
    p.add_argument("--detect",     action="store_true",       help="Run full face detection on each crop (SCRFD + alignment). Default: off — assumes pre-cropped faces, runs landmark alignment + ArcFace directly.")
    p.add_argument("--gpu-id",     type=int, default=cfg.GPU_ID, help=f"GPU device ID  [{cfg.GPU_ID}]")
    return p.parse_args()


# ------------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------------

def load_insightface(det_size: int, gpu_id: int):
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        logger.error("insightface not installed. Run: pip install insightface")
        sys.exit(1)

    logger.info("Loading InsightFace model (buffalo_l)…")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=gpu_id, det_size=(det_size, det_size))
    logger.info("InsightFace ready.")
    return face_app


def load_reid(gpu_id: int, no_reid: bool):
    if no_reid:
        logger.info("ReID skipped (--no-reid).")
        return None
    from reid_model import ReIDModel
    return ReIDModel(gpu_id=gpu_id)


def build_face_db(gallery: str, cache: str, face_app):
    from face_db import FaceDatabase
    face_db = FaceDatabase(cache_path=cache if cache else None)
    gallery_path = Path(gallery)
    if gallery_path.exists():
        logger.info(f"Building face database from '{gallery}'…")
        n = face_db.build_from_gallery(str(gallery_path), face_app=face_app)
        logger.info(f"Face database ready: {n} identit(ies).")
    else:
        logger.warning(f"Gallery '{gallery}' not found. Running with empty DB.")
    return face_db


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Centralized GPU Face Recognition Service")
    logger.info("=" * 60)
    logger.info(f"  Redis URL   : {args.redis_url}")
    logger.info(f"  Req stream  : {args.request_stream}")
    logger.info(f"  Res stream  : {args.result_stream}")
    logger.info(f"  Gallery     : {args.gallery}")
    logger.info(f"  Batch size  : {args.batch_size}")
    logger.info(f"  Threshold   : {args.threshold}")
    logger.info(f"  GPU ID      : {args.gpu_id}")
    logger.info(f"  ReID        : {'disabled' if args.no_reid else 'enabled'}")
    logger.info(f"  Detection   : {'enabled (full SCRFD pipeline)' if args.detect else 'disabled (pre-crop direct ArcFace)'}")
    logger.info("=" * 60)

    # 1. Load InsightFace
    face_app   = load_insightface(args.det_size, args.gpu_id)
    model_lock = threading.Lock()

    # 2. Build face database
    face_db = build_face_db(args.gallery, args.cache, face_app)

    # 3. Load ReID model
    reid_model = load_reid(args.gpu_id, args.no_reid)

    # 4. Create Redis I/O
    from queue_io import RedisStreamReader, RedisStreamWriter
    reader = RedisStreamReader(
        redis_url=args.redis_url,
        stream=args.request_stream,
        group=args.consumer_group,
        consumer=cfg.CONSUMER_NAME,
        batch_size=args.batch_size,
        block_ms=cfg.BATCH_TIMEOUT_MS,
    )
    writer = RedisStreamWriter(
        redis_url=args.redis_url,
        stream=args.result_stream,
        maxlen=args.result_maxlen if args.result_maxlen > 0 else None,
    )

    # 5. Start BatchWorker
    from inference_worker import BatchWorker
    worker = BatchWorker(
        reader=reader,
        writer=writer,
        face_app=face_app,
        face_db=face_db,
        model_lock=model_lock,
        reid_model=reid_model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        batch_timeout_ms=cfg.BATCH_TIMEOUT_MS,
        no_reid=args.no_reid,
        run_detection=args.detect,
    )
    worker.start()
    logger.info("Service running. Press Ctrl+C to stop.")

    # 6. Block — handle Ctrl+C / SIGTERM
    stop_event = threading.Event()

    def _shutdown(signum, frame):
        logger.info(f"Signal {signum} received. Shutting down…")
        stop_event.set()
        worker.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)

    worker.join(timeout=10)
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
