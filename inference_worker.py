"""
inference_worker.py – BatchWorker thread.

Reads face-crop inference requests from a Redis Stream, runs batched
InsightFace + ReID GPU inference, then publishes results back to Redis.
"""

import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger

import config
from face_db import FaceDatabase
from queue_io import RedisStreamReader, RedisStreamWriter
from reid_model import ReIDModel


class BatchWorker(threading.Thread):
    """
    Continuously:
      1. Collects up to BATCH_SIZE messages from Redis within BATCH_TIMEOUT_MS.
      2. Decodes JPEG → numpy (corrupt JPEG → error result).
      3. Runs InsightFace on each image:
           - `face_app.get(img)` first (detection + recognition).
           - If no face detected, falls back to direct recognition model call.
      4. Runs ReID on each image (returns zeros if ReID unavailable).
      5. Publishes results to the result stream.
    """

    def __init__(
        self,
        reader: RedisStreamReader,
        writer: RedisStreamWriter,
        face_app,
        face_db: FaceDatabase,
        model_lock: threading.Lock,
        reid_model: Optional[ReIDModel],
        threshold: float = config.MATCH_THRESHOLD,
        batch_size: int = config.BATCH_SIZE,
        batch_timeout_ms: int = config.BATCH_TIMEOUT_MS,
        no_reid: bool = False,
        run_detection: bool = False,
    ):
        super().__init__(daemon=True)
        self._reader = reader
        self._writer = writer
        self._face_app = face_app
        self._face_db = face_db
        self._model_lock = model_lock
        self._reid_model = reid_model
        self._threshold = threshold
        self._batch_size = batch_size
        self._batch_timeout_ms = batch_timeout_ms
        self._no_reid = no_reid
        self._run_detection = run_detection
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_jpeg(self, data: bytes) -> Optional[np.ndarray]:
        """Decode raw JPEG bytes to a BGR numpy array. Returns None on failure."""
        try:
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                logger.debug(f"_decode_jpeg: cv2.imdecode returned None/empty for {len(data)} bytes")
                return None
            logger.debug(f"_decode_jpeg: OK → shape={img.shape}")
            return img
        except Exception as exc:
            logger.debug(f"_decode_jpeg: exception — {exc}")
            return None

    def _decode_crop(self, fields: dict) -> Optional[np.ndarray]:
        """Extract and decode the face crop from a message (raw JPEG bytes)."""
        raw = fields.get(b"face_crop_jpeg") or fields.get("face_crop_jpeg")

        if raw is None:
            logger.warning("_decode_crop: field 'face_crop_jpeg' is missing entirely. "
                           f"Available keys: {[k if isinstance(k, str) else k.decode(errors='replace') for k in fields.keys()]}")
            return None

        if isinstance(raw, str):
            logger.warning("_decode_crop: payload is str, not bytes — "
                           "Redis connection may have decode_responses=True. Converting via latin-1.")
            raw = raw.encode("latin-1")

        if len(raw) == 0:
            logger.warning("_decode_crop: payload is empty (0 bytes).")
            return None

        head = raw[:4].hex()
        tail = raw[-2:].hex()
        logger.debug(f"_decode_crop: len={len(raw)} head={head} tail={tail}")

        if not raw[:2] == b"\xff\xd8":
            logger.warning(f"_decode_crop: not a valid JPEG — expected header ffd8, got {head}. "
                           "Possible cause: base64 still being sent, truncation, or wrong field.")
            return None
        if not raw[-2:] == b"\xff\xd9":
            logger.warning(f"_decode_crop: JPEG footer missing (got {tail}) — payload may be truncated.")

        return self._decode_jpeg(raw)

    def _get_embedding_insightface(self, img: np.ndarray):
        """
        Dispatcher: routes to detection or direct-recognition path based on
        the `run_detection` flag set at construction time.

        Returns (embedding_512d, face_obj_or_None).
        """
        if self._run_detection:
            return self._embed_with_detection(img)
        return self._embed_direct(img)

    def _embed_with_detection(self, img: np.ndarray):
        """
        Detection path (--detect flag ON):
        Runs the full InsightFace pipeline — SCRFD detection → landmark
        alignment → ArcFace embedding.  Best for scene images where the
        face is not pre-isolated.
        """
        with self._model_lock:
            faces = self._face_app.get(img)

        if faces:
            best = max(faces, key=lambda f: (
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                if f.bbox is not None else 0
            ))
            if best.embedding is not None and np.linalg.norm(best.embedding) > 1e-3:
                return best.embedding, best

        return None, None

    def _embed_direct(self, img: np.ndarray):
        """
        Direct recognition path (default, --detect flag OFF):
        Assumes the input is already a face crop.  Skips SCRFD detection
        and feeds the crop straight to the ArcFace recognition model after
        resizing to 112×112.  Faster and more reliable for pre-cropped inputs.
        """
        try:
            rec_model = self._face_app.models.get("recognition")
            if rec_model is not None:
                aligned = cv2.resize(img, (112, 112))
                with self._model_lock:
                    emb = rec_model.get_feat(aligned[np.newaxis])  # (1, 512)
                if emb is not None:
                    emb = np.array(emb).flatten()
                    if np.linalg.norm(emb) > 1e-3:
                        return emb, None
        except Exception as exc:
            logger.debug(f"Direct rec_model call failed: {exc}")

        return None, None

    def _publish_error(self, fields: dict, reason: str):
        """Publish an error result for a request that could not be processed."""
        result = {
            b"event_id":    fields.get(b"event_id", b""),
            b"camera_id":     fields.get(b"camera_id",  b""),
            b"track_id":      fields.get(b"track_id",   b""),
            b"timestamp":     fields.get(b"timestamp",  b""),
            b"identity":      b"error",
            b"confidence":    b"0.0",
            b"embedding":     b"",
            b"reid_embedding":b"",
            b"error_message": reason.encode(),
        }
        self._writer.write(result)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        logger.info("BatchWorker started.")
        while not self._stop_event.is_set():
            try:
                self._process_batch()
            except Exception as exc:
                logger.exception(f"BatchWorker unexpected error: {exc}")
                time.sleep(0.1)
        logger.info("BatchWorker stopped.")

    def _process_batch(self):
        messages = self._reader.read()
        if not messages:
            return

        msg_ids = [msg_id for msg_id, _ in messages]
        fields_list = [fields for _, fields in messages]

        if self._run_detection:
            # Detection path: InsightFace public API is single-image only
            for fields in fields_list:
                self._process_single_detect(fields)
        else:
            # Direct path: batch all crops into one GPU call
            self._process_batch_direct(fields_list)

        self._reader.ack(msg_ids)

    def _process_batch_direct(self, fields_list: list):
        """
        True GPU batch inference for the direct (no-detection) path.
        Decodes all JPEGs, stacks valid crops into (N, 112, 112, 3),
        calls rec_model.get_feat once, then matches + publishes each result.
        """
        rec_model = self._face_app.models.get("recognition")
        if rec_model is None:
            for fields in fields_list:
                self._publish_error(fields, "no_recognition_model")
            return

        # --- Stage 1: decode + resize (CPU, parallelisable) ---
        decoded = []  # list of (fields, img_112) or (fields, None)
        for fields in fields_list:
            img = self._decode_crop(fields)
            if img is None:
                event_id = fields.get(b"event_id", b"").decode(errors="replace")
                logger.warning(f"[{event_id}] JPEG decode failed.")
                self._publish_error(fields, "decode_failed")
                decoded.append((fields, None))
            else:
                # DEBUG: save raw decoded image for visual inspection (first 5 only)
                import os, tempfile
                _debug_dir = os.path.join(tempfile.gettempdir(), "face_debug")
                os.makedirs(_debug_dir, exist_ok=True)
                existing = len(os.listdir(_debug_dir))
                if existing < 5:
                    event_id = fields.get(b"event_id", b"").decode(errors="replace")
                    debug_path = os.path.join(_debug_dir, f"{existing:03d}_{event_id[:40]}.jpg")
                    cv2.imwrite(debug_path, img)
                    logger.info(f"DEBUG saved crop → {debug_path}  shape={img.shape}")
                decoded.append((fields, cv2.resize(img, (112, 112))))

        valid = [(fields, crop) for fields, crop in decoded if crop is not None]
        if not valid:
            return

        # --- Stage 2: single batched GPU call ---
        # get_feat expects a list of (112,112,3) images, NOT a stacked (N,112,112,3) array.
        # Passing a numpy array causes cv2.dnn.blobFromImages to treat the whole
        # batch as a single malformed image and crash on resize.
        crops_list = [crop for _, crop in valid]
        batch_arr = np.stack(crops_list, axis=0)  # kept only for the debug log
        logger.debug(f"_process_batch_direct: batch_arr shape={batch_arr.shape} dtype={batch_arr.dtype}")
        try:
            logger.debug("_process_batch_direct: calling rec_model.get_feat…")
            with self._model_lock:
                embeddings = rec_model.get_feat(crops_list)  # (N, 512)
            embeddings = np.array(embeddings)
            logger.debug(f"_process_batch_direct: embeddings shape={embeddings.shape} norms={np.linalg.norm(embeddings, axis=1).round(3).tolist()}")
        except Exception as exc:
            logger.exception(f"_process_batch_direct: rec_model.get_feat failed — {exc}")
            for fields, _ in valid:
                self._publish_error(fields, f"insightface_error:{exc}")
            return

        # --- Stage 3: match + ReID + publish ---
        for i, (fields, crop) in enumerate(valid):
            emb = embeddings[i]
            if np.linalg.norm(emb) < 1e-3:
                self._publish_error(fields, "no_face_detected")
                continue
            self._publish_result(fields, emb, crop if not self._no_reid else None)

    def _process_single_detect(self, fields: dict):
        """
        Detection path (--detect ON): single-image full InsightFace pipeline.
        face_app.get() is not batchable via the public API.
        """
        event_id = fields.get(b"event_id", b"").decode(errors="replace")

        img = self._decode_crop(fields)
        if img is None:
            logger.warning(f"[{event_id}] JPEG decode failed.")
            self._publish_error(fields, "decode_failed")
            return

        try:
            embedding, _ = self._embed_with_detection(img)
        except Exception as exc:
            logger.warning(f"[{event_id}] InsightFace error: {exc}")
            self._publish_error(fields, f"insightface_error:{exc}")
            return

        if embedding is None:
            logger.debug(f"[{event_id}] No face detected in crop.")
            self._publish_error(fields, "no_face_detected")
            return

        self._publish_result(fields, embedding, img if not self._no_reid else None)

    def _publish_result(self, fields: dict, embedding: np.ndarray, img_for_reid):
        """Match embedding, run ReID if needed, publish to result stream."""
        event_id = fields.get(b"event_id", b"").decode(errors="replace")

        identity, identity_id, confidence = self._face_db.match(embedding)
        if confidence < self._threshold:
            identity = "Unknown"
            identity_id = ""

        if img_for_reid is not None and self._reid_model is not None:
            try:
                reid_emb = self._reid_model.get_embedding(img_for_reid)
            except Exception as exc:
                logger.warning(f"[{event_id}] ReID error: {exc}")
                reid_emb = np.zeros(256, dtype=np.float32)
        else:
            reid_emb = np.zeros(256, dtype=np.float32)

        result = {
            b"event_id":       fields.get(b"event_id", b""),
            b"camera_id":      fields.get(b"camera_id",  b""),
            b"track_id":       fields.get(b"track_id",   b""),
            b"timestamp":      fields.get(b"timestamp",  b""),
            b"identity":       identity.encode(),
            b"identity_id":    identity_id.encode(),
            b"confidence":     str(round(float(confidence), 6)).encode(),
            b"embedding":      json.dumps(embedding.tolist()).encode(),
            b"reid_embedding": json.dumps(reid_emb.tolist()).encode(),
        }
        self._writer.write(result)
        logger.debug(f"[{event_id}] → identity={identity} conf={confidence:.3f}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        self._stop_event.set()
