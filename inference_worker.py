"""
inference_worker.py – BatchWorker thread.

Reads face-crop inference requests from a Redis Stream, runs batched
InsightFace + ReID GPU inference, then publishes results back to Redis.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import cv2
import numpy as np
from insightface.utils import face_align
from loguru import logger

import config
from face_db import FaceDatabase
from queue_io import RedisStreamReader, RedisStreamWriter
from reid_model import ReIDModel

# 3D reference face model for 5-point landmark pose estimation (solvePnP).
# Points: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
#
# Y-axis matches IMAGE coordinate space (Y increases DOWNWARD), so:
#   eyes   → Y negative (above nose in image = smaller Y value)
#   mouth  → Y positive (below nose in image = larger Y value)
# X-axis: positive = right in image.
# Z-axis: positive = out of screen toward camera.
# Mismatching Y convention with image coords causes ~180° pitch flip.
_FACE_3D_MODEL = np.array([
    [-165.0, -170.0, -135.0],  # left eye
    [ 165.0, -170.0, -135.0],  # right eye
    [   0.0,    0.0,    0.0],  # nose tip
    [-150.0,  150.0, -125.0],  # left mouth
    [ 150.0,  150.0, -125.0],  # right mouth
], dtype=np.float32)


class BatchWorker(threading.Thread):
    """
    Continuously:
      1. Collects up to BATCH_SIZE messages from Redis within BATCH_TIMEOUT_MS.
      2. Drops messages past their deadline_ts_ms (stale frames).
      3. Decodes JPEG → numpy in parallel (ThreadPoolExecutor).
      4. Runs batched ArcFace GPU inference in one call.
      5. Runs ReID on each image (returns zeros if ReID unavailable).
      6. Publishes results to the result stream.
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
        decode_workers: int = 4,
        deadline_grace_ms: int = config.DEADLINE_GRACE_MS,
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
        self._decode_pool = ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix="decode")
        self._deadline_grace_ms = deadline_grace_ms  # 0 = deadline filtering disabled
        self._face_crop_ttl = config.FACE_CROP_TTL  # 0 = disabled

        # Throughput stats
        self._stats_lock = threading.Lock()
        self._stats_frames   = 0   # frames processed since last report
        self._stats_batches  = 0
        self._stats_last_ts  = time.monotonic()
        # identity → (count, max_confidence) since last report
        self._stats_identities: dict[str, tuple[int, float]] = {}
        # identity -> latest (yaw, pitch, roll) in current report window
        self._stats_identity_pose: dict[str, tuple[float, float, float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_jpeg(self, data: bytes) -> Optional[np.ndarray]:
        """Decode raw JPEG bytes to a BGR numpy array. Returns None on failure."""
        try:
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return None
            return img
        except Exception as exc:
            logger.debug(f"_decode_jpeg: exception — {exc}")
            return None

    def _decode_crop(self, fields: dict) -> Optional[np.ndarray]:
        """Extract and decode the face crop from a message (raw JPEG bytes)."""
        raw = fields.get(b"face_crop_jpeg") or fields.get("face_crop_jpeg")

        if raw is None:
            logger.warning("_decode_crop: field 'face_crop_jpeg' missing. "
                           f"Keys: {[k if isinstance(k, str) else k.decode(errors='replace') for k in fields.keys()]}")
            return None

        if isinstance(raw, (bytes, bytearray)) and len(raw) >= 2 and raw[:2] != b"\xff\xd8":
            logger.warning(f"_decode_crop: invalid JPEG header {raw[:4].hex()} — check field encoding.")
            return None

        return self._decode_jpeg(raw)

    def _parse_landmarks(self, fields: dict, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Parse landmark_5_xy from a message field.

        Format accepted (tried in order):
          1. JSON  — b'[[x0,y0],[x1,y1],...]'
          2. CSV   — b'x0,y0,x1,y1,x2,y2,x3,y3,x4,y4'

        Coordinates must be **normalized [0..1] within the face crop**.
        They are scaled to absolute pixel coords using the crop dimensions.

        Returns a (5, 2) float32 array in absolute pixels, or None.
        """
        # landmark_5_xy_crop: crop-local normalized [0..1], used for alignment
        raw = fields.get(b"landmark_5_xy_crop") or fields.get("landmark_5_xy_crop")
        # fallback: legacy field name
        if not raw:
            raw = fields.get(b"landmark_5_xy") or fields.get("landmark_5_xy")
        if not raw:
            return None

        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode(errors="replace")

        h, w = img.shape[:2]
        kps = None

        # --- Try JSON first ---
        try:
            kps = np.array(json.loads(raw), dtype=np.float32).reshape(5, 2)
        except Exception:
            pass

        # --- Fallback: CSV (10 comma-separated floats) ---
        if kps is None:
            try:
                vals = [float(v.strip()) for v in raw.split(",")]
                if len(vals) == 10:
                    kps = np.array(vals, dtype=np.float32).reshape(5, 2)
            except Exception:
                pass

        if kps is None:
            logger.warning(f"_parse_landmarks: could not parse landmark_5_xy_crop: {raw!r}")
            return None

        # Scale normalized [0..1] → absolute pixel coords within crop.
        # Upstream normalizes with (fcw-1)/(fch-1) denominator, so invert correctly.
        kps[:, 0] *= max(w - 1, 1)
        kps[:, 1] *= max(h - 1, 1)
        return kps

    def _align_crop(self, img: np.ndarray, fields: dict) -> np.ndarray:
        """
        Align a decoded face crop to 112×112 for ArcFace.

        Uses norm_crop with landmarks when available (JSON or CSV, normalized [0..1]).
        Falls back to plain resize when landmarks are absent or unparseable.

        Expected landmark order (RetinaFace standard):
          [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
        """
        kps = self._parse_landmarks(fields, img)
        if kps is not None:
            return face_align.norm_crop(img, landmark=kps)
        return cv2.resize(img, (112, 112))

    def _parse_fullframe_landmarks(self, fields: dict) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Parse full-frame absolute landmarks + frame dimensions for accurate solvePnP.

        Expects upstream to send:
          - landmark_5_xy_abs: 10 comma-separated absolute pixel coords in the full camera frame
          - frame_width:  full camera frame width in pixels
          - frame_height: full camera frame height in pixels

        Returns (kps_abs (5,2 float32), frame_w, frame_h), or None if fields absent.
        """
        # landmark_5_xy: full-frame absolute pixel coords (for pose geometry)
        raw = fields.get(b"landmark_5_xy") or fields.get("landmark_5_xy")
        fw  = fields.get(b"full_frame_width")  or fields.get("full_frame_width")
        fh  = fields.get(b"full_frame_height") or fields.get("full_frame_height")
        if raw is None or fw is None or fh is None:
            return None
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode(errors="replace")
            vals = [float(v.strip()) for v in raw.split(",")]
            if len(vals) != 10:
                return None
            kps = np.array(vals, dtype=np.float32).reshape(5, 2)
            return kps, float(fw), float(fh)
        except Exception:
            return None

    def _estimate_pose(self, fields: dict, img: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll) in degrees.

        Preferred path (accurate): upstream sends `landmark_5_xy_abs` (absolute
        full-frame pixel coords) plus `frame_width` / `frame_height`.  The camera
        matrix is built from the full sensor size, giving geometrically correct angles.

        Fallback: crop-local normalized landmarks + crop dimensions for the camera
        matrix.  This is less accurate but avoids breaking existing producers.

        Sign/axis convention follows InsightFace transform.matrix2angle:
          pitch = x = atan2(R[2,1], R[2,2])
          yaw   = y = atan2(-R[2,0], sy)
          roll  = z = atan2(R[1,0], R[0,0])
        """
        # --- Preferred: full-frame absolute landmarks ---
        full = self._parse_fullframe_landmarks(fields)
        if full is not None:
            kps, fw, fh = full
            focal = fw  # focal ≈ frame width (pinhole approximation)
            cam_matrix = np.array([
                [focal, 0,     fw / 2],
                [0,     focal, fh / 2],
                [0,     0,     1     ],
            ], dtype=np.float32)
        else:
            # --- Fallback: crop-local landmarks (less accurate for pose) ---
            kps = self._parse_landmarks(fields, img)
            if kps is None:
                return 0.0, 0.0, 0.0
            h, w = img.shape[:2]
            focal = float(w)
            cam_matrix = np.array([
                [focal, 0,     w / 2],
                [0,     focal, h / 2],
                [0,     0,     1    ],
            ], dtype=np.float32)

        try:
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            _, rvec, _ = cv2.solvePnP(
                _FACE_3D_MODEL, kps, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
            rmat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                # Same as insightface.utils.transform.matrix2angle:
                # x=pitch, y=yaw, z=roll
                pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
                yaw   = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
                roll  = float(np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0])))
            else:
                pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
                yaw   = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
                roll  = 0.0
            return round(yaw, 2), round(pitch, 2), round(roll, 2)
        except Exception as exc:
            logger.debug(f"_estimate_pose failed: {exc}")
            return 0.0, 0.0, 0.0

    @staticmethod
    def _estimate_quality(img: np.ndarray) -> float:
        """
        Estimate face image quality as a Laplacian variance score (blur detection).
        Higher = sharper.  Normalized to [0, 1] with soft saturation at 500.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(min(lap_var / 500.0, 1.0), 4)

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
            # b"embedding":     b"",
            # b"reid_embedding":b"",
            b"error_message": reason.encode(),
        }
        self._writer.write(result)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        logger.info(f"[{self.name}] BatchWorker started.")
        _STATS_INTERVAL = 5.0  # seconds between throughput reports
        _next_report = time.monotonic() + _STATS_INTERVAL
        while not self._stop_event.is_set():
            try:
                self._process_batch()
            except Exception as exc:
                logger.exception(f"BatchWorker unexpected error: {exc}")
                time.sleep(0.1)
            if time.monotonic() >= _next_report:
                self._report_stats()
                _next_report = time.monotonic() + _STATS_INTERVAL
        logger.info(f"[{self.name}] BatchWorker stopped.")

    def _report_stats(self):
        now = time.monotonic()
        with self._stats_lock:
            frames     = self._stats_frames
            batches    = self._stats_batches
            elapsed    = now - self._stats_last_ts
            identities = self._stats_identities.copy()
            identity_pose = self._stats_identity_pose.copy()
            self._stats_frames     = 0
            self._stats_batches    = 0
            self._stats_last_ts    = now
            self._stats_identities = {}
            self._stats_identity_pose = {}

        fps = frames / elapsed if elapsed > 0 else 0.0

        # Request stream lag
        req_lag = "?"
        result_len = "?"
        try:
            client = self._reader._client
            groups = client.xinfo_groups(self._reader._stream)
            for g in groups:
                gname = g.get(b"name", g.get("name", b""))
                if isinstance(gname, bytes):
                    gname = gname.decode()
                if gname == self._reader._group:
                    req_lag = g.get(b"lag", g.get("lag", "?"))
                    break
        except Exception:
            pass
        try:
            result_len = self._writer._client.xlen(self._writer._stream)
        except Exception:
            pass

        logger.info(
            f"[{self.name}] Stats | {fps:.1f} fps  {frames} frames  "
            f"{batches} batches  req_queue={req_lag}  result_stream_len={result_len}"
        )

        if identities:
            # Sort: known identities first (not Unknown), then by count descending
            sorted_ids = sorted(
                identities.items(),
                key=lambda x: (x[0] == "Unknown", -x[1][0])
            )
            parts = []
            for name, (count, best_conf) in sorted_ids:
                yaw, pitch, roll = identity_pose.get(name, (0.0, 0.0, 0.0))
                parts.append(
                    f"{name}×{count}({best_conf:.2f}) (y:{yaw:+.1f},p:{pitch:+.1f},r:{roll:+.1f})"
                )
            logger.info(f"[{self.name}] Seen   | {' | '.join(parts)}")

    def _process_batch(self):
        messages = self._reader.read()
        if not messages:
            return

        now_ms = int(time.time() * 1000)
        live, stale_ids = [], []
        for msg_id, fields in messages:
            if self._deadline_grace_ms > 0:
                deadline_raw = fields.get(b"deadline_ts_ms") or fields.get("deadline_ts_ms")
                if deadline_raw:
                    try:
                        deadline = int(deadline_raw)
                        if deadline + self._deadline_grace_ms < now_ms:
                            stale_ids.append(msg_id)
                            continue
                    except (ValueError, TypeError):
                        pass
            live.append((msg_id, fields))

        if stale_ids:
            logger.debug(f"Dropped {len(stale_ids)} stale message(s) past deadline.")
            self._reader.ack(stale_ids)

        if not live:
            return

        msg_ids = [msg_id for msg_id, _ in live]
        fields_list = [fields for _, fields in live]

        if self._run_detection:
            for fields in fields_list:
                self._process_single_detect(fields)
        else:
            self._process_batch_direct(fields_list)

        self._reader.ack(msg_ids)
        with self._stats_lock:
            self._stats_batches += 1

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

        # --- Stage 1: parallel decode + align (CPU) ---
        def _decode_one(fields):
            img = self._decode_crop(fields)
            if img is None:
                event_id = fields.get(b"event_id", b"").decode(errors="replace")
                logger.warning(f"[{event_id}] JPEG decode failed.")
                self._publish_error(fields, "decode_failed")
                return fields, None, None
            return fields, img, self._align_crop(img, fields)

        decoded = list(self._decode_pool.map(_decode_one, fields_list))

        valid = [(fields, raw_img, crop) for fields, raw_img, crop in decoded if crop is not None]
        if not valid:
            return

        # --- Stage 2: single batched GPU call ---
        # get_feat expects a list of (112,112,3) images, NOT a stacked (N,112,112,3) array.
        crops_list = [crop for _, _, crop in valid]
        try:
            with self._model_lock:
                embeddings = rec_model.get_feat(crops_list)  # (N, 512)
            embeddings = np.array(embeddings)
        except Exception as exc:
            logger.exception(f"_process_batch_direct: rec_model.get_feat failed — {exc}")
            for fields, _, _ in valid:
                self._publish_error(fields, f"insightface_error:{exc}")
            return

        # --- Stage 3: match + ReID + publish ---
        for i, (fields, raw_img, crop) in enumerate(valid):
            emb = embeddings[i]
            if np.linalg.norm(emb) < 1e-3:
                self._publish_error(fields, "no_face_detected")
                continue
            self._publish_result(fields, emb, crop, crop if not self._no_reid else None, pose_img=raw_img)

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

        self._publish_result(fields, embedding, img, img if not self._no_reid else None, pose_img=img)

    def _publish_result(
        self,
        fields: dict,
        embedding: np.ndarray,
        crop: np.ndarray,
        img_for_reid,
        pose_img: Optional[np.ndarray] = None,
    ):
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

        # Pose and quality — best-effort, never block publish on failure
        try:
            yaw, pitch, roll = self._estimate_pose(fields, pose_img if pose_img is not None else crop)
        except Exception:
            yaw, pitch, roll = 0.0, 0.0, 0.0
        try:
            quality = self._estimate_quality(crop)
        except Exception:
            quality = 0.0

        # Store original crop as a short-lived Redis key for on-demand downstream access
        face_crop_key = ""
        if self._face_crop_ttl > 0 and event_id:
            raw_jpeg = fields.get(b"face_crop_jpeg") or fields.get("face_crop_jpeg")
            if raw_jpeg:
                try:
                    face_crop_key = f"face_crop:{event_id}"
                    self._writer.set_key(face_crop_key, bytes(raw_jpeg), self._face_crop_ttl)
                except Exception as exc:
                    logger.warning(f"[{event_id}] Failed to store face_crop key: {exc}")
                    face_crop_key = ""

        result = {
            b"event_id":       fields.get(b"event_id", b""),
            b"camera_id":      fields.get(b"camera_id",  b""),
            b"track_id":       fields.get(b"track_id",   b""),
            b"timestamp":      fields.get(b"timestamp",  b""),
            b"identity":       identity.encode(),
            b"identity_id":    identity_id.encode(),
            b"confidence":     str(round(float(confidence), 6)).encode(),
            b"yaw":            str(round(yaw,   2)).encode(),
            b"pitch":          str(round(pitch, 2)).encode(),
            b"roll":           str(round(roll,  2)).encode(),
            b"quality":        str(round(quality, 4)).encode(),
            b"face_crop_key":  face_crop_key.encode(),
            # b"embedding":      json.dumps(embedding.tolist()).encode(),
            # b"reid_embedding": json.dumps(reid_emb.tolist()).encode(),
        }
        self._writer.write(result)
        with self._stats_lock:
            self._stats_frames += 1
            count, best = self._stats_identities.get(identity, (0, 0.0))
            self._stats_identities[identity] = (count + 1, max(best, float(confidence)))
            self._stats_identity_pose[identity] = (float(yaw), float(pitch), float(roll))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        self._stop_event.set()
