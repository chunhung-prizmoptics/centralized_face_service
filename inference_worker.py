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
        track_cache_ttl: int = config.TRACK_CACHE_TTL,
        track_cache_recheck: int = config.TRACK_CACHE_RECHECK,
        track_cache_unknown_recheck: int = config.TRACK_CACHE_UNKNOWN_RECHECK,
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

        # Per-track identity cache.
        # Once a track is confidently matched, subsequent frames for the same
        # (camera_id, track_id) skip GPU inference and re-publish the cached
        # result directly.  Cache entry expires after TRACK_CACHE_TTL seconds
        # of inactivity (track left scene) or is force-rechecked every
        # TRACK_CACHE_RECHECK seconds to catch genuine identity changes.
        # (camera_id, track_id) → {identity, identity_id, confidence, yaw, pitch, roll,
        #                          quality, last_seen, last_infer}
        self._track_cache_ttl              = track_cache_ttl
        self._track_cache_recheck          = track_cache_recheck
        self._track_cache_unknown_recheck  = track_cache_unknown_recheck if track_cache_unknown_recheck > 0 else track_cache_recheck
        self._track_cache: dict[tuple, dict] = {}
        self._track_cache_lock = threading.Lock()

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

    @staticmethod
    def _has_eye_texture(kps: np.ndarray, img: np.ndarray,
                         threshold: float = 15.0) -> bool:
        """
        Return True if at least one eye-landmark region has real eye texture.

        Extracts a patch around each of the two eye landmarks (patch side =
        30% of eye separation) and checks Laplacian variance.  Returns False
        only when BOTH patches are below the threshold — meaning both look like
        hair/skin rather than iris/sclera edges.  Using AND keeps the check
        forgiving for profile faces where one eye may be partially occluded.
        """
        img_h, img_w = img.shape[:2]
        left_eye, right_eye = kps[0], kps[1]
        eye_sep = abs(right_eye[0] - left_eye[0])
        half = max(int(eye_sep * 0.15), 4)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def _lap(cx: float, cy: float) -> float:
            x0, y0 = max(int(cx) - half, 0), max(int(cy) - half, 0)
            x1, y1 = min(int(cx) + half, img_w), min(int(cy) + half, img_h)
            p = gray[y0:y1, x0:x1]
            return float(cv2.Laplacian(p, cv2.CV_64F).var()) if p.size > 0 else 0.0

        return _lap(*left_eye) >= threshold or _lap(*right_eye) >= threshold

    @staticmethod
    def _is_valid_landmark_geometry(kps: np.ndarray, img: np.ndarray,
                                    eye_patch_lap_threshold: float = 15.0) -> bool:
        """
        Reject crops where the 5-point landmark set is anatomically impossible
        OR where the eye-region texture indicates hair/skin rather than real eyes.

        Two-stage check:

        Stage 1 — Geometric ordering (fast, no pixels):
          1. Nose Y > both eye Ys     (nose is below eyes in image coords)
          2. Mouth Y > nose Y         (mouth is below nose)
          3. Left eye X < right eye X (eyes not horizontally swapped)
          4. Eye separation > 10% of image width

        Stage 2 — Eye-region texture (catches back-of-head where geometry passes):
          For each of the two eye landmarks, extract a square patch whose side
          is 30% of eye separation (scales with crop size).  Compute Laplacian
          variance on the grayscale patch.  Real eyes have high local contrast
          (iris/sclera/eyelid edges).  Hair/skin at fake eye positions is
          low-contrast uniform texture.
          Reject if BOTH eye patches are below `eye_patch_lap_threshold`.
          Using AND (both low) keeps the check conservative — profile faces may
          have one partially occluded eye.
        """
        img_h, img_w = img.shape[:2]
        left_eye, right_eye, nose, left_mouth, right_mouth = kps

        # --- Stage 1: geometric ordering ---
        if nose[1] <= left_eye[1] or nose[1] <= right_eye[1]:
            return False
        mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
        if mouth_y <= nose[1]:
            return False
        if left_eye[0] >= right_eye[0]:
            return False
        eye_sep = right_eye[0] - left_eye[0]
        if eye_sep < 0.10 * img_w:
            return False

        # --- Stage 2: eye-region texture ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        half = max(int(eye_sep * 0.15), 4)  # half-side of patch

        def _patch_lap_var(cx: float, cy: float) -> float:
            x0 = max(int(cx) - half, 0)
            y0 = max(int(cy) - half, 0)
            x1 = min(int(cx) + half, img_w)
            y1 = min(int(cy) + half, img_h)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                return 0.0
            return float(cv2.Laplacian(patch, cv2.CV_64F).var())

        left_lap  = _patch_lap_var(left_eye[0],  left_eye[1])
        right_lap = _patch_lap_var(right_eye[0], right_eye[1])

        # Reject only if BOTH patches are below threshold (conservative)
        if left_lap < eye_patch_lap_threshold and right_lap < eye_patch_lap_threshold:
            return False

        return True

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

    def _track_cache_get(self, camera_id: str, track_id: str, now: float) -> Optional[dict]:
        """
        Return a cached result for (camera_id, track_id) if it is still valid,
        or None if inference should run.

        A cached entry is returned when:
          - It exists and was seen within TRACK_CACHE_TTL seconds (track still active), AND
          - It was last inferred within TRACK_CACHE_RECHECK seconds (not due for recheck).

        Cache is disabled when TRACK_CACHE_TTL == 0.
        """
        if self._track_cache_ttl <= 0:
            return None
        key = (camera_id, track_id)
        with self._track_cache_lock:
            entry = self._track_cache.get(key)
            if entry is None:
                return None
            if now - entry["last_seen"] > self._track_cache_ttl:
                # Track went idle — evict and re-infer
                del self._track_cache[key]
                return None
            entry["last_seen"] = now
            recheck = (self._track_cache_unknown_recheck
                       if entry.get("identity") == "Unknown"
                       else self._track_cache_recheck)
            if recheck > 0 and now - entry["last_infer"] > recheck:
                # Due for a periodic re-inference to catch genuine identity changes
                return None
            return entry

    def _track_cache_put(self, camera_id: str, track_id: str, now: float, entry: dict):
        """Store or update a confident match in the track cache."""
        if self._track_cache_ttl <= 0:
            return
        key = (camera_id, track_id)
        entry = dict(entry, last_seen=now, last_infer=now)
        with self._track_cache_lock:
            self._track_cache[key] = entry
            # Evict stale entries piggyback (no extra thread needed)
            if len(self._track_cache) > 10_000:
                cutoff = now - self._track_cache_ttl
                stale = [k for k, v in self._track_cache.items() if v["last_seen"] < cutoff]
                for k in stale:
                    del self._track_cache[k]

    @staticmethod
    def _estimate_image_metrics(img: np.ndarray) -> dict:
        """
        Compute all image-quality metrics in one pass over the decoded image.

        Returns a dict with:
          sharpness  – Laplacian variance normalised to [0, 1] (saturates at 500)
          brightness – mean pixel intensity normalised to [0, 1]
          dark_ratio – fraction of pixels with intensity < 40  (0 = no dark pixels)
          face_size  – min width or height of the image in pixels
          quality    – alias for sharpness (kept for backward compat)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness  = round(min(lap_var / 500.0, 1.0), 4)
        brightness = round(float(gray.mean()) / 255.0, 4)
        dark_ratio = round(float((gray < 40).sum()) / gray.size, 4)
        h, w       = img.shape[:2]
        face_size  = min(w, h)
        return {
            "sharpness":  sharpness,
            "brightness": brightness,
            "dark_ratio": dark_ratio,
            "face_size":  face_size,
            "quality":    sharpness,   # backward compat
        }

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
        now_s  = now_ms / 1000.0
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

            # Per-track cache check: if this (camera_id, track_id) was already
            # confidently matched recently, re-publish the cached result without
            # running GPU inference again.
            cam_raw   = fields.get(b"camera_id")  or fields.get("camera_id")  or b""
            track_raw = fields.get(b"track_id")   or fields.get("track_id")   or b""
            camera_id = cam_raw.decode(errors="replace")   if isinstance(cam_raw,   (bytes, bytearray)) else str(cam_raw)
            track_id  = track_raw.decode(errors="replace") if isinstance(track_raw, (bytes, bytearray)) else str(track_raw)

            cached = self._track_cache_get(camera_id, track_id, now_s)
            if cached is not None:
                # Re-publish from cache, skip GPU pipeline entirely
                self._publish_cached(fields, cached)
                stale_ids.append(msg_id)  # ack alongside stale
                continue

            live.append((msg_id, fields))

        if stale_ids:
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

        # Pose and quality metrics — best-effort, never block publish on failure
        try:
            yaw, pitch, roll = self._estimate_pose(fields, pose_img if pose_img is not None else crop)
        except Exception:
            yaw, pitch, roll = 0.0, 0.0, 0.0
        try:
            # Use original decoded crop for metrics so face_size reflects
            # upstream crop dimensions (not always ArcFace's aligned 112x112).
            metrics = self._estimate_image_metrics(pose_img if pose_img is not None else crop)
        except Exception:
            metrics = {"sharpness": 0.0, "brightness": 0.0, "dark_ratio": 0.0, "face_size": 0, "quality": 0.0}

        # Store original crop as a short-lived Redis key for on-demand downstream access.
        # For Unknown results: only store if eye-region texture looks like real eyes
        # (not back-of-head hair/skin at fake landmark positions).
        face_crop_key = ""
        if self._face_crop_ttl > 0 and event_id:
            raw_jpeg = fields.get(b"face_crop_jpeg") or fields.get("face_crop_jpeg")
            if raw_jpeg:
                store_crop = True
                if identity == "Unknown":
                    kps = self._parse_landmarks(fields, pose_img if pose_img is not None else crop)
                    if kps is not None and not self._has_eye_texture(kps, pose_img if pose_img is not None else crop):
                        store_crop = False
                        logger.debug(f"[{event_id}] Unknown: skipping crop storage — no eye texture (likely back-of-head).")
                if store_crop:
                    try:
                        face_crop_key = f"face_crop:{event_id}"
                        self._writer.set_key(face_crop_key, bytes(raw_jpeg), self._face_crop_ttl)
                    except Exception as exc:
                        logger.warning(f"[{event_id}] Failed to store face_crop key: {exc}")
                        face_crop_key = ""

        # Pass through the crop-local normalized landmarks so downstream can call
        # /faces/register_from_crop directly without re-running detection.
        landmark_passthrough = (
            fields.get(b"landmark_5_xy_crop") or fields.get("landmark_5_xy_crop") or b""
        )
        if isinstance(landmark_passthrough, str):
            landmark_passthrough = landmark_passthrough.encode()

        result = {
            b"event_id":          fields.get(b"event_id", b""),
            b"camera_id":         fields.get(b"camera_id",  b""),
            b"track_id":          fields.get(b"track_id",   b""),
            b"timestamp":         fields.get(b"timestamp",  b""),
            b"identity":          identity.encode(),
            b"identity_id":       identity.encode(),   # DEBUG: name instead of UUID for easy tracing
            b"confidence":        str(round(float(confidence), 6)).encode(),
            b"yaw":               str(round(yaw,   2)).encode(),
            b"pitch":             str(round(pitch, 2)).encode(),
            b"roll":              str(round(roll,  2)).encode(),
            b"quality":           str(metrics["quality"]).encode(),
            b"sharpness":         str(metrics["sharpness"]).encode(),
            b"brightness":        str(metrics["brightness"]).encode(),
            b"dark_ratio":        str(metrics["dark_ratio"]).encode(),
            b"face_size":         str(metrics["face_size"]).encode(),
            b"face_crop_key":     face_crop_key.encode(),
            b"landmark_5_xy_crop": landmark_passthrough,
            # b"embedding":         json.dumps(embedding.tolist()).encode(),
            # b"reid_embedding":    json.dumps(reid_emb.tolist()).encode(),
        }
        self._writer.write(result)
        with self._stats_lock:
            self._stats_frames += 1
            count, best = self._stats_identities.get(identity, (0, 0.0))
            self._stats_identities[identity] = (count + 1, max(best, float(confidence)))
            self._stats_identity_pose[identity] = (float(yaw), float(pitch), float(roll))

        # Cache all tracks (including Unknown) so prolonged stationary faces
        # don't monopolise GPU inference slots on every frame.
        # Unknown tracks use a shorter recheck so newly enrolled persons are
        # recognised promptly.
        cam_raw   = fields.get(b"camera_id") or fields.get("camera_id") or b""
        track_raw = fields.get(b"track_id")  or fields.get("track_id")  or b""
        camera_id = cam_raw.decode(errors="replace")   if isinstance(cam_raw,   (bytes, bytearray)) else str(cam_raw)
        track_id  = track_raw.decode(errors="replace") if isinstance(track_raw, (bytes, bytearray)) else str(track_raw)
        self._track_cache_put(camera_id, track_id, time.monotonic(), {
            "identity":          identity,
            "identity_id":       identity_id,
            "confidence":        confidence,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "quality":           metrics["quality"],
            "sharpness":         metrics["sharpness"],
            "brightness":        metrics["brightness"],
            "dark_ratio":        metrics["dark_ratio"],
            "face_size":         metrics["face_size"],
            "landmark_5_xy_crop": landmark_passthrough.decode(errors="replace") if landmark_passthrough else "",
        })

    def _publish_cached(self, fields: dict, cached: dict):
        """Re-publish a previously cached identity result without running inference.

        For Unknown tracks: perform a cheap CPU quality check on the raw JPEG.
        If this frame is sharper than the best seen so far for this track, store
        it as a face_crop_key so downstream enrollment can use it as a candidate.
        Known tracks: no crop stored (identity already confirmed, no need).
        """
        event_id = fields.get(b"event_id", b"").decode(errors="replace")
        face_crop_key = b""

        if cached["identity"] == "Unknown" and self._face_crop_ttl > 0 and event_id:
            raw_jpeg = fields.get(b"face_crop_jpeg") or fields.get("face_crop_jpeg")
            if raw_jpeg:
                try:
                    img = self._decode_jpeg(bytes(raw_jpeg))
                    if img is not None:
                        m = self._estimate_image_metrics(img)
                        # Only store if this frame is sharper than the cached best
                        if m["sharpness"] > cached.get("best_crop_quality", 0.0):
                            key = f"face_crop:{event_id}"
                            self._writer.set_key(key, bytes(raw_jpeg), self._face_crop_ttl)
                            face_crop_key = key.encode()
                            cam_raw   = fields.get(b"camera_id") or fields.get("camera_id") or b""
                            track_raw = fields.get(b"track_id")  or fields.get("track_id")  or b""
                            camera_id = cam_raw.decode(errors="replace")   if isinstance(cam_raw,   (bytes, bytearray)) else str(cam_raw)
                            track_id  = track_raw.decode(errors="replace") if isinstance(track_raw, (bytes, bytearray)) else str(track_raw)
                            with self._track_cache_lock:
                                entry = self._track_cache.get((camera_id, track_id))
                                if entry is not None:
                                    entry["best_crop_quality"] = m["sharpness"]
                except Exception:
                    pass

        # Re-use the landmark that was cached alongside the identity so downstream
        # still has it available for register_from_crop on cached frames.
        cached_landmark = cached.get("landmark_5_xy_crop", "")
        if isinstance(cached_landmark, str):
            cached_landmark = cached_landmark.encode()

        result = {
            b"event_id":          fields.get(b"event_id", b""),
            b"camera_id":         fields.get(b"camera_id",  b""),
            b"track_id":          fields.get(b"track_id",   b""),
            b"timestamp":         fields.get(b"timestamp",  b""),
            b"identity":          cached["identity"].encode(),
            b"identity_id":       cached["identity"].encode(),
            b"confidence":        str(round(float(cached["confidence"]), 6)).encode(),
            b"yaw":               str(round(cached["yaw"],   2)).encode(),
            b"pitch":             str(round(cached["pitch"], 2)).encode(),
            b"roll":              str(round(cached["roll"],  2)).encode(),
            b"quality":           str(cached["quality"]).encode(),
            b"sharpness":         str(cached["sharpness"]).encode(),
            b"brightness":        str(cached["brightness"]).encode(),
            b"dark_ratio":        str(cached["dark_ratio"]).encode(),
            b"face_size":         str(cached["face_size"]).encode(),
            b"face_crop_key":     face_crop_key,
            b"landmark_5_xy_crop": cached_landmark,
            b"cached":            b"1",
        }
        self._writer.write(result)
        with self._stats_lock:
            self._stats_frames += 1
            identity = cached["identity"]
            count, best = self._stats_identities.get(identity, (0, 0.0))
            self._stats_identities[identity] = (count + 1, max(best, float(cached["confidence"])))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        self._stop_event.set()
