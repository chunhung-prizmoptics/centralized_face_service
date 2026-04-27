"""
Collect pose results from Redis stream and save crops for manual inspection.

Usage example:
  python centralized_face_service/tests/collect_pose_audit.py \
      --redis-url redis://localhost:6379 \
      --stream face_inference:results \
      --count 300 \
      --output-dir centralized_face_service/tests/pose_audit
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import redis


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return str(v)


def _get_field(fields: dict, key: str) -> str:
    # Redis client returns bytes keys/values with decode_responses=False.
    return _to_str(fields.get(key.encode()) if key.encode() in fields else fields.get(key))


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _to_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except Exception:
        return default


def _to_int(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except Exception:
        return default


def _decode_jpeg(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def _draw_pose_overlay(img: np.ndarray, yaw: str, pitch: str, roll: str, identity: str, confidence: str) -> np.ndarray:
    vis = img.copy()
    text1 = f"id={identity} conf={confidence}"
    text2 = f"yaw={yaw} pitch={pitch} roll={roll}"
    cv2.putText(vis, text1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect yaw/pitch/roll + face crops from face_inference:results")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL [redis://localhost:6379]")
    parser.add_argument("--stream", default="face_inference:results", help="Result stream name [face_inference:results]")
    parser.add_argument("--count", type=int, default=200, help="How many most-recent result rows to inspect [200]")
    parser.add_argument("--output-dir", default="centralized_face_service/tests/pose_audit", help="Output directory")
    parser.add_argument("--only-unknown", action="store_true", help="Keep only entries where identity == Unknown")
    parser.add_argument("--min-abs-yaw", type=float, default=0.0, help="Filter rows by absolute yaw >= value [0]")
    parser.add_argument("--min-abs-pitch", type=float, default=0.0, help="Filter rows by absolute pitch >= value [0]")
    parser.add_argument("--max-abs-yaw", type=float, default=0.0, help="Filter rows by absolute yaw <= value [0 disables]")
    parser.add_argument("--max-abs-pitch", type=float, default=0.0, help="Filter rows by absolute pitch <= value [0 disables]")
    parser.add_argument("--max-abs-roll", type=float, default=0.0, help="Filter rows by absolute roll <= value [0 disables]")
    parser.add_argument(
        "--sharpness-eq",
        type=float,
        default=-1.0,
        help="Keep rows where sharpness is approximately this value (e.g. 1.0). Negative disables.",
    )
    parser.add_argument(
        "--sharpness-eps",
        type=float,
        default=1e-6,
        help="Tolerance for --sharpness-eq comparison [1e-6]",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="When >0, keep polling until at least one matching row is collected or timeout [0 disables]",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval when wait mode is enabled [1.0]",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=120.0,
        help="Skip rows older than this many seconds using the result timestamp field [120] (<=0 disables)",
    )
    args = parser.parse_args()

    r = redis.from_url(args.redis_url, decode_responses=False)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"run_{now}"
    raw_dir = out_root / "raw"
    vis_dir = out_root / "vis"
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "samples.csv"
    saved = 0
    total = 0
    now_ms = int(datetime.now().timestamp() * 1000)
    seen_stream_ids: set[str] = set()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stream_id",
                "event_id",
                "camera_id",
                "track_id",
                "timestamp",
                "identity",
                "confidence",
                "yaw",
                "pitch",
                "roll",
                "quality",
                "sharpness",
                "brightness",
                "dark_ratio",
                "face_size",
                "cached",
                "landmark_5_xy_crop",
                "face_crop_key",
                "face_crop_ttl_sec",
                "age_sec",
                "raw_image_path",
                "vis_image_path",
                "note",
            ],
        )
        writer.writeheader()

        deadline = time.monotonic() + args.wait_seconds if args.wait_seconds > 0 else None
        while True:
            # Most recent first, then reverse so files/csv are chronological.
            rows = r.xrevrange(args.stream, max="+", min="-", count=args.count)
            rows = list(reversed(rows))
            saw_new = False

            for msg_id, fields in rows:
                stream_id = _to_str(msg_id)
                if stream_id in seen_stream_ids:
                    continue
                seen_stream_ids.add(stream_id)
                saw_new = True
                total += 1

                event_id = _get_field(fields, "event_id")
                camera_id = _get_field(fields, "camera_id")
                track_id = _get_field(fields, "track_id")
                ts = _get_field(fields, "timestamp")
                identity = _get_field(fields, "identity")
                confidence = _get_field(fields, "confidence")
                yaw = _get_field(fields, "yaw")
                pitch = _get_field(fields, "pitch")
                roll = _get_field(fields, "roll")
                quality = _get_field(fields, "quality")
                sharpness = _get_field(fields, "sharpness")
                brightness = _get_field(fields, "brightness")
                dark_ratio = _get_field(fields, "dark_ratio")
                face_size = _get_field(fields, "face_size")
                cached = _get_field(fields, "cached")
                landmark = _get_field(fields, "landmark_5_xy_crop")
                face_crop_key = _get_field(fields, "face_crop_key")

                ts_ms = _to_int(ts, default=0)
                age_sec = round((now_ms - ts_ms) / 1000.0, 3) if ts_ms > 0 else -1.0

                if args.only_unknown and identity != "Unknown":
                    continue
                yaw_abs = abs(_to_float(yaw))
                pitch_abs = abs(_to_float(pitch))
                roll_abs = abs(_to_float(roll))
                sharpness_val = _to_float(sharpness, default=-1.0)

                if yaw_abs < args.min_abs_yaw:
                    continue
                if pitch_abs < args.min_abs_pitch:
                    continue
                if args.max_abs_yaw > 0 and yaw_abs > args.max_abs_yaw:
                    continue
                if args.max_abs_pitch > 0 and pitch_abs > args.max_abs_pitch:
                    continue
                if args.max_abs_roll > 0 and roll_abs > args.max_abs_roll:
                    continue
                if args.sharpness_eq >= 0 and abs(sharpness_val - args.sharpness_eq) > args.sharpness_eps:
                    continue
                if args.max_age_seconds > 0 and age_sec >= 0 and age_sec > args.max_age_seconds:
                    continue

                base = _safe_name(f"{saved:05d}_{event_id or stream_id}")
                raw_image_path = ""
                vis_image_path = ""
                note = ""
                face_crop_ttl_sec = ""

                if not face_crop_key:
                    note = "missing_face_crop_key"
                else:
                    try:
                        ttl = int(r.ttl(face_crop_key))
                        face_crop_ttl_sec = str(ttl)
                    except Exception:
                        ttl = -999
                        face_crop_ttl_sec = "-999"

                    raw = r.get(face_crop_key)
                    if not raw:
                        if ttl == -2:
                            note = f"face_crop_expired:{face_crop_key}"
                        else:
                            note = f"face_crop_not_found:{face_crop_key}"
                    else:
                        img = _decode_jpeg(raw)
                        if img is None:
                            note = "face_crop_decode_failed"
                            bin_path = raw_dir / f"{base}.bin"
                            bin_path.write_bytes(raw)
                            raw_image_path = os.fspath(bin_path)
                        else:
                            raw_path = raw_dir / f"{base}.jpg"
                            vis_path = vis_dir / f"{base}.jpg"
                            cv2.imwrite(os.fspath(raw_path), img)
                            cv2.imwrite(
                                os.fspath(vis_path),
                                _draw_pose_overlay(img, yaw=yaw, pitch=pitch, roll=roll, identity=identity, confidence=confidence),
                            )
                            raw_image_path = os.fspath(raw_path)
                            vis_image_path = os.fspath(vis_path)
                            note = "ok"

                writer.writerow(
                    {
                        "stream_id": stream_id,
                        "event_id": event_id,
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "timestamp": ts,
                        "identity": identity,
                        "confidence": confidence,
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "quality": quality,
                        "sharpness": sharpness,
                        "brightness": brightness,
                        "dark_ratio": dark_ratio,
                        "face_size": face_size,
                        "cached": cached,
                        "landmark_5_xy_crop": landmark,
                        "face_crop_key": face_crop_key,
                        "face_crop_ttl_sec": face_crop_ttl_sec,
                        "age_sec": age_sec,
                        "raw_image_path": raw_image_path,
                        "vis_image_path": vis_image_path,
                        "note": note,
                    }
                )
                saved += 1

            if args.wait_seconds <= 0:
                break
            if saved > 0:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not saw_new:
                time.sleep(max(args.poll_interval_seconds, 0.05))

    print(f"Collected {saved} rows (scanned {total})")
    print(f"CSV : {csv_path}")
    print(f"Raw : {raw_dir}")
    print(f"Vis : {vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
