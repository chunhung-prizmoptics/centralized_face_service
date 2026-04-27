# Tests Utilities

This folder contains utility scripts for load testing and pose-audit data collection.

## 1) Collect Pose Audit Samples

Script: `collect_pose_audit.py`

Purpose:
- Read recent rows from `face_inference:results`
- Collect `yaw/pitch/roll` and metadata
- Fetch face crops via `face_crop_key`
- Save raw images, annotated images, and a CSV for manual accuracy inspection

### Usage

```bash
python centralized_face_service/tests/collect_pose_audit.py --count 800 --only-unknown --max-abs-yaw 5 --max-abs-pitch 5 --max-abs-roll 5 --max-age-seconds 45 --output-dir centralized_face_service/tests/pose_audit
```

If strict filters return no rows, enable wait mode so the script keeps polling
new stream entries until a match is collected (or timeout):

```bash
python centralized_face_service/tests/collect_pose_audit.py --count 800 --only-unknown --max-abs-yaw 5 --max-abs-pitch 5 --max-abs-roll 5 --max-age-seconds 45 --wait-seconds 120 --poll-interval-seconds 1.0 --output-dir centralized_face_service/tests/pose_audit
```

If you also want only saturated sharpness rows (`sharpness == 1.0`):

```bash
python centralized_face_service/tests/collect_pose_audit.py --count 800 --only-unknown --max-abs-yaw 5 --max-abs-pitch 5 --max-abs-roll 5 --sharpness-eq 1.0 --max-age-seconds 45 --wait-seconds 120 --poll-interval-seconds 1.0 --output-dir centralized_face_service/tests/pose_audit
```

### Output

Each run creates:
- `run_YYYYmmdd_HHMMSS/samples.csv`
- `run_YYYYmmdd_HHMMSS/raw/` (raw face crops)
- `run_YYYYmmdd_HHMMSS/vis/` (pose text overlay)

Important CSV columns:
- `yaw`, `pitch`, `roll`
- `landmark_5_xy_crop`
- `face_crop_key`
- `face_crop_ttl_sec`
- `age_sec`
- `note`

Common `note` values:
- `ok`
- `missing_face_crop_key`
- `face_crop_expired:<key>`
- `face_crop_not_found:<key>`
- `face_crop_decode_failed`

Why crops may be missing:
- `face_crop_key` is TTL-based (`FACE_CROP_TTL` in config, default short-lived)
- `face_inference:results` keeps much longer history than crop keys

Recommendation:
- Keep `--max-age-seconds` below crop TTL (for example `45` when TTL is `60`)

---

## 2) Redis Replay Load Test

Script: `redis_stream_load_replay.py`

Purpose:
- Re-inject sampled stream rows at controlled rate to stress-test throughput

### Usage

```bash
python centralized_face_service/tests/redis_stream_load_replay.py \
  --redis-url redis://localhost:6379 \
  --stream face_inference:requests \
  --sample-size 100 \
  --rate 100 \
  --burst-size 20 \
  --duration 120 \
  --print-every 2
```

### Staircase Example

1. `--rate 100 --duration 120`
2. `--rate 200 --duration 120`
3. `--rate 300 --duration 120`
4. `--rate 400 --duration 120`

Watch service logs:
- `Stats | ... req_queue=...`
- Aggregate worker FPS
