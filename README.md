# Centralized GPU Face Recognition Service

A Redis Streams–based service that receives pre-cropped face images (with
optional 5-point landmarks) from an upstream C++ tracker, runs batched
InsightFace ArcFace inference on GPU, matches identities against a gallery,
and publishes results back to Redis.

---

## Architecture

```
C++ Producer                         Python Service                   Redis Stream
(ByteTracker + RetinaFace)           centralized_face_service/
    │                                        │
    ├─ face_crop_jpeg (raw bytes)            ├── landmark align (norm_crop)
    ├─ landmark_5_xy  (normalized)    ──►    ├── ArcFace (buffalo_l, GPU, batched)
    └─ metadata                             ├── FaceDatabase (cosine match)
         vision:track_events  ──────────►   └── publish ──► face_inference:results
```

The service reads from `vision:track_events`, aligns each crop using the
producer-supplied 5-point landmarks, runs batched ArcFace on GPU, and writes
the identity match result to `face_inference:results`.

---

## Prerequisites

- NVIDIA GPU with CUDA ≥ 11.8
- Conda environment `face_rtsp_env` (with `insightface`, `onnxruntime-gpu`, `redis`, etc.)
- **Redis 7** running:

```bash
docker run -d -p 6379:6379 redis:7
```

---

## Install

```bash
cd centralized_face_service
conda run -n face_rtsp_env pip install -r requirements.txt
```

---

## Gallery Setup

```
gallery/
├── Alice/
│   ├── front.jpg
│   └── side.jpg
└── Bob/
    └── bob.jpg
```

Each subfolder name becomes the identity label. Supported formats: `.jpg`, `.jpeg`, `.png`.

---

## Running

All defaults are set in `config.py`. Just run:

```bash
conda run -n face_rtsp_env python centralized_face_service/main.py
```

Override specific settings via CLI:

```bash
conda run -n face_rtsp_env python centralized_face_service/main.py \
    --workers 2         \   # parallel BatchWorker threads
    --batch-size 16     \   # GPU inference batch size
    --threshold 0.40        # cosine similarity threshold
```

Full CLI reference (all defaults come from `config.py`):

| Argument            | Default                    | Description                                       |
|---------------------|----------------------------|---------------------------------------------------|
| `--gallery`         | `gallery`                  | Gallery directory                                 |
| `--cache`           | `face_db_cache.npz`        | Embedding cache path                              |
| `--redis-url`       | `redis://localhost:6379`   | Redis connection URL                              |
| `--request-stream`  | `vision:track_events`      | Input Redis stream                                |
| `--result-stream`   | `face_inference:results`   | Output Redis stream                               |
| `--consumer-group`  | `face_inference_workers`   | Redis consumer group                              |
| `--batch-size`      | `8`                        | Max images per GPU inference call                 |
| `--workers`         | `1`                        | Parallel BatchWorker threads                      |
| `--decode-workers`  | `4`                        | CPU threads for JPEG decode per worker            |
| `--threshold`       | `0.35`                     | Min cosine score to count as a match              |
| `--det-size`        | `640`                      | InsightFace detection input size                  |
| `--gpu-id`          | `0`                        | CUDA device index                                 |
| `--no-reid`         | `True`                     | Skip ReID embedding (default on)                  |
| `--detect`          | `False`                    | Run full SCRFD detection instead of landmark path |
| `--result-maxlen`   | `10000`                    | Max entries kept in result stream                 |
| `--request-maxlen`  | `1000`                     | Max JPEG crops kept in request stream             |

> **`--detect` flag:**  
> Off (default) — assumes producer sends a tight face crop + `landmark_5_xy`;
> uses landmark alignment (`norm_crop`) directly into ArcFace. This gives the
> best accuracy and throughput.  
> On — runs full SCRFD detection on each crop first. Use only when crops are
> loose or whole-body images without landmarks.

---

## Configuration

Edit `config.py` directly (preferred) or override via environment variables:

| Variable           | Default                   | Description                            |
|--------------------|---------------------------|----------------------------------------|
| `REDIS_URL`        | `redis://localhost:6379`  | Redis connection URL                   |
| `REQUEST_STREAM`   | `vision:track_events`     | Input stream name                      |
| `RESULT_STREAM`    | `face_inference:results`  | Output stream name                     |
| `CONSUMER_GROUP`   | `face_inference_workers`  | Consumer group name                    |
| `BATCH_SIZE`       | `8`                       | Max images per GPU inference call      |
| `BATCH_TIMEOUT_MS` | `50`                      | Max ms to wait before flushing a batch |
| `WORKERS`          | `1`                       | Parallel BatchWorker threads           |
| `DECODE_WORKERS`   | `4`                       | CPU decode threads per worker          |
| `MATCH_THRESHOLD`  | `0.35`                    | Cosine similarity threshold            |
| `GALLERY_DIR`      | `gallery`                 | Gallery directory                      |
| `CACHE_PATH`       | `face_db_cache.npz`       | Embedding cache file                   |
| `DET_SIZE`         | `640`                     | InsightFace detection size             |
| `GPU_ID`           | `0`                       | CUDA device index                      |
| `NO_REID`          | `1`                       | Set to `0` to enable ReID embeddings   |
| `RUN_DETECTION`    | `0`                       | Set to `1` to enable SCRFD detection   |
| `RESULT_MAXLEN`    | `10000`                   | Max result stream entries              |
| `REQUEST_MAXLEN`   | `1000`                    | Max request stream entries             |

---

## Request Format (Input Stream)

Stream: `vision:track_events`  
All field values are raw bytes (not base64, not JSON-encoded strings).

| Field            | Type        | Required | Description                                                  |
|------------------|-------------|----------|--------------------------------------------------------------|
| `event_id`       | string      | yes      | Unique event ID (e.g. `camera_trackId_timestamp_seq`)        |
| `camera_id`      | string      | yes      | Camera or source identifier                                  |
| `track_id`       | string/int  | yes      | Person track ID from upstream tracker                        |
| `timestamp`      | string/int  | yes      | Unix timestamp in milliseconds                               |
| `face_crop_jpeg` | bytes       | yes      | Raw JPEG-encoded face crop                                   |
| `landmark_5_xy`  | string      | no       | 5-point landmarks, normalized [0..1] within the crop as CSV: `x0,y0,x1,y1,...,x4,y4` (order: left_eye, right_eye, nose, left_mouth, right_mouth). When present, enables precise `norm_crop` alignment. |

---

## Result Format (Output Stream)

Stream: `face_inference:results`

| Field            | Type   | Description                                                                 |
|------------------|--------|-----------------------------------------------------------------------------|
| `event_id`       | string | Echoed from the request                                                     |
| `camera_id`      | string | Echoed from the request                                                     |
| `track_id`       | string | Echoed from the request                                                     |
| `timestamp`      | string | Echoed from the request                                                     |
| `identity`       | string | Matched identity name, or `"Unknown"`                                       |
| `identity_id`    | string | UUID of the matched identity (empty string if Unknown)                      |
| `confidence`     | string | Cosine similarity score (float, 0–1)                                        |
| `yaw`            | string | Head yaw in degrees (positive = face turned right). `0.0` if no landmarks. |
| `pitch`          | string | Head pitch in degrees (positive = face tilted up). `0.0` if no landmarks.  |
| `roll`           | string | Head roll in degrees (positive = face tilted clockwise). `0.0` if no landmarks. |
| `quality`        | string | Sharpness score (float, 0–1). Laplacian variance, saturates at 500. `1.0` = very sharp. |
| `embedding`      | JSON   | 512-D ArcFace embedding as a JSON float array                               |
| `reid_embedding` | JSON   | 256-D ReID embedding (all-zeros if `--no-reid`)                             |

---

## Face Enrollment API

The enrollment API lets you register or remove identities at runtime without
restarting the service.

Run it together with the inference service (same process, shared in-memory DB):

```bash
conda run -n face_rtsp_env python main.py --api --api-host 0.0.0.0 --api-port 8000
```

Or run it as a standalone process:

```bash
conda run -n face_rtsp_env python face_api.py \
    --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Register a face

```bash
# Auto-generate identity UUID
curl -X POST http://localhost:8000/faces/register \
  -F "name=Alice" \
  -F "file=@alice.jpg"

# Provide your own UUID (useful for linking to an external system)
curl -X POST http://localhost:8000/faces/register \
  -F "name=Alice" \
  -F "id=550e8400-e29b-41d4-a716-446655440000" \
  -F "file=@alice.jpg"
```

Response:
```json
{ "name": "Alice", "id": "550e8400-e29b-41d4-a716-446655440000", "embeddings_added": 1 }
```

### List identities

```bash
curl http://localhost:8000/faces
```

```json
[
  { "name": "Alice", "id": "550e8400-e29b-41d4-a716-446655440000", "count": 2 },
  { "name": "Bob",   "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "count": 1 }
]
```

### Delete an identity

```bash
curl -X DELETE http://localhost:8000/faces/Alice
```

### Health check

```bash
curl http://localhost:8000/health
```


### Check Redis

```
docker exec fusion-redis redis-cli --raw XREVRANGE face_inference:results + - COUNT 1
```