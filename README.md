# Centralized GPU Face Recognition Service

A Redis Streams–based service that receives pre-cropped face images, runs
batched InsightFace ArcFace inference on GPU, matches identities against a
gallery, and optionally generates 256-D person ReID embeddings.

---

## Prerequisites

- NVIDIA GPU with CUDA ≥ 11.8
- Conda environment `face_rtsp_env` (with `insightface`, `onnxruntime-gpu`, etc.)
- **Redis 7** running locally:

```bash
docker run -d -p 6379:6379 redis:7
```

---

## Install

```bash
cd centralized_face_service
conda run -n face_rtsp_env pip install -r requirements.txt
```

> **Note on torchreid:** If `pip install torchreid` fails, the service still
> works — ReID embeddings will simply be zero vectors. You can also pass
> `--no-reid` to skip ReID entirely.

---

## Gallery Setup

Same structure as the parent project:

```
gallery/
├── Alice/
│   ├── alice1.jpg
│   └── alice2.jpg
└── Bob/
    └── bob1.jpg
```

Each subfolder name becomes the identity label. Supported image formats:
`.jpg`, `.jpeg`, `.png`.

---

## Running

```bash
# From the face-recognition root directory:
conda run -n face_rtsp_env python centralized_face_service/main.py \
    --gallery gallery

# All options:
conda run -n face_rtsp_env python centralized_face_service/main.py \
    --gallery gallery          \
    --cache   face_db_cache.npz \
    --redis-url redis://localhost:6379 \
    --batch-size 8             \
    --det-size   640           \
    --threshold  0.35          \
    --gpu-id     0             \
    --no-reid                  \  # skip ReID embedding generation
    --detect                      # enable full SCRFD face detection on each crop
```

```bash
conda run -n face_rtsp_env python .\centralized_face_service\main.py --gallery gallery --cache face_db_cache.npz --redis-url redis://localhost:6379 --batch-size 8 --det-size 640 --threshold 0.35 --gpu-id 0 --no-reid
```

> **`--detect` flag explained:**
> - **Off (default):** Assumes each `face_crop` is already a tightly cropped face.
>   Skips SCRFD detection and feeds the crop directly to ArcFace (112×112 resize only).
>   Faster and more reliable for pre-cropped inputs from an upstream tracker.
> - **On (`--detect`):** Runs the full InsightFace pipeline — SCRFD detection →
>   landmark alignment → ArcFace. Use this when the upstream service sends
>   full-frame or loosely cropped scene images.

---

## Request Format

Messages are written to the Redis Stream `face_inference:requests` using
**raw binary values** (not base64).

| Field          | Type        | Description                                  |
|----------------|-------------|----------------------------------------------|
| `request_id`   | string      | Unique UUID for this request                 |
| `camera_id`    | string      | Camera or source identifier                  |
| `track_id`     | string/int  | Person track ID from upstream tracker        |
| `timestamp`    | string/int  | Unix timestamp (seconds or ms)               |
| `face_crop`    | bytes       | Raw JPEG-encoded face crop                   |
| `face_quality` | string/float| *(optional)* Quality score from upstream     |

---

## How to Send a Test Request

```python
import uuid, time, cv2, redis

r = redis.from_url("redis://localhost:6379")

# Load or create a face image
img = cv2.imread("path/to/face.jpg")
ret, buf = cv2.imencode(".jpg", img)
jpeg_bytes = buf.tobytes()

r.xadd("face_inference:requests", {
    b"request_id":  str(uuid.uuid4()).encode(),
    b"camera_id":   b"CAM01",
    b"track_id":    b"42",
    b"timestamp":   str(int(time.time())).encode(),
    b"face_crop":   jpeg_bytes,
    b"face_quality": b"0.88",
})
print("Request sent.")

# Read the result
results = r.xread({"face_inference:results": "0"}, count=1, block=5000)
if results:
    _, messages = results[0]
    msg_id, fields = messages[0]
    print("Result:", {k.decode(): v.decode(errors="replace") for k, v in fields.items()
                      if k != b"embedding" and k != b"reid_embedding"})
```

---

## Result Format

Results are published to `face_inference:results`:

| Field           | Type    | Description                                         |
|-----------------|---------|-----------------------------------------------------|
| `request_id`    | string  | Echoed from the request                             |
| `camera_id`     | string  | Echoed from the request                             |
| `track_id`      | string  | Echoed from the request                             |
| `timestamp`     | string  | Echoed from the request                             |
| `identity`      | string  | Matched identity name, `"Unknown"`, or `"error"`    |
| `confidence`    | string  | Cosine similarity score (float, 0–1)                |
| `embedding`     | JSON    | 512-D ArcFace embedding as JSON float array         |
| `reid_embedding`| JSON    | 256-D ReID embedding as JSON float array            |
| `error_message` | string  | *(only on error)* Reason string                     |

---

## Environment Variables

All settings can be overridden via environment variables:

| Variable          | Default                   | Description                  |
|-------------------|---------------------------|------------------------------|
| `REDIS_URL`       | `redis://localhost:6379`  | Redis connection URL         |
| `BATCH_SIZE`      | `8`                       | Max images per inference batch |
| `BATCH_TIMEOUT_MS`| `50`                      | Max ms to wait to fill batch |
| `MATCH_THRESHOLD` | `0.35`                    | Min cosine score for match   |
| `GALLERY_DIR`     | `../gallery`              | Path to gallery directory    |
| `CACHE_PATH`      | `face_db_cache.npz`       | Embedding cache file         |
| `DET_SIZE`        | `640`                     | InsightFace detection size   |
| `GPU_ID`          | `0`                       | CUDA device index            |
| `RUN_DETECTION`   | `0`                       | Set to `1` to enable full SCRFD detection (same as `--detect`) |

---

## Architecture

```
Redis Stream                  Service                      Redis Stream
face_inference:requests  →  [BatchWorker]  →  face_inference:results
                              |
                              ├── InsightFace (buffalo_l, GPU)
                              ├── FaceDatabase (cosine match)
                              └── ReIDModel (OSNet, GPU/CPU)
```
