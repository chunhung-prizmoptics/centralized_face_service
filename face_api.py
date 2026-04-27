"""
face_api.py – Lightweight FastAPI enrollment server for the face database.

Endpoints:
  POST   /faces/register      Register a new identity from uploaded images.
  POST   /faces/update        Append or replace embeddings for an existing identity.
  DELETE /faces/{name}        Remove an identity.
  GET    /faces               List all registered identities.
  GET    /health              Health check.

The server runs in a background daemon thread and shares the FaceDatabase
instance with the main recognizer. A threading.Lock protects the shared
InsightFace model so both threads can safely call face_app.get().

Standalone usage:
    conda run -n face_rtsp_env python face_api.py --host 127.0.0.1 --port 8000
"""

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import io
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from insightface.utils import face_align
from loguru import logger

from face_db import FaceDatabase

app = FastAPI(title="Face Recognition Enrollment API", version="1.0.0")

# Injected via init_api() before the server thread starts
_face_db: FaceDatabase = None
_face_app = None
_model_lock = threading.Lock()  # shared with realtime_face_recognition.py

# Y-axis aligned to image coordinates (downward-positive) for solvePnP consistency.
_FACE_3D_MODEL = np.array([
    [-165.0, -170.0, -135.0],
    [ 165.0, -170.0, -135.0],
    [   0.0,    0.0,    0.0],
    [-150.0,  150.0, -125.0],
    [ 150.0,  150.0, -125.0],
], dtype=np.float32)


def init_api(
    face_db: FaceDatabase,
    face_app,
    model_lock: threading.Lock = None,
):
    global _face_db, _face_app, _model_lock
    _face_db = face_db
    _face_app = face_app
    if model_lock is not None:
        _model_lock = model_lock


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_pose_from_kps(kps: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) from 5-point landmarks in absolute image pixels."""
    focal = float(img_w)
    cam_matrix = np.array([
        [focal, 0,     img_w / 2],
        [0,     focal, img_h / 2],
        [0,     0,     1        ],
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, _ = cv2.solvePnP(_FACE_3D_MODEL, kps.astype(np.float32), cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)

    # InsightFace matrix2angle convention: x=pitch, y=yaw, z=roll
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
        yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
        yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        roll = 0.0
    return yaw, pitch, roll


def _extract_embeddings(image_bytes: bytes) -> tuple[list, list[dict]]:
    """Decode image bytes, detect faces, and return embeddings plus per-face quality diagnostics."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image – unsupported format or corrupt file.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    with _model_lock:
        faces = _face_app.get(img)
    if not faces:
        raise ValueError("No face detected in the provided image.")

    selected = []
    diagnostics: list[dict] = []
    for f in faces:
        emb = getattr(f, "embedding", None)
        bbox = getattr(f, "bbox", None)
        if emb is None or bbox is None:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        bw = float(x2 - x1)
        bh = float(y2 - y1)

        face_gray = gray[y1:y2, x1:x2]
        if face_gray.size == 0:
            diagnostics.append({"valid_crop": False})
            continue

        sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        brightness = float(face_gray.mean())
        dark_ratio = float((face_gray < 40).mean())
        yaw, pitch, roll = 0.0, 0.0, 0.0

        kps = getattr(f, "kps", None)
        if kps is not None and np.asarray(kps).shape == (5, 2):
            yaw, pitch, roll = _estimate_pose_from_kps(np.asarray(kps, dtype=np.float32), img_w, img_h)

        diagnostics.append(
            {
                "valid_crop": True,
                "bbox": [x1, y1, x2, y2],
                "size_w": round(bw, 1),
                "size_h": round(bh, 1),
                "size_min": round(min(bw, bh), 1),
                "sharpness": round(sharpness, 2),
                "brightness": round(brightness, 2),
                "dark_ratio": round(dark_ratio, 4),
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "roll": round(roll, 2),
            }
        )

        selected.append(emb)

    if not selected:
        raise ValueError("No valid face embeddings extracted from detected faces.")

    return selected, diagnostics


def _parse_landmarks_for_crop(raw: str, img_w: int, img_h: int) -> np.ndarray:
    """
    Parse 5-point landmarks and return absolute pixel coordinates in crop space.

    Accepts either:
      1. JSON: [[x0,y0],...,[x4,y4]]
      2. CSV:  x0,y0,x1,y1,...,x4,y4

    Coordinates may be normalized [0..1] or absolute crop pixels.
    """
    if raw is None or str(raw).strip() == "":
        raise ValueError("landmark_5_xy is required.")

    kps = None
    text = str(raw).strip()

    try:
        kps = np.array(json.loads(text), dtype=np.float32).reshape(5, 2)
    except Exception:
        pass

    if kps is None:
        try:
            vals = [float(v.strip()) for v in text.split(",")]
            if len(vals) != 10:
                raise ValueError("landmark_5_xy must contain exactly 10 numeric values.")
            kps = np.array(vals, dtype=np.float32).reshape(5, 2)
        except Exception as exc:
            raise ValueError(f"Invalid landmark_5_xy format: {exc}")

    # Auto-detect normalized input.
    if float(np.max(kps)) <= 1.5:
        kps[:, 0] *= max(img_w - 1, 1)
        kps[:, 1] *= max(img_h - 1, 1)

    # Clamp into image bounds.
    kps[:, 0] = np.clip(kps[:, 0], 0, max(img_w - 1, 0))
    kps[:, 1] = np.clip(kps[:, 1], 0, max(img_h - 1, 0))
    return kps.astype(np.float32)


def _landmark_geometry_ok(kps: np.ndarray, img_w: int, img_h: int) -> bool:
    """Conservative geometry checks to reject obviously invalid landmark sets."""
    if kps.shape != (5, 2):
        return False

    left_eye, right_eye, nose, left_mouth, right_mouth = kps
    if left_eye[0] >= right_eye[0]:
        return False

    if nose[1] <= left_eye[1] or nose[1] <= right_eye[1]:
        return False

    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
    if mouth_y <= nose[1]:
        return False

    eye_sep = right_eye[0] - left_eye[0]
    if eye_sep < 0.08 * float(max(img_w, 1)):
        return False

    if not np.isfinite(kps).all():
        return False

    return True


def _extract_embedding_from_crop(image_bytes: bytes, landmark_5_xy: str) -> tuple[np.ndarray, dict]:
    """
    Build one ArcFace embedding from a pre-cropped face using provided 5-point landmarks.
    This path intentionally skips face detection to avoid failures on tight crops.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image – unsupported format or corrupt file.")

    img_h, img_w = img.shape[:2]
    if img_h < 24 or img_w < 24:
        raise ValueError("Face crop is too small for reliable embedding extraction.")

    kps = _parse_landmarks_for_crop(landmark_5_xy, img_w, img_h)
    if not _landmark_geometry_ok(kps, img_w, img_h):
        raise ValueError("Invalid facial landmark geometry for this crop.")

    logger.debug(
        f"norm_crop input | img.shape={img.shape} "
        f"kps={kps.tolist()}"
    )

    try:
        aligned = face_align.norm_crop(img, landmark=kps)
    except Exception as exc:
        raise ValueError(
            f"norm_crop failed — likely degenerate landmark positions for this crop size. "
            f"img.shape={img.shape} kps={kps.tolist()} error={exc}"
        )

    if aligned is None or aligned.size == 0 or aligned.shape[:2] != (112, 112):
        raise ValueError(
            f"norm_crop returned unexpected shape {getattr(aligned, 'shape', None)} — "
            f"img.shape={img.shape} kps={kps.tolist()}"
        )

    rec_model = _face_app.models.get("recognition") if _face_app is not None else None
    if rec_model is None:
        raise ValueError("Recognition model is not available.")

    with _model_lock:
        emb = rec_model.get_feat([aligned])  # pass as list, same as batch path in inference worker
    emb = np.array(emb).flatten().astype(np.float32)

    emb_norm = float(np.linalg.norm(emb))
    if emb.size == 0 or emb_norm < 1e-3:
        raise ValueError("Generated embedding is degenerate and cannot be used for recognition.")

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    dark_ratio = float((gray < 40).mean())

    diagnostics = {
        "aligned_size": [int(aligned.shape[1]), int(aligned.shape[0])],
        "embedding_dim": int(emb.shape[0]),
        "embedding_norm": round(emb_norm, 6),
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "dark_ratio": round(dark_ratio, 4),
    }
    return emb, diagnostics


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/faces/register")
async def register_face(
    name: str = Form(..., description="Identity name"),
    id: str = Form(None, description="Optional UUID for this identity. A new UUID is generated if omitted."),
    files: List[UploadFile] = File(..., description="One or more face images"),
):
    """Register a new identity. Fails if the name already exists."""
    if _face_db is None:
        raise HTTPException(503, "Face DB not initialized.")

    # Validate provided ID or generate a fresh one
    if id is not None:
        try:
            identity_id = str(uuid.UUID(id))
        except ValueError:
            raise HTTPException(400, f"Invalid UUID format: '{id}'.")
    else:
        identity_id = str(uuid.uuid4())

    embeddings, warnings, quality_reports = [], [], []
    for f in files:
        try:
            data = await f.read()
            embs, diag = _extract_embeddings(data)
            embeddings.extend(embs)
            quality_reports.append({"file": f.filename, "faces": diag})
        except Exception as exc:
            warnings.append(f"{f.filename}: {exc}")

    if not embeddings:
        raise HTTPException(400, f"No valid face embeddings extracted. Details: {warnings}")

    success = _face_db.register_identity(name, embeddings, identity_id=identity_id)
    if not success:
        raise HTTPException(
            409,
            f"Identity '{name}' already exists. Use POST /faces/update to modify it.",
        )

    return {
        "success": True,
        "message": f"Identity '{name}' registered.",
        "name": name,
        "id": identity_id,
        "embeddings_added": len(embeddings),
        "timestamp": _now_iso(),
        "warnings": warnings,
        "quality_reports": quality_reports,
    }


@app.post("/faces/register_from_crop")
async def register_from_crop(
    name: str = Form(..., description="Identity name"),
    landmark_5_xy: str = Form(
        ...,
        description=(
            "5-point facial landmarks for this crop; JSON [[x,y]x5] or CSV x0,y0,...,x4,y4. "
            "Supports normalized [0..1] or absolute crop pixels."
        ),
    ),
    id: str = Form(None, description="Optional UUID for this identity. A new UUID is generated if omitted."),
    file: UploadFile = File(..., description="One pre-cropped face image."),
):
    """
    Register identity from a tight crop + landmarks without running detector.
    This avoids detector misses on tightly cropped faces.
    """
    logger.debug(f"POST /faces/register_from_crop received | name={name!r} id={id!r} file={file.filename!r} landmark={landmark_5_xy!r}")

    if _face_db is None:
        raise HTTPException(503, "Face DB not initialized.")

    if id is not None:
        try:
            identity_id = str(uuid.UUID(id))
        except ValueError:
            raise HTTPException(400, f"Invalid UUID format: '{id}'.")
    else:
        identity_id = str(uuid.uuid4())

    try:
        data = await file.read()
        emb, diagnostics = _extract_embedding_from_crop(data, landmark_5_xy)
    except Exception as exc:
        raise HTTPException(400, f"register_from_crop failed: {exc}")

    success = _face_db.register_identity(name, [emb], identity_id=identity_id)
    if not success:
        raise HTTPException(
            409,
            f"Identity '{name}' already exists. Use POST /faces/update to modify it.",
        )

    return {
        "success": True,
        "message": f"Identity '{name}' registered from crop.",
        "name": name,
        "id": identity_id,
        "embeddings_added": 1,
        "timestamp": _now_iso(),
        "quality_report": {
            "file": file.filename,
            "mode": "landmark_aligned_crop",
            "diagnostics": diagnostics,
        },
    }


@app.post("/faces/update")
async def update_face(
    name: str = Form(..., description="Identity name"),
    mode: str = Form("append", description="'append' or 'replace'"),
    files: List[UploadFile] = File(..., description="One or more face images"),
):
    """Append new images to or fully replace the embeddings of an existing identity."""
    if _face_db is None:
        raise HTTPException(503, "Face DB not initialized.")
    if mode not in ("append", "replace"):
        raise HTTPException(400, "mode must be 'append' or 'replace'.")

    embeddings, warnings, quality_reports = [], [], []
    for f in files:
        try:
            data = await f.read()
            embs, diag = _extract_embeddings(data)
            embeddings.extend(embs)
            quality_reports.append({"file": f.filename, "faces": diag})
        except Exception as exc:
            warnings.append(f"{f.filename}: {exc}")

    if not embeddings:
        raise HTTPException(400, f"No valid face embeddings extracted. Details: {warnings}")

    _face_db.update_identity(name, embeddings, mode=mode)

    return {
        "success": True,
        "message": f"Identity '{name}' updated ({mode}).",
        "name": name,
        "embeddings_added": len(embeddings),
        "timestamp": _now_iso(),
        "warnings": warnings,
        "quality_reports": quality_reports,
    }


@app.delete("/faces/{name}")
async def delete_face(name: str):
    """Remove an identity from the database."""
    if _face_db is None:
        raise HTTPException(503, "Face DB not initialized.")
    if not _face_db.delete_identity(name):
        raise HTTPException(404, f"Identity '{name}' not found.")
    return {
        "success": True,
        "message": f"Identity '{name}' deleted.",
        "name": name,
        "timestamp": _now_iso(),
    }


@app.get("/faces")
async def list_faces():
    """List all registered identities and their embedding counts."""
    if _face_db is None:
        raise HTTPException(503, "Face DB not initialized.")
    identities = _face_db.list_identities()
    return {
        "success": True,
        "count": len(identities),
        "identities": identities,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now_iso()}


# ------------------------------------------------------------------
# Server launcher
# ------------------------------------------------------------------

def start_api_server(
    face_db: FaceDatabase,
    face_app,
    model_lock: threading.Lock,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Start the enrollment API in a background daemon thread."""
    init_api(face_db, face_app, model_lock)

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True, name="face-api")
    thread.start()

    # Brief wait to let uvicorn bind the socket
    time.sleep(1.5)
    logger.info(f"Face Enrollment API → http://{host}:{port}  (docs: /docs)")
    return thread


# ------------------------------------------------------------------
# Standalone entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from face_db import FaceDatabase

    parser = argparse.ArgumentParser(description="Face Enrollment API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host  [127.0.0.1]")
    parser.add_argument("--port", type=int, default=8000, help="Bind port  [8000]")
    parser.add_argument(
        "--gallery", default="gallery",
        help="Gallery directory to pre-load on startup  [gallery]",
    )
    parser.add_argument(
        "--det-size", type=int, default=640,
        help="InsightFace detection input size  [640]",
    )
    parser.add_argument(
        "--det-thresh", type=float, default=0.5,
        help="InsightFace detection confidence threshold  [0.5]",
    )
    args = parser.parse_args()

    logger.info("Loading InsightFace model (buffalo_l)...")
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)
    logger.info("InsightFace ready.")

    face_db = FaceDatabase(cache_path="face_db_cache.npz")
    from pathlib import Path
    gallery = Path(args.gallery)
    if gallery.exists():
        n = face_db.build_from_gallery(str(gallery), face_app=face_app, dedup_threshold=0.75)
        logger.info(f"Gallery loaded: {n} identities.")
    else:
        logger.warning(f"Gallery '{args.gallery}' not found – starting with empty DB.")

    model_lock = threading.Lock()
    init_api(face_db, face_app, model_lock)

    logger.info(f"Starting Face Enrollment API on http://{args.host}:{args.port} (docs: /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
