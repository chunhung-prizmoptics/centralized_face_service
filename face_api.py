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
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger

from face_db import FaceDatabase

app = FastAPI(title="Face Recognition Enrollment API", version="1.0.0")

# Injected via init_api() before the server thread starts
_face_db: FaceDatabase = None
_face_app = None
_model_lock = threading.Lock()  # shared with realtime_face_recognition.py


def init_api(face_db: FaceDatabase, face_app, model_lock: threading.Lock = None):
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


def _extract_embeddings(image_bytes: bytes) -> list:
    """Decode image bytes, run face detection, return list of embeddings."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image – unsupported format or corrupt file.")
    with _model_lock:
        faces = _face_app.get(img)
    if not faces:
        raise ValueError("No face detected in the provided image.")
    return [f.embedding for f in faces if f.embedding is not None]


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

    embeddings, warnings = [], []
    for f in files:
        try:
            data = await f.read()
            embs = _extract_embeddings(data)
            embeddings.extend(embs)
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

    embeddings, warnings = [], []
    for f in files:
        try:
            data = await f.read()
            embs = _extract_embeddings(data)
            embeddings.extend(embs)
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
