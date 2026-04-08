"""
face_db.py – Thread-safe in-memory face identity store.

Matching strategy:
  1. Normalize all embeddings to unit length.
  2. Per-identity prototype (mean of normalized embeddings) for fast first-pass.
  3. Top-k cosine-similarity voting against individual embeddings for final score.
  4. Quality-aware filtering: embeddings with near-zero norm are discarded on insert.

Cache diff:
  A JSON manifest (alongside the .npz cache) tracks each identity's image files,
  their sizes and modification times. On startup, build_from_gallery:
    - Loads unchanged identities instantly from cache.
    - Re-processes only new or modified identity folders.
    - Removes identities whose gallery folders were deleted.
"""

import json
import threading
import uuid
from pathlib import Path

import numpy as np
from loguru import logger


class FaceDatabase:
    def __init__(self, cache_path: str = None):
        self._lock = threading.RLock()
        # name -> {"id": str, "embeddings": ndarray (N,512), "prototype": ndarray (512,), "count": int}
        self._identities: dict = {}
        self._cache_path = cache_path
        self._manifest_path = (
            Path(cache_path).with_suffix(".manifest.json") if cache_path else None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)

    def _store_identity(self, name: str, embeddings: np.ndarray, identity_id: str = None):
        """Store normalized embeddings and rebuild prototype. Caller may hold lock."""
        with self._lock:
            # Preserve existing ID if already registered and no new ID provided
            if identity_id is None:
                identity_id = self._identities.get(name, {}).get("id") or str(uuid.uuid4())
        prototype = self._normalize(embeddings.mean(axis=0))
        with self._lock:
            self._identities[name] = {
                "id":         identity_id,
                "embeddings": embeddings,
                "prototype":  prototype,
                "count":      len(embeddings),
            }

    def _find_duplicate(self, prototype: np.ndarray, threshold: float) -> str | None:
        """
        Check whether a new identity's prototype is highly similar to any existing
        identity. Returns the name of the best match if similarity >= threshold,
        otherwise None.
        """
        with self._lock:
            if not self._identities:
                return None
            best_name, best_score = None, -1.0
            for name, data in self._identities.items():
                score = float(np.dot(prototype, data["prototype"]))
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score >= threshold:
            return best_name
        return None

    def _assign_ids_from_manifest(self, manifest: dict):
        """
        After loading embeddings from cache, hydrate each identity's `id` field
        from the manifest. Any identity without a manifest entry gets a fresh UUID.
        """
        with self._lock:
            for name, data in self._identities.items():
                if "id" not in data or not data["id"]:
                    data["id"] = manifest.get(name, {}).get("__id__") or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Manifest helpers (cache diff)
    # ------------------------------------------------------------------

    @staticmethod
    def _file_signature(path: Path) -> dict:
        """Return a lightweight fingerprint for a file (size + mtime)."""
        stat = path.stat()
        return {"size": stat.st_size, "mtime": stat.st_mtime}

    @staticmethod
    def _folder_signature(image_files: list) -> dict:
        """Build a manifest entry for a list of image paths."""
        return {
            p.name: FaceDatabase._file_signature(p)
            for p in sorted(image_files)
        }

    @staticmethod
    def _folder_mtime(path: Path) -> float:
        """Return the folder's own modification time (fast pre-check)."""
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    def _load_manifest(self) -> dict:
        """Load the gallery manifest from disk. Returns {} if missing/corrupt."""
        if not self._manifest_path or not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Manifest load failed: {exc}")
            return {}

    def _save_manifest(self, manifest: dict):
        if not self._manifest_path:
            return
        try:
            self._manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            logger.debug(f"Manifest saved → {self._manifest_path}")
        except Exception as exc:
            logger.warning(f"Manifest save failed: {exc}")

    # ------------------------------------------------------------------
    # Gallery build
    # ------------------------------------------------------------------

    def build_from_gallery(self, gallery_dir: str, face_app=None, dedup_threshold: float = 0.75) -> int:
        """
        Scan gallery/<person_name>/*.jpg|jpeg|png, extract embeddings, store.
        If face_app is None, tries to load from cache instead.

        dedup_threshold: cosine similarity above which a new identity is considered
            a duplicate of an existing one and gets merged into it instead of being
            stored separately. Set to 1.0 to disable deduplication.

        Returns number of identities loaded (after deduplication).
        """
        gallery = Path(gallery_dir)
        if not gallery.exists():
            logger.warning(f"Gallery '{gallery_dir}' not found – starting with empty DB.")
            return 0

        # No model → load entirely from cache, then hydrate IDs from manifest
        if face_app is None:
            count = self._load_cache()
            self._assign_ids_from_manifest(self._load_manifest())
            return count

        manifest = self._load_manifest()
        new_manifest = {}

        # --- Step 1: bulk-load ALL cached identities at once (fast) -----
        self._load_cache()
        self._assign_ids_from_manifest(manifest)
        loaded_from_cache = len(self._identities)
        if loaded_from_cache:
            logger.info(f"Cache: {loaded_from_cache} identities loaded instantly.")

        # --- Step 2: scan gallery folders, process new/changed ones -----
        gallery_names = set()
        loaded = 0
        merged = 0

        for person_dir in sorted(gallery.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            gallery_names.add(name)

            # Fast pre-check: compare folder mtime stored in manifest
            folder_mtime = self._folder_mtime(person_dir)
            cached_entry = manifest.get(name, {})
            if cached_entry.get("__folder_mtime__") == folder_mtime:
                # Folder unchanged since last build — copy entry to new manifest and skip
                new_manifest[name] = cached_entry
                continue

            # Folder is new or modified — collect image files and do full check
            image_files = [
                p for p in person_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if not image_files:
                logger.warning(f"  '{name}': no images found, skipping.")
                continue

            current_sig = self._folder_signature(image_files)
            current_sig["__folder_mtime__"] = folder_mtime
            # Preserve existing ID or assign a new one for this identity
            existing_id = manifest.get(name, {}).get("__id__") or str(uuid.uuid4())
            current_sig["__id__"] = existing_id
            new_manifest[name] = current_sig

            # File-level check: if per-file signatures also match, skip
            comparable = {k: v for k, v in cached_entry.items() if k != "__folder_mtime__"}
            comparable_new = {k: v for k, v in current_sig.items() if k != "__folder_mtime__"}
            if comparable == comparable_new:
                continue

            # New or changed → re-extract
            action = "changed" if name in manifest else "new"
            embeddings = self._extract_from_images(image_files, face_app, name)
            if not embeddings:
                logger.warning(f"  '{name}' ({action}): no faces detected, skipping.")
                new_manifest.pop(name, None)
                continue

            new_proto = self._normalize(np.array(embeddings).mean(axis=0))
            duplicate_of = self._find_duplicate(new_proto, dedup_threshold)

            if duplicate_of and duplicate_of != name:
                self.update_identity(duplicate_of, embeddings, mode="append")
                merged += 1
                new_manifest[name] = current_sig
                logger.info(
                    f"  '{name}' ({action}): merged into '{duplicate_of}' "
                    f"(similarity ≥ {dedup_threshold:.2f})."
                )
            else:
                self._store_identity(name, np.array(embeddings), identity_id=existing_id)
                loaded += 1
                logger.info(
                    f"  '{name}' ({action}): {len(embeddings)} embeddings "
                    f"from {len(image_files)} image(s)."
                )

        # --- Step 3: remove identities whose folders were deleted --------
        removed = 0
        stale = set(manifest.keys()) - gallery_names
        for name in stale:
            if name in self._identities:
                self.delete_identity(name)
                removed += 1
                logger.info(f"  '{name}': removed (folder deleted from gallery).")

        # --- Summary -----------------------------------------------------
        total = len(self._identities)
        changed = loaded + merged + removed
        if merged:
            logger.info(f"Deduplication: {merged} folder(s) merged into existing identities.")
        if removed:
            logger.info(f"Removed {removed} stale identit(ies) not in gallery.")
        logger.info(f"Gallery ready: {total} unique identit(ies) in memory.")

        # Only write to disk if something actually changed
        if changed > 0:
            self._save_cache()
            self._save_manifest(new_manifest)
        else:
            logger.info("No gallery changes detected — cache is up to date.")

        return total

    def _extract_from_images(self, image_files: list, face_app, label: str = "") -> list:
        import cv2
        embeddings = []
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.debug(f"    Cannot read: {img_path.name}")
                continue
            faces = face_app.get(img)
            for face in faces:
                if face.embedding is None:
                    continue
                emb = face.embedding
                # Quality gate: skip degenerate embeddings
                if np.linalg.norm(emb) < 1e-3:
                    continue
                embeddings.append(self._normalize(emb))
        return embeddings

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, embedding: np.ndarray, top_k: int = 5) -> tuple:
        """
        Two-stage match:
          Stage 1 – prototype cosine similarity (fast, filters top-3 candidates).
          Stage 2 – top-k mean vote across individual embeddings for final score.

        Returns (name, identity_id, score) where score ∈ [0, 1].
        """
        emb = self._normalize(embedding)

        with self._lock:
            if not self._identities:
                return "Unknown", "", 0.0

            names = list(self._identities.keys())

            # Stage 1: prototype match
            proto_scores = {
                n: float(np.dot(emb, self._identities[n]["prototype"]))
                for n in names
            }
            top3 = sorted(proto_scores, key=proto_scores.get, reverse=True)[:3]

            # Stage 2: top-k vote among shortlisted candidates
            vote_scores = {}
            for n in top3:
                stored = self._identities[n]["embeddings"]  # (N, D)
                sims = stored @ emb  # dot product = cosine similarity (already normalized)
                k = min(top_k, len(sims))
                top_sims = np.partition(sims, -k)[-k:]
                vote_scores[n] = float(top_sims.mean())

            best = max(vote_scores, key=vote_scores.get)
            return best, self._identities[best]["id"], vote_scores[best]

    # ------------------------------------------------------------------
    # Mutation methods (used by HTTP API)
    # ------------------------------------------------------------------

    def register_identity(self, name: str, embeddings: list, identity_id: str = None) -> bool:
        """Register new identity. Returns False if name already exists."""
        with self._lock:
            if name in self._identities:
                return False
        embs = np.array([self._normalize(np.asarray(e)) for e in embeddings])
        self._store_identity(name, embs, identity_id=identity_id or str(uuid.uuid4()))
        self._save_single(name)
        logger.info(f"Registered new identity: '{name}' ({len(embs)} embeddings)")
        return True

    def update_identity(self, name: str, embeddings: list, mode: str = "append") -> bool:
        """Append or replace embeddings for an identity (creates if not exists)."""
        new_embs = np.array([self._normalize(np.asarray(e)) for e in embeddings])
        with self._lock:
            if name in self._identities and mode == "append":
                existing = self._identities[name]["embeddings"]
                combined = np.vstack([existing, new_embs])
                self._store_identity(name, combined)
            else:
                self._store_identity(name, new_embs)
        self._save_single(name)
        logger.info(f"Updated identity: '{name}' mode={mode} (+{len(new_embs)} embeddings)")
        return True

    def delete_identity(self, name: str) -> bool:
        with self._lock:
            if name not in self._identities:
                return False
            del self._identities[name]
        self._delete_from_cache(name)
        logger.info(f"Deleted identity: '{name}'")
        return True

    def list_identities(self) -> list:
        with self._lock:
            return [
                {"id": d["id"], "name": n, "embedding_count": d["count"]}
                for n, d in self._identities.items()
            ]

    # ------------------------------------------------------------------
    # Cache persistence  (single .npz — fast bulk load ~1.5s for 1300 identities)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _save_cache(self):
        if not self._cache_path:
            return
        try:
            data = {}
            with self._lock:
                for name, d in self._identities.items():
                    safe = self._safe_name(name)
                    data[f"{safe}__embs"]  = d["embeddings"]
                    data[f"{safe}__proto"] = d["prototype"]
            np.savez(self._cache_path, **data)
            logger.debug(f"Cache saved → {self._cache_path}")
        except Exception as exc:
            logger.warning(f"Cache save failed: {exc}")

    def _save_single(self, name: str):
        """Convenience — just rewrite the full cache (npz is small enough)."""
        self._save_cache()

    def _delete_from_cache(self, name: str):
        """Remove one identity then rewrite the cache."""
        self._save_cache()

    def _load_cache(self) -> int:
        if not self._cache_path or not Path(self._cache_path).exists():
            return 0
        try:
            npz = np.load(self._cache_path)
            keys = npz.files
            names = {k[:-6] for k in keys if k.endswith("__embs")}
            for name in names:
                embs  = npz[f"{name}__embs"]
                proto_key = f"{name}__proto"
                proto = npz[proto_key] if proto_key in keys else self._normalize(embs.mean(axis=0))
                with self._lock:
                    self._identities[name] = {
                        "embeddings": embs,
                        "prototype":  proto,
                        "count":      len(embs),
                    }
            logger.info(f"Cache loaded: {len(names)} identities from '{self._cache_path}'")
            return len(names)
        except Exception as exc:
            logger.warning(f"Cache load failed: {exc}")
            return 0

