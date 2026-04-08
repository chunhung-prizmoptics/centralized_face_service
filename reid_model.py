"""
reid_model.py – ReID embedding model wrapper using torchreid OSNet.

Returns 256-D L2-normalised float32 embeddings.
Falls back to np.zeros(256) if torchreid is unavailable.
"""

import numpy as np
from loguru import logger

_WARNED = False
_REID_INPUT_H = 256
_REID_INPUT_W = 128

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDModel:
    """
    Wraps torchreid's OSNet-x1.0 for 256-D person ReID embeddings.

    Usage:
        reid = ReIDModel()
        emb = reid.get_embedding(bgr_image)  # np.ndarray shape (256,)
    """

    def __init__(self, gpu_id: int = 0):
        self._model = None
        self._device = None
        self._output_dim: int = 256
        self._load(gpu_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, gpu_id: int):
        global _WARNED
        try:
            import torch
            import torchreid

            device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(device_str)

            model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=1000,
                pretrained=True,
            )
            model.eval()
            model.to(self._device)
            self._model = model

            # Probe output dimension with a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, 3, _REID_INPUT_H, _REID_INPUT_W, device=self._device)
                out = model(dummy)
            self._output_dim = int(out.shape[1])
            logger.info(
                f"ReIDModel: OSNet-x1.0 loaded on {device_str} | output_dim={self._output_dim}"
            )
        except Exception as exc:
            if not _WARNED:
                logger.warning(
                    f"ReIDModel: torchreid unavailable ({exc}). "
                    "ReID embeddings will be zero vectors."
                )
                _WARNED = True
            self._model = None

    def _preprocess(self, img_bgr: np.ndarray) -> "torch.Tensor":
        """BGR ndarray → normalised CHW float32 tensor on device."""
        import cv2
        import torch

        img = cv2.resize(img_bgr, (_REID_INPUT_W, _REID_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD          # HWC float32
        img = img.transpose(2, 0, 1)        # CHW
        tensor = torch.from_numpy(img).unsqueeze(0).to(self._device)  # 1CHW
        return tensor

    @staticmethod
    def _l2_norm(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run a single BGR image through OSNet and return an L2-normalised
        float32 vector of shape (output_dim,).  Returns zeros on failure.
        """
        if self._model is None:
            return np.zeros(self._output_dim, dtype=np.float32)

        try:
            import torch

            tensor = self._preprocess(img_bgr)
            with torch.no_grad():
                feat = self._model(tensor)          # (1, D)
            emb = feat.cpu().numpy()[0].astype(np.float32)
            return self._l2_norm(emb)
        except Exception as exc:
            logger.warning(f"ReIDModel.get_embedding error: {exc}")
            return np.zeros(self._output_dim, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim
