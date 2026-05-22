from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from ..messages import Message, RoutedMessage
from ..video import VideoFrame
from .base import BaseModule, ModuleContext
from .image_enhancer import validate_color_image

EPS = np.finfo(float).eps
REQUIRED_MODEL_KEYS = {
    "query_gmm",
    "non_query_gmm",
    "query_prior",
    "non_query_prior",
}


class GMMColorMaskModule(BaseModule[VideoFrame | np.ndarray]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        model_path: Path | str = Path("data/color_classifier_gmm.joblib"),
        threshold: float = 0.5,
        debug: bool = False,
        debug_dir: Path | str = Path("data/debug"),
    ) -> None:
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")

        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.query_gmm: Any
        self.non_query_gmm: Any
        self.query_prior: float
        self.non_query_prior: float
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"GMM color classifier model not found: {self.model_path}")

        data = joblib.load(self.model_path)
        if not isinstance(data, dict):
            raise ValueError("GMM color classifier model must be a joblib dict.")

        missing_keys = REQUIRED_MODEL_KEYS - data.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"GMM color classifier model is missing key(s): {missing}")

        self.query_gmm = data["query_gmm"]
        self.non_query_gmm = data["non_query_gmm"]
        self.query_prior = float(data["query_prior"])
        self.non_query_prior = float(data["non_query_prior"])

        if self.query_prior < 0.0 or self.non_query_prior < 0.0:
            raise ValueError("GMM color classifier priors must be non-negative.")
        if self.query_prior + self.non_query_prior <= 0.0:
            raise ValueError("GMM color classifier priors must have a positive sum.")

    def _posterior(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_flat = lab.reshape(-1, 3)

        log_prob_query = self.query_gmm.score_samples(lab_flat)
        log_prob_non = self.non_query_gmm.score_samples(lab_flat)
        p_lab_given_query = np.exp(log_prob_query)
        p_lab_given_non = np.exp(log_prob_non)
        p_lab = (
            p_lab_given_query * self.query_prior
            + p_lab_given_non * self.non_query_prior
        )
        p_lab = np.maximum(p_lab, EPS)
        posterior = (p_lab_given_query * self.query_prior) / p_lab
        return posterior.reshape(image.shape[:2])

    def _mask(self, image: np.ndarray) -> np.ndarray:
        posterior = self._posterior(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[posterior > self.threshold] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, None, iterations=3)
        mask = cv2.erode(mask, kernel, None, iterations=3)
        mask = cv2.dilate(mask, kernel, None, iterations=3)
        return mask

    @property
    def _debug_mask_path(self) -> Path:
        return self.debug_dir / "gmm_color_mask.png"

    def _write_debug_mask(self, mask: np.ndarray) -> None:
        if not self.debug:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self._debug_mask_path), mask):
            raise RuntimeError(f"Failed to write GMM color mask debug image: {self._debug_mask_path}")

    async def process(
        self,
        message: Message[VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[np.ndarray]:
        payload = message.payload
        image = payload.image if isinstance(payload, VideoFrame) else payload
        validate_color_image(image)

        metadata: dict[str, Any] = dict(message.metadata)
        if isinstance(payload, VideoFrame):
            metadata.update(
                {
                    "frame_index": payload.frame_index,
                    "timestamp_seconds": payload.timestamp_seconds,
                    "loop_count": payload.loop_count,
                }
            )

        mask = self._mask(image)
        self._write_debug_mask(mask)
        return RoutedMessage(
            destination=self.output_queue,
            message=Message(payload=mask, metadata=metadata),
        )
