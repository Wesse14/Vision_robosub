from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..messages import Message, RoutedMessage
from ..video import VideoFrame
from .base import BaseModule, ModuleContext

logger = logging.getLogger(__name__)

ALLOWED_MARKER_IDS = frozenset(range(101))


@dataclass(frozen=True, slots=True)
class ArucoDetection:
    ids: tuple[int, ...]
    corners: tuple[tuple[tuple[float, float], ...], ...]
    annotated_image: np.ndarray
    preprocessed_image: np.ndarray
    preprocessing: str
    confidence: str
    high_contrast_ids: tuple[int, ...]
    high_contrast_image: np.ndarray
    high_contrast_annotated_image: np.ndarray


@dataclass(frozen=True, slots=True)
class PreprocessVariant:
    name: str
    image: np.ndarray
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    marker_id: int
    corners: np.ndarray
    variant: PreprocessVariant


def _aruco_module() -> Any:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "OpenCV ArUco support is unavailable. Install opencv-contrib-python "
            "instead of opencv-python, then sync the environment."
        )
    return aruco


def _original_dictionary() -> Any:
    aruco = _aruco_module()
    dictionary_id = getattr(aruco, "DICT_ARUCO_ORIGINAL", None)
    if dictionary_id is None:
        raise RuntimeError("This OpenCV build does not expose DICT_ARUCO_ORIGINAL.")
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


def _resize_variant(name: str, image: np.ndarray, scale: float) -> PreprocessVariant:
    if scale == 1.0:
        return PreprocessVariant(name, image, scale)
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    return PreprocessVariant(name, resized, scale)


def _preprocess_variants(image: np.ndarray) -> list[PreprocessVariant]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    sharpened = cv2.addWeighted(equalized, 1.8, cv2.GaussianBlur(equalized, (0, 0), 1.2), -0.8, 0)
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        5,
    )
    adaptive_mean = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35,
        3,
    )
    _, sharpened_otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned_otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cleaned_adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    opened_adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    black_white_strong = cv2.medianBlur(cleaned_adaptive, 3)

    base_variants = [
        ("cleaned_otsu", cleaned_otsu),
        ("black_white_strong", black_white_strong),
        ("otsu", otsu),
        ("cleaned_adaptive", cleaned_adaptive),
        ("adaptive_gaussian", adaptive),
        ("adaptive_mean", adaptive_mean),
        ("sharpened_otsu", sharpened_otsu),
        ("opened_adaptive", opened_adaptive),
        ("sharpened", sharpened),
        ("equalized", equalized),
        ("gray", gray),
    ]

    variants: list[PreprocessVariant] = []
    for name, variant in base_variants:
        variants.append(PreprocessVariant(name, variant))
        if name not in {"gray", "equalized"}:
            variants.append(_resize_variant(f"{name}_2x", variant, 2.0))
    return variants


def _detect_on_image(image: np.ndarray) -> tuple[Any, Any]:
    aruco = _aruco_module()
    dictionary = _original_dictionary()

    if hasattr(aruco, "ArucoDetector"):
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 53
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.minMarkerPerimeterRate = 0.02
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.polygonalApproxAccuracyRate = 0.04
        parameters.errorCorrectionRate = 0.25
        parameters.minCornerDistanceRate = 0.03
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 53
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.minMarkerPerimeterRate = 0.02
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.polygonalApproxAccuracyRate = 0.04
        parameters.errorCorrectionRate = 0.25
        parameters.minCornerDistanceRate = 0.03
        corners, ids, _ = aruco.detectMarkers(image, dictionary, parameters=parameters)

    return corners, ids


def _scaled_to_input_corners(corners: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return corners.astype(np.float32)
    return (corners.astype(np.float32) / scale).astype(np.float32)


def _corner_area(corners: np.ndarray) -> float:
    return float(abs(cv2.contourArea(corners.reshape(-1, 2).astype(np.float32))))


def _high_contrast_black_white(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    boosted = clahe.apply(gray)
    sharpened = cv2.addWeighted(boosted, 2.0, cv2.GaussianBlur(boosted, (0, 0), 1.0), -1.0, 0)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.medianBlur(binary, 3)


def _detect_allowed_candidates(variant: PreprocessVariant) -> list[DetectionCandidate]:
    corners, ids = _detect_on_image(variant.image)
    if ids is None:
        return []

    detections: list[DetectionCandidate] = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in ALLOWED_MARKER_IDS:
            continue
        detections.append(
            DetectionCandidate(
                marker_id=marker_id,
                corners=_scaled_to_input_corners(marker_corners, variant.scale),
                variant=variant,
            )
        )
    return detections


def _high_contrast_retry(annotated_image: np.ndarray) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    binary = _high_contrast_black_white(annotated_image)
    variants = [
        PreprocessVariant("high_contrast", binary),
        _resize_variant("high_contrast_2x", binary, 2.0),
    ]
    detections: list[DetectionCandidate] = []
    for variant in variants:
        detections.extend(_detect_allowed_candidates(variant))

    retry_annotated = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if not detections:
        cv2.putText(
            retry_annotated,
            "No ArUco marker detected",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return (), binary, retry_annotated

    aruco = _aruco_module()
    detections.sort(key=lambda detection: (-_corner_area(detection.corners), detection.variant.scale))
    best = detections[0]
    draw_corners = [best.corners]
    draw_ids = np.asarray([[best.marker_id]], dtype=np.int32)
    aruco.drawDetectedMarkers(retry_annotated, draw_corners, draw_ids)
    return (best.marker_id,), binary, retry_annotated


def detect_original_aruco_markers(image: np.ndarray, min_consensus: int = 2) -> ArucoDetection:
    aruco = _aruco_module()
    variants = _preprocess_variants(image)
    best_variant = variants[0]
    detections: list[DetectionCandidate] = []

    for variant in variants:
        corners, ids = _detect_on_image(variant.image)
        if ids is not None and len(ids) > 0:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                if marker_id not in ALLOWED_MARKER_IDS:
                    continue
                detections.append(
                    DetectionCandidate(
                        marker_id=marker_id,
                        corners=_scaled_to_input_corners(marker_corners, variant.scale),
                        variant=variant,
                    )
                )

    id_counts: dict[int, int] = {}
    for detection in detections:
        id_counts[detection.marker_id] = id_counts.get(detection.marker_id, 0) + 1

    accepted = [
        detection
        for detection in detections
        if id_counts[detection.marker_id] >= min_consensus
    ]
    confidence = "none"
    if accepted:
        accepted.sort(
            key=lambda detection: (
                -id_counts[detection.marker_id],
                -_corner_area(detection.corners),
                detection.variant.scale,
            )
        )
        best_detection = accepted[0]
        best_variant = best_detection.variant
        accepted = [best_detection]
        confidence = "consensus"
    elif detections:
        detections.sort(key=lambda detection: detection.variant.scale)
        best_variant = detections[0].variant
        logger.warning(
            "Ignoring low-confidence ArUco id(s) without preprocessing consensus: %s",
            ", ".join(str(detection.marker_id) for detection in detections),
        )

    annotated = image.copy()
    detected_ids: tuple[int, ...]
    detected_corners: tuple[tuple[tuple[float, float], ...], ...]

    if not accepted:
        detected_ids = ()
        detected_corners = ()
        cv2.putText(
            annotated,
            "No ArUco marker detected",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        draw_corners = [detection.corners for detection in accepted]
        draw_ids = np.asarray([[detection.marker_id] for detection in accepted], dtype=np.int32)
        aruco.drawDetectedMarkers(annotated, draw_corners, draw_ids)
        detected_ids = tuple(detection.marker_id for detection in accepted)
        detected_corners = tuple(
            tuple((float(x), float(y)) for x, y in marker.reshape(-1, 2))
            for marker in draw_corners
        )

    high_contrast_ids, high_contrast_image, high_contrast_annotated_image = _high_contrast_retry(annotated)

    return ArucoDetection(
        ids=detected_ids,
        corners=detected_corners,
        annotated_image=annotated,
        preprocessed_image=best_variant.image,
        preprocessing=best_variant.name,
        confidence=confidence,
        high_contrast_ids=high_contrast_ids,
        high_contrast_image=high_contrast_image,
        high_contrast_annotated_image=high_contrast_annotated_image,
    )


class ArucoDetectionModule(BaseModule[VideoFrame | np.ndarray]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        debug: bool = False,
        debug_dir: Path | str = Path("data/debug"),
    ) -> None:
        super().__init__(name, input_queue)
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        self.output_queue = output_queue
        self.debug = debug
        self.debug_dir = Path(debug_dir)

    def _debug_output_path(self) -> Path:
        return self.debug_dir / "aruco_detected.png"

    def _debug_preprocessed_path(self) -> Path:
        return self.debug_dir / "aruco_preprocessed.png"

    def _debug_high_contrast_path(self) -> Path:
        return self.debug_dir / "aruco_high_contrast.png"

    def _debug_high_contrast_retry_path(self) -> Path:
        return self.debug_dir / "aruco_high_contrast_retry.png"

    def _write_debug_images(self, image: np.ndarray, detection: ArucoDetection) -> None:
        if not self.debug:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self._debug_output_path()), detection.annotated_image):
            logger.warning("Failed to write ArUco debug image: %s", self._debug_output_path())
        if not cv2.imwrite(str(self._debug_preprocessed_path()), detection.preprocessed_image):
            logger.warning(
                "Failed to write ArUco preprocessed image: %s",
                self._debug_preprocessed_path(),
            )
        if not cv2.imwrite(str(self._debug_high_contrast_path()), detection.high_contrast_image):
            logger.warning("Failed to write ArUco high contrast image: %s", self._debug_high_contrast_path())
        if not cv2.imwrite(str(self._debug_high_contrast_retry_path()), detection.high_contrast_annotated_image):
            logger.warning(
                "Failed to write ArUco high contrast retry image: %s",
                self._debug_high_contrast_retry_path(),
            )
        for variant in _preprocess_variants(image):
            variant_path = self.debug_dir / f"aruco_preprocess_{variant.name}.png"
            if not cv2.imwrite(str(variant_path), variant.image):
                logger.warning("Failed to write ArUco preprocess variant: %s", variant_path)

    async def process(
        self,
        message: Message[VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[ArucoDetection]:
        payload = message.payload
        image = payload.image if isinstance(payload, VideoFrame) else payload
        detection = detect_original_aruco_markers(image)
        self._write_debug_images(image, detection)

        metadata: dict[str, Any] = dict(message.metadata)
        metadata["aruco_ids"] = detection.ids
        metadata["aruco_count"] = len(detection.ids)
        metadata["aruco_preprocessing"] = detection.preprocessing
        metadata["aruco_confidence"] = detection.confidence
        metadata["aruco_high_contrast_ids"] = detection.high_contrast_ids
        metadata["aruco_high_contrast_count"] = len(detection.high_contrast_ids)
        if isinstance(payload, VideoFrame):
            metadata.setdefault("frame_index", payload.frame_index)
            metadata.setdefault("timestamp_seconds", payload.timestamp_seconds)
            metadata.setdefault("loop_count", payload.loop_count)

        if detection.ids:
            logger.info(
                "Detected ArUco marker id(s) with %s preprocessing (%s): %s",
                detection.preprocessing,
                detection.confidence,
                ", ".join(map(str, detection.ids)),
            )
        else:
            logger.warning("No ArUco marker detected in rectified cutout.")

        return RoutedMessage(
            destination=self.output_queue,
            message=Message(detection, metadata=metadata),
        )
