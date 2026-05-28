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
DEFAULT_MASK_TEMPLATE_DIR = Path("data/aruco_mask")
MASK_MATCH_SIZE = 160
GRID_SIZE = 7


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
    mask_match_id: int | None
    mask_match_score: float
    mask_match_rotation: int
    mask_match_image: np.ndarray
    grid_match_id: int | None
    grid_match_score: float
    grid_match_rotation: int
    grid_image: np.ndarray
    grid_match_image: np.ndarray


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


@dataclass(frozen=True, slots=True)
class MaskTemplate:
    marker_id: int
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class MaskMatch:
    marker_id: int | None
    score: float
    rotation: int
    candidate_image: np.ndarray
    template_image: np.ndarray | None


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


def _binary_normalize(image: np.ndarray, size: int = MASK_MATCH_SIZE) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def _load_mask_templates(template_dir: Path = DEFAULT_MASK_TEMPLATE_DIR) -> list[MaskTemplate]:
    if not template_dir.exists():
        logger.warning("ArUco mask template directory not found: %s", template_dir)
        return []

    templates: list[MaskTemplate] = []
    for path in sorted(template_dir.glob("*.png")):
        try:
            marker_id = int(path.stem)
        except ValueError:
            continue
        if marker_id not in ALLOWED_MARKER_IDS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Skipping unreadable ArUco mask template: %s", path)
            continue
        templates.append(MaskTemplate(marker_id=marker_id, image=_binary_normalize(image)))
    return templates


def _candidate_crops(binary: np.ndarray) -> list[np.ndarray]:
    height, width = binary.shape[:2]
    side = min(height, width)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    square = binary[y0 : y0 + side, x0 : x0 + side]
    crops = [square]
    for margin_fraction in (0.04, 0.08, 0.12, 0.16):
        margin = int(side * margin_fraction)
        if margin * 2 >= side:
            continue
        crops.append(square[margin : side - margin, margin : side - margin])
    return [_binary_normalize(crop) for crop in crops]


def _grid_from_binary(binary: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    normalized = _binary_normalize(binary, size=grid_size * 24)
    cell_size = normalized.shape[0] // grid_size
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * cell_size
            x0 = col * cell_size
            cell = normalized[y0 : y0 + cell_size, x0 : x0 + cell_size]
            inner_margin = max(1, cell_size // 6)
            inner = cell[inner_margin:-inner_margin, inner_margin:-inner_margin]
            if inner.size == 0:
                inner = cell
            grid[row, col] = 255 if float(np.mean(inner)) >= 127.5 else 0

    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


def _render_grid(grid: np.ndarray, cell_size: int = 28) -> np.ndarray:
    image = cv2.resize(
        grid,
        (grid.shape[1] * cell_size, grid.shape[0] * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for idx in range(grid.shape[0] + 1):
        pos = idx * cell_size
        cv2.line(canvas, (0, pos), (canvas.shape[1], pos), (0, 180, 255), 1)
        cv2.line(canvas, (pos, 0), (pos, canvas.shape[0]), (0, 180, 255), 1)
    return canvas


def _grid_candidates(binary: np.ndarray) -> list[np.ndarray]:
    return [_grid_from_binary(crop) for crop in _candidate_crops(binary)]


def _rotated(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return image
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation}")


def match_aruco_mask(
    high_contrast_image: np.ndarray,
    template_dir: Path = DEFAULT_MASK_TEMPLATE_DIR,
) -> MaskMatch:
    templates = _load_mask_templates(template_dir)
    candidates = _candidate_crops(high_contrast_image)
    if not templates or not candidates:
        return MaskMatch(None, 0.0, 0, _binary_normalize(high_contrast_image), None)

    best = MaskMatch(None, -1.0, 0, candidates[0], None)
    for candidate in candidates:
        for template in templates:
            for rotation in (0, 90, 180, 270):
                rotated_template = _rotated(template.image, rotation)
                mismatch = float(np.mean(candidate != rotated_template))
                inverse_mismatch = float(np.mean(candidate != (255 - rotated_template)))
                score = 1.0 - min(mismatch, inverse_mismatch)
                if score > best.score:
                    best = MaskMatch(
                        marker_id=template.marker_id,
                        score=score,
                        rotation=rotation,
                        candidate_image=candidate,
                        template_image=rotated_template,
                    )
    return best


def match_aruco_grid(
    high_contrast_image: np.ndarray,
    template_dir: Path = DEFAULT_MASK_TEMPLATE_DIR,
) -> MaskMatch:
    templates = _load_mask_templates(template_dir)
    template_grids = [
        MaskTemplate(marker_id=template.marker_id, image=_grid_from_binary(template.image))
        for template in templates
    ]
    candidates = _grid_candidates(high_contrast_image)
    if not template_grids or not candidates:
        fallback = _grid_from_binary(high_contrast_image)
        return MaskMatch(None, 0.0, 0, fallback, None)

    best = MaskMatch(None, -1.0, 0, candidates[0], None)
    for candidate in candidates:
        for template in template_grids:
            for rotation in (0, 90, 180, 270):
                rotated_template = _rotated(template.image, rotation)
                mismatch = float(np.mean(candidate != rotated_template))
                inverse_mismatch = float(np.mean(candidate != (255 - rotated_template)))
                score = 1.0 - min(mismatch, inverse_mismatch)
                if score > best.score:
                    best = MaskMatch(
                        marker_id=template.marker_id,
                        score=score,
                        rotation=rotation,
                        candidate_image=candidate,
                        template_image=rotated_template,
                    )
    return best


def draw_mask_match(match: MaskMatch) -> np.ndarray:
    candidate_bgr = (
        _render_grid(match.candidate_image)
        if match.candidate_image.shape == (GRID_SIZE, GRID_SIZE)
        else cv2.cvtColor(match.candidate_image, cv2.COLOR_GRAY2BGR)
    )
    if match.template_image is None or match.marker_id is None:
        cv2.putText(
            candidate_bgr,
            "No mask match",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return candidate_bgr

    template_bgr = (
        _render_grid(match.template_image)
        if match.template_image.shape == (GRID_SIZE, GRID_SIZE)
        else cv2.cvtColor(match.template_image, cv2.COLOR_GRAY2BGR)
    )
    comparison = cv2.hconcat([candidate_bgr, template_bgr])
    cv2.putText(
        comparison,
        f"mask id {match.marker_id} score {match.score:.3f} rot {match.rotation}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if match.score >= 0.75 else (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        "candidate",
        (10, comparison.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        "template",
        (candidate_bgr.shape[1] + 10, comparison.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return comparison


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

    high_contrast_ids, high_contrast_image, high_contrast_annotated_image = _high_contrast_retry(image)
    mask_match = match_aruco_mask(high_contrast_image)
    mask_match_image = draw_mask_match(mask_match)
    grid_match = match_aruco_grid(high_contrast_image)
    grid_match_image = draw_mask_match(grid_match)
    grid_image = _render_grid(grid_match.candidate_image)

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
        mask_match_id=mask_match.marker_id,
        mask_match_score=mask_match.score,
        mask_match_rotation=mask_match.rotation,
        mask_match_image=mask_match_image,
        grid_match_id=grid_match.marker_id,
        grid_match_score=grid_match.score,
        grid_match_rotation=grid_match.rotation,
        grid_image=grid_image,
        grid_match_image=grid_match_image,
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

    def _debug_mask_match_path(self) -> Path:
        return self.debug_dir / "aruco_mask_match.png"

    def _debug_grid_path(self) -> Path:
        return self.debug_dir / "aruco_grid.png"

    def _debug_grid_match_path(self) -> Path:
        return self.debug_dir / "aruco_grid_match.png"

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
        if not cv2.imwrite(str(self._debug_mask_match_path()), detection.mask_match_image):
            logger.warning("Failed to write ArUco mask match image: %s", self._debug_mask_match_path())
        if not cv2.imwrite(str(self._debug_grid_path()), detection.grid_image):
            logger.warning("Failed to write ArUco grid image: %s", self._debug_grid_path())
        if not cv2.imwrite(str(self._debug_grid_match_path()), detection.grid_match_image):
            logger.warning("Failed to write ArUco grid match image: %s", self._debug_grid_match_path())
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
        metadata["aruco_mask_match_id"] = detection.mask_match_id
        metadata["aruco_mask_match_score"] = detection.mask_match_score
        metadata["aruco_mask_match_rotation"] = detection.mask_match_rotation
        metadata["aruco_grid_match_id"] = detection.grid_match_id
        metadata["aruco_grid_match_score"] = detection.grid_match_score
        metadata["aruco_grid_match_rotation"] = detection.grid_match_rotation
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
        if detection.mask_match_id is not None:
            logger.info(
                "Best ArUco mask match id %s with score %.3f at %s degrees",
                detection.mask_match_id,
                detection.mask_match_score,
                detection.mask_match_rotation,
            )
        if detection.grid_match_id is not None:
            logger.info(
                "Best ArUco grid match id %s with score %.3f at %s degrees",
                detection.grid_match_id,
                detection.grid_match_score,
                detection.grid_match_rotation,
            )

        return RoutedMessage(
            destination=self.output_queue,
            message=Message(detection, metadata=metadata),
        )
