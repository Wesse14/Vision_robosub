from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
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
DEFAULT_EXTENDED_MASK_TEMPLATE_DIR = Path("data/aruco_mask_extend")
MASK_MATCH_SIZE = 160
GRID_SIZE = 7
MASK_MATCH_ACCEPT_THRESHOLD = 0.75
GRID_MATCH_ACCEPT_THRESHOLD = 0.90
STRONG_MASK_MATCH_THRESHOLD = 0.85
STRONG_GRID_MATCH_THRESHOLD = 0.95


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
    mask_match_source: str
    mask_match_image: np.ndarray
    grid_match_id: int | None
    grid_match_score: float
    grid_match_rotation: int
    grid_match_source: str
    grid_image: np.ndarray
    grid_match_image: np.ndarray
    navigation_signal: str
    navigation_marker_id: int | None
    navigation_reason: str


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
    template_source: str


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


@lru_cache(maxsize=1)
def _detector_backend() -> tuple[Any, Any | None, Any | None, Any | None]:
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
        return aruco, aruco.ArucoDetector(dictionary, parameters), None, None

    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.04
    parameters.errorCorrectionRate = 0.25
    parameters.minCornerDistanceRate = 0.03
    return aruco, None, dictionary, parameters


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


def _sharpened_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    equalized = clahe.apply(gray)
    return cv2.addWeighted(equalized, 1.8, cv2.GaussianBlur(equalized, (0, 0), 1.2), -0.8, 0)


def _cleaned_otsu(image: np.ndarray) -> np.ndarray:
    sharpened = _sharpened_gray(image)
    return _cleaned_otsu_from_sharpened(sharpened)


def _cleaned_otsu_from_sharpened(sharpened: np.ndarray) -> np.ndarray:
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)


def _preprocess_variants(image: np.ndarray) -> list[PreprocessVariant]:
    sharpened = _sharpened_gray(image)
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        5,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned_otsu = _cleaned_otsu_from_sharpened(sharpened)
    cleaned_adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    black_white_strong = cv2.medianBlur(cleaned_adaptive, 3)

    base_variants = [
        ("cleaned_otsu", cleaned_otsu),
        ("otsu", otsu),
        ("cleaned_adaptive", cleaned_adaptive),
        ("adaptive_gaussian", adaptive),
        ("black_white_strong", black_white_strong),
    ]

    variants: list[PreprocessVariant] = []
    for name, variant in base_variants:
        variants.append(PreprocessVariant(name, variant))
        variants.append(_resize_variant(f"{name}_2x", variant, 2.0))
    return variants


def _detect_on_image(image: np.ndarray) -> tuple[Any, Any]:
    aruco, detector, dictionary, parameters = _detector_backend()
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(image)
    else:
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


@lru_cache(maxsize=8)
def _load_mask_templates(template_dir: Path = DEFAULT_MASK_TEMPLATE_DIR) -> tuple[MaskTemplate, ...]:
    if not template_dir.exists():
        logger.warning("ArUco mask template directory not found: %s", template_dir)
        return ()

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
    return tuple(templates)


@lru_cache(maxsize=8)
def _load_grid_templates(template_dir: Path = DEFAULT_MASK_TEMPLATE_DIR) -> tuple[MaskTemplate, ...]:
    return tuple(
        MaskTemplate(marker_id=template.marker_id, image=_grid_from_binary(template.image))
        for template in _load_mask_templates(template_dir)
    )


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
    template_source = str(template_dir)
    if not templates or not candidates:
        return MaskMatch(None, 0.0, 0, _binary_normalize(high_contrast_image), None, template_source)

    best = MaskMatch(None, -1.0, 0, candidates[0], None, template_source)
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
                        template_source=template_source,
                    )
    return best


def match_aruco_grid(
    high_contrast_image: np.ndarray,
    template_dir: Path = DEFAULT_MASK_TEMPLATE_DIR,
) -> MaskMatch:
    template_grids = _load_grid_templates(template_dir)
    template_source = str(template_dir)
    candidates = _grid_candidates(high_contrast_image)
    if not template_grids or not candidates:
        fallback = _grid_from_binary(high_contrast_image)
        return MaskMatch(None, 0.0, 0, fallback, None, template_source)

    best = MaskMatch(None, -1.0, 0, candidates[0], None, template_source)
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
                        template_source=template_source,
                    )
    return best


def _match_with_extended_fallback(
    high_contrast_image: np.ndarray,
    matcher: Any,
) -> MaskMatch:
    primary = matcher(high_contrast_image, DEFAULT_MASK_TEMPLATE_DIR)
    if primary.marker_id is not None and primary.score >= MASK_MATCH_ACCEPT_THRESHOLD:
        return primary

    extended = matcher(high_contrast_image, DEFAULT_EXTENDED_MASK_TEMPLATE_DIR)
    if extended.marker_id is not None and extended.score > primary.score:
        return extended
    return primary


def marker_navigation_signal(
    detected_ids: tuple[int, ...],
    high_contrast_ids: tuple[int, ...],
    mask_match: MaskMatch,
    grid_match: MaskMatch,
) -> tuple[str, int | None, str]:
    if (
        mask_match.marker_id is not None
        and mask_match.marker_id == grid_match.marker_id
        and mask_match.score >= STRONG_MASK_MATCH_THRESHOLD
        and grid_match.score >= STRONG_GRID_MATCH_THRESHOLD
    ):
        marker_id = mask_match.marker_id
        if marker_id in detected_ids or marker_id in high_contrast_ids:
            return "NEXT", marker_id, "strong mask/grid agreement with aruco support"
        return "NEXT", marker_id, "strong mask and grid agreement"

    return "TRY_AGAIN_MOVE_CLOSER", None, "mask and grid ids do not strongly agree"


def draw_navigation_signal(image: np.ndarray, signal: str, marker_id: int | None, reason: str) -> np.ndarray:
    canvas = image.copy()
    ok = signal == "NEXT"
    color = (0, 180, 0) if ok else (0, 0, 255)
    label = f"{signal}: ID {marker_id}" if marker_id is not None else signal
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 74), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        reason[:80],
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


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
        match.template_source,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0) if match.score >= MASK_MATCH_ACCEPT_THRESHOLD else (0, 180, 255),
        1,
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
    high_contrast_image = _high_contrast_black_white(image)
    mask_match = _match_with_extended_fallback(high_contrast_image, match_aruco_mask)
    grid_match = _match_with_extended_fallback(high_contrast_image, match_aruco_grid)
    navigation_signal, navigation_marker_id, navigation_reason = marker_navigation_signal(
        (),
        (),
        mask_match,
        grid_match,
    )
    mask_match_image = draw_mask_match(mask_match)
    grid_match_image = draw_mask_match(grid_match)
    grid_image = _render_grid(grid_match.candidate_image)

    if navigation_signal == "NEXT":
        annotated = draw_navigation_signal(
            image.copy(),
            navigation_signal,
            navigation_marker_id,
            navigation_reason,
        )
        high_contrast_annotated_image = draw_navigation_signal(
            cv2.cvtColor(high_contrast_image, cv2.COLOR_GRAY2BGR),
            navigation_signal,
            navigation_marker_id,
            navigation_reason,
        )
        return ArucoDetection(
            ids=(),
            corners=(),
            annotated_image=annotated,
            preprocessed_image=high_contrast_image,
            preprocessing="mask_grid",
            confidence="mask_grid",
            high_contrast_ids=(),
            high_contrast_image=high_contrast_image,
            high_contrast_annotated_image=high_contrast_annotated_image,
            mask_match_id=mask_match.marker_id,
            mask_match_score=mask_match.score,
            mask_match_rotation=mask_match.rotation,
            mask_match_source=mask_match.template_source,
            mask_match_image=mask_match_image,
            grid_match_id=grid_match.marker_id,
            grid_match_score=grid_match.score,
            grid_match_rotation=grid_match.rotation,
            grid_match_source=grid_match.template_source,
            grid_image=grid_image,
            grid_match_image=grid_match_image,
            navigation_signal=navigation_signal,
            navigation_marker_id=navigation_marker_id,
            navigation_reason=navigation_reason,
        )

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
        if logger.isEnabledFor(logging.WARNING):
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
        aruco = _aruco_module()
        draw_corners = [detection.corners for detection in accepted]
        draw_ids = np.asarray([[detection.marker_id] for detection in accepted], dtype=np.int32)
        aruco.drawDetectedMarkers(annotated, draw_corners, draw_ids)
        detected_ids = tuple(detection.marker_id for detection in accepted)
        detected_corners = tuple(
            tuple((float(x), float(y)) for x, y in marker.reshape(-1, 2))
            for marker in draw_corners
        )

    high_contrast_ids, high_contrast_image, high_contrast_annotated_image = _high_contrast_retry(image)
    navigation_signal, navigation_marker_id, navigation_reason = marker_navigation_signal(
        detected_ids,
        high_contrast_ids,
        mask_match,
        grid_match,
    )
    annotated = draw_navigation_signal(annotated, navigation_signal, navigation_marker_id, navigation_reason)

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
        mask_match_source=mask_match.template_source,
        mask_match_image=mask_match_image,
        grid_match_id=grid_match.marker_id,
        grid_match_score=grid_match.score,
        grid_match_rotation=grid_match.rotation,
        grid_match_source=grid_match.template_source,
        grid_image=grid_image,
        grid_match_image=grid_match_image,
        navigation_signal=navigation_signal,
        navigation_marker_id=navigation_marker_id,
        navigation_reason=navigation_reason,
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

    def _debug_cleaned_otsu_2x_path(self) -> Path:
        return self.debug_dir / "aruco_preprocess_cleaned_otsu_2x.png"

    def _write_debug_images(self, image: np.ndarray) -> None:
        if not self.debug:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        cleaned_otsu_2x = _resize_variant("cleaned_otsu_2x", _cleaned_otsu(image), 2.0)
        if not cv2.imwrite(str(self._debug_cleaned_otsu_2x_path()), cleaned_otsu_2x.image):
            logger.warning("Failed to write ArUco cleaned Otsu 2x image: %s", self._debug_cleaned_otsu_2x_path())

    async def process(
        self,
        message: Message[VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[ArucoDetection]:
        payload = message.payload
        image = payload.image if isinstance(payload, VideoFrame) else payload
        detection = detect_original_aruco_markers(image)
        self._write_debug_images(image)

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
        metadata["aruco_mask_match_source"] = detection.mask_match_source
        metadata["aruco_grid_match_id"] = detection.grid_match_id
        metadata["aruco_grid_match_score"] = detection.grid_match_score
        metadata["aruco_grid_match_rotation"] = detection.grid_match_rotation
        metadata["aruco_grid_match_source"] = detection.grid_match_source
        metadata["navigation_signal"] = detection.navigation_signal
        metadata["navigation_marker_id"] = detection.navigation_marker_id
        metadata["navigation_reason"] = detection.navigation_reason
        if isinstance(payload, VideoFrame):
            metadata.setdefault("frame_index", payload.frame_index)
            metadata.setdefault("timestamp_seconds", payload.timestamp_seconds)
            metadata.setdefault("loop_count", payload.loop_count)

        if detection.ids and logger.isEnabledFor(logging.INFO):
            logger.info(
                "Detected ArUco marker id(s) with %s preprocessing (%s): %s",
                detection.preprocessing,
                detection.confidence,
                ", ".join(map(str, detection.ids)),
            )
        elif not detection.ids:
            logger.warning("No ArUco marker detected in rectified cutout.")
        if detection.mask_match_id is not None:
            logger.info(
                "Best ArUco mask match id %s with score %.3f at %s degrees from %s",
                detection.mask_match_id,
                detection.mask_match_score,
                detection.mask_match_rotation,
                detection.mask_match_source,
            )
        if detection.grid_match_id is not None:
            logger.info(
                "Best ArUco grid match id %s with score %.3f at %s degrees from %s",
                detection.grid_match_id,
                detection.grid_match_score,
                detection.grid_match_rotation,
                detection.grid_match_source,
            )
        logger.info(
            "Navigation signal: %s%s (%s)",
            detection.navigation_signal,
            f" id {detection.navigation_marker_id}" if detection.navigation_marker_id is not None else "",
            detection.navigation_reason,
        )

        return RoutedMessage(
            destination=self.output_queue,
            message=Message(detection, metadata=metadata),
        )
