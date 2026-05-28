from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..messages import Message, RoutedMessage
from ..video import VideoFrame
from .base import BaseModule, ModuleContext
from .image_enhancer import EnhancementMode, apply_enhancement, validate_color_image

logger = logging.getLogger(__name__)

ALLOWED_ARUCO_IDS = frozenset(range(101))

@dataclass
class EdgeArtifacts:
    gray: np.ndarray
    blur: np.ndarray
    edges_canny: np.ndarray
    grad_mag: np.ndarray
    dist: np.ndarray


def yellow_pipe_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([12, 45, 70], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def suppress_masked_edges(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return gray
    suppressed = gray.copy()
    background = int(np.median(gray[mask == 0])) if np.any(mask == 0) else int(np.median(gray))
    expanded = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    suppressed[expanded > 0] = background
    return suppressed


def detect_edges(
    gray: np.ndarray,
    low_scale: float = 0.66,
    high_scale: float = 1.33,
    ignore_mask: np.ndarray | None = None,
) -> EdgeArtifacts:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    median = float(np.median(blur))
    low = int(max(0, low_scale * median))
    high = int(min(255, high_scale * median))
    if high <= low:
        high = min(255, low + 32)

    edges_canny = cv2.Canny(blur, low, high)
    if ignore_mask is not None and np.any(ignore_mask):
        blocked = cv2.dilate(ignore_mask, np.ones((7, 7), np.uint8), iterations=1)
        edges_canny[blocked > 0] = 0
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    if ignore_mask is not None and np.any(ignore_mask):
        grad_mag[ignore_mask > 0] = 0.0
    edge_binary = edges_canny > 0
    dist = cv2.distanceTransform((~edge_binary).astype(np.uint8), cv2.DIST_L2, 5)

    return EdgeArtifacts(
        gray=gray,
        blur=blur,
        edges_canny=edges_canny,
        grad_mag=grad_mag,
        dist=dist,
    )


def build_edge_variants(image: np.ndarray, preprocess_mode: EnhancementMode = None) -> tuple[np.ndarray, list[EdgeArtifacts]]:
    working_image = apply_enhancement(image, preprocess_mode) if preprocess_mode is not None else image
    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    yellow_mask = yellow_pipe_mask(working_image)
    gray_without_yellow = suppress_masked_edges(gray, yellow_mask)
    variants = [
        detect_edges(gray_without_yellow, ignore_mask=yellow_mask),
        detect_edges(gray_without_yellow, low_scale=0.5, high_scale=1.1, ignore_mask=yellow_mask),
        detect_edges(gray),
        detect_edges(gray, low_scale=0.5, high_scale=1.1),
    ]
    return working_image, variants


def fallback_edge_retry(artifacts: EdgeArtifacts) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(artifacts.edges_canny, cv2.MORPH_CLOSE, kernel)


def aruco_module() -> Any | None:
    return getattr(cv2, "aruco", None)


def original_aruco_dictionary() -> Any | None:
    aruco = aruco_module()
    if aruco is None:
        return None
    dictionary_id = getattr(aruco, "DICT_ARUCO_ORIGINAL", None)
    if dictionary_id is None:
        return None
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


def aruco_preprocess_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    equalized = clahe.apply(gray)
    sharpened = cv2.addWeighted(equalized, 1.8, cv2.GaussianBlur(equalized, (0, 0), 1.2), -0.8, 0)
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
    cleaned_otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cleaned_adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    return [
        ("cleaned_otsu", cleaned_otsu),
        ("otsu", otsu),
        ("cleaned_adaptive", cleaned_adaptive),
        ("adaptive", adaptive),
        ("sharpened", sharpened),
        ("equalized", equalized),
        ("gray", gray),
    ]


def detect_aruco_quad_candidates(image: np.ndarray, min_area: float) -> list[Candidate]:
    aruco = aruco_module()
    dictionary = original_aruco_dictionary()
    if aruco is None or dictionary is None:
        return []

    candidates: list[Candidate] = []
    for name, candidate_image in aruco_preprocess_variants(image):
        if hasattr(aruco, "ArucoDetector"):
            parameters = aruco.DetectorParameters()
            parameters.errorCorrectionRate = 0.25
            parameters.minMarkerPerimeterRate = 0.02
            parameters.maxMarkerPerimeterRate = 4.0
            detector = aruco.ArucoDetector(dictionary, parameters)
            corners, ids, _ = detector.detectMarkers(candidate_image)
        else:
            parameters = aruco.DetectorParameters_create()
            parameters.errorCorrectionRate = 0.25
            parameters.minMarkerPerimeterRate = 0.02
            parameters.maxMarkerPerimeterRate = 4.0
            corners, ids, _ = aruco.detectMarkers(candidate_image, dictionary, parameters=parameters)

        if ids is None:
            continue
        height, width = image.shape[:2]
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in ALLOWED_ARUCO_IDS:
                continue
            quad = order_corners(marker_corners.reshape(4, 2).astype(np.float32))
            if is_valid_quad(quad, width, height, min_area):
                source = f"aruco:{name}:id-{marker_id}"
                candidates.append(Candidate(quad=quad, source=source, variant_idx=0))

    candidates.sort(key=lambda candidate: -polygon_area(candidate.quad))
    return candidates


EPSILONS = [0.01, 0.02, 0.03, 0.05, 0.08]


@dataclass
class Candidate:
    quad: np.ndarray
    source: str
    variant_idx: int
    score: float | None = None


@dataclass
class FitResult:
    quad: np.ndarray
    score: float
    rejected: list[np.ndarray]
    source: str = "unknown"


@dataclass
class LineDebug:
    variant_idx: int
    lines: np.ndarray | None
    family_a: np.ndarray | None
    family_b: np.ndarray | None
    candidates: list[np.ndarray]
    closed_edges_used: bool = False


def order_corners(pts: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    if len({tuple(map(float, p)) for p in ordered}) != 4:
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        start = np.argmin(pts.sum(axis=1))
        ordered = np.roll(pts, -start, axis=0).astype(np.float32)
        if signed_area(ordered) < 0:
            ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return ordered


def signed_area(quad: np.ndarray) -> float:
    pts = np.asarray(quad, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(quad: np.ndarray) -> float:
    return abs(signed_area(quad))


def is_convex_quad(quad: np.ndarray) -> bool:
    quad = np.asarray(quad, dtype=np.float32)
    crosses = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        c = quad[(i + 2) % 4]
        ab = b - a
        bc = c - b
        crosses.append(ab[0] * bc[1] - ab[1] * bc[0])
    crosses = np.asarray(crosses)
    return np.all(crosses > 0) or np.all(crosses < 0)


def edge_lengths(quad: np.ndarray) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32)
    return np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)


def interior_angles_deg(quad: np.ndarray) -> np.ndarray:
    quad = order_corners(quad)
    angles: list[float] = []
    for idx in range(4):
        prev_pt = quad[(idx - 1) % 4]
        pt = quad[idx]
        next_pt = quad[(idx + 1) % 4]
        v0 = prev_pt - pt
        v1 = next_pt - pt
        denom = max(float(np.linalg.norm(v0) * np.linalg.norm(v1)), 1e-6)
        cos_angle = float(np.clip(np.dot(v0, v1) / denom, -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos_angle)))
    return np.asarray(angles, dtype=np.float32)


def quad_shape_penalty(quad: np.ndarray) -> float:
    lengths = edge_lengths(quad)
    if np.min(lengths) < 1e-3:
        return 1e3
    angles = interior_angles_deg(quad)
    side_ratio = float(np.max(lengths) / max(np.min(lengths), 1e-6))
    opposite_ratio_a = float(max(lengths[0], lengths[2]) / max(min(lengths[0], lengths[2]), 1e-6))
    opposite_ratio_b = float(max(lengths[1], lengths[3]) / max(min(lengths[1], lengths[3]), 1e-6))
    angle_penalty = float(np.sum(np.maximum(0.0, 40.0 - angles) + np.maximum(0.0, angles - 140.0))) * 0.35
    side_penalty = max(0.0, side_ratio - 4.0) * 4.0
    opposite_penalty = (max(0.0, opposite_ratio_a - 3.5) + max(0.0, opposite_ratio_b - 3.5)) * 3.0
    return angle_penalty + side_penalty + opposite_penalty


def is_valid_quad(quad: np.ndarray, width: int, height: int, min_area: float) -> bool:
    quad = order_corners(quad)
    if quad.shape != (4, 2):
        return False
    if not np.all(np.isfinite(quad)):
        return False
    if np.any(quad[:, 0] < -1) or np.any(quad[:, 0] > width):
        return False
    if np.any(quad[:, 1] < -1) or np.any(quad[:, 1] > height):
        return False
    if polygon_area(quad) < min_area:
        return False
    if not is_convex_quad(quad):
        return False
    lengths = edge_lengths(quad)
    min_edge = max(8.0, min(width, height) * 0.025)
    if float(np.min(lengths)) < min_edge:
        return False
    if float(np.max(lengths) / max(np.min(lengths), 1e-6)) > 5.0:
        return False
    angles = interior_angles_deg(quad)
    if float(np.min(angles)) < 35.0 or float(np.max(angles)) > 145.0:
        return False
    skinny = polygon_area(quad) / max(float(np.sum(lengths) ** 2), 1.0)
    if skinny < 0.012:
        return False
    return True


def dedupe_candidates(quads: Iterable[np.ndarray], tol: float = 10.0) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for quad in quads:
        quad = order_corners(quad)
        if any(np.mean(np.linalg.norm(quad - other, axis=1)) < tol for other in unique):
            continue
        unique.append(quad)
    return unique


def find_contour_candidates(edges: np.ndarray, width: int, height: int, min_area: float, variant_idx: int) -> list[Candidate]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[np.ndarray] = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        for eps in EPSILONS:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) != 4:
                continue
            approx = approx.reshape(-1, 2).astype(np.float32)
            if cv2.contourArea(approx) < min_area:
                continue
            if not cv2.isContourConvex(approx.astype(np.int32)):
                continue
            if is_valid_quad(approx, width, height, min_area):
                candidates.append(order_corners(approx))

    return [Candidate(quad=quad, source="contour", variant_idx=variant_idx) for quad in dedupe_candidates(candidates)]


def line_to_abc(line: Sequence[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = map(float, line)
    a = y1 - y2
    b = x2 - x1
    norm = math.hypot(a, b)
    if norm < 1e-6:
        return None
    c = x1 * y2 - x2 * y1
    return np.array([a / norm, b / norm, c / norm], dtype=np.float32)


def line_angle_deg(line: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, line)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return angle


def select_angle_families(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    angles = np.array([line_angle_deg(line) for line in lines], dtype=np.float32)
    if len(angles) < 4:
        return None

    bins = np.linspace(0.0, 180.0, 37)
    hist, _ = np.histogram(angles, bins=bins)
    if hist.max() == 0:
        return None

    first_idx = int(np.argmax(hist))
    first_center = (bins[first_idx] + bins[first_idx + 1]) / 2.0

    distances = np.abs(((angles - first_center + 90.0) % 180.0) - 90.0)
    second_mask = distances > 20.0
    if not np.any(second_mask):
        return None

    second_angles = angles[second_mask]
    second_hist, _ = np.histogram(second_angles, bins=bins)
    second_idx = int(np.argmax(second_hist))
    second_center = (bins[second_idx] + bins[second_idx + 1]) / 2.0

    family_a_mask = np.abs(((angles - first_center + 90.0) % 180.0) - 90.0) <= 15.0
    family_b_mask = np.abs(((angles - second_center + 90.0) % 180.0) - 90.0) <= 15.0

    family_a = lines[family_a_mask]
    family_b = lines[family_b_mask]
    if len(family_a) < 2 or len(family_b) < 2:
        return None
    return family_a, family_b


def pair_extreme_lines(lines: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    coeffs = []
    for line in lines:
        abc = line_to_abc(line)
        if abc is not None:
            coeffs.append(abc)
    if len(coeffs) < 2:
        return []

    coeffs = np.asarray(coeffs, dtype=np.float32)
    c_values = coeffs[:, 2]
    order = np.argsort(c_values)
    candidate_pairs = [(coeffs[order[0]], coeffs[order[-1]])]
    if len(order) >= 4:
        candidate_pairs.append((coeffs[order[0]], coeffs[order[-2]]))
        candidate_pairs.append((coeffs[order[1]], coeffs[order[-1]]))
    return candidate_pairs


def intersect_lines(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(float(det)) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)


def dominant_family_quads(
    edges: np.ndarray,
    family_lines: np.ndarray | None,
    width: int,
    height: int,
    min_area: float,
) -> list[np.ndarray]:
    if family_lines is None or len(family_lines) < 2:
        return []

    directions = []
    endpoints = []
    for x1, y1, x2, y2 in family_lines.astype(np.float32):
        direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm
        if direction[1] > 0:
            direction = -direction
        directions.append(direction)
        endpoints.extend([np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)])

    if len(directions) < 2 or len(endpoints) < 4:
        return []

    axis_u = np.mean(np.asarray(directions, dtype=np.float32), axis=0)
    axis_norm = float(np.linalg.norm(axis_u))
    if axis_norm < 1e-6:
        return []
    axis_u = axis_u / axis_norm
    axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)

    endpoints = np.asarray(endpoints, dtype=np.float32)
    endpoint_u = endpoints @ axis_u
    endpoint_v = endpoints @ axis_v

    ys, xs = np.nonzero(edges > 0)
    if len(xs) < 20:
        return []
    edge_points = np.column_stack([xs, ys]).astype(np.float32)
    edge_u = edge_points @ axis_u
    edge_v = edge_points @ axis_v

    quads: list[np.ndarray] = []
    min_dim = float(min(width, height))
    u_bounds = np.percentile(endpoint_u, [0, 100])

    line_infos: list[tuple[float, np.ndarray]] = []
    for line in family_lines.astype(np.float32):
        abc = line_to_abc(line)
        if abc is None:
            continue
        if float(np.dot(abc[:2], axis_v)) < 0.0:
            abc = -abc
        x1, y1, x2, y2 = line
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        line_infos.append((float(midpoint @ axis_v), abc.astype(np.float32)))

    line_infos.sort(key=lambda item: item[0])
    line_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    if len(line_infos) >= 2:
        line_pairs.append((line_infos[0][1], line_infos[-1][1]))
    if len(line_infos) >= 4:
        line_pairs.append((line_infos[0][1], line_infos[-2][1]))
        line_pairs.append((line_infos[1][1], line_infos[-1][1]))

    for side0, side1 in line_pairs:
        side_offsets = sorted([-float(side0[2]), -float(side1[2])])
        side_mask = (edge_v >= side_offsets[0] - 40) & (edge_v <= side_offsets[1] + 40)
        if int(np.count_nonzero(side_mask)) < 20:
            continue
        for low_pct, high_pct in [(2, 98), (5, 95), (0, 100)]:
            u_low, u_high = np.percentile(edge_u[side_mask], [low_pct, high_pct])
            if abs(float(u_high - u_low)) < min_dim * 0.08:
                continue
            cross0 = np.array([axis_u[0], axis_u[1], -u_low], dtype=np.float32)
            cross1 = np.array([axis_u[0], axis_u[1], -u_high], dtype=np.float32)
            pts = [
                intersect_lines(side0, cross0),
                intersect_lines(side1, cross0),
                intersect_lines(side1, cross1),
                intersect_lines(side0, cross1),
            ]
            if any(p is None for p in pts):
                continue
            quad = order_corners(np.asarray(pts, dtype=np.float32))
            if is_valid_quad(quad, width, height, min_area):
                quads.append(quad)

    for low_pct, high_pct in [(5, 95), (0, 100), (10, 90)]:
        v_low, v_high = np.percentile(endpoint_v, [low_pct, high_pct])
        if abs(float(v_high - v_low)) < min_dim * 0.08:
            continue

        mask = (
            (edge_v >= v_low - 30)
            & (edge_v <= v_high + 30)
            & (edge_u >= u_bounds[0] - min_dim * 0.15)
            & (edge_u <= u_bounds[1] + min_dim * 0.15)
        )
        if int(np.count_nonzero(mask)) < 20:
            continue

        u_low, u_high = np.percentile(edge_u[mask], [2, 98])
        if abs(float(u_high - u_low)) < min_dim * 0.08:
            continue

        quad_uv = np.array(
            [
                [u_low, v_low],
                [u_high, v_low],
                [u_high, v_high],
                [u_low, v_high],
            ],
            dtype=np.float32,
        )
        quad = quad_uv[:, 0, None] * axis_u + quad_uv[:, 1, None] * axis_v
        quad = order_corners(quad)
        if is_valid_quad(quad, width, height, min_area):
            quads.append(quad)

    return dedupe_candidates(quads)


def hough_line_debug(
    edges: np.ndarray,
    width: int,
    height: int,
    min_area: float,
    variant_idx: int,
    closed_edges_used: bool = False,
) -> tuple[list[Candidate], LineDebug]:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min(width, height) * 0.1,
        maxLineGap=20,
    )
    if lines is None:
        return [], LineDebug(variant_idx, None, None, None, [], closed_edges_used)

    lines = lines.reshape(-1, 4).astype(np.float32)
    families = select_angle_families(lines)
    if families is None:
        return [], LineDebug(variant_idx, lines, None, None, [], closed_edges_used)

    family_a, family_b = families
    pair_a = pair_extreme_lines(family_a)
    pair_b = pair_extreme_lines(family_b)
    quads: list[np.ndarray] = []

    for a0, a1 in pair_a:
        for b0, b1 in pair_b:
            pts = [
                intersect_lines(a0, b0),
                intersect_lines(a1, b0),
                intersect_lines(a1, b1),
                intersect_lines(a0, b1),
            ]
            if any(p is None for p in pts):
                continue
            quad = order_corners(np.asarray(pts, dtype=np.float32))
            if is_valid_quad(quad, width, height, min_area):
                quads.append(quad)

    source = "hough"
    if not quads:
        dominant = family_a if len(family_a) >= len(family_b) else family_b
        quads.extend(dominant_family_quads(edges, dominant, width, height, min_area))
        source = "hough_dominant"

    quads = dedupe_candidates(quads)
    candidates = [Candidate(quad=quad, source=source, variant_idx=variant_idx) for quad in quads]
    return candidates, LineDebug(variant_idx, lines, family_a, family_b, quads, closed_edges_used)


def find_hough_candidates(edges: np.ndarray, width: int, height: int, min_area: float, variant_idx: int) -> list[Candidate]:
    candidates, _ = hough_line_debug(edges, width, height, min_area, variant_idx)
    return candidates


def sample_edge_points(quad: np.ndarray, n_per_edge: int = 100) -> np.ndarray:
    points = []
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i + 1) % 4]
        ts = np.linspace(0.0, 1.0, n_per_edge, endpoint=True)
        seg = (1.0 - ts)[:, None] * p0 + ts[:, None] * p1
        points.append(seg)
    return np.vstack(points).astype(np.float32)


def bilinear_sample(image: np.ndarray, points: np.ndarray, outside_value: float) -> np.ndarray:
    h, w = image.shape[:2]
    x = points[:, 0]
    y = points[:, 1]

    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    values = np.full(points.shape[0], outside_value, dtype=np.float32)
    if not np.any(valid):
        return values

    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xv - x0
    wy = yv - y0

    sampled = (
        (1 - wx) * (1 - wy) * image[y0, x0]
        + wx * (1 - wy) * image[y0, x1]
        + (1 - wx) * wy * image[y1, x0]
        + wx * wy * image[y1, x1]
    )
    values[valid] = sampled.astype(np.float32)
    return values


def rectification_penalty(quad: np.ndarray) -> float:
    lengths = edge_lengths(quad)
    if np.min(lengths) < 1e-3:
        return 1e3
    ratio = float(np.max(lengths) / np.min(lengths))
    diag = np.linalg.norm(quad[0] - quad[2]) + np.linalg.norm(quad[1] - quad[3])
    area = polygon_area(quad)
    compactness = area / max(diag * diag, 1.0)
    penalty = max(0.0, ratio - 8.0) * 0.5
    penalty += max(0.0, 0.08 - compactness) * 20.0
    penalty += quad_shape_penalty(quad)
    return penalty


def edge_distance_score(quad: np.ndarray, dist: np.ndarray, grad_mag: np.ndarray, n: int = 100) -> float:
    points = sample_edge_points(quad, n_per_edge=n)
    dist_values = bilinear_sample(dist, points, outside_value=999.0)
    grad_values = bilinear_sample(grad_mag, points, outside_value=0.0)
    mean_dist = float(np.mean(dist_values))
    grad_penalty = max(0.0, 18.0 - float(np.mean(grad_values))) * 0.1
    return mean_dist + grad_penalty + rectification_penalty(quad)


def quad_mask_fraction(mask: np.ndarray, quad: np.ndarray, out_size: int = 64) -> float:
    dst_square = np.array(
        [
            [0, 0],
            [out_size - 1, 0],
            [out_size - 1, out_size - 1],
            [0, out_size - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(order_corners(quad).astype(np.float32), dst_square)
    warped = cv2.warpPerspective(mask, matrix, (out_size, out_size))
    return float(np.count_nonzero(warped > 0)) / float(out_size * out_size)


def marker_likeness_score(image: np.ndarray, quad: np.ndarray, out_size: int = 96) -> float:
    cutout = warp_square_cutout(image, quad, out_size)
    gray = cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    border = max(4, out_size // 12)
    border_pixels = np.concatenate(
        [
            binary[:border, :].reshape(-1),
            binary[-border:, :].reshape(-1),
            binary[:, :border].reshape(-1),
            binary[:, -border:].reshape(-1),
        ]
    )
    border_black = 1.0 - (float(np.mean(border_pixels)) / 255.0)

    inner = binary[border:-border, border:-border]
    if inner.size == 0:
        return 0.0
    black_fraction = 1.0 - (float(np.mean(inner)) / 255.0)
    balance = 1.0 - min(abs(black_fraction - 0.5) / 0.5, 1.0)

    vertical_edges = cv2.Sobel(binary, cv2.CV_32F, 1, 0, ksize=3)
    horizontal_edges = cv2.Sobel(binary, cv2.CV_32F, 0, 1, ksize=3)
    grid_energy = (
        float(np.mean(np.abs(vertical_edges)))
        + float(np.mean(np.abs(horizontal_edges)))
    ) / 255.0
    grid_energy = min(grid_energy, 1.0)

    return 0.45 * border_black + 0.30 * balance + 0.25 * grid_energy


def candidate_selection_score(
    candidate: Candidate,
    edge_score: float,
    width: int,
    height: int,
    yellow_mask: np.ndarray | None = None,
    image: np.ndarray | None = None,
) -> float:
    yellow_penalty = 0.0
    if yellow_mask is not None and np.any(yellow_mask):
        yellow_penalty = quad_mask_fraction(yellow_mask, candidate.quad) * 80.0
    marker_bonus = 0.0
    if image is not None:
        marker_bonus = marker_likeness_score(image, candidate.quad) * 45.0

    if candidate.source != "hough_dominant":
        return edge_score + yellow_penalty - marker_bonus

    # A single dominant line family can lock onto internal stripes. Prefer the
    # larger projected sign without forcing equal side lengths in image space.
    image_area = max(float(width * height), 1.0)
    area_ratio = min(polygon_area(candidate.quad) / image_area, 0.30)
    lengths = edge_lengths(candidate.quad)
    side_ratio = float(np.max(lengths) / max(np.min(lengths), 1e-6))
    stripe_penalty = max(0.0, side_ratio - 4.0) * 2.0
    return edge_score - 180.0 * area_ratio + stripe_penalty + yellow_penalty - marker_bonus


def refine_candidate(
    initial_quad: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    min_area: float,
    reg_weight: float = 0.05,
) -> np.ndarray:
    initial_quad = order_corners(initial_quad).astype(np.float32)
    initial_quad[:, 0] = np.clip(initial_quad[:, 0], 0.0, width - 1.0)
    initial_quad[:, 1] = np.clip(initial_quad[:, 1], 0.0, height - 1.0)
    initial_quad = order_corners(initial_quad).astype(np.float32)
    sample_t = np.linspace(0.0, 1.0, 80, endpoint=True)

    def residuals(params: np.ndarray) -> np.ndarray:
        quad = order_corners(params.reshape(4, 2))
        edge_points = []
        for i in range(4):
            p0 = quad[i]
            p1 = quad[(i + 1) % 4]
            edge_points.append((1.0 - sample_t)[:, None] * p0 + sample_t[:, None] * p1)
        edge_points_arr = np.vstack(edge_points).astype(np.float32)
        residual = list(bilinear_sample(dist, edge_points_arr, outside_value=60.0))

        area = polygon_area(quad)
        lengths = edge_lengths(quad)
        angles = interior_angles_deg(quad)
        side_ratio = float(np.max(lengths) / max(np.min(lengths), 1e-6))
        convex_penalty = 50.0 if not is_convex_quad(quad) else 0.0
        area_penalty = math.sqrt(max(min_area - area, 0.0))
        length_penalty = 20.0 if np.min(lengths) < 8.0 else 0.0
        angle_penalty = float(np.sum(np.maximum(0.0, 35.0 - angles) + np.maximum(0.0, angles - 145.0)))
        side_ratio_penalty = max(0.0, side_ratio - 5.0) * 10.0
        residual.extend([convex_penalty] * 12)
        residual.extend([area_penalty] * 12)
        residual.extend([length_penalty] * 8)
        residual.extend([angle_penalty] * 4)
        residual.extend([side_ratio_penalty] * 4)

        residual.extend(((quad - initial_quad).reshape(-1) * reg_weight).tolist())
        return np.asarray(residual, dtype=np.float32)

    lower = np.tile([0.0, 0.0], 4)
    upper = np.tile([width - 1.0, height - 1.0], 4)
    result = least_squares(
        residuals,
        initial_quad.reshape(-1),
        bounds=(lower, upper),
        max_nfev=200,
    )
    refined = order_corners(result.x.reshape(4, 2))
    if not is_valid_quad(refined, width, height, min_area):
        return initial_quad
    return refined.astype(np.float32)


def collect_line_debug(image: np.ndarray, edge_variants: list[EdgeArtifacts]) -> list[LineDebug]:
    height, width = image.shape[:2]
    min_area = max(0.01 * width * height, 400.0)
    debug_items: list[LineDebug] = []
    for idx, artifacts in enumerate(edge_variants):
        _, debug = hough_line_debug(artifacts.edges_canny, width, height, min_area, idx)
        debug_items.append(debug)
        closed_edges = fallback_edge_retry(artifacts)
        _, closed_debug = hough_line_debug(closed_edges, width, height, min_area, idx, closed_edges_used=True)
        debug_items.append(closed_debug)
    return debug_items


def fit_square(image: np.ndarray, edge_variants: list[EdgeArtifacts]) -> FitResult:
    height, width = image.shape[:2]
    min_area = max(0.01 * width * height, 400.0)
    yellow_mask = yellow_pipe_mask(image)
    all_candidates: list[Candidate] = []
    rejected_debug: list[np.ndarray] = []

    aruco_candidates = detect_aruco_quad_candidates(image, min_area)
    if aruco_candidates:
        best_aruco = max(
            aruco_candidates,
            key=lambda candidate: marker_likeness_score(image, candidate.quad),
        )
        score = edge_distance_score(
            best_aruco.quad,
            edge_variants[0].dist,
            edge_variants[0].grad_mag,
        )
        return FitResult(
            quad=best_aruco.quad,
            score=score,
            rejected=[candidate.quad for candidate in aruco_candidates if candidate is not best_aruco],
            source=best_aruco.source,
        )

    for idx, artifacts in enumerate(edge_variants):
        contour_candidates = find_contour_candidates(artifacts.edges_canny, width, height, min_area, idx)
        hough_candidates, _ = hough_line_debug(artifacts.edges_canny, width, height, min_area, idx)
        closed_edges = fallback_edge_retry(artifacts)
        closed_contour_candidates = find_contour_candidates(closed_edges, width, height, min_area, idx)
        closed_hough_candidates, _ = hough_line_debug(closed_edges, width, height, min_area, idx, closed_edges_used=True)
        candidates = contour_candidates + hough_candidates + closed_contour_candidates + closed_hough_candidates

        for candidate in candidates:
            candidate.score = edge_distance_score(candidate.quad, artifacts.dist, artifacts.grad_mag)
            all_candidates.append(candidate)

    if not all_candidates:
        raise RuntimeError("No valid square candidate found")

    all_candidates.sort(
        key=lambda c: candidate_selection_score(
            c,
            c.score if c.score is not None else float("inf"),
            width,
            height,
            yellow_mask,
            image,
        )
    )
    shortlist = all_candidates[: min(12, len(all_candidates))]

    best_quad = None
    best_score = float("inf")
    best_selection_score = float("inf")

    for candidate in shortlist:
        artifacts = edge_variants[candidate.variant_idx]
        reg_weight = 0.20 if candidate.source == "hough_dominant" else 0.05
        refined = refine_candidate(candidate.quad, artifacts.dist, width, height, min_area, reg_weight=reg_weight)
        refined_score = edge_distance_score(refined, artifacts.dist, artifacts.grad_mag)
        refined_candidate = Candidate(quad=refined, source=candidate.source, variant_idx=candidate.variant_idx)
        selection_score = candidate_selection_score(
            refined_candidate,
            refined_score,
            width,
            height,
            yellow_mask,
            image,
        )
        if selection_score < best_selection_score:
            if best_quad is not None:
                rejected_debug.append(best_quad)
            best_selection_score = selection_score
            best_score = refined_score
            best_quad = refined
        else:
            rejected_debug.append(refined)

    if best_quad is None:
        raise RuntimeError("Candidate refinement failed")

    return FitResult(quad=best_quad, score=best_score, rejected=rejected_debug, source="hough")


def warp_square_cutout(image: np.ndarray, quad: np.ndarray, out_size: int) -> np.ndarray:
    dst_square = np.array(
        [
            [0, 0],
            [out_size - 1, 0],
            [out_size - 1, out_size - 1],
            [0, out_size - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_square)
    return cv2.warpPerspective(image, matrix, (out_size, out_size))


def _draw_line_set(canvas: np.ndarray, lines: np.ndarray | None, color: tuple[int, int, int], thickness: int) -> None:
    if lines is None:
        return
    for x1, y1, x2, y2 in lines.astype(np.int32):
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)


def _draw_hough_debug(image: np.ndarray, debug_items: list[LineDebug]) -> np.ndarray:
    canvas = image.copy()
    for item in debug_items:
        raw_color = (120, 120, 120) if not item.closed_edges_used else (80, 80, 160)
        _draw_line_set(canvas, item.lines, raw_color, 1)
        _draw_line_set(canvas, item.family_a, (255, 0, 0), 2)
        _draw_line_set(canvas, item.family_b, (0, 255, 255), 2)
        for quad in item.candidates:
            cv2.polylines(canvas, [np.rint(quad).astype(np.int32)], True, (0, 180, 0), 1, cv2.LINE_AA)
    return canvas


def _draw_detected_quad(image: np.ndarray, quad: np.ndarray | None, source: str | None = None) -> np.ndarray:
    canvas = image.copy()
    if quad is None:
        cv2.putText(
            canvas,
            "NO MARKER DETECTED",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        return canvas

    points = np.rint(quad).astype(np.int32)
    cv2.polylines(canvas, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    if source:
        cv2.putText(
            canvas,
            source,
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    for idx, point in enumerate(points):
        cv2.circle(canvas, tuple(int(value) for value in point), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(idx),
            tuple(int(value) for value in point + np.array([8, -8], dtype=np.int32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _write_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        logger.warning("Failed to write marker rectifier debug image: %s", path)


class MarkerRectificationModule(BaseModule[VideoFrame | np.ndarray]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        out_size: int = 512,
        preprocess_mode: EnhancementMode = None,
        debug: bool = False,
        debug_dir: Path | str = Path("data/debug"),
    ) -> None:
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        if out_size <= 0:
            raise ValueError("out_size must be greater than zero.")

        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.out_size = out_size
        self.preprocess_mode = preprocess_mode
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _debug_input_path(self) -> Path:
        return self.debug_dir / "marker_input.png"

    @property
    def _debug_hough_lines_path(self) -> Path:
        return self.debug_dir / "marker_hough_lines.png"

    @property
    def _debug_all_hough_lines_path(self) -> Path:
        return self.debug_dir / "marker_hough_lines_all_variants.png"

    @property
    def _debug_detected_quad_path(self) -> Path:
        return self.debug_dir / "marker_detected_quad.png"

    @property
    def _debug_cutout_path(self) -> Path:
        return self.debug_dir / "marker_rectified_cutout.png"

    @property
    def _debug_yellow_mask_path(self) -> Path:
        return self.debug_dir / "marker_yellow_suppression_mask.png"

    def _write_debug_images(
        self,
        image: np.ndarray,
        line_debug: list[LineDebug],
        quad: np.ndarray | None,
        cutout: np.ndarray | None,
        source: str | None = None,
    ) -> None:
        if not self.debug:
            return

        _write_debug_image(self._debug_input_path, image)
        _write_debug_image(self._debug_yellow_mask_path, yellow_pipe_mask(image))
        _write_debug_image(self._debug_hough_lines_path, _draw_hough_debug(image, line_debug[:4]))
        _write_debug_image(self._debug_all_hough_lines_path, _draw_hough_debug(image, line_debug))
        _write_debug_image(self._debug_detected_quad_path, _draw_detected_quad(image, quad, source))
        if cutout is None:
            cutout = np.zeros((self.out_size, self.out_size, image.shape[2]), dtype=image.dtype)
        _write_debug_image(self._debug_cutout_path, cutout)

    async def process(
        self,
        message: Message[VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[np.ndarray] | None:
        payload = message.payload
        image = payload.image if isinstance(payload, VideoFrame) else payload
        validate_color_image(image)

        line_debug: list[LineDebug] = []
        try:
            _, edge_variants = build_edge_variants(image, self.preprocess_mode)
            if self.debug:
                line_debug = collect_line_debug(image, edge_variants)
            fit_result = fit_square(image, edge_variants)
        except RuntimeError as exc:
            self._write_debug_images(image, line_debug, quad=None, cutout=None)
            logger.warning("Dropping frame without detected marker: %s", exc)
            return None

        cutout = warp_square_cutout(image, fit_result.quad, self.out_size)
        self._write_debug_images(
            image,
            line_debug,
            quad=fit_result.quad,
            cutout=cutout,
            source=fit_result.source,
        )
        metadata: dict[str, Any] = dict(message.metadata)
        metadata.update(
            {
                "quad": fit_result.quad.tolist(),
                "score": float(fit_result.score),
                "quad_source": fit_result.source,
                "input_shape": tuple(int(value) for value in image.shape),
            }
        )
        if isinstance(payload, VideoFrame):
            metadata.update(
                {
                    "frame_index": payload.frame_index,
                    "timestamp_seconds": payload.timestamp_seconds,
                    "loop_count": payload.loop_count,
                }
            )

        logger.debug(
            "Rectified marker with score %.3f to %sx%s cutout",
            fit_result.score,
            self.out_size,
            self.out_size,
        )
        return RoutedMessage(
            destination=self.output_queue,
            message=Message(payload=cutout, metadata=metadata),
        )
