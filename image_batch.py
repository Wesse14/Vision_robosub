from __future__ import annotations

import argparse
import asyncio
import logging
import random
import shutil
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src import (
    AsyncProcessor,
    ArucoDetectionModule,
    ImageEnhancementModule,
    MarkerRectificationModule,
    Message,
    VideoFrame,
    configure_logging,
)
from src.modules.image_enhancer import apply_enhancement

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("data/test_images")
DEFAULT_OUTPUT_DIR = Path("data/test_results")
DEFAULT_VIDEO_DIR = Path("data/test_videos")
SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg"}
PIPELINES = ("enhance", "marker", "aruco", "full")
GENERATED_FRAME_PREFIX = "video_frame__"
QUAD_SUMMARY_DIR_NAME = "_marker_detected_quads"
NAVIGATION_SUMMARY_FILENAME = "navigation_signals.txt"


@dataclass(frozen=True, slots=True)
class NavigationSummary:
    image: str
    signal: str
    combined_score: float
    elapsed_ms: float
    marker_id: int | None
    aruco_ids: tuple[int, ...]
    mask_match_id: int | None
    mask_match_score: float
    grid_match_id: int | None
    grid_match_score: float
    reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test images through one or more image processing modules.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        type=Path,
        help="Directory with input images.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where result images are written.",
    )
    parser.add_argument(
        "--video-dir",
        default=DEFAULT_VIDEO_DIR,
        type=Path,
        help="Directory with videos to sample into the input image directory.",
    )
    parser.add_argument(
        "--frames-per-video",
        default=10,
        type=int,
        help="Maximum number of random frames to extract per video on every run.",
    )
    parser.add_argument(
        "--pipeline",
        default="full",
        choices=PIPELINES,
        help="Which module path to run for every input image.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write module debug images under the output directory.",
    )
    parser.add_argument(
        "--save-marker-quads",
        action="store_true",
        help="Write marker detected quad summary images without enabling full debug output.",
    )
    parser.add_argument(
        "--experimental-color-repaint-retry",
        action="store_true",
        help="Retry marker detection after repainting yellow/blue suppression-mask pixels red.",
    )
    parser.add_argument(
        "--max-image-seconds",
        default=1.5,
        type=float,
        help="Skip an image once processing exceeds this many seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep watching the input directory and process new or changed images.",
    )
    parser.add_argument(
        "--watch-poll-seconds",
        default=0.5,
        type=float,
        help="How often to check for new images when --watch is enabled.",
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Minimum log level to show.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in log output.",
    )
    return parser.parse_args(argv)


def iter_video_paths(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        video_dir.mkdir(parents=True)
        logger.info("Created %s. Add videos there to sample random frames.", video_dir)
        return []

    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def iter_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        input_dir.mkdir(parents=True)
        logger.warning(
            "Created %s. Add test images there and run this command again.",
            input_dir,
        )
        return []

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def image_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


async def wait_until_file_is_stable(
    path: Path,
    *,
    poll_seconds: float,
    checks: int = 2,
) -> tuple[int, int] | None:
    previous = image_fingerprint(path)
    if previous is None:
        return None

    stable_checks = 0
    while stable_checks < checks:
        await asyncio.sleep(max(poll_seconds, 0.05))
        current = image_fingerprint(path)
        if current is None:
            return None
        if current == previous:
            stable_checks += 1
        else:
            stable_checks = 0
            previous = current
    return previous


def clear_generated_video_frames(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.iterdir():
        if path.is_file() and path.name.startswith(GENERATED_FRAME_PREFIX):
            path.unlink()


def extract_random_video_frames(
    video_path: Path,
    input_dir: Path,
    *,
    max_frames: int,
) -> int:
    if max_frames <= 0:
        raise ValueError("--frames-per-video must be greater than zero.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.warning("Skipping unreadable video: %s", video_path)
        return 0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        logger.warning("Skipping video with unknown frame count: %s", video_path)
        capture.release()
        return 0

    frame_indices = sorted(random.sample(range(frame_count), min(max_frames, frame_count)))
    written = 0

    try:
        for output_index, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                logger.warning("Could not read frame %s from %s", frame_index, video_path)
                continue

            output_path = input_dir / (
                f"{GENERATED_FRAME_PREFIX}{video_path.stem}__"
                f"{output_index:02d}_of_{len(frame_indices):02d}__"
                f"source_{frame_index:06d}.png"
            )
            write_image(output_path, frame)
            written += 1
    finally:
        capture.release()

    logger.info("Extracted %s random frame(s) from %s", written, video_path)
    return written


def refresh_video_frames(video_dir: Path, input_dir: Path, frames_per_video: int) -> None:
    video_paths = iter_video_paths(video_dir)
    clear_generated_video_frames(input_dir)

    if not video_paths:
        return

    total_written = 0
    for video_path in video_paths:
        total_written += extract_random_video_frames(
            video_path,
            input_dir,
            max_frames=frames_per_video,
        )

    logger.info(
        "Refreshed %s generated video frame(s) in %s",
        total_written,
        input_dir,
    )


def read_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("Skipping unreadable image: %s", path)
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def combined_navigation_score(
    mask_match_id: int | None,
    mask_match_score: float,
    grid_match_id: int | None,
    grid_match_score: float,
) -> float:
    if mask_match_id is None or mask_match_id != grid_match_id:
        return 0.0
    return min(mask_match_score, grid_match_score)


def format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def format_ids(ids: tuple[int, ...]) -> str:
    return ",".join(str(marker_id) for marker_id in ids)


def navigation_summary_text(summaries: Sequence[NavigationSummary]) -> str:
    rows = [
        "\t".join(
            [
                "image",
                "signal",
                "combined_score",
                "elapsed_ms",
                "marker_id",
                "aruco_ids",
                "mask_match_id",
                "mask_match_score",
                "grid_match_id",
                "grid_match_score",
                "reason",
            ]
        )
    ]
    for summary in summaries:
        rows.append(
            "\t".join(
                [
                    summary.image,
                    summary.signal,
                    f"{summary.combined_score:.3f}",
                    f"{summary.elapsed_ms:.1f}",
                    format_optional_int(summary.marker_id),
                    format_ids(summary.aruco_ids),
                    format_optional_int(summary.mask_match_id),
                    f"{summary.mask_match_score:.3f}",
                    format_optional_int(summary.grid_match_id),
                    f"{summary.grid_match_score:.3f}",
                    summary.reason,
                ]
            )
        )
    return "\n".join(rows) + "\n"


def write_navigation_summary(output_dir: Path, summaries: Sequence[NavigationSummary]) -> None:
    write_text(output_dir / NAVIGATION_SUMMARY_FILENAME, navigation_summary_text(summaries))


def image_time_limit_seconds(args: argparse.Namespace) -> float | None:
    limit = float(args.max_image_seconds)
    return limit if limit > 0 else None


def image_timed_out(started_at: float, args: argparse.Namespace) -> bool:
    limit = image_time_limit_seconds(args)
    return limit is not None and (time.perf_counter() - started_at) >= limit


def image_remaining_seconds(started_at: float, args: argparse.Namespace) -> float | None:
    limit = image_time_limit_seconds(args)
    if limit is None:
        return None
    return max(0.001, limit - (time.perf_counter() - started_at))


def timeout_navigation_summary(
    path: Path,
    started_at: float,
    reason: str = "image processing exceeded the configured time limit",
) -> NavigationSummary:
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return NavigationSummary(
        image=path.name,
        signal="SKIPPED_TIMEOUT",
        combined_score=0.0,
        elapsed_ms=elapsed_ms,
        marker_id=None,
        aruco_ids=(),
        mask_match_id=None,
        mask_match_score=0.0,
        grid_match_id=None,
        grid_match_score=0.0,
        reason=reason,
    )


def writes_visual_results(pipeline: str) -> bool:
    return pipeline != "full"


def draw_summary_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2
    padding = 10
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(
        canvas,
        (0, 0),
        (min(canvas.shape[1], text_w + padding * 2), text_h + baseline + padding * 2),
        (0, 0, 0),
        -1,
    )
    color = (0, 200, 0) if label.startswith("NEXT") else (0, 0, 255)
    cv2.putText(
        canvas,
        label,
        (padding, padding + text_h),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def draw_marker_detected_quad_summary(
    image: np.ndarray,
    quad: Sequence[Sequence[float]],
    label: str | None = None,
) -> np.ndarray:
    canvas = image.copy()
    points = np.rint(np.asarray(quad, dtype=np.float32)).astype(np.int32)
    cv2.polylines(canvas, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    if label:
        canvas = draw_summary_label(canvas, label)
    return canvas


def marker_summary_label(signal: str, marker_id: int | None) -> str:
    return f"{signal}: ID {marker_id}" if marker_id is not None else signal


def save_marker_detected_quad_summary(
    output_root: Path,
    source_image_path: Path,
    debug_dir: Path,
    marker_input_image: np.ndarray | None,
    marker_metadata: dict,
    label: str | None = None,
) -> None:
    summary_dir = output_root / QUAD_SUMMARY_DIR_NAME
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = summary_dir / f"{source_image_path.stem}_marker_detected_quad.png"

    quad_path = debug_dir / "marker" / "marker_detected_quad.png"
    if quad_path.exists():
        image = cv2.imread(str(quad_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Could not read marker detected quad debug image: %s", quad_path)
            return
        if label:
            image = draw_summary_label(image, label)
        write_image(output_path, image)
        return

    quad = marker_metadata.get("quad")
    if marker_input_image is None or quad is None:
        return

    write_image(output_path, draw_marker_detected_quad_summary(marker_input_image, quad, label))


def payload_image(payload: VideoFrame | np.ndarray) -> np.ndarray:
    return payload.image if isinstance(payload, VideoFrame) else payload


def message_with_image(
    message: Message[VideoFrame | np.ndarray],
    image: np.ndarray,
    attempt: str,
) -> Message[VideoFrame | np.ndarray]:
    payload = message.payload
    next_payload: VideoFrame | np.ndarray
    if isinstance(payload, VideoFrame):
        next_payload = replace(payload, image=image)
    else:
        next_payload = image
    metadata = dict(message.metadata)
    metadata["marker_preprocess_attempt"] = attempt
    return Message(next_payload, metadata=metadata)


def clahe_bgr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    return cv2.cvtColor(
        cv2.merge([clahe.apply(l_channel), a_channel, b_channel]),
        cv2.COLOR_LAB2BGR,
    )


def high_contrast_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    boosted = clahe.apply(gray)
    sharpened = cv2.addWeighted(boosted, 2.0, cv2.GaussianBlur(boosted, (0, 0), 1.0), -1.0, 0)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def sharpen_bgr(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.8, blurred, -0.8, 0)


def marker_retry_messages(
    original_message: Message[VideoFrame | np.ndarray],
    current_message: Message[VideoFrame | np.ndarray],
) -> list[Message[VideoFrame | np.ndarray]]:
    original_image = payload_image(original_message.payload)
    current_image = payload_image(current_message.payload)
    retry_images = [
        ("raw", original_image),
        ("underwater", apply_enhancement(original_image, "underwater")),
        ("clahe", clahe_bgr(original_image)),
        ("high_contrast", high_contrast_bgr(original_image)),
        ("sharpened", sharpen_bgr(original_image)),
        ("current_clahe", clahe_bgr(current_image)),
        ("current_high_contrast", high_contrast_bgr(current_image)),
    ]

    messages: list[Message[VideoFrame | np.ndarray]] = []
    seen: set[bytes] = {current_image[:: max(current_image.shape[0] // 16, 1), :: max(current_image.shape[1] // 16, 1)].tobytes()}
    for name, enhanced_image in retry_images:
        signature = enhanced_image[
            :: max(enhanced_image.shape[0] // 16, 1),
            :: max(enhanced_image.shape[1] // 16, 1),
        ].tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        messages.append(message_with_image(original_message, enhanced_image, name))
    return messages


def clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


async def run_image(path: Path, args: argparse.Namespace) -> NavigationSummary | None:
    image = read_image(path)
    if image is None:
        return None
    started_at = time.perf_counter()

    stem = path.stem
    output_dir = args.output_dir / stem
    debug_dir = output_dir / "debug"
    write_visual_results = writes_visual_results(args.pipeline)
    context = AsyncProcessor()
    frame = VideoFrame(
        image=image,
        frame_index=0,
        timestamp_seconds=0.0,
        loop_count=0,
    )
    message = Message(frame, metadata={"source_path": str(path)})
    original_message = message

    if args.pipeline in {"enhance", "aruco", "full"}:
        enhancer = ImageEnhancementModule(
            name="image-enhancer",
            input_queue="frames",
            output_queue="enhanced_frames",
        )
        enhanced = await enhancer.process(message, context)
        message = enhanced.message
        if write_visual_results:
            write_image(output_dir / "01_enhanced.png", message.payload.image)

    marker_image = None
    marker_message = None
    marker_input_image = None
    navigation_label = None
    navigation_summary = None
    if args.pipeline in {"marker", "aruco", "full"}:
        marker_input_image = payload_image(message.payload).copy()
        marker = MarkerRectificationModule(
            name="marker-rectifier",
            input_queue="frames",
            output_queue="marker_cutouts",
            debug=args.debug,
            debug_dir=debug_dir / "marker",
            experimental_color_repaint_retry=args.experimental_color_repaint_retry,
            max_processing_seconds=image_remaining_seconds(started_at, args),
        )
        marker_result = await marker.process(message, context)
        if marker_result is None:
            if image_timed_out(started_at, args):
                return timeout_navigation_summary(path, started_at)
            if image_time_limit_seconds(args) is not None:
                return timeout_navigation_summary(
                    path,
                    started_at,
                    "marker fallback skipped to stay under the configured time limit",
                )
            if image_time_limit_seconds(args) is None:
                for retry_message in marker_retry_messages(original_message, message):
                    attempt = retry_message.metadata["marker_preprocess_attempt"]
                    logger.info("Retrying marker detection for %s with %s preprocessing", path, attempt)
                    retry_input_image = payload_image(retry_message.payload).copy()
                    marker = MarkerRectificationModule(
                        name=f"marker-rectifier-{attempt}",
                        input_queue="frames",
                        output_queue="marker_cutouts",
                        debug=args.debug,
                        debug_dir=debug_dir / "marker",
                        experimental_color_repaint_retry=args.experimental_color_repaint_retry,
                        max_processing_seconds=None,
                    )
                    marker_result = await marker.process(retry_message, context)
                    if marker_result is not None:
                        marker_input_image = retry_input_image
                        logger.info("Marker detected in %s after %s preprocessing", path, attempt)
                        break
            if marker_result is None:
                logger.warning("No marker detected in %s", path)

        if marker_result is not None:
            marker_payload = marker_result.message.payload
            marker_image = (
                marker_payload.image
                if isinstance(marker_payload, VideoFrame)
                else marker_payload
            )
            if write_visual_results:
                write_image(output_dir / "02_marker_cutout.png", marker_image)
            marker_message = marker_result.message
            attempt = marker_message.metadata.get("marker_preprocess_attempt", "initial")
            if write_visual_results and attempt != "initial":
                write_image(output_dir / f"02_marker_cutout_{attempt}.png", marker_image)

    if args.pipeline in {"aruco", "full"} and marker_image is not None and marker_message is not None:
        aruco = ArucoDetectionModule(
            name="aruco-detector",
            input_queue="marker_cutouts",
            output_queue="aruco_detections",
            debug=args.debug,
            debug_dir=debug_dir / "aruco",
        )
        aruco_result = await aruco.process(marker_message, context)
        if image_timed_out(started_at, args):
            return timeout_navigation_summary(path, started_at)
        aruco_detection = aruco_result.message.payload
        combined_score = combined_navigation_score(
            aruco_detection.mask_match_id,
            aruco_detection.mask_match_score,
            aruco_detection.grid_match_id,
            aruco_detection.grid_match_score,
        )
        navigation_summary = NavigationSummary(
            image=path.name,
            signal=aruco_detection.navigation_signal,
            combined_score=combined_score,
            elapsed_ms=0.0,
            marker_id=aruco_detection.navigation_marker_id,
            aruco_ids=aruco_detection.ids,
            mask_match_id=aruco_detection.mask_match_id,
            mask_match_score=aruco_detection.mask_match_score,
            grid_match_id=aruco_detection.grid_match_id,
            grid_match_score=aruco_detection.grid_match_score,
            reason=aruco_detection.navigation_reason,
        )
        navigation_label = marker_summary_label(
            aruco_detection.navigation_signal,
            aruco_detection.navigation_marker_id,
        )
        if write_visual_results:
            write_image(
                output_dir / "03_aruco_detected.png",
                aruco_detection.annotated_image,
            )
            write_image(
                output_dir / "04_aruco_high_contrast_retry.png",
                aruco_detection.high_contrast_annotated_image,
            )
            write_image(
                output_dir / "05_aruco_mask_match.png",
                aruco_detection.mask_match_image,
            )
            write_image(
                output_dir / "06_aruco_grid.png",
                aruco_detection.grid_image,
            )
            write_image(
                output_dir / "07_aruco_grid_match.png",
                aruco_detection.grid_match_image,
            )
            write_text(
                output_dir / "09_navigation_signal.txt",
                "\n".join(
                    [
                        f"signal={aruco_detection.navigation_signal}",
                        f"marker_id={aruco_detection.navigation_marker_id}",
                        f"reason={aruco_detection.navigation_reason}",
                        f"aruco_ids={aruco_detection.ids}",
                        f"combined_score={combined_score:.3f}",
                        f"mask_match_id={aruco_detection.mask_match_id}",
                        f"mask_match_score={aruco_detection.mask_match_score:.3f}",
                        f"grid_match_id={aruco_detection.grid_match_id}",
                        f"grid_match_score={aruco_detection.grid_match_score:.3f}",
                    ]
                )
                + "\n",
            )

    if marker_message is not None and (args.debug or args.save_marker_quads):
        save_marker_detected_quad_summary(
            args.output_dir,
            path,
            debug_dir,
            marker_input_image,
            marker_message.metadata,
            navigation_label,
        )

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    logger.info("Processed %s -> %s in %.1f ms", path, output_dir, elapsed_ms)
    if navigation_summary is not None:
        navigation_summary = replace(navigation_summary, elapsed_ms=elapsed_ms)
    if navigation_summary is None and args.pipeline in {"aruco", "full"}:
        return NavigationSummary(
            image=path.name,
            signal="NO_MARKER",
            combined_score=0.0,
            elapsed_ms=elapsed_ms,
            marker_id=None,
            aruco_ids=(),
            mask_match_id=None,
            mask_match_score=0.0,
            grid_match_id=None,
            grid_match_score=0.0,
            reason="marker rectification did not produce a cutout",
        )
    return navigation_summary


async def run_batch(args: argparse.Namespace) -> None:
    refresh_video_frames(args.video_dir, args.input_dir, args.frames_per_video)
    if args.watch:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / NAVIGATION_SUMMARY_FILENAME
        if not summary_path.exists():
            write_navigation_summary(args.output_dir, [])
    else:
        clear_output_dir(args.output_dir)
    processed: dict[Path, tuple[int, int]] = {}
    summaries: list[NavigationSummary] = []

    while True:
        image_paths = iter_image_paths(args.input_dir)
        if not image_paths and not args.watch:
            logger.warning("No test images found in %s", args.input_dir)
            return

        for path in image_paths:
            current_fingerprint = image_fingerprint(path)
            if current_fingerprint is None or processed.get(path) == current_fingerprint:
                continue

            fingerprint = (
                await wait_until_file_is_stable(
                    path,
                    poll_seconds=min(max(args.watch_poll_seconds, 0.05), 0.5),
                )
                if args.watch
                else current_fingerprint
            )
            if fingerprint is None or processed.get(path) == fingerprint:
                continue

            summary = await run_image(path, args)
            processed[path] = fingerprint
            if summary is not None:
                summaries = [existing for existing in summaries if existing.image != summary.image]
                summaries.append(summary)
                summaries.sort(key=lambda summary: summary.image)
                write_navigation_summary(args.output_dir, summaries)

        if not args.watch:
            return

        await asyncio.sleep(max(args.watch_poll_seconds, 0.05))


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, use_colors=not args.no_color)
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
