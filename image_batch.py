from __future__ import annotations

import argparse
import asyncio
import logging
import random
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src import (
    AsyncProcessor,
    ArucoDetectionModule,
    GMMColorMaskModule,
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
DEFAULT_GMM_MODEL_PATH = Path("data/color_classifier_gmm.joblib")
SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg"}
PIPELINES = ("enhance", "marker", "aruco", "gmm", "full")
GENERATED_FRAME_PREFIX = "video_frame__"


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
        "--gmm-model-path",
        default=DEFAULT_GMM_MODEL_PATH,
        type=Path,
        help="Path to the GMM color classifier model.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write module debug images under the output directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
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


async def run_image(path: Path, args: argparse.Namespace) -> None:
    image = read_image(path)
    if image is None:
        return

    stem = path.stem
    output_dir = args.output_dir / stem
    debug_dir = output_dir / "debug"
    context = AsyncProcessor()
    frame = VideoFrame(
        image=image,
        frame_index=0,
        timestamp_seconds=0.0,
        loop_count=0,
    )
    message = Message(frame, metadata={"source_path": str(path)})
    original_message = message

    if args.pipeline in {"enhance", "full"}:
        enhancer = ImageEnhancementModule(
            name="image-enhancer",
            input_queue="frames",
            output_queue="enhanced_frames",
        )
        enhanced = await enhancer.process(message, context)
        message = enhanced.message
        write_image(output_dir / "01_enhanced.png", message.payload.image)

    marker_image = None
    marker_message = None
    if args.pipeline in {"marker", "aruco", "full"}:
        marker = MarkerRectificationModule(
            name="marker-rectifier",
            input_queue="frames",
            output_queue="marker_cutouts",
            debug=args.debug,
            debug_dir=debug_dir / "marker",
        )
        marker_result = await marker.process(message, context)
        if marker_result is None:
            for retry_message in marker_retry_messages(original_message, message):
                attempt = retry_message.metadata["marker_preprocess_attempt"]
                logger.info("Retrying marker detection for %s with %s preprocessing", path, attempt)
                marker = MarkerRectificationModule(
                    name=f"marker-rectifier-{attempt}",
                    input_queue="frames",
                    output_queue="marker_cutouts",
                    debug=args.debug,
                    debug_dir=debug_dir / "marker",
                )
                marker_result = await marker.process(retry_message, context)
                if marker_result is not None:
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
            write_image(output_dir / "02_marker_cutout.png", marker_image)
            marker_message = marker_result.message
            attempt = marker_message.metadata.get("marker_preprocess_attempt", "initial")
            if attempt != "initial":
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
        write_image(
            output_dir / "03_aruco_detected.png",
            aruco_result.message.payload.annotated_image,
        )
        write_image(
            output_dir / "04_aruco_high_contrast_retry.png",
            aruco_result.message.payload.high_contrast_annotated_image,
        )
        write_image(
            output_dir / "05_aruco_mask_match.png",
            aruco_result.message.payload.mask_match_image,
        )
        write_image(
            output_dir / "06_aruco_grid.png",
            aruco_result.message.payload.grid_image,
        )
        write_image(
            output_dir / "07_aruco_grid_match.png",
            aruco_result.message.payload.grid_match_image,
        )

    if args.pipeline in {"gmm", "full"}:
        if not args.gmm_model_path.exists():
            logger.warning(
                "Skipping GMM for %s; model not found at %s",
                path,
                args.gmm_model_path,
            )
        else:
            gmm = GMMColorMaskModule(
                name="gmm-color-mask",
                input_queue="frames",
                output_queue="color_masks",
                model_path=args.gmm_model_path,
                debug=args.debug,
                debug_dir=debug_dir / "gmm",
            )
            gmm_result = await gmm.process(message, context)
            write_image(output_dir / "08_color_mask.png", gmm_result.message.payload)

    logger.info("Processed %s -> %s", path, output_dir)


async def run_batch(args: argparse.Namespace) -> None:
    refresh_video_frames(args.video_dir, args.input_dir, args.frames_per_video)
    clear_output_dir(args.output_dir)
    image_paths = iter_image_paths(args.input_dir)
    if not image_paths:
        logger.warning("No test images found in %s", args.input_dir)
        return

    for path in image_paths:
        await run_image(path, args)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, use_colors=not args.no_color)
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
