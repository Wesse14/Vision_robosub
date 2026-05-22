from __future__ import annotations

import argparse
import asyncio
import logging
import random
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src import (
    AsyncProcessor,
    GMMColorMaskModule,
    ImageEnhancementModule,
    MarkerRectificationModule,
    Message,
    VideoFrame,
    configure_logging,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("data/test_images")
DEFAULT_OUTPUT_DIR = Path("data/test_results")
DEFAULT_VIDEO_DIR = Path("data/test_videos")
DEFAULT_GMM_MODEL_PATH = Path("data/color_classifier_gmm.joblib")
SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg"}
PIPELINES = ("enhance", "marker", "gmm", "full")
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

    if args.pipeline in {"enhance", "full"}:
        enhancer = ImageEnhancementModule(
            name="image-enhancer",
            input_queue="frames",
            output_queue="enhanced_frames",
        )
        enhanced = await enhancer.process(message, context)
        message = enhanced.message
        write_image(output_dir / "01_enhanced.png", message.payload.image)

    if args.pipeline in {"marker", "full"}:
        marker = MarkerRectificationModule(
            name="marker-rectifier",
            input_queue="frames",
            output_queue="marker_cutouts",
            debug=args.debug,
            debug_dir=debug_dir / "marker",
        )
        marker_result = await marker.process(message, context)
        if marker_result is None:
            logger.warning("No marker detected in %s", path)
        else:
            marker_payload = marker_result.message.payload
            marker_image = (
                marker_payload.image
                if isinstance(marker_payload, VideoFrame)
                else marker_payload
            )
            write_image(output_dir / "02_marker_cutout.png", marker_image)

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
            write_image(output_dir / "03_color_mask.png", gmm_result.message.payload)

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
