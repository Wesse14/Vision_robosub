from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Sequence

from src import (
    ArucoDetectionModule,
    AsyncProcessor,
    FrameRateLoggerModule,
    GMMColorMaskModule,
    ImageEnhancementModule,
    LoopingVideoSource,
    MarkerRectificationModule,
    ProcessorLoop,
    QueueFanoutModule,
    configure_logging,
)

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_PATH = (
    Path(__file__).parent / "data" / "1-input.mp4"
)
FRAME_QUEUE = "frames"
ENHANCED_FRAME_QUEUE = "enhanced_frames"
MARKER_CUTOUT_QUEUE = "marker_cutouts"
ARUCO_DETECTION_QUEUE = "aruco_detections"
GMM_MODEL_PATH = Path("data/color_classifier_gmm.joblib")
GMM_FRAME_QUEUE = "gmm_frames"
MARKER_FRAME_QUEUE = "marker_frames"
COLOR_MASK_QUEUE = "color_masks"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the async processor loop.")
    parser.add_argument(
        "--video-path",
        default=DEFAULT_VIDEO_PATH,
        type=Path,
        help="Video file to poll as the input source.",
    )
    parser.add_argument(
        "--queue-size",
        default=2,
        type=int,
        help="Maximum number of queued frames waiting for processing.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Read video frames as fast as processing allows instead of at video FPS.",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write marker-rectifier debug images to data/debug/.",
    )
    return parser.parse_args(argv)


async def run_app(args: argparse.Namespace) -> None:
    processor = AsyncProcessor()
    gmm_model_exists = GMM_MODEL_PATH.exists()
    marker_input_queue = MARKER_FRAME_QUEUE if gmm_model_exists else ENHANCED_FRAME_QUEUE

    processor.create_queue(FRAME_QUEUE, maxsize=args.queue_size)
    processor.create_queue(ENHANCED_FRAME_QUEUE, maxsize=args.queue_size)
    processor.create_queue(MARKER_CUTOUT_QUEUE, maxsize=args.queue_size)
    processor.create_queue(ARUCO_DETECTION_QUEUE, maxsize=args.queue_size)
    if gmm_model_exists:
        processor.create_queue(MARKER_FRAME_QUEUE, maxsize=args.queue_size)
        processor.create_queue(GMM_FRAME_QUEUE, maxsize=args.queue_size)
        processor.create_queue(COLOR_MASK_QUEUE)

    processor.register_module(
        ImageEnhancementModule(
            name="image-enhancer",
            input_queue=FRAME_QUEUE,
            output_queue=ENHANCED_FRAME_QUEUE,
        )
    )
    if gmm_model_exists:
        processor.register_module(
            QueueFanoutModule(
                name="enhanced-frame-fanout",
                input_queue=ENHANCED_FRAME_QUEUE,
                output_queues=[MARKER_FRAME_QUEUE, GMM_FRAME_QUEUE],
            )
        )
        processor.register_module(
            GMMColorMaskModule(
                name="gmm-color-mask",
                input_queue=GMM_FRAME_QUEUE,
                output_queue=COLOR_MASK_QUEUE,
                model_path=GMM_MODEL_PATH,
                debug=args.debug,
                debug_dir=Path("data/debug"),
            )
        )
        logger.info("GMM color mask module enabled with model %s", GMM_MODEL_PATH)
    else:
        logger.info("GMM color classifier model not found at %s; module disabled", GMM_MODEL_PATH)

    processor.register_module(
        MarkerRectificationModule(
            name="marker-rectifier",
            input_queue=marker_input_queue,
            output_queue=MARKER_CUTOUT_QUEUE,
            debug=args.debug,
            debug_dir=Path("data/debug"),
        )
    )
    processor.register_module(
        ArucoDetectionModule(
            name="aruco-detector",
            input_queue=MARKER_CUTOUT_QUEUE,
            output_queue=ARUCO_DETECTION_QUEUE,
            debug=args.debug,
            debug_dir=Path("data/debug"),
        )
    )
    processor.register_module(
        FrameRateLoggerModule(
            name="frame-rate-logger",
            input_queue=ARUCO_DETECTION_QUEUE,
        )
    )

    source = LoopingVideoSource(args.video_path, realtime=not args.no_realtime)
    runner = ProcessorLoop(
        processor,
        input_queue=FRAME_QUEUE,
        source=source,
        poll_interval=0,
    )

    logger.info("Reading frames from %s. Press Ctrl+C to stop.", args.video_path)
    await runner.run_until_interrupted()
    logger.info("Async processor loop stopped cleanly.")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, use_colors=not args.no_color)
    asyncio.run(run_app(args))


if __name__ == "__main__":
    main()
