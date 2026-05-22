from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

import joblib
import numpy as np
import cv2
import pytest

import main as app_main

from src.modules.marker_rectifier import refine_candidate

from src import (
    AsyncProcessor,
    BaseModule,
    ColorFormatter,
    DuplicateModuleError,
    DuplicateQueueError,
    FrameRateLoggerModule,
    ImageEnhancementModule,
    GMMColorMaskModule,
    MarkerRectificationModule,
    LoopingVideoSource,
    Message,
    ModuleContext,
    ProcessorLoop,
    QueueFanoutModule,
    RoutedMessage,
    SignalStopper,
    UnknownQueueError,
    VideoFrame,
)


TEST_VIDEO_PATH = Path(__file__).parents[1] / "data" / "1-input.mp4"


class DistanceGMM:
    def __init__(self, center: np.ndarray) -> None:
        self.center = center.astype(np.float64)

    def score_samples(self, pixels: np.ndarray) -> np.ndarray:
        delta = pixels.astype(np.float64) - self.center
        return -np.sum(delta * delta, axis=1) / 100.0


def write_synthetic_gmm_model(path: Path, query_bgr: tuple[int, int, int]) -> None:
    query_pixel = np.array([[query_bgr]], dtype=np.uint8)
    query_lab = cv2.cvtColor(query_pixel, cv2.COLOR_BGR2LAB).reshape(3)
    non_query_lab = cv2.cvtColor(
        np.array([[[255, 0, 0]]], dtype=np.uint8),
        cv2.COLOR_BGR2LAB,
    ).reshape(3)
    joblib.dump(
        {
            "query_gmm": DistanceGMM(query_lab),
            "non_query_gmm": DistanceGMM(non_query_lab),
            "query_prior": 0.5,
            "non_query_prior": 0.5,
        },
        path,
    )


class UppercaseModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> RoutedMessage[str]:
        return RoutedMessage.from_payload("out", message.payload.upper())


class FanoutModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> list[RoutedMessage[str]]:
        return [
            RoutedMessage.from_payload("out_a", f"{message.payload}:a"),
            RoutedMessage.from_payload("out_b", f"{message.payload}:b"),
        ]


class MissingRouteModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> RoutedMessage[str]:
        return RoutedMessage.from_payload("missing", message.payload)


class SinkModule(BaseModule[str]):
    def __init__(self, name: str, input_queue: str) -> None:
        super().__init__(name, input_queue)
        self.seen: list[str] = []

    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> None:
        self.seen.append(message.payload)
        return None


class ListSource:
    def __init__(self, items: list[str]) -> None:
        self.items = items

    async def poll(self) -> str | None:
        if not self.items:
            return None
        return self.items.pop(0)


def test_queue_creation_and_duplicates() -> None:
    processor = AsyncProcessor()

    processor.create_queue("in")

    with pytest.raises(DuplicateQueueError):
        processor.create_queue("in")


def test_duplicate_module_and_input_queue_validation() -> None:
    processor = AsyncProcessor()
    processor.create_queue("in")
    processor.create_queue("other")
    processor.register_module(SinkModule("sink", "in"))

    with pytest.raises(DuplicateModuleError):
        processor.register_module(SinkModule("sink", "other"))

    with pytest.raises(DuplicateModuleError):
        processor.register_module(SinkModule("second", "in"))

    with pytest.raises(UnknownQueueError):
        processor.register_module(SinkModule("missing", "missing"))


def test_module_consumes_dedicated_queue_and_routes_output() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.create_queue("out")
        processor.register_module(UppercaseModule("upper", "in"))

        await processor.start()
        await processor.submit("in", "hello")

        result = await asyncio.wait_for(processor.queue("out").get(), timeout=1)
        assert result.payload == "HELLO"
        processor.queue("out").task_done()

        await processor.stop()

    asyncio.run(scenario())


def test_multiple_outputs_from_one_input() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.create_queue("out_a")
        processor.create_queue("out_b")
        processor.register_module(FanoutModule("fanout", "in"))

        await processor.start()
        await processor.submit("in", "event")

        result_a = await asyncio.wait_for(processor.queue("out_a").get(), timeout=1)
        result_b = await asyncio.wait_for(processor.queue("out_b").get(), timeout=1)

        assert result_a.payload == "event:a"
        assert result_b.payload == "event:b"

        processor.queue("out_a").task_done()
        processor.queue("out_b").task_done()
        await processor.stop()

    asyncio.run(scenario())


def test_graceful_shutdown_without_hanging() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        module = SinkModule("sink", "in")
        processor.register_module(module)

        await processor.start()
        await processor.submit("in", "one")
        await asyncio.wait_for(processor.queue("in").join(), timeout=1)
        await asyncio.wait_for(processor.stop(), timeout=1)

        assert module.seen == ["one"]

    asyncio.run(scenario())


def test_unknown_route_target_is_surfaced_by_wait() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.register_module(MissingRouteModule("bad-route", "in"))

        await processor.start()
        await processor.submit("in", "hello")

        with pytest.raises(UnknownQueueError):
            await asyncio.wait_for(processor.wait(), timeout=1)

    asyncio.run(scenario())

def test_loop_polls_source_and_submits_to_input_queue() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.create_queue("out")
        processor.register_module(UppercaseModule("upper", "in"))
        source = ListSource(["frame"])
        stop_event = asyncio.Event()
        runner = ProcessorLoop(
            processor,
            input_queue="in",
            source=source,
            poll_interval=0.001,
        )

        task = asyncio.create_task(runner.run(stop_event=stop_event))
        result = await asyncio.wait_for(processor.queue("out").get(), timeout=1)
        assert result.payload == "FRAME"
        processor.queue("out").task_done()

        stop_event.set()
        await asyncio.wait_for(task, timeout=1)
        assert processor.is_running is False

    asyncio.run(scenario())


def test_empty_loop_stops_cleanly_when_stop_event_is_set() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        stop_event = asyncio.Event()
        runner = ProcessorLoop(processor, poll_interval=0.001)

        task = asyncio.create_task(runner.run(stop_event=stop_event))
        await asyncio.sleep(0)
        assert processor.is_running is True

        stop_event.set()
        await asyncio.wait_for(task, timeout=1)
        assert processor.is_running is False

    asyncio.run(scenario())


def test_loop_surfaces_module_failures_and_stops_cleanly() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.register_module(MissingRouteModule("bad-route", "in"))
        source = ListSource(["hello"])
        runner = ProcessorLoop(
            processor,
            input_queue="in",
            source=source,
            poll_interval=0.001,
        )

        with pytest.raises(UnknownQueueError):
            await asyncio.wait_for(runner.run(stop_event=asyncio.Event()), timeout=1)

        assert processor.is_running is False

    asyncio.run(scenario())

@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is unavailable")
def test_signal_stopper_sets_stop_event_from_signal() -> None:
    async def scenario() -> None:
        async with SignalStopper(signals=(signal.SIGUSR1,)) as stop_event:
            os.kill(os.getpid(), signal.SIGUSR1)
            await asyncio.wait_for(stop_event.wait(), timeout=1)

    asyncio.run(scenario())

def test_color_formatter_colors_expected_levels() -> None:
    formatter = ColorFormatter('%(levelname)s:%(message)s', use_colors=True)

    for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
        record = logging.LogRecord(
            name='test',
            level=level,
            pathname=__file__,
            lineno=1,
            msg='message',
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert '\033[' in formatted
        assert logging.getLevelName(level) in formatted
        assert formatted.endswith('\033[0m:message')


def test_color_formatter_can_disable_colors() -> None:
    formatter = ColorFormatter('%(levelname)s:%(message)s', use_colors=False)
    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='message',
        args=(),
        exc_info=None,
    )

    assert formatter.format(record) == 'INFO:message'


def test_looping_video_source_reads_and_loops_test_video() -> None:
    async def scenario() -> None:
        source = LoopingVideoSource(TEST_VIDEO_PATH, realtime=False)
        try:
            first_frame = await source.poll()
            assert first_frame.frame_index == 0
            assert first_frame.loop_count == 0
            assert first_frame.image.shape[:2] == (source.height, source.width)

            looped_frame = first_frame
            for _ in range(source.frame_count):
                looped_frame = await source.poll()

            assert looped_frame.frame_index == 0
            assert looped_frame.loop_count == 1
        finally:
            source.close()

        assert source.is_open is False

    asyncio.run(scenario())


def test_frame_rate_logger_module_logs_processing_rate(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario() -> None:
        module = FrameRateLoggerModule(
            name="fps",
            input_queue="frames",
            log_interval_seconds=0,
        )
        frame = VideoFrame(
            image=object(),
            frame_index=0,
            timestamp_seconds=0.0,
            loop_count=0,
        )

        with caplog.at_level(logging.INFO, logger="src.modules.frame_rate_logger"):
            await module.process(Message(frame), AsyncProcessor())

        assert "Processing frame rate" in caplog.text
        assert "FPS" in caplog.text

    asyncio.run(scenario())


def make_test_image() -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 4
    image[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None] * 4
    image[:, :, 2] = 80
    return image


def test_image_enhancement_module_enhances_raw_bgr_image() -> None:
    async def scenario() -> None:
        image = make_test_image()
        module = ImageEnhancementModule(
            name="enhancer",
            input_queue="frames",
            output_queue="enhanced_frames",
        )

        routed = await module.process(Message(image), AsyncProcessor())
        enhanced = routed.message.payload

        assert routed.destination == "enhanced_frames"
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == image.shape
        assert enhanced.dtype == image.dtype
        assert not np.array_equal(enhanced, image)

    asyncio.run(scenario())


def test_image_enhancement_module_preserves_video_frame_metadata() -> None:
    async def scenario() -> None:
        image = make_test_image()
        frame = VideoFrame(
            image=image,
            frame_index=42,
            timestamp_seconds=1.25,
            loop_count=3,
        )
        module = ImageEnhancementModule(
            name="enhancer",
            input_queue="frames",
            output_queue="enhanced_frames",
        )

        routed = await module.process(Message(frame, metadata={"camera": "test"}), AsyncProcessor())
        enhanced_frame = routed.message.payload

        assert isinstance(enhanced_frame, VideoFrame)
        assert enhanced_frame.frame_index == frame.frame_index
        assert enhanced_frame.timestamp_seconds == frame.timestamp_seconds
        assert enhanced_frame.loop_count == frame.loop_count
        assert enhanced_frame.image.shape == image.shape
        assert enhanced_frame.image.dtype == image.dtype
        assert not np.array_equal(enhanced_frame.image, image)
        assert routed.message.metadata == {"camera": "test"}

    asyncio.run(scenario())


def test_image_enhancement_module_routes_output_to_configured_queue() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("frames")
        processor.create_queue("enhanced_frames")
        processor.register_module(
            ImageEnhancementModule(
                name="enhancer",
                input_queue="frames",
                output_queue="enhanced_frames",
            )
        )

        await processor.start()
        await processor.submit("frames", make_test_image())

        result = await asyncio.wait_for(
            processor.queue("enhanced_frames").get(),
            timeout=1,
        )
        assert isinstance(result.payload, np.ndarray)
        assert result.payload.shape == (32, 32, 3)
        processor.queue("enhanced_frames").task_done()
        await processor.stop()

    asyncio.run(scenario())


def test_queue_fanout_module_routes_message_to_all_output_queues() -> None:
    async def scenario() -> None:
        module = QueueFanoutModule(
            name="fanout",
            input_queue="in",
            output_queues=["out_a", "out_b"],
        )
        message = Message("payload", metadata={"source": "test"})

        routed = await module.process(message, AsyncProcessor())

        assert [item.destination for item in routed] == ["out_a", "out_b"]
        assert all(item.message is message for item in routed)

    asyncio.run(scenario())


def test_gmm_color_mask_module_outputs_binary_mask(tmp_path: Path) -> None:
    async def scenario() -> None:
        query_bgr = (10, 200, 20)
        model_path = tmp_path / "color_classifier_gmm.joblib"
        write_synthetic_gmm_model(model_path, query_bgr)
        image = np.full((20, 24, 3), query_bgr, dtype=np.uint8)
        module = GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=model_path,
        )

        routed = await module.process(Message(image), AsyncProcessor())

        assert routed.destination == "masks"
        assert routed.message.payload.shape == (20, 24)
        assert routed.message.payload.dtype == np.uint8
        assert set(np.unique(routed.message.payload)).issubset({0, 255})
        assert np.any(routed.message.payload == 255)

    asyncio.run(scenario())


def test_gmm_color_mask_module_writes_debug_mask_when_enabled(tmp_path: Path) -> None:
    async def scenario() -> None:
        query_bgr = (10, 200, 20)
        model_path = tmp_path / "color_classifier_gmm.joblib"
        debug_dir = tmp_path / "debug"
        write_synthetic_gmm_model(model_path, query_bgr)
        image = np.full((20, 24, 3), query_bgr, dtype=np.uint8)
        module = GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=model_path,
            debug=True,
            debug_dir=debug_dir,
        )

        routed = await module.process(Message(image), AsyncProcessor())

        debug_mask_path = debug_dir / "gmm_color_mask.png"
        assert debug_mask_path.exists()
        debug_mask = cv2.imread(str(debug_mask_path), cv2.IMREAD_GRAYSCALE)
        assert debug_mask is not None
        assert debug_mask.shape == routed.message.payload.shape
        assert debug_mask.dtype == np.uint8

    asyncio.run(scenario())


def test_gmm_color_mask_module_preserves_video_frame_metadata(tmp_path: Path) -> None:
    async def scenario() -> None:
        query_bgr = (10, 200, 20)
        model_path = tmp_path / "color_classifier_gmm.joblib"
        write_synthetic_gmm_model(model_path, query_bgr)
        frame = VideoFrame(
            image=np.full((12, 14, 3), query_bgr, dtype=np.uint8),
            frame_index=42,
            timestamp_seconds=1.68,
            loop_count=3,
        )
        module = GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=model_path,
        )

        routed = await module.process(Message(frame, metadata={"source": "test"}), AsyncProcessor())

        assert routed.message.payload.shape == (12, 14)
        assert routed.message.metadata["source"] == "test"
        assert routed.message.metadata["frame_index"] == 42
        assert routed.message.metadata["timestamp_seconds"] == 1.68
        assert routed.message.metadata["loop_count"] == 3

    asyncio.run(scenario())


def test_gmm_color_mask_module_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="GMM color classifier model not found"):
        GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=tmp_path / "missing.joblib",
        )


def make_synthetic_marker_image() -> np.ndarray:
    image = np.full((360, 480, 3), 230, dtype=np.uint8)
    marker_quad = np.array(
        [[95, 70], [380, 95], [350, 315], [120, 290]],
        dtype=np.int32,
    )
    inner_quad = np.array(
        [[150, 125], [320, 135], [305, 245], [160, 240]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, marker_quad, (20, 20, 20))
    cv2.fillConvexPoly(image, inner_quad, (240, 240, 240))
    cv2.polylines(image, [marker_quad], True, (0, 0, 0), 8, cv2.LINE_AA)
    return image


def marker_debug_paths(debug_dir: Path) -> list[Path]:
    return [
        debug_dir / "marker_input.png",
        debug_dir / "marker_hough_lines.png",
        debug_dir / "marker_detected_quad.png",
        debug_dir / "marker_rectified_cutout.png",
    ]


def test_marker_candidate_refinement_clips_initial_guess_to_bounds() -> None:
    dist = np.zeros((100, 120), dtype=np.float32)
    initial_quad = np.array(
        [[-6, 10], [126, -3], [118, 108], [4, 96]],
        dtype=np.float32,
    )

    refined = refine_candidate(
        initial_quad,
        dist,
        width=120,
        height=100,
        min_area=400.0,
    )

    assert refined.shape == (4, 2)
    assert np.all(refined[:, 0] >= 0)
    assert np.all(refined[:, 0] <= 119)
    assert np.all(refined[:, 1] >= 0)
    assert np.all(refined[:, 1] <= 99)


def test_marker_rectification_debug_disabled_does_not_create_debug_files(tmp_path: Path) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=False,
            debug_dir=debug_dir,
        )

        routed = await module.process(Message(make_synthetic_marker_image()), AsyncProcessor())

        assert routed is not None
        assert not debug_dir.exists()

    asyncio.run(scenario())


def test_marker_rectification_debug_enabled_writes_processing_images(tmp_path: Path) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=True,
            debug_dir=debug_dir,
        )

        routed = await module.process(Message(make_synthetic_marker_image()), AsyncProcessor())

        assert routed is not None
        for debug_path in marker_debug_paths(debug_dir):
            assert debug_path.exists()
            assert cv2.imread(str(debug_path)) is not None
        assert cv2.imread(str(debug_dir / "marker_input.png")).shape == (360, 480, 3)
        assert cv2.imread(str(debug_dir / "marker_rectified_cutout.png")).shape == (512, 512, 3)

    asyncio.run(scenario())


def test_marker_rectification_debug_enabled_writes_failure_images(tmp_path: Path) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=True,
            debug_dir=debug_dir,
        )
        blank = np.full((240, 320, 3), 127, dtype=np.uint8)

        routed = await module.process(Message(blank), AsyncProcessor())

        assert routed is None
        for debug_path in marker_debug_paths(debug_dir):
            assert debug_path.exists()
            assert cv2.imread(str(debug_path)) is not None
        cutout = cv2.imread(str(debug_dir / "marker_rectified_cutout.png"))
        assert cutout.shape == (512, 512, 3)
        assert int(np.count_nonzero(cutout)) == 0

    asyncio.run(scenario())


def test_main_uses_direct_marker_path_when_gmm_model_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class SpyMarkerRectificationModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["marker_input_queue"] = input_queue
            captured["marker_output_queue"] = output_queue
            captured["marker_kwargs"] = kwargs

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class NoopProcessorLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run_until_interrupted(self) -> None:
            return None

    missing_model_path = tmp_path / "missing.joblib"
    monkeypatch.setattr(app_main, "GMM_MODEL_PATH", missing_model_path)
    monkeypatch.setattr(app_main, "MarkerRectificationModule", SpyMarkerRectificationModule)
    monkeypatch.setattr(app_main, "ProcessorLoop", NoopProcessorLoop)
    args = app_main.parse_args(["--video-path", str(TEST_VIDEO_PATH)])

    asyncio.run(app_main.run_app(args))

    assert captured["marker_input_queue"] == app_main.ENHANCED_FRAME_QUEUE
    assert captured["marker_output_queue"] == app_main.MARKER_CUTOUT_QUEUE


def test_main_registers_gmm_fanout_path_when_model_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class SpyMarkerRectificationModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["marker_input_queue"] = input_queue
            captured["marker_output_queue"] = output_queue

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class SpyGMMColorMaskModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["gmm_input_queue"] = input_queue
            captured["gmm_output_queue"] = output_queue
            captured["gmm_kwargs"] = kwargs

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class SpyQueueFanoutModule(BaseModule[np.ndarray]):
        def __init__(self, name: str, input_queue: str, output_queues: list[str]) -> None:
            super().__init__(name, input_queue)
            captured["fanout_input_queue"] = input_queue
            captured["fanout_output_queues"] = output_queues

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class NoopProcessorLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run_until_interrupted(self) -> None:
            return None

    model_path = tmp_path / "color_classifier_gmm.joblib"
    model_path.write_bytes(b"exists")
    monkeypatch.setattr(app_main, "GMM_MODEL_PATH", model_path)
    monkeypatch.setattr(app_main, "MarkerRectificationModule", SpyMarkerRectificationModule)
    monkeypatch.setattr(app_main, "GMMColorMaskModule", SpyGMMColorMaskModule)
    monkeypatch.setattr(app_main, "QueueFanoutModule", SpyQueueFanoutModule)
    monkeypatch.setattr(app_main, "ProcessorLoop", NoopProcessorLoop)
    args = app_main.parse_args(["--video-path", str(TEST_VIDEO_PATH)])

    asyncio.run(app_main.run_app(args))

    assert captured["fanout_input_queue"] == app_main.ENHANCED_FRAME_QUEUE
    assert captured["fanout_output_queues"] == [app_main.MARKER_FRAME_QUEUE, app_main.GMM_FRAME_QUEUE]
    assert captured["marker_input_queue"] == app_main.MARKER_FRAME_QUEUE
    assert captured["marker_output_queue"] == app_main.MARKER_CUTOUT_QUEUE
    assert captured["gmm_input_queue"] == app_main.GMM_FRAME_QUEUE
    assert captured["gmm_output_queue"] == app_main.COLOR_MASK_QUEUE
    assert captured["gmm_kwargs"] == {"model_path": model_path, "debug": False, "debug_dir": Path("data/debug")}


def test_main_debug_flag_is_parsed_and_wired_to_marker_rectifier(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class SpyMarkerRectificationModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["output_queue"] = output_queue
            captured.update(kwargs)

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class NoopProcessorLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run_until_interrupted(self) -> None:
            return None

    monkeypatch.setattr(app_main, "MarkerRectificationModule", SpyMarkerRectificationModule)
    monkeypatch.setattr(app_main, "ProcessorLoop", NoopProcessorLoop)
    args = app_main.parse_args(["--debug", "--video-path", str(TEST_VIDEO_PATH)])

    assert args.debug is True
    asyncio.run(app_main.run_app(args))

    assert captured["output_queue"] == app_main.MARKER_CUTOUT_QUEUE
    assert captured["debug"] is True
    assert captured["debug_dir"] == Path("data/debug")


def test_marker_rectification_module_outputs_rectified_cutout() -> None:
    async def scenario() -> None:
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )

        routed = await module.process(Message(make_synthetic_marker_image()), AsyncProcessor())

        assert routed is not None
        assert routed.destination == "cutouts"
        assert routed.message.payload.shape == (512, 512, 3)
        assert routed.message.payload.dtype == np.uint8
        assert "quad" in routed.message.metadata
        assert "score" in routed.message.metadata
        assert routed.message.metadata["input_shape"] == (360, 480, 3)

    asyncio.run(scenario())


def test_marker_rectification_module_preserves_video_frame_metadata() -> None:
    async def scenario() -> None:
        frame = VideoFrame(
            image=make_synthetic_marker_image(),
            frame_index=12,
            timestamp_seconds=0.48,
            loop_count=2,
        )
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )

        routed = await module.process(Message(frame, metadata={"source": "test"}), AsyncProcessor())

        assert routed is not None
        assert routed.message.payload.shape == (512, 512, 3)
        assert routed.message.metadata["source"] == "test"
        assert routed.message.metadata["frame_index"] == 12
        assert routed.message.metadata["timestamp_seconds"] == 0.48
        assert routed.message.metadata["loop_count"] == 2

    asyncio.run(scenario())


def test_marker_rectification_module_drops_frame_without_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )
        blank = np.full((240, 320, 3), 127, dtype=np.uint8)

        with caplog.at_level(logging.WARNING, logger="src.modules.marker_rectifier"):
            routed = await module.process(Message(blank), AsyncProcessor())

        assert routed is None
        assert "Dropping frame without detected marker" in caplog.text

    asyncio.run(scenario())


def test_marker_rectification_module_routes_cutout_to_configured_queue() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("frames")
        processor.create_queue("cutouts")
        processor.register_module(
            MarkerRectificationModule(
                name="rectifier",
                input_queue="frames",
                output_queue="cutouts",
            )
        )

        await processor.start()
        await processor.submit("frames", make_synthetic_marker_image())

        result = await asyncio.wait_for(processor.queue("cutouts").get(), timeout=2)
        assert result.payload.shape == (512, 512, 3)
        assert result.payload.dtype == np.uint8
        assert "quad" in result.metadata
        processor.queue("cutouts").task_done()
        await processor.stop()

    asyncio.run(scenario())


def test_frame_rate_logger_module_uses_loop_count_metadata_for_cutouts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        module = FrameRateLoggerModule(
            name="fps",
            input_queue="marker_cutouts",
            log_interval_seconds=0,
        )
        cutout = np.zeros((32, 32, 3), dtype=np.uint8)

        with caplog.at_level(logging.INFO, logger="src.modules.frame_rate_logger"):
            await module.process(Message(cutout, metadata={"loop_count": 7}), AsyncProcessor())

        assert "Processing frame rate" in caplog.text
        assert "source loop 7" in caplog.text

    asyncio.run(scenario())
