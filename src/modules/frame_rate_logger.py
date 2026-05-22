from __future__ import annotations

import logging
import time
from typing import Any

from ..messages import Message
from ..video import VideoFrame
from .base import BaseModule, ModuleContext

logger = logging.getLogger(__name__)


class FrameRateLoggerModule(BaseModule[Any]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        *,
        log_interval_seconds: float = 1.0,
    ) -> None:
        if log_interval_seconds < 0:
            raise ValueError("log_interval_seconds cannot be negative.")

        super().__init__(name=name, input_queue=input_queue)
        self.log_interval_seconds = log_interval_seconds
        self.total_frames = 0
        self._frames_since_log = 0
        self._last_log_at = time.monotonic()

    async def process(
        self,
        message: Message[Any],
        context: ModuleContext,
    ) -> None:
        self.total_frames += 1
        self._frames_since_log += 1

        now = time.monotonic()
        elapsed = now - self._last_log_at
        if elapsed < self.log_interval_seconds:
            return None

        fps = self._frames_since_log / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Processing frame rate: %.2f FPS (%s total item(s), source loop %s)",
            fps,
            self.total_frames,
            self._loop_count(message),
        )
        self._frames_since_log = 0
        self._last_log_at = now
        return None

    def _loop_count(self, message: Message[Any]) -> int | str:
        if isinstance(message.payload, VideoFrame):
            return message.payload.loop_count
        return message.metadata.get("loop_count", "unknown")
