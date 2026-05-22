from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


class VideoSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VideoFrame:
    image: Any
    frame_index: int
    timestamp_seconds: float
    loop_count: int


class LoopingVideoSource:
    def __init__(
        self,
        path: str | Path,
        *,
        realtime: bool = True,
        target_fps: float | None = None,
    ) -> None:
        if target_fps is not None and target_fps <= 0:
            raise ValueError('target_fps must be greater than zero.')

        self.path = Path(path)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise VideoSourceError(f'Could not open video file: {self.path}')

        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.source_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.loop_count = 0

        effective_fps = target_fps if target_fps is not None else self.source_fps
        self._frame_interval = (
            1.0 / effective_fps if realtime and effective_fps > 0 else None
        )
        self._last_frame_at: float | None = None

        logger.info(
            'Opened video %s (%sx%s, %s frame(s), %.2f FPS)',
            self.path,
            self.width,
            self.height,
            self.frame_count,
            self.source_fps,
        )

    @property
    def is_open(self) -> bool:
        return self._capture.isOpened()

    async def poll(self) -> VideoFrame:
        await self._pace()

        ok, frame = self._capture.read()
        if not ok:
            self.loop_count += 1
            logger.info('Video ended; looping %s from the first frame', self.path)
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok:
                raise VideoSourceError(
                    f'Could not read frame from video file: {self.path}'
                )

        position = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        frame_index = max(0, position - 1)
        timestamp_seconds = (
            float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        )

        return VideoFrame(
            image=frame,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            loop_count=self.loop_count,
        )

    def close(self) -> None:
        if self._capture.isOpened():
            self._capture.release()
            logger.info('Closed video %s', self.path)

    async def _pace(self) -> None:
        if self._frame_interval is None:
            return

        now = time.monotonic()
        if self._last_frame_at is not None:
            elapsed = now - self._last_frame_at
            delay = self._frame_interval - elapsed
            if delay > 0:
                await asyncio.sleep(delay)

        self._last_frame_at = time.monotonic()
