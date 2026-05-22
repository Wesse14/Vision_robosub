from __future__ import annotations

import asyncio
import inspect
import logging
import signal
from collections.abc import Sequence
from types import FrameType
from typing import Any, Protocol

from .messages import Message
from .processor import AsyncProcessor, ProcessorError

logger = logging.getLogger(__name__)


class InputSource(Protocol):
    async def poll(self) -> Message[Any] | Any | None:
        ...


class EmptyInputSource:
    async def poll(self) -> None:
        return None


class SignalStopper:
    def __init__(
        self,
        signals: Sequence[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
    ) -> None:
        self._signals = tuple(signals)
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._asyncio_handlers: list[signal.Signals] = []
        self._previous_handlers: dict[
            signal.Signals,
            signal.Handlers | int | None,
        ] = {}

    async def __aenter__(self) -> asyncio.Event:
        self._event = asyncio.Event()
        self._loop = asyncio.get_running_loop()

        for signum in self._signals:
            logger.debug("Registering stop handler for signal %s", signum.name)
            try:
                self._loop.add_signal_handler(
                    signum,
                    self._handle_signal,
                    signum,
                    None,
                )
            except (NotImplementedError, RuntimeError, ValueError):
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            else:
                self._asyncio_handlers.append(signum)

        return self._event

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if self._loop is not None:
            for signum in self._asyncio_handlers:
                self._loop.remove_signal_handler(signum)

        for signum, previous_handler in self._previous_handlers.items():
            signal.signal(signum, previous_handler)

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        logger.warning(
            "Received signal %s; stopping processor loop",
            signal.Signals(signum).name,
        )
        if self._event is None:
            return
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._event.set)
            return
        self._event.set()


class ProcessorLoop:
    def __init__(
        self,
        processor: AsyncProcessor,
        *,
        input_queue: str | None = None,
        source: InputSource | None = None,
        poll_interval: float = 0.01,
    ) -> None:
        if source is not None and input_queue is None:
            raise ValueError('input_queue is required when a source is configured.')
        if poll_interval < 0:
            raise ValueError('poll_interval cannot be negative.')

        self.processor = processor
        self.input_queue = input_queue
        self.source = source
        self.poll_interval = poll_interval

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        active_stop_event = stop_event or asyncio.Event()
        captured_error: BaseException | None = None

        logger.info("Processor loop starting")
        await self.processor.start()
        try:
            while not active_stop_event.is_set():
                self.processor.raise_for_failed_tasks()

                if self.source is None:
                    await self._wait_for_next_poll(active_stop_event)
                    continue

                item = await self.source.poll()
                self.processor.raise_for_failed_tasks()

                if item is None:
                    await self._wait_for_next_poll(active_stop_event)
                    continue

                logger.debug("Submitting polled item to queue %s", self.input_queue)
                await self._submit_item(item)
        except BaseException as exc:
            captured_error = exc
            logger.exception("Processor loop failed")
            raise
        finally:
            logger.info("Processor loop stopping")
            try:
                await self._close_source()
                await self.processor.stop()
                logger.info("Processor loop stopped cleanly")
            except Exception:
                if captured_error is None:
                    raise

    async def run_until_interrupted(
        self,
        signals: Sequence[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
    ) -> None:
        logger.info(
            "Processor loop waiting for interrupt signal(s): %s",
            ", ".join(signum.name for signum in signals),
        )
        async with SignalStopper(signals=signals) as stop_event:
            await self.run(stop_event=stop_event)

    async def _submit_item(self, item: Message[Any] | Any) -> None:
        if self.input_queue is None:
            raise ProcessorError('input_queue is required to submit polled items.')

        if isinstance(item, Message):
            await self.processor.put(self.input_queue, item)
            return

        await self.processor.submit(self.input_queue, item)

    async def _wait_for_next_poll(self, stop_event: asyncio.Event) -> None:
        if self.poll_interval == 0:
            await asyncio.sleep(0)
            return

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval)
        except TimeoutError:
            return

    async def _close_source(self) -> None:
        if self.source is None:
            return

        close = getattr(self.source, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result
