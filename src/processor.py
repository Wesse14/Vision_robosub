from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any, cast

from .messages import Message, RoutedMessage
from .modules import BaseModule, ModuleOutput

logger = logging.getLogger(__name__)


class ProcessorError(RuntimeError):
    pass


class DuplicateQueueError(ProcessorError, ValueError):
    pass


class UnknownQueueError(ProcessorError, KeyError):
    pass


class DuplicateModuleError(ProcessorError, ValueError):
    pass


_STOP = object()


class AsyncProcessor:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Any]] = {}
        self._modules: dict[str, BaseModule[Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._running = False
        self._stopping = False

    @property
    def is_running(self) -> bool:
        return self._running

    def raise_for_failed_tasks(self) -> None:
        for task in self._tasks.values():
            if not task.done():
                continue
            exception = task.exception()
            if exception is not None:
                raise exception

    def create_queue(
        self,
        name: str,
        *,
        maxsize: int = 0,
    ) -> asyncio.Queue[Message[Any]]:
        if not name:
            raise ValueError("Queue name cannot be empty.")
        if name in self._queues:
            raise DuplicateQueueError(f"Queue already exists: {name}")

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._queues[name] = queue
        logger.debug("Created queue %s with maxsize=%s", name, maxsize)
        return cast(asyncio.Queue[Message[Any]], queue)

    def queue(self, name: str) -> asyncio.Queue[Message[Any]]:
        try:
            return cast(asyncio.Queue[Message[Any]], self._queues[name])
        except KeyError as exc:
            raise UnknownQueueError(f"Unknown queue: {name}") from exc

    def register_module(self, module: BaseModule[Any]) -> None:
        if self._running:
            raise ProcessorError("Cannot register modules while processor is running.")
        if module.name in self._modules:
            raise DuplicateModuleError(f"Module already exists: {module.name}")
        if module.input_queue not in self._queues:
            raise UnknownQueueError(f"Unknown input queue: {module.input_queue}")
        if any(
            existing.input_queue == module.input_queue
            for existing in self._modules.values()
        ):
            raise DuplicateModuleError(
                f"Queue is already assigned to another module: {module.input_queue}"
            )

        self._modules[module.name] = module
        logger.info(
            "Registered module %s on input queue %s",
            module.name,
            module.input_queue,
        )

    async def submit(
        self,
        queue_name: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.put(
            queue_name,
            Message(payload=payload, metadata=metadata or {}),
        )

    async def put(self, queue_name: str, message: Message[Any]) -> None:
        await self.queue(queue_name).put(message)

    async def start(self) -> None:
        if self._running:
            raise ProcessorError("Processor is already running.")

        self._running = True
        self._stopping = False
        logger.info("Starting processor with %s module(s)", len(self._modules))
        self._tasks = {
            module.name: asyncio.create_task(
                self._run_module(module),
                name=f"async-processor:{module.name}",
            )
            for module in self._modules.values()
        }

    async def stop(self) -> None:
        if not self._running and not self._tasks:
            return

        logger.info("Stopping processor")
        await self._send_stop_signals()
        await self._collect_tasks(raise_errors=True)

    async def wait(self) -> None:
        await self._collect_tasks(raise_errors=True)

    async def __aenter__(self) -> AsyncProcessor:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.stop()

    async def _run_module(self, module: BaseModule[Any]) -> None:
        queue = self._queues[module.input_queue]
        logger.debug("Module task started: %s", module.name)

        while True:
            item = await queue.get()
            try:
                if item is _STOP:
                    logger.debug("Module task stopping: %s", module.name)
                    return

                result = await module.process(item, self)
                await self._route_outputs(result)
            except Exception:
                logger.exception("Module failed: %s", module.name)
                await self._send_stop_signals(excluding=module.name)
                raise
            finally:
                queue.task_done()

    async def _route_outputs(self, result: ModuleOutput) -> None:
        for routed_message in self._normalize_outputs(result):
            if routed_message.destination not in self._queues:
                logger.error(
                    "Cannot route message to unknown queue %s",
                    routed_message.destination,
                )
                raise UnknownQueueError(
                    f"Unknown destination queue: {routed_message.destination}"
                )
            logger.debug("Routing message to queue %s", routed_message.destination)
            await self._queues[routed_message.destination].put(routed_message.message)

    def _normalize_outputs(self, result: ModuleOutput) -> list[RoutedMessage[Any]]:
        if result is None:
            return []
        if isinstance(result, RoutedMessage):
            return [result]
        if isinstance(result, Iterable):
            outputs = list(result)
            if not all(isinstance(output, RoutedMessage) for output in outputs):
                raise TypeError("Module outputs must be RoutedMessage instances.")
            return outputs

        raise TypeError("Module output must be None, RoutedMessage, or an iterable.")

    async def _send_stop_signals(self, excluding: str | None = None) -> None:
        if self._stopping:
            return

        self._stopping = True
        for module in self._modules.values():
            if module.name == excluding:
                continue
            task = self._tasks.get(module.name)
            if task is not None and task.done():
                continue
            logger.debug("Sending stop signal to module %s", module.name)
            await self._queues[module.input_queue].put(_STOP)

    async def _collect_tasks(self, *, raise_errors: bool) -> None:
        if not self._tasks:
            self._running = False
            return

        tasks = list(self._tasks.values())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._running = False
        self._stopping = False
        logger.info("Processor stopped")

        if raise_errors:
            for result in results:
                if isinstance(result, BaseException):
                    raise result
