from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..messages import Message, RoutedMessage
from .base import BaseModule, ModuleContext


class QueueFanoutModule(BaseModule[Any]):
    def __init__(self, name: str, input_queue: str, output_queues: Iterable[str]) -> None:
        super().__init__(name=name, input_queue=input_queue)
        self.output_queues = tuple(output_queues)
        if not self.output_queues:
            raise ValueError("Module output_queues cannot be empty.")
        if any(not output_queue for output_queue in self.output_queues):
            raise ValueError("Module output_queues cannot contain empty queue names.")

    async def process(
        self,
        message: Message[Any],
        context: ModuleContext,
    ) -> list[RoutedMessage[Any]]:
        return [
            RoutedMessage(destination=output_queue, message=message)
            for output_queue in self.output_queues
        ]
