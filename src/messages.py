from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Message(Generic[T]):
    payload: T
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoutedMessage(Generic[T]):
    destination: str
    message: Message[T]

    @classmethod
    def from_payload(
        cls,
        destination: str,
        payload: T,
        metadata: Mapping[str, Any] | None = None,
    ) -> RoutedMessage[T]:
        return cls(
            destination=destination,
            message=Message(payload=payload, metadata=metadata or {}),
        )
