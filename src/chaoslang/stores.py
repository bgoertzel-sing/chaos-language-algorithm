"""Backend-neutral fact layer skeletons for later Hyperon migration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: tuple[object, ...]
    truth: float = 1.0
    attrs: Mapping[str, object] = field(default_factory=dict)


class FactStore(Protocol):
    def add(self, fact: Fact) -> None: ...

    def query(self, predicate: str | None = None) -> Iterable[Fact]: ...


class MemoryFactStore:
    """Tiny in-memory store; richer projection/query support is a TODO."""

    def __init__(self) -> None:
        self._facts: list[Fact] = []

    def add(self, fact: Fact) -> None:
        self._facts.append(fact)

    def query(self, predicate: str | None = None) -> tuple[Fact, ...]:
        if predicate is None:
            return tuple(self._facts)
        return tuple(f for f in self._facts if f.predicate == predicate)
