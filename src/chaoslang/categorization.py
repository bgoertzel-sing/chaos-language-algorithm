"""Category induction extension seam.

Hard/disjoint context categories are intentionally deferred until the chunk-only
MVP invariants are solid. Implementers should return typed proposals rather than
mutating GrammarState directly.
"""
from __future__ import annotations

from typing import Iterable, Protocol

from .core.types import CategoryProposal, GrammarState


class CategoryInducer(Protocol):
    def propose_categories(self, state: GrammarState) -> Iterable[CategoryProposal]: ...


class NoOpCategoryInducer:
    def propose_categories(self, state: GrammarState) -> tuple[CategoryProposal, ...]:
        return ()
