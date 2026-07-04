"""Category induction extension seams."""
from typing import Iterable, Protocol

from chaoslang.core.types import CategoryProposal, GrammarState
from .context import ContextCategoryInducer


class CategoryInducer(Protocol):
    def propose_categories(self, state: GrammarState) -> Iterable[CategoryProposal]: ...


class NoOpCategoryInducer:
    def propose_categories(self, state: GrammarState) -> tuple[CategoryProposal, ...]:
        return ()


__all__ = ["CategoryInducer", "ContextCategoryInducer", "NoOpCategoryInducer"]
