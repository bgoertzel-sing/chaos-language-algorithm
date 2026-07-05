"""Category induction and clustering seams."""
from typing import Iterable, Protocol

from chaoslang.core.types import CategoryProposal, GrammarState
from .context import ContextCategoryInducer
from .js import cluster_by_js, js_divergence
from .js_inducer import JSCategoryInducer


class CategoryInducer(Protocol):
    def propose_categories(self, state: GrammarState) -> Iterable[CategoryProposal]: ...


class NoOpCategoryInducer:
    def propose_categories(self, state: GrammarState) -> tuple[CategoryProposal, ...]:
        return ()


__all__ = [
    "CategoryInducer",
    "ContextCategoryInducer",
    "JSCategoryInducer",
    "NoOpCategoryInducer",
    "cluster_by_js",
    "js_divergence",
]
