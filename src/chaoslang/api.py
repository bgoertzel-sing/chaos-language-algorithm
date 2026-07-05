"""Public API for the pure-Python symbolic-string CLA MVP."""
from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Iterable, Sequence

from .categorization.context import ContextCategoryInducer
from .categorization.js_inducer import JSCategoryInducer
from .core.edits import EditApplier
from .core.types import GrammarState, ParseEntry, Token, expand_parse
from .mining.ngram import NGramPatternMiner
from .scoring import SimpleMDLScorer


@dataclass(frozen=True)
class CLAModel:
    """A fitted CLA model with exact expansion back to the input stream."""

    state: GrammarState
    scorer: SimpleMDLScorer

    @property
    def score(self):
        return self.state.score

    @property
    def grammar(self):
        return self.state.grammar

    @property
    def parse(self) -> tuple[ParseEntry, ...]:
        return self.state.parse

    def expand_tokens(self) -> tuple[Token, ...]:
        return expand_parse(self.state.parse, self.state.grammar)

    def expand(self) -> tuple[str, ...]:
        return tuple(token.value for token in self.expand_tokens())


class CLA:
    """Greedy symbolic-string Chaos Language Algorithm learner."""

    def __init__(
        self,
        *,
        max_iterations: int = 8,
        seed: int = 0,
        n_min: int = 2,
        n_max: int = 5,
        max_ngram: int | None = None,
        min_uses: int = 2,
        enable_categories: bool = True,
        category_method: str = "exact",
        js_threshold: float = 0.1,
    ) -> None:
        self.max_iterations = max_iterations
        self.seed = seed
        self.enable_categories = enable_categories
        self._rng = random.Random(seed)
        if max_ngram is not None:
            n_max = max_ngram
        self.miner = NGramPatternMiner(n_min=n_min, n_max=n_max, min_uses=min_uses)
        self.applier = EditApplier()
        self.scorer = SimpleMDLScorer()

        if category_method == "js":
            self.category_inducer = JSCategoryInducer(threshold=js_threshold)
        elif category_method == "exact":
            self.category_inducer = ContextCategoryInducer()
        else:
            raise ValueError(f"unknown category_method: {category_method!r}; use 'exact' or 'js'")

    @classmethod
    def simple(cls, max_iterations: int = 8, seed: int = 0, **kwargs: object) -> "CLA":
        """Return the dependency-free deterministic MVP configuration."""

        return cls(max_iterations=max_iterations, seed=seed, **kwargs)

    def fit_symbols(self, symbols: str | Iterable[str | Token]) -> CLAModel:
        """Fit a grammar while preserving exact reconstruction.

        Compact strings are character streams; strings containing whitespace are
        tokenized with ``split()`` for symbolic examples. Pass an iterable for
        explicit pre-tokenized symbols.
        """

        stream: Sequence[str | Token]
        if isinstance(symbols, str):
            stream = tuple(symbols.split()) if any(ch.isspace() for ch in symbols) else tuple(symbols)
        else:
            stream = tuple(symbols)
        state = GrammarState.initial(stream)
        state = state.with_score(self.scorer.score(state))

        for _ in range(self.max_iterations):
            old_score = self.scorer.score(state)
            proposals = list(self.miner.proposals(state))
            if self.enable_categories:
                proposals.extend(self.category_inducer.proposals(state))

            best_state: GrammarState | None = None
            best_score = old_score
            best_key: tuple[str, str] | None = None
            for proposal in proposals:
                candidate = self.applier.apply(state, proposal)
                candidate_score = self.scorer.score(candidate)
                candidate = replace(candidate, score=candidate_score)
                key = (proposal.__class__.__name__, repr(proposal))
                if candidate_score.total < best_score.total - 1e-12 or (
                    abs(candidate_score.total - best_score.total) <= 1e-12
                    and best_state is not None
                    and best_key is not None
                    and key < best_key
                ):
                    best_state = candidate
                    best_score = candidate_score
                    best_key = key

            if best_state is None:
                state = state.with_score(old_score)
                break
            assert expand_parse(best_state.parse, best_state.grammar) == best_state.corpus.symbols
            state = best_state

        state = state.with_score(self.scorer.score(state))
        assert expand_parse(state.parse, state.grammar) == state.corpus.symbols
        return CLAModel(state, self.scorer)
