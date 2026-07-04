"""Approximate deterministic two-part MDL scorers."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from chaoslang.core.edits import EditApplier
from chaoslang.core.types import CategoryOccurrence, GrammarState, ParseEntry, Score, entry_key


@dataclass(frozen=True)
class CodeLengthConfig:
    rule_overhead_bits: float = 0.8
    category_overhead_bits: float = 14.0
    symbol_floor_bits: float = 1.0


class SimpleMDLScorer:
    """Small MDL-like score for strict improvement decisions.

    This sprint-1 proxy deliberately uses token-count units. It rewards shorter
    parses, charges explicit overhead for chunk productions/categories, and keeps
    category occurrences exact via member side-channel cost.
    """

    def __init__(self, rule_overhead: float = 0.8, category_overhead: float = 14.0) -> None:
        self.rule_overhead = rule_overhead
        self.category_overhead = category_overhead

    def score(self, state: GrammarState) -> Score:
        model = 0.0
        for prod in state.grammar.productions.values():
            model += self.rule_overhead + len(prod.rhs)
        for cat in state.grammar.categories.values():
            model += self.category_overhead + len(cat.members)
        # Pruned speculative edits still have an edit-log cost. This makes
        # one-use chunk proposals strictly worse than doing nothing.
        model += 0.1 * len(getattr(state, "edit_log", ()))
        data = 0.0
        cat_counts = Counter()
        cat_member_counts = Counter()
        for entry in state.parse:
            if isinstance(entry, CategoryOccurrence):
                cat_counts[entry.category] += 1
                cat_member_counts[(entry.category, entry.member)] += 1
        for entry in state.parse:
            if isinstance(entry, CategoryOccurrence):
                p = cat_member_counts[(entry.category, entry.member)] / cat_counts[entry.category]
                data += 1.0 - math.log2(max(p, 1e-12))
            else:
                data += 1.0
        return Score(model_bits=model, data_bits=data)


class TwoPartMDLScorer(SimpleMDLScorer):
    """Alias-compatible sprint scorer with a proposal delta helper."""

    def __init__(self, config: CodeLengthConfig = CodeLengthConfig()) -> None:
        super().__init__(rule_overhead=config.rule_overhead_bits, category_overhead=config.category_overhead_bits)
        self.config = config

    def delta(self, state: GrammarState, proposal) -> float:
        old = self.score(state)
        new_state = EditApplier().apply(state, proposal)
        new = self.score(new_state)
        return new.total - old.total

    @staticmethod
    def empirical_bits(parse: tuple[ParseEntry, ...]) -> float:
        if not parse:
            return 0.0
        counts = Counter(entry_key(e) for e in parse)
        total = len(parse)
        return sum(count * (-math.log2(count / total)) for count in counts.values())
