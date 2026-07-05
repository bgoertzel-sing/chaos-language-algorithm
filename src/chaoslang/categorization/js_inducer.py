"""JS-divergence-based category inducer.

Extends the exact-signature ContextCategoryInducer by clustering tokens whose
context distributions are within a Jensen-Shannon divergence threshold, allowing
softer category proposals when context signatures partially overlap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from chaoslang.core.types import CategoryProposal, GrammarState, Token
from .context import BOS, EOS, ContextCategoryInducer
from .js import cluster_by_js, js_divergence


class JSCategoryInducer:
    """Group tokens by JS-divergence similarity of their context histograms.

    Unlike ContextCategoryInducer (exact signature match), this inducer builds
    context distributions per token and clusters tokens whose JS divergence falls
    below a threshold.  This produces softer categories: tokens that share most
    (but not all) contexts can still be grouped together.

    Parameters
    ----------
    threshold : float
        Maximum JS divergence (in bits) for two tokens to be clustered together.
        0.0 means only exact signature matches (equivalent to ContextCategoryInducer).
        Typical useful range: 0.01–0.5 bits.
    min_members : int
        Minimum cluster size to emit a CategoryProposal.
    min_occurrences : int
        Minimum total occurrences for a token to be considered for clustering.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.1,
        min_members: int = 2,
        min_occurrences: int = 2,
    ) -> None:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold
        self.min_members = min_members
        self.min_occurrences = min_occurrences

    def proposals(self, state: GrammarState) -> list[CategoryProposal]:
        parse = state.parse
        contexts: dict[Token, Counter[str]] = defaultdict(Counter)
        positions_by_token: dict[Token, list[int]] = defaultdict(list)

        for i, entry in enumerate(parse):
            if not isinstance(entry, Token):
                continue
            left = self._label(parse[i - 1]) if i > 0 else BOS
            right = self._label(parse[i + 1]) if i + 1 < len(parse) else EOS
            ctx_key = f"{left}/{right}"
            contexts[entry][ctx_key] += 1
            positions_by_token[entry].append(i)

        # Filter tokens by minimum occurrences
        eligible = {
            tok: hist
            for tok, hist in contexts.items()
            if sum(hist.values()) >= self.min_occurrences
        }

        if not eligible:
            return []

        # Cluster by JS divergence
        # Use repr of token as the key for cluster_by_js (must be Hashable)
        histograms: dict[str, dict[str, float]] = {}
        token_by_key: dict[str, Token] = {}
        for tok in sorted(eligible, key=lambda t: t.value):
            key = repr(tok)
            histograms[key] = dict(eligible[tok])
            token_by_key[key] = tok

        clusters = cluster_by_js(histograms, threshold=self.threshold)

        proposals: list[CategoryProposal] = []
        start = len(state.grammar.categories)
        for idx, cluster in enumerate(clusters):
            members = frozenset(token_by_key[k] for k in cluster)
            if len(members) >= self.min_members:
                positions = sorted(
                    i for tok in members for i in positions_by_token[tok]
                )
                name = f"M{start + len(proposals)}"
                proposals.append(
                    CategoryProposal(frozenset(members), name, tuple(positions))
                )
        return proposals

    @staticmethod
    def _label(entry: object) -> str:
        return ContextCategoryInducer._label(entry)
