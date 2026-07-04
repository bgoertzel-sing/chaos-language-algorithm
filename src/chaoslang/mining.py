"""Pure-Python pattern mining for sprint-1 chunk proposals."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .core.types import ChunkProposal, GrammarState, ParseEntry, entry_key


class NGramPatternMiner:
    """Count repeated n-grams and propose deterministic chunk rewrites."""

    def __init__(self, max_ngram: int = 6, min_uses: int = 2) -> None:
        if max_ngram < 2:
            raise ValueError("max_ngram must be at least 2")
        if min_uses < 2:
            raise ValueError("min_uses must be at least 2")
        self.max_ngram = max_ngram
        self.min_uses = min_uses

    def propose_chunks(self, state: GrammarState) -> tuple[ChunkProposal, ...]:
        proposals: list[ChunkProposal] = []
        next_index = len(state.grammar.productions)
        for n in range(2, min(self.max_ngram, len(state.parse)) + 1):
            positions: dict[tuple[tuple[str, str, str], ...], list[int]] = defaultdict(list)
            blocks: dict[tuple[tuple[str, str, str], ...], tuple[ParseEntry, ...]] = {}
            for i in range(0, len(state.parse) - n + 1):
                block = tuple(state.parse[i : i + n])
                key = tuple(entry_key(e) for e in block)
                positions[key].append(i)
                blocks.setdefault(key, block)
            for key in sorted(positions):
                occurrences = select_non_overlapping(positions[key], n)
                if len(occurrences) >= self.min_uses:
                    proposals.append(ChunkProposal(blocks[key], tuple(occurrences), f"N{next_index}"))
                    next_index += 1
        return tuple(proposals)


def select_non_overlapping(starts: Iterable[int], width: int) -> tuple[int, ...]:
    selected: list[int] = []
    next_free = -1
    for start in sorted(starts):
        if start >= next_free:
            selected.append(start)
            next_free = start + width
    return tuple(selected)
