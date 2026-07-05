"""Pure-Python pattern mining for sprint-1 chunk proposals."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from chaoslang.core.types import ChunkProposal, GrammarState, ParseEntry, entry_key


def select_non_overlapping(starts: list[int], width: int) -> tuple[int, ...]:
    return tuple(NGramPatternMiner._non_overlapping(starts, width))


class NGramPatternMiner:
    """Count repeated n-grams and propose deterministic chunk rewrites."""

    def __init__(self, max_ngram: int = 6, min_uses: int = 2, n_min: int = 2, n_max: int | None = None) -> None:
        if n_max is not None:
            max_ngram = n_max
        if max_ngram < 2:
            raise ValueError("max_ngram must be at least 2")
        if min_uses < 2:
            raise ValueError("min_uses must be at least 2")
        self.n_min = n_min
        self.max_ngram = max_ngram
        self.min_uses = min_uses

    def propose_chunks(self, state: GrammarState) -> tuple[ChunkProposal, ...]:
        raw: list[tuple[tuple[tuple[str, str, str], ...], tuple[ParseEntry, ...], tuple[int, ...]]] = []
        for n in range(self.n_min, min(self.max_ngram, len(state.parse)) + 1):
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
                    raw.append((key, blocks[key], tuple(occurrences)))
        raw.sort(key=lambda item: (-(len(item[1]) - 1) * len(item[2]), len(item[1]), item[0]))
        start = _next_chunk_index(state)
        return tuple(ChunkProposal(block, occ, f"N{start + i}") for i, (_key, block, occ) in enumerate(raw))

    def proposals(self, state: GrammarState) -> list[ChunkProposal]:
        return list(self.propose_chunks(state))


def _next_chunk_index(state: GrammarState) -> int:
    highest = -1
    for token in state.grammar.productions:
        if token.kind == "chunk" and token.value.startswith("N") and token.value[1:].isdigit():
            highest = max(highest, int(token.value[1:]))
    return highest + 1


def select_non_overlapping(starts: Iterable[int], width: int) -> tuple[int, ...]:
    selected: list[int] = []
    next_free = -1
    for start in sorted(starts):
        if start >= next_free:
            selected.append(start)
            next_free = start + width
    return tuple(selected)
