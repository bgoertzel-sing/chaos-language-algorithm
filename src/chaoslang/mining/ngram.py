"""Pure-Python pattern mining for chunk proposals."""
from __future__ import annotations

from typing import Iterable

from chaoslang.core.types import ChunkProposal, GrammarState, ParseEntry, entry_key


EntryKey = tuple[str, str, str]
BlockKey = tuple[EntryKey, ...]


class _BoundedSuffixTrieNode:
    """A depth-bounded trie node storing starts sharing this suffix-prefix."""

    __slots__ = ("children", "starts")

    def __init__(self) -> None:
        self.children: dict[EntryKey, _BoundedSuffixTrieNode] = {}
        self.starts: list[int] = []


class NGramPatternMiner:
    """Propose repeated n-gram chunk rewrites using a bounded suffix trie.

    The earlier miner materialized every window as a tuple for each n-gram
    length. That is simple but memory-hostile on high-cardinality compound-symbol
    streams because it duplicates long tuple keys across all n. This miner builds
    one depth-bounded suffix trie over the current parse. Shared prefixes are
    stored once, and repeated substrings are emitted from trie nodes whose path
    depth is within ``[n_min, max_ngram]`` and whose non-overlapping starts meet
    ``min_uses``. It remains intentionally bounded rather than a full suffix
    automaton because CLA currently only proposes chunks up to small n.
    """

    def __init__(self, max_ngram: int = 6, min_uses: int = 2, n_min: int = 2, n_max: int | None = None) -> None:
        if n_max is not None:
            max_ngram = n_max
        if max_ngram < 2:
            raise ValueError("max_ngram must be at least 2")
        if min_uses < 2:
            raise ValueError("min_uses must be at least 2")
        if n_min < 1:
            raise ValueError("n_min must be at least 1")
        if n_min > max_ngram:
            raise ValueError("n_min must not exceed max_ngram")
        self.n_min = n_min
        self.max_ngram = max_ngram
        self.min_uses = min_uses

    def propose_chunks(self, state: GrammarState) -> tuple[ChunkProposal, ...]:
        parse = state.parse
        if len(parse) < self.n_min:
            return ()

        keys = tuple(entry_key(entry) for entry in parse)
        max_depth = min(self.max_ngram, len(parse))
        root = self._build_bounded_suffix_trie(keys, max_depth)

        raw: list[tuple[BlockKey, tuple[ParseEntry, ...], tuple[int, ...]]] = []
        for key, starts in self._repeated_paths(root):
            width = len(key)
            if width < self.n_min:
                continue
            occurrences = select_non_overlapping(starts, width)
            if len(occurrences) >= self.min_uses:
                first = occurrences[0]
                raw.append((key, tuple(parse[first : first + width]), tuple(occurrences)))

        raw.sort(key=lambda item: (-(len(item[1]) - 1) * len(item[2]), len(item[1]), item[0]))
        start = _next_chunk_index(state)
        return tuple(ChunkProposal(block, occ, f"N{start + i}") for i, (_key, block, occ) in enumerate(raw))

    def proposals(self, state: GrammarState) -> list[ChunkProposal]:
        return list(self.propose_chunks(state))

    @staticmethod
    def _non_overlapping(starts: Iterable[int], width: int) -> tuple[int, ...]:
        return select_non_overlapping(starts, width)

    @staticmethod
    def _build_bounded_suffix_trie(keys: tuple[EntryKey, ...], max_depth: int) -> _BoundedSuffixTrieNode:
        root = _BoundedSuffixTrieNode()
        for start in range(len(keys)):
            node = root
            end = min(len(keys), start + max_depth)
            for key in keys[start:end]:
                child = node.children.get(key)
                if child is None:
                    child = _BoundedSuffixTrieNode()
                    node.children[key] = child
                node = child
                node.starts.append(start)
        return root

    @staticmethod
    def _repeated_paths(root: _BoundedSuffixTrieNode) -> Iterable[tuple[BlockKey, list[int]]]:
        stack: list[tuple[BlockKey, _BoundedSuffixTrieNode]] = [((), root)]
        while stack:
            prefix, node = stack.pop()
            if prefix and len(node.starts) >= 2:
                yield prefix, node.starts
            for key in sorted(node.children, reverse=True):
                stack.append((prefix + (key,), node.children[key]))


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
