"""Suffix-trie-based pattern mining for chunk proposals.

This prototype builds a trie of all suffix prefixes up to ``max_ngram``.  For a
fixed ``max_ngram`` this uses O(N) space in the stream length while preserving
the suffix structure needed to enumerate repeated substrings deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from chaoslang.core.types import ChunkProposal, GrammarState, ParseEntry, Token, entry_key
from chaoslang.mining.ngram import select_non_overlapping

EntryKey = tuple[str, str, str]


@dataclass
class SuffixTrieNode:
    """A node representing one prefix of one or more suffixes."""

    depth: int
    children: dict[EntryKey, "SuffixTrieNode"] = field(default_factory=dict)
    positions: list[int] = field(default_factory=list)
    block: tuple[ParseEntry, ...] = ()

    @property
    def frequency(self) -> int:
        """Number of suffix starts sharing this node's path."""

        return len(self.positions)


class SuffixTrieMiner:
    """Mine repeated n-grams by enumerating nodes in a bounded suffix trie."""

    def __init__(self, max_ngram: int = 6, min_uses: int = 2, n_min: int = 2, n_max: int | None = None) -> None:
        if n_max is not None:
            max_ngram = n_max
        if max_ngram < 2:
            raise ValueError("max_ngram must be at least 2")
        if min_uses < 2:
            raise ValueError("min_uses must be at least 2")
        if n_min < 1:
            raise ValueError("n_min must be at least 1")
        self.n_min = n_min
        self.max_ngram = max_ngram
        self.min_uses = min_uses
        self.root = SuffixTrieNode(depth=0)
        self.node_count = 1

    def build_trie(self, entries: Iterable[ParseEntry]) -> SuffixTrieNode:
        """Build and return a suffix trie capped at ``max_ngram`` symbols."""

        stream = tuple(entries)
        root = SuffixTrieNode(depth=0)
        node_count = 1
        for start in range(len(stream)):
            node = root
            stop = min(len(stream), start + self.max_ngram)
            for end in range(start, stop):
                key = entry_key(stream[end])
                child = node.children.get(key)
                if child is None:
                    child = SuffixTrieNode(depth=node.depth + 1, block=stream[start : end + 1])
                    node.children[key] = child
                    node_count += 1
                child.positions.append(start)
                node = child
        self.root = root
        self.node_count = node_count
        return root

    def propose_chunks(self, state: GrammarState) -> tuple[ChunkProposal, ...]:
        root = self.build_trie(state.parse)
        raw: list[tuple[tuple[EntryKey, ...], tuple[ParseEntry, ...], tuple[int, ...]]] = []

        def visit(node: SuffixTrieNode, path: tuple[EntryKey, ...]) -> None:
            if self.n_min <= node.depth <= self.max_ngram:
                occurrences = select_non_overlapping(node.positions, node.depth)
                if len(occurrences) >= self.min_uses:
                    raw.append((path, node.block, occurrences))
            for key in sorted(node.children):
                visit(node.children[key], path + (key,))

        for key in sorted(root.children):
            visit(root.children[key], (key,))

        raw.sort(key=lambda item: (-(len(item[1]) - 1) * len(item[2]), len(item[1]), item[0]))
        start = _next_chunk_index(state)
        return tuple(ChunkProposal(block, occ, f"N{start + i}") for i, (_key, block, occ) in enumerate(raw))

    def proposals(self, state: GrammarState) -> list[ChunkProposal]:
        return list(self.propose_chunks(state))


def _next_chunk_index(state: GrammarState) -> int:
    highest = -1
    for token in state.grammar.productions:
        if isinstance(token, Token) and token.kind == "chunk" and token.value.startswith("N") and token.value[1:].isdigit():
            highest = max(highest, int(token.value[1:]))
    return highest + 1
