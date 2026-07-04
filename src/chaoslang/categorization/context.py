"""Deterministic hard category proposal from immediate contexts."""
from __future__ import annotations

from collections import Counter, defaultdict

from chaoslang.core.types import CategoryProposal, GrammarState, Token

BOS = "<BOS>"
EOS = "<EOS>"


class ContextCategoryInducer:
    """Group base/chunk tokens with identical left/right context signatures.

    This is intentionally conservative and deterministic. It captures simple
    frames such as ``a x b`` / ``a y b`` without treating categories as chunks.
    """

    def __init__(self, min_members: int = 2, min_occurrences: int = 2) -> None:
        self.min_members = min_members
        self.min_occurrences = min_occurrences

    def proposals(self, state: GrammarState) -> list[CategoryProposal]:
        parse = state.parse
        contexts: dict[Token, Counter[tuple[str, str]]] = defaultdict(Counter)
        positions_by_token: dict[Token, list[int]] = defaultdict(list)
        for i, entry in enumerate(parse):
            if not isinstance(entry, Token):
                continue
            left = self._label(parse[i - 1]) if i > 0 else BOS
            right = self._label(parse[i + 1]) if i + 1 < len(parse) else EOS
            contexts[entry][(left, right)] += 1
            positions_by_token[entry].append(i)
        groups: dict[frozenset[tuple[str, str]], list[Token]] = defaultdict(list)
        for tok, counter in contexts.items():
            if sum(counter.values()) >= self.min_occurrences:
                # Sprint-1 hard categories use the set of immediate contexts,
                # not exact occurrence counts, so axb/ayb/axb proposes {x,y}.
                signature = frozenset(counter)
                groups[signature].append(tok)
        proposals: list[CategoryProposal] = []
        start = len(state.grammar.categories)
        for signature, members in sorted(groups.items(), key=lambda kv: (kv[0], [t.value for t in kv[1]])):
            if len(members) >= self.min_members:
                positions = sorted(i for tok in members for i in positions_by_token[tok])
                name = f"M{start + len(proposals)}"
                proposals.append(CategoryProposal(frozenset(members), name, tuple(positions)))
        return proposals

    @staticmethod
    def _label(entry: object) -> str:
        if isinstance(entry, Token):
            return f"{entry.kind}:{entry.value}"
        return str(entry)
