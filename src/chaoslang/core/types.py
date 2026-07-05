"""Backend-neutral CLA domain objects.

The core intentionally has no optional backend dependencies.  Chunks are
sequential nonterminals with productions; categories are first-class objects and
category occurrences retain their chosen member for exact reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence, Union


@dataclass(frozen=True, order=True)
class Token:
    """A vocabulary token.

    ``kind`` is normally ``base`` or ``chunk``. Category membership is expressed
    with :class:`Category` plus :class:`CategoryOccurrence`, not as a chunk RHS.
    """

    value: str
    kind: str = "base"

    def __str__(self) -> str:
        return self.value if self.kind == "base" else f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class CategoryOccurrence:
    """A parse entry ``M[v]`` preserving the observed category member."""

    category: str
    member: Token

    def __str__(self) -> str:
        return f"{self.category}[{self.member.value}]"


ParseEntry = Union[Token, CategoryOccurrence]


@dataclass(frozen=True)
class Production:
    lhs: Token
    rhs: tuple[ParseEntry, ...]


@dataclass(frozen=True)
class Category:
    name: str
    members: frozenset[Token]


@dataclass(frozen=True)
class Corpus:
    """Original observed symbolic corpus.

    Sprint-1 stores one stream and preserves it exactly in ``symbols``.
    """

    symbols: tuple[Token, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_symbols(cls, symbols: Iterable[str | Token]) -> "Corpus":
        out: list[Token] = []
        for sym in symbols:
            out.append(sym if isinstance(sym, Token) else Token(str(sym)))
        return cls(tuple(out))


@dataclass(frozen=True)
class Grammar:
    productions: Mapping[Token, Production] = field(default_factory=dict)
    categories: Mapping[str, Category] = field(default_factory=dict)

    def production_for(self, token: Token) -> Production | None:
        return self.productions.get(token)


@dataclass(frozen=True)
class Score:
    model_bits: float
    data_bits: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.model_bits + self.data_bits


@dataclass(frozen=True)
class Edit:
    """Durable description of an accepted or candidate state transition."""

    kind: str
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GrammarState:
    corpus: Corpus
    parse: tuple[ParseEntry, ...]
    grammar: Grammar = field(default_factory=Grammar)
    score: Score = field(default_factory=lambda: Score(0.0, 0.0))
    history: tuple[str, ...] = ()
    edit_log: tuple[Edit, ...] = ()

    @classmethod
    def initial(cls, symbols: Iterable[str | Token]) -> "GrammarState":
        corpus = Corpus.from_symbols(symbols)
        return cls(corpus=corpus, parse=corpus.symbols)

    def with_score(self, score: Score) -> "GrammarState":
        return replace(self, score=score)


@dataclass(frozen=True)
class ChunkProposal:
    block: tuple[ParseEntry, ...]
    occurrences: tuple[int, ...]
    name: str


@dataclass(frozen=True)
class CategoryProposal:
    members: frozenset[Token]
    name: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class Proposal:
    edit: Edit
    estimated_delta_bits: float | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)


ProposalLike = Union[ChunkProposal, CategoryProposal, Proposal]


def token_text(entry: ParseEntry) -> str:
    if isinstance(entry, CategoryOccurrence):
        return entry.member.value
    return entry.value


def entry_key(entry: ParseEntry) -> tuple[str, str, str]:
    if isinstance(entry, CategoryOccurrence):
        return ("cat", entry.category, entry.member.value)
    return (entry.kind, entry.value, "")


def expand_entry(entry: ParseEntry, grammar: Grammar) -> tuple[Token, ...]:
    if isinstance(entry, CategoryOccurrence):
        return expand_entry(entry.member, grammar)
    prod = grammar.production_for(entry)
    if prod is None:
        return (entry,)
    out: list[Token] = []
    for rhs_entry in prod.rhs:
        out.extend(expand_entry(rhs_entry, grammar))
    return tuple(out)


def expand_parse(parse: Sequence[ParseEntry], grammar: Grammar) -> tuple[Token, ...]:
    out: list[Token] = []
    for entry in parse:
        out.extend(expand_entry(entry, grammar))
    return tuple(out)
