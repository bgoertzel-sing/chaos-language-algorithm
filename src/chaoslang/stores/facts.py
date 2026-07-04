"""Backend-neutral fact projection seam for future Hyperon adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from chaoslang.core.types import (
    Category,
    CategoryOccurrence,
    Corpus,
    Grammar,
    GrammarState,
    Production,
    Score,
    Token,
)


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: tuple[str, ...]
    truth: float = 1.0
    attrs: dict[str, str] = field(default_factory=dict)


class FactStore(Protocol):
    def add(self, fact: Fact) -> None: ...
    def query(self, predicate: str | None = None) -> list[Fact]: ...


class MemoryFactStore:
    def __init__(self, facts: Iterable[Fact] = ()) -> None:
        self._facts = list(facts)

    def add(self, fact: Fact) -> None:
        self._facts.append(fact)

    def query(self, predicate: str | None = None) -> list[Fact]:
        if predicate is None:
            return list(self._facts)
        return [f for f in self._facts if f.predicate == predicate]


def state_to_facts(state: GrammarState) -> list[Fact]:
    facts: list[Fact] = []
    for i, tok in enumerate(state.corpus.symbols):
        facts.append(Fact("corpus_token", (str(i), tok.kind, tok.value)))
    for i, entry in enumerate(state.parse):
        if isinstance(entry, CategoryOccurrence):
            facts.append(Fact("parse_category", (str(i), entry.category, entry.member.kind, entry.member.value)))
        else:
            facts.append(Fact("parse_token", (str(i), entry.kind, entry.value)))
    for lhs, prod in state.grammar.productions.items():
        facts.append(Fact("production", (lhs.kind, lhs.value)))
        for j, entry in enumerate(prod.rhs):
            if isinstance(entry, CategoryOccurrence):
                facts.append(Fact("production_rhs_category", (lhs.kind, lhs.value, str(j), entry.category, entry.member.kind, entry.member.value)))
            else:
                facts.append(Fact("production_rhs_token", (lhs.kind, lhs.value, str(j), entry.kind, entry.value)))
    for name, cat in state.grammar.categories.items():
        for member in sorted(cat.members, key=lambda t: (t.kind, t.value)):
            facts.append(Fact("category_member", (name, member.kind, member.value)))
    facts.append(Fact("score", (repr(state.score.model_bits), repr(state.score.data_bits))))
    return facts


def facts_to_state(facts: Iterable[Fact]) -> GrammarState:
    by_pred: dict[str, list[Fact]] = {}
    for fact in facts:
        by_pred.setdefault(fact.predicate, []).append(fact)
    corpus = Corpus(tuple(Token(f.args[2], f.args[1]) for f in sorted(by_pred.get("corpus_token", []), key=lambda f: int(f.args[0]))))
    parse_entries = []
    parse_facts = by_pred.get("parse_token", []) + by_pred.get("parse_category", [])
    for fact in sorted(parse_facts, key=lambda f: int(f.args[0])):
        if fact.predicate == "parse_token":
            parse_entries.append(Token(fact.args[2], fact.args[1]))
        else:
            parse_entries.append(CategoryOccurrence(fact.args[1], Token(fact.args[3], fact.args[2])))
    productions: dict[Token, Production] = {}
    rhs_by_lhs: dict[Token, list[tuple[int, object]]] = {}
    for fact in by_pred.get("production_rhs_token", []):
        lhs = Token(fact.args[1], fact.args[0])
        rhs_by_lhs.setdefault(lhs, []).append((int(fact.args[2]), Token(fact.args[4], fact.args[3])))
    for fact in by_pred.get("production_rhs_category", []):
        lhs = Token(fact.args[1], fact.args[0])
        rhs_by_lhs.setdefault(lhs, []).append((int(fact.args[2]), CategoryOccurrence(fact.args[3], Token(fact.args[5], fact.args[4]))))
    for lhs, entries in rhs_by_lhs.items():
        productions[lhs] = Production(lhs, tuple(e for _, e in sorted(entries, key=lambda x: x[0])))
    cat_members: dict[str, set[Token]] = {}
    for fact in by_pred.get("category_member", []):
        cat_members.setdefault(fact.args[0], set()).add(Token(fact.args[2], fact.args[1]))
    categories = {name: Category(name, frozenset(members)) for name, members in cat_members.items()}
    score_fact = by_pred.get("score", [Fact("score", ("0.0", "0.0"))])[0]
    return GrammarState(corpus, tuple(parse_entries), Grammar(productions, categories), Score(float(score_fact.args[0]), float(score_fact.args[1])))
