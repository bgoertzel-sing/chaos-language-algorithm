"""Proposal application for CLA grammar states."""
from __future__ import annotations

from dataclasses import replace

from .types import (
    Category,
    CategoryOccurrence,
    CategoryProposal,
    ChunkProposal,
    Edit,
    Grammar,
    GrammarState,
    ParseEntry,
    Production,
    Proposal,
    ProposalLike,
    Token,
    expand_parse,
)


class EditApplier:
    """Pure edit application; miners never mutate state directly."""

    min_rule_uses: int = 2
    min_pruned_rule_len: int = 3

    def apply(self, state: GrammarState, proposal: ProposalLike) -> GrammarState:
        if isinstance(proposal, Proposal):
            proposal = self._proposal_from_edit(proposal.edit)
        if isinstance(proposal, ChunkProposal):
            return self.apply_chunk(state, proposal)
        return self.apply_category(state, proposal)

    def apply_chunk(self, state: GrammarState, proposal: ChunkProposal) -> GrammarState:
        lhs = Token(proposal.name, kind="chunk")
        occurrence_set = set(proposal.occurrences)
        block_len = len(proposal.block)
        new_parse: list[ParseEntry] = []
        i = 0
        while i < len(state.parse):
            if i in occurrence_set and tuple(state.parse[i : i + block_len]) == proposal.block:
                new_parse.append(lhs)
                i += block_len
            else:
                new_parse.append(state.parse[i])
                i += 1
        productions = dict(state.grammar.productions)
        productions[lhs] = Production(lhs, proposal.block)
        edit = Edit(
            "AddChunkRule",
            {
                "name": proposal.name,
                "block": tuple(str(e) for e in proposal.block),
                "occurrences": proposal.occurrences,
            },
        )
        new_state = replace(
            state,
            parse=tuple(new_parse),
            grammar=Grammar(productions, dict(state.grammar.categories)),
            history=state.history + (f"chunk {lhs.value} -> {' '.join(map(str, proposal.block))}",),
            edit_log=state.edit_log + (edit,),
        )
        self._assert_exact(state, new_state)
        if len(proposal.occurrences) >= self.min_rule_uses:
            return self.prune_dead_rules(new_state)
        return new_state

    def apply_category(self, state: GrammarState, proposal: CategoryProposal) -> GrammarState:
        members = proposal.members
        positions = set(proposal.positions)
        new_parse: list[ParseEntry] = []
        for idx, entry in enumerate(state.parse):
            if idx in positions and isinstance(entry, Token) and entry in members:
                new_parse.append(CategoryOccurrence(proposal.name, entry))
            else:
                new_parse.append(entry)
        categories = dict(state.grammar.categories)
        categories[proposal.name] = Category(proposal.name, frozenset(members))
        edit = Edit(
            "AddCategory",
            {"name": proposal.name, "members": tuple(sorted(t.value for t in members)), "positions": proposal.positions},
        )
        new_state = replace(
            state,
            parse=tuple(new_parse),
            grammar=Grammar(dict(state.grammar.productions), categories),
            history=state.history + (f"category {proposal.name} -> {{{','.join(sorted(t.value for t in members))}}}",),
            edit_log=state.edit_log + (edit,),
        )
        self._assert_exact(state, new_state)
        return new_state

    def prune_dead_rules(self, state: GrammarState) -> GrammarState:
        changed = True
        current = state
        while changed:
            changed = False
            uses = self.rule_use_counts(current)
            for lhs in sorted(list(current.grammar.productions), key=lambda t: t.value):
                if uses.get(lhs, 0) < self.min_rule_uses:
                    prod = current.grammar.productions[lhs]
                    new_parse: list[ParseEntry] = []
                    for entry in current.parse:
                        if entry == lhs:
                            new_parse.extend(prod.rhs)
                        else:
                            new_parse.append(entry)
                    productions = dict(current.grammar.productions)
                    del productions[lhs]
                    current = replace(
                        current,
                        parse=tuple(new_parse),
                        grammar=Grammar(productions, dict(current.grammar.categories)),
                        history=current.history + (f"prune {lhs.value}",),
                        edit_log=current.edit_log + (Edit("DeleteUnusedRule", {"name": lhs.value}),),
                    )
                    changed = True
                    break
        assert expand_parse(current.parse, current.grammar) == current.corpus.symbols
        return current

    def rule_use_counts(self, state: GrammarState) -> dict[Token, int]:
        counts = {lhs: 0 for lhs in state.grammar.productions}
        for entry in state.parse:
            if isinstance(entry, Token) and entry in counts:
                counts[entry] += 1
        for prod in state.grammar.productions.values():
            for entry in prod.rhs:
                if isinstance(entry, Token) and entry in counts:
                    counts[entry] += 1
        return counts

    def _assert_exact(self, before: GrammarState, after: GrammarState) -> None:
        assert expand_parse(before.parse, before.grammar) == expand_parse(after.parse, after.grammar)
        assert expand_parse(after.parse, after.grammar) == after.corpus.symbols

    def _proposal_from_edit(self, edit: Edit) -> ChunkProposal | CategoryProposal:
        raise NotImplementedError("generic Edit replay is a persistence TODO; pass typed proposals for now")
