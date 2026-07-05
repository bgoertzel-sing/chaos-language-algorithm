"""Regression test for pruning rules referenced in other productions' RHS.

When a chunk rule A references another chunk rule B in its RHS, and B is
subsequently pruned for under-use, B must be expanded not only in the parse
but also in A's RHS.  Otherwise A retains a dangling reference to the
deleted B and expand_parse fails.

This bug was discovered running CLA on a 256-step Lorenz-63 trajectory.
"""
from __future__ import annotations

import unittest

from chaoslang import CLA
from chaoslang.core.types import (
    Corpus,
    Grammar,
    GrammarState,
    Production,
    Token,
    expand_parse,
)
from chaoslang.core.edits import EditApplier
from chaoslang.mining.ngram import ChunkProposal


class NestedPruneRegressionTests(unittest.TestCase):
    """Reproduce the exact scenario: N1 -> base*5, N168 -> [N1, base]."""

    def test_pruning_rule_referenced_in_other_production_rhs(self) -> None:
        # Build a corpus where N1 (5 base tokens) appears 5 times,
        # and a new rule N168 -> [N1, extra] consumes 4 of those uses.
        # After applying N168, N1 has 1 parse use + 1 RHS reference = 2,
        # but rule_use_counts undercounts (counts 1 from RHS, not 4).
        # N1 gets pruned, and its RHS must be substituted into N168's RHS.
        symbols = tuple(Token(c) for c in "abababababxxxx")  # 5 "ab" + "xxxx"
        # Actually, build a more direct test:
        # Create state with N1 rule, then apply a chunk that references N1

        # Simpler: use symbols that trigger the bug on Lorenz-63
        from chaoslang.benchmarks.attractors import lorenz63, m1_symbolize

        traj = lorenz63(steps=256, discard=64)
        symbols_tuple = m1_symbolize(traj, bins=4)

        # This used to crash with AssertionError
        model = CLA.simple(max_iterations=8, seed=0).fit_symbols(symbols_tuple)

        # Exact reconstruction must hold
        self.assertEqual(model.expand(), symbols_tuple)

    def test_unit_level_nested_prune(self) -> None:
        """Direct unit test: create N1, create N168 referencing N1, prune N1."""

        applier = EditApplier()

        # Corpus: a a a a a b a a a a a b  (N1->a a a a a appears twice,
        # then we create N168 -> [N1, b] which consumes both N1 uses,
        # making N1 have 0 parse uses + 1 RHS ref = 1 < 2, so it gets pruned)
        base = [Token("a"), Token("a"), Token("a"), Token("a"), Token("a"),
                Token("b"), Token("a"), Token("a"), Token("a"), Token("a"), Token("a"),
                Token("b")]

        state = GrammarState.initial(base)

        # Step 1: Create N1 -> a a a a a (at positions 0 and 6)
        n1_proposal = ChunkProposal(
            block=tuple(base[0:5]),
            occurrences=(0, 6),
            name="N1",
        )
        state = applier.apply_chunk(state, n1_proposal)
        # N1 has 2 uses, so it stays

        # Step 2: Create N168 -> [N1, b] (at positions 0 and 6 of the new parse)
        n1_token = Token("N1", kind="chunk")
        n168_block = (n1_token, Token("b"))
        # Find occurrences of [N1, b] in the parse
        occurrences = []
        parse = state.parse
        for i in range(len(parse) - 1):
            if parse[i] == n1_token and parse[i + 1] == Token("b"):
                occurrences.append(i)
        self.assertEqual(len(occurrences), 2)

        n168_proposal = ChunkProposal(
            block=n168_block,
            occurrences=tuple(occurrences),
            name="N168",
        )
        # This applies the chunk and then prunes. N1 should have 0 parse uses
        # but 1 RHS ref from N168. Since rule_use_counts counts 1 (not 2),
        # N1 gets pruned. The fix ensures N1's RHS is substituted into N168.
        state = applier.apply_chunk(state, n168_proposal)

        # N168 should now have RHS = (a, a, a, a, a, b) instead of (N1, b)
        n168_token = Token("N168", kind="chunk")
        self.assertIn(n168_token, state.grammar.productions)
        n168_prod = state.grammar.productions[n168_token]
        rhs_values = [e.value if isinstance(e, Token) else str(e) for e in n168_prod.rhs]
        # After pruning N1, N168's RHS should be the expanded form
        self.assertEqual(rhs_values, ["a", "a", "a", "a", "a", "b"])

        # N1 should no longer be in productions
        self.assertNotIn(n1_token, state.grammar.productions)

        # Exact reconstruction
        self.assertEqual(expand_parse(state.parse, state.grammar), state.corpus.symbols)


if __name__ == "__main__":
    unittest.main()
