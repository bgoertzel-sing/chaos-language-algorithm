import unittest

from chaoslang import CLA
from chaoslang.core.edits import EditApplier
from chaoslang.core.types import ChunkProposal, GrammarState, Token, expand_parse
from chaoslang.mining import NGramPatternMiner, select_non_overlapping
from chaoslang.scoring import TwoPartMDLScorer


class ChunkMVPTests(unittest.TestCase):
    def test_fit_preserves_exact_reconstruction_for_list_and_whitespace(self):
        symbols = "a b c a b c a b c".split()
        model = CLA.simple(max_ngram=3, max_iterations=5).fit_symbols(symbols)
        self.assertEqual(model.expand(), tuple(symbols))

        whitespace_model = CLA.simple(max_ngram=3, max_iterations=5).fit_symbols("x y x y")
        self.assertEqual(whitespace_model.expand(), ("x", "y", "x", "y"))

    def test_chunk_identity_and_use_counts(self):
        state = GrammarState.initial("a b c a b c".split())
        block = state.parse[:3]
        proposal = ChunkProposal(block=block, occurrences=(0, 3), name="N0")
        new_state = EditApplier().apply(state, proposal)
        lhs = Token("N0", kind="chunk")
        self.assertEqual(new_state.grammar.productions[lhs].rhs, block)
        self.assertEqual(tuple(t.value for t in expand_parse((lhs,), new_state.grammar)), ("a", "b", "c"))
        self.assertEqual(EditApplier().rule_use_counts(new_state)[lhs], 2)

    def test_non_overlapping_occurrence_selection(self):
        self.assertEqual(select_non_overlapping([0, 1, 2, 4], 2), (0, 2, 4))

    def test_mdl_rejects_or_avoids_worsening_edit(self):
        symbols = "a b c d".split()
        state = GrammarState.initial(symbols)
        scorer = TwoPartMDLScorer()
        before = scorer.score(state)
        bad = ChunkProposal(block=state.parse[:2], occurrences=(0,), name="N0")
        after_state = EditApplier().apply(state, bad)
        after = scorer.score(after_state)
        # EditApplier prunes single-use chunks back to identity; the key invariant
        # is that the greedy loop does not accept a worsening edit.
        self.assertGreaterEqual(after.total, before.total)

        model = CLA.simple(max_ngram=4, max_iterations=5).fit_symbols(symbols)
        self.assertEqual(model.expand(), tuple(symbols))
        self.assertEqual(model.state.grammar.productions, {})

    def test_miner_avoids_chunk_name_collisions_after_tie_breaks(self):
        # This M1-like stream used to accept N2 before N1, then the next mining
        # round reused N2 and broke exact reconstruction by overwriting a rule.
        symbols = "a a a a a a a a b b b b b b b b b c c c c c".split()
        model = CLA.simple(max_ngram=4, max_iterations=6, enable_categories=False).fit_symbols(symbols)
        self.assertEqual(model.expand(), tuple(symbols))
        names = [token.value for token in model.grammar.productions]
        self.assertEqual(len(names), len(set(names)))

    def test_miner_and_greedy_loop_are_deterministic(self):
        symbols = "a b a b a b c c c c".split()
        state = GrammarState.initial(symbols)
        proposals1 = NGramPatternMiner(max_ngram=3).propose_chunks(state)
        proposals2 = NGramPatternMiner(max_ngram=3).propose_chunks(state)
        self.assertEqual(proposals1, proposals2)

        m1 = CLA.simple(max_ngram=3, max_iterations=10).fit_symbols(symbols)
        m2 = CLA.simple(max_ngram=3, max_iterations=10).fit_symbols(symbols)
        self.assertEqual(m1.parse, m2.parse)
        self.assertEqual(m1.grammar.productions, m2.grammar.productions)
        self.assertEqual(m1.score.total, m2.score.total)


if __name__ == "__main__":
    unittest.main()
