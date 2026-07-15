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

    def test_bounded_suffix_trie_matches_bruteforce_for_compound_symbols(self):
        symbols = [
            f"d{dim}:b{bucket}"
            for _cycle in range(4)
            for dim, bucket in ((0, 1), (1, 3), (2, 5), (0, 1), (1, 3))
        ]
        state = GrammarState.initial(symbols)
        proposals = NGramPatternMiner(max_ngram=5, min_uses=2).propose_chunks(state)

        expected = _bruteforce_chunk_keys(state, max_ngram=5, min_uses=2)
        observed = {
            (tuple((entry.kind, entry.value) for entry in proposal.block), proposal.occurrences)
            for proposal in proposals
        }
        self.assertEqual(observed, expected)
        self.assertTrue(all(len(p.block) <= 5 for p in proposals))

    def test_bounded_suffix_trie_handles_high_cardinality_stream(self):
        motif = tuple(f"d{i}:b{(i * 7) % 23}" for i in range(20))
        symbols = list(motif * 6) + [f"unique:{i}" for i in range(120)] + list(motif * 4)
        state = GrammarState.initial(symbols)
        proposals = NGramPatternMiner(max_ngram=20, min_uses=3).propose_chunks(state)

        self.assertTrue(proposals)
        self.assertEqual(tuple(token.value for token in proposals[0].block), motif)
        self.assertGreaterEqual(len(proposals[0].occurrences), 3)


def _bruteforce_chunk_keys(state, *, max_ngram: int, min_uses: int):
    expected = set()
    parse = state.parse
    for n in range(2, min(max_ngram, len(parse)) + 1):
        positions = {}
        for i in range(0, len(parse) - n + 1):
            block = tuple(parse[i : i + n])
            key = tuple((entry.kind, entry.value) for entry in block)
            positions.setdefault(key, []).append(i)
        for key, starts in positions.items():
            occurrences = select_non_overlapping(starts, n)
            if len(occurrences) >= min_uses:
                expected.add((key, occurrences))
    return expected


if __name__ == "__main__":
    unittest.main()
