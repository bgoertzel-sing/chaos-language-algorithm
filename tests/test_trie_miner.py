import random
import unittest

from chaoslang import CLA
from chaoslang.core.edits import EditApplier
from chaoslang.core.types import CategoryOccurrence, GrammarState, Token, expand_parse
from chaoslang.mining.ngram import NGramPatternMiner, select_non_overlapping
from chaoslang.trie_miner import SuffixTrieMiner


def proposal_signature(proposals):
    return {
        (tuple((entry.kind, entry.value) if isinstance(entry, Token) else (entry.category, entry.member.value) for entry in p.block), p.occurrences)
        for p in proposals
    }


class SuffixTrieMinerTests(unittest.TestCase):
    def test_smoke_empty_single_and_no_repeats(self):
        miner = SuffixTrieMiner(max_ngram=4, min_uses=2)
        for symbols in ([], ["a"], ["a", "b", "c", "d"]):
            state = GrammarState.initial(symbols)
            self.assertEqual(miner.propose_chunks(state), ())

    def test_equivalent_to_ngram_miner_on_representative_inputs(self):
        logistic_like = "a b d d c a a b d c a b d d c a".split()
        repeated = "a b c a b c x a b c a b c".split()
        rng = random.Random(17)
        random_symbols = [rng.choice(["a", "b", "c", "d"]) for _ in range(40)]
        category_stream = [
            Token("a"),
            CategoryOccurrence("M0", Token("x")),
            Token("b"),
            Token("a"),
            CategoryOccurrence("M0", Token("x")),
            Token("b"),
        ]

        for symbols in (logistic_like, repeated, random_symbols, category_stream):
            state = GrammarState.initial(symbols)
            expected = proposal_signature(NGramPatternMiner(max_ngram=5, min_uses=2).propose_chunks(state))
            actual = proposal_signature(SuffixTrieMiner(max_ngram=5, min_uses=2).propose_chunks(state))
            self.assertEqual(actual, expected)

    def test_trie_node_count_is_linear_for_bounded_max_ngram(self):
        n = 128
        max_ngram = 6
        miner = SuffixTrieMiner(max_ngram=max_ngram)
        miner.build_trie(Token(str(i)) for i in range(n))
        # A bounded suffix trie has at most the root plus one new node for each
        # inserted prefix of each suffix: O(N * max_ngram), i.e. O(N) for fixed max_ngram.
        self.assertLessEqual(miner.node_count, 1 + n * max_ngram)

    def test_non_overlapping_selection_matches_ngram_and_applies_cleanly(self):
        symbols = list("aaaaaa")
        state = GrammarState.initial(symbols)
        proposal = SuffixTrieMiner(max_ngram=2, min_uses=3).propose_chunks(state)[0]
        self.assertEqual(select_non_overlapping([0, 1, 2, 3, 4], 2), (0, 2, 4))
        self.assertEqual(proposal.occurrences, (0, 2, 4))
        new_state = EditApplier().apply(state, proposal)
        self.assertEqual(expand_parse(new_state.parse, new_state.grammar), state.corpus.symbols)

    def test_api_switch_uses_suffix_trie_miner(self):
        model = CLA.simple(max_ngram=3, max_iterations=5, enable_categories=False, miner="suffix_trie").fit_symbols("abcabcabc")
        self.assertEqual(model.expand(), tuple("abcabcabc"))


if __name__ == "__main__":
    unittest.main()
