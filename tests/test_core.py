import unittest

from chaoslang import CLA
from chaoslang.benchmarks.attractors import equal_width_symbols, logistic_map
from chaoslang.categorization.context import ContextCategoryInducer
from chaoslang.core.edits import EditApplier
from chaoslang.core.types import CategoryOccurrence, CategoryProposal, ChunkProposal, GrammarState, Token, expand_parse
from chaoslang.scoring.mdl import SimpleMDLScorer
from chaoslang.stores.facts import MemoryFactStore, facts_to_state, state_to_facts


class ChaosLangCoreTests(unittest.TestCase):
    def test_chunk_identity_and_exact_expansion(self):
        state = GrammarState.initial("abcabcabc")
        proposal = ChunkProposal(tuple(state.parse[:3]), (0, 3, 6), "N0")
        new_state = EditApplier().apply(state, proposal)
        self.assertEqual(tuple(t.value for t in expand_parse(new_state.parse, new_state.grammar)), tuple("abcabcabc"))
        prod = next(iter(new_state.grammar.productions.values()))
        self.assertEqual(tuple(e.value for e in prod.rhs), tuple("abc"))
        self.assertEqual(len(new_state.parse), 3)

    def test_category_identity_preserves_member_per_occurrence(self):
        state = GrammarState.initial("axb ayb axb".replace(" ", ""))
        # x positions 1, 7; y position 4
        proposal = CategoryProposal(frozenset({Token("x"), Token("y")}), "M0", (1, 4, 7))
        new_state = EditApplier().apply(state, proposal)
        self.assertIsInstance(new_state.parse[1], CategoryOccurrence)
        self.assertEqual(new_state.parse[1].member, Token("x"))
        self.assertEqual(new_state.parse[4].member, Token("y"))
        self.assertEqual(tuple(t.value for t in expand_parse(new_state.parse, new_state.grammar)), tuple("axbaybaxb"))

    def test_frame_generalization_proposes_xy(self):
        state = GrammarState.initial("axbaybaxb")
        proposals = ContextCategoryInducer(min_occurrences=1).proposals(state)
        member_sets = [{m.value for m in p.members} for p in proposals]
        self.assertIn({"x", "y"}, member_sets)

    def test_mdl_rejects_unrelated_rare_category(self):
        scorer = SimpleMDLScorer()
        state = GrammarState.initial("abcdef")
        state = state.with_score(scorer.score(state))
        proposal = CategoryProposal(frozenset({Token("b"), Token("e")}), "M0", (1, 4))
        candidate = EditApplier().apply(state, proposal)
        candidate = candidate.with_score(scorer.score(candidate))
        self.assertGreater(candidate.score.total, state.score.total)

    def test_rule_pruning_inlines_single_use_rule(self):
        state = GrammarState.initial("abc")
        proposal = ChunkProposal(tuple(state.parse), (0,), "N0")
        new_state = EditApplier().apply(state, proposal)
        pruned = EditApplier().prune_dead_rules(new_state)
        self.assertFalse(pruned.grammar.productions)
        self.assertEqual(tuple(t.value for t in expand_parse(pruned.parse, pruned.grammar)), tuple("abc"))

    def test_determinism(self):
        symbols = tuple("abcabcabcaxbaybaxb")
        a = CLA.simple(max_iterations=8).fit_symbols(symbols)
        b = CLA.simple(max_iterations=8).fit_symbols(symbols)
        self.assertEqual(a.state.parse, b.state.parse)
        self.assertEqual(a.state.history, b.state.history)
        self.assertEqual(a.expand(), b.expand())

    def test_user_api_exact_reconstruction(self):
        model = CLA.simple(max_iterations=8).fit_symbols("abcabcabc")
        self.assertEqual(model.expand(), tuple("abcabcabc"))
        self.assertTrue(model.state.score.total > 0)

    def test_fact_round_trip_core_fields(self):
        model = CLA.simple(max_iterations=4).fit_symbols("abcabcabc")
        facts = state_to_facts(model.state)
        store = MemoryFactStore(facts)
        restored = facts_to_state(store.query())
        self.assertEqual(restored.corpus, model.state.corpus)
        self.assertEqual(restored.parse, model.state.parse)
        self.assertEqual(restored.grammar, model.state.grammar)
        self.assertEqual(tuple(t.value for t in expand_parse(restored.parse, restored.grammar)), tuple("abcabcabc"))

    def test_logistic_symbolization_smoke(self):
        values = logistic_map(steps=16)
        symbols = equal_width_symbols(values, bins=3)
        self.assertEqual(len(symbols), 16)
        self.assertTrue(all(s.startswith("s") for s in symbols))


if __name__ == "__main__":
    unittest.main()
