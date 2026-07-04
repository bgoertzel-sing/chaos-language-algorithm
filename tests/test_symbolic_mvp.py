import unittest

from chaoslang import CLA, Token
from chaoslang.core.types import CategoryOccurrence
from chaoslang.core.edits import EditApplier
from chaoslang.core.types import CategoryProposal, GrammarState


class SymbolicMVPTests(unittest.TestCase):
    def test_readme_example_exact_reconstruction(self):
        model = CLA.simple(max_iterations=8).fit_symbols("abcabcabc")
        self.assertEqual(model.expand(), tuple("abcabcabc"))
        self.assertLessEqual(model.state.score.total, 9.0)

    def test_chunk_identity_production_not_sequence_alias(self):
        model = CLA.simple(max_iterations=1, enable_categories=False).fit_symbols("abcabcabc")
        self.assertEqual(model.expand(), tuple("abcabcabc"))
        self.assertTrue(model.state.grammar.productions)
        prod = next(iter(model.state.grammar.productions.values()))
        self.assertEqual(prod.lhs.kind, "chunk")
        self.assertEqual(tuple(t.value for t in prod.rhs), tuple("abc"))
        self.assertEqual(tuple(t.value for t in model.state.parse), (prod.lhs.value,) * 3)

    def test_simple_repeated_chunk_induction_for_pretokenized_symbols(self):
        model = CLA.simple(max_iterations=4, enable_categories=False).fit_symbols(["red", "blue", "red", "blue", "red", "blue"])
        self.assertEqual(model.expand(), ("red", "blue", "red", "blue", "red", "blue"))
        productions = list(model.state.grammar.productions.values())
        self.assertTrue(any(tuple(t.value for t in p.rhs) == ("red", "blue") for p in productions))

    def test_category_occurrence_preserves_exact_member(self):
        state = GrammarState.initial("axb ayb axb".replace(" ", ""))
        x = Token("x")
        y = Token("y")
        proposal = CategoryProposal(members=frozenset({x, y}), name="M0", positions=(1, 4, 7))
        new_state = EditApplier().apply_category(state, proposal)
        self.assertEqual(tuple(t.value for t in new_state.corpus.symbols), tuple("axbaybaxb"))
        occurrences = [e for e in new_state.parse if isinstance(e, CategoryOccurrence)]
        self.assertEqual([str(e) for e in occurrences], ["M0[x]", "M0[y]", "M0[x]"])
        self.assertEqual(CLA.simple(max_iterations=0).fit_symbols(tuple(t.value for t in new_state.corpus.symbols)).expand(), tuple("axbaybaxb"))

    def test_deterministic_same_input_same_seed(self):
        a = CLA.simple(max_iterations=8, seed=42).fit_symbols("abcabcxyzxyzabcabc")
        b = CLA.simple(max_iterations=8, seed=42).fit_symbols("abcabcxyzxyzabcabc")
        self.assertEqual(a.expand(), b.expand())
        self.assertEqual(a.state.parse, b.state.parse)
        self.assertEqual(a.state.grammar, b.state.grammar)
        self.assertEqual(a.state.history, b.state.history)
        self.assertEqual(a.state.score, b.state.score)


if __name__ == "__main__":
    unittest.main()
