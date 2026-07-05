"""Tests for JS-divergence category induction integration."""
import unittest

from chaoslang.api import CLA
from chaoslang.categorization.js_inducer import JSCategoryInducer
from chaoslang.core.types import GrammarState, Token


class JSCategoryInducerIntegrationTests(unittest.TestCase):
    """The JS inducer should produce valid category proposals that preserve
    exact reconstruction when applied through the CLA learner."""

    def test_js_inducer_proposes_overlap_context_tokens(self):
        """Tokens sharing most but not all contexts should cluster with JS."""
        symbols = ["a", "x", "b", "a", "y", "b", "a", "x", "c", "a", "y", "b"]
        state = GrammarState.initial(symbols)
        inducer = JSCategoryInducer(threshold=0.5, min_members=2, min_occurrences=2)
        proposals = inducer.proposals(state)
        self.assertGreaterEqual(len(proposals), 1)
        for p in proposals:
            self.assertGreaterEqual(len(p.members), 2)
            self.assertGreaterEqual(len(p.positions), 2)

    def test_js_inducer_zero_threshold_matches_exact(self):
        """At threshold=0.0, JS inducer should behave like exact signature matching."""
        symbols = ["a", "x", "b", "a", "y", "b", "a", "x", "b", "a", "y", "b"]
        state = GrammarState.initial(symbols)
        js_inducer = JSCategoryInducer(threshold=0.0, min_members=2, min_occurrences=2)
        from chaoslang.categorization.context import ContextCategoryInducer
        exact_inducer = ContextCategoryInducer(min_members=2, min_occurrences=2)
        js_props = js_inducer.proposals(state)
        exact_props = exact_inducer.proposals(state)
        self.assertEqual(len(js_props), len(exact_props))
        for jp, ep in zip(js_props, exact_props):
            self.assertEqual(jp.members, ep.members)

    def test_js_inducer_no_proposals_for_dissimilar_tokens(self):
        """Tokens with completely different contexts should not cluster."""
        symbols = ["a", "x", "b", "c", "y", "d", "a", "x", "b", "c", "y", "d"]
        state = GrammarState.initial(symbols)
        inducer = JSCategoryInducer(threshold=0.01, min_members=2, min_occurrences=2)
        proposals = inducer.proposals(state)
        self.assertEqual(len(proposals), 0)

    def test_cla_with_js_categories_preserves_exact_reconstruction(self):
        """Full CLA learner with JS categories must still reconstruct exactly."""
        symbols = "axbaybaxcaybaxbayb"
        model = CLA.simple(max_iterations=8, seed=42, category_method="js", js_threshold=0.3)
        fitted = model.fit_symbols(symbols)
        self.assertEqual(fitted.expand(), tuple(symbols))

    def test_cla_with_js_categories_on_attractor_symbols(self):
        """JS categories should work with pre-tokenized attractor symbols."""
        symbols = ["0", "1", "2", "0", "1", "2", "0", "1", "3", "0", "1", "2"]
        model = CLA.simple(max_iterations=8, seed=7, category_method="js", js_threshold=0.3)
        fitted = model.fit_symbols(symbols)
        self.assertEqual(fitted.expand(), tuple(symbols))

    def test_cla_category_method_invalid_raises(self):
        with self.assertRaises(ValueError):
            CLA.simple(category_method="bogus")

    def test_js_inducer_deterministic(self):
        """Same input should produce same proposals every time."""
        symbols = ["a", "x", "b", "a", "y", "b", "a", "x", "b", "a", "y", "b"]
        state = GrammarState.initial(symbols)
        inducer = JSCategoryInducer(threshold=0.3, min_members=2, min_occurrences=2)
        p1 = inducer.proposals(state)
        p2 = inducer.proposals(state)
        self.assertEqual(p1, p2)

    def test_cla_js_vs_exact_same_reconstruction(self):
        """Both methods should produce exact reconstruction on the same input."""
        symbols = ["a", "x", "b", "a", "y", "b", "a", "x", "b", "a", "y", "b"] * 3
        exact_model = CLA.simple(seed=0, category_method="exact")
        js_model = CLA.simple(seed=0, category_method="js", js_threshold=0.3)
        exact_fitted = exact_model.fit_symbols(symbols)
        js_fitted = js_model.fit_symbols(symbols)
        self.assertEqual(exact_fitted.expand(), tuple(symbols))
        self.assertEqual(js_fitted.expand(), tuple(symbols))


if __name__ == "__main__":
    unittest.main()
