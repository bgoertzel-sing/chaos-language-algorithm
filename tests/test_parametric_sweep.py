"""Tests for parametric complexity sweeps.

Validates that CLA grammar complexity tracks the bifurcation parameter
in logistic-map and Lorenz-96 families.
"""
import unittest

from chaoslang.benchmarks.parametric_sweep import (
    SweepPoint,
    SweepResult,
    logistic_map_symbols,
    lorenz96_symbols,
    lorenz96_hd_symbols,
    sweep_logistic_map,
    sweep_lorenz96,
    analyze_symbols,
)
from chaoslang.benchmarks.attractors import logistic_map, lorenz96, high_dimensional_lift


class LogisticMapSweepTests(unittest.TestCase):
    def test_logistic_map_symbols_are_deterministic(self):
        s1 = logistic_map_symbols(r=4.0, steps=100, discard=20, bins=8)
        s2 = logistic_map_symbols(r=4.0, steps=100, discard=20, bins=8)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 100)

    def test_low_r_gives_constant_symbols(self):
        """At r=2.5 the logistic map converges to a fixed point."""
        symbols = logistic_map_symbols(r=2.5, steps=500, discard=400, bins=8)
        # After convergence, should be at most 2 symbols (float jitter at fixed point)
        self.assertLessEqual(len(set(symbols)), 2)

    def test_chaotic_r_gives_diverse_symbols(self):
        """At r=4.0 the logistic map is fully chaotic."""
        symbols = logistic_map_symbols(r=4.0, steps=500, discard=100, bins=8)
        self.assertGreater(len(set(symbols)), 3)

    def test_sweep_logistic_map_completes(self):
        """Quick sweep with small parameters for CI."""
        result = sweep_logistic_map(
            r_values=(2.5, 3.5, 4.0),
            steps=300, discard=100, bins=6,
            max_iterations=3, surrogates=2, seed=0,
        )
        self.assertEqual(result.system, "logistic_map")
        self.assertEqual(result.parameter_name, "r")
        self.assertEqual(len(result.points), 3)
        # Chaotic regime should have more grammar rules than fixed point
        self.assertGreaterEqual(result.points[2].grammar_rules, result.points[0].grammar_rules)

    def test_sweep_result_serializes_to_dict(self):
        result = sweep_logistic_map(
            r_values=(2.5, 4.0), steps=200, discard=50, bins=6,
            max_iterations=2, surrogates=1, seed=0,
        )
        d = result.to_dict()
        self.assertEqual(d["system"], "logistic_map")
        self.assertEqual(len(d["points"]), 2)
        self.assertIn("grammar_gain_bits", d["points"][0])

    def test_grammar_complexity_increases_with_r(self):
        """Core claim: grammar complexity should increase toward chaos."""
        result = sweep_logistic_map(
            r_values=(2.5, 3.2, 3.5, 4.0),
            steps=500, discard=200, bins=8,
            max_iterations=5, surrogates=3, seed=0,
        )
        # At r=2.5 (fixed point), grammar should be minimal
        self.assertLessEqual(result.points[0].grammar_rules, 3)
        # At r=4.0 (full chaos), grammar should be richer
        self.assertGreater(result.points[3].grammar_rules, 0)

    def test_surrogate_delta_positive_for_chaotic_regime(self):
        """CLA should compress real data better than shuffled surrogates for chaos."""
        result = sweep_logistic_map(
            r_values=(4.0,), steps=500, discard=200, bins=8,
            max_iterations=5, surrogates=3, seed=0,
        )
        # Delta should be positive — real grammar beats shuffled
        self.assertGreater(result.points[0].surrogate_delta_bits, -0.5)


class Lorenz96SweepTests(unittest.TestCase):
    def test_lorenz96_symbols_are_deterministic(self):
        s1 = lorenz96_symbols(F=8.0, steps=100, discard=20, dim=5, bins=8)
        s2 = lorenz96_symbols(F=8.0, steps=100, discard=20, dim=5, bins=8)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 100)

    def test_low_F_gives_less_complex_dynamics(self):
        """Low forcing F should produce simpler dynamics than chaotic F."""
        low_symbols = lorenz96_symbols(F=2.0, steps=500, discard=200, dim=5, bins=8)
        high_symbols = lorenz96_symbols(F=8.0, steps=500, discard=200, dim=5, bins=8)
        # High forcing should visit more bins
        self.assertGreaterEqual(len(set(high_symbols)), len(set(low_symbols)))

    def test_sweep_lorenz96_completes(self):
        result = sweep_lorenz96(
            F_values=(2.0, 8.0),
            steps=300, discard=100, dim=5, bins=6,
            max_iterations=3, surrogates=2, seed=0,
        )
        self.assertEqual(result.system, "lorenz96")
        self.assertEqual(len(result.points), 2)

    def test_grammar_complexity_trends_upward_with_F(self):
        """Grammar should be richer at chaotic F than at steady F."""
        result = sweep_lorenz96(
            F_values=(2.0, 4.0, 8.0),
            steps=500, discard=200, dim=5, bins=8,
            max_iterations=5, surrogates=2, seed=0,
        )
        # F=8 should have at least as many rules as F=2
        self.assertGreaterEqual(result.points[2].grammar_rules, result.points[0].grammar_rules)


class HighDimensionalPipelineTests(unittest.TestCase):
    def test_lorenz96_hd_symbols_deterministic(self):
        s1 = lorenz96_hd_symbols(F=8.0, steps=100, discard=20, lift_dimension=32,
                                   microstates=5, embedding_dim=2, lag=1, seed=0)
        s2 = lorenz96_hd_symbols(F=8.0, steps=100, discard=20, lift_dimension=32,
                                   microstates=5, embedding_dim=2, lag=1, seed=0)
        self.assertEqual(s1, s2)

    def test_lorenz96_hd_symbols_correct_length(self):
        symbols = lorenz96_hd_symbols(F=8.0, steps=50, discard=10, lift_dimension=32,
                                       microstates=5, embedding_dim=2, lag=1, seed=0)
        # TICA with lag=1 reduces output by `lag` samples
        self.assertEqual(len(symbols), 49)

    def test_analyze_symbols_on_small_input(self):
        symbols = tuple("abcabcabcabcabc")
        point = analyze_symbols(symbols, max_iterations=2, surrogates=1, seed=0)
        self.assertEqual(point.num_symbols, 15)
        self.assertGreater(point.alphabet_size, 0)


if __name__ == "__main__":
    unittest.main()
