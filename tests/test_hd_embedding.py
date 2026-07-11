import json
import subprocess
import sys
import unittest

from chaoslang import CLA
from chaoslang.benchmarks.attractors import high_dimensional_lift, lorenz63
from chaoslang.embedding import intrinsic_dimension_participation_ratio, tica_vamp_kinetic_map
from chaoslang.evaluation import heldout_next_symbol_log_loss, shuffled_surrogate, surrogate_excess_compression
from chaoslang.symbolization import kmeans_microstate_symbols, kmeans_microstates

try:
    from chaoslang.deeptime_backend import (
        deeptime_kinetic_map,
        deeptime_kmeans_microstate_symbols,
        deeptime_implied_timescales,
        deeptime_pcca_memberships,
    )
    DEEPTIME_AVAILABLE = True
except ImportError:
    DEEPTIME_AVAILABLE = False


class AdaptiveSymbolizationTests(unittest.TestCase):
    def test_kmeans_microstates_choose_alphabet_size_directly(self):
        points = [(0.0, 0.0), (0.1, 0.0), (10.0, 10.0), (10.1, 10.0)]
        result = kmeans_microstates(points, k=2, seed=0)
        symbols = kmeans_microstate_symbols(points, k=2, seed=0)
        self.assertEqual(len(result.assignments), len(points))
        self.assertLessEqual(len(set(symbols)), 2)
        self.assertTrue(all(symbol.startswith("km") for symbol in symbols))

    def test_kmeans_symbols_feed_cla_with_exact_reconstruction(self):
        trajectory = lorenz63(steps=48, discard=8)
        symbols = kmeans_microstate_symbols(trajectory, k=6, seed=1)
        model = CLA.simple(max_iterations=3, miner="suffix_trie").fit_symbols(symbols)
        self.assertEqual(model.expand(), symbols)


class KineticMapTests(unittest.TestCase):
    def test_intrinsic_dimension_spectral_diagnostic(self):
        trajectory = [(float(i), 2.0 * i, 0.0) for i in range(1, 8)]
        estimate = intrinsic_dimension_participation_ratio(trajectory)
        self.assertGreaterEqual(estimate.participation_ratio, 1.0)
        self.assertGreaterEqual(estimate.suggested_dimension, 1)
        self.assertEqual(len(estimate.eigenvalues), 3)

    def test_high_dimensional_lift_preserves_length_and_sets_ambient_dimension(self):
        trajectory = lorenz63(steps=12, discard=4)
        lifted = high_dimensional_lift(trajectory, dimension=8, noise=0.001, seed=7)
        self.assertEqual(len(lifted), len(trajectory))
        self.assertTrue(all(len(point) == 8 for point in lifted))
        self.assertEqual(lifted, high_dimensional_lift(trajectory, dimension=8, noise=0.001, seed=7))

    def test_tica_vamp_kinetic_map_is_low_dimensional_and_directional(self):
        trajectory = lorenz63(steps=40, discard=8)
        result = tica_vamp_kinetic_map(trajectory, dimension=2, lag=2, shrinkage=1e-5)
        self.assertEqual(len(result.coordinates), 38)
        self.assertEqual(len(result.coordinates[0]), 2)
        self.assertEqual(result.input_dimension, 3)
        self.assertEqual(result.lag, 2)
        self.assertEqual(len(result.singular_values), 2)

    def test_tica_kmeans_symbols_feed_cla(self):
        trajectory = lorenz63(steps=48, discard=8)
        kinetic = tica_vamp_kinetic_map(trajectory, dimension=2, lag=1, shrinkage=1e-5)
        symbols = kmeans_microstate_symbols(kinetic.coordinates, k=5, seed=2)
        model = CLA.simple(max_iterations=3, miner="suffix_trie").fit_symbols(symbols)
        self.assertEqual(model.expand(), symbols)


class ValidationHarnessTests(unittest.TestCase):
    def test_shuffle_preserves_marginals(self):
        symbols = tuple("aaabbbcccddd")
        shuffled = shuffled_surrogate(symbols, seed=3)
        self.assertCountEqual(shuffled, symbols)
        self.assertEqual(len(shuffled), len(symbols))

    def test_heldout_next_symbol_log_loss_reports_perplexity(self):
        result = heldout_next_symbol_log_loss(tuple("abcabcabcabc"), train_fraction=0.6, order=1)
        self.assertGreater(result.evaluated_symbols, 0)
        self.assertGreaterEqual(result.perplexity, 1.0)

    def test_surrogate_excess_compression_runs(self):
        symbols = tuple("abcabcabcabc")

        def fit(stream):
            return CLA.simple(max_iterations=2, enable_categories=False).fit_symbols(stream)

        result = surrogate_excess_compression(symbols, runs=2, seed=4, fit=fit)
        self.assertEqual(len(result.surrogate_gain_bits), 2)
        self.assertIsInstance(result.delta_grammar_bits, float)

    def test_cli_supports_tica_kmeans_metrics(self):
        proc = subprocess.run(
            [
                sys.executable, "-m", "chaoslang.benchmarks.m1_controls",
                "--system", "lorenz63", "--steps", "24", "--discard", "4",
                "--lift-dimension", "6", "--lift-noise", "0.001",
                "--symbolizer", "tica-kmeans", "--microstates", "4", "--embedding-dim", "2",
                "--lag", "1", "--iterations", "1", "--heldout", "--surrogates", "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(proc.stdout)
        self.assertEqual(result["symbolizer"], "tica-kmeans")
        self.assertEqual(result["dimension"], 6)
        self.assertEqual(result["source_dimension"], 3)
        self.assertTrue(result["exact_reconstruction"])
        self.assertIn("delta_grammar_bits", result)
        self.assertIn("heldout_next_symbol_perplexity", result)
        self.assertEqual(result["embedding"]["dimension"], 2)


@unittest.skipUnless(DEEPTIME_AVAILABLE, "deeptime not installed")
class DeeptimeBackendTests(unittest.TestCase):
    def test_deeptime_vamp_kinetic_map_on_lorenz63(self):
        traj = lorenz63(steps=80, discard=10)
        lifted = high_dimensional_lift(traj, dimension=50, noise=0.001, seed=0)
        result = deeptime_kinetic_map(lifted, dimension=3, lag=2)
        self.assertEqual(result.backend, "vamp")
        self.assertEqual(len(result.coordinates), 80)
        self.assertEqual(len(result.coordinates[0]), 3)
        self.assertEqual(len(result.singular_values), 3)
        self.assertEqual(len(result.timescales), 3)
        self.assertGreater(len(result.cumulative_kinetic_variance), 0)
        self.assertEqual(result.input_dimension, 50)

    def test_deeptime_tica_reversible(self):
        traj = lorenz63(steps=80, discard=10)
        lifted = high_dimensional_lift(traj, dimension=30, noise=0.001, seed=0)
        result = deeptime_kinetic_map(lifted, dimension=2, lag=2, reversible=True)
        self.assertEqual(result.backend, "tica")
        self.assertEqual(len(result.coordinates[0]), 2)

    def test_deeptime_kmeans_symbols_feed_cla(self):
        traj = lorenz63(steps=48, discard=8)
        lifted = high_dimensional_lift(traj, dimension=20, noise=0.001, seed=0)
        kinetic = deeptime_kinetic_map(lifted, dimension=2, lag=1)
        symbols = deeptime_kmeans_microstate_symbols(kinetic.coordinates, k=5, seed=0)
        model = CLA.simple(max_iterations=3, miner="suffix_trie").fit_symbols(symbols)
        self.assertEqual(model.expand(), symbols)

    def test_deeptime_d200_kinetic_map(self):
        traj = lorenz63(steps=60, discard=10)
        lifted = high_dimensional_lift(traj, dimension=200, noise=0.001, seed=0)
        result = deeptime_kinetic_map(lifted, dimension=3, lag=2)
        self.assertEqual(result.input_dimension, 200)
        self.assertTrue(all(0.0 <= sv <= 1.0 + 1e-6 for sv in result.singular_values))

    def test_deeptime_implied_timescales(self):
        traj = lorenz63(steps=80, discard=10)
        lifted = high_dimensional_lift(traj, dimension=20, noise=0.001, seed=0)
        lags, ts = deeptime_implied_timescales(lifted, lags=[1, 2, 5], dimension=2)
        self.assertEqual(len(lags), 3)
        self.assertEqual(len(ts), 3)
        self.assertTrue(all(len(row) == 2 for row in ts))

    def test_deeptime_pcca_memberships(self):
        traj = lorenz63(steps=80, discard=10)
        lifted = high_dimensional_lift(traj, dimension=20, noise=0.001, seed=0)
        kinetic = deeptime_kinetic_map(lifted, dimension=3, lag=2)
        pcca_result = deeptime_pcca_memberships(kinetic.coordinates, n_microstates=6, n_macrostates=2, lag=2, seed=0)
        self.assertGreater(len(pcca_result.assignments), 0)
        self.assertEqual(len(pcca_result.assignments), len(kinetic.coordinates))

    def test_cli_dt_tica_kmeans(self):
        proc = subprocess.run(
            [
                sys.executable, "-m", "chaoslang.benchmarks.m1_controls",
                "--system", "lorenz63", "--steps", "48", "--discard", "8",
                "--lift-dimension", "20", "--lift-noise", "0.001",
                "--symbolizer", "dt-tica-kmeans", "--microstates", "5", "--embedding-dim", "2",
                "--lag", "1", "--iterations", "1", "--heldout",
            ],
            check=True, capture_output=True, text=True,
        )
        result = json.loads(proc.stdout)
        self.assertEqual(result["symbolizer"], "dt-tica-kmeans")
        self.assertEqual(result["dimension"], 20)
        self.assertTrue(result["exact_reconstruction"])
        self.assertEqual(result["embedding"]["backend"], "vamp")
        self.assertIn("timescales", result["embedding"])


if __name__ == "__main__":
    unittest.main()
