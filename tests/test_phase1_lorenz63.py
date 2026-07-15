import json
import unittest

from chaoslang.benchmarks.attractors import high_dimensional_lift, lorenz63
from chaoslang.benchmarks.phase1_lorenz63 import run


class Phase1Lorenz63Tests(unittest.TestCase):
    def test_lift_is_deterministic_and_has_requested_shape(self):
        source = lorenz63(steps=8, discard=2)
        a = high_dimensional_lift(source, dimension=256, noise=.001, seed=7)
        b = high_dimensional_lift(source, dimension=256, noise=.001, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8)
        self.assertTrue(all(len(point) == 256 for point in a))

    def test_benchmark_output_contains_matched_diagnostics(self):
        result = run(steps=32, discard=8, lift_dimension=16, noise=.001,
                     embedding_dim=2, microstates=4, lag=1, bins=2,
                     iterations=1, surrogates=1, seed=3)
        json.dumps(result)
        self.assertEqual(result["status"], "diagnostic_not_grammar_preservation_claim")
        for name in ("adaptive_tica_microstates", "matched_raw_m1"):
            self.assertTrue(result[name]["exact_reconstruction"])
            self.assertIn("real_minus_shuffled_bits_proxy", result[name])
            self.assertIn("heldout_perplexity", result[name])


if __name__ == "__main__":
    unittest.main()
