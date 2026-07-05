import json
import subprocess
import sys
import unittest

from chaoslang import CLA
from chaoslang.benchmarks.attractors import lorenz63, m1_symbolize, rossler


class M1ControlTests(unittest.TestCase):
    def test_lorenz63_is_deterministic_and_symbolizes_to_fixed_partition(self):
        a = lorenz63(steps=24, discard=4)
        b = lorenz63(steps=24, discard=4)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertTrue(all(len(point) == 3 for point in a))

        symbols = m1_symbolize(a, bins=(3, 3, 3), bounds=((-20.0, 20.0), (-30.0, 30.0), (0.0, 60.0)))
        self.assertEqual(len(symbols), 24)
        self.assertTrue(all(symbol.startswith("m1:d0") for symbol in symbols))
        self.assertEqual(symbols, m1_symbolize(a, bins=(3, 3, 3), bounds=((-20.0, 20.0), (-30.0, 30.0), (0.0, 60.0))))

    def test_rossler_control_is_distinct_and_cla_reconstructs_m1_symbols(self):
        trajectory = rossler(steps=40, discard=8)
        symbols = m1_symbolize(trajectory, bins=4)
        self.assertEqual(len(symbols), 40)
        self.assertGreater(len(set(symbols)), 1)

        model = CLA.simple(max_iterations=4).fit_symbols(symbols)
        self.assertEqual(model.expand(), symbols)

    def test_m1_symbolize_clamps_explicit_bounds(self):
        symbols = m1_symbolize([(-1.0, 2.0), (0.5, 10.0)], bins=(2, 3), bounds=((0.0, 1.0), (0.0, 3.0)))
        self.assertEqual(symbols, ("m1:d00|d12", "m1:d01|d12"))

    def test_m1_controls_cli_emits_json_summary(self):
        proc = subprocess.run(
            [sys.executable, "-m", "chaoslang.benchmarks.m1_controls", "--system", "lorenz63", "--steps", "12", "--discard", "2", "--bins", "3", "--iterations", "2"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(proc.stdout)
        self.assertEqual(result["system"], "lorenz63")
        self.assertEqual(result["symbol_count"], 12)
        self.assertTrue(result["exact_reconstruction"])


if __name__ == "__main__":
    unittest.main()
