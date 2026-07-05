import json
import subprocess
import sys
import unittest

from chaoslang import CLA
from chaoslang.benchmarks.attractors import lorenz96, mackey_glass, m1_symbolize


class MackeyGlassTests(unittest.TestCase):
    def test_deterministic_same_seed(self):
        a = mackey_glass(steps=64, discard=32)
        b = mackey_glass(steps=64, discard=32)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_values_in_reasonable_range(self):
        traj = mackey_glass(steps=200, discard=100, dt=0.1, tau=17)
        self.assertTrue(all(0.0 < x < 2.0 for x in traj))

    def test_m1_symbolize_scalar_trajectory(self):
        traj = mackey_glass(steps=48, discard=16)
        symbols = m1_symbolize(traj, bins=4)
        self.assertEqual(len(symbols), 48)
        self.assertTrue(all(s.startswith("m1:d0") for s in symbols))
        self.assertGreater(len(set(symbols)), 1)

    def test_cla_reconstructs_mackey_glass_symbols(self):
        traj = mackey_glass(steps=64, discard=32)
        symbols = m1_symbolize(traj, bins=4)
        model = CLA.simple(max_iterations=4).fit_symbols(symbols)
        self.assertEqual(model.expand(), symbols)

    def test_tau_must_be_positive(self):
        with self.assertRaises(ValueError):
            mackey_glass(tau=0)


class Lorenz96Tests(unittest.TestCase):
    def test_deterministic_and_correct_dimension(self):
        init = (8.0, 8.0, 8.0, 8.0, 8.01)
        a = lorenz96(steps=40, discard=8, initial=init)
        b = lorenz96(steps=40, discard=8, initial=init)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 40)
        self.assertTrue(all(len(p) == 5 for p in a))

    def test_dimension_from_initial(self):
        traj = lorenz96(steps=20, discard=4, initial=(5.0, 5.0, 5.0, 5.01))
        self.assertTrue(all(len(p) == 4 for p in traj))

    def test_m1_symbolize_vector_trajectory(self):
        traj = lorenz96(steps=64, discard=200, initial=(8.0, 8.0, 8.0, 8.0, 8.01, 8.0))
        symbols = m1_symbolize(traj, bins=3)
        self.assertEqual(len(symbols), 64)
        self.assertTrue(all(s.startswith("m1:") for s in symbols))
        self.assertGreater(len(set(symbols)), 1)

    def test_cla_reconstructs_lorenz96_symbols(self):
        traj = lorenz96(steps=64, discard=200, initial=(8.0, 8.0, 8.0, 8.0, 8.01))
        symbols = m1_symbolize(traj, bins=3)
        model = CLA.simple(max_iterations=4).fit_symbols(symbols)
        self.assertEqual(model.expand(), symbols)

    def test_cli_supports_mackey_glass(self):
        proc = subprocess.run(
            [sys.executable, "-m", "chaoslang.benchmarks.m1_controls",
             "--system", "mackey-glass", "--steps", "32", "--discard", "8",
             "--bins", "4", "--iterations", "2"],
            check=True, capture_output=True, text=True,
        )
        result = json.loads(proc.stdout)
        self.assertEqual(result["system"], "mackey-glass")
        self.assertEqual(result["symbol_count"], 32)
        self.assertTrue(result["exact_reconstruction"])

    def test_cli_supports_lorenz96(self):
        proc = subprocess.run(
            [sys.executable, "-m", "chaoslang.benchmarks.m1_controls",
             "--system", "lorenz96", "--steps", "32", "--discard", "200",
             "--bins", "3", "--iterations", "2"],
            check=True, capture_output=True, text=True,
        )
        result = json.loads(proc.stdout)
        self.assertEqual(result["system"], "lorenz96")
        self.assertEqual(result["symbol_count"], 32)
        self.assertTrue(result["exact_reconstruction"])


if __name__ == "__main__":
    unittest.main()
