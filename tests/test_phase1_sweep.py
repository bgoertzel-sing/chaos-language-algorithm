import json
import math
import tempfile
import unittest
from pathlib import Path

from chaoslang.benchmarks.phase1_sweep import RUN_FIELDS, aggregate_rows, main, run_sweep
from chaoslang.deeptime_backend import validate_kinetic_diagnostics


class Phase1SweepTests(unittest.TestCase):
    def test_reduced_sweep_is_deterministic_and_has_stable_schema(self):
        kwargs = dict(steps=28, discard=4, lift_dimension=8, noise=.001,
                      seeds=(7,), embedding_dims=(2,), microstates=(4,), lags=(1,),
                      iterations=1, surrogates=1)
        first = run_sweep(**kwargs)
        second = run_sweep(**kwargs)
        self.assertEqual(first["run_count"], 4)
        self.assertEqual(first["schema_version"], 2)
        self.assertIn("not_cla_predictive", first["metric_labels"]["heldout_perplexity"])
        for left, right in zip(first["runs"], second["runs"]):
            self.assertEqual(set(left), set(RUN_FIELDS))
            for ignored in ("fit_wall_time_seconds", "configuration_wall_time_seconds"):
                left = {k: v for k, v in left.items() if k != ignored}
                right = {k: v for k, v in right.items() if k != ignored}
            self.assertEqual(left, right)
            self.assertTrue(left["exact_reconstruction"])

    def test_recorded_near_unit_values_fail_closed_without_clamping(self):
        singular_values = (1.0000000000000089, 0.9999999999999762, 0.9999646317546044)
        timescales = (-112589990684262.9, 42089716143649.0, 28273.449943226755)
        valid, status, issues = validate_kinetic_diagnostics(singular_values, timescales)
        self.assertFalse(valid)
        self.assertEqual(status, "invalid_numerical_diagnostics")
        self.assertIn("singular_value_near_one_timescale_undefined", issues)
        self.assertIn("nonpositive_timescale", issues)
        self.assertIn("implausibly_large_timescale", issues)
        self.assertEqual(singular_values[0], 1.0000000000000089)

    def test_nonfinite_out_of_range_and_backend_warning_fail_closed(self):
        valid, _, issues = validate_kinetic_diagnostics(
            (1.0001, float("nan")), (float("inf"),), backend_warnings=("BackendWarning: unstable",)
        )
        self.assertFalse(valid)
        self.assertIn("BackendWarning: unstable", issues)
        self.assertIn("singular_value_outside_valid_range", issues)
        self.assertIn("non_finite_singular_value", issues)
        self.assertIn("non_finite_timescale", issues)

    def test_aggregation_mean_sd_interval_and_robustness(self):
        rows = [{"method": "deeptime_vamp_kmeans", "embedding_dim": 3,
                 "microstates": 8, "lag": 1, "real_minus_shuffled_bits_proxy": value,
                 "heldout_perplexity": perplexity}
                for value, perplexity in ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))]
        result = [r for r in aggregate_rows(rows) if r["method"] == "deeptime_vamp_kmeans"][0]
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["real_minus_shuffled_bits_proxy_mean"], 3.0)
        self.assertAlmostEqual(result["real_minus_shuffled_bits_proxy_std"], 2.0)
        self.assertAlmostEqual(result["heldout_perplexity_ci95_low"], 4 - 1.96 * 2 / math.sqrt(3))
        self.assertTrue(result["all_positive_real_minus_shuffled"])

    def test_cli_writes_json_and_csv_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            code = main(["--output-dir", directory, "--steps", "24", "--discard", "4",
                         "--lift-dimension", "8", "--seeds", "9", "--embedding-dims", "2",
                         "--microstates", "4", "--lags", "1", "--iterations", "1", "--surrogates", "1"])
            self.assertEqual(code, 0)
            for name in ("results.json", "runs.csv", "aggregates.csv"):
                self.assertTrue((Path(directory) / name).is_file())
            result = json.loads((Path(directory) / "results.json").read_text())
            self.assertEqual(result["run_count"], 4)
            self.assertNotIn("\r\n", (Path(directory) / "runs.csv").read_text())


if __name__ == "__main__":
    unittest.main()
