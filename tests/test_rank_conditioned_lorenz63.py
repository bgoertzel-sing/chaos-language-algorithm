import unittest
from unittest.mock import patch

import numpy as np

from chaoslang.benchmarks.rank_conditioned_lorenz63 import (
    circular_block_surrogate, deterministic_split, fit_kinetic, fit_pca,
    fit_symbols, heldout_ngram_loss, kinetic_transform, pca_transform,
    run_experiment,
)


class RankConditionedLorenz63Tests(unittest.TestCase):
    def test_split_is_deterministic_contiguous_and_gapped(self):
        first = deterministic_split(2048, gap=16)
        second = deterministic_split(2048, gap=16)
        self.assertEqual(first, second)
        self.assertEqual((first.train.start, first.train.stop), (0, 1228))
        self.assertEqual(first.validation.start - first.train.stop, 16)
        self.assertEqual(first.test.start - first.validation.stop, 16)

    def test_pca_and_kmeans_fit_train_only(self):
        train = np.arange(60, dtype=float).reshape(20, 3)
        heldout = np.arange(30, dtype=float).reshape(10, 3) + 10000
        pca = fit_pca(train, 2)
        np.testing.assert_allclose(pca.mean, train.mean(axis=0))
        with patch("chaoslang.benchmarks.rank_conditioned_lorenz63.KMeans.fit",
                   autospec=True, side_effect=AssertionError("fit called")) as mocked:
            # Prediction is deliberately not reached: this asserts one attempted fit,
            # and the API takes only train as the fit argument.
            with self.assertRaisesRegex(AssertionError, "fit called"):
                fit_symbols(pca_transform(pca, train), pca_transform(pca, heldout),
                            k=2, seed=7, prefix="x")
            self.assertEqual(mocked.call_count, 1)
            np.testing.assert_array_equal(mocked.call_args.args[1], pca_transform(pca, train))

    def test_rank_conditioning_and_raw_singular_values_fail_closed(self):
        rng = np.random.default_rng(4)
        latent = rng.normal(size=(256, 3))
        data = np.column_stack((latent, latent[:, 0], latent[:, 1]))
        model = fit_kinetic(data, lag=1, dimension=3, method="vamp",
                            ridge_scale=1e-5, max_condition=1e4)
        self.assertLessEqual(model.condition_00, 1e4 * 1.000001)
        self.assertLessEqual(model.condition_11, 1e4 * 1.000001)
        self.assertTrue(all(0 <= value <= 1 + 1e-8 for value in model.singular_values))
        with self.assertRaisesRegex(ValueError, "raw singular value outside"):
            fit_kinetic(data, lag=1, dimension=3, method="vamp",
                        ridge_scale=1e-14, singular_tolerance=-1.0)

    def test_transform_does_not_refit(self):
        rng = np.random.default_rng(9)
        train = rng.normal(size=(100, 3))
        model = fit_kinetic(train, lag=2, dimension=2, method="tica")
        before = model.left.copy()
        transformed = kinetic_transform(model, rng.normal(size=(11, 3)))
        self.assertEqual(transformed.shape, (11, 2))
        np.testing.assert_array_equal(model.left, before)

    def test_surrogate_and_metric_are_deterministic_and_matched(self):
        symbols = tuple("aabbccddeeff" * 4)
        self.assertEqual(circular_block_surrogate(symbols, block_length=4, seed=8),
                         circular_block_surrogate(symbols, block_length=4, seed=8))
        self.assertEqual(sorted(symbols), sorted(circular_block_surrogate(symbols, block_length=4, seed=8)))
        loss = heldout_ngram_loss(tuple("abc" * 10), tuple("abc" * 3))
        self.assertEqual(loss["evaluated_symbols"], 9)
        with self.assertRaisesRegex(ValueError, "outside train-fitted alphabet"):
            heldout_ngram_loss(tuple("abc" * 10), ("z",))

    def test_tiny_end_to_end_schema_has_matched_k_controls(self):
        result = run_experiment(steps=96, discard=64, lift_dimension=12,
            trajectory_seeds=(11,), ranks=(3,), dimensions=(2,), lags=(1,),
            microstates=(4,), block_lengths=(4,), surrogates=1, iterations=1)
        methods = result["trajectories"][0]["methods"]
        self.assertEqual({row["method"] for row in methods},
                         {"vamp", "tica", "pca_lift_kmeans", "direct_xyz_kmeans"})
        self.assertTrue(all(row["k"] == 4 for row in methods))
        self.assertEqual(result["config"]["singular_value_policy"], "raw_fail_closed_no_clipping")


if __name__ == "__main__":
    unittest.main()
