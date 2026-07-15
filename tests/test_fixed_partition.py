"""Tests for deterministic reference-frozen trajectory partitions."""
from dataclasses import FrozenInstanceError
import unittest

from chaoslang.symbolization import (
    FixedPartitionSymbolizer,
    KMeansSymbolizer,
    fixed_partition_symbols,
    kmeans_microstates,
)


class FixedPartitionTests(unittest.TestCase):
    def test_determinism_with_explicit_bounds(self):
        trajectory = (0.0, 0.25, 0.5, 0.75, 1.0)
        first = fixed_partition_symbols(trajectory, bounds=((0.0, 1.0),), bins=4)
        second = fixed_partition_symbols(trajectory, bounds=((0.0, 1.0),), bins=4)
        self.assertEqual(first, second)

    def test_shared_partition_is_independent_of_other_trajectories(self):
        symbolizer = FixedPartitionSymbolizer(((0.0, 10.0),), (5,))
        first = symbolizer.symbolize((1.0, 5.0, 9.0))
        second = symbolizer.symbolize((2.0, 5.0, 7.0))
        self.assertEqual(first[1], second[1])
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[2], second[2])

    def test_out_of_range_values_are_clamped(self):
        symbolizer = FixedPartitionSymbolizer(((0.0, 1.0),), (4,))
        self.assertEqual(
            symbolizer.symbolize((-100.0, 0.0, 1.0, 100.0)),
            ("fp0", "fp0", "fp3", "fp3"),
        )

    def test_scalar_trajectory(self):
        symbolizer = FixedPartitionSymbolizer.from_reference((0.0, 1.0), bins=2)
        self.assertEqual(symbolizer.symbolize((0.1, 0.9)), ("fp0", "fp1"))

    def test_vector_trajectory_and_axis_labels(self):
        symbolizer = FixedPartitionSymbolizer(
            ((0.0, 1.0), (10.0, 20.0)), (2, 5), axis_labels=("x", "y")
        )
        self.assertEqual(
            symbolizer.symbolize(((0.25, 10.0), (0.75, 20.0))),
            ("fpx0_y0", "fpx1_y4"),
        )

    def test_reference_bounds_cover_full_reference_range(self):
        reference = ((-2.0, 10.0), (3.0, 20.0), (1.0, 15.0))
        symbolizer = FixedPartitionSymbolizer.from_reference(reference, bins=(2, 4))
        self.assertEqual(symbolizer.bounds, ((-2.0, 3.0), (10.0, 20.0)))
        self.assertEqual(len(symbolizer.symbolize(reference)), len(reference))

    def test_quantile_bounds_are_robust_to_outliers(self):
        reference = tuple(range(100)) + (1_000_000,)
        ordinary = FixedPartitionSymbolizer.from_reference(reference, bins=4)
        robust = FixedPartitionSymbolizer.from_quantiles(
            reference, bins=4, quantile_range=(0.01, 0.99)
        )
        self.assertEqual(ordinary.bounds[0][1], 1_000_000.0)
        self.assertLess(robust.bounds[0][1], 1_000_000.0)
        self.assertEqual(robust.symbolize((1_000_000,)), ("fp3",))

    def test_symbolizer_is_immutable_and_application_does_not_change_it(self):
        symbolizer = FixedPartitionSymbolizer(((0.0, 1.0),), (4,))
        original = (symbolizer.bounds, symbolizer.bins, symbolizer.prefix)
        symbolizer.symbolize((-10.0, 10.0))
        self.assertEqual(original, (symbolizer.bounds, symbolizer.bins, symbolizer.prefix))
        with self.assertRaises(FrozenInstanceError):
            symbolizer.prefix = "changed"  # type: ignore[misc]

    def test_kmeans_partition_moves_but_fixed_partition_does_not(self):
        first = (0.0, 1.0, 2.0, 9.0, 10.0)
        second = (0.0, 8.0, 9.0, 10.0, 11.0)
        self.assertNotEqual(
            kmeans_microstates(first, k=2).centers,
            kmeans_microstates(second, k=2).centers,
        )
        symbolizer = FixedPartitionSymbolizer(((0.0, 10.0),), (2,))
        bounds_before = symbolizer.bounds
        self.assertEqual(symbolizer.symbolize((0.0, 10.0)), ("fp0", "fp1"))
        symbolizer.symbolize(first)
        symbolizer.symbolize(second)
        self.assertEqual(symbolizer.bounds, bounds_before)
        self.assertEqual(symbolizer.symbolize((0.0, 10.0)), ("fp0", "fp1"))

    def test_fitted_kmeans_symbolizer_does_not_refit_on_test_data(self):
        reference = ((0.0, 0.0), (0.1, 0.1), (9.9, 10.0), (10.0, 9.9))
        symbolizer = KMeansSymbolizer.fit(reference, k=2, seed=3)
        centers = symbolizer.centers

        first = symbolizer.symbolize(((0.2, 0.2), (9.8, 9.8)))
        second = symbolizer.symbolize(((100.0, 100.0), (0.2, 0.2)))

        self.assertNotEqual(first[0], first[1])
        self.assertEqual(first[0], second[1])
        self.assertEqual(symbolizer.centers, centers)

    def test_fitted_kmeans_symbolizer_rejects_wrong_dimension(self):
        symbolizer = KMeansSymbolizer.fit(((0.0, 0.0), (1.0, 1.0)), k=2)
        with self.assertRaisesRegex(ValueError, "dimension"):
            symbolizer.symbolize((0.5,))


if __name__ == "__main__":
    unittest.main()
