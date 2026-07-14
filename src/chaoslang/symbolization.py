"""Dependency-free trajectory symbolizers for CLA.

This module keeps the core dependency-free.  The first adaptive symbolizer is a
small deterministic k-means implementation that chooses alphabet size directly
instead of creating a fixed rectangular b^D grid.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

Point = tuple[float, ...]


@dataclass(frozen=True)
class FixedPartitionSymbolizer:
    """Deterministic fixed-partition trajectory symbolizer.

    Bin edges are frozen at construction time and never recomputed. Identical
    input always produces identical output, regardless of what other
    trajectories are processed.
    """

    bounds: tuple[tuple[float, float], ...]
    bins: tuple[int, ...]
    prefix: str = "fp"
    axis_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.bounds:
            raise ValueError("bounds must contain at least one dimension")
        if len(self.bounds) != len(self.bins):
            raise ValueError("bounds and bins must have the same dimension")
        if self.axis_labels is not None and len(self.axis_labels) != len(self.bounds):
            raise ValueError("axis_labels must have the same dimension as bounds")
        for lo, hi in self.bounds:
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError("bounds must be finite")
            if hi < lo:
                raise ValueError("each upper bound must be at least its lower bound")
        if any(count < 1 for count in self.bins):
            raise ValueError("bin counts must be at least 1")

    @classmethod
    def from_reference(
        cls,
        reference_trajectory: Sequence[Sequence[float] | float],
        *,
        bins: int | Sequence[int] = 4,
        prefix: str = "fp",
        axis_labels: Sequence[str] | None = None,
    ) -> FixedPartitionSymbolizer:
        """Infer per-dimension min/max bounds from a reference and freeze them."""

        points = coerce_points(reference_trajectory)
        if not points:
            raise ValueError("reference trajectory must not be empty")
        dims = len(points[0])
        bounds = tuple(
            (min(point[axis] for point in points), max(point[axis] for point in points))
            for axis in range(dims)
        )
        return cls(bounds, _coerce_bins(bins, dims), prefix, _coerce_labels(axis_labels))

    @classmethod
    def from_quantiles(
        cls,
        reference_trajectory: Sequence[Sequence[float] | float],
        *,
        bins: int | Sequence[int] = 4,
        quantile_range: tuple[float, float] = (0.01, 0.99),
        prefix: str = "fp",
        axis_labels: Sequence[str] | None = None,
    ) -> FixedPartitionSymbolizer:
        """Infer robust quantile bounds from a reference and freeze them."""

        points = coerce_points(reference_trajectory)
        if not points:
            raise ValueError("reference trajectory must not be empty")
        q_lo, q_hi = quantile_range
        if not 0.0 <= q_lo < q_hi <= 1.0:
            raise ValueError("quantile_range must satisfy 0 <= lo < hi <= 1")
        dims = len(points[0])
        bounds = tuple(
            (
                _linear_quantile(sorted(point[axis] for point in points), q_lo),
                _linear_quantile(sorted(point[axis] for point in points), q_hi),
            )
            for axis in range(dims)
        )
        return cls(bounds, _coerce_bins(bins, dims), prefix, _coerce_labels(axis_labels))

    def symbolize(
        self, trajectory: Sequence[Sequence[float] | float]
    ) -> tuple[str, ...]:
        """Map trajectory points to symbols using this frozen partition."""

        points = coerce_points(trajectory)
        if points and len(points[0]) != len(self.bounds):
            raise ValueError("trajectory dimension does not match fixed partition")
        symbols: list[str] = []
        for point in points:
            indices = tuple(
                _fixed_bin(value, lo, hi, count)
                for value, (lo, hi), count in zip(point, self.bounds, self.bins)
            )
            if self.axis_labels is None:
                suffix = "_".join(str(index) for index in indices)
            else:
                suffix = "_".join(
                    f"{label}{index}" for label, index in zip(self.axis_labels, indices)
                )
            symbols.append(f"{self.prefix}{suffix}")
        return tuple(symbols)


def fixed_partition_symbols(
    trajectory: Sequence[Sequence[float] | float],
    *,
    bounds: Sequence[tuple[float, float]] | None = None,
    bins: int | Sequence[int] = 4,
    prefix: str = "fp",
    reference: Sequence[Sequence[float] | float] | None = None,
    quantile_range: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[str, ...]:
    """One-shot or reference-frozen fixed-partition symbolization.

    ``seed`` is accepted and ignored for API compatibility with stochastic
    symbolizers. Explicit ``bounds`` take precedence over ``reference``.
    """

    del seed
    if not trajectory and bounds is None and reference is None:
        return ()
    if bounds is not None:
        frozen_bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        symbolizer = FixedPartitionSymbolizer(
            frozen_bounds, _coerce_bins(bins, len(frozen_bounds)), prefix
        )
    else:
        source = trajectory if reference is None else reference
        if quantile_range is None:
            symbolizer = FixedPartitionSymbolizer.from_reference(
                source, bins=bins, prefix=prefix
            )
        else:
            symbolizer = FixedPartitionSymbolizer.from_quantiles(
                source, bins=bins, quantile_range=quantile_range, prefix=prefix
            )
    return symbolizer.symbolize(trajectory)


@dataclass(frozen=True)
class KMeansResult:
    centers: tuple[Point, ...]
    assignments: tuple[int, ...]
    inertia: float
    iterations: int


def coerce_points(trajectory: Sequence[Sequence[float] | float]) -> tuple[Point, ...]:
    """Return a tuple of float points from scalar or vector trajectory samples."""

    out: list[Point] = []
    for item in trajectory:
        if isinstance(item, (int, float)):
            out.append((float(item),))
        else:
            out.append(tuple(float(v) for v in item))
    if out:
        dims = len(out[0])
        if dims == 0:
            raise ValueError("points must have at least one dimension")
        if any(len(p) != dims for p in out):
            raise ValueError("all trajectory points must have the same dimension")
    return tuple(out)


def kmeans_microstates(
    trajectory: Sequence[Sequence[float] | float],
    *,
    k: int = 32,
    max_iterations: int = 50,
    seed: int = 0,
) -> KMeansResult:
    """Cluster trajectory points into deterministic microstates.

    Initialization is farthest-first from a seed-selected starting point.  This
    avoids optional dependencies and is deterministic for a given seed.
    """

    points = coerce_points(trajectory)
    if k < 1:
        raise ValueError("k must be at least 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if not points:
        return KMeansResult(centers=(), assignments=(), inertia=0.0, iterations=0)
    if k > len(points):
        raise ValueError("k cannot exceed the number of points")

    centers = _initial_centers(points, k, seed)
    assignments = tuple(-1 for _ in points)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        new_assignments = tuple(_nearest_center(point, centers) for point in points)
        new_centers = _recompute_centers(points, new_assignments, centers)
        if new_assignments == assignments or _same_centers(new_centers, centers):
            assignments = new_assignments
            centers = new_centers
            break
        assignments = new_assignments
        centers = new_centers

    inertia = sum(_squared_distance(point, centers[idx]) for point, idx in zip(points, assignments))
    return KMeansResult(centers=tuple(centers), assignments=assignments, inertia=inertia, iterations=iterations)


def kmeans_microstate_symbols(
    trajectory: Sequence[Sequence[float] | float],
    *,
    k: int = 32,
    prefix: str = "km",
    max_iterations: int = 50,
    seed: int = 0,
) -> tuple[str, ...]:
    """Return one low-cardinality symbol per trajectory point."""

    result = kmeans_microstates(trajectory, k=k, max_iterations=max_iterations, seed=seed)
    return tuple(f"{prefix}{idx}" for idx in result.assignments)


def _coerce_bins(bins: int | Sequence[int], dims: int) -> tuple[int, ...]:
    if isinstance(bins, int):
        return (bins,) * dims
    result = tuple(int(count) for count in bins)
    if len(result) != dims:
        raise ValueError("bins must have the same dimension as the trajectory")
    return result


def _coerce_labels(labels: Sequence[str] | None) -> tuple[str, ...] | None:
    return None if labels is None else tuple(str(label) for label in labels)


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _fixed_bin(value: float, lo: float, hi: float, bins: int) -> int:
    if hi == lo:
        return 0
    width = (hi - lo) / bins
    return max(0, min(bins - 1, int((value - lo) / width)))


def _initial_centers(points: tuple[Point, ...], k: int, seed: int) -> list[Point]:
    # Deterministic seed-dependent start without importing random state into tests.
    start = seed % len(points)
    centers = [points[start]]
    while len(centers) < k:
        best_index = 0
        best_distance = -1.0
        for i, point in enumerate(points):
            distance = min(_squared_distance(point, center) for center in centers)
            if distance > best_distance + 1e-15:
                best_distance = distance
                best_index = i
        centers.append(points[best_index])
    return centers


def _nearest_center(point: Point, centers: Sequence[Point]) -> int:
    best_idx = 0
    best_dist = math.inf
    for idx, center in enumerate(centers):
        dist = _squared_distance(point, center)
        if dist < best_dist - 1e-15:
            best_idx = idx
            best_dist = dist
    return best_idx


def _recompute_centers(points: tuple[Point, ...], assignments: tuple[int, ...], old_centers: Sequence[Point]) -> list[Point]:
    dims = len(points[0])
    sums = [[0.0] * dims for _ in old_centers]
    counts = [0] * len(old_centers)
    for point, idx in zip(points, assignments):
        counts[idx] += 1
        for axis, value in enumerate(point):
            sums[idx][axis] += value
    centers: list[Point] = []
    for idx, old in enumerate(old_centers):
        if counts[idx] == 0:
            centers.append(old)
        else:
            centers.append(tuple(value / counts[idx] for value in sums[idx]))
    return centers


def _same_centers(a: Sequence[Point], b: Sequence[Point]) -> bool:
    return all(_squared_distance(x, y) <= 1e-24 for x, y in zip(a, b))


def _squared_distance(a: Point, b: Point) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))
