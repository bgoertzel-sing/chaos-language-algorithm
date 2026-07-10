"""Adaptive low-cardinality trajectory symbolizers for CLA.

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
