"""Jensen-Shannon divergence helpers for later category clustering.

This is a dependency-light seam: it clusters already-computed discrete context
histograms without mutating CLA grammar state.  The context-category inducer can
swap its exact-signature grouping for this thresholded clustering once M1/control
benchmarks are stable enough to tune thresholds.
"""
from __future__ import annotations

from collections import Counter
from math import log2
from typing import Hashable, Mapping

Distribution = Mapping[Hashable, float]


def js_divergence(left: Distribution, right: Distribution) -> float:
    """Return Jensen-Shannon divergence in bits for two non-negative maps."""

    p = _normalize(left)
    q = _normalize(right)
    keys = set(p) | set(q)
    midpoint = {key: 0.5 * p.get(key, 0.0) + 0.5 * q.get(key, 0.0) for key in keys}
    return 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)


def cluster_by_js(histograms: Mapping[Hashable, Distribution], *, threshold: float) -> tuple[tuple[Hashable, ...], ...]:
    """Greedily cluster histogram keys whose JS divergence is within threshold.

    Determinism matters more than optimality for this seam.  Clusters are formed
    in sorted-key order against the current cluster centroid.
    """

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    clusters: list[list[Hashable]] = []
    centroids: list[Counter[Hashable]] = []
    for key in sorted(histograms, key=repr):
        hist = Counter(_normalize(histograms[key]))
        placed = False
        for idx, centroid in enumerate(centroids):
            if js_divergence(hist, centroid) <= threshold:
                clusters[idx].append(key)
                centroids[idx].update(hist)
                placed = True
                break
        if not placed:
            clusters.append([key])
            centroids.append(Counter(hist))
    return tuple(tuple(cluster) for cluster in clusters)


def _normalize(values: Distribution) -> dict[Hashable, float]:
    if any(v < 0.0 for v in values.values()):
        raise ValueError("distribution weights must be non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("distribution must have positive mass")
    return {k: float(v) / total for k, v in values.items() if v > 0.0}


def _kl_divergence(p: Distribution, q: Distribution) -> float:
    total = 0.0
    for key, p_value in p.items():
        if p_value > 0.0:
            total += p_value * log2(p_value / q[key])
    return total
