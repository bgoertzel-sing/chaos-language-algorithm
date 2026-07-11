"""Deeptime-backed TICA/VAMP kinetic map and PCCA+ microstate utilities.

This module provides a numerically robust backend for high-dimensional CLA
embedding using the ``deeptime`` library.  It is imported lazily so that the
core ``chaoslang`` package remains dependency-free; callers should handle
``ImportError`` if ``deeptime`` is not installed.

The pure-Python Phase-0 MVP in ``chaoslang.embedding`` remains the fallback for
environments without ``deeptime`` or for small smoke tests.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .symbolization import Point, coerce_points


@dataclass(frozen=True)
class DeeptimeKineticMapResult:
    coordinates: tuple[Point, ...]
    singular_values: tuple[float, ...]
    timescales: tuple[float, ...]
    cumulative_kinetic_variance: tuple[float, ...]
    lag: int
    input_dimension: int
    dimension: int
    backend: str


@dataclass(frozen=True)
class PCCAMembershipResult:
    n_macrostates: int
    assignments: tuple[int, ...]
    soft_memberships: tuple[tuple[float, ...], ...]
    microstate_centers: tuple[Point, ...]


def _require_deeptime() -> Any:
    try:
        import deeptime  # noqa: F401
        from deeptime.decomposition import TICA, VAMP  # noqa: F401
        from deeptime.markov import pcca  # noqa: F401
        return deeptime
    except ImportError as exc:
        raise ImportError(
            "deeptime is required for this backend. Install with: pip install deeptime"
        ) from exc


def deeptime_kinetic_map(
    trajectory: Sequence[Sequence[float] | float],
    *,
    dimension: int = 5,
    lag: int = 1,
    reversible: bool = False,
) -> DeeptimeKineticMapResult:
    """Embed a trajectory using deeptime's TICA or VAMP kinetic map.

    When ``reversible=False`` (default), uses VAMP singular functions to
    preserve directional/irreversible symbolic grammar structure.  When
    ``reversible=True``, uses standard TICA (symmetrized covariance).

    Parameters
    ----------
    trajectory
        Sequence of scalar or vector trajectory points.
    dimension
        Number of kinetic-map coordinates to keep.
    lag
        Time lag in steps for covariance estimation.
    reversible
        If True, use TICA (symmetrized, detailed balance assumed).
        If False, use VAMP (directional, no detailed balance).
    """

    deeptime = _require_deeptime()
    from deeptime.decomposition import TICA, VAMP

    points = coerce_points(trajectory)
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if lag < 1:
        raise ValueError("lag must be at least 1")
    if len(points) <= lag:
        raise ValueError("trajectory length must exceed lag")

    data = np.array(points, dtype=np.float64)
    input_dim = data.shape[1]

    if reversible:
        estimator = TICA(dim=dimension, lagtime=lag)
        backend_name = "tica"
    else:
        estimator = VAMP(dim=dimension, lagtime=lag)
        backend_name = "vamp"

    model = estimator.fit(data).fetch_model()
    coords = model.transform(data)
    singular_values_raw = np.array(model.singular_values)
    timescales_raw = np.array(model.timescales())

    n_kept = min(dimension, len(singular_values_raw))
    singular_values = tuple(float(sv) for sv in singular_values_raw[:n_kept])
    timescales = tuple(float(ts) for ts in timescales_raw[:n_kept])

    ckv = model.cumulative_kinetic_variance
    ckv_tuple = tuple(float(v) for v in ckv[:n_kept]) if ckv is not None else ()

    coordinates = tuple(tuple(float(v) for v in row) for row in coords)

    return DeeptimeKineticMapResult(
        coordinates=coordinates,
        singular_values=singular_values,
        timescales=timescales,
        cumulative_kinetic_variance=ckv_tuple,
        lag=lag,
        input_dimension=input_dim,
        dimension=n_kept,
        backend=backend_name,
    )


def deeptime_implied_timescales(
    trajectory: Sequence[Sequence[float] | float],
    *,
    lags: Sequence[int] = (1, 2, 5, 10, 20),
    dimension: int = 5,
    reversible: bool = False,
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]:
    """Compute implied timescales across multiple lags.

    Returns (lags, timescales_per_lag) where each inner tuple contains
    the implied timescales for that lag.  Useful for choosing ``lag`` via
    plateau detection.
    """

    _require_deeptime()
    from deeptime.decomposition import VAMP, TICA

    points = coerce_points(trajectory)
    data = np.array(points, dtype=np.float64)

    estimator_cls = TICA if reversible else VAMP
    its_lags: list[int] = []
    its_values: list[tuple[float, ...]] = []
    for lag in lags:
        if lag >= len(data):
            continue
        estimator = estimator_cls(dim=dimension, lagtime=lag)
        model = estimator.fit(data).fetch_model()
        ts = np.array(model.timescales())
        its_lags.append(int(lag))
        its_values.append(tuple(float(v) for v in ts[:dimension]))
    return tuple(its_lags), tuple(its_values)


def deeptime_pcca_memberships(
    coordinates: Sequence[Sequence[float] | float],
    *,
    n_microstates: int = 50,
    n_macrostates: int = 4,
    lag: int = 1,
    seed: int = 0,
) -> PCCAMembershipResult:
    """Cluster coordinates into microstates and compute PCCA+ soft memberships.

    Two-step process:
    1. k-means clustering into ``n_microstates`` microstates.
    2. Estimate a Markov state model on the microstate trajectory.
    3. PCCA+ coarse-graining into ``n_macrostates`` metastable sets.

    The soft memberships are used to seed CLA meta-symbol categories:
    microstates falling in the same metastable set are strong category
    candidates.
    """

    _require_deeptime()
    from deeptime.markov import pcca
    from deeptime.markov.msm import MaximumLikelihoodMSM
    from sklearn.cluster import KMeans as SKKMeans

    points = coerce_points(coordinates)
    data = np.array(points, dtype=np.float64)

    if n_microstates > len(data):
        n_microstates = len(data)

    kmeans = SKKMeans(n_clusters=n_microstates, random_state=seed, n_init=10)
    assignments_raw = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_

    microstate_traj = assignments_raw.astype(np.int32)

    if len(microstate_traj) < lag + 2:
        return PCCAMembershipResult(
            n_macrostates=0,
            assignments=tuple(int(a) for a in microstate_traj),
            soft_memberships=(),
            microstate_centers=tuple(tuple(float(v) for v in c) for c in centers),
        )

    try:
        msm = MaximumLikelihoodMSM(reversible=False, lag=lag).fit(microstate_traj).fetch_model()
        if msm is not None and hasattr(msm, 'pi') and msm.pi is not None:
            pcca_model = pcca(msm, n_macrostates)
            memberships = pcca_model.memberships
            soft = tuple(tuple(float(v) for v in row) for row in memberships)
            return PCCAMembershipResult(
                n_macrostates=n_macrostates,
                assignments=tuple(int(a) for a in microstate_traj),
                soft_memberships=soft,
                microstate_centers=tuple(tuple(float(v) for v in c) for c in centers),
            )
    except Exception:
        pass

    return PCCAMembershipResult(
        n_macrostates=0,
        assignments=tuple(int(a) for a in microstate_traj),
        soft_memberships=(),
        microstate_centers=tuple(tuple(float(v) for v in c) for c in centers),
    )


def deeptime_kmeans_microstate_symbols(
    trajectory: Sequence[Sequence[float] | float],
    *,
    k: int = 32,
    seed: int = 0,
    prefix: str = "km",
) -> tuple[str, ...]:
    """Symbolize using sklearn KMeans (faster for high-D than pure-Python k-means)."""

    _require_deeptime()
    from sklearn.cluster import KMeans as SKKMeans

    points = coerce_points(trajectory)
    data = np.array(points, dtype=np.float64)
    if k > len(data):
        k = len(data)
    kmeans = SKKMeans(n_clusters=k, random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(data)
    return tuple(f"{prefix}{int(a)}" for a in assignments)
