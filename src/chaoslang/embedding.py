"""Dependency-free dynamics-aware embeddings for high-dimensional CLA inputs.

The production roadmap points toward TICA/VAMP libraries such as ``deeptime``.
This module supplies a small pure-Python MVP for tests and local controls:
shrinkage-regularized time-lagged covariances, a non-reversible VAMP/TICA-style
kinetic map, and simple intrinsic-dimension diagnostics.  This is a Phase-0
baseline implementation, not a replacement for a numerically robust VAMP library
on large D≈200–300 traces.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .symbolization import Point, coerce_points

Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class KineticMapResult:
    coordinates: tuple[Point, ...]
    singular_values: tuple[float, ...]
    mean: Point
    lag: int
    shrinkage: float
    input_dimension: int


@dataclass(frozen=True)
class IntrinsicDimensionEstimate:
    participation_ratio: float
    eigenvalues: tuple[float, ...]
    suggested_dimension: int


def intrinsic_dimension_participation_ratio(
    trajectory: Sequence[Sequence[float] | float],
    *,
    variance_fraction: float = 0.95,
) -> IntrinsicDimensionEstimate:
    """Estimate effective dimension from covariance eigenvalue participation.

    This is not a replacement for correlation dimension, but it is cheap and
    useful as the first spectral elbow diagnostic in the high-D embedding path.
    """

    points = coerce_points(trajectory)
    if not points:
        return IntrinsicDimensionEstimate(0.0, (), 0)
    if not 0.0 < variance_fraction <= 1.0:
        raise ValueError("variance_fraction must be in (0, 1]")
    centered, _ = _center(points)
    cov = _covariance(centered, centered, shrinkage=0.0)
    eigvals, _ = _jacobi_eigh(cov)
    eigvals = tuple(max(v, 0.0) for v in sorted(eigvals, reverse=True))
    total = sum(eigvals)
    if total <= 0.0:
        return IntrinsicDimensionEstimate(0.0, eigvals, 0)
    pr = (total * total) / max(sum(v * v for v in eigvals), 1e-300)
    cumulative = 0.0
    suggested = len(eigvals)
    for idx, value in enumerate(eigvals, start=1):
        cumulative += value
        if cumulative / total >= variance_fraction:
            suggested = idx
            break
    return IntrinsicDimensionEstimate(pr, eigvals, suggested)


def tica_vamp_kinetic_map(
    trajectory: Sequence[Sequence[float] | float],
    *,
    dimension: int = 5,
    lag: int = 1,
    shrinkage: float = 1e-6,
) -> KineticMapResult:
    """Embed a trajectory with a small non-reversible TICA/VAMP kinetic map.

    The implementation computes C00, C0tau and Ctautau separately and forms
    ``K = C00^{-1/2} C0tau Ctautau^{-1/2}``.  Coordinates are left singular
    functions scaled by singular values clipped to the ideal VAMP range [0, 1]
    to avoid rank-deficient whitening artifacts in high-D lifted smoke tests.
    We intentionally do not symmetrize C0tau, preserving directional
    information in irreversible symbolic dynamics.

    This routine centers x_t and x_{t+tau} separately for covariance estimation,
    then projects the x_t slice.  Production validation should replace this MVP
    with a tested TICA/VAMP backend such as deeptime.
    """

    points = coerce_points(trajectory)
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if lag < 1:
        raise ValueError("lag must be at least 1")
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must be in [0, 1)")
    if len(points) <= lag:
        raise ValueError("trajectory length must exceed lag")

    x0_raw = points[:-lag]
    xt_raw = points[lag:]
    x0, mean0 = _center(x0_raw)
    xt, _ = _center(xt_raw)
    c00 = _covariance(x0, x0, shrinkage=shrinkage)
    ctt = _covariance(xt, xt, shrinkage=shrinkage)
    c0t = _cross_covariance(x0, xt)
    invsqrt00 = _inverse_sqrt_psd(c00)
    invsqrttt = _inverse_sqrt_psd(ctt)
    k_matrix = _matmul(_matmul(invsqrt00, c0t), invsqrttt)

    kt_k = _matmul(_transpose(k_matrix), k_matrix)
    eigvals, right_vecs = _jacobi_eigh(kt_k)
    order = sorted(range(len(eigvals)), key=lambda i: eigvals[i], reverse=True)
    usable = min(dimension, len(order))
    singular_values: list[float] = []
    left_vectors: list[tuple[float, ...]] = []
    for idx in order[:usable]:
        raw_sigma = math.sqrt(max(eigvals[idx], 0.0))
        sigma = min(raw_sigma, 1.0)
        v = tuple(row[idx] for row in right_vecs)
        kv = _matvec(k_matrix, v)
        if raw_sigma > 1e-12:
            u = tuple(value / raw_sigma for value in kv)
        else:
            u = tuple(0.0 for _ in kv)
        singular_values.append(sigma)
        left_vectors.append(u)

    projection = _matmul(invsqrt00, _columns_to_matrix(left_vectors))
    coordinates: list[Point] = []
    for point in x0:
        raw = _matvec(_transpose(projection), point)
        coordinates.append(tuple(value * singular_values[i] for i, value in enumerate(raw)))

    return KineticMapResult(
        coordinates=tuple(coordinates),
        singular_values=tuple(singular_values),
        mean=mean0,
        lag=lag,
        shrinkage=shrinkage,
        input_dimension=len(points[0]),
    )


def _center(points: Sequence[Point]) -> tuple[tuple[Point, ...], Point]:
    dims = len(points[0])
    mean = tuple(sum(point[i] for point in points) / len(points) for i in range(dims))
    return _center_with_mean(points, mean)


def _center_with_mean(points: Sequence[Point], mean: Point) -> tuple[tuple[Point, ...], Point]:
    return tuple(tuple(value - mean[i] for i, value in enumerate(point)) for point in points), mean


def _covariance(a: Sequence[Point], b: Sequence[Point], *, shrinkage: float) -> Matrix:
    cov = [list(row) for row in _cross_covariance(a, b)]
    if shrinkage > 0.0:
        dims = len(cov)
        trace_mean = sum(cov[i][i] for i in range(dims)) / dims
        for i in range(dims):
            for j in range(dims):
                cov[i][j] *= 1.0 - shrinkage
            cov[i][i] += shrinkage * trace_mean
    return tuple(tuple(row) for row in cov)


def _cross_covariance(a: Sequence[Point], b: Sequence[Point]) -> Matrix:
    if len(a) != len(b):
        raise ValueError("covariance inputs must have equal length")
    dims_a = len(a[0])
    dims_b = len(b[0])
    denom = max(len(a) - 1, 1)
    out = [[0.0] * dims_b for _ in range(dims_a)]
    for pa, pb in zip(a, b):
        for i in range(dims_a):
            ai = pa[i]
            for j in range(dims_b):
                out[i][j] += ai * pb[j]
    return tuple(tuple(value / denom for value in row) for row in out)


def _inverse_sqrt_psd(matrix: Matrix, relative_floor: float = 1e-8) -> Matrix:
    eigvals, eigvecs = _jacobi_eigh(matrix)
    max_eig = max((abs(value) for value in eigvals), default=0.0)
    floor = max(max_eig * relative_floor, 1e-12)
    scales = [1.0 / math.sqrt(max(value, floor)) for value in eigvals]
    return _matmul(_matmul(eigvecs, _diag(scales)), _transpose(eigvecs))


def _jacobi_eigh(matrix: Matrix, *, max_sweeps: int = 100, tol: float = 1e-12) -> tuple[tuple[float, ...], Matrix]:
    n = len(matrix)
    if n == 0:
        return (), ()
    a = [list(row) for row in matrix]
    if any(len(row) != n for row in a):
        raise ValueError("matrix must be square")
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        p, q, max_off = 0, 1 if n > 1 else 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(a[i][j])
                if value > max_off:
                    p, q, max_off = i, j, value
        if max_off < tol or n == 1:
            break
        if abs(a[p][p] - a[q][q]) < 1e-300:
            angle = math.pi / 4.0
        else:
            angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c = math.cos(angle)
        s = math.sin(angle)
        app = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q]
        aqq = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q]
        a[p][q] = a[q][p] = 0.0
        a[p][p] = app
        a[q][q] = aqq
        for r in range(n):
            if r != p and r != q:
                arp = c * a[r][p] - s * a[r][q]
                arq = s * a[r][p] + c * a[r][q]
                a[r][p] = a[p][r] = arp
                a[r][q] = a[q][r] = arq
            vrp = c * v[r][p] - s * v[r][q]
            vrq = s * v[r][p] + c * v[r][q]
            v[r][p] = vrp
            v[r][q] = vrq
    return tuple(a[i][i] for i in range(n)), tuple(tuple(row) for row in v)


def _columns_to_matrix(columns: Sequence[Sequence[float]]) -> Matrix:
    if not columns:
        return ()
    rows = len(columns[0])
    return tuple(tuple(column[i] for column in columns) for i in range(rows))


def _diag(values: Sequence[float]) -> Matrix:
    return tuple(tuple(values[i] if i == j else 0.0 for j in range(len(values))) for i in range(len(values)))


def _transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return ()
    return tuple(tuple(row[i] for row in matrix) for i in range(len(matrix[0])))


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b:
        return ()
    b_t = _transpose(b)
    return tuple(tuple(sum(x * y for x, y in zip(row, col)) for col in b_t) for row in a)


def _matvec(a: Matrix, x: Sequence[float]) -> tuple[float, ...]:
    return tuple(sum(value * x[j] for j, value in enumerate(row)) for row in a)
