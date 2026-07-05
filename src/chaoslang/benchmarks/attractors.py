"""Dependency-free attractor controls and M1 symbolization utilities.

The functions here are intentionally small and deterministic so benchmark slices
can run in the stdlib-only prototype.  Continuous controls use fixed-step RK4;
M1 symbolization maps each trajectory point into a fixed rectangular partition.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

Point = tuple[float, ...]
VectorField = Callable[[Point], Point]


def logistic_map(r: float = 4.0, x0: float = 0.123, steps: int = 128, discard: int = 0) -> list[float]:
    """Return a deterministic logistic-map control trajectory."""

    if steps < 0 or discard < 0:
        raise ValueError("steps and discard must be non-negative")
    x = x0
    out: list[float] = []
    for i in range(steps + discard):
        x = r * x * (1.0 - x)
        if i >= discard:
            out.append(x)
    return out


def lorenz63(
    *,
    steps: int = 128,
    dt: float = 0.01,
    initial: Sequence[float] = (1.0, 1.0, 1.0),
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    discard: int = 0,
) -> tuple[Point, ...]:
    """Return a Lorenz-63 trajectory using deterministic fixed-step RK4."""

    _validate_flow_args(steps, discard, dt, initial, 3)

    def field(p: Point) -> Point:
        x, y, z = p
        return (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)

    return _integrate_rk4(field, tuple(float(v) for v in initial), steps, dt, discard)


def rossler(
    *,
    steps: int = 128,
    dt: float = 0.05,
    initial: Sequence[float] = (0.1, 0.0, 0.0),
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    discard: int = 0,
) -> tuple[Point, ...]:
    """Return a Rössler trajectory using deterministic fixed-step RK4."""

    _validate_flow_args(steps, discard, dt, initial, 3)

    def field(p: Point) -> Point:
        x, y, z = p
        return (-y - z, x + a * y, b + z * (x - c))

    return _integrate_rk4(field, tuple(float(v) for v in initial), steps, dt, discard)



def mackey_glass(
    *,
    steps: int = 128,
    dt: float = 0.1,
    initial: float = 0.5,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    discard: int = 0,
) -> tuple[float, ...]:
    """Return a Mackey-Glass delay-differential trajectory.

    Uses a fixed-step Euler method with a discrete delay buffer of ``tau``
    steps.  The history is initialised to ``initial`` for all delayed samples.
    """
    if steps < 0 or discard < 0:
        raise ValueError("steps and discard must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if tau < 1:
        raise ValueError("tau must be at least 1")

    # History buffer: tau past values all initialised to initial.
    history: list[float] = [float(initial)] * tau
    x = float(initial)
    out: list[float] = []
    for i in range(steps + discard):
        x_tau = history[i]  # value tau steps ago
        dx = beta * x_tau / (1.0 + x_tau ** n) - gamma * x
        x = x + dt * dx
        history.append(x)
        if i >= discard:
            out.append(x)
    return tuple(out)


def lorenz96(
    *,
    steps: int = 128,
    dt: float = 0.01,
    initial: Sequence[float] = (8.0, 8.0, 8.0, 8.0, 8.01),
    F: float = 8.0,
    discard: int = 0,
) -> tuple[Point, ...]:
    """Return a Lorenz-96 trajectory using deterministic fixed-step RK4.

    The system dimension is inferred from ``initial``.  Standard forcing
    ``F=8`` with dimension >= 4 yields chaotic behaviour.
    """
    dims = len(initial)
    _validate_flow_args(steps, discard, dt, initial, dims)

    def field(p: Point) -> Point:
        vals = list(p)
        dx = []
        for i in range(dims):
            # dx_i = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
            x_next = vals[(i + 1) % dims]
            x_prev2 = vals[(i - 2) % dims]
            x_prev1 = vals[(i - 1) % dims]
            dx.append((x_next - x_prev2) * x_prev1 - vals[i] + F)
        return tuple(dx)

    return _integrate_rk4(field, tuple(float(v) for v in initial), steps, dt, discard)


def equal_width_symbols(values: Sequence[float], bins: int = 4, prefix: str = "s") -> tuple[str, ...]:
    """Symbolize scalar values with equal-width bins inferred from the values."""

    if bins < 1:
        raise ValueError("bins must be at least 1")
    if not values:
        return ()
    lo, hi = min(values), max(values)
    if hi == lo:
        return (f"{prefix}0",) * len(values)
    width = (hi - lo) / bins
    symbols = []
    for value in values:
        idx = min(bins - 1, int((value - lo) / width))
        symbols.append(f"{prefix}{idx}")
    return tuple(symbols)


def m1_symbolize(
    trajectory: Sequence[Sequence[float] | float],
    *,
    bins: int | Sequence[int] = 4,
    bounds: Sequence[tuple[float, float]] | None = None,
    prefix: str = "m1",
    axis_prefixes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Map scalar or vector trajectory points into fixed partition symbols.

    If ``bounds`` is omitted, rectangular bounds are inferred once from the full
    trajectory and then held fixed for every point.  Supplying bounds is useful
    for comparing separate controls under the same partition.  Values outside
    supplied bounds are clamped into the nearest edge bin.
    """

    points = _coerce_points(trajectory)
    if not points:
        return ()
    dims = len(points[0])
    if any(len(p) != dims for p in points):
        raise ValueError("all trajectory points must have the same dimension")

    bin_counts = _normalize_bins(bins, dims)
    ranges = tuple(bounds) if bounds is not None else _infer_bounds(points, dims)
    if len(ranges) != dims:
        raise ValueError("bounds length must match trajectory dimension")
    labels = tuple(axis_prefixes) if axis_prefixes is not None else tuple(f"d{i}" for i in range(dims))
    if len(labels) != dims:
        raise ValueError("axis_prefixes length must match trajectory dimension")

    out: list[str] = []
    for point in points:
        parts = []
        for axis, value in enumerate(point):
            idx = _bin_index(value, ranges[axis][0], ranges[axis][1], bin_counts[axis])
            parts.append(f"{labels[axis]}{idx}")
        out.append(f"{prefix}:" + "|".join(parts))
    return tuple(out)


def _validate_flow_args(steps: int, discard: int, dt: float, initial: Sequence[float], dims: int) -> None:
    if steps < 0 or discard < 0:
        raise ValueError("steps and discard must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if len(initial) != dims:
        raise ValueError(f"initial must have dimension {dims}")


def _integrate_rk4(field: VectorField, initial: Point, steps: int, dt: float, discard: int) -> tuple[Point, ...]:
    state = initial
    out: list[Point] = []
    for i in range(steps + discard):
        state = _rk4_step(field, state, dt)
        if i >= discard:
            out.append(state)
    return tuple(out)


def _rk4_step(field: VectorField, state: Point, dt: float) -> Point:
    k1 = field(state)
    k2 = field(_add_scaled(state, k1, dt / 2.0))
    k3 = field(_add_scaled(state, k2, dt / 2.0))
    k4 = field(_add_scaled(state, k3, dt))
    return tuple(s + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4))


def _add_scaled(state: Point, delta: Point, scale: float) -> Point:
    return tuple(s + scale * d for s, d in zip(state, delta))


def _coerce_points(trajectory: Sequence[Sequence[float] | float]) -> tuple[Point, ...]:
    points: list[Point] = []
    for item in trajectory:
        if isinstance(item, (int, float)):
            points.append((float(item),))
        else:
            points.append(tuple(float(v) for v in item))
    return tuple(points)


def _normalize_bins(bins: int | Sequence[int], dims: int) -> tuple[int, ...]:
    if isinstance(bins, int):
        counts = (bins,) * dims
    else:
        counts = tuple(int(b) for b in bins)
    if len(counts) != dims:
        raise ValueError("bins length must match trajectory dimension")
    if any(b < 1 for b in counts):
        raise ValueError("all bin counts must be at least 1")
    return counts


def _infer_bounds(points: Sequence[Point], dims: int) -> tuple[tuple[float, float], ...]:
    return tuple((min(p[axis] for p in points), max(p[axis] for p in points)) for axis in range(dims))


def _bin_index(value: float, lo: float, hi: float, bins: int) -> int:
    if hi < lo:
        raise ValueError("each bound must be ordered as (lo, hi)")
    if hi == lo:
        return 0
    width = (hi - lo) / bins
    idx = int((value - lo) / width)
    return max(0, min(bins - 1, idx))
