"""Dependency-free attractor/symbolization smoke scaffolds."""
from __future__ import annotations


def logistic_map(r: float = 4.0, x0: float = 0.123, steps: int = 128, discard: int = 0) -> list[float]:
    x = x0
    out: list[float] = []
    for i in range(steps + discard):
        x = r * x * (1.0 - x)
        if i >= discard:
            out.append(x)
    return out


def equal_width_symbols(values: list[float], bins: int = 4, prefix: str = "s") -> tuple[str, ...]:
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
