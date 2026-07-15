"""Parametric complexity sweep: CLA across varying attractor complexity.

Two parametric families are provided:

1. **Logistic map** — parameter ``r`` controls the bifurcation route from
   fixed point → period doubling → chaos.  As ``r`` increases from ~2.5 to 4.0,
   the symbolic grammar should become more complex.

2. **Lorenz-96** — parameter ``F`` controls the forcing.  Low ``F`` gives
   steady/periodic dynamics; higher ``F`` (≥~6-8) yields chaotic dynamics with
   richer temporal structure.

Both families can be lifted into R^256 (or any target dimension) and run
through the full CLA pipeline: TICA/VAMP embedding → k-means microstates →
CLA grammar → surrogate and held-out metrics.

The sweep harness runs CLA for each parameter value and collects:
- grammar compression gain (bits)
- surrogate excess compression (delta bits)
- held-out next-symbol perplexity
- number of grammar rules discovered
- intrinsic dimension estimate

This is the parametric complexity benchmark Ben requested (2026-07-11):
"an example with a parameter so that as the param varies toward a certain
limit, the attractor grammar gets more and more complex."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
import json
import math
import sys
from typing import Any

from .attractors import logistic_map, lorenz96, high_dimensional_lift, equal_width_symbols
from ..embedding import intrinsic_dimension_participation_ratio, tica_vamp_kinetic_map
from ..symbolization import (
    FixedPartitionSymbolizer,
    fixed_partition_symbols,
    kmeans_microstate_symbols,
    kmeans_microstates,
)
from ..evaluation import (
    cla_compression_gain_bits,
    empirical_code_length_bits,
    heldout_next_symbol_log_loss,
    shuffled_surrogate,
    surrogate_excess_compression,
)
from ..api import CLA
from ..scoring import TwoPartMDLScorer


@dataclass(frozen=True)
class SweepPoint:
    """Result of CLA analysis at one parameter value."""
    parameter: float
    symbols: tuple[str, ...]
    num_symbols: int
    alphabet_size: int
    grammar_rules: int
    grammar_gain_bits: float
    surrogate_delta_bits: float
    heldout_perplexity: float
    intrinsic_dimension: float
    suggested_embedding_dim: int


@dataclass(frozen=True)
class SweepResult:
    """Collection of sweep points with metadata."""
    system: str
    parameter_name: str
    parameter_values: tuple[float, ...]
    points: tuple[SweepPoint, ...]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "parameter_name": self.parameter_name,
            "parameter_values": list(self.parameter_values),
            "config": self.config,
            "points": [
                {
                    "parameter": p.parameter,
                    "num_symbols": p.num_symbols,
                    "alphabet_size": p.alphabet_size,
                    "grammar_rules": p.grammar_rules,
                    "grammar_gain_bits": p.grammar_gain_bits,
                    "surrogate_delta_bits": p.surrogate_delta_bits,
                    "heldout_perplexity": p.heldout_perplexity,
                    "intrinsic_dimension": p.intrinsic_dimension,
                    "suggested_embedding_dim": p.suggested_embedding_dim,
                }
                for p in self.points
            ],
        }


def logistic_map_symbols(
    r: float = 4.0,
    x0: float = 0.123,
    steps: int = 2000,
    discard: int = 500,
    bins: int = 8,
) -> tuple[str, ...]:
    """Generate logistic-map trajectory and symbolize with equal-width bins."""
    traj = logistic_map(r=r, x0=x0, steps=steps, discard=discard)
    return equal_width_symbols(traj, bins=bins, prefix="L")


def logistic_map_symbols_fixed(
    r: float = 4.0,
    x0: float = 0.123,
    steps: int = 2000,
    discard: int = 500,
    bins: int = 8,
    reference_r: float = 4.0,
) -> tuple[str, ...]:
    """Symbolize a logistic map using bounds frozen at ``reference_r``."""

    trajectory = logistic_map(r=r, x0=x0, steps=steps, discard=discard)
    reference = logistic_map(r=reference_r, x0=x0, steps=steps, discard=discard)
    return fixed_partition_symbols(
        trajectory, reference=reference, bins=bins, prefix="L"
    )


def lorenz96_symbols(
    F: float = 8.0,
    steps: int = 2000,
    discard: int = 500,
    dim: int = 5,
    bins: int = 8,
) -> tuple[str, ...]:
    """Generate Lorenz-96 trajectory and symbolize the first coordinate."""
    initial = tuple(8.0 + 0.01 * i for i in range(dim))
    traj = lorenz96(steps=steps, dt=0.01, initial=initial, F=F, discard=discard)
    first_coord = [p[0] for p in traj]
    return equal_width_symbols(first_coord, bins=bins, prefix="F")


def lorenz96_symbols_fixed(
    F: float = 8.0,
    steps: int = 2000,
    discard: int = 500,
    dim: int = 5,
    bins: int = 8,
    reference_F: float = 8.0,
) -> tuple[str, ...]:
    """Symbolize Lorenz-96's first coordinate using reference-frozen bounds."""

    initial = tuple(8.0 + 0.01 * i for i in range(dim))
    trajectory = lorenz96(
        steps=steps, dt=0.01, initial=initial, F=F, discard=discard
    )
    reference = lorenz96(
        steps=steps, dt=0.01, initial=initial, F=reference_F, discard=discard
    )
    return fixed_partition_symbols(
        [point[0] for point in trajectory],
        reference=[point[0] for point in reference],
        bins=bins,
        prefix="F",
    )


def lorenz96_hd_symbols(
    F: float = 8.0,
    steps: int = 2000,
    discard: int = 500,
    dim: int = 5,
    lift_dimension: int = 256,
    noise: float = 0.001,
    microstates: int = 32,
    embedding_dim: int = 3,
    lag: int = 2,
    seed: int = 0,
) -> tuple[str, ...]:
    """Generate Lorenz-96, lift to R^256, reduce via TICA/VAMP, symbolize with k-means."""
    initial = tuple(8.0 + 0.01 * i for i in range(dim))
    traj = lorenz96(steps=steps, dt=0.01, initial=initial, F=F, discard=discard)
    lifted = high_dimensional_lift(traj, dimension=lift_dimension, noise=noise, seed=seed)
    kinetic = tica_vamp_kinetic_map(lifted, dimension=embedding_dim, lag=lag, shrinkage=1e-6)
    return kmeans_microstate_symbols(kinetic.coordinates, k=microstates, seed=seed)


def analyze_symbols(
    symbols: tuple[str, ...],
    *,
    max_iterations: int = 8,
    surrogates: int = 5,
    seed: int = 0,
) -> SweepPoint:
    """Run the full CLA analysis pipeline on a symbol stream."""
    if not symbols:
        return SweepPoint(
            parameter=0.0, symbols=(), num_symbols=0, alphabet_size=0,
            grammar_rules=0, grammar_gain_bits=0.0, surrogate_delta_bits=0.0,
            heldout_perplexity=float("nan"), intrinsic_dimension=0.0,
            suggested_embedding_dim=0,
        )

    # Fit CLA
    model = CLA.simple(max_iterations=max_iterations, miner="suffix_trie", seed=seed).fit_symbols(symbols)
    num_rules = len(model.state.grammar.productions) + len(model.state.grammar.categories)

    # Grammar compression gain
    baseline_bits = empirical_code_length_bits(symbols)
    grammar_cost = TwoPartMDLScorer().score(model.state).total
    gain_bits = baseline_bits - grammar_cost

    # Surrogate excess compression
    def fit(stream):
        return CLA.simple(max_iterations=max_iterations, miner="suffix_trie", seed=seed).fit_symbols(stream)

    surrogate_result = surrogate_excess_compression(symbols, runs=surrogates, seed=seed, fit=fit)

    # Held-out perplexity
    heldout = heldout_next_symbol_log_loss(symbols, train_fraction=0.7, order=2)

    # Intrinsic dimension (on the raw symbol stream as a simple proxy)
    # For high-D lifted data, this would be called on the lifted trajectory
    alphabet_size = len(set(symbols))

    return SweepPoint(
        parameter=0.0,  # filled by caller
        symbols=symbols,
        num_symbols=len(symbols),
        alphabet_size=alphabet_size,
        grammar_rules=num_rules,
        grammar_gain_bits=gain_bits,
        surrogate_delta_bits=surrogate_result.delta_grammar_bits,
        heldout_perplexity=heldout.perplexity,
        intrinsic_dimension=0.0,  # filled by caller if high-D
        suggested_embedding_dim=0,
    )


def sweep_logistic_map(
    r_values: Sequence[float] | None = None,
    *,
    steps: int = 2000,
    discard: int = 500,
    bins: int = 8,
    max_iterations: int = 8,
    surrogates: int = 5,
    seed: int = 0,
) -> SweepResult:
    """Sweep CLA across logistic-map parameter ``r``.

    Default sweep: r ∈ {2.5, 3.0, 3.2, 3.4, 3.5, 3.56, 3.6, 3.7, 3.8, 3.9, 4.0}
    covering the transition from stable fixed point through period doubling
    to full chaos.
    """
    if r_values is None:
        r_values = (2.5, 3.0, 3.2, 3.4, 3.5, 3.56, 3.6, 3.7, 3.8, 3.9, 4.0)

    points = []
    for r in r_values:
        symbols = logistic_map_symbols(r=r, steps=steps, discard=discard, bins=bins)
        point = analyze_symbols(symbols, max_iterations=max_iterations, surrogates=surrogates, seed=seed)
        point = SweepPoint(parameter=r, **{
            f: getattr(point, f) for f in point.__dataclass_fields__
            if f != "parameter"
        })
        points.append(point)
        print(f"  r={r:.2f}: {point.num_symbols} symbols, {point.grammar_rules} rules, "
              f"gain={point.grammar_gain_bits:.1f} bits, "
              f"Δsurrogate={point.surrogate_delta_bits:.1f} bits, "
              f"ppl={point.heldout_perplexity:.2f}")

    return SweepResult(
        system="logistic_map",
        parameter_name="r",
        parameter_values=tuple(r_values),
        points=tuple(points),
        config={"steps": steps, "discard": discard, "bins": bins,
                "max_iterations": max_iterations, "surrogates": surrogates},
    )


def sweep_logistic_map_fixed(
    r_values: Sequence[float] | None = None,
    *,
    steps: int = 2000,
    discard: int = 500,
    bins: int = 8,
    reference_r: float = 4.0,
    x0: float = 0.123,
    max_iterations: int = 8,
    surrogates: int = 5,
    seed: int = 0,
) -> SweepResult:
    """Sweep logistic-map dynamics through one reference-frozen partition."""

    if r_values is None:
        r_values = (2.5, 3.0, 3.2, 3.4, 3.5, 3.56, 3.6, 3.7, 3.8, 3.9, 4.0)
    reference = logistic_map(
        r=reference_r, x0=x0, steps=steps, discard=discard
    )
    symbolizer = FixedPartitionSymbolizer.from_reference(
        reference, bins=bins, prefix="L"
    )

    points = []
    for r in r_values:
        trajectory = logistic_map(r=r, x0=x0, steps=steps, discard=discard)
        symbols = symbolizer.symbolize(trajectory)
        point = analyze_symbols(
            symbols, max_iterations=max_iterations, surrogates=surrogates, seed=seed
        )
        point = SweepPoint(parameter=r, **{
            field_name: getattr(point, field_name)
            for field_name in point.__dataclass_fields__
            if field_name != "parameter"
        })
        points.append(point)

    return SweepResult(
        system="logistic_map_fixed",
        parameter_name="r",
        parameter_values=tuple(r_values),
        points=tuple(points),
        config={
            "steps": steps,
            "discard": discard,
            "bins": bins,
            "reference_r": reference_r,
            "bounds": symbolizer.bounds,
            "max_iterations": max_iterations,
            "surrogates": surrogates,
        },
    )


def sweep_lorenz96(
    F_values: Sequence[float] | None = None,
    *,
    steps: int = 2000,
    discard: int = 500,
    dim: int = 5,
    bins: int = 8,
    max_iterations: int = 8,
    surrogates: int = 5,
    seed: int = 0,
) -> SweepResult:
    """Sweep CLA across Lorenz-96 forcing parameter ``F``.

    Default: F ∈ {2, 4, 5, 6, 7, 8, 10, 12, 16}
    covering steady → periodic → chaotic regimes.
    """
    if F_values is None:
        F_values = (2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0)

    points = []
    for F in F_values:
        symbols = lorenz96_symbols(F=F, steps=steps, discard=discard, dim=dim, bins=bins)
        point = analyze_symbols(symbols, max_iterations=max_iterations, surrogates=surrogates, seed=seed)
        point = SweepPoint(parameter=F, **{
            f: getattr(point, f) for f in point.__dataclass_fields__
            if f != "parameter"
        })
        points.append(point)
        print(f"  F={F:.1f}: {point.num_symbols} symbols, {point.grammar_rules} rules, "
              f"gain={point.grammar_gain_bits:.1f} bits, "
              f"Δsurrogate={point.surrogate_delta_bits:.1f} bits, "
              f"ppl={point.heldout_perplexity:.2f}")

    return SweepResult(
        system="lorenz96",
        parameter_name="F",
        parameter_values=tuple(F_values),
        points=tuple(points),
        config={"steps": steps, "discard": discard, "dim": dim, "bins": bins,
                "max_iterations": max_iterations, "surrogates": surrogates},
    )


def sweep_lorenz96_hd(
    F_values: Sequence[float] | None = None,
    *,
    steps: int = 2000,
    discard: int = 500,
    dim: int = 5,
    lift_dimension: int = 256,
    noise: float = 0.001,
    microstates: int = 32,
    embedding_dim: int = 3,
    lag: int = 2,
    max_iterations: int = 8,
    surrogates: int = 5,
    seed: int = 0,
) -> SweepResult:
    """Sweep CLA across Lorenz-96 ``F`` with full high-D pipeline.

    Each parameter value:
    1. Generate Lorenz-96 trajectory (dim-dimensional)
    2. Lift to R^256 with small noise
    3. Estimate intrinsic dimension
    4. TICA/VAMP kinetic map to embedding_dim
    5. K-means microstate symbolization
    6. CLA grammar induction
    7. Surrogate and held-out metrics
    """
    if F_values is None:
        F_values = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)

    points = []
    for F in F_values:
        print(f"  F={F:.1f}: generating trajectory...")
        initial = tuple(8.0 + 0.01 * i for i in range(dim))
        traj = lorenz96(steps=steps, dt=0.01, initial=initial, F=F, discard=discard)

        print(f"  F={F:.1f}: lifting to R^{lift_dimension}...")
        lifted = high_dimensional_lift(traj, dimension=lift_dimension, noise=noise, seed=seed)

        # Intrinsic dimension estimate
        id_est = intrinsic_dimension_participation_ratio(lifted)

        print(f"  F={F:.1f}: TICA/VAMP → d={embedding_dim}...")
        kinetic = tica_vamp_kinetic_map(lifted, dimension=embedding_dim, lag=lag, shrinkage=1e-6)

        print(f"  F={F:.1f}: k-means microstates (k={microstates})...")
        symbols = kmeans_microstate_symbols(kinetic.coordinates, k=microstates, seed=seed)

        print(f"  F={F:.1f}: CLA grammar induction...")
        point = analyze_symbols(symbols, max_iterations=max_iterations, surrogates=surrogates, seed=seed)
        point = SweepPoint(
            parameter=F,
            symbols=symbols,
            num_symbols=point.num_symbols,
            alphabet_size=point.alphabet_size,
            grammar_rules=point.grammar_rules,
            grammar_gain_bits=point.grammar_gain_bits,
            surrogate_delta_bits=point.surrogate_delta_bits,
            heldout_perplexity=point.heldout_perplexity,
            intrinsic_dimension=id_est.participation_ratio,
            suggested_embedding_dim=id_est.suggested_dimension,
        )
        points.append(point)
        print(f"  F={F:.1f}: {point.num_symbols} symbols, {point.grammar_rules} rules, "
              f"gain={point.grammar_gain_bits:.1f} bits, "
              f"Δsurrogate={point.surrogate_delta_bits:.1f} bits, "
              f"ppl={point.heldout_perplexity:.2f}, "
              f"intrinsic_dim={point.intrinsic_dimension:.2f}")

    return SweepResult(
        system="lorenz96_hd",
        parameter_name="F",
        parameter_values=tuple(F_values),
        points=tuple(points),
        config={
            "steps": steps, "discard": discard, "dim": dim,
            "lift_dimension": lift_dimension, "noise": noise,
            "microstates": microstates, "embedding_dim": embedding_dim,
            "lag": lag, "max_iterations": max_iterations, "surrogates": surrogates,
        },
    )
