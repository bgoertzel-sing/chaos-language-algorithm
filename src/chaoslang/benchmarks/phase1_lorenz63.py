"""Bounded Phase-1 lifted Lorenz-63 benchmark with matched raw-M1 control."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from collections.abc import Sequence

from chaoslang import CLA
from chaoslang.benchmarks.attractors import high_dimensional_lift, lorenz63, m1_symbolize
from chaoslang.embedding import intrinsic_dimension_participation_ratio, tica_vamp_kinetic_map
from chaoslang.evaluation import heldout_next_symbol_log_loss, surrogate_excess_compression
from chaoslang.symbolization import kmeans_microstate_symbols


def _trajectory_hash(points: Sequence[Sequence[float]]) -> str:
    digest = hashlib.sha256()
    for point in points:
        for value in point:
            digest.update(struct.pack("!d", float(value)))
    return digest.hexdigest()


def _evaluate(symbols: tuple[str, ...], *, iterations: int, surrogates: int, seed: int) -> dict:
    def fit(stream):
        return CLA.simple(max_iterations=iterations, miner="suffix_trie", category_method="js", seed=seed).fit_symbols(stream)

    start = time.perf_counter()
    model = fit(symbols)
    elapsed = time.perf_counter() - start
    surrogate = surrogate_excess_compression(symbols, runs=surrogates, seed=seed, fit=fit)
    heldout = heldout_next_symbol_log_loss(symbols, train_fraction=0.7, order=2)
    return {
        "symbol_count": len(symbols), "unique_symbols": len(set(symbols)),
        "rules_learned": len(model.grammar.productions),
        "categories_learned": len(model.grammar.categories),
        "score_total_proxy_units": model.score.total,
        "exact_reconstruction": model.expand() == symbols,
        "fit_wall_time_seconds": elapsed,
        "surrogate_metric": "current_two_part_proxy_vs_shuffled_marginals_not_calibrated_mdl",
        "real_gain_bits_proxy": surrogate.real_gain_bits,
        "surrogate_gain_bits_proxy_mean": surrogate.surrogate_gain_bits_mean,
        "real_minus_shuffled_bits_proxy": surrogate.delta_grammar_bits,
        "surrogate_gain_bits_proxy": list(surrogate.surrogate_gain_bits),
        "heldout_metric": "smoothed_order_2_ngram_not_cla_predictive_likelihood",
        "heldout_log_loss_bits_per_symbol": heldout.log_loss_bits_per_symbol,
        "heldout_perplexity": heldout.perplexity,
        "heldout_evaluated_symbols": heldout.evaluated_symbols,
    }


def run(*, steps: int = 256, discard: int = 64, lift_dimension: int = 256,
        noise: float = 0.001, embedding_dim: int = 3, microstates: int = 12,
        lag: int = 2, bins: int = 2, iterations: int = 4,
        surrogates: int = 2, seed: int = 0, backend: str = "pure") -> dict:
    source = lorenz63(steps=steps, discard=discard)
    lifted = high_dimensional_lift(source, dimension=lift_dimension, noise=noise, seed=seed)
    intrinsic = intrinsic_dimension_participation_ratio(lifted)
    if backend == "deeptime":
        from chaoslang.deeptime_backend import deeptime_kinetic_map
        kinetic = deeptime_kinetic_map(lifted, dimension=embedding_dim, lag=lag, reversible=False)
    else:
        kinetic = tica_vamp_kinetic_map(lifted, dimension=embedding_dim, lag=lag, shrinkage=1e-6)
    adaptive = kmeans_microstate_symbols(kinetic.coordinates, k=microstates, seed=seed)
    # Matched raw-M1 sees the same lifted observations; two bins avoid changing sample support.
    raw_m1 = m1_symbolize(lifted, bins=bins)
    return {
        "benchmark": "phase1_lifted_lorenz63", "status": "diagnostic_not_grammar_preservation_claim",
        "config": {"steps": steps, "discard": discard, "lift_dimension": lift_dimension,
                   "noise": noise, "embedding_dim": embedding_dim, "microstates": microstates,
                   "lag": lag, "m1_bins": bins, "iterations": iterations,
                   "surrogates": surrogates, "seed": seed, "backend": backend},
        "lifted_trajectory_sha256_f64be": _trajectory_hash(lifted),
        "intrinsic_dimension": {"participation_ratio": intrinsic.participation_ratio,
                                "suggested_dimension": intrinsic.suggested_dimension},
        "embedding": {"backend": backend, "coordinates": len(kinetic.coordinates),
                      "dimension": len(kinetic.coordinates[0]),
                      "singular_values": list(kinetic.singular_values)},
        "adaptive_tica_microstates": _evaluate(adaptive, iterations=iterations, surrogates=surrogates, seed=seed),
        "matched_raw_m1": _evaluate(raw_m1, iterations=iterations, surrogates=surrogates, seed=seed),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=256); p.add_argument("--discard", type=int, default=64)
    p.add_argument("--lift-dimension", type=int, default=256); p.add_argument("--noise", type=float, default=.001)
    p.add_argument("--embedding-dim", type=int, default=3); p.add_argument("--microstates", type=int, default=12)
    p.add_argument("--lag", type=int, default=2); p.add_argument("--bins", type=int, default=2)
    p.add_argument("--iterations", type=int, default=4); p.add_argument("--surrogates", type=int, default=2)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--backend", choices=("pure", "deeptime"), default="pure")
    print(json.dumps(run(**vars(p.parse_args(argv))), sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
