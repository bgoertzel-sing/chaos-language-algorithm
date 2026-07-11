"""Small CLI benchmark for CLA M1 symbolization on control attractors."""
from __future__ import annotations

import argparse
import json
import time

from chaoslang import CLA
from chaoslang.benchmarks.attractors import high_dimensional_lift, lorenz63, mackey_glass, m1_symbolize, rossler, lorenz96, logistic_map
from chaoslang.embedding import intrinsic_dimension_participation_ratio, tica_vamp_kinetic_map
from chaoslang.evaluation import heldout_next_symbol_log_loss, surrogate_excess_compression
from chaoslang.symbolization import kmeans_microstate_symbols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny deterministic M1/control CLA benchmark")
    parser.add_argument("--system", choices=("lorenz63", "rossler", "mackey-glass", "lorenz96", "logistic"), default="lorenz63")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--discard", type=int, default=16)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=0,
                        help="Dimension for Lorenz-96 (default: 5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--miner", choices=("ngram", "suffix_trie"), default="ngram",
                        help="Chunk-proposal miner implementation")
    parser.add_argument("--category-method", choices=("exact", "js"), default="exact",
                        help="Category proposal method")
    parser.add_argument("--js-threshold", type=float, default=0.1,
                        help="Jensen-Shannon threshold when --category-method=js")
    parser.add_argument("--symbolizer", choices=("m1", "kmeans", "tica-kmeans", "dt-kmeans", "dt-tica-kmeans"), default="m1",
                        help="Trajectory-to-symbol stream method (dt-* uses deeptime backend)")
    parser.add_argument("--microstates", type=int, default=32,
                        help="Number of adaptive microstate symbols for kmeans symbolizers")
    parser.add_argument("--embedding-dim", type=int, default=5,
                        help="Kinetic-map dimension for --symbolizer=tica-kmeans")
    parser.add_argument("--lag", type=int, default=1,
                        help="Time lag for --symbolizer=tica-kmeans")
    parser.add_argument("--surrogates", type=int, default=0,
                        help="If positive, run shuffled-surrogate excess-compression metric")
    parser.add_argument("--reversible", action="store_true",
                        help="Use TICA (symmetrized) instead of VAMP for dt-tica-kmeans")
    parser.add_argument("--heldout", action="store_true",
                        help="Report held-out next-symbol log-loss/perplexity")
    parser.add_argument("--lift-dimension", type=int, default=0,
                        help="If positive, lift generated trajectory into this ambient dimension before symbolization")
    parser.add_argument("--lift-noise", type=float, default=0.0,
                        help="Gaussian observation noise for --lift-dimension")
    args = parser.parse_args(argv)

    if args.system == "lorenz63":
        trajectory = lorenz63(steps=args.steps, discard=args.discard)
    elif args.system == "rossler":
        trajectory = rossler(steps=args.steps, discard=args.discard)
    elif args.system == "mackey-glass":
        trajectory = mackey_glass(steps=args.steps, discard=args.discard)
    elif args.system == "lorenz96":
        dim = args.dimension if args.dimension > 0 else 5
        # Build initial condition: F for all dims, with small perturbation on last
        initial = tuple([8.0] * (dim - 1) + [8.01])
        trajectory = lorenz96(steps=args.steps, discard=args.discard, initial=initial)
    elif args.system == "logistic":
        trajectory = logistic_map(steps=args.steps, discard=args.discard)

    source_dimension = len(trajectory[0]) if isinstance(trajectory[0], (tuple, list)) else 1
    if args.lift_dimension > 0:
        trajectory = high_dimensional_lift(
            trajectory,
            dimension=args.lift_dimension,
            noise=args.lift_noise,
            seed=args.seed,
        )

    intrinsic = intrinsic_dimension_participation_ratio(trajectory)
    embedding_summary = None
    if args.symbolizer == "m1":
        symbols = m1_symbolize(trajectory, bins=args.bins)
    elif args.symbolizer == "kmeans":
        symbols = kmeans_microstate_symbols(trajectory, k=args.microstates, seed=args.seed)
    elif args.symbolizer == "tica-kmeans":
        kinetic = tica_vamp_kinetic_map(
            trajectory,
            dimension=args.embedding_dim,
            lag=args.lag,
            shrinkage=1e-6,
        )
        symbols = kmeans_microstate_symbols(kinetic.coordinates, k=min(args.microstates, len(kinetic.coordinates)), seed=args.seed)
        embedding_summary = {
            "coordinates": len(kinetic.coordinates),
            "dimension": len(kinetic.coordinates[0]) if kinetic.coordinates else 0,
            "input_dimension": kinetic.input_dimension,
            "lag": kinetic.lag,
            "shrinkage": kinetic.shrinkage,
            "singular_values": list(kinetic.singular_values),
            "backend": "pure-python-mvp",
        }
    elif args.symbolizer == "dt-kmeans":
        from chaoslang.deeptime_backend import deeptime_kmeans_microstate_symbols
        symbols = deeptime_kmeans_microstate_symbols(trajectory, k=args.microstates, seed=args.seed)
    else:  # dt-tica-kmeans
        from chaoslang.deeptime_backend import deeptime_kinetic_map, deeptime_kmeans_microstate_symbols
        kinetic = deeptime_kinetic_map(
            trajectory,
            dimension=args.embedding_dim,
            lag=args.lag,
            reversible=args.reversible,
        )
        symbols = deeptime_kmeans_microstate_symbols(kinetic.coordinates, k=min(args.microstates, len(kinetic.coordinates)), seed=args.seed)
        embedding_summary = {
            "coordinates": len(kinetic.coordinates),
            "dimension": len(kinetic.coordinates[0]) if kinetic.coordinates else 0,
            "input_dimension": kinetic.input_dimension,
            "lag": kinetic.lag,
            "singular_values": list(kinetic.singular_values),
            "timescales": list(kinetic.timescales),
            "cumulative_kinetic_variance": list(kinetic.cumulative_kinetic_variance),
            "backend": kinetic.backend,
        }
    fit_start = time.perf_counter()
    model = CLA.simple(
        max_iterations=args.iterations,
        seed=args.seed,
        miner=args.miner,
        category_method=args.category_method,
        js_threshold=args.js_threshold,
    ).fit_symbols(symbols)
    fit_wall_time_seconds = time.perf_counter() - fit_start
    result = {
        "system": args.system,
        "steps": args.steps,
        "discard": args.discard,
        "bins": args.bins,
        "dimension": len(trajectory[0]) if isinstance(trajectory[0], (tuple, list)) else 1,
        "source_dimension": source_dimension,
        "lift_dimension": args.lift_dimension,
        "lift_noise": args.lift_noise,
        "seed": args.seed,
        "symbolizer": args.symbolizer,
        "microstates": args.microstates,
        "embedding_dim": args.embedding_dim,
        "lag": args.lag,
        "intrinsic_dimension_participation_ratio": intrinsic.participation_ratio,
        "intrinsic_dimension_suggested": intrinsic.suggested_dimension,
        "embedding": embedding_summary,
        "miner": args.miner,
        "category_method": args.category_method,
        "js_threshold": args.js_threshold,
        "symbol_count": len(symbols),
        "unique_symbols": len(set(symbols)),
        "history": list(model.state.history),
        "score_total": model.score.total,
        "rules_learned": len(model.grammar.productions),
        "fit_wall_time_seconds": fit_wall_time_seconds,
        "exact_reconstruction": model.expand() == symbols,
    }
    if args.heldout:
        heldout = heldout_next_symbol_log_loss(symbols)
        result["heldout_metric"] = "smoothed_ngram_baseline"
        result["heldout_next_symbol_log_loss_bits"] = heldout.log_loss_bits_per_symbol
        result["heldout_next_symbol_perplexity"] = heldout.perplexity
        result["heldout_evaluated_symbols"] = heldout.evaluated_symbols
    if args.surrogates > 0:
        def fit(stream):
            return CLA.simple(
                max_iterations=args.iterations,
                seed=args.seed,
                miner=args.miner,
                category_method=args.category_method,
                js_threshold=args.js_threshold,
            ).fit_symbols(stream)

        surrogate = surrogate_excess_compression(symbols, runs=args.surrogates, seed=args.seed, fit=fit)
        result["surrogate_metric"] = "approx_mdl_vs_shuffled_marginals"
        result["real_gain_bits"] = surrogate.real_gain_bits
        result["surrogate_gain_bits_mean"] = surrogate.surrogate_gain_bits_mean
        result["delta_grammar_bits"] = surrogate.delta_grammar_bits
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
