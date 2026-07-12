"""Preregistered bounded multi-seed sweep for lifted Lorenz-63 Phase-1 diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

from chaoslang.benchmarks.attractors import high_dimensional_lift, lorenz63, m1_symbolize
from chaoslang.benchmarks.phase1_lorenz63 import _evaluate, _trajectory_hash
from chaoslang.deeptime_backend import (deeptime_kmeans_microstate_symbols,
    deeptime_kinetic_map, validate_kinetic_diagnostics)
from chaoslang.embedding import intrinsic_dimension_participation_ratio, tica_vamp_kinetic_map
from chaoslang.symbolization import kmeans_microstate_symbols

METRICS = ("real_minus_shuffled_bits_proxy", "heldout_perplexity")
RUN_FIELDS = (
    "method", "seed", "embedding_dim", "microstates", "lag", "backend",
    "trajectory_sha256", "intrinsic_participation_ratio", "intrinsic_suggested_dimension",
    "symbol_count", "unique_symbols", "rules_learned", "categories_learned",
    "exact_reconstruction", "score_total_proxy_units", "real_gain_bits_proxy",
    "surrogate_gain_bits_proxy_mean", "real_minus_shuffled_bits_proxy",
    "heldout_log_loss_bits_per_symbol", "heldout_perplexity", "heldout_evaluated_symbols",
    "fit_wall_time_seconds", "configuration_wall_time_seconds", "singular_values",
    "timescales", "cumulative_kinetic_variance", "numerical_warnings",
    "diagnostics_valid", "diagnostic_status",
)


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _warning_strings(caught: Iterable[warnings.WarningMessage], values: Iterable[float]) -> list[str]:
    result = [f"{w.category.__name__}: {w.message}" for w in caught]
    if not all(math.isfinite(float(value)) for value in values):
        result.append("non_finite_embedding_diagnostic")
    return result


def _row(method: str, seed: int, embedding_dim: int | None, microstates: int | None,
         lag: int | None, backend: str, trajectory_hash: str, intrinsic: Any,
         evaluation: dict[str, Any], elapsed: float, singular_values: Iterable[float] = (),
         timescales: Iterable[float] = (), cumulative: Iterable[float] = (),
         numerical_warnings: Iterable[str] = ()) -> dict[str, Any]:
    warning_list = list(dict.fromkeys(numerical_warnings))
    row = {field: None for field in RUN_FIELDS}
    row.update({
        "method": method, "seed": seed, "embedding_dim": embedding_dim,
        "microstates": microstates, "lag": lag, "backend": backend,
        "trajectory_sha256": trajectory_hash,
        "intrinsic_participation_ratio": intrinsic.participation_ratio,
        "intrinsic_suggested_dimension": intrinsic.suggested_dimension,
        "configuration_wall_time_seconds": elapsed,
        "singular_values": list(singular_values), "timescales": list(timescales),
        "cumulative_kinetic_variance": list(cumulative),
        "numerical_warnings": warning_list,
        "diagnostics_valid": not warning_list,
        "diagnostic_status": "valid" if not warning_list else "invalid_numerical_diagnostics",
    })
    for key in evaluation:
        if key in row:
            row[key] = evaluation[key]
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["method"] == "deeptime_vamp_kmeans":
            key = (row["method"], row["embedding_dim"], row["microstates"], row["lag"])
        else:
            key = (row["method"], None, row["microstates"], row["lag"])
        groups.setdefault(key, []).append(row)
    deeptime = [r for r in rows if r["method"] == "deeptime_vamp_kmeans"]
    if deeptime:
        groups[("deeptime_vamp_kmeans_overall", None, None, None)] = deeptime
    output = []
    for key in sorted(groups, key=lambda item: tuple("" if v is None else str(v) for v in item)):
        members = groups[key]
        record: dict[str, Any] = {"method": key[0], "embedding_dim": key[1],
                                  "microstates": key[2], "lag": key[3], "n": len(members),
                                  "all_positive_real_minus_shuffled": all(
                                      float(r["real_minus_shuffled_bits_proxy"]) > 0 for r in members)}
        for metric in METRICS:
            values = [float(r[metric]) for r in members]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            half = 1.96 * sd / math.sqrt(len(values))
            record.update({f"{metric}_mean": mean, f"{metric}_std": sd,
                           f"{metric}_min": min(values), f"{metric}_max": max(values),
                           f"{metric}_ci95_low": mean - half,
                           f"{metric}_ci95_high": mean + half})
        output.append(record)
    return output


def run_sweep(*, steps: int = 192, discard: int = 64, lift_dimension: int = 256,
              noise: float = .001, seeds: tuple[int, ...] = (101, 202, 303),
              embedding_dims: tuple[int, ...] = (3, 5),
              microstates: tuple[int, ...] = (8, 16, 24), lags: tuple[int, ...] = (1, 3),
              iterations: int = 2, surrogates: int = 2) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        source = lorenz63(steps=steps, discard=discard)
        lifted = high_dimensional_lift(source, dimension=lift_dimension, noise=noise, seed=seed)
        trajectory_hash = _trajectory_hash(lifted)
        intrinsic = intrinsic_dimension_participation_ratio(lifted)
        for d in embedding_dims:
            for k in microstates:
                for lag in lags:
                    config_start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        kinetic = deeptime_kinetic_map(lifted, dimension=d, lag=lag, reversible=False)
                        symbols = deeptime_kmeans_microstate_symbols(kinetic.coordinates, k=k, seed=seed)
                    evaluation = _evaluate(symbols, iterations=iterations, surrogates=surrogates, seed=seed)
                    diagnostics = (*kinetic.singular_values, *kinetic.timescales,
                                   *kinetic.cumulative_kinetic_variance)
                    rows.append(_row("deeptime_vamp_kmeans", seed, d, k, lag, "deeptime",
                        trajectory_hash, intrinsic, evaluation, time.perf_counter() - config_start,
                        kinetic.singular_values, kinetic.timescales,
                        kinetic.cumulative_kinetic_variance,
                        (*_warning_strings(caught, diagnostics), *kinetic.numerical_warnings)))
        # Roughly alphabet-matched direct/raw baseline: no lift or kinetic embedding.
        config_start = time.perf_counter()
        direct = deeptime_kmeans_microstate_symbols(source, k=16, seed=seed, prefix="raw")
        rows.append(_row("direct_raw_kmeans", seed, None, 16, None, "sklearn",
            trajectory_hash, intrinsic, _evaluate(direct, iterations=iterations,
            surrogates=surrogates, seed=seed), time.perf_counter() - config_start))
        # Deliberate raw compound-symbol failure mode retained from Phase 1.
        config_start = time.perf_counter()
        compound = m1_symbolize(lifted, bins=2)
        rows.append(_row("raw_compound_m1", seed, None, None, None, "dependency_free",
            trajectory_hash, intrinsic, _evaluate(compound, iterations=iterations,
            surrogates=surrogates, seed=seed), time.perf_counter() - config_start))
        # Dependency-free reference at a central preregistered configuration.
        config_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pure = tica_vamp_kinetic_map(lifted, dimension=3, lag=1, shrinkage=1e-6)
            pure_symbols = kmeans_microstate_symbols(pure.coordinates, k=16, seed=seed)
        _, _, pure_issues = validate_kinetic_diagnostics(pure.singular_values)
        rows.append(_row("pure_reference", seed, 3, 16, 1, "dependency_free",
            trajectory_hash, intrinsic, _evaluate(pure_symbols, iterations=iterations,
            surrogates=surrogates, seed=seed), time.perf_counter() - config_start,
            pure.singular_values, (), (), (*_warning_strings(caught, pure.singular_values), *pure_issues)))
    return {
        "schema_version": 2,
        "status": "diagnostic_not_grammar_preservation_claim",
        "metric_labels": {
            "real_minus_shuffled_bits_proxy": "current_two_part_proxy_not_calibrated_cla_mdl",
            "heldout_perplexity": "smoothed_order_2_ngram_not_cla_predictive_likelihood",
        },
        "config": {"steps": steps, "discard": discard, "lift_dimension": lift_dimension,
                   "noise": noise, "seeds": list(seeds), "embedding_dims": list(embedding_dims),
                   "microstates": list(microstates), "lags": list(lags),
                   "iterations": iterations, "surrogates": surrogates},
        "run_count": len(rows), "wall_time_seconds": time.perf_counter() - started,
        "runs": rows, "aggregates": aggregate_rows(rows),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=192); parser.add_argument("--discard", type=int, default=64)
    parser.add_argument("--lift-dimension", type=int, default=256); parser.add_argument("--noise", type=float, default=.001)
    parser.add_argument("--seeds", type=_ints, default=(101, 202, 303))
    parser.add_argument("--embedding-dims", type=_ints, default=(3, 5))
    parser.add_argument("--microstates", type=_ints, default=(8, 16, 24)); parser.add_argument("--lags", type=_ints, default=(1, 3))
    parser.add_argument("--iterations", type=int, default=2); parser.add_argument("--surrogates", type=int, default=2)
    args = vars(parser.parse_args(argv)); output_dir = args.pop("output_dir"); output_dir.mkdir(parents=True, exist_ok=True)
    result = run_sweep(**args)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "runs.csv", result["runs"])
    _write_csv(output_dir / "aggregates.csv", result["aggregates"])
    print(json.dumps({"status": result["status"], "run_count": result["run_count"],
                      "wall_time_seconds": result["wall_time_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
