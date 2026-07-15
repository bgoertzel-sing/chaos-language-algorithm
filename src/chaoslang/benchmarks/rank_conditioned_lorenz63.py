"""Leakage-free rank-conditioned Lorenz-63 VAMP/TICA validation experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import KMeans

from chaoslang import CLA
from chaoslang.benchmarks.attractors import high_dimensional_lift, lorenz63
from chaoslang.evaluation import cla_compression_gain_bits


@dataclass(frozen=True)
class Split:
    train: slice
    validation: slice
    test: slice
    gap: int


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray


@dataclass(frozen=True)
class KineticFit:
    method: str
    lag: int
    dimension: int
    left: np.ndarray
    right: np.ndarray
    mean0: np.ndarray
    mean1: np.ndarray
    singular_values: tuple[float, ...]
    covariance_ridge: float
    condition_00: float
    condition_11: float
    retained_rank_00: int
    retained_rank_11: int


def deterministic_split(n: int, *, gap: int) -> Split:
    """Contiguous 60/20/20 split with gaps removed from val/test starts."""
    if n < 20 or gap < 0:
        raise ValueError("invalid split size or gap")
    train_end = int(.6 * n)
    validation_end = int(.8 * n)
    if train_end + gap >= validation_end or validation_end + gap >= n:
        raise ValueError("gap leaves an empty split")
    return Split(slice(0, train_end), slice(train_end + gap, validation_end),
                 slice(validation_end + gap, n), gap)


def fit_pca(train: np.ndarray, rank: int) -> PCAFit:
    """Fit centering and a thin rank-truncated PCA using train only."""
    if train.ndim != 2 or not 1 <= rank <= min(train.shape):
        raise ValueError("unsupported PCA rank")
    mean = train.mean(axis=0)
    _, singular, vt = np.linalg.svd(train - mean, full_matrices=False)
    return PCAFit(mean, vt[:rank], singular[:rank] ** 2 / max(1, len(train) - 1))


def pca_transform(model: PCAFit, data: np.ndarray) -> np.ndarray:
    return (data - model.mean) @ model.components.T


def _inverse_sqrt(cov: np.ndarray, *, relative_cutoff: float,
                  max_condition: float) -> tuple[np.ndarray, int, float]:
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    maximum = float(values[-1])
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("non-positive covariance spectrum")
    floor = max(maximum * relative_cutoff, maximum / max_condition)
    keep = values >= floor
    if not np.any(keep):
        raise ValueError("rank truncation removed every covariance mode")
    retained = values[keep]
    transform = (vectors[:, keep] / np.sqrt(retained)) @ vectors[:, keep].T
    return transform, int(keep.sum()), float(retained[-1] / retained[0])


def fit_kinetic(train: np.ndarray, *, lag: int, dimension: int, method: str,
                ridge_scale: float = 1e-6, relative_cutoff: float = 1e-10,
                max_condition: float = 1e6,
                singular_tolerance: float = 1e-8) -> KineticFit:
    """Fit regularized rank-conditioned nonsymmetric VAMP or reversible TICA."""
    if method not in {"vamp", "tica"}:
        raise ValueError("method must be vamp or tica")
    if lag < 1 or len(train) <= lag or dimension < 1 or dimension > train.shape[1]:
        raise ValueError("unsupported lag or dimension")
    x0, x1 = train[:-lag], train[lag:]
    mean0, mean1 = x0.mean(axis=0), x1.mean(axis=0)
    a, b = x0 - mean0, x1 - mean1
    denom = max(1, len(a) - 1)
    c00, c11, c01 = a.T @ a / denom, b.T @ b / denom, a.T @ b / denom
    if method == "tica":
        shared = (c00 + c11) / 2
        c00 = c11 = shared
        c01 = (c01 + c01.T) / 2
        mean1 = mean0 = (mean0 + mean1) / 2
    ridge = ridge_scale * float((np.trace(c00) + np.trace(c11)) / (2 * train.shape[1]))
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("invalid covariance ridge")
    w0, rank0, cond0 = _inverse_sqrt(c00 + ridge * np.eye(train.shape[1]),
                                      relative_cutoff=relative_cutoff, max_condition=max_condition)
    w1, rank1, cond1 = _inverse_sqrt(c11 + ridge * np.eye(train.shape[1]),
                                      relative_cutoff=relative_cutoff, max_condition=max_condition)
    koopman = w0 @ c01 @ w1
    u, singular, vt = np.linalg.svd(koopman, full_matrices=False)
    raw = tuple(float(v) for v in singular[:dimension])
    if len(raw) < dimension:
        raise ValueError("requested dimension exceeds retained kinetic rank")
    if any(not math.isfinite(v) or v < -singular_tolerance or v > 1 + singular_tolerance for v in raw):
        raise ValueError(f"raw singular value outside valid range: {raw}")
    return KineticFit(method, lag, dimension, w0 @ u[:, :dimension],
                      w1 @ vt.T[:, :dimension], mean0, mean1, raw, ridge,
                      cond0, cond1, rank0, rank1)


def kinetic_transform(model: KineticFit, data: np.ndarray) -> np.ndarray:
    """Apply the train-fitted left singular functions to arbitrary points."""
    return (data - model.mean0) @ model.left


def validation_vamp2(model: KineticFit, validation: np.ndarray,
                     *, regularization: float = 1e-10) -> float:
    """Blocked out-of-sample VAMP-2 for fixed train-learned subspaces.

    Validation covariances only normalize and score the already learned left/right
    singular-function subspaces; they never alter the fitted feature maps.
    """
    if len(validation) <= model.lag:
        raise ValueError("validation block shorter than lag")
    x0 = (validation[:-model.lag] - model.mean0) @ model.left
    x1 = (validation[model.lag:] - model.mean1) @ model.right
    x0 -= x0.mean(axis=0); x1 -= x1.mean(axis=0)
    denom = max(1, len(x0) - 1)
    c00, c11, c01 = x0.T @ x0 / denom, x1.T @ x1 / denom, x0.T @ x1 / denom
    eye = np.eye(model.dimension)
    w0, _, _ = _inverse_sqrt(c00 + regularization * eye,
                              relative_cutoff=1e-12, max_condition=1e8)
    w1, _, _ = _inverse_sqrt(c11 + regularization * eye,
                              relative_cutoff=1e-12, max_condition=1e8)
    score = float(np.linalg.norm(w0 @ c01 @ w1, ord="fro") ** 2)
    if not math.isfinite(score):
        raise ValueError("non-finite validation VAMP-2")
    return score


def fit_symbols(train: np.ndarray, test: np.ndarray, *, k: int, seed: int,
                prefix: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    model = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300).fit(train)
    return (tuple(f"{prefix}{v}" for v in model.labels_),
            tuple(f"{prefix}{v}" for v in model.predict(test)))


def heldout_ngram_loss(train: Sequence[str], test: Sequence[str], *, order: int = 2,
                       alpha: float = .5) -> dict[str, float | int]:
    """Score test symbols from train-only counts and train-only vocabulary."""
    vocabulary = tuple(sorted(set(train)))
    if not vocabulary or any(symbol not in vocabulary for symbol in test):
        raise ValueError("test contains symbol outside train-fitted alphabet")
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, ...]] = Counter()
    for i in range(order, len(train)):
        context = tuple(train[i-order:i]) if order else ()
        counts[context][train[i]] += 1; totals[context] += 1
    marginal = Counter(train); marginal_total = len(train)
    loss = 0.0
    history = tuple(train[-order:]) if order else ()
    for symbol in test:
        context = history[-order:] if order else ()
        if context in counts:
            probability = (counts[context][symbol] + alpha) / (totals[context] + alpha * len(vocabulary))
        else:
            probability = (marginal[symbol] + alpha) / (marginal_total + alpha * len(vocabulary))
        loss -= math.log2(probability)
        if order:
            history = (*history, symbol)[-order:]
    per_symbol = loss / len(test)
    return {"log_loss_bits_per_symbol": per_symbol, "perplexity": 2 ** per_symbol,
            "evaluated_symbols": len(test), "vocabulary_size": len(vocabulary)}


def circular_block_surrogate(symbols: Sequence[str], *, block_length: int,
                             seed: int) -> tuple[str, ...]:
    """Shuffle circularly-offset blocks while preserving within-block order."""
    import random
    stream = tuple(symbols)
    if block_length < 1 or block_length > len(stream):
        raise ValueError("invalid block length")
    rng = random.Random(seed)
    offset = rng.randrange(block_length)
    rotated = stream[offset:] + stream[:offset]
    blocks = [rotated[i:i + block_length] for i in range(0, len(rotated), block_length)]
    rng.shuffle(blocks)
    shuffled = tuple(value for block in blocks for value in block)
    return shuffled[-offset:] + shuffled[:-offset] if offset else shuffled


def grammar_margin(symbols: Sequence[str], *, block_lengths: Sequence[int],
                   surrogates: int, seed: int, iterations: int) -> dict[str, Any]:
    stream = tuple(symbols)
    def fit(values: Sequence[str]):
        return CLA.simple(max_iterations=iterations, miner="suffix_trie",
                          category_method="js", seed=seed).fit_symbols(values)
    real = cla_compression_gain_bits(stream, fit=fit)
    by_block: dict[str, Any] = {}
    for block in block_lengths:
        gains = [cla_compression_gain_bits(circular_block_surrogate(
                    stream, block_length=block, seed=seed + 1009 * block + i), fit=fit)
                 for i in range(surrogates)]
        mean = float(np.mean(gains))
        by_block[str(block)] = {"surrogate_gain_bits": gains,
                                "surrogate_gain_bits_mean": mean,
                                "margin_bits": real - mean}
    return {"real_gain_bits": real, "by_block_length": by_block,
            "minimum_margin_bits": min(v["margin_bits"] for v in by_block.values())}


def _trajectory(seed: int, *, steps: int, discard: int) -> tuple[tuple[float, ...], ...]:
    rng = np.random.default_rng(seed)
    initial = tuple(float(v) for v in rng.uniform(low=(-15, -20, 5), high=(15, 20, 35)))
    return lorenz63(steps=steps, discard=discard, initial=initial)


def _hash_array(data: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(data, dtype=">f8").tobytes()).hexdigest()


def run_experiment(*, steps: int = 2048, discard: int = 2048, lift_dimension: int = 256,
                   trajectory_seeds: tuple[int, ...] = (101, 202, 303),
                   ranks: tuple[int, ...] = (3, 10), dimensions: tuple[int, ...] = (2, 3),
                   lags: tuple[int, ...] = (1, 4, 16), microstates: tuple[int, ...] = (8, 16),
                   block_lengths: tuple[int, ...] = (4, 16), surrogates: int = 20,
                   iterations: int = 2) -> dict[str, Any]:
    started = time.perf_counter(); trajectories: list[dict[str, Any]] = []
    for trajectory_index, seed in enumerate(trajectory_seeds):
        xyz = np.asarray(_trajectory(seed, steps=steps, discard=discard))
        lifted = np.asarray(high_dimensional_lift(tuple(map(tuple, xyz)), dimension=lift_dimension,
                                                  noise=.001, seed=seed + 10000))
        split = deterministic_split(len(xyz), gap=max(lags))
        selection: list[dict[str, Any]] = []; fitted: dict[tuple[str, int, int, int], tuple[PCAFit, KineticFit]] = {}
        pcas: dict[int, PCAFit] = {}
        for rank in ranks:
            pca = pcas[rank] = fit_pca(lifted[split.train], rank)
            train_rank = pca_transform(pca, lifted[split.train])
            validation_rank = pca_transform(pca, lifted[split.validation])
            for method in ("vamp", "tica"):
                for dimension in dimensions:
                    if dimension > rank:
                        continue
                    for lag in lags:
                        record: dict[str, Any] = {"method": method, "rank": rank,
                            "dimension": dimension, "lag": lag, "status": "valid"}
                        try:
                            model = fit_kinetic(train_rank, lag=lag, dimension=dimension, method=method)
                            score = validation_vamp2(model, validation_rank)
                            fitted[(method, rank, dimension, lag)] = (pca, model)
                            record.update({"validation_vamp2": score,
                                "raw_singular_values": list(model.singular_values),
                                "covariance_ridge": model.covariance_ridge,
                                "condition_00": model.condition_00, "condition_11": model.condition_11,
                                "retained_rank_00": model.retained_rank_00,
                                "retained_rank_11": model.retained_rank_11})
                        except ValueError as exc:
                            record.update({"status": "rejected", "reason": str(exc),
                                           "validation_vamp2": None})
                        selection.append(record)
        selected: dict[str, dict[str, Any]] = {}
        for method in ("vamp", "tica"):
            candidates = [r for r in selection if r["method"] == method and r["status"] == "valid"]
            if not candidates:
                raise RuntimeError(f"no valid {method} candidate for trajectory {seed}")
            best = sorted(candidates, key=lambda r: (-r["validation_vamp2"], r["rank"],
                                                     r["dimension"], r["lag"]))[0]
            selected[method] = best
        methods: list[dict[str, Any]] = []
        for method, chosen in selected.items():
            pca, kinetic = fitted[(method, chosen["rank"], chosen["dimension"], chosen["lag"])]
            train_features = kinetic_transform(kinetic, pca_transform(pca, lifted[split.train]))
            test_features = kinetic_transform(kinetic, pca_transform(pca, lifted[split.test]))
            for k in microstates:
                train_symbols, test_symbols = fit_symbols(train_features, test_features, k=k,
                    seed=seed + k, prefix=f"{method}{k}_")
                methods.append(_evaluate_method(method, chosen["rank"], chosen["dimension"],
                    chosen["lag"], k, train_symbols, test_symbols, block_lengths, surrogates,
                    seed, iterations, chosen["validation_vamp2"]))
        for rank, pca in pcas.items():
            train_features = pca_transform(pca, lifted[split.train])
            test_features = pca_transform(pca, lifted[split.test])
            for k in microstates:
                tr, te = fit_symbols(train_features, test_features, k=k, seed=seed+k,
                                     prefix=f"pca{rank}k{k}_")
                methods.append(_evaluate_method("pca_lift_kmeans", rank, rank, None, k, tr, te,
                    block_lengths, surrogates, seed, iterations, None))
        for k in microstates:
            tr, te = fit_symbols(xyz[split.train], xyz[split.test], k=k, seed=seed+k,
                                 prefix=f"xyz{k}_")
            methods.append(_evaluate_method("direct_xyz_kmeans", None, 3, None, k, tr, te,
                block_lengths, surrogates, seed, iterations, None))
        trajectories.append({"trajectory_index": trajectory_index, "seed": seed,
            "first_integrated_point": list(_trajectory(seed, steps=1, discard=0)[0]),
            "xyz_sha256_f64be": _hash_array(xyz), "lifted_sha256_f64be": _hash_array(lifted),
            "split": {"train": [split.train.start, split.train.stop],
                      "validation": [split.validation.start, split.validation.stop],
                      "test": [split.test.start, split.test.stop], "gap": split.gap},
            "candidate_selection": selection, "selected": selected, "methods": methods})
    stop_go = _stop_go(trajectories, microstates)
    return {"schema_version": 1, "status": "completed", "config": {
        "steps": steps, "discard": discard, "lift_dimension": lift_dimension,
        "trajectory_seeds": list(trajectory_seeds), "ranks": list(ranks),
        "dimensions": list(dimensions), "lags": list(lags), "microstates": list(microstates),
        "block_lengths": list(block_lengths), "surrogates": surrogates,
        "iterations": iterations, "maximum_whitening_condition": 1e6,
        "covariance_ridge_scale": 1e-6, "singular_value_policy": "raw_fail_closed_no_clipping"},
        "metric_labels": {"log_loss": "train-only order-2 ngram test loss",
            "grammar_margin": "test current-two-part proxy minus circular-block-surrogate mean"},
        "trajectories": trajectories, "stop_go": stop_go,
        "wall_time_seconds": time.perf_counter() - started}


def _evaluate_method(method: str, rank: int | None, dimension: int, lag: int | None,
                     k: int, train_symbols: tuple[str, ...], test_symbols: tuple[str, ...],
                     block_lengths: Sequence[int], surrogates: int, seed: int,
                     iterations: int, validation_score: float | None) -> dict[str, Any]:
    return {"method": method, "rank": rank, "dimension": dimension, "lag": lag, "k": k,
        "validation_vamp2": validation_score, "train_symbol_count": len(train_symbols),
        "test_symbol_count": len(test_symbols), "train_unique_symbols": len(set(train_symbols)),
        "test_unique_symbols": len(set(test_symbols)),
        "heldout": heldout_ngram_loss(train_symbols, test_symbols),
        "grammar": grammar_margin(test_symbols, block_lengths=block_lengths,
                                  surrogates=surrogates, seed=seed, iterations=iterations)}


def _stop_go(trajectories: Sequence[dict[str, Any]], microstates: Sequence[int]) -> dict[str, Any]:
    by_k: dict[str, Any] = {}
    for k in microstates:
        outcomes = []
        for trajectory in trajectories:
            rows = [r for r in trajectory["methods"] if r["k"] == k]
            vamp = next(r for r in rows if r["method"] == "vamp")
            pcas = [r for r in rows if r["method"] == "pca_lift_kmeans"]
            best_loss = min(r["heldout"]["log_loss_bits_per_symbol"] for r in pcas)
            best_margin = max(r["grammar"]["minimum_margin_bits"] for r in pcas)
            beats_loss = vamp["heldout"]["log_loss_bits_per_symbol"] < best_loss
            beats_margin = vamp["grammar"]["minimum_margin_bits"] > best_margin
            outcomes.append({"trajectory_index": trajectory["trajectory_index"],
                "vamp_log_loss": vamp["heldout"]["log_loss_bits_per_symbol"],
                "best_pca_log_loss": best_loss, "vamp_minimum_margin": vamp["grammar"]["minimum_margin_bits"],
                "best_pca_minimum_margin": best_margin, "beats_both": beats_loss and beats_margin})
        passes = sum(o["beats_both"] for o in outcomes)
        by_k[str(k)] = {"trajectories_beating_both": passes, "passes_2_of_3": passes >= 2,
                        "outcomes": outcomes}
    overall_go = any(value["passes_2_of_3"] for value in by_k.values())
    return {"criterion_by_k": by_k, "go_linear_vamp": overall_go,
            "conclusion": "continue_linear_vamp" if overall_go else "stop_linear_vamp"}


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value)


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for trajectory in result["trajectories"]:
        for row in trajectory["methods"]:
            rows.append({"trajectory_index": trajectory["trajectory_index"], "seed": trajectory["seed"],
                "method": row["method"], "rank": row["rank"], "dimension": row["dimension"],
                "lag": row["lag"], "k": row["k"], "validation_vamp2": row["validation_vamp2"],
                "test_log_loss_bits_per_symbol": row["heldout"]["log_loss_bits_per_symbol"],
                "test_perplexity": row["heldout"]["perplexity"],
                "minimum_block_surrogate_margin_bits": row["grammar"]["minimum_margin_bits"],
                "block_metrics_json": json.dumps(row["grammar"]["by_block_length"], sort_keys=True)})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2048); parser.add_argument("--discard", type=int, default=2048)
    parser.add_argument("--lift-dimension", type=int, default=256)
    parser.add_argument("--trajectory-seeds", type=_ints, default=(101,202,303))
    parser.add_argument("--ranks", type=_ints, default=(3,10)); parser.add_argument("--dimensions", type=_ints, default=(2,3))
    parser.add_argument("--lags", type=_ints, default=(1,4,16)); parser.add_argument("--microstates", type=_ints, default=(8,16))
    parser.add_argument("--block-lengths", type=_ints, default=(4,16)); parser.add_argument("--surrogates", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=2)
    args = vars(parser.parse_args(argv)); output = args.pop("output_dir"); output.mkdir(parents=True, exist_ok=True)
    result = run_experiment(**args)
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    _write_csv(output / "metrics.csv", result)
    print(json.dumps({"status": result["status"], "wall_time_seconds": result["wall_time_seconds"],
                      "stop_go": result["stop_go"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
