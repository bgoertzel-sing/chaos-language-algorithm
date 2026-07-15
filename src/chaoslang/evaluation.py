"""Phase-0 validation diagnostics for grammar-preserving symbolization.

These metrics are intentionally lightweight baselines.  Held-out loss is a
smoothed n-gram next-symbol baseline, not yet a CLA grammar likelihood; surrogate
compression uses the current approximate scorer and shuffled marginal controls.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import random

from .api import CLA
from .scoring import TwoPartMDLScorer


@dataclass(frozen=True)
class SurrogateCompressionResult:
    real_gain_bits: float
    surrogate_gain_bits_mean: float
    delta_grammar_bits: float
    surrogate_gain_bits: tuple[float, ...]


@dataclass(frozen=True)
class HeldoutLogLossResult:
    log_loss_bits_per_symbol: float
    perplexity: float
    evaluated_symbols: int
    vocabulary_size: int
    order: int


def empirical_code_length_bits(symbols: Sequence[str]) -> float:
    """Optimal iid empirical code length for a symbol stream."""

    if not symbols:
        return 0.0
    counts = Counter(symbols)
    total = len(symbols)
    return sum(count * (-math.log2(count / total)) for count in counts.values())


def cla_compression_gain_bits(symbols: Sequence[str], fit: Callable[[Sequence[str]], object] | None = None) -> float:
    """Return an approximate bit gain from fitting CLA to a symbol stream.

    This uses the current two-part scorer as the model cost and an iid empirical
    code for the raw symbol stream as the no-grammar baseline.  It is a Phase-0
    calibration diagnostic, not a final scientific grammar-survival score.
    """

    if fit is None:
        model = CLA.simple(max_iterations=8, miner="suffix_trie", category_method="js").fit_symbols(symbols)
    else:
        model = fit(symbols)
    baseline = empirical_code_length_bits(tuple(symbols))
    grammar_cost = TwoPartMDLScorer().score(model.state).total
    return baseline - grammar_cost


def shuffled_surrogate(symbols: Sequence[str], *, seed: int = 0) -> tuple[str, ...]:
    """Shuffle a stream, preserving marginal frequencies and destroying order."""

    out = list(symbols)
    random.Random(seed).shuffle(out)
    return tuple(out)


def surrogate_excess_compression(
    symbols: Sequence[str],
    *,
    runs: int = 8,
    seed: int = 0,
    fit: Callable[[Sequence[str]], object] | None = None,
) -> SurrogateCompressionResult:
    """Compute real compression gain minus shuffled-surrogate gain."""

    if runs < 1:
        raise ValueError("runs must be at least 1")
    stream = tuple(symbols)
    real_gain = cla_compression_gain_bits(stream, fit=fit)
    surrogate_gains = tuple(
        cla_compression_gain_bits(shuffled_surrogate(stream, seed=seed + i), fit=fit) for i in range(runs)
    )
    surrogate_mean = sum(surrogate_gains) / len(surrogate_gains)
    return SurrogateCompressionResult(
        real_gain_bits=real_gain,
        surrogate_gain_bits_mean=surrogate_mean,
        delta_grammar_bits=real_gain - surrogate_mean,
        surrogate_gain_bits=surrogate_gains,
    )


def heldout_next_symbol_log_loss(
    symbols: Sequence[str],
    *,
    train_fraction: float = 0.7,
    order: int = 1,
    alpha: float = 0.5,
) -> HeldoutLogLossResult:
    """Evaluate smoothed n-gram next-symbol predictive loss on a holdout suffix.

    This is a baseline predictive diagnostic until CLA exposes a proper grammar
    likelihood / predictive distribution.
    """

    stream = tuple(symbols)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if order < 0:
        raise ValueError("order must be non-negative")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if len(stream) <= order + 1:
        raise ValueError("symbol stream is too short for heldout evaluation")

    split = max(order + 1, min(len(stream) - 1, int(len(stream) * train_fraction)))
    train = stream[:split]
    vocab = tuple(sorted(set(train) | set(stream[split:])))
    vocab_size = len(vocab)
    context_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    context_totals: Counter[tuple[str, ...]] = Counter()
    for i in range(order, len(train)):
        context = train[i - order:i] if order else ()
        context_counts[context][train[i]] += 1
        context_totals[context] += 1

    uniform_context = ()
    if order and uniform_context not in context_counts:
        # Backoff context used when a holdout context was unseen.
        for sym in train:
            context_counts[uniform_context][sym] += 1
            context_totals[uniform_context] += 1

    total_loss = 0.0
    evaluated = 0
    for i in range(split, len(stream)):
        context = stream[i - order:i] if order else ()
        if context not in context_counts:
            context = uniform_context
        count = context_counts[context][stream[i]]
        denom = context_totals[context] + alpha * vocab_size
        prob = (count + alpha) / denom
        total_loss += -math.log2(prob)
        evaluated += 1

    loss = total_loss / evaluated
    return HeldoutLogLossResult(
        log_loss_bits_per_symbol=loss,
        perplexity=2.0 ** loss,
        evaluated_symbols=evaluated,
        vocabulary_size=vocab_size,
        order=order,
    )
