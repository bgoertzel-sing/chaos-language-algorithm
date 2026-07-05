"""Small CLI benchmark for CLA M1 symbolization on control attractors."""
from __future__ import annotations

import argparse
import json

from chaoslang import CLA
from chaoslang.benchmarks.attractors import lorenz63, m1_symbolize, rossler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny deterministic M1/control CLA benchmark")
    parser.add_argument("--system", choices=("lorenz63", "rossler"), default="lorenz63")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--discard", type=int, default=16)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args(argv)

    if args.system == "lorenz63":
        trajectory = lorenz63(steps=args.steps, discard=args.discard)
    else:
        trajectory = rossler(steps=args.steps, discard=args.discard)
    symbols = m1_symbolize(trajectory, bins=args.bins)
    model = CLA.simple(max_iterations=args.iterations).fit_symbols(symbols)
    result = {
        "system": args.system,
        "steps": args.steps,
        "discard": args.discard,
        "bins": args.bins,
        "symbol_count": len(symbols),
        "unique_symbols": len(set(symbols)),
        "history": list(model.state.history),
        "score_total": model.score.total,
        "exact_reconstruction": model.expand() == symbols,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
