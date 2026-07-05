# Experiment: Lorenz-63 M1 Baseline
**Date:** 2026-07-05
**System:** Lorenz-63
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system lorenz63 --steps 256 --discard 64 --bins 4 --iterations 8
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Lorenz-63 |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.01 |
| initial | [1.0, 1.0, 1.0] |
| sigma | 10.0 |
| rho | 28.0 |
| beta | 2.6666666666666665 |

## Seed

deterministic (no RNG; RK4 fixed-step)

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 21 |
| score_total | 187.2 |
| exact_reconstruction | True |
| rules_learned | 8 |

## Conclusion

CLA successfully discovers 8 chunk rules with nested references. The pruning bug (dangling nonterminal in production RHS) was fixed in this run. Exact reconstruction verified. Score 187.2 bits for 256 symbols (0.73 bits/symbol).
