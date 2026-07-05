# Experiment: Rössler M1 Baseline
**Date:** 2026-07-05
**System:** Rössler
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system rossler --steps 256 --discard 64 --bins 4 --iterations 8
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Rössler |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.05 |
| initial | [0.1, 0.0, 0.0] |
| a | 0.2 |
| b | 0.2 |
| c | 5.7 |

## Seed

deterministic (no RNG; RK4 fixed-step)

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 24 |
| score_total | 189.2 |
| exact_reconstruction | True |
| rules_learned | 8 |

## Conclusion

CLA discovers 8 chunk rules. 24 unique M1 symbols from 3D trajectory. Exact reconstruction verified. Score 189.2 bits (0.74 bits/symbol). Similar complexity to Lorenz-63.
