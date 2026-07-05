# Experiment: Lorenz-96 M1 Baseline
**Date:** 2026-07-05
**System:** Lorenz-96
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system lorenz96 --steps 256 --discard 64 --bins 4 --iterations 8
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Lorenz-96 |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.01 |
| initial | [8.0, 8.0, 8.0, 8.0, 8.01] |
| F | 8.0 |
| dimension | 5 |

## Seed

deterministic (no RNG; RK4 fixed-step)

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 56 |
| score_total | 234.2 |
| exact_reconstruction | True |
| rules_learned | 8 |

## Conclusion

CLA discovers 8 chunk rules. 56 unique symbols from 5D trajectory (highest diversity). Score 234.2 bits (0.92 bits/symbol). Higher dimensional system produces richer symbolic vocabulary but lower compression.
