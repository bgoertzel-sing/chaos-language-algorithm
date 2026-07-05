# Experiment: Logistic Map M1 Baseline
**Date:** 2026-07-05
**System:** Logistic Map
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system logistic --steps 256 --discard 64 --bins 4 --iterations 8
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Logistic Map |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| r | 4.0 |
| x0 | 0.123 |

## Seed

deterministic (no RNG; discrete map)

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 4 |
| score_total | 113.2 |
| exact_reconstruction | True |
| rules_learned | 8 |

## Conclusion

CLA discovers 8 chunk rules including nested hierarchy (N0→N3→N9, N0→N4→N5). 4 unique symbols. Score 113.2 bits (0.44 bits/symbol). Chaotic 1D map produces moderate compressibility.
