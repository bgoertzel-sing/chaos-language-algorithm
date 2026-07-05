# Experiment: Lorenz-96 Dimensionality Scaling — dim=24
**Date:** 2026-07-05
**System:** Lorenz-96 (dim=24)
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system lorenz96 --steps 256 --discard 64 --bins 4 --iterations 8 --dimension 24 --seed 0
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Lorenz-96 |
| dimension | 24 |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.01 |
| F | 8.0 |
| initial | [8.0]*23 + [8.01] |
| seed | 0 |

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 148 |
| score_total | 251.5 |
| exact_reconstruction | True |
| rules_learned | 5 |

## Conclusion

CLA discovers 5 chunk rules at dim=24. 148 unique symbols from 24D trajectory. Score 251.5 bits (0.98 bits/symbol). Includes nested chunk (N11 -> N3 N3). Near-incompressible — 58% of symbols are unique. This exceeds OmegaSim starter-vector dimensionality (~20 fields). CLA maintains exact reconstruction at all tested dimensions (1-24).
