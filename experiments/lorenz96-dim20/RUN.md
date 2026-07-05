# Experiment: Lorenz-96 Dimensionality Scaling — dim=20
**Date:** 2026-07-05
**System:** Lorenz-96 (dim=20)
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system lorenz96 --steps 256 --discard 64 --bins 4 --iterations 8 --dimension 20 --seed 0
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Lorenz-96 |
| dimension | 20 |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.01 |
| F | 8.0 |
| initial | [8.0]*19 + [8.01] |
| seed | 0 |

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 127 |
| score_total | 251.4 |
| exact_reconstruction | True |
| rules_learned | 6 |

## Conclusion

CLA discovers 6 chunk rules at dim=20. 127 unique symbols from 20D trajectory. Score 251.4 bits (0.98 bits/symbol). Approaching raw symbol entropy — nearly incompressible at this dimension with 4 bins and 256 steps. This dimensionality is comparable to OmegaSim starter-vector dimensionality (~20-30 state fields per tick). CLA still maintains exact reconstruction.
