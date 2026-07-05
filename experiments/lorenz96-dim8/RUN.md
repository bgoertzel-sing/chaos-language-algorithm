# Experiment: Lorenz-96 Dimensionality Scaling — dim=8
**Date:** 2026-07-05
**System:** Lorenz-96 (dim=8)
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system lorenz96 --steps 256 --discard 64 --bins 4 --iterations 8 --dimension 8 --seed 0
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Lorenz-96 |
| dimension | 8 |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.01 |
| F | 8.0 |
| initial | [8.0]*7 + [8.01] |
| seed | 0 |

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 78 |
| score_total | 229.2 |
| exact_reconstruction | True |
| rules_learned | 8 |

## Conclusion

CLA discovers 8 chunk rules at dim=8. 78 unique symbols from 8D trajectory (4^8 = 65536 possible symbols). Score 229.2 bits (0.90 bits/symbol). Exact reconstruction verified. Compression ratio moderate — higher dim than dim=5 baseline (56 symbols) but still tractable.
