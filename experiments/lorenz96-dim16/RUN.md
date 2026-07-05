# Experiment: Lorenz-96 Dimensionality Scaling — dim=16
**Date:** 2026-07-05
**System:** Lorenz-96 (dim=16)
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system lorenz96 --steps 256 --discard 64 --bins 4 --iterations 8 --dimension 16 --seed 0
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Lorenz-96 |
| dimension | 16 |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.01 |
| F | 8.0 |
| initial | [8.0]*15 + [8.01] |
| seed | 0 |

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 121 |
| score_total | 248.7 |
| exact_reconstruction | True |
| rules_learned | 3 |

## Conclusion

CLA discovers only 3 chunk rules at dim=16. 121 unique symbols from 16D trajectory. Score 248.7 bits (0.97 bits/symbol). Near-incompressible: nearly all symbols are unique. Only highly repeated sequences are captured by chunk rules. This is the regime where dimension reduction or coarser binning would help.
