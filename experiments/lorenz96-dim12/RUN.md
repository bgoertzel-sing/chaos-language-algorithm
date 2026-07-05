# Experiment: Lorenz-96 Dimensionality Scaling — dim=12
**Date:** 2026-07-05
**System:** Lorenz-96 (dim=12)
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system lorenz96 --steps 256 --discard 64 --bins 4 --iterations 8 --dimension 12 --seed 0
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Lorenz-96 |
| dimension | 12 |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.01 |
| F | 8.0 |
| initial | [8.0]*11 + [8.01] |
| seed | 0 |

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 107 |
| score_total | 241.4 |
| exact_reconstruction | True |
| rules_learned | 6 |

## Conclusion

CLA discovers 6 chunk rules at dim=12. 107 unique symbols from 12D trajectory. Score 241.4 bits (0.94 bits/symbol). Includes nested chunk (N19 -> N2 N2). Compression degrading as dimension increases — approaching near-incompressible regime.
