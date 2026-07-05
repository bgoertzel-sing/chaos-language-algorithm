# Experiment: Mackey-Glass M1 Baseline
**Date:** 2026-07-05
**System:** Mackey-Glass
**Status:** Complete

## Command

```bash
PYTHONPATH=src /usr/local/bin/python3.9 -m chaoslang.benchmarks.m1_controls --system mackey-glass --steps 256 --discard 64 --bins 4 --iterations 8
```

## Parameters

| Parameter | Value |
|-----------|-------|
| system | Mackey-Glass |
| steps | 256 |
| discard | 64 |
| bins | 4 |
| iterations | 8 |
| dt | 0.1 |
| initial | 0.5 |
| tau | 17 |
| beta | 0.2 |
| gamma | 0.1 |
| n | 10.0 |

## Seed

deterministic (no RNG; Euler with discrete delay buffer)

## Outputs

| Metric | Value |
|--------|-------|
| symbol_count | 256 |
| unique_symbols | 4 |
| score_total | 44.4 |
| exact_reconstruction | True |
| rules_learned | 6 |

## Conclusion

CLA discovers 6 chunk rules including nested hierarchy (N0→N1→N5). Only 4 unique symbols from scalar trajectory. Best compression ratio (0.17 bits/symbol). Highly repetitive structure as expected for Mackey-Glass with these parameters.
