# CLA Attractor Benchmark Mandate

## Status: ACTIVE (2026-07-05)

## Context

OmegaSim and RelaLeap are paused. The MacBook (`defective-mind-uploader`) is repurposed for Chaos Language Algorithm (CLA) attractor benchmark work. This is the critical-path prerequisite for resuming OmegaSim: CLA must demonstrate it can recognize grammatical strange-attractor structure in known systems before attempting OmegaSim-scale traces.

## Repository

`~/Documents/Codex/2026-06-17/i-want-to-have-you-work/chaos-language-algorithm` (clone of https://github.com/bgoertzel-sing/chaos-language-algorithm)

Use Python 3.9: `/usr/local/bin/python3.9`

```bash
cd ~/Documents/Codex/2026-06-17/i-want-to-have-you-work/chaos-language-algorithm
PYTHONPATH=src /usr/local/bin/python3.9 -m unittest discover -s tests -v
```

## Current state (2026-07-05)

- Sprint-1 symbolic-string MVP complete (chunks, categories, MDL, exact reconstruction)
- M1 fixed-partition symbolization implemented for Lorenz-63, Rossler, Mackey-Glass, Lorenz-96, and logistic map
- Mackey-Glass generator: Euler method with discrete delay buffer (tau=17, beta=0.2, gamma=0.1, n=10)
- Lorenz-96 generator: RK4 with variable dimension from initial, default F=8 with perturbation
- m1_controls CLI supports all 5 systems with --dimension and --seed parameters
- 39 tests passing (includes nested-prune regression tests)
- Latest commit: `1501346` (dimensionality scaling experiments + CLI params)
- Note: git push is done from the OpenClaw host (not the Mac) due to macOS keychain SSH limitations
- Benchmark experiments complete for all 5 attractor systems (Lorenz-63, Rossler, Mackey-Glass, Lorenz-96, logistic map)
- Dimensionality scaling experiments complete: Lorenz-96 dims 8,12,16,20,24 -- all pass exact reconstruction
- Nested prune bug fixed: pruned rules are now expanded in other production RHS (not just in parse)
- Compression degrades with dimension: 0.17 bits/symbol (Mackey-Glass dim=1) to 0.98 bits/symbol (Lorenz-96 dim=24)
- Near-incompressible regime reached at dim~16 with 4 bins and 256 steps

## Mandate: Attractor benchmark sprint

### Remaining tasks

1. ~~Mackey-Glass and Lorenz-96 trajectory generators~~ ✅ DONE
2. ~~Recorded benchmark experiments~~ ✅ DONE — run CLA on Lorenz-63, Rössler, Mackey-Glass, Lorenz-96, and logistic map with recorded seeds/parameters. Create experiment records under `experiments/` with RUN.md (command, parameters, seed, outputs, conclusion).
3. ~~Dimensionality scaling~~ ✅ DONE — include benchmark dimensions up to OmegaSim starter-vector dimensionality. Defer much higher-dimensional traces until a dedicated dimension-reduction step exists.
4. **JS-divergence context clustering** — integrate into category proposal generation (currently a seam only).
5. **Persistence** — JSON grammar/state format, edit-log replay, deterministic seeds, stable text rendering.

### Next task

Task 4: JS-divergence context clustering -- integrate into category proposal generation (currently a seam only). Then Task 5: Persistence (JSON grammar/state format, edit-log replay, deterministic seeds).

### Constraints

- Local-only, no GPU, no paid compute.
- No push to GitHub without running tests first.
- Use `PYTHONPATH=src /usr/local/bin/python3.9` for all Python commands.
- Commit after each tested coherent slice with message describing the change.
- Push to `origin/main` after each tested commit (repo is approved for pushes).
- Push is done from OpenClaw host, not the Mac (macOS keychain SSH limitation).
- Record each benchmark run as an experiment directory with `RUN.md` containing command, parameters, seed, outputs, and conclusion.

### Workflow

1. Pull latest from `origin/main`.
2. Pick the next incomplete task from the list above.
3. Implement + test locally.
4. Commit + push (push from OpenClaw host if Mac keychain is unavailable over SSH).
5. Update this file's "Current state" section.
6. Report progress.

### What to report

- Task completed this cycle.
- Tests run and result.
- Commit hash.
- Next task to pick up.
- Any blockers or issues encountered.
