# Plan 015 — IMAS fast LDAT to LM

Status: planned 2026-11-03 for [spec 015](spec.md). Constitution: [AGENTS.md](../../AGENTS.md). [Ordered tasks](tasks.md).

## Inputs seen at planning

- Legacy `scripts_imas/imas_listmode.py` (848 lines): per-record Python, bunch 20 000 sides, Keras predict per bunch, 32-file spawn pools, `skew_map_ch = {}`.
- NN: `IMASNoCatY.h5`, `IMASNoCatDOI.h5` (337 kB each), inputs `[cat_y (n,1), E (n,8), cat_doi (n,1), E (n,8)]` float32. CPU (TF 2.13, this PC): about 1.3 M sides/s at batch 65 536.
- Map: TIME channel `local_map = (x, 64.513, slab)`; ENERGY channel `(64.513, y, row)`. Pair file `pares_5SR.txt`: 3 900 rows `p s0 s1`.
- `np.dtype(CoincidenceV5)`: 24 B, aligned; `LMHeader`: 176 B.

## Architecture (`src/imas/listmode/`)

1. **`select.py`** (FR-2, FR-3, FR-5 inputs). `LmLookup`: the `ImasLookup` arrays plus ENERGY row per channel, `SLAB_EN_MAP` mu, walk-calibration mu and a dense SM × SM pair table (`pair`, `swapped`). Numba `_lm_select` over one `LdatBatch` reuses `select._keep`, `_walk_ingredients`, `_candidates` and `kernels._min_ch`, `_max_module`, `_time_hits`. Per record a code (0 accepted, 1–7 FR-3); per accepted record ints (record, top channel, SM, MM per side, pair, swapped, raw `dt`, raw `t_min`, tie), keV per side, walk ingredients per side, ENERGY rows `(n, 2, 8)` float32 and candidates `(n, 2, 7)`. Sides stay in LDAT order; the swap is applied when records are built.
2. **`timing.py`** (FR-4). `Timing.from_options`: `em` (IMAS `CanonicalLut` + `WalkLut`, `check_tables`, rule checks), `legacy` (`outputs.read_legacy`, rule `top`), `none`. `Timing.dt(features)` → int16 `dt` (−32 700 where uncorrected or outside int16) and counts, through `walk.correct_dt(policy="drop")`, with the walk keV = 511 / walk mu × ENERGY sum.
3. **`records.py`** (FR-5, FR-6). LM dtype from `CoincidenceV5`; `pixels(v)`; `build(features, dt, y_mm, doi, t0)` applies the swap; `header_bytes(...)` with the legacy fields.
4. **`nn.py`** (FR-5). `Predictor.predict(E (n, 8)) → (y_mm, doi)`; `KerasPredictor` imports TensorFlow only in the NN process, fixed batch.
5. **`job.py`** (FR-7–FR-10). `Settings` + fingerprint. Spawned workers take file indices from a task queue, stream, select and time, and put chunks (file, chunk index, record arrays without positions, ENERGY rows) on a bounded result queue, then a `done` message with counts and debug histograms. The main process is the NN process: predict per chunk, build records, append to `<segment>.lm.part`; at `done`, write `<segment>.json`, rename. Chunks are the stream's fixed batches, so segments do not depend on worker count. Merge: header + segments in natural order → `<base>_all.lm`, `_all.json`, `_log.txt`; debug plots.
6. **`__main__.py`** (FR-1). docopt usage, defaults, `scripts_imas/imas_ldat2lm.py` shim (untracked).

## Decisions

- One NN process (owner); its CPU rate (~0.65 M coincidences/s) may bound many workers, recorded by FR-11 timings. A GPU on the IMAS PC raises it.
- Workers send only what the NN process needs; timing is done in the worker.
- Reference for parity: an untracked copy of the legacy loop with the floor, both fixes and the `listmode_skew_walk.md` timing class (`scripts_imas/imas_listmode_reference_015.py`), frozen as a golden LM with provenance.

## FR → component

| FR | Component | Check |
|---|---|---|
| 1 | `__main__.py` | `tests/test_imas_listmode.py` CLI tests |
| 2, 3 | `select.py` | synthetic cut tests |
| 4 | `timing.py` | synthetic timing tests |
| 5, 6 | `records.py`, `nn.py` | fake-predictor tests, real-data parity |
| 7–10 | `job.py` | job tests (workers 1 vs 3, continue, failure, debug) |
| 11 | `job.py` timings | real-data speed record |
