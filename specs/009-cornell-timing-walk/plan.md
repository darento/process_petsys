# Plan — Cornell time-walk correction

Status: approved plan for [spec 009](spec.md), 2026-10-19. [Ordered tasks](tasks.md). Evidence (untracked): [docs/research/cornell-timing-walk.md](../../docs/research/cornell-timing-walk.md).

## Recommendation

Add one module, `src/cornell/skew/walk.py`, inside the spec 008 package. It holds:
- the walk-input mapping;
- the walk LUT contract;
- the point-source fit (later also the phantom fit);
- the per-event correction used by every consumer.

The existing pieces change only at their edges:
- **ingest:** stores the raw walk ingredients.
- **cache:** schema version 3.
- **job:** applies walk before the pair fits.
- **ctr:** reports with and without walk.
- **listmode:** corrects `dt`.
- **Manager settings:** two profile keys.

The probe already showed the numerical approach works (held-out model-free 423.6 → 394.0 ps). The engineering risk is **definition parity**: the fit and listmode must compute the same four inputs for the same event.

## Architecture

### Walk inputs (FR-1)

The work splits into two stages, so the cache stays independent of listmode calibration files.

**1. Raw ingredients** come from the selected minimodule's hits, inside the existing compiled selection (`ingest._select`, and listmode's `process_batch` arrays):

| Field | Meaning |
|---|---|
| `q_top` | charge of the timestamped TIME hit |
| `q_time` | summed TIME-hit charge |
| `e_raw` | summed ENERGY charge (already the keV numerator) |
| `doi_raw` | `calculate_DOI_vectorized` ratio, listmode definition |
| `y_cog` | listmode `y_centroid_early` |
| `time_channel` | |
| `slab` | |

**2. Mapping** `walk.inputs(ingredients, context) -> (charge, share, doi_mm, kev, status)`. `WalkContext` is built from the listmode files:
- calibration: per-slab or region;
- COG limits;
- DOI limits.

It reuses `listmode.compute_region_vectorized`, `convert_energy_numba`, `_limit_array` and the DOI linear mapping, rather than re-implementing them. Status codes count missing calibration, region, DOI limits and zero sums.

**Parity is defined on shared events.** Listmode and the skew ingest choose the slab of ambiguous events with different RNGs (listmode keeps the reference `np.random` draws; spec 008 uses a seeded hash). A shared event is one where both pipelines picked the same TIME channel and slab on both sides. On those, the four inputs must be equal, the floating-point ones within 1e-6 relative.

### Walk LUT contract (FR-2)

`cornell-walk-lut/1`: one JSON table with a `.json` manifest beside it.

**Blocks:**
- `walk2d`: `charge_edges[21]`, `share_edges[11]`, `offset_ps[20][10]`, `events[20][10]`
- `doi`: `edges[21]`, `offset_ps[20]`, `events[20]`
- `kev`: `edges[21]`, `offset_ps[20]`, `events[20]`

**Rules:**
- **Convention:** `subtract_per_side`.
- **Gauge:** `zero_mean_over_fit_events`.
- **Edges:** `clamp`.
- **Minimum events per bin:** 200. Bins below it are filled from the nearest populated bin (Manhattan distance in the 2D table) and listed.
- **Clamping counts** are reported by the consumer, not stored.

**Correction:** `walk.correction(lut, inputs) -> (ps, clamped flags)`, numba-compiled, the same call in every consumer.

**Manifest:**
- schema, `created_utc`, inputs with SHA-256 (LDATs), skew LUT SHA-256;
- config/map SHA-256, per-slab `.encal` SHA-256, listmode calibration/COG/DOI files SHA-256;
- cuts, fit window [300, 700] keV, seed;
- split rule (`record % 2`), fit/held-out event counts, iterations, core window, `fit_kind` (`point_source` | `phantom`).

The writer refuses an existing path.

### Point-source fit (FR-3, FR-8)

`python -m src.cornell.skew walk WALK.json`, schema `cornell-walk-fit/1`.

**The `WALK.json` file declares:**
- acquisitions (as in the CTR file);
- config and per-slab `.encal`;
- the listmode files;
- the frozen skew LUT and its SHA-256;
- the fit window and output.

**Steps:**
1. Run the spec 008 CTR pipeline (`ctr.frozen_inputs`, `localize`, `apply_geometry`), with the energy window replaced by the fit window. This is the same exception as the spec 008 `energy_range` rule: only `energy_range` may differ from the skew LUT's job config.
2. Residual = LUT-corrected `dt` minus geometric TOF.
3. Joint sparse least squares over the three blocks. Each event contributes +1 to side-1 bins and −1 to side-2 bins. Iterate four times with core |r − model| < 600 ps, add a ridge of 1e-6·trace/P, then zero-mean gauge.
4. Fit on even records; report the CTR on odd records before and after, with `ctr.fit_histogram`.

**Outputs:**
- the LUT and its manifest;
- a report JSON: curves, counts, held-out CTR, iteration residuals;
- PNGs: walk2d heat map with counts, DOI and keV curves, residual-vs-input profiles before and after.

Memory stays bounded: accumulate the normal equations (P ≈ 240 columns, so a 240 × 240 matrix) per batch instead of keeping the events. The core-window iterations re-stream; there are 4 passes over about 1 GB of point-source LDAT, under a minute in total.

### CTR with walk (FR-5)

- **CTR file:** gains `walk_lut`, its `walk_lut_sha256` and the listmode files.
- **Accumulator:** keeps a second histogram set (walk on), on the same events.
- **Report:** `results_without_walk` and `results` (with walk).
- **Round trip:** `previous_reports: [paths]` adds a summary table of each earlier report's pooled and per-position numbers. That is the round-trip summary.
- **Refusals:**
  - a walk LUT fitted with a different skew LUT than the one applied (manifest SHA-256);
  - a skew LUT whose manifest names a walk LUT, applied without it.

### Skew job with walk (FR-4)

- **Cache, schema 3:**
  - `FEATURE_DTYPE` gains the raw ingredients per side: `q_top_*`, `q_time_*`, `e_raw_*`, `doi_raw_*` (`y_*` exists).
  - The job fingerprint adds the schema, so old caches are refused, never reused.
  - The r7 cache is rebuilt once: about 15 min on Windows (8 workers), about +1 GB on top of the current 13 GB.
- **Job file:** optional `walk_lut` plus the listmode files.
- **Observation stage:** maps ingredients to inputs and does `dt_ps -= round(W_i − W_j)`, so `OBSERVATION_DTYPE` stays int64. Pair fits and the solver are unchanged. Walk-input status rejections are counted per run.
- **LUT manifest:** `provenance.walk_lut` and its SHA-256. Every consumer enforces the pairing.

### Listmode (FR-6)

**`ListmodeContext.build`** gains optional skew delays and a walk LUT, plus a `WalkContext` built from the files listmode already loads (calibration, COG and DOI limits).

**`process_batch`:**
- The ingredients come from arrays it already holds. The new ones are `q_top` (charge of the max-energy TIME hit of the final minimodule, beside `get_timestamp_of_max_energy_hit_vectorized`) and `q_time`.
- After `dt_values`: `dt_corr = dt − (b₁ + W₁) + (b₂ + W₂)`, rounded to int32. The int16 swap and wrap logic is unchanged.
- **Events without a skew delay** (channel not applied): `dt` is left uncorrected and counted, matching CTR "no applied delay". This is open to change at T7.
- **Without tables:** the code path is skipped, so the output is byte-identical.

**Provenance (sidecar):** both SHA-256 values, the correction counts (clamped per input, missing, no delay) and the table manifests' calibration SHA-256 checks.

**Refusals:**
- calibration, COG or DOI files that differ from the walk LUT manifest;
- a map or config other than the skew LUT job's, except `energy_range`.

**Manager:**
- `settings.py` gets the profile path keys `skew_lut` and `walk_lut`, optional and checked like `cog_limits_file`.
- `workflow.py` puts them in the listmode request `files`.
- `src/cornell/cli.py` accepts `files.skew_lut` / `files.walk_lut` (`"file?"`).
- No GUI widgets. The profile editor shows the keys only if it lists path keys generically; verify at T7.
- **PETsys Manager:** behaviour changes only when the keys are set. Patch-version bump and CHANGELOG entry at ship (AGENTS.md).

### Phantom fit (FR-7, last)

- `walk.fit_phantom`: on normalization-phantom observations with the skew LUT frozen, it fits one time shift per walk bin.
- **Method:** the same two-peak shell template as `pairs.py`, but grouping events by their side-1 and side-2 walk bins instead of by channel pair. This is linear in the bin offsets; solved like the point-source fit with the template midpoint as the expected TOF.
- **Acceptance:** each populated bin within a declared tolerance (proposal: 10 ps, or 3σ) of the point-source table, and the held-out point-source CTR equal within uncertainty. Otherwise it is recorded as a limitation.

## Data, inputs and sizes

| Data | Use | Size / time |
|---|---|---|
| X0Y0Z0 point source (local, uncorrected) | walk fit | 486k CTR events at T9b cuts; more at 300–700 keV |
| X0Y-1/-6/-10Z0 | held-out validation | 1.35M events |
| r7 (local, 16 splits, ledger) | skew re-solve, phantom fit | rebuild ~15 min, ~14 GB cache |
| T10 skew LUT `time_delays_t10.csv` | round-trip step 1 | SHA-256 `b1b211c1...cd6b74` |

The listmode files are the Manager profile's: calibration, COG limits, DOI limits, pair and region maps. Their local paths will be confirmed at T1. If production listmode uses the per-slab `.encal` (positions 1), the keV of the fit and of listmode coincide exactly.

## Decisions and alternatives

| Decision | Alternative rejected | Reason |
|---|---|---|
| Global 2D + 1D tables | per-channel walk | ~1 ps gain for 30× the parameters (probe) |
| Raw ingredients in cache, limits applied later | cache mapped DOI/keV | cache stays valid when listmode limit files change |
| Walk inside `src/cornell/skew` | new package | reuses ingest/geometry/CTR/LUT code; one timing CLI |
| Corrected `dt_ps` stays int64 (rounded ps) | float observations | pair fits/solver unchanged; 0.5 ps rounding ≪ σ |
| Listmode applies offline | native PETsys | PETsys has no walk correction; native skew table deleted |
| Quantile edges frozen at fit | uniform edges | equal statistics per bin; tails clamp |

## Risks

- **Definition parity** (DOI and region code paths are duplicated today). T1 makes listmode's own functions the source and adds a real-data parity test.
- **The ±5 ns satellites and noise triggers** bias the fits if not excluded. The core window handles them; the counts are reported.
- **Same-day validation only.** The other-day stability check stays open, with a recorded limitation until the new acquisition exists.
- **Phantom fit:** template bias from the finite shell could exceed the tolerance. A limitation is the accepted outcome.

## Requirements to components

| FR | Component | Check |
|---|---|---|
| FR-1 | `walk.inputs`, `WalkContext`; ingredients in `ingest._select` and `listmode.process_batch` | T1 |
| FR-2 | `walk.WalkLut` (read/write/correction) | T2 |
| FR-3 | `walk.fit_point_source`, CLI `walk` | T3, T6 |
| FR-4 | `cache` schema 3, `job` walk option, `lut` manifest pairing | T5, T6 |
| FR-5 | `ctr` walk option, `previous_reports` | T4, T6 |
| FR-6 | `listmode`, Manager `settings`/`workflow`, `cornell.cli` | T7 |
| FR-7 | `walk.fit_phantom` | T8 |
| FR-8 | `walk` plots | T3 |
