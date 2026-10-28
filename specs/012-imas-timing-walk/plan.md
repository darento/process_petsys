# Plan 012 — IMAS time-walk correction

Status: implemented; [spec 012](spec.md) shipped 2026-10-27. Constitution: [AGENTS.md](../../AGENTS.md). [Ordered tasks](tasks.md).

## Inputs seen at planning (2026-10-27)

- **Ship job (spec 011 T16):** `PETSYS_DATA_DIR/IMAS_work/job_ship.json`, work dir `job_ship/`.
  - Runs 7, 8, 9 and 11 on `D:\David_Skew`; split 07 is `encal` + `train`, split 36 is held-out.
  - 47.1 M train and 1.54 M held-out observations; 2.39 h total (EM 56 min, pairs 30 min), peak 9.2 GB.
  - Delivered skew LUT `lut_em.csv` (SHA-256 `8eadd8c7…`): held-out σ 271.8 ps; 4-position 22Na CTR 468.1 ps (legacy 532.2).
- **Walk ingredients already computed:** `src/imas/skew/select.py:_walk_ingredients` fills `WALK_FEATURES` (`q_top`, `q_time`, `e_raw`, `doi_ratio` per side) in every `FeatureBatch`. The feature cache keeps them (`SCHEMA_VERSION` 3), but the streamed observation rows (`OBSERVATION_DTYPE`) do not: the ship job's observation sets cannot feed a walk fit.
- **keV:** the observation's `energy_kev_i/j` is already the run's job calibration (`511 / mu × e_raw`, spec 011 FR-4).
- **Cornell walk (spec 009):** `src/cornell/skew/walk.py` holds a generic table (`WalkLut`, `quantile_edges`, `design`, `fit_tables`, `walk_difference`, `check_pairing`, `compare_tables`), bound to Cornell only through module constants (`BLOCKS` with `doi_mm`, `LUT_SCHEMA`, `CONVENTION`). Its phantom fit (`run_phantom`) assigns each event to the nearer shell branch and re-assigns after each walk update.
- **Disk:** C: 206 GB free; D: (HDD) 332 GB free.

## Architecture

1. **`src/timing/walk.py` — scanner-neutral walk table and fit** (moved from Cornell).
   - A frozen `WalkModel(schema, convention, blocks, default_bins)` replaces the module constants.
   - `WalkLut`, `quantile_edges`, `design`, `fit_tables`, `walk_difference`, `check_pairing` and `compare_tables` take the model.
   - `fit_two_branch(side1, side2, dt, near, far, train, ...)` holds the nearer-branch loop that `run_phantom` runs today.
   - `src/cornell/skew/walk.py` keeps its names (`WalkLut`, `BLOCKS`, `LUT_SCHEMA`, …) bound to the Cornell model, so Cornell files, reports and tests do not change.
2. **`src/imas/skew/walk.py` — IMAS walk** (new).
   - The IMAS model: schema `imas-walk-lut/1`, blocks `walk2d (charge, share)`, `doi (doi_ratio)`, `kev (kev)`, bins 20 × 10, 20, 20.
   - `walk_inputs(q_top, q_time, doi_ratio, kev)` (FR-1), the only place inputs are derived.
   - The bar fit command (FR-3), the correction function (FR-6) and the walk-corrected observation view (FR-4).
3. **Observation rows carry walk inputs** (`src/timing/cache.py`, `src/timing/stream.py`).
   - An extended dtype, `OBSERVATION_WALK_DTYPE`: `OBSERVATION_DTYPE` plus `q_top_i/j`, `q_time_i/j`, `doi_ratio_i/j` (float32, +24 bytes per row, ~19 %).
   - `canonical_observations(..., walk=True)` swaps them with the other per-side columns.
   - `stream_observations(..., carry_walk=True)` writes the extended rows. Its fingerprint records the dtype and `"walk": {"carried": [...]}`, so a set without walk inputs is never reused as one with them.
   - Cornell keeps `OBSERVATION_DTYPE`, `SCHEMA_VERSION` 3 and its fingerprints.
4. **IMAS job and CTR** (`src/imas/skew/job.py`, `ctr.py`, `outputs.py`, `__main__.py`).
   - The job always streams with walk inputs.
   - New job key `walk_lut` and CLI flag `run --observe-only` (stages 1–2 only).
   - The CTR file may declare `walks` (skew LUT + walk LUT); `export --walk-lut`; new commands `walk` and `walk-plot`.

### Alternatives rejected

- **Bumping the shared `SCHEMA_VERSION` (spec FR-4 wording):** every Cornell cache on the Cornell PC (~2.5 TB of runs) would be refused and re-ingested for no change in its rows. The extended dtype and the recorded `walk` entry give the same protection: a set without walk inputs is refused for a walk fit or a walk solve. The spec's FR-4 wording is updated accordingly.
- **Correcting `dt` while streaming (Cornell's walk adapter):** every walk LUT would need its own 1.2 TB HDD pass (~1 h). Correcting when a partition is loaded lets one walk-carrying observation set serve the walk fit, the EM re-solve and the refit.
- **Walk inside the EM, with branch responsibilities (FR-3 option 2):** this changes the shared EM kernels (Cornell's estimator too) and couples two solvers. At IMAS the two bar branches are 2.14 ns apart against a held-out σ of 272 ps. An event lands nearer the wrong branch only beyond 1.07 ns (≈ 3.9 σ, < 1e-4 of Gaussian events; the flat background is cut by the core window). So nearer-branch assignment, re-assigned after each walk update, loses almost nothing. Chosen; the report counts branch flips between iterations.
- **Per-run walk fits:** out of scope; the pooled train events are fitted once.

## Walk inputs (FR-1)

`walk_inputs(q_top, q_time, doi_ratio, kev) -> ({charge, share, doi_ratio, kev}, status)` per side, as float64 with NaN where missing. Statuses, first missing input wins:

| Status | Condition |
| --- | --- |
| `no TIME charge` | `q_top < 0` or `q_time <= 0` or non-finite |
| `no ENERGY charge` | `doi_ratio <= 0` or non-finite (`_walk_ingredients` writes 0 without ENERGY) |
| `no calibration` | `kev` non-finite (no valid `mu` for the TIME channel) |

`share = q_top / q_time`; `charge = q_top` (a.u.). `doi_ratio` stays the raw light-sharing ratio (ENERGY sum / maximum), not mm. Callers: the walk-corrected view, the bar fit, the CTR and the correction function, from observation or feature columns.

## Walk-corrected observation view (FR-4)

`WalkCorrected(observations, walk_lut)` wraps an `ObservationSet` or `PooledObservations`; `partition(p)` returns:

- rows with every walk input on both sides;
- their `dt_ps` corrected by `rint(W_i − W_j)`, as Cornell's `walk_rows`.

Dropped rows (`walk input missing`) and clamps per input are counted per role. The EM, pair estimator and held-out checks read it like any observation set.

## IMAS job with walk (FR-4)

- `walk_lut` names a walk LUT. It is not part of the job fingerprint, which identifies the streamed events, so a walk solve reuses the step-0 work directory and its observation sets. Its outputs are named:
  - `lut_<estimator>_walk-<sha12>.csv`;
  - `job_report_walk-<sha12>.json`.

  An existing output is refused, never overwritten.
- The skew LUT provenance gains `walk_lut_sha256` and the walk LUT's path. The report records the walk counts.
- Before solving, the job refuses:
  - a walk LUT of another map or model;
  - an observation set without walk inputs.
- `run --observe-only` stops after the streams: this is round-trip step 0.

## Bar walk fit (FR-3)

`python -m src.imas.skew walk WALK.json`, schema `imas-walk-fit/1`:

```json
{"schema": "imas-walk-fit/1", "job": "job_ship.json", "work": "job_walk",
 "skew_lut": "job_ship/lut_em.csv", "skew_lut_sha256": "8eadd8c7...",
 "walk_lut": null, "output": "walk/walk1.json",
 "bins": {"charge": 20, "share": 10, "doi_ratio": 20, "kev": 20},
 "min_events": 200, "core_ps": 600, "iterations": 4, "branch_iterations": 3, "max_events": 20000000}
```

1. **Frozen inputs.**
   - The work dir's job fingerprint equals the skew LUT's `job_sha256`, and each run's `encal` SHA-256 equals the LUT's: same events.
   - When the skew LUT was solved with walk, `walk_lut` names that walk LUT, which is used only for the pairing record.
2. **Events.**
   - Train rows (role 0), every `stride`-th row so at most `max_events` are kept.
   - Skew-corrected `dt = dt − b_i + b_j`.
   - Branches `near/far` = midpoints of the two shell segments.
   - Walk inputs from `walk_inputs`.
   - Counts: rows, stride, no applied delay, walk input missing, kept.
3. **Fit.** `fit_two_branch`, for each of `branch_iterations` rounds:
   - nearer branch under the current model;
   - `fit_tables` (quantile edges from the fit events, iterated core window, sparse fill, zero-mean gauge);
   - the model is updated.

   The flip count between rounds is recorded.
4. **Held-out.** Role-2 rows (split 36), never fitted. Report the nearer-branch residual width (core RMS within ±600 ps and robust σ) before and after the walk, and the same on train.
5. **Outputs.** Walk LUT + manifest (FR-2 fields: job and SHA-256, skew LUT SHA-256, config/map SHA-256, per-run encal SHA-256, cuts, split, events, options, iterations) and `<output>.report.json`.

Memory: 20 M events × (8 inputs + 3 times) float32 ≈ 0.9 GB, plus the sparse design matrix (20 M × 6 non-zeros × 12 bytes ≈ 1.4 GB).

## Correction function and export (FR-6)

- `correct_dt(dt_ps, channel_1, channel_2, side_1, side_2, skew, walk, *, policy="keep")`, with `side_s` the raw walk ingredients (`q_top`, `q_time`, `doi_ratio`, `kev`). It returns:
  - the corrected `dt` (float ps);
  - the corrected mask;
  - the counts: no applied delay, walk input missing, clamped per input; under `drop`, also dropped.
- `correct_features(features, skew, walk, policy)` reads the same columns from a `FeatureBatch`.
- `check_pairing(skew_lut, skew_sha256, walk_lut)` from the core, plus the map check. A walk LUT whose model is not IMAS is refused.
- `python -m src.imas.skew export --walk-lut W.json --out DIR` writes:
  - `imas_walk_lut.json` and its manifest, byte copies verified by SHA-256;
  - `imas_walk_curves.tsv`: block, bin indices, bin edges, offset in ps, events;
  - `export.json`, which names the paired skew LUT.

  The native PETsys table gets no walk.

## CTR with walk (FR-5)

- The CTR file gains `walks`, e.g. `{"em2+walk2": {"skew": "em2", "walk_lut": "...", "walk_lut_sha256": "..."}}`.
  - Each entry is a table: skew delays and the walk correction.
  - The pairing is checked with `check_pairing`.
- Same events: when any walk table is evaluated, an event also needs every walk input on both sides. A new stage `walk input missing` counts the rest. The skew-only tables are then on the same events as the walk tables; their CTR may differ slightly from spec 011's 468.1 ps, and the report says so.
- The per-event correction goes through `correct_features`, so the CTR's corrected `dt` is the correction function's.
- **Acceptance:** spec 011's rule with the named pair (`acceptance.compare: ["em", "em2+walk2"]`). The model-free FWHM improvement must exceed the combined uncertainty (paired bootstrap reported); pooled and per position.
- Round-trip summary in the report: the CTR for step 0 skew-only, step 0 skew + walk 1, and step 2 skew + walk 3.

## Diagnostics (FR-7)

`python -m src.imas.skew walk-plot` (core `src/timing/plots.py` style, headless Agg) writes:
- the 2D charge × share table with bin counts;
- the DOI-ratio and keV curves with events per bin;
- residual against each input (median per input bin), before and after, on bar held-out events and on the CTR's 22Na events;
- CTR per position before and after.

## Round trip on real data (spec owner decision)

| Step | Command | Output | Expected time |
| --- | --- | --- | --- |
| 0 | `run job_ship.json --work job_walk --observe-only` | walk-carrying observation sets (same events as `job_ship`) | ~1 h (HDD stream) |
| 1 | `walk walk1.json` (skew `job_ship/lut_em.csv`) | `walk1.json` | minutes |
| 2 | `run job_ship_walk.json --work job_walk` (`walk_lut` walk1, em + pairs) | `lut_em_walk-<sha>.csv`, `lut_pairs_walk-<sha>.csv` | ~1.5 h |
| 3 | `walk walk3.json` (skew from step 2, `walk_lut` walk1 for pairing) | `walk3.json` | minutes |
| 4 | `ctr ctr_walk_4pos.json` | 4-position report: legacy, em, em+walk1, em2+walk3 | ~10 min |

- Step 0 also checks that the per-run accepted observation counts equal `job_ship`'s. The selection and geometry are the same; only the row dtype differs.
- Delivered pair: step-2 skew + step-3 walk (spec). The step-1 walk is fitted with the step-0 skew, so `check_pairing` accepts both pairings.
- The walk fit is a step, not a stage of the job: steps 1 and 3 read the job's observation sets and write separate outputs.

## Data shapes

- **Walk LUT (JSON):** Cornell's layout and keys, with IMAS block inputs (`schema` `imas-walk-lut/1`), units ps, convention string, `edges: clamp`, gauge text, and per block `inputs`, `edges`, `offset_ps`, `events`, `filled_from_nearest`. The manifest holds the table's SHA-256.
- **Observation row with walk:** `OBSERVATION_DTYPE` + 6 float32 columns (~151 bytes per row). The ship job's 48.7 M rows grow from ~6.2 GB to ~7.4 GB on C:.

## FR → component map

| Requirement | Component | Check |
| --- | --- | --- |
| FR-1 | `src/imas/skew/walk.py:walk_inputs` | status per missing input; values from synthetic features |
| FR-2 | `src/timing/walk.py` (model, table, fit); IMAS model | round trip, gauge, sparse fill, clamp counts; Cornell walk tests unchanged |
| FR-3 | `walk` command, `fit_two_branch` | two-branch synthetic recovers injected walk; held-out never fitted; refusals |
| FR-4 | observation dtype, stream, `WalkCorrected`, job `walk_lut` | swap consistency; fingerprint refusal; synthetic job recovers delays with walk; provenance; Cornell cache golden unchanged |
| FR-5 | `ctr.py` `walks` | synthetic 22Na with injected walk improves; pairing refusal; same events |
| FR-6 | `correct_dt`, `correct_features`, `export --walk-lut` | equals CTR corrected `dt`; keep/drop counts; export round trip |
| FR-7 | `walk-plot` | PNGs written headless |
| Real | round trip steps 0–4 | recorded timings, memory, acceptance |

## Risks

- **Walk absorbing skew:** charge and keV distributions differ per channel (gain), so a walk table can take up part of a per-channel offset. The single round trip is the owner's answer, and the report gives the change at each step. A large step-3 change against step 1 is reported, not iterated further (spec).
- **Bar acceptance in walk inputs:** the bar fills the keV window unevenly (Compton in the bar, 408–613 keV only). Sparse keV bins fall back to their nearest populated bin and are listed.
- **Clock satellites (±5 ns) and channel-13 outliers:** outside the core window, so they don't enter the fit; their counts are visible in the held-out report.
- **HDD read time:** step 0 is the only new LDAT pass for the skew runs. It can start once T3/T4 are in, while T5–T8 are coded.
