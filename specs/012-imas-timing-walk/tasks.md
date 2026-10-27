# Tasks — IMAS time-walk correction

Status: in progress, 2026-10-27 (T1 done). [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `012-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("012-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially. Run new and changed tests with `-n 0 --slow-limit 5`.
- `real_data` tests name acquisitions relative to `PETSYS_DATA_DIR` and skip naming the missing variable or file.
- Record each check's selection, reason, command, platform and pytest summary line as `Verified:`. Tick a task only on success.
- No commits, pushes or release builds; the owner runs git.

## Core

- [x] **T1 — Scanner-neutral walk table and fit** (FR-2, FR-3).
  - Move `WalkLut`, `quantile_edges`, `design`, `fit_tables`, `walk_difference`, `check_pairing` and `compare_tables` from `src/cornell/skew/walk.py` into `src/timing/walk.py`, parameterized by a frozen `WalkModel(schema, convention, blocks, default_bins)`.
  - Add `fit_two_branch`, the nearer-branch loop of `run_phantom`, and have `run_phantom` call it.
  - Cornell keeps its names bound to the Cornell model.
  - **Done when:**
    - `tests/test_cornell_walk.py tests/test_manager_listmode.py tests/test_manager_timing_luts.py tests/test_cornell_timing_job.py -m "not real_data"` pass with no test edited, with the same summary as before the change (recorded first).
    - `tests/test_timing_walk.py`:
      - a non-Cornell model's LUT round-trips;
      - a reader with the other model refuses the file;
      - Cornell-model table bytes equal the bytes written by the pre-move code on a fixed fit (golden under `tests/data/golden/timing/` with provenance);
      - `fit_two_branch` on a synthetic two-branch fixture (branches ±1.07 ns, σ 270 ps, 5 % flat background) recovers injected walk2d/DOI/keV curves within 3 standard errors after the gauge;
      - flips between rounds are counted.
  - Deviations:
    - `WalkLut` subclasses carry the model (`model = ...`); `design`, `fit_tables` and `fit_two_branch` take `table=` (the subclass). Cornell keeps `WalkLut`, `design`, `fit_tables`, `_empty` as thin bindings and re-exports `MIN_EVENTS`, `WalkError`, `manifest_path`, `quantile_edges`, `walk_difference`, `check_pairing`, `compare_tables` from the core.
    - `fit_two_branch` also returns the per-round branch flips (outside Cornell's manifest, so its history is unchanged).
    - Goldens `tests/data/golden/timing/012_t1_cornell_walk_{fixed,fit}.json`, captured before the move by local `scripts/capture_012_t1.py` from fixtures in `tests/timing_walk_helpers.py`. Fixed-array table compared byte for byte; the fit compared with edges/events exact and offsets within 1e-6 ps (BLAS order). `test_golden_provenance.py` accepts `012-FR-2` regression records like `011-FR-1`'s.
    - The 3-standard-error criterion became: error equal to an oracle fit given each event's true branch (within 2 %) and below 0.22 x the injected walk spread. On the fixture both fits give 19.7 ps exactly, so the branch assignment loses nothing; the residual is binning and statistics.
  - Verified 2026-10-27 (Windows, this PC). Selection: the Cornell walk consumers (moved code) plus the new tests.
    - Before the change: `tests/test_cornell_walk.py tests/test_manager_listmode.py tests/test_manager_timing_luts.py tests/test_cornell_timing_job.py -m "not real_data"`: 64 passed. After: 64 passed, no test edited.
    - `tests/test_timing_walk.py tests/test_golden_provenance.py -n 0 --slow-limit 5`: 46 passed.
    - `py_compile src/timing/walk.py src/cornell/skew/walk.py` OK.
- [ ] **T2 — IMAS walk inputs and model** (FR-1, FR-2).
  - `src/imas/skew/walk.py`: `IMAS_MODEL` (`imas-walk-lut/1`; inputs charge, share, doi_ratio, kev; 20 × 10, 20, 20 bins) and `walk_inputs(q_top, q_time, doi_ratio, kev)`.
  - **Done when** `tests/test_imas_walk.py`:
    - each missing input gives its status (first missing wins) and NaN, never 0;
    - values from `select_batch` on a synthetic IMAS LDAT match hand-computed charge, share, ratio and keV.

## Observations and job

- [ ] **T3 — Observations carry walk inputs** (FR-4).
  - `OBSERVATION_WALK_DTYPE`, `canonical_observations(..., walk=True)`, `stream_observations(..., carry_walk=True)`; IMAS `stream_observations` and the job always carry walk.
  - **Done when:**
    - `tests/test_imas_walk.py`:
      - swapped rows swap the walk columns with `dt`;
      - a stream directory created without walk is refused when reopened with walk, and the other way round;
      - IMAS stream rows hold the features' walk ingredients.
    - Cornell: `tests/test_timing_core.py tests/test_cornell_timing_cache.py tests/test_imas_skew_cache.py tests/test_imas_skew_job.py -m "not real_data"` pass unchanged. This includes the Cornell cache fingerprint golden; the IMAS stream fingerprint tests are updated only where the fingerprint now records the walk columns.
- [ ] **T4 — Walk-corrected view and job `walk_lut`** (FR-4).
  - `WalkCorrected`; job key `walk_lut` (outside the fingerprint); outputs `lut_<est>_walk-<sha12>.csv` and `job_report_walk-<sha12>.json`, never overwritten; provenance `walk_lut_sha256`; `run --observe-only`.
  - **Done when** `tests/test_imas_walk.py`:
    - a synthetic job with injected delays and injected walk gives EM delays whose error RMS with the true walk LUT is below half the error RMS without it;
    - rows with a missing input are dropped and counted per role;
    - the LUT provenance names the walk LUT;
    - a second run with the same walk LUT is refused (output exists);
    - `--observe-only` writes the observation sets and no LUT;
    - a set without walk inputs is refused.

## Fit, correction, CTR

- [ ] **T5 — Bar walk fit command** (FR-3).
  - `python -m src.imas.skew walk WALK.json` (`imas-walk-fit/1`): frozen inputs, stride to `max_events`, `fit_two_branch`, held-out before/after, LUT + manifest + report.
  - **Done when** `tests/test_imas_walk.py`:
    - the synthetic job's observation set with injected walk: the fitted LUT's per-event correction matches the injected one (RMS below 20 % of the injected RMS);
    - the held-out core RMS drops;
    - held-out rows never enter the fit (a change in held-out `dt` leaves the LUT bytes unchanged);
    - refused: a skew LUT from another job, an output that exists, unknown keys.
- [ ] **T6 — Correction function and export** (FR-6).
  - `correct_dt`, `correct_features`, pairing and map checks, `export --walk-lut`.
  - **Done when** `tests/test_imas_walk.py`:
    - the corrected `dt` equals skew + walk computed by hand;
    - `keep` / `drop` counts for no applied delay and missing input;
    - a skew LUT solved with walk is refused without its walk LUT, and a walk LUT fitted with another skew is refused;
    - the export copies verify by SHA-256 and the TSV reads back to the table's offsets.
- [ ] **T7 — CTR with walk** (FR-5).
  - CTR file `walks`; same events with walk inputs; acceptance pair; round-trip summary.
  - **Done when** `tests/test_imas_skew_ctr.py`:
    - a synthetic 22Na point source with injected walk: skew + walk beats skew-only by more than the combined uncertainty;
    - the CTR's corrected `dt` equals `correct_features`;
    - a pairing mismatch is refused;
    - existing spec 011 CTR tests pass unchanged.
- [ ] **T8 — Diagnostics** (FR-7).
  - `python -m src.imas.skew walk-plot`.
  - **Done when** `tests/test_imas_walk.py`: the PNGs are written headless from a synthetic LUT and report.

## Real data and validation

- [ ] **T9 — Round trip on the bar runs and 22Na** (FR-3, FR-4, FR-5; Windows, this PC, HDD attached).
  - Steps 0–4 of the plan; record timings, peak memory, counts and the change at each step.
  - **Done when:**
    - step 0 per-run accepted counts equal `job_ship`'s;
    - steps 1–3 complete;
    - FR-5 acceptance passes pooled over the four positions, reported per position.
- [ ] **T10 — Validation** (all).
  - Full non-real suite; `-m real_data` (results touched); spec Status `shipped`.
  - **Done when** both suites pass and the spec, plan and tasks record the delivered pair.
