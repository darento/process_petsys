# Tasks — IMAS skew calibration

Status: not started, 2026-10-26. [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `011-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("011-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially. New and changed tests run with `-n 0 --slow-limit 5`.
- `real_data` tests name the skew splits, 22Na and legacy LUT relative to `PETSYS_DATA_DIR` (`IMAS/...`) and skip naming the missing variable or file.
- Record each check's selection, reason, command, platform and the pytest summary line as `Verified:`. Tick a task only on success.
- Tasks after T6 are provisional: the FR-7 decision may change them, through a spec change first.
- No commits, pushes or release builds; the owner runs git.

## Core extraction (Cornell unchanged)

- [ ] **T1 — Scanner-neutral core** (FR-1).
  - Move into `src/timing/`: `dataset.py` (bounded scan), `kernels.py` (side-selection kernels, codes, `FeatureBatch`, `Partition`, `holdout_codes`), `shell.py` (`AnnularSource`, `shell_intervals`, `_shell`, LOR stage on precomputed endpoints), `lut.py`, `ctr.py` core, `plots.py`. `src/cornell/skew/*` re-export them; no behavior change.
  - Split Cornell's `_geometry` into the Cornell endpoint kernel and the core LOR stage.
  - **Done when:**
    - `tests/test_cornell_timing_*.py tests/test_cornell_walk.py tests/test_manager_timing_luts.py tests/test_manager_listmode.py -m "not real_data"` pass with no test edited (summary equal to the pre-change run, recorded first).
    - `tests/test_timing_core.py::test_lor_stage_matches_cornell_geometry`: on synthetic Cornell features, the split stage gives arrays equal (`np.array_equal`) to the pre-split `apply_geometry`, captured as a golden under `tests/data/golden/` with provenance.
- [ ] **T2 — Adapter seam in the cache** (FR-1, FR-6).
  - `src/timing/cache.py` takes an adapter (`select_batch`, `rejection_labels`, `fingerprint_sources`); the Cornell adapter wraps today's lookup.
  - **Done when:**
    - the T1 Cornell selection passes unchanged;
    - `tests/test_timing_core.py::test_cornell_cache_fingerprint_unchanged`: the manifest fingerprint of a synthetic Cornell cache equals the pre-change value (golden);
    - `tests/test_cornell_timing_real.py` with a new `real_data` case `test_run7_job_bytes_unchanged`: a bounded run7 job (first 2 files, prefix) before and after T1–T2 gives equal SHA-256 for feature blocks, observation partitions, pair fits and LUT. The "before" hashes are a baseline under `tests/data/baselines/` recorded from HEAD before T1, with provenance.

## IMAS front end

- [ ] **T3 — IMAS lookup and inventory** (FR-2, FR-6).
  - `src/imas/skew/setup.py` (dense lookup, SM numbering check), `python -m src.imas.skew inventory SPLITS... --out DIR`.
  - **Done when:**
    - `tests/test_imas_skew_setup.py`: synthetic map/config pass; a swapped `ring_yx` entry or a ring-major mismatch raises `GeometryError`; slab index 0–7 per TIME channel.
    - `tests/test_imas_skew_inventory.py`: synthetic LDAT counts records and hits per TIME channel; a channel with no hits is listed unobserved.
    - **Real (operator, Windows):** inventory of the 3 local splits (full files): records, hits per TIME channel and SM, unobserved channels compared with the 796 `9999` channels of `skew_IMAS.txt` and the dark minimodules of `docs/imas/image_regions_down.png`; read rate in MB/s.
- [ ] **T4 — IMAS side selection** (FR-3).
  - `src/imas/skew/select.py`: steps 1–7 of the plan, first-failure codes; reference mode with the legacy hit-energy floor and 15 ns window.
  - **Done when:**
    - `tests/test_imas_skew_select.py`: one synthetic record per rejection reason and per boundary (`min_ch` exact, window `|dt|` = limit, keV at 408 and 613 exclusive, minimodule tie flagged, slab = top TIME channel);
    - every record gets exactly one code (counts sum to records);
    - **Real (operator):** on a 100,000-record prefix of split 08, the reference mode and an untracked harness calling the legacy script's per-event steps agree record by record; the production-vs-reference differences are counted per cause.
- [ ] **T5 — Energy calibration from the dedicated split** (FR-4).
  - `src/imas/skew/encal.py`, `python -m src.imas.skew encal ENCAL.json`.
  - **Done when:**
    - `tests/test_imas_skew_encal.py`: synthetic Gaussian peaks with background recover `mu` within 1 %; < 200 entries → `few entries`; no peak → `no peak`; the written file reads back through `KevConverter(file_type="mu")`; a missing channel converts to NaN, never a number.
    - **Real (operator):** split 08 as the dedicated split: channels fitted / few entries / no peak, per-SM peak spread, and the file SHA-256 recorded here.
- [ ] **T6 — Geometry, source and feature cache** (FR-2, FR-5, FR-6).
  - `src/imas/skew/geometry.py` (flat SM facets, nominal signs `s_t = +1`, `s_a = −1`, 7 mm), `IMAS_SKEW_BAR`; IMAS adapter for the core cache.
  - **Done when:**
    - `tests/test_imas_skew_geometry.py`: known (SM, slab, centroid) → endpoint mm for SMs in rings 0 and 4 and azimuths 0, 5, 23; each sign flips its coordinate; an LOR through the bar crosses the annulus twice with the expected TOF; an LOR missing it is rejected.
    - `tests/test_imas_skew_cache.py`: synthetic IMAS cache resumes after an interrupted file and refuses a changed config, calibration or source.
    - **Real (operator):** feature cache and observation set on the 3 splits with T5's calibration: per-cut counts, accepted observations, cache bytes per LDAT byte, peak memory, time.

## Decision point

- [ ] **T7 — Pair-statistics measurement and estimator decision** (FR-7).
  - `python -m src.imas.skew pairstats OBS_DIR --out REPORT.json`: observations, distinct pairs, observations per pair, branch-resolved fraction at spec 008's thresholds, partners per channel, components, coverage per SM and ring, projection to 1 TB.
  - **Done when:**
    - `tests/test_imas_skew_pairstats.py`: synthetic observation set with known pair counts reproduces the histogram, components and projection.
    - **Real:** the report on the 3 splits recorded here.
    - **Owner decision** written into the spec as a change (estimator, and any revision of T8–T15) before T8 starts.

## Provisional (revised after T7)

- [ ] **T8 — Estimator and solve** (FR-7, FR-8). Per the decision: move `pairs.py`/`solver.py` to the core and run them on IMAS observations, or implement the chosen alternative.
  - **Done when:** `tests/test_imas_skew_solve.py`: injected delays on an IMAS-geometry synthetic set recover exactly without noise (gauge fixed) and within declared uncertainty with noise, two branches and background; components and statuses explicit; no NaN, `9999` or zero enters a correction.
- [ ] **T9 — Sign selection** (FR-2). The four `(s_t, s_a)` combinations on the 3 splits with free branch separation; the evidence and the selected pair recorded here; constants `SELECTED_*` in `src/imas/skew/geometry.py`.
  - **Done when:** `tests/test_imas_skew_geometry.py::test_sign_study_picks_injected_signs` on synthetic data; the real study recorded.
- [ ] **T10 — Outputs** (FR-9).
  - Canonical LUT with `scanner` and map fingerprint; native table from IMAS global IDs (one table); legacy export (`skew = −b`, omitted channels) and import (`9999` = no delay); `correct_timestamps`.
  - **Done when:** `tests/test_timing_lut_outputs.py`: positive injected delays survive each consumer (native parser, `src/utils.py:read_skew` with `t + skew`, `correct_timestamps`), endpoint swaps negate `dt`; importing `skew_IMAS.txt` (`real_data`) gives 14,564 delays and 796 no-delay channels; a LUT from another map is refused by every reader.
- [ ] **T11 — Pooled job** (FR-6, FR-8, FR-12).
  - `python -m src.imas.skew run JOB.json`: runs, splits and roles explicit; per-run 511 keV peaks and pair residuals against the pooled delays; optional per-run solves; disk preflight; stage timings and peak memory.
  - **Done when:** `tests/test_imas_skew_job.py` (synthetic two-run job: refusal of undeclared files, resume, per-run comparison); **real (operator):** the 3-split development job, its report recorded here as a development result.
- [ ] **T12 — Diagnostics** (FR-11). Per-ASIC plots for IMAS IDs; SM/minimodule overview by ring × azimuth; train/held-out residuals; new-vs-legacy delay comparison.
  - **Done when:** `tests/test_imas_skew_plots.py` writes every image headless from a synthetic LUT.
- [ ] **T13 — Point-source CTR** (FR-10).
  - `python -m src.imas.skew ctr CTR.json`: LOR-convergence position, same cuts and calibration rule as the job; uncorrected, legacy and new LUT on the same events; model-free and Gaussian FWHM with uncertainties, tails, counts.
  - **Done when:** `tests/test_imas_skew_ctr.py`: a synthetic point source with injected delays and a legacy file gives the expected ordering and FWHM; **real:** the 22Na splits with the T11 development LUT, recorded here.
- [ ] **T14 — Ship run** (FR-12, FR-10). The job on the owner's ~1 TB HDD runs (read directly from the HDD on this PC), then T13 with the ship LUT.
  - **Done when:** the job report (inventory, coverage, timings, peak memory) and the CTR acceptance (new LUT better than legacy by more than the combined uncertainty) are recorded here.

## Validation

- [ ] **T15 — Validation and closing** (all FRs).
  - **Done when:** `-m "not real_data"` and `-m real_data` pass on Windows (`PETSYS_DATA_DIR`, `PETSYS_CAL_DIR`), `tests/data/` changes only where a task authorized them; each `--fr 011-FR-n` selection count recorded; then the spec `Status` becomes `shipped`.
