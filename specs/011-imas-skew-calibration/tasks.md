# Tasks — IMAS skew calibration

Status: in progress, 2026-10-26 (T1-T6 done; T7 decision pending). [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `011-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("011-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially. New and changed tests run with `-n 0 --slow-limit 5`.
- `real_data` tests name the skew splits, 22Na and legacy LUT relative to `PETSYS_DATA_DIR` (`IMAS/...`) and skip naming the missing variable or file.
- Record each check's selection, reason, command, platform and the pytest summary line as `Verified:`. Tick a task only on success.
- Tasks after T6 are provisional: the FR-7 decision may change them, through a spec change first.
- No commits, pushes or release builds; the owner runs git.

## Core extraction (Cornell unchanged)

- [x] **T1 — Scanner-neutral core** (FR-1).
  - Move into `src/timing/`: `dataset.py` (bounded scan), `kernels.py` (side-selection kernels, codes, `FeatureBatch`, `Partition`, `holdout_codes`), `shell.py` (`AnnularSource`, `shell_intervals`, `_shell`, LOR stage on precomputed endpoints), `lut.py`, `ctr.py` core, `plots.py`. `src/cornell/skew/*` re-export them; no behavior change.
  - Split Cornell's `_geometry` into the Cornell endpoint kernel and the core LOR stage.
  - **Done when:**
    - `tests/test_cornell_timing_*.py tests/test_cornell_walk.py tests/test_manager_timing_luts.py tests/test_manager_listmode.py -m "not real_data"` pass with no test edited (summary equal to the pre-change run, recorded first).
    - `tests/test_timing_core.py::test_lor_stage_matches_cornell_geometry`: on synthetic Cornell features, the split stage gives arrays equal (`np.array_equal`) to the pre-split `apply_geometry`, captured as a golden under `tests/data/golden/` with provenance.
  - Deviations:
    - `pairs.py` and `solver.py` moved too (`lut` imports `solver`); whole-module moves (`pairs`, `solver`, `lut`, `plots`) leave `sys.modules` aliases in `src/cornell/skew`, so Cornell imports and monkeypatches hit the same object. Partial moves (`dataset`, `ingest`, `geometry`) keep `LdatStream`-calling code in Cornell and re-import the moved names.
    - `ctr.py` core move deferred to T13 (only consumer is the IMAS CTR).
    - The run7 byte check from T2 is done here, as `tests/test_timing_core.py::test_run7_pipeline_bytes_unchanged` (`real_data`, `slow`): 8 whole run7 files, all train, `workers=4`; baseline `tests/data/baselines/cornell_run7_pipeline_011.json` captured at d56f49b. Two files gave no pair with ≥30 events. LUT compared rounded to 1e-6 ps (BLAS thread order moves delays by ~1e-11 ps between processes).
    - Cache fingerprints hash module source, so they change with the move: existing Cornell caches rebuild once, same content. T2 compares the fingerprint against the post-T1 value.
  - Verified 2026-10-26: selection above plus `tests/test_timing_core.py`, `-m "not real_data"`: 157 passed, 4 xfailed (before: 156 passed, 4 xfailed; +1 golden test). `tests/test_timing_core.py -m "" -n 0 --slow-limit 5` with `PETSYS_DATA_DIR`/`PETSYS_CAL_DIR`: 2 passed (283 s). `py_compile src/timing/*.py src/cornell/skew/*.py` OK.
- [x] **T2 — Adapter seam in the cache** (FR-1, FR-6).
  - `src/timing/cache.py` takes an adapter (`select_batch`, `rejection_labels`, `fingerprint_sources`); the Cornell adapter wraps today's lookup.
  - **Done when:**
    - the T1 Cornell selection passes unchanged;
    - `tests/test_timing_core.py::test_cornell_cache_fingerprint_unchanged`: the manifest fingerprint of a synthetic Cornell cache equals the post-T1 value (golden);
    - `tests/test_timing_core.py::test_run7_pipeline_bytes_unchanged` (added in T1) still passes against the pre-T1 baseline.
  - Deviations:
    - The adapter API is `kind`, `labels`, `code_label`, `fingerprint()` and `feature_batches(path, partition, *, max_records, seed)`, not `select_batch`/`rejection_labels`/`fingerprint_sources`. Cornell owns its LDAT read (`ingest.production_batches`, patched in tests as `tc.ingest.LdatStream`), so the adapter yields whole `FeatureBatch`es. Level 2 takes a geometry adapter (`kind`, `describe`, `sources`, `apply`) and an optional walk adapter (`fingerprint`, `source`, `rows`). Cornell's `CornellFeatures`, `CornellGeometry` and `CornellWalk` are in `src/cornell/skew/cache.py`, which keeps the spec 008 `build_feature_cache`/`build_observations` signatures and re-exports the core names.
    - Golden `tests/data/golden/timing/011_t2_cornell_cache_manifests.json` holds both manifests, captured at 41421af (post-T1 HEAD) by local `scripts/capture_011_t2.py`. The test recomputes `sources_sha256` from the named files, so later legitimate edits to `ingest.py` and the other hashed sources don't fail it. At capture time the literal values matched: T2 edits no hashed file.
  - Verified 2026-10-26 (Windows, this PC): T1 selection plus `tests/test_timing_core.py`, `-m "not real_data"`: 159 passed, 4 xfailed (157 + bug-cornell-inventory-provenance + fingerprint test). `tests/test_timing_core.py tests/test_cornell_timing_cache.py -m "" -n 0 --slow-limit 5` with `PETSYS_DATA_DIR`/`PETSYS_CAL_DIR`: 8 passed (301 s), including run7 with `workers=4` (adapter pickled to spawned workers).

## IMAS front end

- [x] **T3 — IMAS lookup and inventory** (FR-2, FR-6).
  - `src/imas/skew/setup.py` (dense lookup, SM numbering check), `python -m src.imas.skew inventory SPLITS... --out DIR`.
  - **Done when:**
    - `tests/test_imas_skew_setup.py`: synthetic map/config pass; a swapped `ring_yx` entry or a ring-major mismatch raises `GeometryError`; slab index 0–7 per TIME channel.
    - `tests/test_imas_skew_inventory.py`: synthetic LDAT counts records and hits per TIME channel; a channel with no hits is listed unobserved.
    - **Real (operator, Windows):** inventory of the 3 local splits (full files): records, hits per TIME channel and SM, unobserved channels compared with the 796 `9999` channels of `skew_IMAS.txt` and the dark minimodules of `docs/imas/image_regions_down.png`; read rate in MB/s.
  - Verified 2026-10-26 (Windows, this PC): `tests/test_imas_skew_setup.py tests/test_imas_skew_inventory.py -m "" -n 0 --slow-limit 5` with `PETSYS_DATA_DIR`: 10 passed. Real: `python -m src.imas.skew inventory` on splits 08-10 (full, `PETSYS_DATA_DIR/IMAS_work/``inventory_3splits/`): 155,114,757 records, 49.6 GB in 116 s (423-434 MB/s, local SSD; the HDD rate is measured in T11); no unknown channels, empty sides or non-finite energies. 14,644/15,360 TIME channels observed; 716 unobserved, all among the 796 legacy `9999` channels (80 legacy `9999` channels have hits); 82 dark minimodules, no dark SM; 32 low-statistics channels (< 0.1 x median hits).
  - Deviations: the map frame's SM centre is at 51.61 mm (midpoint of 1.7-101.517 mm on both axes, equal for every SM), not the plan's 48 mm (legacy `local_to_global`); `ImasSetup.local_origin` uses the map value. Inventory also writes `time_channel_hits.csv` and compares with `--legacy-skew`.
- [x] **T4 — IMAS side selection** (FR-3).
  - `src/imas/skew/select.py`: steps 1–7 of the plan, first-failure codes; reference mode with the legacy hit-energy floor and 15 ns window.
  - **Done when:**
    - `tests/test_imas_skew_select.py`: one synthetic record per rejection reason and per boundary (`min_ch` exact, window `|dt|` = limit, keV at 408 and 613 exclusive, minimodule tie flagged, slab = top TIME channel);
    - every record gets exactly one code (counts sum to records);
    - **Real (operator):** on a 100,000-record prefix of split 08, the reference mode and an untracked harness calling the legacy script's per-event steps agree record by record; the production-vs-reference differences are counted per cause.
  - Verified 2026-10-26: `tests/test_imas_skew_select.py -n 0 --slow-limit 5`: 6 passed. Real (operator harness `scripts_imas/parity_011_t4.py`, untracked): first 100,000 records of split 08 with the T5 calibration and 408-613 keV for both: reference mode and the legacy per-event steps (`read_binary_file`, `filter_min_ch`, `get_maxEnergy_sm_mM`, `filter_coincidence` 15 ns, `KevConverter`, `filter_total_energy`, `calculate_centroid`) agree on every record: 0 mismatches, maximum feature difference 0.0 (legacy outcomes: 30,026 passed, 47,868 energy, 19,301 side min channels, 1,833 window, 972 minimodule min channels). Production vs reference: 3,233 records move to `coincidence window` (10 vs 15 ns; 739 of them passed every reference cut), 3 move to `side min channels` (hit floor `en_min_ch` 0.2). Top-hit `|dt|` with no window (2M records): 4.3 % >= 10 ns, 2.0 % >= 15 ns, 0.9 % >= 50 ns: a flat tail, not lost coincidences.
  - Deviation: one zero-ENERGY minimodule alone passes cut 3 with 0 keV (legacy single-module path) and fails the keV window; cut 3 needs two zero minimodules. `calibration=True` mode added for T5.
- [x] **T5 — Energy calibration from the dedicated split** (FR-4).
  - `src/imas/skew/encal.py`, `python -m src.imas.skew encal ENCAL.json`.
  - **Done when:**
    - `tests/test_imas_skew_encal.py`: synthetic Gaussian peaks with background recover `mu` within 1 %; < 200 entries → `few entries`; no peak → `no peak`; the written file reads back through `KevConverter(file_type="mu")`; a missing channel converts to NaN, never a number.
    - **Real (operator):** split 08 as the dedicated split: channels fitted / few entries / no peak, per-SM peak spread, and the file SHA-256 recorded here.
  - Verified 2026-10-26: `tests/test_imas_skew_encal.py -n 0 --slow-limit 5`: 5 passed. Real: `python -m src.imas.skew encal SPLIT08 --full-hash` (`PETSYS_DATA_DIR/IMAS_work/``encal_split08/`, 2 min 47 s): split SHA-256 `2dbe23da…02b0`, `encal.tsv` SHA-256 `dee766bf…8b25`; 38,415,456 records pass cuts 1-5. 14,559 channels fitted, 768 `few entries` (749 never selected), 33 `no peak` (mostly SM 43 and 118). Per-SM median peak 23.3-97.7 a.u. (median 55.6); within an SM, 5-95 % 35.8-73.7 a.u. (median over SMs); median sigma/mu 4.9 %.
  - Deviations: command takes the splits as arguments (`encal SPLITS... --out DIR`), not `ENCAL.json`; the job (T11) declares the dedicated splits. One pass into 0.1 a.u. bins per channel (fit values at bin centres) instead of Cornell's two passes.
- [x] **T6 — Geometry, source and feature cache** (FR-2, FR-5, FR-6).
  - `src/imas/skew/geometry.py` (flat SM facets, nominal signs `s_t = +1`, `s_a = −1`, 7 mm), `IMAS_SKEW_BAR`; IMAS adapter for the core cache.
  - **Done when:**
    - `tests/test_imas_skew_geometry.py`: known (SM, slab, centroid) → endpoint mm for SMs in rings 0 and 4 and azimuths 0, 5, 23; each sign flips its coordinate; an LOR through the bar crosses the annulus twice with the expected TOF; an LOR missing it is rejected.
    - `tests/test_imas_skew_cache.py`: synthetic IMAS cache resumes after an interrupted file and refuses a changed config, calibration or source.
    - **Real (operator):** feature cache and observation set on the 3 splits with T5's calibration: per-cut counts, accepted observations, cache bytes per LDAT byte, peak memory, time.

  - Verified 2026-10-26: `tests/test_imas_skew_geometry.py tests/test_imas_skew_cache.py -n 0 --slow-limit 5`: 9 passed; T1 Cornell selection plus every IMAS file, `-m "not real_data"`: 190 passed, 4 xfailed. Real: `python -m src.imas.skew observe` on splits 08-10 with T5's calibration, nominal geometry (s_t +1, s_a -1, 7 mm), all train, 1 worker (`PETSYS_DATA_DIR/IMAS_work/``observe_3splits/`, 3 min 52 s, peak 0.78 GiB): 155,114,757 records; first failures: 45,100,997 passed pre-geometry, 70,071,812 energy window, 30,556,296 side min channels, 7,818,712 window, 1,519,523 minimodule min channels, 47,417 invalid calibration. Geometry: 2,048,060 accepted, 32,853,707 opposition angle (< 165 degrees, after crossing the shell), 10,199,230 no shell crossing. Features 4.92 GB = 0.099 bytes per LDAT byte (plan assumed 0.059: ~99 GB cache per TB); observations 0.28 GB; features 191 s, observations 40 s.
  - Deviations: `python -m src.imas.skew observe` (development stage report) added; `peak_memory` and `tree_bytes` moved to `src/timing/report.py` (Cornell `job.py` imports them; not a fingerprinted source).
## Decision point

- [ ] **T7 — Pair-statistics measurement and estimator decision** (FR-7).
  - `python -m src.imas.skew pairstats OBS_DIR --out REPORT.json`: observations, distinct pairs, observations per pair, branch-resolved fraction at spec 008's thresholds, partners per channel, components, coverage per SM and ring, projection to 1 TB.
  - **Done when:**
    - `tests/test_imas_skew_pairstats.py`: synthetic observation set with known pair counts reproduces the histogram, components and projection.
    - **Real:** the report on the 3 splits recorded here.
    - **Owner decision** written into the spec as a change (estimator, and any revision of T8–T15) before T8 starts.

  - Verified 2026-10-26: `tests/test_imas_skew_pairstats.py -n 0 --slow-limit 5`: 2 passed. Real report (`PETSYS_DATA_DIR/IMAS_work/``pairstats_3splits.json`, 10 s): 2,048,060 observations over 1,352,415 distinct pairs (median 1, p95 3, max 11 per pair; 1,378 pairs >= 6, 0 >= 30); 14,559 channels with observations, median 282 (p5 163) per channel from median 186 partners; one connected component. Spec 008 pair fits (declared kernel 200 ps, no pair >= 30 for estimation): 2 usable pairs of 1,352,415. Grouped: 46,483 minimodule pairs, median 46 observations; 4,271 ASIC pairs, median 487. Rings 0/4 ~700k, rings 1-3 ~900k observations; no SM without observations. Linear projection to 1 TB (x20.1): ~41M observations, median ~5,700 per channel, ~930 per minimodule pair, ~9,800 per ASIC pair; a zero-truncated Poisson estimate puts ~2.3M reachable pairs at ~18 observations each.
  - Owner decision pending.

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
