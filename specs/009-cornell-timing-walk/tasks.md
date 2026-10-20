# Tasks — Cornell time-walk correction

Status: planned, 2026-10-19. [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `009-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("009-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially.
- Real-data tests name inputs relative to `PETSYS_DATA_DIR` / `PETSYS_CAL_DIR` and skip with a reason.
- Record each check's selection, reason, command, platform and the pytest summary line as `Verified:`. Tick a task only on success.
- Changed tests run with `-n 0 --slow-limit 5`.
- No commits, pushes or release builds; the owner runs git.

## Ordered tasks

- [x] **T1 — Walk inputs and listmode parity** (FR-1).
  - **Ingredients:** `ingest._select` also emits the raw ingredients per side (`q_top`, `q_time`, `e_raw`, `doi_raw`).
  - **New `src/cornell/skew/walk.py`:**
    - `WalkContext.from_files(config, calibration, cog_limits, doi_limits)`, reusing listmode's region, keV and DOI-limit code.
    - `inputs()`, returning charge, share, DOI (mm) and keV with status codes.
  - **Listmode:** exposes the same ingredients from `process_batch` arrays, with no output change.
  - **Confirm** with the owner the profile's listmode files (calibration, COG and DOI limits) and their local paths.
  - **Done when:**
    - `tests/test_cornell_walk.py::test_inputs_match_listmode_on_shared_events`: synthetic LDAT; same TIME channel and slab; equal inputs.
    - `::test_missing_inputs_are_counted`: no TIME hit, zero sums, missing calibration, region or DOI limit.
    - `::test_ingest_outputs_unchanged`: existing feature columns are byte-identical.
    - `tests/test_cornell_timing_ingest.py -m "not real_data"` passes.
    - `tests/test_cornell_walk.py::test_real_point_source_parity` (`real_data`): first 200k records of X0Y0Z0, shared-event inputs equal, shared fraction reported.
  - **Implemented (2026-10-19):**
    - `ingest._select` emits `WALK_FEATURES` (`q_top`, `q_time`, `e_raw`, `doi_ratio` per side) in a separate `FeatureBatch.walk` array; the spec 008 columns are unchanged.
    - `src/cornell/skew/walk.py`: `WalkContext`, `inputs()`, `feature_inputs()`, status codes. The keV, region and DOI mapping reuse listmode's `convert_energy_numba`, `compute_region_vectorized`, `_limit_array`, `calibration_array` and the new `map_doi_mm`.
    - Listmode: `process_batch(..., capture=None)` records the written events' inputs before the pair swap; the new `time_hit_charges` is the same first-maximum TIME hit as the timestamp.
    - Listmode's DOI is its `calculate_DOI_vectorized` ratio, ENERGY sum / ENERGY maximum (`sum_rows_cols: True`), mapped through the DOI limits. Limits are (left, right) with left ≤ right, and the mapping is (ratio − right) · 20 / (left − right).
  - **Verified (Windows, `process_petsys`):**
    - `python -m pytest tests/test_cornell_walk.py -m "not real_data" -n 0 --slow-limit 5`: the four T1 tests pass. Fixture setup 4.3 s, including numba compilation.
    - `python -m pytest tests/test_cornell_timing_ingest.py tests/test_manager_listmode.py tests/test_cornell_walk.py -m "not real_data"` -> **38 passed**.
    - `-k cornell_timing -m "not real_data"` -> **119 passed, 4 xfailed**.
    - Selection reason: the ingest kernel and listmode `process_batch` changed.
    - Byte-identity is checked per batch, as `_process_file` writes it. `np.concatenate` of `CoincidenceV5` arrays leaves padding bytes uninitialised, so a whole-array comparison is meaningless.
  - **Real-data parity: BLOCKED (owner input needed).**
    - The owner's COG and DOI limits (`cog_limits_full_system.txt`, `doi_limits_full_system.txt`, repo root, 2026-02-12; the older copies in `Desktop/data/Cornell/files_for_listmode/` match too) cover exactly the 3,188 TIME channels of `maps/cornell_map_full_system_old.yaml`.
    - Against the current `maps/cornell_map_full_system.yaml` (the map of the skew job and CTR config), 447 limit channels do not exist and 1,099 TIME channels have no limits, so `load_limits` refuses the files.
    - The old → current change is a pure relabelling: 1,280 TIME channels of SMs 9, 10, 11, 15–20 and 25 keep (SM, MM, position) under a new ID; no local coordinates change.
    - 637 old IDs name a different (SM, MM, position) in the current map. A loader that ignored unknown channels would give them another channel's limits.
    - Copies of the two files were placed in `encal_files/` (`PETSYS_CAL_DIR`).
  - **Resolved (owner, 2026-10-20):** the old-map files are now `*_full_system_old.txt`. The new `cog_limits_full_system.txt` (SHA-256 `a5108e34...`) and `doi_limits_full_system.txt` (`c5e712bb...`), dated 2026-03-04, cover 3,188 TIME channels, all in `maps/cornell_map_full_system.yaml`, with no inconsistent slabs. The `encal_files/` copies were replaced with them.
  - **Real-data parity first run:** share differed on 5 of 14,829 shared events, and DOI on 90 (0.6 %, up to 2.4 mm).
    - **Cause:** negative-charge hits (pedestal). The skew ingest drops them before selection (`_keep`, spec 008 legacy parity); listmode's reader keeps them in its sums.
    - **Fix (spec FR-1 updated):** listmode computes its walk ingredients with `walk_ingredients`, over hits with charge >= 0 in the selected minimodule, and maps them with the same `skew.walk.inputs` through `walk_context(ctx)` (its own arrays). The written DOI and keV are unchanged.
  - **Verified (2026-10-20, Windows):**
    - `PETSYS_DATA_DIR=C:/Users/dsanchez/Desktop/data PETSYS_CAL_DIR=<repo>/encal_files python -m pytest tests/test_cornell_walk.py -n 0 -m "real_data or not real_data"` -> **25 passed**.
    - `real parity: 20,073 records in both, 14,829 shared (same TIME channel and slab), 100.0000% with all inputs`; all four inputs equal (rtol 1e-4), and the timestamped TIME channel is the same.
- [x] **T2 — Walk LUT contract** (FR-2).
  - `walk.WalkLut` writes and reads `cornell-walk-lut/1` and its manifest, and refuses an existing path.
  - Minimum events per bin 200, with nearest-populated fill and a listing.
  - Zero-mean gauge, edge clamping.
  - The compiled `correction()` returns ps and clamp flags.
  - **Done when:**
    - `tests/test_cornell_walk.py::test_lut_roundtrip_and_schema`.
    - `::test_refuses_overwrite`.
    - `::test_sparse_bins_take_nearest_and_are_listed`.
    - `::test_clamp_at_edges_counted`.
    - `::test_correction_matches_python_reference`.
  - **Verified (2026-10-19, Windows):** `python -m pytest tests/test_cornell_walk.py -m "not real_data" -n 0 --slow-limit 5` -> the five T2 tests pass, under 1 s each.
    - Files: table `<name>.json` and manifest `<name>.manifest.json`.
    - Nearest-bin ties take the first populated bin in C order.
    - Bin lookup: `searchsorted(edges, v, "right") - 1`, clipped to the edge bins; NaN inputs give NaN corrections.
- [x] **T3 — Point-source walk fit command** (FR-3, FR-8).
  - `python -m src.cornell.skew walk WALK.json`, schema `cornell-walk-fit/1`.
  - The fit window may differ from the skew job config only in `energy_range`.
  - Normal equations are accumulated per batch; 4 core iterations; even/odd split.
  - Outputs: report JSON with held-out CTR before/after, PNGs, LUT and manifest.
  - **Done when:**
    - `tests/test_cornell_walk.py::test_fit_recovers_injected_walk`: synthetic point-source LDAT with known walk, DOI and keV offsets; recovered within the declared uncertainty, gauge kept.
    - `::test_fit_window_only_energy_range_may_differ`.
    - `::test_plots_written`.
    - `tests/test_cornell_timing_cli.py` lists `walk`.
  - **Implemented (2026-10-19):** `walk.load_fit` reuses `ctr.load_ctr`, which now takes `schema=` and `extra=` keys. `walk.collect` runs the CTR cuts with `CompiledLookup.energy_range` replaced by `fit_window_kev`; the config itself still differs from the skew job's only in `energy_range` (`ctr.frozen_inputs`).
    - **Deviation from the plan:** kept events are stored compactly (8 float32 inputs, float32 residual, parity, acquisition; about 40 B per event) instead of re-streaming 4 times. The point-source sets are about 1M events.
  - **Check names as built:**
    - `test_fit_recovers_injected_walk`: arrays with a known 2D walk, DOI and keV truth. The held-out error has mean < 3 ps and spread < min(0.2 · truth spread, 20 ps); the spread is 15.5 ps for a steep exponential truth over 20 quantile bins, i.e. bin discretisation. The residual spread returns to the injected 180 ps noise.
    - `test_walk_command_end_to_end`: covers the CLI, synthetic LDAT, report, LUT, manifest and PNGs (FR-8). The refit is refused because the outputs exist.
    - `test_fit_file_validation`: covers the fit-file checks. The fit-window rule itself is `ctr.frozen_inputs`, tested by spec 008's `test_analysis_energy_window`.
  - **Verified (Windows):** `python -m pytest tests/test_cornell_walk.py tests/test_cornell_timing_cli.py -m "not real_data" -n 0 --slow-limit 5` -> **14 passed**. The longest call is 3.6 s, after caching the synthetic channel lookups.
- [x] **T4 — CTR with walk** (FR-5).
  - The CTR file gains `walk_lut` (+ SHA-256), the listmode files and `previous_reports`.
  - The report gives results with and without walk on the same events, plus the summary of previous reports.
  - Pairing refusals: walk fitted with another skew LUT; a skew-with-walk LUT applied without its walk LUT.
  - **Done when:**
    - `tests/test_cornell_timing_ctr.py::test_walk_lut_improves_injected_walk`: synthetic; with-walk FWHM smaller, without-walk equal to the current result.
    - `::test_walk_pairing_refused`.
    - `::test_previous_reports_summary`.
    - The existing CTR tests pass unchanged.
  - **Pairing rule as built (`walk.check_pairing`):** spec FR-4 says "a skew LUT solved with walk is refused when applied without that walk LUT, and the other way round". For the round trip (walk 2 refitted with the skew re-solved with walk 1), the rule is implemented as follows:
    - A walk LUT is accepted with the skew LUT it was fitted with, or with a skew LUT solved with it.
    - A skew LUT solved with walk is never applied without walk.
    - The listmode files must have the SHA-256 values the walk LUT was fitted with (`check_files`).
  - **Tests as built:** in `tests/test_cornell_walk.py` with the synthetic walk case: `test_ctr_walk_lut_improves_injected_walk`, `test_walk_pairing_refused`, `test_skew_lut_solved_with_walk_needs_walk`, `test_previous_reports_summary`.
    - The CTR gains the stage `walk input missing`; events without inputs are excluded from both result sets.
  - **Verified (Windows):**
    - `python -m pytest tests/test_cornell_timing_ctr.py -m "not real_data" -n 0` -> **4 passed** (unchanged).
    - `python -m pytest tests/test_cornell_walk.py -m "not real_data" -n 0 --slow-limit 5` -> **16 passed**, longest call 4.3 s.
- [x] **T5 — Feature cache schema 3 and job walk option** (FR-4).
  - Cache `SCHEMA_VERSION = 3` with the ingredient columns; old caches refused.
  - The job gains `walk_lut` and the listmode files; observations get `dt_ps -= round(W_i − W_j)`.
  - Status rejections counted per run.
  - The LUT manifest records the walk LUT and its SHA-256.
  - **Done when:**
    - `tests/test_cornell_timing_job.py::test_walk_applied_before_pair_fits`: synthetic run with injected delays and walk; delays recovered only with walk applied.
    - `tests/test_cornell_timing_cache.py::test_schema_3_refuses_old_cache`.
    - `tests/test_cornell_timing_job.py::test_lut_manifest_records_walk`.
    - `-m "not real_data" -k cornell_timing` passes, as the affected consumers.
  - **Implemented (2026-10-19):**
    - Cache `SCHEMA_VERSION = 3`; the `FEATURE_DTYPE` walk columns are stored as float32. This invalidates the existing r7 caches, as planned.
    - `cache.walk_rows` drops rows without walk inputs and corrects `dt_ps -= rint(W_1 - W_2)` before geometry. The observation fingerprint includes the walk LUT and listmode-file SHA-256 values and the `walk.py` source.
    - The job options `walk_lut` + `listmode` must be given together. Observations go to `observations-walk<sha12>`. The LUT provenance records `walk_lut` and `walk_lut_sha256`, and the report records `runs.<run>.walk` counts.
    - Walk with natively corrected runs is refused.
  - **Tests as built:** in `tests/test_cornell_walk.py`: `test_walk_applied_before_pair_fits`, `test_lut_manifest_records_walk`, `test_schema_3_refuses_old_cache`.
    - The synthetic run has low TIME charge on SM 1 and high on SM 16, with a 200 ps step walk LUT.
    - The pair delay is 600 + 200 ps without walk and 600 ps with it, both within 40 ps. Only component 0 is applied.
    - These are marked `slow`: the shared two-job fixture takes 5.4 s to set up.
  - **Verified (Windows):**
    - `python -m pytest tests -k "cornell_timing or cornell_walk or manager_listmode" -m "not real_data"` -> 162 passed, 4 xfailed, 1 failed. The failure was `test_infra::test_no_sys_path_edits_or_script_imports[test_cornell_walk.py]`, because the walk tests imported a test file; fixed by local helpers.
    - `python -m pytest tests/test_infra.py tests/test_cornell_walk.py -m "not real_data"` -> **122 passed**.
- [x] **T6 — Real-data round trip and acceptance** (FR-3, FR-4, FR-5; Windows, local data).
  - **Steps:**
    1. Walk fit on X0Y0Z0 with the T10 skew LUT.
    2. CTR on X0Y-1/-6/-10Z0, T10 LUT ± walk.
    3. r7 skew job with walk: cache rebuild, about 15 min.
    4. Walk refit on X0Y0Z0 with the new skew LUT.
    5. Final CTR at 357–665 keV, with `previous_reports` = steps 2 and 5.
  - **Acceptance:** held-out pooled model-free and Gaussian FWHM each improve by more than the combined uncertainty against T9b: model-free 424.0 ± 3.1 ps, Gaussian 489.9 ± 0.6 ps. Results reported per position and pooled.
  - **Recorded:** command, inputs, SHA-256 values and timings.
  - **Done when:** all five steps succeed and the acceptance rule is met. If not, the numbers are recorded and the outcome is reported to the owner.
  - **Run (2026-10-20, Windows, `process_petsys`; outputs `C:\Users\dsanchez\Desktop\data\Cornell\full_system\point_source\walk_t6\` and `...\full_system\walk_t6_r7\`):**
    - **Common inputs:**
      - config `tests/data/configs/cornell_september.yaml` (T9b cuts, 357–665 keV);
      - r7 `.encal`, also the listmode calibration (per slab), SHA-256 `3a1b86b2...`;
      - limits `cog_limits_full_system.txt` / `doi_limits_full_system.txt` (2026-03-04);
      - sources by LOR convergence.
    1. `python -u -m src.cornell.skew walk walk_t6/step1_walk.json`:
       - X0Y0Z0, T10 skew LUT `b1b211c1...`, fit window 300–700 keV, 570,763 kept, 14 s;
       - held out (odd records): model-free 434.0 -> 392.9 ps, Gaussian 500.0 -> 447.4 ps;
       - walk LUT `2ece204c...`.
    2. `... ctr walk_t6/step2_ctr.json` on X0Y-1/-6/-10Z0 (1,346,822 events):
       - T10 alone: model-free 424.3, Gaussian 490.7 ps;
       - T10 + walk 1: **395.0 ± 2.5 / 448.5 ± 0.7 ps**.
    3. `... run walk_t6_r7/job_r7_walk1.json`:
       - T10 job settings plus `walk_lut` = walk 1;
       - schema-3 cache rebuilt, 19.1 GB; succeeded; kernel 1.6 s, pair fits 9.0 s, solve 13.2 s;
       - LUT `3575e05a...`: 2,281 calibrated / 850 low statistics (T10: 2,094 / 1,037), 3,131 applied;
       - delays moved by 20.5 ps rms against T10 (gauge-aligned);
       - train walk counts: 269 without inputs, 17,614 charge clamps.
    4. `... walk walk_t6/step4_walk.json`:
       - X0Y0Z0 with the new skew LUT;
       - held out: model-free 442.6 -> 391.4, Gaussian 503.1 -> 444.1 ps;
       - walk LUT `529a5543...`.
    5. `... ctr walk_t6/step5_ctr.json` (new skew + walk 2; `previous_reports` = step 2), 1,346,810 events:

       | Scope | Model-free FWHM (ps) | Gaussian FWHM (ps) | FWTM (ps) |
       |---|---|---|---|
       | **Pooled** | **394.1 ± 2.3** | **444.9 ± 0.6** | 789 |
       | X0Y-1 | 390.9 ± 3.9 | 442.5 ± 0.9 | |
       | X0Y-6 | 392.2 ± 3.0 | 442.7 ± 0.9 | |
       | X0Y-10 | 396.0 ± 2.8 | 449.0 ± 0.8 | |

       The same events without walk give 428.8 / 494.6 ps.
  - **Acceptance: PASS.** Against the spec 008 T9b result (424.0 ± 3.1 / 489.9 ± 0.6 ps), the pooled improvement is −29.9 ± 3.9 ps model-free and −45.0 ± 0.8 ps Gaussian, both well beyond the combined uncertainty. Every position also improves against its own no-walk numbers.
  - **Round-trip effect:** the walk-aware skew re-solve plus walk refit gains 0.9 ps model-free and 3.6 ps Gaussian over T10 + walk 1.
- [x] **T7 — Listmode application and Manager profile keys** (FR-6).
  - **Listmode:** `ListmodeContext` gets optional skew and walk tables; `process_batch` corrects `dt` before the int16 write; provenance gets SHA-256 values and counts; mismatched files are refused.
  - **Manager and CLI:** `settings.py` keys `skew_lut` / `walk_lut`; `workflow.py` request `files`; `src/cornell/cli.py` `files.skew_lut` / `files.walk_lut`.
  - **Decide with the owner:** how events whose channel has no skew delay are handled.
  - **Done when:**
    - `tests/test_manager_listmode.py::test_without_tables_byte_identical`.
    - `::test_tables_match_ctr_corrected_dt`: synthetic shared events.
    - `::test_mismatched_tables_refused`.
    - `tests/test_manager_settings.py::test_lut_keys_optional_and_checked`.
    - Affected Manager/listmode/GUI-shell tests pass; Manager modules compile-check.
    - Real data (`real_data`): listmode with and without tables on a local point-source LDAT; corrected `dt` equals the CTR code's on shared events.
    - PETsys Manager patch bump and CHANGELOG entry at ship.
  - **Implemented (2026-10-19):**
    - `listmode.TimingCorrection.load(config, calibration, cog, doi, skew_lut, walk_lut, policy)`:
      - the skew LUT is read as a dense per-channel delay array (NaN where none is applied);
      - natively corrected LUTs are refused;
      - LUT channels that are not TIME channels of the selected map are refused (a LUT of another map);
      - the walk LUT goes through `check_files` and `check_pairing`.
    - **Correction:** `apply` gives `dt = rint(dt - (b1 + W1) + (b2 + W2))` with the timestamped TIME hit's channel and charge (`time_hit_charges` now also returns the channel).
    - **Counters:** observations `timing_corrected`, `timing_uncorrected_kept`, `walk_clamped_<input>`, and rejection `no_timing_correction` with `drop`. They are added only to corrected jobs; the job record gains `timing_correction` only then.
    - **CLI:** `src/cornell/cli.py` takes `files.skew_lut` / `files.walk_lut` (`file?`) and `options.timing_uncorrected` (`keep` | `drop`, a required key). A walk LUT without a skew LUT is refused.
    - **Manager:**
      - Profile fields `skew_lut_file`, `walk_lut_file`, `lm_timing_uncorrected`, saved only when set, because Manager ≤ 1.1.1 rejects unknown fields.
      - The preflight checks the files only when they are set; a walk LUT without a skew LUT is an issue.
      - The workflow request carries them.
      - No GUI widget: the GUI's `replace(self.session.profile, ...)` keeps them.
  - **Tests as built:**
    - In `tests/test_cornell_walk.py` (full-system synthetic): `test_without_tables_byte_identical` (an all-zero skew LUT gives the same bytes as no tables), `test_tables_match_ctr_corrected_dt`, `test_uncorrected_events_kept_or_dropped`, `test_mismatched_tables_refused`.
    - In `tests/test_manager_settings.py`: `test_lut_keys_optional_and_checked`.
    - The no-table path is the unchanged reference code, covered by the spec 003/007 golden listmode tests.
  - **Verified (Windows):**
    - `python -m pytest tests/test_cornell_walk.py tests/test_manager_settings.py -m "not real_data" -n 0 --slow-limit 5` -> **70 passed**.
    - `python -m pytest tests/test_manager_*.py tests/test_ldat_processing.py tests/test_infra.py -m "not real_data"` -> **612 passed, 23 skipped**.
    - `py_compile` of `settings.py`, `workflow.py`, `cornell/cli.py`, `listmode.py`, the skew modules and both GUIs: OK.
    - Selection reason: listmode, the Manager settings/workflow and the request contract changed.
  - **Open:**
    - Real-data listmode check: blocked by the limits files (T1).
    - Manager patch bump at ship (T9).
    - **Owner decision (2026-10-20):** a calibration of the same layout and region count is accepted and flagged (`calibration_differs_from_fit`, with both SHA-256 values); another layout is refused. `check_files` returns the flag, and the CTR, job and listmode records carry it. Test: `test_same_layout_calibration_accepted_and_flagged`.
  - **Real data (2026-10-20, Windows):** `tests/test_cornell_walk.py::test_real_listmode_corrected_dt` runs on the X0Y0Z0 200k-record prefix with the T10 skew LUT and the T6 step-1 walk LUT.
    - Without tables, the output bytes are unchanged.
    - **Result:** `real listmode: 20,073 records in both, 14,796 shared with applied delays, 100.0000% equal; corrected 20,203, kept uncorrected 46` -> **1 passed**.
    - **Release:** PETsys Manager **1.2.0** (MINOR: new optional profile fields), `CHANGELOG.md` entry, 2026-10-20.
- [x] **T8 — Phantom walk fit** (FR-7, last).
  - `walk.fit_phantom` on r7 observations with the skew LUT frozen, using the shell two-peak template per walk bin.
  - **Done when:**
    - `tests/test_cornell_walk.py::test_phantom_fit_recovers_injected_walk`: synthetic shell.
    - Real r7 against the T6 point-source table: populated bins within the declared tolerance (proposal: 10 ps or 3σ), and held-out point-source CTR equal within uncertainty.
    - Otherwise the result is recorded as a limitation in the spec.
  - **Implemented (2026-10-19):**
    - `walk.run_phantom` runs through `python -m src.cornell.skew walk PHANTOM.json`, schema `cornell-walk-phantom-fit/1`. It names the skew LUT's job file (its schema-3 feature caches), the frozen skew LUT, the listmode files and the output.
    - **Events:** train-role feature rows only, as an evenly strided subsample of at most `max_events` (default 5M); no LDAT is read.
    - **Residual:** taken against the nearer shell-branch centre, `(tof0 + tof1) / 2` or `(tof2 + tof3) / 2`, chosen again after each of `branch_iterations` (3) walk updates.
      - This is not CTR, so the FR-13 rule against nearest-branch selection does not apply. With branches about 1.9 ns apart against about 200 ps width and less than 200 ps of walk, misassignment is negligible.
      - The plan's per-bin two-peak template was not needed.
    - `compare_tables` gives the per-event difference of two walk LUTs on the same events (mean, rms, p95).
    - `quantile_edges` gives a single bin for a constant input (no correction after the gauge) instead of failing.
  - **Verified (Windows):** `tests/test_cornell_walk.py::test_phantom_fit_recovers_injected_walk` (`slow`, 8.4 s).
    - **Set-up:** a synthetic shell run, SM 1 vs SM 16, with per-event uniform charges 6–30 and `W(q) = 250 exp(-(q - 6)/6)`. The skew LUT is solved without walk, so it absorbs the mean.
    - **Result:** the bin offsets match the bin-averaged truth within max(15 ps, 0.15 · truth spread).
    - `python -m pytest tests/test_cornell_walk.py -m "not real_data" -n 0 --slow-limit 5` -> **24 passed**.
  - **Real r7 (2026-10-20):** `python -u -m src.cornell.skew walk walk_t6_r7/phantom/phantom_fit_all.json` reads the step-3 job's schema-3 caches with skew LUT `3575e05a...`.
    - The first attempt used the default `max_events` 5M as a stride over all rows (stride 24). It kept only 89k events, because the job geometry accepts 1.85 % of train rows (2.16M of 116M; the opposition-angle cut).
    - With `max_events` 200M: all 116,424,113 train rows, **2,153,498 accepted events**, 2 min; near-branch fraction 0.521; walk LUT `walk_lut_phantom_all.json`.
  - **Comparison with the T6 step-4 point-source table:**
    - **CTR:** on X0Y-1/-6/-10 with the step-3 skew LUT, the phantom walk gives **393.9 ± 1.7 / 444.9 ± 0.6 ps**, against 394.1 ± 2.3 / 444.9 ± 0.6 ps for the point-source walk. Equal within uncertainty.
    - **Per event, on 1,346,826 held-out point-source events:** W1 − W2 differs by 0.3 ps mean, 19.8 ps rms, 46 ps p95; one side alone differs by 13.7 ps rms, 26 ps p95.
    - This exceeds the plan's proposed 10 ps tolerance. No per-bin 3σ was computed, and the two tables have different quantile edges.
    - **FR-7 accepted (owner, 2026-10-20):** on the CTR criterion, with a declared table tolerance of 20 ps rms per side (measured 13.7 ps). Recorded in the spec.
  - **Note:** `max_events` strides over train rows before geometry, so on phantom data it should exceed the train row count. The default 5M is too low for r7; this is recorded, not changed.
- [x] **T9 — Validation** (all FR).
  - Full non-real suite (`-m "not real_data"`) and `-m real_data` on Windows.
  - Walk every `009-FR-n` with its `--fr` selection.
  - Record limitations, including that other-day stability is pending.
  - Set the spec to `shipped`.
  - **Verified (2026-10-20, Windows, `process_petsys`):**
    - `python -m pytest -m "not real_data"` -> **1,330 passed, 24 skipped, 4 xfailed** in 248 s.
    - `PETSYS_DATA_DIR=C:/Users/dsanchez/Desktop/data PETSYS_CAL_DIR=<repo>/encal_files python -m pytest -m real_data -n 0 -rs` -> **30 passed, 0 skipped** in 1,642 s. This includes `test_real_point_source_parity` and `test_real_listmode_corrected_dt`.
    - After the PETsys Manager 1.2.0 bump: `py_compile exe_programs/petsys_manager_gui.py` OK; `tests/test_manager_gui_shell.py` -> **32 passed**.
  - **FR walk:** `--collect-only -m "real_data or not real_data" --fr 009-FR-n` collects every requirement:

    | Requirement | Tests | Checks |
    |---|---|---|
    | FR-1 | 5 | T1 parity, synthetic and real |
    | FR-2 | 5 | T2 |
    | FR-3 | 3 | T3; real fits in T6 |
    | FR-4 | 5 | T5, pairing; real r7 job in T6 |
    | FR-5 | 3 | T4; real CTR in T6 |
    | FR-6 | 7 | T7, including real listmode |
    | FR-7 | 1 | T8 synthetic; real r7, owner-accepted |
    | FR-8 | 1 | T3 plots |

    All of them pass in the runs above.
  - **Linux:** no Linux-only code. The Manager runtime changed (profile fields, request keys), so the Cornell full suite runs before deploying Manager 1.2.0 to Cornell.
  - **Limitations recorded in the spec.**

## Required validation runs

Follow [AGENTS.md's testing policy](../../AGENTS.md#environment-and-checks):
- **Isolated walk tasks (T1–T3, T8):** run `tests/test_cornell_walk.py` or `--fr 009-FR-n`.
- **T4/T5:** these change the shared CTR, cache and job code, so add the `cornell_timing` consumer selection.
- **T7:** touches listmode and the Manager, so add the Manager listmode, settings and GUI-shell tests and compile-check the edited Manager modules.
- **T6/T7 real-data checks** use local LDATs on Windows.
- **No Linux-only code is planned.** If T7 changes the Manager runtime, the Cornell full suite runs before a Manager release to Cornell.
