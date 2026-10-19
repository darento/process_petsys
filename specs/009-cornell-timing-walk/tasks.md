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

- [ ] **T1 — Walk inputs and listmode parity** (FR-1).
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
- [ ] **T2 — Walk LUT contract** (FR-2).
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
- [ ] **T3 — Point-source walk fit command** (FR-3, FR-8).
  - `python -m src.cornell.skew walk WALK.json`, schema `cornell-walk-fit/1`.
  - The fit window may differ from the skew job config only in `energy_range`.
  - Normal equations are accumulated per batch; 4 core iterations; even/odd split.
  - Outputs: report JSON with held-out CTR before/after, PNGs, LUT and manifest.
  - **Done when:**
    - `tests/test_cornell_walk.py::test_fit_recovers_injected_walk`: synthetic point-source LDAT with known walk, DOI and keV offsets; recovered within the declared uncertainty, gauge kept.
    - `::test_fit_window_only_energy_range_may_differ`.
    - `::test_plots_written`.
    - `tests/test_cornell_timing_cli.py` lists `walk`.
- [ ] **T4 — CTR with walk** (FR-5).
  - The CTR file gains `walk_lut` (+ SHA-256), the listmode files and `previous_reports`.
  - The report gives results with and without walk on the same events, plus the summary of previous reports.
  - Pairing refusals: walk fitted with another skew LUT; a skew-with-walk LUT applied without its walk LUT.
  - **Done when:**
    - `tests/test_cornell_timing_ctr.py::test_walk_lut_improves_injected_walk`: synthetic; with-walk FWHM smaller, without-walk equal to the current result.
    - `::test_walk_pairing_refused`.
    - `::test_previous_reports_summary`.
    - The existing CTR tests pass unchanged.
- [ ] **T5 — Feature cache schema 3 and job walk option** (FR-4).
  - Cache `SCHEMA_VERSION = 3` with the ingredient columns; old caches refused.
  - The job gains `walk_lut` and the listmode files; observations get `dt_ps -= round(W_i − W_j)`.
  - Status rejections counted per run.
  - The LUT manifest records the walk LUT and its SHA-256.
  - **Done when:**
    - `tests/test_cornell_timing_job.py::test_walk_applied_before_pair_fits`: synthetic run with injected delays and walk; delays recovered only with walk applied.
    - `tests/test_cornell_timing_cache.py::test_schema_3_refuses_old_cache`.
    - `tests/test_cornell_timing_job.py::test_lut_manifest_records_walk`.
    - `-m "not real_data" -k cornell_timing` passes, as the affected consumers.
- [ ] **T6 — Real-data round trip and acceptance** (FR-3, FR-4, FR-5; Windows, local data).
  - **Steps:**
    1. Walk fit on X0Y0Z0 with the T10 skew LUT.
    2. CTR on X0Y-1/-6/-10Z0, T10 LUT ± walk.
    3. r7 skew job with walk: cache rebuild, about 15 min.
    4. Walk refit on X0Y0Z0 with the new skew LUT.
    5. Final CTR at 357–665 keV, with `previous_reports` = steps 2 and 5.
  - **Acceptance:** held-out pooled model-free and Gaussian FWHM each improve by more than the combined uncertainty against T9b: model-free 424.0 ± 3.1 ps, Gaussian 489.9 ± 0.6 ps. Results reported per position and pooled.
  - **Recorded:** command, inputs, SHA-256 values and timings.
  - **Done when:** all five steps succeed and the acceptance rule is met. If not, the numbers are recorded and the outcome is reported to the owner.
- [ ] **T7 — Listmode application and Manager profile keys** (FR-6).
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
- [ ] **T8 — Phantom walk fit** (FR-7, last).
  - `walk.fit_phantom` on r7 observations with the skew LUT frozen, using the shell two-peak template per walk bin.
  - **Done when:**
    - `tests/test_cornell_walk.py::test_phantom_fit_recovers_injected_walk`: synthetic shell.
    - Real r7 against the T6 point-source table: populated bins within the declared tolerance (proposal: 10 ps or 3σ), and held-out point-source CTR equal within uncertainty.
    - Otherwise the result is recorded as a limitation in the spec.
- [ ] **T9 — Validation** (all FR).
  - Full non-real suite (`-m "not real_data"`) and `-m real_data` on Windows.
  - Walk every `009-FR-n` with its `--fr` selection.
  - Record limitations, including that other-day stability is pending.
  - Set the spec to `shipped`.

## Required validation runs

Follow [AGENTS.md's testing policy](../../AGENTS.md#environment-and-checks):
- **Isolated walk tasks (T1–T3, T8):** run `tests/test_cornell_walk.py` or `--fr 009-FR-n`.
- **T4/T5:** these change the shared CTR, cache and job code, so add the `cornell_timing` consumer selection.
- **T7:** touches listmode and the Manager, so add the Manager listmode, settings and GUI-shell tests and compile-check the edited Manager modules.
- **T6/T7 real-data checks** use local LDATs on Windows.
- **No Linux-only code is planned.** If T7 changes the Manager runtime, the Cornell full suite runs before a Manager release to Cornell.
