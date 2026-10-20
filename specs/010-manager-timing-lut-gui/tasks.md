# Tasks — Timing LUTs in the PETsys Manager GUI

Status: done, 2026-10-20 (spec shipped). [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `010-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("010-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially.
- Record each check's selection, reason, command, platform and the pytest summary line as `Verified:`. Tick a task only on success.
- Changed tests run with `-n 0 --slow-limit 5`.
- No commits, pushes or release builds; the owner runs git.

## Ordered tasks

- [x] **T1 — Table check without a calibration file** (FR-5).
  - `listmode.check_timing_tables` and `TimingTableError(field)`; `TimingCorrection.load` uses it; `walk.check_files(walk_lut, files)` and its callers.
  - **Done when:**
    - `tests/test_manager_timing_luts.py::test_check_tables_fields`: synthetic map; a skew LUT of another map → `skew_lut`; unpaired walk LUT and other DOI limits → `walk_lut`; matching set passes.
    - `::test_check_tables_layout_only`: a calibration entry without SHA-256 accepts the walk fit's layout and refuses another layout/region count.
    - `tests/test_cornell_walk.py` (`-m "not real_data"`), the skew CTR/job tests and `tests/test_manager_listmode.py` pass unchanged.
  - **Implemented (2026-10-20):**
    - `listmode.check_timing_tables` holds the spec 009 checks; `TimingCorrection.load` wraps its `TimingTableError` as `InputError("timing correction: ...")`, so a native/foreign skew LUT message now also carries that prefix.
    - `listmode.calibration_entry`; `walk.check_files(walk_lut, files)` (callers pass `context.files`). A calibration entry without SHA-256 never counts as the fitted one.
    - Test helper `tests/timing_lut_helpers.py` (synthetic skew/walk LUTs for a fixture map).
  - **Verified (Windows):**
    - `python -m pytest tests/test_manager_timing_luts.py -m "not real_data" -n 0 --slow-limit 5` -> **2 passed** (new tests).
    - Shared reader change, consumers: `tests/test_cornell_walk.py tests/test_cornell_timing_*.py tests/test_manager_listmode.py -m "not real_data"` -> **149 passed, 4 xfailed**.
- [x] **T2 — Readiness and START check** (FR-5).
  - `workflow.timing_table_issues(settings)`; called by `prepare` and by the session readiness worker (stamp cache).
  - **Done when:**
    - `tests/test_manager_timing_luts.py::test_readiness_issues_lm_and_pipeline`: LM with a mismatched walk LUT → `walk_lut_file` issue; pipeline with Positions 5 and a per-slab walk LUT → issue; matching tables → no issue; no skew LUT → no check.
    - `::test_prepare_refuses_mismatched_tables`.
    - `tests/test_manager_settings.py tests/test_manager_workflow.py` pass.
  - **Implemented (2026-10-20):** `workflow.timing_table_issues`; `prepare` adds its issues as `"<field>: <message>"` for any graph with an LM stage; `ManagerSession._timing_issues` runs it on the preflight worker after a clean report, cached by (action, positions, file stamps), at most 8 entries. Listmode files that do not load give no timing issue (their own checks report them).
  - **Verified (Windows):**
    - `python -m pytest tests/test_manager_timing_luts.py -m "not real_data" -n 0 --slow-limit 5` -> **4 passed**.
    - `python -m pytest tests/test_manager_settings.py tests/test_manager_workflow.py -m "not real_data"` -> **67 passed**.
- [x] **T3 — CLI summary** (FR-4).
  - **Done when:**
    - `tests/test_manager_timing_luts.py::test_cli_summary_timing_block`: in-process CLI listmode with tables; `summary.timing_correction` equals the job record's block; without tables it is `None`; counts are in `observations`/`rejected`.
    - `tests/test_manager_cli.py` passes.
  - **Implemented (2026-10-20):** the listmode summary gains `timing_correction` (the job record's block or `null`) and `calibration_sha256` (for the FR-4 warning). The job record is unchanged.
  - **Verified (Windows):** `python -m pytest tests/test_manager_timing_luts.py -m "not real_data" -n 0 --slow-limit 5` -> **5 passed**; `tests/test_manager_cli.py`: in the full run (T6).
- [x] **T4 — Plan and result text** (FR-3, FR-4).
  - `timing_plan_text`, `timing_result_text` in the GUI module (pure).
  - **Done when:** `tests/test_manager_timing_luts.py::test_plan_text` (none / skew / skew + walk, keep / drop) and `::test_result_text` (corrected counts, clamps only when non-zero, calibration-differs warning, uncorrected run).
  - **Verified (Windows):** `python -m pytest tests/test_manager_timing_luts.py -m "not real_data" -n 0 --slow-limit 5` -> **7 passed**.
- [x] **T5 — LM tab widgets** (FR-1, FR-2, FR-3, FR-4).
  - File rows, radios, plan label, pipeline plan, stage report.
  - **Done when:**
    - `tests/test_manager_timing_luts.py::TimingWidgets` (`gui`): rows in the LM main section, not in Advanced; save/reload/clear round trip with unset fields absent from the file; radios disabled exactly when the skew field is empty; plan label follows edits; a readiness issue names the field and disables Generate.
    - `tests/test_manager_gui_shell.py tests/test_manager_gui_processing.py` pass; `python -m py_compile exe_programs/petsys_manager_gui.py`.
  - **Deviation:** the widget test is `tests/test_manager_gui_processing.py::ProcessingChecks::test_timing_luts_in_the_lm_tab` (`gui`, `slow`, ~16 s), not a `TimingWidgets` class: it needs that class's real-file LM fixture, and a subclass would rerun all its tests. It also covers FR-5 in the window and FR-4 end to end (real CLI child, request carries the tables, stage report shows the counts).
  - **Implemented (2026-10-20):** `PROFILE_FIELDS` gains the two rows (LM main section, after LM Destination); `timing_policy` radios (`lm_timing_uncorrected`), disabled without a skew LUT; `timing_plan` label and the pipeline plan's last line; `_stage_report` appends `timing_result_text`.
  - **Verified (Windows):**
    - `python -m py_compile exe_programs/petsys_manager_gui.py` -> ok.
    - Changed test files, serial duration check: `python -m pytest tests/test_manager_timing_luts.py tests/test_manager_gui_processing.py -m "not real_data" -n 0 --slow-limit 5` -> **20 passed**.
    - `tests/test_manager_gui_shell.py`: in the full run (T6).
- [x] **T6 — Validation and release** (FR-6).
  - `__version__` 1.3.0; CHANGELOG entry; full non-real suite; spec status.
  - **Manual (owner, Cornell PC):** set the LUTs in the LM tab, save, reload; LM on a representative Cornell LDAT with and without tables; check plan line, counts, job record.
  - **Done (2026-10-20):** `__version__` 1.3.0; CHANGELOG 1.3.0 entry.
  - **Verified (Windows):** full non-real suite `python -m pytest -m "not real_data"` -> **1 failed, 1339 passed, 24 skipped, 4 xfailed**. The failure, `tests/test_ldat_processing.py::test_cancel_returns_control_within_2_s_and_keeps_the_previous_dataset` (LDAT Inspector; no changed module), passes alone (`-n 0` -> **1 passed**): parallel-load timing.
  - **Not run:** `-m real_data`. Listmode numbers are unchanged; the spec 009 `real_data` listmode test still covers the correction.
  - **Verified (Linux, Cornell PC, desktop session, 2026-10-20):** `python -m pytest -m "not real_data"` at `a141124` -> **1364 passed, 4 xfailed** (1368 items, none skipped: `gui` tests ran). A first run failed 138 LDAT tests because the checkout had `configs/imas_1DAQ.yaml` deleted; restored for this run, and fixed separately (bug-ldat-config-outside-tests: the IMAS fixture reads `tests/data/configs/imas_1DAQ.yaml`).
  - **Operator review (owner, Cornell PC, Manager 1.3.0 at `c76bbff`, 2026-10-20): PASS.**
    - **Files:** skew LUT `time_delays_r7_walk1.csv` (`3575e05a`) and walk LUT `walk_lut_phantom_all.json` (`5b7d4728`) with their manifests; COG/DOI limits `a5108e34` / `c5e712bb`; per-slab r7 `.encal` `3a1b86b2`. Processing YAML `configs/cornell_full_system.yaml` (map `maps/cornell_map_full_system.yaml`).
    - **Input:** `NEMA_SR_10142026_X0Y-6Z0_coincCompact.ldat` (`73ea88b6`, uncorrected conversion, not used by any walk fit). The `norm_10082026_r*` LDATs were not used: they are natively corrected.
    - **FR-1/FR-2/FR-3:** rows in the main LM section; radios disabled without a skew LUT; plan line naming both files and the policy; save, close, reopen restored the fields and "Keep".
    - **FR-5:** with `encal_files/20260312_TwoNa22_120s_coincFixed_position_5regions.encal` the only prerequisite issue was "Walk LUT (timing): listmode calibration position with 5 region(s) differs in layout from the walk LUT's (per_slab, 1 region(s))", Generate disabled; with the r7 per-slab `.encal`, prerequisites met.
    - **FR-4, corrected run** (`/mnt/nvme/LMs/NEMA_SR_10142026_X0Y-6Z0_lm-P1_2026-10-20_0443`, 2 min 12 s): 500,449 LM records; 498,636 corrected + 1,813 kept with raw dt (sum = records written); walk clamps charge 2, keV 35; no calibration warning. Job record: skew `3575e05a`, walk `5b7d4728`, `keep`, `calibration_differs_from_fit` false.
    - **Uncorrected run** (`..._0447`, 2 min 05 s): 500,449 LM records, "Timing correction: none (raw dt)", job record without `timing_correction`.
