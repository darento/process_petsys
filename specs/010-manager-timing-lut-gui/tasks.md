# Tasks — Timing LUTs in the PETsys Manager GUI

Status: in progress, 2026-10-20. [Approved spec](spec.md), [plan](plan.md).

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
- [ ] **T6 — Validation and release** (FR-6).
  - `__version__` 1.3.0; CHANGELOG entry; full non-real suite; spec status.
  - **Manual (owner, Cornell PC):** set the LUTs in the LM tab, save, reload; LM on a representative Cornell LDAT with and without tables; check plan line, counts, job record.
  - **Done (2026-10-20):** `__version__` 1.3.0; CHANGELOG 1.3.0 entry.
  - **Verified (Windows):** full non-real suite `python -m pytest -m "not real_data"` -> **1 failed, 1339 passed, 24 skipped, 4 xfailed**. The failure, `tests/test_ldat_processing.py::test_cancel_returns_control_within_2_s_and_keeps_the_previous_dataset` (LDAT Inspector; no changed module), passes alone (`-n 0` -> **1 passed**): parallel-load timing.
  - **Open:**
    - `-m real_data`: not run. Listmode numbers are unchanged; the spec 009 `real_data` listmode test still covers the correction.
    - Linux: `gui` tests and the full suite on the Cornell PC (desktop session) before deploying 1.3.0 there.
    - The owner's interactive review above. The spec ships after it.
