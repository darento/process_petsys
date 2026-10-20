# Plan — Timing LUTs in the PETsys Manager GUI

Status: approved plan for [spec 010](spec.md), 2026-10-20; shipped 2026-10-20 (T6); deviations recorded in [tasks](tasks.md). [Ordered tasks](tasks.md).

## Recommendation

The correction itself stays as spec 009 shipped it. The work is at the edges:

- **listmode:** the table checks of `TimingCorrection.load` become one function that also runs without a calibration file (pipeline readiness).
- **CLI:** the listmode summary carries the job record's `timing_correction` block.
- **Manager readiness:** LM and pipeline prerequisites run that check, cached by file stamps, off the Tk thread.
- **GUI:** two profile file rows, two radios, a plan line, result counts. Text comes from pure functions tested without Tk.

## Architecture

### Table check (FR-5)

`src/cornell/listmode.py`:

- `check_timing_tables(mapping, calibration, cog_limits, doi_limits, skew_lut, walk_lut=None)` holds today's checks in `TimingCorrection.load`, unchanged in order and message:
  - natively corrected skew LUT refused;
  - skew LUT channels must be TIME channels of the map;
  - walk LUT COG/DOI limits and calibration layout (`walk.check_files`);
  - pairing (`walk.check_pairing`).
- `calibration` is a `{path, sha256, layout, num_regions}` entry. `TimingCorrection.load` builds it from the loaded calibration, as `WalkContext.build` does today. Pipeline readiness passes `{path: None, sha256: None, layout, num_regions}` from Positions per Slab (1: `per_slab`, 1; N: `position`, N). Without a SHA-256 only the layout and region count can match, so the "calibration differs" flag is not computed there; the LM stage computes it.
- It raises `TimingTableError(InputError)` with `field` = `skew_lut` or `walk_lut` (pairing: `walk_lut`). `TimingCorrection.load` keeps raising `InputError("timing correction: ...")`, so existing tests and messages stay.
- `walk.check_files(walk_lut, files)` takes the files mapping (`context.files` at its two other callers, `ctr.py` and `job.py`), so no calibration array is built for a check.

### Readiness (FR-5)

`src/petsys_manager/workflow.py`:

- `timing_table_issues(settings)` → tuple of `PrerequisiteIssue("skew_lut_file" | "walk_lut_file", message)`; empty when no skew LUT is set or the action is not LM/pipeline.
  - LM: the LM tab's calibration file, loaded with `load_calibration`.
  - Pipeline: the layout from `options.regions`.
  - Loads the processing config, COG and DOI limits as `prepare` does.
- `prepare` (START) calls it too: the check also guards a START that bypasses readiness.
- `session._check_loop` calls it after a clean `preflight` report for LM and pipeline requests and appends its issues. A small cache keyed by the stamps (path, size, mtime) of the YAML, map, calibration, limits and tables, plus the layout, avoids reloading on every keystroke. It runs on the existing `petsys-preflight` worker thread, never on Tk's.

### CLI summary (FR-4)

`src/cornell/cli.py` listmode summary gains `"timing_correction": None if timing is None else timing.record`. The counts stay where they are (`observations`, `rejected`). Additive key; the job record is unchanged.

### GUI (FR-1–FR-4)

`exe_programs/petsys_manager_gui.py`:

- `PROFILE_FIELDS` gains `skew_lut_file` ("Skew LUT (timing):", file) and `walk_lut_file` ("Walk LUT (timing):", file). `profile_texts`, `profile_from_ui`, Browse, last folders and issue labels follow from the table.
- `self.timing_policy = tk.StringVar(root, "keep")`, in `field_vars`, `profile_texts` (`lm_timing_uncorrected`) and `profile_from_ui`, traced like the other fields.
- LM tab main section: the two file rows after LM Destination, then the radio row, then the plan label (`self.timing_plan`). Radios are disabled when the skew field is empty (`_update_timing_controls`, called from `_edited` and `_show_profile`).
- Pure functions, tested without Tk:
  - `timing_plan_text(skew, walk, policy)`: one line; file names only (full paths are in the fields).
  - `timing_result_text(summary)`: the lines appended to the listmode stage report.
- The pipeline plan appends `timing_plan_text(...)`.
- `_stage_report` listmode branch appends `timing_result_text(summary)`.

### Data shapes

| Item | Shape |
|---|---|
| calibration entry | `{"path": str \| None, "sha256": str \| None, "layout": "per_slab" \| "position", "num_regions": int}` |
| summary `timing_correction` | job record block: `skew_lut {path, sha256, applied_channels}`, `walk_lut {path, sha256, fit_kind, calibration_differs_from_fit, fitted_calibration_sha256} \| null`, `uncorrected_events`, `convention` |
| counts | summary `observations.timing_corrected`, `observations.timing_uncorrected_kept`, `observations.walk_clamped_*`, `rejected.no_timing_correction` |

## Alternatives rejected

- **Readiness reads only the walk manifest's layout** for the pipeline: misses the map, limits and pairing refusals that also fail the LM stage after acquisition.
- **Reading the job record from disk for counts:** the CLI result already carries them; a second reader adds a path to validate.
- **Full paths in the plan line:** long and duplicated by the fields above it.

## FR → component map

| FR | Component |
|---|---|
| FR-1 | GUI `PROFILE_FIELDS`, `_lm_tab` |
| FR-2 | GUI `timing_policy`, `_update_timing_controls` |
| FR-3 | GUI `timing_plan_text`, LM plan label, pipeline plan |
| FR-4 | CLI summary, GUI `timing_result_text`, `_stage_report` |
| FR-5 | listmode `check_timing_tables`, walk `check_files`, workflow `timing_table_issues`, session readiness, `prepare` |
| FR-6 | `__version__`, CHANGELOG |

## Test selection

- Shared reader change (`listmode` check split, `walk.check_files` signature): `tests/test_cornell_walk.py`, `tests/test_cornell_timing_ctr.py`, skew job tests, `tests/test_manager_listmode.py`, `tests/test_manager_cli.py`.
- Manager: `tests/test_manager_settings.py`, `tests/test_manager_workflow.py`, `tests/test_manager_gui_shell.py`, `tests/test_manager_gui_processing.py`, new `tests/test_manager_timing_luts.py`.
- GUI compile check of `exe_programs/petsys_manager_gui.py`.
- Validation: full non-real suite; operator review on the Cornell PC.
