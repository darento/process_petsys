# Plan 005 — PETsys Manager GUI usability

Owner-approved scope: [`spec.md`](spec.md) (approved 2026-10-09). Constitution: [`AGENTS.md`](../../AGENTS.md). Ships as PETsys Manager 1.1.0.

## FR → component map

| Requirement | Pure logic (headless, tracked tests) | Tk / wiring |
| --- | --- | --- |
| FR-1 processing | `src/cornell/parallel.py` shared per-file byte slots; readers report bytes consumed; `src/cornell/cli.py` `Events` aggregates `bytes_read`/`bytes_total`/`phase`/`phases`; `workflow.py` forwards them | `RunPanel` bar, counter, elapsed, estimate |
| FR-1 estimate | `src/petsys_manager/progress.py`: `RateEstimate` (5 s / 1 % gate, per phase), `SplitEstimate` | `RunPanel` |
| FR-1 conversion | `workflow.py` `_ConversionWatch` thread: LDAT bytes, `.rawf` size, closed splits → `stage_progress` | `RunPanel` |
| FR-2 | `progress.py` `StageOverview` from `RunEvent`s | `RunPanel` stage row |
| FR-3 | `src/petsys_manager/recent.py` `read_overview`, `list_recent`, `run_offers`, `main_report` | "Recent Runs" tab (`ttk.Treeview`), session worker job |
| FR-4 | `recent.py` `run_offers(root)` via `artifacts.read_manifest` | Offer buttons in result frames; `apply_offer` fills one tab and switches |
| FR-5 | `progress.py` `changed_fields(values, defaults)` | `AdvancedSection` widget per tab |
| FR-6 | `recent.py` `conversion_outputs(root)`; `settings.py` `last_folders` + `save_last_folders` | "Add conversion run..." button; dialogs start in `last_folders[key]` |
| FR-7 | `progress.py` `Banners` (failure / stopped / bias) | `BannerArea` above the tabs; the existing bias frame moves there |
| FR-8 | `progress.py` `GrowthBanner` state machine from acquisition events | `BannerArea` growth row, updated once per second |

New modules hold no Tk import, so their tests run without a display. Widgets read state from these modules and never compute numbers themselves.

## Layout

```
root window
  BannerArea        (FR-7, FR-8: bias / failure / stopped / growth rows, each shown only when active)
  RunPanel          (FR-1, FR-2: action, bar, counter, elapsed, estimate; stage row for multi-stage runs)
  CTkTabview        (5 existing tabs + "Recent Runs")
  Output Log / DAQD Output   (unchanged)
```

Only one foreground workflow runs at a time (`WorkflowBusy`), so a single `RunPanel` above the tabs serves every tab and stays visible on any of them. Each tab's status label keeps its current text (run directory, outputs, findings). The `RunPanel` keeps the last run's final state until the next run starts.

## Data shapes

**Processing progress event** (CLI stdout, `kind: "progress"`). Existing fields stay; new ones are added:

```
{file_index, files, path, records_read, [records_written], phase, phases, bytes_read, bytes_total, [keys_done, keys_total]}
```

- `bytes_total` is the sum of the input sizes taken at request start.
- `bytes_read` is the sum over files of each file's latest consumed offset. That offset is the position after the last complete record the reader handed on.
- A finished file counts as its whole size. This covers calibration files that stop early at the event limit.
- `phases`: calibration gives `["read", "fits"]` or `["read", "pass 2", "fits"]`, decided by the limit plan before reading starts (`once`). LM and QC give `["read"]`.
- `bytes_*` are absent in the fits phase, which uses keys.

**Conversion progress** (`stage_progress` emitted by the workflow, not the CLI):

```
{phase: "convert", ldat_bytes, rawf_bytes, splits, split_durations_s: [...], converter_exited}
```

**`last_folders`** (profile): `{key: absolute folder}`. Keys come from a fixed set: `raw_input`, `ldat_inputs.calibrate`, `ldat_inputs.listmode`, `ldat_inputs.qc_analyze`, `conversion_run`, and every `PROFILE_FIELDS` name. An unknown key or a relative path raises `ProfileError`, like every other profile field.

**Recent run row:** `(finished: datetime, destination_key, run_root: Path, action, inputs, status, main_output)` parsed from `runs.tsv` (`RUNS_HEADER`).

**Offer:** `(label, target_key, payload)`. Targets: `calibrate` / `listmode` / `qc_analyze` take LDAT descriptors; `lm_calibration` takes an `.encal` path.

## Decisions

- **Byte progress through shared slots (FR-1).** With `workers > 1`, which is the default (`0` = CPU count − 2), calibration, LM and QC call `progress` only from `on_result`, after a file finishes and in input order. With 10 splits on 10+ workers every file finishes at about the same time, so a per-file bar would sit at 0 % and then jump to 100 %.
  - `OrderedPool` gains `progress_slots=n`: a `multiprocessing.Array('q', n, lock=False)` from the spawn context, handed to workers through a wrapping initializer. In-process (`workers == 1`) it is a plain array with the same interface.
  - Each reader writes its file's consumed offset to `slots[index]` once per batch.
  - `OrderedPool.run` gains `on_tick`, called from its existing `POLL_S` (0.2 s) wait loop. The CLI reads the slots there and emits at most every `PROGRESS_INTERVAL_S` (0.5 s).
  - The serial path calls the same hook directly.
  - Only counters change, never data, so results stay byte-identical. The golden and full runs prove it.
  - Rejected: per-file progress only (useless with parallel splits); worker → parent queue messages (more traffic and ordering issues for the same information).
- **Readers report consumed bytes.** Calibration `_Reader` already tracks `pos`. LM `_chunks` and QC `read_pairs` get the same "offset after the last complete record" value. For fixed-format files that is header + records × record size; compact readers track their scan position. No reader changes what it yields.
- **Estimate (FR-1).** `RateEstimate.update(now, done, total)`. With f = done / total and t = time since the phase started, it returns `remaining = t · (1 − f) / f` only when t ≥ 5 s and f ≥ 0.01; otherwise `None`. A new phase restarts the clock. A total of `None`/0 gives the unknown-total state. The GUI uses its own monotonic clock when it receives each event (the delay is at most one poll, 100 ms). Elapsed time counts from the GUI's `workflow_started`, so it shows even before the first event arrives.
- **Conversion watch (FR-1).**
  - A daemon thread starts in `_conversion` before `runner.run` and stops when the converter exits.
  - Every 1 s it scans the fresh attempt directory, which holds only this conversion's files, bounded by `MAX_FOLDER_ENTRIES`. It matches names `<prefix>[_<n>].ldat` (the `SPLIT_NAME` rule, exact prefix) and sums their sizes.
  - Split k counts as closed when split k + 1 appears; the last split closes when the converter exits.
  - It records when each split closed and emits `split_durations_s`.
  - `SplitEstimate`: `(splits − closed) × mean(split_durations_s)` once ≥ 1 split has closed and `splits ≥ 2`, otherwise `None`.
  - The bar shows splits closed / splits; the counter shows "LDAT x.x GB written, RAW y.y GB". With 1 split the bar is indeterminate.
  - The watch only reads sizes and directory entries and never opens LDAT files, so it can't race the converter's writes or the stage's validation.
- **Stage overview (FR-2).** `StageOverview(stages)` is updated by `stage_started` / `stage_finished` (status, `elapsed_s`) and `workflow_finished`.
  - Stages after a failure or STOP are `stopped` if the workflow was cancelled; otherwise they stay `pending` and are shown as "not run".
  - The running stage's elapsed time ticks on the GUI clock and is replaced by the recorded `elapsed_s` when the stage finishes.
  - Shown only for multi-stage actions (`PIPELINE`, `QC`).
- **Recent runs (FR-3).**
  - Destinations are `data_dir`, `calibration_dir`, `lm_dir` and `report_dir` from the saved profile. Relative paths resolve against the processing root, as preflight does. Duplicate destinations are read once.
  - `read_overview` reads at most the last 1 MiB of each `runs.tsv`. It skips the header, a partial last line (no trailing newline: being appended) and lines whose column count isn't 6 or whose run folder isn't a portable name; skipped lines are counted and shown in the tab's status line.
  - A missing file gives "no runs recorded" for that destination.
  - Rows are merged newest first and capped at 500.
  - Reading runs as a session worker job (a new `ShellEvent("recent", ...)`), triggered when the tab is opened, by Refresh and after each `workflow_done`. The view never writes.
- **Main report (FR-3).** `main_report(root)` reads the run record. QC returns the `qc_report` artifact; calibration returns `calibration_plot`; LM returns its first `listmode_debug_plot` if one exists; anything else returns none, and the button is disabled. Opening uses `os.startfile` on Windows and a detached `xdg-open` on Linux. Like the dialogs, it is an attribute (`open_path`) so tests can replace it.
- **Run records (FR-4, FR-6).** "Run record" means the latest revision read by `artifacts.read_manifest`. That is `run.json`'s authoritative copy, and it also reads pre-T29 runs.
  - `run_offers(root)` takes only stages with status `succeeded`.
  - A succeeded `conversion` stage gives its `ldat` artifacts with `validated` structure-checked descriptors (compact coincidence) as three offers.
  - A succeeded `calibration` stage gives its `encal` as "Generate LM with this calibration".
  - Before an offer is applied, each file must exist with the recorded `size_bytes`; otherwise the offer is refused with the file named.
  - Files are never found by name pattern or folder listing, so look-alike files beside them are ignored.
- **Offers (FR-4).** After a successful manual conversion, the conversion result frame shows "Calibrate", "Generate LM" and "Run QC". These replace "Use these outputs as processing inputs". After a successful calibration, the calibration frame shows "Generate LM with this calibration", which replaces "Use this .encal...".
  - Each offer reads `run_offers(outcome.run_root)` when clicked, fills only its target and switches to that tab. Nothing starts.
  - LDAT targets use `InputSelection.use_outputs` (converter declaration and origin, as in spec 003).
  - The calibration target sets `calibration_file` as an unsaved profile edit, as `use_calibration` does today.
  - Recent runs' "Use as input" shows the same offers for the selected run (a pipeline run can offer both kinds). Preflight, structure checks and validation still run when the operator starts the stage.
- **Advanced sections (FR-5).** `AdvancedSection(parent, title, fields)` is a header button ("▸ Advanced", or "▸ Advanced: 2 changed from default") over a frame shown with `grid`/`grid_remove`. Hiding a widget keeps its Tk variable, so no value changes.
  - Defaults come from `MachineProfile()`, `AcquisitionSafety()`, `ProcessingLimits()` and the GUI's own initial values (positions 5, event-limit mode `target`, hit limit 16, LM debug on).
  - The changed count is recomputed on every variable trace (`changed_fields`).
  - The field list is the spec's. On Setup the Advanced section groups the DAQ fields and the Acquisition Safety Limits frame.
  - Readiness messages already name the field, so a problem inside a collapsed section is still reported.
- **Last folders (FR-6).** A dialog's start folder is `last_folders[key]` if that folder exists; otherwise today's rule applies (the current value's folder, or home). The profile-file dialog is excluded: the folder would be stored inside the file being chosen.
  - After a dialog returns, `session.save_last_folders(key, folder)` calls `settings.save_last_folders(path, folders)`. It re-reads the profile file from disk, replaces only `last_folders`, and writes atomically with the same temp-file + `os.replace` path as `save_profile`. Other fields come from disk, never from the UI, so unsaved edits are neither saved nor lost.
  - It also updates `session.profile.last_folders`. `profile_from_ui` builds on `session.profile`, so the "unsaved edits" state doesn't change.
  - With no profile file (defaults), folders are kept in memory only, with one log line, because the Manager never creates a profile unasked.
  - A write failure logs once and keeps the folders in memory.
- **Banners (FR-7).** `Banners` keeps an ordered set of `{kind, text, dismissible}`.
  - `failure`: a workflow ends `failed`, initialization fails, or DAQD goes `FAILED`.
  - `stopped`: a workflow ends `cancelled`.
  - `bias`: `bias_unknown`; it is not dismissible and clears only on `acknowledge_bias`.
  - `new_run()` drops `failure`/`stopped`.
  - No processing warnings go here; they stay in the result text and the log.
- **Growth banner (FR-8).** `GrowthBanner` is driven by acquisition events of the current run:
  - `attempt_started` resets it to hidden and starts the attempt clock;
  - `growth_passed` turns it green;
  - `rawf_progress` with `growing: false` after it passed turns it red and records the last time it grew;
  - `growing: true` turns it green again;
  - `attempt_finished`, `aborting`, the acquisition `stage_finished` or `workflow_finished` hide it.

  Its text has the run name (`run_root` name from `workflow_started`), the attempt's elapsed time, time remaining (`duration_s` − elapsed, at least 0; `duration_s` from the request, or the QC preset), the `.rawf` size and MB/s. A retry starts its own clock. The banner text is refreshed at most once per second from `_poll`.
- **Thread rules.** The conversion watch, CLI slots and Recent-runs reading run off the Tk thread and only enqueue events. Two short reads stay on the Tk thread, both bounded: offers/`main_report` (`read_manifest`, which has `manifest_limit_bytes`) and `save_last_folders` (one small YAML file).
- **Version.** `__version__` becomes `1.1.0` at Validation, with a `CHANGELOG.md` entry. 1.1.0 reads every 1.0.0 profile. The entry notes that a profile with `last_folders` needs 1.1.0 or later.

## Tests (all `@pytest.mark.fr("005-FR-n")`, synthetic fixtures)

| File | Covers |
| --- | --- |
| `tests/test_manager_progress.py` (new) | `RateEstimate` gate, arithmetic, phase reset, unknown total; `SplitEstimate` with 0/1/several closed splits and with 1 split; `StageOverview` from event sequences (success, failure, STOP); `Banners` show / dismiss / `new_run` / bias lock; `GrowthBanner` green → red → green → hidden, retry reset; `changed_fields` |
| `tests/test_manager_recent.py` (new) | `read_overview`: missing, empty, header only, partial last line, wrong column count, non-portable folder, 1 MiB tail; merge order and cap; `run_offers` / `conversion_outputs` from synthetic run stores, look-alike `.ldat` ignored, size change refused, failed stage gives no offer; `main_report` per action |
| `tests/test_manager_cli.py` | `bytes_read`/`bytes_total`/`phases` on synthetic LDATs, serial and `workers=2`; `bytes_read` reaches `bytes_total`; early-stop file counted whole; outputs unchanged |
| `tests/test_manager_workflow.py` | Conversion watch on a fake converter writing 3 splits: `split_durations_s`, `ldat_bytes`, final event after exit; the stage's outputs and run record match the existing expectations |
| `tests/test_manager_settings.py` | `last_folders` round trip; bad key / relative path rejected; `save_last_folders` keeps other on-disk fields byte-equal in content and ignores UI edits; no file → memory only |
| `tests/test_manager_gui_*.py` (`gui`) | `RunPanel` determinate/indeterminate; offers fill one tab, switch, don't start; Advanced collapse keeps values and shows the changed count; banner widgets; growth banner rows; Recent Runs tab populated from a fake session event; dialog start folder from `last_folders` |

Before committing any `src/` or `exe_programs/` change, the full run. Validation runs the full run and `-m real_data` on Windows and the Cornell PC with golden files and baselines unchanged.

## Alternatives rejected

- Per-tab progress bars: only one workflow runs at a time, and a single panel is visible from every tab.
- Estimating conversion from a fixed output/RAW ratio (owner chose split timing).
- Reading `/proc/<pid>/fdinfo` for the converter's RAW offset (owner chose output growth).
- Scanning destinations for run folders without a `runs.tsv` line (out of scope).
- Saving last folders with the operator's Save (owner chose an immediate, separate write).

## Risks

- **Window height.** The banner area and `RunPanel` add about 80–120 px to a 900×950 window on a 1080-pixel display. The panel stays at three lines, banners appear only when active, and the tabs are already scrollable. The Cornell operator check confirms the fit.
- **Shared-memory slots under spawn on Linux and Windows.** Covered by the `workers=2` CLI test on both platforms (the Cornell full run).
- **Converter split timing.** The converter may write splits out of order or buffer them, which would make "split k + 1 appeared" a late signal. The estimate only depends on durations, and the operator check on a real split conversion confirms it.
- **Spec 003 GUI tests that click "Use these outputs" / "Use this .encal".** They move to the offer buttons without changing their expected inputs.
