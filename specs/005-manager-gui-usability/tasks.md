# Tasks 005 — PETsys Manager GUI usability

Spec: [`spec.md`](spec.md). Plan: [`plan.md`](plan.md). Owner approved 2026-10-09. Environment: `conda run -n process_petsys --no-capture-output python -m pytest` from the repo root. Never stage or commit; the owner runs git.

**Every task:**
- New tests carry `@pytest.mark.fr("005-FR-n")`; tests that open Tk carry `gui`.
- Each task records:
  - its own selection;
  - `-m "not real_data" -n 0 --slow-limit 5` over the files it touched, with tests ≥ 5 s marked `slow`;
  - the full run (`-m "not real_data"`) when it changes `src/` or `exe_programs/`.
- Golden files and baselines are never edited.
- A task that changes processing code also records `tests/test_golden_provenance.py` and the golden comparisons passing unchanged.
- A negative check (a deliberately broken scratch copy fails the new tests) is recorded for each pure-logic task.

Order follows the owner's priority: progress and time remaining first (FR-1, FR-2). Banners (FR-7, FR-8), then runs and inputs (FR-3, FR-4, FR-6), then Advanced (FR-5).

## Progress and time remaining

- [x] **T1 — Estimates** (FR-1). New `src/petsys_manager/progress.py` (no Tk import):
  - `RateEstimate`: phase start, `update(now, done, total)` returning fraction and remaining or `None`, with the 5 s / 1 % gate. A new phase restarts it; a `None`/0 total is unknown.
  - `SplitEstimate(splits, split_durations_s)`.

  **Done when:** `python -m pytest tests/test_manager_progress.py -k "rate or split"` passes, covering:
  - below 5 s or below 1 % → `None`; at 10 s and 25 % → 30 s remaining;
  - phase change resets the clock; unknown total → `None` and no fraction;
  - splits = 1 → `None`; 0 closed → `None`; durations [10, 20] of 5 splits → 45 s; 5/5 closed → 0.

  The module imports in a subprocess without `tkinter`.

  Verified 2026-10-09 (Windows), test-first: 9 red → green cycles, each test failing before its code (ImportError; gate ×2; unknown total ×2; phase; no start; split ×3; split clamp; fraction clamp), then the no-Tk guard.
  - `python -m pytest tests/test_manager_progress.py -n 0 --slow-limit 5 -m "not real_data"` → 16 passed; `--fr 005-FR-1` → 16 passed.
  - Full run `-m "not real_data"` → 1018 passed, 23 skipped.
  - Seam: a new phase restarts the clock at the previous event, not at its own first event, which already carries work (plan updated). `update` before `start` starts the clock at that event. Overshooting counters give fraction 1.0 and 0 s; extra converter splits give 0 s.
  - Negatives (scratch copies of `progress.py`), each failing: gate `or` → `and` (4 failed); phase restarts at `now`; no 1-split rule; no split clamp; no fraction clamp; total 0 treated as known; `import tkinter`; no-start branch removed (1 failed each).

- [ ] **T2 — Pool progress slots** (FR-1). `src/cornell/parallel.py` `OrderedPool(progress_slots=n)` keeps a shared `'q'` array (spawn context; a plain array in-process) that workers reach through the wrapping initializer. `run(..., on_tick=None)` calls `on_tick(slots)` from the `POLL_S` loop and once after the last result. Existing callers stay unchanged.

  **Done when:** `python -m pytest tests/test_manager_cli.py -k pool_slots` passes, covering:
  - `workers=1` and `workers=2`: tasks write their index slot, `on_tick` sees intermediate values before the last result, and the final tick sees every slot;
  - results and their order are unchanged;
  - cancellation still raises `PoolCancelled`.

  The full run passes.

- [ ] **T3 — Byte progress in the CLI** (FR-1). Calibration `_Reader`, LM `_chunks` and QC `read_pairs` report the consumed offset per batch, through a hook that writes to the slot or calls progress directly in the serial path.
  - `cli.Events` keeps per-file bytes (a finished file counts whole), `bytes_total` from the input sizes at request start, and `phases`: calibration from `once`, LM and QC `["read"]`.
  - It emits `bytes_read`, `bytes_total`, `phase` and `phases` at most every `PROGRESS_INTERVAL_S`.

  **Done when:** `python -m pytest tests/test_manager_cli.py -k byte_progress` passes for calibrate, listmode and qc on synthetic compact LDATs, serial and `workers=2`, covering:
  - `bytes_read` never decreases within a phase, never exceeds `bytes_total`, and its last value equals `bytes_total`;
  - calibration `phases` is `["read", "fits"]` when the plan reads once and `["read", "pass 2", "fits"]` when it reads twice;
  - a reference-mode file that stops at the event limit still counts whole;
  - the `result.json` outputs are byte-identical to a run with progress disabled.

  `tests/test_golden_provenance.py` and the golden calibration/LM/QC comparisons pass unchanged. The full run passes.

- [ ] **T4 — Workflow forwarding and conversion watch** (FR-1).
  - `workflow.py` forwards `bytes_read`, `bytes_total` and `phases`.
  - `_ConversionWatch` scans the attempt directory every 1 s (bounded, exact prefix). It emits `stage_progress` with `phase: "convert"`, `ldat_bytes`, `rawf_bytes`, `splits`, `split_durations_s` and `converter_exited`, and stops with the converter.

  **Done when:** `python -m pytest tests/test_manager_workflow.py -k "conversion_watch or byte_forward"` passes, covering:
  - a fake converter writing 3 splits at known times on a fake clock gives 2 durations before exit and 3 after;
  - `ldat_bytes` is the sum of the split sizes;
  - a look-alike `<prefix>2_1.ldat` is not counted;
  - the watch thread has ended when the stage returns;
  - the conversion run record and outputs equal those of the existing conversion tests;
  - a calibration stage's events carry the byte fields.

  The full run passes.

- [ ] **T5 — Run panel** (FR-1). A `RunPanel` above the tabs shows the action, `CTkProgressBar` (determinate or indeterminate), counter text, elapsed time and the estimate, refreshed from `_poll` at most every 250 ms.
  - Processing counter: "x.xx / y.yy GB input read (read, phase 1 of 2)". Fits: "fits n / m keys".
  - Conversion: "LDAT x.x GB written, RAW y.y GB; splits k / n closed".
  - The final state stays until the next run. Tab status labels are unchanged.

  **Done when:** `python -m pytest tests/test_manager_gui_shell.py -k run_panel` passes with workflow events fed to `_workflow_event`, covering:
  - an unknown total shows indeterminate with elapsed and no estimate;
  - byte events after 5 s and ≥ 1 % show a determinate bar and an estimate;
  - fits shows keys;
  - 1-split conversion is indeterminate; 3-split conversion gives an estimate after the first closed split;
  - no counter text contains "events", "pairs" or "singles".

  The GUI compile check (`python -m py_compile exe_programs/petsys_manager_gui.py`) and the full run pass.

- [ ] **T6 — Stage overview** (FR-2). `progress.StageOverview(stages)` updated from `stage_started`/`stage_finished`/`workflow_finished`, plus a `RunPanel` stage row for `PIPELINE` and `QC`.

  **Done when:**
  - `python -m pytest tests/test_manager_progress.py -k overview` passes, covering: the success sequence gives all `succeeded` with recorded `elapsed_s`; a failure at stage 2 gives `failed` with the later stages `pending`; STOP during stage 1 gives `stopped` with the later stages `stopped`; the running stage's elapsed time ticks on a fake clock.
  - `python -m pytest tests/test_manager_gui_shell.py -k stage_row` passes: a pipeline shows 4 stage labels with their states, and a single-stage run shows no stage row.

## Banners

- [ ] **T7 — Warning banners** (FR-7). `progress.Banners` and a `BannerArea` above the tabs.
  - The existing bias frame and `acknowledge_bias` move there; the lock is unchanged.
  - Failure banners: a failed workflow, failed initialization, DAQD `FAILED`. A stopped banner: a cancelled workflow.
  - `_start` calls `new_run()`.

  **Done when:**
  - `python -m pytest tests/test_manager_progress.py -k banners` passes, covering: show; dismiss; `new_run` clears failure/stopped and keeps bias; bias can't be dismissed and only `acknowledge_bias` clears it.
  - `python -m pytest tests/test_manager_gui_shell.py -k banner` passes: a failed outcome shows a dismissible red banner on every tab; a new run clears it; a bias banner stays across a new run and live actions stay locked until acknowledged (existing spec 003 bias tests pass unchanged).

- [ ] **T8 — Growth banner** (FR-8). `progress.GrowthBanner` and its `BannerArea` row show run name, attempt elapsed, time remaining (duration from the request or the QC preset), `.rawf` size and MB/s, refreshed once per second.

  **Done when:**
  - `python -m pytest tests/test_manager_progress.py -k growth` passes, covering:
    - hidden before `growth_passed`; green after it;
    - `growing: false` turns it red with the time since the last growth; `growing: true` turns it green again;
    - `attempt_started` (retry) resets it to hidden with a new clock;
    - `aborting`, `attempt_finished`, the acquisition `stage_finished` and `workflow_finished` hide it;
    - time remaining is never negative.
  - `python -m pytest tests/test_manager_gui_acquisition.py -k growth_banner` passes: the acquisition event sequence from the existing fake acquisition shows green → red → green → removed, with the run name and both times in the text, for Acquire, pipeline and live QC.

## Runs and inputs

- [ ] **T9 — Reading runs.tsv** (FR-3). New `src/petsys_manager/recent.py` (no Tk): `read_overview(destination)`, which reads the last 1 MiB and returns rows plus a skipped count, and `list_recent(destinations)` (merge newest first, cap 500, duplicate destinations read once).

  **Done when:** `python -m pytest tests/test_manager_recent.py -k overview` passes, covering:
  - missing file → no rows plus a "no runs recorded" state; empty file and header only → no rows;
  - partial last line skipped; wrong column count skipped and counted; a non-portable `run_folder` (`..`, `a/b`) skipped;
  - a 3 MiB file reads only its tail;
  - merge order across two destinations; cap 500;
  - files written by `workflow.append_overview` parse back to their rows.

- [ ] **T10 — Run records: offers, outputs, main report** (FR-3, FR-4, FR-6). In `recent.py`, `run_offers(root)`, `conversion_outputs(root)` and `main_report(root)` use `artifacts.read_manifest`. Before an offer, each file's existence and recorded `size_bytes` are checked.

  **Done when:** `python -m pytest tests/test_manager_recent.py -k "offers or outputs or main_report"` passes on synthetic `RunStore` runs, covering:
  - a conversion run gives 3 LDAT offers with exactly its recorded outputs in order, and a look-alike `.ldat` in the folder is ignored;
  - a calibration run gives the LM-calibration offer with its `.encal`;
  - a pipeline run gives both kinds; a failed or cancelled stage gives none;
  - a resized output → refused, naming the file; a pre-T29 run folder still reads;
  - a non-run folder → refused;
  - `main_report`: QC → `qc_report`, calibration → `calibration_plot`, LM → first debug plot or none, conversion → none.

- [ ] **T11 — Next-step offers** (FR-4).
  - The conversion result frame's offers ("Calibrate", "Generate LM", "Run QC") replace "Use these outputs as processing inputs".
  - The calibration frame's "Generate LM with this calibration" replaces "Use this .encal...".
  - `apply_offer` fills one target and switches to its tab, never starting a run.
  - The spec 003 GUI tests that clicked the old buttons click the offers; their expected inputs are unchanged.

  **Done when:** `python -m pytest tests/test_manager_gui_conversion.py tests/test_manager_gui_processing.py` passes, including new `-k offer` cases:
  - after a fake successful conversion each offer fills only its tab's list, switches tab, and calls `start_workflow` no time;
  - "Generate LM with this calibration" sets `calibration_file` as an unsaved edit and switches to the LM tab;
  - the offers stay disabled after a failed run;
  - the compact-coincidence confirmation, preflight and validation still gate the start.

  The full run passes.

- [ ] **T12 — Recent Runs tab** (FR-3, FR-4). The tab has a `ttk.Treeview` with the spec columns and buttons Refresh / Open folder / Open report / Use as input. It is filled by a session worker job (`ShellEvent("recent", ...)`) when the tab is opened, on Refresh and after `workflow_done`. `open_path` is an attribute.

  **Done when:** `python -m pytest tests/test_manager_gui_shell.py -k recent_runs` passes, covering:
  - a fake `recent` event fills rows newest first;
  - Open folder / Open report call `open_path` with the run folder / `main_report` path; Open report is disabled without one;
  - Use as input on a conversion row offers the 3 targets and applies one as in T11;
  - the skipped count shows in the status line;
  - no file in the destinations changes (tree hash before and after);
  - reading runs off the Tk thread (the session job is used).

## Last folders and run-folder picking

- [ ] **T13 — Profile `last_folders`** (FR-6). In `settings.py`, `MachineProfile.last_folders` (fixed key set, absolute folders) and `save_last_folders(path, folders)` (re-read from disk, replace only `last_folders`, atomic write); `session.save_last_folders(key, folder)`.

  **Done when:** `python -m pytest tests/test_manager_settings.py -k last_folders` passes, covering:
  - round trip; an unknown key or a relative path → `ProfileError`;
  - a 1.0.0-style profile without the section still loads;
  - saving keeps every other on-disk field equal (parsed) and ignores a different in-memory profile;
  - `session.profile` gets the folders while `profile_from_ui() != session.profile` stays as it was;
  - no profile file → nothing written, folders in memory, one log line;
  - write failure → logged, folders in memory.

- [ ] **T14 — Dialog folders and "Add conversion run..."** (FR-6).
  - Every dialog except the profile file starts in `last_folders[key]` when it exists and saves the chosen folder through `session.save_last_folders`.
  - `InputSelection` gains "Add conversion run...": a directory dialog, then `conversion_outputs`, then `use_outputs`, with refusals logged.

  **Done when:** `python -m pytest tests/test_manager_gui_processing.py -k "last_folder or conversion_run"` passes, covering:
  - a dialog opens in the stored folder; a missing stored folder falls back to today's rule;
  - the choice is saved under its key without marking the profile edited;
  - picking a conversion run folder lists exactly its recorded LDATs in order, with the origin shown;
  - a non-run folder and a run without conversion outputs are refused in the log with the list unchanged.

  The full run passes.

## Advanced settings

- [ ] **T15 — Advanced sections** (FR-5). `progress.changed_fields(values, defaults)` and an `AdvancedSection` widget, collapsed at start, on Setup (DAQ fields and safety limits), Conversion (max hits per side), Calibration (COG limits file, positions per slab, event-limit mode, target T, workers) and LM (COG/DOI limits, pair map, LM debug plots). No section on QC.

  **Done when:**
  - `python -m pytest tests/test_manager_progress.py -k changed_fields` passes.
  - `python -m pytest tests/test_manager_gui_shell.py -k advanced` passes, covering:
    - each listed field sits in its tab's section and the routine fields stay outside;
    - collapsing and expanding leaves every Tk variable and `profile_from_ui()` unchanged;
    - editing a hidden field to a non-default value shows "1 changed from default" and restoring it clears the count;
    - readiness still names a field inside a collapsed section.

  All existing `gui` tests pass. The full run passes.

## Validation

- [ ] **T16 — Windows validation and release files** (all FRs).
  - `__version__ = "1.1.0"`, plus a `CHANGELOG.md` entry noting that a profile with `last_folders` needs 1.1.0 or later.
  - Walk FR-1…FR-8 with `-m "not real_data" --fr 005-FR-n` each.

  **Done when:** on Windows these all pass, with golden files and baselines unchanged (`git status` shows no change under `tests/data/`):
  - `-m "not real_data"`;
  - `-m "not real_data" -n 0 --slow-limit 5`;
  - `-m real_data` (with `PETSYS_DATA_DIR`, `PETSYS_CAL_DIR`);
  - `python -m py_compile exe_programs/petsys_manager_gui.py`.

  Each FR's selection count is recorded.

- [ ] **T17 — Cornell full run and operator check** (all FRs).
  - On a clean checkout with an env matching `process_petsys.yml`, run the full run and `-m real_data` from a desktop-session terminal on the Cornell PC.
  - An operator walks through the Manager:
    - a real conversion with ≥ 3 splits (bar, GB written, estimate after the first split);
    - a calibration (byte bar, phases, fits keys);
    - LM and offline QC (bar and estimate);
    - a pipeline (stage overview);
    - an acquisition (growth banner green; red if growth is interrupted, when safe to test);
    - next-step offers and Recent Runs (open, report, use as input);
    - Advanced sections; dialog folders kept after a restart;
    - the window fits the monitor.

  **Done when:** both runs pass on Linux, with their summary lines recorded. The operator's notes per item go in this task. Then the spec `Status` becomes `shipped`.
