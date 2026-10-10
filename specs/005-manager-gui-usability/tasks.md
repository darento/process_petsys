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

- [x] **T2 — Pool progress slots** (FR-1). `src/cornell/parallel.py` `OrderedPool(progress_slots=n)` keeps a shared `'q'` array (spawn context; a plain array in-process) that workers reach through the wrapping initializer. `run(..., on_tick=None)` calls `on_tick(slots)` from the `POLL_S` loop and once after the last result. Existing callers stay unchanged.

  **Done when:** `python -m pytest tests/test_manager_cli.py -k pool_slots` passes, covering:
  - `workers=1` and `workers=2`: tasks write their index slot, `on_tick` sees intermediate values before the last result, and the final tick sees every slot;
  - results and their order are unchanged;
  - cancellation still raises `PoolCancelled`.

  The full run passes.

  Verified 2026-10-09 (Windows), test-first. Seam: `progress_slots`, `run(..., on_tick)`, module-level `report_progress(index, value)`.
  - Red → green: `test_pool_slots_show_progress_before_results` (`workers` 1 and 2), failing first with `TypeError: unexpected keyword 'progress_slots'`. It makes no timing assumptions: each task waits until the parent has seen its 50 (marker file, 30 s deadline).
  - Guards, green at once and proven by broken copies: `test_pool_slots_absent_reports_are_ignored` (also after an in-process pool with slots, which needs `_SLOTS` restored on exit) and `test_pool_slots_cancellation_still_raises`.
  - `python -m pytest tests/test_manager_cli.py tests/test_manager_calibration.py -m "not real_data" -n 0 --slow-limit 5` → 42 passed, 1 deselected. Full run `-m "not real_data"` → 1024 passed, 23 skipped.
  - Negatives (scratch copies of `parallel.py`), each failing: no tick in the pool loop (timeout); no final tick, which first survived, so an order assertion was added ("tick after the last result"); in-process slots not ticking; `_SLOTS` not restored; workers not given the slots; in-process loop ignoring cancellation.

- [x] **T3 — Byte progress in the CLI** (FR-1). Calibration `_Reader`, LM `_chunks` and QC `read_pairs` report the consumed offset per batch, through a hook that writes to the slot or calls progress directly in the serial path.
  - `cli.Events` keeps per-file bytes (a finished file counts whole), `bytes_total` from the input sizes at request start, and `phases`: calibration from `once`, LM and QC `["read"]`.
  - It emits `bytes_read`, `bytes_total`, `phase` and `phases` at most every `PROGRESS_INTERVAL_S`.

  **Done when:** `python -m pytest tests/test_manager_cli.py -k byte_progress` passes for calibrate, listmode and qc on synthetic compact LDATs, serial and `workers=2`, covering:
  - `bytes_read` never decreases within a phase, never exceeds `bytes_total`, and its last value equals `bytes_total`;
  - calibration `phases` is `["read", "fits"]` when the plan reads once and `["read", "pass 2", "fits"]` when it reads twice;
  - a reference-mode file that stops at the event limit still counts whole;
  - the `result.json` outputs are byte-identical to a run with progress disabled.

  `tests/test_golden_provenance.py` and the golden calibration/LM/QC comparisons pass unchanged. The full run passes.

  Verified 2026-10-10 (Windows), test-first. Seam: the `cli.main` event stream (`ByteProgressChecks`, in-process, `PROGRESS_INTERVAL_S` patched to 0).
  - Readers: calibration `_Reader.offset`; LM `_chunks(..., position)` (compact through `compact_chunks`, fixed through `_fixed_chunks`); QC `read_pairs(..., position)`. Each file's last report carries `finished=True` (counted whole). In workers, `parallel.slot_progress(index)` writes the slot and `tick_progress` turns changed slots into reports; worker ticks have `records_read: null`. A reused LM segment counts as read. Throttling is now global, but a file's end, a new phase and the last fit are always emitted.
  - Red → green, one slice each: calibrate serial (two passes), calibrate `workers=2` decoding once, LM serial (compact and fixed), LM `workers=2`, QC serial, QC `workers=2`. The early stop at `event_limit` 1,000 of about 2,550 events per file was green at once (a guard, proven by a broken copy).
  - Serial calibration and QC reports are checked against record boundaries parsed from the file. LM regroups reader batches, so its check only requires a value inside a file.
  - Byte-identical outputs: the existing CLI-vs-in-process tests (`progress=None`) and the golden comparisons pass unchanged.
  - Callers adapted: QC parallel test counts `finished` reports; LM test lambdas accept keywords; `assertSucceeded` accepts `records_read: null` only on byte ticks.
  - `python -m pytest tests/test_manager_cli.py tests/test_manager_qc.py tests/test_manager_listmode.py tests/test_manager_calibration.py -m "not real_data" -n 0 --slow-limit 5` → 86 passed, 1 failed. The failure is `test_cli_calibrate_child_process_literal_paths_match_in_process` over 5 s under load; it takes 4.6 s alone both at HEAD and with T3 (6.4 s cold at HEAD), so it is borderline already. Full run `-m "not real_data"` → 1031 passed, 23 skipped.
  - Negatives (scratch copies), each failing: finished file not counted whole; compact offset lagging one batch; LM compact position not updated, which first survived through file boundaries, so the check now needs a value inside a file; LM fixed position not updated; no calibration ticks; QC worker without slot; no phase reset; wrong `phases` when decoding once; QC offset miscounted.

- [x] **T4 — Workflow forwarding and conversion watch** (FR-1).
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

  Verified 2026-10-10 (Windows), test-first. Seams: the pure tracker `wf.ConversionWatch.observe(now, entries, exited)` and the coordinator's RunEvents through `ToolWorld`.
  - `ConversionWatch` matches the `discover_ldat` names (exact prefix, optional `_n`, `.ldat`). Splits that close in the same scan share the time since the previous close. The thread (`CONVERSION_WATCH_THREAD`, every `CONVERSION_WATCH_S` = 1 s) scans the attempt directory (files only, bounded by `artifact_limit`). It is stopped and joined when the runner returns; then one final report with `converter_exited: true` comes before `stage_finished`. `rawf_bytes` is the RAW size at the start (null if unreadable).
  - Red → green: `test_conversion_watch_closed_splits` (fake clock: 12/18/11 s, look-alike `<prefix>2_1.ldat` and `.lidx` not counted); `test_workflow_conversion_watch_events` (last report after exit, 3 durations for `_1`, `_2` and the empty `_9`, `ldat_bytes` = split sizes, no watch thread alive at `stage_finished`); `test_workflow_byte_forward` (failed with `KeyError: 'phases'`).
  - Guards: `test_conversion_watch_splits_closing_in_one_scan_share_the_time`; `test_workflow_slow_conversion_watch_reports_while_converting`, where a new `SplitWriterChild` writes a split every 0.3 s and the scan period is patched to 0.02 s.
  - Test helpers: `ToolWorld(converter_interval_s=...)` with `converter_writes`; the fake CLI prints one byte progress event; `execute(on_event=..., converter_interval_s=...)`.
  - The run record and outputs are unchanged: the existing conversion tests pass unmodified.
  - `python -m pytest tests/test_manager_workflow.py tests/test_manager_cli.py -m "not real_data" -n 0 --slow-limit 5` → 48 passed. Full run `-m "not real_data"` → 1036 passed, 23 skipped.
  - Negatives (scratch copies of `workflow.py`), each failing: no final report; thread never stopped; loose prefix; last split not closed at exit; shared time not split; `bytes_read` not forwarded; no scans while converting, which first survived the instant fake converter, hence the slow-converter guard.

- [x] **T5 — Run panel** (FR-1). A `RunPanel` above the tabs shows the action, `CTkProgressBar` (determinate or indeterminate), counter text, elapsed time and the estimate, refreshed from `_poll` at most every 250 ms.
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

  Verified 2026-10-10 (Windows), test-first. Seam: `_workflow_event` / `_start` / `_workflow_done` on a withdrawn window with `app.clock` replaced by a fake clock and `session.start_workflow` stubbed; the panel's widgets are read after `_poll` redraws (`RunPanelChecks`).
  - Logic in the Tk-free `progress.RunTracker` (events → `RunView`); `RunPanel` only renders it. Elapsed counts from `workflow_started` and freezes at `workflow_finished` (or a result without an outcome: "not started"). Each stage's estimate clock starts at its first event. The bar stays indeterminate until the 5 s / 1 % gate opens; a succeeded run fills it, any other end keeps it. Single-phase stages show no "phase k of n". Acquisition shows "RAW x.x MB written" (indeterminate; its time remaining is T8). Redraws at most every `PANEL_REFRESH_S` = 0.25 s, and at once after `_start`.
  - Red → green, one slice each: unknown total; bytes with gate and per-phase reset; fits keys; 1-split conversion; 3-split conversion; final state kept until the next run; failed run keeps its bar; run that never started; acquisition RAW counter. The "events/pairs/singles" check runs on every counter the tests read.
  - Tab status labels unchanged: the existing GUI tests pass unmodified.
  - `python -m py_compile exe_programs/petsys_manager_gui.py` OK. `python -m pytest tests/test_manager_progress.py tests/test_manager_gui_shell.py tests/test_manager_gui_processing.py tests/test_manager_gui_conversion.py tests/test_manager_gui_acquisition.py -m "not real_data" -n 0 --slow-limit 5` → 60 passed, 1 deselected. Full run `-m "not real_data"` → 1053 passed, 23 skipped.
  - Also marked `test_cli_calibrate_child_process_literal_paths_match_in_process` `slow` (~5 s; borderline in T3).
  - Negatives (scratch copies; unmodified copy 17 passed), each failing: no gate; phase index off by one; estimate clock not started with the stage; 1-split conversion gets a bar; elapsed not frozen; succeeded bar not filled; `_start` not resetting the tracker; `_workflow_done` not ending it; no acquisition counter; no redraw at `_start`.

- [x] **T6 — Stage overview** (FR-2). `progress.StageOverview(stages)` updated from `stage_started`/`stage_finished`/`workflow_finished`, plus a `RunPanel` stage row for `PIPELINE` and `QC`.

  **Done when:**
  - `python -m pytest tests/test_manager_progress.py -k overview` passes, covering: the success sequence gives all `succeeded` with recorded `elapsed_s`; a failure at stage 2 gives `failed` with the later stages `pending`; STOP during stage 1 gives `stopped` with the later stages `stopped`; the running stage's elapsed time ticks on a fake clock.
  - `python -m pytest tests/test_manager_gui_shell.py -k stage_row` passes: a pipeline shows 4 stage labels with their states, and a single-stage run shows no stage row.

  Verified 2026-10-10 (Windows), test-first. Seams: `StageOverview.event(now, kind, stage, payload)` / `rows(now)` and the `RunPanel` stage row through the T5 seam.
  - A stage runs from its first event. Acquisition emits no `stage_started`/`stage_finished`, so the next stage's start ends it as succeeded and `workflow_finished` ends it otherwise; its elapsed time is the GUI clock's. `cancelled` → stopped, `launch_error` and other results → failed. After a STOP, stages not run are `stopped`; otherwise they stay `pending`, shown as "not run" once the run ended. `RunTracker` keeps an overview only when `workflow_started` lists more than one stage.
  - Red → green: success sequence; failure at stage 2 (`failed`, `launch_error`); STOP during stage 1 and a failed acquisition; running elapsed ticks then recorded `elapsed_s`; GUI pipeline row (4 labels, states, "not run" after a failure) and no row for a single stage.
  - `python -m pytest tests/test_manager_progress.py -k overview` → 6 passed; `tests/test_manager_gui_shell.py -k stage_row` → 2 passed. Serial and full runs as in T5.
  - Negatives (scratch copies), each failing: earlier running stage not closed; pending not stopped after STOP; running stage without elapsed; `cancelled` read as failed; recorded `elapsed_s` ignored; no "not run"; stage row for a single stage; `workflow_finished` ignored.

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
