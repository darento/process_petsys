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
  - Fix 2026-10-10, found in manual UI check: before the first run the bar spun indeterminate, as if a run with an unknown total were active. `RunView.idle` now marks "no run requested yet"; the panel then shows "● Ready", a hint and a still empty bar whose colour breathes over 3 s (user's choice), and stops breathing on START. Regression `test_run_panel_idle_before_the_first_run_breathes_without_spinning`. Serial run of `test_manager_gui_shell.py` and `test_manager_progress.py` → 41 passed; full run `-m "not real_data"` → 1054 passed, 23 skipped. Negatives each failing: breathing not stopped at START; never breathing; idle while a started run waits for `workflow_started`.

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

- [x] **T7 — Warning banners** (FR-7). `progress.Banners` and a `BannerArea` above the tabs.
  - The existing bias frame and `acknowledge_bias` move there; the lock is unchanged.
  - Failure banners: a failed workflow, failed initialization, DAQD `FAILED`. A stopped banner: a cancelled workflow.
  - `_start` calls `new_run()`.

  **Done when:**
  - `python -m pytest tests/test_manager_progress.py -k banners` passes, covering: show; dismiss; `new_run` clears failure/stopped and keeps bias; bias can't be dismissed and only `acknowledge_bias` clears it.
  - `python -m pytest tests/test_manager_gui_shell.py -k banner` passes: a failed outcome shows a dismissible red banner on every tab; a new run clears it; a bias banner stays across a new run and live actions stay locked until acknowledged (existing spec 003 bias tests pass unchanged).

  Verified 2026-10-10 (Windows), test-first. Seams: `Banners.show/dismiss/new_run/acknowledge_bias/items` and the window (`banner_area` rows, `_workflow_done`, `init_done`/`daqd` shell events, `_start`, `bias_ack`).
  - One banner per kind holds its latest text, shown in the order bias, failure, stopped. Failure covers `failed` and `launch_error`; a cancelled workflow gives "<action> stopped: ...". A run that never started (refused) gives no banner. DAQD `FAILED` shows once per transition into it, so an unchanged FAILED revision doesn't bring a dismissed banner back. The banner area is packed only while a banner is active. `bias_frame`/`bias_label`/`bias_ack` are now the bias row of the area.
  - `python -m pytest tests/test_manager_progress.py -k banners` → 4 passed; `tests/test_manager_gui_shell.py -k banner` → 4 passed.
  - `python -m py_compile exe_programs/petsys_manager_gui.py` OK. `python -m pytest tests/test_manager_gui_shell.py tests/test_manager_gui_acquisition.py tests/test_manager_progress.py -m "not real_data" -n 0 --slow-limit 5` → 56 passed (spec 003 bias test unchanged). Full run `-m "not real_data"` → 1062 passed, 23 skipped.
  - Negatives (scratch copies; unmodified copy 8 passed), each failing: bias dismissible; `new_run` clears bias; `_start` not calling `new_run`; DAQD FAILED re-shown each revision; cancelled shown as failure; failed initialization not shown; empty area kept packed; `launch_error` not shown.

- [x] **T8 — Growth banner** (FR-8). `progress.GrowthBanner` and its `BannerArea` row show run name, attempt elapsed, time remaining (duration from the request or the QC preset), `.rawf` size and MB/s, refreshed once per second.

  **Done when:**
  - `python -m pytest tests/test_manager_progress.py -k growth` passes, covering:
    - hidden before `growth_passed`; green after it;
    - `growing: false` turns it red with the time since the last growth; `growing: true` turns it green again;
    - `attempt_started` (retry) resets it to hidden with a new clock;
    - `aborting`, `attempt_finished`, the acquisition `stage_finished` and `workflow_finished` hide it;
    - time remaining is never negative.
  - `python -m pytest tests/test_manager_gui_acquisition.py -k growth_banner` passes: the acquisition event sequence from the existing fake acquisition shows green → red → green → removed, with the run name and both times in the text, for Acquire, pipeline and live QC.

  Verified 2026-10-10 (Windows), test-first. Seams: `GrowthBanner.reset/event/view` and the window (`banner_area` growth row through `_workflow_event` and `_poll`).
  - `GrowthView` carries numbers only; the window formats "<run>: RAW file growing as expected. Elapsed …, … remaining. RAW x MB at y MB/s" or "<run>: RAW stopped growing … ago. …". Stalled time counts from the last growth (the growth check or a growing `rawf_progress`). The duration comes from the request options of Acquire, pipeline and QC (QC carries its preset). Order in the area: bias, growth, failure, stopped. The row redraws at once when its state changes and otherwise at most once per second; `_workflow_done` also clears it.
  - The GUI test base `RunPanelBase` moved to `tests/manager_gui_helpers.py` (gains `options`, `run_root`). The fake acquisition gains a `pulse` behaviour (grow, pause ~0.6 s, grow).
  - `python -m pytest tests/test_manager_progress.py -k growth` → 12 passed; `tests/test_manager_gui_acquisition.py -k growth_banner` → 4 passed: fed sequences for Acquire, pipeline and live QC, and one real Acquire on the `pulse` fake (green → red → green in order, then removed after STOP). That test passed 8 of 8 when run 8 at once; it checks order, not exact redraws, because a loaded machine may skip one.
  - `python -m py_compile exe_programs/petsys_manager_gui.py` OK. `python -m pytest tests/test_manager_gui_acquisition.py tests/test_manager_gui_shell.py tests/test_manager_progress.py -m "not real_data" -n 0 --slow-limit 5` → 72 passed. Full run `-m "not real_data"` → 1078 passed, 23 skipped.
  - Negatives (scratch copies; unmodified copy 16 passed), each failing: shown before the growth check; never red; retry keeping the clock; stalled time from the stall check; negative remaining; any stage's end hiding it; `workflow_finished` ignored; no once-per-second refresh; duration not taken from the request; red text saying growing; window not feeding events.

## Runs and inputs

- [x] **T9 — Reading runs.tsv** (FR-3). New `src/petsys_manager/recent.py` (no Tk): `read_overview(destination)`, which reads the last 1 MiB and returns rows plus a skipped count, and `list_recent(destinations)` (merge newest first, cap 500, duplicate destinations read once).

  **Done when:** `python -m pytest tests/test_manager_recent.py -k overview` passes, covering:
  - missing file → no rows plus a "no runs recorded" state; empty file and header only → no rows;
  - partial last line skipped; wrong column count skipped and counted; a non-portable `run_folder` (`..`, `a/b`) skipped;
  - a 3 MiB file reads only its tail;
  - merge order across two destinations; cap 500;
  - files written by `workflow.append_overview` parse back to their rows.

  Verified 2026-10-10 (Windows), test-first. Seam: `read_overview(destination, key)` → `Overview(rows, skipped, recorded)` and `list_recent([(key, destination), ...])` → `Recent(rows, skipped, unrecorded)`, rows being `RecentRun`.
  - The header, a partial last line and the line cut at the 1 MiB tail start are dropped without counting; lines without 6 columns, a `%Y-%m-%d %H:%M:%S` time or a portable run folder (`[A-Za-z0-9][A-Za-z0-9_-]*`) are counted. Equal finish times keep the later line first. A destination listed twice (same absolute path) is read once, under its first key.
  - `python -m pytest tests/test_manager_recent.py -k "overview or recent or reading"` → 18 passed, including a check that reading changes neither content nor mtime.
  - Negatives (scratch copies; unmodified copy 18 passed), each failing: whole file read; partial last line kept; header counted; folder not checked; duplicate read twice; no cap; equal times in file order; missing file not reported.
  - Serial and full runs as in T10.

- [x] **T10 — Run records: offers, outputs, main report** (FR-3, FR-4, FR-6). In `recent.py`, `run_offers(root)`, `conversion_outputs(root)` and `main_report(root)` use `artifacts.read_manifest`. Before an offer, each file's existence and recorded `size_bytes` are checked.

  **Done when:** `python -m pytest tests/test_manager_recent.py -k "offers or outputs or main_report"` passes on synthetic `RunStore` runs, covering:
  - a conversion run gives 3 LDAT offers with exactly its recorded outputs in order, and a look-alike `.ldat` in the folder is ignored;
  - a calibration run gives the LM-calibration offer with its `.encal`;
  - a pipeline run gives both kinds; a failed or cancelled stage gives none;
  - a resized output → refused, naming the file; a pre-T29 run folder still reads;
  - a non-run folder → refused;
  - `main_report`: QC → `qc_report`, calibration → `calibration_plot`, LM → first debug plot or none, conversion → none.

  Verified 2026-10-10 (Windows), test-first. Seam: `run_offers(root)` → `Offer(label, target, payload)`, `conversion_outputs(root)`, `main_report(root)`; refusals raise `RunRecordError` with the file or folder named.
  - Each stage's latest attempt counts. Only files in a succeeded attempt's `outputs` are used; a file recorded during the run but not an output is never offered. LDAT descriptors come from the record (format, population, `validated`). `main_report` looks at the run's last stage only (a pipeline gives its LM debug plot) and returns None when that file is gone.
  - `python -m pytest tests/test_manager_recent.py` → 31 passed (13 for T10).
  - Negatives (scratch copies; unmodified copy 13 passed), each failing: failed stage offered; size not checked; `.encal` not checked; last debug plot instead of first; first stage's report; non-output artifacts offered; no refusal without conversion outputs; `validated` dropped.
  - `python -m pytest tests/test_manager_recent.py -m "not real_data" -n 0 --slow-limit 5` → 31 passed. Full run `-m "not real_data"` → 1110 passed, 23 skipped. One earlier full run had 1 failure in `test_injected_failures_and_stop_never_become_success` (pipeline acquisition not launched within 20 s under load); it passed alone 3 of 3, 8 of 8 run at once, and on the next full run. Nothing it uses changed in T9/T10, so it is recorded as a flake.

- [x] **T11 — Next-step offers** (FR-4).
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

  Verified 2026-10-10 (Windows), test-first. Seam: the window (`offer_calibrate`/`offer_listmode`/`offer_qc_analyze`/`offer_lm_calibration` buttons, `apply_offer`).
  - `offer(target)` reads `run_offers(run folder)` of the run shown in the result frame when clicked; a `RunRecordError` is logged as "<offer> not offered: …" and nothing changes. `last_conversion`/`last_calibration` gain the run folder as a third element. `use_conversion_outputs`, `use_calibration` and `OUTPUT_TARGETS` are gone; labels come from `recent.OFFER_LABELS`.
  - Spec 003 tests: the conversion test clicks the three offers instead of "Use these outputs" and the processing test clicks "Generate LM with this calibration" instead of "Use this .encal"; their expected inputs are unchanged. The processing test now also checks the switch to the LM tab with no run started.
  - `python -m pytest tests/test_manager_gui_conversion.py tests/test_manager_gui_processing.py -k offer` → 6 passed: each offer fills only its list, switches tab and starts nothing; a resized output is refused with the file named; offers disabled after a failed conversion; confirmation and validation still gate the start; the calibration offer is an unsaved edit, switches to LM and leaves the profile file unchanged; it is disabled after a failed calibration (synthetic run record fed through `_workflow_done`).
  - `python -m py_compile exe_programs/petsys_manager_gui.py` OK. `python -m pytest tests/test_manager_gui_conversion.py tests/test_manager_gui_processing.py -m "not real_data" -n 0 --slow-limit 5` → 19 passed, 1 deselected.
  - Full run `-m "not real_data"` → 1116 passed, 23 skipped, twice. Two earlier full runs failed `test_conversion_stop_and_failures_never_publish_outputs` at window open ("readiness never settled" within 5 s). It passes serially, with 4 workers and in a verbose full run. The idle breathing measured about 2 % of a core per window, so it is not the cause. `GUIBase.settle` and `settle_edit` now wait up to `SETTLE_S` = 15 s. They return once readiness settles, and AGENTS.md asserts wall-clock bounds only in serial runs.
  - Negatives (scratch copies; unmodified copy 6 passed), each failing: no tab switch; conversion offers enabled after a failure; all three lists filled; refusal not caught; calibration offer switching to the calibration tab; calibration offer enabled after a failure; a run started by the offer.

- [x] **T12 — Recent Runs tab** (FR-3, FR-4). The tab has a `ttk.Treeview` with the spec columns and buttons Refresh / Open folder / Open report / Use as input. It is filled by a session worker job (`ShellEvent("recent", ...)`) when the tab is opened, on Refresh and after `workflow_done`. `open_path` is an attribute.

  **Done when:** `python -m pytest tests/test_manager_gui_shell.py -k recent_runs` passes, covering:
  - a fake `recent` event fills rows newest first;
  - Open folder / Open report call `open_path` with the run folder / `main_report` path; Open report is disabled without one;
  - Use as input on a conversion row offers the 3 targets and applies one as in T11;
  - the skipped count shows in the status line;
  - no file in the destinations changes (tree hash before and after);
  - reading runs off the Tk thread (the session job is used).

  Verified 2026-10-10 (Windows), test-first. Seam: the window (`recent_tree`, `recent_status`, `recent_*` buttons, `open_path`, `ask_offer`) and `session.list_recent()`.
  - `session.list_recent()` reads `recent.profile_destinations(saved profile)` → `list_recent` in a `petsys-recent` worker and posts `ShellEvent("recent", Recent)`; a failure arrives as "refused". The tab refreshes when it becomes the shown tab (checked each poll), on Refresh and after every `workflow_done`. Open report is enabled only when `main_report` finds a file. Use as input calls `run_offers` and asks with `ask_offer`, a small modal dialog by default. The chosen offer goes through `apply_offer` (T11); a run without offers is logged. Refresh is always enabled; Use as input is disabled while a run is active.
  - The synthetic run helpers (`record_stage`, `recorded_run`, `SPLITS`, `CALIBRATION`) moved from `test_manager_recent.py` to `tests/manager_helpers.py`.
  - Spec 003 `test_tabs_log_and_startup` now expects the sixth tab "Recent Runs" and Refresh enabled at startup, as FR-3 requires; nothing else in it changed.
  - The panel now configures labels and the bar only when they change, and the breathing uses a precomputed palette. An idle window's CPU is then back to the pre-T5 level within measurement noise (about 0.1–0.2 s per 5 s, both trees).
  - Readiness waits in the GUI tests use `SETTLE_S` (15 s) instead of a hard-coded 5 s (see T11).
  - `python -m pytest tests/test_manager_gui_shell.py -k recent_runs` → 5 passed: a fake event fills rows in order with the skipped count and the unrecorded destinations; real runs.tsv files read off the Tk thread (thread names recorded); Open folder / Open report call `open_path` with the run folder / `main_report`; Open report disabled for a conversion; Refresh and `workflow_done` re-read; Use as input offers the three targets, a cancel changes nothing and a choice fills only LM and switches tab; a QC run has no offers; file contents and mtimes under the destinations are unchanged; the real dialog returns the clicked offer, or None on Cancel. That test retries until the button exists and closes the dialog after 10 s, because an earlier version left a full run waiting on the open dialog.
  - Serial run of the GUI, recent and progress test files with `-n 0 --slow-limit 5` → 127 passed, 1 deselected. Full run `-m "not real_data"` → 1121 passed, 23 skipped. Two earlier full runs failed `test_incomplete_metadata_options_and_prerequisites_cannot_start` (readiness not settled within 15 s, at two different steps). It passes serially (13 s), and in a scratch export of this tree run as a full suite. The export of HEAD passes it as well. It is recorded as a scheduling-dependent flake to watch in T16.
  - Negatives (scratch copies; unmodified copy 5 passed), each failing: reading on the Tk thread; no refresh when the tab opens; no refresh after a run; Open report enabled without a report; Open folder opening the report; first offer applied without asking; skipped count not shown; rows reversed; Use as input enabled without a selection.

## Last folders and run-folder picking

- [x] **T13 — Profile `last_folders`** (FR-6). In `settings.py`, `MachineProfile.last_folders` (fixed key set, absolute folders) and `save_last_folders(path, folders)` (re-read from disk, replace only `last_folders`, atomic write); `session.save_last_folders(key, folder)`.

  **Done when:** `python -m pytest tests/test_manager_settings.py -k last_folders` passes, covering:
  - round trip; an unknown key or a relative path → `ProfileError`;
  - a 1.0.0-style profile without the section still loads;
  - saving keeps every other on-disk field equal (parsed) and ignores a different in-memory profile;
  - `session.profile` gets the folders while `profile_from_ui() != session.profile` stays as it was;
  - no profile file → nothing written, folders in memory, one log line;
  - write failure → logged, folders in memory.

  Verified 2026-10-10 (Windows), test-first. Seam: `MachineProfile.last_folders`, `settings.save_last_folders`, `session.save_last_folders`.
  - `settings.LAST_FOLDER_KEYS` is the fixed key set: `raw_input`, `ldat_inputs.{calibrate,listmode,qc_analyze}`, `conversion_run` and every profile path field. Folders are stored sorted by key, so key order never makes a different profile.
  - `save_profile` leaves out an empty `last_folders`, so a profile saved by 1.1.0 before any dialog is used still loads in 1.0.0.
  - `save_last_folders` re-reads the file, replaces only the section, validates the whole result as a profile and writes it with the same temp-file + `os.replace` path as `save_profile` (now the shared `_replace_file`). A file that is not a valid profile is refused unchanged.
  - `session.save_last_folders` updates `session.profile` first. Without a profile file it logs once per session; a write failure is logged on each attempt.
  - `python -m pytest tests/test_manager_settings.py -k last_folders` → 9 passed: round trip and key-order equality; unknown key, relative/empty/non-text folder, non-mapping and an unknown key in a file → `ProfileError`; a 1.0.0 profile loads and a save without folders omits the section; `save_last_folders` keeps every other field equal (parsed), also on a 1.0.0 file with a retired capability; a bad folder, a processing YAML and a failing `os.replace` leave the files byte-identical with no temp file; the session writes file and memory; a file changed by another window keeps its fields (the session's in-memory profile is ignored); no file → nothing written, one log line for two calls; write failure → one log line, folders in memory.
  - The `profile_from_ui() != session.profile` item is checked through a real window in T14 (`test_last_folder_choice_saved_without_marking_the_profile_edited`): `test_manager_settings.py` asserts that importing it loads no Tk.
  - Negatives (scratch copies; unmodified copy passed), each failing: no unknown-key check; no absolute-path check; unsorted storage; empty section written; fields taken from defaults instead of disk; no validation before writing; non-atomic write; the no-file line logged on every call; the session saving its whole in-memory profile (killed after adding the "changed by another window" test); `session.profile` not updated; `OSError` not caught.

- [x] **T14 — Dialog folders and "Add conversion run..."** (FR-6).
  - Every dialog except the profile file starts in `last_folders[key]` when it exists and saves the chosen folder through `session.save_last_folders`.
  - `InputSelection` gains "Add conversion run...": a directory dialog, then `conversion_outputs`, then `use_outputs`, with refusals logged.

  **Done when:** `python -m pytest tests/test_manager_gui_processing.py -k "last_folder or conversion_run"` passes, covering:
  - a dialog opens in the stored folder; a missing stored folder falls back to today's rule;
  - the choice is saved under its key without marking the profile edited;
  - picking a conversion run folder lists exactly its recorded LDATs in order, with the origin shown;
  - a non-run folder and a run without conversion outputs are refused in the log with the list unchanged.

  The full run passes.

  Verified 2026-10-10 (Windows), test-first. Seam: the window (`ask_file`, `ask_directory`, `ask_files`, `_browse(name, kind)`, `_browse_raw`, `InputSelection.buttons["run"]`).
  - `start_folder(key, fallback)` returns `last_folders[key]` while it is a folder, else today's start folder; `remember_folder` calls `session.save_last_folders` as soon as a dialog returns a choice. A cancelled dialog remembers nothing.
  - What is remembered: a file dialog stores the chosen file's folder; a profile folder dialog stores the chosen folder; "Add conversion run..." stores the run folder's parent (the destination holding the runs), so the next pick starts among the runs. Its fallback is the Output Data Folder, else home.
  - "Add conversion run..." goes through `conversion_outputs` and then `apply_offer` (T11): the list is replaced by the recorded LDATs, confirmed with the converter's declaration, origin "conversion run <folder>", logged with any replaced count. A refusal logs "Conversion run not added: <reason>" and leaves the list and origin as they were.
  - The profile-file dialog is unchanged.
  - Spec 003 `ProcessingChecks` compared the profile file byte for byte and `session.profile` exactly; picking LDATs now writes `last_folders`, as FR-6 requires. They now compare the parsed profile and `session.profile` with `last_folders` left out (`settings()`, `ui_state`); every other field is still checked.
  - `python -m pytest tests/test_manager_gui_processing.py -k "last_folder or conversion_run"` → 4 passed: five dialogs (COG limits file, Output Data Folder, RAW, LDAT list, conversion run) open in their stored folders, and with stored folders gone in today's folders; cancelling writes nothing; an LDAT choice is written under `ldat_inputs.listmode` with every other on-disk field equal and no "unsaved edits"; with an unsaved Output Data Folder edit pending, a COG file choice writes only the folder, the edit stays in the field and unsaved, `session.profile` keeps the disk value; RAW and Report folder choices are written; a run folder lists exactly its three recorded LDATs in order (an unrecorded LDAT beside them is left out) with the origin shown; a plain folder, a calibration run and a failed conversion are refused with the list unchanged; a cancelled run dialog logs nothing.
  - Serial run of the settings file and the four Manager GUI files with `-n 0 --slow-limit 5` → 106 passed, 2 failed: the two spec 003 tests above, before they were changed. The processing file after the change → 12 passed.
  - Full run `-m "not real_data"` → 1134 passed, 23 skipped. Two earlier full runs each failed spec 003 conversion GUI tests: `test_close_during_conversion_closes_on_the_first_request` in both (10 s wait for "Writing to:") and `test_conversion_request_format_population_and_exact_outputs` in one. Neither opens a dialog. Both pass serially, and the conversion, processing and shell files passed three parallel runs (51 passed each). Recorded as load flakes to watch in T16, with T12's.
  - Negatives (scratch copies; unmodified copy 4 passed), each failing: a stored folder used when gone; stored folders never used; nothing saved; the whole profile saved instead; a file's own path stored; the run folder stored instead of its parent; LDATs globbed from the folder; refusal not logged; RAW choice not saved (killed after extending the save test); LDAT choice not saved; LDAT dialog ignoring the stored folder; RAW dialog reading another key; a cancelled run dialog treated as a folder.

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
