# Spec 005 — PETsys Manager GUI usability

Status: `approved` (2026-10-09)

Constitution: [`AGENTS.md`](../../AGENTS.md). Workflow: [`docs/prompts.md`](../../docs/prompts.md). Builds on spec 003 ([`../003-petsys-manager-migration/spec.md`](../003-petsys-manager-migration/spec.md)), shipped 2026-10-06 as PETsys Manager 1.0.0. This spec ships as PETsys Manager 1.1.0.

## Goal

Make the PETsys Manager quicker to use for its main work (running manual stages on existing files) and clearer during long runs and acquisitions. No numerical result, output file or run-folder layout changes.

## Owner answers (2026-10-05)

- Users: both Cornell operators (routine actions) and experts (tuning).
- Most used: manual stages (convert, calibrate, LM, offline QC) one at a time.
- Long runs: a progress bar with time remaining, and a stage overview. Time remaining comes only from the current stage's live rate.
- Past results: a Recent runs view with "open" and "use as input".
- After a stage: offer the next step with its inputs filled in; the operator starts it.
- Expert settings: a collapsible Advanced section per tab.
- File picking: pick a conversion run to select its LDATs; file dialogs remember their last folder.
- Warnings: a colored banner until dismissed, plus the log line.
- Display: the Cornell PC monitor (full HD or larger). Profiles change rarely, so there is no profile-switching work.
- Priority: progress bar and time remaining first.

## Clarify answers (2026-10-09)

- Processing progress uses byte positions, which the CLI progress events gain (bytes read / bytes total over all inputs). File i/N and record counts alone give no estimate for a single-file run.
- Conversion (external PETsys convert, which reports no progress): show LDAT bytes written next to the RAW size. When there are several splits, estimate the time remaining from how long the closed splits took; the estimate improves as more splits close.
- An estimate is shown only after 5 s and 1 % of progress.
- File dialogs' last folders are saved in the machine profile, in their own section, written immediately and never together with other profile edits.
- The Advanced fields are the ones listed in FR-5.
- Recent runs lists only runs with a `runs.tsv` line. Older run folders can still be picked as an input folder (FR-6).
- No processing warnings go to the banner. The banner shows failures, stopped runs and bias unknown. A new run clears all banners except bias unknown.
- New FR-8: a green banner, visible from every tab, says the acquisition RAW file is growing as expected. Clinicians doing dynamic imaging use it to time the radiotracer injection.

## Data contract

The changes are in the GUI, plus a progress counter and a profile section. System, map, calibration, energy units (a.u./keV), numbering and populations are those of spec 003 and do not change.

- Progress counts are the stage's own counters (bytes, records, files or fit keys read; LDAT or RAW bytes written), each labeled as such. They are never presented as event, pair or singles counts.
- CLI progress events gain `bytes_read`/`bytes_total`. No output file, `result.json`, `run.json` or `runs.tsv` content changes.
- The profile gains an optional `last_folders` section. A profile containing it can't be loaded by Manager 1.0.0, which rejects unknown fields.

## Requirements

- **FR-1 — Stage progress and time remaining.** WHEN a stage reports progress with a known total, the GUI SHALL show a progress bar, the counter it uses and an estimated time remaining computed only from this stage's progress rate so far. Known totals are input bytes, fit keys, acquisition time, or split count for conversion. WHEN the total is unknown, or before 5 s and 1 % of progress, the GUI SHALL show an indeterminate bar and the elapsed time, with no estimate. Elapsed time SHALL always be shown.
  - Processing stages (calibration, LM, offline QC) SHALL use `bytes_read`/`bytes_total` over all inputs from the CLI progress events. A stage with phases (calibration: read, pass 2, fits) SHALL show the current phase ("phase k of n"), and its bar and estimate SHALL cover that phase only.
  - Conversion SHALL show the LDAT bytes written so far next to the `.rawf` size, labeled as bytes, not as a fraction done. WHEN there are 2 or more splits, the estimate SHALL be (splits not yet closed) × (mean time per closed split). It is updated each time a split closes, and no estimate is shown before the first split closes. A split counts as closed when the next split file appears or the converter exits. With 1 split, conversion shows the unknown-total state.
- **FR-2 — Stage overview.** WHEN a multi-stage run (pipeline, live QC) is active or finished, the GUI SHALL list its stages with their state (pending, running, succeeded, failed, stopped) and elapsed time.
- **FR-3 — Recent runs view.** The GUI SHALL list the runs in `runs.tsv` in each of the profile's destinations (data, calibration, LM, report), newest first. Each row SHALL show finish time, run folder, action, inputs, status and main output. Run folders without a `runs.tsv` line are not listed. Each row SHALL offer: open the run folder; open its main report or plot when one exists; use as input (FR-4). The view SHALL be read-only: it never changes, moves or deletes a run.
- **FR-4 — Next step and use as input.** WHEN a manual conversion succeeds, the GUI SHALL offer "Calibrate", "Generate LM" and "Run QC", using the conversion's recorded LDAT outputs as inputs. WHEN a calibration succeeds, it SHALL offer "Generate LM with this calibration". Choosing an offer, or "use as input" in FR-3, SHALL fill the target tab's inputs and switch to that tab, and SHALL NOT start a run. Inputs come from the run record (`run.json`), never from similarly named files. The compact-coincidence confirmation, preflight and the stage's own validation still apply. A calibration is still never saved to the profile without the operator (spec 003).
- **FR-5 — Advanced settings.** Each tab SHALL show routine fields by default and keep expert settings in a collapsible Advanced section, collapsed at start:
  - Setup: DAQ fields and acquisition safety limits.
  - Conversion: max hits per side.
  - Calibration: COG limits file, positions per slab, event-limit mode, target sides per histogram, workers.
  - LM: COG and DOI limits files, pair map file, LM debug plots.
  - QC: none.

  Collapsing SHALL NOT change any value. WHEN a collapsed section holds a value different from its default, the GUI SHALL show that it does.
- **FR-6 — Picking inputs.** Input selection SHALL accept a conversion run folder and select exactly the LDATs recorded as its outputs in `run.json`. Each file dialog SHALL open in the folder last used for that input. The last folders are saved in the profile's `last_folders` section as soon as a dialog returns. Saving them SHALL NOT save, mark as edited or otherwise change any other profile field. A last folder that no longer exists is ignored.
- **FR-7 — Warning banner.** Failures, stopped runs and SiPM bias state unknown SHALL show in a colored banner above the tabs, in addition to the log, until dismissed. Starting a new run SHALL clear the failure and stopped banners. The bias-unknown banner SHALL stay until the operator confirms the bias state (spec 003 lock unchanged); it can't be dismissed and a new run doesn't clear it.
- **FR-8 — Acquisition growing banner.** WHEN an acquisition's RAW growth check passes (spec 003), the GUI SHALL show a green banner above the tabs with the run name, acquisition elapsed time, time remaining (Acq. Time − elapsed), current `.rawf` size and write rate. WHEN the RAW file then stops growing (the existing stall detection), the same banner SHALL turn red and say "RAW stopped growing", with the time since the last growth; it SHALL never stay green while stalled. The banner turns green again if the file resumes growing. The banner SHALL be removed when the acquisition ends; the end is shown in the stage overview and the log. This applies to every acquisition: Acquire, pipeline and live QC.

## Completion criteria

- Each FR-n has tracked tests in `tests/` marked `@pytest.mark.fr("005-FR-n")`, using synthetic fixtures:
  - time-remaining arithmetic, the 5 s / 1 % gate, the unknown-total state and the per-phase bar;
  - CLI `bytes_read`/`bytes_total` events on synthetic LDATs;
  - the conversion split estimate (0, 1, several closed splits; 1 split);
  - stage overview states from workflow events;
  - Recent runs parsing of `runs.tsv` (missing, empty, partial last line, tabs);
  - use as input and run-folder picking read `run.json` outputs and ignore look-alike files;
  - Advanced collapse keeps values and flags non-defaults;
  - `last_folders` is saved without changing other profile fields;
  - banners shown, dismissed, cleared by a new run, and bias locked;
  - growth banner green → red → green → removed, from acquisition events.
- The GUI compile check passes. A person checks the GUI at Cornell on a real conversion (with splits), calibration, LM, offline QC and an acquisition (growth banner).
- No change to any output file, run record or numerical result: the full run (`-m "not real_data"`) and `-m real_data` pass with the existing golden files and baselines unchanged, on Windows and the Cornell PC.
- PETsys Manager `__version__` is 1.1.0, with a `CHANGELOG.md` entry.

## Out of scope

Numerical or output changes; processing warnings in the banner; per-slab LM plots (declined by the owner, 2026-10-05); time remaining from past runs; finished notifications or sounds; repeat-run and run-details views; profile switching; remote/small-screen layouts; listing run folders without a `runs.tsv` line.
