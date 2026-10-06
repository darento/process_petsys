# Spec 005 — PETsys Manager GUI usability

Status: `draft`

Constitution: [`AGENTS.md`](../../AGENTS.md). Workflow: [`docs/prompts.md`](../../docs/prompts.md). Builds on spec 003 ([`../003-petsys-manager-migration/spec.md`](../003-petsys-manager-migration/spec.md)). No implementation before spec 003 is closed (owner, 2026-10-05); a problem that blocks spec 003 acceptance is fixed there.

## Goal

Make the PETsys Manager quicker to use for its main work, manual stages on existing files, and clearer during long runs, without changing any numerical result, output file or run-folder layout.

## Owner answers (2026-10-05)

- Users: both Cornell operators (routine actions) and experts (tuning).
- Most used: manual stages (convert, calibrate, LM, offline QC) one at a time.
- Long runs: progress bar with time remaining, and a stage overview. Time remaining from the current stage's live rate only.
- Past results: a Recent runs view with "open" and "use as input".
- After a stage: offer the next step with its inputs filled in; the operator starts it.
- Expert settings: a collapsible Advanced section per tab.
- File picking: pick a conversion run to select its LDATs; file dialogs remember their last folder.
- Warnings: a colored banner until dismissed, plus the log line.
- Display: the Cornell PC monitor (full HD or larger). Profiles change rarely: no profile switching work.
- Priority: progress bar and time remaining first.

## Data contract

GUI only. System, map, calibration, energy units (a.u./keV), numbering and populations are those of spec 003 and do not change. Progress counts are the stage's own counters (bytes, records or keys read), labeled as such; they are never presented as event, pair or singles counts.

## Requirements

- **FR-1 — Stage progress and time remaining.** WHEN a stage reports progress with a known total (bytes, records, files or fit keys), the GUI SHALL show a progress bar, the counter it uses and an estimated time remaining computed only from this stage's progress rate so far. WHEN the total is unknown, or too little progress exists for an estimate, the GUI SHALL show an indeterminate bar and the elapsed time, with no estimate. Elapsed time SHALL always be shown.
- **FR-2 — Stage overview.** WHEN a multi-stage run (pipeline, live QC) is active or finished, the GUI SHALL list its stages with their state (pending, running, succeeded, failed, stopped) and elapsed time.
- **FR-3 — Recent runs view.** The GUI SHALL list the runs from `runs.tsv` in the profile's destinations (data, calibration, LM, report), newest first, with finish time, run folder, action, inputs, status and main output. Each row SHALL offer: open the run folder; open its main report or plot when one exists; use as input (FR-4). The view SHALL be read-only: it never changes, moves or deletes a run.
- **FR-4 — Next step and use as input.** WHEN a manual conversion succeeds, the GUI SHALL offer "Calibrate", "Generate LM" and "Run QC" with its recorded LDAT outputs as inputs; WHEN a calibration succeeds, it SHALL offer "Generate LM with this calibration". Choosing an offer (or "use as input" in FR-3) SHALL fill the target tab's inputs and switch to it, and SHALL NOT start a run. Inputs come from the run record (`run.json`), never from similarly named files. The compact-coincidence confirmation, preflight and the stage's own validation still apply; a calibration is still never saved to the profile without the operator (spec 003).
- **FR-5 — Advanced settings.** Each tab SHALL show routine fields by default and keep expert settings (e.g. workers, seeds, memory budget, calibration target per key) in a collapsible Advanced section. Collapsing SHALL NOT change any value; WHEN a collapsed section holds a value different from the default, the GUI SHALL show that it does.
- **FR-6 — Picking inputs.** Input selection SHALL accept a conversion run folder, selecting exactly the LDATs recorded as its outputs. Each file dialog SHALL open in the folder last used for that input.
- **FR-7 — Warning banner.** Failures, stopped runs, SiPM bias state unknown and processing warnings (e.g. calibration keys without a fit) SHALL show in a colored banner until dismissed, in addition to the log. The bias-unknown banner SHALL stay until the operator confirms the bias state (spec 003 lock unchanged).

## Completion criteria

- Headless checks: time-remaining arithmetic and the unknown-total state; stage overview states from workflow events; Recent runs parsing of `runs.tsv` (missing, empty, partial last line, tabs); use-as-input reads `run.json` outputs and ignores look-alike files; Advanced collapse keeps values; banner shown/dismissed/locked.
- GUI compile check; a person checks the GUI at Cornell on a real conversion, calibration, LM and offline QC.
- No change to any output file, run record or numerical result (spec 003 checks pass unchanged).

## Out of scope

Numerical or output changes; per-slab LM plots (declined by the owner, 2026-10-05); time remaining from past runs; finished notifications or sounds; repeat-run and run-details views; profile switching; remote/small-screen layouts.

## Open questions (Clarify)

1. Minimum progress before an estimate is shown (e.g. 5 s and 1 %)?
2. Exact list of Advanced settings per tab.
3. Should Recent runs list pre-T29 run folders (no `runs.tsv` line)?
4. Last-used folders: for this session only, or saved in the profile?
5. Which processing warnings go to the banner, and does a new run clear old banners?
