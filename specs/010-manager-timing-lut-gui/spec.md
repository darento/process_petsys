# Spec 010 — Timing LUTs in the PETsys Manager GUI

Status: `shipped` (2026-10-20; T6 validation on Windows and the Cornell PC, [known limitations](#known-limitations-at-closing); PETsys Manager 1.3.0)

Constitution: [AGENTS.md](../../AGENTS.md). Workflow: [docs/prompts.md](../../docs/prompts.md). Builds on shipped [spec 009](../009-cornell-timing-walk/spec.md) FR-6 (listmode applies a skew LUT and a walk LUT to `dt`; profile keys `skew_lut_file`, `walk_lut_file`, `lm_timing_uncorrected`), which shipped with no GUI widget (PETsys Manager 1.2.0).

## Goal and population

Let the operator select, see and save the listmode timing correction in the PETsys Manager window instead of hand-editing the profile file. Show before START which correction applies, and after a run what it did.

- **System:** Cornell; the selected map's TIME channels (global IDs). PETsys Manager processes compact **coincidence** LDAT (two detectors per record), never singles.
- **Tables:** the spec 008 canonical skew LUT and the spec 009 walk LUT, unchanged. Timestamps and `dt` in ps.
- **Calibration:** the listmode calibration of the LM tab (LM action) or of the pipeline's own calibration stage (pipeline). keV, DOI and the correction itself are computed exactly as in spec 009; this spec changes no number written to an LM file.

**Owner decisions, 2026-10-20 (clarification):**
- A new spec, not a reopening of shipped spec 009. Ships as PETsys Manager 1.3.0.
- The two file pickers sit in the LM tab's main "LM File Generation Settings" section, not in its Advanced section.
- The keep/drop policy is two radio buttons.
- Feedback: a plan line before START and the correction counts in the LM result.

## Requirements

- **FR-1 — LUT file fields.**
  - The LM tab's "LM File Generation Settings" section shows two file rows with Browse, after "LM Destination": "Skew LUT (timing):" bound to `skew_lut_file` and "Walk LUT (timing):" bound to `walk_lut_file`.
  - Empty means unset (`None`). Reload shows the profile's values; Save writes them. As in 1.2.0, an unset field is not written to the profile file.
  - Edits mark the profile as having unsaved edits and re-run the prerequisites, like every other profile field.
- **FR-2 — Uncorrected-event policy.**
  - Two radio buttons, "Keep with raw dt (counted)" and "Drop (counted)", bound to `lm_timing_uncorrected` (`keep`, `drop`).
  - They are disabled while the skew LUT field is empty; their value is kept and still saved.
- **FR-3 — Plan line.**
  - The LM tab shows, from the current fields and before START, which correction the next run applies:
    - no skew LUT: `Timing correction: none (raw dt)`;
    - skew LUT only: the skew LUT file name and the policy;
    - both: both file names and the policy.
  - The complete pipeline's plan text states the same.
  - It is display only: it reads no table. Table validity is FR-5's.
- **FR-4 — Result counts.**
  - After a successful LM stage (LM tab or pipeline), the stage report shows, for a corrected run:
    - events corrected; events without a correction kept with raw `dt`, or dropped;
    - walk-input clamp counts, when non-zero;
    - a warning when the calibration differs from the walk fit's (`calibration_differs_from_fit`), with both SHA-256 prefixes.
  - For an uncorrected run it shows `Timing correction: none (raw dt)`.
  - The counts come from the listmode CLI result, never from re-reading the LM file. The CLI listmode summary gains `timing_correction`: the job record's block, or `null` for an uncorrected run.
- **FR-5 — Tables checked before START.**
  - The LM and pipeline prerequisites run the same table checks the LM stage runs: skew LUT of the selected map, natively corrected LUT refused, walk LUT COG/DOI limits, skew/walk pairing, calibration layout and region count.
  - LM action: against the LM tab's calibration file.
  - Pipeline: its calibration does not exist yet, so the layout comes from Positions per Slab (1: one factor per slab; N: N regions). A layout the walk LUT cannot accept is a prerequisite issue, before any acquisition.
  - A failing check is a prerequisite issue naming the field (Skew LUT or Walk LUT) and the reason; START stays disabled.
- **FR-6 — Release.** PETsys Manager `__version__` 1.3.0 and a CHANGELOG entry. Profiles stay readable by 1.2.0 (no new profile key).

## Completion criteria

- **Headless (synthetic):**
  - Profile round trip: set, save, reload and clear the three fields; an unset field is absent from the file.
  - Policy radios disabled exactly when the skew LUT field is empty.
  - Plan text for none / skew only / skew + walk, LM tab and pipeline.
  - Result text from a corrected and an uncorrected synthetic LM run: counts equal the job record's.
  - Prerequisites refuse a skew LUT of another map, an unpaired walk LUT, a walk LUT with other COG/DOI limits, and a pipeline whose positions give a layout the walk LUT refuses; they accept a matching set.
  - LM output bytes with and without tables equal the spec 009 tests' (unchanged code path).
- **Suites:** affected Manager GUI, settings, workflow, CLI and listmode tests; GUI module compile check; full non-real suite at Validation.
- **Operator review (Cornell PC, interactive):** set the LUTs in the LM tab, save, reload; run LM on a representative Cornell LDAT with and without tables; check the plan line, the counts and the job record.

## Known limitations at closing

- **Input correction state is not checked:** the LUTs were solved on LDAT converted without PETsys's native time-offset table. Listmode cannot tell a natively corrected LDAT from its contents, so the operator must apply the LUTs to uncorrected conversions only (`_conv_uncorr`). The Cornell `norm_10082026_r*` LDATs are natively corrected.
- **Pipeline layout check only:** before START, a pipeline's calibration is checked by layout and region count. Whether it differs from the walk fit's calibration is reported after the LM stage.
- **Production calibrations:** as in spec 009, the walk LUT accepts only per-slab calibrations, until a walk LUT is fitted with a region layout.

## Out of scope

- Fitting LUTs, skew/walk jobs or CTR reports from the GUI.
- Showing the correction in the Recent Runs tab.
- Any change to the correction, its refusals, or listmode output.
- IMAS (the Manager is Cornell-only).
- Manufactured singles.
