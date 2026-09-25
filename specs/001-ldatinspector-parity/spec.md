# Spec 001 — LDATInspector offline inspection parity

Status: `shipped`

Owner approved the inspector parity and repo-wide SDD plan. Hardware acquisition is outside this feature; offline analysis and reporting are in scope.

## Goal

Make the PETsys LDATInspector as usable as RAWInspector while retaining LDAT-specific multi-file ingestion, IMAS/Cornell mappings, and SuperModule status. Measurements must describe what the retained LDAT population actually supports.

## Requirements

- **FR-1** — WHEN opening the inspector, the UI SHALL present an integrated processing toolbar, console/status, and clearly named overview, explorer, and timing views in a RAWInspector-like appearance.
- **FR-2** — WHEN selecting files, the inspector SHALL support one or more `.ldat` inputs and show per-file success/failure, provenance, and a combined result without dropping unrelated selected files to match the worker limit.
- **FR-3** — WHEN reading valid records, the engine SHALL retain enough per-detector energy, position, DOI, channel and timestamp information for post-processing views without retaining an unbounded list of Python event dictionaries.
- **FR-4** — WHEN changing display energy/DOI/spatial cuts or selected SuperModule, the views SHALL update from the same retained population; ingestion's min-channel and per-channel-energy cuts SHALL be labelled separately.
- **FR-5** — WHEN selecting IMAS or Cornell, the inspector SHALL use the corresponding mapping, calibration, geometry and active-channel expectations, without a CMB-specific hardcoded module count.
- **FR-6** — WHEN browsing a SuperModule, the inspector SHALL show energy, DOI and flood map together, with adjustable flood-map bins/color scaling, ROI selection, profile projections, and reset controls.
- **FR-7** — WHEN displaying system health, it SHALL show expected vs observed time/energy channels, event counts and a geometry-aware overview, and distinguish insufficient evidence from a confirmed inactive channel.
- **FR-8** — WHEN fitting an energy histogram, the result SHALL include keV centroid/resolution only on a supported fit, explicitly label failure, and never substitute a zero-resolution estimate.
- **FR-9** — WHEN asked for photopeak uniformity, the inspector SHALL show per-SuperModule peaks against a configurable target and tolerance, reporting unavailable fits separately.
- **FR-10** — WHEN experimental background-aware fitting is enabled, the inspector SHALL use a PETsys-calibrated keV search interval, label the derived overlay and diagnostics, and keep the established fit separate.
- **FR-11** — WHEN showing a timing view, the inspector SHALL use the retained LDAT timestamp and its validated unit and label any insufficient or incomparable file spans; it SHALL NOT infer singles from coincidence-only data.
- **FR-12** — WHEN creating system or SuperModule reports, the inspector SHALL include dataset, map, calibration, filter, fit-status and timing provenance, with unsupported results marked unavailable rather than invented.
- **FR-13** — WHEN work is underway, processing SHALL not freeze the window; worker activity SHALL not call Tk widgets directly, and invalid or missing input SHALL leave a useful error state.
- **FR-14** — WHEN keyboard shortcuts or tooltips are displayed, they SHALL match the available actions; the program SHALL show its version in the title.
- **FR-15** — WHEN validating the feature, synthetic and representative IMAS/Cornell files SHALL exercise successful, sparse, missing-channel, corrupt, multi-file, and filtered cases; the operator SHALL review the resulting GUI and reports.
- **FR-16** — WHEN a histogram fit is shown, the predicted counts SHALL use the displayed histogram's bin edges and event selection, and unsupported or unstable photopeak widths/centroids SHALL not be presented as trusted measurements. This SHALL hold on the owner's Cornell acquisition and broad synthetic peaks.
- **FR-17** — WHEN opening a uniformity, fit-settings, profile, or residual window, it SHALL stay above its owning inspector window and receive focus without leaving a detached dialog behind.
- **FR-18** — WHEN showing photopeak uniformity, in-tolerance, out-of-tolerance and unavailable measurements SHALL have distinguishable, accessible row colors as well as text verdicts.
- **FR-19** — WHEN editing explorer energy, DOI or flood colour thresholds, linked slider and numeric fields SHALL allow adjustment and update the views using the selected module and dataset's energy units.
- **FR-20** — WHEN viewing a calibrated spectrum, the default overlay SHALL show the automatic photopeak model in displayed counts per bin (Gaussian plus the fitted local baseline), with its centroid and energy resolution; enabling the background control SHALL distinguish the Gaussian component and continuum, and identify the separate manually adjusted linear background fit and its own centroid/resolution when supported. A Gaussian component SHALL NOT be labelled as a fit to the total histogram.
- **FR-21** — WHEN the flood-map colour minimum is positive, zero-count bins SHALL be white/blank while positive bins at or below the minimum use the minimum colormap colour; the numeric minimum and maximum SHALL control the map's scale.
- **FR-22** — WHEN calibration is disabled, LDAT files SHALL process without an energy-calibration file, use raw PETsys energy (a.u.) for both sides' display cuts, and label raw spectra and reports accordingly. KeV-only fits and uniformity SHALL be explicitly unavailable. Mode-switching behavior is specified in FR-23.
- **FR-23** — WHEN loading coincidence files, the inspector SHALL retain raw energy and both detectors' calibration keys once; toggling calibration or selecting a new calibration file SHALL recompute the view without rereading LDAT, keep the event population and occupancy consistent, and mark missing calibration factors unavailable instead of fabricating keV values. Energy display/report units SHALL track the active mode.
- **FR-24** — WHEN using the explorer, raw energy SHALL have a fixed 0–300 a.u. horizontal plot range and calibrated energy a fixed 0–1500 keV range; energy and DOI cut controls SHALL move threshold markers without changing either histogram's axis limits. The controls SHALL be grouped over their corresponding energy, DOI and 2D plots.
- **FR-25** — WHEN viewing Channel Status, it SHALL show PETsys per-SuperModule summaries (coincidence detector-side counts, mapped/observed channel coverage, minimodule participation, selected energy/DOI measurements when supported) rather than imply CMB ADC status.
- **FR-26** — WHEN a calibration filename is loaded, the input-files card SHALL retain its width, showing a shortened visible name and a way to inspect the full path.
- **FR-27** — WHEN the owner compares the energy overlay with RAWInspector, the automatic total-model curve SHALL track the local photopeak histogram in displayed counts/bin and a supported background-aware fit SHALL show its energy resolution as well as peak centroid; overlapping automatic/manual models SHALL be identified without implying an independent second estimate.

## Out of scope

- Acquisition hardware, RemoteTask/simpleAcquirer connectivity, or RAW/CMB formats.
- Fabricating a singles bucket, SiPM-site maps, calibrated DOI millimetres, or a CMB ADC-level Compton search window from LDAT files.
- Declaring clinical PASS/FAIL thresholds without a PETsys-specific policy approved against suitable baseline data.

## Completion criteria

Each FR has a named verification; no stale views survive a new process/filter selection, reports reproduce the displayed measurements, and the owner accepts the interface. The owner accepted mapped synthetic IMAS validation in place of a real IMAS acquisition and waived separate manual PDF-layout review for this release; the absence of those two checks remains explicit in the validation record.

The owner supplied a Cornell full-system acquisition and requested fit, popup and status-colour corrections while this spec was in progress. FR-16–FR-18 are its approved change scope; the six actual files are `...coincCompact11s_00000003.ldat` through `...00000008.ldat` (the supplied `03-08.ldat` range is not a single file).

The owner requested image-style sliders, distinct Gaussian and opt-in background fits, white zero-count flood bins and a calibration switch; they clarified that the switch must support ingesting raw ADC without a calibration file. FR-19–FR-22 are this change scope. Operator acceptance remains open.

The owner revised calibration switching to use a single ingest and specified 0–300 a.u. and 0–1500 keV axes, fixed cut-line behavior, plot-aligned controls, per-module Channel Status and a non-resizing file card. FR-23–FR-26 supersede FR-22's reprocessing clause and its earlier ADC shorthand; existing report and validation requirements still apply.

The owner identified an apparent under-fit in the Cornell spectrum: the red line was the Gaussian *component* of a background-aware model and therefore sat below measured counts. FR-27 and the FR-20 clarification cover total-model display and separate background-fit resolution. The owner approved closing the feature after this fix and explicitly waived the outstanding real-IMAS acquisition and manual PDF-layout checks; other unobserved operational claims must not be presented as measured evidence.

Final validation and the owner-approved waivers are recorded in `tasks.md` (2026-09-25). The feature is accepted for offline PETsys coincidence inspection; waived checks remain named as unperformed, not passed.
