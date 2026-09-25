# Plan 001 — LDATInspector offline inspection parity

Approved direction: offline analysis parity and PETsys-specific diagnostics, without hardware acquisition. The repo-wide SDD constitution is `AGENTS.md`; phases and checks are in `docs/prompts.md`.

## Architecture

- `src/ldat_inspector.py`: pure input validation, per-file PETsys decoding, bounded columnar per-SuperModule event arrays and channel occupancy, merge, mask selection, histogram fits, status, timing and uniformity. No GUI imports. Retain file index and selected time-channel timestamp for provenance. The population is detector sides of accepted coincidence records; count pairs separately.
- `exe_programs/LDATInspector.py`: CustomTkinter view and worker orchestration, plot interactions, dialogs and report actions. Tk widgets are touched only on the main thread. Multiprocessing is bounded by both number of files and CPU availability.
- `src/ldat_report.py`: report construction from analysis results; state reasons for unavailable metrics and use the same masks as the explorer.

Filters currently run before data is stored in the legacy GUI; retaining pre-energy-window events is necessary for post-process filtering. Default display energy window should match config. DOI remains a ratio. For repeated files, show per-file timestamps rather than silently splicing separate acquisitions into a single live-time axis. Expected channel sets come from the mapping (and the Cornell inactive-minimodule rule where applicable), not a fixed 256-channel assertion. Missing-channel findings are observational and depend on event statistics and acquisition cuts.

## Decisions

- Rebuild the standalone GUI around a small analysis engine rather than copying CMB's `cmb/` imports or building more callbacks into the 2,661-line Tk file. Preserve its existing supported input formats, mapping, calibrated spectra, multi-file merge and status intent.
- Use compact arrays instead of retaining every detector event as a dictionary; store channel activity as per-file counters. Where a channel measurement cannot be recalculated after a display cut, label its ingest-population scope.
- Use `src.fits.fit_gaussian` for the legacy keV fit, guarding its sparse/degenerate cases. An optional continuum-aware fit must validate its own keV interval and disclose fit quality; it must not be called a correction of event energies.
- Use `matplotlib.backends.backend_pdf.PdfPages` for reproducible PDFs and a user-selected destination rather than a CMB-specific hardcoded directory.
- CustomTkinter is a new dependency for the standalone program; add it to `process_petsys.yml`. No CMB asset or machine-specific path is required.

## Requirements to components

| Requirement | Components |
| --- | --- |
| FR-1, FR-6, FR-14 | LDATInspector GUI |
| FR-2, FR-3, FR-5, FR-13 | Engine and GUI worker coordinator |
| FR-4, FR-7, FR-8, FR-9, FR-10, FR-11 | Pure analysis + relevant GUI views |
| FR-12 | Pure analysis, reporter and GUI action |
| FR-15 | Engine checks + human acceptance |
| FR-16 | Bin-aware fit predictions, constrained peak logic, GUI and report overlays |
| FR-17, FR-18 | Parent-owned dialogs, uniformity Treeview tags and GUI regression |
| FR-19–FR-21 | Explorer controls, fit component overlay and masked 2D histogram |
| FR-22 | Optional calibration in ingestion, dataset energy-mode provenance, raw-only unavailable fits in GUI and PDF |
| FR-23–FR-26 | Reversible calibration arrays, explorer axes/controls, PETsys channel view and fixed input card |
| FR-27 | Automatic total-model overlay, opt-in component/adjusted-fit labels and display-bin regression |

## Explorer and raw-energy change (FR-19–FR-22)

Keep paired energy selection on the active energy columns: calibrated keV or unmodified minimodule sums (PETsys a.u.). Use bounded numeric sliders backed by text fields, with redraw on release/Apply to avoid expensive plotting on every movement. The calibrated overlay and component labels follow the later FR-27 revision. Mask exactly zero histogram bins before applying the colour normalization so 0.1 produces white empties and positive counts retain the lower colour. Keep ROI/DOI/paired-energy cuts and report masks in the same units, and explicitly mark keV-dependent findings unavailable in raw reports. Add synthetic checks on both mappings and a hidden-GUI slider/plot check. Calibration switching and Cornell slab behavior follow the revised single-ingest plan below.

## Fit display clarification (FR-20, FR-27)

The automatic `fit_peak` model already includes a positive local constant/linear background. Its Gaussian component alone cannot be expected to follow the total histogram on Cornell data with substantial underlying continuum. Draw `fit_on_display_bins(auto)["total"]` as the default red photopeak model and call it Gaussian + baseline, displaying its supported keV centroid and FWHM resolution. The opt-in control exposes the Gaussian component and estimated background as distinct dashed/dotted lines and evaluates the separately adjustable linear fit. If the manual linear settings reproduce the automatic linear fit, label the equivalence rather than adding a second overlapping total curve; show the resolution of that same supported estimate. If settings differ, draw a contrasting manual total with *its own* centroid and resolution; report fit failure explicitly. Preserve the established pure fit and PDF numerical results. Verify the total curve against display histogram heights on the supplied Cornell prefix and synthetic fixture, and ensure the PDF shows the same total-model curve; never pass off a background-subtracted component as fitted observed counts.

## Revised single-ingest calibration and PETsys presentation (FR-23–FR-26)

FR-23 supersedes the prior invalidation strategy: decode raw event sums and both calibration keys once, including Cornell slab keys, and retain them as bounded columnar arrays. Produce derived calibrated energies by grouping those keys with the selected `KevConverter` factor table; missing/zero factors become NaN so no fake measurement is shown. Build a new dataset view sharing raw arrays (do not mutate the visible dataset in a worker); switching off restores raw energies, and switching on/choosing a different calibration asynchronously creates the calibrated view. The original accepted-pair population, provenance and channel occupancy remain constant across switches; Cornell unresolved slabs are rejected at ingest in either mode. Keep selection/report units in active mode and reset energy cut fields to 0–300 a.u. or the selected keV default. GUI histogram axes use fixed 0–300 a.u. and 0–1500 keV x-ranges, DOI has a stable light-sharing x-range, and redraw replaces threshold markers without x autoscale. Give each plot its own aligned control group. Channel Status shows module-level counts, minimodule participation and mapped coverage with selected energy/DOI availability; do not add CMB-specific status metrics. Constrain the input card's geometry and truncate the displayed calibration name while retaining the full path in the UI. Re-run synthetic IMAS/Cornell, report, hidden GUI, and real Cornell prefix checks; human GUI acceptance remains open.

## Owner-requested correction (FR-16–FR-18)

Preserve automatic and manually adjustable fit modes as separately labelled measurements. Compare each fit to the exact display bin edges, rather than plotting model counts per 2.5-keV fit bin onto a 7.5-keV chart. For the automatic photopeak, prefer a flat continuum unless a linear one yields a material Poisson-deviance improvement (at least 8 for one extra parameter); the explicit experimental linear fit remains an adjustable comparison with residuals. Guard against wrong-peak and narrow-window width estimates; check both a known-width broad synthetic spectrum and real Cornell module spectra. Anchor popup windows to the inspector and test parent association. Attach status-specific, legible Treeview tags; keep text statuses for accessibility. The reproducible check is `scripts/ldat_issue_check.py` (quick fixture and `--real` on the provided Cornell acquisition).

## Checks

Deterministic synthetic LDAT fixtures built with the `read_compact.py` record layout; check exact pair vs detector counts, per-file provenance, masks, units, missing-channel semantics, empty/corrupt file states, Gaussian failures, fit overlays, report text and PDF page output. Compile-check the GUI; use mapped synthetic IMAS and real Cornell prefixes. The owner waived real-IMAS acquisition and manual PDF-layout review for this release. Do not run an interactive window as a side effect or imply those waived checks passed.
