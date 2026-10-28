# Spec 013 — IMAS NEMA NU 2-2018 CTR on the NECR phantom

Status: `approved` (2026-10-28; clarified the same day). [Plan](plan.md), [tasks](tasks.md).

Constitution: [AGENTS.md](../../AGENTS.md). Workflow: [docs/prompts.md](../../docs/prompts.md).
Builds on two shipped specs:
- [spec 011](../011-imas-skew-calibration/spec.md): EM skew LUT, the point-source CTR tool `src/imas/skew/ctr.py`, its selection and energy calibration;
- [spec 012](../012-imas-timing-walk/spec.md): the EM2 skew + walk3 pair and `correct_dt`.

Reference (untracked):
- `scripts_imas/imas_NEMA_CTR_multiprocess.py`, the legacy NEMA timing-resolution script;
- `scripts_imas/NECR_line_determination.py` (also in `docs/imas/CTR_NECR_legacy/`);
- the legacy results in `docs/imas/CTR_NECR_legacy/` (`20241003_NECR_results.txt`, `CTR_vs_activity_NECR.xlsx`, `CTR_NEMA.pptx`).

## Goal and population

Report the IMAS system coincidence time resolution following the NEMA NU 2-2018 procedure the legacy script implements, on the NECR (count-rate) phantom. Report it at several activity concentrations, with the NEMA value at 5.3 kBq/ml. Compare four corrections on the same events:
- none;
- the legacy skew `skew_IMAS.txt`;
- the spec 011 EM skew;
- the spec 012 EM2 skew + walk3.

This is a measurement with an explicit method, not a new calibration. No table is fitted on the NECR data.

**Input:**
- **Data:** PETsys compact **coincidence** LDAT, i.e. two detector hit populations per record, not singles.
  - Acquisition: `20241003_NECR_E0h4_120s_<frame>_coincCompact4s_<split>.ldat`, 3 Oct 2024. NECR phantom (20 cm diameter polyethylene cylinder, 70 cm long, F-18 line source about 45 mm off-axis), 30 frames of 120 s.
  - Owner transfer to `C:\Users\dsanchez\Desktop\data\IMAS\NECR\`: at least 1 GB of splits for frames 1, 5, 7, 10, 15, 20, 25 and 30.
  - On 2026-10-28 frames 7, 25, 26 and 30 are present. Frame 7 runs at about 6 M records/s, so its 3 GB split holds about 1.6 s.
- **Config and map:** `configs/imas_2DAQ.yaml`, `maps/imas2DAQ_map.yaml`. Checked on 100,000 records of frame 7: every hit maps, and all 120 SMs are seen.
- **Units:** timestamps in ps; hit charges in PETsys QDC units (a.u.); keV only through a recorded energy calibration; activity concentration in kBq/ml as given by the owner per frame (`CTR_vs_activity_NECR.xlsx`, column F; frame 7 = 5.31 kBq/ml).
- **Tables (frozen by SHA-256):**
  - legacy `IMAS/skew_IMAS.txt` (`930cd3fc…`);
  - EM `job_ship/lut_em.csv` (`8eadd8c7…`);
  - EM2 `job_walk/lut_em_walk-dd68b558604c.csv` (`daa3b251…`) with walk3 `walk/walk3.json` (`5e8ac010…`), applied only as a pair (spec 012 FR-4).
- **Hardware state:** the NECR data predate the skew runs (19 Nov 2025) by 13 months. The tables held 11 months after them (22Na, Oct 2026); whether they hold before is part of what this spec measures (FR-6).

**Legacy reference results** (legacy skew of Jan 2025, `.lm` input; FWHM with / without background subtraction):
- frame 1: 940 / 905 ps;
- frame 7 (5.31 kBq/ml): **768 / 744 ps**;
- frame 30: 592 / 571 ps.

## Requirements

- **FR-1 — NECR CTR file.**
  - One JSON file declares:
    - the acquisitions per frame (LDAT splits) and each frame's activity concentration with its source;
    - the tables and their SHA-256;
    - the energy calibration;
    - the line-source definition (fitted, FR-3, or declared P1/P2);
    - the output path and the options.
  - Inputs are frozen by SHA-256 as in spec 011's CTR file. Existing outputs are never overwritten.
- **FR-2 — Selection.**
  - Per side as spec 011's production selection: the highest-energy minimodule and its highest-charge TIME hit; hits below `en_min_ch` dropped; both sides 408–613 keV; `|dt|` inside the coincidence window.
  - Endpoints from the spec 011 IMAS geometry (map SM centres, 7 mm depth, frame signs `s_t +1, s_a −1`).
  - No opening-angle cut, as in spec 011's CTR: the line source spans the axial field of view.
  - Every cut is counted per frame, including events without a delay or a walk input. Events count as "same events" only when every evaluated table applies to them.
- **FR-3 — Line-source position.**
  - **Two lines** are evaluated on the same events:
    - the **fitted** line, which gives the NEMA value;
    - the **legacy image** line, `P1 (−37.18, 18.36, −356.11)` and `P2 (−36.60, 11.73, 356.11)` mm, used as given. It was found in the reconstructed image's frame, which may differ from the map frame by a few mm (legacy SM centre 48 mm against the map's 51.61 mm). The report gives its CTR as a sensitivity, with the distance between the two lines along z.
  - **The fit:** the 3D line that a frame's LORs converge to, without timing, in the map frame. It is seeded and robust to the scanner-wide background (LYSO, randoms, phantom scatter), as spec 011's `seeded_converge_point` is for points:
    1. **Seed:** a grid search over the bore for the axis-parallel line `(x0, y0)` that the most LORs pass within a few mm of.
    2. **Per axial slab:** each LOR is assigned to the z where it passes closest to the seed line. In each slab (about 20 mm), the LORs' convergence point is found by robust least squares, as for the point sources.
    3. **Straight line:** a line is fitted through the slab points (principal axis, as `NECR_line_determination.py` does with image points). The spread of the slab points about it tests straightness.
    4. **Refinement:** the four line parameters (`x0, y0` and the two slopes per mm of z) minimise a robust sum of LOR-to-line distances over the LORs within a shrinking window.
  - **Report:** the line per frame and pooled, the median LOR miss, the distance from the scanner axis, and the slab points against the line.
  - A declared line (P1/P2 in the map frame) can replace the fit.
- **FR-4 — NEMA analysis (port of the legacy method).** Per event:
  - **r:** the signed shortest distance between the LOR and the line.
  - **I:** the point on the LOR closest to the line.
  - **Timing error:** `Δt = dt_corrected − (|L1 − I| − |L2 − I|) / c`, with c = 0.299792 mm/ps.
  - Events with `|r| ≤ 20 + 2.0644 mm` and `|Δt| ≤ 2 ns` are kept; the histogram in r spans ±23 mm in 150 bins, and the signal band is `|r| ≤ 20 mm`.
  - The width without background subtraction uses every kept event (`|r| ≤ 22.0644 mm`), as the legacy script does (plan, 2026-10-28).
  - **Histogram:** 100 bins of 40 ps over ±2 ns.
  - **Background per time bin:** the counts in the r bins nearest ±20 mm, linearly interpolated across the band and summed over its bins.
  - **Width:** FWHM and FWTM from a parabola through the peak bin and its two neighbours, with linear interpolation at half and tenth of the parabola's maximum.
  - Reported with and without background subtraction, with the peak position.
  - **Parity:** given the same `(Δt, r)` arrays, the port reproduces the legacy script's histogram, background and widths. The expected values are a golden file computed once by the untracked script, with provenance.
- **FR-5 — Tables, uncertainty and our width.**
  - **Tables:** on the same events per frame: uncorrected, legacy, EM, EM2 + walk3.
  - **Uncertainty:** each FR-4 width has a bootstrap uncertainty. The differences against legacy are paired on the same events.
  - **Second width:** spec 011's model-free FWHM (10 ps bins, flat background within ±3 ns, local quadratic crossings) is reported beside the NEMA width.
- **FR-6 — NEMA report and acceptance.**
  - **Per frame:** CTR against activity concentration for every table. The NEMA value is the background-subtracted FWHM at 5.3 kBq/ml (frame 7).
  - **Transfer check:** on frame 7, EM must beat legacy by more than their combined uncertainty (spec 011 FR-10 rule). Otherwise the 2025 tables do not transfer to the 2024 state; the report says so and gives no EM or EM2 + walk3 NEMA value.
  - **Acceptance:** EM2 + walk3 against EM on frame 7, with the same rule. Pooled over frames and per frame, reported.
  - **Legacy check:** the legacy (`skew_IMAS.txt`) values are reported beside the Jan 2025 `.lm` results. They are not expected to be equal: the table, geometry and position estimator differ.
- **FR-7 — Energy calibration.**
  - **Source:** the bar run 7 calibration (owner, 2026-10-28; see Spec changes). Spec 011 FR-4's calibration from NECR data was the first choice; its coverage was too low.
  - **Rate check:** the 511 keV photopeak position per frame is reported, against a gain shift with rate.
- **FR-8 — Diagnostics.**
  - **Per frame and table:** the timing histogram with and without background subtraction, and the `Δt` against r 2D histogram (as the legacy plots).
  - **Across frames:** CTR against activity, all tables; the line fit; the per-frame photopeak.
  - Diagnostics, not validation evidence.

## Completion criteria

- **Synthetic fixtures:**
  - a line source with known delays and walk: the fitted line is recovered within declared tolerance, and each table's `Δt` is centred with the expected width;
  - the FR-4 background subtraction recovers an injected peak under a flat or sloped background;
  - the parity golden file passes.
- **Existing behaviour:** the spec 011 and 012 point-source CTR tests and goldens pass unchanged.
- **Real data (Windows, this PC):**
  - the transferred frames run, with timings and memory recorded;
  - FR-6 is reported, with the transfer check and the acceptance verdict;
  - missing frames are listed, not invented.
- **Suites:** the required repository suites pass. Missing acquisitions are explicit skips naming `PETSYS_DATA_DIR`.

## Clarify answers (owner, 2026-10-28)

1. **Line position:** both lines are evaluated: the line fitted from the data and the legacy image P1/P2 (FR-3). The fitted line is in the tables' frame, so it gives the NEMA value; the image line is a sensitivity.
2. **Energy calibration:** from the NECR data (FR-7); changed to the run-7 bar calibration at T8 (Spec changes).
3. **NEMA value:** frame 7 (5.31 kBq/ml), background-subtracted FWHM, as measured (no interpolation).
4. **Gates:** FR-6's transfer check (EM against legacy) and acceptance (EM2 + walk3 against EM), both on frame 7 with the spec 011 FR-10 rule.

## Spec changes

- **2026-10-28, FR-7 calibration (owner, T8):** the NECR CTR uses the run-7 bar calibration (`job_walk/skew_20251119_run7/encal/encal.tsv`, the calibration walk3's keV axis was fitted with). The per-slab NECR calibration (`encal_necr_f7`: frame 7, 9,842 of 15,360 slabs fitted; `encal_necr_all`: every split present, 12,281) leaves too many slabs without a 511 keV peak, and the selection rejects their events. The photopeak per frame (FR-7 rate check) is still reported with the run-7 calibration.
- **2026-10-28, plan:** FR-4 states the legacy script's kept band (`|r| ≤ 22.0644 mm`) for the histogram without background subtraction; the summary said 20 mm.

## Out of scope (possible follow-up: method improvements)

These are reported later as sensitivities, not as the NEMA value:
- background from a wider edge strip;
- other bin widths;
- decay correction to the middle of each frame.

Also out of scope:
- **Rate-dependence diagnostics:** CTR against singles rate, walk inputs against rate, pile-up.
- **Calibration:** new skew or walk fits on NECR data.
- **Other paths:** the `.lm` listmode path, the IMAS listmode script, reconstruction, Manager or GUI actions.
- **Other NEMA quantities:** NECR, scatter fraction, sensitivity, spatial resolution.
