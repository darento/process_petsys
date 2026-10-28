# Plan 013 — IMAS NEMA NU 2-2018 CTR on the NECR phantom

Status: approved 2026-10-28; [spec 013](spec.md). Constitution: [AGENTS.md](../../AGENTS.md). [Ordered tasks](tasks.md).

## Inputs seen at planning (2026-10-28)

- **NECR LDAT** in `PETSYS_DATA_DIR/IMAS/NECR/` (`C:\Users\dsanchez\Desktop\data`), transfer in progress:

  | Frame | Splits present | Size | Activity (kBq/ml) |
  | --- | --- | --- | --- |
  | 7 | 24 | 3.07 GB | 5.311 |
  | 20 | 53 | 2.00 GB | 0.996 |
  | 25 | 22, 23 | 1.64 GB | 0.524 |
  | 26 | 22 (misclick, not in the set) | 0.32 GB | 0.461 |
  | 30 | 22, 23, 24 | 1.97 GB | 0.276 |

  Frames 1, 5, 10 and 15 are still to come. Activities are column F of `docs/imas/CTR_NECR_legacy/CTR_vs_activity_NECR.xlsx` (frame 1 11.514, 5 6.872, 10 3.608, 15 1.894 kBq/ml).
- **Legacy results** (same sheet; FWHM raw / background-subtracted): frame 1 940 / 905, 5 811 / 784, 7 768 / 744, 10 717 / 696, 15 654 / 635, 20 621 / 599, 25 603 / 591, 30 592 / 571 ps.
- **Tables:** as in spec 011/012's CTR file `IMAS_work/ctr_walk_4pos_fr7.json`:
  - `job_work` is `IMAS_work/job_walk`, which carries the ship job's fingerprint, so `em` (`job_ship/lut_em.csv`) and `em2` (`job_walk/lut_em_walk-dd68b558604c.csv`) both pass `frozen_inputs`;
  - walk3 is `IMAS_work/walk/walk3.json`;
  - legacy is `IMAS/skew_IMAS.txt`.
- **Legacy script** `scripts_imas/imas_NEMA_CTR_multiprocess.py` (read 2026-10-28). Details that the spec's FR-4 summary leaves out:
  - an event is kept when `|r| <= 20 + 2.0644 mm` and `|Δt| <= 2000 ps`;
  - the **raw** histogram ("w bg" in the results file) uses every kept event, so its band is `|r| <= 22.0644 mm`, not 20 mm;
  - the background-subtracted histogram ("wo bg") comes from `np.histogram2d(r, Δt, [150, 100], [[-23, 23], [-2000, 2000]])`. Signal is the r bins with centre in [−20, 20]. The edge columns are the bins whose centres are nearest ±20 mm, and are weighted `(20 − r)/40` and `(r + 20)/40` for each signal bin;
  - `find_widths` raises at a peak in an edge bin, and fails with `IndexError` when no bin on a side falls below the level;
  - `r = (L1 − P1)·(u × v) / |u × v|`, where `u` is the unit vector P1→P2 and `v` the unit vector L1→L2; `I = L1 + t v` is the closest point on the LOR;
  - the energy filter is commented out, because the `.lm` input was already energy-filtered.
- **Reusable code:**
  - `src/imas/skew/ctr.py`: `frozen_inputs`, `_frozen`, `WalkTable`, `geometries`, `paired_differences`, `_judge`;
  - `src/imas/skew/select.py`: `feature_batches`, `Selection.production`;
  - `src/imas/skew/walk.py`: `correct_features`;
  - `src/timing/ctr.py`: `fit_histogram`, `local_crossings`, `converge_point`, `_near_counts`;
  - `src/imas/skew/encal.py`: `accumulate`, `calibrate`, `run`;
  - `tests/imas_skew_helpers.py`: `side`, `write`, `channels`, `mu_file`, and the two-pass synthetic LDAT pattern of `tests/test_imas_skew_ctr.py:case`.
- **scipy 1.12.0** in the environment.
- **Disk:** C: about 200 GB free.

## Architecture

1. **`src/timing/nema.py` — NEMA timing-resolution core** (scanner-neutral; FR-4).
   - Constants as in the legacy script: `C_MM_PER_PS = 0.299792`, `R_SIGNAL_MM = 20`, `R_KEEP_MM = 22.0644`, `R_RANGE_MM = 23`, `R_BINS = 150`, `T_RANGE_PS = 2000`, `T_BINS = 100`.
   - `line_frame(p, q, origin, direction) -> (r, intercept, Δtof)`, vectorized. It uses the legacy formulas for r and I and returns `(|L1 − I| − |L2 − I|) / c`. LORs parallel to the line get NaN and are counted.
   - `NemaHistograms`: per table, a `(R_BINS, T_BINS)` count array for the band and a `T_BINS` raw array. It is filled per batch with `np.histogram2d` and `np.histogram` on the same ranges as the legacy script, so summing batches equals the legacy call on all events.
   - `background_subtracted(hist2d)`: the legacy loop, vectorized.
   - `nema_widths(hist, centers, fraction)`: the legacy `find_widths`. A peak in an edge bin or a missing crossing returns a status instead of raising.
2. **`src/timing/line.py` — line source from LOR convergence** (scanner-neutral; FR-3). It uses endpoints only.
   - `seed_line(p, q, radius_mm, steps_mm, near_mm, sample)`: grid search for the axis-parallel line `(x0, y0)` that most LORs pass within `near_mm` of. For an axis-parallel line, the 3D LOR distance equals the 2D distance from `(x0, y0)` to the LOR's xy projection, so the grid is 2D, coarse then fine, as in `seeded_converge_point`.
   - `slab_points(p, q, seed, slab_mm, near_mm)`: each LOR within `near_mm` of the seed is assigned to the z of its closest approach to the seed. Per slab, the robust 2D convergence point is found as in `converge_point`: least squares on the xy projections, refitted on the LORs within max(median miss, floor), with z the median closest-approach z.
   - `principal_line(points)`: principal axis through the slab points, weighted by LOR count, with the spread about it.
   - `refine_line(p, q, line, windows_mm=(20, 10, 5))`: four parameters `x = x0 + ax z`, `y = y0 + ay z`. Each window minimises a `soft_l1` sum of LOR-to-line distances (`scipy.optimize.least_squares`) over the LORs within that window of the current line.
   - `locate_line(...)` chains the four steps. It returns the line (point at z = 0, unit direction), the slab points, the median miss, the LORs used, the seed and the step distances.
3. **`src/imas/skew/necr.py` — NECR CTR** (IMAS; FR-1, FR-2, FR-5–FR-7), with the command `python -m src.imas.skew necr FILE`.
   - **Loading.** `load_necr(path)` reads schema `imas-necr-ctr/1`. It shares the CTR file's frozen keys (`job_work`, `encal`, `luts`, `legacy`, `walks`, `walk_acceptance`) and adds:
     - `frames: [{name, frame, ldat: [..], activity_kbq_ml, activity_source}]`;
     - `line: {"fitted": {...options}, "image": {P1, P2}}`;
     - an optional `legacy_results: {frame: {fwhm_raw_ps, fwhm_bg_ps, fwtm_raw_ps, fwtm_bg_ps}, source}`;
     - `gates: {nema_frame, transfer: [ref, cand], acceptance: [ref, cand]}`;
     - options `bootstrap`, `seed`, `max_records`, `localization_events`, `model_free_window_ps` (3000), `model_free_bin_ps` (10).
   - **Frozen inputs.** `ctr.frozen_inputs` takes the loaded file directly. Its fields keep the CTR file's names, so it is reused unchanged.
   - **Passes.**
     1. **Localization:** the first `localization_events` valid LORs per frame. `locate_line` runs per frame and pooled.
     2. **Accumulation:** per frame and batch, the `Accumulator` corrects `dt` for every table (`WalkTable` through `correct_features(policy="drop")`), as `ctr.Accumulator.add` does. It applies the same-events mask, then fills the NEMA histograms for both lines and the model-free histogram (|r| ≤ 20 mm, 10 ps bins over ±3 ns). Bootstrap replicates use one Poisson weight per event, shared by tables and lines.
     3. **Energy check (FR-7):** `encal.accumulate` per frame (bounded by `max_records`). Each channel's a.u. histogram is moved to keV with the frame's `mu` and summed, and the peak is fitted. The result is the frame's photopeak in keV.
   - **Report.** Results go in `necr_ctr.json` per frame, pooled, table and line:
     - NEMA FWHM/FWTM raw and background-subtracted, with the peak position;
     - the model-free FWHM;
     - bootstrap standard deviations and paired differences;
     - the gates (`_judge`, spec 011 FR-10 rule);
     - counts, lines, photopeaks, the legacy results beside ours, timings and peak memory.
   - Histograms go in `necr_histograms.npz` and plots in `plots/`.
4. **`src/imas/skew/necrplots.py` — FR-8 diagnostics.** Per frame and table: the raw and background-subtracted timing histograms, and Δt against r. Across frames: CTR against activity for all tables, with the legacy `.lm` values; the line fit (slab points against the line, x(z) and y(z)); the photopeak per frame.

### Decisions

- **Headline line.** The pooled fitted line is applied to every frame. The phantom did not move between frames, and one line keeps frames comparable. Per-frame lines are reported. A frame whose line is more than 1 mm from the pooled line inside |z| ≤ 350 mm is flagged; it is not refitted.
- **The image line** (P1/P2) is evaluated on the same events as the fitted line, as a sensitivity (spec FR-3). The report gives the distance between the two lines at z = −350, 0 and +350 mm.
- **Same events:** an event counts when both TIME channels have an applied delay in every table and both sides have every walk input (spec 011/012 rule). Each line's band is applied after this mask.
- **Bins as in the legacy script,** so NEMA widths can be compared with the `.lm` results. The model-free width is the method-independent cross-check.
- **Gate value:** the NEMA background-subtracted FWHM on frame 7 with the fitted line. Its uncertainty is the bootstrap standard deviation. "Combined" means quadrature, as spec 011 FR-10 (conservative for shared events); the paired difference is reported beside it.
- **Transfer failure:** when EM does not beat legacy on frame 7, the report gives no NEMA value for `em` or `em2_walk3` and states why. Their histograms are still written as diagnostics.
- **Energy calibration:** `python -m src.imas.skew encal` on the highest-activity transferred frame's splits, as a separate step whose output is frozen in the NECR file. If frame 1 has not arrived, frame 5 or 7 is used and the report records which. The walk's keV input uses this calibration: keV is physical, so the walk3 curves still apply.
- **Memory:** histograms only. The bootstrap arrays (tables × replicates × 150 × 100 float64) take about 96 MB per line and frame for 4 tables and 200 replicates. Frames are reduced to widths and written one at a time. Localization holds `localization_events` × 6 float64 per frame (2 M events ≈ 96 MB).

### Alternatives rejected

- **Extending `src/imas/skew/ctr.py` with a line-source mode:** its accumulator is built around point-source variants (source ± σ, depth, LOR distance) and TOF bins, which have no NEMA meaning. A separate module that reuses the frozen inputs, correction and gates is smaller and leaves spec 011/012 outputs unchanged.
- **3D per-slab points (`converge_point`):** a line segment inside a slab makes the axial coordinate poorly conditioned. The 2D projection is exact for an axis-parallel line, and the refinement removes the slope.
- **Fitting the line per frame for each frame's NEMA value:** this adds frame-to-frame line noise to the CTR comparison. Kept as a check only.
- **Reading `.lm` files:** they lack channel IDs and charges, so skew and walk cannot be reapplied (spec: out of scope).

## FR → component

| FR | Component | Test |
| --- | --- | --- |
| FR-1 | `necr.load_necr`, `ctr.frozen_inputs` | `tests/test_imas_necr.py` |
| FR-2 | `necr.Accumulator` (counts, same events) | `tests/test_imas_necr.py` |
| FR-3 | `src/timing/line.py`, `necr` localization | `tests/test_timing_line.py`, `tests/test_imas_necr.py` |
| FR-4 | `src/timing/nema.py` | `tests/test_timing_nema.py` (parity golden) |
| FR-5 | `necr` summary: bootstrap, paired, model-free | `tests/test_imas_necr.py` |
| FR-6 | `necr` gates and report | `tests/test_imas_necr.py`, real data |
| FR-7 | `encal` on NECR, `necr` photopeak | `tests/test_imas_necr.py`, real data |
| FR-8 | `necrplots` | `tests/test_imas_necr.py` (files written) |

## Parity golden (FR-4)

- **Capture.** The local, untracked script `scripts/capture_013_t1.py` imports `scripts_imas/imas_NEMA_CTR_multiprocess.py` and runs its functions on a fixture from `tests/necr_helpers.py`:
  - `compute_distance_r`, `closest_point_on_line_source` and `compute_timing_error` on 500 seeded LORs and the legacy line;
  - `background_correction`, `np.histogram` and `find_widths` (fraction 0.5 and 0.1) on seeded `(Δt, r)` arrays: 200,000 events, a 300 ps σ peak in the band plus a flat r × Δt background.
- **Output.** `tests/data/golden/timing/013_t1_nema_legacy.json`, with a provenance record (requirement `013-FR-4`; sources: the legacy script, the fixture module, the capture script, each with SHA-256; truth "legacy script output, not physical truth").
- **Provenance test.** `tests/test_golden_provenance.py` accepts `013-FR-4` records under the same rules as `011-FR-1` and `012-FR-2`.

## Synthetic line-source fixture

`tests/necr_helpers.py` builds the synthetic data and is shared by the line, NEMA and NECR tests:
- **Line:** tilted (slopes 0.01 and −0.005, at (−37, 15) mm at z = 0).
- **Events:**
  - random SM–minimodule pairs, kept by rejection when their nominal LOR passes within 25 mm of the line;
  - flat background pairs (20 %);
  - `dt` from the TOF of a point drawn on the line near the LOR, plus injected per-channel delays, a walk offset (from charges written into the hits), and 200 ps σ noise.
- **Files:** written as IMAS compact LDAT, with the two-pass pattern of `test_imas_skew_ctr.py:case`.
- **Frozen inputs:** a job fingerprint, an EM LUT, a worse legacy file and a walk LUT, built as the spec 011/012 CTR tests do.
