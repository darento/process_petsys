# Tasks — IMAS NEMA NU 2-2018 CTR on the NECR phantom

Status: open, 2026-10-28. [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `013-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("013-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially. Run new and changed tests with `-n 0 --slow-limit 5`.
- `real_data` tests name acquisitions relative to `PETSYS_DATA_DIR` (`IMAS/NECR/...`, `IMAS_work/...`) and skip naming the missing variable or file.
- Record each check's selection, reason, command, platform and pytest summary line as `Verified:`. Tick a task only on success.
- No commits, pushes or release builds; the owner runs git.

## Core

- [x] **T1 — NEMA core and parity golden** (FR-4).
  - **Code:** `src/timing/nema.py`: constants, `line_frame`, `NemaHistograms` (batch fill), `background_subtracted`, `nema_widths`.
  - **Fixture:** `tests/necr_helpers.py`: seeded LORs and `(Δt, r)` arrays.
  - **Golden:** the local `scripts/capture_013_t1.py` writes `tests/data/golden/timing/013_t1_nema_legacy.json` from the legacy script, with provenance. `tests/test_golden_provenance.py` accepts `013-FR-4`.
  - **Done when** `tests/test_timing_nema.py tests/test_golden_provenance.py -n 0 --slow-limit 5` pass:
    - **Geometry:** `line_frame` r, I and Δtof equal the golden within 1e-9 mm or ps.
    - **Histograms:** filling in 7 uneven batches gives the golden raw and 2D histograms exactly; the background-subtracted histogram matches within 1e-9.
    - **Widths:** FWHM, FWTM and peak equal the golden within 1e-9 ps.
    - **Failure cases:** a peak in an edge bin, and a side with no crossing, each give a status, never an exception or a number.
    - **Background recovery:** on a flat background and on a sloped one (linear in r), the subtraction recovers an injected peak's counts within 3 % and its FWHM within 2 %.
  - Deviations:
    - `nema_width` is the legacy `find_widths`; `widths` gives FWHM, FWTM and peak with a status (`peak at edge`, `no crossing`; `no parabola peak` guards NaN input only, since the first maximum always has a lower left neighbour). `NemaHistograms.add` takes Poisson weights for the bootstrap and an index for tables/frames.
    - Speed of light is the legacy 0.299792 mm/ps (`src.timing.shell` uses 0.299792458; 5e-9 relative).
    - **Background recovery** is checked two ways:
      - on expected counts: a flat background is removed exactly (1e-9); a sloped one (±80 % over ±30 mm) leaves at most 0.2 % of the background, because the legacy edge bins' centres sit 0.087 mm outside ±20 mm;
      - on seeded events, n = 2,000,000 (seeds 200 flat, 201 sloped): recovered counts within 3 % and FWHM within 2 %.
    - At the golden's 200,000 events, the legacy single-bin edges make the background-subtracted FWHM noisy: 668–719 ps over 6 seeds, against 700 ps peak-only. This is method noise, not a bias. It matters for the uncertainty (bootstrap, FR-5).
    - Golden `tests/data/golden/timing/013_t1_nema_legacy.json`:
      - written by local `scripts/capture_013_t1.py`, importing `scripts_imas/imas_NEMA_CTR_multiprocess.py`;
      - fixture `tests/necr_helpers.py` (`line_lors` seed 13 n 500; `dt_r_events` seed 17 n 200,000, cut as `ctr_correction_per_event`);
      - port and legacy agree exactly: raw 743.34 / 1558.70 ps, background-subtracted 667.98 / 1311.68 ps on that fixture.
    - `test_golden_provenance.py` accepts `013-FR-4` records under the `011-FR-1` / `012-FR-2` rules.
  - Verified 2026-10-28 (Windows, this PC). Selection: the new module and its tests plus the provenance test, which was changed; no existing workflow imports `src.timing.nema`.
    - `tests/test_timing_nema.py tests/test_golden_provenance.py -n 0 --slow-limit 5`: 52 passed in 3.32 s.
    - `py_compile src/timing/nema.py` OK.
- [x] **T2 — Line from LOR convergence** (FR-3).
  - **Code:** `src/timing/line.py` (`seed_line`, `slab_points`, `principal_line`, `refine_line`, `locate_line`).
  - **Done when** `tests/test_timing_line.py -n 0 --slow-limit 5` passes. The fixture is a tilted line plus 30 % background LORs (flat over the bore), with 2 mm endpoint blur:
    - **Recovery:** the fitted line is within 0.5 mm of the true line for |z| ≤ 350 mm.
    - **Steps:** the seed is within 3 mm, and the slab points lie within 1 mm of the line (median).
    - **Axis-parallel line:** an axis-parallel line is recovered with slopes below 1e-3.
    - **Errors:** fewer than three LORs per slab gives an empty slab, counted. All slabs empty, or parallel LORs, give a `CtrError`, never a line.
  - Deviations:
    - Defaults: seed grid 10 / 2 mm, 10 mm near (5 mm on the fine grid), 5,000 sampled LORs; slab near 15 mm, floor 3 mm. A LOR whose xy projection is shorter than 1 mm (parallel to z) counts by its xy midpoint in the seed and is left out of slabs. LORs within sine 0.1 of the line are left out of the refit (`soft_l1`, f_scale 3 mm).
    - `locate_line` refuses fewer than two populated slabs (a line needs two points).
    - Fixtures `tests/line_helpers.py` (seeds 1-11): cylinder r 410 mm, |z| ≤ 380 mm, emission |z| ≤ 350 mm, 2 mm endpoint blur, 30 % flat background.
    - Measured:

      | Fixture | Line error (\|z\| ≤ 350) | Seed error | Slab-point median | Slopes |
      | --- | --- | --- | --- | --- |
      | tilted (114k LORs) | 0.018 mm | 1.4 mm | 0.051 mm (35 slabs, 0 empty) | 0.010015, −0.004979 |
      | axis-parallel (86k LORs) | 0.041 mm | — | 0.064 mm | 2.2e-5, 9.4e-5 |

      Median miss 0.95 mm; `locate_line` 0.5 s.
  - Verified 2026-10-28 (Windows, this PC). Selection: the new module and its tests; nothing imports it yet.
    - `tests/test_timing_line.py -n 0 --slow-limit 5`: 10 passed in 2.10 s.
    - `py_compile src/timing/line.py` OK.

## IMAS NECR tool

- [x] **T3 — NECR file and frozen inputs** (FR-1).
  - **Code:** `src/imas/skew/necr.py:load_necr`; `ctr.frozen_inputs` reused.
  - **Done when** `tests/test_imas_necr.py -k "file or frozen" -n 0 --slow-limit 5` passes:
    - **Paths:** a valid file loads, with paths relative to it.
    - **Refusals:** each is a `CtrError` naming the key:
      - a missing or extra key;
      - a frame without LDAT;
      - a negative or missing activity;
      - an image line with P1 = P2;
      - a gate naming an unknown table;
      - a changed SHA-256 (LUT, walk, legacy, encal);
      - a walk paired with the wrong skew.
    - **Outputs:** an existing output directory is refused.
  - Deviations:
    - File keys: `frames` (`name`, `frame`, `ldat` list, `activity_kbq_ml`, `activity_source`), `lines` (`{method: "lor_convergence", ...options}` or `{p1, p2, source}`), `headline_line`, optional `gates` (`frame`, `transfer`, `acceptance`), `legacy_results`, `bootstrap`, `seed`, `max_records`, `localization_events`, `energy_check_records`, `model_free_window_ps`, `model_free_bin_ps`. `output` is a directory.
    - `load_necr` also exposes the splits as `acquisitions` and the gate pairs as `walk_acceptance`, so `ctr.frozen_inputs` checks, unchanged, that every split exists and is not a job input and that gate tables are evaluated (a skew LUT solved with walk only inside its walk table). `prepare_output` creates the directory and refuses an existing one.
    - Fixtures: `tests/necr_helpers.py:frozen_tables` (job record, encal, EM LUT, walk LUT of `charge_walk`, legacy file with 100 ps errors) and `necr_file`.
  - Verified 2026-10-28 (Windows, this PC). Selection: the new module and its tests; `ctr.py` unchanged.
    - `tests/test_imas_necr.py -n 0 --slow-limit 5`: 19 passed in 2.60 s (15 refusals parametrized; frozen SHA-256 of LUT, walk, legacy, encal; wrong-skew walk; missing split; job split; em2 without its walk).
    - `py_compile src/imas/skew/necr.py` OK.
- [x] **T4 — Accumulation on a synthetic NECR acquisition** (FR-2, FR-3, FR-4, FR-5).
  - **Code:** `necr.Accumulator`, the localization pass, `run_necr`; `tests/necr_helpers.py` synthetic LDAT line source (plan § Synthetic fixture), with injected delays and walk.
  - **Done when** `tests/test_imas_necr.py -k accumulate -n 0 --slow-limit 5` passes:
    - **Line:** the pooled fitted line is within 1 mm of the injected line.
    - **Centring:** every table's background-subtracted Δt is centred within 2 bootstrap SD of 0.
    - **Ordering:** the FWHM orders as uncorrected > legacy > EM > EM + walk.
    - **Counts:** per frame, the cut counts sum to the records. Events without a delay or without a walk input are counted and excluded from every table.
    - **Bootstrap:** replicates are reproducible with the same seed.
    - **Image line:** a declared line offset by 3 mm gives a wider NEMA FWHM than the fitted line, on the same events.
  - Deviations:
    - **Synthetic acquisition** (`tests/necr_helpers.py`):
      - `channel_endpoints` gives the nominal endpoint of every one of the 15,360 TIME channels, from one selected record each;
      - `necr_frame` draws random channel pairs and keeps them as signal with probability exp(−r²/2·2²) when their closest point to the tilted line (−37, 15 mm; slopes 0.01, −0.005) lies within |z| ≤ 350 mm, and as background when |r| < 40 mm;
      - signal dt = line TOF + true delays + `charge_walk` per side + 150 ps noise; background dt is flat over ±3 ns;
      - two frames of 8,500 records; the legacy file has no delay (9999) for SM 50.
    - **Design changes:**
      - `NemaHistograms` keeps only the r × Δt counts (the raw histogram is their sum over r, exactly, since every kept event lies inside ±23 mm); `nema.bin_index` is shared by the nominal fill and the bootstrap replicates, and equals `np.histogram2d` at bin edges (new test).
      - Bootstrap arrays are float32, held for the current frame and summed over frames; the seed is `[seed, frame, split, first record]`, so splits of one frame never share weights.
      - `max_records` applies per split; `localization_events` defaults to 1,000,000.
    - **Centring** is checked on the model-free fit's Gaussian mean: within 5 standard errors (σ/√signal) + 5 ps of 0 for legacy, EM and EM + walk. The NEMA peak position has no bootstrap SD.
    - **Ordering:**
      - legacy > EM > EM + walk is checked on the background-subtracted NEMA FWHM, pooled: 505 > 406 > 359 ps. EM + walk is within 10 % of 2.355 × 150 ps.
      - Uncorrected is checked on the raw NEMA and model-free FWHMs (1,666 and 1,768 ps). Its background-subtracted NEMA width is unstable (251–1,513 ps across frames and lines): its ~1.7 ns peak fills the ±2 ns window, so the edge bins hold signal.
    - **Image line (criterion replaced):** a 3 mm line offset moves the expected TOF by at most ~20 ps, which is not measurable against the bootstrap SD. The test instead checks that:
      - the image line is evaluated on the same events;
      - the reported line distances equal the geometric ones.

      Measured: fitted vs image 3.6, 0.08 and 3.7 mm at z −350, 0 and +350 mm; FWHMs equal within noise.
    - **Line:** the pooled fitted line is within 0.25 mm of the injected line (endpoint quantization); median LOR miss 1.34 mm.
    - **Counts:** pre-geometry labels sum to the records; same events + invalid endpoint + no applied delay + walk input missing = passed pre-geometry. The "walk input missing" path is not reachable with selected synthetic sides (no TIME or ENERGY charge fails the selection first); spec 012's `correct_features` tests cover it.
  - Verified 2026-10-28 (Windows, this PC): see T5.
- [x] **T5 — Summary, gates and report** (FR-5, FR-6).
  - **Code:** the NEMA and model-free widths per frame, pooled, table and line; bootstrap SD; paired differences against legacy; the transfer check and acceptance (`_judge`); the legacy `.lm` results beside ours; and `necr_ctr.json`, `necr_histograms.npz`.
  - **Done when** `tests/test_imas_necr.py -k "report or gate" -n 0 --slow-limit 5` passes:
    - **Fields:** the report has every field of plan § Architecture 3.
    - **Gates:** they are evaluated on the `gates.nema_frame` frame.
    - **Transfer failure:** a fixture where EM is not better than legacy gives `transfer: failed`, no `nema_value` for `em` or `em2_walk3`, and still writes their histograms.
    - **Pass case:** a passing fixture reports both verdicts with their differences and uncertainties.
    - **Model-free width:** equals `core.fit_histogram(..., crossings=local_crossings)` on the same histogram.
  - Deviations:
    - **Gate frame and judge:** the gate frame is `gates.frame`. `judge` applies the spec 011 FR-10 rule to the background-subtracted NEMA FWHM of the headline line (quadrature of the bootstrap SDs), with the paired same-event SD beside it.
    - **Paired differences** against legacy are reported per scope, line and table for the NEMA background-subtracted, NEMA raw and model-free FWHMs.
    - **Transfer failure:** the `nema_value` entries of every frozen table are `withheld: ...`.
    - **Synthetic gate results** (frame 1, 20 replicates):
      - transfer: EM 394 vs legacy 501 ps (combined 54, paired SD 56);
      - acceptance: EM + walk 348 vs EM 394 ps (combined 15.6, paired 17.0).
    - **Transfer-failure fixture:** EM LUT with 300 ps errors.
    - **Run time:** synthetic run 7.2 s, mostly model-free bootstrap fits (about 12 ms each).
    - **Projection for real data:** 200 replicates × 2 lines × 4 tables ≈ 1,600 fits per scope, about 20 s.
  - Verified 2026-10-28 (Windows, this PC). Selection: the new module and its tests (T3-T5), with slow tests; no existing workflow imports `necr`.
    - `tests/test_imas_necr.py -m "not real_data" -n 0 --slow-limit 5`: 24 passed in 27.45 s. The run fixture (10.6 s setup), reproducibility and transfer-failure runs (7.2 s each) are marked `slow`.
    - `tests/test_timing_nema.py tests/test_timing_line.py -n 0 --slow-limit 5`: 11 and 10 passed, after the bin-index change and line wrapping.
    - `py_compile src/imas/skew/necr.py` OK.
- [x] **T6 — Energy check** (FR-7).
  - **Code:** the photopeak per frame (`encal.accumulate` → keV histogram → peak fit), the calibration recorded in the report, and the fallback recorded when the NECR calibration is missing.
  - **Done when** `tests/test_imas_necr.py -k photopeak -n 0 --slow-limit 5` passes:
    - synthetic frames with a 0 % and a 5 % gain shift report their photopeaks at 511 and 536.6 keV (±1 %);
    - a channel without `mu` is excluded and counted.
  - Deviations:
    - `necr.photopeak` moves each channel's `encal.accumulate` histogram (0.1 a.u. bins, 0-220 a.u.) to keV with `511 / mu` and fits the unbinned per-channel keV values with `fit_peak_background` (Gaussian + linear background, 2 keV fit bins) around the highest smoothed 2 keV bin in 350-800 keV. Its start needs bins several times narrower than the peak σ: with 100 bins over 0.55-1.5 × anchor, a narrow peak's start area exceeded its bound ("x0 is infeasible").
    - `energy_check` runs on the first `energy_check_records` (default 2,000,000) records of each split. The report's `energy` block holds the calibration record (`frozen_inputs` origin: job or another encal output with its `encal.json`) and the photopeak per frame. The fallback calibration is whichever encal the NECR file declares; the report says which.
    - Synthetic ENERGY sums are scaled to IMAS-like ~102 a.u. (`mu` 102.2), inside the encal range. The NECR run fixture's sums (511 a.u.) fall outside it, so its photopeak is `few entries`, as reported.
  - Verified 2026-10-28 (Windows, this PC): see T7.
- [x] **T7 — Diagnostics and CLI** (FR-8).
  - **Code:** `src/imas/skew/necrplots.py`; `python -m src.imas.skew necr FILE` in `__main__.py`.
  - **Done when:**
    - `tests/test_imas_necr.py -k plots -n 0 --slow-limit 5` passes: every FR-8 PNG is written for the synthetic run, and the CLI writes the report;
    - `py_compile` succeeds on the new modules;
    - the affected spec 011/012 CTR tests pass unchanged: `tests/test_imas_skew_ctr.py tests/test_imas_walk.py tests/test_timing_ctr_local.py -m "not real_data"`.

  - Deviations:
    - Plots are written by `run_necr` into `<output>/plots/`, after the histograms and before the report, which lists them under `plots`:
      - per frame: `timing_<frame>.png` and `dt_vs_r_<frame>.png`;
      - across frames: `ctr_vs_activity.png` (with the legacy `.lm` results), `line_fit.png`, `photopeak.png`.
    - The command is `python -m src.imas.skew necr FILE`. It prints the NEMA value and the gate verdicts, and returns 2 when refused (e.g. an existing output).
  - Verified 2026-10-28 (Windows, this PC). Selection:
    - the new modules and their tests (T3-T7), with slow tests;
    - the spec 011/012 CTR consumers of `ctr.py`, `walk.py` and `timing/ctr.py`, which are unchanged (`necr` only imports them).
    - Commands:
      - `py_compile` of `necr.py`, `necrplots.py`, `__main__.py`, `timing/nema.py`, `timing/line.py`: OK.
      - `tests/test_imas_skew_ctr.py tests/test_imas_walk.py tests/test_timing_ctr_local.py -m "not real_data"`: 24 passed in 52 s.
      - `tests/test_imas_necr.py tests/test_timing_nema.py tests/test_timing_line.py tests/test_golden_provenance.py -m "not real_data" -n 0 --slow-limit 5`: 90 passed in 71.8 s.
    - Plots were inspected on the synthetic run: the timing histograms and the line fit are as expected.

## Real data

- [x] **T8 — NECR energy calibration and frame 7 run** (FR-3, FR-7).
  - **Steps:**
    1. Run `python -m src.imas.skew encal` on the highest-activity frame present; record which frame was used and why.
    2. Write `IMAS_work/necr_frame7.json` with frame 7 only and the four tables.
    3. Run it.
  - **Done when:**
    - the run completes, with records, timings and peak memory recorded here;
    - the fitted line and its distance to the image line are recorded;
    - the `tests/test_imas_necr.py` `real_data` test passes on frame 7 with `max_records` 2,000,000: the line lies within 10 mm of the image line at z = 0, and every table's NEMA FWHM is finite;
    - otherwise it skips naming `PETSYS_DATA_DIR` or the file.
  - Deviations:
    - **Energy calibration (owner, 2026-10-28):** the run-7 bar calibration `IMAS_work/job_walk/skew_20251119_run7/encal/encal.tsv` (14,546 slabs fitted) instead of a NECR calibration. The NECR calibrations left too many slabs without μ, and the selection rejects their events:
      - `IMAS_work/encal_necr_f7` (frame 7 split 24, 2.9 min): 9,842 fit / 2,202 few entries / 3,316 no peak;
      - `IMAS_work/encal_necr_all` (every NECR split present: frames 7, 20, 25, 26, 30; 2.3 min, 4 workers): 12,281 / 1,487 / 1,592.
      - On common slabs NECR μ / run-7 μ has median 0.969 (p5-p95 0.885-1.143). With the run-7 calibration the frame-7 photopeak is at 494.7 keV (FWHM 15.8 %), inside 408-613 keV. Spec FR-7 updated (Spec changes).
    - The real-data test expects no half-maximum crossing for uncorrected: its ~2.3 ns spread fills the ±2 ns NEMA window. It also checks EM2 + walk3 < legacy on the raw width.
    - The CLI prints each table's status beside its value.
  - **Frame 7 run** (`IMAS_work/necr_frame7.json` → `IMAS_work/necr_frame7/`): split 24 only (3.07 GB, 8,079,742 records), 200 replicates, run-7 calibration.
    - **Time and memory:** 73 s total (localization 13 s, read + accumulate 18.8 s, frame summary 17.8 s, pooled 17.3 s, energy check 3.2 s); peak working set 0.99 GB.
    - **Counts:** 650,421 passed pre-geometry; 518 without a delay; 0 missing a walk input; 649,903 same events. In the NEMA band (fitted line): legacy 113,951, EM 114,263, EM2 + walk3 114,938, uncorrected 107,713.
    - **Line:** (−35.53, 16.05) mm at z = 0, slopes 0.00043 / −0.00827; 39 mm from the axis; 0 empty slabs, slab spread median 0.43 mm. The image line is 1.63 / 1.69 / 1.84 mm away at z −350 / 0 / +350 mm.
    - **Widths (ps, fitted line, ± bootstrap SD):**

      | Table | NEMA raw | NEMA background-subtracted | Model-free |
      | --- | --- | --- | --- |
      | uncorrected | no crossing | no crossing | 2356 ± 36 |
      | legacy | 782 ± 11 | 771 ± 30 | 735 ± 6 |
      | EM | 729 ± 9 | 724 ± 27 | 694 ± 5 |
      | EM2 + walk3 | 693 ± 9 | 669 ± 36 | 668 ± 5 |

      Legacy `.lm` (Jan 2025): 768 raw / 744 background-subtracted.
    - **Gates (background-subtracted, frame 7):** both accepted.
      - Transfer: EM beats legacy by 47.3 ps against a combined uncertainty of 40.3 (paired SD 37.7).
      - Acceptance: EM2 + walk3 beats EM by 54.7 against 44.8 (paired 42.6).
    - **Paired against legacy** (raw / model-free): EM 52.6 ± 13.4 / 40.6 ± 6.1; EM2 + walk3 89.3 ± 13.9 / 67.1 ± 7.1.
    - **Image line:** legacy 757, EM 682, EM2 + walk3 636 ps (background-subtracted).
    - **Observation:** a secondary bump at about +1.5 to +2 ns in every corrected table (`plots/timing_frame7.png`).
  - Verified 2026-10-28 (Windows, this PC). Selection: the real-data test of the new tool.
    - `PETSYS_DATA_DIR=C:/Users/dsanchez/Desktop/data pytest tests/test_imas_necr.py -m real_data -n 0`: 1 passed in 14.1 s. Without the variable it skips naming `PETSYS_DATA_DIR`.
    - `-k command -n 0 --slow-limit 5` after the CLI change: 1 passed.
- [x] **T9 — All transferred frames: FR-6 report** (FR-1–FR-8).
  - **Waits on** the owner's notice that the transfer is complete.
  - **Steps:** write `IMAS_work/necr_all.json` with every transferred frame (activity from `CTR_vs_activity_NECR.xlsx`, legacy `.lm` results) and run it.
  - **Done when:**
    - the report gives the NEMA value at frame 7 (or the transfer-check failure), the acceptance verdict, CTR against activity and the photopeak per frame;
    - missing frames are listed;
    - timings and memory are recorded.
  - **Owner reviews the verdicts.**
  - **Progress, 2026-10-28 (partial: frames present):**
    - **Run file:** `IMAS_work/necr_present.json`:
      - frame 7 (split 24), frame 20 (53), frame 25 (22, 23), frame 26 (22) and frame 30 (22–24);
      - frame 26 is outside the owner's set, kept as an extra point;
      - missing frames: 1, 5, 10, 15.
    - **Run:** 3 min 43 s wall, peak RSS 1.43 GB.
    - **Fitted line (headline), background-subtracted FWHM in ps:**

      | Frame | legacy | EM2 + walk3 |
      | --- | --- | --- |
      | 7 | 740 ± 34 | 649 ± 37 |
      | 20 | 609 | 484 |
      | 25 | 612 | 477 |
      | 30 | 553 | 445 |
      | pooled | 622 ± 12 | 486 ± 7 |

    - **Photopeak:** 494.7–498.9 keV.
    - **Acceptance:** accepted on frame 7 (71 vs 50) and pooled.
    - **Transfer check:**
      - frame 7: not accepted (EM 720 against legacy 740: 20 vs 48), so the EM and EM2 + walk3 NEMA values are withheld;
      - accepted on frames 20, 25 and 30 and pooled (71 vs 15).
    - **Finding: the frame-7 gate is unstable.** The pooled-frame line moves about 0.3 mm from the T8 frame-7-only line. That moves frame-7 legacy from 771 to 740 ps and flips transfer from accepted (47 vs 40) to not accepted. Frame 7 has about 114 k NEMA-band events, so its bootstrap SD (about 35 ps) is the size of the gain under test.
    - **Owner decision (2026-10-28):** keep the frame-7 gate and wait for 2–3 more frame-7 splits. Superseded the same day by the next entry.
  - **Spec change, 2026-10-28 (owner: "use always the model-free"):** FR-5/FR-6 and Spec changes are updated.
    - **What changed:**
      - The headline value, both gates and the CTR-against-activity curves use the model-free FWHM.
      - The NEMA widths are reported beside.
      - The report goes to schema `imas-necr-ctr-report/2`: `gates.headline_value` replaces `gates.nema_value`. Each value carries `nema_background_subtracted` and `nema_raw_fwhm_ps`.
      - The CLI prints the model-free value with the NEMA value beside.
    - **Plots:**
      - `ctr_vs_activity.png` shows model-free (solid) and NEMA background-subtracted (dotted).
      - Uncorrected is left out, off the scale at about 2.3 ns.
      - The legend sits outside the axes.
      - Timing-panel titles give the model-free FWHM.
    - **Verified:**
      - `tests/test_imas_necr.py` + `test_timing_nema.py` + `test_timing_line.py` with `-n 0 --slow-limit 5`: 49 passed, including `real_data` with `PETSYS_DATA_DIR` set.
      - The gate test asserts `judged_on` is model-free, and that the headline value equals the model-free FWHM, with the NEMA width beside.
    - **Rerun** of `necr_present.json`: 3 min 28 s, 1.43 GB. The previous output is kept as `IMAS_work/necr_present_nemagate`.
      - Frame 7, model-free: legacy 734 ± 6, EM 693 ± 5, EM2 + walk3 667 ± 5 ps.
      - Transfer accepted: 41 vs 8. Acceptance accepted: 26 vs 7.
      - Every frame and pooled accepted: pooled 600 / 539 / 493 ps.
      - Frame 7 on the image line: 741 / 696 / 667 ps.
    - **Frame-7 splits:** more are no longer needed for the gate. T9 still waits on frames 1, 5, 10 and 15.
  - **Revised the same day (owner):** the reported value is the NEMA width; the gates stay on model-free.
    - The NEMA value and the CTR-against-activity curves use the NEMA background-subtracted FWHM.
    - `gates.nema_value` carries `model_free` beside it.
    - The CLI labels the gates "(model-free)".
    - The plot shows NEMA solid and model-free dotted.
    - Verified: `tests/test_imas_necr.py` with `-n 0 --slow-limit 5`, 27 passed. Earlier in the same edit, the 49-test selection passed with 1 CLI-label failure, which is now fixed.
    - The real run's report has not been regenerated (same numbers); it will be rerun after the NEMA conformance decisions.
  - **NEMA NU 2-2018 §8 check** (owner's copy in `docs/imas/NEMA NU-2 2018c.pdf`, pp. 30–34):
    - **Matches:**
      - r, I and t as in §8.4.1;
      - background per t bin from the strip edges at ±20 mm, interpolated linearly (§8.4.2);
      - parabola through the peak and its 2 neighbours, then linear interpolation at half maximum (§8.4.3);
      - bins under 1/4 of the expected FWHM;
      - FWHM(j) against a_ave(j) (§8.5a).
    - **Gaps:**
      - (a) The AFOV is over 65 cm (line endpoints at ±356 mm), so §8.3.5 allows only the central 65 cm. Neither the legacy script nor our port cuts on z.
      - (b) §8.4.2a takes C_L and C_R interpolated at exactly r = ±20 mm. The port, like the legacy script, uses the nearest bins, whose centres sit 0.087 mm outside.
      - (c) §8.5b reports the value at 5.3 kBq/ml by linear interpolation between the frames just above and below. That needs frame 8 (4.67 kBq/ml); frame 7 is 5.31.
      - (d) §8.3.5.1 locates the line from image centroids of the first frame below peak NECR. The headline uses the LOR-convergence fit instead.
      - (e) §8.4 analyses whole frames. We use some splits per frame.
      - The raw width, FWTM and the bootstrap are not NEMA quantities. c = 0.3 against 0.299792 mm/ps is negligible.
    - **Owner decisions (2026-10-29):**
      - (a) cut |z_I| ≤ 325 mm;
      - (b) interpolate the strip-edge counts, keeping the legacy nearest-bin variant for parity only;
      - (c) use the §8.5b interpolation; the owner transfers frame 8;
      - (d) keep the fitted line and record the deviation.
      - Spec FR-3, FR-4, FR-6, Clarify 3 and Spec changes are updated.
    - **Code:**
      - `src/timing/nema.py`: `axial_ok`, `edge_counts` and `edges=NEMA|LEGACY` on `background_subtracted` and `summarize`.
      - `necr.py`:
        - z cut per line, with the stage `outside central 65 cm`;
        - `nema_interpolation` and `nema_value`: when frame 8 is missing, the value is unavailable and the nearest frame's values are given;
        - `nema_conformance` report block;
        - the CLI prints the interpolation or the missing frame.
    - **Tests:**
      - The parity test uses `edges=LEGACY`; the sloped background is removed exactly with the NEMA edges.
      - New: `test_edge_counts_interpolated_at_20_mm`, `test_central_axial_range`, `test_nema_interpolation_needs_consecutive_frames` (5 cases).
      - The run tests check the 65 cm stage, the interpolated value with weight 0.325 (synthetic activities are now 8 and 4 kBq/ml) and the conformance block.
      - `tests/test_timing_nema.py`, `test_imas_necr.py`, `test_timing_line.py` and `test_golden_provenance.py` with `-n 0 --slow-limit 5`, real data included: 99 passed and 1 CLI-label failure. After the fix, the CLI test passed on its own. The golden file is unchanged.
    - **Rerun** of `necr_present.json`: 3 min 34 s, 1.43 GB; the previous output is now `IMAS_work/necr_present_v2_nocut`.
      - The NEMA value is unavailable: frame 8 is missing.
      - Frame 7, NEMA legacy / EM / EM2+walk3: 734 ± 27 / 733 ± 28 / 668 ± 24 ps (was 740 / 720 / 649).
      - Frame 7, model-free: 736 / 695 / 672 ps.
      - Both gates accepted on every frame and pooled.
      - 7,935 of 650 k frame-7 same events fall outside the central 65 cm.
    - **T9 waits on** frames 8, 1, 5, 10 and 15.
    - **Results:** `docs/imas/IMAS_NECR_CTR/` (untracked).
  - **Progress, 2026-10-29 (frames 5 and 8 arrived, plus more frame-30 splits):**
    - **Run file:** `IMAS_work/necr_present2.json`: frame 5 (split 23), 7 (24), 8 (41), 20 (53), 25 (22, 23), 26 (22), 30 (22–26). Legacy `.lm` results for frames 5 and 8 from `CTR_vs_activity_NECR.xlsx`.
    - **Missing:** frame 1 (still copying when the run started), frames 10 and 15.
    - **Run:** 6 min 49 s wall, peak 1.65 GB; output `IMAS_work/necr_present2`.
    - **Line:** 38.8 mm from the axis at z = 0, median miss 1.37 mm; per-frame lines within 0.35 mm of the pooled line; 1.86–1.89 mm from the image line.
    - **NEMA value at 5.3 kBq/ml** (§8.5b, frames 7 and 8): legacy 723 ± 21, EM 730 ± 29, EM2 + walk3 665 ± 27 ps; model-free 736 / 695 / 671 ps.
    - **Background-subtracted FWHM (ps), legacy / EM / EM2 + walk3:**

      | Frame | kBq/ml | NEMA | model-free |
      | --- | --- | --- | --- |
      | 5 | 6.87 | 756 / 782 / 742 | 768 / 729 / 712 |
      | 7 | 5.31 | 723 / 731 / 665 | 736 / 695 / 672 |
      | 8 | 4.67 | 750 / 678 / 674 | 721 / 677 / 645 |
      | 20 | 1.00 | 610 / 532 / 485 | 594 / 535 / 491 |
      | 25 | 0.52 | 601 / 520 / 469 | 575 / 509 / 470 |
      | 26 | 0.46 | 600 / 484 / 459 | 584 / 515 / 470 |
      | 30 | 0.28 | 551 / 499 / 455 | 563 / 496 / 455 |
      | pooled | | 648 / 594 / 540 | 636 / 576 / 531 |

    - **Gates (model-free):** transfer and acceptance accepted on every frame and pooled.
    - **Photopeak:** 494.5–498.8 keV.
    - **T9 waits on** frame 1 (then rerun to a new output dir); frames 10 and 15 are not transferred.
  - **Progress, 2026-10-29 (all transferred frames):**
    - **Run file:** `IMAS_work/necr_present3.json`: frames 1 (split 24), 5, 7, 8, 10 (54), 15 (23), 20 (24, 53), 25 (22–24), 26, 30 (22–26) and 32 (39–42).
    - **Frame 32 activity:** 0.2137 kBq/ml, from `docs/imas/CTR_NECR_legacy/20241003_NECR_E0h4_120s.actlog` (0.111781 mCi × 1.91193, the xlsx/actlog factor at frame 30); the xlsx stops at frame 30. No legacy `.lm` result for frame 32.
    - **Run:** 11 min 37 s wall, peak 2.96 GB; output `IMAS_work/necr_present3`.
    - **NEMA value at 5.3 kBq/ml** (frames 7/8): legacy 724 ± 22, EM 731 ± 29, EM2 + walk3 664 ± 24 ps; model-free 736 / 695 / 671 ps.
    - **New frames, NEMA / model-free (ps), legacy / EM / EM2 + walk3:**
      - frame 1 (11.51 kBq/ml): 931 / 913 / 832 and 880 / 856 / 826;
      - frame 10 (3.61): 670 / 647 / 573 and 694 / 648 / 605;
      - frame 15 (1.89): 612 / 575 / 542 and 630 / 572 / 535;
      - frame 32 (0.21): 557 / 481 / 452 and 561 / 496 / 451, level with frame 30.
      - Pooled: 639 / 570 / 521 and 624 / 561 / 517.
    - **Gates (model-free):** transfer and acceptance accepted on all eleven frames and pooled (frame 1: 23 vs 16 and 31 vs 15).
    - **Photopeak:** 494.5–500.0 keV; frame 1 rises to 501.0 keV (pile-up suspected, not investigated).
    - **Line:** unchanged; per-frame lines within 0.35 mm of the pooled line except frame 1 (0.81 mm).
    - Every "done when" item is met. **Owner accepted the verdicts (2026-10-29).**

## Validation

- [x] **T10 — Validation and ship.**
  - **Runs:**
    - full non-real suite: `-m "not real_data"`;
    - `-m real_data`, because the feature touches real-data results.
  - **Recording:** results are summarised in `docs/imas/IMAS_timing/index.html` (gitignored), as a NEMA section.
  - **Status changes:** spec 013 Status → `shipped`; memory updated.
  - **Done when** both suites pass, or their failures are recorded with owner decisions.
  - **Progress, 2026-10-29 (Windows, this PC, clean tree after the T9 commits):**
    - `-m "not real_data"`: 1 failed, 1532 passed, 19 skipped, 4 xfailed in 296 s.
      - The failure: `tests/test_manager_checkout.py::CheckoutChecks::test_checkout_copy_imports_headless_without_private_settings`. The class skips while runtime files are uncommitted, so this was the first run of it since spec 011 (earlier full runs: 24 skipped, now 19).
      - **bug-checkout-probe**, two defects in the test's probe, not in the Manager:
        - it keyed `module_files` by `module.__name__`, so the spec 011 alias shims (`src.cornell.skew.lut`, `plots`, `pairs`, `solver`, which put the `src.timing` module in `sys.modules`) were missing (KeyError);
        - its `matplotlib.get_backend()` resolved the unset backend to QtAgg in the clean environment and imported PyQt5 itself, tripping the forbidden-import check. It now records `rcParams._get_backend_or_none()` (no field is asserted).
      - Fixed in the test; `--fr bug-checkout-probe -n 0 --slow-limit 5`: 1 passed; `tests/test_manager_checkout.py -m "not real_data" -n 0`: 5 passed.
    - `-m real_data` with `PETSYS_DATA_DIR=C:/Users/dsanchez/Desktop/data`, `PETSYS_CAL_DIR=<repo>/encal_files`: 34 passed, 0 skipped in 1,667 s.
    - **Activity correction (owner 2026-10-29):** NEMA §8.5a uses a_ave(j) (§1.2): the frame-start activity averaged over T_acq = 120 s (× 0.99370), over V = 22,000 ml.
      - The frame-start activity comes from `docs/imas/CTR_NECR_legacy/20241003_NECR_E0h4_120s.actlog`. LDAT frame j is actlog entry j − 1, as the legacy xlsx (owner).
      - The xlsx activity is exactly actlog(j − 1) × 37000/22000, without the decay average. The T9 frame-32 note above named the wrong actlog entry; its value (0.2137) matched by coincidence.
      - With a_ave, frame 7 is 5.278 kBq/ml, so §8.5b needs frames 6 (6.00) and 7. Frame 6 (split 54) was transferred.
    - **Final run `IMAS_work/necr_present4.json`** (12 frames: 1, 5, 6, 7, 8, 10, 15, 20, 25, 26, 30, 32; a_ave activities): 13 min 39 s, peak 3.11 GB. The suites ran before it; no code changed between them.
      - **NEMA value at 5.3 kBq/ml**, frames 6/7, weight on frame 6 0.030: legacy 723 ± 23, EM 736 ± 29, **EM2 + walk3 667 ± 28 ps**. Model-free: 737 / 696 / 673 ps.
      - Frame 6 (6.00 kBq/ml): NEMA 720 / 695 / 709, model-free 765 / 723 / 704 ps.
      - Adding frame 6 moved the pooled line slightly. Other frames changed by up to 12 ps NEMA (frame 5 EM 785 → 773) and 1 ps model-free.
      - Gates (model-free): transfer and acceptance accepted on all twelve frames and pooled (frame 6: 42 vs 11 and 19 vs 10).
      - Image-line sensitivity at 5.3 kBq/ml: 776 / 706 / 661 ps.
    - **Recording:**
      - NEMA section in `docs/imas/IMAS_timing/index.html` (final numbers, links the report).
      - NEMA-only test report `docs/imas/IMAS_NECR_CTR/nema_report/index.html` from `scripts_imas/necr_nema_report.py` (owner request): §8.5a/b results, test conditions, §8.4 processing, r profile, bracketing-frame histograms, conformance.
      - Outputs copied to `docs/imas/IMAS_NECR_CTR/run_present4/`.
    - Spec 013 Status → `shipped`; memory updated.
