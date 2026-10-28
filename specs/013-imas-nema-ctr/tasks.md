# Tasks — IMAS NEMA NU 2-2018 CTR on the NECR phantom

Status: open, 2026-10-28. [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `013-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("013-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially. Run new and changed tests with `-n 0 --slow-limit 5`.
- `real_data` tests name acquisitions relative to `PETSYS_DATA_DIR` (`IMAS/NECR/...`, `IMAS_work/...`) and skip naming the missing variable or file.
- Record each check's selection, reason, command, platform and pytest summary line as `Verified:`. Tick a task only on success.
- No commits, pushes or release builds; the owner runs git.

## Core

- [ ] **T1 — NEMA core and parity golden** (FR-4).
  - **Code:** `src/timing/nema.py`: constants, `line_frame`, `NemaHistograms` (batch fill), `background_subtracted`, `nema_widths`.
  - **Fixture:** `tests/necr_helpers.py`: seeded LORs and `(Δt, r)` arrays.
  - **Golden:** the local `scripts/capture_013_t1.py` writes `tests/data/golden/timing/013_t1_nema_legacy.json` from the legacy script, with provenance. `tests/test_golden_provenance.py` accepts `013-FR-4`.
  - **Done when** `tests/test_timing_nema.py tests/test_golden_provenance.py -n 0 --slow-limit 5` pass:
    - **Geometry:** `line_frame` r, I and Δtof equal the golden within 1e-9 mm or ps.
    - **Histograms:** filling in 7 uneven batches gives the golden raw and 2D histograms exactly; the background-subtracted histogram matches within 1e-9.
    - **Widths:** FWHM, FWTM and peak equal the golden within 1e-9 ps.
    - **Failure cases:** a peak in an edge bin, and a side with no crossing, each give a status, never an exception or a number.
    - **Background recovery:** on a flat background and on a sloped one (linear in r), the subtraction recovers an injected peak's counts within 3 % and its FWHM within 2 %.
- [ ] **T2 — Line from LOR convergence** (FR-3).
  - **Code:** `src/timing/line.py` (`seed_line`, `slab_points`, `principal_line`, `refine_line`, `locate_line`).
  - **Done when** `tests/test_timing_line.py -n 0 --slow-limit 5` passes. The fixture is a tilted line plus 30 % background LORs (flat over the bore), with 2 mm endpoint blur:
    - **Recovery:** the fitted line is within 0.5 mm of the true line for |z| ≤ 350 mm.
    - **Steps:** the seed is within 3 mm, and the slab points lie within 1 mm of the line (median).
    - **Axis-parallel line:** an axis-parallel line is recovered with slopes below 1e-3.
    - **Errors:** fewer than three LORs per slab gives an empty slab, counted. All slabs empty, or parallel LORs, give a `CtrError`, never a line.

## IMAS NECR tool

- [ ] **T3 — NECR file and frozen inputs** (FR-1).
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
- [ ] **T4 — Accumulation on a synthetic NECR acquisition** (FR-2, FR-3, FR-4, FR-5).
  - **Code:** `necr.Accumulator`, the localization pass, `run_necr`; `tests/necr_helpers.py` synthetic LDAT line source (plan § Synthetic fixture), with injected delays and walk.
  - **Done when** `tests/test_imas_necr.py -k accumulate -n 0 --slow-limit 5` passes:
    - **Line:** the pooled fitted line is within 1 mm of the injected line.
    - **Centring:** every table's background-subtracted Δt is centred within 2 bootstrap SD of 0.
    - **Ordering:** the FWHM orders as uncorrected > legacy > EM > EM + walk.
    - **Counts:** per frame, the cut counts sum to the records. Events without a delay or without a walk input are counted and excluded from every table.
    - **Bootstrap:** replicates are reproducible with the same seed.
    - **Image line:** a declared line offset by 3 mm gives a wider NEMA FWHM than the fitted line, on the same events.
- [ ] **T5 — Summary, gates and report** (FR-5, FR-6).
  - **Code:** the NEMA and model-free widths per frame, pooled, table and line; bootstrap SD; paired differences against legacy; the transfer check and acceptance (`_judge`); the legacy `.lm` results beside ours; and `necr_ctr.json`, `necr_histograms.npz`.
  - **Done when** `tests/test_imas_necr.py -k "report or gate" -n 0 --slow-limit 5` passes:
    - **Fields:** the report has every field of plan § Architecture 3.
    - **Gates:** they are evaluated on the `gates.nema_frame` frame.
    - **Transfer failure:** a fixture where EM is not better than legacy gives `transfer: failed`, no `nema_value` for `em` or `em2_walk3`, and still writes their histograms.
    - **Pass case:** a passing fixture reports both verdicts with their differences and uncertainties.
    - **Model-free width:** equals `core.fit_histogram(..., crossings=local_crossings)` on the same histogram.
- [ ] **T6 — Energy check** (FR-7).
  - **Code:** the photopeak per frame (`encal.accumulate` → keV histogram → peak fit), the calibration recorded in the report, and the fallback recorded when the NECR calibration is missing.
  - **Done when** `tests/test_imas_necr.py -k photopeak -n 0 --slow-limit 5` passes:
    - synthetic frames with a 0 % and a 5 % gain shift report their photopeaks at 511 and 536.6 keV (±1 %);
    - a channel without `mu` is excluded and counted.
- [ ] **T7 — Diagnostics and CLI** (FR-8).
  - **Code:** `src/imas/skew/necrplots.py`; `python -m src.imas.skew necr FILE` in `__main__.py`.
  - **Done when:**
    - `tests/test_imas_necr.py -k plots -n 0 --slow-limit 5` passes: every FR-8 PNG is written for the synthetic run, and the CLI writes the report;
    - `py_compile` succeeds on the new modules;
    - the affected spec 011/012 CTR tests pass unchanged: `tests/test_imas_skew_ctr.py tests/test_imas_walk.py tests/test_timing_ctr_local.py -m "not real_data"`.

## Real data

- [ ] **T8 — NECR energy calibration and frame 7 run** (FR-3, FR-7).
  - **Steps:**
    1. Run `python -m src.imas.skew encal` on the highest-activity frame present; record which frame was used and why.
    2. Write `IMAS_work/necr_frame7.json` with frame 7 only and the four tables.
    3. Run it.
  - **Done when:**
    - the run completes, with records, timings and peak memory recorded here;
    - the fitted line and its distance to the image line are recorded;
    - the `tests/test_imas_necr.py` `real_data` test passes on frame 7 with `max_records` 2,000,000: the line lies within 10 mm of the image line at z = 0, and every table's NEMA FWHM is finite;
    - otherwise it skips naming `PETSYS_DATA_DIR` or the file.
- [ ] **T9 — All transferred frames: FR-6 report** (FR-1–FR-8).
  - **Waits on** the owner's notice that the transfer is complete.
  - **Steps:** write `IMAS_work/necr_all.json` with every transferred frame (activity from `CTR_vs_activity_NECR.xlsx`, legacy `.lm` results) and run it.
  - **Done when:**
    - the report gives the NEMA value at frame 7 (or the transfer-check failure), the acceptance verdict, CTR against activity and the photopeak per frame;
    - missing frames are listed;
    - timings and memory are recorded.
  - **Owner reviews the verdicts.**

## Validation

- [ ] **T10 — Validation and ship.**
  - **Runs:**
    - full non-real suite: `-m "not real_data"`;
    - `-m real_data`, because the feature touches real-data results.
  - **Recording:** results are summarised in `docs/imas/IMAS_timing/index.html` (gitignored), as a NEMA section.
  - **Status changes:** spec 013 Status → `shipped`; memory updated.
  - **Done when** both suites pass, or their failures are recorded with owner decisions.
