# Tasks — IMAS fast LDAT to LM

Status: in progress 2026-11-03 (T1–T7 done; T8, the owner's run on the IMAS PC, pending). [Spec](spec.md), [plan](plan.md).

**Conventions:** each task cites `015-FR-n` and its check; tests carry `@pytest.mark.fr("015-FR-n")`; new tests run with `-m "not real_data" -n 0 --slow-limit 5`; `real_data` tests name files relative to `PETSYS_DATA_DIR`. No commits; the owner runs git.

**Selection for T1–T6:** new package `src/imas/listmode/` only; it imports `src.imas.skew` (select kernels, setup, timestamp, walk, outputs), `src.timing` and `src.ldat_stream` without changing them, and nothing imports it, so no consumer regression run per task. `tests/test_golden_provenance.py` is affected (new golden, accepted requirement `015-FR-6`).

- [x] **T1 — Selection kernel** (FR-2, FR-3, FR-5 inputs). `src/imas/listmode/select.py`.
  - **Done when** `tests/test_imas_listmode.py -k select`: each FR-3 cut counted on a synthetic LDAT, first failing cut only; hits below `en_min_ch` dropped; ENERGY rows with zeros where a row did not fire; swapped pairs flagged; batches bounded.
  - Tests: `test_each_cut_is_the_first_failing_one`, `test_energy_rows_of_the_selected_minimodule_zero_elsewhere`, `test_batches_are_bounded_and_equal_one_pass`.
  - Deviations:
    - Cut 3 (no positive minimodule energy) cannot happen with a floor > 0 and is not exercised.
    - Missing mu is checked on both sides before invalid mu, as the reference does.
    - A pair listed in both orders uses the listed order, as the legacy does.
- [x] **T2 — Timing** (FR-4). `timing.py`.
  - Tests: `test_dt_is_the_rounded_correct_dt`, `test_uncorrectable_events_get_the_sentinel_and_are_counted`, `test_tables_are_checked_together`, `test_legacy_skew_is_additive_top_rule`.
  - Deviations:
    - The int16 check is symmetric (|dt| ≤ 32 767), so a swapped pair's negated `dt` always fits.
    - `--walk none` applies a skew-only canonical LUT.
- [x] **T3 — Records and NN** (FR-5, FR-6). `records.py`, `nn.py`.
  - Tests: `test_lm_layout_and_header_are_the_legacy_ones`, `test_records_swap_sides_and_negate_dt`, `test_pixels_follow_the_legacy_searchsorted`.
  - The header is checked field by field against the legacy `write_header` values; the legacy script is not imported.
- [x] **T4 — Job** (FR-7–FR-10). `job.py`.
  - Tests:
    - `test_job_merges_segments_with_provenance`;
    - `test_worker_count_does_not_change_the_lm`: 0, 1 and 3 workers give byte-identical output; 3.5 s serial, so not marked slow;
    - `test_continue_skips_completed_segments`;
    - `test_a_failed_file_writes_no_merged_lm`;
    - `test_bad_inputs_stop_before_processing`;
    - `test_timed_debug_run`.
- [x] **T5 — Command line** (FR-1). `__main__.py`, shim `scripts_imas/imas_ldat2lm.py` (untracked).
  - Tests: `test_cli_defaults_follow_the_rule`, `test_default_out_dir_is_the_legacy_one`, `test_cli_names_a_missing_input_and_fails`.
  - Deviation: `-a --ask` keeps the legacy interactive prompts; `--acq-time`, `--activity` and `--isotope` set the same values without prompting.
  - Verified T1–T5 2026-11-03 (Windows, this PC): `python -m pytest tests/test_imas_listmode.py -m "not real_data" -n 0 --slow-limit 5`: 19 passed, 1 deselected.
- [x] **T6 — Real-data parity** (criterion 2).
  - Reference: `scripts_imas/imas_listmode_reference_015.py` (untracked, SHA-256 `2171e9ba…`). It is the legacy loop with the floor, both bug fixes and the `listmode_skew_walk.md` timing class.
  - Golden: `tests/data/golden/timing/015_t6_imas_lm_prefix.bin` (+ provenance), from the first 100 000 records of `IMAS/20261019_Na22_1cm_cFOV_test_300s_coincCompact_00000034.ldat`: 4 433 coincidences.
  - `test_real_prefix_matches_the_legacy_reference`:
    - every field is equal except `dt`;
    - `dt` differs by 1 ps in 164 of 4 433 records (half-ps rounding of the float corrected time);
    - the uncorrected set and the rejection counts are the same.
  - Verified 2026-11-03 (Windows): `PETSYS_DATA_DIR=C:/Users/dsanchez/Desktop/data python -m pytest tests/test_golden_provenance.py tests/test_imas_listmode.py -m "not slow or slow" -n 0 --slow-limit 5`: 63 passed.
- [x] **T7 — Speed and validation** (FR-11, criteria 1–3).
  - Fast run, one split (this PC, CPU NN, `-w 0`, split 34: 1.04 GB, 3 124 015 records):
    - 139 346 coincidences in 7.0 s for the whole run, setup and model load included;
    - read + select + timing at 1.09 M records/s.
  - Fast run, 5 splits (34–38), `-w 5`:
    - 695 640 coincidences from 15.6 M records in 11.9 s (1.32 M records/s for the whole run);
    - the NN takes 6.7 s of that, so one CPU NN process limits how many workers help.
  - Reference on all of split 34:
    - 3 124 015 records; read + select + timing in 105.5 s, i.e. 29 611 records/s;
    - fast / reference = 36.9× (FR-11 asks ≥ 20×);
    - the full-split LM equals the reference on every field and count (139 346 coincidences); `dt` differs by 1 ps in 5 175 records (3.7 %), never more.
  - Verified 2026-11-03 (Windows, this PC), spec Validation gate: `python -m pytest -m "not real_data"`: 1591 passed, 19 skipped, 4 xfailed.
  - Real data: the feature's own `real_data` test passed under T6. No shared code was edited, so no other real-data result changes, and the unrelated `-m real_data` acquisitions wait for integration.
- [ ] **T8 — IMAS PC run** (FR-1, FR-7, FR-11; owner).
  - Confirm the `--luts` default; `/scratch/imas_files_cal/I3M_newSkew_walk/luts` is assumed.
  - Run one acquisition with the GPU NN and several workers.
  - Record the throughput, and whether GPU positions match the CPU golden.
