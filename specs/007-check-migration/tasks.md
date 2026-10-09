# Tasks 007 — Move every remaining check into tests/

Spec: [`spec.md`](spec.md). Plan: [`plan.md`](plan.md). Owner approved 2026-10-07. Environment: env interpreter `-m pytest` from the repo root. Never stage or commit; the owner runs git.

**Migration Done-when (applies to every task marked "migrate"):** in one session on the same fixtures, the script (each listed mode) gives `PASS N` (or `Ran N … OK`) and the new pytest selection gives `N passed` (gui/linux skips counted with their reason); `--durations=0 -n 0` reviewed and tests ≥ 5 s marked `slow`; `test_infra.py` scan passes (no `sys.path`, no `scripts*`/test-file imports); then the script is deleted, or, when a remaining script still imports it, kept until its last importer's task (plan: Deferred deletion). Record both counts, the selection command and any listed exclusion (retired real mode, owner-approved replacement).

## Infrastructure

- [x] **T0 — Default run, calibration fixture, slow guard** (FR-4, FR-5). `addopts` `-m "not real_data and not slow"`; `real_cal_file` fixture (`PETSYS_CAL_DIR`, skip naming it); `--slow-limit SECONDS` option; `test_infra.py` cases for all three; spec 006's `--fr` unchanged.

  **Done when:** `python -m pytest tests/test_infra.py` passes, covering: default run deselects `slow` and `real_data`; `-m "not real_data"` runs `slow`; `real_cal_file` skips naming `PETSYS_CAL_DIR` (unset) and the missing file; `--slow-limit 1` fails a 1.5 s non-slow test and passes a `slow` one. Full suite still passes.

  Verified 2026-10-07 (Windows): `python -m pytest tests/test_infra.py` → 25 passed (7 new: 3 `real_cal_file`, default/full selection, `--slow-limit 1` fails the unmarked 1.5 s test with `slow-limit: took 1.5x s >= 1 s`, off by default); `python -m pytest` → 100 passed; `-m "not real_data" --slow-limit 5` → 100 passed. Negative (scratch copies): dropping the `slow` exemption → 1 failed; `CAL_ENV` → `PETSYS_DATA_DIR` → 3 failed; `addopts` without `not slow` → 2 failed.

- [x] **T1 — Retire three scripts** (FR-1). Delete `scripts/ldat_package_check.py`, `scripts/cornell_slab_en_cal_check.py` (+ `scripts/fixtures_cornell_slab_spectra.json`), `scripts/ldat_real_check.py`.

  **Done when:** no other script imports them (grep); files deleted; their spec 004 / bug evidence stays in earlier `tasks.md`.

  Verified 2026-10-07: grep of all `*.py` under the repo (incl. `scripts_cornell/`, `scripts_imas/`) finds only self-references (usage lines, the en_cal check loading its own JSON); the four files were untracked, deleted with `rm` (local backup kept outside the repo); `scripts/` now holds 27 `*_check.py`; earlier evidence untouched in specs 001, 002, 004, 006 `tasks.md`.

- [x] **T2 — Direct src tests** (FR-9, FR-10). `tests/test_filters.py`, `tests/test_filters_fixed.py`, `tests/test_fem_handler.py`, `tests/test_yaml_handler.py` per spec FR-9.

  **Done when:** `python -m pytest --fr 007-FR-9 -rx` passes with strict xfails only for `bug-filter-channel-list`, `bug-filter-max-sm-minimodules`, `bug-yaml-bool-as-int`, `bug-yaml-tuple-type-message`, each failing before marking (recorded); scalar/vectorized agreement on shared synthetic events; a negative check (deliberately broken copy) fails.

  Verified 2026-10-07 (Windows): `python -m pytest --fr 007-FR-9 -rx` → 54 passed, 6 xfailed (4 bug ids; `bug-filter-channel-list` has 3 cases: det1 valid, timestamps-not-channels, det2 valid). `--runxfail` → exactly those 6 fail, each for its bug (max_sm `False` for one SM in two mM; channel_list judges timestamps; `AttributeError: 'tuple' object has no attribute '__name__'`; bool `DID NOT RAISE`). Vectorized vs scalar agree on 7 shared events incl. padding and the empty event, dict and array forms. Negative (scratch copies of `src`): padding counted in `filter_min_ch_vectorized` → 7 failed; row/col swapped in `get_coordinates` → 3 failed; both-groups check removed → 1 failed; `>=` in `filter_total_energy` → 2 failed; dropping `has_hits` in `filter_single_mM_vectorized` → 0 failed (equivalent: the min/max sentinels already reject an empty event). `python -m pytest` → 158 passed, 6 xfailed; `-m "not real_data" --slow-limit 5` → same. Synthetic maps shared via `tests/helpers.py` (`SYNTH_CHTYPE`, `SYNTH_SM_MM`). FEM128 summed coordinates not asserted (y offset 7 against a 16-channel span gives negative y; no map uses it).

- [x] **T3 — Manager helpers** (FR-6). `tests/manager_helpers.py`: copy the shared builders and fakes listed in the plan; mixins for the shared `TestCase` setup (`CalibrationFixtures`, `ListmodeFixtures`, `QCFixtures`, `CLIFixtures`). Scripts stay untouched.

  **Done when:** module imports without Tk/display; a smoke test builds one fixture of each kind into `tmp_path`; full suite passes.

  Verified 2026-10-07 (Windows): `tests/manager_helpers.py` holds the shared fakes (`FixtureProbe`, `profile_fixture` (was `fixture`), `FakeChild`, `FakeBackend`, `DirectDummyChild`, `FakeDaemon`, `FakeResources`, `ToolWorld`), wire encoders, `CalibrationFixtures` (was `Fixtures`), `sides_for`/`P1_SPECS`/`P3_SPECS`/`pairs`, and the mixins `ListmodeFixtures`, `QCFixtures` (each also `.at(root)` standalone, replacing the scripts' `helper(TestCase(...))` trick), `CLIFixtures`, `PrivateOutput` (private temp folder per class, removed after it unless `PETSYS_KEEP_FIXTURES`); oracle code (`load_reference`, `Oracle`, `reference_loop`, `reference`, `reference_outputs`) left out for T8. AST comparison (docstrings ignored, renames mapped): 71 copied definitions identical to the scripts; only `setUp`/`make`/`listmode_request`/`qc_request` changed for `.at()`. `python -m pytest tests/test_manager_helpers.py` → 10 passed (subprocess import without `tkinter`/`customtkinter`; profile passes `preflight`; encoders round-trip through `read_binary_file`; calibration LDAT passes `validate_ldat` (compact + fixed); listmode/QC standalone; fakes; `ToolWorld` acquisition; both mixins). `--fr 007-FR-6` selects the 2 unittest-style methods too (10 passed). `PrivateOutput` cleanup: temp folder count unchanged after a run (108 older script folders), 2 kept with `PETSYS_KEEP_FIXTURES=1` (then removed). `python -m pytest` → 176 passed; `-m "not real_data" --slow-limit 5` → 176 passed. Scripts untouched.

- [x] **T25 — Parallel default run** (FR-5; owner 2026-10-07, done after T14). `pytest-xdist==3.8.0` + `execnet==2.1.2` in `process_petsys.yml` and the env; `addopts` `-n auto --dist loadscope`; conftest refuses `--slow-limit` with workers; `test_infra.py` inner runs serial (`-n 0`), plus cases for the parallel default and the refusal.

  **Done when:** `python -m pytest tests/test_infra.py` passes, covering: the default run uses workers and keeps each class on one worker; `--slow-limit` with workers exits with a usage error naming `-n 0`. Default run passes in < 120 s with no new failure; `-m "not real_data" -n 0 --slow-limit 5` and `-m "not real_data"` (parallel) pass; the fixture folder count is unchanged.

  Verified 2026-10-07 (Windows, 24 CPUs): `pip install --no-deps pytest-xdist==3.8.0 execnet==2.1.2` (dry run: nothing else; `pip check` clean). `python -m pytest tests/test_infra.py -n 0` → 50 passed (2 new). Default run 512 passed, 18 skipped in 34.7 s (serial `-n 0`: 92.2 s); full `-m "not real_data"` 537 passed, 18 skipped in 211.6 s (serial `-n 0 --slow-limit 5`: 505.2 s, same counts). Fixture folders 908 before and after. Without the guard, a parallel `--slow-limit 5` full run failed 4–6 tests at 5.5–9.0 s that pass serially, which is CPU contention. xdist omits `deselected` from the summary. Negatives (scratch copies): no conftest guard → refusal test fails; `--dist load` → class test fails; no `-n auto` → both fail.

## Manager group (spec 003)

- [x] **T4 — migrate `petsys_manager_artifact_check`** (44 tests) → `tests/test_manager_artifacts.py`.

  Verified 2026-10-07 (Windows): script `python scripts/petsys_manager_artifact_check.py` → `PASS: 44 checks; 1 skipped`; `python -m pytest tests/test_manager_artifacts.py` → 43 passed, 1 skipped (`linux`: the symlink test, `@unittest.skipIf(os.name == "nt")` → `@pytest.mark.linux`, FR-7). `ArtifactChecks` moved unchanged on `PrivateOutput` (replaces its `setUpClass`); AST: all 48 methods identical to the script. Citations (FR-8): class `003-FR-9`, `003-FR-11`, `003-FR-16` (spec 003 T4); the 7 T29 tests add `003-FR-13`, the FR-24 test `003-FR-24`; `--fr 003-FR-16` → 43 passed, 1 skipped; `--fr 003-FR-13` → 7; `--fr 003-FR-24` → 1. `--durations`: slowest 0.06 s, none `slow`. No fixture folder left after the run. `python -m pytest` → 220 passed, 1 skipped; `-m "not real_data" --slow-limit 5` → 220 passed, 1 skipped (includes the `test_infra.py` scan). Script deleted; `petsys_manager_check.py --artifacts`/`--all` (lazy import) no longer load it, aggregate dropped at T6.
- [x] **T5 — migrate `petsys_manager_daqd_check` (22) and `petsys_manager_acquisition_check` (27)** → `tests/test_manager_daqd.py`, `tests/test_manager_acquisition.py`.

  Verified 2026-10-07 (Windows): scripts → `PASS: 22 DAQD checks`, `PASS: 27 acquisition checks`; `python -m pytest tests/test_manager_daqd.py tests/test_manager_acquisition.py` → 49 passed (22 + 27). `DaqdChecks`/`AcquisitionChecks` moved on `PrivateOutput`; fakes from `manager_helpers` (`FakeDaemon`/`FakeResources` were the daqd script's own, AST-identical at T3); `FreshChildBackend`, `wait_for`, `FAST`, `POLICY` and the acquisition `ScaledClock`/`Scenario*` stay in their test files. AST (renames `fixture` → `profile_fixture`, `ROOT` → `REPO`): daqd 32, acquisition 50 definitions identical; only `setUpClass` replaced. Citations: DAQD class `003-FR-5/6/7/16` (T6), T35 test adds `003-FR-1`; acquisition class `003-FR-5/7/8/9/16` (T7), 4 bias-off tests add `003-FR-19`, progress/loss-limit/safety-defaults tests `003-FR-20`; `--fr 003-FR-6` → 22, `003-FR-8` → 27, `003-FR-19` → 4, `003-FR-20` → 3, `003-FR-1` → 1. `--durations`: slowest 1.64 s, none `slow`. No fixture folder left (908 before and after). `python -m pytest` → 271 passed, 1 skipped; `-m "not real_data" --slow-limit 5` → 271 passed, 1 skipped. `petsys_manager_acquisition_check.py` deleted (only the aggregate runner imported it, lazily). **`petsys_manager_daqd_check.py` kept: deletion deferred to T15** (`petsys_manager_gui_check.py` imports `FakeDaemon`, `FakeResources` from it).
- [x] **T6 — migrate `petsys_manager_check`** (75; modes `--settings --commands --runner --artifacts`, plus GUI/Linux parts) → `tests/test_manager_{settings,commands,runner}.py` (artifacts already in T4); aggregate `main()` dropped.

  Verified 2026-10-07 (Windows): script `--settings` → `PASS: 37`, `--commands` → `PASS: 10`, `--runner` → `PASS: 28` (75); `python -m pytest tests/test_manager_settings.py tests/test_manager_commands.py tests/test_manager_runner.py` → 75 passed (37 + 10 + 28; no GUI/Linux-only test in the script, all run on Windows). `CommandChecks`/`RunnerChecks` subclassed `SettingsChecks` (the script picked them by name prefix): the shared setup is now the `SettingsFixtures` mixin in `manager_helpers` and each class holds only its own tests, so nothing runs twice. `PrivateOutput.output_prefix` renamed `fixture_prefix` (clashed with `CommandChecks`' `self.output_prefix` path). `FakeClock`, `DirectDummyBackend` stay in `test_manager_runner.py`. AST (renames `fixture` → `profile_fixture`, `ROOT` → `REPO`): 81 methods identical; one adapted: `test_imports_do_not_launch_processes_threads_or_gui` execs `tests/manager_helpers.py` and itself instead of the local `scripts/petsys_manager_linux_check.py` and the script (negative: a helpers copy starting a thread at import → that test fails). Citations: settings `003-FR-2/3/16` (T2); commands and runner `003-FR-4/5/7/10/16` (T3), FR-23 test adds `003-FR-23`, T36 test `003-FR-1`; `--fr 003-FR-2` → 37, `003-FR-4` → 38, `003-FR-23` → 1, `003-FR-1` → 1. `--durations`: slowest 0.66 s, none `slow`. No fixture folder left. `python -m pytest` → 349 passed, 1 skipped; `-m "not real_data" --slow-limit 5` → 349 passed, 1 skipped. Aggregate `main()` dropped. **`petsys_manager_check.py` kept: deletion deferred to T15** (imported at module level by the daqd, cli, workflow and gui checks).
- [x] **T7 — migrate `petsys_manager_numeric_check`** (41; modes `--formats --calibration --listmode --qc --bounded --scope`) → `tests/test_manager_formats.py`; the other modes map to T9–T12 files.

  Verified 2026-10-07 (Windows): script `--formats` → `PASS: 41 checks`; `python -m pytest tests/test_manager_formats.py` → 41 passed (39 `FormatChecks` + 2 `ValidationSpeedChecks`). The other modes only load the calibration/listmode/QC/bounded/scope scripts (compared at T9–T12). `ValidationSpeedChecks` subclassed `FormatChecks` (the script added its two tests by name): the shared setup is now a `FormatFixtures` mixin in the test file, each class holds only its own tests. `HIT`, `fixed_record`, `compact_record`, `fixture_map` from `manager_helpers` (AST-identical at T3); `SIDE_A`/`SIDE_B` and the `record_by_record` oracle stay in the test file. AST (`ROOT` → `REPO`): 52 definitions identical; `setUp` differs only by `super().setUp()`. Citations: `FormatChecks` `003-FR-3/10/11/12/15/16` (T5), the compact-calibration route test adds `003-FR-21` (T20); `ValidationSpeedChecks` `003-FR-1/15/16` (T23); `--fr 003-FR-10` → 39, `003-FR-21` → 1, `003-FR-1` → 2. `--durations`: slowest 2.20 s (GIL/speed test, 103 MB file), none `slow`. No fixture folder left. `python -m pytest` → 391 passed, 1 skipped; `-m "not real_data" --slow-limit 5` → 391 passed, 1 skipped. **`petsys_manager_numeric_check.py` kept: deletion deferred to T15** (imported at module level by the calibration, listmode, QC and gui checks; the CLI check imports `compact_record` inside a test).
- [x] **T8 — Golden capture** (FR-3). `scripts/capture_golden_007.py` (local): runs the calibration, listmode and QC oracles (`scripts_cornell/cornell_slab_en_cal.py`, `cornell_listmode_cog_fixed_position.py`, `cornell_system_validation.py`) on `manager_helpers` fixtures → `tests/data/golden/…` + provenance; `tests/test_golden_provenance.py`. (`petsys_manager_reference_check --synthetic` contracts dropped from T8, owner 2026-10-07: see spec Clarify and T16.)

  **Done when:** every golden file has a provenance record (oracle path + SHA-256, capture script SHA-256, fixture builder + seed, map, settings, date); capturing twice gives byte-identical files; the provenance test passes; no tracked file imports `scripts_cornell`.

  Verified 2026-10-07 (Windows): `scripts/capture_golden_007.py` (local, untracked) holds the oracle code from the calibration/listmode/QC checks (`load_reference`, `Oracle`, `reference_loop`, `reference`, `reference_outputs`) and writes 33 golden files (1.5 MB) under `tests/data/golden/`: calibration 8 (P = 1 with the reference cap and with the cap patched to 700: `.encal`, status text, sides per key; P = 3 per-event oracle), listmode 17 (legacy header; 7 record streams with per-file counts: default inputs, per-file seeds, non-positive mu, zero-width DOI and COG, controlled pair, pair row 99; loaders with sparse arrays; the two reference `IndexError` cases; debug summaries), QC 8 (manual pairs, default inputs with reference fits, per-file seeds, pair cap, output names per option, PDF/Excel content, bounded, zero-energy `TypeError`). Each has `<name>.provenance.json`: oracle path + SHA-256, capture script SHA-256, repo commit, replaced script tests, `manager_helpers` builders + seeds, map SHA-256, settings, input SHA-256s, date. Captured twice → byte-identical (`diff -r`), 24 s each. Dry run of T9–T11 (scratch script): the Manager's outputs on the same fixtures equal all 33 golden files. `.gitattributes` `tests/data/golden/** -text` (with `core.autocrlf` a Windows checkout would change the bytes). `python -m pytest tests/test_golden_provenance.py` → 36 passed (33 files, areas, orphan records, import scan of `src/`, `exe_programs/`, `tests/`); negatives: one golden byte changed → 1 failed; a provenance record removed → 2 failed; a temporary `src` module importing `scripts_cornell` → 1 failed. All 67 tracked `.py` files: no `scripts_cornell`/`scripts_imas`/`gui_cornell` import. No fixture folder left (908 before and after). `python -m pytest` → 428 passed, 1 skipped; `-m "not real_data" --slow-limit 5` → 428 passed, 1 skipped.

- [x] **T9 — migrate `petsys_manager_calibration_check`** (22, minus `RealCornellChecks` retired) → `tests/test_manager_calibration.py`, oracle comparisons → golden files.

  Verified 2026-10-07 (Windows): script → `PASS: 21 calibration checks` (without `--real`); `python -m pytest tests/test_manager_calibration.py -m "not real_data"` → 21 passed (default run: 17, the 4 `slow` deselected). `CalibrationChecks` on `PrivateOutput` (replaces `setUpClass`'s temp folder); fixtures from `manager_helpers` (`CalibrationFixtures` was `Fixtures`). The 4 oracle tests compare with `tests/data/golden/calibration/` (`golden()` reads bytes, so a CRLF copy fails): per-slab `.encal`/status + accepted sides, the 700-event limit (events passed, `.encal`, status), coverage from the golden sides per key, P = 3 `.encal`/status; the script's "oracle loop = reference loop" assertion checked the oracle, not the Manager, and is gone (the oracle lives only in `scripts/capture_golden_007.py`). The per-slab status assertions now read the Manager's statuses (same text as the golden status file; reference and Manager both omit the `no fit: no resolved events` default). The listmode test uses `ListmodeFixtures.at(...)` instead of the imported `ListmodeChecks`. The owner-`.encal` branch of the outputs test (ran only if `encal_files/` existed) is now the `real_data` test `test_owner_reference_encal_loads` (`real_cal_file`, tracked January map): with `PETSYS_CAL_DIR=encal_files` → 1 passed; unset → skipped `real_data: PETSYS_CAL_DIR is not set`. AST (`ROOT` → `REPO`, decorators ignored): 18 of 25 definitions identical; the 7 that differ are `setUpClass`, the 4 golden tests, the outputs test (owner branch moved) and the listmode test. Negatives (scratch golden copy): changed mu → per-slab failed; changed status → P = 3 failed; CRLF status → limit test failed; accepted sides + 1 → failed; one key dropped from sides per key → coverage failed (+1 on one key: still passes, since min/median/below counts are unchanged). Citations: class `003-FR-10/12/15/16/21` (T20); the two T25.4 parallel tests add `003-FR-1`; the 4 golden tests add `007-FR-3`; `--fr 003-FR-1` → 2, `--fr 007-FR-3` (this file) → 4. `--durations`: `slow` workers 37 s, one decoding 26 s, target limit 23 s, parallel progress 4.9–5.4 s (over 5 s on 3 reruns); the others ≤ 2.2 s. `die_after_marker` stays a module function (spawn workers import it from the test module). No fixture folder left (908 before and after; the script's own run left one, removed). `python -m pytest` → 446 passed, 1 skipped, 5 deselected; `-m "not real_data" --slow-limit 5` → 450 passed, 1 skipped. **`petsys_manager_calibration_check.py` kept: deletion deferred to T15** (imported at module level by the bounded, cli, listmode, qc and scope checks; numeric and real import it lazily).
- [x] **T10 — migrate `petsys_manager_listmode_check`** (21) → `tests/test_manager_listmode.py`, oracle → golden.

  Verified 2026-10-07 (Windows): script → `PASS: 21 listmode checks`; `python -m pytest tests/test_manager_listmode.py -m "not real_data"` → 21 passed (default run: 17, the 4 `slow` deselected). `ListmodeChecks` on `ListmodeFixtures` + `PrivateOutput`; the 14 fixture methods are `manager_helpers.ListmodeFixtures` (AST-identical, `SPECS` → `LISTMODE_SPECS`; only `setUp` differs, for `.at()`); `RECORD`, `HEADER_OFFSETS`, `RECORD_OFFSETS`, `LEGACY`, `decode_header` stay in the test file. The 10 oracle tests compare with `tests/data/golden/listmode/`: legacy header bytes; record bytes + per-file counts for the default inputs (records/merge order and compact = fixed tests), per-file seeds, non-positive mu, zero-width DOI then COG, controlled pair; loaders (pair map, region map, sparse calibration and COG arrays); pair row 99 bytes (the reference writes pair 9); debug energy/SM/flood summaries. The two "reference raises `IndexError`" asserts now read the recorded exception in `pair_row99.json` (the Manager side, counted rejections, is unchanged). AST (`ROOT` → `REPO`, decorators ignored): 11 of 21 tests identical; the 10 that differ are the golden tests. Negatives (scratch golden copy, one change per golden test): a header byte, a record byte (default inputs, controlled pair, zero-width COG), a per-file count (seeds, non-positive mu, zero-width DOI), a calibration-array value, an SM hit count, the recorded exception name → all 10 golden tests failed. Citations: class `003-FR-2/9/12/15/16` (T9); per-file seeds adds `003-FR-22` (T25.5); compact tests add `003-FR-10/13/22` (T21); in-place adds `003-FR-13` (T29.2); the 10 golden tests add `007-FR-3`; `--fr 003-FR-22` → 3, `003-FR-13` → 3, `007-FR-3` (this file) → 10. `--durations`: `slow` debug summaries 14 s, compact = fixed 11 s, per-file seeds 8 s, debug floodmap 5.1–5.3 s (`debug_plots` alone 5.0 s, not import time); the others ≤ 1.1 s. The floodmap test's 3 scipy fit warnings (degenerate 50-count energy spike) were there before and are left alone. No fixture folder left (908 before and after; the script's own run left one, removed). `python -m pytest` → 464 passed, 1 skipped, 9 deselected; `-m "not real_data" --slow-limit 5` → 472 passed, 1 skipped. **`petsys_manager_listmode_check.py` kept: deletion deferred to T15** (imported at module level by the cli, gui, scope and workflow checks; calibration, checkout and numeric import it lazily).
- [x] **T11 — migrate `petsys_manager_qc_check`** (15) → `tests/test_manager_qc.py`, oracle → golden.

  Verified 2026-10-07 (Windows): script → `PASS: 15 checks`; `python -m pytest tests/test_manager_qc.py -m "not real_data"` → 15 passed (default run: 11, the 4 `slow` deselected). `QCChecks` on `QCFixtures` + `PrivateOutput`; the 9 fixture methods are `manager_helpers.QCFixtures` (AST-identical, `SPECS` → `QC_SPECS`; only `setUp` differs, for `.at()`); `histogram`, `fingerprint`, `pdf_lines`, `is_subsequence`, `LEGACY_HALVES` stay in the test file. The 8 oracle tests compare with `tests/data/golden/qc/`: manual pairs (accepted pairs, occupancy, slab counts, minimodule sample counts); default inputs (populations, histograms from the reference energies, the reference `fit_photopeak_worker` mu/sigma/resolution or fallback per key, floods from the reference points); per-file seeds (populations, histograms, sample means); the pair cap at reference counter starts 999,998/999,999/1,000,000; output names per option; PDF line subsequence, Excel rows, expected/missing channels, photopeak keys. Two asserts now read recorded reference behaviour: `bounded.json` flood sides > 4 × 600 (the reference keeps every side), `zero_energy.json` exception `TypeError`. `test_qc_generalized_layout_for_selected_map` swapped the script's global `SPECS`, which the fixtures no longer read; it now patches `manager_helpers.QC_SPECS` (with the default specs, the 7/41 map would raise `KeyError`). AST (`ROOT` → `REPO`, decorators ignored): 6 of 15 tests identical; the 9 that differ are the 8 golden tests and that patch. Negatives (scratch golden copy): one occupancy count, a fit mu (last digit), a per-file slab count, a pair-cap records-read, a removed output name, a PDF line, flood sides 2400, the exception name, a minimodule energy value, an Excel value, a flood point → each golden test failed. Removing `slab_distribution_SM9_no_data.png` from the option outputs still passes: SM 9 is outside the selected map, and the test documents and filters that difference. Citations: class `003-FR-2/9/14/15/16` (T10, T33); pair cap and input contract add `003-FR-24`; the 8 golden tests add `007-FR-3`; `--fr 003-FR-24` → 2, `007-FR-3` (this file) → 8. `--durations`: `slow` options 7.2 s, per-file seeds 6.9 s, PDF/Excel content 5.2–5.3 s, unavailable fits 5.1–5.7 s (3 runs each); the others ≤ 2.4 s. The scipy fit warnings on sparse keys come from `src/fits.py`; they were there before and are left alone. No fixture folder left (908 before and after; the script's own run left one, removed). `python -m pytest` → 476 passed, 1 skipped, 13 deselected; `-m "not real_data" --slow-limit 5` → 488 passed, 1 skipped. **`petsys_manager_qc_check.py` kept: deletion deferred to T15** (imported at module level by the cli, gui, scope and workflow checks; numeric imports it lazily).
- [x] **T12 — migrate `petsys_manager_bounded_check` and `petsys_manager_scope_check`** (3) → `tests/test_manager_bounded.py`, `tests/test_manager_scope.py`.

  Verified 2026-10-07 (Windows): scripts → bounded `Ran 3 tests … OK`, `PASS: 3 FR-24 scope checks`; `python -m pytest tests/test_manager_bounded.py tests/test_manager_scope.py -m "not real_data"` → 6 passed (3 + 3; default run: 3, the 3 bounded tests are `slow`). `BoundedChecks` on `CLIFixtures` (was a `CLIChecks` subclass loaded by name prefix); the scope classes on `ListmodeFixtures`/`QCFixtures`/`PrivateOutput` (were `ListmodeChecks`/`QCChecks` subclasses and a hand-made temp folder). The scope listmode test compares with `tests/data/golden/listmode/inputs_2500.records.bin` instead of `reference_loop`. **Bounded QC deviation (owner 2026-10-07):** under pytest the QC-with-plots peak grew 84.43 → 87.10 MiB (2,738 KiB > 2 MiB budget) after the listmode test; alone 83.6–85.2 MiB for 2–16 files. Plots off, the same run is 4.57 → 4.59 MiB on every repeat, so the swing is plot rendering (fixed-size 500 × 500 floods, denser histograms), not retention; spec 003 T17 passed only because its 2-file baseline was a 101.33 MiB outlier. The plots+slabs run keeps its per-file counts and ×4 totals without a budget (`bounded(..., budget=None)`); the same request without plots/slabs carries the 2 MiB budget (growth 21 KiB) and must give the same totals. Peaks: calibration 58.97 → 59.12 MiB (155 KiB), LM 228.92 → 228.91 MiB (−1 KiB). AST (`ROOT` → `REPO`, decorators ignored): bounded 5 of 7 definitions identical (the `bounded` budget argument and the QC test differ), scope 4 of 6 (`setUpClass` on `PrivateOutput`, the golden listmode test). Negatives: one golden record bit flipped → the scope listmode test failed; a QC sampler that keeps each input's bytes → the plots-off budget failed (7.55 → 16.53 MiB, +9,191 KiB). Citations: bounded `003-FR-15/16` (T17), scope `003-FR-24` (T24), the scope listmode test adds `007-FR-3`; `--fr 003-FR-24` → 6, `003-FR-15` (these files) → 3. `--durations`: `slow` bounded QC 92 s, LM 65 s, calibration 8 s; scope ≤ 1.8 s. No fixture folder left (908 before and after; the scripts' own runs left 4, removed). `python -m pytest` → 481 passed, 1 skipped, 16 deselected; `-m "not real_data" --slow-limit 5` → 496 passed, 1 skipped, 1 deselected (the `real_data` test). Both scripts deleted (only `numeric_check --bounded/--scope` imported them, lazily).
- [x] **T13 — migrate `petsys_manager_cli_check` (15) and `petsys_manager_workflow_check` (15)** → `tests/test_manager_cli.py`, `tests/test_manager_workflow.py`; the "never imports `gui_cornell`/`scripts*`" checks kept.

  Verified 2026-10-07 (Windows): scripts → cli `FAIL: 15 selected checks` (14 passed; see the dependency check below), workflow `PASS: 15 selected checks`; `python -m pytest tests/test_manager_cli.py tests/test_manager_workflow.py -m "not real_data"` → 30 passed (15 + 15; default run: 22, 8 `slow`). `CLIChecks` on `CLIFixtures` (its fixture/launcher methods were already in `manager_helpers`, equivalent up to `CalibrationFixtures`/`ListmodeFixtures`/`QCFixtures`); `WorkflowChecks` on `PrivateOutput`, its `assets()` builds `ListmodeFixtures.at(...)`/`QCFixtures.at(...)` instead of instantiating `ListmodeChecks`/`QCChecks`; `ToolWorld`, `sha256`, `write_output` from `manager_helpers` (equivalent, docstring only). The CLI test's lazy `compact_record` import from the numeric script is a module import from `manager_helpers`. **Stale dependency check:** `test_cli_declared_dependencies_cover_actual_imports` compared the working `process_petsys.yml` with the revision before reportlab was declared and required only the reportlab line added; since spec 006 T1 (`e8b6856`, pytest 8.3.5 + exceptiongroup) the script fails on that last assertion. The test now compares the commit that declared reportlab (`7c3c67b`, adds only `- reportlab==4.4.9`) with its parent; the coverage half (every third-party import of the loaded `src` modules is declared, reportlab version equal) still reads the working file. The "never imports `gui_cornell`/`scripts*`" audits are unchanged: negatives (temporary untracked `src/cornell/_t13_probe.py`) `import gui_cornell` → failed; a `"../scripts_cornell/x.py"` string → failed; probe removed. AST (`ROOT` → `REPO`, decorators ignored): cli 17 definitions identical, 2 differ (the `compact_record` import, the dependency comparison); workflow 29 identical, 2 differ (`setUpClass`, `assets`). Citations: cli class `003-FR-2/4/9/12/14/16` (T11), target limit adds `003-FR-21` (T25.2), LM seed `003-FR-15/22` (T25.5), QC seed `003-FR-15` (T33), in place `003-FR-13` (T29.2), cancellation `003-FR-24`; workflow class `003-FR-5/7/9/10/11/13/14/16` (T12), pipeline and elapsed text add `003-FR-1` (T25.1), limit mode `003-FR-15/21`, compact real-CLI pipeline `003-FR-22`, conversion structure check `003-FR-24`, run names `003-FR-21` (T32); in these files `--fr 003-FR-1` → 2, `003-FR-21` → 3, `003-FR-22` → 2, `003-FR-24` → 2. `--durations` (3 runs): `slow` cli QC seed 12.6–12.9 s, QC report 11.4–11.9 s, LM child 10.9–11.9 s, source audit 6.8–6.9 s, corrupted inputs 5.3–5.4 s, workflow real CLI 11.0–11.1 s, compact pipeline 6.2 s, faults 5.5–5.6 s; the others ≤ 4.7 s. No fixture folder left (908 before and after; the scripts' own runs left 2, removed). `python -m pytest` → 505 passed, 1 skipped, 24 deselected in 86 s; `-m "not real_data" --slow-limit 5` → 528 passed, 1 skipped, 1 deselected. **Deletion deferred:** `petsys_manager_cli_check.py` to T14 (`petsys_manager_checkout_check.py` builds its fixtures with `CLIChecks` inside two tests), `petsys_manager_workflow_check.py` to T15 (`petsys_manager_gui_check.py` imports `ToolWorld` at module level).
- [x] **T14 — migrate `petsys_manager_checkout_check` (5) and `petsys_manager_linux_check` (17)** → `tests/test_manager_checkout.py` (Windows and Linux; owner 2026-10-07: the audit is cross-platform, FR-7 keeps `linux` for Linux-only tests), `tests/test_manager_linux.py` (`linux`); Windows shows the Linux tests skipped with reason; Linux counts confirmed at T24.

  Verified 2026-10-07 (Windows): scripts → checkout `--tracked-runtime` `PASS: 5 checkout checks`; linux `--all` → exit 2 `PENDING: Linux dummy checks require Linux` (no count on Windows; spec 003 T22 recorded 17). `python -m pytest tests/test_manager_checkout.py tests/test_manager_linux.py -m "not real_data" -rs` → 5 passed, 17 skipped `linux: runs only on Linux (Cornell PC), not win32` (default run: 3 passed, 2 `slow`). Checkout: `CheckoutChecks` on `PrivateOutput` (`fixture_prefix = "pm-checkout-"`, replaces `private_root()`, whose folder was never removed); the two copy tests build their requests with `CLIHelper` (a `CLIFixtures` builder, `__test__ = False`, its folder removed by `addCleanup`) instead of instantiating `CLIChecks` (whose folders the script left behind); `METADATA` from `manager_helpers`. **Uncommitted runtime files (owner 2026-10-07):** the copy is `git archive HEAD`, so `setUpClass` skips the class with `checkout: runtime files differ from HEAD (<files>); commit them to audit the checkout` while a closure file is modified or untracked (the script failed instead: spec 003 T24 recorded 4/5 until committed). Negatives: a comment appended to `src/cornell/__init__.py` → 5 skipped naming it; an untracked `src/cornell/_t14_probe.py` → 5 skipped naming it; both restored, `git status --short src exe_programs` clean. Linux: the four classes unchanged, each marked `linux` (pytest skips before `setUpClass`, so the `prctl` subreaper setup never runs on Windows); fixtures stay under `~/.cache/process_petsys` as before. AST (`ROOT` → `REPO`, decorators ignored): checkout 14 definitions identical, 3 differ (`setUpClass`, the two copy tests); linux 30 of 30 identical. Citations: checkout `003-FR-2/16/18` (T18); linux process groups `003-FR-4/5/7/10/16` (T3), DAQD `003-FR-5/6/7/16` (T6), acquisition `003-FR-5/7/8/9/16` (T7), PETsys Python `003-FR-3/4/16/23` (T22); in these files `--fr 003-FR-18` → 5, `003-FR-23` → 1, `003-FR-6` → 7. `--durations` (3 runs): `slow` copy CLI 17.2–17.9 s, copy manual LM 6.0–6.1 s; the others ≤ 1.6 s. No fixture folder left (908 before and after; the checkout script's own run left 3, removed). `python -m pytest` → 510 passed, 18 skipped (17 linux + the artifacts symlink test), 26 deselected in 90 s; `-m "not real_data" --slow-limit 5` → 535 passed, 18 skipped, 1 deselected. `petsys_manager_checkout_check.py` and `petsys_manager_cli_check.py` (deferred from T13; checkout was its last importer) deleted. **`petsys_manager_linux_check.py` kept until T24**: it is the only Linux pass-count evidence, compared with the pytest count on the Cornell PC (nothing imports it; `petsys_manager_check.py` only reads its path in an already migrated test).
- [x] **T15 — migrate `petsys_manager_gui_check`** (then delete the deferred `petsys_manager_daqd_check.py`, `petsys_manager_check.py`, `petsys_manager_listmode_check.py`, `petsys_manager_qc_check.py` and `petsys_manager_workflow_check.py`) (28; modes `--acquisition --conversion --processing --shell`, `--real`) → `tests/test_manager_gui_{acquisition,conversion,processing,shell}.py` (`gui`) + `RealSelectionChecks` as `real_data` (January and September split selection, paths via `PETSYS_DATA_DIR`). Shared bases and fakes → `tests/manager_gui_helpers.py`; tracked `tests/data/configs/cornell_{september,january}.yaml` (owner 2026-10-07).

  Verified 2026-10-07 (Windows):
  - **Counts.** Script `--all` → `PASS: 28 selected checks` (126 s; shell 7, acquisition 7, conversion 6 + calibration route 1, processing 6, real 1). `python -m pytest tests/test_manager_gui_{shell,acquisition,conversion,processing}.py -m "not real_data" -n 0` → 27 passed. `-m real_data` with `PETSYS_DATA_DIR` set → 1 passed (6.5 s); unset → skipped with `real_data: PETSYS_DATA_DIR is not set`. 27 + 1 = 28.
  - **slow (7).** Shell prerequisites ~7 s; conversion split/duration ~14 s; real selection ~6 s; processing manual calibration ~11 s, event limit ~6 s, QC presets ~5 s, metadata ~13 s.
  - **Code.** AST comparison: 99 of 102 definitions identical. The three that differ:
    - `GUIBase.setUpClass`: `PrivateOutput` folder, and a skip with `gui: no display available (...)` when Tk cannot start.
    - `ProcessingChecks.setUpClass`: `ListmodeFixtures.at`/`QCFixtures.at` replace the script classes.
    - The real test: `cornell_data` fixture with `real_data_file`, tracked configs.
  - **Helpers and configs.** `manager_helpers` still imports without Tk (`test_manager_helpers` passes). `.gitignore` re-includes `tests/data/configs/`, which the `configs/` rule had matched.
  - **Negatives** (in-process patches, no file edits): `tkinter.Tk` raising `TclError` → 13 skipped with the reason; `gui.TABS` reordered → tab test fails (`Tuples differ`).
  - **Fixed alongside (plan Decisions).**
    - `--capture=sys` in `addopts`. Under fd capture Tk intermittently failed to start ("Can't find a usable init.tcl"): about once per serial GUI run, 5 times in one parallel full run. Outside pytest, fd 1/2 swaps reproduce it: 2 of 300 and 2 of 600 roots, against 0 of 1,100 without swaps.
    - `test_manager_daqd.py`: `service()` registers `service.close`. Six leftover `petsys-daqd-1` threads had failed the GUI close test in a shared process.
  - **Deleted.** `petsys_manager_gui_check.py`, `_daqd_check.py`, `petsys_manager_check.py`, `_listmode_check.py`, `_qc_check.py` and `_workflow_check.py` (backups in the session scratchpad). `_calibration_check.py` and `_numeric_check.py` are kept until T17: `real_check.run()` imports the calibration check.
  - **Suite.** `python -m pytest` → 538 passed, 18 skipped in 46 s. `-m "not real_data"` → 569 passed, 18 skipped in 196 s. `-m "not real_data" -n 0 --slow-limit 5` → 569 passed, 18 skipped in 634 s. Fixture folders 908 before and after; the script's six `gui_*` folders from earlier runs remain.
- [x] **T16 — retire `petsys_manager_reference_check`** (FR-1, FR-3). Its synthetic contracts assert hand-written expected values, so they need no golden file (T8): each moves onto the tracked code that implements it or is already covered by a T9–T11 test; the contracts on `cornell_slab_en_cal_fixed_position.py` (4, plus the calibration parts of 3 mixed ones) and the `gui_cornell`/PETsys-source inventory are retired (spec Clarify).

  **Done when:** each `--synthetic` contract (17) is mapped in this file to the test that replaces it or listed as retired (spec Clarify); script deleted.

  Verified 2026-10-07 (Windows): script `--synthetic` → `PASS 17/17`. Each contract, by its script label:
  1. **Fixed/compact coincidence/group bytes** → new `tests/test_read_fixed.py` (6: pairs at batch sizes 1/2/100, groups, padding slots, header-only and short files) and `test_read_compact.py::test_group_round_trip_has_no_second_detector` (new), with `test_round_trip_returns_written_pairs`.
  2. **Region boundaries, calibration/LM edge policy** → new `test_manager_listmode.py::test_listmode_region_clipping_versus_calibration_exclusion`: the same points and expected regions on the Manager's `calibration.region_ids` (replaces the fixed-position reference's `compute_region_id_numba`) and `listmode.compute_region_vectorized`.
  3. **Slab convention, scalar/vectorized** → `test_cornell_slab.py::test_slab_convention` (same 7 cases on the tracked Cornell map).
  4. **Seeded one-channel slab randomness** → `test_slab_convention` "middle, one channel" (seeded, both slabs, x per slab); seeded replay: `test_listmode_parallel_files_with_per_file_seeds`, `test_qc_parallel_files_with_per_file_seeds`.
  5. **Calibration side/group populations** → retired (fixed-position reference).
  6. **Calibration 4M cap between side chunks** → retired (fixed-position reference); the Manager's limits: `test_passing_event_limit_matches_reference_semantics`, `test_target_limit_counts_kept_sides`.
  7. **QC resolved-pair vs occupancy denominators** → `test_qc_manual_pair_side_hit_counts_and_occupancy`.
  8. **QC 1,000,001 boundary** → `test_qc_per_file_stopping_before_at_after_limit`.
  9. **Position .encal schema and KevConverter** → retired (the legacy writer is the fixed-position reference); the reader: `test_kev_converter.py::test_cornell_position_known_factor`, `test_outputs_read_by_kevconverter_loader_and_never_overwrite`.
  10. **Fit histogram/gate/fallback** → calibration part retired; QC part (150 bins over 0–250 a.u., fallback unavailable): `test_qc_histograms_fits_occupancy_floods_match_reference`, `test_qc_unavailable_fits_and_raw_units`.
  11. **Reference Gaussian fit on a known spectrum** → retired (fixed-position reference).
  12. **LM header/record sizes and offsets** → `test_listmode_structures_and_every_offset`.
  13. **LM header bytes** → `test_listmode_legacy_metadata_header_bytes_equal_reference`, `test_listmode_supplied_header_fields_decode_independently`.
  14. **LM selected/swapped pair bytes** → `test_listmode_controlled_pair_orientation_energy_and_timestamp`.
  15. **Calibration/LM vs QC channel-energy cut** → calibration part retired (the Manager's per-slab calibration applies `en_min_ch` by design, spec 003 T20); LM: new `test_listmode_does_not_apply_the_channel_energy_cut` (a 0.125 a.u. fourth channel counts towards `min_ch` 4: 1 record; without it: 1 `min_channels` rejection); QC: `test_qc_manual_pair_side_hit_counts_and_occupancy` (0.1 a.u. hit cut, 42 vs 43 hits), `test_qc_reader_equals_reference_reader`.
  16. **QC Excel/PDF schemas** → `test_qc_pdf_excel_and_plot_content_match_reference`.
  17. **Sources unchanged** → retired (fingerprints of the inventory, spec Clarify).

  Mapped 12 (5 with new tests), retired 5 (4 fixed-position contracts, the inventory). The calibration part of mixed contract 2 moved onto the Manager's region rule (same expected regions) instead of being retired. Citations: readers `003-FR-16`, the two listmode tests inherit the class ids (`003-FR-12` among them). Negatives (scratch copies of `src`): LM dropping hits below 0.2 a.u. → channel-cut test fails; LM excluding instead of clipping → region test fails; calibration clipping instead of excluding → region test fails; fixed reader with sides swapped → 4 failed; compact group header read as `2B` → group test fails. `--durations`: new tests ≤ 0.94 s (channel-cut test). Script deleted after T17's baseline run (its `--real` mode runs `petsys_manager_real_check`).

- [x] **T17 — migrate `petsys_manager_real_check`** (FR-4) → `tests/test_manager_t19_real.py` (`real_data`, `slow`): spec 003 T19 calibration/LM/QC on the January files vs `tests/data/baselines/t19_jan2026_{manifest,result}.json`; manifest paths relative to `PETSYS_DATA_DIR`, January config. Then also delete the deferred `petsys_manager_calibration_check.py` and `petsys_manager_numeric_check.py` (`real_check.run()` imports the calibration check for `EN_MIN` and `Oracle`; it imports the numeric check at module level; kept from T15).

  **Done when:** with both variables set on the owner's PC the tests pass and reproduce the 2026-10-05 baseline values; unset → skipped naming the variable; `scripts/petsys_manager_t19_baseline_{jan2026,cornell_f18}.json` moved/retired; script deleted.

  Verified 2026-10-07 (Windows, `PETSYS_DATA_DIR=C:\Users\dsanchez\Desktop\data`, `PETSYS_CAL_DIR=<repo>\encal_files`):
  - **Script.** `petsys_manager_reference_check.py --real` with the 2026-10-05 compact-only manifest → first `FAIL setup: InputError: Not a mapped time channel: 135424`: `configs/cornell_full_system.yaml` selects `maps/cornell_map_full_system.yaml`, the September layout since 2026-10-07 (spec 006 T0). With `config: tests/data/configs/cornell_january.yaml` → `PASS 5/5` (4 fixed-input checks n/a), every evidence value equal to the 2026-10-05 run (only `head`, `input_bytes` from the other config file, and `report_directory` differ), input SHA-256 equal. So the 2026-10-05 run used the January layout.
  - **Tests.** `python -m pytest tests/test_manager_t19_real.py -m real_data -n 0` → 4 passed in 1,350 s:
    - input SHA-256 equal to the 2026-10-05 run's (9 files; the config is not an input here);
    - per-slab calibration (P = 1) byte-identical to the reference-written `.encal`/status, 7,680 rows, status counts, per-file rows;
    - position calibration (P = 5, 400,000 passing events per file): 2,911,163 accepted sides, status counts, 38,400 rows;
    - QC with plots and slabs: 6,000,006 accepted pairs, per-file rows, rejections, slab flags, fit statuses (6,690 fitted, 74 sparse, 12 failed), 30 floods, the 496 report files.
    Each processing test checks its inputs unchanged afterwards (fixture).
  - **Counts.** Script 5 vs pytest 4: the script's last check (inputs unchanged) runs inside each processing test; its reference-source and checkout fingerprints are dropped (no reference is loaded, FR-3). Not compared: the 4 n/a fixed-input checks (no fixed files, spec Clarify).
  - **Replaced.** The reference-function comparisons (position oracle, QC `process_file`/fits) become the 2026-10-05 values they produced (spec Clarify, T17).
  - **Skips.** Unset → 4 skipped `PETSYS_DATA_DIR is not set`; data set, cal unset → `PETSYS_CAL_DIR is not set`; a cal folder without the file → skipped naming the `.encal` and `PETSYS_CAL_DIR`.
  - **Negatives** (scratch copies of `src`, real data): calibration `FIT_BINS` 100 → 99 → per-slab test fails (`(256, 0)` 86.013/7.377 → 86.017/7.337); `EDGE_MULTIPLIER` 1.8 → 1.7 → position test fails (borrowed 121 → 130, no_values 10,574 → 10,566); QC `FIT_MIN_PEAK` 20 → 21 → QC test fails (sparse 74 → 78). `MIN_EVENTS` 200 → 201 changed nothing (no key at the boundary): the position test asserts counts, not fitted values.
  - **Durations.** QC 1,192 s (`run_qc` 482 s + `write_report` 710 s, as in the script run), per-slab 131 s, position 23 s, inputs 4 s; the module is `real_data` + `slow`.
  - **Baselines.** `tests/data/baselines/t19_jan2026_result.json`: verbatim copy of `petsys-manager-real-20261005T070838Z-9fcef188/real_baseline.json` (SHA-256 `408629b1b09586e4dc446a05fb75f26afa84d5425ad028ab58a781132a860f95`); `t19_jan2026_manifest.json`: its compact-only manifest with relative paths and provenance.
  - **Deleted** (backups in the session scratchpad): `petsys_manager_reference_check.py` (T16), `petsys_manager_real_check.py`, the deferred `petsys_manager_calibration_check.py` and `petsys_manager_numeric_check.py`, `petsys_manager_t19_baseline_jan2026.json`, `petsys_manager_t19_baseline_cornell_f18.json`. No remaining script imports them.
  - **Suites (T16 + T17).** Default `python -m pytest` → 549 passed, 18 skipped in 38 s; `-m "not real_data"` → 580 passed, 18 skipped in 166 s; `-m "not real_data" -n 0 --slow-limit 5` → 580 passed, 18 skipped in 566 s. Fixture folders 908 before and after; `git diff --check` clean.

- [x] **T26 — Bounded measurements in a fresh interpreter** (FR-2, FR-5; owner 2026-10-07, found in T15). In a serial full run `test_bounded_listmode_with_debug_end_to_end` failed deterministically with growth 5,112 KiB > 2,048: a `tracemalloc` snapshot diff showed one 5,120 KiB block from `pathlib` `sys.intern`, the interned-string table resizing during the 8-file run after earlier tests had filled it (alone: −10 KiB). Each action's warm/1/BASE/COPIES runs now go to one child interpreter (`MEASURE`); assertions and budgets are unchanged.

  **Done when:** the bounded file passes alone and after the nine earlier files that made it fail; a retention negative (audit hook keeping each opened `bounded_*.ldat` in the child) fails it; serial full run passes.

  Verified 2026-10-07 (Windows):
  - **Alone.** Calibration +163 KiB, LM −3 KiB, QC bare +29 KiB; QC with plots +3,900 KiB, no budget (T12).
  - **After the nine earlier files.** LM +1 KiB (was +5,112 KiB).
  - **Negative.** A `sitecustomize` audit hook (via `PYTHONPATH`) keeps each opened `bounded_*.ldat` in the child → calibration +38,111 KiB and LM +18,083 KiB, both failing.
  - **Serial full run.** `-m "not real_data" -n 0 --slow-limit 5` → 569 passed.

## LDAT group (specs 001, 002)

- [x] **T18 — LDAT helpers and tracked January config** (FR-6). `tests/ldat_helpers.py` (`fixture_files`, `write_pairs`, `build`, `CONFIGS`, `RECOVERY_CASES`, side/LDAT writers); `tests/data/configs/cornell_january.yaml` (added in T15).

  **Done when:** helpers import; `CONFIGS` resolves IMAS (tracked `configs/imas_1DAQ.yaml`) and Cornell (tracked copy) on a fresh clone; smoke test writes one fixture per system.

  Verified 2026-10-09 (Windows):
  - **Helpers.**
    - `CONFIGS`: IMAS `configs/imas_1DAQ.yaml`, Cornell `tests/data/configs/cornell_january.yaml`, a verbatim copy of the local `configs/cornell_full_system_old.yaml`. The local `imas_1DAQ.yaml` differs from HEAD only in line endings.
    - Copied from the scripts: `fixture_files`, `write_pairs` (inspector); `module_channels`, `side`, `RECOVERY_CASES` (scale); `BASE`, `cases`, `is_time`, `build` (views).
    - New: `write_ldat`, which is `helpers.write_ldat` plus the scale script's `tail` bytes; `destroy`, which cancels a hidden window's pending `after` jobs before destroying it, as in T15.
    - Left in the scripts for T21/T22 (each has one user): scale `_fixture_pairs`, `_expected`, `_unmapped_channel`; views `EXPECTED_ROW`, `_expected_states`.
  - **AST** (underscore prefixes dropped): 7 of 9 definitions identical.
    - `fixture_files` reads `CONFIGS`. For Cornell it read `configs/cornell_1cassettes.yaml`: untracked, a 1-cassette map outside the data contract (spec Clarify T18, T19).
    - `is_time` imports at module level.
  - **Tests.** `python -m pytest tests/test_ldat_helpers.py` → 8 passed:
    - imports without Tk;
    - both configs and their maps are tracked (`git ls-files`);
    - `fixture_files` per system loads with its calibration, picks channels on SM 0 and 1, and its 3 records read back;
    - `side` and `write_ldat` with `RECOVERY_CASES` plus a tail (file size);
    - `build` per system (the half-populated case is Cornell-only).
  - **Negative.** Cornell `CONFIGS` → the untracked `configs/cornell_full_system_old.yaml` → the tracked test fails.

- [x] **T19 — migrate `ldat_inspector_check --selftest` (59), `ldat_revision_check` (14), `ldat_ports_check` (12)**.

  Verified 2026-10-09 (Windows):
  - **Scripts.**
    - Inspector `--selftest` → `PASS: 59/59`. With Cornell `fixture_files` on the January config (scratch copy) → `PASS: 59/59`.
    - Revision → `PASS: 14 revision checks`.
    - Ports → `FAIL 11/12`: the Cornell `Dataset.sm_ports` (January config) is compared with `maps/cornell_map_full_system.yaml`, which has held the September layout since 2026-10-07; `mod_feb_map` differs for 10 of 30 SMs. With the January map name (scratch copy) → `PASS 12/12`. The test compares with the map the config selects (spec Clarify T18, T19).
  - **Tests.** `python -m pytest tests/test_ldat_inspector.py tests/test_ldat_revision.py tests/test_ldat_ports.py -m "not real_data"` → 85 passed (59 + 14 + 12). Each script `check` or `checks += 1` block is one test.
    - **Inspector:** a module fixture builds each system's files once; 25 tests per system, 2 Cornell-only tests, and 7 fit tests sharing one seeded generator in the script's draw order.
    - **Revision:** 3 engine tests and 4 GUI steps per system. The GUI steps run in order on one hidden window, so a module fixture runs them once and records what each step saw. One window per system (the script reused one).
    - **Ports:** 3 synthetic, 3 map, 3 overview (`gui`) and 3 report tests. The unpopulated-tile test is now unconditional: the January config declares half-populated SMs.
    - **Fixture roots:** `tempfile`, as in the scripts. The report wraps provenance lines at 118 characters; under `tmp_path_factory` the module-PDF provenance tests failed on the longer paths.
  - **Citations:** inspector and revision `001-FR-15` plus one FR per check (FR-2–5, 7–13, 16, 19–24, 26; channel status `002-FR-5`); ports `002-FR-23`. In these files `--fr 001-FR-15` → 73, `001-FR-23` → 14, `001-FR-22` → 12, `002-FR-23` → 12.
  - **Durations:** the Cornell full report takes 6.2 s with the 30 SMs of the January map (3 SMs in the script's 1-cassette map), so it is marked `slow`; all others ≤ 1.5 s. The default run selects 84.
  - **Negatives** (scratch copies):
    - `TIMESTAMP_SECONDS` 1e-12 → 1e-9: 2 timing tests fail;
    - "Observed timestamp spans" title changed: 2 module-PDF tests fail;
    - raw energy range 300 → 400: 4 revision GUI tests fail;
    - "Unavailable calibrated energy" text changed: 2 tests fail;
    - "SLAVE" → "Slave": 2 ports tests fail.
    - A "No singles" text change was not caught: the phrase is also on another page of the same report.
  - **Tk:** a destroyed window's pending `after` jobs ran in the next window (7 `invalid command name` lines; the script shows 0, with one window until exit). `ldat_helpers.destroy` cancels them → 0.
  - **Order-dependent T13 test.** The first serial full run had 1 failure: `test_manager_cli.py::test_cli_declared_dependencies_cover_actual_imports` → `'llvmlite' not found … imported by ['src/ldat_inspector/fastread.py']`. The test scanned every `src/` module in the process's `sys.modules`. The script ran in its own process, which loaded only the CLI's modules. In a serial pytest run, the new LDAT tests load `src/ldat_inspector` first; in parallel runs that module happened to be on another worker. The test now loads its own imports in a fresh interpreter and scans those. It passes alone and after the LDAT tests. Negative (scratch copy): `import llvmlite` in `src/cornell/cli.py` → fails, naming it. Not changed: `fastread.py` imports `llvmlite`, which `process_petsys.yml` does not declare (it is installed with `numba`).
  - **Deleted** (backups in the session scratchpad): `ldat_revision_check.py`, `ldat_ports_check.py` (no importer). **`ldat_inspector_check.py` kept: deletion deferred to T21** (the gui, issue and pair_choices scripts import `fixture_files` and `write_pairs` at module level).
  - **Suites** (T18 + T19): `python -m pytest` → 646 passed, 18 skipped in 47–55 s (+92 tests, +5 `test_infra.py` scan items); `-m "not real_data"` → 678 passed, 18 skipped in 173 s; `-m "not real_data" -n 0 --slow-limit 5` → 678 passed, 18 skipped. Fixture folders 908 before and after; `git diff --check` clean.
- [x] **T20 — migrate `ldat_processing_check` (17) and `ldat_gui_check` (29)** (`gui`; Linux close-child part `linux`).

  Verified 2026-10-09 (Windows):
  - **Scripts.** Processing → `PASS: 17/17 processing checks passed`; GUI → `PASS: 29/29 hidden-GUI checks passed`. With Cornell `fixture_files` on the January config (scratch copies) the GUI script also gives `PASS: 29/29`.
  - **Tests.** `python -m pytest tests/test_ldat_processing.py tests/test_ldat_gui.py -m "not real_data"` → 46 passed (17 + 29).
    - **Processing:** 9 engine tests on a module fixture (four copies of the IMAS mixed fixture). The fast reader's cancel and progress globals are reset after each test.
    - **Processing GUI:** 7 tests read `gui_run`, which runs the script's steps once in order on one hidden window with real workers: estimate, decline, whole-file run, Cancel.
    - **Close mid-run:** the child is `python -c` calling `ldat_helpers.close_child`; spawn workers import `ldat_helpers.slow_reader`.
    - **GUI:** 1 tab test, then 14 steps × 2 systems on one shared window, as in the script. `run` replays the steps per system and records each step's observations, or the exception it raised, which the step's test re-raises.
  - **Linux marker not applied.** The task named a Linux-only close-child part, but the script runs it on Windows and it passes there (3.7 s of the 5 s limit). It stays unmarked and runs on both platforms (FR-7).
  - **Waiting.** The script waited with a new `BooleanVar` per wait. In a closure cycle, that variable was collected on a worker thread (2 `RuntimeError: main thread is not in main loop` warnings; the script shows 0). The test waits with `ldat_helpers.pump` → 0.
  - **Helpers.** Added to `ldat_helpers`: `fixture_pairs`/`unmapped_channel`, used by processing and scale; the slow reader; `pump`; `workbench`; `console`; `close_child`. AST (underscore prefixes dropped, `_app` → `workbench`, `_log` → `console`): 6 of 7 identical; `slow_reader` imports `fastread` locally, so the module imports without numba.
  - **Citations:** processing `002-FR-2`, plus `002-FR-3` (progress, worker memory, more than 2 workers) and `002-FR-4` (both cancels, close), 3 each. GUI `001-FR-15`, plus one FR per step (FR-1, 4, 6, 7, 11–13, 19, 21–24; status `002-FR-5`). `--fr 002-FR-2` → 17, `001-FR-15` → 29, `001-FR-22` → 6.
  - **Durations.** Setup: GUI `run` 7.9–10.1 s (IMAS) and 5.6–6.6 s (Cornell); processing `gui_run` 6.6–6.9 s. The tests on those fixtures are marked `slow` (28 + 7). Default run: 11 of 46 (9 engine, close 3.7 s, tabs).
  - **Negatives** (scratch copies):
    - "upper bound" estimate label removed: 1 fails;
    - "previous dataset kept" log text changed: the Cancel test fails;
    - `_close` without setting the cancel event: the close test fails (workers run on);
    - flood empty bins black instead of white: 2 fail.
  - **Parallel full runs** (spec Clarify T20). The first runs after this migration failed intermittently.
    - **Close test:** it took 6.2–7.3 s under load against its 5 s bound; alone it takes 3.7 s. Owner: the bound is asserted only outside xdist workers. The exit and exit code 0 are always asserted.
    - **`OSError: [Errno 22]`:** 2 of 4 runs failed with it in a listmode child of existing tests, T14 checkout and T13 workflow. Those requests had `workers: 22` and 2 inputs, and `segments/` stayed empty, so the child died while its workers were spawning.
    - **Cause:** sampling `FreeVirtualMemory` showed free commit falling from 34 GB to 0.07 GB within 4 s of the run starting (67.8 GB limit, 64 GB RAM). Without the two T20 files it fell to 0.03 GB as well, so the suite was already at the limit; T20's extra spawned processes made the failure visible.
    - **Per process:** numpy commits 756 MiB, `src.cornell.listmode` 1,568 MiB (OpenBLAS, 24 threads); with one thread, 18 and 90 MiB.
    - **Fix:** `conftest.py` sets `OPENBLAS/OMP/MKL_NUM_THREADS=1` with `setdefault` (owner). `test_infra.py` gains 2 cases: a parallel worker and its child see `1`; an explicit `OPENBLAS_NUM_THREADS=4` is kept. Negative: without the `setdefault` loop → both fail.
  - **Deleted** (backups in the session scratchpad): `ldat_processing_check.py` (only its own `--close-child` referred to it), `ldat_gui_check.py` (no importer).
  - **Suites:**
    - `python -m pytest` → 661 passed, 18 skipped in 52 s;
    - `-m "not real_data"` → 728 passed, 18 skipped in 175–181 s, twice;
    - `-m "not real_data" -n 0 --slow-limit 5` → 728 passed, 18 skipped in 621 s.

    Free commit stayed at or above 27.2 GB through the default and both parallel runs. Fixture folders 908 before and after; `git diff --check` clean.
- [x] **T21 — migrate `ldat_scale_check` (65 + `--real` T9 counts), `ldat_unpopulated_check` (6 + `--real`), `ldat_issue_check` (8, fixture only), `ldat_pair_choices_check` (15, fixture only)**; retired real modes listed (`--whole`, issue probe, pair_choices `--real`).

  Verified 2026-10-09 (Windows):
  - **Scripts.**
    - Scale `--selftest` → `PASS: 65/65`; `--real` → `PASS: 15/15` (30 s).
    - Unpopulated → `PASS: 6/6`; `--real` → `PASS: 8/8`.
    - Pair choices → `PASS: 15/15`; issue → `PASS: 8/8`.
    - Scale and unpopulated already read `configs/cornell_full_system_old.yaml`, of which the January config is a verbatim copy. Issue and pair choices took `fixture_files` from the inspector script (1-cassette config). With `ldat_helpers.fixture_files` (scratch harness) they give 8/8 and 15/15.
  - **Tests.** `python -m pytest tests/test_ldat_scale.py tests/test_ldat_unpopulated.py tests/test_ldat_pair_choices.py tests/test_ldat_issue.py -m "not real_data"` → 94 passed (65 + 6 + 15 + 8). Each `check` is one test.
    - **Scale:** a module fixture writes each system's fixture files and runs the reference reader on them once. Per system: 8 (IMAS) or 9 (Cornell) reference tests, 6 T3 storage tests, 7 fast = reference tests; T16 adds 2 for IMAS and 19 for Cornell; plus "fast reader available".
    - **Unpopulated:** reads `CONFIGS["CORNELL"]`.
    - **Pair choices:** `gui` records its 6 consecutive window steps.
    - **Issue:** `steps` runs its 6 window steps in order and records each step's exception. The fit-stability and PDF-line tests run on their own.
  - **Real data** (`PETSYS_DATA_DIR`, January compact splits, January map, raw a.u., ≥ 4 channels, ≥ 0.2 a.u.): scale 15 passed, unpopulated 2 passed. Unset → skipped `real_data: PETSYS_DATA_DIR is not set`.
    - **Scale `TestRealCornell`:** the T9 reference results on the six 80,000-pair prefixes are computed once (6 workers): 6 reference, 6 fast, 200,000-pair equality, 2 T16.
    - **Unpopulated:** the 300,000-pair prefix of `00000003` is read once.
  - **AST** (underscore prefixes dropped):
    - Identical: scale `compare`, `cornell_settings`, `expected`, `reindexed`; issue `comparison_overlay`, `descendant`, `gaussian_overlay`; unpopulated `expected`.
    - Differ:
      - `fast_reader` imports the fast reader without the script's `None` fallback; "fast reader available" asserts it.
      - Issue `popup` and `uniformity_colours` import `customtkinter`/`ttk` locally.
      - `single_module_dataset`'s docstring is rewrapped.
  - **Citations:**
    - Scale: `002-FR-1`; T3 `002-FR-2/11`; T16 `002-FR-19`; real `007-FR-4`.
    - Unpopulated: `bug-unpopulated-minimodules` (spec 002 B3), `002-FR-5`.
    - Pair choices: `002-FR-22`.
    - Issue: `001-FR-15`, plus `001-FR-8/10/12/16/17/18/20/27` per check.
    - Counts: `--fr 002-FR-22` → 15, `bug-unpopulated-minimodules` → 8, `001-FR-17` → 2.
  - **Durations.** Fixture setups ≤ 1.4 s; no default-run test is `slow`.
    - Real: the 200,000-pair test 14.5 s (`slow`).
    - Real setups: T9 reference 6.8 s, T16 6.0 s, unpopulated prefix 21 s.
  - **Negatives** (scratch copies):
    - fast reader's `ZeroDivisionError` label changed: 2 scale tests fail;
    - a malformed `unpopulated_minimodules` ignored: 1 fails;
    - whole-file cap raised by 100: 1 fails;
    - `_raise_dialog` without `transient`: 2 fail;
    - two uniformity fills equal: 1 fails.
  - **Tk:** 0 `invalid command name` or main-thread lines.
  - **Retired or not migrated:** scale `--whole`, issue `--real`, pair choices `--real` (spec Clarify); issue `--visible` (owner 2026-10-09, spec Clarify T21, T22).
  - **Deleted** (backups in the session scratchpad):
    - `ldat_unpopulated_check.py`, `ldat_issue_check.py`, `ldat_pair_choices_check.py`;
    - `ldat_inspector_check.py` (deferred from T19; its last importers were the issue and pair-choices scripts).
    - **`ldat_scale_check.py` kept until T22:** the views script imports it inside two functions.
- [x] **T22 — migrate `ldat_views_check`** (171; `--real` retired) → `tests/test_ldat_views_{status,supermodule,overview,coincidences,…}.py` (`gui`), split at existing section boundaries.

  Verified 2026-10-09 (Windows):
  - **Script.** `PASS: 171/171` in 103 s (with `PYTHONIOENCODING=utf-8`: piped under cp1252, a θ in a label raised `UnicodeEncodeError` after 34 checks). Each run left 9 `%TEMP%\ldat_*` folders (`tempfile.mkdtemp`, never removed).
  - **Labels.** An instrumented run recorded each `check` label per section: 18 section functions plus the checks inline in `main`.
  - **Owner files in the default run** (spec Clarify T21, T22). Three checks read untracked files:
    - the list-mode limits copies → `real_data` (`PETSYS_DATA_DIR`);
    - the resolved `.encal` and its status sidecar → `real_data` (`PETSYS_CAL_DIR`);
    - the gitignored repo-root limits files → retired.
  - **Tests.** `python -m pytest tests/test_ldat_views_*.py -m "not real_data"` → 168 passed = 171 − 2 `real_data` − 1 retired. Files, split at the script's sections:

    | File | Tests |
    |---|---|
    | `status` | 29: findings 16, geometry 2, Channel Status tab 11 |
    | `overview` | 35: layout 11, metrics 5, composite 6, tab 13 |
    | `supermodule` | 16 |
    | `coincidences` | 16: 6 engine, 10 tab |
    | `limits` | 30: 8 engine, 18 GUI, 4 report; + 1 `real_data` |
    | `origins` | 18: 9 engine, 7 GUI, 2 report; + 1 `real_data` |
    | `slab_rule` | 9: 7 GUI, 2 report |
    | `report` | 15 |

    With `PETSYS_DATA_DIR`/`PETSYS_CAL_DIR` set → the 2 `real_data` tests pass; unset → skipped, naming each variable.
  - **Structure.**
    - Each section function is the script's, copied by a generator (session scratchpad). The only edits:
      - underscore prefixes dropped;
      - `destroy(app)`;
      - a `tmp` argument from a `TemporaryDirectory` replaces `tempfile.mkdtemp` (tests leave no `ldat_*` folder);
      - the `--png` branches removed;
      - the owner-file parts moved or retired;
      - `scripts.ldat_scale_check` imports → `ldat_helpers`.
    - `main`'s inline checks are `findings_checks`.
    - AST (renames normalised): 21 of 35 functions identical; the 14 others differ only by those edits, reviewed in a diff, plus the overview poll-gap bound below.
    - `EXPECTED_ROW`, `SELECTION`, `metrics_dataset` and `time_channels_by_position` moved to `ldat_helpers` (AST-identical).
  - **Recorder.** `ldat_helpers.Checks` runs a section once in a module fixture and records its checks. Each check is one test (`parametrize` over check ids), so a GUI section's steps still run in order on one window.
    - An exception is re-raised by every check the section did not reach.
    - A label not listed, or recorded twice, fails every check of the section, so the test count stays the script's.
    - `test_ldat_helpers.py` gains 4 recorder tests.
  - **Citations:** module `002-FR-15`; per section the task's FRs:
    - findings `002-FR-5/7`, geometry `002-FR-6`;
    - layout and overview `002-FR-8/21` (tab adds `002-FR-9/23`), metrics `002-FR-8/13`;
    - SuperModule `002-FR-10/13`, coincidences `002-FR-11/12`;
    - limits `002-FR-16/17/18` (GUI and report add `002-FR-14`), origins `002-FR-20`;
    - slab rule `002-FR-19`, report `002-FR-5/13/14`;
    - real data adds `007-FR-4`.
    - Counts in the T21 + T22 files: `--fr 002-FR-19` → 32, `002-FR-20` → 19, `002-FR-14` → 41.
  - **Durations.** A section's time is its fixture's setup. Sections ≥ 5 s are `slow` (64 tests): report 55 s, SuperModule tab 10.6 s, origins report 8.8 s, overview tab 6.4–6.6 s, limits GUI 5.00–5.04 s (3 runs). The others ≤ 3.1 s. The default run gets 104 of the 168.
  - **Parallel failure** (first full run): the overview "slow background job never blocks `_poll_events`" check bounds the longest poll gap at 0.4 s wall clock; under parallel load it was 0.59 s. Following the T20 rule, the bound is asserted only in serial runs; the other conditions (more than 20 polls, job off the Tk thread, one cancelled job, one cache entry) always are.
  - **Negatives** (scratch copies):
    - findings LOW `<` → `<=`: 6 fail;
    - `pair_dt` sign reversed: 5 fail;
    - duplicate limits keys accepted: 1 fails;
    - overview click text without the port address: 1 fails;
    - report "Fitted minimodules" line changed: 2 fail;
    - fitted-only plot title changed: 1 fails.
  - **Deleted** (backups in the session scratchpad): `ldat_views_check.py`; `ldat_scale_check.py` (deferred from T21). `scripts/` now holds only `petsys_manager_linux_check.py` (kept until T24).
  - **Not removed:** 389 `%TEMP%\ldat_*` folders left by earlier script runs since 2026-10-02. This session's own script runs' folders were removed.
  - **Suites** (T21 + T22):
    - `python -m pytest` → 875 passed, 18 skipped in 63 s (+214: 94 T21, 104 T22, 4 recorder, 12 scan items);
    - `-m "not real_data"` → 1006 passed, 18 skipped in 214–217 s, twice after the poll-gap change (before it: 1 failed, 1005 passed);
    - `-m "not real_data" -n 0 --slow-limit 5` → 1006 passed, 18 skipped in 728 s (the limits GUI setup took 4.63 s in this run; it stays `slow` from the 3 runs above).

    Free commit stayed at or above 28.07 GB. Fixture folders 908 before and after; `git diff --check` clean.

## Close

- [ ] **T23 — Docs** (FR-11). `AGENTS.md`, `docs/prompts.md`: default vs full run, `slow` ≥ 5 s, parallel default and `-n 0`, `--slow-limit` (serial), golden files and their change rule, `PETSYS_CAL_DIR`; drop `scripts/*_check.py` as evidence.

  **Done when:** both files state each item once; `git diff --check` clean.

- [ ] **T24 — Validation** (all FR).

  **Done when:** `scripts/` holds no `*_check.py`; fresh clone on Windows: default run passes in < 120 s, full run passes in parallel and with `-n 0 --slow-limit 5`, `linux` skipped with reasons, `git status` unchanged; owner's PC with `PETSYS_DATA_DIR`/`PETSYS_CAL_DIR`: `-m real_data` passes, unset → skips with reasons; source scan: no `scripts_cornell`/`scripts_imas`/`gui_cornell` import; Cornell Linux PC (owner run, env updated with `pytest-xdist`): full run passes incl. `linux`/`gui`; `petsys_manager_linux_check.py --all` there gives the same count as `tests/test_manager_linux.py` (17), then it is deleted (kept from T14). FR walk recorded; Status → `shipped`.

## Bug fixes found by this spec (separate from 007, FR-10)

Fixes verified 2026-10-07 (Windows): `python -m pytest` → 164 passed, 0 xfailed; `-m "not real_data" --slow-limit 5` → 164 passed; all 11 `maps/*.yaml` load through `map_factory`; nothing calls `filter_max_sm`/`filter_channel_list`; `map_factory` is the only `YAMLMapReader` user; local `ldat_scale_check` 65/65, `ldat_unpopulated_check` 6/6.

- `bug-filter-channel-list` — `src/filters.py:filter_channel_list` tests `imp[0]` (timestamp) instead of the channel ID. Fixed 2026-10-07: tests `imp[2]`; `--fr bug-filter-channel-list` → 3 passed.
- `bug-filter-max-sm-minimodules` — `src/filters.py:filter_max_sm` counts `(supermodule, minimodule)` pairs, so one supermodule hit in two minimodules fails `max_sm=1` (T2 finding). Fixed 2026-10-07: counts `sm_mM_map[ch][0]`, docstring names the `(SM, mM)` map; `--fr bug-filter-max-sm-minimodules` → 1 passed.
- `bug-yaml-bool-as-int` — `src/yaml_handler.py` accepts a bool for an integer key (`channels: true` → 1). Fixed 2026-10-07: a bool passes only where `bool` is an accepted type; test also rejects `x_pitch: false` and keeps `sum_rows_cols: false`; `--fr bug-yaml-bool-as-int` → 1 passed.
- `bug-yaml-tuple-type-message` — `src/yaml_handler.py` formats `value_type.__name__`, so a wrong type for an `(int, float)` key (`x_pitch`, `y_pitch`) raises `AttributeError` instead of `RuntimeError` naming the key (T2 finding). Fixed 2026-10-07: tuple types reported as `expected int or float`; `--fr bug-yaml-tuple-type-message` → 1 passed.
