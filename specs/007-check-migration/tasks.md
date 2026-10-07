# Tasks 007 — Move every remaining check into tests/

Spec: [`spec.md`](spec.md). Plan: [`plan.md`](plan.md). Owner approved 2026-10-07. Environment: env interpreter `-m pytest` from the repo root. Never stage or commit; the owner runs git.

**Migration Done-when (applies to every task marked "migrate"):** in one session on the same fixtures, the script (each listed mode) gives `PASS N` (or `Ran N … OK`) and the new pytest selection gives `N passed` (gui/linux skips counted with their reason); `--durations=0` reviewed and tests ≥ 5 s marked `slow`; `test_infra.py` scan passes (no `sys.path`, no `scripts*`/test-file imports); then the script is deleted, or, when a remaining script still imports it, kept until its last importer's task (plan: Deferred deletion). Record both counts, the selection command and any listed exclusion (retired real mode, owner-approved replacement).

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

## Manager group (spec 003)

- [x] **T4 — migrate `petsys_manager_artifact_check`** (44 tests) → `tests/test_manager_artifacts.py`.

  Verified 2026-10-07 (Windows): script `python scripts/petsys_manager_artifact_check.py` → `PASS: 44 checks; 1 skipped`; `python -m pytest tests/test_manager_artifacts.py` → 43 passed, 1 skipped (`linux`: the symlink test, `@unittest.skipIf(os.name == "nt")` → `@pytest.mark.linux`, FR-7). `ArtifactChecks` moved unchanged on `PrivateOutput` (replaces its `setUpClass`); AST: all 48 methods identical to the script. Citations (FR-8): class `003-FR-9`, `003-FR-11`, `003-FR-16` (spec 003 T4); the 7 T29 tests add `003-FR-13`, the FR-24 test `003-FR-24`; `--fr 003-FR-16` → 43 passed, 1 skipped; `--fr 003-FR-13` → 7; `--fr 003-FR-24` → 1. `--durations`: slowest 0.06 s, none `slow`. No fixture folder left after the run. `python -m pytest` → 220 passed, 1 skipped; `-m "not real_data" --slow-limit 5` → 220 passed, 1 skipped (includes the `test_infra.py` scan). Script deleted; `petsys_manager_check.py --artifacts`/`--all` (lazy import) no longer load it, aggregate dropped at T6.
- [x] **T5 — migrate `petsys_manager_daqd_check` (22) and `petsys_manager_acquisition_check` (27)** → `tests/test_manager_daqd.py`, `tests/test_manager_acquisition.py`.

  Verified 2026-10-07 (Windows): scripts → `PASS: 22 DAQD checks`, `PASS: 27 acquisition checks`; `python -m pytest tests/test_manager_daqd.py tests/test_manager_acquisition.py` → 49 passed (22 + 27). `DaqdChecks`/`AcquisitionChecks` moved on `PrivateOutput`; fakes from `manager_helpers` (`FakeDaemon`/`FakeResources` were the daqd script's own, AST-identical at T3); `FreshChildBackend`, `wait_for`, `FAST`, `POLICY` and the acquisition `ScaledClock`/`Scenario*` stay in their test files. AST (renames `fixture` → `profile_fixture`, `ROOT` → `REPO`): daqd 32, acquisition 50 definitions identical; only `setUpClass` replaced. Citations: DAQD class `003-FR-5/6/7/16` (T6), T35 test adds `003-FR-1`; acquisition class `003-FR-5/7/8/9/16` (T7), 4 bias-off tests add `003-FR-19`, progress/loss-limit/safety-defaults tests `003-FR-20`; `--fr 003-FR-6` → 22, `003-FR-8` → 27, `003-FR-19` → 4, `003-FR-20` → 3, `003-FR-1` → 1. `--durations`: slowest 1.64 s, none `slow`. No fixture folder left (908 before and after). `python -m pytest` → 271 passed, 1 skipped; `-m "not real_data" --slow-limit 5` → 271 passed, 1 skipped. `petsys_manager_acquisition_check.py` deleted (only the aggregate runner imported it, lazily). **`petsys_manager_daqd_check.py` kept: deletion deferred to T15** (`petsys_manager_gui_check.py` imports `FakeDaemon`, `FakeResources` from it).
- [ ] **T6 — migrate `petsys_manager_check`** (75; modes `--settings --commands --runner --artifacts`, plus GUI/Linux parts) → `tests/test_manager_{settings,commands,runner}.py` (artifacts already in T4); aggregate `main()` dropped.
- [ ] **T7 — migrate `petsys_manager_numeric_check`** (41; modes `--formats --calibration --listmode --qc --bounded --scope`) → `tests/test_manager_formats.py`; the other modes map to T9–T12 files.
- [ ] **T8 — Golden capture** (FR-3). `scripts/capture_golden_007.py` (local): runs the calibration, listmode and QC oracles (`scripts_cornell/cornell_slab_en_cal.py`, `cornell_listmode_cog_fixed_position.py`, `cornell_system_validation.py`) and `petsys_manager_reference_check --synthetic` contracts on `manager_helpers` fixtures → `tests/data/golden/…` + provenance; `tests/test_golden_provenance.py`.

  **Done when:** every golden file has a provenance record (oracle path + SHA-256, capture script SHA-256, fixture builder + seed, map, settings, date); capturing twice gives byte-identical files; the provenance test passes; no tracked file imports `scripts_cornell`.

- [ ] **T9 — migrate `petsys_manager_calibration_check`** (22, minus `RealCornellChecks` retired) → `tests/test_manager_calibration.py`, oracle comparisons → golden files.
- [ ] **T10 — migrate `petsys_manager_listmode_check`** (21) → `tests/test_manager_listmode.py`, oracle → golden.
- [ ] **T11 — migrate `petsys_manager_qc_check`** (15) → `tests/test_manager_qc.py`, oracle → golden.
- [ ] **T12 — migrate `petsys_manager_bounded_check` and `petsys_manager_scope_check`** (3) → `tests/test_manager_bounded.py`, `tests/test_manager_scope.py`.
- [ ] **T13 — migrate `petsys_manager_cli_check` (15) and `petsys_manager_workflow_check` (15)** → `tests/test_manager_cli.py`, `tests/test_manager_workflow.py`; the "never imports `gui_cornell`/`scripts*`" checks kept.
- [ ] **T14 — migrate `petsys_manager_checkout_check` (5) and `petsys_manager_linux_check` (17)** → `linux`-marked files; Windows shows them skipped with reason; Linux counts confirmed at T24.
- [ ] **T15 — migrate `petsys_manager_gui_check`** (then delete the deferred `petsys_manager_daqd_check.py`) (28; modes `--acquisition --conversion --processing --shell`, `--real`) → `tests/test_manager_gui_{acquisition,conversion,processing,shell}.py` (`gui`) + `RealSelectionChecks` as `real_data` (January and September split selection, paths via `PETSYS_DATA_DIR`).
- [ ] **T16 — retire `petsys_manager_reference_check`** (FR-1, FR-3). Its synthetic parity contracts are covered by T8 golden files and their tests; the `gui_cornell`/PETsys-source inventory is retired.

  **Done when:** each `--synthetic` contract (17) is mapped in this file to the golden-file test that replaces it or listed as inventory-only (retired, owner-approved in spec Clarify); script deleted.

- [ ] **T17 — migrate `petsys_manager_real_check`** (FR-4) → `tests/test_manager_t19_real.py` (`real_data`, `slow`): spec 003 T19 calibration/LM/QC on the January files vs `tests/data/baselines/t19_jan2026_{manifest,result}.json`; manifest paths relative to `PETSYS_DATA_DIR`, January config.

  **Done when:** with both variables set on the owner's PC the tests pass and reproduce the 2026-10-05 baseline values; unset → skipped naming the variable; `scripts/petsys_manager_t19_baseline_{jan2026,cornell_f18}.json` moved/retired; script deleted.

## LDAT group (specs 001, 002)

- [ ] **T18 — LDAT helpers and tracked January config** (FR-6). `tests/ldat_helpers.py` (`fixture_files`, `write_pairs`, `build`, `CONFIGS`, `RECOVERY_CASES`, side/LDAT writers); `tests/data/configs/cornell_january.yaml`.

  **Done when:** helpers import; `CONFIGS` resolves IMAS (tracked `configs/imas_1DAQ.yaml`) and Cornell (tracked copy) on a fresh clone; smoke test writes one fixture per system.

- [ ] **T19 — migrate `ldat_inspector_check --selftest` (59), `ldat_revision_check` (14), `ldat_ports_check` (12)**.
- [ ] **T20 — migrate `ldat_processing_check` (17) and `ldat_gui_check` (29)** (`gui`; Linux close-child part `linux`).
- [ ] **T21 — migrate `ldat_scale_check` (65 + `--real` T9 counts), `ldat_unpopulated_check` (6 + `--real`), `ldat_issue_check` (8, fixture only), `ldat_pair_choices_check` (15, fixture only)**; retired real modes listed (`--whole`, issue probe, pair_choices `--real`).
- [ ] **T22 — migrate `ldat_views_check`** (171; `--real` retired) → `tests/test_ldat_views_{status,supermodule,overview,coincidences,…}.py` (`gui`), split at existing section boundaries.

## Close

- [ ] **T23 — Docs** (FR-11). `AGENTS.md`, `docs/prompts.md`: default vs full run, `slow` ≥ 5 s, `--slow-limit`, golden files and their change rule, `PETSYS_CAL_DIR`; drop `scripts/*_check.py` as evidence.

  **Done when:** both files state each item once; `git diff --check` clean.

- [ ] **T24 — Validation** (all FR).

  **Done when:** `scripts/` holds no `*_check.py`; fresh clone on Windows: default run passes in < 120 s, full run passes with `--slow-limit 5`, `linux` skipped with reasons, `git status` unchanged; owner's PC with `PETSYS_DATA_DIR`/`PETSYS_CAL_DIR`: `-m real_data` passes, unset → skips with reasons; source scan: no `scripts_cornell`/`scripts_imas`/`gui_cornell` import; Cornell Linux PC (owner run): full run passes incl. `linux`/`gui`. FR walk recorded; Status → `shipped`.

## Bug fixes found by this spec (separate from 007, FR-10)

Fixes verified 2026-10-07 (Windows): `python -m pytest` → 164 passed, 0 xfailed; `-m "not real_data" --slow-limit 5` → 164 passed; all 11 `maps/*.yaml` load through `map_factory`; nothing calls `filter_max_sm`/`filter_channel_list`; `map_factory` is the only `YAMLMapReader` user; local `ldat_scale_check` 65/65, `ldat_unpopulated_check` 6/6.

- `bug-filter-channel-list` — `src/filters.py:filter_channel_list` tests `imp[0]` (timestamp) instead of the channel ID. Fixed 2026-10-07: tests `imp[2]`; `--fr bug-filter-channel-list` → 3 passed.
- `bug-filter-max-sm-minimodules` — `src/filters.py:filter_max_sm` counts `(supermodule, minimodule)` pairs, so one supermodule hit in two minimodules fails `max_sm=1` (T2 finding). Fixed 2026-10-07: counts `sm_mM_map[ch][0]`, docstring names the `(SM, mM)` map; `--fr bug-filter-max-sm-minimodules` → 1 passed.
- `bug-yaml-bool-as-int` — `src/yaml_handler.py` accepts a bool for an integer key (`channels: true` → 1). Fixed 2026-10-07: a bool passes only where `bool` is an accepted type; test also rejects `x_pitch: false` and keeps `sum_rows_cols: false`; `--fr bug-yaml-bool-as-int` → 1 passed.
- `bug-yaml-tuple-type-message` — `src/yaml_handler.py` formats `value_type.__name__`, so a wrong type for an `(int, float)` key (`x_pitch`, `y_pitch`) raises `AttributeError` instead of `RuntimeError` naming the key (T2 finding). Fixed 2026-10-07: tuple types reported as `expected int or float`; `--fr bug-yaml-tuple-type-message` → 1 passed.
