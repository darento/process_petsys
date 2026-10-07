# Tasks 007 — Move every remaining check into tests/

Spec: [`spec.md`](spec.md). Plan: [`plan.md`](plan.md). Owner approved 2026-10-07. Environment: env interpreter `-m pytest` from the repo root. Never stage or commit; the owner runs git.

**Migration Done-when (applies to every task marked "migrate"):** in one session on the same fixtures, the script (each listed mode) gives `PASS N` (or `Ran N … OK`) and the new pytest selection gives `N passed` (gui/linux skips counted with their reason); `--durations=0` reviewed and tests ≥ 5 s marked `slow`; `test_infra.py` scan passes (no `sys.path`, no `scripts*`/test-file imports); then the script is deleted. Record both counts, the selection command and any listed exclusion (retired real mode, owner-approved replacement).

## Infrastructure

- [ ] **T0 — Default run, calibration fixture, slow guard** (FR-4, FR-5). `addopts` `-m "not real_data and not slow"`; `real_cal_file` fixture (`PETSYS_CAL_DIR`, skip naming it); `--slow-limit SECONDS` option; `test_infra.py` cases for all three; spec 006's `--fr` unchanged.

  **Done when:** `python -m pytest tests/test_infra.py` passes, covering: default run deselects `slow` and `real_data`; `-m "not real_data"` runs `slow`; `real_cal_file` skips naming `PETSYS_CAL_DIR` (unset) and the missing file; `--slow-limit 1` fails a 1.5 s non-slow test and passes a `slow` one. Full suite still passes.

- [ ] **T1 — Retire three scripts** (FR-1). Delete `scripts/ldat_package_check.py`, `scripts/cornell_slab_en_cal_check.py` (+ `scripts/fixtures_cornell_slab_spectra.json`), `scripts/ldat_real_check.py`.

  **Done when:** no other script imports them (grep); files deleted; their spec 004 / bug evidence stays in earlier `tasks.md`.

- [ ] **T2 — Direct src tests** (FR-9, FR-10). `tests/test_filters.py`, `tests/test_filters_fixed.py`, `tests/test_fem_handler.py`, `tests/test_yaml_handler.py` per spec FR-9.

  **Done when:** `python -m pytest --fr 007-FR-9 -rx` passes with exactly two strict xfails (`bug-filter-channel-list`, `bug-yaml-bool-as-int`), each failing before marking (recorded); scalar/vectorized agreement on shared synthetic events; a negative check (deliberately broken copy) fails.

- [ ] **T3 — Manager helpers** (FR-6). `tests/manager_helpers.py`: copy the shared builders and fakes listed in the plan; mixins for the shared `TestCase` setup (`CalibrationFixtures`, `ListmodeFixtures`, `QCFixtures`, `CLIFixtures`). Scripts stay untouched.

  **Done when:** module imports without Tk/display; a smoke test builds one fixture of each kind into `tmp_path`; full suite passes.

## Manager group (spec 003)

- [ ] **T4 — migrate `petsys_manager_artifact_check`** (44 tests) → `tests/test_manager_artifacts.py`.
- [ ] **T5 — migrate `petsys_manager_daqd_check` (22) and `petsys_manager_acquisition_check` (27)** → `tests/test_manager_daqd.py`, `tests/test_manager_acquisition.py`.
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
- [ ] **T15 — migrate `petsys_manager_gui_check`** (28; modes `--acquisition --conversion --processing --shell`, `--real`) → `tests/test_manager_gui_{acquisition,conversion,processing,shell}.py` (`gui`) + `RealSelectionChecks` as `real_data` (January and September split selection, paths via `PETSYS_DATA_DIR`).
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

- `bug-filter-channel-list` — `src/filters.py:filter_channel_list` tests `imp[0]` (timestamp) instead of the channel ID.
- `bug-yaml-bool-as-int` — `src/yaml_handler.py` accepts a bool for an integer key (`channels: true` → 1).
