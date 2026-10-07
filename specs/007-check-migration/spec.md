# Spec 007 — Move every remaining check into tests/

Status: `in progress`

Constitution: [`AGENTS.md`](../../AGENTS.md). Workflow: [`docs/prompts.md`](../../docs/prompts.md). Builds on spec 006 ([`../006-pytest-checks/spec.md`](../006-pytest-checks/spec.md)), whose suite, markers, `--fr` option and fixtures this spec reuses. This spec changes two spec 006 rules (default run, real-data location: FR-5, FR-4), so owner approval covers those changes.

## Goal

Finish the move started by spec 006: every check in `scripts/*_check.py` becomes a tracked, reproducible test, or is retired, and no test depends on untracked reference scripts, sibling repositories or hardcoded paths. Afterwards `scripts/` holds only local analysis and `scripts/template.py`.

Inventory (2026-10-07): 30 scripts, about 17k lines, three styles (`unittest`, `check()` + `sys.exit`, bare `assert`), heavily cross-importing each other. The calibration, listmode and QC checks take their expected values from untracked `scripts_cornell/` reference scripts; `petsys_manager_reference_check` also reads the sibling `gui_cornell/src/gui.py` (`cli`/`checkout` only assert that the Manager never imports it); `--real` modes read hardcoded `C:\Users\dsanchez\Desktop\data\...` paths and the untracked `encal_files/`.

## Owner answers (2026-10-07)

- Scope: everything — fixture-only checks, the fixture parts of mixed checks, the real-data checks kept below, and the reference/real parity checks.
- Default run: fast tests only; `slow` tests are opt-in.
- Shared helpers: per-area modules (`tests/manager_helpers.py`, `tests/ldat_helpers.py`).
- Reference-based expected values: golden files frozen from the reference scripts once, stored under `tests/data/` with provenance; tests never load the references.
- Real data: two variables, `PETSYS_DATA_DIR` (acquisitions) and `PETSYS_CAL_DIR` (calibrations).
- Retired without migration: `ldat_package_check.py` (one-off spec 004 move audit), `cornell_slab_en_cal_check.py` (tests the untracked `scripts_cornell/cornell_slab_en_cal.py`, not tracked code), `ldat_real_check.py` (interactive viewer, not a check).
- Also add direct synthetic tests for `src/filters.py`, `src/filters_fixed.py`, `src/fem_handler.py` and `src/yaml_handler.py`, which today are covered only indirectly.
- Separately from this spec (done, `ac9d3a5`): deleted the unused `src/write_output.py`.

Clarify (2026-10-07):

- `slow`: a test taking 5 s or more. Default run budget: 120 s on the owner's Windows PC.
- `PETSYS_CAL_DIR` unset: calibration-using real-data tests skip with a reason naming it (no implicit location).
- `PETSYS_DATA_DIR` is the parent data folder (owner's PC: `C:\Users\dsanchez\Desktop\data`); tests name files relative to it, e.g. `Cornell/full_system/<file>`. Another PC provides the same subfolders (a link to its storage is fine).
- Real-data checks kept ("accepted results" only): `ldat_scale_check` spec 001 T9 counts on the January prefixes; `ldat_unpopulated_check` real prefix; `petsys_manager_real_check` T19 parity; `petsys_manager_gui_check` real split-file selection (January and September). Their asserted numbers stay as constants in the test code with a comment naming data, map, calibration and cuts, except T19, whose values come from the newest Windows `real_baseline.json` (2026-10-05) frozen under `tests/data/baselines/`.
- Real-data checks retired: `ldat_issue_check` real photopeak probe, `ldat_scale_check --whole` (timing, memory estimates), `ldat_pair_choices_check --real` (25 GB file, 30 M prefix), `ldat_views_check --real`, `petsys_manager_calibration_check.RealCornellChecks` (compact = fixed calibration on real January files). Their earlier `Verified:` lines stay as history.
- Golden files are computed from the local `scripts_cornell/` copies that spec 003's parity checks used (`cornell_slab_en_cal.py`, `cornell_listmode_cog_fixed_position.py`, `cornell_system_validation.py`), recording each source's SHA-256 (spec 003 T1: not asserted to be the Cornell-installed versions).
- `gui_cornell` is retired (owner): the PETsys Manager replaced it after Cornell acceptance (spec 003, 2026-10-06). `petsys_manager_reference_check`'s static inventory of `gui_cornell/src/gui.py` and the PETsys tool sources (option tokens, `ACQ_*` constants, tool options) was a one-time spec 003 T1 design input, recorded in spec 003; it is retired, not frozen. The Manager's command lines stay covered by `petsys_manager_check --commands`. The `cli`/`checkout` checks that the Manager never imports `gui_cornell` or `scripts*` move unchanged.
- `src/filters.py`: test every function. `filter_channel_list` checks `imp[0]` (the timestamp) instead of the channel ID; its test asserts channel IDs and is a strict `xfail` as `bug-filter-channel-list` until a separate fix. `filter_max_sm` is documented as at most N supermodules but counts distinct `sm_mM_map` values, which are `(supermodule, minimodule)` pairs (T2 finding, owner decision 2026-10-07): its test asserts supermodules and is a strict `xfail` as `bug-filter-max-sm-minimodules`.
- `src/yaml_handler.py`: a bool passes `isinstance(value, int)`, so `channels: true` validates as 1. The test asserts a bool is rejected for integer keys; strict `xfail` as `bug-yaml-bool-as-int` until a separate fix. A wrong type for a multi-type key (`x_pitch`, `y_pitch`: `(int, float)`) raises `AttributeError` (a tuple has no `__name__`) instead of the `RuntimeError` naming the key (T2 finding, same owner rule): strict `xfail` as `bug-yaml-tuple-type-message`.
- `petsys_manager_gui_check.py` (2193 lines) and `ldat_views_check.py` (2535 lines) are split into one test file per GUI area, sharing builders through the helper modules; pass counts are compared per script mode.

## Data contract

Tests only read `src/` and `exe_programs/`; they change no algorithm, cut, output format or GUI behavior. Systems: IMAS and Cornell, each through its selected map (`maps/imas1DAQ_map.yaml`; Cornell `maps/cornell_map_full_system.yaml` September, `maps/cornell_map_full_system_old.yaml` January); no 48-module or 256-channels-per-SuperModule assumption. Synthetic LDAT fixtures follow the `src/read_compact.py` contract: coincidence records of two detectors, each a list of `(timestamp, channel energy, channel ID)` hits; no singles population. Energies are PETsys a.u. unless a test names its calibration; keV conversion uses that calibration file. Real-data tests use the January 2026-01-19 Cornell acquisitions (`coincCompact11s_00000003…8`) with the January map and calibrations, and the September Ge68 acquisitions with the September map. Every golden file and baseline records the data, map, calibration, cuts, prefix and seeds that produced it.

## Requirements

- **FR-1 — Migration set.** Every `scripts/*_check.py` SHALL be moved into `tests/` except the three retired scripts, which SHALL be deleted. `petsys_manager_reference_check.py`'s synthetic parity contracts SHALL become golden-file tests (FR-3) and its `gui_cornell`/PETsys-source inventory SHALL be retired (Clarify) and `petsys_manager_real_check.py` real-data tests (FR-4); both scripts SHALL then be deleted. At completion `scripts/` SHALL contain no `*_check.py` and no fixture file used only by a check.
- **FR-2 — Same coverage.** WHEN a script is migrated, its tests SHALL give the same pass count as the script on the same fixtures (per argparse mode for mode-driven scripts), recorded in `tasks.md` before the script is deleted. Retired real-data modes (Clarify) are excluded from the comparison and listed. `unittest.TestCase` classes MAY move unchanged; argparse modes become markers or files; `check()` tables become `parametrize`. A check whose pass count cannot match (e.g. it asserted on a reference script's internals) SHALL be listed in `tasks.md` with its replacement and the owner's approval.
- **FR-3 — Golden files.** Expected values that today come from `scripts_cornell/` reference scripts SHALL be computed once by those references on the synthetic fixtures and stored under `tests/data/golden/`, each with a provenance record (reference path and SHA-256 of its source, inputs, seeds, settings, date). No test SHALL import or execute `scripts_cornell/`, `scripts_imas/`, `gui_cornell` or the PETsys sources. A golden file SHALL change only in a commit that cites the spec requirement or bug id authorizing the output change.
- **FR-4 — Real data.** The kept real-data checks (Clarify) SHALL carry `real_data` and locate acquisitions relative to `PETSYS_DATA_DIR` (e.g. `Cornell/full_system/<file>`) and calibrations relative to `PETSYS_CAL_DIR`, never through an absolute path. T19 expected values SHALL come from `tests/data/baselines/`; the others keep their asserted constants in the test. WHEN a variable or named file is missing, the test SHALL be skipped with a reason naming it. This extends spec 006 FR-4 with `PETSYS_CAL_DIR`.
- **FR-5 — Default run.** The default `python -m pytest` SHALL exclude `real_data` and `slow` tests and SHALL finish in under 120 s on the owner's Windows PC. A test SHALL carry `slow` when it takes 5 s or more. `-m "not real_data"` (the full run) and `-m real_data` SHALL remain available; a spec's `Validation` cites the full run. This replaces spec 006 FR-3's "`slow` runs by default".
- **FR-6 — Helpers.** Shared builders SHALL live in `tests/conftest.py`, `tests/helpers.py`, `tests/manager_helpers.py` or `tests/ldat_helpers.py`. No test SHALL import another test file, a `scripts*` module, or modify `sys.path` (spec 006 `test_infra.py` scan keeps enforcing this). The two large GUI checks SHALL be split into one test file per GUI area.
- **FR-7 — Markers and platforms.** Tests needing a display carry `gui`; Linux-only tests (process groups, PETsys Python, DAQD) carry `linux`. The full run SHALL pass on Windows (`linux` skipped with reason) and on the Cornell Linux PC (all run).
- **FR-8 — Traceability.** Each migrated test SHALL cite with `@pytest.mark.fr` the requirement ids its script states (e.g. `003-FR-21`, `002-FR-17`, `bug-<name>`). WHEN a script states only a task (e.g. `T25.4`), the test SHALL cite the requirement that task implements.
- **FR-9 — Direct src tests.** New synthetic tests, citing `007-FR-9`, SHALL cover:
  - `src/filters.py`, every function: `filter_total_energy` open bounds `(en_min, en_max)`; `filter_min_ch` with and without `sum_rows_cols` (at least `min_ch` energy channels, and fewer energy channels than hits when summed); `filter_single_mM`; `filter_max_sm` counting supermodules across both detectors (strict `xfail` `bug-filter-max-sm-minimodules`); `filter_specific_mm`; `filter_channel_list` (all hits of either detector among the valid channel IDs; strict `xfail` `bug-filter-channel-list`); `filter_ROI` open bounds; `filter_coincidence` time window between the two detectors' highest-energy time channels.
  - `src/filters_fixed.py`: `filter_min_ch_vectorized`, `filter_total_energy_vectorized`, `filter_single_mM_vectorized` with `-1` padding ignored, an empty event rejected, and the dict and pre-computed array forms agreeing; each agrees with its scalar counterpart on the same synthetic events.
  - `src/fem_handler.py`: `get_FEM_instance` for FEM128/FEM256 and `ValueError` for another type; `get_coordinates` for both `sum_rows_cols` modes at the first, last and a middle channel, computed from pitch and channel count.
  - `src/yaml_handler.py`: missing mandatory key and wrong type rejected with the key named (multi-type key: strict `xfail` `bug-yaml-tuple-type-message`); bool rejected for an integer key (strict `xfail` `bug-yaml-bool-as-int`); exactly one optional group required (none or both rejected); unreadable YAML and missing file give their `RuntimeError`; `get_optional_group_keys` for both groups and neither.
- **FR-10 — No product change.** The spec SHALL NOT change behavior in `src/` or `exe_programs/`. A defect exposed by a migrated or new test SHALL be recorded and fixed as a separate bug fix with a strict `xfail` until then (spec 006 rule).
- **FR-11 — Docs.** `AGENTS.md` and `docs/prompts.md` SHALL state the default/full run split, golden files and their change rule, and `PETSYS_CAL_DIR`; references to `scripts/*_check.py` as evidence SHALL be removed once none remain (earlier `Verified:` lines stay as history).

## Completion criteria

- `scripts/` contains no `*_check.py`; each migrated script's pass count is recorded in `tasks.md` next to its tests' count, with retired modes listed.
- Fresh clone on Windows: default run passes in under 120 s; full run (`-m "not real_data"`) passes with `linux` skipped with reasons; `git status --short` unchanged.
- Same commit on the Cornell Linux PC: full run passes, `linux` and `gui` included.
- Owner's PC with `PETSYS_DATA_DIR` and `PETSYS_CAL_DIR` set: `-m real_data` passes; unset: every real-data test skipped with a reason.
- No tracked test imports `scripts_cornell`, `scripts_imas`, `gui_cornell` or PETsys sources (source scan); every golden file and the T19 baseline have provenance.
- `--fr 007-FR-9` covers the four modules; the four new bug ids show as strict `xfail` until fixed.
- `AGENTS.md` and `docs/prompts.md` updated.

## Out of scope

CI, pre-commit hooks and coverage thresholds; changing any `src/` or `exe_programs/` behavior (bug fixes are separate); the retired real-data modes; tests for `src/plots.py` and `src/slab_nn.py` (no tracked caller); `scripts_cornell/` and `scripts_imas/` analyses; operator visual GUI review; packaging or release changes.

## Open questions

None.
