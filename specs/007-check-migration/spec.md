# Spec 007 — Move every remaining check into tests/

Status: `draft`

Constitution: [`AGENTS.md`](../../AGENTS.md). Workflow: [`docs/prompts.md`](../../docs/prompts.md). Builds on spec 006 ([`../006-pytest-checks/spec.md`](../006-pytest-checks/spec.md)), whose suite, markers, `--fr` option and fixtures this spec reuses. This spec changes two spec 006 rules (default run, real-data location: FR-5, FR-4), so owner approval covers those changes.

## Goal

Finish the move started by spec 006: every check in `scripts/*_check.py` becomes a tracked, reproducible test, and no check depends on untracked reference scripts, sibling repositories or hardcoded paths. Afterwards `scripts/` holds only local analysis and `scripts/template.py`.

Inventory (2026-10-07): 30 scripts, about 17k lines, three styles (`unittest`, `check()` + `sys.exit`, bare `assert`), heavily cross-importing each other. The calibration, listmode and QC checks take their expected values from untracked `scripts_cornell/` reference scripts; `cli`/`checkout` look at the sibling `gui_cornell`; `--real` modes read hardcoded `C:\Users\dsanchez\Desktop\data\...` paths and the untracked `encal_files/`.

## Owner answers (2026-10-07)

- Scope: everything — fixture-only checks, the fixture and real-data parts of mixed checks, and the reference/real parity checks.
- Default run: fast tests only; `slow` tests are opt-in.
- Shared helpers: per-area modules (`tests/manager_helpers.py`, `tests/ldat_helpers.py`).
- Reference-based expected values: golden files frozen from the reference scripts once, stored under `tests/data/` with provenance; tests never load the references.
- Real data: two variables, `PETSYS_DATA_DIR` (acquisitions) and `PETSYS_CAL_DIR` (calibrations). Expected values are the numbers the scripts assert today, frozen as baselines.
- Retired without migration: `ldat_package_check.py` (one-off spec 004 move audit), `cornell_slab_en_cal_check.py` (tests the untracked `scripts_cornell/cornell_slab_en_cal.py`, not tracked code), `ldat_real_check.py` (interactive viewer, not a check).
- Also add direct synthetic tests for `src/filters.py`, `src/filters_fixed.py`, `src/fem_handler.py` and `src/yaml_handler.py`, which today are covered only indirectly.
- Separately from this spec: delete the unused `src/write_output.py` (no caller anywhere).

## Data contract

Tests only read `src/` and `exe_programs/`; they change no algorithm, cut, output format or GUI behavior. Systems: IMAS and Cornell, each through its selected map (`maps/imas1DAQ_map.yaml`, `maps/cornell_map_full_system.yaml` September, `maps/cornell_map_full_system_old.yaml` January); no 48-module or 256-channels-per-SuperModule assumption. Synthetic LDAT fixtures follow the `src/read_compact.py` contract: coincidence records of two detectors, each a list of `(timestamp, channel energy, channel ID)` hits; no singles population. Energies are PETsys a.u. unless a test names its calibration; keV conversion uses that calibration file. Real-data tests use the January 2026-01-19 Cornell acquisitions (`coincCompact11s_3…8`, `coincFixed11s_3…8`) with the January map and calibrations, and the September Ge68 acquisitions with the September map. Every golden file and baseline records the data, map, calibration, cuts, prefix and seeds that produced it.

## Requirements

- **FR-1 — Migration set.** Every `scripts/*_check.py` SHALL be moved into `tests/` except the three retired scripts, which SHALL be deleted. `petsys_manager_reference_check.py` SHALL become golden-file tests (FR-3) and `petsys_manager_real_check.py` real-data tests (FR-4); both scripts SHALL then be deleted. At completion `scripts/` SHALL contain no `*_check.py` and no fixture file used only by a check.
- **FR-2 — Same coverage.** WHEN a script is migrated, its tests SHALL give the same pass count as the script on the same fixtures (per argparse mode for mode-driven scripts), recorded in `tasks.md` before the script is deleted. `unittest.TestCase` classes MAY move unchanged; argparse modes become markers or files; `check()` tables become `parametrize`. A check whose pass count cannot match (e.g. it asserted on a reference script's internals) SHALL be listed in `tasks.md` with its replacement and the owner's approval.
- **FR-3 — Golden files.** Expected values that today come from `scripts_cornell/` reference scripts or the sibling `gui_cornell` SHALL be computed once by those references on the synthetic fixtures and stored under `tests/data/golden/`, each with a provenance record (reference path and SHA-256 of its source, inputs, seeds, settings, date). No test SHALL import or execute `scripts_cornell/`, `scripts_imas/`, `gui_cornell` or the PETsys sources. A golden file SHALL change only in a commit that cites the spec requirement or bug id authorizing the output change.
- **FR-4 — Real data.** Real-data tests SHALL carry `real_data` and locate acquisitions under `PETSYS_DATA_DIR` and calibrations under `PETSYS_CAL_DIR`, never through an absolute path. Their expected values SHALL come from baselines under `tests/data/baselines/` frozen from today's script assertions. WHEN a variable or named file is missing, the test SHALL be skipped with a reason naming it. This extends spec 006 FR-4 with `PETSYS_CAL_DIR`.
- **FR-5 — Default run.** The default `python -m pytest` SHALL exclude `real_data` and `slow` tests and SHALL finish in under 60 s on the owner's Windows PC. A test SHALL carry `slow` when it takes 2 s or more. `-m "not real_data"` (the full run) and `-m real_data` SHALL remain available; a spec's `Validation` cites the full run. This replaces spec 006 FR-3's "`slow` runs by default".
- **FR-6 — Helpers.** Shared builders SHALL live in `tests/conftest.py`, `tests/helpers.py`, `tests/manager_helpers.py` or `tests/ldat_helpers.py`. No test SHALL import another test file, a `scripts*` module, or modify `sys.path` (spec 006 `test_infra.py` scan keeps enforcing this).
- **FR-7 — Markers and platforms.** Tests needing a display carry `gui`; Linux-only tests (process groups, PETsys Python, DAQD) carry `linux`. The full run SHALL pass on Windows (`linux` skipped with reason) and on the Cornell Linux PC (all run).
- **FR-8 — Traceability.** Each migrated test SHALL cite with `@pytest.mark.fr` the requirement ids its script states (e.g. `003-FR-21`, `002-FR-17`, `bug-<name>`). WHEN a script states only a task (e.g. `T25.4`), the test SHALL cite the requirement that task implements.
- **FR-9 — Direct src tests.** New synthetic tests SHALL cover `src/filters.py`, `src/filters_fixed.py`, `src/fem_handler.py` and `src/yaml_handler.py` (behaviours listed at Clarify), citing `007-FR-9`.
- **FR-10 — No product change.** The spec SHALL NOT change behavior in `src/` or `exe_programs/`. A defect exposed by a migrated or new test SHALL be recorded and fixed as a separate bug fix with a strict `xfail` until then (spec 006 rule).
- **FR-11 — Docs.** `AGENTS.md` and `docs/prompts.md` SHALL state the default/full run split, golden files and their change rule, and `PETSYS_CAL_DIR`; references to `scripts/*_check.py` as evidence SHALL be removed once none remain (earlier `Verified:` lines stay as history).

## Completion criteria

- `scripts/` contains no `*_check.py`; each migrated script's pass count is recorded in `tasks.md` next to its tests' count.
- Fresh clone on Windows: default run passes in under 60 s; full run (`-m "not real_data"`) passes with `linux` skipped with reasons; `git status --short` unchanged.
- Same commit on the Cornell Linux PC: full run passes, `linux` and `gui` included.
- Owner's PC with `PETSYS_DATA_DIR` and `PETSYS_CAL_DIR` set: `-m real_data` passes against the frozen baselines; unset: every real-data test skipped with a reason.
- No tracked test imports `scripts_cornell`, `scripts_imas`, `gui_cornell` or PETsys sources (source scan); every golden file and baseline has provenance.
- `AGENTS.md` and `docs/prompts.md` updated.

## Out of scope

CI, pre-commit hooks and coverage thresholds; changing any `src/` or `exe_programs/` behavior; tests for `src/plots.py` and `src/slab_nn.py` (no tracked caller); `scripts_cornell/` and `scripts_imas/` analyses; operator visual GUI review; packaging or release changes. Deleting `src/write_output.py` is a separate small commit.

## Open questions (Clarify)

1. `slow` threshold (2 s per test?) and the default-run budget (60 s?).
2. `PETSYS_CAL_DIR` unset: skip, or default to the repo's `encal_files/`?
3. Layout under `PETSYS_DATA_DIR`: `Cornell/full_system/<file>` as on the owner's PC? Same subfolders on the Cornell Linux PC?
4. Baselines: which real modes to freeze — January only, or also the September Ge68 files? Which existing T19 baseline JSONs (in `%LOCALAPPDATA%\Temp\process_petsys`) become tracked baselines?
5. Golden capture: the local `scripts_cornell/` copies are not asserted to be the Cornell-installed versions (spec 003 T1). Freeze from these local copies, recording their SHA-256?
6. Direct src tests: which behaviours of `filters`, `filters_fixed`, `fem_handler`, `yaml_handler` (listed after reading the modules)?
7. `petsys_manager_gui_check.py` (2193 lines, Tk) and `ldat_views_check.py` (2535 lines): move as-is into `gui`/`slow` tests, or split by view?
