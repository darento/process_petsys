# Spec 006 — Tracked pytest checks

Status: `approved`

Constitution: [`AGENTS.md`](../../AGENTS.md). Workflow: [`docs/prompts.md`](../../docs/prompts.md). Spec 003 shipped 2026-10-06; owner approved this spec 2026-10-07. This spec changes the constitution's check rules (FR-9), so owner approval of the spec covers that change.

## Goal

Make every recorded check reproducible from a clone and rerunnable as one regression run. Today the 31 checks in `scripts/*_check.py` are untracked (`.gitignore` `scripts/*`), mix three styles (bare `assert`; `check()` + `sys.exit`; `unittest` + argparse runner), patch `sys.path`, import each other, and depend on ignored maps/configs and on hardcoded local data paths. A `Verified:` line in `tasks.md` cannot be repeated by anyone else, and nothing reruns old checks after a change.

## Owner answers (2026-10-06)

- Checks were untracked out of habit, not confidentiality: tests are committed. Real acquisition data stays out of git.
- Scope: test infrastructure plus a first migration of the pure `src` checks; the other checks move when their spec is next touched.
- Timing: after spec 003 ships.
- Automation: none for now (no CI, no pre-commit). The owner runs `pytest` and records the result in `tasks.md`.

Clarify (2026-10-06):

- (Superseded 2026-10-07, see below.) Cornell maps: freeze copies of both `maps/cornell_map_full_system.yaml` and `maps/cornell_map_full_system_20260928.yaml` under `tests/data/`; `maps/` stays ignored. The two differ in SuperModule/port assignment of map entries 9–11, 15–20 and 25.
- Real-data baselines (e.g. `petsys_manager_t19_baseline_*.json`) are tracked under `tests/data/`; acquisitions and calibrations stay under `PETSYS_DATA_DIR`.
- Requirement ids: `NNN-FR-n` (e.g. `003-FR-21`); bug fixes `bug-<short-name>`.
- A migrated script is deleted from `scripts/` as soon as the pass counts match; earlier `Verified:` lines stay as history.
- `calculate_DOI` is not tested in this spec; the spec that next touches DOI adds it.
- Default run excludes only `real_data`; `gui` and `slow` tests run by default.
- The suite runs on Windows (`process_petsys` env) and on the Cornell Linux PC; `linux` tests run there.

Clarify (2026-10-07, found while planning):

- The committed `maps/{imas1DAQ,imas2DAQ,default,cpp,erc}_map.yaml` lack the mandatory `channels` key, so `map_factory` fails on a fresh clone. The owner commits the working-tree `channels:` lines as bug fix `bug-map-channels-key` before the fresh-clone check; tests use the tracked `maps/`.
- `src/read_compact.py` truncated last record: the test asserts every complete pair is returned intact and the partial record yields no pair. Today it yields a phantom `([], [])` (cut in detector 1), a half record with detector-1 hits and an empty detector 2 (cut in detector 2) or raises `struct.error` (header cut); the fix is bug fix `bug-read-compact-truncation`.
- `KevConverter.convert_mu` with `mu == 0` returns 0 keV, which is a fabricated value. The test asserts no number is returned; the fix is bug fix `bug-kev-mu-zero`.
- Cornell map: `maps/cornell_map_full_system.yaml` is the single Cornell full-system map; `cornell_map_full_system_20260928.yaml` was deleted by the owner. The owner force-adds `maps/cornell_map_full_system.yaml` to git in the same commit as `bug-map-channels-key`; tests read it from `maps/`, with no copy under `tests/data/`.
- Until its bug fix lands, a test that exposes one of these defects is `xfail(strict=True)` citing the bug id, so the default run passes and the fix flips it to a pass.

## Data contract

Tests only read `src/` and `exe_programs/`; they change no algorithm, cut, output format or GUI behavior. Synthetic LDAT fixtures follow the `src/read_compact.py` contract: coincidence records of two detectors, each a list of `(timestamp, channel energy, channel ID)` hits; no singles population. Maps come from the selected YAML (IMAS and Cornell layouts differ; the Cornell layout is the tracked `maps/cornell_map_full_system.yaml`); energies are PETsys a.u. unless a test names its calibration file. Baselines record which data, map, calibration and cuts produced them.

## Requirements

- **FR-1 — Tracked suite.** Checks SHALL live in a tracked `tests/` directory and run with `conda run -n process_petsys --no-capture-output python -m pytest` from the repo root. `pytest` SHALL be listed in `process_petsys.yml`. Configuration SHALL live in `pyproject.toml` (`[tool.pytest.ini_options]`: `testpaths`, `pythonpath`, registered markers, `--strict-markers`). No test SHALL modify `sys.path` or import from `scripts/`, `scripts_cornell/` or `scripts_imas/`.
- **FR-2 — Self-contained default run.** WHEN run with no options and no environment variables on a fresh clone, the suite SHALL use only tracked files (`tests/data/`, tracked `maps/` and `configs/`) and synthetic fixtures, SHALL write only under pytest's temporary directories, and SHALL pass. It SHALL never write to `configs/`, `maps/`, `encal_files/`, `tests/data/`, data folders or elsewhere in the repo tree.
- **FR-3 — Markers.** The suite SHALL register `real_data`, `gui`, `linux`, `slow` and `fr`. The default run SHALL exclude only `real_data`. `linux` tests SHALL be skipped on other platforms; `gui` tests SHALL use hidden Tk windows and be skipped, with the reason, when no display is available. Every skip SHALL state its reason in the pytest summary.
- **FR-4 — Real data.** Real-data tests SHALL locate acquisitions and calibrations relative to the `PETSYS_DATA_DIR` environment variable, never through an absolute path in a test file; their expected values SHALL come from baselines tracked under `tests/data/`. WHEN the variable or a named file is missing, the test SHALL be skipped with a reason naming what is missing; it SHALL NOT pass, substitute other data or report fabricated numbers.
- **FR-5 — Spec traceability.** Each test SHALL cite the requirement it checks with `@pytest.mark.fr("NNN-FR-n")`; bug-fix regression tests cite `@pytest.mark.fr("bug-<short-name>")`; a test MAY cite several ids. A `--fr <id>` option SHALL select exactly the tests citing that id, so a `Validation` walk can rerun one requirement's checks.
- **FR-6 — Shared fixtures.** `tests/conftest.py` SHALL provide the shared builders now duplicated across checks (synthetic compact LDAT writer with known pairs, timestamps, energies and channel IDs; map/config loaders; a `real_data_dir` fixture per FR-4). Tests SHALL share helpers only through `conftest.py` or a tracked helper module under `tests/`, never by importing another test file.
- **FR-7 — First migration (pure `src`).**
  - `scripts/cornell_slab_convention_check.py` SHALL become a parametrized test over the same eight cases for both `src.utils.get_slab_cornell` and `src.utils_fixed.get_slab_cornell_vectorized`, on the tracked `maps/cornell_map_full_system.yaml`, with the same seeds, giving 16 passing tests where the script reports `PASS: 16/16`. It cites `bug-cornell-slab-convention` (spec 002 B1) and `002-FR-17`.
  - New synthetic tests SHALL cover: `src/read_compact.py` reading a written fixture back to the same pairs (including an empty file and a truncated last record); `src.mapping_generator.map_factory` on every tracked map, IMAS and Cornell (each channel mapped once, SuperModule/minimodule ids and channel types consistent, IMAS and Cornell layouts distinct); `src.utils.get_absolute_id` / `get_electronics_nums` round trip; `src.utils.KevConverter` on a small synthetic calibration file (known factor gives the known keV; a channel without a factor is never given a fabricated value).
- **FR-8 — Gradual migration rule.** WHEN a later spec or bug fix changes an area covered by a `scripts/*_check.py` check, its tasks SHALL move that check to `tests/` first. A migrated test SHALL give the same pass count as the script on the same fixtures, recorded in that spec's `tasks.md`; the script SHALL then be deleted from `scripts/`. Its fixtures and baselines move to `tests/data/`. `unittest.TestCase` classes MAY be moved unchanged (pytest collects them); argparse mode flags become markers or `-k` selections. Until migrated, a script check remains valid evidence for the spec that recorded it.
- **FR-9 — Constitution and workflow.** `AGENTS.md` and `docs/prompts.md` SHALL state that checks live in tracked `tests/`; that `Done when:` names a pytest node id or `--fr` selection; that `Verified:` records the command, platform and its `N passed, M skipped` line; that a bug fix adds a regression test; and that `scripts/` stays for local, untracked analysis and the shared `scripts/template.py`.
- **FR-10 — No product change.** The spec SHALL NOT change behavior in `src/` or `exe_programs/`. WHEN a new test exposes a defect, it SHALL be recorded and fixed as a separate bug fix with its own regression test, not by adjusting the test's expected value.
- **FR-11 — Platforms.** The suite SHALL run in the `process_petsys` env on Windows and in the analysis Python env of the Cornell Linux PC, from a checkout of the same commit. Platform-specific tests SHALL carry `linux` (or a Windows-only skip with its reason) rather than failing on the other platform.

## Completion criteria

- `python -m pytest` in the `process_petsys` env on a fresh clone of the branch (Windows): all tests pass, `real_data` deselected, `linux` skipped with reason, no file written outside temporary directories (`git status --short` unchanged before and after).
- Same commit on the Cornell Linux PC: all tests pass, including `linux`.
- `PETSYS_DATA_DIR` unset with `-m real_data`: each real-data test skipped with a reason; set on the owner's PC: they pass.
- `pytest --fr bug-cornell-slab-convention` selects only the 16 slab-convention tests; `--strict-markers` rejects an unregistered marker.
- Slab-convention test count equals the script's `16/16`; `scripts/cornell_slab_convention_check.py` deleted.
- `AGENTS.md` and `docs/prompts.md` updated; `.gitignore` leaves `tests/` and `tests/data/` tracked (the `*.txt`, `*.csv`, `*.tsv` ignore rules do not hide test data).

## Out of scope

CI and pre-commit hooks; coverage thresholds; migrating all 31 checks now; `calculate_DOI` tests; `scripts/cornell_slab_en_cal_check.py` (it tests `scripts_cornell/cornell_slab_en_cal.py`, which is untracked); operator visual GUI review (still a person at the Cornell/IMAS PC); `scripts_cornell/` and `scripts_imas/` analysis scripts; packaging, build or release changes; changing `setup.py`.

## Open questions

None.
