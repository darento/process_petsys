# Plan 007 — Move every remaining check into tests/

Owner-approved scope: [`spec.md`](spec.md) (approved 2026-10-07). One named task at a time; every migration task compares pass counts with its script before deleting it.

| Requirement | Implementation/check |
| --- | --- |
| FR-1 | One task per script group (tasks T4–T22); the retired scripts deleted in T1; `scripts/` scan for `*_check.py` in T24 |
| FR-2 | Each migration task: run the script (per mode) → N, run the pytest selection → N, record both, delete the script |
| FR-3 | `scripts/capture_golden_007.py` (local, untracked, one-off) runs the three `scripts_cornell/` oracles on helper fixtures → `tests/data/golden/<area>/` + `*.provenance.json`; `tests/test_golden_provenance.py` checks every golden file has provenance |
| FR-4 | `real_cal_file` fixture (`PETSYS_CAL_DIR`) next to spec 006's `real_data_file`; T19 manifest + result baseline under `tests/data/baselines/`; constants inline for the other kept real checks |
| FR-5 | `addopts` `-m "not real_data and not slow"`; `--slow-limit` option (conftest) fails any non-`slow` test ≥ 5 s, used at Validation; `test_infra.py` updated. Parallel (T25): `pytest-xdist` in `process_petsys.yml`, `addopts` `-n auto --dist loadscope`; conftest refuses `--slow-limit` with workers |
| FR-6 | `tests/manager_helpers.py`, `tests/manager_gui_helpers.py` (T15: GUI bases, Tk imports), `tests/ldat_helpers.py`; shared setup as mixins (non-`Test*` classes); GUI checks split per area |
| FR-7 | `gui`/`linux` markers on moved tests; Windows + Cornell full runs (T24) |
| FR-8 | `@pytest.mark.fr` ids copied from each script's docstrings/test names; task-only references mapped to their FR |
| FR-9 | `tests/test_filters.py`, `tests/test_filters_fixed.py`, `tests/test_fem_handler.py`, `tests/test_yaml_handler.py` (T2) |
| FR-10 | Defects → strict `xfail` + separate bug fix |
| FR-11 | `AGENTS.md`, `docs/prompts.md` (T23) |

## Run modes and markers

- Default: `python -m pytest` → `-m "not real_data and not slow" -n auto --dist loadscope`, budget 120 s.
- Full: `python -m pytest -m "not real_data"`; Validation and Done-when cite it.
- Real: `python -m pytest -m real_data` with `PETSYS_DATA_DIR` and `PETSYS_CAL_DIR`.
- Serial: `-n 0` (a later `-n` replaces the default one): one file or test, `-s`/`pdb`, and `--slow-limit`.
- A command-line `-m` replaces the default one (spec 006 decision), so the full run is one flag.
- `--slow-limit 5` (new conftest option): after each call phase, a test without `slow` that took ≥ 5 s fails with its duration. Off by default, so a slow machine never breaks a normal run; on at Validation, with `-n 0`: under parallel workers CPU contention stretches 3–4 s tests past 5 s, so the conftest refuses it with workers.

## Layout

```
tests/
  conftest.py            # + real_cal_file, --slow-limit
  helpers.py             # spec 006 (write_ldat, load_map, ...)
  manager_helpers.py     # FixtureProbe, fixture, FakeChild, FakeBackend, DirectDummyChild, FakeDaemon,
                         # FakeResources, fixture_map, compact_record, calibration specs/encoders,
                         # METADATA, encode_compact, ToolWorld; mixins for shared TestCase setup
  ldat_helpers.py        # fixture_files, write_pairs, build, CONFIGS, RECOVERY_CASES, side/ldat writers
  data/
    configs/cornell_january.yaml     # tracked copy of the January Cornell config (map_file → maps/..._old.yaml)
    configs/cornell_september.yaml   # tracked copy of the September Cornell config (T15 real selection)
    golden/{calibration,listmode,qc}/   # golden outputs + *.provenance.json
    baselines/t19_jan2026_manifest.json, t19_jan2026_result.json
  test_manager_*.py      # one per migrated manager script (gui split per area)
  test_ldat_*.py         # one per migrated LDAT script (views split per area)
  test_filters*.py, test_fem_handler.py, test_yaml_handler.py
```

## Decisions

- **Mixins, not imported test classes.** Today `cli`, `scope`, `workflow`, `gui` and `numeric` import `ListmodeChecks`, `QCChecks`, `CLIChecks`, `CalibrationChecks` to reuse setup. Imported into a test module, pytest would collect those classes twice. Shared `setUp`/helper methods move to `manager_helpers` mixins (`ListmodeFixtures`, `QCFixtures`, …); each test class inherits its mixin. The runner functions that aggregate classes (`petsys_manager_check.main`, `numeric_check --all`) are replaced by markers/paths.
- **Mode mapping.** Argparse modes become per-file selections: `petsys_manager_check --settings/--commands/--runner/--artifacts/...` → `tests/test_manager_<mode>.py`; `numeric_check --formats/--calibration/...` → the matching files. Each mode's script count is compared with its file's pytest count.
- **Golden capture, local and one-off.** The oracle code in the calibration/listmode/qc checks (`Oracle`, `self.ref.process_file`, reference loaders) moves into `scripts/capture_golden_007.py`, which imports `tests/manager_helpers.py` fixtures (same seeds) and writes golden files with provenance (oracle path + SHA-256, capture script SHA-256, fixture builder + seed, map, settings, date). Tests compare the Manager's output with the golden bytes/JSON. The capture script stays untracked; a future intentional output change regenerates golden files from the Manager's own output (FR-3), not from the oracles.
- **Tracked January config.** LDAT fixture checks read `configs/cornell_full_system_old.yaml` (untracked). A tracked copy at `tests/data/configs/cornell_january.yaml` with `map_file: maps/cornell_map_full_system_old.yaml` (resolved against the processing root, i.e. the repo) serves the fixture tests; real tests may use it too. T15 adds it, with the September copy `cornell_september.yaml`, for the GUI real split-selection test.
- **Spec 003 T19 real tests.** `scripts/petsys_manager_t19_baseline_jan2026.json` (operator manifest) becomes `tests/data/baselines/t19_jan2026_manifest.json` with paths relative to `PETSYS_DATA_DIR` and the January config; the newest Windows `real_baseline.json` (2026-10-05) becomes `t19_jan2026_result.json`. `petsys_manager_t19_baseline_cornell_f18.json` is retired with the other Cornell-PC real modes.
- **Retired data.** `scripts/fixtures_cornell_slab_spectra.json` is used only by `cornell_slab_en_cal_check.py` and goes with it.
- **Timing.** Every migration task runs its file with `--durations=0 -n 0`; tests ≥ 5 s get `slow`. The 120 s default budget is checked at T24 on a fresh clone.
- **Parallel default (T25, owner 2026-10-07).** `-n auto --dist loadscope`: each unittest class (one `setUpClass`, serial order) or module runs on one worker, as in a serial run. Every test already writes into private fixture folders. Measured after T14 on 24 CPUs: default 87 → 33 s, full 516 → 220 s. `--dist load` gave 25 s but splits classes across workers, each repeating `setUpClass`, so it was rejected. xdist omits `deselected` from the summary; `test_infra.py` inner runs pass `-n 0` to keep it. A single small file pays about 2 s of worker start-up (`-n 0` avoids it). The Cornell PC needs the updated environment before T24.
- **sys-level capture (T15).** With pytest's default fd-level capture, creating a Tk root on Windows intermittently failed with "Can't find a usable init.tcl/tk.tcl". It happened about once per serial GUI run and 5 times in one parallel full run. It reproduces outside pytest: dup2-swapping fd 1/2 to temp files fails 2–4 of 300–600 roots, against 0 of 1,100 without swaps. `addopts` sets `--capture=sys`; no test uses `capfd`, and the terminal output is unchanged.
- **Thread hygiene (T15).** `test_manager_daqd.py` left fake daemons running: six `petsys-daqd-1` threads stayed alive for the rest of the process. The GUI close test checks that no `petsys-*` thread is alive in its process, so it failed after them. `DaqdChecks.service()` now registers `service.close` as a cleanup; the tests and their expected values are unchanged.
- **Order.** Leaves before dependents: helpers first, then the manager group from `artifact` (no deps) up to `gui`; the golden capture before the oracle-based checks; the LDAT group after; docs and validation last. **Deferred deletion** (owner 2026-10-07): a migrated script that a remaining script still imports at module level stays on disk, unchanged, until its last importer is migrated, and is deleted in that importer's task; lazy imports in aggregate runners (`petsys_manager_check.py`, `numeric_check` modes) do not defer. `tasks.md` records the deferral.
- Rejected: moving test classes unchanged with cross-imports (double collection); live oracles behind an env var (owner chose golden files); one huge file per GUI check (owner chose per area).

## Risks

- Hidden coupling inside the 2.2k/2.5k-line GUI checks: split only at existing class/function boundaries; pass counts per mode guard against lost tests.
- Tk on Windows: spec 004 saw teardown noise between withdrawn windows; the `tk_root` fixture and per-test destroy keep windows isolated.
- Linux-only manager checks can only be confirmed at the Cornell PC (T24).
