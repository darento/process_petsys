# process_petsys development guide

This file is the project's SDD constitution. Feature specs and plans refer to its rules instead of copying them. Preserve user data, configuration, map files, and untracked work.

## Spec-driven development

Follow [`docs/prompts.md`](docs/prompts.md) for feature work: **Spec → Clarify → Plan → Tasks → Implementation → Validation → Change**. Keep each feature in `specs/NNN-name/{spec,plan,tasks}.md`. Number requirements `FR-n` without renumbering; make acceptance criteria observable. A spec's `Status:` records its lifecycle; at most one is `approved` or `in progress`. Implement one named, verifiable task at a time; record check outcomes in `tasks.md`. Changes in scope update the spec before implementation. Small bug fixes need a regression test in `tests/`, but not a new feature spec.

## PETsys boundaries

- `src/read_compact.py` decodes LDAT coincidence records containing *two detectors*, each with `(timestamp, channel energy, channel ID)` hits. It does not expose a singles population. Never manufacture singles counts or label accepted coincidences as singles.
- `src/mapping_generator.py` builds channel-to-SuperModule/minimodule and local-coordinate mappings from YAML. Use the selected map rather than assuming the RAWInspector CMB layout, 48 modules, or 256 channels per SuperModule. IMAS and Cornell have different numbering and layouts.
- `src/utils.py:KevConverter` converts PETsys energy to keV using the selected calibration. Report which calibration and analysis cuts produced a measurement. `src/detector_features.py:calculate_DOI` returns a light-sharing ratio, not calibrated depth in millimetres.
- `scripts/`, `scripts_imas/`, and `scripts_cornell/` are local and untracked, except `scripts/template.py`, which is the starting point for new scripts using `src`. Their scripts may be interactive or use hardcoded paths: inspect them before running. For the GUI, verify numerical logic headlessly first; a person checks the actual UI on representative acquisitions.
- Avoid reading every record into Python dictionaries without a bound. Keep processing off the Tk main thread and update widgets only on that thread. Guard incomplete input and missing calibration without showing fabricated numeric results.

## Environment and checks

Use `conda run -n process_petsys --no-capture-output python ...` or the corresponding environment interpreter; see `process_petsys.yml` for dependencies. Record a deterministic check for each numerical or file-processing task; compile-check the GUI; verify IMAS and Cornell with representative data. A full-system report needs explicit provenance, denominators, and a clear unavailable state for unsupported measurements. Do not silently import CMB-specific hardware or fit models into PETsys analysis.

Checks live in the tracked `tests/` suite: `conda run -n process_petsys --no-capture-output python -m pytest` from the repo root. The default run uses only tracked files and synthetic fixtures, runs in parallel workers (pytest-xdist) and excludes `real_data` and `slow` tests; `-m "not real_data"` is the full run and `-n 0` runs serially. Mark a test `slow` when its setup or call takes 5 s or more serially; `-n 0 --slow-limit 5` fails an unmarked test whose call does, and wall-clock bounds are asserted only in serial runs. `real_data` tests name acquisitions relative to `PETSYS_DATA_DIR` and calibrations relative to `PETSYS_CAL_DIR`, never by absolute path, and skip with a reason naming a missing variable or file. Expected values once computed by untracked reference scripts are golden files under `tests/data/golden/`, and real-data baselines are under `tests/data/baselines/`, each with a provenance record; tests never import the references, and such a file changes only in a commit citing the requirement or bug id that authorizes the output change. Each test cites its requirement with `@pytest.mark.fr("NNN-FR-n")` (bug fixes: `"bug-<short-name>"`), and `--fr <id>` reruns that requirement's tests. A defect found by a new test gets a strict `xfail` citing its bug id until the separate fix lands; never edit the expected value to pass.

Which run:

- While coding: the default run; `-m "not real_data" --fr <id>` reruns one requirement with its `slow` tests.
- Before committing a `src/` or `exe_programs/` change: the full run. The default run skips the `slow` tests, which hold the bounded-memory, end-to-end and report checks.
- After adding or changing tests: `-m "not real_data" -n 0 --slow-limit 5`.
- When real-data results change (calibration, LM, QC numbers, LDAT counts) and in a spec's Validation: the full run plus `-m real_data`.
- When Linux-only parts change (process groups, DAQD, acquisition, PETsys Python) and before deploying a Manager release to Cornell: the full run on the Cornell PC. Run it from a terminal in its desktop session, because `gui` tests skip without a display. Use a clean checkout and an env matching `process_petsys.yml`.

Before modifying files, check `git status --short`; never overwrite existing user changes. Do not run release/build commands as a side effect of implementation.

Each `exe_programs/` program carries its own SemVer `__version__`; when its spec ships, bump it and add the entry by following [`CHANGELOG.md`](CHANGELOG.md).
