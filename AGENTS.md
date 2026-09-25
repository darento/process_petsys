# process_petsys development guide

This file is the project's SDD constitution. Feature specs and plans refer to its rules instead of copying them. Preserve user data, configuration, map files, and untracked work.

## Spec-driven development

Follow [`docs/prompts.md`](docs/prompts.md) for feature work: **Spec → Clarify → Plan → Tasks → Implementation → Validation → Change**. Keep each feature in `specs/NNN-name/{spec,plan,tasks}.md`. Number requirements `FR-n` without renumbering; make acceptance criteria observable. A spec's `Status:` records its lifecycle; at most one is `approved` or `in progress`. Implement one named, verifiable task at a time; record check outcomes in `tasks.md`. Changes in scope update the spec before implementation. Small bug fixes need a reproducible regression check, but not a new feature spec.

## PETsys boundaries

- `src/read_compact.py` decodes LDAT coincidence records containing *two detectors*, each with `(timestamp, channel energy, channel ID)` hits. It does not expose a singles population. Never manufacture singles counts or label accepted coincidences as singles.
- `src/mapping_generator.py` builds channel-to-SuperModule/minimodule and local-coordinate mappings from YAML. Use the selected map rather than assuming the RAWInspector CMB layout, 48 modules, or 256 channels per SuperModule. IMAS and Cornell have different numbering and layouts.
- `src/utils.py:KevConverter` converts PETsys energy to keV using the selected calibration. Report which calibration and analysis cuts produced a measurement. `src/detector_features.py:calculate_DOI` returns a light-sharing ratio, not calibrated depth in millimetres.
- `scripts/`, `scripts_imas/`, and `scripts_cornell/` are local and untracked, except `scripts/template.py`, which is the starting point for new scripts using `src`. Their scripts may be interactive or use hardcoded paths: inspect them before running. Check scripts named in `tasks.md` live in `scripts/`; do not stage them. For the GUI, verify numerical logic headlessly first; a person checks the actual UI on representative acquisitions.
- Avoid reading every record into Python dictionaries without a bound. Keep processing off the Tk main thread and update widgets only on that thread. Guard incomplete input and missing calibration without showing fabricated numeric results.

## Environment and checks

Use `conda run -n process_petsys --no-capture-output python ...` or the corresponding environment interpreter; see `process_petsys.yml` for dependencies. Record a deterministic check for each numerical or file-processing task; compile-check the GUI; verify IMAS and Cornell with representative data. A full-system report needs explicit provenance, denominators, and a clear unavailable state for unsupported measurements. Do not silently import CMB-specific hardware or fit models into PETsys analysis.

Before modifying files, check `git status --short`; never overwrite existing user changes. Do not run release/build commands as a side effect of implementation.
