# Spec-driven development in process_petsys

Adapted from the workflow in sibling `process_cmb`, with [`AGENTS.md`](../AGENTS.md) as the constitution.

`Constitution → Spec → Clarify → Plan → Tasks → Implementation → Validation → Change`

1. **Spec:** interview about observable behavior, datasets, corner cases and scope; write numbered, stable `FR-n` requirements, completion criteria, and out-of-scope items in `specs/NNN-name/spec.md` (`Status: draft`). State the system type, calibration, energy units, channel/minimodule numbering, and sample population explicitly.
2. **Clarify:** identify contradictions and missing cases; resolve them with the owner and set `Status: clarified`. Only one spec may be `approved` or `in progress`. Owner approval sets `approved`.
3. **Plan:** after approval, write `plan.md` with architecture, data shapes, decisions, alternatives and an `FR-n → component` map. For GUI work, separate pure analysis from toolkit widgets and say which data is available only at ingest.
4. **Tasks:** write `tasks.md` in dependency order. Each task cites its `FR-n` and names a concrete `Done when:` check *before* coding: a pytest node id (`tests/test_x.py::test_y`) or a `--fr NNN-FR-n` selection, plus any manual step a person performs. Prefer synthetic fixtures with known expected output and boundary/negative cases to implementation-mirroring tests.
5. **Implementation:** execute the next task and its check. Record the actual outcome as `Verified:` with the command, the platform and pytest's `N passed, M skipped` summary line; tick it only on success. Surface blockers and requirement changes rather than guessing. Use the named `process_petsys` environment for checks.
6. **Validation:** walk every `FR-n` and cite its check (`pytest --fr NNN-FR-n`); require an interactive operator review for GUI behavior on representative IMAS/Cornell LDAT files. State a pass/fail verdict and set `Status: shipped` only when all completion criteria pass.
7. **Change:** when a new feature requirement arrives, assign the next `FR-n`, get approval for the spec delta, and revise plan/tasks before implementing it. Paused specs record why and what to revalidate on resumption.

The workflow is for future features throughout this repository. Bug fixes still require a regression test in `tests/` citing `bug-<short-name>`, without the overhead of a new feature directory.
