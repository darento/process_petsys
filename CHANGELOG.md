# Changelog — exe_programs

This file tracks released versions of the programs under [`exe_programs/`](exe_programs/). Each program is versioned **independently** using [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **MAJOR** — incompatible changes (file formats, profiles, removed features).
- **MINOR** — new functionality, backwards compatible.
- **PATCH** — bug fixes, small tweaks.

## How versioning works

Each program declares its version as a manual `__version__` constant near the top of its GUI module:

| Program | Launcher | Version source |
|---|---|---|
| LDAT Inspector | [`exe_programs/LDATInspector.py`](exe_programs/LDATInspector.py) | [`exe_programs/ldat_inspector_gui.py`](exe_programs/ldat_inspector_gui.py) `__version__` |
| PETsys Manager | [`exe_programs/PETsysManager.py`](exe_programs/PETsysManager.py) | [`exe_programs/petsys_manager_gui.py`](exe_programs/petsys_manager_gui.py) `__version__` |

The version is shown in the window title bar. PETsys Manager also writes it in the first command-log line (`PETsys Manager <version>; checkout <path>`).

A program reaches a new MAJOR/MINOR version when the spec that changes it is `shipped`; a bug fix between specs is a PATCH.

## Release procedure

1. Make and test the change; the spec's checks pass.
2. Bump `__version__` in the program's GUI module.
3. Add an entry under that program's section below (date, version, summary, specs).
4. Stage the release files by name: `git add exe_programs/<program>_gui.py CHANGELOG.md`.
5. Commit: `git commit -m "<Program> v<x.y.z>: <summary>"`.
6. Tag, then push the commit **and** the tag:
   ```
   git tag <Program>-v<x.y.z>
   git push origin main --tags
   ```

Tag convention: `<Program>-v<MAJOR.MINOR.PATCH>`, e.g. `LDATInspector-v1.0.0`, `PETsysManager-v1.0.0`.

## LDAT Inspector

### 1.0.0 — 2026-10-06

First versioned release. Offline LDAT inspection as shipped by:

- [Spec 001](specs/001-ldatinspector-parity/spec.md) — offline inspection parity.
- [Spec 002](specs/002-ldatinspector-scale-views/spec.md) — scale, channel status and minimodule views.
- [Spec 004](specs/004-ldat-inspector-package/spec.md) — engine moved to the `src/ldat_inspector` package.

## PETsys Manager

### Unreleased (0.1.0)

Acquisition, RAW conversion, energy calibration, list-mode and QC workflows for the Cornell system ([spec 003](specs/003-petsys-manager-migration/spec.md), `in progress`). Becomes **1.0.0** when spec 003 is `shipped`.
