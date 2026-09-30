# Plan 004 — LDAT Inspector package

Owner-approved scope: [`spec.md`](spec.md). One named task at a time.

| Requirement | Implementation/check |
| --- | --- |
| FR-1 | Move four existing implementations; source-body audit; keep shared/GUI files byte-identical |
| FR-2 | Package facade delegates legacy engine attributes; three legacy helper modules alias canonical modules; identity/private-name/patch/worker-state checks |
| FR-3 | Engine uses its new depth for checkout-root lookup; alternate-cwd subprocess and explicit Windows spawn checks |
| FR-4 | Independent before/after fixture manifests and AST bodies; unchanged Inspector regression scripts |
| FR-5 | README import map, tasks evidence and spec003 pause/resume note |
| FR-6 | Four GUI import paths use canonical modules; retain shims/facade; source-body and hidden-GUI/process/report regression checks |
| FR-7 | Delete three shims; reduce `__init__.py` to a docstring; rewrite script imports/patch targets mechanically; `ldat_package_check.py --removed` plus existing Inspector regressions |

Canonical imports use `src.ldat_inspector.engine`, `.fastread`, `.memory` and `.report`. Internal imports use explicit relative module paths; no broad import rewrites in ignored scripts. `src.ldat_inspector` remains a package facade, preserving the original engine interface including private names and module-level monkeypatching. Helper compatibility modules refer to the actual canonical module object, preserving worker globals and patches. Package import does not eagerly load numba/report/GUI.

Moving the GUI or shared helpers, splitting the engine into new numerical abstractions, adding dependencies and modifying algorithms are outside this refactor. The only non-import engine edit is the `__file__` depth used to discover the checkout root. Spawn tests use fixture input and existing worker functions; no PETsys hardware.

Owner-approved follow-up: update GUI imports only (`engine`, `fastread`, `memory`, lazy `report`). Keep compatibility modules and local scripts unchanged. Capture the current GUI source body and protected source/config/map hashes before the import edit; check the body and hashes afterward. Compile the GUI/launcher and run existing hidden-GUI and processing checks, which exercise PDF output, estimates, legacy patches, spawned workers, cancellation and close. Prior T1–T3 evidence stays historical; the original GUI byte fingerprint is intentionally superseded by the import-only T4 change.

Change 1 (FR-7): nothing writes pickles to disk and the GUI already uses canonical imports, so the aliases only served local scripts. Rewrite `from src.ldat_inspector import` → `.engine`, `src.ldat_{fastread,memory,report}` → `src.ldat_inspector.{...}`, and engine patch targets → `src.ldat_inspector.engine.*`. Only docstring/comment references change in the implementations. `ldat_package_check.py` drops its legacy-compatibility assertions and gains `--removed`: legacy modules unimportable, package exposes no engine names, cold import loads no numba/report/Tk, no legacy references remain in tracked or local Python, implementation hashes match a pre-edit snapshot except fastread docstring/comment lines. T1–T4 evidence stays historical.
