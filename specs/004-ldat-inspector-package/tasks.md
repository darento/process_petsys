# Tasks 004 — LDAT Inspector package

Spec: [`spec.md`](spec.md). Plan: [`plan.md`](plan.md). User approved 2026-09-30; spec003 paused after T5. Local checks live in ignored `scripts/`; never stage automatically.

- [x] **T1 — Before-move fixture/source baseline** (FR-4). Snapshot source ASTs/hashes and deterministic IMAS/Cornell fixture results before moving files.

  **Done when:** `python scripts/ldat_package_check.py --baseline` records original four modules, shared/GUI fingerprints and exact fixture count/counter/array digests, without changing input/config/maps.

  **Verified 2026-09-30:** environment interpreter `scripts/ldat_package_check.py --baseline` → **PASS**. Original four modules' hashes and normalized AST bodies, shared/GUI/config/map fingerprints and exact IMAS/Cornell 120-pair fixture results captured before any source move. Relative `maps/...` fixture configuration exercises checkout-root fallback. Private baseline: `C:\Users\dsanchez\AppData\Local\Temp\opencode\ldat-package-baseline-e08y3mgw\baseline.json`. Check script remains ignored; no acquisition/hardware or user-input writes.

- [x] **T2 — Package move and import compatibility** (FR-1–FR-3). Relocate four implementations, change only internal imports/root depth, add compatible package/helper import paths.

  **Done when:** `python scripts/ldat_package_check.py --verify <baseline.json>` proves before/after numerical parity, unchanged non-import bodies, public/private import and patch compatibility, helper identity/shared worker state, pickle round-trip, actual Windows spawn processing and alternate-cwd relative-map lookup. Compile GUI/package/shims/launcher. No shared or GUI file changes.

  **Verified 2026-09-30:** environment interpreter `scripts/ldat_package_check.py --verify C:/Users/dsanchez/AppData/Local/Temp/opencode/ldat-package-baseline-e08y3mgw/baseline.json` → **PASS 24/24**. Before/after IMAS/Cornell array digests, counts, channel counters, selected sides and input bytes agree exactly. Normalized ASTs prove all four non-import source bodies unchanged, excluding only the intentional checkout-root depth fix. Package facade preserves public/private engine attributes and patch/restoration behavior; helper aliases share canonical module identity, patches and worker cancellation/progress globals. Settings and real spawned worker results pickle successfully for both systems; alternate-cwd subprocesses resolve relative selected maps. Shared/config/map/GUI/launcher hashes unchanged. Package/shims/GUI/launcher/check `py_compile` passes. Initial checks exposed facade patch restoration (fixed with stable original engine-name membership) and a check-only JSON integer-counter-key comparison mismatch (normalized; baseline numerical expectations unchanged). No numerical algorithm change. Full regressions remain T3.

- [x] **T3 — Regressions, documentation and migration revalidation** (FR-4–FR-5). Run inspected existing Inspector checks and spec003 T2–T5 checks, update README and acceptance evidence.

  **Done when:** environment checks pass: `ldat_inspector_check.py --selftest`, `ldat_scale_check.py --selftest`, `ldat_revision_check.py`, `ldat_issue_check.py`, `ldat_views_check.py`, `ldat_unpopulated_check.py`, `ldat_pair_choices_check.py`, `ldat_ports_check.py`, `cornell_slab_convention_check.py`, `ldat_processing_check.py`, `ldat_gui_check.py`; `petsys_manager_check.py --settings --commands --runner --artifacts`, `petsys_manager_numeric_check.py --formats`. T1 reference/helper fingerprint audit passes. README documents canonical/legacy imports; outcome table covers FR-1–FR-5. Spec003 stays paused until owner resumes.

  **Verified 2026-09-30:** Windows `process_petsys` environment interpreter (`C:/Users/dsanchez/AppData/Local/anaconda3/envs/process_petsys/python.exe`), unchanged fixture scripts; no `--real` or `--visible` acquisition runs:

  | Check (under `scripts/`) | Outcome |
  | --- | --- |
  | `ldat_inspector_check.py --selftest` | PASS 59/59 |
  | `ldat_scale_check.py --selftest` | PASS 65/65 |
  | `ldat_revision_check.py` | PASS 14/14 |
  | `ldat_issue_check.py` | PASS 8/8 |
  | `ldat_views_check.py` | PASS 171/171 |
  | `ldat_unpopulated_check.py` | PASS 6/6 |
  | `ldat_pair_choices_check.py` | PASS 15/15 |
  | `ldat_ports_check.py` | PASS 12/12 |
  | `cornell_slab_convention_check.py` | PASS 16/16 |
  | `ldat_processing_check.py` | PASS 17/17 |
  | `ldat_gui_check.py` | PASS 29/29 |
  | `petsys_manager_check.py --settings --commands --runner --artifacts` | PASS: 106 passed, 1 Linux-only skip / 107 selected |
  | `petsys_manager_numeric_check.py --formats` | PASS 38/38 |
  | Spec003 T1 source SHA-256 audit | PASS 24/24 unchanged |

  Inspector regressions total **412/412**. Processing cancellation returned control in 0.19 s and retained the previous dataset; closing during processing exited in 0.34 s. Both IMAS/Cornell hidden-GUI and PDF fixtures pass. An additional fresh-interpreter check confirms a cold package import leaves numba, fast reader, report and Tk unloaded; **104 legacy pickle class/function globals** resolve to the canonical objects. README documents canonical and compatible import paths; check script is confirmed ignored. `git diff --check` passes.

  First view invocation stopped while printing theta under Windows cp1252; reran unchanged with Python `-X utf8`, completing 171/171. That run emitted Tk scheduled-callback teardown diagnostics between withdrawn fixture windows, with exit 0 and all assertions passing; recorded without modifying the byte-identical GUI or widening expectations. No new visible UI, representative acquisition or hardware run was performed for this source-only move; shipped spec001/spec002 operator acceptance remains applicable to unchanged behavior. Spec003 remains paused at T6; its previously recorded Linux checks are not represented as freshly rerun here.

- [x] **T4 — Canonical GUI imports; retain compatibility aliases** (FR-2, FR-4, FR-6). Owner approved this follow-up on 2026-09-30. Change only the four GUI import paths; retain the root-level shims and engine facade.

  **Done when:** before/after normalized GUI ASTs match after excluding imports; protected source/config/map hashes remain unchanged; GUI uses canonical imports including lazy PDF loading. GUI/launcher/package/shims compile. Existing `ldat_gui_check.py` and `ldat_processing_check.py` pass unchanged for IMAS/Cornell fixtures, PDF writing, estimates, legacy patches, spawned workers and cancel/close. Legacy helper identities and pickle globals still resolve. Update README and FR-6 acceptance; leave spec003 paused.

  **Verified 2026-09-30:** changed exactly four import paths in `exe_programs/ldat_inspector_gui.py`: worker initializer → `src.ldat_inspector.fastread`, analysis → `.engine`, memory estimate → `.memory`, lazy PDF writer → `.report`. Root helper aliases, package facade and local scripts were retained unchanged. Before/after normalized GUI ASTs agree with imports excluded; all **59 protected source/config/map files** remain byte-identical. Snapshot: `C:/Users/dsanchez/AppData/Local/Temp/opencode/ldat-gui-imports-feff1fky/baseline.json`. Canonical engine/worker/memory bindings have identical objects; three legacy helper modules retain canonical identity and **104 legacy pickle class/function globals** resolve correctly.

  Environment interpreter `C:/Users/dsanchez/AppData/Local/anaconda3/envs/process_petsys/python.exe`: GUI/launcher/package/shims `py_compile` → exit 0; unchanged `-X utf8 scripts/ldat_gui_check.py` → **PASS 29/29**, `-X utf8 scripts/ldat_processing_check.py` → **PASS 17/17**. These cover both IMAS/Cornell fixture wiring and PDF output, legacy memory patches, worker progress/merge, four spawned workers, cancellation preserving prior data (0.20 s), and closing during processing (0.34 s, exit 0). `git diff` confirms import-only GUI edits; `git diff --check` passes. README explains canonical GUI imports and retained aliases. No numerical, widget or visible-behavior change; T1–T3 outcomes remain historical rather than newly rerun. Spec003 remains paused; no commits/pushes/builds.

- [x] **T5 — Remove compatibility aliases and package facade** (FR-7, Change 1). Owner approved on 2026-09-30. Delete the three root-level helper aliases and the `__init__.py` engine facade, and move scripts to canonical imports.

  **Done when:** `python scripts/ldat_package_check.py --removed <pre-edit snapshot>` passes: legacy modules absent and unimportable; package exposes no engine names and a cold import loads no engine/numba/matplotlib/Tk; canonical modules import and own the pickled classes; engine/memory/report byte-identical to the snapshot; fastread identical apart from docstrings/comments; no legacy import path left in `src/`, `exe_programs/`, `scripts/`, `scripts_imas/` or `scripts_cornell/` Python. The T3 regression list passes with only import/patch-path edits. Protected GUI/launcher/shared-source/config/map hashes are unchanged.

  **Verified 2026-09-30:** deleted `src/ldat_{fastread,memory,report}.py`; `src/ldat_inspector/__init__.py` is now a docstring only. Updated two docstring/comment references in `fastread.py` to `src.ldat_inspector.engine.*`. Mechanical import/patch-target rewrites went into 13 `scripts/` files (including tracked `scripts/template.py`, one line) and `scripts_cornell/cornell_slab_en_cal.py`. `ldat_package_check.py --verify` lost its legacy facade/alias assertions (T2 evidence stays historical) and gained `--removed`. Original CRLF/LF line endings were preserved. Nothing writes pickles to disk, so no saved-file compatibility is lost.

  Environment interpreter `C:/Users/dsanchez/AppData/Local/anaconda3/envs/process_petsys/python.exe`, `-X utf8`: `py_compile` of GUI/launcher/package/checks/template → exit 0; `ldat_package_check.py --removed` → **PASS 19/19**. The first run flagged `from src.ldat_inspector import fastread as ldat_fastread` in `ldat_processing_check.py`; rewritten as `import src.ldat_inspector.fastread as ldat_fastread`.

  | Check (under `scripts/`) | Outcome |
  | --- | --- |
  | `ldat_inspector_check.py --selftest` | PASS 59/59 |
  | `ldat_scale_check.py --selftest` | PASS 65/65 |
  | `ldat_revision_check.py` | PASS 14/14 |
  | `ldat_issue_check.py` | PASS 8/8 |
  | `ldat_views_check.py` | PASS 171/171 |
  | `ldat_unpopulated_check.py` | PASS 6/6 |
  | `ldat_pair_choices_check.py` | PASS 15/15 |
  | `ldat_ports_check.py` | PASS 12/12 |
  | `cornell_slab_convention_check.py` | PASS 16/16 |
  | `ldat_processing_check.py` | PASS 17/17 |
  | `ldat_gui_check.py` | PASS 29/29 |
  | `petsys_manager_check.py --settings --commands --runner --artifacts` | PASS 107 selected |
  | `petsys_manager_numeric_check.py --formats` | PASS 38/38 |

  Inspector regressions total **412/412**, with no tracebacks in the logs. A SHA-256 check of 47 protected files (GUI/launcher, `src/*.py`, configs, maps, engine/memory/report) against the pre-edit snapshot passes. `git diff --check` passes. No visible-UI, `--real` or hardware run; no numerical or behaviour change. Spec003 remains paused; no commits/pushes/builds.

## Requirement acceptance

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| FR-1 | Four canonical implementation files; T2 protected shared/GUI/launcher fingerprints; T4 GUI import-only body audit and protected files | PASS |
| FR-2 | T2 public/private identities, engine patch/restoration, helper identity/patch/state; T3/T4 104 legacy pickle globals; T4 retained aliases and legacy-patch processing checks | PASS |
| FR-3 | T2 alternate-cwd selected-map lookup, settings round-trip and actual Windows spawn results for IMAS/Cornell; processing 17/17 | PASS |
| FR-4 | T1 before-move fixture baseline; T2 unchanged normalized source bodies and exact result digests; T3 412 Inspector regressions | PASS |
| FR-5 | README imports; spec003 paused with T2–T5 revalidation and T1 fingerprint audit; no commits/pushes/builds | PASS |
| FR-6 | T4 canonical GUI imports/bindings, unchanged non-import AST/59 protected files, retained aliases/pickle globals, compile and 46/46 GUI/processing checks; alias-retention clause superseded by FR-7 | PASS |
| FR-7 | T5 `--removed` 19/19, canonical-only imports in all local Python, 412 Inspector regressions, protected hashes unchanged | PASS |

**Verdict: PASS.** T1–T5 complete; spec004 shipped. Spec003 progress and all existing user config/map changes preserved.
