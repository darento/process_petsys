# Spec 004 — LDAT Inspector package

Status: `shipped`

Owner approved the proposed four-module package move on 2026-09-30. Constitution: [`AGENTS.md`](../../AGENTS.md); workflow: [`docs/prompts.md`](../../docs/prompts.md). Spec003 is paused after T5 at the owner's request.

Owner approved the follow-up GUI import cleanup and retention of compatibility aliases on 2026-09-30 (FR-6).

**Change 1 (2026-09-30, owner-approved):** remove the three root-level helper aliases and the package's legacy engine facade (FR-7). This supersedes the compatibility clauses of FR-2 and FR-6; their other clauses stand.

## Scope and data contract

Move only Inspector-specific engine, fast reader, memory estimate and PDF report implementation into `src/ldat_inspector/`. GUI and launcher stay in `exe_programs/`; GUI imports use the canonical package modules, including its lazy report import. Shared PETsys readers, mappings, calibration utilities, detector helpers, filters and fits stay in `src/`. No algorithm, GUI behavior, cut, format, output schema or dependency change. IMAS/Cornell selected maps, a.u./keV/calibration provenance and coincidence pair/side/hit populations retain their existing contracts. No singles population is introduced. Existing data/config/maps, local scripts and sibling repositories remain untouched.

## Requirements

- **FR-1** — Implementations SHALL live in `src/ldat_inspector/{engine,fastread,memory,report}.py`; launcher/GUI and shared helpers SHALL retain their paths and behavior.
- **FR-2** — Existing `src.ldat_inspector`, `src.ldat_fastread`, `src.ldat_memory` and `src.ldat_report` imports SHALL remain compatible, including used private names, module patches and fast-reader worker state. Canonical and legacy imports SHALL share classes/functions and helper module identity. *Superseded by FR-7 (Change 1).*
- **FR-3** — Relative processing-map fallback SHALL continue to resolve against the checkout root; launching/importing from another working directory SHALL work. Windows spawn workers and returned results SHALL remain importable/pickleable.
- **FR-4** — Analysis/filtering/calibration/report behavior SHALL remain unchanged. Before/after deterministic IMAS/Cornell fixture summaries/arrays and source bodies SHALL agree; existing numerical, hidden-GUI and process/cancel/close checks SHALL pass unchanged. Unavailable checks SHALL be recorded explicitly.
- **FR-5** — Documentation SHALL identify canonical and compatible import paths, record all validation outcomes, and preserve spec003 progress. No commit/push/build/release or external launch change SHALL occur.
- **FR-6** — The Inspector GUI SHALL import engine, worker initializer, memory estimate and lazy report writer from their canonical `src.ldat_inspector` submodules. Its non-import source body SHALL remain unchanged. ~~The three root-level helper aliases and package facade SHALL remain available for existing scripts, patches and pickle references.~~ *Superseded by FR-7 (Change 1).*
- **FR-7** — `src/ldat_fastread.py`, `src/ldat_memory.py` and `src/ldat_report.py` SHALL be removed, and `src.ldat_inspector` SHALL be a plain package exposing no engine names. Only `src.ldat_inspector.{engine,fastread,memory,report}` imports SHALL work. `scripts/template.py` and local scripts under `scripts/` and `scripts_cornell/` SHALL use canonical imports and patch targets. The four implementations' non-docstring source SHALL be unchanged; existing Inspector regression checks SHALL pass with only import/patch-path edits. No legacy module path is persisted to disk, so no saved-file compatibility is lost.

## Completion

All seven requirements checked and documented. GUI behavior is unchanged; hidden fixture GUI checks verify wiring, with existing shipped Inspector operator acceptance retained. Any numerical/visible-behavior change discovered requires explicit scope revision rather than becoming part of this move.
