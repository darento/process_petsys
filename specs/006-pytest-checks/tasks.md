# Tasks 006 — Tracked pytest checks

Spec: [`spec.md`](spec.md). Plan: [`plan.md`](plan.md). Owner approved 2026-10-07. Environment: `conda run -n process_petsys --no-capture-output python -m pytest` (or the env interpreter `-m pytest`) from the repo root. Never stage or commit; the owner runs git.

- [ ] **T0 — Owner: commit `bug-map-channels-key` and track the Cornell map** (precondition for T9). Commit the working-tree `channels:` lines in `maps/{imas1DAQ,imas2DAQ,default,cpp,erc}_map.yaml` and force-add `maps/cornell_map_full_system.yaml` (single Cornell full-system map; `_20260928` deleted by the owner). The `configs/*.yaml` diffs are line endings only; leave them out.

  **Done when:** `git show HEAD:maps/imas1DAQ_map.yaml` loaded by `map_factory` returns without `Missing mandatory key`, for all five maps; `git ls-files maps/cornell_map_full_system.yaml` lists it. T6's test on a fresh clone repeats both.

- [ ] **T1 — pytest install and config** (FR-1, FR-2, FR-3). Add `pytest==8.3.5` to `process_petsys.yml`; `pip install pytest==8.3.5` in the env; add `pyproject.toml` (`[tool.pytest.ini_options]` only) per plan; add `!tests/data/**` to `.gitignore`; create `tests/conftest.py` (empty apart from `pytest_plugins`).

  **Done when:** `python -m pytest --markers` lists `real_data`, `gui`, `linux`, `slow`, `fr`; `python -m pytest` exits 5 ("no tests ran") with no `.pytest_cache/` created; `git check-ignore -v tests/data/x.tsv tests/data/x.txt tests/data/x.csv` reports none ignored.

- [ ] **T2 — conftest fixtures, options and infra tests** (FR-3, FR-4, FR-5, FR-6). `--fr` option and deselection; `linux` collection skip; `tk_root` fixture; `real_data_dir` / `real_data_file`; `tests/helpers.py` with `write_ldat`, `load_map`, `REPO`, `DATA`. `tests/test_infra.py` uses pytester.

  **Done when:** `python -m pytest tests/test_infra.py` passes and covers: `--fr X` selects exactly the citing tests (one test citing two ids is selected by either); unregistered marker → collection error; default run deselects `real_data`; `-m real_data` with `PETSYS_DATA_DIR` unset → skipped, reason names the variable; set to an empty dir → skipped, reason names the missing file; `linux` test skipped with reason on Windows; `tk_root` gives a withdrawn window. No test edits `sys.path`.

- ~~**T3 — Freeze Cornell maps**~~ Dropped 2026-10-07: the Cornell map is tracked in `maps/` (T0); no frozen copies.

- [ ] **T4 — Migrate slab convention check** (FR-7, FR-8; cites `bug-cornell-slab-convention`, `002-FR-17`). `tests/test_cornell_slab.py` on `maps/cornell_map_full_system.yaml`.

  **Done when:** `python scripts/cornell_slab_convention_check.py` → `PASS: 16/16` and `python -m pytest tests/test_cornell_slab.py` → `16 passed`, same run; `python -m pytest --fr bug-cornell-slab-convention` → `16 passed`, rest deselected. Then `scripts/cornell_slab_convention_check.py` deleted.

- [ ] **T5 — read_compact tests** (FR-6, FR-7, FR-10). `tests/test_read_compact.py` using `write_ldat`.

  **Done when:** round-trip, empty-file and `en_filter` tests pass; truncated-data and truncated-header tests report `xfailed` with reason `bug-read-compact-truncation` (and fail before marking: phantom `([], [])` / `struct.error`, recorded here).

- [ ] **T6 — map_factory tests** (FR-7). `tests/test_mapping.py` over the six tracked maps (five IMAS-family + Cornell).

  **Done when:** all pass: shared key set of size `len(mod_feb_map) × 2 × len(channels_1)`; SuperModule/minimodule ranges; channel types per `sum_rows_cols`; IMAS vs Cornell types distinct. Counts in the test come from each YAML.

- [ ] **T7 — ID and KevConverter tests** (FR-7, FR-10). `tests/test_utils_ids.py`, `tests/test_kev_converter.py`, calibration files in `tmp_path`.

  **Done when:** ID round trips pass on boundaries + seeded sample; `mu`, `cornell` and `poly` known values pass; missing ID → `KeyError` passes; `mu == 0` test reports `xfailed` with reason `bug-kev-mu-zero` (fails before marking: returns `0`, recorded here).

- [ ] **T8 — Constitution and workflow docs** (FR-8, FR-9). Edit `AGENTS.md` and `docs/prompts.md` per plan.

  **Done when:** both files state: checks in tracked `tests/`; `Done when:` names a node id or `--fr`; `Verified:` = command, platform, `N passed, M skipped`; bug fix adds a regression test; gradual migration rule; `scripts/` local + `template.py`; `PETSYS_DATA_DIR`. `git diff --check` passes.

- [ ] **T9 — Validation** (all FR). Walk FR-1–FR-11 with checks.

  **Done when:** fresh clone of the owner's commit on Windows: `python -m pytest` → all pass, `real_data` deselected, `linux` skipped with reason, xfails only the two bug ids above; `git status --short` identical before/after. `-m real_data` without `PETSYS_DATA_DIR` → skips with reasons. `--strict-markers` rejection and `--fr` selection shown (T2). Same commit on the Cornell Linux PC: all pass including `linux` (owner run). Spec → `shipped`.

## Bug fixes found by this spec (separate from 006, FR-10)

Each one gets its own regression test (the xfail test above, with its marker removed) and is recorded here only as a cross-reference.

- `bug-map-channels-key` — committed maps lack `channels` (T0, owner; same commit tracks the Cornell map).
- `bug-read-compact-truncation` — phantom `([], [])` / `struct.error` on a truncated last record; fix: yield complete pairs, then raise `ValueError` with the byte offset.
- `bug-kev-mu-zero` — `convert_mu` returns 0 keV for `mu == 0`; fix: return `nan`, like `ldat_inspector.engine`.
