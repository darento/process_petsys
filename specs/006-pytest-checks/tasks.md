# Tasks 006 — Tracked pytest checks

Spec: [`spec.md`](spec.md). Plan: [`plan.md`](plan.md). Owner approved 2026-10-07. Environment: `conda run -n process_petsys --no-capture-output python -m pytest` (or the env interpreter `-m pytest`) from the repo root. Never stage or commit; the owner runs git.

- [x] **T0 — Owner: commit `bug-map-channels-key` and track the Cornell map** (precondition for T9). Commit the working-tree `channels:` lines in `maps/{imas1DAQ,imas2DAQ,default,cpp,erc}_map.yaml` and force-add `maps/cornell_map_full_system.yaml` (single Cornell full-system map; `_20260928` deleted by the owner). The `configs/*.yaml` diffs are line endings only; leave them out.

  **Done when:** `git show HEAD:maps/imas1DAQ_map.yaml` loaded by `map_factory` returns without `Missing mandatory key`, for all five maps; `git ls-files maps/cornell_map_full_system.yaml` lists it. T6's test on a fresh clone repeats both.

  **Verified 2026-10-07:** owner commit `f0ae742`. `git show HEAD:` copies of the six maps loaded by `map_factory` in the env interpreter: cornell 7680, cpp 256, default 1792, erc 1792, imas1DAQ 30720, imas2DAQ 30720 channels, no error; `git ls-files` lists `maps/cornell_map_full_system.yaml`.

- [x] **T1 — pytest install and config** (FR-1, FR-2, FR-3). Add `pytest==8.3.5` to `process_petsys.yml`; `pip install pytest==8.3.5` in the env; add `pyproject.toml` (`[tool.pytest.ini_options]` only) per plan; add `!tests/data/**` to `.gitignore`; create `tests/conftest.py` (empty apart from `pytest_plugins`).

  **Done when:** `python -m pytest --markers` lists `real_data`, `gui`, `linux`, `slow`, `fr`; `python -m pytest` exits 5 ("no tests ran") with no `.pytest_cache/` created; `git check-ignore -v tests/data/x.tsv tests/data/x.txt tests/data/x.csv` reports none ignored.

  **Verified 2026-10-07 (Windows, env interpreter `C:/Users/dsanchez/AppData/Local/anaconda3/envs/process_petsys/python.exe`):** `pip install pytest==8.3.5` also upgraded `typing-extensions` to 4.16.0 (via `exceptiongroup` 1.3.1), breaking tensorflow 2.13's `<4.6.0` pin; restored with `exceptiongroup==1.2.2 typing-extensions==4.5.0`, `pip check` → no broken requirements, tensorflow 2.13.0 imports. `exceptiongroup==1.2.2` pinned in `process_petsys.yml` next to pytest. `python -m pytest --markers` lists all five markers; `python -m pytest` → `collected 0 items`, `no tests ran`, exit 5, no `.pytest_cache/`. `git check-ignore --no-index tests/data/x.{tsv,txt,csv,png} tests/data/sub/x.json tests/conftest.py pyproject.toml` → none ignored (exit 1); `tests/__pycache__/*.pyc` still ignored. pytester is loaded with `-p pytester` in `addopts` rather than `pytest_plugins` in `conftest.py` (same effect, no conftest-level plugin rule); `conftest.py` is a docstring only until T2. `git diff --check` passes.

- [x] **T2 — conftest fixtures, options and infra tests** (FR-3, FR-4, FR-5, FR-6). `--fr` option and deselection; `linux` collection skip; `tk_root` fixture; `real_data_dir` / `real_data_file`; `tests/helpers.py` with `write_ldat`, `load_map`, `REPO`, `DATA`. `tests/test_infra.py` uses pytester.

  **Done when:** `python -m pytest tests/test_infra.py` passes and covers: `--fr X` selects exactly the citing tests (one test citing two ids is selected by either); unregistered marker → collection error; default run deselects `real_data`; `-m real_data` with `PETSYS_DATA_DIR` unset → skipped, reason names the variable; set to an empty dir → skipped, reason names the missing file; `linux` test skipped with reason on Windows; `tk_root` gives a withdrawn window. No test edits `sys.path`.

  **Verified 2026-10-07 (Windows, env interpreter):** `python -m pytest tests/test_infra.py -rs` → **13 passed**. Each infra check runs a pytester subprocess project with copies of the real `pyproject.toml` and `tests/conftest.py`: `--fr 006-FR-5` / `--fr bug-x` each select exactly their two citing tests (one test cites both), 2 deselected; no `--fr` runs all 4; `@pytest.mark.unknownmark` → error `'unknownmark' not found in markers`; default run deselects `real_data`; `-m real_data` → skipped `real_data: PETSYS_DATA_DIR is not set`, with an empty dir → skipped `real_data: missing acq/run.ldat under PETSYS_DATA_DIR=…`, with the file present → passed; `linux` test skipped `linux: runs only on Linux (Cornell PC), not win32` (passes on Linux); `tk_root` state `withdrawn`; AST scan of every `tests/*.py`: no `sys.path` use, no `scripts*`/test-file imports. Negative check: a temporary conftest copy with `--fr` filtering, the `linux` skip and the unset-variable message broken → 4 failed / 9 passed; original restored byte-identical (`cmp`). Full `python -m pytest -q` → 13 passed, no `.pytest_cache/`, `git status --short` identical before/after; `--fr 006-FR-4` → 3 passed, 10 deselected; `-m real_data` → 13 deselected (no real-data tests yet). `git diff --check` passes. `helpers.py`: `write_ldat` (`2B` + `qfi`, read_compact layout, ≤255 hits), `load_map`, `load_config`, `REPO`/`DATA`/`MAPS`/`CONFIGS`; first exercised in T4–T6.

- ~~**T3 — Freeze Cornell maps**~~ Dropped 2026-10-07: the Cornell map is tracked in `maps/` (T0); no frozen copies.

- [x] **T4 — Migrate slab convention check** (FR-7, FR-8; cites `bug-cornell-slab-convention`, `002-FR-17`). `tests/test_cornell_slab.py` on `maps/cornell_map_full_system.yaml`.

  **Done when:** `python scripts/cornell_slab_convention_check.py` → `PASS: 16/16` and `python -m pytest tests/test_cornell_slab.py` → `16 passed`, same run; `python -m pytest --fr bug-cornell-slab-convention` → `16 passed`, rest deselected. Then `scripts/cornell_slab_convention_check.py` deleted.

  **Verified 2026-10-07 (Windows, env interpreter):** same session, same map file: `python -X utf8 scripts/cornell_slab_convention_check.py` → `PASS: 16/16`; `python -m pytest tests/test_cornell_slab.py` → **16 passed**; `--fr bug-cornell-slab-convention` → 16 passed, 14 deselected; `--fr 002-FR-17` → the same 16. Cases, geometry lookup, seeds (`random.seed(1)`, `np.random.seed(1)` per case and implementation) and 200 trials for the one-channel middle case are unchanged; the `check()` table became `parametrize` (8 cases × scalar/vectorized). Negative check: a scratchpad copy with the half-slab sign flipped and the p-1 case expecting slab 7 → 14 failed, 2 passed (the two unresolved cases have no x). Script deleted from `scripts/` (copy kept in the session scratchpad); no other script imports it; references in specs 001–004 stay as history. Full `python -m pytest -q` → 30 passed.

- [x] **T5 — read_compact tests** (FR-6, FR-7, FR-10). `tests/test_read_compact.py` using `write_ldat`.

  **Done when:** round-trip, empty-file and `en_filter` tests pass; truncated-data and truncated-header tests report `xfailed` with reason `bug-read-compact-truncation` (and fail before marking: phantom `([], [])` / `struct.error`, recorded here).

  **Verified 2026-10-07 (Windows, env interpreter):** `python -m pytest tests/test_read_compact.py -rx` → **3 passed, 3 xfailed** (reason `bug-read-compact-truncation`). Passing: round trip of 3 known pairs (multi-hit, timestamps to 2^40+9, channel IDs 0/63/64/4095/4096/131071/131072 and port 7/slave 31/chip 63/channel 63); empty file → no pairs; `en_filter=4.0` keeps hits ≥ 4.0 (boundary kept), drops the rest, keeps all 3 records. Before marking (`--runxfail`) → 3 failed: cut in the header → `struct.error: unpack requires a buffer of 2 bytes`; cut in detector-1 hits → DID NOT RAISE, extra `([], [])`; cut in detector-2 hits → DID NOT RAISE, extra record with detector 1's 3 hits and an empty detector 2 (half a coincidence, which reads like a single). The test asserts the planned fix contract: the complete pair intact, then `ValueError`.

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
- `bug-read-compact-truncation` — on a truncated last record: phantom `([], [])` (cut in detector 1), a half record with detector-1 hits and empty detector 2 (cut in detector 2), or `struct.error` (cut in header); fix: yield complete pairs, then raise `ValueError` with the byte offset.
- `bug-kev-mu-zero` — `convert_mu` returns 0 keV for `mu == 0`; fix: return `nan`, like `ldat_inspector.engine`.
