# Plan 006 — Tracked pytest checks

Owner-approved scope: [`spec.md`](spec.md) (approved 2026-10-07). One named task at a time.

| Requirement | Implementation/check |
| --- | --- |
| FR-1 | `pytest==8.3.5` in `process_petsys.yml` (pip block); `pyproject.toml` `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `pythonpath = [".", "tests"]`, markers, `addopts = ["--strict-markers", "-m", "not real_data", "-p", "no:cacheprovider", "-p", "pytester"]`; `exceptiongroup==1.2.2` pinned (tensorflow 2.13 needs `typing-extensions<4.6`) |
| FR-2 | Tests write only to `tmp_path`/`tmp_path_factory`; `-p no:cacheprovider` (no `.pytest_cache` in the repo); `git status --short` before/after the fresh-clone run |
| FR-3 | Markers registered in `pyproject.toml`; `conftest.py` collection hook skips `linux` off Linux with reason; `tk_root` fixture creates a withdrawn `Tk()` and skips on `TclError` with reason |
| FR-4 | `real_data_dir` fixture reads `PETSYS_DATA_DIR`, skips naming the variable; `real_data_file(rel)` helper skips naming the missing file; baselines read from `tests/data/` |
| FR-5 | `fr(*ids)` marker; `--fr <id>` option in `conftest.py` deselects every item not citing `<id>` (reported as deselected) |
| FR-6 | `tests/conftest.py` fixtures; shared non-fixture code in `tests/helpers.py` (LDAT writer, map loaders) |
| FR-7 | `tests/test_cornell_slab.py`, `tests/test_read_compact.py`, `tests/test_mapping.py`, `tests/test_utils_ids.py`, `tests/test_kev_converter.py` |
| FR-8, FR-9 | `AGENTS.md` "Environment and checks" + PETsys boundaries `scripts/` bullet; `docs/prompts.md` steps 4–5 |
| FR-10 | Defects found by tests: `xfail(strict=True, reason="bug-…")` + separate bug fix; expected values never edited to match |
| FR-11 | `linux` marker + collection skip; Windows and Cornell Linux runs recorded in `tasks.md` |

## Layout

```
pyproject.toml            # pytest config only; setup.py unchanged
tests/
  conftest.py             # options/fixtures (pytester loaded by -p pytester in addopts)
  helpers.py              # write_ldat(), load_map(), REPO, DATA
  data/                   # future real-data baselines (none in this spec)
  test_infra.py           # pytester checks of FR-3/FR-4/FR-5 behavior
  test_cornell_slab.py
  test_read_compact.py
  test_mapping.py
  test_utils_ids.py
  test_kev_converter.py
```

`tests/` has no `__init__.py`; `pythonpath = [".", "tests"]` makes `src` and `helpers` (`from helpers import …`) importable without `sys.path` edits. Maps are read from tracked `maps/`; nothing goes in a `tests/**/maps/` directory, because `.gitignore` pattern `maps/` matches at any depth. `.gitignore` gains `!tests/data/**` after the `*.txt`/`*.csv`/`*.tsv`/`*.png` rules so test data stays tracked; checked with `git check-ignore -v`.

## Data shapes

- **Synthetic LDAT** (`helpers.write_ldat(path, pairs)`): `pairs` is a list of `(det1, det2)`, each a list of `(timestamp:int, energy:float, channel_id:int)`; written as `struct.pack("2B", len(det1), len(det2))` then `"qfi"` per hit, the `read_compact.read_binary_file` layout. At most 255 hits per detector (header byte). Coincidence records only; no singles population.
- **Maps:** tracked `maps/*.yaml`: IMAS 1DAQ/2DAQ, default, cpp, erc and Cornell `cornell_map_full_system.yaml` (force-added in T0). `map_factory` returns `(local_map, sm_mM_map, chtype_map, fem)`.
- **Calibration fixture:** written to `tmp_path` per test: `mu` type TSV `ID\tmu`, `cornell` type `ID(t_ch, slab)\tmu\tsigma`. Energies in PETsys a.u.; expected keV is `511 / mu × energy`.

## Tests and expected outcomes

- **Slab convention:** `parametrize` over the script's 8 `CASES` × `{scalar, vectorized}` = 16 ids; same geometry helper, `random.seed(1)`/`np.random.seed(1)` per case, 200 trials for the `random` case. Cites `bug-cornell-slab-convention`, `002-FR-17`. Must give `16 passed` against the script's `PASS: 16/16` on the same file, `maps/cornell_map_full_system.yaml`.
- **read_compact:** round trip of known pairs (multi-hit, both detectors, max channel id, negative-free timestamps); empty file → no pairs; `en_filter` drops low hits only; truncated data → complete pairs intact and no pair for the partial record; truncated header → same. The two truncation tests are `xfail(strict=True, reason="bug-read-compact-truncation")` until the fix. Fix contract (for the bug fix, not this spec): yield every complete pair, then raise `ValueError` naming the byte offset. That matches `ldat_inspector.engine.iter_pairs` and `cornell.qc`, which already reject truncation. Its only caller is `exe_programs/LDATInspector_legacy.py`.
- **map_factory:** per map: the three dicts share one key set, of size `len(mod_feb_map) × 2 × len(channels_1)` (no channel overwritten); every value's SuperModule is a `mod_feb_map` key and minimodule in `[0, n_mM)`; `chtype_map` is `[TIME, ENERGY]` for both channels when `sum_rows_cols` is false, and `[TIME]`/`[ENERGY]` when it is true; IMAS maps (`sum_rows_cols` false) and Cornell maps (true) differ in channel types. Invariants are computed from each YAML, not hard-coded per-map counts, so a map edit changes the expectation rather than breaking it silently.
- **IDs:** `get_absolute_id(*get_electronics_nums(c)) == c` over boundary channel ids (0, 63, 64, 4095, 4096, 131071, 131072, and the maximum for port 7/slave 31/chip 63/channel 63) plus a seeded sample of 1000; the reverse direction over every chip/channel at port 0–1, slave 0–1.
- **KevConverter:** `mu` and `cornell` files: known factor → `511/mu × E`; missing ID → `KeyError` (no value); `mu == 0` → no number returned: `xfail(strict=True, reason="bug-kev-mu-zero")`. Fix contract (for the bug fix): return `nan`, matching `ldat_inspector.engine` `converted()`, which leaves non-finite or ≤0 factors as `nan`. `poly` type: known coefficients → known value.
- **Infra (`test_infra.py`, pytester):** `--fr X` selects exactly the citing tests; an unregistered marker errors under `--strict-markers`; default run deselects `real_data`; `-m real_data` without `PETSYS_DATA_DIR` skips with a reason naming it; a `linux` test is skipped with reason on Windows. The first migration has no real-data test, so FR-4's "set on the owner's PC: they pass" is covered by this infra test until a real-data check migrates (FR-8).

## Decisions and alternatives

- **pytester over manual CLI checks** for FR-3/FR-5: the marker/option behavior becomes a tracked regression test instead of a one-off `Verified:` line.
- **`-p no:cacheprovider`** keeps pytest from writing `.pytest_cache/` into the repo (FR-2). Python's `__pycache__/` bytecode is interpreter behavior, already ignored, and outside FR-2's data-write intent; not suppressed.
- **`addopts -m 'not real_data'`:** a command-line `-m` replaces it (last `-m` wins), so `-m real_data` selects only real-data tests. `--fr X` composes with the default `-m`; to include real-data tests for an id use `--fr X -m "real_data or not real_data"`. This goes in the docs.
- **pytest 8.3.5:** supports the env's Python 3.10.14; reads `pyproject.toml` through the bundled `tomli`. Installed with `pip install pytest==8.3.5` into the env (no other package changes).
- **Defects use strict xfail** rather than skipping or changing expected values (FR-10); the bug fix removes the marker, and `strict` makes an unnoticed fix fail the run.
- **Maps:** tracked `maps/` used directly (owner commits `bug-map-channels-key` and force-adds the Cornell map first); frozen `tests/data/` copies were declined (2026-10-07): a local map edit now shows in `git status` and in test results instead of hiding behind a stale copy.
- Rejected: `unittest` style, a custom runner script, `tox`/`nox`, CI (out of scope).

## Docs (FR-9)

`AGENTS.md`: replace "Check scripts named in `tasks.md` live in `scripts/`; do not stage them." with tracked `tests/`, `scripts/` for local analysis + `template.py`; "Environment and checks" gets the pytest command, `--fr`, `PETSYS_DATA_DIR`, `Verified:` format (command, platform, `N passed, M skipped`), bug fix → regression test, gradual migration rule. `docs/prompts.md` steps 4–5 name a pytest node id or `--fr` selection in `Done when:`, and the `Verified:` format.
