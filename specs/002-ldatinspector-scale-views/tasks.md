# Tasks 002 — LDATInspector scale, channel status and minimodule views

Dependency order. Each task cites its FRs and states its `Done when:` check before coding. Record the actual output under `Verified:` and tick a task only on success. Run checks with the `process_petsys` interpreter. The owner commits; tasks never commit.

## Prerequisite bug fix (outside the FR list; owner decision 2026-09-25)

- [x] **B1 — Consistent Cornell slab convention** (affects FR-1 oracle and FR-17). The owner first confirms the physical convention for two-adjacent-time-channel events. *Done when:*
  - a regression check reproduces the current disagreement (red): for position p with neighbour p−1, scalar `get_slab_cornell` gives slab `2p` with X + 0.8 mm, and vectorized `get_slab_cornell_vectorized` gives slab `2p+1` with X + 0.8 mm;
  - after the fix, both functions give the confirmed slab index and a matching X shift on edge, random, adjacent and non-adjacent cases;
  - the report states which calibration/limits files were generated with which convention (from the owner's generation records or scripts).

  **Owner convention 2026-09-25:** slab `2p` at X_p − 0.8 mm; the event is in slab `2p` when neighbour p−1 also fires. **Red baseline:** `python scripts/cornell_slab_convention_check.py` → **FAIL 12/16**. The four adjacent-neighbour cases failed: scalar gave slab 6 at X 89.52 (expected 87.92) and slab 7 at X 87.92 (expected 89.52); vectorized gave slab 7 at X 89.52 and slab 6 at X 87.92. **Verified 2026-09-25:** after fixing the X sign in `src/utils.py:get_slab_cornell` and the slab offset plus X sign in `src/utils_fixed.py:get_slab_cornell_vectorized`, the check gives **PASS 16/16**, covering edges, one-channel random (200 trials, both slabs, X consistent), both adjacent directions and non-adjacent. Existing checks are unchanged: engine/report 59/59, revision 14/14, issue 8/8, hidden GUI 29/29. **Provenance (owner):** the inspector's `…coincFixed11s_fixed.encal` was generated with the vectorized `*_fixed` scripts (old, swapped keys for two-adjacent events). The COG/DOI limits files come from `cornell_cog_decompress_params.py` (scalar; slab index unchanged by this fix). **Consequence:** the scalar slab index is unchanged, so inspector keys and the limits files keep their meaning; only scalar X outputs move. The vectorized slab IDs change for two-adjacent events, so existing `*_fixed` encal files no longer match the fixed vectorized scripts and should be regenerated. Until then the inspector applies that encal's factors to about 44 % of sides whose keys were swapped when it was built.

- [x] **B2 — Cornell slab energy calibration picks the wrong peak** (bug fix in `scripts_cornell/cornell_slab_en_cal.py`; owner decision 2026-09-25). *Done when:*
  - `scripts/cornell_slab_en_cal_check.py` pins, on real whole-file spectra: the old finder's failure, resolved-only selection, correct μ for an inner edge slab and an ordinary slab, flagging of a higher-energy photopeak, an explicit no-fit, outer-slab borrowing, and `.encal` compatibility with `KevConverter`;
  - existing checks are unchanged.

  **Evidence (whole files `coincCompact11s_00000003`–`08`: 14.86 M pairs, 24.1 M resolved sides):**
  - **Why one-channel sides are left out:** their energy is mostly below the photopeak. Median energy: coin flip 40.8 a.u., outer edge one channel 47.9, versus 93.5 for two adjacent channels and 93.7 for edge with two channels.
  - **Outer slabs 0/15:** they only receive one-channel sides, and the half-height finder then locks onto the low bump (slab `(256, 0)`: μ 36.8 a.u.; `(787291, 0)`: 32.8).
  - **The fit model is not the issue:** on resolved-only spectra the old Gaussian and the background + Gaussian agree to a median 0.64 %.
  - **Silent fallback:** 163 slabs fell back to the mean energy without warning.
  - **Correction to an earlier explanation:** double photopeaks persist with two-adjacent-channel sides only (e.g. `(842, 8)`: ~59 and ~99 a.u.), so they are not caused by the coin flip.

  **Red baseline:** the check against the original script (backup in the session scratchpad; `scripts_cornell/` is gitignored) fails with `AttributeError: … no attribute 'is_resolved_slab'`.

  **Verified 2026-09-25:**
  - `python scripts/cornell_slab_en_cal_check.py` → **PASS 10/10**.
  - The check's (256, 1) expectation was first too strict: its lower bump at ~54 a.u. was flagged. The flag rule was narrowed to *higher-energy* peaks after measuring, over 5,470 fitted slabs, ~2,660 lower bumps (expected) against ~115 higher-energy peaks.
  - Shared `src/ldat_inspector.py:fit_peak_background` gained `sigma0` / `mu_halfwidth` with unchanged keV defaults. Engine/report 59/59, revision 14/14, issue 8/8, hidden GUI 29/29, slab convention 16/16; compile exit 0.
  - Whole-file run: 5,470 slabs fitted, 791 outer slabs borrowing slab 1/14, 87 explicit no-fits. Output: `encal_files/…coincCompact11s_resolved.encal` plus `…_resolved_status.txt`; the earlier encal files are unchanged.
  - **Owner follow-up 2026-09-25:**
    - **Per-channel cut applied:** `process_file` passes `en_min_ch` (0.2 a.u.) to `read_binary_file`. The check pins it: 5 / 0 synthetic pairs pass at cut 0.0 / 0.2; the old script could not take the argument. On this acquisition the cut removes 945 of 12,557,213 events and changes no fitted μ.
    - **Estimates for populated slabs without a fit:** `estimate_missing_slabs` uses the mean of the cleanly fitted physical neighbours (slab ±1), else the minimodule median, and needs ≥ 4 clean fits per minimodule. Leave-one-out on 5,356 clean fits: neighbours 2.8 % median error (96.3 % within 10 %); minimodule median 3.2 % (94.0 %). The check has 15/15.
    - **Whole-file result:** 5,470 fitted (114 to check), 791 borrowed, 139 estimated (80 neighbours, 59 minimodule median). Exactly the 1,280 unpopulated slabs have no factor, and none were estimated.
    - **Files:** the owner-reviewed version is kept as `…_resolved_nocut.encal` / `_status.txt`.

- [x] **B3 — Half-populated Cornell SuperModules from the config** (bug in spec 001's `src/ldat_inspector.py:merge_results`, copied from `LDATInspector_legacy.py:37`; owner-confirmed 2026-09-25). *Done when:*
  - `scripts/ldat_unpopulated_check.py` shows the expected minimodules from `configs/cornell_full_system.yaml:unpopulated_minimodules`: SM 2, 5, …, 29 without {2, 3, 6, 7, 10, 11, 14, 15}, every other SM with 16;
  - a config without the key expects everything, and a malformed key is rejected;
  - with `--real`, a Cornell prefix has no sides in unpopulated minimodules and sides in every expected one.

  **Evidence:** the whole-file 2026-01-19 acquisition has no sides in exactly those 80 minimodules. The old rule expected all 16 in SM 2–17 and ignored populated minimodules in SM 21, 22, 24, 25, 27 and 28.

  **Red baseline:** `--real` → **FAIL 1/8** (SM 2 expected 16 minimodules, SM 21 expected 8; no config key; malformed key accepted).

  **Verified 2026-09-25:** **PASS 8/8** (234,284 accepted pairs in a 300,000-pair prefix of `00000003`). Engine/report 59/59, revision 14/14, issue 8/8, hidden GUI 29/29, slab convention 16/16, scale selftest 17/32 (red until T2); compile exit 0.
  - **Config backup:** `configs/cornell_full_system.yaml` is gitignored; the original is in the session scratchpad.
  - **Other Cornell configs** (`cornell_1cassettes`, `cornell_2cassettes`, `cornell_FEM256*`) do not declare the key. They need it if their setups include half-populated SMs.

## Scale

- [x] **T1 — Reference oracle and edge fixtures** (FR-1, FR-15). Keep spec 001's reader as `process_file_reference`. Add `scripts/ldat_scale_check.py` with the synthetic IMAS/Cornell edge fixtures from the plan and a comparison harness. *Done when:* `--selftest` runs the reference on every fixture and prints expected counts/error labels. The fast-reader comparisons are present and fail as not implemented (red baseline). The four existing checks still report 59/59, 14/14, 8/8 and 29/29.

  **Verified 2026-09-25:** `python scripts/ldat_scale_check.py --selftest` → **FAIL 17/32**, the intended red baseline. All 17 reference checks pass and the 15 fast-reader comparisons fail as "not implemented".
  - **IMAS mixed file:** 36 read, 18 accepted, `min channels` 9, `KeyError` 6, `ValueError` 3.
  - **Cornell mixed file:** 42 read, 18 accepted, the same labels plus `unresolved Cornell slab` 6.
  - **Cornell SM 0 slabs:** {0, 6, 7, 14}, covering the edge, adjacent and random cases.
  - **Failures with no partial data:** truncated header, truncated hit and empty file.
  - **Other pinned behaviour:** a labelled 7-pair prefix, counters that exclude the unselected minimodule, and zero-energy minimodules rejected with `ZeroDivisionError`.
  - **Precedence caught by a wrong first expectation:** zero-energy minimodules fail at DOI (sum/max) before the `energy > 0` check. The fast reader must keep that order.

  `--real` → the reference reproduces spec 001 T9 exactly for all six 80,000-pair Cornell prefixes (e.g. `00000003`: 62,460 accepted, 11,691 min channels, 5,457 unresolved slab, 392 `ValueError`). The 200,000-pair fast comparison is red until T2.

  Existing checks: 59/59, 14/14, 8/8, 29/29. Compile exit 0.
- [x] **T2 — numba fast reader** (FR-1). Write `src/ldat_fastread.py` and declare `numba==0.63.1` in `process_petsys.yml`. *Done when:*
  - `ldat_scale_check.py --selftest` shows fast = reference on all fixtures: exact accepted/read counts, error-label counts and channel counters; columns identical, except slab bits of `random_slab` sides, whose counts match;
  - `--real` matches the reference on the 200,000-pair Cornell `00000003` prefix and reproduces spec 001 T9's six 80,000-pair rows;
  - single-core throughput is recorded against the 14,500 pairs/s baseline;
  - the tie count is reported.

  **Red baseline:** T1's 15 fast comparisons failed ("not implemented"). The first real run matched every count and label exactly but failed on centroid Y: numba lowers `(E + 1e-5) ** 2` to `x * x`, while CPython calls the C runtime `pow`. That gave 49 of 78,062 sides at ≤ 3 ulp (2.8e-14 mm). `x * x` differs from Python's `**` on 115 of 200,000 random values, and `ucrtbase` `pow` on 0.

  **Verified 2026-09-25:**
  - `src/ldat_fastread.py:process_file_fast`: numba header scan, NumPy header stripping into aligned hit arrays in 250,000-pair chunks, and a numba per-pair kernel following the reference's order. The squared weight calls the C `pow` through an external symbol.
  - `python scripts/ldat_scale_check.py --selftest` → **PASS 32/32**.
  - `--real` → **PASS 13/13**: the fast reader reproduces all six spec 001 T9 80,000-pair rows, and equals the reference exactly on the 200,000-pair Cornell `00000003` prefix (all columns, counters and labels; slab only compared as a pair on the 67,179 coin-flip sides).
  - **Throughput, single core:** reference 14,349 pairs/s, fast **589,619 pairs/s (41×)**.
  - **Exact minimodule-energy ties:** 0.
  - `numba==0.63.1` is declared in `process_petsys.yml`.
  - **Not yet:** `process_file` still points to the reference; it switches in T4, where `ldat_inspector_check.py`'s `get_slab_cornell` patch must be adapted. `ModuleEvents.random_slab` is set per file and is not merged until T3.
- [x] **T3 — Compact SM-sorted side table with partner SM/mM** (FR-2, FR-11). *Done when:*
  - fixtures give the expected `partner_sm`/`partner_mm` for every side;
  - measured table bytes/side ≤ 64;
  - calibration switching replaces only `energy`;
  - the four existing checks pass with unchanged counts. The `ldat_issue_check.py:260` fixture is updated and noted.

  **Verified 2026-09-25:**
  - `src/ldat_inspector.py` has `SideTable`: one row per side, sorted by (SM, file, record), and `partner` rows instead of duplicated partner columns. `ModuleEvents` is now a view: its own columns are zero-copy slices, and partner columns (including the new `partner_sm` / `partner_mm`) are gathered on access and cannot be assigned. `FileResult.modules` comes from `FileResult.table`. Both readers build the table from pair-ordered sides, and the reference now also keeps its coin-flip flag. `merge_results` concatenates per-file tables and drops them from `dataset.files`; `apply_calibration` swaps only the energy column.
  - `python scripts/ldat_scale_check.py --selftest` → **PASS 44/44**, with 12 new T3 checks on both mappings: partner rows pair up, partner SM/mM on every side, partner values from the partner row, a two-file merge keeping pairs and file order, calibration replacing only energy (`table.columns` shared), and **62.0 B/side** with a calibrated energy column. Fast = reference now also covers `partner_sm`, `partner_mm` and `random_slab`.
  - `--real` → **13/13**; fast reader 605,429 pairs/s single core.
  - **Changed spec 001 check:** `ldat_issue_check.py` no longer overwrites module attributes. `_single_module_dataset` builds the same 30,000-side normal(511, 20) SM 0 spectrum as a `SideTable`. The paired partner energies are now that spectrum's other side instead of a constant 511.
  - **Results:** issue 8/8 and `--real` 8/8; engine/report 59/59; revision 14/14; hidden GUI 29/29; unpopulated 6/6 and `--real` 8/8; slab convention 16/16; compile exit 0.
- [x] **T4 — Whole files, all cores, Cancel, memory estimate** (FR-2–FR-4). *Done when:*
  - the estimate appears after file/pairs changes, and a warning dialog appears when a synthetic estimate exceeds the mocked available RAM;
  - a hidden-GUI check processes a multi-file synthetic set in whole-file mode with more than 2 workers;
  - Cancel mid-run returns control within 2 s, keeps the previous dataset and logs the cancel;
  - a subprocess test closes the window mid-run and exits within 5 s;
  - prefix and whole-file labels appear in the log and provenance.

  **Verified 2026-09-25:** `python scripts/ldat_processing_check.py` → **PASS 15/15**.
  - **Engine:** whole-file mode reads all 36 fixture pairs; the fast reader reports per-file progress (1.000) and stops between chunks on Cancel with no partial data. `merge_results(consume=True)` gives the same table while freeing per-file columns: `SideTable.concatenate` now copies each (SM, file) block straight to its final rows, one column at a time. The estimate gives sampled pairs and accepted sides exactly on fixtures, equals baseline + sides × 80 B, and is labelled an upper bound without a config.
  - **Hidden GUI:**
    - the estimate appears after file selection;
    - an estimate above a mocked free-RAM figure asks first, and declining does not process;
    - the whole-file 4-file run used **4 worker processes**, with "(whole files)" and "(whole file)" in the log and "whole files (no pair limit)" in the PDF provenance;
    - **Cancel returned control in 0.03 s** and kept the previous dataset ("previous dataset kept" in the log).
  - **Subprocess:** closing the window mid-run exited in **0.36 s**.
  - **Implementation:**
    - `process_file` now calls `src.ldat_fastread.process_file_fast`;
    - the GUI uses a spawn-context `ProcessPoolExecutor` with `min(files, cpu_count - 2)` workers, a shared cancel `Event` and a per-file progress `Array` (`init_worker`);
    - memory figures live in `src/ldat_memory.py` (`GlobalMemoryStatusEx` / `GetProcessMemoryInfo`, no new dependency);
    - "Whole files" is off by default, since FR-2 says "when the operator requests it".
  - **Changed spec 001 check:** `ldat_inspector_check.py` runs its patched `get_slab_cornell` case on `process_file_reference`, because the patch cannot reach the fast reader; the fast reader's label is compared in the scale check.
  - **All other checks:** engine/report 59/59, revision 14/14, issue 8/8 (`--real` 8/8), hidden GUI 29/29, scale 44/44 (`--real` 13/13), unpopulated 6/6 (`--real` 8/8), slab convention 16/16, slab calibration 15/15; compile exit 0.
- [x] **T5 — Real whole-file acceptance** (FR-1, FR-2). *Done when:* `ldat_scale_check.py --real --whole` processes all six Cornell `coincCompact11s_00000003`–`08` files in under 5 min. It records wall time, accepted/read pairs per file, the pre-run estimate, main-process peak working set (estimate within ±25 %) and worker peaks.

  **Red baseline:** the first `--whole` run met the time target but read the process memory as `None`. `src/ldat_memory.working_set` passed the 64-bit `GetCurrentProcess` pseudo-handle through undeclared ctypes types, so it was truncated. The estimate the owner saw in the GUI (1.9 GB) therefore had no baseline. Fixed by declaring the argument and return types. `ldat_processing_check.py` now requires a real working set, free RAM and a non-zero baseline on Windows (16/16).

  **Verified 2026-09-25:** `python scripts/ldat_scale_check.py --whole` → **PASS 8/8**. Setup: config `cornell_full_system.yaml`, calibration `…coincCompact11s_resolved.encal`, ≥ 4 channels, ≥ 0.2 a.u., whole files.
  - **Counts:** all six whole files match the owner's GUI run exactly: `00000003` 2,471,032 / 1,926,842 read/accepted through `00000008` 2,486,035 / 1,935,729. That is 14,857,750 pairs read, 11,576,118 accepted and 23,152,236 detector sides.
  - **Time:** **11.7 s total** (9.5 s reading with 6 worker processes, 2.2 s merge + keV calibration), against the < 5 min target.
  - **Memory:** estimate **2.00 GB** vs measured main-process peak **2.08 GB (−3.8 %)**, within ±25 %. Baseline 0.14 GB; 83.5 B/side above baseline against the assumed 80. Estimated sides 23,166,235 vs actual 23,152,236. Table 62.0 B/side (1.44 GB).
  - **keV:** available on 23,152,236 / 23,152,236 sides.
  - **Worker peaks (recorded, not capped):** working set 1.93–1.95 GB each, 11.61 GB summed. A single-file probe shows ~1.1 GB private memory added by processing one whole file; the rest is the memory-mapped 760 MB file (reclaimable page cache). About 1.64 GB is committed before reading starts, likely numerical-library thread buffers on 24 cores (not verified).
  - **Worker-memory follow-up (owner-approved, 2026-09-25):**
    - `process_file_fast` copies only each chunk's accepted rows and frees the full-size chunk buffers. The private memory added for one whole file fell from 1.12 to 0.68 GB (177 B/side).
    - The estimate now adds the concurrent workers: the largest files read at once × `WORKER_BYTES_PER_SIDE` = 180 B. The memory-mapped file is not counted, since it is reclaimable page cache. The GUI line and the free-RAM warning use main + workers.
    - Re-measured `--whole` → **PASS 9/9**: 11.6 s total; main estimate 2.00 GB vs peak 2.08 GB (−3.7 %). Worker estimate **4.17 GB vs measured 4.20 GB (−0.8 %)**, 0.70 GB private per worker. Worker working-set peaks 1.58–1.59 GB, down from 1.93–1.95 GB.
    - `ldat_processing_check.py` pins the worker part of the estimate (17/17). Scale 44/44 (`--real` 13/13), engine/report 59/59, hidden GUI 29/29, revision 14/14, issue 8/8; compile exit 0.

## Views

- [x] **T6 — Channel findings engine** (FR-5, FR-7). *Done when:* `scripts/ldat_views_check.py` fixtures with known not-observed, low, high, OK and insufficient channels (plus Cornell inactive minimodules) give exact per-channel states, row priority and threshold changes on both maps.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 16/16**; `--real` → **PASS 17/17**.
  - **Engine** (`src/ldat_inspector.py`): `FindingThresholds` (defaults 0.15 / 3.0 / 20 hits / 100 sides, validated), `channel_findings`, `system_channel_findings`, `FINDING_PRIORITY`, `FINDING_COLOURS`. Counts are the existing per-SM `time_counts` / `energy_counts`: one hit per channel of each accepted side's selected minimodule after the per-channel cut. Spec 001's `channel_status` stays until T7/T10/T12 replace its uses.
  - **Fixtures:** synthetic datasets on the real IMAS 1DAQ (120 SMs) and Cornell full-system (30 SMs) maps, with 8 SM cases on IMAS and 9 on Cornell. Per-channel states and medians match a hand-written rule exactly on both maps. Pinned cases:
    - boundaries 14 → LOW, 15 → OK, 300 → OK, 301 → HIGH, 0 → NOT OBSERVED (median 100);
    - 98 ingest sides, and a time median of 19, read insufficient events with their reasons (FR-7), not not-observed;
    - a time median of 0 (more than half the channels empty) is insufficient, while an energy channel at 0 still makes the row NOT OBSERVED;
    - row priority NOT OBSERVED > HIGH > LOW > INSUFFICIENT > OK, and NO DATA for a mapped SM without sides;
    - threshold changes: low 0.04 clears LOW; high 10 clears HIGH (1000 is not > 1000) and LOW remains; min median 101 and 19, and min sides 98, move the rows as expected;
    - invalid thresholds are rejected (low 1 or −0.1, high 1, median −1, NaN, sides 0 or 2.5);
    - Cornell half-populated SM 2: only its 8 populated minimodules are assessed (64 + 64 channels, median 100). 4 channels in unpopulated minimodules with 5,000 hits are listed as unexpected and do not change the row (OK).
  - **Mutations:** swapping HIGH/LOW priority fails 2 checks; inclusive thresholds (`>=`, `<=`) fail 6.
  - **Real Cornell (owner data, observational):**
    - `--real` prefix of 500,000 pairs of `00000003` (390,203 accepted): 30 rows, no hits in unpopulated minimodules.
    - Whole six files (11,576,118 pairs, recorded, not a pass criterion): rows 12 OK, 9 LOW, 9 NOT OBSERVED; channels 6,366 OK, 19 LOW, 15 NOT OBSERVED, 0 HIGH.
    - The 15 not-observed channels have exactly 0 hits in the whole files: SM 3 time 4368/4419, SM 9 time 135562, SM 12 time 262457/262624/262648 and energy 262422, SM 13 time 262989 and energy 263031, SM 15 energy 528703, SM 20 time 525717, SM 23 time 656768/656791, SM 24 time 659786, SM 28 energy 787417.
    - All 19 low channels are time channels, at 0.03–0.149 × median, except SM 16 time 529182 with only 5 hits (0.0003 × median).
  - **Other checks:** engine/report 59/59, scale 44/44; compile exit 0.
- [ ] **T7 — Channel Status tab** (FR-5, FR-6). *Done when:* the hidden-GUI check shows RAWInspector-style row tags and text states, threshold Apply updates rows, clicking a row draws the channel map and bars for that SM, and "Open in SuperModule" selects it. It also checks the compile.
- [ ] **T8 — Minimodule layout and metrics engine** (FR-8, FR-13). *Done when:*
  - both real maps give 4×4 grids;
  - a synthetic non-4×4 map gives its own grid (proves the layout is derived, not assumed);
  - SM placement is unique;
  - per-minimodule counts are exact;
  - per-minimodule fits match `fit_peak` on the same masks, and raw mode is unavailable.
- [ ] **T9 — System Overview minimodule tiles** (FR-8, FR-9). *Done when:* the hidden-GUI check verifies all four metrics, inactive vs zero vs unavailable rendering, pixel→(SM, mM) resolution on click and "Open SM", and that background fits never block `_poll_events`.
- [ ] **T10 — SuperModule tab stepping and summary** (FR-10, FR-13). *Done when:* the hidden-GUI check covers Prev/Next, wheel and Page Up/Page Down stepping (clamped at the ends), the summary panel and per-minimodule table. The Status tab is retired with no lost content, and stepping redraws only the visible tab.
- [ ] **T11 — Coincidence matrix and Δt** (FR-11, FR-12). *Done when:*
  - fixtures with a known pair list give the exact symmetric SM × SM counts under cuts, counting each pair once;
  - Δt sign, median and central-68 % width match fixtures;
  - the hidden GUI selects a pair from a cell click;
  - labels state the observational limits.

## Cornell slab view and decompression

- [ ] **T14 — Limits loader and derived values** (FR-16–FR-18; after B1). *Done when:*
  - `ldat_views_check.py` loads synthetic limits files and gives exact slab X, clipped decompressed Y and decompressed DOI for known sides;
  - missing keys and out-of-range DOI give NaN with exact per-reason counts;
  - malformed files are rejected;
  - swapping files recomputes without calling the reader;
  - the repo-root `*_limits_full_system.txt` files (6,375 keys each) load.
- [ ] **T15 — Flood/DOI selectors and provenance** (FR-16–FR-18). *Done when:*
  - the hidden-GUI check covers both file pickers, the flood-view and DOI-unit selectors, IMAS/no-file unavailable states, the DOI cut reset (0–15 ratio vs 0–20 mm), and the excluded counts in titles;
  - PDF provenance lists the limits paths and exclusion counts;
  - on a real Cornell prefix with the owner-chosen limits files, the slab flood map shows per-slab stripes for owner review.

- [ ] **T16 — Non-adjacent slab recovery option** (FR-19; after B1, with T2). *Done when:*
  - `ldat_scale_check.py` fixtures pin every recovery branch (only p−1 fired, only p+1 fired, both with a stronger side, a tie, neither fired, and edges) in both the reference and the fast reader;
  - legacy mode stays the default and its counts are unchanged;
  - on a real Cornell prefix, the recovered-side count is recorded against the 3.9 % non-adjacent baseline;
  - provenance and the GUI label show the active rule.

- [ ] **T17 — Estimated calibration factors in the inspector** (FR-20; with T12). *Done when:*
  - `ldat_views_check.py` loads a synthetic `.encal` with a `_status.txt` sidecar and counts sides using fitted, borrowed and estimated factors per SM exactly;
  - estimated/borrowed slabs are excluded from per-slab fit and uniformity inputs when so selected;
  - an `.encal` without a sidecar reports provenance "unknown" rather than "fitted";
  - the energy view, Channel Status and PDF provenance show the counts;
  - on the real `…coincCompact11s_resolved.encal`, counts match its status file.

## Reports and validation

- [ ] **T12 — Reports and provenance** (FR-14). *Done when:* `pypdf` extraction of synthetic IMAS/Cornell PDFs finds whole-file/prefix scope, channel-findings thresholds and table, and per-minimodule tables matching the engine values, with raw-mode fits marked unavailable.
- [ ] **T13 — Integrated validation and owner review** (FR-1–FR-20; after T14–T17). *Done when:* all new and existing checks pass, compile exits 0 and the T5 real run is recorded. Owner GUI review is recorded on the Cornell set, and real IMAS is either performed or explicitly waived by the owner. Only then set the spec to `shipped`.
