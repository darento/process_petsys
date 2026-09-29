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
- [x] **T7 — Channel Status tab** (FR-5, FR-6). *Done when:* the hidden-GUI check shows RAWInspector-style row tags and text states, threshold Apply updates rows, clicking a row draws the channel map and bars for that SM, and "Open in SuperModule" selects it. It also checks the compile.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 29/29**; `--real` → **PASS 30/30**; GUI compile exit 0.
  - **Engine:** `channel_geometry(dataset, sm)` places each channel from the selected map. `Dataset` now keeps the map's `coordinates`, `channel_modules` and `channel_types`. On both maps every expected channel sits at its map X (time) or Y (energy) inside its minimodule box. Each SM has 16 minimodule boxes, with populated ones equal to `expected_mm` (Cornell SM 2: 8).
  - **Hidden GUI** (Cornell fixtures from T6):
    - 30 rows with the RAWInspector tag colour and the text state for every fixture case;
    - row columns pinned (the mixed SM reads `400`, `1/16`, `1 / 0 / 0`, median `100`, `0 / 1 / 1`; half-populated SM 2 reads `1/8` and `4 ch` unexpected; the 98-side SM reads `insufficient`);
    - the summary shows totals, the population wording and the thresholds;
    - Apply with low 0.04 turns the LOW row OK; high 0.9 is rejected with a dialog and the rows are kept;
    - clicking a row draws 128 vertical time segments at the map's fine X and 128 horizontal energy segments, one flagged outline, 128 bars per type with state colours, and median / 0.15× / 3× lines. The tab does not change;
    - the flag list names each flagged channel; the half-populated SM shows 8 hatched minimodules and 4 unexpected channels;
    - "Open in SuperModule" selects the SM in the SuperModule Explorer (renamed "SuperModule" in T10); invalidating inputs clears the tab; no error dialogs.
  - **Found while checking:** in a screenshot on a real Cornell prefix, the SM table area was blank. The Treeview was created before the frame it is packed into, so the frame covered it. The spec 001 table used the same construction. The frame is now created first; the screenshot shows the rows.
  - **Changed spec 001 check:** `ldat_revision_check.py` read the old columns (`Selected sides`) and `channel_ax`. It now checks SM, ingest sides, minimodules and the finding (14 revision checks pass).
  - **Other checks:** hidden GUI 29/29, issue 8/8, processing 17/17, engine/report 59/59, scale 44/44, unpopulated 6/6.
  - **Owner review pending:** layout and readability on representative IMAS and Cornell acquisitions.
- [x] **T8 — Minimodule layout and metrics engine** (FR-8, FR-13). *Done when:*
  - both real maps give 4×4 grids;
  - a synthetic non-4×4 map gives its own grid (proves the layout is derived, not assumed);
  - SM placement is unique;
  - per-minimodule counts are exact;
  - per-minimodule fits match `fit_peak` on the same masks, and raw mode is unavailable.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 43/43** (13 new for T8); `--real` → **PASS 45/45**.
  - **Engine** (`src/ldat_inspector.py`): `minimodule_layout`, `supermodule_layout` and `minimodule_metrics`, plus `RAW_FIT`.
  - **Layout:**
    - both real maps give a 4 × 4 grid with unique cells for every SM; populated cells equal `expected_mm` (Cornell half-populated SMs: 8);
    - orientation: SM 0 reads, top to bottom, `[3 2 1 0] [7 6 5 4] [11 10 9 8] [15 14 13 12]` on both maps;
    - synthetic 2 × 3, 3 × 1 and 1 × 5 maps give their own shapes and cells;
    - two minimodules at one centre raise an error.
  - **SuperModule placement:** unique and equal to spec 001's GUI `_layout` for every SM: IMAS 5 × 24 (120 SMs), Cornell 3 × 10 (30 SMs).
  - **Metrics:** synthetic Cornell sides in SMs 0–2 (pairs inside one minimodule, 75 % peak + flat background) are compared with a per-side Python recount:
    - ingest, selected (paired energy + DOI + ROI) and fit-population counts are exact for all 400 expected minimodules, including 0 for SMs and minimodules without sides;
    - fits equal `fit_peak` on the same ROI/DOI mask with the energy window off (status, μ and resolution; 38 FIT, and low-statistics minimodules unavailable);
    - raw mode gives `RAW_FIT` for every minimodule and keeps the counts;
    - a cancel callback returns None; `fits=False` gives counts only.
  - **Real Cornell** (recorded; not a pass criterion): six whole files with `…_resolved.encal` and GUI default cuts (400–650 keV, DOI 0–15):
    - all **400/400** minimodules FIT; centroid median 509.4 keV (5–95 %: 504.5–511.6); resolution median 15.5 % (14.1–17.8);
    - 40,450–78,673 ingest sides per minimodule;
    - **timing: counts only 1.6 s, counts + fits 6.7 s (17 ms per minimodule).** Both run off the Tk thread in T9. The `--real` prefix check covers every side and gives every expected minimodule a grid cell.
  - **Other checks:** engine/report 59/59, revision 14/14, hidden GUI 29/29; compile exit 0.
- [x] **T9 — System Overview minimodule tiles** (FR-8, FR-9). *Done when:* the hidden-GUI check verifies all four metrics, inactive vs zero vs unavailable rendering, pixel→(SM, mM) resolution on click and "Open SM", and that background fits never block `_poll_events`.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 59/59** (16 new for T9); `--real` → **PASS 61/61**; GUI compile exit 0.
  - **Engine (`overview_grid`), on the T8 synthetic Cornell sides:**
    - the composite is 14 × 49 pixels: 3 rings × 10 cassettes of 4 × 4 tiles, one-pixel gaps between SMs. IMAS is 24 × 119;
    - SM 4 mM 0 is at pixel (5, 8); count tiles equal the metrics;
    - zero-side minimodules are values (0), and the 80 unpopulated tiles are a separate kind;
    - fit tiles equal the fits; failed fits are unavailable with their reason, never a low value;
    - raw mode makes all 400 fit tiles unavailable (raw a.u.) and keeps the count tiles.
  - **Hidden GUI:**
    - the tab shows "Computing per-minimodule counts…" and then the tiles from the background job;
    - all four metrics draw the expected per-minimodule values and label. The comparison uses the app's selection: loading applies the config's `energy_range`;
    - unpopulated tiles are hatched light grey and unavailable fits dark grey, both outside the colormap and in the legend; zero sides are the colormap minimum;
    - real canvas clicks resolve (row 5, col 8) → "SM 4 · mM 0" and (1, 2) → "SM 0 · mM 5", and a gap click is ignored; the tile text gives ingest, selected and the photopeak;
    - "Open SM" selects SM 0 in the SuperModule tab; invalidating inputs clears the cache and pick.
  - **Non-blocking:** a slowed metrics job (2 s, polling cancel) runs on a worker thread. `_poll_events` kept running (longest gap 0.16 s). A second request superseded the first job, which stopped (1 cancel), and only the new selection was cached.
  - **Real Cornell:**
    - one whole file (`00000003`, 3,853,684 sides, resolved calibration): counts + fits for 400 minimodules in 5.3 s on the worker thread; `_poll_events` median gap 0.094 s, longest 0.172 s;
    - screenshots of a 500,000-pair prefix show the centroid (all 400 fitted, 500–518 keV) and ingest views. The half-populated SMs (ring 2) are hatched on the left two columns, matching the config.
  - **Changed spec 001 check:** `ldat_gui_check.py` used the removed "Counts" mode; it now uses "Ingest sides" (29/29). "Flood maps" is unchanged.
  - **Other checks:** engine/report 59/59, revision 14/14, issue 8/8, processing 17/17, scale 44/44, unpopulated 6/6.
  - **Owner review pending:** readability on IMAS (120 SMs) and Cornell.
- [x] **T10 — SuperModule tab stepping and summary** (FR-10, FR-13). *Done when:* the hidden-GUI check covers Prev/Next, wheel and Page Up/Page Down stepping (clamped at the ends), the summary panel and per-minimodule table. The Status tab is retired with no lost content, and stepping redraws only the visible tab.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 76/76** (17 new for T10); `--real` → **PASS 78/78**; GUI compile exit 0.
  - **Engine:** `minimodule_metrics(..., sms=[1, 2, 99])` gives exactly the rows of SMs 1 and 2 (99 is not mapped), equal to the full computation.
  - **Hidden GUI, on the T6 Cornell fixtures** (drawing is counted per tab):
    - the tabs are Channel Status, SuperModule, System Overview, Timestamps; the Status tab and its text box are gone;
    - loading shows SM 0 at "1 / 30" with Prev disabled. Prev/Next step one SM and redraw at once; both stop at the ends (the end button is disabled and a step returns False);
    - wheel events on the SM box and on the toolbar background: wheel up at SM 0 is clamped, and three wheel-downs move the box to SM 3 at once with no redraw. One redraw follows 150 ms later; wheel up then goes back one;
    - Page Down twice from the second-last SM gives the last SM with one redraw; Page Up steps back. On the Channel Status tab, Page Down does nothing;
    - three steps redraw only the SuperModule tab (3 draws). A cut change redraws only the visible tab, and Timestamps and System Overview draw once when shown. After a further step, showing them again does not redraw them;
    - the summary keeps the spec 001 Status-tab content for all 9 fixture SMs: ingest sides, time and energy observed / expected, every unobserved channel ID from `channel_status`, and the minimodule occupancy (in the table). The header gives the SM, the system and the finding in its Channel Status colour;
    - the summary lists flagged channels by state (not observed / low / high with IDs), the 4 stray channels of the half-populated SM, and "insufficient: 98 ingest sides < 100" with the zero-hit IDs. Threshold Apply updates the visible summary (LOW → OK → LOW).
  - **Hidden GUI, per-minimodule table** (T8 synthetic sides, calibrated; energy window from the config):
    - SM 0 shows "computing…" first, then per-mM ingest, selected, centroid, resolution and FIT from one background job for SM 0 (not on the Tk thread). "16/16 fitted"; the selected total equals the summary line and the explorer label;
    - SM 1 mM 3 (24 sides) and mM 7 (0 sides) read "insufficient events" with no value;
    - with a 0.4 s job, stepping SM 0 → 1 → 2 during the job runs jobs for SM 0 and SM 2 only. The half-populated SM 2 has 8 grey rows reading "unpopulated (config)";
    - a revisited SM and a System Overview result with fits are reused without a new job;
    - raw mode: every fit reads `unavailable: raw a.u. (no keV calibration)` with the counts kept. Invalidating inputs clears the summary, table and stepping controls; no error dialogs.
  - **Test note:** Tk drops generated wheel and key events for an unmapped window, so the check maps the window fully transparent for those two checks.
  - **Mutation checks:** without the wheel debounce, with Page Up/Down active on every tab, or with a step marking every tab stale, one check fails each; unmodified, none fail.
  - **Real Cornell** (`00000003`, resolved calibration, 1600 × 960 window; recorded, not a pass criterion):
    - stepping through all 30 SMs: a step takes median 0.45 s (max 0.60 s) on the 500,000-pair prefix and 0.50 s (max 0.61 s) on the whole file (3,853,684 sides), including the canvas render;
    - the summary panel takes 8 ms of that. The rest is the spec 001 energy/DOI/flood plots (0.27 s to build, 0.33 s to render); previously an SM change also redrew System Overview and Timestamps;
    - the per-mM table job (16 fits) finishes within the same step (max 0.10 s after it);
    - screenshots: SM 12 shows the NOT OBSERVED header and channels 262457, 262624, 262648 and 262422, which are among the T6 whole-file zero-hit channels; SM 2 shows 8/8 fitted and 8 unpopulated rows.
  - **Found while checking:** at 960 px the summary text left room for only two table rows. The text and table now share a draggable vertical split (8 rows visible by default), and the Fit column was widened so "unpopulated (config)" fits.
  - **Changed spec 001 checks:** `ldat_gui_check.py` expected 5 tabs; it now expects 4 and shows the SuperModule tab first, because only the visible tab redraws (29/29). `ldat_revision_check.py` also shows that tab first (14/14). The T7 and T9 "Open …" checks now also confirm that the SuperModule tab is drawn for that SM.
  - **Other checks:** engine/report 59/59, processing 17/17, issue 8/8, scale 44/44, unpopulated 6/6.
  - **Owner review pending:** stepping feel, the summary layout and the split position on the owner's screen.
- [x] **T11 — Coincidence matrix and Δt** (FR-11, FR-12). *Done when:*
  - fixtures with a known pair list give the exact symmetric SM × SM counts under cuts, counting each pair once;
  - Δt sign, median and central-68 % width match fixtures;
  - the hidden GUI selects a pair from a cell click;
  - labels state the observational limits.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 92/92** (16 new for T11); `--real` → **PASS 94/94**; GUI compile exit 0. FR-11's partner column was already in the table (T1/T2); T11 uses it.
  - **Engine, on a hand-built Cornell pair list** (153 pairs, timestamps near 2.9 × 10¹⁴ ps):
    - SM 0–1: 101 passing pairs, half stored with the SM 1 side first. SM 0–2: 20 passing, plus 5 failing the partner energy, 5 failing DOI on the SM 2 side only and 5 failing ROI X on the SM 0 side only. SM 3–3: 10 passing. SM 4–5: 7 passing;
    - the matrix equals a pair-by-pair recount: symmetric, 138 of 153 pairs, 0–1 = 101, 0–2 = 20, 3–3 = 10 (diagonal counted once). Without cuts all 153 pairs are counted once, and 0–2 = 35;
    - Δt(0, 1) equals the fixture values exactly, with median +3.0 ns; Δt(1, 0) is the negation (−3.0 ns). p16 −14.0, p84 20.0 and width 34.0 ns equal `np.percentile` of the fixture;
    - SM 0 against all partners gives 121 pairs; SM 3–3 gives 10 (each pair once, also against all partners); an SM without pairs gives no statistics (None);
    - a precomputed mask gives the same matrix and Δt.
  - **Hidden GUI:**
    - the tab shows "Computing the SM × SM pair matrix…" and then the matrix from a background job (not the Tk thread), equal to `pair_matrix`, titled "both sides pass the display cuts", "each counted once";
    - the default Δt is SM 0 against all partners (121 pairs, row highlighted);
    - a canvas click on (SM 0, SM 1) selects the pair and gives "101 pairs • median +3.00 ns • central 68 % width 34.00 ns", with 101 pairs in the histogram and both cells highlighted. The mirrored cell gives −3.00 ns; the diagonal cell (SM 3) gives 10 pairs, titled "sign arbitrary";
    - the tab label and the Δt title state that geometry and time of flight contribute and that Δt is not a clock or CTR calibration; an SM without passing pairs reads "no pairs pass the cuts";
    - a cut change (DOI high 0.9 → 1.0) recomputes the matrix (0–2 becomes 25), and going back redraws at once from the cache;
    - stepping SMs does not make the tab stale; invalidating inputs clears it; no error dialogs.
  - **Mutation checks:** requiring only the own side to pass fails 3 checks, counting same-SM pairs per side fails 2 and flipping the Δt sign fails 2; unmodified, none fail.
  - **Real Cornell** (`00000003` whole file, 3,853,684 sides, resolved calibration, default cuts; recorded, not a pass criterion):
    - 389,780 of 1,926,842 pairs pass the cuts. The mask and matrix take 0.10 s (engine), and the tab is drawn 0.65 s after loading. 315 SM pairs have coincidences; the diagonal and each SM's nearby SMs have none;
    - the largest cell is SM 10–SM 25 with 15,446 pairs: median +3.93 ns, central-68 % width 1.58 ns, and −3.93 ns reversed. It computes in 2 ms, and the GUI redraw takes 0.14 s;
    - the histogram shows two peaks, near +3.2 and +4.5 ns (two axially separated sources: time of flight). 175 pairs fall outside the plotted range (all pairs span −100.3 to +69.0 ns).
  - **Found while checking:** at first the histogram spanned the full data range, so on real data the peak took one or two of 100 bins. It now spans the core ± 4 central-68 % widths and states how many pairs are outside.
  - **Changed spec 001 check:** `ldat_gui_check.py` now expects 5 tabs again (29/29).
  - **Other checks:** engine/report 59/59, revision 14/14, processing 17/17, issue 8/8, scale 44/44, unpopulated 6/6.
  - **Owner review pending:** matrix and Δt readability on Cornell and IMAS (120 × 120).

## Cornell slab view and decompression

- [x] **T14 — Limits loader and derived values** (FR-16–FR-18; after B1). *Done when:*
  - `ldat_views_check.py` loads synthetic limits files and gives exact slab X, clipped decompressed Y and decompressed DOI for known sides;
  - missing keys and out-of-range DOI give NaN with exact per-reason counts;
  - malformed files are rejected;
  - swapping files recomputes without calling the reader;
  - the repo-root `*_limits_full_system.txt` files (6,375 keys each) load.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 101/101** (9 new for T14); `--real` → **PASS 104/104**; GUI compile exit 0. Engine: `load_limits`, `Limits`, `slab_view`, `decompressed_doi`, `unresolved_slab_pairs` in `src/ldat_inspector.py`; no GUI change (T15).
  - **Slab X:** equals `src/utils.py:get_slab_cornell`'s X on hand-built hits for all 8 positions of a Cornell minimodule, alone and with each neighbour (22 cases: 4 edge, 6 seeded coin flips, 12 adjacent). It also equals the written-out rule, fine X − 0.8 mm for even slabs and + 0.8 mm for odd slabs (B1).
  - **Decompressed Y and DOI** on 10 known sides in rows 1 and 3 (to 1e-9 mm):
    - Y 57.6, 51.2 and 76.8 (clipped low and high; 2 clipped), 64.0, 76.8 (exactly at the right limit, not clipped), 6.4 and 25.6 (row 3, no offset);
    - DOI 10.0, 0.0 and 20.0 (both bounds kept); −0.4 and 22.4 mm are out of range.
  - **Exclusions:** Y {missing key 2, invalid limits 1} and DOI {missing key 2, invalid limits 1, out of range 2}. The NaN count equals the excluded total, so nothing is substituted. "Invalid limits" (left == right) is a reason not in the task text: the list-mode DOI copy has one such entry.
  - **Malformed files:** 13 are rejected whole: header line, missing column, not a number, NaN, inf, slab 16, negative channel, spaces instead of tabs, duplicate key, a bad line after good ones, empty, blank lines only, and an absent file. CRLF and blank lines load.
  - **Swap:** with `iter_pairs`, `process_file`, `process_file_reference` and `process_file_fast` patched to fail, a second COG file moves row 0 from 57.600 to 54.400 mm and leaves the other rows unchanged. Going back to the first file reproduces the first values; the reader is called 0 times and the stored columns are unchanged.
  - **IMAS:** a dataset marked IMAS is refused by both functions. `unresolved_slab_pairs` reads the ingest counter (9 in the fixture).
  - **Owner files:** every key belongs to a Cornell time channel and is one of its own slabs (slab // 2 = position).
    - the repo-root `cog_`/`doi_limits_full_system.txt` have **6,376 keys each**, not the 6,375 in the task text, with no left == right;
    - the list-mode copies in `files_for_listmode\` have 6,375 keys each, and their DOI copy has 1 left == right entry.
  - **Mutation checks:** each mutation fails 1–3 checks:
    - slab X sign swapped;
    - Y not clipped;
    - row from `mm % 4`;
    - missing key given fallback limits;
    - DOI 0 mm excluded;
    - DOI left/right swapped;
    - duplicates accepted;
    - slab 16 accepted;
    - left == right not excluded.

    Unmodified, none fail.
  - **Real Cornell prefix** (`00000003`, 500,000 pairs per file, 780,406 sides, repo-root files; recorded, not a pass criterion):
    - loading both files takes 0.03 s; all SMs take 0.10 s;
    - Y: 1,588 missing keys (0.20 %), 37,234 clipped (4.8 %);
    - DOI: 33,429 out of range (4.3 %), 1,588 missing keys;
    - 33,790 unresolved-slab pairs were rejected at ingest;
    - slab X − COG X: median +0.001 mm, |Δ| p95 0.98 mm.
  - **Other checks:** engine/report 59/59, hidden GUI 29/29, revision 14/14, processing 17/17, issue 8/8, scale 44/44 (selftest 44/44), unpopulated 6/6, slab convention 16/16.
- [x] **T15 — Flood/DOI selectors and provenance** (FR-16–FR-18). *Done when:*
  - the hidden-GUI check covers both file pickers, the flood-view and DOI-unit selectors, IMAS/no-file unavailable states, the DOI cut reset (0–15 ratio vs 0–20 mm), and the excluded counts in titles;
  - PDF provenance lists the limits paths and exclusion counts;
  - on a real Cornell prefix with the owner-chosen limits files, the slab flood map shows per-slab stripes for owner review.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 124/124** (23 new for T15); `--real` → **PASS 127/127**; GUI compile exit 0.
  - **Code:** engine `apply_doi_view`, `SideTable.with_doi`, `ModuleEvents.doi_ratio`, `Selection.mask(doi=False)`, `slab_totals`, `slab_x_edges`, and `flood_counts(extent=, x_edges=)`; report `write_report(cog_limits=, doi_limits=, slab_flood=)`; GUI pickers, selectors and views (see plan). Owner-chosen files: the repo-root `cog_`/`doi_limits_full_system.txt`.
  - **Hidden GUI** (Cornell SM 0 fixture: 652 sides on minimodules 4 and 13, 32 keys; 5 sides on a key in neither file, 3 above their COG right limit, 4 with DOI ratio 8; 7 unresolved-slab pairs in the ingest counters):
    - **unavailable states:** IMAS reads "n/a: Cornell only" on both selectors; Cornell without files reads "n/a: no COG / DOI limits file"; the selectors are disabled, and COG / RTP and Ratio stay the defaults;
    - **COG picker:** the file loads (short name shown) and enables both flood selectors (SuperModule tab and System Overview);
    - **slab flood:** equals the engine: 647 of 652 sides, 0–102.4 mm, the note "excluded: missing key 5 • clipped to row 3 • 7 unresolved-slab pairs"; before switching, the COG flood had all 652;
    - **ROI tools:** the slab view has no ROI rectangle or selector, and a note says the ROI uses COG/RTP coordinates;
    - **Overview flood mode** follows the selector: the suptitle is "Slab flood maps (decompressed Y, cog_fixture.txt) • 647 sides passing the cuts • excluded: missing key 5…";
    - **COG file changes:**
      - a malformed file gives one dialog and the previous file is kept;
      - a swapped file redraws exactly `flood_counts` of the new limits;
      - removing the file (label click, confirmed) returns to COG / RTP and brings the ROI rectangle and selector back;
      - the reader, patched to fail, is called 0 times throughout;
    - **DOI picker** enables the unit selector, and Ratio stays the view;
    - **mm:**
      - the cut resets to 0–20; the view comes from a background thread and equals `decompressed_doi`, with `doi_excluded` {missing key 5, out of range 4};
      - the DOI plot spans 0–20 mm, with an "mm-equivalent" label and a note naming the file, "not a validated depth" and "9 excluded: missing key 5, out of range 4";
      - 643 sides are selected (the excluded ones fail the cut); a 0–10 mm cut keeps exactly the sides with decompressed DOI in [0, 10] (324);
    - **back to Ratio:** the cut returns to 0–15 and the view is the stored ratio column. Removing the DOI file in mm mode falls back to Ratio (0–15);
    - **GUI SuperModule report:** carries the slab and mm views and both file names;
    - **IMAS:** an IMAS dataset with both files loaded makes both selectors unavailable;
    - no unexpected dialogs.
  - **PDF** (`pypdf`, SM 0, mm view and slab flood):
    - the provenance holds both full paths, "Flood view: slab-assigned … excluded: none; clipped to the row: 3" (in mm the missing-key sides already fail the DOI cut), "unresolved-slab pairs rejected at ingest (all files): 7", "DOI view: decompressed mm-equivalent … missing key 5, out of range 4", "DOI mm 0..20" and "not a validated depth";
    - the SM page titles "Selected slab flood (decompressed Y)" and labels "Decompressed DOI (mm-equivalent)";
    - a default-view PDF reads "Flood view: COG/RTP centroid", "DOI limits: none" and "DOI view: light-sharing ratio";
    - a slab-flood report without a COG file is refused.
  - **Slab columns:** 64 per full Cornell SM, with each of the 16 fixture slabs in its own column.
  - **Mutation checks** (GUI + PDF checks):
    - DOI cut not reset: 4 fail;
    - slab view never active: 5;
    - ratio kept instead of mm: 5;
    - regular X bins in the slab view: 1;
    - no limits lines in the PDF: 3;
    - selectors never disabled: 7.

    Unmodified, none fail.
  - **Found while checking:**
    - two-line titles overlapped on the real canvas, so the counts moved to in-axes notes;
    - regular X bins aliased the discrete 1.6 mm slab X (moiré), so the slab view now has one column per slab.
  - **Real Cornell** (`00000003`, resolved encal, min 4 channels, 0.2 a.u.; mm and slab chosen before processing, so the DOI view is applied in the processing thread; recorded, not a pass criterion; screenshots in the session scratchpad `t15/`):
    - **prefix** (500,000 pairs/file, 780,406 sides):
      - processed in 3.2 s; mm DOI excludes {missing key 1,588, out of range 33,429};
      - slab totals take 0.08 s under the cuts (151,882 shown, 0 excluded, 3,859 clipped);
      - an SM redraw takes 0.52 s; the overview slab floods 0.48 s;
    - **whole file** (3,853,684 sides):
      - processed in 7.1 s; mm excludes {missing key 7,747, out of range 164,681};
      - under the cuts 749,022 sides are shown, 0 excluded and 18,341 clipped; 167,568 unresolved-slab pairs were rejected at ingest;
      - the SM 13 redraw takes 0.59 s; the overview 1.0 s;
      - SM 13 shows 64 distinct slab X columns at 1.600 mm (median) spacing: per-slab stripes, with empty columns and blocks where the Channel Status reads NOT OBSERVED.
  - **Other checks:** hidden GUI 29/29, revision 14/14, processing 17/17, issue 8/8, unpopulated 6/6, slab convention 16/16, scale 44/44 (selftest 44/44), engine/report 59/59.
  - **Owner review pending:** the slab flood and mm DOI on the real Cornell set (screenshots `t15/sm13_slab_mm.png`, `t15/sm13_figure.png`, `t15/overview_slab.png`). Also pending: the design choices of a display-only COG file, the ROI kept on COG/RTP coordinates, exclusions counted among sides passing the cuts, and one column per slab.

- [x] **T16 — Non-adjacent slab recovery option** (FR-19; after B1, with T2). *Done when:*
  - `ldat_scale_check.py` fixtures pin every recovery branch (only p−1 fired, only p+1 fired, both with a stronger side, a tie, neither fired, and edges) in both the reference and the fast reader;
  - legacy mode stays the default and its counts are unchanged;
  - on a real Cornell prefix, the recovered-side count is recorded against the 3.9 % non-adjacent baseline;
  - provenance and the GUI label show the active rule.

  **Verified 2026-09-25:** `python scripts/ldat_scale_check.py --selftest` → **PASS 65/65** (21 new for T16); `--real` → **PASS 15/15**; `--whole` → **PASS 9/9**; `python scripts/ldat_views_check.py` → **PASS 152/152** (9 new); GUI compile exit 0. Engine `Settings.slab_rule`, `SLAB_RULES`, `cornell_slab`, the `recovered_slab` column, `recovered_slab_sides` and `slab_rule_text`; fast-kernel branch; GUI selector and labels; report provenance (see plan).
  - **Fixtures** (Cornell, a separate 88-pair file; det1 on SM 0, the strongest time channel p = 3 unless stated, a non-adjacent second at 6):
    - only p−1 fired → slab 6; only p+1 → 7; both with p−1 stronger → 6; both with p+1 stronger → 7;
    - exact tie and no neighbour (24 pairs each) → both 6 and 7 appear, `random_slab` set;
    - a neighbour tying the non-adjacent second: listed after it → recovered 6; listed before it → legacy-adjacent 6 under both rules (the stable sort order);
    - p = 1 with neighbour 0 → 2; p = 6 with neighbour 7 → 13;
    - edges p = 0 / 7 with a non-adjacent second → edge rule 1 / 14 under both rules, not recovered.

    Legacy keeps exactly the 12 pairs with a legacy slab and rejects 76 as unresolved. Recover accepts all 88, with `recovered_slab` on exactly the 76, and det2 unchanged. The fast reader equals the reference under both rules, and on the mixed file under recover. There, the 6 non-adjacent pairs are accepted and every other outcome is unchanged. IMAS ingest is identical under either rule.
  - **Legacy unchanged:** the mixed-fixture outcomes, spec 001 T9's 80,000-pair tables for all six files (both readers) and the six whole-file counts match the owner's GUI run. The default is "legacy".
  - **Mutation checks** (scratch `t16_mutate.py`), each caught:
    - kernel: tie → 2p; sides swapped; no recovered flag; recover ignored; neighbour window ±2; random flag lost;
    - reference: tie → 2p; stronger side reversed; legacy recovering.

    Unmodified, none fail.
  - **Real Cornell `00000003`, 80,000-pair prefix** (both readers equal under recover):
    - 5,457 unresolved pairs → 5,444 accepted (62,460 → 67,904) and 13 `ValueError` on the other side;
    - 5,590 recovered sides = 4.12 % of 135,808 sides, against the 3.9 % non-adjacent baseline: 4,398 (78.7 %) from a fired neighbour, 1,192 coin flips.
  - **Real whole file `00000003`** (recorded, not a pass criterion):
    - legacy 1,926,842 pairs (unchanged); recover 2,094,087 pairs (+8.7 %) and 4,188,174 sides, with 171,758 recovered (4.1 %): 133,848 from a neighbour (77.9 %), 37,910 coin flips. `ValueError` rises 11,366 → 11,689, and no unresolved pairs remain;
    - read in 4.0 s (legacy 3.8 s); GUI processing with keV 7.6 s; the six-file table is 64.0 B/side;
    - on the legacy-built resolved encal, the whole-file photopeak is μ 509.3 keV / 15.6 % for other sides, 519.5 keV / 17.2 % for neighbour-recovered sides and 509.4 keV / 18.9 % for coin-flipped ones;
    - screenshots are in the session scratchpad `t16/`.
  - **Hidden GUI and PDF:**
    - the selector defaults to Legacy (reject), is disabled for IMAS, and the Process settings carry the chosen rule (IMAS always legacy);
    - the SM summary reads "Slab rule recover non-adjacent / Recovered sides 76 (86.4 % of SM; legacy-built keV/limits)";
    - the log gives the totals and the legacy-calibration warning, and the status keeps "Merged …";
    - the plot labels read "recover rule: 76 recovered sides" / "legacy rule: 76 unresolved pairs rejected" (also in the T15 slab-flood checks);
    - a change after loading says it applies on the next Process;
    - the SM 0 PDF provenance reads "Slab rule: recover non-adjacent (FR-19); 76 recovered sides (86.4 % of sides) … (SM 0)"; the whole-system lines hold for both rules, and IMAS has none.
  - **Found while checking:**
    - on the real canvas the selector text was cut and the slab-flood note ran into the colorbar, so the label became "Slab rule" and the notes were shortened;
    - "Merged …" stopped being the status line, so the rule is now logged first.
  - **Other checks:** hidden GUI 29/29, revision 14/14, processing 17/17, issue 8/8, unpopulated 6/6, slab convention 16/16, engine/report 59/59.
  - **Owner review 2026-09-28:** the owner checked the real-file screenshots (`t16/`). The whole-SM energy resolution barely changes under the recover rule, because recovered sides are about 4 % of the total. The rule recovers extra sides (+8.7 % accepted pairs on `00000003`), which the owner expects to raise sensitivity; sensitivity itself was not measured.
    - Not separately decided, so the current behaviour is kept:
      - recovered sides use the legacy-built calibration;
      - ties and sides without a fired neighbour are coin-flipped;
      - a legacy calibration gets a warning rather than a refusal.

- [x] **T17 — Estimated calibration factors in the inspector** (FR-20; with T12). *Done when:*
  - `ldat_views_check.py` loads a synthetic `.encal` with a `_status.txt` sidecar and counts sides using fitted, borrowed and estimated factors per SM exactly;
  - estimated/borrowed slabs are excluded from per-slab fit and uniformity inputs when so selected;
  - an `.encal` without a sidecar reports provenance "unknown" rather than "fitted";
  - the energy view, Channel Status and PDF provenance show the counts;
  - on the real `…coincCompact11s_resolved.encal`, counts match its status file.

  **Verified 2026-09-25:** `python scripts/ldat_views_check.py` → **PASS 143/143** (19 new for T17); `--real` → **PASS 146/146**; GUI compile exit 0. Engine `load_calibration_status`, `CalibrationStatus`, `FACTOR_ORIGINS`, `factor_origins`, `slab_origins`, `Selection.fitted_only` and `SideTable.origin`; GUI switches, note, column and summary; report provenance (see plan).
  - **Engine**, on a synthetic `.encal` plus status for every mapped Cornell slab:
    - SM 0 sides: fitted 60, check 8, borrowed 12, est. neighbours 10, est. median 6, no fit 4 and unknown 4 (a key in the `.encal` but not in the status). SM 1: fitted 10, borrowed 6. The counts are exact per SM and in total;
    - slabs: SM 0 has 256 (250 fitted, one of each other origin); the half-populated SM 2 has 128 unpopulated (not "no fit") and 128 fitted; over the map 7,680, with 1,280 unpopulated;
    - fitted only keeps exactly the fitted and check sides (68 of 100 in the window), and the per-minimodule selected count follows it;
    - origins survive the DOI view, are absent in raw a.u. and are restored when keV returns;
    - without a sidecar every side and slab is "unknown" and fitted only keeps 0; IMAS reports nothing;
    - 5 malformed sidecars reject the calibration (unrecognised status, missing tab, slab 16, duplicate, header only).
  - **Hidden GUI:**
    - Channel Status reads "12 / 16" (SM 0) and "6 / 0" (SM 1), and its summary gives the ingest sides and mapped slabs by origin (with unpopulated 1,280);
    - the SM summary lists the origin counts of the energy plot's sides (none drawn on the plot), and the checkbox is enabled;
    - "Fitted keV factors only" gives 68 selected sides and says so on the plot;
    - Photopeak Uniformity defaults to fitted only: SM 0 has 68 of 104 sides, equal to the engine;
    - without a sidecar: "unknown" in the column, the note and the summary, and the switch is disabled with a "no _status.txt" note;
    - raw a.u.: "—", no note, the switch disabled;
    - no error dialogs.
  - **PDF:**
    - the provenance has the sidecar path, "Ingest sides by keV factor origin (SM 0): fitted 60, fitted (check) 8, borrowed 12, estimated (neighbours) 10, estimated (median) 6, no fit 4, unknown 4", the mapped slabs (fitted 250…) and "fitted keV factors only (borrowed and estimated left out)"; the SM page lists its origins;
    - without a sidecar: "keV factor origins: unknown … never assumed fitted".
  - **Real resolved calibration (sidecar):**
    - its counts equal the status file: fitted 5,356, check 114, borrowed 791, est. neighbours 80, est. median 59 and no fit 1,280;
    - mapped over the Cornell map, the 1,280 "no fit" slabs are exactly the unpopulated ones (no fit 0).
  - **Mutation checks:**
    - borrowed read as fitted: 10 fail;
    - fitted only dropping "check": 3;
    - no sidecar assumed fitted: 1;
    - unpopulated counted as no fit: 3;
    - unknown status text accepted: 1;
    - GUI never offering origins: 3;
    - no origin lines in the PDF: 2.

    Unmodified, none fail.
  - **Found while checking:** on the real canvas the Channel Status line was cut off, so the summary drops the file name and uses two lines.
  - **Follow-up (owner request 2026-09-25: too much text on the SM energy histogram):**
    - the origin counts moved to the SM summary panel, and the title became "Energy • N ROI/DOI sides" (plus "without keV" / "fitted keV factors" only when they apply);
    - the always-on legend, which repeated the readout, now appears only with "Show background fit", using short names;
    - the background-fit readout wraps onto two lines.

    Checks: views 143/143 (the origin checks now read the summary panel), issue 8/8, hidden GUI 29/29, revision 14/14, processing 17/17, engine/report 59/59; compile exit 0. Real-prefix screenshots are in the session scratchpad `energy/`.
  - **Real Cornell whole file** (`00000003`, 3,853,684 sides, resolved encal; recorded, not a pass criterion):
    - sides: fitted 3,182,098, check 62,472, borrowed 586,004 (15.2 %), est. neighbours 17,863, est. median 5,247; none without keV;
    - the side counts take 0.012 s, and the Channel Status redraw 0.09 s;
    - fitted-only uniformity leaves out a median 15.8 % of sides per SM (max 22.4 %, SM 23), and moves μ by a median of +1.84 keV (max +3.24 keV: SM 23, 506.3 → 509.6 keV). The borrowed outer slabs pull the SM photopeak down;
    - screenshots are in the session scratchpad `t17/`.
  - **Other checks:** hidden GUI 29/29, revision 14/14, processing 17/17, issue 8/8, unpopulated 6/6, slab convention 16/16, slab calibration 15/15, scale 44/44 (selftest 44/44), engine/report 59/59.
  - **Owner review pending:**
    - the defaults: fitted only off in the views and on in uniformity, and "fitted (check)" counted as fitted;
    - that a malformed sidecar rejects the calibration.

## Reports and validation

- [x] **T12 — Reports and provenance** (FR-14). *Done when:* `pypdf` extraction of synthetic IMAS/Cornell PDFs finds whole-file/prefix scope, channel-findings thresholds and table, and per-minimodule tables matching the engine values, with raw-mode fits marked unavailable.

  **Verified 2026-09-28:** `python scripts/ldat_views_check.py` → **PASS 166/166** (14 new for T12); `--real` → **PASS 169/169**; GUI compile exit 0.
  - **Code** (`src/ldat_report.py`):
    - after the SuperModule summary, a channel-findings table (34 SMs per page);
    - after each SM detail page, a minimodule page: the per-minimodule table, then the SM's flagged channel IDs by state;
    - the provenance gives the scope ("whole files (no pair limit)" / "prefix, max N coincidence pairs / file") and the findings thresholds;
    - `write_report(…, thresholds=)` refuses invalid thresholds, and the GUI passes the Channel Status thresholds.

    The values come from `system_channel_findings`, `minimodule_metrics` and `minimodule_layout`, as in the GUI.
  - **Findings pages** (T6 fixtures, raw, report thresholds low 0.04):
    - every row equals the engine for all 30 Cornell and 120 IMAS SMs, and low 0.04 turns the LOW SM OK;
    - findings pages: 1 for Cornell, 4 for IMAS; total pages equal 1 + ⌈n/32⌉ + ⌈n/34⌉ + 2n (63 and 249);
    - the pages state the population ("… ingest population, before display cuts"), the scope, "not a dead/hot hardware verdict" and the thresholds, which also appear in the provenance;
    - the mixed SM's page lists its not-observed and high channel IDs; with low 0.04 it lists no low channels. Its fits read unavailable (raw a.u.);
    - the half-populated Cornell SM 2 has 8 "unpopulated (config)" rows and its unexpected hits.
  - **Minimodule pages** (T8 fixtures, calibrated, prefix 5,000 pairs/file):
    - in the SM 1 and system reports, every row (ingest, selected, fit sides, centroid, resolution, status) equals `minimodule_metrics`;
    - SM 1 mM 3 reads insufficient events; "Fitted minimodules: k/16 populated"; SM 2 has 8 unpopulated rows;
    - each page states the prefix scope, "keV with synthetic.encal", the selected population (both energies 400..650 keV, DOI, ROI) and "energy window off";
    - raw SM 0: the counts are kept and all 16 fits read "unavailable: raw a.u. (no keV calibration)".
  - **Mutation checks** (scratch `t12_mutate.py`), each caught:
    - findings ignoring the report thresholds: 4 fail;
    - selected column showing ingest: 3;
    - scope always whole files: 2;
    - unpopulated minimodules shown as populated: 2;
    - fits dropped: 5.

    Unmodified, none fail.
  - **Found while checking** (real PDF read back):
    - the calibration file name ran off the minimodule page, so header lines wrap at 140 characters;
    - the T17 "keV factor sides" line on the SM detail page ran into the energy plot, so that panel wraps at 80 characters;
    - an SM report under the legacy rule labelled the all-file unresolved-slab count "(SM n)". It now reads "(all files; pairs, not per SM)", pinned by a new check.
  - **Real Cornell** (`00000003` whole file, 3,853,684 sides, resolved encal, 400–650 keV, DOI 0–15; recorded, not a pass criterion):
    - the system PDF has 63 pages and takes 19.9 s (background thread in the GUI); the SM 12 PDF has 5 pages and takes 1.4 s;
    - SM 12: 16/16 minimodules fitted (506.5–521.3 keV); findings NOT OBSERVED with time 262457/262624/262648, energy 262422 and low 262413, matching the Channel Status tab;
    - files are in the session scratchpad `t12/`.
  - **Changed spec 001 checks** (page counts only): SM reports have 5 pages instead of 3 (`ldat_inspector_check.py`, `ldat_gui_check.py`, `ldat_real_check.py`). The Cornell full report has 3 + 2 × 30 pages instead of 2 + 30.
  - **Other checks:**
    - engine/report 59/59, hidden GUI 29/29, revision 14/14, issue 8/8 (`--real` 8/8), processing 17/17;
    - scale 65/65 (`--real` 15/15, `--whole` 9/9), unpopulated 6/6 (`--real` 8/8), slab convention 16/16, slab calibration 15/15;
    - `ldat_real_check.py` PASS on the six-file prefix.
  - **Owner review 2026-09-28:** content OK. On a six-file SM report, the SM detail page's info text ran into the DOI histogram (owner screenshot).
  - **Follow-up fix (owner item 9):**
    - **Red:** a new check in `ldat_views_check.py` renders the SM report of a ten-file fixture with long paths (the real six-file set overflows the same way). It fails with the info text over "Selected DOI" and text off the provenance and minimodule pages. The real six-file PDF also showed that provenance lines for files [3]–[5] were cut off the bottom of page 1 (a spec 001 layout limit).
    - **Fix:**
      - the per-file timestamp spans moved to the SM's minimodule page, one line per file, so the detail page's info panel has a fixed length;
      - text pages (provenance, minimodule) shrink to at least 6 pt and then continue on "(continued)" pages. A page break never splits a wrapped entry: a mutation that allows it fails the check.
    - **Green:** views 170/170 (`--real` 173/173). The real six-file SM 0 PDF has 6 pages (provenance continues onto page 2) and lists all six files. `ldat_real_check.py` now requires every input file in the SM PDF and accepts continuation pages (6 pages on the six-file prefix).
- [x] **T18 — Cornell System Overview in the real-system orientation** (FR-21; Change 2). *Done when:*
  - `ldat_views_check.py` recomputes every Cornell minimodule centre's global Z and tangential offset with the formula of `cornell_lor_display.py:local_to_global` / `sm_map_gen` from the config. Across the composite, a lower pixel row always means a smaller Z, and a pixel column further right means a larger (θ, tangential offset). SM 2 is at the top-left, SM 0 at the bottom-left and SM 29 at the top-right;
  - IMAS placement and minimodule orientation are unchanged, equal to T8/T9;
  - the hidden GUI shows Z / θ ticks and titles, a canvas click still resolves each tile to its (SM, mM), and Cornell flood thumbnails draw local Y horizontally with local X increasing downward;
  - the T8/T9 checks that pinned the old Cornell orientation are updated and noted.

  **Red baseline:** after the engine change, exactly the 4 checks pinning the old Cornell orientation failed (163/167): the T8 flood-map orientation and spec 001 placement, the T9 "SM 4 mM 0 at (5, 8)" and the canvas click.

  **Verified 2026-09-28:** `python scripts/ldat_views_check.py` → **PASS 170/170**; `--real` → **PASS 173/173**; GUI compile exit 0.
  - **Engine:**
    - `supermodule_layout`: Cornell rows by `ring_z`, largest first; columns by the cassette angle atan2(Y, X) of `ring_yx`;
    - `minimodule_layout`: Cornell rows follow local X, columns local Y;
    - new `supermodule_axis_labels`.

    **GUI:** the overview ticks and titles, and the flood thumbnails (transposed mesh, local X downward).
  - **Geometry check:** each of the 480 Cornell tiles gets (Z, θ, tangential offset) recomputed with `local_to_global`'s formula (Z = sm_z + 48 − x, offset = y − 48, θ from `ring_yx`):
    - every lower pixel row has a smaller Z, and every column further right a larger (θ, offset);
    - rows share Z and columns share θ and offset within 0.5 mm;
    - SM 2 is at the top-left (0, 0), SM 0 at (10, 0), SM 29 at (0, 45) and SM 27 at (10, 45). SM 0's grid, top to bottom: `[15 11 7 3] [14 10 6 2] [13 9 5 1] [12 8 4 0]`;
    - labels: "Z +102 mm / Z +0 mm / Z -102 mm" and "θ 0°" … "θ 324°".
  - **IMAS unchanged:** the flood-map orientation and spec 001 placement checks still pass.
  - **Hidden GUI:**
    - ticks and titles read Z and θ, and the x label gives "→ local Y (θ), ↓ local X (−Z)";
    - clicks at (8, 8) → SM 4 mM 0 and (12, 2) → SM 0 mM 5;
    - SM 0's flood thumbnail mesh equals `flood_counts(…)` transposed (not symmetric), with y limits (102, 0).
  - **Changed checks:** the T8/T9 Cornell expectations above now use the new pixels. SM 4 mM 0 is (8, 8) (was (5, 8)), SM 0's origin is (10, 0) (was (0, 0)), and the second click is (12, 2) (was (1, 2)).
  - **Mutation checks** (scratch `t18_mutate.py`), each caught with 3 failing checks:
    - Cornell minimodules in the IMAS orientation;
    - Z ascending;
    - θ = atan2(X, Y).

    Unmodified, none fail.
  - **Found while checking:** on the real canvas the "Z +102 mm" ticks were clipped, so the Cornell left margin is 0.07.
  - **Real Cornell** (`00000003` whole file, resolved encal; screenshots in the session scratchpad `t18/`: `overview_centroid.png`, `overview_ingest.png`, `overview_flood.png`): the half-populated SMs (Z = +102) show their unpopulated half at the outer axial end (top) in both the tiles and the flood thumbnails.
  - **Other checks:** engine/report 59/59, hidden GUI 29/29, revision 14/14, issue 8/8 (`--real` 8/8), processing 17/17, scale 65/65 (`--real` 15/15, `--whole` 9/9), unpopulated 6/6 (`--real` 8/8), slab convention 16/16, slab calibration 15/15, `ldat_real_check.py` PASS.
  - **Owner review 2026-09-28:** the orientation and the report look fine.
  - **Follow-up: absolute flood-mode colour scale** (owner: "I would like to see it absolute so the colormap gives an idea on the source position"):
    - the metric modes were already one system-wide scale; flood mode autoscaled each SM thumbnail separately and had no colour bar;
    - **red:** a new hidden-GUI check found three different scales ((0.1, 18), (0.1, 24), (0.1, 31)) and no colour bar;
    - **fix:** every thumbnail uses vmin 0, vmax = the system's peak bin, with one colour bar "sides per bin (one scale, all SMs)" and "one colour scale for all SMs" in the title. This applies to COG/RTP and slab views on both systems;
    - **green:** views 171/171 (`--real` 174/174). All other checks unchanged. Real screenshot: `t18/overview_flood.png`.
  - **Geometry provenance:** the placement follows `ring_z` and `ring_yx` in `cornell_full_system.yaml`. The owner confirmed them as the real Cornell geometry on 2026-09-28, and the config's TODO was removed. `ring_r` (the skew-phantom distance) is not used.

- [x] **T13 — Integrated validation and owner review** (FR-1–FR-21; after T14–T17). *Done when:* all new and existing checks pass, compile exits 0 and the T5 real run is recorded. Owner GUI review is recorded on the Cornell set, and real IMAS is either performed or explicitly waived by the owner. Only then set the spec to `shipped`.

  **Automated part, 2026-09-28** (after T12, the item 9 fix, T18 and the absolute flood scale): every check passes and compile exits 0.
  - views 171/171 (`--real` 174/174), scale 65/65 (`--real` 15/15), engine/report 59/59, hidden GUI 29/29, revision 14/14, issue 8/8 (`--real` 8/8), processing 17/17, unpopulated 6/6 (`--real` 8/8), slab convention 16/16, slab calibration 15/15, `ldat_real_check.py` PASS;
  - T5 re-run: `ldat_scale_check.py --whole` **9/9** on the six whole Cornell files.

  **Owner review 2026-09-28** (Cornell set, GUI):
  - **Real IMAS:** waived (no IMAS data available; spec Change 2).
  - **Channel Status (T7):** OK. The time/energy channel map on a clicked SM is very clear, and "Open in SuperModule" works.
  - **System Overview (T9):** readable and working. The owner asked for the real-system orientation, which became FR-21 / T18.
  - **SuperModule tab (T10):** stepping smooth; the summary and table are well placed and readable.
  - **Coincidences (T11):** OK.
  - **Slab flood and mm DOI (T15):** OK. The four T15 design choices (display-only COG file, ROI on COG/RTP, exclusions among sides passing the cuts, one column per slab) are accepted.
  - **T17:** the defaults (fitted only off in the views and on in uniformity, "fitted (check)" counted as fitted) and a malformed sidecar rejecting the calibration are accepted.
  - **PDF (T12):** content OK; text overlapped the DOI histogram on a six-file SM report. Fixed (T12 follow-up).

  **FR walk** (✓ = automated check passes; *owner* = GUI review still open):

  | FR | Evidence | State |
  | --- | --- | --- |
  | 1 | T2: 41× single core, fast = reference; T5: six whole files in 11.6 s | ✓ |
  | 2 | T3 62 B/side; T4 whole-file mode, estimate and warning; T5 main −3.7 %, workers −0.8 % | ✓ |
  | 3, 4 | T4: 4 workers; Cancel 0.03 s with the previous dataset kept; close 0.36 s | ✓ |
  | 5–7 | T6 exact states, boundaries and insufficient; T7 tab | ✓, owner OK |
  | 8, 9 | T8 derived grids; T9 tiles, click, Open SM, non-blocking | ✓, owner OK |
  | 10, 13 | T10 stepping, summary, minimodule table; T8 fits = `fit_peak` | ✓, owner OK |
  | 11, 12 | T11 partner SM/mM, exact matrix, Δt | ✓, owner OK |
  | 14 | view labels (T7–T17); T12 PDF scope, findings, minimodule tables, no overflow | ✓, owner OK |
  | 15 | synthetic IMAS/Cornell fixtures in every task; real Cornell whole files (T5) | ✓; real IMAS waived |
  | 16–18 | T14 limits engine; T15 selectors and PDF | ✓, owner OK |
  | 19 | T16 | ✓, owner OK |
  | 20 | T17 | ✓, owner OK |
  | 21 | T18 geometry check against the scripts' `local_to_global`; one absolute flood scale | ✓, owner OK (orientation); owner OK (orientation and absolute flood scale) |

  **Owner sign-off 2026-09-28:** "everything now looks good, ready to ship" (orientation, report and absolute flood scale).

  **Verdict: PASS.** Every FR has a passing check and owner review; real IMAS is waived. Spec status set to `shipped`.

## Change 3

- [x] **T19 — Prefix choices up to the worker-result limit** (FR-22; Change 3). *Done when:* the GUI offers only the FR-22 choices (default 10k, disabled in whole-file mode); `Settings.validate` accepts 30 M and rejects 30 M + 1 and 0; the worst-case result size for 30 M is below 4 GiB; a real 30 M Cornell prefix returns through a spawn worker; whole files estimated above 30 M are listed in the estimate and refused before any worker starts, prefixes never are; the existing checks and the GUI compile still pass; the owner checks the combo box in the GUI.

  **Reproduction 2026-09-28** (`Source_200microCi_60s_coincCompact.ldat`, 25.2 GB, raw mode): the whole file read in one process gives 99,273,393 pairs, 60,081,663 accepted, in 137 s; peak private 22.2 GB (peak working set 45.8 GB counts the mapped file). Its pickled result is 6.61 GB (55.0 B/side). Returning bytes from a spawn `ProcessPoolExecutor`: 1.5, 2.5, 3.9 GiB OK; 4.1 and 6 GiB fail with `OSError: [WinError 87] El parámetro no es correcto`, the GUI's error.

  **Verified 2026-09-28:** `python scripts/ldat_pair_choices_check.py --real` → **PASS 12/12**: choices 10k … 30M parse, 30M = `MAX_PAIRS_PER_FILE`, validation bounds, worst case 59 B/side × 60 M sides = 3.54 GB, GUI combo box values/default/whole-file state, and the real 30 M prefix through a spawn worker: 18,111,524/30,000,000 accepted, 1.99 GB result. GUI compile exit 0; hidden GUI 29/29, engine/report 59/59, revision 14/14, processing 17/17. **Owner GUI check 2026-09-28:** the dropdown is OK; the owner asked to refuse whole files above 30 M.

  **Verified 2026-09-28 (refusal):** `python scripts/ldat_pair_choices_check.py --real` → **PASS 19/19**. The new checks: a synthetic 140-pair file against a cap patched to 100 is listed whole but not as a prefix; the GUI estimate label shows the refusal, and `_confirm_and_launch` shows the error without starting workers; the real 25 GB file is refused whole ("Source_200microCi_60s_coincCompact.ldat ~100 M") and allowed as a 30 M prefix. GUI compile exit 0; hidden GUI 29/29, engine/report 59/59, revision 14/14, processing 17/17, scale 65/65. **Owner GUI check 2026-09-28:** "it looks fine and the error appears trying to process the whole file". Spec status set back to `shipped`.

- [ ] **T20 — SuperModule hardware address** (FR-23; Change 4). *Done when:* `read_sm_ports` on a synthetic map returns each SM's (port, slave, FEB/D port) and leaves out SMs with a missing or malformed entry; `sm_port_text` gives "DAQ port 4 · SLAVE · FEB/D port 1" for (4, 1, 1), MASTER for slave 0, "slave ID 2" for 2 and "ports unavailable" for none; on the real `cornell_map_full_system.yaml` SM 15 is (4, 1, 1) and SM 18 is (4, 0, 1); the hidden-GUI tile text for a clicked tile contains the address; the report summary table has DAQ/M-S/FEB-D columns matching the map and "--" for an SM without an entry; the existing checks and the GUI compile still pass; the owner checks the tile line and a Full System Report on Cornell data.

  **Verified 2026-09-29 (headless):** `python scripts/ldat_ports_check.py` → **PASS 12/12**: synthetic valid/short/non-integer/bool/empty entries, missing map and key; labels MASTER/SLAVE/slave ID n/ports unavailable; Cornell and IMAS `Dataset.sm_ports` equal the maps' `mod_feb_map` (Cornell SM 15 = (4, 1, 1), SM 18 = (4, 0, 1)); hidden-GUI tile line "SM 15 · DAQ port 4 · SLAVE · FEB/D port 1 · mM …", unpopulated tile and missing entry; SuperModule report row "15 4 SLAVE 1", "--" without an entry, column alignment kept. `ldat_views_check.py` tile-text expectation updated for the address → **PASS 171/171**. GUI compile exit 0; hidden GUI 29/29, engine/report 59/59, revision 14/14, processing 17/17, issue 8/8, unpopulated 6/6, pair choices 15/15, scale 65/65. Pending: owner GUI check on Cornell data.

