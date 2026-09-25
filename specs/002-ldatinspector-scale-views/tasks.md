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
- [ ] **T2 — numba fast reader** (FR-1). Write `src/ldat_fastread.py` and declare `numba==0.63.1` in `process_petsys.yml`. *Done when:*
  - `ldat_scale_check.py --selftest` shows fast = reference on all fixtures: exact accepted/read counts, error-label counts and channel counters; columns identical, except slab bits of `random_slab` sides, whose counts match;
  - `--real` matches the reference on the 200,000-pair Cornell `00000003` prefix and reproduces spec 001 T9's six 80,000-pair rows;
  - single-core throughput is recorded against the 14,500 pairs/s baseline;
  - the tie count is reported.
- [ ] **T3 — Compact SM-sorted side table with partner SM/mM** (FR-2, FR-11). *Done when:*
  - fixtures give the expected `partner_sm`/`partner_mm` for every side;
  - measured table bytes/side ≤ 64;
  - calibration switching replaces only `energy`;
  - the four existing checks pass with unchanged counts. The `ldat_issue_check.py:260` fixture is updated and noted.
- [ ] **T4 — Whole files, all cores, Cancel, memory estimate** (FR-2–FR-4). *Done when:*
  - the estimate appears after file/pairs changes, and a warning dialog appears when a synthetic estimate exceeds the mocked available RAM;
  - a hidden-GUI check processes a multi-file synthetic set in whole-file mode with more than 2 workers;
  - Cancel mid-run returns control within 2 s, keeps the previous dataset and logs the cancel;
  - a subprocess test closes the window mid-run and exits within 5 s;
  - prefix and whole-file labels appear in the log and provenance.
- [ ] **T5 — Real whole-file acceptance** (FR-1, FR-2). *Done when:* `ldat_scale_check.py --real --whole` processes all six Cornell `coincCompact11s_00000003`–`08` files in under 5 min. It records wall time, accepted/read pairs per file, the pre-run estimate, main-process peak working set (estimate within ±25 %) and worker peaks.

## Views

- [ ] **T6 — Channel findings engine** (FR-5, FR-7). *Done when:* `scripts/ldat_views_check.py` fixtures with known not-observed, low, high, OK and insufficient channels (plus Cornell inactive minimodules) give exact per-channel states, row priority and threshold changes on both maps.
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
