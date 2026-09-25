# Plan 002 — LDATInspector scale, channel status and minimodule views

Approved spec: [`spec.md`](spec.md). Constitution: [`AGENTS.md`](../../AGENTS.md). Spec 001's engine/GUI split stays: pure analysis in `src/`, CustomTkinter in `exe_programs/ldat_inspector_gui.py`, reports in `src/ldat_report.py`.

## Measured starting point (2026-09-25)

- Reference reader: 14,500 pairs/s, one core, raw mode (Cornell `...coincCompact11s_00000003.ldat`, 50,000-pair prefix, config `cornell_full_system.yaml`, ≥4 channels, ≥0.2 a.u.).
- Cornell files: ~323 bytes/pair, ~2.5 M pairs per 760 MB file; at most 16 hits/side, mean 10. Channel IDs: Cornell 256–787,967, IMAS 0–270,335; both maps `sum_rows_cols = True`.
- Workstation: 24 logical CPUs. `numba 0.63.1` is installed in `process_petsys` and already used by `scripts_cornell/`, but it is not declared in `process_petsys.yml`. `psutil` is not installed.
- Spec 001 layout: 13 per-side columns, ~96 B/side, with partner energy/raw/key/timestamp duplicated on both sides. 15 M pairs = 30 M sides → ~2.9 GB before partner SM/mM or a calibrated copy. That breaks FR-2's 4 GB budget, so the storage changes too.

## Architecture

### Fast reader — `src/ldat_fastread.py` (FR-1, FR-2)

- Memory-map the file as `uint8` and walk records in order inside an `@njit` kernel: 2-byte header, then `(n1 + n2)` 16-byte `<qfi` hits, like `read_compact.py`.
- The kernel runs in chunks (default 250,000 pairs) and returns the next byte offset. That bounds buffers, allows progress and cancel between chunks, and turns a truncated tail into the same "no prefix from a corrupt file" error as spec 001.
- Map lookups become dense arrays indexed by absolute channel ID (size max ID + 1, ≤ 1 M entries): `sm`, `mm`, `is_time`, `is_energy`, `fine`/`coarse` coordinates, Cornell slab position, `mapped`. A missing entry reproduces the reference's `KeyError`.
- For each pair the kernel copies the reference's order and precedence exactly:
  1. per-channel energy cut;
  2. `filter_min_ch` on det1, then det2, short-circuiting on the first failure, including the `sum_rows_cols` rule `min_ch <= n_energy < n_hits`;
  3. `get_maxEnergy_sm_mM`: strict `>` against a running max that starts at 0, and a single minimodule returns all hits;
  4. selected-minimodule min-ch, then the max-energy time channel;
  5. Cornell slab;
  6. centroid (TIME → fine X, power 1; ENERGY → Y, power 2; offset 1e-5);
  7. DOI (`sum/max` of energy channels);
  8. finiteness check and `energy > 0`.
- Each pair gets one error code, mapped to spec 001's error labels (`min channels`, `unresolved Cornell slab`, `ValueError`, `KeyError`, `ZeroDivisionError`). Sums accumulate in hit order in float64, as the Python reference does.
- Per-channel time/energy hit counts are added only for accepted pairs.
- Cornell one-time-channel slabs keep the random-neighbour rule, but use a numba RNG seeded from the file index, so runs are reproducible. The kernel records a `random_slab` flag.
- **Known deviation:** `get_maxEnergy_sm_mM` breaks exact ties by Python set order. The kernel takes the first minimodule in hit order. The oracle check counts ties; any non-zero count on real data is reported, not hidden.

### Reference oracle (FR-1, FR-15)

`process_file_reference` is spec 001's `process_file` logic, unchanged. It is not deleted: checks compare the fast reader against it.

### Compact storage (FR-2, FR-11)

`Dataset` holds one side table sorted by (SM, file, record order). Each SuperModule is a contiguous slice.

| Column | dtype | Bytes |
| --- | --- | ---: |
| `raw_energy` | f8 | 8 |
| `energy` (active view; the same array as `raw_energy` when uncalibrated) | f8 | 8 |
| `x`, `y`, `doi` | f8 | 24 |
| `timestamp` | i8 | 8 |
| `calibration_key` (`(ch << 5) + slab`, max 25.2 M) | i4 | 4 |
| `partner` (row of the other side) | i4 | 4 |
| `sm` | i2 | 2 |
| `file_index` | i2 | 2 |
| `mm` | i1 | 1 |
| `random_slab` | bool | 1 |

That is ≈ 62 B/side → ≈ 1.9 GB for 30 M sides. Keeping f8 means no display or report value changes (the spec's precision clause is not needed).

- `ModuleEvents` keeps its attribute names. Own columns are zero-copy slices. `partner_energy`, `partner_raw_energy`, `partner_calibration_key`, `partner_timestamp`, `partner_sm` and `partner_mm` are gathered on access through `partner`. Assigning them is not supported; `scripts/ldat_issue_check.py:260` gets an explicit fixture instead.
- `apply_calibration` replaces only `energy` (one new f8 column); raw mode restores the shared raw array.
- **Merge:** each worker returns its per-file table. The main process concatenates column by column, freeing each part as it goes, then orders by SM with one `argsort` and gathers column by column. Peak ≈ table + one column + order array. If the memory check fails, the fallback is workers writing `.npy` columns to a session temp folder that the main process loads one at a time.

### Processing orchestration (FR-2–FR-4)

- `Settings.max_pairs = None` means whole file. The GUI gets a "Whole files" switch next to the pairs field; prefix mode keeps its label.
- Workers: `min(len(files), os.cpu_count() - 2)`. Each is a `ProcessPoolExecutor` process with an initializer that receives a shared cancel `Event` and a per-file progress `Array`. The kernel checks cancel between chunks.
- **Cancel:** set the event, then `shutdown(wait=False, cancel_futures=True)`. The GUI keeps the previous dataset (or its empty state) and logs the cancel.
- **Closing the window:** does the same, so close never waits for a whole-file worker.
- The spec 001 rule that Tk widgets are only touched on the main thread still holds; the progress `Array` is polled from `_poll_events`.

### Channel findings — engine `channel_findings(dataset, sm, low_frac, high_frac, min_median)` (FR-5–FR-7)

- Works per channel type (time/energy) over the expected channels (Cornell inactive minimodules are already excluded). Each channel's hit count is compared with the median of its type in that SM.
- States: `NOT OBSERVED` (0 hits), `LOW` (< `low_frac` × median), `HIGH` (> `high_frac` × median), `OK`.
- `INSUFFICIENT EVENTS` applies when the SM has fewer than 100 ingest sides (spec 001's rule) or its per-type median is below `min_median`.
- **Provisional defaults**, editable in the tab and printed in reports: `low_frac = 0.15` (RAWInspector's count-based dead test), `high_frac = 3.0`, `min_median = 20` hits.
- Row state priority: not observed > high > low > insufficient > OK. A type that could not be assessed ranks below any finding but above OK, so a row reads OK only when both types were assessed. Colours follow RAWInspector: OK `#b3ffb3`, not observed `#ffb3b3`, high `#ffc4e1`, low `#ffd9b3`, insufficient `#dce0e4`. Text states stay alongside the colours.
- A mapped SM with no ingest sides reads `NO DATA` (coloured like not observed); its channels read insufficient events.
- The median is taken over the SM's expected channels of that type, zeros included. Hits on mapped channels the config does not expect (unpopulated minimodules) are not assessed; they are listed per SM as unexpected hits.
- Engine API: `FindingThresholds(low_frac, high_frac, min_median, min_events=100)`, `channel_findings(dataset, sm, thresholds)` and `system_channel_findings(dataset, thresholds)`. Each insufficient type carries its reason (too few sides, or the median below the minimum).
- Wording is always "occupancy in the coincidence ingest population".

### Minimodule layout and metrics (FR-8, FR-9, FR-13)

- `minimodule_layout(dataset)` is derived from the selected map, which the dataset now keeps (coordinates, channel modules and types; T7). It is computed on demand, not at merge; it takes milliseconds. Row 0 is the largest Y and column 0 the smallest X, as the flood map is viewed. Two minimodules at one centre are rejected rather than merged. Each minimodule's centre is (mean fine X of its time channels, mean fine Y of its energy channels). Grid rows/columns come from the distinct rounded centres, so it is 4×4 for the current IMAS and Cornell maps without being assumed. `mM_disposition` is not used. Mapped minimodules listed in the config's `unpopulated_minimodules` (Cornell SM 2, 5, …, 29; B3) are kept and drawn as unpopulated.
- SuperModule placement moves spec 001's `_layout` into the engine as `supermodule_layout(dataset)`. Cornell: ring = `sm % 3`, cassette = `sm // 3`. IMAS: ring-major, `ncols = len(ring_yx)`. A check confirms every mapped SM gets a unique cell.
- `minimodule_metrics(dataset, selection)` gives, per SM and minimodule:
  - ingest sides;
  - selected sides;
  - when calibrated, `fit_peak` on the ROI/DOI-selected energies (energy cut off, as in `uniformity`).
- Fits (~480 for Cornell, ~1,900 for IMAS) run in a background thread and are cached per (dataset identity, selection). Views show "computing…" until they arrive.

### GUI (FR-5, FR-6, FR-8–FR-10, FR-12)

- **Tabs:** Channel Status, SuperModule, System Overview, Coincidences (T11), Timestamps. The "SuperModule Status" text tab is removed; its content is in the SuperModule summary panel (T10).
- **Redraw:** `_refresh_all` validates the cuts, marks the SuperModule, System Overview and Timestamps tabs stale and draws the visible one; the tab-change command draws a stale tab when it is shown. Changing the SM marks only the SuperModule tab stale, so stepping never recomputes overview tiles or timestamps. Channel Status keeps its own rule (new data and Apply). A finished overview job redraws only when its tab is visible.
- **Channel Status:** SM table with the counts from the findings (ingest sides, minimodules seen/expected, per type: assessed channels, not observed / low / high, median hits; unexpected hits; finding). Threshold entries plus Apply. A detail figure for the selected row:
  - left, the SM's minimodule outline from `channel_geometry` (unpopulated minimodules hatched), drawn twice: time channels as vertical segments at their fine X, and energy channels as horizontal segments at their fine Y. They are separate maps because the two sets cross over the same area and their medians differ by about 3×. Segments are coloured by hits / type median and outlined by state;
  - right, per-channel bars for each type, grouped by minimodule in map order and coloured by state, with median and threshold lines; zero-hit channels are marked with ×;
  - a list of the flagged channels (ID, hits, state) and unexpected hits;
  - an "Open in SuperModule" button. Clicking a row only draws its detail; it no longer switches tab.
  - The findings use the ingest population only, so the tab is redrawn on new data and on Apply, not on display-cut changes. The spec 001 "Selected sides" column and SM participation bar chart are dropped from this tab; selected counts belong to the SuperModule summary (T10) and System Overview (T9).
- **SuperModule** (the renamed Explorer):
  - Stepping: "◀ Prev" / "Next ▶" buttons and an "n / N" position label beside the SM box, clamped at the ends (the button at an end is disabled). The mouse wheel over the toolbar and SM box and Page Up/Page Down (only while this tab is visible) also step; they update the SM box at once and redraw 150 ms after the last step, so a fast scroll draws once. Buttons redraw at once. Tooltips give the shortcuts.
  - A summary panel beside the plots. A header gives the SM, the system and its finding, coloured as in Channel Status. Below it, in a vertical split the operator can drag:
    - occupancy: ingest sides (and share of the system), selected sides, minimodules seen / expected;
    - channel findings per type: observed / expected, median, not observed / low / high counts with the channel IDs, or the insufficient reason with the zero-hit IDs; unexpected hits; the thresholds. This keeps everything the spec 001 Status tab showed;
    - the per-minimodule Treeview (mM, ingest, selected, centroid keV, resolution %, fit status). Unpopulated minimodules are grey rows reading "unpopulated (config)"; failed fits show their status, never a value; raw mode shows every fit as unavailable (raw a.u.). Selecting a row shows its full fit status.
  - The table comes from `minimodule_metrics(dataset, selection, sms=[sm])` on a daemon thread: one job at a time, and when it ends the latest wanted (SM, selection) runs next, so SMs skipped while stepping are never computed. Results are cached per (dataset, selection, SM) for 32 entries; a System Overview result with fits for the same selection is reused. The table shows "computing…" until then. The spec 001 energy, DOI and flood plots are unchanged and still drawn on the Tk thread.
- **System Overview:**
  - One composite `imshow` for the whole system: SM cells laid out by ring/column, each drawn as its minimodule grid, with one-pixel NaN gaps between SMs.
  - Metric selector: ingest counts, selected counts, centroid (keV) and resolution (%), the last two calibrated only.
  - Inactive/unmapped minimodules are hatched grey; zero counts use the colormap minimum; unavailable fits are a distinct grey with a legend.
  - Clicking a pixel resolves to (SM, mM) and shows its identity and value, plus "Open SM".
  - Engine: `overview_grid(dataset, metrics, metric)` builds the composite (`TILE_VALUE / EMPTY / UNPOPULATED / UNAVAILABLE`, pixel → (SM, mM), reasons). The GUI computes `minimodule_metrics` in a daemon thread: counts only for the count views, and counts plus fits for the fit views. The result arrives through the existing event queue and is cached per (dataset, selection) for the last four requests. A newer request supersedes the running job, which stops at its next SuperModule.
  - The spec 001 "Flood maps" view stays as a fifth option, placed by `supermodule_layout`.
- **Coincidences:**
  - SM × SM matrix of accepted pairs where both sides pass the current display cuts, each pair counted once. Every mapped SM is on both axes; the diagonal holds pairs with both sides in one SM. Log colour scale; 0 pairs is grey.
  - Clicking a cell selects an SM pair (SM a = row, SM b = column); SM a / SM b boxes also choose it, SM b may be "All partners".
  - A Δt histogram (`t_a − t_b`, ns, fixed sign for the ordered pair; or one SM against all partners) with median and central-68 % width (p84 − p16). The histogram spans the core ± 4 central-68 % widths, clipped to the data; pairs outside are counted on the plot, and the statistics use every pair.
  - Labelled: geometry and time of flight contribute; not a clock or CTR calibration.
  - Engine: `pair_mask(dataset, selection)` (per side: both sides of its pair pass `Selection.mask`), `pair_matrix(dataset, selection, mask=)` and `pair_dt(dataset, sm_a, sm_b=None, mask=)`, all through the `partner` column. A pair with both sides in one SM is counted once in Δt, with the lower row as side a (sign arbitrary, and labelled so). Differences are taken in integer ps, then converted to ns.
  - GUI: the mask and matrix come from a daemon thread and are cached per (dataset, selection) for two selections (the mask is one bool per side). Δt is computed on the Tk thread from the cached mask (milliseconds per SM). The tab is redrawn on cut changes when visible; stepping SMs does not touch it.

### Limits files and slab view (FR-16–FR-18)

- **Loader:** `load_limits(path)` parses `(ch, slab)\tleft\tright` lines, as in `scripts_cornell/cornell_listmode_cog_fixed_position.py:143-157`, into a `Limits` object: sorted calibration keys `(ch << 5) | slab` with left/right arrays, looked up with `searchsorted` (NaN where there is no entry). This replaces the dense `[time channel, 16, 2]` array: Cornell channel IDs reach ~788,000, and the keys are the table's own `calibration_key` encoding. Malformed lines, a slab outside 0–15, non-finite limits, duplicate keys and files without entries are errors; the file is not partially accepted. Blank lines and CRLF are accepted.
- **Derived values:** computed on demand per SuperModule from stored columns (`slab_view(dataset, data, cog)`, `decompressed_doi(dataset, data, doi)`; Cornell datasets only). The time channel is `calibration_key >> 5` and the slab is `calibration_key & 31`. Nothing is added to ingest or storage, and loading, swapping or removing a file only recomputes views.
  - **Slab X:** time-channel fine X ± 0.8 mm by slab parity (even −0.8, odd +0.8), using the convention fixed in B1.
  - **Decompressed Y:** `clip((y − left)·25.6/(right − left), 0, 25.6) + (3 − mm // 4)·25.6`, from `cornell_floodmaps.py:173-183`. The stored Y is the same centroid (`calculate_centroid(…, 1, 2)`) that `cornell_cog_decompress_params.py` used to build the limits. The clipped count is returned.
  - **Decompressed DOI:** `(doi − right)·20/(left − right)`; outside [0, 20] is NaN and counted ("out of range"; 0 and 20 are kept, as in list mode).
  - **Exclusions:** NaN, counted per reason: "missing key" (no entry) and "invalid limits" (left == right, which the scripts' `+ 1e-9` would turn into a clipped or huge value). The scripts' `+ 1e-9` in the denominator is otherwise omitted.
  - **Unresolved slabs:** those pairs were rejected at ingest and are not in the table; `unresolved_slab_pairs(dataset)` gives their count from the ingest counters for titles and provenance.
- **GUI:**
  - Two optional file pickers ("COG limits…", "DOI limits…") in the inputs card with shortened names. Clicking a name shows the full path and offers to remove the file. A malformed file is rejected with a dialog, and the previous file is kept. Pickers are refused while processing or writing a report.
  - Flood view selector: "COG / RTP" or "Slab (decompressed)", Cornell with a COG file only. It sits in the SuperModule tab's ROI group and in the System Overview toolbar (one shared value).
  - DOI unit selector: "Ratio" or "Decompressed mm", with a DOI file only. The cut range resets to 0–15 (ratio) or 0–20 mm.
  - When a selector is unavailable it is disabled, with a grey note ("n/a: Cornell only" / "n/a: no COG limits file" / "n/a: no DOI limits file"), and it falls back to the default view (COG / RTP, Ratio with the cut reset). An IMAS dataset keeps the files loaded but makes both views unavailable.
  - Excluded-side counts appear on the plots. The DOI and flood titles stay short (the summary panel narrows the canvas), and the counts per reason go in an in-axes note. For the SM flood, these are the sides passing the cuts, with the clipped count and the unresolved-slab pairs from ingest (all SMs). For the DOI, they are the energy/ROI sides. The System Overview flood mode follows the same selector and puts the totals in its suptitle.
  - **COG limits are display-only:** loading, swapping or removing one only redraws the SuperModule and System Overview tabs. The dataset and its cached per-minimodule, SM and pair results are unchanged.
  - **The mm DOI view is a dataset view:** `apply_doi_view(dataset, doi)` sets `SideTable.doi` to the decompressed depth; the stored ratio stays in `columns["doi"]` and `ModuleEvents.doi_ratio`. So the DOI cut, counts, fits, pair matrix and reports all use mm, and an excluded side (NaN) fails every DOI cut. `Selection.mask(doi=False)` gives the DOI plot its population.
    - The view is applied by a background job (like calibration), or in the processing thread when mm is chosen before processing. `apply_calibration` keeps it.
    - As a result, in mm mode sides without a DOI entry have already failed the DOI cut, and the slab flood reports them as "excluded: none".
  - **The ROI stays on COG/RTP coordinates:** in the slab view the ROI rectangle and the drag-to-select / profile region are off, and a note says so. Changing the flood view never changes any population.
  - **Slab-view flood columns:** one column per mapped slab X of the SM (`slab_x_edges`: edges halfway between neighbouring slab positions; 64 for a full Cornell SM). Slab X is discrete, and the regular bins alias against the 1.6 mm pitch. Y uses the bins control over 0–102.4 mm (`SLAB_EXTENT_MM`).
- **Reports:** `write_report(…, cog_limits=, doi_limits=, slab_flood=)`.
  - **Provenance** lists both limits paths (key counts, left == right entries) or "none", the active flood view with the slab exclusions among sides passing the cuts in the report scope, the clipped count and the unresolved-slab pairs, and the DOI view with `doi_excluded`. The display line reads "DOI mm a..b".
  - **SM pages** use the active views, and a slab-flood report without a COG file is refused.

### Non-adjacent slab recovery (FR-19)

- `Settings.slab_rule`: `"legacy"` (default) or `"recover_non_adjacent"`. It is an ingest setting, because accepted pairs change; switching rules re-reads the files.
- Reference oracle: a wrapper around the fixed `get_slab_cornell` applies the FR-19 branches before falling back to it. The fast kernel mirrors the same branches, and the oracle comparison covers both rules.
- A `recovered_slab` flag column (bool, 1 B/side) feeds the counts and flags calibration mismatch.

### Calibration factor provenance (FR-20)

- **Loader:** `load_calibration_status(encal_path)` reads `<encal stem>_status.txt` when present into a `CalibrationStatus` (sorted calibration keys and `int8` codes, `FACTOR_ORIGINS`): fitted / fitted (check) / borrowed / estimated (neighbours) / estimated (median) / no fit / unknown.
  - Each status is matched exactly against the texts `cornell_slab_en_cal.py` writes: `fit`, `fit; check: …`, `borrowed from slab N`, `estimated from neighbour slabs [..]`, `estimated from minimodule median (N slabs)` and `no fit…`.
  - An unrecognised status, a bad line, slab ≥ 16, a duplicate key or a file with no entries rejects the sidecar, and with it the calibration (the calibration error dialog). No origin is guessed.
  - A key missing from the sidecar is "unknown".
- **Storage:** `apply_calibration` loads the sidecar (Cornell, keV only) and sets a per-side `int8` origin view on the table (`SideTable.origin`, `ModuleEvents.origin`; about 3.9 MB on the 3.85 M-side whole file). It replaces the planned "no per-side storage" lookup, because the fitted-only cut needs a per-side test in `Selection.mask`. The DOI view keeps the origins, raw a.u. has none, and `Dataset.calibration_status` records the file.
- **Counts:**
  - `factor_origins(dataset, data, mask)` counts sides by origin (a `bincount` on the SM slice). It is empty in raw a.u. and for IMAS (no slab factors). Without a sidecar every side is "unknown".
  - `slab_origins(dataset, sms)` counts mapped slabs by origin, two per time channel (B1: slabs 2p and 2p + 1). Slabs of the config's unpopulated minimodules are "unpopulated", not "no fit".
- **Fitted-only cut:** `Selection.fitted_only` keeps only sides whose *own* factor is fitted or fitted (check); the partner is still subject to the energy window. An unknown origin is never assumed fitted, so with no sidecar the cut keeps nothing, and the GUI disables it then.
  - **GUI:** a "Fitted keV factors only" checkbox in the energy group, off by default, applies to every view's population. Photopeak Uniformity has its own switch, on by default. Both are disabled with a grey "n/a: raw a.u. / Cornell only / no _status.txt" note when origins are unavailable.
- **Display:**
  - The SuperModule summary panel, beside the energy plot, has a "KEV FACTORS (energy plot sides)" block with the origin counts of the plotted sides ("no fit" sides have no keV, so they are not plotted), or "origin unknown: no _status.txt". The energy title ends in "• fitted keV factors" when that cut is on. The counts are not drawn on the plot (owner request 2026-09-25: less text on the energy histogram).
  - **Energy plot text (owner request 2026-09-25):**
    - the title is "Energy • N ROI/DOI sides", adding "(k without keV)" only when some sides have no factor;
    - the fit readout box keeps the numbers, with the background-fit line wrapped onto two lines;
    - the legend is drawn only with "Show background fit" (several curves), using short names ("Photopeak model", "Gaussian component", …). The line labels keep the full values that spec 001's issue check reads.
  - Channel Status has a "keV factor sides borrowed / est." column per SM, plus summary lines with the ingest sides and the mapped slabs by origin.
  - The console logs the sidecar and its mapped-slab counts on processing or calibration.
- **Reports:** the provenance lists the sidecar path, the ingest sides and mapped slabs by origin in the report scope, and whether the fits used fitted factors only. Without a sidecar it reads "keV factor origins: unknown (… never assumed fitted)". SM pages list their sides by origin.
- **Unpopulated minimodules (B3):** they come from the config, so their slabs are neither expected nor counted as unavailable.

### Reports (FR-14)

- Provenance says "whole file" or "prefix N pairs/file".
- New system page: channel-findings table with thresholds.
- Each SM page gets a per-minimodule table (ingest, selected, centroid, resolution, status).
- All from the same engine functions as the GUI.

## Decisions and alternatives

- **numba kernel** over a pure-NumPy segmented decode. The pure-NumPy route needs header removal plus segmented argmax per (side, minimodule), which is hard to make match the reference's error precedence exactly. A sequential kernel mirrors the reference line by line.
  - Add `numba==0.63.1` (the installed version) to `process_petsys.yml`.
  - A kernel in `src/` keeps the GUI importable without compiling at import time; the first call compiles (cached with `cache=True`).
- **Seeded Cornell random slab:** the same rule as spec 001, owner-retained, but reproducible per file. The oracle compares slab keys only for non-random sides and compares random-side counts.
- **Memory estimate (FR-2 revised):** `estimate_memory(files, max_pairs, settings)` scans the first 20,000 records of each file with the fast reader for bytes/pair and acceptance, then estimates `sides × (62 B table + 16 B merge transient) + baseline`, with the baseline measured once. Available RAM comes from `GlobalMemoryStatusEx` via `ctypes`. The GUI shows the estimate after files/pairs change and asks before proceeding when it exceeds available RAM. Measured peak = main-process `PeakWorkingSetSize` (`GetProcessMemoryInfo`); worker peaks are recorded separately. No new dependency.
- **Out of scope here:** faster fitting or new fit models. Per-minimodule fits reuse `fit_peak` unchanged.

## Requirements to components

| FR | Components |
| --- | --- |
| 1 | `ldat_fastread` kernel, reference oracle, scale check |
| 2 | Compact side table, whole-file settings, merge, memory estimate and warning |
| 3, 4 | Worker pool, cancel event, GUI Cancel and close handling |
| 5–7 | `channel_findings`, Channel Status tab |
| 8, 9 | `minimodule_layout`, `supermodule_layout`, `minimodule_metrics`, System Overview |
| 10, 13 | SuperModule tab stepping, summary panel, per-minimodule table |
| 11, 12 | Partner column, `pair_matrix`, `pair_dt`, Coincidences tab |
| 14 | View labels, `ldat_report` pages |
| 16–18 | `load_limits`, derived slab/decompressed columns, flood and DOI selectors, provenance |
| 19 | `Settings.slab_rule`, oracle wrapper, kernel branch, `recovered_slab` flag |
| 20 | `load_calibration_status`, origin codes, `fitted_factors_only`, view/report counts |
| 15 | `scripts/ldat_scale_check.py`, `scripts/ldat_views_check.py`, existing four checks, owner review |

## Checks

- `scripts/ldat_scale_check.py --selftest`: synthetic IMAS/Cornell native-layout fixtures covering unmapped channel IDs on either side, a zero-energy minimodule, the min-ch `sum_rows_cols` boundary, one-time-channel and non-adjacent Cornell slabs, truncated header/hit, an empty file and prefix limits. The fast reader must match the reference oracle exactly on counts, error labels, channel counters and columns.
- `--real`:
  - fast vs reference on a 200,000-pair prefix of Cornell `00000003`;
  - reproduces spec 001 T9's 80,000-pair accepted/rejection table for all six files (e.g. `00000003`: 62,460 accepted, 11,691 min-channel, 5,457 unresolved slab, 392 `ValueError`);
  - the whole six-file run within 5 min, with the pre-run memory estimate within ±25 % of the measured main-process peak.
- `scripts/ldat_views_check.py`:
  - fixtures with known hot/low/missing/inactive channels, a synthetic non-4×4 map to prove the layout is derived, known pair matrices and Δt, and per-minimodule fits;
  - hidden-GUI stepping, row/tile/cell clicks, stale-tab redraw and cancel.
- The existing `ldat_inspector_check`, `ldat_revision_check`, `ldat_issue_check` and `ldat_gui_check` stay green, with spec 001's counts unchanged.
- Owner GUI review on the Cornell set. Real-IMAS availability is asked for at validation; there is no silent waiver.
