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

- `minimodule_layout(dataset)` is built at merge from the selected map. Each minimodule's centre is (mean fine X of its time channels, mean fine Y of its energy channels). Grid rows/columns come from the distinct rounded centres, so it is 4×4 for the current IMAS and Cornell maps without being assumed. `mM_disposition` is not used. Mapped minimodules listed in the config's `unpopulated_minimodules` (Cornell SM 2, 5, …, 29; B3) are kept and drawn as unpopulated.
- SuperModule placement moves spec 001's `_layout` into the engine as `supermodule_layout(dataset)`. Cornell: ring = `sm % 3`, cassette = `sm // 3`. IMAS: ring-major, `ncols = len(ring_yx)`. A check confirms every mapped SM gets a unique cell.
- `minimodule_metrics(dataset, selection)` gives, per SM and minimodule:
  - ingest sides;
  - selected sides;
  - when calibrated, `fit_peak` on the ROI/DOI-selected energies (energy cut off, as in `uniformity`).
- Fits (~480 for Cornell, ~1,900 for IMAS) run in a background thread and are cached per (dataset identity, selection). Views show "computing…" until they arrive.

### GUI (FR-5, FR-6, FR-8–FR-10, FR-12)

- **Tabs:** Channel Status, SuperModule, System Overview, Coincidences, Timestamps. The "SuperModule Status" text tab is removed after its content moves to the SuperModule summary panel.
- **Redraw:** `_refresh_all` redraws only the visible tab and marks the others stale; switching tabs draws a stale tab. Stepping through SMs at the new scale needs this: fits and overview tiles must not be recomputed on every SM change.
- **Channel Status:** SM table with the counts from the findings. Threshold entries plus Apply. A detail figure for the selected row:
  - left, the SM's 4×4 minimodule outline, with time channels as vertical segments at their fine X and energy channels as horizontal segments at their fine Y, coloured by hit count and outlined by state;
  - right, per-channel bars grouped by minimodule, with median and threshold lines.
  - An "Open in SuperModule" button.
- **SuperModule** (the renamed Explorer):
  - Prev/Next buttons; mouse wheel over the toolbar and SM combobox (debounced 150 ms); Page Up/Page Down shortcuts. Tooltips and labels match.
  - A summary panel beside the plots: occupancy, channel findings and a per-minimodule Treeview (mM, ingest, selected, centroid, resolution, fit status). Raw mode shows the fit columns as unavailable.
- **System Overview:**
  - One composite `imshow` for the whole system: SM cells laid out by ring/column, each drawn as its minimodule grid, with one-pixel NaN gaps between SMs.
  - Metric selector: ingest counts, selected counts, centroid (keV) and resolution (%), the last two calibrated only.
  - Inactive/unmapped minimodules are hatched grey; zero counts use the colormap minimum; unavailable fits are a distinct grey with a legend.
  - Clicking a pixel resolves to (SM, mM) and shows its identity and value, plus "Open SM".
- **Coincidences:**
  - SM × SM matrix of accepted pairs where both sides pass the current display cuts, each pair counted once.
  - Clicking a cell selects an SM pair.
  - A Δt histogram (`t_a − t_b`, ns, fixed sign for the ordered pair; or one SM against all partners) with median and central-68 % width.
  - Labelled: geometry and time of flight contribute; not a clock or CTR calibration.

### Limits files and slab view (FR-16–FR-18)

- **Loader:** `load_limits(path)` parses `(ch, slab)\tleft\tright` lines, as in `scripts_cornell/cornell_listmode_cog_fixed_position.py:143-157`, into a dense f8 array `[dense time channel, 16, 2]`, NaN where there is no entry. Malformed lines are an error; the file is not partially accepted.
- **Derived values:** computed on demand per SuperModule from stored columns. The time channel is `calibration_key >> 5` and the slab is `calibration_key & 31`. Nothing is added to ingest or storage, and loading, swapping or removing a file only recomputes views.
  - **Slab X:** time-channel fine X ± 0.8 mm by slab parity, using the convention fixed in B1.
  - **Decompressed Y:** `clip((y − left)·25.6/(right − left), 0, 25.6) + (3 − mm // 4)·25.6`, from `cornell_floodmaps.py:173-183`.
  - **Decompressed DOI:** `(doi − right)·20/(left − right)`; outside [0, 20] is NaN and counted.
  - **Missing keys:** NaN, counted per reason.
- **GUI:**
  - Two optional file pickers ("COG limits…", "DOI limits…") with shortened names that show the full path, like spec 001's calibration label.
  - Flood view selector: "COG / RTP" or "Slab (decompressed)", Cornell with a COG file only.
  - DOI unit selector: "Ratio" or "Decompressed mm", with a DOI file only. The cut range resets to 0–15 (ratio) or 0–20 mm.
  - Excluded-side counts appear in the plot titles.
  - The System Overview flood mode follows the same selector.
- **Reports:** limits paths and exclusion counts go in provenance. SM flood/DOI pages use the active views.

### Non-adjacent slab recovery (FR-19)

- `Settings.slab_rule`: `"legacy"` (default) or `"recover_non_adjacent"`. It is an ingest setting, because accepted pairs change; switching rules re-reads the files.
- Reference oracle: a wrapper around the fixed `get_slab_cornell` applies the FR-19 branches before falling back to it. The fast kernel mirrors the same branches, and the oracle comparison covers both rules.
- A `recovered_slab` flag column (bool, 1 B/side) feeds the counts and flags calibration mismatch.

### Calibration factor provenance (FR-20)

- **Loader:** `load_calibration_status(encal_path)` reads `<encal stem>_status.txt` when present, giving `(time channel, slab)` → one of fitted / check / borrowed / estimated-neighbours / estimated-median / no-fit.
- **Storage:** the dense key array built at calibration time gains a parallel `int8` origin code, so a side's origin is one lookup on `calibration_key`. Nothing is added to per-side storage.
- **Views and reports:** per-SM origin counts come from a `bincount` on the active SM slice. `Selection` gets an optional `fitted_factors_only` flag used by photopeak/uniformity fits, off by default in views and on by default in uniformity.
- **No sidecar:** origin is "unknown" for every key.
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
