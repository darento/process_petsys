# Spec 002 — LDATInspector scale, channel status and minimodule views

Status: `in progress`

Builds on shipped spec 001. Offline PETsys coincidence inspection only; the constitution is `AGENTS.md`. Reference UI: `process_cmb/exe_programs/RAWInspector.py` ("ADC Status", "System Overview", "Module Explorer"), adapted to what LDAT coincidence records support.

## Context

- Measured on 2026-09-25: ingest runs at ~14,500 coincidence pairs/s on one core (Cornell `...coincCompact11s_00000003.ldat`, 50,000-pair prefix, raw mode). That file is 760 MB, ~323 bytes/pair, ~2.5 M pairs; the six-file set is ~15 M pairs. Spec 001 caps reading at 1 M pairs/file, defaults to 10,000, and uses at most 2 worker processes.
- Retained per detector side: minimodule energy sum, calibration key, local X/Y, DOI ratio, time-channel timestamp, SM/mM, file index, partner energy and partner timestamp. **Not retained:** partner SM/mM, per-channel energies. Per-channel *hit counts* are retained per SM for the ingest population.
- IMAS and Cornell SuperModules are both 4×4 minimodules: the map's channel coordinates for SM 0 give 16 minimodules at 4 distinct X × 4 distinct Y positions in `imas1DAQ_map.yaml` and `cornell_map_full_system.yaml` (`_get_local_mapping` always uses `TbpetGeom`; `BrainGeom`'s 2×8 is unused). The maps' `mM_disposition` (`[2, 1]` for Cornell) does not describe this grid. Cornell SM 2, 5, 8, …, 29 are half-populated: minimodules {2, 3, 6, 7, 10, 11, 14, 15} have no sensors (owner-confirmed 2026-09-25), declared in the config as `unpopulated_minimodules` (bug fix B3). Spec 001 had wrongly hardcoded SM 20–29.

## Requirements

### Speed and scale

- **FR-1** — WHEN processing representative LDAT files, ingest SHALL be substantially faster than spec 001's measured 14,500 pairs/s single-core rate on the same file and settings, with identical accepted-pair counts, rejection counts and retained values. On the owner's workstation, the six-file Cornell `coincCompact11s_00000003`–`08` set (~15 M pairs) SHALL complete whole-file processing in under 5 minutes.
- **FR-2** — WHEN the operator requests it, the inspector SHALL read entire files without a per-file pair cap, with retained per-side columns for every accepted pair. Before processing starts, the inspector SHALL show an estimated memory use for the selected files and pairs-per-file setting (from file sizes and sampled bytes per pair), warn when the estimate exceeds the available physical memory, and let the operator proceed or change the selection; the estimate SHALL be within ±25 % of the measured main-process peak on the six-file Cornell set. There is no fixed memory cap. Reduced column precision is allowed only where it cannot change a displayed or reported value beyond its printed precision. The prefix mode remains available and labelled.
- **FR-3** — WHEN several files are selected, processing SHALL use the available CPU cores (leaving the UI responsive) instead of a fixed two-worker limit.
- **FR-4** — WHEN processing is running, a Cancel action SHALL stop the workers, leave the previous dataset (or an empty state) intact, and log what was cancelled; closing the window SHALL not wait for full-file workers.
- **FR-22** — WHEN the operator reads a prefix, the pairs per file SHALL be chosen from 10k, 100k, 1M, 2M, 5M, 10M, 20M and 30M (default 10k). 30M is the largest prefix whose per-file worker result stays below the 4 GiB Windows inter-process message limit even if every pair is accepted. In whole-file mode (FR-2), the pre-run estimate SHALL list every selected file whose estimated pair count exceeds 30M, and processing SHALL be refused with that list and the advice to read a prefix or split the acquisition into several files (Change 3).

### Channel status (first tab, RAWInspector "ADC Status" analogue)

- **FR-5** — WHEN viewing the first tab, it SHALL show one row per mapped SuperModule listing channels that are *not observed*, *low* and *high* relative to that SM's per-channel hit-count median for the same channel type (time/energy), plus totals, with RAWInspector-like row colours and text states. Thresholds SHALL be shown and configurable; states are observational occupancy findings from the coincidence ingest population, never a dead/hot hardware verdict.
- **FR-6** — WHEN selecting a SuperModule row, the tab SHALL show that SM's per-channel hit counts laid out on the mapped channel geometry (time and energy channels distinguished), with flagged channels highlighted, and allow opening the SM in the SuperModule view.
- **FR-7** — WHEN statistics are too low for a channel finding (configurable minimum), the state SHALL read "insufficient events" rather than low/not observed.

### Whole-system minimodule overview

- **FR-8** — WHEN viewing System Overview, each mapped SuperModule SHALL be drawn as its minimodule grid derived from the selected map's channel coordinates (not a hardcoded shape or `mM_disposition`; 4×4 for current IMAS and Cornell maps), placed by the config geometry, with each minimodule tile coloured by a selectable metric: detector-side counts (ingest population or current display cuts) or, when calibrated, photopeak centroid or resolution (spec 001 fit guards; unavailable fits distinct from low values; raw mode marks them unavailable). Inactive or unmapped minimodules SHALL be visibly distinct from zero-count ones.
- **FR-9** — WHEN clicking a minimodule tile, the inspector SHALL show its SM/mM identity and value, and SHALL be able to open that SuperModule in the SuperModule view.

### SuperModule browsing

- **FR-10** — WHEN inspecting SuperModules, the operator SHALL step through them one at a time (previous/next buttons, mouse wheel and keyboard shortcuts) in the SuperModule Explorer (renamed "SuperModule"), which adds a summary panel with occupancy, channel findings and the per-minimodule table next to the energy/DOI/flood plots. The text-only "SuperModule Status" tab is retired once its content is shown there.

### Coincidence views

- **FR-11** — WHEN ingesting, the engine SHALL retain each side's partner SM and mM so pair-level views are possible, without changing spec 001 counts.
- **FR-12** — WHEN viewing coincidences, the inspector SHALL show an SM × SM matrix of accepted pairs (under the current display cuts), and a paired timestamp-difference histogram for a selected SM or SM pair, labelled as observational (geometry and time of flight contribute; not a clock or CTR calibration).
- **FR-13** — WHEN calibration is active, the SuperModule view SHALL offer a per-minimodule photopeak table (centroid, resolution, counts, fit status) using the spec 001 fit guards; raw mode marks it unavailable.

### Cornell slab-assigned flood map and decompression (change proposed 2026-09-25)

- **FR-16** — WHEN loading inputs, the operator SHALL be able to select optional COG-limits and DOI-limits files (tab-separated `(time channel, slab)`, left, right — e.g. `cog_limits_full_system.txt`, `doi_limits_full_system.txt`); their paths SHALL appear in provenance, and changing or removing them SHALL recompute the affected views without rereading LDAT. Keys missing from a file SHALL leave that side's decompressed value unavailable and be counted, never replaced by a fallback value.
- **FR-17** — WHEN viewing a Cornell flood map (SuperModule view and System Overview flood mode), the operator SHALL switch between the COG/RTP view (spec 001 centroid) and a slab-assigned view: X from the assigned slab, Y from the COG Y decompressed with the COG limits into its minimodule row (`(y − left)·25.6/(right − left)` plus the row offset used by `scripts_cornell/cornell_floodmaps.py`). The slab view SHALL be unavailable for IMAS and when no COG-limits file is loaded; decompressed Y outside its row SHALL be clipped to the row. It SHALL state how many sides were excluded (missing key, out of range, unresolved slab).
- **FR-18** — WHEN a DOI-limits file is loaded, the DOI view and DOI cut SHALL offer a decompressed DOI in mm-equivalent crystal depth (0–20 mm, `(doi − right)·20/(left − right)` as in `scripts_cornell/cornell_listmode_cog_fixed_position.py`), labelled with its limits file as a linear light-sharing mapping, not an independently validated depth; values outside 0–20 mm SHALL be unavailable and counted. The DOI ratio remains the default.

### Cornell slab assignment option (owner-selected 2026-09-25)

- **FR-19** — WHEN the operator selects "recover non-adjacent" slab assignment, a Cornell side whose two strongest time channels are not adjacent SHALL be resolved from the strongest channel p and its fired adjacent neighbours instead of being rejected:
  - only p−1 fired → slab 2p;
  - only p+1 fired → slab 2p+1;
  - both fired → the stronger neighbour's side;
  - an exact tie, or neither fired → the legacy one-channel rule for p (random, or the edge rule at positions 0/7).

  The legacy rule (reject) SHALL remain the default. The active rule SHALL appear in provenance and view labels, with the number of recovered sides. Energy calibration and limits files built under the legacy rule SHALL be flagged as possibly inconsistent for recovered sides.

### Calibration factor provenance (owner request, approved 2026-09-25)

- **FR-20** — WHEN a Cornell energy calibration has a `_status.txt` sidecar (written by `scripts_cornell/cornell_slab_en_cal.py`), the inspector SHALL read each slab's factor origin (fitted, fitted-with-check, borrowed from slab 1/14, estimated from neighbours, estimated from minimodule median). It SHALL count the detector sides using each kind per SuperModule and in total, and show those counts in the energy view, Channel Status and report provenance. Unpopulated minimodules SHALL be shown as unpopulated, not as unavailable keV. Borrowed and estimated slabs SHALL be excludable from photopeak and uniformity measurements. Without a sidecar the origin SHALL be "unknown", never assumed fitted.

### Cornell system orientation (owner request, approved 2026-09-28)

- **FR-21** — WHEN viewing System Overview on a Cornell system, SuperModules and minimodule tiles SHALL be laid out as the unrolled cylinder of the config geometry used by `scripts_cornell/cornell_lor_display.py` and `cornell_skew_cal_bigdata.py` (`sm_map_gen`, `local_to_global`):
  - rows follow axial Z from the config's `ring_z`, with +Z at the top;
  - columns follow the cassette angle θ = atan2(Y, X) of `ring_yx`, increasing to the right;
  - inside each SM, local X (axial, Z = sm_z − (x − 48)) runs downward and local Y (tangential, towards increasing θ) runs to the right;
  - the axis labels give Z and θ, and the flood-map mode thumbnails use the same orientation;
  - the flood-map mode thumbnails of every system share one absolute colour scale (0 to the system's peak bin) with one colour bar, so relative intensity across SMs shows the source position (owner, 2026-09-28). The metric modes already use one system-wide scale.

  The SuperModule tab and PDF flood maps stay in local X/Y. The IMAS layout is unchanged. The placement uses only the order of the `ring_z` values and the cassette angles of `ring_yx` in `cornell_full_system.yaml`, which the owner confirmed as the real Cornell geometry on 2026-09-28. `ring_r` is not used: it is the distance used for the skew-calibration phantom (the diameter of the activity there), not a cassette radius.

### SuperModule hardware address (owner request, approved 2026-09-29)

- **FR-23** — WHEN a minimodule tile is clicked in System Overview, and in the Full System/SuperModule report's SuperModule summary table, the inspector SHALL show that SuperModule's hardware address from the selected map's `mod_feb_map` entry `[PortID, SlaveID, FEB/D port]`: the FEB/D DAQ port ID, MASTER for slave ID 0 or SLAVE for slave ID 1 (any other value as "slave ID n"), and the FEB/D port. A SuperModule without a valid entry SHALL show the address as unavailable, never a default.

### Common

- **FR-14** — WHEN any new view or report measurement is shown, it SHALL state its population (ingest vs display-cut), units, calibration and prefix/whole-file scope; the PDF reports SHALL include the new channel findings and minimodule summary.
- **FR-15** — WHEN validating, synthetic IMAS/Cornell fixtures with known hot, low, missing and inactive channels/minimodules, known pair matrices and known Δt SHALL pin every new measurement; the real Cornell six-file set SHALL verify FR-1–FR-4 on whole files; the owner reviews the GUI.

## Out of scope

- Singles, per-SiPM maps, DOI depth beyond the owner-supplied limits-file mapping (FR-18), CMB charge-share/gain metrics, acquisition hardware.
- Clinical PASS/FAIL thresholds; channel states are observational.

## Clarifications (owner, 2026-09-25)

- **Q1** — Whole files, all six Cornell files, ≤ 4 GB, < 5 min (FR-1, FR-2).
- **Q2** — Channel status uses per-channel hit counts only; no new per-channel energy data (FR-5–FR-7).
- **Q3** — Minimodule tiles offer counts (ingest/selected) and calibrated photopeak centroid/resolution (FR-8).
- **Q4** — Stepping merges into the Explorer; the text-only SuperModule Status tab is retired (FR-10).

## Change 1 (owner request, approved 2026-09-25)

- **FR-2 revised** — The fixed 4 GB budget is replaced by a pre-processing memory estimate and warning (owner: "be more flexible… estimate the total GB usage to advertise the user").
- **FR-16–FR-18 added** — Slab-assigned flood map with COG/DOI decompression from the `*_limits_full_system.txt` files. The repo-root copies are gitignored and differ from `C:\Users\dsanchez\Desktop\data\Cornell\files_for_listmode\`; the owner selects the file.
- **Out-of-range values** — decompressed COG Y is clipped to its minimodule row (as `cornell_floodmaps.py`); decompressed DOI outside 0–20 mm is unavailable for that side and counted (as list mode).
- **Slab convention** — the scalar (`src/utils.py`) and vectorized (`src/utils_fixed.py`) `get_slab_cornell` disagree for two-adjacent-time-channel events. The owner chose a separate bug fix, with its own regression check, before T1; FR-17's slab X follows the convention the owner confirms there.
- **Alternative slab algorithms** — the owner selected non-adjacent recovery (FR-19); weighted random and the classifier study are not in scope.
- **FR-20 added** — estimated/borrowed calibration factors must stay distinguishable from fitted ones in the inspector (owner: "is there anything we can do with the keV unavailable?").

## Change 2 (owner review, approved 2026-09-28)

- **FR-21 added** — owner, reviewing T9: "I would like to have the same orientation as in the real system". Checked against `cornell_lor_display.py` / `cornell_skew_cal_bigdata.py`: the T9 overview put Z vertically between SMs but local X (axial) horizontally inside them. The owner chose Z vertical / θ horizontal and kept the SuperModule-tab flood in local X/Y.
- **Real IMAS waived** (owner, 2026-09-28: no IMAS data available). IMAS stays covered by synthetic fixtures on the real IMAS maps only.

## Change 3 (owner request, approved 2026-09-28)

- **FR-22 added** — reading the 25 GB Cornell `Source_200microCi_60s_coincCompact.ldat` whole failed with `[WinError 87] El parámetro no es correcto`. Measured: the file reads in one process (99.3 M pairs, 60.1 M accepted, 22.2 GB peak private memory), but its 6.61 GB result cannot be returned from a worker; a spawn-pool transfer fails above 4 GiB (3.9 GiB OK, 4.1 GiB fails). The owner declined workers writing results to disk (extra complexity; large acquisitions should be several files) and asked for prefix choices up to that limit instead of a free number with a 1 M cap. After checking the dropdown in the GUI (OK), the owner asked to refuse whole files above 30M pairs as well.

## Change 4 (owner request, approved 2026-09-29)

- **FR-23 added** — owner: "in the Whole system setup, can we somewhere note the portID information per Supermodule? the information in the map yaml file [FEBD_DAQ_PORTID, MASTER/SLAVE, FEBD_PORTID]". The owner chose the System Overview tile info line and the report summary table (not the grid labels or the SuperModule tab), and the labels MASTER for 0 and SLAVE for 1.
