# Plan 011 — IMAS skew calibration

Status: draft plan for approved [spec 011](spec.md), 2026-10-26. Constitution: [AGENTS.md](../../AGENTS.md). [Ordered tasks](tasks.md).

## Inputs seen at planning (2026-10-26)

- **Skew splits:** `IMAS/20251119_Skew_400s_0_coinCompact_00000008/9/10.ldat` (16.54, 16.49, 16.60 GB). A bounded read of the first 20,000 records of each: every channel in `maps/imas2DAQ_map.yaml`, all 120 SMs present, ~9.9 hits per side, first timestamps 13.34 s apart. LDAT Inspector on 5M-record prefixes: ~79 % pass `min_ch`.
- **Legacy production LUT:** `IMAS/skew_IMAS.txt`, SHA-256 `930cd3fc5630e216…`, one `ch skew` line for each of the 15,360 TIME channels; 796 hold `9999` (no delay); finite values −3416…+4984 ps, per-SM medians −2312…+318 ps. The sign agrees with `t + skew` (SMs whose paired local-minus-partner time is about +2 ns in the Inspector get about −1.8 ns).
- **22Na:** `IMAS/20261019_Na22_1cm_cFOV_test_300s_coincCompact_00000038.ldat` (1.03 GB), still locked by the transfer at planning time.
- **Energy calibration format:** `KevConverter(file_type="mu")` reads a tab-separated table with columns `ID` and `mu` (511 keV peak position in a.u.); a non-positive or non-finite `mu` gives NaN.

## Architecture

Three layers. Cornell modules keep their import paths; IMAS gets its own package and CLI.

1. **`src/timing/` — scanner-neutral core** (new). Code moved out of `src/cornell/skew/` without change in behavior:
   - `kernels.py`: numba side-selection building blocks (`_keep`, `_min_ch`, `_max_module`, `_time_hits`, `_centroid`), the record codes, `FeatureBatch`, `Partition` and `holdout_codes`.
   - `shell.py`: `AnnularSource`, `shell_intervals`, the numba `_shell` and the LOR stage (shell crossings, 3D opposition angle, codes) on precomputed endpoints.
   - `cache.py`: feature cache and observation sets, parameterized by a scanner adapter instead of a Cornell lookup.
   - `lut.py`: canonical LUT, native table, conversion manifests, plus the new legacy-file export/import and the timestamp correction function (FR-9).
   - `ctr.py` core: `converge_point`, histogram fits, level crossings, uncertainty combination, accumulators.
   - `plots.py`: per-ASIC diagnostics.
   - `pairs.py` and `solver.py` move only if FR-7 keeps spec 008's estimator (T8).
2. **Scanner adapters.** An adapter supplies what differs between scanners:
   - `select_batch(batch, file_index, seed) -> FeatureBatch` (pre-geometry codes and features) and its rejection labels;
   - `endpoints(features, hypothesis) -> (codes, endpoints_mm)`;
   - energy-calibration keys and reading;
   - fingerprint sources (files whose bytes enter cache manifests);
   - channel classes (operational, unobserved) for coverage.
   - **Cornell adapter** stays in `src/cornell/skew/` (`ingest.py`, `geometry.py`): slab coin flip, per-(TIME channel, slab) `.encal`, faceted cassette frames. Its modules re-export what moved, so `python -m src.cornell.skew`, Manager listmode and every Cornell test keep their imports.
   - **IMAS adapter** in new `src/imas/skew/`.
3. **`src/imas/skew/` — IMAS front end and commands** (new), `python -m src.imas.skew COMMAND`:
   - `inventory`: bounded scan of declared splits (records, hits per TIME channel and SM, unobserved channels).
   - `encal`: per-TIME-channel photopeak calibration from the dedicated splits (FR-4).
   - `pairstats`: the FR-7 measurement.
   - `run JOB.json`: pooled job (FR-6, FR-8, FR-9, FR-12).
   - `export`: canonical LUT to native table and legacy file.
   - `ctr CTR.json`: point-source CTR against the legacy LUT (FR-10).

### Alternatives rejected

- **IMAS fork of `src/cornell/skew/`:** fastest, but two copies of the cache, LUT and CTR code would drift; rejected by the owner.
- **Moving everything at once, pairs and solver included:** FR-7 may replace the estimator, so moving them first could be wasted work.
- **Generalizing `job.py`:** the Cornell job is bound to Manager conversion ledgers, `.encal` reuse and spec 008/009 options (native second pass, walk). The IMAS job reuses the core but has its own job schema.

## IMAS adapter details

### Lookup (`src/imas/skew/setup.py`)

Built once from `configs/imas_2DAQ.yaml`, `maps/imas2DAQ_map.yaml` and the energy calibration:
- dense per-channel arrays: kind (TIME/ENERGY), SM, minimodule, local x/y (mm), slab index within the minimodule (0–7 from the TIME channel's map position);
- `mu` per TIME channel and its state (valid, missing, invalid);
- `min_ch`, `en_min_ch`, the job's keV window (408–613 keV), coincidence window.
- SM numbering check: SM = azimuthal index + 24 × ring for every mapped channel, against `ring_yx` (24 keys) and `ring_z` (5 entries); a mismatch is a `GeometryError`.

### Side selection (`select.py`, numba)

Per record, reusing the core kernels, in this order (first failing cut counted):
1. unmapped channel;
2. both sides pass `min_ch` (ENERGY hits ≥ `min_ch`, at least one TIME hit);
3. highest-ENERGY-sum minimodule per side (first seen on ties, flagged);
4. that minimodule passes `min_ch`;
5. `|t_top1 − t_top2| <` coincidence window, `t_top` from the highest-charge TIME hit;
6. TIME channel has a valid `mu` (else "no calibration" / "invalid calibration");
7. both sides' minimodule ENERGY sum in keV inside 408–613.

No coin flip: the slab is the top TIME channel. Features keep spec 008's columns (`slab_*` = slab index 0–7) plus the raw ENERGY sum, so the cache schema is shared.

**Hit energy floor:** hits below `en_min_ch` (0.2 a.u. in the config) are dropped before selection, as the compact reader does in the Inspector. The legacy skew script reads with `read_binary_file`'s default (energy ≥ 0); the reference mode keeps that, and the difference is reported (FR-3).

### Positions and endpoints (`geometry.py`)

- **Local coordinates:** transaxial `u` = local x of the top TIME channel (slab centre); axial `v` = squared-energy-weighted centroid of the minimodule's ENERGY-channel y, as `_centroid` does. Both in the map's mm frame, centred on the SM (local origin 48 mm, from 32 positions at 3 mm).
- **SM frame:** flat facet. Centre azimuth `φ = atan2(Y, X)` from `ring_yx` (stored as `[Y, X]`), axial centre `ring_z[ring]`, entrance-face apothem 410 mm (`ring_r`, bore 82 cm), endpoint at apothem + 7 mm.
- **Global endpoint:** `r·(cos φ, sin φ, 0) + s_t·(u − 48)·(−sin φ, cos φ, 0) + (0, 0, z_ring + s_a·(v − 48))`, with transverse sign `s_t` and axial sign `s_a` in {−1, +1}.
- **Sign selection (FR-2):** nominal `s_t = +1`, `s_a = −1` (the legacy transform). The data choice is made after the estimator exists, with spec 008 T6's method: free branch separation per pair, compared across the four sign combinations, the expected separation being the TOF difference between the two shell crossings. Before that, FR-7 statistics are measured with the nominal signs; acceptance counts change little with the signs and the difference is reported.
- **Source:** `AnnularSource(155, 165, −350, 350, (0, 0, 0), (0, 0, 1))` as `IMAS_SKEW_BAR`, recorded in every manifest as nominal.
- The LOR stage (shell segments, 3D angle ≥ 165°, codes) is the core `shell.py` stage shared with Cornell.

### Energy calibration (`encal.py`, FR-4)

- Input: the job's dedicated energy-calibration splits (one of the 3 local splits during development).
- Per side passing selection steps 1–5, the minimodule ENERGY sum is histogrammed per top TIME channel: 150 bins over 0–220 a.u. as in `src/cornell/calibration.py`, with the same anchor (highest 5-bin-smoothed bin), the same Gaussian plus constant or linear background fit (`fit_peak_background`) and the same ≥ 200-entry rule.
- Not reused from Cornell: slab keys, the two-TIME-channel side rule, outer-slab borrowing and neighbour estimates. A TIME channel without a fit gets status `no peak` or `few entries`, no factor, and its events are rejected.
- Output: `ID\tmu` (KevConverter `mu`), a status file and a JSON manifest (input splits with SHA-256, selection, histogram, fit parameters per channel, counts).
- Check against the legacy calibration: not available (none supplied); the per-SM 511 keV peak position spread is reported instead.

## Core extraction (FR-1)

- Moves are mechanical (same source text, new module); the old modules import from the new ones. No numerical change.
- `cache.py` takes an adapter object with `select_batch`, `fingerprint_sources`, `rejection_labels` instead of `(config_path, encal_path)`; the Cornell call sites wrap their current lookup in a Cornell adapter. The cache manifest keeps its fields, so existing Cornell caches stay reusable: the fingerprint must not change. If it would, `SCHEMA_VERSION` is not bumped silently: the task stops and reports.
- Cornell's geometry stage splits into the Cornell endpoint kernel and the core LOR stage. The arithmetic and order are unchanged, so results are bit-identical; the check compares arrays exactly.
- **Checks (FR-1):** the full Cornell timing test set (`tests/test_cornell_timing_*.py`, `test_cornell_walk.py`, `test_manager_timing_luts.py`, `test_manager_listmode.py`), unchanged; and an exact before/after comparison on run7 (`PETSYS_DATA_DIR`): feature-cache blocks, observation set, pair fits and LUT from a bounded run7 job, by SHA-256.

## Data shapes

- **Feature row (shared):** spec 008 `FEATURE_DTYPE` columns: record, TIME channels, SM/minimodule, slab, dt (ps, int64), module tie, keV, local x/y per side; IMAS adds nothing new to the row except the raw ENERGY sum already in the walk columns (`e_raw_*`).
- **Observation set:** spec 008's canonical rows (lower TIME channel first, shell TOF intervals), partitioned by channel pair; pair counts uint64.
- **Pair statistics (FR-7):** JSON with observations, distinct pairs, histogram of observations per pair (log bins), branch-resolved fraction at spec 008's thresholds, partners per channel, components, coverage per SM and ring, and a projection per TB from the bytes read.
- **Canonical LUT:** spec 008's schema `cornell-time-delay-lut/1` is kept: it names the format, not the scanner, and renaming it would invalidate existing Cornell LUTs. The manifest gains `scanner: "imas"` and the map fingerprint. LUT readers must refuse a LUT whose map differs from the run's; T10 checks whether the Cornell readers already do and adds the check where they do not.
- **Legacy file:** `ch skew` per line, `skew = −b` with 3 decimals; channels without `apply` are omitted. The importer reads the production file, turns `9999` into "no delay" and anything else into `b = −skew`, recording the SHA-256.
- **Timestamp correction:** `correct_timestamps(channel, timestamp, lut) -> (corrected, applied_mask)` in `src/timing/lut.py`, vectorized over an `LdatBatch` or the two sides of features; integer ps rounding is explicit.

## Scale and memory (FR-6)

- **Reading:** `src.ldat_stream.LdatStream` batches; one worker per file, at most `workers` files open. The HDD read rate is measured first (T3) and sets the worker count, since one USB disk does not benefit from many parallel readers.
- **Cache location:** a local SSD directory named in the job; `_disk_preflight` estimates the cache from the bytes to read (Cornell's ratio 0.059 feature bytes per LDAT byte, re-measured on IMAS in T6).
- **Bounds:** per-file feature blocks, observation partitions with a bounded write buffer, no per-event Python objects. Peak memory per stage is reported.

## FR → component map

| Requirement | Component | Check |
| --- | --- | --- |
| FR-1 | `src/timing/*`, Cornell re-exports, adapter seam in `cache.py` and the geometry stage | Cornell timing suite unchanged; run7 before/after SHA-256 equality (`real_data`) |
| FR-2 | `src/imas/skew/geometry.py`, sign selection | synthetic endpoint fixtures (known SM/slab/centroid → mm); sign study on real data |
| FR-3 | `src/imas/skew/select.py`, reference mode | synthetic records per cut and boundary; legacy parity on a prefix (operator, untracked harness) |
| FR-4 | `src/imas/skew/encal.py` | synthetic peaks recover `mu`; status for empty and few-entry channels; real dedicated split |
| FR-5 | `IMAS_SKEW_BAR`, core `shell.py` | synthetic LORs with known crossings |
| FR-6 | `inventory`, job, core cache | resume and refusal fixtures; bounded-memory test; real 3-split job |
| FR-7 | `pairstats` | synthetic counts fixture; real 3-split report |
| FR-8 | core solver or the FR-7 alternative | injected delays on IMAS-geometry synthetic data |
| FR-9 | core `lut.py` (canonical, native, legacy, correction) | consumer round trips; legacy import of `skew_IMAS.txt` |
| FR-10 | `src/imas/skew/ctr.py` over core `ctr.py` | synthetic point source; real 22Na with legacy, uncorrected and new LUT |
| FR-11 | core `plots.py`, IMAS SM overview | image files written, headless |
| FR-12 | job on the HDD runs | recorded job report |

## Risks

- **Pair statistics (FR-7):** the rough estimate is ~10 observations per pair on 3 splits and ~100–200 on 1 TB, below what spec 008's two-branch pair fits need per branch. The measurement may move the plan to another estimator; tasks after T5 are provisional until then.
- **Bar coverage:** 3 splits ≈ 40 s ≈ ⅓ turn; LORs through the bar only occupy part of each channel's fan. The 1 TB pool should cover many turns; per-run coverage is reported.
- **Nominal geometry:** no NN positions, no gap model between minimodules; axial position from the energy centroid compresses near the edges. Its effect on the delays is bounded in validation by the sign study and held-out residuals, not removed.
- **Large initial offsets:** whole SMs are off by up to ±2.5 ns (Inspector), plus up to ±2.7 ns of TOF; spec 008's ±15 ns window covers this, but a narrower energy-calibration or pair window would not.
- **22Na split 38 only** may be short of statistics per SM; more 22Na splits may be needed (owner transferring).
