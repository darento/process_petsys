# Plan — Cornell streaming timing calibration

Status: approved plan for [spec 008](spec.md), shipped 2026-10-19 (T11); updated 2026-10-13. Owner authorizes isolated skew implementation; spec 005 is paused awaiting operator verification. Run7 inputs pass bounded preparation checks; full data/physical/converter qualification pending. [Ordered tasks](tasks.md). [Paper research](../../docs/research/cornell-skew-calibration.md). [Local probe evidence](../../docs/research/cornell-skew-probes.md). [PETsys native-offset contract](../../docs/research/petsys-time-offset-contract.md).

## Recommendation

T1b1 first scans the successful converter ledger's named LDAT outputs with bounded buffers, reusing existing fastread kernels read-only. It freezes file identities and 11/2/3 train/validation/test membership before fitting; time intervals use measured bounds and a declared 1 microsecond guard at each file edge. Qualification requires full scans, nonempty finite/mapped records and non-overlapping guarded intervals. Prefix reports cannot qualify a holdout. Raw channel presence and down-region observations are reported separately from calibrated acceptance. T1b2's calibrated cuts and complete-pipeline/storage benchmarks depend on later ingest and remain pending. No Manager/shared-reader behavior changes in this task.

Replace the original script's ingestion with the compiled approach already proven in `src/ldat_inspector/fastread.py`, exposed as a streaming core. Keep one entry point. Combine a two-peak shell initializer with an uncertainty-weighted sparse channel-offset solve. Add finite-shell likelihood refinement only if it improves held-out results. Validate CTR independently of calibration and source-branch selection.

The stationary thin shell is a particularly useful calibration source: along a suitable LOR it gives two narrow source-position peaks. It is not a uniformly filled cylindrical volume, and an individual coincidence's emitting branch is unknown.

Owner-confirmed selection policy: retain the photopeak energy window on both selected detector minimodules and the legacy opposition cut for matched reference comparisons. Current YAML requires strictly `357 < E < 665` keV after slab-specific calibration. The legacy angle is computed in 3D between the two scanner-center-to-reconstructed-endpoint vectors and accepts angles >=165 degrees (within 15 degrees of opposition); it is neither a transverse-only angle nor a nominal SM-center pair cut. Report energy and angle rejections separately. A wider or transverse-only angular policy, or a different energy window, is an explicit later experiment assessed for offset bias/uncertainty and graph connectivity on frozen validation data; preserve the baseline while improving throughput.

T2a implements nominal geometry independently of ingestion: explicit `time-axial` versus `time-transverse` local axes, caller-declared local origin and axial sign, selected-map SM IDs and configured centers. Original and optimized scripts both use 48 mm origins; the selected map's coordinate midpoint is 51.61 mm. Owner's suggestion to inspect the optimized script establishes an alternate transform to compare, not a measured orientation/origin. No origin/axis policy is silently promoted to qualified geometry. Finite-shell intervals are continuous source segments clipped by radial and axial boundaries, with floating geometric TOF; angle selection remains a separate policy. Source-fingerprinted legacy probes are frozen for strict-xfail regression evidence without importing local reference scripts in tests. T2b retains physical and deployed-PETsys checks.

T2a verified on 2026-10-13: scalar `timing_geometry.py` and exact-difference `timing_convention.py`, 56 focused/provenance checks passed and four strict legacy-defect xfails retained. [Audit contract](geometry-contract.json) freezes configured centers, axis/origin variants, provisional finite-shell intervals and local PETsys source fingerprints. It explicitly selects no production physical axis/origin policy and claims no native conversion qualification. T2b remains open; geometry compilation belongs to T4.

## First real dataset: run7, 16 actual LDAT outputs

Owner selected `norm_10082026_r7_acq_2026-10-08_1630/attempt-1/norm_10082026_r7.rawf` from eight nominal 900 s F18 annulus/normalization acquisitions on 2026-10-08. Listed RAWF size is `185G`. Local inventory now contains 16 run7 LDAT files totaling 222,836,867,310 bytes; names/sizes match the successful `run.json` conversion ledger. Requested split setting was 15; the Manager builder uses `--splitTime 60.1`. Use every actual ledger output rather than assuming 15 filenames. Prefix timestamps confirm approximately 60.1 s spacing for most starts, with a partial first interval; full interval ends/900 s coverage remain unverified. Two co-located run6 files are outside this run7 dataset. Input skews are unapplied; matching `.encal` and status sidecar are supplied.

Local transfer target is `C:\Users\dsanchez\Desktop\data\Cornell\full_system\skew`. The [input-preflight snapshot](input-preflight.json) preserves the earlier preparation-only state and input fingerprints. It is not a completed-file or full-dataset validation. Initial T1a adds an isolated bounded decoder benchmark using existing fastread kernels without editing their implementation or Manager code. It compares raw decode plus aligned-array packing on identical bytes, with no analysis cuts; feature/geometry/solver performance is deferred. T1b retains full-run coverage and Cornell storage/worker qualification. No geometry qualification is needed to measure raw decoding.

Bounded preparation checks on 2026-10-13 read the first 10,000 physical coincidence records of each run7 file: 160,000 records total, passing structure, finite hit-energy and selected-map channel checks. File sizes remained stable during reads and match the ledger. This verifies readable prefixes, not full-file integrity/counts, calibrated acceptance, channel adequacy or CTR. [Detailed evidence](../../docs/research/cornell-run7-input-checks.json).

The supplied `encal_files/norm_10082026_r7_coincCompact_resolved.encal` has 7,680 unique mapped slab keys and a valid matching status sidecar. The YAML marks 6,400 nominally populated keys and 1,280 unpopulated. Of the former, 6,336 have valid factors; 64 zero/no-fit keys are SM14 minimodules 0,1,4,5. Owner confirms these four minimodules were down due to a board short circuit. Carry this run7-specific availability mask in provenance/coverage reporting: 3,168 operational TIME channels / 6,336 operational slab keys, all with valid factors. Retain unavailable LUT status for the 32 down TIME channels; preserve global map/configuration. Track borrowed/estimated/check-flagged factor origins and assess their selection sensitivity later. The current calibration script documents `_resolved` as fitting from resolved two-TIME-channel sides under its legacy slab function; actual generating command/cuts remain to record.

The supplied conversion INI is `Desktop/data/Cornell/20260304_final_system_newFEBDConf/config.ini`, now fingerprinted in that snapshot. Native channel-offset/energy table entries are commented out; TDC/QDC tables are configured. Its software coincidence window is 1 clock unit and group window 20 clock units under inspected upstream semantics. Run7 RAWF header gives 200 MHz, so the coincidence gate is `abs(delta_t) <= 5 ns` (10 ns full width) and group window 100 ns; Cornell's installed sources match upstream `21dc58b..3d13196` (T2b record). Do not retain 15 ns as a verified conversion gate or confuse it with the legacy offline cut. The local INI copy lacks its referenced TSVs; these are provenance/native-reconversion inputs, not a reason to reprocess fine-time calibration in existing LDAT.

This is a substantial development dataset and may also suffice for full calibration. There is no requirement to use all eight acquisitions. The decision is driven by per-channel information, two-peak fit quality, connected channel coverage and uncertainty, not raw bytes or activity alone. A later/lower-activity acquisition may have different random/background and rate effects; do not presume that decay improves measured quality or that its correction table transfers unchanged to earlier/higher-rate data.

Qualification sequence:

1. Preserve the converter command/config, acquisition timing window, calibration provenance, timestamp correction state, split order/time ranges and selected map. Inventory split sizes and validate record boundaries. Count compact **coincidence records**, not singles, and do not infer counts from RAWF size.
2. Use a bounded prefix of one split for decoder/selection debugging and matched performance comparison. Scan headers/counts across all splits early to detect incomplete transfer and temporal/channel coverage; do not extrapolate a first-split result to the entire run. The existing hidden 500,000-record/file limit would retain at most 8 million processed records across these 16 files and is unsuitable as an undisclosed full-run result.
3. Collect per-stage rejection counts and accepted counts per active TIME channel, distinct usable partners, shell geometry groups/branch quality and connected components. Distinguish low statistics from missing sensors/map/calibration. Histogram views/branches and shared-hit records must not inflate effective independent statistics.
4. If split boundaries permit, reserve 11 splits for training, 2 for model/cut validation and 3 for final held-out evaluation, adjusting for the first/last partial intervals. Distribute the groups across early/middle/late portions of the run; record membership before fitting. Use non-overlapping time blocks within files if splitting is not chronological. Do not mix records from the same coincidence across groups. Repeated prefixes/overlapping samples are not independent validation.
5. Fit progressively larger training sets, for example 1/3/6/11 splits chosen to cover the run. Compare channel coverage, graph structure, uncertainty and offsets after fixing a common gauge. Bootstrap independent time blocks for stability. Partial fits use the same reference/gauge; disconnected components do not inherit an assumed common origin. Stop adding data only when the chosen precision/coverage criteria and held-out residual checks pass.
6. Report sufficient / sufficient for named channels only / insufficient, with reasons per channel. If insufficient, determine whether more coincidences, broader valid LOR coverage, missing calibration or geometry correction is needed. More files cannot repair disconnected constraints caused by a restrictive cut. Consider run 8 as a later independent annulus stability check, if available; request earlier runs only if the coverage/precision/rate comparison requires them.

For intuition only, an ideal Gaussian location estimate has standard error `sigma/sqrt(N)`. A hypothetical 500 ps FWHM timing kernel has sigma about 212 ps, so roughly 450 independent useful observations give 10 ps **one-standard-error** precision for a simple location estimate. This is not a universal channel/pair event threshold: unequal shell branches, nuisance geometry/background, shared hits and global graph conditioning change the information. Real sufficiency uses fitted/bootstrap channel uncertainty. A 95% interval has a larger width than one standard error.

Held-out annulus data can establish offset stability and a model-dependent timing-kernel width. It does not replace independent known-position CTR validation. Existing January two-Na-source fixtures use a different map/provenance and cannot automatically serve as the October acquisition's calibration or identical-condition CTR reference.

## Confirmed nominal geometry and intended output

The selected YAML describes 10 cassette XY centers through `ring_yx`, at nominal detector radius 160 mm, replicated at z = -102, 0 and 102 mm. Its Cornell SM identity is cassette-first (`SM = 3*cassette + z_index`); preserve the selected map rather than extrapolating this numbering to other scanners. Owner selects these SM centers as the nominal reference and confirms 20 mm radial crystal thickness on 2026-10-13. Opposite XY entries 0/5 are cassette indices; same-z hardware SMIDs are 0/15, 1/16 and 2/17. Construct global positions from the configured centers plus verified local transforms, checking consistency with `ring_r`/angles. Do not shift the reference by an assumed 10 mm or convert light sharing into calibrated depth. Phantom drawing gives fluid radii 137.5/142.5 mm (middle 140 mm) and 300 mm length; owner centers it at YAML z=0 (±150 mm). Owner confirms slab index axial (original time-axial transform). A faceted two-tiles-per-SM sketch candidate (entrance apothem 155 mm, declared interaction depth) is kept beside the planar YAML candidate until data/evidence selects one. Effective interaction-plane uncertainty and source alignment remain precision qualifications.

Deliver first a canonical LUT: global TIME-channel ID, positive subtractive delay `b` in ps, uncertainty, status, gauge and provenance. Invalid/unobserved channels stay explicit. Native PETsys adaptation can follow once this LUT is qualified. A known-position point source is an additional CTR reference; its absence does not prevent annulus calibration or delivery of a qualified LUT.

Upstream PETsys already provides `time_offset_calibration_table`: five tab-separated fields for port, slave, chip, channel and offset. Its hit processor subtracts the per-channel offset in ps when TDC/timing-offset calibration is enabled, before hit grouping and coincidence formation. Confirm the installed Cornell version and channel encoding before export; see the [pinned source evidence](../../docs/research/petsys-time-offset-contract.md). Fine-time TDC coefficients are a separate calibration and must be preserved.

The later native adapter needs a bounded RAWF-to-LDAT roundtrip: positive delay/sign, channel identity, activation, timestamp quantization and correction applied once. Native offsets can change grouping and coincidence membership. Validate differences on matched retained events and report population changes separately. A TIME-channel solve does not determine ENERGY-channel offsets; preserve existing values or choose an explicit, justified policy rather than copying TIME offsets. Store correction state in the manifest to prevent offline subtraction after a corrected conversion.

## Findings that determine priority

| Current code | Consequence | Planned change |
|---|---|---|
| Both scripts call `read_binary_file`; Python unpacking, lists, filters and per-event geometry remain | Histogram storage alone leaves ingestion work unchanged | Compile decoding **and** feature extraction/geometry; measure stage costs |
| Original stops after 500,000 processed records/file; optimized reads full files | Different workloads can mask improvements; original loses statistics | Explicit limit, matched prefixes, full production mode |
| Optimized `local_to_global` swaps axial/transaxial roles compared with original | It is not a physics-equivalent reference | Retire that variant; verify geometry against measured scanner placement |
| Optimized intersection rejection is commented out | Rejected LORs can enter with zero theoretical TOF | Restore a verified geometry/selection contract in the replacement |
| Both read `en_min_ch` without applying it | Printed cut differs from analysis | Preserve legacy mode for parity; compare deliberate cut change separately |
| Original stores two residuals and both channel views: four CSV rows/event | Large text I/O; counts are not unique physical events | One physical observation; canonical pair orientation and explicit branch model |
| Optimized accumulates `int16` bins | Counts can wrap before the late warning | uint64 totals/checked accumulators; bounded sparse partitions |
| Original reopens channel CSVs each iteration and rebuilds spawn pools; ingest has fixed 32 workers | Parsing/startup/serialization and disk contention | One tuned worker pool, numeric cache, small worker results |
| Original fits a peak of mixed candidate residuals | Wrong-branch structures, partner biases and peak switching can distort updates | Fit both physical TOF peaks, then solve pair differences globally |
| Calibration uses `dt + skew_i - skew_j`; `apply_skew` subtracts stored values; pointSource uses `(t1-sk1)-(t2+sk2)` | Export/consumption conventions are inconsistent | Synthetic sign/swap tests through deployed consumer before evaluating CTR |
| Channels freeze after small change in fitted mean | Stable nonzero residual can be called converged; partner changes can invalidate freezing | Global residual/uncertainty checks; assess the complete solution |
| Infinite cylinder and 15-degree opposition cut based on a 3D angle | Axial source limits omitted; oblique coverage/connectivity can be lost | Finite shell, explicit angle policy, graph and acceptance diagnostics |

These are source-inspection findings, not a profile of the Cornell run. The original reads raw LDAT during extraction, then cached CSVs during iterations; the optimized docstring's claim of repeated raw scans does not describe the original accurately.

## 1. Establish an honest speed baseline

On Cornell local disk, select a representative prefix and complete files. Use identical events, maps, calibration, cuts and deterministic slab policy. Measure raw sequential disk throughput; Python decoding; decoding plus features; geometry; cache/histogram accumulation; solver; output; complete-job wall time and concurrent memory. Record actual bytes processed, not whole-file bytes for a limited run. Include first-call compilation/cache load, imports, worker startup and repeated warm measurements. Do not clear operating-system caches as an implementation side effect; state whether a cold-storage measurement is feasible.

Use 1/2/4/8 workers, extending only while total throughput improves within memory and storage limits. The present 32-worker choice is not evidence of optimality. Keep one pool initialized with immutable lookup arrays and limit BLAS/Numba thread multiplication. Parallelize complete files first; a large single LDAT file needs indexed record-aligned ranges, not arbitrary byte splits. Start with sequential chunks for such files and add range indexing only when the measured CPU/storage tradeoff warrants it.

Local planning probe, Windows, synthetic 90,000 variable-length pairs / 990,000 hits / 16.02 MB:

- Python decode/count/checksum median 0.416 s; existing compiled scan/aligned decode 0.0205 s: **20.3x**.
- Inspector reference ingest 3.818 s; warm fast ingest 0.148 s: **25.8x**. Exact tested retained columns, counters and rejection labels; 60,000 accepted pairs.
- Small synthetic file is warm/cache resident; checksums are reduced in Python versus NumPy. No Cornell disk, shell geometry, solver or complete skew-job speedup was measured.
- Existing project history reports 41x Inspector ingest on a Cornell prefix on 2026-09-25. This independently supports feasibility, not a promised skew speedup.

## 2. Create a reusable bounded ingestion core

Tracked components (layout after T10, owner decision 2026-10-14; modules were `src/cornell/timing_*.py` before):

```text
src/ldat_stream.py                validated record boundaries and compiled hit batches
src/cornell/skew/__main__.py      the one command: python -m src.cornell.skew {run,validate,plot,export,conversion,roundtrip}
src/cornell/skew/dataset.py       raw inventory, ledger inputs, frozen holdout (dev command: inventory)
src/cornell/skew/geometry.py      detector transforms and finite-shell intervals/TOF
src/cornell/skew/ingest.py        reference/compiled selection and calibration
src/cornell/skew/cache.py         feature and observation caches
src/cornell/skew/pairs.py         two-branch pair fits
src/cornell/skew/solver.py        graph, gauge, sparse solve, reproducibility floor
src/cornell/skew/lut.py           canonical LUT, offline sign contract, native table, conversion manifest, composition
src/cornell/skew/plots.py         per-ASIC diagnostics
src/cornell/skew/job.py           pooled multi-run job and second pass
src/cornell/skew/validation.py    frozen-LUT test evaluation
src/cornell/skew/roundtrip.py     native conversion roundtrip
src/cornell/skew/benchmark.py     decoder baseline (development)
scripts_cornell/cornell_skew_cal_bigdata.py  thin local launcher (untracked)
```

Extract/delegate shared low-level logic rather than copying private Inspector functions into another script. Initially preserve Inspector's API and results. A direct call to `process_file_fast` is unsuitable for unbounded calibration: it collects accepted chunk arrays and concatenates all file rows. Streaming must yield/reduce each chunk immediately.

Batch shape: raw hit columns plus offsets/counts for each detector; signed int64 timestamps remain integers until **after subtraction**. Dense/remapped lookups describe channel types, SM/minimodule, local positions, slab and calibration validity, constructed from the selected map. Bound lookup allocation for sparse high channel IDs. Build per-SM rotation sine/cosine and origins once; compute transforms and intersections in compiled scalar/array operations rather than constructing SciPy Rotation/temporary arrays per event. Keep geometric TOF floating point, not truncated integer ps.

Fuse the hot selection passes where beneficial: minimum energy-channel count; selected maximum-energy minimodule; strongest TIME hit; slab/ambiguity; calibration; strict energy window; coordinate reconstruction; coincidence window; shell acceptance. Preserve legacy order/semantics during performance parity. Do not silently inherit Inspector's extra DOI validity rejection, different slab recovery or tie-breaking policy. Track deterministic ambiguity and rejection counts. Exclude absent calibration before reporting keV.

Start with the current aligned-copy approach: it is simple and already validated. Prototype direct byte decoding into compact output only if copying remains material after compilation. Do not assume memory mapping bypasses disk I/O.

## 3. Replace CSV spill with bounded reusable numeric data

The first reliable implementation should stream one accepted-pair row to numeric block files: dense channel indices, int64 `dt`, detector geometry or predicted shell intersections/intervals, calibrated energies, ambiguity flags, input/record identity, and features required by the chosen model. No reversed pair duplicate. Canonicalizing pair order must swap endpoints/features and negate `dt` consistently.

Use uncompressed `.npy`/equivalent typed blocks with a JSON manifest initially; benchmark lightweight compression separately. Include complete markers, versions and fingerprints. A rerun may reuse valid blocks but cannot append into old unrelated output. Solver changes can reuse ingest features; geometry/selection changes invalidate derived blocks unless their raw features permit a verified rebuild.

Then reduce to sparse pair/geometry-stratum histograms or pair-fit summaries. Fixed allocated bounds and partitioned merge prevent RAM growth with dataset size/pair diversity; workers spill their own independent blocks and return metadata rather than large pickled dictionaries. Retain **physical** event counts separately from histogram views/latent branch terms. Histograms need uint64 counters or checked overflow; a post-overflow maximum check cannot repair counts.

For likelihoods depending on event geometry, a single histogram of `dt_error` per channel pair is insufficient. Group geometrically similar LORs under a tested TOF error tolerance, or make passes over compact numeric blocks without redecoding LDAT. Time-walk/optical features must survive compression if they will be studied later.

## 4. Estimate offsets from the shell, then solve globally

Fix the sign contract first:

```text
y = t_i - t_j
y = mu(source position, detector endpoints) + b_i - b_j + noise
t_i_corrected = t_i - b_i
y_corrected = y - b_i + b_j
```

Existing solver tables are additive corrections (`a=-b`) according to their update equation. The intended consumer is PETsys RAWF conversion; upstream uses positive subtractive offsets in ps. Verify the installed version before exporting/migrating tables. Unit tests must verify a positive injected delay corrects to zero and an endpoint swap negates the result. The pointSource script's mixed-sign expression is not a trusted CTR oracle until tested/fixed under its own bug id.

For a transverse diametric LOR through a centered 140 mm shell, ideal peak centers are approximately **-934 and +934 ps**, separated by **1868 ps**. A 5 mm wall contributes roughly 33 ps full branch extent at normal incidence; its distribution is not itself Gaussian CTR. Oblique/tangential LORs can have much larger source extent and overlapping branches. Apply the finite axial extent in 3D.

**Preferred initializer:** within channel-pair/geometry groups, fit both measured peaks with independent amplitudes and a common timing shift relative to expected shell templates. For narrow, resolved branches:

```text
d_ij = (q_minus + q_plus)/2 - (mu_minus + mu_plus)/2
d_ij = b_i - b_j
```

This uses the midpoint of fitted peak positions, not a count-weighted average of events. Unequal source activity/attenuation can change peak populations without moving that midpoint under a valid resolved-peak model. Correct the geometric midpoint for measured miscentering/unequal detector radii. Geometry variation within a channel pair must be retained/grouped; do not pretend all its events share two identical TOFs. If peaks merge or a branch is absent, use a validated distribution fit or mark the constraint unavailable.

**Global solve:** build a sparse incidence matrix with +1 for channel i and -1 for j; estimate `b` from weighted pair shifts using sparse least squares. Weights use fitted shift uncertainty, not duplicated counts. Add robust reweighting only with contamination evidence. Fix a reference or zero-mean gauge; assess uncertainty/graph conditioning and loop-closure residuals. Bootstrap across independent time blocks for empirical stability.

Channel count/top-100 partner gates do not establish identifiability. Check graph components explicitly. With only strictly opposite cassettes, separate opposing cassette groups can be disconnected. Broaden valid LOR coverage using the physical shell model when needed; do not claim relative calibration across components from a common arbitrary zero. Do not freeze channels just because their update stopped changing.

**Optional refinement:** match finite-shell expected TOF distributions to observed distributions with a Poisson likelihood or stable correlation/template shift, including a timing kernel and stated background assumptions. Integrate source position over shell segments and sum branch likelihoods; never enter both alternatives as independent events. Fit offsets first, widths later; joint offset/width/background fitting can be unstable. Accept only better held-out fit/stability or independent CTR beyond uncertainty.

## Paper choice and other improvements

| Approach | Suitability for this acquisition |
|---|---|
| [Werner & Karp 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3826954/) | Expected-vs-observed TOF distribution matching from reconstructed clinical activity. Adapt the distribution principle to known shell geometry; clinical reconstruction is unnecessary for the first shell solution. |
| [Perkins et al.](https://doi.org/10.1109/NSSMIC.2005.1596845) | Known moving-source position can provide event-specific flight time. A stationary shell has two possible branches. Full original text was unavailable; details are limited to accessible primary follow-up descriptions. |
| [Li et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4629519/) | Closest direct match: thin radioactive shell, two peaks per LOR and channel-pair differences. Start here. Published scanner results are not Cornell targets. |
| Sparse global solve | Proposed improvement over channel-by-channel averaging/relaxation; exposes connectivity, residual inconsistency and uncertainty. |

After fixed offsets pass, study held-out residuals versus energy, local impact position, light-sharing ratio, angle, channel/slab and acquisition time. Add anchored time-walk/optical-path corrections only for reproducible trends. Light sharing is not depth in mm without calibration. Within-event differences among simultaneously fired TIME channels can add local relative constraints, but optical transit and time walk must be modeled; they do not alone establish offsets between modules. A combined timestamp from multiple calibrated TIME hits may improve CTR, but correlation and optical bias require a separate comparison with the actual downstream timing estimator. Never assume a square-root-of-hit-count improvement.

## 5. Validate timing without an optimistic CTR

Split by acquisition/time blocks before model selection. Train offsets on one set and freeze them for evaluation. Keep energy/timing cuts, populations and weighting identical across variants. Report residual bias by channel/pair plus pooled residual FWHM, tails, fit uncertainty and channel coverage. A mean of pair FWHMs is not the pooled system FWHM.

Prefer an independent, known-position point source or time-position-recorded rotating line for CTR; subtract its physical TOF. Audit the actual CTR consumer and stored correction sign. Shell-only data can assess offsets and a **model-dependent** convolved timing-kernel width on held-out data. The full two-peak shell histogram's FWHM is not CTR. Selecting the nearest shell branch using measured timing truncates the residuals and can artificially narrow the result, even on held-out events; exclude that procedure from independent CTR claims.

Validate finite thickness, miscentering, axial edges, asymmetric peak amplitudes, unknown calibration, overlap/background, drift and disconnected channels with known synthetic truth. Real acquisition coincidence-window selection may already truncate available events; report it rather than claiming calibration recovers discarded coincidences. Offsets cannot remove intrinsic timing jitter.

## Requirements to components

| Requirement | Component/check area |
|---|---|
| FR-1, FR-14 | Stage instrumentation, reproducible performance manifest, Cornell benchmark |
| FR-2 | Geometry loader, transforms, finite-shell templates |
| FR-3 | Solver export adapter and actual downstream consumer |
| FR-4 | Shared compiled LDAT stream and Inspector parity |
| FR-5, FR-6 | Reference/compiled Cornell ingestion and rejection accounting |
| FR-7, FR-8 | Canonical pair records, bounded numeric cache/accumulator |
| FR-9 | Pair/geometry fits and finite-source model |
| FR-10, FR-12 | Sparse solve, graph/gauge, uncertainties, channel-status output |
| FR-11 | Optional template/likelihood refinement, held-out decision |
| FR-13 | Independent CTR pipeline and frozen evaluation manifest |
| FR-18 | `src/cornell/skew/job.py`: job file, per-run inventory/roles/caches, pooled pair fits/solve/LUT, per-run energy-peak and held-out residual reports |
| FR-19 | `src/cornell/skew/plots.py`: per-ASIC decoding, pattern residuals and outlier list, PNG plots and summary; called by the job after the LUT and as `python -m src.cornell.skew plot LUT.csv` |
| FR-15 | `src/cornell/skew/__main__.py` one command; thin local launcher; `tests/test_cornell_timing_cli.py`; retirement checklist |
| FR-16 | Run 7 inventory, selection/coverage report, fixed holdout manifest and uncertainty/convergence assessment; `src/cornell/skew/validation.py`: frozen-LUT test-file pair residuals (once), nested event-fraction training-size study and per-channel sufficiency |
| FR-17 | Canonical channel LUT; `skew/lut.py` `write_native_table`/`read_native_table`/`activated_ini` native adapter, conversion manifest and `compose_lut` (FR-20); `src/cornell/skew/roundtrip.py` matched-event conversion roundtrip |

Implement in the [task order](tasks.md). First deliver baseline/sign/geometry agreement and a fast streaming reader; then change the estimator. Performance and physics changes need separately attributable comparisons. Input files, precise interaction/source geometry and installed converter qualification remain pending. Independent known-position CTR validation is additional to annulus LUT qualification.
