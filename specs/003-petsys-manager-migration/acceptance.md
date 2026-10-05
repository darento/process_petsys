# Acceptance 003 — T19 Cornell operator acceptance record

Spec: [`spec.md`](spec.md) (FR-17 and completion criteria). Task: [`tasks.md`](tasks.md) T19. Deployment: [`docs/petsys_manager.md`](../../docs/petsys_manager.md).

Each line is a pass, a fail or pending, with the evidence that decides it. Pending is never a pass. `Status: shipped` needs every line to pass; retiring `gui_cornell` is a separate owner decision.

## A. Real-data parity (owner workstation) — pass, 2026-10-02

Final run on the final code (uncommitted runtime diff recorded by SHA-256 in the results): **PASS 8/8**, private results `C:\Users\dsanchez\AppData\Local\Temp\process_petsys\petsys-manager-real-20261002T092346Z-2a5a5253\real_baseline.json`. About 45 min on the owner PC.

`python scripts/petsys_manager_reference_check.py --real --manifest scripts/petsys_manager_t19_baseline_jan2026.json` (local, untracked). It runs the reference scripts as in-process oracles on the operator-selected files and compares their results with the migrated code. Random slab draws are seeded identically for both, and the inputs are read only.

Dataset: Cornell full system, 2026-01-19, two Na-22 sources, 300 s, splits 3–8, in fixed and compact coincidence from the same acquisition. `configs/cornell_full_system.yaml`, owner COG/DOI limits and pair/region maps.

| Comparison | Criterion | Result |
|---|---|---|
| Per-slab calibration (P = 1), compact | `.encal` and status byte-identical to the reference-written files | **pass**: 7,680 rows; 5,356 fitted, 114 higher-peak checks, 791 borrowed, 139 estimated, 1,280 without values; 2.47–2.49 M records and 2.09–2.10 M passing events per file |
| Per-slab calibration, fixed | identical to compact | **pass** |
| Position calibration (P = 5), first 400,000 passing events per file | `.encal`/status byte-identical to the reference-function oracle | **pass**: 2,911,163 sides, 27,705 keys sampled, 38,400 rows |
| Position calibration (P = 5), whole files | fixed identical to compact; written for review | **pass**: 23,672 fitted, 216 higher-peak checks, 3,759 borrowed, 4,119 estimated, 220 without a fit, 6,414 without values |
| LM, six fixed files, owner 5-region calibration | records byte-identical to the reference loop; header = supplied metadata | **pass**: 2,267,361 records (376,785–378,560 per file); 4 μ ≤ 0 keys read as no factor and recorded |
| LM, six compact files (FR-22, T21) | records byte-identical to the fixed-input LM | **pass**: 2,267,361 records, hit limit 16; `real_baseline.json` of run `…20261002T123710Z-2a8af9d2` |
| QC, six compact files, plots + slabs | counts, occupancy, histograms, fits and floods equal to the reference | **pass**: 6,000,006 accepted pairs (1,000,001 per file, limit reached); 6,690 fitted, 74 sparse, 12 failed fits (fallback means within 1e-15 relative); 30 flood SuperModules |

Findings from these files, both decided by the owner on 2026-10-02 and added to FR-12:

- **Failed fits in the owner's January 5-region calibration.** It has 4 rows with μ ≤ 0: `(4360, 2, 0)` −2878.171, `(132067, 0, 2)` −114.864, `(528867, 0, 1)` −213.217 and `(656640, 0, 0)` −17.952. The reference LM uses only μ > 0, so the manager now reads these rows as no factor. Their pairs are rejected and counted (missing calibration, 17,596 in total), and the keys are recorded in the LM provenance and shown as a warning. Previously the manager refused the file.
- **A zero-width DOI-limits row.** The owner DOI limits contain `(136027, 0) 3.2 3.2` (left = right). The manager now uses it with the reference formulas unchanged, so its pairs fall out of range and are counted. Previously the manager refused the file.

**For information only** (a different algorithm, so not a parity criterion): the new P = 5 calibration compared with the legacy `fit_gaussian` file, over 31,553 common keys.
- The median μ ratio is +1.0 %.
- The absolute relative difference is 3.0 % at the 50th percentile, 78 % at the 95th and 188 % at the 99th. The legacy file has many poor fits.
- For the 4 failed keys, the new calibration fits `(4360, 2, *)` (μ 81–88 a.u.) and borrows slab 1 for the three slab-0 keys.

## B. Live acceptance (Cornell Linux machine, operator)

**Pre-start record, 2026-10-02 (operator, by terminal; the scanner front end was off, DAQ cards on):**
- Checkout `~/sw/process_petsys` at `4b6d4e81746d646fa607fff77138fcad37d97d18`. The five CMB/IMAS configs and their maps are deleted in that working tree (owner's local state, not used by Cornell); untracked torch files are the owner's. Environment check `python -s -c "import …"` passed.
- A leftover `daqd` (pid 288705, socket from 30 Sep) was stopped by the operator before the manager was launched; `/tmp/d.sock` and `/dev/shm/daqd_shm` were then absent. `set_bias --power off` was not run first, so the bias state at that moment was unknown (no acquisition was running).
- Tools `/home/sie/sw/sw_daq_tofpet2_20260731a/build` (all six present). SHA-256: `daqd` `c8996c4c…676e`, `init_system` `31f63688…a4c718`, `acquire_sipm_data` `2954848e…8993`, `set_bias` `3e9f3592…625c`, `convert_raw_to_coincidence` `5b026d33…68a7`, `convert_raw_to_group` `7d84cd4b…230`. INI `/home/sie/sw/20260304_final_system_newFEBDConf/config.ini` `b55722cd…b6f58`. Cards `/dev/psdaq0`, `/dev/psdaq1`.
- **Finding:** the installed converter's help lists `--writeBinary` and `--writeBinaryCompact` only, no `--writeBinaryFixed` (that flag exists only in the owner's fork). The owner asked for compact LM and a compact pipeline: FR-22, T21. Whether this build accepts the fork flag (`strings … | grep -i fixed`) is still to be recorded; until then `fixed_output_confirmed` stays false.
- **Finding:** the reference LM's hardcoded header (120 modules, 5 rings, 820 mm) is wrong for Cornell. Owner values: 30 modules, 3 rings, ring distance 320 mm, 51.61 × 51.61 mm, 100 × 100 pixels, timestamp unit ps; isotope per acquisition (Ge68 for the test RAW). Section A compared bytes against the reference with its own header values, so its result is unaffected.
- Profile written to `~/.config/process_petsys/petsys_manager.yaml`, outputs in `/mnt/nvme/petsys_manager_t19/`. Test RAW: `/mnt/nvme/decay_serie6/Ge68_10012026_test1/run_0001_attempt_01.rawf` (20.7 GB, `.idxf` present).
- **Finding (step 2, first attempt):** Initialize failed with `ModuleNotFoundError: No module named 'bitarray'`; the failure was shown and Acquire stayed locked. `init_system`, `acquire_sipm_data` and `set_bias` are `#!/usr/bin/env python3` scripts for the system `/usr/bin/python3` (has `bitarray`, `pandas`); launched from `process_petsys`, they ran with the env Python. The owner chose a profile interpreter: FR-23, T22 (`petsys_python: /usr/bin/python3`). Step 2 is to be repeated with it.
- **Environment fixes (operator, 2026-10-02):** a legacy editable `event-petsys` install broke `conda list`; replaced by `pip install --no-deps -e .` (`process_petsys`), stale `event_petsys.egg-info` moved to `~/event_petsys.egg-info.bak`. The env's `tk 8.6.13 noxft` (47 font families, pixelated GUI) was replaced by `tk 8.6.13 xft_h891c84d_3` (336 families); conda also updated `openssl` 3.4.1 → 3.6.3 and `ca-certificates`.
- The installed converter build also lacks the fork's fixed output (`strings … | grep -i fixed` prints nothing): this machine runs compact (FR-22).
- **Finding (step 5):** converter-output validation was a pure-Python record loop on the workflow thread: ~10 µs/record (≈12 min for the 23 GB output) and it starved the Tk thread of the GIL, so the window went blank. Fixed as a bug (T23): compiled, GIL-free validation, same checks and errors, ~0.2 µs/record.
- With the scanner off, steps 3, 4, 9, 10 and 12, the initialized half of step 2, the acquisition part of 11 and 13 wait for a session with the scanner on.

Before starting:
- `git pull` on `main`; record `git rev-parse HEAD`.
- Check the environment: `python -s -c "import customtkinter, numpy, numba, yaml, reportlab, openpyxl"`.
- Stop `gui_cornell` and any `daqd` it started. The manager refuses to start while `/tmp/d.sock` or `/dev/shm/daqd_shm` exists.
- Record the PETsys tools folder, the SHA-256 of `daqd`, `acquire_sipm_data`, `convert_raw_to_coincidence`, `convert_raw_to_group` and `set_bias`, the INI path and SHA-256, and the cards.
- Note the files already in `data_dir` (`ls -l`) so step 13 can show they survived.

Launch with `python exe_programs/PETsysManager.py --profile <profile.yaml>`. Record each run folder.

| # | Step | Pass when | FR | Result |
|---|---|---|---|---|
| 1 | Start DAQD | READY only after the daemon answers; the log shows the owned pid | FR-6 | **pass** 2026-10-02: STARTING → `DAQD ready; pid 799615 answered the shared-memory query (/daqd_shm)` → DAQD ON (not initialized) |
| 2 | Initialize | ready; Acquire unlocks; edit the INI → initialization invalidated | FR-5, FR-6 | failure path **pass** 2026-10-02 (scanner off): with `petsys_python: /usr/bin/python3`, `init_system` reached the daemon and failed `No Trigger unit has been found`; shown, not initialized, Acquire locked. Success path pending (scanner on) |
| 3 | Acquire 30 s with source | RAW started, growth passed, live size/rate, frame loss shown (or "unknown"); `.rawf`/`.idxf` in a new attempt folder | FR-8, FR-20 | pending |
| 4 | STOP during an acquisition | the child stops; `set_bias --power off` runs and its outcome is logged; no retry; bias actually off (operator check) | FR-7, FR-19 | pending |
| 5 | Convert the step-3 RAW to compact coincidence (fixed and group no longer offered, FR-10 amended 2026-10-05) | the output with the shown file list; existing files untouched | FR-10, FR-11 | pending. First attempt 2026-10-02 (Ge68 `run_0001_attempt_01.rawf`, compact, 2 splits, duration left at 10 s): the converter finished in ~2 min (3 nonempty splits `_4/_5/_6`, 23 GB, plus an empty `_0`: the converter numbers splits by absolute time), RAW folder unchanged; then the window went blank during output validation (see finding) and the operator closed it. Nothing left running; outputs kept; manifest `partial`, no artifacts recorded. Repeated 2026-10-04 after T23–T24: **compact pass** on `/mnt/nvme/Norm_10_02_2026/NormRun1/run_0024_attempt_01.rawf` (36.7 GB; Ge68 RAW deleted outside the manager meanwhile), splits 1: 155,860,134 coincidences in one output `convert-20261004-041330-575fa18e/…/run_0024_attempt_01_coincCompact.ldat`, first 10,000 records checked, window responsive; RAW files dated 2 Oct, unchanged. Fixed and group: N/A on this converter build (no fixed output) |
| 6 | Create energy cal file, P = 1 and P = 5, from the compact output | `.encal`, status, sidecar, plot; P = 1 byte-identical to `python scripts_cornell/cornell_slab_en_cal.py <yaml> <compact files>` on the compact output of the same RAW | FR-12, FR-21 | **pass** 2026-10-04 (compact input; fixed N/A on this converter). P = 1 from the step-5 compact output written (`encal/calibrate-20261004-042042-bc001f69`; 6,191/7,680 keys with a factor); **P = 1 pass**: `.encal` and status byte-identical (`cmp`) to `scripts_cornell/cornell_slab_en_cal.py configs/cornell_full_system.yaml` on the same compact file; sidecar 13,006,580 records validated, 10,000,001 events passed, stopped at limit (reference semantics: stop once more than 10 M have passed). **P = 5 pass** (review; no reference): 34 F18 compact files `20260930_F18_950uCi_Run1_60s_coincCompact_*` (not run_0024), `encal/calibrate-20261004-055004-814232cf`; `.encal`, status, sidecar and plot written; 38,400 rows: fitted 27,067, borrowed 3,923, estimated 690, without values 6,720; whole files read (1.55–1.99 M passing events per file, none at the 10 M limit); 17 min 34 s (05:50:04 → 06:07:38, request/result file times; the log has no timestamps, added to T25); window responsive |
| 7 | Generate LM file from the compact output, profile metadata | `.lm` + provenance; the reconstruction software reads it, with the expected header and timestamp units | FR-12 | in progress 2026-10-04: `run_0024` compact (155.9 M records), P = 5 F18 calibration `calibrate-20261004-055004-814232cf`, metadata F18, 900 s acquisition and measurement time (run summary `completed`, `daq_seconds` 900.0), debug plots on. Segment and merged `.lm` written (365,586,368 bytes, 63 min 46 s, window responsive), then the floodmap debug plot failed (`colorbar` nested-list error, T26): run `failed`, no LM sidecar. Fixed in T26. **Repeat 2026-10-04 (after pull, GUI left open): manager side pass**: `listmode-20261004-093632-1aab7fd8`, 6 verified outputs incl. all 3 debug plots (floodmap as expected, operator); 15,232,042 records, file size = 176-byte header + 24 × records; header read back from the `.lm` equals the sidecar (Cornell, F18, acqTime = measurementTime = 900, 51.61 mm, 30 modules, 3 rings, 320 mm, 100 × 100 px, v9.5); all 155,860,134 records accounted for (written + rejected: energy window 88.6 M, min channels 34.9 M, no pair 7.1 M, unresolved slab 6.2 M); zero-width COG/DOI key `(529219, 10)` reported; μ ≤ 0 keys 0; 62 min 53 s; window responsive. Written count differs from the failed run by 716 (15,232,758): the unseeded reference slab rule for ambiguous sides (`np.random`); cuts before it are identical. **Pending:** reconstruction software reads the `.lm` (header, timestamp unit ps) |
| 8 | Section-A check on Cornell compact data | `--real --manifest <cornell baseline>` passes for the compact-only groups (P = 1 calibration vs reference-written files, P = 5 vs the reference-function oracle, QC); fixed-twin and LM-oracle checks N/A without fixed files (owner decision 2026-10-05: LM parity rests on Section A and step 7) | FR-17 | **numerical pass** 2026-10-05 (Cornell PC, `scripts/petsys_manager_t19_baseline_cornell_f18.json`: F18 950 µCi Run1 60 s, compact splits 33–38 of `convert-20261004-044459-cae4ba21`; reference P = 1 written by `cornell_slab_en_cal.py` on the same 6 files, 3 min 23 s, 11,454,707 passing events, 5,410 fitted / 786 borrowed / 140 estimated): P = 1 compact byte-identical to the reference-written `.encal`/status; P = 5 (first 400,000 passing events per file) equals the reference-function oracle; QC (plots, slabs, seeded) equals the reference; inputs and reference sources unchanged. Fixed-twin and LM-oracle checks n/a (compact only, FR-10). Timings: P = 1 187 s, oracle reference 277 s / migrated 36 s, QC migrated 701 s / reference 668 s. Reported **FAIL 4/5** only because the checkout-provenance check refused an untracked runtime file on that PC, `src/slab_nn_torch.py` (not part of this repository); the input fingerprints were taken before the refusal. Results `/home/sie/.cache/process_petsys/petsys-manager-real-20261005T080143Z-47bb16dd/real_baseline.json`. Clean provenance rerun: owner decision pending |
| 9 | Run quality control 60 s with source, plots + slabs; then 180 s without source, plots | PDF, Excel and plots in the shown run folder; source mode and denominators in the report | FR-14 | pending |
| 10 | Run complete pipeline (short Acq. Time, 2 splits) | acquire → fixed → calibration → LM in one run folder; LM header times = Acq. Time | FR-12, FR-13 | pending |
| 11 | Pipeline STOP during conversion; pipeline with a missing COG-limits path | STOP: no later stage starts, controls usable. Missing file: refused before anything starts, with the reason | FR-2, FR-7, FR-13 | in progress 2026-10-05: missing COG limits checked on the manual Create Energy cal file action (P = 5, 34 F18 files, `/tmp/missing_cog.txt`): action unavailable before anything started, reason "COG Limits File: File not found: /tmp/missing_cog.txt"; restoring the path → "prerequisites met". Pipeline variant and pipeline STOP during conversion pending (need initialization, scanner on) |
| 12 | Retry: set the minimum growth above what the system writes, acquire | each attempt in a new folder, bias off after each, retry reason logged, stops after `max_attempts`; restore the setting | FR-8, FR-19 | pending |
| 13 | Close the window during an acquisition | cancellation, bias off, DAQD stopped; `pgrep -af 'daqd\|acquire_sipm_data\|convert_raw'` empty; earlier `data_dir` files unchanged | FR-6, FR-7, FR-9 | partial 2026-10-05 (scanner off: close during a **conversion**): `run_0002` compact, closed after ~51 s (`convert-20261005-023414-cca2dc07`, 02:34:14 → 02:35:05): "Closing" shown, STOP sent, manifest `cancelled` "conversion cancelled: Cancelled; owned child reaped; later stages not started"; no `convert_raw` left; partial outputs kept (split `_00000031` 6.36 GB, unsplit `.ldat` 4.26 GB being written, empty `_00000000`); earlier run folders unchanged; the operator's own `./daqd` (not manager-owned) left running, as FR-6 requires. **Finding:** the window needed a second close (T27); repeated after the fix (`convert-20261005-025333-7f726399`): one close, run and attempt `cancelled`, no `convert_raw` left. Close during an acquisition (bias off, owned DAQD stopped) pending |

## C. Inspector regressions on representative data

| Data | Check | Result |
|---|---|---|
| Cornell (owner January/September files; Inspector source unchanged) | `ldat_inspector_check --selftest` 59/59; `ldat_scale_check --real` 15/15; `ldat_unpopulated_check --real` 8/8; `ldat_views_check --real` 174/174; `ldat_pair_choices_check --real` 19/19 (25 GB file as a 30 M prefix, 1.99 GB result); `ldat_issue_check --real` 8/8; slab convention 16/16 | **pass** (2026-10-02) |
| IMAS | representative IMAS acquisition | pending (no IMAS acquisition on the owner workstation) |

## D. FR-by-FR

| FR | Headless evidence (tasks.md) | Live / real data | Status |
|---|---|---|---|
| FR-1 | T13, T17 GUI `--all` | B launch | pending |
| FR-2 | T18 checkout audit (Windows + Cornell) | B 11 | pending |
| FR-3 | T2, T13 | B launch (profile) | pending |
| FR-4 | T3 | B 1–13 | pending |
| FR-5 | T3, T6, T12 | B 2 | pending |
| FR-6 | T6, Linux dummy 16/16 | B 1, 2, 13 | pending |
| FR-7 | T12–T14, T17 | B 4, 11, 13 | pending |
| FR-8 | T7 | B 3, 12 | pending |
| FR-9 | T4 | B 5, 13 | pending |
| FR-10 | T5, T15 | B 5 | pending |
| FR-11 | T5, T15 | B 5 | pending |
| FR-12 | T8, T9, T16, T19 μ ≤ 0 / zero-width checks | A pass; B 6, 7, 10 | pending (live) |
| FR-13 | T12, T16, T17 | B 10, 11 | pending |
| FR-14 | T10, T16 | A pass; B 9 | pending (live) |
| FR-15 | T8–T10, T17 bounded storage | A pass | **pass** |
| FR-16 | T1–T17, T19 regressions | — | **pass** (headless) |
| FR-17 | — | A, B | pending |
| FR-18 | T18 | sibling untouched | pending |
| FR-19 | T7, T14 | B 4, 12, 13 | pending |
| FR-20 | T7, T14 | B 3 | pending |
| FR-21 | T20 | A pass; B 6 | pending (live) |
| FR-22 | T21 | A pass (compact LM = fixed); B 5–7, 10 with compact | pending (live) |
| FR-23 | T22 | B 2 (repeat), 3, 4, 12 | pending (live) |
