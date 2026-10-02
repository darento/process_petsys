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
- The installed converter build also lacks the fork's fixed output (`strings … | grep -i fixed` prints nothing): this machine runs compact (FR-22).
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
| 2 | Initialize | ready; Acquire unlocks; edit the INI → initialization invalidated | FR-5, FR-6 | pending |
| 3 | Acquire 30 s with source | RAW started, growth passed, live size/rate, frame loss shown (or "unknown"); `.rawf`/`.idxf` in a new attempt folder | FR-8, FR-20 | pending |
| 4 | STOP during an acquisition | the child stops; `set_bias --power off` runs and its outcome is logged; no retry; bias actually off (operator check) | FR-7, FR-19 | pending |
| 5 | Convert the step-3 RAW: fixed coincidence, then compact coincidence, then group | three outputs with the shown file lists; existing files untouched | FR-10, FR-11 | pending |
| 6 | Create energy cal file, P = 1 and P = 5, from the fixed output | `.encal`, status, sidecar, plot; P = 1 byte-identical to `python scripts_cornell/cornell_slab_en_cal.py <yaml> <compact files>` on the compact output of the same RAW | FR-12, FR-21 | pending |
| 7 | Generate LM file from the fixed output, profile metadata | `.lm` + provenance; the reconstruction software reads it, with the expected header and timestamp units | FR-12 | pending |
| 8 | Section-A check on the new acquisition | `--real --manifest <cornell baseline>` passes (calibration, LM, QC) | FR-17 | pending |
| 9 | Run quality control 60 s with source, plots + slabs; then 180 s without source, plots | PDF, Excel and plots in the shown run folder; source mode and denominators in the report | FR-14 | pending |
| 10 | Run complete pipeline (short Acq. Time, 2 splits) | acquire → fixed → calibration → LM in one run folder; LM header times = Acq. Time | FR-12, FR-13 | pending |
| 11 | Pipeline STOP during conversion; pipeline with a missing COG-limits path | STOP: no later stage starts, controls usable. Missing file: refused before anything starts, with the reason | FR-2, FR-7, FR-13 | pending |
| 12 | Retry: set the minimum growth above what the system writes, acquire | each attempt in a new folder, bias off after each, retry reason logged, stops after `max_attempts`; restore the setting | FR-8, FR-19 | pending |
| 13 | Close the window during an acquisition | cancellation, bias off, DAQD stopped; `pgrep -af 'daqd\|acquire_sipm_data\|convert_raw'` empty; earlier `data_dir` files unchanged | FR-6, FR-7, FR-9 | pending |

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
