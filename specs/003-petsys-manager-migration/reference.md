# Reference 003 — T1 inventory and parity contract

Recorded 2026-09-30. Reference-only checks; no manager runtime, acquisition, hardware commands, real acquisition processing or source migration. Constitution: `AGENTS.md`.

## Confirmation and provenance

- Owner confirmed the three local `scripts_cornell/` processing scripts are the Cornell-used versions, and that Cornell uses a modified PETsys converter supporting `--writeBinaryFixed`.
- These are **operator declarations**, not remotely measured hashes or a successful hardware/converter execution. Shared helpers and tool sources below were fingerprinted locally; their installed-machine identity remains unmeasured.
- The sibling `sw_daq_tofpet2` converter source does not implement `--writeBinaryFixed`. Its `--writeBinary` is a different layout, not an equivalent substitute. Do not build/deploy it as the Cornell fixed-output converter.
- Exact modified converter version/source/binary hash, LM reconstruction metadata/timestamp units and representative RAW/fixed/compact baseline data remain pending. Hardware/operator acceptance remains T19.
- Candidate `configs/cornell_full_system_20260928.yaml` and its map were not changed or implicitly selected. Its geometry comments are not a confirmed LM reconstruction profile.

### Primary source fingerprints (SHA-256, file bytes)

| Source | SHA-256 |
| --- | --- |
| `../gui_cornell/src/gui.py` | `907ad50ee50b7d3f5c0779e9278dc0f20b98510f890becba13ad922f54f1412d` |
| `scripts_cornell/cornell_slab_en_cal_fixed_position.py` | `8e5bb5cfdd4a287b5646b2a81587c6c1a044f760daa3d1cccafa6f4e38533ec4` |
| `scripts_cornell/cornell_listmode_cog_fixed_position.py` | `f7adbaf8cef6677ac01ebb312562822ee9a284e9450a7af26b81160641311f81` |
| `scripts_cornell/cornell_system_validation.py` | `3c1b9c57d41de8aabeb63fb6145dae43c4c6d83efbce3053d6da9e15138e1607` |

The private manifest also records 13 shared helpers (`read_fixed`, `read_compact`, `utils`, `utils_fixed`, `filters`, `filters_fixed`, `detector_features`, `detector_features_fixed`, `fits`, `mapping_generator`, `fem_handler`, `listmode`, `yaml_handler`) and seven sibling PETsys source files (DAQD, writer source/header, both converters, acquisition and initialization scripts): **24 unchanged sources total**. Byte hashes include local line endings; cross-machine comparison must distinguish line-ending changes from algorithm changes.

## Reproducible check

Local check (intentionally ignored/untracked): `scripts/petsys_manager_reference_check.py`.

```powershell
conda run -n process_petsys --no-capture-output python scripts/petsys_manager_reference_check.py --synthetic --confirm-local-scripts --confirm-modified-converter
```

Confirmation flags record the owner's declarations above; omitting them leaves confirmations pending. Neither flag runs or verifies hardware. Reference numerical modules are imported after inspection; reference `main()`, GUI initialization and multiprocessing pools are never invoked. Boundary tests explicitly inject counters/aggregate counts, as described below; they are not million-record acquisitions.

Each run reserves a new exclusive private directory and refuses an existing `--output-dir`. It emits miniature fixtures, controlled reference outputs, source fingerprints, environment versions, static GUI/tool options, policies, results and pending gates to `baseline.json`. No private inputs/results are added to Git.

**Verified:** 17/17 checks passed in Python 3.10.14, NumPy 1.24.3, numba 0.63.1, reportlab 4.4.9, using the `process_petsys` environment. Repeat run with the corresponding environment interpreter produced identical check outcomes/evidence and fixture digests; the latest manifest also fingerprints the check script itself. Import-only and compile checks passed. The check script is ignored by Git. A concurrent `conda run` import smoke check initially failed on conda's temporary activation-file collision; the same check passed with the corresponding environment interpreter. No application change was required.

Private verified manifest:

`C:\Users\dsanchez\AppData\Local\Temp\opencode\petsys-manager-reference-20260930T103452Z-c907d4be\baseline.json`

## Independent fixture contract

- Two synthetic SuperModules, one minimodule each; explicit time positions 0–7 and four energy channels, no real map files or hardware count assumptions. Map metadata is saved with the fixtures.
- Four hand-encoded coincidence pairs: ordinary resolved, reverse-oriented resolved, unresolved-slab pair and an energy-channel-count failure. Fixed and compact readers reproduce every manually supplied timestamp/energy/channel hit.
- Eight detector sides, 47 input hits; fixed coincidence hit limit 16 gives 514 bytes/record plus the 4-byte file header. Hits are little-endian `(int64, float32, int32)`, 16 bytes. Separate fixed and compact group encoders/read checks exercise four groups; group records are not called singles.
- With four minimum energy channels, calibration accepts **6 sides** independently (a valid partner survives a rejected side); group calibration accepts **2 groups**. Accepted region keys are `(4, 6, 2)` with 2 values and `(104, 6, 2)` with 4 values.
- QC accepts **2 resolved pairs / 4 energy/flood sides**. Occupancy includes **3 pairs / 36 channel hits**, because the unresolved pair contributes hits before slab rejection. The channel-count failure contributes neither population. Do not equate these denominators.

## Calibration contracts

- Five default position regions, edge multiplier 1.8; manual normalized boundaries `[0, 3/11, 14/33, 19/33, 8/11, 1]`. Calibration excludes normalized COG outside `[0, 1]`, including 1 in the last region; LM region assignment clips to `[0, 0.999]` instead. These intentionally remain separate policies.
- Reference calibration reads `min_ch`, but does not apply a per-channel energy threshold. Its acceptance counter is sides/groups, not coincidence pairs.
- Nominal limit is 4,000,000 accepted sides per file, tested between side chunks. A real miniature cap of 3 accepts 4 sides; aggregate-count injection into the unchanged orchestration gives 3,999,999 then 2, stopping at 4,000,001. The test does **not** retain/decode four million actual sides. Default batch is 5,000 records; the actual overshoot/sample must be recorded.
- Fit input: 100 bins over 0–200 a.u., `cb=6`, `pk_finder='peak'`, at least 50 samples per key. Forced `RuntimeError` uses population mean/std (`ddof=0`); the legacy file does not distinguish this fallback from a fit. Migration must add status/provenance, not pretend fallback factors are fitted.
- Independent analytic Gaussian histogram has true μ=100, σ=10 a.u.; reference fit gives μ=`99.99996478570006`, σ=`10.004009872999005`, within predeclared 0.05 a.u. truth bounds. Histogram digest is in the manifest.
- `.encal` starts with `# Position-dependent energy calibration (5 regions per slab)` and `ID(time_ch, slab, region)\tmu\tsigma`, sorted three-part keys, μ/σ to three decimals. Existing `KevConverter(..., 'cornell_position')` reads it. Region count is present; exact boundaries/generation cuts/fit status need provenance outside the unchanged data schema.

## QC contracts

- Compact reader applies selected `en_min_ch` in a.u.; energy channel count is checked before/after selecting the highest-energy minimodule. Occupancy counting precedes unresolved-slab rejection; energy/flood/slab samples require both sides resolved.
- Loop stops when accepted pairs **exceed** 1,000,000, normally 1,000,001 per file. Seeded counters at 999,999 / 1,000,000 / 1,000,001 process respectively 2 / 1 / 0 new valid pairs. This executes the actual loop with injected initial counters, not a full 1 M record dataset. The generator may yield the next pair before the break; record processed/sample counts separately from iterator reads.
- Raw photopeak fit uses 150 bins over 0–250 a.u. with caller-selected `cb`; failed fit returns mean/std plus an error flag. Do not label raw μ as keV or a failure estimate as a valid photopeak measurement.
- Existing output schemas are checked via an actual fixture PDF and Excel output: `missing_channels_report.pdf`, `photopeak_values.xlsx` with columns `SM`, `Minimodule`, `Mu`, `Sigma`, `Energy Resolution (%)`. This schema check is not a manual PDF layout review or a real detector QC verdict.
- Plot/layout helpers contain hardcoded counts and cassette selections. Preserve computations/output intent, but use the selected map/config in migrated expectations and report any intentional differences.

## Slab and LM contracts

- Current scalar/vectorized helpers agree on edge, lower/upper adjacent and unresolved cases: slab `2p` is at `X_p - 0.8 mm`, slab `2p+1` at `X_p + 0.8 mm`. Seed 741 reproduces each engine's one-channel assignments across 128 trials, seeing both slabs (scalar even: 64; vectorized even: 66). Cross-engine random draws need not match; neither production rule was changed.
- Independent structure oracle pins **176-byte `LMHeader`** (`_pack_=4`) and **24-byte `CoincidenceV5`**, all field offsets, padding and bytes. Record format: `<fHHfbbbbbbHh2x`; pair offset 18, signed Δt offset 20. Header/version stays compatible with `(9, 5)`; no schema redesign.
- Controlled forward/reversed pairs both write ordered energies 511/1022 keV, amount 1, pixel X/Y 3/1, DOI code 10, pair ID 7 and Δt 10. Raw input time numeric values 90 and 110 are assigned to float32. `first_tstp` is extracted in the CLI but unused; no time normalization/scaling occurs in the tested loop. Consumer timestamp units/precision require confirmation before LM migration acceptance.
- Reference header hardcodes acquisition/measurement time 10, module number 120, ring number 5, ring distance 820; several optional fields are zero. These bytes are the **legacy oracle**, not evidence those values describe the actual system/acquisition. Required corrections must use supplied profile/duration, retain field layout and be explicitly compared/documented.
- `en_min_ch` is read/printed by LM `main()` but not passed/applied to its fixed processing loop. A fixture with a 0.125 a.u. fourth energy hit is accepted by fixed calibration and LM (wide keV window), accepted by QC with cut 0, but rejected by QC at cut 0.2. Preserve/disclose actual reference cuts; adding a fixed-path channel cut would be a scope/behavior change, not a harmless refactor.
- Static inspection flags inconsistent partial-decompression masking in legacy LM. Negative cases must be exercised in T9; T1 does not claim a demonstrated real-input failure or fix it.

## Tool/lifecycle inventory

- GUI DAQD command uses `--socket-name`, `--daq-type PFP_KX7`, two `--card` arguments (`/dev/psdaq1`, then `/dev/psdaq0`). Its acquisition command uses `--config`, `-o`, `--time`, `--mode qdc`, optional `--enable-hw-trigger`.
- Fixed conversion requests `--writeBinaryFixed --writeMultipleHits 16`; QC requests `--writeBinaryCompact --writeMultipleHits 16`; splits use `--splitTime` with duration/splits + 0.1 s. Requested splitting is not a reliable guarantee of an exact file count.
- GUI safety defaults: startup 45 s, growth 20 s, interval 5 s, minimum growth 20 MB, loss threshold 5%, maximum attempts 3, retry delay 2 s. Source presets: 60/180 s. These are recorded policies, not measured acquisition performance.
- Local DAQD source supports socket/type/cards/debug, hardcodes shared memory `/daqd_shm`, limits cards to two and creates the listening socket **before** card/shared-memory initialization. Socket existence alone is not readiness.
- Local `init_system` and acquisition scripts connect through default `daqd.Connection()` without a selectable socket argument. Do not promise end-to-end custom-socket support from the DAQD flag alone; installed-tool capabilities must be checked in T6.
- **T3 argv clarification:** inspected `init_system` takes no INI/config argument at all; acquisition/conversion load the selected INI. DAQD has no shared-memory-name flag. Command builders preserve these known argv contracts and reject unrecorded custom socket/shared-memory behavior; no tool source or installed hardware was changed/executed.

## Migration comparison rules / remaining gates

- Exact counts, keys, histogram bins/counts, layouts and controlled LM record bytes.
- Calibration output agreement at its written precision (0.001 a.u.). Scalar/vectorized coordinate truth tolerance 1e-4 mm; region boundaries 1e-14 absolute; fallback moments 1e-12 absolute; same-environment fit oracle comparison 1e-6 absolute. Synthetic Gaussian truth check is separately bounded at 0.05 a.u.; do not widen parity tolerances after a mismatch.
- Supplied duration/profile may intentionally replace incorrect hardcoded header fields; record exactly which fields changed and why. Missing required metadata blocks generation.
- T1 is complete with operator-confirmed local processing sources and fixed-flag support declaration. T2 settings/contracts can proceed. T8/T10 reference-script gate is cleared; T9 still needs the reconstruction metadata/timestamp contract. Exact installed tool provenance, shared-helper parity, real acquisition comparisons and Linux hardware/operator acceptance stay pending through their respective tasks.
