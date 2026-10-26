# Spec 011 — IMAS skew calibration

Status: `draft` (2026-10-26)

Constitution: [AGENTS.md](../../AGENTS.md). Workflow: [docs/prompts.md](../../docs/prompts.md). Builds on shipped [spec 008](../008-cornell-skew-calibration/spec.md) (Cornell per-channel skew LUT, shell pair fits, global solve, native table, point-source CTR). Evidence (untracked, `docs/imas/`): manuscript `IMAS_TBPET_Dec24.docx`, poster `IMAS_IEEE2025_poster.pptx`, readout map `ModuleMap_FEM256_RR.pptx`, LDAT Inspector screenshots `image_regions_down.png` and `image_FEM_skew.png`.

## Goal and population

Compute one subtractive timing delay per IMAS TIME channel (slab) from a rotating-bar acquisition, as spec 008 does for Cornell, and deliver it in the forms IMAS uses. Check it on an independent 22Na point-source acquisition against the legacy IMAS skew LUT.

**System:** IMAS long-axial-FOV PET. 5 rings × 24 SuperModules = 120 SMs, all populated, ~3.9 % of channels failing (owner/manuscript). Each SM has 4 × 4 minimodules; each minimodule has 8 LYSO slabs of 25.8 × 3 × 20 mm and 8 + 8 reduced signals.

**Input:**
- **Data:** PETsys compact **coincidence** LDAT, i.e. two detector hit populations per record, not singles. Coincidence window 10 ns (owner). The PETsys version does not change compact LDAT (owner).
- **Config and map:** `configs/imas_2DAQ.yaml`, `maps/imas2DAQ_map.yaml`: 30,720 channels, 15,360 TIME and 15,360 ENERGY, global channel IDs; ring-major SM numbering (SM = azimuthal index + 24 × ring), FEM256, two DAQ boards.
- **Units:** timestamps in ps; hit energies in PETsys QDC units (a.u.); keV only through the energy calibration of FR-4.
- **Development data (local):** `20251119_Skew_400s_0_coinCompact_00000008/9/10.ldat` under `PETSYS_DATA_DIR/IMAS/` (3 of about 30 splits, ~16.5 GB and ~13.3 s each, about a third of a bar turn).
- **Ship data:** about 1 TB of skew acquisitions brought by the owner on an external HDD.
- **Validation data:** an IMAS 22Na point-source acquisition (owner is transferring it; name and position pending) and the legacy IMAS skew LUT (file pending).

**Owner decisions, 2026-10-26:**

- **Detector:** one slab per TIME channel; the highest-charge TIME channel gives the slab. Slabs run axially: the monolithic coordinate is axial, the slab index transaxial. Crystal thickness 20 mm. Hardware unchanged since the acquisition.
- **Source:** PMMA bar, 10 mm diameter, ~700 mm long, axially centred, filled with FDG, rotated at constant speed (120 s per turn) with its centre at 160 mm radius. Over several turns it forms a corona as in Cornell. `pipe_radius` in `imas_2DAQ.yaml` corrected from 321 to 160 mm.
- **Source model:** corona shell, two branch hypotheses per LOR, as in spec 008. No rotating-line model.
- **Interaction depth:** fixed 7 mm from the entrance face, as for Cornell.
- **Code:** generalize `src/cornell/skew/` into scanner-neutral code with Cornell and IMAS adapters. Cornell results stay identical.
- **Energy calibration:** generated from the skew data, not taken from another acquisition.
- **Estimator:** measure the real per-pair statistics first; the estimator is decided from that measurement (FR-7).
- **Down regions:** from the data (counts per channel), as the LDAT Inspector shows them; no hand-written list is required.

## Requirements

- **FR-1 — Scanner-neutral skew code.**
  - Inventory, ingest, cache, pair fits, solver, LUT, native export, CTR and diagnostics run for a declared scanner adapter (Cornell or IMAS). The adapter supplies SM numbering, detector frames, side selection, energy-calibration keys and the source model.
  - Cornell keeps `python -m src.cornell.skew` and its job, CTR and walk files. Its tracked tests, golden files and baselines pass unchanged, and a Cornell job gives byte-identical caches and LUTs before and after the change.
  - Nothing Cornell-specific (cassette numbering, 105 mm SM pitch, slab coin flip, per-(TIME channel, slab) `.encal`) applies to IMAS.
- **FR-2 — IMAS geometry contract.**
  - SM centres come from `ring_yx` and `ring_z`, ring-major, checked against the map's channel-to-SM assignment.
  - `ring_r` = 410 mm is read as the entrance-face radius (bore 82 cm, manuscript). Endpoints are at 7 mm depth.
  - Transaxial position: the centre of the selected slab within its SM. Axial position: from the selected minimodule's ENERGY-channel centroid across the minimodule's monolithic length. Both are nominal; no NN or DOI calibration is used.
  - The transaxial and axial sign of the local frame are selected by the data, with the evidence recorded, as Cornell's handedness was (spec 008 T6).
  - The geometry is reported as nominal: SM face positions, minimodule pitch and ring positions are not physically qualified.
- **FR-3 — IMAS side selection.**
  - Per side: `min_ch` and `en_min_ch` from the config, the highest-energy minimodule, its highest-charge TIME hit as timestamp and slab.
  - Both sides inside the declared keV window; LOR opening angle at least 165° in 3D; `|dt|` inside the declared coincidence window.
  - A reference mode reproduces the selection of `scripts_imas/imas_skew_cal_bigdata_optimized.py` on a bounded prefix; its differences from the production selection (slab from the highest TIME channel instead of the TIME-channel centroid, 15 ns legacy window) are listed with counts.
  - Every rejection is counted once, by its first failing cut.
- **FR-4 — Energy calibration from the skew data.**
  - A per-TIME-channel 511 keV photopeak calibration is fitted from the skew acquisition's own events, from a declared subset of files or records that is recorded in the job.
  - Channels without a usable peak get an explicit status; their events are rejected and counted, never converted with a guessed factor.
  - The output is a file `KevConverter` reads and a manifest with its inputs and fit parameters. Every skew and CTR result names the calibration and keV window that produced it.
- **FR-5 — Corona source model.**
  - An annulus from 155 to 165 mm radius (bar centre 160 mm ± 5 mm), axially from −350 to +350 mm about the scanner centre, coaxial with the scanner. Its parameters, and the fact that they are nominal, are recorded in every output.
  - Each accepted LOR crosses the annulus in two segments, the two branch hypotheses of spec 008; an LOR that does not cross it twice is rejected and counted.
- **FR-6 — Inventory, scale and cache.**
  - A job file lists every split explicitly (paths on the external HDD included), with frozen roles (energy calibration, train, held-out).
  - The inventory reports per file: physical coincidence records, first-failure counts, accepted observations, and counts per TIME channel and SM. Channels with no or few observations are listed as unobserved or low-statistics, so down regions appear from the data.
  - Memory is bounded by the configured chunk and cache buffers, never by the number of records. The job resumes from completed cache blocks and never reuses a cache whose inputs, config, map, calibration, geometry, cuts or schema differ.
  - The job reports stage timings, throughput and peak memory; on ~1 TB it must finish on the machine that holds the HDD (a time budget is set after the FR-7 measurement).
- **FR-7 — Pair-statistics measurement and estimator decision.**
  - Before any IMAS solver work, a measurement on the 3 local splits reports: accepted observations, distinct channel pairs, observations per pair (distribution), fraction of pairs with both branches resolved, partners per channel, connected components, and a linear projection to ~1 TB.
  - The owner then chooses the estimator, recorded here as a spec change: spec 008's per-pair two-branch fits and global solve, a per-channel estimator over all partners, or grouped pairs (minimodule or ASIC pairs first).
  - Whatever is chosen recovers injected delays on synthetic IMAS-geometry fixtures within its declared uncertainty, keeps a gauge, and reports per-channel uncertainty.
- **FR-8 — Solve, statuses and gauge.**
  - As spec 008 FR-10 and FR-12: connected components, an explicit gauge, uncertainties, statuses (calibrated, low statistics, disconnected, ambiguous, failed fit, unobserved).
  - No missing value, `9999` or invented zero enters any correction.
- **FR-9 — Outputs.** From one solved LUT:
  - **Canonical LUT:** spec 008's CSV and manifest, with IMAS global TIME-channel IDs, subtractive delays `b` in ps (`t_corrected = t − b`), uncertainty, status and `apply` flag.
  - **PETsys native table:** `time_offset_calibration_table` rows `port slave chip channel offset_ps` for applied channels, decoded from the IMAS global IDs. Whether the two DAQ boards need one table or two is settled in Clarify.
  - **Legacy skew file:** `ch skew` per line as `src/utils.py:read_skew` reads it. The legacy consumers add the value (`t + skew`), so `skew = −b`. Channels without an applied delay are omitted; that legacy readers then use `9999` is documented, not hidden.
  - **Timestamp correction:** a tracked function applies the canonical LUT to the timestamps of an LDAT batch or a coincidence's two sides and counts uncorrected hits, so any script can use the LUT without converting files.
  - Each export is checked through its consumer: positive injected delays, endpoint swaps, and the native parser of spec 008.
- **FR-10 — Point-source CTR.**
  - The LUT is frozen and evaluated on the 22Na acquisition with the job's cuts and calibration (or one fitted the same way from the point-source data; recorded).
  - The source position comes from timing-blind LOR convergence, as in spec 008; its declared nominal position is recorded too.
  - On the same events: uncorrected, legacy IMAS LUT and the new LUT, with model-free and Gaussian FWHM, uncertainties, tails and counts.
  - **Acceptance:** the new LUT's FWHM is better than the legacy LUT's by more than their combined uncertainty. The manuscript's legacy values (2293 ps uncorrected, 563 ps corrected, NECR phantom at 0.5 kBq/mL) are context, not a comparison on the same events.
- **FR-11 — Diagnostics.**
  - Per-ASIC diagnostics as spec 008 FR-19, decoding IMAS global IDs.
  - A per-SM and per-minimodule delay overview laid out by ring and azimuthal SM, like the Inspector's System Overview.
  - Residual consistency between train and held-out files, and between the new and legacy LUTs.
  - These are diagnostics, not validation evidence.
- **FR-12 — Ship run.**
  - The job runs on the owner's ~1 TB HDD acquisitions with the FR-7 estimator. Its LUT is the deliverable that FR-10 validates.
  - Results from the 3 local splits are development results and are reported as such.

## Completion criteria

- **Synthetic fixtures:** injected delays on an IMAS-geometry fixture recover exactly in the noiseless case after fixing the gauge, and within declared uncertainty with noise, both branches and background. Exports reproduce them through each consumer.
- **Cornell:** FR-1's unchanged-results checks pass.
- **Real data:**
  - FR-7 measurement recorded and the estimator decision written into this spec.
  - The ship job over ~1 TB completes with recorded inventory, coverage, timings and peak memory.
  - FR-10 acceptance passes on the 22Na acquisition against the legacy IMAS LUT.
- **Suites:** the required repository suites pass; missing acquisitions are explicit skips naming `PETSYS_DATA_DIR`.

## Open questions (Clarify)

1. **Legacy IMAS LUT:** which file, from which acquisition and with which cuts; confirm it uses the additive `t + skew` convention of `imas_listmode.py`.
2. **22Na acquisition:** name, nominal position, activity and duration; was it acquired after the skew run, with the same hardware state?
3. **~1 TB contents:** the 400 s skew run is ~30 splits (~500 GB). Is the rest more splits of the same run, or other skew runs (different day, activity, bar radius)? If several runs, are they pooled as in spec 008 FR-18?
4. **keV window:** the config uses 450–570 keV, the manuscript's skew used 408–613 keV. Which one for the skew job and for the CTR?
5. **Native table and two DAQs:** is the IMAS RAW → LDAT conversion one PETsys configuration with one `time_offset_calibration_table`, or one per DAQ board?
6. **Energy calibration subset:** a dedicated set of splits, or a prefix of every train file (spec 008's held-out prefix rule)?
7. **Run location for the ship job:** this Windows PC reading the HDD, or another machine?

## Out of scope

- Fitting models: rotating-line source model, per-LOR skews, ML timing models, time-walk correction for IMAS (the spec 009 analogue is a later spec).
- Position: NN or DOI calibration, physical qualification of the IMAS geometry.
- Other components: reconstruction, IMAS listmode changes beyond the correction function, Manager actions, GUI widgets, conversion of RAW files.
- Manufactured singles.
