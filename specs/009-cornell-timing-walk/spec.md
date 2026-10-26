# Spec 009 — Cornell time-walk correction

Status: `shipped` (2026-10-20; T9 validation on Windows, [known limitations](#known-limitations-at-closing); PETsys Manager 1.2.0)

Constitution: [AGENTS.md](../../AGENTS.md). Workflow: [docs/prompts.md](../../docs/prompts.md). Builds on shipped [spec 008](../008-cornell-skew-calibration/spec.md) (per-channel skew LUT, point-source CTR). Evidence (untracked): [docs/cornell/research/cornell-timing-walk.md](../../docs/cornell/research/cornell-timing-walk.md).

## Goal and population

Reduce the Cornell system CTR beyond the per-channel skew LUT by correcting each side's timestamp for its **time walk**, DOI and photon energy. The walk depends on the timing-hit charge and its light-sharing share. Deliver a second frozen table, the **walk LUT**, beside the skew LUT. Both are applied to any later acquisition, patient runs included, from quantities present in every event, without a source position.

**Input:**
- **Data:** PETsys compact **coincidence** LDAT, i.e. two detector hit populations per record, not singles.
- **Map:** the Cornell full-system map `maps/cornell_map_full_system.yaml`. TIME/ENERGY channel types come from the selected map; TIME channels use global IDs.
- **Units:** timestamps in ps. Hit energies in PETsys QDC units (a.u.).

**Selection:** as in spec 008. Per side, the selected minimodule is used, and the timestamp comes from its highest-charge TIME hit.

**The four walk inputs per side:**
- **Timing-hit charge:** the raw QDC charge of that top TIME hit.
- **Light-sharing share:** that charge divided by the summed TIME-hit charge of the selected minimodule.
- **DOI:** the value Cornell listmode already computes (its light-sharing DOI mapped through the DOI limits file). It is not an independently calibrated depth.
- **keV:** listmode's keV, from its (TIME channel, slab, region) calibration file.

**Owner decisions, 2026-10-19 (research probe and clarification):**

- **Tables:**
  - **Two tables:** the per-channel skew LUT and a new walk LUT, each with its own manifest. Per side, `t_corrected = t − b[TIME channel] − W[charge, share] − D[DOI] − K[keV]` (subtractive, ps).
  - **Model:** one global walk LUT: a 2D table over charge × share (20 × 10 bins) plus 1D DOI and keV curves (20 bins each). It is not per channel; in the probe, per-ASIC curves gained about 1 ps for about 30 times the parameters.
  - **Bins:** quantile edges frozen from the fit events and stored in the LUT, with a minimum-events rule per bin.
  - **Out of range:** inputs outside the table use the edge bin. They are counted per input and never rejected.
- **Fitting:**
  - **Calibration source:** the centred point source X0Y0Z0, where the expected TOF is about 0 on every LOR. A normalization-phantom fit is the last task of this spec and must reproduce the point-source tables.
  - **Fit energy window:** 300–700 keV, wider than any analysis window in use (357–665, 408–613), so analysis runs do not clamp. Skew and CTR cuts are unchanged.
  - **Skew LUT:** re-solved on r7 with the walk LUT applied before the pair fits. The current LUT holds each channel's average walk under the phantom spectrum. The result is still a separate skew LUT.
  - **One round trip:**
    1. Walk fitted with the T10 skew LUT.
    2. Skew re-solved with walk.
    3. Walk refitted with the new skew.
    4. Final CTR.

    The change at each step is reported.
- **Commands and data:**
  - **Command:** a new subcommand `python -m src.cornell.skew walk WALK.json`.
  - **Cache:** the feature-cache schema is bumped to carry the walk inputs. The r7 cache is rebuilt once.
  - **Validation:** X0Y-1/-6/-10Z0 (same day) now. A point-source acquisition on another day is a later stability check.
  - **Acceptance:** held-out model-free and Gaussian FWHM must both improve by more than their combined uncertainty on X0Y-1/-6/-10. No fixed target.
- **Production:**
  - **Where:** `src/cornell/listmode.py` applies both tables to the `dt` it writes. Today it writes raw `dt`; the native PETsys table was deleted, and PETsys has no walk correction.
  - **Inputs:** table paths come from Manager profile keys and the listmode CLI/job. No new GUI widgets.
  - **Real-data checks:** on Windows with local LDATs.
  - **Recalibration:** with every skew recalibration, using one short centred point-source run. The CTR command with frozen tables is the drift check.
- **Not pursued:**
  - Multi-timestamp averaging: measured not viable on this readout.
  - Per-channel walk tables.
  - ML timing models.
  - `vth_t1` threshold tuning (an acquisition setting).

## Requirements

- **FR-1 — Walk inputs.**
  - One tracked function computes the four walk inputs per side. The walk fit, the skew job, the CTR and listmode all use it.
  - DOI and keV use listmode's definitions and its calibration and DOI-limits files.
  - **Hits with charge >= 0 only (found at T1, 2026-10-20):** the skew ingest drops negative QDC charges (pedestal noise) before selection, while listmode's reader keeps them. The walk inputs are therefore computed by one function, `skew.walk.inputs`, from the selected minimodule's hits with charge >= 0 in both pipelines.
    - Listmode's written DOI and keV keep the reference hits and are unchanged.
    - On the X0Y0Z0 prefix, the written DOI differs from the walk DOI on 0.6 % of events, by up to 2.4 mm.
  - Missing inputs (no TIME hit, zero sums, no calibration or DOI limit for the key) are counted rejections, never zero corrections.
- **FR-2 — Walk LUT contract.** Versioned schema containing:
  - bin edges, offsets in ps, events per bin;
  - the minimum-events rule: bins below it take the nearest populated bin and are listed;
  - the subtractive per-side convention;
  - the gauge: only side differences are observable, so offsets are zero-mean over the fit events;
  - edge clamping;
  - a manifest: inputs and their SHA-256, skew LUT SHA-256, config, map, listmode calibration and DOI-limits files, per-slab `.encal`, cuts, fit window, seed, fit/held-out split, events.

  Existing outputs are never overwritten.
- **FR-3 — Point-source walk fit.** `python -m src.cornell.skew walk WALK.json` fits the walk LUT from declared point-source acquisitions with a frozen skew LUT:
  - **Cuts:** spec 008 CTR cuts, except the declared 300–700 keV fit window.
  - **Source:** located by timing-blind LOR convergence.
  - **Residual:** corrected `dt` minus geometric TOF.
  - **Fit:** joint least squares over all blocks on fit records, with an iterated core window.
  - **Reports:** curves, per-bin counts and held-out CTR before and after.
- **FR-4 — Skew job with walk.**
  - A job option names a walk LUT. Observations are corrected before the pair fits, using the walk inputs from the bumped feature cache (schema version change, cache rebuilt, never silently reused).
  - The skew LUT manifest records the walk LUT SHA-256.
  - A skew LUT solved with walk is refused when applied without that walk LUT, and the other way round.
- **FR-5 — CTR with walk.**
  - The CTR file may name a walk LUT. The report records it and gives the CTR with and without it on the same events.
  - Acceptance uses positions not used for the walk fit.
  - The report includes the round-trip summary: the CTR after each step.
- **FR-6 — Listmode application.**
  - `src/cornell/listmode.py` optionally applies a skew LUT and a walk LUT to `dt`, in ps, rounded before the int16 write.
  - Table paths come from Manager profile keys (`skew_lut`, `walk_lut`) and the listmode CLI/job.
  - Provenance records both SHA-256 values and the counts (clamped per input, missing inputs).
  - Tables whose map, COG limits or DOI limits do not match the listmode run are refused.
  - **Calibration (owner decision 2026-10-20, superseding "listmode refuses another calibration"):** production listmode converts keV with each run's own calibration. Any calibration with the same layout and region count as the one the walk LUT was fitted with is accepted. Both SHA-256 values are recorded and the run is flagged `calibration_differs_from_fit`. A different layout or region count is refused.
  - **Owner decision (2026-10-19, at T7):** events without a correction are handled by an explicit policy:
    - **Which events:** either timestamped TIME channel has no applied skew delay, or a walk input is missing.
    - **`keep`** (default): written with their uncorrected `dt` and counted.
    - **`drop`**: rejected and counted.
    - The policy comes from profile `lm_timing_uncorrected` and request option `timing_uncorrected`.
    - The owner also mentioned marking such events with a "characteristic" `dt`. That is not implemented: a sentinel `dt` would be read as TOF unless reconstruction knows it (reconstruction is out of scope).
  - Without tables, the output is byte-identical to the current writer. With tables, the corrected `dt` matches the CTR code's corrected `dt` on shared events.
- **FR-7 — Phantom walk fit (last task).**
  - Fit the walk LUT from normalization-phantom data, with no known source.
  - Accepted only if it reproduces the point-source tables within a declared tolerance on populated bins, and gives the same held-out point-source CTR within its uncertainty. Otherwise it is recorded as a limitation.
  - **Owner decision (2026-10-20, after T8):** accepted on the CTR criterion.
    - The r7 phantom walk gives 393.9 ± 1.7 / 444.9 ± 0.6 ps on X0Y-1/-6/-10. The point-source walk gives 394.1 ± 2.3 / 444.9 ± 0.6 ps.
    - The declared table tolerance is **20 ps rms** per side on point-source events; measured 13.7 ps rms (p95 26 ps). The tables have different quantile edges, so a per-bin comparison is not used.
- **FR-8 — Diagnostics.**
  - PNGs of the fitted curves and the 2D table with bin counts, residual-vs-input profiles before and after, and per-position CTR before and after.
  - These are diagnostics, not validation evidence.

## Completion criteria

- **Synthetic fixtures:**
  - Injected walk, DOI and keV offsets recover the tables within declared uncertainty, with the gauge kept.
  - Skew and walk together recover the injected delays and walk.
  - Edge clamping and missing inputs are counted.
- **Real data (Windows, local LDATs, after the round trip):**
  - Held-out model-free and Gaussian FWHM on X0Y-1/-6/-10Z0 each improve by more than their combined uncertainty. The comparison is against the spec 008 result at 357–665 keV: model-free 424.0 ± 3.1 ps, Gaussian 489.9 ± 0.6 ps. The research probe expects about 394 and 446 ps.
  - The results are reported per position and pooled.
- **Listmode:**
  - Without tables, the output stays byte-identical.
  - With tables, the corrected `dt` matches the CTR code on shared events.
- **Phantom fit:** FR-7 passes or is recorded as a limitation.
- **Suites:** the required repository suites run; missing acquisitions are explicit skips.

## Known limitations at closing

- **Same-day validation only:** X0Y-1/-6/-10Z0 were acquired on the same day as the X0Y0Z0 fit source. A point-source run from another day (stability) is pending.
- **One r7 job, one point-source set:** the round trip used the r7-only job, as in spec 008 T10.
- **The DOI walk input** excludes negative-charge hits. On 0.6 % of events it differs from listmode's written DOI, by up to 2.4 mm.
- **Phantom table vs point-source table:** 13.7 ps rms per side (owner-accepted tolerance 20 ps). `max_events` strides over train rows before geometry, so phantom fits must set it above the train row count (r7: 116M rows, 1.85 % accepted).
- **Production calibrations:** the walk LUT was fitted with the per-slab r7 calibration. Production listmode with a region-layout calibration is refused until a walk LUT is fitted with that layout.
- **Manager 1.2.0** is not yet run on the Cornell PC.

## Out of scope

- Fitting models and readout:
  - ML timing models.
  - Multi-timestamp estimators.
  - Per-channel walk tables.
  - Native PETsys walk correction.
- Acquisition: threshold changes (`vth_t1`).
- Separate studies:
  - The ±5 ns satellites.
  - Walk/skew drift tooling beyond the CTR check (the spec 008 gap).
- Other components:
  - Reconstruction changes.
  - New GUI widgets.
  - Manager actions.
  - Manufactured singles.
