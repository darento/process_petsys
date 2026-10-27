# Spec 012 — IMAS time-walk correction

Status: `in progress` (approved by owner 2026-10-27; T1 started 2026-10-27)

Constitution: [AGENTS.md](../../AGENTS.md). Workflow: [docs/prompts.md](../../docs/prompts.md). Builds on shipped [spec 011](../011-imas-skew-calibration/spec.md) (IMAS skew LUT from the rotating bar, global EM, four-position 22Na CTR) and follows shipped [spec 009](../009-cornell-timing-walk/spec.md) (Cornell walk LUT; its phantom fit, FR-7, is the model for a fit without a known source). Evidence (untracked): `docs/imas/IMAS_CTR/index.html` (spec 011 CTR report: 468 ps with the EM skew LUT, flat over four positions).

## Goal and population

Reduce the IMAS system CTR beyond the per-channel skew LUT by correcting each side's timestamp for its **time walk**, DOI and photon energy, as spec 009 does for Cornell (424 → 393 ps there). Deliver a frozen **walk LUT** beside the skew LUT; both apply to any later acquisition from quantities present in every event, without a source position.

**Input:**
- **Data:** PETsys compact **coincidence** LDAT, i.e. two detector hit populations per record, not singles.
- **Config and map:** `configs/imas_2DAQ.yaml`, `maps/imas2DAQ_map.yaml` (spec 011).
- **Units:** timestamps in ps; hit charges in PETsys QDC units (a.u.); keV only through the job's energy calibration.
- **Fit data:** the spec 011 ship runs (rotating bar, `D:\David_Skew`, runs 7, 8, 9 and 11, 1.18 TB), with the ship job's roles (energy calibration, train, held-out).
- **Validation data:** the 2026-10-19 22Na acquisitions at four positions (radius 0 / 198 mm × z 0 / 267 mm, five files each), never used in a fit.

**Selection:** as in spec 011 (FR-3): per side the highest-energy minimodule and its highest-charge TIME hit, both sides in 408–613 keV, opening angle >= 165°, `|dt|` inside the coincidence window.

**The four walk inputs per side** (the ingredients spec 011's selection already computes):
- **Timing-hit charge:** raw QDC charge of the timestamped TIME hit.
- **Light-sharing share:** that charge over the selected minimodule's summed TIME charge.
- **DOI ratio:** ENERGY sum over ENERGY maximum of the minimodule (`src/detector_features.py:calculate_DOI` semantics): a light-sharing ratio, not a depth in mm. IMAS has no DOI-limits file.
- **keV:** the minimodule's energy through the run's job calibration (spec 011 FR-4).

**Owner decisions, 2026-10-27:**

- **Fit source:** the bar runs only. No point source enters any fit; production recalibration needs only a bar run. The 22Na runs are validation at all four positions.
- **Round trip:** one, as spec 009:
  1. Walk fitted on the bar runs with the shipped EM skew LUT.
  2. EM skew re-solved on the bar runs with that walk applied.
  3. Walk refitted with the new skew.
  4. Final CTR at the four 22Na positions.

  The change at each step is reported. The delivered pair is the step-2 skew and step-3 walk, each with its own manifest naming the other.
- **Production:** the walk LUT file plus a tracked correction function in `src/` (as spec 011's `correct_timestamps`), applied by the CTR tool and exported. The untracked IMAS listmode script is not changed.
- **Energy calibration:** the skew-run job calibration (per run, from its `encal` split), for the walk fit and for the 22Na CTR (run 7's, as in spec 011). The 22Na spectra have no usable 511 keV peak of their own.

## Requirements

- **FR-1 — Walk inputs.**
  - One tracked function computes the four walk inputs per side from the selection's ingredients; the walk fit, the skew job with walk, the CTR and the correction function all use it.
  - Missing inputs (no TIME charge, zero sums, no calibration for the channel) are counted rejections, never zero corrections.
- **FR-2 — Walk LUT contract.**
  - Same model and file contract as spec 009 FR-2, with IMAS identities: a 2D table over charge × share plus 1D DOI-ratio and keV curves; quantile bin edges frozen from the fit events; a minimum-events rule (sparse bins take the nearest populated bin and are listed); subtractive per side, `t_corrected = t − b[TIME channel] − W[charge, share] − D[DOI ratio] − K[keV]` (ps); zero-mean gauge over the fit events; edge clamping, counted per input.
  - The manifest records the job and its SHA-256, the skew LUT SHA-256, config and map fingerprints, the energy calibrations, cuts, fit split, events and options. Existing outputs are never overwritten.
  - The format is shared with Cornell where it can be without changing Cornell's files or results.
- **FR-3 — Walk fit on the bar runs.**
  - Fitted from the job's train observations, without a known source: each event's expected `dt` is one of the two shell crossings (spec 011 FR-5); the walk fit takes the crossing assignment from the current model and re-assigns it after each walk update, or fits walk inside the EM with branch responsibilities (the plan chooses and records why).
  - Held-out splits never enter the fit; the report gives the held-out residual width before and after.
- **FR-4 — Skew re-solve with walk.**
  - A job option names a walk LUT; observations are corrected before the EM (and the pair cross-check). The walk inputs are carried by the streamed observations: an extended observation dtype recorded in the set's fingerprint, so a set without them is refused for walk, never silently reused (plan 2026-10-27: not a shared `SCHEMA_VERSION` bump, which would refuse every Cornell cache without changing its rows).
  - The skew LUT manifest records the walk LUT SHA-256; a skew LUT solved with walk is refused when applied without that walk LUT, and the other way round.
- **FR-5 — CTR with walk.**
  - The spec 011 CTR file may name a walk LUT. On the same events, the report gives uncorrected, legacy, skew-only and skew + walk, per position and pooled, with the round-trip summary.
  - **Acceptance:** at the four 22Na positions, the skew + walk model-free FWHM is better than skew-only by more than their combined uncertainty (spec 011 FR-10 rule); the Gaussian FWHM is reported.
- **FR-6 — Correction function and exports.**
  - A tracked function applies skew + walk to hits or `dt` from the event's walk inputs; table pairing, map and calibration identity are checked as in FR-4.
  - Events without a correction (no applied delay, missing walk input) follow an explicit, counted policy (`keep` uncorrected by default, or `drop`), as spec 009 FR-6.
  - A CLI exports the walk LUT with its manifest; the PETsys native table carries no walk (PETsys has none).
- **FR-7 — Diagnostics.**
  - PNGs of the curves and the 2D table with bin counts, residual-against-input profiles before and after (bar held-out and 22Na), per-position CTR before and after. Diagnostics, not validation evidence.

## Completion criteria

- **Synthetic fixtures:** injected walk, DOI and keV offsets on a bar-like (two-branch) fixture recover within declared uncertainty with the gauge kept; skew and walk together recover the injected delays and walk; clamping and missing inputs are counted.
- **Cornell:** spec 009's tests, goldens and baselines pass unchanged.
- **Real data (Windows, this PC, HDD attached):** the round trip completes with recorded timings and memory; FR-5 acceptance passes pooled over the four 22Na positions, reported per position.
- **Suites:** the required repository suites pass; missing acquisitions are explicit skips naming `PETSYS_DATA_DIR`.

## Clarify answers (owner, 2026-10-27)

1. **Fit keV window:** the job's 408–613 keV (no new stream cut; production windows in use, 408–613 and 450–570, sit inside). Spec 009 used a wider 300–700 keV fit window for Cornell.
2. **Acceptance:** model-free FWHM only, as spec 011 FR-10 (Gaussian reported). Spec 009 required both.
3. **DOI input:** the raw DOI ratio with global quantile bins (no per-slab normalisation; IMAS has no DOI-limits file).
4. **Table model:** spec 009's (20 × 10 charge × share, 20 DOI, 20 keV bins).

## Out of scope

- Point-source walk fits; per-channel or per-ASIC walk tables; ML timing models; multi-timestamp estimators.
- The IMAS listmode script, Manager actions, GUI widgets, reconstruction.
- The ±5 ns clock satellites and the ASIC channel-13 delay outliers (−2.6 to −5.7 ns) seen in spec 011's diagnostics.
- Acquisition settings (`vth_t1`). Manufactured singles.
