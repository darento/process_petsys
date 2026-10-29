# Spec 014 — IMAS side timestamp from the top and earliest TIME hits

Status: `draft` (2026-10-29)

Constitution: [AGENTS.md](../../AGENTS.md). Workflow: [docs/prompts.md](../../docs/prompts.md). Builds on shipped [spec 011](../011-imas-skew-calibration/spec.md) (selection, skew LUT, FR-10 acceptance rule) and shipped [spec 012](../012-imas-timing-walk/spec.md) (walk LUT, round trip, `correct_dt`). Spec 012 put multi-timestamp estimators out of scope; this spec adds one.

Evidence (untracked):
- `docs/imas/IMAS_NECR_CTR/rate_study.html`, section "Side finding: which TIME hit to timestamp".
- Scripts `scripts_imas/necr_time_hit.py` and `scripts_imas/necr_time_avg.py`.
- Outputs `time_hit/` and `time_avg/`.

## Goal and population

Each side's timestamp is now the highest-charge TIME hit of the selected minimodule (the **top** hit; spec 011 FR-3). In about 10 % of events another TIME hit of that minimodule fires earlier, after its own skew and walk correction, and those events time badly.

**Goal:** time each side with the mean of the top hit and the earliest hit, when the earliest is a different hit. Then refit the skew and walk tables under that rule, so the rule and the tables are consistent.

**Input:**
- **Data:** PETsys compact **coincidence** LDAT, i.e. two detector hit populations per record, not singles.
- **Config and map:** `configs/imas_2DAQ.yaml`, `maps/imas2DAQ_map.yaml` (spec 011).
- **Units:** timestamps in ps; hit charges in PETsys QDC units (a.u.); keV only through the job's energy calibration.
- **Fit data:** the spec 011/012 bar runs (`D:\David_Skew`, runs 7, 8, 9 and 11), with the ship job's roles.
- **Validation data:** the 2026-10-19 22Na acquisitions at four positions (spec 011/012), never used in a fit. Additional evidence: the spec 013 NECR frames.

**Selection:** unchanged from spec 011 FR-3: minimodule, top TIME hit, keV window, opening angle and coincidence window. The top hit still sets the slab (transaxial position), the minimodule, the walk inputs other than charge, and the keV. Only the side timestamp changes.

**Exploratory result** (frozen em2 + walk3 tables fitted on top hits, same events, same LOR; NECR phantom, model-free FWHM):

| Frame | kBq/ml | top (now) | top + earliest | Events where the rule acts |
|---|---|---|---|---|
| 7 | 5.31 | 672 | 661 | 11.1 % |
| 20 | 1.00 | 491 | 483 | 10.2 % |
| 25 | 0.52 | 470 | 459 | 10.0 % |
| 30 | 0.28 | 453 | 444 | 10.0 % |

- **Significance:** the paired bootstrap SD of each difference is 1–2.5 ps.
- **Affected events alone:** at frame 30 they improve from 586 to 434 ps.
- **Other choices tested:**
  - Taking the earliest hit outright is worse than the mean with em2 + walk3. It is better with the legacy and EM skew-only tables (−7 to −14 ps).
  - Averaging every TIME hit within ±0.2–1 ns of the top gains nothing: the other hits are mostly late triggers (median 5.3 ns after the top, 12 % within ±0.5 ns).
- **Known limitation:** the walk LUT was fitted on top hits, so the earliest hit's walk is extrapolated in its charge.

## Requirements

- **FR-1 — Side timestamp rule.**
  - Per side, every TIME hit of the selected minimodule that passes the hit energy floor (`en_min_ch`) is a candidate. Each candidate is corrected as if it were the timestamped hit: its own channel's skew delay, and the walk with its own charge as the charge input. The share, DOI ratio and keV inputs stay the side's.
  - The **earliest** hit is the candidate with the smallest corrected time.
  - If the earliest is the top hit, the side time is the top's corrected time (identical to now).
  - Otherwise the side time is the mean of the top and earliest corrected times. The rule SHALL take a lower bound on how much earlier the earliest may be (`earliest_window_ps`; clarify 1). An earlier hit beyond it is ignored and counted.
  - A candidate without an applied delay or with a missing walk input is excluded and counted. The top hit's handling is unchanged (spec 012 FR-6 policy).
  - Ties prefer the top hit.
- **FR-2 — Features and caches.**
  - The selection carries, per side, the earliest candidate's channel, raw timestamp and charge beside the top's. With no earlier candidate, it carries the top's own values.
  - The observation dtype and the stream-set fingerprint record the timestamp rule. A cache built without the earliest-hit columns is refused for the new rule, never silently reused (as spec 012 FR-4).
  - Cornell features, caches, goldens and results are unchanged.
- **FR-3 — Correction function and table identity.**
  - `correct_dt` (or a sibling) applies FR-1 from the carried columns for one skew + walk pair.
  - Skew and walk LUT manifests record the timestamp rule they were fitted under (`top` for every existing table). Applying a table under a different rule is refused unless an explicit option names the mismatch; that option is used only for the frozen-table comparison in FR-5. The refusal message names both rules.
- **FR-4 — Refit under the rule.**
  - The spec 012 round trip on the bar runs with the rule applied throughout:
    1. walk fitted with the shipped em2 skew;
    2. skew re-solved with that walk;
    3. walk refitted.
  - Held-out widths and the change at each step are reported, as in spec 012 FR-3.
  - New outputs never overwrite the spec 011/012 deliverables.
- **FR-5 — CTR comparison and acceptance.**
  - The spec 011/012 CTR tool, on the same events, reports at least:
    - em2 + walk3 under `top`;
    - em2 + walk3 under the rule (frozen, mismatch option);
    - the refitted pair under the rule.
  - It reports these per 22Na position and pooled, with the share of events where the rule acts, ignored earlier hits and excluded candidates.
  - **Acceptance:** pooled over the four 22Na positions, the refitted pair under the rule is better than em2 + walk3 under `top` by more than their combined uncertainty (spec 011 FR-10 rule, model-free FWHM; Gaussian reported). No position may be worse by more than its uncertainty.
  - The spec 013 NECR tool, given the new pair and rule, reports the NECR frames without breaking its gates. This is evidence, not acceptance.
- **FR-6 — Exports.**
  - The rule is software-only, like em2 + walk3 (spec 012 spec change): PETsys native and legacy skew exports carry no rule and stay spec 011's EM LUT.
  - The walk LUT export records the rule in its manifest.
- **FR-7 — Diagnostics.**
  - PNGs of the earliest-minus-top corrected offset distribution, per position and pooled.
  - CTR against the offset bin, before and after the rule.
  - The share of events where the rule acts, per minimodule.
  - These are diagnostics, not validation evidence.

## Completion criteria

- **Synthetic fixtures:**
  - A side with an injected late top timestamp and an earlier neighbour gets the mean.
  - A side whose top hit is earliest is bit-identical to `top`.
  - Hits beyond the window, uncorrectable candidates and ties are counted and handled per FR-1.
  - Mismatched table rules are refused (FR-3).
  - A cache without the new columns is refused (FR-2).
- **Cornell:** spec 008/009 tests, goldens and baselines pass unchanged.
- **IMAS regressions:** spec 011/012/013 tests pass. Under `top`, existing tables reproduce their recorded results exactly.
- **Real data (Windows, this PC, HDD attached):** the FR-4 round trip completes with recorded timings and memory, and FR-5 acceptance passes.
- **Suites:** the required repository suites pass; missing acquisitions are explicit skips naming `PETSYS_DATA_DIR`.

## Clarify questions (open)

1. **Earliest window:** no limit, or ignore hits more than 500 ps earlier? On the NECR frames they give the same width (444 vs 444 ps at frame 30; 661 vs 667 at frame 7). A limit guards against dark counts.
2. **Combination:** the plain mean of top and earliest, or a weighted form? The charge²-weighted mean within ±500 ps matched the plain mean (444 ps at frame 30).
3. **Interim use:** may the frozen em2 + walk3 be used under the rule before the refit (gain about 8–11 ps on NECR), or only the refitted pair?
4. **Legacy and EM skew-only tables:** for these the earliest hit outright did better than the mean. Is that variant in scope, or only the em2 + walk3 lineage?
5. **Spec 013:** after shipping, does the NECR CTR report switch to the new pair, or keep em2 + walk3 as its frozen headline table?

## Out of scope

- Averaging more TIME hits (tested, no gain); ML timing models; per-channel or per-ASIC walk tables.
- The high-rate CTR degradation (`rate_study.html`), which this rule does not address: the share of affected events barely changes with rate.
- The IMAS listmode script, Manager actions, GUI widgets, reconstruction, acquisition settings (`vth_t1`), manufactured singles.
