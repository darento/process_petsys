# Spec 014 — IMAS side timestamp from the top and earliest TIME hits

Status: `in progress` (2026-10-30, T1; approved 2026-10-30, clarified 2026-10-30, drafted 2026-10-29)

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
  - Per side, every TIME hit of the selected minimodule that passes the hit energy floor (`en_min_ch`) is a candidate. Each candidate is corrected as if it were the timestamped hit: its own channel's skew delay, and the walk with its own charge as the charge input and its own share (its charge over the side's summed TIME charge; clarify 8). The DOI ratio and keV inputs stay the side's.
  - The **earliest** hit is the candidate with the smallest corrected time.
  - Three rules, named in every table manifest and report:
    - `top`: the top's corrected time (spec 011/012, unchanged).
    - `top_earliest_mean`: if the earliest is the top hit, the top's corrected time (identical to `top`); otherwise the plain mean of the top and earliest corrected times. No limit on how much earlier the earliest may be (clarify 1).
    - `earliest`: the earliest candidate's corrected time. For the skew-only tables (legacy, EM) in the FR-5 comparison only (clarify 4).
  - A candidate without an applied delay or with a missing walk input is excluded and counted. The top hit's handling is unchanged (spec 012 FR-6 policy).
  - Ties prefer the top hit.
- **FR-2 — Features and caches.**
  - The earliest hit depends on the tables, which change during the refit, so it is chosen when the tables are applied, not at selection. The selection carries, per side, every other candidate of the minimodule (at most 7): its TIME channel, its timestamp relative to the top's (ps) and its charge (clarify 7).
  - The observation dtype and the stream-set fingerprint record the carried candidates. A set built without them is refused for a rule other than `top`, never silently reused (as spec 012 FR-4).
  - Cornell features, caches, goldens and results are unchanged.
- **FR-3 — Correction function and table identity.**
  - `correct_dt` (or a sibling) applies FR-1 from the carried columns for one skew + walk pair.
  - Skew and walk LUT manifests record the timestamp rule they were fitted under (`top` for every existing table). Applying a table under a different rule is refused unless an explicit option names the mismatch; that option is used only for the frozen-table comparison in FR-5. The refusal message names both rules.
- **FR-4 — Refit under the rule.**
  - The spec 012 round trip on the bar runs with the rule applied throughout:
    1. walk fitted with the shipped em2 skew;
    2. skew re-solved with that walk;
    3. walk refitted.
  - Under `top_earliest_mean` a side's time involves two channels' delays and walks (½ each), while the shared EM and walk fit assume one per side. Each solve and fit is a fixed point (clarify 9): the solver sees the top's time adjusted by the rule's offset from the previous tables, re-computed after each pass until the tables stop changing (or a pass limit). Passes and changes are reported; the shared EM and walk kernels are unchanged.
  - Held-out widths and the change at each step are reported, as in spec 012 FR-3.
  - New outputs never overwrite the spec 011/012 deliverables.
- **FR-5 — CTR comparison and acceptance.**
  - The spec 011/012 CTR tool, on the same events, reports at least:
    - em2 + walk3 under `top`;
    - em2 + walk3 under the rule (frozen, mismatch option);
    - the refitted pair under the rule.
  - It reports these per 22Na position and pooled, with the share of events where the rule acts and the excluded candidates.
  - Under the mismatch option it also reports the frozen legacy and EM skew-only tables under `top` and `earliest` (clarify 4); no skew-only table is refitted (clarify 6).
  - **Acceptance:** pooled over the four 22Na positions, the refitted pair under the rule is better than em2 + walk3 under `top` by more than their combined uncertainty (spec 011 FR-10 rule, model-free FWHM; Gaussian reported). No position may be worse by more than its uncertainty.
  - The spec 013 NECR tool, given the new pair and rule, reports the NECR frames without breaking its gates. This is evidence, not acceptance. Spec 013's headline stays em2 + walk3 under `top`; the new pair is reported beside it (clarify 5).
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
  - Uncorrectable candidates and ties are counted and handled per FR-1; an earliest hit far earlier than the top (no window) still gives the mean; `earliest` returns the earliest candidate.
  - Mismatched table rules are refused (FR-3).
  - A cache without the new columns is refused (FR-2).
- **Cornell:** spec 008/009 tests, goldens and baselines pass unchanged.
- **IMAS regressions:** spec 011/012/013 tests pass. Under `top`, existing tables reproduce their recorded results exactly.
- **Real data (Windows, this PC, HDD attached):** the FR-4 round trip completes with recorded timings and memory, and FR-5 acceptance passes.
- **Suites:** the required repository suites pass; missing acquisitions are explicit skips naming `PETSYS_DATA_DIR`.

## Clarifications (owner, 2026-10-30)

1. **Earliest window:** no limit. Every earlier candidate counts (FR-1).
2. **Combination:** plain mean of the top and earliest corrected times.
3. **Interim use:** frozen em2 + walk3 under the rule only in the FR-5 comparison (mismatch option). The deliverable is the refitted pair.
4. **Skew-only tables:** in scope. The `earliest` rule is applied to the frozen legacy and EM tables in the FR-5 comparison.
5. **Spec 013:** keeps em2 + walk3 under `top` as its headline; the new pair is reported beside it as spec 014 evidence.
6. **Skew-only fit:** none. Legacy and EM are compared frozen; PETsys exports unchanged (FR-6).
7. **Candidates (plan, 2026-10-30):** carry every other TIME hit of the selected minimodule (≤ 7 per side). Measured on NECR frames 7 and 30: up to 8 TIME hits per minimodule, about 1.7 besides the top per side; EM2 delays spread up to 5.3 ns within a minimodule (median 0.3 ns), the walk2d block 0.4 ns, so any candidate can become the earliest.
8. **Share input (plan, 2026-10-30):** a candidate's share is its own charge over the side's summed TIME charge, as `walk_inputs` derives it and as the exploratory gain was measured.
9. **Refit method (plan, 2026-10-30):** fixed point on the top's adjusted time; shared EM and walk kernels unchanged.

## Out of scope

- Averaging more TIME hits (tested, no gain); ML timing models; per-channel or per-ASIC walk tables.
- The high-rate CTR degradation (`rate_study.html`), which this rule does not address: the share of affected events barely changes with rate.
- The IMAS listmode script, Manager actions, GUI widgets, reconstruction, acquisition settings (`vth_t1`), manufactured singles.
