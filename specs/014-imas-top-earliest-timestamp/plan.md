# Plan 014 — IMAS side timestamp from the top and earliest TIME hits

Status: planned 2026-10-30 for the approved [spec 014](spec.md). Constitution: [AGENTS.md](../../AGENTS.md). [Ordered tasks](tasks.md).

## Inputs seen at planning (2026-10-30)

- **Shipped tables:**
  - `em` = `IMAS_work/job_ship/lut_em.csv` (`8eadd8c7…`, rule `top`);
  - `em2` = `IMAS_work/job_walk/lut_em_walk-dd68b558604c.csv` (`daa3b251…`, solved with walk1);
  - `walk3` = `IMAS_work/walk/walk3.json` (`5e8ac010…`);
  - legacy `IMAS/skew_IMAS.txt`.
- **Bar observation sets:** `job_walk/` carries walk inputs (`OBSERVATION_WALK_DTYPE`, ~151 B/row, 48.7 M rows) but no other TIME hits, so a new stream is required (FR-2).
- **Candidates (measured, NECR frames 7 and 30, 400 k records each):**
  - TIME hits per selected minimodule side: 1 (17 %), 2 (32 %), 3 (33 %), 4 (11 %), 5–8 (6 %); about 1.7 besides the top.
  - Of the other hits, 4 % are raw-earlier than the top; 0.4–0.8 % more than 2 ns earlier.
  - EM2 delay spread within a minimodule: median 282 ps, p95 906 ps, max 5.3 ns. walk3 block ranges: walk2d 411 ps, doi 59 ps, keV 77 ps.
- **Disk:** C: 140 GB free. D: (bar runs, HDD) must be attached for the stream.

## Architecture

1. **`src/imas/skew/timestamp.py` — the rule** (new, FR-1, FR-3).
   - `RULES = ("top", "top_earliest_mean", "earliest")`.
   - `side_offset(top, candidates, side, skew, walk, rule)` returns the side's time relative to the top's corrected time (`δ`, float ps), plus per-event flags (rule acts, candidates excluded).
     - `top` gives `δ = 0` everywhere, without reading the candidates.
     - Every candidate is corrected as the top: `c = lag − b[channel] − W(charge, share = charge / q_time, doi, keV)`. The top's own value is `0 − b_top − W_top`.
     - The earliest is the candidate with the smallest `c`; on a tie, the top.
     - `top_earliest_mean`: `δ = (c_earliest − c_top) / 2`; `earliest`: `δ = c_earliest − c_top`.
   - A candidate with no applied delay or a missing walk input is excluded and counted.
   - Vectorised over the 7 slots in NumPy; masked slots have channel −1.
   - `check_rule(table_rule, rule, *, mismatch=False)` raises with both names unless `mismatch` is set.
   - `table_rule(manifest)` returns the manifest's `timestamp_rule`, absent meaning `top`.
2. **Selection carries candidates** (`src/imas/skew/select.py`, `src/timing/kernels.py`, FR-2).
   - `_select` also fills `out_c` per passing record: per side, up to 7 other TIME hits of the selected minimodule (energy ≥ floor), sorted by raw lag. Each slot holds:
     - the channel (−1 when empty);
     - `lag = t_hit − t_top` (int32 ps; a lag outside int32 is refused, never wrapped);
     - the charge (float32).
   - `FeatureBatch` gains an optional `candidates` (structured `(n, 2, 7)` array; None where not computed). The existing columns are bit-identical.
   - The holdout filter in `feature_batches` slices it with the rest.
   - The feature cache (`FEATURE_DTYPE`, `SCHEMA_VERSION` 3, shared with Cornell) does not store candidates. A feature-cache-built set therefore never carries them, and a rule other than `top` refuses it.
3. **Observation rows carry candidates** (`src/timing/cache.py`, `src/timing/stream.py`, FR-2).
   - `OBSERVATION_RULE_DTYPE` = `OBSERVATION_WALK_DTYPE` plus, per canonical side `i`/`j`: `cand_channel_s` (i4 × 7), `cand_lag_s` (i4 × 7) and `cand_charge_s` (f4 × 7). That is 168 B, about 319 B per row.
   - `canonical_observations(..., carry_candidates=True)` swaps the candidate blocks with the sides. Lags are relative to their own side's top, so they need no sign change.
   - `stream_observations(..., carry_candidates=True)` requires `carry_walk`. Its fingerprint records the dtype and `"candidates": {"slots": 7, "order": "raw lag"}`. A set without them is refused for a rule other than `top`.
   - Cornell keeps `OBSERVATION_DTYPE`, its fingerprints and its goldens. The new arguments default to off, so default fingerprint entries are unchanged.
4. **Correction and views** (`src/imas/skew/walk.py`, `timestamp.py`, FR-3).
   - `correct_dt(..., rule="top", candidates_1=None, candidates_2=None)`: the top-corrected `dt` plus `δ_1 − δ_2`. `correct_features(..., rule=...)` reads `features.candidates`.
   - The counts gain `rule acts`, `candidates excluded (no delay)` and `candidates excluded (walk input)`. Under `top` the output is bit-identical to today's.
   - `RuleAdjusted(observations, skew, walk, rule)` is a partition view like `WalkCorrected`. It returns rows whose `dt_ps` is the top-only correction plus `rint(δ_i − δ_j)`, so the EM and walk fit see the rule through the top's channel and inputs (the fixed point below). The rule's counts are kept per role.
5. **Fixed-point refit** (`job.py`, `walk.py`, FR-4).
   - **Walk fit:** an outer loop around `core.fit_two_branch`.
     - Round 0 computes `δ` from the starting pair. Each round fits a walk LUT on the skew-corrected `dt` plus `δ_i − δ_j` (top inputs), then recomputes `δ` with the new LUT and the frozen skew.
     - It stops when the per-event correction changes by less than `rule_tolerance_ps` (RMS 0.5 ps) or after `rule_rounds` (default 4).
     - Recorded per round: RMS/max change, share acting, branch flips.
   - **Skew solve:** job key `timestamp_rule` plus `rule_start_skew` (the LUT that gives the first `δ`).
     - Each pass runs the job's EM (only `em`; the pair estimator is skipped under a rule) on `RuleAdjusted(WalkCorrected(...))` with the previous pass's delays, then recomputes `δ`.
     - It stops when the applied delays change by less than 0.5 ps RMS, or after `rule_passes` (default 3).
     - Per pass: RMS/max delay change and held-out widths.
   - Neither the shared EM (`src/timing/em.py`) nor the shared walk fit changes.
6. **CTR** (`ctr.py`, `necr.py`, FR-5).
   - Entries in `luts`, `walks` and `legacy` gain an optional `rule` (default the table's manifest rule; legacy and every existing table are `top`).
   - New `rule_tables` = `{name: {"base": <table name>, "rule": <rule>}}` evaluates a frozen table under another rule (mismatch option, recorded), e.g. `em2_walk3@mean`, `em@earliest`, `legacy@earliest`.
   - Same events as now: candidate exclusions never drop an event.
   - Per table and position: share acting, candidates excluded.
   - `walk_acceptance` gains a per-position rule: no position worse by more than its uncertainty (`positions_not_worse`, gate).
   - NECR (`necr.py`) shares `frozen_inputs` and the per-table correction, so the same entries apply there. Its gates and headline are unchanged.
7. **Exports** (`outputs.py`, `walk.export_walk`, FR-6).
   - Native and legacy exports refuse a skew LUT whose rule is not `top`.
   - The walk export manifest copies `timestamp_rule`.
8. **Diagnostics** (`src/imas/skew/timestampplots.py`, FR-7). Headless Agg PNGs from the CTR's histograms:
   - the earliest-minus-top corrected offset distribution (per position, pooled);
   - CTR per offset bin under `top` and under the rule;
   - share acting per minimodule.

### Alternatives rejected

- **Pick the earliest at selection** (the draft FR-2): the earliest depends on skew and walk, which the refit changes, so a stored choice would be stale after the first pass.
- **3 raw-earliest slots per side:** about 36 % smaller rows, but not exact. Exactness was chosen (owner, clarify 7).
- **Fractional multi-channel sides in the EM:** an exact joint fit, but it changes the shared EM kernels (Cornell's estimator too). Fixed point chosen (owner, clarify 9): the rule acts on about 10 % of events with weight ½, so each pass shrinks the inconsistency by roughly 20×.
- **Re-solving only on events where the rule does not act:** that selection depends on the delays and biases the late-top channels.

## Data shapes

- **Candidates in features:** `np.dtype([("channel", "<i4"), ("lag_ps", "<i4"), ("charge", "<f4")])`, shape `(n, 2, 7)`.
- **Observation row with candidates:** about 319 B. The bar job's 48.7 M rows are about 15.5 GB on C:, in a new work dir (`job_rule/`); `job_walk/` is kept.
- **Manifests:**
  - skew LUT provenance `timestamp_rule`, `rule_passes` and `rule_start_skew_sha256`;
  - walk LUT manifest `timestamp_rule`, `rule_rounds` and `rule_start`.
  - Absent means `top`.

## Real-data round trip (FR-4, FR-5)

| Step | Command | Output | Expected time |
| --- | --- | --- | --- |
| 0 | `run job_rule.json --work job_rule --observe-only` (`job_ship` plus `"candidates": true`) | candidate-carrying observation sets, same events as `job_ship` | ~1 h (HDD) |
| 1 | `walk walk_rule1.json` (skew `em2`, rule `top_earliest_mean`, start `em2`+`walk3`) | `walk_rule1.json` | ~10–20 min |
| 2 | `run job_rule_walk.json --work job_rule` (`walk_lut` walk_rule1, rule, start `em2`) | `lut_em_walk-<sha>_rule-mean.csv` (`em3`) | 1–3 h (≤ 3 EM passes) |
| 3 | `walk walk_rule3.json` (skew `em3`) | `walk_rule3.json` | ~10–20 min |
| 4 | `ctr ctr_rule_4pos.json` | legacy, em, em2+walk3 under `top`; legacy@earliest, em@earliest, em2+walk3@mean; em3+walk_rule3 under the rule; acceptance | ~15 min |
| 5 | `necr necr_rule.json` (`necr_present5` plus the new pair) | NECR evidence | ~20 min |

- Step 0 checks that the per-run accepted counts equal `job_ship`'s.
- The job fingerprint is unchanged by `candidates`, which only selects the row dtype (as `carry_walk` did). The set fingerprint protects the rows.

## FR → component map

| Requirement | Component | Check |
| --- | --- | --- |
| FR-1 | `timestamp.side_offset` | synthetic late top gives the mean; top earliest is bit-identical; far-earlier hit with no window; ties; exclusions counted; `earliest` |
| FR-2 | `_select` candidates, `FeatureBatch.candidates`, `OBSERVATION_RULE_DTYPE`, stream fingerprint | synthetic LDAT candidates by hand; existing features bit-identical; swap; refusal of sets without candidates; Cornell cache goldens unchanged |
| FR-3 | `correct_dt(rule=)`, `RuleAdjusted`, `check_rule`, manifests | `top` equals today's output; rule mismatch refused naming both rules; option recorded |
| FR-4 | walk outer loop, job passes | synthetic bar with injected late tops: the fixed point converges and recovers delays/walk; outputs never overwrite |
| FR-5 | `ctr.py` `rule_tables`, rule entries, per-position gate; `necr.py` | synthetic 22Na with injected late tops: rule improves; same events; counts |
| FR-6 | exports | rule skew refused for native/legacy; walk export manifest has the rule |
| FR-7 | `timestampplots.py` | PNGs written headless |
| Real | steps 0–5 | timings, memory, acceptance |

## Risks

- **Walk extrapolation:** walk3 was fitted on top hits, so low-charge candidates fall in its low-charge bins (clamped, counted). The frozen comparison inherits this. The refit fits on top hits only (the rule enters through `δ`), so low-charge candidates stay extrapolated. The clamp counts are reported per candidate.
- **Ordering includes walk:** the spec orders candidates after skew and walk. The NECR exploratory work ordered them after skew only, so the FR-5 frozen comparison may differ slightly from the NECR evidence.
- **Pass cost:** each EM pass is about 1 h. If 3 passes do not converge, the report says so and the owner decides.
- **Row size:** 319 B/row doubles the partition memory in the EM (~1 GB per partition at 16 partitions). Raising `partitions` to 32 is the fallback.
