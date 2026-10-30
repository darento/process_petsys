# Tasks — IMAS side timestamp from the top and earliest TIME hits

Status: in progress 2026-10-30 (T1–T3 done). [Approved spec](spec.md), [plan](plan.md).

**Conventions:**
- Each task cites `014-FR-n` and names its check before coding.
- Tests carry `@pytest.mark.fr("014-FR-n")`. Mark a test `slow` if its setup or call takes 5 s or more serially. Run new and changed tests with `-n 0 --slow-limit 5`.
- `real_data` tests name acquisitions relative to `PETSYS_DATA_DIR` and skip naming the missing variable or file.
- Record each check's selection, reason, command, platform and pytest summary line as `Verified:`. Tick a task only on success.
- No commits, pushes or release builds; the owner runs git.

## Core

- [x] **T1 — The rule** (FR-1, FR-3).
  - `src/imas/skew/timestamp.py`: `RULES`, `CANDIDATE_DTYPE`, `side_offset`, `check_rule`, `table_rule`.
  - **Done when** `tests/test_imas_timestamp.py`:
    - a side with an injected late top and an earlier neighbour gets the mean (`top_earliest_mean`) or the neighbour (`earliest`);
    - a side whose top is earliest gives `δ = 0`, and `top` never reads the candidates;
    - an earliest hit 20 ns before the top still counts (no window);
    - a candidate without a delay, or with a missing walk input, is excluded and counted;
    - a tie keeps the top;
    - the candidate's walk uses its own charge and share and the side's DOI and keV;
    - `check_rule` refuses a mismatch, naming both rules, and passes with `mismatch=True`.
  - Deviations:
    - `empty_candidates(n)` builds empty slots.
    - `earliest` with a walk LUT is refused (`RuleError`, clarify 4).
    - A rule other than `top` without candidates is refused.
    - Counts add `candidates clamped <input>` (plan risk: walk extrapolation).
  - Verified 2026-10-30 (Windows, this PC). Selection: the new module only; nothing imports it yet. `tests/test_imas_timestamp.py -n 0 --slow-limit 5`: 8 passed.
- [x] **T2 — Selection carries candidates** (FR-2).
  - `_select` fills up to 7 other TIME hits per side (sorted by raw lag); `FeatureBatch.candidates`; the holdout filter slices them; an int32 lag overflow is refused.
  - **Done when:**
    - `tests/test_imas_timestamp.py`: on a synthetic IMAS LDAT, the candidates equal a hand-built list (channels, lags, charges, empty slots −1, energy floor, other minimodules excluded).
    - Existing IMAS selection tests pass unchanged: `tests/test_imas_select.py tests/test_imas_walk.py tests/test_imas_cache.py -m "not real_data"` (names checked at the task), with ints/floats/walk bit-identical on the synthetic file.
    - Cornell consumers of `FeatureBatch` pass unchanged (selection recorded at the task).
  - Deviations:
    - The refusal covers both cases: more than 7 other TIME hits (duplicate hits on one channel), and a lag outside int32. It raises `ValueError` naming the record.
    - Scanned real data never triggers it: all of NECR frame 1 split 24 (628 k passing production, 700 k reference) has at most 7 candidates.
    - Candidates come from the kept hits, so reference mode (floor 0) can list more of them than production does.
    - Consequence: `select.py` and `kernels.py` source hashes are in the streamed-set fingerprint, so re-running a job on an existing work dir (e.g. `job_walk/`) is refused. This is existing behaviour for any selection edit; T10 streams into a new dir.
  - Verified 2026-10-30 (Windows, this PC).
    - Selection: shared `FeatureBatch` (`kernels.py`) and IMAS `select.py`, so their consumers: every IMAS test, Cornell timing/walk, core timing and golden provenance.
    - `tests/test_imas_timestamp.py -n 0 --slow-limit 5`: 10 passed.
    - `tests/test_imas_*.py tests/test_cornell_timing_*.py tests/test_cornell_walk.py tests/test_timing_*.py tests/test_golden_provenance.py -m "not real_data"`: 340 passed, 4 xfailed.
    - `PETSYS_DATA_DIR=C:/Users/dsanchez/Desktop/data`, `tests/test_imas_*.py -m real_data`: 2 passed.
    - Bit-identity against the HEAD selection: codes, ints, floats and walk identical on 500 k-record prefixes of NECR frame 30 split 22 and frame 1 split 24, production and reference modes (137,736 passing). One-off check, `$CLAUDE_JOB_DIR/tmp/bitident.py`.
- [x] **T3 — Observation rows carry candidates** (FR-2).
  - `OBSERVATION_RULE_DTYPE`, `canonical_observations(..., carry_candidates)`, `stream_observations(..., carry_candidates)` with its fingerprint entry; `carries_candidates` on the set.
  - **Done when:**
    - Tests: the swap moves the candidate blocks with their side; a stream with candidates round-trips; a set without them is refused for a rule other than `top`; `carry_candidates` without `carry_walk` is refused.
    - Cornell cache/stream goldens and fingerprints unchanged (`tests/test_cornell_*cache* tests/test_timing_stream*`, names checked at the task).
  - Deviations:
    - There is no fixed `OBSERVATION_RULE_DTYPE`. `observation_dtype(carry_walk, candidate_slots)` appends `cand_channel_s` / `cand_lag_s` (i4) and `cand_charge_s` (f4) per side. Its shape is the slot count, so the core cache stays scanner-neutral.
    - The core takes `candidate_slots` (IMAS passes `timestamp.SLOTS`). The IMAS adapter keeps `carry_candidates`.
    - With 0 slots the dtype and fingerprint are unchanged, which keeps the Cornell and spec 011/012 fingerprints.
    - A stream refuses an adapter whose batches carry no candidates of that slot count.
    - `timestamp.require_candidates(set, rule)` holds the refusal; `timestamp.observation_candidates(rows, "i"|"j")` reads one side.
  - Verified 2026-10-30 (Windows, this PC).
    - Selection: shared `src/timing/cache.py` and `stream.py`, so every IMAS, Cornell, core timing and Manager timing test, plus golden provenance.
    - `tests/test_imas_timestamp.py -n 0 --slow-limit 5`: 12 passed.
    - `tests/test_imas_*.py tests/test_cornell_*.py tests/test_timing_*.py tests/test_manager_timing*.py tests/test_golden_provenance.py -m "not real_data"`: 365 passed, 4 xfailed. This covers the Cornell cache and stream goldens (`test_cornell_timing_cache.py`, `test_timing_stream.py`), with no test edited.
    - `PETSYS_DATA_DIR=C:/Users/dsanchez/Desktop/data`, `tests/test_imas_*.py -m real_data`: 2 passed.
- [ ] **T4 — Correction and identity** (FR-3).
  - `correct_dt(..., rule, candidates_1, candidates_2)`, `correct_features(..., rule)`, `RuleAdjusted`, `timestamp_rule` read from skew and walk manifests.
  - **Done when:**
    - Under `top`, the output equals today's `correct_dt` bit for bit on the spec 012 fixtures.
    - Under the rule, it equals `side_offset` by hand.
    - `RuleAdjusted` partitions equal `correct_dt` on the same rows.
    - A table fitted under `top` applied under the rule is refused unless the mismatch option is set.
    - `tests/test_imas_walk.py` passes unchanged.

## Refit

- [ ] **T5 — Walk fit under the rule** (FR-4).
  - `imas-walk-fit/1` gains optional `timestamp_rule`, `rule_start` (skew + walk pair), `rule_rounds` and `rule_tolerance_ps`; outer fixed-point loop; manifest fields.
  - **Done when** on a synthetic two-branch bar with injected late tops (10 %, 600 ps) and a known walk:
    - the rule fit converges within the round limit;
    - it recovers the walk within the T1 (spec 012) criterion;
    - it beats the `top` fit on held-out rows;
    - rounds and changes are recorded;
    - an existing output is refused.
- [ ] **T6 — Skew solve under the rule** (FR-4).
  - Job keys `candidates`, `timestamp_rule`, `rule_start_skew`, `rule_passes`; EM passes on `RuleAdjusted`; output names with `_rule-mean`; provenance.
  - **Done when:**
    - On a synthetic job with injected delays and late tops, the passes converge and recover the delays within the spec 011 synthetic criterion.
    - `candidates` does not change the job fingerprint.
    - A rule without a candidate-carrying set is refused.
    - Existing outputs are never overwritten.
    - Spec 011/012 job tests pass unchanged.

## Evaluation

- [ ] **T7 — CTR and NECR under the rule** (FR-5).
  - Rule entries, `rule_tables`, counts per position, the `positions_not_worse` gate; NECR passes the entries through.
  - **Done when:**
    - On synthetic 22Na with injected late tops, the rule table improves and the tables share the same events.
    - Mismatch is recorded.
    - The per-position gate fails on a fixture where one position is made worse.
    - Spec 011/012/013 CTR and NECR tests pass unchanged.
- [ ] **T8 — Exports** (FR-6).
  - **Done when:**
    - Native and legacy exports refuse a rule skew LUT, naming the rule.
    - The walk export manifest carries `timestamp_rule`.
    - Existing export tests pass unchanged.
- [ ] **T9 — Diagnostics** (FR-7). **Done when** the PNGs are written headless from a synthetic CTR run.

## Real data and validation

- [ ] **T10 — Step 0 stream** (FR-2, FR-4). Run `run job_rule.json --work job_rule --observe-only` on the bar runs (D: attached).
  - **Done when:**
    - The per-run accepted counts equal `job_ship`'s.
    - Time, peak memory and set size are recorded.
- [ ] **T11 — Steps 1–3 round trip** (FR-4).
  - **Done when:**
    - Every fit and pass converges or reports its limit.
    - Held-out widths and changes are recorded.
    - No spec 011/012 output is touched.
- [ ] **T12 — Steps 4–5: CTR acceptance and NECR evidence** (FR-5, FR-7).
  - **Done when:**
    - The 4-position report gives the acceptance verdict (pooled gate and per-position rule).
    - The NECR run reports the new pair beside em2 + walk3 with its gates intact.
    - The diagnostics are written.
    - The owner reviews the verdict.
- [ ] **T13 — Validation** (all FR).
  - Full non-real suite plus `-m real_data`; every `--fr 014-FR-n` cited.
  - `IMAS_timing` page section for the new pair.
  - Spec `shipped` only when all completion criteria pass.
