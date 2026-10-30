"""Spec 014: IMAS side timestamp from the top and earliest TIME hits."""
import numpy as np
import pytest

from imas_skew_helpers import CONFIG, T0, channels, mu_file, side as ldat_side, write
from src.imas.skew import select as sel
from src.imas.skew import timestamp as ts
from src.imas.skew import walk as iw
from src.imas.skew.setup import ImasSetup
from src.timing.lut import DelayTable

# top channels 10 and 20; candidates 11, 12, 21; 13 known without a delay; 99 not in the table
SKEW = DelayTable({10: 100.0, 11: 300.0, 12: -50.0, 20: 0.0, 21: 40.0}, missing=(13,))
W00, W01, W10, W11, DOI, KEV = 30.0, -70.0, 110.0, -20.0, 7.0, 3.0


def walk_lut():
    """charge bins [0, 10) [10, 100); share bins [0, 0.5) [0.5, 1]; one DOI and one keV bin (cancel per side)."""
    edges = {"walk2d": [[0.0, 10.0, 100.0], [0.0, 0.5, 1.0]], "doi": [[0.0, 10.0]], "kev": [[0.0, 1000.0]]}
    offsets = {"walk2d": [[W00, W01], [W10, W11]], "doi": [DOI], "kev": [KEV]}
    events = {"walk2d": [[1, 1], [1, 1]], "doi": [1], "kev": [1]}
    return iw.WalkLut(edges, offsets, events, {b: [] for b in edges}, {})


def candidates(*rows):
    """One event per row: [(channel, lag_ps, charge), ...] (at most SLOTS)."""
    out = ts.empty_candidates(len(rows))
    for e, row in enumerate(rows):
        for k, slot in enumerate(row):
            out[e, k] = slot
    return out


def side(n, q_top=15.0, q_time=20.0, doi_ratio=2.0, kev=511.0):
    return {name: np.full(n, value, np.float64) for name, value in
            (("q_top", q_top), ("q_time", q_time), ("doi_ratio", doi_ratio), ("kev", kev))}


@pytest.mark.fr("014-FR-1")
def test_late_top_gets_mean_or_earliest():
    # top 10 corrected -100; candidate 11 at -200 ps corrected -500: the earliest
    cands = candidates([(11, -200, 9.0), (12, 600, 5.0)])
    mean, acts, counts = ts.side_offset([10], cands, side(1), SKEW, None, "top_earliest_mean")
    assert acts.tolist() == [True] and mean.tolist() == [(-500.0 + 100.0) / 2]
    early, _, _ = ts.side_offset([10], cands, side(1), SKEW, None, "earliest")
    assert early.tolist() == [-400.0]
    assert counts["rule acts"] == 1 and counts["candidates"] == 2


@pytest.mark.fr("014-FR-1")
def test_top_earliest_and_top_rule_are_identity():
    cands = candidates([(11, 500, 9.0)], [])     # corrected 200 > -100; no candidate
    offset, acts, counts = ts.side_offset([10, 10], cands, side(2), SKEW, None, "top_earliest_mean")
    assert offset.tolist() == [0.0, 0.0] and not acts.any() and counts["rule acts"] == 0
    offset, acts, counts = ts.side_offset([10, 10], None, side(2), SKEW, walk_lut(), "top")
    assert offset.tolist() == [0.0, 0.0] and not acts.any() and counts["candidates"] == 0


@pytest.mark.fr("014-FR-1")
def test_no_window_far_earlier_hit_counts():
    offset, acts, _ = ts.side_offset([20], candidates([(21, -20_000, 9.0)]), side(1), SKEW, None,
                                     "top_earliest_mean")
    assert acts.tolist() == [True] and offset.tolist() == [(-20_040.0 - 0.0) / 2]


@pytest.mark.fr("014-FR-1")
def test_uncorrectable_candidates_excluded_and_counted():
    cands = candidates([(13, -5000, 9.0), (99, -5000, 9.0), (21, -5000, np.nan), (21, 100, 9.0)])
    offset, acts, counts = ts.side_offset([20], cands, side(1), SKEW, walk_lut(), "top_earliest_mean")
    assert not acts.any() and offset.tolist() == [0.0]
    assert counts["candidates"] == 4
    assert counts["candidates excluded (no delay)"] == 2
    assert counts["candidates excluded (walk input)"] == 1
    # an event whose top has no delay keeps offset 0 (its handling is the top's)
    offset, acts, _ = ts.side_offset([13], candidates([(21, -5000, 9.0)]), side(1), SKEW, None, "earliest")
    assert offset.tolist() == [0.0] and not acts.any()


@pytest.mark.fr("014-FR-1")
def test_tie_keeps_top():
    # top 10 corrected -100; candidate 12 at -150 ps corrected -150 + 50 = -100
    offset, acts, _ = ts.side_offset([10], candidates([(12, -150, 9.0)]), side(1), SKEW, None, "earliest")
    assert offset.tolist() == [0.0] and not acts.any()


@pytest.mark.fr("014-FR-1")
def test_candidate_walk_uses_own_charge_and_share():
    # top: charge 15, share 0.75 -> W11; candidate charge 5, share 5/20 -> W00 (the side's share would give W01)
    lag = -100
    offset, acts, counts = ts.side_offset([20], candidates([(21, lag, 5.0)]), side(1), SKEW, walk_lut(),
                                          "top_earliest_mean")
    c_top = -0.0 - (W11 + DOI + KEV)
    c_cand = lag - 40.0 - (W00 + DOI + KEV)
    assert acts.tolist() == [True]
    assert offset[0] == pytest.approx((c_cand - c_top) / 2)
    assert counts["candidates clamped charge"] == 0


@pytest.mark.fr("014-FR-3")
def test_rule_identity():
    assert ts.table_rule({}) == "top" and ts.table_rule({"provenance": {}}) == "top"
    assert ts.table_rule({"provenance": {"timestamp_rule": "top_earliest_mean"}}) == "top_earliest_mean"
    assert ts.table_rule({"timestamp_rule": "earliest"}) == "earliest"
    with pytest.raises(ts.RuleError, match="'top'.*'top_earliest_mean'"):
        ts.check_rule("top", "top_earliest_mean")
    assert ts.check_rule("top", "top_earliest_mean", mismatch=True) == {
        "table_rule": "top", "rule": "top_earliest_mean", "mismatch": True}
    assert ts.check_rule("top", "top")["mismatch"] is False
    with pytest.raises(ts.RuleError, match="one of"):
        ts.check_rule("top", "first")
    with pytest.raises(ts.RuleError, match="one of"):
        ts.table_rule({"timestamp_rule": "mean"})


@pytest.mark.fr("014-FR-1")
def test_earliest_is_skew_only_and_rules_need_candidates():
    with pytest.raises(ts.RuleError, match="skew-only"):
        ts.side_offset([10], candidates([]), side(1), SKEW, walk_lut(), "earliest")
    with pytest.raises(ts.RuleError, match="needs the candidates"):
        ts.side_offset([10], None, side(1), SKEW, None, "top_earliest_mean")


# --- T2: the selection carries candidates (FR-2) ---------------------------------------------
A, B = (0, 5), (60, 6)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    probe = ImasSetup.from_files(CONFIG)
    path = mu_file(tmp_path_factory.mktemp("mu") / "mu.tsv", {ch: 511.0 for ch in probe.time_channels.tolist()})
    return ImasSetup.from_files(CONFIG, path)


def select(tmp_path, setup, records):
    lookup = setup.lookup()
    path = write(tmp_path / "f.ldat", records)
    return list(sel.feature_batches(path, lookup, sel.Selection.production(lookup), batch_records=2))


@pytest.mark.fr("014-FR-2")
def test_selection_carries_other_time_hits_by_lag(tmp_path, setup):
    t_a, _ = channels(setup, *A)
    t_b, _ = channels(setup, *B)
    t_other, _ = channels(setup, A[0], 6)
    extra = [(T0 - 300, 4.0, t_a[6]),          # earlier, above the 0.2 floor
             (T0 - 900, 0.1, t_a[7]),          # below the floor: not a candidate
             (T0 - 500, 8.0, t_other[2])]      # another minimodule: not a candidate
    records = [(ldat_side(setup, *A, time={3: 12.0, 4: 6.0, 5: 9.0}, extra=extra), ldat_side(setup, *B)),
               (ldat_side(setup, *A, time={3: 12.0}), ldat_side(setup, *B, time={2: 30.0, 0: 1.0})),
               (ldat_side(setup, *A), ldat_side(setup, *B))]
    batches = select(tmp_path, setup, records)
    features = [b for b in batches if len(b.ints)]
    assert all((b.codes == 0).all() for b in batches)
    cands = np.concatenate([b.candidates for b in features])
    assert cands.shape == (3, 2, ts.SLOTS) and cands.dtype == ts.CANDIDATE_DTYPE

    def listed(event, s):
        row = cands[event, s]
        return [(int(c), int(l), float(q)) for c, l, q in zip(row["channel"], row["lag_ps"], row["charge"]) if c >= 0]
    assert listed(0, 0) == [(t_a[6], -300, 4.0), (t_a[4], 10, 6.0), (t_a[5], 20, 9.0)]
    assert listed(0, 1) == [(t_b[4], 10, 6.0)]
    assert listed(1, 0) == []
    assert listed(1, 1) == [(t_b[0], 10, 1.0)]
    assert (cands["channel"][1, 0] == -1).all() and (cands["lag_ps"][1, 0] == 0).all()
    tops = np.concatenate([b.column("time_channel_1") for b in features])
    assert tops.tolist() == [t_a[3]] * 3


@pytest.mark.fr("014-FR-2")
def test_candidates_beyond_slots_or_int32_are_refused(tmp_path, setup):
    t_a, _ = channels(setup, *A)
    full = {slab: 20.0 - slab for slab in range(8)}             # top slab 0 and 7 others
    duplicate = ldat_side(setup, *A, time=full, extra=[(T0 + 900, 1.0, t_a[1])])
    with pytest.raises(ValueError, match="more than 7 other TIME hits"):
        select(tmp_path, setup, [(duplicate, ldat_side(setup, *B))])
    far = ldat_side(setup, *A, extra=[(T0 + 3_000_000_000, 1.0, t_a[6])])
    with pytest.raises(ValueError, match="int32"):
        select(tmp_path, setup, [(far, ldat_side(setup, *B))])
    seven = ldat_side(setup, *A, time=full)
    (features,) = [b for b in select(tmp_path, setup, [(seven, ldat_side(setup, *B))]) if len(b.ints)]
    assert (features.candidates["channel"][0, 0] >= 0).sum() == 7
