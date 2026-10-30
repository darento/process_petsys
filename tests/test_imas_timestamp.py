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


# --- T3: observation rows carry candidates (FR-2) --------------------------------------------
@pytest.fixture(scope="module")
def candidate_files(tmp_path_factory, setup):
    """Diametric coincidences with 1-4 TIME hits per side at random lags, so swapped sides carry different
    candidates."""
    base = tmp_path_factory.mktemp("imas_candidates")
    rng = np.random.default_rng(23)
    records = []
    for _ in range(240):
        k, ring = int(rng.integers(0, 24)), int(rng.integers(0, 5))
        a, b = 24 * ring + k, 24 * (4 - ring) + (k + 12) % 24
        sides = []
        for sm, stamp in ((a, T0), (b, T0 - int(rng.integers(-900, 900)))):
            mm = int(rng.integers(0, 16))
            slabs = rng.permutation(8)[:int(rng.integers(1, 5))]
            t_ch, _ = channels(setup, sm, mm)
            time = {int(slabs[0]): float(rng.uniform(20, 30))}
            extra = [(stamp + int(rng.integers(-2000, 6000)), float(rng.uniform(0.5, 15)), t_ch[int(s)])
                     for s in slabs[1:]]
            sides.append(ldat_side(setup, sm, mm, time=time, stamp=stamp, extra=extra))
        records.append(tuple(sides))
    return [write(base / f"split_{i}.ldat", records[i::2]) for i in range(2)]


def _stream(directory, setup, paths, **options):
    from src.imas.skew import cache as ic
    from src.imas.skew.geometry import ImasGeometry
    return ic.stream_observations(directory, [{"index": i, "path": p} for i, p in enumerate(paths)],
                                  ImasGeometry.nominal(setup), config_path=CONFIG, encal_path=setup.encal_path,
                                  holdout=ic.all_train(range(len(paths))), energy_calibration_prefix={},
                                  partitions=3, buffer_rows=40, **options)


def _rows(observations):
    rows = np.concatenate([observations.partition(p) for p in range(observations.partitions)])
    return rows[np.lexsort((rows["record"], rows["file"]))]


@pytest.mark.fr("014-FR-2")
def test_observations_carry_candidates_with_their_side(tmp_path, setup, candidate_files):
    from src.timing import cache as core_cache
    walk_only = _rows(_stream(tmp_path / "walk", setup, candidate_files, carry_walk=True))
    carried_set = _stream(tmp_path / "cand", setup, candidate_files, carry_walk=True, carry_candidates=True)
    carried = _rows(carried_set)
    assert carried_set.carries_candidates and carried_set.candidate_slots == ts.SLOTS
    assert carried.dtype == core_cache.observation_dtype(True, ts.SLOTS)
    assert len(carried) > 150 and carried["swapped"].any() and not carried["swapped"].all()
    assert all(np.array_equal(carried[n], walk_only[n]) for n in core_cache.OBSERVATION_WALK_DTYPE.names)
    lookup = setup.lookup()
    features = {}
    for index, path in enumerate(candidate_files):
        for batch in sel.feature_batches(path, lookup, sel.Selection.production(lookup)):
            for k, record in enumerate(batch.column("record").tolist()):
                features[(index, record)] = batch.candidates[k]
    assert sum((c["channel"] >= 0).sum() for c in features.values()) > 300
    side_i, side_j = ts.observation_candidates(carried, "i"), ts.observation_candidates(carried, "j")
    assert side_i.dtype == ts.CANDIDATE_DTYPE and side_i.shape == (len(carried), ts.SLOTS)
    for n, row in enumerate(carried):
        f = features[(int(row["file"]), int(row["record"]))]
        i, j = (1, 0) if row["swapped"] else (0, 1)
        assert side_i[n].tobytes() == f[i].tobytes() and side_j[n].tobytes() == f[j].tobytes()
    assert (ts.observation_candidates(carried_set.partition(0)[:0], "i").shape == (0, ts.SLOTS))


@pytest.mark.fr("014-FR-2")
def test_sets_without_candidates_are_refused(tmp_path, setup, candidate_files):
    from src.timing import cache as core_cache
    walk_only = _stream(tmp_path / "walk", setup, candidate_files, carry_walk=True)
    with pytest.raises(core_cache.CacheMismatch, match="different inputs"):
        _stream(tmp_path / "walk", setup, candidate_files, carry_walk=True, carry_candidates=True)
    with pytest.raises(ValueError, match="carry_walk"):
        _stream(tmp_path / "plain", setup, candidate_files, carry_candidates=True)
    with pytest.raises(ts.RuleError, match="carry_candidates"):
        ts.require_candidates(walk_only, "top_earliest_mean")
    ts.require_candidates(walk_only, "top")
    carried = _stream(tmp_path / "cand", setup, candidate_files, carry_walk=True, carry_candidates=True)
    ts.require_candidates(carried, "top_earliest_mean")
    assert "candidates" not in walk_only.manifest["fingerprint"] and carried.manifest["fingerprint"]["candidates"][
        "slots"] == ts.SLOTS


# --- T4: correction under a rule and table identity (FR-3) -------------------------------------
def full_skew(setup, seed=5, provenance=None):
    rng = np.random.default_rng(seed)
    return DelayTable({int(c): float(rng.normal(0, 400)) for c in setup.time_channels.tolist()},
                      provenance=provenance)


def ruled_walk(rule):
    lut = walk_lut()
    lut.manifest = {"timestamp_rule": rule} if rule else {}
    return lut


def _candidate_features(setup, paths):
    lookup = setup.lookup()
    return [b for path in paths for b in sel.feature_batches(path, lookup, sel.Selection.production(lookup))
            if len(b.ints)]


@pytest.mark.fr("014-FR-3")
def test_top_rule_is_todays_correction(setup, candidate_files):
    skew = full_skew(setup)
    for features in _candidate_features(setup, candidate_files):
        for walk in (None, walk_lut()):
            before = iw.correct_features(features, skew, walk)
            after = iw.correct_features(features, skew, walk, rule="top")
            assert before[0].tobytes() == after[0].tobytes() and np.array_equal(before[1], after[1])
            assert before[2] == after[2] and "rule acts" not in after[2]


@pytest.mark.fr("014-FR-3")
def test_rule_correction_adds_side_offsets(setup, candidate_files):
    skew, walk, acted = full_skew(setup), walk_lut(), 0
    for features in _candidate_features(setup, candidate_files):
        top, status, _ = iw.correct_features(features, skew, walk)
        ruled, status_r, counts = iw.correct_features(features, skew, walk, rule="top_earliest_mean", mismatch=True)
        assert np.array_equal(status, status_r)
        offsets = []
        for s in (1, 2):
            side = {"q_top": features.column(f"q_top_{s}"), "q_time": features.column(f"q_time_{s}"),
                    "doi_ratio": features.column(f"doi_ratio_{s}"), "kev": features.column(f"energy_kev_{s}")}
            offsets.append(ts.side_offset(features.column(f"time_channel_{s}"), features.candidates[:, s - 1], side,
                                          skew, walk, "top_earliest_mean"))
        ok = status == 0
        np.testing.assert_array_equal(ruled[ok], top[ok] + (offsets[0][0] - offsets[1][0])[ok])
        assert counts["rule acts"] == int(np.count_nonzero(ok & (offsets[0][1] | offsets[1][1])))
        assert counts["timestamp rule"] == {"table_rule": "top", "rule": "top_earliest_mean", "mismatch": True,
                                            "tables": ["top", "top"]}
        acted += counts["rule acts"]
    assert acted > 10


@pytest.mark.fr("014-FR-3")
def test_tables_refused_under_another_rule(setup, candidate_files):
    (features, *_) = _candidate_features(setup, candidate_files)
    skew = full_skew(setup)
    with pytest.raises(ts.RuleError, match="'top'.*'top_earliest_mean'"):
        iw.correct_features(features, skew, walk_lut(), rule="top_earliest_mean")
    ruled_skew = full_skew(setup, provenance={"lut_sha256": "x", "timestamp_rule": "top_earliest_mean"})
    with pytest.raises(ts.RuleError, match="'top_earliest_mean'.*'top'"):
        iw.correct_features(features, ruled_skew, None)
    with pytest.raises(ts.RuleError, match="'top'.*'top_earliest_mean'"):        # the walk LUT's rule is checked
        iw.correct_features(features, ruled_skew, ruled_walk(None), rule="top_earliest_mean")
    iw.correct_features(features, ruled_skew, ruled_walk("top_earliest_mean"), rule="top_earliest_mean")
    assert ts.skew_rule(skew) == "top" and ts.skew_rule(ruled_skew) == "top_earliest_mean"


@pytest.mark.fr("014-FR-3")
def test_delay_table_keeps_the_lut_rule():
    from types import SimpleNamespace
    rows = np.array([(3, True), (4, False)], [("time_channel", "<i8"), ("apply", "?")])
    for provenance, rule in (({}, "top"), ({"timestamp_rule": "top_earliest_mean"}, "top_earliest_mean")):
        table = SimpleNamespace(rows=rows, _delay={3: 1.0}, manifest={"table": {"sha256": "s"},
                                                                       "provenance": provenance})
        delays = DelayTable.from_lut(table)
        assert ts.skew_rule(delays) == rule
        assert delays.provenance == ({"lut_sha256": "s"} | ({"timestamp_rule": rule} if provenance else {}))


@pytest.mark.fr("014-FR-3")
def test_rule_adjusted_partitions_match_correct_dt(tmp_path, setup, candidate_files):
    observations = _stream(tmp_path / "cand", setup, candidate_files, carry_walk=True, carry_candidates=True)
    skew, walk = full_skew(setup), walk_lut()
    view = ts.RuleAdjusted(iw.WalkCorrected(observations, walk), skew, walk, "top_earliest_mean")
    acted = 0
    for p in range(observations.partitions):
        raw, adjusted = observations.partition(p), view.partition(p)
        assert np.array_equal(raw["record"], adjusted["record"])          # every walk input present here
        expected, status, counts = iw.correct_dt(
            raw["dt_ps"], raw["channel_i"], raw["channel_j"], ts.observation_sides(raw, "i"),
            ts.observation_sides(raw, "j"), skew, walk, rule="top_earliest_mean", mismatch=True,
            candidates_1=ts.observation_candidates(raw, "i"), candidates_2=ts.observation_candidates(raw, "j"))
        b_i, _ = skew.lookup(adjusted["channel_i"])
        b_j, _ = skew.lookup(adjusted["channel_j"])
        assert (status == 0).all()
        assert np.abs(adjusted["dt_ps"] - b_i + b_j - expected).max() <= 1.0     # two roundings to integer ps
        acted += counts["rule acts"]
    assert view.counts()["train"]["rule acts"] == acted > 10
    assert view.counts()["train"]["rows"] == sum(len(observations.partition(p)) for p in range(observations.partitions))
    walk_only = _stream(tmp_path / "walk", setup, candidate_files, carry_walk=True)
    with pytest.raises(ts.RuleError, match="carry_candidates"):
        ts.RuleAdjusted(iw.WalkCorrected(walk_only, walk), skew, walk, "top_earliest_mean")


# --- T5: walk fit under the rule (FR-4) ------------------------------------------------------
RULE_FIT = {"bins": dict(iw.DEFAULT_BINS), "min_events": 50, "core_ps": 600.0, "iterations": 3,
            "branch_iterations": 2, "max_events": 10 ** 9, "timestamp_rule": "top_earliest_mean", "rule_rounds": 6,
            "rule_tolerance_ps": 0.5}


def truth_walk():
    edges = {"walk2d": [[0.0, 15.0, 22.0, 100.0], [0.0, 0.75, 1.01]], "doi": [[0.0, 3.0, 10.0]],
             "kev": [[0.0, 500.0, 1000.0]]}
    offsets = {"walk2d": [[120.0, 60.0], [0.0, -20.0], [-80.0, -120.0]], "doi": [15.0, -15.0], "kev": [10.0, -10.0]}
    events = {"walk2d": [[1, 1], [1, 1], [1, 1]], "doi": [1, 1], "kev": [1, 1]}
    return iw.WalkLut(edges, offsets, events, {b: [] for b in edges}, {})


def late_top_events(rng, n, truth, late_ps=600.0, late_share=0.1, sigma=150.0):
    """``bar_events`` arrays (rule layout) of skew-corrected two-branch events: a share of tops fire ``late_ps``
    late and then have an on-time neighbour; other sides have a neighbour 30 % of the time."""
    branch = np.where(rng.random(n) < 0.5, -1070.0, 1070.0)
    events = {"near": np.full(n, -1070.0, np.float32), "far": np.full(n, 1070.0, np.float32)}
    arrival = {"i": branch + rng.normal(0, 30, n), "j": np.zeros(n)}
    stamps = {}
    for s in ("i", "j"):
        charge, share = rng.uniform(10, 30, n), rng.uniform(0.55, 1.0, n)
        q_time, doi, kev = charge / share, rng.uniform(1.5, 5.5, n), rng.uniform(420, 600, n)
        late = rng.random(n) < late_share
        values, _ = iw.walk_inputs(charge, q_time, doi, kev)
        stamps[s] = arrival[s] + truth.correction(values)[0] + rng.normal(0, sigma, n) + np.where(late, late_ps, 0)
        neighbour = late | (rng.random(n) < 0.3)
        c_charge = rng.uniform(2, 9, n)
        c_values, _ = iw.walk_inputs(c_charge, q_time, doi, kev)
        c_stamp = arrival[s] + truth.correction(c_values)[0] + rng.normal(0, sigma, n)
        rel = np.full((n, ts.SLOTS), np.nan, np.float32)
        rel[:, 0] = np.where(neighbour, c_stamp - stamps[s], np.nan)
        cand_charge = np.zeros((n, ts.SLOTS), np.float32)
        cand_charge[:, 0] = c_charge
        events |= {f"charge_{s}": charge, f"share_{s}": share, f"doi_ratio_{s}": doi, f"kev_{s}": kev,
                   f"q_time_{s}": q_time, f"rel_{s}": rel, f"cand_charge_{s}": cand_charge}
    events["dt"] = stamps["i"] - stamps["j"]
    return {k: np.asarray(v, np.float32) for k, v in events.items()}


@pytest.mark.fr("014-FR-4")
def test_walk_fit_under_rule_converges_and_beats_top():
    rng = np.random.default_rng(31)
    truth = truth_walk()
    events = {"train": late_top_events(rng, 120_000, truth), "heldout": late_top_events(rng, 40_000, truth)}
    top_options = {k: v for k, v in RULE_FIT.items() if not k.startswith(("timestamp_rule", "rule_"))}
    top_lut, top_widths, _ = iw.fit_bar(events, top_options, {"fit_kind": "synthetic"})
    lut, widths, _ = iw.fit_bar(events, RULE_FIT, {"fit_kind": "synthetic"})
    rounds = lut.manifest["rule_rounds"]
    assert lut.manifest["timestamp_rule"] == "top_earliest_mean" and lut.manifest["rule_start_walk_sha256"] is None
    assert lut.manifest["rule_converged"] and 2 <= len(rounds) <= RULE_FIT["rule_rounds"] + 1
    assert rounds[-1]["rms_change_ps"] < 0.5 and rounds[1]["rule_acts_share"] > 0.1
    assert "rule_rounds" not in top_lut.manifest and "timestamp_rule" not in top_lut.manifest
    h1, h2 = iw.sides(events["heldout"])
    fitted, _, _ = iw.core.walk_difference(lut, h1, h2)
    injected, _, _ = iw.core.walk_difference(truth, h1, h2)
    assert np.sqrt(np.mean((fitted - injected) ** 2)) < 0.2 * np.sqrt(np.mean(injected ** 2))
    rule_after, top_after = widths["heldout"]["after"], top_widths["heldout"]["after"]
    assert rule_after["core_rms_ps"] < top_after["core_rms_ps"] - 5
    assert rule_after["core_rms_ps"] < widths["heldout"]["after (top)"]["core_rms_ps"]


@pytest.mark.fr("014-FR-4")
def test_rule_fit_options(tmp_path):
    import json
    base = {"schema": iw.FIT_SCHEMA, "job": "j.json", "work": "w", "skew_lut": "s.csv", "skew_lut_sha256": "a",
            "output": "o.json"}
    for extra, match in (({"timestamp_rule": "earliest"}, "skew-only"), ({"timestamp_rule": "mean"}, "top_earliest"),
                         ({"rule_rounds": 0}, "positive integer"), ({"rule_tolerance_ps": 0}, "positive"),
                         ({"timestamp_rule": "top_earliest_mean", "rule_start_walk": "w3.json"}, "go together"),
                         ({"rule_start_walk": "w3.json", "rule_start_walk_sha256": "b"}, "other than 'top'")):
        path = tmp_path / "fit.json"
        path.write_text(json.dumps(base | extra), encoding="utf-8")
        with pytest.raises(iw.core.WalkError, match=match):
            iw.load_fit(path)
    path.write_text(json.dumps(base | {"timestamp_rule": "top_earliest_mean", "rule_start_walk": "w3.json",
                                       "rule_start_walk_sha256": "b"}), encoding="utf-8")
    spec = iw.load_fit(path)
    assert spec["rule_start_walk"] == tmp_path / "w3.json" and spec["rule_rounds"] == 4
