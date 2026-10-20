"""Spec 009: Cornell time-walk inputs, walk LUT and fits."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.cornell import listmode as lm
from src.cornell.inputs import describe_legacy, load_calibration, load_limits, load_processing_config
from src.cornell.skew import walk
from src.cornell.skew.ingest import CompiledLookup, LegacySetup, compiled_batches
from src.mapping_generator import ChannelType
from src.petsys_manager.contracts import DataFormat, Population
from cornell_timing_helpers import CONFIG, INVALID_SM, MISSING_SM, ROOT, T0, channels, make_setup, write
from manager_helpers import METADATA

LM_SEED = 7
TAB, NL = "\t", "\n"


def _time_slabs(setup):
    """(TIME channel, slab) keys of the map: each TIME channel covers slabs 2p and 2p + 1."""
    return [(ch, slab) for ch, types in setup.chtype_map.items() if ChannelType.TIME in types
            for slab in (2 * setup.local_map[ch][2], 2 * setup.local_map[ch][2] + 1)]


def _limits(path, setup, value, skip=()):
    rows = [f"({ch}, {slab}){TAB}{value[0]}{TAB}{value[1]}" for ch, slab in _time_slabs(setup)
            if (ch, slab) not in skip]
    path.write_text(NL.join(rows) + NL, encoding="utf-8")
    return path


def _encal(path, setup):
    """Per-slab factors mu = 511 (keV = raw sum); SM MISSING_SM absent, SM INVALID_SM non-positive."""
    rows = [f"({ch}, {slab}){TAB}{0.0 if setup.sm_mM_map[ch][0] == INVALID_SM else 511.0}{TAB}1.0"
            for ch, slab in _time_slabs(setup) if setup.sm_mM_map[ch][0] != MISSING_SM]
    path.write_text(f"ID(t_ch, slab){TAB}mu{TAB}sigma{NL}" + NL.join(rows) + NL, encoding="utf-8")
    return path


def _records(setup, count, seed):
    """Opposed sides with 1-3 TIME hits (adjacent positions resolve the slab) and 4-8 ENERGY hits near 511."""
    rng = np.random.default_rng(seed)
    out = []

    def one(sm, stamp):
        t_ch, e_ch = channels(setup, sm, int(rng.integers(0, 16)))
        p = int(rng.integers(0, 7))
        n_time = int(rng.choice([1, 2, 2, 2, 3]))
        hits = [(stamp + int(rng.integers(0, 50)), float(rng.uniform(5, 30)), t_ch[min(p + k, 7)])
                for k in range(n_time)]
        n_energy = int(rng.integers(4, 9))
        for i, value in zip(rng.choice(len(e_ch), n_energy, replace=False),
                            rng.dirichlet(np.ones(n_energy)) * rng.normal(511, 40)):
            hits.append((stamp + 500, float(value), e_ch[i]))
        return [hits[i] for i in rng.permutation(len(hits))]

    for _ in range(count):
        cassette, z = int(rng.integers(0, 10)), int(rng.integers(0, 3))
        sm1, sm2 = 3 * cassette + z, 3 * ((cassette + 5) % 10) + z
        stamp = T0 + int(rng.integers(0, 10 ** 9))
        out.append((one(sm1, stamp), one(sm2, stamp - int(rng.integers(-2000, 2000)))))
    return out


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    base = tmp_path_factory.mktemp("walk")
    encal = _encal(base / "walk.encal", make_setup(base))
    setup = LegacySetup.from_files(CONFIG, encal, root=ROOT)
    ldat = write(base / "walk_coinc_0.ldat", _records(setup, 3000, seed=11))
    cog = _limits(base / "cog.txt", setup, (-1000.0, 1000.0))
    doi = _limits(base / "doi.txt", setup, (1.0, 8.0))
    modules = sorted(set(map(tuple, setup.sm_mM_map.values())))
    (base / "regions.tsv").write_text("".join(f"({sm}, {mm}){TAB}0{TAB}(0, 0){NL}" for sm, mm in modules),
                                      encoding="utf-8")
    (base / "pairs.txt").write_text("0 0 0" + NL, encoding="utf-8")
    return {"base": base, "setup": setup, "ldat": ldat, "encal": encal, "cog": cog, "doi": doi,
            "regions": base / "regions.tsv", "pairs": base / "pairs.txt"}


def _listmode(files):
    config = load_processing_config(CONFIG, processing_root=ROOT, action="listmode")
    mapping = config.mapping
    ctx = lm.ListmodeContext.build(config, load_calibration(Path(files["encal"]), mapping),
                                   load_limits(Path(files["cog"]), mapping, kind="cog"),
                                   load_limits(Path(files["doi"]), mapping, kind="doi"),
                                   lm.load_pair_map(files["pairs"]), lm.load_region_map(files["regions"], mapping),
                                   METADATA)
    return ctx, mapping


def _chunks(ctx, mapping, ldat):
    descriptor = describe_legacy([Path(ldat)], format=DataFormat.COMPACT, population=Population.COINCIDENCE,
                                 confirmed=True)[0]
    return lm._chunks(descriptor, ctx, mapping, 16, 1000, [0])


def listmode_capture(files, ldat):
    """Listmode's walk inputs of its written events (keyed by record index), its output bytes (per batch, as
    ``_process_file`` writes them) and rejections."""
    ctx, mapping = _listmode(files)
    rejected, observations, flags, capture = dict.fromkeys(lm.REJECTIONS, 0), dict.fromkeys(lm.OBSERVATIONS, 0), {}, []
    np.random.seed(LM_SEED)
    start, written = 0, []
    for chunk in _chunks(ctx, mapping, ldat):
        before = len(capture)
        written.append(lm.process_batch(chunk, ctx, rejected, observations, flags, capture=capture))
        for item in capture[before:]:
            item["record"] = start + item.pop("rows")
        start += len(chunk)
    joined = {key: np.concatenate([c[key] for c in capture]) for key in capture[0]}
    return joined, b"".join(w.tobytes() for w in written), rejected


def skew_inputs(files, ldat):
    setup = LegacySetup.from_files(CONFIG, files["encal"], root=ROOT)
    lookup = CompiledLookup.from_setup(setup)
    context = walk.WalkContext.from_files(CONFIG, files["encal"], files["cog"], files["doi"])
    parts = []
    for features in compiled_batches(ldat, lookup, file_index=0, seed=0):
        if not len(features.ints):
            continue
        values, status1, status2 = walk.feature_inputs(context, features)
        values.update(record=features.column("record"), status_1=status1, status_2=status2,
                      dt_ps=features.column("dt_ps"),
                      **{f"{c}_{s}": features.column(f"{c}_{s}") for c in ("time_channel", "slab") for s in (1, 2)})
        parts.append(values)
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def shared(ours, theirs):
    """Events both pipelines accept with the same TIME channel and slab on both sides."""
    common, i, j = np.intersect1d(ours["record"], theirs["record"], return_indices=True)
    same = np.ones(len(common), bool)
    for c in ("time_channel_1", "time_channel_2", "slab_1", "slab_2"):
        same &= ours[c][i] == theirs[c][j]
    return i[same], j[same], len(common)


def assert_inputs_equal(ours, capture, i, j, rtol):
    """Both pipelines' walk inputs (listmode: ``walk_inputs`` of its non-negative hits) on shared events."""
    for s in (1, 2):
        assert np.all(capture[f"status_{s}"][j] == 0)
        np.testing.assert_array_equal(capture[f"timestamp_channel_{s}"][j], ours[f"time_channel_{s}"][i])
        np.testing.assert_allclose(ours[f"charge_{s}"][i], capture[f"charge_{s}"][j], rtol=1e-6)
        np.testing.assert_allclose(ours[f"share_{s}"][i], capture[f"share_{s}"][j], rtol=rtol)
        np.testing.assert_allclose(ours[f"doi_mm_{s}"][i], capture[f"doi_mm_{s}"][j], rtol=rtol, atol=rtol)
        np.testing.assert_allclose(ours[f"kev_{s}"][i], capture[f"kev_{s}"][j], rtol=1e-5)


@pytest.mark.fr("009-FR-1")
def test_inputs_match_listmode_on_shared_events(synthetic):
    capture, _, _ = listmode_capture(synthetic, synthetic["ldat"])
    ours = skew_inputs(synthetic, synthetic["ldat"])
    i, j, common = shared(ours, capture)
    assert common > 1000 and len(i) > 0.7 * common            # single-TIME-hit slabs draw differently
    assert np.all(ours["status_1"][i] == 0) and np.all(ours["status_2"][i] == 0)
    np.testing.assert_array_equal(ours["dt_ps"][i], capture["dt_ps"][j])
    assert_inputs_equal(ours, capture, i, j, rtol=1e-5)


@pytest.mark.fr("009-FR-1")
def test_capture_leaves_listmode_output_unchanged(synthetic):
    _, with_capture, rejected = listmode_capture(synthetic, synthetic["ldat"])
    ctx, mapping = _listmode(synthetic)
    plain = dict.fromkeys(lm.REJECTIONS, 0)
    np.random.seed(LM_SEED)
    out = b"".join(lm.process_batch(chunk, ctx, plain, dict.fromkeys(lm.OBSERVATIONS, 0), {}).tobytes()
                   for chunk in _chunks(ctx, mapping, synthetic["ldat"]))
    assert out == with_capture and plain == rejected and len(out) > 1000 * lm.RECORD_DTYPE.itemsize


@pytest.mark.fr("009-FR-1")
def test_missing_inputs_are_counted(synthetic):
    setup = synthetic["setup"]
    ch = channels(setup, 1, 5)[0][3]
    doi = _limits(synthetic["base"] / "doi_skip.txt", setup, (1.0, 8.0), skip={(ch, 6)})
    context = walk.WalkContext.from_files(CONFIG, synthetic["encal"], synthetic["cog"], doi)
    bad_ch = channels(setup, MISSING_SM, 5)[0][3]
    #               ok  no TIME zero TIME no ENERGY no factor no DOI limit
    tch = np.array([ch, ch, ch, ch, bad_ch, ch])
    slab = np.array([7, 7, 7, 7, 7, 6])
    values, status = walk.inputs(context, tch, slab, q_top=[10, -1, 0, 10, 10, 10], q_time=[20, -1, 0, 20, 20, 20],
                                 e_raw=[511.0] * 6, doi_ratio=[2.5, 2.5, 2.5, 0.0, 2.5, 2.5], y_cog=[10.0] * 6)
    assert [walk.STATUS[s] for s in status] == ["ok", "no TIME charge", "no TIME charge", "no ENERGY charge",
                                                "no calibration factor", "no DOI limits"]
    assert values["share"][0] == 0.5 and values["kev"][0] == pytest.approx(511.0)
    assert values["doi_mm"][0] == pytest.approx((2.5 - 8.0) * 20 / (1.0 - 8.0), rel=1e-5)
    assert all(np.isnan(values[name][1:]).all() for name in walk.INPUTS)


@pytest.mark.fr("009-FR-1")
def test_ingest_outputs_unchanged(synthetic):
    """The walk columns are extra (the spec 008 feature columns keep their golden checks in
    tests/test_cornell_timing_ingest.py); with mu = 511 the ENERGY sum equals the selection keV."""
    lookup = CompiledLookup.from_setup(synthetic["setup"])
    features = next(f for f in compiled_batches(synthetic["ldat"], lookup, file_index=0, seed=0) if len(f.ints))
    assert features.walk.shape == (len(features.ints), 8)
    for s in (1, 2):
        q_top, q_time = features.column(f"q_top_{s}"), features.column(f"q_time_{s}")
        assert np.all(q_top > 0) and np.all(q_time >= q_top) and np.all(features.column(f"doi_ratio_{s}") >= 1)
        np.testing.assert_allclose(features.column(f"e_raw_{s}"), features.column(f"energy_kev_{s}"), rtol=1e-12)


def _prefix(ldat, out, records):
    """First ``records`` compact coincidence records of ``ldat``."""
    with open(ldat, "rb") as src, open(out, "wb") as dst:
        for _ in range(records):
            head = src.read(2)
            if len(head) < 2:
                break
            dst.write(head + src.read(16 * (head[0] + head[1])))
    return out


@pytest.mark.real_data
@pytest.mark.fr("009-FR-1")
def test_real_point_source_parity(tmp_path):
    data, cal = os.environ.get("PETSYS_DATA_DIR"), os.environ.get("PETSYS_CAL_DIR")
    if not data or not cal:
        pytest.skip("PETSYS_DATA_DIR and PETSYS_CAL_DIR are required")
    source = (Path(data) / "Cornell/full_system/point_source/NEMA_SR_10142026_X0Y0Z0_conv_uncorr"
              / "NEMA_SR_10142026_X0Y0Z0_coincCompact.ldat")
    files = {"encal": Path(cal) / "norm_10082026_r7_coincCompact_resolved.encal",
             "cog": Path(cal) / "cog_limits_full_system.txt", "doi": Path(cal) / "doi_limits_full_system.txt",
             "regions": Path(data) / "Cornell/files_for_listmode/region_sm_mm_map.tsv",
             "pairs": Path(data) / "Cornell/files_for_listmode/pairs_map_cornell.txt"}
    for path in [source, *files.values()]:
        if not path.is_file():
            pytest.skip(f"missing {path}")
    prefix = _prefix(source, tmp_path / "x0y0z0_prefix_coinc_0.ldat", 200_000)
    capture, _, _ = listmode_capture(files, prefix)
    ours = skew_inputs(files, prefix)
    i, j, common = shared(ours, capture)
    ok = (ours["status_1"][i] == 0) & (ours["status_2"][i] == 0)
    print(f"real parity: {common:,} records in both, {len(i):,} shared (same TIME channel and slab), "
          f"{ok.mean():.4%} with all inputs")
    assert common > 10_000 and len(i) > 0.5 * common and ok.mean() > 0.99
    assert_inputs_equal(ours, capture, i[ok], j[ok], rtol=1e-4)


# --- T2: walk LUT contract (FR-2) ---------------------------------------------
def _lut(rng, *, events=None):
    edges = {"walk2d": [np.linspace(5, 35, 21), np.linspace(0.3, 1.0, 11)], "doi": [np.linspace(0, 20, 21)],
             "kev": [np.linspace(300, 700, 21)]}
    offsets = {"walk2d": rng.normal(0, 50, (20, 10)), "doi": rng.normal(0, 10, 20), "kev": rng.normal(0, 10, 20)}
    events = events or {"walk2d": rng.integers(300, 900, (20, 10)), "doi": rng.integers(300, 900, 20),
                        "kev": rng.integers(300, 900, 20)}
    return walk.WalkLut.from_fit(edges, offsets, events, {"fit_kind": "synthetic"})


@pytest.mark.fr("009-FR-2")
def test_lut_roundtrip_and_schema(tmp_path):
    lut = _lut(np.random.default_rng(1))
    manifest = lut.write(tmp_path / "walk_lut.json")
    back = walk.WalkLut.read(tmp_path / "walk_lut.json")
    assert manifest["schema"] == walk.LUT_SCHEMA and back.sha256 == manifest["table"]["sha256"]
    assert back.manifest["min_events"] == walk.MIN_EVENTS and back.manifest["fit_kind"] == "synthetic"
    for b in walk.BLOCKS:
        np.testing.assert_array_equal(back.offset_ps[b], lut.offset_ps[b])
        np.testing.assert_array_equal(back.events[b], lut.events[b])
        weights = back.events[b]
        assert abs((weights * back.offset_ps[b]).sum() / weights.sum()) < 1e-9      # zero-mean gauge
    table = (tmp_path / "walk_lut.json").read_bytes()
    (tmp_path / "walk_lut.json").write_bytes(table.replace(b'"units": "ps"', b'"units": "ns"'))
    with pytest.raises(walk.WalkError, match="does not match|unknown"):
        walk.WalkLut.read(tmp_path / "walk_lut.json")


@pytest.mark.fr("009-FR-2")
def test_refuses_overwrite(tmp_path):
    lut = _lut(np.random.default_rng(2))
    lut.write(tmp_path / "walk_lut.json")
    with pytest.raises(walk.WalkError, match="output exists"):
        _lut(np.random.default_rng(3)).write(tmp_path / "walk_lut.json")
    (tmp_path / "walk_lut.json").unlink()
    with pytest.raises(walk.WalkError, match="output exists"):       # the manifest alone also blocks
        _lut(np.random.default_rng(3)).write(tmp_path / "walk_lut.json")


@pytest.mark.fr("009-FR-2")
def test_sparse_bins_take_nearest_and_are_listed():
    rng = np.random.default_rng(4)
    events = {"walk2d": np.full((20, 10), 500), "doi": np.full(20, 500), "kev": np.full(20, 500)}
    events["walk2d"][0, 0] = events["walk2d"][0, 1] = 10
    events["doi"][19] = 0
    edges = {"walk2d": [np.linspace(5, 35, 21), np.linspace(0.3, 1.0, 11)], "doi": [np.linspace(0, 20, 21)],
             "kev": [np.linspace(300, 700, 21)]}
    offsets = {"walk2d": rng.normal(0, 50, (20, 10)), "doi": np.arange(20.0), "kev": np.zeros(20)}
    lut = walk.WalkLut.from_fit(edges, offsets, events, {})
    assert lut.filled["walk2d"] == [[0, 0], [0, 1]] and lut.filled["doi"] == [[19]]
    two = lut.offset_ps["walk2d"]
    assert two[0, 0] == two[1, 0] and two[0, 1] == two[0, 2]     # nearest populated; ties: first in C order
    assert lut.offset_ps["doi"][19] == lut.offset_ps["doi"][18]
    with pytest.raises(walk.WalkError, match="no bin reaches"):
        walk.WalkLut.from_fit(edges, offsets, {**events, "kev": np.zeros(20, int)}, {})


@pytest.mark.fr("009-FR-2")
def test_clamp_at_edges_counted():
    lut = _lut(np.random.default_rng(5))
    values = {"charge": np.array([1.0, 20.0, 99.0, np.nan]), "share": np.array([0.5, 1.2, 0.5, 0.5]),
              "doi_mm": np.array([-3.0, 10.0, 10.0, 10.0]), "kev": np.array([500.0, 500.0, 800.0, 500.0])}
    correction, outside = lut.correction(values)
    assert outside["charge"].tolist() == [True, False, True, False]
    assert outside["share"].tolist() == [False, True, False, False]
    assert outside["doi_mm"].tolist() == [True, False, False, False] and outside["kev"].tolist()[2]
    assert np.isfinite(correction[:3]).all() and np.isnan(correction[3])
    edge = {"charge": np.array([5.0]), "share": np.array([0.5]), "doi_mm": np.array([0.0]), "kev": np.array([500.0])}
    assert lut.correction(values={k: v[:1] for k, v in values.items()})[0][0] == pytest.approx(
        lut.correction(edge)[0][0])                              # below the first edge = the first bin


@pytest.mark.fr("009-FR-2")
def test_correction_matches_python_reference():
    rng = np.random.default_rng(6)
    lut = _lut(rng)
    n = 500
    values = {"charge": rng.uniform(0, 40, n), "share": rng.uniform(0.2, 1.1, n), "doi_mm": rng.uniform(-2, 22, n),
              "kev": rng.uniform(250, 750, n)}
    correction, _ = lut.correction(values)

    def index(edges, v):
        for i in range(len(edges) - 1):
            if v < edges[i + 1]:
                return i
        return len(edges) - 2

    for k in range(n):
        ec, es = lut.edges["walk2d"]
        expected = (lut.offset_ps["walk2d"][index(ec, values["charge"][k]), index(es, values["share"][k])]
                    + lut.offset_ps["doi"][index(lut.edges["doi"][0], values["doi_mm"][k])]
                    + lut.offset_ps["kev"][index(lut.edges["kev"][0], values["kev"][k])])
        assert correction[k] == pytest.approx(expected, abs=1e-9)


# --- T3: point-source walk fit (FR-3, FR-8) -------------------------------------
FULL_ENERGY = np.array([200.0, 150.0, 100.0, 61.0])


def _true_walk(charge, share, doi, kev):
    return 250.0 * np.exp(-(charge - 5.0) / 6.0) + 150.0 * (0.9 - share) + 1.5 * doi + 0.15 * np.clip(480 - kev, 0, None)


@pytest.mark.fr("009-FR-3")
def test_fit_recovers_injected_walk():
    rng = np.random.default_rng(21)
    n = 80_000

    def side_inputs():
        share = np.where(rng.random(n) < 0.4, 1.0, rng.uniform(0.4, 0.95, n))
        return {"charge": rng.uniform(5, 35, n), "share": share, "doi_mm": rng.uniform(0, 20, n),
                "kev": np.clip(rng.normal(500, 60, n), 300, 700)}

    side1, side2 = side_inputs(), side_inputs()
    truth = _true_walk(*side1.values()) - _true_walk(*side2.values())
    noise = rng.normal(0, 180, n)
    background = rng.random(n) < 0.03
    residual = np.where(background, rng.uniform(-3000, 3000, n), truth + noise)
    train = np.arange(n) % 2 == 0
    lut, history = walk.fit_tables(side1, side2, residual, train, walk.DEFAULT_BINS, core_ps=600.0, iterations=4,
                                   minimum=200, manifest={"fit_kind": "synthetic"})
    estimate, _, _ = walk.walk_difference(lut, side1, side2)
    held = ~train & ~background
    error = estimate[held] - truth[held]
    assert abs(error.mean()) < 3.0                          # side differences: no gauge left
    assert error.std() < min(0.2 * truth[held].std(), 20.0)    # bin discretization of a steep, smooth truth
    assert len(history) == 4 and history[-1]["events"] == history[-2]["events"]   # core window converged
    before, after = np.std(residual[held]), np.std((residual - estimate)[held])
    assert after < 0.92 * before and after == pytest.approx(180, rel=0.05)   # 98 ps of walk removed in quadrature
    for b in walk.BLOCKS:                                    # zero-mean gauge per block
        weights = lut.events[b]
        assert abs((weights * lut.offset_ps[b]).sum() / weights.sum()) < 1e-6


def _all_features(path, setup):
    from src.cornell.skew.ingest import FeatureBatch
    batches = list(compiled_batches(path, CompiledLookup.from_setup(setup), file_index=0, seed=0))
    return FeatureBatch(0, *(np.concatenate([getattr(b, k) for b in batches]) for k in ("codes", "ints", "floats", "walk")))


DELAY_SM1, DELAY_SM16 = 300.0, -200.0      # frozen skew LUT delays (as tests/test_cornell_timing_ctr.py)


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _walk_case(tmp_path, count=3000, seed=5):
    """Frozen skew LUT + its job, listmode files, and a point-source LDAT: TOF + delays + injected walk."""
    from src.cornell.skew import ctr
    from src.cornell.skew import solver as ts
    from src.cornell.skew.ingest import GeometryHypothesis, apply_geometry
    from src.cornell.skew.lut import write_lut
    rng = np.random.default_rng(seed)
    setup = make_setup(tmp_path)
    t1, t2 = channels(setup, 1, 5)[0], channels(setup, 16, 5)[0]
    rows = np.zeros(len(t1) + len(t2), ts.CHANNEL_DTYPE)
    rows["time_channel"] = t1 + t2
    rows["delay_ps"] = [DELAY_SM1] * len(t1) + [DELAY_SM16] * len(t2)
    rows["sigma_ps"] = rows["raw_sigma_ps"] = 5.0
    rows["status"], rows["apply"] = ts.CALIBRATED, True
    job = tmp_path / "job.json"
    (tmp_path / "norm").mkdir()
    job.write_text(json.dumps({"config": str(CONFIG), "runs": [{"name": "r", "directory": "norm", "prefix": "norm_"}]}))
    lut = tmp_path / "lut.csv"
    write_lut(lut, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}),
              {"job": str(job), "job_sha256": _sha(job), "runs": ["r"], "encal_sha256": {"r": _sha(tmp_path / "r.encal")}})
    q = rng.uniform(6, 30, (count, 2))
    second = q * rng.uniform(0.05, 0.9, (count, 2))
    scale = rng.uniform(0.8, 1.2, (count, 2, 4))

    e1, e2 = channels(setup, 1, 5)[1], channels(setup, 16, 5)[1]

    def one(t_ch, e_ch, k, s, stamp):       # cornell_timing_helpers.side layout, channels looked up once
        hits = [(stamp + 3, q[k, s], t_ch[3]), (stamp + 4, second[k, s], t_ch[4])]
        return hits + [(stamp + 500 + i, value, e_ch[i]) for i, value in enumerate(scale[k, s] * FULL_ENERGY)]

    def records(dt_of):
        return [(one(t1, e1, k, 0, T0 + k * 1_000_000), one(t2, e2, k, 1, T0 + k * 1_000_000 - dt_of(k)))
                for k in range(count)]

    probe = write(tmp_path / "probe.ldat", records(lambda k: 0))
    features = _all_features(probe, setup)
    assert len(features.ints) == count
    geometry = apply_geometry(features, GeometryHypothesis.selected(CONFIG, setup))
    p, qq = geometry.endpoints_mm[:, :3], geometry.endpoints_mm[:, 3:]
    source = p[0] + 0.5 * (qq[0] - p[0])
    tof = ctr.geometric_tof_ps(source, p, qq)
    files = {"encal": _encal(tmp_path / "listmode.encal", setup),
             "cog": _limits(tmp_path / "cog.txt", setup, (-1000.0, 1000.0)),
             "doi": _limits(tmp_path / "doi.txt", setup, (1.0, 8.0))}
    context = walk.WalkContext.from_files(CONFIG, files["encal"], files["cog"], files["doi"])
    values, s1, s2 = walk.feature_inputs(context, features)
    assert np.all(s1 == 0) and np.all(s2 == 0)
    truth = (_true_walk(values["charge_1"], values["share_1"], values["doi_mm_1"], values["kev_1"])
             - _true_walk(values["charge_2"], values["share_2"], values["doi_mm_2"], values["kev_2"]))
    noise = rng.normal(0, 100, count)
    target = np.rint(tof + DELAY_SM1 - DELAY_SM16 + truth + noise).astype(np.int64)
    base_dt = features.column("dt_ps")
    ldat = write(tmp_path / "point.ldat", records(lambda k: int(target[k] - base_dt[k])))
    spec = {"schema": walk.FIT_SCHEMA, "config": str(CONFIG), "encal": str(tmp_path / "r.encal"), "lut": str(lut),
            "lut_sha256": _sha(lut), "output": str(tmp_path / "out" / "walk_lut.json"),
            "acquisitions": [{"name": "point", "ldat": str(ldat), "source_mm": [float(v) for v in source],
                              "source_sigma_mm": [1.0, 1.0, 1.0], "input_correction": "uncorrected"}],
            "bootstrap": 5, "listmode": {"calibration": str(files["encal"]), "cog_limits": str(files["cog"]),
                                         "doi_limits": str(files["doi"])},
            "bins": {"charge": 8, "share": 4, "doi_mm": 4, "kev": 4}, "min_events": 50}
    path = tmp_path / "walk.json"
    path.write_text(json.dumps(spec))
    return path, truth


@pytest.mark.fr("009-FR-3")
@pytest.mark.fr("009-FR-8")
def test_walk_command_end_to_end(tmp_path):
    from src.cornell.skew.__main__ import main
    path, truth = _walk_case(tmp_path)
    assert main(["walk", str(path)]) == 0
    out = tmp_path / "out"
    report = json.loads((out / "walk_lut_report.json").read_text())
    lut = walk.WalkLut.read(out / "walk_lut.json")
    assert report["walk_lut_sha256"] == lut.sha256 and lut.manifest["fit_kind"] == "point_source"
    assert lut.manifest["skew_lut_sha256"] == report["frozen_inputs"]["lut_sha256"]
    assert set(lut.manifest["listmode_files"]) == {"calibration", "cog_limits", "doi_limits"}
    stages = report["acquisitions"][0]["stages"]
    assert stages["kept"] == stages["records"] == 3000 and stages["walk input missing"] == 0
    before, after = report["results"]["pooled"]["before"], report["results"]["pooled"]["after"]
    assert after["fwhm_ps"] < 0.8 * before["fwhm_ps"]
    assert after["sigma_ps"] == pytest.approx(100, rel=0.25)
    for name in ("walk_lut_tables.png", "walk_lut_profiles.png"):
        assert (out / name).stat().st_size > 10_000
    assert main(["walk", str(path)]) == 1                  # outputs are never overwritten


@pytest.mark.fr("009-FR-3")
def test_fit_file_validation(tmp_path):
    path, _ = _walk_case(tmp_path, count=50)
    spec = json.loads(path.read_text())
    assert walk.load_fit(path).extra["fit_window_kev"] == [300.0, 700.0]
    bad = tmp_path / "bad.json"
    for change, match in (({"fit_window_kev": [700, 300]}, "fit_window_kev"), ({"bins": {"charge": 1}}, "bins"),
                          ({"min_events": 0}, "min_events"), ({"core_ps": -1}, "core_ps"),
                          ({"listmode": {"calibration": "x"}}, "listmode"), ({"schema": "cornell-ctr-point/1"}, "schema")):
        bad.write_text(json.dumps(spec | change))
        with pytest.raises(ValueError, match=match):
            walk.load_fit(bad)
    bad.write_text(json.dumps({k: v for k, v in spec.items() if k != "listmode"}))
    with pytest.raises(ValueError, match="required"):
        walk.load_fit(bad)


# --- T4: CTR with walk (FR-5) ---------------------------------------------------
@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Synthetic point source with injected walk, its walk LUT, and a CTR file template."""
    from src.cornell.skew import ctr
    tmp_path = tmp_path_factory.mktemp("walk_ctr")
    path, _ = _walk_case(tmp_path)
    walk.run_fit(walk.load_fit(path), log=lambda *_: None)
    fit = json.loads(path.read_text())
    lut = walk.WalkLut.read(tmp_path / "out" / "walk_lut.json")
    ctr_spec = {k: fit[k] for k in ("config", "encal", "lut", "lut_sha256", "acquisitions", "bootstrap")}
    ctr_spec |= {"schema": ctr.SCHEMA, "depth_variants_mm": [3.0]}
    walk_keys = {"walk_lut": str(tmp_path / "out" / "walk_lut.json"), "walk_lut_sha256": lut.sha256,
                 "listmode": fit["listmode"]}
    return {"tmp": tmp_path, "ctr": ctr_spec, "walk": walk_keys, "lut": lut}


def _run_ctr(fitted, name, **keys):
    from src.cornell.skew import ctr
    path = fitted["tmp"] / f"{name}.json"
    path.write_text(json.dumps(fitted["ctr"] | {"output": str(fitted["tmp"] / name / "ctr.json")} | keys))
    return ctr.run_ctr(ctr.load_ctr(path), log=lambda *_: None)


@pytest.mark.fr("009-FR-5")
def test_ctr_walk_lut_improves_injected_walk(fitted):
    plain = _run_ctr(fitted, "plain")
    walked = _run_ctr(fitted, "walked", **fitted["walk"])
    without, with_walk = walked["results_without_walk"]["pooled"]["fit"], walked["results"]["pooled"]["fit"]
    assert without["fwhm_ps"] == pytest.approx(plain["results"]["pooled"]["fit"]["fwhm_ps"], rel=1e-9)
    assert with_walk["fwhm_ps"] < 0.8 * without["fwhm_ps"] and with_walk["events"] == without["events"]
    assert walked["walk"]["walk_lut_sha256"] == fitted["lut"].sha256
    stages = walked["counts"]["point"]["stages"]
    assert stages["walk input missing"] == 0 and stages["in window"] == 3000
    assert "walk" not in plain and "results_without_walk" not in plain


@pytest.mark.fr("009-FR-5")
@pytest.mark.fr("009-FR-4")
def test_walk_pairing_refused(fitted):
    from src.cornell.skew import ctr
    keys = fitted["walk"]
    with pytest.raises(ctr.CtrError, match="go together"):
        _run_ctr(fitted, "partial", walk_lut=keys["walk_lut"])
    with pytest.raises(ctr.CtrError, match="not the frozen"):
        _run_ctr(fitted, "sha", **keys | {"walk_lut_sha256": "0" * 64})
    other = dict(keys["listmode"], doi_limits=str(_limits(fitted["tmp"] / "doi2.txt", fitted_setup(fitted), (1.5, 8.0))))
    with pytest.raises(ctr.CtrError, match="differ from those the walk LUT was fitted with"):
        _run_ctr(fitted, "files", **keys | {"listmode": other})
    lut = walk.WalkLut.read(keys["walk_lut"])
    lut.manifest = dict(lut.manifest, skew_lut_sha256="f" * 64)
    lut.write(fitted["tmp"] / "other_walk.json")
    with pytest.raises(ctr.CtrError, match="was fitted with skew LUT"):
        _run_ctr(fitted, "pair", **keys | {"walk_lut": str(fitted["tmp"] / "other_walk.json"),
                                           "walk_lut_sha256": lut.sha256})


def fitted_setup(fitted):
    return LegacySetup.from_files(CONFIG, fitted["tmp"] / "r.encal", root=ROOT)


@pytest.mark.fr("009-FR-4")
def test_skew_lut_solved_with_walk_needs_walk():
    from src.cornell.skew.lut import CanonicalLut
    skew = CanonicalLut(np.zeros(0), {"provenance": {"walk_lut_sha256": "a" * 64}})
    with pytest.raises(walk.WalkError, match="solved with walk LUT"):
        walk.check_pairing(skew, "b" * 64, None)
    paired = walk.WalkLut.__new__(walk.WalkLut)
    paired.manifest = {"skew_lut_sha256": "c" * 64, "table": {"sha256": "a" * 64}}
    walk.check_pairing(skew, "b" * 64, paired)          # round trip: the walk LUT the skew LUT was solved with
    paired.manifest = {"skew_lut_sha256": "b" * 64, "table": {"sha256": "d" * 64}}
    walk.check_pairing(skew, "b" * 64, paired)          # a walk LUT refitted with this skew LUT


@pytest.mark.fr("009-FR-5")
def test_previous_reports_summary(fitted):
    plain = _run_ctr(fitted, "first")
    report = _run_ctr(fitted, "second", **fitted["walk"], previous_reports=[str(fitted["tmp"] / "first" / "ctr.json")])
    rows = report["round_trip"]
    assert [r["report"].endswith("ctr.json") for r in rows] == [True, False] and rows[1]["report"] == "this report"
    assert rows[0]["walk_lut_sha256"] is None and rows[1]["walk_lut_sha256"] == fitted["lut"].sha256
    assert rows[0]["results"]["pooled"]["fwhm_ps"] == pytest.approx(plain["results"]["pooled"]["fit"]["fwhm_ps"])
    assert rows[1]["results"]["pooled"]["fwhm_ps"] < rows[0]["results"]["pooled"]["fwhm_ps"]


# --- T5: skew job with walk (FR-4) ----------------------------------------------
WALK_BIAS_PS = 200.0       # W(SM 1 side, low charge) - W(SM 16 side, high charge)


def _step_walk_lut(path):
    """Known walk LUT: +200 ps below charge 15, 0 above; DOI and keV flat."""
    edges = {"walk2d": [np.array([0.0, 15.0, 100.0]), np.array([0.0, 0.75, 1.0])],
             "doi": [np.array([-100.0, 100.0])], "kev": [np.array([0.0, 2000.0])]}
    offsets = {"walk2d": np.array([[WALK_BIAS_PS, WALK_BIAS_PS], [0.0, 0.0]]), "doi": np.zeros(1), "kev": np.zeros(1)}
    events = {"walk2d": np.full((2, 2), 1000), "doi": np.full(1, 1000), "kev": np.full(1, 1000)}
    lut = walk.WalkLut.from_fit(edges, offsets, events, {"fit_kind": "synthetic"})
    return lut


def _walk_run(base, setup, name, *, seed, shift, accepted=70):
    """make_run with low TIME charge on SM 1 and high on SM 16: dt = shift + walk bias +- TOF + noise."""
    from cornell_timing_helpers import FILES, TOF_PS, shifted, side
    directory = base / name
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    prefix = f"norm_test_{name}_coincCompact_"
    artifacts = []
    for index in range(FILES):
        records = []
        for k in range(accepted):
            mm = (5, 6)[k % 2]
            dt = int(round(shift + WALK_BIAS_PS + rng.choice((-TOF_PS, TOF_PS)) + rng.normal(0, 150)))
            stamp = T0 + k * 1_000_000 + index * 10 ** 12
            records.append((side(setup, 1, mm=mm, time={3: 8.0, 4: 4.0}, stamp=stamp),
                            side(setup, 16, mm=mm, time={3: 25.0, 4: 4.0}, stamp=stamp - dt)))
        path = write(directory / f"{prefix}{index:08d}.ldat", records)
        artifacts.append({"kind": "ldat", "cleanup": "retained", "path": str(path), "size_bytes": path.stat().st_size,
                          "input_descriptor": {"format": "compact", "population": "coincidence"}})
    ledger = {"status": "succeeded", "attempts": [{"stage_id": "conversion", "status": "succeeded", "exit_code": 0,
                                                  "artifacts": artifacts}]}
    (directory / "run.json").write_text(json.dumps(ledger))
    return {"name": name, "directory": str(directory), "ledger": str(directory / "run.json"), "prefix": prefix,
            "energy_calibration_prefix": "none"}


@pytest.fixture(scope="module")
def walk_jobs(tmp_path_factory):
    """One synthetic norm run solved without and with the walk LUT (job outputs and reports)."""
    from cornell_timing_helpers import job_file
    from src.cornell.skew import job as tj
    base = tmp_path_factory.mktemp("walk_job")
    setup = make_setup(base)
    item = _walk_run(base, setup, "wa", seed=31, shift=600.0)
    files = {"calibration": str(_encal(base / "listmode.encal", setup)),
             "cog_limits": str(_limits(base / "cog.txt", setup, (-1000.0, 1000.0))),
             "doi_limits": str(_limits(base / "doi.txt", setup, (1.0, 8.0)))}
    context = walk.WalkContext.from_files(CONFIG, files["calibration"], files["cog_limits"], files["doi_limits"])
    lut = _step_walk_lut(base)
    lut.manifest["listmode_files"] = context.files
    lut.write(base / "walk_lut.json")
    out = {}
    for label, extra in (("plain", {}), ("walk", {"walk_lut": str(base / "walk_lut.json"), "listmode": files})):
        folder = base / label
        folder.mkdir()
        job = tj.load_job(job_file(folder / "job.json", base, [item], **extra))
        out[label] = (job, tj.run_job(job, log=lambda message: None))
    return {"base": base, "setup": setup, "jobs": out, "walk": walk.WalkLut.read(base / "walk_lut.json"),
            "files": files}


def _pair_delays(job, setup):
    from src.cornell.skew.lut import CanonicalLut
    lut = CanonicalLut.read(job.lut)
    applied = set(lut.rows["time_channel"][lut.rows["apply"]].tolist())
    pairs = [(channels(setup, 1, mm)[0][3], channels(setup, 16, mm)[0][3]) for mm in (5, 6)]
    # the two minimodule pairs are separate components; only component 0 is applied (spec 008 FR-10)
    return [lut.delay(a) - lut.delay(b) for a, b in pairs if a in applied and b in applied]


@pytest.mark.slow  # walk_jobs fixture: two synthetic jobs, ~5.4 s setup
@pytest.mark.fr("009-FR-4")
def test_walk_applied_before_pair_fits(walk_jobs):
    setup = walk_jobs["setup"]
    (plain, plain_report), (walked, walk_report) = walk_jobs["jobs"]["plain"], walk_jobs["jobs"]["walk"]
    assert plain_report["status"] == walk_report["status"] == "succeeded"
    pairs = list(zip(_pair_delays(plain, setup), _pair_delays(walked, setup)))
    assert len(pairs) == 1
    for without, with_walk in pairs:
        assert without == pytest.approx(600.0 + WALK_BIAS_PS, abs=40)      # walk biases the pair estimate
        assert with_walk == pytest.approx(600.0, abs=40)                    # removed when applied first
    info = walk_report["runs"]["wa"]["walk"]
    assert all(counts["walk input missing"] == 0 for counts in info.values())
    assert walk_report["walk"]["walk_lut_sha256"] == walk_jobs["walk"].sha256 and "walk" not in plain_report


@pytest.mark.slow  # walk_jobs fixture: two synthetic jobs, ~5.4 s setup
@pytest.mark.fr("009-FR-4")
def test_lut_manifest_records_walk(walk_jobs):
    from src.cornell.skew.lut import CanonicalLut
    walked, _ = walk_jobs["jobs"]["walk"]
    plain, _ = walk_jobs["jobs"]["plain"]
    skew = CanonicalLut.read(walked.lut)
    assert skew.manifest["provenance"]["walk_lut_sha256"] == walk_jobs["walk"].sha256
    assert "walk_lut_sha256" not in CanonicalLut.read(plain.lut).manifest["provenance"]
    sha = skew.manifest["table"]["sha256"]
    with pytest.raises(walk.WalkError, match="solved with walk LUT"):
        walk.check_pairing(skew, sha, None)
    walk.check_pairing(skew, sha, walk_jobs["walk"])            # the walk LUT it was solved with


@pytest.mark.slow  # walk_jobs fixture: two synthetic jobs, ~5.4 s setup
@pytest.mark.fr("009-FR-4")
def test_schema_3_refuses_old_cache(walk_jobs, tmp_path):
    from src.cornell.skew import cache as tc
    plain, _ = walk_jobs["jobs"]["plain"]
    features = plain.work / "wa" / "features"
    manifest = json.loads((features / "manifest.json").read_text())
    assert manifest["fingerprint"]["schema_version"] == tc.SCHEMA_VERSION == 3
    assert {"q_top_1", "doi_ratio_2"} <= {name for name, _ in manifest["fingerprint"]["feature_dtype"]}
    old = tmp_path / "features"
    old.mkdir()
    manifest["fingerprint"]["schema_version"] = 2
    (old / "manifest.json").write_text(json.dumps(manifest))
    current = json.loads((features / "manifest.json").read_text())
    with pytest.raises(tc.CacheMismatch):       # a version-2 cache directory is never reused by version 3
        tc._prepare(old, current)


# --- T7: listmode application (FR-6) --------------------------------------------
def _skew_lut(path, setup, rng, *, skip_sm=None, provenance=None):
    """Canonical skew LUT over every TIME channel of the map (random delays); SM ``skip_sm`` unobserved."""
    from src.cornell.skew import solver as ts
    from src.cornell.skew.lut import write_lut
    time = sorted(ch for ch, types in setup.chtype_map.items() if ChannelType.TIME in types)
    rows = np.zeros(len(time), ts.CHANNEL_DTYPE)
    rows["time_channel"] = time
    applied = np.array([setup.sm_mM_map[ch][0] != skip_sm for ch in time])
    rows["delay_ps"] = np.where(applied, rng.normal(0, 300, len(time)), np.nan)
    rows["sigma_ps"] = rows["raw_sigma_ps"] = np.where(applied, 5.0, np.nan)
    rows["status"] = np.where(applied, ts.CALIBRATED, ts.UNOBSERVED)
    rows["apply"] = applied
    write_lut(path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), provenance or {"job": "synthetic"})
    return path


def _timing(synthetic, skew, walk_path=None, policy="keep"):
    config = load_processing_config(CONFIG, processing_root=ROOT, action="listmode")
    mapping = config.mapping
    return lm.TimingCorrection.load(config, load_calibration(Path(synthetic["encal"]), mapping),
                                    load_limits(Path(synthetic["cog"]), mapping, kind="cog"),
                                    load_limits(Path(synthetic["doi"]), mapping, kind="doi"), skew, walk_path, policy)


def _run_listmode(synthetic, timing):
    from dataclasses import replace
    ctx, mapping = _listmode(synthetic)
    ctx = replace(ctx, timing=timing)
    rejected = dict.fromkeys(lm.REJECTIONS + (lm.TIMING_REJECTION,), 0)
    observations = dict.fromkeys(lm.OBSERVATIONS + lm.TIMING_OBSERVATIONS, 0)
    capture, out, start = [], [], 0
    np.random.seed(LM_SEED)
    for chunk in _chunks(ctx, mapping, synthetic["ldat"]):
        before = len(capture)
        out.append(lm.process_batch(chunk, ctx, rejected, observations, {}, capture=capture).tobytes())
        for item in capture[before:]:
            item["record"] = start + item.pop("rows")
        start += len(chunk)
    joined = {key: np.concatenate([c[key] for c in capture]) for key in capture[0]}
    return b"".join(out), joined, rejected, observations


@pytest.fixture(scope="module")
def listmode_tables(synthetic):
    """A skew LUT over every TIME channel, a walk LUT fitted with it (listmode files recorded), and one with an
    unobserved SM."""
    from src.cornell.skew.lut import CanonicalLut
    base, setup = synthetic["base"], synthetic["setup"]
    skew = _skew_lut(base / "skew.csv", setup, np.random.default_rng(41))
    holes = _skew_lut(base / "skew_holes.csv", setup, np.random.default_rng(41), skip_sm=16)
    table = _lut(np.random.default_rng(42))
    context = walk.WalkContext.from_files(CONFIG, synthetic["encal"], synthetic["cog"], synthetic["doi"])
    table.manifest.update(listmode_files=context.files, skew_lut_sha256=_sha(skew))
    table.write(base / "walk_listmode.json")
    hole_walk = _lut(np.random.default_rng(42))
    hole_walk.manifest.update(listmode_files=context.files, skew_lut_sha256=_sha(holes))
    hole_walk.write(base / "walk_holes.json")
    return {"skew": skew, "holes": holes, "walk": base / "walk_listmode.json", "hole_walk": base / "walk_holes.json",
            "skew_lut": CanonicalLut.read(skew)}


@pytest.mark.fr("009-FR-6")
def test_without_tables_byte_identical(synthetic, listmode_tables, tmp_path):
    plain, _, rejected, _ = _run_listmode(synthetic, None)
    zero = _skew_lut(tmp_path / "zero.csv", synthetic["setup"], np.random.default_rng(1))
    from src.cornell.skew.lut import CanonicalLut, write_lut
    from src.cornell.skew import solver as ts
    source = CanonicalLut.read(zero)
    flat = source.rows.copy()
    flat["delay_ps"] = np.where(flat["apply"], 0.0, np.nan)
    write_lut(tmp_path / "flat.csv", ts.Solution(flat, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), {"job": "zero"})
    corrected, _, rejected_flat, observations = _run_listmode(synthetic, _timing(synthetic, tmp_path / "flat.csv"))
    assert corrected == plain and rejected_flat == rejected            # all-zero delays leave every byte as is
    assert observations["timing_corrected"] == len(plain) // lm.RECORD_DTYPE.itemsize


@pytest.mark.fr("009-FR-6")
def test_tables_match_ctr_corrected_dt(synthetic, listmode_tables):
    timing = _timing(synthetic, listmode_tables["skew"], listmode_tables["walk"])
    _, capture, _, observations = _run_listmode(synthetic, timing)
    ours = skew_inputs(synthetic, synthetic["ldat"])
    i, j, common = shared(ours, capture)
    assert len(i) > 0.7 * common > 700
    table = walk.WalkLut.read(listmode_tables["walk"])
    skew = listmode_tables["skew_lut"]
    b1 = np.array([skew.delay(c) for c in ours["time_channel_1"][i]])
    b2 = np.array([skew.delay(c) for c in ours["time_channel_2"][i]])
    side1 = {n: ours[f"{n}_1"][i] for n in walk.INPUTS}
    side2 = {n: ours[f"{n}_2"][i] for n in walk.INPUTS}
    difference, _, _ = walk.walk_difference(table, side1, side2)
    expected = np.rint(ours["dt_ps"][i] - b1 + b2 - difference)        # the CTR code's corrected t_1 - t_2
    np.testing.assert_array_equal(capture["dt_corrected_ps"][j], expected)
    assert np.any(capture["dt_corrected_ps"] != capture["dt_ps"])
    assert observations["timing_corrected"] == len(capture["dt_ps"]) and observations["timing_uncorrected_kept"] == 0


@pytest.mark.fr("009-FR-6")
def test_uncorrected_events_kept_or_dropped(synthetic, listmode_tables):
    plain, _, _, _ = _run_listmode(synthetic, None)
    kept, kept_capture, _, kept_obs = _run_listmode(
        synthetic, _timing(synthetic, listmode_tables["holes"], listmode_tables["hole_walk"], "keep"))
    dropped, _, dropped_rejected, dropped_obs = _run_listmode(
        synthetic, _timing(synthetic, listmode_tables["holes"], listmode_tables["hole_walk"], "drop"))
    size = lm.RECORD_DTYPE.itemsize
    uncorrected = kept_obs["timing_uncorrected_kept"]
    assert uncorrected > 50 and len(kept) == len(plain)                # SM 16 has no delays: kept, raw dt
    raw = kept_capture["dt_corrected_ps"] == kept_capture["dt_ps"]
    assert raw.sum() >= uncorrected
    assert dropped_rejected[lm.TIMING_REJECTION] == uncorrected and len(dropped) == len(plain) - uncorrected * size
    assert dropped_obs["timing_corrected"] == kept_obs["timing_corrected"] == len(dropped) // size


@pytest.mark.fr("009-FR-6")
def test_mismatched_tables_refused(synthetic, listmode_tables, tmp_path):
    from src.cornell.inputs import InputError
    other_doi = _limits(tmp_path / "doi_other.txt", synthetic["setup"], (1.5, 8.0))
    with pytest.raises(InputError, match="differ from those the walk LUT was fitted with"):
        _timing(synthetic | {"doi": other_doi}, listmode_tables["skew"], listmode_tables["walk"])
    with pytest.raises(InputError, match="was fitted with skew LUT"):
        _timing(synthetic, listmode_tables["holes"], listmode_tables["walk"])
    with pytest.raises(InputError, match="policy"):
        _timing(synthetic, listmode_tables["skew"], None, "zero")
    foreign = _skew_lut(tmp_path / "foreign.csv", synthetic["setup"], np.random.default_rng(3))
    from src.cornell.skew.lut import CanonicalLut, write_lut
    from src.cornell.skew import solver as ts
    rows = CanonicalLut.read(foreign).rows.copy()
    rows["time_channel"][0] = 999_999_999 % (2 ** 31)                # not a channel of the map
    write_lut(tmp_path / "foreign2.csv", ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), {"job": "x"})
    with pytest.raises(InputError, match="not TIME channels"):
        _timing(synthetic, tmp_path / "foreign2.csv")
    solved = _skew_lut(tmp_path / "solved.csv", synthetic["setup"], np.random.default_rng(4),
                       provenance={"job": "x", "walk_lut_sha256": "e" * 64})
    with pytest.raises(InputError, match="solved with walk LUT"):
        _timing(synthetic, solved)


@pytest.mark.real_data
@pytest.mark.fr("009-FR-6")
def test_real_listmode_corrected_dt(tmp_path):
    """Listmode with the T10 skew LUT and the T6 step-1 walk LUT on an X0Y0Z0 prefix: its corrected dt equals the
    CTR code's (dt - b1 + b2 - (W1 - W2)) on shared events; without tables the bytes are unchanged."""
    data, cal = os.environ.get("PETSYS_DATA_DIR"), os.environ.get("PETSYS_CAL_DIR")
    if not data or not cal:
        pytest.skip("PETSYS_DATA_DIR and PETSYS_CAL_DIR are required")
    data, cal = Path(data), Path(cal)
    source = (data / "Cornell/full_system/point_source/NEMA_SR_10142026_X0Y0Z0_conv_uncorr"
              / "NEMA_SR_10142026_X0Y0Z0_coincCompact.ldat")
    skew = data / "Cornell/full_system/t10_replacement/t10/time_delays_t10.csv"
    table = data / "Cornell/full_system/point_source/walk_t6/step1/walk_lut_step1.json"
    files = {"encal": cal / "norm_10082026_r7_coincCompact_resolved.encal",
             "cog": cal / "cog_limits_full_system.txt", "doi": cal / "doi_limits_full_system.txt",
             "regions": data / "Cornell/files_for_listmode/region_sm_mm_map.tsv",
             "pairs": data / "Cornell/files_for_listmode/pairs_map_cornell.txt"}
    for path in [source, skew, table, *files.values()]:
        if not path.is_file():
            pytest.skip(f"missing {path}")
    files["ldat"] = _prefix(source, tmp_path / "x0y0z0_prefix_coinc_0.ldat", 200_000)
    plain, _, _, _ = _run_listmode(files, None)
    _, raw, _ = listmode_capture(files, files["ldat"])
    assert plain == raw                                             # the timing path off: unchanged bytes
    timing = _timing(files, skew, table)
    _, capture, _, observations = _run_listmode(files, timing)
    ours = skew_inputs(files, files["ldat"])
    i, j, common = shared(ours, capture)
    from src.cornell.skew.lut import CanonicalLut
    lut = CanonicalLut.read(skew)
    applied = set(lut.rows["time_channel"][lut.rows["apply"]].tolist())
    both = np.array([a in applied and b in applied for a, b in zip(ours["time_channel_1"][i], ours["time_channel_2"][i])])
    i, j = i[both], j[both]
    b1 = np.array([lut.delay(c) for c in ours["time_channel_1"][i]])
    b2 = np.array([lut.delay(c) for c in ours["time_channel_2"][i]])
    difference, _, _ = walk.walk_difference(walk.WalkLut.read(table), {n: ours[f"{n}_1"][i] for n in walk.INPUTS},
                                            {n: ours[f"{n}_2"][i] for n in walk.INPUTS})
    expected = np.rint(ours["dt_ps"][i] - b1 + b2 - difference)
    equal = capture["dt_corrected_ps"][j] == expected
    print(f"real listmode: {common:,} records in both, {len(i):,} shared with applied delays, "
          f"{equal.mean():.4%} equal; corrected {observations['timing_corrected']:,}, "
          f"kept uncorrected {observations['timing_uncorrected_kept']:,}")
    assert len(i) > 10_000 and equal.all()


@pytest.mark.fr("009-FR-6")
def test_same_layout_calibration_accepted_and_flagged(synthetic, listmode_tables, tmp_path):
    """Owner 2026-10-20: each run's own calibration of the walk LUT's layout is accepted and flagged."""
    from src.cornell.inputs import InputError
    same = _timing(synthetic, listmode_tables["skew"], listmode_tables["walk"])
    assert same.record["walk_lut"]["calibration_differs_from_fit"] is False
    lines = Path(synthetic["encal"]).read_text().splitlines()
    key, mu, sigma = lines[1].split(TAB)
    other = tmp_path / "run_own.encal"
    other.write_text(NL.join([lines[0], TAB.join([key, str(float(mu) * 1.01), sigma])] + lines[2:]) + NL)
    flagged = _timing(synthetic | {"encal": other}, listmode_tables["skew"], listmode_tables["walk"])
    assert flagged.record["walk_lut"]["calibration_differs_from_fit"] is True
    assert flagged.record["walk_lut"]["fitted_calibration_sha256"] == _sha(synthetic["encal"])
    regions = tmp_path / "regions_5.encal"
    rows = [f"({ch}, {slab}, {r}){TAB}511.0{TAB}1.0" for ch, slab in _time_slabs(synthetic["setup"]) for r in range(5)]
    regions.write_text("# Position-dependent energy calibration (5 regions per slab)" + NL
                       + f"ID(time_ch, slab, region){TAB}mu{TAB}sigma" + NL + NL.join(rows) + NL)
    with pytest.raises(InputError, match="differs in layout"):
        _timing(synthetic | {"encal": regions}, listmode_tables["skew"], listmode_tables["walk"])


# --- T8: normalization-phantom walk fit (FR-7) ----------------------------------
def _charge_walk(q):
    return 250.0 * np.exp(-(q - 6.0) / 6.0)


def _phantom_run(base, setup, name, *, seed, shift, accepted):
    """SM 1 vs SM 16 (mm 5): per-event random TIME charges; dt = shift + W(q_1) - W(q_2) +- shell TOF + noise."""
    from cornell_timing_helpers import FILES, TOF_PS
    directory = base / name
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    prefix = f"norm_test_{name}_coincCompact_"
    (t1, e1), (t2, e2) = channels(setup, 1, 5), channels(setup, 16, 5)
    artifacts, truth = [], []

    def one(t_ch, e_ch, q, stamp):
        hits = [(stamp + 3, q, t_ch[3]), (stamp + 4, 0.3 * q, t_ch[4])]
        return hits + [(stamp + 500 + i, value, e_ch[i]) for i, value in enumerate(FULL_ENERGY)]

    for index in range(FILES):
        records = []
        for k in range(accepted):
            q1, q2 = rng.uniform(6, 30, 2)
            dt = int(round(shift + _charge_walk(q1) - _charge_walk(q2) + rng.choice((-TOF_PS, TOF_PS))
                           + rng.normal(0, 150)))
            stamp = T0 + k * 1_000_000 + index * 10 ** 12
            records.append((one(t1, e1, q1, stamp), one(t2, e2, q2, stamp - dt)))
        path = write(directory / f"{prefix}{index:08d}.ldat", records)
        artifacts.append({"kind": "ldat", "cleanup": "retained", "path": str(path), "size_bytes": path.stat().st_size,
                          "input_descriptor": {"format": "compact", "population": "coincidence"}})
    ledger = {"status": "succeeded", "attempts": [{"stage_id": "conversion", "status": "succeeded", "exit_code": 0,
                                                  "artifacts": artifacts}]}
    (directory / "run.json").write_text(json.dumps(ledger))
    return {"name": name, "directory": str(directory), "ledger": str(directory / "run.json"), "prefix": prefix,
            "energy_calibration_prefix": "none"}


@pytest.mark.slow  # synthetic job (12k records) and phantom fit, ~10 s
@pytest.mark.fr("009-FR-7")
def test_phantom_fit_recovers_injected_walk(tmp_path):
    from cornell_timing_helpers import job_file
    from src.cornell.skew import job as tj
    setup = make_setup(tmp_path)
    item = _phantom_run(tmp_path, setup, "ph", seed=51, shift=600.0, accepted=4000)
    job = tj.load_job(job_file(tmp_path / "job.json", tmp_path, [item]))
    assert tj.run_job(job, log=lambda message: None)["status"] == "succeeded"
    files = {"calibration": str(_encal(tmp_path / "listmode.encal", setup)),
             "cog_limits": str(_limits(tmp_path / "cog.txt", setup, (-1000.0, 1000.0))),
             "doi_limits": str(_limits(tmp_path / "doi.txt", setup, (1.0, 8.0)))}
    spec = {"schema": walk.PHANTOM_SCHEMA, "job": str(tmp_path / "job.json"), "lut": str(job.lut),
            "lut_sha256": _sha(job.lut), "listmode": files, "output": str(tmp_path / "out" / "walk_phantom.json"),
            "bins": {"charge": 6, "share": 2, "doi_mm": 2, "kev": 2}, "min_events": 20}
    path = tmp_path / "phantom.json"
    path.write_text(json.dumps(spec))
    from src.cornell.skew.__main__ import main
    assert main(["walk", str(path)]) == 0
    lut = walk.WalkLut.read(tmp_path / "out" / "walk_phantom.json")
    assert lut.manifest["fit_kind"] == "phantom" and lut.manifest["skew_lut_sha256"] == _sha(job.lut)
    # per charge bin: fitted offset against the truth averaged over the bin (charges uniform; share, DOI and keV
    # are constant in the synthetic hits, so those blocks are single bins)
    edges = lut.edges["walk2d"][0]
    a, b = np.clip(edges[:-1], 6, 30), np.clip(edges[1:], 6, 30)
    truth = 250.0 * 6.0 * (np.exp(-(a - 6.0) / 6.0) - np.exp(-(b - 6.0) / 6.0)) / (b - a)
    counts = lut.events["walk2d"]                       # share: one value up to float32 rounding (1 or 2 bins)
    weights = counts.sum(axis=1)
    estimate = (lut.offset_ps["walk2d"] * counts).sum(axis=1) / np.maximum(weights, 1)
    error = (estimate - np.average(estimate, weights=weights)) - (truth - np.average(truth, weights=weights))
    assert np.sqrt(np.average(error ** 2, weights=weights)) < max(15.0, 0.15 * truth.std())
    # refused: a skew LUT from another job; a changed LUT
    other = dict(spec, lut_sha256="0" * 64, output=str(tmp_path / "out" / "other.json"))
    path.write_text(json.dumps(other))
    with pytest.raises(walk.WalkError, match="not the frozen"):
        walk.run_phantom(walk.load_phantom(path))
