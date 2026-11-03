"""IMAS fast LDAT -> LM (spec 015): selection, timing, records, job, command line, real-data parity."""
import ctypes
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, FULL, T0, channels, mu_file, side, write
from src.imas.listmode import job, records, timing as lm_timing
from src.imas.listmode import select as lsel
from src.imas.listmode.__main__ import PAIRS_BY_RULE, WALK_ENCAL, main, settings_from_args
from src.imas.skew import timestamp as ts
from src.imas.skew import walk as iw
from src.imas.skew.setup import ImasSetup
from src.listmode import CoincidenceV5, LMHeader
from src.timing import solver
from src.timing.lut import DelayTable, LutError, write_lut

ROOT = Path(__file__).resolve().parents[1]
SM_A, SM_B, SM_C, SM_D = 0, 12, 30, 50          # pairs (A, B) listed, (C, A) listed reversed, D unpaired


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG, root=ROOT)


def top(setup, sm, mm=5, slab=3):
    return channels(setup, sm, mm)[0][slab]


def pairs_file(path, pairs):
    path.write_text("".join(f"{p}\t{a}\t{b}\n" for (a, b), p in pairs.items()), encoding="utf-8")
    return path


PAIRS = {(SM_A, SM_B): 7, (SM_C, SM_A): 9, (SM_B, SM_C): 11}


def slab_mu(setup, **override):
    """mu = 511 for every TIME channel of the fixture minimodules (keV = ENERGY sum)."""
    values = {ch: 511.0 for sm in (SM_A, SM_B, SM_C, SM_D) for mm in (5, 6) for ch in channels(setup, sm, mm)[0]}
    values.update(override)
    return values


def lookup(setup, mu=None, pairs=PAIRS, walk_mu=None):
    return lsel.LmLookup.build(setup, slab_mu(setup) if mu is None else mu, pairs, (450, 570), walk_mu)


def features_of(tmp_path, setup, recs, lk=None, **options):
    path = write(tmp_path / "f.ldat", recs)
    lk = lk or lookup(setup)
    out = [f for _, f in lsel.feature_batches(path, lk, **options)]
    return out, lk


# --- T1: selection (FR-2, FR-3, FR-5 inputs) ---------------------------------------------------------------------

@pytest.mark.fr("015-FR-3")
def test_each_cut_is_the_first_failing_one(tmp_path, setup):
    unmapped = int(setup.lookup().kind.size) + 5
    ok = (side(setup, SM_A), side(setup, SM_B, stamp=T0 + 300))
    two_mm = side(setup, SM_A, energy=[200.0, 150.0, 100.0, 61.0]) + side(setup, SM_A, mm=6, time={2: 3.0},
                                                                              energy=[30.0, 20.0, 10.0])
    recs = [
        ok,                                                                              # accepted
        (side(setup, SM_A, extra=[(T0, 1.0, unmapped)]), side(setup, SM_B)),             # 1 unmapped
        (side(setup, SM_A, extra=[(T0, 0.1, unmapped)]), side(setup, SM_B)),             # below floor: accepted
        (side(setup, SM_A, energy=[200.0, 150.0, 100.0, 61.0]), side(setup, SM_B)),      # 2 side min ch
        (side(setup, SM_A, energy=[200.0, 150.0, 100.0, 61.0, 0.1]), side(setup, SM_B)),  # floor first, then 2
        (two_mm, side(setup, SM_B)),                                                     # 4 minimodule min ch
        (side(setup, SM_A, time={1: 12.0}), side(setup, SM_B)),                          # 5 missing mu
        (side(setup, SM_A, time={2: 12.0}), side(setup, SM_B)),                          # 6 invalid mu
        (side(setup, SM_A, energy=[2 * e for e in FULL]), side(setup, SM_B)),            # 7 energy window
        (side(setup, SM_A), side(setup, SM_D)),                                          # 8 pair
        (side(setup, SM_A), side(setup, SM_C)),                                          # accepted, swapped
    ]
    mu = slab_mu(setup)
    del mu[top(setup, SM_A, slab=1)]
    mu[top(setup, SM_A, slab=2)] = -1.0
    (features,), _ = features_of(tmp_path, setup, recs, lookup(setup, mu))
    assert features.codes.tolist() == [0, 1, 0, 2, 2, 4, 5, 6, 7, 8, 0]
    assert lsel.count_codes(features.codes) == {"accepted": 3, "unmapped channel": 1, "side min channels": 2,
                                                "minimodule min channels": 1, "missing energy calibration": 1,
                                                "invalid energy calibration": 1, "energy window": 1, "pair": 1}
    assert features.column("record").tolist() == [0, 2, 10]
    assert features.column("pair").tolist() == [7, 7, 9]
    assert features.column("swapped").tolist() == [0, 0, 1]
    assert features.column("time_channel_1").tolist() == [top(setup, SM_A)] * 3
    assert features.column("dt_ps").tolist() == [-300, 0, 0]
    assert features.column("t_min").tolist() == [T0, T0, T0]
    assert np.allclose(features.floats, 511.0)


@pytest.mark.fr("015-FR-5")
def test_energy_rows_of_the_selected_minimodule_zero_elsewhere(tmp_path, setup):
    first = side(setup, SM_A, energy=[200.0, 150.0, 100.0, 40.0, 21.0, 0.1]) + side(
        setup, SM_A, mm=6, time={}, energy=[5.0])
    (features,), _ = features_of(tmp_path, setup, [(first, side(setup, SM_B, energy=[100, 100, 100, 100, 111]))])
    rows = features.energy_rows
    assert rows.shape == (1, 2, 8) and rows.dtype == np.float32
    by_row = {setup.local_map[ch][2]: e for ch, e in zip(channels(setup, SM_A, 5)[1], [200.0, 150.0, 100.0, 40.0, 21.0])}
    assert rows[0, 0].tolist() == [by_row.get(r, 0.0) for r in range(8)]       # 0.1 below floor, mm 6 excluded
    assert rows[0, 1, :5].tolist() == [100, 100, 100, 100, 111] and not rows[0, 1, 5:].any()
    assert features.column("q_time_1")[0] == 18.0 and features.column("doi_ratio_1")[0] == pytest.approx(511 / 200)


@pytest.mark.fr("015-FR-2")
def test_batches_are_bounded_and_equal_one_pass(tmp_path, setup):
    recs = [(side(setup, SM_A, stamp=T0 + 1000 * k), side(setup, SM_B, stamp=T0 + 1000 * k + k)) for k in range(7)]
    small, _ = features_of(tmp_path, setup, recs, batch_records=3)
    (whole,), _ = features_of(tmp_path, setup, recs)
    assert [len(f.codes) for f in small] == [3, 3, 1]
    assert np.array_equal(np.concatenate([f.ints for f in small]), whole.ints)
    assert np.array_equal(np.concatenate([f.energy_rows for f in small]), whole.energy_rows)


# --- T2: timing (FR-4) ---------------------------------------------------------------------------------------------

def walk_lut(rule="top_earliest_mean"):
    edges = {"walk2d": [[0.0, 10.0, 100.0], [0.0, 0.5, 1.0]], "doi": [[0.0, 10.0]], "kev": [[0.0, 1000.0]]}
    offsets = {"walk2d": [[30.0, -70.0], [110.0, -20.0]], "doi": [7.0], "kev": [3.0]}
    events = {"walk2d": [[1, 1], [1, 1]], "doi": [1], "kev": [1]}
    lut = iw.WalkLut(edges, offsets, events, {b: [] for b in edges}, {})
    lut.manifest = {"timestamp_rule": rule}
    return lut


def full_skew(setup, rule="top_earliest_mean", drop=()):
    rng = np.random.default_rng(5)
    return DelayTable({int(c): float(rng.normal(0, 400)) for c in setup.time_channels.tolist() if c not in drop},
                      provenance={"timestamp_rule": rule})


def rule_records(setup, n=6):
    """Sides whose second TIME hit is sometimes earlier than the top (the rule acts)."""
    out = []
    for k in range(n):
        t = T0 + 10_000 * k
        a = side(setup, SM_A, time={3: 12.0, 4: 6.0}, stamp=t)
        a[1] = (t - 400 * (k % 3), 6.0, a[1][2])
        out.append((a, side(setup, SM_B, time={5: 9.0, 6: 4.0}, stamp=t + 150 * k)))
    return out


@pytest.mark.fr("015-FR-4")
def test_dt_is_the_rounded_correct_dt(tmp_path, setup):
    (features,), lk = features_of(tmp_path, setup, rule_records(setup))
    walk_mu = np.where(np.isfinite(lk.mu), 600.0, np.nan)
    skew, walk = full_skew(setup), walk_lut()
    dt, counts = lm_timing.Timing("em", skew, walk, "top_earliest_mean").dt(features, walk_mu)
    kev = {s: 511.0 / 600.0 * features.column(f"e_raw_{s}") for s in (1, 2)}
    expected, status, details = iw.correct_dt(
        features.column("dt_ps"), features.column("time_channel_1"), features.column("time_channel_2"),
        *[{"q_top": features.column(f"q_top_{s}"), "q_time": features.column(f"q_time_{s}"),
           "doi_ratio": features.column(f"doi_ratio_{s}"), "kev": kev[s]} for s in (1, 2)], skew, walk,
        rule="top_earliest_mean", candidates_1=features.candidates[:, 0], candidates_2=features.candidates[:, 1])
    assert (status == 0).all() and dt.dtype == np.int16
    assert dt.tolist() == np.rint(expected).astype(int).tolist()
    assert counts["corrected"] == len(dt) and counts["rule acts"] == details["rule acts"] > 0
    top_only = lm_timing.Timing("em", full_skew(setup, "top"), walk_lut("top"), "top").dt(features, walk_mu)[0]
    assert not np.array_equal(top_only, dt)


@pytest.mark.fr("015-FR-4")
def test_uncorrectable_events_get_the_sentinel_and_are_counted(tmp_path, setup):
    recs = rule_records(setup, 3)
    recs.append((side(setup, SM_C, stamp=T0), side(setup, SM_B, stamp=T0 + 40_000)))    # |dt| > int16
    (features,), lk = features_of(tmp_path, setup, recs)
    walk_mu = np.where(np.isfinite(lk.mu), 511.0, np.nan)
    walk_mu[top(setup, SM_B, slab=5)] = np.nan                       # record 0..2 side 2: walk input missing
    skew = full_skew(setup, drop=(top(setup, SM_C),))               # record 3 side 1: no delay
    timing = lm_timing.Timing("em", skew, walk_lut(), "top_earliest_mean")
    dt, counts = timing.dt(features, walk_mu)
    assert dt.tolist() == [lm_timing.UNCORRECTED_DT] * 4
    assert (counts["walk input missing"], counts["no applied delay"], counts["corrected"]) == (3, 1, 0)
    skew = full_skew(setup)
    dt, counts = lm_timing.Timing("em", skew, walk_lut(), "top_earliest_mean").dt(
        features, np.where(np.isfinite(lk.mu), 511.0, np.nan))
    assert counts["outside int16"] == 1 and dt[3] == lm_timing.UNCORRECTED_DT and counts["corrected"] == 3
    dt, counts = lm_timing.Timing.none().dt(features, walk_mu)
    assert (dt == lm_timing.UNCORRECTED_DT).all() and counts["no applied delay"] == 4


def lut_files(tmp_path, setup, *, rule="top_earliest_mean", walk_rule=None, walk_skew_sha=None, map_sha=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = np.zeros(len(setup.time_channels), solver.CHANNEL_DTYPE)
    rows["time_channel"] = setup.time_channels
    rows["delay_ps"], rows["sigma_ps"], rows["raw_sigma_ps"] = 10.0, 1.0, 1.0
    rows["status"], rows["apply"] = solver.CALIBRATED, True
    skew = tmp_path / "skew.csv"
    write_lut(skew, solver.Solution(rows, np.zeros(0, solver.PAIR_RESIDUAL_DTYPE), {}),
              {"scanner": "imas", "map_sha256": map_sha or setup.provenance["map_sha256"], "timestamp_rule": rule})
    walk = walk_lut(walk_rule or rule)
    walk.manifest = {"timestamp_rule": walk_rule or rule, "map_sha256": setup.provenance["map_sha256"],
                     "skew_lut_sha256": walk_skew_sha or lm_timing.sha256(skew)}
    walk.write(tmp_path / "walk.json")
    return skew, tmp_path / "walk.json"


@pytest.mark.fr("015-FR-4")
def test_tables_are_checked_together(tmp_path, setup):
    skew, walk = lut_files(tmp_path / "ok", setup)
    timing = lm_timing.from_options(setup, skew=skew, walk=walk, rule="top_earliest_mean")
    assert timing.describe()["walk_rule"] == "top_earliest_mean" and timing.describe()["skew_applied_channels"] > 0
    with pytest.raises(ts.RuleError, match="'top_earliest_mean', applied under 'top'"):
        lm_timing.from_options(setup, skew=skew, walk=walk, rule="top")
    skew2, walk2 = lut_files(tmp_path / "rule", setup, walk_rule="top")
    with pytest.raises(ts.RuleError):
        lm_timing.from_options(setup, skew=skew2, walk=walk2, rule="top_earliest_mean")
    skew3, walk3 = lut_files(tmp_path / "pair", setup, walk_skew_sha="0" * 64)
    with pytest.raises(iw.core.WalkError, match="was fitted with skew LUT"):
        lm_timing.from_options(setup, skew=skew3, walk=walk3, rule="top_earliest_mean")
    skew4, walk4 = lut_files(tmp_path / "map", setup, map_sha="f" * 64)
    with pytest.raises(LutError, match="another map"):
        lm_timing.from_options(setup, skew=skew4, walk=walk4, rule="top_earliest_mean")
    with pytest.raises(ValueError, match="exactly one timing"):
        lm_timing.from_options(setup, skew=skew, walk=walk, none=True)


@pytest.mark.fr("015-FR-4")
def test_legacy_skew_is_additive_top_rule(tmp_path, setup):
    a, b = top(setup, SM_A), top(setup, SM_B)
    legacy = tmp_path / "skew.txt"
    legacy.write_text(f"{a}\t120\n{b}\t9999\n", encoding="utf-8")
    timing = lm_timing.from_options(setup, legacy=legacy)
    (features,), lk = features_of(tmp_path, setup, [(side(setup, SM_A), side(setup, SM_B, stamp=T0 + 300))])
    assert timing.dt(features, lk.walk_mu)[0].tolist() == [lm_timing.UNCORRECTED_DT]
    legacy.write_text(f"{a}\t120\n{b}\t-30\n", encoding="utf-8")
    dt, counts = lm_timing.from_options(setup, legacy=legacy).dt(features, lk.walk_mu)
    assert dt.tolist() == [(T0 + 120) - (T0 + 300 - 30)] and counts["corrected"] == 1


# --- T3: records (FR-5, FR-6) -------------------------------------------------------------------------------------

@pytest.mark.fr("015-FR-6")
def test_lm_layout_and_header_are_the_legacy_ones():
    assert records.LM_DTYPE == np.dtype(CoincidenceV5) and records.LM_DTYPE.itemsize == 24
    raw = records.header_bytes(120, 2.5, "Na22")
    header = LMHeader.from_buffer_copy(raw)
    assert len(raw) == ctypes.sizeof(LMHeader) == 176
    assert (header.identifier, header.isotope, header.acqTime, header.activity) == (b"IMAS", b"Na22", 120.0, 2.5)
    assert (header.moduleNumber, header.ringNumber, header.ringDistance, list(header.version)) == (120, 5, 820, [9, 5])
    assert (header.detectorPixelsX, header.detectorPixelsY) == (100, 100)
    assert header.detectorPixelSizeX == pytest.approx(1.0322, rel=1e-6) and header.measurementTime == 10
    with pytest.raises(ValueError):
        records.header_bytes(1, 0, "C11")


@pytest.mark.fr("015-FR-5", "015-FR-6")
def test_records_swap_sides_and_negate_dt():
    chunk = np.zeros(3, job.CHUNK_DTYPE)
    chunk["t_min"] = [5_000_000_000, 127_500_000_000, 5_000_000_000]
    chunk["pair"], chunk["swapped"], chunk["dt"] = [7, 9, 9], [0, 1, 1], [-300, 250, lm_timing.UNCORRECTED_DT]
    chunk["kev"] = [[500.4, 520.6], [460.5, 461.5], [511, 512]]
    chunk["x_mm"] = [[0.5, 103.3], [50.0, 2.0], [1, 2]]
    chunk["y0_mm"] = 64.513
    y_nn = np.array([[-64.6, 0.0], [10.0, -30.0], [0, 0]], np.float32)
    doi = np.array([[3.9, -0.7], [12.2, 19.99], [1, 2]], np.float32)
    out = records.build_chunk(chunk, y_nn, doi, t0=4_000_000_000)
    assert out["time"].tolist() == pytest.approx([0.001, 0.1235, 0.001])
    assert out["energy1"].tolist() == [500, 462, 512] and out["energy2"].tolist() == [521, 460, 511]
    assert out["xPosition1"].tolist() == [0, 1, 1] and out["xPosition2"].tolist() == [100, 48, 0]
    assert out["yPosition1"].tolist() == [-1, 33, 62] and out["yPosition2"].tolist() == [62, 72, 62]
    assert out["zPosition1"].tolist() == [3, 19, 2] and out["zPosition2"].tolist() == [0, 12, 1]
    assert out["dt"].tolist() == [-300, -250, lm_timing.UNCORRECTED_DT]
    assert out["pair"].tolist() == [7, 9, 9] and (out["amount"] == 1.0).all()


@pytest.mark.fr("015-FR-6")
def test_pixels_follow_the_legacy_searchsorted():
    edges = np.linspace(0, 103.22, 101)
    values = np.array([-0.1, 0.0, 1.0321, 1.0322, 51.6, 103.21, 103.22, 104.0])
    expected = [int(np.searchsorted(edges, v, side="right")) - 1 for v in values]
    assert records.pixels(values).tolist() == expected == [-1, 0, 0, 1, 49, 99, 100, 100]


# --- T4: job (FR-7 to FR-10) --------------------------------------------------------------------------------------

def job_inputs(base, setup, sizes=(5, 3, 4), *, truncate=None):
    base.mkdir(parents=True, exist_ok=True)
    paths = []
    for f, n in enumerate(sizes):
        recs = []
        for k in range(n):
            t = T0 + 10**9 * f + 20_000 * k
            partner = (SM_B, SM_C, SM_D)[k % 3]
            recs.append((side(setup, SM_A, stamp=t, time={3: 12.0, 4: 6.0}), side(setup, partner, stamp=t + 37 * k)))
        path = write(base / f"acq_run_coincCompact_{f:08d}.ldat", recs)
        if truncate == f:
            path.write_bytes(path.read_bytes()[:-5])
        paths.append(str(path))
    return paths[::-1]                       # natural order is restored by the job


def job_settings(base, setup, paths, out, **changes):
    values = dict(config=str(CONFIG), slab_en_map=str(mu_file(base / "mu.tsv", slab_mu(setup))), inputs=paths,
                  out_dir=str(out), nn=("linear",), pairs=str(pairs_file(base / "pairs.txt", PAIRS)),
                  no_timing=True, rule="top", workers=0, root=str(ROOT))
    values.update(changes)
    return job.Settings(**values)


def lm_records(path):
    data = Path(path).read_bytes()
    return np.frombuffer(data[176:], records.LM_DTYPE)


@pytest.mark.fr("015-FR-6", "015-FR-7", "015-FR-9")
def test_job_merges_segments_with_provenance(tmp_path, setup):
    paths = job_inputs(tmp_path / "in", setup)
    provenance = job.run(job_settings(tmp_path, setup, paths, tmp_path / "out"), log=lambda m: None)
    final = tmp_path / "out" / "acq_run_coincCompact_all.lm"
    assert provenance["output"] == str(final)
    out = lm_records(final)
    # partners B, C, D cycle; D is unpaired. Natural order of files, then record order
    assert len(out) == 4 + 2 + 3 and provenance["totals"]["coincidences"] == 9
    assert provenance["totals"]["records"] == 12 and {k: v for k, v in provenance["totals"]["codes"].items() if v} == {"accepted": 9, "pair": 3}
    assert out["pair"].tolist() == [7, 9, 7, 9, 7, 9, 7, 9, 7]
    assert out["time"].tolist() == pytest.approx([0, 0, 0, 0, 0.001, 0.001, 0.002, 0.002, 0.002])
    assert (out["energy1"] == 511).all() and (out["dt"] == lm_timing.UNCORRECTED_DT).all()
    assert [f["name"] for f in provenance["files"]] == [Path(p).name for p in sorted(paths)]
    assert provenance["inputs"]["pairs"]["sha256"] == hashlib.sha256(Path(tmp_path / "pairs.txt").read_bytes()).hexdigest()
    assert not list((tmp_path / "out").glob("*_0000000?.lm*"))                  # segments removed
    assert (tmp_path / "out" / "acq_run_coincCompact_log.txt").is_file()
    assert json.loads((tmp_path / "out" / "acq_run_coincCompact_all.json").read_text())["t0_ps"] == T0


@pytest.mark.fr("015-FR-7")
def test_worker_count_does_not_change_the_lm(tmp_path, setup):
    paths = job_inputs(tmp_path / "in", setup, sizes=(40, 25, 33, 10))
    outputs = []
    for workers in (0, 1, 3):
        job.run(job_settings(tmp_path, setup, paths, tmp_path / f"w{workers}", workers=workers, batch_records=7),
                log=lambda m: None)
        outputs.append((tmp_path / f"w{workers}" / "acq_run_coincCompact_all.lm").read_bytes())
    assert outputs[0] == outputs[1] == outputs[2] and len(outputs[0]) > 176


@pytest.mark.fr("015-FR-8")
def test_continue_skips_completed_segments(tmp_path, setup):
    paths = job_inputs(tmp_path / "in", setup)
    settings = job_settings(tmp_path, setup, paths, tmp_path / "out", keep_segments=True)
    first = job.run(settings, log=lambda m: None)
    with pytest.raises(job.JobError, match="already holds LM segments"):
        job.run(settings, log=lambda m: None)
    changed = sorted(paths)[1]
    Path(changed).write_bytes(Path(changed).read_bytes() * 2)
    messages = []
    again = job.run(job_settings(tmp_path, setup, paths, tmp_path / "out", keep_segments=True, resume=True),
                    log=messages.append)
    assert "3 files, 2 already complete, 1 to process" in messages[0]
    assert [m for m in messages if m.startswith("done")] == [m for m in messages if Path(changed).name in m]
    fresh = job.run(job_settings(tmp_path, setup, paths, tmp_path / "fresh"), log=lambda m: None)
    assert Path(again["output"]).read_bytes() == Path(fresh["output"]).read_bytes()
    assert again["totals"]["coincidences"] == first["totals"]["coincidences"] + 2


@pytest.mark.fr("015-FR-2", "015-FR-9")
def test_a_failed_file_writes_no_merged_lm(tmp_path, setup):
    paths = job_inputs(tmp_path / "in", setup, truncate=1)
    messages = []
    with pytest.raises(job.JobError, match="1 file\\(s\\) failed.*00000001"):
        job.run(job_settings(tmp_path, setup, paths, tmp_path / "out"), log=messages.append)
    assert any("truncated LDAT hit record" in m for m in messages)
    out = tmp_path / "out"
    assert not list(out.glob("*_all.lm")) and not list(out.glob("*.part"))
    assert sorted(p.name for p in out.glob("*.lm")) == ["acq_run_coincCompact_00000000.lm",
                                                       "acq_run_coincCompact_00000002.lm"]


@pytest.mark.fr("015-FR-1", "015-FR-9")
def test_bad_inputs_stop_before_processing(tmp_path, setup):
    paths = job_inputs(tmp_path / "in", setup)
    with pytest.raises(job.JobError, match="is not a file"):
        job.run(job_settings(tmp_path, setup, paths + [str(tmp_path / "missing.ldat")], tmp_path / "o1"))
    with pytest.raises(job.JobError, match="walk encal"):
        skew, walk = lut_files(tmp_path / "luts", setup)
        job.run(job_settings(tmp_path, setup, paths, tmp_path / "o2", no_timing=False, skew=str(skew),
                             walk=str(walk), rule="top_earliest_mean"))
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists()


@pytest.mark.fr("015-FR-4", "015-FR-10")
def test_timed_debug_run(tmp_path, setup):
    paths = job_inputs(tmp_path / "in", setup, sizes=(6,))
    skew, walk = lut_files(tmp_path / "luts", setup)
    encal = mu_file(tmp_path / "walk_mu.tsv", slab_mu(setup))
    provenance = job.run(job_settings(tmp_path, setup, paths, tmp_path / "out", no_timing=False, skew=str(skew),
                                      walk=str(walk), walk_encal=str(encal), rule="top_earliest_mean", debug=True),
                         log=lambda m: None)
    out = lm_records(provenance["output"])
    # equal delays and walk offsets cancel: dt = raw top difference, negated for swapped pairs
    raw = [-(37 * k) for k in range(6) if k % 3 != 2]
    swapped = [k % 3 == 1 for k in range(6) if k % 3 != 2]
    assert out["dt"].tolist() == [-d if s else d for d, s in zip(raw, swapped)]
    assert provenance["totals"]["timing"]["corrected"] == 4 and provenance["timing"]["rule"] == "top_earliest_mean"
    assert (tmp_path / "out" / "total_energy.png").is_file() and (tmp_path / "out" / "sm_hits.png").is_file()


# --- T5: command line (FR-1) --------------------------------------------------------------------------------------

def args_for(*extra):
    from docopt import docopt
    from src.imas.listmode import __main__ as cli
    return docopt(cli.__doc__, argv=[*extra, "conf.yaml", "mu.tsv", "b_2.ldat", "a_10.ldat"])


@pytest.mark.fr("015-FR-1")
def test_cli_defaults_follow_the_rule():
    settings = settings_from_args(args_for("-w", "2", "--luts", "L", "-o", "OUT"))
    assert settings.skew == str(Path("L") / PAIRS_BY_RULE["top_earliest_mean"][0])
    assert settings.walk == str(Path("L") / PAIRS_BY_RULE["top_earliest_mean"][1])
    assert settings.walk_encal == str(Path("L") / WALK_ENCAL) and settings.rule == "top_earliest_mean"
    assert settings.nn == ("keras", "/scratch/imas_files_cal/IMASNoCatY.h5", "/scratch/imas_files_cal/IMASNoCatDOI.h5",
                           16384)
    assert settings.pairs == "/scratch/imas_files_cal/pares_5SR.txt" and settings.workers == 2
    assert (settings.acq_time, settings.activity, settings.isotope, settings.out_dir) == (300.0, 0.0, "F18", "OUT")
    top = settings_from_args(args_for("--rule", "top", "--luts", "L", "-o", "OUT"))
    assert top.skew == str(Path("L") / PAIRS_BY_RULE["top"][0]) and top.walk == str(Path("L") / PAIRS_BY_RULE["top"][1])
    legacy = settings_from_args(args_for("--legacy-skew", "s.txt", "-o", "OUT"))
    assert (legacy.legacy_skew, legacy.skew, legacy.walk, legacy.rule) == ("s.txt", None, None, "top")
    none = settings_from_args(args_for("--no-timing", "-o", "OUT", "--walk", "none"))
    assert none.no_timing and none.skew is None
    skew_only = settings_from_args(args_for("--walk", "none", "--skew", "em.csv", "--rule", "top", "-o", "OUT"))
    assert (skew_only.skew, skew_only.walk, skew_only.walk_encal) == ("em.csv", None, None)
    auto = settings_from_args(args_for("--lm-root", "R"), now=datetime(2026, 11, 3, 12, 0, 0))
    assert Path(auto.out_dir) == Path("R") / "lm-20261103-120000-"        # first natural input "a_10.ldat"
    with pytest.raises(job.JobError, match="exclude each other"):
        settings_from_args(args_for("--no-timing", "--legacy-skew", "s.txt"))
    with pytest.raises(job.JobError, match="either --out or --continue"):
        settings_from_args(args_for("-o", "A", "-c", "B"))


@pytest.mark.fr("015-FR-1")
def test_default_out_dir_is_the_legacy_one():
    path = job.default_out_dir("R", "/x/20261019_Na22_1cm_cFOV_test_300s_coincCompact_00000034.ldat",
                               datetime(2026, 11, 3, 12, 0, 5))
    assert path == Path("R") / "lm-20261103-120005-Na22_1cm_cFOV_test_300s"
    assert job.base_name(Path("D") / "20261019_Na22_test_coincCompact_00000034.lm") == Path("D") / "20261019_Na22_test_coincCompact"


@pytest.mark.fr("015-FR-1")
def test_cli_names_a_missing_input_and_fails(tmp_path, capsys):
    code = main(["--no-timing", "-o", str(tmp_path / "o"), "-w", "0", str(CONFIG), str(tmp_path / "mu.tsv"),
                 str(tmp_path / "x.ldat")])
    assert code == 1 and "x.ldat is not a file" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--help"])


# --- T6: real-data parity (criterion 2) ----------------------------------------------------------------------------

GOLDEN = ROOT / "tests/data/golden/timing/015_t6_imas_lm_prefix.bin"
NA22 = "IMAS/20261019_Na22_1cm_cFOV_test_300s_coincCompact_00000034.ldat"
REAL_TABLES = {"skew": "IMAS_work/job_rule/lut_em_walk-f5d870aa6390_rule-mean-freeze200.csv",
               "walk": "IMAS_work/walk/walk_rule4.json",
               "encal": "IMAS_work/job_rule/skew_20251119_run7/encal/encal.tsv",
               "nn_y": "IMAS/LUTs/IMASNoCatY.h5", "nn_doi": "IMAS/LUTs/IMASNoCatDOI.h5", "pairs": "IMAS/LUTs/pares_5SR.txt"}


@pytest.mark.real_data
@pytest.mark.slow
@pytest.mark.fr("015-FR-3", "015-FR-4", "015-FR-5", "015-FR-6")
def test_real_prefix_matches_the_legacy_reference(tmp_path, real_data_file):
    files = {key: str(real_data_file(rel)) for key, rel in REAL_TABLES.items()}
    ldat = real_data_file(NA22)
    record = json.loads(GOLDEN.with_name(GOLDEN.name + ".provenance.json").read_text(encoding="utf-8"))
    for key in ("skew", "walk", "encal"):
        assert lm_timing.sha256(files[key]) == record["sources"][REAL_TABLES[key]]
    settings = job.Settings(config=str(CONFIG), slab_en_map=files["encal"], inputs=[str(ldat)], out_dir=str(tmp_path),
                            nn=("keras", files["nn_y"], files["nn_doi"]), pairs=files["pairs"], skew=files["skew"],
                            walk=files["walk"], walk_encal=files["encal"], rule="top_earliest_mean", workers=0,
                            max_records=100_000, root=str(ROOT))
    provenance = job.run(settings, log=lambda m: None)
    fast = lm_records(provenance["output"])
    golden = np.fromfile(GOLDEN, records.LM_DTYPE)
    assert len(fast) == len(golden) == record["settings"]["accepted"]
    for name in records.LM_DTYPE.names:
        if name != "dt":
            assert np.array_equal(fast[name], golden[name]), name
    difference = np.abs(fast["dt"].astype(int) - golden["dt"].astype(int))
    assert difference.max() <= 1 and np.array_equal(fast["dt"] == -32700, golden["dt"] == -32700)
    assert provenance["totals"]["codes"] == record["settings"]["codes"]
