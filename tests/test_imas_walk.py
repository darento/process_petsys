"""Spec 012: IMAS time-walk inputs, observations with walk, job, bar fit, correction and export."""
import numpy as np
import pytest

from imas_skew_helpers import CONFIG, FULL, T0, channels, job_file, job_splits, mu_file, side, write
from src.imas.skew import select as sel
from src.imas.skew import walk as iw
from src.imas.skew.setup import ImasSetup
import timing_walk_helpers as h

A, B = (0, 5), (60, 6)


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    return job_splits(tmp_path_factory.mktemp("job"))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    probe = ImasSetup.from_files(CONFIG)
    path = mu_file(tmp_path_factory.mktemp("mu") / "mu.tsv", {ch: 511.0 for ch in probe.time_channels.tolist()})
    return ImasSetup.from_files(CONFIG, path)


# --- T2: walk inputs (FR-1) ---------------------------------------------------------------
@pytest.mark.fr("012-FR-1")
def test_missing_inputs_get_status_never_zero():
    q_top = np.array([12.0, -1.0, 12.0, 12.0, 12.0, 0.0, np.nan, 12.0])
    q_time = np.array([18.0, 18.0, 0.0, 18.0, 18.0, 18.0, 18.0, 18.0])
    ratio = np.array([2.5, 0.0, 2.5, 0.0, 2.5, 2.5, 2.5, np.inf])
    kev = np.array([511.0, np.nan, 511.0, np.nan, np.nan, 511.0, 511.0, 511.0])
    values, status = iw.walk_inputs(q_top, q_time, ratio, kev)
    assert [iw.STATUS[s] for s in status] == ["ok", "no TIME charge", "no TIME charge", "no ENERGY charge",
                                              "no calibration", "ok", "no TIME charge", "no ENERGY charge"]
    assert values["share"][0] == pytest.approx(12 / 18) and values["share"][5] == 0.0   # zero charge is a value
    for name in iw.INPUTS:
        assert np.isnan(values[name][status != 0]).all()
    assert iw.count_status(status) == {"no TIME charge": 3, "no ENERGY charge": 2, "no calibration": 1}


@pytest.mark.fr("012-FR-1")
def test_inputs_from_selected_features(tmp_path, setup):
    lookup = setup.lookup()
    records = [(side(setup, *A), side(setup, *B, time={5: 20.0, 6: 5.0, 4: 5.0}, stamp=T0 - 250)),
               (side(setup, *A, time={2: 9.0}, energy=[150.0, 150.0, 100.0, 80.0, 31.0]), side(setup, *B))]
    path = write(tmp_path / "f.ldat", records)
    features = next(sel.feature_batches(path, lookup, sel.Selection.production(lookup)))
    assert (features.codes == 0).all()
    one, status1 = iw.side_inputs(features.column, 1)
    two, status2 = iw.side_inputs(features.column, 2)
    assert not status1.any() and not status2.any()
    np.testing.assert_allclose(one["charge"], [12.0, 9.0])
    np.testing.assert_allclose(one["share"], [12 / 18, 1.0])
    np.testing.assert_allclose(one["doi_ratio"], [sum(FULL) / 200, 511 / 150])
    np.testing.assert_allclose(one["kev"], [511.0, 511.0], rtol=1e-6)
    np.testing.assert_allclose(two["charge"], [20.0, 12.0])
    np.testing.assert_allclose(two["share"], [20 / 30, 12 / 18])
    np.testing.assert_allclose(two["kev"], [511.0, 511.0], rtol=1e-6)


@pytest.mark.fr("012-FR-2")
def test_imas_model_is_spec_009_table():
    assert iw.MODEL.inputs == iw.INPUTS and iw.WalkLut.model is iw.MODEL
    assert iw.DEFAULT_BINS == {"charge": 20, "share": 10, "doi_ratio": 20, "kev": 20}
    assert iw.LUT_SCHEMA == "imas-walk-lut/1" and "doi[doi_ratio]" in iw.CONVENTION


# --- T3: observations carry walk inputs (FR-4) ----------------------------------------------
@pytest.fixture(scope="module")
def walk_files(tmp_path_factory, setup):
    """Diametric coincidences with varied TIME charges, so swapped sides carry different walk ingredients."""
    base = tmp_path_factory.mktemp("imas_walk_stream")
    rng = np.random.default_rng(17)
    records = []
    for _ in range(240):
        k, ring = int(rng.integers(0, 24)), int(rng.integers(0, 5))
        a, b = 24 * ring + k, 24 * (4 - ring) + (k + 12) % 24
        ta = {int(rng.integers(0, 8)): float(rng.uniform(5, 30)), 7: float(rng.uniform(0, 4))}
        tb = {int(rng.integers(0, 7)): float(rng.uniform(5, 30)), 7: float(rng.uniform(0, 4))}
        energy = sorted(rng.uniform(20, 160, 5).tolist(), reverse=True)
        energy = [e * 511 / sum(energy) for e in energy]
        records.append((side(setup, a, int(rng.integers(0, 16)), time=ta, energy=energy),
                        side(setup, b, int(rng.integers(0, 16)), time=tb, stamp=T0 - int(rng.integers(-900, 900)))))
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


@pytest.mark.fr("012-FR-4")
def test_observations_carry_walk_ingredients(tmp_path, setup, walk_files):
    from src.timing import cache as core_cache
    plain = _rows(_stream(tmp_path / "plain", setup, walk_files))
    carried_set = _stream(tmp_path / "walk", setup, walk_files, carry_walk=True)
    carried = _rows(carried_set)
    assert carried_set.carries_walk and carried.dtype == core_cache.OBSERVATION_WALK_DTYPE
    assert len(carried) > 150 and carried["swapped"].any() and not carried["swapped"].all()
    base = list(core_cache.OBSERVATION_DTYPE.names)
    assert all(np.array_equal(carried[n], plain[n]) for n in base)            # same events and values
    lookup = setup.lookup()
    from src.imas.skew import select as sel
    features = {}
    for index, path in enumerate(walk_files):
        for batch in sel.feature_batches(path, lookup, sel.Selection.production(lookup)):
            for k, record in enumerate(batch.column("record").tolist()):
                features[(index, record)] = {n: batch.column(n)[k] for n in ("q_top_1", "q_time_1", "doi_ratio_1",
                                                                             "q_top_2", "q_time_2", "doi_ratio_2")}
    for row in carried:
        f = features[(int(row["file"]), int(row["record"]))]
        i, j = ("2", "1") if row["swapped"] else ("1", "2")
        for name in ("q_top", "q_time", "doi_ratio"):
            assert row[f"{name}_i"] == np.float32(f[f"{name}_{i}"]) and row[f"{name}_j"] == np.float32(f[f"{name}_{j}"])
    values, status = iw.side_inputs(lambda name: carried[name], "i")
    assert not status.any() and np.allclose(values["kev"], carried["energy_kev_i"])


@pytest.mark.fr("012-FR-4")
def test_sets_with_and_without_walk_refuse_each_other(tmp_path, setup, walk_files):
    from src.timing import cache as core_cache
    _stream(tmp_path / "plain", setup, walk_files)
    with pytest.raises(core_cache.CacheMismatch, match="different inputs"):
        _stream(tmp_path / "plain", setup, walk_files, carry_walk=True)
    _stream(tmp_path / "walk", setup, walk_files, carry_walk=True)
    with pytest.raises(core_cache.CacheMismatch, match="different inputs"):
        _stream(tmp_path / "walk", setup, walk_files)


@pytest.mark.fr("012-FR-4")
def test_cached_and_streamed_walk_rows_agree(tmp_path, setup, walk_files):
    from src.imas.skew import cache as ic
    from src.imas.skew.geometry import ImasGeometry
    inputs = [{"index": i, "path": p} for i, p in enumerate(walk_files)]
    cache = ic.build_feature_cache(tmp_path / "features", inputs, config_path=CONFIG, encal_path=setup.encal_path,
                                   holdout=ic.all_train(range(2)), energy_calibration_prefix={}, block_rows=50)
    built = ic.build_observations(cache, ImasGeometry.nominal(setup), tmp_path / "obs", partitions=3, buffer_rows=40,
                                  carry_walk=True)
    streamed = _stream(tmp_path / "s", setup, walk_files, carry_walk=True)
    assert built.carries_walk and _rows(built).tobytes() == _rows(streamed).tobytes()


# --- T4: walk-corrected observations and the job's walk_lut (FR-4) ---------------------------
def _walk_columns(rng, channels, gain):
    """One side's raw walk ingredients with a per-channel charge gain (so walk biases skew), and its true walk."""
    n = len(channels)
    charge = rng.uniform(5, 35, n) * gain[channels]
    share = np.where(rng.random(n) < 0.4, 1.0, rng.uniform(0.4, 0.95, n))
    ratio, kev = rng.uniform(1.2, 3.0, n), np.clip(rng.normal(500, 60, n), 409, 612)
    return (charge, share, ratio, kev), h.true_walk(charge, share, ratio, kev)


@pytest.fixture(scope="module")
def walk_set(tmp_path_factory):
    """Bar-like synthetic set: injected delays and injected walk, 2 % of rows missing a walk input, 10 % held out."""
    from imas_skew_helpers import channel_table, lors, observation_set, skew_events
    from src.imas.skew.geometry import ImasGeometry
    probe = ImasSetup.from_files(CONFIG)
    geometry = ImasGeometry.nominal(probe)
    rng = np.random.default_rng(31)
    table = channel_table(probe, (48, 49, 60, 61), (5,))
    sm = {s: rng.uniform(-2500, 2500) for s in np.unique(table["sm"])}
    delays = {int(r["ch"]): sm[r["sm"]] + rng.uniform(-300, 300) for r in table}
    ci, cj, tof = lors(geometry, table, 60_000, rng)
    ci, cj, dt, tof = skew_events(ci, cj, tof, delays, rng, background=0.05)
    gain = np.ones(int(table["ch"].max()) + 1)
    gain[table["ch"]] = rng.uniform(0.6, 1.4, len(table))
    (qi, si, ri, ki), wi = _walk_columns(rng, ci, gain)
    (qj, sj, rj, kj), wj = _walk_columns(rng, cj, gain)
    dt = dt + wi - wj
    missing = rng.random(len(ci)) < 0.02
    walk = {"q_top_i": qi, "q_time_i": np.where(missing, 0.0, qi / si), "doi_ratio_i": ri, "energy_kev_i": ki,
            "q_top_j": qj, "q_time_j": qj / sj, "doi_ratio_j": rj, "energy_kev_j": kj}
    roles = np.where(rng.random(len(ci)) < 0.1, 2, 0)
    raw = {"ci": ci, "cj": cj, "dt": dt, "tof": tof, "walk": walk, "roles": roles}
    observations = observation_set(tmp_path_factory.mktemp("walk_set") / "obs", ci, cj, dt, tof, walk=walk,
                                   roles=roles)
    side_i = {"charge": qi, "share": si, "doi_ratio": ri, "kev": ki}
    side_j = {"charge": qj, "share": sj, "doi_ratio": rj, "kev": kj}
    truth, _ = iw.core.fit_tables(side_i, side_j, wi - wj, ~missing, iw.DEFAULT_BINS, core_ps=1e6, iterations=1,
                                  minimum=20, manifest={"fit_kind": "synthetic truth"}, table=iw.WalkLut)
    return probe, observations, delays, truth, missing, roles, raw


def _delay_error(solution, delays):
    rows = solution.channels[solution.channels["apply"]]
    error = rows["delay_ps"] - np.array([delays[c] for c in rows["time_channel"].tolist()])
    return error - error.mean()


@pytest.mark.fr("012-FR-4")
def test_walk_corrected_em_recovers_delays(walk_set):
    from src.imas.skew import solve
    setup, observations, delays, truth, missing, roles, _ = walk_set
    plain, _ = solve.em_estimator(observations, setup)
    view = iw.WalkCorrected(observations, truth)
    corrected, _ = solve.em_estimator(view, setup)
    without, with_walk = _delay_error(plain, delays), _delay_error(corrected, delays)
    assert len(with_walk) >= 30 and np.sqrt(np.mean(with_walk ** 2)) < 0.5 * np.sqrt(np.mean(without ** 2))
    counts = view.counts()
    assert counts["train"]["walk input missing"] == int(np.count_nonzero(missing & (roles == 0)))
    assert counts["test"]["walk input missing"] == int(np.count_nonzero(missing & (roles == 2)))
    assert counts["train"]["rows"] + counts["test"]["rows"] == len(missing)


@pytest.mark.fr("012-FR-4")
def test_rows_without_walk_inputs_refused(tmp_path):
    from imas_skew_helpers import observation_set
    plain = observation_set(tmp_path / "obs", np.array([1, 2]), np.array([3, 4]), np.zeros(2), np.zeros((2, 4)))
    assert not plain.carries_walk
    with pytest.raises(iw.core.WalkError, match="without walk inputs"):
        iw.WalkCorrected(plain, None).partition(0)


@pytest.mark.slow
@pytest.mark.fr("012-FR-4")
def test_job_observe_only_then_walk_solve(tmp_path, splits):
    import json
    from src.imas.skew import job as ij
    from src.timing.lut import CanonicalLut
    work = tmp_path / "work"
    plain_job = ij.load_job(job_file(tmp_path / "job.json", splits, estimators=["em"], per_run_solves=False))
    report = ij.run_job(plain_job, work, progress=lambda _: None, observe_only=True)
    assert (work / "job_report_observe.json").is_file() and not list(work.glob("lut_*.csv"))
    assert "estimators" not in report and set(report["runs"]) == {"r1", "r2"}
    probe = ImasSetup.from_files(CONFIG)
    edges = {"walk2d": [np.array([0.0, 8, 20]), np.array([0.5, 0.9, 1.1])], "doi": [np.array([1.0, 4, 6])],
             "kev": [np.array([300.0, 500, 700])]}
    offsets = {"walk2d": np.array([[0.0, 30], [0, -30]]), "doi": np.array([0.0, 0]), "kev": np.array([5.0, -5])}
    events = {b: np.full(v.shape, 500) for b, v in offsets.items()}
    stamp = ij.fingerprint(plain_job)["sha256"]
    lut = iw.WalkLut.from_fit(edges, offsets, events, {"map_sha256": probe.provenance["map_sha256"],
                                                       "skew_lut_sha256": "0" * 64, "job_sha256": stamp})
    lut.write(tmp_path / "walk.json")
    walk_job = ij.load_job(job_file(tmp_path / "job_walk.json", splits, estimators=["em"], per_run_solves=False,
                                    walk_lut=str(tmp_path / "walk.json")))
    assert ij.fingerprint(walk_job) == ij.fingerprint(plain_job)            # same streamed events
    report = ij.run_job(walk_job, work, progress=lambda _: None)
    suffix = f"_walk-{lut.sha256[:12]}"
    skew = CanonicalLut.read(work / f"lut_em{suffix}.csv")
    assert skew.manifest["provenance"]["walk_lut_sha256"] == lut.sha256
    assert json.loads((work / f"job_report{suffix}.json").read_text(encoding="utf-8"))["walk"]["walk_lut_sha256"] \
        == lut.sha256
    counts = report["walk"]["pooled_counts"]
    assert counts["train"]["rows"] > 0 and counts["train"]["walk input missing"] == 0
    assert not (work / "lut_em.csv").exists()
    with pytest.raises(ij.JobError, match="never overwritten"):
        ij.run_job(walk_job, work, progress=lambda _: None)
    for name, manifest, match in (("map", {"map_sha256": "f" * 64, "job_sha256": stamp}, "another map"),
                                  ("job", {"map_sha256": probe.provenance["map_sha256"], "job_sha256": "e" * 64},
                                   "another job")):
        other = iw.WalkLut.from_fit(edges, offsets, events, manifest)
        other.write(tmp_path / f"other_{name}.json")
        with pytest.raises(ij.JobError, match=match):
            ij.run_job(ij.load_job(job_file(tmp_path / f"job_{name}.json", splits, estimators=["em"],
                                            walk_lut=str(tmp_path / f"other_{name}.json"))), work,
                       progress=lambda _: None)
    old = tmp_path / "old"                      # a spec 011 work directory: streamed without walk inputs
    (old / "r1" / "observations").mkdir(parents=True)
    (old / "job_fingerprint.json").write_bytes((work / "job_fingerprint.json").read_bytes())
    (old / "r1" / "observations" / "fingerprint.json").write_text('{"kind": "spec 011"}', encoding="utf-8")
    with pytest.raises(ij.JobError, match="without walk inputs"):
        ij.run_job(plain_job, old, progress=lambda _: None, observe_only=True)


# --- T5: bar walk fit (FR-3) ----------------------------------------------------------------
FAST = {"bins": dict(iw.DEFAULT_BINS), "min_events": 50, "core_ps": 600.0, "iterations": 3, "branch_iterations": 2,
        "max_events": 10 ** 9}


def _bar_fit(observations, delays):
    from src.timing.lut import DelayTable
    events, counts, sampling = iw.bar_events([observations], DelayTable(delays), max_events=FAST["max_events"])
    lut, widths, profiles = iw.fit_bar(events, FAST, {"fit_kind": "synthetic"})
    return lut, widths, profiles, counts, events


@pytest.mark.fr("012-FR-3")
def test_bar_fit_recovers_injected_walk(walk_set):
    setup, observations, delays, truth, missing, roles, _ = walk_set
    lut, widths, profiles, counts, events = _bar_fit(observations, delays)
    assert counts["train"]["walk input missing"] == int(np.count_nonzero(missing & (roles == 0)))
    assert counts["heldout"]["kept"] == len(events["heldout"]["dt"]) > 3000
    h1, h2 = iw.sides(events["heldout"])
    fitted, _, _ = iw.core.walk_difference(lut, h1, h2)
    injected, _, _ = iw.core.walk_difference(truth, h1, h2)
    assert np.sqrt(np.mean((fitted - injected) ** 2)) < 0.2 * np.sqrt(np.mean(injected ** 2))
    before, after = widths["heldout"]["before"], widths["heldout"]["after"]
    assert after["core_rms_ps"] < before["core_rms_ps"] - 10 and after["robust_sigma_ps"] < before["robust_sigma_ps"]
    assert lut.manifest["branch_flips"][0] is None and len(lut.manifest["branch_iterations"]) == 2
    profile = profiles["bar held-out (side i)"]
    assert set(profile) == set(iw.INPUTS) and sum(profile["charge"]["events"]) == len(events["heldout"]["dt"])


@pytest.mark.fr("012-FR-3")
def test_heldout_rows_never_enter_the_fit(walk_set, tmp_path):
    from imas_skew_helpers import observation_set
    setup, observations, delays, truth, missing, roles, raw = walk_set
    dt = np.where(raw["roles"] == 2, raw["dt"] + 3000.0, raw["dt"])
    changed = observation_set(tmp_path / "obs", raw["ci"], raw["cj"], dt, raw["tof"], walk=raw["walk"],
                              roles=raw["roles"])
    first, *_ = _bar_fit(observations, delays)
    second, widths, *_ = _bar_fit(changed, delays)
    assert first.table() == second.table()
    assert widths["heldout"]["before"]["events"] == int(np.count_nonzero((raw["roles"] == 2) & ~missing))


@pytest.fixture(scope="module")
def solved_job(tmp_path_factory, splits):
    """Spec 011's synthetic job: walk-carrying observation sets (observe-only) and an EM skew LUT without walk."""
    from src.imas.skew import job as ij
    base = tmp_path_factory.mktemp("solved_job")
    job = ij.load_job(job_file(base / "job.json", splits, estimators=["em"], per_run_solves=False))
    ij.run_job(job, base / "work", progress=lambda _: None)
    return base


def _fit_file(path, base, **changes):
    import json
    from src.imas.skew.walk import _sha256
    data = {"schema": iw.FIT_SCHEMA, "job": str(base / "job.json"), "work": str(base / "work"),
            "skew_lut": str(base / "work" / "lut_em.csv"), "skew_lut_sha256": _sha256(base / "work" / "lut_em.csv"),
            "output": str(path.parent / "walk.json"), "min_events": 20, **changes}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.slow
@pytest.mark.fr("012-FR-3")
def test_walk_command_frozen_inputs(tmp_path, solved_job):
    import json
    import shutil
    from src.imas.skew import __main__ as cli
    base = solved_job
    assert cli.main(["walk", str(_fit_file(tmp_path / "fit.json", base))]) == 0
    lut = iw.WalkLut.read(tmp_path / "walk.json")
    manifest = lut.manifest
    job_sha = json.loads((base / "work" / "job_fingerprint.json").read_text(encoding="utf-8"))["sha256"]
    assert manifest["job_sha256"] == job_sha and manifest["fit_kind"] == "bar"
    assert set(manifest["energy_calibration"]) == {"r1", "r2"} and manifest["map_sha256"]
    assert manifest["skew_solved_with_walk_lut_sha256"] is None
    report = json.loads(iw.report_path(tmp_path / "walk.json").read_text(encoding="utf-8"))
    assert report["walk_lut_sha256"] == lut.sha256 and report["widths"]["heldout"]["before"]["events"] > 0
    with pytest.raises(iw.core.WalkError, match="output exists"):
        iw.run_fit(iw.load_fit(_fit_file(tmp_path / "fit.json", base)))
    with pytest.raises(iw.core.WalkError, match="SHA-256"):
        iw.run_fit(iw.load_fit(_fit_file(tmp_path / "a.json", base, skew_lut_sha256="0" * 64,
                                         output=str(tmp_path / "walk_a.json"))))
    other = tmp_path / "other"
    other.mkdir()
    for name in ("lut_em.csv", "lut_em.json"):
        shutil.copy(base / "work" / name, other / name)
    record = json.loads((other / "lut_em.json").read_text(encoding="utf-8"))
    record["provenance"]["job_sha256"] = "f" * 64
    (other / "lut_em.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(iw.core.WalkError, match="not solved by this job"):
        iw.run_fit(iw.load_fit(_fit_file(other / "fit.json", base, skew_lut=str(other / "lut_em.csv"))))
    with pytest.raises(iw.core.WalkError, match="give it exactly when"):
        iw.run_fit(iw.load_fit(_fit_file(tmp_path / "b.json", base, walk_lut=str(tmp_path / "walk.json"),
                                         output=str(tmp_path / "walk_b.json"))))
    with pytest.raises(iw.core.WalkError, match="file keys"):
        iw.load_fit(_fit_file(tmp_path / "c.json", base, extra=1))


# --- T6: correction function and export (FR-6) ----------------------------------------------
def _small_lut(map_sha256, **manifest):
    edges = {"walk2d": [np.array([0.0, 10, 40]), np.array([0.0, 0.6, 1.0])], "doi": [np.array([1.0, 2, 4])],
             "kev": [np.array([400.0, 500, 620])]}
    offsets = {"walk2d": np.array([[40.0, 10], [-10, -40]]), "doi": np.array([6.0, -6]), "kev": np.array([-3.0, 3])}
    events = {b: np.full(v.shape, 300) for b, v in offsets.items()}
    return iw.WalkLut.from_fit(edges, offsets, events, {"map_sha256": map_sha256, **manifest})


@pytest.mark.fr("012-FR-6")
def test_correct_dt_by_hand_and_policy(setup):
    from src.timing.lut import DelayTable
    lut = _small_lut("m")
    skew = DelayTable({10: 100.0, 20: -50.0, 30: 7.0})
    dt = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    side_1 = {"q_top": np.array([5.0, 5.0, 5.0, 5.0]), "q_time": np.array([10.0, 10, 10, 0]),
              "doi_ratio": np.array([1.5, 1.5, 1.5, 1.5]), "kev": np.array([450.0, 450, 450, 450])}
    side_2 = {"q_top": np.array([30.0, 30, 30, 30]), "q_time": np.array([30.0, 30, 30, 30]),
              "doi_ratio": np.array([3.0, 3, 3, 3]), "kev": np.array([550.0, 550, 550, 550])}
    out, status, counts = iw.correct_dt(dt, [10, 10, 99, 10], [20, 30, 20, 20], side_1, side_2, skew, lut)
    w1 = lut.offset_ps["walk2d"][0, 0] + lut.offset_ps["doi"][0] + lut.offset_ps["kev"][0]
    w2 = lut.offset_ps["walk2d"][1, 1] + lut.offset_ps["doi"][1] + lut.offset_ps["kev"][1]
    assert out[0] == pytest.approx(1000 - 100 + -50 - (w1 - w2))
    assert out[1] == pytest.approx(1000 - 100 + 7 - (w1 - w2))
    assert [iw.CORRECTION_STATUS[s] for s in status] == ["corrected", "corrected", "no applied delay",
                                                         "walk input missing"]
    assert out[2] == 1000 and out[3] == 1000 and counts["uncorrected kept"] == 2
    dropped, _, counts = iw.correct_dt(dt, [10, 10, 99, 10], [20, 30, 20, 20], side_1, side_2, skew, lut, policy="drop")
    assert np.isnan(dropped[2:]).all() and counts["uncorrected dropped"] == 2 and counts["walk input missing"] == 1
    skew_only, status, _ = iw.correct_dt(dt, [10, 10, 99, 10], [20, 30, 20, 20], side_1, side_2, skew, None)
    assert skew_only[3] == 1000 - 100 - 50 and status[3] == 0
    with pytest.raises(iw.core.WalkError, match="policy"):
        iw.correct_dt(dt, [10] * 4, [20] * 4, side_1, side_2, skew, lut, policy="mark")


@pytest.mark.fr("012-FR-6")
def test_correct_features_equals_correct_dt(tmp_path, setup):
    from src.timing.lut import DelayTable
    lookup = setup.lookup()
    records = [(side(setup, *A, time={2: 9.0, 3: 3.0}), side(setup, *B, stamp=T0 - 250)),
               (side(setup, *A), side(setup, *B, time={5: 25.0}, stamp=T0 + 400))]
    features = next(sel.feature_batches(write(tmp_path / "f.ldat", records), lookup,
                                        sel.Selection.production(lookup)))
    channels_ = np.concatenate([features.column("time_channel_1"), features.column("time_channel_2")])
    skew = DelayTable({int(c): float(k * 13) for k, c in enumerate(channels_)})
    lut = _small_lut("m")
    out, status, _ = iw.correct_features(features, skew, lut)
    one, _ = iw.side_inputs(features.column, 1)
    two, _ = iw.side_inputs(features.column, 2)
    b1, _ = skew.lookup(features.column("time_channel_1"))
    b2, _ = skew.lookup(features.column("time_channel_2"))
    walk, _, _ = iw.core.walk_difference(lut, one, two)
    assert not status.any()
    np.testing.assert_allclose(out, features.column("dt_ps") - b1 + b2 - walk)


@pytest.mark.fr("012-FR-6")
def test_pairing_and_map_refusals(setup):
    from types import SimpleNamespace
    walk = _small_lut(setup.provenance["map_sha256"], skew_lut_sha256="a" * 64)
    walk.manifest["table"] = {"sha256": "w" * 64}
    plain = SimpleNamespace(manifest={"provenance": {}})
    solved = SimpleNamespace(manifest={"provenance": {"walk_lut_sha256": "w" * 64}})
    iw.check_tables(plain, "a" * 64, walk, setup)                     # fitted with this skew LUT
    iw.check_tables(solved, "b" * 64, walk, setup)                    # skew solved with this walk LUT
    with pytest.raises(iw.core.WalkError, match="not with"):
        iw.check_tables(plain, "b" * 64, walk, setup)                 # walk fitted with another skew
    with pytest.raises(iw.core.WalkError, match="apply a walk LUT"):
        iw.check_tables(solved, "b" * 64, None, setup)                # skew solved with walk, applied without
    other = _small_lut("f" * 64, skew_lut_sha256="a" * 64)
    with pytest.raises(iw.core.WalkError, match="another map"):
        iw.check_tables(plain, "a" * 64, other, setup)


@pytest.mark.fr("012-FR-6")
def test_export_round_trip(tmp_path):
    import csv
    import json
    from src.imas.skew import __main__ as cli
    lut = _small_lut("m", skew_lut_sha256="a" * 64)
    lut.write(tmp_path / "walk.json")
    assert cli.main(["walk-export", str(tmp_path / "walk.json"), "--out", str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    assert iw.WalkLut.read(out / "imas_walk_lut.json").sha256 == lut.sha256
    record = json.loads((out / "export.json").read_text(encoding="utf-8"))
    assert record["walk_lut_sha256"] == lut.sha256 and record["skew_lut_sha256"] == "a" * 64
    with open(out / "imas_walk_curves.tsv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == sum(v.size for v in lut.offset_ps.values())
    for row in rows:
        index = tuple(int(i) for i in row["bin"].split(","))
        assert float(row["offset_ps"]) == lut.offset_ps[row["block"]][index]
        assert float(row["low_1"]) == lut.edges[row["block"]][0][index[0]]
    assert cli.main(["walk-export", str(tmp_path / "walk.json"), "--out", str(out)]) == 1     # never overwritten


# --- T7: CTR with walk (FR-5) -----------------------------------------------------------------
NOISE_PS = 150.0


def _charge_walk(q):
    return 250.0 * np.exp(-(q - 5.0) / 6.0)


@pytest.fixture(scope="module")
def walk_ctr(tmp_path_factory):
    """Opposed ring-2 SMs through a point near the centre: injected delays plus a charge walk, a frozen skew LUT
    (true delays) and a walk LUT tabulating the injected walk (spec 011's CTR fixture with walk)."""
    import json
    from src.imas.skew import job as ij, outputs
    from src.imas.skew.geometry import ImasGeometry
    from src.timing import solver as ts
    from src.timing.cache import _sha256
    from src.timing.shell import C_MM_PER_PS
    base = tmp_path_factory.mktemp("walk_ctr")
    probe = ImasSetup.from_files(CONFIG)
    rng = np.random.default_rng(9)
    ring2 = range(48, 72)
    times = sorted({ch for sm in ring2 for mm in (5, 6) for ch in channels(probe, sm, mm)[0]})
    encal = mu_file(base / "encal.tsv", {ch: 511.0 for ch in times})
    delays = dict(zip(times, rng.normal(0.0, 500.0, len(times))))
    pairs = []
    for _ in range(5000):
        sm = int(rng.choice(ring2[:12]))
        pairs.append(((sm, int(rng.integers(5, 7)), int(rng.integers(8)), float(rng.uniform(5, 35))),
                      (sm + 12, int(rng.integers(5, 7)), int(rng.integers(8)), float(rng.uniform(5, 35)))))

    def records(stamps):
        return [(side(probe, a[0], a[1], time={a[2]: a[3]}, stamp=T0 + s1),
                 side(probe, b[0], b[1], time={b[2]: b[3]}, stamp=T0 + s2)) for (a, b), (s1, s2) in zip(pairs, stamps)]

    point = write(base / "na22_00000040.ldat", records([(0, 0)] * len(pairs)))
    lookup = ImasSetup.from_files(CONFIG, encal).lookup()
    geometry = ImasGeometry.from_setup(probe)
    source = np.array([0.0, 0.0, probe.ring_z[2] + 3.0])
    stamps = [(0, 0)] * len(pairs)
    for f in sel.feature_batches(point, lookup, sel.Selection.production(lookup)):
        _, rows = geometry.endpoints(f)
        tof = (np.linalg.norm(source - rows[:, :3], axis=1) - np.linalg.norm(source - rows[:, 3:], axis=1)) / C_MM_PER_PS
        walk = _charge_walk(f.column("q_top_1")) - _charge_walk(f.column("q_top_2"))
        for k, (record, c1, c2) in enumerate(zip(f.column("record"), f.column("time_channel_1"),
                                                 f.column("time_channel_2"))):
            dt = int(round(tof[k] + delays[int(c1)] - delays[int(c2)] + walk[k] + rng.normal(0.0, NOISE_PS)))
            a = pairs[record][0]
            stamps[record] = (dt, 0) if channels(probe, a[0], a[1])[0][a[2]] == int(c1) else (0, dt)
    write(point, records(stamps))
    splits = [write(base / f"skew_{k}.ldat", []) for k in range(2)]
    job_path = base / "job.json"
    job_path.write_text(json.dumps({"schema": ij.SCHEMA, "config": str(CONFIG), "runs": [{"name": "r1", "splits": [
        {"path": str(splits[0]), "roles": ["encal", "train"]}, {"path": str(splits[1]), "roles": ["heldout"]}]}]}),
        encoding="utf-8")
    work = base / "job"
    work.mkdir()
    stamp = ij.fingerprint(ij.load_job(job_path))
    (work / "job_fingerprint.json").write_text(json.dumps(stamp), encoding="utf-8")

    def skew_lut(path, **extra):
        rows = np.zeros(len(times), ts.CHANNEL_DTYPE)
        rows["time_channel"], rows["delay_ps"] = times, [delays[c] for c in times]
        rows["sigma_ps"] = rows["raw_sigma_ps"] = 5.0
        rows["status"], rows["apply"] = ts.LOW_STATISTICS, True
        provenance = {"job_sha256": stamp["sha256"], "estimator": "em",
                      "runs": {"r1": {"encal_sha256": _sha256(encal)}}, **extra}
        outputs.write_imas_lut(path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), provenance, probe)
        return path

    lut = skew_lut(work / "lut_em.csv")
    edges = {"walk2d": [np.linspace(5.0, 35.0, 61), np.array([0.0, 2.0])], "doi": [np.array([0.0, 100.0])],
             "kev": [np.array([0.0, 2000.0])]}
    centres = (edges["walk2d"][0][:-1] + edges["walk2d"][0][1:]) / 2
    offsets = {"walk2d": _charge_walk(centres)[:, None], "doi": np.zeros(1), "kev": np.zeros(1)}
    events = {b: np.full(v.shape, 500) for b, v in offsets.items()}
    walk_lut = iw.WalkLut.from_fit(edges, offsets, events, {"map_sha256": probe.provenance["map_sha256"],
                                                            "skew_lut_sha256": _sha256(lut), "fit_kind": "synthetic"})
    walk_lut.write(base / "walk.json")
    legacy = base / "skew_legacy.txt"
    legacy.write_text("".join(f"{c} {-delays[c]}\n" for c in times), encoding="utf-8")
    return {"base": base, "point": point, "work": work, "encal": encal, "lut": lut, "walk": base / "walk.json",
            "legacy": legacy, "skew_lut": skew_lut}


def _ctr_file(case, name="ctr.json", **changes):
    import json
    from src.imas.skew import ctr
    from src.timing.cache import _sha256
    data = {"schema": ctr.SCHEMA, "job_work": str(case["work"]), "encal": str(case["encal"]),
            "luts": {"em": {"path": str(case["lut"]), "sha256": _sha256(case["lut"])}},
            "legacy": {"path": str(case["legacy"]), "sha256": _sha256(case["legacy"])},
            "output": str(case["base"] / "out" / name),
            "acquisitions": [{"name": "na22_40", "ldat": str(case["point"]), "source_mm": "lor_convergence",
                              "source_sigma_mm": [1, 1, 1], "declared": "centre"}],
            "walks": {"em+walk": {"skew": "em", "path": str(case["walk"]), "sha256": _sha256(case["walk"])}},
            "walk_acceptance": [["em", "em+walk"]],
            "lor_distance_mm": 40.0, "sensitivity_lor_mm": [30.0, 50.0], "bootstrap": 20, **changes}
    path = case["base"] / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.slow
@pytest.mark.fr("012-FR-5", "012-FR-7")
def test_ctr_walk_table_beats_skew_only(walk_ctr):
    from src.imas.skew import ctr
    report = ctr.run_ctr(ctr.load_ctr(_ctr_file(walk_ctr)), log=lambda *_: None)
    fits = {name: report["results"][name]["pooled"]["fit"] for name in ("legacy", "em", "em+walk")}
    assert len({f["events"] for f in fits.values()}) == 1                   # same events in every table
    width = 2 * np.sqrt(2 * np.log(2))
    assert fits["em+walk"]["fwhm_ps"] == pytest.approx(width * NOISE_PS, rel=0.15)
    assert fits["em+walk"]["model_free_fwhm_ps"] < fits["em"]["model_free_fwhm_ps"]
    judged = report["walk_acceptance"][0]
    assert (judged["reference"], judged["candidate"]) == ("em", "em+walk") and judged["accepted"]
    assert judged["pooled"]["paired"]["significant"] and judged["per_position"]["centre"]["accepted"]
    assert report["results_by_position"]["em+walk"]["centre"]["fit"]["events"] == fits["em+walk"]["events"]
    assert report["counts"]["na22_40"]["stages"]["walk input missing"] == 0
    assert set(report["counts"]["na22_40"]["stages"]["walk clamped"]["em+walk"]) == set(iw.INPUTS)
    assert report["frozen_inputs"]["walks"]["em+walk"]["skew"] == "em"
    assert "every walk input" in report["cuts"]["same_events"]
    # FR-7: the injected charge walk shows in the 22Na profile before (em) and is gone after (em+walk)
    profile = report["profiles"]["22Na em -> em+walk"]["charge"]
    assert report["profiles_by_position"]["22Na em -> em+walk"]["centre"]["charge"] == profile
    before, after = np.array(profile["before_median_ps"], float), np.array(profile["after_median_ps"], float)
    edges = np.array(profile["edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    low, high = centres < 10, centres > 30
    expected = np.mean(_charge_walk(centres[low])) - np.mean(_charge_walk(centres[high]))     # ~165 ps
    assert before[low].mean() - before[high].mean() == pytest.approx(expected, abs=30)
    assert abs(after[low].mean() - after[high].mean()) < 25
    assert sum(profile["events"]) > 0.9 * 2 * fits["em+walk"]["events"]       # both sides, |residual| < 600 ps


@pytest.mark.fr("012-FR-7")
def test_profile_median_from_histogram():
    from src.imas.skew import ctr
    bins = int(2 * ctr.PROFILE_CORE_PS / ctr.PROFILE_BIN_PS)
    hist = np.zeros((3, bins))
    hist[0, bins // 2] = 4                                  # all events in [0, 2) ps: median at 1 ps
    hist[1, [bins // 2 - 1, bins // 2]] = 1                # one event each side of 0 ps: the middle
    assert ctr._median(hist) == [pytest.approx(1.0), pytest.approx(0.0), None]


@pytest.mark.fr("012-FR-5")
def test_ctr_walk_pairing_refused(walk_ctr):
    import json
    from src.imas.skew import ctr
    from src.timing.cache import _sha256
    case = walk_ctr
    other = iw.WalkLut.read(case["walk"])
    other.manifest = {k: v for k, v in other.manifest.items() if k != "table"} | {"skew_lut_sha256": "0" * 64}
    other.write(case["base"] / "walk_other.json")
    walks = {"em+walk": {"skew": "em", "path": str(case["base"] / "walk_other.json"),
                         "sha256": _sha256(case["base"] / "walk_other.json")}}
    with pytest.raises(ctr.CtrError, match="was fitted with skew LUT"):
        ctr.frozen_inputs(ctr.load_ctr(_ctr_file(case, "a.json", walks=walks, walk_acceptance=[])))
    solved = case["skew_lut"](case["work"] / "lut_em_walk.csv", walk_lut_sha256="9" * 64)
    luts = {"em": {"path": str(case["lut"]), "sha256": _sha256(case["lut"])},
            "em2": {"path": str(solved), "sha256": _sha256(solved)}}
    with pytest.raises(ctr.CtrError, match="evaluate it in a walks entry"):
        ctr.frozen_inputs(ctr.load_ctr(_ctr_file(case, "b.json", luts=luts)))
    with pytest.raises(ctr.CtrError, match="walk_acceptance"):
        ctr.load_ctr(_ctr_file(case, "c.json", walk_acceptance=[["em", "nope"]]))
    walk_sha = iw.WalkLut.read(case["walk"]).sha256
    em2 = case["skew_lut"](case["work"] / "lut_em_walk2.csv", walk_lut_sha256=walk_sha)
    luts = {"em": luts["em"], "em2": {"path": str(em2), "sha256": _sha256(em2)}}
    walks = {"em2+walk": {"skew": "em2", "path": str(case["walk"]), "sha256": _sha256(case["walk"])}}
    ctr.frozen_inputs(ctr.load_ctr(_ctr_file(case, "f.json", luts=luts, walks=walks,
                                             walk_acceptance=[["em", "em2+walk"]])))    # solved with this walk LUT
    with pytest.raises(ctr.CtrError, match="not evaluated"):
        ctr.frozen_inputs(ctr.load_ctr(_ctr_file(case, "g.json", luts=luts, walks=walks,
                                                 walk_acceptance=[["em", "em2"]])))
    from src.imas.skew import outputs
    from src.timing.lut import LutError
    probe = ImasSetup.from_files(CONFIG)
    with pytest.raises(LutError, match="solved with walk"):
        outputs.write_native(em2, case["base"] / "native.tsv", probe)
    with pytest.raises(LutError, match="solved with walk"):
        outputs.write_legacy(em2, case["base"] / "legacy.txt", probe)
    with pytest.raises(ctr.CtrError, match="'skew' must name"):
        ctr.load_ctr(_ctr_file(case, "d.json", walks={"w": {"skew": "x", "path": "p", "sha256": "0"}}))
    assert json.loads(_ctr_file(case, "e.json").read_text(encoding="utf-8"))["walks"]
