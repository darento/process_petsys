"""Independent CTR from known-position point sources (spec 008 T9b, FR-13)."""
import hashlib
import json
import math

import numpy as np
import pytest

from src.cornell.skew import ctr
from src.cornell.skew import solver as ts
from src.cornell.skew.geometry import geometric_tof_ps
from src.cornell.skew.ingest import CompiledLookup, GeometryHypothesis, apply_geometry, compiled_batches
from src.cornell.skew.lut import corrected_dt_ps, write_lut
from cornell_timing_helpers import CONFIG, T0, channels, make_setup, side, write

pytestmark = pytest.mark.fr("008-FR-13")
DELAY_SM1, DELAY_SM16 = 300.0, -200.0
SIGMA_PS = 120.0


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(setup, count, dt_of):
    """SM 1 vs SM 16 (mm 5); detector 2 shifted so the record's raw t_1 - t_2 becomes ``dt_of(k)``."""
    return [(side(setup, 1, stamp=T0 + k * 1_000_000), side(setup, 16, stamp=T0 + k * 1_000_000 - dt_of(k)))
            for k in range(count)]


def _features(path, setup):
    lookup = CompiledLookup.from_setup(setup)
    batches = list(compiled_batches(path, lookup, file_index=0, seed=0))
    ints = np.concatenate([b.ints for b in batches])
    floats = np.concatenate([b.floats for b in batches])
    from src.cornell.skew.ingest import FeatureBatch
    return FeatureBatch(0, np.concatenate([b.codes for b in batches]), ints, floats)


def _case(tmp_path, count=400, seed=3):
    """Frozen LUT + its job, and a point-source LDAT whose truth is TOF + delays + Gaussian noise."""
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
              {"job": str(job), "job_sha256": _sha(job), "runs": ["r"],
               "encal_sha256": {"r": _sha(tmp_path / "r.encal")}})

    # endpoints do not depend on timestamps: a first pass gives each record's endpoints and raw dt offset
    probe = write(tmp_path / "probe.ldat", _records(setup, count, lambda k: 0))
    features = _features(probe, setup)
    assert len(features.ints) == count
    geometry = apply_geometry(features, GeometryHypothesis.selected(CONFIG, setup))
    p, q = geometry.endpoints_mm[:, :3], geometry.endpoints_mm[:, 3:]
    source = p[0] + 0.3 * (q[0] - p[0])  # on the LOR, off centre: |TOF| of several hundred ps
    tof = ctr.geometric_tof_ps(source, p, q)
    base_dt = features.column("dt_ps")
    delay = {c: DELAY_SM1 for c in t1} | {c: DELAY_SM16 for c in t2}
    b1 = np.array([delay[c] for c in features.column("time_channel_1")])
    b2 = np.array([delay[c] for c in features.column("time_channel_2")])
    noise = np.random.default_rng(seed).normal(0, SIGMA_PS, count)
    target = np.rint(tof + b1 - b2 + noise).astype(np.int64)
    ldat = write(tmp_path / "point.ldat", _records(setup, count, lambda k: int(target[k] - base_dt[k])))
    return {"setup": setup, "lut": lut, "job": job, "ldat": ldat, "source": source, "tof": tof, "p": p, "q": q}


def _spec(tmp_path, case, **overrides):
    acquisition = {"name": "point", "ldat": str(case["ldat"]), "source_mm": [float(v) for v in case["source"]],
                   "source_sigma_mm": [1.0, 1.0, 1.0], "input_correction": "uncorrected"}
    acquisition |= overrides.pop("acquisition", {})
    spec = {"schema": ctr.SCHEMA, "config": str(CONFIG), "encal": str(tmp_path / "r.encal"), "lut": str(case["lut"]),
            "lut_sha256": _sha(case["lut"]), "output": str(tmp_path / "out" / "ctr.json"),
            "acquisitions": [acquisition], "bootstrap": 20} | overrides
    path = tmp_path / "ctr.json"
    path.write_text(json.dumps(spec))
    return path


def test_frozen_holdout_and_geometric_tof(tmp_path):
    case = _case(tmp_path)
    # sign conventions: scalar geometry and LUT contracts agree with the vectorized residual
    p, q, source = case["p"][0], case["q"][0], case["source"]
    assert ctr.geometric_tof_ps(source, p[None], q[None])[0] == pytest.approx(geometric_tof_ps(source, p, q))
    raw = 10_000 + int(round(geometric_tof_ps(source, p, q) + DELAY_SM1 - DELAY_SM16))
    corrected = corrected_dt_ps(raw, 10_000, DELAY_SM1, DELAY_SM16)
    assert ctr.residual_ps([corrected], source, p[None], q[None])[0] == pytest.approx(0, abs=0.5)
    assert abs(case["tof"]).min() > 300

    report = ctr.run_ctr(ctr.load_ctr(_spec(tmp_path, case)), log=lambda *_: None)
    stages = report["counts"]["point"]["stages"]
    assert stages["records"] == stages["passed pre-geometry"] == stages["in window"] == 400
    fit = report["results"]["pooled"]["fit"]
    assert fit["status"] == "fitted"
    # TOF and delays removed: zero mean, the injected width; without TOF subtraction the mean would be ~TOF
    assert abs(fit["mean_ps"]) < 4 * SIGMA_PS / math.sqrt(400)
    assert fit["sigma_ps"] == pytest.approx(SIGMA_PS, rel=0.15)
    assert report["coverage"]["applied_channels_seen"] >= 2 and report["coverage"]["applied_channel_pairs_seen"] >= 1
    assert report["frozen_inputs"]["lut_sha256"] == _sha(case["lut"])
    assert report["acquisitions"][0]["sha256"] == _sha(case["ldat"])
    assert report["results"]["pooled"]["uncertainty"]["fwhm_ps"]["statistical"] > 0
    out = tmp_path / "out"
    assert (out / "ctr_histograms.csv").is_file() and (out / "ctr.png").is_file()

    # frozen: another table, a LUT job input, another .encal, a changed job, native correction -> refused
    with pytest.raises(ctr.CtrError, match="not the frozen"):
        ctr.run_ctr(ctr.load_ctr(_spec(tmp_path, case, lut_sha256="0" * 64)))
    inside = tmp_path / "norm" / "norm_point.ldat"
    inside.write_bytes(case["ldat"].read_bytes())
    with pytest.raises(ctr.CtrError, match="input of the LUT job"):
        ctr.run_ctr(ctr.load_ctr(_spec(tmp_path, case, acquisition={"ldat": str(inside)})))
    other = tmp_path / "other.encal"
    other.write_text((tmp_path / "r.encal").read_text() + "\n")
    with pytest.raises(ctr.CtrError, match="encal"):
        ctr.run_ctr(ctr.load_ctr(_spec(tmp_path, case, encal=str(other))))
    with pytest.raises(ctr.CtrError, match="corrected twice"):
        ctr.load_ctr(_spec(tmp_path, case, acquisition={"input_correction": "native"}))
    case["job"].write_text(case["job"].read_text() + " ")
    with pytest.raises(ctr.CtrError, match="missing or changed"):
        ctr.run_ctr(ctr.load_ctr(_spec(tmp_path, case)))

    # timing-blind localization: the point closest to noisy LORs through a known source, outliers rejected
    rng = np.random.default_rng(2)
    truth = np.array([-58.0, 19.0, -25.0])
    u = rng.normal(size=(2_000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    p, q = truth - 160 * u + rng.normal(0, 1.5, (2_000, 3)), truth + 160 * u + rng.normal(0, 1.5, (2_000, 3))
    q[:200] += rng.uniform(-80, 80, (200, 3))  # scatter-like LORs that miss the source
    point, miss, used = ctr.converge_point(p, q)
    assert np.linalg.norm(point - truth) < 0.3 and miss < 2.0 and 1_000 < used < 1_900
    with pytest.raises(ctr.CtrError, match="parallel"):
        ctr.converge_point(np.zeros((5, 3)), np.tile([1.0, 0.0, 0.0], (5, 1)))
    located = ctr.load_ctr(_spec(tmp_path, case, acquisition={"source_mm": "lor_convergence", "declared": "X0Y-6Z0 cm"}))
    assert located.acquisitions[0].source_mm is None and located.acquisitions[0].declared == "X0Y-6Z0 cm"


def _spec_object(tmp_path, names, bootstrap, window=1500.0):
    return ctr.CtrSpec(tmp_path / "c.json", "", CONFIG, tmp_path / "r.encal", tmp_path / "l.csv", "", tmp_path / "o.json",
                       tuple(ctr.Acquisition(n, tmp_path / f"{n}.ldat", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), None)
                             for n in names), 15.0, (10.0, 25.0), window, 10.0, bootstrap, 0, (3.0, 11.0), None)


class _NoDelays:
    channels = np.zeros(0, np.int64)


def test_pooled_ctr_and_uncertainty(tmp_path):
    rng = np.random.default_rng(11)
    spec = _spec_object(tmp_path, ("a", "b"), bootstrap=50)
    edges = spec.edges_ps
    bins = len(edges) - 1

    def sample(n_signal, sigma, n_background, mean=0.0):
        values = np.concatenate((rng.normal(mean, sigma, n_signal), rng.uniform(edges[0], edges[-1], n_background)))
        return values, np.histogram(values, edges)[0].astype(float)

    # fit: Gaussian width, background fraction, model-free width, tails
    _, hist = sample(20_000, 150.0, 4_000)
    fit = ctr.fit_histogram(hist, edges)
    assert fit["status"] == "fitted" and fit["fwhm_ps"] == pytest.approx(ctr.FWHM_PER_SIGMA * 150, rel=0.04)
    assert fit["background_fraction"] == pytest.approx(4_000 / 24_000, abs=0.02)
    assert fit["model_free_fwhm_ps"] == pytest.approx(ctr.FWHM_PER_SIGMA * 150, rel=0.08)
    assert fit["fwtm_to_fwhm"] == pytest.approx(ctr.GAUSSIAN_FWTM_TO_FWHM, rel=0.06)
    wide = np.linspace(-3000.0, 3000.0, 601)  # the default window: tails stand above the background
    flat = np.concatenate((rng.normal(0, 150, 100_000), rng.uniform(-3000, 3000, 20_000)))
    laplace = np.concatenate((rng.normal(0, 150, 100_000), rng.laplace(0, 300, 20_000)))
    ratios = [ctr.fit_histogram(np.histogram(v, wide)[0], wide)["fwtm_to_fwhm"] for v in (flat, laplace)]
    assert ratios[0] == pytest.approx(ctr.GAUSSIAN_FWTM_TO_FWHM, rel=0.03)
    assert ratios[1] > 1.03 * ctr.GAUSSIAN_FWTM_TO_FWHM
    assert ctr.fit_histogram(hist[:1] * 0 + 5, edges[:2])["status"] == "insufficient events"

    # pooled = event-weighted sum; bootstrap spread matches independent repetitions; systematics in quadrature
    accumulator = ctr.Accumulator(spec, _NoDelays())
    names = [v.name for v in accumulator.variants]
    for a, n in enumerate((6_000, 3_000)):
        values, hist = sample(n, 150.0, n // 5)
        accumulator.hist[:, a] = hist
        b = np.floor((values + spec.window_ps) / spec.bin_ps).astype(int)
        for r in range(spec.bootstrap):
            accumulator.boot[r, a] = np.bincount(b, weights=rng.poisson(1.0, len(b)), minlength=bins)
        accumulator.hist[names.index("source_x+"), a] = sample(n, 160.0, n // 5)[1]
        accumulator.hist[names.index("depth_11mm"), a] = sample(n, 140.0, n // 5)[1]
    results = ctr.summarize(accumulator)
    pooled, alone = results["pooled"], results["a"]
    assert pooled["fit"]["events"] == results["a"]["fit"]["events"] + results["b"]["fit"]["events"]
    assert alone["fit"]["events"] == accumulator.hist[0, 0].sum()
    repeats = [ctr.fit_histogram(sample(9_000, 150.0, 1_800)[1], edges)["fwhm_ps"] for _ in range(30)]
    statistical = pooled["uncertainty"]["fwhm_ps"]["statistical"]
    assert 0.6 < statistical / np.std(repeats, ddof=1) < 1.6
    components = pooled["uncertainty"]["fwhm_ps"]["systematic_components"]
    assert components["source x"] > components["source y"] and components["depth"] > 0
    total = pooled["uncertainty"]["fwhm_ps"]
    assert total["total"] == pytest.approx(math.hypot(statistical, total["systematic"]))
    assert set(pooled["lor_sensitivity_fwhm_ps"]) == {"lor_10mm", "lor_25mm"}
    assert ctr.combine_uncertainty(100.0, {"x": [103.0, 96.0], "d": [101.0]}, 2.0) == {
        "statistical": 2.0, "systematic_components": {"x": 4.0, "d": 1.0}, "systematic": math.sqrt(17.0),
        "total": math.sqrt(21.0)}

    # residual medians against geometric TOF: a units error would show a slope
    tof_hist = np.zeros((len(ctr.TOF_EDGES_PS) - 1, bins))
    centers = (ctr.TOF_EDGES_PS[:-1] + ctr.TOF_EDGES_PS[1:]) / 2
    for k, t in enumerate(centers):
        tof_hist[k] = np.histogram(rng.normal(0.1 * t, 150, 2_000), edges)[0]
    trend = ctr.tof_dependence(tof_hist, edges, 150.0)
    assert trend["slope"] == pytest.approx(0.1, abs=5 * trend["slope_sigma"])


def test_shell_width_is_model_dependent(tmp_path):
    # an extended (shell) source is not a CTR input
    case_spec = {"schema": ctr.SCHEMA, "config": str(CONFIG), "encal": "r.encal", "lut": "l.csv", "lut_sha256": "0",
                 "output": "o.json", "acquisitions": [{"name": "norm", "ldat": "n.ldat", "input_correction": "uncorrected",
                                                       "source": {"inner_radius_mm": 137.5, "outer_radius_mm": 142.5}}]}
    path = tmp_path / "ctr.json"
    path.write_text(json.dumps(case_spec))
    with pytest.raises(ctr.CtrError, match="model dependent and is not CTR"):
        ctr.load_ctr(path)

    # the same events: residual width against an assumed source depends on that assumption; only the
    # known emission point gives the timing resolution
    rng = np.random.default_rng(5)
    n = 20_000

    def shell_events(radius):
        phi, direction = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, np.pi, n)
        emission = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.zeros(n)))
        u = np.column_stack((np.cos(direction), np.sin(direction), np.zeros(n)))
        b = np.einsum("ij,ij->i", emission, u)
        reach = -b + np.sqrt(b * b - (np.einsum("ij,ij->i", emission, emission) - 330.0 ** 2))
        back = -b - np.sqrt(b * b - (np.einsum("ij,ij->i", emission, emission) - 330.0 ** 2))
        p, q = emission + reach[:, None] * u, emission + back[:, None] * u
        dt = ctr.geometric_tof_ps(emission, p, q) + rng.normal(0, 150.0, n)
        return emission, p, q, dt

    widths = {}
    for radius in (100.0, 140.0):
        emission, p, q, dt = shell_events(radius)
        assert np.std(ctr.residual_ps(dt, emission, p, q)) == pytest.approx(150.0, rel=0.03)
        widths[radius] = np.std(ctr.residual_ps(dt, (0.0, 0.0, 0.0), p, q))
    assert widths[100.0] > 2 * 150.0 and widths[140.0] > widths[100.0] * 1.2
