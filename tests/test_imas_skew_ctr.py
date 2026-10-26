"""IMAS point-source CTR on a synthetic 22Na-like source (spec 011 T15, FR-10)."""
import json
import math

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, T0, channels, mu_file, side, write
from src.imas.skew import ctr, job as ij, outputs, select as sel
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing import solver as ts
from src.timing.cache import _sha256
from src.timing.shell import C_MM_PER_PS

pytestmark = pytest.mark.fr("011-FR-10")

NOISE_PS, LEGACY_ERROR_PS = 150.0, 100.0
RING2 = range(48, 72)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """Opposed ring-2 SMs through the centre: injected delays, a frozen LUT, a worse legacy file, a job record."""
    base = tmp_path_factory.mktemp("ctr")
    setup = ImasSetup.from_files(CONFIG)
    rng = np.random.default_rng(5)
    times = sorted({ch for sm in RING2 for mm in (5, 6) for ch in channels(setup, sm, mm)[0]})
    encal = mu_file(base / "encal.tsv", {ch: 511.0 for ch in times})
    delays = dict(zip(times, rng.normal(0.0, 500.0, len(times))))
    pairs = []
    for _ in range(4000):
        sm = int(rng.choice(RING2[:12]))
        a = (sm, int(rng.integers(5, 7)), int(rng.integers(8)))
        b = (sm + 12, int(rng.integers(5, 7)), int(rng.integers(8)))
        pairs.append((a, b))

    def records(stamps):
        return [(side(setup, a[0], a[1], time={a[2]: 9.0}, stamp=T0 + s1),
                 side(setup, b[0], b[1], time={b[2]: 9.0}, stamp=T0 + s2)) for (a, b), (s1, s2) in zip(pairs, stamps)]

    # pass 1: the selection's sides and endpoints (independent of the timestamps)
    point = write(base / "na22_00000035.ldat", records([(0, 0)] * len(pairs)))
    lookup = ImasSetup.from_files(CONFIG, encal).lookup()
    geometry = ImasGeometry.from_setup(setup)
    features = list(sel.feature_batches(point, lookup, sel.Selection.production(lookup)))
    assert sum(len(f.ints) for f in features) == len(pairs)
    source = np.array([0.0, 0.0, setup.ring_z[2] + 3.0])
    stamps = [(0, 0)] * len(pairs)
    for f in features:
        _, rows = geometry.endpoints(f)
        tof = (np.linalg.norm(source - rows[:, :3], axis=1) - np.linalg.norm(source - rows[:, 3:], axis=1)) / C_MM_PER_PS
        for k, (record, c1, c2) in enumerate(zip(f.column("record"), f.column("time_channel_1"),
                                                 f.column("time_channel_2"))):
            dt = int(round(tof[k] + delays[int(c1)] - delays[int(c2)] + rng.normal(0.0, NOISE_PS)))
            first = channels(setup, pairs[record][0][0], pairs[record][0][1])[0][pairs[record][0][2]]
            stamps[record] = (dt, 0) if first == int(c1) else (0, dt)
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
    rows = np.zeros(len(times), ts.CHANNEL_DTYPE)
    rows["time_channel"], rows["delay_ps"] = times, [delays[c] for c in times]
    rows["sigma_ps"] = rows["raw_sigma_ps"] = 5.0
    rows["status"], rows["apply"] = ts.LOW_STATISTICS, True
    provenance = {"job_sha256": stamp["sha256"], "estimator": "em", "runs": {"r1": {"encal_sha256": _sha256(encal)}}}
    lut_path = work / "lut_em.csv"
    outputs.write_imas_lut(lut_path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), provenance, setup)
    legacy = base / "skew_legacy.txt"
    legacy.write_text("".join(f"{c} {-(delays[c] + rng.normal(0.0, LEGACY_ERROR_PS))}\n" for c in times),
                      encoding="utf-8")
    return {"base": base, "point": point, "splits": splits, "work": work, "encal": encal, "lut": lut_path,
            "legacy": legacy, "source": source}


def ctr_file(case, name="ctr.json", **changes):
    data = {"schema": ctr.SCHEMA, "job_work": str(case["work"]), "encal": str(case["encal"]),
            "luts": {"em": {"path": str(case["lut"]), "sha256": _sha256(case["lut"])}},
            "legacy": {"path": str(case["legacy"]), "sha256": _sha256(case["legacy"])},
            "output": str(case["base"] / "out" / "ctr.json"),
            "acquisitions": [{"name": "na22", "ldat": str(case["point"]), "source_mm": "lor_convergence",
                              "source_sigma_mm": [1, 1, 1]}],
            "lor_distance_mm": 40.0, "sensitivity_lor_mm": [30.0, 50.0], "bootstrap": 20, **changes}
    path = case["base"] / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.slow
def test_new_lut_beats_legacy_beats_uncorrected(case):
    report = ctr.run_ctr(ctr.load_ctr(ctr_file(case)), log=lambda *_: None)
    located = report["acquisitions"][0]["source"]["position_mm"]
    assert np.allclose(located[:2], case["source"][:2], atol=3.0)
    fits = {name: report["results"][name]["pooled"]["fit"] for name in (ctr.UNCORRECTED, ctr.LEGACY, "em")}
    events = {f["events"] for f in fits.values()}
    assert len(events) == 1                                  # same events in every table
    width = 2 * math.sqrt(2 * math.log(2))
    assert fits["em"]["fwhm_ps"] == pytest.approx(width * NOISE_PS, rel=0.12)
    assert fits[ctr.LEGACY]["fwhm_ps"] == pytest.approx(width * math.hypot(NOISE_PS, math.sqrt(2) * LEGACY_ERROR_PS),
                                                        rel=0.12)
    assert fits["em"]["fwhm_ps"] < fits[ctr.LEGACY]["fwhm_ps"] < fits[ctr.UNCORRECTED]["fwhm_ps"]
    accepted = report["acceptance"]["em"]
    assert accepted["accepted"] and accepted["judged_on"] == "model_free_fwhm_ps"
    assert accepted["model_free_fwhm_ps"]["accepted"] and accepted["model_free_fwhm_ps"]["paired"]["significant"]
    paired = accepted["fwhm_ps"]["paired"]                   # shared events: common fluctuations cancel
    assert paired["significant"] and paired["statistical"] < accepted["fwhm_ps"]["combined_uncertainty_ps"]
    out = case["base"] / "out"
    assert (out / "ctr.png").is_file() and (out / "ctr_histograms.csv").is_file()


def test_frozen_inputs_refused(case, tmp_path):
    with pytest.raises(ctr.CtrError, match="not the frozen"):
        ctr.frozen_inputs(ctr.load_ctr(ctr_file(case, luts={"em": {"path": str(case["lut"]), "sha256": "0" * 64}})))
    inside = [{"name": "skew", "ldat": str(case["splits"][1]), "source_mm": [0, 0, 0], "source_sigma_mm": [1, 1, 1]}]
    with pytest.raises(ctr.CtrError, match="input of the LUT job"):
        ctr.frozen_inputs(ctr.load_ctr(ctr_file(case, acquisitions=inside)))
    other = tmp_path / "job"
    other.mkdir()
    (other / "job_fingerprint.json").write_text(json.dumps({"sha256": "1" * 64, "declared": json.loads(
        (case["work"] / "job_fingerprint.json").read_text(encoding="utf-8"))["declared"]}), encoding="utf-8")
    with pytest.raises(ctr.CtrError, match="another job"):
        ctr.frozen_inputs(ctr.load_ctr(ctr_file(case, job_work=str(other))))
    with pytest.raises(ctr.CtrError, match="reserved"):
        ctr.load_ctr(ctr_file(case, luts={"legacy": {"path": str(case["lut"]), "sha256": "0" * 64}}))
    unrelated = tmp_path / "encal.tsv"
    unrelated.write_text("ID\tmu\n", encoding="utf-8")
    with pytest.raises(ctr.CtrError, match="no encal.json"):
        ctr.frozen_inputs(ctr.load_ctr(ctr_file(case, encal=str(unrelated))))
