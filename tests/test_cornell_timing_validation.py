"""Frozen-LUT validation: training size, sufficiency, test files once (spec 008 T9, FR-16)."""
import json

import numpy as np
import pytest

from src.cornell.skew import cache as tc
from src.cornell.skew import job as tj
from src.cornell.skew import pairs as tp
from src.cornell.skew import solver as ts
from src.cornell.skew import validation as tv
from cornell_timing_helpers import job_file, runs  # noqa: F401  (synthetic two-run job fixture)

pytestmark = pytest.mark.fr("008-FR-16")


def solution(channels, delays, sigma):
    rows = np.zeros(len(channels), ts.CHANNEL_DTYPE)
    rows["time_channel"], rows["delay_ps"], rows["sigma_ps"], rows["raw_sigma_ps"] = channels, delays, sigma, sigma
    rows["status"], rows["apply"] = ts.CALIBRATED, True
    return ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {})


class FakeSet:
    partitions = 2

    def __init__(self, rows):
        self.rows = rows

    def partition(self, p):
        return self.rows[self.rows["record"] % 2 == p]


def test_nested_fractions_and_stability():
    rows = np.zeros(200_000, tc.OBSERVATION_DTYPE)
    rows["file"], rows["record"] = np.arange(len(rows)) % 7, np.arange(len(rows))
    rows["role"] = np.where(np.arange(len(rows)) % 10 == 0, tc.ROLES.index("test"), tc.ROLES.index("train"))
    u = tv.event_uniform(rows, 0)
    assert np.array_equal(u, tv.event_uniform(rows, 0)) and (0 <= u).all() and (u < 1).all()
    assert abs((u < 0.25).mean() - 0.25) < 0.005
    other = tv.event_uniform(rows, 1)
    assert abs(np.corrcoef(u, other)[0, 1]) < 0.01  # another run thins independently
    # nested: every event kept at 1/8 is kept at 1/4; test rows never enter
    sets = [FakeSet(rows), FakeSet(rows.copy())]
    kept = {}
    for fraction in (0.125, 0.25, 1.0):
        view = tv.Thinned(sets, fraction)
        part = np.concatenate([view.partition(p) for p in range(view.partitions)])
        assert (part["role"] == tc.ROLES.index("train")).all()
        kept[fraction] = set(zip(part["file"].tolist(), part["record"].tolist()))
    assert kept[0.125] <= kept[0.25] <= kept[1.0] and len(kept[1.0]) == 180_000
    # nested comparison: gauge removed, z uses sigma_f^2 - sigma_1^2, a jump is unstable
    rng = np.random.default_rng(16)
    channels = np.arange(3000)
    truth = rng.uniform(-3000, 3000, len(channels))
    full = solution(channels, truth + rng.normal(0, 4, len(channels)), 4.0)
    half_delays = full.channels["delay_ps"] + 100.0 + rng.normal(0, np.sqrt(8.0 ** 2 - 4.0 ** 2), len(channels))
    half_delays[17] += 300.0
    half = solution(channels, half_delays, 8.0)
    summary, z = tv.nested_comparison(half, full)
    assert summary["channels"] == 3000 and abs(summary["gauge_offset_ps"] - 100) < 1
    assert abs(summary["z_rms"] - 1) < 0.25 and abs(summary["predicted_rms_ps"] - np.sqrt(48)) < 1e-9
    assert abs(z[17]) > 20 and summary["abs_z_gt_3"] < 20
    half.channels["apply"][5] = False
    assert 5 not in tv.nested_comparison(half, full)[1]


def test_sufficiency_uses_precision_stability_and_outliers():
    rows = np.zeros(8, ts.CHANNEL_DTYPE)
    rows["time_channel"] = np.arange(10, 18)
    rows["status"] = [ts.CALIBRATED] * 5 + [ts.LOW_STATISTICS, ts.CHANNEL_STATUSES.index("unobserved"),
                                            ts.CHANNEL_STATUSES.index("hardware unavailable")]
    rows["apply"] = [True] * 6 + [False, False]
    half_z = {10: 0.5, 11: -4.0, 13: 1.0, 14: 0.2, 15: 0.1}  # 12 not solved at half data
    classes = tv.classify(rows, half_z, cross_run={13}, pattern={14})
    assert 17 not in classes and len(classes) == 7  # hardware unavailable excluded
    assert classes[10] == (True, [])
    assert classes[11] == (False, ["unstable at half data"])
    assert classes[12] == (False, ["not solved at half data"])
    assert classes[13] == (False, ["cross-run outlier"]) and classes[14] == (False, ["ASIC pattern outlier"])
    assert classes[15] == (False, ["not calibrated"])
    assert classes[16] == (False, ["not calibrated", "not solved at half data"])
    assert tv.dataset_class(classes) == "sufficient for a named subset"
    assert tv.dataset_class({1: (True, [])}) == "sufficient"
    assert tv.dataset_class({1: (False, ["not calibrated"])}) == "insufficient"
    assert tv.classify(rows, half_z, set(), set(), stability_z=5.0)[11] == (True, [])


def validation_file(path, **overrides):
    spec = {"schema": tv.SCHEMA, "job": "job.json", "lut": "out/lut.csv", "output": "validation.json",
            "evaluate_test": True, "prior_test_looks": ["synthetic earlier look"]} | overrides
    path.write_text(json.dumps(spec))
    return path


def test_frozen_lut_binding_and_test_evaluated_once(runs, tmp_path, monkeypatch):  # noqa: F811
    base, items = runs
    job = tj.load_job(job_file(tmp_path / "job.json", base, items))
    assert tj.run_job(job, log=lambda message: None)["status"] == "succeeded"

    def no_reads(*args, **kwargs):
        raise AssertionError("LDAT read during validation")
    monkeypatch.setattr(tj.td, "scan_file", no_reads)
    monkeypatch.setattr(tc.ingest, "LdatStream", no_reads)
    marker_seen, fit = [], tp.fit_observation_set

    def watched(observations, model, *args, roles=(0,), **kwargs):
        marker_seen.append((roles, any(job.work.glob("test-evaluated-*.json"))))
        return fit(observations, model, *args, roles=roles, **kwargs)
    monkeypatch.setattr(tv.tp, "fit_observation_set", watched)

    report = tv.run_validation(tv.load_validation(validation_file(tmp_path / "v.json")), log=lambda message: None)
    assert [e["fraction"] for e in report["training_size"]] == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    events = [e["train_events"] for e in report["training_size"]]
    assert events == sorted(events, reverse=True) and events[-1] < events[0]
    assert report["training_size"][0]["vs_full"]["diff_rms_ps"] == 0.0
    assert report["prior_test_looks"] == ["synthetic earlier look"] and "not run" in report["bootstrap"]
    sufficiency = report["sufficiency"]
    assert sufficiency["operational_channels"] + sufficiency["hardware_unavailable"] == report_rows(tmp_path)
    assert sufficiency["dataset"] in ("sufficient", "sufficient for a named subset", "insufficient")
    lines = (tmp_path / "validation_channels.csv").read_text().splitlines()
    assert lines[0] == "time_channel,lut_status,sufficient,half_data_z,reasons"
    assert len(lines) == 1 + sufficiency["operational_channels"]
    # test files: both runs evaluated after the marker; train fits never see test rows
    assert set(report["test"]["runs"]) == {"ra", "rb"} and all(r["test_events"] > 0 for r in report["test"]["runs"].values())
    assert [seen for roles, seen in marker_seen if roles == (tc.ROLES.index("test"),)] == [True, True]
    assert all(not seen for roles, seen in marker_seen if roles == (tc.ROLES.index("train"),))
    assert {roles for roles, _ in marker_seen} == {(0,), (2,)}
    # opened once: a second evaluation of the same LUT is refused; without the test part it runs
    with pytest.raises(tv.ValidationError, match="already evaluated"):
        tv.run_validation(tv.load_validation(validation_file(tmp_path / "v2.json", output="r2.json")))
    again = tv.run_validation(tv.load_validation(validation_file(tmp_path / "v3.json", output="r3.json",
                                                                 evaluate_test=False)), log=lambda message: None)
    assert again["test"] == {"status": "not evaluated in this validation run"}
    assert again["sufficiency"] == dict(sufficiency, channels_csv=str(tmp_path / "r3_channels.csv"))
    # outputs are never overwritten; a LUT from another job file is refused
    with pytest.raises(tv.ValidationError, match="output exists"):
        tv.run_validation(tv.load_validation(validation_file(tmp_path / "v4.json", output="r3.json", evaluate_test=False)))
    job_file(tmp_path / "other.json", base, items, seed=1)
    with pytest.raises(tv.ValidationError, match="job SHA-256"):
        tv.run_validation(tv.load_validation(validation_file(tmp_path / "v5.json", job="other.json", output="r5.json",
                                                             evaluate_test=False)))
    for bad, message in (({"fractions": [0.25, 1.0]}, "0.5"), ({"evaluate_test": "yes"}, "evaluate_test"),
                         ({"surprise": 1}, "unknown")):
        with pytest.raises(tv.ValidationError, match=message):
            tv.load_validation(validation_file(tmp_path / "bad.json", **bad))


def report_rows(folder):
    return json.loads((folder / "out" / "lut.json").read_text())["table"]["rows"]
