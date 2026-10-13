"""Pooled multi-run Cornell timing job (spec 008 T7a, FR-18).

Synthetic runs are small compact LDAT files with Manager-style ``run.json``
ledgers. Accepted coincidences come from diametric SM 1/SM 16 sides whose
``dt`` follows the two shell branches of the selected geometry with a known
channel-pair shift, so every stage from inventory to LUT runs for real.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cornell_timing_helpers import ROOT, T0, make_setup, shifted, side, write
from src.cornell import timing_cache as tc
from src.cornell import timing_job as tj
from src.cornell import timing_pairs as tp
from src.cornell import timing_solver as ts
from src.cornell.timing_petsys import CanonicalLut

pytestmark = pytest.mark.fr("008-FR-18")
FILES = 3
TOF_PS = 934.0  # diametric branch TOF for the 140 mm mid-radius shell


def coincidence_records(setup, rng, count, shift):
    """SM 1 vs SM 16, two minimodules: dt = shift +- branch TOF + kernel noise."""
    out = []
    for k in range(count):
        mm = (5, 6)[k % 2]
        dt = int(round(shift + rng.choice((-TOF_PS, TOF_PS)) + rng.normal(0, 150)))
        stamp = T0 + k * 1_000_000
        out.append((side(setup, 1, mm=mm, stamp=stamp), side(setup, 16, mm=mm, stamp=stamp - dt)))
    return out


def make_run(base, setup, name, *, seed, accepted=70, shift=600.0):
    """FILES chronological splits plus a succeeded conversion ledger."""
    directory = base / name
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    prefix = f"norm_test_{name}_coincCompact_"
    artifacts = []
    for index in range(FILES):
        records = coincidence_records(setup, rng, accepted, shift)
        offset = index * 10 ** 12  # one second per split, chronological
        records = [(shifted(a, offset), shifted(b, offset)) for a, b in records]
        records.sort(key=lambda r: min(h[0] for h in r[0] + r[1]))
        path = write(directory / f"{prefix}{index:08d}.ldat", records)
        artifacts.append({"kind": "ldat", "cleanup": "retained", "path": str(path), "size_bytes": path.stat().st_size,
                          "input_descriptor": {"format": "compact", "population": "coincidence"}})
    ledger = {"status": "succeeded", "attempts": [{"stage_id": "conversion", "status": "succeeded", "exit_code": 0,
                                                  "artifacts": artifacts}]}
    (directory / "run.json").write_text(json.dumps(ledger))
    return {"name": name, "directory": str(directory), "ledger": str(directory / "run.json"), "prefix": prefix,
            "energy_calibration_prefix": "none" if name != "ra" else {"1": 5, "2": 0}}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("job")
    setup = make_setup(base)
    return base, [make_run(base, setup, "ra", seed=1), make_run(base, setup, "rb", seed=2)]


def job_file(path, base, run_items, **overrides):
    spec = {"schema": tj.JOB_SCHEMA, "config": "configs/cornell_full_system.yaml", "encal": str(base / "r.encal"),
            "reference_run": run_items[0]["name"], "down_regions": [[14, 0]], "runs": run_items,
            "roles": {"validation": [1], "test": [2]}, "geometry": "selected", "work_directory": "work",
            "lut": "out/lut.csv", "workers": 1} | overrides
    path.write_text(json.dumps(spec))
    return path


def test_job_file_is_explicit_and_validated(tmp_path, runs):
    base, items = runs
    job = tj.load_job(job_file(tmp_path / "job.json", base, items))
    assert job.config == ROOT / "configs/cornell_full_system.yaml"
    assert job.work == tmp_path / "work" and job.lut == tmp_path / "out/lut.csv"
    assert [r.name for r in job.runs] == ["ra", "rb"] and job.validation == (1,) and job.test == (2,)
    assert job.energy_peak_tolerance == 0.02 and job.partitions == 16
    bad = [
        ({"schema": "other"}, "schema"), ({"surprise": 1}, "unknown"), ({"geometry": "planar"}, "selected"),
        ({"runs": [items[0], items[0]]}, "unique"), ({"reference_run": "r9"}, "reference_run"),
        ({"roles": {"validation": [1]}}, "roles"), ({"energy_peak_tolerance": 0}, "tolerance"),
        ({"down_regions": [[14]]}, "down regions"), ({"runs": [{"name": "x", "directory": "d"}]}, "run entries"),
        ({"runs": [dict(items[0], energy_calibration_prefix={"1": -1})]}, "energy_calibration_prefix"),
        ({"reproducibility_floor": True}, "per_run_solves"),
    ]
    for override, message in bad:
        with pytest.raises(tj.JobError, match=message):
            tj.load_job(job_file(tmp_path / "bad.json", base, items, **override))
    spec = json.loads((tmp_path / "job.json").read_text())
    del spec["lut"]
    (tmp_path / "bad.json").write_text(json.dumps(spec))
    with pytest.raises(tj.JobError, match="missing"):
        tj.load_job(tmp_path / "bad.json")


@pytest.fixture(scope="module")
def finished(runs, tmp_path_factory):
    base, items = runs
    folder = tmp_path_factory.mktemp("finished")
    job = tj.load_job(job_file(folder / "job.json", base, items))
    return job, tj.run_job(job, log=lambda message: None)


def test_runs_roles_and_resume(runs, finished, monkeypatch):
    base, items = runs
    job, report = finished
    assert report["status"] == "succeeded" and json.loads((job.work / "job-report.json").read_text()) == \
        json.loads(json.dumps(report, default=str))
    for name in ("ra", "rb"):
        info = report["runs"][name]
        assert info["files"] == FILES and info["physical_coincidences"] == FILES * 70
        assert info["geometry"]["train"]["accepted"] > 0 and info["geometry"]["validation"]["accepted"] > 0
        inventory = json.loads((job.work / name / "inventory.json").read_text())
        assert [s["index"] for s in inventory["scans"]] == list(range(FILES))
        assert set(info["residuals_vs_pooled"]) == {"train", "validation"}  # test stays closed
    # the run that fed the .encal drops its declared prefix; the other declares none
    assert report["runs"]["ra"]["energy_calibration_prefix"] == {"1": 5, "2": 0}
    assert report["runs"]["ra"]["first_failure"]["validation"]["energy calibration prefix"] == 5
    assert report["runs"]["rb"]["energy_calibration_prefix"] == {"1": 0, "2": 0}
    assert report["pooled"]["usable_pairs"] >= 1 and report["lut"]["rows"] > 3000
    assert job.lut.exists() and job.lut.with_suffix(".json").exists()
    # an existing LUT is never overwritten
    with pytest.raises(tj.JobError, match="output LUT exists"):
        tj.run_job(job, log=lambda message: None)
    # resume with a new output: every run stage is reused without reading LDAT
    again = tj.load_job(job_file(job.path, base, items, lut="out/lut2.csv"))

    def no_reads(*args, **kwargs):
        raise AssertionError("LDAT read during resume")
    monkeypatch.setattr(tj.td, "scan_file", no_reads)
    monkeypatch.setattr(tc.timing_ingest, "LdatStream", no_reads)
    resumed = tj.run_job(again, log=lambda message: None)
    assert resumed["status"] == "succeeded"
    first, second = CanonicalLut.read(job.lut).rows, CanonicalLut.read(again.lut).rows
    for name in ("time_channel", "delay_ps", "sigma_ps", "status", "apply"):
        assert np.array_equal(first[name], second[name], equal_nan=name in ("delay_ps", "sigma_ps")), name
    # two disjoint channel pairs (minimodules 5 and 6): the larger component is applied, the other is not
    statuses = {ts.CHANNEL_STATUSES[k] for k in first["status"][first["usable_pairs"] > 0]}
    assert first["apply"].sum() == 2 and "disconnected" in statuses
    assert statuses <= {"calibrated", "low statistics", "disconnected"}


def test_pooled_fit_equals_union_of_runs():
    rng = np.random.default_rng(91)

    def rows(count, role=0, channels=(10, 20), shift=500.0):
        out = np.zeros(count, tc.OBSERVATION_DTYPE)
        out["channel_i"], out["channel_j"], out["role"] = channels[0], channels[1], role
        out["record"] = rng.integers(0, 10 ** 9, count)
        out["shell_tof_ps"] = [-951.0, -917.0, 917.0, 951.0]
        branch = rng.choice((-934.0, 934.0), count)
        out["dt_ps"] = np.rint(shift + branch + rng.normal(0, 200, count))
        return out

    class Fake:
        def __init__(self, parts):
            self.parts, self.partitions = parts, len(parts)

        def partition(self, p):
            return self.parts[p]

    a = Fake([np.concatenate([rows(20), rows(5, channels=(1, 2))]), rows(12, channels=(3, 4))])
    b = Fake([np.concatenate([rows(20), rows(5, channels=(1, 2), role=1)]), rows(25, channels=(3, 4))])
    pooled = tj.PooledObservations([a, b])
    model = tp.PairModel(200.0, 0.05)
    fits = tp.fit_observation_set(pooled, model)
    fits = fits[np.lexsort((fits["channel_j"], fits["channel_i"]))]
    union = tp.fit_pairs(np.concatenate([a.parts[0], b.parts[0], a.parts[1], b.parts[1]]), model)
    union = union[union["role"] == 0]
    assert len(fits) == len(union) == 3
    for name in tp.PAIR_FIT_DTYPE.names:
        assert np.array_equal(fits[name], union[name], equal_nan=True), name
    assert list(fits["events"]) == [5, 37, 40]
    # neither run alone has a 30-event pair; pooled, two pairs qualify for the kernel
    chosen = tj.kernel_rows(pooled, min_events=30)
    assert sorted({(int(r["channel_i"]), int(r["channel_j"])) for r in chosen}) == [(3, 4), (10, 20)]
    assert len(tj.kernel_rows(a, min_events=30)) == 0
    with pytest.raises(tj.JobError, match="partitions"):
        tj.PooledObservations([a, Fake([rows(3)])])


def test_energy_peak_check_flags_shift():
    rng = np.random.default_rng(92)

    def observations(peaks, count=40_000):
        out = np.zeros(count, tc.OBSERVATION_DTYPE)
        sm = rng.choice(list(peaks), count)
        out["sm_i"], out["sm_j"] = sm, 29
        out["energy_kev_i"] = rng.normal([peaks[s] for s in sm], 25)
        out["energy_kev_j"] = rng.normal(511, 25, count)
        part = type("Obs", (), {"partitions": 1, "partition": lambda self, p: out})()
        return part

    reference = tj.energy_peaks(observations({1: 511.0, 2: 511.0, 3: 511.0}))
    moved = tj.energy_peaks(observations({1: 511.0, 2: 506.0, 3: 540.0}))
    assert all(abs(reference[sm]["peak_kev"] - 511) <= 6 for sm in (1, 2, 3, 29))
    flags = tj.peak_flags(moved, reference, 0.02)
    assert [f["sm"] for f in flags] == [3] and flags[0]["relative_shift"] > 0.02
    assert tj.peak_flags(moved, reference, 0.08) == []
    sparse = tj.energy_peaks(observations({4: 511.0}, count=50), min_entries=200)
    assert sparse[4]["peak_kev"] is None and tj.peak_flags(sparse, reference, 0.02) == []


def test_disk_preflight_and_failed_run_stop(runs, tmp_path, monkeypatch):
    base, items = runs
    job = tj.load_job(job_file(tmp_path / "job.json", base, items))
    real = tj.shutil.disk_usage
    monkeypatch.setattr(tj.shutil, "disk_usage", lambda path: real(path)._replace(free=10))
    with pytest.raises(tj.JobError, match="GB free"):
        tj.run_job(job, log=lambda message: None)
    report = json.loads((job.work / "job-report.json").read_text())
    assert report["status"] == "failed" and "ra" in report["error"] and not job.lut.exists()
    monkeypatch.undo()
    # a run whose ledger names a missing split stops the job; nothing pooled is written
    broken = dict(items[1], name="rc", energy_calibration_prefix="none")
    ledger = json.loads(open(items[1]["ledger"]).read())
    ledger["attempts"][0]["artifacts"][0]["size_bytes"] += 1
    (tmp_path / "rc.json").write_text(json.dumps(ledger))
    broken["ledger"] = str(tmp_path / "rc.json")
    job = tj.load_job(job_file(tmp_path / "job2.json", base, [broken, items[0]], work_directory="work2"))
    assert tj.main([str(job.path)]) == 2
    report = json.loads((tmp_path / "work2/job-report.json").read_text())
    assert report["status"] == "failed" and report["error"].startswith("JobError: rc:")
    assert "size-mismatched" in report["error"] and not job.lut.exists()
    assert not (tmp_path / "work2/ra").exists()  # the job stops at the failed run


def calibrate_ledger(path, encal, run_item, validated):
    """Manager calibrate run.json: its .encal artifact and records_validated per split."""
    names = [p.name for p in sorted(Path(run_item["directory"]).glob(run_item["prefix"] + "*.ldat"))]
    ledger = {"attempts": [{"details": {"cli_status": "succeeded", "summary": {"inputs": [
        {"path": f"/mnt/nvme/data/x/{name}", "records_validated": validated + k} for k, name in enumerate(names)]}},
        "artifacts": [{"kind": "encal", "path": f"/mnt/nvme/encal/{encal.name}", "size_bytes": encal.stat().st_size}]}]}
    path.write_text(json.dumps(ledger))
    return path


def test_per_run_encal_prefix_and_solves(runs, tmp_path):
    base, items = runs
    (tmp_path / "setup").mkdir()
    setup = make_setup(tmp_path / "setup")
    shifted_run = make_run(tmp_path, setup, "rc", seed=3, shift=900.0)  # a different channel-pair offset
    own = tmp_path / "rc_resolved.encal"
    own.write_bytes((base / "r.encal").read_bytes())
    ledger = calibrate_ledger(tmp_path / "rc_cal.json", own, shifted_run, 7)
    shifted_run |= {"encal": str(own), "energy_calibration_prefix": {"calibration_run": str(ledger)}}
    job = tj.load_job(job_file(tmp_path / "job.json", base, [items[0], items[1], shifted_run], per_run_solves=True))
    report = tj.run_job(job, log=lambda message: None)
    rc = report["runs"]["rc"]
    assert rc["encal"] == str(own) and rc["encal_sha256"] == report["runs"]["ra"]["encal_sha256"]
    # held-out splits drop the calibrate run's records_validated; the run-specific .encal gets its own caches
    assert rc["energy_calibration_prefix"] == {"1": 8, "2": 9}
    assert rc["first_failure"]["validation"]["energy calibration prefix"] == 8
    tag = rc["encal_sha256"][:12]
    assert (job.work / "rc" / f"features-{tag}" / "manifest.json").exists()
    assert (job.work / "ra" / "features" / "manifest.json").exists()
    # per-run solves: consistent runs agree; the run with another offset stands out against the reference
    solves = report["per_run_solves"]
    assert set(solves) == {"ra", "rb", "rc"} and solves["ra"]["vs_reference_run"] is None
    assert solves["rb"]["vs_reference_run"]["channels"] == 2 and solves["rb"]["vs_reference_run"]["z_rms"] < 4
    assert solves["rc"]["vs_reference_run"]["z_rms"] > max(4, 2 * solves["rb"]["vs_reference_run"]["z_rms"])
    assert abs(abs(solves["rc"]["vs_reference_run"]["diff_ps_quantiles_1_50_99"][0]) - 150) < 60
    assert (job.work / "rc" / "per_run_delays.csv").exists() and "per_run_solve_seconds" in report["stages"]
    # the calibrate run must describe this .encal and every split
    with pytest.raises(tj.JobError, match="does not match"):
        tj.calibration_prefix(ledger, [{"index": 0, "path": Path(items[0]["directory"]) / "x.ldat"}],
                              base / "r.encal")
    with pytest.raises(tj.JobError, match="no records_validated"):
        tj.calibration_prefix(ledger, [{"index": 0, "path": Path(items[0]["directory"]) / "x.ldat"}], own)
