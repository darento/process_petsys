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

from cornell_timing_helpers import (FILES, ROOT, T0, channels, job_file, make_run, make_setup, runs,  # noqa: F401
                                    shifted, side, write)
from src.cornell.skew import cache as tc
from src.cornell.skew import job as tj
from src.cornell.skew import pairs as tp
from src.cornell.skew import plots as tpl
from src.cornell.skew import solver as ts
from src.cornell.skew.lut import (CanonicalLut, LutError, activated_ini, read_native_table,
                                  write_conversion_manifest, write_native_table)
from src.utils import get_absolute_id, get_electronics_nums

pytestmark = pytest.mark.fr("008-FR-18")
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
        ({"reproducibility_floor": True}, "per_run_solves"), ({"pattern_outlier_ps": 0}, "pattern outlier"),
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
    # FR-19: per-ASIC plots next to the LUT; pattern outliers listed in the report and manifest
    plots = tpl.plot_paths(job.lut)
    assert report["asic_pattern"]["files"] == [str(p) for p in plots] and all(p.stat().st_size > 0 for p in plots)
    assert report["asic_pattern"]["threshold_ps"] == 1000.0 and report["asic_pattern"]["outliers"] == []
    assert json.loads(job.lut.with_suffix(".json").read_text())["provenance"]["asic_pattern_outliers"] == \
        {"threshold_ps": 1000.0, "channels": []}
    # an existing LUT is never overwritten
    with pytest.raises(tj.JobError, match="output LUT exists"):
        tj.run_job(job, log=lambda message: None)
    # resume with a new output: every run stage is reused without reading LDAT
    again = tj.load_job(job_file(job.path, base, items, lut="out/lut2.csv"))

    def no_reads(*args, **kwargs):
        raise AssertionError("LDAT read during resume")
    monkeypatch.setattr(tj.td, "scan_file", no_reads)
    monkeypatch.setattr(tc.ingest, "LdatStream", no_reads)
    resumed = tj.run_job(again, log=lambda message: None)
    assert resumed["status"] == "succeeded" and all(p.exists() for p in tpl.plot_paths(again.lut))
    # an existing plot blocks a new LUT of that name before any work
    blocked = tj.load_job(job_file(job.path, base, items, lut="out/lut3.csv"))
    tpl.plot_paths(blocked.lut)[2].write_bytes(b"keep")
    with pytest.raises(tj.JobError, match="per-ASIC plots"):
        tj.run_job(blocked, log=lambda message: None)
    assert tpl.plot_paths(blocked.lut)[2].read_bytes() == b"keep" and not blocked.lut.exists()
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


def corrected_run(base, setup, name, *, seed, shift, mm, table, ini):
    """A run 'converted with the native table': splits plus a native-conversion manifest, no ledger."""
    item = make_run(base, setup, name, seed=seed, shift=shift, mms=(mm,))
    Path(item.pop("ledger")).unlink()
    manifest = base / f"{name}_conversion.json"
    write_conversion_manifest(item["directory"], item["prefix"], table, ini, manifest,
                              raw_source=f"/raw/{name}", command="convert_raw_to_coincidence --writeBinaryCompact")
    return item | {"conversion": str(manifest)}


@pytest.mark.fr("008-FR-20")
def test_second_pass_on_corrected_runs(runs, finished, tmp_path):
    base, items = runs
    first, _ = finished
    # the first pass (shift 600 ps) gives the source LUT and its native table
    table = tmp_path / "time_offset_v3.tsv"
    write_native_table(first.lut, table)
    (tmp_path / "config.ini").write_text("[main]\n#time_offset_calibration_table = %CDIR%/x.tsv\n")
    ini = tmp_path / "config_v3.ini"
    activated_ini(tmp_path / "config.ini", table, ini)
    native, _, _ = read_native_table(table)
    (tmp_path / "setup").mkdir()
    setup = make_setup(tmp_path / "setup")
    mm = next(m for m in (5, 6) if channels(setup, 1, m)[0][3] in native)
    c1, c16 = channels(setup, 1, mm)[0][3], channels(setup, 16, mm)[0][3]
    # the true uncorrected shift is 700 ps (first pass: 600); corrected data keep 700 - (b1 - b16)
    shift = 700.0 - (native[c1] - native[c16])
    ca = corrected_run(tmp_path, setup, "ca", seed=11, shift=shift, mm=mm, table=table, ini=ini)
    ca["energy_calibration_prefix"] = {"1": 50, "2": 0}
    cb = corrected_run(tmp_path, setup, "cb", seed=12, shift=shift, mm=mm, table=table, ini=ini)
    # cb's own .encal came from a calibrate run over the uncorrected files: other names, same split indices
    own = tmp_path / "cb_resolved.encal"
    own.write_bytes((base / "r.encal").read_bytes())
    ledger = {"attempts": [{"details": {"cli_status": "succeeded", "summary": {"inputs": [
        {"path": f"/mnt/nvme/data/x/norm_test_rb_coincCompact_{k:08d}.ldat", "records_validated": 8 + k}
        for k in range(FILES)]}},
        "artifacts": [{"kind": "encal", "path": f"/x/{own.name}", "size_bytes": own.stat().st_size}]}]}
    (tmp_path / "cb_cal.json").write_text(json.dumps(ledger))
    cb |= {"encal": str(own), "energy_calibration_prefix": {"calibration_run": str(tmp_path / "cb_cal.json")}}

    job = tj.load_job(job_file(tmp_path / "job.json", base, [ca, cb], native_source_lut=str(first.lut),
                               lut="out/residual.csv", per_run_solves=True))
    report = tj.run_job(job, log=lambda message: None)
    assert report["status"] == "succeeded"
    lut_sha = json.loads(first.lut.with_suffix(".json").read_text())["table"]["sha256"]
    state = report["input_correction"]
    assert state["state"] == "native" and state["source_lut_sha256"] == lut_sha
    assert set(state["conversion_manifests"]) == {"ca", "cb"}
    # held-out prefixes: uncorrected counts by split index plus 2 %
    assert report["runs"]["ca"]["energy_calibration_prefix"] == {"1": 51, "2": 0}
    assert report["runs"]["ca"]["first_failure"]["validation"]["energy calibration prefix"] == 51
    assert report["runs"]["cb"]["energy_calibration_prefix"] == {"1": 10, "2": 11}  # ceil(9 * 1.02), ceil(10 * 1.02)
    assert report["runs"]["cb"]["energy_calibration_prefix_margin"] == 0.02
    # residual LUT: the remaining ~100 ps, never exported natively
    residual = CanonicalLut.read(job.lut)
    assert residual.input_correction["state"] == "native"
    with pytest.raises(LutError, match="residual LUT"):
        write_native_table(job.lut, tmp_path / "never.tsv")
    assert sorted(residual.rows["time_channel"][residual.rows["apply"]]) == sorted([c1, c16])
    # composed LUT for uncorrected data recovers the true 700 ps, not the first pass
    composed_path = job.lut.with_name("residual_composed.csv")
    composed = CanonicalLut.read(composed_path)
    assert composed.input_correction == {"state": "uncorrected"}
    rows = {int(r["time_channel"]): r for r in composed.rows}
    sigma = float(np.hypot(rows[c1]["sigma_ps"], rows[c16]["sigma_ps"]))
    difference = composed.delay(c1) - composed.delay(c16)
    assert abs(difference - 700.0) < 4 * sigma and abs(difference - 600.0) > 8 * sigma and sigma < 30
    assert report["composed_lut"]["composition"] == {"residual plus native": 2, "native only": 0,
                                                     "residual only": 0, "not applied": len(composed.rows) - 2}
    assert report["composed_lut"]["sha256"] == composed.manifest["table"]["sha256"]
    assert composed.manifest["provenance"]["residual_lut"]["sha256"] == report["lut"]["sha256"]
    assert all(p.exists() for p in tpl.plot_paths(composed_path))
    native_v4 = write_native_table(composed_path, tmp_path / "time_offset_v4.tsv")
    assert native_v4["table"]["rows"] == 2
    assert report["second_pass"]["applied_channels"] == 2 and report["second_pass"]["listed_in_native_table"] == 2

    # refusals before any work: mixed states, missing/unexpected/wrong source LUT, ledger and conversion together
    for run_items, extra, message in (
            ([ca, items[0]], {"native_source_lut": str(first.lut)}, "mix"),
            ([ca, cb], {}, "native_source_lut"),
            (items, {"native_source_lut": str(first.lut)}, "native_source_lut"),
            ([ca, cb], {"native_source_lut": str(job.lut)}, "not the source LUT")):
        bad = tj.load_job(job_file(tmp_path / "bad.json", base, run_items, work_directory="bad", lut="out/bad.csv",
                                   **extra))
        with pytest.raises(tj.JobError, match=message):
            tj.run_job(bad, log=lambda message: None)
        assert not (tmp_path / "bad" / "ca").exists() and not (tmp_path / "out/bad.csv").exists()
    with pytest.raises(tj.JobError, match="run entries"):
        tj.load_job(job_file(tmp_path / "bad.json", base, [dict(ca, ledger=items[0]["ledger"])]))


@pytest.mark.fr("bug-composed-provenance")
def test_composed_lut_carries_second_pass_listings(runs, finished, tmp_path, monkeypatch):
    """The composed LUT lists the second pass's cross-run and ASIC pattern outliers (v4 manifest showed 0)."""
    listed_run = [{"time_channel": 4877, "runs": {"r7": {"difference_ps": 7944.5, "z": 66.8}}}]
    monkeypatch.setattr(tj.ts, "cross_run_outliers", lambda *args, **kwargs: listed_run)
    test_second_pass_on_corrected_runs(runs, finished, tmp_path)
    report = json.loads((tmp_path / "work" / "job-report.json").read_text())
    composed = CanonicalLut.read(tmp_path / "out" / "residual_composed.csv")
    provenance = composed.manifest["provenance"]
    assert report["cross_run_outliers"] == listed_run
    assert provenance["cross_run_outliers"] == report["cross_run_outliers"]
    listed = [o["time_channel"] for o in report["composed_lut"]["asic_pattern"]["outliers"]]
    assert [o["time_channel"] for o in provenance["asic_pattern_outliers"]["channels"]] == listed
    native = json.loads((tmp_path / "time_offset_v4.json").read_text())
    assert native["policy"]["listed_cross_run_outliers"] == len(report["cross_run_outliers"])


@pytest.mark.fr("008-FR-20")
def test_second_pass_diagnostics():
    rng = np.random.default_rng(93)
    ids = [get_absolute_id(0, 0, chip, ch) for chip in range(40) for ch in range(64)]
    rows = np.zeros(len(ids), ts.CHANNEL_DTYPE)
    rows["time_channel"], rows["status"], rows["apply"], rows["sigma_ps"] = ids, ts.CALIBRATED, True, 5.0
    native = {c: float(v) for c, v in zip(ids, rng.uniform(-5000, 5000, len(ids)))}
    asic_channel = np.array([get_electronics_nums(c)[3] for c in ids])
    # clipping-like shrinkage (1 % of the native offset), +30 ps on ASIC channel 11, a common 7 ps gauge
    rows["delay_ps"] = (0.01 * np.array([native[c] for c in ids]) + 30.0 * (asic_channel == 11) + 7.0
                        + rng.normal(0, 5, len(ids)))
    rows["apply"][:3] = False  # not applied in the second pass: excluded
    rows["status"][:3] = ts.CHANNEL_STATUSES.index("unobserved")
    pairs = np.zeros(4000, ts.PAIR_RESIDUAL_DTYPE)
    pairs["channel_i"], pairs["channel_j"] = rng.choice(ids, (2, 4000))
    shift = np.array([native[int(i)] - native[int(j)] for i, j in zip(pairs["channel_i"], pairs["channel_j"])])
    pairs["normalized_residual"] = (rng.normal(0, 1, 4000) * (1 + np.abs(shift) / 4000)
                                    + 0.5 * np.sign(shift) * (np.abs(shift) >= 4000))
    source = type("Lut", (), {"manifest": {"provenance": {
        "cross_run_outliers": [{"time_channel": ids[11], "runs": {}}, {"time_channel": ids[0], "runs": {}}],
        "asic_pattern_outliers": {"channels": [{"time_channel": ids[13]}]}}}})()
    out = tj.second_pass_report(ts.Solution(rows, pairs), native, source)
    assert out["applied_channels"] == out["listed_in_native_table"] == len(ids) - 3
    trend = out["trend_vs_native"]
    assert trend["slope"] == pytest.approx(0.01, abs=5 * trend["slope_sigma"])
    assert [b["from"] for b in trend["bins_abs_native_ps"]] == [0, 1000, 2000, 4000]
    assert trend["bins_abs_native_ps"][-1]["to"] is None
    groups = out["asic_channel_groups"]
    assert groups["11"]["count"] == 40 and groups["13"]["count"] == 40
    assert groups["11"]["mean_ps"] - groups["other"]["mean_ps"] == pytest.approx(30, abs=10)
    assert abs(groups["13"]["mean_ps"] - groups["other"]["mean_ps"]) < 10
    chi2 = [b["chi2_per_pair"] for b in out["pair_chi2_by_abs_native_pair_shift"]]
    assert chi2 == sorted(chi2) and chi2[0] < 1.6
    assert out["pair_chi2_by_abs_native_pair_shift"][-1]["mean_signed_z"] == pytest.approx(0.5, abs=0.25)
    assert [e["time_channel"] for e in out["source_cross_run_outliers"]] == [ids[11], ids[0]]
    assert out["source_cross_run_outliers"][0]["residual_ps"] > 15  # ASIC channel 11 carries the +30 ps
    assert out["source_cross_run_outliers"][1]["residual_ps"] is None  # not applied in the second pass
    assert out["source_asic_pattern_outliers"][0]["time_channel"] == ids[13]
    assert not out["source_asic_pattern_outliers_computed"]
    json.dumps(out, allow_nan=False)
    # a source LUT from before FR-19 lists no pattern outliers: computed from its applied delays
    source_rows = rows.copy()
    source_rows["apply"], source_rows["delay_ps"] = True, rng.normal(0, 20, len(ids))
    source_rows["delay_ps"][100] += 3000.0
    old = type("Lut", (), {"manifest": {"provenance": {}}, "rows": source_rows})()
    out = tj.second_pass_report(ts.Solution(rows, pairs), native, old)
    assert out["source_asic_pattern_outliers_computed"] and out["source_cross_run_outliers"] == []
    assert [e["time_channel"] for e in out["source_asic_pattern_outliers"]] == [ids[100]]
