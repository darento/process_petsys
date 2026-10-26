"""Pooled IMAS skew job: declared files only, resume, per-run comparison (spec 011 T13)."""
import json

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, T0, side, write
from src.imas.skew import job as ij
from src.imas.skew.setup import ImasSetup

pytestmark = pytest.mark.fr("011-FR-6", "011-FR-8")

SIDES = [(mm, slab) for mm in (5, 6) for slab in range(4)]     # 8 TIME channels per SM, SMs 50 and 62


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    """Two runs x three splits of diametric ring-2 coincidences with a 60 a.u. photopeak."""
    base = tmp_path_factory.mktemp("job")
    setup = ImasSetup.from_files(CONFIG)
    rng = np.random.default_rng(13)

    def energy():
        return rng.normal(60.0, 4.0) if rng.random() < 0.85 else rng.uniform(5.0, 120.0)

    paths = []
    for index in range(6):
        records = []
        for _ in range(3200):
            (mm_a, slab_a), (mm_b, slab_b) = SIDES[rng.integers(8)], SIDES[rng.integers(8)]
            sm_a, sm_b = 50, 62
            ea, eb = energy(), energy()
            records.append((side(setup, sm_a, mm_a, time={slab_a: 9.0}, energy=[ea / 5] * 5),
                            side(setup, sm_b, mm_b, time={slab_b: 9.0}, energy=[eb / 5] * 5,
                                 stamp=T0 - int(rng.integers(-900, 900)))))
        paths.append(write(base / f"run{index // 3}_s{index % 3}.ldat", records))
    return paths


def job_file(path, splits, **changes):
    runs = []
    for r in range(2):
        a, b, c = splits[3 * r:3 * r + 3]
        runs.append({"name": f"r{r + 1}", "splits": [{"path": str(a), "roles": ["encal", "train"]},
                                                      {"path": str(b), "roles": ["train"]},
                                                      {"path": str(c), "roles": ["heldout"]}]})
    data = {"schema": ij.SCHEMA, "config": str(CONFIG), "runs": runs, "estimators": ["em", "pairs"],
            "per_run_solves": True, **changes}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_job_file_refusals(tmp_path, splits):
    good = json.loads(job_file(tmp_path / "j.json", splits).read_text(encoding="utf-8"))

    def refused(data, match):
        (tmp_path / "bad.json").write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ij.JobError, match=match):
            ij.load_job(tmp_path / "bad.json")

    refused({**good, "extra": 1}, "unknown keys")
    refused({**good, "schema": "other"}, "schema")
    run = good["runs"][0]
    refused({**good, "runs": [{**run, "splits": run["splits"] + [{"path": "x.ldat"}]}]}, "roles")
    refused({**good, "runs": [{**run, "splits": run["splits"] + [{"path": "x.ldat", "roles": ["dark"]}]}]}, "roles")
    refused({**good, "runs": [{**run, "splits": [run["splits"][0], {**run["splits"][2], "roles": ["heldout", "train"]}]}]},
            "held-out")
    refused({**good, "runs": [{**run, "splits": run["splits"] + [run["splits"][1]]}]}, "declared twice")
    refused({**good, "runs": [{**run, "splits": run["splits"][1:]}]}, "encal")
    refused({**good, "runs": [run, {**good["runs"][1], "name": run["name"]}]}, "unique")
    missing = {**good, "runs": [{**run, "splits": run["splits"] + [{"path": str(tmp_path / "gone.ldat"),
                                                                       "roles": ["train"]}]}]}
    (tmp_path / "missing.json").write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ij.JobError, match="not found"):
        ij.run_job(ij.load_job(tmp_path / "missing.json"), tmp_path / "work")
    assert not (tmp_path / "work").exists()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ij.JobError, match="not empty"):
        ij.run_job(ij.load_job(tmp_path / "j.json"), unrelated)


@pytest.mark.slow
def test_job_runs_resumes_and_compares_runs(tmp_path, splits):
    work = tmp_path / "work"
    job = ij.load_job(job_file(tmp_path / "j.json", splits))
    report = ij.run_job(job, work, progress=lambda _: None)
    for name in ("em", "pairs"):
        assert (work / f"lut_{name}.csv").exists() and (work / f"lut_{name}.json").exists()
    assert set(report["runs"]) == {"r1", "r2"}
    for run in report["runs"].values():
        assert run["encal"]["status_counts"]["fit"] == 16 and not run["encal"]["reused"]
        assert run["observations"]["observations"]["train"] > 3000 and run["observations"]["observations"]["test"] > 1400
    em = report["estimators"]["em"]
    assert em["applied"] == 16 and em["status_counts"]["unobserved"] == len(ImasSetup.from_files(CONFIG).time_channels) - 16
    peaks = report["per_run"]["peaks_511_kev"]["r2 / r1"]
    assert peaks["channels"] == 16 and abs(peaks["median_ratio"] - 1) < 0.02
    for name in ("r1", "r2"):
        entry = report["per_run"][name]
        assert entry["train"]["events"] > 3000 and entry["heldout"]["events"] > 1400
        assert entry["alone_versus_pooled"]["channels"] == 16
    assert report["held_out"]["em"]["events"] > 2800
    # resume: calibrations and streamed files are reused, not redone
    stamps = {p: p.stat().st_mtime_ns for p in work.glob("r*/observations/files/f*/done.json")}
    again = ij.run_job(job, work, progress=lambda _: None)
    assert all(r["encal"]["reused"] for r in again["runs"].values())
    assert {p: p.stat().st_mtime_ns for p in work.glob("r*/observations/files/f*/done.json")} == stamps
    assert len(stamps) == 6          # encal+train, train and held-out split of each run
    # another job (one split fewer) is refused in the same work directory
    other = json.loads((tmp_path / "j.json").read_text(encoding="utf-8"))
    other["runs"][1]["splits"] = other["runs"][1]["splits"][:2]
    (tmp_path / "other.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ij.JobError, match="another job"):
        ij.run_job(ij.load_job(tmp_path / "other.json"), work)
