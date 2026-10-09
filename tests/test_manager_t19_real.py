"""Spec 003 T19 real-data checks on the January 2026 Cornell acquisition (spec 007 T17).

Moved from scripts/petsys_manager_real_check.py (run through petsys_manager_reference_check.py --real).
The inputs are named by tests/data/baselines/t19_jan2026_manifest.json: compact splits 3-8 of the
2026-01-19 two-Na-22-source acquisition and the COG limits under PETSYS_DATA_DIR, the owner's
reference-written per-slab .encal/status under PETSYS_CAL_DIR, and the January config
(tests/data/configs/cornell_january.yaml, January map). Fixed inputs are not used (fixed format
discontinued, spec 007 Clarify). Expected values come from t19_jan2026_result.json, the
2026-10-05 Windows run in which every value below equalled the reference scripts (scripts_cornell/
cornell_slab_en_cal.py, cornell_system_validation.py) run on the same files; no test loads them
(007-FR-3). Cuts: min_ch 4, en_min_ch 0.2 a.u. (calibration, QC); QC seed 19. Inputs are read only.
"""

import hashlib
import json
import os
from pathlib import Path
import random

import pytest

from helpers import DATA, REPO
from src.cornell import calibration as cal, qc, qc_report
from src.cornell.inputs import load_limits, load_processing_config
from src.petsys_manager.contracts import InputDescriptor

BASELINES = DATA / "baselines"
MANIFEST = json.loads((BASELINES / "t19_jan2026_manifest.json").read_text(encoding="utf-8"))
RESULT = json.loads((BASELINES / "t19_jan2026_result.json").read_text(encoding="utf-8"))

pytestmark = [pytest.mark.real_data, pytest.mark.slow, pytest.mark.fr("007-FR-4")]


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def evidence(label):
    """The baseline run's evidence for the check whose label starts with ``label``."""
    (found,) = [check["evidence"] for check in RESULT["checks"] if check["check"].startswith(label)]
    return found


def identity(paths):
    return {path: (os.stat(path).st_size, os.stat(path).st_mtime_ns) for path in paths}


@pytest.fixture
def inputs(real_data_file, real_cal_file):
    """Manifest inputs by the names of the baseline's ``inputs`` record; skips naming a missing one."""
    named = {f"compact_inputs[{i}]": real_data_file(path) for i, path in enumerate(MANIFEST["compact_inputs"])}
    named["cog_limits"] = real_data_file(MANIFEST["cog_limits"])
    for key, path in MANIFEST["reference_outputs"].items():
        named[f"reference_output:{key}"] = real_cal_file(path)
    return named


@pytest.fixture
def unchanged(inputs):
    """The inputs, checked unchanged (size, mtime) after the test."""
    before = identity(inputs.values())
    yield inputs
    assert identity(inputs.values()) == before, "inputs changed during the run"


def compact(named):
    return [InputDescriptor(named[f"compact_inputs[{i}]"], "compact", "coincidence")
            for i in range(len(MANIFEST["compact_inputs"]))]


def config(action):
    return load_processing_config(REPO / MANIFEST["config"], processing_root=REPO, action=action)


def test_inputs_are_the_baseline_files(inputs):
    expected = {key: RESULT["inputs"][key]["sha256"] for key in inputs}
    assert {key: sha256(path) for key, path in inputs.items()} == expected


@pytest.mark.fr("003-FR-12", "003-FR-21")  # spec 003 T19, T20
def test_per_slab_calibration_equals_the_reference_written_files(unchanged):
    settings = config("calibrate")
    assert (int(settings.values["min_ch"]), float(settings.values["en_min_ch"])) == (4, 0.2)
    result = cal.calibrate(compact(unchanged), settings, positions=1)
    encal, status = cal.encal_text(result), cal.status_text(result)
    assert encal == unchanged["reference_output:per_slab_encal"].read_text(encoding="utf-8")
    assert status == unchanged["reference_output:per_slab_status"].read_text(encoding="utf-8")
    expected = evidence("per-slab calibration (P = 1, compact)")
    assert len(encal.splitlines()) - 1 == expected["rows"]
    assert result.status_counts() == expected["status_counts"]
    assert [[Path(f.path).name, f.records_read, f.events_passed, f.stopped_at_limit] for f in result.files] == expected["files"]


@pytest.mark.fr("003-FR-12", "003-FR-21")  # spec 003 T19, T20
def test_position_calibration_with_event_limit_reproduces_the_baseline(unchanged):
    settings = config("calibrate")
    cog = load_limits(unchanged["cog_limits"], settings.mapping, kind="cog")
    expected = evidence("position calibration (P = 5, first passing events")
    limit = MANIFEST["position_oracle_event_limit"]
    assert expected["event_limit_per_file"] == limit
    result = cal.calibrate(compact(unchanged), settings, cog, positions=MANIFEST["positions"], event_limit=limit)
    assert sum(f.accepted_sides for f in result.files) == expected["accepted_sides"]
    assert result.status_counts() == expected["status_counts"]
    assert len(cal.encal_text(result).splitlines()) - 2 == expected["encal_rows"]


@pytest.mark.fr("003-FR-14")  # spec 003 T19
def test_qc_with_plots_and_slabs_reproduces_the_baseline(unchanged, tmp_path):
    random.seed(MANIFEST["seeds"]["qc"])
    result = qc.run_qc(compact(unchanged), config("qc"), plots=True, slabs=True)
    written = qc_report.write_report(result, tmp_path / "qc")
    expected = evidence("QC (compact, plots and slabs)")
    totals = result.totals()
    assert totals["accepted_pairs"] == expected["accepted_pairs"]
    assert (totals["rejected"], totals["slab_flags"]) == (expected["rejected"], expected["slab_flags"])
    assert [[Path(f.path).name, f.records_read, f.pairs_processed, f.occupancy_pairs, f.accepted_pairs, f.stopped_at_limit]
            for f in result.files] == expected["files"]
    statuses = {}
    for entry in [*result.minimodule_fits, *result.slab_fits]:
        statuses[entry.status] = statuses.get(entry.status, 0) + 1
    assert statuses == expected["fit_statuses"]
    assert len(result.floods) == expected["flood_sms"]
    assert [Path(path).name for path in written] == expected["report_files"]
