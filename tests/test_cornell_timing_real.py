"""Point-source CTR on the Cornell centre acquisition (spec 008 T9b, FR-13; real data, bounded prefix)."""
import json
from pathlib import Path

import pytest

from src.cornell.skew import ctr

pytestmark = [pytest.mark.fr("008-FR-13"), pytest.mark.real_data]
CENTRE = "Cornell/full_system/point_source/NEMA_SR_10142026_X0Y0Z0_conv_uncorr/NEMA_SR_10142026_X0Y0Z0_coincCompact.ldat"
LUT = "Cornell/full_system/t10_replacement/t10/time_delays_t10.csv"
LUT_SHA256 = "b1b211c1eb41c02d888272656d2c2a549fc051107933acf4a486d0be84cd6b74"  # frozen r7-only LUT (T10 job)
ENCAL = "norm_10082026_r7_coincCompact_resolved.encal"
RECORDS = 300_000


def test_centre_point_source_prefix(real_data_file, real_cal_file, tmp_path):
    ldat, lut, encal = real_data_file(CENTRE), real_data_file(LUT), real_cal_file(ENCAL)
    job = Path(json.loads(lut.with_suffix(".json").read_text(encoding="utf-8"))["provenance"]["job"])
    if not job.is_file():
        pytest.skip(f"real_data: the LUT's job file {job} is missing on this machine")
    spec = {"schema": ctr.SCHEMA, "config": "tests/data/configs/cornell_september.yaml", "encal": str(encal),
            "lut": str(lut), "lut_sha256": LUT_SHA256, "output": str(tmp_path / "ctr.json"),
            "acquisitions": [{"name": "X0Y0Z0", "ldat": str(ldat), "source_mm": "lor_convergence",
                              "source_sigma_mm": [1.5, 1.5, 1.5], "input_correction": "uncorrected"}],
            "bootstrap": 20, "max_records": RECORDS}
    (tmp_path / "ctr_in.json").write_text(json.dumps(spec))
    report = ctr.run_ctr(ctr.load_ctr(tmp_path / "ctr_in.json"), log=lambda *_: None)
    stages = report["counts"]["X0Y0Z0"]["stages"]
    assert stages["records"] == RECORDS
    assert stages["passed pre-geometry"] == sum(stages[k] for k in ("invalid endpoint", "no applied delay",
                                                                     "LOR outside distance", "outside window",
                                                                     "in window"))
    assert sum(report["counts"]["X0Y0Z0"]["pre_geometry"].values()) == RECORDS
    source = report["acquisitions"][0]["source"]
    assert source["method"] == "lor_convergence" and source["median_miss_mm"] < 5
    fit = report["results"]["pooled"]["fit"]
    assert stages["in window"] > 0.5 * stages["passed pre-geometry"] and fit["status"] == "fitted"
    assert 100 < fit["fwhm_ps"] < 2_000
    assert report["frozen_inputs"]["lut_sha256"] == LUT_SHA256
