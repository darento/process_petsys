"""Spec 013: IMAS NEMA NU 2-2018 CTR on the NECR phantom (``src.imas.skew.necr``)."""
import json

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, channels, write
from src.imas.skew import ctr, necr
from src.imas.skew.setup import ImasSetup
from src.timing.ctr import CtrError
import necr_helpers as h


# --- T3: NECR file and frozen inputs (FR-1) ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    base = tmp_path_factory.mktemp("necr_tables")
    setup = ImasSetup.from_files(CONFIG)
    times = sorted({ch for sm in (48, 60) for mm in (5, 6) for ch in channels(setup, sm, mm)[0]})
    delays = dict(zip(times, np.random.default_rng(2).normal(0.0, 500.0, len(times))))
    case = h.frozen_tables(base, setup, times, delays)
    case["frame"] = write(base / "frame7_00000024.ldat", [])
    return case


def _edit(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.fr("013-FR-1")
def test_file_loads_with_relative_paths(tables):
    path = h.necr_file(tables, {"frame7": ["frame7_00000024.ldat"]}, "relative.json")
    spec = necr.load_necr(path)
    assert spec.frames[0].ldat == (tables["base"] / "frame7_00000024.ldat",)
    assert spec.frames[0].activity_kbq_ml == 5.0 and spec.headline_line == "fitted"
    assert spec.lines["fitted"].method == necr.FITTED and spec.lines["image"].p1 == tuple(h.LEGACY_P1)
    assert (spec.gates.frame, spec.gates.transfer, spec.gates.acceptance) == ("frame7", ("legacy", "em"),
                                                                              ("em", "em_walk"))
    assert spec.table_names == {"uncorrected", "legacy", "em", "em_walk"}
    inputs, _, delays = ctr.frozen_inputs(spec)
    assert set(delays) == {"uncorrected", "legacy", "em", "em_walk"}
    assert inputs["walks"]["em_walk"]["skew"] == "em"


def _drop(key):
    return lambda d: d.pop(key)


REFUSALS = {
    "missing key": (_drop("lines"), "file keys"),
    "extra key": (lambda d: d.update(sources={}), "file keys"),
    "frame without ldat": (lambda d: d["frames"][0].update(ldat=[]), "ldat"),
    "negative activity": (lambda d: d["frames"][0].update(activity_kbq_ml=-1.0), "activity_kbq_ml"),
    "missing activity": (lambda d: d["frames"][0].pop("activity_kbq_ml"), "frames need"),
    "duplicate frame": (lambda d: d["frames"].append(dict(d["frames"][0])), "unique"),
    "image line P1 = P2": (lambda d: d["lines"]["image"].update(p2=d["lines"]["image"]["p1"]), "lines.image"),
    "unknown line option": (lambda d: d["lines"]["fitted"].update(radius=3), "lines.fitted"),
    "headline not a line": (lambda d: d.update(headline_line="pet"), "headline_line"),
    "gate unknown table": (lambda d: d["gates"].update(acceptance=["em", "pairs"]), "gates.acceptance"),
    "gate unknown frame": (lambda d: d["gates"].update(frame="frame9"), "gates.frame"),
    "legacy results unknown frame": (lambda d: d.update(legacy_results={"source": "x", "frames": {"f9": {
        "fwhm_raw_ps": 1, "fwtm_raw_ps": 1, "fwhm_bg_ps": 1, "fwtm_bg_ps": 1}}}), "legacy_results"),
    "one bootstrap replicate": (lambda d: d.update(bootstrap=1), "bootstrap"),
    "reserved LUT name": (lambda d: d["luts"].update(legacy=d["luts"]["em"]), "reserved"),
    "walk without its skew": (lambda d: d["walks"]["em_walk"].update(skew="em2"), "walks.em_walk"),
}


@pytest.mark.fr("013-FR-1")
@pytest.mark.parametrize("case", REFUSALS, ids=list(REFUSALS))
def test_file_refusals_name_the_key(tables, case):
    change, message = REFUSALS[case]
    path = _edit(h.necr_file(tables, {"frame7": [tables["frame"]]}, "refused.json"), change)
    with pytest.raises(CtrError, match=message):
        necr.load_necr(path)


@pytest.mark.fr("013-FR-1")
def test_frozen_inputs_refused(tables, tmp_path):
    from src.imas.skew import walk as iw
    from src.timing.cache import _sha256

    def check(change, message):
        path = _edit(h.necr_file(tables, {"frame7": [tables["frame"]]}, "frozen.json"), change)
        with pytest.raises(CtrError, match=message):
            ctr.frozen_inputs(necr.load_necr(path))

    check(lambda d: d["luts"]["em"].update(sha256="0" * 64), "LUT em: SHA-256")
    check(lambda d: d["walks"]["em_walk"].update(sha256="0" * 64), "walk em_walk: SHA-256")
    check(lambda d: d["legacy"].update(sha256="0" * 64), "legacy: SHA-256")
    encal = tmp_path / "encal.tsv"
    encal.write_text(tables["encal"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    check(lambda d: d.update(encal=str(encal)), "encal")
    other = tables["skew_lut"](tmp_path / "lut_other.csv", {c: d + 1.0 for c, d in tables["delays"].items()})
    walk = iw.WalkLut.read(tables["walk"])
    assert walk.manifest["skew_lut_sha256"] != _sha256(other)
    check(lambda d: d["luts"].update(em={"path": str(other), "sha256": _sha256(other)}), "walk em_walk")


@pytest.mark.fr("013-FR-1")
def test_existing_output_is_refused(tables):
    spec = necr.load_necr(h.necr_file(tables, {"frame7": [tables["frame"]]}, "output.json"))
    assert necr.prepare_output(spec).is_dir()
    with pytest.raises(CtrError, match="already exists"):
        necr.prepare_output(spec)


@pytest.mark.fr("013-FR-1")
def test_splits_and_gate_tables_checked(tables, tmp_path):
    def check(change, message):
        path = _edit(h.necr_file(tables, {"frame7": [tables["frame"]]}, "splits.json"), change)
        with pytest.raises(CtrError, match=message):
            ctr.frozen_inputs(necr.load_necr(path))

    check(lambda d: d["frames"][0].update(ldat=[str(tmp_path / "missing.ldat")]), "missing LDAT")
    check(lambda d: d["frames"][0].update(ldat=[str(tables["base"] / "skew_0.ldat")]), "input of the LUT job")
    em2 = tables["skew_lut"](tmp_path / "lut_em2.csv", walk_lut_sha256="f" * 64)
    from src.timing.cache import _sha256
    check(lambda d: (d["luts"].update(em2={"path": str(em2), "sha256": _sha256(em2)}),
                     d["gates"].update(acceptance=["em", "em2"])), "solved with walk LUT")


# --- T4/T5: synthetic NECR acquisition (FR-2 to FR-6) -------------------------------------------------------------

def _necr_case(base, *, em_error_ps=0.0, legacy_missing_sm=None):
    setup = ImasSetup.from_files(CONFIG)
    times = h.all_time_channels(setup)
    delays = dict(zip(times, np.random.default_rng(2).normal(0.0, 500.0, len(times))))
    missing = [] if legacy_missing_sm is None else [ch for mm in range(16)
                                                     for ch in channels(setup, legacy_missing_sm, mm)[0]]
    case = h.frozen_tables(base, setup, times, delays, em_error_ps=em_error_ps, legacy_missing=missing)
    endpoints = h.channel_endpoints(base, setup, case["encal"])
    frames = {}
    for k in (1, 2):
        h.necr_frame(base / f"necr_{k}_00000001.ldat", setup, endpoints, delays, seed=k)
        frames[f"frame{k}"] = [base / f"necr_{k}_00000001.ldat"]
    case["frames"], case["missing"] = frames, missing
    return case


@pytest.fixture(scope="module")
def necr_run(tmp_path_factory):
    case = _necr_case(tmp_path_factory.mktemp("necr_run"), legacy_missing_sm=50)
    spec = necr.load_necr(h.necr_file(case, case["frames"], legacy_results={"source": "synthetic .lm", "frames": {
        "frame1": {"fwhm_raw_ps": 768, "fwtm_raw_ps": 1603, "fwhm_bg_ps": 744, "fwtm_bg_ps": 1513}}}))
    return case, spec, necr.run_necr(spec, log=lambda *a: None)


@pytest.mark.slow
@pytest.mark.fr("013-FR-3")
def test_accumulate_fits_the_line(necr_run):
    _, _, report = necr_run
    line = report["lines"]["fitted"]
    z = np.array([-350.0, 0.0, 350.0])
    error = np.linalg.norm(necr.xy_at(line["p1_mm"], line["p2_mm"], z) - necr.xy_at(*h.true_line(), z), axis=1)
    assert error.max() < 1.0
    assert set(line["per_frame"]) == {"frame1", "frame2"}
    assert all(f["status"] == "fitted" for f in line["per_frame"].values())
    image = report["lines"]["image"]
    expected = np.linalg.norm(necr.xy_at(line["p1_mm"], line["p2_mm"], z) - necr.xy_at(image["p1_mm"],
                                                                                         image["p2_mm"], z), axis=1)
    np.testing.assert_allclose(report["line_distances"]["distance_mm"]["fitted - image"], expected)


@pytest.mark.slow
@pytest.mark.fr("013-FR-2", "013-FR-4", "013-FR-5")
def test_accumulate_counts_and_widths(necr_run):
    case, spec, report = necr_run
    for frame in spec.frames:
        stages = report["counts"][frame.name]["stages"]
        assert stages["records"] == sum(report["counts"][frame.name]["pre_geometry"].values())
        assert stages["passed pre-geometry"] == (stages["invalid endpoint"] + stages["no applied delay"]
                                                 + stages["walk input missing"] + stages["same events"])
        assert stages["no applied delay"] > 0          # the legacy table has no delay for SM 50
        for line in ("fitted", "image"):                # the same events for every table and line
            assert stages["kept (NEMA)"][line]["em"] <= stages["same events"]
    pooled = report["results"]["pooled"]["fitted"]
    bg = {name: entry["nema"]["background_subtracted"] for name, entry in pooled.items()}
    assert bg["legacy"]["fwhm_ps"] > bg["em"]["fwhm_ps"] > bg["em_walk"]["fwhm_ps"]
    for kind in ("raw",):
        assert pooled["uncorrected"]["nema"][kind]["fwhm_ps"] > pooled["legacy"]["nema"][kind]["fwhm_ps"]
    assert pooled["uncorrected"]["model_free"]["model_free_fwhm_ps"] > pooled["legacy"]["model_free"][
        "model_free_fwhm_ps"]
    for name in ("legacy", "em", "em_walk"):             # centred: Gaussian mean within 5 standard errors of 0
        fit = pooled[name]["model_free"]
        assert abs(fit["mean_ps"]) < 5 * fit["sigma_ps"] / np.sqrt(fit["signal_events"]) + 5.0
    walk = pooled["em_walk"]["model_free"]["model_free_fwhm_ps"]
    assert walk == pytest.approx(2.3548 * h.NOISE_PS, rel=0.1)


@pytest.mark.slow
@pytest.mark.fr("013-FR-5")
def test_bootstrap_reproducible(necr_run, tmp_path):
    case, spec, report = necr_run
    path = h.necr_file(case, case["frames"], "again.json")
    again = necr.run_necr(necr.load_necr(path), log=lambda *a: None)
    assert again["results"] == report["results"]


@pytest.mark.slow
@pytest.mark.fr("013-FR-5", "013-FR-6")
def test_report_fields_and_gates(necr_run):
    from src.timing import ctr as core
    _, spec, report = necr_run
    assert {"frozen_inputs", "frames", "lines", "line_distances", "cuts", "method", "counts", "results",
            "paired_against_legacy", "gates", "legacy_results", "energy", "seconds", "peak_memory"} <= set(report)
    assert set(report["energy"]["frames"]) == {"frame1", "frame2"}
    assert report["energy"]["calibration"]["origin"] == "job"
    assert report["frames"][0]["splits"][0]["sha256"] and report["frames"][0]["activity_kbq_ml"] == 5.0
    gates = report["gates"]
    assert gates["frame"] == "frame1" and gates["line"] == "fitted" and gates["transfer_passed"]
    for gate, (reference, candidate) in (("transfer", ("legacy", "em")), ("acceptance", ("em", "em_walk"))):
        verdict = gates[gate]["verdict"]
        assert gates[gate]["accepted"] and verdict["accepted"]
        entry = report["results"]["frame1"]["fitted"]
        assert verdict["judged_on"] == "model_free model_free_fwhm_ps"      # spec change at T9
        assert verdict["table_ps"] == entry[candidate]["model_free"]["model_free_fwhm_ps"]
        assert verdict["improvement_ps"] > verdict["combined_uncertainty_ps"] > 0
        assert verdict["paired"]["sd_ps"] > 0 and set(gates[gate]["per_scope"]) == {"frame1", "frame2", "pooled"}
    values = gates["headline_value"]["values"]
    assert set(values) == {"uncorrected", "legacy", "em", "em_walk"}
    for name in ("legacy", "em", "em_walk"):
        entry = report["results"]["frame1"]["fitted"][name]
        assert values[name]["fwhm_ps"] == entry["model_free"]["model_free_fwhm_ps"] and values[name]["sd_ps"] > 0
        assert values[name]["nema_background_subtracted"]["fwhm_ps"] == entry["nema"]["background_subtracted"][
            "fwhm_ps"]
    paired = report["paired_against_legacy"]["pooled"]["fitted"]["em_walk"]["background_subtracted fwhm_ps"]
    assert paired["improvement_ps"] > 0 and paired["replicates"] == spec.bootstrap
    with np.load(spec.output / "necr_histograms.npz") as saved:
        mf = saved["model_free"][0, list(saved["tables"]).index("em"), 0]
        fit = core.fit_histogram(mf, saved["model_free_edges"], crossings=ctr.MODEL_FREE)
        assert fit["model_free_fwhm_ps"] == report["results"]["frame1"]["fitted"]["em"]["model_free"][
            "model_free_fwhm_ps"]
        assert saved["band"].shape[:3] == (2, 4, 2)


@pytest.mark.slow
@pytest.mark.fr("013-FR-6")
def test_transfer_failure_withholds_frozen_tables(tmp_path):
    case = _necr_case(tmp_path, em_error_ps=300.0)
    spec = necr.load_necr(h.necr_file(case, case["frames"], bootstrap=10))
    report = necr.run_necr(spec, log=lambda *a: None)
    gates = report["gates"]
    assert gates["transfer_passed"] is False and not gates["transfer"]["accepted"]
    values = gates["headline_value"]["values"]
    assert values["em"]["status"].startswith("withheld") and values["em_walk"]["status"].startswith("withheld")
    assert values["legacy"]["fwhm_ps"] > 0
    with np.load(spec.output / "necr_histograms.npz") as saved:
        assert {"em", "em_walk"} <= set(saved["tables"]) and saved["band"].sum() > 0


# --- T6: energy check (FR-7) -------------------------------------------------------------------------------------

@pytest.mark.fr("013-FR-7")
def test_photopeak_follows_a_gain_shift(tmp_path):
    from imas_skew_helpers import FULL, mu_file, side
    from src.imas.skew import encal
    probe = ImasSetup.from_files(CONFIG)
    rng = np.random.default_rng(4)
    times = sorted({ch for sm in range(48, 72) for mm in (5, 6) for ch in channels(probe, sm, mm)[0]})
    missing = set(channels(probe, 50, 5)[0])
    scale = 0.2                       # IMAS-like ENERGY sums (~102 a.u.), inside the encal histogram range
    calibration = mu_file(tmp_path / "encal.tsv", {ch: 511.0 * scale for ch in times if ch not in missing})
    setup = ImasSetup.from_files(CONFIG, calibration)
    peaks = {}
    for gain in (1.0, 1.05):
        records = []
        for _ in range(3000):
            sm, mm = int(rng.integers(48, 60)), int(rng.choice([5, 6]))
            s1, s2 = rng.normal(gain, 0.04, 2)
            records.append((side(probe, sm, mm, time={int(rng.integers(8)): 20.0},
                                 energy=[v * scale * s1 for v in FULL]),
                            side(probe, sm + 12, mm, time={int(rng.integers(8)): 20.0},
                                 energy=[v * scale * s2 for v in FULL])))
        path = write(tmp_path / f"gain_{gain}.ldat", records)
        histogram, _, _, _ = encal.accumulate(path, setup)
        peaks[gain] = necr.photopeak(histogram, setup)
    assert peaks[1.0]["status"] == "fitted" and peaks[1.0]["peak_kev"] == pytest.approx(511.0, rel=0.01)
    assert peaks[1.05]["peak_kev"] == pytest.approx(536.55, rel=0.01)
    assert peaks[1.0]["excluded_channels"] > 0 and peaks[1.0]["excluded_entries"] > 0
    assert peaks[1.0]["entries"] + peaks[1.0]["excluded_entries"] == 2 * 3000


# --- T7: diagnostics and command (FR-8) ----------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.fr("013-FR-8")
def test_plots_written(necr_run):
    _, spec, report = necr_run
    expected = {f"{kind}_{frame}.png" for kind in ("timing", "dt_vs_r") for frame in ("frame1", "frame2")}
    expected |= {"ctr_vs_activity.png", "line_fit.png", "photopeak.png"}
    assert {path.split("/")[-1] for path in report["plots"]} == expected
    assert all((spec.output / path).stat().st_size > 1000 for path in report["plots"])


@pytest.mark.slow
@pytest.mark.fr("013-FR-1", "013-FR-8")
def test_command_writes_the_report(necr_run, capsys):
    from src.imas.skew.__main__ import main
    case, _, _ = necr_run
    path = h.necr_file(case, case["frames"], "command.json", bootstrap=4)
    assert main(["necr", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "Headline value (frame1, line fitted, model-free FWHM" in printed and "acceptance: em_walk vs em" in printed
    assert json.loads((case["base"] / "out" / "command" / "necr_ctr.json").read_text(encoding="utf-8"))["plots"]
    assert main(["necr", str(path)]) == 2            # the output exists: refused, never overwritten


# --- T8: real NECR frame 7 (FR-3, FR-6, FR-7) ---------------------------------------------------------------------

FROZEN = {"IMAS/skew_IMAS.txt": "930cd3fc", "IMAS_work/job_ship/lut_em.csv": "8eadd8c7",
          "IMAS_work/job_walk/lut_em_walk-dd68b558604c.csv": "daa3b251", "IMAS_work/walk/walk3.json": "5e8ac010"}


@pytest.mark.real_data
@pytest.mark.slow
@pytest.mark.fr("013-FR-3", "013-FR-6", "013-FR-7")
def test_real_frame7_prefix(tmp_path, real_data_file):
    from src.timing.cache import _sha256
    split = real_data_file("IMAS/NECR/20241003_NECR_E0h4_120s_7_coincCompact4s_00000024.ldat")
    encal = real_data_file("IMAS_work/job_walk/skew_20251119_run7/encal/encal.tsv")   # owner 2026-10-28
    work = real_data_file("IMAS_work/job_walk/job_fingerprint.json").parent
    paths = {name: real_data_file(name) for name in FROZEN}
    sha = {name: _sha256(path) for name, path in paths.items()}
    assert all(sha[name].startswith(prefix) for name, prefix in FROZEN.items())

    def frozen(name):
        return {"path": str(paths[name]), "sha256": sha[name]}

    data = {"schema": necr.SCHEMA, "job_work": str(work), "encal": str(encal), "output": str(tmp_path / "out"),
            "luts": {"em": frozen("IMAS_work/job_ship/lut_em.csv"),
                     "em2": frozen("IMAS_work/job_walk/lut_em_walk-dd68b558604c.csv")},
            "walks": {"em2_walk3": {"skew": "em2", **frozen("IMAS_work/walk/walk3.json")}},
            "legacy": frozen("IMAS/skew_IMAS.txt"),
            "frames": [{"name": "frame7", "frame": 7, "ldat": [str(split)], "activity_kbq_ml": 5.311,
                        "activity_source": "CTR_vs_activity_NECR.xlsx"}],
            "lines": {"fitted": {"method": "lor_convergence"},
                      "image": {"p1": h.LEGACY_P1.tolist(), "p2": h.LEGACY_P2.tolist(), "source": "legacy image"}},
            "headline_line": "fitted", "gates": {"frame": "frame7", "transfer": ["legacy", "em"],
                                                 "acceptance": ["em", "em2_walk3"]},
            "bootstrap": 10, "max_records": 2_000_000, "localization_events": 1_000_000,
            "energy_check_records": 500_000}
    path = tmp_path / "necr.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    report = necr.run_necr(necr.load_necr(path), log=lambda *a: None)
    distance = report["line_distances"]["distance_mm"]["fitted - image"]
    assert distance[1] < 10.0                           # at z = 0
    pooled = report["results"]["frame7"]["fitted"]
    assert set(pooled) == {"uncorrected", "legacy", "em", "em2_walk3"}
    for name, entry in pooled.items():
        assert entry["model_free"]["status"] == "fitted"
        if name != "uncorrected":     # uncorrected (~2.3 ns) fills the +-2 ns NEMA window: no half-maximum crossing
            for kind in ("raw", "background_subtracted"):
                assert entry["nema"][kind]["status"] == "fitted" and np.isfinite(entry["nema"][kind]["fwhm_ps"])
    assert pooled["em2_walk3"]["nema"]["raw"]["fwhm_ps"] < pooled["legacy"]["nema"]["raw"]["fwhm_ps"]
    assert report["energy"]["frames"]["frame7"]["status"] == "fitted"
