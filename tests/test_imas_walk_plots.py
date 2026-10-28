"""IMAS walk diagnostics written headless from a synthetic table and report (spec 012 T8, FR-7)."""
import json

import numpy as np
import pytest

from src.imas.skew import walkplots
from src.imas.skew.__main__ import main as skew_main
from src.imas.skew.walk import WalkLut

pytestmark = pytest.mark.fr("012-FR-7")

PNG = b"\x89PNG"


def synthetic_lut():
    rng = np.random.default_rng(5)
    edges = {"walk2d": [np.linspace(0, 40, 5), np.linspace(0.3, 1, 4)], "doi": [np.linspace(1, 3, 6)],
             "kev": [np.linspace(400, 620, 6)]}
    offsets = {b: rng.normal(0, 40, tuple(len(e) - 1 for e in edges[b])) for b in edges}
    events = {b: rng.integers(300, 5000, v.shape) for b, v in offsets.items()}
    events["walk2d"][0, 0] = events["doi"][-1] = events["kev"][0] = 10       # sparse: filled from the nearest bin
    return WalkLut.from_fit(edges, offsets, events, {"fit_kind": "synthetic"}, minimum=100)


def written_lut(tmp_path):
    synthetic_lut().write(tmp_path / "walk.json")
    return tmp_path / "walk.json"


def assert_png(path):
    assert path.stat().st_size > 1000 and path.read_bytes()[:4] == PNG


def test_plot_table_written_for_a_lut_with_sparse_bins(tmp_path):
    lut = synthetic_lut()
    assert all(lut.filled[b] for b in lut.filled)
    walkplots.plot_table(lut, tmp_path / "table.png")        # no manifest table entry: no sha in the title
    assert_png(tmp_path / "table.png")
    lut.write(tmp_path / "walk.json")
    walkplots.plot_table(lut, tmp_path / "table_sha.png")
    assert_png(tmp_path / "table_sha.png")
    assert lut.sha256[:12] in walkplots._title(lut)


def test_residual_profiles_known_offsets_empty_bins_and_clamping():
    edges = {"kev": np.array([400.0, 500.0, 600.0, 700.0])}
    kev = np.array([450.0] * 5 + [550.0] * 3 + [650.0] * 2 + [100.0, 900.0, np.nan, 450.0])
    before = np.array([100.0] * 5 + [-50.0] * 3 + [10.0, 12.0] + [30.0, 40.0, 1.0, 5000.0])
    after = before - 100.0
    out = walkplots.residual_profiles({"kev": kev}, before, after, edges, core_ps=600.0)["kev"]
    # clamped: 100 keV -> first bin, 900 keV -> last bin; NaN input dropped; the 5000 ps event is outside the core
    assert out["edges"] == [400.0, 500.0, 600.0, 700.0]
    assert out["events"] == [7, 3, 3]
    assert out["before_median_ps"] == [100.0, -50.0, 12.0]       # first: 5 x 100, 30, and the 5000 excluded
    assert out["after_median_ps"] == [0.0, -150.0, -88.0]
    empty = walkplots.residual_profiles({"kev": np.array([450.0, 460.0])}, [1.0, 2.0], [0.0, 0.0], edges)["kev"]
    assert empty["before_median_ps"] == [1.5, None, None] and empty["events"] == [2, 0, 0]
    assert empty["after_median_ps"] == [0.0, None, None]
    json.dumps(out), json.dumps(empty)


def test_residual_profiles_core_applies_before_and_after_separately():
    edges = {"charge": np.array([0.0, 10.0])}
    out = walkplots.residual_profiles({"charge": np.array([1.0, 2.0, 3.0])}, [10.0, 700.0, 20.0],
                                      [5.0, 10.0, 900.0], edges, core_ps=600.0)["charge"]
    assert out["events"] == [3] and out["before_median_ps"] == [15.0] and out["after_median_ps"] == [7.5]
    assert isinstance(out["events"][0], int) and type(out["before_median_ps"][0]) is float


def test_plot_profiles_written(tmp_path):
    edges = {"charge": np.linspace(0, 40, 5), "share": np.linspace(0.3, 1, 4), "doi_ratio": np.linspace(1, 3, 5),
             "kev": np.linspace(400, 620, 5)}
    rng = np.random.default_rng(1)
    n = 3000
    values = {k: rng.uniform(e[0], e[-1], n) for k, e in edges.items()}
    before = rng.normal(0, 200, n)
    one = walkplots.residual_profiles(values, before, before * 0.5, edges)
    two = walkplots.residual_profiles({k: v[:50] for k, v in values.items()}, before[:50], before[:50], edges)
    walkplots.plot_profiles({"bar held-out": json.loads(json.dumps(one)), "22Na": two}, tmp_path / "p.png")
    assert_png(tmp_path / "p.png")


def fit(value):
    return {"fit": {"status": "fitted", "model_free_fwhm_ps": value}}


def synthetic_report():
    names = ["1cm_cFOV_34", "1cm_cFOV_35", "20cm_3d8cFOV_26"]

    def table(scale):
        return {"pooled": fit(1000.0 * scale), "1cm_cFOV_34": fit(900.0 * scale), "1cm_cFOV_35": fit(1100.0 * scale),
                "20cm_3d8cFOV_26": {"fit": {"status": "no peak", "model_free_fwhm_ps": None}}}
    assert set(table(1)) == {"pooled", *names}
    return {"results": {"em": table(1.0), "em2+walk3": table(0.9), "legacy": table(1.2)}}


def test_plot_ctr_groups_positions_and_skips_missing_tables(tmp_path):
    summary = walkplots.plot_ctr(synthetic_report(), ["em", "em2+walk3", "nope"], tmp_path / "ctr.png")
    assert_png(tmp_path / "ctr.png")
    assert list(summary["positions"]) == ["1cm_cFOV", "20cm_3d8cFOV"]
    assert summary["positions"]["1cm_cFOV"] == {"em": pytest.approx(1000.0), "em2+walk3": pytest.approx(900.0)}
    assert summary["positions"]["20cm_3d8cFOV"] == {"em": None, "em2+walk3": None}
    assert summary["pooled"] == {"em": 1000.0, "em2+walk3": pytest.approx(900.0)}
    assert summary["skipped_tables"] == ["nope"] and "nope" in summary["warnings"][0]
    json.dumps(summary)


def test_plot_ctr_with_no_known_table_still_writes(tmp_path):
    summary = walkplots.plot_ctr(synthetic_report(), ["nope"], tmp_path / "ctr.png")
    assert_png(tmp_path / "ctr.png")
    assert summary["positions"] == {}


def test_cli_end_to_end_and_refuses_overwrite(tmp_path, capsys):
    lut_path = written_lut(tmp_path)
    edges = {"charge": np.linspace(0, 40, 5), "share": np.linspace(0.3, 1, 4), "doi_ratio": np.linspace(1, 3, 5),
             "kev": np.linspace(400, 620, 5)}
    rng = np.random.default_rng(2)
    values = {k: rng.uniform(e[0], e[-1], 500) for k, e in edges.items()}
    before = rng.normal(0, 150, 500)
    profile = walkplots.residual_profiles(values, before, 0.5 * before, edges)
    (tmp_path / "p.json").write_text(json.dumps({"profiles": {"bar held-out": profile}}), encoding="utf-8")
    (tmp_path / "ctr.json").write_text(json.dumps(synthetic_report()), encoding="utf-8")
    out = tmp_path / "plots"
    argv = ["walk-plot", "--walk-lut", str(lut_path), "--profiles", str(tmp_path / "p.json"),
            "--ctr", str(tmp_path / "ctr.json"), "--tables", "em", "em2+walk3", "--out", str(out)]
    assert skew_main(argv) == 0
    printed = capsys.readouterr().out.split()
    assert [p.split("\\")[-1].split("/")[-1] for p in printed] == list(walkplots.OUTPUTS)
    for name in walkplots.OUTPUTS:
        assert_png(out / name)
    before_bytes = (out / "walk_table.png").read_bytes()
    assert skew_main(argv) == 1
    assert "output exists" in capsys.readouterr().err
    assert (out / "walk_table.png").read_bytes() == before_bytes


def test_cli_table_only_and_bad_lut(tmp_path, capsys):
    lut_path = written_lut(tmp_path)
    assert skew_main(["walk-plot", "--walk-lut", str(lut_path), "--out", str(tmp_path / "o")]) == 0
    assert [p.name for p in (tmp_path / "o").iterdir()] == ["walk_table.png"]
    capsys.readouterr()
    lut_path.write_text("{}", encoding="utf-8")
    assert skew_main(["walk-plot", "--walk-lut", str(lut_path), "--out", str(tmp_path / "o2")]) == 2
    assert "walk-plot failed" in capsys.readouterr().err and not (tmp_path / "o2").exists()
