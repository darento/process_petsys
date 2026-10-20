"""Spec 010: timing LUTs in the PETsys Manager (table checks, readiness, CLI summary, GUI text and widgets).

Synthetic fixture map and LUTs (``timing_lut_helpers``); the timing numbers themselves are spec 009's tests.
"""

import io
import json

import pytest

from helpers import REPO
from src.cornell import cli, listmode as lm
from src.cornell.inputs import load_limits
from src.petsys_manager import workflow as wf
from src.petsys_manager.contracts import Action
from src.petsys_manager.settings import MachineProfile, RunOptions, ToolCapabilities, preflight
from manager_helpers import METADATA, FixtureProbe, ListmodeFixtures
from timing_lut_helpers import write_skew_lut, write_walk_lut


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """Fixture map and listmode files (position calibration, 5 regions), a skew LUT, a walk LUT fitted with them."""
    h = ListmodeFixtures.at(tmp_path_factory.mktemp("timing") / "lm")
    descriptors, maps = h.inputs(count=200, random_slabs=False)
    skew = write_skew_lut(h.root / "skew.csv", h.mapping)
    walk = write_walk_lut(h.root / "walk.json", skew, maps["calibration"], maps["cog_limits"], maps["doi_limits"])
    compact = tuple(h.compact_twins(descriptors, count=200, random_slabs=False))
    return {"h": h, "descriptors": descriptors, "compact": compact, "maps": maps, "skew": skew, "walk": walk}


def _check(tables, skew=None, walk=None, *, calibration=None, doi=None):
    maps = tables["maps"]
    return lm.check_timing_tables(tables["h"].mapping, calibration or lm.calibration_entry(maps["calibration"]),
                                  maps["cog_limits"], doi or maps["doi_limits"], skew or tables["skew"], walk)


@pytest.mark.fr("010-FR-5")
def test_check_tables_fields(tables, tmp_path):
    h = tables["h"]
    delays, sha, applied, table, checked = _check(tables, walk=tables["walk"])
    assert table is not None and applied > 0 and checked["calibration_differs_from_fit"] is False
    foreign = write_skew_lut(tmp_path / "foreign.csv", h.mapping, channels=[2 ** 30])
    with pytest.raises(lm.TimingTableError, match="not TIME channels") as error:
        _check(tables, foreign)
    assert error.value.field == "skew_lut"
    other = write_skew_lut(tmp_path / "other.csv", h.mapping, seed=5)
    with pytest.raises(lm.TimingTableError, match="was fitted with skew LUT") as error:
        _check(tables, other, tables["walk"])
    assert error.value.field == "walk_lut"
    path = tmp_path / "doi_other.txt"
    path.write_text(tables["maps"]["doi_limits"].path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(lm.TimingTableError, match="differ from those the walk LUT was fitted with") as error:
        _check(tables, walk=tables["walk"], doi=load_limits(path, h.mapping, kind="doi"))
    assert error.value.field == "walk_lut"
    with pytest.raises(lm.TimingTableError) as error:
        _check(tables, tmp_path / "missing.csv")
    assert error.value.field == "skew_lut"


@pytest.mark.fr("010-FR-5")
def test_check_tables_layout_only(tables):
    """A pipeline's calibration does not exist before START: only its layout and region count are checked."""
    _check(tables, walk=tables["walk"], calibration={"path": None, "sha256": None, "layout": "position",
                                                     "num_regions": 5})
    for layout, regions in (("per_slab", 1), ("position", 3)):
        with pytest.raises(lm.TimingTableError, match="differs in layout") as error:
            _check(tables, walk=tables["walk"], calibration={"path": None, "sha256": None, "layout": layout,
                                                             "num_regions": regions})
        assert error.value.field == "walk_lut"


# --- T2: readiness and START (FR-5) ----------------------------------------------
def _profile(tables, root, **changes):
    h, maps = tables["h"], tables["maps"]
    for name in ("tools", "data", "encal", "reports", "lm", "private"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for tool in ("daqd", "init_system", "acquire_sipm_data", "set_bias", "convert_raw_to_coincidence"):
        (root / "tools" / tool).write_text("fixture marker", encoding="utf-8")
    for name in ("card0", "card1", "selected.ini"):
        (root / "private" / name).write_text("fixture", encoding="utf-8")
    values = dict(petsys_folder=str(root / "tools"), processing_root=str(h.root),
                  ini_file=str(root / "private/selected.ini"), yaml_file=str(h.root / "configs/processing.yaml"),
                  data_dir=str(root / "data"), calibration_dir=str(root / "encal"),
                  report_dir=str(root / "reports"), lm_dir=str(root / "lm"),
                  cards=(str(root / "private/card0"), str(root / "private/card1")),
                  capabilities=ToolCapabilities(), lm_metadata=METADATA,
                  cog_limits_file=str(maps["cog_limits"].path), doi_limits_file=str(maps["doi_limits"].path),
                  calibration_file=str(maps["calibration"].path), pair_map_file=str(maps["pairs"].path),
                  region_map_file=str(maps["regions"].path), skew_lut_file=str(tables["skew"]),
                  walk_lut_file=str(tables["walk"]))
    values.update(changes)
    return MachineProfile(**values)


def _settings(tables, root, action, options=None, **changes):
    inputs = tables["compact"] if action == Action.LISTMODE else ()
    report = preflight(_profile(tables, root, **changes), action, options, inputs, repo_root=REPO,
                       probe=FixtureProbe())
    assert report.ready, report.issues
    return report.settings


@pytest.mark.fr("010-FR-5")
def test_readiness_issues_lm_and_pipeline(tables, tmp_path):
    other = write_skew_lut(tmp_path / "other.csv", tables["h"].mapping, seed=5)
    assert wf.timing_table_issues(_settings(tables, tmp_path, Action.LISTMODE)) == ()
    assert wf.timing_table_issues(_settings(tables, tmp_path, Action.PIPELINE, RunOptions(regions=5))) == ()
    assert wf.timing_table_issues(_settings(tables, tmp_path, Action.LISTMODE, skew_lut_file=None,
                                            walk_lut_file=None)) == ()
    (issue,) = wf.timing_table_issues(_settings(tables, tmp_path, Action.LISTMODE, skew_lut_file=str(other)))
    assert issue.field == "walk_lut_file" and "was fitted with skew LUT" in issue.message
    (issue,) = wf.timing_table_issues(_settings(tables, tmp_path, Action.PIPELINE, RunOptions(regions=1)))
    assert issue.field == "walk_lut_file" and "differs in layout" in issue.message
    foreign = write_skew_lut(tmp_path / "foreign.csv", tables["h"].mapping, channels=[2 ** 30])
    (issue,) = wf.timing_table_issues(_settings(tables, tmp_path, Action.LISTMODE, skew_lut_file=str(foreign),
                                                walk_lut_file=None))
    assert issue.field == "skew_lut_file" and "not TIME channels" in issue.message


@pytest.mark.fr("010-FR-5")
def test_prepare_refuses_mismatched_tables(tables, tmp_path):
    wf.prepare(_settings(tables, tmp_path, Action.LISTMODE))
    other = write_skew_lut(tmp_path / "other.csv", tables["h"].mapping, seed=5)
    with pytest.raises(wf.WorkflowError, match="walk_lut_file: .*was fitted with skew LUT"):
        wf.prepare(_settings(tables, tmp_path, Action.LISTMODE, skew_lut_file=str(other)))
    with pytest.raises(wf.WorkflowError, match="walk_lut_file: .*differs in layout"):
        wf.prepare(_settings(tables, tmp_path, Action.PIPELINE, RunOptions(regions=1)))


# --- T3: CLI summary (FR-4) ------------------------------------------------------
def _cli_listmode(tables, out, skew=None, walk=None):
    h, maps = tables["h"], tables["maps"]
    request = {"schema_version": 1, "action": "listmode", "processing_root": str(h.root),
               "processing_config": str(h.root / "configs/processing.yaml"),
               "inputs": [{"path": str(d.path), "format": d.format.value, "population": d.population.value}
                          for d in tables["descriptors"]],
               "files": {"calibration": str(maps["calibration"].path), "calibration_sidecar": None,
                         "cog_limits": str(maps["cog_limits"].path), "doi_limits": str(maps["doi_limits"].path),
                         "pair_map": str(maps["pairs"].path), "region_map": str(maps["regions"].path),
                         "skew_lut": None if skew is None else str(skew),
                         "walk_lut": None if walk is None else str(walk)},
               "options": {"num_regions": 5, "region_boundaries": None,
                           "metadata": {name: getattr(METADATA, name) for name in METADATA.__dataclass_fields__},
                           "batch_records": 1000, "debug": False, "resume": False, "hit_limit": None,
                           "lm_seed": 3, "workers": 1, "in_place": False, "timing_uncorrected": "keep"},
               "outputs": {"directory": str(out / "job")}}
    out.mkdir()
    (out / "request.json").write_text(json.dumps(request), encoding="utf-8")
    code = cli.main(["listmode", "--request", str(out / "request.json"), "--result", str(out / "result.json")],
                    stdout=io.StringIO(), stderr=io.StringIO())
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert code == 0, result["errors"]
    return result["summary"], json.loads((out / "job" / lm.JOB_FILE).read_text(encoding="utf-8"))["job"]


@pytest.mark.fr("010-FR-4")
def test_cli_summary_timing_block(tables, tmp_path):
    summary, job = _cli_listmode(tables, tmp_path / "corrected", tables["skew"], tables["walk"])
    assert summary["timing_correction"] == job["timing_correction"]
    assert summary["timing_correction"]["walk_lut"]["calibration_differs_from_fit"] is False
    assert summary["observations"]["timing_corrected"] > 0
    assert summary["observations"]["timing_corrected"] + summary["observations"]["timing_uncorrected_kept"] \
        == summary["records_written"]
    summary, job = _cli_listmode(tables, tmp_path / "plain")
    assert summary["timing_correction"] is None and "timing_correction" not in job
    assert "timing_corrected" not in summary["observations"]


# --- T4: plan and result text (FR-3, FR-4) ---------------------------------------
@pytest.mark.fr("010-FR-3")
def test_plan_text():
    from exe_programs.petsys_manager_gui import timing_plan_text
    assert timing_plan_text("", "", "keep") == "Timing correction: none (raw dt)"
    assert timing_plan_text(None, "/x/walk.json", "drop") == \
        "Timing correction: none (raw dt); a walk LUT needs its skew LUT"
    assert timing_plan_text("/x/skew.csv", "", "keep") == \
        "Timing correction: skew LUT skew.csv; events without a correction are kept with raw dt (counted)"
    assert timing_plan_text("/x/skew.csv", "/y/walk.json", "drop") == \
        "Timing correction: skew LUT skew.csv + walk LUT walk.json; events without a correction are dropped (counted)"


@pytest.mark.fr("010-FR-4")
def test_result_text(tables, tmp_path):
    from exe_programs.petsys_manager_gui import timing_result_text
    summary, job = _cli_listmode(tables, tmp_path / "corrected", tables["skew"], tables["walk"])
    observed = summary["observations"]
    text = timing_result_text(summary)
    assert text.startswith(f"\n  Timing correction (skew + walk LUTs): {observed['timing_corrected']:,} events "
                           f"corrected, {observed['timing_uncorrected_kept']:,} without a correction kept with raw dt")
    assert job["timing_correction"]["uncorrected_events"] == "keep" and "Warning" not in text
    clamped = [name for name in lm.TIMING_OBSERVATIONS if name.startswith("walk_clamped_") and observed[name]]
    assert ("clamped" in text) == bool(clamped)
    assert timing_result_text({"timing_correction": None}) == "\n  Timing correction: none (raw dt)"
    drop = {"timing_correction": {"skew_lut": {}, "walk_lut": {"calibration_differs_from_fit": True,
                                                               "fitted_calibration_sha256": "a" * 64},
                                  "uncorrected_events": "drop"},
            "observations": {"timing_corrected": 1200, "walk_clamped_charge": 0, "walk_clamped_kev": 3},
            "rejected": {"no_timing_correction": 7}, "calibration_sha256": "b" * 64}
    assert timing_result_text(drop) == (
        "\n  Timing correction (skew + walk LUTs): 1,200 events corrected, 7 without a correction dropped"
        "\n  Walk inputs clamped to the table edges: kev 3"
        f"\n  Warning: the LM calibration ({'b' * 12}...) is not the one the walk LUT was fitted with "
        f"({'a' * 12}...); same layout, accepted")
