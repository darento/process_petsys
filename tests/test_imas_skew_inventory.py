"""Raw IMAS inventory per TIME channel, SM and minimodule (spec 011 T3)."""
import json

import pytest

from imas_skew_helpers import CONFIG, channels, side, write
from src.imas.skew import inventory
from src.imas.skew.setup import ImasSetup

pytestmark = pytest.mark.fr("011-FR-6")


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG)


def test_counts_unobserved_and_legacy_comparison(tmp_path, setup):
    records = [(side(setup, 0, 5), side(setup, 60, 6))] * 3 + [(side(setup, 1, 0, time={7: 9.0}), [])]
    paths = [write(tmp_path / "a.ldat", records[:2]), write(tmp_path / "b.ldat", records[2:])]
    t0, _ = channels(setup, 0, 5)
    t60, _ = channels(setup, 60, 6)
    t1, _ = channels(setup, 1, 0)
    unobserved_example = channels(setup, 2, 0)[0][0]
    legacy = tmp_path / "skew.txt"
    legacy.write_text(f"{t0[3]} 12.5\n{unobserved_example} 9999\n{t1[7]} 9999\n", encoding="utf-8")
    out = tmp_path / "inv"
    assert inventory.main([str(p) for p in paths] + ["--config", str(CONFIG), "--out", str(out),
                                                      "--legacy-skew", str(legacy)]) == 0
    report = json.loads((out / "inventory.json").read_text(encoding="utf-8"))
    assert report["total_records"] == 4 and [s["physical_coincidences"] for s in report["splits"]] == [2, 2]
    summary = report["summary"]
    observed = {t0[3], t0[4], t60[3], t60[4], t1[7]}
    assert summary["observed"] == len(observed) and summary["unobserved"] == 15_360 - len(observed)
    assert unobserved_example in summary["unobserved_channels"]
    assert not observed & set(summary["unobserved_channels"])
    assert summary["hits_per_sm"]["0"] == 6 and summary["hits_per_sm"]["60"] == 6 and summary["hits_per_sm"]["1"] == 1
    assert [0, 5] not in summary["dark_minimodules"] and [2, 0] in summary["dark_minimodules"]
    assert 0 not in summary["dark_sms"] and 2 in summary["dark_sms"]
    comparison = summary["legacy_comparison"]
    assert comparison["legacy_no_delay"] == 2 and comparison["unobserved_and_legacy_no_delay"] == 1
    assert comparison["legacy_no_delay_but_observed"] == 1
    rows = (out / "time_channel_hits.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "channel,sm,minimodule,slab,hits" and len(rows) == 15_361
    assert f"{t0[3]},0,5,3,3" in rows
    with pytest.raises(SystemExit):
        inventory.main([str(paths[0]), "--config", str(CONFIG), "--out", str(out)])


@pytest.mark.real_data
@pytest.mark.slow
def test_real_split_prefix(tmp_path, real_data_file):
    split = real_data_file("IMAS/20251119_Skew_400s_0_coinCompact_00000008.ldat")
    legacy = real_data_file("IMAS/skew_IMAS.txt")
    out = tmp_path / "inv"
    inventory.main([str(split), "--max-records", "2000000", "--legacy-skew", str(legacy), "--out", str(out),
                    "--config", str(CONFIG)])
    summary = json.loads((out / "inventory.json").read_text(encoding="utf-8"))["summary"]
    # every TIME channel without hits is a legacy no-delay channel; unobserved ~ 4-5 % of TIME channels
    assert summary["legacy_comparison"]["unobserved_only"] == []
    assert 0.03 < summary["unobserved"] / summary["time_channels"] < 0.06
