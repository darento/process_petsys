"""One Cornell skew command: options, restart and export contract (spec 008 T10, FR-15)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from cornell_timing_helpers import ROOT, job_file, runs  # noqa: F401  (synthetic two-run job fixture)
from src.cornell.skew import cache as tc
from src.cornell.skew import dataset as td
from src.cornell.skew import plots as tpl
from src.cornell.skew.__main__ import COMMANDS, main
from src.cornell.skew.lut import CanonicalLut, read_native_table

pytestmark = pytest.mark.fr("008-FR-15")


def test_options_restart_and_export_contract(runs, tmp_path, capsys, monkeypatch):
    # options: every command listed; no command is a usage error; an unknown one is refused
    assert main(["-h"]) == 0
    listed = capsys.readouterr().out
    assert all(f"  {name} " in listed for name in COMMANDS)
    assert set(COMMANDS) == {"run", "validate", "plot", "export", "conversion", "roundtrip", "inventory", "benchmark"}
    assert main([]) == 2 and "usage" in capsys.readouterr().err
    assert main(["calibrate"]) == 2 and "unknown command 'calibrate'" in capsys.readouterr().err
    for name in COMMANDS:  # each command's own options, under the one program name
        with pytest.raises(SystemExit) as done:
            main([name, "-h"])
        assert done.value.code == 0 and f"python -m src.cornell.skew {name}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as done:
        main(["run"])
    assert done.value.code == 2 and "job" in capsys.readouterr().err

    # run: a job file to a LUT, report and plots
    base, items = runs
    job = job_file(tmp_path / "job.json", base, items)
    assert main(["run", str(job)]) == 0
    out = capsys.readouterr().out  # progress lines, then the summary JSON
    printed = json.loads(out[out.index("{\n"):])
    lut = tmp_path / "out" / "lut.csv"
    assert printed["status"] == "succeeded" and printed["lut"]["path"] == str(lut)
    assert CanonicalLut.read(lut).manifest["table"]["sha256"] == printed["lut"]["sha256"]

    # restart: the same output is refused untouched; a new output reuses every cache without reading LDAT
    before = lut.read_bytes()
    assert main(["run", str(job)]) == 2 and "output LUT exists" in capsys.readouterr().err
    assert lut.read_bytes() == before

    def no_reads(*args, **kwargs):
        raise AssertionError("LDAT read on restart")
    monkeypatch.setattr(td, "scan_file", no_reads)
    monkeypatch.setattr(tc.ingest, "LdatStream", no_reads)
    again = job_file(tmp_path / "job.json", base, items, lut="out/lut2.csv")
    assert main(["run", str(again)]) == 0
    capsys.readouterr()
    first, second = CanonicalLut.read(lut).rows, CanonicalLut.read(tmp_path / "out" / "lut2.csv").rows
    for name in ("time_channel", "delay_ps", "sigma_ps", "status", "apply"):
        assert np.array_equal(first[name], second[name], equal_nan=name in ("delay_ps", "sigma_ps")), name

    # export: native table (+ activated INI) once; existing outputs are never overwritten
    (tmp_path / "config.ini").write_text("[main]\n#time_offset_calibration_table = %CDIR%/x.tsv\n")
    table, ini = tmp_path / "native" / "time_offset.tsv", tmp_path / "native" / "config_time_offset.ini"
    table.parent.mkdir()
    assert main(["export", str(lut), str(table), "--ini", str(tmp_path / "config.ini"), "--out-ini", str(ini)]) == 0
    exported = json.loads(capsys.readouterr().out)
    offsets, skipped, duplicates = read_native_table(table)
    applied = first[first["apply"]]
    assert exported["table"]["rows"] == len(offsets) == len(applied) and not skipped and not duplicates
    assert all(abs(offsets[int(r["time_channel"])] - float(r["delay_ps"])) <= 0.01 for r in applied)
    assert f"time_offset_calibration_table = {table.resolve().as_posix()}" in ini.read_text()
    kept = table.read_bytes()
    assert main(["export", str(lut), str(table)]) == 2 and "exists" in capsys.readouterr().err
    assert table.read_bytes() == kept
    with pytest.raises(SystemExit) as done:
        main(["export", str(lut), str(tmp_path / "x.tsv"), "--ini", str(tmp_path / "config.ini")])
    assert done.value.code == 2 and "--out-ini" in capsys.readouterr().err
    assert not (tmp_path / "x.tsv").exists()
    # plots: written by the job, never overwritten by the plot command
    assert all(p.exists() for p in tpl.plot_paths(lut))
    assert main(["plot", str(lut)]) == 2 and "exist" in capsys.readouterr().err


@pytest.mark.fr("008-FR-15")
def test_module_entry_point():
    """``python -m src.cornell.skew`` is wired; the old per-module commands are gone (clean break, T10)."""
    done = subprocess.run([sys.executable, "-m", "src.cornell.skew", "-h"], cwd=ROOT, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0 and "commands:" in done.stdout
    assert not list((ROOT / "src" / "cornell").glob("timing_*.py"))
