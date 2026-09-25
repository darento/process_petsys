"""Deterministic, hardware-free LDATInspector engine checks.

Run: python scripts/ldat_inspector_check.py --selftest (from repo root).
"""

import argparse
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from src.ldat_inspector import (
    Selection, Settings, channel_status, fit_on_display_bins, fit_peak, fit_peak_background,
    flood_counts, load_setup, merge_results,
    pair_offset_series, process_file, rate_series, uniformity,
)
from src.mapping_generator import ChannelType
from src.ldat_report import report_rows, write_report
from pypdf import PdfReader


def fixture_files(root: Path, system: str):
    repo = Path(__file__).resolve().parent.parent
    original = repo / "configs" / ("imas_1DAQ.yaml" if system == "IMAS" else "cornell_1cassettes.yaml")
    config = yaml.safe_load(original.read_text(encoding="utf-8"))
    config["map_file"] = str(repo / config["map_file"])
    config_file = root / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    # Calibration is built from real mapped channel IDs, not assumed IDs.
    from src.mapping_generator import map_factory
    _, modules, types, _ = map_factory(config["map_file"])
    groups = {}
    for ch, (sm, mm) in modules.items():
        if sm not in (0, 1):
            continue
        group = groups.setdefault(sm, {}).setdefault(mm, {})
        if ChannelType.TIME in types[ch]:
            group.setdefault("t", ch)
        if ChannelType.ENERGY in types[ch]:
            group.setdefault("e", ch)
    selected = []
    for sm in (0, 1):
        mm = next(mm for mm, channels in groups[sm].items() if "t" in channels and "e" in channels)
        selected.append(groups[sm][mm])
    calibration = root / "calibration.txt"
    if system == "IMAS":
        calibration.write_text("ID\tmu\n" + "".join(f"{part['t']}\t100\n" for part in selected), encoding="utf-8")
    else:
        calibration.write_text("ID(t_ch, slab)\tmu\n" + "".join(
            f"({part['t']}, {slab})\t100\n" for part in selected for slab in range(16)), encoding="utf-8")
    return config_file, calibration, selected


def write_pairs(path, channels, count, *, energy=100.0, truncate=False, second_offset_ps=0):
    with open(path, "wb") as handle:
        for i in range(count):
            handle.write(struct.pack("2B", 2, 2))
            for side_index, part in enumerate(channels):
                for ch in (part["t"], part["e"]):
                    handle.write(struct.pack("qfi", 1_000_000_000_000 + i * 10_000_000_000
                                             + (second_offset_ps if side_index else 0), energy, ch))
        if truncate:
            handle.write(b"\x02\x02\x00")


def selftest():
    checks = 0

    def check(label, condition):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)
        print(f"[ok] {label}")

    for system in ("IMAS", "CORNELL"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, calibration, channels = fixture_files(root, system)
            settings = Settings(str(config), str(calibration), system, max_pairs=300)
            setup = load_setup(settings)
            check(f"{system} mapped two channels", all(p["t"] in setup.channel_modules for p in channels))
            first, second, broken = (root / f"{name}.ldat" for name in ("a", "b", "broken"))
            write_pairs(first, channels, 120)
            write_pairs(second, channels, 80, energy=120.0)
            write_pairs(broken, channels, 5, truncate=True)
            results = [process_file(str(path), settings, i) for i, path in enumerate((first, second, broken))]
            check(f"{system} valid pair and detector counts", [(r.pairs_read, r.pairs_accepted) for r in results[:2]] == [(120, 120), (80, 80)])
            threshold_result = process_file(str(first), replace(settings, min_channel_energy=110), 0)
            check(f"{system} ingest per-channel energy cut applied", threshold_result.pairs_read == 120 and threshold_result.pairs_accepted == 0)
            prefix_result = process_file(str(first), replace(settings, max_pairs=5), 0)
            check(f"{system} prefix is explicitly marked", prefix_result.pairs_read == 5 and prefix_result.prefix_limited)
            if system == "CORNELL":
                with patch("src.ldat_inspector.get_slab_cornell", return_value=(None, 2, None)):
                    unresolved = process_file(str(first), replace(settings, max_pairs=5), 0)
                check("CORNELL unresolved slab has a named rejection without partial sides",
                      unresolved.pairs_accepted == 0 and unresolved.errors == {"unresolved Cornell slab": 5}
                      and not unresolved.modules)
            check(f"{system} corrupt file has no partial measurements", not results[2].success and not results[2].modules)
            empty = root / "empty.ldat"
            empty.write_bytes(b"")
            empty_result = process_file(str(empty), settings, 4)
            check(f"{system} empty file is a named failure", not empty_result.success and "Empty LDAT" in empty_result.error)
            dataset = merge_results(settings, results, setup)
            raw_settings = replace(settings, calibration_path="", calibrated=False)
            raw_files = [process_file(str(first), raw_settings, 0),
                         process_file(str(second), raw_settings, 1)]
            raw = merge_results(raw_settings, raw_files)
            check(f"{system} raw mode opens without calibration and retains native energy",
                  all(f.pairs_accepted == expected for f, expected in zip(raw_files, (120, 80)))
                  and all(np.allclose(data.energy[data.file_index == 0], 100)
                          and np.allclose(data.energy[data.file_index == 1], 120)
                          for data in raw.modules.values()))
            check(f"{system} raw paired ADC cut selects the second file only",
                  sum(Selection(110, 130).mask(data).sum() for data in raw.modules.values()) == 160)
            check(f"{system} no fabricated keV fit without calibration",
                  all(row["fit"]["mu"] is None and "raw a.u." in row["fit"]["status"]
                      for row in uniformity(raw, Selection(0, 200)) if row["events"] > 0))
            raw_report = root / "raw.pdf"
            write_report(raw_report, raw, Selection(110, 130), sm=0)
            raw_text = "\n".join(page.extract_text() for page in PdfReader(str(raw_report)).pages)
            check(f"{system} raw report labels units and unavailable fit",
                  "110..130 a.u." in raw_text and "no keV calibration" in raw_text
                  and "OFF (raw PETsys a.u.)" in raw_text)
            try:
                load_setup(replace(settings, calibration_path=""))
            except FileNotFoundError:
                check(f"{system} calibrated mode still requires a calibration", True)
            else:
                check(f"{system} calibrated mode still requires a calibration", False)
            check(f"{system} failed file excluded", sum(map(len, dataset.modules.values())) == 400)
            check(f"{system} file provenance", all(set(data.file_index) == {0, 1} for data in dataset.modules.values()))
            check(f"{system} pair energy filter reversible", sum(Selection(400, 650).mask(d).sum() for d in dataset.modules.values()) == 400 and sum(Selection(550, 650).mask(d).sum() for d in dataset.modules.values()) == 160)
            status = channel_status(dataset, 0)
            check(f"{system} mapped time/energy occupancy", channels[0]["t"] in status["active_time"] and channels[0]["e"] in status["active_energy"])
            check(f"{system} other mapped channels remain unobserved", status["unobserved_time"] and status["unobserved_energy"])
            check(f"{system} per-file timing in seconds", (lambda rate: rate is not None and np.isclose(rate[0][-1], 1.19))(rate_series(dataset, 0, 0)))
            check(f"{system} modules share file time axis", np.array_equal(rate_series(dataset, 0, 0)[0], rate_series(dataset, 1, 0)[0]))
            shifted = root / "shifted.ldat"
            write_pairs(shifted, channels, 100, second_offset_ps=200_000_000_000)
            shifted_dataset = merge_results(settings, [process_file(str(shifted), settings, 3)], setup)
            early = rate_series(shifted_dataset, 0, 3)
            late = rate_series(shifted_dataset, 1, 3)
            check(f"{system} timing uses common acquisition origin", early is not None and
                  np.array_equal(early[0], late[0]) and np.count_nonzero(late[1][:5]) == 0 and
                  np.count_nonzero(early[1][:5]) > 0)
            early_offset = pair_offset_series(shifted_dataset, 0, 3)
            late_offset = pair_offset_series(shifted_dataset, 1, 3)
            check(f"{system} pair time offsets are signed, not clock drift", early_offset is not None and
                  np.isclose(np.nanmedian(early_offset[1]), -200_000_000) and
                  np.isclose(np.nanmedian(late_offset[1]), +200_000_000))
            check(f"{system} unsupported file timing omitted", rate_series(dataset, 0, 2) is None)
            check(f"{system} sparse fit explicitly unavailable", fit_peak(dataset.modules[0].energy[:20])["mu"] is None)
            check(f"{system} uniformity labels sparse results", all(row["result"] == "UNAVAILABLE" for row in uniformity(dataset, Selection())))
            rows = report_rows(dataset, Selection(550, 650))
            check(f"{system} report uses same paired-energy mask", sum(row["selected"] for row in rows) == 160)
            pdf_path = root / "module.pdf"
            write_report(pdf_path, dataset, Selection(550, 650), sm=0)
            reader = PdfReader(str(pdf_path))
            report_text = "\n".join(page.extract_text() for page in reader.pages)
            check(f"{system} module PDF pages and provenance", len(reader.pages) == 3 and
                  str(config) in report_text and str(calibration) in report_text and
                  "No singles" in report_text and "SuperModule 0" in report_text and
                  "Observed timestamp spans per file" in report_text)
            if system == "CORNELL":
                whole = root / "whole.pdf"
                write_report(whole, dataset, Selection())
                reader = PdfReader(str(whole))
                check("Cornell full report has every mapped module detail", len(reader.pages) == 2 + len(rows))
    rng = np.random.default_rng(20260924)
    fit = fit_peak(rng.normal(511, 20, 30_000))
    check("known-width fit measured in keV", fit["status"] == "FIT" and abs(fit["mu"] - 511) < 3 and abs(fit["resolution"] - 9.2) < 1.5)
    with_continuum = np.concatenate((rng.normal(511, 22, 18_000), rng.uniform(350, 700, 6000)))
    background_fit = fit_peak_background(with_continuum)
    check("continuum fit locates 511 keV", background_fit["status"] == "FIT" and abs(background_fit["mu"] - 511) < 3)
    check("continuum estimate remains distinct from total", background_fit["background"].sum() > 0 and background_fit["gaussian"].sum() > 0)
    display = fit_on_display_bins(background_fit, np.linspace(0, 1200, 161))
    check("Gaussian and background are distinct display-count components",
          display is not None and np.allclose(display["gaussian"] + display["background"], display["total"])
          and display["gaussian"].max() > display["background"].max())
    flood, _, _ = flood_counts([1, 1, 50], [1, 1, 50], 10)
    check("flood zeros are masked; minimum-positive bins are coloured",
          int(flood.mask.sum()) == 98 and flood[0, 0] == 2 and flood[4, 4] == 1)
    check("flat continuum refused", fit_peak_background(rng.uniform(350, 700, 30000))["status"] != "FIT")
    check("invalid experimental window refused", fit_peak_background(with_continuum, search=(200, 300))["status"] != "FIT")
    print(f"PASS: {checks}/{checks} checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    if parser.parse_args().selftest:
        selftest()
    else:
        parser.print_help()
