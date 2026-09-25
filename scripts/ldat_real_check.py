"""Bounded real-data inspection of the six-file Cornell acquisition.

Example:
  python scripts/ldat_real_check.py --directory .../full_system \
      --config configs/cornell_full_system.yaml --calibration encal_files/...encal

Reads only a configured prefix of each file and never modifies acquisition data.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pypdf import PdfReader

from src.ldat_inspector import Selection, Settings, merge_results, process_file, uniformity
from src.ldat_report import write_report


def run(args):
    folder = Path(args.directory)
    paths = sorted(folder.glob("*coincCompact11s_0000000[3-8].ldat"))
    if len(paths) != 6:
        raise ValueError(f"Expected separate 03–08 Cornell Compact11s files; found {len(paths)}")
    settings = Settings(str(Path(args.config).resolve()),
                        str(Path(args.calibration).resolve()),
                        "CORNELL", args.pairs, 4, 0.2)
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process_file, str(path), settings, index): index
                   for index, path in enumerate(paths)}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: result.index)
    for result in results:
        print(f"{Path(result.path).name}: {result.pairs_accepted:,}/{result.pairs_read:,} "
              f"pairs accepted, prefix={result.prefix_limited}, "
              f"rejections={dict(result.errors)}, error={result.error!r}")
        assert result.success and result.pairs_read == args.pairs and result.pairs_accepted > 0
        assert "KeyError" not in result.errors and result.errors["unresolved Cornell slab"] > 0
    dataset = merge_results(settings, results)
    assert len(dataset.modules) == 30
    selection = Selection(357, 665, 0, 15, 0, 102, 0, 102)
    rows = uniformity(dataset, selection)
    fits = [row for row in rows if row["result"] != "UNAVAILABLE"]
    peaks = np.array([row["fit"]["mu"] for row in fits])
    resolutions = np.array([row["fit"]["resolution"] for row in fits])
    print(f"Full-system sample: {sum(result.pairs_accepted for result in results):,} "
          f"pairs; {sum(map(len, dataset.modules.values())):,} detector sides; "
          f"{len(fits)}/{len(rows)} supported fits")
    print(f"Centroid range/median: {peaks.min():.1f}/{np.median(peaks):.1f}/{peaks.max():.1f} keV; "
          f"resolution range/median: {resolutions.min():.1f}/{np.median(resolutions):.1f}/{resolutions.max():.1f}%")
    print("Uniformity:", [(row["sm"], row["result"],
                          round(row["fit"]["mu"], 1) if row["fit"]["mu"] else None)
                          for row in rows])
    assert len(fits) >= 25 and np.all(np.isfinite(peaks))
    with tempfile.TemporaryDirectory() as name:
        pdf = Path(name) / "Cornell_SM0_sample.pdf"
        write_report(pdf, dataset, selection, sm=0)
        reader = PdfReader(str(pdf))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert (len(reader.pages) == 3 and "SuperModule 0" in text and
                "Energy calibration:" in text and "legacy random neighbour" in text), (
            f"SM0 PDF pages={len(reader.pages)}; scope/provenance present="
            f"{('SuperModule 0' in text, 'Energy calibration:' in text, 'legacy random neighbour' in text)}")
        print("Sample SM0 PDF: 3 pages; provenance and fit text extracted")
    if args.preview:
        from exe_programs.ldat_inspector_gui import LDATWorkbench
        app = LDATWorkbench()
        app.withdraw()
        try:
            app.dataset = dataset
            app.energy_low.set("357")
            app.energy_high.set("665")
            app._update_after_processing()
            app.module_var.set("SM 1")
            app.experimental.set(True)
            app._refresh_selected()
            app.fig.savefig(args.preview, dpi=130)
            print(f"Preview: {args.preview}")
        finally:
            app.destroy()
    print("PASS: six-file Cornell prefix, fits, module PDF and plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--pairs", type=int, default=80_000)
    parser.add_argument("--preview", help="Optional output PNG path outside the repo")
    run(parser.parse_args())
