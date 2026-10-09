"""LDATInspector engine and report checks on native-layout fixtures (spec 001; spec 007 T19).

Moved from ``scripts/ldat_inspector_check.py --selftest``: each ``check`` is one test. The
fixture files and datasets are built once per system (``worlds``); the fit tests share one
seeded generator in the script's draw order (``samples``).
"""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pypdf import PdfReader

from ldat_helpers import fixture_files, write_pairs
from src.ldat_inspector.engine import (
    Selection, Settings, channel_status, fit_on_display_bins, fit_peak, fit_peak_background,
    flood_counts, load_setup, merge_results,
    pair_offset_series, process_file, process_file_reference, rate_series, uniformity,
)
from src.ldat_inspector.report import report_rows, write_report

pytestmark = pytest.mark.fr("001-FR-15")
fr = pytest.mark.fr


def pdf_text(path):
    return "\n".join(page.extract_text() for page in PdfReader(str(path)).pages)


def build_world(system, root):
    config, calibration, channels = fixture_files(root, system)
    settings = Settings(str(config), str(calibration), system, max_pairs=300)
    setup = load_setup(settings)
    first, second, broken = (root / f"{name}.ldat" for name in ("a", "b", "broken"))
    write_pairs(first, channels, 120)
    write_pairs(second, channels, 80, energy=120.0)
    write_pairs(broken, channels, 5, truncate=True)
    results = [process_file(str(path), settings, i) for i, path in enumerate((first, second, broken))]
    raw_settings = replace(settings, calibration_path="", calibrated=False)
    raw_files = [process_file(str(first), raw_settings, 0), process_file(str(second), raw_settings, 1)]
    shifted = root / "shifted.ldat"
    write_pairs(shifted, channels, 100, second_offset_ps=200_000_000_000)
    return SimpleNamespace(
        system=system, root=root, config=config, calibration=calibration, channels=channels,
        settings=settings, setup=setup, first=first, results=results,
        dataset=merge_results(settings, results, setup), raw_files=raw_files,
        raw=merge_results(raw_settings, raw_files),
        shifted=merge_results(settings, [process_file(str(shifted), settings, 3)], setup))


@pytest.fixture(scope="module")
def worlds():
    """System -> fixture world, built on first use. A short tempfile root, as in the script:
    the report wraps provenance lines at 118 characters, which pytest's tmp paths can exceed."""
    with tempfile.TemporaryDirectory() as temporary:
        built = {}

        def get(system):
            if system not in built:
                root = Path(temporary) / system
                root.mkdir()
                built[system] = build_world(system, root)
            return built[system]

        yield get


@pytest.fixture(scope="module", params=["IMAS", "CORNELL"])
def world(request, worlds):
    return worlds(request.param)


@pytest.fixture(scope="module")
def cornell(worlds):
    return worlds("CORNELL")


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(20260924)
    known = rng.normal(511, 20, 30_000)
    with_continuum = np.concatenate((rng.normal(511, 22, 18_000), rng.uniform(350, 700, 6000)))
    flat = rng.uniform(350, 700, 30000)
    return SimpleNamespace(known=known, with_continuum=with_continuum, flat=flat)


# --- ingest ------------------------------------------------------------------

@fr("001-FR-5")
def test_mapped_two_channels(world):
    assert all(p["t"] in world.setup.channel_modules for p in world.channels)


@fr("001-FR-2", "001-FR-3")
def test_valid_pair_and_detector_counts(world):
    assert [(r.pairs_read, r.pairs_accepted) for r in world.results[:2]] == [(120, 120), (80, 80)]


@fr("001-FR-4")
def test_ingest_per_channel_energy_cut_applied(world):
    threshold_result = process_file(str(world.first), replace(world.settings, min_channel_energy=110), 0)
    assert threshold_result.pairs_read == 120 and threshold_result.pairs_accepted == 0


@fr("001-FR-3")
def test_prefix_is_explicitly_marked(world):
    prefix_result = process_file(str(world.first), replace(world.settings, max_pairs=5), 0)
    assert prefix_result.pairs_read == 5 and prefix_result.prefix_limited


@fr("001-FR-5")
def test_cornell_unresolved_slab_has_a_named_rejection_without_partial_sides(cornell):
    # The patch reaches the reference reader only; the fast reader's label is compared
    # with it in the scale checks (non-adjacent fixture).
    with patch("src.ldat_inspector.engine.get_slab_cornell", return_value=(None, 2, None)):
        unresolved = process_file_reference(str(cornell.first), replace(cornell.settings, max_pairs=5), 0)
    assert (unresolved.pairs_accepted == 0 and unresolved.errors == {"unresolved Cornell slab": 5}
            and not unresolved.modules)


@fr("001-FR-2", "001-FR-13")
def test_corrupt_file_has_no_partial_measurements(world):
    assert not world.results[2].success and not world.results[2].modules


@fr("001-FR-13")
def test_empty_file_is_a_named_failure(world):
    empty = world.root / "empty.ldat"
    empty.write_bytes(b"")
    empty_result = process_file(str(empty), world.settings, 4)
    assert not empty_result.success and "Empty LDAT" in empty_result.error


# --- raw (uncalibrated) mode -------------------------------------------------

@fr("001-FR-22", "001-FR-23")
def test_raw_mode_opens_without_calibration_and_retains_native_energy(world):
    assert (all(f.pairs_accepted == expected for f, expected in zip(world.raw_files, (120, 80)))
            and all(np.allclose(data.energy[data.file_index == 0], 100)
                    and np.allclose(data.energy[data.file_index == 1], 120)
                    for data in world.raw.modules.values()))


@fr("001-FR-4", "001-FR-22")
def test_raw_paired_adc_cut_selects_the_second_file_only(world):
    assert sum(Selection(110, 130).mask(data).sum() for data in world.raw.modules.values()) == 160


@fr("001-FR-8", "001-FR-22")
def test_no_fabricated_kev_fit_without_calibration(world):
    assert all(row["fit"]["mu"] is None and "raw a.u." in row["fit"]["status"]
               for row in uniformity(world.raw, Selection(0, 200)) if row["events"] > 0)


@fr("001-FR-12", "001-FR-22")
def test_raw_report_labels_units_and_unavailable_fit(world):
    raw_report = world.root / "raw.pdf"
    write_report(raw_report, world.raw, Selection(110, 130), sm=0)
    raw_text = pdf_text(raw_report)
    assert ("110..130 a.u." in raw_text and "no keV calibration" in raw_text
            and "OFF (raw PETsys a.u.)" in raw_text)


@fr("001-FR-22")
def test_calibrated_mode_still_requires_a_calibration(world):
    with pytest.raises(FileNotFoundError):
        load_setup(replace(world.settings, calibration_path=""))


# --- merged dataset ----------------------------------------------------------

@fr("001-FR-2")
def test_failed_file_excluded(world):
    assert sum(map(len, world.dataset.modules.values())) == 400


@fr("001-FR-2")
def test_file_provenance(world):
    assert all(set(data.file_index) == {0, 1} for data in world.dataset.modules.values())


@fr("001-FR-4")
def test_pair_energy_filter_reversible(world):
    modules = world.dataset.modules.values()
    assert (sum(Selection(400, 650).mask(d).sum() for d in modules) == 400
            and sum(Selection(550, 650).mask(d).sum() for d in modules) == 160)


@fr("001-FR-7")
def test_mapped_time_energy_occupancy(world):
    status = channel_status(world.dataset, 0)
    assert world.channels[0]["t"] in status["active_time"] and world.channels[0]["e"] in status["active_energy"]


@fr("001-FR-7")
def test_other_mapped_channels_remain_unobserved(world):
    status = channel_status(world.dataset, 0)
    assert status["unobserved_time"] and status["unobserved_energy"]


# --- timing ------------------------------------------------------------------

@fr("001-FR-11")
def test_per_file_timing_in_seconds(world):
    rate = rate_series(world.dataset, 0, 0)
    assert rate is not None and np.isclose(rate[0][-1], 1.19)


@fr("001-FR-11")
def test_modules_share_file_time_axis(world):
    assert np.array_equal(rate_series(world.dataset, 0, 0)[0], rate_series(world.dataset, 1, 0)[0])


@fr("001-FR-11")
def test_timing_uses_common_acquisition_origin(world):
    early = rate_series(world.shifted, 0, 3)
    late = rate_series(world.shifted, 1, 3)
    assert (early is not None and np.array_equal(early[0], late[0]) and np.count_nonzero(late[1][:5]) == 0
            and np.count_nonzero(early[1][:5]) > 0)


@fr("001-FR-11")
def test_pair_time_offsets_are_signed_not_clock_drift(world):
    early_offset = pair_offset_series(world.shifted, 0, 3)
    late_offset = pair_offset_series(world.shifted, 1, 3)
    assert (early_offset is not None and np.isclose(np.nanmedian(early_offset[1]), -200_000_000)
            and np.isclose(np.nanmedian(late_offset[1]), +200_000_000))


@fr("001-FR-11")
def test_unsupported_file_timing_omitted(world):
    assert rate_series(world.dataset, 0, 2) is None


# --- fits and reports --------------------------------------------------------

@fr("001-FR-8")
def test_sparse_fit_explicitly_unavailable(world):
    assert fit_peak(world.dataset.modules[0].energy[:20])["mu"] is None


@fr("001-FR-9")
def test_uniformity_labels_sparse_results(world):
    assert all(row["result"] == "UNAVAILABLE" for row in uniformity(world.dataset, Selection()))


@fr("001-FR-4", "001-FR-12")
def test_report_uses_same_paired_energy_mask(world):
    assert sum(row["selected"] for row in report_rows(world.dataset, Selection(550, 650))) == 160


@fr("001-FR-12")
def test_module_pdf_pages_and_provenance(world):
    pdf_path = world.root / "module.pdf"
    write_report(pdf_path, world.dataset, Selection(550, 650), sm=0)
    reader = PdfReader(str(pdf_path))
    report_text = "\n".join(page.extract_text() for page in reader.pages)
    assert (len(reader.pages) == 5 and str(world.config) in report_text and str(world.calibration) in report_text
            and "No singles" in report_text and "SuperModule 0" in report_text
            and "Observed timestamp spans per file" in report_text)


@pytest.mark.slow  # 30 SuperModules in the January map: 6 s
@fr("001-FR-12")
def test_cornell_full_report_has_every_mapped_module_detail(cornell):
    rows = report_rows(cornell.dataset, Selection(550, 650))
    whole = cornell.root / "whole.pdf"
    write_report(whole, cornell.dataset, Selection())
    # provenance, summary table and findings page (30 SMs fit on one each), then two pages per SM
    assert len(PdfReader(str(whole)).pages) == 3 + 2 * len(rows)


# --- photopeak fits on seeded samples ----------------------------------------

@fr("001-FR-8")
def test_known_width_fit_measured_in_kev(samples):
    fit = fit_peak(samples.known)
    assert fit["status"] == "FIT" and abs(fit["mu"] - 511) < 3 and abs(fit["resolution"] - 9.2) < 1.5


@fr("001-FR-10")
def test_continuum_fit_locates_511_kev(samples):
    background_fit = fit_peak_background(samples.with_continuum)
    assert background_fit["status"] == "FIT" and abs(background_fit["mu"] - 511) < 3


@fr("001-FR-10")
def test_continuum_estimate_remains_distinct_from_total(samples):
    background_fit = fit_peak_background(samples.with_continuum)
    assert background_fit["background"].sum() > 0 and background_fit["gaussian"].sum() > 0


@fr("001-FR-16", "001-FR-20")
def test_gaussian_and_background_are_distinct_display_count_components(samples):
    display = fit_on_display_bins(fit_peak_background(samples.with_continuum), np.linspace(0, 1200, 161))
    assert (display is not None and np.allclose(display["gaussian"] + display["background"], display["total"])
            and display["gaussian"].max() > display["background"].max())


@fr("001-FR-21")
def test_flood_zeros_are_masked_minimum_positive_bins_are_coloured():
    flood, _, _ = flood_counts([1, 1, 50], [1, 1, 50], 10)
    assert int(flood.mask.sum()) == 98 and flood[0, 0] == 2 and flood[4, 4] == 1


@fr("001-FR-8", "001-FR-10")
def test_flat_continuum_refused(samples):
    assert fit_peak_background(samples.flat)["status"] != "FIT"


@fr("001-FR-10")
def test_invalid_experimental_window_refused(samples):
    assert fit_peak_background(samples.with_continuum, search=(200, 300))["status"] != "FIT"
