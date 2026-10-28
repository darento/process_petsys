"""Spec 013 T1: NEMA NU 2-2018 line-source timing resolution (``src.timing.nema``), parity with the legacy script."""
import json

import numpy as np
import pytest

from src.timing import nema
from helpers import DATA
import necr_helpers as h

GOLDEN = json.loads((DATA / "golden" / "timing" / "013_t1_nema_legacy.json").read_text(encoding="utf-8"))
pytestmark = pytest.mark.fr("013-FR-4")


def test_line_frame_matches_legacy():
    l1, l2, dt = h.line_lors()
    r, intercept, tof = nema.line_frame(l1, l2, h.LEGACY_P1, h.LEGACY_P2)
    expected = GOLDEN["geometry"]
    np.testing.assert_allclose(r, expected["r_mm"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(intercept, expected["intercept_mm"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(dt - tof, expected["dt_error_ps"], rtol=0, atol=1e-9)


def test_parallel_lor_is_nan_and_never_kept():
    l1 = np.array([[100.0, 0.0, -300.0], [0.0, -410.0, 0.0]])
    r, intercept, tof = nema.line_frame(l1, l1 + [[0.0, 0.0, 600.0], [0.0, 820.0, 0.0]], [0, 0, -1], [0, 0, 1])
    assert np.isnan([r[0], tof[0]]).all() and np.isnan(intercept[0]).all()
    assert np.isfinite([r[1], tof[1]]).all()
    assert nema.kept(np.zeros(2), r).tolist() == [False, True]


def test_batched_histograms_match_legacy():
    dt, r, _ = h.dt_r_events()
    hist = nema.NemaHistograms()
    kept = sum(hist.add((), dt[s], r[s]) for s in np.array_split(np.arange(len(dt)), [3, 900, 50_000, 50_001,
                                                                                    120_000, 199_000]))
    expected = GOLDEN["histograms"]
    assert kept == expected["kept"]
    assert hist.raw.tolist() == expected["raw"]
    assert hist.band.tolist() == expected["band"]
    np.testing.assert_allclose(nema.background_subtracted(hist.band), expected["background_subtracted"],
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(nema.T_CENTERS, expected["t_centers"])


def test_widths_match_legacy():
    expected = GOLDEN["histograms"]
    for name, hist in (("raw_widths", expected["raw"]), ("background_subtracted_widths",
                                                         expected["background_subtracted"])):
        result = nema.widths(np.array(hist, np.float64))
        assert result["status"] == nema.FITTED
        for key in ("fwhm_ps", "fwtm_ps", "peak_ps"):
            assert result[key] == pytest.approx(expected[name][key], abs=1e-9)


def test_width_failures_are_statuses():
    centers = nema.T_CENTERS
    gaussian = np.exp(-0.5 * (centers / 300.0) ** 2) * 1000
    assert nema.widths(gaussian)["status"] == nema.FITTED
    assert nema.widths(np.roll(gaussian, 50))["status"] == nema.EDGE_PEAK
    shoulder = np.where(centers < -300, 600.0, gaussian)    # never falls below half on the left
    assert nema.widths(shoulder)["status"] == nema.NO_CROSSING
    result = nema.widths(np.zeros_like(centers))             # empty: the first bin is the maximum
    assert result["status"] == nema.EDGE_PEAK and result["fwhm_ps"] is None


@pytest.mark.parametrize("slope", [0.0, 0.8])
def test_background_subtraction_of_expected_counts(slope):
    """Expected counts: a Gaussian peak plus a background linear in r. A flat background is removed exactly; a
    sloped one nearly, since the edge bins' centres sit 0.087 mm outside +-20 mm."""
    peak = np.outer(np.exp(-0.5 * (nema.R_CENTERS / 2.0) ** 2), 50 * np.exp(-0.5 * (nema.T_CENTERS / 300.0) ** 2))
    background = np.outer(1 + slope * nema.R_CENTERS / 30.0, np.full(nema.T_BINS, 3.0))
    signal = (nema.R_CENTERS >= -20) & (nema.R_CENTERS <= 20)
    residual = nema.background_subtracted(peak + background) - peak[signal].sum(axis=0)
    assert np.abs(residual).max() <= (1e-9 if slope == 0 else 2e-3) * background[signal].sum(axis=0).max()


@pytest.mark.parametrize("seed, slope", [(200, 0.0), (201, 0.8)])
def test_background_subtraction_recovers_injected_peak(seed, slope):
    dt, r, is_peak = h.dt_r_events(seed=seed, n=2_000_000, slope=slope)
    total, peak = nema.NemaHistograms(), nema.NemaHistograms()
    total.add((), dt, r)
    peak.add((), dt[is_peak], r[is_peak])
    recovered, injected = nema.background_subtracted(total.band), nema.background_subtracted(peak.band)
    assert recovered.sum() == pytest.approx(injected.sum(), rel=0.03)
    assert nema.widths(recovered)["fwhm_ps"] == pytest.approx(nema.widths(peak.raw)["fwhm_ps"], rel=0.02)
    assert nema.widths(total.raw)["fwhm_ps"] > 1.03 * nema.widths(peak.raw)["fwhm_ps"]


def test_weighted_and_indexed_histograms():
    dt, r, _ = h.dt_r_events(n=20_000)
    hist = nema.NemaHistograms((2, 3))
    weights = np.random.default_rng(1).poisson(1.0, len(dt))
    hist.add((1, 2), dt, r, weights)
    one = nema.NemaHistograms()
    one.add((), np.repeat(dt, weights), np.repeat(r, weights))
    np.testing.assert_array_equal(hist.raw[1, 2], one.raw)
    np.testing.assert_array_equal(hist.band[1, 2], one.band)
    assert not hist.raw[0].any() and not hist.band[1, :2].any()


def test_bin_index_equals_histogram2d_at_edges():
    edges_t, edges_r = nema.T_EDGES, nema.R_EDGES
    dt = np.concatenate((edges_t, edges_t[:20], np.nextafter(edges_t[1:], 0)))
    r = np.concatenate((np.clip(np.linspace(-22.0644, 22.0644, len(edges_t)), -22.0644, 22.0644),
                        edges_r[10:30], np.zeros(len(edges_t) - 1)))
    ok, flat = nema.bin_index(dt, r)
    expected = np.histogram2d(r[ok], dt[ok], bins=[nema.R_BINS, nema.T_BINS],
                              range=[[-nema.R_RANGE_MM, nema.R_RANGE_MM], [-nema.T_RANGE_PS, nema.T_RANGE_PS]])[0]
    np.testing.assert_array_equal(np.bincount(flat, minlength=expected.size).reshape(expected.shape), expected)
    np.testing.assert_array_equal(expected.sum(axis=0), np.histogram(dt[ok], bins=nema.T_BINS,
                                                                     range=(-nema.T_RANGE_PS, nema.T_RANGE_PS))[0])
