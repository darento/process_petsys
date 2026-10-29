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
    np.testing.assert_allclose(nema.background_subtracted(hist.band, edges=nema.LEGACY),
                               expected["background_subtracted"], rtol=1e-9, atol=1e-9)
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


@pytest.mark.parametrize("r_rebin", [1, nema.NEMA_R_REBIN])
@pytest.mark.parametrize("edges", [nema.NEMA, nema.LEGACY])
@pytest.mark.parametrize("slope", [0.0, 0.8])
def test_background_subtraction_of_expected_counts(slope, edges, r_rebin):
    """Expected counts: a Gaussian peak plus a background linear in r. The NEMA edges (interpolated at +-20 mm,
    section 8.4.2a) remove any linear background exactly, on the native or summed r bins; the legacy nearest bins
    only a flat one exactly, since their centres sit outside +-20 mm."""
    peak = np.outer(np.exp(-0.5 * (nema.R_CENTERS / 2.0) ** 2), 50 * np.exp(-0.5 * (nema.T_CENTERS / 300.0) ** 2))
    background = np.outer(1 + slope * nema.R_CENTERS / 30.0, np.full(nema.T_BINS, 3.0))
    coarse_peak, centers = nema.rebinned(peak, r_rebin)
    coarse_background, _ = nema.rebinned(background, r_rebin)
    signal = (centers >= -20) & (centers <= 20)
    residual = (nema.background_subtracted(peak + background, edges=edges, r_rebin=r_rebin)
                - coarse_peak[signal].sum(axis=0))
    exact = slope == 0 or edges == nema.NEMA
    assert np.abs(residual).max() <= (1e-9 if exact else 2e-3 * r_rebin) * coarse_background[signal].sum(axis=0).max()


def test_r_rebin_factor():
    """Spec 013 FR-4 (2026-10-29): factor 1 is the native (legacy) binning; the factor must divide the r bins."""
    band = np.random.default_rng(3).poisson(5.0, (nema.R_BINS, nema.T_BINS)).astype(float)
    np.testing.assert_array_equal(nema.background_subtracted(band, r_rebin=1), nema.background_subtracted(band))
    coarse, centers = nema.rebinned(band, nema.NEMA_R_REBIN)
    assert coarse.shape == (nema.R_BINS // nema.NEMA_R_REBIN, nema.T_BINS) and coarse.sum() == band.sum()
    assert centers[1] - centers[0] == pytest.approx(3 * 46.0 / 150)
    for bad in (4, 0, 1.0, True):
        with pytest.raises(ValueError, match="r_rebin"):
            nema.background_subtracted(band, r_rebin=bad)


def test_r_profile_fwhm():
    """Section 8.4.1 bin limit: FWHM of a Gaussian line profile in r above a flat floor."""
    band = np.outer(1000 * np.exp(-0.5 * (nema.R_CENTERS / 2.0) ** 2) + 40, np.ones(nema.T_BINS))
    assert nema.r_profile_fwhm(band) == pytest.approx(2.0 * 2.3548, rel=0.02)
    assert nema.r_profile_fwhm(np.ones((nema.R_BINS, nema.T_BINS))) is None


def test_edge_counts_interpolated_at_20_mm():
    """Section 8.4.2a: counts per bin at exactly r = -20 and +20 mm, linear between the neighbouring r bins."""
    band = np.outer(nema.R_CENTERS, np.ones(nema.T_BINS))       # counts equal to the bin's r
    left, right = nema.edge_counts(band)
    np.testing.assert_allclose(left, -20.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(right, 20.0, rtol=0, atol=1e-12)
    legacy = nema.edge_counts(band, edges=nema.LEGACY)
    assert legacy[0][0] == pytest.approx(-20.087, abs=1e-3) and legacy[1][0] == pytest.approx(20.087, abs=1e-3)
    with pytest.raises(ValueError, match="edges"):
        nema.edge_counts(band, edges="nearest")


def test_central_axial_range():
    """Section 8.3.5: with an AFOV over 65 cm only the central 65 cm is used; NaN (parallel LOR) never is."""
    z = np.array([-325.0, -325.01, 0.0, 325.0, 400.0, np.nan])
    assert nema.axial_ok(z).tolist() == [True, False, True, True, False, False]


@pytest.mark.parametrize("r_rebin", [1, nema.NEMA_R_REBIN])
@pytest.mark.parametrize("seed, slope", [(200, 0.0), (201, 0.8)])
def test_background_subtraction_recovers_injected_peak(seed, slope, r_rebin):
    dt, r, is_peak = h.dt_r_events(seed=seed, n=2_000_000, slope=slope)
    total, peak = nema.NemaHistograms(), nema.NemaHistograms()
    total.add((), dt, r)
    peak.add((), dt[is_peak], r[is_peak])
    recovered = nema.background_subtracted(total.band, r_rebin=r_rebin)
    injected = nema.background_subtracted(peak.band, r_rebin=r_rebin)
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
