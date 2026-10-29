"""NEMA NU 2-2018 timing resolution with a line source (spec 013 FR-4), scanner-neutral.

NEMA NU 2-2018 section 8, ported from the legacy IMAS script
``scripts_imas/imas_NEMA_CTR_multiprocess.py`` (untracked) and checked against
its output by a golden file:

- per LOR, the signed shortest distance ``r`` to the line, the point ``I`` on
  the LOR closest to it, and the timing error ``Δt = dt - (|L1 - I| - |L2 - I|) / c``;
- events with ``|r| <= 22.0644 mm`` (20 mm plus the legacy 2.0644 mm margin) and
  ``|Δt| <= 2 ns`` are kept;
- raw histogram: every kept event, 100 bins over +-2 ns;
- background-subtracted histogram: r x Δt counts (150 x 100 bins over +-23 mm,
  +-2 ns); per Δt bin, the r bins with centres in [-20, 20] mm minus a linear
  interpolation between the strip-edge counts at r = -20 and +20 mm, each
  interpolated between its neighbouring r bins (section 8.4.2a; the legacy
  script takes the nearest r bins: ``edges=LEGACY``, kept for parity);
- the background-subtracted width sums the r bins by ``NEMA_R_REBIN`` (0.92 mm; section 8.4.1 asks only for
  bins below a quarter of the r-profile FWHM, and single 0.307 mm edge bins dominated its noise); factor 1 is
  the legacy binning;
- with an axial field of view over 65 cm, only LORs whose point I lies in the
  central 65 cm are used (section 8.3.5, ``axial_ok``);
- width: a parabola through the peak bin and its neighbours, then linear
  interpolation at a fraction of its maximum.

Histograms filled batch by batch equal one fill on all events; the bin index is shared by
bootstrap replicates. Unlike the
legacy script, a peak in an edge bin or a missing crossing gives a status, and
LORs parallel to the line get NaN and are never kept.
"""
from __future__ import annotations

import numpy as np

C_MM_PER_PS = 0.299792             # legacy value (src.timing.shell uses 0.299792458); 5e-9 relative
R_SIGNAL_MM = 20.0
R_KEEP_MM = R_SIGNAL_MM + 2.0644   # legacy "r_max + hist_bin_size"
R_RANGE_MM, R_BINS = 23.0, 150
T_RANGE_PS, T_BINS = 2000.0, 100
R_EDGES = np.linspace(-R_RANGE_MM, R_RANGE_MM, R_BINS + 1)
T_EDGES = np.linspace(-T_RANGE_PS, T_RANGE_PS, T_BINS + 1)
R_CENTERS = (R_EDGES[:-1] + R_EDGES[1:]) / 2
T_CENTERS = (T_EDGES[:-1] + T_EDGES[1:]) / 2
FITTED, EDGE_PEAK, NO_PEAK, NO_CROSSING = "fitted", "peak at edge", "no parabola peak", "no crossing"
NEMA, LEGACY = "nema", "legacy"    # strip-edge counts: interpolated at +-20 mm (section 8.4.2a) or nearest r bin
NEMA_AXIAL_HALF_MM = 325.0         # section 8.3.5: an AFOV over 65 cm uses only the central 65 cm
NEMA_R_REBIN = 3                   # 0.92 mm r bins for the background subtraction (spec 013 FR-4, 2026-10-29)


def line_frame(endpoint_1_mm, endpoint_2_mm, p1_mm, p2_mm):
    """Per LOR: signed distance ``r`` (mm) to the line P1-P2, closest point ``I`` on the LOR, and
    ``(|L1 - I| - |L2 - I|) / c`` (ps). NaN for LORs parallel to the line."""
    l1, l2 = np.asarray(endpoint_1_mm, np.float64), np.asarray(endpoint_2_mm, np.float64)
    p1, p2 = np.asarray(p1_mm, np.float64), np.asarray(p2_mm, np.float64)
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    v = (l2 - l1) / np.linalg.norm(l2 - l1, axis=1)[:, None]
    cross = np.cross(u, v)
    norm = np.linalg.norm(cross, axis=1)
    parallel = np.isclose(norm, 0)
    norm = np.where(parallel, np.nan, norm)
    w0 = l1 - p1
    r = np.einsum("ij,ij->i", w0, cross) / norm
    a, b, c = np.dot(u, u), v @ u, np.einsum("ij,ij->i", v, v)
    d, e = w0 @ u, np.einsum("ij,ij->i", v, w0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (b * d - a * e) / (a * c - b * b)
    t[parallel] = np.nan
    intercept = l1 + t[:, None] * v
    tof = (np.linalg.norm(l1 - intercept, axis=1) - np.linalg.norm(l2 - intercept, axis=1)) / C_MM_PER_PS
    return r, intercept, tof


def kept(dt_error_ps, r_mm):
    """The legacy per-event cuts: ``|r| <= R_KEEP_MM`` and ``|Δt| <= T_RANGE_PS`` (NaN never kept)."""
    with np.errstate(invalid="ignore"):
        return (np.abs(r_mm) <= R_KEEP_MM) & (np.abs(dt_error_ps) <= T_RANGE_PS)


def axial_ok(z_mm, half_mm=NEMA_AXIAL_HALF_MM):
    """Section 8.3.5: ``|z| <= half_mm`` (NaN never)."""
    with np.errstate(invalid="ignore"):
        return np.abs(np.asarray(z_mm, np.float64)) <= half_mm


def bin_index(dt_error_ps, r_mm):
    """(kept mask, flat r x Δt bin index of the kept events), binned as ``np.histogram2d`` on the legacy ranges
    (right edges open except the last)."""
    ok = kept(dt_error_ps, r_mm)
    r = np.searchsorted(R_EDGES, np.asarray(r_mm)[ok], side="right") - 1
    t = np.searchsorted(T_EDGES, np.asarray(dt_error_ps)[ok], side="right") - 1
    r[r == R_BINS] = R_BINS - 1
    t[t == T_BINS] = T_BINS - 1
    return ok, r * T_BINS + t


class NemaHistograms:
    """r x Δt counts (``R_BINS x T_BINS``) per index of ``shape``; the raw histogram is their sum over r (every
    kept event lies inside the r range)."""

    def __init__(self, shape=(), dtype=np.float64):
        self.band = np.zeros(tuple(shape) + (R_BINS, T_BINS), dtype)

    @property
    def raw(self):
        return self.band.sum(axis=-2)

    def add(self, index, dt_error_ps, r_mm, weights=None):
        """Add the kept events; ``index`` selects the histograms (``()`` for an unshaped set)."""
        ok, flat = bin_index(dt_error_ps, r_mm)
        w = None if weights is None else np.asarray(weights)[ok]
        self.band[index] += np.bincount(flat, weights=w, minlength=R_BINS * T_BINS).reshape(R_BINS, T_BINS)
        return int(np.count_nonzero(ok))


def rebinned(band, r_rebin=1):
    """(band with every ``r_rebin`` r bins summed, the new r-bin centres); the factor must divide ``R_BINS``."""
    if isinstance(r_rebin, bool) or not isinstance(r_rebin, int) or r_rebin < 1 or R_BINS % r_rebin:
        raise ValueError(f"r_rebin must be a whole divisor of {R_BINS}, not {r_rebin!r}")
    band = np.asarray(band, np.float64)
    if r_rebin == 1:
        return band, R_CENTERS
    shape = band.shape[:-2] + (R_BINS // r_rebin, r_rebin, band.shape[-1])
    edges = R_EDGES[::r_rebin]
    return band.reshape(shape).sum(axis=-2), (edges[:-1] + edges[1:]) / 2


def edge_counts(band, edges=NEMA, r_rebin=1):
    """Per Δt bin, the counts per r bin at the strip edges r = -20 and +20 mm: interpolated linearly between the
    two neighbouring r bins (section 8.4.2a), or the nearest r bin (``LEGACY``, the legacy script); r bins summed
    by ``r_rebin`` first."""
    if edges not in (NEMA, LEGACY):
        raise ValueError(f"edges must be {NEMA!r} or {LEGACY!r}, not {edges!r}")
    band, centers = rebinned(band, r_rebin)
    out = []
    for r0 in (-R_SIGNAL_MM, R_SIGNAL_MM):
        if edges == LEGACY:
            out.append(band[..., np.argmin(np.abs(centers - r0)), :])
            continue
        i = int(np.searchsorted(centers, r0)) - 1
        f = (r0 - centers[i]) / (centers[i + 1] - centers[i])
        out.append((1 - f) * band[..., i, :] + f * band[..., i + 1, :])
    return out


def background_subtracted(band, edges=NEMA, r_rebin=1):
    """Per Δt bin: signal r bins minus the background interpolated linearly between the strip-edge counts
    (section 8.4.2; ``edges=LEGACY`` with ``r_rebin=1`` is the legacy ``background_correction``)."""
    left, right = edge_counts(band, edges, r_rebin)
    band, centers = rebinned(band, r_rebin)
    signal = (centers >= -R_SIGNAL_MM) & (centers <= R_SIGNAL_MM)
    r = centers[signal]
    w_left, w_right = np.sum((R_SIGNAL_MM - r) / (2 * R_SIGNAL_MM)), np.sum((r + R_SIGNAL_MM) / (2 * R_SIGNAL_MM))
    return band[..., signal, :].sum(axis=-2) - w_left * left - w_right * right


def r_profile_fwhm(band):
    """FWHM (mm) of the r profile (counts summed over Δt) above the mean of its two outermost bins, from linear
    interpolation at half maximum; None without a crossing on both sides. For the section 8.4.1 bin limit."""
    profile = np.asarray(band, np.float64).sum(axis=-1)
    level = (profile[0] + profile[-1]) / 2
    status, width, _ = nema_width(profile - level, R_CENTERS, 0.5)
    return width if status == FITTED else None


def nema_width(hist, centers=T_CENTERS, fraction=0.5):
    """Legacy ``find_widths``: (status, width, parabola peak position); width and peak None unless fitted."""
    hist = np.asarray(hist, np.float64)
    peak = int(np.argmax(hist))
    if peak == 0 or peak == len(hist) - 1:
        return EDGE_PEAK, None, None
    a, b, c = np.polyfit(centers[peak - 1:peak + 2], hist[peak - 1:peak + 2], 2)
    if not a < 0:
        return NO_PEAK, None, None
    x_vertex, y_vertex = -b / (2 * a), c - b ** 2 / (4 * a)
    target = y_vertex * fraction
    below_left, below_right = np.flatnonzero(hist[:peak] <= target), np.flatnonzero(hist[peak:] <= target)
    if not len(below_left) or not len(below_right):
        return NO_CROSSING, None, None
    i, j = below_left[-1], below_right[0] + peak
    left = centers[i] + (target - hist[i]) * ((centers[i + 1] - centers[i]) / (hist[i + 1] - hist[i]))
    right = centers[j] + (target - hist[j]) * ((centers[j - 1] - centers[j]) / (hist[j - 1] - hist[j]))
    return FITTED, float(right - left), float(x_vertex)


def widths(hist, centers=T_CENTERS):
    """FWHM, FWTM and peak (ps) of one timing histogram; status is the first failure, else ``fitted``."""
    half, fwhm, peak = nema_width(hist, centers, 0.5)
    tenth, fwtm, _ = nema_width(hist, centers, 0.1)
    status = half if half != FITTED else tenth
    return {"status": status, "fwhm_ps": fwhm, "fwtm_ps": fwtm, "peak_ps": peak,
            "events": float(np.sum(hist))}


def summarize(raw, band, edges=NEMA, r_rebin=1):
    """NEMA widths without and with background subtraction (r bins summed by ``r_rebin`` for the latter)."""
    return {"raw": widths(raw), "background_subtracted": widths(background_subtracted(band, edges, r_rebin))}
