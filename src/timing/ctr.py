"""Scanner-neutral point-source CTR core (spec 011 T15, moved from spec 008's ``src/cornell/skew/ctr.py``).

Known-position residuals, timing-blind LOR convergence, the binned Gaussian +
flat fit with model-free FWHM/FWTM, uncertainty combination and the
TOF dependence of the residual. Scanner front ends (``src.cornell.skew.ctr``,
``src.imas.skew.ctr``) supply selection, endpoints, LUTs and reports.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from src.timing.shell import C_MM_PER_PS

LOCALIZATION_ITERATIONS = 6
LOCALIZATION_FLOOR_MM = 5.0
SEED_STEPS_MM = (25.0, 4.0)       # coarse grid over the bore, then a fine grid around the coarse best
SEED_NEAR_MM = 20.0               # LORs counted per grid point (half on the fine grid); kept for the refit
SEED_SAMPLE = 5_000
FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))
GAUSSIAN_FWTM_TO_FWHM = math.sqrt(math.log(10) / math.log(2))  # 1.823
MIN_FIT_EVENTS = 100
TOF_EDGES_PS = np.arange(-2000.0, 2000.0 + 1, 100.0)
MIN_TOF_BIN_EVENTS = 100


class CtrError(ValueError):
    pass


def geometric_tof_ps(source_mm, endpoint_1_mm, endpoint_2_mm):
    """Expected ``t_1 - t_2`` (ps) for annihilation at the source; rows of endpoints in mm."""
    source = np.asarray(source_mm, np.float64)
    return (np.linalg.norm(source - endpoint_1_mm, axis=-1)
            - np.linalg.norm(source - endpoint_2_mm, axis=-1)) / C_MM_PER_PS


def lor_distance_mm(source_mm, endpoint_1_mm, endpoint_2_mm):
    """Distance (mm) of the source from each LOR and its position along it (0 at endpoint 1, 1 at endpoint 2)."""
    source = np.asarray(source_mm, np.float64)
    direction = endpoint_2_mm - endpoint_1_mm
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.einsum("ij,ij->i", source - endpoint_1_mm, direction) / np.einsum("ij,ij->i", direction, direction)
        distance = np.linalg.norm(source - (endpoint_1_mm + along[:, None] * direction), axis=-1)
    return distance, along


def residual_ps(dt_corrected_ps, source_mm, endpoint_1_mm, endpoint_2_mm):
    """Corrected ``t_1 - t_2`` minus the known-position geometric TOF."""
    return np.asarray(dt_corrected_ps, np.float64) - geometric_tof_ps(source_mm, endpoint_1_mm, endpoint_2_mm)


def converge_point(endpoint_1_mm, endpoint_2_mm, *, iterations=LOCALIZATION_ITERATIONS,
                   floor_mm=LOCALIZATION_FLOOR_MM):
    """Point closest to the LORs (least squares), refitted on LORs within max(median miss, floor) of it.

    Uses endpoints only, never timestamps. Returns (point, median miss of the last fit's LORs, LORs used).
    """
    direction = endpoint_2_mm - endpoint_1_mm
    length = np.linalg.norm(direction, axis=1)
    ok = np.isfinite(length) & (length > 0)
    if np.count_nonzero(ok) < 3:
        raise CtrError("fewer than three LORs to locate the source")
    p, d = endpoint_1_mm[ok], direction[ok] / length[ok, None]
    used = np.ones(len(p), bool)
    for _ in range(iterations):
        pu, du = p[used], d[used]
        # sum over LORs of (I - d d^T), and of (I - d d^T) p
        a = len(pu) * np.eye(3) - du.T @ du
        b = pu.sum(axis=0) - du.T @ np.einsum("ij,ij->i", du, pu)
        if abs(np.linalg.det(a)) < 1e-9 * len(pu) ** 3:
            raise CtrError("LORs are parallel: the source cannot be located")
        point = np.linalg.solve(a, b)
        along = np.einsum("ij,ij->i", point - p, d)
        miss = np.linalg.norm(point - (p + along[:, None] * d), axis=1)
        used = miss <= max(float(np.median(miss)), floor_mm)
    return point, float(np.median(miss[used])), int(np.count_nonzero(used))


def _near_counts(grid, p, d, radius_mm):
    """For each grid point, the number of LORs (``p + t d``, unit ``d``) passing within ``radius_mm``."""
    pd, pp = np.einsum("ij,ij->i", p, d), np.einsum("ij,ij->i", p, p)
    counts = np.empty(len(grid), np.int64)
    chunk = max(1, 4_000_000 // len(p))
    for start in range(0, len(grid), chunk):
        s = grid[start:start + chunk]
        along = s @ d.T - pd
        distance2 = np.einsum("ij,ij->i", s, s)[:, None] - 2 * s @ p.T + pp - along * along
        counts[start:start + chunk] = np.count_nonzero(distance2 <= radius_mm * radius_mm, axis=1)
    return counts


def seeded_converge_point(endpoint_1_mm, endpoint_2_mm, *, radius_mm, half_length_mm,
                          steps_mm=SEED_STEPS_MM, near_mm=SEED_NEAR_MM, sample=SEED_SAMPLE, seed=0):
    """``converge_point`` on the LORs within ``near_mm`` of the point most LORs pass (bug-imas-ctr-localization).

    For sources among scanner-wide background (natural LYSO radioactivity, a weak source), where the
    all-LOR least-squares start of ``converge_point`` is pulled off the source. The densest point is found on
    a coarse grid over the cylinder ``radius_mm`` x ``|z| <= half_length_mm``, then on a fine grid around it,
    using up to ``sample`` LORs. Uses endpoints only. Returns (point, median miss, LORs used, seed point).
    """
    direction = endpoint_2_mm - endpoint_1_mm
    length = np.linalg.norm(direction, axis=1)
    ok = np.isfinite(length) & (length > 0)
    if np.count_nonzero(ok) < 3:
        raise CtrError("fewer than three LORs to locate the source")
    p, q, d = endpoint_1_mm[ok], endpoint_2_mm[ok], direction[ok] / length[ok, None]
    pick = np.random.default_rng(seed).permutation(len(p))[:sample]
    coarse, fine = steps_mm
    axis = np.arange(-radius_mm, radius_mm + coarse / 2, coarse)
    x, y, z = np.meshgrid(axis, axis, np.arange(-half_length_mm, half_length_mm + coarse / 2, coarse),
                          indexing="ij")
    grid = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    grid = grid[grid[:, 0] ** 2 + grid[:, 1] ** 2 <= radius_mm ** 2]
    best = grid[np.argmax(_near_counts(grid, p[pick], d[pick], near_mm))]
    offsets = np.arange(-coarse, coarse + fine / 2, fine)
    grid = best + np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3)
    best = grid[np.argmax(_near_counts(grid, p[pick], d[pick], near_mm / 2))]
    near = lor_distance_mm(best, p, q)[0] <= near_mm
    if np.count_nonzero(near) < 3:
        raise CtrError("fewer than three LORs near the densest point")
    point, miss, used = converge_point(p[near], q[near])
    return point, miss, used, best


def _model(params, edges):
    mean, sigma, signal, background = params
    return signal * np.diff(ndtr((edges - mean) / sigma)) + background / (len(edges) - 1)


def _width_at(counts, centers, background_per_bin, fraction, crossings=None):
    """Model-free full width at ``fraction`` of the peak height above the background (3-bin smoothing)."""
    crossing = (crossings or level_crossings)(counts, centers, background_per_bin, fraction)
    return None if crossing is None else float(crossing[1] - crossing[0])


def level_crossings(counts, centers, background_per_bin, fraction):
    """(left, right, level) where the smoothed histogram crosses ``fraction`` of its height above background."""
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    k = int(np.argmax(smooth))
    height = smooth[k] - background_per_bin
    if not height > 0:
        return None
    level = background_per_bin + height * fraction
    width = centers[1] - centers[0]
    left = k
    while left > 0 and smooth[left - 1] >= level:
        left -= 1
    right = k
    while right < len(smooth) - 1 and smooth[right + 1] >= level:
        right += 1
    if left == 0 or right == len(smooth) - 1:
        return None
    x_left = centers[left - 1] + width * (level - smooth[left - 1]) / (smooth[left] - smooth[left - 1])
    x_right = centers[right] + width * (smooth[right] - level) / (smooth[right] - smooth[right + 1])
    return float(x_left), float(x_right), float(level)


PEAK_WINDOW, EDGE_WINDOW = 0.35, 0.25   # local model-free windows, in units of a rough FWHM (spec 011 FR-10)


def _local_fit(counts, centers, x0, half):
    """Quadratic in (x - x0) through the bins within ``half`` of ``x0`` (Poisson weights)."""
    near = np.abs(centers - x0) <= half
    return np.polyfit(centers[near] - x0, counts[near], 2, w=1 / np.sqrt(np.maximum(counts[near], 1.0)))


def local_crossings(counts, centers, background_per_bin, fraction):
    """``level_crossings`` from local quadratic fits (spec 011 FR-10): the peak height from the bins within
    ``PEAK_WINDOW`` x a rough FWHM of the mode, each crossing from the bins within ``EDGE_WINDOW`` x it (refined
    twice). Model free (no global shape); about half the replicate spread of the 3-bin-smoothed crossing."""
    counts = np.asarray(counts, np.float64)
    rough = level_crossings(counts, centers, background_per_bin, 0.5)
    if rough is None:
        return None
    step = centers[1] - centers[0]
    peak_half = max(PEAK_WINDOW * (rough[1] - rough[0]), 2 * step)
    edge_half = max(EDGE_WINDOW * (rough[1] - rough[0]), 2 * step)
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    k = int(np.argmax(smooth))
    a, b, c = _local_fit(counts, centers, centers[k], peak_half)
    vertex = -b / (2 * a) if a < 0 and abs(b / (2 * a)) <= peak_half else 0.0
    height = float(np.polyval((a, b, c), vertex))
    if not height > background_per_bin:
        return None
    level = background_per_bin + fraction * (height - background_per_bin)
    left, right = k, k
    while left > 0 and smooth[left - 1] >= level:
        left -= 1
    while right < len(smooth) - 1 and smooth[right + 1] >= level:
        right += 1
    if left == 0 or right == len(smooth) - 1:
        return None
    found = []
    for x0 in (centers[left] - step / 2, centers[right] + step / 2):
        for _ in range(2):
            poly = _local_fit(counts, centers, x0, edge_half)
            poly[-1] -= level
            roots = np.roots(poly)
            roots = roots[np.isreal(roots)].real
            roots = roots[np.abs(roots) <= edge_half]
            if not len(roots):
                return None
            x0 = x0 + roots[np.argmin(np.abs(roots))]
        found.append(float(x0))
    return found[0], found[1], float(level)


def fit_histogram(counts, edges, *, start=None, crossings=None):
    """Gaussian + flat background, binned Poisson likelihood; ``status`` says why numbers are absent.

    ``crossings`` is the model-free rule: ``level_crossings`` (spec 008, default) or ``local_crossings``."""
    counts = np.asarray(counts, np.float64)
    edges = np.asarray(edges, np.float64)
    total = float(counts.sum())
    if total < MIN_FIT_EVENTS:
        return {"status": "insufficient events", "events": total}
    centers, width, bins = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0], len(counts)
    if start is None:
        outer = max(bins // 10, 1)
        level = float(np.concatenate((counts[:outer], counts[-outer:])).mean())
        smooth = np.convolve(counts, np.ones(5) / 5, mode="same")
        k = int(np.argmax(smooth))
        above = smooth - level
        sigma = max(np.count_nonzero(above > above[k] / 2) * width / FWHM_PER_SIGMA, width)
        background = max(level * bins, 1.0)
        start = (centers[k], sigma, max(total - background, 1.0), background)
    mean, sigma, signal, background = start
    x0 = np.array([mean, math.log(sigma), math.log(max(signal, 1.0)), math.log(max(background, 1e-3))])

    def nll(x):
        mu = np.maximum(_model((x[0], math.exp(x[1]), math.exp(x[2]), math.exp(x[3])), edges), 1e-300)
        return float(np.sum(mu - counts * np.log(mu)))

    result = minimize(nll, x0, method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 6000, "maxfev": 12000})
    mean, sigma, signal, background = result.x[0], *np.exp(result.x[1:])
    mu = _model((mean, sigma, signal, background), edges)
    inside = (sigma > width / 2) and (sigma < edges[-1] - edges[0]) and (edges[0] < mean < edges[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance = 2 * np.sum(mu - counts + np.where(counts > 0, counts * np.log(counts / mu), 0.0))
    half = _width_at(counts, centers, background / bins, 0.5, crossings)
    tenth = _width_at(counts, centers, background / bins, 0.1, crossings)
    return {"status": "fitted" if result.success and inside else "fit failed",
            "events": total, "mean_ps": float(mean), "sigma_ps": float(sigma),
            "fwhm_ps": float(FWHM_PER_SIGMA * sigma), "model_free_fwhm_ps": half, "model_free_fwtm_ps": tenth,
            "fwtm_to_fwhm": None if half is None or tenth is None else tenth / half,
            "signal_events": float(signal), "background_events": float(background),
            "background_fraction": float(background / (signal + background)),
            "deviance_per_dof": float(deviance / max(bins - 4, 1)), "_start": (mean, sigma, signal, background)}


def _public(fit):
    return {k: v for k, v in fit.items() if not k.startswith("_")}


def combine_uncertainty(nominal, variants, statistical):
    """Systematic components (largest |change| per group), quadrature total with the statistical part."""
    components = {group: max(abs(v - nominal) for v in values) for group, values in variants.items() if values}
    systematic = math.sqrt(sum(c * c for c in components.values()))
    return {"statistical": statistical, "systematic_components": components, "systematic": systematic,
            "total": math.hypot(statistical, systematic) if statistical is not None else None}


def tof_dependence(tof_hist, edges, sigma_ps):
    """Median residual per geometric-TOF bin and its weighted linear slope (ps per ps of TOF)."""
    rows = []
    for k in range(tof_hist.shape[0]):
        counts = tof_hist[k]
        n = counts.sum()
        if n < MIN_TOF_BIN_EVENTS:
            continue
        cumulative = np.cumsum(counts)
        j = int(np.searchsorted(cumulative, n / 2))
        below = cumulative[j - 1] if j else 0.0
        median = edges[j] + (n / 2 - below) / counts[j] * (edges[1] - edges[0])
        rows.append({"tof_ps": float((TOF_EDGES_PS[k] + TOF_EDGES_PS[k + 1]) / 2), "events": float(n),
                     "median_residual_ps": float(median), "median_sigma_ps": float(1.2533 * sigma_ps / math.sqrt(n))})
    out = {"bins": rows, "slope": None, "slope_sigma": None, "intercept_ps": None}
    if len(rows) >= 3 and sigma_ps:
        x = np.array([r["tof_ps"] for r in rows])
        y = np.array([r["median_residual_ps"] for r in rows])
        w = 1 / np.array([r["median_sigma_ps"] for r in rows]) ** 2
        design = np.column_stack((np.ones_like(x), x))
        cov = np.linalg.inv(design.T @ (design * w[:, None]))
        coef = cov @ design.T @ (w * y)
        out.update(slope=float(coef[1]), slope_sigma=float(math.sqrt(cov[1, 1])), intercept_ps=float(coef[0]))
    return out


@dataclass(frozen=True)
class Variant:
    name: str
    depth_index: int      # 0 nominal depth, k > 0 depth_variants_mm[k - 1]
    shift: tuple          # in units of each acquisition's source_sigma_mm
    lor_distance_mm: float
    group: str | None     # systematic group; None for nominal and sensitivity


def variants(spec):
    out = [Variant("nominal", 0, (0.0, 0.0, 0.0), spec.lor_distance_mm, None)]
    out += [Variant(f"lor_{d:g}mm", 0, (0.0, 0.0, 0.0), d, None) for d in spec.sensitivity_lor_mm]
    for axis, label in enumerate("xyz"):
        for sign in (-1.0, 1.0):
            shift = [0.0, 0.0, 0.0]
            shift[axis] = sign
            out.append(Variant(f"source_{label}{'+' if sign > 0 else '-'}", 0, tuple(shift), spec.lor_distance_mm,
                               f"source {label}"))
    out += [Variant(f"depth_{d:g}mm", k + 1, (0.0, 0.0, 0.0), spec.lor_distance_mm, "depth")
            for k, d in enumerate(spec.depth_variants_mm)]
    return out


def summarize(accumulator, crossings=None):
    """Fits, bootstrap and systematics per acquisition and pooled (event weighted)."""
    spec, edges = accumulator.spec, accumulator.edges
    names = [a.name for a in spec.acquisitions]
    scopes = [("pooled", slice(None))] + [(name, k) for k, name in enumerate(names)]
    results = {}
    for scope, selector in scopes:
        def pick(array):
            return array[..., selector, :].sum(axis=-2) if scope == "pooled" else array[..., selector, :]

        hist = pick(accumulator.hist)
        nominal = fit_histogram(hist[0], edges, crossings=crossings)
        entry = {"fit": _public(nominal)}
        if nominal["status"] != "fitted":
            entry["uncertainty"] = None
            results[scope] = entry
            continue
        boot = pick(accumulator.boot) if spec.bootstrap else np.zeros((0, len(edges) - 1))
        replicate = [fit_histogram(h, edges, start=nominal["_start"], crossings=crossings) for h in boot]
        good = [f for f in replicate if f["status"] == "fitted"]
        entry["bootstrap"] = {"replicates": len(replicate), "fitted": len(good)}
        variant_fits = {v.name: fit_histogram(hist[i], edges, start=nominal["_start"], crossings=crossings)
                        for i, v in enumerate(accumulator.variants) if i}
        entry["variants"] = {name: _public(f) for name, f in variant_fits.items()}
        for key in ("fwhm_ps", "model_free_fwhm_ps", "model_free_fwtm_ps"):
            values = [f[key] for f in good if f[key] is not None]
            statistical = float(np.std(values, ddof=1)) if len(values) >= 2 else None
            groups = {}
            for v in accumulator.variants:
                fit = variant_fits.get(v.name)
                if v.group and fit and fit["status"] == "fitted" and fit[key] is not None:
                    groups.setdefault(v.group, []).append(fit[key])
            entry.setdefault("uncertainty", {})[key] = (
                combine_uncertainty(nominal[key], groups, statistical) if nominal[key] is not None else None)
        entry["lor_sensitivity_fwhm_ps"] = {v.name: variant_fits[v.name].get("fwhm_ps")
                                            for v in accumulator.variants[1:] if v.group is None}
        tof = accumulator.tof.sum(axis=0) if scope == "pooled" else accumulator.tof[selector]
        entry["tof_dependence"] = tof_dependence(tof, edges, nominal["sigma_ps"])
        results[scope] = entry
    return results
