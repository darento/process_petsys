"""Synthetic NECR line-source fixtures shared by the spec 013 tests and the local golden capture."""
import numpy as np

LEGACY_P1 = np.array([-37.18432338, 18.35763082, -356.109009])   # legacy image line (imas_NEMA_CTR_multiprocess)
LEGACY_P2 = np.array([-36.60248657, 11.73070676, 356.109009])
RING_R_MM = 410.0
HALF_Z_MM = 380.0


def cylinder_hits(origin, direction, radius=RING_R_MM):
    """Both intersections of lines ``origin + s direction`` with the cylinder x^2 + y^2 = radius^2."""
    a = direction[:, 0] ** 2 + direction[:, 1] ** 2
    b = 2 * (origin[:, 0] * direction[:, 0] + origin[:, 1] * direction[:, 1])
    c = origin[:, 0] ** 2 + origin[:, 1] ** 2 - radius ** 2
    root = np.sqrt(b * b - 4 * a * c)
    return (origin + ((-b - root) / (2 * a))[:, None] * direction,
            origin + ((-b + root) / (2 * a))[:, None] * direction)


def line_lors(seed=13, n=500, p1=LEGACY_P1, p2=LEGACY_P2, spread_mm=15.0):
    """LOR endpoints (L1, L2) through points within ``spread_mm`` of the line P1-P2, and a measured dt (ps).

    Emission points lie along the line inside the axial field of view, displaced transversally by up to
    ``spread_mm``; directions are isotropic, and LORs leaving the cylinder beyond |z| 380 mm are redrawn.
    """
    rng = np.random.default_rng(seed)
    out1, out2 = [], []
    while sum(len(a) for a in out1) < n:
        m = 4 * n
        s = rng.uniform(0.05, 0.95, m)
        point = p1 + s[:, None] * (p2 - p1) + rng.uniform(-spread_mm, spread_mm, (m, 3)) * [1, 1, 0]
        direction = rng.normal(size=(m, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        l1, l2 = cylinder_hits(point, direction)
        keep = (np.abs(l1[:, 2]) <= HALF_Z_MM) & (np.abs(l2[:, 2]) <= HALF_Z_MM)
        out1.append(l1[keep])
        out2.append(l2[keep])
    l1, l2 = np.concatenate(out1)[:n], np.concatenate(out2)[:n]
    return l1, l2, rng.normal(0.0, 400.0, n)


def dt_r_events(seed=17, n=200_000, *, peak_fraction=0.6, sigma_ps=300.0, mean_ps=20.0, sigma_r_mm=4.0,
                slope=0.0):
    """``(Δt, r)`` of NEMA events: a Gaussian peak at the line plus a background flat in Δt.

    The background is spread over -30 < r < 30 mm with density ``1 + slope * r / 30`` (``slope`` in [-1, 1]);
    the peak's Δt is N(mean, sigma), its r N(0, sigma_r). Events outside the legacy kept region are included,
    so the cuts are exercised. Returns (dt, r, is_peak).
    """
    rng = np.random.default_rng(seed)
    k = int(round(n * peak_fraction))
    dt = np.concatenate((rng.normal(mean_ps, sigma_ps, k), rng.uniform(-2500.0, 2500.0, n - k)))
    u = rng.random(n - k)
    if slope:                                   # inverse CDF of a linear density on [-30, 30]
        grid = np.linspace(-30.0, 30.0, 6001)
        cdf = np.cumsum(1 + slope * grid / 30.0)
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        r_bg = np.interp(u, cdf, grid)
    else:
        r_bg = -30.0 + 60.0 * u
    r = np.concatenate((rng.normal(0.0, sigma_r_mm, k), r_bg))
    peak = np.arange(n) < k
    order = rng.permutation(n)
    return dt[order], r[order], peak[order]


def charge_walk(q):
    """Injected time walk (ps) of one side against its timing-hit charge (spec 012 CTR fixture)."""
    return 250.0 * np.exp(-(np.asarray(q, np.float64) - 5.0) / 6.0)


def frozen_tables(base, setup, times, delays, *, legacy_error_ps=100.0, em_error_ps=0.0, legacy_missing=(), seed=3):
    """A job record, an energy calibration (mu 511 a.u.), an EM skew LUT of the true ``delays`` (plus
    ``em_error_ps`` Gaussian errors), a walk LUT tabulating ``charge_walk`` paired with it, and a legacy file
    (``t + skew``) with Gaussian errors and 9999 (no delay) for the ``legacy_missing`` channels.

    Returns the paths, ``times``, ``delays`` and the ``skew_lut(path, values, **provenance)`` writer.
    """
    import json
    from imas_skew_helpers import CONFIG, mu_file, write
    from src.imas.skew import job as ij, outputs
    from src.imas.skew import walk as iw
    from src.timing import solver as ts
    from src.timing.cache import _sha256
    rng = np.random.default_rng(seed)
    encal = mu_file(base / "encal.tsv", {ch: 511.0 for ch in times})
    splits = [write(base / f"skew_{k}.ldat", []) for k in range(2)]
    job_path = base / "job.json"
    job_path.write_text(json.dumps({"schema": ij.SCHEMA, "config": str(CONFIG), "runs": [{"name": "r1", "splits": [
        {"path": str(splits[0]), "roles": ["encal", "train"]}, {"path": str(splits[1]), "roles": ["heldout"]}]}]}),
        encoding="utf-8")
    work = base / "job"
    work.mkdir()
    stamp = ij.fingerprint(ij.load_job(job_path))
    (work / "job_fingerprint.json").write_text(json.dumps(stamp), encoding="utf-8")

    def skew_lut(path, values=None, **extra):
        values = delays if values is None else values
        rows = np.zeros(len(times), ts.CHANNEL_DTYPE)
        rows["time_channel"], rows["delay_ps"] = times, [values[c] for c in times]
        rows["sigma_ps"] = rows["raw_sigma_ps"] = 5.0
        rows["status"], rows["apply"] = ts.LOW_STATISTICS, True
        provenance = {"job_sha256": stamp["sha256"], "estimator": "em",
                      "runs": {"r1": {"encal_sha256": _sha256(encal)}}, **extra}
        outputs.write_imas_lut(path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), provenance, setup)
        return path

    lut = skew_lut(work / "lut_em.csv", {c: d + rng.normal(0.0, em_error_ps) for c, d in delays.items()}
                   if em_error_ps else None)
    edges = {"walk2d": [np.linspace(5.0, 35.0, 61), np.array([0.0, 2.0])], "doi": [np.array([0.0, 100.0])],
             "kev": [np.array([0.0, 2000.0])]}
    centres = (edges["walk2d"][0][:-1] + edges["walk2d"][0][1:]) / 2
    offsets = {"walk2d": charge_walk(centres)[:, None], "doi": np.zeros(1), "kev": np.zeros(1)}
    events = {b: np.full(v.shape, 500) for b, v in offsets.items()}
    walk = base / "walk.json"
    iw.WalkLut.from_fit(edges, offsets, events, {"map_sha256": setup.provenance["map_sha256"],
                                                 "skew_lut_sha256": _sha256(lut), "fit_kind": "synthetic"}).write(walk)
    legacy = base / "skew_legacy.txt"
    missing = set(legacy_missing)
    legacy.write_text("".join(f"{c} 9999\n" if c in missing
                              else f"{c} {-(delays[c] + rng.normal(0.0, legacy_error_ps))}\n" for c in times),
                      encoding="utf-8")
    return {"base": base, "work": work, "encal": encal, "lut": lut, "walk": walk, "legacy": legacy,
            "skew_lut": skew_lut, "times": times, "delays": delays}


def necr_file(case, frames, name="necr.json", **changes):
    """A NECR CTR file over ``frames`` ({name: [ldat paths]}) with the ``frozen_tables`` tables."""
    import json
    from src.imas.skew import necr
    from src.timing.cache import _sha256

    def frozen(path):
        return {"path": str(path), "sha256": _sha256(path)}

    data = {"schema": necr.SCHEMA, "job_work": str(case["work"]), "encal": str(case["encal"]),
            "luts": {"em": frozen(case["lut"])}, "legacy": frozen(case["legacy"]),
            "walks": {"em_walk": {"skew": "em", **frozen(case["walk"])}},
            "output": str(case["base"] / "out" / name.removesuffix(".json")),
            "frames": [{"name": frame, "frame": k + 1, "ldat": [str(p) for p in paths],
                        "activity_kbq_ml": 8.0 / (k + 1), "activity_source": "synthetic"}
                       for k, (frame, paths) in enumerate(frames.items())],
            "lines": {"fitted": {"method": "lor_convergence"},
                      "image": {"p1": LEGACY_P1.tolist(), "p2": LEGACY_P2.tolist(), "source": "legacy image"}},
            "headline_line": "fitted",
            "gates": {"frame": next(iter(frames)), "transfer": ["legacy", "em"], "acceptance": ["em", "em_walk"]},
            "bootstrap": 20, "localization_events": 1000, "energy_check_records": 1000, **changes}
    path = case["base"] / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- synthetic NECR acquisition (spec 013 T4) ----------------------------------------------------------------

TRUE_POINT = np.array([-37.0, 15.0, 0.0])          # tilted line source of the synthetic NECR acquisition
TRUE_SLOPES = (0.01, -0.005)
NOISE_PS = 150.0


def true_line():
    """P1, P2 of the synthetic line at z = -/+ 356.109 mm."""
    ends = []
    for z in (-356.109009, 356.109009):
        ends.append(TRUE_POINT + [TRUE_SLOPES[0] * z, TRUE_SLOPES[1] * z, z])
    return ends


def all_time_channels(setup):
    from imas_skew_helpers import channels
    return sorted({ch for sm in range(120) for mm in range(16) for ch in channels(setup, sm, mm)[0]})


def channel_endpoints(base, setup, encal):
    """Nominal endpoint (mm) of every TIME channel (slab) of the map, from one selected record per channel."""
    from imas_skew_helpers import CONFIG, channels, side, write
    from src.imas.skew import select as sel
    from src.imas.skew.geometry import ImasGeometry
    from src.imas.skew.setup import ImasSetup
    sides = [(sm, mm, slab) for sm in range(120) for mm in range(16) for slab in range(len(channels(setup, sm, mm)[0]))]
    partner = {s: ((s[0] // 24) * 24 + (s[0] % 24 + 12) % 24, s[1], s[2]) for s in sides}
    records = [(side(setup, *a[:2], time={a[2]: 20.0}), side(setup, *partner[a][:2], time={partner[a][2]: 20.0}))
               for a in sides if partner[a] in partner]
    path = write(base / "endpoints.ldat", records)
    lookup = ImasSetup.from_files(CONFIG, encal).lookup()
    geometry = ImasGeometry.from_setup(setup)
    table = {}
    for f in sel.feature_batches(path, lookup, sel.Selection.production(lookup)):
        codes, rows = geometry.endpoints(f)
        for c1, c2, code, row in zip(f.column("time_channel_1"), f.column("time_channel_2"), codes, rows):
            if code != 1:
                table[int(c1)], table[int(c2)] = row[:3], row[3:]
    sided = {}
    for a in sides:
        sided[channels(setup, a[0], a[1])[0][a[2]]] = a
    keys = sorted(set(table) & set(sided))
    return np.array(keys), np.array([table[k] for k in keys]), [sided[k] for k in keys]


def necr_frame(path, setup, endpoints, delays, *, seed, signal=6000, background=2500, sigma_r_mm=2.0,
               line=None):
    """A synthetic NECR frame: LORs drawn from random channel pairs, kept as signal with probability
    exp(-r^2 / 2 sigma_r^2) when their closest point to the line lies within |z| <= 350 mm, and as background
    when |r| < 40 mm (flat in r). Signal dt is the line TOF plus the true delays (t + b), the charge walk of
    each side and NOISE_PS noise; background dt is flat over +-3 ns. Returns the record count."""
    from imas_skew_helpers import T0, side, write
    from src.timing import nema
    keys, points, sides = endpoints
    p1, p2 = true_line() if line is None else line
    rng = np.random.default_rng(seed)
    chosen = {"signal": [], "background": []}
    while len(chosen["signal"]) < signal or len(chosen["background"]) < background:
        i, j = rng.integers(len(keys), size=(2, 200_000))
        far = np.array([sides[a][0] // 24 != sides[b][0] // 24 or abs(sides[a][0] - sides[b][0]) > 2
                        for a, b in zip(i, j)])
        i, j = i[far], j[far]
        r, intercept, tof = nema.line_frame(points[i], points[j], p1, p2)
        keep = rng.random(len(r)) < np.exp(-0.5 * (r / sigma_r_mm) ** 2)
        keep &= np.abs(intercept[:, 2]) <= 350.0
        for k in np.flatnonzero(keep):
            chosen["signal"].append((i[k], j[k], tof[k]))
        flat = np.abs(r) < 40.0
        for k in np.flatnonzero(flat)[: max(background - len(chosen["background"]), 0)]:
            chosen["background"].append((i[k], j[k], None))
    events = chosen["signal"][:signal] + chosen["background"][:background]
    records = []
    for i, j, tof in events:
        q1, q2 = rng.uniform(5.0, 35.0, 2)
        c1, c2 = int(keys[i]), int(keys[j])
        if tof is None:
            dt = rng.uniform(-3000.0, 3000.0)
        else:
            dt = tof + delays[c1] - delays[c2] + charge_walk(q1) - charge_walk(q2) + rng.normal(0.0, NOISE_PS)
        a, b = sides[i], sides[j]
        records.append((side(setup, a[0], a[1], time={a[2]: float(q1)}, stamp=T0 + int(round(dt))),
                        side(setup, b[0], b[1], time={b[2]: float(q2)}, stamp=T0)))
    order = rng.permutation(len(records))
    write(path, [records[k] for k in order])
    return len(records)
