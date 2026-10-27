"""Synthetic walk-table fixtures shared by the walk tests and local golden capture (spec 012)."""
import numpy as np

CORNELL_FIXED_BINS = {"walk2d": (6, 4), "doi": (5,), "kev": (5,)}
CORNELL_FIT_BINS = {"charge": 6, "share": 4, "doi_mm": 5, "kev": 5}


def true_walk(charge, share, depth, kev):
    """Smooth injected walk (ps) of one side: charge and share 2D, a depth-like ramp and a low-keV tail."""
    return (250.0 * np.exp(-(charge - 5.0) / 6.0) + 150.0 * (0.9 - share) + 1.5 * depth
            + 0.15 * np.clip(480 - kev, 0, None))


def side_inputs(rng, n, names=("charge", "share", "doi_mm", "kev"), depth=(0.0, 20.0)):
    """One side's inputs; ``names`` maps (charge, share, depth, keV) to the model's input names."""
    share = np.where(rng.random(n) < 0.4, 1.0, rng.uniform(0.4, 0.95, n))
    values = (rng.uniform(5, 35, n), share, rng.uniform(*depth, n), np.clip(rng.normal(500, 60, n), 300, 700))
    return dict(zip(names, values))


def cornell_fixed_arrays():
    """Deterministic edges, offsets and events of a Cornell-layout table (no fit, no BLAS): some bins below
    200 events, so the sparse fill and the gauge both act."""
    edges = {"walk2d": [np.linspace(2.0, 40.0, 7), np.linspace(0.3, 1.0, 5)],
             "doi": [np.linspace(0.0, 20.0, 6)], "kev": [np.linspace(300.0, 700.0, 6)]}
    offsets, events = {}, {}
    for b, shape in CORNELL_FIXED_BINS.items():
        size = int(np.prod(shape))
        offsets[b] = (np.arange(size, dtype=np.float64) * 1.25 - 7.0).reshape(shape) / 3.0
        events[b] = ((np.arange(size) * 37) % 11 * 60).reshape(shape)
    return edges, offsets, events


def cornell_fit_events(seed=12, n=20_000):
    """Point-source-like residuals with injected walk: (side1, side2, residual, train)."""
    rng = np.random.default_rng(seed)
    side1, side2 = side_inputs(rng, n), side_inputs(rng, n)
    truth = true_walk(*side1.values()) - true_walk(*side2.values())
    residual = truth + rng.normal(0, 150, n)
    return side1, side2, residual, np.arange(n) % 3 != 0


def two_branch_events(seed=5, n=120_000, names=("charge", "share", "doi_ratio", "kev"), half_gap_ps=1070.0,
                      sigma_ps=270.0, background=0.05, depth=(1.2, 3.0)):
    """Bar-like events: dt = branch (+-half gap around a per-event midpoint) + W1 - W2 + noise, with a flat
    background; returns (side1, side2, dt, near, far, truth, is_background)."""
    rng = np.random.default_rng(seed)
    side1, side2 = side_inputs(rng, n, names, depth), side_inputs(rng, n, names, depth)
    truth = true_walk(*side1.values()) - true_walk(*side2.values())
    centre = rng.uniform(-600, 600, n)
    near, far = centre - half_gap_ps, centre + half_gap_ps
    branch = np.where(rng.random(n) < 0.5, near, far)
    is_background = rng.random(n) < background
    dt = np.where(is_background, rng.uniform(-5000, 5000, n), branch + truth + rng.normal(0, sigma_ps, n))
    return side1, side2, dt, near, far, truth, is_background
