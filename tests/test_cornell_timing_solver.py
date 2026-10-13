"""Sparse global TIME-channel solve from pair constraints (spec 008 T7, FR-10/FR-12).

Pair constraints are synthesized directly as T6 output rows with known
channel delays, so the expected answer is fixed by truth after the gauge.
"""
import csv
import json

import numpy as np
import pytest

from src.cornell import timing_pairs as tp
from src.cornell import timing_solver as ts
from src.cornell.timing_petsys import CanonicalLut, LutError, write_lut

pytestmark = pytest.mark.fr("008-FR-10", "008-FR-12")


def pairs(truth, edges, *, sigma=50.0, rng=None, status=tp.USABLE, events=20):
    """PAIR_FIT_DTYPE rows d_ij = b_i - b_j (+ noise) for canonical edges."""
    rows = np.zeros(len(edges), tp.PAIR_FIT_DTYPE)
    for k, (a, b) in enumerate(edges):
        i, j = min(a, b), max(a, b)
        rows[k]["channel_i"], rows[k]["channel_j"], rows[k]["events"] = i, j, events
        rows[k]["status"] = status
        noise = rng.normal(0, sigma) if rng is not None else 0.0
        rows[k]["shift_ps"] = truth[i] - truth[j] + noise if status == tp.USABLE else np.nan
        rows[k]["shift_sigma_ps"] = sigma if status == tp.USABLE else np.nan
    return rows


def random_graph(rng, channels, degree):
    edges = {(int(a), int(b)) for a, b in zip(channels[:-1], channels[1:])}  # a path keeps it connected
    while len(edges) < degree * len(channels) // 2:
        a, b = rng.choice(channels, 2, replace=False)
        edges.add((int(min(a, b)), int(max(a, b))))
    return sorted(edges)


def gauged(truth, channels):
    values = np.array([truth[c] for c in channels])
    return values - values.mean()


def test_known_delays_after_gauge():
    rng = np.random.default_rng(71)
    channels = np.arange(100, 160)
    truth = {int(c): float(rng.uniform(-5000, 5000)) for c in channels}
    edges = random_graph(rng, channels, 12)
    exact = ts.solve(pairs(truth, edges, sigma=5.0), channels)
    row = exact.channels
    assert list(row["time_channel"]) == list(channels)
    assert np.abs(row["delay_ps"] - gauged(truth, channels)).max() < 1e-6
    assert abs(row["delay_ps"].mean()) < 1e-9 and (row["status"] == ts.CALIBRATED).all() and row["apply"].all()
    assert exact.diagnostics["components"][0]["chi2"] < 1e-12
    # noisy constraints: pulls against truth follow the reported uncertainty
    pulls = []
    for seed in range(20):
        noisy = ts.solve(pairs(truth, edges, rng=np.random.default_rng(seed)), channels,
                         options=ts.SolveOptions(huber_k=None))
        pulls.append((noisy.channels["delay_ps"] - gauged(truth, channels)) / noisy.channels["raw_sigma_ps"])
        info = noisy.diagnostics["components"][0]
        assert info["dof"] == len(edges) - (len(channels) - 1)
        assert 0.7 < info["birge_ratio"] < 1.3
    pulls = np.concatenate(pulls)
    assert abs(pulls.mean()) < 0.1 and 0.85 < pulls.std() < 1.15
    # the gauge is a choice, the differences are not
    shifted = {c: v + 777.0 for c, v in truth.items()}
    assert np.allclose(ts.solve(pairs(shifted, edges, sigma=5.0), channels).channels["delay_ps"], row["delay_ps"],
                       atol=1e-6)


def test_disconnected_components_and_missing_channels():
    rng = np.random.default_rng(72)
    big, small = np.arange(0, 30), np.arange(40, 46)
    truth = {int(c): float(rng.uniform(-3000, 3000)) for c in np.r_[big, small, 60, 61, 62, 63, 64, 70]}
    big_edges = random_graph(rng, big, 8)
    rows = np.concatenate([
        pairs(truth, big_edges, sigma=5.0), pairs(truth, random_graph(rng, small, 4), sigma=5.0),
        pairs(truth, [(60, 3)], status=tp.ONE_BRANCH), pairs(truth, [(61, 4)], status=tp.ALIAS_UNRESOLVED),
        pairs(truth, [(62, 5)], status=tp.FIT_FAILED), pairs(truth, [(63, 6)], status=tp.MERGED),
        pairs(truth, [(6, 70)]),  # usable, but 70 is a down channel
    ])
    operational = np.r_[big, small, 60, 61, 62, 63, 64]
    solution = ts.solve(rows, operational, unavailable_channels=[70])
    by = {int(r["time_channel"]): r for r in solution.channels}
    expected = {60: ts.INSUFFICIENT, 61: ts.AMBIGUOUS, 62: ts.FAILED_FIT, 63: ts.AMBIGUOUS, 64: ts.UNOBSERVED,
                70: ts.HARDWARE_UNAVAILABLE}
    for channel, status in expected.items():
        assert by[channel]["status"] == status and not by[channel]["apply"], channel
        assert np.isnan(by[channel]["delay_ps"]) and np.isnan(by[channel]["sigma_ps"]), channel
    # the larger component is applied with its own zero-mean gauge, unaffected by the other
    assert all(by[c]["status"] == ts.CALIBRATED and by[c]["component"] == 0 for c in big)
    assert np.abs(np.array([by[c]["delay_ps"] for c in big]) - gauged(truth, big)).max() < 1e-6
    # the smaller one is solved relative to itself only and never applied
    for c in small:
        assert by[c]["status"] == ts.DISCONNECTED and by[c]["component"] == 1 and not by[c]["apply"]
    assert np.abs(np.array([by[c]["delay_ps"] for c in small]) - gauged(truth, small)).max() < 1e-6
    # the down channel's usable pair is excluded from the solve
    assert by[6]["usable_pairs"] == sum(6 in e for e in big_edges) and by[6]["pairs"] == by[6]["usable_pairs"] + 2
    assert solution.diagnostics["status_counts"]["hardware unavailable"] == 1
    assert len(solution.diagnostics["components"]) == 2 and solution.diagnostics["components"][1]["applied"] is False
    assert set(solution.delays()) == set(int(c) for c in big)
    with pytest.raises(ValueError, match="outside the operational"):
        ts.solve(rows, big)
    with pytest.raises(ValueError, match="one role"):
        mixed = rows.copy()
        mixed["role"][0] = 1
        ts.solve(mixed, operational, [70])


def test_contamination_and_loop_residuals():
    rng = np.random.default_rng(73)
    channels = np.arange(0, 50)
    truth = {int(c): float(rng.uniform(-4000, 4000)) for c in channels}
    edges = random_graph(rng, channels, 14)
    rows = pairs(truth, edges, rng=rng)
    bad = rng.choice(len(rows), 4, replace=False)
    rows["shift_ps"][bad] += np.array([1900.0, -1900.0, 3800.0, 2500.0])  # alias slips / wrong constraints
    plain = ts.solve(rows, channels, options=ts.SolveOptions(huber_k=None))
    robust = ts.solve(rows, channels)
    truth_g = gauged(truth, channels)
    err_plain = np.abs(plain.channels["delay_ps"] - truth_g).max()
    err_robust = np.abs(robust.channels["delay_ps"] - truth_g).max()
    assert err_plain > 100 and err_robust < 4 * robust.channels["raw_sigma_ps"].max()
    # contaminated pairs carry the large normalized residuals and are downweighted
    flagged = np.flatnonzero(np.abs(robust.pairs["normalized_residual"]) > 5)
    assert set(flagged) == set(bad)
    assert (robust.pairs["huber_weight"][bad] < 0.1).all()
    assert robust.diagnostics["components"][0]["flagged_pairs"] == 4
    # loop closure: an inconsistent triangle shows up in chi2/dof and inflates sigma (Birge)
    tri = {0: 0.0, 1: 300.0, 2: -200.0}
    loop = pairs(tri, [(0, 1), (1, 2), (0, 2)], sigma=10.0)
    loop["shift_ps"][2] += 90.0
    solved = ts.solve(loop, [0, 1, 2], options=ts.SolveOptions(huber_k=None))
    info = solved.diagnostics["components"][0]
    assert info["dof"] == 1 and info["chi2"] == pytest.approx(90.0 ** 2 / (3 * 10.0 ** 2))
    ratio = info["birge_ratio"]
    assert ratio > 1 and np.allclose(solved.channels["sigma_ps"], ratio * solved.channels["raw_sigma_ps"])
    # each loop edge absorbs a third of the closure error
    assert np.allclose(np.abs(solved.pairs["residual_ps"]), 30.0)
    # a consistent loop is not inflated
    clean = ts.solve(pairs(tri, [(0, 1), (1, 2), (0, 2)], sigma=10.0), [0, 1, 2])
    assert np.allclose(clean.channels["sigma_ps"], clean.channels["raw_sigma_ps"])


def test_no_sentinel_in_correction_arithmetic(tmp_path):
    truth = {0: 100.0, 1: -50.0, 2: 400.0, 3: 0.0, 4: 0.0, 5: 0.0}
    rows = np.concatenate([pairs(truth, [(0, 1), (1, 2), (0, 2)], sigma=5.0),
                           pairs(truth, [(3, 0)], status=tp.ONE_BRANCH)])
    solution = ts.solve(rows, [0, 1, 2, 3, 4], unavailable_channels=[5],
                        options=ts.SolveOptions(precision_gate_ps=1.0))
    path = tmp_path / "lut.csv"
    write_lut(path, solution, {"source": "synthetic"})
    with path.open(newline="") as handle:
        table = list(csv.DictReader(handle))
    for row in table:
        if row["status"] not in ("calibrated", "low statistics"):
            assert row["delay_ps"] == "" and row["sigma_ps"] == "" and row["apply"] == "0"
        assert row["delay_ps"] not in ("0", "0.0", "9999", "9999.0")
    lut = CanonicalLut.read(path)
    # 5 ps pair sigma against a 1 ps gate: solved, applied, but marked low statistics
    assert {ts.CHANNEL_STATUSES[s] for s in lut.rows["status"][:3]} == {"low statistics"}
    assert lut.corrected_dt_ps(1000, 500, 0, 2) == pytest.approx(500 - lut.delay(0) + lut.delay(2))
    for channel in (3, 4, 5, 99):
        with pytest.raises(LutError, match="no applicable delay"):
            lut.corrected_dt_ps(1000, 500, 0, channel)
    # tampering: any edit without the manifest is rejected; a consistent but sentinel-carrying table too
    import hashlib
    text = path.read_text()
    unsolved = next(r for r in table if r["status"] == "insufficient statistics")
    line = ",".join(unsolved[c] for c in table[0])
    fields = line.split(",")
    fields[1:4] = ["9999", "1", "1"]
    sentinel = text.replace(line, ",".join(fields))
    manifest = json.loads((tmp_path / "lut.json").read_text())
    tampered = tmp_path / "tampered.csv"
    for body, digest, message in ((sentinel, manifest["table"]["sha256"], "does not match"),
                                  (sentinel, hashlib.sha256(sentinel.encode()).hexdigest(), "contradicts status"),
                                  (text.replace(",1,0,", ",0,0,", 1), None, "apply flag")):
        tampered.write_text(body, newline="")
        manifest["table"]["sha256"] = digest or hashlib.sha256(body.encode()).hexdigest()
        (tmp_path / "tampered.json").write_text(json.dumps(manifest))
        with pytest.raises(LutError, match=message):
            CanonicalLut.read(tampered)


def run_solution(channels, delays, sigma):
    rows = np.zeros(len(channels), ts.CHANNEL_DTYPE)
    rows["time_channel"], rows["delay_ps"], rows["sigma_ps"], rows["raw_sigma_ps"] = channels, delays, sigma, sigma
    rows["status"], rows["apply"] = ts.CALIBRATED, True
    return ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {"status_counts": {}})


@pytest.mark.fr("008-FR-18")
def test_reproducibility_floor_and_cross_run_outliers():
    rng = np.random.default_rng(75)
    channels = np.arange(1000, 4000)
    truth = rng.uniform(-4000, 4000, len(channels))
    stat = {"a": 6.0, "b": 6.0, "c": 4.0, "d": 3.0}
    floor = {"a": 2.0, "b": 6.0, "c": 6.0, "d": 18.0}
    per_run = {}
    for k, name in enumerate(stat):
        delays = truth + 50.0 * k + rng.normal(0, np.hypot(stat[name], floor[name]), len(channels))
        per_run[name] = run_solution(channels, delays, stat[name])
    per_run["d"].channels["delay_ps"][7] += 2000.0  # one channel jumps in one run
    floors, pairs = ts.reproducibility_floor(per_run)
    assert len(pairs) == 6
    for name in stat:
        assert abs(floors[name] - floor[name]) < 2.5, (name, floors[name])
    # pooled: statistics-weighted mean with its own (small) statistical uncertainty
    weights = {n: 1 / stat[n] ** 2 for n in stat}
    pooled_delay = sum(weights[n] * (per_run[n].channels["delay_ps"] - 50.0 * k)
                       for k, n in enumerate(stat)) / sum(weights.values())
    pooled = run_solution(channels, pooled_delay, 1 / np.sqrt(sum(weights.values())))
    pooled.channels["status"][-5:] = ts.UNOBSERVED
    pooled.channels["apply"][-5:] = False
    pooled.channels["delay_ps"][-5:] = pooled.channels["sigma_ps"][-5:] = np.nan
    adjusted = ts.apply_reproducibility_floor(pooled, per_run, floors, precision_gate_ps=8.0)
    expected = np.sqrt(sum((weights[n] * floors[n]) ** 2 for n in stat)) / sum(weights.values())
    assert adjusted.channels["sigma_ps"][0] == pytest.approx(np.hypot(pooled.channels["sigma_ps"][0], expected))
    assert expected > 8.0 > pooled.channels["sigma_ps"][0]  # the floor, not statistics, decides the gate here
    assert (adjusted.channels["status"][:-5] == ts.LOW_STATISTICS).all()
    assert (adjusted.channels["status"][-5:] == ts.UNOBSERVED).all() and np.isnan(adjusted.channels["sigma_ps"][-5:]).all()
    assert pooled.channels["status"][0] == ts.CALIBRATED  # the input solution is not modified
    assert adjusted.diagnostics["status_counts"]["low statistics"] == len(channels) - 5
    # the jumping channel is listed, stays applied
    flagged = ts.cross_run_outliers(adjusted, per_run, floors, threshold=5.0)
    # the jump also pulls the pooled value, so every run disagrees with it on that channel
    jump = next(f for f in flagged if f["time_channel"] == 1007)
    assert set(jump["runs"]) == set(stat) and jump["runs"]["d"]["difference_ps"] > 900
    assert len(flagged) - 1 < 10 and adjusted.channels["apply"][7]
    # two runs share their excess equally; one run cannot define a floor
    two, _ = ts.reproducibility_floor({n: per_run[n] for n in ("b", "c")})
    assert two["b"] == two["c"] and abs(two["b"] - 6.0) < 2.5
    with pytest.raises(ValueError, match="at least two runs"):
        ts.reproducibility_floor({"a": per_run["a"]})
