"""Native time-offset roundtrip on matched LDAT events (spec 008 T9a, FR-3/FR-17).

A synthetic "conversion" draws each hit's exact time T (ps, double) once; the
uncorrected file stores trunc(T) and the corrected one trunc(T - b) for listed
channels, as PETsys's ProcessHit + compact writer do. In QDC mode ProcessHit
also lowers a listed hit's energy by b/T (T = 5000 ps), because the
integration end keeps the uncorrected time. Records may change detector order
or exist in only one conversion (grouping/window changes).
"""
import json

import numpy as np
import pytest

from cornell_timing_helpers import T0, write
from src.cornell import timing_roundtrip as tr
from src.utils import get_absolute_id

pytestmark = pytest.mark.fr("008-FR-17", "008-FR-3")
CHANNELS = [get_absolute_id(p, s, c, k) for p, s, c, k in
            [(0, 0, 0, 0), (0, 0, 0, 13), (0, 1, 4, 34), (1, 0, 2, 52), (4, 3, 1, 7), (5, 0, 11, 61),
             (6, 2, 0, 20), (2, 0, 8, 40), (0, 0, 1, 3), (4, 0, 0, 59)]]
OFFSETS = {CHANNELS[k]: v for k, v in enumerate([-4647.25, 2436.125, 803.5, 1642.0, -1554.75, 11.25])}


def physical_events(rng, count):
    """Per event: two sides of (channel, exact time, energy) hits, sorted by energy as PETsys writes them."""
    events = []
    for k in range(count):
        start = T0 + k * 1_000_000 + rng.uniform(0, 1000)
        sides = []
        n1, n2 = rng.integers(1, 4, 2)
        chosen = rng.choice(CHANNELS, n1 + n2, replace=False)  # a channel fires once per event
        for chans in (chosen[:n1], chosen[n1:]):
            hits = [(int(c), start + rng.normal(0, 300), float(np.float32(rng.uniform(1, 60)))) for c in chans]
            sides.append(sorted(hits, key=lambda h: -h[2]))
        events.append(sides)
    return events


def convert(events, factor=1.0, *, swap=(), drop=(), extra=(), energy_factor=None):
    """Records as PETsys would write them with offsets scaled by ``factor`` (1 = applied once).

    Listed energies move by ``-energy_factor * b / 5000`` (default ``factor``, as ProcessHit does).
    """
    energy_factor = factor if energy_factor is None else energy_factor
    records = []
    for k, sides in enumerate(events):
        if k in drop:
            continue
        out = [[(int(np.floor(t - factor * np.float32(OFFSETS.get(c, 0.0)))),
                 float(np.float32(e - energy_factor * np.float32(OFFSETS.get(c, 0.0)) / 5000)), c)
                for c, t, e in side] for side in sides]
        records.append((out[1], out[0]) if k in swap else (out[0], out[1]))
    return records + list(extra)


def hits_of(path, **kwargs):
    return tr.read_hits(path, **kwargs)[0]


def test_applies_correction_once(tmp_path):
    rng = np.random.default_rng(9)
    events = physical_events(rng, 3000)
    a = write(tmp_path / "a.ldat", convert(events, 0.0))
    once = write(tmp_path / "once.ldat", convert(events, 1.0))
    report = tr.compare(hits_of(a), hits_of(once), OFFSETS)
    hyp = report["hypotheses_on_listed_hits"]
    assert report["passed"] and hyp["applied once"] == hyp["hits"] > 1000
    assert hyp["not applied"] + hyp["applied twice"] + hyp["sign flipped"] <= hyp["hits"] // 1000  # chance coincidences
    low, high = report["hits"]["residual_ps_min_max"]
    assert -1 <= low and high <= 1  # integer-ps truncation of t and t - b
    assert report["unlisted_channel_hits"]["matched"] > 1000 and report["unlisted_channel_hits"]["changed"] == 0
    energy = report["energy"]
    assert energy["unlisted_changed"] == 0 and energy["listed_within_model"] == energy["listed_matched"] > 1000
    assert energy["fitted_clock_ps"] == pytest.approx(5000, rel=1e-3)
    assert report["channels"]["max_abs_median_minus_offset_ps"] <= 1
    assert report["channels"]["listed_seen"] == len(OFFSETS)
    assert report["records"]["only_corrected"] == report["records"]["only_uncorrected"] == 0
    # wrong conversions are diagnosed, never passed
    for name, factor in (("applied twice", 2.0), ("sign flipped", -1.0), ("not applied", 0.0)):
        bad = write(tmp_path / f"{factor}.ldat", convert(events, factor))
        result = tr.compare(hits_of(a), hits_of(bad), OFFSETS)
        counts = result["hypotheses_on_listed_hits"]
        assert not result["passed"] and counts["applied once"] <= counts["hits"] // 1000, name
        assert counts[name] == counts["hits"], name
    # listed energies unchanged (no PETsys side effect) or an unlisted energy changed: not the model
    for kwargs in ({"energy_factor": 0.0}, {"energy_factor": 3.0}):
        bad = write(tmp_path / "energy.ldat", convert(events, 1.0, **kwargs))
        result = tr.compare(hits_of(a), hits_of(bad), OFFSETS)
        assert not result["passed"] and result["energy"]["listed_outside_model"] > 0.5 * result["energy"]["listed_matched"]
        (tmp_path / "energy.ldat").unlink()
    records = convert(events, 1.0)
    unlisted = next(k for k in range(len(records) // 2, len(records)) if records[k][0][0][2] not in OFFSETS)
    (t, e, c), *rest = records[unlisted][0]
    records[unlisted] = ([(t, e + 1.0, c), *rest], records[unlisted][1])
    result = tr.compare(hits_of(a), hits_of(write(tmp_path / "unlisted.ldat", records)), OFFSETS)
    assert not result["passed"] and result["energy"]["unlisted_changed"] == 1


def test_membership_changes_and_endpoint_swaps(tmp_path, capsys):
    rng = np.random.default_rng(10)
    events = physical_events(rng, 2000)
    # PETsys can write one photon into several coincidence records: identical hit copies
    for k in range(50, 2000, 101):
        used = {c for side in events[k] for c, _, _ in side}
        other = next(c for c in CHANNELS if c not in used)
        events.append([events[k][0], [(other, events[k][0][0][1] + 700.0, 5.0)]])
    swap, drop = set(range(5, 2000, 37)), {100, 700, 701, 1500}
    gained = [([(T0 + 1000 * 1_000_000 + 500, 7.0, CHANNELS[1])], [(T0 + 1000 * 1_000_000 + 900, 9.0, CHANNELS[2])])]
    a = write(tmp_path / "a.ldat", convert(events, 0.0, drop={300, 1200}))
    b = write(tmp_path / "b.ldat", convert(events, 1.0, swap=swap, drop=drop, extra=gained))
    report = tr.compare(hits_of(a), hits_of(b), OFFSETS)
    assert report["passed"]
    assert report["endpoints"]["detector_order_swapped"] == len(swap) and report["endpoints"]["dt_mismatch"] == 0
    # records only in one conversion are counted, not failures (edges excluded by the compared span)
    assert report["records"]["only_uncorrected"] == len(drop) and report["records"]["only_corrected"] == 3
    assert report["records"]["matched"] == report["corrected_records"] - 3
    assert report["records"]["resolved_with_duplicate_hits"] >= 2 * 20  # both copies of 20 shared photons
    # CLI: bounded corrected prefix, uncorrected read until the span is passed
    table = tmp_path / "t.tsv"
    table.write_text("# native\n" + "".join(f"{(c >> 17)}\t{(c >> 12) & 31}\t{(c >> 6) & 63}\t{c & 63}\t{v}\n"
                                             for c, v in OFFSETS.items()))
    assert tr.main([str(a), str(b), str(table), "--records", "800"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["passed"] and printed["files"]["records_read"]["corrected"] == 800
    assert printed["files"]["records_read"]["uncorrected"] < 2000
