"""Spec 012 T1: scanner-neutral walk table and fits (``src.timing.walk``); Cornell bound unchanged."""
import json

import numpy as np
import pytest

from src.cornell.skew import walk as cornell_walk
from src.timing import walk
from helpers import DATA
import timing_walk_helpers as h

GOLDEN = DATA / "golden" / "timing"
MANIFEST = {"fit_kind": "synthetic", "created_utc": "2026-10-27T00:00:00+00:00"}
BINS = {"charge": 20, "share": 10, "doi_ratio": 20, "kev": 20}


class OtherLut(walk.WalkLut):
    model = walk.WalkModel("test-walk-lut/1", "per side t - walk2d[charge, share] - doi[doi_ratio] - kev[kev] (ps)",
                           {"walk2d": ("charge", "share"), "doi": ("doi_ratio",), "kev": ("kev",)}, BINS)


@pytest.mark.fr("012-FR-2")
def test_cornell_table_bytes_unchanged(tmp_path):
    edges, offsets, events = h.cornell_fixed_arrays()
    cornell_walk.WalkLut.from_fit(edges, offsets, events, MANIFEST).write(tmp_path / "fixed.json")
    assert (tmp_path / "fixed.json").read_bytes() == (GOLDEN / "012_t1_cornell_walk_fixed.json").read_bytes()


@pytest.mark.fr("012-FR-2")
def test_cornell_fit_unchanged():
    side1, side2, residual, train = h.cornell_fit_events()
    lut, history = cornell_walk.fit_tables(side1, side2, residual, train, h.CORNELL_FIT_BINS, core_ps=600.0,
                                           iterations=3, minimum=50, manifest=MANIFEST)
    golden = json.loads((GOLDEN / "012_t1_cornell_walk_fit.json").read_text(encoding="utf-8"))
    table = lut.table()
    assert {k: v for k, v in table.items() if k != "blocks"} == {k: v for k, v in golden["table"].items()
                                                                  if k != "blocks"}
    for b, expected in golden["table"]["blocks"].items():
        ours = table["blocks"][b]
        assert (ours["inputs"], ours["edges"], ours["events"], ours["filled_from_nearest"]) == (
            expected["inputs"], expected["edges"], expected["events"], expected["filled_from_nearest"])
        np.testing.assert_allclose(ours["offset_ps"], expected["offset_ps"], rtol=0, atol=1e-6)   # BLAS order
    assert [e["events"] for e in history] == [e["events"] for e in golden["history"]]
    np.testing.assert_allclose([e["rms_ps"] for e in history], [e["rms_ps"] for e in golden["history"]], rtol=1e-9)


@pytest.mark.fr("012-FR-2")
def test_other_model_roundtrip_and_refusal(tmp_path):
    rng = np.random.default_rng(3)
    edges = {"walk2d": [np.linspace(0, 40, 5), np.linspace(0.3, 1, 3)], "doi": [np.linspace(1, 3, 4)],
             "kev": [np.linspace(400, 620, 4)]}
    offsets = {b: rng.normal(0, 30, tuple(len(e) - 1 for e in edges[b])) for b in edges}
    events = {b: np.full(v.shape, 500) for b, v in offsets.items()}
    lut = OtherLut.from_fit(edges, offsets, events, MANIFEST)
    manifest = lut.write(tmp_path / "other.json")
    back = OtherLut.read(tmp_path / "other.json")
    assert manifest["schema"] == "test-walk-lut/1" and back.sha256 == manifest["table"]["sha256"]
    for b in edges:
        np.testing.assert_array_equal(back.offset_ps[b], lut.offset_ps[b])
    with pytest.raises(walk.WalkError, match="schema"):
        cornell_walk.WalkLut.read(tmp_path / "other.json")
    cornell_edges, cornell_offsets, cornell_events = h.cornell_fixed_arrays()
    cornell_walk.WalkLut.from_fit(cornell_edges, cornell_offsets, cornell_events, MANIFEST).write(tmp_path / "c.json")
    with pytest.raises(walk.WalkError, match="schema"):
        OtherLut.read(tmp_path / "c.json")
    with pytest.raises(walk.WalkError, match="output exists"):
        lut.write(tmp_path / "other.json")


@pytest.mark.fr("012-FR-2")
def test_missing_input_and_clamp_flags():
    edges = {"walk2d": [np.array([0.0, 10, 20]), np.array([0.0, 0.5, 1])], "doi": [np.array([1.0, 2, 3])],
             "kev": [np.array([400.0, 500, 600])]}
    offsets = {"walk2d": np.array([[1.0, 2], [3, 4]]), "doi": np.array([10.0, 20]), "kev": np.array([100.0, 200])}
    lut = OtherLut(edges, offsets, {b: np.ones_like(v, np.int64) for b, v in offsets.items()},
                   {b: [] for b in offsets}, {})
    values = {"charge": np.array([5.0, 25.0, np.nan]), "share": np.array([0.7, 0.2, 0.5]),
              "doi_ratio": np.array([1.5, 0.5, 2.5]), "kev": np.array([550.0, 450.0, 450.0])}
    correction, outside = lut.correction(values)
    assert correction[0] == 2 + 10 + 200 and correction[1] == 3 + 10 + 100    # clamped charge and DOI
    assert np.isnan(correction[2])                                             # missing input: no correction
    assert outside["charge"].tolist() == [False, True, False] and outside["doi_ratio"].tolist() == [False, True, False]


@pytest.mark.fr("012-FR-3")
def test_two_branch_fit_recovers_injected_walk():
    side1, side2, dt, near, far, truth, background = h.two_branch_events()
    train = np.arange(len(dt)) % 2 == 0
    lut, model, history, flips = walk.fit_two_branch(side1, side2, dt, near, far, train, BINS, rounds=3,
                                                     core_ps=600.0, iterations=4, minimum=200, manifest=MANIFEST,
                                                     table=OtherLut)
    held = ~train & ~background
    error = model[held] - truth[held]
    assert abs(error.mean()) < 3.0
    assert error.std() < 0.22 * truth[held].std()       # bin discretization of a steep, smooth truth
    # as good as knowing each event's branch: the assignment loses nothing at a 2.14 ns gap and 270 ps
    branch = np.where(np.abs(dt - truth - near) <= np.abs(dt - truth - far), near, far)
    oracle, _ = walk.fit_tables(side1, side2, dt - branch, train, BINS, core_ps=600.0, iterations=4, minimum=200,
                                manifest=MANIFEST, table=OtherLut)
    known, _, _ = walk.walk_difference(oracle, side1, side2)
    assert error.std() <= 1.02 * (known[held] - truth[held]).std()
    assert len(history) == 3 and flips[0] is None and all(isinstance(f, int) for f in flips[1:])
    assert flips[-1] <= flips[1]                                               # assignments settle
    assert 0.45 < history[-1]["near_fraction"] < 0.55
    for b in OtherLut.model.blocks:                                            # zero-mean gauge per block
        weights = lut.events[b]
        assert abs((weights * lut.offset_ps[b]).sum() / weights.sum()) < 1e-6
    assert lut.manifest["core_ps"] == 600.0 and len(lut.manifest["iterations"]) == 4
