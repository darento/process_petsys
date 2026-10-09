"""Fast-reader equivalence and scale checks for LDATInspector (spec 002 T2, T3, T16; spec 007 T21).

Moved from ``scripts/ldat_scale_check.py``: each ``check`` is one test. The spec 001 per-pair
reader, ``process_file_reference``, is the oracle. Every fixture pins its outcome by hand, then the
fast reader must reproduce it exactly: counts, rejection labels, channel counters and retained
columns. Cornell one-time-channel slabs are random in both readers; for those sides only the slab
pair (``slab >> 1``) must match. T16 adds the FR-19 slab rule: every recovery branch is pinned under
both rules in both readers. The fixture files and reference results are built once per system
(``worlds``).

``real_data`` tests (``--real``) read the January 2026-01-19 Cornell acquisitions under
PETSYS_DATA_DIR. ``--whole`` (timing, memory estimates) is retired (spec 007 Clarify).
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from ldat_helpers import CONFIGS, RECOVERY_CASES, fixture_pairs, module_channels, side, write_ldat
from src.ldat_inspector.engine import (_COLUMNS, Settings, apply_calibration, load_setup, merge_results,
                                       process_file_reference)

pytestmark = pytest.mark.fr("002-FR-1")
fr = pytest.mark.fr
# Run name -> fixture file; "mixed, 7-pair prefix" reads the mixed file.
RUNS = ("mixed", "truncated header", "truncated hit", "empty", "weaker minimodule only",
        "zero-energy minimodule", "mixed, 7-pair prefix")
FAILURES = ("truncated header", "truncated hit", "empty")
RECOVER = "recover_non_adjacent"


def fast_reader():
    from src.ldat_inspector.fastread import process_file_fast

    return process_file_fast


def expected(fixtures):
    outcome = Counter()
    for _, count, result, _ in fixtures:
        outcome[result] += count
    accepted = outcome.pop("accepted", 0)
    return accepted, outcome


def compare(reference, fast):
    """Return a list of differences between two FileResult objects."""
    diffs = []
    for name in ("pairs_read", "pairs_accepted", "prefix_limited", "success", "error"):
        if getattr(reference, name) != getattr(fast, name):
            diffs.append(f"{name}: {getattr(reference, name)!r} != {getattr(fast, name)!r}")
    if dict(reference.errors) != dict(fast.errors):
        diffs.append(f"errors: {dict(reference.errors)} != {dict(fast.errors)}")
    for name in ("time_counts", "energy_counts"):
        ref = {sm: dict(c) for sm, c in getattr(reference, name).items()}
        new = {sm: dict(c) for sm, c in getattr(fast, name).items()}
        if ref != new:
            diffs.append(f"{name} differ")
    if set(reference.modules) != set(fast.modules):
        diffs.append(f"SuperModules: {sorted(reference.modules)} != {sorted(fast.modules)}")
        return diffs
    for sm, ref in reference.modules.items():
        new = fast.modules[sm]
        random_side = getattr(new, "random_slab", None)
        if random_side is not None and not np.array_equal(random_side, ref.random_slab):
            diffs.append(f"SM {sm} random_slab differs")
        if not np.array_equal(new.recovered_slab, ref.recovered_slab):
            diffs.append(f"SM {sm} recovered_slab differs")
        for column in (*_COLUMNS, "partner_sm", "partner_mm"):
            a, b = getattr(ref, column), getattr(new, column)
            if a.shape != b.shape:
                diffs.append(f"SM {sm} {column}: length {a.shape} != {b.shape}")
                continue
            if column in ("calibration_key", "partner_calibration_key"):
                same = (a >> 1) == (b >> 1)  # time channel and slab pair
                if column == "calibration_key" and random_side is not None:
                    same &= (a == b) | random_side
                elif column == "calibration_key":
                    same &= a == b
                ok = bool(same.all())
            else:
                ok = np.array_equal(a, b)
            if not ok:
                diffs.append(f"SM {sm} {column} differs")
    return diffs


def reindexed(table, index):
    columns = dict(table.columns)
    columns["file_index"] = np.full(len(table), index, dtype=columns["file_index"].dtype)
    return type(table)(columns)


def build_world(system, root):
    base = Settings(str(CONFIGS[system]), "", system, max_pairs=10_000,
                    min_channels=4, min_channel_energy=0.2, calibrated=False)
    setup = load_setup(base)
    fixtures = fixture_pairs(system, setup)
    pairs = [builder(i) for _, count, _, builder in fixtures for i in range(count)]
    files = {"mixed": root / "mixed.ldat", "truncated header": root / "th.ldat",
             "truncated hit": root / "tt.ldat", "empty": root / "empty.ldat",
             "weaker minimodule only": root / "mm.ldat", "zero-energy minimodule": root / "zero.ldat"}
    write_ldat(files["mixed"], pairs)
    write_ldat(files["truncated header"], pairs[:5], tail=b"\x04")
    write_ldat(files["truncated hit"], pairs[:5], tail=struct.pack("2B", 2, 2) + b"\x00" * 20)
    files["empty"].write_bytes(b"")
    weaker = next(b for name, _, _, b in fixtures if name == "second minimodule weaker")
    write_ldat(files["weaker minimodule only"], [weaker(i) for i in range(4)])
    zero_pair = lambda i: (side(setup, 0, 0, {3: 12.0}, [0.0] * 4, 10 ** 12 + i),
                           side(setup, 1, 0, {3: 12.0, 2: 5.0}, [30.0, 20.0, 10.0, 5.0], 10 ** 12 + i))
    write_ldat(files["zero-energy minimodule"], [zero_pair(i) for i in range(4)])
    files["mixed, 7-pair prefix"] = files["mixed"]

    settings = {name: base for name in RUNS}
    settings["zero-energy minimodule"] = replace(base, min_channel_energy=0.0)
    settings["mixed, 7-pair prefix"] = replace(base, max_pairs=7)
    results = {name: process_file_reference(str(files[name]), settings[name], 0) for name in RUNS}
    recover = replace(base, slab_rule=RECOVER)
    world = SimpleNamespace(system=system, root=root, base=base, recover=recover, setup=setup,
                            fixtures=fixtures, pairs=pairs, files=files, settings=settings, results=results,
                            mixed_recover=process_file_reference(str(files["mixed"]), recover, 0))
    if system == "CORNELL":
        def build(time, i):
            stamp = 2_000_000_000_000 + i * 5_000_000
            full = [30.0, 20.0, 10.0, 5.0]
            return (side(setup, 0, 0, time, full, stamp),
                    side(setup, 1, 0, {3: 12.0, 2: 5.0}, full, stamp + 1_234))

        world.non_adjacent = root / "non_adjacent.ldat"
        world.rows = [case for case in RECOVERY_CASES for _ in range(case[1])]
        write_ldat(world.non_adjacent, [build(case[2], i) for i, case in enumerate(world.rows)])
        world.legacy, world.recovered = (process_file_reference(str(world.non_adjacent), rule, 0)
                                         for rule in (base, recover))
    return world


@pytest.fixture(scope="module")
def worlds():
    """System -> fixture world, built on first use under a tempfile root, as in the script."""
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
def imas(worlds):
    return worlds("IMAS")


def test_fast_reader_available():
    assert callable(fast_reader())


# --- reference reader on the fixtures ----------------------------------------

def test_reference_mixed_fixture_outcomes(world):
    accepted, errors = expected(world.fixtures)
    ref = world.results["mixed"]
    assert (ref.pairs_read, ref.pairs_accepted, dict(ref.errors)) == (len(world.pairs), accepted, dict(errors))


def test_reference_stores_both_sides_per_accepted_pair(world):
    ref = world.results["mixed"]
    accepted, _ = expected(world.fixtures)
    assert sum(map(len, ref.modules.values())) == 2 * accepted and set(ref.modules) == {0, 1}


def test_cornell_fixture_slabs_cover_edges_adjacent_and_random(cornell):
    slabs = sorted(set((cornell.results["mixed"].modules[0].calibration_key & 31).tolist()))
    assert {0, 6, 7, 14}.issubset(slabs) and set(slabs) <= {0, 6, 7, 14}, f"SM 0 slabs {slabs}"


def test_reference_prefix_is_labelled(world):
    prefix = world.results["mixed, 7-pair prefix"]
    assert prefix.pairs_read == 7 and prefix.prefix_limited


@pytest.mark.parametrize("name", FAILURES)
def test_reference_failure_has_no_partial_data(world, name):
    result = world.results[name]
    assert not result.success and not result.modules and result.pairs_accepted == 0, result.error or ""


def test_reference_counters_exclude_the_unselected_minimodule(world):
    result = world.results["weaker minimodule only"]
    mm1_time, mm1_energy = module_channels(world.setup, 0, 1)
    counted = set(result.time_counts.get(0, ())) | set(result.energy_counts.get(0, ()))
    assert result.pairs_accepted == 4 and not counted & set(mm1_time + mm1_energy)


def test_reference_zero_energy_minimodule_rejected_at_doi(world):
    # DOI (sum / max energy) divides by zero before the "energy > 0" check runs.
    zero = world.results["zero-energy minimodule"]
    assert zero.pairs_accepted == 0 and dict(zero.errors) == {"ZeroDivisionError": 4}, f"{dict(zero.errors)}"


# --- T3: compact SM-sorted side table with partner SM/mM ----------------------

@pytest.fixture(scope="module")
def merged(world):
    """The mixed result merged with the weaker-minimodule result as file 1, and its calibration."""
    mixed, weaker = world.results["mixed"], world.results["weaker minimodule only"]
    dataset = merge_results(world.base, [mixed, replace(weaker, index=1, table=reindexed(weaker.table, 1))],
                            world.setup)
    calibration = world.root / "calibration.txt"
    keys = np.unique(dataset.table.calibration_key)
    if world.system == "CORNELL":
        calibration.write_text("ID(t_ch, slab)\tmu\n" + "".join(
            f"({k >> 5}, {k & 31})\t100\n" for k in keys.tolist()), encoding="utf-8")
    else:
        calibration.write_text("ID\tmu\n" + "".join(f"{t}\t100\n" for t in np.unique(keys >> 5).tolist()),
                               encoding="utf-8")
    return SimpleNamespace(dataset=dataset, calibrated=apply_calibration(dataset, str(calibration), True),
                           pairs=mixed.pairs_accepted + weaker.pairs_accepted)


@fr("002-FR-2", "002-FR-11")
def test_t3_partner_rows_pair_up(world):
    table = world.results["mixed"].table
    rows = np.arange(len(table))
    assert np.array_equal(table.partner[table.partner], rows) and not np.any(table.partner == rows)


@fr("002-FR-2", "002-FR-11")
def test_t3_partner_sm_and_mm_on_every_side(world):
    sm0, sm1 = world.results["mixed"].modules[0], world.results["mixed"].modules[1]
    assert ((sm0.partner_sm == 1).all() and (sm1.partner_sm == 0).all()
            and (sm0.partner_mm == 0).all() and (sm1.partner_mm == 0).all())


@fr("002-FR-2", "002-FR-11")
def test_t3_partner_values_come_from_the_partner_row(world):
    mixed = world.results["mixed"]
    table, sm0 = mixed.table, mixed.modules[0]
    partner_rows = table.partner[:len(sm0)]  # SM 0 is the first slice of the sorted table
    assert (np.array_equal(sm0.partner_timestamp, table.timestamp[partner_rows])
            and np.array_equal(sm0.partner_energy, table.energy[partner_rows])
            and np.array_equal(sm0.partner_calibration_key, table.calibration_key[partner_rows]))


@fr("002-FR-2", "002-FR-11")
def test_t3_two_file_merge_keeps_pairs_sm_order_and_file_order(merged):
    dataset, table = merged.dataset, merged.dataset.table
    assert (len(table) == 2 * merged.pairs
            and np.array_equal(table.partner[table.partner], np.arange(len(table)))
            and np.all(np.diff(table.sm) >= 0)
            and all(np.all(np.diff(d.file_index) >= 0) for d in dataset.modules.values())
            and all(f.table is None for f in dataset.files))


@fr("002-FR-2", "002-FR-11")
def test_t3_calibration_replaces_only_the_energy_column(merged):
    calibrated, table = merged.calibrated, merged.dataset.table
    assert (calibrated.table.columns is table.columns and calibrated.table.energy is not table.raw_energy
            and np.allclose(calibrated.table.energy, table.raw_energy * 511 / 100)
            and np.allclose(calibrated.modules[0].partner_energy, calibrated.table.energy[
                calibrated.table.partner[:len(calibrated.modules[0])]]))


@fr("002-FR-2", "002-FR-11")
def test_t3_stored_bytes_per_side_with_calibrated_energy(merged):
    table = merged.calibrated.table
    per_side = table.nbytes / len(table)
    assert per_side <= 64, f"{per_side:.1f} B/side ({table.nbytes:,} B for {len(table):,} sides)"


# --- T16 (FR-19): the non-adjacent recovery rule ------------------------------

@fr("002-FR-19")
def test_t16_legacy_is_the_default_slab_rule(world):
    assert Settings(world.base.config_path, "", world.system).slab_rule == "legacy" == world.base.slab_rule


@fr("002-FR-19")
def test_t16_slab_rule_does_not_change_imas_ingest(imas):
    recovered = imas.mixed_recover
    assert (not compare(imas.results["mixed"], recovered) and recovered.success
            and not any(d.recovered_slab.any() for d in recovered.modules.values()))


@fr("002-FR-19")
def test_t16_cornell_recovery_accepts_the_mixed_non_adjacent_pairs_only(cornell):
    legacy, recovered = cornell.results["mixed"], cornell.mixed_recover
    non_adjacent = sum(count for _, count, outcome, _ in cornell.fixtures if outcome == "unresolved Cornell slab")
    errors = dict(legacy.errors)
    errors.pop("unresolved Cornell slab", None)
    assert (recovered.pairs_accepted == legacy.pairs_accepted + non_adjacent
            and dict(recovered.errors) == errors
            and int(recovered.modules[0].recovered_slab.sum()) == non_adjacent
            and not recovered.modules[1].recovered_slab.any()), \
        f"accepted {legacy.pairs_accepted} -> {recovered.pairs_accepted}, errors {dict(recovered.errors)}"


@fr("002-FR-19")
def test_t16_cornell_legacy_rejects_non_adjacent_keeps_edge_rule_and_adjacent_tie(cornell):
    legacy, rows = cornell.legacy, cornell.rows
    kept = [case for case in rows if case[3] is not None]
    sm0 = legacy.modules[0]
    assert (legacy.pairs_accepted == len(kept)
            and dict(legacy.errors) == {"unresolved Cornell slab": len(rows) - len(kept)}
            and (sm0.calibration_key & 31).tolist() == [case[3] for case in kept]
            and not sm0.random_slab.any() and not sm0.recovered_slab.any()), \
        f"accepted {legacy.pairs_accepted}/{len(rows)}, {dict(legacy.errors)}"


@fr("002-FR-19")
@pytest.mark.parametrize("case", RECOVERY_CASES, ids=[case[0] for case in RECOVERY_CASES])
def test_t16_cornell_recover(cornell, case):
    name, _, _, legacy_slab, expected_slabs, random_side = case
    recovered, rows = cornell.recovered, cornell.rows
    sm0 = recovered.modules[0]
    slabs = (sm0.calibration_key & 31).tolist() if recovered.success else []
    at = [i for i, row in enumerate(rows) if row[0] == name]
    got = sorted({slabs[i] for i in at}) if len(slabs) == len(rows) else []
    ok = (set(got) <= expected_slabs and all(bool(sm0.random_slab[i]) == random_side for i in at)
          and all(bool(sm0.recovered_slab[i]) == (legacy_slab is None) for i in at))
    if random_side:
        ok &= set(got) == expected_slabs  # both sides of the coin within 24 pairs
    assert ok, f"slabs {got}, random {sorted({bool(sm0.random_slab[i]) for i in at}) if got else '-'}"


@fr("002-FR-19")
def test_t16_cornell_recover_accepts_every_pair_det2_unchanged(cornell):
    recovered = cornell.recovered
    assert (recovered.pairs_accepted == len(cornell.rows) and not recovered.errors
            and (recovered.modules[1].calibration_key & 31 == 6).all()
            and not recovered.modules[1].recovered_slab.any())


@fr("002-FR-19")
@pytest.mark.parametrize("label", ["non-adjacent fixture, legacy", "non-adjacent fixture, recover",
                                   "mixed, recover"])
def test_t16_cornell_fast_equals_reference(cornell, label):
    settings, target, reference = {
        "non-adjacent fixture, legacy": (cornell.base, cornell.non_adjacent, cornell.legacy),
        "non-adjacent fixture, recover": (cornell.recover, cornell.non_adjacent, cornell.recovered),
        "mixed, recover": (cornell.recover, cornell.files["mixed"], cornell.mixed_recover)}[label]
    diffs = compare(reference, fast_reader()(str(target), settings, 0))
    assert not diffs, "; ".join(diffs[:4])


# --- fast reader = reference ---------------------------------------------------

@pytest.mark.parametrize("name", RUNS)
def test_fast_equals_reference(world, name):
    diffs = compare(world.results[name], fast_reader()(str(world.files[name]), world.settings[name], 0))
    assert not diffs, "; ".join(diffs[:4])


# --- real Cornell data ----------------------------------------------------------
# Data: PETSYS_DATA_DIR/Cornell/full_system, the six January 2026-01-19 compact acquisitions below.
# Map: the January map through the tracked copy of the operator config (CONFIGS["CORNELL"]).
# No calibration (raw a.u.). Cuts: >= 4 energy channels, >= 0.2 a.u. per channel.
# Spec 001 T9: first 80,000 pairs of each file -> (accepted, min channels, unresolved slab, ValueError).

CORNELL_PATTERN = ("Cornell/full_system/20260119_2NaSourcesAxialSeparated_vBiasCompDiscCalibAdjusted2hits_300s_"
                   "coincCompact11s_{:08d}.ldat")
T9_PREFIX = {3: (62_460, 11_691, 5_457, 392), 4: (62_253, 11_833, 5_510, 404),
             5: (62_214, 11_922, 5_472, 392), 6: (62_365, 11_818, 5_460, 357),
             7: (62_576, 11_656, 5_352, 416), 8: (62_393, 11_854, 5_331, 422)}


def cornell_settings(pairs):
    return Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=pairs,
                    min_channels=4, min_channel_energy=0.2, calibrated=False)


def counts(result):
    return (result.pairs_read, result.pairs_accepted, result.errors.get("min channels", 0),
            result.errors.get("unresolved Cornell slab", 0), result.errors.get("ValueError", 0))


@pytest.fixture(scope="module")
def t9_cache():
    return {}


@pytest.fixture
def t9(real_data_file, t9_cache):
    """File number -> path, and the reference results on the 80,000-pair prefixes (computed once)."""
    paths = {n: real_data_file(CORNELL_PATTERN.format(n)) for n in T9_PREFIX}
    if not t9_cache:
        settings = cornell_settings(80_000)
        with ProcessPoolExecutor(max_workers=len(paths)) as pool:
            futures = {n: pool.submit(process_file_reference, str(p), settings, n) for n, p in paths.items()}
            t9_cache.update(paths=paths, settings=settings, results={n: f.result() for n, f in futures.items()})
    return SimpleNamespace(**t9_cache)


@pytest.fixture
def t16_real(t9, t9_cache):
    """Spec 002 T16: the recovery rule on the 80,000-pair 00000003 prefix (reference, computed once)."""
    if "recover" not in t9_cache:
        recover = replace(t9.settings, slab_rule=RECOVER)
        t9_cache.update(recover=recover, recovered=process_file_reference(str(t9.paths[3]), recover, 3))
    return SimpleNamespace(**t9_cache)


class TestRealCornell:
    pytestmark = [pytest.mark.real_data, fr("007-FR-4")]

    @pytest.mark.parametrize("n", T9_PREFIX, ids=lambda n: f"{n:08d}")
    def test_reference_reproduces_spec001_t9(self, t9, n):
        accepted, min_ch, slab, value = T9_PREFIX[n]
        got = counts(t9.results[n])
        assert got == (80_000, accepted, min_ch, slab, value), f"read/accepted/min-ch/slab/ValueError = {got}"

    @pytest.mark.parametrize("n", T9_PREFIX, ids=lambda n: f"{n:08d}")
    def test_fast_reader_reproduces_spec001_t9(self, t9, n):
        accepted, min_ch, slab, value = T9_PREFIX[n]
        result = fast_reader()(str(t9.paths[n]), t9.settings, n)
        got = counts(result)
        assert got == (80_000, accepted, min_ch, slab, value) and not compare(t9.results[n], result), \
            f"read/accepted/min-ch/slab/ValueError = {got}"

    @pytest.mark.slow
    def test_fast_equals_reference_on_200k_prefix(self, t9):
        settings = cornell_settings(200_000)
        diffs = compare(process_file_reference(str(t9.paths[3]), settings, 0),
                        fast_reader()(str(t9.paths[3]), settings, 0))
        assert not diffs, "; ".join(diffs[:4])

    @fr("002-FR-19")
    def test_t16_fast_equals_reference_recover_rule(self, t16_real):
        diffs = compare(t16_real.recovered, fast_reader()(str(t16_real.paths[3]), t16_real.recover, 3))
        assert not diffs, "; ".join(diffs[:4])

    @fr("002-FR-19")
    def test_t16_unresolved_pairs_accepted_or_fail_later_nothing_else_moves(self, t16_real):
        # A formerly unresolved pair is now accepted or fails a later check (e.g. on its other side).
        legacy, recovered = t16_real.results[3], t16_real.recovered
        unresolved = legacy.errors.get("unresolved Cornell slab", 0)
        later = {k: recovered.errors.get(k, 0) - legacy.errors.get(k, 0) for k in recovered.errors}
        assert ("unresolved Cornell slab" not in recovered.errors and min(later.values(), default=0) >= 0
                and set(legacy.errors) - {"unresolved Cornell slab"} <= set(recovered.errors)
                and recovered.pairs_accepted - legacy.pairs_accepted + sum(later.values()) == unresolved), \
            (f"accepted {legacy.pairs_accepted:,} -> {recovered.pairs_accepted:,} pairs of {unresolved:,} "
             f"unresolved; later rejections {later}")
