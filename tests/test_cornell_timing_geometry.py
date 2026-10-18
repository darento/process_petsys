"""Analytic source truth and declared Cornell frame policies (008 T2a)."""
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

from cornell_timing_helpers import CONFIG
from src.cornell.skew.geometry import (AnnularSource, C_MM_PER_PS, GeometryError,
                                       build_detector_frames, build_faceted_detector_frames,
                                       geometric_tof_ps,
                                       legacy_opposition_accepts, shell_intervals)
from src.mapping_generator import map_factory, ChannelType

pytestmark = pytest.mark.fr("008-FR-2", "008-FR-5")
ROOT = Path(__file__).resolve().parents[1]


def shell(**kwargs):
    values = dict(inner_radius_mm=137.5, outer_radius_mm=142.5,
                  axial_min_mm=-130.0, axial_max_mm=130.0,
                  center_mm=(0, 0, 0), axis=(0, 0, 1))
    return AnnularSource(**(values | kwargs))


def config_and_map():
    config = yaml.safe_load(CONFIG.read_text())
    local, modules, types, _ = map_factory(str(ROOT / config["map_file"]))
    return config, local, modules, types


@pytest.mark.fr("bug-timing-tests-untracked-config")
def test_timing_config_and_its_map_are_tracked():
    """A fresh clone failed 45 timing tests: they read the gitignored configs/cornell_full_system.yaml."""
    config = yaml.safe_load(CONFIG.read_text())
    for path in (CONFIG, ROOT / config["map_file"]):
        relative = path.resolve().relative_to(ROOT).as_posix()
        assert subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT,
                              capture_output=True).returncode == 0, relative


@pytest.mark.parametrize("origin", [(48.0, 48.0), (51.61, 51.61)])
@pytest.mark.parametrize("axes", ["time-axial", "time-transverse"])
def test_transforms_match_nominal_geometry(origin, axes):
    config, _, modules, _ = config_and_map()
    frames = build_detector_frames(config, modules, local_origin_mm=origin,
                                   local_axes=axes, axial_sign=-1, crystal_thickness_mm=20)
    assert set(frames) == set(range(30))
    for sm, frame in frames.items():
        cassette, z_index = divmod(sm, 3)
        y, x = config["ring_yx"][cassette]
        assert frame.to_global(*origin) == pytest.approx((x, y, config["ring_z"][z_index]))
        assert frame.crystal_thickness_mm == 20 and "pending" in frame.qualification
    # Opposite cassette entries 0/5 correspond to hardware SM 0/15 at same z.
    assert frames[0].center_mm == (160, 0, -102)
    assert frames[15].center_mm == (-160, 0, -102)
    # A local displacement moves along only its declared physical axis.
    moved = frames[2].to_global(origin[0] + 12, origin[1] - 8)
    expected = (160, -8, 90) if axes == "time-axial" else (160, 12, 110)
    assert moved == pytest.approx(expected)
    assert frames[17].to_global(origin[0] + 12, origin[1] - 8) == pytest.approx(
        (-160, -expected[1], expected[2]))


def test_selected_map_midpoint_differs_from_legacy_origin():
    _, local, modules, types = config_and_map()
    for sm in range(30):
        for kind, coordinate in ((ChannelType.TIME, 0), (ChannelType.ENERGY, 1)):
            values = [local[ch][coordinate] for ch in local if modules[ch][0] == sm and kind in types[ch]]
            assert (min(values) + max(values)) / 2 == pytest.approx(51.61)
            assert (min(values) + max(values)) / 2 != 48
    # Geometric midpoint is map evidence, not a measured position calibration.


def test_finite_shell_intervals():
    i, j = (160, 0, 0), (-160, 0, 0)
    intervals = shell_intervals(i, j, shell())
    assert len(intervals) == 2
    assert [(x.fraction_start, x.fraction_end) for x in intervals] == pytest.approx(
        [(17.5 / 320, 22.5 / 320), (297.5 / 320, 302.5 / 320)])
    assert [(x.tof_start_ps, x.tof_end_ps) for x in intervals] == pytest.approx(
        [(-285 / C_MM_PER_PS, -275 / C_MM_PER_PS),
         (275 / C_MM_PER_PS, 285 / C_MM_PER_PS)])
    assert intervals[0].tof_end_ps - intervals[0].tof_start_ps == pytest.approx(10 / C_MM_PER_PS)
    # Physical middle-radius peaks remain floating point, not integer-truncated.
    assert geometric_tof_ps((140, 0, 0), i, j) == pytest.approx(-280 / C_MM_PER_PS)
    assert geometric_tof_ps((-140, 0, 0), i, j) == pytest.approx(280 / C_MM_PER_PS)
    reverse = shell_intervals(j, i, shell())
    for forward, backward in zip(intervals, reversed(reverse)):
        assert backward.fraction_start == pytest.approx(1 - forward.fraction_end)
        assert backward.tof_start_ps == pytest.approx(-forward.tof_end_ps)
        assert backward.tof_end_ps == pytest.approx(-forward.tof_start_ps)


def test_axial_edges_parallel_lines_tangency_and_segment_clip():
    assert shell_intervals((160, 0, 131), (-160, 0, 131), shell()) == ()
    clipped = shell_intervals((160, 0, 0), (-160, 0, 100), shell(axial_min_mm=0, axial_max_mm=6))
    assert len(clipped) == 1
    assert clipped[0].fraction_start == pytest.approx(17.5 / 320)
    assert clipped[0].fraction_end == pytest.approx(0.06)
    parallel = shell_intervals((140, 0, -200), (140, 0, 200), shell())
    assert len(parallel) == 1
    assert (parallel[0].fraction_start, parallel[0].fraction_end) == pytest.approx((0.175, 0.825))
    assert shell_intervals((0, 0, -200), (0, 0, 200), shell()) == ()
    assert shell_intervals((160, 142.5, 0), (-160, 142.5, 0), shell()) == ()
    # Inner-cavity tangency doesn't create a spurious zero-width missing segment.
    tangent = shell_intervals((160, 137.5, 0), (-160, 137.5, 0), shell())
    assert len(tangent) == 1
    assert tangent[0].fraction_end > tangent[0].fraction_start
    # The finite detector segment never extrapolates into a source elsewhere.
    assert shell_intervals((160, 0, 0), (150, 0, 0), shell()) == ()
    inside = shell_intervals((142, 0, 0), (138, 0, 0), shell())
    assert len(inside) == 1 and (inside[0].fraction_start, inside[0].fraction_end) == (0, 1)


def test_miscentered_rotated_source_and_asymmetric_detector_radii():
    original = shell_intervals((160, 0, 0), (-170, 0, 0), shell())
    # Unequal detector distance shifts both physical peak midpoints by -10/c.
    means = [(x.tof_start_ps + x.tof_end_ps) / 2 for x in original]
    assert means == pytest.approx([-290 / C_MM_PER_PS, 270 / C_MM_PER_PS])
    assert sum(means) / 2 == pytest.approx(-10 / C_MM_PER_PS)
    # Rigid rotation x→z, z→-x plus translation preserves every flight distance.
    center = np.array([20., -30., 5.])
    rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    transformed = shell_intervals(rotation @ np.array([160., 0., 0.]) + center,
                                  rotation @ np.array([-170., 0., 0.]) + center,
                                  shell(center_mm=tuple(center), axis=(-1, 0, 0)))
    assert [(x.tof_start_ps, x.tof_end_ps) for x in transformed] == pytest.approx(
        [(x.tof_start_ps, x.tof_end_ps) for x in original])


def test_oblique_shell_tof_agrees_with_flight_distances():
    # A noncentral, oblique LOR: flight distances provide independent TOF truth.
    source = shell(center_mm=(5, -7, 11))
    i, j = np.array([165., 23., -59.]), np.array([-155., 23., 81.])
    intervals = shell_intervals(i, j, source)
    assert len(intervals) == 2
    for interval in intervals:
        for fraction in np.linspace(interval.fraction_start, interval.fraction_end, 7):
            point = i + fraction * (j - i)
            expected_tof = interval.tof_start_ps + (fraction - interval.fraction_start) * (
                interval.tof_end_ps - interval.tof_start_ps) / (interval.fraction_end - interval.fraction_start)
            assert geometric_tof_ps(point, i, j) == pytest.approx(expected_tof, abs=1e-9)
            if interval.fraction_start < fraction < interval.fraction_end:
                distance = math.hypot(*(point[:2] - source.center_mm[:2]))
                assert source.inner_radius_mm < distance < source.outer_radius_mm
                assert source.axial_min_mm < point[2] - source.center_mm[2] < source.axial_max_mm


def test_legacy_angle_boundary_and_axial_obliquity():
    i = (160, 0, 0)
    at_angle = lambda deg: (160 * math.cos(math.radians(deg)), 160 * math.sin(math.radians(deg)), 0)
    assert legacy_opposition_accepts(i, at_angle(165))
    assert legacy_opposition_accepts(i, at_angle(165.001))
    assert not legacy_opposition_accepts(i, at_angle(164.999))
    assert legacy_opposition_accepts(i, (-160, 0, 0))
    assert type(legacy_opposition_accepts(i, (-160, 0, 0))) is bool
    # XY positions are opposite; same-z outer ring fails the actual 3D angle.
    assert not legacy_opposition_accepts((160, 0, 102), (-160, 0, 102))
    assert legacy_opposition_accepts((160, 0, -102), (-160, 0, 102))
    # Angle rejection is distinct from a geometrically valid shell crossing.
    assert len(shell_intervals((160, 0, 102), (-160, 0, 102), shell())) == 2


def test_rejects_unknown_geometry():
    config, _, modules, _ = config_and_map()
    options = dict(local_origin_mm=(48, 48), local_axes="time-axial", axial_sign=-1, crystal_thickness_mm=20)
    for invalid in ({"local_origin_mm": None}, {"local_axes": "unknown"}, {"axial_sign": 0},
                    {"crystal_thickness_mm": 0}, {"crystal_thickness_mm": None}):
        with pytest.raises(GeometryError):
            build_detector_frames(config, modules, **(options | invalid))
    with pytest.raises(GeometryError, match="selected map"):
        build_detector_frames(config, {11: (999, 0)}, **options)
    for changed in ({"ring_r": 150}, {"ring_yx": {}}, {"ring_z": [0, 0]}, {"ring_z": [float('nan')]}):
        with pytest.raises(GeometryError):
            build_detector_frames(config | changed, modules, **options)
    for invalid in ({"axis": (0, 0, 0)}, {"center_mm": None}, {"inner_radius_mm": 142.5},
                    {"outer_radius_mm": float('nan')}, {"outer_radius_mm": None}, {"axial_max_mm": -130}):
        with pytest.raises(GeometryError):
            shell(**invalid)
    for i, j in (((0, 0, 0), (0, 0, 0)), ((float('nan'), 0, 0), (160, 0, 0))):
        with pytest.raises(GeometryError):
            shell_intervals(i, j, shell())
    with pytest.raises(GeometryError):
        shell_intervals((160, 0, 0), (-160, 0, 0), None)
    with pytest.raises(GeometryError):
        legacy_opposition_accepts((0, 0, 0), (-160, 0, 0))


# --- T2c: faceted candidate from owner-supplied Cornell Sketch r01 231010 ---
# Expected values follow the drawing dimensions: entrance faces Ø310 mm,
# 20 mm crystals, 18° tiles of two 25.4 mm minimodules (0.5 mm gap), 51.3 mm
# axial rows with 1.2 mm gaps (row pitch 52.5 mm, SM pitch 105 mm).
MM_OFFSET = (25.4 + 0.5) / 2
SKETCH_AXIAL = (-52.5 / 2 - MM_OFFSET, -52.5 / 2 + MM_OFFSET,
                52.5 / 2 - MM_OFFSET, 52.5 / 2 + MM_OFFSET)


def map_minimodule_centers():
    """Map (input) minimodule centers: TIME coordinate 0, ENERGY coordinate 1."""
    _, local, modules, types = config_and_map()
    centers = {}
    for kind, coordinate in ((ChannelType.TIME, 0), (ChannelType.ENERGY, 1)):
        values = {}
        for ch, (sm, mm) in modules.items():
            if sm == 1 and kind in types[ch]:
                values.setdefault(mm, []).append(local[ch][coordinate])
        centers[coordinate] = sorted({round((min(v) + max(v)) / 2, 6) for v in values.values()})
    return centers[0], centers[1]


def faceted(depth=5.0, **kwargs):
    config, local, modules, types = config_and_map()
    options = dict(interaction_depth_mm=depth, axial_sign=-1, transverse_sign=1) | kwargs
    return build_faceted_detector_frames(config, local, modules, types, **options)


@pytest.mark.parametrize("depth", [0.0, 5.0, 20.0])
def test_faceted_tiles_match_sketch(depth):
    config, _, _, _ = config_and_map()
    frames = faceted(depth)
    assert set(frames) == set(range(30))
    centers_x, centers_y = map_minimodule_centers()
    assert len(centers_x) == len(centers_y) == 4
    apothem = 155.0 + depth
    azimuths = set()
    for sm, frame in frames.items():
        y_c, x_c = config["ring_yx"][sm // 3]
        theta = math.atan2(y_c, x_c)
        for k, local_y in enumerate(centers_y):
            phi = theta + math.radians(-9.0 if k < 2 else 9.0)
            along = -MM_OFFSET if k % 2 == 0 else MM_OFFSET
            point = frame.to_global(centers_x[1], local_y)
            expected = (apothem * math.cos(phi) - along * math.sin(phi),
                        apothem * math.sin(phi) + along * math.cos(phi))
            assert point[:2] == pytest.approx(expected, abs=1e-9)
            # Every position on a tile lies on its declared interaction plane.
            assert point[0] * math.cos(phi) + point[1] * math.sin(phi) == pytest.approx(apothem)
            # Within-minimodule offsets follow the map along the tile.
            shifted = frame.to_global(centers_x[1], local_y + 3.2)
            assert shifted - point == pytest.approx((-3.2 * math.sin(phi), 3.2 * math.cos(phi), 0))
            azimuths.add(round(math.degrees(phi) % 360, 9))
        assert "pending" in frame.qualification and frame.crystal_thickness_mm == 20
    # Twenty flat tiles at uniform 18° pitch around the ring; YAML ring_yx
    # coordinates are rounded to 1 µm, i.e. azimuth errors below 1e-3°.
    assert len(azimuths) == 20
    ordered = sorted(azimuths)
    assert np.diff(ordered + [ordered[0] + 360]) == pytest.approx([18.0] * 20, abs=1e-3)


def test_faceted_axial_rows_match_sketch():
    config, _, _, _ = config_and_map()
    frames = faceted()
    centers_x, centers_y = map_minimodule_centers()
    populated = []
    for sm, frame in frames.items():
        z_sm = (sm % 3 - 1) * 105.0
        unpopulated = set(config["unpopulated_minimodules"].get(sm, ()))
        for k, local_x in enumerate(centers_x):
            z = frame.to_global(local_x, centers_y[0])[2]
            # axial_sign=-1: increasing slab position moves toward lower z.
            assert z == pytest.approx(z_sm - SKETCH_AXIAL[k])
            assert frame.to_global(local_x + 1.6, centers_y[0])[2] == pytest.approx(z - 1.6)
            if not (unpopulated and k < 2):
                populated.append(z)
    # Unpopulated half (low slab positions) lies at the outer end of SM z-index 2.
    assert set(config["unpopulated_minimodules"][2]) == {2, 3, 6, 7, 10, 11, 14, 15}
    rows = sorted(set(round(z, 6) for z in populated))
    assert len(rows) == 10
    assert rows[0] - 12.7 == pytest.approx(-156.9)
    assert rows[-1] + 12.7 == pytest.approx(104.4)
    assert rows[-1] + 12.7 - (rows[0] - 12.7) == pytest.approx(261.3)
    # Alternating 0.5 mm intra-row and 1.2 mm inter-row gaps between 25.4 mm minimodules.
    assert np.diff(rows) == pytest.approx([25.9, 26.6] * 4 + [25.9])


def test_faceted_differs_from_planar_by_tile_position():
    config, _, modules, _ = config_and_map()
    centers_x, centers_y = map_minimodule_centers()
    planar_frames = build_detector_frames(config, modules, local_origin_mm=(51.61, 51.61),
                                          local_axes="time-axial", axial_sign=-1,
                                          crystal_thickness_mm=20)
    tiled_frames = faceted(depth=5.0)
    c, s = math.cos(math.radians(9)), math.sin(math.radians(9))
    # Equal 160 mm apothem: inner minimodule nearly agrees, outer moves ~4 mm inward.
    inner = tiled_frames[1].to_global(centers_x[1], centers_y[2])
    outer = tiled_frames[1].to_global(centers_x[1], centers_y[3])
    assert inner[:2] == pytest.approx((160 * c + MM_OFFSET * s, 160 * s - MM_OFFSET * c))
    assert outer[:2] == pytest.approx((160 * c - MM_OFFSET * s, 160 * s + MM_OFFSET * c))
    planar_inner = planar_frames[1].to_global(centers_x[1], centers_y[2])
    planar_outer = planar_frames[1].to_global(centers_x[1], centers_y[3])
    assert np.linalg.norm(inner[:2] - planar_inner[:2]) < 1.0
    assert 3.5 < np.linalg.norm(outer[:2] - planar_outer[:2]) < 4.5
    # Outer-to-inner opposite minimodules (a symmetric diametric LOR would have
    # endpoint-independent TOF): candidate choice changes the shell TOF peaks.
    source = shell(axial_min_mm=-150.0, axial_max_mm=150.0)
    a = shell_intervals(planar_outer, planar_frames[16].to_global(centers_x[1], centers_y[2]), source)
    b = shell_intervals(outer, tiled_frames[16].to_global(centers_x[1], centers_y[2]), source)
    assert len(a) == len(b) == 2
    assert all(abs(x.tof_start_ps - y.tof_start_ps) > 1 for x, y in zip(a, b))


def test_faceted_rejects_undeclared_or_invalid_geometry():
    config, local, modules, types = config_and_map()
    for depth in (None, -0.1, 20.1, float("nan"), True):
        with pytest.raises(GeometryError):
            faceted(depth)
    # Handedness has no default; depth defaults only to the owner-declared 7 mm.
    with pytest.raises(TypeError):
        build_faceted_detector_frames(config, local, modules, types, axial_sign=-1)
    for sign in (0, 2, True, None):
        with pytest.raises(GeometryError):
            faceted(transverse_sign=sign)
    with pytest.raises(GeometryError):
        faceted(axial_sign=0)
    for changed in ({"ring_z": [-102.0, 102.0]}, {"ring_z": [102.0, 0.0, -102.0]},
                    {"ring_z": [-100.0, 5.0, 100.0]}, {"ring_r": 150}, {"ring_yx": {}}):
        with pytest.raises(GeometryError):
            build_faceted_detector_frames(config | changed, local, modules, types,
                                          interaction_depth_mm=5, axial_sign=-1, transverse_sign=1)
    # A map without four minimodule centers per axis is not this sketch layout.
    moved = dict(local)
    ch = next(ch for ch in local if modules[ch] == (1, 0) and ChannelType.TIME in types[ch])
    moved[ch] = (local[ch][0] + 7.0, local[ch][1])
    with pytest.raises(GeometryError, match="minimodule"):
        build_faceted_detector_frames(config, moved, modules, types,
                                      interaction_depth_mm=5, axial_sign=-1, transverse_sign=1)
    frame = faceted()[1]
    centers_x, centers_y = map_minimodule_centers()
    for x, y in ((centers_x[0] - 13.0, centers_y[0]), (centers_x[1], centers_y[3] + 13.0),
                 ((centers_x[0] + centers_x[1]) / 2, centers_y[0]), (float("nan"), centers_y[0])):
        with pytest.raises(GeometryError):
            frame.to_global(x, y)


# --- T2d: owner-declared 7 mm depth; both transverse handedness candidates ---
def test_faceted_handedness_mirrors_each_sm():
    config, local, modules, types = config_and_map()
    centers_x, centers_y = map_minimodule_centers()
    ccw = faceted(transverse_sign=1)
    cw = faceted(transverse_sign=-1)
    for sm in range(30):
        y_c, x_c = config["ring_yx"][sm // 3]
        u = np.array([x_c, y_c]) / math.hypot(x_c, y_c)
        for local_x in centers_x:
            for local_y in (centers_y[0] - 5.0, centers_y[1], centers_y[2] + 3.2, centers_y[3]):
                a = ccw[sm].to_global(local_x, local_y)
                b = cw[sm].to_global(local_x, local_y)
                # Clockwise is the mirror image across the cassette's radial plane.
                assert b[:2] == pytest.approx(2 * (a[:2] @ u) * u - a[:2], abs=1e-9)
                assert b[2] == a[2]
    # A minimodule off the cassette centre moves to the other side of it.
    a = ccw[0].to_global(centers_x[0], centers_y[3])
    b = cw[0].to_global(centers_x[0], centers_y[3])
    assert a[1] > 30 and b[1] < -30


def test_faceted_nominal_depth_closes_tile_edges():
    config, local, modules, types = config_and_map()
    frames = build_faceted_detector_frames(config, local, modules, types,
                                           axial_sign=-1, transverse_sign=1)
    centers_x, centers_y = map_minimodule_centers()
    half = 25.4 / 2
    frame = frames[1]
    # Owner-declared depth: 7 mm below the Ø310 entrance face.
    centre = frame.to_global(centers_x[1], centers_y[2])
    phi = math.radians(9)
    assert centre[0] * math.cos(phi) + centre[1] * math.sin(phi) == pytest.approx(162.0)
    # At 162 mm the 18° chord equals the 51.3 mm tile: the two tiles of one SM
    # (and of neighbouring cassettes) meet within 0.1 mm at the interaction plane.
    lower = frame.to_global(centers_x[1], centers_y[1] + half)
    upper = frame.to_global(centers_x[1], centers_y[2] - half)
    assert np.linalg.norm(lower - upper) < 0.1
    neighbour = frames[4].to_global(centers_x[1], centers_y[0] - half)
    edge = frame.to_global(centers_x[1], centers_y[3] + half)
    assert np.linalg.norm(neighbour - edge) < 0.1


# --- T2b: data-selected transverse handedness (T6 evidence, owner-accepted 2026-10-13) ---
PAIR_FITS = ROOT / "specs/008-cornell-skew-calibration/run7-pair-fits.json"


def test_transforms_match_measured_geometry():
    from src.cornell.skew.geometry import (NORMALIZATION_PHANTOM, OWNER_INTERACTION_DEPTH_MM,
                                           SELECTED_AXIAL_SIGN, SELECTED_TRANSVERSE_SIGN)
    import json
    assert (SELECTED_AXIAL_SIGN, SELECTED_TRANSVERSE_SIGN, OWNER_INTERACTION_DEPTH_MM) == (-1, 1, 7.0)
    assert NORMALIZATION_PHANTOM == shell(axial_min_mm=-150.0, axial_max_mm=150.0)
    # Recorded run7 evidence: on events only one hypothesis accepts, the selected one
    # predicts the measured branch separation; the mirror is off by hundreds of ps.
    evidence = json.loads(PAIR_FITS.read_text(encoding="utf-8"))["handedness"]
    assert evidence["selected"] == f"{SELECTED_TRANSVERSE_SIGN:+d}"
    for role in ("train", "validation"):
        exclusive = evidence["exclusive_events"][role]
        right, wrong = exclusive["+1"], exclusive["-1"]
        assert abs(right["separation_weighted_mean_ps"]) < max(10.0, 3 * right["separation_mean_sigma_ps"])
        assert right["separation_pull_rms"] < 1.3
        assert abs(wrong["separation_weighted_mean_ps"]) > 100 + 10 * wrong["separation_mean_sigma_ps"]
        # Events both accept are mirror images with identical shell TOFs: no information.
        assert evidence["common_events_tof_identical"][role]["max_abs_tof_diff_ps"] < 1e-6
    # Why: flipping handedness mirrors each SM across its cassette's radial plane, a
    # symmetry of the centred shell for opposite cassettes only.
    ccw, cw = faceted(OWNER_INTERACTION_DEPTH_MM), faceted(OWNER_INTERACTION_DEPTH_MM, transverse_sign=-1)
    centers_x, centers_y = map_minimodule_centers()
    a_local, b_local = (centers_x[1], centers_y[0] + 4.0), (centers_x[2], centers_y[3] - 7.0)

    def tofs(frames, sm_a, sm_b):
        return [t for s in shell_intervals(frames[sm_a].to_global(*a_local), frames[sm_b].to_global(*b_local),
                                           NORMALIZATION_PHANTOM) for t in (s.tof_start_ps, s.tof_end_ps)]
    opposite, neighbour = (1, 16), (1, 13)  # cassettes 0/5 and 0/4
    assert tofs(ccw, *opposite) == pytest.approx(tofs(cw, *opposite), abs=1e-6)
    assert len(tofs(ccw, *neighbour)) != len(tofs(cw, *neighbour)) or \
        max(abs(x - y) for x, y in zip(tofs(ccw, *neighbour), tofs(cw, *neighbour))) > 50
