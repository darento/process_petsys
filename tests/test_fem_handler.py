"""src.fem_handler: FEM selection and channel coordinates (spec 007 FR-9).

Expected coordinates are worked out by hand from the pitch and the channel count:
unsummed grids have channels // 16 columns; summed layouts span channels // 8 channels
and flip y about the top row. Distinct x and y pitches catch swapped axes.
"""

import pytest

from src.fem_handler import FEM128, FEM256, get_FEM_instance

XP, YP = 3.2, 3.3


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("fem_type, cls, asics", [("FEM128", FEM128, 2), ("FEM256", FEM256, 4)])
def test_get_fem_instance(fem_type, cls, asics):
    fem = get_FEM_instance(fem_type, XP, YP, 8, True, 64)
    assert type(fem) is cls
    assert (fem.x_pitch, fem.y_pitch, fem.mM_channels, fem.sum_rows_cols) == (XP, YP, 8, True)
    assert fem.channels_populated == 64
    assert fem.num_ASICS == asics


@pytest.mark.fr("007-FR-9")
def test_unknown_fem_type_rejected():
    with pytest.raises(ValueError, match="Unsupported FEM type"):
        get_FEM_instance("FEM512", XP, YP, 8, False, 512)


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("fem_type, channels, pos, expected", [
    ("FEM128", 128, 0, (1.6, 1.65)),
    ("FEM128", 128, 37, (17.6, 14.85)),     # 8 columns: row 4, col 5
    ("FEM128", 128, 127, (24.0, 51.15)),    # row 15, col 7
    ("FEM256", 256, 0, (1.6, 1.65)),
    ("FEM256", 256, 37, (17.6, 8.25)),      # 16 columns: row 2, col 5
    ("FEM256", 256, 255, (49.6, 51.15)),    # row 15, col 15
])
def test_unsummed_coordinates(fem_type, channels, pos, expected):
    fem = get_FEM_instance(fem_type, XP, YP, 8, False, channels)
    assert fem.get_coordinates(pos) == pytest.approx(expected)


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("pos, expected", [
    (0, (1.6, 103.95)),     # top row offset 31
    (37, (17.6, 87.45)),    # 37 % 32 = 5
    (255, (100.8, 1.65)),   # 255 % 32 = 31
])
def test_summed_coordinates_fem256(pos, expected):
    fem = get_FEM_instance("FEM256", XP, YP, 8, True, 256)
    assert fem.get_coordinates(pos) == pytest.approx(expected)
