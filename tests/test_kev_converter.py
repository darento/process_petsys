"""src.utils.KevConverter on small synthetic calibration files (spec 006 FR-7).

Energies are PETsys a.u.; a mu factor maps the 511 keV photopeak, so keV = 511 / mu * E.
A channel without a factor must never receive a number.
"""

import math

import pytest

from src.utils import KevConverter


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def mu_file(tmp_path):
    return _write(tmp_path, "mu.tsv", "ID\tmu\n5\t100.0\n6\t255.5\n7\t0.0\n8\t-5.0\n9\tnan\n")


@pytest.fixture
def cornell_file(tmp_path):
    return _write(tmp_path, "slab.encal", "ID(t_ch, slab)\tmu\tsigma\n(10, 3)\t200.0\t5.0\n(10, 4)\t0.0\t0.0\n(10, 6)\t-2.0\t0.0\n")


@pytest.mark.fr("006-FR-7")
def test_mu_known_factor(mu_file):
    converter = KevConverter(mu_file, "mu")
    assert converter.convert(5, 50.0) == pytest.approx(511.0 / 100.0 * 50.0)
    assert converter.convert(6, 255.5) == pytest.approx(511.0)


@pytest.mark.fr("006-FR-7")
def test_cornell_slab_known_factor(cornell_file):
    assert KevConverter(cornell_file, "cornell").convert((10, 3), 100.0) == pytest.approx(255.5)


@pytest.mark.fr("006-FR-7")
def test_cornell_position_known_factor(tmp_path):
    path = _write(tmp_path, "pos.encal",
                  "# Position-dependent energy calibration (4 regions per slab)\n"
                  "ID(time_ch, slab, region)\tmu\n(10, 3, 1)\t150.0\n")
    converter = KevConverter(path, "cornell_position")
    assert converter.num_regions == 4
    assert converter.convert((10, 3, 1), 300.0) == pytest.approx(1022.0)


@pytest.mark.fr("006-FR-7")
def test_poly_known_coefficients(tmp_path):
    path = _write(tmp_path, "poly.tsv", "ID\tcoef0\tcoef1\tcoef2\n5\t0.001\t2.0\t-3.0\n")
    assert KevConverter(path, "poly").convert(5, 100.0) == pytest.approx(0.001 * 100.0**2 + 2.0 * 100.0 - 3.0)


@pytest.mark.fr("006-FR-7")
@pytest.mark.parametrize("file_type, key", [("mu", 99), ("cornell", (10, 5)), ("poly", 99)])
def test_missing_channel_gets_no_value(tmp_path, mu_file, cornell_file, file_type, key):
    path = {"mu": mu_file, "cornell": cornell_file,
            "poly": _write(tmp_path, "poly.tsv", "ID\tcoef0\tcoef1\tcoef2\n5\t0.0\t1.0\t0.0\n")}[file_type]
    with pytest.raises(KeyError):
        KevConverter(path, file_type).convert(key, 100.0)


@pytest.mark.fr("006-FR-7")
def test_unknown_file_type_rejected(mu_file):
    with pytest.raises(ValueError, match="Unknown file type"):
        KevConverter(mu_file, "linear")


@pytest.mark.fr("006-FR-7", "bug-kev-mu-zero")
@pytest.mark.parametrize("file_type, key", [("mu", 7), ("mu", 8), ("mu", 9), ("cornell", (10, 4)), ("cornell", (10, 6))],
                         ids=["mu 0", "mu negative", "mu nan", "cornell 0", "cornell negative"])
def test_unusable_factor_gets_no_number(mu_file, cornell_file, file_type, key):
    # Before the fix: mu == 0 gave 0 keV and a negative mu a negative keV.
    path = {"mu": mu_file, "cornell": cornell_file}[file_type]
    assert math.isnan(KevConverter(path, file_type).convert(key, 100.0))
