from .fem_handler import FEM128, FEM256, get_FEM_instance


def test_get_FEM_instance():
    fem128 = FEM128(1.1, 1.1, 8, False)
    fem128_f = get_FEM_instance("FEM128", 1.1, 1.1, 8, False)

    fem256 = FEM256(1.1, 1.1, 8, False)
    fem256_f = get_FEM_instance("FEM256", 1.1, 1.1, 8, False)

    assert fem128.__dict__ == fem128_f.__dict__
    assert fem256.__dict__ == fem256_f.__dict__

    assert fem128.get_coordinates(0) == fem128_f.get_coordinates(0) == (0.55, 0.55)
    assert fem128.get_coordinates(150) == fem128_f.get_coordinates(150) == (7.15, 20.35)
    assert fem256.get_coordinates(0) == fem256_f.get_coordinates(0) == (0.55, 0.55)
    assert fem256.get_coordinates(256) == fem256_f.get_coordinates(256) == (0.55, 18.15)
