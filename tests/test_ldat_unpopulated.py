"""Half-populated Cornell SuperModules come from the config (spec 002 B3; spec 007 T21).

Moved from ``scripts/ldat_unpopulated_check.py``: each ``check`` is one test. Bug (spec 001, from
``LDATInspector_legacy.py``): SM 20-29 were hardcoded as missing minimodules {2, 3, 6, 7, 10, 11, 14,
15}. The owner confirmed on 2026-09-25 that SM 2, 5, 8, ..., 29 are the half-populated ones, and the
whole-file 2026-01-19 acquisition has no sides in exactly those minimodules. The fix reads
``unpopulated_minimodules`` from the selected config: here the tracked January config
(``CONFIGS["CORNELL"]``, the script's ``configs/cornell_full_system_old.yaml``).
"""

from pathlib import Path
import tempfile

import pytest
import yaml

from helpers import REPO
from ldat_helpers import CONFIGS
from src.ldat_inspector.engine import Settings, load_setup, merge_results, process_file_reference

pytestmark = pytest.mark.fr("bug-unpopulated-minimodules", "002-FR-5")
CONFIG = CONFIGS["CORNELL"]
HALF = [2, 3, 6, 7, 10, 11, 14, 15]
HALF_SMS = list(range(2, 30, 3))
FULL = set(range(16))


def expected(config_path):
    settings = Settings(str(config_path), "", "CORNELL", calibrated=False)
    return merge_results(settings, [], load_setup(settings))


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset():
    return expected(CONFIG)


def test_config_declares_sm_2_5_to_29_half_populated(config):
    declared = {int(sm): sorted(mms) for sm, mms in (config.get("unpopulated_minimodules") or {}).items()}
    assert declared == {sm: HALF for sm in HALF_SMS}, f"{declared or 'no unpopulated_minimodules key'}"


def test_half_populated_sms_expect_only_their_populated_minimodules(dataset):
    assert all(dataset.expected_mm.get(sm) == FULL - set(HALF) for sm in HALF_SMS), \
        f"SM 2 expects {sorted(dataset.expected_mm.get(2, ()))}"


def test_all_other_sms_including_21_to_28_expect_16_minimodules(dataset):
    assert all(dataset.expected_mm.get(sm) == FULL for sm in range(30) if sm not in HALF_SMS), \
        f"SM 21 expects {len(dataset.expected_mm.get(21, ()))}"


def test_half_populated_sm_expects_64_time_and_64_energy_channels(dataset):
    assert len(dataset.expected_time.get(2, ())) == 64 and len(dataset.expected_energy.get(2, ())) == 64, \
        f"{len(dataset.expected_time.get(2, ()))} / {len(dataset.expected_energy.get(2, ()))}"


@pytest.fixture
def plain(config):
    """The config without ``unpopulated_minimodules`` (absolute map path), and where to write it."""
    with tempfile.TemporaryDirectory() as temporary:
        plain = dict(config)
        plain.pop("unpopulated_minimodules", None)
        plain["map_file"] = str(REPO / config["map_file"])
        yield plain, Path(temporary) / "plain.yaml"


def test_config_without_the_key_expects_every_mapped_minimodule(plain):
    config, path = plain
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert all(mms == FULL for mms in expected(path).expected_mm.values())


def test_malformed_key_is_rejected(plain):
    config, path = plain
    path.write_text(yaml.safe_dump({**config, "unpopulated_minimodules": [2, 3]}), encoding="utf-8")
    with pytest.raises(ValueError):
        expected(path)


# --- real Cornell prefix ----------------------------------------------------------
# Data: PETSYS_DATA_DIR/Cornell/full_system, January 2026-01-19 compact split 00000003, first 300,000
# pairs. Map: January, through CONFIGS["CORNELL"]. No calibration. Cuts: >= 4 energy channels,
# >= 0.2 a.u. per channel.

REAL = ("Cornell/full_system/20260119_2NaSourcesAxialSeparated_vBiasCompDiscCalibAdjusted2hits_300s_"
        "coincCompact11s_00000003.ldat")


@pytest.fixture(scope="module")
def real_cache():
    return {}


@pytest.fixture
def seen(real_data_file, real_cache):
    """SM -> minimodules with sides on the real prefix (computed once)."""
    path = real_data_file(REAL)
    if not real_cache:
        settings = Settings(str(CONFIG), "", "CORNELL", max_pairs=300_000, min_channels=4,
                            min_channel_energy=0.2, calibrated=False)
        result = process_file_reference(str(path), settings, 0)
        real_cache.update({sm: set(data.mm.tolist()) for sm, data in result.modules.items()})
    return real_cache


@pytest.mark.real_data
@pytest.mark.fr("007-FR-4")
class TestRealPrefix:
    def test_no_sides_in_unpopulated_minimodules(self, seen):
        assert all(not seen.get(sm, set()) & set(HALF) for sm in HALF_SMS)

    def test_every_expected_minimodule_has_sides(self, seen, dataset):
        assert all(seen.get(sm, set()) >= mms for sm, mms in dataset.expected_mm.items())
