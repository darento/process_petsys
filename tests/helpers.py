"""Shared test helpers (spec 006): repo paths, synthetic LDAT writer, map/config loaders.

Import as ``from helpers import ...``; ``pyproject.toml`` puts ``tests/`` on the path.
"""

import struct
from pathlib import Path

import yaml

from src.mapping_generator import map_factory

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "data"
MAPS = REPO / "maps"
CONFIGS = REPO / "configs"

# Same native layout as src/read_compact.py: per record a "2B" header with the hit
# counts of the two detectors, then one "qfi" (timestamp, energy, channel ID) per hit.
HEADER = "2B"
HIT = "qfi"


def write_ldat(path, pairs):
    """Write coincidence records ``[(det1, det2), ...]`` of ``(timestamp, energy, channel_id)`` hits.

    Each detector holds at most 255 hits (one header byte). Returns ``path``.
    """
    with open(path, "wb") as handle:
        for det1, det2 in pairs:
            if len(det1) > 255 or len(det2) > 255:
                raise ValueError("a detector holds at most 255 hits per record")
            handle.write(struct.pack(HEADER, len(det1), len(det2)))
            for timestamp, energy, channel in (*det1, *det2):
                handle.write(struct.pack(HIT, timestamp, energy, channel))
    return path


def load_map(name):
    """``map_factory`` on ``maps/<name>``: ``(local_map, sm_mM_map, chtype_map, fem)``."""
    return map_factory(str(MAPS / name))


def load_config(name):
    """Parsed ``configs/<name>`` YAML."""
    with open(CONFIGS / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
