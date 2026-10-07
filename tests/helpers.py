"""Shared test helpers (specs 006, 007): repo paths, synthetic LDAT writer, map/config
loaders, small synthetic channel maps.

Import as ``from helpers import ...``; ``pyproject.toml`` puts ``tests/`` on the path.
"""

import struct
from pathlib import Path

import yaml

from src.mapping_generator import ChannelType, map_factory

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "data"
MAPS = REPO / "maps"
CONFIGS = REPO / "configs"

# Same native layout as src/read_compact.py: per record a "2B" header with the hit
# counts of the two detectors, then one "qfi" (timestamp, energy, channel ID) per hit.
HEADER = "2B"
HIT = "qfi"

# Small maps in map_factory's form (channel -> [ChannelType], channel -> (SM, mM)).
# Channels 0-3 energy only and 4-7 time only (summed rows/cols); 8-9 both types (FEM128 style).
SYNTH_CHTYPE = {**{c: [ChannelType.ENERGY] for c in range(4)},
                **{c: [ChannelType.TIME] for c in range(4, 8)},
                8: [ChannelType.TIME, ChannelType.ENERGY], 9: [ChannelType.TIME, ChannelType.ENERGY]}
# SM 0 holds minimodules 0 (channels 0, 1, 4, 5) and 1 (2, 3, 6, 7); SM 1 minimodule 0 holds 8, 9.
SYNTH_SM_MM = {0: (0, 0), 1: (0, 0), 4: (0, 0), 5: (0, 0), 2: (0, 1), 3: (0, 1), 6: (0, 1), 7: (0, 1),
               8: (1, 0), 9: (1, 0)}


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
