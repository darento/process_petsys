"""Strict manager input contracts. Existing Inspector/readers remain unchanged.

LDAT has no universal format/population signature. Operator or converter-supplied
descriptors are mandatory; structural validation cannot prove indistinguishable
layouts. Full scans retain at most one bounded byte batch, never event lists.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType

import yaml

from src.fem_handler import get_FEM_instance
from src.mapping_generator import ChannelType, _get_local_mapping
from src.petsys_manager.contracts import (Action, DataFormat, FrozenMapping,
    InputDescriptor, Population, freeze, route_accepts)


MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 100000
MAX_BATCH_BYTES = 16 * 1024 * 1024
HIT = struct.Struct("<qfi")


class InputError(ValueError):
    def __init__(self, *issues):
        self.issues = tuple(str(issue) for issue in issues)
        super().__init__("; ".join(self.issues))


class ValidationCancelled(InputError):
    pass


def _integer(value, label, lower=0, upper=2**31 - 1):
    if type(value) is not int or not lower <= value <= upper:
        raise InputError(f"{label}: integer in [{lower}, {upper}] required")
    return value


def _number(value, label, *, positive=False, nonnegative=False):
    if type(value) not in (int, float):
        raise InputError(f"{label}: finite number required")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise InputError(f"{label}: finite number required") from None
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        raise InputError(f"{label}: invalid/nonfinite number")
    return number


def _absolute(path, root):
    root = Path(root)
    if not root.is_absolute():
        raise InputError("Processing root must be absolute")
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


def describe_legacy(paths, *, format=None, population=None, confirmed=False):
    """Require an explicit operator decision, even for a file named *.ldat."""
    if confirmed is not True or format is None or population is None:
        raise InputError("Confirm legacy input format and population explicitly")
    return tuple(InputDescriptor(path, format, population) for path in paths)


def calibration_layout(path):
    """("per_slab", 1) or ("position", regions) from a calibration header; full checks are load_calibration's."""
    with open(path, "rb") as stream:
        head = stream.read(4096).decode("utf-8", "replace").splitlines()
    head = [line.strip() for line in head if line.strip()]
    if head and head[0] == "ID(t_ch, slab)\tmu\tsigma":
        return "per_slab", 1
    match = re.fullmatch(r"# Position-dependent energy calibration \((\d+) regions per slab\)", head[0]) if head else None
    if match is None:
        raise InputError(f"Unsupported calibration header/key schema: {path}")
    return "position", int(match[1])


def select_inputs(descriptors, action, *, processing_root):
    """Resolve only the given ordered list. Report every missing/duplicate/wrong route."""
    action = Action(action)
    if action not in (Action.CALIBRATE, Action.LISTMODE, Action.QC_ANALYZE):
        raise InputError("Select inputs for calibration, listmode or offline QC")
    selected, issues, seen = [], [], set()
    for descriptor in descriptors:
        if len(selected) >= MAX_ENTRIES:
            raise InputError("Input list exceeds its metadata bound")
        if not isinstance(descriptor, InputDescriptor):
            raise InputError("Inputs need explicit typed descriptors; extensions are ambiguous")
        path = _absolute(descriptor.path, processing_root)
        if not path.is_file():
            issues.append(f"Missing selected input: {path}")
        if path in seen:
            issues.append(f"Duplicate selected input: {path}")
        seen.add(path)
        if not route_accepts(action, descriptor.format, descriptor.population):
            issues.append(f"Wrong format/population for {action.value}: {path}")
        selected.append(replace(descriptor, path=path, validated=False))
    if not selected:
        issues.append("Select the exact ordered input list")
    if action == Action.CALIBRATE and len({(d.format, d.population) for d in selected}) > 1:
        issues.append("Do not mix formats or group/coincidence calibration inputs")
    if issues:
        raise InputError(*issues)
    return tuple(selected)


@dataclass(frozen=True)
class NumericTable(Mapping):
    """Immutable, sparse, pickle-friendly metadata keyed by actual map IDs."""
    entries: tuple
    _lookup: Mapping = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        entries = tuple((key, tuple(value)) for key, value in self.entries)
        if len(entries) > MAX_ENTRIES or len(dict(entries)) != len(entries):
            raise InputError("Duplicate/oversized metadata table")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_lookup", MappingProxyType(dict(entries)))

    def __getitem__(self, key):
        return self._lookup[key]

    def __iter__(self):
        return iter(self._lookup)

    def __len__(self):
        return len(self._lookup)

    def __reduce__(self):
        return type(self), (self.entries,)

    def missing(self, keys):
        return tuple(key for key in keys if key not in self)


@dataclass(frozen=True)
class ChannelMap:
    path: Path
    sha256: str
    config: FrozenMapping
    local: NumericTable
    modules: NumericTable
    types: NumericTable

    def time_slab_key(self, key):
        channel, slab = key[:2]
        _integer(channel, "time channel")
        if channel not in self.types or ChannelType.TIME not in self.types[channel]:
            raise InputError(f"Not a mapped time channel: {channel}")
        if not self.config["sum_rows_cols"] or self.config["mM_channels"] != 8:
            raise InputError("Cornell position/slab route requires the supported 8-position summed map")
        _integer(slab, "Cornell slab", 0, 15)
        position = self.local[channel][2]
        if slab not in (2 * position, 2 * position + 1):
            raise InputError(f"Slab {slab} is inconsistent with time channel {channel} position {position}")


@dataclass(frozen=True)
class ProcessingConfig:
    path: Path
    sha256: str
    root: Path
    values: FrozenMapping
    mapping: ChannelMap


class _Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._nodes = 0
        super().__init__(stream)

    def compose_node(self, parent, index):
        self._nodes += 1
        if self._nodes > MAX_ENTRIES:
            raise InputError("YAML nodes exceed the metadata bound")
        if self.check_event(yaml.AliasEvent):
            raise InputError("YAML aliases are unsupported in processing metadata")
        return super().compose_node(parent, index)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        try:
            duplicate = key in result
        except TypeError:
            raise InputError("Metadata keys must be scalar") from None
        if duplicate:
            raise InputError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _yaml_mapping(loader, node):
    # No merge keys: keep duplicate detection explicit and bounded.
    return _unique_pairs((loader.construct_object(key, deep=True),
                          loader.construct_object(value, deep=True)) for key, value in node.value)


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def _metadata_bytes(path):
    path = Path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise InputError(f"Metadata input is not a regular file: {path}")
    with path.open("rb") as stream:
        content = stream.read(MAX_METADATA_BYTES + 1)
    if len(content) > MAX_METADATA_BYTES:
        raise InputError(f"Metadata file exceeds {MAX_METADATA_BYTES} bytes: {path}")
    return content


def _finite_tree(value, depth=0):
    if depth > 32:
        raise InputError("Metadata nesting exceeds its bound")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) not in (int, str):
                raise InputError("Metadata keys must be integers or text")
            _finite_tree(child, depth + 1)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_tree(child, depth + 1)
    elif type(value) in (int, float):
        _number(value, "metadata value")
    elif value is not None and type(value) not in (str, bool):
        raise InputError("Unsupported metadata scalar")


def _yaml(path):
    content = _metadata_bytes(path)
    try:
        value = yaml.load(content.decode("utf-8"), Loader=_Loader)
        _finite_tree(value)
    except (yaml.YAMLError, RecursionError, UnicodeError) as exc:
        raise InputError(f"Invalid YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(f"Expected a YAML mapping: {path}")
    return value, hashlib.sha256(content).hexdigest()


def load_channel_map(path, *, processing_root):
    path = _absolute(path, processing_root)
    value, digest = _yaml(path)
    required = ("FEM", "FEBD", "channels", "mM_channels", "sum_rows_cols", "x_pitch",
                "y_pitch", "mod_feb_map", "mM_disposition")
    absent = [key for key in required if key not in value]
    if absent:
        raise InputError(f"Map missing keys: {', '.join(absent)}")
    capacity = {"FEM128": 128, "FEM256": 256}.get(value["FEM"]) if isinstance(value["FEM"], str) else None
    if capacity is None or not isinstance(value["FEBD"], str) or not value["FEBD"].strip():
        raise InputError("Unsupported FEM or missing FEBD identity")
    count = _integer(value["channels"], "populated FEM channels", 16, capacity)
    width = _integer(value["mM_channels"], "minimodule channel width", 1, count // 2)
    if count % 16 or count // 2 % width or type(value["sum_rows_cols"]) is not bool:
        raise InputError("Inconsistent channel layout or sum_rows_cols flag")
    for key in ("x_pitch", "y_pitch"):
        _number(value[key], key, positive=True)
    disposition = value["mM_disposition"]
    if not isinstance(disposition, list) or len(disposition) != 2:
        raise InputError("mM_disposition requires two dimensions")
    for dimension in disposition:
        _integer(dimension, "map disposition", 1, count)
    groups = [("Time", "Energy"), ("channels_j1", "channels_j2")]
    present = [group for group in groups if any(key in value for key in group)]
    if len(present) != 1 or not all(key in value for key in present[0]):
        raise InputError("Exactly one complete channel-list pair is required")
    lists = [value[key] for key in present[0]]
    if any(not isinstance(channels, list) or len(channels) != count // 2 for channels in lists):
        raise InputError("Channel lists must have equal declared half-FEM lengths")
    channels = lists[0] + lists[1]
    for channel in channels:
        _integer(channel, "local channel", 0, capacity - 1)
    if len(set(channels)) != len(channels):
        raise InputError("Channel lists overlap or contain duplicates")
    modules = value["mod_feb_map"]
    if not isinstance(modules, dict) or not modules or len(modules) * count > MAX_ENTRIES:
        raise InputError("Empty/oversized module map")
    absolute_ids = set()
    for module, address in modules.items():
        _integer(module, "SuperModule ID")
        if not isinstance(address, list) or len(address) != 3:
            raise InputError("Module address requires port/slave/FEB port")
        port, slave, feb = address
        _integer(port, "port", 0, 31)
        _integer(slave, "slave", 0, 31)
        _integer(feb, "FEB port", 0, 4096 // capacity - 1)
        ids = {131072 * port + 4096 * slave + capacity * feb + channel for channel in channels}
        if absolute_ids.intersection(ids):
            raise InputError("Module electronics addresses overlap")
        absolute_ids.update(ids)
    fem = get_FEM_instance(value["FEM"], value["x_pitch"], value["y_pitch"], width,
                           value["sum_rows_cols"], count)
    try:
        local, sm_mm, types = _get_local_mapping(modules, *lists, fem)
    except (KeyError, IndexError, ZeroDivisionError) as exc:
        raise InputError(f"Unsupported selected map layout: {exc}") from exc
    if not (set(local) == set(sm_mm) == set(types) == absolute_ids):
        raise InputError("Generated map is inconsistent")
    for coordinates in local.values():
        for coordinate in coordinates:
            _number(coordinate, "generated map coordinate")
    return ChannelMap(path, digest, freeze(value), NumericTable(tuple(local.items())),
                      NumericTable(tuple(sm_mm.items())), NumericTable(tuple(types.items())))


def load_processing_config(path, *, processing_root, action):
    root = Path(processing_root)
    path = _absolute(path, root)
    value, digest = _yaml(path)
    name = value.get("map_file")
    if not isinstance(name, str) or not name.strip():
        raise InputError("Processing config requires map_file")
    mapping = load_channel_map(name, processing_root=root)
    _integer(value.get("min_ch"), "min_ch", 1, mapping.config["mM_channels"])
    action = Action(action)
    if action not in (Action.CALIBRATE, Action.LISTMODE, Action.QC_ANALYZE, Action.PIPELINE, Action.QC):
        raise InputError("Unsupported processing action")
    if not mapping.config["sum_rows_cols"] or mapping.config["mM_channels"] != 8:
        raise InputError("Cornell processing requires the supported 8-position summed slab map")
    if action != Action.CALIBRATE or "en_min_ch" in value:
        _number(value.get("en_min_ch"), "en_min_ch (a.u.)", nonnegative=True)
    if action in (Action.LISTMODE, Action.PIPELINE) or "energy_range" in value:
        window = value.get("energy_range")
        if window is not None or action in (Action.LISTMODE, Action.PIPELINE):
            if not isinstance(window, list) or len(window) != 2:
                raise InputError("energy_range requires two keV bounds")
            lower = _number(window[0], "energy lower (keV)", nonnegative=True)
            if _number(window[1], "energy upper (keV)", positive=True) <= lower:
                raise InputError("Energy upper must exceed lower")
    absent = value.get("unpopulated_minimodules", {})
    known = set(mapping.modules.values())
    known_modules = {module for module, _ in known}
    if not isinstance(absent, dict):
        raise InputError("unpopulated_minimodules must be a mapping")
    for module, minimodules in absent.items():
        _integer(module, "unpopulated SuperModule")
        if module not in known_modules:
            raise InputError(f"Unpopulated SuperModule is not in selected map: {module}")
        if not isinstance(minimodules, list) or any(type(mm) is not int for mm in minimodules) or len(set(minimodules)) != len(minimodules):
            raise InputError("Unpopulated minimodules require a unique list")
        for mm in minimodules:
            _integer(mm, "unpopulated minimodule")
            if (module, mm) not in known:
                raise InputError(f"Unpopulated minimodule is not in selected map: {(module, mm)}")
    return ProcessingConfig(path, digest, root, freeze(value), mapping)


def _rows(path):
    content = _metadata_bytes(path)
    try:
        text = content.decode("utf-8")
    except UnicodeError as exc:
        raise InputError(f"Invalid UTF-8 metadata: {path}") from exc
    rows = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    if any(len(line) > 4096 for _, line in rows):
        raise InputError("Metadata row exceeds its bound")
    if len(rows) > MAX_ENTRIES + 2:
        raise InputError("Table exceeds its metadata row bound")
    return rows, hashlib.sha256(content).hexdigest()


def _key(text, length, mapping):
    try:
        key = ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError) as exc:
        raise InputError(f"Invalid tuple key: {text}") from exc
    if not isinstance(key, tuple) or len(key) != length or any(type(v) is not int for v in key):
        raise InputError(f"Expected {length}-integer tuple key: {text}")
    mapping.time_slab_key(key)
    return key


@dataclass(frozen=True)
class Limits:
    path: Path
    sha256: str
    kind: str
    values: NumericTable
    zero_width: tuple = ()          # keys with left == right: kept, their sides fall out of range (FR-12)


def load_limits(path, mapping, *, kind):
    if not isinstance(mapping, ChannelMap) or not Path(path).is_absolute():
        raise InputError("Limits require an absolute selected path and typed channel map")
    if kind not in ("cog", "doi"):
        raise InputError("Limits require explicit COG or DOI kind")
    rows, digest = _rows(path)
    entries, zero_width = [], []
    for number, line in rows:
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise InputError(f"Limits row {number}: expected key/left/right")
        key = _key(fields[0], 2, mapping)
        try:
            left, right = (_number(float(value), f"limits row {number}") for value in fields[1:])
        except ValueError as exc:
            raise InputError(f"Invalid limits row {number}: {exc}") from exc
        if right < left:
            raise InputError(f"Limits row {number}: right must not be below left")
        if right == left:     # the reference formulas put every side of this key out of range
            zero_width.append(key)
        entries.append((key, (left, right)))
    if not entries:
        raise InputError("Limits contain no entries")
    return Limits(Path(path).resolve(), digest, kind, NumericTable(tuple(entries)), tuple(zero_width))


def _boundaries(values, count):
    if not isinstance(values, (list, tuple)) or len(values) != count + 1:
        raise InputError("Region boundaries must have num_regions + 1 values")
    values = tuple(_number(value, "region boundary") for value in values)
    if values[0] != 0 or values[-1] != 1 or any(b <= a for a, b in zip(values, values[1:])):
        raise InputError("Region boundaries must strictly increase from 0 to 1")
    return values


@dataclass(frozen=True)
class Calibration:
    path: Path
    sha256: str
    num_regions: int
    values: NumericTable            # factors only; "0\t0" rows are keys without a factor
    region_boundaries: tuple | None
    region_provenance: str
    metadata: FrozenMapping
    layout: str = "position"        # "position": (time_ch, slab, region); "per_slab": (time_ch, slab)
    unfitted: int = 0               # "0\t0" rows (the reference writes every mapped key)
    non_positive: tuple = ()        # keys whose finite mu <= 0 (failed legacy fits) read as no factor


def load_calibration(path, mapping, *, expected_regions=None, region_boundaries=None,
                     metadata_path=None):
    if not isinstance(mapping, ChannelMap) or not Path(path).is_absolute():
        raise InputError("Calibration requires an absolute selected path and typed channel map")
    if metadata_path is not None and not Path(metadata_path).is_absolute():
        raise InputError("Calibration sidecar requires an absolute selected path")
    rows, digest = _rows(path)
    if rows and rows[0][1] == "ID(t_ch, slab)\tmu\tsigma":   # per-slab (KevConverter "cornell")
        layout, regions, body, arity = "per_slab", 1, rows[1:], 2
        if len(body) < 1:
            raise InputError("Per-slab calibration requires factors")
    else:
        if len(rows) < 3:
            raise InputError("Position calibration requires region header, columns and factors")
        match = re.fullmatch(r"# Position-dependent energy calibration \((\d+) regions per slab\)", rows[0][1])
        if not match or rows[1][1] != "ID(time_ch, slab, region)\tmu\tsigma":
            raise InputError("Unsupported calibration header/key schema")
        layout, regions, body, arity = "position", _integer(int(match[1]), "calibration regions", 1, 1000), rows[2:], 3
    if expected_regions is not None and _integer(expected_regions, "selected regions", 1, 1000) != regions:
        raise InputError("Selected region count differs from calibration")
    entries, unfitted, non_positive = [], 0, []
    for number, line in body:
        fields = line.split("\t")
        if len(fields) != 3:
            raise InputError(f"Calibration row {number}: expected tuple/mu/sigma")
        key = _key(fields[0], arity, mapping)
        if arity == 3:
            _integer(key[2], "calibration region", 0, regions - 1)
        try:
            if float(fields[1]) == 0 and float(fields[2]) == 0:
                unfitted += 1     # the reference's "no factor" row: a missing key, never a factor
                entries.append((key, None))
                continue
            mu = _number(float(fields[1]), "mu (a.u.)")
            sigma = _number(float(fields[2]), "sigma (a.u.)", nonnegative=True)
            if mu <= 0:           # the reference LM uses only mu > 0: a key without a factor (FR-12)
                non_positive.append(key)
                entries.append((key, None))
                continue
        except ValueError as exc:
            raise InputError(f"Calibration row {number}: {exc}") from exc
        entries.append((key, (mu, sigma)))
    seen = set()
    for key, _ in entries:
        if key in seen:
            raise InputError(f"Duplicate calibration key: {key}")
        seen.add(key)
    entries = [(key, value) for key, value in entries if value is not None]
    boundaries = None if region_boundaries is None else _boundaries(region_boundaries, regions)
    provenance = "header only; region boundaries unavailable" if boundaries is None else "operator supplied"
    metadata = {}
    if metadata_path is not None:
        try:
            metadata = json.loads(_metadata_bytes(metadata_path), object_pairs_hook=_unique_pairs,
                                  parse_constant=lambda value: (_ for _ in ()).throw(InputError("Nonfinite JSON")))
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise InputError(f"Invalid calibration sidecar: {exc}") from exc
        _finite_tree(metadata)
        if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
            raise InputError("Unsupported calibration sidecar schema")
        if _integer(metadata.get("num_regions"), "sidecar region count", 1, 1000) != regions:
            raise InputError("Sidecar region count differs from calibration")
        if metadata.get("layout", "position") != layout:
            raise InputError("Sidecar layout differs from calibration")
        supplied = _boundaries(metadata.get("region_boundaries"), regions)
        if boundaries is not None and any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                                           for a, b in zip(supplied, boundaries)):
            raise InputError("Sidecar and selected region boundaries disagree")
        if metadata.get("calibration_sha256") != digest:
            raise InputError("Calibration sidecar fingerprint is missing/stale")
        boundaries, provenance = supplied, str(Path(metadata_path).resolve())
    return Calibration(Path(path).resolve(), digest, regions, NumericTable(tuple(entries)),
                       boundaries, provenance, freeze(metadata), layout, unfitted, tuple(non_positive))


@dataclass(frozen=True)
class ValidationSummary:
    descriptor: InputDescriptor
    records: int
    detector_sides: int
    channel_hits: int
    bytes_read: int
    hit_limit: int | None
    peak_buffer_bytes: int
    file_size: int
    mtime_ns: int


def validate_ldat(descriptor, channels, *, batch_records=5000, max_batch_bytes=MAX_BATCH_BYTES,
                  expected_hit_limit=None, cancelled=None):
    """Full strict scan. Counts describe input pairs/groups/sides/hits before cuts.

    Only active fixed slots are checked; padding has no population. Byte layout
    alone never changes the explicit descriptor. A failed/cancelled scan returns
    no successful summary. Revalidate changed files at processing/ingest.
    """
    records, hits, consumed, peak, limit, before, _ = _scan(
        descriptor, channels, batch_records=batch_records, max_batch_bytes=max_batch_bytes,
        expected_hit_limit=expected_hit_limit, cancelled=cancelled, max_records=None)
    sides = 1 if descriptor.population == Population.GROUP else 2
    return ValidationSummary(replace(descriptor, validated=True), records, records * sides, hits,
                             consumed, limit, peak, before.st_size, before.st_mtime_ns)


@dataclass(frozen=True)
class ProbeSummary:
    """Bounded operator feedback, never a validation: the descriptor stays unvalidated."""
    descriptor: InputDescriptor
    records_checked: int
    channel_hits_checked: int
    complete: bool              # the whole file was checked (equivalent to a full scan)
    records_total: int | None   # fixed: exact from the size; compact: known only when complete
    file_size: int
    hit_limit: int | None


def probe_ldat(descriptor, channels, *, max_records=10000, expected_hit_limit=None, cancelled=None):
    """The validate_ldat checks on the first ``max_records`` records only.

    Catches a wrong declared format/population, truncation, unmapped channels or
    nonfinite energy near the start; a passing probe is not a full validation.
    """
    _integer(max_records, "max_records", 1, 1000000)
    records, hits, _, _, limit, before, complete = _scan(
        descriptor, channels, batch_records=min(max_records, 5000), max_batch_bytes=MAX_BATCH_BYTES,
        expected_hit_limit=expected_hit_limit, cancelled=cancelled, max_records=max_records)
    total = records if complete else None
    if descriptor.format == DataFormat.FIXED:
        sides = 1 if descriptor.population == Population.GROUP else 2
        total = (before.st_size - 4) // (sides + sides * limit * HIT.size)
    return ProbeSummary(replace(descriptor, validated=False), records, hits, complete, total, before.st_size, limit)


def _scan(descriptor, channels, *, batch_records, max_batch_bytes, expected_hit_limit, cancelled, max_records):
    if not isinstance(descriptor, InputDescriptor) or not descriptor.path.is_absolute():
        raise InputError("Validation requires an absolute explicit input descriptor")
    if descriptor.format == DataFormat.COMPACT and descriptor.population != Population.COINCIDENCE:
        raise InputError("Manager supports compact coincidence only")
    _integer(batch_records, "batch_records", 1, 1000000)
    _integer(max_batch_bytes, "max_batch_bytes", 8192, MAX_BATCH_BYTES)
    if expected_hit_limit is not None:
        _integer(expected_hit_limit, "expected_hit_limit", 1, 255)
    if not isinstance(channels, Mapping) or not channels:
        raise InputError("Validation requires the selected channel mapping")
    if not stat.S_ISREG(descriptor.path.stat().st_mode):
        raise InputError("LDAT input must be a regular file")
    sides = 1 if descriptor.population == Population.GROUP else 2
    records = hits = consumed = peak = 0
    limit = None
    stopped = False
    def check_stop():
        if cancelled is not None and cancelled():
            raise ValidationCancelled("Input validation cancelled")
    def hit_check(payload, count, offset, record, side):
        for index in range(count):
            _, energy, channel = HIT.unpack_from(payload, offset + index * HIT.size)
            if channel not in channels:
                raise InputError(f"{descriptor.path}: record {record}, side {side}: unmapped channel {channel}")
            if not math.isfinite(energy):
                raise InputError(f"{descriptor.path}: record {record}, side {side}: nonfinite energy")
    with descriptor.path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size == 0:
            raise InputError(f"Empty LDAT: {descriptor.path}")
        check_stop()
        if descriptor.format == DataFormat.FIXED:
            header = stream.read(4)
            if len(header) != 4:
                raise InputError("Truncated fixed hit-limit header")
            limit = _integer(struct.unpack("<i", header)[0], "fixed hit limit", 1, 255)
            if expected_hit_limit is not None and limit != expected_hit_limit:
                raise InputError("Fixed hit limit differs from explicit conversion settings")
            record_bytes = sides + sides * limit * HIT.size
            if before.st_size <= 4 or (before.st_size - 4) % record_bytes:
                raise InputError("Empty/truncated fixed records or inconsistent remainder/population")
            batch_bytes = min(batch_records, max_batch_bytes // record_bytes) * record_bytes
            consumed = 4
            while not stopped:
                check_stop()
                payload = stream.read(batch_bytes)
                if not payload:
                    break
                if len(payload) % record_bytes:
                    raise InputError("Truncated fixed record during reading")
                peak = max(peak, len(payload))
                for offset in range(0, len(payload), record_bytes):
                    if max_records is not None and records >= max_records:
                        stopped = True
                        break
                    for side in range(sides):
                        count = payload[offset + side]
                        if not 1 <= count <= limit:
                            raise InputError(f"Invalid fixed hit count at record {records}, side {side}")
                        hit_check(payload, count, offset + sides + side * limit * HIT.size, records, side)
                        hits += count
                    records += 1
                consumed += len(payload)
                del payload
            complete = 4 + records * record_bytes == before.st_size
        else:
            while True:
                check_stop()
                if max_records is not None and records >= max_records:
                    break
                header = stream.read(2)
                if not header:
                    break
                if len(header) != 2:
                    raise InputError("Truncated compact coincidence header")
                consumed += 2
                for side, count in enumerate(header):
                    if not 1 <= count <= (expected_hit_limit or 255):
                        raise InputError(f"Invalid compact hit count at record {records}, side {side}")
                    payload = stream.read(count * HIT.size)
                    if len(payload) != count * HIT.size:
                        raise InputError(f"Truncated compact hits at record {records}, side {side}")
                    hit_check(payload, count, 0, records, side)
                    peak = max(peak, len(payload) + 2)
                    hits += count
                    consumed += len(payload)
                    del payload
                records += 1
            complete = consumed == before.st_size
        after = os.fstat(stream.fileno())
    current = descriptor.path.stat()
    fingerprint = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(current) or (
            max_records is None and consumed != before.st_size):
        raise InputError("Input changed during validation")
    check_stop()
    return records, hits, consumed, peak, limit, before, complete
