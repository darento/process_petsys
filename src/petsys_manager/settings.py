"""Versioned profiles and action-specific, read-only prerequisite checks.

No GUI/hardware imports, process launches, input mutations or numeric decoding.
Profile paths are literal; relative processing paths use an explicit root, not
the current directory or the directory containing a selected processing YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import math
import os
import re
from pathlib import Path, PurePosixPath
import sys
import tempfile

import yaml

from .contracts import (MANAGER_ROUTE, Action, DataFormat, FrozenMapping, InputDescriptor, Population,
                        SourceMode, freeze, to_plain)


SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 4 * 1024 * 1024


class ProfileError(ValueError):
    """Invalid profile shape/version/value; never silently reset a profile."""


def _number(value, name, *, minimum=0, strict=True):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ProfileError(f"{name} must be a finite number")
    if value < minimum or (strict and value == minimum):
        raise ProfileError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")


def _integer(value, name, *, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ProfileError(f"{name} must be an integer in {minimum}..{maximum or 'unbounded'}")


def _bool(value, name):
    if type(value) is not bool:
        raise ProfileError(f"{name} must be boolean")


def _text(value, name, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ProfileError(f"{name} must be nonempty text without NUL")


def _absolute_path(value):
    # Linux profiles may be loaded for headless checks on Windows.
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute()


def validate_calibration_factor(value):
    """Validation primitive for future calibration-file checks, not a fallback."""
    _number(value, "calibration factor")
    return value


@dataclass(frozen=True)
class AcquisitionSafety:
    startup_timeout_s: float = 45.0
    growth_window_s: float = 20.0
    poll_interval_s: float = 5.0
    min_growth_bytes: int = 20_000_000
    max_loss_percent: float = 5.0
    max_attempts: int = 3
    retry_delay_s: float = 2.0
    terminate_grace_s: float = 3.0

    def __post_init__(self):
        for name in ("startup_timeout_s", "growth_window_s", "poll_interval_s", "terminate_grace_s"):
            _number(getattr(self, name), name)
        _number(self.retry_delay_s, "retry_delay_s", strict=False)
        _number(self.max_loss_percent, "max_loss_percent", strict=False)
        if self.max_loss_percent > 100:
            raise ProfileError("max_loss_percent must not exceed 100")
        _integer(self.min_growth_bytes, "min_growth_bytes", minimum=0)
        _integer(self.max_attempts, "max_attempts")


@dataclass(frozen=True)
class ProcessingLimits:
    workers: int = 0                            # FR-15: worker processes; 0 = automatic (CPU count - 2, at least 1)
    batch_records: int = 5000
    calibration_event_limit: int = 10_000_000   # reference mode: a file stops once more events have passed
    calibration_target_per_key: int = 3_000     # target mode: T kept sides per histogram (FR-21)
    calibration_memory_mb: int = 8_192          # target mode: decode each file once within this budget (FR-15)
    lm_seed: int = 0                            # LM: per-file random streams for ambiguous slabs (FR-22)
    qc_pair_limit: int = 1_000_001
    log_tail_lines: int = 1000

    def __post_init__(self):
        for item in fields(self):
            _integer(getattr(self, item.name), item.name, minimum=0 if item.name in ("workers", "lm_seed") else 1)


@dataclass(frozen=True)
class ToolCapabilities:
    custom_socket_confirmed: bool = False
    installed_version: str | None = None

    def __post_init__(self):
        _bool(self.custom_socket_confirmed, "custom_socket_confirmed")
        _text(self.installed_version, "installed_version", optional=True)


@dataclass(frozen=True)
class LMMetadata:
    # No default measured duration/geometry inferred from the legacy writer.
    isotope: str | None = None
    acquisition_time_s: float | None = None
    measurement_time_s: float | None = None
    detector_size_x_mm: float | None = None
    detector_size_y_mm: float | None = None
    module_number: int | None = None
    ring_number: int | None = None
    ring_distance_mm: float | None = None
    detector_pixels_x: int | None = None
    detector_pixels_y: int | None = None
    timestamp_unit: str | None = None

    def __post_init__(self):
        for name in ("isotope", "timestamp_unit"):
            _text(getattr(self, name), name, optional=True)
        if self.isotope is not None and len(self.isotope.encode("utf-8")) > 16:
            raise ProfileError("isotope must fit the 16-byte LM header field")
        for name in ("acquisition_time_s", "measurement_time_s", "detector_size_x_mm",
                     "detector_size_y_mm", "ring_distance_mm"):
            if getattr(self, name) is not None:
                _number(getattr(self, name), name)
        for name in ("module_number", "ring_number"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, maximum=2**31 - 1)
        for name in ("detector_pixels_x", "detector_pixels_y"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, maximum=127)

    def missing(self):
        return tuple(item.name for item in fields(self) if getattr(self, item.name) is None)


# PETsys tools that are Python scripts (FR-23): run with the profile's petsys_python when set.
PETSYS_PYTHON_TOOLS = ("init_system", "acquire_sipm_data", "set_bias")

_PATH_FIELDS = ("petsys_folder", "petsys_python", "processing_root", "ini_file", "yaml_file", "data_dir",
                "calibration_dir", "report_dir", "lm_dir", "cog_limits_file", "doi_limits_file",
                "calibration_file", "pair_map_file", "region_map_file")


@dataclass(frozen=True)
class MachineProfile:
    schema_version: int = SCHEMA_VERSION
    petsys_folder: str | None = None
    petsys_python: str | None = None      # interpreter PETsys was installed for (FR-23); None: tool shebang
    processing_root: str | None = None
    ini_file: str | None = None
    yaml_file: str | None = None
    data_dir: str | None = None
    calibration_dir: str | None = None
    report_dir: str | None = None
    lm_dir: str | None = None
    cog_limits_file: str | None = None
    doi_limits_file: str | None = None
    calibration_file: str | None = None
    pair_map_file: str | None = None
    region_map_file: str | None = None
    daq_type: str = "PFP_KX7"
    cards: tuple[str, ...] = ("/dev/psdaq1", "/dev/psdaq0")
    socket_path: str = "/tmp/d.sock"
    shared_memory_path: str = "/dev/shm/daqd_shm"
    safety: AcquisitionSafety = field(default_factory=AcquisitionSafety)
    limits: ProcessingLimits = field(default_factory=ProcessingLimits)
    capabilities: ToolCapabilities = field(default_factory=ToolCapabilities)
    lm_metadata: LMMetadata = field(default_factory=LMMetadata)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ProfileError(f"Unsupported profile schema_version: {self.schema_version!r}")
        for name in _PATH_FIELDS:
            _text(getattr(self, name), name, optional=True)
        for name in ("daq_type", "socket_path", "shared_memory_path"):
            _text(getattr(self, name), name)
        if self.petsys_python is not None and not _absolute_path(self.petsys_python):
            raise ProfileError("petsys_python must be an absolute interpreter path")
        if not _absolute_path(self.socket_path) or not _absolute_path(self.shared_memory_path):
            raise ProfileError("DAQ socket/shared-memory paths must be absolute")
        if isinstance(self.cards, str) or not isinstance(self.cards, (tuple, list)):
            raise ProfileError("cards must be a sequence of device paths")
        object.__setattr__(self, "cards", tuple(self.cards))
        for card in self.cards:
            _text(card, "card")
            if not _absolute_path(card):
                raise ProfileError("DAQ card paths must be absolute")
        if len(self.cards) > 2 or len(set(self.cards)) != len(self.cards):
            raise ProfileError("Select at most two distinct DAQ cards")
        for name, cls in (("safety", AcquisitionSafety), ("limits", ProcessingLimits),
                          ("capabilities", ToolCapabilities), ("lm_metadata", LMMetadata)):
            if not isinstance(getattr(self, name), cls):
                raise ProfileError(f"{name} must be {cls.__name__}")


ACQUISITION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}")


@dataclass(frozen=True)
class RunOptions:
    duration_s: float = 10.0
    acquisition_mode: str = "qdc"
    hardware_trigger: bool = False
    splits: int = 1
    hit_limit: int = 16
    output_format: DataFormat = DataFormat.COMPACT     # the only manager route (FR-10)
    population: Population = Population.COINCIDENCE
    source_mode: SourceMode = SourceMode.WITH
    plots: bool = False
    slabs: bool = False
    debug: bool = False
    regions: int = 5
    calibration_limit_mode: str = "target"      # FR-21: "target" (from T per key) or "reference" (10 M per file)
    raw_input: str | None = None
    acquisition_name: str = "acquisition"   # RAW basename and run-folder data name of acquire/pipeline/QC (T29)

    def __post_init__(self):
        _number(self.duration_s, "duration_s")
        if self.acquisition_mode not in ("qdc", "tot", "mixed"):
            raise ProfileError("acquisition_mode must be qdc, tot or mixed")
        _bool(self.hardware_trigger, "hardware_trigger")
        _integer(self.splits, "splits")
        _integer(self.hit_limit, "hit_limit", maximum=255)
        _integer(self.regions, "regions", maximum=127)
        if self.calibration_limit_mode not in ("target", "reference"):
            raise ProfileError("calibration_limit_mode must be target or reference")
        for name in ("plots", "slabs", "debug"):
            _bool(getattr(self, name), name)
        if self.slabs and not self.plots:
            raise ProfileError("Slab analysis requires plots")
        _text(self.raw_input, "raw_input", optional=True)
        if not isinstance(self.acquisition_name, str) or not ACQUISITION_NAME.fullmatch(self.acquisition_name):
            raise ProfileError("Acquisition name: 1-48 letters, digits, '_' or '-', starting with a letter or digit")
        try:
            object.__setattr__(self, "output_format", DataFormat(self.output_format))
            object.__setattr__(self, "population", Population(self.population))
            object.__setattr__(self, "source_mode", SourceMode(self.source_mode))
        except ValueError as exc:
            raise ProfileError(str(exc)) from exc
        if (self.output_format, self.population) != MANAGER_ROUTE:
            raise ProfileError("The manager converts and processes compact coincidence only (FR-10); "
                               f"{self.output_format.value} {self.population.value} is not offered")


PIPELINE_LM_TIME_FIELDS = ("acquisition_time_s", "measurement_time_s")


def lm_header_metadata(profile, action, options):
    """LM header metadata a run writes. The pipeline writes its own Acq. Time as both header times
    (owner, 2026-10-02); manual LM uses the profile values as given."""
    if Action(action) != Action.PIPELINE:
        return profile.lm_metadata
    return replace(profile.lm_metadata, **{name: options.duration_s for name in PIPELINE_LM_TIME_FIELDS})


class _StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in result:
                raise ProfileError(f"Duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=True)
        except TypeError as exc:
            raise ProfileError("YAML mapping key must be scalar") from exc
    return result


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read_yaml(path):
    try:
        with path.open("rb") as source:
            content = source.read(MAX_CONFIG_BYTES + 1)
        if len(content) > MAX_CONFIG_BYTES:
            raise ProfileError(f"Configuration exceeds {MAX_CONFIG_BYTES} bytes: {path}")
        return yaml.load(content, Loader=_StrictLoader)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ProfileError(f"Invalid YAML: {path}: {exc}") from exc


def _construct(cls, value, label):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProfileError(f"{label} must be a mapping with text keys")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise ProfileError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    try:
        return cls(**value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"Invalid {label}: {exc}") from exc


RETIRED_CAPABILITIES = ("fixed_output_confirmed",)   # FR-10 (2026-10-05): read from old profiles, ignored


def profile_from_mapping(value):
    if not isinstance(value, dict) or "schema_version" not in value:
        raise ProfileError("Profile requires an explicit schema_version")
    data = dict(value)
    if isinstance(data.get("capabilities"), dict):
        data["capabilities"] = {key: item for key, item in data["capabilities"].items()
                                if key not in RETIRED_CAPABILITIES}
    for name, cls in (("safety", AcquisitionSafety), ("limits", ProcessingLimits),
                      ("capabilities", ToolCapabilities), ("lm_metadata", LMMetadata)):
        if name in data:
            data[name] = _construct(cls, data[name], name)
    return _construct(MachineProfile, data, "profile")


def default_profile_path():
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".config"
    return base / "process_petsys" / "petsys_manager.yaml"


def load_profile(path=None):
    target = default_profile_path() if path is None else Path(path)
    if path is None and not target.exists():
        return MachineProfile()
    return profile_from_mapping(_read_yaml(target))


def save_profile(profile, path=None, *, overwrite=False):
    """Write only a selected application profile, never processing config/maps.

    New paths use exclusive creation. Replacing an existing path requires an
    explicit overwrite request AND an existing valid manager-profile schema.
    """
    if not isinstance(profile, MachineProfile):
        raise ProfileError("Expected a typed manager profile")
    target = default_profile_path() if path is None else Path(path)
    content = yaml.safe_dump(to_plain(profile), sort_keys=False, allow_unicode=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite or not target.exists():
        with target.open("x", encoding="utf-8") as out:
            out.write(content)
        return target
    load_profile(target)  # Refuse replacing arbitrary INI/YAML/data with a profile.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".petsys-profile-", delete=False) as out:
            temporary = Path(out.name)
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


@dataclass(frozen=True)
class PrerequisiteIssue:
    field: str
    message: str


@dataclass(frozen=True)
class RunSettings:
    action: Action
    profile: MachineProfile
    options: RunOptions
    processing_root: Path
    paths: FrozenMapping
    inputs: tuple[InputDescriptor, ...]
    processing_config: FrozenMapping

    def __post_init__(self):
        object.__setattr__(self, "action", Action(self.action))
        if not isinstance(self.profile, MachineProfile) or not isinstance(self.options, RunOptions):
            raise ProfileError("Run snapshot requires a typed profile and options")
        root = Path(self.processing_root)
        if not root.is_absolute():
            raise ProfileError("Run snapshot processing root must be absolute")
        object.__setattr__(self, "processing_root", root)
        inputs = tuple(self.inputs)
        if any(not isinstance(item, InputDescriptor) for item in inputs):
            raise ProfileError("Run snapshot inputs must be typed descriptors")
        object.__setattr__(self, "inputs", inputs)
        for name in ("paths", "processing_config"):
            value = freeze(getattr(self, name))
            if not isinstance(value, FrozenMapping):
                raise ProfileError(f"Run snapshot {name} must be a mapping")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PreflightReport:
    action: Action
    issues: tuple[PrerequisiteIssue, ...]
    paths: FrozenMapping
    settings: RunSettings | None

    @property
    def ready(self):
        return not self.issues and self.settings is not None


class SystemProbe:
    """Read-only filesystem/OS probes; tests substitute these, never DAQ tools."""

    platform_name = {"linux": "Linux", "win32": "Windows", "darwin": "Darwin"}.get(sys.platform, sys.platform)

    def file(self, path):
        return path.is_file()

    def executable(self, path):
        return path.is_file() and os.access(path, os.X_OK)

    def device(self, path):
        return path.is_char_device()

    def writable_directory(self, path):
        candidate = path
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK)


def _resolve(value, root):
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def preflight(profile, action, options=None, inputs=(), *, repo_root=None, probe=None):
    """Collect all known prerequisites; defer numerical/file schemas to T5.

    Does not launch hardware, create destinations, remove daemon resources or
    declare readiness. Live daemon/initialization state belongs to T6/T12.
    """
    if not isinstance(profile, MachineProfile):
        raise ProfileError("Expected a typed manager profile")
    action = Action(action)
    options = RunOptions() if options is None else options
    if not isinstance(options, RunOptions):
        raise ProfileError("Expected typed run options")
    inputs = tuple(inputs)
    if any(not isinstance(item, InputDescriptor) for item in inputs):
        raise ProfileError("Inputs must have explicit format/population descriptors")
    probe = SystemProbe() if probe is None else probe
    checkout = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    root = _resolve(profile.processing_root, checkout) or checkout
    paths = {name: _resolve(getattr(profile, name), root) for name in _PATH_FIELDS}
    paths["processing_root"] = root
    if profile.petsys_python is not None:      # as given: resolving a venv's symlinked python would leave the venv
        paths["petsys_python"] = Path(profile.petsys_python).expanduser()
    if options.raw_input is not None:
        paths["raw_input"] = _resolve(options.raw_input, root)
    issues = []
    def issue(name, message):
        issues.append(PrerequisiteIssue(name, message))
    def need_file(name):
        path = paths.get(name)
        if path is None or not probe.file(path):
            issue(name, "Select an existing file" if path is None else f"File not found: {path}")
            return False
        return True
    def need_output(name):
        path = paths.get(name)
        if path is None or not probe.writable_directory(path):
            issue(name, "Select a writable destination" if path is None else f"Destination not writable: {path}")

    live = action in (Action.DAQD, Action.INITIALIZE, Action.ACQUIRE, Action.QC, Action.PIPELINE)
    processing = action in (Action.CALIBRATE, Action.LISTMODE, Action.QC, Action.QC_ANALYZE, Action.PIPELINE)
    conversion = action in (Action.CONVERT, Action.QC, Action.PIPELINE)
    if live and probe.platform_name != "Linux":
        issue("platform", "Hardware operations require the Cornell Linux machine")
    tool_names = set()
    if live:
        tool_names.add("daqd")
        if profile.daq_type == "PFP_KX7":
            if not profile.cards:
                issue("cards", "Select one or two DAQ character devices")
            for card in profile.cards:
                if not probe.device(Path(card)):
                    issue("cards", f"DAQ character device not found: {card}")
        if len(os.fsencode(profile.socket_path)) > 107:
            issue("socket_path", "Unix socket path exceeds 107 bytes")
        if profile.socket_path != "/tmp/d.sock" and not profile.capabilities.custom_socket_confirmed:
            issue("socket_path", "Custom-socket support must be confirmed for DAQD, init and acquisition tools")
    if action in (Action.INITIALIZE, Action.ACQUIRE, Action.QC, Action.PIPELINE):
        tool_names.add("init_system")
        need_file("ini_file")
    if action in (Action.ACQUIRE, Action.QC, Action.PIPELINE):
        tool_names.update(("acquire_sipm_data", "set_bias"))  # set_bias: FR-19 bias-off after an abort
    if conversion:
        tool_names.add("convert_raw_to_coincidence")   # compact coincidence only (FR-10)
        need_file("ini_file")
    if action == Action.CONVERT and need_file("raw_input"):
        # PETsys RawReader opens <prefix>.rawf plus <prefix>.tmpf or <prefix>.idxf.
        raw = paths["raw_input"]
        if raw.suffix != ".rawf":
            issue("raw_input", f"Select the acquisition's .rawf file; converters read <prefix>.rawf: {raw}")
        elif not (probe.file(raw.with_suffix(".idxf")) or probe.file(raw.with_suffix(".tmpf"))):
            issue("raw_input", f"RAW index not found (the converter reads it): {raw.with_suffix('.idxf')}")
    if tool_names & set(PETSYS_PYTHON_TOOLS) and paths["petsys_python"] is not None             and not probe.executable(paths["petsys_python"]):
        issue("petsys_python", f"PETsys Python interpreter not found or not executable: {paths['petsys_python']}")
    for name in sorted(tool_names):
        path = paths["petsys_folder"] / name if paths["petsys_folder"] is not None else None
        paths[f"tool:{name}"] = path
        if path is None or not probe.executable(path):
            issue(f"tool:{name}", "Select the PETsys tools folder" if path is None else f"Tool unavailable/not executable: {path}")

    if action in (Action.ACQUIRE, Action.CONVERT, Action.QC, Action.PIPELINE):
        need_output("data_dir")
    if action in (Action.CALIBRATE, Action.PIPELINE):
        need_output("calibration_dir")
    if action in (Action.CALIBRATE, Action.QC, Action.QC_ANALYZE, Action.PIPELINE) or (
            action == Action.LISTMODE and options.debug):
        need_output("report_dir")
    if action in (Action.LISTMODE, Action.PIPELINE):
        need_output("lm_dir")
        for name in lm_header_metadata(profile, action, options).missing():
            issue(f"lm_metadata.{name}", "Required LM profile/measurement metadata is unavailable")
    if action in (Action.LISTMODE, Action.PIPELINE) or (action == Action.CALIBRATE and options.regions > 1):
        need_file("cog_limits_file")   # per-slab calibration (positions 1) needs no COG limits
    if action in (Action.LISTMODE, Action.PIPELINE):
        for name in ("doi_limits_file", "pair_map_file", "region_map_file"):
            need_file(name)
    if action == Action.LISTMODE:
        need_file("calibration_file")

    config = FrozenMapping()
    if processing and need_file("yaml_file"):
        try:
            value = _read_yaml(paths["yaml_file"])
            if not isinstance(value, dict):
                raise ProfileError("Processing YAML must be a mapping")
            _text(value.get("map_file"), "map_file")
            paths["map_file"] = _resolve(value["map_file"], root)
            need_file("map_file")
            _integer(value.get("min_ch"), "min_ch")
            if "en_min_ch" in value:
                _number(value["en_min_ch"], "en_min_ch", strict=False)
            if "en_min_ch" not in value:
                raise ProfileError("Processing YAML requires en_min_ch for this action")
            if action in (Action.LISTMODE, Action.PIPELINE):
                window = value.get("energy_range")
                if not isinstance(window, (list, tuple)) or len(window) != 2:
                    raise ProfileError("LM requires a two-value keV energy_range")
                _number(window[0], "energy_range lower", strict=False)
                _number(window[1], "energy_range upper")
                if window[1] <= window[0]:
                    raise ProfileError("energy_range upper must exceed lower")
            config = freeze(value)
        except (OSError, ValueError, TypeError) as exc:
            issue("yaml_file", str(exc))

    if action in (Action.CALIBRATE, Action.LISTMODE, Action.QC_ANALYZE):
        if not inputs:
            issue("inputs", "Select the exact ordered input files")
        resolved = []
        seen = set()
        for item in inputs:
            path = _resolve(str(item.path), root)
            if not probe.file(path):
                issue("inputs", f"Input file not found: {path}")
            if path in seen:
                issue("inputs", f"Duplicate selected input: {path}")
            seen.add(path)
            if (item.format, item.population) != MANAGER_ROUTE:   # FR-10: the library still reads fixed
                issue("inputs", f"Unsupported format/population for {action.value}: {path}")
            resolved.append(replace(item, path=path))
        inputs = tuple(resolved)
    if action == Action.QC:
        options = replace(options, duration_s=60.0 if options.source_mode == SourceMode.WITH else 180.0)
    issues = list(dict.fromkeys(issues))
    frozen_paths = freeze(paths)
    settings = None if issues else RunSettings(action, profile, options, root, frozen_paths, inputs, config)
    return PreflightReport(action, tuple(issues), frozen_paths, settings)
