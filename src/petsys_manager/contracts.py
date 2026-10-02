"""Immutable, toolkit-independent contracts for PETsys Manager."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Action(str, Enum):
    DAQD = "daqd"
    INITIALIZE = "initialize"
    ACQUIRE = "acquire"
    CONVERT = "convert"
    CALIBRATE = "calibrate"
    LISTMODE = "listmode"
    QC = "qc"
    QC_ANALYZE = "qc_analyze"
    PIPELINE = "pipeline"


class DataFormat(str, Enum):
    FIXED = "fixed"
    COMPACT = "compact"


class Population(str, Enum):
    COINCIDENCE = "coincidence"
    GROUP = "group"


class SourceMode(str, Enum):
    WITH = "with"
    WITHOUT = "without"


def route_accepts(action, data_format, population):
    """Format routes: calibration takes fixed coincidence/group or compact coincidence (FR-21);
    LM fixed coincidence; offline QC compact coincidence."""
    action, data_format, population = Action(action), DataFormat(data_format), Population(population)
    if action == Action.CALIBRATE:
        return data_format == DataFormat.FIXED or population == Population.COINCIDENCE
    if action in (Action.LISTMODE, Action.PIPELINE):
        return (data_format, population) == (DataFormat.FIXED, Population.COINCIDENCE)
    if action in (Action.QC_ANALYZE, Action.QC):
        return (data_format, population) == (DataFormat.COMPACT, Population.COINCIDENCE)
    return False


class ResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LAUNCH_ERROR = "launch_error"


@dataclass(frozen=True)
class FrozenMapping(Mapping):
    """Small, pickle-friendly immutable mapping, including nested values."""

    entries: tuple[tuple[Any, Any], ...] = ()

    def __post_init__(self):
        entries = tuple((key, freeze(value)) for key, value in self.entries)
        if any(type(key) not in (str, int) for key, _ in entries):
            raise ValueError("Configuration keys must be strings or integers")
        if len({key for key, _ in entries}) != len(entries):
            raise ValueError("Duplicate immutable mapping key")
        object.__setattr__(self, "entries", entries)

    def __getitem__(self, key):
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator:
        return (key for key, _ in self.entries)

    def __len__(self):
        return len(self.entries)


def freeze(value, _ancestors=None):
    """Copy configuration containers; reject recursive/unsupported payloads."""
    if isinstance(value, FrozenMapping):
        return value
    if value is None or isinstance(value, (str, bool, int, float, Path, Enum)):
        return value
    ancestors = set() if _ancestors is None else _ancestors
    if id(value) in ancestors:
        raise ValueError("Recursive configuration container")
    if isinstance(value, (Mapping, list, tuple)):
        ancestors = ancestors | {id(value)}
        if isinstance(value, Mapping):
            if any(type(key) not in (str, int) for key in value):
                raise ValueError("Configuration keys must be strings or integers")
            return FrozenMapping(tuple((key, freeze(item, ancestors)) for key, item in value.items()))
        return tuple(freeze(item, ancestors) for item in value)
    raise ValueError(f"Unsupported configuration value: {type(value).__name__}")


def to_plain(value):
    """Explicit JSON/YAML representation without mutable aliases to a snapshot."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: to_plain(item) for key, item in value.items()}
    if is_dataclass(value):
        return {field.name: to_plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [to_plain(item) for item in value]
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{name} must be nonempty text without NUL")
    return value


def _path(value, name):
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a path or text")
    _text(str(value), name)
    return Path(value)


@dataclass(frozen=True)
class InputDescriptor:
    path: Path
    format: DataFormat
    population: Population
    validated: bool = False

    def __post_init__(self):
        object.__setattr__(self, "path", _path(self.path, "Input path"))
        object.__setattr__(self, "format", DataFormat(self.format))
        object.__setattr__(self, "population", Population(self.population))
        if type(self.validated) is not bool:
            raise ValueError("Input validation state must be boolean")


@dataclass(frozen=True)
class Identity:
    run_id: str
    stage_id: str
    attempt_id: str

    def __post_init__(self):
        for name in ("run_id", "stage_id", "attempt_id"):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    input_descriptor: InputDescriptor | None = None
    disposable: bool = False

    def __post_init__(self):
        object.__setattr__(self, "path", _path(self.path, "Artifact path"))
        _text(self.kind, "Artifact kind")
        if type(self.disposable) is not bool:
            raise ValueError("Artifact disposable state must be boolean")
        if self.input_descriptor is not None and not isinstance(self.input_descriptor, InputDescriptor):
            raise ValueError("Artifact input descriptor has an invalid type")


@dataclass(frozen=True)
class CommandSpec:
    """Literal argv and a complete environment snapshot, not shell text."""
    argv: tuple[str, ...]
    cwd: Path
    identity: Identity
    environment: FrozenMapping = FrozenMapping()

    def __post_init__(self):
        if isinstance(self.argv, str) or not self.argv:
            raise ValueError("Command argv must be a nonempty argument sequence, not a shell string")
        args = tuple(_text(arg, "Command argument") for arg in self.argv)
        object.__setattr__(self, "argv", args)
        object.__setattr__(self, "cwd", _path(self.cwd, "Command cwd"))
        if not self.cwd.is_absolute() or not isinstance(self.identity, Identity):
            raise ValueError("Command requires an absolute cwd and typed identity")
        env = freeze(self.environment)
        if not isinstance(env, FrozenMapping) or any(
                not isinstance(key, str) or not isinstance(value, str) or "\0" in key + value
                for key, value in env.items()):
            raise ValueError("Command environment must contain text keys/values without NUL")
        object.__setattr__(self, "environment", env)


@dataclass(frozen=True)
class CommandResult:
    identity: Identity
    status: ResultStatus
    exit_code: int | None = None
    message: str = ""
    artifacts: tuple[Artifact, ...] = ()
    log_tail: tuple[str, ...] = ()
    outputs_validated: bool = False

    def __post_init__(self):
        object.__setattr__(self, "status", ResultStatus(self.status))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "log_tail", tuple(self.log_tail))
        if not isinstance(self.identity, Identity):
            raise ValueError("Result requires a typed identity")
        if not isinstance(self.message, str):
            raise ValueError("Result message must be text")
        if type(self.outputs_validated) is not bool:
            raise ValueError("Output validation state must be boolean")
        if self.outputs_validated and self.status != ResultStatus.SUCCEEDED:
            raise ValueError("Only successful results can have validated outputs")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("Exit code must be an integer or unavailable")
        if self.status == ResultStatus.SUCCEEDED and self.exit_code != 0:
            raise ValueError("Successful command must have exit code zero")
        if self.status == ResultStatus.FAILED and self.exit_code in (None, 0) and not self.message:
            raise ValueError("Failure without nonzero exit requires a validation/cleanup reason")
        if self.status == ResultStatus.LAUNCH_ERROR and self.exit_code is not None:
            raise ValueError("Launch error has no child exit code")
        if any(not isinstance(item, Artifact) for item in self.artifacts):
            raise ValueError("Result artifacts must be typed")
        if any(not isinstance(line, str) for line in self.log_tail):
            raise ValueError("Result log lines must be text")

    @property
    def can_advance(self):
        return self.status == ResultStatus.SUCCEEDED and self.outputs_validated


@dataclass(frozen=True)
class OutputValidation:
    """A stage validator's explicit decision; no filesystem guessing in runner."""

    valid: bool
    message: str = ""
    artifacts: tuple[Artifact, ...] = ()

    def __post_init__(self):
        if type(self.valid) is not bool or not isinstance(self.message, str):
            raise ValueError("Validation requires a boolean decision and text reason")
        if not self.valid and not self.message.strip():
            raise ValueError("Rejected outputs require a reason")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, Artifact) for item in artifacts):
            raise ValueError("Validated artifacts must be typed")
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True)
class RunEvent:
    identity: Identity
    sequence: int
    kind: str
    message: str = ""
    payload: FrozenMapping = FrozenMapping()

    def __post_init__(self):
        if not isinstance(self.identity, Identity) or type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Event requires an identity and nonnegative integer sequence")
        _text(self.kind, "Event kind")
        if not isinstance(self.message, str):
            raise ValueError("Event message must be text")
        payload = freeze(self.payload)
        if not isinstance(payload, FrozenMapping):
            raise ValueError("Event payload must be a mapping")
        object.__setattr__(self, "payload", payload)
