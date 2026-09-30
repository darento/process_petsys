"""Exclusive run storage, append-only manifests and conservative file ownership.

One worker owns each store. No existing run can be reopened for writing. Recovery
is read-only until a workflow-specific resume contract exists. Records are file
metadata, never acquisition records. Numerical validation belongs to T5 onward.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from uuid import uuid4

from .contracts import (Artifact, CommandResult, Identity, InputDescriptor,
                        ResultStatus, freeze, to_plain)


class ArtifactError(ValueError):
    """Invalid ownership, state, input or publication contract."""


def _name(value):
    # Portable identities: no traversal, Windows drive/stream syntax or devices.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ArtifactError("Identity must be a portable path component")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))}:
        raise ArtifactError("Reserved device identity")
    return value


def _absolute(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ArtifactError("Expected an absolute path without traversal")
    return path


def _stat(path, *, directory=False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ArtifactError(f"Not a plain {'directory' if directory else 'file'}: {path}")
    return info


def _parents(path):
    for parent in reversed((path, *path.parents)):
        _stat(parent, directory=True)


def _token(info):
    return (info.st_dev, info.st_ino)


def _file_token(info):
    return (*_token(info), info.st_size, info.st_mtime_ns)


def _sync_directory(path):
    # Windows does not expose directory fsync through this API. Linux deployment
    # persists both file contents and directory entries; Windows fixtures fsync files.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _link(source, target):
    """Atomic no-replace publication on the same filesystem; no unsafe fallback."""
    os.link(source, target, follow_symlinks=False)


def _sync_file(path, expected):
    flags = (os.O_RDWR if os.name == "nt" else os.O_RDONLY) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if _file_token(os.fstat(fd)) != expected:
            raise ArtifactError("File identity changed before flushing")
        os.fsync(fd)
    finally:
        os.close(fd)


def _artifact_contract(value):
    descriptor = value["input_descriptor"]
    if descriptor is not None:
        descriptor = {k: v for k, v in descriptor.items() if k != "validated"}
    return value["kind"], value["disposable"], descriptor


def _natural(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", path.name)) + ((0, path.name),)


@dataclass(frozen=True)
class Attempt:
    identity: Identity
    directory: Path


class RunStore:
    """Single-writer store. A snapshot becomes current only after durable publication.

    Manifests contain settings, ordered inputs, attempts and exact artifacts.
    Active/interrupted work is explicitly ``partial``. ``succeeded`` requires
    a validated successful final attempt for every stage and unchanged outputs.
    """

    manifest_limit_bytes = 4 * 1024 * 1024
    artifact_limit = 10000

    def __init__(self):
        raise TypeError("Use RunStore.reserve()")

    @classmethod
    def reserve(cls, destination, settings, inputs=(), *, run_id=None):
        destination = _absolute(destination)
        _parents(destination)  # Destination is operator-selected, never auto-created.
        run_id = _name(run_id or f"run-{uuid4().hex}")
        plain_settings = to_plain(settings)
        if not isinstance(plain_settings, dict):
            raise ArtifactError("Settings must be a mapping or typed settings snapshot")
        records = []
        seen = set()
        for item in inputs:
            if not isinstance(item, InputDescriptor):
                raise ArtifactError("Inputs require explicit format/population descriptors")
            path = _absolute(item.path)
            _parents(path.parent)
            info = _stat(path)
            if path in seen:
                raise ArtifactError("Duplicate exact input")
            seen.add(path)
            records.append({**to_plain(item), "size_bytes": info.st_size,
                            "mtime_ns": info.st_mtime_ns})
            if len(records) > cls.artifact_limit:
                raise ArtifactError("Input inventory exceeds its bound")
        data = {"schema_version": 1, "run_id": run_id, "status": "partial",
                "message": "Reserved; unfinished work is partial", "settings": plain_settings,
                "inputs": records, "attempts": []}
        cls._encode(data)  # Reject nonfinite/unsupported settings before creating anything.
        root = destination / run_id
        root.mkdir()  # Exclusive. Never adopt a pre-existing run, even an empty one.
        _sync_directory(destination)
        store = object.__new__(cls)
        store._root = root
        store._run_id = run_id
        store._directories = {root: _token(_stat(root, directory=True))}
        store._attempts = {}
        store._lock = RLock()
        store._revision = 0
        store._data = data
        store._commit(data)
        return store

    @staticmethod
    def _encode(data):
        content = (json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True,
                              indent=2) + "\n").encode("utf-8")
        if len(content) > RunStore.manifest_limit_bytes:
            raise ArtifactError("Manifest metadata exceeds its bound")
        return content

    @property
    def root(self):
        return self._root

    @property
    def run_id(self):
        return self._run_id

    def _guard(self):
        _parents(self.root)
        for path, token in self._directories.items():
            if _token(_stat(path, directory=True)) != token:
                raise ArtifactError(f"Reserved directory identity changed: {path}")

    def _active(self):
        self._guard()
        if self._data["status"] != "partial":
            raise ArtifactError("A terminal run is immutable")

    def _copy(self):
        return to_plain(self.snapshot)

    @property
    def snapshot(self):
        with self._lock:
            return freeze(self._data)

    @property
    def manifest_path(self):
        return self.root / f"manifest-{self._revision:06d}.json"

    def _commit(self, data):
        self._guard()
        revision = self._revision + 1
        data = {**data, "revision": revision}
        content = self._encode(data)
        temporary = self.root / f".manifest-{uuid4().hex}.partial"
        target = self.root / f"manifest-{revision:06d}.json"
        linked = False
        with temporary.open("xb") as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        token = _token(_stat(temporary))
        try:
            self._guard()
            _link(temporary, target)
            linked = True
            temporary.unlink()
            _sync_directory(self.root)
        except BaseException:
            # Remove only our just-linked metadata revision on failed durability.
            # Preserve the temporary and every earlier partial/failure revision.
            if linked and _token(_stat(target)) == token:
                target.unlink()
            raise
        self._data = data
        self._revision = revision

    def reserve_attempt(self, stage_id, *, attempt_id=None):
        with self._lock:
            self._active()
            if len(self._data["attempts"]) >= self.artifact_limit:
                raise ArtifactError("Attempt inventory exceeds its bound")
            stage_id = _name(stage_id)
            attempt_id = _name(attempt_id or f"attempt-{uuid4().hex}")
            stage = self.root / stage_id
            if stage not in self._directories:
                stage.mkdir()  # No adoption of existing stage directories.
                self._directories[stage] = _token(_stat(stage, directory=True))
                _sync_directory(self.root)
            if any(record["stage_id"] == stage_id and record["status"] == "partial"
                   for record in self._data["attempts"]):
                raise ArtifactError("Finish the previous stage attempt before retrying")
            directory = stage / attempt_id
            directory.mkdir()
            self._directories[directory] = _token(_stat(directory, directory=True))
            _sync_directory(stage)
            attempt = Attempt(Identity(self.run_id, stage_id, attempt_id), directory)
            data = self._copy()
            data["attempts"].append({"stage_id": stage_id, "attempt_id": attempt_id,
                                     "directory": str(directory), "status": "partial",
                                     "message": "Reserved", "artifacts": []})
            self._commit(data)
            self._attempts[(stage_id, attempt_id)] = attempt
            return attempt

    def _record(self, attempt, data):
        if not isinstance(attempt, Attempt):
            raise ArtifactError("Expected a reserved attempt")
        key = (attempt.identity.stage_id, attempt.identity.attempt_id)
        if attempt.identity.run_id != self.run_id or self._attempts.get(key) != attempt:
            raise ArtifactError("Foreign or unrecorded attempt")
        return next(record for record in data["attempts"] if
                    (record["stage_id"], record["attempt_id"]) == key)

    def _path(self, attempt, path):
        self._guard()
        self._record(attempt, self._data)
        path = _absolute(path)
        if not path.is_relative_to(attempt.directory) or path == attempt.directory:
            raise ArtifactError("Output is outside its reserved attempt")
        _parents(path.parent)
        return path

    def _artifact_record(self, attempt, artifact, *, validated=False):
        if not isinstance(artifact, Artifact):
            raise ArtifactError("Expected a typed artifact")
        path = self._path(attempt, artifact.path)
        info = _stat(path)
        descriptor = artifact.input_descriptor
        if artifact.kind == "ldat" and descriptor is None:
            raise ArtifactError("LDAT requires an explicit format/population descriptor")
        if descriptor is not None and _absolute(descriptor.path) != path:
            raise ArtifactError("Artifact descriptor path does not match its file")
        if validated and descriptor is not None and not descriptor.validated:
            raise ArtifactError("Successful data artifact needs a validated descriptor")
        if artifact.disposable and (artifact.kind != "ldat" or path.suffix != ".ldat"
                                    or info.st_size != 0):
            raise ArtifactError("Only explicitly empty LDAT may be disposable")
        if validated:
            _sync_file(path, _file_token(info))
            _sync_directory(path.parent)
        return {**to_plain(artifact), "size_bytes": info.st_size,
                "file_token": list(_file_token(info)), "validated": validated,
                "cleanup": "retained"}

    def record_artifacts(self, attempt, artifacts):
        """Explicit exact list, including partial/failed output. No prefix expansion."""
        with self._lock:
            self._active()
            data = self._copy()
            record = self._record(attempt, data)
            if record["status"] != "partial":
                raise ArtifactError("Attempt is already terminal")
            additions = []
            for artifact in artifacts:
                if len(additions) >= self.artifact_limit:
                    raise ArtifactError("Artifact inventory exceeds its bound")
                additions.append(self._artifact_record(attempt, artifact))
            existing = {item["path"] for item in record["artifacts"]}
            if len(existing) + len(additions) != len(existing | {a["path"] for a in additions}):
                raise ArtifactError("Artifact already recorded or duplicated")
            record["artifacts"].extend(additions)
            record["artifacts"].sort(key=lambda item: _natural(Path(item["path"])))
            if sum(len(a["artifacts"]) for a in data["attempts"]) > self.artifact_limit:
                raise ArtifactError("Artifact inventory exceeds its bound")
            self._commit(data)

    def finish_attempt(self, attempt, result):
        with self._lock:
            self._active()
            if not isinstance(result, CommandResult) or result.identity != attempt.identity:
                raise ArtifactError("Result identity does not match its attempt")
            data = self._copy()
            record = self._record(attempt, data)
            if record["status"] != "partial":
                raise ArtifactError("Attempt is already terminal")
            success = result.status == ResultStatus.SUCCEEDED
            if success and (not result.can_advance or not result.artifacts):
                raise ArtifactError("Success requires validated exact output artifacts")
            additions = [self._artifact_record(attempt, artifact, validated=success)
                         for artifact in result.artifacts]
            if len({a["path"] for a in additions}) != len(additions):
                raise ArtifactError("Duplicate result artifact")
            by_path = {item["path"]: item for item in record["artifacts"]}
            for item in additions:
                previous = by_path.get(item["path"])
                if previous and (previous["file_token"] != item["file_token"] or
                                 previous["cleanup"] != "retained"):
                    raise ArtifactError("Recorded artifact changed or was removed")
                if previous and _artifact_contract(previous) != _artifact_contract(item):
                    raise ArtifactError("Recorded artifact format/population/ownership changed")
                by_path[item["path"]] = item
            record.update(status=result.status.value, message=result.message,
                          exit_code=result.exit_code,
                          artifacts=sorted(by_path.values(), key=lambda a: _natural(Path(a["path"]))),
                          outputs=[item["path"] for item in sorted(additions,
                                   key=lambda a: _natural(Path(a["path"])))])
            if sum(len(a["artifacts"]) for a in data["attempts"]) > self.artifact_limit:
                raise ArtifactError("Artifact inventory exceeds its bound")
            self._commit(data)

    def finish(self, status, message=""):
        with self._lock:
            self._active()
            status = ResultStatus(status)
            data = self._copy()
            if status == ResultStatus.SUCCEEDED:
                latest = {}
                for record in data["attempts"]:
                    latest[record["stage_id"]] = record
                if not latest or any(record["status"] != "succeeded" for record in latest.values()):
                    raise ArtifactError("Every stage needs a successful final attempt")
                for record in latest.values():
                    for item in record["artifacts"]:
                        if item["path"] not in record["outputs"]:
                            continue
                        self._path(self._attempts[(record["stage_id"], record["attempt_id"])],
                                   item["path"])
                        if (not item["validated"] or item["cleanup"] != "retained" or
                                list(_file_token(_stat(Path(item["path"])))) != item["file_token"]):
                            raise ArtifactError("Successful artifact is missing, changed or unvalidated")
            elif not message.strip():
                raise ArtifactError("Failure/cancellation requires a reason")
            if any(record["status"] == "partial" for record in data["attempts"]):
                raise ArtifactError("Finish active attempts before finalizing the run")
            data.update(status=status.value, message=message)
            self._commit(data)

    def publish(self, attempt, temporary, destination):
        """Publish one completed file exclusively; preserve temporary on failure.

        Callers write/close a uniquely reserved ``*.partial`` file, then validate.
        Publication alone never marks an attempt/run successful or disposable.
        """
        with self._lock:
            self._active()
            if self._record(attempt, self._data)["status"] != "partial":
                raise ArtifactError("Attempt is already terminal")
            source = self._path(attempt, temporary)
            target = self._path(attempt, destination)
            if source == target or source.suffix != ".partial":
                raise ArtifactError("Publication needs a separate owned .partial source")
            info = _stat(source)
            if info.st_nlink != 1:
                raise ArtifactError("Partial output has an external hard link")
            _sync_file(source, _file_token(info))
            self._guard()
            _link(source, target)
            _sync_directory(target.parent)
            # Keep .partial too: even a later manifest/storage failure preserves it.
            return target

    def discover_ldat(self, attempt, prefix, format, population, *, before):
        """Exact converter basename, optional numeric split, in this attempt only.

        Requires its pre-command directory inventory. A pre-existing matching
        file is a collision, never a successful output. No legacy input discovery.
        """
        with self._lock:
            self._active()
            prefix = self._path(attempt, prefix)
            if prefix.parent != attempt.directory:
                raise ArtifactError("Converter prefix must be in the attempt directory")
            pattern = re.compile(re.escape(prefix.name) + r"(?:_(\d+))?\.ldat\Z")
            old = {_absolute(path) for path in before}
            paths = []
            with os.scandir(attempt.directory) as entries:
                for entry in entries:
                    if not pattern.fullmatch(entry.name):
                        continue
                    path = Path(entry.path)
                    self._path(attempt, path)
                    _stat(path)
                    if path in old:
                        raise ArtifactError(f"Pre-existing converter output: {path}")
                    paths.append(path)
                    if len(paths) > self.artifact_limit:
                        raise ArtifactError("Converter inventory exceeds its bound")
            return tuple(Artifact(path, "ldat", InputDescriptor(path, format, population))
                         for path in sorted(paths, key=_natural))

    def inventory(self, attempt):
        with self._lock:
            self._active()
            self._record(attempt, self._data)
            paths = []
            with os.scandir(attempt.directory) as entries:
                for entry in entries:
                    paths.append(Path(entry.path))
                    if len(paths) > self.artifact_limit:
                        raise ArtifactError("Attempt inventory exceeds its bound")
            return tuple(sorted(paths, key=_natural))

    def remove_empty_ldat(self, attempt, path):
        """Only a recorded, unchanged, disposable, zero-byte LDAT is eligible.

        No other data/index/RAW file cleanup is implemented. Linux uses a no-follow
        directory handle for the final ownership/size check and unlink operation.
        """
        with self._lock:
            self._active()
            path = self._path(attempt, path)
            data = self._copy()
            record = self._record(attempt, data)
            item = next((a for a in record["artifacts"] if a["path"] == str(path)), None)
            info = _stat(path)
            if (item is None or not item["disposable"] or item["kind"] != "ldat"
                    or path.suffix != ".ldat" or info.st_size != 0 or info.st_nlink != 1
                    or list(_file_token(info)) != item["file_token"]
                    or item["cleanup"] != "retained"):
                raise ArtifactError("Cleanup refused: not an unchanged recorded empty disposable LDAT")
            item["cleanup"] = "pending"
            self._commit(data)  # Persist intent before removal; a crash never invents success.
            self._path(attempt, path)
            if os.name != "nt":
                fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    if _token(os.fstat(fd)) != _token(_stat(path.parent, directory=True)):
                        raise ArtifactError("Cleanup directory identity changed")
                    current = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                    if _file_token(current) != _file_token(info) or current.st_nlink != 1:
                        raise ArtifactError("Cleanup file changed")
                    os.unlink(path.name, dir_fd=fd)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            else:
                current = _stat(path)
                if _file_token(current) != _file_token(info) or current.st_nlink != 1:
                    raise ArtifactError("Cleanup file changed")
                path.unlink()
            data = self._copy()
            record = self._record(attempt, data)
            next(a for a in record["artifacts"] if a["path"] == str(path))["cleanup"] = "removed"
            self._commit(data)


def read_manifest(root):
    """Read the latest bounded complete revision. Never reopen it for writing."""
    root = _absolute(root)
    _parents(root)
    path = None
    revision = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if re.fullmatch(r"manifest-\d{6,}\.json", entry.name):
                candidate = int(Path(entry.name).stem.split("-")[-1])
                if candidate > revision:
                    path, revision = Path(entry.path), candidate
    if path is None:
        raise ArtifactError("No complete manifest revision; run is unavailable/partial")
    info = _stat(path)
    if info.st_size > RunStore.manifest_limit_bytes:
        raise ArtifactError("Manifest exceeds its metadata bound")
    with path.open("rb") as stream:
        if _file_token(os.fstat(stream.fileno())) != _file_token(info):
            raise ArtifactError("Manifest identity changed")
        content = stream.read(RunStore.manifest_limit_bytes + 1)
    if len(content) > RunStore.manifest_limit_bytes:
        raise ArtifactError("Manifest exceeds its metadata bound")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError("Duplicate manifest key")
            result[key] = value
        return result
    data = json.loads(content, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ArtifactError("Nonfinite manifest")))
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("run_id") != root.name or data.get("revision") != int(path.stem.split("-")[-1])
            or data.get("status") not in {"partial", *(status.value for status in ResultStatus)}):
        raise ArtifactError("Invalid run manifest")
    return freeze(data)
