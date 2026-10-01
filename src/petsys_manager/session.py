"""Toolkit-free manager session: profile file, prerequisite checks and UI event queue.

Workers only put `ShellEvent`s on `events`; the GUI drains that queue on its Tk
thread. Nothing here launches processes, touches DAQ resources, creates
destinations or writes processing configuration/maps. Prerequisite checks are
read-only `settings.preflight` calls, run off the GUI thread and identified by a
generation so a slower, older answer cannot replace a newer one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading

from .contracts import Action
from .settings import (MachineProfile, PrerequisiteIssue, SystemProbe, default_profile_path,
                       load_profile, preflight, save_profile)


CHECKOUT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ShellEvent:
    kind: str  # "log" or "readiness"; later tasks add workflow/DAQD kinds
    payload: object


@dataclass(frozen=True)
class Readiness:
    generation: int
    issues: dict  # check key -> tuple[PrerequisiteIssue, ...]; empty means prerequisites met


class ManagerSession:
    def __init__(self, profile_path=None, *, probe=None, repo_root=None):
        self.repo_root = Path(repo_root or CHECKOUT).resolve()
        self.explicit_profile = profile_path is not None
        self.profile_path = Path(profile_path) if profile_path is not None else default_profile_path()
        self.probe = SystemProbe() if probe is None else probe
        self.profile = MachineProfile()
        self.events = queue.SimpleQueue()
        self._wake = threading.Condition()
        self._pending = None
        self._generation = 0
        self._worker = None
        self._closed = False

    @property
    def generation(self):
        return self._generation

    def log(self, message):
        self.events.put(ShellEvent("log", str(message)))

    def open(self):
        """Load the startup profile, or keep defaults; never creates or rewrites a file."""
        if not self.profile_path.exists():
            self.log(f"No profile at {self.profile_path}; defaults in use, nothing written")
            return "defaults"
        try:
            self.load()
        except (OSError, ValueError) as exc:
            self.log(f"Profile not loaded, defaults in use and file left unchanged: {exc}")
            return "error"
        self.log(f"Loaded profile {self.profile_path}")
        return "loaded"

    def load(self, path=None):
        """Read a profile; on failure the current profile and path stay unchanged."""
        target = self.profile_path if path is None else Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Profile not found: {target}")
        profile = load_profile(target)
        self.profile_path, self.profile = target, profile
        return profile

    def save(self, profile, path=None):
        """Write only a manager profile; an existing file must already be a valid profile."""
        target = self.profile_path if path is None else Path(path)
        written = save_profile(profile, target, overwrite=target.exists())
        self.profile_path, self.profile = written, profile
        return written

    def check(self, profile, requests):
        """Queue read-only prerequisite checks: requests maps key -> (Action, RunOptions)."""
        requests = {key: (Action(action), options) for key, (action, options) in requests.items()}
        with self._wake:
            if self._closed:
                raise RuntimeError("Session is closed")
            self._generation += 1
            self._pending = (self._generation, profile, requests)
            if self._worker is None:
                self._worker = threading.Thread(target=self._check_loop, name="petsys-preflight", daemon=True)
                self._worker.start()
            self._wake.notify()
            return self._generation

    def _check_loop(self):
        while True:
            with self._wake:
                while self._pending is None and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                generation, profile, requests = self._pending
                self._pending = None
            issues = {}
            for key, (action, options) in requests.items():
                try:
                    issues[key] = preflight(profile, action, options, repo_root=self.repo_root,
                                            probe=self.probe).issues
                except Exception as exc:  # A probe/profile fault is a reason, never readiness.
                    issues[key] = (PrerequisiteIssue("settings", f"{type(exc).__name__}: {exc}"),)
            self.events.put(ShellEvent("readiness", Readiness(generation, issues)))

    def close(self, timeout=2.0):
        with self._wake:
            self._closed = True
            self._pending = None
            self._wake.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return worker is None or not worker.is_alive()
