"""Recent runs from each destination's ``runs.tsv`` (spec 005 FR-3), and what a run record offers as the next
step's inputs (FR-4, FR-6). Read only; no Tk import.

Only runs with a ``runs.tsv`` line are listed; run folders without one are not searched for. Offers, conversion
outputs and the main report come from the run record (``artifacts.read_manifest``), never from file names or a
folder listing, and every offered file must still have its recorded size.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re

from .artifacts import ArtifactError, read_manifest
from .contracts import InputDescriptor
from .settings import _resolve
from .workflow import RUNS, RUNS_HEADER

TAIL_BYTES = 2**20          # at most the last 1 MiB of each runs.tsv is read
RECENT_LIMIT = 500
FINISHED_FORMAT = "%Y-%m-%d %H:%M:%S"
RUN_FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")   # a portable name: never "..", a path or empty
DESTINATIONS = ("data_dir", "calibration_dir", "lm_dir", "report_dir")


@dataclass(frozen=True)
class RecentRun:
    finished: datetime
    destination: str        # the profile field of the destination it was listed in
    run_root: Path
    action: str
    inputs: str
    status: str
    main_output: str        # relative to run_root, or "-"


@dataclass(frozen=True)
class Overview:
    rows: tuple             # RecentRun, newest first
    skipped: int            # malformed lines
    recorded: bool          # False: no runs.tsv ("no runs recorded")


@dataclass(frozen=True)
class Recent:
    rows: tuple             # RecentRun over all destinations, newest first, at most RECENT_LIMIT
    skipped: int
    unrecorded: tuple       # destination keys without a runs.tsv


def read_overview(destination, key=""):
    """The runs of one ``runs.tsv``, newest first. The header, a partial last line (still being appended) and
    the line cut by the 1 MiB tail are dropped silently; other lines without 6 columns, a finish time or a
    portable run folder are skipped and counted."""
    destination = Path(destination)
    try:
        with open(destination / RUNS, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            start = max(0, size - TAIL_BYTES)
            handle.seek(max(0, start - 1))
            data = handle.read(size - max(0, start - 1))
    except FileNotFoundError:
        return Overview((), 0, False)
    if start > 0:
        boundary = data[:1] == b"\n"
        data = data[1:] if boundary else data[data.find(b"\n") + 1:] if b"\n" in data else b""
    lines = data.split(b"\n")
    lines.pop()                 # empty after a final newline, or a partial line
    header = "\t".join(RUNS_HEADER)
    rows, skipped = [], 0
    for raw in lines:
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        if text == header:
            continue
        row = _row(text, destination, key)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    return Overview(tuple(reversed(rows)), skipped, True)


def _row(text, destination, key):
    cells = text.split("\t")
    if len(cells) != len(RUNS_HEADER) or not RUN_FOLDER.fullmatch(cells[1]):
        return None
    try:
        finished = datetime.strptime(cells[0], FINISHED_FORMAT)
    except ValueError:
        return None
    return RecentRun(finished, key, destination / cells[1], *cells[2:])


def profile_destinations(profile, repo_root):
    """``(key, folder)`` of the profile's destinations; relative paths resolve against the processing root,
    as preflight does."""
    checkout = Path(repo_root).resolve()
    root = _resolve(profile.processing_root, checkout) or checkout
    return [(key, _resolve(getattr(profile, key), root)) for key in DESTINATIONS if getattr(profile, key)]


def list_recent(destinations):
    """Runs of every ``(key, destination)``, newest first and capped at 500; a destination listed twice is
    read once, under its first key. Equal finish times keep the later line first."""
    seen, overviews, unrecorded = set(), [], []
    for key, destination in destinations:
        identity = os.path.normcase(os.path.abspath(destination))
        if identity in seen:
            continue
        seen.add(identity)
        overview = read_overview(destination, key)
        overviews.append(overview)
        if not overview.recorded:
            unrecorded.append(key)
    ordered = sorted(((row, position) for overview in overviews for position, row in enumerate(overview.rows)),
                     key=lambda item: (item[0].finished, -item[1]), reverse=True)
    rows = tuple(row for row, _ in ordered[:RECENT_LIMIT])
    return Recent(rows, sum(overview.skipped for overview in overviews), tuple(unrecorded))


class RunRecordError(ValueError):
    """A folder that is not a readable run, or a recorded file that is gone or changed."""


@dataclass(frozen=True)
class Offer:
    label: str
    target: str             # calibrate, listmode, qc_analyze (LDAT descriptors) or lm_calibration (an .encal)
    payload: object


LDAT_OFFERS = (("Calibrate", "calibrate"), ("Generate LM", "listmode"), ("Run QC", "qc_analyze"))
CALIBRATION_OFFER = ("Generate LM with this calibration", "lm_calibration")
OFFER_LABELS = {target: label for label, target in (*LDAT_OFFERS, CALIBRATION_OFFER)}
REPORT_KINDS = {"qc": "qc_report", "calibration": "calibration_plot", "listmode": "listmode_debug_plot"}


def run_offers(root):
    """Next-step offers of a run's succeeded stages: a conversion's LDATs for calibration, LM and offline QC,
    and a calibration's .encal for LM."""
    stages = _stages(root)
    offers = []
    ldats = _ldats(stages)
    if ldats:
        offers += [Offer(label, target, ldats) for label, target in LDAT_OFFERS]
    encal = next(iter(_outputs(stages.get("calibration"), "encal")), None)
    if encal is not None:
        offers.append(Offer(*CALIBRATION_OFFER, _checked(encal)))
    return tuple(offers)


def conversion_outputs(root):
    """Exactly the LDATs a conversion run recorded as its outputs, in their recorded order."""
    ldats = _ldats(_stages(root))
    if not ldats:
        raise RunRecordError(f"{Path(root).name} has no conversion outputs (no succeeded conversion stage)")
    return ldats


def main_report(root):
    """The report or plot of the run's last stage when it succeeded: QC report, calibration plot or the first
    LM debug plot; None otherwise."""
    stages = _stages(root)
    if not stages:
        return None
    stage_id, record = list(stages.items())[-1]
    kind = REPORT_KINDS.get(stage_id)
    for item in _outputs(record, kind) if kind else ():
        path = Path(item["path"])
        if path.is_file():
            return path
    return None


def _stages(root):
    """Each stage's latest attempt record, in run order."""
    try:
        manifest = read_manifest(Path(root).absolute())
    except (ArtifactError, OSError, ValueError) as exc:
        raise RunRecordError(f"{root} is not a run folder (no readable run record: {exc})") from None
    latest = {}
    for record in manifest.get("attempts", ()):
        latest[record["stage_id"]] = record
    return latest


def _outputs(record, kind):
    if record is None or record.get("status") != "succeeded":
        return []
    outputs = set(record.get("outputs") or ())
    return [item for item in record.get("artifacts", ())
            if item["path"] in outputs and item["kind"] == kind and not item.get("disposable")]


def _ldats(stages):
    return tuple(InputDescriptor(_checked(item), item["input_descriptor"]["format"],
                                 item["input_descriptor"]["population"], bool(item["input_descriptor"].get("validated")))
                 for item in _outputs(stages.get("conversion"), "ldat"))


def _checked(item):
    path = Path(item["path"])
    try:
        size = path.stat().st_size
    except OSError:
        raise RunRecordError(f"Recorded output is missing: {path}") from None
    if size != item["size_bytes"]:
        raise RunRecordError(f"Recorded output changed size ({size:,} bytes, recorded {item['size_bytes']:,}): {path}")
    return path
