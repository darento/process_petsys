"""IMAS time-walk correction (spec 012): walk inputs and the IMAS walk table.

Per side, ``t_corrected = t - b[TIME channel] - walk2d[charge, share] - doi[doi_ratio] - kev[kev]`` (ps,
subtractive; table and fits in ``src.timing.walk``). The four walk inputs (FR-1) come from spec 011's
selection ingredients (``select._walk_ingredients``), so features, observations, the CTR and the
correction function derive them through ``walk_inputs`` alone:

* charge: raw QDC charge (a.u.) of the timestamped TIME hit (the minimodule's highest-charge TIME hit);
* share: that charge over the minimodule's summed TIME charge;
* doi_ratio: ENERGY sum over ENERGY maximum of the minimodule, a light-sharing ratio, not a depth
  (IMAS has no DOI-limits file);
* kev: the minimodule's ENERGY sum through the run's job calibration (``511 / mu x e_raw``).

Missing inputs are status codes, never zero corrections.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from src.timing import walk as core
from src.timing.cache import ROLES

INPUTS = ("charge", "share", "doi_ratio", "kev")
# status per side: 0 ok, else the first missing input
STATUS = ("ok", "no TIME charge", "no ENERGY charge", "no calibration")
_STATUS = {label: code for code, label in enumerate(STATUS)}

LUT_SCHEMA = "imas-walk-lut/1"
CONVENTION = "per side t_corrected = t - b[TIME channel] - walk2d[charge, share] - doi[doi_ratio] - kev[kev] (ps)"
BLOCKS = {"walk2d": ("charge", "share"), "doi": ("doi_ratio",), "kev": ("kev",)}
DEFAULT_BINS = {"charge": 20, "share": 10, "doi_ratio": 20, "kev": 20}     # spec 009's table (clarify 4)
MODEL = core.WalkModel(LUT_SCHEMA, CONVENTION, BLOCKS, DEFAULT_BINS)


class WalkLut(core.WalkLut):
    """IMAS walk table: ``src.timing.walk.WalkLut`` with ``BLOCKS``."""
    model = MODEL


def walk_inputs(q_top, q_time, doi_ratio, kev):
    """Walk inputs of one side for arrays of events: ({name: float64 array, NaN where missing}, status int8)."""
    q_top, q_time = np.asarray(q_top, np.float64), np.asarray(q_time, np.float64)
    doi_ratio, kev = np.asarray(doi_ratio, np.float64), np.asarray(kev, np.float64)
    status = np.zeros(len(q_top), np.int8)
    with np.errstate(invalid="ignore"):
        for label, bad in (("no calibration", ~np.isfinite(kev)),
                           ("no ENERGY charge", ~(doi_ratio > 0) | ~np.isfinite(doi_ratio)),
                           ("no TIME charge", ~(q_top >= 0) | ~(q_time > 0) | ~np.isfinite(q_top)
                            | ~np.isfinite(q_time))):
            status[bad] = _STATUS[label]       # applied in reverse order: the first missing input wins
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(q_time > 0, q_top / q_time, np.nan)
    values = {"charge": q_top, "share": share, "doi_ratio": doi_ratio, "kev": kev}
    return {name: np.where(status == 0, value, np.nan) for name, value in values.items()}, status


def side_inputs(column, suffix):
    """``walk_inputs`` of one side read through ``column(name)``: features (suffix ``1``/``2``) or walk-carrying
    observations (``i``/``j``)."""
    return walk_inputs(column(f"q_top_{suffix}"), column(f"q_time_{suffix}"), column(f"doi_ratio_{suffix}"),
                       column(f"energy_kev_{suffix}"))


def count_status(status, mask=None):
    """{label: events} of the non-ok statuses (within ``mask``)."""
    status = status if mask is None else status[mask]
    codes, found = np.unique(status[status != 0], return_counts=True)
    return {STATUS[c]: int(n) for c, n in zip(codes.tolist(), found.tolist())}


# --- Walk-corrected observations (FR-4) ---------------------------------------------------
def correct_rows(rows, lut):
    """Walk-carrying observation rows with every walk input on both sides, ``dt_ps`` corrected by
    round(W(i) - W(j)); and per role the counts (missing inputs, clamped per input)."""
    if "q_top_i" not in (rows.dtype.names or ()):
        raise core.WalkError("observation rows without walk inputs: stream the set with carry_walk")
    values_i, status_i = side_inputs(rows.__getitem__, "i")
    values_j, status_j = side_inputs(rows.__getitem__, "j")
    keep = (status_i == 0) & (status_j == 0)
    difference, out_i, out_j = core.walk_difference(lut, values_i, values_j)
    counts = {}
    for code, role in enumerate(ROLES):
        mine = rows["role"] == code
        if not mine.any():
            continue
        entry = {"rows": int(mine.sum()), "walk input missing": int(np.count_nonzero(mine & ~keep))}
        for name in INPUTS:
            entry[f"clamped {name}"] = int(np.count_nonzero(mine & keep & out_i[name])
                                           + np.count_nonzero(mine & keep & out_j[name]))
        counts[role] = entry
    rows = rows[keep].copy()
    rows["dt_ps"] -= np.rint(difference[keep]).astype(np.int64)
    return rows, counts


class WalkCorrected:
    """An observation set (or pooled sets) read with ``correct_rows``: ``partition(p)`` like the set's own, so the
    EM, the pair estimator and the held-out checks see walk-corrected ``dt``. ``counts()`` sums the partitions
    read so far (each counted once)."""

    def __init__(self, observations, lut):
        self.observations, self.lut = observations, lut
        self.partitions = observations.partitions
        self._counts = {}

    @property
    def carries_candidates(self):
        return getattr(self.observations, "carries_candidates", False)

    def partition(self, p):
        rows, self._counts[p] = correct_rows(self.observations.partition(p), self.lut)
        return rows

    def counts(self):
        total = {}
        for counts in self._counts.values():
            for role, entry in counts.items():
                mine = total.setdefault(role, {})
                for key, n in entry.items():
                    mine[key] = mine.get(key, 0) + n
        return total


# --- Bar walk fit (FR-3) ------------------------------------------------------------------
FIT_SCHEMA = "imas-walk-fit/1"
REPORT_SCHEMA = "imas-walk-fit-report/1"
_FIT_REQUIRED = {"schema", "job", "work", "skew_lut", "skew_lut_sha256", "output"}
_FIT_OPTIONAL = {"walk_lut": None, "bins": dict(DEFAULT_BINS), "min_events": core.MIN_EVENTS, "core_ps": 600.0,
                 "iterations": 4, "branch_iterations": 3, "max_events": 20_000_000}
EVENT_STAGES = ("rows read", "no applied delay", "walk input missing", "kept")


def report_path(output):
    output = Path(output)
    return output.with_name(output.stem + "_report.json")


def load_fit(path):
    """Walk fit file: the job file and its work directory (walk-carrying observation sets), the frozen skew LUT
    (solved by that job), the output and fit options; ``walk_lut`` names the walk LUT the skew LUT was solved
    with, if it was (pairing record only). Paths relative to the file."""
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec, dict) or spec.get("schema") != FIT_SCHEMA:
        raise core.WalkError(f"file schema must be {FIT_SCHEMA!r}")
    if not _FIT_REQUIRED <= set(spec) <= _FIT_REQUIRED | set(_FIT_OPTIONAL):
        raise core.WalkError(f"file keys: required {sorted(_FIT_REQUIRED)}, optional {sorted(_FIT_OPTIONAL)}")
    base = path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    options = _FIT_OPTIONAL | {k: spec[k] for k in _FIT_OPTIONAL if k in spec}
    bins = options["bins"]
    if not isinstance(bins, dict) or set(bins) != set(DEFAULT_BINS) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in bins.values()):
        raise core.WalkError(f"bins: integers >= 2 for {sorted(DEFAULT_BINS)}")
    for key in ("min_events", "iterations", "branch_iterations", "max_events"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise core.WalkError(f"{key} must be a positive integer")
    if not float(options["core_ps"]) > 0:
        raise core.WalkError("core_ps must be positive")
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "job": local(spec["job"]),
            "work": local(spec["work"]), "skew_lut": local(spec["skew_lut"]),
            "skew_lut_sha256": str(spec["skew_lut_sha256"]).lower(), "output": local(spec["output"]),
            **options, "walk_lut": None if options["walk_lut"] is None else local(options["walk_lut"])}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_rows(observations):
    counts = observations.pair_counts()
    return int(counts["count"][counts["role"] == 0].sum())


def bar_events(sets, delays, *, max_events):
    """Skew-corrected events of walk-carrying observation sets: every ``stride``-th train row (at most
    ``max_events``) and every held-out (``test``) row, with both TIME channels delayed and every walk input.
    Returns {"train"|"heldout": {dt, near, far, <input>_i, <input>_j} float32}, counts and the stride."""
    total = sum(_train_rows(s) for s in sets)
    stride = max(1, -(-total // max_events))
    parts = {"train": [], "heldout": []}
    counts = {role: dict.fromkeys(EVENT_STAGES, 0) for role in parts}
    for observations in sets:
        if not observations.carries_walk:
            raise core.WalkError(f"{observations.directory}: observation set without walk inputs; stream it with "
                                 "carry_walk")
        for p in range(observations.partitions):
            rows = observations.partition(p)
            for role, code in (("train", 0), ("heldout", 2)):
                chosen = rows[rows["role"] == code]
                if role == "train":
                    chosen = chosen[::stride]
                b_i, f_i = delays.lookup(chosen["channel_i"])
                b_j, f_j = delays.lookup(chosen["channel_j"])
                values_i, status_i = side_inputs(chosen.__getitem__, "i")
                values_j, status_j = side_inputs(chosen.__getitem__, "j")
                applied, ok = f_i & f_j, (status_i == 0) & (status_j == 0)
                keep = applied & ok
                stage = counts[role]
                stage["rows read"] += len(chosen)
                stage["no applied delay"] += int(np.count_nonzero(~applied))
                stage["walk input missing"] += int(np.count_nonzero(applied & ~ok))
                stage["kept"] += int(np.count_nonzero(keep))
                tof = chosen["shell_tof_ps"][keep]
                part = {"dt": (chosen["dt_ps"][keep] - b_i[keep] + b_j[keep]).astype(np.float32),
                        "near": ((tof[:, 0] + tof[:, 1]) / 2).astype(np.float32),
                        "far": ((tof[:, 2] + tof[:, 3]) / 2).astype(np.float32)}
                for name in INPUTS:
                    part[f"{name}_i"] = values_i[name][keep].astype(np.float32)
                    part[f"{name}_j"] = values_j[name][keep].astype(np.float32)
                parts[role].append(part)
    keys = ["dt", "near", "far"] + [f"{n}_{s}" for s in ("i", "j") for n in INPUTS]
    events = {role: {k: np.concatenate([c[k] for c in chunks]) if chunks else np.empty(0, np.float32) for k in keys}
              for role, chunks in parts.items()}
    return events, counts, {"train_rows": total, "stride": stride}


def sides(events):
    """float64 side dicts (i, j) of ``bar_events`` arrays."""
    return ({n: events[f"{n}_i"].astype(np.float64) for n in INPUTS},
            {n: events[f"{n}_j"].astype(np.float64) for n in INPUTS})


def branch_residual(events, model):
    """``dt - model`` minus the nearer shell branch (ps)."""
    dt = events["dt"].astype(np.float64) - model
    near, far = events["near"].astype(np.float64), events["far"].astype(np.float64)
    return dt - np.where(np.abs(dt - near) <= np.abs(dt - far), near, far)


def width(residual, core_ps):
    """Core RMS (|r| < ``core_ps``), robust sigma (1.4826 x MAD) and events."""
    residual = residual[np.isfinite(residual)]
    if not len(residual):
        return {"events": 0}
    core_rows = residual[np.abs(residual) < core_ps]
    return {"events": int(len(residual)), "core_events": int(len(core_rows)),
            "core_rms_ps": float(np.sqrt(np.mean(core_rows ** 2))) if len(core_rows) else None,
            "robust_sigma_ps": float(1.4826 * np.median(np.abs(residual - np.median(residual))))}


def fit_bar(events, options, manifest):
    """FR-3: the walk LUT from the train events (nearer branch re-chosen after each walk update) and the
    before/after residual widths on train and held-out events, with held-out residual profiles."""
    from src.imas.skew import walkplots
    train = events["train"]
    side1, side2 = sides(train)
    lut, model, history, flips = core.fit_two_branch(
        side1, side2, train["dt"].astype(np.float64), train["near"].astype(np.float64),
        train["far"].astype(np.float64), np.ones(len(train["dt"]), bool), options["bins"],
        rounds=options["branch_iterations"], core_ps=float(options["core_ps"]), iterations=options["iterations"],
        minimum=options["min_events"], manifest=manifest, table=WalkLut)
    lut.manifest["branch_iterations"] = history
    lut.manifest["branch_flips"] = flips
    core_ps = float(options["core_ps"])
    widths = {"train": {"before": width(branch_residual(train, 0.0), core_ps),
                        "after": width(branch_residual(train, model), core_ps)}}
    held, profiles = events["heldout"], {}
    if len(held["dt"]):
        h1, h2 = sides(held)
        held_model, _, _ = core.walk_difference(lut, h1, h2)
        before, after = branch_residual(held, 0.0), branch_residual(held, held_model)
        widths["heldout"] = {"before": width(before, core_ps), "after": width(after, core_ps)}
        edges = {"charge": lut.edges["walk2d"][0], "share": lut.edges["walk2d"][1], "doi_ratio": lut.edges["doi"][0],
                 "kev": lut.edges["kev"][0]}
        profiles["bar held-out (side i)"] = walkplots.residual_profiles(h1, before, after, edges, core_ps=core_ps)
    return lut, widths, profiles


def run_fit(spec, *, log=print):
    """FR-3: fit and write the walk LUT, its manifest and ``<output>_report.json`` from the job's observation sets."""
    from src.imas.skew import job as ij, outputs
    from src.imas.skew.setup import ImasSetup
    from src.timing.cache import ObservationSet
    from src.timing.lut import DelayTable, LutError
    started = time.perf_counter()
    output = Path(spec["output"])
    for target in (output, core.manifest_path(output), report_path(output)):
        if target.exists():
            raise core.WalkError(f"output exists: {target}; choose a new name")
    job = ij.load_job(spec["job"])
    stamp = ij.fingerprint(job)
    marker = Path(spec["work"]) / "job_fingerprint.json"
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8"))["sha256"] != stamp["sha256"]:
        raise core.WalkError(f"{spec['work']}: not the work directory of {spec['job']}")
    setup = ImasSetup.from_files(job.config)
    if _sha256(spec["skew_lut"]) != spec["skew_lut_sha256"]:
        raise core.WalkError(f"skew LUT SHA-256 of {spec['skew_lut']} is not the frozen {spec['skew_lut_sha256']}")
    try:
        skew = outputs.read_imas_lut(spec["skew_lut"], setup)
    except LutError as error:
        raise core.WalkError(f"skew LUT: {error}") from None
    provenance = skew.manifest.get("provenance") or {}
    if provenance.get("job_sha256") != stamp["sha256"]:
        raise core.WalkError("the skew LUT was not solved by this job: its observation sets are not the LUT's events")
    if skew.input_correction != {"state": "uncorrected"}:
        raise core.WalkError("bar walk fits need an uncorrected-data skew LUT")
    encal = {}
    for run in job.runs:
        path = Path(spec["work"]) / run.name / "encal" / "encal.tsv"
        encal[run.name] = {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
        if encal[run.name]["sha256"] != (provenance.get("runs", {}).get(run.name) or {}).get("encal_sha256"):
            raise core.WalkError(f"{run.name}: energy calibration differs from the skew LUT's (other events)")
    skew_walk = provenance.get("walk_lut_sha256")
    if skew_walk or spec["walk_lut"] is not None:
        if spec["walk_lut"] is None or not skew_walk:
            raise core.WalkError("walk_lut names the walk LUT the skew LUT was solved with: give it exactly when the "
                                 "skew LUT was solved with walk")
        if WalkLut.read(spec["walk_lut"]).sha256 != skew_walk:
            raise core.WalkError(f"walk_lut is not the walk LUT {skew_walk[:12]}... the skew LUT was solved with")
    sets = [ObservationSet(Path(spec["work"]) / run.name / "observations") for run in job.runs]
    events, counts, sampling = bar_events(sets, DelayTable.from_lut(skew), max_events=spec["max_events"])
    log(f"{counts['train']['kept']:,} train events (stride {sampling['stride']}), "
        f"{counts['heldout']['kept']:,} held-out events")
    options = {k: spec[k] for k in _FIT_OPTIONAL if k != "walk_lut"}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fit_kind": "bar",
                "requirement": "012-FR-3", "fit_file": str(spec["path"]), "fit_file_sha256": spec["sha256"],
                "job": str(spec["job"]), "job_sha256": stamp["sha256"], "work": str(spec["work"]),
                "skew_lut": str(spec["skew_lut"]), "skew_lut_sha256": spec["skew_lut_sha256"],
                "skew_lut_table_sha256": skew.manifest["table"]["sha256"],
                "skew_solved_with_walk_lut_sha256": skew_walk, "config": str(job.config),
                "config_sha256": setup.provenance["config_sha256"], "map_sha256": setup.provenance["map_sha256"],
                "energy_calibration": encal, "cuts": sets[0].manifest["fingerprint"]["features"].get("selection"),
                "split": "train rows of the job's runs (every stride-th); held-out rows never fitted",
                "events": counts, "sampling": sampling, "options": options}
    lut, widths, profiles = fit_bar(events, options, manifest)
    lut.write(output)
    report = {"schema": REPORT_SCHEMA, "walk_lut": str(output), "walk_lut_sha256": lut.sha256, "manifest": lut.manifest,
              "widths": widths, "profiles": profiles, "seconds": round(time.perf_counter() - started, 1)}
    report_path(output).write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    held = widths.get("heldout")
    if held and None not in (held["before"].get("core_rms_ps"), held["after"].get("core_rms_ps")):
        log(f"held-out core RMS {held['before']['core_rms_ps']:.1f} -> {held['after']['core_rms_ps']:.1f} ps")
    log(f"walk LUT {output} ({lut.sha256[:12]})")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew walk",
                                     description=f"Fit an IMAS walk LUT on the bar runs of a skew job ({FIT_SCHEMA}).")
    parser.add_argument("fit_file", help="walk fit file")
    args = parser.parse_args(argv)
    try:
        run_fit(load_fit(args.fit_file), log=lambda message: print(message, flush=True))
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


# --- Correction function and export (FR-6) --------------------------------------------------
POLICIES = ("keep", "drop")
CORRECTION_STATUS = ("corrected", "no applied delay", "walk input missing")
EXPORT_FILES = ("imas_walk_lut.json", "imas_walk_lut.manifest.json", "imas_walk_curves.tsv", "export.json")


def correct_dt(dt_ps, channel_1, channel_2, side_1, side_2, skew, walk, *, policy="keep", rule="top",
               candidates_1=None, candidates_2=None, mismatch=False):
    """``t_1 - t_2`` corrected per side by the skew delays (``DelayTable``) and the walk LUT (None: skew only).

    ``side_s``: raw ingredients {q_top, q_time, doi_ratio, kev}. Returns the corrected ``dt`` (float ps), a status
    per event (``CORRECTION_STATUS``) and counts. An event without both delays or (with walk) without every walk
    input is not corrected: ``keep`` returns its uncorrected ``dt``, ``drop`` NaN (spec 009 FR-6 policy).

    Spec 014: each side is timestamped by ``rule`` (``timestamp.RULES``) from its ``candidates_s``
    (``timestamp.CANDIDATE_DTYPE`` slots); the corrected events get ``offset_1 - offset_2`` (``timestamp.
    side_offset``) and the counts ``rule acts`` (events) and the candidate counts per side summed. Both tables
    must have been fitted under ``rule`` unless ``mismatch`` names the difference (``timestamp.check_rule``)."""
    from src.imas.skew import timestamp
    if policy not in POLICIES:
        raise core.WalkError(f"policy must be one of {POLICIES}")
    rules = [timestamp.check_rule(timestamp.skew_rule(skew), rule, mismatch=mismatch)]
    if walk is not None:
        rules.append(timestamp.check_rule(timestamp.table_rule(walk.manifest), rule, mismatch=mismatch))
    dt = np.asarray(dt_ps, np.float64)
    b1, f1 = skew.lookup(np.asarray(channel_1))
    b2, f2 = skew.lookup(np.asarray(channel_2))
    corrected = dt - b1 + b2
    status = np.where(f1 & f2, 0, 1).astype(np.int8)
    counts = {"events": int(len(dt))}
    if walk is not None:
        values_1, status_1 = walk_inputs(**side_1)
        values_2, status_2 = walk_inputs(**side_2)
        difference, out_1, out_2 = core.walk_difference(walk, values_1, values_2)
        status[(status == 0) & ((status_1 != 0) | (status_2 != 0))] = 2
        corrected = corrected - np.where(status == 0, difference, 0.0)
        for name in INPUTS:
            counts[f"clamped {name}"] = int(np.count_nonzero((status == 0) & out_1[name])
                                            + np.count_nonzero((status == 0) & out_2[name]))
    if rule != "top":
        offset_1, acts_1, counts_1 = timestamp.side_offset(channel_1, candidates_1, side_1, skew, walk, rule)
        offset_2, acts_2, counts_2 = timestamp.side_offset(channel_2, candidates_2, side_2, skew, walk, rule)
        corrected = corrected + np.where(status == 0, offset_1 - offset_2, 0.0)
        counts["rule acts"] = int(np.count_nonzero((status == 0) & (acts_1 | acts_2)))
        for key in counts_1:
            if key != "rule acts":
                counts[key] = counts_1[key] + counts_2[key]
        counts["timestamp rule"] = {**rules[-1], "tables": [r["table_rule"] for r in rules]}
    for code, label in enumerate(CORRECTION_STATUS):
        counts[label] = int(np.count_nonzero(status == code))
    counts[f"uncorrected {'kept' if policy == 'keep' else 'dropped'}"] = int(np.count_nonzero(status != 0))
    return np.where(status == 0, corrected, dt if policy == "keep" else np.nan), status, counts


def correct_features(features, skew, walk, *, policy="keep", rule="top", mismatch=False):
    """``correct_dt`` of a ``FeatureBatch`` (sides 1 and 2, keV from the selection's calibration; under a spec 014
    rule the candidates from ``features.candidates``)."""
    def side(s):
        return {"q_top": features.column(f"q_top_{s}"), "q_time": features.column(f"q_time_{s}"),
                "doi_ratio": features.column(f"doi_ratio_{s}"), "kev": features.column(f"energy_kev_{s}")}
    candidates = features.candidates if rule != "top" else None
    return correct_dt(features.column("dt_ps"), features.column("time_channel_1"), features.column("time_channel_2"),
                      side(1), side(2), skew, walk, policy=policy, rule=rule, mismatch=mismatch,
                      candidates_1=None if candidates is None else candidates[:, 0],
                      candidates_2=None if candidates is None else candidates[:, 1])


def check_tables(skew_lut, skew_sha256, walk_lut, setup):
    """FR-4/FR-6 pairing and identity: the walk LUT's map is the run's and ``core.check_pairing`` holds (a skew
    LUT solved with walk never without its walk LUT; a walk LUT only with the skew LUT it was fitted with or one
    solved with it). ``skew_lut``: an IMAS ``CanonicalLut`` already read against ``setup``."""
    if walk_lut is not None and walk_lut.manifest.get("map_sha256") != setup.provenance["map_sha256"]:
        raise core.WalkError("the walk LUT was fitted for another map (map SHA-256 differs)")
    core.check_pairing(skew_lut, skew_sha256, walk_lut)


def export_walk(walk_path, out):
    """FR-6: the walk LUT and its manifest (byte copies, SHA-256 checked), ``imas_walk_curves.tsv`` (one row per
    bin) and ``export.json``; existing outputs are refused. PETsys's native table has no walk."""
    walk_path, out = Path(walk_path), Path(out)
    lut = WalkLut.read(walk_path)
    targets = [out / name for name in EXPORT_FILES]
    existing = [t.name for t in targets if t.exists()]
    if existing:
        raise core.WalkError(f"outputs exist in {out}: {existing}")
    out.mkdir(parents=True, exist_ok=True)
    table, manifest = walk_path.read_bytes(), core.manifest_path(walk_path).read_bytes()
    if hashlib.sha256(table).hexdigest() != lut.sha256:
        raise core.WalkError("walk LUT changed while it was exported")
    targets[0].write_bytes(table)
    targets[1].write_bytes(manifest)
    lines = ["block\tbin\tinput_1\tlow_1\thigh_1\tinput_2\tlow_2\thigh_2\toffset_ps\tevents\tfilled_from_nearest"]
    for block, axes in BLOCKS.items():
        filled = {tuple(i) for i in lut.filled[block]}
        for index in np.ndindex(lut.offset_ps[block].shape):
            cells = []
            for axis, name in enumerate(axes):
                edges = lut.edges[block][axis]
                cells += [name, repr(float(edges[index[axis]])), repr(float(edges[index[axis] + 1]))]
            cells += [""] * (6 - len(cells))
            lines.append("\t".join([block, ",".join(map(str, index)), *cells, repr(float(lut.offset_ps[block][index])),
                                    str(int(lut.events[block][index])), str(int(index in filled))]))
    targets[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = {"schema": "imas-walk-export/1", "requirement": "012-FR-6",
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": str(walk_path),
              "walk_lut_sha256": lut.sha256, "skew_lut_sha256": lut.manifest.get("skew_lut_sha256"),
              "skew_solved_with_walk_lut_sha256": lut.manifest.get("skew_solved_with_walk_lut_sha256"),
              "map_sha256": lut.manifest.get("map_sha256"), "convention": CONVENTION, "edges": "clamp",
              "files": {t.name: hashlib.sha256(t.read_bytes()).hexdigest() for t in targets[:3]},
              "native_table": "none: PETsys applies no walk correction"}
    targets[3].write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def export_main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew walk-export",
                                     description="Export an IMAS walk LUT: the table and manifest, a per-bin TSV and "
                                                 "export.json (spec 012 FR-6).")
    parser.add_argument("walk_lut", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = export_walk(args.walk_lut, args.out)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"walk LUT {record['walk_lut_sha256'][:12]} -> {args.out}")
    return 0
