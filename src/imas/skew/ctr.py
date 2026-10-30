"""IMAS point-source CTR: ``python -m src.imas.skew ctr CTR.json`` (spec 011 T15, FR-10).

The CTR file names the job work directory of the frozen LUTs, the energy
calibration, each new LUT and the legacy skew file (each with its SHA-256),
and uncorrected 22Na LDATs::

    {"schema": "imas-ctr-point/1", "job_work": ".../job", "encal": ".../job/r1/encal/encal.tsv",
     "luts": {"em": {"path": ".../job/lut_em.csv", "sha256": "..."}},
     "legacy": {"path": ".../skew_IMAS.txt", "sha256": "..."}, "output": ".../ctr/ctr.json",
     "acquisitions": [{"name": "na22_35", "ldat": ".../x_00000035.ldat", "source_mm": "lor_convergence",
                       "source_sigma_mm": [1, 1, 1], "declared": "1 cm cFOV"}]}

Frozen inputs: each LUT is an IMAS LUT of the same map, solved by the job
whose ``job_fingerprint.json`` is in ``job_work``; the configuration and frame
signs are that job's; no acquisition is one of its declared splits. The
energy calibration is one of the job's (FR-10) or, recorded as such, another
``encal`` output whose ``encal.json`` lies beside it (fitted the same way,
e.g. from the point-source data).

Cuts (as the job): IMAS production selection (10 ns window, 408-613 keV),
endpoints from the job's frame signs at 7 mm depth. As spec 008 FR-13: an
event is kept when its LOR passes within ``lor_distance_mm`` of the source,
between its endpoints; the bar shell and opening-angle cuts do not apply.
The source is located from the acquisition's own LORs (timing blind) unless
declared.

Tables: uncorrected, legacy and each LUT, on the same events: those whose two
TIME channels have an applied delay in every compared table (a LUT without
any applied delay is reported unavailable and not compared). Per table:
residual ``t_1 - t_2`` minus the geometric TOF, fitted as spec 008
(``src.timing.ctr``) except the model-free widths, which come from local
quadratic fits (``local_crossings``): Gaussian and model-free FWHM, FWTM, background,
bootstrap and systematic uncertainty (source +- sigma per axis, depth
variants), LOR-distance sensitivity and the TOF dependence. The bootstrap
draws one weight per event, shared by all tables.

Walk (spec 012 FR-5): ``walks`` = {name: {"skew": <luts name>, "path", "sha256"}} adds a table per entry:
that skew LUT's delays plus the walk LUT's correction (``walk.correct_features``), the pairing checked
(``walk.check_tables``). A skew LUT solved with walk is evaluated only inside a walk table. With walk
tables, an event also needs every walk input on both sides (stage ``walk input missing``), so every table
keeps the same events. ``walk_acceptance`` = [[reference, candidate], ...] judges each candidate against its
reference with the FR-10 rule, pooled (the gate) and per declared position (reported). FR-7: per pair, the
median residual (dt - TOF, |residual| < ``PROFILE_CORE_PS``) per bin of each walk input, both sides pooled
(side 2 with the residual negated), before (reference) and after (candidate), pooled and per position
(``profiles``, ``profiles_by_position``; bins of the first walk table's LUT). Diagnostics, not acceptance.

Timestamp rule (spec 014 FR-5): each LUT and walk table is applied under the rule recorded in its manifest
(``timestamp_rule``, absent = ``top``; a skew LUT and its walk LUT must agree). ``rule_tables`` = {name: {"base":
<legacy, luts or walks name>, "rule": <rule>}} evaluates a frozen table under another rule, the mismatch named
and recorded (e.g. ``em2_walk3@mean``, ``legacy@earliest``; ``earliest`` only for skew-only tables). Candidate
exclusions never drop an event, so every table keeps the same events. Per rule table and position: the share
of near-source events where the rule acts, and the candidate counts of the corrected events (before the LOR
cut). Each ``walk_acceptance`` pair also reports ``positions_not_worse``: no declared position worse than its
reference by more than their combined uncertainty (spec 014 acceptance with the pooled FR-10 rule).

Acceptance (FR-10): a LUT passes when its model-free FWHM is below the
legacy one by more than their uncertainties added in quadrature
(conservative: the tables share events). The Gaussian FWHM is reported for
each table but not judged: the residual peak is not Gaussian (owner). Beside it, the paired improvement:
the same bootstrap replicates and systematic variants fitted for both tables,
so their common fluctuations cancel. The width is a system CTR at these cuts,
not an intrinsic detector CTR.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from src.imas.skew import outputs, select as sel
from src.imas.skew import timestamp
from src.imas.skew import walk as iw
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing import ctr as core
from src.timing.cache import _sha256
from src.timing.ctr import CtrError
from src.timing.lut import DelayTable, LutError

SCHEMA = "imas-ctr-point/1"
REPORT_SCHEMA = "imas-ctr-point-report/1"
LOCALIZE = "lor_convergence"
UNCORRECTED = "uncorrected"
LEGACY = "legacy"
_REQUIRED = {"schema", "job_work", "encal", "luts", "legacy", "output", "acquisitions"}
_OPTIONAL = {"lor_distance_mm": 15.0, "sensitivity_lor_mm": [10.0, 25.0], "window_ps": 3000.0, "bin_ps": 10.0,
             "bootstrap": 200, "seed": 0, "depth_variants_mm": [4.0, 10.0], "max_records": None,
             "localization_events": 500_000, "walks": {}, "walk_acceptance": [], "rule_tables": {}}
_ACQUISITION_KEYS = {"name", "ldat", "source_mm", "source_sigma_mm"}
STAGES = ("records", "passed pre-geometry", "invalid endpoint", "no applied delay", "LOR outside distance",
          "near source")
WALK_STAGE = "walk input missing"       # spec 012: only with walk tables, after "no applied delay"
RULE_STAGE = "timestamp rule"           # spec 014: per table applied under a rule other than top
PROFILE_CORE_PS, PROFILE_BIN_PS = 600.0, 2.0   # spec 012 FR-7 profiles: the bar fit's core window
FWHM_KEYS = ("model_free_fwhm_ps", "fwhm_ps")
MODEL_FREE = core.local_crossings      # owner 2026-10-26: local quadratic crossings (spec 011 FR-10)
ACCEPTANCE_KEY = "model_free_fwhm_ps"   # owner 2026-10-26: the peak is not Gaussian; Gaussian FWHM reported only


def _frozen(entry, label, base):
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise CtrError(f"{label}: {{path, sha256}} required")
    path = Path(entry["path"])
    return path if path.is_absolute() else base / path, str(entry["sha256"]).lower()


def load_ctr(path):
    """Validated CTR file (paths relative to it); the options as ``SimpleNamespace`` fields."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise CtrError(f"file schema must be {SCHEMA!r}")
    if not _REQUIRED <= set(raw) <= _REQUIRED | set(_OPTIONAL):
        raise CtrError(f"file keys: required {sorted(_REQUIRED)}, optional {sorted(_OPTIONAL)}")
    base = path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    if not isinstance(raw["luts"], dict) or not raw["luts"]:
        raise CtrError("luts: at least one {name: {path, sha256}}")
    if {UNCORRECTED, LEGACY} & set(raw["luts"]):
        raise CtrError(f"LUT names {UNCORRECTED!r} and {LEGACY!r} are reserved")
    acquisitions = []
    for item in raw["acquisitions"] if isinstance(raw["acquisitions"], list) else []:
        if not isinstance(item, dict) or not _ACQUISITION_KEYS <= set(item) <= _ACQUISITION_KEYS | {"declared"}:
            raise CtrError(f"acquisitions need {sorted(_ACQUISITION_KEYS)} (optional 'declared'): {item!r}")
        source = None if item["source_mm"] == LOCALIZE else _vector(item["source_mm"], f"{item['name']} source_mm")
        acquisitions.append(SimpleNamespace(
            name=str(item["name"]), ldat=local(item["ldat"]), source_mm=source,
            source_sigma_mm=_vector(item["source_sigma_mm"], f"{item['name']} source_sigma_mm", nonnegative=True),
            declared=None if item.get("declared") is None else str(item["declared"])))
    if not acquisitions:
        raise CtrError("at least one point-source acquisition required")
    if len({a.name for a in acquisitions}) != len(acquisitions):
        raise CtrError("acquisition names must be unique")
    options = {key: raw.get(key, default) for key, default in _OPTIONAL.items()}
    window, width = float(options["window_ps"]), float(options["bin_ps"])
    bins = 2 * window / width if window > 0 and width > 0 else 0
    if abs(bins - round(bins)) > 1e-9 or round(bins) < 10:
        raise CtrError("2 * window_ps must be a whole number (>= 10) of bin_ps")
    bootstrap = options["bootstrap"]
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0 or bootstrap == 1:
        raise CtrError("bootstrap: 0 (none) or at least 2 replicates")
    if not float(options["lor_distance_mm"]) > 0 or not all(float(d) > 0 for d in options["sensitivity_lor_mm"]):
        raise CtrError("LOR distances must be positive")
    walks = {}
    if not isinstance(options["walks"], dict):
        raise CtrError("walks: {name: {skew, path, sha256}}")
    for name, entry in options["walks"].items():
        if name in (UNCORRECTED, LEGACY) or name in raw["luts"]:
            raise CtrError(f"walk table name {name!r} is reserved or a LUT name")
        if not isinstance(entry, dict) or entry.get("skew") not in raw["luts"]:
            raise CtrError(f"walks.{name}: 'skew' must name one of the luts")
        path, sha = _frozen({k: v for k, v in entry.items() if k != "skew"}, f"walks.{name}", base)
        walks[name] = SimpleNamespace(skew=entry["skew"], path=path, sha256=sha)
    rules = rule_entries(options["rule_tables"], {LEGACY} | set(raw["luts"]) | set(walks))
    names = {UNCORRECTED, LEGACY} | set(raw["luts"]) | set(walks) | set(rules)
    pairs = options["walk_acceptance"]
    if not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2 or not set(pair) <= names
                                          or pair[0] == pair[1] for pair in pairs):
        raise CtrError("walk_acceptance: [[reference, candidate], ...] of table names")
    return SimpleNamespace(
        path=path, sha256=_sha256(path), job_work=local(raw["job_work"]), encal=local(raw["encal"]),
        luts={name: _frozen(entry, f"luts.{name}", base) for name, entry in raw["luts"].items()},
        legacy=_frozen(raw["legacy"], "legacy", base), output=local(raw["output"]), acquisitions=tuple(acquisitions),
        lor_distance_mm=float(options["lor_distance_mm"]),
        sensitivity_lor_mm=tuple(float(d) for d in options["sensitivity_lor_mm"]), window_ps=window, bin_ps=width,
        bootstrap=bootstrap, seed=int(options["seed"]),
        depth_variants_mm=tuple(float(d) for d in options["depth_variants_mm"]), max_records=options["max_records"],
        localization_events=int(options["localization_events"]), walks=walks, rule_tables=rules,
        walk_acceptance=[tuple(pair) for pair in pairs], edges_ps=np.linspace(-window, window, int(round(bins)) + 1))


def rule_entries(value, bases):
    """``rule_tables`` = {name: {base, rule}} (spec 014 FR-5): ``base`` one of ``bases``, ``rule`` one of
    ``timestamp.RULES``; names never reuse a table name."""
    if not isinstance(value, dict):
        raise CtrError("rule_tables: {name: {base, rule}}")
    out = {}
    for name, entry in value.items():
        if name in bases or name == UNCORRECTED:
            raise CtrError(f"rule table name {name!r} is reserved or a table name")
        if not isinstance(entry, dict) or set(entry) != {"base", "rule"} or entry["base"] not in bases:
            raise CtrError(f"rule_tables.{name}: {{base, rule}} with base one of {sorted(bases)}")
        if entry["rule"] not in timestamp.RULES:
            raise CtrError(f"rule_tables.{name}: rule one of {timestamp.RULES}")
        out[name] = SimpleNamespace(base=entry["base"], rule=entry["rule"])
    return out


def _vector(value, label, *, nonnegative=False):
    try:
        vector = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise CtrError(f"{label}: three numbers in mm required") from None
    if len(vector) != 3 or not all(math.isfinite(v) for v in vector) or (nonnegative and min(vector) < 0):
        raise CtrError(f"{label}: three finite{' nonnegative' if nonnegative else ''} numbers in mm required")
    return vector


def frozen_inputs(spec):
    """Job, configuration, signs, calibration and tables, each checked against the frozen record."""
    marker = spec.job_work / "job_fingerprint.json"
    if not marker.is_file():
        raise CtrError(f"{spec.job_work}: no job_fingerprint.json; the LUTs' job cannot be verified")
    job = json.loads(marker.read_text(encoding="utf-8"))
    declared = job["declared"]
    config = Path(declared["config"])
    setup = ImasSetup.from_files(config)
    inputs = {"job_work": str(spec.job_work), "job_sha256": job["sha256"], "config": str(config),
              "config_sha256": setup.provenance["config_sha256"], "map_sha256": setup.provenance["map_sha256"],
              "signs": declared["signs"], "luts": {}}
    tables, canonical, encal_job = {}, {}, set()
    for name, (path, sha) in spec.luts.items():
        if _sha256(path) != sha:
            raise CtrError(f"LUT {name}: SHA-256 of {path} is not the frozen {sha}")
        try:
            table = outputs.read_imas_lut(path, setup)
        except LutError as error:
            raise CtrError(f"LUT {name}: {error}") from None
        provenance = table.manifest["provenance"]
        if provenance.get("job_sha256") != job["sha256"]:
            raise CtrError(f"LUT {name}: solved by another job than {marker}")
        if table.input_correction != {"state": "uncorrected"}:
            raise CtrError(f"LUT {name}: solved from corrected data; offline application needs an uncorrected LUT")
        encal_job |= {run["encal_sha256"] for run in provenance.get("runs", {}).values()}
        tables[name], canonical[name] = DelayTable.from_lut(table), table
        inputs["luts"][name] = {"path": str(path), "sha256": sha, "table_sha256": table.manifest["table"]["sha256"],
                                "estimator": provenance.get("estimator"), "applied": len(tables[name].delays),
                                "solved_with_walk_lut_sha256": provenance.get("walk_lut_sha256"),
                                "timestamp_rule": timestamp.skew_rule(tables[name])}
    walk_tables = _walk_tables(spec, setup, tables, canonical, inputs)
    for name in list(tables):
        walk_sha = inputs["luts"][name]["solved_with_walk_lut_sha256"]
        if walk_sha:
            if not any(w.skew == name for w in spec.walks.values()):
                raise CtrError(f"LUT {name}: solved with walk LUT {walk_sha[:12]}...; evaluate it in a walks entry "
                               "with its walk LUT")
            del tables[name]           # evaluated only inside its walk table(s)
        elif inputs["luts"][name]["timestamp_rule"] != "top":
            tables[name] = WalkTable(tables[name], None, inputs["luts"][name]["timestamp_rule"])
    legacy_path, legacy_sha = spec.legacy
    if _sha256(legacy_path) != legacy_sha:
        raise CtrError(f"legacy: SHA-256 of {legacy_path} is not the frozen {legacy_sha}")
    legacy = outputs.read_legacy(legacy_path, setup)
    inputs["legacy"] = {"path": str(legacy_path), "sha256": legacy_sha, "applied": len(legacy.delays),
                        **legacy.provenance}
    encal_sha = _sha256(spec.encal)
    if encal_sha in encal_job:
        calibration = {"origin": "job", "encal": str(spec.encal), "encal_sha256": encal_sha}
    else:
        manifest = spec.encal.with_name("encal.json")
        if not manifest.is_file():
            raise CtrError(f"{spec.encal}: not the job's calibration and no encal.json beside it")
        written = json.loads(manifest.read_text(encoding="utf-8"))
        if written.get("provenance", {}).get("map_sha256") != setup.provenance["map_sha256"]:
            raise CtrError(f"{spec.encal}: calibration of another map")
        calibration = {"origin": "other encal output (recorded)", "encal": str(spec.encal), "encal_sha256": encal_sha,
                       "inputs": written.get("inputs")}
    inputs["calibration"] = calibration
    splits = {Path(s["path"]).resolve() for run in declared["runs"] for s in run["splits"]}
    names = {(s["name"], s["size_bytes"]) for run in declared["runs"] for s in run["splits"]}
    for acquisition in spec.acquisitions:
        if not acquisition.ldat.is_file():
            raise CtrError(f"{acquisition.name}: missing LDAT {acquisition.ldat}")
        if (acquisition.ldat.resolve() in splits
                or (acquisition.ldat.name, acquisition.ldat.stat().st_size) in names):
            raise CtrError(f"{acquisition.name}: {acquisition.ldat} is an input of the LUT job")
    zero = DelayTable({int(c): 0.0 for c in setup.time_channels})
    tables = {UNCORRECTED: zero, LEGACY: legacy, **tables, **walk_tables}
    tables |= _rule_tables(getattr(spec, "rule_tables", {}), tables, inputs)
    named = {name for pair in spec.walk_acceptance for name in pair}
    if named - set(tables):
        raise CtrError(f"walk_acceptance names tables that are not evaluated: {sorted(named - set(tables))} (a LUT "
                       "solved with walk is evaluated only inside its walk table)")
    return inputs, setup, tables


class WalkTable:
    """A skew ``DelayTable`` with a walk LUT (None: skew only) under a timestamp rule (``mismatch``: not the
    tables' own), corrected through ``walk.correct_features`` (spec 012 FR-5, spec 014 FR-5)."""

    def __init__(self, delays, walk, rule="top", mismatch=False):
        self.skew, self.walk, self.rule, self.mismatch = delays, walk, rule, mismatch
        self.delays = delays.delays


def table_rule(table):
    """The timestamp rule a table of ``frozen_inputs`` is applied under."""
    return table.rule if isinstance(table, WalkTable) else "top"


def correct(table, features):
    """One table's corrected ``dt`` (NaN where a walk table does not correct), whether both delays apply,
    whether a walk input is missing, the correction counts (None for a plain delay table) and whether the rule
    acts (None under ``top``)."""
    if not isinstance(table, WalkTable):
        b1, f1 = table.lookup(features.column("time_channel_1"))
        b2, f2 = table.lookup(features.column("time_channel_2"))
        # t_1 - t_2 exact in integers, then - b_1 + b_2
        return features.column("dt_ps").astype(np.float64) - b1 + b2, f1 & f2, np.zeros(len(b1), bool), None, None
    values, status, counts, acts = iw.correct_features(features, table.skew, table.walk, policy="drop",
                                                       rule=table.rule, mismatch=table.mismatch, with_acts=True)
    return values, status != 1, status == 2, counts, None if table.rule == "top" else acts


def _rule_tables(entries, tables, inputs):
    """``rule_tables`` as ``WalkTable``s of their base's delays and walk under the named rule (mismatch
    recorded)."""
    out = {}
    if entries:
        inputs["rule_tables"] = {}
    for name, entry in entries.items():
        if entry.base not in tables:
            raise CtrError(f"rule_tables.{name}: base {entry.base!r} is not evaluated (a LUT solved with walk is "
                           "evaluated only inside its walk table)")
        base = tables[entry.base]
        skew, walk = (base.skew, base.walk) if isinstance(base, WalkTable) else (base, None)
        fitted = table_rule(base)
        if entry.rule == fitted:
            raise CtrError(f"rule_tables.{name}: {entry.base!r} is already applied under {fitted!r}")
        if entry.rule == "earliest" and walk is not None:
            raise CtrError(f"rule_tables.{name}: rule 'earliest' is for skew-only tables (spec 014 clarify 4)")
        out[name] = WalkTable(skew, walk, entry.rule, mismatch=True)
        inputs["rule_tables"][name] = {"base": entry.base, **timestamp.check_rule(fitted, entry.rule, mismatch=True)}
    return out


def _walk_tables(spec, setup, tables, canonical, inputs):
    """``spec.walks`` as ``WalkTable``s, each walk LUT checked (SHA-256, map, pairing with its skew LUT)."""
    out = {}
    inputs["walks"] = {}
    for name, entry in spec.walks.items():
        if _sha256(entry.path) != entry.sha256:
            raise CtrError(f"walk {name}: SHA-256 of {entry.path} is not the frozen {entry.sha256}")
        try:
            walk = iw.WalkLut.read(entry.path)
            iw.check_tables(canonical[entry.skew], spec.luts[entry.skew][1], walk, setup)
        except (iw.core.WalkError, OSError, KeyError) as error:
            raise CtrError(f"walk {name}: {error}") from None
        rule = timestamp.table_rule(walk.manifest)
        if rule != timestamp.skew_rule(tables[entry.skew]):
            raise CtrError(f"walk {name}: walk LUT fitted under timestamp rule {rule!r}, its skew LUT "
                           f"{entry.skew!r} under {timestamp.skew_rule(tables[entry.skew])!r}")
        out[name] = WalkTable(tables[entry.skew], walk, rule)
        inputs["walks"][name] = {"skew": entry.skew, "path": str(entry.path), "sha256": entry.sha256,
                                 "walk_lut_sha256": walk.sha256, "fitted_with_skew_lut_sha256":
                                 walk.manifest.get("skew_lut_sha256"), "fit_kind": walk.manifest.get("fit_kind"),
                                 "timestamp_rule": rule}
    return out


class Accumulator:
    """Per table x variant x acquisition residual histograms on shared events; bootstrap weights shared."""

    def __init__(self, spec, tables, sources):
        self.spec, self.names, self.tables = spec, list(tables), list(tables.values())
        self.sources = [np.asarray(s, np.float64) for s in sources]
        self.variants = core.variants(spec)
        self.edges = spec.edges_ps
        t, n, bins = len(self.tables), len(spec.acquisitions), len(self.edges) - 1
        self.hist = np.zeros((t, len(self.variants), n, bins))
        self.boot = np.zeros((t, spec.bootstrap, n, bins))
        self.tof = np.zeros((t, n, len(core.TOF_EDGES_PS) - 1, bins))
        self.walked = any(isinstance(table, WalkTable) and table.walk is not None for table in self.tables)
        stages = STAGES[:4] + (WALK_STAGE,) + STAGES[4:] if self.walked else STAGES
        self.stages = [dict.fromkeys(stages, 0) | {"in window": dict.fromkeys(self.names, 0)} for _ in range(n)]
        self.ruled = [name for name, table in zip(self.names, self.tables) if table_rule(table) != "top"]
        for stage in self.stages if self.ruled else ():
            stage[RULE_STAGE] = {name: {"near source": 0, "rule acts near source": 0} for name in self.ruled}
        self.diagnostics = None
        if self.ruled:                          # spec 014 FR-7: lead and CTR per lead bin, per position
            from src.imas.skew.timestampplots import RuleDiagnostics
            self.diagnostics = RuleDiagnostics({name: self.tables[self.names.index(name)] for name in self.ruled},
                                               positions(spec), self.edges)
        self.pre_geometry = [{} for _ in range(n)]
        self.channels_seen = set()
        self.profile = None
        if self.walked:                         # FR-7: residual histograms per walk-input bin, per position
            walk = next(table.walk for table in self.tables if isinstance(table, WalkTable)
                        and table.walk is not None)
            self.profile_edges = {name: np.asarray(walk.edges[block][k], np.float64)
                                  for block, names in walk.model.blocks.items() for k, name in enumerate(names)}
            self.groups = positions(spec)
            self.group_of = {k: g for g, index in enumerate(self.groups.values()) for k in index}
            self.profile_bins = int(round(2 * PROFILE_CORE_PS / PROFILE_BIN_PS))
            self.profile = {name: np.zeros((t, len(self.groups), len(e) - 1, self.profile_bins))
                            for name, e in self.profile_edges.items()}

    def add(self, index, features, endpoints):
        """``endpoints``: (codes, endpoints) per depth (nominal first)."""
        spec, stages = self.spec, self.stages[index]
        codes, found = np.unique(features.codes, return_counts=True)
        for code, count in zip(codes.tolist(), found.tolist()):
            label = sel.code_label(code)
            self.pre_geometry[index][label] = self.pre_geometry[index].get(label, 0) + count
        stages["records"] += len(features.codes)
        stages["passed pre-geometry"] += len(features.ints)
        if not len(features.ints):
            return
        ch1, ch2 = features.column("time_channel_1"), features.column("time_channel_2")
        corrected, applied, walk_missing, acting = correction(self.names, self.tables, features, stages)
        delayed, applied = applied, applied & ~walk_missing
        bins = len(self.edges) - 1
        sigma = np.array(spec.acquisitions[index].source_sigma_mm)
        for v, variant in enumerate(self.variants):
            codes, rows = endpoints[variant.depth_index]
            valid = codes != 1
            p, q = rows[:, :3], rows[:, 3:]
            source = self.sources[index] + np.array(variant.shift) * sigma
            distance, along = core.lor_distance_mm(source, p, q)
            with np.errstate(invalid="ignore"):
                near = valid & applied & (distance <= variant.lor_distance_mm) & (along > 0) & (along < 1)
            tof = core.geometric_tof_ps(source, p[near], q[near])
            binned = []
            for t in range(len(self.tables)):
                b = np.floor((corrected[t][near] - tof + spec.window_ps) / spec.bin_ps).astype(np.int64)
                inside = (b >= 0) & (b < bins)
                self.hist[t, v, index] += np.bincount(b[inside], minlength=bins)
                binned.append((b, inside))
            if v:
                continue
            stages["invalid endpoint"] += int(np.count_nonzero(~valid))
            stages["no applied delay"] += int(np.count_nonzero(valid & ~delayed))
            if self.walked:
                stages[WALK_STAGE] += int(np.count_nonzero(valid & delayed & walk_missing))
            stages["LOR outside distance"] += int(np.count_nonzero(valid & applied & ~near))
            stages["near source"] += int(np.count_nonzero(near))
            for name, acts in acting.items():
                stages[RULE_STAGE][name]["near source"] += int(np.count_nonzero(near))
                stages[RULE_STAGE][name]["rule acts near source"] += int(np.count_nonzero(acts & near))
            if self.diagnostics is not None:
                self.diagnostics.add(index, features, near, tof,
                                     {name: corrected[self.names.index(name)] for name in self.ruled})
            if self.profile is not None:
                self._profile(index, features, near, tof, corrected)
            self.channels_seen.update(np.unique(np.concatenate((ch1[near], ch2[near]))).tolist())
            k = np.searchsorted(core.TOF_EDGES_PS, tof, side="right") - 1
            in_tof = (k >= 0) & (k < len(core.TOF_EDGES_PS) - 1)
            for t, (b, inside) in enumerate(binned):
                stages["in window"][self.names[t]] += int(np.count_nonzero(inside))
                ok = inside & in_tof
                self.tof[t, index] += np.bincount(k[ok] * bins + b[ok], minlength=self.tof[t, index].size).reshape(
                    self.tof[t, index].shape)
            if spec.bootstrap:
                rng = np.random.default_rng([spec.seed, index, features.first_record])
                for r in range(spec.bootstrap):
                    weights = rng.poisson(1.0, len(tof))
                    for t, (b, inside) in enumerate(binned):
                        self.boot[t, r, index] += np.bincount(b[inside], weights=weights[inside], minlength=bins)

    def _profile(self, index, features, near, tof, corrected):
        g, bins = self.group_of[index], self.profile_bins
        sides = [(iw.side_inputs(features.column, s)[0], sign) for s, sign in (("1", 1.0), ("2", -1.0))]
        for t, values in enumerate(corrected):
            residual = values[near] - tof
            for inputs, sign in sides:
                r = np.floor((sign * residual + PROFILE_CORE_PS) / PROFILE_BIN_PS).astype(np.int64)
                core_window = (r >= 0) & (r < bins)
                for name, edges in self.profile_edges.items():
                    x = inputs[name][near]
                    ok = core_window & np.isfinite(x)
                    k = np.clip(np.searchsorted(edges, x[ok], side="right") - 1, 0, len(edges) - 2)
                    self.profile[name][t, g] += np.bincount(k * bins + r[ok], minlength=(len(edges) - 1) * bins
                                                            ).reshape(len(edges) - 1, bins)

    def view(self, t):
        """One table's histograms in the layout ``src.timing.ctr.summarize`` reads."""
        return SimpleNamespace(spec=self.spec, edges=self.edges, variants=self.variants, hist=self.hist[t],
                               boot=self.boot[t], tof=self.tof[t])


def correction(names, tables, features, stages):
    """Every table's ``correct``ed ``dt`` on one batch, the events with both delays in every table, those
    missing a walk input in a walk table, and per rule table whether the rule acts. Walk clamps and rule counts
    are added to ``stages`` (``walk clamped``, ``RULE_STAGE``)."""
    n = len(features.ints)
    applied, walk_missing, corrected, acting = np.ones(n, bool), np.zeros(n, bool), [], {}
    for name, table in zip(names, tables):
        values, delayed, missing, counts, acts = correct(table, features)
        applied &= delayed
        walk_missing |= missing
        corrected.append(values)
        if counts is None:
            continue
        if table.walk is not None:
            clamped = stages.setdefault("walk clamped", {}).setdefault(name, dict.fromkeys(iw.INPUTS, 0))
            for key in iw.INPUTS:
                clamped[key] += counts[f"clamped {key}"]
        if acts is not None:
            acting[name] = acts
            mine = stages.setdefault(RULE_STAGE, {}).setdefault(name, {})
            for key, value in counts.items():
                if key == "timestamp rule":
                    mine[key] = value
                elif key.startswith(("rule acts", "candidates")):
                    mine[key] = mine.get(key, 0) + value
    return corrected, applied, walk_missing, acting


def geometries(setup, signs, depths):
    return [ImasGeometry.from_setup(setup, transverse_sign=signs[0], axial_sign=signs[1], depth_mm=d,
                                    name=f"imas facets s_t {signs[0]:+d} s_a {signs[1]:+d} depth {d:g} mm")
            for d in depths]


def localize(acquisition, spec, lookup, selection, geometry):
    """Timing-blind source position from the first ``localization_events`` events with valid endpoints."""
    p, q, events = [], [], 0
    for features in sel.feature_batches(acquisition.ldat, lookup, selection, max_records=spec.max_records):
        if not len(features.ints):
            continue
        codes, rows = geometry.endpoints(features)
        rows = rows[codes != 1][:spec.localization_events - events]
        p.append(rows[:, :3])
        q.append(rows[:, 3:])
        events += len(rows)
        if events >= spec.localization_events:
            break
    if not events:
        raise CtrError(f"{acquisition.name}: no LOR with valid endpoints to locate the source")
    # bug-imas-ctr-localization: seeded by a grid search over the bore (scanner-wide background)
    point, miss, used, seed = core.seeded_converge_point(
        np.concatenate(p), np.concatenate(q), radius_mm=geometry.ring_r - 10.0,
        half_length_mm=max(abs(z) for z in geometry.ring_z) + geometry.local_origin[1])
    return {"method": LOCALIZE, "position_mm": [float(v) for v in point], "lors": events, "lors_used": used,
            "median_miss_mm": miss, "seed_mm": [float(v) for v in seed], "geometry": geometry.name}


def acceptance(results, paired=None):
    """Each LUT against the legacy table: improvement and combined uncertainty per FWHM kind.

    ``paired`` (``paired_differences``) adds the same-event uncertainty of the improvement, reported beside
    the FR-10 quadrature criterion.
    """
    legacy = results[LEGACY]["pooled"]
    out = {}
    for name, result in results.items():
        if name in (UNCORRECTED, LEGACY):
            continue
        pooled, entry = result["pooled"], {}
        for key in FWHM_KEYS:
            new, old = pooled["fit"].get(key), legacy["fit"].get(key)
            u_new = ((pooled.get("uncertainty") or {}).get(key) or {}).get("total")
            u_old = ((legacy.get("uncertainty") or {}).get(key) or {}).get("total")
            if None in (new, old, u_new, u_old):
                entry[key] = {"status": "unavailable"}
                continue
            combined = math.hypot(u_new, u_old)
            entry[key] = {"lut_ps": new, "legacy_ps": old, "improvement_ps": old - new, "combined_uncertainty_ps":
                          combined, "accepted": bool(old - new > combined)}
            if paired and name in paired:
                entry[key]["paired"] = paired[name][key]
        entry["accepted"] = bool(entry[ACCEPTANCE_KEY].get("accepted"))
        entry["judged_on"] = ACCEPTANCE_KEY
        out[name] = entry
    return out


def paired_differences(accumulator, reference=LEGACY, names=None):
    """Reference-minus-table FWHM on the same events (default: legacy minus each LUT): bootstrap SD (shared event
    weights) and systematic shifts."""
    pooled = accumulator.hist.sum(axis=2)                       # (table, variant, bins)
    boot = accumulator.boot.sum(axis=2)                          # (table, replicate, bins)
    legacy = accumulator.names.index(reference)
    edges = accumulator.edges
    out = {}
    for t, name in enumerate(accumulator.names):
        if name in (UNCORRECTED, reference) or (names is not None and name not in names):
            continue
        fits = {k: core.fit_histogram(pooled[k, 0], edges, crossings=MODEL_FREE) for k in (t, legacy)}
        if any(f["status"] != "fitted" for f in fits.values()):
            out[name] = {key: {"status": "unavailable"} for key in FWHM_KEYS}
            continue
        replicates = [{k: core.fit_histogram(boot[k, r], edges, start=fits[k]["_start"], crossings=MODEL_FREE)
                       for k in (t, legacy)}
                      for r in range(boot.shape[1])]
        variants = {v.name: {k: core.fit_histogram(pooled[k, i], edges, start=fits[k]["_start"], crossings=MODEL_FREE)
                             for k in (t, legacy)}
                    for i, v in enumerate(accumulator.variants) if v.group}
        entry = {}
        for key in FWHM_KEYS:
            def improvement(pair):
                ok = all(f["status"] == "fitted" and f[key] is not None for f in pair.values())
                return pair[legacy][key] - pair[t][key] if ok else None
            nominal = improvement(fits)
            draws = [d for d in (improvement(pair) for pair in replicates) if d is not None]
            statistical = float(np.std(draws, ddof=1)) if len(draws) >= 2 else None
            groups = {}
            for v in accumulator.variants:
                d = improvement(variants[v.name]) if v.group else None
                if d is not None:
                    groups.setdefault(v.group, []).append(d)
            combined = core.combine_uncertainty(nominal, groups, statistical) if nominal is not None else None
            entry[key] = {"improvement_ps": nominal, "replicates": len(draws), **(combined or {}),
                          "significant": None if not combined or combined["total"] is None
                          else bool(nominal > combined["total"])}
        out[name] = entry
    return out


def positions(spec):
    """Acquisition indices per declared position (the acquisition name when none is declared), in file order."""
    groups = {}
    for k, acquisition in enumerate(spec.acquisitions):
        groups.setdefault(acquisition.declared or acquisition.name, []).append(k)
    return groups


def by_position(accumulator, t, groups):
    """One table's results with each position's acquisitions summed (``core.summarize`` per position)."""
    def merge(array):
        return np.stack([array[..., index, :].sum(axis=-2) for index in groups.values()], axis=-2)
    spec = SimpleNamespace(**{**vars(accumulator.spec), "acquisitions": tuple(SimpleNamespace(name=name)
                                                                             for name in groups)})
    view = SimpleNamespace(spec=spec, edges=accumulator.edges, variants=accumulator.variants,
                           hist=merge(accumulator.hist[t]), boot=merge(accumulator.boot[t]),
                           tof=np.stack([accumulator.tof[t][index].sum(axis=0) for index in groups.values()]))
    results = core.summarize(view, crossings=MODEL_FREE)
    results.pop("pooled")
    return results


def _judge(new, old):
    """FR-10 rule on ``ACCEPTANCE_KEY`` between two results entries (fit + uncertainty)."""
    value, reference = new["fit"].get(ACCEPTANCE_KEY), old["fit"].get(ACCEPTANCE_KEY)
    u_new = ((new.get("uncertainty") or {}).get(ACCEPTANCE_KEY) or {}).get("total")
    u_old = ((old.get("uncertainty") or {}).get(ACCEPTANCE_KEY) or {}).get("total")
    if None in (value, reference, u_new, u_old):
        return {"status": "unavailable"}
    combined = math.hypot(u_new, u_old)
    return {"table_ps": value, "reference_ps": reference, "improvement_ps": reference - value,
            "combined_uncertainty_ps": combined, "accepted": bool(reference - value > combined)}


def positions_not_worse(per_position):
    """Spec 014 FR-5: no position worse than its reference by more than their combined uncertainty. ``passed``
    is None when a position is unavailable and none is worse."""
    worse = [name for name, entry in per_position.items() if entry.get("status") != "unavailable"
             and -entry["improvement_ps"] > entry["combined_uncertainty_ps"]]
    unavailable = [name for name, entry in per_position.items() if entry.get("status") == "unavailable"]
    return {"passed": False if worse else (None if unavailable else True), "worse": worse,
            "unavailable": unavailable}


def walk_acceptance(spec, accumulator, results):
    """Spec 012 FR-5: each [reference, candidate] pair, pooled (the gate, with the paired same-event uncertainty)
    and per declared position (reported)."""
    groups = positions(spec)
    cache, out = {}, []
    for reference, candidate in spec.walk_acceptance:
        for name in (reference, candidate):
            if name not in cache:
                cache[name] = by_position(accumulator, accumulator.names.index(name), groups)
        paired = paired_differences(accumulator, reference, [candidate]).get(candidate, {}).get(ACCEPTANCE_KEY)
        pooled = _judge(results[candidate]["pooled"], results[reference]["pooled"])
        per_position = {name: _judge(cache[candidate][name], cache[reference][name]) for name in groups}
        not_worse = positions_not_worse(per_position)
        out.append({"reference": reference, "candidate": candidate, "judged_on": ACCEPTANCE_KEY,
                    "pooled": pooled | {"paired": paired}, "accepted": bool(pooled.get("accepted")),
                    "per_position": per_position, "positions_not_worse": not_worse,
                    "accepted_pooled_and_positions": bool(pooled.get("accepted")) and not_worse["passed"] is True,
                    "positions": {name: [spec.acquisitions[k].name for k in index] for name, index in groups.items()}})
    return out, {name: {p: {"fit": r["fit"], "uncertainty": r.get("uncertainty")} for p, r in by_pos.items()}
                 for name, by_pos in cache.items()}


def run_ctr(spec, *, log=print):
    started = time.perf_counter()
    inputs, setup, tables = frozen_inputs(spec)
    unavailable = {name: "no applied delay" for name, table in tables.items() if not table.delays}
    tables = {name: table for name, table in tables.items() if name not in unavailable}
    if set(tables) <= {UNCORRECTED, LEGACY}:
        raise CtrError(f"no LUT with applied delays to evaluate ({sorted(unavailable)})")
    setup = ImasSetup.from_files(inputs["config"], spec.encal)
    lookup = setup.lookup()
    selection = sel.Selection.production(lookup)
    hypotheses = geometries(setup, inputs["signs"], (ImasGeometry.from_setup(setup).depth_mm,)
                            + spec.depth_variants_mm)
    acquisitions, sources = [], []
    for acquisition in spec.acquisitions:
        entry = {"name": acquisition.name, "ldat": str(acquisition.ldat),
                 "size_bytes": acquisition.ldat.stat().st_size, "sha256": _sha256(acquisition.ldat),
                 "declared": acquisition.declared, "source_sigma_mm": list(acquisition.source_sigma_mm)}
        if acquisition.source_mm is None:
            entry["source"] = localize(acquisition, spec, lookup, selection, hypotheses[0])
            log(f"{acquisition.name}: source at {np.round(entry['source']['position_mm'], 2).tolist()} mm "
                f"(median LOR miss {entry['source']['median_miss_mm']:.2f} mm)")
        else:
            entry["source"] = {"method": "declared", "position_mm": list(acquisition.source_mm)}
        acquisitions.append(entry)
        sources.append(entry["source"]["position_mm"])
    accumulator = Accumulator(spec, tables, sources)
    for index, acquisition in enumerate(spec.acquisitions):
        clock = time.perf_counter()
        for features in sel.feature_batches(acquisition.ldat, lookup, selection, max_records=spec.max_records):
            rows = [h.endpoints(features) for h in hypotheses] if len(features.ints) else None
            accumulator.add(index, features, rows)
        stages = accumulator.stages[index]
        log(f"{acquisition.name}: {stages['records']:,} records, {stages['near source']:,} near the source "
            f"({time.perf_counter() - clock:.1f} s)")
    results = {name: core.summarize(accumulator.view(t), crossings=MODEL_FREE)
               for t, name in enumerate(accumulator.names)}
    report = {
        "schema": REPORT_SCHEMA, "requirement": "011-FR-10",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ctr_file": str(spec.path), "ctr_file_sha256": spec.sha256, "frozen_inputs": inputs,
        "acquisitions": acquisitions, "unavailable_tables": unavailable,
        "cuts": {"pre_geometry": selection.describe(), "geometry": hypotheses[0].name,
                 "lor_distance_mm": spec.lor_distance_mm, "lor_between_endpoints": True,
                 "window_ps": spec.window_ps, "bin_ps": spec.bin_ps, "max_records": spec.max_records,
                 "same_events": "both TIME channels with an applied delay in every compared table"
                 + (", and every walk input on both sides" if accumulator.walked else "")
                 + ("; candidate exclusions never drop an event" if accumulator.ruled else "")},
        "model": "Gaussian + flat background, binned Poisson likelihood (reported only); model-free FWHM/FWTM "
                 "above the fitted background from local quadratic fits (peak within +-"
                 f"{core.PEAK_WINDOW:g} x rough FWHM, crossings within +-{core.EDGE_WINDOW:g} x rough FWHM); "
                 f"Gaussian FWTM/FWHM {core.GAUSSIAN_FWTM_TO_FWHM:.3f}",
        "uncertainty": f"statistical: {spec.bootstrap} seeded Poisson bootstrap replicates, one weight per event "
                       "shared by all tables; systematic: source +- sigma per axis, depth "
                       f"{'/'.join(f'{d:g}' for d in spec.depth_variants_mm)} mm; LOR "
                       f"{'/'.join(f'{d:g}' for d in spec.sensitivity_lor_mm)} mm sensitivity not added",
        "counts": {a.name: {"stages": accumulator.stages[k], "pre_geometry": accumulator.pre_geometry[k]}
                   for k, a in enumerate(spec.acquisitions)},
        "coverage": {name: {"applied_channels": len(table.delays)} for name, table in tables.items()}
        | {"channels_near_source": len(accumulator.channels_seen)},
        "results": results, "acceptance": acceptance(results, paired_differences(accumulator)),
        **_walk_report(spec, accumulator, results), **_rule_report(spec, accumulator),
        "interpretation": "system CTR at these cuts (includes residual offset errors and endpoint blur); not an "
                          "intrinsic detector CTR; the acceptance adds uncertainties in quadrature although the "
                          "tables share events (conservative)",
        "seconds": round(time.perf_counter() - started, 1),
    }
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    if accumulator.diagnostics is not None:
        from src.imas.skew import timestampplots
        report["timestamp_plots"] = [path.name for path in timestampplots.write_plots(
            spec.output, report["timestamp_diagnostics"])]
    spec.output.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    _write_histograms(spec, accumulator)
    _plot(spec, accumulator, results)
    return report


def _median(hist):
    """Median residual (ps) of histograms over the profile core window (last axis); None when empty."""
    out = []
    for counts in hist:
        total = counts.sum()
        if not total:
            out.append(None)
            continue
        cumulative = np.cumsum(counts)
        k = int(np.searchsorted(cumulative, total / 2))
        below = cumulative[k] - counts[k]
        out.append(float(-PROFILE_CORE_PS + PROFILE_BIN_PS * (k + (total / 2 - below) / counts[k])))
    return out


def walk_profiles(spec, accumulator):
    """FR-7: per ``walk_acceptance`` pair, residual-vs-input profiles before (reference) and after (candidate),
    pooled and per position, in ``walkplots.plot_profiles``'s layout."""
    pooled, by_position = {}, {}
    for reference, candidate in spec.walk_acceptance:
        label = f"22Na {reference} -> {candidate}"
        a, b = accumulator.names.index(reference), accumulator.names.index(candidate)

        def row(name, select):
            before, after = select(accumulator.profile[name][a]), select(accumulator.profile[name][b])
            return {"edges": accumulator.profile_edges[name].tolist(),
                    "events": after.sum(axis=-1).astype(int).tolist(),
                    "before_median_ps": _median(before), "after_median_ps": _median(after)}
        pooled[label] = {name: row(name, lambda h: h.sum(axis=0)) for name in accumulator.profile}
        by_position[label] = {position: {name: row(name, lambda h, g=g: h[g]) for name in accumulator.profile}
                              for g, position in enumerate(accumulator.groups)}
    return {"profiles": pooled, "profiles_by_position": by_position,
            "profiles_note": f"median of dt - TOF within |residual| < {PROFILE_CORE_PS:g} ps per input bin, both "
                             "sides pooled (side 2 negated); before = reference, after = candidate; diagnostics"}


def _walk_report(spec, accumulator, results):
    if not spec.walk_acceptance:
        return {}
    judged, by_pos = walk_acceptance(spec, accumulator, results)
    return {"walk_acceptance": judged, "results_by_position": by_pos, **walk_profiles(spec, accumulator)}


def _rule_report(spec, accumulator):
    """Spec 014 FR-5: per table applied under a rule other than ``top``, its rule, the share of near-source
    events where it acts and the candidate counts, pooled and per declared position."""
    if not accumulator.ruled:
        return {}
    out = {}
    for name in accumulator.ruled:
        table = accumulator.tables[accumulator.names.index(name)]

        def merged(index):
            total = {}
            for k in index:
                for key, value in accumulator.stages[k][RULE_STAGE][name].items():
                    if key != "timestamp rule":
                        total[key] = total.get(key, 0) + value
            near = total.get("near source", 0)
            return total | {"rule_acts_share": total.get("rule acts near source", 0) / near if near else None}
        out[name] = {"rule": table.rule, "mismatch": table.mismatch,
                     "pooled": merged(range(len(spec.acquisitions))),
                     "per_position": {p: merged(index) for p, index in positions(spec).items()}}
    return {"timestamp_rules": out, "timestamp_rules_note": "rule acts: on near-source events (nominal variant); "
            "candidate counts: on the corrected events of the selection, before the LOR cut (spec 014 FR-5)",
            "timestamp_diagnostics": accumulator.diagnostics.summary(),
            "timestamp_diagnostics_note": "spec 014 FR-7 diagnostics, not validation evidence: per side the "
            "earliest-minus-top corrected time (lead); CTR per event lead bin before (the same tables under top) "
            "and after (the table's rule); rule acts per minimodule"}


def _write_histograms(spec, accumulator):
    edges = accumulator.edges
    pooled = accumulator.hist[:, 0].sum(axis=1)
    with open(spec.output.with_name(spec.output.stem + "_histograms.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["residual_low_ps", "residual_high_ps", *accumulator.names])
        for k in range(len(edges) - 1):
            writer.writerow([f"{edges[k]:g}", f"{edges[k + 1]:g}", *(int(pooled[t, k]) for t in range(len(pooled)))])


def _plot(spec, accumulator, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = accumulator.edges
    centers = (edges[:-1] + edges[1:]) / 2
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, scale in zip(axes, ("linear", "log")):
        for t, name in enumerate(accumulator.names):
            pooled = accumulator.hist[t, 0].sum(axis=0)
            fit = results[name]["pooled"]["fit"]
            label = name if fit.get("status") != "fitted" else (
                f"{name}: FWHM {fit['fwhm_ps']:.0f} ps, model-free "
                + (f"{fit['model_free_fwhm_ps']:.0f}" if fit["model_free_fwhm_ps"] is not None else "n/a") + " ps")
            ax.step(centers, pooled, where="mid", lw=0.9, label=label)
        ax.set_yscale(scale)
        ax.set_xlabel("corrected t1 - t2 minus geometric TOF (ps)")
        ax.set_ylabel(f"events / {spec.bin_ps:g} ps")
        ax.legend(fontsize=8)
    figure.suptitle(f"IMAS point-source CTR, same events, LOR <= {spec.lor_distance_mm:g} mm")
    figure.tight_layout()
    figure.savefig(spec.output.with_suffix(".png"), dpi=120)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew ctr", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ctr_file", type=Path, help=f"CTR file (schema {SCHEMA})")
    args = parser.parse_args(argv)
    try:
        spec = load_ctr(args.ctr_file)
        report = run_ctr(spec)
    except CtrError as error:
        print(f"ctr refused: {error}", file=sys.stderr)
        return 2
    for name, result in report["results"].items():
        fit, unc = result["pooled"]["fit"], result["pooled"].get("uncertainty") or {}
        if fit.get("status") != "fitted":
            print(f"{name}: {fit.get('status')}")
            continue
        print(f"{name}: model-free FWHM {fit['model_free_fwhm_ps']} +- "
              f"{(unc.get('model_free_fwhm_ps') or {}).get('total')} ps, Gaussian FWHM {fit['fwhm_ps']:.1f} +- "
              f"{(unc.get('fwhm_ps') or {}).get('total')} ps, {fit['events']:.0f} events")
    print(json.dumps(report["acceptance"], indent=1, default=float))
    print(f"-> {spec.output}")
    return 0
