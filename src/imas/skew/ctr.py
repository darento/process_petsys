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
(``src.timing.ctr``): Gaussian and model-free FWHM, FWTM, background,
bootstrap and systematic uncertainty (source +- sigma per axis, depth
variants), LOR-distance sensitivity and the TOF dependence. The bootstrap
draws one weight per event, shared by all tables.

Acceptance (FR-10): a LUT passes when its FWHM (Gaussian and model-free) is
below the legacy one by more than their uncertainties added in quadrature
(conservative: the tables share events). Beside it, the paired improvement:
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
             "localization_events": 500_000}
_ACQUISITION_KEYS = {"name", "ldat", "source_mm", "source_sigma_mm"}
STAGES = ("records", "passed pre-geometry", "invalid endpoint", "no applied delay", "LOR outside distance",
          "near source")
FWHM_KEYS = ("model_free_fwhm_ps", "fwhm_ps")


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
    return SimpleNamespace(
        path=path, sha256=_sha256(path), job_work=local(raw["job_work"]), encal=local(raw["encal"]),
        luts={name: _frozen(entry, f"luts.{name}", base) for name, entry in raw["luts"].items()},
        legacy=_frozen(raw["legacy"], "legacy", base), output=local(raw["output"]), acquisitions=tuple(acquisitions),
        lor_distance_mm=float(options["lor_distance_mm"]),
        sensitivity_lor_mm=tuple(float(d) for d in options["sensitivity_lor_mm"]), window_ps=window, bin_ps=width,
        bootstrap=bootstrap, seed=int(options["seed"]),
        depth_variants_mm=tuple(float(d) for d in options["depth_variants_mm"]), max_records=options["max_records"],
        localization_events=int(options["localization_events"]),
        edges_ps=np.linspace(-window, window, int(round(bins)) + 1))


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
    tables, encal_job = {}, set()
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
        tables[name] = DelayTable.from_lut(table)
        inputs["luts"][name] = {"path": str(path), "sha256": sha, "table_sha256": table.manifest["table"]["sha256"],
                                "estimator": provenance.get("estimator"), "applied": len(tables[name].delays)}
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
    return inputs, setup, {UNCORRECTED: zero, LEGACY: legacy, **tables}


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
        self.stages = [dict.fromkeys(STAGES, 0) | {"in window": dict.fromkeys(self.names, 0)} for _ in range(n)]
        self.pre_geometry = [{} for _ in range(n)]
        self.channels_seen = set()

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
        dt = features.column("dt_ps").astype(np.float64)
        applied = np.ones(len(dt), bool)
        corrected = []
        for table in self.tables:
            b1, f1 = table.lookup(ch1)
            b2, f2 = table.lookup(ch2)
            applied &= f1 & f2
            corrected.append(dt - b1 + b2)       # t_1 - t_2 exact in integers, then - b_1 + b_2
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
            stages["no applied delay"] += int(np.count_nonzero(valid & ~applied))
            stages["LOR outside distance"] += int(np.count_nonzero(valid & applied & ~near))
            stages["near source"] += int(np.count_nonzero(near))
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

    def view(self, t):
        """One table's histograms in the layout ``src.timing.ctr.summarize`` reads."""
        return SimpleNamespace(spec=self.spec, edges=self.edges, variants=self.variants, hist=self.hist[t],
                               boot=self.boot[t], tof=self.tof[t])


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
    point, miss, used = core.converge_point(np.concatenate(p), np.concatenate(q))
    return {"method": LOCALIZE, "position_mm": [float(v) for v in point], "lors": events, "lors_used": used,
            "median_miss_mm": miss, "geometry": geometry.name}


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
        entry["accepted"] = all(entry[k].get("accepted") for k in FWHM_KEYS)
        out[name] = entry
    return out


def paired_differences(accumulator):
    """Legacy-minus-LUT FWHM on the same events: bootstrap SD (shared event weights) and systematic shifts."""
    pooled = accumulator.hist.sum(axis=2)                       # (table, variant, bins)
    boot = accumulator.boot.sum(axis=2)                          # (table, replicate, bins)
    legacy = accumulator.names.index(LEGACY)
    edges = accumulator.edges
    out = {}
    for t, name in enumerate(accumulator.names):
        if name in (UNCORRECTED, LEGACY):
            continue
        fits = {k: core.fit_histogram(pooled[k, 0], edges) for k in (t, legacy)}
        if any(f["status"] != "fitted" for f in fits.values()):
            out[name] = {key: {"status": "unavailable"} for key in FWHM_KEYS}
            continue
        replicates = [{k: core.fit_histogram(boot[k, r], edges, start=fits[k]["_start"]) for k in (t, legacy)}
                      for r in range(boot.shape[1])]
        variants = {v.name: {k: core.fit_histogram(pooled[k, i], edges, start=fits[k]["_start"]) for k in (t, legacy)}
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
    results = {name: core.summarize(accumulator.view(t)) for t, name in enumerate(accumulator.names)}
    report = {
        "schema": REPORT_SCHEMA, "requirement": "011-FR-10",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ctr_file": str(spec.path), "ctr_file_sha256": spec.sha256, "frozen_inputs": inputs,
        "acquisitions": acquisitions, "unavailable_tables": unavailable,
        "cuts": {"pre_geometry": selection.describe(), "geometry": hypotheses[0].name,
                 "lor_distance_mm": spec.lor_distance_mm, "lor_between_endpoints": True,
                 "window_ps": spec.window_ps, "bin_ps": spec.bin_ps, "max_records": spec.max_records,
                 "same_events": "both TIME channels with an applied delay in every compared table"},
        "model": "Gaussian + flat background, binned Poisson likelihood; model-free FWHM/FWTM above the fitted "
                 f"background (3-bin smoothing); Gaussian FWTM/FWHM {core.GAUSSIAN_FWTM_TO_FWHM:.3f}",
        "uncertainty": f"statistical: {spec.bootstrap} seeded Poisson bootstrap replicates, one weight per event "
                       "shared by all tables; systematic: source +- sigma per axis, depth "
                       f"{'/'.join(f'{d:g}' for d in spec.depth_variants_mm)} mm; LOR "
                       f"{'/'.join(f'{d:g}' for d in spec.sensitivity_lor_mm)} mm sensitivity not added",
        "counts": {a.name: {"stages": accumulator.stages[k], "pre_geometry": accumulator.pre_geometry[k]}
                   for k, a in enumerate(spec.acquisitions)},
        "coverage": {name: {"applied_channels": len(table.delays)} for name, table in tables.items()}
        | {"channels_near_source": len(accumulator.channels_seen)},
        "results": results, "acceptance": acceptance(results, paired_differences(accumulator)),
        "interpretation": "system CTR at these cuts (includes residual offset errors and endpoint blur); not an "
                          "intrinsic detector CTR; the acceptance adds uncertainties in quadrature although the "
                          "tables share events (conservative)",
        "seconds": round(time.perf_counter() - started, 1),
    }
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    spec.output.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    _write_histograms(spec, accumulator)
    _plot(spec, accumulator, results)
    return report


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
