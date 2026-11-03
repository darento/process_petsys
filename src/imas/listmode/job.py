"""IMAS LDAT -> LM job: parallel file workers and one NN process (spec 015 T4, FR-7 to FR-11).

Workers (spawned processes; ``workers=0`` runs the same steps in this process) take whole files from one task
queue. Per ``LdatStream`` batch they select (``select``) and time (``timing``) and put one chunk on a bounded
result queue; at the end of a file they put its counts. This process is the NN process: per chunk it predicts
the axial y and DOI, builds the LM records and appends them to the file's segment (``<name>.lm.part``,
renamed to ``<name>.lm`` with a ``<name>.lm.json`` record when the file completes). Chunks are the stream's
fixed batches, so a segment does not depend on the worker count. The merge writes the legacy header and the
segments in natural order to ``<base>_all.lm`` with ``<base>_all.json`` (provenance) and ``<base>_log.txt``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
import traceback

import numpy as np
import yaml
from natsort import natsorted

from src.imas.listmode import nn as lm_nn
from src.imas.listmode import records, timing as lm_timing
from src.imas.listmode.select import (REJECTIONS, LmLookup, count_codes, read_mu, read_pairs, select_batch)
from src.imas.skew.setup import ImasSetup
from src.ldat_stream import DEFAULT_BATCH_RECORDS, LdatStream

SCHEMA = "imas-ldat-lm-job/1"
KEV_BINS = (1500, 0.0, 1500.0)
# legacy QUADRANT_MAP: minimodule (4 x 4) -> SM quadrant
QUADRANT = np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3], np.int64)
CHUNK_DTYPE = np.dtype([("t_min", "<i8"), ("pair", "<u2"), ("swapped", "u1"), ("dt", "<i2"),
                        ("kev", "<f8", 2), ("x_mm", "<f8", 2), ("y0_mm", "<f8", 2), ("rows", "<f4", (2, 8))])


class JobError(RuntimeError):
    """A run that cannot start or did not complete; the message names the cause."""


@dataclass
class Settings:
    config: str
    slab_en_map: str
    inputs: list
    out_dir: str
    nn: tuple                                   # predictor spec for nn.make_predictor
    pairs: str
    skew: str | None = None
    walk: str | None = None
    walk_encal: str | None = None
    rule: str = "top_earliest_mean"
    legacy_skew: str | None = None
    no_timing: bool = False
    workers: int = 1
    acq_time: float = 300.0
    activity: float = 0.0
    isotope: str = "F18"
    debug: bool = False
    resume: bool = False
    keep_segments: bool = False
    max_records: int | None = None
    batch_records: int = DEFAULT_BATCH_RECORDS
    root: str = "."
    extra: dict = field(default_factory=dict)


# --- per-process setup ---------------------------------------------------------------------------------------

class Context:
    """Setup, lookup and timing built once per process from ``Settings``."""

    def __init__(self, settings):
        config = yaml.safe_load(Path(settings.config).read_text(encoding="utf-8"))
        self.setup = ImasSetup.from_files(settings.config, root=settings.root)
        walk_mu = read_mu(settings.walk_encal) if settings.walk_encal else None
        self.lookup = LmLookup.build(self.setup, read_mu(settings.slab_en_map), read_pairs(settings.pairs),
                                     config["energy_range"], walk_mu)
        self.timing = lm_timing.from_options(self.setup, skew=settings.skew, walk=settings.walk, rule=settings.rule,
                                             legacy=settings.legacy_skew, none=settings.no_timing)
        if self.timing.needs_walk_kev and walk_mu is None:
            raise JobError("the walk LUT needs its energy calibration (walk encal)")
        self.energy_range = tuple(map(float, config["energy_range"]))


def first_timestamp(path):
    """t0: the earlier of the first hit of each side of the file's first record (raw ps, every hit)."""
    for batch in LdatStream(path, max_records=1):
        k = batch.first_hit[0]
        stamps = [batch.timestamp[k]] if batch.n1[0] else []
        if batch.n2[0]:
            stamps.append(batch.timestamp[k + batch.n1[0]])
        if stamps:
            return int(min(stamps))
    raise JobError(f"{path}: no record to take the first timestamp from")


def _chunk(features, dt, lookup):
    n = len(features)
    out = np.empty(n, CHUNK_DTYPE)
    ch = np.stack([features.column("time_channel_1"), features.column("time_channel_2")], axis=1)
    out["t_min"], out["pair"] = features.column("t_min"), features.column("pair")
    out["swapped"], out["dt"] = features.column("swapped"), dt
    out["kev"] = np.stack([features.column("energy_kev_1"), features.column("energy_kev_2")], axis=1)
    out["x_mm"], out["y0_mm"] = lookup.base.coord_x[ch], lookup.time_y[ch]
    out["rows"] = features.energy_rows
    return out


def process_file(index, path, settings, context, emit):
    """Stream one file; ``emit(message)`` gets ("chunk", index, number, array) then ("done", index, counts)."""
    started = time.perf_counter()
    stream = LdatStream(path, max_records=settings.max_records, batch_records=settings.batch_records)
    codes = {label: 0 for label in ("accepted",) + REJECTIONS}
    timing_counts, ties = {}, 0
    kev_hist = np.zeros(KEV_BINS[0], np.int64)
    n_sm = context.lookup.pair.shape[0]
    quadrant_hits = np.zeros((n_sm, 4), np.int64)
    select_s = timing_s = 0.0
    number = 0
    for batch in stream:
        t = time.perf_counter()
        features = select_batch(batch, context.lookup)
        select_s += time.perf_counter() - t
        t = time.perf_counter()
        dt, counts = context.timing.dt(features, context.lookup.walk_mu)
        timing_s += time.perf_counter() - t
        lm_timing.merge_counts(timing_counts, counts)
        lm_timing.merge_counts(codes, count_codes(features.codes))
        ties += int(features.column("module_tie").sum())
        if settings.debug and len(features):
            kev = features.floats.ravel()
            kev_hist += np.histogram(kev, bins=KEV_BINS[0], range=KEV_BINS[1:])[0]
            for s in (1, 2):
                np.add.at(quadrant_hits, (features.column(f"sm_{s}"), QUADRANT[features.column(f"mm_{s}") % 16]), 1)
        emit(("chunk", index, number, _chunk(features, dt, context.lookup)))
        number += 1
    if not stream.complete and settings.max_records is None:
        raise JobError(f"{path}: stream ended before the end of file")
    emit(("done", index, {
        "path": str(path), "size_bytes": stream.size_bytes, "records": stream.records, "chunks": number,
        "prefix_only": settings.max_records is not None and not stream.complete,
        "codes": codes, "timing": timing_counts, "module_ties": ties,
        "seconds": {"read_decode": stream.read_seconds + stream.decode_seconds, "select": select_s,
                    "timing": timing_s, "worker_total": time.perf_counter() - started},
        "kev_hist": kev_hist.tolist() if settings.debug else None,
        "quadrant_hits": quadrant_hits.tolist() if settings.debug else None}))


def _worker(settings, tasks, results):
    """Worker process: files from ``tasks`` until None; errors become ("error", index, text) messages."""
    try:
        context = Context(settings)
    except Exception:                      # noqa: BLE001 - reported to the NN process, which stops the run
        results.put(("fatal", -1, traceback.format_exc()))
        return
    while True:
        item = tasks.get()
        if item is None:
            return
        index, path = item
        try:
            process_file(index, path, settings, context, results.put)
        except Exception as error:         # noqa: BLE001 - the file fails, the run continues and reports it
            results.put(("error", index, f"{type(error).__name__}: {error}"))


# --- the NN process ------------------------------------------------------------------------------------------

def segment_path(out_dir, path):
    return Path(out_dir) / Path(path).name.replace(".ldat", ".lm")


def base_name(first_segment):
    """Legacy ``_all.lm`` base: the first segment's name without its last ``_`` field (the split index)."""
    stem = Path(first_segment).name
    return Path(first_segment).with_name("_".join(stem.split("_")[:-1]) if "_" in stem else Path(stem).stem)


def default_out_dir(lm_root, first_input, now=None):
    """Legacy ``LM_PATH_DIR/lm-<date>-<experiment>/``."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    experiment = "_".join(Path(first_input).name.split("_")[1:-2])
    return Path(lm_root) / f"lm-{stamp}-{experiment}"


def input_records(settings):
    """SHA-256 of every table, model and configuration the LM depends on."""
    files = {"config": settings.config, "slab_en_map": settings.slab_en_map, "pairs": settings.pairs}
    for key in ("skew", "walk", "walk_encal", "legacy_skew"):
        if getattr(settings, key):
            files[key] = getattr(settings, key)
    if settings.walk:
        files["walk_manifest"] = str(Path(settings.walk).with_name(Path(settings.walk).stem + ".manifest.json"))
    if settings.skew:
        files["skew_manifest"] = str(Path(settings.skew).with_suffix(".json"))
    if settings.nn[0] == "keras":
        files["nn_y"], files["nn_doi"] = settings.nn[1], settings.nn[2]
    out = {}
    for key, path in files.items():
        if not Path(path).is_file():
            raise JobError(f"{key}: {path} is not a file")
        out[key] = {"path": str(path), "sha256": lm_timing.sha256(path)}
    return out


def fingerprint(settings, inputs, t0, map_sha):
    """Everything that changes a segment's bytes (FR-8)."""
    record = {"schema": SCHEMA, "inputs": {k: v["sha256"] for k, v in inputs.items()}, "map_sha256": map_sha,
              "rule": settings.rule, "no_timing": settings.no_timing, "nn": list(map(str, settings.nn)),
              "max_records": settings.max_records, "batch_records": settings.batch_records, "t0": t0}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _segment_record_path(segment):
    return segment.with_name(segment.name + ".json")


def _completed(segment, path, fp):
    record_path = _segment_record_path(segment)
    if not (segment.is_file() and record_path.is_file()):
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (record.get("fingerprint") != fp or record.get("input", {}).get("size_bytes") != Path(path).stat().st_size
            or record.get("input", {}).get("name") != Path(path).name
            or segment.stat().st_size != record.get("coincidences", -1) * records.LM_DTYPE.itemsize):
        return None
    return record


class _Segments:
    """Open segment writers of the NN process, one per file in progress."""

    def __init__(self, out_dir, paths, predictor, t0, fp):
        self.out_dir, self.paths, self.predictor, self.t0, self.fp = Path(out_dir), paths, predictor, t0, fp
        self.open, self.next_chunk, self.written = {}, {}, {}
        self.nn_seconds = self.write_seconds = 0.0

    def chunk(self, index, number, chunk):
        if number != self.next_chunk.get(index, 0):
            raise JobError(f"{self.paths[index]}: chunk {number} out of order")
        self.next_chunk[index] = number + 1
        if index not in self.open:
            part = segment_path(self.out_dir, self.paths[index]).with_suffix(".lm.part")
            self.open[index] = part.open("wb")
            self.written[index] = 0
        if not len(chunk):
            return
        t = time.perf_counter()
        y, doi = self.predictor.predict(chunk["rows"].reshape(-1, 8))
        self.nn_seconds += time.perf_counter() - t
        t = time.perf_counter()
        out = records.build_chunk(chunk, y.reshape(-1, 2), doi.reshape(-1, 2), self.t0)
        self.open[index].write(out.tobytes())
        self.written[index] += len(out)
        self.write_seconds += time.perf_counter() - t

    def done(self, index, counts):
        if index not in self.open and counts["chunks"] == 0:
            self.chunk(index, 0, np.empty(0, CHUNK_DTYPE))
            self.next_chunk[index] = 0
        if self.next_chunk.get(index, 0) != counts["chunks"]:
            raise JobError(f"{self.paths[index]}: {self.next_chunk.get(index, 0)} chunks of {counts['chunks']}")
        handle = self.open.pop(index)
        handle.close()
        segment = segment_path(self.out_dir, self.paths[index])
        record = {"schema": SCHEMA, "fingerprint": self.fp, "coincidences": self.written[index],
                  "input": {"name": Path(self.paths[index]).name, "size_bytes": counts["size_bytes"]}, **counts}
        os.replace(handle.name, segment)
        _segment_record_path(segment).write_text(json.dumps(record, indent=1), encoding="utf-8")
        return record

    def abort(self, index):
        handle = self.open.pop(index, None)
        if handle is not None:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)

    def close(self):
        for index in list(self.open):
            self.abort(index)


def _prepare_dir(settings, segments):
    out = Path(settings.out_dir)
    if out.exists() and not settings.resume:
        stale = [p.name for p in out.iterdir() if p.name.endswith((".lm", ".lm.part", ".lm.json"))]
        if stale:
            raise JobError(f"{out} already holds LM segments ({stale[:3]}); continue it or choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        segment.with_suffix(".lm.part").unlink(missing_ok=True)


def run(settings, *, log=print):
    """Convert ``settings.inputs`` to ``<base>_all.lm``; returns the provenance record. Raises ``JobError``
    (no ``_all.lm`` written) when a file fails."""
    started = time.perf_counter()
    paths = natsorted(str(p) for p in settings.inputs)
    if not paths:
        raise JobError("no input files")
    for path in paths:
        if not Path(path).is_file():
            raise JobError(f"input {path} is not a file")
    if settings.workers < 0:
        raise JobError("workers must be >= 0")
    context = Context(settings)                 # validates config, calibrations, pairs and tables up front
    inputs = input_records(settings)
    t0 = first_timestamp(paths[0])
    fp = fingerprint(settings, inputs, t0, context.setup.provenance["map_sha256"])
    segments = [segment_path(settings.out_dir, p) for p in paths]
    if len(set(segments)) != len(segments):
        raise JobError("two inputs map to the same segment name")
    _prepare_dir(settings, segments)
    done = {i: r for i, (s, p) in enumerate(zip(segments, paths)) if settings.resume
            and (r := _completed(s, p, fp)) is not None}
    todo = [i for i in range(len(paths)) if i not in done]
    log(f"{len(paths)} files, {len(done)} already complete, {len(todo)} to process with "
        f"{settings.workers or 'no'} worker processes")
    predictor = lm_nn.make_predictor(settings.nn)
    writer = _Segments(settings.out_dir, paths, predictor, t0, fp)
    failures = {}
    try:
        if todo:
            _dispatch(settings, paths, todo, context, writer, done, failures, log)
    finally:
        writer.close()
    if failures:
        for index, text in failures.items():
            log(f"FAILED {paths[index]}: {text}")
        raise JobError(f"{len(failures)} file(s) failed: {[Path(paths[i]).name for i in failures]}; no _all.lm written")
    return _merge(settings, paths, segments, done, inputs, context, t0, fp, writer, predictor, started, log)


def _dispatch(settings, paths, todo, context, writer, done, failures, log):
    def handle(message):
        kind, index = message[0], message[1]
        if kind == "chunk":
            writer.chunk(index, message[2], message[3])
        elif kind == "done":
            done[index] = writer.done(index, message[2])
            log(f"done {Path(paths[index]).name}: {message[2]['codes']['accepted']} coincidences of "
                f"{message[2]['records']} records ({len(done)}/{len(paths)})")
        elif kind == "error":
            writer.abort(index)
            failures[index] = message[2]
        else:
            raise JobError(f"worker setup failed:\n{message[2]}")

    if settings.workers == 0:
        for index in todo:
            try:
                process_file(index, paths[index], settings, context, handle)
            except Exception as error:      # noqa: BLE001 - same handling as a worker
                handle(("error", index, f"{type(error).__name__}: {error}"))
        return
    spawn = mp.get_context("spawn")
    tasks = spawn.Queue()
    results = spawn.Queue(maxsize=2 * settings.workers + 2)   # bounded: workers wait for the NN process
    for index in todo:
        tasks.put((index, paths[index]))
    workers = [spawn.Process(target=_worker, args=(settings, tasks, results), daemon=True)
               for _ in range(min(settings.workers, len(todo)))]
    for _ in workers:
        tasks.put(None)
    for w in workers:
        w.start()
    pending = set(todo)
    try:
        while pending:
            try:
                message = results.get(timeout=5)
            except queue.Empty:
                if not any(w.is_alive() for w in workers):
                    for index in pending:
                        failures.setdefault(index, "worker process ended without a result")
                    return
                continue
            handle(message)
            if message[0] in ("done", "error"):
                pending.discard(message[1])
    finally:
        for w in workers:
            if w.is_alive():
                w.terminate()
            w.join(timeout=5)


def _merge(settings, paths, segments, done, inputs, context, t0, fp, writer, predictor, started, log):
    base = base_name(segments[0])
    final = base.with_name(base.name + "_all.lm")
    totals = {"records": 0, "coincidences": 0, "codes": {}, "timing": {}, "module_ties": 0}
    seconds = {"read_decode": 0.0, "select": 0.0, "timing": 0.0}
    kev_hist, quadrant_hits = None, None
    with open(final, "wb") as out:
        out.write(records.header_bytes(settings.acq_time, settings.activity, settings.isotope))
        for index, segment in enumerate(segments):
            record = done[index]
            with open(segment, "rb") as part:
                shutil.copyfileobj(part, out, 1 << 24)
            totals["records"] += record["records"]
            totals["coincidences"] += record["coincidences"]
            totals["module_ties"] += record["module_ties"]
            lm_timing.merge_counts(totals["codes"], record["codes"])
            lm_timing.merge_counts(totals["timing"], record["timing"])
            for key in seconds:
                seconds[key] += record["seconds"][key]
            if settings.debug and record.get("kev_hist") is not None:
                kev_hist = np.asarray(record["kev_hist"]) + (0 if kev_hist is None else kev_hist)
                quadrant_hits = np.asarray(record["quadrant_hits"]) + (0 if quadrant_hits is None else quadrant_hits)
    elapsed = time.perf_counter() - started
    work = seconds["read_decode"] + seconds["select"] + seconds["timing"]
    provenance = {
        "schema": SCHEMA, "output": str(final), "size_bytes": final.stat().st_size, "fingerprint": fp,
        "created": datetime.now().isoformat(timespec="seconds"), "inputs": inputs,
        "map": context.setup.provenance, "timing": context.timing.describe(),
        "nn": getattr(predictor, "describe", {}),
        "cuts": {"min_ch": context.lookup.min_ch, "hit_energy_floor": f"energy >= {context.lookup.en_min_ch} a.u.",
                 "energy_range_kev_open": list(context.energy_range), "energy_calibration": settings.slab_en_map,
                 "coincidence_window": "none (legacy)"},
        "header": {"acq_time_s": settings.acq_time, "activity_mci": settings.activity, "isotope": settings.isotope},
        "t0_ps": t0, "workers": settings.workers, "max_records": settings.max_records,
        "files": [{"name": Path(p).name, "records": done[i]["records"], "coincidences": done[i]["coincidences"],
                   "codes": done[i]["codes"], "timing": done[i]["timing"], "prefix_only": done[i]["prefix_only"],
                   "seconds": done[i]["seconds"]} for i, p in enumerate(paths)],
        "totals": totals,
        "seconds": {"elapsed": elapsed, **seconds, "nn": writer.nn_seconds, "write": writer.write_seconds},
        "records_per_s": {"read_select_timing_per_worker": totals["records"] / work if work else None,
                          "whole_run": totals["records"] / elapsed if elapsed else None},
    }
    json_path = base.with_name(base.name + "_all.json")
    json_path.write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    _write_log(base.with_name(base.name + "_log.txt"), segments, final, provenance)
    if settings.debug:
        debug_plots(kev_hist, quadrant_hits, Path(settings.out_dir))
    if not settings.keep_segments:
        for segment in segments:
            segment.unlink()
            _segment_record_path(segment).unlink(missing_ok=True)
    log(f"Final output file: {final.name}, {provenance['size_bytes'] / 2**20:.2f} MB, "
        f"{totals['coincidences']} coincidences of {totals['records']} records in {elapsed:.1f} s")
    return provenance


def _write_log(path, segments, final, provenance):
    lines = [f"INFO:root:Input file: {s}, size: {s.stat().st_size / 2**20:.2f} MB" for s in segments]
    lines += [f"INFO:root:Final output file: {final}, size: {provenance['size_bytes'] / 2**20:.2f} MB",
              f"INFO:root:Time taken ldat to lm: {provenance['seconds']['elapsed']} seconds",
              f"INFO:root:Total number of events processed to LM: {provenance['totals']['coincidences']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def debug_plots(kev_hist, quadrant_hits, out_dir):
    """FR-10: accepted-side keV histogram with its Gaussian fit, and hits per SM quadrant (rings x 2 rows)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from src.fits import fit_gaussian
    if kev_hist is None:
        return
    n_bins, low, high = KEV_BINS
    bins = np.linspace(low, high, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    figure = plt.figure()
    plt.bar(centres, kev_hist, width=bins[1] - bins[0], align="center", color="blue", alpha=0.7)
    try:
        x, y, pars, _, _ = fit_gaussian(kev_hist, bins, cb=16)
        plt.plot(x, y, "-r")
        plt.legend([f"Energy res: {round(2.35 * pars[2] / pars[1] * 100, 2)}%\nCentroid {round(pars[1], 2)}"])
    except Exception as error:              # noqa: BLE001 - a failed fit is shown as such, never as a value
        plt.title(f"Total energy (fit failed: {type(error).__name__})")
    else:
        plt.title("Total energy")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    figure.savefig(out_dir / "total_energy.png")
    plt.close(figure)
    hits = np.asarray(quadrant_hits)
    n_sm = hits.shape[0]
    rings = n_sm // 24
    grid = np.zeros((2 * rings, 48), np.int64)
    for sm in range(n_sm):
        for q in range(4):
            grid[2 * (sm // 24) + q // 2, 2 * (sm % 24) + q % 2] = hits[sm, q]
    figure = plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap="viridis")
    plt.colorbar(label="Hits per SM")
    plt.xticks(range(48), [f"{i // 2}" if i % 2 == 0 else "" for i in range(48)])
    plt.yticks(range(2 * rings), [f"SM {i // 2 * 24}" if i % 2 == 0 else "" for i in range(2 * rings)])
    plt.xlabel("SuperModule")
    plt.ylabel("Row")
    plt.title("Hits per SuperModule")
    figure.savefig(out_dir / "sm_hits.png")
    plt.close(figure)


def settings_record(settings):
    return asdict(settings)
