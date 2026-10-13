"""Bounded numeric cache for Cornell timing calibration (spec 008 T5).

Level 1 (feature cache): every record of each run7 file passing the holdout
and pre-geometry cuts, as geometry-free typed ``.npy`` blocks, plus each file's
first-failure counts. Owner decision (2026-10-13): all pre-geometry rows are
kept, so geometry hypotheses and alternative angle cuts rerun without LDAT
reads. A file's directory is renamed into place only after all its blocks and
``done.json`` are written; an interrupted ``.partial`` directory never looks
complete and is redone.

Level 2 (observation set): one canonical row per geometry-accepted physical
coincidence for one declared hypothesis (owner decision: per-event rows, no
histogram commitment before T6). Canonical orientation puts the lower TIME
channel first as ``i``: endpoints, features and shell TOF intervals swap and
``dt = t_i - t_j`` is negated consistently. Rows are partitioned by channel
pair with a bounded write buffer; physical pair counts are uint64.

A cache is reused only when its fingerprint (inputs, config/map/``.encal``
bytes, ingest/reader sources, holdout partitions, seed, prefix, schema)
matches; anything else is refused rather than overwritten.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import yaml

from src.cornell import timing_geometry, timing_ingest
from src import ldat_stream

SCHEMA_VERSION = 1
FEATURE_DTYPE = np.dtype([
    ("file", "<i2"), ("record", "<i8"), ("time_channel_1", "<i4"), ("time_channel_2", "<i4"),
    ("sm_1", "<i2"), ("mm_1", "<i2"), ("sm_2", "<i2"), ("mm_2", "<i2"), ("slab_1", "i1"), ("slab_2", "i1"),
    ("module_tie", "i1"), ("dt_ps", "<i8"), ("energy_kev_1", "<f4"), ("energy_kev_2", "<f4"),
    ("x_1", "<f8"), ("y_1", "<f8"), ("x_2", "<f8"), ("y_2", "<f8")])
OBSERVATION_DTYPE = np.dtype([
    ("file", "<i2"), ("record", "<i8"), ("role", "i1"), ("swapped", "i1"),
    ("channel_i", "<i4"), ("channel_j", "<i4"), ("dt_ps", "<i8"),
    ("sm_i", "<i2"), ("mm_i", "<i2"), ("sm_j", "<i2"), ("mm_j", "<i2"), ("slab_i", "i1"), ("slab_j", "i1"),
    ("module_tie", "i1"), ("energy_kev_i", "<f4"), ("energy_kev_j", "<f4"),
    ("endpoint_i_mm", "<f8", (3,)), ("endpoint_j_mm", "<f8", (3,)), ("shell_tof_ps", "<f8", (4,))])
PAIR_COUNT_DTYPE = np.dtype([("channel_i", "<i4"), ("channel_j", "<i4"), ("role", "i1"), ("count", "<u8")])
ROLES = ("train", "validation", "test")
_INT_COLUMNS = {"time_channel_1", "time_channel_2", "sm_1", "mm_1", "sm_2", "mm_2", "slab_1", "slab_2",
                "module_tie", "dt_ps", "record"}


class CacheMismatch(ValueError):
    """Existing cache content was produced from different inputs or settings."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    """JSON round trip, so fresh and stored fingerprints compare equal."""
    return json.loads(json.dumps(value))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class _Columns:
    """``column(name)`` view of structured rows for ``apply_geometry``."""

    def __init__(self, rows):
        self.rows = rows

    def column(self, name):
        dtype = np.int64 if name in _INT_COLUMNS else np.float64
        return np.ascontiguousarray(self.rows[name], dtype=dtype)


def feature_rows(features, file_index):
    """``FeatureBatch`` rows as ``FEATURE_DTYPE`` (keV rounded to float32)."""
    rows = np.empty(len(features.ints), FEATURE_DTYPE)
    rows["file"] = file_index
    for name in timing_ingest.FEATURE_INT:
        rows[name] = features.column(name)
    for name in timing_ingest.FEATURE_FLOAT:
        rows[name] = features.column(name)
    return rows


def _prepare(directory, manifest):
    """Create or reopen a cache directory; refuse unrelated or mismatched content."""
    directory = Path(directory)
    path = directory / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            raise CacheMismatch(f"{directory}: cache fingerprint differs; use a new directory")
        return existing
    if directory.exists() and any(directory.iterdir()):
        raise CacheMismatch(f"{directory}: not empty and not a timing cache")
    directory.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def _ingest_file(directory, item, config_path, encal_path, root, partition, seed, max_records, block_rows):
    """Worker: one file into ``features/fNNN``; returns the small ``done`` record."""
    setup = timing_ingest.LegacySetup.from_files(config_path, encal_path, root=root)
    lookup = timing_ingest.CompiledLookup.from_setup(setup)
    partition = timing_ingest.Partition(**partition)
    index, path = item["index"], Path(item["path"])
    final = Path(directory) / "features" / f"f{index:03d}"
    work = final.with_name(final.name + ".partial")
    if work.exists():
        shutil.rmtree(work)            # this cache's own interrupted file
    work.mkdir(parents=True)
    before = path.stat()
    if (before.st_size, before.st_mtime_ns) != (item["size_bytes"], item["mtime_ns"]):
        raise CacheMismatch(f"{path.name}: input identity changed since the cache was created")
    counts, buffered, blocks, records = {}, [], [], 0

    def flush(rows):
        name = f"b{len(blocks):05d}.npy"
        np.save(work / name, rows)
        blocks.append({"name": name, "rows": int(len(rows)), "sha256": _sha256(work / name)})

    pending = 0
    for features in timing_ingest.production_batches(path, lookup, partition, max_records=max_records, seed=seed):
        records += len(features.codes)
        codes, found = np.unique(features.codes, return_counts=True)
        for code, n in zip(codes.tolist(), found.tolist()):
            label = timing_ingest.code_label(code)
            counts[label] = counts.get(label, 0) + n
        if len(features.ints):
            buffered.append(feature_rows(features, index))
            pending += len(features.ints)
        while pending >= block_rows:
            rows = np.concatenate(buffered)
            flush(rows[:block_rows])
            buffered, pending = [rows[block_rows:]], len(rows) - block_rows
    if pending:
        flush(np.concatenate(buffered))
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise CacheMismatch(f"{path.name}: input changed during ingest")
    done = {"file": index, "name": path.name, "role": partition.role, "records": records,
            "rows": sum(b["rows"] for b in blocks), "first_failure": counts, "blocks": blocks}
    (work / "done.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
    os.rename(work, final)
    return done


def build_feature_cache(directory, inputs, *, config_path, encal_path, root, holdout,
                        energy_calibration_prefix, seed=0, max_records=None, workers=1, block_rows=1_000_000):
    """Ingest each input once into level-1 blocks; complete files are reused, never redone.

    ``inputs`` are ``{"index", "path"}`` items (``timing_dataset.ledger_inputs``).
    One process pool serves all pending files; workers return metadata only.
    """
    root = Path(root)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    items = []
    for item in sorted(inputs, key=lambda value: value["index"]):
        stat = Path(item["path"]).stat()
        items.append({"index": int(item["index"]), "path": str(item["path"]), "name": Path(item["path"]).name,
                      "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    partitions = {item["index"]: asdict(timing_ingest.Partition.from_holdout(
        holdout, item["index"], energy_calibration_prefix)) for item in items}
    fingerprint = {
        "schema_version": SCHEMA_VERSION, "kind": "cornell timing features",
        "inputs": [{k: item[k] for k in ("index", "name", "size_bytes", "mtime_ns")} for item in items],
        "config_sha256": _sha256(config_path), "map_sha256": _sha256(root / config["map_file"]),
        "encal_sha256": _sha256(encal_path),
        "sources_sha256": {name: _sha256(module.__file__) for name, module in
                           (("timing_ingest", timing_ingest), ("ldat_stream", ldat_stream))},
        "partitions": {str(k): v for k, v in partitions.items()}, "seed": seed, "max_records": max_records,
        "feature_dtype": FEATURE_DTYPE.descr}
    fingerprint = _plain(fingerprint)
    manifest = {"fingerprint": fingerprint, "fingerprint_sha256": _digest(fingerprint),
                "labels": list(timing_ingest.PRODUCTION_LABELS),
                "notes": ["rows: holdout-kept records passing every pre-geometry cut",
                          "energy_kev stored as float32"]}
    _prepare(directory, manifest)
    cache = FeatureCache(directory)
    pending = [item for item in items if item["index"] not in cache.files]
    arguments = [(str(directory), item, str(config_path), str(encal_path), str(root), partitions[item["index"]],
                  seed, max_records, block_rows) for item in pending]
    if workers > 1 and len(pending) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            list(pool.map(_ingest_file, *zip(*arguments)))
    else:
        for argument in arguments:
            _ingest_file(*argument)
    return FeatureCache(directory)


class FeatureCache:
    """Read-only view of completed level-1 files; opening it reads no LDAT."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.files = {}
        features = self.directory / "features"
        for item in self.manifest["fingerprint"]["inputs"]:
            done = features / f"f{item['index']:03d}" / "done.json"
            if done.exists():
                self.files[item["index"]] = json.loads(done.read_text(encoding="utf-8"))

    @property
    def complete(self):
        return len(self.files) == len(self.manifest["fingerprint"]["inputs"])

    def require_complete(self):
        if not self.complete:
            missing = sorted({i["index"] for i in self.manifest["fingerprint"]["inputs"]} - set(self.files))
            raise CacheMismatch(f"feature cache incomplete; files pending: {missing}")
        return self

    def blocks(self, *, verify=False):
        """Yield (file index, role, memory-mapped rows) block by block."""
        for index in sorted(self.files):
            done = self.files[index]
            for block in done["blocks"]:
                path = self.directory / "features" / f"f{index:03d}" / block["name"]
                if verify and _sha256(path) != block["sha256"]:
                    raise CacheMismatch(f"{path}: block content changed")
                rows = np.load(path, mmap_mode="r")
                if rows.dtype != FEATURE_DTYPE or len(rows) != block["rows"]:
                    raise CacheMismatch(f"{path}: unexpected block layout")
                yield index, done["role"], rows


def describe_hypothesis(hypothesis):
    """Stable description of every geometry input of a ``GeometryHypothesis``."""
    frames = {str(sm): repr(frame) for sm, frame in sorted(hypothesis.frames.items())}
    return {"name": hypothesis.name, "frames_sha256": _digest(frames), "source": repr(hypothesis.source),
            "opposition_tolerance_deg": hypothesis.opposition_tolerance_deg}


def canonical_observations(rows, result, role):
    """Accepted rows of one geometry result, lower TIME channel first as ``i``."""
    accepted = result.codes == 0
    rows = rows[accepted]
    endpoints, tof = result.endpoints_mm[accepted], result.shell_tof_ps[accepted]
    swap = rows["time_channel_1"] > rows["time_channel_2"]
    out = np.empty(len(rows), OBSERVATION_DTYPE)
    out["file"], out["record"], out["role"] = rows["file"], rows["record"], ROLES.index(role)
    out["swapped"], out["module_tie"] = swap, rows["module_tie"]
    for i_name, j_name, first, second in (("channel_i", "channel_j", "time_channel_1", "time_channel_2"),
                                          ("sm_i", "sm_j", "sm_1", "sm_2"), ("mm_i", "mm_j", "mm_1", "mm_2"),
                                          ("slab_i", "slab_j", "slab_1", "slab_2"),
                                          ("energy_kev_i", "energy_kev_j", "energy_kev_1", "energy_kev_2")):
        out[i_name] = np.where(swap, rows[second], rows[first])
        out[j_name] = np.where(swap, rows[first], rows[second])
    out["dt_ps"] = np.where(swap, -rows["dt_ps"], rows["dt_ps"])
    out["endpoint_i_mm"] = np.where(swap[:, None], endpoints[:, 3:], endpoints[:, :3])
    out["endpoint_j_mm"] = np.where(swap[:, None], endpoints[:, :3], endpoints[:, 3:])
    # Reversing the LOR maps TOF t -> -t and reverses segment order.
    out["shell_tof_ps"] = np.where(swap[:, None], -tof[:, ::-1], tof)
    return out


def pair_partition(channel_i, channel_j, partitions):
    return ((channel_i.astype(np.int64) * 1_000_003 + channel_j) % partitions).astype(np.int64)


def _pair_key(observations):
    return ((observations["channel_i"].astype(np.int64) << 26) | (observations["channel_j"].astype(np.int64) << 2)
            | observations["role"].astype(np.int64))


def _merge_counts(keys, counts, new_keys, new_counts):
    keys = np.concatenate([keys, new_keys])
    counts = np.concatenate([counts, new_counts])
    unique, inverse = np.unique(keys, return_inverse=True)
    merged = np.zeros(len(unique), np.uint64)
    np.add.at(merged, inverse, counts)
    return unique, merged


def build_observations(feature_cache, hypothesis, directory, *, partitions=16, buffer_rows=250_000,
                       slice_rows=250_000):
    """Level 2 for one hypothesis from cached features only (zero LDAT reads).

    At most ``buffer_rows`` observations are held before the largest partition
    buffer is spilled; pair counts are merged per partition (memory grows with
    distinct pairs, not with events). Built in ``<directory>.partial`` and
    renamed into place when complete.
    """
    feature_cache.require_complete()
    directory = Path(directory)
    fingerprint = {"schema_version": SCHEMA_VERSION, "kind": "cornell timing observations",
                   "features_sha256": feature_cache.manifest["fingerprint_sha256"],
                   "hypothesis": describe_hypothesis(hypothesis), "partitions": partitions,
                   "sources_sha256": {name: _sha256(module.__file__) for name, module in
                                      (("timing_ingest", timing_ingest), ("timing_geometry", timing_geometry))},
                   "observation_dtype": OBSERVATION_DTYPE.descr}
    fingerprint = _plain(fingerprint)
    if directory.exists():
        existing = ObservationSet(directory)
        if existing.manifest["fingerprint"] != fingerprint:
            raise CacheMismatch(f"{directory}: observation set from different inputs; use a new directory")
        return existing
    work = directory.with_name(directory.name + ".partial")
    if work.exists():
        manifest = work / "partial.json"
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != fingerprint:
            raise CacheMismatch(f"{work}: unrelated partial directory")
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "partial.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    buffers = [[] for _ in range(partitions)]
    sizes = np.zeros(partitions, np.int64)
    chunks = [0] * partitions
    keys = [np.empty(0, np.int64) for _ in range(partitions)]
    counts = [np.empty(0, np.uint64) for _ in range(partitions)]
    geometry = {role: dict.fromkeys((timing_ingest.ACCEPTED,) + timing_ingest.GEOMETRY_REJECTIONS, 0)
                for role in ROLES}

    def spill(p):
        rows = np.concatenate(buffers[p])
        np.save(work / f"p{p:03d}_c{chunks[p]:05d}.npy", rows)
        found, n = np.unique(_pair_key(rows), return_counts=True)
        keys[p], counts[p] = _merge_counts(keys[p], counts[p], found, n.astype(np.uint64))
        chunks[p] += 1
        buffers[p], sizes[p] = [], 0

    for _, role, block in feature_cache.blocks():
        for start in range(0, len(block), slice_rows):
            rows = np.array(block[start:start + slice_rows])
            result = timing_ingest.apply_geometry(_Columns(rows), hypothesis)
            for label, n in result.counts().items():
                geometry[role][label] += n
            observations = canonical_observations(rows, result, role)
            part = pair_partition(observations["channel_i"], observations["channel_j"], partitions)
            for p in np.unique(part).tolist():
                selected = observations[part == p]
                buffers[p].append(selected)
                sizes[p] += len(selected)
            while sizes.sum() > buffer_rows:
                spill(int(np.argmax(sizes)))
    for p in range(partitions):
        if sizes[p]:
            spill(p)
    pair_counts = []
    for p in range(partitions):
        table = np.empty(len(keys[p]), PAIR_COUNT_DTYPE)
        table["channel_i"], table["channel_j"] = keys[p] >> 26, (keys[p] >> 2) & ((1 << 24) - 1)
        table["role"], table["count"] = keys[p] & 3, counts[p]
        pair_counts.append(table)
    pair_counts = np.concatenate(pair_counts)
    np.save(work / "pair_counts.npy", pair_counts[np.lexsort((pair_counts["role"], pair_counts["channel_j"],
                                                             pair_counts["channel_i"]))])
    manifest = {"fingerprint": fingerprint, "chunks": chunks, "geometry": geometry,
                "observations": {role: int(pair_counts["count"][pair_counts["role"] == k].sum())
                                 for k, role in enumerate(ROLES)},
                "first_failure": _sum_first_failure(feature_cache)}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (work / "partial.json").unlink()
    os.rename(work, directory)
    return ObservationSet(directory)


def _sum_first_failure(feature_cache):
    totals = {role: {} for role in ROLES}
    for done in feature_cache.files.values():
        for label, n in done["first_failure"].items():
            totals[done["role"]][label] = totals[done["role"]].get(label, 0) + n
    return totals


class ObservationSet:
    """Completed level-2 rows; T6 loads one pair partition at a time."""

    def __init__(self, directory):
        self.directory = Path(directory)
        manifest = self.directory / "manifest.json"
        if not manifest.exists():
            raise CacheMismatch(f"{self.directory}: observation set incomplete")
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))

    @property
    def partitions(self):
        return self.manifest["fingerprint"]["partitions"]

    def partition(self, p):
        chunks = [np.load(self.directory / f"p{p:03d}_c{c:05d}.npy") for c in range(self.manifest["chunks"][p])]
        return np.concatenate(chunks) if chunks else np.empty(0, OBSERVATION_DTYPE)

    def pair_counts(self):
        return np.load(self.directory / "pair_counts.npy")
