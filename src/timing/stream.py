"""Streamed observation sets: LDAT -> selection -> geometry -> observations, no feature cache (spec 011 T8, FR-6).

For multi-TB inputs the level-1 feature cache (~0.1 byte per LDAT byte) is
optional. Here each input file is read once by one worker, through a feature
adapter's batches and a geometry adapter, into canonical observation chunks
partitioned by channel pair (``src.timing.cache`` rows and rules). A file's
output directory is renamed into place only when complete, so a restart redoes
interrupted files only. When every file is done, the per-file pair counts are
merged and ``manifest.json`` lists each partition's chunk files; the result
opens as an ordinary ``ObservationSet``.

The fingerprint holds everything a feature cache plus ``build_observations``
would (inputs, adapter fingerprint, holdout partitions, seed, prefix, geometry
description and sources, partitions, schema); a directory with a different
fingerprint is refused, never overwritten.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

import numpy as np

from src.timing.cache import (FEATURE_DTYPE, PAIR_COUNT_DTYPE, ROLES, SCHEMA_VERSION, CacheMismatch, ObservationSet,
                              _Columns, _digest, _merge_counts, _pair_key, _plain, _sha256, canonical_observations,
                              feature_rows, observation_dtype, pair_partition, walk_fingerprint)
from src.timing.kernels import ACCEPTED, Partition
from src.timing.shell import GEOMETRY_REJECTIONS


def _stream_file(directory, item, adapter, geometry, partition, seed, max_records, partitions, buffer_rows,
                 carry_walk=False):
    """Worker: one input file into ``files/fNNN``; returns its small ``done`` record."""
    partition = Partition(**partition)
    index, path = item["index"], Path(item["path"])
    final = Path(directory) / "files" / f"f{index:03d}"
    work = final.with_name(final.name + ".partial")
    if work.exists():
        shutil.rmtree(work)            # this set's own interrupted file
    work.mkdir(parents=True)
    before = path.stat()
    if (before.st_size, before.st_mtime_ns) != (item["size_bytes"], item["mtime_ns"]):
        raise CacheMismatch(f"{path.name}: input identity changed since the set was created")
    first_failure, geometry_counts, records = {}, dict.fromkeys((ACCEPTED,) + GEOMETRY_REJECTIONS, 0), 0
    buffers = [[] for _ in range(partitions)]
    sizes = np.zeros(partitions, np.int64)
    chunks = [[] for _ in range(partitions)]
    keys, counts = np.empty(0, np.int64), np.empty(0, np.uint64)

    def spill(p):
        nonlocal keys, counts
        rows = np.concatenate(buffers[p])
        name = f"p{p:03d}_c{len(chunks[p]):05d}.npy"
        np.save(work / name, rows)
        chunks[p].append(name)
        found, n = np.unique(_pair_key(rows), return_counts=True)
        keys, counts = _merge_counts(keys, counts, found, n.astype(np.uint64))
        buffers[p], sizes[p] = [], 0

    for features in adapter.feature_batches(path, partition, max_records=max_records, seed=seed):
        records += len(features.codes)
        codes, found = np.unique(features.codes, return_counts=True)
        for code, n in zip(codes.tolist(), found.tolist()):
            label = adapter.code_label(code)
            first_failure[label] = first_failure.get(label, 0) + n
        if not len(features.ints):
            continue
        rows = feature_rows(features, index)
        result = geometry.apply(_Columns(rows))
        for label, n in result.counts().items():
            geometry_counts[label] += n
        observations = canonical_observations(rows, result, partition.role, carry_walk)
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
    table = np.empty(len(keys), PAIR_COUNT_DTYPE)
    table["channel_i"], table["channel_j"] = keys >> 26, (keys >> 2) & ((1 << 24) - 1)
    table["role"], table["count"] = keys & 3, counts
    np.save(work / "pair_counts.npy", table)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise CacheMismatch(f"{path.name}: input changed while it was read")
    done = {"file": index, "name": path.name, "role": partition.role, "records": records,
            "first_failure": first_failure, "geometry": geometry_counts, "chunks": chunks,
            "observations": int(counts.sum())}
    (work / "done.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
    os.rename(work, final)
    return done


def stream_fingerprint(items, adapter, geometry, partitions_by_file, *, seed, max_records, partitions,
                       carry_walk=False):
    features = {"schema_version": SCHEMA_VERSION, "kind": adapter.kind,
                "inputs": [{k: item[k] for k in ("index", "name", "size_bytes", "mtime_ns")} for item in items],
                **adapter.fingerprint(),
                "partitions": {str(k): v for k, v in partitions_by_file.items()}, "seed": seed,
                "max_records": max_records, "feature_dtype": FEATURE_DTYPE.descr}
    return _plain(walk_fingerprint({
        "schema_version": SCHEMA_VERSION, "kind": geometry.kind + " (streamed)", "features": features,
        "features_sha256": _digest(_plain(features)), "hypothesis": geometry.describe(), "partitions": partitions,
        "sources_sha256": {name: _sha256(path) for name, path in geometry.sources().items()},
        "observation_dtype": observation_dtype(carry_walk).descr, "walk": None}, carry_walk))


def stream_observations(directory, inputs, adapter, geometry, *, holdout, energy_calibration_prefix, seed=0,
                        max_records=None, workers=1, partitions=16, buffer_rows=250_000, progress=None,
                        carry_walk=False):
    """Build (or resume) a streamed observation set; returns the completed ``ObservationSet``.

    ``progress(done_record)`` is called in the parent as each file completes. ``carry_walk`` keeps the raw
    walk ingredients in the rows (spec 012 FR-4).
    """
    directory = Path(directory)
    items = []
    for item in sorted(inputs, key=lambda value: value["index"]):
        stat = Path(item["path"]).stat()
        items.append({"index": int(item["index"]), "path": str(item["path"]), "name": Path(item["path"]).name,
                      "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    roles = {item["index"]: asdict(Partition.from_holdout(holdout, item["index"], energy_calibration_prefix))
             for item in items}
    fingerprint = stream_fingerprint(items, adapter, geometry, roles, seed=seed, max_records=max_records,
                                     partitions=partitions, carry_walk=carry_walk)
    marker = directory / "fingerprint.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != fingerprint:
            raise CacheMismatch(f"{directory}: streamed set from different inputs; use a new directory")
        if (directory / "manifest.json").exists():
            return ObservationSet(directory)
    elif directory.exists() and any(directory.iterdir()):
        raise CacheMismatch(f"{directory}: not empty and not a streamed observation set")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        with open(marker, "x", encoding="utf-8") as handle:
            json.dump(fingerprint, handle, indent=1)
    done = {}
    for item in items:
        record = directory / "files" / f"f{item['index']:03d}" / "done.json"
        if record.exists():
            done[item["index"]] = json.loads(record.read_text(encoding="utf-8"))
    pending = [item for item in items if item["index"] not in done]
    arguments = [(str(directory), item, adapter, geometry, roles[item["index"]], seed, max_records, partitions,
                  buffer_rows, carry_walk) for item in pending]
    if workers > 1 and len(pending) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            for result in pool.map(_stream_file, *zip(*arguments)):
                done[result["file"]] = result
                if progress:
                    progress(result)
    else:
        for argument in arguments:
            result = _stream_file(*argument)
            done[result["file"]] = result
            if progress:
                progress(result)
    return _finish(directory, fingerprint, [done[item["index"]] for item in items], partitions)


def _finish(directory, fingerprint, done, partitions):
    keys, counts = np.empty(0, np.int64), np.empty(0, np.uint64)
    chunk_paths = [[] for _ in range(partitions)]
    first_failure = {role: {} for role in ROLES}
    geometry = {role: dict.fromkeys((ACCEPTED,) + GEOMETRY_REJECTIONS, 0) for role in ROLES}
    for record in done:
        folder = f"files/f{record['file']:03d}"
        for p in range(partitions):
            chunk_paths[p] += [f"{folder}/{name}" for name in record["chunks"][p]]
        table = np.load(directory / folder / "pair_counts.npy")
        found = ((table["channel_i"].astype(np.int64) << 26) | (table["channel_j"].astype(np.int64) << 2)
                 | table["role"].astype(np.int64))
        keys, counts = _merge_counts(keys, counts, found, table["count"])
        for label, n in record["first_failure"].items():
            first_failure[record["role"]][label] = first_failure[record["role"]].get(label, 0) + n
        for label, n in record["geometry"].items():
            geometry[record["role"]][label] += n
    table = np.empty(len(keys), PAIR_COUNT_DTYPE)
    table["channel_i"], table["channel_j"] = keys >> 26, (keys >> 2) & ((1 << 24) - 1)
    table["role"], table["count"] = keys & 3, counts
    np.save(directory / "pair_counts.npy", table[np.lexsort((table["role"], table["channel_j"], table["channel_i"]))])
    manifest = {"fingerprint": fingerprint, "chunk_paths": chunk_paths,
                "chunks": [len(paths) for paths in chunk_paths], "geometry": geometry, "walk": None,
                "observations": {role: int(table["count"][table["role"] == k].sum()) for k, role in enumerate(ROLES)},
                "first_failure": first_failure, "files": [{k: r[k] for k in ("file", "name", "role", "records",
                                                                            "observations")} for r in done]}
    partial = directory / "manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(partial, directory / "manifest.json")
    return ObservationSet(directory)
