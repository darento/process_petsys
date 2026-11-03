# Spec 015 — IMAS fast LDAT to LM with the delivered timing tables

Status: `in progress` (2026-11-03: T1–T7 done and validated on this PC, IMAS PC run T8 pending; approved 2026-11-03, owner; clarified 2026-11-03 from owner answers to two question rounds recorded below; drafted 2026-11-03)

Constitution: [AGENTS.md](../../AGENTS.md). Workflow: [docs/prompts.md](../../docs/prompts.md). Builds on shipped [spec 011](../011-imas-skew-calibration/spec.md) (IMAS setup, selection kernels, skew LUT), [spec 012](../012-imas-timing-walk/spec.md) (walk LUT, `correct_dt`) and [spec 014](../014-imas-top-earliest-timestamp/spec.md) (side timestamp rule, EM4 + walk_rule4). Reuses the spec 002 compact reader (`src/ldat_stream.py`).

## Goal and population

`scripts_imas/imas_listmode.py` (untracked) converts IMAS compact LDAT to listmode (`.lm`) in pure Python, one record at a time, with a Keras NN for the axial position and the DOI. It writes `dt = -32700` for every event (its skew map is empty). It is slow, and the timing tables delivered by specs 011–014 are not applied.

**Goal:** a tracked, fast converter, run as a docopt script like the IMAS scripts, that writes the same LM V5 records with the delivered timing correction, and processes several LDAT files in parallel.

- **Input:** PETsys compact **coincidence** LDAT (two detectors per record, never singles), IMAS 2DAQ (`configs/imas_2DAQ.yaml`, `maps/imas2DAQ_map.yaml`). Hit charges in PETsys QDC units (a.u.), timestamps in ps.
- **Calibrations:** the LM energy uses `SLAB_EN_MAP` (`ID mu` per TIME channel, keV = 511 / mu × minimodule ENERGY sum). The walk's keV input uses the walk tables' own calibration (`encal_skew_run7.tsv`).
- **Timing tables (default):** EM4 skew + walk_rule4 under `top_earliest_mean` (spec 014; 22Na 389.9 ps).
- **Positions:** the combined Y + DOI model of `imas_listmode.py` (`src.slab_nn`, `IMASNoCatY.h5`, `IMASNoCatDOI.h5`, categories 0). The DOI output is the NN's value, not a validated depth in mm.
- **Representative data (this PC):** `PETSYS_DATA_DIR/IMAS/20261019_Na22_*` splits; NN models and pair file in `PETSYS_DATA_DIR/IMAS/LUTs/`.

## Requirements

- **FR-1 — Command line.** `python -m src.imas.listmode` and a shim `scripts_imas/imas_ldat2lm.py` SHALL parse one docopt usage: positional `YAMLCONF SLAB_EN_MAP INFILES...`, and options for the output directory, skew LUT, walk LUT, walk calibration, timestamp rule, skew-only legacy file, no timing, NN Y and DOI models, pair file, workers, LM header acquisition time / activity / isotope, debug, continue, keep segments and a per-file record prefix. Defaults SHALL be the IMAS PC paths of the legacy script (NN, pair file, LM directory) and of the delivered package. Inputs are natural-sorted. A missing or unreadable input, config, calibration, table or model SHALL stop the run before any file is processed, naming it.
- **FR-2 — Bounded fast reading.** Each file SHALL be streamed in bounded record batches (`LdatStream`) and selected by compiled kernels; memory SHALL not grow with file length. Truncated or malformed input SHALL fail that file with its byte offset, never write a partial segment as complete.
- **FR-3 — Selection.** Per record, in order, the first failing cut SHALL be counted:
  1. a hit with charge ≥ `en_min_ch` on an unmapped channel;
  2. either side fails `min_ch` (sum rows/cols rule: ENERGY hits ≥ `min_ch` and fewer than all hits);
  3. no minimodule with positive ENERGY sum (highest sum; first seen on ties, counted);
  4. the selected minimodule fails `min_ch`;
  5. the top TIME channel (highest charge; the slab) has no / an invalid `SLAB_EN_MAP` mu;
  6. either side's keV outside the open `energy_range` of `YAMLCONF`;
  7. the SuperModule pair is not in the pair file (either order).
  Hits below `en_min_ch` are dropped before all cuts (owner 2026-11-03: same floor as the timing tables and the LDAT Inspector; the legacy read every hit). There is no coincidence-window cut (the legacy has none).
- **FR-4 — Timing.** `dt` (ps, int16) SHALL be side 1 minus side 2 after FR-6 ordering, corrected by `src.imas.skew.walk.correct_dt` under the tables' rule, with the walk keV from the walk calibration. The skew LUT, walk LUT and rule SHALL be checked together (map SHA-256, skew/walk pairing, rule fitted equals rule applied); a mismatch refuses the run. An event without both delays or without every walk input gets `dt = -32700` and is counted, as are clamped walk inputs and the events where the rule acts. Alternatives: a skew-only legacy `ch skew` file (`t + skew`, rule `top`, no walk), or no timing (every `dt = -32700`, as the legacy today). A corrected `dt` outside int16 SHALL be written as `-32700` and counted.
- **FR-5 — Positions.** Each accepted side SHALL feed the NN its 8 ENERGY charges indexed by the map row (`local_map[ch][2]`) of the selected minimodule, zero where a row did not fire, with categories 0. Local x = the top TIME channel's map x; local y = NN y + the top TIME channel's map y. Pixels: index = `searchsorted(edges, v, "right") - 1` over 101 edges from 0 to 103.22 mm; z = the NN DOI truncated to int8. *(Fixes legacy bug (a): the legacy reused stale charges of an earlier event for rows that did not fire.)*
- **FR-6 — Records.** One `CoincidenceV5` record per accepted coincidence *(fixes legacy bug (b): the last coincidence of every full bunch was dropped)*: `time` = round((min of the raw top timestamps − t0) · 1e-12, 4) s, t0 = min of the first hit of each side of the first record of the first input file (all hits); `energy1/2` = round(keV) from `SLAB_EN_MAP`; `amount` = 1.0; `pair` from the pair file; sides swapped (energy, position, `dt` sign) when the pair is listed in the other order. Records follow input natural order, then record order.
- **FR-7 — Parallel files.** N worker processes SHALL each process whole files from one queue of all inputs; one NN process SHALL predict every chunk and write per-file segments. Queues SHALL be bounded. The merged `<base>_all.lm` (legacy name: first segment name without its split index, plus `_all.lm`) SHALL start with the legacy `LMHeader` fields, with acquisition time, activity and isotope from the options, followed by the segments in natural order. The `.lm` bytes SHALL not depend on N (CPU inference). Segments are deleted after a complete merge unless kept.
- **FR-8 — Continue.** With continue, a file whose segment and its record exist and match the input (path, size) and the run's settings fingerprint SHALL be skipped; others are reprocessed. Without continue, an existing output directory with segments SHALL be refused.
- **FR-9 — Provenance.** `<base>_all.json` SHALL record inputs with SHA-256 (config, map, `SLAB_EN_MAP`, skew, walk, walk calibration or legacy skew, NN models, pair file), cuts, rule, header values, worker count, per-file and total records read, FR-3 rejection counts, accepted coincidences, FR-4 timing status counts, elapsed time and throughput; `<base>_log.txt` as the legacy log. A run with any failed file SHALL not write `_all.lm`, SHALL name the failed files and exit non-zero.
- **FR-10 — Debug.** With debug, the run SHALL save the legacy plots from full counts: accepted-side keV histogram (1500 bins, 0–1500 keV) with its Gaussian fit, and hits per SuperModule quadrant (10 × 48).
- **FR-11 — Speed.** On one 22Na split, the run SHALL report records/s for read + select + timing and for the whole run, and the read + select + timing SHALL be at least 20× the legacy loop's record rate on the same file prefix and PC (recorded).

## Completion criteria

1. Synthetic LDAT tests: each FR-3 cut, FR-4 statuses (corrected, no delay, walk missing, int16 overflow), FR-5 Esignals and pixels with a fake predictor, FR-6 swap and time, FR-7 worker independence and merge, FR-8 skip, FR-9 failure handling.
2. Parity (real data, first records of a 22Na split): records equal a reference computed from the legacy loop with the floor and the two bug fixes and the listmode_skew_walk.md timing class, frozen as a golden file with provenance. `dt` within 1 ps (float rounding of absolute timestamps), all other fields equal.
3. FR-11 speed recorded on this PC; a full 22Na split converted.

## Out of scope

The NN_independent variant (per-category models; `src.slab_nn_NN_independent` is not in the repo), COG positions, category maps, temperature compensation, event subsampling, GUI or Manager integration, Cornell, GPU determinism (recorded, not guaranteed).

## Clarifications (owner, 2026-11-03)

1. Positions: `imas_listmode.py` combined NN.
2. Timing default: EM4 + walk_rule4; options for EM2 + walk3, skew-only and none.
3. Hit floor: `en_min_ch` everywhere.
4. Parallelism: N workers + one NN process.
5. Inputs: docopt options with defaults, recorded in provenance.
6. Legacy bugs (a) and (b): fixed.
7. Validation data: NN models and pair file are in `PETSYS_DATA_DIR/IMAS/LUTs/`.
8. Run shape: one pool over all files, merged `_all.lm`, continue skips completed segments.
