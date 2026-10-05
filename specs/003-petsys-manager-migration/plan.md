# Plan 003 — PETsys Manager migration

Spec: [`spec.md`](spec.md), owner-approved 2026-09-30. Constitution: [`AGENTS.md`](../../AGENTS.md). Implement one named task from [`tasks.md`](tasks.md) at a time; do not modify the sibling repository or user-owned inputs. Architecture below is planned; only verified tasks describe implemented capabilities.

## Direction

- Separate Linux manager, retaining the five Cornell workflow areas and command log. LDAT Inspector remains independent and unchanged.
- Migrate runtime processing into tracked modules, not a copied monolithic GUI that still shells out to ignored scripts.
- First establish reference behavior; then settings, command/file safety and headless processing; wire the UI only after these checks pass.
- No hardware commands, acquisitions, daemon cleanup, builds or external launch changes as a side effect of development or test discovery.

**T2 outcome (2026-09-30):** additive `src/petsys_manager/` package now provides immutable contracts, versioned machine profiles and read-only action prerequisites. **35/35** synthetic settings checks and compile/import guards pass; evidence in [`tasks.md`](tasks.md). Existing flat `src` modules/imports remain unchanged. Numerical schema validation, live readiness, execution and GUI integration are still future tasks; preflight success alone authorizes none of them.

**T3 outcome (2026-09-30):** pure command builders and blocking, toolkit-free runner implemented; **35/35** command/runner checks, **70/70** combined T2/T3 checks and **5/5** actual WSL Linux process checks pass. Native dummy children exercise real pipe flooding/cancellation; production execution remains Linux-only. WSL's standard-library-only process checks use system Python 3.12.3 because Linux conda is not installed; this is not the Cornell deployment/environment acceptance. No hardware or numerical workflows executed. Output validation is explicit: exit zero alone cannot advance; returned `can_advance` plus workflow cancellation is authoritative, not a completion event. Runtime reaps its direct child; Linux reaps orphaned killed descendants, with a check-only temporary subreaper pinning descendant reaping. Evidence/reproducible commands in [`tasks.md`](tasks.md).

## Architecture

**T4 outcome (2026-09-30):** exclusive run/stage/attempt storage and immutable manifest revisions implemented in `artifacts.py`. Final combined checks: **106 passed / 1 Linux-only skip**; standalone WSL filesystem checks: **37/37 passed**. Windows fsyncs files; Linux additionally fsyncs directory entries. Atomic publication uses same-filesystem hard links and fails closed if unsupported; `.partial` data is retained. Manifest readers stream revision discovery with a 4 MiB JSON bound; inventories cap at 10,000 entries. Success still requires later stage-specific numerical validation. Evidence in [`tasks.md`](tasks.md).

**T5 outcome (2026-09-30):** `src/cornell/inputs.py` implements exact typed selection, bounded full LDAT validation, processing/selected-map parsing and sparse immutable limits/position-calibration contracts. **38/38** fixture checks plus **59/59 Inspector Cornell/IMAS**, **16/16 slab** and **6/6 unpopulated-map** regressions pass. Existing readers, mapping and Inspector remain unchanged. Full scans count input records/sides/hits before cuts; neither format nor population is inferred from the extension or claimed proven when bytes are ambiguous. Settings-to-workflow integration and numerical processing remain later tasks. Evidence/bounds in [`tasks.md`](tasks.md).

**T6 outcome (2026-09-30):** `acquisition.py` now owns DAQD lifetime and initialization validity. READY requires the owned child to answer daqd's shared-memory-name query (command `0x02`, the `daqd.Connection()` handshake), not just a socket file. An existing socket or shared memory blocks launch, because the inspected daqd `shm_unlink`s an existing name after its own `O_EXCL` create fails. Initialization is bound to the daemon generation, argv and INI digest. **21/21** fake and **7/7** WSL dummy-socket checks pass. Installed-tool behavior stays pending until T19; evidence in [`tasks.md`](tasks.md).

**T7 outcome (2026-09-30):** `AcquisitionService` runs monitored attempts in worker threads, each in a new run-store attempt directory, with the reference startup/growth/loss defaults. It retries only stall, no-data and frame-loss symptoms, never a nonzero exit or STOP. Missing loss telemetry stays `unknown`, and an unreached growth window is recorded as `not_exercised`. **21/21** fake-clock and **4/4** WSL real-process acquisition checks pass. Installed-tool output and loss formats stay pending until T19; evidence in [`tasks.md`](tasks.md).

| Planned location | Responsibility |
| --- | --- |
| `exe_programs/PETsysManager.py` | Thin entry point, repository import path, `freeze_support`, manager startup |
| `exe_programs/petsys_manager_gui.py` | CustomTkinter views, dialogs and main-thread queue polling; no numerical algorithms or subprocess ownership |
| `src/petsys_manager/__init__.py` | Side-effect-free package |
| `src/petsys_manager/settings.py` | Versioned machine profile, immutable validated run settings, action-specific prerequisites |
| `src/petsys_manager/contracts.py` | Typed command/result/event, input descriptor, artifact and run/attempt identifiers |
| `src/petsys_manager/commands.py` | Pure argv/cwd builders for DAQD, initialization, acquisition, conversion and internal processing |
| `src/petsys_manager/runner.py` | Linux subprocess groups, bounded output streaming, termination/reaping; injectable process/clock interfaces |
| `src/petsys_manager/artifacts.py` | Exclusive run directories, exact input/output inventory, manifests and scoped cleanup |
| `src/petsys_manager/acquisition.py` | DAQD ownership/readiness, initialization validity, acquisition monitoring and bounded retries |
| `src/petsys_manager/workflow.py` | Single-workflow coordinator, settings snapshot, stage dependencies, cancellation and result validation |
| `src/cornell/__init__.py` | Side-effect-free package |
| `src/cornell/inputs.py` | Format/population validation, processing-config/map resolution, limits and calibration contracts |
| `src/cornell/calibration.py` | Tracked fixed-position calibration extracted from the reference script |
| `src/cornell/listmode.py` | Tracked fixed-position LM processing; reuse `src/listmode.py` binary structures |
| `src/cornell/qc.py` | Tracked compact-coincidence QC extraction/counting/fits |
| `src/cornell/qc_report.py` | Existing QC report/output types with explicit scope, units and provenance |
| `src/cornell/cli.py` | Headless calibration/listmode/QC dispatch, request loading, progress/result emission |

Reuse `src/read_fixed.py`, `src/read_compact.py`, mapping, scalar/vectorized detector helpers, `src/fits.py` and `KevConverter` where their validated behavior fits. Prefer manager-specific validation wrappers over changing shared Inspector behavior. A necessary shared fix needs its own reproducible regression check and existing Inspector checks.

### Boundaries and data shapes

- `MachineProfile`: tool root, INI/YAML paths, configured processing root for relative map paths, cards/type/socket/shared-memory resource settings, destinations, limits/pair/region maps, acquisition safety settings, worker/batch bounds and explicit LM metadata/profile. Save a versioned YAML under the user's Linux application-config directory; allow an explicit profile path. Never rewrite the selected processing YAML or maps.
- `RunSettings`: frozen validated copy of the action-specific profile plus exact input descriptors, ordered files, source mode, format/population, duration, split/hit settings and processing options. Input files stay untouched; record their paths, sizes and relevant configuration/calibration digests. Do not hash whole large acquisitions on the UI thread.
- `InputDescriptor`: path, `fixed`/`compact`, `coincidence`/`group`, scope and validation status. Extension is not format metadata. User confirmation supplies descriptors for legacy files; converter-owned artifacts record them automatically.
- `CommandSpec`: argv list, absolute cwd, explicit environment, run/stage/attempt IDs. Internal stages run `[sys.executable, '-u', '-m', 'src.cornell.cli', ...]` from the discovered checkout; manager itself starts in the documented conda environment. No nested shell/conda activation strings.
- `CommandResult`: success/failed/cancelled/launch-error, exit code where known, bounded log tail, actual artifacts and validation/rejection counts. A zero exit is necessary, not sufficient, for stage success.
- `RunEvent`: run/stage/attempt ID, monotonically ordered event sequence, kind, message/progress/payload. The GUI drains a queue using `after` scheduled only on its main thread. Workers never call Tk APIs, including `after`.
- `RunManifest`: settings/provenance, exact inputs, stage outcomes, actual output paths, format/population, sample bounds, pair/side/hit denominators, calibration/fit status, partial/final state and owned disposable artifacts. Machine-specific manifests belong in selected output directories, not Git.
- Numerical outputs are batch arrays, bounded sampled columns, histogram/statistic summaries or streamed LM records. No unlimited event dictionaries or debug coordinate lists.

## Lifecycle and process safety

### Single foreground workflow

Coordinator states: `idle → validating → running → succeeded | failed | cancelled`. Stage states have the same terminal distinction. DAQD has a separate `off → starting → ready | failed → stopping → off` lifetime. Initialization is tied to the current daemon identity/configuration, not a checkbox; daemon death or changing hardware/INI settings invalidates it.

- Reserve the workflow before launching work. Only STOP, logs and harmless navigation remain active during conflicting operations.
- Preflight all known pipeline prerequisites before acquisition. Runtime artifact checks still guard every transition; missing output cannot trigger a later stage.
- Freeze settings for the workflow; manual edits affect the next run only. Pipelines never monkey-patch callbacks or temporarily override persistent output fields.
- Require a new run identity and a new acquisition-attempt identity. The coordinator ignores stale completion/monitor events; cancellation is sticky for that identity.
- Runner uses `Popen(argv, shell=False, cwd=..., start_new_session=True)` on Linux. Internal processing children/pools inherit the stage process group. Drain stdout/stderr without deadlock and keep only a bounded UI tail; a full log can stream to the owned run directory.
- STOP sets cancellation, invalidates scheduled retries, sends TERM to owned groups, escalates to KILL after a configured grace interval and reaps processes off the UI thread. Daemon death aborts dependent hardware work, not unrelated offline processing. Window close cancels foreground work and stops owned DAQD before completing shutdown; failure to reap is visible, never silently abandoned.
- Readiness requires a live owned process and the expected socket becoming usable; successful initialization provides protocol/tool evidence. A toggle or socket file existing by itself is not readiness. Confirm actual supported PETsys flags/readiness mechanism in T1/T6.
- T1 tool-source inspection shows local initialization/acquisition use a default daemon connection without a socket-selection flag. Confirm installed capabilities before enabling a custom socket; a DAQD socket flag alone is insufficient. Local DAQD creates its socket before initializing cards/shared memory, so readiness must not rely on file existence.
- T3 builders do not invent an `init_system --config` flag: the inspected initializer accepts no INI argument; acquisition/conversion actually load the selected INI. Custom-socket init/acquisition argv remains unsupported until the installed tool contract is recorded, even if a profile capability boolean is set. DAQD's shared-memory name is hardcoded by the inspected source; a custom selection is rejected rather than silently ignored.
- Reject pre-existing socket/shared-memory conflicts. Do not delete global resources on startup/close. Any future stale-resource removal requires explicit operator resolution and verified ownership; it is not an automatic migration feature.

### Acquisition monitoring

Preserve reference defaults as editable safety settings: startup timeout 45 s, growth window 20 s, poll interval 5 s, minimum growth 20 MB, maximum reported frame loss 5%, maximum attempts 3, retry delay 2 s. These are acquisition safety policy, not detector QC thresholds. Use monotonic timing and cancellation-aware waits, never main-thread sleeps.

Monitor output growth for the active attempt. Completion requires successful exit plus nonempty expected RAW artifacts; reported frame loss over threshold can retry. Missing loss reporting remains `unknown` and is logged. Each attempt uses a distinct basename/directory, retaining failed attempts. Short acquisitions may complete before a growth window; record that the growth check was not exercised rather than falsely claiming it passed.

Operator feedback (FR-20): the monitor publishes `growth_started`, `growth_passed` and, at every poll after the first data, `rawf_progress` (size, bytes/s since the previous poll, `growing`). A stall after the check passed is a visible warning, not an abort: the reference has no such rule. Safety limits live in the profile (`safety:`), are editable in the setup UI (T14) and are recorded with the run settings.

SiPM bias (FR-19): `acquire_sipm_data` switches bias off only at the end of a normal run, and TERM ends the Python tool without that step. After an attempt whose child was launched but did not exit 0 by itself, the service runs the inspected `set_bias --power off` (same default DAQD connection; bounded by a timeout, not cancelled by STOP) and records `bias_off`. Failure gives `bias.state = unknown`, a `bias_unknown` warning event and no further attempt. Window close must let this finish before stopping DAQD (T14). `set_bias` becomes a required tool for acquiring actions.

## Artifact and configuration policy

- Create a unique run directory exclusively under the operator-selected destination, with stage/attempt subdirectories; a collision is an error or causes a new identifier. Writers use exclusive creation or publish within that owned directory without replacing existing user files.
- **Readable run folders (T29, FR-9/FR-13, 2026-10-05; replaces the `<action>-<stamp>-<hex>/<stage>/attempt-1/` layout for new runs).**
  - *Name.* `<data>_<action>[-<options>]_<YYYY-MM-DD>_<HHMM>` (local time). Action codes: `acq`, `conv`, `cal`, `lm`, `qc` (live and offline), `pipeline`. Options: calibration `P<n>-<target|reference>`, LM and pipeline `P<n>`, live QC `with-source`/`without-source`. Data: the RAW name (conversion); the acquisition name (acquisition, pipeline, live QC: `RunOptions.acquisition_name`, default `acquisition`, also the RAW basename; live QC keeps its `_qc_<mode>_source` RAW suffix); otherwise the common base of the input stems: their longest common prefix, cut back to a `_`/`-` boundary when the stems differ, a trailing converter suffix (`_coincCompact`, `_coincFixed`, `_groupCompact`, `_groupFixed`) and trailing `_`/`-` removed, `data` if empty. Characters outside `[A-Za-z0-9_-]` become `-`; the data part is cut to 60 characters. Examples: `run_0024_lm-P5_2026-10-04_0936`, `20260930_F18_950uCi_Run1_60s_cal-P5-target_2026-10-05_0533`.
  - *Allocation.* `RunStore.reserve(..., name=…, stages=…)` creates `<name>` exclusively, else `<name>_2` … `<name>_99`, then refuses; the folder name is the run id. An existing folder is never adopted.
  - *Layout.* From the run's stage tuple: one stage → the run folder itself; several → `<i>_<stage>/`. Acquisition attempts → `attempt-N/` in their stage folder (one stage: in the run folder). Every other stage has one attempt (`attempt-1` in the record, no folder); a second attempt of such a stage is refused. LM keeps `lm-job.json` and `segments/`.
    ```
    <lm_dir>/run_0024_lm-P5_2026-10-04_0936/      run.json  .history/  request.json  result.json  run_all.lm  run_all.lm.json  lm-job.json  segments/
    <data_dir>/F18_Run1_pipeline-P5_2026-10-05_0533/  run.json  .history/  1_acquisition/attempt-1/  2_conversion/  3_calibration/  4_listmode/
    <data_dir>/runs.tsv
    ```
  - *Records.* Each commit links `.history/manifest-NNNNNN.json` exclusively from a flushed temporary in `.history/` (as T4), then replaces `run.json` with the same content (flushed temporary, `os.replace`). `.history/` is authoritative: `read_manifest` reads its latest revision; a pre-T29 folder (`manifest-NNNNNN.json` in its root) is still read. A failed `run.json` replacement fails the commit and removes the just-linked revision. `run.json`, `.history/` and the store's temporaries are reserved: no artifact may name them, and inventories and partial-output recording skip them.
  - *Overview.* After the run is finalized (any status), the coordinator appends one tab-separated line to `<destination>/runs.tsv` (header written when the file is created exclusively): finished (`YYYY-MM-DD HH:MM:SS`), run folder, action, inputs (RAW name, or `<n> file(s): <first>`), status, main output (relative to the run folder: RAW, first LDAT and count, `.encal`, `.lm`, QC PDF). Tabs and newlines in values become spaces. One `O_APPEND` write per line, flushed; a failure is logged and never changes the run's verdict.
  - *LM/QC in place.* The processing CLI gains `options.in_place` (listmode, QC): `outputs.directory` is then the caller's existing stage folder, created exclusively by the store, instead of a new directory. Every LM/QC file is still created exclusively (`xb`), so a name collision fails and nothing is replaced. An in-place LM resume checks only LM-named entries (job file, segments, merged output and its partials, sidecar, debug plots) and ignores the caller's files. QC `options.report_title` (the run folder name) titles the PDF; null keeps the directory name.
- T4 stores append-only `manifest-NNNNNN.json` revisions, atomically linked from flushed private metadata; only a fully persisted revision updates the in-memory snapshot. Run state stays `partial` while any work is unfinished. Failed publication/durability preserves earlier evidence; no arbitrary existing run is adopted. `read_manifest` is read-only; guarded resume remains T9. Ordered external inputs are recorded as explicit descriptors with sizes/mtime, without hashing large acquisitions.
- The converter output prefix is isolated to its run. Inventory files before/after execution in that directory, restrict discovery to that exact stage prefix, validate outputs and record the actual ordered set. Empty split files may be reported/ignored; deletion, if enabled, requires zero size and recorded disposable ownership. Do not delete `.lidx` merely because a wildcard matches.
- Conversion duration is explicit, not parsed from a basename. Validate positive duration, split count/hit limit and actual PETsys support. Never use `cpu_count() - n` without a safe lower bound; split count is distinct from worker count.
- Relative `map_file` resolution uses an explicit processing root, initially the discovered repo root for legacy `maps/...` values. Do not silently switch semantics to the YAML directory. Show resolved paths in preflight and record them. Absolute external paths remain valid.
- `configs/cornell_full_system_20260928.yaml` and its map are available local candidate validation inputs, not an auto-selected profile or tracked requirement. Their geometry comments are not authority for LM metadata; confirm the reconstruction profile explicitly.
- Assets, if retained, resolve relative to the module, not the current directory. An optional logo failing to load cannot block operations. Do not introduce an ignored top-level `gui/` runtime dependency.

## Format routes and stage contracts

| Workflow | Input | Output/consumer |
| --- | --- | --- |
| Manual coincidence conversion | RAW acquisition + selected INI | Compact coincidence descriptor (FR-10, 2026-10-05: fixed and group conversion no longer offered) |
| Energy calibration (FR-21) | Selected compact coincidence files (the library still reads fixed coincidence/group for existing files and checks), positions P, processing config; COG limits when P ≥ 2 | Per-slab (P = 1) or position (P ≥ 2) `.encal`, per-key status file, sidecar provenance, summary plot |
| Manual LM generation | Compact coincidence files with their hit limit (FR-22; fixed only in the library), per-slab or position calibration, COG/DOI limits, pair/region maps, metadata | Compatible LM header/records plus provenance/debug summaries |
| Complete pipeline | Acquire → compact coincidence conversion (FR-22) → position calibration → LM | Actual artifacts from each successful predecessor |
| QC | 60 s with-source or 180 s without-source acquisition → compact coincidence conversion → legacy QC | Existing QC output types with actual results directory |

Full structural validation is fused with bounded reading and occurs before publishing final numerical output. Fixed validation checks the hit-limit header, supported record layout, remainder length, hit counts and mapped IDs; compact validation detects incomplete headers/hits and invalid/missing mapping. Group versus coincidence and arbitrary legacy format cannot always be inferred from bytes: explicit descriptors remain mandatory. Invalid files never yield an apparently successful partial result; retain owned partial artifacts as failed.

T5 provides a full read-only scan before numerical output publication; later processing must also reject files that change after validation and validate any streamed ingest before final success. Active fixed slots alone contribute counts; padded slots do not. Generic selected maps are parsed through the unchanged mapping generator using the validated in-memory YAML, avoiding a second unchecked read. Cornell slab workflows require the existing summed eight-position convention, not a fixed number of SuperModules/channels.

Position `.encal` metadata is explicit: header region count is mandatory; absent exact boundaries remain `None`/unavailable. An operator may supply boundaries, or explicitly select a JSON sidecar with `schema_version: 1`, `num_regions`, `region_boundaries` and `calibration_sha256`; all must agree with the selected calibration and settings. T8 writers will add fit/cut/status provenance using this sidecar contract without changing `.encal` keys. Sparse missing limits/factors remain missing entries for later rejection accounting, never zero-filled measured results. LM pair/region lookup semantics remain T9.

## Numerical migration

### Reference gate

T1 inventories/hashes the three local reference scripts and relevant shared helpers without running their CLI blindly. Operator confirmation is needed that these are the Cornell-installed versions. Capture tool flags/version, slab convention, cut behavior, baseline fixture results, LM schema and metadata. Changes from already-swapped historical slab calibrations are a provenance issue, not a reason to silently change current slab rules.

**T1 outcome (2026-09-30):** [`reference.md`](reference.md) records 17/17 passing synthetic contracts and 24 unchanged source fingerprints. Owner confirmed the three local processing scripts and modified converter fixed-output support. Sibling converter source is not that installed version. Exact tool provenance, installed helper parity and real data remain pending; LM profile/timestamp units additionally gate T9.

Exact counts, accepted/rejected selection, calibration keys and record layouts are parity oracles. Compare fitted values at the reference file's written precision (0.001 a.u. for position calibration); numerical tolerances for other quantities must be recorded before a task, not widened after failure. Controlled fixtures avoid ambiguous random slabs; seeded reference comparisons separately exercise randomness. LM records should be byte-identical on controlled fixtures; changed header values are allowed only where the supplied metadata intentionally replaces incorrect hardcoded reference values.

### Position calibration

- Extract pure region assignment, accumulation, fitting and writing; keep existing 100-bin 0–200 a.u. histogram, minimum 50 samples, fit call, five-region default and edge multiplier 1.8 unless confirmed reference differs.
- Preserve the current distinct boundary behavior: calibration excludes COG outside its limits; LM clips normalized COG to its region-assignment interval. Do not unify them as a cleanup refactor.
- The reference has a nominal 4 M accepted-side limit per file, checked between side chunks and able to overshoot. Record the exact stopping semantics, batch size and resulting sample, not an invented exact cap or pair denominator. Keep it configurable and bounded.
- Replace accumulating scalar energy lists with per-key fixed histograms and sufficient moments/counts for the existing mean/std fallback. Merge summaries in deterministic order; verify written values/fit selection and report fallback estimates in a sidecar without changing `.encal` keys/header compatibility. Sidecar records region boundaries and generation cuts.
- Do not silently add a channel-energy cut absent from the confirmed reference; report the actual calibration cut separately from LM/QC cuts.

**T8 outcome (2026-10-01):** `src/cornell/calibration.py` reproduces the reference's accepted sides, keys, histograms and fitted values exactly on seeded synthetic data. Fallbacks agree within 0.001 a.u. and are now labelled. Storage is fixed per mapped key. The `.encal` is unchanged and readable by `KevConverter`; the T5-compatible sidecar records boundaries, cuts, sampling, rejections and non-fitted keys. 14/14 checks pass; real-data parity is T19. Evidence in [`tasks.md`](tasks.md).

### Energy calibration (FR-21, T20)

Owner decision 2026-10-01: one algorithm for every input format, from `scripts_cornell/cornell_slab_en_cal.py`, plus a positions-per-slab count P.

- **Readers:** fixed and compact batches are decoded into the same padded per-side arrays (empty slots are channel −1; fixed slots beyond the header count are ignored). Hits below `en_min_ch` are emptied in place, keeping the order of the remaining hits, as the compact reader drops them. One vectorized selection then serves both formats, so equal events give equal results by construction.
- **Selection** mirrors the per-event reference:
  - both sides pass `filter_min_ch`;
  - the highest-energy minimodule is found (an exact energy tie falls back to the reference function, whose result depends on Python set order);
  - both maximum minimodules pass `filter_min_ch`;
  - `get_slab_cornell` is applied; an event with both slabs undetermined is skipped;
  - one-time-channel sides are dropped;
  - the side's key is its highest-energy time channel and slab (plus region when P ≥ 2).

  A side with no time channel in its maximum minimodule is a counted rejection; the reference aborts there. A group is calibrated as one side. A file stops once more than 10,000,000 events have passed, as the reference does.
- **Regions (P ≥ 2):** y COG of the maximum minimodule normalized by the selected COG limits; outside [0, 1] excluded (T8 calibration convention, edge multiplier 1.8).
- **Bounded exact fit:** `fit_peak_background` depends on the values only through their count and a 100-bin histogram over an anchor-dependent interval (0.55–1.5 × anchor). Pass 1 accumulates per key the count and the 150-bin 0–220 a.u. anchor histogram; pass 2 re-reads the same selection and fills each key's interval histogram with `np.histogram` bin edges. Storage is keys × 250 bins, never per event. The unchanged Inspector `fit_peak_background` then runs on stand-in values built from that histogram, one key at a time: each count at its bin centre, plus fillers beyond the interval to keep the count. These give it exactly the same histogram and count; the Inspector source is not modified.
- **Outputs:** the `.encal` lists every mapped key (unfitted ones as `0\t0`, as the reference); P = 1 writes `ID(t_ch, slab)\tmu\tsigma` and P ≥ 2 `# Position-dependent energy calibration (P regions per slab)` + `ID(time_ch, slab, region)\tmu\tsigma`. A per-key status file, a JSON sidecar (layout, P, boundaries, cuts, limits, inputs, status counts) and the summary plot follow. Loaders treat `0\t0` rows as no factor.
- **LM:** a per-slab factor is used for the whole slab (a one-region lookup); sides without COG limits are still rejected, since LM decompression needs them.

### Listmode

- Reuse the existing vectorized filtering, slab/region convention, `cornell_position` calibration keys, pair/region lookup, coordinates, DOI mapping and `CoincidenceV5`/`LMHeader` structures. Validate finite positive calibration factors and complete required metadata before writing.
- Stream records in batches to owned per-file outputs; merge in the confirmed natural file order. Record the timestamp reference explicitly. Do not retain every debug point; accumulate the same plot-bin summaries or use explicitly bounded numeric samples where a reference computation needs them.
- T1 proves the reference reads/prints `en_min_ch` but does not apply it in the fixed LM loop, unlike QC. Preserve and disclose that actual cut policy; do not silently add a threshold. The reference writes raw timestamp numeric values to float32, with an unused extracted first timestamp; confirm consumer units/precision before accepting LM metadata behavior.
- Preserve the binary schema; supply known duration/isotope/geometry/pixel/profile fields explicitly. Missing required header values block generation; optional unknown metadata follows the confirmed schema convention and is disclosed in provenance. No fabricated 10 s measurement or guessed module count.
- **Compact input (FR-22, T21):** each compact record is decoded into the fixed coincidence layout of the conversion hit limit H (empty slots channel −1, time 0, energy 0), regrouped into the same 1000-record batches, and passed to the unchanged batch function. Padding width matters: the reference's float32 sums use numpy pairwise summation, whose grouping depends on the row width, so H must be the fixed width. The slab draws (`np.random.randint(size=rows)`) do not depend on batching. Fixed input keeps `read_fixed_file_numpy` unchanged. The owner's fork writes side-1 padding as channel 0 (side 2: −1); channel 0 is in no minimodule of the Cornell map, so this has no effect there (byte-identical output with it masked, January split 3).
- Existing resume support is not a new recovery system: retain it only for validated matching manifest/settings/artifacts. Never trust arbitrary existing LM files as successful predecessor outputs.

### Throughput, event limit and progress (FR-1, FR-15, FR-21, FR-22, T25)

Owner decisions 2026-10-04/05. Profile on six January compact files (14.9 M records): P = 1 157 s (pass 1 50 s, pass 2 53 s with no GUI progress, fits 54 s on one core); P = 5 395 s (fits 281 s, 27,772 keys). The LM loops over files one at a time, while every `scripts_cornell/cornell_listmode*.py` ran CPU − 1 files at once; the reference calibration reads its files in a pool too (Cornell 2026-10-05: 6 F18 files, 3 min 23 s wall for 13 min 30 s CPU). Library defaults stay the reference (per-file 10 M, one worker, current LM stream) so the existing parity checks keep pinning reference behaviour; the manager passes the new modes explicitly.

- **Settings (`ProcessingLimits`, profile):** `workers` (0 = auto: `max(1, os.cpu_count() - 2)`, as the Inspector), `calibration_target_per_key` (T, default 3,000), `calibration_memory_mb` (default 8,192), `lm_seed` (fixed default, e.g. 0). `RunOptions.calibration_limit_mode` = `target` (default) or `reference`. Old profiles without these keys load with the defaults.
- **Event limit (`src/cornell/calibration.py`):** `calibrate(..., limit_mode="reference", event_limit=EVENT_LIMIT, target_per_key=None, memory_budget=None, ...)`. Reference mode limits passing events per file (the reference rule, `before <= limit`). Target mode (amended 2026-10-05) limits **kept sides**: S = K × P × T with K = `len(_mapped_keys(mapping, 1))`, ⌈S / n⌉ per file; `_sample` counts the sides `select` keeps per record (`np.bincount` of their record rows) and stops at the first record after which the file's kept sides exceed its share, with the same truncate-and-reselect step as the event rule. `CalibrationResult`/sidecar `"sampling"` record mode, unit, T, K, S, the per-file limit, events and kept sides per file; `"coverage"` records sides per key (min, median, below T, below `MIN_EVENTS`). The session's readiness worker computes K from the selected map (off the Tk thread) so the LDAT Processing tab shows S and the share before the run.
- **One decoding (target mode):** upper bound of kept pairs = n × (share + 2) sides in target mode (a record adds at most 2 sides past the share) × 16 bytes (int64 code + float64 energy). Within `memory_budget`, each file's pass returns its selected `codes`/`energies` exactly as `context.select` produced them; the parent concatenates them in input order and runs the existing `first`, `prepare_fit` and `second` from memory, so histograms and fits equal the two-pass path. The two-pass path stays for reference mode, no limit, or a bound above the budget; the pass-consistency check becomes the existing file-fingerprint check when decoding once.
- **Parallel reading and fits:** a small shared helper `src/cornell/parallel.py` (`ProcessPoolExecutor`, `spawn` context as `exe_programs/ldat_inspector_gui.py`; `workers == 1` runs in-process, no pool). Per-file work is a picklable top-level function building its own `_Context` and `_Reader`; results are consumed in input order. Two-pass parallel: workers return per-file integer histograms by code, merged exactly (integer adds; output order is already the sorted mapped keys, so discovery order does not matter). Fits: `_fit_key` is pure per key; chunks of rows go to the same pool with only the arrays they read, and results are reassembled by row. Calibration makes no random draws (it keeps only two-time-channel sides), so identical bytes for any worker count follow from exact merging.
- **Parallel LM (`src/cornell/listmode.py` via the helper):** `_process_file` per file in the pool; segments, debug summaries and counts merged in the existing natural order; `listmode.py` itself still imports no `multiprocessing` (the listmode check's ban stays). Each file sets the global NumPy stream once before its first batch, `np.random.seed(SeedSequence([lm_seed, file_index]).generate_state(1)[0])`, so `get_slab_cornell_vectorized` keeps its reference call order within the file; the seed and the per-file seeds are recorded in the segment records and the LM provenance and enter the job digest (resume reuses only segments of the same seed). Library default `lm_seed=None` keeps the current continuing stream.
- **Cancellation:** the parent checks `cancelled()` between completions, then `shutdown(cancel_futures=True)` and raises the stage's cancelled error; STOP also terminates the CLI child's process group, which holds the spawned workers (Linux).
- **Progress (FR-1):** the CLI progress event gains `phase` (`read`, `pass 2`, `fits`), `keys_done`, `keys_total`; the workflow relays them in `stage_progress`; the GUI status line shows `pass 2: file i/n` and `fits k/K`. Workers report through completions (per file, per fit chunk), not shared state.
- **Log timestamps (FR-1):** `PETsysManager.log` prefixes each line with local `HH:MM:SS`; the coordinator measures each stage with its clock and appends the elapsed time to `stage_finished` and the manifest attempt record.
- **Checks:** calibration: target mode share/stop points; one file target ≡ per-file; reference mode byte-identical to the reference functions (existing tests unchanged); workers 1 vs 2/4, both modes, P = 1 and 5: identical files; one vs two decodings identical, one decoding reads each file once and stays within its bound; budget fallback; cancellation in each phase publishes nothing; coverage numbers on a synthetic map. LM: workers 1 vs 2/4 byte-identical with `lm_seed`; same seed twice identical; different seeds differ only in ambiguous-side slabs; reference loop with the same per-file seeds byte-identical (the real check re-pins its LM item to per-file seeding). GUI/workflow: timestamps, elapsed text, progress phases, limit-mode radio, T/N display. Real: January compact, workers auto vs 1 identical, with timings; Cornell: the 34-file calibration and a multi-file LM, timed.

### QC and reports

- Extract the reference compact-coincidence filtering/counting/fits and existing PDF/Excel/plot generation into tracked functions. Do not replace it with Inspector algorithms.
- Pin the legacy per-file stopping condition (`accepted > 1,000,000`, normally 1,000,001 accepted pairs). Crucially channel occupancy is counted after minimodule channel cuts but before rejecting unresolved slab pairs; energy/flood/slab counts use resolved pairs. Report these as separate populations.
- Process bounded per-file samples into numeric columns/summaries. Avoid simultaneous full Python-list results for all selected files; use fixed histogram summaries when bins are known, or bounded arrays/two-pass reading for adaptive reference histogram ranges. Preserve reference histogram edges/fits, not just a similar-looking plot.
- Expected population comes from the selected map/config, including declared unpopulated minimodules. Record any intentional difference from the legacy hardcoded half-module rule on synthetic/real checks.
- Raw QC photopeak values stay in a.u.; keV is not invented. Add provenance and explicit sparse/invalid-fit states. Keep source mode and process success separate from observational findings.
- Add only actually required declared dependencies (notably `reportlab`, currently imported by QC but absent from the environment file); do not run environment updates as a planning side effect.

## Information available only at execution/ingest

Actual DAQD readiness, acquisition growth/loss text, successful duration/artifacts, converter split filenames, record format validation, decoded populations, rejected keys, calibration fit status and sample stopping counts cannot come from settings alone. Persist them in the run manifest while observed. The GUI/report uses these facts, never reconstructs rates from filenames or calls a prefix a whole acquisition. Missing telemetry is unavailable.

## Requirements to components/tasks

| Requirement | Components | Tasks |
| --- | --- | --- |
| FR-1 | Entry point, GUI (log timestamps, stage elapsed time, progress); unchanged Inspector | T13–T17, T25 |
| FR-2 | Tracked Cornell code, CLI, dependencies/preflight | T1–T2, T8–T11, T18 |
| FR-3 | Settings/profile, resolved inputs, GUI | T2, T5, T13 |
| FR-4 | Commands/runner/CLI | T3, T11 |
| FR-5 | Typed results, DAQD/init, artifact validation, workflow | T3, T6–T7, T12, T14, T16 |
| FR-6 | Acquisition service, ownership lifecycle | T6, T14, T19 |
| FR-7 | Runner/coordinator, queue-only GUI, shutdown | T3, T6–T7, T12–T14, T17, T19 |
| FR-8 | Acquisition attempt monitor/retry | T7, T14, T19 |
| FR-9 | Artifact reservation, attempts, processing writers | T4, T7–T12, T17 |
| FR-10 | Commands, input validators, conversion GUI | T3, T5, T12, T15 |
| FR-11 | File descriptors/manifests, GUI selection | T4–T5, T12, T15–T16 |
| FR-12 | Calibration/listmode/metadata/provenance | T1, T5, T8–T9, T11, T16, T19 |
| FR-13 | Pipeline coordinator and actual artifacts | T12, T16–T17, T19 |
| FR-14 | QC, reports, source settings | T1, T10–T12, T16, T19 |
| FR-15 | Bounded readers/accumulators/debug summaries; parallel workers, identical results | T5, T8–T10, T17, T25 |
| FR-16 | Local deterministic checks and Inspector regression | T1–T18 |
| FR-17 | Baseline comparison, Linux operator review | T1, T19 |
| FR-18 | Deployment docs, checkout audit, retirement boundary | T18–T19 |
| FR-19 | Bias-off after abnormal acquisition end, unknown-bias warning | T7, T14, T19 |
| FR-20 | Growth/progress feedback, editable safety limits | T7, T14, T19 |
| FR-21 | Compact/per-slab/position calibration, LM per-slab lookup; target/reference event limit | T20, T16, T19, T25 |
| FR-22 | Compact LM and pipeline conversion format; parallel LM, per-file streams from the LM seed | T21, T19, T25 |
| FR-23 | PETsys Python interpreter for init/acquire/bias tools | T22, T19 |
| FR-24 | Each stage validates only the records it reads, in one pass | T24, T19 |

## Alternatives rejected

- Copy the manager unchanged: retains missing ignored scripts, deletion hazards and false-success callbacks.
- Merge it into Inspector: explicitly outside owner intent and increases hardware/analysis coupling.
- Compact-only conversion: rejected 2026-09-30 (fixed-position workflows); adopted for the manager on 2026-10-05 (owner decision, FR-10: fixed is a discontinued fork feature). The library keeps the fixed readers, so no reader/algorithm change.
- Shell command strings: unsafe quoting and unnecessary dependency on machine-specific conda setup.
- Glob guessed split/calibration names: can include unrelated acquisitions and advance using stale output.
- Broad package reorganization or new fit/LM models: unnecessary scope and risks Inspector regression.

## Validation and gates

Use `conda run -n process_petsys --no-capture-output python ...` for checks. Named local check scripts live in `scripts/` and stay untracked; runtime modules never depend on them. Headless scripts generate deterministic fixtures rather than requiring private maps. Real validation uses explicit external inputs/baseline manifests.

1. T1 source inventory and synthetic/reference contracts; operator-installed-version confirmation gates numerical parity work.
2. Deterministic mock process/clock tests for all failure/ownership/cancellation paths, plus Linux dummy-process-group tests that never invoke hardware.
3. Fixed/compact/group negative fixtures; calibration/LM/QC mathematical and binary checks; bounded-storage checks independent of acquisition length.
4. Compile checks and withdrawn/hidden Tk checks with a fake backend; no real acquisitions started by tests. Run existing Inspector numerical and GUI regression scripts unchanged where available.
5. Clean runtime-checkout audit (tracked files only, no ignored scripts); Linux machine deployment/operator workflow review. A clean-checkout audit is not a build or release.

T19 cannot be marked verified from this Windows workstation. Real input comparison and operator review remain pending until versions, RAW/fixed/compact acquisitions and reconstruction metadata are supplied. If a reference discrepancy requires changing algorithms/schema/scope, update the spec before implementing it. Approval of this plan does not waive those gates.
