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
- T4 stores append-only `manifest-NNNNNN.json` revisions, atomically linked from flushed private metadata; only a fully persisted revision updates the in-memory snapshot. Run state stays `partial` while any work is unfinished. Failed publication/durability preserves earlier evidence; no arbitrary existing run is adopted. `read_manifest` is read-only; guarded resume remains T9. Ordered external inputs are recorded as explicit descriptors with sizes/mtime, without hashing large acquisitions.
- The converter output prefix is isolated to its run. Inventory files before/after execution in that directory, restrict discovery to that exact stage prefix, validate outputs and record the actual ordered set. Empty split files may be reported/ignored; deletion, if enabled, requires zero size and recorded disposable ownership. Do not delete `.lidx` merely because a wildcard matches.
- Conversion duration is explicit, not parsed from a basename. Validate positive duration, split count/hit limit and actual PETsys support. Never use `cpu_count() - n` without a safe lower bound; split count is distinct from worker count.
- Relative `map_file` resolution uses an explicit processing root, initially the discovered repo root for legacy `maps/...` values. Do not silently switch semantics to the YAML directory. Show resolved paths in preflight and record them. Absolute external paths remain valid.
- `configs/cornell_full_system_20260928.yaml` and its map are available local candidate validation inputs, not an auto-selected profile or tracked requirement. Their geometry comments are not authority for LM metadata; confirm the reconstruction profile explicitly.
- Assets, if retained, resolve relative to the module, not the current directory. An optional logo failing to load cannot block operations. Do not introduce an ignored top-level `gui/` runtime dependency.

## Format routes and stage contracts

| Workflow | Input | Output/consumer |
| --- | --- | --- |
| Manual coincidence conversion | RAW acquisition + selected INI | Explicit fixed or compact coincidence descriptor |
| Manual group conversion | RAW acquisition + selected INI | Fixed group descriptor |
| Position calibration | Selected fixed group or fixed coincidence files, COG limits, processing config | Position `.encal`, status/provenance, summary plot |
| Manual LM generation | Fixed coincidence files, position calibration, COG/DOI limits, pair/region maps, metadata | Compatible LM header/records plus provenance/debug summaries |
| Complete pipeline | Acquire → fixed coincidence conversion → position calibration → LM | Actual artifacts from each successful predecessor |
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

### Listmode

- Reuse the existing vectorized filtering, slab/region convention, `cornell_position` calibration keys, pair/region lookup, coordinates, DOI mapping and `CoincidenceV5`/`LMHeader` structures. Validate finite positive calibration factors and complete required metadata before writing.
- Stream records in batches to owned per-file outputs; merge in the confirmed natural file order. Record the timestamp reference explicitly. Do not retain every debug point; accumulate the same plot-bin summaries or use explicitly bounded numeric samples where a reference computation needs them.
- T1 proves the reference reads/prints `en_min_ch` but does not apply it in the fixed LM loop, unlike QC. Preserve and disclose that actual cut policy; do not silently add a threshold. The reference writes raw timestamp numeric values to float32, with an unused extracted first timestamp; confirm consumer units/precision before accepting LM metadata behavior.
- Preserve the binary schema; supply known duration/isotope/geometry/pixel/profile fields explicitly. Missing required header values block generation; optional unknown metadata follows the confirmed schema convention and is disclosed in provenance. No fabricated 10 s measurement or guessed module count.
- Existing resume support is not a new recovery system: retain it only for validated matching manifest/settings/artifacts. Never trust arbitrary existing LM files as successful predecessor outputs.

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
| FR-1 | Entry point, GUI; unchanged Inspector | T13–T17 |
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
| FR-15 | Bounded readers/accumulators/debug summaries | T5, T8–T10, T17 |
| FR-16 | Local deterministic checks and Inspector regression | T1–T18 |
| FR-17 | Baseline comparison, Linux operator review | T1, T19 |
| FR-18 | Deployment docs, checkout audit, retirement boundary | T18–T19 |
| FR-19 | Bias-off after abnormal acquisition end, unknown-bias warning | T7, T14, T19 |
| FR-20 | Growth/progress feedback, editable safety limits | T7, T14, T19 |

## Alternatives rejected

- Copy the manager unchanged: retains missing ignored scripts, deletion hazards and false-success callbacks.
- Merge it into Inspector: explicitly outside owner intent and increases hardware/analysis coupling.
- Compact-only conversion: breaks fixed-position workflows and adds unwanted reader/algorithm changes.
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
