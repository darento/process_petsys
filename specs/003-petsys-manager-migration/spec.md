# Spec 003 — PETsys Manager migration

Status: `in progress`

Constitution: [`AGENTS.md`](../../AGENTS.md). Workflow: [`docs/prompts.md`](../../docs/prompts.md). This feature brings the sibling Cornell manager into this repository as a separate application; it does not extend LDAT Inspector with acquisition controls.

Paused by owner after T5 (2026-09-30) for the separately approved Inspector package refactor, spec004. Resumed by owner the same day, after spec004 shipped; T2–T5 checks and the T1 fingerprint audit were rerun and passed before T6. No remaining migration task is waived or implicitly completed.

## Goal

Run the existing Cornell acquisition, conversion, position-dependent energy calibration, listmode generation and quality-control workflows from one maintained `process_petsys` checkout. Preserve the recognizable manager UI and numerical workflows while removing deployment dependencies on ignored scripts, machine-specific paths, misleading success states and unsafe deletion.

## Confirmed scope (owner, 2026-09-30)

- Same repository, separate applications: `exe_programs/PETsysManager.py` and the existing `exe_programs/LDATInspector.py`.
- Initial operational target: the Cornell Linux acquisition machine. Windows runtime support is not required; headless numerical and import/compile checks may run on this workstation.
- Keep both fixed and compact PETsys formats; do not standardize everything on compact.
- Preserve Cornell QC, including its source/no-source acquisition and optional plots/slab analysis; do not substitute LDAT Inspector for it.
- Leave the sibling `gui_cornell` repository and all existing local scripts, configuration, mappings, calibration files and acquisitions intact during migration.

## Reference behavior and migration hazards

- Reference UI: `../gui_cornell/src/gui.py` (`PETsysGUIApp`), with setup/acquisition, RAW conversion, LDAT processing, LM generation and QC tabs, command logs, monitored acquisition/retries and a complete acquire → convert → calibrate → LM pipeline.
- Calibration/listmode calls currently target nonexistent `scripts_gui/` paths. Local counterparts are `scripts_cornell/cornell_slab_en_cal_fixed_position.py` and `scripts_cornell/cornell_listmode_cog_fixed_position.py`; QC uses `scripts_cornell/cornell_system_validation.py`. These scripts are ignored, not reproducible checkout dependencies.
- Main coincidence/group conversion writes fixed binary; QC conversion writes compact coincidence binary. The `.ldat` extension alone does not distinguish these formats or group/coincidence populations.
- Command callbacks currently execute regardless of exit status. Initialization and later pipeline stages can consequently report success after a failed command.
- Acquisition retries delete existing same-basename data. Conversion cleanup deletes presumed-empty files without checking size. Startup/close unconditionally clean global DAQD socket/shared-memory paths.
- Paths, Linux card addresses, conda initialization, listmode destinations and some LM header fields are hardcoded. Worker threads sometimes touch Tk widgets directly; retry sleeps can block the UI.
- Legacy QC stops after a bounded accepted-pair sample rather than necessarily inspecting the entire acquisition. Migration must disclose its actual scope, not relabel it as whole-run analysis.

## Data contract

- **System:** Cornell PETsys. DAQD/acquisition use the selected PETsys INI; processing uses a separately selected YAML and its matching channel/geometry map. IMAS manager workflows are out of scope, but existing IMAS inspection must remain unaffected.
- **Numbering:** channel IDs, SuperModules and minimodules follow the selected map. Cornell slab assignment and position regions retain their existing conventions, explicitly identified in output provenance; do not introduce CMB numbering or layout assumptions.
- **Formats/populations:** fixed coincidence records contain two detector sides; fixed group records have one group. Compact QC/Inspector input here is coincidence data. Accepted pairs, detector sides and per-channel hits have distinct denominators. None may be called a singles population.
- **Energy:** uncalibrated PETsys energies/photopeak parameters are in a.u.; calibrated energies/cuts are in keV. The fixed-position calibration is keyed by `(time channel, slab, region)` and includes region count/boundaries. Preserve this contract; do not silently pass it to a slab-only consumer.
- **Position/DOI:** use selected COG/DOI limits and map geometry. DOI ratio is light sharing; limits-derived mm-equivalent values must state their linear mapping, not claim independently calibrated depth.
- **External inputs:** PETsys software, device cards, INI/YAML/maps, calibration/limits and LM pair/region maps remain operator-supplied. Code and declared Python dependencies must be tracked; private data and machine settings must not be required in Git.

## Requirements

### Application and deployment

- **FR-1** — WHEN launched from the documented `process_petsys` environment on supported Linux, `exe_programs/PETsysManager.py` SHALL open a separate manager with the five reference workflow areas and command log. LDAT Inspector SHALL retain its existing entry point, behavior and independent lifetime.
- **FR-2** — WHEN installed from a clean checkout, the manager SHALL use tracked processing code for calibration, listmode and QC, with required dependencies declared in `process_petsys.yml`; it SHALL not import or execute ignored `scripts_cornell/`, `scripts_gui/`, `gui/` or the sibling repository. Missing external tools/data SHALL be listed before the affected action starts, rather than discovered halfway through a pipeline.
- **FR-3** — WHEN configuring a workflow, the operator SHALL select PETsys tools, DAQD socket/cards/type, INI, processing YAML, data/calibration/report/LM destinations, COG/DOI limits and LM pair/region maps as applicable. Settings SHALL be reusable across launches without editing Python; relocating the checkout SHALL not require selecting another `process_petsys` folder. Missing/stale settings SHALL disable or reject only affected actions with a specific reason. No active path SHALL require `/home/sie/` or `/data/nvmDisk/`.

### Commands, hardware and lifecycle

- **FR-4** — WHEN executing a command, paths and arguments SHALL be passed without shell interpolation, with an explicit working directory and interpreter/environment. Paths containing spaces or shell metacharacters SHALL remain literal. Commands SHALL not depend on sourcing a hardcoded conda initialization script.
- **FR-5** — WHEN a command completes, its result SHALL distinguish successful exit, failed exit, launch failure and cancellation. Initialization SHALL become ready only after successful initialization with a live owned DAQD; pipeline stages and success callbacks SHALL run only after successful prerequisites and validated required outputs. A failed QC operation SHALL not appear as completed QC or a detector PASS verdict.
- **FR-6** — WHEN DAQD is started/stopped or exits, the manager SHALL reflect actual process readiness/state, disable actions whose prerequisite is lost and manage only processes/resources it owns. Startup or close SHALL not remove an existing daemon's socket/shared memory. A stale-resource conflict SHALL be reported for explicit operator resolution, never silently deleted.
- **FR-7** — WHEN acquisition, conversion, calibration, listmode, QC or retry waiting is active, work SHALL remain off the Tk main thread and widget updates SHALL occur only on that thread. A single active foreground workflow SHALL prevent conflicting operations; DAQD remains a separately managed prerequisite. STOP SHALL cancel the active workflow and pending retries, terminate its owned child processes and prevent later stages from starting. Closing SHALL request safe cancellation/cleanup rather than abandon acquisition children.
- **FR-8** — WHEN monitoring acquisition, configurable startup/growth/frame-loss limits and bounded retries SHALL preserve the reference safety intent. Each attempt SHALL have a distinct identity; stale monitor/completion events SHALL not affect a newer attempt. Nonzero exit or missing/empty required acquisition output SHALL not count as success. Missing frame-loss information SHALL be reported as unknown, not zero. Logs SHALL identify attempts and retry reasons.
- **FR-9** — WHEN an output destination already contains acquisition or generated results, the manager SHALL refuse the collision or allocate a distinct run/attempt destination, never silently overwrite/delete the existing files. Automatic cleanup SHALL be restricted to explicitly recorded disposable artifacts of the current run; any LDAT removed as empty SHALL first be verified zero-length. Raw data, nonempty LDAT, failed attempts and unrelated index files SHALL be preserved by default.
- **FR-19** — WHEN an acquisition child was launched but did not finish by its own normal exit (monitor abort, STOP, window close, nonzero exit or runner failure), the manager SHALL request SiPM bias power off through the installed PETsys bias tool before any retry or later stage, and SHALL record the outcome with the attempt. If bias-off cannot be requested successfully (tool missing or failing, DAQD unavailable, timeout), the operator SHALL see a prominent "SiPM bias state unknown" warning, no further attempt SHALL start and nothing SHALL imply that the hardware is safe. *(Owner request, 2026-09-30.)*
- **FR-20** — WHEN an acquisition is running, the manager SHALL show the operator when the RAW file starts writing, when the growth check passes (the file is growing as expected) and the observed RAW size and recent growth rate at each monitor poll, including a visible warning when growth stops after the check has passed. Startup/growth/frame-loss/retry limits SHALL be editable operator settings, the frame-loss limit included, and each run SHALL record the values it used. *(Owner request, 2026-09-30; the default loss limit stays the reference 5% until the owner chooses another.)*

### Conversion and processing

- **FR-10** — WHEN converting RAW data, the manager SHALL offer explicitly labelled fixed/compact coincidence output and preserve fixed group conversion. Split settings and hit limits SHALL be validated independently of filename parsing. The fixed calibration/listmode pipeline SHALL consume fixed coincidence output; QC SHALL consume compact coincidence output. Wrong-format, wrong-population, empty, truncated or inconsistent inputs SHALL fail validation before numerical results are published; an ambiguous `.ldat` SHALL require an explicit format/population selection rather than guessing from its extension.
- **FR-11** — WHEN selecting LDAT inputs, the operator SHALL see the exact ordered file list to be processed. Selecting one split file MAY offer its siblings for confirmation, but SHALL not silently widen selection using a broad basename wildcard. Pipeline stages SHALL consume the exact outputs recorded by their predecessor, not guessed filenames or a presumed first split.
- **FR-12** — WHEN generating position-dependent calibration or listmode, tracked implementations SHALL preserve the supported reference algorithm, region convention, calibration key/file format and LM binary contract. Missing/invalid calibration factors, limits or map keys SHALL produce explicit rejection/unavailable counts, not fabricated measurements. Failed fits or estimates retained for compatibility SHALL be identified as such, never presented as fitted factors. Output provenance SHALL identify calibration, regions, maps, limits, cuts and selected population. LM metadata SHALL use supplied acquisition/profile values; unknown values SHALL not silently become hardcoded measured duration/geometry.
- **FR-13** — WHEN running the complete pipeline, stages SHALL use a validated snapshot of settings and record actual artifacts at every successful stage. Failure/cancellation SHALL stop advancement, preserve completed/partial outputs with their status and return controls to a consistent usable state. Manual workflows SHALL remain available separately; pipeline execution SHALL not mutate persistent user settings merely to coordinate stages.
- **FR-14** — WHEN running Cornell QC, the manager SHALL preserve with-source (60 s) and without-source (180 s) presets, compact conversion, optional plots and slab analysis (requiring plots), and the supported legacy QC computations/output types. Reports SHALL state actual files/sample bounds, accepted-pair/side/hit denominators, cuts, energy units, mapping and calibration status, source mode and thresholds where used. Occupancy findings SHALL remain observations of the coincidence sample, not hardware dead/hot or singles claims. Process completion SHALL be distinct from QC findings, and the displayed results location SHALL be the actual generated directory.
- **FR-15** — WHEN processing data, migrated numerical/file code SHALL operate with explicit bounds or batches and SHALL not retain every record as unbounded Python dictionaries/lists. QC's bounded-sample intent SHALL be retained and disclosed. Required changes to accumulation SHALL preserve validated numerical results, including reference slab randomness where present; performance claims SHALL be measured, not inferred from UI responsiveness.

### Validation and migration

- **FR-16** — BEFORE acceptance, deterministic headless checks SHALL cover command arguments, missing prerequisites, success/failure/cancellation, retries/stale events, DAQD ownership, output collisions/cleanup, exact file selection and fixed/compact/group boundaries. Synthetic Cornell fixtures SHALL pin calibration region assignment/factor validity, pair/side/hit counts, QC sample scope, LM record/header compatibility and numerical parity. Random reference behavior SHALL use a controlled seed or deterministic fixture without changing the production rule. GUI files SHALL pass compile checks; existing Inspector checks SHALL detect Cornell/IMAS regressions.
- **FR-17** — BEFORE declaring the migration shipped, an operator SHALL exercise initialization, monitored acquisition, conversion in both coincidence formats, fixed-position calibration/listmode, source/no-source QC with optional analyses, pipeline failure/STOP and window close on the Cornell Linux machine with representative acquisitions. Numerical results and output compatibility SHALL be compared with the reference workflows using recorded settings/inputs. Missing hardware/data checks SHALL stay explicitly pending, not be represented as passed.
- **FR-18** — WHEN documenting deployment, instructions SHALL describe the manager launch, Linux prerequisites, per-machine settings, supported format routes and required private input files. The sibling repository SHALL remain untouched until operator acceptance; retiring/archiving it or changing external launch shortcuts requires a separate explicit owner decision.

## Completion criteria

1. A clean checkout plus declared environment, PETsys tools and operator-supplied configuration/data runs every in-scope workflow without ignored script dependencies.
2. Every FR has a named reproducible check and recorded outcome; failures, cancellation and collisions never produce false success or loss of existing data.
3. Fixed calibration/listmode and compact QC preserve validated reference results and binary contracts, with truthful provenance and unavailable states.
4. Cornell Linux operator review passes FR-17. Existing Cornell/IMAS LDAT Inspector behavior remains unchanged.
5. Deployment documentation is complete; original repositories and user-owned configuration/data remain intact.

## Out of scope

- Embedding acquisition in LDAT Inspector, redesigning its UI, or adding fixed/group readers to it.
- Windows DAQ/runtime support, remote/SSH acquisition, IMAS manager workflows, a unified launcher or compiled executable packaging.
- A general repo/package reorganization, promoting all local scripts/configuration into Git, new calibration/fit models, clinical/hardware PASS/FAIL certification or singles analysis.
- Changing LM schema/reconstruction conventions, automatic external shortcut updates, deleting/archiving `gui_cornell`, or release/build commands.

## Clarification / approval gate

- Platform, conversion formats and preserved QC are owner-confirmed above. Owner approved proceeding to the plan/tasks and then T1 on 2026-09-30, followed by continuation with additive migration, T3, continued spec003 tasks and explicitly T5. T1 reference inventory, T2 settings/contracts, T3 commands/runner, T4 artifact storage, T5 input validation and (after the owner resumed the migration) T6 DAQD ownership/readiness, T7 monitored acquisition attempts/retries, T8 fixed-position calibration, T9 streamed listmode, T10 legacy QC, T11 headless processing CLI, T12 workflow coordinator and T13 manager shell checks are complete, including WSL Linux dummy-process/filesystem and synthetic Cornell/IMAS regressions. Live hardware orchestration and wiring of the GUI actions (T14–T16) remain pending. T10 takes expected channels from the selected map and declared unpopulated minimodules instead of the legacy hardcoded half-SuperModule rule, and reports the difference. T9 keeps the reference timestamp encoding (raw LDAT value as float32); consumer unit/precision acceptance stays with T19. This approval does not waive the remaining reference-version or Cornell Linux operator acceptance gates below.
- Before numerical migration, verify that the local fixed-position/QC scripts are the versions used on the Cornell machine. Record representative RAW/fixed/compact inputs, settings and baseline outputs. If installed scripts or LM metadata requirements differ, clarify/update this spec before implementation.
- Before hardware validation, record the Cornell PETsys tool version, device/card/socket settings, required LM acquisition/profile metadata and available operator test datasets. These are deployment/validation inputs, not invented defaults.
- **T1 clarification (2026-09-30):** owner confirmed the local fixed-position calibration/LM and QC scripts are the Cornell-used versions, and the installed converter is modified to support `--writeBinaryFixed`. This is an operator declaration, not remote hash/tool execution evidence. Synthetic contracts and the remaining tool/LM metadata/real-data gates are recorded in [`reference.md`](reference.md); no requirement or numerical behavior was changed.
