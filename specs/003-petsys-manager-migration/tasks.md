# Tasks 003 — PETsys Manager migration

Spec: [`spec.md`](spec.md). Architecture: [`plan.md`](plan.md). Owner requested plan/tasks and T1 on 2026-09-30, then continuation with additive migration and T3, continued spec003 work and explicitly T5. T1–T6 are complete; T7 acquisition attempts/monitoring is next. Live hardware, numerical algorithm migration and GUI wiring have not started.

**Resumed 2026-09-30** (owner request) after spec004 shipped, including its alias removal (Change 1). Revalidated before T6: T2–T4 → 107 selected pass (the Linux-only case passes under WSL), T5 → 38/38, WSL `--process-groups` → 5/5, WSL artifacts → 37/37; T1 reference/helper fingerprints unchanged 24/24.

**Paused after T5 (2026-09-30):** owner requested the Inspector package move, completed as [spec004](../004-ldat-inspector-package/tasks.md). After the move, T2–T4 checks passed (106 passed, one Linux-only skip / 107 selected), T5 formats passed 38/38, and T1 reference/shared-helper fingerprints remained unchanged 24/24. Inspector regressions passed 412/412. No T6 work started; resume only when the owner returns to this migration.

Execute in dependency order, one named task at a time. Each task cites its FRs and defines `Done when:` before coding. Record actual commands, inputs, outcomes and deviations under `Verified:`; tick only on success. Never commit automatically. Preserve user work, ignored scripts/data and the sibling repository.

## Check conventions

- Windows check fixtures now default to `%LOCALAPPDATA%/Temp/process_petsys`; Linux uses `~/.cache/process_petsys`. Owner requested no further use of the OpenCode temporary directory on 2026-09-30. Five local check scripts' defaults changed; previous baseline paths below remain historical evidence. Verified: all five scripts compile; T2–T4 → 106 passed, one Linux-only skip / 107 selected; T5 → 38/38. Fresh fixture paths are under the project-specific directory; no application logic or old evidence was changed.
- Run Python commands below with `conda run -n process_petsys --no-capture-output python ...` (or the environment interpreter). The abbreviated commands below begin at `python`.
- Planned check scripts: `scripts/petsys_manager_check.py` (settings/commands/lifecycle/artifacts), `scripts/petsys_manager_numeric_check.py` (fixtures/parity), `scripts/petsys_manager_gui_check.py` (hidden GUI), `scripts/petsys_manager_linux_check.py` (dummy processes only), `scripts/petsys_manager_reference_check.py` (explicit reference/real inputs), `scripts/petsys_manager_checkout_check.py` (tracked runtime audit). Create them in their owning task; they are local/untracked and must not be staged.
- Script flags/check modes named below are planned interfaces, not existing checks or claims of success. All scripts must guard `__main__`; importing them must not launch hardware or UI. Fixture baselines specify expected outputs independently of implementation.
- Runtime modules must never import these check scripts or ignored reference scripts. Reference comparisons explicitly load the selected originals in check-only code after inspection; preserve them unchanged.
- `Verified: pending` means untested, including hardware-gated work. Counts/fit tolerances/metadata exceptions are established before numerical code changes, not adjusted merely to pass.

## Foundations

- [x] **T1 — Reference inventory and parity contract** (FR-2, FR-12, FR-14, FR-16, FR-17). Inspect/hash the sibling GUI, local fixed-position calibration/LM and QC scripts plus shared numerical helpers. Add the reference check with deterministic miniature Cornell maps, matched fixed/compact/group encoders and manually expected slab/region/count/header fixtures. Record tool flags, processing cuts, sample semantics and source fingerprints in a private baseline manifest; request confirmation of Cornell-installed versions/LM profile.

  **Done when:** `python scripts/petsys_manager_reference_check.py --synthetic` passes reference-only contracts: calibration boundary exclusion versus LM clipping; five-region/1.8-edge layout; fixed 4 M side-cap batch semantics; QC 1,000,001 accepted-pair stopping and pre-slab occupancy; output schemas. Independent fixture bytes are decoded correctly by reference readers. A controlled LM fixture pins structure sizes/offsets and bytes, and declared header corrections/tolerances are recorded. Installed-version/tool/real-data confirmation is recorded as confirmed or explicitly pending; only confirmation unlocks numerical tasks T8–T10 against those versions.

  **Verified 2026-09-30:** `conda run -n process_petsys --no-capture-output python scripts/petsys_manager_reference_check.py --synthetic --confirm-local-scripts --confirm-modified-converter` → **PASS 17/17**. Repeat with the corresponding environment interpreter gives identical outcomes/evidence and fixture digests. Script lives in `scripts/`, stays ignored/untracked; import-only and `py_compile` checks pass. Independent bytes decode 4 pairs / 8 sides / 47 hits and four groups; calibration selects 6 independent sides, QC selects 2 resolved pairs / 4 energy sides but occupancy from 3 pairs / 36 hits. Manual region boundaries, exclusion/clipping difference, calibration cap overshoot, QC 1,000,001 stopping, real Gaussian/fallback/schema, seeded slab replay and every LM header/record offset/controlled byte pass. Cap boundaries use explicitly injected aggregate/counter seams, not full million-record acquisitions. All 24 reference/helper/tool source byte hashes unchanged. Owner confirmed local processing script versions and modified Cornell converter fixed-flag support (declarations, not remote measurements). Full contract/tolerances/hazards: [`reference.md`](reference.md). Private manifest: `C:\Users\dsanchez\AppData\Local\Temp\opencode\petsys-manager-reference-20260930T103452Z-c907d4be\baseline.json`. Exact modified tool provenance, reconstruction profile/timestamp units, shared-helper installed identity and representative real datasets/operator checks remain pending. No runtime source, user inputs, sibling code, hardware or release/build changes.

- [x] **T2 — Settings, typed contracts and prerequisites** (FR-2, FR-3, FR-16). Add manager package/contracts/settings; profile schema/versioning, immutable action snapshots and validation of paths, devices, safety/split/worker bounds, format/source flags and required LM metadata. No runtime GUI/hardware imports at module load.

  **Depends on:** T1 inventory (hardware/version confirmation not required for this task).

  **Done when:** `python scripts/petsys_manager_check.py --settings` proves YAML round-trip, unsupported/malformed profile rejection, stale paths affecting only relevant actions, separate INI/YAML, relocated-root resolution and literal paths containing spaces/metacharacters. Invalid durations/NaN factors, zero workers and slabs-without-plots are rejected. Settings snapshots cannot be mutated by later UI/profile edits; private inputs/config files remain byte-identical.

  **Verified 2026-09-30:** `conda run -n process_petsys --no-capture-output python scripts/petsys_manager_check.py --settings` → **PASS 35/35**. Synthetic fixtures only, under `C:\Users\dsanchez\AppData\Local\Temp\opencode\petsys-manager-settings-mpgam13e`; check script remains ignored/untracked (`git check-ignore scripts/petsys_manager_check.py`). Covers versioned YAML round-trip/exclusive save/explicit update, malformed/duplicate/unsafe/oversized profiles, stale action-specific paths, separate INI/YAML/map resolution, relocation/explicit root, literal metacharacters, unavailable LM metadata, fixed/compact/group routes, QC snapshot presets, invalid numeric/worker/batch/split/hit/flag bounds, NaN/overflow factors, immutable nested/direct-constructor snapshots, later config edits, pickle/JSON representation and private fixture byte preservation. Read-only filesystem/device probes are substituted; no actual hardware readiness is claimed. Structural map/limits/calibration/input schemas remain T5; commands/lifecycle remain T3/T6.

  **Compile/import:** `conda run -n process_petsys --no-capture-output python -m py_compile src/petsys_manager/__init__.py src/petsys_manager/contracts.py src/petsys_manager/settings.py scripts/petsys_manager_check.py` → **PASS**. Guarded cold import and suite re-import checks reject subprocess/thread starts and GUI imports → **PASS**. Initial cold import exposed Python 3.10 Windows `platform.system()` invoking a subprocess; changed to side-effect-free `sys.platform` and added the import regression. A parallel conda activation encountered a temporary-file conflict; sequential retries passed. SHA-256 audit against T1's private manifest: **24/24 reference/helper/tool files unchanged**. Existing flat `src` modules/imports, Inspector, sibling code and user configurations/maps/data untouched. No commit/tag/branch, hardware execution or release/build changes. T3 not started.

- [x] **T3 — Literal commands and owned command runner** (FR-4, FR-5, FR-7, FR-10, FR-16). Add argv/cwd builders and typed runner results/events; output draining/bounded logs, process-group cancellation/escalation/reaping and fake process/clock seams. Use the running environment interpreter for internal stages.

  **Depends on:** T2.

  **Done when:** `python scripts/petsys_manager_check.py --commands --runner` pins fixed/compact coincidence and fixed-group flags, hit/split settings and literal arguments; asserts no shell interpolation or sourced conda path. Exit 0/nonzero, spawn error and cancel yield distinct results; only validated success is eligible to advance. Full stdout/stderr pipes do not deadlock; log tail is bounded. `python scripts/petsys_manager_linux_check.py --process-groups` on Linux terminates/reaps a dummy parent/grandchild, including TERM-resistant children; mock tests cover the same decisions elsewhere. A missing Linux result remains pending, never inferred from mocks.

  **Verified 2026-09-30:** `conda run -n process_petsys --no-capture-output python scripts/petsys_manager_check.py --commands --runner` → **PASS 35/35** (9 command / 26 runner checks). `--settings --commands --runner` → **PASS 70/70**, including all 35 T2 regressions and guarded imports without GUI/process/thread starts. Fixtures only: latest named T3 run under `C:\Users\dsanchez\AppData\Local\Temp\opencode\petsys-manager-settings-ujiix0us` (commands) and `petsys-manager-settings-1staoch7` (runner). Checks pin DAQD card order, literal argv/cwd/environment, selected INI, acquisition mode/optional trigger, fixed/compact coincidence and fixed group flags, explicit duration/splits + 0.1 s offset and hit limits, known `.rawf` prefix handling, QC snapshot presets and internal `sys.executable -u -m src.cornell.cli` request/result argv (CLI execution remains T11). Mock clock/process checks cover TERM/KILL grace, parent exit with surviving descendants, cancellation at validation/publication, invalid/missing output decisions and reader/sink/poll/startup/reaping failures. Real native **dummy direct children only** verify literal arguments/environment/cwd, validated exclusive fixture output, simultaneous stdout/stderr pipe floods, nonzero/spawn errors and cancellation/reaping. Queue size, line fragments, UTF-8 decoding and returned log tail are bounded; only explicit `OutputValidation` success makes `CommandResult.can_advance` true. Completed events are informational; workflow advancement must use the returned result plus sticky cancellation.

  **Linux verified:** initially unavailable (WSL absent; Linux check correctly returned **2 / PENDING**). Owner installed Ubuntu-24.04 WSL2, then `wsl --distribution Ubuntu-24.04 --cd /mnt/c/Users/dsanchez/Desktop/Git/process_petsys --exec /usr/bin/python3 -B scripts/petsys_manager_linux_check.py --process-groups` → **PASS 5/5**, repeated with the same outcomes. First Linux run was orchestrated through the Windows `process_petsys` interpreter invoking that WSL argv. **Environment exception:** fresh WSL has no conda; these standard-library-only OS/process checks use `/usr/bin/python3` **3.12.3**. No packages/environment installed or numerical/PETsys deployment validation inferred. Private Linux fixtures: `/home/dsanchez/.cache/process_petsys/petsys-manager-linux-grzrrwar` (repeat; first `_4k52wlm`). Actual production Linux backend terminates owned cooperative/TERM-resistant groups, including a resistant descendant after its parent exits, refuses successful exit with live descendants, preserves an unrelated dummy process, and drains both pipes. Runner reaps its direct child; the check-only temporary subreaper explicitly reaps adopted dummy grandchildren and restores its prior state. No runtime global subreaper policy is introduced.

  **Compile/protection:** `conda run -n process_petsys --no-capture-output python -m py_compile src/petsys_manager/__init__.py src/petsys_manager/contracts.py src/petsys_manager/settings.py src/petsys_manager/commands.py src/petsys_manager/runner.py scripts/petsys_manager_check.py scripts/petsys_manager_linux_check.py` → **PASS**; embedded Linux dummy programs compile too. Both scripts stay ignored/untracked. T1 manifest SHA-256 audit → **24/24 reference/helper/tool files unchanged**. Inspected tools accept no INI argument in `init_system`, no custom-socket argv in init/acquisition, and no custom shared-memory name in DAQD: builders refuse unsupported contracts rather than inventing flags. Internal processing CLI, structural validators, output reservation and GUI coordination remain later tasks. No PETsys hardware/tools executed, user configurations/maps/data changed, sibling edits, commit/tag/branch or release/build commands. T4 not started.

- [x] **T4 — Run reservation, manifests and safe artifacts** (FR-9, FR-11, FR-16). Add exclusive run/stage/attempt destinations, exact artifact descriptors, durable statuses and an ownership-limited disposable policy.

  **Depends on:** T2.

  **Done when:** `python scripts/petsys_manager_check.py --artifacts` uses temporary fixtures to prove collisions do not overwrite, retry paths differ, failed RAW/nonempty LDAT/index files survive, unrelated similarly named files are not discovered/deleted, and only recorded current-run zero-byte disposable LDAT is eligible for removal. Path traversal/symlink escape is refused. Interrupted/failed publication leaves a failed/partial manifest, not a final success. Actual stage outputs are naturally ordered, not presumed to start at split 1.

  **Verified 2026-09-30:** `C:\Users\dsanchez\AppData\Local\anaconda3\envs\process_petsys\python.exe scripts/petsys_manager_check.py --artifacts` initially passed **33 checks / 1 Linux-only skip**; final expanded `--settings --commands --runner --artifacts` passed **106 checks / 1 Linux-only skip (107 selected: 70 T2/T3 + 37 T4)**. The corresponding environment interpreter is used because `conda` is absent from this shell's PATH. `src/petsys_manager/artifacts.py` adds exclusive run/stage/attempt reservation, immutable settings/ordered input snapshots, exact natural-ordered artifact/output inventories, bounded metadata, atomic no-replace publication and append-only fsynced manifest revisions. Incomplete work remains `partial`; validated successful latest attempts and unchanged outputs are necessary for final success. Recorded format/population/ownership cannot change during completion. Collision, output flush/link failure, interruption, manifest revision collision and directory-sync failure never advance the durable manifest to success. Failed/cancelled/launch-error states remain distinct. Failed RAW, nonempty LDAT, indices, unrelated files and partial outputs are retained. Only an unchanged recorded disposable zero-byte LDAT is removable; cleanup intent/result are persisted. Directory/file identity, traversal, reparse/symlink and external hard-link guards constrain ownership. No existing run is reopened for writing; workflow-specific resume remains T9.

  **Linux verified:** `wsl --distribution Ubuntu-24.04 --cd /mnt/c/Users/dsanchez/Desktop/Git/process_petsys --exec /usr/bin/python3 -B scripts/petsys_manager_artifact_check.py` → **PASS 37/37, no skips**. Exercises actual Linux file/directory fsync, hard links, no-follow cleanup and symlink destination/attempt/output/manifest escape rejection. Fixtures are written on WSL's Linux filesystem, not the shared checkout. Same T3 environment exception: WSL system Python 3.12.3 for standard-library filesystem checks; no conda/packages installed or Cornell deployment acceptance inferred. Windows fixtures: `C:\Users\dsanchez\AppData\Local\Temp\opencode\petsys-manager-artifacts-s96a3tm4`; Linux: `/home/dsanchez/.cache/process_petsys/petsys-manager-artifacts-0uha_vgh`. The local helper `scripts/petsys_manager_artifact_check.py` implements `--artifacts` fixtures and can run standalone without YAML/numerical dependencies. Both check scripts remain ignored/untracked.

  **Compile/protection:** environment-interpreter `-m py_compile` on all six manager modules plus `scripts/petsys_manager_check.py` and `scripts/petsys_manager_artifact_check.py` → **PASS**. Guarded module import covers artifacts without GUI/process/thread launches. SHA-256 protection audit → **29/29 existing config/map/T2–T3 runtime files unchanged**; T1 baseline audit → **24/24 reference/helper/sibling tool files unchanged**. The first fixture run exposed Windows fsync requiring a writable file descriptor and a mock method-binding error; both corrected before final checks. Numerical input validity, actual PETsys conversion output compatibility and GUI/hardware integration remain later tasks. No user settings/data, sibling edits, hardware execution, commits/pushes or release/build operations. T5 not started.

- [x] **T5 — Input contracts and structural validation** (FR-3, FR-10, FR-11, FR-12, FR-15, FR-16). Add bounded fixed/compact validation, explicit population descriptors, exact input selection and processing-config/map/limits/calibration parsing. Keep existing readers/Inspector APIs unless a independently checked shared fix is necessary.

  **Depends on:** T1 fixture contracts, T2, T4.

  **Done when:** `python scripts/petsys_manager_numeric_check.py --formats` accepts hand-encoded fixed coincidence, fixed group and compact coincidence fixtures; rejects empty/truncated headers/hits, invalid hit-limit/count/remainder, wrong route, missing/unmapped channels, malformed limits, duplicate/nonfinite/invalid calibration keys and inconsistent region metadata. Ambiguous legacy `.ldat` requires confirmation; no automatic extension-based detection. Relative `maps/...` resolves against the configured processing root; missing input lists are exact. Validation memory is bounded by batch/metadata, including repeated fixture records.

  **Verified 2026-09-30:** `C:\Users\dsanchez\AppData\Local\anaconda3\envs\process_petsys\python.exe scripts/petsys_manager_numeric_check.py --formats` → **PASS 38/38**. Adds only `src/cornell/__init__.py` and `src/cornell/inputs.py` runtime files; existing readers and Inspector APIs stay unchanged. Hand-encoded little-endian fixtures pin fixed coincidence (130-byte record at hit limit 4), fixed group (65-byte record), compact coincidence (66 bytes for two 2-hit sides), exact record/side/hit counts and file byte totals. Empty/header-only/truncated headers/hits/remainders, zero/over-limit hit counts, invalid/mismatched fixed limits, wrong routes, unmapped active channels, nonfinite active energy and late corruption reject without a successful summary. Fixed padding contributes no hits and is ignored. Cancellation or input size/mtime/identity changes also reject. Exact ordered selection never expands to siblings, lists every missing selected path and requires explicit legacy format/population confirmation. A deliberately indistinguishable fixed fixture passes as either two groups or one coincidence: byte validation cannot prove its population, and never changes the supplied descriptor.

  **Metadata/provenance:** processing YAML resolves relative `maps/...` against the configured root, not the YAML directory. Strict loaders reject duplicate keys, aliases, unsafe tags, malformed layouts/addresses/channel lists, invalid cuts and unknown unpopulated module/minimodule keys. Generated maps match unchanged `map_factory` for selected Cornell full-system and IMAS 1DAQ maps; Cornell processing rejects unsupported nonsummed/non-eight-position slab layouts without changing shared map support. COG/DOI limits require finite increasing bounds and unique mapped time-channel/slab keys. Position calibration preserves the three-part `.encal` schema/region header and reads identically through `KevConverter(..., 'cornell_position')`; zero/nonfinite/negative mu, negative/nonfinite sigma, duplicate/invalid keys and inconsistent region counts/boundaries reject. Zero sigma remains a valid compatibility value. Sparse limits/factors retain explicit missing-key results; legacy region boundaries remain unavailable, never guessed. Optional explicit JSON sidecars require schema 1, matching region count, increasing 0–1 boundaries and the actual calibration SHA-256. No file is modified by validation.

  **Bounds:** full fixed scans retain one byte batch (at most 16 MiB; positive bounded batch/record settings); compact scans retain one side's at most 255 16-byte hits. Metadata files cap at 4 MiB, YAML at 100,000 nodes, tables at 100,000 rows/entries, individual table rows at 4,096 characters and nesting at 32. Sparse immutable/pickleable lookup tables use selected IDs rather than dense arrays indexed to the largest channel. Repeated fixed fixture: **100 → 12,000 records**, batch 17 (**2,210 bytes**), `tracemalloc` peak **12,488 → 12,544 bytes**; all counts agree. Compact **5,000 pairs / 20,000 hits**, reported payload/header buffer **34 bytes**. These are measured fixture/storage results, not throughput or real-acquisition claims. Private fixtures/evidence: `C:\Users\dsanchez\AppData\Local\Temp\opencode\petsys-manager-formats-cc7atpjy` (`memory-evidence.json` under its repeated-file check directory). Check script remains ignored/untracked.

  **Regressions/compile/protection:** same environment interpreter: `scripts/petsys_manager_check.py --settings --commands --runner --artifacts` → **106 passed / 1 existing Linux-only skip**; `scripts/ldat_inspector_check.py --selftest` → **59/59 Cornell/IMAS synthetic checks**; `scripts/cornell_slab_convention_check.py` → **16/16**; `scripts/ldat_unpopulated_check.py` (without `--real`) → **6/6**. Inspected each regression script before running. New package/input/check files pass `-m py_compile`; guarded input-module import launches no process/thread/GUI. SHA-256 audits → **35/35 pre-existing config/map/shared-reader/helper/T2–T4 runtime files unchanged**, and **24/24 T1 reference/helper/sibling tool fingerprints unchanged**. No shared fixes, new dependencies, acquisitions, sibling edits, commits/pushes or release/build commands. Actual Cornell/IMAS acquisition checks remain in T17/T19; processing CLI/writer integration remains later tasks. T6 not started.

## Hardware orchestration (headless first)

- [x] **T6 — DAQD ownership/readiness and initialization** (FR-5, FR-6, FR-7, FR-16). Add daemon state/service tied to process identity and selected hardware/INI; configuration changes or daemon death invalidate initialization. Never clean global resources at launch/close.

  **Depends on:** T2–T3; verify PETsys flags/readiness contract from T1 before live validation.

  **Done when:** `python scripts/petsys_manager_check.py --daqd` proves pre-existing socket/shared-memory conflicts cause no deletion/foreign-process termination, launch failure/death toggles state correctly, readiness requires more than a checked box, failed initialization never unlocks acquisition and changed daemon/config invalidates prior ready state. Stop/close only signal owned identities. Linux dummy socket/process check in `petsys_manager_linux_check.py --daqd` covers lifecycle without calling PETsys hardware.

  **Readiness contract (from the inspected sibling `sw_daq_tofpet2` source; installed Cornell version unmeasured):** `daqd` binds its socket *before* opening cards and shared memory, so a socket file is not readiness. Its client loop, which starts only after the cards and frame server succeed, answers command `0x02` (GetDataFrameSharedMemoryName). That is the same query `daqd.Connection()` sends on connect. **Hazard found:** `FrameServer::freeSharedMemory` calls `shm_unlink` even when its own `O_EXCL` create failed, so launching a second `daqd` while `/dev/shm/daqd_shm` exists deletes the other daemon's shared memory. An existing socket *or* shared-memory path therefore blocks start. An existing socket alone already makes `daqd` fail `bind` before touching shared memory.

  **Verified 2026-09-30:** new `src/petsys_manager/acquisition.py` (DAQD part; T7 adds monitoring) implements `DaqdService` with the plan's `off → starting → ready | failed → stopping → off` states.
  - **Start:** refused while either resource path exists; the refusal lists them and removes nothing. The owned `daqd` is launched through the T3 runner in its own thread.
  - **READY:** requires the owned pid to exist, the socket path to be a socket, and a reply to the `0x02` query that names the expected shared memory (`/` + basename). On Linux the reply must also come from the owned pid (`SO_PEERCRED`), and the shared-memory path must exist. A single probe connection waits until the startup deadline (default 30 s). A wrong name, a foreign peer or the deadline stops the daemon and ends FAILED.
  - **Initialization:** allowed only while READY, with a matching daemon config (exact argv/cwd/socket/shm) and no initialization already running. `init_system` exit 0 is followed by a re-query of the same daemon. The result is recorded against the daemon generation and the init config: argv/cwd, INI path and INI SHA-256, since the plan says INI changes invalidate. `acquisition_ready()` discards it permanently on any mismatch, so the operator must initialize again.
  - **Daemon exit:** any unrequested exit ends FAILED, clears initialization and lists leftover resources, which then block restart until the operator resolves them. Stop/close only cancel the owned runner child: TERM, then KILL after 10 s by default.
  - **Other properties:** status snapshots carry a revision number (reads do not bump it). Sink failures cannot break ownership. The module imports no Tk/`subprocess`/settings, and a guarded AST audit shows no `unlink`/`remove`/`kill`/`killpg`/`shm_unlink`/`system` calls. The builders are imported lazily (PyYAML), so the Linux check can inject dummies.

  Environment interpreter `-X utf8 scripts/petsys_manager_check.py --daqd` → **PASS 21/21**; repeated 10× standalone with identical results.
  - **Fake children/resources:** cover socket and shared-memory conflicts without launch or deletion; a read-only real-file probe; launch error; death during startup; socket without a protocol reply staying STARTING until timeout TERM; late reply → READY; wrong name and foreign peer; initialization refused before READY; failed init (exit 1); successful init; daemon death after or during init; INI edit and card change invalidation (sticky); owned-only TERM with leftovers kept and reported; TERM-resistant KILL; close reaping; start refused while running; generation/revision ordering; the source audit; side-effect-free import; policy bounds.
  - **Linux:** `wsl --distribution Ubuntu-24.04 ... /usr/bin/python3 -B scripts/petsys_manager_linux_check.py --daqd` → **PASS 7/7**, repeated 3×. It uses a real Unix socket plus a shared-memory file under `~/.cache/process_petsys/pm-daqd-*` and a dummy daemon mimicking bind → shm `O_EXCL` → delayed service. Never `/tmp/d.sock`, `/dev/shm/daqd_shm`, PETsys tools or hardware.
    - The service stays STARTING while the socket is bound, reaches READY only after the reply, needs a single probe connection, initializes, then closes with the owned pid gone and no leftovers.
    - A test-launched foreign daemon on the same paths is refused and stays alive with its resources intact.
    - A stale shared-memory file is untouched.
    - A SIGKILL crash after initialization → FAILED, not initialized, both leftovers reported and blocking restart.
    - A TERM-resistant dummy is killed and reaped, with leftovers reported.
    - An early exit (card failure) → FAILED with its exit code; a never-serving daemon → startup timeout and TERM.
    - A wrong name → FAILED; a failed init leaves acquisition locked.
  - The first Linux run exposed a dummy crash on EPIPE from an abandoned probe connection; real `daqd` sends with `MSG_NOSIGNAL`. Fixed in the dummy, and readiness changed from repeated short probes to one connection waiting until the deadline, so `daqd` does not accumulate abandoned clients.
  - Same T3 environment exception: WSL system Python 3.12.3, standard library only.

  **Regressions/compile/protection:** `-m py_compile` on all seven manager modules, `src/cornell` and the three check scripts → PASS. `--settings --commands --runner --artifacts --daqd` → **128 selected pass**. `--formats` → 38/38. `ldat_inspector_check.py --selftest` → 59/59, `cornell_slab_convention_check.py` → 16/16, `ldat_unpopulated_check.py` → 6/6. WSL `--process-groups --daqd` → 12/12; WSL artifacts → 37/37. T1 fingerprints unchanged 24/24. 47 GUI/launcher/shared-source/config/map/Inspector files match this session's pre-edit SHA-256 snapshot. T2–T5 runtime modules were not edited. The sibling repositories were only read. `git check-ignore` confirms the new and changed check scripts stay ignored.

  **Still pending:** installed Cornell `daqd`/`init_system` behaviour (command `0x02`, `SO_PEERCRED` from the direct child, startup time with two cards, TERM shutdown time) is unmeasured. Custom-socket support stays refused (T3). An explicit operator stale-resource resolution action is not implemented. GUI wiring is T14; live acceptance is T19.

- [ ] **T7 — Acquisition attempts, monitoring and retries** (FR-5, FR-7, FR-8, FR-9, FR-16). Add monotonic growth/startup/loss monitoring, cancellation-aware bounded retry waits and immutable attempt IDs with new output paths.

  **Depends on:** T3–T4, T6.

  **Done when:** `python scripts/petsys_manager_check.py --acquisition` with fake clock/process/files covers startup timeout, adequate/insufficient growth, loss below/equal/above 5%, absent/malformed loss telemetry, nonzero exit, empty/missing output, three-attempt exhaustion, cancel during retry delay and stale completion events. No retry follows STOP; missing telemetry is unknown; short successful acquisition records an unexercised growth check. Prior attempt data persists and no sleep/widget work occurs on the main thread.

  **Verified:** pending.

## Tracked Cornell numerical workflows

- [ ] **T8 — Position calibration extraction and bounded summaries** (FR-2, FR-9, FR-12, FR-15, FR-16). Extract confirmed fixed-position calibration into tracked functions; replace per-event Python energy lists with histogram/moment summaries, retain reference limits/sample semantics and `.encal` contract, and expose fit/fallback status/provenance.

  **Depends on:** T1 installed-reference confirmation, T4–T5.

  **Done when:** `python scripts/petsys_manager_numeric_check.py --calibration` pins region-edge cases, out-of-range rejection, sample denominator/cap overshoot, >=50 criterion, missing keys and failed-fit estimate labeling. Histogram counts match reference exactly and written fitted/fallback values agree at 0.001 a.u. precision; repeated data length changes do not expand accumulator storage beyond mapped keys/bins. Existing `KevConverter(..., 'cornell_position')` reads generated output; sidecar records boundaries/cuts/status without changing keys or overwriting files. Group/coincidence sample routes are independently exercised.

  **Verified:** pending; T1 processing-script confirmation and T4–T5 checks complete; calibration algorithm migration remains.

- [ ] **T9 — Streamed compatible listmode** (FR-2, FR-9, FR-12, FR-15, FR-16). Extract confirmed LM processing with explicit pair/region maps, destinations and metadata. Preserve record encoding, timestamp/order/selection and bounded debug summaries; guard legacy resume using matching manifests.

  **Depends on:** T1 installed-reference/LM schema confirmation, T4–T5, T8 calibration contract.

  **Done when:** `python scripts/petsys_manager_numeric_check.py --listmode` decodes output independently, matches controlled reference record bytes and verifies documented `LMHeader` fields/offsets plus supplied duration/profile values. Counts/energy cuts/positions/DOI/pair IDs and file merge order match; calibration/limits/pair/region failures are counted or block output as specified. Header metadata missing/overflow is rejected; no guessed 10 s/module count. Debug storage remains bounded on repeated inputs. Resume refuses mismatched settings/input manifests and never trusts unrelated LM files.

  **Verified:** pending; T1 processing-script confirmation and T4–T5 checks complete; reconstruction profile/timestamp contract and T8 remain.

- [ ] **T10 — Legacy QC extraction and truthful reports** (FR-2, FR-9, FR-14, FR-15, FR-16). Extract compact QC and existing report types; retain numerical histogram/fitting behavior with bounded sampled columns/summaries and selected-map expectations. Add accurate provenance/population/unavailable states, not a new QC model.

  **Depends on:** T1 installed-reference confirmation, T4–T5.

  **Done when:** `python scripts/petsys_manager_numeric_check.py --qc` gives manually expected pair/side/hit counts including occupancy from an unresolved slab pair excluded from energy counts. It pins per-file stopping before/at/after the legacy limit and reports actual sampled/read counts. All four option combinations (slabs alone rejected) are checked, no-source/source metadata stays distinct, sparse/failed fits are unavailable and raw peak units are a.u. Reference histogram edges/counts/fits and PDF/Excel/plot output content match documented tolerances; changed inactive-module expectation is explicitly explained. Bounded-storage/merge tests do not keep every selected file's full Python lists concurrently.

  **Verified:** pending; T1 processing-script confirmation and T4–T5 checks complete; QC algorithm migration remains.

- [ ] **T11 — Headless processing CLI and declared dependencies** (FR-2, FR-4, FR-9, FR-12, FR-14, FR-16). Add tracked internal CLI dispatch for calibration/LM/QC, request/result manifests and structured progress; declare only needed missing Python dependencies.

  **Depends on:** T2–T5, T8–T10.

  **Done when:** `python scripts/petsys_manager_check.py --cli` launches each action through `sys.executable` with a fixture request and validates actual output/result paths. Invalid requests, corrupted inputs and numerical failure exit nonzero without success results; output paths with spaces/metacharacters remain literal. Source audit proves runtime imports/calls do not reference ignored scripts/sibling GUI; processing modules import headlessly. Dependencies cover actual imports, including QC's `reportlab`.

  **Verified:** pending.

## Workflow and separate GUI

- [ ] **T12 — Manual actions and fail-closed pipelines** (FR-5, FR-7, FR-9–FR-11, FR-13–FR-14, FR-16). Add workflow coordinator with settings snapshots and manifest-derived exact artifacts. Implement acquire/fixed/calibrate/LM and acquire/compact/QC stage graphs plus manual actions; preflight the entire chosen graph.

  **Depends on:** T3–T7, T11.

  **Done when:** `python scripts/petsys_manager_check.py --workflows` runs fake successful graphs and injects spawn error/nonzero exit/invalid or missing output/STOP at every stage. No successor starts after failure/cancel; old similarly named files cannot substitute for new output. Only one foreground workflow starts; all paths end in consistent controls/state and unchanged persistent settings. QC uses correct 60/180 s presets and compact descriptors, full pipeline uses fixed coincidence, manual group action never feeds LM. Stage logs/manifests distinguish completed processing from QC findings.

  **Verified:** pending.

- [ ] **T13 — Manager shell, profiles and input UI** (FR-1, FR-3, FR-7, FR-16). Add separate launcher/GUI shell, five tabs, main-thread event poller, settings/profile controls and optional safe asset loading. Do not instantiate another Inspector root or change its code.

  **Depends on:** T2, T5, T12.

  **Done when:** `python -m compileall -q exe_programs/PETsysManager.py exe_programs/petsys_manager_gui.py src/petsys_manager src/cornell` succeeds. `python scripts/petsys_manager_gui_check.py --shell` with withdrawn window/fake backend checks tab names/log/profile reload, relocated module-relative assets, separate INI/YAML controls and prerequisite reasons without private paths. Instrumented widget access occurs only on the main thread; no hardware/processes launch at startup. Inspector imports/entry point remain independent.

  **Verified:** pending.

- [ ] **T14 — DAQD/acquisition/STOP/close UI** (FR-5–FR-8, FR-16). Connect GUI controls to owned backend states, initialization and monitored acquisition; block conflicting actions and implement asynchronous close.

  **Depends on:** T6–T7, T12–T13.

  **Done when:** `python scripts/petsys_manager_gui_check.py --acquisition` proves failed init/dead daemon never unlock acquisition, readiness is backend-driven, retry wait remains responsive, stale events cannot restore active buttons and STOP during acquisition/retry prevents new attempts. Fake subprocess close/cancel leaves no owned children; no cleanup of global socket/shm is invoked. GUI event polling remains live while shutdown completes/failure is shown.

  **Verified:** pending.

- [ ] **T15 — Conversion controls and exact file selection UI** (FR-10–FR-11, FR-16). Add explicit fixed/compact coincidence selection and fixed-group conversion, independent duration/split/hit controls and ordered LDAT selection/validation feedback.

  **Depends on:** T3–T5, T12–T13.

  **Done when:** `python scripts/petsys_manager_gui_check.py --conversion` verifies generated request format/population, positive split/duration validation without basename parsing, wrong-route rejection and visible exact inputs/outputs. An unsuffixed or differently named acquisition converts correctly. A selected split without index 1 is supported; unrelated prefix files remain unselected; ambiguous legacy format requires confirmation.

  **Verified:** pending.

- [ ] **T16 — Calibration, LM, QC and pipeline UI** (FR-1, FR-5, FR-11–FR-14, FR-16). Connect remaining tabs/manual actions/pipeline with actual artifacts, region/LM metadata fields, source/options and report destinations. Keep the reference layout recognizable; no Inspector embedding.

  **Depends on:** T8–T13, T15.

  **Done when:** `python scripts/petsys_manager_gui_check.py --processing` proves actual calibration/results paths are displayed, next stages use recorded artifacts, incomplete metadata/options/prerequisites cannot start work, slabs require plots and both source presets dispatch correctly. Every injected stage failure/cancel remains failure/cancel, never pipeline success/QC PASS. Persistent settings and selected external files are unchanged after successful/failed pipelines.

  **Verified:** pending.

## Acceptance and deployment

- [ ] **T17 — Integrated deterministic and Inspector regressions** (FR-1, FR-7, FR-9, FR-13, FR-15–FR-16). Run the complete manager fixture/mock/hidden-GUI checks and compile checks; exercise bounded storage and failures end-to-end. Run existing Inspector checks without quietly changing their expected output.

  **Depends on:** T1–T16.

  **Done when:** manager check scripts' `--all` modes pass, Linux dummy process-group results are recorded, numerical fixture outputs have exact/manual expectations and large repeated fixtures demonstrate bounded accumulation/debug storage. Run `scripts/ldat_inspector_check.py`, `scripts/ldat_revision_check.py`, `scripts/ldat_issue_check.py`, `scripts/ldat_scale_check.py --selftest`, `scripts/ldat_views_check.py`, `scripts/ldat_unpopulated_check.py`, `scripts/ldat_pair_choices_check.py`, `scripts/ldat_ports_check.py`, `scripts/cornell_slab_convention_check.py`, `scripts/ldat_processing_check.py` and `scripts/ldat_gui_check.py` with the named environment. Record unavailable scripts/data/display checks as pending blockers, not passes. Shared fixes/deviations have explicit regression evidence.

  **Verified:** pending.

- [ ] **T18 — Deployment docs and clean-runtime-checkout audit** (FR-2, FR-16, FR-18). Update README/add repo deployment documentation with Linux launch/prerequisites/profile/format routes, external input checklist, runtime module map and safety semantics. Preserve original repository and external shortcuts.

  **Depends on:** T11–T17.

  **Done when:** `python scripts/petsys_manager_checkout_check.py --tracked-runtime` audits intended tracked runtime files/dependencies without ignored-script references, writes an isolated runtime copy for headless imports/fixture CLI checks and proves no original private settings/data are needed for import/startup validation. Include new intended tracked files explicitly if not yet staged; do not stage automatically. Linux startup/offline fixture checks run from a different cwd using explicit external settings. README distinguishes supported fixed/compact/group routes and unavailable hardware/data. No build/release, sibling edits or shortcut changes occur.

  **Verified:** pending.

- [ ] **T19 — Cornell Linux baseline comparison and operator acceptance** (FR-6–FR-8, FR-12–FR-14, FR-17–FR-18). Using confirmed tool/script versions and operator-selected profile/representative data, compare migrated numerical outputs and perform all live GUI workflows with the operator. Keep this as an explicit external gate.

  **Depends on:** T1 installed-version/metadata/real-data confirmation, T17–T18.

  **Done when:** `python scripts/petsys_manager_reference_check.py --real --manifest <operator-baseline.json>` records input/settings/version fingerprints and parity for fixed calibration/LM and compact QC at predefined tolerances. Operator records live initialization, monitored acquisition, both coincidence conversions, group conversion/manual group calibration where used, LM compatibility with its consumer, 60/180 s QC with plot/slab options, complete pipeline, failure/STOP/retry and close. Existing data survives; no owned children remain; actual result locations and scope match reports. Representative Cornell/IMAS Inspector regression results and the FR-by-FR pass/fail table are recorded. Only all completion criteria passing permits `Status: shipped`; sibling retirement remains a separate owner decision.

  **Verified:** pending; requires Cornell Linux hardware/operator and representative data, not available as a demonstrated check on this workstation.

## Planning validation

**Verified 2026-09-30:** PowerShell structural check passed: FR-1–FR-18 uniquely numbered and mapped in the plan; T1–T19 consecutively numbered, each with `Done when:` and pending `Verified:`; exactly one approved/in-progress spec; no trailing whitespace in the three documents. No application/numerical/hardware checks were run for this documentation-only change. Creating these documents does not complete T1 or any other implementation task.
