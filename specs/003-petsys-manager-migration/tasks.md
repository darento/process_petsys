# Tasks 003 — PETsys Manager migration

Spec: [`spec.md`](spec.md). Architecture: [`plan.md`](plan.md). Owner requested plan/tasks and T1 on 2026-09-30, then continuation with additive migration and T3, continued spec003 work and explicitly T5. T1–T14 are complete; T15 conversion controls and exact file selection UI is next. Live hardware, conversion/processing GUI wiring have not started.

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

- [x] **T7 — Acquisition attempts, monitoring and retries** (FR-5, FR-7, FR-8, FR-9, FR-16). Add monotonic growth/startup/loss monitoring, cancellation-aware bounded retry waits and immutable attempt IDs with new output paths.

  **Depends on:** T3–T4, T6.

  **Done when:** `python scripts/petsys_manager_check.py --acquisition` with fake clock/process/files covers startup timeout, adequate/insufficient growth, loss below/equal/above 5%, absent/malformed loss telemetry, nonzero exit, empty/missing output, three-attempt exhaustion, cancel during retry delay and stale completion events. No retry follows STOP; missing telemetry is unknown; short successful acquisition records an unexercised growth check. Prior attempt data persists and no sleep/widget work occurs on the main thread.

  **Tool contract (inspected sibling `sw_daq_tofpet2`; installed Cornell version unmeasured):** `acquire_sipm_data -o PREFIX` writes `PREFIX.rawf/.idxf/.tmpf` (`write_raw`) and `PREFIX.modf`. At the end of each step `write_raw` prints on stderr `writeRaw:: some events were lost for N (x%) frames; all events were lost for M (y%) frames`, formatted `%5.1f`; a step without frames prints `nan`. Like the reference, only the "all events" percentage is compared, and a value above the limit retries (equal passes).

  **Verified 2026-09-30:** `src/petsys_manager/acquisition.py` adds `AcquisitionService`.
  - **Threads:** `start(settings, store, basename=, prerequisite=)` validates the argv (T3 `build_acquisition`) before reserving anything, then returns an `AcquisitionHandle`. Attempts, the monitor and retry waits run in worker threads. Updates are `RunEvent`s with one ordered sequence; `AttemptFilter` drops older sequences and superseded attempts for the GUI queue. STOP (`handle.stop()`) is sticky.
  - **Attempts:** each attempt is a new T4 run-store attempt `acquisition/attempt-N/` with prefix `attempt-N/<basename>`, so no retry deletes anything. Failed attempts keep and record their partial files; the manifest stores status/exit code plus `details` (attempt, retry reason, growth, frame loss). `RunStore.finish_attempt` gained that optional `details` field.
  - **Monitor:** reference defaults from `AcquisitionSafety` (45 s startup, 20 s window, 5 s poll, 20 MB, 5%, 3 attempts, 2 s delay), monotonic clock. Early abort on no `.rawf` data within the startup timeout or on growth below the minimum over the window from first data; TERM/KILL through the T3 runner. The runner gained an `exited` event so the monitor stops judging a finished child. A lost prerequisite (callable, e.g. `DaqdService.acquisition_ready`) aborts without retry.
  - **Verdicts:** success needs exit 0, nonempty `.rawf` and `.idxf` and loss not above the limit. Retry: startup timeout, insufficient growth, exit 0 without data (`no_data`), frame loss. No retry: nonzero exit (the reference treated it as success), launch error, lost prerequisite, STOP, storage failure. Missing or unparseable loss is `unknown` (`None`, never 0) and does not block success. A run ending before the window records growth `not_exercised`.
  - **Logs:** tool lines are tagged `[attempt-N]`, and service decisions `[acquisition attempt-N] kind: message`, including retry reasons.

  Environment interpreter `-X utf8 scripts/petsys_manager_check.py --acquisition` → **PASS 21/21**, repeated 10× with identical results.
  - A scaled fake clock (100× real time, real blocking waits) runs the reference defaults. Fake children write small real files into their attempt directory; a fake size probe scripts `.rawf` growth.
  - Covered: adequate growth; short success (`not_exercised`); startup timeout ×3 → exhausted with TERM on each and all three attempt directories kept; insufficient growth → retry → success, with attempt 1 bytes intact and tagged log lines.
  - Also: loss parsing below/equal/above 5%, multi-step maximum, absent, `-nan`, >100 and mixed; loss 7.5% → retry, then 5.0% → success; absent/malformed loss → `unknown`, stored as `null`; nonzero exit terminal with 4 partial artifacts recorded; missing/empty `.rawf`/`.idxf` never success.
  - Also: launch error terminal; unmet prerequisite → no launch or attempt; prerequisite lost mid-attempt → TERM, no retry; invalid argv/basename → nothing reserved; second start refused.
  - Also: STOP during an attempt → cancelled, TERM, no retry; STOP during a 100 000 s retry delay → returns in under 10 s real time with no second launch; a blocked attempt-1 monitor released during attempt 2 emits nothing and changes nothing; `AttemptFilter` ordering.
  - Also: all clock waits and update-sink calls happen off the main thread, and `start` returns at once; the runner `exited` event arrives once, before `completed`; the module has no `sleep`/Tk `after`/deletion/kill calls and no Tk/`subprocess` import; reference defaults are pinned.
  - **Linux:** `wsl ... /usr/bin/python3 -B scripts/petsys_manager_linux_check.py --process-groups --daqd --acquisition` → **PASS 16/16**, repeated 3×. It adds 4 real-clock checks with a dummy acquisition that writes a real growing `.rawf` under `~/.cache/process_petsys/pm-acquisition-*`:
    - growth passes and 4 artifacts are recorded;
    - a stall times out, then the retry succeeds;
    - slow growth, then 12.5% loss, exhausts 2 attempts;
    - STOP terminates the dummy without a retry.
    Every dummy pid is gone afterwards.

  **Regressions/compile/protection:** `-m py_compile` on the manager modules and the three check scripts → PASS. `--settings --commands --runner --artifacts --daqd --acquisition` → **149 selected pass**. `--formats` → 38/38. `ldat_inspector_check.py --selftest` → 59/59, `cornell_slab_convention_check.py` → 16/16, `ldat_unpopulated_check.py` → 6/6. WSL artifacts → 37/37. Tracked changes are limited to `acquisition.py`, `artifacts.py` (optional `details`) and `runner.py` (one `exited` event), plus these spec notes. The pre-existing config/map edits were not touched. `git check-ignore` confirms the new and changed check scripts stay ignored.

  **Still pending:**
  - Installed `acquire_sipm_data`/`write_raw` output names, loss-line format and stall behaviour with Cornell hardware (T19).
  - Superseded by the amendment below: bias after TERM.

  **Amendment (owner, 2026-09-30; FR-19, FR-20):** after an abnormal attempt end, request SiPM bias off (`set_bias --power off`); if that fails, warn prominently and stop retrying. Publish RAW growth progress for the operator. Keep the loss limit an editable profile setting (default 5%).

  **Amendment verified 2026-09-30:**
  - **Bias-off:** `commands.build_bias_off` gives the literal argv `set_bias --power off` (default DAQD connection only), and preflight now requires `tool:set_bias` for acquire/QC/pipeline. When a launched attempt did not exit 0 by itself (abort, STOP, nonzero exit, runner failure), `AcquisitionService` runs it before any retry, bounded by its own 30 s timeout rather than by STOP. The result goes in `AttemptSummary.bias` and in manifest `details.bias`. A `bias_unknown` event, `outcome.bias_unknown` and an `UNKNOWN` message report a failure, and no further attempt starts.
  - **Progress:** `rawf_progress` events (size, bytes/s, `growing`) are published at every poll after the first data. The growth-pass message reads "RAW file growing as expected". A stall after the pass is a logged warning, not an abort.
  - **Loss limit:** stays `safety.max_loss_percent` in the profile (default 5), is recorded in the run manifest settings, and T14 exposes it.
  - **Checks:** `--acquisition` → **PASS 27/27**, repeated 10×. New checks:
    - bias-off after a startup-timeout abort, placed before the retry;
    - bias-off still runs after STOP, and after a nonzero exit;
    - no bias-off after a normal end, `no_data`, frame loss or launch error;
    - bias-off failure, a 0.3 s timeout or a missing tool → `unknown` with no retry;
    - the bias-off argv, the custom-socket refusal and the preflight requirement;
    - growing progress, and a stall warning without an abort;
    - a 10% limit accepting 7.5% loss, with the limit recorded, and the profile YAML field.
  - **Linux:** WSL `--process-groups --daqd --acquisition` → **PASS 16/16** ×3. The STOP check confirms a real dummy bias-off process ran; a normal run launches none.
  - **Regressions:** all manager modes → **155**; `--formats` 38/38; Inspector 59/59, 16/16, 6/6; WSL artifacts 37/37; compile PASS.
  - **Pending:** the installed `set_bias` flags and their effect on Cornell hardware (T19); GUI display (T14).
  - Coordinator and pipeline use (T12) and GUI wiring (T14).

## Tracked Cornell numerical workflows

- [x] **T8 — Position calibration extraction and bounded summaries** (FR-2, FR-9, FR-12, FR-15, FR-16). Extract confirmed fixed-position calibration into tracked functions; replace per-event Python energy lists with histogram/moment summaries, retain reference limits/sample semantics and `.encal` contract, and expose fit/fallback status/provenance.

  **Depends on:** T1 installed-reference confirmation, T4–T5.

  **Done when:** `python scripts/petsys_manager_numeric_check.py --calibration` pins region-edge cases, out-of-range rejection, sample denominator/cap overshoot, >=50 criterion, missing keys and failed-fit estimate labeling. Histogram counts match reference exactly and written fitted/fallback values agree at 0.001 a.u. precision; repeated data length changes do not expand accumulator storage beyond mapped keys/bins. Existing `KevConverter(..., 'cornell_position')` reads generated output; sidecar records boundaries/cuts/status without changing keys or overwriting files. Group/coincidence sample routes are independently exercised.

  **Verified 2026-10-01:** new tracked `src/cornell/calibration.py` ports the owner-confirmed reference `scripts_cornell/cornell_slab_en_cal_fixed_position.py`, fingerprint `8e5bb5cf…` unchanged.
  - **Reused unchanged:** the shared `src` helpers (`read_fixed`, `filters_fixed`, `utils_fixed`, `detector_features_fixed`, `fits`, `mapping_generator`).
  - **Kept from the reference:**
    - selection: min-channel filter, max-energy minimodule and its energy-channel rule, slab, and Y-COG with power 2;
    - the float32 COG-limit arrays (slab < 16);
    - region boundaries (5 regions, edge 1.8); out-of-[0, 1] excluded, exactly 1 → last region;
    - a per-file accepted-side limit (default 4 M) tested before each record batch and each side batch, batch 5000;
    - `np.histogram` 100 bins over 0–200 a.u. into `fit_gaussian(cb=6, pk_finder='peak')`, at least 50 samples;
    - mean/std fallback on `RuntimeError`;
    - `.encal` header/key/order/3-decimal format.
  - **Changed:**
    - Per-key Python energy lists → a fixed int64 histogram plus count and float64 mean/M2 (103×8 bytes per key), merged in input order.
    - Each entry gets a status: `fitted`, `fallback_mean_std` (written, labelled), `insufficient_samples` (not written, as in the reference), `fit_error` (the reference would abort), or `invalid_result` (non-finite or non-positive μ; not written). A negative fitted σ is written as |σ| and flagged.
    - Side rejections are counted by reason: min channels, minimodule channels, unresolved slab, missing COG limits, COG out of range.
    - `calibrate()` first runs a full T5 `validate_ldat` on each input, and rejects an input that changes before sampling ends.
    - `write_calibration` uses exclusive creation for the `.encal` and the JSON sidecar. The sidecar uses the T5 `schema_version 1` contract: boundaries, `calibration_sha256`, cuts (min_ch; no channel-energy cut, as in the reference), histogram/fit, sampling semantics, source digests, per-file samples/rejections, status counts and non-fitted keys only (bounded).
    - `plot_summary` is the reference plot, also exclusive, with the fallback count in its title.

  Environment interpreter `-X utf8 scripts/petsys_manager_numeric_check.py --calibration` → **PASS 14/14**; `--formats --calibration` → **52/52** (38 + 14), repeated 3×. The reference script is loaded only as a check oracle on seeded synthetic fixed LDAT (fixture map, 2 SuperModules). The checks cover:
  - Boundaries for 1–11 regions match exactly.
  - `region_ids` equals `compute_region_id_numba` on 20 000 random float32 points plus exact boundary, 0.9995, NaN, invalid channel/slab and zero-width limits, and gives the manual edge vector.
  - Histogram binning equals `np.histogram` exactly (float32 edges, 0/200/near-edge values, chunked adds). Fallback moments match exact float64 within 1e-12.
  - On 6000 pairs, accepted sides, keys, per-key counts and per-key histograms equal the reference exactly. All four rejection reasons occur and `considered = accepted + rejected`.
  - Fitted μ/σ equal the reference fit bit-for-bit. Fallbacks agree within 0.001 a.u. The ≥ 50 written set equals the reference's. `fitted`, fallback and insufficient keys all occur.
  - 49 vs 50 samples; forced `RuntimeError`/`ValueError`, negative σ and NaN/negative μ get the right labels.
  - Limit overshoot equals the reference for (limit, batch) = (1, 2), (3, 2), (7, 5), (50, 7) and unlimited.
  - The group route matches the reference histograms.
  - Missing and narrow limits are rejected and counted.
  - Bytes per key are constant for 500 and 5000 records, and keys stay within the mapped bound.
  - Output: `KevConverter(…, 'cornell_position')` and T5 `load_calibration(metadata_path=)` read it. The reference `write_position_cal` gives byte-identical text for the same factors. A second write refuses either path and leaves both files unchanged. The plot is exclusive.
  - Mixed populations, compact input, an empty list, invalid regions, unmapped channels and an input changed after validation are rejected.
  - The tracked module imports no `scripts`/`docopt`/`multiprocessing`/Tk.

  **Regressions:** Inspector 59/59, slab 16/16, unpopulated 6/6; manager modes 155; compile PASS. Shared helpers are unchanged per `git diff`.

  **Still pending:**
  - Real Cornell data parity (T19).
  - Full T5 validation is a per-hit Python scan (slow on multi-GB files; not measured).
  - The reference relies on channel ID −1 for padding slots and does not use the header count; this is preserved, not re-verified against the installed converter.
  - Fits are sequential; worker pooling and the CLI/request/RunStore publication are T11/T12.
  - Calibration has no sample-limit field yet beyond `ProcessingLimits.calibration_side_limit` / `batch_records`; the caller passes them.

- [x] **T9 — Streamed compatible listmode** (FR-2, FR-9, FR-12, FR-15, FR-16). Extract confirmed LM processing with explicit pair/region maps, destinations and metadata. Preserve record encoding, timestamp/order/selection and bounded debug summaries; guard legacy resume using matching manifests.

  **Depends on:** T1 installed-reference/LM schema confirmation, T4–T5, T8 calibration contract.

  **Done when:** `python scripts/petsys_manager_numeric_check.py --listmode` decodes output independently, matches controlled reference record bytes and verifies documented `LMHeader` fields/offsets plus supplied duration/profile values. Counts/energy cuts/positions/DOI/pair IDs and file merge order match; calibration/limits/pair/region failures are counted or block output as specified. Header metadata missing/overflow is rejected; no guessed 10 s/module count. Debug storage remains bounded on repeated inputs. Resume refuses mismatched settings/input manifests and never trusts unrelated LM files.

  **Verified 2026-10-01:** new tracked `src/cornell/listmode.py` ports the owner-confirmed reference `scripts_cornell/cornell_listmode_cog_fixed_position.py`, fingerprint `f7adbaf8…` unchanged.
  - **Reused unchanged:** the shared `src` helpers (`read_fixed`, `filters_fixed`, `utils_fixed`, `detector_features_fixed`, `listmode` structures, `fits`) and T8 `CalibrationMaps`/`create_region_boundaries`.
  - **Kept from the reference:**
    - every selection/numerical expression per 1000-record reader batch, in the same call order, so its `np.random` slab draws are preserved;
    - Y-COG power 2; LM region clipping to [0, 0.999] (calibration still excludes);
    - `511 / mu * E` and the keV window; Y decompression 25.6 mm; DOI linear 20 mm mapping; region offsets; pair lookup with swap; timestamp of the max-energy time hit; every `CoincidenceV5` field cast;
    - `en_min_ch` read but not applied (recorded as such);
    - natural basename merge order; reference per-file and `_all.lm` naming.
  - **Changed:**
    - The header comes from supplied `LMMetadata`: acquisition/measurement time, isotope, detector size, modules, rings, ring distance and the pixel grid `linspace(0, size, pixels + 1)`. `identifier` "Cornell", `startTime` 0 and version (9, 5) are kept; the reference zero fields stay zero and are listed. Missing metadata, header overflow (float32 pixel size included) and an energy window above the uint16 field are rejected before any output.
    - Each rejected pair is counted once, at its first failing stage: min channels, minimodule channels, unresolved slab, no position region, missing calibration, energy window, missing DOI limits, Y/DOI out of range, unmapped region, no pair, missing timestamp. `records_read = written + rejected` is enforced per file.
    - Reference failure cases become counted rejections: partial decompression masks (the reference raises `IndexError`), a minimodule beyond its region array (raises), and region −1 reading pair row 99 (the reference writes a fabricated pair). A wrapped int16 `dt` and pixels outside the grid are written as before, and counted.
    - Strict pair/region map loaders: exact columns, regions 0–99, uint16 pair IDs, no duplicates, region keys must be minimodules of the selected map.
    - Debug keeps a fixed 1500-bin energy histogram, per-SuperModule hit counts and 100 × 64 × 64 flood histograms instead of event lists; plots are exclusive. The SuperModule plot is a bar chart over the selected map's SuperModules, not the hardcoded 3 × 10 grid.
    - Inputs: full T5 validation first, explicit fixed coincidence descriptors, distinct basenames, and the list must already be in natural order (no silent reordering). An input changed after validation is refused.
    - Storage: an exclusive job directory with `lm-job.json` (settings, source digests, metadata, input size and mtime); one segment `*.part` per input plus a completion record written last; header + segments merged with per-segment SHA-256 re-verification and published by a no-replace link; a JSON sidecar (cuts, timestamp contract, region/position/DOI policy, sources, per-file counts, resume) written last. Segments are retained; nothing is deleted.
    - Resume replaces the legacy `-c`: it requires an identical job record and reuses only segments whose record, input identity, size and SHA-256 match. Orphan partials of this job are ignored and reported. Any other file, a completed job or a directory without a job record refuses the resume.
  - **Timestamp contract (reference behavior kept):** `time` is the raw LDAT timestamp (minimum of the two sides) as float32, with no offset or scaling. `timestamp_unit` is the operator's declared consumer unit and is only recorded. Consumer interpretation/precision stays with T19.

  Environment interpreter `-X utf8 scripts/petsys_manager_numeric_check.py --listmode` → **PASS 14/14**; `--formats --calibration --listmode` → **66/66** (38 + 14 + 14). The reference is loaded only as a check oracle on seeded synthetic fixed LDAT (fixture map, SuperModules 7 and 21). The checks cover:
  - `LMHeader` 176 / `CoincidenceV5` 24 bytes and every field offset against an independent table. Legacy-equivalent metadata gives header bytes identical to the reference `write_header`; other supplied values decode independently, and only the supplied fields differ from the legacy header.
  - Each of the 11 metadata fields missing, overflowing isotope/pixels/module, NaN/negative values, float32 overflow and a 70 000 keV window are rejected without creating the job directory.
  - Two files × 2500 pairs, one-time-channel (random-slab) sides included: merged records are byte-identical to the reference loop run in natural order with the same seed, and per-file counts are equal. Every rejection reason except missing timestamp occurs, and the int16 `dt` wrap occurs. An independent `struct` decode checks every record (amount, pair IDs, keV window, DOI range, pixel range).
  - A controlled forward/reversed pair gives the same pair ID, ordered energies/positions, Δt −123 both ways and time 5000.0.
  - Pair/region loaders equal the reference pandas parsers; calibration and COG arrays equal the reference arrays; malformed pair/region rows are rejected.
  - Pair row 99 (the reference writes fabricated pair 9; ours rejects), region-array overflow and partial decompression limits (the reference raises; ours counts).
  - Debug energy/SuperModule/flood histograms equal histograms of the reference debug lists; summary size is identical for 300 and 3000 pairs; plots are exclusive.
  - Sidecar provenance fields; exclusive job directory.
  - Resume: after cancelling during file 2, resume refuses a changed config, metadata, pair map, input list or input mtime, a stray `.lm`, a same-size tampered segment, a completed job and a non-job directory, leaving files byte-identical after each refusal. It then reuses file 1, ignores one orphan and reproduces the fresh output bytes. A second resume of the complete job is refused.
  - Input contract rejections: compact, group, reversed order, empty, duplicate basenames, unmapped channels, non-reference calibration boundaries, wrong limits kind. Natural order equals `natsort`. The tracked module imports no `scripts`/`docopt`/`multiprocessing`/Tk/pandas/natsort.

  **Regressions:** manager modes 155; Inspector 59/59, slab 16/16, unpopulated 6/6; compile PASS. Shared helpers are unchanged (`git diff -- src` shows only the new file); the four T1 primary fingerprints are unchanged.

  **Still pending:**
  - Real Cornell data, LM consumer comparison and timestamp unit/precision acceptance (T19).
  - Removing segments after a verified merge would halve disk use but needs an owner decision under FR-9; segments are kept.
  - Processing is sequential, and validation is the slow per-hit T5 scan (not measured). Worker pooling and CLI/RunStore publication are T11/T12; the pipeline must pass the acquisition duration as `acquisition_time_s`.

- [x] **T10 — Legacy QC extraction and truthful reports** (FR-2, FR-9, FR-14, FR-15, FR-16). Extract compact QC and existing report types; retain numerical histogram/fitting behavior with bounded sampled columns/summaries and selected-map expectations. Add accurate provenance/population/unavailable states, not a new QC model.

  **Depends on:** T1 installed-reference confirmation, T4–T5.

  **Done when:** `python scripts/petsys_manager_numeric_check.py --qc` gives manually expected pair/side/hit counts including occupancy from an unresolved slab pair excluded from energy counts. It pins per-file stopping before/at/after the legacy limit and reports actual sampled/read counts. All four option combinations (slabs alone rejected) are checked, no-source/source metadata stays distinct, sparse/failed fits are unavailable and raw peak units are a.u. Reference histogram edges/counts/fits and PDF/Excel/plot output content match documented tolerances; changed inactive-module expectation is explicitly explained. Bounded-storage/merge tests do not keep every selected file's full Python lists concurrently.

  **Verified 2026-10-01:** new tracked `src/cornell/qc.py` (extraction/counting/fits) and `src/cornell/qc_report.py` (output types) port the owner-confirmed reference `scripts_cornell/cornell_system_validation.py`, fingerprint `3c1b9c57…` unchanged.
  - **Reused unchanged:** `filter_min_ch`, `get_maxEnergy_sm_mM`, `get_slab_cornell` (its `random` draws keep the reference call order), `get_max_num_ch`, `calculate_centroid`, `fit_gaussian`, `get_electronics_nums`.
  - **Kept from the reference:**
    - per-pair selection: `en_min_ch` applied to every hit at reading, `min_ch` before and after picking the max-energy minimodule;
    - occupancy counted before the unresolved-slab rejection; energy, flood and slab samples use both sides of resolved pairs;
    - per-file stop once accepted pairs reach 1,000,001 (reference `> 1,000,000`), with the extra iterator read;
    - photopeak histogram 150 bins over 0–250 a.u.; `fit_gaussian(cb=12 minimodule / 10 slab, min_peak=20, pk_finder='peak')`; resolution `2.35·σ/μ·100`;
    - flood: Y-COG power 2 over 8 energy + 2 time channels, 40–200 a.u. window, 500 × 500 bins over 0–105 mm;
    - output names and types: `missing_channels_report.pdf` always; with plots `photopeak_values.xlsx`, `photopeak_SM_*`, `channels_present_per_cassette`, `slab_distribution_SM*`, `floodmap_SM_*`, `floodmap_all_SM`; with slabs also `photopeak_<sm>_<mm>` and `photopeak_distribution`.
  - **Changed:**
    - Storage: per-key fixed histograms + count + float64 mean/M2, and per-SuperModule fixed flood histograms. Files merge as they finish, with a bounded flush buffer; no per-side lists. The unused reference x-COG profile is not computed.
    - Populations are reported separately: records read, pairs processed, occupancy pairs/hits (unresolved-slab pairs included), accepted pairs and sides. Rejections are counted by reason; slab-assignment flags are counted (random single-time-channel choices included). A multi-minimodule side with no positive energy (the reference raises `TypeError`) is a counted rejection.
    - Fits: `fitted`, or unavailable as `sparse` (< 20 counts in the highest bin), `fit_failed`, `fit_error` or `invalid_result`. Unavailable fits have no μ/σ/resolution and no plot line. The reference wrote the population mean/std as `Mu`; it is now a labelled sample moment. A negative fitted σ is reported as its absolute value and flagged. Values stay in a.u.; QC applies no calibration.
    - Expected channels: selected map minus the config's declared `unpopulated_minimodules`. The legacy hardcoded rule (SuperModule `(sm+1) % 3 == 0` populated only in minimodules 0, 1, 4, 5, 8, 9, 12, 13) is not applied. The summary and PDF list the minimodules it would skip that are now expected (and vice versa). Declaring those minimodules reproduces the legacy expectation exactly. Hits on declared-unpopulated minimodules are reported.
    - Layouts follow the selected map's SuperModules: slab distribution per map SuperModule (not SM 0–29), channel frequency per cassette `sm // 3` (not cassettes 0–1), and the combined flood places the 0.21 mm per-SuperModule histograms side by side, highest SM on the left (not three SuperModules re-binned at 0.315 mm). Photopeak grids grow beyond 16 slots.
    - Reports: the PDF keeps every reference line in order and adds a sample/provenance section (source mode, cuts, units, sample bound and populations, per-file read/accepted/stopped, map/config digests, expectation rule) and an "observed in the sample, not a dead-channel verdict" note. The Excel keeps the five reference columns (fitted rows sorted by Mu), then adds unavailable rows, status, samples, labelled sample moments and a provenance sheet.
    - Output directory: created exclusively (`default_directory` keeps the reference `YYYYmmdd-HHMMSS` name); every file is created exclusively; `qc_summary.json` (process completion separate from findings, source mode/duration or "not recorded", options, cuts, populations, per-file counts, fit statuses, expectation differences, output SHA-256) is written last.
    - Inputs: full T5 validation of explicit compact coincidence descriptors in the given order; an input changed after validation is refused. Slabs without plots are rejected. The tracked modules import reportlab/openpyxl/matplotlib lazily and never import `scripts`, docopt, multiprocessing, Tk, natsort, colorama or tqdm.

  Environment interpreter `-X utf8 scripts/petsys_manager_numeric_check.py --qc` → **PASS 14/14**; `--formats --calibration --listmode --qc` → **80/80** (38 + 14 + 14 + 14). The reference is loaded only as a check oracle (pools replaced by an in-process map) on seeded synthetic compact LDAT (fixture map, SuperModules 1 and 2; 7 and 41 for the layout check). The checks cover:
  - Manual 5-pair file: 5 read, 3 occupancy pairs / 42 hits (the unresolved pair included), 2 accepted pairs / 4 sides, one rejection per reason, slab flags 5 adjacent / 1 non-adjacent. A 0.1 a.u. hit is cut at `en_min_ch` 0.2 (43 hits and mean 100.05 a.u. with cut 0). Equal to the reference.
  - The reader equals `read_compact.read_binary_file`.
  - Two files × 1500 pairs, plots + slabs: occupancy, slab counts, accepted pairs, every minimodule/slab histogram and flood histogram are exactly equal to the reference. Every fitted μ/σ is exactly equal. Every reference fallback key is unavailable here, with its sample mean/std within 1e-9 relative. Sparse fits occur.
  - Stopping at limits 5/6/7/100 on a 6-valid-pair file gives read/accepted/stopped 6/5/yes, 7/6/yes, 8/6/no, 8/6/no. The reference counter seeded at 999,998/999,999/1,000,000 matches limits 3/2/1 in accepted pairs and iterator reads. The default equals settings `qc_pair_limit` 1,000,001.
  - Options: none / plots / plots + slabs give the same file set as the reference (apart from the documented slab-distribution SuperModules). Slabs alone are rejected by `run_qc` and `RunOptions`. An existing results directory is refused.
  - Source mode with (60 s), without (180 s) and not recorded give distinct summary/PDF metadata and identical totals/findings.
  - Unavailable fits: blank Mu/Sigma/resolution in the Excel, status and samples, a.u. units in the Excel/summary/PDF. Forced fit failure, error and non-positive μ are unavailable; a negative σ becomes its absolute value and is flagged.
  - Content: with the legacy halves declared, expected channels and missing-channel summaries are equal to the reference. Every reference PDF line appears in order in ours. The Excel fitted rows (5 columns) are equal to the reference rows.
  - Without declarations the 8 legacy-skipped minimodules are reported as not observed and listed in the summary/PDF; a declared-but-hit minimodule is reported.
  - Bounded storage: 4 files vs 1 → identical accumulator bytes, traced peak growth < 256 KiB, exactly 4× histogram/flood/slab totals (the reference keeps every side's list).
  - Rejections: fixed/empty/duplicate/untyped inputs, bad options/limit/duration/source mode, unmapped channel, truncated file, missing `en_min_ch`, missing results parent. Cancellation raises `QCCancelled`.
  - Tracked-module import audit.

  **Regressions:** manager modes 155; Inspector 59/59, slab 16/16, unpopulated 6/6; compile PASS. Shared helpers are unchanged (`git diff -- src` empty; only the two new files). The four T1 primary fingerprints are unchanged.

  **Still pending:**
  - Real Cornell source/no-source acquisitions and an operator review of the PDF/plots (T19).
  - Processing is sequential and per-pair Python, like each reference worker; file-level pooling and CLI/RunStore publication are T11/T12. `reportlab` is still undeclared in `process_petsys.yml` (T11).
  - The Cornell config must declare the half-populated minimodules to reproduce the legacy expectation; until then they are reported as not observed.

- [x] **T11 — Headless processing CLI and declared dependencies** (FR-2, FR-4, FR-9, FR-12, FR-14, FR-16). Add tracked internal CLI dispatch for calibration/LM/QC, request/result manifests and structured progress; declare only needed missing Python dependencies.

  **Depends on:** T2–T5, T8–T10.

  **Done when:** `python scripts/petsys_manager_check.py --cli` launches each action through `sys.executable` with a fixture request and validates actual output/result paths. Invalid requests, corrupted inputs and numerical failure exit nonzero without success results; output paths with spaces/metacharacters remain literal. Source audit proves runtime imports/calls do not reference ignored scripts/sibling GUI; processing modules import headlessly. Dependencies cover actual imports, including QC's `reportlab`.

  **Verified 2026-10-01:** new tracked `src/cornell/cli.py`; `python -u -m src.cornell.cli {calibrate,listmode,qc} --request R --result S` is exactly the existing `commands.build_internal` argv.
  - **Request (schema 1):** bounded JSON (4 MiB, unique keys, finite numbers) with exact keys per action, all required and none defaulted:
    - common: `processing_root`, `processing_config`, ordered typed `inputs`, `files`, `options`, `outputs`;
    - calibrate: `cog_limits`; `num_regions`, `side_limit` (or null), `batch_records`; `encal`, `sidecar`, `plot` (or null);
    - listmode: calibration (+ optional sidecar/boundaries), COG/DOI limits, pair/region maps, the complete `LMMetadata`, `batch_records`, `debug`, `resume`; job `directory`;
    - qc: `plots`, `slabs` (needs plots), `source_mode` (`with`/`without`/null), `acquisition_time_s` (or null), `pair_limit`; results `directory`.

    Paths must be absolute and are used literally. Outputs must not exist (a resumed LM job directory excepted), need an existing parent, and may not repeat each other, the request or the result. Inputs go through T5 `select_inputs`/full validation inside the T8–T10 modules.
  - **Result:** published once by exclusive temp + hard link, never replaced, after output verification:
    - status `succeeded`/`failed`/`cancelled`, exit code, request path/SHA-256, timestamps, interpreter, errors and `error_kind`;
    - outputs (kind, absolute path, size, SHA-256) and counts summary only on success.

    Verification per action:
    - the `.encal` + sidecar must re-load through `load_calibration`;
    - the LM size must equal header + records × 24;
    - every QC file must match its `qc_summary.json` digest.

    `read_result` accepts a success only if every output is unchanged. Exit codes: 0 succeeded, 1 input/processing/numerical/verification failure, 2 invalid request or result path, 3 cancelled (SIGTERM/SIGINT set a flag the modules poll; a cancel arriving after processing still publishes no success). Failed/cancelled results list no outputs; files already written stay as unlisted evidence.
  - **Progress:** `@petsys-event {json}` stdout lines (sequence, kind `started`/`progress`/`output`/`finished`, file index, records read/written), each < 4096 characters; intermediate progress is throttled to 0.5 s per file. `parse_event` reads them back, also from the runner's `[stdout] ` log lines. Additive hooks: optional `progress` in `calibration.calibrate`/`sample_file` (per batch) and one final per-file `progress` call in `qc.sample_file`; numerics unchanged.
  - **Dependencies:** `process_petsys.yml` gains only `reportlab==4.4.9` (installed version; imported lazily by `qc_report`). Every other third-party import of the loaded tracked modules is already declared: numpy, scipy, matplotlib, numba, openpyxl, pyyaml, pandas (`src.utils`) and tqdm (`src.read_fixed`). No environment update was run.

  Environment interpreter `-X utf8 scripts/petsys_manager_check.py --cli` → **PASS 11/11**. Real child processes are built with `build_internal` and run without `MPLBACKEND`/`DISPLAY`, on synthetic fixtures under a directory named `cli ; & $HOME %PATH% [x] (y) 'q' #é`:
  - Calibrate (2 fixed files): the `.encal`, sidecar and PNG land at the literal requested paths; `.encal` text, status counts and accepted sides equal in-process `calibrate`.
  - Listmode (2 files, debug): LM SHA-256, name, record count and rejections equal in-process `generate_listmode`; outputs are the LM, provenance, job record and debug plots.
  - QC (plots + slabs, 60 s with source): the output file set equals in-process `write_report`. Summary totals, findings, minimodule fits, expectation, cuts, sampling and source metadata are equal. Slab-dependent values are excluded because the unseeded Python `random` single-time-channel draw differs in the child.
  - Events: contiguous sequence, started…finished, progress with records and an output event.
  - Invalid requests → exit 2, no outputs and no results directory. Cases: 18 schema/type/key/path cases and an existing results directory (left empty), plus duplicate-key, NaN, non-JSON and oversized bytes as children, plus incomplete LM metadata. An existing result is left byte-identical, a relative or parentless result writes nothing, and an unknown action fails argparse with 2.
  - Corrupted inputs → exit 1, `input_or_processing`, no success. Cases: truncated compact QC file, unmapped channel, compact file declared fixed, missing input, LM second file truncated (no merged LM; job record kept).
  - Numerical failures → exit 1. Cases: all keys < 50 samples (no `.encal`/sidecar/plot); forced fit exception (`unexpected`, traceback on stderr, no results directory); plot exception after the report started (directory kept, unlisted, no summary); forced calibration read-back failure.
  - Cancellation (pre-set, mid-QC, mid-LM, after processing) → exit 3, `cancelled`, no outputs.
  - Result contract: one flipped or truncated byte in an output rejects the success; forged failed/empty/nonzero/other-schema results are rejected; no `.partial` remains.
  - Source audit (AST, docstrings excluded) of `src/cornell/*.py` and `src/petsys_manager/*.py`: every file is tracked or the intended new `cli.py`. There are no imports of scripts/sibling GUI/docopt/natsort/colorama/Tk/runpy and no strings naming `scripts_cornell`/`scripts_imas`/`gui_cornell`/`scripts/`. `src/cornell` uses no subprocess/multiprocessing/importlib/`os.system`-type calls; the only `shell=` is the runner's `shell=False`.
  - Headless runtime: a child process runs QC with plots and slabs. The matplotlib backend is Agg; no Tk, PyQt or pyqtgraph is loaded; every repository module loaded is tracked `src/` (or `cli.py`). colorama is loaded only by the shared reader's tqdm on Windows.
  - Dependencies: AST imports of every loaded tracked module, lazy ones included, map to declared `process_petsys.yml` entries. The diff from HEAD is exactly the added `reportlab==<installed version>` line.

  **Regressions:**
  - numeric `--formats --calibration --listmode --qc` 80/80;
  - manager modes 155;
  - T1 reference `--synthetic` 17/17 with fingerprints unchanged;
  - Inspector 59/59, slab 16/16, unpopulated 6/6;
  - compile PASS.

  **Still pending:**
  - Linux SIGTERM delivery to a real CLI child: Windows has no SIGTERM, and WSL lacks the numerical stack. This goes to T17 Linux, then T19.
  - Request generation from settings snapshots, runner/result wiring and stage graphs (T12).
  - File-level worker pooling.
  - Real-data runs (T19).

## Workflow and separate GUI

- [x] **T12 — Manual actions and fail-closed pipelines** (FR-5, FR-7, FR-9–FR-11, FR-13–FR-14, FR-16). Add workflow coordinator with settings snapshots and manifest-derived exact artifacts. Implement acquire/fixed/calibrate/LM and acquire/compact/QC stage graphs plus manual actions; preflight the entire chosen graph.

  **Depends on:** T3–T7, T11.

  **Done when:** `python scripts/petsys_manager_check.py --workflows` runs fake successful graphs and injects spawn error/nonzero exit/invalid or missing output/STOP at every stage. No successor starts after failure/cancel; old similarly named files cannot substitute for new output. Only one foreground workflow starts; all paths end in consistent controls/state and unchanged persistent settings. QC uses correct 60/180 s presets and compact descriptors, full pipeline uses fixed coincidence, manual group action never feeds LM. Stage logs/manifests distinguish completed processing from QC findings.

  **Verified 2026-10-01:** new tracked `src/petsys_manager/workflow.py`.
  - **Graphs:**
    - manual `acquire`, `convert` (fixed/compact coincidence or fixed group), `calibrate`, `listmode` and `qc_analyze` are single stages;
    - `qc` is acquisition (60/180 s preset) → compact coincidence conversion → QC;
    - `pipeline` is acquisition → fixed coincidence conversion → calibration → listmode.
  - **`prepare(settings)`:** checks the whole graph from the immutable T2 preflight snapshot before anything is created or launched:
    - the destination exists;
    - every argv contract builds (acquisition, bias-off, conversion, internal CLI);
    - the processing YAML/map loads, for conversion-output validation too;
    - COG/DOI limits, pair/region maps, the manual calibration file and LM metadata load and are complete;
    - manual inputs pass T5 `select_inputs`;
    - the QC preset and the pipeline/QC formats are right.

    A live graph also needs `prerequisite()` (DAQD initialized) at `start`.
  - **`WorkflowCoordinator`:**
    - one foreground workflow at a time (`WorkflowBusy`);
    - one worker thread and one `RunStore` run per workflow, in `data_dir` for acquire/convert/qc/pipeline, `calibration_dir`, `lm_dir` or `report_dir` for manual processing;
    - STOP (`handle.stop()`/`close`) cancels the running stage, the acquisition and its retries, and every later stage;
    - sinks only receive `RunEvent`s (never Tk).
  - **Exact artifacts:**
    - Acquisition is the T7 service; the conversion consumes exactly its successful attempt's `.rawf` prefix.
    - Conversion outputs come from `discover_ldat` against the pre-launch inventory of a fresh attempt directory. A pre-existing match fails the stage. Every nonempty file is fully T5-validated (selected map, hit limit) and gets a validated descriptor. Zero-length splits are recorded disposable, kept and excluded.
    - Processing stages write an exclusive `request.json` from the snapshot (inputs are the predecessor's exact descriptors; listmode gets the calibration stage's `.encal` + sidecar). They run `build_internal` and accept only a `cli.read_result` whose request path/SHA-256 and action match, with every output (hash-verified) inside the attempt directory and the required kinds present.
  - **Failure handling:** failure, launch error, invalid/missing output or STOP ends the graph and finalizes the run failed/cancelled; files are kept as unvalidated `partial_output`. QC success records `process: completed` and the findings separately; the outcome message says the findings are not a detector verdict.

  Environment interpreter `-X utf8 scripts/petsys_manager_check.py --workflows` → **PASS 9/9**. The real `CommandRunner`/`AcquisitionService` run against a check-only backend: fake acquisition/converter/`set_bias` children; converters copy seeded T9/T10 synthetic LDAT.
  - Pipeline:
    - argv: `--time` 10, `convert_raw_to_coincidence --writeBinaryFixed -i <this attempt's RAW prefix>`;
    - outputs: `_1`/`_2` validated fixed coincidence; `_9` (empty) disposable, kept and not an output; an old look-alike in `data_dir` is unused;
    - downstream: the calibration and listmode requests use exactly those inputs; listmode uses the calibration stage's `.encal`/sidecar, the profile LM metadata and batch 1000.
  - QC with/without source: `--time` 60/180, `--writeBinaryCompact`, compact QC inputs, request options exact (plots, slabs, source, duration, pair limit). Findings are separate from `process: completed`, and no "PASS" appears in manifests or events.
  - Real CLI children (pipeline: calibrate → listmode; QC 180 s): LM provenance names the calibration stage's `.encal`; records > 0; inputs exact; QC summary in the stage's actual results directory, 1200 records, without/180 s; progress events forwarded.
  - Faults: spawn error, nonzero exit, invalid output, missing output, STOP during, and STOP right after each stage, at all 4 pipeline and 3 QC stages (42 cases).
    - No successor launches.
    - The status is `launch_error`/`failed`/`cancelled`, never success; the run manifest is terminal with no partial attempts.
    - An invalid/missing acquisition is retried (2 attempts), then stops.
    - Afterwards a new workflow succeeds.
  - Stale outputs: a pre-existing converter-named file, a foreign successful CLI result (old request digest), an output outside the attempt and a hash-mismatched output all fail.
  - Single workflow: a second `start` raises `WorkflowBusy`; `close` stops the blocked acquisition, `set_bias` bias-off runs and the run ends `cancelled`; then a new workflow succeeds.
  - Preflight, before any run directory or launch: corrupted pair map, missing destination, compact pipeline, wrong QC preset, convert without processing YAML, LM with group inputs, incomplete LM metadata, untyped settings, DAQD action, prerequisite not met.
  - Manual actions:
    - acquire, calibrate, listmode (profile calibration, no sidecar) and offline QC (source/duration not recorded) run one stage in their destination;
    - fixed group conversion (RAW path with spaces/metacharacters) runs only `convert_raw_to_group`, gives group descriptors, and is refused by listmode `prepare`/preflight.
  - Every run:
    - the profile YAML, processing YAML/map/limits/maps/inputs are byte-identical and the settings snapshot is unchanged and recorded in the manifest;
    - events are strictly ordered, carry this run's identity and bracket each stage;
    - `workflow.py` imports without Tk/numba/matplotlib.

  **Regressions:**
  - manager modes plus `--cli --workflows` 175;
  - numeric 80/80;
  - T1 reference `--synthetic` 17/17;
  - Inspector 59/59, slab 16/16, unpopulated 6/6;
  - compile PASS.

  The local T11 CLI check was made commit-state independent: uncommitted runtime files are audited, and the dependency diff is taken against the revision before `reportlab` was added.

  **Design note:** a pipeline/QC run keeps every stage under its run directory in `data_dir`, because the run store accepts only outputs inside its attempts. `calibration_dir`, `lm_dir` and `report_dir` are destinations for the manual actions only; the GUI must show the recorded actual paths (T16).

  **Still pending:**
  - Linux `LinuxProcessBackend` runs of these graphs (T17), then live hardware (T19).
  - GUI wiring (T13–T16).
  - Empty-LDAT removal: `remove_empty_ldat` exists but is not enabled by default.
  - File-level worker pooling.

- [x] **T13 — Manager shell, profiles and input UI** (FR-1, FR-3, FR-7, FR-16). Add separate launcher/GUI shell, five tabs, main-thread event poller, settings/profile controls and optional safe asset loading. Do not instantiate another Inspector root or change its code.

  **Depends on:** T2, T5, T12.

  **Done when:** `python -m compileall -q exe_programs/PETsysManager.py exe_programs/petsys_manager_gui.py src/petsys_manager src/cornell` succeeds. `python scripts/petsys_manager_gui_check.py --shell` with withdrawn window/fake backend checks tab names/log/profile reload, relocated module-relative assets, separate INI/YAML controls and prerequisite reasons without private paths. Instrumented widget access occurs only on the main thread; no hardware/processes launch at startup. Inspector imports/entry point remain independent.

  **Verified 2026-10-01:** new tracked files:
  - `exe_programs/PETsysManager.py`: thin launcher (repository import path, `freeze_support`, `--profile PATH`), mirroring `LDATInspector.py`.
  - `exe_programs/petsys_manager_gui.py`: `PETsysManager(root, session)` on a caller-owned root.
    - The five reference tabs: System Setup & Acquisition, RAWF to LDAT Conversion, LDAT Processing, LM File Generation, System Quality Control.
    - A bounded "Output Log:" (profile `log_tail_lines`, appended in one widget update per poll).
    - The reference control layout. DAQD, Initialize, Acquire, pipeline/STOP, conversion, calibration, LM and QC buttons stay disabled until T14–T16 connect them.
    - A "Prerequisites" panel per tab: a specific reason per action from read-only `settings.preflight`, or "prerequisites met (control not connected yet)".
  - `exe_programs/assets/onco_logo.jpeg`: optional logo, a byte-identical copy of the sibling's `imgs/onco_logo.jpeg`.
  - `src/petsys_manager/session.py`: toolkit-free `ManagerSession`.
    - Profile open/load/save via T2 `load_profile`/`save_profile`.
    - A coalescing `petsys-preflight` worker thread whose results carry a generation; the GUI shows only the generation it is awaiting.
    - A `SimpleQueue` that the window drains with `after` (100 ms, ≤ 200 events per tick).

  Profile controls:
  - Profile path, Browse, Reload, Save.
  - Separate fields: PETsys tools folder, PETsys INI (DAQ/conversion), processing YAML (cal/LM/QC), optional processing root, DAQ type/cards/socket, every destination, COG/DOI limits, calibration file and LM pair/region maps. Shared fields (COG limits, report destination) use one variable across tabs.
  - Run inputs: acquisition time, hardware trigger, RAW file, split count, QC source preset and plots/slabs; slabs need plots.
  - Edits never autosave and keep fields the window does not show: safety, limits, capabilities, LM metadata, shared memory.
  - A missing profile leaves defaults and writes nothing. A broken profile is reported and left byte-identical. Save replaces only a valid manager profile.
  - The relative map root defaults to the module's checkout, so a moved checkout needs no `process_petsys` folder selection.
  - The logo resolves relative to the module; a failure is logged and never blocks startup.

  Environment interpreter `-X utf8 -m compileall -q exe_programs/PETsysManager.py exe_programs/petsys_manager_gui.py src/petsys_manager src/cornell` → **PASS**. `-X utf8 scripts/petsys_manager_gui_check.py --shell` → **PASS 7/7**. Withdrawn real windows with a fixture probe (Linux platform, marker tools/cards/files):
  - **Tabs/log/startup:**
    - tab names equal the reference;
    - startup logs the version/checkout and the loaded profile;
    - all action buttons are disabled; a complete fixture shows DAQD/init/acquire/pipeline/QC met, conversion needs the RAW file and calibrate/LM/offline QC need the exact inputs;
    - 1,500 queued lines leave exactly the last 1,000;
    - closing joins the preflight thread.
  - **No launches at startup:** while the window is built, `subprocess.Popen`, `os.system`, `posix_spawn`, `multiprocessing.Process.start`, `CommandRunner`, `DaqdService`, `AcquisitionService` and `WorkflowCoordinator` are patched to fail, and none is called. The window creates no second Tk root, and the only new thread is `petsys-preflight`.
  - **Profile reload/save:**
    - entries show the saved profile;
    - edits mark "unsaved edits" without touching the file;
    - Reload restores the file values;
    - Save writes the edited data folder/processing root and preserves hidden fields;
    - a broken file is reported and kept byte-identical, with the window values kept;
    - Save refuses to replace the processing YAML, which stays byte-identical;
    - a new path is created exclusively;
    - a missing explicit profile writes nothing;
    - a broken startup profile shows "not loaded (see log)".
  - **Separate INI/YAML:** distinct variables and entries. A missing INI is named only for init, acquire, pipeline, both conversions and QC; a missing YAML only for pipeline, calibrate, LM, QC and offline QC. Neither reason mentions the other. An explicit processing root moves the YAML's relative map lookup.
  - **Reasons without private paths:**
    - the default profile with the real probe on this workstation gives every action specific bulleted reasons (platform, tools folder with the tools it needs, destinations, LM metadata missing from the profile);
    - neither the window text nor the GUI/session source contains `/home/sie` or `nvmDisk`;
    - an invalid acquisition time affects only acquire/pipeline, a nonpositive duration or split count gives the option error, and a missing RAW file is named;
    - an invalid card path marks every action with the profile error until fixed.
  - **Main thread only:** the root's Tcl interpreter is proxied, so every widget call records its thread.
    - More than 100 calls, none off the main thread.
    - A blocked check keeps the poller draining log lines.
    - A stale older check (INI reason) is received and discarded; only the newer one (data folder reason) is shown.
    - Probe calls happen only on `petsys-preflight`.
    - Negative control: a `rogue` worker calling `log` is caught.
  - **Relocated assets:** `exe_programs/` (with assets) and `src/petsys_manager` are copied to `re lo ; [x]` and started from an unrelated working directory.
    - The module, logo and checkout all resolve inside the copy, and every manager module loaded is from it.
    - The YAML's relative `maps/...` resolves in the copy, and the original checkout appears nowhere.
    - Without `assets/`, startup logs "Optional logo not shown" and keeps all tabs.
    - The relocated launcher's `--help` exits 0.
  - **Inspector independence:**
    - importing the manager loads no `ldat_inspector`/matplotlib modules; the session imports no Tk;
    - importing the Inspector GUI loads no manager module;
    - no manager file imports Inspector code;
    - Inspector files are unchanged against HEAD and have no untracked additions;
    - the compile check passes.

  **Regressions:**
  - manager modes plus `--cli --workflows` 175;
  - numeric 80/80;
  - T1 reference `--synthetic` 17/17;
  - Inspector `--selftest` 59/59, slab 16/16, unpopulated 6/6, Inspector hidden GUI `ldat_gui_check.py` 29/29.

  Screenshots of all five tabs on a fixture profile were reviewed for layout only; the operator review of the real UI is still due on representative acquisitions.

  **Still pending:**
  - DAQD/initialize/acquisition/STOP/asynchronous close and safety-limit editing (T14).
  - Conversion format controls and exact file selection (T15).
  - Processing/pipeline/QC actions, LM metadata fields and results paths (T16).
  - Linux run (T17) and operator review (T19).

- [x] **T14 — DAQD/acquisition/STOP/close UI** (FR-5–FR-8, FR-16). Connect GUI controls to owned backend states, initialization and monitored acquisition; block conflicting actions and implement asynchronous close. Also FR-19/FR-20: show RAW-started, growth-passed ("file growing") and live size/rate messages with a stall warning; show a persistent "SiPM bias state unknown" warning; expose the safety limits, the frame-loss limit included, as editable settings; on close, let the acquisition's bias-off finish before stopping DAQD.

  **Depends on:** T6–T7, T12–T13.

  **Done when:** `python scripts/petsys_manager_gui_check.py --acquisition` proves failed init/dead daemon never unlock acquisition, readiness is backend-driven, retry wait remains responsive, stale events cannot restore active buttons and STOP during acquisition/retry prevents new attempts. Fake subprocess close/cancel leaves no owned children; no cleanup of global socket/shm is invoked. GUI event polling remains live while shutdown completes/failure is shown.

  **Verified 2026-10-01:**
  - **`src/petsys_manager/session.py`** (extended), toolkit-free:
    - `start_daqd`/`stop_daqd`/`initialize`/`start_workflow`/`stop_workflow`/`shutdown`.
    - Each runs preflight plus the T6 `DaqdService` or T12 `prepare` + `WorkflowCoordinator` in its own worker thread.
    - Results go back as queue events: DAQD statuses (with the service revision), initialization outcomes, refusals, workflow `RunEvent`s and a token-carrying `WorkflowResult`.
    - Services are created on first use, so startup still launches nothing.
    - Acquisition is gated by `DaqdService.acquisition_ready` on the frozen snapshot.
    - STOP is sticky, including before the coordinator has started.
    - `shutdown(timeout)`:
      - sends STOP and waits for the workflow, its bias-off included;
      - only then joins the DAQD/initialization requests (initialization is cancelled by the same flag) and calls `DaqdService.close`.
      - If the workflow is still finishing at the timeout, DAQD is left running and a failed `ShutdownResult` asks the operator to close again.
  - **`exe_programs/petsys_manager_gui.py`:**
    - DAQD checkbox, Initialize System, Acquire Data and STOP are connected.
    - Control state is recomputed after every event batch, from:
      - the newest DAQD status (an older revision is dropped);
      - the prerequisite generation in view;
      - the window's own workflow token (an older `workflow_done` is logged as stale and ignored; workflow events from other run IDs are ignored);
      - pending requests and the bias warning.
    - The checkbox always shows the backend state, never the click. A DAQD status line shows state, pid, initialization and the service message.
    - **Live status (FR-20):** attempt n/m, "RAW file started writing", "File growing: growth check passed", size and MB/s per poll, a red "RAW file stopped growing" warning after the check passed, abort/retry/bias-off messages and the final status with the run directory.
    - **Unknown bias (FR-19):** a persistent red banner. Acquisition stays blocked until the operator presses "I checked the bias", and the confirmation is logged.
    - **Safety limits (FR-20):** editable "Acquisition Safety Limits" fields (startup, growth window/minimum in MB, poll, max frame loss %, attempts, retry delay, stop grace) are saved in the profile and validated through `AcquisitionSafety`. Invalid values mark every action with the profile reason. Runs record the values used in the T12 settings snapshot.
    - **Close:** immediate when idle; otherwise asynchronous through `session.shutdown`, with controls disabled and polling live. A failure is shown in the status line and log, and the window stays open.
  - **`src/petsys_manager/workflow.py`:** forwarded `acquisition_*` events are no longer logged a second time (`_emit(..., log=False)`); events and manifests are unchanged.

  Environment interpreter `-X utf8 scripts/petsys_manager_gui_check.py --acquisition` → **PASS 6/6**; `--shell --acquisition` → 13/13 in three consecutive runs.
  - **Setup:** withdrawn real windows run the real `DaqdService`, `WorkflowCoordinator`, `AcquisitionService` and `CommandRunner` against fake children:
    - a DAQD daemon recording TERM;
    - an in-memory socket/shm table with no delete operation and a gated protocol reply;
    - `init_system` with a chosen exit code;
    - `acquire_sipm_data` behaviours ok/nodata/fail/block/grow;
    - a gated or failing `set_bias`.

    The Tk interpreter is proxied, so every widget call records its thread.
  - **Backend-driven readiness:**
    - Before DAQD only its checkbox is enabled.
    - While the socket exists but DAQD does not answer, the state stays STARTING for 0.5 s with Initialize/Acquire disabled.
    - READY enables only Initialize.
    - A failed `init_system` (exit 1) logs "acquisition stays locked" and keeps Acquire disabled; a successful one enables it.
    - Daemon death gives FAILED with Initialize/Acquire disabled. Re-injecting the earlier READY+initialized status and initialization outcome changes nothing, and no acquisition was launched.
  - **Progress/STOP/bias-off:** with a growing then stalling RAW file:
    - the status shows attempt 1/3, RAW started, "File growing", MB/s and the stopped-growing warning;
    - during the run only STOP is enabled;
    - STOP gives "Acquisition cancelled", the last tools launched are `acquire_sipm_data` then `set_bias`, there is no second attempt and every owned non-daemon child has exited;
    - controls return to Acquire enabled;
    - the manifest is `cancelled` and records the edited limits: the frame-loss limit set to 2.5 in the UI, after "abc" and "101" were rejected with specific reasons.
  - **Retry wait:**
    - A no-data attempt with a 30 s retry delay shows "Retrying" while the UI still drains log lines.
    - STOP ends the run in under 10 s, with "Stopped during the retry delay; no further attempts" and one launch.
    - STOP of a running attempt gives TERM and bias-off, and no further attempt.
  - **Stale events:** during a running acquisition, an injected older `workflow_done`, an older DAQD revision and another run's `workflow_finished` are ignored, with STOP the only enabled control.
  - **Unknown bias:** a failing bias-off after STOP shows the banner and keeps Acquire disabled until confirmed.
  - **Failure is not success:** a nonzero acquisition exit reports "Acquisition failed" and is not retried.
  - **Existing resources:** a pre-existing socket/shm blocks DAQD start with "Existing DAQD resources block start", launches nothing and leaves the table unchanged.
  - **Close during acquisition with a slow bias-off:**
    - With a 0.6 s timeout, controls are disabled, polling stays live (marker lines shown during and after) and "Close incomplete" is shown; the window stays open and DAQD is not stopped.
    - After the bias-off finishes, a second close completes: `set_bias` precedes `daqd TERM`, every owned child (daemon included) has exited and the manager threads have ended.
    - No `os.unlink`/`remove`/`rmdir`/`rmtree` call names the socket or shared memory, and the fake leftovers are reported, not removed.
  - **Main thread:** no off-thread Tk call in any test.

  **Regressions:**
  - `--shell` 7/7, now with only the DAQD checkbox enabled at startup;
  - manager modes plus `--cli --workflows` 175;
  - numeric 80/80;
  - T1 reference `--synthetic` 17/17;
  - Inspector `--selftest` 59/59, slab 16/16, unpopulated 6/6, Inspector hidden GUI 29/29;
  - compile PASS.

  **Still pending:**
  - Pipeline/QC controls (T16) and conversion (T15).
  - Linux process-group runs of the real backend from the GUI (T17).
  - Installed-tool behaviour, real bias-off and operator review of the live UI (T19).

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

  **Done when:** `python scripts/petsys_manager_reference_check.py --real --manifest <operator-baseline.json>` records input/settings/version fingerprints and parity for fixed calibration/LM and compact QC at predefined tolerances. Operator records live initialization, monitored acquisition (including that `set_bias --power off` exists and switches bias off after a STOP/abort), both coincidence conversions, group conversion/manual group calibration where used, LM compatibility with its consumer, 60/180 s QC with plot/slab options, complete pipeline, failure/STOP/retry and close. Existing data survives; no owned children remain; actual result locations and scope match reports. Representative Cornell/IMAS Inspector regression results and the FR-by-FR pass/fail table are recorded. Only all completion criteria passing permits `Status: shipped`; sibling retirement remains a separate owner decision.

  **Verified:** pending; requires Cornell Linux hardware/operator and representative data, not available as a demonstrated check on this workstation.

## Planning validation

**Verified 2026-09-30:** PowerShell structural check passed: FR-1–FR-18 uniquely numbered and mapped in the plan; T1–T19 consecutively numbered, each with `Done when:` and pending `Verified:`; exactly one approved/in-progress spec; no trailing whitespace in the three documents. No application/numerical/hardware checks were run for this documentation-only change. Creating these documents does not complete T1 or any other implementation task.
