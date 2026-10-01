# Tasks 003 — PETsys Manager migration

Spec: [`spec.md`](spec.md). Architecture: [`plan.md`](plan.md). Owner requested plan/tasks and T1 on 2026-09-30, then continuation with additive migration and T3, continued spec003 work and explicitly T5. T1–T10 are complete; T11 headless processing CLI is next. Live hardware and GUI wiring have not started.

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

- [ ] **T14 — DAQD/acquisition/STOP/close UI** (FR-5–FR-8, FR-16). Connect GUI controls to owned backend states, initialization and monitored acquisition; block conflicting actions and implement asynchronous close. Also FR-19/FR-20: show RAW-started, growth-passed ("file growing") and live size/rate messages with a stall warning; show a persistent "SiPM bias state unknown" warning; expose the safety limits, the frame-loss limit included, as editable settings; on close, let the acquisition's bias-off finish before stopping DAQD.

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

  **Done when:** `python scripts/petsys_manager_reference_check.py --real --manifest <operator-baseline.json>` records input/settings/version fingerprints and parity for fixed calibration/LM and compact QC at predefined tolerances. Operator records live initialization, monitored acquisition (including that `set_bias --power off` exists and switches bias off after a STOP/abort), both coincidence conversions, group conversion/manual group calibration where used, LM compatibility with its consumer, 60/180 s QC with plot/slab options, complete pipeline, failure/STOP/retry and close. Existing data survives; no owned children remain; actual result locations and scope match reports. Representative Cornell/IMAS Inspector regression results and the FR-by-FR pass/fail table are recorded. Only all completion criteria passing permits `Status: shipped`; sibling retirement remains a separate owner decision.

  **Verified:** pending; requires Cornell Linux hardware/operator and representative data, not available as a demonstrated check on this workstation.

## Planning validation

**Verified 2026-09-30:** PowerShell structural check passed: FR-1–FR-18 uniquely numbered and mapped in the plan; T1–T19 consecutively numbered, each with `Done when:` and pending `Verified:`; exactly one approved/in-progress spec; no trailing whitespace in the three documents. No application/numerical/hardware checks were run for this documentation-only change. Creating these documents does not complete T1 or any other implementation task.
