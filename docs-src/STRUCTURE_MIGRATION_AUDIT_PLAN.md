# NVIDIA/Efimon structure migration audit plan

Date: 2026-10-27  
Scope: audit the restructuring of this NVIDIA profiler checkout against the
layout agreed in this session, including the later removal of root Python
scripts. This document is an audit plan, not a declaration that every check
has passed. Commands run from the repository root unless stated otherwise.

## 1. Requirements and boundaries

The latest user instructions take precedence over earlier compatibility plans:

- NVIDIA library headers belong under `include/efimon/gpu/nvidia-gpu/`.
- NVIDIA library implementations belong under `src/efimon/gpu/nvidia-gpu/`.
- GPU session orchestration belongs under `src/tools/gpu/`.
- Daemon and launcher mains belong under `src/tools/`.
- Analyzer entry scripts and their implementation belong under `analyzers/`.
- No analyzer `.py` files remain directly in the repository root. Root script
  wrappers were explicitly removed after the initial migration.
- Installed command names, capture schemas, report filenames, numerical
  behavior, and existing results must remain compatible.
- Apply the supplied Python and C++ style rules, respecting their precedence
  for existing local conventions and avoiding unrelated reformatting.

This checkout does not contain the rest of Efimon: its Observer interfaces,
existing `gpu/nvidia` implementation, CPU backends, and other CLI binaries are
not available here. Directory organization alone does not establish integration
with those components. Do not create placeholder backends or claim that full
Efimon integration has been verified.

The Efimon reference tree is available at `/home/kevinl/efimon` and was used
for the structural comparison recorded in section 5 (STRUCT-02 to STRUCT-07).

## 2. Current inventory and expected placement

The paths below were inspected while preparing this plan. Their existence does
not establish runtime correctness.

| Component | Current location | Audit expectation |
| --- | --- | --- |
| Preload collector | `src/efimon/gpu/nvidia-gpu/process-metrics.cpp` | Owns collection lifecycle; review the unresolved separation below |
| CUPTI sampling helper | `include/efimon/gpu/nvidia-gpu/pm-sampling.hpp` | Includes resolve from the project include root |
| CUDA error helpers | `include/efimon/gpu/nvidia-gpu/cuda-api-checks.hpp` | Existing callers resolve the renamed header |
| PM sampler executable | `src/tools/gpu/pm-sampling.cu` | Tool entry point stays in the tools layer |
| GPU session and worker | `src/tools/gpu/session.{hpp,cpp}`, `worker.{hpp,cpp}` | Session orchestration stays outside collection code |
| GPU IPC helpers | `src/tools/gpu/ipc.hpp` | Daemon and launcher use the same protocol |
| CLI mains | `src/tools/daemon.cpp`, `launcher.cpp` | Installed executable names remain stable |
| Service template | `misc/nvidia-process-metrics-daemon.service.in` | Meson and source packages include it |
| Analyzer scripts | `analyzers/*.py` | All eight scripts run from their new location |
| Python package | `analyzers/energy_attribution/` | Installed package remains `energy_attribution` |
| Source import compatibility | Root `energy_attribution` relative symlink | Imports resolve to the canonical package; no duplicate implementation |
| Python configuration | Root `pyproject.toml` | Discovers packages under `analyzers/`; remains the single configuration |
| Manual build outputs | `nvidia/` | Generated artifacts only; scripts and documentation agree on this location |

### Unresolved structural difference

The earlier proposed tree included separate `telemetry.hpp/.cpp` and
`kernel-activity.hpp/.cpp` components. The current implementation still combines
NVML telemetry, CUPTI activity, CSV writing, and preload lifecycle in
`process-metrics.cpp`. The previous migration moved this implementation without
performing that separation.

- [ ] Record whether that component separation remains required.
- [ ] If required, define interfaces and state ownership before extracting code.
- [ ] Preserve callback lifetime, synchronization, sampling shutdown, and final
  CSV writes during extraction.
- [ ] Verify the extraction on a real CUDA build and capture before closing it.

Do not mark the full originally proposed component structure complete while
this difference remains unresolved. Keeping the PM executable's `main` in
`src/tools/gpu/` is intentional; collector helper code belongs in the library
subsystem.

## 3. Evidence and status tracking

Use these statuses for each audit item: **pending**, **pass**, **fail**, or
**unavailable**. An unavailable tool or GPU is not a passing result. Record the
exact command, exit status, environment, and evidence location for each check.

Earlier session checks reported the following results. They are historical
baseline evidence and should be rerun against the final tree before completion:

| Check | Earlier result | Remaining coverage |
| --- | --- | --- |
| Python regression suite after root script removal | 22 tests passed | Rerun after any subsequent changes |
| Eight relocated script entry points | `--help` worked from `/tmp` | Exercise their actual analysis paths |
| CUDA-free daemon and launcher | Build and staged installation passed | Does not compile the collector or PM sampler |
| Native package source archive | CUDA-free build from extracted archive passed | Actual RPM/DEB packages and CUDA-enabled archives |
| Python wheel | Built, installed under `/tmp`, eight commands loaded | Repeat after final packaging changes |
| Style | Available Pylint check passed for wrappers before root removal | Full scoped style checks remain pending |
| CUDA compilation and hardware capture | Unavailable | Required on a suitable NVIDIA machine |

## 4. Audit sequence

### A. Establish a safe baseline

- [ ] Record the current commit, working tree, tool versions, and Python interpreter.
- [ ] Distinguish pre-existing user changes from migration changes. This working
  tree already contained modified code, untracked package files, and results.
- [ ] Record the existing `results/` status and verify the audit does not alter it.
- [ ] Keep generated captures, logs, stages, and packages outside existing results.
- [ ] Do not reset files, update golden fixtures, or stage unrelated changes as
  part of the audit.

```bash
git rev-parse HEAD
git status --short
python3 --version
meson --version
```

### B. Verify placement and references

- [ ] Confirm the inventory in section 2 against the actual tree.
- [ ] Confirm no `.py` files remain directly in the repository root.
- [ ] Check all eight analyzer scripts delegate to package implementations.
- [ ] Check the source import symlink is relative and resolves correctly.
- [ ] Search tracked and untracked source/documentation files for obsolete paths.
- [ ] Classify old-path matches: historical migration documentation and generated
  artifacts are different from active build or usage instructions.
- [ ] Review the rename mapping so no modified or untracked implementation was
  omitted during the move. Git may show deletions and new paths until staging;
  that alone is not evidence of lost content.

```bash
rg --files -g '*.py' -g '!results/**' -g '!build*/**' -g '!dist/**'
rg -n 'nvidia/nvidia-process-metrics\.cpp|nvidia/pm_sampling_simple\.cu|tools/nvidia-process-metrics' README.md docs-src packaging src include build_gpus.sh
rg -n 'python3? (analyze_|bayesian_energy_model\.py|plot_pm_sampling\.py)' README.md ANALYSIS_GUIDE.txt analyzers
readlink energy_attribution
git diff --check
```

The obsolete-reference searches should have no active broken references.
References inside this audit plan's search commands are expected matches.

### C. Verify Meson and installation

- [ ] Trace every moved source and header through Meson dependency declarations.
- [ ] Verify generated configuration headers resolve from nested tool directories.
- [ ] Build a fresh CUDA-free configuration; do not rely on an old build tree.
- [ ] Stage installation and inspect binaries, service paths, and executable modes.
- [ ] Repeat with PM Sampling enabled and disabled on a CUDA-equipped machine.
- [ ] Verify toolkit layouts supported by the platform registry, including target
  include/library directories and the legacy CUPTI layout.
- [ ] Confirm manual build instructions compile the same sources and preserve
  the output locations expected by `run_pm_sampling_benchmark.sh`.

Example isolated local check; use fresh paths if these already exist:

```bash
CCACHE_DIR=/tmp/nvidia-audit-ccache meson setup /tmp/nvidia-audit-build -Dbuild_cuda_components=false -Denable_pm_sampling=false
CCACHE_DIR=/tmp/nvidia-audit-ccache meson compile -C /tmp/nvidia-audit-build
DESTDIR=/tmp/nvidia-audit-stage meson install -C /tmp/nvidia-audit-build --no-rebuild
bash -n build_gpus.sh run_pm_sampling_benchmark.sh
```

Expected installed names remain `nvidia-process-metrics-daemon`,
`nvidia-process-metrics-launcher`, `libnvidia-process-metrics.so`, and, when
built, `pm_sampling_simple`. A CUDA-free installation cannot prove the latter
two work.

### D. Verify packaging

- [ ] Validate all entries in `packaging/supported-platforms.yaml`.
- [ ] Inspect both native package source-copy paths for the relocated `include/`,
  `src/`, and `misc/` trees.
- [ ] Build from an extracted source archive, not just the development checkout.
- [ ] Verify generic Linux, Fedora, Ubuntu/Debian, Rocky, and ORFEO installation
  paths against their registry entries.
- [ ] Build and inspect RPM/DEB artifacts on supported packaging hosts.
- [ ] Build a Python wheel in a clean packaging environment; install it outside
  the checkout with dependencies available.
- [ ] Confirm the wheel contains `energy_attribution`, all command modules, and
  no dependency on the root compatibility symlink or repository scripts.
- [ ] Check a Python source distribution also contains what wheel building needs.

```bash
python3 packaging/scripts/validate-platforms.py
python3 -m pip wheel --no-deps --wheel-dir /tmp/nvidia-audit-wheels .
```

Use an environment with the configured build backend available. Missing build
or packaging tools must be recorded as unavailable, not silently skipped.

### E. Verify analyzers and output contracts

- [ ] Run the regression suite through the relocated scripts.
- [ ] Run all eight scripts with `--help` from outside the repository.
- [ ] Check all eight installed console commands outside the checkout.
- [ ] Run actual PM association, weighted-energy analysis, and PM plotting on a
  small capture; these are not all covered by the Bayesian golden scenarios.
- [ ] Compare output filenames, CSV column names/order, JSON keys, numerical
  values, and plots with the saved pre-migration baseline where available.
- [ ] Confirm `bayesian_kernel_coefficients.csv` and existing downstream joins
  remain usable without moving or rewriting historical results.
- [ ] Verify malformed inputs and missing optional dependencies produce useful
  errors and preserve expected exit behavior.
- [ ] Confirm test discovery and architecture checks inspect `analyzers/` rather
  than passing vacuously because the root scripts were deleted.

```bash
python3 -m pytest
python3 analyzers/analyze_bayesian_kernel_energy.py --help
python3 analyzers/analyze_kernel_pm_metrics.py --help
```

Do not regenerate golden fixtures to make a layout-only migration pass.
Investigate any numerical or schema differences first.

### F. Verify collection lifecycle on NVIDIA hardware

- [ ] Capture one short benchmark through the launcher and daemon.
- [ ] Confirm successful target exit and profiler completion reporting.
- [ ] Verify nonempty telemetry and kernel files, timestamps, and required columns.
- [ ] Verify PM output when supported and telemetry/kernel capture without PM.
- [ ] Check CUDA device selection matches NVML by PCI identity, including changed
  `CUDA_VISIBLE_DEVICES` ordering.
- [ ] Check observe mode, concurrent-session rejection, termination, and failed
  workload behavior against the existing documented contract.
- [ ] Check the final sampling interval, CUPTI flush, thread shutdown, and CSV
  completion are retained before process teardown.
- [ ] Run the analyzers against the new capture to validate collection-to-report
  integration, not merely individual command startup.

Store captures in a new audit directory. Record GPU model, driver, CUDA/CUPTI
versions, and exact capture commands. A CUDA-free build cannot close this phase.

### G. Verify style and documentation

Style sources supplied by the user:

- `/home/kevinl/Downloads/python-style-prompt.md`
- `/home/kevinl/Downloads/cpp-style-prompt.md`

- [ ] Read those documents when auditing; record their versions or hashes.
- [ ] Separate newly written/edited lines from unchanged relocated code.
- [ ] Apply Google Python conventions and the supplied Pylint configuration,
  formatting, and type checks to the appropriate changed scope.
- [ ] Apply Google C++ formatting and cpplint to the appropriate changed scope.
- [ ] Record existing-convention deviations explicitly: C++17, `.cpp/.hpp`, and
  any retained exception, include-guard, or formatting conventions.
- [ ] Review existing exceptions and global lifetime management separately;
  retaining them during relocation is not proof of full style compliance.
- [ ] Record missing tools and unresolved findings. A default Pylint run does
  not establish compliance with the requested Google configuration.
- [ ] Verify README, analyzer documentation, manual build instructions, and
  analysis guide use the current paths and explain the root script removal.

Typical commands once the specified tools/configuration are available:

```bash
pyink --check --line-length 80 <changed-python-files>
pylint --rcfile=<google-pylintrc> <changed-python-files>
pytype <changed-python-files>
clang-format --style=Google --dry-run --Werror <changed-cpp-files>
cpplint <changed-cpp-files>
```

The angle-bracket arguments are placeholders. Review findings before applying
formatting across whole moved files; the supplied rules prohibit unrelated
reformatting.

## 5. Findings register

Use one entry per issue:

| ID | Requirement | Severity | Evidence/path | Expected behavior | Observed behavior | Fix | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STRUCT-01 | Collector component separation | Open scope decision | `process-metrics.cpp` | Resolve earlier telemetry/activity proposal | Implementation remains combined (1,157 lines) | Define required boundary before extraction | CUDA build and capture if changed | Pending |
| STRUCT-02 | Efimon integration | High (open scope decision) | `src/efimon/gpu/nvidia-gpu/`, `/home/kevinl/efimon/include/efimon/gpu/nvidia.hpp` | Code in namespace `efimon`, collector implements `Observer` | Namespace `nvidia_process_metrics`; no `Observer`, `Status` or `Readings` use | Decide whether to wrap the collector as an Efimon observer | Build against Efimon | Pending |
| STRUCT-03 | Efimon directory naming | Low (open scope decision) | `include/efimon/gpu/nvidia-gpu/` | Efimon keeps NVIDIA code at `gpu/nvidia.{hpp,cpp}` | Extra `nvidia-gpu/` level repeats "gpu" | Decide whether to flatten | Rebuild and path search | Pending |
| STRUCT-04 | Efimon Meson layout | Medium | `meson.build`, `src/` | `include/efimon/**/meson.build` installs headers; `src/meson.build` | Headers are not installed; no `src/meson.build` | Add header install rules if headers become public API | Staged install | Pending |
| STRUCT-05 | Efimon tooling and docs | Medium | Repository root | Efimon has `.pre-commit-config.yaml`, Doxygen `docs-src/`, `examples/`, `test/` | Pre-commit added (Efimon hooks plus pyink/pylint); no Doxygen, examples or C++ tests | Add Doxygen config if docs are required | Build docs | Partial |
| STRUCT-06 | Tool include convention | Medium | `src/tools/**` | `<efimon/...>` for library headers, `"subdir/file.hpp"  // NOLINT` for tool headers | Was `"worker.hpp"` style | Converted to Efimon convention | CUDA-free build, cpplint | Pass |
| STRUCT-07 | Leftover directories | Low | `nvidia/`, `tools/` | Removed or documented | Both exist and are empty | Remove `tools/`; keep `nvidia/` only if build scripts still write there | `ls` | Pending |
| VERIFY-01 | CUDA collection validation | High | Section 4F | Real capture and analysis succeed | Not verified; no CUDA toolkit on the audit host | Run on suitable hardware | Attach logs and output checks | Pending |
| STYLE-01 | C++ style (project-owned files) | Medium | 9 files under `include/` and `src/`, excluding `pm-sampling.*` | 0 cpplint errors, no clang-format diff | Was 80 cpplint errors; now 0 (`CPPLINT.cfg`), clang-format clean, Doxygen headers and `#ifndef` guards added | Done 2026-10-27 | Fresh CUDA-free build with and without PM sampling: 0 warnings; `process-metrics.cpp` diffed only, not compiled | Pass (CUDA compile pending) |
| STYLE-02 | Vendored CUPTI sample style | Low (decision) | `pm-sampling.hpp`, `pm-sampling.cu` | Third-party code excluded or reformatted | 274 cpplint errors; excluded in pre-commit as adapted NVIDIA sample | Decide whether to move under a third-party path, as Efimon does | — | Pending |
| STYLE-03 | Python style | Medium | `analyzers/` (48 files) | pyink 80 clean, Google `pylintrc` clean | Was 279 pylint findings and 33 unformatted files; now 0 and 0 | Done 2026-10-27; targeted disables with reasons for lazy matplotlib imports and f-string quotes | `pytest`: 22 passed, 86 subtests | Pass |
| STYLE-04 | Recorded deviations | Low | Whole tree | Deviations listed | Exceptions (Efimon also throws), `from x import Name` imports, missing license headers, 128 unannotated public functions, `pytype` not run, mutable globals in `process-metrics.cpp` | Address with STRUCT-01/02 or accept as local convention | — | Pending |
| STRUCT-08 | Single responsibility (Python) | Medium | `analyzers/energy_attribution/` | One job per module; functions near 40 lines | `weighted_kernel_energy.py` (742 lines) mixed CSV input, attribution sweep, output and plots; 309-line `create_multisession_plots`; 222- and 180-line `main` functions | Split into `streaming_io.py`, `streaming_attribution.py`, `pm_association.py`, `reporting/weighted_outputs.py`, `reporting/pm_plots.py`; one function per figure | All 40 outputs (CSV, PNG, stdout) byte-identical on capture `65c0d52ac7833-5f090-33231704` and the synthetic telemetry fixture; edge cases match; pytest 22 passed | Pass |

Severity guidance: high blocks functional migration completion; medium needs a
fix or documented rationale; low covers documentation or maintenance findings.
An open scope decision must be resolved explicitly rather than counted as pass.

## 6. Completion criteria

The audit is complete only when:

1. Every requirement is mapped to evidence or an explicitly unresolved item.
2. No root analyzer `.py` files or active obsolete path references remain.
3. Build, installation, packaging, and analyzer regression checks pass for the
   supported configurations actually claimed as verified.
4. Hardware collection is verified before claiming CUDA runtime compatibility.
5. Capture/report contracts and existing result contents are preserved.
6. The collector separation difference has a recorded disposition.
7. Style findings, local-convention deviations, and unavailable checks are listed.
8. The final report identifies remaining gaps without describing partial checks
   as complete Efimon integration.
