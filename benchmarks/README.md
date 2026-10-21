# NVIDIA GPU energy-profiling microbenchmarks

This directory contains small, controlled workloads for characterizing NVIDIA GPU
energy and performance behavior. The executables only generate load. CUPTI, NVML,
data collection, and profiler-specific logic intentionally remain outside them.

Each benchmark performs setup and warm-up first, then runs a stable experimental
phase for 10 seconds by default. The duration is controlled on the host with
`std::chrono::steady_clock`. Memory allocation, initialization, warm-up, and cleanup
are excluded from the reported experimental time.

## Workloads

| Benchmark | Category | Primary resource | Main variable |
|---|---|---|---|
| `bench_idle` | Baseline | GPU idle / context idle | `duration`, `mode` |
| `bench_compute` | Compute-bound | SM / FP32 pipelines | `iterations` |
| `bench_memory` | Memory-bound | Global memory / DRAM | `size` |
| `bench_stride` | Memory access | L1/L2/DRAM | `stride` |
| `bench_transfer` | Transfer | PCIe/DMA path | `size`, `direction` |
| `bench_power_cycle` | Phased power sweep | Compute / memory / mixed | `duty-cycle`, `period-ms` |

`bench_compute` runs independent FP32 FMA dependency chains in registers and writes
one result per thread. `bench_memory` repeatedly performs a full streaming copy.
`bench_stride` copies the in-bounds elements selected by `index = thread_id * stride`,
so larger values progressively space out global-memory accesses. `bench_transfer`
repeats synchronous CUDA copies using pinned or pageable host memory. `bench_idle` either
sleeps without using CUDA (`system`) or creates a CUDA context before sleeping
(`cuda-context`).

## Build

Requirements are a C++17 compiler, CMake 3.21 or newer, and the CUDA Toolkit.

```bash
cd benchmarks
cmake -S . -B build
cmake --build build -j
```

The five executables are written to `benchmarks/build/` for a normal single-config
build.

## Common arguments

All programs accept:

```text
--device <id>          CUDA device, default 0
--duration <seconds>   stable experimental phase, default 10
--warmup <iterations>  warm-up operations, default 5
--block-size <threads> CUDA block size, default 256
--help                 usage information
```

`bench_idle` accepts the warm-up and block-size options for interface consistency;
they do not cause GPU work. `bench_transfer` likewise accepts block size, although
CUDA runtime copies do not use a kernel block size. A zero warm-up count is valid.

Memory sizes are specified in bytes and must fit in host/device memory. Kernel-based
memory benchmarks require a multiple of four bytes because their elements are
`float`. The default is 268435456 bytes (256 MiB) per buffer.

## Examples

```bash
./build/bench_idle --duration 10 --mode system
./build/bench_idle --duration 10 --mode cuda-context --device 0

./build/bench_compute --device 0 --duration 10 \
  --iterations 10000 --block-size 256

./build/bench_memory --device 0 --duration 10 \
  --size 536870912 --mode copy

./build/bench_stride --device 0 --duration 10 \
  --size 536870912 --stride 8

./build/bench_transfer --device 0 --duration 10 \
  --size 268435456 --direction h2d --memory pinned
```

Currently, memory mode is `copy`; its CLI leaves room for later read/write modes.
Transfer directions are `h2d` and `d2h`; D2D can be added later without changing
the shared infrastructure.

## Machine-readable output

Configuration and results use one `KEY=VALUE` pair per line. A successful run ends
with fields like:

```text
STATUS=OK
EXECUTION_TIME_SECONDS=10.012345
OPERATIONS_OR_KERNEL_LAUNCHES=1234
```

The final synchronization is part of the stable phase, so the measured duration can
slightly exceed the requested duration by one bounded batch. This ensures all work
counted in the phase has completed before its endpoint is reported. Kernel launches
are checked immediately, and every CUDA runtime error includes its source file, line,
error name, numeric code, and description.

The actual resource behavior is intended to be verified externally using CUPTI
Activity, CUPTI PM Sampling, and NVML. No benchmark claims a fixed utilization level.

## Layout and extension

```text
benchmarks/
├── CMakeLists.txt
├── README.md
├── baseline/bench_idle.cpp
├── common/
│   ├── benchmark_utils.cpp
│   └── cuda_utils.cu
├── compute/bench_compute.cu
├── include/
│   ├── benchmark_config.hpp
│   ├── benchmark_utils.hpp
│   ├── cuda_utils.hpp
│   └── timing.hpp
├── memory/
│   ├── bench_memory.cu
│   └── bench_stride.cu
└── transfer/bench_transfer.cu
```

Additional compute, memory, and mixed workloads can reuse the common parsing,
timing, and CUDA error helpers without coupling them to a profiler.

## Reproducible power sweeps

`bench_power_cycle` executes one condition as a single CUDA process with explicit
warm-up, measurement, and cooldown phases. Small CUDA marker kernels delimit the
phases in `kernel_activity.csv`, so offline analysis uses the CUPTI timestamp domain
instead of approximating boundaries from host time. Load is controlled by alternating
GPU work and context-idle time; the benchmark does not change clocks or power limits
and does not require root privileges.

Build it with the rest of the benchmarks:

```bash
cmake -S benchmarks -B benchmarks/build
cmake --build benchmarks/build -j --target bench_power_cycle
```

Create a deterministic pilot plan without running the GPU:

```bash
python3 benchmarks/scripts/run_power_sweep.py \
  --config benchmarks/config/power_sweep.default.json \
  --mode pilot \
  --output-dir results/power-sweep/thor-pilot \
  --plan-only
```

Execute that plan through the installed daemon and launcher:

```bash
python3 benchmarks/scripts/run_power_sweep.py \
  --config benchmarks/config/power_sweep.default.json \
  --mode pilot \
  --launcher "$HOME/.local/bin/nvidia-process-metrics-launcher" \
  --benchmark benchmarks/build/bench_power_cycle \
  --output-dir results/power-sweep/thor-pilot
```

If the launcher requires non-default installation paths, add `--socket PATH` and
`--library PATH`. Interrupted studies can be continued with `--resume`; completed
conditions are not repeated.

Summarize only the marker-delimited phases:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
python3 benchmarks/scripts/analyze_power_sweep.py \
  --study-dir results/power-sweep/thor-pilot \
  --target-step-w 20 \
  --plot --strict
```

Use the pilot-selected levels directly for the randomized production study:

```bash
python3 benchmarks/scripts/run_power_sweep.py \
  --config benchmarks/config/power_sweep.default.json \
  --mode production \
  --levels-file \
    results/power-sweep/thor-pilot/summary/recommended_power_levels.csv \
  --launcher "$HOME/.local/bin/nvidia-process-metrics-launcher" \
  --benchmark benchmarks/build/bench_power_cycle \
  --output-dir results/power-sweep/thor-production
```

The production mode uses five block-randomized repetitions. The default configuration
contains 15-second warm-up and cooldown phases, a 45-second stable phase, 200 ms NVML
windows, and a 100 ms duty-control period. Run the pilot first: requested duty cycle
is a control variable, not a claim of a particular wattage on every GPU.
