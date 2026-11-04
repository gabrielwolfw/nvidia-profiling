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
| `bench_molecular_dynamics` | Scientific application | FP32/FP64 pipelines, irregular memory | `precision`, `lattice-cells` |
| `bench_quantized_dnn` | Neural-network inference | FP32 FMA / FP16 HFMA2 / INT8 DP4A | `precision`, layer widths |
| `bench_vpi` | Computer vision (NVIDIA VPI) | Image filters, corner detection, stereo matching | `algorithm`, `backend` |

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
├── neural/bench_quantized_dnn.cu
├── simulation/bench_molecular_dynamics.cu
├── transfer/bench_transfer.cu
└── vision/bench_vpi.cu
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

## Vision Transformer (ViT) inference

`pytorch/vit_inference.py` runs `timm`'s `vit_base_patch16_224` (12 blocks,
dimension 768, about 86 M parameters, 197 tokens per 224x224 image) with
random weights and inputs, so a capture needs no download. Each process runs
one precision, so every launcher session maps to one configuration:

| `--dtype` | Path |
|---|---|
| `fp32` | Baseline; TF32 stays off unless `--tf32` is given |
| `fp16`, `bf16` | Native PyTorch half precision |
| `fp8` | `bf16` model with torchao dynamic float8 on every Linear layer whose dimensions are multiples of 16 (the 1000-class head stays `bf16`); needs compute capability 8.9 or newer |

Like `bench_power_cycle`, it emits `benchmark_*_marker` kernels between
warm-up, measurement and cooldown, so the measured phase is found in
`kernel_activity.csv`. Model construction and `torch.compile` run before the
first marker. The log reports `IMAGES_PER_SECOND`, `MS_PER_BATCH` and
`REL_L2_ERROR_VS_FP32` (logit error against the same model in fp32).

Two choices keep it measurable with the launcher:

- It removes the profiler library from `LD_PRELOAD` for its child processes.
  `torch.compile` starts compiler subprocesses; with the preload each would
  start its own profiler and overwrite the session CSV files when it exits.
- `torch.compile` modes that use CUDA graphs are not offered, because CUDA
  graph replay crashed inside CUPTI on Thor (see the llama.cpp notes in the
  top-level README).

### Set up on Thor

Use a virtual environment, and install `torch`, `torchvision` and `torchao`
from the same CUDA 13 index *before* `timm`. Otherwise pip pulls a
`torchvision` that does not match `torch` (`operator torchvision::nms does
not exist`).

```bash
cd ~/klobo
python3 -m venv .venv-vit
.venv-vit/bin/pip install torch torchvision torchao \
  --index-url https://download.pytorch.org/whl/cu130
.venv-vit/bin/pip install timm
```

Check the installation outside the launcher. It prints the device, compute
capability and the CUDA architectures in the wheel, compiles the marker
kernels and runs every precision for two seconds:

```bash
.venv-vit/bin/python benchmarks/pytorch/vit_inference.py --preflight
.venv-vit/bin/python benchmarks/pytorch/vit_inference.py --preflight --compile
```

Every line must read `PREFLIGHT <dtype>: OK` and `PHASE_MARKERS=ok`. A
warning that the wheel has no binary for the GPU architecture means the wheel
does not target Thor; install one that does before capturing. If only `fp8`
fails, capture the other precisions and record why.

### Capture

One session per precision and repetition. The order alternates between
repetitions, and the launcher duration is a cap (the session ends when the
benchmark exits):

```bash
benchmarks/scripts/run_vit_sweep.sh \
  --python "$PWD/.venv-vit/bin/python" \
  --output-dir results/vit-sweep/thor \
  --repetitions 3 --batch-size 64 --duration 40 --compile
```

A single manual run:

```bash
nvidia-process-metrics-launcher --device 0 --duration 300 --window-ms 200 \
  --command -- "$PWD/.venv-vit/bin/python" \
  "$PWD/benchmarks/pytorch/vit_inference.py" --dtype fp16 --batch-size 64 \
  --duration 40 --compile
```

### Analyze

```bash
python3 benchmarks/scripts/analyze_vit_sweep.py --study-dir results/vit-sweep/thor
```

It writes `summary/vit_sessions.csv` (one row per session) and
`summary/vit_precision_summary.csv` (mean and standard deviation per
precision) with throughput, mean NVML power, energy per image, dynamic
energy per image (cooldown power as the idle baseline) and images per joule.
Each session directory is also a normal capture, so the kernel-energy
analyzers in `analyzers/` accept it as an unseen workload.

## Molecular dynamics (Lennard-Jones liquid)

`simulation/bench_molecular_dynamics.cu` simulates a Lennard-Jones liquid in
reduced units with the settings of the LAMMPS `in.lj` benchmark: an FCC
lattice at density 0.8442 melted from T = 1.44, cutoff 2.5 sigma, neighbor
skin 0.3 sigma, neighbor list rebuilt every 20 steps without a displacement
check, and NVE velocity Verlet with dt = 0.005 tau. The default
`--lattice-cells 32` gives 4 x 32^3 = 131072 atoms.

Each step launches these kernels, so a capture mixes compute-bound and
memory-bound work:

| Kernel | Work |
|---|---|
| `md_integrate_first_half_kernel` | half-kick and drift, periodic wrap |
| `md_bin_atoms_kernel` | cell list (every `--neighbor-every` steps) |
| `md_build_neighbor_list_kernel` | full neighbor list from 27 cells (same steps) |
| `md_lj_force_kernel` | pair forces, about 78 neighbors per atom |
| `md_integrate_second_half_kernel` | second half-kick |

`--precision fp32|fp64` selects the type of positions, velocities and forces
(the kernels are templates, so the kernel name shows the type). Energies are
always reduced in double, so `RELATIVE_ENERGY_DRIFT` (total energy change over
the measurement phase) compares how well each precision conserves energy.

Like `bench_power_cycle`, it emits `benchmark_*_marker` kernels between
warm-up, measurement and cooldown. The two energy reductions run outside the
measured interval. Throughput is reported as `STEPS_PER_SECOND`,
`MS_PER_STEP` and `ATOM_STEPS_PER_SECOND`; energy per atom-step is the NVML
energy of the measurement phase divided by `MEASUREMENT_STEPS x ATOMS`.

Cell and neighbor capacities are fixed (96 atoms per cell, 192 neighbors per
atom). An overflow stops the run with `STATUS=ERROR` instead of dropping pairs.

### Verify

`--mode verify` runs `--verify-steps` steps (rounded up to a rebuild step, so
the list is fresh) and compares every GPU force with an exact O(N^2) host
computation in double. It needs at most 32000 atoms:

```bash
./build/bench_molecular_dynamics --mode verify --precision fp32 --lattice-cells 12
./build/bench_molecular_dynamics --mode verify --precision fp64 --lattice-cells 12
```

`VERIFY=OK` requires a relative force error of at most 1e-4 (fp32) or 1e-9
(fp64), relative to the largest force.

### Capture

```bash
nvidia-process-metrics-launcher --device 0 --duration 120 --window-ms 200 \
  --command -- ./build/bench_molecular_dynamics --precision fp64 \
  --warmup-seconds 10 --duration 40 --cooldown-seconds 10
```

FP64 throughput depends strongly on the GPU: data-center parts such as the
V100 run FP64 at half the FP32 rate, while consumer and Jetson GPUs run it at
1/32 or 1/64.

## Quantized deep neural network (QDNN) inference

`neural/bench_quantized_dnn.cu` runs batched inference of a deep multilayer
perceptron, by default 1024 -> 4096 -> 4096 -> 4096 -> 1024 with ReLU between
layers and a batch of 256 (41.9 M multiply-accumulates per sample). Weights
and inputs are random and deterministic (`--seed`), so no model file is
needed. `--precision` selects one of three versions of the same network:

| Precision | Weights and activations | Inner product |
|---|---|---|
| `fp32` | float | `fmaf`, one MAC per 32-bit word |
| `fp16` | half | `__hfma2`, two MACs per word; partial sums promoted to float every 16 products |
| `int8` | symmetric INT8, per-output-channel weight scales, per-layer activation scales | `__dp4a`, four MACs per word, INT32 accumulation; float dequantization and INT8 requantization in the epilogue |

All three use the same tiled GEMM kernel (`qdnn_linear_kernel`, a 64 x 64
output tile per 256-thread block) and move the same number of 32-bit words per
tile, so they differ only in arithmetic density and memory footprint. Tensor
cores are not used: the V100 has none for INT8, while `__hfma2` and `__dp4a`
exist on every GPU from compute capability 6.1, so the same source builds for
the V100 (`70`) and Jetson Thor (`101` with CUDA 12.8, `110` with CUDA 13).

The INT8 activation scales are calibrated on the fp32 activations of the
input batch (max |x| / 127 per layer). Each run reports the accuracy cost
against the fp32 network as `REL_L2_ERROR_VS_FP32` (logits) and
`TOP1_AGREEMENT_VS_FP32_PCT` (same arg-max class). Throughput is reported as
`SAMPLES_PER_SECOND`, `MS_PER_BATCH` and `TERA_MACS_PER_SECOND`. Like
`bench_power_cycle`, phase markers delimit warm-up, measurement and cooldown;
model construction and calibration happen before the first marker.

`--mode verify` runs one 64 x 128 x 512 layer of each precision and compares
it with a host reference built from the same rounded or quantized inputs
(INT8 sums are exact on the host). It reports `VERIFY=OK` when the relative
L2 error is at most 1e-5 (fp32, int8) or 5e-3 (fp16).

```bash
./build/bench_quantized_dnn --mode verify
nvidia-process-metrics-launcher --device 0 --duration 120 --window-ms 200 \
  --command -- ./build/bench_quantized_dnn --precision int8 \
  --warmup-seconds 10 --duration 40 --cooldown-seconds 10
```

## Computer vision with NVIDIA VPI

`vision/bench_vpi.cu` streams synthetic frames through NVIDIA's Vision
Programming Interface (VPI 4, the version in JetPack 7). `--algorithm` selects
the workload:

| Algorithm | Frames | Work per frame |
|---|---|---|
| `gaussian` | U8 3840 x 2160 | 7 x 7 Gaussian filter (stencil, memory bound) |
| `rescale` | U8 3840 x 2160 -> 1920 x 1080 | bilinear downscale by two |
| `harris` | U8 1920 x 1080 | Harris corner detector with non-maximum suppression |
| `stereo` | U8 1920 x 1080 pair | semi-global matching disparity, 64 disparities (compute bound) |
| `pipeline` | BGR8 3840 x 2160 | grey conversion, 5 x 5 Gaussian, downscale by two, Harris corners |

`--width` and `--height` override the frame size. `--backend` selects the VPI
backend: `cuda` (default) runs on any GPU VPI supports, from the V100 (`sm_70`)
to Jetson Thor; `cpu` runs on the host; `pva`, `vic` and `ofa` exist only on
Jetson. Those three are accelerators outside the GPU, so CUPTI records none of
their work and NVML GPU power does not include it.

The frames are deterministic (`--seed`): bright squares on a textured
background, whose corners are known, and a random-texture stereo pair whose
right view is the left view shifted by 20 pixels. Throughput is reported as
`FRAMES_PER_SECOND`, `MS_PER_FRAME` and `MEGAPIXELS_PER_SECOND` (input
pixels), together with `KEYPOINTS_LAST_FRAME` (Harris, pipeline) or
`DISPARITY_WITHIN_1PX_PCT` (stereo). The Harris strength threshold defaults
to 10 for `harris` and 5 for `pipeline` (`--harris-strength`), the values at
which each scene's square corners are detected: VPI's default of 20 finds a
third fewer on the `harris` scene and none after the pipeline's blur and
downscale. Phase markers delimit warm-up,
measurement and cooldown as in `bench_power_cycle`; VPI's kernels appear in
the capture under VPI's own names.

`--mode verify` runs each algorithm on the CUDA backend. The Gaussian filter,
the downscale and the grey conversion are compared with the CPU backend: at
most two grey levels apart and at most 1e-4 of the pixels more than one level
apart, because the two backends round the filter taps differently. Harris
corners must match the squares' corners within 3 pixels (precision and recall
at least 90 %), and at least 90 % of the stereo disparities must be within one
pixel of the true 20.

VPI is not part of the CUDA Toolkit. On Jetson it is installed with JetPack
under `/opt/nvidia/vpi4`. On x86_64 NVIDIA publishes Ubuntu packages
(`libnvvpi4`, `vpi4-dev`) in `https://repo.download.nvidia.com/jetson/x86_64`;
they need glibc 2.35 (Ubuntu 22.04 or newer). The target is only configured
when CMake finds VPI:

```bash
cmake -S . -B build -Dvpi_DIR=/opt/nvidia/vpi4/lib/x86_64-linux-gnu/cmake/vpi
cmake --build build -j --target bench_vpi
./build/bench_vpi --mode verify
nvidia-process-metrics-launcher --device 0 --duration 120 --window-ms 200 \
  --command -- ./build/bench_vpi --algorithm pipeline \
  --warmup-seconds 10 --duration 40 --cooldown-seconds 10
```
