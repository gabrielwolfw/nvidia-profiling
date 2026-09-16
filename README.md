# NVIDIA Process Metrics

A non-instrumented profiler for NVIDIA GPUs. It combines CUPTI Activity for
the kernel timeline, NVML for device telemetry, and CUPTI PM Sampling for
microarchitectural metrics.

Run capture and installation commands from the repository root on NVIDIA Thor.
Python analyses can run on Thor or on another machine that has the CSV files
and Python dependencies.

## 1. Build the benchmarks

Benchmarks are independent from the profiler and must be built first.

```bash
cmake -S benchmarks -B benchmarks/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j
```

The resulting executables are:

```text
benchmarks/build/bench_compute
benchmarks/build/bench_memory
benchmarks/build/bench_stride
benchmarks/build/bench_transfer
benchmarks/build/bench_idle
```

Each executable accepts `--help`. `--warmup 5` means five warmup launches, not
five seconds.

## 2. Manually build the NVIDIA components

Use this path for development and for the temporary
`run_pm_sampling_benchmark.sh` test. On Thor, CUPTI is installed inside the
CUDA toolkit: headers and libraries are in `/usr/local/cuda/include` and
`/usr/local/cuda/lib64`; only `helper_cupti.h` is under
`extras/CUPTI/samples/common`.

```bash
g++ -std=c++17 -fPIC -shared nvidia/nvidia-process-metrics.cpp \
  -I/usr/local/cuda/include \
  -I/usr/local/cuda/extras/CUPTI/samples/common \
  -L/usr/local/cuda/lib64 \
  -lcupti -lnvidia-ml -lcuda -pthread \
  -o nvidia/libnvidia-process-metrics.so

nvcc -std=c++17 nvidia/pm_sampling_simple.cu \
  -I/usr/local/cuda/include \
  -I/usr/local/cuda/extras/CUPTI/samples/common \
  -L/usr/local/cuda/lib64 \
  -lcupti -lcuda \
  -Xcompiler -pthread \
  -o nvidia/pm_sampling_simple
```

To capture a benchmark without PM Sampling, load the library directly:

```bash
mkdir -p results/manual-compute
NVIDIA_METRICS_DEVICE=0 \
NVIDIA_METRICS_OUTPUT_DIR="$PWD/results/manual-compute" \
LD_PRELOAD="$PWD/nvidia/libnvidia-process-metrics.so" \
  ./benchmarks/build/bench_compute --duration 10 --warmup 5
```

This produces `gpu_telemetry.csv` and `kernel_activity.csv` in the selected
result directory.

## 3. Temporary benchmark and PM Sampling test

`run_pm_sampling_benchmark.sh` is the quick way to test all three collection
streams before installing the daemon. It starts PM Sampling as root, runs the
benchmark with `LD_PRELOAD`, and organizes the CSV files under `results/`.

```bash
BENCHMARK_NAME=compute ./run_pm_sampling_benchmark.sh
```

Valid names are `compute`, `memory`, `stride`, `transfer`, and `idle`. Useful
optional variables are:

```bash
DEVICE=0 DURATION_SECONDS=10 PM_DURATION_SECONDS=12 \
PM_MAX_SAMPLES=100000 BENCHMARK_NAME=compute \
  ./run_pm_sampling_benchmark.sh
```

A compute run creates a directory like this:

```text
results/<run-id>-compute/
├── gpu_telemetry_compute.csv
├── kernel_activity_compute.csv
├── pm_sampling_compute.csv
└── pm_sampling_compute.log
```

The PM log contains a continuity summary. Fewer gaps and less total gap time
increase the number of kernels with PM coverage.

## 4. Production installation: daemon and launcher

Meson installation is the production path. The daemon runs as root because
CUPTI PM Sampling requires it. The launcher runs as the regular user and
places results in a user-owned session directory.

```bash
meson setup --wipe build-daemon \
  --prefix=/usr/local \
  --libdir=lib \
  -Dbuild_cuda_components=true \
  -Dnvidia_toolkit_root=/usr/local/cuda

meson compile -C build-daemon
sudo "$(command -v meson)" install -C build-daemon
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-process-metrics-daemon
```

Verify the service and its IPC socket:

```bash
systemctl --no-pager --full status nvidia-process-metrics-daemon
ls -l /run/nvidia-process-metrics/daemon.sock
```

Installed binaries:

```text
/usr/local/sbin/nvidia-process-metrics-daemon
/usr/local/bin/nvidia-process-metrics-launcher
```

### Fully profile a command

The launcher creates a session, enables PM Sampling, and releases the target
process with `LD_PRELOAD` only after PM Sampling is ready. Use a PM duration
slightly longer than the benchmark duration.

```bash
nvidia-process-metrics-launcher \
  --device 0 \
  --duration 12 \
  --window-ms 200 \
  --command -- \
  ./benchmarks/build/bench_compute --duration 10 --warmup 5
```

The launcher prints the session and generated paths. Files are stored in:

```text
/tmp/nvidia-process-metrics/<uid>/<session-id>/
├── gpu_telemetry.csv
├── kernel_activity.csv
├── pm_sampling.csv
└── pm_sampling.log
```

### Observe an existing PID

This mode captures global PM Sampling while an existing PID remains alive or
until `--duration` expires.

```bash
./benchmarks/build/bench_compute --duration 30 --warmup 5 &
TARGET_PID=$!

nvidia-process-metrics-launcher \
  --pid "$TARGET_PID" --observe \
  --device 0 --duration 10
```

A PID that did not start with `LD_PRELOAD` cannot retroactively produce
`kernel_activity.csv`. Therefore, `--observe` returns global PM Sampling and
does not support per-kernel energy attribution.

Only one PM Sampling session can be active per GPU.

## 5. Offline analysis

Activate a Python environment with `matplotlib` before requesting plots:

```bash
source .venv/bin/activate
```

### Associate NVML energy with kernels

The energy analysis integrates NVML power by time window and attributes it to
active kernels. With the default `--idle-power-mw 0`, results must be
interpreted as **Attributed Device Energy**, not physical energy measured
directly for each kernel.

```bash
python3 analyze_kernel_energy.py \
  --telemetry results/20260911-184551-107039-compute/gpu_telemetry_compute.csv \
  --kernels results/20260911-184551-107039-compute/kernel_activity_compute.csv \
  --window-ms 200 \
  --output-dir results/20260911-184551-107039-compute/energy-analysis \
  --plot
```

For a daemon session, replace the `results/...` paths with the directory
printed by the launcher:

```bash
SESSION=/tmp/nvidia-process-metrics/2002/<session-id>

python3 analyze_kernel_energy.py \
  --telemetry "$SESSION/gpu_telemetry.csv" \
  --kernels "$SESSION/kernel_activity.csv" \
  --window-ms 200 \
  --output-dir "$SESSION/energy-analysis" \
  --plot
```

### Associate PM Sampling with kernels

This analysis calculates PM time coverage and time-weighted PM metrics for
each kernel.

```bash
python3 analyze_kernel_pm_metrics.py \
  --pm-samples results/20260911-184551-107039-compute/pm_sampling_compute.csv \
  --kernels results/20260911-184551-107039-compute/kernel_activity_compute.csv \
  --output-dir results/20260911-184551-107039-compute/pm-kernel-analysis \
  --plot
```

Results are written to `kernel_pm_metrics.csv`, `pm_kernel_summary.csv`,
`pm_coverage_timeline.png`, and `pm_sampling_gaps.png`.

See [ANALYSIS_GUIDE.txt](ANALYSIS_GUIDE.txt) for CSV and plot interpretation.
