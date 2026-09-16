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

The temporary script appends the benchmark name to each CSV, for example
`gpu_telemetry_compute.csv`. Daemon sessions use the unsuffixed names shown
above. Always use the file names that exist in the captured directory.

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

The analysis can run on Thor or on a laptop. When capture and analysis use
different machines, copy the complete session directory after the capture has
finished. The energy analysis needs the telemetry and kernel CSV files; the
PM-weighted analysis additionally needs the PM CSV file.

```bash
# Run on the laptop. Replace the host, source path, and run name as needed.
mkdir -p results
scp -r nvidia@<thor-host>:~/klobo/results/<run-id>-compute results/
```

For a daemon capture, copy the session printed by the launcher instead:

```bash
mkdir -p results
scp -r nvidia@<thor-host>:/tmp/nvidia-process-metrics/<uid>/<session-id> \
  results/<session-id>
```

Activate a Python environment with `matplotlib` before requesting plots. On a
new laptop environment, install it once with `python3 -m pip install matplotlib`.

```bash
source .venv/bin/activate
```

### Associate NVML energy with kernels

The energy analysis integrates NVML power by time window and attributes it to
active kernels. With the default `--idle-power-mw 0`, results must be
interpreted as **Attributed Device Energy**, not physical energy measured
directly for each kernel.

`--idle-power-mw` is optional. It subtracts a device idle baseline before
energy attribution. Use `0` when reporting **Attributed Device Energy**. Use
a measured value only when reporting energy above idle as dynamic energy.
Measure it on the same GPU, power mode, clock policy, and display state as the
experiment; do not reuse an idle value from another machine. For example:

```bash
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -l 1
```

Run this while the GPU is idle, before starting the benchmark. Use the median
of several readings, converted from watts to milliwatts. For the NVIDIA Jetson
AGX Thor Developer Kit readings centered around `2.37 W`, pass
`--idle-power-mw 2370`.

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


### PM-weighted energy attribution

Pass `kernel_pm_metrics.csv` to the energy analysis to use the PM instruction
activity metric as an energy-allocation weight. The model uses PM weights only
when every concurrent kernel has at least 99.9% PM coverage; otherwise it
preserves the equal-share time-based allocation as a fallback.

```bash
python3 analyze_kernel_energy.py \
  --telemetry results/20260911-184551-107039-compute/gpu_telemetry_compute.csv \
  --kernels results/20260911-184551-107039-compute/kernel_activity_compute.csv \
  --pm-kernel-metrics results/20260911-184551-107039-compute/pm-kernel-analysis/kernel_pm_metrics.csv \
  --pm-weight-metric sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed \
  --pm-min-coverage-pct 99.9 \
  --window-ms 200 \
  --output-dir results/20260911-184551-107039-compute/pm-weighted-energy-analysis \
  --plot
```

`kernel_energy.csv` adds `pm_coverage_pct`, `pm_energy_weight`,
`pm_weighted_energy_mj`, and `equal_share_fallback_energy_mj`. The energy
summary reports how much energy used PM weighting and how much required the
equal-share fallback.

For a copied daemon session, run the two analysis steps in this order:

```bash
SESSION=results/<session-id>

python3 analyze_kernel_pm_metrics.py \
  --pm-samples "$SESSION/pm_sampling.csv" \
  --kernels "$SESSION/kernel_activity.csv" \
  --output-dir "$SESSION/pm-kernel-analysis" \
  --plot

python3 analyze_kernel_energy.py \
  --telemetry "$SESSION/gpu_telemetry.csv" \
  --kernels "$SESSION/kernel_activity.csv" \
  --pm-kernel-metrics "$SESSION/pm-kernel-analysis/kernel_pm_metrics.csv" \
  --window-ms 200 \
  --output-dir "$SESSION/pm-weighted-energy-analysis" \
  --plot
```

The PM weighting remains an attribution model: NVML power and PM counters are
device-wide, so keep the GPU free of unrelated workloads during capture.

See [ANALYSIS_GUIDE.txt](ANALYSIS_GUIDE.txt) for CSV and plot interpretation.
