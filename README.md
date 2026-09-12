# NVIDIA profiling

Run these commands on NVIDIA Thor from the repository root.

## Build

```bash
cmake -S benchmarks -B benchmarks/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j

g++ -std=c++17 -fPIC -shared nvidia/nvidia-process-metrics.cpp \
  -I/usr/local/cuda/include \
  -I/usr/local/cuda/extras/CUPTI/include \
  -L/usr/local/cuda/lib64 \
  -L/usr/local/cuda/extras/CUPTI/lib64 \
  -lcupti -lnvidia-ml -lcuda -pthread \
  -o nvidia/libnvidia-process-metrics.so

nvcc -std=c++17 nvidia/pm_sampling_simple.cu \
  -I/usr/local/cuda/extras/CUPTI/include \
  -I/usr/local/cuda/extras/CUPTI/samples/common \
  -L/usr/local/cuda/extras/CUPTI/lib64 \
  -lcupti -lcuda \
  -Xcompiler -pthread \
  -o nvidia/pm_sampling_simple
```

## Run a benchmark with process metrics

```bash
LD_PRELOAD="$PWD/nvidia/libnvidia-process-metrics.so" \
  ./benchmarks/build/bench_compute --duration 10 --warmup 5
```

The profiler also writes `gpu_telemetry.csv` and `kernel_activity.csv`. Set
`NVIDIA_METRICS_OUTPUT_DIR` to keep the files for each run in a separate
directory:

```bash
mkdir -p results/compute-run
NVIDIA_METRICS_OUTPUT_DIR="$PWD/results/compute-run" \
LD_PRELOAD="$PWD/nvidia/libnvidia-process-metrics.so" \
  ./benchmarks/build/bench_compute --duration 10 --warmup 5
```

Replace `bench_compute` with `bench_idle`, `bench_memory`, `bench_stride`, or
`bench_transfer`. Use `--help` to see each benchmark's options.

## Run PM Sampling

PM Sampling defaults to 10 seconds and writes `pm_sampling.csv` in the current
directory.

```bash
sudo ./nvidia/pm_sampling_simple --device 0 --duration 10
```

To capture a complete 10-second benchmark, start a 12-second sampler first:

```bash
sudo ./nvidia/pm_sampling_simple --device 0 --duration 12
```
Let's mention about what pm sampling option there are:

Then, in another terminal, run the benchmark:

```bash
cd ~/tfg/nvidia-profiling
LD_PRELOAD="$PWD/nvidia/libnvidia-process-metrics.so" \
  ./benchmarks/build/bench_compute --duration 10 --warmup 5
```

The collection script accepts a benchmark name and only stores raw collection
files; it does not run the laptop-side analysis:

```bash
BENCHMARK_NAME=memory ./run_pm_sampling_benchmark.sh
```

Valid names are `compute`, `memory`, `stride`, `transfer`, and `idle`.
The benchmark label is appended to the result directory and every collected
file. A compute run, for example, produces:

```text
results/20260904-195154-24452-compute/
├── gpu_telemetry_compute.csv
├── kernel_activity_compute.csv
├── pm_sampling_compute.csv
└── pm_sampling_compute.log
```

## Associate energy with kernels

Copy `gpu_telemetry.csv` and `kernel_activity.csv` to the analysis computer,
then run local.

```bash
python3 analyze_kernel_energy.py \
  --telemetry results/<file>/gpu_telemetry_<type_data>.csv \
  --kernels results/<file>/kernel_activity_<type_data>.csv \
  --window-ms 200 \
  --output-dir results/<file>/energy-analysis \
  --plot
```



python3 analyze_kernel_energy.py \
  --telemetry results/20260910-005938-86151-compute/gpu_telemetry_compute.csv \
  --kernels results/20260910-005938-86151-compute/kernel_activity_compute.csv \
  --window-ms 200 \
  --output-dir results/20260910-005938-86151-compute/energy-analysis \
  --plot

20260910-015643-87139-compute

python3 analyze_kernel_energy.py \
  --telemetry results/20260910-015643-87139-compute/gpu_telemetry_compute.csv \
  --kernels results/20260910-015643-87139-compute/kernel_activity_compute.csv \
  --window-ms 200 \
  --output-dir results/20260910-015643-87139-compute/energy-analysis \
  --plot



python3 analyze_kernel_pm_metrics.py \
  --pm-samples results/20260910-015643-87139-compute/pm_sampling_compute.csv \
  --kernels results/20260910-015643-87139-compute/kernel_activity_compute.csv \
  --output-dir results/20260910-015643-87139-compute/pm-kernel-analysis

The analysis produces:

- `kernel_energy.csv`: estimated dynamic energy for every kernel.
- `energy_windows.csv`: active-time ratio, power, and energy per window.
- `energy_summary.csv`: totals, conservation error, and the Pearson
  active-time–power correlation.
- `energy_timeline.png`: energy and kernel activity over time.
- `utilization_energy_correlation.png`: utilization–energy scatter plot.


BENCHMARK_NAME=compute ./run_pm_sampling_benchmark.sh
BENCHMARK_NAME=memory ./run_pm_sampling_benchmark.sh
BENCHMARK_NAME=stride ./run_pm_sampling_benchmark.sh
BENCHMARK_NAME=transfer ./run_pm_sampling_benchmark.sh
BENCHMARK_NAME=idle ./run_pm_sampling_benchmark.sh
