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
  -I/usr/local/cuda/extras/CUPTI/samples/Common \
  -L/usr/local/cuda/extras/CUPTI/lib64 \
  -lcupti -Xcompiler -pthread \
  -o nvidia/pm_sampling_simple
```

## Run a benchmark with process metrics

```bash
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

Then, in another terminal, run the benchmark:

```bash
cd ~/tfg/nvidia-profiling
LD_PRELOAD="$PWD/nvidia/libnvidia-process-metrics.so" \
  ./benchmarks/build/bench_compute --duration 10 --warmup 5
```
