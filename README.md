# NVIDIA Process Metrics

A profiler for NVIDIA GPUs that does not require changing the target's source.
It combines CUPTI Activity for the kernel timeline, NVML for device telemetry,
and CUPTI PM Sampling for microarchitectural metrics.

## Platform-aware installation layouts

Linux installation layouts are defined in
`packaging/supported-platforms.yaml`. The build helper detects the current
distribution from `/etc/os-release`, or accepts an explicit platform:

```bash
python3 packaging/scripts/build-package.py detect
python3 packaging/scripts/build-package.py stage --platform generic-linux
python3 packaging/scripts/build-package.py stage --platform fedora
```

`generic-linux` preserves the `/usr/local` installation documented below.
Fedora stages under `/usr` with `lib64`; Ubuntu stages under `/usr` with its
multiarch library directory. See `packaging/README.md` for configuration and
CUDA-free layout validation. Direct Meson builds remain supported.

The examples below use `~/klobo` as the repository on NVIDIA Thor. Run build
and capture commands from that directory. Run Python analysis on Thor or copy
the completed result directory to a laptop.

## 1. Build the benchmarks on Thor

Build these for synthetic benchmark runs. Skip this step for a Llama-only run.

```bash
cd ~/klobo
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

## 2. Build the collectors for the shell script

Use this path for development and for the temporary
`run_pm_sampling_benchmark.sh` test. On Thor, CUPTI is installed inside the
CUDA toolkit: headers and libraries are in `/usr/local/cuda/include` and
`/usr/local/cuda/lib64`. The project provides its own CUDA/CUPTI error-checking
helper and does not depend on files from the CUPTI samples directory.

```bash
cd ~/klobo
g++ -std=c++17 -fPIC -shared nvidia/nvidia-process-metrics.cpp \
  -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 \
  -lcupti -lnvidia-ml -lcuda -pthread \
  -o nvidia/libnvidia-process-metrics.so

nvcc -std=c++17 nvidia/pm_sampling_simple.cu \
  -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 \
  -lcupti -lcuda \
  -Xcompiler -pthread \
  -o nvidia/pm_sampling_simple
```

To capture a benchmark without PM Sampling, load the library directly:

```bash
mkdir -p results/manual-compute
NVIDIA_METRICS_DEVICE=0 \
NVIDIA_METRICS_WINDOW_MS=200 \
NVIDIA_METRICS_OUTPUT_DIR="$PWD/results/manual-compute" \
LD_PRELOAD="$PWD/nvidia/libnvidia-process-metrics.so" \
  ./benchmarks/build/bench_compute --duration 10 --warmup 5
```

This produces `gpu_telemetry.csv` and `kernel_activity.csv` in the selected
result directory. Set `NVIDIA_METRICS_WINDOW_MS` to any positive number of
milliseconds to adjust the kernel-utilization summary window; it defaults to
`200` when omitted.

## 3. Run a benchmark with the shell script

`run_pm_sampling_benchmark.sh` is the quick way to test all three collection
streams before installing the daemon. It starts PM Sampling as root, runs the
benchmark with `LD_PRELOAD`, and organizes the CSV files under `results/`.

```bash
cd ~/klobo
BENCHMARK_NAME=compute ./run_pm_sampling_benchmark.sh
```

Valid names are `compute`, `memory`, `stride`, `transfer`, and `idle`. Useful
optional variables are:

```bash
DEVICE=0 DURATION_SECONDS=10 PM_DURATION_SECONDS=12 \
PM_MAX_SAMPLES=100000 WINDOW_MS=500 BENCHMARK_NAME=compute \
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

## 4. Install and run the daemon

Meson installation is the production path. The daemon runs as root because
CUPTI PM Sampling requires it. The launcher runs as the regular user and
places results in a user-owned session directory.

```bash
cd ~/klobo
meson setup build-daemon \
  --prefix=/usr/local \
  --libdir=lib \
  -Dbuild_cuda_components=true \
  -Dnvidia_toolkit_root=/usr/local/cuda

meson compile -C build-daemon
sudo "$(command -v meson)" install -C build-daemon
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-process-metrics-daemon
```

On GPUs that do not support PM Sampling, add
`-Denable_pm_sampling=false` to the `meson setup` command. The daemon will
still collect GPU telemetry and kernel activity.

For later source changes, run only `meson compile -C build-daemon` and
`sudo "$(command -v meson)" install -C build-daemon`. Restart the service if
the daemon executable changed. The manual binaries from step 2 are separate
from this Meson installation.

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

## 5. Install and profile Llama 3.1 8B

Build llama.cpp in this repository's `workloads/` directory. If it is already
cloned, skip the `git clone` line. Keep this build separate from the benchmark
build and the Meson daemon build.

```bash
cd ~/klobo
mkdir -p workloads
git clone https://github.com/ggml-org/llama.cpp.git workloads/llama.cpp
cmake -S workloads/llama.cpp -B workloads/llama.cpp/build-no-graphs \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build workloads/llama.cpp/build-no-graphs -j4
```

CUDA graphs reduce launch overhead during repeated token generation. In the
observed Thor run, the crash stack passed through CUPTI and `cuGraphLaunch()`;
the same 512-token workload completed with graphs disabled. This build option
avoids that path. Record "CUDA graphs disabled" with results because it can
change runtime and energy use. The `-hf` model downloads on first use. To
keep the download outside the measured run, fetch it with a short unprofiled
inference first:

```bash
cd ~/klobo
./workloads/llama.cpp/build-no-graphs/bin/llama-cli \
  -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
  --gpu-layers 99 --ctx-size 2048 \
  -p "Hello" -n 1 --single-turn --simple-io
```

After installing the daemon in step 4, run the finite Llama workload from the
repository root. No server or port is required.

```bash
cd ~/klobo
nvidia-process-metrics-launcher \
  --device 0 \
  --duration 60 \
  --window-ms 200 \
  --command -- \
  "$PWD/workloads/llama.cpp/build-no-graphs/bin/llama-cli" \
    -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
    --gpu-layers 99 \
    --ctx-size 2048 \
    -p "Explain GPU energy profiling, including power, energy, utilization, hardware metrics, memory traffic, and methodological limitations." \
    -n 512 \
    --temp 0 \
    --single-turn \
    --simple-io
```




A complete capture prints `Target exit code: 0` and paths for
`gpu_telemetry.csv`, `kernel_activity.csv`, `pm_sampling.csv`, and
`pm_sampling.log`.

## 6. Copy and analyze results on the laptop

Copy the complete result directory after capture finishes. On the laptop,
from its copy of this repository, choose the source that matches the capture
method. Replace the angle-bracket placeholders with the path printed on Thor.

```bash
cd ~/tfg/nvidia-profiling
mkdir -p results

# Shell script from step 3:
scp -r <username>@<thor-host>:~/klobo/results/<run-id>-compute results/

# Daemon launcher from steps 4 or 5:
scp -r <username>@<thor-host>:/tmp/nvidia-process-metrics/<uid>/<session-id> results/
```

For `--plot`, activate the laptop's Python environment with `matplotlib`
installed (`python3 -m pip install matplotlib` if needed).

### Analyze a daemon session

Run PM association first; the energy analyzer needs its
`kernel_pm_metrics.csv` output for PM weighting. Set `SESSION` to the copied
directory, using the session ID printed by the launcher.

```bash
source .venv/bin/activate
SESSION="results/<session-id>"

Example:
SESSION="results/65ce5662fe4a8-8480c-a0776b95"

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

### Analyze a shell-script benchmark

The shell script adds the benchmark name to each CSV filename. For a compute
run, use the run ID printed by the script:

```bash
source .venv/bin/activate
RUN="results/<run-id>-compute"

python3 analyze_kernel_pm_metrics.py \
  --pm-samples "$RUN/pm_sampling_compute.csv" \
  --kernels "$RUN/kernel_activity_compute.csv" \
  --output-dir "$RUN/pm-kernel-analysis" \
  --plot

python3 analyze_kernel_energy.py \
  --telemetry "$RUN/gpu_telemetry_compute.csv" \
  --kernels "$RUN/kernel_activity_compute.csv" \
  --pm-kernel-metrics "$RUN/pm-kernel-analysis/kernel_pm_metrics.csv" \
  --window-ms 200 \
  --output-dir "$RUN/pm-weighted-energy-analysis" \
  --plot
```

NVML power and PM counters are device-wide. Per-kernel energy is an
attribution model, including when PM weighting is used. To report energy above
idle, measure the GPU's idle power under the same power and display settings:

```bash
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -l 1
```

Use the median in milliwatts with `--idle-power-mw` on the energy command.
For the earlier Thor reading of `2.37 W`, that value was `2370`; measure it
again for a new setup. Without this option, interpret the output as
**Attributed Device Energy**. See [ANALYSIS_GUIDE.txt](ANALYSIS_GUIDE.txt) for
CSV and plot interpretation.

## 7. Install on the `gpu002` V100 node

This cluster requires an explicit GPU type in the Slurm request. The V100
installation below disables PM Sampling, installs into `~/.local` without
`sudo`, and keeps the daemon socket and results in `/tmp` on the compute node.

### Copy a clean source tree

```bash
# Run on the development PC from the user's home directory.
cd "$HOME"
git clone --branch nvml-cupti-prof-kernel \
  https://github.com/gabrielwolfw/nvidia-profiling.git \
  nvidia-profiling
cd "$HOME/nvidia-profiling"
git pull --ff-only
```

The repository is now located at `~/nvidia-profiling` in the development
PC's home directory. Create a clean source archive because `scp` does not
provide exclude patterns. The archive leaves out `.env`, `.venv`, Git
metadata, build products, plots, data, and previous results.

```bash
cd "$HOME/nvidia-profiling"

tar -czf /tmp/nvidia-profiling-source.tar.gz \
  --exclude='./.git' \
  --exclude='./.vscode' \
  --exclude='./.env' \
  --exclude='./.env.*' \
  --exclude='./.venv' \
  --exclude='./.venv.*' \
  --exclude='./__pycache__' \
  --exclude='./build-daemon' \
  --exclude='./benchmarks/build' \
  --exclude='./benchmarks/build-gpu' \
  --exclude='./nvidia/libnvidia-process-metrics.so' \
  --exclude='./nvidia/pm_sampling_simple' \
  --exclude='./data' \
  --exclude='./plots' \
  --exclude='./results' \
  .

scp /tmp/nvidia-profiling-source.tar.gz username@x.x.x.x:~/

ssh username@x.x.x.x 'rm -rf "$HOME/nvidia-profiling" && mkdir -p "$HOME/nvidia-profiling" && tar -xzf "$HOME/nvidia-profiling-source.tar.gz" -C "$HOME/nvidia-profiling" && rm "$HOME/nvidia-profiling-source.tar.gz"'
```

### Allocate the V100

Connect to the cluster and request one V100 on `gpu002`:

```bash
ssh username@x.x.x.x

srun \
  --partition=GPU \
  --nodelist=gpu002 \
  --gres=gpu:V100:1 \
  --time=01:00:00 \
  --mem=32G \
  --cpus-per-task=2 \
  --pty bash
```

Load CUDA and verify that Slurm assigned the GPU:

```bash
module load cuda/12.8
cd ~/nvidia-profiling
nvidia-smi -L
```

The last command must display a `Tesla V100` before continuing.

### Build and install without PM Sampling

```bash
meson setup build-daemon \
  --prefix="$HOME/.local" \
  --libdir=lib \
  -Dbuild_cuda_components=true \
  -Denable_pm_sampling=false \
  -Dnvidia_toolkit_root=/orfeo/cephfs/opt/programs/intel/almalinux9/cuda/12.8

meson compile -C build-daemon
meson install -C build-daemon
```

This installs the launcher, daemon, and preload library under `~/.local`.
The configuration should report three Meson build targets and
`enable_pm_sampling: false`.

### Build all benchmarks

Build CUDA benchmark binaries for V100 (`70`), A100 (`80`), and H100/H200
(`90`):

```bash
cmake \
  -S benchmarks \
  -B benchmarks/build-gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES='70;80;90'

cmake --build benchmarks/build-gpu -j2
```

### Start the daemon in `/tmp`

The daemon must be started inside the active GPU allocation before running
the launcher:

```bash
mkdir -p "/tmp/nvidia-process-metrics-$UID/results"
rm -f "/tmp/nvidia-process-metrics-$UID/daemon.sock"

"$HOME/.local/sbin/nvidia-process-metrics-daemon" \
  --socket "/tmp/nvidia-process-metrics-$UID/daemon.sock" \
  --results-root "/tmp/nvidia-process-metrics-$UID/results" \
  >/tmp/nvidia-process-metrics-daemon.log 2>&1 &

sleep 1
ls -l "/tmp/nvidia-process-metrics-$UID/daemon.sock"
```

### Verify the installation

```bash
"$HOME/.local/bin/nvidia-process-metrics-launcher" \
  --socket "/tmp/nvidia-process-metrics-$UID/daemon.sock" \
  --library "$HOME/.local/lib/nvidia-process-metrics/libnvidia-process-metrics.so" \
  --device 0 \
  --duration 15 \
  --window-ms 150 \
  --command -- \
  ./benchmarks/build-gpu/bench_compute \
    --duration 10 \
    --warmup 2
```

A successful run displays `NVML device: 0`, `STATUS=OK`, a nonzero kernel
count, and `Target exit code: 0`. Session files are written below:

```text
/tmp/nvidia-process-metrics-<uid>/results/<uid>/<session-id>/
```

The `/tmp` directory is node-local and may be cleaned when the allocation or
node is reset. Copy any results that must be retained before leaving the node.
