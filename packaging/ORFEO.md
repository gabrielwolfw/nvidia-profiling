# Build and installation on ORFEO (Rocky Linux 9)

This guide covers two profiles on an ORFEO V100 compute node:

- `rocky-orfeo`: complete system installation with root PM Sampling.
- `rocky-orfeo-user`: user daemon and launcher without sudo or PM Sampling.

Run the Slurm allocation command from an ORFEO login node. Do not compile or
run GPU measurements on the login node.

| Profile | Build directory | Sudo | PM Sampling | Other metrics |
| --- | --- | --- | --- | --- |
| `rocky-orfeo` | `build-rocky-orfeo` | Required to install/run daemon | Enabled | NVML and kernel activity |
| `rocky-orfeo-user` | `build-rocky-orfeo-user` | Not required | Disabled | NVML and kernel activity |

## 1. Request the GPU node

```bash
srun \
  --partition=GPU \
  --nodelist=gpu002 \
  --gres=gpu:V100:1 \
  --time=01:00:00 \
  --mem=32G \
  --cpus-per-task=2 \
  --pty bash
```

Confirm that the allocation placed the shell on the expected node:

```bash
hostname
echo "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
```

## 2. Load CUDA 12.8

Run these commands inside the allocated `gpu002` shell:

```bash
module load cuda/12.8
module list
which nvcc
nvcc --version
```

The ORFEO profile expects this toolkit:

```text
/orfeo/cephfs/opt/programs/intel/almalinux9/cuda/12.8
```

## 3. Build and stage the complete profiler

```bash
cd ~/nvidia-profiling-test

python3 packaging/scripts/build-package.py detect \
  --platform rocky-orfeo

python3 packaging/scripts/build-package.py stage \
  --platform rocky-orfeo
```

The build directory is:

```text
~/nvidia-profiling-test/build-rocky-orfeo
```

The staged filesystem is:

```text
~/nvidia-profiling-test/dist/rocky-orfeo/root
```

Verify it without modifying the operating system:

```bash
find dist/rocky-orfeo/root -type f | sort

ldd \
  dist/rocky-orfeo/root/usr/lib64/nvidia-process-metrics/libnvidia-process-metrics.so

ldd \
  dist/rocky-orfeo/root/usr/lib64/nvidia-process-metrics/pm_sampling_simple
```

There should be no `not found` dependencies.

## 4. Measurements without sudo

Without administrative privileges, the preload library can collect NVML
device telemetry and CUPTI kernel activity for a command started by the same
user. This path does not install files or start a system service.

Create a result directory and preload the staged library:

```bash
cd ~/nvidia-profiling-test
mkdir -p results/orfeo-user-test

NVIDIA_METRICS_DEVICE=0 \
NVIDIA_METRICS_WINDOW_MS=200 \
NVIDIA_METRICS_OUTPUT_DIR="$PWD/results/orfeo-user-test" \
LD_PRELOAD="$PWD/dist/rocky-orfeo/root/usr/lib64/nvidia-process-metrics/libnvidia-process-metrics.so" \
  <PROGRAM> <ARGUMENTS>
```

For example, after building the included compute benchmark:

```bash
cmake -S benchmarks -B benchmarks/build-gpu \
  -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build-gpu -j2

NVIDIA_METRICS_DEVICE=0 \
NVIDIA_METRICS_WINDOW_MS=200 \
NVIDIA_METRICS_OUTPUT_DIR="$PWD/results/orfeo-user-test" \
LD_PRELOAD="$PWD/dist/rocky-orfeo/root/usr/lib64/nvidia-process-metrics/libnvidia-process-metrics.so" \
  ./benchmarks/build-gpu/bench_compute --duration 10 --warmup 5
```

Check the output:

```bash
find results/orfeo-user-test -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected files are:

```text
gpu_telemetry.csv
kernel_activity.csv
```

### Use the launcher without sudo

The launcher requires a daemon, but that daemon does not need root when it is
built with PM Sampling disabled. CUDA telemetry, NVML telemetry, and CUPTI
kernel activity remain enabled.

Build the dedicated user profile. Its default build and staging directories
are separate from the complete PM-enabled build:

```bash
cd ~/nvidia-profiling-test
module load cuda/12.8

python3 packaging/scripts/build-package.py stage \
  --platform rocky-orfeo-user
```

This creates `build-rocky-orfeo-user/` and
`dist/rocky-orfeo-user/root/` automatically.

Start the unprivileged daemon with a private socket and results directory:

```bash
cd ~/nvidia-profiling-test

USER_DAEMON_ROOT="${TMPDIR:-/tmp}/nvidia-process-metrics-$USER"
USER_STAGE="$PWD/dist/rocky-orfeo-user/root"

mkdir -p "$USER_DAEMON_ROOT/results"
chmod 700 "$USER_DAEMON_ROOT"

"$USER_STAGE/usr/sbin/nvidia-process-metrics-daemon" \
  --socket "$USER_DAEMON_ROOT/daemon.sock" \
  --results-root "$USER_DAEMON_ROOT/results" \
  >"$USER_DAEMON_ROOT/daemon.log" 2>&1 &

USER_DAEMON_PID=$!
echo "daemon PID: $USER_DAEMON_PID"
ls -l "$USER_DAEMON_ROOT/daemon.sock"
```

Run a command through the staged launcher:

```bash
"$USER_STAGE/usr/bin/nvidia-process-metrics-launcher" \
  --socket "$USER_DAEMON_ROOT/daemon.sock" \
  --library "$USER_STAGE/usr/lib64/nvidia-process-metrics/libnvidia-process-metrics.so" \
  --device 0 \
  --duration 12 \
  --window-ms 200 \
  --command -- \
  ./benchmarks/build-gpu/bench_compute --duration 10 --warmup 5
```

The launcher reports `running_without_pm_sampling` and prints the session
result directory. Expected outputs are `gpu_telemetry.csv` and
`kernel_activity.csv`; `pm_sampling.csv` is intentionally absent.

Stop the user daemon when finished:

```bash
kill "$USER_DAEMON_PID"
wait "$USER_DAEMON_PID" 2>/dev/null || true
```

The daemon and its `/tmp` results disappear independently of the system
service and do not modify `/usr`.

The two builds can coexist. Rebuilding `rocky-orfeo-user` does not modify
`build-rocky-orfeo`, and staging either profile does not install it.

### PM Sampling permissions

`pm_sampling_simple` uses NVIDIA hardware performance counters. Whether a
regular user may access those counters is controlled by the NVIDIA driver and
the cluster administrator. Building the binary as a regular user does not
grant that permission.

Test the current policy from the allocated GPU node:

```bash
mkdir -p /tmp/nvidia-pm-$USER
cd /tmp/nvidia-pm-$USER

~/nvidia-profiling-test/dist/rocky-orfeo/root/usr/lib64/nvidia-process-metrics/pm_sampling_simple \
  --device 0 \
  --duration 2 \
  --maxsamples 10000
```

If this reports insufficient permission or restricted profiling counters,
there is no user-space workaround. Ask the ORFEO administrator either to allow
performance-counter access for regular users or to install and operate the
root daemon. Do not use setuid on the profiler binaries.

## 5. System installation with administrator access

The complete launcher workflow uses a system daemon because PM Sampling may
require elevated access. After the build in section 3, an administrator can
install it with:

```bash
cd ~/nvidia-profiling-test
module load cuda/12.8

sudo "$(command -v meson)" install \
  -C build-rocky-orfeo

sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-process-metrics-daemon
```

Verify the installation:

```bash
sudo systemctl --no-pager --full status \
  nvidia-process-metrics-daemon

ls -l /run/nvidia-process-metrics/daemon.sock
nvidia-process-metrics-launcher --help
```

The installed files are:

```text
/usr/bin/nvidia-process-metrics-launcher
/usr/sbin/nvidia-process-metrics-daemon
/usr/lib64/nvidia-process-metrics/libnvidia-process-metrics.so
/usr/lib64/nvidia-process-metrics/pm_sampling_simple
/usr/lib/systemd/system/nvidia-process-metrics-daemon.service
```

## 6. Profile a command through the installed daemon

Run this inside a Slurm GPU allocation:

```bash
nvidia-process-metrics-launcher \
  --device 0 \
  --duration 12 \
  --window-ms 200 \
  --command -- \
  ./benchmarks/build-gpu/bench_compute --duration 10 --warmup 5
```

The launcher prints the session directory containing the resulting CSV files.
Only one PM Sampling session can be active on a GPU at a time.
