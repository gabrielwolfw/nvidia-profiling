# NVIDIA Process Metrics daemon

`nvidia-process-metrics-daemon` owns the privileged PM Sampling collector.
`nvidia-process-metrics-launcher` is the user-facing client. They communicate
through a Unix domain socket and create one result directory per session.

## Build and install on Thor

From the repository root:

```bash
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

Check that the daemon is ready:

```bash
systemctl status nvidia-process-metrics-daemon
ls -l /run/nvidia-process-metrics/daemon.sock
```

The service runs as root because CUPTI PM Sampling requires it. The launcher
uses the caller's Unix-socket credentials to create results under:

```text
/tmp/nvidia-process-metrics/<uid>/<session-id>/
```

## Full profiling of a launched command

```bash
nvidia-process-metrics-launcher \
  --device 0 --duration 10 --window-ms 200 \
  --command -- ./benchmarks/build/bench_compute --duration 10 --warmup 5
```

The launcher starts PM Sampling, then releases the target process with
`LD_PRELOAD=libnvidia-process-metrics.so`. The result directory can contain
`kernel_activity.csv`, `gpu_telemetry.csv`, `pm_sampling.csv`, and
`pm_sampling.log`.

## Observe an existing process

```bash
nvidia-process-metrics-launcher \
  --pid 12345 --observe --device 0 --duration 10
```

This mode can observe the lifetime of an already-running PID and collect the
device-global PM Sampling output. It cannot retroactively collect CUPTI kernel
activity from that process, so it does not provide per-kernel attribution.

Only one active PM Sampling session is allowed per GPU.
