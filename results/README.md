# Profiling result sessions

These sessions were captured on NVIDIA Jetson AGX Thor. They include NVML
telemetry, CUPTI kernel activity, PM Sampling data, derived CSV files, and
plots where available.

| Session | Workload | Analyzed time | Notes |
| --- | --- | ---: | --- |
| `20260911-184551-107039-compute` | Compute benchmark | 10.2 s | FP32 FMA synthetic workload |
| `65c0d52ac7833-5f090-33231704` | Memory benchmark | 10.2 s | Streaming-copy workload |
| `65c0d53d7067d-5f090-b27eb01b` | Stride benchmark | 10.2 s | Strided global-memory reads |
| `65c0d5504eb29-5f090-afe9075d` | Transfer benchmark | 10.6 s | Pinned H2D copies; no kernels captured |
| `65bd02b5b30d3-1a875-dcfed0a0` | Llama 3.1 8B | 10.4 s | Short inference capture |
| `65bfa110407c3-5f090-6968b83e` | Llama 3.1 8B | 14.4 s | 316,220 captured kernels |
| `65c3099f8acaf-5f090-9659a2de` | Llama 3.1 8B | 30.4 s | Extended inference capture; 671,080 kernels |

The raw power and PM counters describe the complete GPU. Per-kernel energy is
an attribution estimate based on the process kernel timeline, rather than a
physical energy measurement isolated to one kernel.

CSV files are stored with Git LFS because several raw captures exceed GitHub's
regular 100 MB file limit.
