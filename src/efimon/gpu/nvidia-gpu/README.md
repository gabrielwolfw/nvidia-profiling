# NVIDIA collection library

`process-metrics.cpp` implements the existing preload library: NVML telemetry
and CUPTI kernel activity share a lifecycle and remain together in this layout
migration. Installed output remains `libnvidia-process-metrics.so`.

Shared CUDA/CUPTI helper headers live under
`include/efimon/gpu/nvidia-gpu/`. The PM sampler executable's main lives in
`src/tools/gpu/pm-sampling.cu`. Session orchestration belongs to
`src/tools/gpu/`; daemon and launcher mains live directly under `src/tools/`.

These directories prepare the standalone profiler for Efimon's subsystem
layout. This checkout does not contain Efimon's existing Observer interfaces
or `gpu/nvidia` implementation; integration with those interfaces is separate.
