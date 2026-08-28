#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <vector>

#include <cupti.h>
#include <nvml.h>

struct KernelEvent {
  uint64_t start;
  uint64_t end;
};

static std::vector<KernelEvent> kernels;

static uint64_t bytes_h2d = 0;
static uint64_t bytes_d2h = 0;
static uint64_t bytes_d2d = 0;

constexpr size_t BUFFER_SIZE = 1024 * 1024;


/* CUPTI buffer */
void CUPTIAPI BufferRequested(uint8_t** buffer,
                              size_t* size,
                              size_t* max_records) {
  *buffer = static_cast<uint8_t*>(std::malloc(BUFFER_SIZE));
  *size = BUFFER_SIZE;
  *max_records = 0;
}


/* Process CUPTI records */
void CUPTIAPI BufferCompleted(CUcontext,
                              uint32_t,
                              uint8_t* buffer,
                              size_t,
                              size_t valid_size) {
  CUpti_Activity* record = nullptr;

  while (CUPTI_SUCCESS ==
         cuptiActivityGetNextRecord(buffer,
                                    valid_size,
                                    &record)) {

    /* Kernel */
    if (record->kind ==
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {

      auto* kernel =
          reinterpret_cast<CUpti_ActivityKernel9*>(record);

      kernels.push_back({
          kernel->start,
          kernel->end
      });

      std::printf(
          "Kernel: %s | Duration: %llu ns\n",
          kernel->name,
          static_cast<unsigned long long>(
              kernel->end - kernel->start));
    }

    /* Memory copies */
    if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {

      auto* copy =
          reinterpret_cast<CUpti_ActivityMemcpy*>(record);

      switch (copy->copyKind) {

        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
          bytes_h2d += copy->bytes;
          break;

        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
          bytes_d2h += copy->bytes;
          break;

        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
          bytes_d2d += copy->bytes;
          break;

        default:
          break;
      }
    }
  }

  std::free(buffer);
}


/* NVML memory used by this PID */
uint64_t GetProcessMemory(nvmlDevice_t device, pid_t pid) {

  unsigned int count = 0;

  nvmlReturn_t res =
      nvmlDeviceGetComputeRunningProcesses_v3(
          device,
          &count,
          nullptr);

  if (NVML_ERROR_INSUFFICIENT_SIZE != res &&
      NVML_SUCCESS != res) {
    return 0;
  }

  std::vector<nvmlProcessInfo_t> processes(count);

  res =
      nvmlDeviceGetComputeRunningProcesses_v3(
          device,
          &count,
          processes.data());

  if (NVML_SUCCESS != res)
    return 0;

  for (const auto& process : processes) {

    if (process.pid ==
        static_cast<unsigned int>(pid)) {

      return process.usedGpuMemory;
    }
  }

  return 0;
}


/* Merge kernel intervals to avoid double-counting concurrency */
uint64_t GetActiveKernelTime() {

  if (kernels.empty())
    return 0;

  std::sort(
      kernels.begin(),
      kernels.end(),
      [](const KernelEvent& a,
         const KernelEvent& b) {
        return a.start < b.start;
      });

  uint64_t total = 0;

  uint64_t start = kernels[0].start;
  uint64_t end = kernels[0].end;

  for (size_t i = 1; i < kernels.size(); ++i) {

    if (kernels[i].start <= end) {

      end = std::max(
          end,
          kernels[i].end);

    } else {

      total += end - start;

      start = kernels[i].start;
      end = kernels[i].end;
    }
  }

  total += end - start;

  return total;
}


int main() {

  const pid_t pid = getpid();

  std::printf("PID: %d\n\n", pid);


  /* ---------- NVML ---------- */

  nvmlInit_v2();

  nvmlDevice_t device;

  nvmlDeviceGetHandleByIndex_v2(
      0,
      &device);


  /* ---------- CUPTI ---------- */

  cuptiActivityRegisterCallbacks(
      BufferRequested,
      BufferCompleted);

  cuptiActivityEnable(
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

  cuptiActivityEnable(
      CUPTI_ACTIVITY_KIND_MEMCPY);


  /* Start measurement window */
  const auto start =
      std::chrono::steady_clock::now();


  /*
   * Aquí corre la aplicación CUDA.
   *
   * Para esta primera prueba puedes integrar este código
   * dentro de una aplicación o convertirlo después en
   * biblioteca LD_PRELOAD.
   */

  std::printf("Profiler running...\n");

  std::getchar();


  const auto end =
      std::chrono::steady_clock::now();


  /* Flush pending CUPTI events */
  cuptiActivityFlushAll(
      CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);


  /* ---------- Results ---------- */

  const uint64_t memory_bytes =
      GetProcessMemory(device, pid);

  const uint64_t active_ns =
      GetActiveKernelTime();

  const uint64_t window_ns =
      std::chrono::duration_cast<
          std::chrono::nanoseconds>(
          end - start)
          .count();

  double utilization = 0.0;

  if (window_ns > 0) {

    utilization =
        static_cast<double>(active_ns) /
        static_cast<double>(window_ns) *
        100.0;
  }


  std::printf("\n--- Process metrics ---\n");

  std::printf(
      "GPU memory: %.2f MB\n",
      memory_bytes /
          (1024.0 * 1024.0));

  std::printf(
      "Kernel active time: %.3f ms\n",
      active_ns / 1e6);

  std::printf(
      "Kernel-active utilization: %.2f %%\n",
      utilization);

  std::printf(
      "H2D: %.2f MB\n",
      bytes_h2d /
          (1024.0 * 1024.0));

  std::printf(
      "D2H: %.2f MB\n",
      bytes_d2h /
          (1024.0 * 1024.0));

  std::printf(
      "D2D: %.2f MB\n",
      bytes_d2d /
          (1024.0 * 1024.0));


  nvmlShutdown();

  return 0;
}


/* g++ -std=c++17 nvidia-process-metrics.cpp \
  -I/usr/local/cuda/include \
  -I/usr/local/cuda/extras/CUPTI/include \
  -L/usr/local/cuda/lib64 \
  -L/usr/local/cuda/extras/CUPTI/lib64 \
  -lcupti \
  -lnvidia-ml \
  -lcuda \
  -o nvidia-process-metrics */