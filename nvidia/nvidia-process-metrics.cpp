#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cxxabi.h>
#include <exception>
#include <string>
#include <thread>
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

static nvmlDevice_t device;
static bool nvml_initialized = false;

static std::atomic<bool> metrics_sampling{false};
static std::thread* metrics_thread = nullptr;

constexpr auto METRICS_SAMPLE_INTERVAL =
    std::chrono::milliseconds(50);

constexpr size_t BUFFER_SIZE = 1024 * 1024;


/* ------------------------------------------------ */
/* Local NVML sampling                              */
/* ------------------------------------------------ */

void SampleGpuMetrics() {

  bool error_reported = false;

  while (metrics_sampling.load(
      std::memory_order_acquire)) {

    unsigned int power_mw = 0;
    unsigned int temperature_c = 0;
    unsigned int graphics_clock_mhz = 0;
    nvmlUtilization_t utilization{};

    uint64_t timestamp_ns = 0;

    const CUptiResult timestamp_result =
        cuptiGetTimestamp(
            &timestamp_ns);

    const nvmlReturn_t power_result =
        nvmlDeviceGetPowerUsage(
            device,
            &power_mw);

    const nvmlReturn_t temperature_result =
        nvmlDeviceGetTemperature(
            device,
            NVML_TEMPERATURE_GPU,
            &temperature_c);

    const nvmlReturn_t utilization_result =
        nvmlDeviceGetUtilizationRates(
            device,
            &utilization);

    const nvmlReturn_t clock_result =
        nvmlDeviceGetClockInfo(
            device,
            NVML_CLOCK_GRAPHICS,
            &graphics_clock_mhz);

    if (timestamp_result == CUPTI_SUCCESS &&
        power_result == NVML_SUCCESS &&
        temperature_result == NVML_SUCCESS &&
        utilization_result == NVML_SUCCESS &&
        clock_result == NVML_SUCCESS) {

      std::printf(
          "timestamp_ns: %lld | "
          "power: %u mW | "
          "temperature: %u C | "
          "gpu: %u %% | "
          "memory: %u %% | "
          "graphics_clock: %u MHz\n",
          static_cast<long long>(timestamp_ns),
          power_mw,
          temperature_c,
          utilization.gpu,
          utilization.memory,
          graphics_clock_mhz);

    } else if (!error_reported) {

      std::fprintf(
          stderr,
          "[NVML] One or more local GPU metrics could not be sampled\n");

      error_reported = true;
    }

    std::this_thread::sleep_for(
        METRICS_SAMPLE_INTERVAL);
  }
}


void StartGpuMetricsSampling() {

  if (!nvml_initialized ||
      metrics_thread != nullptr) {
    return;
  }

  metrics_sampling.store(
      true,
      std::memory_order_release);

  try {

    metrics_thread =
        new std::thread(
            SampleGpuMetrics);

  } catch (const std::exception& error) {

    metrics_sampling.store(
        false,
        std::memory_order_release);

    std::fprintf(
        stderr,
        "[NVML] Failed to start local sampling: %s\n",
        error.what());
  }
}


void StopGpuMetricsSampling() {

  metrics_sampling.store(
      false,
      std::memory_order_release);

  if (metrics_thread == nullptr)
    return;

  if (metrics_thread->joinable())
    metrics_thread->join();

  delete metrics_thread;
  metrics_thread = nullptr;
}


/* ------------------------------------------------ */
/* Demangle kernel name                             */
/* ------------------------------------------------ */

std::string Demangle(const char* name) {

  if (nullptr == name)
    return "unknown";

  int status = 0;

  char* result =
      abi::__cxa_demangle(
          name,
          nullptr,
          nullptr,
          &status);

  if (0 != status || nullptr == result)
    return name;

  std::string demangled(result);

  std::free(result);

  return demangled;
}


/* ------------------------------------------------ */
/* CUPTI buffer                                     */
/* ------------------------------------------------ */

void CUPTIAPI BufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* max_records) {

  *buffer =
      static_cast<uint8_t*>(
          std::malloc(BUFFER_SIZE));

  *size = BUFFER_SIZE;
  *max_records = 0;
}


/* ------------------------------------------------ */
/* CUPTI records                                    */
/* ------------------------------------------------ */

void CUPTIAPI BufferCompleted(
    CUcontext,
    uint32_t,
    uint8_t* buffer,
    size_t,
    size_t valid_size) {

  CUpti_Activity* record = nullptr;

  while (CUPTI_SUCCESS ==
         cuptiActivityGetNextRecord(
             buffer,
             valid_size,
             &record)) {

    /* Kernel */
    if (record->kind ==
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {

      auto* kernel =
          reinterpret_cast<CUpti_ActivityKernel9*>(
              record);

      kernels.push_back({
          kernel->start,
          kernel->end
      });

      const std::string name =
          Demangle(kernel->name);

      std::printf(
          "[CUPTI] Kernel: %s | Duration: %.3f us\n",
          name.c_str(),
          (kernel->end - kernel->start) / 1000.0);
    }


    /* Memory copies */
    if (record->kind ==
        CUPTI_ACTIVITY_KIND_MEMCPY) {

      auto* copy =
          reinterpret_cast<CUpti_ActivityMemcpy*>(
              record);

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


/* ------------------------------------------------ */
/* NVML process memory                              */
/* ------------------------------------------------ */

uint64_t GetProcessMemory(pid_t pid) {

  if (!nvml_initialized)
    return 0;

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

  if (count == 0)
    return 0;

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


/* ------------------------------------------------ */
/* Active kernel time inside window                 */
/* ------------------------------------------------ */

uint64_t GetActiveKernelTimeInWindow(
    uint64_t window_start,
    uint64_t window_end) {

  std::vector<KernelEvent> events;

  for (const auto& kernel : kernels) {

    const uint64_t start =
        std::max(
            kernel.start,
            window_start);

    const uint64_t end =
        std::min(
            kernel.end,
            window_end);

    if (start < end) {

      events.push_back({
          start,
          end
      });
    }
  }


  if (events.empty())
    return 0;


  std::sort(
      events.begin(),
      events.end(),
      [](const KernelEvent& a,
         const KernelEvent& b) {

        return a.start < b.start;
      });


  uint64_t total = 0;

  uint64_t current_start =
      events[0].start;

  uint64_t current_end =
      events[0].end;


  for (size_t i = 1;
       i < events.size();
       ++i) {

    if (events[i].start <= current_end) {

      current_end =
          std::max(
              current_end,
              events[i].end);

    } else {

      total +=
          current_end -
          current_start;

      current_start =
          events[i].start;

      current_end =
          events[i].end;
    }
  }


  total +=
      current_end -
      current_start;

  return total;
}


/* ------------------------------------------------ */
/* Library loaded                                   */
/* ------------------------------------------------ */

__attribute__((constructor))
void ProfilerStart() {

  std::printf(
      "\n=== NVIDIA Process Profiler ===\n");

  std::printf(
      "PID: %d\n",
      getpid());


  /* NVML */

  nvmlReturn_t res =
      nvmlInit_v2();

  if (NVML_SUCCESS == res) {

    res =
        nvmlDeviceGetHandleByIndex_v2(
            0,
            &device);

    if (NVML_SUCCESS == res)
      nvml_initialized = true;
  }


  /* CUPTI */

  CUptiResult cupti_result =
      cuptiActivityRegisterCallbacks(
          BufferRequested,
          BufferCompleted);

  if (CUPTI_SUCCESS != cupti_result) {

    std::fprintf(
        stderr,
        "[CUPTI] Failed to register callbacks\n");

    return;
  }


  cuptiActivityEnable(
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

  cuptiActivityEnable(
      CUPTI_ACTIVITY_KIND_MEMCPY);


  StartGpuMetricsSampling();


  std::printf(
      "Profiler started\n\n");
}


/* ------------------------------------------------ */
/* Library unloaded                                 */
/* ------------------------------------------------ */

__attribute__((destructor))
void ProfilerStop() {

  StopGpuMetricsSampling();

  cuptiActivityFlushAll(
      CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);


  const uint64_t memory_bytes =
      GetProcessMemory(
          getpid());


  std::printf(
      "\n=== Process Metrics ===\n");

  std::printf(
      "PID: %d\n",
      getpid());

  std::printf(
      "Kernels total: %zu\n",
      kernels.size());

  std::printf(
      "GPU memory: %.2f MB\n",
      memory_bytes /
          (1024.0 * 1024.0));


  /* ---------------------------------------------- */
  /* 200 ms windows                                 */
  /* ---------------------------------------------- */

  if (!kernels.empty()) {

    const uint64_t window_size_ns =
        200000000ULL;  // 200 ms


    uint64_t first_kernel_ns =
        kernels[0].start;

    uint64_t last_kernel_ns =
        kernels[0].end;


    for (const auto& kernel : kernels) {

      first_kernel_ns =
          std::min(
              first_kernel_ns,
              kernel.start);

      last_kernel_ns =
          std::max(
              last_kernel_ns,
              kernel.end);
    }


    std::printf(
        "\n--- 200 ms Windows ---\n");


    uint64_t window_number = 0;


    for (uint64_t window_start =
             first_kernel_ns;
         window_start < last_kernel_ns;
         window_start += window_size_ns) {


      uint64_t window_end =
          window_start +
          window_size_ns;


      /*
       * Last window can be shorter
       */
      if (window_end > last_kernel_ns) {
        window_end =
            last_kernel_ns;
      }


      const uint64_t current_window_ns =
          window_end -
          window_start;


      const uint64_t active_ns =
          GetActiveKernelTimeInWindow(
              window_start,
              window_end);


      double utilization = 0.0;

      if (current_window_ns > 0) {

        utilization =
            static_cast<double>(
                active_ns) /
            static_cast<double>(
                current_window_ns) *
            100.0;
      }


      const double relative_start_ms =
          (window_start -
           first_kernel_ns) /
          1e6;


      const double relative_end_ms =
          (window_end -
           first_kernel_ns) /
          1e6;


      std::printf(
          "Window %llu "
          "[%.0f-%.0f ms] | "
          "Active: %.3f ms | "
          "Utilization: %.2f %%\n",

          static_cast<unsigned long long>(
              window_number),

          relative_start_ms,

          relative_end_ms,

          active_ns / 1e6,

          utilization);


      ++window_number;
    }
  }

  else {

    std::printf(
        "\nNo kernels captured\n");
  }


  /* ---------------------------------------------- */
  /* Memory copies                                  */
  /* ---------------------------------------------- */

  std::printf(
      "\n--- Memory copies ---\n");

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


  if (nvml_initialized)
    nvmlShutdown();


  std::printf(
      "=======================\n");
}


/* g++ -std=c++17 -fPIC -shared \
  nvidia-process-metrics.cpp \
  -I/usr/local/cuda/include \
  -I/usr/local/cuda/extras/CUPTI/include \
  -L/usr/local/cuda/lib64 \
  -L/usr/local/cuda/extras/CUPTI/lib64 \
  -lcupti \
  -lnvidia-ml \
  -lcuda \
  -pthread \
  -o libnvidia-process-metrics.so */


// LD_PRELOAD=$PWD/libnvidia-process-metrics.so \
python gpu-regression.py


/* LD_PRELOAD=$PWD/nvidia/libnvidia-process-metrics.so \
./benchmarks/build/bench_compute \
  --duration 10 \
  --warmup 5 */