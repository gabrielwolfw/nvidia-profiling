#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <cupti.h>
#include <cuda.h>
#include <nvml.h>


struct KernelEvent {
  uint64_t start;
  uint64_t end;
  uint32_t device_id = 0;
  uint32_t context_id = 0;
  uint32_t stream_id = 0;
  uint32_t correlation_id = 0;
  std::string name;
};

struct ActiveKernelInterval {
  uint64_t start;
  uint64_t end;
};

struct GpuTelemetrySample {
  uint64_t timestamp_ns;
  uint64_t query_start_ns;
  uint64_t query_end_ns;
  std::optional<unsigned int> power_mw;
  std::optional<unsigned int> temperature_c;
  std::optional<unsigned int> gpu_utilization;
  std::optional<unsigned int> memory_utilization;
  std::optional<unsigned int> graphics_clock_mhz;
};

static std::vector<KernelEvent> kernels;
static std::vector<GpuTelemetrySample> telemetry_samples;
static std::mutex activity_mutex;
static std::atomic<size_t> dropped_activity_records{0};
static std::atomic<bool> dropped_records_query_failed{false};

static uint64_t bytes_h2d = 0;
static uint64_t bytes_d2h = 0;
static uint64_t bytes_d2d = 0;

static nvmlDevice_t device;
static bool nvml_initialized = false;

static std::atomic<bool> metrics_sampling{false};
static std::thread* metrics_thread = nullptr;
static pid_t profiler_owner_pid = -1;
static int profiler_completion_fd = -1;
static std::atomic<bool> profiler_failed{false};
static std::atomic<bool> activity_allocation_error_reported{false};

constexpr char PROFILER_COMPLETION_SUCCESS = '0';
constexpr char PROFILER_COMPLETION_FAILURE = '1';

constexpr auto METRICS_SAMPLE_INTERVAL =
    std::chrono::milliseconds(50);

constexpr int DEFAULT_METRICS_WINDOW_MS = 200;
static std::chrono::milliseconds metrics_window_duration(
    DEFAULT_METRICS_WINDOW_MS);

constexpr size_t BUFFER_SIZE = 1024 * 1024;
constexpr size_t KERNELS_RESERVE_SIZE = 16384;
constexpr size_t TELEMETRY_RESERVE_SIZE = 512;


static void ProfilerStop();


static void InitializeCompletionChannel() {
  profiler_owner_pid = getpid();
  const char* configured = std::getenv("NVIDIA_METRICS_COMPLETION_FD");
  if (configured == nullptr || configured[0] == '\0')
    return;

  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(configured, &end, 10);
  unsetenv("NVIDIA_METRICS_COMPLETION_FD");
  if (errno != 0 || end == configured || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "[Profiler] Invalid completion descriptor: %s\n",
                 configured);
    return;
  }

  const int descriptor = static_cast<int>(parsed);
  const int flags = fcntl(descriptor, F_GETFD);
  if (flags < 0 || fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0) {
    std::fprintf(stderr, "[Profiler] Cannot configure completion descriptor: %s\n",
                 std::strerror(errno));
    return;
  }
  profiler_completion_fd = descriptor;
}


static void ReportCompletion() {
  if (profiler_completion_fd < 0 || getpid() != profiler_owner_pid)
    return;

  const char status = profiler_failed.load(std::memory_order_acquire)
                          ? PROFILER_COMPLETION_FAILURE
                          : PROFILER_COMPLETION_SUCCESS;
  ssize_t written = -1;
  do {
    written = write(profiler_completion_fd, &status, sizeof(status));
  } while (written < 0 && errno == EINTR);
  if (written != sizeof(status)) {
    std::fprintf(stderr, "[Profiler] Cannot report completion: %s\n",
                 written < 0 ? std::strerror(errno) : "short write");
  }
  close(profiler_completion_fd);
  profiler_completion_fd = -1;
}


static bool GetConfiguredDeviceIndex(int* device_index) {
  const char* configured =
      std::getenv("NVIDIA_METRICS_DEVICE");

  if (configured == nullptr || configured[0] == '\0') {
    *device_index = 0;
    return true;
  }

  char* end = nullptr;
  errno = 0;
  const long parsed =
      std::strtol(configured, &end, 10);

  if (errno != 0 || end == configured || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(
        stderr,
        "[Profiler] Invalid NVIDIA_METRICS_DEVICE: %s\n",
        configured);
    return false;
  }

  *device_index = static_cast<int>(parsed);
  return true;
}


static void PrintCudaError(const char* operation, CUresult result) {
  const char* name = "CUDA_ERROR_UNKNOWN";
  const char* message = "unknown CUDA driver error";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "[CUDA] %s: %s (%s)\n", operation, name, message);
}


static bool GetCudaPciBusId(int device_index, std::string* pci_bus_id) {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    PrintCudaError("Cannot initialize the CUDA driver", result);
    return false;
  }

  CUdevice cuda_device;
  result = cuDeviceGet(&cuda_device, device_index);
  if (result != CUDA_SUCCESS) {
    PrintCudaError("Cannot select the configured CUDA device", result);
    return false;
  }

  char identifier[32]{};
  result = cuDeviceGetPCIBusId(identifier, sizeof(identifier), cuda_device);
  if (result != CUDA_SUCCESS) {
    PrintCudaError("Cannot resolve the CUDA device PCI bus ID", result);
    return false;
  }

  *pci_bus_id = identifier;
  return true;
}


static bool GetConfiguredWindowDuration(
    std::chrono::milliseconds* window_duration) {
  const char* configured =
      std::getenv("NVIDIA_METRICS_WINDOW_MS");

  if (configured == nullptr || configured[0] == '\0') {
    *window_duration =
        std::chrono::milliseconds(DEFAULT_METRICS_WINDOW_MS);
    return true;
  }

  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(configured, &end, 10);

  if (errno != 0 || end == configured || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(
        stderr,
        "[Profiler] Invalid NVIDIA_METRICS_WINDOW_MS: %s; using %d ms\n",
        configured,
        DEFAULT_METRICS_WINDOW_MS);
    *window_duration =
        std::chrono::milliseconds(DEFAULT_METRICS_WINDOW_MS);
    return false;
  }

  *window_duration = std::chrono::milliseconds(parsed);
  return true;
}


static std::filesystem::path GetOutputDirectory() {
  const char* configured =
      std::getenv("NVIDIA_METRICS_OUTPUT_DIR");

  if (configured != nullptr && configured[0] != '\0')
    return configured;

  return ".";
}


static std::string EscapeCsv(const std::string& value) {
  std::string escaped = "\"";

  for (const char character : value) {
    if (character == '\"')
      escaped += '\"';
    escaped += character;
  }

  escaped += "\"";
  return escaped;
}


static void PrintCuptiError(
    const char* operation,
    CUptiResult result) {
  const char* message = "unknown CUPTI error";
  cuptiGetResultString(result, &message);
  std::fprintf(stderr, "[CUPTI] %s: %s\n", operation, message);
}


static bool FinishCsvFile(std::ofstream* file,
                          const std::filesystem::path& path) {
  file->flush();
  if (!*file) {
    std::fprintf(stderr, "[Profiler] Cannot write CSV file %s\n",
                 path.c_str());
    file->close();
    return false;
  }
  file->close();
  if (!*file) {
    std::fprintf(stderr, "[Profiler] Cannot close CSV file %s\n",
                 path.c_str());
    return false;
  }
  return true;
}


static bool SaveTimelineCsvFiles() {
  const std::filesystem::path output_directory =
      GetOutputDirectory();
  std::error_code error;
  std::filesystem::create_directories(
      output_directory,
      error);

  if (error) {
    std::fprintf(
        stderr,
        "[Profiler] Cannot create output directory %s: %s\n",
        output_directory.c_str(),
        error.message().c_str());
    return false;
  }

  bool success = true;

  const auto telemetry_path =
      output_directory / "gpu_telemetry.csv";
  std::ofstream telemetry_file(telemetry_path);
  if (!telemetry_file.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 telemetry_path.c_str());
    success = false;
  } else {
    telemetry_file <<
        "timestamp_ns,query_start_ns,query_end_ns,power_mw,temperature_c,"
        "gpu_utilization_pct,"
        "memory_utilization_pct,graphics_clock_mhz\n";

    for (const auto& sample : telemetry_samples) {
      telemetry_file
          << sample.timestamp_ns << ','
          << sample.query_start_ns << ','
          << sample.query_end_ns << ',';
      if (sample.power_mw)
        telemetry_file << *sample.power_mw;
      telemetry_file << ',';
      if (sample.temperature_c)
        telemetry_file << *sample.temperature_c;
      telemetry_file << ',';
      if (sample.gpu_utilization)
        telemetry_file << *sample.gpu_utilization;
      telemetry_file << ',';
      if (sample.memory_utilization)
        telemetry_file << *sample.memory_utilization;
      telemetry_file << ',';
      if (sample.graphics_clock_mhz)
        telemetry_file << *sample.graphics_clock_mhz;
      telemetry_file << '\n';
    }
    if (!FinishCsvFile(&telemetry_file, telemetry_path))
      success = false;
  }

  const auto kernels_path =
      output_directory / "kernel_activity.csv";
  std::ofstream kernels_file(kernels_path);
  if (!kernels_file.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 kernels_path.c_str());
    success = false;
  } else {
    kernels_file <<
        "event_id,pid,device_id,context_id,stream_id,correlation_id,"
        "start_ns,end_ns,duration_ns,kernel_name\n";

    for (size_t index = 0; index < kernels.size(); ++index) {
      const auto& kernel = kernels[index];
      kernels_file
          << index << ','
          << getpid() << ','
          << kernel.device_id << ','
          << kernel.context_id << ','
          << kernel.stream_id << ','
          << kernel.correlation_id << ','
          << kernel.start << ','
          << kernel.end << ','
          << (kernel.end - kernel.start) << ','
          << EscapeCsv(kernel.name) << '\n';
    }
    if (!FinishCsvFile(&kernels_file, kernels_path))
      success = false;
  }

  if (success) {
    std::printf(
        "Timeline CSV: %s\nKernel CSV: %s\n",
        telemetry_path.c_str(),
        kernels_path.c_str());
  }
  return success;
}


/* ------------------------------------------------ */
/* Local NVML sampling                              */
/* ------------------------------------------------ */

void SampleGpuMetrics() {

  bool timestamp_error_reported = false;
  bool power_error_reported = false;
  bool temperature_error_reported = false;
  bool utilization_error_reported = false;
  bool clock_error_reported = false;

  while (true) {

    unsigned int power_mw = 0;
    unsigned int temperature_c = 0;
    unsigned int graphics_clock_mhz = 0;
    nvmlUtilization_t utilization{};

    uint64_t query_start_ns = 0;
    uint64_t query_end_ns = 0;

    const CUptiResult timestamp_start_result =
        cuptiGetTimestamp(
            &query_start_ns);

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

    const CUptiResult timestamp_end_result =
        cuptiGetTimestamp(
            &query_end_ns);

    if (power_result != NVML_SUCCESS && !power_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample power: %s\n",
                   nvmlErrorString(power_result));
      power_error_reported = true;
    }
    if (temperature_result != NVML_SUCCESS && !temperature_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample temperature: %s\n",
                   nvmlErrorString(temperature_result));
      temperature_error_reported = true;
    }
    if (utilization_result != NVML_SUCCESS && !utilization_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample utilization: %s\n",
                   nvmlErrorString(utilization_result));
      utilization_error_reported = true;
    }
    if (clock_result != NVML_SUCCESS && !clock_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample graphics clock: %s\n",
                   nvmlErrorString(clock_result));
      clock_error_reported = true;
    }

    if (timestamp_start_result == CUPTI_SUCCESS &&
        timestamp_end_result == CUPTI_SUCCESS) {

      const uint64_t timestamp_ns =
          query_start_ns +
          (query_end_ns - query_start_ns) / 2;

      telemetry_samples.push_back({
          timestamp_ns,
          query_start_ns,
          query_end_ns,
          power_result == NVML_SUCCESS
              ? std::optional<unsigned int>(power_mw)
              : std::nullopt,
          temperature_result == NVML_SUCCESS
              ? std::optional<unsigned int>(temperature_c)
              : std::nullopt,
          utilization_result == NVML_SUCCESS
              ? std::optional<unsigned int>(utilization.gpu)
              : std::nullopt,
          utilization_result == NVML_SUCCESS
              ? std::optional<unsigned int>(utilization.memory)
              : std::nullopt,
          clock_result == NVML_SUCCESS
              ? std::optional<unsigned int>(graphics_clock_mhz)
              : std::nullopt
      });

    } else if (!timestamp_error_reported) {
      if (timestamp_start_result != CUPTI_SUCCESS)
        PrintCuptiError("Cannot timestamp the start of an NVML query",
                        timestamp_start_result);
      if (timestamp_end_result != CUPTI_SUCCESS)
        PrintCuptiError("Cannot timestamp the end of an NVML query",
                        timestamp_end_result);
      timestamp_error_reported = true;
    }

    // Take one final sample after the stop request to bracket the tail.
    if (!metrics_sampling.load(std::memory_order_acquire))
      break;

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
/* CUPTI buffer                                     */
/* ------------------------------------------------ */

void CUPTIAPI BufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* max_records) {

  *buffer = nullptr;
  *size = 0;
  *max_records = 0;

  uint8_t* requested_buffer =
      static_cast<uint8_t*>(std::malloc(BUFFER_SIZE));
  if (requested_buffer == nullptr) {
    profiler_failed.store(true, std::memory_order_release);
    if (!activity_allocation_error_reported.exchange(
            true, std::memory_order_relaxed)) {
      std::fprintf(stderr,
                   "[CUPTI] Cannot allocate activity buffer; profiling data "
                   "will be incomplete\n");
    }
    return;
  }

  *buffer = requested_buffer;
  *size = BUFFER_SIZE;
}


/* ------------------------------------------------ */
/* CUPTI records                                    */
/* ------------------------------------------------ */

void CUPTIAPI BufferCompleted(
    CUcontext context,
    uint32_t stream_id,
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
#if CUPTI_API_VERSION <= 26
          reinterpret_cast<CUpti_ActivityKernel9*>(
#else
          reinterpret_cast<CUpti_ActivityKernel10*>(
#endif
              record);

      std::lock_guard<std::mutex> lock(activity_mutex);
      kernels.push_back({
          kernel->start,
          kernel->end,
          kernel->deviceId,
          kernel->contextId,
          kernel->streamId,
          kernel->correlationId,
          kernel->name != nullptr ? kernel->name : "unknown"
      });

    }


    /* Memory copies */
    if (record->kind ==
        CUPTI_ACTIVITY_KIND_MEMCPY) {

      auto* copy =
          reinterpret_cast<CUpti_ActivityMemcpy*>(
              record);

      std::lock_guard<std::mutex> lock(activity_mutex);
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

  size_t dropped = 0;
  const CUptiResult dropped_result =
      cuptiActivityGetNumDroppedRecords(
          context,
          stream_id,
          &dropped);

  if (dropped_result == CUPTI_SUCCESS) {
    dropped_activity_records.fetch_add(
        dropped,
        std::memory_order_relaxed);
  } else if (!dropped_records_query_failed.exchange(
                 true,
                 std::memory_order_relaxed)) {
    PrintCuptiError(
        "Failed to query dropped activity records",
        dropped_result);
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

static std::vector<ActiveKernelInterval> MergeActiveKernelIntervals() {
  std::vector<ActiveKernelInterval> intervals;
  intervals.reserve(kernels.size());

  for (const auto& kernel : kernels) {
    if (kernel.start >= kernel.end)
      continue;

    if (intervals.empty() || kernel.start > intervals.back().end) {
      intervals.push_back({kernel.start, kernel.end});
      continue;
    }

    intervals.back().end = std::max(intervals.back().end, kernel.end);
  }

  return intervals;
}


static uint64_t GetActiveKernelTimeInWindow(
    const std::vector<ActiveKernelInterval>& intervals,
    size_t& interval_index,
    uint64_t window_start,
    uint64_t window_end) {

  uint64_t total = 0;
  while (interval_index < intervals.size() &&
         intervals[interval_index].end <= window_start) {
    ++interval_index;
  }

  size_t index = interval_index;
  while (index < intervals.size() && intervals[index].start < window_end) {
    const uint64_t overlap_start =
        std::max(intervals[index].start, window_start);
    const uint64_t overlap_end =
        std::min(intervals[index].end, window_end);
    total += overlap_end - overlap_start;

    if (intervals[index].end > window_end)
      break;

    ++index;
    interval_index = index;
  }

  return total;
}


/* ------------------------------------------------ */
/* Library loaded                                   */
/* ------------------------------------------------ */

static void ProfilerStart() {

  InitializeCompletionChannel();

  std::printf(
      "\n=== NVIDIA Process Profiler ===\n");

  std::printf(
      "PID: %d\n",
      getpid());

  kernels.reserve(KERNELS_RESERVE_SIZE);
  telemetry_samples.reserve(TELEMETRY_RESERVE_SIZE);

  GetConfiguredWindowDuration(&metrics_window_duration);
  std::printf(
      "Kernel window: %lld ms\n",
      static_cast<long long>(metrics_window_duration.count()));


  /* NVML */

  int device_index = 0;

  if (GetConfiguredDeviceIndex(&device_index)) {
    std::string pci_bus_id;
    if (GetCudaPciBusId(device_index, &pci_bus_id)) {
      nvmlReturn_t res = nvmlInit_v2();

      if (NVML_SUCCESS == res) {
        res = nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id.c_str(), &device);

        if (NVML_SUCCESS == res) {
          nvml_initialized = true;
          std::printf("CUDA device: %d; NVML PCI device: %s\n",
                      device_index, pci_bus_id.c_str());
        } else {
          std::fprintf(
              stderr,
              "[NVML] Cannot match CUDA device %d at PCI bus %s: %s\n",
              device_index,
              pci_bus_id.c_str(),
              nvmlErrorString(res));
          nvmlShutdown();
        }
      } else {
        std::fprintf(stderr, "[NVML] Cannot initialize NVML: %s\n",
                     nvmlErrorString(res));
      }
    }
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

    profiler_failed = true;
    return;
  }


  cupti_result = cuptiActivityEnable(
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError(
        "Failed to enable concurrent kernel activity",
        cupti_result);
    profiler_failed = true;
    return;
  }

  cupti_result = cuptiActivityEnable(
      CUPTI_ACTIVITY_KIND_MEMCPY);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError(
        "Failed to enable memory copy activity",
        cupti_result);
    profiler_failed = true;
    return;
  }


  std::printf(
      "\n--- Live GPU telemetry ---\n");

  StartGpuMetricsSampling();
}


/* ------------------------------------------------ */
/* Profiler shutdown                                */
/* ------------------------------------------------ */

static void ProfilerStop() {

  if (profiler_owner_pid >= 0 && getpid() != profiler_owner_pid)
    return;

  const CUptiResult flush_result =
      cuptiActivityFlushAll(
          CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

  if (flush_result != CUPTI_SUCCESS) {
    PrintCuptiError(
        "Failed to flush activity records",
        flush_result);
    profiler_failed = true;
  }

  // Keep real power samples beyond the last kernel for a full final window.
  if (metrics_thread != nullptr)
    std::this_thread::sleep_for(metrics_window_duration);

  StopGpuMetricsSampling();

  std::sort(
      kernels.begin(),
      kernels.end(),
      [](const KernelEvent& left,
         const KernelEvent& right) {
        return left.start < right.start;
      });

  if (!SaveTimelineCsvFiles())
    profiler_failed = true;


  const uint64_t memory_bytes =
      GetProcessMemory(
          getpid());


  std::printf(
      "\n--- Process Summary ---\n");

  std::printf(
      "PID: %d\n",
      getpid());

  std::printf(
      "Kernels total: %zu\n",
      kernels.size());

  const size_t dropped =
      dropped_activity_records.load(
          std::memory_order_relaxed);

  if (dropped > 0) {
    std::fprintf(
        stderr,
        "[CUPTI] WARNING: %zu activity records were dropped\n",
        dropped);
  }

  std::printf(
      "GPU memory: %.2f MB\n",
      memory_bytes /
          (1024.0 * 1024.0));


  /* ---------------------------------------------- */
  /* Configurable kernel windows                    */
  /* ---------------------------------------------- */

  if (!kernels.empty()) {

    const std::vector<ActiveKernelInterval> active_kernel_intervals =
        MergeActiveKernelIntervals();

    const uint64_t window_size_ns =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                metrics_window_duration).count());


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
        "\n--- %lld ms Kernel Windows ---\n",
        static_cast<long long>(metrics_window_duration.count()));


    uint64_t window_number = 0;
    size_t active_interval_index = 0;


    for (uint64_t window_start =
             first_kernel_ns;
         window_start < last_kernel_ns;
         window_start += window_size_ns) {


      uint64_t window_end =
          window_start +
          window_size_ns;


      const uint64_t current_window_ns =
          window_end -
          window_start;


      const uint64_t active_ns =
          GetActiveKernelTimeInWindow(
              active_kernel_intervals,
              active_interval_index,
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

  ReportCompletion();
}


class ProfilerLifecycle {
 public:
  ProfilerLifecycle() {
    ProfilerStart();
  }

  ~ProfilerLifecycle() {
    ProfilerStop();
  }

  ProfilerLifecycle(const ProfilerLifecycle&) = delete;
  ProfilerLifecycle& operator=(const ProfilerLifecycle&) = delete;
};


// This object must remain after the profiler state declarations in this
// translation unit. C++ destroys it first, ensuring ProfilerStop() joins the
// sampling thread before the vectors and synchronization objects are destroyed.
static ProfilerLifecycle profiler_lifecycle;


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


// LD_PRELOAD=$PWD/libnvidia-process-metrics.so python gpu-regression.py


/* LD_PRELOAD=$PWD/nvidia/libnvidia-process-metrics.so \
./benchmarks/build/bench_compute \
  --duration 10 \
  --warmup 5 */
