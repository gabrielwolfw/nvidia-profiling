/**
 * @file process-metrics.cpp
 * @brief Preloaded collector for NVML telemetry and CUPTI kernel activity
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cuda.h>
#include <cupti.h>
#include <fcntl.h>
#include <nvml.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>  // NOLINT
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

// Kernel names are interned: name_id indexes kernel_names.
struct KernelEvent {
  uint64_t start;
  uint64_t end;
  uint32_t device_id = 0;
  uint32_t context_id = 0;
  uint32_t stream_id = 0;
  uint32_t correlation_id = 0;
  uint32_t name_id = 0;
};

struct MemcpyEvent {
  uint64_t start;
  uint64_t end;
  uint64_t bytes;
  uint32_t device_id;
  uint32_t context_id;
  uint32_t stream_id;
  uint32_t correlation_id;
  uint8_t copy_kind;
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
  std::optional<uint64_t> energy_mj;
};

struct CompletedActivityBuffer {
  uint8_t* buffer;
  size_t valid_size;
};

// Runtime switches read from NVIDIA_METRICS_* variables. Every behavior added
// by the profiler audit can be disabled to compare against the previous one.
struct ProfilerOptions {
  // Profile child processes too; each writes under children/<pid>.
  bool follow_children = false;
  // Parse CUPTI buffers in a dedicated thread instead of the CUPTI callback.
  bool activity_worker = true;
  // Preallocated CUPTI activity buffers; 0 allocates one per request.
  size_t activity_pool_buffers = 8;
  // CUPTI periodic flush in milliseconds; 0 keeps the CUPTI heuristics.
  uint32_t activity_flush_ms = 1000;
  // Stream unsorted kernel rows to kernel_activity.partial.csv during the run.
  bool kernel_journal = true;
  // Write individual memory copies to memcpy_activity.csv.
  bool memcpy_csv = true;
  // Read the NVML energy counter next to every power sample.
  bool nvml_fields = true;
  // Bracket only the power and energy reads with the query timestamps.
  bool power_bracket = true;
};

// Profiler state is global because the library has no entry point other than
// its load-time constructor and exit-time destructor.

// Kernel records live in a deque so that growth never copies existing ones.
static std::deque<KernelEvent> kernels;
// Deque elements stay in place, so kernel_name_ids can view their characters.
static std::deque<std::string> kernel_names;
static std::vector<std::string> escaped_kernel_names;
static std::unordered_map<std::string_view, uint32_t> kernel_name_ids;
static std::deque<MemcpyEvent> memcpy_events;
static std::mutex activity_mutex;
static std::atomic<size_t> dropped_activity_records{0};
static std::atomic<bool> dropped_records_query_failed{false};
static std::atomic<bool> activity_record_error_reported{false};

static uint64_t bytes_h2d = 0;
static uint64_t bytes_d2h = 0;
static uint64_t bytes_d2d = 0;

static nvmlDevice_t device;
static bool nvml_initialized = false;

static std::atomic<bool> metrics_sampling{false};
static std::thread* metrics_thread = nullptr;
static pid_t profiler_owner_pid = -1;
// False in processes that only inherited LD_PRELOAD; see ConfigureProcessScope.
static bool profiler_active = false;
static int profiler_completion_fd = -1;
static std::atomic<bool> profiler_failed{false};
static std::atomic<bool> activity_allocation_error_reported{false};

static ProfilerOptions options;
static std::filesystem::path output_directory(".");
static bool output_directory_ready = false;
// Written only by the sampling thread until ProfilerStop() joins it.
static std::ofstream telemetry_file;
// Guarded by activity_mutex.
static std::ofstream kernel_journal_file;
static uint64_t kernel_journal_rows = 0;

// Every pooled buffer, and the subset that CUPTI does not currently hold.
static std::mutex activity_pool_mutex;
static std::vector<uint8_t*> pooled_activity_buffers;
static std::vector<uint8_t*> free_activity_buffers;
static std::atomic<size_t> activity_pool_misses{0};

// Buffers handed over by BufferCompleted() for the activity worker.
static std::mutex completed_buffers_mutex;
static std::condition_variable completed_buffers_ready;
static std::deque<CompletedActivityBuffer> completed_buffers;
static bool activity_worker_accepting = false;
static std::thread* activity_worker_thread = nullptr;

constexpr char PROFILER_COMPLETION_SUCCESS = '0';
constexpr char PROFILER_COMPLETION_FAILURE = '1';

constexpr auto METRICS_SAMPLE_INTERVAL = std::chrono::milliseconds(50);
constexpr auto TELEMETRY_FLUSH_INTERVAL = std::chrono::seconds(1);

constexpr int DEFAULT_METRICS_WINDOW_MS = 200;
static std::chrono::milliseconds metrics_window_duration(
    DEFAULT_METRICS_WINDOW_MS);

constexpr size_t BUFFER_SIZE = 1024 * 1024;
constexpr size_t ACTIVITY_BUFFER_ALIGNMENT = 8;
constexpr size_t DEFAULT_ACTIVITY_POOL_BUFFERS = 8;
constexpr size_t MAX_ACTIVITY_POOL_BUFFERS = 256;
constexpr size_t MIN_QUEUED_ACTIVITY_BUFFERS = 2;
constexpr uint32_t DEFAULT_ACTIVITY_FLUSH_MS = 1000;

constexpr char KERNEL_CSV_HEADER[] =
    "event_id,pid,device_id,context_id,stream_id,correlation_id,"
    "start_ns,end_ns,duration_ns,kernel_name\n";

static void ProfilerStop();

static void InitializeCompletionChannel() {
  profiler_owner_pid = getpid();
  const char* configured = std::getenv("NVIDIA_METRICS_COMPLETION_FD");
  if (configured == nullptr || configured[0] == '\0') return;

  char* end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(configured, &end, 10);
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
    std::fprintf(stderr,
                 "[Profiler] Cannot configure completion descriptor: %s\n",
                 std::strerror(errno));
    return;
  }
  profiler_completion_fd = descriptor;
}

static void ReportCompletion() {
  if (profiler_completion_fd < 0 || getpid() != profiler_owner_pid) return;

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

// Returns a 0/1 switch, or fallback when the variable is unset or invalid.
static bool GetEnvSwitch(const char* name, bool fallback) {
  const char* configured = std::getenv(name);
  if (configured == nullptr || configured[0] == '\0') return fallback;
  if (std::strcmp(configured, "0") == 0) return false;
  if (std::strcmp(configured, "1") == 0) return true;

  std::fprintf(stderr, "[Profiler] Invalid %s: %s; using %d\n", name,
               configured, fallback ? 1 : 0);
  return fallback;
}

// Returns a non-negative integer no larger than max_value, or fallback when
// the variable is unset or invalid.
static uint64_t GetEnvInteger(const char* name, uint64_t fallback,
                              uint64_t max_value) {
  const char* configured = std::getenv(name);
  if (configured == nullptr || configured[0] == '\0') return fallback;

  char* end = nullptr;
  errno = 0;
  const uint64_t parsed = std::strtoull(configured, &end, 10);
  if (configured[0] == '-' || errno != 0 || end == configured || *end != '\0' ||
      parsed > max_value) {
    std::fprintf(stderr, "[Profiler] Invalid %s: %s; using %" PRIu64 "\n", name,
                 configured, fallback);
    return fallback;
  }
  return parsed;
}

static void LoadProfilerOptions() {
  options.follow_children =
      GetEnvSwitch("NVIDIA_METRICS_FOLLOW_CHILDREN", false);
  options.activity_worker =
      GetEnvSwitch("NVIDIA_METRICS_ACTIVITY_WORKER", true);
  options.activity_pool_buffers = static_cast<size_t>(
      GetEnvInteger("NVIDIA_METRICS_ACTIVITY_POOL_BUFFERS",
                    DEFAULT_ACTIVITY_POOL_BUFFERS, MAX_ACTIVITY_POOL_BUFFERS));
  options.activity_flush_ms = static_cast<uint32_t>(GetEnvInteger(
      "NVIDIA_METRICS_ACTIVITY_FLUSH_MS", DEFAULT_ACTIVITY_FLUSH_MS,
      std::numeric_limits<uint32_t>::max()));
  options.kernel_journal = GetEnvSwitch("NVIDIA_METRICS_KERNEL_JOURNAL", true);
  options.memcpy_csv = GetEnvSwitch("NVIDIA_METRICS_MEMCPY_CSV", true);
  options.nvml_fields = GetEnvSwitch("NVIDIA_METRICS_NVML_FIELDS", true);
  options.power_bracket = GetEnvSwitch("NVIDIA_METRICS_POWER_BRACKET", true);
}

// Decides whether this process profiles and where it writes. The first
// process that loads the library owns the top-level output and records its
// PID in NVIDIA_METRICS_OWNER_PID; a wrapper that execs the target keeps that
// PID and stays the owner. Other processes inherit LD_PRELOAD but stay idle,
// so that they never overwrite the owner's CSVs, unless
// NVIDIA_METRICS_FOLLOW_CHILDREN=1 sends them to children/<pid>.
// Returns false when this process must not profile.
static bool ConfigureProcessScope() {
  const char* configured = std::getenv("NVIDIA_METRICS_OUTPUT_DIR");
  if (configured != nullptr && configured[0] != '\0') {
    output_directory = configured;
  }

  const std::string own_pid = std::to_string(getpid());
  const char* owner = std::getenv("NVIDIA_METRICS_OWNER_PID");
  if (owner == nullptr || owner[0] == '\0') {
    setenv("NVIDIA_METRICS_OWNER_PID", own_pid.c_str(), 1);
    return true;
  }
  if (own_pid == owner) return true;

  if (!options.follow_children) return false;
  output_directory = output_directory / "children" / own_pid;
  return true;
}

static bool GetConfiguredDeviceIndex(int* device_index) {
  const char* configured = std::getenv("NVIDIA_METRICS_DEVICE");

  if (configured == nullptr || configured[0] == '\0') {
    *device_index = 0;
    return true;
  }

  char* end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(configured, &end, 10);

  if (errno != 0 || end == configured || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "[Profiler] Invalid NVIDIA_METRICS_DEVICE: %s\n",
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
  const char* configured = std::getenv("NVIDIA_METRICS_WINDOW_MS");

  if (configured == nullptr || configured[0] == '\0') {
    *window_duration = std::chrono::milliseconds(DEFAULT_METRICS_WINDOW_MS);
    return true;
  }

  char* end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(configured, &end, 10);

  if (errno != 0 || end == configured || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(
        stderr,
        "[Profiler] Invalid NVIDIA_METRICS_WINDOW_MS: %s; using %d ms\n",
        configured, DEFAULT_METRICS_WINDOW_MS);
    *window_duration = std::chrono::milliseconds(DEFAULT_METRICS_WINDOW_MS);
    return false;
  }

  *window_duration = std::chrono::milliseconds(parsed);
  return true;
}

static std::string EscapeCsv(const std::string& value) {
  std::string escaped = "\"";

  for (const char character : value) {
    if (character == '\"') escaped += '\"';
    escaped += character;
  }

  escaped += "\"";
  return escaped;
}

static void PrintCuptiError(const char* operation, CUptiResult result) {
  const char* message = "unknown CUPTI error";
  cuptiGetResultString(result, &message);
  std::fprintf(stderr, "[CUPTI] %s: %s\n", operation, message);
}

static bool FinishCsvFile(std::ofstream* file,
                          const std::filesystem::path& path) {
  file->flush();
  if (!*file) {
    std::fprintf(stderr, "[Profiler] Cannot write CSV file %s\n", path.c_str());
    file->close();
    return false;
  }
  file->close();
  if (!*file) {
    std::fprintf(stderr, "[Profiler] Cannot close CSV file %s\n", path.c_str());
    return false;
  }
  return true;
}

static std::filesystem::path TelemetryPath() {
  return output_directory / "gpu_telemetry.csv";
}

static std::filesystem::path KernelJournalPath() {
  return output_directory / "kernel_activity.partial.csv";
}

// Creates the output directory and opens the files that are written while
// the profiled process runs.
static bool PrepareOutputFiles() {
  std::error_code error;
  std::filesystem::create_directories(output_directory, error);
  if (error) {
    std::fprintf(stderr, "[Profiler] Cannot create output directory %s: %s\n",
                 output_directory.c_str(), error.message().c_str());
    return false;
  }
  output_directory_ready = true;

  telemetry_file.open(TelemetryPath());
  if (!telemetry_file.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 TelemetryPath().c_str());
    return false;
  }
  telemetry_file
      << "timestamp_ns,query_start_ns,query_end_ns,power_mw,temperature_c,"
         "gpu_utilization_pct,"
         "memory_utilization_pct,graphics_clock_mhz,"
         "energy_mj\n";

  if (options.kernel_journal) {
    kernel_journal_file.open(KernelJournalPath());
    if (!kernel_journal_file.is_open()) {
      std::fprintf(stderr,
                   "[Profiler] Cannot open %s; kernels are kept in memory "
                   "until exit\n",
                   KernelJournalPath().c_str());
      options.kernel_journal = false;
    } else {
      kernel_journal_file << KERNEL_CSV_HEADER;
    }
  }
  return true;
}

static void WriteKernelRow(std::ofstream* file, uint64_t event_id,
                           const KernelEvent& kernel) {
  *file << event_id << ',' << profiler_owner_pid << ',' << kernel.device_id
        << ',' << kernel.context_id << ',' << kernel.stream_id << ','
        << kernel.correlation_id << ',' << kernel.start << ',' << kernel.end
        << ',' << (kernel.end - kernel.start) << ','
        << escaped_kernel_names[kernel.name_id] << '\n';
}

static const char* MemcpyKindName(uint8_t copy_kind) {
  switch (copy_kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      return "unknown";
  }
}

// Writes the sorted kernel and memory copy CSVs that the analyzers read.
static bool SaveActivityCsvFiles() {
  if (!output_directory_ready) return false;

  bool success = true;

  const auto kernels_path = output_directory / "kernel_activity.csv";
  std::ofstream kernels_file(kernels_path);
  if (!kernels_file.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 kernels_path.c_str());
    success = false;
  } else {
    kernels_file << KERNEL_CSV_HEADER;
    for (size_t index = 0; index < kernels.size(); ++index) {
      WriteKernelRow(&kernels_file, index, kernels[index]);
    }
    if (!FinishCsvFile(&kernels_file, kernels_path)) success = false;
  }

  if (options.memcpy_csv) {
    const auto memcpy_path = output_directory / "memcpy_activity.csv";
    std::ofstream memcpy_file(memcpy_path);
    if (!memcpy_file.is_open()) {
      std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                   memcpy_path.c_str());
      success = false;
    } else {
      memcpy_file << "event_id,pid,device_id,context_id,stream_id,"
                     "correlation_id,start_ns,end_ns,duration_ns,bytes,"
                     "copy_kind\n";
      for (size_t index = 0; index < memcpy_events.size(); ++index) {
        const auto& copy = memcpy_events[index];
        memcpy_file << index << ',' << profiler_owner_pid << ','
                    << copy.device_id << ',' << copy.context_id << ','
                    << copy.stream_id << ',' << copy.correlation_id << ','
                    << copy.start << ',' << copy.end << ','
                    << (copy.end - copy.start) << ',' << copy.bytes << ','
                    << MemcpyKindName(copy.copy_kind) << '\n';
      }
      if (!FinishCsvFile(&memcpy_file, memcpy_path)) success = false;
    }
  }

  if (success) {
    std::printf("Timeline CSV: %s\nKernel CSV: %s\n", TelemetryPath().c_str(),
                kernels_path.c_str());
  }
  return success;
}

/* ------------------------------------------------ */
/* Local NVML sampling                              */
/* ------------------------------------------------ */

// Reads the energy counter in millijoules since the driver loaded. A device
// that does not support it is disabled so that later samples skip the call.
static std::optional<uint64_t> ReadEnergyCounter(bool* enabled,
                                                 bool* error_reported) {
  if (!*enabled) return std::nullopt;

  unsigned long long energy_mj = 0;  // NOLINT(runtime/int): NVML API type.
  const nvmlReturn_t result =
      nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
  if (result == NVML_SUCCESS) return static_cast<uint64_t>(energy_mj);

  if (!*error_reported) {
    std::fprintf(stderr, "[NVML] Cannot sample total energy: %s\n",
                 nvmlErrorString(result));
    *error_reported = true;
  }
  if (result == NVML_ERROR_NOT_SUPPORTED) *enabled = false;
  return std::nullopt;
}

static void WriteTelemetryRow(const GpuTelemetrySample& sample) {
  telemetry_file << sample.timestamp_ns << ',' << sample.query_start_ns << ','
                 << sample.query_end_ns << ',';
  if (sample.power_mw) telemetry_file << *sample.power_mw;
  telemetry_file << ',';
  if (sample.temperature_c) telemetry_file << *sample.temperature_c;
  telemetry_file << ',';
  if (sample.gpu_utilization) telemetry_file << *sample.gpu_utilization;
  telemetry_file << ',';
  if (sample.memory_utilization) telemetry_file << *sample.memory_utilization;
  telemetry_file << ',';
  if (sample.graphics_clock_mhz) telemetry_file << *sample.graphics_clock_mhz;
  telemetry_file << ',';
  if (sample.energy_mj) telemetry_file << *sample.energy_mj;
  telemetry_file << '\n';
}

void SampleGpuMetrics() {
  bool timestamp_error_reported = false;
  bool power_error_reported = false;
  bool temperature_error_reported = false;
  bool utilization_error_reported = false;
  bool clock_error_reported = false;
  bool energy_error_reported = false;
  bool energy_enabled = options.nvml_fields;
  bool write_error_reported = false;

  auto last_flush = std::chrono::steady_clock::now();

  while (true) {
    unsigned int power_mw = 0;
    unsigned int temperature_c = 0;
    unsigned int graphics_clock_mhz = 0;
    nvmlUtilization_t utilization{};

    uint64_t query_start_ns = 0;
    uint64_t query_end_ns = 0;
    CUptiResult timestamp_end_result = CUPTI_SUCCESS;

    const CUptiResult timestamp_start_result =
        cuptiGetTimestamp(&query_start_ns);

    const nvmlReturn_t power_result =
        nvmlDeviceGetPowerUsage(device, &power_mw);

    const std::optional<uint64_t> energy_mj =
        ReadEnergyCounter(&energy_enabled, &energy_error_reported);

    // The energy model integrates power, so keep the slower driver-averaged
    // counters outside the interval that dates the power sample.
    if (options.power_bracket) {
      timestamp_end_result = cuptiGetTimestamp(&query_end_ns);
    }

    const nvmlReturn_t temperature_result =
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature_c);

    const nvmlReturn_t utilization_result =
        nvmlDeviceGetUtilizationRates(device, &utilization);

    const nvmlReturn_t clock_result = nvmlDeviceGetClockInfo(
        device, NVML_CLOCK_GRAPHICS, &graphics_clock_mhz);

    if (!options.power_bracket) {
      timestamp_end_result = cuptiGetTimestamp(&query_end_ns);
    }

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
          query_start_ns + (query_end_ns - query_start_ns) / 2;

      WriteTelemetryRow({timestamp_ns, query_start_ns, query_end_ns,
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
                             : std::nullopt,
                         energy_mj});

    } else if (!timestamp_error_reported) {
      if (timestamp_start_result != CUPTI_SUCCESS)
        PrintCuptiError("Cannot timestamp the start of an NVML query",
                        timestamp_start_result);
      if (timestamp_end_result != CUPTI_SUCCESS)
        PrintCuptiError("Cannot timestamp the end of an NVML query",
                        timestamp_end_result);
      timestamp_error_reported = true;
    }

    // Flush periodically so that an abnormal exit keeps the samples so far.
    const auto now = std::chrono::steady_clock::now();
    if (now - last_flush >= TELEMETRY_FLUSH_INTERVAL) {
      telemetry_file.flush();
      last_flush = now;
    }
    if (!telemetry_file && !write_error_reported) {
      std::fprintf(stderr, "[Profiler] Cannot write CSV file %s\n",
                   TelemetryPath().c_str());
      profiler_failed.store(true, std::memory_order_release);
      write_error_reported = true;
    }

    // Take one final sample after the stop request to bracket the tail.
    if (!metrics_sampling.load(std::memory_order_acquire)) break;

    std::this_thread::sleep_for(METRICS_SAMPLE_INTERVAL);
  }
}

void StartGpuMetricsSampling() {
  if (!nvml_initialized || !telemetry_file.is_open() ||
      metrics_thread != nullptr) {
    return;
  }

  metrics_sampling.store(true, std::memory_order_release);

  try {
    metrics_thread = new std::thread(SampleGpuMetrics);
  } catch (const std::exception& error) {
    metrics_sampling.store(false, std::memory_order_release);

    std::fprintf(stderr, "[NVML] Failed to start local sampling: %s\n",
                 error.what());
  }
}

void StopGpuMetricsSampling() {
  metrics_sampling.store(false, std::memory_order_release);

  if (metrics_thread == nullptr) return;

  if (metrics_thread->joinable()) metrics_thread->join();

  delete metrics_thread;
  metrics_thread = nullptr;
}

/* ------------------------------------------------ */
/* CUPTI buffer                                     */
/* ------------------------------------------------ */

static void PreallocateActivityBuffers() {
  std::lock_guard<std::mutex> lock(activity_pool_mutex);
  for (size_t index = 0; index < options.activity_pool_buffers; ++index) {
    auto* buffer = static_cast<uint8_t*>(
        std::aligned_alloc(ACTIVITY_BUFFER_ALIGNMENT, BUFFER_SIZE));
    if (buffer == nullptr) {
      std::fprintf(stderr,
                   "[CUPTI] Preallocated %zu of %zu activity buffers; the "
                   "rest are allocated on demand\n",
                   index, options.activity_pool_buffers);
      break;
    }
    pooled_activity_buffers.push_back(buffer);
    free_activity_buffers.push_back(buffer);
  }
}

// Returns a pooled buffer to the pool, or frees one allocated on demand.
static void ReleaseActivityBuffer(uint8_t* buffer) {
  {
    std::lock_guard<std::mutex> lock(activity_pool_mutex);
    if (std::find(pooled_activity_buffers.begin(),
                  pooled_activity_buffers.end(),
                  buffer) != pooled_activity_buffers.end()) {
      free_activity_buffers.push_back(buffer);
      return;
    }
  }
  std::free(buffer);
}

// Frees the pooled buffers that CUPTI returned. A buffer CUPTI still holds is
// no longer pooled, so ReleaseActivityBuffer() frees it when it comes back,
// and any later request allocates on demand.
static void ReleaseActivityBufferPool() {
  std::lock_guard<std::mutex> lock(activity_pool_mutex);
  for (uint8_t* buffer : free_activity_buffers) std::free(buffer);
  free_activity_buffers.clear();
  pooled_activity_buffers.clear();
}

void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size,
                              size_t* max_records) {
  *buffer = nullptr;
  *size = 0;
  *max_records = 0;

  uint8_t* requested_buffer = nullptr;
  if (options.activity_pool_buffers > 0) {
    std::lock_guard<std::mutex> lock(activity_pool_mutex);
    if (!free_activity_buffers.empty()) {
      requested_buffer = free_activity_buffers.back();
      free_activity_buffers.pop_back();
    }
  }

  if (requested_buffer == nullptr) {
    if (options.activity_pool_buffers > 0) {
      activity_pool_misses.fetch_add(1, std::memory_order_relaxed);
    }
    requested_buffer = static_cast<uint8_t*>(
        std::aligned_alloc(ACTIVITY_BUFFER_ALIGNMENT, BUFFER_SIZE));
  }

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

// Returns the id of name, adding it on first use. Caller holds activity_mutex.
static uint32_t InternKernelName(const char* name) {
  const std::string_view key = name != nullptr ? name : "unknown";
  const auto found = kernel_name_ids.find(key);
  if (found != kernel_name_ids.end()) return found->second;

  const auto id = static_cast<uint32_t>(kernel_names.size());
  kernel_names.emplace_back(key);
  escaped_kernel_names.push_back(EscapeCsv(kernel_names.back()));
  kernel_name_ids.emplace(std::string_view(kernel_names.back()), id);
  return id;
}

// Copies the needed fields of every record in a completed CUPTI buffer.
static void ParseActivityBuffer(uint8_t* buffer, size_t valid_size) {
  std::lock_guard<std::mutex> lock(activity_mutex);

  CUpti_Activity* record = nullptr;
  CUptiResult result = CUPTI_SUCCESS;
  while (CUPTI_SUCCESS ==
         (result = cuptiActivityGetNextRecord(buffer, valid_size, &record))) {
    /* Kernel */
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      auto* kernel =
#if CUPTI_API_VERSION <= 26
          reinterpret_cast<CUpti_ActivityKernel9*>(
#else
          reinterpret_cast<CUpti_ActivityKernel10*>(
#endif
              record);

      kernels.push_back({kernel->start, kernel->end, kernel->deviceId,
                         kernel->contextId, kernel->streamId,
                         kernel->correlationId,
                         InternKernelName(kernel->name)});
      if (options.kernel_journal) {
        WriteKernelRow(&kernel_journal_file, kernel_journal_rows++,
                       kernels.back());
      }
    }

    /* Memory copies */
    if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* copy = reinterpret_cast<CUpti_ActivityMemcpy*>(record);

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

      if (options.memcpy_csv) {
        memcpy_events.push_back({copy->start, copy->end, copy->bytes,
                                 copy->deviceId, copy->contextId,
                                 copy->streamId, copy->correlationId,
                                 copy->copyKind});
      }
    }
  }

  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED &&
      !activity_record_error_reported.exchange(true,
                                               std::memory_order_relaxed)) {
    PrintCuptiError("Failed to read an activity record", result);
    profiler_failed.store(true, std::memory_order_release);
  }

  if (options.kernel_journal) {
    kernel_journal_file.flush();
    if (!kernel_journal_file) {
      std::fprintf(stderr,
                   "[Profiler] Cannot write %s; kernels are kept in memory "
                   "until exit\n",
                   KernelJournalPath().c_str());
      options.kernel_journal = false;
    }
  }
}

static void ProcessCompletedBuffers() {
  while (true) {
    CompletedActivityBuffer completed{};
    {
      std::unique_lock<std::mutex> lock(completed_buffers_mutex);
      completed_buffers_ready.wait(lock, [] {
        return !completed_buffers.empty() || !activity_worker_accepting;
      });
      if (completed_buffers.empty()) return;
      completed = completed_buffers.front();
      completed_buffers.pop_front();
    }
    ParseActivityBuffer(completed.buffer, completed.valid_size);
    ReleaseActivityBuffer(completed.buffer);
  }
}

static void StartActivityWorker() {
  if (!options.activity_worker) return;

  {
    std::lock_guard<std::mutex> lock(completed_buffers_mutex);
    activity_worker_accepting = true;
  }
  try {
    activity_worker_thread = new std::thread(ProcessCompletedBuffers);
  } catch (const std::exception& error) {
    {
      std::lock_guard<std::mutex> lock(completed_buffers_mutex);
      activity_worker_accepting = false;
    }
    std::fprintf(stderr,
                 "[CUPTI] Cannot start the activity worker; buffers are "
                 "parsed in the CUPTI callback: %s\n",
                 error.what());
  }
}

static size_t MaxQueuedActivityBuffers() {
  return std::max(options.activity_pool_buffers, MIN_QUEUED_ACTIVITY_BUFFERS);
}

// Stops accepting buffers and waits until the worker parsed every queued one.
static void StopActivityWorker() {
  {
    std::lock_guard<std::mutex> lock(completed_buffers_mutex);
    activity_worker_accepting = false;
  }
  completed_buffers_ready.notify_all();

  if (activity_worker_thread == nullptr) return;
  if (activity_worker_thread->joinable()) activity_worker_thread->join();
  delete activity_worker_thread;
  activity_worker_thread = nullptr;
}

void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream_id,
                              uint8_t* buffer, size_t, size_t valid_size) {
  size_t dropped = 0;
  const CUptiResult dropped_result =
      cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped);

  if (dropped_result == CUPTI_SUCCESS) {
    dropped_activity_records.fetch_add(dropped, std::memory_order_relaxed);
  } else if (!dropped_records_query_failed.exchange(
                 true, std::memory_order_relaxed)) {
    PrintCuptiError("Failed to query dropped activity records", dropped_result);
  }

  if (buffer == nullptr) return;

  // Hand the buffer to the worker so that CUPTI's thread only enqueues it.
  // A full queue means the worker is behind; parsing here then bounds the
  // memory held by queued buffers, as before the worker existed.
  {
    std::lock_guard<std::mutex> lock(completed_buffers_mutex);
    if (activity_worker_accepting &&
        completed_buffers.size() < MaxQueuedActivityBuffers()) {
      completed_buffers.push_back({buffer, valid_size});
      completed_buffers_ready.notify_one();
      return;
    }
  }

  ParseActivityBuffer(buffer, valid_size);
  ReleaseActivityBuffer(buffer);
}

/* ------------------------------------------------ */
/* NVML process memory                              */
/* ------------------------------------------------ */

uint64_t GetProcessMemory(pid_t pid) {
  if (!nvml_initialized) return 0;

  unsigned int count = 0;

  nvmlReturn_t res =
      nvmlDeviceGetComputeRunningProcesses_v3(device, &count, nullptr);

  if (NVML_ERROR_INSUFFICIENT_SIZE != res && NVML_SUCCESS != res) {
    return 0;
  }

  if (count == 0) return 0;

  std::vector<nvmlProcessInfo_t> processes(count);

  res =
      nvmlDeviceGetComputeRunningProcesses_v3(device, &count, processes.data());

  if (NVML_SUCCESS != res) return 0;

  for (const auto& process : processes) {
    if (process.pid == static_cast<unsigned int>(pid)) {
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
    if (kernel.start >= kernel.end) continue;

    if (intervals.empty() || kernel.start > intervals.back().end) {
      intervals.push_back({kernel.start, kernel.end});
      continue;
    }

    intervals.back().end = std::max(intervals.back().end, kernel.end);
  }

  return intervals;
}

static uint64_t GetActiveKernelTimeInWindow(
    const std::vector<ActiveKernelInterval>& intervals, size_t& interval_index,
    uint64_t window_start, uint64_t window_end) {
  uint64_t total = 0;
  while (interval_index < intervals.size() &&
         intervals[interval_index].end <= window_start) {
    ++interval_index;
  }

  size_t index = interval_index;
  while (index < intervals.size() && intervals[index].start < window_end) {
    const uint64_t overlap_start =
        std::max(intervals[index].start, window_start);
    const uint64_t overlap_end = std::min(intervals[index].end, window_end);
    total += overlap_end - overlap_start;

    if (intervals[index].end > window_end) break;

    ++index;
    interval_index = index;
  }

  return total;
}

/* ------------------------------------------------ */
/* Library loaded                                   */
/* ------------------------------------------------ */

static void ProfilerStart() {
  LoadProfilerOptions();
  if (!ConfigureProcessScope()) return;
  profiler_active = true;
  InitializeCompletionChannel();

  std::printf("\n=== NVIDIA Process Profiler ===\n");

  std::printf("PID: %d\n", getpid());

  GetConfiguredWindowDuration(&metrics_window_duration);
  std::printf("Kernel window: %" PRId64 " ms\n",
              static_cast<int64_t>(metrics_window_duration.count()));
  std::printf(
      "Options: follow_children=%d activity_worker=%d pool_buffers=%zu "
      "flush_ms=%" PRIu32
      " kernel_journal=%d memcpy_csv=%d nvml_fields=%d "
      "power_bracket=%d\n",
      options.follow_children, options.activity_worker,
      options.activity_pool_buffers, options.activity_flush_ms,
      options.kernel_journal, options.memcpy_csv, options.nvml_fields,
      options.power_bracket);

  if (!PrepareOutputFiles()) profiler_failed = true;

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
          std::printf("CUDA device: %d; NVML PCI device: %s\n", device_index,
                      pci_bus_id.c_str());
        } else {
          std::fprintf(stderr,
                       "[NVML] Cannot match CUDA device %d at PCI bus %s: %s\n",
                       device_index, pci_bus_id.c_str(), nvmlErrorString(res));
          nvmlShutdown();
        }
      } else {
        std::fprintf(stderr, "[NVML] Cannot initialize NVML: %s\n",
                     nvmlErrorString(res));
      }
    }
  }

  /* CUPTI */

  PreallocateActivityBuffers();
  StartActivityWorker();

  CUptiResult cupti_result =
      cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);

  if (CUPTI_SUCCESS != cupti_result) {
    std::fprintf(stderr, "[CUPTI] Failed to register callbacks\n");

    profiler_failed = true;
    return;
  }

  cupti_result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError("Failed to enable concurrent kernel activity",
                    cupti_result);
    profiler_failed = true;
    return;
  }

  cupti_result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError("Failed to enable memory copy activity", cupti_result);
    profiler_failed = true;
    return;
  }

  // The periodic flush only returns full buffers; exit still flushes all.
  if (options.activity_flush_ms > 0) {
    cupti_result = cuptiActivityFlushPeriod(options.activity_flush_ms);
    if (CUPTI_SUCCESS != cupti_result) {
      PrintCuptiError("Failed to set the activity flush period", cupti_result);
    }
  }

  std::printf("\n--- Live GPU telemetry ---\n");

  StartGpuMetricsSampling();
}

/* ------------------------------------------------ */
/* Profiler shutdown                                */
/* ------------------------------------------------ */

static void ProfilerStop() {
  if (!profiler_active || getpid() != profiler_owner_pid) return;

  const CUptiResult flush_result =
      cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

  if (flush_result != CUPTI_SUCCESS) {
    PrintCuptiError("Failed to flush activity records", flush_result);
    profiler_failed = true;
  }

  StopActivityWorker();
  ReleaseActivityBufferPool();

  // Keep real power samples beyond the last kernel for a full final window.
  if (metrics_thread != nullptr)
    std::this_thread::sleep_for(metrics_window_duration);

  StopGpuMetricsSampling();

  if (telemetry_file.is_open() &&
      !FinishCsvFile(&telemetry_file, TelemetryPath())) {
    profiler_failed = true;
  }

  std::sort(kernels.begin(), kernels.end(),
            [](const KernelEvent& left, const KernelEvent& right) {
              return left.start < right.start;
            });
  std::sort(memcpy_events.begin(), memcpy_events.end(),
            [](const MemcpyEvent& left, const MemcpyEvent& right) {
              return left.start < right.start;
            });

  const bool activity_saved = SaveActivityCsvFiles();
  if (!activity_saved) profiler_failed = true;

  // The sorted CSV supersedes the journal; keep the journal if it failed.
  if (kernel_journal_file.is_open()) {
    kernel_journal_file.close();
    if (activity_saved) {
      std::error_code error;
      std::filesystem::remove(KernelJournalPath(), error);
      if (error) {
        std::fprintf(stderr, "[Profiler] Cannot remove %s: %s\n",
                     KernelJournalPath().c_str(), error.message().c_str());
      }
    }
  }

  const uint64_t memory_bytes = GetProcessMemory(getpid());

  std::printf("\n--- Process Summary ---\n");

  std::printf("PID: %d\n", getpid());

  std::printf("Kernels total: %zu\n", kernels.size());
  std::printf("Distinct kernel names: %zu\n", kernel_names.size());

  const size_t dropped =
      dropped_activity_records.load(std::memory_order_relaxed);

  if (dropped > 0) {
    std::fprintf(stderr, "[CUPTI] WARNING: %zu activity records were dropped\n",
                 dropped);
  }

  const size_t pool_misses =
      activity_pool_misses.load(std::memory_order_relaxed);
  if (pool_misses > 0) {
    std::printf("Activity buffers allocated beyond the pool: %zu\n",
                pool_misses);
  }

  std::printf("GPU memory: %.2f MB\n", memory_bytes / (1024.0 * 1024.0));

  /* ---------------------------------------------- */
  /* Configurable kernel windows                    */
  /* ---------------------------------------------- */

  if (!kernels.empty()) {
    const std::vector<ActiveKernelInterval> active_kernel_intervals =
        MergeActiveKernelIntervals();

    const uint64_t window_size_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            metrics_window_duration)
            .count());

    uint64_t first_kernel_ns = kernels[0].start;

    uint64_t last_kernel_ns = kernels[0].end;

    for (const auto& kernel : kernels) {
      first_kernel_ns = std::min(first_kernel_ns, kernel.start);

      last_kernel_ns = std::max(last_kernel_ns, kernel.end);
    }

    std::printf("\n--- %" PRId64 " ms Kernel Windows ---\n",
                static_cast<int64_t>(metrics_window_duration.count()));

    uint64_t window_number = 0;
    size_t active_interval_index = 0;

    for (uint64_t window_start = first_kernel_ns; window_start < last_kernel_ns;
         window_start += window_size_ns) {
      uint64_t window_end = window_start + window_size_ns;

      const uint64_t current_window_ns = window_end - window_start;

      const uint64_t active_ns = GetActiveKernelTimeInWindow(
          active_kernel_intervals, active_interval_index, window_start,
          window_end);

      double utilization = 0.0;

      if (current_window_ns > 0) {
        utilization = static_cast<double>(active_ns) /
                      static_cast<double>(current_window_ns) * 100.0;
      }

      const double relative_start_ms = (window_start - first_kernel_ns) / 1e6;

      const double relative_end_ms = (window_end - first_kernel_ns) / 1e6;

      std::printf("Window %" PRIu64
                  " [%.0f-%.0f ms] | "
                  "Active: %.3f ms | "
                  "Utilization: %.2f %%\n",

                  window_number,

                  relative_start_ms,

                  relative_end_ms,

                  active_ns / 1e6,

                  utilization);

      ++window_number;
    }
  } else {
    std::printf("\nNo kernels captured\n");
    if (!options.follow_children) {
      std::printf(
          "If the command only starts the CUDA program, set "
          "NVIDIA_METRICS_FOLLOW_CHILDREN=1\n");
    }
  }

  /* ---------------------------------------------- */
  /* Memory copies                                  */
  /* ---------------------------------------------- */

  std::printf("\n--- Memory copies ---\n");

  std::printf("H2D: %.2f MB\n", bytes_h2d / (1024.0 * 1024.0));

  std::printf("D2H: %.2f MB\n", bytes_d2h / (1024.0 * 1024.0));

  std::printf("D2D: %.2f MB\n", bytes_d2d / (1024.0 * 1024.0));

  if (nvml_initialized) nvmlShutdown();

  std::printf("=======================\n");

  ReportCompletion();
}

class ProfilerLifecycle {
 public:
  ProfilerLifecycle() { ProfilerStart(); }

  ~ProfilerLifecycle() { ProfilerStop(); }

  ProfilerLifecycle(const ProfilerLifecycle&) = delete;
  ProfilerLifecycle& operator=(const ProfilerLifecycle&) = delete;
};

// This object must remain after the profiler state declarations in this
// translation unit. C++ destroys it first, ensuring ProfilerStop() joins the
// sampling and activity threads before the containers and synchronization
// objects are destroyed.
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
