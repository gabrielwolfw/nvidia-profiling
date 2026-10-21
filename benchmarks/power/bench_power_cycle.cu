#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kDefaultWarmupSeconds = 15.0;
constexpr double kDefaultMeasurementSeconds = 45.0;
constexpr double kDefaultCooldownSeconds = 15.0;
constexpr double kDefaultDutyCyclePct = 100.0;
constexpr double kDefaultPeriodMs = 100.0;
constexpr unsigned int kDefaultComputeIterations = 10000;

enum class Workload { kIdle, kCompute, kMemory, kMixed };

struct Config {
  int device = benchmarks::kDefaultDevice;
  int block_size = benchmarks::kDefaultBlockSize;
  double warmup_seconds = kDefaultWarmupSeconds;
  double measurement_seconds = kDefaultMeasurementSeconds;
  double cooldown_seconds = kDefaultCooldownSeconds;
  double duty_cycle_pct = kDefaultDutyCyclePct;
  double period_ms = kDefaultPeriodMs;
  std::size_t buffer_bytes = benchmarks::kDefaultBufferBytes;
  unsigned int compute_iterations = kDefaultComputeIterations;
  unsigned int mixed_compute_pct = 50;
  Workload workload = Workload::kCompute;
};

struct DeviceResources {
  float* compute_output = nullptr;
  float* memory_source = nullptr;
  float* memory_destination = nullptr;
  int compute_grid_size = 0;
  int memory_grid_size = 0;
  std::size_t memory_elements = 0;
};

struct PhaseStats {
  double elapsed_seconds = 0.0;
  double active_seconds = 0.0;
  std::uint64_t operations = 0;
  std::uint64_t period_overruns = 0;
};

__global__ void benchmark_warmup_start_marker() {}
__global__ void benchmark_measurement_start_marker() {}
__global__ void benchmark_cooldown_start_marker() {}
__global__ void benchmark_complete_marker() {}

__global__ void power_cycle_fma_kernel(float* output,
                                       unsigned int iterations) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float seed = 1.0F + static_cast<float>(index & 0xffU) * 0.0001F;
  float a0 = seed;
  float a1 = seed + 0.01F;
  float a2 = seed + 0.02F;
  float a3 = seed + 0.03F;
  float a4 = seed + 0.04F;
  float a5 = seed + 0.05F;
  float a6 = seed + 0.06F;
  float a7 = seed + 0.07F;

  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    a0 = fmaf(a0, 1.0000001F, 0.0000001F);
    a1 = fmaf(a1, 0.9999999F, 0.0000002F);
    a2 = fmaf(a2, 1.0000002F, -0.0000001F);
    a3 = fmaf(a3, 0.9999998F, 0.0000003F);
    a4 = fmaf(a4, 1.0000001F, -0.0000002F);
    a5 = fmaf(a5, 0.9999999F, 0.0000004F);
    a6 = fmaf(a6, 1.0000002F, -0.0000003F);
    a7 = fmaf(a7, 0.9999998F, 0.0000005F);
  }
  output[index] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

__global__ void power_cycle_memory_kernel(float* destination,
                                          const float* source,
                                          std::size_t elements) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = first; index < elements; index += step) {
    destination[index] = source[index];
  }
}

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Phased, duty-cycle-controlled CUDA power benchmark.\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --workload <name>         idle, compute, memory, or mixed\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 15)\n"
      << "  --duration <seconds>      Stable measurement phase (default: 45)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 15)\n"
      << "  --duty-cycle <percent>    Active part of each period [0, 100]\n"
      << "  --period-ms <value>       Duty-control period (default: 100)\n"
      << "  --block-size <threads>    CUDA block size (default: 256)\n"
      << "  --iterations <N>          Compute-kernel FMA iterations\n"
      << "  --size <bytes>            Bytes per memory buffer\n"
      << "  --mixed-compute-pct <N>   Compute operation share [0, 100]\n"
      << "  --help                     Show this help\n";
}

Workload parse_workload(const std::string& value) {
  if (value == "idle") return Workload::kIdle;
  if (value == "compute") return Workload::kCompute;
  if (value == "memory") return Workload::kMemory;
  if (value == "mixed") return Workload::kMixed;
  throw std::invalid_argument(
      "--workload must be idle, compute, memory, or mixed");
}

const char* workload_name(Workload workload) {
  switch (workload) {
    case Workload::kIdle:
      return "idle";
    case Workload::kCompute:
      return "compute";
    case Workload::kMemory:
      return "memory";
    case Workload::kMixed:
      return "mixed";
  }
  return "unknown";
}

Config parse_config(const benchmarks::ArgumentParser& arguments) {
  arguments.validate_options(
      {"--device", "--workload", "--warmup-seconds", "--duration",
       "--cooldown-seconds", "--duty-cycle", "--period-ms",
       "--block-size", "--iterations", "--size", "--mixed-compute-pct"});

  Config config;
  config.device = arguments.integer_value("--device", config.device);
  config.block_size =
      arguments.integer_value("--block-size", config.block_size);
  config.warmup_seconds =
      arguments.double_value("--warmup-seconds", config.warmup_seconds);
  config.measurement_seconds =
      arguments.double_value("--duration", config.measurement_seconds);
  config.cooldown_seconds =
      arguments.double_value("--cooldown-seconds", config.cooldown_seconds);
  config.duty_cycle_pct =
      arguments.double_value("--duty-cycle", config.duty_cycle_pct);
  config.period_ms = arguments.double_value("--period-ms", config.period_ms);
  config.workload =
      parse_workload(arguments.string_value("--workload", "compute"));

  const auto size_value =
      arguments.unsigned_value("--size", benchmarks::kDefaultBufferBytes);
  const auto iterations_value =
      arguments.unsigned_value("--iterations", kDefaultComputeIterations);
  const auto mixed_compute_value =
      arguments.unsigned_value("--mixed-compute-pct", 50);

  if (config.device < 0) {
    throw std::invalid_argument("--device must be non-negative");
  }
  if (config.block_size <= 0 || config.block_size > 1024) {
    throw std::invalid_argument("--block-size must be in the range [1, 1024]");
  }
  if (!(config.warmup_seconds >= 0.0) ||
      !(config.measurement_seconds > 0.0) ||
      !(config.cooldown_seconds >= 0.0)) {
    throw std::invalid_argument(
        "phase durations must be finite; measurement must be positive");
  }
  if (config.duty_cycle_pct < 0.0 || config.duty_cycle_pct > 100.0) {
    throw std::invalid_argument("--duty-cycle must be in the range [0, 100]");
  }
  if (!(config.period_ms > 0.0)) {
    throw std::invalid_argument("--period-ms must be greater than zero");
  }
  if (size_value == 0 ||
      size_value > std::numeric_limits<std::size_t>::max() ||
      size_value % sizeof(float) != 0) {
    throw std::invalid_argument(
        "--size must be a positive multiple of 4 representable by size_t");
  }
  if (iterations_value == 0 ||
      iterations_value > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument("--iterations must be in the range [1, 2^32-1]");
  }
  if (mixed_compute_value > 100) {
    throw std::invalid_argument("--mixed-compute-pct must be in [0, 100]");
  }
  if (config.workload == Workload::kIdle && config.duty_cycle_pct != 0.0) {
    throw std::invalid_argument("idle workload requires --duty-cycle 0");
  }

  config.buffer_bytes = static_cast<std::size_t>(size_value);
  config.compute_iterations = static_cast<unsigned int>(iterations_value);
  config.mixed_compute_pct = static_cast<unsigned int>(mixed_compute_value);
  return config;
}

void print_config(const Config& config) {
  std::cout << std::setprecision(12)
            << "BENCHMARK=power_cycle\n"
            << "WORKLOAD=" << workload_name(config.workload) << '\n'
            << "DEVICE=" << config.device << '\n'
            << "WARMUP_SECONDS=" << config.warmup_seconds << '\n'
            << "MEASUREMENT_SECONDS=" << config.measurement_seconds << '\n'
            << "COOLDOWN_SECONDS=" << config.cooldown_seconds << '\n'
            << "DUTY_CYCLE_PCT=" << config.duty_cycle_pct << '\n'
            << "PERIOD_MS=" << config.period_ms << '\n'
            << "BLOCK_SIZE=" << config.block_size << '\n'
            << "ITERATIONS=" << config.compute_iterations << '\n'
            << "BYTES=" << config.buffer_bytes << '\n'
            << "MIXED_COMPUTE_PCT=" << config.mixed_compute_pct << '\n';
}

enum class PhaseMarker { kWarmup, kMeasurement, kCooldown, kComplete };

void emit_phase(const char* phase, PhaseMarker marker) {
  switch (marker) {
    case PhaseMarker::kWarmup:
      benchmark_warmup_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kMeasurement:
      benchmark_measurement_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kCooldown:
      benchmark_cooldown_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kComplete:
      benchmark_complete_marker<<<1, 1>>>();
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "PHASE=" << phase << '\n' << std::flush;
}

void allocate_resources(const Config& config, const cudaDeviceProp& properties,
                        DeviceResources* resources) {
  const bool needs_compute = config.workload == Workload::kCompute ||
                             config.workload == Workload::kMixed;
  const bool needs_memory = config.workload == Workload::kMemory ||
                            config.workload == Workload::kMixed;
  if (needs_compute) {
    resources->compute_grid_size = properties.multiProcessorCount * 8;
    const std::size_t output_elements =
        static_cast<std::size_t>(resources->compute_grid_size) *
        static_cast<std::size_t>(config.block_size);
    CUDA_CHECK(cudaMalloc(&resources->compute_output,
                          output_elements * sizeof(float)));
  }
  if (needs_memory) {
    resources->memory_elements = config.buffer_bytes / sizeof(float);
    const std::size_t required_blocks =
        (resources->memory_elements + config.block_size - 1) /
        static_cast<std::size_t>(config.block_size);
    resources->memory_grid_size = static_cast<int>(std::min<std::size_t>(
        required_blocks,
        static_cast<std::size_t>(properties.multiProcessorCount) * 32));
    CUDA_CHECK(cudaMalloc(&resources->memory_source, config.buffer_bytes));
    CUDA_CHECK(cudaMalloc(&resources->memory_destination, config.buffer_bytes));
    CUDA_CHECK(cudaMemset(resources->memory_source, 0x3f, config.buffer_bytes));
    CUDA_CHECK(cudaMemset(resources->memory_destination, 0, config.buffer_bytes));
  }
}

void release_resources(DeviceResources* resources) noexcept {
  if (resources->memory_destination != nullptr) {
    cudaFree(resources->memory_destination);
    resources->memory_destination = nullptr;
  }
  if (resources->memory_source != nullptr) {
    cudaFree(resources->memory_source);
    resources->memory_source = nullptr;
  }
  if (resources->compute_output != nullptr) {
    cudaFree(resources->compute_output);
    resources->compute_output = nullptr;
  }
}

void launch_compute(const Config& config, const DeviceResources& resources) {
  power_cycle_fma_kernel<<<resources.compute_grid_size, config.block_size>>>(
      resources.compute_output, config.compute_iterations);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_memory(const Config& config, const DeviceResources& resources) {
  power_cycle_memory_kernel<<<resources.memory_grid_size, config.block_size>>>(
      resources.memory_destination, resources.memory_source,
      resources.memory_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_operation(const Config& config, const DeviceResources& resources,
                      unsigned int* mixed_accumulator) {
  if (config.workload == Workload::kCompute) {
    launch_compute(config, resources);
  } else if (config.workload == Workload::kMemory) {
    launch_memory(config, resources);
  } else if (config.workload == Workload::kMixed) {
    *mixed_accumulator += config.mixed_compute_pct;
    if (*mixed_accumulator >= 100) {
      *mixed_accumulator -= 100;
      launch_compute(config, resources);
    } else {
      launch_memory(config, resources);
    }
  }
}

Clock::duration seconds_to_duration(double seconds) {
  return std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(seconds));
}

PhaseStats run_paced_phase(const Config& config,
                           const DeviceResources& resources,
                           double phase_seconds) {
  PhaseStats stats;
  if (phase_seconds <= 0.0) return stats;

  const auto phase_start = Clock::now();
  const auto phase_end = phase_start + seconds_to_duration(phase_seconds);
  const auto period = seconds_to_duration(config.period_ms / 1000.0);
  const auto active = std::chrono::duration_cast<Clock::duration>(
      period * (config.duty_cycle_pct / 100.0));
  auto period_start = phase_start;
  unsigned int mixed_accumulator = 0;

  while (Clock::now() < phase_end) {
    const auto period_end = std::min(period_start + period, phase_end);
    const auto active_end = std::min(period_start + active, period_end);
    const auto active_start = Clock::now();
    const std::uint64_t operations_before_period = stats.operations;
    if (config.workload != Workload::kIdle &&
        config.duty_cycle_pct > 0.0) {
      while (Clock::now() < active_end) {
        launch_operation(config, resources, &mixed_accumulator);
        ++stats.operations;
      }
    }
    const auto work_end = Clock::now();
    if (stats.operations > operations_before_period) {
      stats.active_seconds +=
          std::chrono::duration<double>(work_end - active_start).count();
    }
    if (work_end > period_end) {
      ++stats.period_overruns;
      period_start = work_end;
    } else {
      std::this_thread::sleep_until(period_end);
      period_start = period_end;
    }
  }
  stats.elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - phase_start).count();
  return stats;
}

void print_phase_stats(const char* prefix, const PhaseStats& stats) {
  std::cout << std::fixed << std::setprecision(6)
            << prefix << "_ELAPSED_SECONDS=" << stats.elapsed_seconds << '\n'
            << prefix << "_ACTIVE_SECONDS=" << stats.active_seconds << '\n'
            << prefix << "_OPERATIONS=" << stats.operations << '\n'
            << prefix << "_PERIOD_OVERRUNS=" << stats.period_overruns << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  DeviceResources resources;
  try {
    const benchmarks::ArgumentParser arguments(argc, argv);
    if (arguments.help_requested()) {
      print_help(argv[0]);
      return 0;
    }
    const Config config = parse_config(arguments);
    print_config(config);

    const cudaDeviceProp properties =
        benchmarks::initialize_cuda_device(config.device);
    if (config.block_size > properties.maxThreadsPerBlock) {
      throw std::invalid_argument("--block-size exceeds the device limit");
    }
    allocate_resources(config, properties, &resources);

    emit_phase("warmup", PhaseMarker::kWarmup);
    const PhaseStats warmup =
        run_paced_phase(config, resources, config.warmup_seconds);

    emit_phase("measurement", PhaseMarker::kMeasurement);
    const PhaseStats measurement =
        run_paced_phase(config, resources, config.measurement_seconds);

    emit_phase("cooldown", PhaseMarker::kCooldown);
    const auto cooldown_start = Clock::now();
    std::this_thread::sleep_for(
        std::chrono::duration<double>(config.cooldown_seconds));
    const double cooldown_elapsed =
        std::chrono::duration<double>(Clock::now() - cooldown_start).count();

    emit_phase("complete", PhaseMarker::kComplete);
    release_resources(&resources);

    print_phase_stats("WARMUP", warmup);
    print_phase_stats("MEASUREMENT", measurement);
    std::cout << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed << '\n';
    benchmarks::print_execution_summary(measurement.elapsed_seconds,
                                        measurement.operations);
    return 0;
  } catch (const std::exception& error) {
    release_resources(&resources);
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
