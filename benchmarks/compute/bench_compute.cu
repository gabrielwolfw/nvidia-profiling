#include "benchmark_config.hpp"
#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"
#include "timing.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

__global__ void fma_fp32_kernel(float* output, unsigned int iterations) {
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

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "FP32 FMA compute-bound CUDA microbenchmark.\n\n"
      << "Options:\n"
      << "  --device <id>          CUDA device (default: 0)\n"
      << "  --duration <seconds>   Stable phase duration (default: 10)\n"
      << "  --warmup <iterations>  Warm-up kernel launches (default: 5)\n"
      << "  --block-size <threads> Threads per block (default: 256)\n"
      << "  --iterations <N>       FMA loop iterations (default: 10000)\n"
      << "  --help                  Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  float* output = nullptr;
  try {
    const benchmarks::ArgumentParser arguments(argc, argv);
    if (arguments.help_requested()) {
      print_help(argv[0]);
      return 0;
    }
    arguments.validate_options(
        {"--device", "--duration", "--warmup", "--block-size", "--iterations"});
    const auto config = benchmarks::parse_common_config(arguments);
    const auto iterations_value = arguments.unsigned_value("--iterations", 10000);
    if (iterations_value == 0 ||
        iterations_value > std::numeric_limits<unsigned int>::max()) {
      throw std::invalid_argument("--iterations must be in the range [1, 2^32-1]");
    }
    const auto iterations = static_cast<unsigned int>(iterations_value);

    std::cout << "BENCHMARK=compute\nVARIANT=fma_fp32\n";
    benchmarks::print_common_config(config);
    std::cout << "ITERATIONS=" << iterations << '\n';

    const auto properties = benchmarks::initialize_cuda_device(config.device);
    if (config.block_size > properties.maxThreadsPerBlock) {
      throw std::invalid_argument("--block-size exceeds the device limit");
    }
    const int grid_size = properties.multiProcessorCount * 8;
    const std::size_t output_elements =
        static_cast<std::size_t>(grid_size) * config.block_size;
    CUDA_CHECK(cudaMalloc(&output, output_elements * sizeof(float)));

    for (std::size_t launch = 0; launch < config.warmup_iterations; ++launch) {
      fma_fp32_kernel<<<grid_size, config.block_size>>>(output, iterations);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::uint64_t launches = 0;
    const benchmarks::SteadyTimer timer;
    do {
      for (std::size_t batch = 0;
           batch < benchmarks::kLaunchesBetweenSynchronizations; ++batch) {
        fma_fp32_kernel<<<grid_size, config.block_size>>>(output, iterations);
        CUDA_CHECK(cudaGetLastError());
        ++launches;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    } while (timer.elapsed_seconds() < config.duration_seconds);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = timer.elapsed_seconds();

    CUDA_CHECK(cudaFree(output));
    output = nullptr;
    benchmarks::print_execution_summary(elapsed, launches);
    return 0;
  } catch (const std::exception& error) {
    if (output != nullptr) {
      cudaFree(output);
    }
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
