#include "benchmark_config.hpp"
#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"
#include "timing.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

__global__ void strided_read_kernel(float* destination, const float* source,
                                    std::size_t logical_elements,
                                    std::size_t stride) {
  const std::size_t first_logical =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t logical = first_logical; logical < logical_elements;
       logical += step) {
    const std::size_t index = logical * stride;
    destination[index] = source[index];
  }
}

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Controlled strided-read CUDA memory microbenchmark.\n\n"
      << "Options:\n"
      << "  --device <id>          CUDA device (default: 0)\n"
      << "  --duration <seconds>   Stable phase duration (default: 10)\n"
      << "  --warmup <iterations>  Warm-up kernel launches (default: 5)\n"
      << "  --block-size <threads> Threads per block (default: 256)\n"
      << "  --size <bytes>         Bytes per buffer (default: 268435456)\n"
      << "  --stride <N>           Source stride in float elements (default: 1)\n"
      << "  --help                  Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  float* source = nullptr;
  float* destination = nullptr;
  try {
    const benchmarks::ArgumentParser arguments(argc, argv);
    if (arguments.help_requested()) {
      print_help(argv[0]);
      return 0;
    }
    arguments.validate_options(
        {"--device", "--duration", "--warmup", "--block-size", "--size", "--stride"});
    const auto config = benchmarks::parse_common_config(arguments);
    const auto size_value =
        arguments.unsigned_value("--size", benchmarks::kDefaultBufferBytes);
    const auto stride_value = arguments.unsigned_value("--stride", 1);
    if (size_value == 0 || size_value > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("--size is out of range for this platform");
    }
    if (stride_value == 0 ||
        stride_value > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("--stride is out of range for this platform");
    }
    const auto bytes = static_cast<std::size_t>(size_value);
    const auto stride = static_cast<std::size_t>(stride_value);
    if (bytes % sizeof(float) != 0) {
      throw std::invalid_argument("--size must be a multiple of 4 bytes");
    }
    const std::size_t elements = bytes / sizeof(float);

    std::cout << "BENCHMARK=stride\nVARIANT=strided_read\n";
    benchmarks::print_common_config(config);
    std::cout << "BYTES=" << bytes << "\nSTRIDE=" << stride << '\n';

    const auto properties = benchmarks::initialize_cuda_device(config.device);
    if (config.block_size > properties.maxThreadsPerBlock) {
      throw std::invalid_argument("--block-size exceeds the device limit");
    }
    const std::size_t logical_elements = (elements - 1) / stride + 1;
    const std::size_t required_blocks =
        (logical_elements + static_cast<std::size_t>(config.block_size) - 1) /
        static_cast<std::size_t>(config.block_size);
    const auto grid_size = static_cast<int>(std::min<std::size_t>(
        required_blocks, static_cast<std::size_t>(properties.multiProcessorCount) * 32));

    CUDA_CHECK(cudaMalloc(&source, bytes));
    CUDA_CHECK(cudaMalloc(&destination, bytes));
    CUDA_CHECK(cudaMemset(source, 0x3f, bytes));
    CUDA_CHECK(cudaMemset(destination, 0, bytes));

    for (std::size_t launch = 0; launch < config.warmup_iterations; ++launch) {
      strided_read_kernel<<<grid_size, config.block_size>>>(destination, source,
                                                            logical_elements, stride);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::uint64_t launches = 0;
    const benchmarks::SteadyTimer timer;
    do {
      for (std::size_t batch = 0;
           batch < benchmarks::kLaunchesBetweenSynchronizations; ++batch) {
        strided_read_kernel<<<grid_size, config.block_size>>>(destination, source,
                                                              logical_elements, stride);
        CUDA_CHECK(cudaGetLastError());
        ++launches;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    } while (timer.elapsed_seconds() < config.duration_seconds);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = timer.elapsed_seconds();

    CUDA_CHECK(cudaFree(destination));
    destination = nullptr;
    CUDA_CHECK(cudaFree(source));
    source = nullptr;
    benchmarks::print_execution_summary(elapsed, launches);
    return 0;
  } catch (const std::exception& error) {
    if (destination != nullptr) cudaFree(destination);
    if (source != nullptr) cudaFree(source);
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
