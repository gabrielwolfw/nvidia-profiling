#include "benchmark_config.hpp"
#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"
#include "timing.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Repeated host/device CUDA transfer microbenchmark.\n\n"
      << "Options:\n"
      << "  --device <id>          CUDA device (default: 0)\n"
      << "  --duration <seconds>   Stable phase duration (default: 10)\n"
      << "  --warmup <iterations>  Warm-up transfers (default: 5)\n"
      << "  --block-size <threads> Accepted for CLI consistency (default: 256)\n"
      << "  --size <bytes>         Bytes per transfer (default: 268435456)\n"
      << "  --direction <h2d|d2h>  Transfer direction (default: h2d)\n"
      << "  --memory <type>        pinned or pageable (default: pinned)\n"
      << "  --help                  Show this help\n";
}

cudaMemcpyKind copy_kind(const std::string& direction) {
  if (direction == "h2d") return cudaMemcpyHostToDevice;
  if (direction == "d2h") return cudaMemcpyDeviceToHost;
  throw std::invalid_argument("--direction must be 'h2d' or 'd2h'");
}

}  // namespace

int main(int argc, char** argv) {
  void* device_buffer = nullptr;
  void* pinned_buffer = nullptr;
  std::unique_ptr<unsigned char[]> pageable_buffer;
  try {
    const benchmarks::ArgumentParser arguments(argc, argv);
    if (arguments.help_requested()) {
      print_help(argv[0]);
      return 0;
    }
    arguments.validate_options({"--device", "--duration", "--warmup",
                                "--block-size", "--size", "--direction",
                                "--memory"});
    const auto config = benchmarks::parse_common_config(arguments);
    const auto size_value =
        arguments.unsigned_value("--size", benchmarks::kDefaultBufferBytes);
    if (size_value == 0 || size_value > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("--size is out of range for this platform");
    }
    const auto bytes = static_cast<std::size_t>(size_value);
    const std::string direction =
        arguments.string_value("--direction", "h2d");
    const cudaMemcpyKind kind = copy_kind(direction);
    const std::string memory = arguments.string_value("--memory", "pinned");
    if (memory != "pinned" && memory != "pageable") {
      throw std::invalid_argument("--memory must be 'pinned' or 'pageable'");
    }

    std::cout << "BENCHMARK=transfer\nVARIANT=cuda_memcpy\n";
    benchmarks::print_common_config(config);
    std::cout << "BYTES=" << bytes << "\nDIRECTION=" << direction
              << "\nMEMORY=" << memory << '\n';

    benchmarks::initialize_cuda_device(config.device);
    CUDA_CHECK(cudaMalloc(&device_buffer, bytes));
    if (memory == "pinned") {
      CUDA_CHECK(cudaMallocHost(&pinned_buffer, bytes));
    } else {
      pageable_buffer = std::make_unique<unsigned char[]>(bytes);
    }
    auto* host_buffer = memory == "pinned"
                            ? static_cast<unsigned char*>(pinned_buffer)
                            : pageable_buffer.get();
    std::fill_n(host_buffer, bytes, static_cast<unsigned char>(0x5a));
    CUDA_CHECK(cudaMemset(device_buffer, 0x3c, bytes));

    void* destination = direction == "h2d" ? device_buffer : host_buffer;
    const void* source = direction == "h2d" ? host_buffer : device_buffer;
    for (std::size_t copy = 0; copy < config.warmup_iterations; ++copy) {
      CUDA_CHECK(cudaMemcpy(destination, source, bytes, kind));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::uint64_t copies = 0;
    const benchmarks::SteadyTimer timer;
    do {
      CUDA_CHECK(cudaMemcpy(destination, source, bytes, kind));
      ++copies;
    } while (timer.elapsed_seconds() < config.duration_seconds);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = timer.elapsed_seconds();

    if (pinned_buffer != nullptr) {
      CUDA_CHECK(cudaFreeHost(pinned_buffer));
      pinned_buffer = nullptr;
    }
    pageable_buffer.reset();
    CUDA_CHECK(cudaFree(device_buffer));
    device_buffer = nullptr;
    benchmarks::print_execution_summary(elapsed, copies);
    return 0;
  } catch (const std::exception& error) {
    if (pinned_buffer != nullptr) cudaFreeHost(pinned_buffer);
    if (device_buffer != nullptr) cudaFree(device_buffer);
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
