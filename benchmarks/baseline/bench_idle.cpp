#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"
#include "timing.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Idle baseline with an optional pre-created CUDA context.\n\n"
      << "Options:\n"
      << "  --device <id>          CUDA device for cuda-context mode (default: 0)\n"
      << "  --duration <seconds>   Idle phase duration (default: 10)\n"
      << "  --warmup <iterations>  Accepted for CLI consistency (default: 5)\n"
      << "  --block-size <threads> Accepted for CLI consistency (default: 256)\n"
      << "  --mode <mode>           system or cuda-context (default: system)\n"
      << "  --help                  Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const benchmarks::ArgumentParser arguments(argc, argv);
    if (arguments.help_requested()) {
      print_help(argv[0]);
      return 0;
    }
    arguments.validate_options(
        {"--device", "--duration", "--warmup", "--block-size", "--mode"});
    const auto config = benchmarks::parse_common_config(arguments);
    const std::string mode = arguments.string_value("--mode", "system");
    if (mode != "system" && mode != "cuda-context") {
      throw std::invalid_argument("--mode must be 'system' or 'cuda-context'");
    }

    std::cout << "BENCHMARK=idle\nVARIANT=" << mode << '\n';
    benchmarks::print_common_config(config);
    std::cout << "MODE=" << mode << '\n';

    if (mode == "cuda-context") {
      benchmarks::initialize_cuda_device(config.device);
    }

    const benchmarks::SteadyTimer timer;
    std::this_thread::sleep_for(
        std::chrono::duration<double>(config.duration_seconds));
    const double elapsed = timer.elapsed_seconds();

    benchmarks::print_execution_summary(elapsed, 0);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
