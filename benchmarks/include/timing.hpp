#pragma once

#include <chrono>

namespace benchmarks {

class SteadyTimer {
 public:
  SteadyTimer() : start_(std::chrono::steady_clock::now()) {}

  double elapsed_seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace benchmarks
