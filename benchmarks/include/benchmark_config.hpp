#pragma once

#include <cstddef>

namespace benchmarks {

inline constexpr int kDefaultDevice = 0;
inline constexpr double kDefaultDurationSeconds = 10.0;
inline constexpr std::size_t kDefaultWarmupIterations = 5;
inline constexpr int kDefaultBlockSize = 256;
inline constexpr std::size_t kDefaultBufferBytes = 256ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kLaunchesBetweenSynchronizations = 16;

struct CommonConfig {
  int device = kDefaultDevice;
  double duration_seconds = kDefaultDurationSeconds;
  std::size_t warmup_iterations = kDefaultWarmupIterations;
  int block_size = kDefaultBlockSize;
};

}  // namespace benchmarks
