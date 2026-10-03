#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include <cstdio>
#include <cstdlib>

namespace nvidia_process_metrics {

inline void CheckCupti(CUptiResult result, const char* expression,
                       const char* file, int line) {
  if (result == CUPTI_SUCCESS) return;
  const char* message = "unknown CUPTI error";
  cuptiGetResultString(result, &message);
  std::fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expression,
               message);
  std::exit(EXIT_FAILURE);
}

inline void CheckCudaDriver(CUresult result, const char* expression,
                            const char* file, int line) {
  if (result == CUDA_SUCCESS) return;
  const char* name = "CUDA_ERROR_UNKNOWN";
  const char* message = "unknown CUDA driver error";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "%s:%d: %s failed: %s (%s)\n", file, line,
               expression, name, message);
  std::exit(EXIT_FAILURE);
}

inline void CheckCudaRuntime(cudaError_t result, const char* expression,
                             const char* file, int line) {
  if (result == cudaSuccess) return;
  std::fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expression,
               cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

}  // namespace nvidia_process_metrics

#define CUPTI_API_CALL(expression)                                           \
  ::nvidia_process_metrics::CheckCupti((expression), #expression, __FILE__, \
                                        __LINE__)
#define DRIVER_API_CALL(expression)                                         \
  ::nvidia_process_metrics::CheckCudaDriver((expression), #expression,      \
                                             __FILE__, __LINE__)
#define RUNTIME_API_CALL(expression)                                        \
  ::nvidia_process_metrics::CheckCudaRuntime((expression), #expression,     \
                                              __FILE__, __LINE__)
