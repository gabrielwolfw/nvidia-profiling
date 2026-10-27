/**
 * @file cuda-api-checks.hpp
 * @brief Error-checking helpers for CUDA driver, runtime and CUPTI calls
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_CUDA_API_CHECKS_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_CUDA_API_CHECKS_HPP_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include <stdexcept>
#include <string>

namespace nvidia_process_metrics {

inline void CheckCupti(CUptiResult result, const char* expression,
                       const char* file, int line) {
  if (result == CUPTI_SUCCESS) return;
  const char* message = "unknown CUPTI error";
  cuptiGetResultString(result, &message);
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                           ": " + expression + " failed: " + message);
}

inline void CheckCudaDriver(CUresult result, const char* expression,
                            const char* file, int line) {
  if (result == CUDA_SUCCESS) return;
  const char* name = "CUDA_ERROR_UNKNOWN";
  const char* message = "unknown CUDA driver error";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                           ": " + expression + " failed: " + name + " (" +
                           message + ")");
}

inline void CheckCudaRuntime(cudaError_t result, const char* expression,
                             const char* file, int line) {
  if (result == cudaSuccess) return;
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                           ": " + expression +
                           " failed: " + cudaGetErrorString(result));
}

}  // namespace nvidia_process_metrics

#define CUPTI_API_CALL(expression)                                          \
  ::nvidia_process_metrics::CheckCupti((expression), #expression, __FILE__, \
                                       __LINE__)
#define DRIVER_API_CALL(expression)                                    \
  ::nvidia_process_metrics::CheckCudaDriver((expression), #expression, \
                                            __FILE__, __LINE__)
#define RUNTIME_API_CALL(expression)                                    \
  ::nvidia_process_metrics::CheckCudaRuntime((expression), #expression, \
                                             __FILE__, __LINE__)

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_CUDA_API_CHECKS_HPP_
