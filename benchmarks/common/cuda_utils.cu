#include "cuda_utils.hpp"

#include <sstream>

namespace benchmarks {

void check_cuda(cudaError_t result, const char* expression, const char* file,
                int line) {
  if (result == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": CUDA call " << expression
          << " failed with " << cudaGetErrorName(result) << " ("
          << static_cast<int>(result) << "): " << cudaGetErrorString(result);
  throw CudaError(message.str());
}

cudaDeviceProp initialize_cuda_device(int device) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device >= device_count) {
    std::ostringstream message;
    message << "requested CUDA device " << device << ", but only "
            << device_count << " device(s) are available";
    throw std::invalid_argument(message.str());
  }
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaFree(nullptr));  // Materialize the primary context before timing.

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  return properties;
}

}  // namespace benchmarks
