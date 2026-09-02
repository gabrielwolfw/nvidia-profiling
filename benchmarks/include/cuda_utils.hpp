#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace benchmarks {

class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void check_cuda(cudaError_t result, const char* expression, const char* file,
                int line);
cudaDeviceProp initialize_cuda_device(int device);

}  // namespace benchmarks

#define CUDA_CHECK(expression)                                                \
  ::benchmarks::check_cuda((expression), #expression, __FILE__, __LINE__)
