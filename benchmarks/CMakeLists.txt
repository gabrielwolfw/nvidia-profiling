cmake_minimum_required(VERSION 3.21)

project(nvidia_energy_microbenchmarks LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_library(benchmark_common STATIC common/benchmark_utils.cpp)
target_include_directories(benchmark_common PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(benchmark_common PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_library(benchmark_cuda_common STATIC common/cuda_utils.cu)
target_include_directories(benchmark_cuda_common PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(benchmark_cuda_common PUBLIC CUDA::cudart)

function(add_cuda_benchmark target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE benchmark_common benchmark_cuda_common)
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endfunction()

add_cuda_benchmark(bench_compute compute/bench_compute.cu)
add_cuda_benchmark(bench_memory memory/bench_memory.cu)
add_cuda_benchmark(bench_stride memory/bench_stride.cu)
add_cuda_benchmark(bench_transfer transfer/bench_transfer.cu)
add_cuda_benchmark(bench_power_cycle power/bench_power_cycle.cu)
add_cuda_benchmark(bench_molecular_dynamics
  simulation/bench_molecular_dynamics.cu)
add_cuda_benchmark(bench_quantized_dnn neural/bench_quantized_dnn.cu)

# NVIDIA VPI ships with JetPack and as Ubuntu packages for x86_64; pass
# -Dvpi_DIR=<prefix>/lib/<arch>/cmake/vpi when it is not installed system-wide.
find_package(vpi 4 QUIET)
if(vpi_FOUND)
  add_cuda_benchmark(bench_vpi vision/bench_vpi.cu)
  target_link_libraries(bench_vpi PRIVATE vpi)
else()
  message(STATUS "VPI 4 not found: bench_vpi is not built")
endif()

add_executable(bench_idle baseline/bench_idle.cpp)
target_link_libraries(bench_idle PRIVATE benchmark_common benchmark_cuda_common)
target_compile_options(bench_idle PRIVATE -Wall -Wextra -Wpedantic)
