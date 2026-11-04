// Quantized deep neural network (QDNN) inference benchmark.
//
// Runs batched inference of a deep multilayer perceptron (default
// 1024 -> 4096 -> 4096 -> 4096 -> 1024, ReLU between layers) in one of three
// precisions with the same tiled GEMM on CUDA cores:
//
//   fp32  float weights and activations, one FMA per 32-bit word
//   fp16  half weights and activations, __hfma2 (two MACs per word), partial
//         sums promoted to float every tile
//   int8  symmetric INT8 weights (per output channel) and activations (per
//         layer), __dp4a (four MACs per word), INT32 accumulation, float
//         dequantization and INT8 requantization in the epilogue
//
// Every precision moves the same number of 32-bit words per tile, so the
// comparison isolates arithmetic density and memory footprint. Tensor cores
// are not used: the V100 has none for INT8, and __hfma2 and __dp4a exist on
// every GPU from compute capability 6.1, including Jetson Thor.
//
// The INT8 activation scales are calibrated on the fp32 activations of the
// input batch. Each run reports the logit error and top-1 agreement against
// fp32. Like bench_power_cycle, the run has warm-up, measurement and cooldown
// phases delimited by empty marker kernels. --mode verify checks one layer of
// each precision against a host reference.

#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kDefaultWarmupSeconds = 10.0;
constexpr double kDefaultMeasurementSeconds = 40.0;
constexpr double kDefaultCooldownSeconds = 10.0;
constexpr int kDefaultBatch = 256;
constexpr int kDefaultInputFeatures = 1024;
constexpr int kDefaultHiddenFeatures = 4096;
constexpr int kDefaultHiddenLayers = 3;
constexpr int kDefaultOutputs = 1024;
constexpr std::uint64_t kDefaultSeed = 2024;
constexpr int kForwardsBetweenSynchronizations = 5;

// GEMM tiling: a 256-thread block computes a 64 x 64 output tile, 4 x 4
// outputs per thread, reading 16 32-bit words of K per tile.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileWords = 16;
constexpr int kThreads = 256;
constexpr int kThreadsPerRow = 16;
constexpr int kOutputsPerThread = 4;
constexpr int kDimensionMultiple = 64;
constexpr int kInt8Max = 127;

constexpr int kVerifyRows = 64;
constexpr int kVerifyColumns = 128;
constexpr int kVerifyDepth = 512;
constexpr double kVerifyToleranceFp32 = 1e-5;
constexpr double kVerifyToleranceFp16 = 5e-3;
constexpr double kVerifyToleranceInt8 = 1e-5;

enum class Precision { kFp32, kFp16, kInt8 };
enum class Mode { kMeasure, kVerify };

struct Config {
  int device = benchmarks::kDefaultDevice;
  double warmup_seconds = kDefaultWarmupSeconds;
  double measurement_seconds = kDefaultMeasurementSeconds;
  double cooldown_seconds = kDefaultCooldownSeconds;
  int batch = kDefaultBatch;
  int input_features = kDefaultInputFeatures;
  int hidden_features = kDefaultHiddenFeatures;
  int hidden_layers = kDefaultHiddenLayers;
  int outputs = kDefaultOutputs;
  std::uint64_t seed = kDefaultSeed;
  Precision precision = Precision::kInt8;
  Mode mode = Mode::kMeasure;
};

// Epilogue inputs of one layer. The scales are used by INT8 only.
struct LayerParams {
  const float* bias = nullptr;
  const float* weight_scale = nullptr;  // per output channel
  float input_scale = 1.0F;
  float inverse_output_scale = 1.0F;
  bool relu = true;
};

__global__ void benchmark_warmup_start_marker() {}
__global__ void benchmark_measurement_start_marker() {}
__global__ void benchmark_cooldown_start_marker() {}
__global__ void benchmark_complete_marker() {}

__device__ __forceinline__ __half2 bits_as_half2(unsigned int bits) {
  __half2 value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

struct Fp32Policy {
  using Value = float;
  using Acc = float;
  using Sum = float;
  static constexpr int kValuesPerWord = 1;
  static const char* name() { return "fp32"; }
  __device__ static Acc zero_acc() { return 0.0F; }
  __device__ static Sum zero_sum() { return 0.0F; }
  __device__ static Acc mac(Acc acc, unsigned int a, unsigned int w) {
    return fmaf(__uint_as_float(a), __uint_as_float(w), acc);
  }
  __device__ static void flush(Sum* sum, Acc* acc) {
    *sum += *acc;
    *acc = 0.0F;
  }
  __device__ static float dequantize(Sum sum, const LayerParams&, int) {
    return sum;
  }
  __device__ static Value quantize(float value, const LayerParams&) {
    return value;
  }
};

struct Fp16Policy {
  using Value = __half;
  using Acc = __half2;
  using Sum = float;
  static constexpr int kValuesPerWord = 2;
  static const char* name() { return "fp16"; }
  __device__ static Acc zero_acc() { return __float2half2_rn(0.0F); }
  __device__ static Sum zero_sum() { return 0.0F; }
  __device__ static Acc mac(Acc acc, unsigned int a, unsigned int w) {
    return __hfma2(bits_as_half2(a), bits_as_half2(w), acc);
  }
  // Each half lane holds at most 16 products before it is promoted.
  __device__ static void flush(Sum* sum, Acc* acc) {
    *sum += __low2float(*acc) + __high2float(*acc);
    *acc = zero_acc();
  }
  __device__ static float dequantize(Sum sum, const LayerParams&, int) {
    return sum;
  }
  __device__ static Value quantize(float value, const LayerParams&) {
    return __float2half_rn(value);
  }
};

struct Int8Policy {
  using Value = std::int8_t;
  using Acc = int;
  using Sum = int;
  static constexpr int kValuesPerWord = 4;
  static const char* name() { return "int8"; }
  __device__ static Acc zero_acc() { return 0; }
  __device__ static Sum zero_sum() { return 0; }
  __device__ static Acc mac(Acc acc, unsigned int a, unsigned int w) {
    return __dp4a(static_cast<int>(a), static_cast<int>(w), acc);
  }
  __device__ static void flush(Sum* sum, Acc* acc) {
    *sum += *acc;
    *acc = 0;
  }
  __device__ static float dequantize(Sum sum, const LayerParams& params,
                                     int column) {
    return static_cast<float>(sum) * params.input_scale *
           params.weight_scale[column];
  }
  __device__ static Value quantize(float value, const LayerParams& params) {
    const int level = __float2int_rn(value * params.inverse_output_scale);
    return static_cast<Value>(min(max(level, -kInt8Max), kInt8Max));
  }
};

// output[m][n] = act(dequantize(sum_k input[m][k] * weights[n][k]) + bias[n]).
// input and weights are row-major with K contiguous, packed in 32-bit words.
// kFinal writes float logits instead of the next layer's activations.
template <typename Policy, bool kFinal>
__global__ void __launch_bounds__(kThreads)
    qdnn_linear_kernel(const unsigned int* __restrict__ input,
                       const unsigned int* __restrict__ weights,
                       void* __restrict__ output, int columns, int k_words,
                       LayerParams params) {
  __shared__ unsigned int input_tile[kTileWords][kTileM + 1];
  __shared__ unsigned int weight_tile[kTileWords][kTileN + 1];
  const int tx = threadIdx.x % kThreadsPerRow;
  const int ty = threadIdx.x / kThreadsPerRow;
  const int row0 = blockIdx.y * kTileM;
  const int column0 = blockIdx.x * kTileN;

  typename Policy::Acc acc[kOutputsPerThread][kOutputsPerThread];
  typename Policy::Sum sum[kOutputsPerThread][kOutputsPerThread];
  for (int i = 0; i < kOutputsPerThread; ++i) {
    for (int j = 0; j < kOutputsPerThread; ++j) {
      acc[i][j] = Policy::zero_acc();
      sum[i][j] = Policy::zero_sum();
    }
  }

  for (int k0 = 0; k0 < k_words; k0 += kTileWords) {
    for (int element = threadIdx.x; element < kTileM * kTileWords;
         element += kThreads) {
      const int row = element / kTileWords;
      const int word = element % kTileWords;
      input_tile[word][row] =
          input[static_cast<std::size_t>(row0 + row) * k_words + k0 + word];
      weight_tile[word][row] =
          weights[static_cast<std::size_t>(column0 + row) * k_words + k0 +
                  word];
    }
    __syncthreads();
#pragma unroll
    for (int word = 0; word < kTileWords; ++word) {
      unsigned int a[kOutputsPerThread];
      unsigned int w[kOutputsPerThread];
#pragma unroll
      for (int i = 0; i < kOutputsPerThread; ++i) {
        a[i] = input_tile[word][ty * kOutputsPerThread + i];
        w[i] = weight_tile[word][tx * kOutputsPerThread + i];
      }
#pragma unroll
      for (int i = 0; i < kOutputsPerThread; ++i) {
#pragma unroll
        for (int j = 0; j < kOutputsPerThread; ++j) {
          acc[i][j] = Policy::mac(acc[i][j], a[i], w[j]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kOutputsPerThread; ++i) {
#pragma unroll
      for (int j = 0; j < kOutputsPerThread; ++j) {
        Policy::flush(&sum[i][j], &acc[i][j]);
      }
    }
  }

  for (int i = 0; i < kOutputsPerThread; ++i) {
    const int row = row0 + ty * kOutputsPerThread + i;
    for (int j = 0; j < kOutputsPerThread; ++j) {
      const int column = column0 + tx * kOutputsPerThread + j;
      float value =
          Policy::dequantize(sum[i][j], params, column) + params.bias[column];
      if (params.relu) value = fmaxf(value, 0.0F);
      const std::size_t index = static_cast<std::size_t>(row) * columns + column;
      if constexpr (kFinal) {
        static_cast<float*>(output)[index] = value;
      } else {
        static_cast<typename Policy::Value*>(output)[index] =
            Policy::quantize(value, params);
      }
    }
  }
}

template <typename T>
T* device_allocate(std::size_t count) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

template <typename T>
T* device_copy(const std::vector<T>& values) {
  T* pointer = device_allocate<T>(values.size());
  CUDA_CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return pointer;
}

// Host values of one precision packed into 32-bit words.
template <typename Value>
std::vector<unsigned int> pack_words(const std::vector<Value>& values) {
  std::vector<unsigned int> words(values.size() * sizeof(Value) /
                                  sizeof(unsigned int));
  std::memcpy(words.data(), values.data(), words.size() * sizeof(unsigned int));
  return words;
}

std::vector<__half> to_half(const std::vector<float>& values) {
  std::vector<__half> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[i] = __float2half_rn(values[i]);
  }
  return result;
}

float max_abs(const std::vector<float>& values) {
  float result = 0.0F;
  for (float value : values) result = std::max(result, std::abs(value));
  return result;
}

std::int8_t quantize_level(float value, float inverse_scale) {
  const long level = std::lround(value * inverse_scale);
  return static_cast<std::int8_t>(
      std::clamp<long>(level, -kInt8Max, kInt8Max));
}

// Symmetric per-row (output channel) INT8 weights.
void quantize_weights(const std::vector<float>& weights, int rows, int depth,
                      std::vector<std::int8_t>* levels,
                      std::vector<float>* scales) {
  levels->resize(weights.size());
  scales->resize(rows);
  for (int row = 0; row < rows; ++row) {
    const auto begin = weights.begin() + static_cast<std::size_t>(row) * depth;
    float largest = 0.0F;
    for (auto value = begin; value != begin + depth; ++value) {
      largest = std::max(largest, std::abs(*value));
    }
    const float scale = largest > 0.0F ? largest / kInt8Max : 1.0F;
    (*scales)[row] = scale;
    for (int k = 0; k < depth; ++k) {
      const std::size_t index = static_cast<std::size_t>(row) * depth + k;
      (*levels)[index] = quantize_level(weights[index], 1.0F / scale);
    }
  }
}

std::vector<std::int8_t> quantize_activations(const std::vector<float>& values,
                                              float scale) {
  std::vector<std::int8_t> levels(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    levels[i] = quantize_level(values[i], 1.0F / scale);
  }
  return levels;
}

// fp32 master weights and input batch, generated deterministically.
struct HostNetwork {
  int batch = 0;
  std::vector<int> widths;  // input, hidden..., outputs
  std::vector<std::vector<float>> weights;  // [layer] (out x in), row-major
  std::vector<std::vector<float>> biases;
  std::vector<float> input;

  int layers() const { return static_cast<int>(weights.size()); }
  double macs_per_sample() const {
    double macs = 0.0;
    for (int layer = 0; layer < layers(); ++layer) {
      macs += static_cast<double>(widths[layer]) * widths[layer + 1];
    }
    return macs;
  }
};

HostNetwork make_network(const Config& config) {
  HostNetwork network;
  network.batch = config.batch;
  network.widths.push_back(config.input_features);
  for (int layer = 0; layer < config.hidden_layers; ++layer) {
    network.widths.push_back(config.hidden_features);
  }
  network.widths.push_back(config.outputs);
  std::mt19937_64 generator(config.seed);
  std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
  for (std::size_t layer = 0; layer + 1 < network.widths.size(); ++layer) {
    const int depth = network.widths[layer];
    const int rows = network.widths[layer + 1];
    // He-uniform initialization keeps ReLU activations at a stable scale.
    const float limit = std::sqrt(6.0F / depth);
    std::vector<float> weights(static_cast<std::size_t>(rows) * depth);
    for (float& weight : weights) weight = limit * uniform(generator);
    std::vector<float> bias(rows);
    for (float& value : bias) value = 0.01F * uniform(generator);
    network.weights.push_back(std::move(weights));
    network.biases.push_back(std::move(bias));
  }
  network.input.resize(static_cast<std::size_t>(config.batch) *
                       config.input_features);
  for (float& value : network.input) value = uniform(generator);
  return network;
}

// Device copy of the network in one precision; runs batched forward passes.
template <typename Policy>
class DeviceModel {
 public:
  using Value = typename Policy::Value;

  // input_scales[l] is the INT8 scale of layer l's input (ignored otherwise).
  DeviceModel(const HostNetwork& network,
              const std::vector<float>& input_scales)
      : batch_(network.batch), widths_(network.widths) {
    const int widest = *std::max_element(widths_.begin(), widths_.end());
    const std::size_t activation_bytes =
        static_cast<std::size_t>(batch_) * widest * sizeof(Value);
    for (auto& buffer : activations_) {
      buffer = device_allocate<unsigned int>(activation_bytes /
                                             sizeof(unsigned int));
    }
    logits_ = device_allocate<float>(static_cast<std::size_t>(batch_) *
                                     widths_.back());
    for (int layer = 0; layer < network.layers(); ++layer) {
      Layer device_layer;
      device_layer.rows = widths_[layer + 1];
      device_layer.depth = widths_[layer];
      device_layer.bias = device_copy(network.biases[layer]);
      device_layer.params.bias = device_layer.bias;
      device_layer.params.relu = layer + 1 < network.layers();
      if constexpr (std::is_same_v<Policy, Int8Policy>) {
        std::vector<std::int8_t> levels;
        std::vector<float> scales;
        quantize_weights(network.weights[layer], device_layer.rows,
                         device_layer.depth, &levels, &scales);
        device_layer.weights = device_copy(pack_words(levels));
        device_layer.weight_scale = device_copy(scales);
        device_layer.params.weight_scale = device_layer.weight_scale;
        device_layer.params.input_scale = input_scales[layer];
        if (layer + 1 < network.layers()) {
          device_layer.params.inverse_output_scale =
              1.0F / input_scales[layer + 1];
        }
      } else if constexpr (std::is_same_v<Policy, Fp16Policy>) {
        device_layer.weights =
            device_copy(pack_words(to_half(network.weights[layer])));
      } else {
        device_layer.weights = device_copy(pack_words(network.weights[layer]));
      }
      layers_.push_back(device_layer);
    }
    std::vector<unsigned int> input_words;
    if constexpr (std::is_same_v<Policy, Int8Policy>) {
      input_words =
          pack_words(quantize_activations(network.input, input_scales[0]));
    } else if constexpr (std::is_same_v<Policy, Fp16Policy>) {
      input_words = pack_words(to_half(network.input));
    } else {
      input_words = pack_words(network.input);
    }
    input_ = device_copy(input_words);
  }

  ~DeviceModel() {
    for (Layer& layer : layers_) {
      cudaFree(layer.weights);
      cudaFree(layer.bias);
      cudaFree(layer.weight_scale);
    }
    for (unsigned int* buffer : activations_) cudaFree(buffer);
    cudaFree(input_);
    cudaFree(logits_);
  }
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  // One batched forward pass. With capture, the fp32 activations entering
  // every layer are copied back (used to calibrate the INT8 scales).
  void Forward(std::vector<std::vector<float>>* capture = nullptr) {
    const unsigned int* input = input_;
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
      const Layer& current = layers_[layer];
      const dim3 grid(current.rows / kTileN, batch_ / kTileM);
      const int k_words = current.depth / Policy::kValuesPerWord;
      const bool final_layer = layer + 1 == layers_.size();
      unsigned int* output = activations_[layer % 2];
      if (final_layer) {
        qdnn_linear_kernel<Policy, true><<<grid, kThreads>>>(
            input, current.weights, logits_, current.rows, k_words,
            current.params);
      } else {
        qdnn_linear_kernel<Policy, false><<<grid, kThreads>>>(
            input, current.weights, output, current.rows, k_words,
            current.params);
      }
      CUDA_CHECK(cudaGetLastError());
      if (capture != nullptr && !final_layer) {
        if constexpr (std::is_same_v<Policy, Fp32Policy>) {
          std::vector<float> values(static_cast<std::size_t>(batch_) *
                                    current.rows);
          CUDA_CHECK(cudaMemcpy(values.data(), output,
                                values.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
          capture->push_back(std::move(values));
        }
      }
      input = output;
    }
  }

  std::vector<float> Logits() const {
    std::vector<float> values(static_cast<std::size_t>(batch_) *
                              widths_.back());
    CUDA_CHECK(cudaMemcpy(values.data(), logits_, values.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return values;
  }

 private:
  struct Layer {
    int rows = 0;
    int depth = 0;
    unsigned int* weights = nullptr;
    float* bias = nullptr;
    float* weight_scale = nullptr;
    LayerParams params;
  };

  int batch_ = 0;
  std::vector<int> widths_;
  std::vector<Layer> layers_;
  unsigned int* activations_[2] = {nullptr, nullptr};
  unsigned int* input_ = nullptr;
  float* logits_ = nullptr;
};

// INT8 input scales: max |activation| / 127 of each layer's fp32 input.
std::vector<float> calibrate(const HostNetwork& network,
                             std::vector<float>* fp32_logits) {
  DeviceModel<Fp32Policy> reference(network, {});
  std::vector<std::vector<float>> hidden;
  reference.Forward(&hidden);
  *fp32_logits = reference.Logits();
  std::vector<float> scales;
  scales.push_back(max_abs(network.input) / kInt8Max);
  for (const auto& values : hidden) {
    const float largest = max_abs(values);
    scales.push_back(largest > 0.0F ? largest / kInt8Max : 1.0F);
  }
  return scales;
}

struct Accuracy {
  double relative_l2_error = 0.0;
  double top1_agreement_pct = 0.0;
};

Accuracy compare_logits(const std::vector<float>& reference,
                        const std::vector<float>& logits, int batch) {
  const std::size_t classes = reference.size() / batch;
  double error_sq = 0.0;
  double reference_sq = 0.0;
  int agreements = 0;
  for (int sample = 0; sample < batch; ++sample) {
    const auto first = reference.begin() + sample * classes;
    const auto other = logits.begin() + sample * classes;
    agreements += std::max_element(first, first + classes) - first ==
                  std::max_element(other, other + classes) - other;
    for (std::size_t c = 0; c < classes; ++c) {
      const double difference = other[c] - first[c];
      error_sq += difference * difference;
      reference_sq += static_cast<double>(first[c]) * first[c];
    }
  }
  return {std::sqrt(error_sq / reference_sq), 100.0 * agreements / batch};
}

// One final layer (no ReLU) of each precision against a host reference built
// from the same rounded or quantized inputs; INT8 sums are exact on the host.
template <typename Policy>
double verify_layer(std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
  std::vector<float> input(static_cast<std::size_t>(kVerifyRows) * kVerifyDepth);
  std::vector<float> weights(static_cast<std::size_t>(kVerifyColumns) *
                             kVerifyDepth);
  std::vector<float> bias(kVerifyColumns);
  for (float& value : input) value = uniform(generator);
  for (float& value : weights) value = uniform(generator) / 16.0F;
  for (float& value : bias) value = uniform(generator);

  std::vector<unsigned int> input_words;
  std::vector<unsigned int> weight_words;
  std::vector<double> input_used(input.begin(), input.end());
  std::vector<double> weight_used(weights.begin(), weights.end());
  std::vector<float> weight_scales(kVerifyColumns, 1.0F);
  float input_scale = 1.0F;
  if constexpr (std::is_same_v<Policy, Int8Policy>) {
    input_scale = max_abs(input) / kInt8Max;
    const auto input_levels = quantize_activations(input, input_scale);
    std::vector<std::int8_t> weight_levels;
    quantize_weights(weights, kVerifyColumns, kVerifyDepth, &weight_levels,
                     &weight_scales);
    input_words = pack_words(input_levels);
    weight_words = pack_words(weight_levels);
    input_used.assign(input_levels.begin(), input_levels.end());
    weight_used.assign(weight_levels.begin(), weight_levels.end());
  } else if constexpr (std::is_same_v<Policy, Fp16Policy>) {
    const auto input_half = to_half(input);
    const auto weight_half = to_half(weights);
    input_words = pack_words(input_half);
    weight_words = pack_words(weight_half);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input_used[i] = __half2float(input_half[i]);
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weight_used[i] = __half2float(weight_half[i]);
    }
  } else {
    input_words = pack_words(input);
    weight_words = pack_words(weights);
  }

  unsigned int* device_input = device_copy(input_words);
  unsigned int* device_weights = device_copy(weight_words);
  float* device_bias = device_copy(bias);
  float* device_scales = device_copy(weight_scales);
  float* device_output =
      device_allocate<float>(static_cast<std::size_t>(kVerifyRows) *
                             kVerifyColumns);
  LayerParams params;
  params.bias = device_bias;
  params.weight_scale = device_scales;
  params.input_scale = input_scale;
  params.relu = false;
  qdnn_linear_kernel<Policy, true>
      <<<dim3(kVerifyColumns / kTileN, kVerifyRows / kTileM), kThreads>>>(
          device_input, device_weights, device_output, kVerifyColumns,
          kVerifyDepth / Policy::kValuesPerWord, params);
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> output(static_cast<std::size_t>(kVerifyRows) *
                            kVerifyColumns);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (void* pointer : {static_cast<void*>(device_input),
                        static_cast<void*>(device_weights),
                        static_cast<void*>(device_bias),
                        static_cast<void*>(device_scales),
                        static_cast<void*>(device_output)}) {
    cudaFree(pointer);
  }

  double error_sq = 0.0;
  double reference_sq = 0.0;
  for (int row = 0; row < kVerifyRows; ++row) {
    for (int column = 0; column < kVerifyColumns; ++column) {
      double sum = 0.0;
      for (int k = 0; k < kVerifyDepth; ++k) {
        sum += input_used[static_cast<std::size_t>(row) * kVerifyDepth + k] *
               weight_used[static_cast<std::size_t>(column) * kVerifyDepth + k];
      }
      const double expected =
          sum * input_scale * weight_scales[column] + bias[column];
      const double difference =
          output[static_cast<std::size_t>(row) * kVerifyColumns + column] -
          expected;
      error_sq += difference * difference;
      reference_sq += expected * expected;
    }
  }
  return std::sqrt(error_sq / reference_sq);
}

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Quantized deep neural network (MLP) inference: fp32, fp16 or int8.\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --precision <name>        fp32, fp16 or int8 (default: int8)\n"
      << "  --mode <name>             measure or verify (default: measure)\n"
      << "  --batch <N>               Samples per forward pass (default: 256)\n"
      << "  --input-features <N>      Input width (default: 1024)\n"
      << "  --hidden-features <N>     Hidden width (default: 4096)\n"
      << "  --hidden-layers <N>       Hidden layers (default: 3)\n"
      << "  --outputs <N>             Output width (default: 1024)\n"
      << "                            Widths and batch must be multiples of 64\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 10)\n"
      << "  --duration <seconds>      Measurement phase (default: 40)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 10)\n"
      << "  --seed <N>                Weight and input seed (default: 2024)\n"
      << "  --help                    Show this help\n";
}

Config parse_config(const benchmarks::ArgumentParser& arguments) {
  arguments.validate_options(
      {"--device", "--precision", "--mode", "--batch", "--input-features",
       "--hidden-features", "--hidden-layers", "--outputs",
       "--warmup-seconds", "--duration", "--cooldown-seconds", "--seed"});
  Config config;
  config.device = arguments.integer_value("--device", config.device);
  config.batch = arguments.integer_value("--batch", config.batch);
  config.input_features =
      arguments.integer_value("--input-features", config.input_features);
  config.hidden_features =
      arguments.integer_value("--hidden-features", config.hidden_features);
  config.hidden_layers =
      arguments.integer_value("--hidden-layers", config.hidden_layers);
  config.outputs = arguments.integer_value("--outputs", config.outputs);
  config.warmup_seconds =
      arguments.double_value("--warmup-seconds", config.warmup_seconds);
  config.measurement_seconds =
      arguments.double_value("--duration", config.measurement_seconds);
  config.cooldown_seconds =
      arguments.double_value("--cooldown-seconds", config.cooldown_seconds);
  config.seed = arguments.unsigned_value("--seed", config.seed);

  const std::string precision = arguments.string_value("--precision", "int8");
  if (precision == "fp32") {
    config.precision = Precision::kFp32;
  } else if (precision == "fp16") {
    config.precision = Precision::kFp16;
  } else if (precision == "int8") {
    config.precision = Precision::kInt8;
  } else {
    throw std::invalid_argument("--precision must be fp32, fp16 or int8");
  }
  const std::string mode = arguments.string_value("--mode", "measure");
  if (mode == "measure") {
    config.mode = Mode::kMeasure;
  } else if (mode == "verify") {
    config.mode = Mode::kVerify;
  } else {
    throw std::invalid_argument("--mode must be measure or verify");
  }

  if (config.device < 0) {
    throw std::invalid_argument("--device must be non-negative");
  }
  for (int value : {config.batch, config.input_features,
                    config.hidden_features, config.outputs}) {
    if (value <= 0 || value % kDimensionMultiple != 0) {
      throw std::invalid_argument(
          "--batch and every layer width must be positive multiples of 64");
    }
  }
  if (config.hidden_layers < 0) {
    throw std::invalid_argument("--hidden-layers must be non-negative");
  }
  if (!(config.warmup_seconds >= 0.0) || !(config.measurement_seconds > 0.0) ||
      !(config.cooldown_seconds >= 0.0)) {
    throw std::invalid_argument(
        "phase durations must be finite; measurement must be positive");
  }
  return config;
}

const char* precision_name(Precision precision) {
  switch (precision) {
    case Precision::kFp32:
      return Fp32Policy::name();
    case Precision::kFp16:
      return Fp16Policy::name();
    case Precision::kInt8:
      return Int8Policy::name();
  }
  return "unknown";
}

void print_config(const Config& config, const cudaDeviceProp& properties) {
  std::cout << std::setprecision(12) << "BENCHMARK=quantized_dnn\n"
            << "MODEL=mlp\n"
            << "PRECISION=" << precision_name(config.precision) << '\n'
            << "MODE=" << (config.mode == Mode::kMeasure ? "measure" : "verify")
            << '\n'
            << "DEVICE=" << config.device << '\n'
            << "DEVICE_NAME=" << properties.name << '\n'
            << "COMPUTE_CAPABILITY=" << properties.major << '.'
            << properties.minor << '\n'
            << "BATCH_SIZE=" << config.batch << '\n'
            << "INPUT_FEATURES=" << config.input_features << '\n'
            << "HIDDEN_FEATURES=" << config.hidden_features << '\n'
            << "HIDDEN_LAYERS=" << config.hidden_layers << '\n'
            << "OUTPUTS=" << config.outputs << '\n'
            << "WARMUP_SECONDS=" << config.warmup_seconds << '\n'
            << "MEASUREMENT_SECONDS=" << config.measurement_seconds << '\n'
            << "COOLDOWN_SECONDS=" << config.cooldown_seconds << '\n';
}

enum class PhaseMarker { kWarmup, kMeasurement, kCooldown, kComplete };

void emit_phase(const char* phase, PhaseMarker marker) {
  switch (marker) {
    case PhaseMarker::kWarmup:
      benchmark_warmup_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kMeasurement:
      benchmark_measurement_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kCooldown:
      benchmark_cooldown_start_marker<<<1, 1>>>();
      break;
    case PhaseMarker::kComplete:
      benchmark_complete_marker<<<1, 1>>>();
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "PHASE=" << phase << '\n' << std::flush;
}

struct PhaseStats {
  double elapsed_seconds = 0.0;
  std::uint64_t forwards = 0;
};

template <typename Policy>
PhaseStats run_phase(DeviceModel<Policy>* model, double seconds) {
  PhaseStats stats;
  const auto start = Clock::now();
  const auto deadline =
      start + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(seconds));
  while (Clock::now() < deadline) {
    for (int i = 0; i < kForwardsBetweenSynchronizations; ++i) {
      model->Forward();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.forwards += kForwardsBetweenSynchronizations;
  }
  stats.elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  return stats;
}

template <typename Policy>
int measure(const Config& config) {
  const HostNetwork network = make_network(config);
  std::vector<float> fp32_logits;
  const std::vector<float> scales = calibrate(network, &fp32_logits);
  DeviceModel<Policy> model(network, scales);
  model.Forward();
  const Accuracy accuracy =
      compare_logits(fp32_logits, model.Logits(), config.batch);

  emit_phase("warmup", PhaseMarker::kWarmup);
  const PhaseStats warmup = run_phase(&model, config.warmup_seconds);
  emit_phase("measurement", PhaseMarker::kMeasurement);
  const PhaseStats measurement = run_phase(&model, config.measurement_seconds);
  emit_phase("cooldown", PhaseMarker::kCooldown);
  const auto cooldown_start = Clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.cooldown_seconds));
  const double cooldown_elapsed =
      std::chrono::duration<double>(Clock::now() - cooldown_start).count();
  emit_phase("complete", PhaseMarker::kComplete);

  const double samples =
      static_cast<double>(measurement.forwards) * config.batch;
  const double macs = samples * network.macs_per_sample();
  std::cout << std::fixed << std::setprecision(6)
            << "LAYERS=" << network.layers() << '\n'
            << "MACS_PER_SAMPLE=" << std::setprecision(0)
            << network.macs_per_sample() << '\n'
            << std::setprecision(6)
            << "WARMUP_ELAPSED_SECONDS=" << warmup.elapsed_seconds << '\n'
            << "WARMUP_FORWARDS=" << warmup.forwards << '\n'
            << "MEASUREMENT_ELAPSED_SECONDS=" << measurement.elapsed_seconds
            << '\n'
            << "MEASUREMENT_FORWARDS=" << measurement.forwards << '\n'
            << "MEASUREMENT_SAMPLES=" << std::setprecision(0) << samples << '\n'
            << std::setprecision(6) << "SAMPLES_PER_SECOND="
            << samples / measurement.elapsed_seconds << '\n'
            << "MS_PER_BATCH="
            << 1000.0 * measurement.elapsed_seconds / measurement.forwards
            << '\n'
            << "TERA_MACS_PER_SECOND="
            << macs / measurement.elapsed_seconds / 1e12 << '\n'
            << std::scientific
            << "REL_L2_ERROR_VS_FP32=" << accuracy.relative_l2_error << '\n'
            << std::fixed << std::setprecision(3)
            << "TOP1_AGREEMENT_VS_FP32_PCT=" << accuracy.top1_agreement_pct
            << '\n'
            << std::setprecision(6)
            << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed << '\n';
  benchmarks::print_execution_summary(measurement.elapsed_seconds,
                                      measurement.forwards);
  return 0;
}

int verify(const Config& config) {
  const double errors[3] = {verify_layer<Fp32Policy>(config.seed),
                            verify_layer<Fp16Policy>(config.seed),
                            verify_layer<Int8Policy>(config.seed)};
  const double tolerances[3] = {kVerifyToleranceFp32, kVerifyToleranceFp16,
                                kVerifyToleranceInt8};
  const char* names[3] = {"FP32", "FP16", "INT8"};
  bool passed = true;
  std::cout << std::scientific << std::setprecision(6);
  for (int i = 0; i < 3; ++i) {
    const bool ok = std::isfinite(errors[i]) && errors[i] <= tolerances[i];
    passed = passed && ok;
    std::cout << "VERIFY_" << names[i] << "_REL_L2_ERROR=" << errors[i] << '\n'
              << "VERIFY_" << names[i] << "_TOLERANCE=" << tolerances[i] << '\n'
              << "VERIFY_" << names[i] << '=' << (ok ? "OK" : "FAILED") << '\n';
  }
  std::cout << "VERIFY=" << (passed ? "OK" : "FAILED") << '\n';
  if (!passed) {
    std::cout << "STATUS=FAILED\n";
    return 1;
  }
  benchmarks::print_execution_summary(0.0, 3);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const benchmarks::ArgumentParser arguments(argc, argv);
    if (arguments.help_requested()) {
      print_help(argv[0]);
      return 0;
    }
    const Config config = parse_config(arguments);
    const cudaDeviceProp properties =
        benchmarks::initialize_cuda_device(config.device);
    print_config(config, properties);
    if (config.mode == Mode::kVerify) return verify(config);
    switch (config.precision) {
      case Precision::kFp32:
        return measure<Fp32Policy>(config);
      case Precision::kFp16:
        return measure<Fp16Policy>(config);
      case Precision::kInt8:
        return measure<Int8Policy>(config);
    }
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
