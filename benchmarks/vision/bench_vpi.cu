// NVIDIA VPI (Vision Programming Interface) computer-vision benchmark.
//
// Streams synthetic frames through one VPI algorithm or a small pipeline on a
// selectable VPI backend:
//
//   gaussian  7x7 Gaussian filter, U8, 3840x2160 (stencil, memory bound)
//   rescale   bilinear downscale by two, U8, 3840x2160 -> 1920x1080
//   harris    Harris corner detector, U8, 1920x1080 (several kernels and a
//             keypoint compaction)
//   stereo    semi-global matching stereo disparity, U8, 1920x1080, 64
//             disparities (compute bound)
//   pipeline  BGR8 3840x2160 -> grey U8 -> 5x5 Gaussian -> downscale by two
//             -> Harris corners, the front end of a feature tracker
//
// The CUDA backend runs on every NVIDIA GPU VPI supports, including the
// Tesla V100 (sm_70) and Jetson Thor. On Jetson the PVA, VIC and OFA backends
// are separate accelerators: their work is invisible to CUPTI, so a profiler
// built on CUPTI attributes none of it to kernels.
//
// Like bench_power_cycle, the run has warm-up, measurement and cooldown phases
// delimited by empty marker kernels launched with the CUDA runtime. The input
// frames are synthetic and deterministic: rectangles on a textured background
// with known corners, and a random-texture stereo pair whose right view is
// the left view shifted by a known disparity. --mode verify checks each
// algorithm on the CUDA backend against the CPU backend or against that
// ground truth.

#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/Version.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianFilter.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kDefaultWarmupSeconds = 10.0;
constexpr double kDefaultMeasurementSeconds = 40.0;
constexpr double kDefaultCooldownSeconds = 10.0;
constexpr std::uint64_t kDefaultSeed = 2024;
constexpr int kFramesBetweenSynchronizations = 8;

constexpr int kLargeWidth = 3840;
constexpr int kLargeHeight = 2160;
constexpr int kSmallWidth = 1920;
constexpr int kSmallHeight = 1080;

constexpr int kGaussianKernelSize = 7;
constexpr float kGaussianSigma = 1.7F;
constexpr int kPipelineKernelSize = 5;
constexpr float kPipelineSigma = 1.0F;
constexpr int kKeypointCapacity = 16384;
// Harris strength thresholds at which each scene's square corners are found:
// VPI's default (20) misses a third of them on the harris scene and all of
// them once the pipeline has blurred and downscaled the frame.
constexpr float kHarrisStrength = 10.0F;
constexpr float kPipelineHarrisStrength = 5.0F;
constexpr int kStereoMaxDisparity = 64;
constexpr int kStereoTrueDisparity = 20;
constexpr int kStereoFixedPointScale = 32;  // Q10.5 disparity output

// Scene: bright squares on a darker textured background.
constexpr int kSquareSize = 48;
constexpr int kSquareSpacing = 96;
constexpr int kBackgroundLevel = 40;
constexpr int kSquareLevel = 210;
constexpr int kSceneNoise = 24;
constexpr int kVerifyHarrisNoise = 4;

// Verification tolerances.
// CPU and CUDA accumulate the filter taps with different arithmetic, so a
// 49-tap Gaussian can round two steps apart on a few pixels; a real defect
// shows up as larger or widespread differences.
constexpr int kVerifyMaxPixelDifference = 2;
constexpr double kVerifyMaxAboveOneFraction = 1e-4;
constexpr double kVerifyCornerRadius = 3.0;
constexpr double kVerifyMinCornerFraction = 0.9;
constexpr double kVerifyMinDisparityFraction = 0.9;
constexpr int kStereoBorder = 8;

enum class Algorithm { kGaussian, kRescale, kHarris, kStereo, kPipeline };
enum class Mode { kMeasure, kVerify };

struct Config {
  int device = benchmarks::kDefaultDevice;
  double warmup_seconds = kDefaultWarmupSeconds;
  double measurement_seconds = kDefaultMeasurementSeconds;
  double cooldown_seconds = kDefaultCooldownSeconds;
  std::uint64_t seed = kDefaultSeed;
  Algorithm algorithm = Algorithm::kPipeline;
  std::string backend_name = "cuda";
  std::uint64_t backend = VPI_BACKEND_CUDA;
  int width = 0;   // 0: the algorithm's default
  int height = 0;
  float harris_strength = -1.0F;  // < 0: the workload's default
  Mode mode = Mode::kMeasure;
};

__global__ void benchmark_warmup_start_marker() {}
__global__ void benchmark_measurement_start_marker() {}
__global__ void benchmark_cooldown_start_marker() {}
__global__ void benchmark_complete_marker() {}

void check_vpi(VPIStatus status, const char* expression) {
  if (status == VPI_SUCCESS) return;
  char message[VPI_MAX_STATUS_MESSAGE_LENGTH];
  vpiGetLastStatusMessage(message, sizeof(message));
  throw std::runtime_error(std::string(expression) + " failed: " +
                           vpiStatusGetName(status) + ": " + message);
}

#define VPI_CHECK(expression) check_vpi((expression), #expression)

// Owns one VPI handle and destroys it with the matching VPI function.
template <typename Handle, void (*Destroy)(Handle)>
class VpiHandle {
 public:
  VpiHandle() = default;
  ~VpiHandle() { reset(); }
  VpiHandle(const VpiHandle&) = delete;
  VpiHandle& operator=(const VpiHandle&) = delete;
  VpiHandle(VpiHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  VpiHandle& operator=(VpiHandle&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  Handle get() const { return handle_; }
  Handle* out() {
    reset();
    return &handle_;
  }

 private:
  void reset() {
    if (handle_ != nullptr) Destroy(handle_);
    handle_ = nullptr;
  }

  Handle handle_ = nullptr;
};

using Image = VpiHandle<VPIImage, vpiImageDestroy>;
using Array = VpiHandle<VPIArray, vpiArrayDestroy>;
using Payload = VpiHandle<VPIPayload, vpiPayloadDestroy>;
using Stream = VpiHandle<VPIStream, vpiStreamDestroy>;

// Host copy of a single-plane pitch-linear image.
template <typename T>
struct HostImage {
  int width = 0;
  int height = 0;
  int channels = 1;
  std::vector<T> pixels;

  HostImage() = default;
  HostImage(int image_width, int image_height, int image_channels)
      : width(image_width),
        height(image_height),
        channels(image_channels),
        pixels(static_cast<std::size_t>(image_width) * image_height *
               image_channels) {}

  T& at(int x, int y, int channel = 0) {
    return pixels[(static_cast<std::size_t>(y) * width + x) * channels +
                  channel];
  }
  const T& at(int x, int y, int channel = 0) const {
    return pixels[(static_cast<std::size_t>(y) * width + x) * channels +
                  channel];
  }
};

// Keeps an image locked for host access for the lifetime of the object.
class HostLock {
 public:
  HostLock(VPIImage image, VPILockMode mode) : image_(image) {
    VPI_CHECK(vpiImageLockData(image, mode, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                               &data_));
    if (data_.buffer.pitch.numPlanes != 1) {
      vpiImageUnlock(image_);
      throw std::runtime_error("only single-plane images are supported");
    }
  }
  ~HostLock() { vpiImageUnlock(image_); }
  HostLock(const HostLock&) = delete;
  HostLock& operator=(const HostLock&) = delete;

  VPIByte* row(int y) const {
    const VPIImagePlanePitchLinear& plane = data_.buffer.pitch.planes[0];
    return plane.pBase + plane.offsetBytes +
           static_cast<std::ptrdiff_t>(y) * plane.pitchBytes;
  }

 private:
  VPIImage image_;
  VPIImageData data_{};
};

template <typename T>
void upload(VPIImage image, const HostImage<T>& host) {
  const HostLock lock(image, VPI_LOCK_WRITE);
  const std::size_t row_bytes =
      static_cast<std::size_t>(host.width) * host.channels * sizeof(T);
  for (int y = 0; y < host.height; ++y) {
    std::memcpy(lock.row(y), &host.at(0, y), row_bytes);
  }
}

template <typename T>
HostImage<T> download(VPIImage image, int channels = 1) {
  int width = 0;
  int height = 0;
  VPI_CHECK(vpiImageGetSize(image, &width, &height));
  HostImage<T> host(width, height, channels);
  const HostLock lock(image, VPI_LOCK_READ);
  const std::size_t row_bytes =
      static_cast<std::size_t>(width) * channels * sizeof(T);
  for (int y = 0; y < height; ++y) {
    std::memcpy(&host.at(0, y), lock.row(y), row_bytes);
  }
  return host;
}

std::vector<VPIKeypointF32> download_keypoints(VPIArray array) {
  VPIArrayData data{};
  VPI_CHECK(vpiArrayLockData(array, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                             &data));
  const auto* keypoints =
      static_cast<const VPIKeypointF32*>(data.buffer.aos.data);
  std::vector<VPIKeypointF32> result(keypoints,
                                     keypoints + *data.buffer.aos.sizePointer);
  vpiArrayUnlock(array);
  return result;
}

// Deterministic per-pixel noise in [0, 256), independent of evaluation order.
std::uint32_t hash_noise(std::uint64_t seed, int x, int y, int channel) {
  std::uint64_t value = seed ^ (static_cast<std::uint64_t>(x) << 1) ^
                        (static_cast<std::uint64_t>(y) << 21) ^
                        (static_cast<std::uint64_t>(channel) << 42);
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  value ^= value >> 31;
  return static_cast<std::uint32_t>(value & 0xFFU);
}

bool inside_square(int x, int y) {
  const int offset = kSquareSpacing / 2;
  if (x < offset || y < offset) return false;
  return (x - offset) % kSquareSpacing < kSquareSize &&
         (y - offset) % kSquareSpacing < kSquareSize;
}

// Squares on a textured background; `noise` is the texture amplitude.
HostImage<std::uint8_t> make_scene(int width, int height, int channels,
                                   int noise, std::uint64_t seed) {
  HostImage<std::uint8_t> image(width, height, channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int base = inside_square(x, y) ? kSquareLevel : kBackgroundLevel;
      for (int c = 0; c < channels; ++c) {
        const int texture =
            static_cast<int>(hash_noise(seed, x, y, c) * noise / 256);
        image.at(x, y, c) =
            static_cast<std::uint8_t>(std::min(255, base + texture));
      }
    }
  }
  return image;
}

// Corners of every square that lies fully inside the image.
std::vector<VPIKeypointF32> scene_corners(int width, int height) {
  std::vector<VPIKeypointF32> corners;
  const int offset = kSquareSpacing / 2;
  for (int y0 = offset; y0 + kSquareSize <= height; y0 += kSquareSpacing) {
    for (int x0 = offset; x0 + kSquareSize <= width; x0 += kSquareSpacing) {
      const float x1 = static_cast<float>(x0 + kSquareSize - 1);
      const float y1 = static_cast<float>(y0 + kSquareSize - 1);
      for (const VPIKeypointF32& corner :
           {VPIKeypointF32{static_cast<float>(x0), static_cast<float>(y0)},
            VPIKeypointF32{x1, static_cast<float>(y0)},
            VPIKeypointF32{static_cast<float>(x0), y1},
            VPIKeypointF32{x1, y1}}) {
        corners.push_back(corner);
      }
    }
  }
  return corners;
}

// A random-texture stereo pair: right(x, y) = left(x + disparity, y), so every
// left pixel with x >= disparity has the true disparity `disparity`.
void make_stereo_pair(int width, int height, int disparity, std::uint64_t seed,
                      HostImage<std::uint8_t>* left,
                      HostImage<std::uint8_t>* right) {
  *left = HostImage<std::uint8_t>(width, height, 1);
  *right = HostImage<std::uint8_t>(width, height, 1);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      left->at(x, y) = static_cast<std::uint8_t>(hash_noise(seed, x, y, 0));
      right->at(x, y) = static_cast<std::uint8_t>(
          x + disparity < width ? hash_noise(seed, x + disparity, y, 0)
                                : hash_noise(seed ^ 1U, x, y, 0));
    }
  }
}

std::uint64_t image_flags(std::uint64_t backend) {
  return backend | VPI_BACKEND_CPU;  // the CPU flag allows host upload
}

// One algorithm or pipeline with its images, arrays and payloads.
class Workload {
 public:
  virtual ~Workload() = default;
  virtual void Submit(VPIStream stream) = 0;
  virtual double input_pixels() const = 0;
  virtual const char* name() const = 0;
  // Per-frame result printed after the measurement (KEY=value lines).
  virtual void PrintResult() const {}
};

class GaussianWorkload : public Workload {
 public:
  GaussianWorkload(std::uint64_t backend, int width, int height,
                   std::uint64_t seed)
      : backend_(backend), width_(width), height_(height) {
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8,
                             image_flags(backend), input_.out()));
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8,
                             image_flags(backend), output_.out()));
    upload(input_.get(), make_scene(width, height, 1, kSceneNoise, seed));
  }

  void Submit(VPIStream stream) override {
    VPI_CHECK(vpiSubmitGaussianFilter(
        stream, backend_, input_.get(), output_.get(), kGaussianKernelSize,
        kGaussianKernelSize, kGaussianSigma, kGaussianSigma,
        VPI_BORDER_CLAMP));
  }
  double input_pixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* name() const override { return "gaussian"; }
  VPIImage output() const { return output_.get(); }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  Image input_;
  Image output_;
};

class RescaleWorkload : public Workload {
 public:
  RescaleWorkload(std::uint64_t backend, int width, int height,
                  std::uint64_t seed)
      : backend_(backend), width_(width), height_(height) {
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8,
                             image_flags(backend), input_.out()));
    VPI_CHECK(vpiImageCreate(width / 2, height / 2, VPI_IMAGE_FORMAT_U8,
                             image_flags(backend), output_.out()));
    upload(input_.get(), make_scene(width, height, 1, kSceneNoise, seed));
  }

  void Submit(VPIStream stream) override {
    VPI_CHECK(vpiSubmitRescale(stream, backend_, input_.get(), output_.get(),
                               VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  }
  double input_pixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* name() const override { return "rescale"; }
  VPIImage output() const { return output_.get(); }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  Image input_;
  Image output_;
};

class HarrisWorkload : public Workload {
 public:
  HarrisWorkload(std::uint64_t backend, int width, int height, int noise,
                 float strength, std::uint64_t seed)
      : backend_(backend), width_(width), height_(height) {
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8,
                             image_flags(backend), input_.out()));
    VPI_CHECK(vpiArrayCreate(kKeypointCapacity, VPI_ARRAY_TYPE_KEYPOINT_F32,
                             image_flags(backend), keypoints_.out()));
    VPI_CHECK(vpiArrayCreate(kKeypointCapacity, VPI_ARRAY_TYPE_U32,
                             image_flags(backend), scores_.out()));
    VPI_CHECK(vpiCreateHarrisCornerDetector(backend, width, height,
                                            payload_.out()));
    VPI_CHECK(vpiInitHarrisCornerDetectorParams(&params_));
    if (strength >= 0.0F) params_.strengthThresh = strength;
    upload(input_.get(), make_scene(width, height, 1, noise, seed));
  }

  void Submit(VPIStream stream) override {
    VPI_CHECK(vpiSubmitHarrisCornerDetector(stream, backend_, payload_.get(),
                                            input_.get(), keypoints_.get(),
                                            scores_.get(), &params_));
  }
  double input_pixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* name() const override { return "harris"; }
  void PrintResult() const override {
    std::cout << "KEYPOINTS_LAST_FRAME=" << keypoints().size() << '\n';
  }
  std::vector<VPIKeypointF32> keypoints() const {
    return download_keypoints(keypoints_.get());
  }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  Image input_;
  Array keypoints_;
  Array scores_;
  Payload payload_;
  VPIHarrisCornerDetectorParams params_{};
};

class StereoWorkload : public Workload {
 public:
  StereoWorkload(std::uint64_t backend, int width, int height,
                 std::uint64_t seed)
      : backend_(backend), width_(width), height_(height) {
    for (Image* image : {&left_, &right_}) {
      VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8,
                               image_flags(backend), image->out()));
    }
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16,
                             image_flags(backend), disparity_.out()));
    VPIStereoDisparityEstimatorCreationParams creation{};
    VPI_CHECK(vpiInitStereoDisparityEstimatorCreationParams(&creation));
    creation.maxDisparity = kStereoMaxDisparity;
    VPI_CHECK(vpiCreateStereoDisparityEstimator(backend, width, height,
                                                VPI_IMAGE_FORMAT_U8, &creation,
                                                payload_.out()));
    VPI_CHECK(vpiInitStereoDisparityEstimatorParams(&params_));
    params_.maxDisparity = kStereoMaxDisparity;
    HostImage<std::uint8_t> left;
    HostImage<std::uint8_t> right;
    make_stereo_pair(width, height, kStereoTrueDisparity, seed, &left, &right);
    upload(left_.get(), left);
    upload(right_.get(), right);
  }

  void Submit(VPIStream stream) override {
    VPI_CHECK(vpiSubmitStereoDisparityEstimator(
        stream, backend_, payload_.get(), left_.get(), right_.get(),
        disparity_.get(), nullptr, &params_));
  }
  double input_pixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* name() const override { return "stereo"; }
  void PrintResult() const override {
    std::cout << std::fixed << std::setprecision(3)
              << "DISPARITY_WITHIN_1PX_PCT=" << 100.0 * accuracy() << '\n'
              << "TRUE_DISPARITY=" << kStereoTrueDisparity << '\n';
  }

  // Fraction of interior pixels whose disparity is within one pixel of the
  // true disparity. Pixels with x < max disparity have no full search range.
  double accuracy() const {
    const HostImage<std::int16_t> disparity =
        download<std::int16_t>(disparity_.get());
    std::uint64_t good = 0;
    std::uint64_t total = 0;
    for (int y = kStereoBorder; y < height_ - kStereoBorder; ++y) {
      for (int x = kStereoMaxDisparity;
           x < width_ - kStereoTrueDisparity - kStereoBorder; ++x) {
        const double value =
            static_cast<double>(disparity.at(x, y)) / kStereoFixedPointScale;
        good += std::abs(value - kStereoTrueDisparity) <= 1.0 ? 1 : 0;
        ++total;
      }
    }
    return total == 0 ? 0.0 : static_cast<double>(good) / total;
  }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  Image left_;
  Image right_;
  Image disparity_;
  Payload payload_;
  VPIStereoDisparityEstimatorParams params_{};
};

class PipelineWorkload : public Workload {
 public:
  PipelineWorkload(std::uint64_t backend, int width, int height,
                   float strength, std::uint64_t seed)
      : backend_(backend), width_(width), height_(height) {
    const std::uint64_t flags = image_flags(backend);
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_BGR8, flags,
                             colour_.out()));
    VPI_CHECK(
        vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, grey_.out()));
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags,
                             blurred_.out()));
    VPI_CHECK(vpiImageCreate(width / 2, height / 2, VPI_IMAGE_FORMAT_U8, flags,
                             small_.out()));
    VPI_CHECK(vpiArrayCreate(kKeypointCapacity, VPI_ARRAY_TYPE_KEYPOINT_F32,
                             flags, keypoints_.out()));
    VPI_CHECK(
        vpiArrayCreate(kKeypointCapacity, VPI_ARRAY_TYPE_U32, flags, scores_.out()));
    VPI_CHECK(vpiCreateHarrisCornerDetector(backend, width / 2, height / 2,
                                            harris_.out()));
    VPI_CHECK(vpiInitHarrisCornerDetectorParams(&params_));
    if (strength >= 0.0F) params_.strengthThresh = strength;
    upload(colour_.get(), make_scene(width, height, 3, kSceneNoise, seed));
  }

  void Submit(VPIStream stream) override {
    VPI_CHECK(vpiSubmitConvertImageFormat(stream, backend_, colour_.get(),
                                          grey_.get(), nullptr));
    VPI_CHECK(vpiSubmitGaussianFilter(stream, backend_, grey_.get(),
                                      blurred_.get(), kPipelineKernelSize,
                                      kPipelineKernelSize, kPipelineSigma,
                                      kPipelineSigma, VPI_BORDER_CLAMP));
    VPI_CHECK(vpiSubmitRescale(stream, backend_, blurred_.get(), small_.get(),
                               VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
    VPI_CHECK(vpiSubmitHarrisCornerDetector(stream, backend_, harris_.get(),
                                            small_.get(), keypoints_.get(),
                                            scores_.get(), &params_));
  }
  double input_pixels() const override {
    return static_cast<double>(width_) * height_;
  }
  const char* name() const override { return "pipeline"; }
  void PrintResult() const override {
    std::cout << "KEYPOINTS_LAST_FRAME="
              << download_keypoints(keypoints_.get()).size() << '\n';
  }
  VPIImage grey() const { return grey_.get(); }

 private:
  std::uint64_t backend_;
  int width_;
  int height_;
  Image colour_;
  Image grey_;
  Image blurred_;
  Image small_;
  Array keypoints_;
  Array scores_;
  Payload harris_;
  VPIHarrisCornerDetectorParams params_{};
};

const char* algorithm_name(Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::kGaussian:
      return "gaussian";
    case Algorithm::kRescale:
      return "rescale";
    case Algorithm::kHarris:
      return "harris";
    case Algorithm::kStereo:
      return "stereo";
    case Algorithm::kPipeline:
      return "pipeline";
  }
  return "unknown";
}

bool uses_large_frames(Algorithm algorithm) {
  return algorithm == Algorithm::kGaussian ||
         algorithm == Algorithm::kRescale || algorithm == Algorithm::kPipeline;
}

int frame_width(const Config& config) {
  if (config.width > 0) return config.width;
  return uses_large_frames(config.algorithm) ? kLargeWidth : kSmallWidth;
}

int frame_height(const Config& config) {
  if (config.height > 0) return config.height;
  return uses_large_frames(config.algorithm) ? kLargeHeight : kSmallHeight;
}

std::unique_ptr<Workload> make_workload(const Config& config) {
  const int width = frame_width(config);
  const int height = frame_height(config);
  const auto strength = [&config](float workload_default) {
    return config.harris_strength >= 0.0F ? config.harris_strength
                                          : workload_default;
  };
  switch (config.algorithm) {
    case Algorithm::kGaussian:
      return std::make_unique<GaussianWorkload>(config.backend, width, height,
                                                config.seed);
    case Algorithm::kRescale:
      return std::make_unique<RescaleWorkload>(config.backend, width, height,
                                               config.seed);
    case Algorithm::kHarris:
      return std::make_unique<HarrisWorkload>(config.backend, width, height,
                                              kSceneNoise,
                                              strength(kHarrisStrength),
                                              config.seed);
    case Algorithm::kStereo:
      return std::make_unique<StereoWorkload>(config.backend, width, height,
                                              config.seed);
    case Algorithm::kPipeline:
      return std::make_unique<PipelineWorkload>(
          config.backend, width, height, strength(kPipelineHarrisStrength),
          config.seed);
  }
  throw std::invalid_argument("unknown algorithm");
}

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "NVIDIA VPI computer-vision algorithms on synthetic frames.\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --algorithm <name>        gaussian, rescale, harris, stereo or\n"
      << "                            pipeline (default: pipeline)\n"
      << "  --backend <name>          cuda, cpu, pva, vic or ofa (default: cuda)\n"
      << "  --mode <name>             measure or verify (default: measure)\n"
      << "  --width <N>               Frame width (default: 3840 for gaussian,\n"
      << "                            rescale and pipeline, otherwise 1920)\n"
      << "  --height <N>              Frame height (default: 2160 or 1080)\n"
      << "  --harris-strength <val>   Harris strength threshold (default: 10\n"
      << "                            for harris, 5 for pipeline)\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 10)\n"
      << "  --duration <seconds>      Measurement phase (default: 40)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 10)\n"
      << "  --seed <N>                Synthetic frame seed (default: 2024)\n"
      << "  --help                    Show this help\n";
}

Config parse_config(const benchmarks::ArgumentParser& arguments) {
  arguments.validate_options({"--device", "--algorithm", "--backend", "--mode",
                              "--width", "--height", "--harris-strength",
                              "--warmup-seconds",
                              "--duration", "--cooldown-seconds", "--seed"});
  Config config;
  config.device = arguments.integer_value("--device", config.device);
  config.width = arguments.integer_value("--width", config.width);
  config.height = arguments.integer_value("--height", config.height);
  config.warmup_seconds =
      arguments.double_value("--warmup-seconds", config.warmup_seconds);
  config.measurement_seconds =
      arguments.double_value("--duration", config.measurement_seconds);
  config.cooldown_seconds =
      arguments.double_value("--cooldown-seconds", config.cooldown_seconds);
  config.seed = arguments.unsigned_value("--seed", config.seed);
  config.harris_strength = static_cast<float>(
      arguments.double_value("--harris-strength", config.harris_strength));

  const std::string algorithm = arguments.string_value("--algorithm", "pipeline");
  bool known = false;
  for (Algorithm candidate : {Algorithm::kGaussian, Algorithm::kRescale,
                              Algorithm::kHarris, Algorithm::kStereo,
                              Algorithm::kPipeline}) {
    if (algorithm == algorithm_name(candidate)) {
      config.algorithm = candidate;
      known = true;
    }
  }
  if (!known) {
    throw std::invalid_argument(
        "--algorithm must be gaussian, rescale, harris, stereo or pipeline");
  }

  config.backend_name = arguments.string_value("--backend", "cuda");
  if (config.backend_name == "cuda") {
    config.backend = VPI_BACKEND_CUDA;
  } else if (config.backend_name == "cpu") {
    config.backend = VPI_BACKEND_CPU;
  } else if (config.backend_name == "pva") {
    config.backend = VPI_BACKEND_PVA;
  } else if (config.backend_name == "vic") {
    config.backend = VPI_BACKEND_VIC;
  } else if (config.backend_name == "ofa") {
    config.backend = VPI_BACKEND_OFA;
  } else {
    throw std::invalid_argument("--backend must be cuda, cpu, pva, vic or ofa");
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
  if (config.width < 0 || config.height < 0 || config.width % 2 != 0 ||
      config.height % 2 != 0) {
    throw std::invalid_argument("--width and --height must be even and positive");
  }
  if (config.algorithm == Algorithm::kStereo &&
      frame_width(config) <= kStereoMaxDisparity + kStereoTrueDisparity +
                                 kStereoBorder) {
    throw std::invalid_argument("stereo frames are too narrow");
  }
  if (!(config.warmup_seconds >= 0.0) || !(config.measurement_seconds > 0.0) ||
      !(config.cooldown_seconds >= 0.0)) {
    throw std::invalid_argument(
        "phase durations must be finite; measurement must be positive");
  }
  return config;
}

void print_config(const Config& config, const cudaDeviceProp& properties) {
  const std::int32_t version = vpiGetVersion();
  std::cout << std::setprecision(12) << "BENCHMARK=vpi\n"
            << "ALGORITHM=" << algorithm_name(config.algorithm) << '\n'
            << "BACKEND=" << config.backend_name << '\n'
            << "MODE=" << (config.mode == Mode::kMeasure ? "measure" : "verify")
            << '\n'
            << "VPI_VERSION=" << version / 1000000 << '.'
            << version / 10000 % 100 << '.' << version / 100 % 100 << '\n'
            << "DEVICE=" << config.device << '\n'
            << "DEVICE_NAME=" << properties.name << '\n'
            << "COMPUTE_CAPABILITY=" << properties.major << '.'
            << properties.minor << '\n'
            << "FRAME_WIDTH=" << frame_width(config) << '\n'
            << "FRAME_HEIGHT=" << frame_height(config) << '\n'
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
  std::uint64_t frames = 0;
};

PhaseStats run_phase(Workload* workload, VPIStream stream, double seconds) {
  PhaseStats stats;
  const auto start = Clock::now();
  const auto deadline =
      start + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(seconds));
  while (Clock::now() < deadline) {
    for (int i = 0; i < kFramesBetweenSynchronizations; ++i) {
      workload->Submit(stream);
    }
    VPI_CHECK(vpiStreamSync(stream));
    stats.frames += kFramesBetweenSynchronizations;
  }
  stats.elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  return stats;
}

int measure(const Config& config) {
  Stream stream;
  VPI_CHECK(vpiStreamCreate(config.backend, stream.out()));
  const std::unique_ptr<Workload> workload = make_workload(config);
  workload->Submit(stream.get());  // first frame: lazy allocations, JIT
  VPI_CHECK(vpiStreamSync(stream.get()));

  emit_phase("warmup", PhaseMarker::kWarmup);
  const PhaseStats warmup =
      run_phase(workload.get(), stream.get(), config.warmup_seconds);
  emit_phase("measurement", PhaseMarker::kMeasurement);
  const PhaseStats measurement =
      run_phase(workload.get(), stream.get(), config.measurement_seconds);
  emit_phase("cooldown", PhaseMarker::kCooldown);
  const auto cooldown_start = Clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.cooldown_seconds));
  const double cooldown_elapsed =
      std::chrono::duration<double>(Clock::now() - cooldown_start).count();
  emit_phase("complete", PhaseMarker::kComplete);

  const double frames = static_cast<double>(measurement.frames);
  std::cout << std::fixed << std::setprecision(6)
            << "WARMUP_ELAPSED_SECONDS=" << warmup.elapsed_seconds << '\n'
            << "WARMUP_FRAMES=" << warmup.frames << '\n'
            << "MEASUREMENT_ELAPSED_SECONDS=" << measurement.elapsed_seconds
            << '\n'
            << "MEASUREMENT_FRAMES=" << measurement.frames << '\n'
            << "FRAMES_PER_SECOND=" << frames / measurement.elapsed_seconds
            << '\n'
            << "MS_PER_FRAME=" << 1000.0 * measurement.elapsed_seconds / frames
            << '\n'
            << "MEGAPIXELS_PER_SECOND="
            << frames * workload->input_pixels() /
                   measurement.elapsed_seconds / 1e6
            << '\n';
  workload->PrintResult();
  std::cout << std::fixed << std::setprecision(6)
            << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed << '\n';
  benchmarks::print_execution_summary(measurement.elapsed_seconds,
                                      measurement.frames);
  return 0;
}

// Runs one frame of `workload` on its own stream and waits for it.
void run_once(Workload* workload) {
  Stream stream;
  VPI_CHECK(vpiStreamCreate(0, stream.out()));
  workload->Submit(stream.get());
  VPI_CHECK(vpiStreamSync(stream.get()));
}

struct PixelDifference {
  int max_abs = 0;
  double mean_abs = 0.0;
  double above_one_fraction = 0.0;  // pixels that differ by more than 1
};

PixelDifference compare_images(const HostImage<std::uint8_t>& a,
                               const HostImage<std::uint8_t>& b) {
  if (a.pixels.size() != b.pixels.size() || a.pixels.empty()) {
    throw std::runtime_error("verification images differ in size");
  }
  PixelDifference result;
  std::uint64_t sum = 0;
  std::uint64_t above_one = 0;
  for (std::size_t i = 0; i < a.pixels.size(); ++i) {
    const int difference = std::abs(static_cast<int>(a.pixels[i]) -
                                    static_cast<int>(b.pixels[i]));
    result.max_abs = std::max(result.max_abs, difference);
    sum += difference;
    above_one += difference > 1 ? 1 : 0;
  }
  result.mean_abs = static_cast<double>(sum) / a.pixels.size();
  result.above_one_fraction = static_cast<double>(above_one) / a.pixels.size();
  return result;
}

// Fraction of points in `from` that have a point of `to` within `radius`.
double matched_fraction(const std::vector<VPIKeypointF32>& from,
                        const std::vector<VPIKeypointF32>& to, double radius) {
  if (from.empty()) return 0.0;
  std::size_t matched = 0;
  for (const VPIKeypointF32& a : from) {
    for (const VPIKeypointF32& b : to) {
      if (std::hypot(a.x - b.x, a.y - b.y) <= radius) {
        ++matched;
        break;
      }
    }
  }
  return static_cast<double>(matched) / from.size();
}

class VerifyReport {
 public:
  void check(const std::string& name, double value, double limit,
             bool at_least) {
    const bool ok = std::isfinite(value) &&
                    (at_least ? value >= limit : value <= limit);
    passed_ = passed_ && ok;
    std::cout << std::setprecision(6) << "VERIFY_" << name << '=' << value
              << '\n'
              << "VERIFY_" << name << (at_least ? "_MIN=" : "_MAX=") << limit
              << '\n'
              << "VERIFY_" << name << "_STATUS=" << (ok ? "OK" : "FAILED")
              << '\n';
  }
  void check_pixels(const std::string& name, VPIImage cpu, VPIImage cuda) {
    const PixelDifference difference = compare_images(
        download<std::uint8_t>(cpu), download<std::uint8_t>(cuda));
    std::cout << std::setprecision(6) << "VERIFY_" << name
              << "_MEAN_ABS_DIFF_VS_CPU=" << difference.mean_abs << '\n';
    check(name + "_MAX_ABS_DIFF_VS_CPU", difference.max_abs,
          kVerifyMaxPixelDifference, false);
    check(name + "_ABOVE_1_FRACTION_VS_CPU", difference.above_one_fraction,
          kVerifyMaxAboveOneFraction, false);
  }
  bool passed() const { return passed_; }

 private:
  bool passed_ = true;
};

int verify(const Config& config) {
  const std::uint64_t seed = config.seed;
  VerifyReport report;
  const int large_width = config.width > 0 ? config.width : kLargeWidth;
  const int large_height = config.height > 0 ? config.height : kLargeHeight;
  const int small_width = config.width > 0 ? config.width : kSmallWidth;
  const int small_height = config.height > 0 ? config.height : kSmallHeight;

  {
    GaussianWorkload cpu(VPI_BACKEND_CPU, large_width, large_height, seed);
    GaussianWorkload cuda(VPI_BACKEND_CUDA, large_width, large_height, seed);
    run_once(&cpu);
    run_once(&cuda);
    report.check_pixels("GAUSSIAN", cpu.output(), cuda.output());
  }
  {
    RescaleWorkload cpu(VPI_BACKEND_CPU, large_width, large_height, seed);
    RescaleWorkload cuda(VPI_BACKEND_CUDA, large_width, large_height, seed);
    run_once(&cpu);
    run_once(&cuda);
    report.check_pixels("RESCALE", cpu.output(), cuda.output());
  }
  {
    PipelineWorkload cpu(VPI_BACKEND_CPU, large_width, large_height, -1.0F,
                         seed);
    PipelineWorkload cuda(VPI_BACKEND_CUDA, large_width, large_height, -1.0F,
                          seed);
    run_once(&cpu);
    run_once(&cuda);
    report.check_pixels("BGR_TO_GREY", cpu.grey(), cuda.grey());
  }
  {
    // Low texture so that the only corners are the squares' corners.
    HarrisWorkload cuda(VPI_BACKEND_CUDA, small_width, small_height,
                        kVerifyHarrisNoise, -1.0F, seed);
    run_once(&cuda);
    const std::vector<VPIKeypointF32> detected = cuda.keypoints();
    const std::vector<VPIKeypointF32> truth =
        scene_corners(small_width, small_height);
    std::cout << "VERIFY_HARRIS_KEYPOINTS=" << detected.size() << '\n'
              << "VERIFY_HARRIS_TRUE_CORNERS=" << truth.size() << '\n';
    report.check("HARRIS_PRECISION",
                 matched_fraction(detected, truth, kVerifyCornerRadius),
                 kVerifyMinCornerFraction, true);
    report.check("HARRIS_RECALL",
                 matched_fraction(truth, detected, kVerifyCornerRadius),
                 kVerifyMinCornerFraction, true);
  }
  {
    StereoWorkload cuda(VPI_BACKEND_CUDA, small_width, small_height, seed);
    run_once(&cuda);
    report.check("STEREO_WITHIN_1PX_FRACTION", cuda.accuracy(),
                 kVerifyMinDisparityFraction, true);
  }

  std::cout << "VERIFY=" << (report.passed() ? "OK" : "FAILED") << '\n';
  if (!report.passed()) {
    std::cout << "STATUS=FAILED\n";
    return 1;
  }
  benchmarks::print_execution_summary(0.0, 5);
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
    return measure(config);
  } catch (const std::exception& error) {
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
