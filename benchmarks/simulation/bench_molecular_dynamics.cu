// Lennard-Jones molecular dynamics benchmark.
//
// Simulates a Lennard-Jones liquid in reduced units with the settings of the
// LAMMPS "in.lj" benchmark: an FCC lattice at density 0.8442 melted from
// T = 1.44, cutoff 2.5 sigma, neighbor skin 0.3 sigma, neighbor list rebuilt
// every 20 steps without a displacement check, NVE velocity Verlet with
// dt = 0.005 tau. Each step runs four kernels (first half-kick and drift,
// optional cell binning and neighbor-list build, pair forces, second
// half-kick), so a capture shows a realistic mix of compute-bound force
// kernels and memory-bound integration kernels.
//
// Like bench_power_cycle, the run is split into warm-up, measurement and
// cooldown phases delimited by empty marker kernels. --mode verify instead
// checks the GPU forces against an exact O(N^2) host computation.

#include "benchmark_utils.hpp"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kDefaultWarmupSeconds = 10.0;
constexpr double kDefaultMeasurementSeconds = 40.0;
constexpr double kDefaultCooldownSeconds = 10.0;
// 4 * 32^3 = 131072 atoms, the size of the LAMMPS in.lj benchmark x4.
constexpr int kDefaultLatticeCells = 32;
constexpr int kMinLatticeCells = 2;
constexpr int kMaxLatticeCells = 128;
constexpr double kDefaultDensity = 0.8442;
constexpr double kDefaultTemperature = 1.44;
constexpr double kDefaultCutoff = 2.5;
constexpr double kDefaultSkin = 0.3;
constexpr double kDefaultTimestep = 0.005;
constexpr int kDefaultNeighborEvery = 20;
constexpr int kDefaultVerifySteps = 200;
constexpr std::uint64_t kDefaultSeed = 12345;

// Fixed capacities; overflow is detected and reported, never silent. At the
// default density a cell holds about 19 atoms and an atom about 78 neighbors.
constexpr int kCellCapacity = 96;
constexpr int kMaxNeighbors = 192;
constexpr int kStepsBetweenChecks = 20;
constexpr int kReductionBlock = 256;
constexpr int kVerifyMaxAtoms = 32000;
constexpr double kVerifyToleranceFp32 = 1e-4;
constexpr double kVerifyToleranceFp64 = 1e-9;

constexpr int kStatusCellOverflow = 1;
constexpr int kStatusNeighborOverflow = 2;

enum class Precision { kFp32, kFp64 };
enum class Mode { kMeasure, kVerify };

struct Config {
  int device = benchmarks::kDefaultDevice;
  int block_size = benchmarks::kDefaultBlockSize;
  double warmup_seconds = kDefaultWarmupSeconds;
  double measurement_seconds = kDefaultMeasurementSeconds;
  double cooldown_seconds = kDefaultCooldownSeconds;
  int lattice_cells = kDefaultLatticeCells;
  double density = kDefaultDensity;
  double temperature = kDefaultTemperature;
  double cutoff = kDefaultCutoff;
  double skin = kDefaultSkin;
  double timestep = kDefaultTimestep;
  int neighbor_every = kDefaultNeighborEvery;
  int verify_steps = kDefaultVerifySteps;
  std::uint64_t seed = kDefaultSeed;
  Precision precision = Precision::kFp32;
  Mode mode = Mode::kMeasure;
};

struct PhaseStats {
  double elapsed_seconds = 0.0;
  std::uint64_t steps = 0;
};

struct Energies {
  double potential_per_atom = 0.0;
  double kinetic_per_atom = 0.0;
  double total_per_atom = 0.0;
  double temperature = 0.0;
};

// Plain view of the device state, passed to kernels by value.
template <typename Real>
struct SystemView {
  int atoms = 0;
  int cells_per_side = 0;
  Real box = 0;
  Real inverse_box = 0;
  Real cells_per_length = 0;
  Real cutoff_sq = 0;
  Real list_cutoff_sq = 0;
  Real timestep = 0;
  Real* x = nullptr;
  Real* y = nullptr;
  Real* z = nullptr;
  Real* vx = nullptr;
  Real* vy = nullptr;
  Real* vz = nullptr;
  Real* fx = nullptr;
  Real* fy = nullptr;
  Real* fz = nullptr;
  int* cell_counts = nullptr;
  int* cell_atoms = nullptr;
  int* neighbor_counts = nullptr;
  int* neighbors = nullptr;  // neighbors[k * atoms + i], coalesced over i
  int* status = nullptr;
  int* max_neighbors = nullptr;
};

__global__ void benchmark_warmup_start_marker() {}
__global__ void benchmark_measurement_start_marker() {}
__global__ void benchmark_cooldown_start_marker() {}
__global__ void benchmark_complete_marker() {}

template <typename Real>
__device__ Real wrap_position(Real position, const SystemView<Real>& system) {
  return position - system.box * floor(position * system.inverse_box);
}

template <typename Real>
__device__ Real minimum_image(Real delta, const SystemView<Real>& system) {
  return delta - system.box * rint(delta * system.inverse_box);
}

template <typename Real>
__device__ int cell_coordinate(Real position, const SystemView<Real>& system) {
  const int cell = static_cast<int>(position * system.cells_per_length);
  return min(max(cell, 0), system.cells_per_side - 1);
}

template <typename Real>
__global__ void md_integrate_first_half_kernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const Real half_step = Real(0.5) * system.timestep;
  const Real vx = system.vx[i] + half_step * system.fx[i];
  const Real vy = system.vy[i] + half_step * system.fy[i];
  const Real vz = system.vz[i] + half_step * system.fz[i];
  system.vx[i] = vx;
  system.vy[i] = vy;
  system.vz[i] = vz;
  system.x[i] = wrap_position(system.x[i] + system.timestep * vx, system);
  system.y[i] = wrap_position(system.y[i] + system.timestep * vy, system);
  system.z[i] = wrap_position(system.z[i] + system.timestep * vz, system);
}

template <typename Real>
__global__ void md_integrate_second_half_kernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const Real half_step = Real(0.5) * system.timestep;
  system.vx[i] += half_step * system.fx[i];
  system.vy[i] += half_step * system.fy[i];
  system.vz[i] += half_step * system.fz[i];
}

template <typename Real>
__global__ void md_bin_atoms_kernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const int n = system.cells_per_side;
  const int cell = cell_coordinate(system.x[i], system) +
                   n * (cell_coordinate(system.y[i], system) +
                        n * cell_coordinate(system.z[i], system));
  const int slot = atomicAdd(&system.cell_counts[cell], 1);
  if (slot < kCellCapacity) {
    system.cell_atoms[cell * kCellCapacity + slot] = i;
  } else {
    atomicOr(system.status, kStatusCellOverflow);
  }
}

// Full neighbor list (each pair stored twice): forces need no atomics, so a
// step is deterministic for a given list order.
template <typename Real>
__global__ void md_build_neighbor_list_kernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const int n = system.cells_per_side;
  const Real xi = system.x[i];
  const Real yi = system.y[i];
  const Real zi = system.z[i];
  const int cx = cell_coordinate(xi, system);
  const int cy = cell_coordinate(yi, system);
  const int cz = cell_coordinate(zi, system);
  int count = 0;
  for (int oz = -1; oz <= 1; ++oz) {
    const int z = (cz + oz + n) % n;
    for (int oy = -1; oy <= 1; ++oy) {
      const int y = (cy + oy + n) % n;
      for (int ox = -1; ox <= 1; ++ox) {
        const int cell = (cx + ox + n) % n + n * (y + n * z);
        const int members = min(system.cell_counts[cell], kCellCapacity);
        for (int slot = 0; slot < members; ++slot) {
          const int j = system.cell_atoms[cell * kCellCapacity + slot];
          if (j == i) continue;
          const Real dx = minimum_image(system.x[j] - xi, system);
          const Real dy = minimum_image(system.y[j] - yi, system);
          const Real dz = minimum_image(system.z[j] - zi, system);
          if (dx * dx + dy * dy + dz * dz < system.list_cutoff_sq) {
            if (count < kMaxNeighbors) {
              system.neighbors[static_cast<std::size_t>(count) * system.atoms +
                               i] = j;
            }
            ++count;
          }
        }
      }
    }
  }
  system.neighbor_counts[i] = min(count, kMaxNeighbors);
  if (count > kMaxNeighbors) atomicOr(system.status, kStatusNeighborOverflow);
  atomicMax(system.max_neighbors, count);
}

// F_i = sum_j 48 r^-8 (r^-6 - 1/2) (r_i - r_j), epsilon = sigma = mass = 1.
template <typename Real>
__global__ void md_lj_force_kernel(SystemView<Real> system) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= system.atoms) return;
  const Real xi = system.x[i];
  const Real yi = system.y[i];
  const Real zi = system.z[i];
  Real fx = 0;
  Real fy = 0;
  Real fz = 0;
  const int count = system.neighbor_counts[i];
  for (int k = 0; k < count; ++k) {
    const int j =
        system.neighbors[static_cast<std::size_t>(k) * system.atoms + i];
    const Real dx = minimum_image(xi - system.x[j], system);
    const Real dy = minimum_image(yi - system.y[j], system);
    const Real dz = minimum_image(zi - system.z[j], system);
    const Real r_sq = dx * dx + dy * dy + dz * dz;
    if (r_sq < system.cutoff_sq) {
      const Real inverse_r2 = Real(1) / r_sq;
      const Real inverse_r6 = inverse_r2 * inverse_r2 * inverse_r2;
      const Real force_over_r =
          Real(48) * inverse_r6 * (inverse_r6 - Real(0.5)) * inverse_r2;
      fx += force_over_r * dx;
      fy += force_over_r * dy;
      fz += force_over_r * dz;
    }
  }
  system.fx[i] = fx;
  system.fy[i] = fy;
  system.fz[i] = fz;
}

// Per-block sums of potential and kinetic energy, accumulated in double for
// both precisions so the drift reflects the dynamics, not the reduction.
template <typename Real>
__global__ void md_energy_kernel(SystemView<Real> system, double* partials) {
  __shared__ double potential[kReductionBlock];
  __shared__ double kinetic[kReductionBlock];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double atom_potential = 0.0;
  double atom_kinetic = 0.0;
  if (i < system.atoms) {
    const int count = system.neighbor_counts[i];
    for (int k = 0; k < count; ++k) {
      const int j =
          system.neighbors[static_cast<std::size_t>(k) * system.atoms + i];
      const Real dx = minimum_image(system.x[i] - system.x[j], system);
      const Real dy = minimum_image(system.y[i] - system.y[j], system);
      const Real dz = minimum_image(system.z[i] - system.z[j], system);
      const Real r_sq = dx * dx + dy * dy + dz * dz;
      if (r_sq < system.cutoff_sq) {
        const double inverse_r6 = 1.0 / (static_cast<double>(r_sq) * r_sq * r_sq);
        // Each pair appears twice in the full list.
        atom_potential += 2.0 * inverse_r6 * (inverse_r6 - 1.0);
      }
    }
    const double vx = system.vx[i];
    const double vy = system.vy[i];
    const double vz = system.vz[i];
    atom_kinetic = 0.5 * (vx * vx + vy * vy + vz * vz);
  }
  potential[threadIdx.x] = atom_potential;
  kinetic[threadIdx.x] = atom_kinetic;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      potential[threadIdx.x] += potential[threadIdx.x + stride];
      kinetic[threadIdx.x] += kinetic[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[2 * blockIdx.x] = potential[0];
    partials[2 * blockIdx.x + 1] = kinetic[0];
  }
}

template <typename T>
T* device_allocate(std::size_t count) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

int grid_size(int items, int block_size) {
  return (items + block_size - 1) / block_size;
}

// Owns the device buffers of one simulation and advances it step by step.
template <typename Real>
class Simulation {
 public:
  explicit Simulation(const Config& config)
      : block_size_(config.block_size), neighbor_every_(config.neighbor_every) {
    const int n = config.lattice_cells;
    view_.atoms = 4 * n * n * n;
    const double lattice = std::cbrt(4.0 / config.density);
    const double box = n * lattice;
    const double list_cutoff = config.cutoff + config.skin;
    view_.cells_per_side = static_cast<int>(std::floor(box / list_cutoff));
    if (view_.cells_per_side < 3) {
      throw std::invalid_argument(
          "box is smaller than three neighbor cells; raise --lattice-cells");
    }
    view_.box = static_cast<Real>(box);
    view_.inverse_box = static_cast<Real>(1.0 / box);
    view_.cells_per_length = static_cast<Real>(view_.cells_per_side / box);
    view_.cutoff_sq = static_cast<Real>(config.cutoff * config.cutoff);
    view_.list_cutoff_sq = static_cast<Real>(list_cutoff * list_cutoff);
    view_.timestep = static_cast<Real>(config.timestep);
    box_length_ = box;
    Allocate();
    Initialize(config, lattice);
    Rebuild();
    LaunchForces();
    Check();
  }

  ~Simulation() { Release(); }
  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;

  // One velocity Verlet step, with the neighbor list rebuilt every
  // neighbor_every steps before the force evaluation (as LAMMPS does).
  void Step() {
    const int grid = grid_size(view_.atoms, block_size_);
    md_integrate_first_half_kernel<<<grid, block_size_>>>(view_);
    ++steps_;
    if (steps_ % neighbor_every_ == 0) Rebuild();
    LaunchForces();
    md_integrate_second_half_kernel<<<grid, block_size_>>>(view_);
    CUDA_CHECK(cudaGetLastError());
  }

  // Waits for queued work and throws if a fixed capacity was exceeded.
  void Check() {
    int status = 0;
    CUDA_CHECK(cudaMemcpy(&status, view_.status, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (status & kStatusCellOverflow) {
      throw std::runtime_error("cell capacity exceeded (kCellCapacity)");
    }
    if (status & kStatusNeighborOverflow) {
      throw std::runtime_error("neighbor capacity exceeded (kMaxNeighbors)");
    }
  }

  Energies ComputeEnergies() {
    const int blocks = grid_size(view_.atoms, kReductionBlock);
    md_energy_kernel<<<blocks, kReductionBlock>>>(view_, partials_);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> host(2 * static_cast<std::size_t>(blocks));
    CUDA_CHECK(cudaMemcpy(host.data(), partials_, host.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double potential = 0.0;
    double kinetic = 0.0;
    for (int block = 0; block < blocks; ++block) {
      potential += host[2 * block];
      kinetic += host[2 * block + 1];
    }
    Energies energies;
    energies.potential_per_atom = potential / view_.atoms;
    energies.kinetic_per_atom = kinetic / view_.atoms;
    energies.total_per_atom =
        energies.potential_per_atom + energies.kinetic_per_atom;
    energies.temperature = 2.0 * kinetic / (3.0 * (view_.atoms - 1));
    return energies;
  }

  int MaxNeighbors() const {
    int value = 0;
    CUDA_CHECK(cudaMemcpy(&value, view_.max_neighbors, sizeof(int),
                          cudaMemcpyDeviceToHost));
    return value;
  }

  // Positions and forces as double, for the host force check.
  void Download(std::vector<double>* positions,
                std::vector<double>* forces) const {
    positions->assign(3 * static_cast<std::size_t>(view_.atoms), 0.0);
    forces->assign(3 * static_cast<std::size_t>(view_.atoms), 0.0);
    const Real* sources[6] = {view_.x,  view_.y,  view_.z,
                              view_.fx, view_.fy, view_.fz};
    std::vector<Real> buffer(view_.atoms);
    for (int axis = 0; axis < 6; ++axis) {
      CUDA_CHECK(cudaMemcpy(buffer.data(), sources[axis],
                            buffer.size() * sizeof(Real),
                            cudaMemcpyDeviceToHost));
      std::vector<double>* target = axis < 3 ? positions : forces;
      for (int i = 0; i < view_.atoms; ++i) {
        (*target)[3 * static_cast<std::size_t>(i) + axis % 3] = buffer[i];
      }
    }
  }

  int atoms() const { return view_.atoms; }
  int cells_per_side() const { return view_.cells_per_side; }
  double box_length() const { return box_length_; }
  std::uint64_t rebuilds() const { return rebuilds_; }

 private:
  void Allocate() {
    const std::size_t atoms = view_.atoms;
    const std::size_t cells = static_cast<std::size_t>(view_.cells_per_side) *
                              view_.cells_per_side * view_.cells_per_side;
    Real** arrays[9] = {&view_.x,  &view_.y,  &view_.z,  &view_.vx, &view_.vy,
                        &view_.vz, &view_.fx, &view_.fy, &view_.fz};
    for (Real** array : arrays) *array = device_allocate<Real>(atoms);
    view_.cell_counts = device_allocate<int>(cells);
    view_.cell_atoms = device_allocate<int>(cells * kCellCapacity);
    view_.neighbor_counts = device_allocate<int>(atoms);
    view_.neighbors = device_allocate<int>(atoms * kMaxNeighbors);
    view_.status = device_allocate<int>(1);
    view_.max_neighbors = device_allocate<int>(1);
    partials_ = device_allocate<double>(
        2 * static_cast<std::size_t>(grid_size(view_.atoms, kReductionBlock)));
    cells_ = cells;
    CUDA_CHECK(cudaMemset(view_.status, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(view_.max_neighbors, 0, sizeof(int)));
  }

  void Release() noexcept {
    void* pointers[] = {view_.x,           view_.y,         view_.z,
                        view_.vx,          view_.vy,        view_.vz,
                        view_.fx,          view_.fy,        view_.fz,
                        view_.cell_counts, view_.cell_atoms,
                        view_.neighbor_counts, view_.neighbors,
                        view_.status,      view_.max_neighbors, partials_};
    for (void* pointer : pointers) {
      if (pointer != nullptr) cudaFree(pointer);
    }
  }

  // FCC lattice and random velocities with zero total momentum, scaled to
  // the requested temperature (3N - 3 degrees of freedom).
  void Initialize(const Config& config, double lattice) {
    const int n = config.lattice_cells;
    const std::size_t atoms = view_.atoms;
    std::vector<double> host[6];
    for (auto& values : host) values.reserve(atoms);
    constexpr double kBasis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          for (const auto& basis : kBasis) {
            host[0].push_back((i + basis[0]) * lattice);
            host[1].push_back((j + basis[1]) * lattice);
            host[2].push_back((k + basis[2]) * lattice);
          }
        }
      }
    }
    std::mt19937_64 generator(config.seed);
    std::uniform_real_distribution<double> uniform(-0.5, 0.5);
    double momentum[3] = {0.0, 0.0, 0.0};
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      for (int axis = 0; axis < 3; ++axis) {
        host[3 + axis].push_back(uniform(generator));
        momentum[axis] += host[3 + axis].back();
      }
    }
    double kinetic_twice = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      for (double& velocity : host[3 + axis]) {
        velocity -= momentum[axis] / atoms;
        kinetic_twice += velocity * velocity;
      }
    }
    const double scale =
        std::sqrt(config.temperature * 3.0 * (atoms - 1) / kinetic_twice);
    Real* targets[6] = {view_.x, view_.y, view_.z, view_.vx, view_.vy, view_.vz};
    std::vector<Real> buffer(atoms);
    for (int array = 0; array < 6; ++array) {
      const double factor = array < 3 ? 1.0 : scale;
      for (std::size_t atom = 0; atom < atoms; ++atom) {
        buffer[atom] = static_cast<Real>(host[array][atom] * factor);
      }
      CUDA_CHECK(cudaMemcpy(targets[array], buffer.data(),
                            atoms * sizeof(Real), cudaMemcpyHostToDevice));
    }
  }

  void Rebuild() {
    const int grid = grid_size(view_.atoms, block_size_);
    CUDA_CHECK(cudaMemsetAsync(view_.cell_counts, 0, cells_ * sizeof(int)));
    md_bin_atoms_kernel<<<grid, block_size_>>>(view_);
    md_build_neighbor_list_kernel<<<grid, block_size_>>>(view_);
    CUDA_CHECK(cudaGetLastError());
    ++rebuilds_;
  }

  void LaunchForces() {
    md_lj_force_kernel<<<grid_size(view_.atoms, block_size_), block_size_>>>(
        view_);
    CUDA_CHECK(cudaGetLastError());
  }

  SystemView<Real> view_;
  double* partials_ = nullptr;
  std::size_t cells_ = 0;
  double box_length_ = 0.0;
  int block_size_ = 0;
  int neighbor_every_ = 0;
  std::uint64_t steps_ = 0;
  std::uint64_t rebuilds_ = 0;
};

// Largest |F_gpu - F_exact| over atoms, relative to the largest |F_exact|.
double max_relative_force_error(const std::vector<double>& positions,
                                const std::vector<double>& forces,
                                double box, double cutoff) {
  const std::size_t atoms = positions.size() / 3;
  const double cutoff_sq = cutoff * cutoff;
  double max_error = 0.0;
  double max_force = 0.0;
  for (std::size_t i = 0; i < atoms; ++i) {
    double exact[3] = {0.0, 0.0, 0.0};
    for (std::size_t j = 0; j < atoms; ++j) {
      if (j == i) continue;
      double delta[3];
      for (int axis = 0; axis < 3; ++axis) {
        const double d = positions[3 * i + axis] - positions[3 * j + axis];
        delta[axis] = d - box * std::nearbyint(d / box);
      }
      const double r_sq =
          delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
      if (r_sq >= cutoff_sq) continue;
      const double inverse_r2 = 1.0 / r_sq;
      const double inverse_r6 = inverse_r2 * inverse_r2 * inverse_r2;
      const double force_over_r = 48.0 * inverse_r6 * (inverse_r6 - 0.5) *
                                  inverse_r2;
      for (int axis = 0; axis < 3; ++axis) {
        exact[axis] += force_over_r * delta[axis];
      }
    }
    double error_sq = 0.0;
    double force_sq = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double difference = forces[3 * i + axis] - exact[axis];
      error_sq += difference * difference;
      force_sq += exact[axis] * exact[axis];
    }
    max_error = std::max(max_error, std::sqrt(error_sq));
    max_force = std::max(max_force, std::sqrt(force_sq));
  }
  return max_force > 0.0 ? max_error / max_force : max_error;
}

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Lennard-Jones molecular dynamics (LAMMPS in.lj settings).\n\n"
      << "Options:\n"
      << "  --device <id>             CUDA device (default: 0)\n"
      << "  --precision <name>        fp32 or fp64 (default: fp32)\n"
      << "  --mode <name>             measure or verify (default: measure)\n"
      << "  --lattice-cells <N>       FCC cells per side; atoms = 4 N^3\n"
      << "                            (default: 32, 131072 atoms)\n"
      << "  --warmup-seconds <value>  Timed warm-up phase (default: 10)\n"
      << "  --duration <seconds>      Measurement phase (default: 40)\n"
      << "  --cooldown-seconds <val>  Context-idle phase (default: 10)\n"
      << "  --density <value>         Reduced density (default: 0.8442)\n"
      << "  --temperature <value>     Initial temperature (default: 1.44)\n"
      << "  --cutoff <value>          Pair cutoff in sigma (default: 2.5)\n"
      << "  --skin <value>            Neighbor skin in sigma (default: 0.3)\n"
      << "  --timestep <value>        Step in tau (default: 0.005)\n"
      << "  --neighbor-every <N>      Rebuild period in steps (default: 20)\n"
      << "  --verify-steps <N>        Steps before the force check\n"
      << "                            (default: 200)\n"
      << "  --seed <N>                Velocity seed (default: 12345)\n"
      << "  --block-size <threads>    CUDA block size (default: 256)\n"
      << "  --help                    Show this help\n";
}

Config parse_config(const benchmarks::ArgumentParser& arguments) {
  arguments.validate_options(
      {"--device", "--precision", "--mode", "--lattice-cells",
       "--warmup-seconds", "--duration", "--cooldown-seconds", "--density",
       "--temperature", "--cutoff", "--skin", "--timestep", "--neighbor-every",
       "--verify-steps", "--seed", "--block-size"});

  Config config;
  config.device = arguments.integer_value("--device", config.device);
  config.block_size = arguments.integer_value("--block-size", config.block_size);
  config.warmup_seconds =
      arguments.double_value("--warmup-seconds", config.warmup_seconds);
  config.measurement_seconds =
      arguments.double_value("--duration", config.measurement_seconds);
  config.cooldown_seconds =
      arguments.double_value("--cooldown-seconds", config.cooldown_seconds);
  config.lattice_cells =
      arguments.integer_value("--lattice-cells", config.lattice_cells);
  config.density = arguments.double_value("--density", config.density);
  config.temperature =
      arguments.double_value("--temperature", config.temperature);
  config.cutoff = arguments.double_value("--cutoff", config.cutoff);
  config.skin = arguments.double_value("--skin", config.skin);
  config.timestep = arguments.double_value("--timestep", config.timestep);
  config.neighbor_every =
      arguments.integer_value("--neighbor-every", config.neighbor_every);
  config.verify_steps =
      arguments.integer_value("--verify-steps", config.verify_steps);
  config.seed = arguments.unsigned_value("--seed", config.seed);

  const std::string precision = arguments.string_value("--precision", "fp32");
  if (precision == "fp32") {
    config.precision = Precision::kFp32;
  } else if (precision == "fp64") {
    config.precision = Precision::kFp64;
  } else {
    throw std::invalid_argument("--precision must be fp32 or fp64");
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
  if (config.block_size <= 0 || config.block_size > 1024) {
    throw std::invalid_argument("--block-size must be in the range [1, 1024]");
  }
  if (!(config.warmup_seconds >= 0.0) || !(config.measurement_seconds > 0.0) ||
      !(config.cooldown_seconds >= 0.0)) {
    throw std::invalid_argument(
        "phase durations must be finite; measurement must be positive");
  }
  if (config.lattice_cells < kMinLatticeCells ||
      config.lattice_cells > kMaxLatticeCells) {
    throw std::invalid_argument("--lattice-cells must be in the range [2, 128]");
  }
  if (!(config.density > 0.0) || !(config.temperature > 0.0) ||
      !(config.cutoff > 0.0) || !(config.skin >= 0.0) ||
      !(config.timestep > 0.0)) {
    throw std::invalid_argument(
        "density, temperature, cutoff and timestep must be positive; skin "
        "must be non-negative");
  }
  if (config.neighbor_every < 1 || config.verify_steps < 0) {
    throw std::invalid_argument(
        "--neighbor-every must be positive and --verify-steps non-negative");
  }
  // The force check needs a freshly built list, so it runs on a rebuild step.
  config.verify_steps = (config.verify_steps + config.neighbor_every - 1) /
                        config.neighbor_every * config.neighbor_every;
  return config;
}

const char* precision_name(Precision precision) {
  return precision == Precision::kFp32 ? "fp32" : "fp64";
}

void print_config(const Config& config, const cudaDeviceProp& properties) {
  std::cout << std::setprecision(12) << "BENCHMARK=molecular_dynamics\n"
            << "POTENTIAL=lennard_jones\n"
            << "PRECISION=" << precision_name(config.precision) << '\n'
            << "MODE="
            << (config.mode == Mode::kMeasure ? "measure" : "verify") << '\n'
            << "DEVICE=" << config.device << '\n'
            << "DEVICE_NAME=" << properties.name << '\n'
            << "COMPUTE_CAPABILITY=" << properties.major << '.'
            << properties.minor << '\n'
            << "LATTICE_CELLS=" << config.lattice_cells << '\n'
            << "DENSITY=" << config.density << '\n'
            << "TEMPERATURE=" << config.temperature << '\n'
            << "CUTOFF=" << config.cutoff << '\n'
            << "SKIN=" << config.skin << '\n'
            << "TIMESTEP=" << config.timestep << '\n'
            << "NEIGHBOR_EVERY=" << config.neighbor_every << '\n'
            << "BLOCK_SIZE=" << config.block_size << '\n'
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

template <typename Real>
PhaseStats run_phase(Simulation<Real>* simulation, double seconds) {
  PhaseStats stats;
  const auto start = Clock::now();
  const auto deadline =
      start + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(seconds));
  while (Clock::now() < deadline) {
    for (int step = 0; step < kStepsBetweenChecks; ++step) simulation->Step();
    simulation->Check();
    stats.steps += kStepsBetweenChecks;
  }
  stats.elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  return stats;
}

void print_energies(const char* prefix, const Energies& energies) {
  std::cout << std::scientific << std::setprecision(10) << prefix
            << "_POTENTIAL_PER_ATOM=" << energies.potential_per_atom << '\n'
            << prefix << "_KINETIC_PER_ATOM=" << energies.kinetic_per_atom
            << '\n'
            << prefix << "_TOTAL_ENERGY_PER_ATOM=" << energies.total_per_atom
            << '\n'
            << prefix << "_TEMPERATURE=" << energies.temperature << '\n';
}

void print_system(int atoms, int cells_per_side, double box_length) {
  std::cout << std::setprecision(12) << "ATOMS=" << atoms << '\n'
            << "BOX_LENGTH=" << box_length << '\n'
            << "CELLS_PER_SIDE=" << cells_per_side << '\n';
}

template <typename Real>
int verify(const Config& config) {
  const int n = config.lattice_cells;
  if (4 * n * n * n > kVerifyMaxAtoms) {
    throw std::invalid_argument(
        "verify mode needs at most 32000 atoms; use --lattice-cells 20 or "
        "fewer");
  }
  Simulation<Real> simulation(config);
  print_system(simulation.atoms(), simulation.cells_per_side(),
               simulation.box_length());
  const Energies initial = simulation.ComputeEnergies();
  for (int step = 0; step < config.verify_steps; ++step) simulation.Step();
  simulation.Check();
  const Energies final_energies = simulation.ComputeEnergies();

  std::vector<double> positions;
  std::vector<double> forces;
  simulation.Download(&positions, &forces);
  const double error = max_relative_force_error(
      positions, forces, simulation.box_length(), config.cutoff);
  const double tolerance = config.precision == Precision::kFp32
                               ? kVerifyToleranceFp32
                               : kVerifyToleranceFp64;
  const bool passed = std::isfinite(error) && error <= tolerance;

  print_energies("INITIAL", initial);
  print_energies("FINAL", final_energies);
  std::cout << std::scientific << std::setprecision(6)
            << "VERIFY_STEPS=" << config.verify_steps << '\n'
            << "MAX_NEIGHBORS=" << simulation.MaxNeighbors() << '\n'
            << "VERIFY_MAX_RELATIVE_FORCE_ERROR=" << error << '\n'
            << "VERIFY_TOLERANCE=" << tolerance << '\n'
            << "VERIFY_RELATIVE_ENERGY_DRIFT="
            << (final_energies.total_per_atom - initial.total_per_atom) /
                   std::abs(initial.total_per_atom)
            << '\n'
            << "VERIFY=" << (passed ? "OK" : "FAILED") << '\n';
  if (!passed) {
    std::cout << "STATUS=FAILED\n";
    return 1;
  }
  benchmarks::print_execution_summary(0.0, config.verify_steps);
  return 0;
}

template <typename Real>
int measure(const Config& config) {
  Simulation<Real> simulation(config);
  print_system(simulation.atoms(), simulation.cells_per_side(),
               simulation.box_length());

  emit_phase("warmup", PhaseMarker::kWarmup);
  const PhaseStats warmup = run_phase(&simulation, config.warmup_seconds);
  // Energy reductions sit outside the measured phase.
  const Energies start = simulation.ComputeEnergies();

  emit_phase("measurement", PhaseMarker::kMeasurement);
  const std::uint64_t rebuilds_before = simulation.rebuilds();
  const PhaseStats measurement =
      run_phase(&simulation, config.measurement_seconds);
  const std::uint64_t rebuilds = simulation.rebuilds() - rebuilds_before;

  emit_phase("cooldown", PhaseMarker::kCooldown);
  const Energies end = simulation.ComputeEnergies();
  const auto cooldown_start = Clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(config.cooldown_seconds));
  const double cooldown_elapsed =
      std::chrono::duration<double>(Clock::now() - cooldown_start).count();
  emit_phase("complete", PhaseMarker::kComplete);

  const double atom_steps =
      static_cast<double>(measurement.steps) * simulation.atoms();
  print_energies("MEASUREMENT_START", start);
  print_energies("MEASUREMENT_END", end);
  std::cout << std::fixed << std::setprecision(6)
            << "WARMUP_ELAPSED_SECONDS=" << warmup.elapsed_seconds << '\n'
            << "WARMUP_STEPS=" << warmup.steps << '\n'
            << "MEASUREMENT_ELAPSED_SECONDS=" << measurement.elapsed_seconds
            << '\n'
            << "MEASUREMENT_STEPS=" << measurement.steps << '\n'
            << "MEASUREMENT_NEIGHBOR_REBUILDS=" << rebuilds << '\n'
            << "STEPS_PER_SECOND="
            << measurement.steps / measurement.elapsed_seconds << '\n'
            << "MS_PER_STEP="
            << 1000.0 * measurement.elapsed_seconds / measurement.steps << '\n'
            << "ATOM_STEPS_PER_SECOND="
            << atom_steps / measurement.elapsed_seconds << '\n'
            << "MAX_NEIGHBORS=" << simulation.MaxNeighbors() << '\n'
            << std::scientific << std::setprecision(6)
            << "RELATIVE_ENERGY_DRIFT="
            << (end.total_per_atom - start.total_per_atom) /
                   std::abs(start.total_per_atom)
            << '\n'
            << std::fixed << "COOLDOWN_ELAPSED_SECONDS=" << cooldown_elapsed
            << '\n';
  benchmarks::print_execution_summary(measurement.elapsed_seconds,
                                      measurement.steps);
  return 0;
}

template <typename Real>
int run(const Config& config) {
  return config.mode == Mode::kVerify ? verify<Real>(config)
                                      : measure<Real>(config);
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
    if (config.block_size > properties.maxThreadsPerBlock) {
      throw std::invalid_argument("--block-size exceeds the device limit");
    }
    print_config(config, properties);
    return config.precision == Precision::kFp32 ? run<float>(config)
                                                : run<double>(config);
  } catch (const std::exception& error) {
    std::cerr << "STATUS=ERROR\nERROR=" << error.what() << '\n';
    return 1;
  }
}
