#include "nvidia-process-metrics-worker.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <thread>

namespace nvidia_process_metrics {
namespace {

constexpr auto kSamplerReadyTimeout = std::chrono::seconds(10);
constexpr auto kSamplerInterruptTimeout = std::chrono::seconds(5);
constexpr auto kSamplerTerminateTimeout = std::chrono::seconds(2);
constexpr auto kSamplerKillTimeout = std::chrono::seconds(2);
constexpr uint64_t kSamplerSamplingInterval = 1'000'000;
std::mutex sampler_fork_mutex;

bool ProcessExists(pid_t pid) {
  return kill(pid, 0) == 0 || errno == EPERM;
}

enum class ChildWaitResult { kExited, kTimedOut, kError };

struct SamplerStopResult {
  bool success;
  std::string error;
};

std::string DescribeChildStatus(int status) {
  if (WIFEXITED(status)) {
    return "exited with status " + std::to_string(WEXITSTATUS(status));
  }
  if (WIFSIGNALED(status)) {
    return "terminated by signal " + std::to_string(WTERMSIG(status));
  }
  return "ended with an unknown process status";
}

bool IsSuccessfulChildStatus(int status) {
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

ChildWaitResult WaitForChild(pid_t pid,
                             std::chrono::steady_clock::duration timeout,
                             int* status, std::string* error) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const pid_t waited = waitpid(pid, status, WNOHANG);
    if (waited == pid) {
      return ChildWaitResult::kExited;
    }
    if (waited < 0 && errno != EINTR) {
      *error = std::string("cannot wait for PM sampler: ") +
               std::strerror(errno);
      return ChildWaitResult::kError;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return ChildWaitResult::kTimedOut;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void SignalChild(pid_t pid, int signal_number) {
  if (kill(pid, signal_number) != 0 && errno != ESRCH) {
    std::fprintf(stderr, "cannot signal PM sampler %d: %s\n", pid,
                 std::strerror(errno));
  }
}

void ReapChildInBackground(pid_t pid) {
  try {
    std::thread([pid]() {
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
    }).detach();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "cannot create PM sampler reaper for %d: %s\n", pid,
                 error.what());
  }
}

SamplerStopResult StopSamplerProcess(pid_t pid) {
  int status = 0;
  std::string wait_error;
  SignalChild(pid, SIGINT);
  ChildWaitResult waited =
      WaitForChild(pid, kSamplerInterruptTimeout, &status, &wait_error);
  if (waited == ChildWaitResult::kExited) {
    return {IsSuccessfulChildStatus(status),
            "PM sampler " + DescribeChildStatus(status)};
  }
  if (waited == ChildWaitResult::kError) {
    return {false, wait_error};
  }

  std::fprintf(stderr, "PM sampler %d did not stop after SIGINT; sending SIGTERM\n",
               pid);
  SignalChild(pid, SIGTERM);
  waited = WaitForChild(pid, kSamplerTerminateTimeout, &status, &wait_error);
  if (waited == ChildWaitResult::kExited) {
    return {IsSuccessfulChildStatus(status),
            "PM sampler " + DescribeChildStatus(status)};
  }
  if (waited == ChildWaitResult::kError) {
    return {false, wait_error};
  }

  std::fprintf(stderr, "PM sampler %d did not stop after SIGTERM; sending SIGKILL\n",
               pid);
  SignalChild(pid, SIGKILL);
  waited = WaitForChild(pid, kSamplerKillTimeout, &status, &wait_error);
  if (waited == ChildWaitResult::kExited) {
    return {false, "PM sampler required SIGKILL"};
  }
  if (waited == ChildWaitResult::kError) {
    return {false, wait_error};
  }
  if (waited == ChildWaitResult::kTimedOut) {
    std::fprintf(stderr,
                 "PM sampler %d could not be reaped promptly; reaping asynchronously\n",
                 pid);
    ReapChildInBackground(pid);
  }
  return {false, "PM sampler did not terminate within the shutdown deadline"};
}

}  // namespace

const char* SessionStateName(SessionState state) {
  switch (state) {
    case SessionState::kCreated:
      return "created";
    case SessionState::kStarting:
      return "starting";
    case SessionState::kRunning:
      return "running";
    case SessionState::kCompleted:
      return "completed";
    case SessionState::kFailed:
      return "failed";
  }
  return "unknown";
}

NvidiaProcessMetricsWorker::NvidiaProcessMetricsWorker(
    std::string session_id, pid_t pid, uid_t owner_uid, gid_t owner_gid,
    int device, std::chrono::seconds max_duration,
    std::filesystem::path output_dir, std::filesystem::path sampler_path)
    : session_id_(std::move(session_id)),
      target_pid_(pid),
      owner_uid_(owner_uid),
      owner_gid_(owner_gid),
      device_(device),
      max_duration_(max_duration),
      output_dir_(std::move(output_dir)),
      sampler_path_(std::move(sampler_path)) {}

NvidiaProcessMetricsWorker::~NvidiaProcessMetricsWorker() { Stop(); }

bool NvidiaProcessMetricsWorker::Start(std::string* error) {
  std::scoped_lock lifecycle_lock(lifecycle_mutex_);
  const auto fail_start = [this, error]() {
    MakeResultsReadable();
    std::string ownership_error;
    if (!TransferOutputDirectoryOwnership(&ownership_error)) {
      if (!error->empty()) {
        *error += "; ";
      }
      *error += ownership_error;
    }
    SetState(SessionState::kFailed, *error);
    return false;
  };

  if (!ProcessExists(target_pid_)) {
    *error = "target PID does not exist";
    return fail_start();
  }
#if NVIDIA_PROCESS_METRICS_ENABLE_PM_SAMPLING
  SetState(SessionState::kStarting, "starting PM Sampling");
  if (!StartSampler(error)) {
    return fail_start();
  }
  if (!WaitForSamplerReady(error)) {
    SetState(SessionState::kFailed, *error);
    Finish("PM Sampling did not become ready");
    return false;
  }
#endif

  if (!TransferOutputDirectoryOwnership(error)) {
#if NVIDIA_PROCESS_METRICS_ENABLE_PM_SAMPLING
    Finish("cannot transfer output directory ownership");
#endif
    SetState(SessionState::kFailed, *error);
    return false;
  }

  SetState(SessionState::kRunning,
#if NVIDIA_PROCESS_METRICS_ENABLE_PM_SAMPLING
           "running"
#else
           "running_without_pm_sampling"
#endif
  );
  monitor_thread_ = std::thread(&NvidiaProcessMetricsWorker::Monitor, this);
  return true;
}

void NvidiaProcessMetricsWorker::Stop() {
  std::scoped_lock lifecycle_lock(lifecycle_mutex_);
  stop_requested_.store(true);
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
    return;
  }
  if (sampler_pid_ > 0) {
    Finish("stopped");
  }
}

void NvidiaProcessMetricsWorker::Fail(const std::string& reason) {
  std::scoped_lock lifecycle_lock(lifecycle_mutex_);
  SetState(SessionState::kFailed, reason);
  stop_requested_.store(true);
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
    return;
  }
  if (sampler_pid_ > 0) {
    Finish(reason);
  }
}

SessionState NvidiaProcessMetricsWorker::state() const {
  std::scoped_lock lock(mutex_);
  return state_;
}

std::string NvidiaProcessMetricsWorker::reason() const {
  std::scoped_lock lock(mutex_);
  return reason_;
}

std::filesystem::path NvidiaProcessMetricsWorker::output_dir() const {
  return output_dir_;
}

bool NvidiaProcessMetricsWorker::StartSampler(std::string* error) {
  if (!std::filesystem::is_regular_file(sampler_path_)) {
    *error = "PM sampler not found: " + sampler_path_.string();
    return false;
  }

  const int output_directory_fd =
      open(output_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (output_directory_fd < 0) {
    *error = std::string("cannot open sampler output directory: ") +
             std::strerror(errno);
    return false;
  }

  const int log_fd =
      openat(output_directory_fd, "pm_sampling.log",
             O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (log_fd < 0) {
    *error = std::string("cannot create PM sampler log: ") +
             std::strerror(errno);
    close(output_directory_fd);
    return false;
  }

  const int samples_fd =
      openat(output_directory_fd, "pm_sampling.csv",
             O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (samples_fd < 0) {
    *error = std::string("cannot create PM sampler output: ") +
             std::strerror(errno);
    close(log_fd);
    close(output_directory_fd);
    return false;
  }
  // This duplicate intentionally remains open across exec. The sampler writes
  // through /proc/self/fd instead of reopening a user-controlled pathname.
  // Serialize this non-CLOEXEC descriptor's lifetime in the parent so another
  // concurrently started sampler cannot inherit it.
  std::scoped_lock fork_lock(sampler_fork_mutex);
  const int inherited_samples_fd = fcntl(samples_fd, F_DUPFD, 64);
  close(samples_fd);
  if (inherited_samples_fd < 0) {
    *error = std::string("cannot retain PM sampler output: ") +
             std::strerror(errno);
    close(log_fd);
    close(output_directory_fd);
    return false;
  }

  const std::string device = std::to_string(device_);
  const std::string duration = std::to_string(max_duration_.count());
  const std::string sampling_interval =
      std::to_string(kSamplerSamplingInterval);
  const std::string samples_path =
      "/proc/self/fd/" + std::to_string(inherited_samples_fd);
  const char* sampler_path = sampler_path_.c_str();

  sampler_pid_ = fork();
  if (sampler_pid_ < 0) {
    *error = std::string("cannot fork PM sampler: ") + std::strerror(errno);
    close(inherited_samples_fd);
    close(log_fd);
    close(output_directory_fd);
    return false;
  }
  if (sampler_pid_ == 0) {
    if (fchdir(output_directory_fd) != 0 ||
        dup2(log_fd, STDOUT_FILENO) < 0 ||
        dup2(log_fd, STDERR_FILENO) < 0) {
      _exit(127);
    }
    if (log_fd != STDOUT_FILENO && log_fd != STDERR_FILENO) {
      close(log_fd);
    }
    close(output_directory_fd);

    execl(sampler_path, sampler_path, "--device", device.c_str(), "--duration",
          duration.c_str(), "--samplingInterval", sampling_interval.c_str(),
          "--maxsamples", "100000", "--output", samples_path.c_str(),
          static_cast<char*>(nullptr));
    _exit(127);
  }
  close(inherited_samples_fd);
  close(log_fd);
  close(output_directory_fd);
  return true;
}

bool NvidiaProcessMetricsWorker::WaitForSamplerReady(std::string* error) {
  const std::filesystem::path log_path = output_dir_ / "pm_sampling.log";
  const auto deadline = std::chrono::steady_clock::now() + kSamplerReadyTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream log(log_path);
    std::string line;
    while (std::getline(log, line)) {
      if (line.find("PM Sampling active") != std::string::npos) {
        return true;
      }
    }
    int status = 0;
    const pid_t waited = waitpid(sampler_pid_, &status, WNOHANG);
    if (waited == sampler_pid_) {
      sampler_pid_ = -1;
      *error = "PM sampler exited before becoming ready: " +
               DescribeChildStatus(status);
      return false;
    }
    if (waited < 0 && errno != EINTR) {
      sampler_pid_ = -1;
      *error = std::string("cannot monitor PM sampler startup: ") +
               std::strerror(errno);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  *error = "timed out waiting for PM sampler";
  return false;
}

bool NvidiaProcessMetricsWorker::PollSamplerExit(std::string* error) {
  if (sampler_pid_ <= 0) {
    return false;
  }

  int status = 0;
  const pid_t waited = waitpid(sampler_pid_, &status, WNOHANG);
  if (waited == 0 || (waited < 0 && errno == EINTR)) {
    return false;
  }
  if (waited == sampler_pid_) {
    sampler_pid_ = -1;
    *error = "PM sampler exited before session completion: " +
             DescribeChildStatus(status);
    return true;
  }

  sampler_pid_ = -1;
  *error = std::string("cannot monitor PM sampler: ") + std::strerror(errno);
  return true;
}

bool NvidiaProcessMetricsWorker::TransferOutputDirectoryOwnership(
    std::string* error) const {
  const int directory_fd =
      open(output_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_fd < 0) {
    *error = std::string("cannot open output directory for ownership transfer: ") +
             std::strerror(errno);
    return false;
  }

  if (fchmod(directory_fd, 0700) != 0 ||
      fchown(directory_fd, owner_uid_, owner_gid_) != 0) {
    *error = std::string("cannot transfer output directory ownership: ") +
             std::strerror(errno);
    close(directory_fd);
    return false;
  }
  close(directory_fd);
  return true;
}

void NvidiaProcessMetricsWorker::Monitor() {
  const auto deadline = std::chrono::steady_clock::now() + max_duration_;
  while (!stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
#if NVIDIA_PROCESS_METRICS_ENABLE_PM_SAMPLING
    std::string sampler_error;
    if (PollSamplerExit(&sampler_error)) {
      SetState(SessionState::kFailed, sampler_error);
      Finish(sampler_error);
      return;
    }
#endif
    if (!ProcessExists(target_pid_)) {
      Finish("process_exited");
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  Finish(stop_requested_.load() ? "stopped" : "duration_elapsed");
}

void NvidiaProcessMetricsWorker::Finish(const std::string& reason) {
  const pid_t sampler_pid = sampler_pid_;
  sampler_pid_ = -1;
  SamplerStopResult sampler_result{true, {}};
  if (sampler_pid > 0) {
    sampler_result = StopSamplerProcess(sampler_pid);
  }
  MakeResultsReadable();
  std::string ownership_error;
  if (!TransferOutputDirectoryOwnership(&ownership_error)) {
    std::fprintf(stderr, "%s\n", ownership_error.c_str());
  }
  if (state() != SessionState::kFailed) {
    if (sampler_result.success) {
      SetState(SessionState::kCompleted, reason);
    } else {
      SetState(SessionState::kFailed, sampler_result.error);
    }
  }
}

void NvidiaProcessMetricsWorker::SetState(SessionState state,
                                           const std::string& reason) {
  std::scoped_lock lock(mutex_);
  state_ = state;
  reason_ = reason;
}

void NvidiaProcessMetricsWorker::MakeResultsReadable() const {
  const int directory_fd =
      open(output_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_fd < 0) {
    std::fprintf(stderr, "cannot open result directory %s: %s\n",
                 output_dir_.c_str(), std::strerror(errno));
    return;
  }

  DIR* directory = fdopendir(directory_fd);
  if (directory == nullptr) {
    std::fprintf(stderr, "cannot enumerate result directory %s: %s\n",
                 output_dir_.c_str(), std::strerror(errno));
    close(directory_fd);
    return;
  }

  while (true) {
    errno = 0;
    dirent* entry = readdir(directory);
    if (entry == nullptr) {
      if (errno != 0) {
        std::fprintf(stderr, "cannot enumerate result directory %s: %s\n",
                     output_dir_.c_str(), std::strerror(errno));
      }
      break;
    }

    const std::string filename = entry->d_name;
    if (filename == "." || filename == "..") {
      continue;
    }

    const int file_fd = openat(dirfd(directory), filename.c_str(),
                               O_PATH | O_CLOEXEC | O_NOFOLLOW);
    if (file_fd < 0) {
      if (errno != ELOOP) {
        std::fprintf(stderr, "cannot open result file %s/%s: %s\n",
                     output_dir_.c_str(), filename.c_str(),
                     std::strerror(errno));
      }
      continue;
    }

    struct stat file_status {};
    if (fstat(file_fd, &file_status) != 0 || !S_ISREG(file_status.st_mode)) {
      close(file_fd);
      continue;
    }

    if (fchownat(file_fd, "", owner_uid_, owner_gid_, AT_EMPTY_PATH) != 0) {
      std::fprintf(stderr, "cannot set result ownership for %s/%s: %s\n",
                   output_dir_.c_str(), filename.c_str(),
                   std::strerror(errno));
    }
    close(file_fd);
  }
  closedir(directory);
}

}  // namespace nvidia_process_metrics
