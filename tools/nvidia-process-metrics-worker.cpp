#include "nvidia-process-metrics-worker.hpp"

#include <fcntl.h>
#include <signal.h>
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <thread>

namespace nvidia_process_metrics {
namespace {

constexpr auto kSamplerReadyTimeout = std::chrono::seconds(10);

bool ProcessExists(pid_t pid) {
  return kill(pid, 0) == 0 || errno == EPERM;
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
  if (!ProcessExists(target_pid_)) {
    *error = "target PID does not exist";
    SetState(SessionState::kFailed, *error);
    return false;
  }
  SetState(SessionState::kStarting, "starting PM Sampling");
  if (!StartSampler(error)) {
    SetState(SessionState::kFailed, *error);
    return false;
  }
  if (!WaitForSamplerReady(error)) {
    Finish("PM Sampling did not become ready");
    SetState(SessionState::kFailed, *error);
    return false;
  }

  SetState(SessionState::kRunning, "running");
  monitor_thread_ = std::thread(&NvidiaProcessMetricsWorker::Monitor, this);
  return true;
}

void NvidiaProcessMetricsWorker::Stop() {
  stop_requested_.store(true);
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
    return;
  }
  if (sampler_pid_ > 0) {
    Finish("stopped");
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

  sampler_pid_ = fork();
  if (sampler_pid_ < 0) {
    *error = "cannot fork PM sampler";
    return false;
  }
  if (sampler_pid_ == 0) {
    const std::filesystem::path log_path = output_dir_ / "pm_sampling.log";
    const int log_fd = open(log_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC,
                            0644);
    if (log_fd < 0 || chdir(output_dir_.c_str()) != 0) {
      _exit(127);
    }
    dup2(log_fd, STDOUT_FILENO);
    dup2(log_fd, STDERR_FILENO);
    close(log_fd);

    const std::string device = std::to_string(device_);
    const std::string duration = std::to_string(max_duration_.count());
    execl(sampler_path_.c_str(), sampler_path_.c_str(), "--device",
          device.c_str(), "--duration", duration.c_str(),
          "--maxsamples", "100000", static_cast<char*>(nullptr));
    _exit(127);
  }
  return true;
}

bool NvidiaProcessMetricsWorker::WaitForSamplerReady(std::string* error) const {
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
    if (waitpid(sampler_pid_, &status, WNOHANG) == sampler_pid_) {
      *error = "PM sampler exited before becoming ready";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  *error = "timed out waiting for PM sampler";
  return false;
}

void NvidiaProcessMetricsWorker::Monitor() {
  const auto deadline = std::chrono::steady_clock::now() + max_duration_;
  while (!stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
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
  if (sampler_pid > 0) {
    kill(sampler_pid, SIGINT);
    waitpid(sampler_pid, nullptr, 0);
    sampler_pid_ = -1;
  }
  MakeResultsReadable();
  if (state() != SessionState::kFailed) {
    SetState(SessionState::kCompleted, reason);
  }
}

void NvidiaProcessMetricsWorker::SetState(SessionState state,
                                           const std::string& reason) {
  std::scoped_lock lock(mutex_);
  state_ = state;
  reason_ = reason;
}

void NvidiaProcessMetricsWorker::MakeResultsReadable() const {
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(output_dir_, error)) {
    if (error) {
      return;
    }
    chown(entry.path().c_str(), owner_uid_, owner_gid_);
  }
}

}  // namespace nvidia_process_metrics
