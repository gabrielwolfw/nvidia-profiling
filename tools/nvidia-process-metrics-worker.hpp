#pragma once

#include <sys/types.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

namespace nvidia_process_metrics {

enum class SessionState { kCreated, kStarting, kRunning, kCompleted, kFailed };

class NvidiaProcessMetricsWorker {
 public:
  NvidiaProcessMetricsWorker(std::string session_id, pid_t pid, uid_t owner_uid,
                             gid_t owner_gid, int device,
                             std::chrono::seconds max_duration,
                             std::filesystem::path output_dir,
                             std::filesystem::path sampler_path);
  ~NvidiaProcessMetricsWorker();

  NvidiaProcessMetricsWorker(const NvidiaProcessMetricsWorker&) = delete;
  NvidiaProcessMetricsWorker& operator=(const NvidiaProcessMetricsWorker&) = delete;

  bool Start(std::string* error);
  void Stop();
  SessionState state() const;
  std::string reason() const;
  std::filesystem::path output_dir() const;

 private:
  bool StartSampler(std::string* error);
  bool WaitForSamplerReady(std::string* error) const;
  void Monitor();
  void Finish(const std::string& reason);
  void SetState(SessionState state, const std::string& reason);
  void MakeResultsReadable() const;

  std::string session_id_;
  pid_t target_pid_;
  uid_t owner_uid_;
  gid_t owner_gid_;
  int device_;
  std::chrono::seconds max_duration_;
  std::filesystem::path output_dir_;
  std::filesystem::path sampler_path_;
  pid_t sampler_pid_ = -1;
  std::atomic<bool> stop_requested_{false};
  std::thread monitor_thread_;
  mutable std::mutex mutex_;
  SessionState state_ = SessionState::kCreated;
  std::string reason_;
};

const char* SessionStateName(SessionState state);

}  // namespace nvidia_process_metrics
