#pragma once

#include <sys/types.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "nvidia-process-metrics-worker.hpp"

namespace nvidia_process_metrics {

class NvidiaProcessMetricsSessionManager {
 public:
  NvidiaProcessMetricsSessionManager(std::filesystem::path results_root,
                                     std::filesystem::path sampler_path);

  bool Create(uid_t uid, gid_t gid, int device,
              std::chrono::seconds max_duration, int window_ms,
              std::string* session_id, std::filesystem::path* output_dir,
              std::string* error);
  bool Start(const std::string& session_id, pid_t pid, uid_t requester_uid,
             std::string* error);
  bool Stop(const std::string& session_id, uid_t requester_uid,
            std::string* error);
  bool Get(const std::string& session_id, uid_t requester_uid,
           SessionState* state, std::string* reason,
           std::filesystem::path* output_dir, std::string* error) const;

 private:
  struct Session {
    uid_t uid;
    gid_t gid;
    int device;
    int window_ms;
    std::chrono::seconds max_duration;
    std::filesystem::path output_dir;
    std::shared_ptr<NvidiaProcessMetricsWorker> worker;
  };

  std::string NewSessionId() const;
  bool GetOwnedSession(const std::string& session_id, uid_t requester_uid,
                       Session** session, std::string* error);
  bool GetOwnedSession(const std::string& session_id, uid_t requester_uid,
                       const Session** session, std::string* error) const;

  std::filesystem::path results_root_;
  std::filesystem::path sampler_path_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Session> sessions_;
};

}  // namespace nvidia_process_metrics
