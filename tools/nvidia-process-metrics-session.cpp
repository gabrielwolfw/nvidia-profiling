#include "nvidia-process-metrics-session.hpp"

#include <unistd.h>
#include <sys/stat.h>

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

namespace nvidia_process_metrics {

NvidiaProcessMetricsSessionManager::NvidiaProcessMetricsSessionManager(
    std::filesystem::path results_root, std::filesystem::path sampler_path)
    : results_root_(std::move(results_root)), sampler_path_(std::move(sampler_path)) {}

bool NvidiaProcessMetricsSessionManager::Create(
    uid_t uid, gid_t gid, int device, std::chrono::seconds max_duration,
    int window_ms, std::string* session_id, std::filesystem::path* output_dir,
    std::string* error) {
  if (device < 0 || max_duration.count() <= 0 || window_ms <= 0) {
    *error = "device, duration, and window must be greater than zero";
    return false;
  }
  std::scoped_lock lock(mutex_);
  *session_id = NewSessionId();
  *output_dir = results_root_ / std::to_string(uid) / *session_id;
  std::error_code filesystem_error;
  std::filesystem::create_directories(*output_dir, filesystem_error);
  if (filesystem_error || chown(output_dir->c_str(), uid, gid) != 0 ||
      chmod(output_dir->c_str(), 0700) != 0) {
    *error = "cannot create session output directory";
    return false;
  }
  sessions_.emplace(*session_id,
                    Session{uid, gid, device, window_ms, max_duration,
                            *output_dir, nullptr});
  return true;
}

bool NvidiaProcessMetricsSessionManager::Start(const std::string& session_id,
                                                pid_t pid,
                                                uid_t requester_uid,
                                                std::string* error) {
  std::shared_ptr<NvidiaProcessMetricsWorker> worker;
  {
    std::scoped_lock lock(mutex_);
    Session* session = nullptr;
    if (!GetOwnedSession(session_id, requester_uid, &session, error)) {
      return false;
    }
    if (session->worker) {
      *error = "session has already started";
      return false;
    }
    for (const auto& [other_id, other] : sessions_) {
      if (other_id == session_id || other.device != session->device ||
          !other.worker) {
        continue;
      }
      const SessionState other_state = other.worker->state();
      if (other_state == SessionState::kStarting ||
          other_state == SessionState::kRunning) {
        *error = "PM Sampling is already active on this GPU";
        return false;
      }
    }
    worker = std::make_shared<NvidiaProcessMetricsWorker>(
        session_id, pid, session->uid, session->gid, session->device,
        session->max_duration, session->output_dir, sampler_path_);
    session->worker = worker;
  }
  if (!worker->Start(error)) {
    return false;
  }
  return true;
}

bool NvidiaProcessMetricsSessionManager::Stop(const std::string& session_id,
                                               uid_t requester_uid,
                                               std::string* error) {
  std::shared_ptr<NvidiaProcessMetricsWorker> worker;
  {
    std::scoped_lock lock(mutex_);
    Session* session = nullptr;
    if (!GetOwnedSession(session_id, requester_uid, &session, error)) {
      return false;
    }
    if (!session->worker) {
      *error = "session has not started";
      return false;
    }
    worker = session->worker;
  }
  worker->Stop();
  return true;
}

bool NvidiaProcessMetricsSessionManager::Get(
    const std::string& session_id, uid_t requester_uid, SessionState* state,
    std::string* reason, std::filesystem::path* output_dir,
    std::string* error) const {
  std::scoped_lock lock(mutex_);
  const Session* session = nullptr;
  if (!GetOwnedSession(session_id, requester_uid, &session, error)) {
    return false;
  }
  *output_dir = session->output_dir;
  if (!session->worker) {
    *state = SessionState::kCreated;
    *reason = "created";
  } else {
    *state = session->worker->state();
    *reason = session->worker->reason();
  }
  return true;
}

std::string NvidiaProcessMetricsSessionManager::NewSessionId() const {
  const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  std::random_device device;
  std::ostringstream stream;
  stream << std::hex << now << '-' << getpid() << '-' << device();
  return stream.str();
}

bool NvidiaProcessMetricsSessionManager::GetOwnedSession(
    const std::string& session_id, uid_t requester_uid, Session** session,
    std::string* error) {
  auto found = sessions_.find(session_id);
  if (found == sessions_.end()) {
    *error = "session not found";
    return false;
  }
  if (found->second.uid != requester_uid) {
    *error = "session belongs to a different user";
    return false;
  }
  *session = &found->second;
  return true;
}

bool NvidiaProcessMetricsSessionManager::GetOwnedSession(
    const std::string& session_id, uid_t requester_uid, const Session** session,
    std::string* error) const {
  auto found = sessions_.find(session_id);
  if (found == sessions_.end()) {
    *error = "session not found";
    return false;
  }
  if (found->second.uid != requester_uid) {
    *error = "session belongs to a different user";
    return false;
  }
  *session = &found->second;
  return true;
}

}  // namespace nvidia_process_metrics
