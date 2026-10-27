/**
 * @file session.cpp
 * @brief Daemon-side manager for GPU profiling sessions
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include "gpu/session.hpp"  // NOLINT

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>

namespace nvidia_process_metrics {
namespace {

constexpr size_t kMaxSessionsPerUser = 64;
constexpr size_t kMaxSessionsGlobal = 1024;
constexpr auto kUnstartedSessionTimeout = std::chrono::seconds(60);

class ScopedFd {
 public:
  explicit ScopedFd(int fd = -1) : fd_(fd) {}
  ~ScopedFd() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  int get() const { return fd_; }

  void Reset(int fd) {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_;
};

bool IsTrustedDirectory(int directory_fd, std::string* error) {
  struct stat status{};
  if (fstat(directory_fd, &status) != 0) {
    *error =
        std::string("cannot inspect results path: ") + std::strerror(errno);
    return false;
  }

  const uid_t effective_uid = geteuid();
  if (status.st_uid != 0 && status.st_uid != effective_uid) {
    *error = "results path contains a directory owned by another user";
    return false;
  }

  const bool writable_by_others = (status.st_mode & (S_IWGRP | S_IWOTH)) != 0;
  const bool trusted_sticky_directory =
      status.st_uid == 0 && (status.st_mode & S_ISVTX) != 0;
  if (writable_by_others && !trusted_sticky_directory) {
    *error = "results path contains an unsafe writable directory";
    return false;
  }
  return true;
}

bool IsPrivateDaemonDirectory(int directory_fd, std::string* error) {
  struct stat status{};
  if (fstat(directory_fd, &status) != 0) {
    *error = std::string("cannot inspect results directory: ") +
             std::strerror(errno);
    return false;
  }
  if (status.st_uid != geteuid() ||
      (status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    *error = "results directory has unsafe ownership or permissions";
    return false;
  }
  return true;
}

int OpenOrCreateDirectoryAt(int parent_fd, const std::string& name, mode_t mode,
                            std::string* error) {
  int directory_fd = openat(parent_fd, name.c_str(),
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_fd < 0 && errno == ENOENT) {
    if (mkdirat(parent_fd, name.c_str(), mode) != 0 && errno != EEXIST) {
      *error = std::string("cannot create results directory: ") +
               std::strerror(errno);
      return -1;
    }
    directory_fd = openat(parent_fd, name.c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  }
  if (directory_fd < 0) {
    *error =
        std::string("cannot open results directory: ") + std::strerror(errno);
    return -1;
  }
  return directory_fd;
}

int OpenSecureResultsRoot(const std::filesystem::path& results_root,
                          std::string* error) {
  if (results_root.empty()) {
    *error = "results root cannot be empty";
    return -1;
  }

  ScopedFd current(open(results_root.is_absolute() ? "/" : ".",
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (current.get() < 0) {
    *error =
        std::string("cannot open results path root: ") + std::strerror(errno);
    return -1;
  }
  if (!IsTrustedDirectory(current.get(), error)) {
    return -1;
  }

  for (const auto& component : results_root) {
    const std::string name = component.string();
    if (name.empty() || name == "/" || name == ".") {
      continue;
    }
    if (name == "..") {
      *error = "results root cannot contain '..'";
      return -1;
    }

    const int next = OpenOrCreateDirectoryAt(current.get(), name, 0755, error);
    if (next < 0) {
      return -1;
    }
    current.Reset(next);
    if (!IsTrustedDirectory(current.get(), error)) {
      return -1;
    }
  }

  const int result_fd = fcntl(current.get(), F_DUPFD_CLOEXEC, 0);
  if (result_fd < 0) {
    *error =
        std::string("cannot retain results directory: ") + std::strerror(errno);
  }
  return result_fd;
}

bool CreateSessionDirectory(const std::filesystem::path& results_root,
                            uid_t uid, const std::string& session_id,
                            std::filesystem::path* output_dir,
                            std::string* error) {
  ScopedFd results_fd(OpenSecureResultsRoot(results_root, error));
  if (results_fd.get() < 0) {
    return false;
  }

  const std::string uid_directory = std::to_string(uid);
  ScopedFd uid_fd(
      OpenOrCreateDirectoryAt(results_fd.get(), uid_directory, 0755, error));
  if (uid_fd.get() < 0 || !IsPrivateDaemonDirectory(uid_fd.get(), error)) {
    return false;
  }

  if (mkdirat(uid_fd.get(), session_id.c_str(), 0700) != 0) {
    *error = std::string("cannot create session output directory: ") +
             std::strerror(errno);
    return false;
  }

  ScopedFd session_fd(openat(uid_fd.get(), session_id.c_str(),
                             O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (session_fd.get() < 0 || fchmod(session_fd.get(), 0700) != 0) {
    *error = std::string("cannot initialize session output directory: ") +
             std::strerror(errno);
    unlinkat(uid_fd.get(), session_id.c_str(), AT_REMOVEDIR);
    return false;
  }

  *output_dir = results_root / uid_directory / session_id;
  return true;
}

}  // namespace

NvidiaProcessMetricsSessionManager::NvidiaProcessMetricsSessionManager(
    std::filesystem::path results_root, std::filesystem::path sampler_path)
    : results_root_(std::move(results_root)),
      sampler_path_(std::move(sampler_path)) {}

bool NvidiaProcessMetricsSessionManager::Create(
    uid_t uid, gid_t gid, int device, std::chrono::seconds max_duration,
    int window_ms, std::string* session_id, std::filesystem::path* output_dir,
    std::string* error) {
  if (device < 0 || max_duration.count() <= 0 || window_ms <= 0) {
    *error = "device, duration, and window must be greater than zero";
    return false;
  }
  std::scoped_lock lock(mutex_);
  if (!MakeCapacityForSession(uid, error)) {
    return false;
  }
  *session_id = NewSessionId();
  if (!CreateSessionDirectory(results_root_, uid, *session_id, output_dir,
                              error)) {
    return false;
  }
  sessions_.emplace(
      *session_id,
      Session{uid, gid, device, window_ms, max_duration, *output_dir, nullptr,
              std::chrono::steady_clock::now(), next_sequence_++});
  return true;
}

bool NvidiaProcessMetricsSessionManager::MakeCapacityForSession(
    uid_t uid, std::string* error) {
  size_t user_sessions = 0;
  for (const auto& [session_id, session] : sessions_) {
    (void)session_id;
    if (session.uid == uid) {
      ++user_sessions;
    }
  }

  while (user_sessions >= kMaxSessionsPerUser) {
    if (!RemoveOldestEvictableSession(uid, true)) {
      *error = "per-user session limit reached";
      return false;
    }
    --user_sessions;
  }

  while (sessions_.size() >= kMaxSessionsGlobal) {
    if (!RemoveOldestEvictableSession(0, false)) {
      *error = "global session limit reached";
      return false;
    }
  }
  return true;
}

bool NvidiaProcessMetricsSessionManager::RemoveOldestEvictableSession(
    uid_t uid, bool filter_by_uid) {
  auto oldest_abandoned = sessions_.end();
  auto oldest_terminal = sessions_.end();
  const auto now = std::chrono::steady_clock::now();

  for (auto session = sessions_.begin(); session != sessions_.end();
       ++session) {
    if (filter_by_uid && session->second.uid != uid) {
      continue;
    }

    if (!session->second.worker) {
      if (now - session->second.created_at >= kUnstartedSessionTimeout &&
          (oldest_abandoned == sessions_.end() ||
           session->second.sequence < oldest_abandoned->second.sequence)) {
        oldest_abandoned = session;
      }
      continue;
    }

    const SessionState state = session->second.worker->state();
    if (state != SessionState::kCompleted && state != SessionState::kFailed) {
      continue;
    }
    if (oldest_terminal == sessions_.end() ||
        session->second.sequence < oldest_terminal->second.sequence) {
      oldest_terminal = session;
    }
  }

  if (oldest_abandoned != sessions_.end()) {
    std::error_code filesystem_error;
    std::filesystem::remove(oldest_abandoned->second.output_dir,
                            filesystem_error);
    sessions_.erase(oldest_abandoned);
    return true;
  }

  if (oldest_terminal != sessions_.end()) {
    sessions_.erase(oldest_terminal);
    return true;
  }
  return false;
}

bool NvidiaProcessMetricsSessionManager::Start(const std::string& session_id,
                                               pid_t pid, uid_t requester_uid,
                                               std::string* error) {
  std::shared_ptr<NvidiaProcessMetricsWorker> worker;
  int reserved_device = -1;
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
    if (reserved_devices_.count(session->device) != 0) {
      *error = "PM Sampling is already active on this GPU";
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
    reserved_device = session->device;
    reserved_devices_.insert(reserved_device);
    try {
      worker = std::make_shared<NvidiaProcessMetricsWorker>(
          session_id, pid, session->uid, session->gid, session->device,
          session->max_duration, session->output_dir, sampler_path_);
    } catch (...) {
      reserved_devices_.erase(reserved_device);
      throw;
    }
    session->worker = worker;
  }

  bool started = false;
  try {
    started = worker->Start(error);
  } catch (...) {
    std::scoped_lock lock(mutex_);
    reserved_devices_.erase(reserved_device);
    throw;
  }
  {
    std::scoped_lock lock(mutex_);
    reserved_devices_.erase(reserved_device);
  }
  return started;
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

bool NvidiaProcessMetricsSessionManager::Fail(const std::string& session_id,
                                              uid_t requester_uid,
                                              const std::string& reason,
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
  worker->Fail(reason);
  return true;
}

bool NvidiaProcessMetricsSessionManager::Get(const std::string& session_id,
                                             uid_t requester_uid,
                                             SessionState* state,
                                             std::string* reason,
                                             std::filesystem::path* output_dir,
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
