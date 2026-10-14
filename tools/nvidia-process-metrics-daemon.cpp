#include "nvidia-process-metrics-ipc.hpp"
#include "nvidia-process-metrics-session.hpp"
#include "nvidia-process-metrics-config.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using nvidia_process_metrics::NvidiaProcessMetricsSessionManager;
using nvidia_process_metrics::SessionState;
using nvidia_process_metrics::SessionStateName;

constexpr size_t kMaxConcurrentClients = 32;
constexpr time_t kClientTimeoutSeconds = 10;
volatile std::sig_atomic_t shutdown_requested = 0;
volatile std::sig_atomic_t shutdown_notification_fd = -1;
std::mutex active_clients_mutex;
std::condition_variable active_clients_drained;
size_t active_clients = 0;

void RequestShutdown(int) {
  shutdown_requested = 1;
  if (shutdown_notification_fd < 0) {
    return;
  }
  const int saved_errno = errno;
  const char notification = 1;
  const ssize_t ignored =
      write(shutdown_notification_fd, &notification, sizeof(notification));
  static_cast<void>(ignored);
  errno = saved_errno;
}

void InstallSignalHandlers(int notification_fd) {
  shutdown_notification_fd = notification_fd;
  struct sigaction action {};
  action.sa_handler = RequestShutdown;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGINT, &action, nullptr) != 0 ||
      sigaction(SIGTERM, &action, nullptr) != 0) {
    throw std::runtime_error(std::string("cannot install signal handlers: ") +
                             std::strerror(errno));
  }
}

struct DaemonOptions {
  std::string socket_path = nvidia_process_metrics::kDefaultSocketPath;
  std::filesystem::path results_root = "/tmp/nvidia-process-metrics";
  std::filesystem::path sampler_path =
      NVIDIA_PROCESS_METRICS_SAMPLER_PATH;
};

DaemonOptions ParseArgs(int argc, char** argv) {
  DaemonOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--socket" && index + 1 < argc) {
      options.socket_path = argv[++index];
    } else if (argument == "--results-root" && index + 1 < argc) {
      options.results_root = argv[++index];
    } else if (argument == "--sampler" && index + 1 < argc) {
      options.sampler_path = argv[++index];
    } else if (argument == "--help") {
      std::cout << "Usage: nvidia-process-metrics-daemon [--socket PATH] "
                   "[--results-root PATH] [--sampler PATH]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("invalid daemon option: " + argument);
    }
  }
  return options;
}

int CreateListener(const std::string& socket_path) {
  std::filesystem::create_directories(
      std::filesystem::path(socket_path).parent_path());
  unlink(socket_path.c_str());

  const int listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (listener < 0) {
    throw std::runtime_error(std::strerror(errno));
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(address.sun_path)) {
    close(listener);
    throw std::runtime_error("socket path is too long");
  }
  std::strncpy(address.sun_path, socket_path.c_str(),
               sizeof(address.sun_path) - 1);
  if (bind(listener, reinterpret_cast<const sockaddr*>(&address),
           sizeof(address)) != 0 ||
      listen(listener, 32) != 0) {
    const std::string error = std::strerror(errno);
    close(listener);
    throw std::runtime_error("cannot create daemon socket: " + error);
  }
  // Peer credentials identify the session owner. The protocol does not accept
  // arbitrary commands or output paths, so users can only create their own
  // profiling sessions.
  chmod(socket_path.c_str(), 0666);
  return listener;
}

bool GetPeerCredentials(int client, uid_t* uid, gid_t* gid) {
  ucred credentials{};
  socklen_t length = sizeof(credentials);
  if (getsockopt(client, SOL_SOCKET, SO_PEERCRED, &credentials, &length) != 0) {
    return false;
  }
  *uid = credentials.uid;
  *gid = credentials.gid;
  return true;
}

bool ConfigureClientTimeouts(int client) {
  const timeval timeout{kClientTimeoutSeconds, 0};
  return setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                    sizeof(timeout)) == 0 &&
         setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                    sizeof(timeout)) == 0;
}

bool AcquireClientSlot() {
  std::scoped_lock lock(active_clients_mutex);
  if (active_clients >= kMaxConcurrentClients) {
    return false;
  }
  ++active_clients;
  return true;
}

void ReleaseClientSlot() {
  std::scoped_lock lock(active_clients_mutex);
  --active_clients;
  if (active_clients == 0) {
    active_clients_drained.notify_all();
  }
}

void WaitForClients() {
  std::unique_lock lock(active_clients_mutex);
  active_clients_drained.wait(lock, [] { return active_clients == 0; });
}

std::string Error(const std::string& message) { return "ERR " + message; }

std::string HandleRequest(NvidiaProcessMetricsSessionManager& manager,
                          uid_t uid, gid_t gid, const std::string& request) {
  const std::vector<std::string> fields =
      nvidia_process_metrics::SplitFields(request);
  if (fields.empty()) {
    return Error("empty_request");
  }

  try {
    std::string error;
    if (fields[0] == "CREATE" && fields.size() == 4) {
      std::string session_id;
      std::filesystem::path output_dir;
      if (!manager.Create(uid, gid, std::stoi(fields[1]),
                          std::chrono::seconds(std::stoll(fields[2])),
                          std::stoi(fields[3]), &session_id, &output_dir,
                          &error)) {
        return Error(error);
      }
      return "OK " + session_id + " " + output_dir.string();
    }
    if (fields[0] == "START" && fields.size() == 3) {
      if (!manager.Start(fields[1], static_cast<pid_t>(std::stol(fields[2])),
                         uid, &error)) {
        return Error(error);
      }
      return "OK ready";
    }
    if (fields[0] == "STOP" && fields.size() == 2) {
      if (!manager.Stop(fields[1], uid, &error)) {
        return Error(error);
      }
      return "OK stopped";
    }
    if (fields[0] == "FAIL" && fields.size() == 3) {
      if (!manager.Fail(fields[1], uid, fields[2], &error)) {
        return Error(error);
      }
      return "OK failed";
    }
    if (fields[0] == "STATUS" && fields.size() == 2) {
      SessionState state;
      std::string reason;
      std::filesystem::path output_dir;
      if (!manager.Get(fields[1], uid, &state, &reason, &output_dir, &error)) {
        return Error(error);
      }
      return "OK " + std::string(SessionStateName(state)) + " " + reason +
             " " + output_dir.string();
    }
  } catch (const std::exception& exception) {
    return Error(std::string("invalid_request_") + exception.what());
  }
  return Error("unsupported_request");
}

void ServeClient(int client, NvidiaProcessMetricsSessionManager& manager) {
  uid_t uid = 0;
  gid_t gid = 0;
  if (!GetPeerCredentials(client, &uid, &gid)) {
    send(client, "ERR peer_credentials", 20, MSG_NOSIGNAL);
    close(client);
    return;
  }
  char buffer[4096]{};
  const ssize_t received = recv(client, buffer, sizeof(buffer) - 1, 0);
  if (received > 0) {
    const std::string reply =
        HandleRequest(manager, uid, gid, std::string(buffer, received));
    send(client, reply.data(), reply.size(), MSG_NOSIGNAL);
  }
  close(client);
}

}  // namespace

int main(int argc, char** argv) {
  try {
#if NVIDIA_PROCESS_METRICS_ENABLE_PM_SAMPLING
    if (geteuid() != 0) {
      std::cerr << "nvidia-process-metrics-daemon must run as root\n";
      return 1;
    }
#endif
    const DaemonOptions options = ParseArgs(argc, argv);
    int shutdown_pipe[2]{};
    if (pipe2(shutdown_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
      throw std::runtime_error(std::string("cannot create shutdown pipe: ") +
                               std::strerror(errno));
    }
    InstallSignalHandlers(shutdown_pipe[1]);
    const int listener = CreateListener(options.socket_path);
    NvidiaProcessMetricsSessionManager manager(options.results_root,
                                                options.sampler_path);
    std::cout << "NVIDIA Process Metrics daemon listening on "
              << options.socket_path << '\n';
    while (!shutdown_requested) {
      pollfd descriptors[2] = {
          {shutdown_pipe[0], POLLIN, 0},
          {listener, POLLIN, 0},
      };
      const int poll_result = poll(descriptors, 2, -1);
      if (poll_result < 0) {
        if (errno == EINTR) {
          continue;
        }
        std::cerr << "poll failed: " << std::strerror(errno) << '\n';
        break;
      }
      if (shutdown_requested || (descriptors[0].revents & POLLIN) != 0) {
        break;
      }
      if ((descriptors[1].revents & POLLIN) == 0) {
        continue;
      }
      const int client = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
      if (client < 0) {
        if (errno == EINTR) {
          continue;
        }
        std::cerr << "accept failed: " << std::strerror(errno) << '\n';
        continue;
      }
      if (!ConfigureClientTimeouts(client)) {
        std::cerr << "cannot configure client socket: " << std::strerror(errno)
                  << '\n';
        close(client);
        continue;
      }
      if (!AcquireClientSlot()) {
        constexpr char kBusyReply[] = "ERR server_busy";
        send(client, kBusyReply, sizeof(kBusyReply) - 1, MSG_NOSIGNAL);
        close(client);
        continue;
      }
      try {
        std::thread([client, &manager]() {
          try {
            ServeClient(client, manager);
          } catch (const std::exception& exception) {
            std::cerr << "client request failed: " << exception.what() << '\n';
            close(client);
          } catch (...) {
            std::cerr << "client request failed with an unknown error\n";
            close(client);
          }
          ReleaseClientSlot();
        }).detach();
      } catch (const std::exception& exception) {
        ReleaseClientSlot();
        close(client);
        std::cerr << "cannot create client thread: " << exception.what() << '\n';
      }
    }
    shutdown_notification_fd = -1;
    close(shutdown_pipe[0]);
    close(shutdown_pipe[1]);
    close(listener);
    unlink(options.socket_path.c_str());
    std::cout << "NVIDIA Process Metrics daemon shutting down\n";
    WaitForClients();
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "nvidia-process-metrics-daemon: " << exception.what() << '\n';
    return 1;
  }
}
