/**
 * @file ipc.hpp
 * @brief Unix-socket protocol shared by the daemon and the launcher
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef SRC_TOOLS_GPU_IPC_HPP_
#define SRC_TOOLS_GPU_IPC_HPP_

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvidia_process_metrics {

constexpr const char* kDefaultSocketPath =
    "/run/nvidia-process-metrics/daemon.sock";  // NOLINT

inline std::vector<std::string> SplitFields(const std::string& message) {
  std::vector<std::string> fields;
  std::string current;
  for (char character : message) {
    if (character == ' ' || character == '\n' || character == '\t') {
      if (!current.empty()) {
        fields.push_back(current);
        current.clear();
      }
    } else {
      current += character;
    }
  }
  if (!current.empty()) {
    fields.push_back(current);
  }
  return fields;
}

inline int ConnectToDaemon(const std::string& socket_path) {
  const int socket_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (socket_fd < 0) {
    throw std::runtime_error(std::strerror(errno));
  }

  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(address.sun_path)) {
    close(socket_fd);
    throw std::runtime_error("daemon socket path is too long");
  }
  std::strncpy(address.sun_path, socket_path.c_str(),
               sizeof(address.sun_path) - 1);

  int connected = -1;
  do {
    connected = connect(socket_fd, reinterpret_cast<const sockaddr*>(&address),
                        sizeof(address));
  } while (connected != 0 && errno == EINTR);
  if (connected != 0) {
    const std::string error = std::strerror(errno);
    close(socket_fd);
    throw std::runtime_error("cannot connect to daemon: " + error);
  }
  return socket_fd;
}

inline std::string Request(const std::string& socket_path,
                           const std::string& request) {
  const int socket_fd = ConnectToDaemon(socket_path);
  ssize_t sent = -1;
  do {
    sent = send(socket_fd, request.data(), request.size(), MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  if (sent != static_cast<ssize_t>(request.size())) {
    const std::string error = std::strerror(errno);
    close(socket_fd);
    throw std::runtime_error("cannot send daemon request: " + error);
  }

  char buffer[4096]{};
  ssize_t received = -1;
  do {
    received = recv(socket_fd, buffer, sizeof(buffer) - 1, 0);
  } while (received < 0 && errno == EINTR);
  if (received < 0) {
    const std::string error = std::strerror(errno);
    close(socket_fd);
    throw std::runtime_error("cannot receive daemon response: " + error);
  }
  close(socket_fd);
  return std::string(buffer, static_cast<size_t>(received));
}

}  // namespace nvidia_process_metrics

#endif  // SRC_TOOLS_GPU_IPC_HPP_
