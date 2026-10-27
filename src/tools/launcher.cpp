#include "gpu/ipc.hpp"
#include "nvidia-process-metrics-config.hpp"

#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

volatile sig_atomic_t requested_signal = 0;
constexpr char kProfilerCompletionSuccess = '0';
constexpr char kProfilerCompletionFailure = '1';

struct ProfilerCompletion {
  bool success;
  std::string reason;
};

void RequestShutdown(int signal_number) {
  if (requested_signal == 0) {
    requested_signal = signal_number;
  }
}

void InstallSignalHandlers() {
  struct sigaction action {};
  action.sa_handler = RequestShutdown;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGINT, &action, nullptr) != 0 ||
      sigaction(SIGTERM, &action, nullptr) != 0) {
    throw std::runtime_error(std::string("cannot install signal handler: ") +
                             std::strerror(errno));
  }
}

struct LauncherOptions {
  std::string socket_path = nvidia_process_metrics::kDefaultSocketPath;
  std::string library_path =
      NVIDIA_PROCESS_METRICS_LIBRARY_PATH;
  int device = 0;
  int duration_seconds = 10;
  int window_ms = 200;
  pid_t pid = -1;
  bool observe = false;
  std::vector<std::string> command;
};

void PrintHelp() {
  std::cout << "Usage:\n"
            << "  nvidia-process-metrics-launcher --command -- PROGRAM [ARGS...]\n"
            << "  nvidia-process-metrics-launcher --pid PID --observe\n\n"
            << "Options: --device N --duration SECONDS --window-ms MS "
               "--socket PATH --library PATH\n";
}

LauncherOptions ParseArgs(int argc, char** argv) {
  LauncherOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--command") {
      ++index;
      if (index < argc && std::string(argv[index]) == "--") {
        ++index;
      }
      for (; index < argc; ++index) {
        options.command.emplace_back(argv[index]);
      }
      break;
    }
    if (argument == "--pid" && index + 1 < argc) {
      options.pid = static_cast<pid_t>(std::stol(argv[++index]));
    } else if (argument == "--observe") {
      options.observe = true;
    } else if (argument == "--device" && index + 1 < argc) {
      options.device = std::stoi(argv[++index]);
    } else if (argument == "--duration" && index + 1 < argc) {
      options.duration_seconds = std::stoi(argv[++index]);
    } else if (argument == "--window-ms" && index + 1 < argc) {
      options.window_ms = std::stoi(argv[++index]);
    } else if (argument == "--socket" && index + 1 < argc) {
      options.socket_path = argv[++index];
    } else if (argument == "--library" && index + 1 < argc) {
      options.library_path = argv[++index];
    } else if (argument == "--help") {
      PrintHelp();
      std::exit(0);
    } else {
      throw std::runtime_error("invalid launcher option: " + argument);
    }
  }
  if (options.duration_seconds <= 0 || options.window_ms <= 0 ||
      options.device < 0) {
    throw std::runtime_error("device, duration, and window must be positive");
  }
  if (options.command.empty() && (options.pid <= 0 || !options.observe)) {
    throw std::runtime_error("use --command or --pid PID --observe");
  }
  if (!options.command.empty() && options.pid > 0) {
    throw std::runtime_error("--command and --pid are mutually exclusive");
  }
  return options;
}

std::vector<std::string> RequireOk(const std::string& response) {
  const auto fields = nvidia_process_metrics::SplitFields(response);
  if (fields.empty() || fields[0] != "OK") {
    throw std::runtime_error("daemon rejected request: " + response);
  }
  return fields;
}

std::pair<std::string, std::string> CreateSession(const LauncherOptions& options) {
  const std::string response = nvidia_process_metrics::Request(
      options.socket_path, "CREATE " + std::to_string(options.device) + " " +
                               std::to_string(options.duration_seconds) + " " +
                               std::to_string(options.window_ms));
  const auto fields = RequireOk(response);
  if (fields.size() != 3) {
    throw std::runtime_error("invalid CREATE response: " + response);
  }
  return {fields[1], fields[2]};
}

void StartSession(const LauncherOptions& options, const std::string& session_id,
                  pid_t pid) {
  RequireOk(nvidia_process_metrics::Request(
      options.socket_path,
      "START " + session_id + " " + std::to_string(static_cast<long>(pid))));
}

void StopSession(const LauncherOptions& options, const std::string& session_id) {
  RequireOk(nvidia_process_metrics::Request(options.socket_path,
                                             "STOP " + session_id));
}

void FailSession(const LauncherOptions& options, const std::string& session_id,
                 const std::string& reason) {
  RequireOk(nvidia_process_metrics::Request(
      options.socket_path, "FAIL " + session_id + " " + reason));
}

bool StopSessionForCleanup(const LauncherOptions& options,
                           const std::string& session_id) {
  try {
    StopSession(options, session_id);
    return true;
  } catch (const std::exception& exception) {
    std::cerr << "cannot stop interrupted session: " << exception.what()
              << '\n';
    return false;
  }
}

bool FailSessionForCleanup(const LauncherOptions& options,
                           const std::string& session_id,
                           const std::string& reason) {
  try {
    FailSession(options, session_id, reason);
    return true;
  } catch (const std::exception& exception) {
    std::cerr << "cannot fail profiling session: " << exception.what() << '\n';
    return false;
  }
}

ProfilerCompletion ReadProfilerCompletion(int completion_fd) {
  char status = 0;
  ssize_t received = -1;
  do {
    received = read(completion_fd, &status, sizeof(status));
  } while (received < 0 && errno == EINTR);
  const int read_error = errno;
  close(completion_fd);

  if (received == sizeof(status)) {
    if (status == kProfilerCompletionSuccess) {
      return {true, {}};
    }
    if (status == kProfilerCompletionFailure) {
      return {false, "preload_output_failed"};
    }
    return {false, "preload_completion_invalid"};
  }
  if (received == 0 || (received < 0 && read_error == EAGAIN)) {
    return {false, "preload_completion_missing"};
  }
  return {false, "preload_completion_read_failed"};
}

void PrintResults(const LauncherOptions& options, const std::string& session_id) {
  const auto fields = RequireOk(nvidia_process_metrics::Request(
      options.socket_path, "STATUS " + session_id));
  if (fields.size() < 4) {
    throw std::runtime_error("invalid STATUS response");
  }
  const std::filesystem::path output_dir = fields.back();
  std::cout << "Session: " << session_id << '\n'
            << "State: " << fields[1] << " (" << fields[2] << ")\n"
            << "Results: " << output_dir << '\n';
  for (const char* filename : {"gpu_telemetry.csv", "kernel_activity.csv",
                               "pm_sampling.csv", "pm_sampling.log"}) {
    const auto path = output_dir / filename;
    if (std::filesystem::exists(path)) {
      std::cout << filename << ": " << path << '\n';
    }
  }
}

pid_t LaunchWithBarrier(const LauncherOptions& options,
                        const std::string& output_dir, int release_fd,
                        int parent_release_fd, int completion_fd,
                        int parent_completion_fd) {
  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("cannot fork target process");
  }
  if (pid != 0) {
    if (setpgid(pid, pid) != 0) {
      const std::string error = std::strerror(errno);
      kill(pid, SIGKILL);
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
      throw std::runtime_error("cannot create target process group: " + error);
    }
    return pid;
  }

  close(parent_release_fd);
  close(parent_completion_fd);
  if (setpgid(0, 0) != 0) {
    _exit(127);
  }
  struct sigaction default_action {};
  default_action.sa_handler = SIG_DFL;
  sigemptyset(&default_action.sa_mask);
  sigaction(SIGINT, &default_action, nullptr);
  sigaction(SIGTERM, &default_action, nullptr);

  char release = 0;
  ssize_t received = -1;
  do {
    received = read(release_fd, &release, 1);
  } while (received < 0 && errno == EINTR);
  if (received != 1) {
    _exit(127);
  }
  close(release_fd);
  const int completion_flags = fcntl(completion_fd, F_GETFD);
  if (completion_flags < 0 ||
      fcntl(completion_fd, F_SETFD, completion_flags & ~FD_CLOEXEC) != 0) {
    _exit(127);
  }
  setenv("NVIDIA_METRICS_OUTPUT_DIR", output_dir.c_str(), 1);
  setenv("NVIDIA_METRICS_DEVICE", std::to_string(options.device).c_str(), 1);
  setenv("NVIDIA_METRICS_WINDOW_MS",
         std::to_string(options.window_ms).c_str(), 1);
  setenv("NVIDIA_METRICS_COMPLETION_FD",
         std::to_string(completion_fd).c_str(), 1);
  const char* existing_preload = getenv("LD_PRELOAD");
  const std::string preload = existing_preload == nullptr
                                  ? options.library_path
                                  : options.library_path + ":" + existing_preload;
  setenv("LD_PRELOAD", preload.c_str(), 1);

  std::vector<char*> arguments;
  for (const auto& argument : options.command) {
    arguments.push_back(const_cast<char*>(argument.c_str()));
  }
  arguments.push_back(nullptr);
  execvp(arguments.front(), arguments.data());
  _exit(127);
}

int Observe(const LauncherOptions& options) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(options.duration_seconds);
  while (requested_signal == 0 &&
         std::chrono::steady_clock::now() < deadline &&
         (kill(options.pid, 0) == 0 || errno == EPERM)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return requested_signal;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const LauncherOptions options = ParseArgs(argc, argv);
    if (!options.command.empty() &&
        !std::filesystem::is_regular_file(options.library_path)) {
      throw std::runtime_error("metrics library not found: " +
                               options.library_path);
    }
    InstallSignalHandlers();
    const auto [session_id, output_dir] = CreateSession(options);

    if (options.command.empty()) {
      StartSession(options, session_id, options.pid);
      const int signal_number = Observe(options);
      if (signal_number == 0) {
        StopSession(options, session_id);
      } else {
        StopSessionForCleanup(options, session_id);
      }
      try {
        PrintResults(options, session_id);
      } catch (const std::exception& exception) {
        if (requested_signal == 0) {
          throw;
        }
        std::cerr << "cannot read interrupted session results: "
                  << exception.what() << '\n';
      }
      return requested_signal == 0 ? 0 : 128 + requested_signal;
    }

    int barrier[2]{};
    if (pipe2(barrier, O_CLOEXEC) != 0) {
      throw std::runtime_error("cannot create process barrier");
    }
    int completion_pipe[2]{};
    if (pipe2(completion_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
      close(barrier[0]);
      close(barrier[1]);
      throw std::runtime_error("cannot create profiler completion pipe");
    }
    const pid_t pid =
        LaunchWithBarrier(options, output_dir, barrier[0], barrier[1],
                          completion_pipe[1], completion_pipe[0]);
    close(barrier[0]);
    close(completion_pipe[1]);
    try {
      StartSession(options, session_id, pid);
    } catch (...) {
      kill(-pid, SIGKILL);
      close(barrier[1]);
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
      close(completion_pipe[0]);
      StopSessionForCleanup(options, session_id);
      throw;
    }

    std::string release_error;
    if (requested_signal == 0) {
      ssize_t released = -1;
      do {
        released = write(barrier[1], "1", 1);
      } while (released < 0 && errno == EINTR && requested_signal == 0);
      if (released != 1 && requested_signal == 0) {
        release_error = std::strerror(errno);
      }
    }
    close(barrier[1]);

    if (!release_error.empty()) {
      kill(-pid, SIGKILL);
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
      close(completion_pipe[0]);
      StopSessionForCleanup(options, session_id);
      throw std::runtime_error("cannot release target process: " +
                               release_error);
    }

    int status = 0;
    int signal_number = 0;
    bool session_stopped = false;
    while (true) {
      if (requested_signal != 0 && signal_number == 0) {
        signal_number = requested_signal;
        kill(-pid, signal_number);
        session_stopped = StopSessionForCleanup(options, session_id);
      }

      const pid_t waited = waitpid(pid, &status, 0);
      if (waited == pid) {
        break;
      }
      if (waited < 0 && errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("cannot wait for target process: ") +
                               std::strerror(errno));
    }

    const ProfilerCompletion profiler_completion =
        ReadProfilerCompletion(completion_pipe[0]);
    if (!profiler_completion.success) {
      std::cerr << "Profiler failed: " << profiler_completion.reason << '\n';
      if (requested_signal == 0) {
        FailSession(options, session_id, profiler_completion.reason);
      } else {
        FailSessionForCleanup(options, session_id, profiler_completion.reason);
      }
      session_stopped = true;
    } else if (!session_stopped) {
      if (requested_signal == 0) {
        StopSession(options, session_id);
      } else {
        StopSessionForCleanup(options, session_id);
      }
    }
    try {
      PrintResults(options, session_id);
    } catch (const std::exception& exception) {
      if (requested_signal == 0) {
        throw;
      }
      std::cerr << "cannot read interrupted session results: "
                << exception.what() << '\n';
    }
    if (requested_signal != 0) {
      return 128 + requested_signal;
    }
    if (WIFEXITED(status)) {
      const int exit_code = WEXITSTATUS(status);
      std::cout << "Target exit code: " << exit_code << '\n';
      return exit_code == 0 && !profiler_completion.success ? 1 : exit_code;
    }
    if (WIFSIGNALED(status)) {
      const int signal_number = WTERMSIG(status);
      std::cerr << "Target terminated by signal " << signal_number << '\n';
      return 128 + signal_number;
    }
    std::cerr << "Target ended with an unknown wait status\n";
    return 1;
  } catch (const std::exception& exception) {
    std::cerr << "nvidia-process-metrics-launcher: " << exception.what() << '\n';
    return requested_signal == 0 ? 1 : 128 + requested_signal;
  }
}
