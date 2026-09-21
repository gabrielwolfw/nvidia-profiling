#include "nvidia-process-metrics-ipc.hpp"

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

struct LauncherOptions {
  std::string socket_path = nvidia_process_metrics::kDefaultSocketPath;
  std::string library_path =
      "/usr/local/lib/nvidia-process-metrics/libnvidia-process-metrics.so";
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
                        const std::string& output_dir, int release_fd) {
  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("cannot fork target process");
  }
  if (pid != 0) {
    return pid;
  }

  char release = 0;
  if (read(release_fd, &release, 1) != 1) {
    _exit(127);
  }
  close(release_fd);
  setenv("NVIDIA_METRICS_OUTPUT_DIR", output_dir.c_str(), 1);
  setenv("NVIDIA_METRICS_DEVICE", std::to_string(options.device).c_str(), 1);
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

void Observe(const LauncherOptions& options, const std::string& session_id) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(options.duration_seconds);
  while (std::chrono::steady_clock::now() < deadline &&
         (kill(options.pid, 0) == 0 || errno == EPERM)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  StopSession(options, session_id);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const LauncherOptions options = ParseArgs(argc, argv);
    const auto [session_id, output_dir] = CreateSession(options);

    if (options.command.empty()) {
      StartSession(options, session_id, options.pid);
      Observe(options, session_id);
      PrintResults(options, session_id);
      return 0;
    }

    if (!std::filesystem::is_regular_file(options.library_path)) {
      throw std::runtime_error("metrics library not found: " +
                               options.library_path);
    }
    int barrier[2]{};
    if (pipe2(barrier, O_CLOEXEC) != 0) {
      throw std::runtime_error("cannot create process barrier");
    }
    const pid_t pid = LaunchWithBarrier(options, output_dir, barrier[0]);
    close(barrier[0]);
    try {
      StartSession(options, session_id, pid);
    } catch (...) {
      kill(pid, SIGKILL);
      close(barrier[1]);
      waitpid(pid, nullptr, 0);
      throw;
    }
    if (write(barrier[1], "1", 1) != 1) {
      throw std::runtime_error("cannot release target process");
    }
    close(barrier[1]);
    int status = 0;
    waitpid(pid, &status, 0);
    StopSession(options, session_id);
    PrintResults(options, session_id);
    if (WIFEXITED(status)) {
      const int exit_code = WEXITSTATUS(status);
      std::cout << "Target exit code: " << exit_code << '\n';
      return exit_code;
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
    return 1;
  }
}
