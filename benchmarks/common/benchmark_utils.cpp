#include "benchmark_utils.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace benchmarks {
namespace {

std::uint64_t parse_unsigned(const std::string& option,
                             const std::string& text) {
  std::uint64_t value = 0;
  const auto* begin = text.data();
  const auto* end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (text.empty() || result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument("invalid unsigned integer for " + option +
                                ": " + text);
  }
  return value;
}

}  // namespace

ArgumentParser::ArgumentParser(int argc, char** argv) {
  for (int index = 1; index < argc; ++index) {
    std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      help_requested_ = true;
      continue;
    }
    if (argument.rfind("--", 0) != 0 || argument.size() == 2) {
      throw std::invalid_argument("unexpected positional argument: " + argument);
    }

    std::string option;
    std::string value;
    const auto equals = argument.find('=');
    if (equals != std::string::npos) {
      option = argument.substr(0, equals);
      value = argument.substr(equals + 1);
      if (value.empty()) {
        throw std::invalid_argument("missing value for " + option);
      }
    } else {
      option = argument;
      if (index + 1 >= argc || std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
        throw std::invalid_argument("missing value for " + option);
      }
      value = argv[++index];
    }

    if (!options_.emplace(option, value).second) {
      throw std::invalid_argument("duplicate option: " + option);
    }
  }
}

bool ArgumentParser::help_requested() const { return help_requested_; }

bool ArgumentParser::has(const std::string& option) const {
  return options_.find(option) != options_.end();
}

std::string ArgumentParser::string_value(
    const std::string& option, const std::string& default_value) const {
  const auto found = options_.find(option);
  return found == options_.end() ? default_value : found->second;
}

std::uint64_t ArgumentParser::unsigned_value(
    const std::string& option, std::uint64_t default_value) const {
  const auto found = options_.find(option);
  return found == options_.end() ? default_value
                                : parse_unsigned(option, found->second);
}

int ArgumentParser::integer_value(const std::string& option,
                                  int default_value) const {
  const auto found = options_.find(option);
  if (found == options_.end()) {
    return default_value;
  }
  int value = 0;
  const auto* begin = found->second.data();
  const auto* end = begin + found->second.size();
  const auto result = std::from_chars(begin, end, value);
  if (found->second.empty() || result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument("invalid integer for " + option + ": " +
                                found->second);
  }
  return value;
}

double ArgumentParser::double_value(const std::string& option,
                                    double default_value) const {
  const auto found = options_.find(option);
  if (found == options_.end()) {
    return default_value;
  }
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(found->second, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid number for " + option + ": " +
                                found->second);
  }
  if (consumed != found->second.size() || !std::isfinite(value)) {
    throw std::invalid_argument("invalid number for " + option + ": " +
                                found->second);
  }
  return value;
}

void ArgumentParser::validate_options(
    std::initializer_list<const char*> allowed) const {
  std::unordered_set<std::string> allowed_set;
  for (const char* option : allowed) {
    allowed_set.emplace(option);
  }
  for (const auto& [option, unused_value] : options_) {
    (void)unused_value;
    if (allowed_set.find(option) == allowed_set.end()) {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
}

CommonConfig parse_common_config(const ArgumentParser& arguments) {
  CommonConfig config;
  config.device = arguments.integer_value("--device", kDefaultDevice);
  config.duration_seconds =
      arguments.double_value("--duration", kDefaultDurationSeconds);
  const auto warmup =
      arguments.unsigned_value("--warmup", kDefaultWarmupIterations);
  config.block_size =
      arguments.integer_value("--block-size", kDefaultBlockSize);

  if (config.device < 0) {
    throw std::invalid_argument("--device must be non-negative");
  }
  if (!(config.duration_seconds > 0.0)) {
    throw std::invalid_argument("--duration must be greater than zero");
  }
  if (warmup > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("--warmup is too large for this platform");
  }
  if (config.block_size <= 0 || config.block_size > 1024) {
    throw std::invalid_argument("--block-size must be in the range [1, 1024]");
  }
  config.warmup_iterations = static_cast<std::size_t>(warmup);
  return config;
}

void print_common_config(const CommonConfig& config) {
  std::cout << "DEVICE=" << config.device << '\n'
            << "DURATION_SECONDS=" << std::setprecision(12)
            << config.duration_seconds << '\n'
            << "BLOCK_SIZE=" << config.block_size << '\n'
            << "WARMUP=" << config.warmup_iterations << '\n';
}

void print_execution_summary(double elapsed_seconds,
                             std::uint64_t operation_count) {
  std::cout << "STATUS=OK\n"
            << "EXECUTION_TIME_SECONDS=" << std::fixed << std::setprecision(6)
            << elapsed_seconds << '\n'
            << "OPERATIONS_OR_KERNEL_LAUNCHES=" << operation_count << '\n';
}

double elapsed_seconds_since(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace benchmarks
