/**
 * @file kernel-journal-test.cpp
 * @brief Tests for the kernel journal and the active-time windows
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include "efimon/gpu/nvidia-gpu/kernel-journal.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace nvidia_process_metrics {
namespace {

int failures = 0;

void Check(bool condition, const char* expression, int line) {
  if (condition) return;
  std::fprintf(stderr, "kernel-journal-test.cpp:%d: NVPM_VERIFY(%s) failed\n",
               line, expression);
  ++failures;
}

#define NVPM_VERIFY(condition) Check((condition), #condition, __LINE__)

std::filesystem::path TemporaryPath(const char* name) {
  return std::filesystem::temp_directory_path() /
         (std::string(name) + "-" + std::to_string(getpid()) + ".journal");
}

bool SameRecord(const KernelRecord& left, const KernelRecord& right) {
  return left.start_ns == right.start_ns && left.end_ns == right.end_ns &&
         left.device_id == right.device_id &&
         left.context_id == right.context_id &&
         left.stream_id == right.stream_id &&
         left.correlation_id == right.correlation_id &&
         left.name_id == right.name_id;
}

// The window computation of the original collector, used as the reference.
std::vector<uint64_t> ReferenceWindows(std::vector<KernelRecord> kernels,
                                       uint64_t window_ns) {
  std::stable_sort(kernels.begin(), kernels.end(),
                   [](const KernelRecord& left, const KernelRecord& right) {
                     return left.start_ns < right.start_ns;
                   });
  std::vector<std::pair<uint64_t, uint64_t>> intervals;
  for (const KernelRecord& kernel : kernels) {
    if (kernel.start_ns >= kernel.end_ns) continue;
    if (intervals.empty() || kernel.start_ns > intervals.back().second) {
      intervals.push_back({kernel.start_ns, kernel.end_ns});
    } else {
      intervals.back().second =
          std::max(intervals.back().second, kernel.end_ns);
    }
  }

  uint64_t first = kernels[0].start_ns;
  uint64_t last = kernels[0].end_ns;
  for (const KernelRecord& kernel : kernels) {
    first = std::min(first, kernel.start_ns);
    last = std::max(last, kernel.end_ns);
  }

  std::vector<uint64_t> windows;
  for (uint64_t start = first; start < last; start += window_ns) {
    const uint64_t end = start + window_ns;
    uint64_t active = 0;
    for (const auto& interval : intervals) {
      const uint64_t overlap_start = std::max(interval.first, start);
      const uint64_t overlap_end = std::min(interval.second, end);
      if (overlap_end > overlap_start) active += overlap_end - overlap_start;
    }
    windows.push_back(active);
  }
  return windows;
}

std::vector<KernelRecord> RandomKernels(std::mt19937_64* random, size_t count) {
  std::uniform_int_distribution<uint64_t> start(1'000'000, 9'000'000);
  std::uniform_int_distribution<uint64_t> duration(0, 40'000);
  std::uniform_int_distribution<uint32_t> small(0, 7);
  std::vector<KernelRecord> kernels(count);
  for (size_t index = 0; index < count; ++index) {
    KernelRecord& kernel = kernels[index];
    // Coarse starts create ties between runs.
    kernel.start_ns = start(*random) / 1000 * 1000;
    kernel.end_ns = kernel.start_ns + duration(*random);
    kernel.device_id = small(*random);
    kernel.context_id = small(*random);
    kernel.stream_id = small(*random);
    kernel.correlation_id = static_cast<uint32_t>(index);
    kernel.name_id = small(*random);
  }
  return kernels;
}

void TestMergesRunsInStartOrder() {
  std::mt19937_64 random(2026);
  const std::vector<KernelRecord> kernels = RandomKernels(&random, 20'000);
  const std::filesystem::path path = TemporaryPath("kernel-journal-merge");

  KernelJournal journal;
  NVPM_VERIFY(journal.Open(path, 1234));
  NVPM_VERIFY(journal.AppendName(0, "kernel_a"));
  // Runs of uneven sizes, including runs smaller than the merge buffer.
  std::vector<KernelRecord> run;
  size_t next = 0;
  for (size_t run_size = 1; next < kernels.size();
       run_size = run_size * 3 % 997 + 1) {
    run.assign(kernels.begin() + next,
               kernels.begin() + std::min(kernels.size(), next + run_size));
    next += run.size();
    NVPM_VERIFY(journal.AppendRun(&run));
    NVPM_VERIFY(run.empty());
  }
  NVPM_VERIFY(journal.record_count() == kernels.size());

  std::vector<KernelRecord> merged;
  NVPM_VERIFY(journal.ForEachSorted(
      [&merged](const KernelRecord& kernel) { merged.push_back(kernel); }));

  std::vector<KernelRecord> expected = kernels;
  std::stable_sort(expected.begin(), expected.end(),
                   [](const KernelRecord& left, const KernelRecord& right) {
                     return left.start_ns < right.start_ns;
                   });
  NVPM_VERIFY(merged.size() == expected.size());
  bool sorted = true;
  for (size_t index = 1; index < merged.size(); ++index) {
    sorted = sorted && merged[index - 1].start_ns <= merged[index].start_ns;
  }
  NVPM_VERIFY(sorted);
  // Same multiset of records: compare after a total order.
  auto total_order = [](const KernelRecord& left, const KernelRecord& right) {
    return left.correlation_id < right.correlation_id;
  };
  std::sort(merged.begin(), merged.end(), total_order);
  std::sort(expected.begin(), expected.end(), total_order);
  bool same = merged.size() == expected.size();
  for (size_t index = 0; same && index < merged.size(); ++index) {
    same = SameRecord(merged[index], expected[index]);
  }
  NVPM_VERIFY(same);

  NVPM_VERIFY(journal.Remove());
  NVPM_VERIFY(!std::filesystem::exists(path));
}

void TestWritesDocumentedLayout() {
  const std::filesystem::path path = TemporaryPath("kernel-journal-layout");
  KernelJournal journal;
  NVPM_VERIFY(journal.Open(path, 77));
  NVPM_VERIFY(journal.AppendName(5, "gemm"));
  std::vector<KernelRecord> run(1);
  run[0].start_ns = 10;
  run[0].end_ns = 25;
  run[0].stream_id = 3;
  run[0].name_id = 5;
  NVPM_VERIFY(journal.AppendRun(&run));
  NVPM_VERIFY(journal.Flush());

  std::ifstream input(path, std::ios::binary);
  const std::vector<unsigned char> bytes(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  // Header (16) + name chunk (1 + 4 + 4 + 4) + run chunk (1 + 4 + 36).
  NVPM_VERIFY(bytes.size() == 16 + 13 + 41);
  NVPM_VERIFY(std::memcmp(bytes.data(), "NVPMKJ01", 8) == 0);
  NVPM_VERIFY(bytes[16] == 'N');
  NVPM_VERIFY(bytes[29] == 'R');
  uint64_t start = 0;
  std::memcpy(&start, bytes.data() + 34, sizeof(start));
  NVPM_VERIFY(start == 10);

  NVPM_VERIFY(journal.Remove());
}

void TestEmptyJournalVisitsNothing() {
  const std::filesystem::path path = TemporaryPath("kernel-journal-empty");
  KernelJournal journal;
  NVPM_VERIFY(journal.Open(path, 1));
  size_t visited = 0;
  NVPM_VERIFY(
      journal.ForEachSorted([&visited](const KernelRecord&) { ++visited; }));
  NVPM_VERIFY(visited == 0);
  NVPM_VERIFY(journal.Remove());
}

void TestWindowsMatchOriginalComputation() {
  std::mt19937_64 random(7);
  for (const uint64_t window_ns : {1'000ULL, 50'000ULL, 200'000ULL}) {
    std::vector<KernelRecord> kernels = RandomKernels(&random, 3'000);
    const std::vector<uint64_t> expected = ReferenceWindows(kernels, window_ns);

    std::stable_sort(kernels.begin(), kernels.end(),
                     [](const KernelRecord& left, const KernelRecord& right) {
                       return left.start_ns < right.start_ns;
                     });
    ActiveTimeWindows windows(window_ns);
    for (const KernelRecord& kernel : kernels) {
      windows.Add(kernel.start_ns, kernel.end_ns);
    }
    windows.Finish();
    NVPM_VERIFY(windows.ActiveNs() == expected);
  }
}

void TestWindowsWithoutKernels() {
  ActiveTimeWindows windows(200);
  windows.Finish();
  NVPM_VERIFY(windows.empty());
  NVPM_VERIFY(windows.ActiveNs().empty());
}

}  // namespace
}  // namespace nvidia_process_metrics

int main() {
  nvidia_process_metrics::TestMergesRunsInStartOrder();
  nvidia_process_metrics::TestWritesDocumentedLayout();
  nvidia_process_metrics::TestEmptyJournalVisitsNothing();
  nvidia_process_metrics::TestWindowsMatchOriginalComputation();
  nvidia_process_metrics::TestWindowsWithoutKernels();
  if (nvidia_process_metrics::failures != 0) {
    std::fprintf(stderr, "%d checks failed\n",
                 nvidia_process_metrics::failures);
    return 1;
  }
  std::printf("kernel-journal-test: all checks passed\n");
  return 0;
}
