/**
 * @file kernel-journal.hpp
 * @brief Binary journal of CUPTI kernel records and kernel active-time windows
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_KERNEL_JOURNAL_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_KERNEL_JOURNAL_HPP_

#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace nvidia_process_metrics {

// Fields of one CUPTI concurrent-kernel record. name_id indexes the kernel
// names that the journal stores separately.
struct KernelRecord {
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint32_t device_id = 0;
  uint32_t context_id = 0;
  uint32_t stream_id = 0;
  uint32_t correlation_id = 0;
  uint32_t name_id = 0;
};

// Orders kernel records by start time, then by CUPTI correlation id, which
// follows launch order. Records with the same start therefore always come
// out in the same order, whatever the order in which CUPTI delivered them.
bool KernelRecordLess(const KernelRecord& left, const KernelRecord& right);

// Appends kernel names and runs of kernel records to a binary file while the
// profiled process runs, then merges the runs back in start order.
//
// File layout, little-endian: the 8-byte magic "NVPMKJ01", a uint32 format
// version and the int32 PID; then chunks tagged 'N' (uint32 id, uint32 length,
// name bytes) or 'R' (uint32 count, then count records of kRecordBytes with
// start, end, device, context, stream, correlation and name id). Every run is
// sorted with KernelRecordLess. analyzers/recover_kernel_journal.py reads the
// same layout.
//
// Not thread-safe: callers serialize every call.
class KernelJournal {
 public:
  static constexpr uint32_t kFormatVersion = 1;
  static constexpr size_t kRecordBytes = 36;

  KernelJournal() = default;
  ~KernelJournal();

  KernelJournal(const KernelJournal&) = delete;
  KernelJournal& operator=(const KernelJournal&) = delete;

  // Creates or truncates the journal at path and writes its header. Returns
  // false if the file cannot be written.
  bool Open(const std::filesystem::path& path, int32_t pid);

  // Records the name of a kernel id. Returns false after a write error.
  bool AppendName(uint32_t name_id, std::string_view name);

  // Sorts *records with KernelRecordLess, appends them as one run and flushes
  // the file.
  // On success *records is cleared and keeps its capacity for reuse. Returns
  // false after a write error, leaving *records unchanged apart from its
  // order; every later call also fails.
  bool AppendRun(std::vector<KernelRecord>* records);

  // Flushes buffered chunks to the file. Returns false after a write error.
  bool Flush();

  // Calls visit for every record of the runs that AppendRun() wrote, in
  // KernelRecordLess order. The read buffers of all runs take 4 MiB, or 16
  // records per run beyond 7,281 runs.
  // This includes the runs written before a write error. Returns false if
  // the journal cannot be read.
  bool ForEachSorted(const std::function<void(const KernelRecord&)>& visit);

  // Closes and deletes the journal file. Returns false if it cannot be
  // removed.
  bool Remove();

  bool is_open() const { return file_ != nullptr; }
  bool failed() const { return failed_; }
  uint64_t record_count() const { return record_count_; }
  const std::filesystem::path& path() const { return path_; }

 private:
  struct Run {
    uint64_t offset;
    uint64_t count;
  };

  bool Write(const void* data, size_t size);
  void Close();

  std::FILE* file_ = nullptr;
  std::filesystem::path path_;
  std::vector<Run> runs_;
  std::vector<unsigned char> encoded_;
  uint64_t offset_ = 0;
  uint64_t record_count_ = 0;
  bool failed_ = false;
};

// Appends one kernel_activity.csv row to *out: event id, PID, device,
// context, stream and correlation ids, start, end, duration and the kernel
// name, which must already be quoted for CSV. The text matches what
// std::ostream writes for the same values.
void AppendKernelCsvRow(uint64_t event_id, int64_t pid,
                        const KernelRecord& record,
                        std::string_view quoted_name, std::string* out);

// Accumulates the union of kernel intervals into fixed windows that start at
// the first kernel, without storing the kernels. Records must arrive in start
// order. Intervals that touch are merged, and empty intervals are ignored.
// Records with a zero start or an end before the start are not counted, so
// they cannot move the first window.
class ActiveTimeWindows {
 public:
  explicit ActiveTimeWindows(uint64_t window_ns);

  void Add(uint64_t start_ns, uint64_t end_ns);

  // Adds the last merged interval; call once after the final Add().
  void Finish();

  bool empty() const { return kernel_count_ == 0; }
  uint64_t first_start_ns() const { return first_start_ns_; }
  uint64_t last_end_ns() const { return last_end_ns_; }
  uint64_t window_ns() const { return window_ns_; }

  // Returns the active time of each window from first_start_ns() until the
  // last window that starts before last_end_ns().
  std::vector<uint64_t> ActiveNs() const;

 private:
  void AddMergedInterval(uint64_t start_ns, uint64_t end_ns);

  uint64_t window_ns_;
  uint64_t kernel_count_ = 0;
  uint64_t first_start_ns_ = 0;
  uint64_t last_end_ns_ = 0;
  bool has_interval_ = false;
  uint64_t interval_start_ns_ = 0;
  uint64_t interval_end_ns_ = 0;
  std::vector<uint64_t> active_ns_;
};

}  // namespace nvidia_process_metrics

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_KERNEL_JOURNAL_HPP_
