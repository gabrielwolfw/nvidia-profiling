/**
 * @file kernel-journal.cpp
 * @brief Binary journal of CUPTI kernel records and kernel active-time windows
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include "efimon/gpu/nvidia-gpu/kernel-journal.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <functional>
#include <queue>
#include <string>
#include <system_error>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

// The journal stores integers in host order and documents them as
// little-endian, which holds on the x86-64 and AArch64 hosts it targets.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "KernelJournal requires a little-endian host");

namespace nvidia_process_metrics {
namespace {

constexpr char kMagic[8] = {'N', 'V', 'P', 'M', 'K', 'J', '0', '1'};
constexpr unsigned char kNameTag = 'N';
constexpr unsigned char kRunTag = 'R';
// Records read at once from each run while merging.
constexpr size_t kMergeRecordsPerRead = 512;

template <typename T>
void AppendValue(std::vector<unsigned char>* bytes, T value) {
  unsigned char encoded[sizeof(value)];
  std::memcpy(encoded, &value, sizeof(value));
  bytes->insert(bytes->end(), encoded, encoded + sizeof(value));
}

template <typename T>
T ReadValue(const unsigned char* bytes) {
  T value;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

// Writes the kRecordBytes encoding of record at bytes.
void EncodeRecord(const KernelRecord& record, unsigned char* bytes) {
  std::memcpy(bytes, &record.start_ns, 8);
  std::memcpy(bytes + 8, &record.end_ns, 8);
  std::memcpy(bytes + 16, &record.device_id, 4);
  std::memcpy(bytes + 20, &record.context_id, 4);
  std::memcpy(bytes + 24, &record.stream_id, 4);
  std::memcpy(bytes + 28, &record.correlation_id, 4);
  std::memcpy(bytes + 32, &record.name_id, 4);
}

template <typename T>
void AppendDecimal(T value, std::string* out) {
  char digits[24];
  const std::to_chars_result result =
      std::to_chars(digits, digits + sizeof(digits), value);
  out->append(digits, result.ptr);
}

KernelRecord DecodeRecord(const unsigned char* bytes) {
  KernelRecord record;
  record.start_ns = ReadValue<uint64_t>(bytes);
  record.end_ns = ReadValue<uint64_t>(bytes + 8);
  record.device_id = ReadValue<uint32_t>(bytes + 16);
  record.context_id = ReadValue<uint32_t>(bytes + 20);
  record.stream_id = ReadValue<uint32_t>(bytes + 24);
  record.correlation_id = ReadValue<uint32_t>(bytes + 28);
  record.name_id = ReadValue<uint32_t>(bytes + 32);
  return record;
}

// Reads one run of a journal through a small buffer.
class RunReader {
 public:
  RunReader(int descriptor, uint64_t offset, uint64_t count)
      : descriptor_(descriptor), offset_(offset), remaining_(count) {}

  // Moves to the next record. Returns false at the end of the run or after a
  // read error, which error() reports.
  bool Next() {
    if (position_ == buffered_) {
      if (remaining_ == 0 || !Refill()) return false;
    }
    current_ =
        DecodeRecord(buffer_.data() + position_ * KernelJournal::kRecordBytes);
    ++position_;
    return true;
  }

  const KernelRecord& current() const { return current_; }
  bool error() const { return error_; }

 private:
  bool Refill() {
    const uint64_t records =
        std::min<uint64_t>(remaining_, kMergeRecordsPerRead);
    const size_t bytes = records * KernelJournal::kRecordBytes;
    buffer_.resize(bytes);
    size_t done = 0;
    while (done < bytes) {
      const ssize_t result = pread(descriptor_, buffer_.data() + done,
                                   bytes - done, offset_ + done);
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) {
        error_ = true;
        return false;
      }
      done += result;
    }
    offset_ += bytes;
    remaining_ -= records;
    buffered_ = records;
    position_ = 0;
    return true;
  }

  int descriptor_;
  uint64_t offset_;
  uint64_t remaining_;
  std::vector<unsigned char> buffer_;
  size_t buffered_ = 0;
  size_t position_ = 0;
  KernelRecord current_;
  bool error_ = false;
};

}  // namespace

bool KernelRecordLess(const KernelRecord& left, const KernelRecord& right) {
  if (left.start_ns != right.start_ns) return left.start_ns < right.start_ns;
  return left.correlation_id < right.correlation_id;
}

KernelJournal::~KernelJournal() { Close(); }

bool KernelJournal::Open(const std::filesystem::path& path, int32_t pid) {
  Close();
  path_ = path;
  runs_.clear();
  offset_ = 0;
  record_count_ = 0;
  failed_ = false;

  file_ = std::fopen(path.c_str(), "wb");
  if (file_ == nullptr) {
    failed_ = true;
    return false;
  }

  encoded_.clear();
  encoded_.insert(encoded_.end(), kMagic, kMagic + sizeof(kMagic));
  AppendValue(&encoded_, kFormatVersion);
  AppendValue(&encoded_, pid);
  return Write(encoded_.data(), encoded_.size());
}

bool KernelJournal::AppendName(uint32_t name_id, std::string_view name) {
  encoded_.clear();
  encoded_.push_back(kNameTag);
  AppendValue(&encoded_, name_id);
  AppendValue(&encoded_, static_cast<uint32_t>(name.size()));
  encoded_.insert(encoded_.end(), name.begin(), name.end());
  return Write(encoded_.data(), encoded_.size());
}

bool KernelJournal::AppendRun(std::vector<KernelRecord>* records) {
  if (records->empty()) return !failed_;

  // CUPTI buffers are usually in order already.
  if (!std::is_sorted(records->begin(), records->end(), KernelRecordLess)) {
    std::sort(records->begin(), records->end(), KernelRecordLess);
  }

  constexpr size_t kRunHeaderBytes = 1 + sizeof(uint32_t);
  const auto count = static_cast<uint32_t>(records->size());
  encoded_.resize(kRunHeaderBytes + records->size() * kRecordBytes);
  encoded_[0] = kRunTag;
  std::memcpy(encoded_.data() + 1, &count, sizeof(count));
  unsigned char* next = encoded_.data() + kRunHeaderBytes;
  for (const KernelRecord& record : *records) {
    EncodeRecord(record, next);
    next += kRecordBytes;
  }

  // A run counts only once it reached the file, so that a later write error
  // never hides the runs written before it. On failure *records is kept for
  // the caller to store elsewhere.
  const uint64_t records_offset = offset_ + 1 + sizeof(uint32_t);
  if (!Write(encoded_.data(), encoded_.size()) || !Flush()) return false;

  runs_.push_back({records_offset, records->size()});
  record_count_ += records->size();
  records->clear();
  return true;
}

bool KernelJournal::Flush() {
  if (file_ == nullptr || failed_) return false;
  if (std::fflush(file_) != 0) failed_ = true;
  return !failed_;
}

bool KernelJournal::ForEachSorted(
    const std::function<void(const KernelRecord&)>& visit) {
  // Every listed run was flushed by AppendRun(), so the runs are readable
  // even if a later write failed.
  const int descriptor = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) return false;

  std::vector<RunReader> readers;
  readers.reserve(runs_.size());
  for (const Run& run : runs_) {
    readers.emplace_back(descriptor, run.offset, run.count);
  }

  // Ordered as KernelRecordLess, then by run for identical keys.
  using HeapEntry = std::tuple<uint64_t, uint32_t, size_t>;
  std::priority_queue<HeapEntry, std::vector<HeapEntry>,
                      std::greater<HeapEntry>>
      heap;
  bool read_failed = false;
  for (size_t index = 0; index < readers.size(); ++index) {
    if (readers[index].Next()) {
      heap.push({readers[index].current().start_ns,
                 readers[index].current().correlation_id, index});
    } else if (readers[index].error()) {
      read_failed = true;
    }
  }

  while (!heap.empty() && !read_failed) {
    const size_t index = std::get<2>(heap.top());
    heap.pop();
    visit(readers[index].current());
    if (readers[index].Next()) {
      heap.push({readers[index].current().start_ns,
                 readers[index].current().correlation_id, index});
    } else if (readers[index].error()) {
      read_failed = true;
    }
  }

  close(descriptor);
  return !read_failed;
}

bool KernelJournal::Remove() {
  Close();
  if (path_.empty()) return true;
  std::error_code error;
  std::filesystem::remove(path_, error);
  return !error;
}

bool KernelJournal::Write(const void* data, size_t size) {
  if (file_ == nullptr || failed_) return false;
  if (std::fwrite(data, 1, size, file_) != size) {
    failed_ = true;
    return false;
  }
  offset_ += size;
  return true;
}

void KernelJournal::Close() {
  if (file_ == nullptr) return;
  if (std::fclose(file_) != 0) failed_ = true;
  file_ = nullptr;
}

void AppendKernelCsvRow(uint64_t event_id, int64_t pid,
                        const KernelRecord& record,
                        std::string_view quoted_name, std::string* out) {
  AppendDecimal(event_id, out);
  out->push_back(',');
  AppendDecimal(pid, out);
  out->push_back(',');
  AppendDecimal(record.device_id, out);
  out->push_back(',');
  AppendDecimal(record.context_id, out);
  out->push_back(',');
  AppendDecimal(record.stream_id, out);
  out->push_back(',');
  AppendDecimal(record.correlation_id, out);
  out->push_back(',');
  AppendDecimal(record.start_ns, out);
  out->push_back(',');
  AppendDecimal(record.end_ns, out);
  out->push_back(',');
  // Unsigned subtraction, as the stream-based writer did.
  AppendDecimal(record.end_ns - record.start_ns, out);
  out->push_back(',');
  out->append(quoted_name);
  out->push_back('\n');
}

ActiveTimeWindows::ActiveTimeWindows(uint64_t window_ns)
    : window_ns_(std::max<uint64_t>(window_ns, 1)) {}

void ActiveTimeWindows::Add(uint64_t start_ns, uint64_t end_ns) {
  if (kernel_count_ == 0) {
    first_start_ns_ = start_ns;
    last_end_ns_ = end_ns;
  } else {
    last_end_ns_ = std::max(last_end_ns_, end_ns);
  }
  ++kernel_count_;

  if (start_ns >= end_ns) return;
  if (!has_interval_) {
    has_interval_ = true;
    interval_start_ns_ = start_ns;
    interval_end_ns_ = end_ns;
    return;
  }
  if (start_ns > interval_end_ns_) {
    AddMergedInterval(interval_start_ns_, interval_end_ns_);
    interval_start_ns_ = start_ns;
    interval_end_ns_ = end_ns;
    return;
  }
  interval_end_ns_ = std::max(interval_end_ns_, end_ns);
}

void ActiveTimeWindows::Finish() {
  if (!has_interval_) return;
  AddMergedInterval(interval_start_ns_, interval_end_ns_);
  has_interval_ = false;
}

std::vector<uint64_t> ActiveTimeWindows::ActiveNs() const {
  std::vector<uint64_t> windows = active_ns_;
  const uint64_t span =
      last_end_ns_ > first_start_ns_ ? last_end_ns_ - first_start_ns_ : 0;
  windows.resize((span + window_ns_ - 1) / window_ns_, 0);
  return windows;
}

void ActiveTimeWindows::AddMergedInterval(uint64_t start_ns, uint64_t end_ns) {
  const uint64_t first_window = (start_ns - first_start_ns_) / window_ns_;
  const uint64_t last_window = (end_ns - 1 - first_start_ns_) / window_ns_;
  if (active_ns_.size() <= last_window) active_ns_.resize(last_window + 1, 0);

  for (uint64_t window = first_window; window <= last_window; ++window) {
    const uint64_t window_start = first_start_ns_ + window * window_ns_;
    const uint64_t window_end = window_start + window_ns_;
    active_ns_[window] +=
        std::min(end_ns, window_end) - std::max(start_ns, window_start);
  }
}

}  // namespace nvidia_process_metrics
