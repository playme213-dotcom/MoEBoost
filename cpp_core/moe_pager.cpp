#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <thread>
#include <cstdlib>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace py = pybind11;

class Pager {
 public:
  double bw_bytes_per_ms_ = 0.0;
  int best_threads_ = 4;
  std::mutex bw_mutex_;
  ~Pager() {
    std::lock_guard<std::mutex> guard(mutex_);
    files_.clear();
  }

  std::uint64_t open_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("open failed for '" + path + "': " + std::strerror(errno));

    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) {
      const auto message = std::string("not a readable regular file: '") + path + "'";
      ::close(fd);
      throw std::runtime_error(message);
    }

    ::posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);

    std::lock_guard<std::mutex> guard(mutex_);
    const auto id = next_id_++;
    files_.emplace(id, std::make_shared<File>(fd, static_cast<std::uint64_t>(st.st_size), path));
    return id;
  }

  void close_file(std::uint64_t id) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = files_.find(id);
    if (it == files_.end()) throw std::runtime_error("unknown or already closed file handle");
    files_.erase(it);
  }

  std::uint64_t file_size(std::uint64_t id) {
    std::lock_guard<std::mutex> guard(mutex_);
    return get_file(id)->size;
  }

  py::bytes read(std::uint64_t id, std::uint64_t offset, std::uint64_t length) {
    std::shared_ptr<File> file;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      file = get_file(id);
      if (offset > file->size || length > file->size - offset)
        throw std::out_of_range("requested byte range exceeds file bounds");
      if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
          length > static_cast<std::uint64_t>(PY_SSIZE_T_MAX))
        throw std::out_of_range("requested byte range exceeds platform limits");
    }

    std::string data(static_cast<std::size_t>(length), '\0');
    std::size_t done = 0;
    const auto start = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      while (done < data.size()) {
        const auto remaining = data.size() - done;
        const auto chunk = ::pread(file->fd, data.data() + done, remaining,
                                   static_cast<off_t>(offset + done));
        if (chunk < 0 && errno == EINTR) continue;
        if (chunk < 0) throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        if (chunk == 0) throw std::runtime_error("unexpected end of file during pread");
        done += static_cast<std::size_t>(chunk);
      }
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    {
      std::lock_guard<std::mutex> guard(mutex_);
      bytes_read_ += done;
      read_ms_ += elapsed;
      ++read_ops_;
    }
    return py::bytes(data.data(), static_cast<py::ssize_t>(data.size()));
  }

  double benchmark_ram_bandwidth_locked(int threads) {
    constexpr std::size_t kBytes = 32u * 1024u * 1024u;
    constexpr int kPasses = 16, kTrials = 3;
    const int kThreads = std::max(1, std::min(threads, 16));
    void* raw = nullptr;
    if (::posix_memalign(&raw, 2u * 1024u * 1024u, kBytes) != 0) return 0.0;
    ::madvise(raw, kBytes, MADV_HUGEPAGE);
    std::memset(raw, 0x5a, kBytes);
    auto* base = static_cast<const std::uint8_t*>(raw);
    std::vector<std::uint64_t> sinks(kThreads, 0);
    double best = 0.0;
    {
      py::gil_scoped_release release;
      std::atomic<int> ready_count{0};
      std::atomic<int> start_generation{0};
      std::atomic<int> done_count{0};
      std::atomic<bool> stop_workers{false};
      std::vector<std::thread> pool;
      pool.reserve(kThreads);
      const std::size_t part = (kBytes / kThreads) & ~std::size_t(255);
      for (int t = 0; t < kThreads; ++t) {
        pool.emplace_back([&, t, part]() {
          const auto* p = base + t * part;
          int my_gen = 0;
          while (true) {
            ready_count.fetch_add(1, std::memory_order_release);
            while (start_generation.load(std::memory_order_acquire) == my_gen) {
              if (stop_workers.load(std::memory_order_acquire)) return;
#if defined(__AVX2__) || defined(__SSE2__)
              _mm_pause();
#endif
            }
            ++my_gen;
            if (stop_workers.load(std::memory_order_acquire)) return;

            std::uint64_t acc = 0;
            for (int pass = 0; pass < kPasses; ++pass) {
#if defined(__AVX2__)
              __m256i v0 = _mm256_setzero_si256();
              __m256i v1 = _mm256_setzero_si256();
              __m256i v2 = _mm256_setzero_si256();
              __m256i v3 = _mm256_setzero_si256();
              __m256i v4 = _mm256_setzero_si256();
              __m256i v5 = _mm256_setzero_si256();
              __m256i v6 = _mm256_setzero_si256();
              __m256i v7 = _mm256_setzero_si256();
              for (std::size_t i = 0; i < part; i += 256) {
                _mm_prefetch(reinterpret_cast<const char*>(p + i + 512), _MM_HINT_T0);
                const auto* vp = reinterpret_cast<const __m256i*>(p + i);
                v0 = _mm256_xor_si256(v0, _mm256_load_si256(vp + 0));
                v1 = _mm256_xor_si256(v1, _mm256_load_si256(vp + 1));
                v2 = _mm256_xor_si256(v2, _mm256_load_si256(vp + 2));
                v3 = _mm256_xor_si256(v3, _mm256_load_si256(vp + 3));
                v4 = _mm256_xor_si256(v4, _mm256_load_si256(vp + 4));
                v5 = _mm256_xor_si256(v5, _mm256_load_si256(vp + 5));
                v6 = _mm256_xor_si256(v6, _mm256_load_si256(vp + 6));
                v7 = _mm256_xor_si256(v7, _mm256_load_si256(vp + 7));
              }
              const __m256i v01 = _mm256_xor_si256(v0, v1);
              const __m256i v23 = _mm256_xor_si256(v2, v3);
              const __m256i v45 = _mm256_xor_si256(v4, v5);
              const __m256i v67 = _mm256_xor_si256(v6, v7);
              const __m256i v = _mm256_xor_si256(
                  _mm256_xor_si256(v01, v23),
                  _mm256_xor_si256(v45, v67));
              alignas(32) std::uint64_t out[4];
              _mm256_store_si256(reinterpret_cast<__m256i*>(out), v);
              acc ^= out[0] ^ out[1] ^ out[2] ^ out[3];
#else
              for (std::size_t i = 0; i < part; i += 8)
                acc ^= *reinterpret_cast<const std::uint64_t*>(p + i);
#endif
            }
            sinks[t] = acc;
            done_count.fetch_add(1, std::memory_order_release);
          }
        });
      }

      const std::size_t total_part_bytes = part * static_cast<std::size_t>(kThreads);
      for (int trial = 0; trial < kTrials; ++trial) {
        while (ready_count.load(std::memory_order_acquire) < kThreads) {
#if defined(__AVX2__) || defined(__SSE2__)
          _mm_pause();
#endif
        }
        ready_count.store(0, std::memory_order_relaxed);
        done_count.store(0, std::memory_order_relaxed);
        const auto t0 = std::chrono::steady_clock::now();
        start_generation.fetch_add(1, std::memory_order_release);
        while (done_count.load(std::memory_order_acquire) < kThreads) {
#if defined(__AVX2__) || defined(__SSE2__)
          _mm_pause();
#endif
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        const double rate = static_cast<double>(total_part_bytes) * kPasses / ms;
        if (rate > best) best = rate;
      }

      while (ready_count.load(std::memory_order_acquire) < kThreads) {
#if defined(__AVX2__) || defined(__SSE2__)
        _mm_pause();
#endif
      }
      stop_workers.store(true, std::memory_order_release);
      start_generation.fetch_add(1, std::memory_order_release);
      for (auto& th : pool) th.join();
    }
    volatile std::uint64_t keep = 0;
    for (auto v : sinks) keep = keep ^ v;
    ::free(raw);
    return best;
  }

  // Multi-threaded pure-READ DDR bandwidth probe. Returns ms to read `size` bytes at measured speed.
  double measure_ram_bandwidth(py::buffer input) {
    auto info = input.request();
    const double size = static_cast<double>(info.size * info.itemsize);
    std::lock_guard<std::mutex> guard(bw_mutex_);
    if (bw_bytes_per_ms_ <= 0.0) {
      bw_bytes_per_ms_ = benchmark_ram_bandwidth_locked(best_threads_);
    }
    return bw_bytes_per_ms_ > 0.0 ? size / bw_bytes_per_ms_ : 0.0;
  }

  py::dict calibrate_ram_bandwidth(const std::vector<int>& thread_candidates = {2, 4, 8}) {
    std::lock_guard<std::mutex> guard(bw_mutex_);
    std::vector<int> candidates = thread_candidates;
    if (candidates.empty()) candidates = {2, 4, 8};
    double best_rate = 0.0;
    int best_t = best_threads_;
    py::list results;
    for (int t : candidates) {
      if (t <= 0) continue;
      const double rate = benchmark_ram_bandwidth_locked(t);
      const double mbs = (rate * 1000.0) / (1024.0 * 1024.0);
      py::dict row;
      row["threads"] = t;
      row["bandwidth_mbs"] = mbs;
      row["bandwidth_gbs"] = mbs / 1024.0;
      results.append(row);
      if (rate > best_rate) {
        best_rate = rate;
        best_t = t;
      }
    }
    if (best_rate > 0.0) {
      bw_bytes_per_ms_ = best_rate;
      best_threads_ = best_t;
    }
    const double best_mbs = (bw_bytes_per_ms_ * 1000.0) / (1024.0 * 1024.0);
    py::dict out;
    out["best_threads"] = best_threads_;
    out["bandwidth_mbs"] = best_mbs;
    out["bandwidth_gbs"] = best_mbs / 1024.0;
    out["candidates"] = results;
    return out;
  }

  bool advise_ram_buffer(py::buffer input) {
    auto info = input.request();
    const auto len = static_cast<std::size_t>(info.size * info.itemsize);
    if (!info.ptr || len == 0) return false;
    const auto page_size = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
    const auto raw_addr = reinterpret_cast<std::uintptr_t>(info.ptr);
    const auto aligned_addr = raw_addr & ~(page_size - 1);
    const auto aligned_len = len + (raw_addr - aligned_addr);
    const int rc_will = ::madvise(reinterpret_cast<void*>(aligned_addr), aligned_len, MADV_WILLNEED);
    int rc_huge = 0;
    if (len >= 2u * 1024u * 1024u) {
      rc_huge = ::madvise(reinterpret_cast<void*>(aligned_addr), aligned_len, MADV_HUGEPAGE);
    }
    return rc_will == 0 && rc_huge == 0;
  }

  bool advise_file_sequential(std::uint64_t id) {
    std::shared_ptr<File> file;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      file = get_file(id);
    }
    const int rc_seq = ::posix_fadvise(file->fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    const int rc_will = ::posix_fadvise(file->fd, 0, 0, POSIX_FADV_WILLNEED);
    return rc_seq == 0 && rc_will == 0;
  }

  void read_into(std::uint64_t id, std::uint64_t offset, py::buffer output) {
    auto info = output.request(true);
    if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1)
      throw std::invalid_argument("output buffer must be a contiguous one-dimensional byte array");
    const auto length = static_cast<std::uint64_t>(info.size);
    if (length > static_cast<std::uint64_t>(PY_SSIZE_T_MAX))
      throw std::out_of_range("output buffer exceeds platform limits");
    std::shared_ptr<File> file;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      file = get_file(id);
      if (offset > file->size || length > file->size - offset)
        throw std::out_of_range("requested byte range exceeds file bounds");
      if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::out_of_range("requested byte offset exceeds platform limits");
    }
    std::size_t done = 0;
    if (length >= 2u * 1024u * 1024u) ::madvise(info.ptr, static_cast<std::size_t>(length), MADV_HUGEPAGE);
    const auto start = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto* target = static_cast<char*>(info.ptr);
      while (done < static_cast<std::size_t>(length)) {
        const auto remaining = static_cast<std::size_t>(length) - done;
        const auto count = ::pread(file->fd, target + done, remaining,
                                   static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::runtime_error(std::string("pread into buffer failed: ") + std::strerror(errno));
        if (count == 0) throw std::runtime_error("unexpected end of file during pread into buffer");
        done += static_cast<std::size_t>(count);
      }
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    {
      std::lock_guard<std::mutex> guard(mutex_);
      bytes_read_ += done;
      read_ms_ += elapsed;
      ++read_ops_;
    }
  }

  void read_segments_into(
      std::uint64_t id,
      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& segments,
      py::buffer output) {
    auto info = output.request(true);
    if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1)
      throw std::invalid_argument("output buffer must be a contiguous one-dimensional byte array");
    const auto total_output = static_cast<std::uint64_t>(info.size);
    std::shared_ptr<File> file;
    std::uint64_t expected_total = 0;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      file = get_file(id);
      for (const auto& seg : segments) {
        const auto offset = seg.first;
        const auto length = seg.second;
        if (offset > file->size || length > file->size - offset)
          throw std::out_of_range("requested segment byte range exceeds file bounds");
        if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            length > static_cast<std::uint64_t>(PY_SSIZE_T_MAX))
          throw std::out_of_range("requested segment byte range exceeds platform limits");
        if (length > total_output - expected_total)
          throw std::invalid_argument("segment lengths exceed output buffer size");
        expected_total += length;
      }
    }
    if (expected_total != total_output)
      throw std::invalid_argument("sum of segment lengths does not match output buffer size");

    std::size_t total_done = 0;
    if (total_output >= 2u * 1024u * 1024u) ::madvise(info.ptr, static_cast<std::size_t>(total_output), MADV_HUGEPAGE);
    const auto start = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto* target = static_cast<char*>(info.ptr);
      std::size_t cursor = 0;
      for (const auto& seg : segments) {
        const auto offset = seg.first;
        const auto length = static_cast<std::size_t>(seg.second);
        std::size_t done = 0;
        while (done < length) {
          const auto remaining = length - done;
          const auto count = ::pread(file->fd, target + cursor + done, remaining,
                                     static_cast<off_t>(offset + done));
          if (count < 0 && errno == EINTR) continue;
          if (count < 0) throw std::runtime_error(std::string("pread_segments_into failed: ") + std::strerror(errno));
          if (count == 0) throw std::runtime_error("unexpected end of file during pread_segments_into");
          done += static_cast<std::size_t>(count);
        }
        cursor += length;
        total_done += done;
      }
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    {
      std::lock_guard<std::mutex> guard(mutex_);
      bytes_read_ += total_done;
      read_ms_ += elapsed;
      read_ops_ += segments.size();
    }
  }

  void record_expert(const std::string& expert_id) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++expert_heat_[expert_id];
  }

  py::dict metrics() {
    std::lock_guard<std::mutex> guard(mutex_);
    py::dict heat;
    for (const auto& item : expert_heat_) heat[py::str(item.first)] = item.second;
    const double throughput = read_ms_ > 0.0
        ? (static_cast<double>(bytes_read_) / (1024.0 * 1024.0)) / (read_ms_ / 1000.0)
        : 0.0;
    py::dict result;
    result["time_used_for_reading_ms"] = read_ms_;
    result["bytes_read"] = bytes_read_;
    result["read_ops"] = read_ops_;
    result["actual_ssd_throughput_mbs"] = throughput;
    result["expert_heat"] = heat;
    return result;
  }

 private:
  struct File {
    int fd;
    std::uint64_t size;
    std::string path;
    File(int descriptor, std::uint64_t length, std::string file_path)
        : fd(descriptor), size(length), path(std::move(file_path)) {}
    ~File() { if (fd >= 0) ::close(fd); }
  };
  std::shared_ptr<File> get_file(std::uint64_t id) {
    auto it = files_.find(id);
    if (it == files_.end()) throw std::runtime_error("unknown or already closed file handle");
    return it->second;
  }
  std::mutex mutex_;
  std::unordered_map<std::uint64_t, std::shared_ptr<File>> files_;
  std::uint64_t next_id_ = 1;
  std::uint64_t bytes_read_ = 0;
  std::uint64_t read_ops_ = 0;
  double read_ms_ = 0.0;
  std::unordered_map<std::string, std::uint64_t> expert_heat_;
};

PYBIND11_MODULE(moe_pager_cpp, m) {
  m.doc() = "Bounds-checked positional file reader for MoE expert data";
  py::class_<Pager>(m, "Pager")
      .def(py::init<>())
      .def("open", &Pager::open_file, py::arg("path"))
      .def("close", &Pager::close_file, py::arg("handle"))
      .def("file_size", &Pager::file_size, py::arg("handle"))
      .def("pread", &Pager::read, py::arg("handle"), py::arg("offset"), py::arg("length"))
      .def("pread_into", &Pager::read_into, py::arg("handle"), py::arg("offset"), py::arg("output"))
      .def("pread_segments_into", &Pager::read_segments_into, py::arg("handle"), py::arg("segments"), py::arg("output"))
      .def("measure_ram_bandwidth", &Pager::measure_ram_bandwidth, py::arg("input"))
      .def("calibrate_ram_bandwidth", &Pager::calibrate_ram_bandwidth,
           py::arg("thread_candidates") = std::vector<int>{2, 4, 8})
      .def("advise_ram_buffer", &Pager::advise_ram_buffer, py::arg("input"))
      .def("advise_file_sequential", &Pager::advise_file_sequential, py::arg("handle"))
      .def("record_expert", &Pager::record_expert, py::arg("expert_id"))
      .def("metrics", &Pager::metrics);
}
