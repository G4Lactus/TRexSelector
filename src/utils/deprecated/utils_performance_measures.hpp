#ifndef UTILS_PERF_HPP
#define UTILS_PERF_HPP

// ============================================================================
// Performance, Timing, and Memory Utilities (Header-only)
// ============================================================================

#include <chrono>
#include <iostream>
#include <cstddef>
#include <iomanip>

// --- Platform-specific includes for memory/disk stats ---
#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <psapi.h>
#endif

#ifdef __APPLE__
    #include <mach/mach.h>
#endif

#if !defined(_WIN32) && !defined(_WIN64)
    #include <sys/statvfs.h>   // Disk space check (POSIX)
    #include <sys/resource.h>  // getrusage()
#endif

// --- High-resolution timing alias ---
using Clock = std::chrono::high_resolution_clock;

namespace utils_perf {

/**
 * @brief Get the default output stream for logging.
 *
 * This function provides a configurable default output stream, which is set to std::cout by default.
 *
 * @return Reference to the default output stream.
 */
inline std::ostream*& get_default_stream_ptr() {
    static std::ostream *default_stream = &std::cout;
    return default_stream;
}

inline std::ostream &get_default_stream() {
    return *get_default_stream_ptr();
}

/**
 * @brief Set the default output stream for logging.
 *
 * Allows redirection of the default output stream to a custom stream.
 *
 * @param stream Reference to the new output stream.
 */
inline void set_default_stream(std::ostream &stream) {
    get_default_stream_ptr() = &stream;
}

/**
 * @brief Snapshot of memory usage.
 *
 * This struct captures the current memory usage of the process, including resident RAM and RSS (Resident Set Size).
 */
struct MemorySnapshot {
    double ram_bytes;      ///< Resident process RAM (excludes most file cache)
    double rss_bytes;      ///< Resident Set Size (includes file cache)

    /**
     * @brief Get RAM usage in megabytes (MB).
     * @return RAM usage in MB.
     */
    double ram_mb() const { return ram_bytes / 1e6; }

    /**
     * @brief Get RSS usage in megabytes (MB).
     * @return RSS usage in MB.
     */
    double rss_mb() const { return rss_bytes / 1e6; }

    /**
     * @brief Get RAM usage in mebibytes (MiB).
     * @return RAM usage in MiB.
     */
    double ram_mib() const { return ram_bytes / (1024. * 1024.); }

    /**
     * @brief Get RSS usage in mebibytes (MiB).
     * @return RSS usage in MiB.
     */
    double rss_mib() const { return rss_bytes / (1024. * 1024.); }

    /**
     * @brief Capture the current memory usage of the process.
     *
     * This function provides a platform-agnostic way to capture memory usage, with specific implementations for Windows,
     * macOS, and Linux.
     *
     * @return A MemorySnapshot instance with the current memory usage.
     */
    static MemorySnapshot capture() {
        double ram = 0.0, rss = 0.0;
        #if defined(_WIN32) || defined(_WIN64)
            // Windows: Use GetProcessMemoryInfo from psapi.h
            PROCESS_MEMORY_COUNTERS_EX pmc;
            HANDLE hProc = GetCurrentProcess();
            if (GetProcessMemoryInfo(hProc, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
                ram = static_cast<double>(pmc.PrivateUsage);      // "Private" = actual RAM not shared with other procs
                rss = static_cast<double>(pmc.WorkingSetSize);    // Entirety of process in RAM (shared + private)
            }
        #elif defined(__APPLE__)
            struct mach_task_basic_info info;
            mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
            if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                          (task_info_t)&info, &count) == KERN_SUCCESS) {
                ram = info.resident_size;  // bytes
                rss = info.resident_size;
            }
            // Fallback: getrusage (may return peak rather than current)
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                // macOS (getrusage returns bytes, not KB)
                rss = static_cast<double>(usage.ru_maxrss);
                // Do not overwrite ram if task_info succeeded
            }
        #elif defined(__linux__)
            // Linux: Parse /proc/self/status for VmRSS (in KB)
            FILE* file = fopen("/proc/self/status", "r");
            if (file) {
                char line[256]; double vmrss_kb = 0.0;
                while (fgets(line, sizeof(line), file)) {
                    if (strncmp(line, "VmRSS:", 6) == 0) {
                        sscanf(line + 6, "%lf", &vmrss_kb);
                        break;
                    }
                }
                fclose(file);
                ram = rss = vmrss_kb * 1024.0; // to bytes
            }
        #else
            // Generic fallback: getrusage
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                #ifdef __APPLE__
                    rss = static_cast<double>(usage.ru_maxrss);
                #else
                    rss = static_cast<double>(usage.ru_maxrss) * 1024.0;
                #endif
                ram = rss;
            }
        #endif
        return {ram, rss};
    }

    MemorySnapshot operator-(const MemorySnapshot& o) const {
        return {ram_bytes - o.ram_bytes, rss_bytes - o.rss_bytes};
    }

    /**
     * @brief Print the memory snapshot to the output stream.
     *
     * @param label Label for the memory snapshot.
     * @param os Output stream (default: std::cout).
     */
    void print(const std::string& label, std::ostream &os = get_default_stream()) const {
        os << label << ":\n"
           << "  RAM:  " << std::fixed << std::setprecision(2)
           << ram_mb() << " MB (" << ram_mib() << " MiB)\n"
           << "  RSS:  " << rss_mb() << " MB (" << rss_mib() << " MiB)\n";
    }
};


/**
 * @brief Result of profiling a code block, including time and memory usage.
 */
struct ProfileResult {
    long long time_ms;             ///< Wall-clock time in ms
    MemorySnapshot memory_delta;   ///< Change in resident memory

    /**
     * @brief Print the profiling result to the output stream.
     *
     * @param label Label for the profiling result.
     * @param os Output stream (default: std::cout).
     */
    void print(const std::string& label, std::ostream &os = get_default_stream()) const {
        os << label << ":\n"
           << "  Time: " << time_ms << " ms\n";
        memory_delta.print("  Memory change", os);
    }
};


/**
 * @brief Measure the elapsed time (in milliseconds) for a zero-argument functor or lambda.
 *
 * @tparam Func Type of the functor or lambda.
 * @param f The functor or lambda to measure.
 * @return Elapsed time in milliseconds.
 */
template <typename Func>
inline double timeit(Func&& f) {
    auto start = Clock::now();
    f();
    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}


/**
 * @brief Profile a code block, capturing both wall time and memory usage changes.
 *
 * @tparam Func Type of the functor or lambda.
 * @param f The functor or lambda to profile.
 * @return A ProfileResult containing the elapsed time and memory usage changes.
 */
template <typename Func>
ProfileResult profileit(Func&& f) {
    auto mem_before = MemorySnapshot::capture();
    auto time_start = Clock::now();
    f();
    auto time_end = Clock::now();
    auto mem_after = MemorySnapshot::capture();
    long long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start).count();
    MemorySnapshot mem_delta = mem_after - mem_before;
    return {time_ms, mem_delta};
}




} // namespace utils_perf

#endif // UTILS_PERF_HPP
