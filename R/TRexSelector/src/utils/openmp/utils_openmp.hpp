// ===================================================================================
// utils_openmp.hpp
// ===================================================================================
#ifndef UTILS_OPENMP_HPP
#define UTILS_OPENMP_HPP
// ===================================================================================
/**
 * @file utils_openmp.hpp
 *
 * @brief OpenMP Compatibility Layer
 *
 * @details
 * The primary reason for this file is to act as a compilation safeguard.
 * If the T-Rex framework is compiled on a system or compiler without OpenMP support
 * (e.g., standard macOS Clang without Homebrew), directly including <omp.h> or calling
 * OpenMP functions would cause fatal compilation errors.
 *
 * This header intercepts those calls and provides graceful single-threaded stubs
 * (e.g., forcing max_threads to 1). This guarantees that the entire codebase can
 * compile flawlessly and execute sequentially on any machine, regardless of its
 * OpenMP capabilities.
 *
 * Usage: rather than including raw <omp.h>, include this header in your source files:
 * #include <utils/openmp/utils_openmp.hpp>
 *
 * Compiler flags:
 * - GCC/Clang: -fopenmp
 * - MSVC: /openmp
 * - Disable fallback warning: -DDISABLE_OPENMP_WARNING
 */
// ===================================================================================

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

// ===================================================================================

#if defined(_OPENMP)
// OpenMP is available - use native implementation
#include <omp.h>
#define HAS_OPENMP 1
#else
// OpenMP is not available - provide single-threaded stubs
#define HAS_OPENMP 0

// Stub implementations for OpenMP API if omp.h is not available
// ----------------------------------------------------------------

/** @brief Get the thread number (always 0 in single-threaded mode) */
inline int omp_get_thread_num() {
    return 0;
}

/** @brief Get the number of threads (always 1 in single-threaded mode) */
inline int omp_get_num_threads() {
    return 1;
}

/** @brief Get the maximum number of threads (always 1 in single-threaded mode) */
inline int omp_get_max_threads() {
    return 1;
}

/** @brief Get the number of processors (always 1 in single-threaded mode) */
inline int omp_get_num_procs() {
    return 1;
}

/** @brief Set the number of threads (no-op in single-threaded mode) */
inline void omp_set_num_threads(int /* nthread */) {}

/** @brief Set dynamic threads flag (no-op in single-threaded mode) */
inline void omp_set_dynamic(int /* flag */) {}

/** @brief Get dynamic threads flag (always 0 in single-threaded mode) */
inline int omp_get_dynamic() { return 0; }

/** @brief Check if currently in parallel region (always 0 in single-threaded mode) */
inline int omp_in_parallel() { return 0; }

/**
 * @brief Wall-clock timer stub using std::chrono (seconds since first call).
 * @return Elapsed seconds as a double.
 */
inline double omp_get_wtime() {
    static const auto epoch = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - epoch).count();
}

/** @brief Timer precision stub (returns 1 ns as a conservative estimate). */
inline double omp_get_wtick() {
    return 1e-9;
}

/** @brief Schedule kind enum stub matching OpenMP ABI values. */
typedef enum omp_sched_t {
    omp_sched_static  = 1,
    omp_sched_dynamic = 2,
    omp_sched_guided  = 3,
    omp_sched_auto    = 4
} omp_sched_t;

/** @brief Set loop schedule (no-op in single-threaded mode). */
inline void omp_set_schedule(omp_sched_t /* kind */, int /* chunk_size */) {}

/** @brief Get loop schedule (returns static/0 in single-threaded mode). */
inline void omp_get_schedule(omp_sched_t* kind, int* chunk_size) {
    if (kind)       *kind       = omp_sched_static;
    if (chunk_size) *chunk_size = 0;
}

#endif


// ============================================================================
// Utility Functions
// ============================================================================

// Embedded into namespace trex::utils::openmp
namespace trex::utils::openmp {

// ============================================================================


/**
 * @brief Check if OpenMP is available at compile-time
 * @return True if OpenMP is available, false otherwise.
 */
inline constexpr bool is_available() {
    return HAS_OPENMP;
}


/**
 * @brief Check if custom reductions are supported
 * @return True if OpenMP 4.0+ is available, false otherwise.
 */
inline constexpr bool has_reductions() {
#if defined(_OPENMP) && _OPENMP >= 201307
        return true;
#else
    return false;
#endif
}


/**
 * @brief Get OpenMP version as an integer
 * @return The OpenMP version (e.g., 201307 for OpenMP 4.0).
 */
inline constexpr int version() {
#if defined(_OPENMP)
        return _OPENMP;
#else
    return 0;
#endif
}


/**
 * @brief Get a human-readable status string
 * @return A string indicating whether OpenMP is enabled or disabled.
 */
inline const char *status_string() {
    return HAS_OPENMP ? "OpenMP enabled" :
                        "Single-threaded (OpenMP disabled)";
}


/**
 * @brief Print OpenMP configuration info
 *
 * Prints details about the OpenMP configuration, including version,
 * maximum threads, and processor count.
 */
inline void print_info() {
#if defined(_OPENMP)
    std::cout << "OpenMP version: " << _OPENMP << " (";

    // Decode version
    if (_OPENMP >= 202011) {
        std::cout << "5.1+";
    }
    else if (_OPENMP >= 201811) {
        std::cout << "5.0";
    }
    else if (_OPENMP >= 201511) {
        std::cout << "4.5";
    }
    else if (_OPENMP >= 201307) {
        std::cout << "4.0";
    }
    else if (_OPENMP >= 201107) {
        std::cout << "3.1";
    }
    else {
        std::cout << "3.0 or earlier";
    }
    std::cout << ")\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";
    std::cout << "Num processors: " << omp_get_num_procs() << "\n";

    #if defined(_OPENMP) && _OPENMP >= 201307
        std::cout << "Custom reductions: Supported\n";
    #else
        std::cout << "Custom reductions: Not supported (requires OpenMP 4.0+)\n";
    #endif
#else
    std::cout << "OpenMP: Not available (single-threaded mode)\n";
#endif
}

// ============================================================================
// The library's thread budget
// ============================================================================
//
// Every OpenMP region of the library carries an explicit num_threads clause.
// The T-solvers and the selectors take their budget from their options
// (n_threads, max_inner_threads); the one-shot kernels without an option of
// their own (the dummy and data generators, the normalizer, the scalers, the
// clustering, the ML regressions' Eigen products) take the library's default
// budget below, which starts at 1. The OpenMP runtime's own default team is
// never used, so a default call is single-threaded on every machine.
// ============================================================================

/**
 * @brief Work threshold (matrix entries) below which a budgeted region stays
 *        serial, because forking a team costs more than the work.
 */
inline constexpr double kParallelWorkThreshold = 1.0e6;

namespace detail {
    /** @brief Storage of the library's default thread budget. */
    inline std::atomic<int>& default_threads() {
        static std::atomic<int> value{1};
        return value;
    }
}

/**
 * @brief The library's default thread budget, initially 1.
 *
 * Kernels without a thread option of their own use it; set_num_threads()
 * changes it for the process.
 */
inline int get_num_threads() {
    return detail::default_threads().load(std::memory_order_relaxed);
}

/**
 * @brief Set the library's default thread budget.
 * @param num_threads The budget, at least 1.
 * @return The previous budget.
 * @throws std::invalid_argument when num_threads is below 1.
 */
inline int set_num_threads(int num_threads) {
    if (num_threads < 1) {
        throw std::invalid_argument(
            "set_num_threads: num_threads must be >= 1. Got: " + std::to_string(num_threads));
    }
    return detail::default_threads().exchange(num_threads, std::memory_order_relaxed);
}

/**
 * @brief Threads the OpenMP runtime can run on this machine (1 without
 *        OpenMP). An upper bound for a budget, not the budget itself.
 */
inline int get_max_threads() {
    #if HAS_OPENMP
        return omp_get_max_threads();
    #else
        return 1;
    #endif
}

/**
 * @brief Team size of a budgeted region.
 * @param budget The caller's thread budget (an option or get_num_threads()).
 * @param work The region's work in matrix entries.
 * @return 1 when the budget is 1, the work is below kParallelWorkThreshold or
 *         the caller already runs inside a team; the budget otherwise.
 */
inline int threads_for(int budget, double work) noexcept {
    if (budget <= 1 || work < kParallelWorkThreshold || omp_in_parallel() != 0) {
        return 1;
    }
    return budget;
}

/**
 * @brief Team-size probe for the tests: records the largest team any
 *        budgeted region of the library ran on since the last reset.
 *
 * Off by default; record() is one relaxed load per call while off.
 */
namespace team_probe {

    namespace detail {
        inline std::atomic<bool>& enabled() {
            static std::atomic<bool> value{false};
            return value;
        }
        inline std::atomic<int>& max_team() {
            static std::atomic<int> value{0};
            return value;
        }
    }

    /** @brief Switch the probe on or off. */
    inline void enable(bool on) { detail::enabled().store(on, std::memory_order_relaxed); }

    /** @brief Forget the recorded maximum. */
    inline void reset() { detail::max_team().store(0, std::memory_order_relaxed); }

    /** @brief The largest team recorded since the last reset (0 when none ran). */
    inline int max_team() { return detail::max_team().load(std::memory_order_relaxed); }

    /** @brief Called inside a region: records the size of the current team. */
    inline void record() {
        if (!detail::enabled().load(std::memory_order_relaxed)) return;
        const int team = omp_get_num_threads();
        auto& recorded = detail::max_team();
        int current = recorded.load(std::memory_order_relaxed);
        while (team > current &&
               !recorded.compare_exchange_weak(current, team, std::memory_order_relaxed)) {}
    }

} /* End of namespace team_probe */

// ===================================================================================
} /* End of namespace trex::utils::openmp */
// ===================================================================================
#endif /* UTILS_OPENMP_HPP */
