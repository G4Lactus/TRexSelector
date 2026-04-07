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

#include <chrono>
#include <iostream>

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

/** @brief Set the number of threads for OpenMP */
inline void set_num_threads(int num_threads) {
    #if HAS_OPENMP
        omp_set_num_threads(num_threads);
    #endif
}

/** @brief Get the maximum number of threads available for OpenMP */
inline int get_max_threads() {
    #if HAS_OPENMP
        return omp_get_max_threads();
    #else
        return 1;
    #endif
}

// ===================================================================================
} /* End of namespace trex::utils::openmp */
// ===================================================================================
#endif /* UTILS_OPENMP_HPP */
