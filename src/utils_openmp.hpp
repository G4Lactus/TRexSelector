#ifndef UTILS_OPENMP_HPP
#define UTILS_OPENMP_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

// ============================================================================
// OpenMP Compatibility Layer
// ============================================================================
// This header provides a unified interface for OpenMP functionality, with
// graceful fallback to single-threaded stubs when OpenMP is not available.
//
// Usage:
//   #include "utils_openmp.hpp"
//
// Compiler flags:
//   - GCC/Clang: -fopenmp
//   - MSVC: /openmp
//   - Disable warning: -DDISABLE_OPENMP_WARNING
// ============================================================================

#if defined(_OPENMP)
// OpenMP is available - use native implementation
#include <omp.h>

#define HAS_OPENMP 1

// Introduce Custom OpenMP Reduction for OpenMP 4.0+
// =======================================================
#if _OPENMP >= 201307

/**
         * @brief Custom reduction for std::vector<T> element-wise sum
         *
         * @details Enables parallel accumulation without atomic operations.
         * Each thread maintains a private vector
         * initialized to zeros, then all copies are reduced via element-wise
         * addition.
         *
         * @tparam T The data type of the vector elements.
         *
         * @example
         * @code {.cpp}
         *  std::vector<double> result(n, 0.0);
         *  #pragma omp parallel for reduction(vec_plus: results)
         *  for (std::size_t i = 0; i < iterations; ++i) {
         *      // accumulate into result
         *  }
         * @endcode
         */
// Note: OpenMP declare reduction for std::vector is not fully portable
// Commented out to avoid compilation issues on some compilers
// template <typename T>
// #pragma omp declare reduction(vec_plus : std::vector<T> :
//     std::transform(omp_in.begin(), omp_in.end(), omp_out.begin(),
//                    omp_out.begin(), std::plus<>()))
//     initializer(omp_priv = std::vector<T>(omp_orig.size(), T{}))
#endif // OpenMP 4.0+

#else
// OpenMP not available - provide single-threaded stubs
#define HAS_OPENMP 0

// Emit compile-time warning (can be disabled with -DDISABLE_OPENMP_WARNING)
#ifndef DISABLE_OPENMP_WARNING
#if defined(__GNUC__) || defined(__clang__)
#warning "OpenMP not detected. Compiling in single-threaded mode. Use -fopenmp to enable parallelism."
#elif defined(_MSC_VER)
#pragma message("Warning: OpenMP not detected. Compiling in single-threaded mode. Use /openmp to enable parallelism.")
#endif
#endif

// Stub implementations for OpenMP API
inline int omp_get_thread_num() {
    return 0;
}
inline int omp_get_num_threads() {
    return 1;
}
inline int omp_get_max_threads() {
    return 1;
}
inline int omp_get_num_procs() {
    return 1;
}
inline void omp_set_num_threads(int /* nthread */) {}
inline void omp_set_dynamic(int /* flag */) {}
inline int omp_get_dynamic() { return 0; }
inline int omp_in_parallel() { return 0; }

#endif

// ============================================================================
// Utility Functions
// ============================================================================

namespace openmp
{

    /**
     * @brief Check if OpenMP is available at compile-time
     * @return True if OpenMP is available, false otherwise.
     */
    inline constexpr bool is_available()
    {
        return HAS_OPENMP;
    }

    /**
     * @brief Check if custom reductions are supported
     * @return True if OpenMP 4.0+ is available, false otherwise.
     */
    inline constexpr bool has_reductions()
    {
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
    inline constexpr int version()
    {
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
    inline const char *status_string()
    {
        return HAS_OPENMP ? "OpenMP enabled" :
                            "Single-threaded (OpenMP disabled)";
    }

    /**
     * @brief Print OpenMP configuration info
     *
     * Prints details about the OpenMP configuration, including version,
     * maximum threads, and processor count.
     */
    inline void print_info()
    {
#if defined(_OPENMP)
        std::cout << "OpenMP version: " << _OPENMP << " (";

        // Decode version
        if (_OPENMP >= 202011)
        {
            std::cout << "5.1+";
        }
        else if (_OPENMP >= 201811)
        {
            std::cout << "5.0";
        }
        else if (_OPENMP >= 201511)
        {
            std::cout << "4.5";
        }
        else if (_OPENMP >= 201307)
        {
            std::cout << "4.0";
        }
        else if (_OPENMP >= 201107)
        {
            std::cout << "3.1";
        }
        else
        {
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

} // namespace openmp

#endif /* UTILS_OPENMP_HPP */
