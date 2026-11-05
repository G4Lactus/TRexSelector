//===================================================================================
/**
 * @brief Logging utilities for TREX framework to support console outputs in
 *        compliance with Rcpp and CRAN policies.
 *        Provides macros for info and warning messages that adapt to
 *        the build environment (Rcpp or standard C++), plus a process-wide
 *        runtime mute for the warning stream (MC simulation harnesses spawn
 *        hundreds of solver instances; repeated per-instance warnings would
 *        otherwise drown the simulation output). Muting only affects the
 *        console print — solvers still record every warning in their
 *        warnings_ vector for programmatic retrieval (getWarnings()).
 */
//===================================================================================
#ifndef TREX_UTILS_LOGGING_LOGGER_HPP
#define TREX_UTILS_LOGGING_LOGGER_HPP
// ===================================================================================

#include <atomic>

namespace trex::utils::logging {

/** @brief Process-wide switch for the warning stream (default: enabled). */
inline std::atomic<bool>& warningsEnabledFlag() noexcept
{
    static std::atomic<bool> enabled{true};
    return enabled;
}

/** @brief Whether TREX_WARN currently prints to the warning stream. */
inline bool warningsEnabled() noexcept
{
    return warningsEnabledFlag().load(std::memory_order_relaxed);
}

/** @brief Enable/disable the warning stream (warnings stay recorded). */
inline void setWarningsEnabled(bool enabled) noexcept
{
    warningsEnabledFlag().store(enabled, std::memory_order_relaxed);
}

} // namespace trex::utils::logging

#ifdef CRAN_BUILD
    #include <Rcpp.h>

    #define TREX_INFO(msg) do { Rcpp::Rcout << msg << "\n"; } while(0)
    #define TREX_WARN(msg) do { \
        if (trex::utils::logging::warningsEnabled()) { \
            Rcpp::Rcerr << msg << "\n"; \
        } \
    } while(0)

#else
    #include <iostream> // IWYU pragma: keep

    #define TREX_INFO(msg) do { std::cout << msg << "\n"; } while(0)
    #define TREX_WARN(msg) do { \
        if (trex::utils::logging::warningsEnabled()) { \
            std::cerr << msg << "\n"; \
        } \
    } while(0)
#endif

//===================================================================================
#endif /* End of TREX_UTILS_LOGGING_LOGGER_HPP */
