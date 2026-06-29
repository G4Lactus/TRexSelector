//===================================================================================
/**
 * @brief Logging utilities for TREX framework to support console outputs in
 *        compliance with Rcpp and CRAN policies.
 *        Provides macros for info and warning messages that adapt to
 *        the build environment (Rcpp or standard C++).
 */
//===================================================================================
#ifndef TREX_UTILS_LOGGING_LOGGER_HPP
#define TREX_UTILS_LOGGING_LOGGER_HPP
// ===================================================================================

#ifdef CRAN_BUILD
    #include <Rcpp.h>

    #define TREX_INFO(msg) do { Rcpp::Rcout << msg << "\n"; } while(0)
    #define TREX_WARN(msg) do { Rcpp::Rcerr << msg << "\n"; } while(0)

#else
    #include <iostream> // IWYU pragma: keep

    #define TREX_INFO(msg) do { std::cout << msg << "\n"; } while(0)
    #define TREX_WARN(msg) do { std::cerr << msg << "\n"; } while(0)
#endif

//===================================================================================
#endif /* End of TREX_UTILS_LOGGING_LOGGER_HPP */
