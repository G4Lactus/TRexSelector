#ifndef TREX_UTILS_LOGGING_LOGGER_HPP
#define TREX_UTILS_LOGGING_LOGGER_HPP

#ifdef CRAN_BUILD
    #include <Rcpp.h>
    #define TREX_INFO(msg) do { Rcpp::Rcout << msg << "\n"; } while(0)
    #define TREX_WARN(msg) do { Rcpp::Rcerr << msg << "\n"; } while(0)
#else
    #include <iostream>
    #define TREX_INFO(msg) do { std::cout << msg << "\n"; } while(0)
    #define TREX_WARN(msg) do { std::cerr << msg << "\n"; } while(0)
#endif

#endif // TREX_UTILS_LOGGING_LOGGER_HPP
