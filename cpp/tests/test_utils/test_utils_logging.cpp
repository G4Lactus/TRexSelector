// ========================================================================================
/**
 * @file test_utils_logging.cpp
 *
 * @brief Unit tests for logging utilities in utils_logging.hpp
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <string>

// project utils includes
#include <utils/logging/logger.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::utils {

/** @brief Test the TREX_INFO macro. */
TEST(LoggingTest, InfoMacro) {
#ifndef CRAN_BUILD
    testing::internal::CaptureStdout();
    TREX_INFO("Test Info Message");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("Test Info Message"), std::string::npos);
#else
    // Behavior under CRAN build directs to Rcpp::Rcout,
    // which cannot be trivially captured natively here.
    SUCCEED();
#endif
}


/** @brief Test the TREX_WARN macro. */
TEST(LoggingTest, WarnMacro) {
#ifndef CRAN_BUILD
    testing::internal::CaptureStderr();
    TREX_WARN("Test Warning Message");
    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Test Warning Message"), std::string::npos);
#else
    // Behavior under CRAN build directs to Rcpp::Rcerr
    SUCCEED();
#endif
}

// ========================================================================================
} /* End of namespace trex::test::utils */
