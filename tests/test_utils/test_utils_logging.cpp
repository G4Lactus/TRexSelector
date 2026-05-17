#include <gtest/gtest.h>
#include <utils/logging/logger.hpp>
#include <string>

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
