// =========================================================================================
/**
 * @file test_utils_fp_classify.cpp
 *
 * @brief Unit tests for evaluation of floating-point classification utilities in
 *        utils/fp_classify.
 */
// =========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <limits>

// Eigen includes
#include <Eigen/Dense>

// project utils includes
#include <utils/fp_classify/fp_classify.hpp>

// =========================================================================================

// Embed into test namespace
namespace trex::test::utils::fp_classify {

// Namespace alias for convenience
using namespace trex::utils::fp_classify;

// =========================================================================================

/** @brief Test for checking if a value is NaN. */
TEST(FPClassifyTest, IsNaN) {
    double q_nan = std::numeric_limits<double>::quiet_NaN();
    double s_nan = std::numeric_limits<double>::signaling_NaN();
    double inf = std::numeric_limits<double>::infinity();
    double normal = 3.14159;

    EXPECT_TRUE(isnan(q_nan));
    EXPECT_TRUE(isnan(s_nan));
    EXPECT_FALSE(isnan(inf));
    EXPECT_FALSE(isnan(normal));
    EXPECT_FALSE(isnan(0.0));
}


/** @brief Test for checking if a value is infinite. */
TEST(FPClassifyTest, IsInf) {
    double inf = std::numeric_limits<double>::infinity();
    double n_inf = -std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();
    double normal = 3.14159;

    EXPECT_TRUE(isinf(inf));
    EXPECT_TRUE(isinf(n_inf));
    EXPECT_FALSE(isinf(nan));
    EXPECT_FALSE(isinf(normal));
    EXPECT_FALSE(isinf(0.0));
}


/** @brief Test for checking if a value is finite. */
TEST(FPClassifyTest, IsFinite) {
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();
    double normal = 3.14159;

    EXPECT_FALSE(trex::utils::fp_classify::isfinite(inf));
    EXPECT_FALSE(trex::utils::fp_classify::isfinite(nan));
    EXPECT_TRUE(trex::utils::fp_classify::isfinite(normal));
    EXPECT_TRUE(trex::utils::fp_classify::isfinite(0.0));
    EXPECT_TRUE(trex::utils::fp_classify::isfinite(-0.0));
}


/** @brief Test for checking if all elements in a vector are finite. */
TEST(FPClassifyTest, AllFinite) {
    Eigen::VectorXd valid(3);
    valid << 1.0, 2.0, 3.0;
    EXPECT_TRUE(allFinite(valid));

    Eigen::VectorXd with_nan(3);
    with_nan << 1.0, std::numeric_limits<double>::quiet_NaN(), 3.0;
    EXPECT_FALSE(allFinite(with_nan));

    Eigen::VectorXd with_inf(3);
    with_inf << 1.0, 2.0, std::numeric_limits<double>::infinity();
    EXPECT_FALSE(allFinite(with_inf));
}

// =========================================================================================
} /* End ofnamespace trex::test::utils::fp_classify */
