// ========================================================================================
/**
 * @file test_utils_noisegen.cpp
 * @brief Unit tests for noise generation utilities in utils_noisegen.hpp
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// project utils includes
#include <utils/datageneration/utils_noisegen.hpp>

// Eigen includes
#include <Eigen/Dense>


// ========================================================================================

// Embed into test namespace
namespace trex::test::utils::datageneration {

// Namespace alias for convenience
using namespace trex::utils::datageneration::noisegen;

// ========================================================================================

/** @brief Tests for noise policy validation */
TEST(NoiseGenTest, NoisePolicyValidation) {
    EXPECT_NO_THROW(noise_policy::StudentT(5.0));
    EXPECT_THROW(noise_policy::StudentT(-1.0), std::invalid_argument);
    EXPECT_THROW(noise_policy::StudentT(0.0), std::invalid_argument);

    EXPECT_NO_THROW(noise_policy::StudentTMixture(5.0, 2.0));
    EXPECT_THROW(noise_policy::StudentTMixture(-1.0, 2.0), std::invalid_argument);
}


/** @brief Tests for noise parameter calculation */
TEST(NoiseGenTest, CalculateNoiseParams) {
    Eigen::VectorXd y(5);
    // Values: 1, 2, 3, 4, 5. Mean is 3.
    // Squared diffs: (-2)^2 + (-1)^2 + 0 + 1^2 + 2^2 = 4 + 1 + 0 + 1 + 4 = 10
    // signal_power: n <= 100, variance = sum / (n-1) = 10 / 4 = 2.5
    y << 1.0, 2.0, 3.0, 4.0, 5.0;

    double snr = 2.0;
    auto [signal_power, noise_std] = detail::calculate_noise_params(y, 5, snr);

    EXPECT_DOUBLE_EQ(signal_power, 2.5);
    // noise_std = sqrt(signal_power / snr) = sqrt(2.5 / 2.0) = sqrt(1.25)
    EXPECT_DOUBLE_EQ(noise_std, std::sqrt(1.25));

    // Edge cases SNR 0
    auto [sp0, nstd0] = detail::calculate_noise_params(y, 5, 0.0);
    EXPECT_DOUBLE_EQ(nstd0, 0.0); // should safely handle and return 0
}


/** @brief Tests for noise addition determinism */
TEST(NoiseGenTest, AddNoiseDeterminism) {
    Eigen::VectorXd y_base(10);
    y_base.setOnes();

    Eigen::VectorXd y1 = y_base;
    Eigen::VectorXd y2 = y_base;
    Eigen::VectorXd y3 = y_base;

    unsigned int seed1 = 1111;
    unsigned int seed2 = 1111;
    unsigned int seed3 = 2222;

    double noise_std = 1.0;

    // Normal Noise policy
    add_noise(y1, 10, noise_std, noise_policy::Normal(), seed1);
    add_noise(y2, 10, noise_std, noise_policy::Normal(), seed2);
    add_noise(y3, 10, noise_std, noise_policy::Normal(), seed3);

    // Identical seeds yield identical outputs
    EXPECT_TRUE(y1.isApprox(y2, 1e-15));
    // Different seeds yield different outputs
    EXPECT_FALSE(y1.isApprox(y3, 1e-15));
    // Noise was actually added
    EXPECT_FALSE(y1.isApprox(y_base, 1e-15));
}


/** @brief Tests for noise addition with invalid standard deviation */
TEST(NoiseGenTest, AddNoiseThrowsOnNegativeStd) {
    Eigen::VectorXd y(5);
    y.setOnes();

    EXPECT_THROW(add_noise(y, 5, -1.0, noise_policy::Normal(), 123),
                 std::invalid_argument);
    EXPECT_THROW(add_noise(y, 5, 0.0, noise_policy::Normal(), 123),
                 std::invalid_argument);
}

// ========================================================================================
} /* End of namespace trex::test::utils::datageneration */
