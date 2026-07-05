// ========================================================================================
/**
 * @file test_utils_dummygen.cpp
 *
 * @brief Unit tests for dummy generator utilities in utils_dummygen.hpp
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <numbers>

// project utils includes
#include <utils/datageneration/utils_dummygen.hpp>

// Eigen includes
#include <Eigen/Dense>

// ========================================================================================

// Embed into test namespace
namespace trex::test::utils::datageneration {

// Namespace alias for convenience
using namespace trex::utils::datageneration::dummygen;

// ========================================================================================

/** @brief Unit tests for mix_seed function. */
TEST(DummyGenTest, MixSeedDeterminism) {
    // Tests that mix_seed is deterministic and unique for different bases/indices
    std::uint32_t seed1 = mix_seed(42, 0);
    std::uint32_t seed2 = mix_seed(42, 0);
    std::uint32_t seed3 = mix_seed(42, 1);
    std::uint32_t seed4 = mix_seed(43, 0);

    EXPECT_EQ(seed1, seed2) << "Same inputs must produce the same seed";
    EXPECT_NE(seed1, seed3) << "Different indices must produce different seeds";
    EXPECT_NE(seed1, seed4) << "Different bases must produce different seeds";
}


/** @brief Unit tests for distribution factories. */
TEST(DummyGenTest, DistributionFactories) {
    // Normal Check
    auto norm = Distribution::Normal();
    EXPECT_EQ(norm.type, Distribution::Type::Normal);

    // Uniform check (valid and invalid)
    auto uni = Distribution::Uniform(2.0);
    EXPECT_EQ(uni.type, Distribution::Type::Uniform);
    EXPECT_DOUBLE_EQ(uni.uniform_a, 2.0);
    EXPECT_THROW(Distribution::Uniform(-1.0), std::invalid_argument);

    // Student T check
    auto t = Distribution::StudentT(4.0);
    EXPECT_EQ(t.type, Distribution::Type::StudentT);
    EXPECT_DOUBLE_EQ(t.student_t_df, 4.0);
    EXPECT_THROW(Distribution::StudentT(0.0), std::invalid_argument);

    // Laplace check
    auto lap = Distribution::Laplace(1.0, 2.0);
    EXPECT_EQ(lap.type, Distribution::Type::Laplace);
    EXPECT_DOUBLE_EQ(lap.laplace_location, 1.0);
    EXPECT_DOUBLE_EQ(lap.laplace_scale, 2.0);
    EXPECT_THROW(Distribution::Laplace(0.0, -1.0), std::invalid_argument);

    // Constrained Sparse Rademacher check
    auto csr = Distribution::ConstrainedSparseRademacher(0.5);
    EXPECT_EQ(csr.type, Distribution::Type::ConstrainedSparseRademacher);
    EXPECT_DOUBLE_EQ(csr.constrained_sparse_rademacher_s, 0.5);
    EXPECT_THROW(Distribution::ConstrainedSparseRademacher(1.5), std::invalid_argument);
    EXPECT_THROW(Distribution::ConstrainedSparseRademacher(-0.1), std::invalid_argument);
}


/** @brief Unit tests for Rademacher dummy generation bounds. */
TEST(DummyGenTest, GenerateDummiesRademacherBounds) {
    Eigen::MatrixXd D(100, 5);
    unsigned int seed = 12345;

    generate_dummies(D, 100, 5, seed, Distribution::Rademacher());

    // Rademacher should only be exactly -1.0 or 1.0
    for(int i = 0; i < D.rows(); ++i) {
        for(int j = 0; j < D.cols(); ++j) {
            EXPECT_TRUE(D(i, j) == 1.0 || D(i, j) == -1.0);
        }
    }
}


/** @brief Unit tests for Normal dummy generation properties. */
TEST(DummyGenTest, GenerateDummiesDeterminism) {
    Eigen::MatrixXd D1(10, 3);
    Eigen::MatrixXd D2(10, 3);
    Eigen::MatrixXd D3(10, 3);

    unsigned int seed1 = 999;
    unsigned int seed2 = 999;
    unsigned int seed3 = 888;

    auto dist = Distribution::Normal();

    generate_dummies(D1, 10, 3, seed1, dist);
    generate_dummies(D2, 10, 3, seed2, dist);
    generate_dummies(D3, 10, 3, seed3, dist);

    // Same seed -> completely identical matrix
    EXPECT_TRUE(D1.isApprox(D2, 1e-15));
    // Different seed -> matrix should be different
    EXPECT_FALSE(D1.isApprox(D3, 1e-15));
}

/** @brief Unit tests for Gumbel dummy centering and variance. */
TEST(DummyGenTest, GenerateDummiesGumbelCentered) {
    constexpr std::size_t n = 20000;
    constexpr std::size_t p = 4;
    Eigen::MatrixXd D(n, p);

    // Default parameters: samples must be centered at 0 with
    // variance pi^2/6 * scale^2
    generate_dummies(D, n, p, 4242, Distribution::Gumbel());
    EXPECT_NEAR(D.mean(), 0.0, 0.05);
    double var = (D.array() - D.mean()).square().sum()
                 / static_cast<double>(n * p - 1);
    EXPECT_NEAR(var, std::numbers::pi * std::numbers::pi / 6.0, 0.1);

    // Non-default location/scale: mean must equal the requested location
    generate_dummies(D, n, p, 4242, Distribution::Gumbel(1.0, 2.0));
    EXPECT_NEAR(D.mean(), 1.0, 0.1);
}


/** @brief Unit tests for constrained sparse Rademacher balance and small-n bounds. */
TEST(DummyGenTest, GenerateDummiesConstrainedSparseRademacher) {
    constexpr std::size_t n = 100;
    constexpr std::size_t p = 3;
    Eigen::MatrixXd D(n, p);

    generate_dummies(D, n, p, 7, Distribution::ConstrainedSparseRademacher(0.1));

    for (std::size_t j = 0; j < p; ++j) {
        // k = 2 * floor(n * s / 2) = 10 non-zeros, balanced +1/-1
        EXPECT_EQ((D.col(j).array() == 1.0).count(), 5);
        EXPECT_EQ((D.col(j).array() == -1.0).count(), 5);
        EXPECT_EQ((D.col(j).array() == 0.0).count(), 90);
    }

    // Degenerate n = 1: minimum-non-zeros bump must not write out of bounds
    Eigen::MatrixXd D_tiny(1, 2);
    EXPECT_NO_THROW(generate_dummies(D_tiny, 1, 2, 7,
        Distribution::ConstrainedSparseRademacher(0.5)));
    EXPECT_TRUE((D_tiny.array() == 0.0).all());
}


// ========================================================================================
} /* End of namespace trex::test::utils::datageneration */
