#include <gtest/gtest.h>
#include <utils/datageneration/utils_datagen.hpp>
#include <Eigen/Dense>

using namespace trex::utils::datageneration::datagen;
using namespace trex::utils::datageneration;

TEST(DataGenTest, ValidateSupport) {
    std::vector<std::size_t> support = {0, 2};
    std::vector<double> coefs = {1.5, 2.5};

    // Valid: 3 columns, max index is 2
    EXPECT_NO_THROW(detail::validate_support(support, coefs, 3));

    // Invalid: index out of bounds
    EXPECT_THROW(detail::validate_support(support, coefs, 2), std::out_of_range);

    // Invalid: size mismatch
    std::vector<double> bad_coefs = {1.5};
    EXPECT_THROW(detail::validate_support(support, bad_coefs, 3), std::exception);
}

TEST(DataGenTest, PredictorPolicyValidation) {
    EXPECT_NO_THROW(predictor_policy::StudentT(3.0));
    EXPECT_THROW(predictor_policy::StudentT(-1.0), std::exception);

    EXPECT_NO_THROW(predictor_policy::Gumbel(0.0, 1.0));
    EXPECT_THROW(predictor_policy::Gumbel(0.0, -1.0), std::exception);
}

TEST(DataGenTest, SyntheticDataConstruction) {
    std::vector<std::size_t> support = {0, 2};
    std::vector<double> coefs = {1.0, 1.0};
    Eigen::Index n = 50;
    Eigen::Index p = 10;
    Eigen::Index d = 5;

    // With Dummies
    SyntheticData data_w_dummies(
        n, p, d, support, coefs, 2.0, 42
    );

    EXPECT_EQ(data_w_dummies.rows(), 50);
    EXPECT_EQ(data_w_dummies.cols(), 10);
    // SyntheticData actually exposes `getD().cols()` or similar? 
    // Wait, let's just assert getD().cols()
    EXPECT_EQ(data_w_dummies.getD().cols(), 5);
    EXPECT_TRUE(data_w_dummies.hasDummies());
    
    // Check SNR calculation closeness 
    double real_snr = data_w_dummies.snr();
    EXPECT_NEAR(real_snr, 2.0, 1e-1);

    // Without Dummies
    SyntheticData data_no_dummies(
        n, p, support, coefs, 2.0, 42
    );

    EXPECT_EQ(data_no_dummies.rows(), 50);
    EXPECT_EQ(data_no_dummies.cols(), 10);
    EXPECT_EQ(data_no_dummies.getD().cols(), 0);
    EXPECT_FALSE(data_no_dummies.hasDummies());
}

TEST(DataGenTest, SyntheticDataDeterminism) {
    std::vector<std::size_t> support = {1, 3};
    std::vector<double> coefs = {2.0, 3.0};
    
    // Identical seeds
    SyntheticData data1(20, 5, 2, support, coefs, 1.5, 999);
    SyntheticData data2(20, 5, 2, support, coefs, 1.5, 999);


    EXPECT_TRUE(data1.getX().isApprox(data2.getX(), 1e-15));
    EXPECT_TRUE(data1.getD().isApprox(data2.getD(), 1e-15));
    EXPECT_TRUE(data1.getY().isApprox(data2.getY(), 1e-15));

    // Different seeds
    SyntheticData data3(20, 5, 2, support, coefs, 1.5, 888);
    EXPECT_FALSE(data1.getX().isApprox(data3.getX(), 1e-15));
    EXPECT_FALSE(data1.getY().isApprox(data3.getY(), 1e-15));
}

TEST(DataGenTest, SyntheticDataValidation) {
    auto pred_policy = trex::utils::datageneration::datagen::predictor_policy::Normal();
    auto noise_policy = trex::utils::datageneration::noisegen::noise_policy::Normal();

    // n=0 should throw
    EXPECT_THROW((trex::utils::datageneration::datagen::SyntheticData(
        0, 10, 0, {0, 1}, {1.0, 1.0}, 1.0, 42, -1, -1, pred_policy, trex::utils::datageneration::dummygen::Distribution::Normal(), noise_policy)), std::exception);

    // p=0 should throw
    EXPECT_THROW((trex::utils::datageneration::datagen::SyntheticData(
        10, 0, 0, {0, 1}, {1.0, 1.0}, 1.0, 42, -1, -1, pred_policy, trex::utils::datageneration::dummygen::Distribution::Normal(), noise_policy)), std::exception);
}

TEST(DataGenTest, AR1CorrelationTest) {
    auto pred_policy = trex::utils::datageneration::datagen::predictor_policy::AR1(0.9);
    auto noise_policy = trex::utils::datageneration::noisegen::noise_policy::Normal();

    std::size_t n = 10000;
    std::size_t p = 5;
    std::size_t num_dummies = 0;
    double snr = 1.0;
    unsigned int seed = 123;

    trex::utils::datageneration::datagen::SyntheticData data(
            n, p, num_dummies, {0, 1}, {1.0, 1.0}, snr, seed, -1, -1, pred_policy,
            trex::utils::datageneration::dummygen::Distribution::Normal(), noise_policy);

    const auto& X = data.getX();

    // Calculate empirical correlation between column 0 and column 1
    double mean_0 = X.col(0).mean();
    double mean_1 = X.col(1).mean();

    Eigen::VectorXd zero_centered = X.col(0).array() - mean_0;
    Eigen::VectorXd one_centered = X.col(1).array() - mean_1;

    double cov = (zero_centered.array() * one_centered.array()).mean();
    double var_0 = (zero_centered.array() * zero_centered.array()).mean();
    double var_1 = (one_centered.array() * one_centered.array()).mean();

    double correlation = cov / std::sqrt(var_0 * var_1);

    // The theoretical AR1 correlation is 0.9.
    // 10000 samples is well enough to hit Within 0.05
    EXPECT_NEAR(correlation, 0.9, 0.05);
}
