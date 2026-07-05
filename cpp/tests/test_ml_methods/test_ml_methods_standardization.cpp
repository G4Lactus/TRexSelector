// ========================================================================================
/**
 * @file test_ml_methods_standardization.cpp
 * @brief Unit tests for standardization techniques in ml_methods.
 */
// ========================================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <cereal/archives/portable_binary.hpp>

#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::scaler_methods {

using namespace trex::ml_methods::scaler_methods;

// ========================================================================================
// ZScoreScaler Tests
// ========================================================================================

/** @brief Test for basic functionality of ZScoreScaler */
TEST(StandardizationTest, ZScoreScalerBasic) {
    Eigen::MatrixXd data(3, 2);
    // Col 0: 1, 2, 3 -> mean=2.0, var=1.0, std=1.0 (with Bessel: sum(sq_diffs) = 2, df = 2)
    // Col 1: 4, 6, 8 -> mean=6.0, var=4.0, std=2.0 (with Bessel: sum(sq_diffs) = 4+4=8, df = 2, var=4)
    data << 1.0, 4.0,
            2.0, 6.0,
            3.0, 8.0;

    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler scaler(true, true);
    scaler.fit(map_data);

    EXPECT_TRUE(scaler.is_fitted());

    // Check parameters
    const auto& means = scaler.get_means();
    const auto& scales = scaler.get_scales();
    EXPECT_DOUBLE_EQ(means(0), 2.0);
    EXPECT_DOUBLE_EQ(means(1), 6.0);
    EXPECT_DOUBLE_EQ(scales(0), 1.0);
    EXPECT_DOUBLE_EQ(scales(1), 2.0);

    // Transform
    scaler.transform_inplace(map_data);

    // Col 0 should be -1, 0, 1
    EXPECT_DOUBLE_EQ(data(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(data(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(data(2, 0), 1.0);

    // Col 1 should be -1, 0, 1
    EXPECT_DOUBLE_EQ(data(0, 1), -1.0);
    EXPECT_DOUBLE_EQ(data(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(data(2, 1), 1.0);

    // Inverse transform
    scaler.inverse_transform_inplace(map_data);
    EXPECT_DOUBLE_EQ(data(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(data(2, 1), 8.0);
}


/**** @brief Test for ZScoreScaler with no centering and only scaling */
TEST(StandardizationTest, ZScoreScalerNoCentering) {
    Eigen::MatrixXd data(3, 1);
    data << 1.0, 3.0, 5.0; // std = 2.0 (bessel: sq_diff(2)=4, sq_diff(-2)=4 -> sum 8. 8/2 = 4 var, std = 2)
    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler scaler(false, true); // no mean, yes std
    scaler.fit(map_data);

    const auto& means = scaler.get_means();
    const auto& scales = scaler.get_scales();

    // If no centering, the mean recorded by the scaler is 0.0 internally if with_mean_ = false
    // But wait, the standard deviation WITHOUT mean subtraction is computed as:
    // variance = sum(X_i^2) / (n-1) = (1 + 9 + 25) / 2 = 35 / 2 = 17.5
    // std_val = sqrt(17.5) = 4.1833001326703778
    EXPECT_DOUBLE_EQ(means(0), 0.0);
    EXPECT_NEAR(scales(0), 4.1833001326703778, 1e-12);

    scaler.transform_inplace(map_data);

    EXPECT_NEAR(data(0, 0), 1.0 / 4.1833001326703778, 1e-12);
    EXPECT_NEAR(data(1, 0), 3.0 / 4.1833001326703778, 1e-12);
    EXPECT_NEAR(data(2, 0), 5.0 / 4.1833001326703778, 1e-12);
}

/** @brief Test for ZScoreScaler with a constant column (should handle zero std) */
TEST(StandardizationTest, ZScoreScalerConstantColumn) {
    Eigen::MatrixXd data(3, 1);
    data << 42.0, 42.0, 42.0;
    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler scaler(true, true);
    scaler.fit(map_data);

    // Constant column has 0.0 std, so scale should be set to 1.0 internally.
    // It should also be marked as a dropped index.
    const auto& scales = scaler.get_scales();
    EXPECT_DOUBLE_EQ(scales(0), 1.0);
    EXPECT_EQ(scaler.get_dropped_indices().size(), 1);
    EXPECT_EQ(scaler.get_dropped_indices()[0], 0);

    scaler.transform_inplace(map_data);
    // Because it is dropped, transform_inplace just skips the column entirely.
    EXPECT_DOUBLE_EQ(data(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(data(1, 0), 42.0);
    EXPECT_DOUBLE_EQ(data(2, 0), 42.0);
}


// ========================================================================================
// LpNormScaler Tests
// ========================================================================================

/** @brief Test for LpNormScaler with L2 norm */
TEST(StandardizationTest, LpNormScalerL2) {
    Eigen::MatrixXd data(3, 1);
    data << 1.0, 2.0, 3.0; // mean = 2, centered = -1, 0, 1 -> L2 norm = sqrt(1+1) = sqrt(2) = 1.414213562

    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    LpNormScaler scaler(LpNormScaler::NormType::L2, true, true);
    scaler.fit(map_data);

    const auto& means = scaler.get_means();
    const auto& scales = scaler.get_scales();

    EXPECT_DOUBLE_EQ(means(0), 2.0);
    EXPECT_DOUBLE_EQ(scales(0), std::sqrt(2.0));

    scaler.transform_inplace(map_data);

    EXPECT_DOUBLE_EQ(data(0, 0), -1.0 / std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(data(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(data(2, 0), 1.0 / std::sqrt(2.0));

    // Inverse
    scaler.inverse_transform_inplace(map_data);
    EXPECT_DOUBLE_EQ(data(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(data(2, 0), 3.0);
}


/** @brief Test for LpNormScaler with L1 norm */
TEST(StandardizationTest, LpNormScalerL1) {
    Eigen::MatrixXd data(3, 1);
    data << 1.0, 2.0, 3.0; // mean = 2, centered = -1, 0, 1 -> L1 norm = |-1| + 0 + |1| = 2.0

    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    LpNormScaler scaler(LpNormScaler::NormType::L1, true, true);
    scaler.fit(map_data);

    const auto& scales = scaler.get_scales();
    EXPECT_DOUBLE_EQ(scales(0), 2.0);

    scaler.transform_inplace(map_data);

    EXPECT_DOUBLE_EQ(data(0, 0), -0.5);
    EXPECT_DOUBLE_EQ(data(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(data(2, 0), 0.5);
}


/** @brief Test for LpNormScaler with no centering and only scaling */
TEST(StandardizationTest, LpNormScalerNoCenteringL2) {
    Eigen::MatrixXd data(3, 1);
    data << 3.0, 4.0, 0.0; // Norm is sqrt(9 + 16 + 0) = 5

    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    LpNormScaler scaler(LpNormScaler::NormType::L2, false, true); // No mean
    scaler.fit(map_data);

    EXPECT_DOUBLE_EQ(scaler.get_means()(0), 0.0);
    EXPECT_DOUBLE_EQ(scaler.get_scales()(0), 5.0);

    scaler.transform_inplace(map_data);

    EXPECT_DOUBLE_EQ(data(0, 0), 0.6);
    EXPECT_DOUBLE_EQ(data(1, 0), 0.8);
    EXPECT_DOUBLE_EQ(data(2, 0), 0.0);
}


/** @brief Test for LpNormScaler with a constant column (should handle zero norm) */
TEST(StandardizationTest, LpNormScalerConstantColumn) {
    Eigen::MatrixXd data(3, 1);
    data << 5.0, 5.0, 5.0; // mean = 5. Centered: 0, 0, 0 -> Norm = 0. Scale should be 1.0.

    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    LpNormScaler scaler(LpNormScaler::NormType::L2, true, true);
    scaler.fit(map_data);

    // Constant column means centered values are 0, norm expands to 0.
    // The scaler should catch this, set scale to 1.0, and add to dropped indices.
    EXPECT_DOUBLE_EQ(scaler.get_scales()(0), 1.0);
    EXPECT_EQ(scaler.get_dropped_indices().size(), 1);
    EXPECT_EQ(scaler.get_dropped_indices()[0], 0);

    scaler.transform_inplace(map_data);

    // Because it is dropped, transform_inplace skips it.
    EXPECT_DOUBLE_EQ(data(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(data(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(data(2, 0), 5.0);
}


// ========================================================================================
// Error handling
// ========================================================================================

/** @brief Transform/inverse before fit and column-count mismatches must throw. */
TEST(StandardizationTest, NotFittedAndDimensionMismatchThrow) {
    Eigen::MatrixXd data(3, 2);
    data << 1.0, 4.0,
            2.0, 6.0,
            3.0, 8.0;
    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler z_scaler;
    EXPECT_THROW(z_scaler.transform_inplace(map_data), std::logic_error);
    EXPECT_THROW(z_scaler.inverse_transform_inplace(map_data), std::logic_error);

    LpNormScaler lp_scaler;
    EXPECT_THROW(lp_scaler.transform_inplace(map_data), std::logic_error);
    EXPECT_THROW(lp_scaler.inverse_transform_inplace(map_data), std::logic_error);

    // Fitted on 2 columns, applied to 3 columns -> invalid_argument
    z_scaler.fit(map_data);
    Eigen::MatrixXd wide(3, 3);
    wide.setOnes();
    Eigen::Map<Eigen::MatrixXd> map_wide(wide.data(), wide.rows(), wide.cols());
    EXPECT_THROW(z_scaler.transform_inplace(map_wide), std::invalid_argument);
    EXPECT_THROW(z_scaler.inverse_transform_inplace(map_wide), std::invalid_argument);
}


/**
 * @brief A single sample makes the Bessel-corrected standard deviation undefined
 *        (division by n - 1 = 0): fit must throw instead of emitting NaN scales.
 */
TEST(StandardizationTest, ZScoreScalerSingleSampleThrows) {
    Eigen::MatrixXd data(1, 3);
    data << 1.0, 2.0, 3.0;
    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler with_std(true, true);
    EXPECT_THROW(with_std.fit(map_data), std::invalid_argument);

    // Mean-only scaling has no n - 1 division and stays valid for a single sample.
    ZScoreScaler mean_only(true, false);
    EXPECT_NO_THROW(mean_only.fit(map_data));
}


/**
 * @brief Multi-column round trip with a constant (dropped) column in the middle:
 *        live columns standardise to mean 0 / unit SD and invert exactly; the
 *        dropped column passes through untouched.
 */
TEST(StandardizationTest, MixedColumnsRoundTrip) {
    const Eigen::Index n = 5;
    Eigen::MatrixXd data(n, 3);
    data.col(0) = Eigen::VectorXd::LinSpaced(n, -2.0, 6.0);
    data.col(1) = Eigen::VectorXd::Constant(n, 7.0);   // constant -> dropped
    data.col(2) = Eigen::VectorXd::LinSpaced(n, 10.0, 2.0);
    Eigen::MatrixXd original = data;

    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler scaler(true, true);
    scaler.fit(map_data);
    ASSERT_EQ(scaler.get_dropped_indices().size(), 1);
    EXPECT_EQ(scaler.get_dropped_indices()[0], 1);

    scaler.transform_inplace(map_data);

    // Live columns: mean 0, Bessel-corrected SD 1
    for (Eigen::Index j : {Eigen::Index{0}, Eigen::Index{2}}) {
        EXPECT_NEAR(data.col(j).mean(), 0.0, 1e-12);
        const double sd = std::sqrt(data.col(j).squaredNorm()
                                    / static_cast<double>(n - 1));
        EXPECT_NEAR(sd, 1.0, 1e-12);
    }
    // Dropped column untouched
    EXPECT_TRUE(data.col(1).isApprox(original.col(1), 0.0));

    scaler.inverse_transform_inplace(map_data);
    EXPECT_TRUE(data.isApprox(original, 1e-12));
}


// ========================================================================================
// Serialization Tests
// ========================================================================================

/** @brief Test for ZScoreScaler serialization and deserialization */
TEST(StandardizationTest, ZScoreScalerSerialization) {
    Eigen::MatrixXd data(3, 2);
    data << 1.0, 4.0,
            2.0, 6.0,
            3.0, 8.0;

    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler original_scaler(true, true);
    original_scaler.fit(map_data);

    std::stringstream ss;
    {
        cereal::PortableBinaryOutputArchive oarchive(ss);
        oarchive(original_scaler);
    }

    ZScoreScaler restored_scaler(false, false);
    {
        cereal::PortableBinaryInputArchive iarchive(ss);
        iarchive(restored_scaler);
    }

    EXPECT_TRUE(restored_scaler.is_fitted());
    EXPECT_EQ(original_scaler.get_means(), restored_scaler.get_means());
    EXPECT_EQ(original_scaler.get_scales(), restored_scaler.get_scales());
    EXPECT_EQ(original_scaler.get_dropped_indices(), restored_scaler.get_dropped_indices());
}


/** @brief Test for LpNormScaler serialization and deserialization */
TEST(StandardizationTest, LpNormScalerSerialization) {
    Eigen::MatrixXd data(3, 1);
    data << 1.0, 2.0, 3.0;
    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    LpNormScaler original_scaler(LpNormScaler::NormType::L2, true, true);
    original_scaler.fit(map_data);

    std::stringstream ss;
    {
        cereal::PortableBinaryOutputArchive oarchive(ss);
        oarchive(original_scaler);
    }

    LpNormScaler restored_scaler(LpNormScaler::NormType::L1, false, false);
    {
        cereal::PortableBinaryInputArchive iarchive(ss);
        iarchive(restored_scaler);
    }

    EXPECT_TRUE(restored_scaler.is_fitted());
    EXPECT_EQ(original_scaler.get_means(), restored_scaler.get_means());
    EXPECT_EQ(original_scaler.get_scales(), restored_scaler.get_scales());
    EXPECT_EQ(original_scaler.get_dropped_indices(), restored_scaler.get_dropped_indices());
}

/**
 * @brief The public save()/load() file API: round trip restores the fitted state,
 *        the type-name guard rejects archives written by a different scaler type,
 *        and loading a missing file throws.
 */
TEST(StandardizationTest, SaveLoadFileRoundTripAndTypeMismatch) {
    Eigen::MatrixXd data(4, 2);
    data << 1.0,  4.0,
            2.0,  6.0,
            3.0,  8.0,
            6.0, 12.0;
    Eigen::Map<Eigen::MatrixXd> map_data(data.data(), data.rows(), data.cols());

    ZScoreScaler original(true, true);
    original.fit(map_data);

    const std::string path = ::testing::TempDir() + "trex_zscore_scaler_test.bin";
    original.save(path);

    // Round trip through the file restores the fitted state exactly.
    ZScoreScaler restored(false, false);
    restored.load(path);
    EXPECT_TRUE(restored.is_fitted());
    EXPECT_EQ(original.get_means(), restored.get_means());
    EXPECT_EQ(original.get_scales(), restored.get_scales());
    EXPECT_EQ(original.get_dropped_indices(), restored.get_dropped_indices());
    EXPECT_EQ(restored.get_with_mean(), original.get_with_mean());
    EXPECT_EQ(restored.get_with_std(), original.get_with_std());

    // The restored scaler must transform identically to the original.
    Eigen::MatrixXd a = data, b = data;
    Eigen::Map<Eigen::MatrixXd> map_a(a.data(), a.rows(), a.cols());
    Eigen::Map<Eigen::MatrixXd> map_b(b.data(), b.rows(), b.cols());
    original.transform_inplace(map_a);
    restored.transform_inplace(map_b);
    EXPECT_TRUE(a.isApprox(b, 0.0));

    // Type-name guard: a ZScoreScaler archive must be rejected by LpNormScaler.
    LpNormScaler wrong_type;
    EXPECT_THROW(wrong_type.load(path), std::runtime_error);

    // Missing file throws.
    ZScoreScaler no_file;
    EXPECT_THROW(no_file.load(path + ".does_not_exist"), std::runtime_error);

    std::remove(path.c_str());
}

// ========================================================================================
} /* End of namespace trex::test::ml_methods::scaler_methods */
