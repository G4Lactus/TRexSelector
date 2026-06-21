// ========================================================================================
// test_trex_spca.cpp
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <stdexcept>
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_spca/trex_spca.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_spca {

// Namespace alias for convenience
using namespace trex::trex_selector_methods::trex_spca;
namespace tc = trex::trex_selector_methods::trex_core;

// ========================================================================================

/**
 * @brief Test fixture providing a minimal (n=30, p=10) design matrix shared
 *        across all TRexSPCA tests.
 *
 * @details Dimensions are kept deliberately small so that the gtest_discover_tests
 *          post-build enumeration step (5 s limit) is never exceeded even under
 *          a heavily-loaded parallel build. A fixed seed ensures reproducibility.
 */
class TRexSPCATest : public ::testing::Test {
protected:
    static constexpr Eigen::Index n = 30;
    static constexpr Eigen::Index p = 10;
    static constexpr Eigen::Index M = 2;

    Eigen::MatrixXd X;
    Eigen::Map<Eigen::MatrixXd> X_map;

    /** @brief Returns a TRexSPCAControlParameter tuned for fast unit testing. */
    static TRexSPCAControlParameter fast_ctrl(SPCAMode mode = SPCAMode::ActiveSet) {
        TRexSPCAControlParameter ctrl;
        ctrl.mode          = mode;
        ctrl.lambda2       = 1e-6;
        ctrl.seed          = 42;
        ctrl.trex_ctrl.K   = 3;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        return ctrl;
    }

    TRexSPCATest()
        : X(Eigen::MatrixXd::Random(n, p)),
          X_map(X.data(), n, p) {}
};


// ========================================================================================
// Execution — does not throw
// ========================================================================================

/** @brief ActiveSet mode runs without throwing on valid inputs. */
TEST_F(TRexSPCATest, Execution_ActiveSetModeDoesNotThrow) {
    EXPECT_NO_THROW({
        TRexSPCA::select(X_map, M, 0.2, fast_ctrl(SPCAMode::ActiveSet));
    });
}


/** @brief Thresholded mode runs without throwing on valid inputs. */
TEST_F(TRexSPCATest, Execution_ThresholdedModeDoesNotThrow) {
    EXPECT_NO_THROW({
        TRexSPCA::select(X_map, M, 0.2, fast_ctrl(SPCAMode::Thresholded));
    });
}


// ========================================================================================
// Output dimensions
// ========================================================================================

/** @brief Result matrices have the expected shapes. */
TEST_F(TRexSPCATest, Dimensions_ResultMatricesCorrectShape) {
    auto result = TRexSPCA::select(
        X_map, M, 0.2, fast_ctrl()
    );

    EXPECT_EQ(result.Z.rows(), n);
    EXPECT_EQ(result.Z.cols(), M);
    EXPECT_EQ(result.V.rows(), p);
    EXPECT_EQ(result.V.cols(), M);
    EXPECT_EQ(result.active_sets.size(), static_cast<std::size_t>(M));
    EXPECT_EQ(result.adjusted_ev.size(),  M);
    EXPECT_EQ(result.cumulative_ev.size(), M);
}


/** @brief Extracting only one sparse PC (M=1) produces correct shapes. */
TEST_F(TRexSPCATest, Dimensions_SingleComponentCorrectShape) {
    auto result = TRexSPCA::select(
        X_map, 1, 0.2, fast_ctrl()
    );

    EXPECT_EQ(result.Z.rows(), n);
    EXPECT_EQ(result.Z.cols(), 1);
    EXPECT_EQ(result.V.rows(), p);
    EXPECT_EQ(result.V.cols(), 1);
    EXPECT_EQ(result.active_sets.size(), 1u);
}


// ========================================================================================
// Active sets
// ========================================================================================

/** @brief All active-set indices are valid column indices of X. */
TEST_F(TRexSPCATest, ActiveSets_IndicesInBounds) {
    auto result = TRexSPCA::select(
        X_map, M, 0.2, fast_ctrl()
    );

    for (const auto& as : result.active_sets) {
        for (Eigen::Index idx : as) {
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, p);
        }
    }
}


// ========================================================================================
// Explained variance
// ========================================================================================

/** @brief Adjusted explained variances are non-negative. */
TEST_F(TRexSPCATest, ExplainedVariance_AdjustedNonNegative) {
    auto result = TRexSPCA::select(
        X_map, M, 0.2, fast_ctrl()
    );

    for (Eigen::Index k = 0; k < M; ++k) {
        EXPECT_GE(result.adjusted_ev(k), 0.0);
    }
}


/** @brief Cumulative explained variance is monotonically non-decreasing and in [0, 1]. */
TEST_F(TRexSPCATest, ExplainedVariance_CumulativeMonotoneInUnitInterval) {
    auto result = TRexSPCA::select(
        X_map, M, 0.2, fast_ctrl()
    );

    double prev = 0.0;
    for (Eigen::Index k = 0; k < M; ++k) {
        double cum = result.cumulative_ev(k);
        EXPECT_GE(cum, prev - 1e-12); // non-decreasing (with numerical tolerance)
        EXPECT_GE(cum, 0.0);
        EXPECT_LE(cum, 1.0 + 1e-12);
        prev = cum;
    }
}


// ========================================================================================
// Data integrity
// ========================================================================================

/** @brief X is restored to its original values after select() returns. */
TEST_F(TRexSPCATest, DataIntegrity_XRestoredAfterSelect) {
    Eigen::MatrixXd X_copy = X;
    TRexSPCA::select(
        X_map, M, 0.2, fast_ctrl()
    );

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored after TRexSPCA::select().";
}


// ========================================================================================
// Exception handling
// ========================================================================================

/** @brief M > min(n, p) triggers std::invalid_argument. */
TEST_F(TRexSPCATest, Exceptions_InvalidM) {
    TRexSPCAControlParameter ctrl = fast_ctrl();
    EXPECT_THROW(TRexSPCA::select(
        X_map, p + 1, 0.2, ctrl
    ), std::invalid_argument);
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_spca */
