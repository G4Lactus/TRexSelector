// ========================================================================================
// test_trex_spca.cpp
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <algorithm>
#include <random>
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
    static constexpr int          seed = 42;

    Eigen::MatrixXd X;
    Eigen::Map<Eigen::MatrixXd> X_map;

    /** @brief Returns a TRexSPCAControlParameter tuned for fast unit testing. */
    static TRexSPCAControlParameter fast_ctrl(SPCAMode mode = SPCAMode::ActiveSet) {
        TRexSPCAControlParameter ctrl;
        ctrl.mode                           = mode;
        ctrl.lambda2_ridge_loadings         = 1e-6;
        ctrl.gvs_ctrl.trex_ctrl.K                    = 3;
        ctrl.gvs_ctrl.trex_ctrl.max_dummy_multiplier = 2;
        return ctrl;
    }

    TRexSPCATest()
        : X(Eigen::MatrixXd::Random(n, p)),
          X_map(X.data(), n, p) {}

    /** @brief Deterministic single-factor design engineered so the GVS active
     *         set of PC1 and the top-|A| ordinary-loading support diverge.
     *
     *  Ordinary loadings rank variables by covariance with z_1 (X^T z_1 is
     *  proportional to v_1), while the GVS sub-selector ranks by normalized
     *  correlation. The design therefore plants, next to the strong factor
     *  columns {0, 1, 2}:
     *    - columns {3, 4, 5}: tiny norm, near-unit correlation with the factor
     *      (GVS selects them, but their ordinary loadings are small), and
     *    - columns {6, 7, 8}: large norm, weak correlation (GVS rejects them,
     *      but their ordinary loadings can outrank columns {3, 4, 5}).
     *  The remaining columns are pure noise, so that the FDR-controlled active
     *  set is a proper subset of {1, ..., p} (with very small p the selector
     *  admits every column and the two supports trivially coincide).
     *  Uses its own RNG rather than Eigen::Random so the design is independent
     *  of test order. */
    static constexpr Eigen::Index n_f = 40;
    static constexpr Eigen::Index p_f = 20;

    static Eigen::MatrixXd plantedFactorX() {
        std::mt19937 rng(7u);
        std::normal_distribution<double> N01(0.0, 1.0);

        Eigen::VectorXd z(n_f);
        for (Eigen::Index i = 0; i < n_f; ++i) {
            z(i) = 3.0 * N01(rng);
        }

        Eigen::MatrixXd Xf(n_f, p_f);
        for (Eigen::Index j = 0; j < p_f; ++j) {
            for (Eigen::Index i = 0; i < n_f; ++i) {
                Xf(i, j) = N01(rng);
            }
        }
        for (Eigen::Index j = 0; j < 3; ++j) {
            Xf.col(j) = 0.9 * z + 0.3 * Xf.col(j);   // strong factor columns
        }
        for (Eigen::Index j = 3; j < 6; ++j) {
            Xf.col(j) = 0.05 * z + 0.02 * Xf.col(j); // tiny norm, corr ~ 1
        }
        for (Eigen::Index j = 6; j < 9; ++j) {
            Xf.col(j) = 0.05 * z + 1.5 * Xf.col(j);  // large norm, weak corr
        }
        return Xf;                                   // cols 9..19: pure noise
    }

    /** @brief Shared invariant check: after select(), Z must equal
     *         X_centered * V (scores computed over the full support of V, not
     *         merely the GVS active set), and every assembled loading column
     *         must be L2-normalized. */
    static void expectScoresMatchCenteredXTimesV(SPCAMode mode) {
        Eigen::MatrixXd Xf = plantedFactorX();
        Eigen::Map<Eigen::MatrixXd> Xf_map(Xf.data(), n_f, p_f);

        TRexSPCA spca(Xf_map, M, 0.1, fast_ctrl(mode), seed);
        auto result = spca.select();

        // The planted factor must be picked up, otherwise the invariant below
        // would only be checked on the trivial all-zero support.
        ASSERT_FALSE(result.active_sets[0].empty());

        const Eigen::RowVectorXd mu = Xf.colwise().mean();
        const Eigen::MatrixXd Xc = Xf.rowwise() - mu;
        const Eigen::MatrixXd Z_expected = Xc * result.V;

        EXPECT_LE((result.Z - Z_expected).norm(),
                  1e-10 * std::max(1.0, Z_expected.norm()))
            << "Z != X_centered * V for mode "
            << (mode == SPCAMode::ActiveSet ? "ActiveSet" : "Thresholded");

        for (Eigen::Index m = 0; m < M; ++m) {
            const double norm_m = result.V.col(m).norm();
            if (norm_m > 0.0) {
                EXPECT_NEAR(norm_m, 1.0, 1e-12);
            }
        }
    }
};


// ========================================================================================
// Execution — does not throw
// ========================================================================================

/** @brief ActiveSet mode runs without throwing on valid inputs. */
TEST_F(TRexSPCATest, Execution_ActiveSetModeDoesNotThrow) {
    EXPECT_NO_THROW({
        TRexSPCA spca(X_map, M, 0.2, fast_ctrl(SPCAMode::ActiveSet), seed);
        spca.select();
    });
}


/** @brief Thresholded mode runs without throwing on valid inputs. */
TEST_F(TRexSPCATest, Execution_ThresholdedModeDoesNotThrow) {
    EXPECT_NO_THROW({
        TRexSPCA spca(X_map, M, 0.2, fast_ctrl(SPCAMode::Thresholded), seed);
        spca.select();
    });
}


// ========================================================================================
// Output dimensions
// ========================================================================================

/** @brief Result matrices have the expected shapes. */
TEST_F(TRexSPCATest, Dimensions_ResultMatricesCorrectShape) {
    TRexSPCA spca(X_map, M, 0.2, fast_ctrl(), seed);
    auto result = spca.select();

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
    TRexSPCA spca(X_map, 1, 0.2, fast_ctrl(), seed);
    auto result = spca.select();

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
    TRexSPCA spca(X_map, M, 0.2, fast_ctrl(), seed);
    auto result = spca.select();

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
    TRexSPCA spca(X_map, M, 0.2, fast_ctrl(), seed);
    auto result = spca.select();

    for (Eigen::Index k = 0; k < M; ++k) {
        EXPECT_GE(result.adjusted_ev(k), 0.0);
    }
}


/** @brief Cumulative explained variance is monotonically non-decreasing and in [0, 1]. */
TEST_F(TRexSPCATest, ExplainedVariance_CumulativeMonotoneInUnitInterval) {
    TRexSPCA spca(X_map, M, 0.2, fast_ctrl(), seed);
    auto result = spca.select();

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
// Score / loading consistency
// ========================================================================================

/** @brief ActiveSet mode: Z == X_centered * V and V columns are unit-norm. */
TEST_F(TRexSPCATest, Scores_MatchCenteredXTimesV_ActiveSet) {
    expectScoresMatchCenteredXTimesV(SPCAMode::ActiveSet);
}


/** @brief Thresholded mode: Z == X_centered * V even when the thresholded
 *         support of V differs from the GVS active set (regression test for
 *         the score loop formerly iterating over the active set only). */
TEST_F(TRexSPCATest, Scores_MatchCenteredXTimesV_Thresholded) {
    expectScoresMatchCenteredXTimesV(SPCAMode::Thresholded);
}


// ========================================================================================
// Data integrity
// ========================================================================================

/** @brief X is restored to its original values after select() returns. */
TEST_F(TRexSPCATest, DataIntegrity_XRestoredAfterSelect) {
    Eigen::MatrixXd X_copy = X;
    TRexSPCA spca(X_map, M, 0.2, fast_ctrl(), seed);
    spca.select();

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored after TRexSPCA::select().";
}


/** @brief The constructor does not mutate X — centering only happens inside select(). */
TEST_F(TRexSPCATest, DataIntegrity_ConstructorDoesNotMutateX) {
    Eigen::MatrixXd X_copy = X;
    { TRexSPCA spca(X_map, M, 0.2, fast_ctrl(), seed); }

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was mutated by TRexSPCA constructor or destructor without select().";
}


/** @brief Destructor does not mutate X when select() was never called. */
TEST_F(TRexSPCATest, DataIntegrity_XUnchangedOnDestructionWithoutSelect) {
    Eigen::MatrixXd X_copy = X;

    {
        TRexSPCA spca(X_map, M, 0.2, fast_ctrl(), seed);
        // Destructor fires here; X_is_centered_ = false, so no inverse_transform.
    }

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was incorrectly modified by TRexSPCA destructor.";
}


// ========================================================================================
// GVS type recorded in result
// ========================================================================================

// ========================================================================================
// Exception handling
// ========================================================================================

/** @brief M > min(n, p) triggers std::invalid_argument in the constructor. */
TEST_F(TRexSPCATest, Exceptions_InvalidM) {
    EXPECT_THROW(
        (TRexSPCA(X_map, p + 1, 0.2, fast_ctrl(), seed)),
        std::invalid_argument
    );
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_spca */
