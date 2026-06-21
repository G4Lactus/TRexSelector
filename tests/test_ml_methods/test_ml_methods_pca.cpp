// ========================================================================================
/**
 * @file test_ml_methods_pca.cpp
 * @brief Unit tests for the PCA class.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <stdexcept>

// project ml_methods includes
#include <ml_methods/pca/pca.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::pca {

using namespace trex::ml_methods::pca;

// ========================================================================================
// Output dimensions
// ========================================================================================

/** @brief Test fit() returns results with correct shapes. */
TEST(PCATest, Dimensions_ResultShapes) {
    const Eigen::Index n = 50, p = 20, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    PCA pca;
    PCAResult res = pca.fit(X, M);

    EXPECT_EQ(res.Z.rows(), n);
    EXPECT_EQ(res.Z.cols(), M);
    EXPECT_EQ(res.V.rows(), p);
    EXPECT_EQ(res.V.cols(), M);
    EXPECT_EQ(res.explained_variance.size(), M);
}


/** @brief Test getLoadings() and getMean() are accessible after fit(). */
TEST(PCATest, Dimensions_GetterShapesAfterFit) {
    const Eigen::Index n = 40, p = 15, M = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    PCA pca;
    pca.fit(X, M);

    EXPECT_EQ(pca.getLoadings().rows(), p);
    EXPECT_EQ(pca.getLoadings().cols(), M);
    EXPECT_EQ(pca.getMean().size(), p);
    EXPECT_EQ(pca.getExplainedVariance().size(), M);
}


// ========================================================================================
// Centering
// ========================================================================================

/**
 * @brief Test when center=true, fit() subtracts column means from X in-place.
 *        The stored mean equals the original column means of X.
 */
TEST(PCATest, Centering_MeanStoredCorrectly) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::RowVectorXd expected_mean = X.colwise().mean();

    PCA pca;
    pca.fit(X, M);

    EXPECT_TRUE(pca.getMean().isApprox(expected_mean, 1e-12));
}


/** @brief Test when center=false, getMean() returns a zero vector. */
TEST(PCATest, Centering_NoCenterMeanIsZero) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    // Pre-center manually
    X.rowwise() -= X.colwise().mean();

    PCA pca(false);
    pca.fit(X, M);

    EXPECT_TRUE(
        pca.getMean().isApprox(Eigen::RowVectorXd::Zero(p), 1e-12)
    );
}


// ========================================================================================
// RAII restoration
// ========================================================================================

/** @brief Test X is restored to its original values when the PCA object is destroyed. */
TEST(PCATest, RAII_XRestoredOnDestruction) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_original = X;

    {
        PCA pca;
        pca.fit(X, M);
        // X is centered here — not equal to original
    } // pca destroyed → X should be restored

    EXPECT_TRUE(X.isApprox(X_original, 1e-12))
        << "X was not restored after PCA went out of scope.";
}


/** @brief Test restore() can be called explicitly and is idempotent. */
TEST(PCATest, RAII_ExplicitRestoreIsIdempotent) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_original = X;

    PCA pca;
    pca.fit(X, M);

    pca.restore();
    EXPECT_TRUE(X.isApprox(X_original, 1e-12));

    // Second call should be safe (no double-restoration)
    EXPECT_NO_THROW(pca.restore());
    EXPECT_TRUE(X.isApprox(X_original, 1e-12));
}


/** @brief Test center=false: X is not modified or restored (no RAII side-effects). */
TEST(PCATest, RAII_NoCenterDoesNotModifyX) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.rowwise() -= X.colwise().mean(); // pre-center
    Eigen::MatrixXd X_snapshot = X;

    {
        PCA pca(false);
        pca.fit(X, M);
    }

    EXPECT_TRUE(X.isApprox(X_snapshot, 1e-12))
        << "X was modified despite center=false.";
}


// ========================================================================================
// Loadings orthogonality
// ========================================================================================

/** @brief Test Loadings are orthonormal: V^T V ≈ I_M. */
TEST(PCATest, Loadings_Orthonormal) {
    const Eigen::Index n = 50, p = 20, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    PCA pca;
    PCAResult res = pca.fit(X, M);

    Eigen::MatrixXd VtV = res.V.transpose() * res.V;
    EXPECT_TRUE(
        VtV.isApprox(Eigen::MatrixXd::Identity(M, M), 1e-10)
    ) << "Loading matrix V is not orthonormal.";
}


// ========================================================================================
// Explained variance
// ========================================================================================

/** @brief Test explained variances are non-negative and non-increasing. */
TEST(PCATest, ExplainedVariance_NonNegativeAndNonIncreasing) {
    const Eigen::Index n = 50, p = 20, M = 6;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    PCA pca;
    PCAResult res = pca.fit(X, M);

    for (Eigen::Index k = 0; k < M; ++k) {
        EXPECT_GE(res.explained_variance(k), 0.0);
    }
    for (Eigen::Index k = 1; k < M; ++k) {
        EXPECT_GE(res.explained_variance(k - 1),
                  res.explained_variance(k) - 1e-12);
    }
}


/**
 * @brief Test the sum of all explained variances equals the total variance of X:
 *        trace(X_centered^T X_centered) / (n-1) when using the full SVD.
 */
TEST(PCATest, ExplainedVariance_SumEqualsTotalVariance) {
    const Eigen::Index n = 30, p = 10;
    Eigen::Index M = std::min(n, p); // full decomposition
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::RowVectorXd mu = X.colwise().mean();
    Eigen::MatrixXd Xc = X.rowwise() - mu;
    double total_var = Xc.colwise().squaredNorm().sum() / static_cast<double>(n - 1);

    PCA pca;
    PCAResult res = pca.fit(X, M);

    double sum_ev = res.explained_variance.sum();
    EXPECT_NEAR(sum_ev, total_var, 1e-9);
}


// ========================================================================================
// transform()
// ========================================================================================

/**
 * @brief Test scores from fit() equal transform() applied to the original (uncentered) X.
 *
 * @details transform() subtracts the stored mean internally, so it must receive
 *          the original X. We snapshot X before fit() centers it in-place.
 */
TEST(PCATest, Transform_ConsistentWithFitScores) {
    const Eigen::Index n = 40, p = 15, M = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_original = X;  // snapshot before fit() modifies X in-place

    PCA pca;
    PCAResult res = pca.fit(X, M);

    // transform() expects the original (uncentered) X and subtracts the mean itself.
    Eigen::MatrixXd Z_via_transform = pca.transform(X_original);
    EXPECT_TRUE(Z_via_transform.isApprox(res.Z, 1e-10))
        << "transform() does not reproduce fit() scores on original X.";
}


/** @brief Test transform() on held-out data has the expected shape. */
TEST(PCATest, Transform_HeldOutDataShape) {
    const Eigen::Index n = 40, p = 15, M = 4;
    Eigen::MatrixXd X_train = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_test  = Eigen::MatrixXd::Random(20, p);

    PCA pca;
    pca.fit(X_train, M);

    Eigen::MatrixXd Z_test = pca.transform(X_test);
    EXPECT_EQ(Z_test.rows(), 20);
    EXPECT_EQ(Z_test.cols(), M);
}


// ========================================================================================
// Exception handling
// ========================================================================================

/** @brief Test M > min(n, p) throws std::invalid_argument. */
TEST(PCATest, Exceptions_InvalidM) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 10);
    PCA pca;
    EXPECT_THROW(pca.fit(X, 11), std::invalid_argument);
}


/** @brief Test n <= 1 throws std::invalid_argument (variance undefined). */
TEST(PCATest, Exceptions_SingleSample) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 5);
    PCA pca;
    EXPECT_THROW(pca.fit(X, 1), std::invalid_argument);
}


/** @brief Test transform() before fit() throws std::runtime_error. */
TEST(PCATest, Exceptions_TransformBeforeFit) {
    Eigen::MatrixXd X_test = Eigen::MatrixXd::Random(10, 5);
    PCA pca;
    EXPECT_THROW(pca.transform(X_test), std::runtime_error);
}

// ========================================================================================
} /* End of namespace trex::test::ml_methods::pca */
