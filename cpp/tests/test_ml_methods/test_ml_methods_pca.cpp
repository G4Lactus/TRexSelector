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

    PCA pca(X, M);
    PCAResult res = pca.fit();

    EXPECT_EQ(res.Z.rows(), n);
    EXPECT_EQ(res.Z.cols(), M);
    EXPECT_EQ(res.V.rows(), p);
    EXPECT_EQ(res.V.cols(), M);
    EXPECT_EQ(res.explained_variance.size(), M);
}


/** @brief Test getLoadings() and getMeans() are accessible after fit(). */
TEST(PCATest, Dimensions_GetterShapesAfterFit) {
    const Eigen::Index n = 40, p = 15, M = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    PCA pca(X, M);
    pca.fit();

    EXPECT_EQ(pca.getLoadings().rows(), p);
    EXPECT_EQ(pca.getLoadings().cols(), M);
    EXPECT_EQ(pca.getMeans().size(), p);
    EXPECT_EQ(pca.getNorms().size(), p);
    EXPECT_EQ(pca.getExplainedVariance().size(), M);
}


// ========================================================================================
// Centering
// ========================================================================================

/**
 * @brief Test when center=true, fit() subtracts column means from X in-place.
 *        The stored means equal the original column means of X.
 */
TEST(PCATest, Centering_MeansStoredCorrectly) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::RowVectorXd expected_mean = X.colwise().mean();

    PCA pca(X, M, /*center=*/true, /*normalize=*/false);
    pca.fit();

    EXPECT_TRUE(pca.getMeans().isApprox(expected_mean, 1e-12));
}


/** @brief Test when center=false, getMeans() returns a zero vector. */
TEST(PCATest, Centering_NoCenterMeansIsZero) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.rowwise() -= X.colwise().mean();  // Pre-center manually

    PCA pca(X, M, /*center=*/false, /*normalize=*/false);
    pca.fit();

    EXPECT_TRUE(
        pca.getMeans().isApprox(Eigen::RowVectorXd::Zero(p), 1e-12)
    );
}


// ========================================================================================
// Normalisation
// ========================================================================================

/** @brief Test when normalize=true, getNorms() stores the correct column L2 norms. */
TEST(PCATest, Normalisation_NormsStoredCorrectly) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    // Snapshot of column norms of the already-centered X
    Eigen::MatrixXd Xc = X.rowwise() - X.colwise().mean();
    Eigen::RowVectorXd expected_norms = Xc.colwise().norm();

    PCA pca(X, M, /*center=*/true, /*normalize=*/true);
    pca.fit();

    EXPECT_TRUE(pca.getNorms().isApprox(expected_norms, 1e-10));
}


/** @brief Test when normalize=false, getNorms() returns a ones vector. */
TEST(PCATest, Normalisation_NoNormalizeNormsAreOnes) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    PCA pca(X, M, /*center=*/true, /*normalize=*/false);
    pca.fit();

    EXPECT_TRUE(
        pca.getNorms().isApprox(Eigen::RowVectorXd::Ones(p), 1e-12)
    );
}


// ========================================================================================
// Restoration
// ========================================================================================

/** @brief Test restore(X) undoes centering correctly. */
TEST(PCATest, Restoration_CenteringUndone) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_original = X;

    PCA pca(X, M, /*center=*/true, /*normalize=*/false);
    pca.fit();
    pca.restore(X);

    EXPECT_TRUE(X.isApprox(X_original, 1e-12))
        << "X was not restored after centering.";
}


/** @brief Test restore(X) undoes both centering and normalisation correctly. */
TEST(PCATest, Restoration_CenterAndNormaliseUndone) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_original = X;

    PCA pca(X, M, /*center=*/true, /*normalize=*/true);
    pca.fit();
    pca.restore(X);

    EXPECT_TRUE(X.isApprox(X_original, 1e-10))
        << "X was not restored after centering + normalisation.";
}


/** @brief Test restore(X) is a no-op when both flags are false. */
TEST(PCATest, Restoration_BothFalseDoesNotModifyX) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    X.rowwise() -= X.colwise().mean();  // pre-center
    Eigen::MatrixXd X_snapshot = X;

    PCA pca(X, M, /*center=*/false, /*normalize=*/false);
    pca.fit();

    EXPECT_TRUE(X.isApprox(X_snapshot, 1e-12))
        << "X was modified despite center=false, normalize=false.";

    pca.restore(X);  // no-op
    EXPECT_TRUE(X.isApprox(X_snapshot, 1e-12));
}


/** @brief Test restore(X) is idempotent: second call is a safe no-op (state flags). */
TEST(PCATest, Restoration_Idempotent) {
    const Eigen::Index n = 30, p = 10, M = 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_original = X;

    PCA pca(X, M);
    pca.fit();

    pca.restore(X);
    EXPECT_TRUE(X.isApprox(X_original, 1e-10));

    // Second call: is_centered_ / is_normalized_ are now false — no-op.
    EXPECT_NO_THROW(pca.restore(X));
    EXPECT_TRUE(X.isApprox(X_original, 1e-10))
        << "X was modified by a second restore() call (should be a no-op).";
}


// ========================================================================================
// Loadings orthogonality
// ========================================================================================

/** @brief Test Loadings are orthonormal: V^T V ≈ I_M. */
TEST(PCATest, Loadings_Orthonormal) {
    const Eigen::Index n = 50, p = 20, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    PCA pca(X, M);
    PCAResult res = pca.fit();

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

    PCA pca(X, M);
    PCAResult res = pca.fit();

    for (Eigen::Index k = 0; k < M; ++k) {
        EXPECT_GE(res.explained_variance(k), 0.0);
    }
    for (Eigen::Index k = 1; k < M; ++k) {
        EXPECT_GE(res.explained_variance(k - 1),
                  res.explained_variance(k) - 1e-12);
    }
}


/**
 * @brief Test the sum of all explained variances equals the total variance of X
 *        (center only, no normalisation, full decomposition).
 */
TEST(PCATest, ExplainedVariance_SumEqualsTotalVariance) {
    const Eigen::Index n = 30, p = 10;
    Eigen::Index M = std::min(n, p);  // full decomposition
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::RowVectorXd mu = X.colwise().mean();
    Eigen::MatrixXd Xc = X.rowwise() - mu;
    double total_var = Xc.colwise().squaredNorm().sum() / static_cast<double>(n - 1);

    PCA pca(X, M, /*center=*/true, /*normalize=*/false);
    PCAResult res = pca.fit();

    double sum_ev = res.explained_variance.sum();
    EXPECT_NEAR(sum_ev, total_var, 1e-9);
}


// ========================================================================================
// transform()
// ========================================================================================

/**
 * @brief Test scores from fit() equal transform() applied to the original X.
 *
 * @details transform() applies the same centering and normalisation internally,
 *          so it must receive the original (un-preprocessed) X.
 *          We snapshot X before construction (before fit() modifies it).
 */
TEST(PCATest, Transform_ConsistentWithFitScores) {
    const Eigen::Index n = 40, p = 15, M = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_original = X;  // snapshot before constructor preprocesses X in-place

    PCA pca(X, M);
    PCAResult res = pca.fit();

    // transform() expects the original (un-preprocessed) X.
    Eigen::MatrixXd Z_via_transform = pca.transform(X_original);
    EXPECT_TRUE(Z_via_transform.isApprox(res.Z, 1e-10))
        << "transform() does not reproduce fit() scores on original X.";
}


/** @brief Test transform() on held-out data has the expected shape. */
TEST(PCATest, Transform_HeldOutDataShape) {
    const Eigen::Index n = 40, p = 15, M = 4;
    Eigen::MatrixXd X_train = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd X_test  = Eigen::MatrixXd::Random(20, p);

    PCA pca(X_train, M);
    pca.fit();

    Eigen::MatrixXd Z_test = pca.transform(X_test);
    EXPECT_EQ(Z_test.rows(), 20);
    EXPECT_EQ(Z_test.cols(), M);
}


// ========================================================================================
// Exception handling
// ========================================================================================

/** @brief Test M > min(n, p) throws std::invalid_argument in the constructor. */
TEST(PCATest, Exceptions_InvalidM) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 10);
    EXPECT_THROW((PCA{X, 11}), std::invalid_argument);
}


/** @brief Test n <= 1 throws std::invalid_argument in the constructor. */
TEST(PCATest, Exceptions_SingleSample) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 5);
    EXPECT_THROW((PCA{X, 1}), std::invalid_argument);
}


/** @brief Test fit() throws std::runtime_error if called a second time. */
TEST(PCATest, Exceptions_FitCalledTwice) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 10);
    PCA pca(X, 3);
    pca.fit();
    EXPECT_THROW(pca.fit(), std::runtime_error);
}


/** @brief Test transform() before fit() throws std::runtime_error. */
TEST(PCATest, Exceptions_TransformBeforeFit) {
    Eigen::MatrixXd X      = Eigen::MatrixXd::Random(20, 10);
    Eigen::MatrixXd X_test = Eigen::MatrixXd::Random(10, 10);
    PCA pca(X, 3);
    EXPECT_THROW(pca.transform(X_test), std::runtime_error);
}


/** @brief Test restore() before fit() throws std::runtime_error. */
TEST(PCATest, Exceptions_RestoreBeforeFit) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 10);
    PCA pca(X, 3);
    EXPECT_THROW(pca.restore(X), std::runtime_error);
}


/** @brief Test restore() with wrong dimensions throws std::invalid_argument. */
TEST(PCATest, Exceptions_RestoreWrongDimensions) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(30, 10);
    PCA pca(X, 3);
    pca.fit();
    Eigen::MatrixXd X_wrong = Eigen::MatrixXd::Random(20, 10);
    EXPECT_THROW(pca.restore(X_wrong), std::invalid_argument);
}


/**
 * @brief Test getMeans()/getNorms() are accessible before fit() (set in constructor);
 *        getLoadings()/getExplainedVariance() still throw before fit().
 */
TEST(PCATest, Exceptions_GettersBeforeFit) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 10);
    PCA pca(X, 3);
    // means_ and norms_ are set by the constructor — no throw expected.
    EXPECT_NO_THROW(pca.getMeans());
    EXPECT_NO_THROW(pca.getNorms());
    // loadings_ and explained_variance_ require fit() — throw expected.
    EXPECT_THROW(pca.getLoadings(),          std::runtime_error);
    EXPECT_THROW(pca.getExplainedVariance(), std::runtime_error);
}

// ========================================================================================
} /* End of namespace trex::test::ml_methods::pca */
