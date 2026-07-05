// ========================================================================================
/**
 * @file test_ml_methods_svd.cpp
 * @brief Unit tests for the SVDSolver class.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// project ml_methods includes
#include <ml_methods/svd/svd.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::svd {

using namespace trex::ml_methods::svd;

// ========================================================================================
// Helpers
// ========================================================================================

/**
 * @brief Construct a rank-M matrix X = U_true * diag(S_true) * V_true^T
 *        with known singular values, useful for verifying exact recovery.
 */
Eigen::MatrixXd make_rank_m_matrix(Eigen::Index n, Eigen::Index p, Eigen::Index M) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, M);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(p, M);
    // Thin QR to get orthonormal columns
    Eigen::HouseholderQR<Eigen::MatrixXd> qrA(A), qrB(B);
    Eigen::MatrixXd U = qrA.householderQ() * Eigen::MatrixXd::Identity(n, M);
    Eigen::MatrixXd V = qrB.householderQ() * Eigen::MatrixXd::Identity(p, M);
    Eigen::VectorXd S = Eigen::VectorXd::LinSpaced(M, static_cast<double>(M), 1.0); // M, M-1, ..., 1
    return U * S.asDiagonal() * V.transpose();
}


// ========================================================================================
// Output dimensions
// ========================================================================================

/** @brief Computed factors have the expected shapes (tall matrix). */
TEST(SVDSolverTest, Dimensions_TallMatrix) {
    const Eigen::Index n = 50, p = 20, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    EXPECT_EQ(res.U.rows(), n);
    EXPECT_EQ(res.U.cols(), M);
    EXPECT_EQ(res.S.size(), M);
    EXPECT_EQ(res.V.rows(), p);
    EXPECT_EQ(res.V.cols(), M);
}


/** @brief Computed factors have the expected shapes (wide matrix → Gram path). */
TEST(SVDSolverTest, Dimensions_WideMatrix) {
    const Eigen::Index n = 20, p = 80, M = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    EXPECT_EQ(res.U.rows(), n);
    EXPECT_EQ(res.U.cols(), M);
    EXPECT_EQ(res.S.size(), M);
    EXPECT_EQ(res.V.rows(), p);
    EXPECT_EQ(res.V.cols(), M);
}


// ========================================================================================
// Orthogonality
// ========================================================================================

/** @brief Left singular vectors are orthonormal: U^T U ≈ I_M. */
TEST(SVDSolverTest, Orthogonality_UOrthonormal) {
    const Eigen::Index n = 60, p = 30, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    Eigen::MatrixXd UtU = res.U.transpose() * res.U;
    EXPECT_TRUE(UtU.isApprox(Eigen::MatrixXd::Identity(M, M), 1e-10))
        << "U^T U deviates from I_M";
}


/** @brief Right singular vectors are orthonormal: V^T V ≈ I_M. */
TEST(SVDSolverTest, Orthogonality_VOrthonormal) {
    const Eigen::Index n = 60, p = 30, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    Eigen::MatrixXd VtV = res.V.transpose() * res.V;
    EXPECT_TRUE(VtV.isApprox(Eigen::MatrixXd::Identity(M, M), 1e-10))
        << "V^T V deviates from I_M";
}


/** @brief Gram-path singular vectors are also orthonormal (wide matrix). */
TEST(SVDSolverTest, Orthogonality_GramPathVOrthonormal) {
    const Eigen::Index n = 15, p = 60, M = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    Eigen::MatrixXd VtV = res.V.transpose() * res.V;
    EXPECT_TRUE(VtV.isApprox(Eigen::MatrixXd::Identity(M, M), 1e-10))
        << "Gram-path V^T V deviates from I_M";
}


// ========================================================================================
// Singular values
// ========================================================================================

/** @brief Singular values are non-negative. */
TEST(SVDSolverTest, SingularValues_NonNegative) {
    const Eigen::Index n = 40, p = 25, M = 6;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    for (Eigen::Index k = 0; k < M; ++k) {
        EXPECT_GE(res.S(k), 0.0);
    }
}


/** @brief Singular values are returned in non-increasing order. */
TEST(SVDSolverTest, SingularValues_NonIncreasing) {
    const Eigen::Index n = 50, p = 30, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    for (Eigen::Index k = 1; k < M; ++k) {
        EXPECT_GE(res.S(k - 1), res.S(k) - 1e-12);
    }
}


// ========================================================================================
// Reconstruction accuracy
// ========================================================================================

/**
 * @brief Full-rank reconstruction: the rank-min(n,p) factorization reconstructs X exactly
 *        (direct-SVD path).
 */
TEST(SVDSolverTest, Reconstruction_FullRank) {
    const Eigen::Index n = 20, p = 15;
    Eigen::Index M = std::min(n, p);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    Eigen::MatrixXd X_approx = res.U * res.S.asDiagonal() * res.V.transpose();
    EXPECT_TRUE(X_approx.isApprox(X, 1e-9))
        << "Full-rank SVD reconstruction does not match X";
}


/**
 * @brief Rank-M reconstruction is the best rank-M approximation:
 *        the residual norm equals the (M+1)-th singular value of X.
 */
TEST(SVDSolverTest, Reconstruction_LowRankResidual) {
    const Eigen::Index n = 30, p = 20, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    // Full SVD to find the true next singular value
    Eigen::JacobiSVD<Eigen::MatrixXd> full_svd(X,
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    double sigma_next = full_svd.singularValues()(M);

    SVDResult res = SVDSolver{}.compute(X, M);
    Eigen::MatrixXd residual = X - res.U * res.S.asDiagonal() * res.V.transpose();

    // ||X - X_M||_2 ≈ sigma_{M+1}
    Eigen::JacobiSVD<Eigen::MatrixXd> res_svd(residual, Eigen::ComputeThinU);
    double residual_sigma1 = res_svd.singularValues()(0);

    EXPECT_NEAR(residual_sigma1, sigma_next, 1e-9);
}


// ========================================================================================
// Dispatch paths
// ========================================================================================

/**
 * @brief Low min_dim_thresh forces the Randomized SVD path.
 *        We only verify shapes and approximate orthonormality (randomized method).
 */
TEST(SVDSolverTest, Dispatch_RandomizedPathShapesAndOrthogonality) {
    const Eigen::Index n = 60, p = 60, M = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    // min_dim_thresh = 10 forces the randomized path for any matrix with min(n,p) > 10
    SVDSolver solver(/*min_dim_thresh=*/10, /*block_size=*/32);
    SVDResult res = solver.compute(X, M);

    EXPECT_EQ(res.U.rows(), n);
    EXPECT_EQ(res.S.size(), M);
    EXPECT_EQ(res.V.rows(), p);

    Eigen::MatrixXd VtV = res.V.transpose() * res.V;
    EXPECT_TRUE(VtV.isApprox(Eigen::MatrixXd::Identity(M, M), 1e-8))
        << "Randomized-path V^T V deviates from I_M";
}


/**
 * @brief Gram path (p > 2n) must reproduce the direct JacobiSVD numerically:
 *        same singular values and the same rank-M approximation of X.
 */
TEST(SVDSolverTest, GramPathMatchesDirectSVD) {
    const Eigen::Index n = 15, p = 60, M = 6;  // p > 2n -> Gram/kernel path
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);

    SVDResult res = SVDSolver{}.compute(X, M);

    Eigen::JacobiSVD<Eigen::MatrixXd> ref(X,
        Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE(res.S.isApprox(ref.singularValues().head(M), 1e-8))
        << "Gram-path singular values deviate from JacobiSVD.";

    // Compare the rank-M approximations (basis-independent).
    Eigen::MatrixXd approx_gram = res.U * res.S.asDiagonal() * res.V.transpose();
    Eigen::MatrixXd approx_ref  = ref.matrixU().leftCols(M)
                                * ref.singularValues().head(M).asDiagonal()
                                * ref.matrixV().leftCols(M).transpose();
    EXPECT_TRUE(approx_gram.isApprox(approx_ref, 1e-8))
        << "Gram-path rank-M approximation deviates from JacobiSVD.";
}


/**
 * @brief Randomized path on an exactly rank-M matrix: the sketch captures the full
 *        range, so recovery of the known singular values {M, ..., 1} is exact
 *        (up to floating-point error), as is the reconstruction.
 */
TEST(SVDSolverTest, RandomizedPathExactOnLowRank) {
    const Eigen::Index n = 200, p = 120, M = 5;
    Eigen::MatrixXd X = make_rank_m_matrix(n, p, M);

    // min_dim_thresh = 10 forces the randomized path (M = 5 < 120/10)
    SVDSolver solver(/*min_dim_thresh=*/10, /*block_size=*/64);
    SVDResult res = solver.compute(X, M);

    Eigen::VectorXd expected_S =
        Eigen::VectorXd::LinSpaced(M, static_cast<double>(M), 1.0);
    EXPECT_TRUE(res.S.isApprox(expected_S, 1e-8))
        << "Randomized path does not recover the known singular values.";

    Eigen::MatrixXd X_approx = res.U * res.S.asDiagonal() * res.V.transpose();
    EXPECT_TRUE(X_approx.isApprox(X, 1e-8))
        << "Randomized path does not reconstruct a rank-M matrix exactly.";
}


/**
 * @brief Gram path on a rank-deficient matrix: trailing singular values are ~0,
 *        the S^{-1} guard must keep V free of NaN/Inf, and the reconstruction is
 *        unaffected by the null components.
 */
TEST(SVDSolverTest, GramPathRankDeficient) {
    const Eigen::Index n = 10, p = 30, rank = 3, M = 5;
    Eigen::MatrixXd X = make_rank_m_matrix(n, p, rank);  // singular values {3, 2, 1}

    SVDResult res = SVDSolver{}.compute(X, M);

    EXPECT_NEAR(res.S(0), 3.0, 1e-6);
    EXPECT_NEAR(res.S(1), 2.0, 1e-6);
    EXPECT_NEAR(res.S(2), 1.0, 1e-6);
    EXPECT_LT(res.S(3), 1e-5);
    EXPECT_LT(res.S(4), 1e-5);

    EXPECT_TRUE(res.V.allFinite())
        << "Null components produced NaN/Inf in V (S^{-1} guard failed).";

    Eigen::MatrixXd X_approx = res.U * res.S.asDiagonal() * res.V.transpose();
    EXPECT_TRUE(X_approx.isApprox(X, 1e-6));
}


// ========================================================================================
// Exception handling
// ========================================================================================

/** @brief M > min(n, p) throws std::invalid_argument. */
TEST(SVDSolverTest, Exceptions_InvalidM) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(10, 15);
    EXPECT_THROW(SVDSolver{}.compute(X, 11), std::invalid_argument);
}


/** @brief M < 1 throws std::invalid_argument (0 and negative). */
TEST(SVDSolverTest, Exceptions_NonPositiveM) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(10, 15);
    EXPECT_THROW(SVDSolver{}.compute(X, 0), std::invalid_argument);
    EXPECT_THROW(SVDSolver{}.compute(X, -1), std::invalid_argument);
}

// ========================================================================================
} /* End of namespace trex::test::ml_methods::svd */
