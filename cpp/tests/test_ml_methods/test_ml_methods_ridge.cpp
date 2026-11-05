// ========================================================================================
/**
 * @file test_ml_methods_ridge.cpp
 * @brief Unit tests for the RidgeSolver class.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <stdexcept>

// ml_methods includes
#include <ml_methods/ridge_regression/ridge.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::ridge {

using namespace trex::ml_methods::ridge;

// ========================================================================================
// Output dimensions
// ========================================================================================

/** @brief Test solve() returns a vector of length p (primal path, n >= p). */
TEST(RidgeSolverTest, Dimensions_PrimalPath) {
    const Eigen::Index n = 20, p = 8;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    Eigen::VectorXd beta = RidgeSolver::solve(X, y, 1.0);

    EXPECT_EQ(beta.size(), p);
}


/** @brief Test solve() returns a vector of length p (dual path, n < p). */
TEST(RidgeSolverTest, Dimensions_DualPath) {
    const Eigen::Index n = 8, p = 20;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    Eigen::VectorXd beta = RidgeSolver::solve(X, y, 1.0);

    EXPECT_EQ(beta.size(), p);
}


// ========================================================================================
// Correctness — known solutions
// ========================================================================================

/**
 * @brief Test Ridge satisfies its own KKT (optimality) condition: X^T(y - X*β) = λ*β.
 *
 * @details This is a deterministic correctness check holding to near-machine-precision
 *          for any correct Ridge solution. Uses a sin-based deterministic design matrix.
 */
TEST(RidgeSolverTest, Correctness_KKTConditionPrimal) {
    const Eigen::Index n = 50, p = 15;
    double lambda = 0.5;

    // Deterministic, non-random design matrix via trigonometric filling.
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, p);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < p; ++j)
            X(i, j) = std::sin(static_cast<double>(i + 1) / static_cast<double>(j + 1));

    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, -2.0, 3.0);

    Eigen::VectorXd beta_hat = RidgeSolver::solve(X, y, lambda);

    // KKT condition: X^T (y - X*beta) == lambda * beta
    Eigen::VectorXd kkt_residual = X.transpose() * (y - X * beta_hat) - lambda * beta_hat;
    EXPECT_NEAR(kkt_residual.norm(), 0.0, 1e-8);
}


/**
 * @brief Test noise-free recovery (dual): same KKT check for the wide-matrix (n < p) path.
 *
 * @details Uses a deterministic sin-based design matrix to avoid RNG state issues.
 */
TEST(RidgeSolverTest, Correctness_KKTConditionDual) {
    const Eigen::Index n = 15, p = 40;
    double lambda = 0.5;

    // Deterministic wide design matrix
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, p);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < p; ++j)
            X(i, j) = std::cos(static_cast<double>(i + 1)
                             * static_cast<double>(j + 1) * 0.1);

    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, 1.0, 5.0);

    Eigen::VectorXd beta_hat = RidgeSolver::solve(X, y, lambda);

    // KKT condition: X^T (y - X*beta) == lambda * beta
    Eigen::VectorXd kkt_residual = X.transpose() * (y - X * beta_hat) - lambda * beta_hat;
    EXPECT_NEAR(kkt_residual.norm(), 0.0, 1e-8);
}


/**
 * @brief Test manual solution check: for a 2x2 identity design matrix the Ridge solution
 *        has a closed form beta_i = y_i / (1 + lambda).
 */
TEST(RidgeSolverTest, Correctness_AnalyticalIdentityDesign) {
    const Eigen::Index n = 2;
    Eigen::MatrixXd X = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd y(n);
    y << 4.0, 6.0;
    double lambda = 2.0;

    Eigen::VectorXd beta = RidgeSolver::solve(X, y, lambda);

    EXPECT_NEAR(beta(0), 4.0 / (1.0 + lambda), 1e-12);
    EXPECT_NEAR(beta(1), 6.0 / (1.0 + lambda), 1e-12);
}


// ========================================================================================
// Shrinkage behaviour
// ========================================================================================

/**
 * @brief Test larger lambda → smaller coefficient norms (shrinkage property).
 */
TEST(RidgeSolverTest, Shrinkage_LargerLambdaReducesNorm) {
    const Eigen::Index n = 20, p = 8;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    double norm_small = RidgeSolver::solve(X, y, 1e-3).norm();
    double norm_large = RidgeSolver::solve(X, y, 1e3).norm();

    EXPECT_LT(norm_large, norm_small)
        << "Larger lambda should produce a smaller coefficient norm.";
}


/**
 * @brief Test as lambda → ∞ the coefficients approach zero.
 */
TEST(RidgeSolverTest, Shrinkage_VeryLargeLambdaApproachesZero) {
    const Eigen::Index n = 20, p = 8;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    Eigen::VectorXd beta = RidgeSolver::solve(X, y, 1e12);

    EXPECT_NEAR(beta.norm(), 0.0, 1e-6);
}


// ========================================================================================
// Primal / dual consistency
// ========================================================================================

/**
 * @brief For lambda > 0 the ridge solution is unique, so the dual path (n < p) must
 *        match the primal closed form (X^T X + lambda I)^{-1} X^T y, which is
 *        well-posed for lambda > 0 regardless of the shape of X.
 */
TEST(RidgeSolverTest, PrimalDualConsistency_ClosedFormMatch) {
    const Eigen::Index n = 8, p = 20;  // n < p -> RidgeSolver dispatches to the dual path
    const double lambda = 0.5;

    // Deterministic wide design matrix
    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < p; ++j)
            X(i, j) = std::sin(static_cast<double>(i + 1)
                             * static_cast<double>(j + 1) * 0.2);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, -1.0, 2.0);

    Eigen::VectorXd beta_dual = RidgeSolver::solve(X, y, lambda);

    // Primal closed form computed independently
    Eigen::MatrixXd A = X.transpose() * X
                      + lambda * Eigen::MatrixXd::Identity(p, p);
    Eigen::VectorXd beta_primal = A.ldlt().solve(X.transpose() * y);

    EXPECT_TRUE(beta_dual.isApprox(beta_primal, 1e-8))
        << "Dual-path solution deviates from the primal closed form.";
}


/**
 * @brief The "lambda_2 trap" from the solver's documentation: a perfectly duplicated
 *        column with a tiny lambda keeps X^T X + lambda I positive-definite but
 *        extremely ill-conditioned. Whether LLT succeeds or the LDLT fallback kicks
 *        in, the returned solution must satisfy the ridge KKT condition.
 */
TEST(RidgeSolverTest, RankDeficient_TinyLambdaStaysSolvable) {
    const Eigen::Index n = 12, p = 4;

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < p - 1; ++j)
            X(i, j) = std::cos(static_cast<double>(i + 1)
                             * static_cast<double>(j + 1) * 0.3);
    X.col(p - 1) = X.col(0);  // exact duplicate -> X^T X singular, X^T X + lambda I PD

    // Response in the column space of X
    Eigen::VectorXd w(p);
    w << 1.0, -2.0, 0.5, 0.0;
    Eigen::VectorXd y = X * w;

    const double lambda = 1e-8;
    Eigen::VectorXd beta = RidgeSolver::solve(X, y, lambda);

    // KKT residual (tolerance reflects the ~1/lambda conditioning of the system)
    Eigen::VectorXd kkt = X.transpose() * (y - X * beta) - lambda * beta;
    EXPECT_NEAR(kkt.norm(), 0.0, 1e-4);

    // With lambda = 0 exactly the system is singular: depending on the platform's
    // LDLT behavior the solver either reports failure or returns a valid solution
    // of the (consistent) normal equations. Both are acceptable; garbage is not.
    try {
        Eigen::VectorXd beta0 = RidgeSolver::solve(X, y, 0.0);
        EXPECT_NEAR((X.transpose() * (y - X * beta0)).norm(), 0.0, 1e-6);
    } catch (const std::runtime_error&) {
        SUCCEED() << "Factorizations rejected the exactly singular system.";
    }
}


// ========================================================================================
// Exception handling
// ========================================================================================

/** @brief Test dimension mismatch between X and y throws std::invalid_argument. */
TEST(RidgeSolverTest, Exceptions_DimensionMismatch) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(10, 5);
    Eigen::VectorXd y = Eigen::VectorXd::Random(11); // wrong size

    EXPECT_THROW(RidgeSolver::solve(X, y, 1.0), std::invalid_argument);
}


/** @brief Test negative lambda throws std::invalid_argument. */
TEST(RidgeSolverTest, Exceptions_NegativeLambda) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(10, 5);
    Eigen::VectorXd y = Eigen::VectorXd::Random(10);

    EXPECT_THROW(RidgeSolver::solve(X, y, -0.1), std::invalid_argument);
}


/** @brief Test lambda = 0 is valid and returns a solution. */
TEST(RidgeSolverTest, Exceptions_ZeroLambdaIsValid) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(15, 5);
    Eigen::VectorXd y = Eigen::VectorXd::Random(15);

    EXPECT_NO_THROW({
        Eigen::VectorXd beta = RidgeSolver::solve(X, y, 0.0);
        EXPECT_EQ(beta.size(), 5);
    });
}

// ========================================================================================
} /* End of namespace trex::test::ml_methods::ridge */
