// ==================================================================================
/**
 * @file test_ml_methods_model_selection.cpp
 *
 * @brief Unit tests for ML methods model selection
 */
// ==================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <tuple>

// project ml_methods includes
#include "ml_methods/model_selection/ridge_gcv.hpp"
#include "ml_methods/model_selection/ridge_cv.hpp"

// ==================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::model_selection {

// Namespace includes for convenience
using namespace trex::ml_methods::model_selection;

// ========================================================================================

// Helper to create synthetic data for regression
auto create_synthetic_data(
    Eigen::Index n,
    Eigen::Index p,
    double noise_std,
    unsigned int seed = 42
) {
    std::srand(seed); // Quick determinism
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::VectorXd true_beta = Eigen::VectorXd::Random(p);

    // Create random noise
    Eigen::VectorXd noise = Eigen::VectorXd::Random(n) * noise_std;
    Eigen::VectorXd y = X * true_beta + noise;
    return std::make_tuple(X, y, true_beta);
}

// -----------------------------------------------------------------------------------
// ridge_gcv Tests
// -----------------------------------------------------------------------------------

/** @brief Test for fitting ridge_gcv on a tall matrix (n > p) */
TEST(RidgeGCVTest, FitTallMatrix) {
    // n > p: Covariance matrix eigen-decomposition path
    const Eigen::Index n = 100;
    const Eigen::Index p = 10;
    auto [X, y, beta] = create_synthetic_data(n, p, 0.1);

    ridge_gcv model;
    EXPECT_NO_THROW(model.fit(X, y));

    EXPECT_EQ(model.n_samples(), n);
    EXPECT_EQ(model.n_features(), p);
    EXPECT_LE(model.rank(), p); // Rank cannot exceed p

    // Test lambda grid generation
    Eigen::VectorXd grid = model.default_lambda_grid(50, 1000.0);
    EXPECT_EQ(grid.size(), 50);
    // Grid should be monotonically increasing
    for (Eigen::Index i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid(i), grid(i-1));
    }

    // Solve for a specific lambda
    double lambda = grid(10);
    Eigen::VectorXd coefs = model.solve(lambda);
    EXPECT_EQ(coefs.size(), p);

    // Predict
    Eigen::VectorXd preds = model.predict(X, lambda);
    EXPECT_EQ(preds.size(), n);

    // GCV optimal
    double opt_lambda = model.gcv_optimal(50, 1e-4);
    EXPECT_GT(opt_lambda, 0.0);
}


/** @brief Test for fitting ridge_gcv on a wide matrix (n < p) */
TEST(RidgeGCVTest, FitWideMatrix) {
    // n < p: Gram matrix eigen-decomposition path
    const Eigen::Index n = 20;
    const Eigen::Index p = 50;
    auto [X, y, beta] = create_synthetic_data(n, p, 0.1);

    ridge_gcv model;
    EXPECT_NO_THROW(model.fit(X, y)); // Should trigger the Gram path

    EXPECT_EQ(model.n_samples(), n);
    EXPECT_EQ(model.n_features(), p);
    EXPECT_LE(model.rank(), n); // Rank cannot exceed n when n < p

    double opt_lambda = model.gcv_optimal(100, 1e-4);
    EXPECT_GT(opt_lambda, 0.0);

    // Check paths prediction
    Eigen::VectorXd grid = model.default_lambda_grid(10);
    Eigen::MatrixXd path_preds = model.predict_path(X, grid);
    EXPECT_EQ(path_preds.rows(), n);
    EXPECT_EQ(path_preds.cols(), 10);
}


/** @brief Test for edge cases in ridge_gcv */
TEST(RidgeGCVTest, EdgeCases) {
    ridge_gcv model;

    Eigen::MatrixXd X_bad(10, 5);
    Eigen::VectorXd y_bad(11);

    // Dimension mismatch
    EXPECT_THROW(model.fit(X_bad, y_bad), std::invalid_argument);

    auto [X, y, beta] = create_synthetic_data(10, 5, 0.0);
    model.fit(X, y);

    // Invalid negative lambda solve
    EXPECT_THROW(model.solve(-1.0), std::invalid_argument);
    EXPECT_THROW(model.gcv(-1.0), std::invalid_argument);
    EXPECT_THROW(model.predict(X, -1.0), std::invalid_argument);
}

// -----------------------------------------------------------------------------------
// ridge_cv Tests
// -----------------------------------------------------------------------------------

/** @brief Test for fitting ridge_cv with 5-fold CV */
TEST(RidgeCVTest, CrossValidationRules) {
    const Eigen::Index n = 150;
    const Eigen::Index p = 20;
    auto [X, y, beta] = create_synthetic_data(n, p, 2.0);

    ridge_cv cv_model;

    // 5-fold CV
    const int folds = 5;
    const int n_lambdas = 100;
    EXPECT_NO_THROW(cv_model.fit(X, y, folds, n_lambdas, 1000.0, 42));

    const Eigen::VectorXd& mse = cv_model.cv_mse();
    const Eigen::VectorXd& sem = cv_model.cv_sem();

    EXPECT_EQ(mse.size(), n_lambdas);
    EXPECT_EQ(sem.size(), n_lambdas);

    double l_min = cv_model.cv_min();
    double l_1se = cv_model.cv_1se();

    // The grid is increasing.
    // `lambda.1se` should pick a more regularised model, which is a larger lambda
    // than `lambda.min`. So its index should be >= min index.
    Eigen::Index idx_min = cv_model.index_min();
    Eigen::Index idx_1se = cv_model.index_1se();

    EXPECT_GE(idx_1se, idx_min);
    EXPECT_GE(l_1se, l_min);

    // Validate the 1SE logic manually
    double threshold = mse(idx_min) + sem(idx_min);
    EXPECT_LE(mse(idx_1se), threshold);
}


/** @brief Test for edge cases in ridge_cv */
TEST(RidgeCVTest, EdgeCases) {
    ridge_cv model;

    auto [X, y, beta] = create_synthetic_data(20, 5, 0.0);

    // Dimension mismatch
    EXPECT_THROW(model.fit(X, Eigen::VectorXd::Random(21)), std::invalid_argument);

    // Invalid folds
    EXPECT_THROW(model.fit(X, y, 1), std::invalid_argument); // < 2
    EXPECT_THROW(model.fit(X, y, 25), std::invalid_argument); // > n
}

// ==================================================================================
} /* End of namespace trex::test::ml_methods::model_selection */
// ==================================================================================
