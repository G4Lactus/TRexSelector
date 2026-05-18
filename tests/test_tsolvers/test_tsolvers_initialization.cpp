// ========================================================================================
// test_tsolvers_initialization.cpp
// ========================================================================================
/**
 * @file test_tsolvers_initialization.cpp
 * @brief Unit tests verifying initialization behavior (preprocessing, mappings,
 * collinearity checks) for TSolver_Base via the concrete proxy TLARS_Solver.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;

// ========================================================================================

class TSolverInitTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;

    void SetUp() override {
        // Use a fixed random seed for reproducible testing
        std::srand(42);

        X = Eigen::MatrixXd::Random(200, 10);
        D = Eigen::MatrixXd::Random(200, 10);
        y = Eigen::VectorXd::Random(200);

        // Add extreme offsets to effectively test centering
        X.rowwise() += Eigen::VectorXd::Constant(10, 15.0).transpose();
        D.rowwise() += Eigen::VectorXd::Constant(10, -50.0).transpose();
        y.array() += 100.0;
    }
};

// ========================================================================================

/** @brief Verify unit L2-normalization and centroid subtraction during initialization */
TEST_F(TSolverInitTest, NormalizationAndCentering) {

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // Creating the solver triggers the preprocessing inside TSolver_Base
    TLARS_Solver solver(X_map, D_map, y_map,
                        true /*normalize*/, true /*intercept*/,
                        false /*verbose*/);

    // After centering and scaling, each column should have a zero mean and unit L2-norm
    for (int j = 0; j < X.cols(); ++j) {
        EXPECT_NEAR(X_map.col(j).mean(), 0.0, 1e-10);
        EXPECT_NEAR(X_map.col(j).norm(), 1.0, 1e-10);
    }

    for (int j = 0; j < D.cols(); ++j) {
        EXPECT_NEAR(D_map.col(j).mean(), 0.0, 1e-10);
        EXPECT_NEAR(D_map.col(j).norm(), 1.0, 1e-10);
    }
}


/** @brief Verify collinear columns are caught and tracked correctly to avoid singularity */
TEST_F(TSolverInitTest, CollinearDroppingLogic) {
    // Force a perfect collinearity (Column 1 is exactly 3x Column 0)
    X.col(1) = X.col(0) * 3.0;

    // Force a constant (zero variance) column
    X.col(2) = Eigen::VectorXd::Constant(200, 42.0);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map,
                        true /*normalize*/, true /*intercept*/,
                        false /*verbose*/);

    // Since we forced dependent columns, there should be dropped indices
    const auto& dropped = solver.getDroppedIndices();
    EXPECT_FALSE(dropped.empty());

    // The perfectly collinear and constant columns should be safely dropped
    // ensuring executeStep doesn't encounter floating point limits
    EXPECT_NO_THROW(solver.executeStep(2, true));
}
