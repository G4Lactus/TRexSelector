// ========================================================================================
// test_tsolvers_execution.cpp
// ========================================================================================
/**
 * @file test_tsolvers_execution.cpp
 *
 * @brief Unit tests verifying core execution path integrity (bounds checking, monotonicity)
 *        for TSolver_Base via the concrete proxy TLARS_Solver.
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

class TSolverExecutionTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;

    void SetUp() override {
        // Use datagen utilities to get well-conditioned design matrices
        SyntheticData data(
            200, 50, 50,
            {0, 10, 20},
            {5.0, -3.0, 2.0},
            1.0, 42, -1, -1,
            predictor_policy::Normal(),
            dummygen::Distribution::Normal(),
            noisegen::noise_policy::Normal()
        );

        X = data.getX();
        D = data.getD();
        y = data.getY();
    }
};

// ========================================================================================

/** @brief Validate that the solver properly respects early T_stop halting constraints. */
TEST_F(TSolverExecutionTest, ExecuteEarlyStoppingAtTStop) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map, true,
                        true, false);

    // Goal: Stop when 4 dummy variables have entered the active set.
    int T_stop = 4;
    solver.executeStep(T_stop, true);

    // It should have executed some steps...
    EXPECT_GT(solver.getNumSteps(), 0);

    // Check dummy capacity actively constrained to exactly what we requested
    auto active_dummies = solver.getActiveDummyIndices();
    EXPECT_EQ(active_dummies.size(), T_stop);
}


/** @brief Ensure the residual sum of squares shrinks monotonically along the inclusion path. */
TEST_F(TSolverExecutionTest, MonotonicRSSDecrease) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map, true,
                        true, false);

    // Execute multiple steps
    solver.executeStep(10, true);

    const auto& rss_path = solver.getRSS();
    EXPECT_FALSE(rss_path.empty());

    // RSS should strictly decrease (or at least remain non-increasing) over successful
    // variable inclusions
    for (std::size_t i = 1; i < rss_path.size(); ++i) {
        // We use EXPECT_LE due to minor precision limits on edge updates
        EXPECT_LE(rss_path[i], rss_path[i-1] + 1e-12);
    }
}
