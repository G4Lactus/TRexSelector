// ========================================================================================
// test_tsolvers_getters.cpp
// ========================================================================================
/**
 * @file test_tsolvers_getters.cpp
 *
 * @brief Unit tests verifying API read methods for model state, path iteration,
 *        and intercepts for TSolver_Base.
 */
// ========================================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <utils/datageneration/utils_datagen.hpp>

using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;

// ========================================================================================

/** @brief Verify extraction of specific iteration path coefficients */
TEST(TSolverGettersTest, ValidateCoefficientPathExtraction) {
    SyntheticData data(
        100, 20, 20,
        {1, 2},
        {5.0, 2.0},
        1.0, 42, -1, -1,
        predictor_policy::Normal(),
        dummygen::Distribution::Normal(),
        noisegen::noise_policy::Normal()
    );

    auto X = data.getX();
    auto D = data.getD();
    auto y = data.getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // Intercept activated
    TLARS_Solver solver(X_map, D_map, y_map,
                        true, true, false);
    solver.executeStep(2, true);

    // Check sizes match step execution
    std::size_t steps = solver.getNumSteps();
    Eigen::MatrixXd beta_path = solver.getBetaPath();

    // Path columns should correspond to tracked actions/steps + 1
    // (for the zero-initialization step 0)
    EXPECT_EQ(beta_path.cols(), solver.getActions().size() + 1);

    for (int step = 0; step < steps; ++step) {
        // Individual beta vector uncentering scale retrieval vs full uncentered path retrieval
        auto beta_at_step = solver.getBeta(step);
        EXPECT_TRUE(beta_path.col(step).isApprox(beta_at_step, 1e-10));
    }
}
