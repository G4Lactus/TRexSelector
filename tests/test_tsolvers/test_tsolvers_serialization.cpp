// ========================================================================================
// test_tsolvers_serialization.cpp
// ========================================================================================
/**
 * @file test_tsolvers_serialization.cpp
 *
 * @brief Unit tests verifying serialization and warm-start capabilities
 *        for TSolver_Base via the concrete proxy TLARS_Solver.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <filesystem>
#include <string>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;
namespace fs = std::filesystem;

// ========================================================================================

class TSolverSerializationTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;

    void SetUp() override {
        // Construct a small, well-behaved dataset for testing
        SyntheticData data(
            150, 40, 40,
            {5, 15, 25},
            {4.0, -2.5, 3.1},
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

/** @brief Verify resuming essentially replicates a continuous run */
TEST_F(TSolverSerializationTest, ExpectResumePathMatchesContinuousPath) {
    // CRUCIAL: TSolver_Base mutates the input matrices via Eigen::Map because of normalization!
    // If we run `ref_solver` on X, X is permanently centered and scaled.
    // `partial_solver` would then receive ALREADY NORMALIZED data, calculate new normalization
    // weights, and diverge!
    // We must pass a copy to the second workflow.

    auto X2 = X; auto D2 = D; auto y2 = y;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // 1. Continuous Reference execution to T_stop = 10
    TLARS_Solver ref_solver(X_map, D_map, y_map,
                            true, true, false);
    ref_solver.executeStep(10, true);

    Eigen::Map<Eigen::MatrixXd> X_map2(X2.data(), X2.rows(), X2.cols());
    Eigen::Map<Eigen::MatrixXd> D_map2(D2.data(), D2.rows(), D2.cols());
    Eigen::Map<Eigen::VectorXd> y_map2(y2.data(), y2.size());

    // 2. Partial execution followed by save and resume
    TLARS_Solver partial_solver(X_map2, D_map2, y_map2,
                                true, true, false);
    partial_solver.executeStep(5, true);

    std::string checkpoint = "temp_tlars_checkpoint.bin";
    partial_solver.save(checkpoint);

    // Ensure checkpoint exists
    EXPECT_TRUE(fs::exists(checkpoint));

    // Reload from file (Provide the tracking maps which hold their internally preprocessed state)
    TLARS_Solver resumed_solver = TLARS_Solver::load(checkpoint, X_map2, D_map2);

    // Verify it picked up correctly
    EXPECT_EQ(resumed_solver.getNumSteps(), partial_solver.getNumSteps());

    // Continue execution to T_stop = 10
    resumed_solver.executeStep(10, true);

    // Assert parity against the uninterrupted run
    EXPECT_EQ(ref_solver.getNumSteps(), resumed_solver.getNumSteps());

    auto ref_beta = ref_solver.getBeta(-1);
    auto res_beta = resumed_solver.getBeta(-1);

    // Exact state mapping implies minimal floating point drift
    EXPECT_TRUE(ref_beta.isApprox(res_beta, 1e-9));

    // Cleanup
    fs::remove(checkpoint);
}
