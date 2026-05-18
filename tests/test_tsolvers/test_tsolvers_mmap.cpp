// ========================================================================================
// test_tsolvers_mmap.cpp
// ========================================================================================
/**
 * @file test_tsolvers_mmap.cpp
 *
 * @brief Unit tests verifying memory-mapping behavior interaction with solver preprocessing.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <filesystem>

namespace fs = std::filesystem;

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

/** @brief Verify that memory-mapped data is correctly mutated by the solver */
TEST(TSolverMmapTest, ValidateMapMutation) {
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

    // Modify to be un-centered
    X.array() += 10.0;

    Eigen::MatrixXd X_copy = X; // Keep copy for comparison

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map,
                        true, true, false);

    // Initialization changes the original X memory (via Map) because of centering/scaling
    EXPECT_FALSE(X.isApprox(X_copy, 1e-10));
    EXPECT_NEAR(X.col(0).mean(), 0.0, 1e-10);
    EXPECT_NEAR(X.col(0).norm(), 1.0, 1e-10);
}

// ========================================================================================


/** @brief Verify that execution over disk-mapped data succeeds correctly */
TEST(TSolverMmapTest, ValidateDiskMappedDataExecution) {
    const std::string X_file = "test_tsolvers_mmap_X.bin";
    const std::string D_file = "test_tsolvers_mmap_D.bin";
    const std::string y_file = "test_tsolvers_mmap_y.bin";

    // Clean up if files left over from previous aborted run
    if (fs::exists(X_file)) fs::remove(X_file);
    if (fs::exists(D_file)) fs::remove(D_file);
    if (fs::exists(y_file)) fs::remove(y_file);

    {
        // 1. Generate mapped data directly on disk using SyntheticDataMapped
        SyntheticDataMapped data(
            X_file, D_file, y_file,
            100, 20, 20,
            {1, 2}, {5.0, 2.0},
            1.0, 42
        );

        // 2. Retrieve Eigen::Map objects mapped over the file descriptors
        auto X_map = data.getX();
        auto D_map = data.getD();
        Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

        // 3. Feed the memory-mapped views to the solver
        TLARS_Solver solver(X_map, D_map, y_map,
                            true, true, false);
        solver.executeStep(5, true);

        // 4. Verify that execution over mapped memory succeeds correctly
        EXPECT_EQ(solver.getActiveDummyIndices().size(), 5);
        EXPECT_FALSE(solver.getActions().empty());
        EXPECT_GT(solver.getRSS().front(), solver.getRSS().back());

    } // the scoping block ensures SyntheticDataMapped instances destruct
      // and close their file handles before we attempt fs::remove()

    // 5. Clean up temporary files
    EXPECT_TRUE(fs::remove(X_file));
    EXPECT_TRUE(fs::remove(D_file));
    EXPECT_TRUE(fs::remove(y_file));
}
