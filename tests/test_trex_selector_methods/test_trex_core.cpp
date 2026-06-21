// ========================================================================================
// test_trex_core.cpp
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_core/trex.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_core {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;

// ========================================================================================


/**
 * @brief Helper to set up TRexSelector boilerplate for testing.
 *
 * @param n Number of samples
 * @param p Number of features
 * @param solver Solver type for TRex
 * @param l_strategy L loop strategy
 * @param multi_threaded Whether to use multi-threading
 */
void run_trex_core_test(Eigen::Index n,
                        Eigen::Index p,
                        sd::SolverTypeForTRex solver = sd::SolverTypeForTRex::TLARS,
                        LLoopStrategy l_strategy = LLoopStrategy::HCONCAT,
                        bool multi_threaded = false)
{
    // Explicit vectors prevent template resolution errors in the constructor
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};

    SyntheticData data(n, p, support, coefs, 1.0, 42);

    auto X = data.getX();
    auto y = data.getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3; // Keep K small for fast testing
    trex_ctrl.max_dummy_multiplier = 2;
    trex_ctrl.solver_type = solver;
    trex_ctrl.lloop_strategy = l_strategy;

    if (multi_threaded) {
        omp_set_num_threads(2);
        trex_ctrl.parallel_rnd_experiments = true;
        trex_ctrl.autoConfigResources();
    }

    TRexSelector trex(X_map, y_map, 0.1, trex_ctrl, 42,
                      false);

    // Assert configuration getters
    EXPECT_EQ(trex.getTFDR(), 0.1);
    EXPECT_EQ(trex.getK(), 3);
    EXPECT_EQ(trex.getMaxNumDummies(), 2);

    // Test the select() execution safely
    auto result = trex.select();

    // Dimensions shouldn't crash and MUST match `p`
    EXPECT_EQ(result.Phi_prime.size(), p);
    EXPECT_GE(result.selected_var.size(), 0);

    if (result.T_stop > 0) {
        EXPECT_EQ(result.Phi_mat.cols(), p);
        EXPECT_EQ(result.Phi_mat.rows(), result.T_stop);
    }

    // Assert bounds on Phi_prime
    for (Eigen::Index j = 0; j < p; ++j) {
        EXPECT_GE(result.Phi_prime(j), 0.0);
        EXPECT_LE(result.Phi_prime(j), 1.0);
    }
}

/** @brief Test TRexSelector with low-dimensional data. */
TEST(TRexCoreTest, LowDimensional) {
    run_trex_core_test(300, 100);
}


/** @brief Test TRexSelector with high-dimensional data. */
TEST(TRexCoreTest, HighDimensional) {
    run_trex_core_test(150, 300);
}


/** @brief Test TRexSelector with TLASSO solver. */
TEST(TRexCoreTest, Param_Solvers_TLASSO) {
    run_trex_core_test(150, 200, sd::SolverTypeForTRex::TLASSO);
}


/** @brief Test TRexSelector with SKIPL L loop strategy. */
TEST(TRexCoreTest, Param_LStrategies_SKIPL) {
    run_trex_core_test(150, 200, sd::SolverTypeForTRex::TLARS,
                       LLoopStrategy::SKIPL);
}


/** @brief Test TRexSelector with concurrency enabled. */
TEST(TRexCoreTest, Param_Concurrency) {
    // Tests autoConfigResources logic and threaded OpenMP execution
    run_trex_core_test(150, 200, sd::SolverTypeForTRex::TLARS,
                       LLoopStrategy::HCONCAT, true);
}


/** @brief Test TRexSelector with invalid input dimensions. */
TEST(TRexCoreTest, Exceptions_InvalidDimensions) {
    Eigen::MatrixXd X_empty(0, 0);
    Eigen::VectorXd y_empty(0);
    Eigen::Map<Eigen::MatrixXd> X_map(X_empty.data(), 0, 0);
    Eigen::Map<Eigen::VectorXd> y_map(y_empty.data(), 0);

    TRexControlParameter trex_ctrl;
    EXPECT_THROW(TRexSelector(X_map, y_map, 0.1, trex_ctrl),
                 std::invalid_argument);
}


/** @brief Test TRexSelector with mismatched input dimensions. */
TEST(TRexCoreTest, Exceptions_MismatchedDimensions) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 50);
    Eigen::VectorXd y = Eigen::VectorXd::Random(99); // Mismatched
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    EXPECT_THROW(TRexSelector(X_map, y_map, 0.1, trex_ctrl),
                 std::invalid_argument);
}


/** @brief Test TRexSelector with invalid parameters. */
TEST(TRexCoreTest, Exceptions_InvalidParameters) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 50);
    Eigen::VectorXd y = Eigen::VectorXd::Random(100);
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;

    // Test invalid tFDR bounds
    EXPECT_THROW(TRexSelector(X_map, y_map, -0.1, trex_ctrl),
                 std::invalid_argument);
    EXPECT_THROW(TRexSelector(X_map, y_map, 1.5, trex_ctrl),
                 std::invalid_argument);

    // Test invalid K
    trex_ctrl.K = 0;
    EXPECT_THROW(TRexSelector(X_map, y_map, 0.1, trex_ctrl),
                 std::invalid_argument);
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_core */
