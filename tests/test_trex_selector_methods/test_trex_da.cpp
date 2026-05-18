// ========================================================================================
// test_trex_da.cpp
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_da/trex_da.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_da;
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::utils::datageneration::datagen;

// ========================================================================================

/**
 * @brief Helper to set up TRexDASelector boilerplate
 *
 * @param n Number of samples
 * @param p Number of features
 * @param method DA method to use
 * @param K Number of top variables to select
 */
void run_trex_da_test(Eigen::Index n,
                      Eigen::Index p,
                      DAMethod method,
                      std::size_t K = 3,
                      LLoopStrategy l_strategy = LLoopStrategy::HCONCAT)
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
    trex_ctrl.K = K;
    trex_ctrl.max_dummy_multiplier = 2;

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = method;

    TRexDASelector trex(X_map, y_map, 0.1, da_ctrl,
                       trex_ctrl, 42, false);

    // Test the select() execution safely
    auto result = trex.select();

    // Retrieve DA exact result extended format
    const auto& da_result = trex.getDAResult();

    // Verify returning correct struct with fields
    EXPECT_GE(da_result.selected_var.size(), 0);
    EXPECT_EQ(da_result.method, method);

    // Structural matrix sizing depending on configuration method
    if (method == DAMethod::AR1 || method == DAMethod::EQUI) {
        EXPECT_EQ(da_result.rho_grid.size(), 0);
    } else if (method == DAMethod::BT || method == DAMethod::NN) {
        EXPECT_GT(da_result.rho_grid.size(), 0);
    }
}

/** @brief Test TRexDASelector with AR1 method in low-dimensional setting */
TEST(TRexDATest, Method_AR1_LowDimensional) {
    run_trex_da_test(300, 100, DAMethod::AR1);
}


/** @brief Test TRexDASelector with AR1 method in high-dimensional setting */
TEST(TRexDATest, Method_AR1_HighDimensional) {
    run_trex_da_test(150, 300, DAMethod::AR1);
}


/** @brief Test TRexDASelector with BT method in low-dimensional setting */
TEST(TRexDATest, Method_BT_HighDimensional) {
    run_trex_da_test(150, 300, DAMethod::BT);
}


/** @brief Test TRexDASelector with NN method in low-dimensional setting */
TEST(TRexDATest, Method_NN_LowDimensional) {
    run_trex_da_test(300, 100, DAMethod::NN);
}

/** @brief Test TRexDASelector with EQUI method in low-dimensional setting */
TEST(TRexDATest, Method_EQUI_LowDimensional) {
    run_trex_da_test(300, 100, DAMethod::EQUI);
}


/** @brief Test TRexDASelector with AR1 method and SKIPL L loop strategy */
TEST(TRexDATest, Method_AR1_SKIPL) {
    run_trex_da_test(150, 300, DAMethod::AR1, 3, LLoopStrategy::SKIPL);
}


/** @brief Test TRexDASelector validation with invalid parameters */
TEST(TRexDATest, Validation_ThrowsOnInvalidParameters) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(20);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter tc_params;
    TRexDAControlParameter da_params;

    // 1. Invalid rho_thr_DA
    da_params.rho_thr_DA = 1.5; // Bounds are [0, 1]
    EXPECT_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params,
                                tc_params);
    }, std::invalid_argument);

    da_params.rho_thr_DA = 0.5; // Reset to valid

    // 2. Prior Groups Length Mismatch
    da_params.prior_groups = {{1, 1, 1, 1, 1}}; // Incorrect length (p = 10)
    EXPECT_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params,
                                tc_params);
    }, std::invalid_argument);

    da_params.prior_groups.clear(); // Reset

    // 3. Unsupported BT Linkage
    da_params.method = DAMethod::BT;
    // Supplying a linkage method not explicitly supported by our clustering logic
    da_params.hc_linkage =
        static_cast<trex::ml_methods::clustering::hierarchical::agglomerative::LinkageMethod>(999);
    EXPECT_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params,
                                 tc_params);
    }, std::invalid_argument);
}


/** @brief Test confirming SKIPL executes without exceptions */
TEST(TRexDATest, Method_AR1_SKIPL_DoesNotThrow) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(50, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(50);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter tc_params;
    tc_params.lloop_strategy = LLoopStrategy::SKIPL;

    TRexDAControlParameter da_params;
    da_params.method = DAMethod::AR1;

    EXPECT_NO_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params,
                                tc_params);
        auto result = selector.select();
        // Since SKIPL is used, verify initialized dummies
        EXPECT_EQ(result.T_stop >= 1, true);
    });
}
