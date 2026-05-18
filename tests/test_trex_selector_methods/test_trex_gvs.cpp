// =================================================================================
// test_trex_gvs.cpp
// =================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen include
#include <Eigen/Dense>

// std includes
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>

// =================================================================================

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_gvs;
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::trex_selector_methods::utils::solver_dispatch;
using namespace trex::ml_methods::clustering::hierarchical::agglomerative;

// =================================================================================

class TRexGVSTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::Map<Eigen::MatrixXd> X_map;
    Eigen::Map<Eigen::VectorXd> y_map;

    TRexGVSTest()
        : X(Eigen::MatrixXd::Random(20, 10)),
          y(Eigen::VectorXd::Random(20)),
          X_map(X.data(), X.rows(), X.cols()),
          y_map(y.data(), y.size()) {}
};


/** @brief Construct a new test f object to validate constructor constraints. */
TEST_F(TRexGVSTest, Validation_ThrowsOnSolverMismatch) {

    TRexControlParameter tc_params;
    TRexGVSControlParameter gvs_params;

    // EN requires TENET. Giving it TLASSO should throw.
    gvs_params.gvs_type = GVSType::EN;
    tc_params.solver_type = SolverTypeForTRex::TLASSO;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
                                 tc_params), std::invalid_argument);

    // IEN requires TLASSO. Giving it TENET should throw.
    gvs_params.gvs_type = GVSType::IEN;
    tc_params.solver_type = SolverTypeForTRex::TENET;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
                                 tc_params), std::invalid_argument);
}


/** @brief Construct a new test f object to validate permutation L-Loop constraints. */
TEST_F(TRexGVSTest, Validation_ThrowsOnPermutationLLoop) {
    TRexControlParameter tc_params;
    TRexGVSControlParameter gvs_params;
    tc_params.solver_type = SolverTypeForTRex::TENET;

    // Permutation is mechanically invalid for GVS MVN dummies
    tc_params.lloop_strategy = LLoopStrategy::PERMUTATION;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
                                 tc_params), std::invalid_argument);

    tc_params.lloop_strategy = LLoopStrategy::PERMUTATION_DIRECT;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
                                 tc_params), std::invalid_argument);
}


/** @brief Construct a new test f object to validate GVS-specific controls. */
TEST_F(TRexGVSTest, Validation_ThrowsOnInvalidGVSParameters) {
    TRexControlParameter tc_params;
    tc_params.solver_type = SolverTypeForTRex::TENET; // Default to valid
    TRexGVSControlParameter gvs_params;

    // Invalid corr_max
    gvs_params.corr_max = 1.5;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
                                 tc_params), std::invalid_argument);
    gvs_params.corr_max = 0.5; // Reset

    // Bad prior groups length
    gvs_params.prior_groups = {0, 1, 1}; // length must match p = 10
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
                                 tc_params), std::invalid_argument);

    // Correct length, but non-contiguous
    gvs_params.prior_groups = {
        0, 0, 0, 0, 0, 2, 2, 2, 2, 2}; // Missing 1
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
                                 tc_params), std::invalid_argument);

    // Right length, contiguous
    gvs_params.prior_groups = {
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    EXPECT_NO_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params,
         tc_params));
}


/** @brief Test that unsupported solver types (e.g., TLARS) are rejected. */
TEST_F(TRexGVSTest, Validation_ThrowsOnUnsupportedSolver_TLARS) {
    TRexControlParameter tc_params;
    tc_params.solver_type = SolverTypeForTRex::TLARS; // TLARS is illegal for GVS

    TRexGVSControlParameter gvs_params;

    // Test rejection for EN
    gvs_params.gvs_type = GVSType::EN;
    EXPECT_THROW({
        TRexGVSSelector selector(X_map, y_map, 0.1, gvs_params,
                                tc_params);
    }, std::invalid_argument);

    // Test rejection for IEN
    gvs_params.gvs_type = GVSType::IEN;
    EXPECT_THROW({
        TRexGVSSelector selector(X_map, y_map, 0.1, gvs_params,
                                tc_params);
    }, std::invalid_argument);
}
