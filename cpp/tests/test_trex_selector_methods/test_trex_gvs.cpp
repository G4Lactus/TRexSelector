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

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_gvs {

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

    TRexGVSControlParameter gvs_params;

    // EN requires TENET. Giving it TLASSO should throw.
    gvs_params.gvs_type = GVSType::EN;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TLASSO;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);

    // IEN requires TLASSO. Giving it TENET should throw.
    gvs_params.gvs_type = GVSType::IEN;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TENET;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);
}


/** @brief Construct a new test f object to validate permutation L-Loop constraints. */
TEST_F(TRexGVSTest, Validation_ThrowsOnPermutationLLoop) {
    TRexGVSControlParameter gvs_params;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TENET;

    // Permutation is mechanically invalid for GVS MVN dummies
    gvs_params.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);

    gvs_params.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION_DIRECT;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);
}


/** @brief Construct a new test f object to validate GVS-specific controls. */
TEST_F(TRexGVSTest, Validation_ThrowsOnInvalidGVSParameters) {
    TRexGVSControlParameter gvs_params;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TENET; // Default to valid

    // Invalid corr_max
    gvs_params.corr_max = 1.5;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);
    gvs_params.corr_max = 0.5; // Reset

    // Bad prior groups length
    gvs_params.prior_groups = {0, 1, 1}; // length must match p = 10
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);

    // Correct length, but non-contiguous
    gvs_params.prior_groups = {
        0, 0, 0, 0, 0, 2, 2, 2, 2, 2}; // Missing 1
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);

    // Right length, contiguous
    gvs_params.prior_groups = {
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    EXPECT_NO_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params));
}


/** @brief Test that unsupported solver types (e.g., TLARS) are rejected. */
TEST_F(TRexGVSTest, Validation_ThrowsOnUnsupportedSolver_TLARS) {
    TRexGVSControlParameter gvs_params;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TLARS; // TLARS is illegal for GVS

    // Test rejection for EN
    gvs_params.gvs_type = GVSType::EN;
    EXPECT_THROW({
        TRexGVSSelector selector(X_map, y_map, 0.1, gvs_params);
    }, std::invalid_argument);

    // Test rejection for IEN
    gvs_params.gvs_type = GVSType::IEN;
    EXPECT_THROW({
        TRexGVSSelector selector(X_map, y_map, 0.1, gvs_params);
    }, std::invalid_argument);
}


/** @brief A fully-defaulted TRexGVSControlParameter (no field overrides) must
 *  construct without throwing: the nested trex_ctrl's smart default
 *  (solver_type = TENET) matches the default gvs_type = EN / en_solver =
 *  TENET, unlike the base class's own default (TLARS). */
TEST_F(TRexGVSTest, Construction_DefaultParametersDoNotThrow) {
    EXPECT_NO_THROW({
        TRexGVSSelector selector(X_map, y_map, 0.1, TRexGVSControlParameter(),
                                 42, false);
    });
}


// ========================================================================================
// Data Integrity
// ========================================================================================

/** @brief X is restored to its original values after select() returns (object still alive). */
TEST_F(TRexGVSTest, DataIntegrity_XRestoredAfterSelect) {
    Eigen::MatrixXd X_copy = X;

    TRexGVSControlParameter gvs_params;
    gvs_params.gvs_type = GVSType::EN;
    gvs_params.trex_ctrl.K = 3;
    gvs_params.trex_ctrl.max_dummy_multiplier = 2;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TENET;

    TRexGVSSelector selector(X_map, y_map, 0.1, gvs_params, 42, false);
    selector.select();

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored after TRexGVSSelector::select().";
}


/** @brief X is restored to its original values when the object is destroyed without
 *         calling select() (normalization happens in the constructor). */
TEST_F(TRexGVSTest, DataIntegrity_XRestoredOnDestruction) {
    Eigen::MatrixXd X_copy = X;

    TRexGVSControlParameter gvs_params;
    gvs_params.gvs_type = GVSType::EN;
    gvs_params.trex_ctrl.K = 3;
    gvs_params.trex_ctrl.max_dummy_multiplier = 2;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TENET;

    {
        TRexGVSSelector selector(X_map, y_map, 0.1, gvs_params, 42, false);
        // X is now normalized. Destructor fires here.
    }

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored by TRexGVSSelector destructor.";
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_gvs */
