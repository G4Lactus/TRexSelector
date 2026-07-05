// =================================================================================
// test_trex_gvs.cpp
// =================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen include
#include <Eigen/Dense>

// std includes
#include <cmath>
#include <random>
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

    // IEN requires TIENET_AUG. Giving it TENET should throw.
    gvs_params.gvs_type = GVSType::IEN;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TENET;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);

    // ... and so should the pre-rework TLASSO requirement.
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TLASSO;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);

    // IEN + TIENET_AUG constructs.
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TIENET_AUG;
    EXPECT_NO_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params));
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
// End-to-end selection (grouped data)
// ========================================================================================

namespace {

/** @brief Grouped synthetic design: two equicorrelated active groups (cols
 *  0-2 and 3-5, within-group correlation ~rho) among iid null predictors. */
struct GroupedData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    GroupedData(Eigen::Index n, Eigen::Index p, double rho, double sigma,
                unsigned int seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> N01(0.0, 1.0);

        X.resize(n, p);
        const double a = std::sqrt(rho);
        const double b = std::sqrt(1.0 - rho);
        for (int g = 0; g < 2; ++g) {
            Eigen::VectorXd shared(n);
            for (Eigen::Index i = 0; i < n; ++i) { shared(i) = N01(rng); }
            for (Eigen::Index j = 0; j < 3; ++j) {
                for (Eigen::Index i = 0; i < n; ++i) {
                    X(i, 3 * g + j) = a * shared(i) + b * N01(rng);
                }
            }
        }
        for (Eigen::Index j = 6; j < p; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) { X(i, j) = N01(rng); }
        }

        // y = sum(group A) - sum(group B) + noise.
        y.resize(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            double signal = 0.0;
            for (Eigen::Index j = 0; j < 3; ++j) { signal += X(i, j); }
            for (Eigen::Index j = 3; j < 6; ++j) { signal -= X(i, j); }
            y(i) = signal + sigma * N01(rng);
        }
    }
};

/** @brief Run select() end-to-end for one GVS variant and check that the
 *  planted active groups are recovered with few false selections. */
void run_gvs_select_e2e(GVSType gvs_type,
                        ENSolverType en_solver,
                        SolverTypeForTRex solver_type,
                        bool use_lars_inner = false,
                        double fixed_lambda2 = -1.0) {
    const Eigen::Index n = 150, p = 40;
    GroupedData d(n, p, 0.75, 1.0, 7);
    Eigen::Map<Eigen::MatrixXd> X_map(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(d.y.data(), d.y.size());

    TRexGVSControlParameter ctrl;
    ctrl.gvs_type  = gvs_type;
    ctrl.en_solver = en_solver;
    ctrl.tenet_aug_use_lars = use_lars_inner;
    ctrl.lambda_2 = fixed_lambda2;   // < 0 -> auto (ridge CV)
    ctrl.trex_ctrl.solver_type = solver_type;
    ctrl.trex_ctrl.K = 5;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;

    TRexGVSSelector trex(X_map, y_map, 0.2, ctrl, 42, false);
    auto result = trex.select();
    const auto& g = trex.getGVSResult();

    // Structural checks.
    ASSERT_EQ(result.selected_var.size(), p);
    EXPECT_EQ(g.gvs_type, gvs_type);
    EXPECT_GT(g.lambda2_used, 0.0);
    if (fixed_lambda2 > 0.0) { EXPECT_DOUBLE_EQ(g.lambda2_used, fixed_lambda2); }
    EXPECT_GE(g.max_clusters, 2u);
    EXPECT_EQ(g.groups_vec.size(), p);
    for (Eigen::Index j = 0; j < result.Phi_prime.size(); ++j) {
        EXPECT_GE(result.Phi_prime(j), 0.0);
        EXPECT_LE(result.Phi_prime(j), 1.0);
    }

    // Recovery: most planted actives (cols 0-5) selected, few nulls.
    int hits = 0, false_sel = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
        if (result.selected_var(j) == 1) {
            if (j < 6) { ++hits; } else { ++false_sel; }
        }
    }
    EXPECT_GE(hits, 4)
        << "GVS variant failed to recover the planted active groups.";
    EXPECT_LE(false_sel, 4)
        << "GVS variant selected too many null variables.";
}

} // namespace


/** @brief EN with the Gram-based TENET solver recovers the planted groups. */
TEST_F(TRexGVSTest, EndToEnd_EN_TENET) {
    run_gvs_select_e2e(GVSType::EN, ENSolverType::TENET,
                       SolverTypeForTRex::TENET);
}

/** @brief EN with the augmented-LASSO TENETAug solver recovers the planted
 *  groups. */
TEST_F(TRexGVSTest, EndToEnd_EN_TENETAug) {
    run_gvs_select_e2e(GVSType::EN, ENSolverType::TENET_AUG,
                       SolverTypeForTRex::TENET_AUG);
}

/** @brief EN-aug with the pure-LARS inner solver (R type="lar" parity flag)
 *  recovers the planted groups — also exercises tenet_aug_use_lars. */
TEST_F(TRexGVSTest, EndToEnd_EN_TENETAug_LarsInner) {
    run_gvs_select_e2e(GVSType::EN, ENSolverType::TENET_AUG,
                       SolverTypeForTRex::TENET_AUG, /*use_lars_inner=*/true);
}

/** @brief IEN (TIENETAug solver) recovers the planted groups.
 *
 *  lambda_2 is pinned instead of CV-resolved: the IEN group-indicator rows
 *  scale with sqrt(lambda_2), and the ridge-CV value (~26 in LARS units on
 *  this n > p design; identical to what the R reference would choose, since
 *  both use the same CV and the same * p/2 conversion for EN and IEN) makes
 *  the shared bottom rows dominate the augmented geometry — dummies become
 *  nearly collinear with their actives and nothing is selectable. Recovery
 *  is perfect for lambda_2 <~ 2 on this design. Known lambda_2 sensitivity
 *  of the IEN track, inherited from the reference (see computeLambda2()). */
TEST_F(TRexGVSTest, EndToEnd_IEN_TIENETAug) {
    run_gvs_select_e2e(GVSType::IEN, ENSolverType::TENET,
                       SolverTypeForTRex::TIENET_AUG,
                       /*use_lars_inner=*/false, /*fixed_lambda2=*/1.0);
}


/** @brief TENET (Gram-based EN) and TENETAug (augmented LASSO) are
 *  mathematically equivalent for lambda2 > 0; with identical seeds and
 *  dummies the two EN sub-variants must select the same variables. */
TEST_F(TRexGVSTest, EndToEnd_TENETAugMatchesTENET) {
    const Eigen::Index n = 150, p = 40;

    auto run_variant = [&](ENSolverType en_solver, SolverTypeForTRex st) {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::EN;
        ctrl.en_solver = en_solver;
        ctrl.trex_ctrl.solver_type = st;
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_tenet =
        run_variant(ENSolverType::TENET, SolverTypeForTRex::TENET);
    const Eigen::VectorXi sel_aug =
        run_variant(ENSolverType::TENET_AUG, SolverTypeForTRex::TENET_AUG);

    EXPECT_TRUE((sel_tenet.array() == sel_aug.array()).all())
        << "TENET and TENETAug selected different variable sets.";
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_gvs */
