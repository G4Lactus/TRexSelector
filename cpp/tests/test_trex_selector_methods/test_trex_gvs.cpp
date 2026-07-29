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

    // IEN requires TIENET or TIENET_AUG. Giving it TENET should throw.
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

    // IEN + TIENET (native pathwise) constructs.
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TIENET;
    EXPECT_NO_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params));
}


/** @brief EN selects its solver via en_solver alone; solver_type is derived.
 *  The legacy pattern (EN-family solver_type conflicting with en_solver)
 *  must throw instead of being silently downgraded. */
TEST_F(TRexGVSTest, Validation_EnSolverIsTheSingleENAxis) {

    // en_solver alone selects each variant (solver_type left at default).
    for (auto en : {ENSolverType::TENET, ENSolverType::TENET_AUG,
                    ENSolverType::TCENET}) {
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::EN;
        ctrl.en_solver = en;
        EXPECT_NO_THROW(TRexGVSSelector(X_map, y_map, 0.1, ctrl));
    }

    // Consistent explicit solver_type is tolerated.
    {
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::EN;
        ctrl.en_solver = ENSolverType::TCENET;
        ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TCENET;
        EXPECT_NO_THROW(TRexGVSSelector(X_map, y_map, 0.1, ctrl));
    }

    // Legacy conflict: solver_type = TCENET with en_solver = TENET throws
    // (pre-rework this silently selected the CCD variant via solver_type).
    {
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::EN;
        ctrl.en_solver = ENSolverType::TENET;
        ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TCENET;
        EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, ctrl),
                     std::invalid_argument);
    }

    // Conflict in the other direction: solver_type = TENET_AUG with
    // en_solver = TCENET throws as well.
    {
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::EN;
        ctrl.en_solver = ENSolverType::TCENET;
        ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TENET_AUG;
        EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, ctrl),
                     std::invalid_argument);
    }
}


/** @brief Construct a new test f object to validate permutation L-Loop constraints. */
TEST_F(TRexGVSTest, Validation_ThrowsOnPermutationLLoop) {
    TRexGVSControlParameter gvs_params;
    gvs_params.trex_ctrl.solver_type = SolverTypeForTRex::TENET;

    // Permutation is mechanically invalid for GVS MVN dummies
    gvs_params.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION;
    EXPECT_THROW(TRexGVSSelector(X_map, y_map, 0.1, gvs_params),
                 std::invalid_argument);

    gvs_params.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION_SEEDED;
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
                        double fixed_lambda2 = -1.0,
                        LambdaSelectionMethod lambda2_method =
                            LambdaSelectionMethod::CV_1SE_CCD) {
    const Eigen::Index n = 150, p = 40;
    GroupedData d(n, p, 0.75, 1.0, 7);
    Eigen::Map<Eigen::MatrixXd> X_map(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(d.y.data(), d.y.size());

    TRexGVSControlParameter ctrl;
    ctrl.gvs_type  = gvs_type;
    ctrl.en_solver = en_solver;
    ctrl.tenet_aug_use_lars = use_lars_inner;
    ctrl.lambda_2 = fixed_lambda2;   // < 0 -> auto (CV via lambda2_method)
    ctrl.lambda2_method = lambda2_method;
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

/** @brief IEN with the native pathwise TIENET solver recovers the planted
 *  groups (same lambda_2 pinning rationale as the TIENETAug test above). */
TEST_F(TRexGVSTest, EndToEnd_IEN_TIENET) {
    run_gvs_select_e2e(GVSType::IEN, ENSolverType::TENET,
                       SolverTypeForTRex::TIENET,
                       /*use_lars_inner=*/false, /*fixed_lambda2=*/1.0);
}

/** @brief EN with the CCD solver (TCENET) recovers the planted groups
 *  (selected via en_solver, the single EN axis). */
TEST_F(TRexGVSTest, EndToEnd_EN_TCENET) {
    run_gvs_select_e2e(GVSType::EN, ENSolverType::TCENET,
                       SolverTypeForTRex::TENET);
}

/** @brief IEN with the CCD solver (TCIENET) recovers the planted groups
 *  (same lambda_2 pinning rationale as the other IEN tests). */
TEST_F(TRexGVSTest, EndToEnd_IEN_TCIENET) {
    run_gvs_select_e2e(GVSType::IEN, ENSolverType::TENET,
                       SolverTypeForTRex::TCIENET,
                       /*use_lars_inner=*/false, /*fixed_lambda2=*/1.0);
}


/** @brief THE auto-lambda_2 milestone: with the IEN-geometry profiled CV
 *  (ienet_cv_ccd, 1SE) the IEN track recovers the planted groups WITHOUT a
 *  hand-pinned lambda_2 — the EN-shaped ridge CV chose ~26 on this design
 *  and collapsed the selection (see EndToEnd_IEN_TIENETAug's rationale). */
TEST_F(TRexGVSTest, EndToEnd_IEN_AutoLambda2_IenCcd) {
    run_gvs_select_e2e(GVSType::IEN, ENSolverType::TENET,
                       SolverTypeForTRex::TIENET,
                       /*use_lars_inner=*/false, /*fixed_lambda2=*/-1.0,
                       LambdaSelectionMethod::CV_1SE_IEN_CCD);
}

/** @brief Auto-lambda_2 via the analytic Tikhonov backbone (n > p here, so
 *  the contrast block cannot interpolate and the curve is informative). */
TEST_F(TRexGVSTest, EndToEnd_IEN_AutoLambda2_TikSvd) {
    run_gvs_select_e2e(GVSType::IEN, ENSolverType::TENET,
                       SolverTypeForTRex::TIENET,
                       /*use_lars_inner=*/false, /*fixed_lambda2=*/-1.0,
                       LambdaSelectionMethod::CV_1SE_TIK_SVD);
}


/** @brief The IEN-geometry lambda_2 methods require gvs_type = IEN (they
 *  consume the group structure); EN must reject them at select() time. */
TEST_F(TRexGVSTest, Validation_ThrowsOnIenLambdaMethodForEN) {
    GroupedData d(150, 40, 0.75, 1.0, 7);
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    TRexGVSControlParameter ctrl;
    ctrl.gvs_type = GVSType::EN;
    ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_IEN_CCD;
    ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TENET;
    ctrl.trex_ctrl.K = 3;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;

    TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 42, false);
    EXPECT_THROW(trex.select(), std::invalid_argument);
}


/** @brief The CCD IEN solver and the native pathwise LARS-IEN solve the same
 *  problem; with identical seeds and dummies the crossing supports — and
 *  hence the selections — must coincide. */
TEST_F(TRexGVSTest, EndToEnd_TCIENETMatchesTIENET) {
    const Eigen::Index n = 150, p = 40;

    auto run_variant = [&](SolverTypeForTRex st) {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::IEN;
        ctrl.lambda_2  = 1.0;                    // FIXED, > 0
        ctrl.trex_ctrl.solver_type = st;
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_ccd  = run_variant(SolverTypeForTRex::TCIENET);
    const Eigen::VectorXi sel_lars = run_variant(SolverTypeForTRex::TIENET);

    EXPECT_TRUE((sel_ccd.array() == sel_lars.array()).all())
        << "TCIENET and TIENET selected different variable sets.\n"
        << "  TCIENET n_selected = " << sel_ccd.sum() << "\n"
        << "  TIENET  n_selected = " << sel_lars.sum();
}


/** @brief The CCD EN solver and the Gram-based pathwise TENET solve the same
 *  problem; with identical seeds, dummies, and a fixed lambda_2 the
 *  selections must coincide. */
TEST_F(TRexGVSTest, EndToEnd_TCENETMatchesTENET) {
    const Eigen::Index n = 150, p = 40;

    auto run_variant = [&](ENSolverType en_solver) {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::EN;
        ctrl.en_solver = en_solver;              // single EN axis
        ctrl.lambda_2  = 1.0;                    // FIXED, > 0
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_ccd  = run_variant(ENSolverType::TCENET);
    const Eigen::VectorXi sel_lars = run_variant(ENSolverType::TENET);

    EXPECT_TRUE((sel_ccd.array() == sel_lars.array()).all())
        << "TCENET and TENET selected different variable sets.\n"
        << "  TCENET n_selected = " << sel_ccd.sum() << "\n"
        << "  TENET  n_selected = " << sel_lars.sum();
}


/** @brief The native pathwise TIENET and the row-augmented TIENETAug are
 *  mathematically equivalent; with identical seeds and dummies the two IEN
 *  sub-variants must select the same variables. */
TEST_F(TRexGVSTest, EndToEnd_TIENETMatchesTIENETAug) {
    const Eigen::Index n = 150, p = 40;

    auto run_variant = [&](SolverTypeForTRex st) {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::IEN;
        ctrl.lambda_2  = 1.0;                    // FIXED, > 0
        ctrl.trex_ctrl.solver_type = st;
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_native = run_variant(SolverTypeForTRex::TIENET);
    const Eigen::VectorXi sel_aug    = run_variant(SolverTypeForTRex::TIENET_AUG);

    EXPECT_TRUE((sel_native.array() == sel_aug.array()).all())
        << "TIENET and TIENETAug selected different variable sets.\n"
        << "  TIENET    n_selected = " << sel_native.sum() << "\n"
        << "  TIENETAug n_selected = " << sel_aug.sum();
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


/** @brief DIAGNOSTIC: equivalence must also hold at a FIXED lambda_2 (the doc
 *  claims mathematical equivalence for any lambda2 > 0). Reproduces the R
 *  conditions where a divergence was observed: fixed lambda_2 = 0.1, small
 *  design, default-ish K / dummy multiplier. */
TEST_F(TRexGVSTest, EndToEnd_TENETAugMatchesTENET_FixedLambda2) {
    const Eigen::Index n = 60, p = 15;

    // Unstructured iid-Gaussian design with signal in a few columns, mirroring
    // the R reproduction (which diverged). Built once, shared by both variants.
    Eigen::MatrixXd Xd(n, p);
    Eigen::VectorXd yd(n);
    {
        std::mt19937 rng(123);
        std::normal_distribution<double> N01(0.0, 1.0);
        for (Eigen::Index j = 0; j < p; ++j)
            for (Eigen::Index i = 0; i < n; ++i) Xd(i, j) = N01(rng);
        for (Eigen::Index i = 0; i < n; ++i)
            yd(i) = 3.0 * Xd(i, 0) - 2.0 * Xd(i, 1) + 2.0 * Xd(i, 4) + N01(rng);
    }

    auto run_variant = [&](ENSolverType en_solver, SolverTypeForTRex st) {
        Eigen::Map<Eigen::MatrixXd> Xm(Xd.data(), Xd.rows(), Xd.cols());
        Eigen::Map<Eigen::VectorXd> ym(yd.data(), yd.size());
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type  = GVSType::EN;
        ctrl.en_solver = en_solver;
        ctrl.lambda_2  = 0.1;                    // FIXED, > 0
        ctrl.trex_ctrl.solver_type = st;
        ctrl.trex_ctrl.K = 20;
        ctrl.trex_ctrl.max_dummy_multiplier = 10;
        TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 1, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_tenet =
        run_variant(ENSolverType::TENET, SolverTypeForTRex::TENET);
    const Eigen::VectorXi sel_aug =
        run_variant(ENSolverType::TENET_AUG, SolverTypeForTRex::TENET_AUG);

    EXPECT_TRUE((sel_tenet.array() == sel_aug.array()).all())
        << "TENET and TENETAug diverged at fixed lambda_2 = 0.1.\n"
        << "  TENET     n_selected = " << sel_tenet.sum() << "\n"
        << "  TENETAug  n_selected = " << sel_aug.sum();
}


// ========================================================================================
// ZSCORE scaling mode (2026-07: GVS made mode-complete)
// ========================================================================================

namespace {

/** @brief Run one GVS EN selection on shared grouped data with the given
 *         scaling mode; auto-CV lambda_2 (< 0 sentinel). Copies X/y because
 *         the selector normalizes X in place. */
Eigen::VectorXi run_gvs_with_mode(
    const GroupedData& d,
    trex::trex_selector_methods::utils::data_normalizer::ScalingMode mode,
    ENSolverType en_solver,
    SolverTypeForTRex st,
    int seed) {

    Eigen::MatrixXd Xc = d.X;
    Eigen::VectorXd yc = d.y;
    Eigen::Map<Eigen::MatrixXd> Xm(Xc.data(), Xc.rows(), Xc.cols());
    Eigen::Map<Eigen::VectorXd> ym(yc.data(), yc.size());

    TRexGVSControlParameter ctrl;
    ctrl.gvs_type  = GVSType::EN;
    ctrl.en_solver = en_solver;
    ctrl.lambda_2  = -1.0;  // auto (ridge CV)
    ctrl.trex_ctrl.solver_type  = st;
    ctrl.trex_ctrl.K = 5;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;
    ctrl.trex_ctrl.scaling_mode = mode;

    TRexGVSSelector trex(Xm, ym, 0.2, ctrl, seed, false);
    return trex.select().selected_var;
}

} // namespace


/** @brief GVS end-to-end under ZSCORE: the whole pipeline (X normalization,
 *         scale-adaptive Sigma, MVN dummies, working-scale lambda_2) must
 *         recover the planted groups just like the L2 run. */
TEST_F(TRexGVSTest, EndToEnd_EN_TENET_ZScore) {
    namespace dnm = trex::trex_selector_methods::utils::data_normalizer;
    GroupedData d(150, 40, 0.75, 1.0, 7);

    const Eigen::VectorXi sel = run_gvs_with_mode(
        d, dnm::ScalingMode::ZSCORE,
        ENSolverType::TENET, SolverTypeForTRex::TENET, 42);

    int hits = 0, false_sel = 0;
    for (Eigen::Index j = 0; j < sel.size(); ++j) {
        if (sel(j) == 1) { (j < 6 ? hits : false_sel)++; }
    }
    EXPECT_GE(hits, 4) << "ZSCORE GVS failed to recover the planted actives.";
    EXPECT_LE(false_sel, 4) << "ZSCORE GVS selected too many nulls.";
}


/** @brief Scaling-mode parity: with auto-CV lambda_2 the ZSCORE run must
 *         select EXACTLY the same variables as the L2 run. lambda_cv is
 *         computed on the CV's internal unit-L2 scale in both modes; the
 *         (n-1) working-scale conversion in computeLambda2 makes the ZSCORE
 *         augmented system a global scalar multiple of the L2 one, and LARS
 *         paths are invariant under global column scaling. */
TEST_F(TRexGVSTest, ScalingModeParity_EN_AutoLambda2) {
    namespace dnm = trex::trex_selector_methods::utils::data_normalizer;
    GroupedData d(150, 40, 0.75, 1.0, 7);

    const std::pair<ENSolverType, SolverTypeForTRex> variants[] = {
        {ENSolverType::TENET,     SolverTypeForTRex::TENET},
        {ENSolverType::TENET_AUG, SolverTypeForTRex::TENET_AUG},
    };
    for (const auto& [en_solver, st] : variants) {
        const Eigen::VectorXi sel_l2 =
            run_gvs_with_mode(d, dnm::ScalingMode::L2, en_solver, st, 42);
        const Eigen::VectorXi sel_z =
            run_gvs_with_mode(d, dnm::ScalingMode::ZSCORE, en_solver, st, 42);

        EXPECT_TRUE((sel_l2.array() == sel_z.array()).all())
            << "L2 vs ZSCORE selections diverged (solver "
            << static_cast<int>(st) << ").\n"
            << "  L2     n_selected = " << sel_l2.sum() << "\n"
            << "  ZSCORE n_selected = " << sel_z.sum();
    }
}


/** @brief IEN scaling-mode parity at FIXED lambda_2. A fixed lambda_2 is
 *         interpreted in working-scale units, so L2 with lambda_2 and ZSCORE
 *         with (n-1) * lambda_2 describe the same group-augmented geometry
 *         up to one global column-scale factor sqrt(n-1) (the MVN dummies
 *         scale along automatically: Sigma_m is estimated on the normalized
 *         X, and the same seed reproduces the same standard-normal draws).
 *         LARS paths are invariant under global column scaling, so the
 *         selections must be identical — for BOTH IEN solver variants. */
TEST_F(TRexGVSTest, ScalingModeParity_IEN_FixedLambda2) {
    namespace dnm = trex::trex_selector_methods::utils::data_normalizer;
    const Eigen::Index n = 150, p = 40;
    const double lambda2_l2 = 1.0;

    auto run_ien = [&](dnm::ScalingMode mode, double lambda2,
                       SolverTypeForTRex st) {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());
        TRexGVSControlParameter ctrl;
        ctrl.gvs_type = GVSType::IEN;
        ctrl.lambda_2 = lambda2;                 // FIXED, working-scale units
        ctrl.trex_ctrl.solver_type = st;
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        ctrl.trex_ctrl.scaling_mode = mode;
        TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    for (SolverTypeForTRex st : {SolverTypeForTRex::TIENET,
                                 SolverTypeForTRex::TIENET_AUG}) {
        const Eigen::VectorXi sel_l2 =
            run_ien(dnm::ScalingMode::L2, lambda2_l2, st);
        const Eigen::VectorXi sel_z =
            run_ien(dnm::ScalingMode::ZSCORE,
                    static_cast<double>(n - 1) * lambda2_l2, st);

        EXPECT_TRUE((sel_l2.array() == sel_z.array()).all())
            << "IEN L2 vs ZSCORE selections diverged (solver "
            << static_cast<int>(st) << ").\n"
            << "  L2     n_selected = " << sel_l2.sum() << "\n"
            << "  ZSCORE n_selected = " << sel_z.sum();
    }
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_gvs */
