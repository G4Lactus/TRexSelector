// ========================================================================================
// test_trex_screening.cpp
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <stdexcept>
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_screening/trex_biobank_screening.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>


// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_screening {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace dn = trex::trex_selector_methods::utils::data_normalizer;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;

// ========================================================================================

class TRexScreeningTest : public::testing::Test {
protected:
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::Map<Eigen::MatrixXd> X_map;
    Eigen::Map<Eigen::VectorXd> y_map;

    TRexScreeningTest()
        : X(Eigen::MatrixXd::Random(50, 20)),
          y(Eigen::VectorXd::Random(50)),
          X_map(X.data(), X.rows(), X.cols()),
          y_map(y.data(), y.size()) {}
};


/** @brief Test of validation constraints */
TEST_F(TRexScreeningTest, Validation_ThrowsOnInvalidLLoopStrategy) {
    ScreenTRexControlParameter screen_params;

    // Supported: STANDARD, SEEDED, PERMUTATION, PERMUTATION_SEEDED.
    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
    EXPECT_NO_THROW(ScreenTRexSelector(X_map, y_map, screen_params));

    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::SEEDED;
    EXPECT_NO_THROW(ScreenTRexSelector(X_map, y_map, screen_params));

    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION;
    EXPECT_NO_THROW(ScreenTRexSelector(X_map, y_map, screen_params));

    screen_params.trex_ctrl.lloop_strategy =
        LLoopStrategy::PERMUTATION_SEEDED;
    EXPECT_NO_THROW(ScreenTRexSelector(X_map, y_map, screen_params));

    // Unsupported: the L-loop growth strategies.
    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::SKIPL;
    EXPECT_THROW(ScreenTRexSelector(X_map, y_map, screen_params),
                                    std::invalid_argument);

    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::HCONCAT;
    EXPECT_THROW(ScreenTRexSelector(X_map, y_map, screen_params),
                 std::invalid_argument);
}


/** @brief PERMUTATION and PERMUTATION_SEEDED key their base off the same
 *  resolved permutation_base_seed_, so for a fixed seed the stored and the
 *  stateless variants must produce bit-identical screening results. */
TEST_F(TRexScreeningTest, Equivalence_PermutationSeededMatchesPermutation) {

    auto run_with = [&](LLoopStrategy strategy) {
        Eigen::MatrixXd Xc = X;
        Eigen::VectorXd yc = y;
        Eigen::Map<Eigen::MatrixXd> Xm(Xc.data(), Xc.rows(), Xc.cols());
        Eigen::Map<Eigen::VectorXd> ym(yc.data(), yc.size());

        ScreenTRexControlParameter screen_params;
        screen_params.trex_method = ScreenTRexMethod::TREX;
        screen_params.use_bootstrap_CI = false;
        screen_params.trex_ctrl.K = 5;
        screen_params.trex_ctrl.lloop_strategy = strategy;

        ScreenTRexSelector selector(Xm, ym, screen_params, /*seed=*/42);
        return selector.select();
    };

    const auto res_perm = run_with(LLoopStrategy::PERMUTATION);
    const auto res_pod  = run_with(LLoopStrategy::PERMUTATION_SEEDED);

    ASSERT_EQ(res_perm.selected_var.size(), res_pod.selected_var.size());
    EXPECT_TRUE((res_perm.selected_var.array()
                 == res_pod.selected_var.array()).all())
        << "PERMUTATION_SEEDED selection diverged from PERMUTATION.";
    ASSERT_EQ(res_perm.Phi_prime.size(), res_pod.Phi_prime.size());
    EXPECT_TRUE(res_perm.Phi_prime.isApprox(res_pod.Phi_prime, 1e-14))
        << "PERMUTATION_SEEDED Phi diverged from PERMUTATION.";
}


/** @brief SEEDED regenerates the identical tag-0 draws STANDARD stores
 *  (screening runs at exactly L = p), one D_k at a time — the two must
 *  produce bit-identical screening results. */
TEST_F(TRexScreeningTest, Equivalence_SeededMatchesStandard) {

    auto run_with = [&](LLoopStrategy strategy) {
        Eigen::MatrixXd Xc = X;
        Eigen::VectorXd yc = y;
        Eigen::Map<Eigen::MatrixXd> Xm(Xc.data(), Xc.rows(), Xc.cols());
        Eigen::Map<Eigen::VectorXd> ym(yc.data(), yc.size());

        ScreenTRexControlParameter screen_params;
        screen_params.trex_method = ScreenTRexMethod::TREX;
        screen_params.use_bootstrap_CI = false;
        screen_params.trex_ctrl.K = 5;
        screen_params.trex_ctrl.lloop_strategy = strategy;

        ScreenTRexSelector selector(Xm, ym, screen_params, /*seed=*/42);
        return selector.select();
    };

    const auto res_std = run_with(LLoopStrategy::STANDARD);
    const auto res_sd  = run_with(LLoopStrategy::SEEDED);

    ASSERT_EQ(res_std.selected_var.size(), res_sd.selected_var.size());
    EXPECT_TRUE((res_std.selected_var.array()
                 == res_sd.selected_var.array()).all())
        << "SEEDED selection diverged from STANDARD.";
    ASSERT_EQ(res_std.Phi_prime.size(), res_sd.Phi_prime.size());
    EXPECT_TRUE(res_std.Phi_prime.isApprox(res_sd.Phi_prime, 1e-14))
        << "SEEDED Phi diverged from STANDARD.";
}


/** @brief Test execution of Ordinary Screening */
TEST_F(TRexScreeningTest, Execution_OrdinaryScreenTRexDoesNotThrow) {
    ScreenTRexControlParameter screen_params;
    screen_params.trex_method = ScreenTRexMethod::TREX;
    screen_params.use_bootstrap_CI = false;
    screen_params.trex_ctrl.K = 5; // Reduced K for quick unit testing
    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;

    EXPECT_NO_THROW({
        ScreenTRexSelector selector(X_map, y_map, screen_params);
        auto result = selector.select();

        // Ensure static execution values
        EXPECT_EQ(result.T_stop, 1);

        // num_dummies is locked to p
        EXPECT_EQ(result.num_dummies, 20);
    });
}


/** @brief Screening with the CD solvers (TCCD / TCENET): T is fixed at 1
 *  and no warm start engages, so this exercises the plain fresh-construction
 *  dispatch of the CD family under the screening loop. */
TEST_F(TRexScreeningTest, Execution_CdFamilyDoesNotThrow) {
    for (auto solver : {sd::SolverTypeForTRex::TCCD,
                        sd::SolverTypeForTRex::TCENET}) {
        ScreenTRexControlParameter screen_params;
        screen_params.trex_method = ScreenTRexMethod::TREX;
        screen_params.use_bootstrap_CI = false;
        screen_params.trex_ctrl.K = 5;
        screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
        screen_params.trex_ctrl.solver_type = solver;

        EXPECT_NO_THROW({
            ScreenTRexSelector selector(X_map, y_map, screen_params);
            auto result = selector.select();
            EXPECT_EQ(result.T_stop, 1);
            EXPECT_EQ(result.num_dummies, 20);
        });
    }
}


/** @brief Test of Boostrap CI Screening */
TEST_F(TRexScreeningTest, Execution_BootstrapScreenTRexDoesNotThrow) {
    ScreenTRexControlParameter screen_params;
    screen_params.trex_method = ScreenTRexMethod::TREX;
    screen_params.use_bootstrap_CI = true;
    screen_params.R_boot = 50; // Small bootstrap resamples for fast testing
    screen_params.trex_ctrl.K = 5;
    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;

    EXPECT_NO_THROW({
        ScreenTRexSelector selector(X_map, y_map, screen_params);

        auto result = selector.getScreenResult();
        selector.select();
    });
}

/// ========================================================================================
// Data Integrity
// ========================================================================================

/** @brief X is restored to its original values after select() returns (object still alive). */
TEST_F(TRexScreeningTest, DataIntegrity_XRestoredAfterSelect) {
    Eigen::MatrixXd X_copy = X;

    ScreenTRexControlParameter screen_params;
    screen_params.trex_ctrl.K = 3;
    screen_params.trex_ctrl.max_dummy_multiplier = 2;
    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;

    ScreenTRexSelector selector(X_map, y_map, screen_params, 42, false);
    selector.select();

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored after ScreenTRexSelector::select().";
}


/** @brief X is restored to its original values when the object is destroyed without
 *         calling select() (normalization happens in the constructor). */
TEST_F(TRexScreeningTest, DataIntegrity_XRestoredOnDestruction) {
    Eigen::MatrixXd X_copy = X;

    ScreenTRexControlParameter screen_params;
    screen_params.trex_ctrl.K = 3;
    screen_params.trex_ctrl.max_dummy_multiplier = 2;
    screen_params.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;

    {
        ScreenTRexSelector selector(X_map, y_map, screen_params, 42, false);
        // X is now normalized. Destructor fires here.
    }

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored by ScreenTRexSelector destructor.";
}

// ========================================================================================
// Scaling-mode parity (DA-EQUI / DA-BLOCK-EQUI)
// ========================================================================================

/**
 * @brief Runs ScreenTRexSelector under a given scaling mode and returns the
 *        selected-variable indicator vector.
 *
 * estimateEquiCorrelation() / estimateBlockEquiCorrelation() accumulate
 * unit-norm columns (scale-mode agnostic, mirroring
 * TRexDASelector::estimateEquiCorrelation), so L2 and ZSCORE scaling must
 * yield bit-identical selections for the DA_EQUI and DA_BLOCK_EQUI variants.
 */
Eigen::VectorXi run_screen_trex_select(Eigen::Index n,
                                       Eigen::Index p,
                                       ScreenTRexMethod method,
                                       dn::ScalingMode scaling_mode)
{
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};

    SyntheticData data(n, p, support, coefs, 1.0, 42);

    auto X = data.getX();
    auto y = data.getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3;
    trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
    trex_ctrl.scaling_mode = scaling_mode;

    ScreenTRexControlParameter screen_ctrl;
    screen_ctrl.trex_method = method;
    screen_ctrl.trex_ctrl = trex_ctrl;

    ScreenTRexSelector selector(X_map, y_map, screen_ctrl, 42, false);
    auto result = selector.select();

    return result.selected_var;
}

/** @brief Screen-TRex DA-EQUI selections must be identical under L2 and ZSCORE scaling. */
TEST(TRexScreeningScalingTest, Method_DA_EQUI_ScalingModeParity) {
    Eigen::VectorXi sel_l2 =
        run_screen_trex_select(300, 100, ScreenTRexMethod::TREX_DA_EQUI, dn::ScalingMode::L2);
    Eigen::VectorXi sel_zscore =
        run_screen_trex_select(300, 100, ScreenTRexMethod::TREX_DA_EQUI, dn::ScalingMode::ZSCORE);

    ASSERT_EQ(sel_l2.size(), sel_zscore.size());
    EXPECT_TRUE((sel_l2.array() == sel_zscore.array()).all())
        << "L2 and ZSCORE scaling produced different Screen-TRex DA-EQUI selections.";
}

/** @brief Screen-TRex DA-BLOCK-EQUI selections must be identical under L2 and ZSCORE scaling. */
TEST(TRexScreeningScalingTest, Method_DA_BLOCK_EQUI_ScalingModeParity) {
    Eigen::VectorXi sel_l2 =
        run_screen_trex_select(300, 100, ScreenTRexMethod::TREX_DA_BLOCK_EQUI, dn::ScalingMode::L2);
    Eigen::VectorXi sel_zscore =
        run_screen_trex_select(300, 100, ScreenTRexMethod::TREX_DA_BLOCK_EQUI, dn::ScalingMode::ZSCORE);

    ASSERT_EQ(sel_l2.size(), sel_zscore.size());
    EXPECT_TRUE((sel_l2.array() == sel_zscore.array()).all())
        << "L2 and ZSCORE scaling produced different Screen-TRex DA-BLOCK-EQUI selections.";
}

/// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_screening */
