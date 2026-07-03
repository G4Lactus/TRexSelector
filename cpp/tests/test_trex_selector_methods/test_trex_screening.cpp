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
    TRexControlParameter tc_params;
    ScreenTRexControlParameter screen_params;

    // Supported
    tc_params.lloop_strategy = LLoopStrategy::STANDARD;
    EXPECT_NO_THROW(ScreenTRexSelector(X_map, y_map,
                                       screen_params, tc_params));

    tc_params.lloop_strategy = LLoopStrategy::PERMUTATION;
    EXPECT_NO_THROW(ScreenTRexSelector(X_map, y_map,
                                       screen_params, tc_params));

    // Unsupported: SKIPL, HCONCAT, DIRECT, etc.
    tc_params.lloop_strategy = LLoopStrategy::SKIPL;
    EXPECT_THROW(ScreenTRexSelector(X_map, y_map,
                                    screen_params, tc_params),
                                    std::invalid_argument);

    tc_params.lloop_strategy = LLoopStrategy::HCONCAT;
    EXPECT_THROW(ScreenTRexSelector(X_map, y_map, screen_params,
         tc_params), std::invalid_argument);
}


/** @brief Test execution of Ordinary Screening */
TEST_F(TRexScreeningTest, Execution_OrdinaryScreenTRexDoesNotThrow) {
    TRexControlParameter tc_params;
    tc_params.K = 5; // Reduced K for quick unit testing
    tc_params.lloop_strategy = LLoopStrategy::STANDARD;

    ScreenTRexControlParameter screen_params;
    screen_params.trex_method = ScreenTRexMethod::TREX;
    screen_params.use_bootstrap_CI = false;

    EXPECT_NO_THROW({
        ScreenTRexSelector selector(X_map, y_map,
                                    screen_params, tc_params);
        auto result = selector.select();

        // Ensure static execution values
        EXPECT_EQ(result.T_stop, 1);

        // num_dummies is locked to p
        EXPECT_EQ(result.num_dummies, 20);
    });
}


/** @brief Test of Boostrap CI Screening */
TEST_F(TRexScreeningTest, Execution_BootstrapScreenTRexDoesNotThrow) {
    TRexControlParameter tc_params;
    tc_params.K = 5;
    tc_params.lloop_strategy = LLoopStrategy::STANDARD;

    ScreenTRexControlParameter screen_params;
    screen_params.trex_method = ScreenTRexMethod::TREX;
    screen_params.use_bootstrap_CI = true;
    screen_params.R_boot = 50; // Small bootstrap resamples for fast testing

    EXPECT_NO_THROW({
        ScreenTRexSelector selector(X_map, y_map,
                                    screen_params, tc_params);

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

    TRexControlParameter tc_params;
    tc_params.K = 3;
    tc_params.max_dummy_multiplier = 2;
    tc_params.lloop_strategy = LLoopStrategy::STANDARD;

    ScreenTRexControlParameter screen_params;

    ScreenTRexSelector selector(X_map, y_map, screen_params, tc_params, 42, false);
    selector.select();

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored after ScreenTRexSelector::select().";
}


/** @brief X is restored to its original values when the object is destroyed without
 *         calling select() (normalization happens in the constructor). */
TEST_F(TRexScreeningTest, DataIntegrity_XRestoredOnDestruction) {
    Eigen::MatrixXd X_copy = X;

    TRexControlParameter tc_params;
    tc_params.K = 3;
    tc_params.max_dummy_multiplier = 2;
    tc_params.lloop_strategy = LLoopStrategy::STANDARD;

    ScreenTRexControlParameter screen_params;

    {
        ScreenTRexSelector selector(X_map, y_map, screen_params, tc_params, 42, false);
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

    ScreenTRexSelector selector(X_map, y_map, screen_ctrl, trex_ctrl, 42, false);
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
