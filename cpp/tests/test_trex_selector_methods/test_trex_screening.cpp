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


// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_screening {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_core;

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

/// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_screening */
