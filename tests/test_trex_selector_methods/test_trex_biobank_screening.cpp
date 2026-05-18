// =============================================================================
// test_trex_biobank_screening.cpp
// =============================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <stdexcept>

// project trex includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_screening/trex_screening.hpp>
#include <trex_selector_methods/trex_screening/trex_biobank_screening.hpp>

// ===================================================================================

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_biobank_screening;


// ===================================================================================


class TRexBiobankScreeningTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::MatrixXd Y_matrix;

    Eigen::Map<Eigen::MatrixXd> X_map;
    Eigen::Map<Eigen::VectorXd> y_map;
    Eigen::Map<Eigen::MatrixXd> Y_matrix_map;

    TRexBiobankScreeningTest()
        : X(Eigen::MatrixXd::Random(50, 20)),
          y(Eigen::VectorXd::Random(50)),
          Y_matrix(Eigen::MatrixXd::Random(50, 3)),
          X_map(X.data(), X.rows(), X.cols()),
          y_map(y.data(), y.size()),
          Y_matrix_map(Y_matrix.data(), Y_matrix.rows(), Y_matrix.cols()) {}
};


/** @brief Ensure invalid phenotypic calls map to runtime_errors */
TEST_F(TRexBiobankScreeningTest, Validation_ThrowsOnMismatchedPhenotypeCalls) {
    BiobankScreenTRexControl control;

    // Single Phenotype constructor
    BiobankScreenTRex single_pheno_selector(X_map, y_map, control);
    EXPECT_THROW(single_pheno_selector.screenPhenotypes(), std::runtime_error);

    // Multiple Phenotype constructor
    BiobankScreenTRex multi_pheno_selector(X_map, Y_matrix_map, control);
    EXPECT_THROW(multi_pheno_selector.screenPhenotype(), std::runtime_error);
}


/** @brief Standard single execution works cleanly */
TEST_F(TRexBiobankScreeningTest, Execution_SinglePhenotypeDoesNotThrow) {
    BiobankScreenTRexControl control;
    control.screen_ctrl.trex_method = ScreenTRexMethod::TREX;

    // As identified, we must override internal defaults to standard!
    control.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
    control.trex_ctrl.K = 5;

    BiobankScreenTRex selector(X_map, y_map, control);
    EXPECT_NO_THROW({
        auto result = selector.screenPhenotype();
    });
}


/** @brief Execute fallback testing under impossible bounds */
TEST_F(TRexBiobankScreeningTest, Execution_FallbackLogicActivation) {
    BiobankScreenTRexControl control;
    // Force the decision logic to abandon Screen-TRex by setting impossible bounds:
    control.lower_bound_FDR = 0.99;
    control.upper_bound_FDR = 0.00;

    // Prevent the Screen-TRex components from crashing under SKIPL
    control.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
    control.trex_ctrl.K = 2; // Keep loop overhead low

    BiobankScreenTRex selector(X_map, y_map, control);
    auto result = selector.screenPhenotype();

    // Verify Algorithm 1 dumped us into the full T-Rex calculation tree
    EXPECT_EQ(result.method_used, "T-Rex (fallback)");
    EXPECT_TRUE(result.used_fallback_trex);
}
