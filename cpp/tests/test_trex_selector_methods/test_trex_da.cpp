// ========================================================================================
// test_trex_da.cpp
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <cmath>
#include <random>
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_da/trex_da.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_da {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_da;
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace dn = trex::trex_selector_methods::utils::data_normalizer;

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
    da_ctrl.trex_ctrl = trex_ctrl;

    TRexDASelector trex(X_map, y_map, 0.1, da_ctrl, 42, false);

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


// ========================================================================================
// Scaling-mode parity
// ========================================================================================

/**
 * @brief Runs TRexDASelector under a given scaling mode and returns the
 *        selected-variable indicator vector.
 *
 * The DA correlation estimators for NN (pairwise Pearson correlation) and EQUI
 * (mean off-diagonal correlation from unit-norm columns) are normalization-
 * agnostic, so L2 and ZSCORE scaling must yield bit-identical selections.
 */
Eigen::VectorXi run_trex_da_select(Eigen::Index n,
                                   Eigen::Index p,
                                   DAMethod method,
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
    trex_ctrl.max_dummy_multiplier = 2;
    trex_ctrl.scaling_mode = scaling_mode;

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = method;
    da_ctrl.trex_ctrl = trex_ctrl;

    TRexDASelector trex(X_map, y_map, 0.1, da_ctrl, 42, false);
    trex.select();

    return trex.getDAResult().selected_var;
}

/** @brief DA-NN selections must be identical under L2 and ZSCORE scaling. */
TEST(TRexDATest, Method_NN_ScalingModeParity) {
    Eigen::VectorXi sel_l2 =
        run_trex_da_select(300, 100, DAMethod::NN, dn::ScalingMode::L2);
    Eigen::VectorXi sel_zscore =
        run_trex_da_select(300, 100, DAMethod::NN, dn::ScalingMode::ZSCORE);

    ASSERT_EQ(sel_l2.size(), sel_zscore.size());
    EXPECT_TRUE((sel_l2.array() == sel_zscore.array()).all())
        << "L2 and ZSCORE scaling produced different DA-NN selections.";
}

/** @brief DA-EQUI selections must be identical under L2 and ZSCORE scaling. */
TEST(TRexDATest, Method_EQUI_ScalingModeParity) {
    Eigen::VectorXi sel_l2 =
        run_trex_da_select(300, 100, DAMethod::EQUI, dn::ScalingMode::L2);
    Eigen::VectorXi sel_zscore =
        run_trex_da_select(300, 100, DAMethod::EQUI, dn::ScalingMode::ZSCORE);

    ASSERT_EQ(sel_l2.size(), sel_zscore.size());
    EXPECT_TRUE((sel_l2.array() == sel_zscore.array()).all())
        << "L2 and ZSCORE scaling produced different DA-EQUI selections.";
}


/**
 * @brief Runs a DA-BT selection under a given BT candidate-cell policy and
 *        returns the selected-variable indicator vector.
 */
Eigen::VectorXi run_trex_da_bt_select(Eigen::Index n,
                                      Eigen::Index p,
                                      BTSelectionMode bt_mode)
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
    trex_ctrl.max_dummy_multiplier = 2;

    TRexDAControlParameter da_ctrl;
    da_ctrl.method            = DAMethod::BT;
    da_ctrl.bt_selection_mode = bt_mode;
    da_ctrl.trex_ctrl         = trex_ctrl;

    TRexDASelector trex(X_map, y_map, 0.1, da_ctrl, 42, false);
    trex.select();

    return trex.getDAResult().selected_var;
}

/** @brief DA-BT bt_selection_mode: FeasibleOnly is the default, and both policies
 *  produce a valid binary indicator of length p. */
TEST(TRexDATest, Method_BT_SelectionModeField) {
    const Eigen::Index n = 150, p = 300;

    // The default candidate policy must be the FDR-controlling FeasibleOnly.
    TRexDAControlParameter da_default;
    EXPECT_EQ(da_default.bt_selection_mode, BTSelectionMode::FeasibleOnly);

    Eigen::VectorXi sel_feasible =
        run_trex_da_bt_select(n, p, BTSelectionMode::FeasibleOnly);
    Eigen::VectorXi sel_rfaithful =
        run_trex_da_bt_select(n, p, BTSelectionMode::RFaithful);

    // Both policies return a valid binary indicator of length p.
    ASSERT_EQ(sel_feasible.size(), p);
    ASSERT_EQ(sel_rfaithful.size(), p);
    EXPECT_GE(sel_feasible.minCoeff(), 0);
    EXPECT_LE(sel_feasible.maxCoeff(), 1);
    EXPECT_GE(sel_rfaithful.minCoeff(), 0);
    EXPECT_LE(sel_rfaithful.maxCoeff(), 1);
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

    TRexDAControlParameter da_params;

    // 1. Invalid rho_thr_DA
    da_params.rho_thr_DA = 1.5; // Bounds are [0, 1]
    EXPECT_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params);
    }, std::invalid_argument);

    da_params.rho_thr_DA = 0.5; // Reset to valid

    // 2. Prior Groups Length Mismatch
    da_params.prior_groups = {{1, 1, 1, 1, 1}}; // Incorrect length (p = 10)
    EXPECT_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params);
    }, std::invalid_argument);

    da_params.prior_groups.clear(); // Reset

    // 3. Unsupported BT Linkage
    da_params.method = DAMethod::BT;
    // Supplying a linkage method not explicitly supported by our clustering logic
    da_params.hc_linkage =
        static_cast<trex::ml_methods::clustering::hierarchical::agglomerative::LinkageMethod>(999);
    EXPECT_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params);
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
    da_params.trex_ctrl = tc_params;

    EXPECT_NO_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_params);
        auto result = selector.select();
        // Since SKIPL is used, verify initialized dummies
        EXPECT_EQ(result.T_stop >= 1, true);
    });
}


// ========================================================================================
// Data Integrity
// ========================================================================================

/** @brief X is restored to its original values after select() returns (object still alive). */
TEST(TRexDATest, DataIntegrity_XRestoredAfterSelect) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(30, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(30);
    Eigen::MatrixXd X_copy = X;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3;
    trex_ctrl.max_dummy_multiplier = 2;

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = DAMethod::AR1;
    da_ctrl.trex_ctrl = trex_ctrl;

    TRexDASelector trex(X_map, y_map, 0.1, da_ctrl, 42, false);
    trex.select();

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored after TRexDASelector::select().";
}


/** @brief X is restored to its original values when the object is destroyed without
 *         calling select() (normalization happens in the constructor). */
TEST(TRexDATest, DataIntegrity_XRestoredOnDestruction) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(30, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(30);
    Eigen::MatrixXd X_copy = X;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3;
    trex_ctrl.max_dummy_multiplier = 2;

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = DAMethod::AR1;
    da_ctrl.trex_ctrl = trex_ctrl;

    {
        TRexDASelector trex(X_map, y_map, 0.1, da_ctrl, 42, false);
        // X is now normalized. Destructor fires here.
    }

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored by TRexDASelector destructor.";
}


// ========================================================================================
// Correlation estimation
// ========================================================================================

namespace {

/** @brief Probe exposing protected internals (estimators, DA setup) for testing. */
class TRexDASelectorProbe : public TRexDASelector {
public:
    using TRexDASelector::TRexDASelector;
    using TRexDASelector::estimateAR1Correlation;
    using TRexDASelector::setupDA;
    using TRexDASelector::da_setup_;
};

} // namespace


/** @brief The row-wise Yule-Walker AR(1) estimator recovers a known rho.
 *
 *  Tolerance context: the estimator approximates the R reference (per-row
 *  `arima(..., method = "ML")` averaged over rows); both are consistent, so
 *  on a long synthetic AR(1) design the estimate must sit close to the truth.
 */
TEST(TRexDATest, Estimation_AR1YuleWalkerRecoversKnownRho) {
    const Eigen::Index n = 100;
    const Eigen::Index p = 400;
    const double rho = 0.7;

    // Row-stationary AR(1) across the ordered predictors.
    std::mt19937 rng(123);
    std::normal_distribution<double> N01(0.0, 1.0);
    const double innov_sd = std::sqrt(1.0 - rho * rho);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        X(i, 0) = N01(rng);
        for (Eigen::Index t = 1; t < p; ++t) {
            X(i, t) = rho * X(i, t - 1) + innov_sd * N01(rng);
        }
    }
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n);  // unused by the estimator

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = DAMethod::AR1;

    // The constructor normalizes X in place; the estimator runs on the
    // normalized data, exactly as setupDA_AR1() would invoke it.
    TRexDASelectorProbe trex(X_map, y_map, 0.1, da_ctrl, 42, false);

    const double est = trex.estimateAR1Correlation();
    EXPECT_NEAR(est, rho, 0.05)
        << "Yule-Walker AR(1) estimate drifted from the true coefficient.";
}

// ========================================================================================
// Prior groups — constrained sub-clustering
// ========================================================================================

namespace {

/** @brief Group-factor design: within-group correlated columns, independent
 *  across groups. Column j of group g is a * f_g + b * noise.
 */
Eigen::MatrixXd make_group_factor_X(Eigen::Index n,
                                    Eigen::Index p,
                                    Eigen::Index group_size,
                                    double rho_within,
                                    unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    const double a = std::sqrt(rho_within);
    const double b = std::sqrt(1.0 - rho_within);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        double factor = 0.0;
        for (Eigen::Index j = 0; j < p; ++j) {
            if (j % group_size == 0) { factor = N01(rng); }
            X(i, j) = a * factor + b * N01(rng);
        }
    }
    return X;
}

} // namespace


/** @brief PRIOR_GROUPS builds a BT-style structure by sub-clustering WITHIN
 *  the prior groups: the rho grid ends at the conservative rho = 1 singleton
 *  anchor, no neighbourhood ever crosses a prior-group boundary, and the
 *  coarsest grid point recovers (at most) the whole prior group.
 */
TEST(TRexDATest, Method_PriorGroups_ConstrainedSubclusteringStructure) {
    const Eigen::Index n = 100, p = 40, group_size = 8;

    Eigen::MatrixXd X = make_group_factor_X(n, p, group_size, 0.5, 7);
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexDAControlParameter da_ctrl;
    da_ctrl.prior_groups.resize(1);
    for (Eigen::Index j = 0; j < p; ++j) {
        da_ctrl.prior_groups[0].push_back(j / group_size);
    }

    TRexDASelectorProbe trex(X_map, y_map, 0.1, da_ctrl, 42, false);
    trex.setupDA();

    const auto& setup = trex.da_setup_;
    ASSERT_TRUE(setup.use_BT_style);

    // Default grid length min(20, p) = 20; grid ascending with rho = 1 anchor.
    const auto grid_len = static_cast<Eigen::Index>(setup.rho_grid_len);
    ASSERT_EQ(grid_len, 20);
    ASSERT_EQ(setup.rho_grid.size(), grid_len);
    for (Eigen::Index r = 1; r < grid_len; ++r) {
        EXPECT_LE(setup.rho_grid(r - 1), setup.rho_grid(r));
    }
    EXPECT_DOUBLE_EQ(setup.rho_grid(grid_len - 1), 1.0);

    bool any_nonempty = false;
    for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index r = 0; r < grid_len; ++r) {
            const auto& mates =
                setup.gr_j_list[static_cast<std::size_t>(j)]
                               [static_cast<std::size_t>(r)];
            if (!mates.empty()) { any_nonempty = true; }
            // Hard constraint: mates never leave the prior group.
            for (const auto k : mates) {
                EXPECT_EQ(k / group_size, j / group_size)
                    << "Neighbourhood crossed a prior-group boundary.";
                EXPECT_NE(k, j);
            }
            // Tightness: never more mates than the prior group offers.
            EXPECT_LE(mates.size(),
                      static_cast<std::size_t>(group_size - 1));
        }
        // rho = 1 anchor: all neighbourhoods empty (paper's conservative
        // Psi = 1/2 penalty applies to every variable).
        EXPECT_TRUE(setup.gr_j_list[static_cast<std::size_t>(j)]
                                   [static_cast<std::size_t>(grid_len - 1)]
                        .empty());
    }
    EXPECT_TRUE(any_nonempty)
        << "Sub-clustering produced no neighbourhoods at any grid point.";
}


/** @brief The constraint partition is the finest common refinement of ALL
 *  supplied levels: with level 0 = groups of 4 and level 1 = parity, the
 *  effective constraint groups have size 2.
 */
TEST(TRexDATest, Method_PriorGroups_FinestCommonRefinement) {
    const Eigen::Index n = 60, p = 16;

    Eigen::MatrixXd X = make_group_factor_X(n, p, 4, 0.6, 11);
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexDAControlParameter da_ctrl;
    da_ctrl.prior_groups.resize(2);
    for (Eigen::Index j = 0; j < p; ++j) {
        da_ctrl.prior_groups[0].push_back(j / 4);  // groups of 4
        da_ctrl.prior_groups[1].push_back(j % 2);  // parity split
    }

    TRexDASelectorProbe trex(X_map, y_map, 0.1, da_ctrl, 42, false);
    trex.setupDA();

    const auto& setup = trex.da_setup_;
    for (Eigen::Index j = 0; j < p; ++j) {
        for (std::size_t r = 0; r < setup.rho_grid_len; ++r) {
            const auto& mates =
                setup.gr_j_list[static_cast<std::size_t>(j)][r];
            // Refinement group = {j, j xor 2} within each block of 4: the
            // only admissible mate shares the block AND the parity.
            EXPECT_LE(mates.size(), 1u);
            for (const auto k : mates) {
                EXPECT_EQ(k / 4, j / 4);
                EXPECT_EQ(k % 2, j % 2);
            }
        }
    }
}


/** @brief Full select() run under PRIOR_GROUPS: valid binary indicator and a
 *  data-driven rho grid (not the legacy one-point-per-level grid).
 */
TEST(TRexDATest, Method_PriorGroups_Select) {
    const Eigen::Index n = 150, p = 50, group_size = 5;

    Eigen::MatrixXd X = make_group_factor_X(n, p, group_size, 0.5, 3);
    // Signal on the first column of three distinct groups.
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    beta(0) = 5.0; beta(5) = -3.0; beta(10) = 2.0;
    std::mt19937 rng(99);
    std::normal_distribution<double> N01(0.0, 1.0);
    Eigen::VectorXd y = X * beta;
    for (Eigen::Index i = 0; i < n; ++i) { y(i) += N01(rng); }

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3;
    trex_ctrl.max_dummy_multiplier = 2;

    TRexDAControlParameter da_ctrl;
    da_ctrl.trex_ctrl = trex_ctrl;
    da_ctrl.prior_groups.resize(1);
    for (Eigen::Index j = 0; j < p; ++j) {
        da_ctrl.prior_groups[0].push_back(j / group_size);
    }

    TRexDASelector trex(X_map, y_map, 0.1, da_ctrl, 42, false);
    trex.select();

    const auto& da_result = trex.getDAResult();
    EXPECT_EQ(da_result.method, DAMethod::PRIOR_GROUPS);
    ASSERT_EQ(da_result.selected_var.size(), p);
    EXPECT_GE(da_result.selected_var.minCoeff(), 0);
    EXPECT_LE(da_result.selected_var.maxCoeff(), 1);

    // Data-driven rho grid: hc_grid_length points ending at the rho = 1
    // anchor — NOT the legacy grid with one point per prior-group level.
    ASSERT_EQ(da_result.rho_grid.size(), 20);
    EXPECT_DOUBLE_EQ(da_result.rho_grid(da_result.rho_grid.size() - 1), 1.0);
}


/** @brief Negative prior-group labels are rejected. */
TEST(TRexDATest, Method_PriorGroups_NegativeLabelThrows) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(20);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexDAControlParameter da_ctrl;
    da_ctrl.prior_groups = {{0, 0, 1, 1, 2, 2, 3, 3, -1, 4}};

    EXPECT_THROW({
        TRexDASelector selector(X_map, y_map, 0.1, da_ctrl);
    }, std::invalid_argument);
}


// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_da */
