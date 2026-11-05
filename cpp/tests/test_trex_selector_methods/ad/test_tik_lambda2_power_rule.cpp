// ========================================================================================
// test_tik_lambda2_power_rule.cpp
// ========================================================================================
/**
 * @file test_tik_lambda2_power_rule.cpp
 *
 * @brief Unit tests for the lambda_2 power rule (tik_lambda2_power_rule.hpp)
 * and the post-selection fit criteria (post_selection_criteria.hpp): the
 * pick logic on synthetic profiles, the driver's grid handling, the rule on
 * the Tikhonov selector with a fitting prior (K_center on whole active
 * blocks: a positive pick with more power than the lasso) and with a
 * misfitting one (the group-mean prior on the same truth: the rule switches
 * it off), reproducibility under a shared seed, the GVS overload, and the
 * closed forms of BIC / EBIC / adjusted R^2.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// std includes
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

// project trex includes
#include <ml_methods/model_selection/post_selection_criteria.hpp>
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>
#include <trex_selector_methods/ad/trex_tikhonov/tik_lambda2_power_rule.hpp>
#include <trex_selector_methods/ad/trex_tikhonov/trex_tikhonov.hpp>

// ========================================================================================

namespace trex::test::trex_selector_methods::ad::trex_tikhonov {

using namespace trex::trex_selector_methods::ad::trex_tikhonov;
using namespace trex::trex_selector_methods::ad::trex_core;
namespace gvs = trex::trex_selector_methods::ad::trex_gvs;
namespace ms  = trex::ml_methods::model_selection;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;

// ========================================================================================

namespace {

constexpr double kTol = 1e-9;

/** @brief Equicorrelated blocks of `block` columns (within-block correlation
 *  `rho`), the first `active_blocks` blocks WHOLE-active with coefficient
 *  `beta`, Gaussian noise with variance sigma^2. */
struct BlockData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    std::vector<Eigen::Index> groups;
    std::vector<std::size_t>  support;

    BlockData(Eigen::Index n, Eigen::Index p, Eigen::Index block, double rho,
              Eigen::Index active_blocks, double beta, double sigma, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<double> N01(0.0, 1.0);
        X.resize(n, p);
        groups.resize(static_cast<std::size_t>(p));
        for (Eigen::Index g = 0; g * block < p; ++g) {
            Eigen::VectorXd z(n);
            for (Eigen::Index i = 0; i < n; ++i) { z(i) = N01(rng); }
            for (Eigen::Index j = g * block; j < std::min((g + 1) * block, p); ++j) {
                groups[static_cast<std::size_t>(j)] = g;
                for (Eigen::Index i = 0; i < n; ++i) {
                    X(i, j) = std::sqrt(rho) * z(i) + std::sqrt(1.0 - rho) * N01(rng);
                }
            }
        }
        y = Eigen::VectorXd::Zero(n);
        for (Eigen::Index j = 0; j < active_blocks * block; ++j) {
            support.push_back(static_cast<std::size_t>(j));
            y += beta * X.col(j);
        }
        for (Eigen::Index i = 0; i < n; ++i) { y(i) += sigma * N01(rng); }
    }
};

/** @brief Group-mean (informed elastic net) matrix W = sum_m 1_m 1_m^T / p_m. */
Eigen::SparseMatrix<double> groupMeanK(const std::vector<Eigen::Index>& groups)
{
    const auto p = static_cast<Eigen::Index>(groups.size());
    std::map<Eigen::Index, std::vector<Eigen::Index>> members;
    for (Eigen::Index j = 0; j < p; ++j) { members[groups[static_cast<std::size_t>(j)]].push_back(j); }
    std::vector<Eigen::Triplet<double>> trip;
    for (const auto& [g, cols] : members) {
        const double w = 1.0 / static_cast<double>(cols.size());
        for (const Eigen::Index a : cols) {
            for (const Eigen::Index b : cols) { trip.emplace_back(a, b, w); }
        }
    }
    Eigen::SparseMatrix<double> W(p, p);
    W.setFromTriplets(trip.begin(), trip.end());
    return W;
}

/** @brief The family's base control at a test-sized budget. */
TRexControlParameter baseControl()
{
    TRexControlParameter ctrl;
    ctrl.K                        = 10;
    ctrl.max_dummy_multiplier     = 2;
    ctrl.use_max_T_stop           = true;
    ctrl.lloop_strategy           = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop    = true;
    ctrl.tloop_max_stagnant_steps = 5;
    ctrl.use_memory_mapping       = false;
    return ctrl;
}

/** @brief K_center on the prior groups through the GROUP_CENTERING route. */
TRexTikhonovControlParameter centeringControl(const std::vector<Eigen::Index>& groups)
{
    TRexTikhonovControlParameter ctrl;
    ctrl.penalty_structure = TikPenaltyStructure::GROUP_CENTERING;
    ctrl.prior_groups      = groups;
    ctrl.cluster_dummies   = true;
    ctrl.trex_ctrl         = baseControl();
    ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TIKNET;
    return ctrl;
}

/** @brief The group-mean prior through the SPARSE_GENERAL route. */
TRexTikhonovControlParameter groupMeanControl(const std::vector<Eigen::Index>& groups)
{
    TRexTikhonovControlParameter ctrl;
    ctrl.penalty_structure = TikPenaltyStructure::SPARSE_GENERAL;
    ctrl.tikhonov_K        = groupMeanK(groups);
    ctrl.prior_groups      = groups;
    ctrl.cluster_dummies   = true;
    ctrl.trex_ctrl         = baseControl();
    ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TIKNET;
    return ctrl;
}

/** @brief The test design: n = 100, p = 200 in 40 blocks of 5 at rho 0.7,
 *  two whole-active blocks at beta = 1, SNR 1 (signal variance
 *  2 (5 + 20 rho) = 38). */
BlockData testDesign(unsigned seed = 11)
{
    return BlockData(100, 200, 5, 0.7, 2, 1.0, std::sqrt(38.0), seed);
}

std::size_t truePositives(const std::vector<std::size_t>& sel, const std::vector<std::size_t>& support)
{
    std::size_t tp = 0;
    for (const std::size_t j : sel) {
        if (std::find(support.begin(), support.end(), j) != support.end()) { ++tp; }
    }
    return tp;
}

} // namespace


// ========================================================================================
// The pick on synthetic profiles
// ========================================================================================

TEST(TikLambda2PowerRule, PickIndex_ToleranceZeroTiesToSmallest) {
    const std::vector<std::size_t> counts = {3, 5, 5, 4};
    const auto pick = pick_lambda2_index(counts, 0.0);
    EXPECT_EQ(pick.index, 1U);
    EXPECT_EQ(pick.max_selected, 5U);
    EXPECT_EQ(pick.n_ties, 2U);
}

TEST(TikLambda2PowerRule, PickIndex_KneeTakesTheFirstOnThePlateau) {
    const std::vector<std::size_t> a = {3, 5, 5, 4};
    const auto pa = pick_lambda2_index(a, 1.0);
    EXPECT_EQ(pa.index, 1U);
    EXPECT_EQ(pa.n_ties, 3U);   // 5, 5, 4 are within one selection of the maximum

    const std::vector<std::size_t> b = {4, 5, 5, 4};
    EXPECT_EQ(pick_lambda2_index(b, 1.0).index, 0U);
    EXPECT_EQ(pick_lambda2_index(b, 0.0).index, 1U);
    EXPECT_EQ(pick_lambda2_index(b).index, 0U);   // the default tolerance is one selection
}

TEST(TikLambda2PowerRule, PickIndex_Validation) {
    EXPECT_THROW(pick_lambda2_index({}, 0.0), std::invalid_argument);
    EXPECT_THROW(pick_lambda2_index({1, 2}, -1.0), std::invalid_argument);
}


// ========================================================================================
// The driver
// ========================================================================================

/** @brief An unsorted grid with a duplicate is profiled ascending and
 *  de-duplicated; the pick, the ties and the boundary flags follow the
 *  probe's counts. */
TEST(TikLambda2PowerRule, Driver_SortsDeduplicatesAndFlagsBoundaries) {
    Lambda2PowerRuleConfig cfg;
    cfg.grid = {10.0, 0.0, 1.0, 10.0, 100.0};
    cfg.knee_tolerance = 0.0;   // the largest selection, ties toward the smallest lambda_2
    const std::map<double, std::size_t> counts = {{0.0, 2}, {1.0, 6}, {10.0, 7}, {100.0, 7}};

    std::vector<double> visited;
    const auto res = run_lambda2_power_rule(cfg, [&](double l2) {
        visited.push_back(l2);
        Lambda2Probe pr;
        pr.selected.assign(counts.at(l2), 0);
        for (std::size_t i = 0; i < pr.selected.size(); ++i) { pr.selected[i] = pr.selected.size() - i; }
        pr.T_stop = 3;
        return pr;
    });

    ASSERT_EQ(visited.size(), 4U);
    EXPECT_DOUBLE_EQ(visited[0], 0.0);
    EXPECT_DOUBLE_EQ(visited[3], 100.0);
    ASSERT_EQ(res.probes.size(), 4U);
    EXPECT_DOUBLE_EQ(res.lambda2, 10.0);
    EXPECT_EQ(res.index, 2U);
    EXPECT_EQ(res.pick.max_selected, 7U);
    EXPECT_EQ(res.pick.n_ties, 2U);
    EXPECT_FALSE(res.at_lower_end);
    EXPECT_FALSE(res.at_upper_end);
    EXPECT_EQ(res.selection().size(), 7U);
    EXPECT_TRUE(std::is_sorted(res.selection().begin(), res.selection().end()));
    EXPECT_GE(res.seconds, 0.0);
    for (const auto& pr : res.probes) { EXPECT_EQ(pr.T_stop, 3U); }

    cfg.knee_tolerance = 1.0;   // the knee: within one selection of the maximum
    const auto knee = run_lambda2_power_rule(cfg, [&](double l2) {
        Lambda2Probe pr;
        pr.selected.assign(counts.at(l2), 0);
        return pr;
    });
    EXPECT_DOUBLE_EQ(knee.lambda2, 1.0);
}

TEST(TikLambda2PowerRule, Driver_Validation) {
    Lambda2PowerRuleConfig cfg;
    const auto probe = [](double) { return Lambda2Probe{}; };
    EXPECT_THROW(run_lambda2_power_rule(cfg, probe), std::invalid_argument);
    cfg.grid = {-1.0};
    EXPECT_THROW(run_lambda2_power_rule(cfg, probe), std::invalid_argument);
    cfg.grid = {std::numeric_limits<double>::infinity()};
    EXPECT_THROW(run_lambda2_power_rule(cfg, probe), std::invalid_argument);
}


// ========================================================================================
// On the selectors
// ========================================================================================

/** @brief K_center on whole active blocks: the rule picks a positive
 *  lambda_2 whose selection is larger than the lasso's (lambda_2 = 0) and
 *  carries more true positives. */
TEST(TikLambda2PowerRule, Tikhonov_CenteringPrior_PicksPositiveLambda2WithMorePower) {
    const BlockData d = testDesign();
    Lambda2PowerRuleConfig cfg;
    cfg.grid = {0.0, 1.0, 10.0, 100.0};

    const auto res = tikhonov_lambda2_power_rule(d.X, d.y, 0.1, centeringControl(d.groups), 2026, cfg);

    ASSERT_EQ(res.probes.size(), 4U);
    EXPECT_GT(res.lambda2, 0.0);
    EXPECT_GT(res.pick.max_selected, res.probes[0].selected.size());
    EXPECT_GT(truePositives(res.selection(), d.support), truePositives(res.probes[0].selected, d.support));
    for (const auto& pr : res.probes) { EXPECT_GT(pr.seconds, 0.0); }
}

/** @brief The group-mean prior on the same truth loses power as lambda_2
 *  grows (one representative per block); the rule switches it off. */
TEST(TikLambda2PowerRule, Tikhonov_GroupMeanPrior_SwitchedOff) {
    const BlockData d = testDesign();
    Lambda2PowerRuleConfig cfg;
    cfg.grid = {0.0, 10.0, 100.0};

    const auto res = tikhonov_lambda2_power_rule(d.X, d.y, 0.1, groupMeanControl(d.groups), 2026, cfg);

    ASSERT_EQ(res.probes.size(), 3U);
    EXPECT_DOUBLE_EQ(res.lambda2, 0.0);
    EXPECT_TRUE(res.at_lower_end);
    EXPECT_LT(res.probes[2].selected.size(), res.probes[0].selected.size());
}

/** @brief The same seed reproduces every probe and the pick. */
TEST(TikLambda2PowerRule, Tikhonov_SharedSeedReproduces) {
    const BlockData d = testDesign();
    Lambda2PowerRuleConfig cfg;
    cfg.grid = {0.0, 10.0};

    const auto a = tikhonov_lambda2_power_rule(d.X, d.y, 0.1, centeringControl(d.groups), 7, cfg);
    const auto b = tikhonov_lambda2_power_rule(d.X, d.y, 0.1, centeringControl(d.groups), 7, cfg);

    EXPECT_DOUBLE_EQ(a.lambda2, b.lambda2);
    ASSERT_EQ(a.probes.size(), b.probes.size());
    for (std::size_t k = 0; k < a.probes.size(); ++k) {
        EXPECT_EQ(a.probes[k].selected, b.probes[k].selected);
        EXPECT_EQ(a.probes[k].T_stop, b.probes[k].T_stop);
    }
}

/** @brief The GVS overload (elastic-net ridge weight) runs the grid and
 *  returns a consistent pick. */
TEST(TikLambda2PowerRule, Gvs_ElasticNetRuleRuns) {
    const BlockData d = testDesign();
    gvs::TRexGVSControlParameter ctrl;
    ctrl.gvs_type     = gvs::GVSType::EN;
    ctrl.en_solver    = gvs::ENSolverType::TENET;
    ctrl.prior_groups = d.groups;
    ctrl.trex_ctrl    = baseControl();
    ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TENET;

    Lambda2PowerRuleConfig cfg;
    cfg.grid = {0.0, 1.0, 10.0};
    const auto res = gvs_lambda2_power_rule(d.X, d.y, 0.1, ctrl, 2026, cfg);

    ASSERT_EQ(res.probes.size(), 3U);
    const auto counts = res.counts();
    EXPECT_EQ(res.pick.max_selected, *std::max_element(counts.begin(), counts.end()));
    EXPECT_EQ(res.selection().size(), res.pick.max_selected);
    EXPECT_DOUBLE_EQ(res.probes[res.index].lambda2, res.lambda2);
}


// ========================================================================================
// Post-selection criteria
// ========================================================================================

TEST(PostSelectionCriteria, ClosedForms) {
    // RSS = n makes the deviance term vanish: BIC = k log n.
    EXPECT_NEAR(ms::bic(100.0, 100, 3), 3.0 * std::log(100.0), kTol);
    // EBIC adds 2 gamma log C(p, k): log C(5, 2) = log 10.
    EXPECT_NEAR(ms::ebic(100.0, 100, 2, 5, 1.0) - ms::bic(100.0, 100, 2), 2.0 * std::log(10.0), kTol);
    EXPECT_NEAR(ms::log_binomial(5, 2), std::log(10.0), kTol);
    EXPECT_NEAR(ms::log_binomial(10, 0), 0.0, kTol);
    // Saturated or degenerate fits are inadmissible.
    EXPECT_TRUE(std::isinf(ms::bic(1.0, 10, 10)));
    EXPECT_TRUE(std::isinf(ms::bic(0.0, 10, 2)));
    EXPECT_TRUE(std::isinf(ms::ebic(0.0, 10, 2, 20, 0.5)));
    EXPECT_THROW(ms::ebic(1.0, 10, 2, 20, -0.5), std::invalid_argument);
    // Adjusted R^2: a perfect fit gives 1; exhausted degrees of freedom -inf.
    EXPECT_NEAR(ms::adjusted_r2(0.0, 10.0, 20, 3), 1.0, kTol);
    EXPECT_NEAR(ms::adjusted_r2(5.0, 10.0, 11, 4), 1.0 - (5.0 / 6.0) / (10.0 / 10.0), kTol);
    EXPECT_TRUE(std::isinf(ms::adjusted_r2(1.0, 10.0, 5, 4)));
    // argmin / argmax take the first on ties.
    EXPECT_EQ(ms::argmin_first({2.0, 1.0, 1.0}), 1U);
    EXPECT_EQ(ms::argmax_first({2.0, 3.0, 3.0}), 1U);
    EXPECT_THROW(ms::argmin_first({}), std::invalid_argument);
}

/** @brief A noiseless linear response is refit exactly on its support; an
 *  empty support returns the total sum of squares; a wrong index throws. */
TEST(PostSelectionCriteria, OlsRefit_RecoversAndValidates) {
    std::mt19937 rng(3);
    std::normal_distribution<double> N01(0.0, 1.0);
    Eigen::MatrixXd X(50, 6);
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) { X(i, j) = N01(rng); }
    }
    const Eigen::VectorXd y = 2.0 * X.col(1) - 1.5 * X.col(4) + Eigen::VectorXd::Constant(50, 0.7);

    const auto fit = ms::ols_refit(X, y, {1, 4});
    EXPECT_EQ(fit.n, 50);
    EXPECT_EQ(fit.k, 2);
    EXPECT_NEAR(fit.coef(1), 2.0, 1e-8);
    EXPECT_NEAR(fit.coef(4), -1.5, 1e-8);
    EXPECT_NEAR(fit.coef(0), 0.0, kTol);
    EXPECT_NEAR(fit.rss, 0.0, 1e-8);
    EXPECT_GT(fit.tss, 0.0);

    const auto none = ms::ols_refit(X, y, {});
    EXPECT_NEAR(none.rss, none.tss, kTol);
    EXPECT_EQ(none.k, 0);

    EXPECT_THROW(ms::ols_refit(X, y, {6}), std::invalid_argument);
    EXPECT_THROW(ms::ols_refit(X, Eigen::VectorXd::Zero(49), {1}), std::invalid_argument);
}


// ========================================================================================
// The in-selector pilot rule (PILOT_KNEE)
// ========================================================================================

namespace {

/** @brief A control that requests the pilot rule on a grid. */
TRexTikhonovControlParameter pilotControl(TRexTikhonovControlParameter ctrl, TikLambda2Method method,
                                          std::vector<double> grid, std::size_t K_pilot, std::size_t T_pilot = 3)
{
    ctrl.lambda_2       = -1.0;
    ctrl.lambda2_method = method;
    ctrl.pilot_grid     = std::move(grid);
    ctrl.pilot_K        = K_pilot;
    ctrl.pilot_T        = T_pilot;
    return ctrl;
}

/** @brief One full selection on a private copy of the design. */
struct SelectorRun {
    std::vector<std::size_t> selected;
    std::size_t              T_stop  = 0;
    double                   lambda2 = 0.0;
    Lambda2PilotResult       pilot;
    Eigen::MatrixXd          phi;
};

SelectorRun runSelector(const BlockData& d, const TRexTikhonovControlParameter& ctrl, int seed)
{
    Eigen::MatrixXd X = d.X;
    Eigen::VectorXd y = d.y;
    Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
    TRexTikhonovSelector selector(Xm, ym, 0.1, ctrl, seed, false);
    selector.select();
    SelectorRun r;
    r.selected = selector.getSelectedIndices();
    r.T_stop   = selector.getTStop();
    r.lambda2  = selector.getLambda2Used();
    r.pilot    = selector.getLambda2PilotResult();
    r.phi      = selector.getPhiMat();
    return r;
}

const std::vector<double> kPilotGrid = {0.0, 1.0, 10.0, 100.0};

} // namespace

/** @brief K_center on whole active blocks: the pilot picks a positive
 *  lambda_2 whose count beats the lambda_2 = 0 probe's, and the selection
 *  at the pick carries more true positives than the lasso's. */
TEST(TikLambda2Pilot, CenteringPrior_PicksPositiveLambda2WithMorePower) {
    const BlockData d = testDesign();
    const SelectorRun pilot = runSelector(d, pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE,
                                                          kPilotGrid, 5), 2026);
    ASSERT_TRUE(pilot.pilot.ran);
    EXPECT_GT(pilot.lambda2, 0.0);
    EXPECT_GT(pilot.pilot.pick.max_selected, pilot.pilot.counts[0]);

    auto lasso = centeringControl(d.groups);
    lasso.lambda_2 = 0.0;
    const SelectorRun lasso_run = runSelector(d, lasso, 2026);
    EXPECT_GT(truePositives(pilot.selected, d.support), truePositives(lasso_run.selected, d.support));
}

/** @brief The group-mean prior loses power as lambda_2 grows; the pilot
 *  switches it off. */
TEST(TikLambda2Pilot, GroupMeanPrior_SwitchedOff) {
    const BlockData d = testDesign();
    const SelectorRun r = runSelector(d, pilotControl(groupMeanControl(d.groups), TikLambda2Method::PILOT_KNEE,
                                                      {0.0, 10.0, 100.0}, 5), 2026);
    EXPECT_DOUBLE_EQ(r.lambda2, 0.0);
    EXPECT_TRUE(r.pilot.at_lower_end);
    EXPECT_LT(r.pilot.counts[2], r.pilot.counts[0]);
}

/** @brief The main run after a pilot equals a fixed-lambda_2 run at the
 *  pick with the same seed: the pilot leaves the main dummy streams
 *  untouched, with independent and with shared pilot dummies. */
TEST(TikLambda2Pilot, MainRunEqualsFixedLambda2Run) {
    const BlockData d = testDesign();
    for (const bool shared : {false, true}) {
        auto ctrl = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
        ctrl.pilot_shared_dummies = shared;
        const SelectorRun pilot = runSelector(d, ctrl, 2026);

        auto fixed = centeringControl(d.groups);
        fixed.lambda_2 = pilot.lambda2;
        const SelectorRun ref = runSelector(d, fixed, 2026);
        EXPECT_EQ(pilot.selected, ref.selected) << "shared = " << shared;
        EXPECT_EQ(pilot.T_stop, ref.T_stop) << "shared = " << shared;
        EXPECT_TRUE(pilot.phi.isApprox(ref.phi)) << "shared = " << shared;
    }
}

/** @brief The same seed reproduces the pilot and the selection. */
TEST(TikLambda2Pilot, SeedReproduces) {
    const BlockData d = testDesign();
    const auto ctrl = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    const SelectorRun a = runSelector(d, ctrl, 7);
    const SelectorRun b = runSelector(d, ctrl, 7);
    EXPECT_EQ(a.pilot.counts, b.pilot.counts);
    EXPECT_EQ(a.pilot.curve_majority, b.pilot.curve_majority);
    EXPECT_DOUBLE_EQ(a.lambda2, b.lambda2);
    EXPECT_EQ(a.selected, b.selected);
}

/** @brief The pick is consistent with the pure pick function on the
 *  recorded counts, and the result echoes its budget. */
TEST(TikLambda2Pilot, PickConsistentWithPickIndex) {
    const BlockData d = testDesign();
    {
        const SelectorRun r = runSelector(d, pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE,
                                                          kPilotGrid, 5), 2026);
        EXPECT_EQ(r.pilot.pick.index, pick_lambda2_index(r.pilot.counts, 1.0).index);
        EXPECT_EQ(r.pilot.index, r.pilot.pick.index);
        EXPECT_DOUBLE_EQ(r.pilot.lambda2, r.pilot.grid[r.pilot.index]);
        EXPECT_DOUBLE_EQ(r.lambda2, r.pilot.lambda2);
        EXPECT_EQ(r.pilot.grid.size(), kPilotGrid.size());
        EXPECT_EQ(r.pilot.K_pilot, 5U);
        EXPECT_EQ(r.pilot.T_pilot, 3U);
        EXPECT_EQ(r.pilot.L_0, 200U);
        EXPECT_EQ(r.pilot.curve_majority.size(), kPilotGrid.size());
        EXPECT_EQ(r.pilot.curve_majority.front().size(), 3U);
        EXPECT_EQ(r.pilot.calibrated.size(), kPilotGrid.size());
        EXPECT_GT(r.pilot.seconds, 0.0);
        for (const double s : r.pilot.probe_seconds) { EXPECT_GT(s, 0.0); }
    }
}

/** @brief Under the global null every count is zero and the pick is the
 *  smallest lambda_2 at any tolerance. */
TEST(TikLambda2Pilot, NullCountsPickSmallestLambda2) {
    const BlockData d(100, 200, 5, 0.7, 0, 1.0, std::sqrt(38.0), 11);
    for (const double tol : {0.0, 1.0}) {
        auto ctrl = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
        ctrl.pilot_knee_tolerance = tol;
        const SelectorRun r = runSelector(d, ctrl, 2026);
        for (const std::size_t c : r.pilot.counts) { EXPECT_EQ(c, 0U); }
        EXPECT_DOUBLE_EQ(r.lambda2, 0.0);
    }
}

/** @brief The knee tolerance maps as documented. */
TEST(TikLambda2Pilot, KneeToleranceMaps) {
    const BlockData d = testDesign();
    auto loose = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    loose.pilot_knee_tolerance = 1e9;
    EXPECT_EQ(runSelector(d, loose, 2026).pilot.index, 0U);

    auto tight = loose;
    tight.pilot_knee_tolerance = 0.0;   // the largest count, ties toward the smallest lambda_2
    const SelectorRun t   = runSelector(d, tight, 2026);
    const auto first_max  = std::max_element(t.pilot.counts.begin(), t.pilot.counts.end());
    EXPECT_EQ(t.pilot.index, static_cast<std::size_t>(std::distance(t.pilot.counts.begin(), first_max)));
}

/** @brief A user value bypasses the pilot. */
TEST(TikLambda2Pilot, UserLambda2Bypasses) {
    const BlockData d = testDesign();
    auto ctrl = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    ctrl.lambda_2 = 3.0;
    const SelectorRun r = runSelector(d, ctrl, 2026);
    EXPECT_DOUBLE_EQ(r.lambda2, 3.0);
    EXPECT_FALSE(r.pilot.ran);
}

/** @brief Malformed pilot controls are refused at construction. */
TEST(TikLambda2Pilot, Validation) {
    const BlockData d = testDesign();
    const auto expect_throw = [&](const TRexTikhonovControlParameter& ctrl) {
        Eigen::MatrixXd X = d.X;
        Eigen::VectorXd y = d.y;
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        EXPECT_THROW(TRexTikhonovSelector(Xm, ym, 0.1, ctrl, 1, false), std::invalid_argument);
    };
    expect_throw(pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, {}, 5));
    expect_throw(pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, {-1.0, 1.0}, 5));
    expect_throw(pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 1));
    auto bad_L = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    bad_L.pilot_L_multiplier = 0;
    expect_throw(bad_L);
    bad_L.pilot_L_multiplier = bad_L.trex_ctrl.max_dummy_multiplier + 1;
    expect_throw(bad_L);
    auto bad_T = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5, 0);
    expect_throw(bad_T);
    auto bad_tol = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    bad_tol.pilot_knee_tolerance = -1.0;
    expect_throw(bad_tol);
    auto perm = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    perm.cluster_dummies         = false;
    perm.pilot_shared_dummies    = true;
    perm.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION;
    expect_throw(perm);
}

/** @brief The pure group-majority count: a group counts with its members
 *  ahead only when at least the fraction of it is ahead. */
TEST(TikLambda2Pilot, GroupMajorityCountPure) {
    // groups of 4: {3 of 4 ahead}, {1 of 4}, {2 of 4}, {0 of 4}
    const std::vector<Eigen::Index> groups = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    Eigen::VectorXd phi(16);
    phi << 1.0, 0.9, 0.6, 0.2,  0.8, 0.1, 0.0, 0.3,  0.7, 0.55, 0.5, 0.0,  0.4, 0.3, 0.2, 0.1;
    const GroupMajorityCount half = group_majority_count(phi, groups, 0.5);
    EXPECT_EQ(half.members, 5U);   // group 0 (3) + group 2 (2); phi = 0.5 is not ahead
    EXPECT_EQ(half.groups, 2U);
    const GroupMajorityCount all = group_majority_count(phi, groups, 1.0);
    EXPECT_EQ(all.members, 0U);
    EXPECT_EQ(all.groups, 0U);
    const GroupMajorityCount any = group_majority_count(phi, groups, 0.25);
    EXPECT_EQ(any.members, 6U);    // groups 0, 1, 2
    EXPECT_EQ(any.groups, 3U);
    EXPECT_THROW(group_majority_count(phi, {0, 1}, 0.5), std::invalid_argument);
    EXPECT_THROW(group_majority_count(phi, groups, 0.0), std::invalid_argument);
    EXPECT_THROW(group_majority_count(phi, groups, 1.5), std::invalid_argument);
    std::vector<Eigen::Index> neg = groups;
    neg[0] = -1;
    EXPECT_THROW(group_majority_count(phi, neg, 0.5), std::invalid_argument);
}

/** @brief The group-majority statistic in the pilot: never above the member
 *  count, the pick read off it, the main run still the fixed-lambda_2 run at
 *  the pick; refused without a partition or with a fraction outside (0, 1]. */
TEST(TikLambda2Pilot, GroupMajorityStatistic) {
    const BlockData d = testDesign();
    auto ctrl = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    ctrl.pilot_statistic = PilotStatistic::GROUP_MAJORITY;
    const SelectorRun r = runSelector(d, ctrl, 2026);
    ASSERT_TRUE(r.pilot.ran);
    ASSERT_EQ(r.pilot.group_majority.size(), kPilotGrid.size());
    ASSERT_EQ(r.pilot.groups_qualifying.size(), kPilotGrid.size());
    for (std::size_t g = 0; g < kPilotGrid.size(); ++g) {
        EXPECT_LE(r.pilot.group_majority[g], r.pilot.curve_majority[g][r.pilot.T_pilot - 1]);
        EXPECT_EQ(r.pilot.counts[g], r.pilot.group_majority[g]);
    }
    EXPECT_EQ(r.pilot.pick.index, pick_lambda2_index(r.pilot.counts, 1.0).index);
    EXPECT_GT(r.lambda2, 0.0);   // whole active blocks enter as majorities: the prior still wins

    // the member statistic on the same seed records the same group counts (diagnostic columns)
    auto plain = ctrl;
    plain.pilot_statistic = PilotStatistic::MAJORITY;
    const SelectorRun m = runSelector(d, plain, 2026);
    EXPECT_EQ(m.pilot.group_majority, r.pilot.group_majority);
    EXPECT_EQ(m.pilot.curve_majority, r.pilot.curve_majority);

    auto fixed = centeringControl(d.groups);
    fixed.lambda_2 = r.lambda2;
    EXPECT_EQ(r.selected, runSelector(d, fixed, 2026).selected);

    const auto expect_throw = [&](const TRexTikhonovControlParameter& c) {
        Eigen::MatrixXd X = d.X;
        Eigen::VectorXd y = d.y;
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        EXPECT_THROW(TRexTikhonovSelector(Xm, ym, 0.1, c, 1, false), std::invalid_argument);
    };
    auto no_groups = ctrl;
    no_groups.penalty_structure = TikPenaltyStructure::SPARSE_GENERAL;
    no_groups.tikhonov_K        = groupMeanK(d.groups);
    no_groups.prior_groups.clear();
    no_groups.cluster_dummies   = false;
    expect_throw(no_groups);
    auto bad_fraction = ctrl;
    bad_fraction.pilot_group_fraction = 0.0;
    expect_throw(bad_fraction);
    bad_fraction.pilot_group_fraction = 1.5;
    expect_throw(bad_fraction);
}

/** @brief runPilotOnly() gives the pick of a full select() with the same
 *  seed, without the main selection. */
TEST(TikLambda2Pilot, RunPilotOnly) {
    const BlockData d = testDesign();
    const auto ctrl = pilotControl(centeringControl(d.groups), TikLambda2Method::PILOT_KNEE, kPilotGrid, 5);
    const SelectorRun full = runSelector(d, ctrl, 2026);

    Eigen::MatrixXd X = d.X;
    Eigen::VectorXd y = d.y;
    Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
    TRexTikhonovSelector selector(Xm, ym, 0.1, ctrl, 2026, false);
    const Lambda2PilotResult& only = selector.runPilotOnly();
    EXPECT_TRUE(only.ran);
    EXPECT_EQ(only.counts, full.pilot.counts);
    EXPECT_DOUBLE_EQ(only.lambda2, full.pilot.lambda2);
    EXPECT_DOUBLE_EQ(selector.getLambda2Used(), full.lambda2);
    EXPECT_TRUE(selector.getLambda2CVDiagnostics().ran);
    EXPECT_TRUE(X.isApprox(d.X));   // X released and restored
}

} // namespace trex::test::trex_selector_methods::ad::trex_tikhonov
