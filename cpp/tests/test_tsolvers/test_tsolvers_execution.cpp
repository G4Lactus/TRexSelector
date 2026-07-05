// ========================================================================================
// test_tsolvers_execution.cpp
// ========================================================================================
/**
 * @file test_tsolvers_execution.cpp
 *
 * @brief Unit tests verifying core execution path integrity (bounds checking, monotonicity)
 *        for TSolver_Base via the concrete proxy TLARS_Solver.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <map>
#include <set>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tstagewise_solver.hpp>
#include <tsolvers/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tgp_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::linear_model {

using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;

// ========================================================================================

class TSolverExecutionTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;

    void SetUp() override {
        // Use datagen utilities to get well-conditioned design matrices
        SyntheticData data(
            200, 50, 50,
            {0, 10, 20},
            {5.0, -3.0, 2.0},
            1.0, 42, -1, -1,
            predictor_policy::Normal(),
            dummygen::Distribution::Normal(),
            noisegen::noise_policy::Normal()
        );

        X = data.getX();
        D = data.getD();
        y = data.getY();
    }
};

// ========================================================================================

/** @brief Test to validate that the solver properly respects early T_stop halting constraints. */
TEST_F(TSolverExecutionTest, ExecuteEarlyStoppingAtTStop) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map, true,
                        true, false);

    // Goal: Stop when 4 dummy variables have entered the active set.
    int T_stop = 4;
    solver.executeStep(T_stop, true);

    // It should have executed some steps...
    EXPECT_GT(solver.getNumSteps(), 0);

    // Check dummy capacity actively constrained to exactly what we requested
    auto active_dummies = solver.getActiveDummyIndices();
    EXPECT_EQ(active_dummies.size(), T_stop);
}


/** @brief Test to ensure the residual sum of squares shrinks monotonically along the inclusion path. */
TEST_F(TSolverExecutionTest, MonotonicRSSDecrease) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map, true,
                        true, false);

    // Execute multiple steps
    solver.executeStep(10, true);

    const auto& rss_path = solver.getRSS();
    EXPECT_FALSE(rss_path.empty());

    // RSS should strictly decrease (or at least remain non-increasing) over successful
    // variable inclusions
    for (std::size_t i = 1; i < rss_path.size(); ++i) {
        // We use EXPECT_LE due to minor precision limits on edge updates
        EXPECT_LE(rss_path[i], rss_path[i-1] + 1e-12);
    }
}

/** @brief MP and GP must re-select atoms (their defining difference from
 *         OMP): on a full path, some atom is picked in more than one step,
 *         while the active set keeps each index exactly once. */
TEST_F(TSolverExecutionTest, MPAndGPReSelectAtoms) {
    using trex::tsolvers::linear_model::omp_based::TMP_Solver;
    using trex::tsolvers::linear_model::omp_based::TGP_Solver;

    auto max_selection_count = [this](auto make_solver) {
        Eigen::MatrixXd Xc = X, Dc = D;
        Eigen::VectorXd yc = y;
        Eigen::Map<Eigen::MatrixXd> X_map(Xc.data(), Xc.rows(), Xc.cols());
        Eigen::Map<Eigen::MatrixXd> D_map(Dc.data(), Dc.rows(), Dc.cols());
        Eigen::Map<Eigen::VectorXd> y_map(yc.data(), yc.size());
        auto solver = make_solver(X_map, D_map, y_map);
        solver.executeStep(0, false); // full path

        std::map<int, int> counts;
        for (const auto& step : solver.getActions()) {
            for (int a : step) {
                if (a >= 0) { counts[a]++; }
            }
        }
        int max_count = 0;
        for (const auto& [idx, c] : counts) { max_count = std::max(max_count, c); }

        // Active set holds each index exactly once despite repeated picks
        std::set<std::size_t> unique_actives(solver.getActives().begin(),
                                             solver.getActives().end());
        EXPECT_EQ(unique_actives.size(), solver.getActives().size());
        return max_count;
    };

    int mp_max = max_selection_count([](auto& Xm, auto& Dm, auto& ym) {
        return TMP_Solver(Xm, Dm, ym, true, true, false);
    });
    EXPECT_GE(mp_max, 2) << "MP never re-selected an atom on a full path";

    int gp_max = max_selection_count([](auto& Xm, auto& Dm, auto& ym) {
        return TGP_Solver(Xm, Dm, ym, true, true, false);
    });
    EXPECT_GE(gp_max, 2) << "GP never re-selected an atom on a full path";
}


/** @brief Pin the documented equivalence (Locatello et al. 2017, Alg. 4): for
 *         the least-squares objective (L = 1), NCGMP-LineSearch is exactly
 *         T-MP and NCGMP-FullyCorrective is exactly T-OMP. */
TEST_F(TSolverExecutionTest, NCGMPVariantsMatchMPAndOMP) {
    using trex::tsolvers::linear_model::omp_based::TMP_Solver;
    using trex::tsolvers::linear_model::omp_based::TOMP_Solver;
    using trex::tsolvers::linear_model::omp_based::TNCGMP_Solver;
    using trex::tsolvers::linear_model::omp_based::NCGMPVariant;

    auto fit = [this](auto make_solver) {
        Eigen::MatrixXd Xc = X, Dc = D;
        Eigen::VectorXd yc = y;
        Eigen::Map<Eigen::MatrixXd> X_map(Xc.data(), Xc.rows(), Xc.cols());
        Eigen::Map<Eigen::MatrixXd> D_map(Dc.data(), Dc.rows(), Dc.cols());
        Eigen::Map<Eigen::VectorXd> y_map(yc.data(), yc.size());
        auto solver = make_solver(X_map, D_map, y_map);
        solver.executeStep(4, true);
        return solver.getBetaPath();
    };

    Eigen::MatrixXd mp_path = fit([](auto& Xm, auto& Dm, auto& ym) {
        return TMP_Solver(Xm, Dm, ym, true, true, false);
    });
    Eigen::MatrixXd ncg_ls_path = fit([](auto& Xm, auto& Dm, auto& ym) {
        return TNCGMP_Solver(Xm, Dm, ym, NCGMPVariant::LineSearch, true, true, false);
    });
    ASSERT_EQ(mp_path.cols(), ncg_ls_path.cols());
    EXPECT_TRUE(mp_path.isApprox(ncg_ls_path, 1e-12))
        << "max abs diff: " << (mp_path - ncg_ls_path).cwiseAbs().maxCoeff();

    Eigen::MatrixXd omp_path = fit([](auto& Xm, auto& Dm, auto& ym) {
        return TOMP_Solver(Xm, Dm, ym, true, true, false);
    });
    Eigen::MatrixXd ncg_fc_path = fit([](auto& Xm, auto& Dm, auto& ym) {
        return TNCGMP_Solver(Xm, Dm, ym, NCGMPVariant::FullyCorrective, true, true, false);
    });
    ASSERT_EQ(omp_path.cols(), ncg_fc_path.cols());
    EXPECT_TRUE(omp_path.isApprox(ncg_fc_path, 1e-12))
        << "max abs diff: " << (omp_path - ncg_fc_path).cwiseAbs().maxCoeff();
}


/** @brief Coverage for the Fully Corrective (OMP) variant of TNCGMP, which the
 *         typed suite does not reach (its factory hardcodes LineSearch). Also
 *         asserts the actions/diagnostics alignment invariant that collinear
 *         rejections must not break. */
TEST_F(TSolverExecutionTest, TNCGMPFullyCorrectiveExecutesAndStaysAligned) {
    using trex::tsolvers::linear_model::omp_based::TNCGMP_Solver;
    using trex::tsolvers::linear_model::omp_based::NCGMPVariant;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TNCGMP_Solver solver(X_map, D_map, y_map, NCGMPVariant::FullyCorrective,
                         true, true, false);
    solver.executeStep(4, true);

    EXPECT_GT(solver.getNumSteps(), 0u);
    EXPECT_EQ(solver.getActiveDummyIndices().size(), 4u);

    // One actions entry per recorded step, aligned with the diagnostics arrays
    EXPECT_EQ(solver.getActions().size(), solver.getNumSteps());
    EXPECT_EQ(solver.getActions().size(), solver.getRSS().size() - 1);
    EXPECT_EQ(solver.getRSS().size(), solver.getDoF().size());

    // Fully corrective refits must decrease RSS monotonically
    const auto& rss = solver.getRSS();
    for (std::size_t i = 1; i < rss.size(); ++i) {
        EXPECT_LE(rss[i], rss[i - 1] + 1e-12);
    }
}


/** @brief Regression test: executeStep() with defaults (T_stop=0, early_stop=true)
 *         must compute the full solution path as documented — historically it was
 *         a silent no-op because the loop guard treated T_stop=0 as "zero dummies
 *         allowed". T_stop=0 now means "no dummy limit". */
TEST_F(TSolverExecutionTest, DefaultExecuteStepComputesFullPath) {
    auto X2 = X; auto D2 = D; auto y2 = y;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver by_default(X_map, D_map, y_map, true, true, false);
    by_default.executeStep(); // T_stop = 0, early_stop = true

    Eigen::Map<Eigen::MatrixXd> X_map2(X2.data(), X2.rows(), X2.cols());
    Eigen::Map<Eigen::MatrixXd> D_map2(D2.data(), D2.rows(), D2.cols());
    Eigen::Map<Eigen::VectorXd> y_map2(y2.data(), y2.size());

    TLARS_Solver explicit_full(X_map2, D_map2, y_map2, true, true, false);
    explicit_full.executeStep(0, false); // documented full-path form

    EXPECT_GT(by_default.getNumSteps(), 0u);
    EXPECT_EQ(by_default.getNumSteps(), explicit_full.getNumSteps());
    EXPECT_EQ(by_default.getActions(), explicit_full.getActions());
}


/** @brief Pin T-LASSO cycling behavior: zero-crossing drops must keep the
 *         lambda sequence monotone, zero the dropped coefficient at the drop
 *         step, and keep the actions/diagnostics arrays aligned. */
TEST_F(TSolverExecutionTest, LassoZeroCrossingDropsBehaveConsistently) {
    std::srand(3);

    // Strongly correlated predictors with opposing effects provoke
    // coefficient sign changes and hence LASSO drops.
    Eigen::MatrixXd Xc = Eigen::MatrixXd::Random(200, 20);
    Xc.col(1) = 0.95 * Xc.col(0) + 0.05 * Xc.col(1);
    Xc.col(2) = 0.90 * Xc.col(0) + 0.10 * Xc.col(2);
    Eigen::MatrixXd Dc = Eigen::MatrixXd::Random(200, 20);
    Eigen::VectorXd yc = 5.0 * Xc.col(0) - 4.0 * Xc.col(1) + 2.0 * Xc.col(2)
                         + 0.1 * Eigen::VectorXd::Random(200);

    Eigen::Map<Eigen::MatrixXd> X_map(Xc.data(), Xc.rows(), Xc.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(Dc.data(), Dc.rows(), Dc.cols());
    Eigen::Map<Eigen::VectorXd> y_map(yc.data(), yc.size());

    TLASSO_Solver solver(X_map, D_map, y_map, true, true, false);
    solver.executeStep(0, false); // full path

    // The design must actually trigger drops, otherwise this test is vacuous.
    ASSERT_GT(solver.getNumRemovals(), 0u);

    // Max correlation must be non-increasing along the path
    const auto& lambda = solver.getLambda();
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LE(lambda[k], lambda[k - 1] + 1e-8) << "lambda jumped upward at step " << k;
    }

    // One actions entry per recorded step
    const auto& actions = solver.getActions();
    EXPECT_EQ(actions.size(), solver.getRSS().size() - 1);

    // Every drop must zero the coefficient at its step (beta column k+1)
    Eigen::MatrixXd path = solver.getBetaPath();
    for (std::size_t k = 0; k < actions.size(); ++k) {
        for (int a : actions[k]) {
            if (a < 0) {
                EXPECT_EQ(path(static_cast<Eigen::Index>(-a),
                                static_cast<Eigen::Index>(k + 1)), 0.0)
                    << "dropped variable " << -a << " has nonzero beta at step " << k + 1;
            }
        }
    }
}


/** @brief Regression test: NNLS drops in T-Stagewise must refresh the dropped
 *         variable's correlation entry. A stale (inflated) entry re-enters
 *         findTiedMaxCorrelations and shows up as an upward jump in the lambda
 *         (max correlation) sequence, which must be non-increasing. */
TEST_F(TSolverExecutionTest, StagewiseNNLSDropKeepsLambdaMonotone) {
    std::srand(7);

    // Strongly correlated predictors with opposing effects provoke
    // direction-sign violations and hence NNLS removals.
    Eigen::MatrixXd Xc = Eigen::MatrixXd::Random(200, 20);
    Xc.col(1) = 0.95 * Xc.col(0) + 0.05 * Xc.col(1);
    Xc.col(2) = 0.90 * Xc.col(0) + 0.10 * Xc.col(2);
    Eigen::MatrixXd Dc = Eigen::MatrixXd::Random(200, 20);
    Eigen::VectorXd yc = 5.0 * Xc.col(0) - 4.0 * Xc.col(1) + 2.0 * Xc.col(2)
                         + 0.1 * Eigen::VectorXd::Random(200);

    Eigen::Map<Eigen::MatrixXd> X_map(Xc.data(), Xc.rows(), Xc.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(Dc.data(), Dc.rows(), Dc.cols());
    Eigen::Map<Eigen::VectorXd> y_map(yc.data(), yc.size());

    TSTAGEWISE_Solver solver(X_map, D_map, y_map, true, true, false);
    solver.executeStep(0, false); // full path

    // The design must actually trigger NNLS removals, otherwise this test is vacuous.
    EXPECT_GT(solver.getNumRemovals(), 0u);

    // Max correlation must be non-increasing along the path.
    const auto& lambda = solver.getLambda();
    ASSERT_FALSE(lambda.empty());
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LE(lambda[k], lambda[k - 1] + 1e-8) << "lambda jumped upward at step " << k;
    }
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::linear_model */
