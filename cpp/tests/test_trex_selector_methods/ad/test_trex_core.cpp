// ========================================================================================
// test_trex_core.cpp
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <optional>
#include <vector>

// project trex includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::ad::trex_core {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::ad::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace disp = trex::trex_selector_methods::ad::utils::solver_dispatch;

// ========================================================================================


/**
 * @brief Helper to set up TRexSelector boilerplate for testing.
 *
 * @param n Number of samples
 * @param p Number of features
 * @param solver Solver type for TRex
 * @param l_strategy L loop strategy
 * @param multi_threaded Whether to use multi-threading
 */
void run_trex_core_test(Eigen::Index n,
                        Eigen::Index p,
                        disp::SolverTypeForTRex solver = disp::SolverTypeForTRex::TLARS,
                        LLoopStrategy l_strategy = LLoopStrategy::HCONCAT,
                        bool multi_threaded = false)
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
    trex_ctrl.K = 3; // Keep K small for fast testing
    trex_ctrl.max_dummy_multiplier = 2;
    trex_ctrl.solver_type = solver;
    trex_ctrl.lloop_strategy = l_strategy;

    if (multi_threaded) {
        trex_ctrl.parallel_rnd_experiments = true;
        trex_ctrl.max_outer_threads = 2;
    }

    TRexSelector trex(X_map, y_map, 0.1, trex_ctrl, 42,
                      false);

    // Assert configuration getters
    EXPECT_EQ(trex.getTFDR(), 0.1);
    EXPECT_EQ(trex.getK(), 3);
    EXPECT_EQ(trex.getMaxNumDummies(), 2);

    // Test the select() execution safely
    auto result = trex.select();

    // Dimensions shouldn't crash and MUST match `p`
    EXPECT_EQ(result.Phi_prime.size(), p);
    EXPECT_GE(result.selected_var.size(), 0);

    if (result.T_stop > 0) {
        EXPECT_EQ(result.Phi_mat.cols(), p);
        EXPECT_EQ(result.Phi_mat.rows(), result.T_stop);
    }

    // Phi_prime is capped at 1 and may be negative (no lower clamp, see computePhiPrime)
    for (Eigen::Index j = 0; j < p; ++j) {
        EXPECT_LE(result.Phi_prime(j), 1.0);
    }
}

/** @brief Test TRexSelector with low-dimensional data. */
TEST(TRexCoreTest, LowDimensional) {
    run_trex_core_test(300, 100);
}


/** @brief Test TRexSelector with high-dimensional data. */
TEST(TRexCoreTest, HighDimensional) {
    run_trex_core_test(150, 300);
}


/** @brief Test TRexSelector with TLASSO solver. */
TEST(TRexCoreTest, Param_Solvers_TLASSO) {
    run_trex_core_test(150, 200, disp::SolverTypeForTRex::TLASSO);
}


/** @brief Test TRexSelector with the CCD lasso solver (TCDLASSO). */
TEST(TRexCoreTest, Param_Solvers_TCDLASSO) {
    run_trex_core_test(150, 200, disp::SolverTypeForTRex::TCDLASSO);
}


/** @brief Test TRexSelector with the CCD elastic-net solver (TCDENET). */
TEST(TRexCoreTest, Param_Solvers_TCDENET) {
    run_trex_core_test(150, 200, disp::SolverTypeForTRex::TCDENET);
}


/** @brief Test TRexSelector with the row-augmented elastic net (TENET_AUG). */
TEST(TRexCoreTest, Param_Solvers_TENET_AUG) {
    run_trex_core_test(150, 200, disp::SolverTypeForTRex::TENET_AUG);
}


/** @brief A select-time C++ exception must surface as a catchable exception,
 *  not a process abort. Regression test for the OpenMP structured-block
 *  rule: an exception may not leave an `omp parallel` region — even an
 *  INACTIVE one (`if(false)`, single thread) — or the runtime calls
 *  std::terminate, which killed the R host session before the runner and
 *  GVS loops gained the capture-and-rethrow discipline. TCDIENET without a
 *  Tikhonov matrix throws inside the experiment loop at dispatch time,
 *  making it the canonical trigger. */
TEST(TRexCoreTest, SelectTimeThrowPropagatesInsteadOfTerminating) {
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(60, 20, support, coefs, 1.0, 42);

    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3;
    trex_ctrl.max_dummy_multiplier = 2;
    trex_ctrl.solver_type = disp::SolverTypeForTRex::TCDIENET;

    // Serial path (parallel_rnd_experiments = false): the omp region is
    // inactive but still a structured block.
    TRexSelector trex(X_map, y_map, 0.1, trex_ctrl, 42, false);
    EXPECT_THROW(trex.select(), std::invalid_argument);

    // Parallel path: the same discipline must hold with live threads.
    TRexControlParameter par_ctrl = trex_ctrl;
    par_ctrl.parallel_rnd_experiments = true;
    par_ctrl.max_outer_threads = 2;
    SyntheticData data2(60, 20, support, coefs, 1.0, 42);
    auto X2 = data2.getX();
    auto y2 = data2.getY();
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), X2.rows(), X2.cols());
    Eigen::Map<Eigen::VectorXd> y2_map(y2.data(), y2.size());
    TRexSelector trex_par(X2_map, y2_map, 0.1, par_ctrl, 42, false);
    EXPECT_THROW(trex_par.select(), std::invalid_argument);
}


/** @brief TENET_AUG is the row-augmented formulation of the SAME elastic-net
 *  problem TENET solves via the Gram-based path (proved equivalent for
 *  lambda2 >= 0), so with identical data and seed the two must select the
 *  same variables in the core selector. */
TEST(TRexCoreTest, Equivalence_TenetAugMatchesTenet) {

    auto run_with = [](disp::SolverTypeForTRex solver) {
        std::vector<std::size_t> support = {1, 2, 3};
        std::vector<double> coefs = {5.0, -3.0, 2.0};
        SyntheticData data(150, 60, support, coefs, 1.0, 42);

        auto X = data.getX();
        auto y = data.getY();
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

        TRexControlParameter trex_ctrl;
        trex_ctrl.K = 3;
        trex_ctrl.max_dummy_multiplier = 2;
        trex_ctrl.solver_type = solver;

        TRexSelector trex(X_map, y_map, 0.1, trex_ctrl, 42, false);
        return trex.select().selected_var;
    };

    const auto sel_tenet = run_with(disp::SolverTypeForTRex::TENET);
    const auto sel_aug   = run_with(disp::SolverTypeForTRex::TENET_AUG);

    ASSERT_EQ(sel_tenet.size(), sel_aug.size());
    EXPECT_TRUE((sel_tenet.array() == sel_aug.array()).all())
        << "TENET_AUG selection diverged from TENET in the core selector.";
}


/** @brief Test TRexSelector with SKIPL L loop strategy. */
TEST(TRexCoreTest, Param_LStrategies_SKIPL) {
    run_trex_core_test(150, 200, disp::SolverTypeForTRex::TLARS,
                       LLoopStrategy::SKIPL);
}


/** @brief Test TRexSelector with concurrency enabled. */
TEST(TRexCoreTest, Param_Concurrency) {
    // Parallel random experiments on two threads
    run_trex_core_test(150, 200, disp::SolverTypeForTRex::TLARS,
                       LLoopStrategy::HCONCAT, true);
}


/** @brief The thread options are validated at construction: a value below
 *  1 throws (0 included: there is no "every core" value), the outer count is
 *  capped at K, and the defaults are one thread each (no parallelism unless
 *  asked for). */
TEST(TRexCoreTest, ThreadOptionsResolveAtConstruction) {
    TRexControlParameter ctrl;
    ctrl.K = 4;
    ctrl.resolveThreadOptions();
    EXPECT_EQ(ctrl.max_outer_threads, 1);
    EXPECT_EQ(ctrl.max_inner_threads, 1);

    TRexControlParameter zero_outer;
    zero_outer.K = 4;
    zero_outer.max_outer_threads = 0;
    EXPECT_THROW(zero_outer.resolveThreadOptions(), std::invalid_argument);
    TRexControlParameter zero_inner;
    zero_inner.K = 4;
    zero_inner.max_inner_threads = 0;
    EXPECT_THROW(zero_inner.resolveThreadOptions(), std::invalid_argument);

    TRexControlParameter capped;
    capped.K = 2;
    capped.max_outer_threads = 64;
    capped.resolveThreadOptions();
    EXPECT_EQ(capped.max_outer_threads, 2);

    TRexControlParameter negative;
    negative.max_inner_threads = -1;
    EXPECT_THROW(negative.resolveThreadOptions(), std::invalid_argument);

    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(60, 20, support, coefs, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());
    TRexControlParameter bad;
    bad.max_outer_threads = -2;
    EXPECT_THROW(TRexSelector(X_map, y_map, 0.1, bad, 42, false), std::invalid_argument);
}


/** @brief A seeded selector selects the same variables with the experiments
 *  in sequence and in parallel: every experiment derives its own seed, so
 *  the dummies do not depend on the thread that draws them. */
TEST(TRexCoreTest, ParallelExperimentsSelectAsSerial) {
    if (!trex::utils::openmp::is_available() || omp_get_num_procs() < 2) {
        GTEST_SKIP() << "needs OpenMP and at least two cores";
    }
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(100, 40, support, coefs, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();

    auto run = [&](bool parallel, int outer, int inner) {
        Eigen::MatrixXd Xc = X;
        Eigen::VectorXd yc = y;
        Eigen::Map<Eigen::MatrixXd> X_map(Xc.data(), Xc.rows(), Xc.cols());
        Eigen::Map<Eigen::VectorXd> y_map(yc.data(), yc.size());
        TRexControlParameter ctrl;
        ctrl.K = 4;
        ctrl.max_dummy_multiplier = 2;
        ctrl.parallel_rnd_experiments = parallel;
        ctrl.max_outer_threads = outer;
        ctrl.max_inner_threads = inner;
        TRexSelector trex(X_map, y_map, 0.1, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const auto serial   = run(false, 1, 1);
    const auto parallel = run(true, 2, 1);
    const auto budget   = run(false, 1, 2);
    ASSERT_EQ(serial.size(), parallel.size());
    EXPECT_TRUE((serial.array() == parallel.array()).all())
        << "parallel experiments selected differently from the sequential run";
    EXPECT_TRUE((serial.array() == budget.array()).all())
        << "the solvers' thread budget changed the selection";
}


/** @brief Test TRexSelector with invalid input dimensions. */
TEST(TRexCoreTest, Exceptions_InvalidDimensions) {
    Eigen::MatrixXd X_empty(0, 0);
    Eigen::VectorXd y_empty(0);
    Eigen::Map<Eigen::MatrixXd> X_map(X_empty.data(), 0, 0);
    Eigen::Map<Eigen::VectorXd> y_map(y_empty.data(), 0);

    TRexControlParameter trex_ctrl;
    EXPECT_THROW(TRexSelector(X_map, y_map, 0.1, trex_ctrl),
                 std::invalid_argument);
}


/** @brief Test TRexSelector with mismatched input dimensions. */
TEST(TRexCoreTest, Exceptions_MismatchedDimensions) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 50);
    Eigen::VectorXd y = Eigen::VectorXd::Random(99); // Mismatched
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    EXPECT_THROW(TRexSelector(X_map, y_map, 0.1, trex_ctrl),
                 std::invalid_argument);
}


/** @brief Test TRexSelector with invalid parameters. */
TEST(TRexCoreTest, Exceptions_InvalidParameters) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 50);
    Eigen::VectorXd y = Eigen::VectorXd::Random(100);
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;

    // Test invalid tFDR bounds
    EXPECT_THROW(TRexSelector(X_map, y_map, -0.1, trex_ctrl),
                 std::invalid_argument);
    EXPECT_THROW(TRexSelector(X_map, y_map, 1.5, trex_ctrl),
                 std::invalid_argument);

    // Test invalid K
    trex_ctrl.K = 0;
    EXPECT_THROW(TRexSelector(X_map, y_map, 0.1, trex_ctrl),
                 std::invalid_argument);
}


// ========================================================================================
// Data Integrity
// ========================================================================================

/** @brief X is restored to its original values after select() returns (object still alive). */
TEST(TRexCoreTest, DataIntegrity_XRestoredAfterSelect) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(30, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(30);
    Eigen::MatrixXd X_copy = X;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3;
    trex_ctrl.max_dummy_multiplier = 2;

    TRexSelector trex(X_map, y_map, 0.1, trex_ctrl, 42, false);
    trex.select();

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored after TRexSelector::select().";
}


/** @brief X is restored to its original values when the object is destroyed without
 *         calling select() (normalization happens in the constructor). */
TEST(TRexCoreTest, DataIntegrity_XRestoredOnDestruction) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(30, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(30);
    Eigen::MatrixXd X_copy = X;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 3;
    trex_ctrl.max_dummy_multiplier = 2;

    {
        TRexSelector trex(X_map, y_map, 0.1, trex_ctrl, 42, false);
        // X is now normalized. Destructor fires here.
    }

    EXPECT_TRUE(X.isApprox(X_copy, 1e-12))
        << "X was not restored by TRexSelector destructor.";
}


// ========================================================================================
// Row bound of the solvers on the selector-centered design
// ========================================================================================

/** @brief Exposes the protected runner configuration of the selector. */
class RunnerConfigProbe : public TRexSelector {
public:
    using TRexSelector::TRexSelector;
    using TRexSelector::buildRunnerConfig;
};

/** @brief The selector centers X and y in every ScalingMode except NONE, so the
 *         runner marks the design precentered (the solvers cap their active sets
 *         at n - 1, the rank of the centered design); NONE leaves the design
 *         untouched and keeps the bound n. The dispatch forwards the flag into
 *         the solver options.
 */
TEST(TRexCoreTest, RunnerMarksCenteredDesignPrecentered) {
    namespace tdn = trex::trex_selector_methods::ad::utils::data_normalizer;
    namespace ter = trex::trex_selector_methods::ad::utils::experiment_runner;

    for (const auto mode : {tdn::ScalingMode::L2, tdn::ScalingMode::ZSCORE, tdn::ScalingMode::NONE}) {
        Eigen::MatrixXd X = Eigen::MatrixXd::Random(30, 10);
        Eigen::VectorXd y = Eigen::VectorXd::Random(30);
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

        TRexControlParameter trex_ctrl;
        trex_ctrl.K = 3;
        trex_ctrl.max_dummy_multiplier = 2;
        trex_ctrl.scaling_mode = mode;

        RunnerConfigProbe trex(X_map, y_map, 0.1, trex_ctrl, 42, false);
        const ter::ExperimentRunnerConfig cfg =
            trex.buildRunnerConfig(10, 1, false, ter::ExperimentStrategy::Standard);
        EXPECT_EQ(cfg.precentered, mode != tdn::ScalingMode::NONE)
            << "scaling mode " << static_cast<int>(mode);
    }

    Eigen::MatrixXd Xs = Eigen::MatrixXd::Zero(4, 2), Ds = Eigen::MatrixXd::Zero(4, 2);
    Eigen::VectorXd ys = Eigen::VectorXd::Zero(4);
    Eigen::Map<Eigen::MatrixXd> Xs_map(Xs.data(), 4, 2), Ds_map(Ds.data(), 4, 2);
    Eigen::Map<Eigen::VectorXd> ys_map(ys.data(), 4);
    disp::SolverConfig solver_cfg{.X = Xs_map, .D = Ds_map, .y = ys_map, .precentered = true};
    EXPECT_TRUE(disp::toSolverOptions(solver_cfg).precentered);
    solver_cfg.precentered = false;
    EXPECT_FALSE(disp::toSolverOptions(solver_cfg).precentered);
}


// ========================================================================================
// Stagnation-stop AUTO default (resolved by solver family in the constructor)
// ========================================================================================

/** @brief The AUTO default disables stagnation control for equiangular
 *         LARS-path solvers (R reference / paper behavior) and enables it for
 *         greedy solvers (noise-trap guard); an explicit value always wins. */
TEST(TRexCoreTest, StagnationStop_AutoDefaultResolvesBySolverFamily) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(30, 10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(30);
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    auto resolved_default = [&](disp::SolverTypeForTRex solver,
                                std::optional<bool> user_choice) {
        TRexControlParameter ctrl;
        ctrl.K = 3;
        ctrl.solver_type = solver;
        ctrl.tloop_stagnation_stop = user_choice;
        TRexSelector trex(X_map, y_map, 0.1, ctrl, 42, false);
        return trex.getStagnationCheck();
    };

    // AUTO: off for the equiangular LARS family, on for greedy solvers.
    EXPECT_FALSE(resolved_default(disp::SolverTypeForTRex::TLARS,  std::nullopt));
    EXPECT_FALSE(resolved_default(disp::SolverTypeForTRex::TLASSO, std::nullopt));
    EXPECT_FALSE(resolved_default(disp::SolverTypeForTRex::TENET,  std::nullopt));
    EXPECT_TRUE(resolved_default(disp::SolverTypeForTRex::TOMP,    std::nullopt));
    EXPECT_TRUE(resolved_default(disp::SolverTypeForTRex::TSTEPWISE, std::nullopt));
    EXPECT_TRUE(resolved_default(disp::SolverTypeForTRex::TAFS,    std::nullopt));

    // CCD family: currently classified non-greedy (exact penalized-lasso
    // minimizers per crossing -> AUTO off, same as the equiangular paths).
    // Whether CD paths can stagnate in the T-loop is an open question under
    // investigation; these entries lock in the CURRENT default, not a
    // settled answer.
    EXPECT_FALSE(resolved_default(disp::SolverTypeForTRex::TCDLASSO,   std::nullopt));
    EXPECT_FALSE(resolved_default(disp::SolverTypeForTRex::TCDENET, std::nullopt));

    // Explicit user choice overrides the family default in both directions.
    EXPECT_TRUE(resolved_default(disp::SolverTypeForTRex::TLARS, true));
    EXPECT_FALSE(resolved_default(disp::SolverTypeForTRex::TOMP, false));
}


// ========================================================================================
// Warm-start wiring (regression: solvers were never retained, so warm starts
// silently never engaged and every T-step re-solved from scratch)
// ========================================================================================

namespace {

/** @brief Shared setup for the ExperimentRunner warm-start tests. */
struct RunnerTestData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    RunnerTestData(Eigen::Index n, Eigen::Index p) {
        std::vector<std::size_t> support = {1, 2, 3};
        std::vector<double> coefs = {5.0, -3.0, 2.0};
        SyntheticData data(n, p, support, coefs, 1.0, 42);
        X = data.getX();
        y = data.getY();
    }
};

/** @brief Baseline ExperimentRunnerConfig for warm-start tests. */
er::ExperimentRunnerConfig make_runner_cfg(std::size_t K,
                                           std::size_t p,
                                           er::ExperimentStrategy strategy) {
    er::ExperimentRunnerConfig cfg;
    cfg.K = K;
    cfg.num_dummies = p;
    cfg.T_stop = 1;
    cfg.p = p;
    cfg.strategy = strategy;
    cfg.use_warm_start = true;
    return cfg;
}

} // namespace


/** @brief In-memory warm start engages (solvers retained after the first
 *         warm run), the continued T-step reproduces a cold run exactly, and
 *         invalidate() keeps the slots repopulatable. */
TEST(TRexCoreTest, WarmStart_EngagesAndMatchesColdRun) {
    const Eigen::Index n = 60, p = 20;
    const std::size_t K = 3;
    RunnerTestData d(n, p);
    Eigen::Map<Eigen::MatrixXd> X_map(d.X.data(), d.X.rows(), d.X.cols());

    dg::DummyGenerator dummy_gen(static_cast<std::size_t>(n),
                                 dummygen::Distribution::Normal(), 42, false);
    dummy_gen.generateAndStore(K, static_cast<std::size_t>(p), 1);

    wsm::WarmStartManager warm_mgr(K, wsm::WarmStartMode::IN_MEMORY, false);
    er::ExperimentRunner runner(X_map, d.y, dummy_gen, warm_mgr, nullptr);

    auto cfg = make_runner_cfg(K, static_cast<std::size_t>(p),
                               er::ExperimentStrategy::Standard);

    // T = 1: fresh solvers must be retained for continuation.
    runner.run(cfg);
    for (std::size_t k = 0; k < K; ++k) {
        EXPECT_TRUE(warm_mgr.hasSolver(k))
            << "Solver " << k << " was not retained after a warm-start run.";
    }

    // T = 2: warm continuation.
    cfg.T_stop = 2;
    auto res_warm = runner.run(cfg);

    // Cold reference: fresh manager, identical stored dummies.
    wsm::WarmStartManager cold_mgr(K, wsm::WarmStartMode::IN_MEMORY, false);
    er::ExperimentRunner cold_runner(X_map, d.y, dummy_gen, cold_mgr, nullptr);
    auto cold_cfg = cfg;
    cold_cfg.use_warm_start = false;
    auto res_cold = cold_runner.run(cold_cfg);

    ASSERT_EQ(res_warm.phi_T_mat.rows(), res_cold.phi_T_mat.rows());
    ASSERT_EQ(res_warm.phi_T_mat.cols(), res_cold.phi_T_mat.cols());
    EXPECT_LE((res_warm.phi_T_mat - res_cold.phi_T_mat).cwiseAbs().maxCoeff(),
              1e-12)
        << "Warm-continued phi_T_mat differs from the cold reference run.";

    // invalidate() must keep the K slots usable (regression: clear() shrank
    // the vectors to zero and hasSolver() stayed false forever).
    warm_mgr.invalidate();
    for (std::size_t k = 0; k < K; ++k) {
        EXPECT_FALSE(warm_mgr.hasSolver(k));
    }
    runner.run(cfg);
    for (std::size_t k = 0; k < K; ++k) {
        EXPECT_TRUE(warm_mgr.hasSolver(k))
            << "Slot " << k << " could not be repopulated after invalidate().";
    }
}


/** @brief DIRECT strategy: warm continuation matches a cold run — exercises
 *         the co-retained dummy buffers (the retained solver views a per-call
 *         temporary D_k) and the deterministic re-derivation of dummies. */
TEST(TRexCoreTest, WarmStart_DirectStrategyContinuationMatchesCold) {
    const Eigen::Index n = 60, p = 20;
    const std::size_t K = 3;
    RunnerTestData d(n, p);
    Eigen::Map<Eigen::MatrixXd> X_map(d.X.data(), d.X.rows(), d.X.cols());

    dg::DummyGenerator dummy_gen(static_cast<std::size_t>(n),
                                 dummygen::Distribution::Normal(), 7, false);
    wsm::WarmStartManager warm_mgr(K, wsm::WarmStartMode::IN_MEMORY, false);
    er::ExperimentRunner runner(X_map, d.y, dummy_gen, warm_mgr, nullptr);

    auto cfg = make_runner_cfg(K, static_cast<std::size_t>(p),
                               er::ExperimentStrategy::Seeded);

    runner.run(cfg);           // T = 1, retains solvers + dummy buffers
    cfg.T_stop = 2;
    auto res_warm = runner.run(cfg);   // warm continuation

    // Cold reference with a separate, equally seeded generator: DIRECT
    // dummies must be reproducible across instances for the same seed.
    dg::DummyGenerator cold_gen(static_cast<std::size_t>(n),
                                dummygen::Distribution::Normal(), 7, false);
    wsm::WarmStartManager cold_mgr(K, wsm::WarmStartMode::IN_MEMORY, false);
    er::ExperimentRunner cold_runner(X_map, d.y, cold_gen, cold_mgr, nullptr);
    auto cold_cfg = cfg;
    cold_cfg.use_warm_start = false;
    auto res_cold = cold_runner.run(cold_cfg);

    ASSERT_EQ(res_warm.phi_T_mat.rows(), res_cold.phi_T_mat.rows());
    ASSERT_EQ(res_warm.phi_T_mat.cols(), res_cold.phi_T_mat.cols());
    EXPECT_LE((res_warm.phi_T_mat - res_cold.phi_T_mat).cwiseAbs().maxCoeff(),
              1e-12)
        << "DIRECT warm continuation differs from the cold reference run.";
}


/** @brief Memory-mapped runs use SERIALIZED warm starts (save/load per
 *         T-step). With identical data and seed they must reproduce the
 *         in-memory (retention-based) run exactly. */
TEST(TRexCoreTest, WarmStart_MemoryMappedMatchesInMemory) {
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(200, 50, support, coefs, 1.0, 42);

    auto X_mem = data.getX();
    auto y_mem = data.getY();
    auto X_map_data = X_mem;   // independent copies: X is normalized in-place
    auto y_map_data = y_mem;

    TRexControlParameter ctrl;
    ctrl.K = 3;
    ctrl.max_dummy_multiplier = 2;

    // tFDR = 0.2: with K = 3 and few strong actives the FDP estimate cannot
    // drop below ~0.15, so a tighter target (e.g. 0.1) would end the T-loop
    // at T_stop = 1 — leaving the warm-start continuation unexercised
    // (the EXPECT_GT below guards this).
    const double tFDR = 0.2;

    Eigen::Map<Eigen::MatrixXd> X1(X_mem.data(), X_mem.rows(), X_mem.cols());
    Eigen::Map<Eigen::VectorXd> y1(y_mem.data(), y_mem.size());
    TRexSelector trex_in_memory(X1, y1, tFDR, ctrl, 42, false);
    auto res_in_memory = trex_in_memory.select();

    ctrl.use_memory_mapping = true;
    Eigen::Map<Eigen::MatrixXd> X2(X_map_data.data(),
                                   X_map_data.rows(), X_map_data.cols());
    Eigen::Map<Eigen::VectorXd> y2(y_map_data.data(), y_map_data.size());
    TRexSelector trex_memmap(X2, y2, tFDR, ctrl, 42, false);
    auto res_memmap = trex_memmap.select();

    EXPECT_EQ(res_in_memory.T_stop, res_memmap.T_stop);
    // The comparison is only meaningful if the T-loop actually continued
    // (T_stop > 1), i.e. the serialized load-and-continue cycle fired.
    EXPECT_GT(res_in_memory.T_stop, 1u)
        << "Test data no longer drives the T-loop past T = 1; "
           "the warm-start continuation path is not being exercised.";
    ASSERT_EQ(res_in_memory.selected_var.size(), res_memmap.selected_var.size());
    EXPECT_TRUE((res_in_memory.selected_var.array()
                 == res_memmap.selected_var.array()).all())
        << "Memory-mapped selection differs from the in-memory selection.";
    if (res_in_memory.Phi_mat.size() > 0 &&
        res_in_memory.Phi_mat.rows() == res_memmap.Phi_mat.rows()) {
        EXPECT_LE((res_in_memory.Phi_mat - res_memmap.Phi_mat)
                      .cwiseAbs().maxCoeff(), 1e-9);
    }
}


/** @brief CD-family serialized warm start at the TRex level: memory-mapped
 *         runs save/load TCDLASSO / TCDENET checkpoints per T-step, so the
 *         continuation goes through ensureCdState()'s gram re-entry after
 *         deserialization. The selection must reproduce the in-memory
 *         (retention-based) run exactly. */
TEST(TRexCoreTest, WarmStart_MemoryMappedMatchesInMemory_CdFamily) {
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};

    for (auto solver : {disp::SolverTypeForTRex::TCDLASSO,
                        disp::SolverTypeForTRex::TCDENET}) {
        SyntheticData data(200, 50, support, coefs, 1.0, 42);

        auto X_mem = data.getX();
        auto y_mem = data.getY();
        auto X_map_data = X_mem;   // independent copies: X is normalized in-place
        auto y_map_data = y_mem;

        TRexControlParameter ctrl;
        ctrl.K = 3;
        ctrl.max_dummy_multiplier = 2;
        ctrl.solver_type = solver;
        const double tFDR = 0.2;   // keeps the T-loop past T = 1 (see above)

        Eigen::Map<Eigen::MatrixXd> X1(X_mem.data(), X_mem.rows(), X_mem.cols());
        Eigen::Map<Eigen::VectorXd> y1(y_mem.data(), y_mem.size());
        TRexSelector trex_in_memory(X1, y1, tFDR, ctrl, 42, false);
        auto res_in_memory = trex_in_memory.select();

        ctrl.use_memory_mapping = true;
        Eigen::Map<Eigen::MatrixXd> X2(X_map_data.data(),
                                       X_map_data.rows(), X_map_data.cols());
        Eigen::Map<Eigen::VectorXd> y2(y_map_data.data(), y_map_data.size());
        TRexSelector trex_memmap(X2, y2, tFDR, ctrl, 42, false);
        auto res_memmap = trex_memmap.select();

        EXPECT_EQ(res_in_memory.T_stop, res_memmap.T_stop);
        EXPECT_GT(res_in_memory.T_stop, 1u)
            << "CD solver run ended at T = 1; the serialized "
               "load-and-continue cycle is not being exercised.";
        ASSERT_EQ(res_in_memory.selected_var.size(),
                  res_memmap.selected_var.size());
        EXPECT_TRUE((res_in_memory.selected_var.array()
                     == res_memmap.selected_var.array()).all())
            << "Memory-mapped CD selection differs from the in-memory one.";
        if (res_in_memory.Phi_mat.size() > 0 &&
            res_in_memory.Phi_mat.rows() == res_memmap.Phi_mat.rows()) {
            EXPECT_LE((res_in_memory.Phi_mat - res_memmap.Phi_mat)
                          .cwiseAbs().maxCoeff(), 1e-9);
        }
    }
}


// ========================================================================================
// DIRECT dummy seeding (regression: the user seed was ignored — dummies were
// derived from the PERMUTATION-only base seed, which is 0 for DIRECT — and
// unseeded runs drew fresh entropy on every call, changing the dummies
// between T-steps)
// ========================================================================================

/** @brief DIRECT dummies honour the user seed, are reproducible across
 *         equally seeded instances, and are stable within one run even when
 *         unseeded. */
TEST(TRexCoreTest, DirectDummies_HonourUserSeedAndAreStableWithinRun) {
    const std::size_t n = 50, num_dummies = 30;

    dg::DummyGenerator g_seed1(n, dummygen::Distribution::Normal(), 1, false);
    dg::DummyGenerator g_seed2(n, dummygen::Distribution::Normal(), 2, false);
    dg::DummyGenerator g_seed1_again(n, dummygen::Distribution::Normal(), 1, false);

    Eigen::MatrixXd D1  = g_seed1.generateSeeded(num_dummies, 0, /*l_tag=*/1);
    Eigen::MatrixXd D2  = g_seed2.generateSeeded(num_dummies, 0, /*l_tag=*/1);
    Eigen::MatrixXd D1b = g_seed1_again.generateSeeded(num_dummies, 0, /*l_tag=*/1);

    EXPECT_FALSE(D1.isApprox(D2, 1e-12))
        << "Different user seeds produced identical DIRECT dummies.";
    EXPECT_TRUE(D1.isApprox(D1b, 1e-12))
        << "Equal user seeds did not reproduce the same DIRECT dummies.";

    // Unseeded (seed < 0): the base is resolved once per generator, so
    // repeated derivation of the same experiment's dummies must be identical
    // (the T-loop re-derives D_k at every step).
    dg::DummyGenerator g_random(n, dummygen::Distribution::Normal(), -1, false);
    Eigen::MatrixXd Da = g_random.generateSeeded(num_dummies, 3, /*l_tag=*/1);
    Eigen::MatrixXd Db = g_random.generateSeeded(num_dummies, 3, /*l_tag=*/1);
    EXPECT_TRUE(Da.isApprox(Db, 1e-12))
        << "Unseeded DIRECT dummies changed between calls within one run.";
}

// ========================================================================================
// Shared-buffer claim guard
// ========================================================================================

/** @brief Two selectors on the SAME Eigen::Map must not coexist: the second
 *  construction throws, because both would normalize/restore the one buffer in
 *  place and silently corrupt each other. */
TEST(TRexCoreTest, SharedBufferClaim_ThrowsOnSecondSelector) {
    SyntheticData data(60, 15, {1, 2, 3}, {5.0, -3.0, 2.0}, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = 3;
    ctrl.max_dummy_multiplier = 2;

    TRexSelector first(X_map, y_map, 0.1, ctrl, 42, false);
    // Second selector on the SAME buffer while `first` still holds it normalized.
    EXPECT_THROW(
        { TRexSelector second(X_map, y_map, 0.1, ctrl, 42, false); },
        std::runtime_error);
}

/** @brief After a selector finishes select() (which restores X and releases the
 *  claim) a new selector on the same buffer is allowed — mirrors the biobank
 *  screen->fallback pattern. */
TEST(TRexCoreTest, SharedBufferClaim_ReleasedAfterSelect) {
    SyntheticData data(60, 15, {1, 2, 3}, {5.0, -3.0, 2.0}, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = 3;
    ctrl.max_dummy_multiplier = 2;

    {
        TRexSelector first(X_map, y_map, 0.1, ctrl, 42, false);
        first.select();   // restores X, releases the claim
        // Second selector on the same buffer is now permitted.
        EXPECT_NO_THROW(
            { TRexSelector second(X_map, y_map, 0.1, ctrl, 42, false); });
    }
}

/** @brief Selectors on DISTINCT buffers never conflict. */
TEST(TRexCoreTest, SharedBufferClaim_DistinctBuffersOk) {
    SyntheticData data(60, 15, {1, 2, 3}, {5.0, -3.0, 2.0}, 1.0, 42);
    auto Xa = data.getX();
    auto Xb = data.getX();   // independent copy
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> Xam(Xa.data(), Xa.rows(), Xa.cols());
    Eigen::Map<Eigen::MatrixXd> Xbm(Xb.data(), Xb.rows(), Xb.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = 3;
    ctrl.max_dummy_multiplier = 2;

    TRexSelector s1(Xam, y_map, 0.1, ctrl, 42, false);
    EXPECT_NO_THROW(
        { TRexSelector s2(Xbm, y_map, 0.1, ctrl, 42, false); });
}

// ========================================================================================
// FDP estimator: Phi_prime keeps negative values
// ========================================================================================

/** @brief Exposes the protected FDP estimator of the selector. */
class FdpEstimatorProbe : public TRexSelector {
public:
    using TRexSelector::TRexSelector;
    using TRexSelector::computePhiPrime;
    using TRexSelector::computeFDPHat;
};


/** @brief SKIPL with L >= 10 warns only for a path solver running without the stagnation stop. */
TEST(TRexCoreTest, SkiplLargeBudgetWithoutStopWarns) {
    SyntheticData data(20, 10, {0, 1, 2}, {5.0, -3.0, 2.0}, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    const auto warns = [&](LLoopStrategy strategy, std::size_t multiplier, std::optional<bool> stop) {
        TRexControlParameter ctrl;
        ctrl.K = 5;
        ctrl.solver_type = disp::SolverTypeForTRex::TLARS;
        ctrl.lloop_strategy = strategy;
        ctrl.max_dummy_multiplier = multiplier;
        ctrl.tloop_stagnation_stop = stop;
        testing::internal::CaptureStderr();
        TRexSelector trex(X_map, y_map, 0.1, ctrl, 42, false);
        return testing::internal::GetCapturedStderr().find("lloop_strategy SKIPL") != std::string::npos;
    };

    EXPECT_TRUE(warns(LLoopStrategy::SKIPL, 10, std::nullopt));
    EXPECT_TRUE(warns(LLoopStrategy::SKIPL, 20, false));
    EXPECT_FALSE(warns(LLoopStrategy::SKIPL, 5, std::nullopt));
    EXPECT_FALSE(warns(LLoopStrategy::SKIPL, 20, true));
    EXPECT_FALSE(warns(LLoopStrategy::STANDARD, 20, std::nullopt));
}

/** @brief A step in which the variables above Phi = 0.5 gain less mass than the
 *         expected number of null entries gives a negative Phi_prime. It stays
 *         negative, as in the T-Rex paper, so the variable carries more than one
 *         expected false discovery (a clamp at 0 lowered FDP_hat).
 */
TEST(TRexCoreTest, PhiPrimeKeepsNegativeValues) {
    SyntheticData data(20, 10, {0, 1, 2}, {5.0, -3.0, 2.0}, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());
    TRexControlParameter ctrl;
    ctrl.K = 5;
    FdpEstimatorProbe trex(X_map, y_map, 0.1, ctrl, 42, false);

    // p x T occurrences: variables 0-2 in every experiment from T = 1, variable 3
    // joins at T = 2 in 60 % of the experiments, variables 4-9 in 10 %
    Eigen::MatrixXd phi_T = Eigen::MatrixXd::Zero(10, 2);
    phi_T.topRows(3).setOnes();
    phi_T(3, 1) = 0.6;
    phi_T.block(4, 1, 6, 1).setConstant(0.1);
    const Eigen::VectorXd Phi = phi_T.col(1);

    const Eigen::VectorXd phi_prime = trex.computePhiPrime(phi_T, Phi, /*num_dummies=*/10);

    // scale_t = 1 - ((p - sum Phi_t) / (num_dummies - (t - 1))) / delta_t
    const double scale1 = 1.0 - ((10.0 - 3.0) / 10.0) / 3.0;
    const double scale2 = 1.0 - ((10.0 - 4.2) / 9.0) / 0.6;
    EXPECT_NEAR(phi_prime(0), scale1, 1e-12);
    EXPECT_NEAR(phi_prime(3), 0.6 * scale2, 1e-12);
    EXPECT_LT(phi_prime(3), 0.0);

    // At v = 0.5 variables 0-3 are selected, and variable 3 counts 1 - Phi' > 1
    Eigen::VectorXd grid(1);
    grid << 0.5;
    const double expected = (3.0 * (1.0 - scale1) + (1.0 - 0.6 * scale2)) / 4.0;
    EXPECT_NEAR(trex.computeFDPHat(grid, Phi, phi_prime)(0), expected, 1e-12);
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::ad::trex_core */
