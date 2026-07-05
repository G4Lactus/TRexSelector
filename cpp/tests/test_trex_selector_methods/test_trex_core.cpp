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
#include <trex_selector_methods/trex_core/trex.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_core {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;

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
                        sd::SolverTypeForTRex solver = sd::SolverTypeForTRex::TLARS,
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
        omp_set_num_threads(2);
        trex_ctrl.parallel_rnd_experiments = true;
        trex_ctrl.autoConfigResources();
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

    // Assert bounds on Phi_prime
    for (Eigen::Index j = 0; j < p; ++j) {
        EXPECT_GE(result.Phi_prime(j), 0.0);
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
    run_trex_core_test(150, 200, sd::SolverTypeForTRex::TLASSO);
}


/** @brief Test TRexSelector with SKIPL L loop strategy. */
TEST(TRexCoreTest, Param_LStrategies_SKIPL) {
    run_trex_core_test(150, 200, sd::SolverTypeForTRex::TLARS,
                       LLoopStrategy::SKIPL);
}


/** @brief Test TRexSelector with concurrency enabled. */
TEST(TRexCoreTest, Param_Concurrency) {
    // Tests autoConfigResources logic and threaded OpenMP execution
    run_trex_core_test(150, 200, sd::SolverTypeForTRex::TLARS,
                       LLoopStrategy::HCONCAT, true);
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

    auto resolved_default = [&](sd::SolverTypeForTRex solver,
                                std::optional<bool> user_choice) {
        TRexControlParameter ctrl;
        ctrl.K = 3;
        ctrl.solver_type = solver;
        ctrl.tloop_stagnation_stop = user_choice;
        TRexSelector trex(X_map, y_map, 0.1, ctrl, 42, false);
        return trex.getStagnationCheck();
    };

    // AUTO: off for the equiangular LARS family, on for greedy solvers.
    EXPECT_FALSE(resolved_default(sd::SolverTypeForTRex::TLARS,  std::nullopt));
    EXPECT_FALSE(resolved_default(sd::SolverTypeForTRex::TLASSO, std::nullopt));
    EXPECT_FALSE(resolved_default(sd::SolverTypeForTRex::TENET,  std::nullopt));
    EXPECT_TRUE(resolved_default(sd::SolverTypeForTRex::TOMP,    std::nullopt));
    EXPECT_TRUE(resolved_default(sd::SolverTypeForTRex::TSTEPWISE, std::nullopt));
    EXPECT_TRUE(resolved_default(sd::SolverTypeForTRex::TAFS,    std::nullopt));

    // Explicit user choice overrides the family default in both directions.
    EXPECT_TRUE(resolved_default(sd::SolverTypeForTRex::TLARS, true));
    EXPECT_FALSE(resolved_default(sd::SolverTypeForTRex::TOMP, false));
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
                               er::ExperimentStrategy::Direct);

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

    Eigen::MatrixXd D1  = g_seed1.generateDirect(num_dummies, 0);
    Eigen::MatrixXd D2  = g_seed2.generateDirect(num_dummies, 0);
    Eigen::MatrixXd D1b = g_seed1_again.generateDirect(num_dummies, 0);

    EXPECT_FALSE(D1.isApprox(D2, 1e-12))
        << "Different user seeds produced identical DIRECT dummies.";
    EXPECT_TRUE(D1.isApprox(D1b, 1e-12))
        << "Equal user seeds did not reproduce the same DIRECT dummies.";

    // Unseeded (seed < 0): the base is resolved once per generator, so
    // repeated derivation of the same experiment's dummies must be identical
    // (the T-loop re-derives D_k at every step).
    dg::DummyGenerator g_random(n, dummygen::Distribution::Normal(), -1, false);
    Eigen::MatrixXd Da = g_random.generateDirect(num_dummies, 3);
    Eigen::MatrixXd Db = g_random.generateDirect(num_dummies, 3);
    EXPECT_TRUE(Da.isApprox(Db, 1e-12))
        << "Unseeded DIRECT dummies changed between calls within one run.";
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_core */
