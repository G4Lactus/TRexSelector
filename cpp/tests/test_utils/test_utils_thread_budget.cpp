// ========================================================================================
// test_utils_thread_budget.cpp
// ========================================================================================
/**
 * @file test_utils_thread_budget.cpp
 *
 * @brief The library's thread budget: single-threaded by default on every
 *        machine, explicit budgets honoured, the OpenMP runtime's own team
 *        never used.
 *
 * @details The team probe of utils_openmp.hpp records the largest team any
 *  budgeted region ran on. A default call (the generators, the scalers, the
 *  clustering, a selector) must record one thread even when the OpenMP
 *  runtime's default team is raised; a budget of four on a design above the
 *  work threshold must form a team. The threshold rule and the Eigen guard
 *  are pinned directly.
 */
// ========================================================================================

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp>
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/datageneration/utils_dummygen.hpp>
#include <utils/openmp/utils_eigen_threads.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ========================================================================================

namespace trex::test::utils::thread_budget {

namespace omp = trex::utils::openmp;
namespace agg = trex::ml_methods::clustering::hierarchical::agglomerative;
using trex::utils::datageneration::datagen::SyntheticData;
using trex::ml_methods::scaler_methods::ZScoreScaler;
using namespace trex::trex_selector_methods::ad::trex_core;

namespace {

// A design above the work threshold (n * p >= 1e6), so a budget above 1 forms a team
constexpr Eigen::Index kN = 1000;
constexpr Eigen::Index kP = 1200;

const std::vector<std::size_t> kSupport = {1, 5, 9};
const std::vector<double> kCoefs = {1.0, 1.0, 1.0};

// The probe records teams while an instance lives
struct ProbeScope {
    ProbeScope() { omp::team_probe::reset(); omp::team_probe::enable(true); }
    ~ProbeScope() { omp::team_probe::enable(false); }
};

// The OpenMP runtime's own default team, raised: the library must not follow it
struct RuntimeTeamScope {
    int previous;
    RuntimeTeamScope() : previous(omp_get_max_threads()) { omp_set_num_threads(4); }
    ~RuntimeTeamScope() { omp_set_num_threads(previous); }
};

bool canFormTeams() {
    return omp::is_available() && omp_get_num_procs() >= 2;
}

SyntheticData makeData(int n_threads = omp::get_num_threads()) {
    namespace dg = trex::utils::datageneration;
    return SyntheticData(kN, kP, kSupport, kCoefs, 1.0, 7,
                         dg::datagen::predictor_policy::Normal(),
                         dg::noisegen::noise_policy::Normal(), n_threads);
}

TRexControlParameter smallControl(int max_inner_threads = 1) {
    TRexControlParameter ctrl;
    ctrl.K = 2;
    ctrl.max_dummy_multiplier = 1;
    ctrl.max_inner_threads = max_inner_threads;
    return ctrl;
}

}  // namespace

// ========================================================================================

TEST(ThreadBudget, DefaultBudgetIsOneAndTheSetterValidates) {
    EXPECT_EQ(omp::get_num_threads(), 1);
    EXPECT_THROW(omp::set_num_threads(0), std::invalid_argument);
    EXPECT_THROW(omp::set_num_threads(-3), std::invalid_argument);
    EXPECT_EQ(omp::set_num_threads(3), 1);
    EXPECT_EQ(omp::get_num_threads(), 3);
    EXPECT_EQ(omp::set_num_threads(1), 3);
    EXPECT_EQ(omp::get_num_threads(), 1);
    EXPECT_GE(omp::get_max_threads(), 1);
}

TEST(ThreadBudget, ThreadsForFollowsTheBudgetAndTheThreshold) {
    EXPECT_EQ(omp::threads_for(1, 1e12), 1);
    EXPECT_EQ(omp::threads_for(0, 1e12), 1);
    EXPECT_EQ(omp::threads_for(4, omp::kParallelWorkThreshold - 1.0), 1);
    EXPECT_EQ(omp::threads_for(4, omp::kParallelWorkThreshold), 4);
}

// Eigen's thread count is inert under a BLAS backend (EIGEN_USE_BLAS), so the
// guard's effect is observable only in the builds without one.
#if !defined(EIGEN_USE_BLAS)
TEST(ThreadBudget, EigenGuardSetsAndRestores) {
    const int before = Eigen::nbThreads();
    {
        omp::EigenThreadBudgetGuard guard(3);
        EXPECT_EQ(Eigen::nbThreads(), 3);
    }
    EXPECT_EQ(Eigen::nbThreads(), before);
}
#endif

TEST(ThreadBudget, GeneratorAndScalerDefaultToOneThread) {
    RuntimeTeamScope runtime;
    ProbeScope probe;
    SyntheticData data = makeData();
    Eigen::MatrixXd X = data.getX();
    ZScoreScaler scaler;
    scaler.fit(X);
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    scaler.transform_inplace(X_map);
    EXPECT_LE(omp::team_probe::max_team(), 1);
}

TEST(ThreadBudget, DefaultSelectionStaysOnOneThread) {
    RuntimeTeamScope runtime;
    SyntheticData data = makeData();
    Eigen::MatrixXd X = data.getX();
    Eigen::VectorXd y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    ProbeScope probe;
    TRexSelector sel(X_map, y_map, 0.2, smallControl(), 3, /*verbose=*/false);
    (void)sel.select();
    EXPECT_LE(omp::team_probe::max_team(), 1);
}

TEST(ThreadBudget, ExplicitBudgetFormsATeam) {
    if (!canFormTeams()) { GTEST_SKIP() << "one core or no OpenMP"; }

    ProbeScope probe;
    SyntheticData data = makeData(4);
    EXPECT_GE(omp::team_probe::max_team(), 2) << "the generator's budget";

    omp::team_probe::reset();
    Eigen::MatrixXd X = data.getX();
    ZScoreScaler scaler(true, true, 4);
    scaler.fit(X);
    EXPECT_GE(omp::team_probe::max_team(), 2) << "the scaler's budget";

    omp::team_probe::reset();
    Eigen::VectorXd y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());
    TRexSelector sel(X_map, y_map, 0.2, smallControl(4), 3, /*verbose=*/false);
    (void)sel.select();
    EXPECT_GE(omp::team_probe::max_team(), 2) << "the selector's max_inner_threads";
}

TEST(ThreadBudget, LibraryDefaultBudgetReachesTheKernels) {
    if (!canFormTeams()) { GTEST_SKIP() << "one core or no OpenMP"; }

    const int previous = omp::set_num_threads(4);
    {
        ProbeScope probe;
        SyntheticData data = makeData();   // the library's default budget, now 4
        EXPECT_GE(omp::team_probe::max_team(), 2);
    }
    EXPECT_EQ(omp::set_num_threads(previous), 4);
    EXPECT_EQ(omp::get_num_threads(), 1);
}

TEST(ThreadBudget, SeededLoopWorkCountsDrawsAndEngines) {
    namespace dg = trex::utils::datageneration::dummygen;
    // 5,000 columns of 150 draws: 7.5e5 entries, far above the threshold once counted as work
    EXPECT_GE(dg::seeded_loop_work(5000, 150, dg::kSeedSeqEngineWork), omp::kParallelWorkThreshold);
    // 20 columns of 50 draws stay below it
    EXPECT_LT(dg::seeded_loop_work(20, 50, dg::kSeedSeqEngineWork), omp::kParallelWorkThreshold);
    EXPECT_GT(dg::seeded_loop_work(10, 10, dg::kSeedSeqEngineWork),
              dg::seeded_loop_work(10, 10, dg::kWordSeededEngineWork));
}

TEST(ThreadBudget, SeedingDominatedGeneratorFormsATeam) {
    if (!canFormTeams()) { GTEST_SKIP() << "one core or no OpenMP"; }
    namespace dg = trex::utils::datageneration::dummygen;
    namespace dgen = trex::utils::datageneration::datagen;

    // 150 x 5,000: below the entry threshold, far above it with the draws and seedings counted
    const std::size_t n = 150;
    const std::size_t p = 5000;
    Eigen::MatrixXd D(n, p);
    {
        ProbeScope probe;
        dg::generate_dummies(D, n, p, 11u, dg::Distribution::Normal(), 0, 4);
        EXPECT_GE(omp::team_probe::max_team(), 2) << "the dummy columns";
    }
    {
        ProbeScope probe;
        dgen::detail::generate_predictor_matrix(D, n, p, 11u, dgen::predictor_policy::AR1(0.5), 4);
        EXPECT_GE(omp::team_probe::max_team(), 2) << "the AR(1) rows";
    }
    {
        // The scaler's plain passes over the same 7.5e5 entries stay serial
        ProbeScope probe;
        ZScoreScaler scaler(true, true, 4);
        scaler.fit(D);
        EXPECT_LE(omp::team_probe::max_team(), 1) << "the scaler's passes";
    }
}

TEST(ThreadBudget, ClusteringDefaultsToOneThreadAndTakesABudget) {
    RuntimeTeamScope runtime;
    // 250 objects of 6,000 entries: the norm precomputation touches 1.5e6 entries
    const Eigen::Index rows = 6000;
    const Eigen::Index objs = 250;
    Eigen::MatrixXd data = Eigen::MatrixXd::Random(rows, objs);
    using Policy = agg::DistancePolicy<Eigen::MatrixXd, agg::DistanceMetric::Euclidean>;
    {
        ProbeScope probe;
        (void)agg::AgglomerativeClustering::cluster<Eigen::MatrixXd, Policy,
                                                   agg::LinkageMethod::Average>(
            data, /*use_mmap=*/false, /*verbose=*/false);
        EXPECT_LE(omp::team_probe::max_team(), 1);
    }
    if (!canFormTeams()) { GTEST_SKIP() << "one core or no OpenMP"; }
    {
        ProbeScope probe;
        (void)agg::AgglomerativeClustering::cluster<Eigen::MatrixXd, Policy,
                                                   agg::LinkageMethod::Average>(
            data, /*use_mmap=*/false, /*verbose=*/false, /*n_threads=*/4);
        EXPECT_GE(omp::team_probe::max_team(), 2);
    }
}

}  // namespace trex::test::utils::thread_budget
