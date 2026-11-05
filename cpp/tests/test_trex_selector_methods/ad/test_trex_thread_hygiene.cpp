// ========================================================================================
/**
 * @file test_trex_thread_hygiene.cpp
 *
 * @brief The process-global thread settings must be exactly what they were
 *  before a selection once select() returns, on every runner path: the
 *  library sets its budget once from the calling thread,
 *  never inside an OpenMP team, and restores it. A leak here changes the
 *  threading of the host process (an R or Python session) behind its back.
 *
 *  Two globals are watched:
 *   - OpenMP's nthreads-var of the calling thread (`omp_get_max_threads()`).
 *     The parallel runners call `omp_set_num_threads()` INSIDE their team
 *     (a per-thread ICV for the nested solver regions), which must not reach
 *     the host thread. Live in every OpenMP build.
 *   - Eigen's global thread count (`Eigen::nbThreads()`). Live only when
 *     Eigen manages its own products: with EIGEN_USE_BLAS (or without
 *     OpenMP) Eigen's Parallelizer.h stubs setNbThreads()/nbThreads() to a
 *     no-op returning 1, so the leak cannot exist there and the tests are
 *     compiled out (CI's cpp-tests job, the R package and the Python wheel
 *     are no-BLAS builds where they run; locally cpp/build/release-noblas).
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <vector>

// project includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ========================================================================================

namespace trex::test::trex_selector_methods::ad::trex_core {

using namespace trex::trex_selector_methods::ad::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace disp = trex::trex_selector_methods::ad::utils::solver_dispatch;
namespace gvs = trex::trex_selector_methods::ad::trex_gvs;

namespace {

/** @brief A sentinel no code path would choose on its own, restored afterwards. */
constexpr int kSentinelThreads = 3;

void runCore(LLoopStrategy strategy, bool parallel) {
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(60, 20, support, coefs, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = 3;
    ctrl.max_dummy_multiplier = 2;
    ctrl.lloop_strategy = strategy;
    if (parallel) {
        ctrl.parallel_rnd_experiments = true;
        ctrl.max_outer_threads = 2;
        ctrl.max_inner_threads = 1;
    }
    TRexSelector trex(X_map, y_map, 0.1, ctrl, 42, false);
    trex.select();
}

void runGvs(bool parallel) {
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(60, 20, support, coefs, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    gvs::TRexGVSControlParameter ctrl;
    ctrl.gvs_type = gvs::GVSType::EN;
    ctrl.lambda_2 = 1.0;
    ctrl.trex_ctrl.K = 3;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;
    if (parallel) {
        ctrl.trex_ctrl.parallel_rnd_experiments = true;
        ctrl.trex_ctrl.max_outer_threads = 2;
        ctrl.trex_ctrl.max_inner_threads = 1;
    }
    gvs::TRexGVSSelector sel(X_map, y_map, 0.1, ctrl, 42, false);
    sel.select();
}

constexpr LLoopStrategy kSequentialStrategies[] = {
    LLoopStrategy::HCONCAT, LLoopStrategy::STANDARD, LLoopStrategy::SKIPL,
    LLoopStrategy::SEEDED, LLoopStrategy::PERMUTATION, LLoopStrategy::PERMUTATION_SEEDED};

constexpr LLoopStrategy kParallelStrategies[] = {
    LLoopStrategy::HCONCAT, LLoopStrategy::STANDARD};

}  // namespace

// ========================================================================================
// OpenMP's nthreads-var of the host thread — live in every OpenMP build
// ========================================================================================

namespace {

/** @brief Pin the host thread's OpenMP thread count to the sentinel; restored. */
struct OmpSentinelScope {
    int previous;
    OmpSentinelScope() : previous(omp_get_max_threads()) { omp_set_num_threads(kSentinelThreads); }
    ~OmpSentinelScope() { omp_set_num_threads(previous); }
};

#ifdef _OPENMP
constexpr bool kOpenMpEnabled = true;
#else
constexpr bool kOpenMpEnabled = false;   // the shims in utils_openmp.hpp return 1
#endif

#define TREX_SKIP_WITHOUT_OPENMP()                                                    \
    if (!kOpenMpEnabled) {                                                            \
        GTEST_SKIP() << "OpenMP is not enabled in this build.";                       \
    }

}  // namespace

/** @brief The sequential runners never touch OpenMP's global. */
TEST(TRexThreadHygieneTest, SequentialRunnersRestoreOpenMpMaxThreads) {
    TREX_SKIP_WITHOUT_OPENMP();
    OmpSentinelScope sentinel;
    for (const LLoopStrategy s : kSequentialStrategies) {
        runCore(s, /*parallel=*/false);
        EXPECT_EQ(omp_get_max_threads(), kSentinelThreads)
            << "strategy " << static_cast<int>(s) << " leaked the OpenMP thread count";
    }
}


/** @brief The parallel-experiments runner sets the nested budget per worker
 *  inside its team; the host thread's value must survive. */
TEST(TRexThreadHygieneTest, ParallelRunnerRestoresOpenMpMaxThreads) {
    TREX_SKIP_WITHOUT_OPENMP();
    OmpSentinelScope sentinel;
    for (const LLoopStrategy s : kParallelStrategies) {
        runCore(s, /*parallel=*/true);
        EXPECT_EQ(omp_get_max_threads(), kSentinelThreads)
            << "strategy " << static_cast<int>(s) << " leaked the OpenMP thread count";
    }
}


/** @brief The GVS in-memory K-loop, which runs its own parallel region. */
TEST(TRexThreadHygieneTest, GvsRunnerRestoresOpenMpMaxThreads) {
    TREX_SKIP_WITHOUT_OPENMP();
    OmpSentinelScope sentinel;
    for (const bool parallel : {false, true}) {
        runGvs(parallel);
        EXPECT_EQ(omp_get_max_threads(), kSentinelThreads)
            << (parallel ? "parallel" : "serial") << " GVS run leaked the OpenMP thread count";
    }
}


// ========================================================================================
// Eigen's global thread count — live only when Eigen manages its own products
// ========================================================================================

#ifndef EIGEN_USE_BLAS

namespace {

struct EigenSentinelScope {
    int previous;
    EigenSentinelScope() : previous(Eigen::nbThreads()) { Eigen::setNbThreads(kSentinelThreads); }
    ~EigenSentinelScope() { Eigen::setNbThreads(previous); }
};

/**
 * @brief Whether Eigen's global thread count is live in this build.
 *
 * @details Compiled without EIGEN_USE_BLAS, but Eigen also stubs the count
 *  when it has no OpenMP (Parallelizer.h), so probe at runtime as a fallback.
 */
bool eigenThreadCountIsLive() {
    const int previous = Eigen::nbThreads();
    Eigen::setNbThreads(kSentinelThreads);
    const bool live = (Eigen::nbThreads() == kSentinelThreads);
    Eigen::setNbThreads(previous);
    return live;
}

#define TREX_SKIP_UNLESS_EIGEN_THREADS_LIVE()                                        \
    if (!eigenThreadCountIsLive()) {                                                  \
        GTEST_SKIP() << "Eigen's thread count is inert in this build (no OpenMP); "  \
                        "run an OpenMP-enabled no-BLAS configuration.";              \
    }

}  // namespace

/** @brief Every sequential runner (stored, on-demand and permutation
 *  strategies) leaves Eigen's thread count as it found it. */
TEST(TRexThreadHygieneTest, SequentialRunnersRestoreEigenThreadCount) {
    TREX_SKIP_UNLESS_EIGEN_THREADS_LIVE();
    EigenSentinelScope sentinel;
    for (const LLoopStrategy s : kSequentialStrategies) {
        runCore(s, /*parallel=*/false);
        EXPECT_EQ(Eigen::nbThreads(), kSentinelThreads)
            << "strategy " << static_cast<int>(s) << " leaked the Eigen thread count";
    }
}


/** @brief The parallel-experiments runner sets its budget once and restores it. */
TEST(TRexThreadHygieneTest, ParallelRunnerRestoresEigenThreadCount) {
    TREX_SKIP_UNLESS_EIGEN_THREADS_LIVE();
    EigenSentinelScope sentinel;
    for (const LLoopStrategy s : kParallelStrategies) {
        runCore(s, /*parallel=*/true);
        EXPECT_EQ(Eigen::nbThreads(), kSentinelThreads)
            << "strategy " << static_cast<int>(s) << " leaked the Eigen thread count";
    }
}


/** @brief The GVS in-memory K-loop, which runs its own parallel region. */
TEST(TRexThreadHygieneTest, GvsRunnerRestoresEigenThreadCount) {
    TREX_SKIP_UNLESS_EIGEN_THREADS_LIVE();
    EigenSentinelScope sentinel;
    for (const bool parallel : {false, true}) {
        runGvs(parallel);
        EXPECT_EQ(Eigen::nbThreads(), kSentinelThreads)
            << (parallel ? "parallel" : "serial") << " GVS run leaked the Eigen thread count";
    }
}

#endif  // EIGEN_USE_BLAS

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::ad::trex_core */
