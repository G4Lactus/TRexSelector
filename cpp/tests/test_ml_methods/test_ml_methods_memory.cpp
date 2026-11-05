// ==================================================================================
/**
 * @file test_ml_methods_memory.cpp
 *
 * @brief Peak-memory budget of the ML cores, in units of the design matrix
 *        (the allocation guard).
 *
 * @details Eigen allocates through malloc, not operator new, so a global
 *          operator-new counter cannot see it; the measure is the process
 *          peak resident set (getrusage ru_maxrss: bytes on macOS, kilobytes
 *          on Linux). getrusage is POSIX, so on Windows this file holds one
 *          skipped test. Each test takes the peak before a call and
 *          after it on a design of 32 MB; the growth is the working memory
 *          the call needed on top of what was already resident. A full
 *          copy of X raises it by at least one X. The budgets below are the
 *          CURRENT state of each core, tightened as copies are removed; a
 *          regression that adds a copy of X fails here.
 *
 *          The measurement is a process-wide peak, so each test must run in
 *          its own process: ctest does (gtest_discover_tests invokes the
 *          binary per test); a manual run needs --gtest_filter=MLMemory.<name>,
 *          otherwise the first test's peak hides the later ones. Set
 *          TREX_ML_MEMORY_VERBOSE=1 to print the measured growth per call.
 *
 *          Measured on the copying cores before the in-place rewrites
 *          (macOS, Accelerate BLAS, Eigen 5.0), wide 500 x 8000: ridge CV
 *          4.55 x X, elastic net 1.09 x X, elastic net CV 2.17 x X, SVD
 *          0.35 x X, ridge 0.33 x X, PCA 0.35 x X, scaler 0.03 x X.
 */
// ==================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

// system includes (getrusage is POSIX)
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

// Eigen includes
#include <Eigen/Dense>

// project ml_methods includes
#include <ml_methods/model_selection/enet_cv_ccd.hpp>
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/svd/svd.hpp>
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// ==================================================================================

namespace trex::test::ml_methods::memory {

using trex::ml_methods::model_selection::enet_cv_ccd;
using trex::ml_methods::model_selection::enet_gaussian;
using trex::ml_methods::model_selection::ridge_cv_svd;
using trex::ml_methods::pca::PCA;
using trex::ml_methods::ridge::RidgeSolver;
using trex::ml_methods::scaler_methods::LpNormScaler;
using trex::ml_methods::svd::SVDSolver;

#if defined(_WIN32)

/** @brief The peak measure is POSIX; the budgets are measured on macOS and Linux. */
TEST(MLMemory, SkippedOnWindows) {
    GTEST_SKIP() << "peak resident set via getrusage: POSIX only";
}

#else

namespace {

/** @brief Peak resident set size of the process, in bytes. */
double peak_rss_bytes() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return static_cast<double>(ru.ru_maxrss);
#else
    return static_cast<double>(ru.ru_maxrss) * 1024.0;
#endif
}

Eigen::MatrixXd random_matrix(Eigen::Index n, Eigen::Index p, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    return Eigen::MatrixXd::NullaryExpr(
        n, p, [&](Eigen::Index, Eigen::Index) { return dist(gen); });
}

/** @brief The wide regime: n = 500, p = 8000 -> X is 32 MB. */
constexpr Eigen::Index kN = 500;
constexpr Eigen::Index kP = 8000;

struct Design {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    double bytes;
    Design() : X(random_matrix(kN, kP, 5u)), y(X.col(0) * 2.0 - X.col(5) + random_matrix(kN, 1, 6u)),
               bytes(static_cast<double>(kN) * static_cast<double>(kP) * 8.0) {}
};

/** @brief Runs f, returns the peak growth in units of X, prints if asked. */
template <class F>
double growth_in_X(const Design& d, const char* what, F&& f) {
    // Touch every element so X itself is resident before the baseline.
    volatile double sink = d.X.sum(); (void)sink;
    const double before = peak_rss_bytes();
    f();
    const double after = peak_rss_bytes();
    const double units = (after - before) / d.bytes;
    if (const char* v = std::getenv("TREX_ML_MEMORY_VERBOSE"); v != nullptr && *v == '1') {
        std::cout << "[  MEMORY  ] " << what << ": peak growth " << units << " x X ("
                  << (after - before) / 1e6 << " MB)\n";
    }
    return units;
}

} // namespace

// ==================================================================================
// Budgets: the current state of each core
// ==================================================================================

/** @brief RidgeCV: per fold X_train + standardized copy + Jacobi internals. */
TEST(MLMemory, RidgeCV) {
    Design d;
    const double g = growth_in_X(d, "ridge_cv_svd::fit", [&] {
        ridge_cv_svd cv;
        cv.fit(d.X, d.y, 5, 20, 1000.0, 1u);
    });
    // The Jacobi SVD keeps one internal copy of the fold under Eigen 5.0 (the
    // QR preconditioner) and two under Eigen 3.4 (plus its scaled copy):
    // measured 3.64 X on macOS / Eigen 5.0 and 4.50 X on Ubuntu / Eigen 3.4.
    const double budget = EIGEN_VERSION_AT_LEAST(5, 0, 0) ? 4.0 : 5.0;
    EXPECT_LT(g, budget) << "ridge CV allocated more than its budget of " << budget << " x X";
}

/** @brief ElasticNet: the centred member copy Xs_ plus p x K paths. */
TEST(MLMemory, ElasticNetPath) {
    Design d;
    const double g = growth_in_X(d, "enet_gaussian::fit", [&] {
        enet_gaussian en;
        en.fit(d.X, d.y, 1.0, 20, -1.0, true, true, false, 1000, 1e-5);
    });
    EXPECT_LT(g, 1.6) << "elastic net allocated more than its budget of 1.6 x X";
}

/** @brief RidgeCV, kernel engine: n x n kernels and column blocks only. */
TEST(MLMemory, RidgeCVLowMemory) {
    Design d;
    const double g = growth_in_X(d, "ridge_cv_svd::fit (low_memory)", [&] {
        ridge_cv_svd cv(/*low_memory=*/true);
        cv.fit(d.X, d.y, 5, 20, 1000.0, 1u);
    });
    // O(n^2) kernels plus the eigen solver's workspaces and the column-block
    // buffers: about 11 MB here (0.35 X of a 32 MB design), independent of p.
    EXPECT_LT(g, 0.5) << "low-memory ridge CV allocated more than its budget of 0.5 x X";
}

/** @brief ElasticNet, raw-X engine: no allocation of the size of X. */
TEST(MLMemory, ElasticNetPathLowMemory) {
    Design d;
    const double g = growth_in_X(d, "enet_gaussian::fit (low_memory)", [&] {
        enet_gaussian en(/*low_memory=*/true);
        en.fit(d.X, d.y, 1.0, 20, -1.0, true, true, false, 1000, 1e-5);
    });
    EXPECT_LT(g, 0.2) << "low-memory elastic net allocated more than its budget of 0.2 x X";
}

/** @brief ElasticNetCV, raw-X engine: masks and row-wise predictions only. */
TEST(MLMemory, ElasticNetCVLowMemory) {
    Design d;
    const double g = growth_in_X(d, "enet_cv_ccd::fit (low_memory)", [&] {
        enet_cv_ccd cv(/*low_memory=*/true);
        cv.fit(d.X, d.y, 0.5, 5, 20, -1.0, 2u, true, true, 1000, 1e-5);
    });
    // One fit needs about 0.1 X (the p x K paths, 0.095 X in ElasticNetPathLowMemory).
    // The six fits of a 5-fold CV run one after another, and whether the allocator
    // reuses the freed paths varies between runs: measured 0.12-0.23 X on macOS /
    // Eigen 5.0 for identical input. A copy of X would add 1.0 X.
    EXPECT_LT(g, 0.3) << "low-memory elastic net CV allocated more than its budget of 0.3 x X";
}

/** @brief ElasticNetCV: full-data Xs_, then per fold X_tr + X_te + the fold's Xs_. */
TEST(MLMemory, ElasticNetCV) {
    Design d;
    const double g = growth_in_X(d, "enet_cv_ccd::fit", [&] {
        enet_cv_ccd cv;
        cv.fit(d.X, d.y, 0.5, 5, 20, -1.0, 2u, true, true, 1000, 1e-5);
    });
    EXPECT_LT(g, 1.6) << "elastic net CV allocated more than its budget of 1.6 x X";
}

/** @brief SVD, wide: the Gram path forms n x n and reads X blockwise. */
TEST(MLMemory, SVDWide) {
    Design d;
    const double g = growth_in_X(d, "SVDSolver::compute (wide, Gram path)", [&] {
        auto r = SVDSolver{}.compute(d.X, 5);
        volatile double sink = r.S.sum(); (void)sink;
    });
    EXPECT_LT(g, 0.5) << "wide SVD allocated more than its budget of 0.5 x X";
}

/** @brief Ridge, wide: the n x n kernel and its factorization only. */
TEST(MLMemory, RidgeWide) {
    Design d;
    const double g = growth_in_X(d, "RidgeSolver::solve (wide, dual)", [&] {
        Eigen::VectorXd b = RidgeSolver::solve(d.X, d.y, 1.0);
        volatile double sink = b.sum(); (void)sink;
    });
    EXPECT_LT(g, 0.5) << "wide ridge allocated more than its budget of 0.5 x X";
}

/** @brief PCA on a wide design: in place, then the Gram-path SVD. */
TEST(MLMemory, PCAWide) {
    Design d;
    const double g = growth_in_X(d, "PCA (wide)", [&] {
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        PCA pca(Xm, 5, true, true);
        auto r = pca.fit();
        volatile double sink = r.explained_variance.sum(); (void)sink;
        pca.restore(Xm);
    });
    EXPECT_LT(g, 0.5) << "wide PCA allocated more than its budget of 0.5 x X";
}

/** @brief Scalers: in place, no allocation of the size of X. */
TEST(MLMemory, Scaler) {
    Design d;
    const double g = growth_in_X(d, "LpNormScaler::fit_transform_inplace", [&] {
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        LpNormScaler s;
        s.fit_transform_inplace(Xm);
        s.inverse_transform_inplace(Xm);
    });
    EXPECT_LT(g, 0.1) << "scaler allocated more than its budget of 0.1 x X";
}

#endif  // !_WIN32

} // namespace trex::test::ml_methods::memory
