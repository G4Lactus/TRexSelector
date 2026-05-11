// ===================================================================================
// demo_mlm_ms_02_ridge_cv.cpp
// ===================================================================================
/**
 * @file demo_mlm_ms_02_ridge_cv.cpp
 *
 * @brief Demonstration of Ridge Regression with K-Fold Cross-Validation (CV).
 *
 * Organization:
 * -------------
 * |-- ml_methods/
 * |   |-- model_selection/
 * |       |-- demo_mlm_ms_02_ridge_cv.cpp
 *
 * Scenarios covered:
 * ------------------
 *  1. Low-dimensional  : n=1000, p=500  (n > p,  10-fold)
 *  2. High-dimensional : n=300,  p=500  (p > n,  10-fold)
 *  3. Very high-dim    : n=300,  p=1000 (p >> n,  5-fold — fewer folds for small n)
 *  4. Memory-mapped    : n=5000, p=500  — dataset on disk; ridge_cv::fit() reads
 *                        it via ConstMapType (zero-copy Eigen::Ref binding).
 *
 * Memory-mapping notes:
 * ---------------------
 *   ridge_cv::fit() accepts Eigen::Ref<const MatrixXd>, which binds to
 *   MemoryMappedMatrix::ConstMapType without copying the full matrix.
 *   However, fit() assembles dense fold sub-matrices (X_train, X_test) in RAM for each
 *   fold — there is no out-of-core path. For K folds the peak RAM usage is
 *   O(n * p) (one full-fold train matrix at a time). The mmap benefit is that
 *   the raw dataset can persist on disk across runs or be shared between processes.
 */
// ===================================================================================

// std includes
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/model_selection/ridge_cv.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>


// ===================================================================================
// Namespace aliases
// ===================================================================================

namespace ms          = trex::ml_methods::model_selection;
namespace memmap      = trex::utils::memmap;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;

using MmapMatrix    = memmap::MemoryMappedMatrix<double>;
using EigenMap      = MmapMatrix::MapType;
using ConstEigenMap = MmapMatrix::ConstMapType;


// ===================================================================================
// Shared data-generation helper (same scheme as GCV demo)
// ===================================================================================

/**
 * @brief Generate a synthetic regression dataset with known sparse signal.
 *
 * @details True coefficients: @p n_active entries equal to 1.0, at evenly-spaced
 *          column indices. Observations: y = X * beta_true + N(0, sigma^2).
 */
void generate_regression_data(
    Eigen::Index        n,
    Eigen::Index        p,
    double              sigma,
    unsigned int        seed,
    Eigen::MatrixXd&    X_out,
    Eigen::VectorXd&    y_out,
    Eigen::VectorXd&    beta_true,
    int                 n_active = 10)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    X_out.resize(n, p);
    for (Eigen::Index j = 0; j < p; ++j)
        for (Eigen::Index i = 0; i < n; ++i)
            X_out(i, j) = nd(rng);

    beta_true.setZero(p);
    Eigen::Index stride = std::max(Eigen::Index{1}, p / n_active);
    for (int k = 0; k < n_active; ++k)
        beta_true(k * stride) = 1.0;

    y_out = X_out * beta_true;
    for (Eigen::Index i = 0; i < n; ++i)
        y_out(i) += sigma * nd(rng);
}


// ===================================================================================
// Shared print helper
// ===================================================================================

/**
 * @brief Print a condensed summary of K-fold CV results.
 *
 * Prints lambda.min, lambda.1se, their CV-MSE and SEM, and a condensed MSE
 * curve (every @p step_size-th grid point).
 */
void print_cv_summary(const ms::ridge_cv& cv, Eigen::Index step_size = 10) {
    const Eigen::VectorXd& lambdas = cv.lambdas();
    const Eigen::VectorXd& mse     = cv.cv_mse();
    const Eigen::VectorXd& sem     = cv.cv_sem();
    const Eigen::Index     K       = lambdas.size();
    const Eigen::Index     i_min   = cv.index_min();
    const Eigen::Index     i_1se   = cv.index_1se();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  lambda.min : " << cv.cv_min()
              << "  (CV-MSE=" << mse(i_min) << " +/- " << sem(i_min) << ")\n";
    std::cout << "  lambda.1se : " << cv.cv_1se()
              << "  (CV-MSE=" << mse(i_1se) << " +/- " << sem(i_1se) << ")\n\n";

    std::cout << "  CV-MSE curve (every " << step_size << " steps):\n";
    std::cout << "    " << std::left
              << std::setw(14) << "lambda"
              << std::setw(12) << "CV-MSE"
              << std::setw(12) << "SEM"
              << "marker\n";

    for (Eigen::Index k = 0; k < K; k += step_size) {
        const char* marker =
            (k == i_min) ? "  <-- min" :
            (k == i_1se) ? "  <-- 1se" : "";
        std::cout << "    "
                  << std::setw(14) << lambdas(k)
                  << std::setw(12) << mse(k)
                  << std::setw(12) << sem(k)
                  << marker << "\n";
    }
    // Print min / 1se rows if they were skipped by step_size
    if (i_min % step_size != 0) {
        std::cout << "    "
                  << std::setw(14) << lambdas(i_min)
                  << std::setw(12) << mse(i_min)
                  << std::setw(12) << sem(i_min)
                  << "  <-- min\n";
    }
    if (i_1se % step_size != 0 && i_1se != i_min) {
        std::cout << "    "
                  << std::setw(14) << lambdas(i_1se)
                  << std::setw(12) << mse(i_1se)
                  << std::setw(12) << sem(i_1se)
                  << "  <-- 1se\n";
    }
    std::cout << "\n";
}


// ===================================================================================
// Demo 1: Low-dimensional scenario (n=1000, p=500, n > p, 10-fold)
// ===================================================================================

void demo_kfold_low_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | Ridge K-Fold CV | Low-Dimensional (n=1000, p=500, 10-fold)");

    constexpr Eigen::Index n = 1'000;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/111, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    ms::ridge_cv cv;
    cv.fit(X, y, n_folds);

    print_cv_summary(cv, 10);

    // Check: lambda.min <= lambda.1se (glmnet convention: 1se is more regularised)
    const bool order_ok = cv.cv_min() <= cv.cv_1se();
    std::cout << "  Ordering check (lambda.min <= lambda.1se): "
              << (order_ok ? "PASS" : "FAIL") << "\n\n";
}


// ===================================================================================
// Demo 2: High-dimensional scenario (n=300, p=500, p > n, 10-fold)
// ===================================================================================

void demo_kfold_high_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | Ridge K-Fold CV | High-Dimensional (n=300, p=500, 10-fold)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/222, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n";
    std::cout << "  Note: per-fold training set has ~" << (n - n/n_folds)
              << " rows; rank = min(n_train, p) = " << (n - n/n_folds) << "\n\n";

    ms::ridge_cv cv;
    cv.fit(X, y, n_folds);

    print_cv_summary(cv, 10);

    const bool order_ok = cv.cv_min() <= cv.cv_1se();
    std::cout << "  Ordering check (lambda.min <= lambda.1se): "
              << (order_ok ? "PASS" : "FAIL") << "\n\n";
}


// ===================================================================================
// Demo 3: Very high-dimensional scenario (n=300, p=1000, p >> n, 5-fold)
// ===================================================================================

void demo_kfold_very_high_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | Ridge K-Fold CV | Very High-Dimensional (n=300, p=1000, 5-fold)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 1000;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 5;  // Fewer folds: each fold train set has ~240 rows

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/333, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n";
    std::cout << "  Note: 5 folds used (rather than 10) to ensure each fold's "
                 "training set remains large enough relative to n.\n\n";

    ms::ridge_cv cv;
    cv.fit(X, y, n_folds);

    print_cv_summary(cv, 10);

    const bool order_ok = cv.cv_min() <= cv.cv_1se();
    std::cout << "  Ordering check (lambda.min <= lambda.1se): "
              << (order_ok ? "PASS" : "FAIL") << "\n\n";
}


// ===================================================================================
// Demo 4: Memory-mapped scenario (n=5000, p=500, 10-fold)
// ===================================================================================
/**
 * @brief Demonstrates K-fold CV when the design matrix is backed by a mmap file.
 *
 * @details
 *   ridge_cv::fit() accepts Eigen::Ref<const MatrixXd>, which binds a
 *   MemoryMappedMatrix::ConstMapType without copying the full matrix.  Internally,
 *   each fold materializes dense X_train / X_test sub-matrices in RAM — the peak
 *   working-set RAM per fold is O((n - n/K) * p) for training and O(n/K * p) for
 *   testing.  There is no out-of-core streaming path.  At n=5000, p=500, 10-fold:
 *   X_train peak RAM ≈ 4500 * 500 * 8 bytes ≈ 17 MB — trivially in-memory.
 */
void demo_kfold_mmap() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | Ridge K-Fold CV | Memory-Mapped Dataset (n=5000, p=500, 10-fold)");

    constexpr Eigen::Index n = 5000;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;

#ifdef TREX_DEMO_MMAP_PATH
    const std::string MMAP_PATH_X = TREX_DEMO_MMAP_PATH;
#else
    const std::string MMAP_PATH_X = "trex_mlm_demo_ms02_cv_mmap.bin";
#endif

    // -------------------------------------------------------------------------
    // Phase 1: Generate data and write X into the mmap file
    // -------------------------------------------------------------------------

    Eigen::MatrixXd X_dense;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/444, X_dense, y, beta_true);

    const std::size_t file_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(p) * sizeof(double);

    std::cout << "Writing X to mmap file: " << MMAP_PATH_X << "\n";
    std::cout << "  File size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(file_bytes) / (1 << 20) << " MB\n";
    std::cout << "  Active coefs: " << (beta_true.array() != 0.0).count() << "\n\n";

    {
        MmapMatrix mmap_w(MMAP_PATH_X,
                          static_cast<std::size_t>(n),
                          static_cast<std::size_t>(p),
                          memmap::AccessMode::ReadWrite);
        EigenMap X_rw = mmap_w.getMap();
        X_rw = X_dense;
    }   // MmapMatrix destructor closes + flushes here (RAII)

    // -------------------------------------------------------------------------
    // Phase 2: Re-open read-only and fit ridge_cv via ConstMapType
    // -------------------------------------------------------------------------

    std::cout << "Re-opening mmap read-only and fitting ridge_cv (" << n_folds << "-fold)...\n\n";

    {
        const MmapMatrix mmap_r(MMAP_PATH_X,
                                static_cast<std::size_t>(n),
                                static_cast<std::size_t>(p),
                                memmap::AccessMode::ReadOnly);
        const ConstEigenMap X_ro = mmap_r.getMap();

        // X_ro is Eigen::Map<const MatrixXd>; ridge_cv::fit takes
        // Eigen::Ref<const MatrixXd> — binding is zero-copy at the call site.
        // Each fold internally copies its ~4500-row train sub-matrix into RAM.
        ms::ridge_cv cv;
        cv.fit(X_ro, y, n_folds);

        print_cv_summary(cv, 10);

        const bool order_ok = cv.cv_min() <= cv.cv_1se();
        std::cout << "  Ordering check (lambda.min <= lambda.1se): "
                  << (order_ok ? "PASS" : "FAIL") << "\n\n";
    }   // mmap_r destructor closes the file here

    // -------------------------------------------------------------------------
    // Cleanup: remove the mmap file
    // -------------------------------------------------------------------------
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::remove(MMAP_PATH_X, ec))
            std::cout << "Cleaned up mmap file: " << MMAP_PATH_X << "\n";
        else
            std::cerr << "[WARN] Could not remove mmap file: " << ec.message() << "\n";
    }

    std::cout << "\n";
}


// ===================================================================================
// Main
// ===================================================================================

int main() {
    std::cout << std::unitbuf;  // Flush after every write for live log output.

    cdiagnostics::print_section_header(
        "Ridge K-Fold CV Demo Suite  (demo_mlm_ms_02_ridge_cv)");

    if (true) {
        demo_kfold_low_dim();
    }

    if (true) {
        demo_kfold_high_dim();
    }

    if (true) {
        demo_kfold_very_high_dim();
    }

    if (true) {
        demo_kfold_mmap();
    }


    std::cout << "\nAll Ridge K-Fold CV demos complete.\n\n";
    return 0;
}
