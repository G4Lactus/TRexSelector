// ===================================================================================
// demo_mlm_ms_01_ridge_gcv.cpp
// ===================================================================================
/**
 * @file demo_mlm_ms_01_ridge_gcv.cpp
 *
 * @brief Demo of Ridge Regression with Generalized Cross-Validation (GCV).
 *
 * Organization:
 * -------------
 * |-- ml_methods/
 * |   |-- model_selection/
 * |       |-- demo_mlm_ms_01_ridge_gcv.cpp
 *
 * Scenarios covered:
 * ------------------
 *  1. Low-dimensional  : n=1000, p=500  (n > p)
 *  2. High-dimensional : n=300,  p=500  (p > n)
 *  3. Very high-dim    : n=300,  p=1000 (p >> n, rank = n)
 *  4. Memory-mapped    : n=5000, p=500  — dataset lives on disk; fit() reads it
 *                        via ConstMapType (zero-copy Eigen::Ref binding).
 */
// ===================================================================================

// std includes
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/model_selection/ridge_gcv.hpp>
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
// Shared data-generation helpers
// ===================================================================================

/**
 * @brief Generate a synthetic regression dataset X, y with known sparse signal.
 *
 * @details The true coefficient vector has p n_active non-zero entries, all equal
 *          to 1.0, with indices evenly spaced in [0, p). Observations are:
 *          y = X * beta_true + noise, where noise ~ N(0, sigma^2).
 *
 * @param n               Number of observations.
 * @param p               Number of predictors.
 * @param sigma           Noise standard deviation.
 * @param seed            RNG seed.
 * @param[out] X_out      (n x p) design matrix — columns ~ N(0,1), standardised.
 * @param[out] y_out      (n)     response vector.
 * @param[out] beta_true  (p) true coefficient vector (for reference).
 */
void generate_regression_data(
    Eigen::Index n,
    Eigen::Index p,
    double sigma,
    unsigned int seed,
    Eigen::MatrixXd& X_out,
    Eigen::VectorXd& y_out,
    Eigen::VectorXd& beta_true,
    int n_active = 10)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    // 1. Generate X
    X_out.resize(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = 0; i < n; ++i) {
            X_out(i, j) = nd(rng);
        }
    }

    // 2. Generate true coefficients
    beta_true.setZero(p);
    Eigen::Index stride = std::max(Eigen::Index{1}, p / n_active);
    for (int k = 0; k < n_active; ++k) {
        beta_true(k * stride) = 1.0;
    }

    // 3. Generate response y
    y_out = X_out * beta_true;
    for (Eigen::Index i = 0; i < n; ++i) {
        y_out(i) += sigma * nd(rng);
    }
}


/**
 * @brief Print a condensed summary of a GCV solve-path result.
 *
 * Prints the best lambda, its GCV score, effective df, and a miniature GCV
 * curve (every step_size-th entry) so the output remains readable.
 */
void print_gcv_path_summary(const ms::ridge_path& path, Eigen::Index step_size = 10) {
    const Eigen::Index K = path.lambdas.size();
    const Eigen::Index bi = path.best_index;

    // Format a GCV score for display: ridge_gcv::gcv() returns
    // std::numeric_limits<double>::max() as a sentinel for the degenerate
    // region |1 - df/n| < 1e-2.  Render it as "inf" instead of the raw
    // ~1.8e308 numeric value.
    auto gcv_str = [](double v) -> std::string {
        if (v >= 0.5 * std::numeric_limits<double>::max()) return "inf";
        std::ostringstream os;
        os << std::fixed << std::setprecision(4) << v;
        return os.str();
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Best lambda (grid)  : " << path.lambdas(bi) << "\n";
    std::cout << "  GCV score at best   : " << gcv_str(path.gcv_scores(bi)) << "\n";
    std::cout << "  Effective df at best: " << path.df_effective(bi) << "\n\n";

    std::cout << "  GCV curve (every " << step_size << " steps):\n";
    std::cout << "    " << std::left
              << std::setw(14) << "lambda"
              << std::setw(14) << "GCV"
              << std::setw(10) << "df_eff"
              << (std::string(bi / step_size * 14, ' ').empty() ? "" : "  (* = best)\n")
              << "\n";

    for (Eigen::Index k = 0; k < K; k += step_size) {
        std::cout << "    "
                  << std::setw(14) << path.lambdas(k)
                  << std::setw(14) << gcv_str(path.gcv_scores(k))
                  << std::setw(10) << path.df_effective(k)
                  << (k == bi ? "  <-- best" : "")
                  << "\n";
    }

    // Print best if it was skipped by step_size
    if (bi % step_size != 0) {
        std::cout << "    "
                  << std::setw(14) << path.lambdas(bi)
                  << std::setw(14) << gcv_str(path.gcv_scores(bi))
                  << std::setw(10) << path.df_effective(bi)
                  << "  <-- best\n";
    }
    std::cout << "\n";
}



// ===================================================================================
// Demo 1: Low-dimensional scenario (n=1000, p=500, n > p)
// ===================================================================================

void demo_gcv_low_dim() {

    constexpr Eigen::Index n = 1'000;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;

    cdiagnostics::print_section_header(
        "Demo MS-01 | Ridge GCV |\n Low-Dimensional (n=" +
        std::to_string(n) + ", p=" + std::to_string(p) + ", n > p)");

    // --- Generate data ---
    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/101, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    // --- Fit ridge_gcv ---
    ms::ridge_gcv gcv;
    gcv.fit(X, y);

    std::cout << "SVD rank: " << gcv.rank() << " (expected min(n,p)="
          << std::min(n,p) << ")\n\n";

    // --- Solve path over default lambda grid ---
    const Eigen::VectorXd lambdas = gcv.default_lambda_grid(100);
    const ms::ridge_path path = gcv.solve_path(lambdas);
    print_gcv_path_summary(path, 10);

    // --- Golden-section refinement of GCV optimum ---
    const double lambda_star = gcv.gcv_optimal(200, 1e-8);
    std::cout << "  GCV-optimal lambda (golden-section): "
              << lambda_star << "\n";
    std::cout << "  GCV score at optimal               : "
              << gcv.gcv(lambda_star) << "\n";
    std::cout << "  Effective df at optimal            : "
              << gcv.effective_df(lambda_star) << "\n\n";

    // --- Prediction MSE on a held-out test set ---
    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test, beta_tmp;
    generate_regression_data(n, p, sigma,
        /*seed=*/202, X_test, y_test, beta_tmp);

    const Eigen::VectorXd y_pred = gcv.predict(X_test, lambda_star);
    const double test_mse = (y_pred - y_test).squaredNorm() / static_cast<double>(n);
    std::cout << "  Test MSE at lambda_star: " << test_mse
              << "  (noise variance=" << sigma*sigma << ")\n\n";
}



// ===================================================================================
// Demo 2: High-dimensional scenario (n=300, p=500, p > n)
// ===================================================================================

void demo_gcv_high_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-01 | Ridge GCV | High-Dimensional (n=300, p=500, p > n)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/303, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n";
    std::cout << "  Note: thin SVD yields rank = min(n,p) = " << std::min(n,p) << "\n\n";

    ms::ridge_gcv gcv;
    gcv.fit(X, y);

    std::cout << "SVD rank: " << gcv.rank()
              << "  (= n=" << n << " because p > n; beta lives in a low-dim subspace)\n\n";

    const Eigen::VectorXd lambdas = gcv.default_lambda_grid(100);
    const ms::ridge_path path = gcv.solve_path(lambdas);
    print_gcv_path_summary(path, 10);

    const double lambda_star = gcv.gcv_optimal(200, 1e-8);
    std::cout << "  GCV-optimal lambda (golden-section): "
              << lambda_star << "\n";
    std::cout << "  GCV score at optimal               : "
              << gcv.gcv(lambda_star)
              << "\n";
    std::cout << "  Effective df at optimal            : "
              << gcv.effective_df(lambda_star)
              << "\n\n";

    // Prediction on held-out test set (same p, new n observations)
    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test, beta_tmp;
    generate_regression_data(n, p, sigma,
                             /*seed=*/404,
                             X_test,
                             y_test,
                             beta_tmp);

    const Eigen::VectorXd y_pred = gcv.predict(X_test, lambda_star);
    const double test_mse = (y_pred - y_test).squaredNorm() / static_cast<double>(n);

    std::cout << "  Test MSE at lambda_star: " << test_mse
              << "  (noise variance=" << sigma*sigma << ")\n\n";
}



// ===================================================================================
// Demo 3: Very high-dimensional scenario (n=300, p=1000, p >> n)
// ===================================================================================

void demo_gcv_very_high_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-01 | Ridge GCV | Very High-Dimensional (n=300, p=1000, p >> n)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 1000;
    constexpr double sigma = 1.0;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/505, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n";
    std::cout << "  Note: rank is capped at n=" << n << "; "
              << (p - n)
              << " extra features receive zero coefficient (in the hat-matrix sense)\n\n";

    ms::ridge_gcv gcv;
    gcv.fit(X, y);

    std::cout << "SVD rank: " << gcv.rank() << "  (expected " << std::min(n,p) << ")\n\n";

    const Eigen::VectorXd lambdas = gcv.default_lambda_grid(100);
    const ms::ridge_path path = gcv.solve_path(lambdas);
    print_gcv_path_summary(path, 10);

    const double lambda_star = gcv.gcv_optimal(200, 1e-8);
    std::cout << "  GCV-optimal lambda (golden-section): "
              << lambda_star << "\n";
    std::cout << "  GCV score at optimal               : "
              << gcv.gcv(lambda_star) << "\n";
    std::cout << "  Effective df at optimal            : "
              << gcv.effective_df(lambda_star) << "\n\n";

    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test, beta_tmp;
    generate_regression_data(n, p, sigma,
                             /*seed=*/606, X_test, y_test, beta_tmp);

    const Eigen::VectorXd y_pred = gcv.predict(X_test, lambda_star);
    const double test_mse = (y_pred - y_test).squaredNorm() / static_cast<double>(n);
    std::cout << "  Test MSE at lambda_star: " << test_mse
              << "  (noise variance=" << sigma*sigma << ")\n\n";
}



// ===================================================================================
// Demo 4: Memory-mapped scenario (n=5000, p=500)
// ===================================================================================
/**
 * @brief Demonstrates GCV fitting when the dataset is backed by a memory-mapped file.
 *
 * @details
 *   The design matrix X is written into a MemoryMappedMatrix<double> on disk.
 *   A ConstMapType (Eigen::Map<const MatrixXd>) is then passed directly to
 *   ridge_gcv::fit(), which accepts Eigen::Ref<const MatrixXd> — Eigen's Ref
 *   binds the Map without any copy at the call site.
 *
 *   RAM usage during fit() is still O(n*p) because fit() must centre X and
 *   compute a thin SVD of the centred matrix.  The mmap benefit here is that
 *   the raw dataset resides on disk and can be shared across processes or
 *   re-opened across runs without re-generation.
 */
void demo_gcv_mmap() {
    cdiagnostics::print_section_header(
        "Demo MS-01 | Ridge GCV | Memory-Mapped Dataset (n=5000, p=500)");

    constexpr Eigen::Index n = 5000;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;

#ifdef TREX_DEMO_MMAP_PATH
    const std::string MMAP_PATH_X = TREX_DEMO_MMAP_PATH;
#else
    const std::string MMAP_PATH_X = "trex_mlm_demo_ms01_gcv_mmap.bin";
#endif

    // -------------------------------------------------------------------------
    // Phase 1: Generate data and write X into the mmap file
    // -------------------------------------------------------------------------

    // Generate in-memory first (y and beta_true are small, kept in RAM)
    Eigen::MatrixXd X_dense;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/707, X_dense, y, beta_true);

    const std::size_t file_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(p) * sizeof(double);

    std::cout << "Writing X to mmap file: " << MMAP_PATH_X << "\n";
    std::cout << "  File size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(file_bytes) / (1 << 20) << " MB\n\n";

    {
        MmapMatrix mmap_w(MMAP_PATH_X,
                          static_cast<std::size_t>(n),
                          static_cast<std::size_t>(p),
                          memmap::AccessMode::ReadWrite);
        EigenMap X_rw = mmap_w.getMap();
        X_rw = X_dense;   // single copy from RAM → disk-backed mmap
    }   // MmapMatrix destructor closes + flushes here (RAII)

    // -------------------------------------------------------------------------
    // Phase 2: Re-open read-only and fit ridge_gcv via ConstMapType
    // -------------------------------------------------------------------------

    std::cout << "Re-opening mmap read-only and fitting ridge_gcv...\n";

    {
        const MmapMatrix mmap_r(MMAP_PATH_X,
                                static_cast<std::size_t>(n),
                                static_cast<std::size_t>(p),
                                memmap::AccessMode::ReadOnly);
        const ConstEigenMap X_ro = mmap_r.getMap();

        // X_ro is Eigen::Map<const MatrixXd>; ridge_gcv::fit takes
        // Eigen::Ref<const MatrixXd> — binding is zero-copy.
        ms::ridge_gcv gcv;
        gcv.fit(X_ro, y);   // <-- mmap ConstMapType passed as Ref

        std::cout << "  SVD rank: " << gcv.rank() << "\n\n";

        const double lambda_star = gcv.gcv_optimal(200, 1e-8);
        std::cout << "  GCV-optimal lambda: " << lambda_star << "\n";
        std::cout << "  GCV score         : " << gcv.gcv(lambda_star) << "\n";
        std::cout << "  Effective df      : " << gcv.effective_df(lambda_star) << "\n\n";

        // Test-set prediction (test X kept in RAM — smaller than training set)
        Eigen::MatrixXd X_test;
        Eigen::VectorXd y_test, beta_tmp;
        generate_regression_data(300, p, sigma,
                                 /*seed=*/808, X_test, y_test, beta_tmp);

        const Eigen::VectorXd y_pred = gcv.predict(X_test, lambda_star);
        const double test_mse = (y_pred - y_test).squaredNorm() / 300.0;
        std::cout << "  Test MSE at lambda_star: " << test_mse
                  << "  (noise variance=" << sigma*sigma << ")\n\n";
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
        "Ridge GCV Demo Suite  (demo_mlm_ms_01_ridge_gcv)");

    if (true) {
        demo_gcv_low_dim();
    }

    if (true) {
        demo_gcv_high_dim();
    }

    if (true) {
        demo_gcv_very_high_dim();
    }

    if (true) {
        demo_gcv_mmap();
    }


    std::cout << "\nAll Ridge GCV demos complete.\n\n";
    return 0;
}
