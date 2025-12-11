#ifndef UTILS_TALGOS_HPP
#define UTILS_TALGOS_HPP

#include <atomic>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>

#include <Eigen/Dense>

#include "utils_eval.hpp"
#include "utils_perf.hpp"
#include "utils_memmap.hpp"
#include "utils_openmp.hpp"


namespace utils_talgos {


/**
 * @brief Generate synthetic regression data with known support in-memory
 */

struct SyntheticData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::VectorXd beta_true;
    std::vector<std::size_t> true_support;

    SyntheticData(std::size_t n,
                  std::size_t p,
                  const std::vector<std::size_t>& support,
                  const std::vector<double>& coefs,
                  double snr = 1.0,
                  unsigned int seed = 42)
        : true_support(support) {

        // Set random seed for reproducibility
        std::srand(seed);

        X = Eigen::MatrixXd::Random(n, p);
        y = Eigen::VectorXd::Zero(n);
        beta_true = Eigen::VectorXd::Zero(p);

        // Generate signal
        for (std::size_t i = 0; i < support.size(); ++i) {
            beta_true[support[i]] = coefs[i];
            y += coefs[i] * X.col(support[i]);
        }

        // Calculate noise level from SNR
        double signal_power = y.squaredNorm() / n;
        double noise_std = std::sqrt(signal_power / snr);

        // Add noise
        y += noise_std * Eigen::VectorXd::Random(n);
    }
};


/**
 * @brief Fill matrix with normal(mu = 0, sigma = 1) random values in column-major order
 *
 * @param X_data Pointer to column-major matrix data (Eigen::MatrixXd layout)
 * @param n Number of rows
 * @param p Number of columns
 * @param base_seed Base random seed for reproducibility
 *
 * @note Eigen matrices are column-major by default.
 */

inline void fill_matrix_column_major_normal(
    double* X_data,
    std::size_t n,
    std::size_t p,
    unsigned int base_seed = 42)
{
    std::cout << "  [DEBUG] fill_matrix_column_major_normal:\n";
    std::cout << "    Using seeds " << base_seed << " to " << (base_seed + p - 1) << "\n";

    // Check what seed is actually used for problematic columns
    std::vector<std::size_t> check_cols = {4, 27, 149, 398, 420};
    for (std::size_t col : check_cols) {
        std::cout << "    X.col(" << col << ") uses seed: " << (base_seed + col) << "\n";
    }

    std::atomic<std::size_t> completed_columns{0};
    std::size_t last_reported = 0;

    #pragma omp parallel
    {

        #pragma omp for schedule(static, 1000)
        for (std::size_t j = 0; j < p; ++j) {
            std::mt19937 gen(base_seed + 100 * p + j);
            std::normal_distribution<double> dist(0.0, 1.0);

            // Fill column j (Eigen column-major: elements are contiguous)
            for (std::size_t i = 0; i < n; ++i) {
                X_data[i + j * n] = dist(gen);
            }

            // Progress reporting
            std::size_t current = ++completed_columns;
            if (current % 1000 == 0 || current == p) {
                #pragma omp critical(progress_reporting)
                {
                    if (current > last_reported) {
                        std::cout << "  Progress: " << current << "/" << p
                                  << " columns (" << std::fixed
                                  << std::setprecision(1)
                                  << (100.0 * current / p) << "%)\r"
                                  << std::flush;
                        last_reported = current;
                    }
                }
            }
        }
    }
    std::cout << "\n";
}


/**
 * @brief Step 1: Create memory-mapped X matrix and y response vector
 *
 * Creates X ~ N(0,1) and y = X * beta_true + noise on disk as memory-mapped files.
 *
 * @param X_filename Path to X matrix file (will be created)
 * @param y_filename Path to y vector file (will be created)
 * @param n Number of observations (rows)
 * @param p Number of predictors (columns)
 * @param true_support Indices of true non-zero coefficients
 * @param true_support_coefs Values of true non-zero coefficients
 * @param snr Signal-to-noise ratio
 * @param seed Random seed for reproducibility
 */

inline void create_mmapped_X_and_y(
    const char* X_filename,
    const char* y_filename,
    std::size_t n,
    std::size_t p,
    const std::vector<std::size_t>& true_support,
    const std::vector<double>& true_support_coefs,
    double snr = 1.0,
    unsigned int seed = 42,
    bool reproducible_noise = true
    )
{
    std::cout << "=== Step 1: Creating memory-mapped X and y ===\n";
    std::cout << "  Dimensions: n=" << n << ", p=" << p << "\n";
    std::cout << "  True support size: " << true_support.size() << "\n";
    std::cout << "  SNR: " << snr << "\n\n";

    // Validate input
    if (true_support.size() != true_support_coefs.size()) {
        throw std::invalid_argument("true_support and true_support_coefs must have same size");
    }
    for (std::size_t idx : true_support) {
        if (idx >= p) {
            throw std::invalid_argument("Support index exceeds number of predictors");
        }
    }

    // Step 1.1: Check disk space
    std::size_t X_size = n * p * sizeof(double);
    std::size_t y_size = n * sizeof(double);
    std::cout << "  X size: " << (X_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  y size: " << (y_size / 1024.0) << " KB\n";

    if (!utils_perf::check_disk_space(X_size + y_size)) {
        throw std::runtime_error("Insufficient disk space for memory-mapped files");
    }

    // Step 1.2: Create memory-mapped files
    // Phase 1: Create and fill X with N(0,1) random values
    std::cout << "  Creating files...\n";
    {
        auto X_map = utils_memmap::create_empty_map<double>(X_filename, n * p);
        fill_matrix_column_major_normal(X_map.data, n, p, seed);
        utils_memmap::flush_mapping(X_map);
    }

    // Phase 2: Reopen X and generate y
    std::cout << "  Reopening X and creating y...\n";
    auto X_map_readonly = utils_memmap::map_file_rw<double>(X_filename, n * p);
    auto y_map = utils_memmap::create_empty_map<double>(y_filename, n);

    // Step 1.4: Create Eigen::Map views
    Eigen::Map<Eigen::MatrixXd> X(X_map_readonly.data, n, p);
    Eigen::Map<Eigen::VectorXd> y(y_map.data, n);

    // Step 1.5: Generate response y = X * beta_true + noise
    std::cout << "  Generating response y = X*beta_true + noise...\n";
    y.setZero();

    // Add signal from true support
    for (std::size_t i = 0; i < true_support.size(); ++i) {
        std::size_t idx = true_support[i];
        double coef = true_support_coefs[i];
        y.noalias() += coef * X.col(idx);
        std::cout << "    beta[" << idx << "] = " << coef << "\n";
    }


    // Add noise
    // Calculate noise level from SNR
    double signal_power = y.squaredNorm() / n;
    double noise_std = std::sqrt(signal_power / snr);
    std::cout << "  Signal power: " << signal_power << "\n";
    std::cout << "  Noise std: " << noise_std << "\n";

    std::mt19937 gen;
    if (reproducible_noise) {
        gen.seed(seed + 100 * p); // Large separation, reproducible, avoid spurious correlation
        std::cout << "  Using reproducible noise with seed " << (seed + 100 * p) << "\n";
    } else {
        std::random_device rd;
        gen.seed(rd());
        std::cout << "  Using random noise (non-reproducible)\n";
    }

    std::normal_distribution<double> noise_dist(0.0, noise_std);
    for (std::size_t i = 0; i < n; ++i) {
        y(i) += noise_dist(gen);
    }

    // Step 1.6: Flush to disk
    std::cout << "  Flushing to disk...\n";
    utils_memmap::flush_mapping(X_map_readonly);
    utils_memmap::flush_mapping(y_map);

    std::cout << "  ✓ Files created: " << X_filename << ", " << y_filename << "\n\n";
}



/**
 * @brief Create X_aug = [X | D] from existing memory-mapped X
 *
 * @param X_filename Path to existing X matrix file (n x p)
 * @param X_aug_filename Path to X_aug file to create (n x (p + num_dummies))
 * @param n Number of observations
 * @param p Number of original predictors
 * @param num_dummies Number of dummy variables to append
 * @param seed Random seed for dummy generation (reproducible)
 * @return MappedFile<double> handle to X_aug
 */
inline utils_memmap::MappedFile<double> augment_with_dummies(
    const char* X_filename,
    const char* X_aug_filename,
    std::size_t n,
    std::size_t p,
    std::size_t num_dummies,
    unsigned int seed = 42)
{
    std::cout << "=== Augmenting X with " << num_dummies << " dummies ===\n";

    std::cout << "=== Augmenting X with " << num_dummies << " dummies ===\n";

    // DIAGNOSTIC: Print seed ranges
    std::cout << "  [SEED INFO]\n";
    std::cout << "    X columns use seeds: " << seed << " to " << (seed + p - 1) << "\n";
    std::cout << "    Dummy columns will use seeds: "
              << (seed + 100*p + 1) << " to "
              << (seed + 100*p + num_dummies) << "\n";
    std::cout << "    Separation: " << (100*p + 1 - p) << " seeds\n";

    // Step 1: Load X
    auto X_map = utils_memmap::map_file_rw<double>(X_filename, n * p);
    const double* X_data = X_map.data;

    // Step 2: Create X_aug file
    auto X_aug_map = utils_memmap::create_empty_map<double>(
        X_aug_filename, n * (p + num_dummies)
    );
    double* X_aug_data = X_aug_map.data;

    // Step 3: Copy X to first p columns (parallel)
    std::cout << "  Copying X..." << std::flush;
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < p; ++j) {
        const double* src = X_data + j * n;
        double* dst = X_aug_data + j * n;
        std::memcpy(dst, src, n * sizeof(double));
    }
    std::cout << " done\n";

    // Step 4: Generate dummies (parallel, deterministic)
    std::cout << "  Generating " << num_dummies << " dummies..." << std::flush;
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < num_dummies; ++j) {
        std::mt19937 gen(seed + 1000003UL + j);  // 1000003 is prime
        std::normal_distribution<double> dist(0.0, 1.0);

        double* col_start = X_aug_data + (p + j) * n;
        for (std::size_t i = 0; i < n; ++i) {
            col_start[i] = dist(gen);
        }
    }
    std::cout << " done\n";

    // DIAGNOSTIC: Check for collinearity
    std::cout << "  [COLLINEARITY CHECK]\n";
    Eigen::Map<Eigen::MatrixXd> X_aug_check(X_aug_data, n, p + num_dummies);

    // Check suspicious dummies vs their corresponding X columns
    std::vector<std::size_t> check_pairs = {3, 26, 148, 397, 419};
    for (std::size_t idx : check_pairs) {
        double corr = X_aug_check.col(idx + 1).dot(X_aug_check.col(p + idx)) /
                      (X_aug_check.col(idx + 1).norm() * X_aug_check.col(p + idx).norm());
        std::cout << "    Corr(X.col(" << (idx+1) << "), Dummy.col(" << idx
                  << ")) = " << corr << "\n";
    }

    // Step 5: Flush
    utils_memmap::flush_mapping(X_aug_map);
    std::cout << "  ✓ X_aug created (" << n << " x " << (p + num_dummies) << ")\n";

    return X_aug_map;
}


/**
 * @brief Print T-LARS configuration
 */
inline void print_tlars_config(std::size_t n,
                               std::size_t p,
                               std::size_t num_dummies,
                               std::size_t T_stop,
                               const std::vector<std::size_t>& support,
                               const std::vector<double>& coefs,
                               double snr = 1.0) {

    std::cout << "Configuration:\n"
              << "  Observations (n): " << n << "\n"
              << "  Predictors (p): " << p << "\n"
              << "  Dummies: " << num_dummies << "\n"
              << "  T_stop: " << T_stop << "\n"
              << "  True support: ";
    for (const auto& idx : support) {
        std::cout << idx << " ";
    }
    std::cout << "\n  Coefficients: ";
    for (const auto& coef : coefs) {
        std::cout << coef << " ";
    }
    std::cout << "\n  SNR: " << snr << "\n\n";
}


/**
 * @brief Print selected variables
 */
template<typename SolverType>
inline void print_selection(
    const SolverType& solver,
    const std::vector<std::size_t>& true_support)
{
    std::cout << "\nSelection:\n";

    // 1. Predictors in active set
    std::cout << "  Predictors: ";
    utils_eval::print_vector(solver.getActivePredictorIndices());

    // 2. Dummies in active set
    std::cout << "\n  Dummies: ";
    utils_eval::print_vector(solver.getActiveDummyIndices());

    // 3. Full selection path (with dummies colored)
    auto actions = solver.getActions();
    std::size_t dummy_start = solver.getDummyStartIndex();
    std::cout << "\n  Path: [";
    bool first = true;
    for (std::size_t step = 0; step < actions.size(); ++step) {
        for (int action : actions[step]) {
            if (!first) std::cout << ", ";
            first = false;

            if (action < 0) {
                // REMOVAL EVENT
                std::size_t removed_idx = static_cast<std::size_t>(-action);
                if (removed_idx >= dummy_start) {
                    // Dummy removal: magenta
                    std::cout << "\033[35m-D" << (removed_idx - dummy_start)
                              << "\033[0m";
                } else {
                    // Variable removal: yellow
                    std::cout << "\033[33m-v" << removed_idx << "\033[0m";
                }
            } else {
                // ADDITION EVENT
                std::size_t idx = static_cast<std::size_t>(action);
                bool is_tp = std::find(true_support.begin(),
                                       true_support.end(), idx)
                                       != true_support.end();
                if (idx >= dummy_start) {
                    // Dummy addition: blue
                    std::cout << "\033[34m+D" << (idx - dummy_start)
                              << "\033[0m";
                } else if (is_tp) {
                    // True positive variable addition: green
                    std::cout << "\033[32m+v" << idx << "\033[0m";
                } else {
                    // False positive variable addition: red
                    std::cout << "\033[31m+v" << idx << "\033[0m";
                }
            }
        }
    }
    std::cout << "]\n";
}


/**
 * @brief Print selection quality
 */
template<typename SolverType>
inline void print_quality(const SolverType& solver,
                          const std::vector<std::size_t>& true_support)
{

    auto selected = solver.getActivePredictorIndices();
    std::size_t tp = 0;
    for (std::size_t idx : selected) {
        if (std::find(true_support.begin(),
                      true_support.end(), idx) != true_support.end()) {
            tp++;
        }
    }
    double precision = selected.size() > 0 ? static_cast<double>(tp)
                        / selected.size() : 0.0;
    double recall = true_support.size() > 0 ? static_cast<double>(tp)
                        / true_support.size() : 0.0;

    std::cout << "\nQuality: TP=" << tp << "/" << true_support.size()
              << ", FP=" << (selected.size() - tp)
              << ", Precision=" << std::fixed << std::setprecision(1)
              << (precision * 100) << "%"
              << ", Recall=" << (recall * 100) << "%\n";
}

} /* End of namespace utils_talgos */

#endif /* UTILS_TALGOS_HPP */
