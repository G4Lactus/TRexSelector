// ===================================================================================
// demo_mlm_01_scaler.cpp
// ===================================================================================
/**
 * @file demo_mlm_01_scaler.cpp
 *
 * @brief Demonstration of ZScoreScaler and LpNormScaler usage.
 * Organiation:
 * ------------
 * |-- ml_methods/
 *.    |-- demo_01_scaler.cpp
 */
// ===================================================================================

// std includes
#include <iostream>
#include <random>
#include <string>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/standardization/lp_norm_scaler.hpp>
#include <ml_methods/standardization/z_score_scaler.hpp>
#include <utils/openmp/utils_openmp.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace scaler = trex::ml_methods::standardization;
namespace omp_utils = trex::utils::openmp;
namespace memmap = trex::utils::memmap;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;


// ============================================================================
// Demo 1: In-Memory Computations
// ============================================================================

void demo_scaler_comparison() {
    cdiagnostics::print_section_header("Scaler Comparison - ZScoreScaler vs LpNormScaler");

    // Step 1. Generate random data
    const Eigen::Index n = 2'000, p = 10'000;
    std::mt19937 rng(123);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    // Setup data matrix
    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }


    // Step 2. Create Map Wrapper
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);


    // Step 3. Fit Scalers
    scaler::ZScoreScaler zscaler;
    scaler::LpNormScaler l2scaler(scaler::LpNormScaler::NormType::L2, true);
    scaler::LpNormScaler l1scaler(scaler::LpNormScaler::NormType::L1, true);

    std::cout << "Fitting scalers...\n";
    zscaler.fit(X_map);
    l2scaler.fit(X_map);
    l1scaler.fit(X_map);
    std::cout << "✓ Scalers fitted\n\n";


    // Step 4. Transform performance
    std::cout << "Transforming data...\n";
    Eigen::MatrixXd X1 = X, X2 = X, X3 = X;
    Eigen::Map<Eigen::MatrixXd> X1_map(X1.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X3_map(X3.data(), n, p);

    zscaler.transform_inplace(X1_map);
    l2scaler.transform_inplace(X2_map);
    l1scaler.transform_inplace(X3_map);
    std::cout << "✓ Data transformed\n\n";

    std::cout << "\n\n";
}


// =============================================================================
// Demo 2: Transform and Inverse Transform Test
// =============================================================================

void demo_scaler_inverse_transform() {
    cdiagnostics::print_section_header("Scaler Inverse Transform Accuracy");

    // Step 1. Generate random data
    const Eigen::Index n = 1'000, p = 5'000;
    std::cout << "Matrix size: " << n << " x " << p << "\n\n";

    std::mt19937 rng(456);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    Eigen::MatrixXd X_orig(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X_orig(i, j) = norm_dist(rng);
        }
    }

    // Step 2. Test ZScoreScaler
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        scaler::ZScoreScaler zscaler;

        zscaler.fit(X_map);
        zscaler.transform_inplace(X_map);
        zscaler.inverse_transform_inplace(X_map);

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "ZScoreScaler:\n"
                  << "  Max Error: " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    // Step 3. Test LpNormScaler L2
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        scaler::LpNormScaler l2scaler(scaler::LpNormScaler::NormType::L2, true);


        l2scaler.fit(X_map);
        l2scaler.transform_inplace(X_map);
        l2scaler.inverse_transform_inplace(X_map);

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "LpNormScaler L2:\n"
                  << "  Max Error: " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }
    std::cout << "\n\n";
}


// =============================================================================
// Demo 3: Serialization
// =============================================================================

void demo_scaler_serialization() {
    cdiagnostics::print_section_header("Scaler Serialization");

    // Step 1. Generate data
    const Eigen::Index n = 1'000, p = 500;

    std::mt19937 rng(999);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);

    // Step 2. Fit and save
    std::string filename = "scaler_test.bin";
    scaler::ZScoreScaler scaler;
    scaler.fit(X_map);
    scaler.save(filename);
    std::cout << "✓ Saved ZScoreScaler to '" << filename << "'\n";

    // Step 3. Load and verify
    scaler::ZScoreScaler loaded_scaler;
    loaded_scaler.load(filename);
    std::cout << "✓ Loaded ZScoreScaler from '" << filename << "'\n\n";

    // Verify means and scales match
    double mean_diff = (scaler.get_means() - loaded_scaler.get_means()).norm();
    double scale_diff = (scaler.get_scales() - loaded_scaler.get_scales()).norm();

    std::cout << "Verification:\n"
              << "  Mean difference: " << std::scientific << mean_diff << "\n"
              << "  Scale difference: " << scale_diff << "\n\n";

    if (mean_diff < 1e-10 && scale_diff < 1e-10) {
        std::cout << "✓ Serialization perfect!\n\n";
    }

    // Step 4. Clean up
    std::remove(filename.c_str());
    std::cout << "\n\n";
}


// ============================================================================
// Demo 04: Memory-Mapped Matrix
// ============================================================================

void demo_scaler_on_mmap_matrix() {
    cdiagnostics::print_section_header("Scaler on Memory-Mapped Matrix");

    // Step 1. Create memory-mapped matrix
    std::cout << "Creating memory-mapped matrix...\n";
    const Eigen::Index n = 20'000, p = 100'000;
    std::cout << "Matrix size: " << n << " x " << p << "\n";
    std::string mmap_filename = "mmap_matrix.bin";
    memmap::MemoryMappedMatrix<double> mmap_matrix(
        mmap_filename, n, p, memmap::AccessMode::ReadWrite);
    auto X_map = mmap_matrix.getMap();

    // Step 2. Populate matrix
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (Eigen::Index j = 0; j < p; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) {
                X_map(i, j) = static_cast<double>(2.0 * static_cast<double>(i) +
                                                           static_cast<double>(j) / 100);
            }
        }
    }

    // Step 3. Fit and run LpNormScaler
    scaler::LpNormScaler l2scaler(scaler::LpNormScaler::NormType::L2, true);
    std::cout << "Fitting L2 LpNormScaler on memory-mapped matrix...\n";
    l2scaler.fit(X_map);
    std::cout << "✓ LpNormScaler fitted\n";
    l2scaler.transform_inplace(X_map);
    std::cout << "✓ Data transformed\n";
    l2scaler.inverse_transform_inplace(X_map);
    std::cout << "✓ Data inverse transformed\n";

    // Step 4. Clean up
    std::remove(mmap_filename.c_str());
    std::cout << "✓ Memory-mapped file removed\n";
    std::cout << "\n\n";
}



// ============================================================================
// Main
// ============================================================================

void init_omp_environment(int threads = 6) {
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, 0);
}


int main() {

    std::cout << "\n";
    cdiagnostics::print_section_header("DataTransformer Demo Suite");

    // Print OpenMP info
    omp_utils::print_info();
    init_omp_environment(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {
        demo_scaler_comparison();
        demo_scaler_inverse_transform();
        demo_scaler_serialization();
        demo_scaler_on_mmap_matrix();

        std::cout << "\n";
        cdiagnostics::print_section_header("All Demos Completed Successfully");

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
