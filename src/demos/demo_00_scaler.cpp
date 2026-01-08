// ===================================================================================
// demo_00_scaler.cpp
// ===================================================================================
/**
 * @file demo_00_scaler.cpp
 *
 * @brief Demonstration of StandardScaler (Zscore) and Normalizer usage.
 */
// ===================================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/Normalizer.hpp>
#include <ml_methods/StandardScaler.hpp>
#include <utils/openMP/utils_openmp.hpp>
#include <utils/memmap/MemoryMappedMatrix.hpp>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace ml_methods = trex::ml_methods;
namespace omp_utils = trex::utils::openmp;
namespace memmap = trex::utils::memmap;

// ============================================================================
// Demo 1: In-Memory Computations
// ============================================================================

void demo_scaler_comparison() {

    std::cout << std::string(60, '=') << "\n";
    std::cout << "Scaler Comparison - StandardScaler vs Normalizers\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Step 1. Generate random data
    const std::size_t n = 2'000, p = 10'000;
    std::mt19937 rng(123);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    // Setup data matrix
    Eigen::MatrixXd X(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }


    // Step 2. Create Map Wrapper
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);


    // Step 3. Fit Scalers
    ml_methods::StandardScaler zscaler;
    ml_methods::Normalizer norm_l2_scaler(ml_methods::Normalizer::NormType::L2, true);
    ml_methods::Normalizer norm_l1_scaler(ml_methods::Normalizer::NormType::L1, true);

    std::cout << "Fitting scalers...\n";
    zscaler.fit(X_map);
    norm_l2_scaler.fit(X_map);
    norm_l1_scaler.fit(X_map);
    std::cout << "✓ Scalers fitted\n\n";


    // Step 4. Transform performance
    std::cout << "Transforming data...\n";
    Eigen::MatrixXd X1 = X, X2 = X, X3 = X;
    Eigen::Map<Eigen::MatrixXd> X1_map(X1.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X3_map(X3.data(), n, p);

    zscaler.transform_inplace(X1_map);
    norm_l2_scaler.transform_inplace(X2_map);
    norm_l1_scaler.transform_inplace(X3_map);
    std::cout << "✓ Data transformed\n\n";

    std::cout << "\n\n";
}


// =============================================================================
// Demo 2: Transform and Inverse Transform Test
// =============================================================================

void demo_scaler_inverse_transform() {

    std::cout << std::string(60, '=') << "\n";
    std::cout << "Demo 2: Inverse Transform Accuracy\n";
    std::cout << std::string(60, '=') << "\n\n";

    const std::size_t n = 1'000, p = 5'000;
    std::cout << "Matrix size: " << n << " x " << p << "\n\n";

    // Generate random data
    std::mt19937 rng(456);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    Eigen::MatrixXd X_orig(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X_orig(i, j) = norm_dist(rng);
        }
    }

    // Test StandardScaler
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        ml_methods::StandardScaler scaler;

        scaler.fit(X_map);
        scaler.transform_inplace(X_map);
        scaler.inverse_transform_inplace(X_map);

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "StandardScaler:\n"
                  << "  Max Error: " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    // Test Normalizer L2
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        ml_methods::Normalizer norm_l2(ml_methods::Normalizer::NormType::L2, true, true);

        norm_l2.fit(X_map);
        norm_l2.transform_inplace(X_map);
        norm_l2.inverse_transform_inplace(X_map);

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "Normalizer L2:\n"
                  << "  Max Error: " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }
    std::cout << "\n\n";
}


// =============================================================================
// Demo 3: Serialization
// =============================================================================

void demo_scaler_serialization() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Demo 3: Serialization\n";
    std::cout << std::string(60, '=') << "\n\n";

    const std::size_t n = 1'000, p = 500;

    // Generate data and fit scaler
    std::mt19937 rng(999);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    Eigen::MatrixXd X(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);

    // Fit and save
    std::string filename = "scaler_test.bin";
    ml_methods::StandardScaler scaler;
    scaler.fit(X_map);
    scaler.save(filename);
    std::cout << "✓ Saved StandardScaler to '" << filename << "'\n";

    // Load and verify
    ml_methods::StandardScaler loaded_scaler;
    loaded_scaler.load(filename);
    std::cout << "✓ Loaded StandardScaler from '" << filename << "'\n\n";

    // Verify means and scales match
    double mean_diff = (scaler.get_means() - loaded_scaler.get_means()).norm();
    double scale_diff = (scaler.get_scales() - loaded_scaler.get_scales()).norm();

    std::cout << "Verification:\n"
              << "  Mean difference: " << std::scientific << mean_diff << "\n"
              << "  Scale difference: " << scale_diff << "\n\n";

    if (mean_diff < 1e-10 && scale_diff < 1e-10) {
        std::cout << "✓ Serialization perfect!\n\n";
    }

    // Clean up
    std::remove(filename.c_str());
    std::cout << "\n\n";
}


// ============================================================================
// Demo 04: Memory-Mapped Matrix
// ============================================================================

void demo_scaler_on_mmap_matrix() {

    std::cout << std::string(60, '=') << "\n";
    std::cout << "L2 Normalizer on Memory-Mapped Matrix\n";
    std::cout << std::string(60, '=') << "\n\n";

    std::cout << "Creating memory-mapped matrix...\n";
    const std::size_t n = 20'000, p = 100'000;
    std::cout << "Matrix size: " << n << " x " << p << "\n";
    std::string mmap_filename = "mmap_matrix.bin";
    memmap::MemoryMappedMatrix<double> mmap_matrix(
        mmap_filename, n, p, memmap::AccessMode::ReadWrite);
    auto X_map = mmap_matrix.getMap();
    #pragma omp parallel
    {
        // Populate matrix in major-column order
        #pragma omp for schedule(static)
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                X_map(i, j) = static_cast<double>(2 * i + j / 100);
            }
        }
    }

    // Setup & Run Normalizer
    ml_methods::Normalizer norm_l2_scaler(ml_methods::Normalizer::NormType::L2, true);
    std::cout << "Fitting L2 Normalizer on memory-mapped matrix...\n";
    norm_l2_scaler.fit(X_map);
    std::cout << "✓ Normalizer fitted\n";
    norm_l2_scaler.transform_inplace(X_map);
    std::cout << "✓ Data transformed\n";
    norm_l2_scaler.inverse_transform_inplace(X_map);
    std::cout << "✓ Data inverse transformed\n";

    // Clean up
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
    std::cout << std::string(60, '=') << "\n";
    std::cout << "DataTransformer Demo Suite\n";
    std::cout << std::string(60, '=') << "\n\n";

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
        std::cout << std::string(60, '=') << "\n";
        std::cout << "  ✓ All demos completed successfully\n";
        std::cout << std::string(60, '=') << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
