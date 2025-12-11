#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <Eigen/Dense>

#include "Normalizer.hpp"
#include "StandardScaler.hpp"

#include "utils_openmp.hpp"
#include "utils_eval.hpp"
#include "utils_perf.hpp"
#include "utils_memmap.hpp"


// ============================================================================
// Demo 1: In-Memory Computations
// ============================================================================

void demo_scaler_comparison() {

    utils_eval::print_section_header("Scaler Comparison - StandardScaler vs Normalizers");

    // Step 1. Generate random data
    const std::size_t n = 2'000, p = 10'000;
    std::mt19937 rng(123);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }


    // Step 2. Create Map Wrapper
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);

    // Step 3. Fit Scalers
    StandardScaler scaler;
    Normalizer norm_l2(Normalizer::NormType::L2, true);
    Normalizer norm_l1(Normalizer::NormType::L1, true);

    auto t1 = utils_perf::profileit([&]() { scaler.fit(X_map); });
    auto t2 = utils_perf::profileit([&]() { norm_l2.fit(X_map); });
    auto t3 = utils_perf::profileit([&]() { norm_l1.fit(X_map); });

    // Step 4. Report Fit Performance
    utils_eval::print_section_header("Fit Performance");
    std::cout << std::left << std::setw(20) << "Scaler"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "RAM (MiB)" << "\n"
              << std::string(50, '-') << "\n"
              << std::setw(20) << "StandardScaler"
              << std::setw(15) << t1.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2)
              << t1.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L2"
              << std::setw(15) << t2.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2)
              << t2.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L1"
              << std::setw(15) << t3.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2)
              << t3.memory_delta.ram_mib() << "\n\n";

    // Transform performance (make copies to preserve original)
    utils_eval::print_section_header("Transform Performance");

    Eigen::MatrixXd X1 = X, X2 = X, X3 = X;
    Eigen::Map<Eigen::MatrixXd> X1_map(X1.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X3_map(X3.data(), n, p);

    auto tt1 = utils_perf::profileit([&]() { scaler.transform_inplace(X1_map); });
    auto tt2 = utils_perf::profileit([&]() { norm_l2.transform_inplace(X2_map); });
    auto tt3 = utils_perf::profileit([&]() { norm_l1.transform_inplace(X3_map); });

    std::cout << std::left << std::setw(20) << "Scaler"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "RAM (MiB)" << "\n"
              << std::string(50, '-') << "\n"
              << std::setw(20) << "StandardScaler"
              << std::setw(15) << tt1.time_ms
              << std::setw(15) << tt1.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L2"
              << std::setw(15) << tt2.time_ms
              << std::setw(15) << tt2.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L1"
              << std::setw(15) << tt3.time_ms
              << std::setw(15) << tt3.memory_delta.ram_mib() << "\n\n";

    std::cout << "\n\n";
}


// ============================================================================
// Demo 2: Transform and Inverse Transform Test
// ============================================================================

void demo_inverse_transform() {

    utils_eval::print_section_header("Demo 2: Inverse Transform Accuracy");

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

        StandardScaler scaler;
        scaler.fit(X_map);

        auto t_transform = utils_perf::profileit([&]() {
            scaler.transform_inplace(X_map);
        });
        auto t_inverse = utils_perf::profileit([&]() {
            scaler.inverse_transform_inplace(X_map);
        });

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "StandardScaler:\n"
                  << "  Transform:  " << t_transform.time_ms << " ms\n"
                  << "  Inverse:    " << t_inverse.time_ms << " ms\n"
                  << "  Max Error:  " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    // Test Normalizer L2
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);

        Normalizer norm_l2(Normalizer::NormType::L2, true);
        norm_l2.fit(X_map);

        auto t_transform = utils_perf::profileit([&]() {
            norm_l2.transform_inplace(X_map);
        });
        auto t_inverse = utils_perf::profileit([&]() {
            norm_l2.inverse_transform_inplace(X_map);
        });

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "Normalizer L2:\n"
                  << "  Transform:  " << t_transform.time_ms << " ms\n"
                  << "  Inverse:    " << t_inverse.time_ms << " ms\n"
                  << "  Max Error:  " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    // Test Normalizer L1
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);

        Normalizer norm_l1(Normalizer::NormType::L1, true);
        norm_l1.fit(X_map);

        auto t_transform = utils_perf::profileit([&]() {
            norm_l1.transform_inplace(X_map);
        });
        auto t_inverse = utils_perf::profileit([&]() {
            norm_l1.inverse_transform_inplace(X_map);
        });

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "Normalizer L1:\n"
                  << "  Transform:  " << t_transform.time_ms << " ms\n"
                  << "  Inverse:    " << t_inverse.time_ms << " ms\n"
                  << "  Max Error:  " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    std::cout << "\n\n";
}


// ============================================================================
// Demo 03: Memory Mapped Data
// ============================================================================

void demo_memory_mapped() {
    utils_eval::print_section_header("Demo 3: Scaler with Memory-Mapped Data");

    const std::size_t n = 5'000, p = 20'000;
    const std::size_t n_elements = n * p;
    const std::string filename = "scaler_mmap_data.bin";

    std::cout << "Matrix size: " << n << " x " << p << "\n";
    std::cout << "File: " << filename << "\n\n";

    // ===============================================
    // Step 1: Create and populate memory-mapped file
    // ===============================================

    std::cout << "Step 1: Creating and populating memory-mapped file...\n";

    auto create_time = utils_perf::profileit([&]() {
        auto mmap_handle = utils_memmap::create_empty_map<double>(filename.c_str(), n_elements);
        double* data_ptr = mmap_handle.data;

        // Fill with random data
        std::mt19937 rng(789);
        std::normal_distribution<double> norm_dist(0.0, 1.0);

        for (std::size_t i = 0; i < n_elements; ++i) {
            data_ptr[i] = norm_dist(rng);
        }

        // Flush to disk
        utils_memmap::flush_mapping(mmap_handle, true);
    });
    std::cout << "  Created and populated in " << create_time.time_ms << " ms\n\n";

    // ============================================================
    // Step 2: Load memory-mapped file and apply scaler (read-only)
    // =============================================================

    std::cout << "Step 2: Mapping file and fitting scaler..\n";

    auto mapped = utils_memmap::map_file_rw<double>(filename.c_str(), n_elements);
    Eigen::Map<Eigen::MatrixXd> X_map(mapped.data, n, p);

    StandardScaler scaler;
    auto fit_time = utils_perf::profileit([&]() {
        scaler.fit(X_map);
    });

    std::cout << "  Fitted scaler in " << fit_time.time_ms << " ms\n\n"
              << " (RAM delta: " << std::fixed << std::setprecision(2)
              << fit_time.memory_delta.ram_mib() << " MiB)\n\n";

    std::cout << "  Dropped " << scaler.get_dropped_indices().size()
              << " features due to zero variance.\n\n";

    // ============================================================
    // Step 3: Transform data in place
    // ============================================================

    std::cout << "Step 3: Transforming data in place...\n";

    auto transform_time = utils_perf::profileit([&]() {
        scaler.transform_inplace(X_map);  // Modifies mmap region = writes to file!
    });

    std::cout << "  Transformed in " << transform_time.time_ms << " ms "
              << "(RAM delta: " << transform_time.memory_delta.ram_mib() << " MiB)\n\n";

    // Verify transformation
    double mean_val = X_map.col(0).mean();
    double std_val = std::sqrt((X_map.col(0).array() - mean_val).square().sum() / (n - 1));
    std::cout << "  Verification (column 0):\n"
              << "    Mean: " << std::scientific << mean_val << "\n"
              << "    Std:  " << std_val << "\n\n";


    // ============================================================
    // Step 4: Inverse transform (restore original data)
    // ============================================================

    std::cout << "Step 4: Applying inverse transform...\n";

    auto inverse_time = utils_perf::profileit([&]() {
        scaler.inverse_transform_inplace(X_map);
    });

    std::cout << "  Inverse transformed in " << inverse_time.time_ms << " ms\n";
    std::cout << "  Data restored to original scale\n\n";

    // Flush changes to disk
    utils_memmap::flush_mapping(mapped, true);

    std::cout << "✓ Memory-mapped operations completed\n";
    std::cout << "✓ File written to: " << filename << "\n\n";

    // Clean up
    std::remove(filename.c_str());

}


// ============================================================================
// Demo 4: Serialization
// ============================================================================

void demo_serialization() {
    utils_eval::print_section_header("Demo 4: Serialization");

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
    StandardScaler scaler;
    scaler.fit(X_map);
    scaler.save(filename.c_str());
    std::cout << "✓ Saved StandardScaler to '" + filename + "'\n";

    // Load and verify
    StandardScaler loaded_scaler;
    loaded_scaler.load(filename.c_str());
    std::cout << "✓ Loaded StandardScaler from '" + filename + "'\n\n";

    // Verify means and scales match
    double mean_diff = (scaler.get_means() - loaded_scaler.get_means()).norm();
    double scale_diff = (scaler.get_scales() - loaded_scaler.get_scales()).norm();

    std::cout << "Verification:\n"
              << "  Mean difference:  " << std::scientific << mean_diff << "\n"
              << "  Scale difference: " << scale_diff << "\n\n";

    if (mean_diff < 1e-10 && scale_diff < 1e-10) {
        std::cout << "✓ Serialization perfect!\n\n";
    }

    // Clean up
    std::remove(filename.c_str());
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
    std::cout << "========================================\n";
    std::cout << "  DataTransformer Demo Suite\n";
    std::cout << "========================================\n\n";

    openmp::print_info();
    init_omp_environment(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {
        demo_scaler_comparison();
        demo_inverse_transform();
        demo_memory_mapped();
        demo_serialization();

        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  ✓ All demos completed successfully\n";
        std::cout << "========================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
