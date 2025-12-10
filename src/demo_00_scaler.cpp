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
#include "utils_talgos.hpp"


// ============================================================================
// Demo 1: In-Memory Computations
// ============================================================================

void demo_scaler_comparison() {

    utils_eval::print_section_header("Scaler Comparison - StandardScaler vs Normalizers");

    const std::size_t n = 2'000, p = 10'000;
    utils_eval::print_config(n, p, /*sigma=*/0, {}, {});

    // Generate random data
    std::mt19937 rng(123);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    // Create Map wrappers
    Eigen::Map<const Eigen::MatrixXd> X_map(X.data(), n, p);

    // Fit Scalers
    StandardScaler scaler;
    Normalizer norm_l2(Normalizer::NormType::L2, true);
    Normalizer norm_l1(Normalizer::NormType::L1, true);

    auto t1 = utils_perf::profileit([&]() { scaler.fit(X_map); });
    auto t2 = utils_perf::profileit([&]() { norm_l2.fit(X_map); });
    auto t3 = utils_perf::profileit([&]() { norm_l1.fit(X_map); });

    utils_eval::print_section_header("Fit Performance Comparison");
    std::cout << std::left << std::setw(20) << "Scaler"
              << std::setw(15) << "Fit Time (ms)"
              << std::setw(15) << "RAM (MiB)" << "\n"
              << std::string(50, '-') << "\n"
              << std::setw(20) << "StandardScaler" << std::setw(15)
              << t1.time_ms << std::setw(15) << t1.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L2"  << std::setw(15)
              << t2.time_ms << std::setw(15) << t2.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L1"  << std::setw(15)
              << t3.time_ms << std::setw(15) << t3.memory_delta.ram_mib() << "\n\n";


    // Copy based comparison
    utils_eval::print_section_header("Transform Performance Comparison");
    Eigen::MatrixXd X1 = X, X2 = X, X3 = X;
    Eigen::Map<Eigen::MatrixXd> X1_map(X1.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> X3_map(X3.data(), n, p);

    auto tt1 = utils_perf::profileit([&]() { scaler.transform_inplace(X1_map); });
    auto tt2 = utils_perf::profileit([&]() { norm_l2.transform_inplace(X2_map); });
    auto tt3 = utils_perf::profileit([&]() { norm_l1.transform_inplace(X3_map); });

    std::cout << std::left << std::setw(20) << "Scaler"
              << std::setw(20) << "Transform Time (ms)"
              << std::setw(15) << "RAM (MiB)" << "\n"
              << std::string(55, '-') << "\n"
              << std::setw(20) << "StandardScaler" << std::setw(20) << tt1.time_ms
              << std::setw(15) << tt1.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L2"  << std::setw(20) << tt2.time_ms
              << std::setw(15) << tt2.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L1"  << std::setw(20) << tt3.time_ms
              << std::setw(15) << tt3.memory_delta.ram_mib() << "\n\n";
}


// ============================================================================
// Demo 2: Transform and Inverse Transform Test
// ============================================================================

void demo_scaler_inverse_transform() {

    utils_eval::print_section_header("Scaler Inverse Transform Test");

    const std::size_t n = 1'000, p = 5'000;
    std::cout << "Testing with n=" << n << ", p=" << p << "\n\n";

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
        Eigen::Map<const Eigen::MatrixXd> X_orig_map(X_orig.data(), n, p);

        StandardScaler scaler;
        scaler.fit(X_orig_map);

        auto tt_transform = utils_perf::profileit([&]() { scaler.transform_inplace(X_map); });
        auto tt_inverse = utils_perf::profileit([&]() { scaler.inverse_transform_inplace(X_map); });

        // Check reconstruction error
        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "StandardScaler:\n"
                  << "  Transform:  " << tt_transform.time_ms << " ms\n"
                  << "  Inverse:    " << tt_inverse.time_ms << " ms\n"
                  << "  Max Error:  " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    // Test Normalizer L2
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        Eigen::Map<const Eigen::MatrixXd> X_orig_map(X_orig.data(), n, p);

        Normalizer norm_l2(Normalizer::NormType::L2, true);
        norm_l2.fit(X_orig_map);

        auto tt_transform = utils_perf::profileit([&]() { norm_l2.transform_inplace(X_map); });
        auto tt_inverse = utils_perf::profileit([&]() { norm_l2.inverse_transform_inplace(X_map); });

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "Normalizer L2:\n"
                  << "  Transform:  " << tt_transform.time_ms << " ms\n"
                  << "  Inverse:    " << tt_inverse.time_ms << " ms\n"
                  << "  Max Error:  " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    // Test Normalizer L1
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        Eigen::Map<const Eigen::MatrixXd> X_orig_map(X_orig.data(), n, p);

        Normalizer norm_l1(Normalizer::NormType::L1, true);
        norm_l1.fit(X_orig_map);

        auto tt_transform = utils_perf::profileit([&]() { norm_l1.transform_inplace(X_map); });
        auto tt_inverse = utils_perf::profileit([&]() { norm_l1.inverse_transform_inplace(X_map); });

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "Normalizer L1:\n"
                  << "  Transform:  " << tt_transform.time_ms << " ms\n"
                  << "  Inverse:    " << tt_inverse.time_ms << " ms\n"
                  << "  Max Error:  " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }
}


// ============================================================================
// Demo 3: Large Scale Performance Test
// ============================================================================

void demo_scaler_large_scale() {

    utils_eval::print_section_header("Large Scale Performance Test");

    const std::size_t n = 10'000, p = 50'000;
    const std::vector<std::size_t> support = {100, 5'000, 25'000};
    const std::vector<double> coefs = {2.5, -1.8, 3.2};
    double sigma = 0.5;

    utils_eval::print_config(n, p, sigma, support, coefs);

    // Generate sparse regression data
    std::mt19937 rng(789);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    utils_eval::print_section_header("Creating Dataset");
    Eigen::MatrixXd X(n, p);
    Eigen::VectorXd y(n);

    auto data_prof = utils_perf::profileit([&]() {
        // Generate X
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                X(i, j) = norm_dist(rng);
            }
        }

        // Generate y with sparse signal
        y.setZero();
        for (std::size_t idx = 0; idx < support.size(); ++idx) {
            y += X.col(support[idx]) * coefs[idx];
        }

        // Add noise
        for (std::size_t i = 0; i < n; ++i) {
            y(i) += sigma * norm_dist(rng);
        }
    });
    std::cout << "✓ Data creation: " << data_prof.time_ms << " ms\n\n";

    Eigen::Map<const Eigen::MatrixXd> X_map(X.data(), n, p);

    // Fit scalers
    StandardScaler scaler;
    Normalizer norm_l2(Normalizer::NormType::L2, true);
    Normalizer norm_l1(Normalizer::NormType::L1, true);

    auto t1 = utils_perf::profileit([&]() { scaler.fit(X_map); });
    auto t2 = utils_perf::profileit([&]() { norm_l2.fit(X_map); });
    auto t3 = utils_perf::profileit([&]() { norm_l1.fit(X_map); });

    utils_eval::print_section_header("Fit Performance (Large Scale)");
    std::cout << std::left << std::setw(20) << "Scaler"
              << std::setw(15) << "Fit Time (ms)"
              << std::setw(15) << "RAM (MiB)" << "\n"
              << std::string(50, '-') << "\n"
              << std::setw(20) << "StandardScaler"
              << std::setw(15) << t1.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2)
              << t1.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L2"
              << std::setw(15) << t2.time_ms
              << std::setw(15) << t2.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L1"
              << std::setw(15) << t3.time_ms
              << std::setw(15) << t3.memory_delta.ram_mib() << "\n\n";

    // Transform + Inverse performance
    utils_eval::print_section_header("Transform Performance (Large Scale)");

    // Test 1: StandardScaler
    Eigen::MatrixXd X1 = X;
    Eigen::Map<Eigen::MatrixXd> X1_map(X1.data(), n, p);
    auto tt1_transform = utils_perf::profileit([&]() { scaler.transform_inplace(X1_map); });
    auto tt1_inverse = utils_perf::profileit([&]() { scaler.inverse_transform_inplace(X1_map); });

    // Test 2: Normalizer L2
    Eigen::MatrixXd X2 = X;
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), n, p);
    auto tt2_transform = utils_perf::profileit([&]() { norm_l2.transform_inplace(X2_map); });
    auto tt2_inverse = utils_perf::profileit([&]() { norm_l2.inverse_transform_inplace(X2_map); });

    // Test 3: Normalizer L1
    Eigen::MatrixXd X3 = X;
    Eigen::Map<Eigen::MatrixXd> X3_map(X3.data(), n, p);
    auto tt3_transform = utils_perf::profileit([&]() { norm_l1.transform_inplace(X3_map); });
    auto tt3_inverse = utils_perf::profileit([&]() { norm_l1.inverse_transform_inplace(X3_map); });

    std::cout << std::left << std::setw(20) << "Scaler"
              << std::setw(20) << "Transform (ms)"
              << std::setw(20) << "Inverse (ms)"
              << std::setw(15) << "RAM (MiB)" << "\n"
              << std::string(75, '-') << "\n"
              << std::setw(20) << "StandardScaler"
              << std::setw(20) << tt1_transform.time_ms
              << std::setw(20) << tt1_inverse.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2)
              << tt1_transform.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L2"
              << std::setw(20) << tt2_transform.time_ms
              << std::setw(20) << tt2_inverse.time_ms
              << std::setw(15) << tt2_transform.memory_delta.ram_mib() << "\n"
              << std::setw(20) << "Normalizer L1"
              << std::setw(20) << tt3_transform.time_ms
              << std::setw(20) << tt3_inverse.time_ms
              << std::setw(15) << tt3_transform.memory_delta.ram_mib() << "\n\n";
}

// ============================================================================
// Demo 04: Memory Mapped Data
// ============================================================================



// ============================================================================
// Main
// ============================================================================

void init_omp_environment(int threads = 6) {
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, 0);
}

int main() {
    openmp::print_info();
    init_omp_environment(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {
        demo_scaler_comparison();

        demo_scaler_inverse_transform();

        demo_scaler_large_scale();

        std::cout << "\n✓ All scaler demos completed\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
