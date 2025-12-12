#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <Eigen/Dense>

#include "Normalizer.hpp"

#include "TLASSO_Solver.hpp"

#include "utils_openmp.hpp"
#include "utils_eval.hpp"
#include "utils_perf.hpp"
#include "utils_talgos.hpp"


// ============================================================================
// Demo 1: Basic T-LASSO with Early Stopping
// ============================================================================

void demo_TLASSO_early_stopping(bool high_dim, std::size_t T_stop = 5) {

    std::cout << "=== Demo 1: Basic T-LASSO with Early Stopping ===\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    std::size_t n, p;
    if (high_dim) {
        std::cout << "High-dimensional setting (p > n)" << "\n";
        n = 1000;
        p = 5000;
    } else {
        std::cout << "Low-dimensional setting (n > p)" << "\n";
        n = 5000;
        p = 1000;
    }
    const std::size_t num_dummies = 10 * p;
    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs = {-0.4, -0.2, -0.8, 1.1, 2.5};
    const double snr = 1.0;

    utils_talgos::print_talgo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr);
    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/42);


    // Create augmented matrix X_aug = [X | D]
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug.leftCols(p) = data.X;
    utils_talgos::generate_dummies_inplace(X_aug, n, p, num_dummies, 42);

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(), X_aug.rows(), X_aug.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    // Normalization is handled by the TLASSO solver internally
    TLASSO_Solver tlasso(X_aug_map, y_map, num_dummies, true, true, true);
    auto t1 = utils_perf::profileit([&]() { tlasso.executeStep(T_stop, /*early_stop=*/true); });
    std::cout << "T-LASSO early stopping at T=" << T_stop << " took " << t1.time_ms << " ms\n";

    utils_talgos::print_selection(tlasso, true_support);
    utils_talgos::print_quality(tlasso, true_support);

    // T-LASSO specific diagnostics
    std::cout << "\nT-LASSO Diagnostics:\n";
    std::cout << "  Removals: " << tlasso.getNumRemovals() << "\n";
    std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(4)
              << tlasso.getCyclingRatio() << "\n";

    std::cout << "\n\n";
}



// ============================================================================
// Demo 2: External Normalization with DataTransformer
// ============================================================================

void demo_TLASSO_with_external_normalizer(bool high_dim, std::size_t T_stop = 5) {

    std::cout << "=== Demo 2: T-LASSO with External Normalization ===\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    std::size_t n, p;
    if (high_dim) {
        std::cout << "High-dimensional setting (p > n)" << "\n";
        n = 500;
        p = 1000;
    } else {
        std::cout << "Low-dimensional setting (n > p)" << "\n";
        n = 1000;
        p = 500;
    }

    const std::size_t num_dummies = 10 * p;
    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs = {-0.4, -0.2, -0.8, 1.1, 2.5};
    const double snr = 1.0;

    utils_talgos::print_talgo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr);
    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/42);

    // Create augmented matrix X_aug = [X | D]
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug.leftCols(p) = data.X;

    utils_talgos::generate_dummies_inplace(X_aug, n, p, num_dummies, 42);

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(), X_aug.rows(), X_aug.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    // External normalization
    Normalizer normalizer(Normalizer::NormType::L2, true);
    normalizer.fit(X_aug_map);
    normalizer.transform_inplace(X_aug_map);
    y_map.array() -= y_map.mean();

    // Solver with external normalization
    TLASSO_Solver tlasso(X_aug_map, y_map, num_dummies, false, false, true);
    auto t1 = utils_perf::profileit([&]() { tlasso.executeStep(T_stop, /*early_stop=*/true); });
    std::cout << "T-LASSO early stopping at T=" << T_stop << " took " << t1.time_ms << " ms\n";

    utils_talgos::print_selection(tlasso, true_support);
    utils_talgos::print_quality(tlasso, true_support);

    // T-LASSO specific diagnostics
    std::cout << "\nT-LASSO Diagnostics:\n";
    std::cout << "  Removals: " << tlasso.getNumRemovals() << "\n";
    std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(4)
              << tlasso.getCyclingRatio() << "\n";

    std::cout << "\n\n";

}



// ============================================================================
// Demo 3: Serialization
// ============================================================================

void demo_TLASSO_serialization() {

    std::cout << "=== Demo 3: T-LASSO Serialization & Path Consistency ===\n\n";

    const std::size_t n = 100, p = 50;
    const std::size_t num_dummies = p;
    const std::size_t T_stop_final = 15;
    const std::size_t T_stop_partial = 7;
    const std::vector<std::size_t> true_support = {10, 25, 40};
    const std::vector<double> true_coefs = {2.5, -1.8, 3.2};

    utils_talgos::print_talgo_config(n, p, num_dummies, T_stop_final, true_support, true_coefs);

    utils_talgos::SyntheticData data(n, p, true_support, true_coefs);

    // Create augmented matrix X_aug = [X | D]
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug.leftCols(p) = data.X;
    utils_talgos::generate_dummies_inplace(X_aug, n, p, num_dummies, 42);

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(), X_aug.rows(), X_aug.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    // First, create reference solver, run to full solution path
    TLASSO_Solver solver_ref(X_aug_map, y_map, num_dummies, true, true, true);
    solver_ref.executeStep(T_stop_final, /*early_stop=*/true);

    std::string filename = "tlasso_checkpoint.bin";

    // --------- STEP 1: Run partial path and save checkpoint (scoped) ---------
    {
        TLASSO_Solver solver1(X_aug_map, y_map, num_dummies, true, true, true);
        solver1.executeStep(T_stop_partial, /*early_stop=*/true);

        std::cout << "Checkpoint at partial stop: " << solver1.getNumSteps() << " steps\n";
        const char* checkpoint = filename.c_str();
        solver1.save(checkpoint);
        std::cout << "✓ Checkpoint saved to '" << checkpoint << "'\n";
    }

    // --------- STEP 2: Load from checkpoint and continue (scoped) ---------
    {
        TLASSO_Solver solver2 = TLASSO_Solver::load(filename.c_str(), X_aug_map);
        std::cout << "Loaded from checkpoint: "
                  << solver2.getNumSteps() << " steps\n";
        solver2.executeStep(T_stop_final, /*early_stop=*/true); // Continue to full path

        // --------- Comparison to reference ---------
        std::cout << "\nCOMPARISON:\n";
        std::cout << "Reference steps: " << solver_ref.getNumSteps() << "\n";
        std::cout << "Reloaded steps:  " << solver2.getNumSteps() << "\n";
        std::cout << "RSS diff:   "
                  << std::abs(solver_ref.getRSS().back() - solver2.getRSS().back()) << "\n";
        std::cout << "R2 diff:    "
                  << std::abs(solver_ref.getR2().back() - solver2.getR2().back()) << "\n";
        auto& ref_path = solver_ref.getActions();
        auto& loaded_path = solver2.getActions();
        std::cout << (ref_path == loaded_path ? "✓ Paths match\n" : "✗ Paths differ!\n");

        utils_talgos::print_selection(solver2, true_support);
        utils_talgos::print_quality(solver2, true_support);

        // T-LASSO specific diagnostics
        std::cout << "\nT-LASSO Diagnostics:\n";
        std::cout << "  Removals: " << solver2.getNumRemovals() << "\n";
        std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(4)
                  << solver2.getCyclingRatio() << "\n";
    }

    // Cleanup
    std::remove(filename.c_str());
    std::cout << "✓ Checkpoint file removed\n";

    std::cout << "\n\n";
}



// ============================================================================
// Demo 4: Controlled Comparison In-Memory vs Memory-Mapped
// ============================================================================

void demo_TLASSO_controlled_comparison() {
    std::cout << "=== Demo 4: Controlled Comparison: In-Memory vs Memory-Mapped ===\n\n";

    // Configuration
    const std::size_t n = 1000, p = 5000;
    const std::size_t num_dummies = 50 * p;
    const std::size_t T_stop = 10;
    std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    std::vector<double> true_support_coefs = {-0.4, -0.2, -0.8, 1.1, 2.5};


    // ============================================================
    // Step 1: Generate data in-memory
    // ============================================================
    std::cout << "\n=== Step 1: Generating data in-memory ===\n";
    utils_talgos::SyntheticData data(n, p, true_support, true_support_coefs, 1.0, 42);

    // Create X_aug in-memory
    Eigen::MatrixXd X_aug_inmem(n, p + num_dummies);
    X_aug_inmem.leftCols(p) = data.X;
    utils_talgos::generate_dummies_inplace(X_aug_inmem, n, p, num_dummies, 42);

    Eigen::VectorXd y_inmem = data.y;

    std::cout << "  ✓ In-memory data created\n\n";


    // ============================================================
    // Step 2: Writing same data to memory-mapped files
    // ============================================================
    std::cout << "\n=== Step 2: Writing same data to memory-mapped files ===\n";

    const std::string X_aug_file = "test_X_aug.bin";
    const std::string y_file = "test_y.bin";

    auto X_aug_mmap = utils_memmap::create_empty_map<double>(X_aug_file.c_str(),
                                                             n * (p + num_dummies));
    auto y_mmap = utils_memmap::create_empty_map<double>(y_file.c_str(), n);

    // Copy data
    std::memcpy(X_aug_mmap.data, X_aug_inmem.data(), n * (p + num_dummies) * sizeof(double));
    std::memcpy(y_mmap.data, y_inmem.data(), n * sizeof(double));

    utils_memmap::flush_mapping(X_aug_mmap);
    utils_memmap::flush_mapping(y_mmap);

    std::cout << "  ✓ Memory-mapped files created\n\n";


    // ============================================================
    // Step 3: Creating Eigen::Map views
    // ============================================================
    std::cout << "\n=== Step 3: Creating Eigen::Map views ===\n";

    Eigen::Map<Eigen::MatrixXd> X_aug_inmem_map(X_aug_inmem.data(), n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y_inmem_map(y_inmem.data(), n);

    Eigen::Map<Eigen::MatrixXd> X_aug_mmap_map(X_aug_mmap.data, n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y_mmap_map(y_mmap.data, n);


    // ============================================================
    // Step 4: Applying external normalization
    // ============================================================
    std::cout << "\n=== Step 4: Applying external normalization ===\n";

    // In-memory normalization
    Normalizer norm1(Normalizer::NormType::L2, false);
    norm1.fit(X_aug_inmem_map);
    norm1.transform_inplace(X_aug_inmem_map);
    y_inmem_map.array() -= y_inmem_map.mean();

    // Memory-mapped normalization
    Normalizer norm2(Normalizer::NormType::L2, false);
    norm2.fit(X_aug_mmap_map);
    norm2.transform_inplace(X_aug_mmap_map);
    y_mmap_map.array() -= y_mmap_map.mean();

    std::cout << "  ✓ Both normalized\n\n";

    // Verify data is still identical
    std::cout << "Verifying data identity...\n";
    double X_diff = (X_aug_inmem_map - X_aug_mmap_map).norm();
    double y_diff = (y_inmem_map - y_mmap_map).norm();
    std::cout << "  ||X_inmem - X_mmap|| = " << X_diff << "\n";
    std::cout << "  ||y_inmem - y_mmap|| = " << y_diff << "\n\n";


    // ============================================================
    // Step 5: Running T-LASSO for both versions
    // ============================================================
    std::cout << "\n=== Step 5: Running T-LASSO on in-memory data ===\n";
    TLASSO_Solver tlasso_inmem(X_aug_inmem_map, y_inmem_map, num_dummies, false, false, true);
    tlasso_inmem.executeStep(T_stop, true);
    std::cout << "  ✓ In-memory T-LASSO complete\n\n";

    std::cout << "Running T-LASSO on memory-mapped data...\n";
    TLASSO_Solver tlasso_mmap(X_aug_mmap_map, y_mmap_map, num_dummies, false, false, true);
    tlasso_mmap.executeStep(T_stop, true);
    std::cout << "  ✓ Memory-mapped T-LASSO complete\n\n";


    // ============================================================
    // Step 6: Compare results
    // ============================================================
    std::cout << "\n=== Step 6: Comparing results ===\n";

    auto path_inmem = tlasso_inmem.getActions();
    auto path_mmap = tlasso_mmap.getActions();

    std::cout << "In-memory path:\n  ";
    utils_talgos::print_selection(tlasso_inmem, true_support);

    std::cout << "\nMemory-mapped path:\n  ";
    utils_talgos::print_selection(tlasso_mmap, true_support);

    std::cout << "\nPaths match: " << (path_inmem == path_mmap ? "YES ✓" : "NO ✗") << "\n";

    // Cleanup
    std::remove(X_aug_file.c_str());
    std::remove(y_file.c_str());

}


// ============================================================================
// Demo 5: Production T-LASSO Workflow
// ============================================================================

void demo_production_TLASSO_workflow() {

    utils_eval::print_section_header("=== Demo 5: Production T-LASSO Workflow ===\n\n");

    // Configuration
    const std::size_t n = 20'000, p = 100'000;
    const std::size_t num_dummies = 10 * p;
    const std::size_t T_stop = 10;
    std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const bool strong_coefs = true;
    std::vector<double> true_support_coefs;
    if (strong_coefs) {
        // strong coefficients
        true_support_coefs = {1, 1, 1, 1, 1};
    } else {
        // weak coefficients
        true_support_coefs = {-0.4, -0.2, -0.8, 1.1, 2.5};
    }

    const std::string X_file = "production_X.bin";
    const std::string y_file = "production_y.bin";
    const std::string X_aug_file = "production_X_aug.bin";


    // ========================================================================
    // Step 1: Create X and y on disk (simulating "already existing" data)
    // ========================================================================

    std::cout << "\n=== Step 1: Creating X and y (simulating existing data) ===\n";

    utils_talgos::create_mmapped_X_and_y(
        X_file.c_str(),
        y_file.c_str(),
        n, p,
        true_support,
        true_support_coefs,
        1.0,   // SNR
        42,    // seed
        false  // reproducible_noise
    );

    std::cout << "✓ X and y now exist on disk\n";
    std::cout << "  (In real use case, these would already exist)\n\n";

    // ========================================================================
    // Step 2: Augment X with dummies: X_aug = [X | D]
    // ========================================================================
    std::cout << "=== Step 2: Creating X_aug = [X | D] ===\n";

    auto X_aug_handle = utils_talgos::augment_with_dummies(
        X_file.c_str(),
        X_aug_file.c_str(),
        n, p,
        num_dummies,
        42  // seed for dummies
    );

    // ========================================================================
    // Step 3: Load y and create Eigen maps
    // ========================================================================
    std::cout << "\n=== Step 3: Loading y and creating maps ===\n";

    auto y_handle = utils_memmap::map_file_rw<double>(y_file.c_str(), n);

    Eigen::Map<Eigen::MatrixXd> X_aug(X_aug_handle.data, n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y(y_handle.data, n);

    std::cout << "  X_aug: " << X_aug.rows() << " x " << X_aug.cols() << "\n";
    std::cout << "  y: " << y.size() << "\n";

    // ========================================================================
    // Step 4: External normalization
    // ========================================================================
    std::cout << "\n=== Step 4: Normalizing data ===\n";

    Normalizer normalizer(Normalizer::NormType::L2, false);
    normalizer.fit(X_aug);
    normalizer.transform_inplace(X_aug);
    y.array() -= y.mean();

    std::cout << "  ✓ X_aug normalized (L2)\n";
    std::cout << "  ✓ y centered\n";

    // Verification
    std::cout << "\nVerification:\n";
    for (std::size_t idx : true_support) {
        double corr = X_aug.col(idx).dot(y) / (X_aug.col(idx).norm() * y.norm());
        std::cout << "  Corr(col " << idx << ", y) = " << corr << "\n";
    }

    // ========================================================================
    // Step 5: Run T-LASSO
    // ========================================================================
    std::cout << "\n=== Step 5: Running T-LASSO ===\n";

    TLASSO_Solver tlasso(X_aug, y, num_dummies, false, false, true);

    auto t1 = utils_perf::profileit([&]() {
        tlasso.executeStep(T_stop, true);
    });

    std::cout << "T-LASSO completed in " << t1.time_ms << " ms\n\n";

    // Results
    utils_talgos::print_selection(tlasso, true_support);
    utils_talgos::print_quality(tlasso, true_support);

    // ========================================================================
    // Step 6: Cleanup
    // ========================================================================
    std::cout << "\n=== Step 6: Cleanup ===\n";
    std::remove(X_file.c_str());
    std::remove(y_file.c_str());
    std::remove(X_aug_file.c_str());
    std::cout << "✓ All files removed\n";

    std::cout << "\n\n";
}



// ===========================================================================
// Main
// ===========================================================================

int main() {
    openmp::print_info();
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {

        // T-LASSO with early stopping - low and high dimensional
        const bool run_early_stopping_demo = false;
        if (run_early_stopping_demo) {
            demo_TLASSO_early_stopping(/*high_dim=*/false, /*T_stop=*/10);
            demo_TLASSO_early_stopping(/*high_dim=*/true, /*T_stop=*/10);
        }


        // T-LASSO with external normalization
        const bool run_external_normalizer_demo = false;
        if (run_external_normalizer_demo) {
            demo_TLASSO_with_external_normalizer(/*high_dim=*/false, /*T_stop=*/5);
            demo_TLASSO_with_external_normalizer(/*high_dim=*/true, /*T_stop=*/5);
        }


        // T-LASSO serialization and warm-start
        const bool run_serialization_demo = false;
        if (run_serialization_demo) demo_TLASSO_serialization();


        // T-LASSO controlled comparison
        const bool run_controlled_comparison = false;
        if (run_controlled_comparison) demo_TLASSO_controlled_comparison();


        // T-LASSO production workflow
        const bool run_production_tlasso_with_mmap = true;
        if (run_production_tlasso_with_mmap) demo_production_TLASSO_workflow();


        std::cout << "\n✓ All T-LASSO demos completed\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
