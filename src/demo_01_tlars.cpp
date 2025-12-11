#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <Eigen/Dense>

#include "Normalizer.hpp"

#include "TLARS_Solver.hpp"

#include "utils_openmp.hpp"
#include "utils_eval.hpp"
#include "utils_perf.hpp"
#include "utils_talgos.hpp"


// Add to utils_talgos.hpp
inline void generate_dummies_inplace(
    Eigen::MatrixXd& X_aug,
    std::size_t n,
    std::size_t p,
    std::size_t num_dummies,
    unsigned int seed = 42)
{
    for (std::size_t j = 0; j < num_dummies; ++j) {
        std::mt19937 rng(seed + 1000003UL + j);  // seed per column
        std::normal_distribution<double> normal(0.0, 1.0);

        for (std::size_t i = 0; i < n; ++i) {
            X_aug(i, p + j) = normal(rng);
        }
    }
}


// ============================================================================
// Demo 1: Basic T-LARS with Early Stopping
// ============================================================================

void demo_TLARS_early_stopping(bool high_dim, std::size_t T_stop = 5) {

    std::cout << "Demo 1: Basic T-LARS with Early Stopping" << "\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> rnorm(0.0, 1.0);

    std::size_t n, p;
    if (high_dim) {
        std::cout << "High-dimensional setting (p > n)" << "\n";
        n = 5000;
        p = 1000;
    } else {
        std::cout << "Low-dimensional setting (n > p)" << "\n";
        n = 1000;
        p = 5000;
    }
    const std::size_t num_dummies = 10 * p;
    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs = {-0.4, -0.2, -0.8, 1.1, 2.5};
    const double snr = 1.0;

    utils_talgos::print_tlars_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr);

    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/42);

    // Create augmented matrix X_aug = [X | D]
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug.leftCols(p) = data.X;
    generate_dummies_inplace(X_aug, n, p, num_dummies, 42);

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(),
                                          X_aug.rows(),
                                          X_aug.cols());

    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    TLARS_Solver tlars(X_aug_map, y_map, num_dummies, true, true, true);
    auto t1 = utils_perf::profileit([&]() { tlars.executeStep(T_stop, /*early_stop=*/true); });
    std::cout << "T-LARS early stopping at T=" << T_stop << " took " << t1.time_ms << " ms\n";

    utils_talgos::print_selection(tlars, true_support);
    utils_talgos::print_quality(tlars, true_support);

    std::cout << "\n\n";
}



// ============================================================================
// Demo 2: External Normalization with DataTransformer
// ============================================================================

void demo_TLARS_with_external_normalizer(bool high_dim,
                                         bool normalize_internal,
                                         std::size_t T_stop = 5) {

    std::cout << "Demo 2: T-LARS with External Normalization\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

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
    const std::vector<std::size_t> true_support = {10, 25, 40};
    const std::vector<double> true_coefs = {0.5, -0.8, 1.2};
    const double snr = 1.0;

    utils_talgos::print_tlars_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr);

    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/42);

    // Create augmented matrix X_aug = [X | dummies]
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug.leftCols(p) = data.X;

    generate_dummies_inplace(X_aug, n, p, num_dummies, 42);

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(),
                                          X_aug.rows(),
                                          X_aug.cols());

    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    Normalizer normalizer(Normalizer::NormType::L2, true);
    normalizer.fit(X_aug_map);
    normalizer.transform_inplace(X_aug_map);

    TLARS_Solver tlars(X_aug_map, y_map, num_dummies, false, false, true);
    auto t1 = utils_perf::profileit([&]() { tlars.executeStep(T_stop, /*early_stop=*/true); });
    std::cout << "T-LARS early stopping at T=" << T_stop << " took " << t1.time_ms << " ms\n";

    utils_talgos::print_selection(tlars, true_support);
    utils_talgos::print_quality(tlars, true_support);

    std::cout << "\n\n";
}



// ============================================================================
// Demo 3: Serialization
// ============================================================================

void demo_TLARS_serialization() {

    std::cout << "Demo 3: T-LARS Serialization & Path Consistency\n";

    const std::size_t n = 100, p = 50;
    const std::size_t num_dummies = p;
    const std::size_t T_stop = 15;
    const std::vector<std::size_t> true_support = {10, 25, 40};
    const std::vector<double> true_coefs = {2.5, -1.8, 3.2};

    utils_talgos::print_tlars_config(n, p, num_dummies, T_stop, true_support, true_coefs);

    utils_talgos::SyntheticData data(n, p, true_support, true_coefs);

    // Create augmented matrix X_aug = [X | D]
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug.leftCols(p) = data.X;
    generate_dummies_inplace(X_aug, n, p, num_dummies, 42);

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(), X_aug.rows(),
                                          X_aug.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    // First, create reference solver, run to full solution path
    TLARS_Solver solver_ref(X_aug_map, y_map, num_dummies, true, true, true);
    solver_ref.executeStep(T_stop, /*early_stop=*/true);

    std::string filename = "tlars_checkpoint.bin";

    // --------- PART 1: Run partial path and save checkpoint (scoped) ---------
    {
        TLARS_Solver solver1(X_aug_map, y_map, num_dummies, true, true, true);
        solver1.executeStep(/*T_stop=*/7, /*early_stop=*/true);

        std::cout << "Checkpoint at partial stop: "
                  << solver1.getNumSteps() << " steps\n";
        const char* checkpoint = filename.c_str();
        solver1.save(checkpoint);
        std::cout << "✓ Checkpoint saved to '" << checkpoint << "'\n";
    }

    // --------- PART 2: Load from checkpoint and continue (scoped) ---------
    TLARS_Solver solver2(X_aug_map, y_map, num_dummies, true, true, true);
    {
        TLARS_Solver solver2 = TLARS_Solver::load(filename.c_str(), X_aug_map);
        std::cout << "Loaded from checkpoint: "
                  << solver2.getNumSteps() << " steps\n";
        solver2.executeStep(T_stop, /*early_stop=*/true); // Continue to full path

        // --------- Comparison to reference ---------
        std::cout << "\nCOMPARISON:\n";
        std::cout << "Reference steps: " << solver_ref.getNumSteps() << "\n";
        std::cout << "Reloaded steps:  " << solver2.getNumSteps() << "\n";
        std::cout << "RSS diff:   "
                  << std::abs(solver_ref.getRSS().back() - solver2.getRSS().back())
                  << "\n";
        std::cout << "R2 diff:    "
                  << std::abs(solver_ref.getR2().back() - solver2.getR2().back())
                  << "\n";
        auto& ref_path = solver_ref.getActions();
        auto& loaded_path = solver2.getActions();
        std::cout << (ref_path == loaded_path ? "✓ Paths match\n" :
                                                "✗ Paths differ!\n");

        utils_talgos::print_selection(solver2, true_support);
        utils_talgos::print_quality(solver2, true_support);
    }

    // Cleanup
    std::remove(filename.c_str());
    std::cout << "✓ Checkpoint file removed\n";

    std::cout << "\n\n";
}



// ============================================================================
// Demo 4: Controlled Comparison In-Memory vs Memory-Mapped
// ============================================================================

void demo_TLARS_controlled_comparison() {
    std::cout << "CONTROLLED COMPARISON: In-Memory vs Memory-Mapped ===\n\n";

    // Configuration
    const std::size_t n = 1000, p = 5000;
    const std::size_t num_dummies = 50 * p;
    const std::size_t T_stop = 10;
    std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    std::vector<double> true_support_coefs = {-0.4, -0.2, -0.8, 1.1, 2.5};


    // ============================================================
    // STEP 1: Generate data in-memory (like working demos)
    // ============================================================
    std::cout << "Step 1: Generating data in-memory...\n";
    utils_talgos::SyntheticData data(n, p, true_support, true_support_coefs, 1.0, 42);

    // Create X_aug in-memory
    Eigen::MatrixXd X_aug_inmem(n, p + num_dummies);
    X_aug_inmem.leftCols(p) = data.X;
    generate_dummies_inplace(X_aug_inmem, n, p, num_dummies, 42);

    Eigen::VectorXd y_inmem = data.y;

    std::cout << "  ✓ In-memory data created\n\n";


    // ============================================================
    // STEP 2: Write SAME data to memory-mapped files
    // ============================================================
    std::cout << "Step 2: Writing same data to memory-mapped files...\n";

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
    // STEP 3: Create Eigen::Maps
    // ============================================================
    Eigen::Map<Eigen::MatrixXd> X_aug_inmem_map(X_aug_inmem.data(), n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y_inmem_map(y_inmem.data(), n);

    Eigen::Map<Eigen::MatrixXd> X_aug_mmap_map(X_aug_mmap.data, n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y_mmap_map(y_mmap.data, n);


    // ============================================================
    // STEP 4: External normalization (both versions)
    // ============================================================
    std::cout << "Step 3: Applying external normalization...\n";

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
    std::cout << "Step 4: Verifying data identity...\n";
    double X_diff = (X_aug_inmem_map - X_aug_mmap_map).norm();
    double y_diff = (y_inmem_map - y_mmap_map).norm();
    std::cout << "  ||X_inmem - X_mmap|| = " << X_diff << "\n";
    std::cout << "  ||y_inmem - y_mmap|| = " << y_diff << "\n\n";


    // ============================================================
    // STEP 5: Run T-LARS on BOTH
    // ============================================================
    std::cout << "Step 5: Running T-LARS on in-memory data...\n";
    TLARS_Solver tlars_inmem(X_aug_inmem_map, y_inmem_map, num_dummies, false, false, true);
    tlars_inmem.executeStep(T_stop, true);
    std::cout << "  ✓ In-memory T-LARS complete\n\n";

    std::cout << "Step 6: Running T-LARS on memory-mapped data...\n";
    TLARS_Solver tlars_mmap(X_aug_mmap_map, y_mmap_map, num_dummies, false, false, true);
    tlars_mmap.executeStep(T_stop, true);
    std::cout << "  ✓ Memory-mapped T-LARS complete\n\n";


    // ============================================================
    // STEP 6: Compare results
    // ============================================================
    std::cout << "=== COMPARISON ===\n";

    auto path_inmem = tlars_inmem.getActions();
    auto path_mmap = tlars_mmap.getActions();

    std::cout << "In-memory path:\n  ";
    utils_talgos::print_selection(tlars_inmem, true_support);

    std::cout << "\nMemory-mapped path:\n  ";
    utils_talgos::print_selection(tlars_mmap, true_support);

    std::cout << "\nPaths match: " << (path_inmem == path_mmap ? "YES ✓" : "NO ✗") << "\n";

    // Cleanup
    std::remove(X_aug_file.c_str());
    std::remove(y_file.c_str());

}


// ============================================================================
// Demo 5: Production T-LARS Workflow
// ============================================================================

void demo_production_tlars_workflow() {

    utils_eval::print_section_header("Production T-LARS Workflow");

    // Configuration
    const std::size_t n = 1000, p = 5000;
    const std::size_t num_dummies = 50 * p;
    const std::size_t T_stop = 10;
    std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
        // strong coefficients
    std::vector<double> true_support_coefs = {1, 1, 1, 1, 1};
        // weak coefficients
//    std::vector<double> true_support_coefs = {-0.4, -0.2, -0.8, 1.1, 2.5};

    const std::string X_file = "production_X.bin";
    const std::string y_file = "production_y.bin";
    const std::string X_aug_file = "production_X_aug.bin";


    // ========================================================================
    // PHASE 1: Create X and y on disk (simulating "already existing" data)
    // ========================================================================

    std::cout << "\n=== PHASE 1: Creating X and y (simulating existing data) ===\n";

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
    // PHASE 2: Augment X with dummies: X_aug = [X | D]
    // ========================================================================
    std::cout << "=== PHASE 2: Creating X_aug = [X | D] ===\n";

    auto X_aug_handle = utils_talgos::augment_with_dummies(
        X_file.c_str(),
        X_aug_file.c_str(),
        n, p,
        num_dummies,
        42  // seed for dummies
    );

    // ========================================================================
    // PHASE 3: Load y and create Eigen maps
    // ========================================================================
    std::cout << "\n=== PHASE 3: Loading y and creating maps ===\n";

    auto y_handle = utils_memmap::map_file_rw<double>(y_file.c_str(), n);

    Eigen::Map<Eigen::MatrixXd> X_aug(X_aug_handle.data, n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y(y_handle.data, n);

    std::cout << "  X_aug: " << X_aug.rows() << " x " << X_aug.cols() << "\n";
    std::cout << "  y: " << y.size() << "\n";

    // ========================================================================
    // PHASE 4: External normalization
    // ========================================================================
    std::cout << "\n=== PHASE 4: Normalizing data ===\n";

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
    // PHASE 5: Run T-LARS
    // ========================================================================
    std::cout << "\n=== PHASE 5: Running T-LARS ===\n";

    TLARS_Solver tlars(X_aug, y, num_dummies, false, false, true);

    auto t1 = utils_perf::profileit([&]() {
        tlars.executeStep(T_stop, true);
    });

    std::cout << "T-LARS completed in " << t1.time_ms << " ms\n\n";

    // Results
    utils_talgos::print_selection(tlars, true_support);
    utils_talgos::print_quality(tlars, true_support);

    // ========================================================================
    // PHASE 6: Cleanup
    // ========================================================================
    std::cout << "\n=== PHASE 6: Cleanup ===\n";
    std::remove(X_file.c_str());
    std::remove(y_file.c_str());
    std::remove(X_aug_file.c_str());
    std::cout << "✓ All files removed\n";

}


// ===========================================================================
// Main
// ===========================================================================

int main() {
    openmp::print_info();
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {

        // T-LARS with early stopping - low and high dimensional
        const bool run_early_stopping_demo = true;
        if (run_early_stopping_demo) {
            demo_TLARS_early_stopping(/*high_dim=*/false, /*T_stop=*/10);
            demo_TLARS_early_stopping(/*high_dim=*/true, /*T_stop=*/10);
        }


        // T-LARS with external normalization
        const bool run_external_normalizer_demo = true;
        if (run_external_normalizer_demo) {
            demo_TLARS_with_external_normalizer(/*high_dim=*/false, /*T_stop=*/5);
            demo_TLARS_with_external_normalizer(/*high_dim=*/true, /*T_stop=*/5);
        }


        // T-LARS serialization and warm-start
        const bool run_serialization_demo = true;
        if (run_serialization_demo) demo_TLARS_serialization();


        // T-LARS controlled comparison
        const bool run_controlled_comparison = true;
        if (run_controlled_comparison) demo_TLARS_controlled_comparison();


        // T-LARS production workflow
        const bool run_production_tlars_with_mmap = true;
        if (run_production_tlars_with_mmap) demo_production_tlars_workflow();


        std::cout << "\n✓ All T-LARS demos completed\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
