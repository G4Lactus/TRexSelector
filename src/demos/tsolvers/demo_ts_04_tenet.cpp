// ============================================================================
// demo_ts_04_tenet.cpp
// ============================================================================
/**
 * @file demo_ts_04_tenet.cpp
 *
 * @brief Demonstration of T-Elastic Net (Terminating Elastic Net) solver.
 *
 * @details Shows basic usage, external/internal normalization, serialization,
 *          and comparison between in-memory and memory-mapped workflows.
 */
// ============================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>

// ml_methods includes
#include <ml_methods/standardization/lp_norm_scaler.hpp>

// utils includes
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/openmp/utils_openmp.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>
#include <utils/eval_metrics/utils_eval_composites.hpp>


// ============================================================================
// Namespace aliases
// ============================================================================
namespace datagen = trex::utils::datageneration::datagen;
namespace memmap = trex::utils::memmap;
namespace scaler = trex::ml_methods::standardization;
namespace omp_utils = trex::utils::openmp;
namespace cdiagnost = trex::utils::eval::cdiagnostics;
namespace rates = trex::utils::eval::rates;
namespace counts = trex::utils::eval::counts;
namespace composites = trex::utils::eval::composites;
namespace fs = std::filesystem;
namespace tsolvers = trex::tsolvers::linear_model::lars_based;



// ============================================================================
// Demo 1: Basic T-Elastic Net with Early Stopping
// ============================================================================

void demo_TENET_early_stopping(bool high_dim, bool rnd_coef, std::size_t T_stop) {

    // Print Demo Header
    cdiagnost::print_section_header("Demo 1: Basic T-Elastic Net with Early Stopping");

    // Setup
    const std::size_t n = high_dim ? 1000 : 5000;;
    const std::size_t p = high_dim ? 5000 : 1000;
    const std::size_t num_dummies = 10 * p;
    const double lambda2 = 0.5;

    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs =
        rnd_coef ?
        std::vector<double>{-0.4, -0.2, -0.8, 1.1, 2.5} :
        std::vector<double>{1, 1, 1, 1, 1};
    const double snr = 1.0;

    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    // Print demo config
    cdiagnost::print_talgo_demo_config(
        n,
        p,
        num_dummies,
        T_stop,
        true_support,
        true_coefs,
        snr
    );

    // Generate data
    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        static_cast<Eigen::Index>(num_dummies),
        true_support,
        true_coefs,
        snr,
        /*seed=*/42,
        /*X_seed=*/-1,
        /*dummy_seed=*/-1,
        datagen::predictor_policy::Normal(),
        datagen::dummygen::Distribution::Normal(),
        datagen::noisegen::noise_policy::Normal()
    );

    // Create Maps
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(data.getD().data(), data.rows(),
                                      data.num_dummies());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // Run T-Elastic Net solver with internal normalization
    tsolvers::TENET_Solver tenet(
        X_map,
        D_map,
        y_map,
        lambda2,
        /*normalize=*/true,
        /*intercept=*/true,
        /*verbose=*/true
    );

    // Execute with early stopping
    tenet.executeStep(T_stop, /*early_stop=*/true);

    // Results
    // Print selection path
    cdiagnost::print_selection(tenet, true_support);
    // Print selection quality measures
    cdiagnost::print_selection_quality(tenet, true_support);

    // T-Elastic Net specific diagnostics
    std::cout << "\nT-Elastic Net Diagnostics:\n";
    std::cout << "  Removals: " << tenet.getNumRemovals() << "\n";
    std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(4)
              << tenet.getCyclingRatio() << "\n";

    std::cout << "\n\n";
}



// ============================================================================
// Demo 2: External Normalization with DataTransformer
// ============================================================================

void demo_TENET_with_external_normalizer(bool high_dim, bool rnd_coef, std::size_t T_stop) {

    // Print Demo Header
    cdiagnost::print_section_header("=== Demo 2: T-Elastic Net with External Normalization ===");

    // Setup
    const std::size_t n = high_dim ? 1'000 : 5'000;
    const std::size_t p = high_dim ? 5'000 : 1'000;
    const std::size_t num_dummies = 10 * p;
    const double lambda2 = 0.5;

    const std::vector<std::size_t> true_support = {4, 27, 149, 398, 420};
    const std::vector<double> true_coefs = rnd_coef ?
                                           std::vector<double>{2.5, -0.4, -0.2, -0.8, 1.1} :
                                           std::vector<double>{1, 1, 1, 1, 1};
    const double snr = 1.0;

    std::cout << (high_dim ? "High-dimensional (p > n)\n" : "Low-dimensional (n > p)\n");
    cdiagnost::print_talgo_demo_config(n, p, num_dummies, T_stop, true_support,
                                        true_coefs, snr);

    // Generate data
    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        static_cast<Eigen::Index>(num_dummies),
        true_support,
        true_coefs,
        snr,
        /*seed=*/42,
        /*X_seed=*/-1,
        /*dummy_seed=*/-1,
        datagen::predictor_policy::Normal(),
        datagen::dummygen::Distribution::Normal(),
        datagen::noisegen::noise_policy::Normal()
    );


    // Create maps
    Eigen::Map<Eigen::MatrixXd> X_map(
        data.getX().data(),
        data.rows(),
        data.cols()
    );

    Eigen::Map<Eigen::MatrixXd> D_map(
        data.getD().data(),
        data.rows(),
        data.num_dummies()
    );

    Eigen::Map<Eigen::VectorXd> y_map(
        data.getY().data(),
        data.rows()
    );

    // External normalization
    std::cout << "Applying external L2 normalization...\n";

    // Normalize X
    scaler::LpNormScaler l2scaler_X(scaler::LpNormScaler::NormType::L2, true);
    l2scaler_X.fit(X_map);
    l2scaler_X.transform_inplace(X_map);

    // Normalize D
    scaler::LpNormScaler l2scaler_D(scaler::LpNormScaler::NormType::L2, true);
    l2scaler_D.fit(D_map);
    l2scaler_D.transform_inplace(D_map);

    // Center y
    y_map.array() -= y_map.mean();

    // Run T-Elastic Net
    tsolvers::TENET_Solver tenet(
        X_map,
        D_map,
        y_map,
        lambda2,
        /*normalize=*/false,
        /*intercept=*/false,
        /*verbose=*/true
    );

    // Execute with early stopping
    tenet.executeStep(T_stop, /*early_stop=*/true);

    // Results
    // Print selection path
    cdiagnost::print_selection(tenet, true_support);
    // Print selection quality measures
    cdiagnost::print_selection_quality(tenet, true_support);

    // T-Elastic Net specific diagnostics
    std::cout << "\nT-Elastic Net Diagnostics:\n";
    std::cout << "  Removals: " << tenet.getNumRemovals() << "\n";
    std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(4)
              << tenet.getCyclingRatio() << "\n";

    std::cout << "\n\n";
}



// ============================================================================
// Demo 3: Serialization & Warm-Start
// ============================================================================

void demo_TENET_serialization() {

    // Print Demo Header
    cdiagnost::print_section_header("=== Demo 3: T-Elastic Net Serialization & Path Consistency ===");

    // Setup
    const std::size_t n = 100, p = 50;
    const std::size_t num_dummies = p;
    const std::size_t T_stop_final = 15;
    const std::size_t T_stop_partial = 7;
    const double lambda2 = 0.5;
    const double snr = 1.0;

    const std::vector<std::size_t> true_support = {10, 25, 40};
    const std::vector<double> true_coefs = {2.5, -1.8, 3.2};

    cdiagnost::print_talgo_demo_config(n, p, num_dummies,
                                       T_stop_final,
                                       true_support,
                                       true_coefs,
                                       /*snr=*/snr);

    // Generate data
    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        static_cast<Eigen::Index>(num_dummies),
        true_support,
        true_coefs,
        snr,
        /*seed=*/42,
        /*X_seed=*/-1,
        /*dummy_seed=*/-1,
        datagen::predictor_policy::Normal(),
        datagen::dummygen::Distribution::Normal(),
        datagen::noisegen::noise_policy::Normal()
    );

    // Create Maps
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(),
                                        data.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(data.getD().data(), data.rows(),
                                        data.num_dummies());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // Create reference solver: run until T_stop_final
    tsolvers::TENET_Solver solver_ref(
        X_map,
        D_map,
        y_map,
        lambda2,
        /*normalize=*/true,
        /*intercept=*/true,
        /*verbose=*/false
    );
    solver_ref.executeStep(T_stop_final, /*early_stop=*/true);
    std::cout << "✓ Reference completed with " << solver_ref.getNumSteps() << " steps\n\n";

    std::string filename = "tenet_checkpoint.bin";

    // --------- STEP 1: Run partial path and save checkpoint (scoped) ---------
    {
        tsolvers::TENET_Solver solver1(
            X_map,
            D_map,
            y_map,
            lambda2,
            /*normalize=*/true,
            /*intercept=*/true,
            /*verbose=*/false
        );
        solver1.executeStep(/*T_stop=*/T_stop_partial, /*early_stop=*/true);

        std::cout << "Checkpoint at partial stop: " << solver1.getNumSteps() << " steps\n";
        const char* checkpoint = filename.c_str();
        solver1.save(checkpoint);
        std::cout << "✓ Checkpoint saved at '" << checkpoint << "'\n";
    }

    // --------- STEP 2: Load from checkpoint and continue (scoped) ---------
    {
        tsolvers::TENET_Solver solver2 =
            tsolvers::TENET_Solver::load(filename, X_map, D_map);
        std::cout << "Loaded from checkpoint: "
                  << solver2.getNumSteps() << " steps\n";
        solver2.executeStep(T_stop_final, /*early_stop=*/true); // Continue to full path

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

        // T-Elastic Net specific diagnostics
        std::cout << "\nT-Elastic Net Diagnostics:\n";
        std::cout << "  Removals: " << solver2.getNumRemovals() << "\n";
        std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(4)
                  << solver2.getCyclingRatio() << "\n";
    }

    // Cleanup
    fs::remove(filename);
    std::cout << "✓ Checkpoint file removed\n";

    std::cout << "\n\n";
}



// =============================================================================
// Demo 4: Memory-Mapped Data
// =============================================================================

void demo_memory_mapped(bool high_dim, bool rnd_coef, std::size_t T_stop) {

    // Print Demo Header
    cdiagnost::print_section_header("=== Demo 4: T-Elastic Net with Memory-Mapped Data ===");

    // Setup
    const std::size_t n = high_dim ? 1'000 : 5'000;
    const std::size_t p = high_dim ? 500'000 : 1'000;
    const std::size_t num_dummies = 10 * p;
    const double lambda2 = 0.5;
    const double snr = 1.0;

    const std::vector<std::size_t> true_support = {4, 27, 149, 398, 420};
    const std::vector<double> true_coefs = rnd_coef ?
                            std::vector<double>{2.5, -0.4, -0.2, -0.8, 1.1} :
                            std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0};

    // Print demo config
    cdiagnost::print_talgo_demo_config(n, p, num_dummies, T_stop, true_support,
                                       true_coefs, /*snr=*/snr);

    std::cout << "Generating memory-mapped data...\n";

    // We now define separate filepaths for X, D, and y
    const std::string X_file = "demo_tenet_X.bin";
    const std::string D_file = "demo_tenet_D.bin";
    const std::string y_file = "demo_tenet_y.bin";

    // Generate memory-mapped data directly on disk using the unified class
    datagen::SyntheticDataMapped data(
        X_file,
        D_file,
        y_file,
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        static_cast<Eigen::Index>(num_dummies),
        true_support,
        true_coefs,
        snr,
        /*seed=*/42
    );
    std::cout << "✓ Data generated on disk\n\n";

    // Retrieve Maps from the memory-mapped files
    auto X_map = data.getX();
    auto D_map = data.getD();
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // Run T-Elastic Net
    std::cout << "Running T-Elastic Net on memory-mapped data...\n";
    tsolvers::TENET_Solver tenet(
        X_map,
        D_map,
        y_map,
        lambda2,
        /*normalize=*/true,
        /*intercept=*/true,
        /*verbose=*/true
    );

    // Execute
    tenet.executeStep(T_stop, /*early_stop=*/true);
    std::cout << "✓ T-Elastic Net completed\n\n";

    // Results
    // Print selection path
    cdiagnost::print_selection(tenet, true_support);
    // Print selection quality measures
    cdiagnost::print_selection_quality(tenet, true_support);

    // T-Elastic Net specific diagnostics
    std::cout << "\nT-Elastic Net Diagnostics:\n";
    std::cout << "  Removals: " << tenet.getNumRemovals() << "\n";
    std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(4)
              << tenet.getCyclingRatio() << "\n";

    // Cleanup
    std::cout << "\nCleaning up files...\n";
    fs::remove(X_file);
    fs::remove(D_file);
    fs::remove(y_file);
    std::cout << "✓ All files removed\n\n";
}


// ===========================================================================
// Main
// ===========================================================================

int main() {

    std::cout << "\n";
    // Print header
    cdiagnost::print_section_header("T-Elastic Net Demo Suite");

    // Print OpenMP info
    omp_utils::print_info();
    omp_utils::set_num_threads(6);
    std::cout << "Running with " << omp_utils::get_max_threads() << " threads\n\n";

    try {

        // Demo 1: Basic usage with internal normalization
        if (true) {
            demo_TENET_early_stopping(/*high_dim=*/true, /*rnd_coef=*/false, /*T_stop=*/10);
            demo_TENET_early_stopping(/*high_dim=*/false, /*rnd_coef=*/false, /*T_stop=*/10);
        }

        // Demo 2: External normalization
        if (true) {
            demo_TENET_with_external_normalizer(/*high_dim=*/false, /*rnd_coef=*/false,
                                                /*T_stop=*/10);
            demo_TENET_with_external_normalizer(/*high_dim=*/true, /*rnd_coef=*/false,
                                                /*T_stop=*/10);
        }

        // Demo 3: Serialization
        if (true) {
            demo_TENET_serialization();
        }

        // Demo 4: Memory-mapped
        if (true) {
            demo_memory_mapped(/*high_dim=*/true, /*rnd_coef=*/false, /*T_stop=*/10);
        }

        cdiagnost::print_section_header("✓ All demos completed successfully");

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
