#include <iomanip>
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "Normalizer.hpp"
#include "utils_openmp.hpp"

#include "TSTEPWISE_Solver.hpp"

#include "utils_eval.hpp"
#include "utils_perf.hpp"
#include "utils_talgos.hpp"


// ============================================================================
// Demo 1: Basic T-Stepwise with Early Stopping
// ============================================================================

void demo_TSTEPWISE_early_stopping(bool high_dim, std::size_t T_stop = 5) {

    std::cout << "Demo 1: Basic T-Stepwise with Early Stopping" << "\n";

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
    const std::vector<std::size_t> true_support = {10, 25, 40};
    const std::vector<double> true_coefs = {2.5, -1.8, 3.2};
    const double sigma{1};

    utils_talgos::print_tlars_config(n, p, num_dummies, T_stop, true_support, true_coefs, 1);

    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, sigma);

    // Generate dummy variables
    Eigen::MatrixXd X_dummies(n, num_dummies);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < num_dummies; ++j) {
            X_dummies(i, j) = norm_dist(rng);
        }
    }

    // Concatenate X and dummies
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug << data.X, X_dummies;

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(), n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), n);

    TSTEPWISE_Solver tstepwise(X_aug_map, y_map, num_dummies, true, true, true);
    auto t1 = utils_perf::profileit([&]() { tstepwise.executeStep(T_stop, /*early_stop=*/true); });
    std::cout << "T-STEPWISE early stopping at T=" << T_stop << " took " << t1.time_ms << " ms\n";

    utils_talgos::print_selection(tstepwise, true_support);
    utils_talgos::print_quality(tstepwise, true_support);

    std::cout << "\n\n";
}


// ============================================================================
// Demo 2: Serialization
// ============================================================================

void demo_TSTEPWISE_serialization() {
    std::cout << "Demo 2: T-Stepwise Serialization & Path Consistency Demo" << "\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    const std::size_t n = 100, p = 50, num_dummies = 50, T_stop = 3;
    const std::vector<std::size_t> true_support = {10, 25, 40};
    const std::vector<double> true_coefs = {2.5, -1.8, 3.2};

    utils_talgos::print_tlars_config(n, p, num_dummies, T_stop, true_support, true_coefs);

    utils_talgos::SyntheticData data(n, p, true_support, true_coefs);

    // Generate dummy variables
    Eigen::MatrixXd X_dummies(n, num_dummies);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < num_dummies; ++j) {
            X_dummies(i, j) = norm_dist(rng);
        }
    }

    // Concatenate X and dummies
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug << data.X, X_dummies;

    Eigen::Map<Eigen::MatrixXd> X_aug_map(X_aug.data(), n, p + num_dummies);
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), n);

    // First, create reference solver, run to full solution path
    TSTEPWISE_Solver solver_ref(X_aug_map, y_map, num_dummies, true, true, true);
    solver_ref.executeStep(0, /*early_stop=*/false);

    std::string filename = "tstepwise_checkpoint.bin";

    // --------- PART 1: Run partial path and save checkpoint (scoped) ---------
    {
        TSTEPWISE_Solver solver1(X_aug_map, y_map, num_dummies, true, true, true);
        solver1.executeStep(T_stop, /*early_stop=*/true);

        std::cout << "Checkpoint at partial stop: " << solver1.getNumSteps() << " steps\n";
        const char* checkpoint = filename.c_str();
        solver1.save(checkpoint);
        std::cout << "✓ Checkpoint saved to '" << checkpoint << "'\n";
    }
    // At this point, solver1 is destroyed, no state remains


    // --------- PART 2: Load from checkpoint and continue (scoped) ---------
    {
        TSTEPWISE_Solver solver2 = TSTEPWISE_Solver::load(filename.c_str(), X_aug_map);
        std::cout << "Loaded from checkpoint: " << solver2.getNumSteps() << " steps\n";
        solver2.executeStep(0, /*early_stop=*/false); // Continue to full path

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
    }

    // Cleanup
    std::remove(filename.c_str());
    std::cout << "✓ Checkpoint file removed\n";
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
        // T-Stepwise with early stopping
        demo_TSTEPWISE_early_stopping(/*high_dim=*/true,/*T_stop=*/5);
        demo_TSTEPWISE_early_stopping(/*high_dim=*/false,/*T_stop=*/5);

        demo_TSTEPWISE_serialization();

        std::cout << "\n✓ All T-Stepwise demos completed\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
