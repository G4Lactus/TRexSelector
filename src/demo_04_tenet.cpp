#include <iomanip>
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "Normalizer.hpp"
#include "utils_openmp.hpp"

#include "TENET_Solver.hpp"

#include "utils_eval.hpp"
#include "utils_perf.hpp"
#include "utils_talgos.hpp"


// ============================================================================
// Demo 1: Basic T-Elastic Net with Early Stopping
// ============================================================================

void demo_TENET_early_stopping(bool high_dim, double lambda2, std::size_t T_stop = 5) {

    std::cout << "Demo 1: Basic T-Elastic Net with Early Stopping (λ₂="
              << lambda2 << ")" << "\n";

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

    TENET_Solver tenet(X_aug_map, y_map, num_dummies, lambda2, true, true, true);
    auto t1 = utils_perf::profileit([&]() { tenet.executeStep(T_stop, /*early_stop=*/true); });
    std::cout << "T-Elastic Net early stopping at T=" << T_stop << " took "
              << t1.time_ms << " ms\n";

    utils_talgos::print_selection(tenet, true_support);
    utils_talgos::print_quality(tenet, true_support);

    std::cout << "Removals: " << tenet.getNumRemovals() << "\n";
    std::cout << "Cycling ratio: " << std::fixed << std::setprecision(4)
              << tenet.getCyclingRatio() << "\n";
    std::cout << "Alpha (mixing): " << std::fixed << std::setprecision(4)
              << tenet.getAlpha() << "\n";
    std::cout << "\n\n";
}


// ============================================================================
// Demo 2: Serialization
// ============================================================================

void demo_TENET_serialization() {
    std::cout << "Demo 2: T-Elastic Net Serialization & Path Consistency Demo" << "\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    const std::size_t n = 100, p = 50, num_dummies = 50, T_stop = 3;
    const double lambda2 = 0.5;
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
    TENET_Solver solver_ref(X_aug_map, y_map, num_dummies, lambda2, true, true, true);
    solver_ref.executeStep(0, /*early_stop=*/false);

    std::string filename = "tenet_checkpoint.bin";

    // --------- PART 1: Run partial path and save checkpoint (scoped) ---------
    {
        TENET_Solver solver1(X_aug_map, y_map, num_dummies, lambda2, true, true, true);
        solver1.executeStep(T_stop, /*early_stop=*/true);

        std::cout << "Checkpoint at partial stop: " << solver1.getNumSteps() << " steps\n";
        const char* checkpoint = filename.c_str();
        solver1.save(checkpoint);
        std::cout << "✓ Checkpoint saved to '" << checkpoint << "'\n";
    }
    // At this point, solver1 is destroyed, no state remains


    // --------- PART 2: Load from checkpoint and continue (scoped) ---------
    {
        TENET_Solver solver2 = TENET_Solver::load(filename.c_str(), X_aug_map);
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
        std::cout << "Removals: " << solver2.getNumRemovals() << "\n";
        std::cout << "Cycling ratio: " << std::fixed << std::setprecision(4)
                  << solver2.getCyclingRatio() << "\n";
        std::cout << "Alpha (mixing): " << std::fixed << std::setprecision(4)
                  << solver2.getAlpha() << "\n";
    }

    // Cleanup
    std::remove(filename.c_str());
    std::cout << "✓ Checkpoint file removed\n";
    std::cout << "\n\n";
}


// ============================================================================
// Demo 3: Lambda2 Comparison (LASSO vs Ridge-heavy)
// ============================================================================

void demo_TENET_lambda2_comparison() {
    std::cout << "Demo 3: T-Elastic Net λ₂ Comparison" << "\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    const std::size_t n = 200, p = 100, num_dummies = 500, T_stop = 5;
    const std::vector<std::size_t> true_support = {5, 20, 35};
    const std::vector<double> true_coefs = {3.0, -2.0, 2.5};

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

    std::vector<double> lambda2_values = {0.0, 0.1, 0.5, 1.0, 5.0};

    for (double lambda2 : lambda2_values) {
        std::cout << "\n--- λ₂ = " << lambda2 << " ---\n";

        TENET_Solver tenet(X_aug_map, y_map, num_dummies, lambda2, true, true, false);
        tenet.executeStep(T_stop, /*early_stop=*/true);

        std::cout << "Steps: " << tenet.getNumSteps()
                  << ", Actives: " << tenet.getNumActives()
                  << ", Removals: " << tenet.getNumRemovals()
                  << ", Cycling: " << std::fixed << std::setprecision(3)
                  << tenet.getCyclingRatio()
                  << ", Alpha: " << std::fixed << std::setprecision(3)
                  << tenet.getAlpha()
                  << ", RSS: " << std::fixed << std::setprecision(2)
                  << tenet.getRSS().back() << "\n";

        auto selected = tenet.getActivePredictorIndices();
        std::cout << "Selected: {";
        for (size_t i = 0; i < std::min(selected.size(), size_t(10)); ++i) {
            std::cout << selected[i] << (i < selected.size() - 1 ? ", " : "");
        }
        if (selected.size() > 10) std::cout << ", ...";
        std::cout << "}\n";
    }

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

        // T-Elastic Net early stopping demos
        demo_TENET_early_stopping(/*high_dim=*/false, /*lambda2=*/0.5, /*T_stop=*/5);
        demo_TENET_early_stopping(/*high_dim=*/true, /*lambda2=*/0.5, /*T_stop=*/5);

        // T-Elastic Net with different lambda2 values
        demo_TENET_early_stopping(/*high_dim=*/true, /*lambda2=*/0.0, /*T_stop=*/5); // LASSO limit
        demo_TENET_early_stopping(/*high_dim=*/true, /*lambda2=*/0.5, /*T_stop=*/5); // Balanced
        demo_TENET_early_stopping(/*high_dim=*/true, /*lambda2=*/2.0, /*T_stop=*/5); // Ridge-heavy

        demo_TENET_serialization();

        demo_TENET_lambda2_comparison();

        std::cout << "\n✓ All T-Elastic Net demos completed\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
