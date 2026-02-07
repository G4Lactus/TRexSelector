/**
 * @file demo_09_screen_trex.cpp
 *
 * @brief Demos for the Screen-TRex Selector.
 */


 // std includes
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <random>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include "utils_talgos.hpp"
#include "utils_fdr_control.hpp"
#include "Screen_TRex_Selector.hpp"


// ======================================================================================
// Demo 01: Basic Screen-T-Rex functionality
// ======================================================================================

void demo_Screen_TRex_Selector(bool high_dim, bool rnd_coef) {

    std::cout << "=== Demo: Screen-T-Rex Selector Basic Functionality ===\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

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

    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    std::vector<double> true_coefs;
    if (rnd_coef) {
        std::cout << "Using random coefficients for true support\n";
        true_coefs =  {-0.4, -0.25, -0.8, 1.1, 2.5};
    } else {
        std::cout << "Using fixed coefficients for true support\n";
        true_coefs =  {1.0, 1.0, 1.0, 1.0, 1.0};
    }
    const double snr = 1.0;

    std::cout << "Generating synthetic data...\n";
    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/58);

    // Create mapped views as lvalues (required by TRexSelector constructor)
    std::cout << "Creating maps of data...\n";
    Eigen::Map<Eigen::MatrixXd> X_map(data.X.data(), data.X.rows(), data.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    // Setup Control Structures
    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 20;
    trex_ctrl.max_num_dummies = 10;
    trex_ctrl.max_T_stop = true;

    SolverControl solver_ctrl;
    solver_ctrl.solver_type = SolverTypeForTRex::TLARS;

    ScreenTRexControlParameter screen_ctrl;
    screen_ctrl.trex_method = ScreenTRexMethod::TREX;

    // Setup Screen-TRex Selector
    std::cout << "Creating Screen-T-Rex Selector instance...\n";
    ScreenTRexSelector sctrex(
        X_map,
        y_map,
        screen_ctrl,
        trex_ctrl,
        std::move(solver_ctrl),
        /*seed=*/42,
        /*verbose=*/true
    );

    // Execute Screen-T-Rex Selector
    std::cout << "Executing Screen-T-Rex Selector...\n";
    sctrex.select();

    // Display Results
    ScreenTRexSelectionResult screen_result = sctrex.getScreenResult();

    std::cout << "Estimated FDR: " <<  screen_result.estimated_FDR << "\n";

    auto selected_indices = sctrex.getSelectedIndices();
    std::cout << "Selected indices: ";
    for (const auto& idx : selected_indices) {
        std::cout << idx << " ";
    }
    std::cout << "\n";

    const double fdp = utils_fdr_control::compute_fdp(
        /*selected_indices=*/selected_indices,
        /*true_support=*/true_support
    );

    const double tpp = utils_fdr_control::compute_tpp(
        /*selected_indices=*/selected_indices,
        /*true_support=*/true_support
    );

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "False Discovery Proportion (FDP): " << fdp << "\n";
    std::cout << "True Positive Proportion (TPP):   " << tpp << "\n";

    std::cout << "\n\n";
}


// ======================================================================================
// Demo 2: Monte Carlo Simulation for Screen-T-Rex Method Comparison - Variable Support
// ======================================================================================

void demo_ScreenTRex_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {
    std::cout.setf(std::ios::unitbuf);
    std::cout << "=== Demo: Screen-T-Rex Method Comparison ===\n\n";

    if (high_dim) {
        std::cout << "High-dimensional setting (p > n)\n";
    } else {
        std::cout << "Low-dimensional setting (n > p)\n";
    }
    const std::size_t n = high_dim ? 300 : 1000;
    const std::size_t p = high_dim ? 1000 : 300;

    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    // =================================================================
    // Define Screen-TRex variants to test
    // =================================================================

    struct ScreenMethodInfo {
        bool bootstrap;
        std::string name;
    };

    const std::vector<ScreenMethodInfo> methods_to_test = {
        {/*bootstrap=*/false, "Screen-TRex"},
        {/*bootstrap=*/true,  "Screen-TRex+Bootstrap"}
    };

    // Result storage: method x SNR
    std::map<std::string, Eigen::VectorXd> est_fdr_results_map;
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;

    for (const auto& method: methods_to_test) {
        est_fdr_results_map[method.name] = Eigen::VectorXd(snr_values.size());
        fdr_results_map[method.name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[method.name] = Eigen::VectorXd(snr_values.size());
    }

    // Setup Screen-TRex Parameters
    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 20;
    trex_ctrl.max_num_dummies = 10;
    trex_ctrl.max_T_stop = true;


    // =============================================================
    // RNG for support and coefficients generation
    // =============================================================
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    std::normal_distribution<double> normal(0.0, 1.0);

    // =============================================================
    // Method loop
    // =============================================================
    for (const auto& current_method : methods_to_test) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << current_method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        // =============================================================
        // Loop over SNR values
        // =============================================================
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            double total_est_fdp = 0.0;
            double total_fdp = 0.0;
            double total_tpp = 0.0;
            const double snr = snr_values[snr_idx];

            std::cout << "SNR = " << std::fixed << std::setprecision(3) << snr << "\n";

            // =============================================================
            // Monte Carlo loop
            // =============================================================
            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Generate unique support indices for this MC run
                std::unordered_set<std::size_t> support_set;
                while (support_set.size() < cardinality_true_support) {
                    support_set.insert(uniform_dist(rng));
                }
                std::vector<std::size_t> true_support(support_set.begin(), support_set.end());

                // Generate coefficient values
                std::vector<double> true_coefs;
                true_coefs.reserve(cardinality_true_support);
                if (rnd_coef) {
                    // Random (heterogeneous) coefficients
                    for (std::size_t i = 0; i < cardinality_true_support; ++i) {
                        true_coefs.push_back(normal(rng));
                    }
                } else {
                    // Fixed (deterministic/homogeneous) coefficients
                    for (std::size_t i = 0; i < cardinality_true_support; ++i) {
                        true_coefs.push_back(1);
                    }
                }

                // Clear progress line and print true support
                std::cout << std::string(80, ' ') << "\r";
                std::cout << "  Run [" << std::setw(3) << (mc + 1) << "/"
                          << num_MC << "] Support: ";
                for (const auto idx : true_support) {
                    std::cout << idx << " ";
                }
                std::cout << "\n";

                // Generate data
                utils_talgos::SyntheticData data(
                    n, p, true_support, true_coefs, snr,
                    /*seed=*/24 + snr_idx * 1000 + mc
                );

                Eigen::Map<Eigen::MatrixXd> X_map(data.X.data(), data.X.rows(), data.X.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

                // Setup Screen Control Parameters
                ScreenTRexControlParameter screen_ctrl;
                screen_ctrl.trex_method = ScreenTRexMethod::TREX;
                screen_ctrl.use_bootstrap_CI = current_method.bootstrap;
                screen_ctrl.R_boot = 1000;
                screen_ctrl.ci_grid_step = 0.001;

                // Create Screen-TRex Selector instance
                ScreenTRexSelector sctrex(
                    /*X=*/X_map,
                    /*y=*/y_map,
                    /*screen_control=*/screen_ctrl,
                    /*trex_control=*/trex_ctrl,
                    /*solver_control=*/SolverControl(),
                    /*seed=*/-1,
                    /*verbose=*/false
                );

                // Exceute Screen-TRex selection
                sctrex.select();

                // Evaluate performance
                auto selected_indices = sctrex.getSelectedIndices();

                const double fdp = utils_fdr_control::compute_fdp(
                    selected_indices,
                    true_support
                );

                const double tpp = utils_fdr_control::compute_tpp(
                    selected_indices,
                    true_support
                );

                total_est_fdp += sctrex.getScreenResult().estimated_FDR;
                total_fdp += fdp;
                total_tpp += tpp;

                // Update progress indicator (will be overwritten or cleared)
                std::cout << "  Progress: " << std::fixed << std::setprecision(1)
                          << (100.0 * (mc + 1) / num_MC) << "% complete\r";
            }

            // Clear progress line
            std::cout << std::string(80, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";

            // Store average results
            est_fdr_results_map[current_method.name](snr_idx) =
                total_est_fdp / static_cast<double>(num_MC);
            fdr_results_map[current_method.name](snr_idx) =
                total_fdp / static_cast<double>(num_MC);
            tpr_results_map[current_method.name](snr_idx) =
                total_tpp / static_cast<double>(num_MC);
        }
        std::cout << "\n";
    }

    // ===================================================================
    // Print results table
    // ===================================================================

    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "=== Screen-TRex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n";
    std::cout << std::string(80, '=') << "\n\n";

    const int name_width = 25;
    const int col_width = 10;

    // Print header
    std::cout << std::setw(name_width) << "Method";
    std::cout << std::setw(name_width) << "Metric";
    for (const auto snr : snr_values) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(col_width) << snr;
    }
    std::cout << "\n";
    std::cout << std::string(name_width * 2 + col_width * snr_values.size(), '-') << "\n";

    // Print each method's results
    for (const auto& method : methods_to_test) {
        // FDR row
        std::cout << std::setw(name_width) << method.name;
        std::cout << std::setw(name_width) << "FDR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << fdr_results_map[method.name](i);
        }
        std::cout << "\n";

        // TPR row
        std::cout << std::setw(name_width) << "";
        std::cout << std::setw(name_width) << "TPR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << tpr_results_map[method.name](i);
        }
        std::cout << "\n";

        // Estimated FDR row
        std::cout << std::setw(name_width) << "";
        std::cout << std::setw(name_width) << "Estimated FDR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << est_fdr_results_map[method.name](i);
        }
        std::cout << "\n\n";
    }

    std::cout << "\n\n";
}


// ----------------------------------------------------------------------


int main() {
    std::cout.setf(std::ios::unitbuf);

    // Demo 01: Run basic Screen-T-Rex Selector demo
    // --------------------------------------------------------------------------------------
    // demo_Screen_TRex_Selector(/*high_dim=*/true, /*rnd_coef=*/true);


    // Demo 02:  Screen-T-Rex Monte Carlo simulation - Ordinary vs. Bootstrap
    // --------------------------------------------------------------------------------------
    demo_ScreenTRex_MonteCarlo(/*num_MC=*/200, /*high_dim=*/true, /*rnd_coef=*/true);


    // Demo 03: Screen-DA-TRex Monte Carlo simulation - Ordinary vs. Bootstrap
    // --------------------------------------------------------------------------------------

    return 0;
}
