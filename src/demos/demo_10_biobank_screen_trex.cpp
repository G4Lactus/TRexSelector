/**
 * @file demo_10_biobank_screen_trex.cpp
 *
 * @brief Demo for Biobank Screen-TRex Selector (Algorithm 1).
 *
 * @details Demonstrates the Biobank Screen-TRex orchestrator that
 *          coordinates Screen-TRex and T-Rex selectors for efficient
 *          multi-phenotype screening.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include "utils_talgos.hpp"
#include "utils_fdr_control.hpp"
#include "BiobankScreenTRex.hpp"


// ======================================================================================
// Demo 01: Biobank Screen-TRex Selector
// ======================================================================================

void demo_BiobankScreenTrex_SinglePhenotype(bool rnd_coef = false) {

    std::cout << "=== Demo: Biobank Screen-TRex Selector ===\n\n";

    const std::size_t n = 1000;
    const std::size_t p = 5000;
    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 815, 1025};
    std::vector<double> true_coefs;
    if (rnd_coef) {
        std::cout << "Using random coefficients for true support\n";
        true_coefs =  {-0.4, -0.25, -0.8, 1.1, 2.5};
    } else {
        std::cout << "Using fixed coefficients for true support\n";
        true_coefs =  {1.0, 1.0, 1.0, 1.0, 1.0};
    }
    const double snr = 1.0;

    // Generate synthetic data
    std::cout << "Generating synthetic data...\n";
    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/123);

    // Create mapped views
    Eigen::Map<Eigen::MatrixXd> X_map(data.X.data(), data.X.rows(), data.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    // Setup Biobank Screen-TRex Control Parameters
    BiobankScreenTRexControl biosctrex_ctrl;
    biosctrex_ctrl.lower_bound_FDR = 0.05;
    biosctrex_ctrl.upper_bound_FDR = 0.15;
    biosctrex_ctrl.target_FDR_trex = 0.10;

    // Screen-TRex parameters
    biosctrex_ctrl.screen_ctrl.trex_method = ScreenTRexMethod::TREX;
    biosctrex_ctrl.screen_ctrl.R_boot = 1000;
    biosctrex_ctrl.screen_ctrl.ci_grid_step = 0.001;

    // T-Rex parameters
    biosctrex_ctrl.trex_ctrl.K = 20;
    biosctrex_ctrl.trex_ctrl.max_num_dummies = 10;
    biosctrex_ctrl.trex_ctrl.max_T_stop = true;

    // Solver parameters
    biosctrex_ctrl.solver_ctrl.solver_type = SolverTypeForTRex::TLARS;

    // Create Biobank Screen-TRex Selector instance
    std::cout << "Initializing Biobank Screen-TRex selector...\n";
    BiobankScreenTRex biosctrex(
        X_map,
        y_map,
        biosctrex_ctrl,
        /*seed=*/42,
        /*verbose=*/true
    );

    // Exceute screening
    std::cout << "\nExecuting Biobank Screen-TRex selection...\n";
    auto result = biosctrex.screenPhenotype();

    // Display results
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "=== Results ===\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "Method used: " << result.method_used << "\n";
    std::cout << "Estimated FDR: " << std::fixed << std::setprecision(4)
              << result.estimated_FDR << "\n";
    std::cout << "Number of selected variables: " << result.selected_indices.size() << "\n";

    std::cout << "\nIntermediate estimates:\n";
    std::cout << "  α̂ (ordinary):  " << result.estimated_FDR_screen_ordinary << "\n";
    std::cout << "  α̂_C (bootstrap): " << result.estimated_FDR_screen_bootstrap << "\n";

    std::cout << "\nSelected indices: ";
    for (const auto& idx : result.selected_indices) {
        std::cout << idx << " ";
    }
    std::cout << "\n";

    // Compute performance metrics
    const double fdp = utils_fdr_control::compute_fdp(
        result.selected_indices,
        true_support
    );
    const double tpp = utils_fdr_control::compute_tpp(
        result.selected_indices,
        true_support
    );

    std::cout << "\nPerformance:\n";
    std::cout << "  FDP (actual): " << fdp << "\n";
    std::cout << "  TPP (power):  " << tpp << "\n";

    std::cout << "\n\n";
}



// ======================================================================================
// Demo 02: Multiple Phenotype Screening
// ======================================================================================

void demo_BiobankScreenTrex_MultiplePhenotypes(bool rnd_coef) {

    std::cout << "=== Demo: Biobank Screen-TRex - Multiple Phenotypes ===\n\n";

    // Problem dimensions
    const std::size_t n = 500;
    const std::size_t p = 2000;
    const std::size_t q = 5;  // Number of phenotypes

    // Generate design matrix (shared across all phenotypes)
    std::mt19937 rng(4212);
    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X(i, j) = normal(rng);
        }
    }

    // Generate multiple phenotypes with different support sets and SNR
    Eigen::MatrixXd Y(n, q);
    std::vector<std::vector<std::size_t>> true_supports(q);
    std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};

    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);

    for (std::size_t pheno_idx = 0; pheno_idx < q; ++pheno_idx) {

        // Generate unique support for this phenotype
        std::unordered_set<std::size_t> support_set;
        const std::size_t support_size = 5 + (pheno_idx % 3);  // 5, 6, or 7

        while (support_set.size() < support_size) {
            support_set.insert(uniform_dist(rng));
        }

        true_supports[pheno_idx] = std::vector<std::size_t>(
            support_set.begin(),
            support_set.end()
        );

        // Generate coefficients
        std::vector<double> true_coefs(support_size);
        for (auto& coef : true_coefs) {
            coef = rnd_coef ? normal(rng) : 1.0;
        }

        // Generate phenotype data
        const double snr = snr_values[pheno_idx];
        utils_talgos::SyntheticData data(
            n, p, true_supports[pheno_idx], true_coefs, snr,
            /*seed=*/100 + pheno_idx
        );

        Y.col(pheno_idx) = data.y;

        std::cout << "Phenotype " << pheno_idx << ": "
                  << "support size = " << support_size
                  << ", SNR = " << std::fixed << std::setprecision(2) << snr << "\n";
    }

    std::cout << "\n";

    // Create mapped views
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> Y_map(Y.data(), Y.rows(), Y.cols());

    // Setup Biobank Screen-TRex Control Parameters
    BiobankScreenTRexControl biosctrex_ctrl;
    biosctrex_ctrl.lower_bound_FDR = 0.05;
    biosctrex_ctrl.upper_bound_FDR = 0.15;
    biosctrex_ctrl.target_FDR_trex = 0.1;

    biosctrex_ctrl.screen_ctrl.trex_method = ScreenTRexMethod::TREX;
    biosctrex_ctrl.screen_ctrl.R_boot = 500;  // Reduced for speed
    biosctrex_ctrl.screen_ctrl.ci_grid_step = 0.005;

    biosctrex_ctrl.trex_ctrl.K = 20;
    biosctrex_ctrl.trex_ctrl.max_num_dummies = 10;
    biosctrex_ctrl.trex_ctrl.max_T_stop = true;

    biosctrex_ctrl.solver_ctrl.solver_type = SolverTypeForTRex::TLARS;

    // Create Biobank Screen-TRex instance
    BiobankScreenTRex biobank_sctrex(
        X_map,
        Y_map,
        biosctrex_ctrl,
        /*seed=*/42,
        /*verbose=*/true
    );

    // Execute screening for all phenotypes
    std::cout << "Executing Biobank Screen-TRex for all phenotypes...\n\n";
    auto results = biobank_sctrex.screenPhenotypes();

    // Get summary statistics
    auto stats = biobank_sctrex.getBiobankScreenTRexResult(results);

    // Display detailed results
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "=== Detailed Results by Phenotype ===\n";
    std::cout << std::string(90, '=') << "\n\n";

    const int col_width = 18;
    std::cout << std::setw(10) << "Pheno"
              << std::setw(col_width) << "Method"
              << std::setw(12) << "Est. FDR"
              << std::setw(12) << "Selected"
              << std::setw(12) << "FDP"
              << std::setw(12) << "TPP"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];

        // Compute actual FDP and TPP
        const double fdp = utils_fdr_control::compute_fdp(
            res.selected_indices,
            true_supports[i]
        );
        const double tpp = utils_fdr_control::compute_tpp(
            res.selected_indices,
            true_supports[i]
        );

        std::cout << std::setw(10) << i
                  << std::setw(col_width) << res.method_used
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << res.estimated_FDR
                  << std::setw(12) << res.selected_indices.size()
                  << std::setw(12) << fdp
                  << std::setw(12) << tpp
                  << "\n";
    }

    // Display summary statistics
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "=== Summary Statistics ===\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "Total phenotypes: " << stats.num_phenotypes << "\n";
    std::cout << "Method usage:\n";
    std::cout << "  Screen-TRex (ordinary):  " << stats.num_used_screen_ordinary << "\n";
    std::cout << "  Screen-TRex (bootstrap): " << stats.num_used_screen_bootstrap << "\n";
    std::cout << "  T-Rex (fallback):        " << stats.num_used_trex << "\n\n";
    std::cout << "Average estimated FDR: " << std::fixed << std::setprecision(4)
              << stats.avg_estimated_FDR << "\n";
    std::cout << "Average selected count: " << std::fixed << std::setprecision(2)
              << stats.avg_selected_count << "\n";

    std::cout << "\n\n";
}



// ======================================================================================
// Demo 03: Monte Carlo Simulation for Biobank Screen-TRex
// ======================================================================================

void demo_BiobankScreenTrex_MonteCarlo(std::size_t num_MC, bool rnd_coef) {
    std::cout << "=== Demo: Biobank Screen-TRex Monte Carlo Simulation ===\n\n";
    std::cout << "Number of Monte Carlo runs: " << num_MC << "\n";

    // Problem dimensions
    const std::size_t n = 300;
    const std::size_t p = 1000;
    const std::size_t support_size = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};

    // Trach results
    std::map<std::string, Eigen::VectorXd> fdp_results;
    std::map<std::string, Eigen::VectorXd> fdp_estimates;
    std::map<std::string, Eigen::VectorXd> tpp_results;
    std::map<std::string, Eigen::VectorXd> method_usage;

    const std::vector<std::string> method_names = {
        "Screen-TRex (ordinary)",
        "Screen-TRex (bootstrap-CI)",
        "T-Rex (fallback)"
    };

    for (const auto& name : method_names) {
        fdp_results[name] = Eigen::VectorXd::Zero(snr_values.size());
        fdp_estimates[name] = Eigen::VectorXd::Zero(snr_values.size());
        tpp_results[name] = Eigen::VectorXd::Zero(snr_values.size());
        method_usage[name] = Eigen::VectorXd::Zero(snr_values.size());
    }

    // RNG for support generation
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    std::normal_distribution<double> normal(0.0, 1.0);

    // =================================================
    // Loop over SNR values
    // =================================================
    for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
        const double snr = snr_values[snr_idx];

        std::cout << "SNR = " << std::fixed << std::setprecision(2) << snr << "\n";

        // =================================================
        // Loop over Monte Carlo runs
        // =================================================
        for (std::size_t mc = 0; mc < num_MC; ++mc) {

            // Generate unique support
            std::unordered_set<std::size_t> support_set;
            while (support_set.size() < support_size) {
                support_set.insert(uniform_dist(rng));
            }
            std::vector<std::size_t> true_support(support_set.begin(), support_set.end());

            // Generate coefficients
            std::vector<double> true_coefs(support_size);
            for (auto& coef : true_coefs) {
                coef = rnd_coef ? normal(rng) : 1.0;
            }

            // Generate data
            utils_talgos::SyntheticData data(
                n, p, true_support, true_coefs, snr,
                /*seed=*/1000 * snr_idx + mc
            );

            // Create mapped views
            Eigen::Map<Eigen::MatrixXd> X_map(data.X.data(), data.X.rows(), data.X.cols());
            Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

            // Setup Biobank Screen-TRex
            BiobankScreenTRexControl biosctrex_ctrl;
            biosctrex_ctrl.lower_bound_FDR = 0.05;
            biosctrex_ctrl.upper_bound_FDR = 0.15;
            biosctrex_ctrl.target_FDR_trex = 0.10;

            biosctrex_ctrl.screen_ctrl.trex_method = ScreenTRexMethod::TREX;
            biosctrex_ctrl.screen_ctrl.R_boot = 1000;
            biosctrex_ctrl.screen_ctrl.ci_grid_step = 0.001;

            biosctrex_ctrl.trex_ctrl.K = 20;
            biosctrex_ctrl.trex_ctrl.max_num_dummies = 10;
            biosctrex_ctrl.trex_ctrl.max_T_stop = true;

            biosctrex_ctrl.solver_ctrl.solver_type = SolverTypeForTRex::TLARS;

            // Create Biobank Screen-TRex instance
            BiobankScreenTRex biobank_sctrex(
                X_map,
                y_map,
                biosctrex_ctrl,
                /*seed=*/-1,
                /*verbose=*/false
            );

            // Execute screening
            auto result = biobank_sctrex.screenPhenotype();


            // Compute metrics
            const double fdp = utils_fdr_control::compute_fdp(
                result.selected_indices, true_support
            );
            const double tpp = utils_fdr_control::compute_tpp(
                result.selected_indices, true_support
            );

            // 1. Accumulate performance metrics
            fdp_results[result.method_used](snr_idx) += fdp;
            tpp_results[result.method_used](snr_idx) += tpp;
            method_usage[result.method_used](snr_idx) += 1.0;

            // 2. Accumulate estimated FDR for diagnostics
            fdp_estimates["Screen-TRex (ordinary)"](snr_idx) += result.estimated_FDR_screen_ordinary;
            fdp_estimates["Screen-TRex (bootstrap-CI)"](snr_idx) += result.estimated_FDR_screen_bootstrap;
            fdp_estimates["T-Rex (fallback)"](snr_idx) += biosctrex_ctrl.target_FDR_trex;


            // Progress indicator
            std::cout << "  Progress: " << std::fixed << std::setprecision(1)
                      << (100.0 * (mc + 1) / num_MC) << "% complete\r" << std::flush;
        }

        std::cout << std::string(80, ' ') << "\r";
        std::cout << "  Completed " << num_MC << " runs\n\n";
    }

    // Compute averages
    for (const auto& name : method_names) {
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            // Performance metrics to average: performance only if method was used
            if (method_usage[name](i) > 0) {
                fdp_results[name](i) /= method_usage[name](i);
                tpp_results[name](i) /= method_usage[name](i);
            }

            // Diagnostic estimates to average: always average over all runs (unconditional)
            fdp_estimates[name](i) /= static_cast<double>(num_MC);
        }
    }


    // Display results
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "=== Biobank Screen-TRex Results (averaged over " << num_MC << " runs) ===\n";
    std::cout << std::string(90, '=') << "\n\n";

    const int name_width = 30;
    const int col_width = 12;

    // Print header
    std::cout << std::setw(name_width) << "Method"
              << std::setw(15) << "Metric";
    for (const auto snr : snr_values) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(col_width) << snr;
    }
    std::cout << "\n";
    std::cout << std::string(name_width + 15 + col_width * snr_values.size(), '-') << "\n";

    // Print results
    for (const auto& name : method_names) {
        // Usage count
        std::cout << std::setw(name_width) << name
                  << std::setw(15) << "Usage (%)";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(col_width)
                      << (100.0 * method_usage[name](i) / num_MC);
        }
        std::cout << "\n";

        // FDP
        std::cout << std::setw(name_width) << ""
                  << std::setw(15) << "FDR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << fdp_results[name](i);
        }
        std::cout << "\n";

        // Estimated FDR (Diagnostic)
        std::cout << std::setw(name_width) << ""
                  << std::setw(15) << "Est. FDR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << fdp_estimates[name](i);
        }
        std::cout << "\n";

        // TPP
        std::cout << std::setw(name_width) << ""
                  << std::setw(15) << "TPR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << tpp_results[name](i);
        }
        std::cout << "\n\n";
    }

    std::cout << std::string(name_width + 15 + col_width * snr_values.size(), '-') << "\n";
    std::cout << "\n\n";
}


// ----------------------------------------------------------------------


int main() {
    std::cout.setf(std::ios::unitbuf);

    // Demo 01: Biobank Screen-TRex Selector
    // ------------------------------------------------------------
    // demo_BiobankScreenTrex_SinglePhenotype(/*rnd_coef=*/false);


    // Demo 02: Multiple phenotype screening
    // ------------------------------------------------------------
    // demo_BiobankScreenTrex_MultiplePhenotypes(/*rnd_coef=*/false);


    // Demo 03: Monte Carlo simulation
    // ------------------------------------------------------------
    demo_BiobankScreenTrex_MonteCarlo(/*num_MC=*/200, /*rnd_coef=*/false);

    return 0;
}
