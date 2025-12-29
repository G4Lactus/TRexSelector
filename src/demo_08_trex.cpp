#include <iostream>
#include <iomanip>
#include <vector>

#include <Eigen/Dense>

#include "TRex_Selector.hpp"

#include "utils_talgos.hpp"
#include "utils_fdr_control.hpp"



void demo_TRexSelector(bool high_dim) {

    std::cout << "=== Demo: T-Rex Selector Basic Functionality ===\n\n";

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
    const std::vector<double> true_coefs = {-0.4, -0.25, -0.8, 1.1, 2.5};
    const double snr = 1.0;

    std::cout << "Generating synthetic data...\n";
    utils_talgos::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/58);

    // Create mapped views as lvalues (required by TRexSelector constructor)
    std::cout << "Creating maps of data...\n";
    Eigen::Map<Eigen::MatrixXd> X_map(data.X.data(), data.X.rows(), data.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

    // Create T-Rex Selector instance
    std::cout << "Creating T-Rex Selector instance...\n";
    TRexSelector trex(
        /*X=*/X_map,
        /*y=*/y_map,
        /*tFDR=*/0.1,
        /*K=*/20,
        /*solver=*/SolverTypeForTRex::TLARS,
        /*max_num_dummies=*/10,
        /*max_T_stop=*/true,
        /*lloop_strategy=*/LLoopStrategy::ADAPTIVE,
        /*seed=*/-1,
        /*verbose=*/true,
        /*solver_config=*/nullptr
    );

    // Execute T-Rex Selector
    std::cout << "Executing T-Rex Selector...\n";
    trex.select();

    auto selected_indices = trex.getSelectedIndices();
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


// ----------------------------------------------------------------------


void demo_TRexSelector_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    std::cout << "=== Demo: T-Rex Selector Monte Carlo Simulation ===\n\n";

    if (high_dim) {
        std::cout << "High-dimensional setting (p > n)\n";
    } else {
        std::cout << "Low-dimensional setting (n > p)\n";
    }

    const std::size_t n = high_dim ? 300 : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};

    // ===================================================================
    // Define solvers to test
    // ===================================================================

    struct SolverConfig {
        SolverTypeForTRex solver_type;
        std::string solver_name;
        double lambda2;  // For TENET solver
    };

    const std::vector<SolverConfig> solvers_to_test = {
       {SolverTypeForTRex::TLARS,      "TLARS", /*lambda2=*/{}},
       {SolverTypeForTRex::TLASSO,     "TLASSO", /*lambda2=*/{}},
       {SolverTypeForTRex::TENET,      "TENET",      /*lambda2=*/0.1},
       {SolverTypeForTRex::TSTEPWISE,  "TSTEPWISE", /*lambda2=*/{}},
       {SolverTypeForTRex::TOMP,       "TOMP", /*lambda2=*/{}},
       {SolverTypeForTRex::TGP,        "TGP", /*lambda2=*/{}},
       {SolverTypeForTRex::TACGP,      "TACGP", /*lambda2=*/{}}
    };

    // Results: solver x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // ===================================================================
    // Generate true support once and shared across all tests
    // ===================================================================
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Generate unique support indices
    std::unordered_set<std::size_t> support_set;
    while (support_set.size() < cardinality_true_support) {
        support_set.insert(uniform_dist(rng));
    }
    std::vector<std::size_t> true_support(support_set.begin(), support_set.end());

    // Generate coefficient values
    std::vector<double> true_coefs;
    true_coefs.reserve(cardinality_true_support);
    if (rnd_coef) {
        // Random coefficients
        for (std::size_t i = 0; i < cardinality_true_support; ++i) {
            true_coefs.push_back(normal(rng));
        }
    } else {
        // Fixed coefficients
        for (std::size_t i = 0; i < cardinality_true_support; ++i) {
            true_coefs.push_back(1);
        }
    }


    std::cout << "True support (cardinality " << cardinality_true_support << "): ";
    for (const auto idx : true_support) {
        std::cout << idx << " ";
    }
    std::cout << "\n\n";

    // ===================================================================
    // Solver loop
    // ===================================================================
    for (const auto& solver_config : solvers_to_test) {

        std::cout << "===================================================\n";
        std::cout << "Solver: " << solver_config.solver_name << "\n";
        std::cout << "===================================================\n\n";

        // ===================================================================
        // Loop over SNR values
        // ===================================================================
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];
            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            double total_fdp = 0.0;
            double total_tpp = 0.0;

            // ===================================================================
            // Monte Carlo loop
            // ===================================================================
            for (std::size_t mc = 0; mc < num_MC; ++mc) {
                // Progress
                std::cout << "  Progress: [" << std::setw(3) << (mc + 1)
                          << "/" << num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * (mc + 1) / num_MC) << "%\r";

                utils_talgos::SyntheticData data(
                    n, p, true_support, true_coefs, snr,
                    /*seed=*/24 + snr_idx * 1000 + mc
                );

                Eigen::Map<Eigen::MatrixXd> X_map(data.X.data(), data.X.rows(), data.X.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

                // Create solver-specific config if needed
                std::unique_ptr<::SolverConfig> solver_cfg = nullptr;
                if (solver_config.solver_type == SolverTypeForTRex::TENET) {
                    solver_cfg = std::make_unique<TENETConfig>(solver_config.lambda2);
                }

                TRexSelector trex(
                    /*X=*/X_map,
                    /*y=*/y_map,
                    /*tFDR=*/0.1,
                    /*K=*/20,
                    /*solver=*/solver_config.solver_type,
                    /*max_num_dummies=*/10,
                    /*max_T_stop=*/true,
                    /*lloop_strategy=*/LLoopStrategy::ADAPTIVE,
                    /*seed=*/-1,
                    /*verbose=*/false,
                    /*solver_config=*/std::move(solver_cfg)
                );

                trex.select();

                auto selected_indices = trex.getSelectedIndices();

                const double fdp = utils_fdr_control::compute_fdp(
                    selected_indices,
                    true_support
                );

                const double tpp = utils_fdr_control::compute_tpp(
                    selected_indices,
                    true_support
                );

                total_fdp += fdp;
                total_tpp += tpp;
            }

            // Store in map
            fdr_results_map[solver_config.solver_name](snr_idx) =
                total_fdp / static_cast<double>(num_MC);
            tpr_results_map[solver_config.solver_name](snr_idx) =
                total_tpp / static_cast<double>(num_MC);

            // Clear progress line
            std::cout << std::string(50, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";
        }

        std::cout << "\n";
    }

    // ===================================================================
    // Print results table
    // ===================================================================
    std::cout << "\n";
    std::cout << "======================================================================\n";
    std::cout << "=== T-Rex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n";
    std::cout << "======================================================================\n\n";

    const int name_width = 15;
    const int col_width = 10;

    // Print SNR
    std::cout << std::setw(name_width) << "Solver";
    std::cout << std::setw(name_width) << "Metric";
    for (const auto snr : snr_values) {
        std::cout << std::fixed << std::setprecision(1) << std::setw(col_width) << snr;
    }
    std::cout << "\n";
    std::cout << std::string(name_width * 2 + col_width * snr_values.size(), '-') << "\n";

    // Print each solver's results
    for (const auto& solver_config : solvers_to_test) {
        // FDR row
        std::cout << std::setw(name_width) << solver_config.solver_name;
        std::cout << std::setw(name_width) << "FDR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << fdr_results_map[solver_config.solver_name](i);
        }
        std::cout << "\n";

        // TPR row
        std::cout << std::setw(name_width) << "";  // Empty solver name
        std::cout << std::setw(name_width) << "TPR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << tpr_results_map[solver_config.solver_name](i);
        }
        std::cout << "\n\n";
    }

    std::cout << "\n\n";
}


// ----------------------------------------------------------------------


void demo_TRexSelector_varMonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    std::cout << "=== Demo: T-Rex Selector Monte Carlo Simulation ===\n\n";

    if (high_dim) {
        std::cout << "High-dimensional setting (p > n)\n";
    } else {
        std::cout << "Low-dimensional setting (n > p)\n";
    }

    const std::size_t n = high_dim ? 300 : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};

    // ===================================================================
    // Define solvers to test
    // ===================================================================

    struct SolverConfig {
        SolverTypeForTRex solver_type;
        std::string solver_name;
        double lambda2;  // For TENET solver
    };

    const std::vector<SolverConfig> solvers_to_test = {
       {SolverTypeForTRex::TLARS,      "TLARS", {}},
       {SolverTypeForTRex::TLASSO,     "TLASSO", {}},
       {SolverTypeForTRex::TENET,      "TENET",      /*lambda2=*/0.1},
       {SolverTypeForTRex::TSTEPWISE,  "TSTEPWISE", {}},
       {SolverTypeForTRex::TOMP,       "TOMP", {}},
       {SolverTypeForTRex::TGP,        "TGP", {}},
       {SolverTypeForTRex::TACGP,      "TACGP", {}}
    };

    // Results: solver x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // ===================================================================
    // Generate true support once and shared across all tests
    // ===================================================================
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    std::normal_distribution<double> normal(0.0, 1.0);


    // ===================================================================
    // Solver loop
    // ===================================================================
    for (const auto& solver_config : solvers_to_test) {

        std::cout << "===================================================\n";
        std::cout << "Solver: " << solver_config.solver_name << "\n";
        std::cout << "===================================================\n\n";

        // ===================================================================
        // Loop over SNR values
        // ===================================================================
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {


            double total_fdp = 0.0;
            double total_tpp = 0.0;

            // ===================================================================
            // Monte Carlo loop
            // ===================================================================
            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Generate unique support indices
                std::unordered_set<std::size_t> support_set;
                while (support_set.size() < cardinality_true_support) {
                    support_set.insert(uniform_dist(rng));
                }
                std::vector<std::size_t> true_support(support_set.begin(), support_set.end());

                // Generate coefficient values
                std::vector<double> true_coefs;
                true_coefs.reserve(cardinality_true_support);
                if (rnd_coef) {
                    // Random coefficients
                    for (std::size_t i = 0; i < cardinality_true_support; ++i) {
                        true_coefs.push_back(normal(rng));
                    }
                } else {
                    // Fixed coefficients
                    for (std::size_t i = 0; i < cardinality_true_support; ++i) {
                        true_coefs.push_back(1);
                    }
                }

                std::cout << "True support (cardinality " << cardinality_true_support << "): ";
                for (const auto idx : true_support) {
                    std::cout << idx << " ";
                }
                std::cout << "\n\n";


                const double snr = snr_values[snr_idx];
                std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

                // Progress
                std::cout << "  Progress: [" << std::setw(3) << (mc + 1)
                          << "/" << num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * (mc + 1) / num_MC) << "%\r";

                utils_talgos::SyntheticData data(
                    n, p, true_support, true_coefs, snr,
                    /*seed=*/24 + snr_idx * 1000 + mc
                );

                Eigen::Map<Eigen::MatrixXd> X_map(data.X.data(), data.X.rows(), data.X.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.y.data(), data.y.size());

                // Create solver-specific config if needed
                std::unique_ptr<::SolverConfig> solver_cfg = nullptr;
                if (solver_config.solver_type == SolverTypeForTRex::TENET) {
                    solver_cfg = std::make_unique<TENETConfig>(solver_config.lambda2);
                }

                TRexSelector trex(
                    /*X=*/X_map,
                    /*y=*/y_map,
                    /*tFDR=*/0.1,
                    /*K=*/20,
                    /*solver=*/solver_config.solver_type,
                    /*max_num_dummies=*/10,
                    /*max_T_stop=*/true,
                    /*lloop_strategy=*/LLoopStrategy::ADAPTIVE,
                    /*seed=*/-1,
                    /*verbose=*/true,
                    /*solver_config=*/std::move(solver_cfg)
                );

                trex.select();

                auto selected_indices = trex.getSelectedIndices();

                const double fdp = utils_fdr_control::compute_fdp(
                    selected_indices,
                    true_support
                );

                const double tpp = utils_fdr_control::compute_tpp(
                    selected_indices,
                    true_support
                );

                total_fdp += fdp;
                total_tpp += tpp;
            }

            // Store in map
            fdr_results_map[solver_config.solver_name](snr_idx) =
                total_fdp / static_cast<double>(num_MC);
            tpr_results_map[solver_config.solver_name](snr_idx) =
                total_tpp / static_cast<double>(num_MC);

            // Clear progress line
            std::cout << std::string(50, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";
        }

        std::cout << "\n";
    }

    // ===================================================================
    // Print results table
    // ===================================================================
    std::cout << "\n";
    std::cout << "======================================================================\n";
    std::cout << "=== T-Rex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n";
    std::cout << "======================================================================\n\n";

    const int name_width = 15;
    const int col_width = 10;

    // Print SNR
    std::cout << std::setw(name_width) << "Solver";
    std::cout << std::setw(name_width) << "Metric";
    for (const auto snr : snr_values) {
        std::cout << std::fixed << std::setprecision(1) << std::setw(col_width) << snr;
    }
    std::cout << "\n";
    std::cout << std::string(name_width * 2 + col_width * snr_values.size(), '-') << "\n";

    // Print each solver's results
    for (const auto& solver_config : solvers_to_test) {
        // FDR row
        std::cout << std::setw(name_width) << solver_config.solver_name;
        std::cout << std::setw(name_width) << "FDR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << fdr_results_map[solver_config.solver_name](i);
        }
        std::cout << "\n";

        // TPR row
        std::cout << std::setw(name_width) << "";  // Empty solver name
        std::cout << std::setw(name_width) << "TPR";
        for (std::size_t i = 0; i < snr_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_width)
                      << tpr_results_map[solver_config.solver_name](i);
        }
        std::cout << "\n\n";
    }

    std::cout << "\n\n";
}


// ----------------------------------------------------------------------

int main() {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    // Run basic T-Rex Selector demo
    // --------------------------------------------------------------
    // demo_TRexSelector(/*high_dim=*/true);


    // Run T-Rex Selector Monte Carlo simulation
    // --------------------------------------------------------------
    // high-dimensional setting
    // demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/true, /*rnd_coef=*/false);

    // low-dimensional setting
    // demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/false);


    // Run T-Rex Selector with variable data Monte Carlo simulation
    // --------------------------------------------------------------
    // high-dimensional setting
    demo_TRexSelector_varMonteCarlo(/*num_MC=*/100, /*high_dim=*/true, /*rnd_coef=*/true);


    return 0;
}
