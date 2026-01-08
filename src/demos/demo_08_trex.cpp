// ==============================================================================
// demo_08_trex.cpp
// ==============================================================================
/**
 * @file demo_08_trex.cpp
 *
 * @brief Demonstration of T-Rex Selector.
 *
 * @details Shows basic usage of the T-Rex Selector for both low- and high-
 *          dimensional settings and comparison between in-memory and
 *          memory-mapped workflows.
 */
// ==============================================================================

// std includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <unordered_set>

// Eigen includes
#include <Eigen/Dense>

// T-Rex Selector includes
#include <trex/trex_core/TRex_Selector.hpp>
#include <utils/datagen/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
// Namespace aliases
// ==============================================================================
namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace datagen = trex::utils::datagen;
namespace rates = trex::utils::eval::rates;

// ==============================================================================


// ==============================================================================
// Demo 1: Basic T-Rex Selector functionality
// ==============================================================================

void demo_TRexSelector(bool high_dim, bool rnd_coef) {

    // Print Demo Header
    cdianostics::print_section_header("Demo: T-Rex Selector Basic Functionality");

    // Setup
    const std::size_t n = high_dim ? 1000 : 5000;
    const std::size_t p = high_dim ? 5000 : 1000;

    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs = rnd_coef ?
                                           std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5} :
                                           std::vector<double>{1, 1, 1, 1, 1};
    const double snr = 1.0;

    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    std::cout << "Generating synthetic data...\n";
    datagen::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/58);

    // Create mapped views as lvalues (required by TRexSelector constructor)
    std::cout << "Creating maps of data...\n";

    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // Setup Control Structures
    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 20;
    trex_ctrl.max_num_dummies = 10;
    trex_ctrl.max_T_stop = true;
    trex_ctrl.dummy_distribution = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy = LLoopStrategy::ADAPTIVE;
    trex_ctrl.tloop_stagnation_stop = true;

    SolverControl solver_ctrl;
    solver_ctrl.solver_type = SolverTypeForTRex::TLARS;

    // Create T-Rex Selector instance
    std::cout << "Creating T-Rex Selector instance...\n";
    TRexSelector trex(
        /*X=*/X_map,
        /*y=*/y_map,
        /*tFDR=*/0.1,
        /*trex_control=*/trex_ctrl,
        /*solver_control=*/std::move(solver_ctrl),
        /*seed=*/-1,
        /*verbose=*/true
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

    const double fdp = rates::compute_fdp(
        /*selected_indices=*/selected_indices,
        /*true_support=*/true_support
    );

    const double tpp = rates::compute_tpp(
        /*selected_indices=*/selected_indices,
        /*true_support=*/true_support
    );

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "False Discovery Proportion (FDP): " << fdp << "\n";
    std::cout << "True Positive Proportion (TPP):   " << tpp << "\n";

    std::cout << "\n\n";
}


// ==============================================================================

struct DemoSolverInfo {
    SolverTypeForTRex solver_type;
    std::string solver_name;
    double lambda2;  // For TENET solver
};

// ==============================================================================
// Demo 2: T-Rex Selector Monte Carlo Simulation
// ==============================================================================

void demo_TRexSelector_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    cdianostics::print_section_header("Demo: T-Rex Selector Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::size_t n = high_dim ? 300 : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};

    // ===================================================================
    // Define solvers to test & T-Rex control parameters
    // ===================================================================

    const std::vector<DemoSolverInfo> solvers_to_test = {
       {SolverTypeForTRex::TLARS,      "TLARS",     /*lambda2=*/{}  },
       {SolverTypeForTRex::TLASSO,     "TLASSO",    /*lambda2=*/{}  },
       {SolverTypeForTRex::TENET,      "TENET",     /*lambda2=*/0.1 },
       {SolverTypeForTRex::TSTEPWISE,  "TSTEPWISE", /*lambda2=*/{}  },
       {SolverTypeForTRex::TOMP,       "TOMP",      /*lambda2=*/{}  },
       {SolverTypeForTRex::TGP,        "TGP",       /*lambda2=*/{}  },
       {SolverTypeForTRex::TACGP,      "TACGP",     /*lambda2=*/{}  }
    };

    // Results: dim = solver x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // Setup TRex Parameters
    TRexControlParameter trex_control;
    trex_control.K = 20;
    trex_control.max_num_dummies = 10;
    trex_control.max_T_stop = true;
    trex_control.dummy_distribution = dummygen::Distribution::Normal();
    trex_control.lloop_strategy = LLoopStrategy::ADAPTIVE;
    trex_control.tloop_stagnation_stop = true;
    trex_control.max_stagnant_steps = 3;

    // ===================================================================
    // Generate true support once and shared across all Monte Carlo runs
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


    std::cout << "True support (cardinality " << cardinality_true_support << "): ";
    for (const auto idx : true_support) {
        std::cout << idx << " ";
    }
    std::cout << "\n\n";

    // ===================================================================
    // Solver loop
    // ===================================================================
    for (const auto& current_solver : solvers_to_test) {

        std::cout << "===================================================\n";
        std::cout << "Solver: " << current_solver.solver_name << "\n";
        std::cout << "===================================================\n\n";

        // ===================================================================
        // Loop over SNR values
        // ===================================================================
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {

            double total_fdp = 0.0;
            double total_tpp = 0.0;
            const double snr = snr_values[snr_idx];

            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            // ===================================================================
            // Monte Carlo loop
            // ===================================================================
            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Progress
                std::cout << "  Progress: [" << std::setw(3) << (mc + 1)
                          << "/" << num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * (mc + 1) / num_MC) << "%\r";

                // Generate data
                datagen::SyntheticData data(
                    n, p, true_support, true_coefs, snr,
                    /*seed=*/24 + snr_idx * 1000 + mc
                );

                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

                // Setup Solver Control Parameters
                SolverControl solver_control;
                solver_control.solver_type = current_solver.solver_type;
                solver_control.tenet_lambda2 =
                    current_solver.solver_type == SolverTypeForTRex::TENET
                    ? current_solver.lambda2
                    : 1.0;

                // Create T-Rex Selector instance
                TRexSelector trex(
                    /*X=*/X_map,
                    /*y=*/y_map,
                    /*tFDR=*/0.1,
                    /*trex_control=*/trex_control,
                    /*solver_control=*/std::move(solver_control),
                    /*seed=*/-1,
                    /*verbose=*/false
                );

                // Execute T-Rex Selector
                trex.select();

                // Evaluate performance
                auto selected_indices = trex.getSelectedIndices();

                const double fdp = rates::compute_fdp(
                    selected_indices,
                    true_support
                );

                const double tpp = rates::compute_tpp(
                    selected_indices,
                    true_support
                );

                total_fdp += fdp;
                total_tpp += tpp;
            }

            // Store in map
            fdr_results_map[current_solver.solver_name](snr_idx) =
                total_fdp / static_cast<double>(num_MC);
            tpr_results_map[current_solver.solver_name](snr_idx) =
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

    cdianostics::print_section_header("Demo: T-Rex Selector Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::size_t n = high_dim ? 300 : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};

    // ===================================================================
    // Define solvers to test & T-Rex control parameters
    // ===================================================================

    const std::vector<DemoSolverInfo> solvers_to_test = {
       {SolverTypeForTRex::TLARS,      "TLARS",     /*lambda2=*/{}  },
       {SolverTypeForTRex::TLASSO,     "TLASSO",    /*lambda2=*/{}  },
       {SolverTypeForTRex::TENET,      "TENET",     /*lambda2=*/0.1 },
       {SolverTypeForTRex::TSTEPWISE,  "TSTEPWISE", /*lambda2=*/{}  },
       {SolverTypeForTRex::TOMP,       "TOMP",      /*lambda2=*/{}  },
       {SolverTypeForTRex::TGP,        "TGP",       /*lambda2=*/{}  },
       {SolverTypeForTRex::TACGP,      "TACGP",     /*lambda2=*/{}  }
    };

    // Results: solver x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // Setup TRex Parameters
    TRexControlParameter trex_control;
    trex_control.K = 20;
    trex_control.max_num_dummies = 10;
    trex_control.max_T_stop = true;
    trex_control.lloop_strategy = LLoopStrategy::ADAPTIVE;
    trex_control.tloop_stagnation_stop = true;
    trex_control.max_stagnant_steps = 3;

    // ===================================================================
    // RNG for support and coefficients generation
    // ===================================================================
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    std::normal_distribution<double> normal(0.0, 1.0);


    // ===================================================================
    // Solver loop
    // ===================================================================
    for (const auto& current_solver : solvers_to_test) {

        std::cout << std::string(70, '=') << "\n";
        std::cout << "Solver: " << current_solver.solver_name << "\n";
        std::cout << std::string(70, '=') << "\n";

        // ===================================================================
        // Loop over SNR values
        // ===================================================================
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            double total_fdp = 0.0;
            double total_tpp = 0.0;
            const double snr = snr_values[snr_idx];

            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            // ===================================================================
            // Monte Carlo loop
            // ===================================================================
            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Progress indicator
                std::cout << "  Progress: [" << std::setw(3) << (mc + 1)
                          << "/" << num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * (mc + 1) / num_MC) << "%\r";

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

                std::cout << "True support (cardinality " << cardinality_true_support << "): ";
                for (const auto idx : true_support) {
                    std::cout << idx << " ";
                }
                std::cout << "\n\n";

                // Generate data
                datagen::SyntheticData data(
                    n, p, true_support, true_coefs, snr,
                    /*seed=*/24 + snr_idx * 1000 + mc
                );

                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

                // Setup Solver Control Parameters
                SolverControl solver_control;
                solver_control.solver_type = current_solver.solver_type;
                solver_control.tenet_lambda2 =
                    current_solver.solver_type == SolverTypeForTRex::TENET
                    ? current_solver.lambda2
                    : 1.0;

                // Create T-Rex Selector instance
                TRexSelector trex(
                    /*X=*/X_map,
                    /*y=*/y_map,
                    /*tFDR=*/0.1,
                    /*trex_control=*/trex_control,
                    /*solver_control=*/std::move(solver_control),
                    /*seed=*/-1,
                    /*verbose=*/false
                );

                // Execute T-Rex Selector
                trex.select();

                // Evaluate performance
                auto selected_indices = trex.getSelectedIndices();

                const double fdp = rates::compute_fdp(
                    selected_indices,
                    true_support
                );

                const double tpp = rates::compute_tpp(
                    selected_indices,
                    true_support
                );

                total_fdp += fdp;
                total_tpp += tpp;
            }

            // Store in map
            fdr_results_map[current_solver.solver_name](snr_idx) =
                total_fdp / static_cast<double>(num_MC);
            tpr_results_map[current_solver.solver_name](snr_idx) =
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
    std::cout << std::string(80, '=') << "\n";
    std::cout << "=== T-Rex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n";
    std::cout << std::string(80, '=') << "\n";

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
    // --------------------------------------------------------------------------------------
    demo_TRexSelector(/*high_dim=*/true, /*rnd_coef=*/false);


    // Monte Carlo simulation: Run T-Rex Selector with fixed support & coefficients
    // --------------------------------------------------------------------------------------
    // high-dimensional setting
    if (false)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/true, /*rnd_coef=*/false);

    // low-dimensional setting
    if (false)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/false, /*rnd_coef=*/false);


    // Monte Carlo simulation: Run T-Rex Selector with variable data, support & coefficients
    // --------------------------------------------------------------------------------------
    // high-dimensional setting
    if (false)
        demo_TRexSelector_varMonteCarlo(/*num_MC=*/100, /*high_dim=*/true, /*rnd_coef=*/false);


    return 0;
}
