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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>
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
namespace fs = std::filesystem;

// ==============================================================================



// ==============================================================================
// Demo 1: Basic T-Rex Selector functionality
// ==============================================================================

void demo_TRexSelector(bool high_dim, bool rnd_coef) {

    // Print Demo Header
    cdianostics::print_section_header("Demo: T-Rex Selector Basic Functionality");

    // Setup
    const std::size_t n = high_dim ? 50: 5000;
    const std::size_t p = high_dim ? 150 : 1000;

    const std::vector<std::size_t> true_support = {27, 149, 128, 42, 4};
    const std::vector<double> true_coefs = rnd_coef ?
                                           std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5} :
                                           std::vector<double>{1, 1, 1, 1, 1};
    const double snr = 0.1;
    const double tFDR = 0.1;

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
    trex_ctrl.max_dummy_multiplier = 10;
    trex_ctrl.use_max_T_stop = true;
    trex_ctrl.dummy_distribution = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
    trex_ctrl.tloop_stagnation_stop = false;
    trex_ctrl.tloop_max_stagnant_steps = 5;
    trex_ctrl.parallel_rnd_experiments = false;
    trex_ctrl.solver_type = SolverTypeForTRex::TSTEPWISE;

    // Create T-Rex Selector instance
    std::cout << "Creating T-Rex Selector instance...\n";
    TRexSelector trex(
        /*X=*/X_map,
        /*y=*/y_map,
        /*tFDR=*/tFDR,
        /*trex_control=*/trex_ctrl,
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

    auto phi_prime = trex.getPhiPrime();
    std::cout << "\nAdjusted Relative Occurrences (Phi_prime):\n";
    std::cout << phi_prime.transpose() << "\n";

    auto phi_mat = trex.getPhiMat();
    std::cout << "\nPhi Matrix (Phi):\n";
    std::cout << phi_mat << "\n";

    auto fdp_hat_mat = trex.getFDPHatMat();
    std::cout << "\nEstimated FDP Matrix (FDP_hat):\n";
    std::cout << fdp_hat_mat << "\n";

    auto r_mat = trex.getRMat();
    std::cout << "\nR Matrix (R):\n";
    std::cout << r_mat << "\n";

    auto voting_grid = trex.getVotingGrid();
    std::cout << "\nVoting Grid:\n";
    std::cout << voting_grid.transpose() << "\n";

    double dummy_ratio = trex.getDummyMultiplierL();
    std::cout << "\nDummy to original variable ratio: " << dummy_ratio << "\n";

    std::cout << "\n\n";
}


// ==============================================================================

struct DemoSolverInfo {
    SolverTypeForTRex solver_type;
    std::string solver_name;
    double lambda2;  // For TENET solver
};

void save_and_print_results(
    std::size_t num_MC,
    std::size_t n,
    std::size_t p,
    std::size_t stagnation_window,
    const std::vector<double>& snr_values,
    const std::vector<DemoSolverInfo>& solvers_to_test,
    const std::map<std::string, Eigen::VectorXd>& fdr_results_map,
    const std::map<std::string, Eigen::VectorXd>& tpr_results_map,
    const std::map<std::string, Eigen::VectorXd>& avg_L_results_map,
    const std::map<std::string, Eigen::VectorXd>& avg_T_results_map
) {
    // 1. Setup File Output
    std::string folder = "/simulations/";
    std::string filename = "trex_results_n" + std::to_string(n) + "_p" + std::to_string(p) +
                           "_stagnation_window_" + std::to_string(stagnation_window);
    std::ofstream out_file(filename);

    // Lambda to print to both console and file
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // 2. Print Headers
    std::stringstream ss;
    ss << "\n";
    ss << "======================================================================\n";
    ss << "T-Rex Results (averaged over " << num_MC << " Monte Carlo runs)\n";
    ss << "======================================================================\n";
    print_dual(ss.str());

    // 3. Setup Table Dimensions
    const int name_width = 15;
    const int col_width = 10;

    // 4. Print Table Header (SNR Values)
    std::stringstream ss_header;
    ss_header << std::setw(name_width) << "Solver" << std::setw(name_width) << "Metric";
    for (double snr : snr_values) {
        ss_header << std::fixed << std::setprecision(1) << std::setw(col_width) << snr;
    }
    ss_header << "\n" << std::string(name_width * 2 + col_width * snr_values.size(), '-') << "\n";
    print_dual(ss_header.str());

    // 5. Print Data Rows
    for (const auto& solver : solvers_to_test) {
        std::string name = solver.solver_name;
        std::stringstream ss_row;

        // --- FDR ---
        ss_row << std::setw(name_width) << name << std::setw(name_width) << "FDR";
        for (int i = 0; i < snr_values.size(); ++i) {
            ss_row << std::fixed << std::setprecision(4) << std::setw(col_width) << fdr_results_map.at(name)(i);
        }
        ss_row << "\n";

        // --- TPR ---
        ss_row << std::setw(name_width) << "" << std::setw(name_width) << "TPR";
        for (int i = 0; i < snr_values.size(); ++i) {
            ss_row << std::fixed << std::setprecision(4) << std::setw(col_width) << tpr_results_map.at(name)(i);
        }
        ss_row << "\n";

        // --- Avg L ---
        ss_row << std::setw(name_width) << "" << std::setw(name_width) << "Avg L";
        for (int i = 0; i < snr_values.size(); ++i) {
            ss_row << std::fixed << std::setprecision(4) << std::setw(col_width) << avg_L_results_map.at(name)(i);
        }
        ss_row << "\n";

        // --- Avg T ---
        ss_row << std::setw(name_width) << "" << std::setw(name_width) << "Avg T";
        for (int i = 0; i < snr_values.size(); ++i) {
            ss_row << std::fixed << std::setprecision(4) << std::setw(col_width) << avg_T_results_map.at(name)(i);
        }
        ss_row << "\n";

        print_dual(ss_row.str());
    }

    // 6. Footer
    print_dual("\n");
    if (out_file.is_open()) {
        std::cout << "[Info] Results successfully saved to: " << filename << "\n\n";
        out_file.close();
    }
}


// ==============================================================================
// Demo 2: T-Rex Selector Monte Carlo Simulation with fixed active support
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
    const double tFDR = 0.1;

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
    trex_control.max_dummy_multiplier = 10;
    trex_control.use_max_T_stop = true;
    trex_control.dummy_distribution = dummygen::Distribution::Normal();
    trex_control.lloop_strategy = LLoopStrategy::HCONCAT;
    trex_control.tloop_stagnation_stop = false;
    trex_control.tloop_max_stagnant_steps = 3;

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
                trex_control.solver_type = current_solver.solver_type;
                trex_control.tenet_lambda2 =
                    current_solver.solver_type == SolverTypeForTRex::TENET
                    ? current_solver.lambda2
                    : 1.0;

                // Create T-Rex Selector instance
                TRexSelector trex(
                    /*X=*/X_map,
                    /*y=*/y_map,
                    /*tFDR=*/tFDR,
                    /*trex_control=*/trex_control,
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


// ===============================================================================
// Demo 3: T-Rex Selector Monte Carlo Simulation with variable active support
// ===============================================================================

void demo_TRexSelector_varMonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    cdianostics::print_section_header("Demo: T-Rex Selector Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::size_t n = high_dim ? 300 : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                                            5.0};
    const double tFDR = 0.1;

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
    std::map<std::string, Eigen::VectorXd> average_L_results_map;
    std::map<std::string, Eigen::VectorXd> average_T_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        average_L_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        average_T_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // Setup TRex Parameters
    TRexControlParameter trex_control;
    trex_control.K = 20;
    trex_control.max_dummy_multiplier = 10;
    trex_control.use_max_T_stop = true;
    trex_control.dummy_distribution = dummygen::Distribution::Normal();
    trex_control.lloop_strategy = LLoopStrategy::HCONCAT;

    // TODO: if false, only LAR path algorithms (later on defaulted -> already on list)
    trex_control.tloop_stagnation_stop = true;
    trex_control.tloop_max_stagnant_steps = 7;

    // ===================================================================
    // RNG for support and coefficients generation
    // ===================================================================

    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);

    // ===================================================================
    // Solver loop
    // ===================================================================

    for (const auto& current_solver : solvers_to_test) {

        cdianostics::print_section_header("Solver: " + current_solver.solver_name);

        // ===================================================================
        // Loop over SNR values
        // ===================================================================

        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            double total_fdp = 0.0;
            double total_tpp = 0.0;
            double total_L = 0.0;
            double total_T = 0.0;
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
                    std::normal_distribution<double> normal(0.0, 1.0);
                    for (std::size_t i = 0; i < cardinality_true_support; ++i) {
                        true_coefs.push_back(normal(rng));
                    }
                } else {
                    // Fixed (deterministic/homogeneous) coefficients
                    for (std::size_t i = 0; i < cardinality_true_support; ++i) {
                        true_coefs.push_back(1);
                    }
                }

                // Generate data
                datagen::SyntheticData data(
                    n, p, true_support, true_coefs, snr,
                    /*seed=*/24 + snr_idx * 1000 + mc,
                    /*predictor_config=*/datagen::detail::PredictorConfig::AR1(0.6),
                    /*noise_config=*/datagen::NoiseConfig::Gaussian()
                );

                // Generate maps of data
                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

                // Setup Solver Control Parameters
                trex_control.solver_type = current_solver.solver_type;
                trex_control.tenet_lambda2 =
                    (current_solver.solver_type == SolverTypeForTRex::TENET) ?
                    current_solver.lambda2
                    : 1.0;

                // Create T-Rex Selector instance
                TRexSelector trex(
                    /*X=*/X_map,
                    /*y=*/y_map,
                    /*tFDR=*/tFDR,
                    /*trex_control=*/trex_control,
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
                total_L += trex.getDummyMultiplierL();
                total_T += trex.getStoppingTimeT();
            }

            // Store in map
            fdr_results_map[current_solver.solver_name](snr_idx) =
                total_fdp / static_cast<double>(num_MC);
            tpr_results_map[current_solver.solver_name](snr_idx) =
                total_tpp / static_cast<double>(num_MC);
            average_L_results_map[current_solver.solver_name](snr_idx) =
                total_L / static_cast<double>(num_MC);
            average_T_results_map[current_solver.solver_name](snr_idx) =
                total_T / static_cast<double>(num_MC);

            // Clear progress line
            std::cout << std::string(50, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";
        }

        std::cout << "\n";
    }

    // ===================================================================
    // Print & Save Results
    // ===================================================================
    save_and_print_results(
        num_MC,
        n,
        p,
        trex_control.tloop_max_stagnant_steps,
        snr_values,
        solvers_to_test,
        fdr_results_map,
        tpr_results_map,
        average_L_results_map,
        average_T_results_map
    );
    std::cout << "\n\n";
}



void demo_TRexSelector_PhiMetaAnalysis(int num_MC) {

    // 1. Setup
    const int n = 300;
    const int p = 1000;
    const int k_true = 10;
    const int max_T = 50;
    const double tFDR = 0.1;

    // SNR levels to test
    const std::vector<double> snr_list = {0.1, 0.5, 0.7, 1.0};

    // 2. Fixed Random Support (Same support indices for ALL SNRs)
    std::vector<std::size_t> support(k_true);
    std::random_device rng{};
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    for (int i = 0; i < k_true; ++i) { support[i] = uniform_dist(rng); }

    // Fixed Coefficients
    std::vector<double> coefs(k_true, 1.0);

    // Header
    std::cout << "\n========================================\n";
    std::cout << " Meta-Stability Analysis (LARS vs Greedy)\n";
    std::cout << " Support Indices: ";
    for(auto s : support) { std::cout << s << " "; }
    std::cout << "\n========================================\n\n";

    // 3. Loop over SNRs
    for (double snr : snr_list) {

        // Reset Accumulators for this SNR
        Eigen::MatrixXd Phi_Sum_LARS = Eigen::MatrixXd::Zero(max_T, p);
        Eigen::MatrixXd Phi_Sum_Greedy = Eigen::MatrixXd::Zero(max_T, p);

        // Setup TRex Control (Standard)
        TRexControlParameter trex_ctrl;
        trex_ctrl.K = 20;
        trex_ctrl.tloop_stagnation_stop = true;
        trex_ctrl.tloop_max_stagnant_steps = 5;

        std::cout << ">>> Simulating SNR = " << snr << " (" << num_MC << " runs)...\n";

        // 4. Monte Carlo Loop
        for (int mc = 0; mc < num_MC; ++mc) {
            // Progress
            if (mc % 10 == 0 || mc == num_MC - 1) {
                std::cout << "Run " << (mc+1) << "/" << num_MC << "\r" << std::flush;
            }

            // Generate Data
            datagen::SyntheticData data(
                n, p, support, coefs, snr,
                24 + mc, // Vary seed for noise
                datagen::detail::PredictorConfig::Normal(),
                datagen::NoiseConfig::Gaussian()
            );

            Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), n, p);
            Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), n);

            // -----------------------------------------------------
            // A. Run LARS
            // -----------------------------------------------------
            trex_ctrl.solver_type = SolverTypeForTRex::TLARS;
            TRexSelector trex_tlars(X_map, y_map, tFDR, trex_ctrl, -1, false);
            trex_tlars.select();

            // Accumulate
            const Eigen::MatrixXd& phi_L = trex_tlars.getPhiMat();
            long t_stop_L = phi_L.rows();

            for (int t = 0; t < max_T; ++t) {
                int source_row = std::min((long)t, t_stop_L - 1);
                if (source_row >= 0) {
                    Phi_Sum_LARS.row(t) += phi_L.row(source_row);
                }
            }

            // -----------------------------------------------------
            // B. Run Greedy (Stepwise)
            // -----------------------------------------------------
            trex_ctrl.solver_type = SolverTypeForTRex::TSTEPWISE;
            TRexSelector trex_greedy(X_map, y_map, tFDR, trex_ctrl, -1, false);
            trex_greedy.select();

            // Accumulate
            const Eigen::MatrixXd& phi_G = trex_greedy.getPhiMat();
            long t_stop_G = phi_G.rows();

            for (int t = 0; t < max_T; ++t) {
                int source_row = std::min((long)t, t_stop_G - 1);
                if (source_row >= 0) {
                    Phi_Sum_Greedy.row(t) += phi_G.row(source_row);
                }
            }
        } // End MC Loop

        // 5. Averaging
        Phi_Sum_LARS /= static_cast<double>(num_MC);
        Phi_Sum_Greedy /= static_cast<double>(num_MC);

        // 6. Print Results Table
        auto get_metrics = [&](const Eigen::MatrixXd& Phi_Sum, int t) {
            double sum_signal_phi = 0.0;
            for (size_t idx : support) sum_signal_phi += Phi_Sum(t, idx);
            double avg_signal = sum_signal_phi / static_cast<double>(support.size());

            double max_noise = 0.0;
            for (int j = 0; j < p; ++j) {
                bool is_signal = false;
                for(size_t s : support) if(s == (size_t)j) { is_signal = true; break; }
                if (!is_signal && Phi_Sum(t, j) > max_noise) max_noise = Phi_Sum(t, j);
            }
            return std::make_pair(avg_signal, max_noise);
        };

        std::cout << "\n\nResults (Avg Phi) at SNR=" << snr << ":\n";
        std::cout << "T | LARS Signal | LARS Noise | Greedy Signal | Greedy Noise\n";
        std::cout << "-----|-------------|------------|---------------|-------------\n";

        for (int t : {1, 5, 10, 15, 20, 30, 40}) {
            if (t >= max_T) break;
            auto [lars_sig, lars_noise] = get_metrics(Phi_Sum_LARS, t);
            auto [greedy_sig, greedy_noise] = get_metrics(Phi_Sum_Greedy, t);

            std::cout << std::setw(4) << t << " | "
                      << std::fixed << std::setprecision(3)
                      << lars_sig << "       | "
                      << lars_noise << "      | "
                      << greedy_sig << "         | "
                      << greedy_noise << "\n";
        }
        std::cout << "\n";

    } // End SNR Loop
}



// ==============================================================================
// ==============================================================================


void demo_TRexSelector_memmap(bool high_dim, bool rnd_coef) {

    cdianostics::print_section_header("Demo: T-Rex Selector with Memory-Mapped Data");

    const std::size_t n = high_dim ? 1000 : 5000;
    const std::size_t p = high_dim ? 5000 : 1000;
    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs = rnd_coef ?
                                           std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5} :
                                           std::vector<double>{1, 1, 1, 1, 1};
    const double snr = 1.0;

    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::string X_filepath = "X_mmap.dat";
    const std::string y_filepath = "y_mmap.dat";

    std::cout << "Generating synthetic data...\n";
    datagen::SyntheticDataMapped data(
        X_filepath,
        y_filepath,
        n,
        p,
        true_support,
        true_coefs,
        snr,
        /*seed=*/58
    );

    // Create mapped views as lvalues (required by TRexSelector constructor)
    std::cout << "Creating maps of data...\n";
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // Setup Control Structures
    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 20;
    trex_ctrl.max_dummy_multiplier = 10;
    trex_ctrl.use_max_T_stop = true;
    trex_ctrl.dummy_distribution = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy = LLoopStrategy::HCONCAT;
    trex_ctrl.tloop_stagnation_stop = true;
    trex_ctrl.use_memory_mapping = true;
    trex_ctrl.solver_type = SolverTypeForTRex::TLARS;

    // Create T-Rex Selector instance
    std::cout << "Creating T-Rex Selector instance...\n";
    TRexSelector trex(
        /*X=*/X_map,
        /*y=*/y_map,
        /*tFDR=*/0.1,
        /*trex_control=*/trex_ctrl,
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

    // Clean up memory-mapped files
    fs::remove(X_filepath);
    fs::remove(y_filepath);

    std::cout << "\n\n";
}



// ==============================================================================


void demo_TRexSelector_memmap_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {
    cdianostics::print_section_header("Demo: T-Rex Selector Memory-Mapped Monte Carlo Simulation");

    std::cout << "Not yet implemented.\n\n";
}


// ==============================================================================
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // Run basic T-Rex Selector demo
    // --------------------------------------------------------------------------------------
    if (false)
        demo_TRexSelector(/*high_dim=*/true, /*rnd_coef=*/false);

    // Monte Carlo simulation: Run T-Rex Selector with fixed support & coefficients
    // --------------------------------------------------------------------------------------
    // high-dimensional setting
    if (false)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/true, /*rnd_coef=*/false);

    // low-dimensional setting
    if (false)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/false, /*rnd_coef=*/false);


    // STANDARD SIMULATION
    // -----------------------
    // Monte Carlo simulation: Run T-Rex Selector with variable data, support & coefficients
    // --------------------------------------------------------------------------------------
    // high-dimensional setting
    if (true)
        demo_TRexSelector_varMonteCarlo(/*num_MC=*/500, /*high_dim=*/true, /*rnd_coef=*/false);

    // Phi-Matrix Meta-Stability Analysis
    // --------------------------------------------------------------------------------------
    if (false)
        demo_TRexSelector_PhiMetaAnalysis(/*num_MC=*/500);



    // ===============================================
    // Memory-Mapped DATA WORKFLOWS
    // ===============================================

    // Memory-mapped T-Rex Selector demo
    // --------------------------------------------------------------------------------------
    if (false)
        demo_TRexSelector_memmap(/*high_dim=*/false, /*rnd_coef=*/false);

    // Memory-mapped T-Rex Selector Monte Carlo demo
    // --------------------------------------------------------------------------------------
    if (false)
        demo_TRexSelector_memmap_MonteCarlo(/*num_MC=*/100, /*high_dim=*/false, /*rnd_coef=*/false);


    return 0;
}
