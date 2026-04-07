// ==============================================================================
// demo_trx_01_classic_trex_inmem.cpp
// ==============================================================================
/**
 * @file demo_trx_01_classic_trex_inmem.cpp
 *
 * @brief Demonstration of classical T-Rex Selector.
 *
 * @details Shows basic usage of the classical T-Rex Selector for both low- and
 *          high-dimensional settings.
 *          Furthermore, we compare between in-memory and memory-mapped workflows.
 */
// ==============================================================================

// std includes
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// T-Rex Selector includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Demo utilities
#include "demo_trx_utils.hpp"


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace datagen = trex::utils::datageneration::datagen;
namespace dummygen= trex::utils::datageneration::dummygen;
namespace rates = trex::utils::eval::rates;
namespace fs = std::filesystem;

// T-Rex types
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================

// ==============================================================================
// Demo 1: Basic T-Rex Selector functionality
// ==============================================================================

void demo_TRexSelector(bool high_dim, bool rnd_coef) {

    // Print Demo Header
    cdianostics::print_section_header("Demo: T-Rex Selector Basic Functionality");

    // Setup
    const Eigen::Index n = high_dim ? 150: 5000;
    const Eigen::Index p = high_dim ? 300 : 1000;

    const std::vector<std::size_t> true_support = {
        27, 149, 43 , 128, 42, 4};
    const std::vector<double> true_coefs = rnd_coef ?
        std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5, -1.2} :
        std::vector<double>{1, 1, 1, 1, 1, 1};
    const double snr = 1.0;
    const double tFDR = 0.1;

    // Setup dual output (console + file)
    const std::string folder = "simulations/";
    const std::string filename = "trex_basic_n" + std::to_string(n) +
                                 "_p" + std::to_string(p) + ".txt";
    std::ofstream out_file(folder + filename);
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    print_dual(std::string(high_dim ? "High-dimensional (p > n)"
                                    : "Low-dimensional (n > p)") + "\n");

    // Generate synthetic data
    print_dual("Generating synthetic data...\n");
    datagen::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/58);

    // Create mapped views as lvalues (required by TRexSelector constructor)
    print_dual("Creating maps of data...\n");
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                      data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(),
                                      data.rows());

    // Setup Control Structures
    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 20;
    trex_ctrl.max_dummy_multiplier = 10;
    trex_ctrl.use_max_T_stop = true;
    trex_ctrl.dummy_distribution = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy = LLoopStrategy::HCONCAT;
    trex_ctrl.tloop_stagnation_stop = false;
    trex_ctrl.tloop_max_stagnant_steps = 5;
    trex_ctrl.parallel_rnd_experiments = false;
    trex_ctrl.solver_type = SolverTypeForTRex::TLARS;

    // Create T-Rex Selector instance
    print_dual("Creating T-Rex Selector instance...\n");
    TRexSelector trex(X_map, y_map, tFDR, trex_ctrl, -1, true);

    // Execute T-Rex Selector
    print_dual("Executing T-Rex Selector...\n");
    trex.select();

    const auto selected_indices = trex.getSelectedIndices();
    {
        std::ostringstream ss;
        ss << "Selected indices: ";
        for (const auto& idx : selected_indices) ss << idx << " ";
        ss << "\n";
        print_dual(ss.str());
    }

    const double fdp = rates::compute_fdp(selected_indices, true_support);
    const double tpp = rates::compute_tpp(selected_indices, true_support);
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "False Discovery Proportion (FDP): " << fdp << "\n";
        ss << "True Positive Proportion (TPP):   " << tpp << "\n";
        print_dual(ss.str());
    }

    {
        const auto& phi_prime = trex.getPhiPrime();
        std::ostringstream ss;
        ss << "\nAdjusted Relative Occurrences (Phi_prime):\n"
           << phi_prime.transpose() << "\n";
        print_dual(ss.str());
    }

    {
        const auto& phi_mat = trex.getPhiMat();
        std::ostringstream ss;
        ss << "\nPhi Matrix (Phi):\n" << phi_mat << "\n";
        print_dual(ss.str());
    }

    {
        const auto& fdp_hat_mat = trex.getFDPHatMat();
        std::ostringstream ss;
        ss << "\nEstimated FDP Matrix (FDP_hat):\n" << fdp_hat_mat << "\n";
        print_dual(ss.str());
    }

    {
        const auto& r_mat = trex.getRMat();
        std::ostringstream ss;
        ss << "\nR Matrix (R):\n" << r_mat << "\n";
        print_dual(ss.str());
    }

    {
        const auto& voting_grid = trex.getVotingGrid();
        std::ostringstream ss;
        ss << "\nVoting Grid:\n" << voting_grid.transpose() << "\n";
        print_dual(ss.str());
    }

    print_dual("\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] Results saved to: " << folder + filename << "\n";
        out_file.close();
    }
}


// ==============================================================================
// More sophisticated demos
// ==============================================================================

// ==============================================================================
// Shared configuration helpers
// ==============================================================================

static std::vector<DemoSolverInfo> make_default_solvers_to_test() {
    return {
        {SolverTypeForTRex::TLARS,      "TLARS"},
        {SolverTypeForTRex::TLASSO,     "TLASSO"},
        {SolverTypeForTRex::TENET,      "TENET",
            0.1},
        {SolverTypeForTRex::TSTEPWISE,  "TSTEPWISE"},
        {SolverTypeForTRex::TSTAGEWISE, "TSTAGEWISE"},
        {SolverTypeForTRex::TOMP,       "TOMP"},
        {SolverTypeForTRex::TGP,        "TGP"},
        {SolverTypeForTRex::TACGP,      "TACGP"},
        {SolverTypeForTRex::TMP,        "TMP"},
        {SolverTypeForTRex::TAFS,       "TAFS_rho_0.3",
            0.0, 0.3},
        {SolverTypeForTRex::TAFS,       "TAFS_rho_1.0",
            0.0, 1.0},
        {SolverTypeForTRex::TNCGMP,     "TNCGMP_v1",
            0.0, 0.0, 1},
        {SolverTypeForTRex::TNCGMP,     "TNCGMP_v0",
            0.0, 0.0, 0},
        {SolverTypeForTRex::TOOLS,      "TOOLS"}
    };
}

static TRexControlParameter make_base_trex_control() {
    TRexControlParameter ctrl;
    ctrl.K = 20;
    ctrl.max_dummy_multiplier = 10;
    ctrl.use_max_T_stop = true;
    ctrl.dummy_distribution = dummygen::Distribution::Normal();
    ctrl.lloop_strategy = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop = true;
    return ctrl;
}

// ==============================================================================
// ==============================================================================


// ==============================================================================
// Demo 2: T-Rex Selector Monte Carlo Simulation with fixed active support
// ==============================================================================

void demo_TRexSelector_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    cdianostics::print_section_header("Demo: T-Rex Selector Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const Eigen::Index n = high_dim ? 300 : 1000;
    const Eigen::Index p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    const double tFDR = 0.1;

    // ===================================================================
    // Define solvers to test & T-Rex control parameters
    // ===================================================================

    const std::vector<DemoSolverInfo> solvers_to_test = make_default_solvers_to_test();

    // Results: dim = solver x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // Setup TRex Parameters
    TRexControlParameter trex_control = make_base_trex_control();
    trex_control.tloop_max_stagnant_steps = 7;
    trex_control.parallel_rnd_experiments = false;
    trex_control.opt_threshold = 0.75;
    trex_control.use_memory_mapping = false;


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
                          << (100.0 * static_cast<double>(mc + 1) / static_cast<double>(num_MC))
                          << "%\r";

                // Generate data
                datagen::SyntheticData data(
                    n, p,
                    true_support,
                    true_coefs, snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc)
                );

                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(),
                                                  data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

                // Setup Solver Control Parameters
                trex_control.solver_type = current_solver.solver_type;
                trex_control.solver_params.lambda2 = current_solver.lambda2;

                // Create T-Rex Selector instance
                TRexSelector trex(
                    X_map,
                    y_map,
                    tFDR,
                    trex_control,
                    -1,
                    false
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
            fdr_results_map[current_solver.solver_name](static_cast<Eigen::Index>(snr_idx)) =
                total_fdp / static_cast<double>(num_MC);
            tpr_results_map[current_solver.solver_name](static_cast<Eigen::Index>(snr_idx)) =
                total_tpp / static_cast<double>(num_MC);

            // Clear progress line
            std::cout << std::string(50, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";
        }

        std::cout << "\n";
    }

    // ===================================================================
    // Print and save results
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
        {}, // average_L_results_map not computed in this demo
        {}  // average_T_results_map not computed in this demo
    );

    std::cout << "\n\n";
}


// ==============================================================================
// ==============================================================================


// ===============================================================================
// Demo 3: T-Rex Selector Monte Carlo Simulation with variable active support
// ===============================================================================

void demo_TRexSelector_varMonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    cdianostics::print_section_header("Demo: T-Rex Selector Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const Eigen::Index n = high_dim ? 300 : 1000;
    const Eigen::Index p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        1.9, 2.0, 5.0};
    const double tFDR = 0.1;

    // ===================================================================
    // Define solvers to test & T-Rex control parameters
    // ===================================================================

    const std::vector<DemoSolverInfo> solvers_to_test = make_default_solvers_to_test();

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
    TRexControlParameter trex_control = make_base_trex_control();
    trex_control.tloop_max_stagnant_steps = 3;


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

            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr
                      << "\n";

            // ===================================================================
            // Monte Carlo loop
            // ===================================================================

            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Progress indicator
                std::cout << "  Progress: [" << std::setw(3) << (mc + 1)
                          << "/" << num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * static_cast<double>(mc + 1) / static_cast<double>(num_MC))
                          << "%\r";

                // Generate unique support indices for this MC run
                std::unordered_set<std::size_t> support_set;
                while (support_set.size() < cardinality_true_support) {
                    support_set.insert(uniform_dist(rng));
                }
                std::vector<std::size_t> true_support(support_set.begin(),
                                                      support_set.end());

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
                    n, p,
                    true_support,
                    true_coefs,
                    snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc),
                    datagen::predictor_policy::Normal(),
                    datagen::noisegen::noise_policy::Normal()
                );


                // Generate maps of data
                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                                  data.rows(),
                                                  data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(),
                                                  data.rows());

                // Setup Solver Control Parameters
                trex_control.solver_type = current_solver.solver_type;
                trex_control.solver_params.lambda2 = current_solver.lambda2;

                // Create T-Rex Selector instance
                TRexSelector trex(
                    X_map,
                    y_map,
                    tFDR,
                    trex_control,
                    -1,
                    false
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
                total_L += static_cast<double>(trex.getDummyMultiplierL());
                total_T += static_cast<double>(trex.getStoppingTimeT());
            }

            // Store in map
            fdr_results_map[current_solver.solver_name](
                static_cast<Eigen::Index>(snr_idx)) = total_fdp / static_cast<double>(num_MC);
            tpr_results_map[current_solver.solver_name](
                static_cast<Eigen::Index>(snr_idx)) = total_tpp / static_cast<double>(num_MC);
            average_L_results_map[current_solver.solver_name](
                static_cast<Eigen::Index>(snr_idx)) = total_L / static_cast<double>(num_MC);
            average_T_results_map[current_solver.solver_name](
                static_cast<Eigen::Index>(snr_idx)) = total_T / static_cast<double>(num_MC);

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


// ==============================================================================
// ==============================================================================


int main() {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // Run basic T-Rex Selector demo without Monte Carlo simulation
    // --------------------------------------------------------------------------------------
    if (false)
        demo_TRexSelector(/*high_dim=*/true, /*rnd_coef=*/false);

    // Monte Carlo simulation: Run T-Rex Selector with fixed support & variable data
    // --------------------------------------------------------------------------------------
    // high-dimensional setting
    if (true)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/true, /*rnd_coef=*/false);

    // low-dimensional setting
    if (true)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/100, /*high_dim=*/false, /*rnd_coef=*/false);


    // STANDARD SIMULATION
    // -----------------------
    // Monte Carlo simulation: Run T-Rex Selector with variable data, support & coefficients
    // --------------------------------------------------------------------------------------
    // high-dimensional setting
    if (true)
        demo_TRexSelector_varMonteCarlo(/*num_MC=*/100, /*high_dim=*/true, /*rnd_coef=*/false);

    return 0;
}
// ==============================================================================
