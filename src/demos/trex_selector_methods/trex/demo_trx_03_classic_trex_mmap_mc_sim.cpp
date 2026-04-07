// ==============================================================================
// demo_trx_03_classic_trex_mmap_mc_sim.cpp
// ==============================================================================
/**
 * @file demo_trx_03_classic_trex_mmap_mc_sim.cpp
 *
 * @brief Monte Carlo simulations for T-Rex Selector with memory-mapped workflows.
 *
 * @details Contains two Monte Carlo simulation scenarios that mirror the
 *          corresponding single-run variants in demo_trx_02_classic_trex_mmap.cpp:
 *
 *  (C) MC with in-memory X + use_memory_mapping=true:
 *      Activates solver serialization and memory-mapped dummy matrices D.
 *      Purpose: verify that the D-mmap + solver-serialization pipeline produces
 *      stable, reproducible FDR/TPR statistics over many runs.
 *
 *  (D) MC with fully memory-mapped pipeline (X + D + solver serialization):
 *      Each MC iteration creates its own mmap-backed X/y files and destroys them
 *      via a RAII guard at the end of the iteration scope — exception-safe.
 *
 * Results are written as:
 *  - Aligned text table (.txt) to simulations/ (dual console + file output)
 *  - Tidy long-format CSV (.csv) to simulations/ for plotting
 */
// ==============================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

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
namespace datagen     = trex::utils::datageneration::datagen;
namespace dummygen    = trex::utils::datageneration::dummygen;
namespace rates       = trex::utils::eval::rates;
namespace fs          = std::filesystem;

// T-Rex types
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================


// ==============================================================================
// Shared helpers (self-contained; mirrors demo_trx_02_classic_trex_mmap.cpp)
// ==============================================================================

/**
 * @brief Shared TRexControlParameter baseline for all mmap demos.
 * @note  use_memory_mapping = true couples two behaviours:
 *   (1) Internal dummy matrices D are memory-mapped (TRexMemMapManager).
 *   (2) Solver warm-start state is serialized to disk between T-loop iterations
 *       (SERIALIZED WarmStartMode). Solver footprint scales as O(T * (p + L)).
 */
static TRexControlParameter make_mmap_trex_ctrl() {
    TRexControlParameter ctrl;
    ctrl.K                     = 20;
    ctrl.max_dummy_multiplier  = 10;
    ctrl.use_max_T_stop        = true;
    ctrl.dummy_distribution    = dummygen::Distribution::Normal();
    ctrl.lloop_strategy        = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop = true;
    ctrl.use_memory_mapping    = true;  // enables D mmap + solver serialization
    ctrl.solver_type           = SolverTypeForTRex::TLARS;
    return ctrl;
}

// ==============================================================================


// ==============================================================================
// Demo C: Monte Carlo — In-memory X + use_memory_mapping=true
//         Fixed true support; SNR sweep; TLARS solver.
//         Purpose: verify that the D-mmap + solver-serialization pipeline
//         produces stable, reproducible FDR/TPR statistics over many runs.
// ==============================================================================

void demo_TRexSelector_d_mmap_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf);

    cdianostics::print_section_header(
        "Demo C: MC — In-Memory X, Solver Serialization + D Memory-Mapping");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const Eigen::Index n = high_dim ? 300 : 1000;
    const Eigen::Index p = high_dim ? 1000 : 300;
    const std::size_t  s = 10;    // cardinality of true support
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    const double tFDR = 0.1;
    std::cout << "n = " << n << ", p = " << p << "\n";

    const std::vector<std::string> solver_names = {"TLARS"};
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;
    std::map<std::string, Eigen::VectorXd> avg_L_results_map;
    std::map<std::string, Eigen::VectorXd> avg_T_results_map;
    for (const auto& name : solver_names) {
        fdr_results_map[name]   = Eigen::VectorXd(snr_values.size());
        tpr_results_map[name]   = Eigen::VectorXd(snr_values.size());
        avg_L_results_map[name] = Eigen::VectorXd(snr_values.size());
        avg_T_results_map[name] = Eigen::VectorXd(snr_values.size());
    }

    TRexControlParameter trex_ctrl = make_mmap_trex_ctrl();
    trex_ctrl.tloop_max_stagnant_steps = 7;
    trex_ctrl.parallel_rnd_experiments = false;
    trex_ctrl.opt_threshold            = 0.75;

    // ===================================================================
    // Generate fixed true support and coefficients (shared across all MC runs)
    // ===================================================================
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> udist(0, static_cast<std::size_t>(p) - 1);
    std::normal_distribution<double> ndist(0.0, 1.0);

    std::unordered_set<std::size_t> support_set;
    while (support_set.size() < s) support_set.insert(udist(rng));
    const std::vector<std::size_t> true_support(support_set.begin(), support_set.end());

    std::vector<double> true_coefs;
    true_coefs.reserve(s);
    for (std::size_t i = 0; i < s; ++i)
        true_coefs.push_back(rnd_coef ? ndist(rng) : 1.0);

    std::cout << "True support (cardinality " << s << "): ";
    for (const auto idx : true_support) std::cout << idx << " ";
    std::cout << "\n\n";

    // ===================================================================
    // Solver loop → SNR loop → MC loop
    // ===================================================================
    for (const auto& solver_name : solver_names) {

        cdianostics::print_section_header("Solver: " + solver_name);

        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {

            double total_fdp = 0.0, total_tpp = 0.0;
            double total_L   = 0.0, total_T   = 0.0;
            const double snr = snr_values[snr_idx];
            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                std::cout << "  Progress: [" << std::setw(3) << (mc + 1)
                          << "/" << num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * static_cast<double>(mc + 1) / static_cast<double>(num_MC))
                          << "%\r";

                datagen::SyntheticData data(
                    n, p, true_support, true_coefs, snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc));

                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                                  data.rows(),
                                                  data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(),
                                                  data.rows());

                trex_ctrl.solver_type = SolverTypeForTRex::TLARS;

                TRexSelector trex(X_map,
                                  y_map, tFDR,
                                  trex_ctrl,
                                  -1,
                                  false);
                trex.select();

                const auto sel = trex.getSelectedIndices();
                total_fdp += rates::compute_fdp(sel, true_support);
                total_tpp += rates::compute_tpp(sel, true_support);
                total_L   += static_cast<double>(trex.getDummyMultiplierL());
                total_T   += static_cast<double>(trex.getStoppingTimeT());
            }

            const auto ei = static_cast<Eigen::Index>(snr_idx);
            fdr_results_map[solver_name](ei)   = total_fdp / static_cast<double>(num_MC);
            tpr_results_map[solver_name](ei)   = total_tpp / static_cast<double>(num_MC);
            avg_L_results_map[solver_name](ei) = total_L   / static_cast<double>(num_MC);
            avg_T_results_map[solver_name](ei) = total_T   / static_cast<double>(num_MC);

            std::cout << std::string(50, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";
        }
        std::cout << "\n";
    }

    const std::string stem = "trex_mmap_demo_c_n" + std::to_string(n) +
                             "_p" + std::to_string(p) +
                             "_sw" + std::to_string(trex_ctrl.tloop_max_stagnant_steps);
    save_and_print_mc_results(num_MC, stem, snr_values, solver_names,
                              fdr_results_map, tpr_results_map,
                              avg_L_results_map, avg_T_results_map);
    std::cout << "\n\n";
}


// ==============================================================================
// Demo D: Monte Carlo — Fully memory-mapped X + use_memory_mapping=true
//         Fixed true support; SNR sweep; TLARS solver.
//         Each MC iteration creates and destroys its own mmap-backed X/y files.
//
//         SAFETY NOTE: MmapFileGuard is declared BEFORE SyntheticDataMapped
//         inside the MC scope so that C++ reverse-LIFO destruction gives the
//         correct order: data dtor closes mmap handles → guard dtor removes files.
//         This holds even if TRexSelector::select() throws.
// ==============================================================================

void demo_TRexSelector_full_mmap_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf);

    cdianostics::print_section_header(
        "Demo D: MC — Fully Memory-Mapped X + D + Solver Serialization");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const Eigen::Index n = high_dim ? 300 : 1000;
    const Eigen::Index p = high_dim ? 1000 : 300;
    const std::size_t  s = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    const double tFDR = 0.1;
    std::cout << "n = " << n << ", p = " << p << "\n";

    const std::vector<std::string> solver_names = {"TLARS"};
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;
    std::map<std::string, Eigen::VectorXd> avg_L_results_map;
    std::map<std::string, Eigen::VectorXd> avg_T_results_map;
    for (const auto& name : solver_names) {
        fdr_results_map[name]   = Eigen::VectorXd(snr_values.size());
        tpr_results_map[name]   = Eigen::VectorXd(snr_values.size());
        avg_L_results_map[name] = Eigen::VectorXd(snr_values.size());
        avg_T_results_map[name] = Eigen::VectorXd(snr_values.size());
    }

    TRexControlParameter trex_ctrl = make_mmap_trex_ctrl();
    trex_ctrl.tloop_max_stagnant_steps = 7;
    trex_ctrl.parallel_rnd_experiments = false;
    trex_ctrl.opt_threshold            = 0.75;

    // ===================================================================
    // Generate fixed true support and coefficients (shared across all MC runs)
    // ===================================================================
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> udist(0, static_cast<std::size_t>(p) - 1);
    std::normal_distribution<double> ndist(0.0, 1.0);

    std::unordered_set<std::size_t> support_set;
    while (support_set.size() < s) support_set.insert(udist(rng));
    const std::vector<std::size_t> true_support(support_set.begin(),
                                                support_set.end());

    std::vector<double> true_coefs;
    true_coefs.reserve(s);
    for (std::size_t i = 0; i < s; ++i)
        true_coefs.push_back(rnd_coef ? ndist(rng) : 1.0);

    std::cout << "True support (cardinality " << s << "): ";
    for (const auto idx : true_support) std::cout << idx << " ";
    std::cout << "\n\n";

    // Fixed filenames — safe because iterations are sequential.
    // If parallelised with OpenMP, use per-thread names (e.g. X_mmap_mc_tid{N}.dat).
    const std::string X_filepath = "X_mmap_mc.dat";
    const std::string y_filepath = "y_mmap_mc.dat";

    // ===================================================================
    // Solver loop → SNR loop → MC loop
    // ===================================================================
    for (const auto& solver_name : solver_names) {

        cdianostics::print_section_header("Solver: " + solver_name);

        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {

            double total_fdp = 0.0, total_tpp = 0.0;
            double total_L   = 0.0, total_T   = 0.0;
            const double snr = snr_values[snr_idx];
            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                std::cout << "  Progress: [" << std::setw(3) << (mc + 1)
                          << "/" << num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * static_cast<double>(mc + 1) / static_cast<double>(num_MC))
                          << "%\r";

                // Guard declared BEFORE data: LIFO destruction order:
                // (1) data dtor closes mmap handles,
                // (2) mc_guard dtor removes files.
                MmapFileGuard mc_guard{{X_filepath, y_filepath}};
                datagen::SyntheticDataMapped data(
                    X_filepath, y_filepath,
                    n, p, true_support, true_coefs, snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc));

                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                                  data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

                trex_ctrl.solver_type = SolverTypeForTRex::TLARS;

                TRexSelector trex(X_map,
                                  y_map,
                                  tFDR,
                                  trex_ctrl,
                                  -1,
                                  false);
                trex.select();

                const auto sel = trex.getSelectedIndices();
                total_fdp += rates::compute_fdp(sel, true_support);
                total_tpp += rates::compute_tpp(sel, true_support);
                total_L   += static_cast<double>(trex.getDummyMultiplierL());
                total_T   += static_cast<double>(trex.getStoppingTimeT());
                // Scope end: trex → X_map/y_map → data (closes handles) → mc_guard (removes files)
            }

            const auto ei = static_cast<Eigen::Index>(snr_idx);
            fdr_results_map[solver_name](ei)   = total_fdp / static_cast<double>(num_MC);
            tpr_results_map[solver_name](ei)   = total_tpp / static_cast<double>(num_MC);
            avg_L_results_map[solver_name](ei) = total_L   / static_cast<double>(num_MC);
            avg_T_results_map[solver_name](ei) = total_T   / static_cast<double>(num_MC);

            std::cout << std::string(50, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";
        }
        std::cout << "\n";
    }

    const std::string stem = "trex_mmap_demo_d_n" + std::to_string(n) +
                             "_p" + std::to_string(p) +
                             "_sw" + std::to_string(trex_ctrl.tloop_max_stagnant_steps);
    save_and_print_mc_results(num_MC, stem, snr_values, solver_names,
                              fdr_results_map, tpr_results_map,
                              avg_L_results_map, avg_T_results_map);
    std::cout << "\n\n";
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // ============================================================
    // Demo C: MC — in-memory X + solver serialization + D mmap
    // ============================================================
    if (false)
        demo_TRexSelector_d_mmap_MonteCarlo(
            50,
            true,
            false
        );

    // ============================================================
    // Demo D: MC — fully memory-mapped pipeline (X + D + solver serialization)
    // ============================================================
    if (false)
        demo_TRexSelector_full_mmap_MonteCarlo(
            50,
            true,
            false
        );

    return 0;
}
// ==============================================================================
