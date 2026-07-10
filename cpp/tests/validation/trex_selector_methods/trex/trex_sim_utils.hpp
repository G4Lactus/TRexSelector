// ==============================================================================
// trex_sim_utils.hpp
// ==============================================================================
/**
 * @file trex_sim_utils.hpp
 *
 * @brief Shared simulation utilities for classic T-Rex demo files.
 *
 * @details Provides all types, output helpers, and MC loop infrastructure
 *          shared by the classic T-Rex demo files, all under namespace trex_sim:
 *
 *   Types:
 *    - TrexDGPData               — data container (owned X, y, true_support)
 *    - TrexGridPointResult       — MC aggregate (avg/sd FDR, TPR, L, T)
 *    - TrexDGPFactory            — factory callable type: seed → TrexDGPData
 *    - TrexMmapDGPFactory        — mmap factory type: seed, X_path, y_path → TrexDGPData
 *    - DemoSolverInfo            — solver descriptor (type, name, tuning parameters)
 *    - MmapFileGuard             — RAII guard: deletes mmap backing files on scope exit
 *    - PrintFn                   — dual console+file output callback type
 *
 *  Output functions:
 *    - make_default_solvers_to_test() — canonical list of 14 T-Rex base solvers
 *    - save_selection_csv()           — writes a per-variable tidy CSV
 *    - print_results()                — formats a single-run selection result block
 *    - save_and_print_results()       — MC summary table + tidy CSV (DemoSolverInfo API)
 *    - save_and_print_mc_results()    — MC summary table + tidy CSV (string-name API)
 *
 *  MC loop functions:
 *    - run_mc_trials_trex()      — parallel MC loop using in-memory TRexSelector
 *    - run_mc_trials_trex_mmap() — parallel MC loop using memory-mapped X
 *
 * Parallelism pattern:
 *   #pragma omp parallel for schedule(dynamic)
 *   for (int mc = 0; mc < iMC; ++mc) { ... }
 *
 * Thread safety:
 *   - Each trial owns its data (returned by-value from factory).
 *   - TRexControlParameter is passed by const-ref; TRexSelector copies it by value.
 *   - mmap variant uses per-thread file names derived from omp_get_thread_num().
 *   - WarmStartManager / TRexMemMapManager use TempDirManager::create() which
 *     generates unique directories via std::random_device — safe under OpenMP.
 *
 * @note Demo-internal header — not part of the TRexSelector library interface.
 *       Never include from library code.
 */
// ==============================================================================

#ifndef TREX_DEMOS_TREX_SIM_UTILS_HPP
#define TREX_DEMOS_TREX_SIM_UTILS_HPP

// std includes
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
// Namespace
// ==============================================================================

namespace trex_sim {

namespace rates = trex::utils::eval::rates;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;


// ==============================================================================
// Data and result types
// ==============================================================================

/** @brief Data container returned by a DGP factory for one MC trial. */
struct TrexDGPData {
    Eigen::MatrixXd          X;
    Eigen::VectorXd          y;
    std::vector<std::size_t> true_support;
};

/** @brief Aggregate MC result for one grid point (solver × parameter value). */
struct TrexGridPointResult {
    double avg_fdr = 0.0;   ///< Mean FDR across MC trials
    double avg_tpr = 0.0;   ///< Mean TPR across MC trials
    double avg_L   = 0.0;   ///< Mean dummy multiplier L
    double avg_T   = 0.0;   ///< Mean stopping time T
    double sd_fdr  = 0.0;   ///< Bessel-corrected sd of FDR
    double sd_tpr  = 0.0;   ///< Bessel-corrected sd of TPR
};

/** @brief Factory callable: given a trial seed, produce one MC dataset. */
using TrexDGPFactory = std::function<TrexDGPData(unsigned seed)>;

/**
 * @brief Factory callable for fully memory-mapped X.
 *
 * Given a trial seed and per-thread file paths, produce one MC dataset with
 * data copied into owned matrices. The factory is responsible for the complete
 * lifecycle of the SyntheticDataMapped object and its backing files (via
 * MmapFileGuard declared before SyntheticDataMapped inside the lambda body,
 * ensuring LIFO cleanup: mmap handles closed before files are removed).
 */
using TrexMmapDGPFactory =
    std::function<TrexDGPData(unsigned seed,
                              const std::string& X_path,
                              const std::string& y_path)>;


// ==============================================================================
// Solver descriptor and RAII helpers
// ==============================================================================

/**
 * @brief Solver descriptor used by demo MC simulations.
 */
struct DemoSolverInfo {

    SolverTypeForTRex solver_type;
    std::string       solver_name;
    double            lambda2       = 0.0;
    double            rho_afs       = 0.0;
    int               ncgmp_variant = 0;
};


/**
 * @brief RAII guard that removes memory-mapped backing files on scope exit.
 * @note  Declare the guard BEFORE the SyntheticDataMapped object so that
 *        C++ reverse-LIFO destruction closes the mmap handles (data dtor)
 *        before the files are removed (guard dtor). Exception-safe.
 */
struct MmapFileGuard {
    std::vector<std::filesystem::path> paths;
    ~MmapFileGuard() {
        for (const auto& p : paths) {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    }
};


/** @brief Dual console + file output callback type. */
using PrintFn = std::function<void(const std::string&)>;


// ==============================================================================
// Shared solver list
// ==============================================================================

/**
 * @brief Returns the canonical list of 14 T-Rex base solvers used across MC demos.
 *
 * Shared by demo_trex_02_mc_sim_fixed_support.cpp and
 * demo_trex_03_mc_sim_variable_support.cpp.
 */
inline std::vector<DemoSolverInfo> make_default_solvers_to_test() {
    return {
        {SolverTypeForTRex::TLARS,      "TLARS"},
        {SolverTypeForTRex::TLASSO,     "TLASSO"},
        {SolverTypeForTRex::TENET,      "TENET",
            0.1},
        {SolverTypeForTRex::TSTAGEWISE, "TSTAGEWISE"},
        {SolverTypeForTRex::TSTEPWISE,  "TSTEPWISE"},
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


// ==============================================================================
// Output functions
// ==============================================================================

/**
 * @brief Write per-variable selection results as a tidy CSV for plotting.
 * @details Columns: variable_index, phi_prime, selected, true_positive
 */
inline void save_selection_csv(
    const std::string&              filepath,
    Eigen::Index                    p,
    const Eigen::VectorXd&          phi_prime,
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support)
{
    std::ofstream csv(filepath);
    if (!csv.is_open()) {
        std::cout << "[Warning] Could not open CSV file: " << filepath << "\n";
        return;
    }
    csv << "variable_index,phi_prime,selected,true_positive\n";
    csv << std::fixed << std::setprecision(6);
    for (Eigen::Index i = 0; i < p; ++i) {
        const auto ui      = static_cast<std::size_t>(i);
        const bool is_sel  = std::find(selected_indices.begin(),
                                       selected_indices.end(), ui)
                             != selected_indices.end();
        const bool is_true = std::find(true_support.begin(),
                                       true_support.end(), ui)
                             != true_support.end();
        csv << i << "," << phi_prime(i) << ","
            << (is_sel ? 1 : 0) << "," << (is_true ? 1 : 0) << "\n";
    }
    std::cout << "[Info] CSV saved to:                  " << filepath << "\n";
}


/**
 * @brief Print and save all selection outputs (indices, metrics, matrices).
 *
 * @param selected_indices  Indices selected by T-Rex.
 * @param true_support      Ground-truth active variables.
 * @param selector          Completed TRexSelector instance (for matrix accessors).
 * @param out               Dual output callback (console + file).
 */
inline void print_results(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support,
    TRexSelector&                   selector,
    const PrintFn&                  out)
{
    {
        std::ostringstream ss;
        ss << "Selected indices: ";
        for (const auto& idx : selected_indices) { ss << idx << " "; }
        ss << "\n";
        out(ss.str());
    }

    const double fdp = rates::compute_fdp(selected_indices, true_support);
    const double tpp = rates::compute_tpp(selected_indices, true_support);
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "False Discovery Proportion (FDP): " << fdp << "\n";
        ss << "True Positive Proportion (TPP):   " << tpp << "\n";
        out(ss.str());
    }

    {
        std::ostringstream ss;
        ss << "\nAdjusted Relative Occurrences (Phi_prime):\n"
           << selector.getPhiPrime().transpose() << "\n";
        out(ss.str());
    }

    {
        std::ostringstream ss;
        ss << "\nPhi Matrix (Phi):\n" << selector.getPhiMat() << "\n";
        out(ss.str());
    }

    {
        std::ostringstream ss;
        ss << "\nEstimated FDP Matrix (FDP_hat):\n" << selector.getFDPHatMat() << "\n";
        out(ss.str());
    }

    {
        std::ostringstream ss;
        ss << "\nR Matrix (R):\n" << selector.getRMat() << "\n";
        out(ss.str());
    }

    {
        std::ostringstream ss;
        ss << "\nVoting Grid:\n" << selector.getVotingGrid().transpose() << "\n";
        out(ss.str());
    }
}


/**
 * @brief Print averaged MC results as an aligned table (console + .txt) and
 *        write a tidy long-format CSV. Uses DemoSolverInfo for solver metadata.
 *
 * @param num_MC               Number of Monte Carlo repetitions.
 * @param n                    Number of observations.
 * @param p                    Number of predictors.
 * @param stagnation_window    Value of tloop_max_stagnant_steps (used in filename).
 * @param snr_values           SNR grid.
 * @param solvers_to_test      Solver descriptors (determines row order).
 * @param fdr_results_map      Averaged FDR per solver x SNR.
 * @param tpr_results_map      Averaged TPR per solver x SNR.
 * @param avg_L_results_map    Averaged dummy multiplier L (empty = omit).
 * @param avg_T_results_map    Averaged stopping time T   (empty = omit).
 */
inline void save_and_print_results(
    std::size_t num_MC,
    std::size_t n,
    std::size_t p,
    std::size_t stagnation_window,
    const std::vector<double>&                     snr_values,
    const std::vector<DemoSolverInfo>&             solvers_to_test,
    const std::map<std::string, Eigen::VectorXd>&  fdr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_L_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_T_results_map,
    const std::string&                             file_prefix = "")
{
    // 1. Setup File Output
    // -----------------------------------
    std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);
    std::string filename = file_prefix + "trex_results_n" + std::to_string(n) + "_p" +
                           std::to_string(p) + "_stagnation_window_" +
                           std::to_string(stagnation_window) + ".txt";

    std::ofstream out_file(folder + filename);

    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // 2. Print Headers
    // -----------------------------------
    std::stringstream ss;
    ss << "\n";
    ss << "======================================================================\n";
    ss << "=== T-Rex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n";
    ss << "======================================================================\n\n";
    print_dual(ss.str());

    // 3. Setup Table Dimensions
    // -----------------------------------
    const std::size_t solver_width = 15;
    const std::size_t metric_width = 8;
    const std::size_t snr_col_width = 5;
    const std::size_t col_width = 10;

    // 4. Print Table Header
    // -----------------------------------
    std::stringstream ss_header;

    ss_header << std::left << std::setw(solver_width) << "Solver"
              << std::left << std::setw(metric_width) << "Metric"
              << std::right << std::setw(snr_col_width) << "SNR";

    for (double snr : snr_values) {
        ss_header << std::fixed << std::setprecision(1) << std::setw(col_width) << snr;
    }
    ss_header << "\n";

    ss_header << std::string(solver_width + metric_width + snr_col_width +
        col_width * snr_values.size(), '-') << "\n";
    print_dual(ss_header.str());

    auto print_metric_row = [&](const std::string& solver_name,
        const std::string& metric_name, const Eigen::VectorXd& data, bool is_first_row) {
        std::stringstream ss_row;

        ss_row << std::left << std::setw(solver_width) << (is_first_row ? solver_name : "")
               << std::left << std::setw(metric_width) << metric_name
               << std::setw(snr_col_width) << "";

        ss_row << std::right;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i) {
            ss_row << std::fixed << std::setprecision(4)
                   << std::setw(col_width) << data(i);
        }
        ss_row << "\n";
        print_dual(ss_row.str());
    };

    // 5. Print Data Rows
    // -----------------------------------
    for (const auto& solver : solvers_to_test) {
        std::string name = solver.solver_name;

        print_metric_row(name, "FDR",
                        fdr_results_map.at(name),
                        true);

        print_metric_row(name,
                         "TPR",
                         tpr_results_map.at(name),
                         false);

        if (avg_L_results_map.count(name) > 0) {
            print_metric_row(name, "Avg L",
                avg_L_results_map.at(name), false);
        }

        if (avg_T_results_map.count(name) > 0) {
            print_metric_row(name, "Avg T",
                avg_T_results_map.at(name), false);
        }

        print_dual("\n");
    }

    // 6. Write tidy CSV (long/stacked format) for plotting
    // -----------------------------------
    std::string csv_filename = file_prefix + "trex_results_n" + std::to_string(n) + "_p" +
                               std::to_string(p) + "_stagnation_window_" +
                               std::to_string(stagnation_window) + ".csv";
    std::ofstream csv_file(folder + csv_filename);
    if (csv_file.is_open()) {
        csv_file << "solver,metric,snr,value\n";
        for (const auto& solver : solvers_to_test) {
            const std::string& name = solver.solver_name;
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                std::ostringstream snr_ss;
                snr_ss << std::fixed << std::setprecision(6) << snr_values[i];
                const std::string snr_str = snr_ss.str();
                csv_file << std::fixed << std::setprecision(6);
                csv_file << name << ",FDR," << snr_str << ","
                         << fdr_results_map.at(name)(ei) << "\n";
                csv_file << name << ",TPR," << snr_str << ","
                         << tpr_results_map.at(name)(ei) << "\n";
                if (avg_L_results_map.count(name) > 0)
                    csv_file << name << ",AvgL," << snr_str << ","
                             << avg_L_results_map.at(name)(ei) << "\n";
                if (avg_T_results_map.count(name) > 0)
                    csv_file << name << ",AvgT," << snr_str << ","
                             << avg_T_results_map.at(name)(ei) << "\n";
            }
        }
        std::cout << "[Info] CSV results saved to:             " << folder + csv_filename << "\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: " << folder + csv_filename << "\n";
    }

    // 7. Footer (text file)
    // -----------------------------------
    if (out_file.is_open()) {
        std::cout << "[Info] Results successfully saved to: " << folder + filename << "\n\n";
        out_file.close();
    } else {
        std::cout << "[Warning] Could not open output file: " << folder + filename << "\n\n";
    }
}


/**
 * @brief Print averaged MC results as an aligned table (console + .txt) and
 *        write a tidy long-format CSV. Uses string names for solvers.
 *
 * @param num_MC             Number of Monte Carlo repetitions used.
 * @param file_stem          Output file base name (no folder, no extension).
 * @param snr_values         SNR grid used during the simulation.
 * @param solver_names       Ordered list of solver names (keys into result maps).
 * @param fdr_results_map    Averaged FDR per solver, indexed by SNR.
 * @param tpr_results_map    Averaged TPR per solver, indexed by SNR.
 * @param avg_L_results_map  Averaged dummy multiplier L (empty map = omit).
 * @param avg_T_results_map  Averaged stopping time T  (empty map = omit).
 */
inline void save_and_print_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     snr_values,
    const std::vector<std::string>&                solver_names,
    const std::map<std::string, Eigen::VectorXd>&  fdr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_L_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_T_results_map)
{
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);

    // 1. Open text file (dual output)
    std::ofstream out_file(folder + file_stem + ".txt");
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // 2. Header
    {
        std::ostringstream ss;
        ss << "\n"
           << "======================================================================\n"
           << "=== T-Rex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n"
           << "======================================================================\n\n";
        print_dual(ss.str());
    }

    // 3. Table dimensions
    const std::size_t solver_width  = 15;
    const std::size_t metric_width  = 8;
    const std::size_t snr_col_width = 5;
    const std::size_t col_width     = 10;

    // 4. Column header + dashed separator
    {
        std::ostringstream hdr;
        hdr << std::left  << std::setw(solver_width)  << "Solver"
            << std::left  << std::setw(metric_width)  << "Metric"
            << std::right << std::setw(snr_col_width) << "SNR";
        for (double snr : snr_values)
            hdr << std::fixed << std::setprecision(1) << std::setw(col_width) << snr;
        hdr << "\n"
            << std::string(solver_width + metric_width + snr_col_width +
                           col_width * snr_values.size(), '-') << "\n";
        print_dual(hdr.str());
    }

    // 5. Data rows
    auto print_row = [&](const std::string& sname, const std::string& metric,
                         const Eigen::VectorXd& data, bool first_row) {
        std::ostringstream row;
        row << std::left  << std::setw(solver_width)  << (first_row ? sname : "")
            << std::left  << std::setw(metric_width)  << metric
            << std::setw(snr_col_width) << "" << std::right;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i)
            row << std::fixed << std::setprecision(4) << std::setw(col_width)
                << data(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& name : solver_names) {
        print_row(name, "FDR", fdr_results_map.at(name), true);
        print_row(name, "TPR", tpr_results_map.at(name), false);
        if (avg_L_results_map.count(name) > 0)
            print_row(name, "Avg L", avg_L_results_map.at(name), false);
        if (avg_T_results_map.count(name) > 0)
            print_row(name, "Avg T", avg_T_results_map.at(name), false);
        print_dual("\n");
    }

    // 6. Tidy long-format CSV (solver, metric, snr, value)
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "solver,metric,snr,value\n" << std::fixed << std::setprecision(6);
        for (const auto& name : solver_names) {
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                std::ostringstream snr_ss;
                snr_ss << std::fixed << std::setprecision(6) << snr_values[i];
                const std::string s = snr_ss.str();
                csv << name << ",FDR,"  << s << ","
                    << fdr_results_map.at(name)(ei) << "\n";
                csv << name << ",TPR,"  << s << ","
                    << tpr_results_map.at(name)(ei) << "\n";
                if (avg_L_results_map.count(name) > 0)
                    csv << name << ",AvgL," << s << ","
                        << avg_L_results_map.at(name)(ei) << "\n";
                if (avg_T_results_map.count(name) > 0)
                    csv << name << ",AvgT," << s << ","
                        << avg_T_results_map.at(name)(ei) << "\n";
            }
        }
        std::cout << "[Info] CSV results saved to:             "
                  << folder + file_stem + ".csv\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: " << folder + file_stem + ".csv\n";
    }

    // 7. Footer
    if (out_file.is_open()) {
        std::cout << "[Info] Results successfully saved to: " << folder + file_stem + ".txt\n\n";
        out_file.close();
    } else {
        std::cout << "[Warning] Could not open output file: " << folder + file_stem + ".txt\n\n";
    }
}


// ==============================================================================
// Parallel MC inner loop — in-memory DGP
// ==============================================================================

/**
 * @brief Run num_MC parallel MC trials using TRexSelector.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in the start / done lines.
 * @param make_data         Factory: seed → TrexDGPData (called per trial).
 * @param tFDR              Target FDR for the selector.
 * @param trex_ctrl         T-Rex control parameters (read-only; TRexSelector copies
 *                          by value so multiple threads are safe).
 * @param base_seed_offset  Base RNG seed offset; trial mc uses base_seed_offset + mc.
 *
 * @return TrexGridPointResult with avg/sd FDR, TPR and avg L, T.
 */
inline TrexGridPointResult run_mc_trials_trex(
    std::size_t                  num_MC,
    const std::string&           progress_label,
    const TrexDGPFactory&        make_data,
    double                       tFDR,
    const TRexControlParameter&  trex_ctrl,
    unsigned                     base_seed_offset)
{
    const int iMC = static_cast<int>(num_MC);

    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);
    std::vector<double> L_vec  (num_MC, 0.0);
    std::vector<double> T_vec  (num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexSelector selector(X_map, y_map, tFDR, trex_ctrl,
                              -1, false);
        selector.select();

        const auto sel  = selector.getSelectedIndices();
        fdp_vec[mc] = rates::compute_fdp(sel, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel, dat.true_support);
        L_vec[mc]   = static_cast<double>(selector.getDummyMultiplierL());
        T_vec[mc]   = static_cast<double>(selector.getTStop());
    }

    // Compute means
    const double dMC = static_cast<double>(num_MC);
    double avg_fdr = 0.0, avg_tpr = 0.0, avg_L = 0.0, avg_T = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_fdr += fdp_vec[mc];
        avg_tpr += tpp_vec[mc];
        avg_L   += L_vec[mc];
        avg_T   += T_vec[mc];
    }
    avg_fdr /= dMC;  avg_tpr /= dMC;  avg_L /= dMC;  avg_T /= dMC;

    // Bessel-corrected standard deviations
    double sd_fdr = 0.0, sd_tpr = 0.0;
    if (num_MC > 1) {
        double ssq_fdr = 0.0, ssq_tpr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            ssq_fdr += (fdp_vec[mc] - avg_fdr) * (fdp_vec[mc] - avg_fdr);
            ssq_tpr += (tpp_vec[mc] - avg_tpr) * (tpp_vec[mc] - avg_tpr);
        }
        sd_fdr = std::sqrt(ssq_fdr / (dMC - 1.0));
        sd_tpr = std::sqrt(ssq_tpr / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " \u2014 done. TPP=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDR=" << avg_fdr << "\n\n" << std::flush;

    return {avg_fdr, avg_tpr, avg_L, avg_T, sd_fdr, sd_tpr};
}


// ==============================================================================
// Parallel MC inner loop — fully memory-mapped X
// ==============================================================================

/**
 * @brief Run num_MC parallel MC trials using TRexSelector with memory-mapped X.
 *
 * Each OpenMP thread uses unique backing-file paths derived from mmap_filestem
 * and the thread ID (omp_get_thread_num()), preventing file-name collisions.
 * The factory lambda is responsible for the full lifecycle of SyntheticDataMapped
 * and its files: declare MmapFileGuard BEFORE SyntheticDataMapped inside the
 * lambda body so LIFO destruction closes mmap handles before removing files.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in the start / done lines.
 * @param make_data         Factory: (seed, X_path, y_path) → TrexDGPData.
 *                          Must copy X and y into owned matrices and clean up files.
 * @param tFDR              Target FDR for the selector.
 * @param trex_ctrl         T-Rex control parameters (read-only; TRexSelector copies
 *                          by value so multiple threads are safe).
 * @param base_seed_offset  Base RNG seed offset; trial mc uses base_seed_offset + mc.
 * @param mmap_filestem     Stem for per-thread backing files.
 *                          Thread t uses: filestem_X_tid{t}.dat / filestem_y_tid{t}.dat
 *
 * @return TrexGridPointResult with avg/sd FDR, TPR and avg L, T.
 */
inline TrexGridPointResult run_mc_trials_trex_mmap(
    std::size_t                  num_MC,
    const std::string&           progress_label,
    const TrexMmapDGPFactory&    make_data,
    double                       tFDR,
    const TRexControlParameter&  trex_ctrl,
    unsigned                     base_seed_offset,
    const std::string&           mmap_filestem)
{
    const int iMC = static_cast<int>(num_MC);

    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);
    std::vector<double> L_vec  (num_MC, 0.0);
    std::vector<double> T_vec  (num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials (mmap) ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const int    tid      = omp_get_thread_num();
        const std::string X_path = mmap_filestem + "_X_tid" + std::to_string(tid) + ".dat";
        const std::string y_path = mmap_filestem + "_y_tid" + std::to_string(tid) + ".dat";

        const unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed, X_path, y_path);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexSelector selector(X_map, y_map, tFDR, trex_ctrl, -1, false);
        selector.select();

        const auto sel  = selector.getSelectedIndices();
        fdp_vec[mc] = rates::compute_fdp(sel, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel, dat.true_support);
        L_vec[mc]   = static_cast<double>(selector.getDummyMultiplierL());
        T_vec[mc]   = static_cast<double>(selector.getTStop());
    }

    // Compute means
    const double dMC = static_cast<double>(num_MC);
    double avg_fdr = 0.0, avg_tpr = 0.0, avg_L = 0.0, avg_T = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_fdr += fdp_vec[mc];
        avg_tpr += tpp_vec[mc];
        avg_L   += L_vec[mc];
        avg_T   += T_vec[mc];
    }
    avg_fdr /= dMC;  avg_tpr /= dMC;  avg_L /= dMC;  avg_T /= dMC;

    // Bessel-corrected standard deviations
    double sd_fdr = 0.0, sd_tpr = 0.0;
    if (num_MC > 1) {
        double ssq_fdr = 0.0, ssq_tpr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            ssq_fdr += (fdp_vec[mc] - avg_fdr) * (fdp_vec[mc] - avg_fdr);
            ssq_tpr += (tpp_vec[mc] - avg_tpr) * (tpp_vec[mc] - avg_tpr);
        }
        sd_fdr = std::sqrt(ssq_fdr / (dMC - 1.0));
        sd_tpr = std::sqrt(ssq_tpr / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " \u2014 done. TPP=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDR=" << avg_fdr << "\n\n" << std::flush;

    return {avg_fdr, avg_tpr, avg_L, avg_T, sd_fdr, sd_tpr};
}

} // namespace trex_sim

// ==============================================================================
#endif /* End of TREX_DEMOS_TREX_SIM_UTILS_HPP */
// ==============================================================================
