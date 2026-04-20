// ==============================================================================
// demo_trx_scr_common.hpp
// ==============================================================================
/**
 * @file demo_trx_scr_common.hpp
 *
 * @brief Shared infrastructure for Screen-TRex demo files.
 *
 * @details Provides output-generating helpers, control-parameter factories,
 *          and type aliases shared by:
 *            demo_trx_scr_01_screen_trex.cpp
 *            demo_trx_scr_02_screen_trex_mmap.cpp
 *            demo_trx_scr_03_screen_trex_correlated.cpp
 *
 *  Types / constants:
 *    - ScreenMethodInfo          method descriptor (name + bootstrap flag)
 *    - kScreenMethods            default method list (Ordinary + Bootstrap)
 *
 *  Control-parameter factories:
 *    - make_trex_control()       default TRexControlParameter for Screen-TRex
 *    - make_trex_control_mmap()  same, with use_memory_mapping = true
 *    - make_screen_control()     default ScreenTRexControlParameter
 *
 *  Output functions:
 *    - print_screen_result()     single-run result block (FDR, indices, FDP, TPP)
 *    - print_mc_results_table()  MC summary table to console
 *
 * @note Demo-internal header — not part of the TRexSelector library.
 */
// ==============================================================================

#ifndef TREX_DEMOS_TRX_SCR_COMMON_HPP
#define TREX_DEMOS_TRX_SCR_COMMON_HPP

// std includes
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <trex_selector_methods/trex_screening/trex_screening.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace scr_demo {

namespace cdiag    = trex::utils::eval::cdiagnostics;
namespace dummygen = trex::utils::datageneration::dummygen;
namespace rates    = trex::utils::eval::rates;

using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::trex_screening::ScreenTRexControlParameter;
using trex::trex_selector_methods::trex_screening::ScreenTRexMethod;
using trex::trex_selector_methods::trex_screening::ScreenTRexSelector;
using trex::trex_selector_methods::trex_screening::ScreenTRexSelectionResult;


// ==============================================================================
// Types
// ==============================================================================

/** @brief Method descriptor for MC comparisons. */
struct ScreenMethodInfo {
    std::string     name;
    ScreenTRexMethod method    = ScreenTRexMethod::TREX;
    bool            bootstrap = false;
};

/** @brief Default method set for ordinary-vs-bootstrap comparisons. */
inline std::vector<ScreenMethodInfo> default_methods() {
    return {
        {"Screen-TRex Ordinary",  ScreenTRexMethod::TREX, false},
        {"Screen-TRex Bootstrap", ScreenTRexMethod::TREX, true }
    };
}


// ==============================================================================
// Control-parameter factories
// ==============================================================================

/** @brief Default TRexControlParameter for Screen-TRex (in-memory). */
inline TRexControlParameter make_trex_control(std::size_t K = 20) {
    TRexControlParameter ctrl;
    ctrl.K                    = K;
    ctrl.lloop_strategy       = LLoopStrategy::STANDARD;
    ctrl.solver_type          = SolverTypeForTRex::TLARS;
    ctrl.dummy_distribution   = dummygen::Distribution::Normal();
    ctrl.use_memory_mapping   = false;
    return ctrl;
}

/** @brief Default TRexControlParameter for Screen-TRex (memory-mapped). */
inline TRexControlParameter make_trex_control_mmap(std::size_t K = 20) {
    auto ctrl = make_trex_control(K);
    ctrl.use_memory_mapping = true;
    return ctrl;
}

/** @brief Default ScreenTRexControlParameter. */
inline ScreenTRexControlParameter make_screen_control(
    ScreenTRexMethod method    = ScreenTRexMethod::TREX,
    bool             bootstrap = false,
    std::size_t      n_blocks  = 5)
{
    ScreenTRexControlParameter ctrl;
    ctrl.trex_method      = method;
    ctrl.use_bootstrap_CI = bootstrap;
    ctrl.R_boot           = 1000;
    ctrl.ci_grid_step     = 0.001;
    ctrl.rho_thr_DA       = 0.02;
    ctrl.n_blocks         = n_blocks;
    return ctrl;
}


// ==============================================================================
// Single-run result output
// ==============================================================================

/**
 * @brief Print a single Screen-TRex selection result block.
 *
 * @param result           ScreenTRexSelectionResult from getScreenResult().
 * @param selected_indices Indices returned by getSelectedIndices().
 * @param true_support     Ground-truth active variable indices.
 */
inline void print_screen_result(
    const ScreenTRexSelectionResult&  result,
    const std::vector<std::size_t>&  selected_indices,
    const std::vector<std::size_t>&  true_support)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n  Estimated FDR:  " << result.estimated_FDR << "\n";

    if (result.estimated_correlation != 0.0) {
        std::cout << "  Estimated rho:  " << result.estimated_correlation << "\n";
    }
    if (result.used_bootstrap) {
        std::cout << "  Bootstrap:      yes\n";
        std::cout << "  Conf. level:    " << result.confidence_level << "\n";
    }

    std::cout << "  Selected (" << selected_indices.size() << "):  ";
    for (const auto idx : selected_indices) { std::cout << idx << " "; }
    std::cout << "\n";

    const double fdp = rates::compute_fdp(selected_indices, true_support);
    const double tpp = rates::compute_tpp(selected_indices, true_support);

    std::cout << "  FDP: " << fdp << "  |  TPP: " << tpp << "\n";

    std::cout << "  True support:   ";
    for (const auto idx : true_support) { std::cout << idx << " "; }
    std::cout << "\n\n";
}


// ==============================================================================
// MC results: dual console + file output (TXT + CSV)
// ==============================================================================

/**
 * @brief Save and print a sweep-variable MC results table.
 *
 * @details Writes a human-readable table (matching the T-Rex results format)
 *          to **console + TXT file** and a machine-readable tidy-long CSV.
 *          Files are placed in `simulations/demos/trex_screening/` under the supplied stem.
 *
 * @param num_MC      Number of Monte Carlo runs.
 * @param file_stem   Output file stem (no folder, no extension).
 * @param x_values    Sweep variable grid (SNR values, rho values, etc.).
 * @param methods     Method descriptors.
 * @param fdr_map     Method name -> FDR vector.
 * @param tpr_map     Method name -> TPR vector.
 * @param est_fdr_map Method name -> estimated FDR vector.
 * @param sweep_label Column label for the sweep variable (default: "SNR").
 */
inline void save_and_print_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     x_values,
    const std::vector<ScreenMethodInfo>&            methods,
    const std::map<std::string, Eigen::VectorXd>&  fdr_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_map,
    const std::map<std::string, Eigen::VectorXd>&  est_fdr_map,
    const std::string&                             sweep_label = "SNR")
{
    const std::string folder = "simulations/demos/trex_screening/";

    // ── Open TXT file for dual output ────────────────────────────────────
    std::ofstream out_file(folder + file_stem + ".txt");

    auto print_dual = [&](const std::string& text) {
        std::cout  << text;
        if (out_file.is_open()) out_file << text;
    };

    // ── Table dimensions ─────────────────────────────────────────────────
    const int name_w   = 31;   // left-aligned method name
    const int metric_w = 10;   // left-aligned metric label
    const int sweep_w  = 10;   // sweep variable label / empty in data rows
    const int col_w    = 10;   // right-aligned values
    const std::size_t sep_w =
        static_cast<std::size_t>(name_w + metric_w + sweep_w)
        + col_w * x_values.size();

    // ── Header ───────────────────────────────────────────────────────────
    {
        std::ostringstream ss;
        ss << "\n"
           << std::string(70, '=') << "\n"
           << "=== Screen-TRex Results (averaged over "
           << num_MC << " Monte Carlo runs) ===\n"
           << std::string(70, '=') << "\n\n";
        print_dual(ss.str());
    }

    // ── Column header + separator ────────────────────────────────────────
    {
        std::ostringstream hdr;
        hdr << std::left
            << std::setw(name_w)   << "Method"
            << std::setw(metric_w) << "Metric"
            << std::setw(sweep_w)  << sweep_label;
        for (const double x : x_values) {
            hdr << std::right << std::fixed << std::setprecision(2)
                << std::setw(col_w) << x;
        }
        hdr << "\n" << std::string(sep_w, '-') << "\n";
        print_dual(hdr.str());
    }

    // ── Data rows ────────────────────────────────────────────────────────
    auto print_row = [&](const std::string& nm, const std::string& label,
                         const Eigen::VectorXd& vec, bool first_row) {
        std::ostringstream row;
        row << std::left
            << std::setw(name_w)   << (first_row ? nm : "")
            << std::setw(metric_w) << label;
        row << std::string(static_cast<std::size_t>(sweep_w), ' ');
        for (Eigen::Index i = 0;
             i < static_cast<Eigen::Index>(x_values.size()); ++i) {
            row << std::right << std::fixed << std::setprecision(4)
                << std::setw(col_w) << vec(i);
        }
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& m : methods) {
        const std::string& nm = m.name;
        print_row(nm, "FDR",      fdr_map.at(nm),     true);
        print_row(nm, "TPR",      tpr_map.at(nm),     false);
        print_row(nm, "Est. FDR", est_fdr_map.at(nm), false);
        print_dual("\n");
    }

    // ── Close TXT ────────────────────────────────────────────────────────
    if (out_file.is_open()) {
        out_file.close();
        std::cout << "[Info] TXT results saved to: "
                  << folder + file_stem + ".txt\n";
    }

    // ── CSV file: tidy long format ───────────────────────────────────────
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "method,metric," << sweep_label << ",value\n"
            << std::fixed << std::setprecision(6);
        for (const auto& m : methods) {
            const std::string& nm = m.name;
            for (std::size_t i = 0; i < x_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                const double x = x_values[i];
                csv << nm << ",FDR,"      << x << "," << fdr_map.at(nm)(ei)     << "\n";
                csv << nm << ",TPR,"      << x << "," << tpr_map.at(nm)(ei)     << "\n";
                csv << nm << ",Est. FDR," << x << "," << est_fdr_map.at(nm)(ei) << "\n";
            }
        }
        csv.close();
        std::cout << "[Info] CSV results saved to: "
                  << folder + file_stem + ".csv\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: "
                  << folder + file_stem + ".csv\n";
    }
    std::cout << "\n";
}


// ==============================================================================
// MC progress helper
// ==============================================================================

/** @brief Print in-place MC progress line (carriage-return overwrite). */
inline void print_mc_progress(double val,
                              std::size_t mc,
                              std::size_t num_MC,
                              const std::string& sweep_label = "SNR") {
    std::cout << "  " << sweep_label << " "
              << std::fixed << std::setprecision(2) << val
              << " \u2014 Progress: ["
              << std::setw(3) << (mc + 1) << "/" << num_MC << "] "
              << std::fixed << std::setprecision(1)
              << (100.0 * static_cast<double>(mc + 1)
                  / static_cast<double>(num_MC))
              << "%    \r" << std::flush;
}

/** @brief Print MC completion line for a given sweep point. */
inline void print_mc_done(double val,
                          std::size_t num_MC,
                          const std::string& sweep_label = "SNR") {
    std::cout << "  " << sweep_label << " "
              << std::fixed << std::setprecision(2) << val
              << " — Completed " << num_MC << " runs.      \n";
}


} // namespace scr_demo

// ==============================================================================
#endif /* End of TREX_DEMOS_TRX_SCR_COMMON_HPP */
// ==============================================================================
