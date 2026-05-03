// ==============================================================================
// demo_trx_scr_bb_common.hpp
// ==============================================================================
/**
 * @file demo_trx_scr_bb_common.hpp
 *
 * @brief Shared infrastructure for Biobank Screen-TRex demo files (04–05).
 *
 * @details Extends demo_trx_scr_common.hpp with biobank-specific helpers:
 *   - makeBiobankControl()               BiobankScreenTRexControl factory
 *   - print_biobank_single_result()      single-phenotype result block
 *   - print_biobank_phenotype_table()    per-phenotype results table
 *   - print_biobank_summary()            multi-phenotype summary statistics
 *   - save_and_print_biobank_mc_results() MC table (with Usage %) + file output
 *
 * @note Demo-internal header — not part of the TRexSelector library.
 */
// ==============================================================================

#ifndef TREX_DEMOS_TRX_SCR_BB_COMMON_HPP
#define TREX_DEMOS_TRX_SCR_BB_COMMON_HPP

// TRex includes
#include <trex_selector_methods/trex_screening/trex_biobank_screening.hpp>

// Screen-TRex demo common header (brings in Eigen, std IO, scr_demo namespace)
#include "demo_trex_scr_common.hpp"

// ==============================================================================
namespace scr_demo {

/** @brief Namespace alias for Biobank Screen-TRex types. */
namespace tbs = trex::trex_selector_methods::trex_biobank_screening;


// ==============================================================================
// Control factory
// ==============================================================================

/**
 * @brief Build a default BiobankScreenTRexControl.
 *
 * @param use_mmap  Use memory-mapped D-file storage (default: false = in-memory).
 * @param K         Number of experiments per phenotype (default: 20).
 */
inline tbs::BiobankScreenTRexControl makeBiobankControl(
    bool use_mmap = false, std::size_t K = 20)
{
    tbs::BiobankScreenTRexControl ctrl;
    ctrl.lower_bound_FDR = 0.05;
    ctrl.upper_bound_FDR = 0.15;
    ctrl.target_FDR_trex = 0.10;

    ctrl.screen_ctrl.trex_method  = ScreenTRexMethod::TREX;
    ctrl.screen_ctrl.R_boot       = 1000;
    ctrl.screen_ctrl.ci_grid_step = 0.001;

    ctrl.trex_ctrl.K                    = K;
    ctrl.trex_ctrl.max_dummy_multiplier = 10;
    ctrl.trex_ctrl.use_max_T_stop       = true;
    ctrl.trex_ctrl.dummy_distribution   = dummygen::Distribution::Normal();
    ctrl.trex_ctrl.lloop_strategy       = LLoopStrategy::STANDARD;
    ctrl.trex_ctrl.solver_type          = SolverTypeForTRex::TLARS;
    ctrl.trex_ctrl.use_memory_mapping   = use_mmap;

    return ctrl;
}


// ==============================================================================
// Single-phenotype result output
// ==============================================================================

/**
 * @brief Print a single Biobank Screen-TRex phenotype result.
 *
 * @details Aligned with print_screen_result() format: method, FDR diagnostics,
 *          selected indices, FDP/TPP on one line, true support.
 *
 * @param result       BiobankScreenTRexResult from screenPhenotype().
 * @param true_support Ground-truth active variable indices.
 */
inline void print_biobank_single_result(
    const tbs::BiobankScreenTRexResult&  result,
    const std::vector<std::size_t>&      true_support)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n  Method used:       " << result.method_used                   << "\n";
    std::cout << "  Estimated FDR:     " << result.estimated_FDR                   << "\n";
    std::cout << "  α̂  (ordinary):     " << result.estimated_FDR_screen_ordinary   << "\n";
    std::cout << "  α̂_C (bootstrap):   " << result.estimated_FDR_screen_bootstrap  << "\n";

    std::cout << "  Selected (" << result.selected_indices.size() << "):  ";
    for (const auto idx : result.selected_indices) { std::cout << idx << " "; }
    std::cout << "\n";

    const double fdp = rates::compute_fdp(result.selected_indices, true_support);
    const double tpp = rates::compute_tpp(result.selected_indices, true_support);
    std::cout << "  FDP: " << fdp << "  |  TPP: " << tpp << "\n";

    std::cout << "  True support:   ";
    for (const auto idx : true_support) { std::cout << idx << " "; }
    std::cout << "\n\n";
}


// ==============================================================================
// Multi-phenotype table + summary output
// ==============================================================================

/**
 * @brief Print per-phenotype results table.
 *
 * @param results       Per-phenotype screening results.
 * @param true_supports Ground-truth supports (one entry per phenotype).
 */
inline void print_biobank_phenotype_table(
    const std::vector<tbs::BiobankScreenTRexResult>&      results,
    const std::vector<std::vector<std::size_t>>&           true_supports)
{
    constexpr int pw = 8, nw = 28, cw = 12;

    std::cout << "\n" << std::string(80, '=') << "\n"
              << "=== Results by Phenotype ===\n"
              << std::string(80, '=') << "\n\n";

    std::cout << std::right << std::setw(pw) << "Pheno"
              << std::left  << std::setw(nw) << "  Method"
              << std::right << std::setw(cw) << "Est.FDR"
              << std::setw(cw) << "Selected"
              << std::setw(cw) << "FDP"
              << std::setw(cw) << "TPP" << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r   = results[i];
        const double fdp = rates::compute_fdp(r.selected_indices, true_supports[i]);
        const double tpp = rates::compute_tpp(r.selected_indices, true_supports[i]);
        std::cout << std::right << std::setw(pw) << i
                  << std::left  << std::setw(nw) << ("  " + r.method_used)
                  << std::right
                  << std::setw(cw) << r.estimated_FDR
                  << std::setw(cw) << r.selected_indices.size()
                  << std::setw(cw) << fdp
                  << std::setw(cw) << tpp << "\n";
    }
    std::cout << "\n";
}


/**
 * @brief Print multi-phenotype screening summary statistics.
 *
 * @param stats BiobankScreenTRexStatistics from getBiobankScreenTRexResult().
 */
inline void print_biobank_summary(
    const tbs::BiobankScreenTRex::BiobankScreenTRexStatistics& stats)
{
    std::cout << std::string(70, '=') << "\n"
              << "=== Summary ===\n"
              << std::string(70, '=') << "\n";
    std::cout << "  Total phenotypes:        " << stats.num_phenotypes           << "\n";
    std::cout << "  Screen-TRex (ordinary):  " << stats.num_used_screen_ordinary  << "\n";
    std::cout << "  Screen-TRex (bootstrap): " << stats.num_used_screen_bootstrap << "\n";
    std::cout << "  T-Rex (fallback):        " << stats.num_used_trex             << "\n";
    std::cout << "  Avg estimated FDR:       "
              << std::fixed << std::setprecision(4) << stats.avg_estimated_FDR    << "\n";
    std::cout << "  Avg selected count:      "
              << std::fixed << std::setprecision(2) << stats.avg_selected_count   << "\n\n";
}


// ==============================================================================
// MC results: save + print (TXT + CSV)
// ==============================================================================

/**
 * @brief Save and print a Biobank Screen-TRex SNR-sweep MC results table.
 *
 * @details Writes a human-readable table (console + TXT) and a tidy CSV.
 *          Files are placed in `simulations/demos/trex_screening/`.
 *          Includes a "Usage (%)" row per method — the fraction of MC runs
 *          in which Algorithm 1 routed to that method.
 *
 * @param num_MC       Number of Monte Carlo runs.
 * @param file_stem    Output file stem (no folder, no extension).
 * @param snr_values   SNR grid.
 * @param method_names Ordered method names (keys into all result maps).
 * @param fdp_map      Method → conditional-mean FDP vector.
 * @param tpp_map      Method → conditional-mean TPP vector.
 * @param est_fdr_map  Method → mean estimated FDR vector.
 * @param usage_map    Method → usage fraction [0, 1] vector.
 */
inline void save_and_print_biobank_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     snr_values,
    const std::vector<std::string>&                method_names,
    const std::map<std::string, Eigen::VectorXd>&  fdp_map,
    const std::map<std::string, Eigen::VectorXd>&  tpp_map,
    const std::map<std::string, Eigen::VectorXd>&  est_fdr_map,
    const std::map<std::string, Eigen::VectorXd>&  usage_map)
{
    const std::string folder = "simulations/demos/trex_screening/";

    // ── Open TXT file for dual output ─────────────────────────────────────
    std::ofstream out_file(folder + file_stem + ".txt");
    auto print_dual = [&](const std::string& text) {
        std::cout  << text;
        if (out_file.is_open()) out_file << text;
    };

    // ── Table dimensions ──────────────────────────────────────────────────
    const int name_w   = 31;
    const int metric_w = 12;
    const int snr_w    = 10;
    const int col_w    = 10;
    const std::size_t sep_w =
        static_cast<std::size_t>(name_w + metric_w + snr_w)
        + col_w * snr_values.size();

    // ── Header ────────────────────────────────────────────────────────────
    {
        std::ostringstream ss;
        ss << "\n"
           << std::string(70, '=') << "\n"
           << "=== Biobank Screen-TRex Results (averaged over "
           << num_MC << " Monte Carlo runs) ===\n"
           << std::string(70, '=') << "\n\n";
        print_dual(ss.str());
    }

    // ── Column header + separator ─────────────────────────────────────────
    {
        std::ostringstream hdr;
        hdr << std::left
            << std::setw(name_w)   << "Method"
            << std::setw(metric_w) << "Metric"
            << std::setw(snr_w)    << "SNR";
        for (const double snr : snr_values)
            hdr << std::right << std::fixed << std::setprecision(2)
                << std::setw(col_w) << snr;
        hdr << "\n" << std::string(sep_w, '-') << "\n";
        print_dual(hdr.str());
    }

    // ── Data rows ─────────────────────────────────────────────────────────
    auto print_row = [&](const std::string& nm, const std::string& label,
                         const Eigen::VectorXd& vec, bool first_row, int dp) {
        std::ostringstream row;
        row << std::left
            << std::setw(name_w)   << (first_row ? nm : "")
            << std::setw(metric_w) << label
            << std::string(static_cast<std::size_t>(snr_w), ' ');
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i)
            row << std::right << std::fixed << std::setprecision(dp)
                << std::setw(col_w) << vec(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& nm : method_names) {
        const Eigen::VectorXd usage_pct = usage_map.at(nm) * 100.0;
        print_row(nm, "Usage (%)", usage_pct,            true,  1);
        print_row(nm, "FDR",       fdp_map.at(nm),       false, 4);
        print_row(nm, "TPR",       tpp_map.at(nm),       false, 4);
        print_row(nm, "Est. FDR",  est_fdr_map.at(nm),   false, 4);
        print_dual("\n");
    }

    // ── Close TXT ─────────────────────────────────────────────────────────
    if (out_file.is_open()) {
        out_file.close();
        std::cout << "[Info] TXT results saved to: " << folder + file_stem + ".txt\n";
    }

    // ── CSV: tidy long format ─────────────────────────────────────────────
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "method,metric,snr,value\n" << std::fixed << std::setprecision(6);
        for (const auto& nm : method_names) {
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto   ei  = static_cast<Eigen::Index>(i);
                const double snr = snr_values[i];
                csv << nm << ",Usage,"    << snr << "," << usage_map.at(nm)(ei)   << "\n";
                csv << nm << ",FDR,"      << snr << "," << fdp_map.at(nm)(ei)     << "\n";
                csv << nm << ",TPR,"      << snr << "," << tpp_map.at(nm)(ei)     << "\n";
                csv << nm << ",Est. FDR," << snr << "," << est_fdr_map.at(nm)(ei) << "\n";
            }
        }
        csv.close();
        std::cout << "[Info] CSV results saved to: " << folder + file_stem + ".csv\n";
    }
    std::cout << "\n";
}


} // namespace scr_demo
// ==============================================================================
#endif /* TREX_DEMOS_TRX_SCR_BB_COMMON_HPP */
// ==============================================================================
