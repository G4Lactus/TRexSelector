// ==============================================================================
// trex_gvs_simulation_utils.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_GVS_SIMULATION_UTILS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_GVS_SIMULATION_UTILS_HPP
// ==============================================================================
/**
 * @file trex_gvs_simulation_utils.hpp
 *
 * @brief Monte Carlo infrastructure and sweep utilities for T-Rex+GVS demos.
 *
 * Provides:
 *   - GVSSimConfig          — simulation parameter struct
 *   - GVSSingleRunResult    — single-run diagnostics (FDP, TPP, T_stop, …)
 *   - GVSGridPointResult    — MC-aggregated (mean/sd FDP, TPP) per grid point
 *   - make_gvs_trex_control() — default TRexControlParameter factory
 *   - run_gvs_single()      — single selector run, returns GVSSingleRunResult
 *   - print_single_run_result() — R-style formatted output block
 *   - GVSDGPFactory         — type alias for DGP closure
 *   - run_gvs_mc_trials()   — OpenMP-parallel MC inner loop
 *   - run_gvs_snr_sweep()   — outer SNR-grid sweep
 *   - print_mc_snr_table()  — aligned results table for SNR sweeps
 *   - run_gvs_rho_sweep()   — outer rho-grid sweep
 *   - print_mc_rho_table()  — aligned results table for rho sweeps
 *   - GVS2DDGPFactory       — type alias for 2-D DGP closure (snr, rho, seed)
 *   - run_gvs_2d_sweep()    — nested SNR × rho grid sweep
 *   - print_mc_matrix()     — named matrix table for 2-D sweep results
 *
 * Transitively includes trex_gvs_dgps.hpp (GVSDGPResult + all DGP generators).
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"

// std includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP
#include <utils/openmp/utils_openmp.hpp>

// T-Rex+GVS includes
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

// Eval metrics
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
namespace gvs_demo {
// ==============================================================================


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace rates       = trex::utils::eval::rates;
namespace cdiag       = trex::utils::eval::cdiagnostics;
namespace hac         = trex::ml_methods::clustering::hierarchical::agglomerative;

using trex::trex_selector_methods::trex_gvs::TRexGVSSelector;
using trex::trex_selector_methods::trex_gvs::TRexGVSControlParameter;
using trex::trex_selector_methods::trex_gvs::GVSType;
using trex::trex_selector_methods::trex_gvs::LambdaSelectionMethod;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::utils::solver_dispatch::SolverHyperparameters;


// ==============================================================================
// Simulation configuration
// ==============================================================================

/** @brief Parameter bundle for MC simulations (SNR sweep, rho sweep, 2-D grid). */
struct GVSSimConfig {
    int    n         = 200;    ///< Observations.
    int    p         = 500;    ///< Predictors.
    double sd_x      = 0.1;    ///< sqrt(0.01) — within-group predictor noise.
    double tFDR      = 0.1;    ///< Target FDR level.
    std::size_t K    = 20;     ///< Random experiments per selector run.
    std::size_t num_MC = 200;  ///< Monte Carlo trials per grid point.
    int base_seed    = 2026;   ///< Base RNG seed (trial i uses base_seed + i).
    double corr_max  = 0.98;   ///< corr_max for auto-clustering HAC.
    double snr       = 1.0;    ///< Fixed SNR for rho-sweep simulations.
};


// ==============================================================================
// Single-run result
// ==============================================================================

/** @brief Diagnostics returned by run_gvs_single(). */
struct GVSSingleRunResult {
    /** @brief Selected variable indices (0-based). */
    std::vector<std::size_t> selected;

    double      fdp          = 0.0;  ///< Realised false discovery proportion.
    double      tpp          = 0.0;  ///< Realised true positive proportion.
    double      lambda2_used = 0.0;  ///< lambda_2 resolved by the selector.
    std::size_t T_stop       = 0;    ///< Calibrated stopping time.
    std::size_t num_dummies  = 0;    ///< Total dummy variables used.
    std::size_t M_found      = 0;    ///< Number of clusters detected.
    double      elapsed_sec  = 0.0;  ///< Wall-clock time in seconds.
};


// ==============================================================================
// MC aggregate result
// ==============================================================================

/** @brief Aggregated (mean/sd FDP, TPP) across MC trials for one grid point. */
struct GVSGridPointResult {
    double mean_fdp = 0.0;
    double mean_tpp = 0.0;
    double sd_fdp   = 0.0;
    double sd_tpp   = 0.0;
};


// ==============================================================================
// Default control parameter factory
// ==============================================================================

/**
 * @brief Build a TRexControlParameter with sensible GVS defaults.
 *
 * @param K Number of random experiments (default: 20).
 */
inline TRexControlParameter make_gvs_trex_control(std::size_t K = 20) {
    TRexControlParameter ctrl;
    ctrl.K                        = K;
    ctrl.max_dummy_multiplier     = 10;
    ctrl.use_max_T_stop           = true;
    ctrl.lloop_strategy           = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop    = true;
    ctrl.tloop_max_stagnant_steps = 5;
    ctrl.use_memory_mapping       = false;
    return ctrl;
}


// ==============================================================================
// Single-run helper
// ==============================================================================

/**
 * @brief Run TRexGVSSelector once and collect diagnostics.
 *
 * @param X_map      Non-owning Eigen::Map over the design matrix.
 * @param y_map      Non-owning Eigen::Map over the response.
 * @param true_sup   True support for FDP/TPP computation.
 * @param tFDR       Target FDR level.
 * @param gvs_ctrl   GVS-specific control parameters.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param seed       RNG seed passed to the selector.
 *
 * @return GVSSingleRunResult with realised FDP, TPP, and selector diagnostics.
 */
inline GVSSingleRunResult run_gvs_single(
    Eigen::Map<Eigen::MatrixXd>&        X_map,
    Eigen::Map<Eigen::VectorXd>&        y_map,
    const std::vector<std::size_t>&     true_sup,
    double                              tFDR,
    const TRexGVSControlParameter&      gvs_ctrl,
    const TRexControlParameter&         trex_ctrl,
    int                                 seed)
{
    const auto t0 = std::chrono::steady_clock::now();

    // EN requires TENET solver; IEN requires TLASSO (augmented system).
    TRexControlParameter trex_ctrl_run = trex_ctrl;
    if (gvs_ctrl.gvs_type == GVSType::EN)
        trex_ctrl_run.solver_type = SolverTypeForTRex::TENET;
    else
        trex_ctrl_run.solver_type = SolverTypeForTRex::TLASSO;

    TRexGVSSelector selector(X_map, y_map, tFDR, gvs_ctrl, trex_ctrl_run,
                              seed, /*verbose=*/false);
    selector.select();

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    const auto& gvs_res = selector.getGVSResult();
    const auto& sel_idx = selector.getSelectedIndices();

    GVSSingleRunResult out;
    out.selected     = sel_idx;
    out.fdp          = rates::compute_fdp(sel_idx, true_sup);
    out.tpp          = rates::compute_tpp(sel_idx, true_sup);
    out.lambda2_used = gvs_res.lambda2_used;
    out.T_stop       = gvs_res.T_stop;
    out.num_dummies  = gvs_res.num_dummies;
    out.M_found      = gvs_res.max_clusters;
    out.elapsed_sec  = elapsed;

    return out;
}


// ==============================================================================
// Print helpers
// ==============================================================================

/**
 * @brief Print a formatted single-run result block (mirrors R's .print_result()).
 *
 * @param tag    Short scenario / method label (e.g. "Hastie GVS [EN]").
 * @param dat    DGP metadata.
 * @param res    Single-run result to display.
 * @param tFDR   Target FDR (printed for reference).
 */
inline void print_single_run_result(const std::string&       tag,
                                    const GVSDGPResult&       dat,
                                    const GVSSingleRunResult& res,
                                    double                    tFDR)
{
    const int n_sel = static_cast<int>(res.selected.size());
    const int n_tp  = static_cast<int>(
        std::count_if(res.selected.begin(), res.selected.end(),
                      [&](std::size_t idx) {
                          return std::binary_search(dat.true_support.begin(),
                                                    dat.true_support.end(),
                                                    idx);
                      }));
    const int n_fp = n_sel - n_tp;

    std::cout << std::string(70, '=') << "\n";
    std::cout << "  " << tag << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Data:  n = " << dat.n << ", p = " << dat.p
              << ", tFDR = " << tFDR << ", s = " << dat.s << "\n";
    std::cout << "         SNR = " << std::setprecision(2) << dat.snr
              << ",  sigma_y = " << std::setprecision(4) << dat.sigma_y
              << ",  sd_x = " << dat.sd_x << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "  Calibration:  T_stop = " << res.T_stop
              << ",  dummies = " << res.num_dummies
              << ",  M = " << res.M_found
              << ",  lambda2 = " << std::setprecision(4) << res.lambda2_used << "\n";
    std::cout << "  Selection:    " << n_sel << " selected"
              << "  |  TP = " << n_tp << "  FP = " << n_fp << "\n";
    std::cout << "  Rates:        TPP = " << std::setprecision(3) << res.tpp
              << "  |  FDP = " << res.fdp
              << "  (target tFDR <= " << tFDR << ")\n";
    std::cout << "  Wall-clock:   " << std::setprecision(2) << res.elapsed_sec << " s\n";
    std::cout << std::string(70, '=') << "\n\n";
}


// ==============================================================================
// MC inner loop
// ==============================================================================

/**
 * @brief Type alias for a DGP factory function.
 *
 * Signature: (snr_or_rho, seed) → GVSDGPResult.
 * Captures scenario-specific parameters (n, p, sd_x, …) via closure.
 */
using GVSDGPFactory = std::function<GVSDGPResult(double snr, unsigned seed)>;


/**
 * @brief Run num_MC parallel trials at a single parameter point.
 *
 * Each trial:
 *  1. Generates a fresh dataset via dgp_fn(param, trial_seed).
 *  2. Runs TRexGVSSelector (verbose=false).
 *  3. Collects FDP and TPP.
 *
 * @param dgp_fn         DGP factory — must be thread-safe (no shared mutable state).
 * @param param          Parameter value passed to dgp_fn (SNR or rho).
 * @param num_MC         Number of trials.
 * @param progress_label Label printed before/after the run.
 * @param tFDR           Target FDR level.
 * @param gvs_ctrl       GVS control parameters.
 * @param trex_ctrl      Base T-Rex control parameters.
 * @param base_seed      Seed for trial 0; trial i uses base_seed + i.
 *
 * @return GVSGridPointResult with mean and sd of FDP and TPP.
 */
inline GVSGridPointResult run_gvs_mc_trials(
    const GVSDGPFactory&           dgp_fn,
    double                         param,
    std::size_t                    num_MC,
    const std::string&             progress_label,
    double                         tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    unsigned                       base_seed)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials (param="
              << std::fixed << std::setprecision(2) << param << ") ...\n"
              << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed + static_cast<unsigned>(mc);
        auto dat = dgp_fn(param, trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        // EN requires TENET solver; IEN requires TLASSO (augmented system).
        TRexControlParameter trex_ctrl_run = trex_ctrl;
        if (gvs_ctrl.gvs_type == GVSType::EN)
            trex_ctrl_run.solver_type = SolverTypeForTRex::TENET;
        else
            trex_ctrl_run.solver_type = SolverTypeForTRex::TLASSO;

        TRexGVSSelector selector(X_map, y_map, tFDR, gvs_ctrl, trex_ctrl_run,
                                  static_cast<int>(trial_seed),
                                  /*verbose=*/false);
        selector.select();

        const auto& sel_idx = selector.getSelectedIndices();
        fdp_vec[static_cast<std::size_t>(mc)] =
            rates::compute_fdp(sel_idx, dat.true_support);
        tpp_vec[static_cast<std::size_t>(mc)] =
            rates::compute_tpp(sel_idx, dat.true_support);
    }

    const double dMC = static_cast<double>(num_MC);
    double mean_fdp = 0.0, mean_tpp = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        mean_fdp += fdp_vec[static_cast<std::size_t>(mc)];
        mean_tpp += tpp_vec[static_cast<std::size_t>(mc)];
    }
    mean_fdp /= dMC;
    mean_tpp /= dMC;

    double sd_fdp = 0.0, sd_tpp = 0.0;
    if (num_MC > 1) {
        double sq_fdp = 0.0, sq_tpp = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            const std::size_t idx = static_cast<std::size_t>(mc);
            sq_fdp += (fdp_vec[idx] - mean_fdp) * (fdp_vec[idx] - mean_fdp);
            sq_tpp += (tpp_vec[idx] - mean_tpp) * (tpp_vec[idx] - mean_tpp);
        }
        sd_fdp = std::sqrt(sq_fdp / (dMC - 1.0));
        sd_tpp = std::sqrt(sq_tpp / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " — done.  TPP=" << std::fixed << std::setprecision(3) << mean_tpp
              << "  FDP=" << mean_fdp << "\n\n" << std::flush;

    return {mean_fdp, mean_tpp, sd_fdp, sd_tpp};
}


// ==============================================================================
// SNR sweep
// ==============================================================================

/**
 * @brief Run a full SNR-grid MC sweep for one (EN or IEN) method variant.
 *
 * @param dgp_fn     DGP factory — must be thread-safe.
 * @param snr_grid   Vector of SNR values to sweep over.
 * @param cfg        Simulation config (num_MC, base_seed, tFDR, …).
 * @param gvs_ctrl   GVS control parameters (gvs_type is set by caller).
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param label      Short label for progress messages (e.g. "EN", "IEN").
 *
 * @return One GVSGridPointResult per entry in snr_grid.
 */
inline std::vector<GVSGridPointResult> run_gvs_snr_sweep(
    const GVSDGPFactory&           dgp_fn,
    const std::vector<double>&     snr_grid,
    const GVSSimConfig&            cfg,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    const std::string&             label)
{
    std::vector<GVSGridPointResult> results;
    results.reserve(snr_grid.size());

    for (double snr : snr_grid) {
        results.push_back(
            run_gvs_mc_trials(
                dgp_fn, snr, cfg.num_MC, label,
                cfg.tFDR, gvs_ctrl, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed)));
    }

    return results;
}


// ==============================================================================
// Print MC 1-D method table (shared helper for SNR and rho sweeps)
// ==============================================================================

/**
 * @brief Print a table for one method variant with all four metrics as columns.
 *
 * Columns: mean_FDP  sd_FDP  mean_TPP  sd_TPP
 *
 * @param method_label  Method name displayed as table heading (e.g. "EN").
 * @param param_labels  Labels for each row (e.g. "snr=0.10", "rho=0.30").
 * @param results       One GVSGridPointResult per row.
 */
inline void print_mc_1d_method_table(
    const std::string&                     method_label,
    const std::vector<std::string>&        param_labels,
    const std::vector<GVSGridPointResult>& results)
{
    constexpr int rw = 12;  // row-label width
    constexpr int cw = 10;  // value column width
    const int total_w = rw + 4 * cw;

    std::cout << "\n  " << method_label << "\n"
              << "  " << std::string(total_w, '-') << "\n"
              << "  " << std::left << std::setw(rw) << ""
              << std::right
              << std::setw(cw) << "mean_FDP"
              << std::setw(cw) << "sd_FDP"
              << std::setw(cw) << "mean_TPP"
              << std::setw(cw) << "sd_TPP"
              << "\n"
              << "  " << std::string(total_w, '-') << "\n";

    for (std::size_t r = 0; r < results.size() && r < param_labels.size(); ++r) {
        std::cout << "  " << std::left  << std::setw(rw) << param_labels[r]
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(cw) << results[r].mean_fdp
                  << std::setw(cw) << results[r].sd_fdp
                  << std::setw(cw) << results[r].mean_tpp
                  << std::setw(cw) << results[r].sd_tpp
                  << "\n";
    }
    std::cout << "\n";
}


// ==============================================================================
// Print MC SNR table
// ==============================================================================

/**
 * @brief Print two per-method tables for an SNR sweep (EN and IEN).
 *
 * Each table shows mean_TPP, sd_TPP, mean_FDP, sd_FDP as columns.
 *
 * @param scenario_tag  Short label printed in the header.
 * @param snr_grid      The swept SNR values.
 * @param en_results    EN method results per SNR point.
 * @param ien_results   IEN method results per SNR point.
 * @param cfg           Simulation config (num_MC, tFDR, …).
 */
inline void print_mc_snr_table(
    const std::string&                     scenario_tag,
    const std::vector<double>&             snr_grid,
    const std::vector<GVSGridPointResult>& en_results,
    const std::vector<GVSGridPointResult>& ien_results,
    const GVSSimConfig&                    cfg)
{
    std::cout << "\n" << std::string(78, '=') << "\n"
              << "  T-Rex+GVS MC: " << scenario_tag
              << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
              << ", n=" << cfg.n << ", p=" << cfg.p << ")\n"
              << std::string(78, '=') << "\n";

    std::vector<std::string> labels;
    labels.reserve(snr_grid.size());
    for (double s : snr_grid)
        labels.push_back("snr=" + std::to_string(s).substr(0, 4));

    print_mc_1d_method_table("EN",  labels, en_results);
    print_mc_1d_method_table("IEN", labels, ien_results);

    std::cout << std::string(78, '=') << "\n\n";
}


// ==============================================================================
// rho sweep
// ==============================================================================

/**
 * @brief Type alias documenting that the DGP factory's first argument is rho.
 *
 * Functionally identical to GVSDGPFactory — the alias exists to document
 * caller intent.
 */
using GVSRhoDGPFactory = GVSDGPFactory;


/**
 * @brief Run a full rho-grid MC sweep for one (EN or IEN) method variant.
 *
 * @param dgp_fn     DGP factory whose first argument is rho (not SNR).
 * @param rho_grid   Vector of rho values to sweep over.
 * @param cfg        Simulation config (num_MC, base_seed, tFDR, snr, …).
 * @param gvs_ctrl   GVS control parameters.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param label      Short label for progress messages (e.g. "EN", "IEN").
 *
 * @return One GVSGridPointResult per entry in rho_grid.
 */
inline std::vector<GVSGridPointResult> run_gvs_rho_sweep(
    const GVSRhoDGPFactory&        dgp_fn,
    const std::vector<double>&     rho_grid,
    const GVSSimConfig&            cfg,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    const std::string&             label)
{
    std::vector<GVSGridPointResult> results;
    results.reserve(rho_grid.size());

    for (double rho : rho_grid) {
        results.push_back(
            run_gvs_mc_trials(
                dgp_fn, rho, cfg.num_MC, label,
                cfg.tFDR, gvs_ctrl, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed)));
    }

    return results;
}


// ==============================================================================
// Print MC rho table
// ==============================================================================

/**
 * @brief Print two per-method tables for a rho sweep (EN and IEN).
 *
 * Each table shows mean_TPP, sd_TPP, mean_FDP, sd_FDP as columns.
 *
 * @param scenario_tag  Short label printed in the header.
 * @param rho_grid      The swept rho values.
 * @param en_results    EN method results per rho point.
 * @param ien_results   IEN method results per rho point.
 * @param cfg           Simulation config (num_MC, tFDR, snr, …).
 */
inline void print_mc_rho_table(
    const std::string&                     scenario_tag,
    const std::vector<double>&             rho_grid,
    const std::vector<GVSGridPointResult>& en_results,
    const std::vector<GVSGridPointResult>& ien_results,
    const GVSSimConfig&                    cfg)
{
    std::cout << "\n" << std::string(78, '=') << "\n"
              << "  T-Rex+GVS MC: " << scenario_tag
              << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
              << ", SNR=" << cfg.snr
              << ", n=" << cfg.n << ", p=" << cfg.p << ")\n"
              << std::string(78, '=') << "\n";

    std::vector<std::string> labels;
    labels.reserve(rho_grid.size());
    for (double r : rho_grid)
        labels.push_back("rho=" + std::to_string(r).substr(0, 4));

    print_mc_1d_method_table("EN",  labels, en_results);
    print_mc_1d_method_table("IEN", labels, ien_results);

    std::cout << std::string(78, '=') << "\n\n";
}


// ==============================================================================
// 2-D (SNR × rho) sweep
// ==============================================================================

/**
 * @brief Type alias for a 2-D DGP factory: (snr, rho, seed) → GVSDGPResult.
 */
using GVS2DDGPFactory =
    std::function<GVSDGPResult(double snr, double rho, unsigned seed)>;


/**
 * @brief Run a nested SNR × rho grid MC sweep for one method variant.
 *
 * @param dgp_fn      2-D DGP factory — must be thread-safe.
 * @param snr_grid    Outer SNR axis.
 * @param rho_grid    Inner rho axis.
 * @param cfg         Simulation config (num_MC, base_seed, tFDR, …).
 * @param gvs_ctrl    GVS control parameters.
 * @param trex_ctrl   Base T-Rex control parameters.
 * @param label       Short label for progress messages.
 *
 * @return results[i_snr][i_rho] — GVSGridPointResult for each cell.
 */
inline std::vector<std::vector<GVSGridPointResult>> run_gvs_2d_sweep(
    const GVS2DDGPFactory&         dgp_fn,
    const std::vector<double>&     snr_grid,
    const std::vector<double>&     rho_grid,
    const GVSSimConfig&            cfg,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    const std::string&             label)
{
    const std::size_t n_snr = snr_grid.size();
    const std::size_t n_rho = rho_grid.size();
    const int         iMC   = static_cast<int>(cfg.num_MC);
    const unsigned base_seed = static_cast<unsigned>(cfg.base_seed);

    std::vector<std::vector<GVSGridPointResult>> results(
        n_snr, std::vector<GVSGridPointResult>(n_rho));

    for (std::size_t i_snr = 0; i_snr < n_snr; ++i_snr) {
        for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
            const double snr = snr_grid[i_snr];
            const double rho = rho_grid[i_rho];

            std::cout << "  [" << label << "] ["
                      << (i_snr * n_rho + i_rho + 1) << "/"
                      << (n_snr * n_rho) << "]"
                      << "  SNR=" << std::fixed << std::setprecision(2) << snr
                      << "  rho=" << rho
                      << "  running " << cfg.num_MC << " MC trials...\n"
                      << std::flush;

            std::vector<double> fdp_vec(cfg.num_MC, 0.0);
            std::vector<double> tpp_vec(cfg.num_MC, 0.0);

            #pragma omp parallel for schedule(dynamic)
            for (int mc = 0; mc < iMC; ++mc) {
                const unsigned trial_seed = base_seed + static_cast<unsigned>(mc);
                auto dat = dgp_fn(snr, rho, trial_seed);

                Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                                   dat.X.rows(), dat.X.cols());
                Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

                TRexControlParameter trex_ctrl_run = trex_ctrl;
                if (gvs_ctrl.gvs_type == GVSType::EN)
                    trex_ctrl_run.solver_type = SolverTypeForTRex::TENET;
                else
                    trex_ctrl_run.solver_type = SolverTypeForTRex::TLASSO;

                TRexGVSSelector selector(X_map, y_map, cfg.tFDR, gvs_ctrl,
                                         trex_ctrl_run,
                                         static_cast<int>(trial_seed),
                                         /*verbose=*/false);
                selector.select();

                const auto& sel_idx = selector.getSelectedIndices();
                fdp_vec[static_cast<std::size_t>(mc)] =
                    rates::compute_fdp(sel_idx, dat.true_support);
                tpp_vec[static_cast<std::size_t>(mc)] =
                    rates::compute_tpp(sel_idx, dat.true_support);
            }

            const double dMC = static_cast<double>(cfg.num_MC);
            double mean_fdp = 0.0, mean_tpp = 0.0;
            for (int mc = 0; mc < iMC; ++mc) {
                mean_fdp += fdp_vec[static_cast<std::size_t>(mc)];
                mean_tpp += tpp_vec[static_cast<std::size_t>(mc)];
            }
            mean_fdp /= dMC;
            mean_tpp /= dMC;

            double sd_fdp = 0.0, sd_tpp = 0.0;
            if (cfg.num_MC > 1) {
                double sq_fdp = 0.0, sq_tpp = 0.0;
                for (int mc = 0; mc < iMC; ++mc) {
                    const std::size_t idx = static_cast<std::size_t>(mc);
                    sq_fdp += (fdp_vec[idx] - mean_fdp) * (fdp_vec[idx] - mean_fdp);
                    sq_tpp += (tpp_vec[idx] - mean_tpp) * (tpp_vec[idx] - mean_tpp);
                }
                sd_fdp = std::sqrt(sq_fdp / (dMC - 1.0));
                sd_tpp = std::sqrt(sq_tpp / (dMC - 1.0));
            }

            results[i_snr][i_rho] = {mean_fdp, mean_tpp, sd_fdp, sd_tpp};

            std::cout << "    [" << label << "] done."
                      << "  TPP=" << std::fixed << std::setprecision(3) << mean_tpp
                      << "  FDP=" << mean_fdp << "\n\n" << std::flush;
        }
    }

    return results;
}


// ==============================================================================
// Print MC 2-D matrix
// ==============================================================================

/**
 * @brief Print a named grid table from a 2-D sweep result.
 *
 * @param title      Table title string.
 * @param row_labels Labels for each row (SNR levels).
 * @param col_labels Labels for each column (rho levels).
 * @param values     Result matrix indexed [row][col].
 * @param print_tpp  If true, print mean_tpp; if false, print mean_fdp.
 */
inline void print_mc_matrix(
    const std::string&                                  title,
    const std::vector<std::string>&                     row_labels,
    const std::vector<std::string>&                     col_labels,
    const std::vector<std::vector<GVSGridPointResult>>& values,
    bool                                                print_tpp)
{
    constexpr int rw = 12;  // row-label width
    constexpr int cw = 8;   // value column width

    std::cout << "\n  " << title << "\n"
              << "  " << std::string(rw + static_cast<int>(col_labels.size()) * cw, '-')
              << "\n";

    // Header
    std::cout << "  " << std::left << std::setw(rw) << "";
    for (const auto& cl : col_labels)
        std::cout << std::right << std::setw(cw) << cl;
    std::cout << "\n";

    // Data rows
    for (std::size_t r = 0; r < values.size() && r < row_labels.size(); ++r) {
        std::cout << "  " << std::left << std::setw(rw) << row_labels[r];
        for (std::size_t c = 0; c < values[r].size() && c < col_labels.size(); ++c) {
            const double val = print_tpp ? values[r][c].mean_tpp
                                         : values[r][c].mean_fdp;
            std::cout << std::right << std::fixed << std::setprecision(4)
                      << std::setw(cw) << val;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


// ==============================================================================
// Print MC generic parameter sweep table
// ==============================================================================

/**
 * @brief Print two per-method tables for a generic 1-D parameter sweep.
 *
 * Generalises print_mc_rho_table() by accepting a custom parameter name for
 * the row-label prefix (e.g. "ar_coef", "rho_sc").  Row labels are formed as
 * param_name + "=" + std::to_string(v).substr(0, 4).
 *
 * @param scenario_tag  Short label printed in the header.
 * @param param_name    Label prefix for each row (max ~7 chars fits rw=12).
 * @param param_grid    The swept parameter values.
 * @param en_results    EN method results per parameter point.
 * @param ien_results   IEN method results per parameter point.
 * @param cfg           Simulation config (num_MC, tFDR, snr, n, p, …).
 */
inline void print_mc_param_sweep_table(
    const std::string&                     scenario_tag,
    const std::string&                     param_name,
    const std::vector<double>&             param_grid,
    const std::vector<GVSGridPointResult>& en_results,
    const std::vector<GVSGridPointResult>& ien_results,
    const GVSSimConfig&                    cfg)
{
    std::cout << "\n" << std::string(78, '=') << "\n"
              << "  T-Rex+GVS MC: " << scenario_tag
              << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
              << ", SNR=" << cfg.snr
              << ", n=" << cfg.n << ", p=" << cfg.p << ")\n"
              << std::string(78, '=') << "\n";

    std::vector<std::string> labels;
    labels.reserve(param_grid.size());
    for (double v : param_grid)
        labels.push_back(param_name + "=" + std::to_string(v).substr(0, 4));

    print_mc_1d_method_table("EN",  labels, en_results);
    print_mc_1d_method_table("IEN", labels, ien_results);

    std::cout << std::string(78, '=') << "\n\n";
}


// ==============================================================================
} // namespace gvs_demo
// ==============================================================================

#endif // DEMOS_TREX_SELECTOR_METHODS_GVS_SIMULATION_UTILS_HPP
