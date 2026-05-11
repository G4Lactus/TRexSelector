// ==============================================================================
// trex_gvs_block_bench_utils.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_GVS_BLOCK_BENCH_UTILS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_GVS_BLOCK_BENCH_UTILS_HPP
// ==============================================================================
/**
 * @file trex_gvs_block_bench_utils.hpp
 *
 * @brief Monte Carlo infrastructure for the block-equicorrelated benchmark
 *        (demo_trex_gvs_08 / sim_trex_gvs_08).
 *
 * Provides:
 *   - BlockBenchConfig           — simulation parameter bundle
 *   - BlockMethodResult          — all metrics for one method on one trial
 *   - BlockTrialResult           — four BlockMethodResults (M1–M4) for one trial
 *   - BlockGridResult            — MC-aggregated metrics per grid cell
 *   - block metric helpers       — compute_block_hit_rate(), …, compute_block_purity_rate()
 *   - run_block_method()         — single selector run → BlockMethodResult
 *   - run_block_single()         — one trial → BlockTrialResult (all 4 methods)
 *   - run_block_mc()             — OpenMP-parallel MC loop → BlockGridResult
 *   - print_block_trial_result() — formatted single-run output (demo use)
 *   - print_block_grid_table()   — formatted MC aggregate table (sim use)
 *
 * Methods compared:
 *   M1  T-Rex + EN,  HAC-clustered groups
 *   M2  T-Rex + EN,  oracle groups (dat.prior_groups)
 *   M3  T-Rex + IEN, HAC-clustered groups
 *   M4  T-Rex + IEN, oracle groups (dat.prior_groups)
 *
 * Note: IEN is known to collapse at ρ ≈ 0.99 (T_stop ≈ 6, TPR ≈ 0).
 * The default rho_grid uses {0.5, 0.9} which avoids this regime.
 */
// ==============================================================================

#include "trex_gvs_simulation_utils.hpp"  // includes trex_gvs_dgps.hpp

// std includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen
#include <Eigen/Dense>

// OpenMP
#include <utils/openmp/utils_openmp.hpp>


// ==============================================================================
namespace gvs_demo {
// ==============================================================================


// ==============================================================================
// Simulation configuration
// ==============================================================================

/** @brief Parameter bundle for the block-equicorrelated MC benchmark. */
struct BlockBenchConfig {
    int    n               = 200;  ///< Observations.
    int    p               = 500;  ///< Predictors.
    int    G               = 100;  ///< Number of blocks.
    int    block_size      = 5;    ///< Variables per block.
    int    n_active_blocks = 10;   ///< Active blocks per trial.
    double tFDR            = 0.1;  ///< Target FDR level.
    std::size_t K          = 20;   ///< Random experiments per selector run.
    std::size_t num_MC     = 500;  ///< Monte Carlo trials per grid cell.
    int    base_seed       = 2026; ///< Seed for trial 0; trial i uses base_seed + i.
    double corr_max        = 0.5;  ///< corr_max for HAC auto-clustering.
    double b               = 3.0;  ///< Nonzero coefficient magnitude.
};


// ==============================================================================
// Per-method result (one trial)
// ==============================================================================

/**
 * @brief All metrics collected for one method on one MC trial.
 *
 * Coordinate metrics (evaluated against the true variable support):
 *   coord_fdp    = |S_hat \ S_true| / max(|S_hat|, 1)
 *   coord_tpr    = |S_hat ∩ S_true| / |S_true|
 *   exact_support = 1( S_hat == S_true )
 *
 * Block metrics (evaluated against the active block set):
 *   block_hit_rate        = (active blocks with ≥1 selected var) / n_active_blocks
 *   block_fdp             = (null blocks hit) / max(total blocks hit, 1)
 *   full_block_rate       = (active blocks fully in S_hat) / n_active_blocks
 *   null_block_activation = (null blocks with ≥1 selected var) / null_block_count
 *
 * Group-recovery diagnostic:
 *   block_purity_rate = (true blocks where all members share same groups_vec value)
 *                       / G
 *   (= 1.0 by definition for oracle runs)
 *
 * Selector diagnostics:
 *   lambda2_used, T_stop, M_found, elapsed_sec
 */
struct BlockMethodResult {
    // Coordinate-level metrics
    double coord_fdp     = 0.0;
    double coord_tpr     = 0.0;
    double exact_support = 0.0;

    // Block-level metrics
    double block_hit_rate        = 0.0;
    double block_fdp             = 0.0;
    double full_block_rate       = 0.0;
    double null_block_activation = 0.0;

    // Group-recovery diagnostic
    double block_purity_rate = 0.0;

    // Selector diagnostics
    double      lambda2_used = 0.0;
    std::size_t T_stop       = 0;
    std::size_t M_found      = 0;
    double      elapsed_sec  = 0.0;
};


// ==============================================================================
// Trial result (all 4 methods)
// ==============================================================================

/** @brief Results for all four method variants on a single MC trial. */
struct BlockTrialResult {
    BlockMethodResult M1;  ///< EN  + HAC-clustered groups
    BlockMethodResult M2;  ///< EN  + oracle groups
    BlockMethodResult M3;  ///< IEN + HAC-clustered groups
    BlockMethodResult M4;  ///< IEN + oracle groups
};


// ==============================================================================
// MC aggregate result
// ==============================================================================

/** @brief Aggregated statistics for one method over num_MC trials at one grid cell. */
struct BlockMethodAggregate {
    // Coordinate-level metrics
    double mean_coord_fdp = 0.0;   double sd_coord_fdp = 0.0;   double mcse_coord_fdp = 0.0;
    double mean_coord_tpr = 0.0;   double sd_coord_tpr = 0.0;   double mcse_coord_tpr = 0.0;
    double mean_exact_sup = 0.0;   double sd_exact_sup = 0.0;

    // Block-level metrics
    double mean_block_hit  = 0.0;  double sd_block_hit  = 0.0;
    double mean_block_fdp  = 0.0;  double sd_block_fdp  = 0.0;
    double mean_full_block = 0.0;  double sd_full_block = 0.0;
    double mean_null_act   = 0.0;  double sd_null_act   = 0.0;

    // Group-recovery diagnostic
    double mean_purity = 0.0;      double sd_purity = 0.0;

    // Selector diagnostics (means only)
    double      mean_lambda2 = 0.0;
    double      mean_T_stop  = 0.0;
    double      mean_M_found = 0.0;
    double      mean_elapsed = 0.0;
};

/** @brief All four method aggregates for one grid cell. */
struct BlockGridResult {
    BlockMethodAggregate M1;
    BlockMethodAggregate M2;
    BlockMethodAggregate M3;
    BlockMethodAggregate M4;
};


// ==============================================================================
// Block metric helpers
// ==============================================================================

/**
 * @brief Fraction of active blocks that contain at least one selected variable.
 *
 * @param sel      Selected variable indices (0-based).
 * @param active   Active block indices (0-based block IDs).
 * @param block_size  Variables per block.
 * @return block_hit_rate in [0, 1].
 */
inline double compute_block_hit_rate(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             block_size)
{
    if (active.empty()) return 0.0;
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    int hits = 0;
    for (int blk : active) {
        const int base = blk * block_size;
        for (int jj = 0; jj < block_size; ++jj) {
            if (sel_set.count(static_cast<std::size_t>(base + jj))) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(active.size());
}


/**
 * @brief False discovery proportion at the block level.
 *
 * block_fdp = (# null blocks with ≥1 selected var) / max(# total blocks hit, 1)
 *
 * @param sel         Selected variable indices (0-based).
 * @param active      Active block indices.
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @return block_fdp in [0, 1].
 */
inline double compute_block_fdp(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             G,
    int                             block_size)
{
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    const std::unordered_set<int>         active_set(active.begin(), active.end());

    int total_hit = 0, null_hit = 0;
    for (int blk = 0; blk < G; ++blk) {
        const int base = blk * block_size;
        bool hit = false;
        for (int jj = 0; jj < block_size; ++jj) {
            if (sel_set.count(static_cast<std::size_t>(base + jj))) {
                hit = true;
                break;
            }
        }
        if (hit) {
            ++total_hit;
            if (!active_set.count(blk)) ++null_hit;
        }
    }
    return static_cast<double>(null_hit) /
           static_cast<double>(std::max(total_hit, 1));
}


/**
 * @brief Fraction of active blocks that are fully contained in the selection.
 *
 * @param sel         Selected variable indices (0-based).
 * @param active      Active block indices.
 * @param block_size  Variables per block.
 * @return full_block_rate in [0, 1].
 */
inline double compute_full_block_rate(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             block_size)
{
    if (active.empty()) return 0.0;
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    int fully_hit = 0;
    for (int blk : active) {
        const int base = blk * block_size;
        bool full = true;
        for (int jj = 0; jj < block_size; ++jj) {
            if (!sel_set.count(static_cast<std::size_t>(base + jj))) {
                full = false;
                break;
            }
        }
        if (full) ++fully_hit;
    }
    return static_cast<double>(fully_hit) / static_cast<double>(active.size());
}


/**
 * @brief Fraction of null blocks that contain at least one selected variable.
 *
 * @param sel         Selected variable indices (0-based).
 * @param active      Active block indices.
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @return null_block_activation in [0, 1].
 */
inline double compute_null_block_activation(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             G,
    int                             block_size)
{
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    const std::unordered_set<int>         active_set(active.begin(), active.end());

    int null_total = 0, null_hit = 0;
    for (int blk = 0; blk < G; ++blk) {
        if (active_set.count(blk)) continue;
        ++null_total;
        const int base = blk * block_size;
        for (int jj = 0; jj < block_size; ++jj) {
            if (sel_set.count(static_cast<std::size_t>(base + jj))) {
                ++null_hit;
                break;
            }
        }
    }
    return (null_total == 0) ? 0.0
         : static_cast<double>(null_hit) / static_cast<double>(null_total);
}


/**
 * @brief Fraction of true blocks whose members all share the same groups_vec value.
 *
 * A block is "pure" if groups_vec(base) == groups_vec(base+1) == … == groups_vec(base+block_size-1).
 * This is always 1.0 for oracle runs (prior_groups supplied).
 *
 * @param groups_vec  Cluster assignment per variable (length p, from GVSSelectionResult).
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @return block_purity_rate in [0, 1].
 */
inline double compute_block_purity_rate(
    const Eigen::VectorXi& groups_vec,
    int                    G,
    int                    block_size)
{
    int pure = 0;
    for (int k = 0; k < G; ++k) {
        const int base  = k * block_size;
        const int first = groups_vec(base);
        bool is_pure    = true;
        for (int jj = 1; jj < block_size; ++jj) {
            if (groups_vec(base + jj) != first) { is_pure = false; break; }
        }
        if (is_pure) ++pure;
    }
    return static_cast<double>(pure) / static_cast<double>(G);
}


/**
 * @brief Exact support recovery: 1.0 if S_hat == S_true, else 0.0.
 *
 * @param sel        Selected variable indices (0-based, need not be sorted).
 * @param true_sup   True support (sorted, 0-based).
 * @return 1.0 or 0.0.
 */
inline double compute_exact_support(
    const std::vector<std::size_t>& sel,
    const std::vector<std::size_t>& true_sup)
{
    if (sel.size() != true_sup.size()) return 0.0;
    std::vector<std::size_t> sorted_sel = sel;
    std::sort(sorted_sel.begin(), sorted_sel.end());
    return (sorted_sel == true_sup) ? 1.0 : 0.0;
}


// ==============================================================================
// Single-method run helper
// ==============================================================================

/**
 * @brief Run TRexGVSSelector once and collect all BlockMethodResult metrics.
 *
 * @param X_map       Non-owning map over the design matrix (n × p).
 * @param y_map       Non-owning map over the response (n).
 * @param dat         Full DGP result (true_support, active_blocks, prior_groups).
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @param tFDR        Target FDR level.
 * @param gvs_ctrl    GVS control parameters (gvs_type and prior_groups set by caller).
 * @param trex_ctrl   Base T-Rex control parameters.
 * @param seed        RNG seed passed to the selector.
 * @return Populated BlockMethodResult.
 */
inline BlockMethodResult run_block_method(
    Eigen::Map<Eigen::MatrixXd>&   X_map,
    Eigen::Map<Eigen::VectorXd>&   y_map,
    const GVSDGPResult&            dat,
    int                            G,
    int                            block_size,
    double                         tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    int                            seed)
{
    const auto t0 = std::chrono::steady_clock::now();

    // Solver dispatch: EN → TENET, IEN → TLASSO
    TRexControlParameter trex_run = trex_ctrl;
    if (gvs_ctrl.gvs_type == GVSType::EN)
        trex_run.solver_type = SolverTypeForTRex::TENET;
    else
        trex_run.solver_type = SolverTypeForTRex::TLASSO;

    TRexGVSSelector selector(X_map, y_map, tFDR, gvs_ctrl, trex_run,
                              seed, /*verbose=*/false);
    selector.select();

    const auto t1 = std::chrono::steady_clock::now();

    const auto& sel_idx  = selector.getSelectedIndices();
    const auto& gvs_res  = selector.getGVSResult();

    BlockMethodResult out;
    out.coord_fdp     = rates::compute_fdp(sel_idx, dat.true_support);
    out.coord_tpr     = rates::compute_tpp(sel_idx, dat.true_support);
    out.exact_support = compute_exact_support(sel_idx, dat.true_support);

    out.block_hit_rate        = compute_block_hit_rate(sel_idx, dat.active_blocks,
                                                        block_size);
    out.block_fdp             = compute_block_fdp(sel_idx, dat.active_blocks,
                                                   G, block_size);
    out.full_block_rate       = compute_full_block_rate(sel_idx, dat.active_blocks,
                                                         block_size);
    out.null_block_activation = compute_null_block_activation(sel_idx, dat.active_blocks,
                                                               G, block_size);
    out.block_purity_rate     = compute_block_purity_rate(gvs_res.groups_vec,
                                                           G, block_size);

    out.lambda2_used = gvs_res.lambda2_used;
    out.T_stop       = gvs_res.T_stop;
    out.M_found      = gvs_res.max_clusters;
    out.elapsed_sec  =
        std::chrono::duration<double>(t1 - t0).count();

    return out;
}


// ==============================================================================
// Single-trial runner (all 4 methods)
// ==============================================================================

/**
 * @brief Run all four method variants on one generated dataset.
 *
 * M1: EN  + HAC-clustered groups (prior_groups empty)
 * M2: EN  + oracle groups        (prior_groups = dat.prior_groups)
 * M3: IEN + HAC-clustered groups (prior_groups empty)
 * M4: IEN + oracle groups        (prior_groups = dat.prior_groups)
 *
 * @param dat        Generated dataset (from make_block_equicorr_dgp).
 * @param G          Total number of blocks.
 * @param block_size Variables per block.
 * @param tFDR       Target FDR level.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param corr_max   corr_max passed to HAC auto-clustering methods.
 * @param seed       RNG seed passed to the selector.
 * @return BlockTrialResult with results for all four methods.
 */
inline BlockTrialResult run_block_single(
    const GVSDGPResult&         dat,
    int                         G,
    int                         block_size,
    double                      tFDR,
    const TRexControlParameter& trex_ctrl,
    double                      corr_max,
    int                         seed)
{
    Eigen::Map<Eigen::MatrixXd> X_map(
        const_cast<double*>(dat.X.data()), dat.X.rows(), dat.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(
        const_cast<double*>(dat.y.data()), dat.y.rows());

    // --- Base GVS control parameters ---
    TRexGVSControlParameter gvs_en;
    gvs_en.gvs_type       = GVSType::EN;
    gvs_en.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_en.corr_max       = corr_max;
    gvs_en.hc_linkage     = hac::LinkageMethod::Single;

    TRexGVSControlParameter gvs_ien;
    gvs_ien.gvs_type       = GVSType::IEN;
    gvs_ien.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ien.corr_max       = corr_max;
    gvs_ien.hc_linkage     = hac::LinkageMethod::Single;

    // Oracle variants carry the true prior_groups
    TRexGVSControlParameter gvs_en_oracle  = gvs_en;
    TRexGVSControlParameter gvs_ien_oracle = gvs_ien;
    gvs_en_oracle.prior_groups  = dat.prior_groups;
    gvs_ien_oracle.prior_groups = dat.prior_groups;

    BlockTrialResult out;
    out.M1 = run_block_method(X_map, y_map, dat, G, block_size, tFDR,
                               gvs_en,        trex_ctrl, seed);
    out.M2 = run_block_method(X_map, y_map, dat, G, block_size, tFDR,
                               gvs_en_oracle,  trex_ctrl, seed);
    out.M3 = run_block_method(X_map, y_map, dat, G, block_size, tFDR,
                               gvs_ien,       trex_ctrl, seed);
    out.M4 = run_block_method(X_map, y_map, dat, G, block_size, tFDR,
                               gvs_ien_oracle, trex_ctrl, seed);
    return out;
}


// ==============================================================================
// MC aggregation helper
// ==============================================================================

namespace detail {

/** @brief Compute mean and standard deviation of a vector. */
inline void mean_sd(const std::vector<double>& v, double& mean, double& sd)
{
    if (v.empty()) { mean = sd = 0.0; return; }
    const double dN = static_cast<double>(v.size());
    mean = std::accumulate(v.begin(), v.end(), 0.0) / dN;
    if (v.size() < 2) { sd = 0.0; return; }
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    sd = std::sqrt(sq / (dN - 1.0));
}

} // namespace detail


// ==============================================================================
// MC loop
// ==============================================================================

/**
 * @brief Run num_MC parallel trials at one (scenario, snr, rho) point.
 *
 * @param dgp_fn     DGP factory — captures scenario, n, p, G, block_size,
 *                   n_active_blocks, rho, snr, b via closure.
 *                   Signature: (seed) → GVSDGPResult.
 * @param cfg        Benchmark configuration.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param label      Progress label for console output.
 * @return BlockGridResult aggregating all four methods.
 */
inline BlockGridResult run_block_mc(
    const std::function<GVSDGPResult(unsigned)>& dgp_fn,
    const BlockBenchConfig&                      cfg,
    const TRexControlParameter&                  trex_ctrl,
    const std::string&                           label)
{
    const int iMC = static_cast<int>(cfg.num_MC);
    const unsigned base_seed = static_cast<unsigned>(cfg.base_seed);

    // Accumulators for all metrics, all 4 methods
    std::vector<double>
        m1_cfdp(cfg.num_MC), m1_ctpr(cfg.num_MC), m1_esup(cfg.num_MC),
        m1_bhr (cfg.num_MC), m1_bfdp(cfg.num_MC),
        m1_fbr (cfg.num_MC), m1_nba (cfg.num_MC), m1_pur (cfg.num_MC),

        m2_cfdp(cfg.num_MC), m2_ctpr(cfg.num_MC), m2_esup(cfg.num_MC),
        m2_bhr (cfg.num_MC), m2_bfdp(cfg.num_MC),
        m2_fbr (cfg.num_MC), m2_nba (cfg.num_MC), m2_pur (cfg.num_MC),

        m3_cfdp(cfg.num_MC), m3_ctpr(cfg.num_MC), m3_esup(cfg.num_MC),
        m3_bhr (cfg.num_MC), m3_bfdp(cfg.num_MC),
        m3_fbr (cfg.num_MC), m3_nba (cfg.num_MC), m3_pur (cfg.num_MC),

        m4_cfdp(cfg.num_MC), m4_ctpr(cfg.num_MC), m4_esup(cfg.num_MC),
        m4_bhr (cfg.num_MC), m4_bfdp(cfg.num_MC),
        m4_fbr (cfg.num_MC), m4_nba (cfg.num_MC), m4_pur (cfg.num_MC),

        m1_lam(cfg.num_MC), m1_ts(cfg.num_MC), m1_mf(cfg.num_MC), m1_el(cfg.num_MC),
        m2_lam(cfg.num_MC), m2_ts(cfg.num_MC), m2_mf(cfg.num_MC), m2_el(cfg.num_MC),
        m3_lam(cfg.num_MC), m3_ts(cfg.num_MC), m3_mf(cfg.num_MC), m3_el(cfg.num_MC),
        m4_lam(cfg.num_MC), m4_ts(cfg.num_MC), m4_mf(cfg.num_MC), m4_el(cfg.num_MC);

    std::cout << "  " << label << " — running " << cfg.num_MC
              << " MC trials...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed + static_cast<unsigned>(mc);
        auto dat = dgp_fn(trial_seed);

        const auto res = run_block_single(dat, cfg.G, cfg.block_size,
                                          cfg.tFDR, trex_ctrl,
                                          cfg.corr_max,
                                          static_cast<int>(trial_seed));
        const std::size_t idx = static_cast<std::size_t>(mc);

        m1_cfdp[idx] = res.M1.coord_fdp;    m1_ctpr[idx] = res.M1.coord_tpr;
        m1_esup[idx] = res.M1.exact_support;
        m1_bhr[idx]  = res.M1.block_hit_rate;  m1_bfdp[idx] = res.M1.block_fdp;
        m1_fbr[idx]  = res.M1.full_block_rate; m1_nba[idx]  = res.M1.null_block_activation;
        m1_pur[idx]  = res.M1.block_purity_rate;
        m1_lam[idx]  = res.M1.lambda2_used;
        m1_ts[idx]   = static_cast<double>(res.M1.T_stop);
        m1_mf[idx]   = static_cast<double>(res.M1.M_found);
        m1_el[idx]   = res.M1.elapsed_sec;

        m2_cfdp[idx] = res.M2.coord_fdp;    m2_ctpr[idx] = res.M2.coord_tpr;
        m2_esup[idx] = res.M2.exact_support;
        m2_bhr[idx]  = res.M2.block_hit_rate;  m2_bfdp[idx] = res.M2.block_fdp;
        m2_fbr[idx]  = res.M2.full_block_rate; m2_nba[idx]  = res.M2.null_block_activation;
        m2_pur[idx]  = res.M2.block_purity_rate;
        m2_lam[idx]  = res.M2.lambda2_used;
        m2_ts[idx]   = static_cast<double>(res.M2.T_stop);
        m2_mf[idx]   = static_cast<double>(res.M2.M_found);
        m2_el[idx]   = res.M2.elapsed_sec;

        m3_cfdp[idx] = res.M3.coord_fdp;    m3_ctpr[idx] = res.M3.coord_tpr;
        m3_esup[idx] = res.M3.exact_support;
        m3_bhr[idx]  = res.M3.block_hit_rate;  m3_bfdp[idx] = res.M3.block_fdp;
        m3_fbr[idx]  = res.M3.full_block_rate; m3_nba[idx]  = res.M3.null_block_activation;
        m3_pur[idx]  = res.M3.block_purity_rate;
        m3_lam[idx]  = res.M3.lambda2_used;
        m3_ts[idx]   = static_cast<double>(res.M3.T_stop);
        m3_mf[idx]   = static_cast<double>(res.M3.M_found);
        m3_el[idx]   = res.M3.elapsed_sec;

        m4_cfdp[idx] = res.M4.coord_fdp;    m4_ctpr[idx] = res.M4.coord_tpr;
        m4_esup[idx] = res.M4.exact_support;
        m4_bhr[idx]  = res.M4.block_hit_rate;  m4_bfdp[idx] = res.M4.block_fdp;
        m4_fbr[idx]  = res.M4.full_block_rate; m4_nba[idx]  = res.M4.null_block_activation;
        m4_pur[idx]  = res.M4.block_purity_rate;
        m4_lam[idx]  = res.M4.lambda2_used;
        m4_ts[idx]   = static_cast<double>(res.M4.T_stop);
        m4_mf[idx]   = static_cast<double>(res.M4.M_found);
        m4_el[idx]   = res.M4.elapsed_sec;
    }

    const double dMC = static_cast<double>(cfg.num_MC);

    auto build_agg = [&](
        const std::vector<double>& cfdp, const std::vector<double>& ctpr,
        const std::vector<double>& esup,
        const std::vector<double>& bhr,  const std::vector<double>& bfdp,
        const std::vector<double>& fbr,  const std::vector<double>& nba,
        const std::vector<double>& pur,
        const std::vector<double>& lam,  const std::vector<double>& ts,
        const std::vector<double>& mf,   const std::vector<double>& el)
    {
        BlockMethodAggregate a;
        detail::mean_sd(cfdp, a.mean_coord_fdp, a.sd_coord_fdp);
        detail::mean_sd(ctpr, a.mean_coord_tpr, a.sd_coord_tpr);
        detail::mean_sd(esup, a.mean_exact_sup, a.sd_exact_sup);
        detail::mean_sd(bhr,  a.mean_block_hit, a.sd_block_hit);
        detail::mean_sd(bfdp, a.mean_block_fdp, a.sd_block_fdp);
        detail::mean_sd(fbr,  a.mean_full_block, a.sd_full_block);
        detail::mean_sd(nba,  a.mean_null_act,  a.sd_null_act);
        detail::mean_sd(pur,  a.mean_purity,    a.sd_purity);
        a.mcse_coord_fdp = (cfg.num_MC > 1) ? a.sd_coord_fdp / std::sqrt(dMC) : 0.0;
        a.mcse_coord_tpr = (cfg.num_MC > 1) ? a.sd_coord_tpr / std::sqrt(dMC) : 0.0;
        a.mean_lambda2 = std::accumulate(lam.begin(), lam.end(), 0.0) / dMC;
        a.mean_T_stop  = std::accumulate(ts.begin(),  ts.end(),  0.0) / dMC;
        a.mean_M_found = std::accumulate(mf.begin(),  mf.end(),  0.0) / dMC;
        a.mean_elapsed = std::accumulate(el.begin(),  el.end(),  0.0) / dMC;
        return a;
    };

    BlockGridResult out;
    out.M1 = build_agg(m1_cfdp, m1_ctpr, m1_esup, m1_bhr, m1_bfdp,
                        m1_fbr, m1_nba, m1_pur, m1_lam, m1_ts, m1_mf, m1_el);
    out.M2 = build_agg(m2_cfdp, m2_ctpr, m2_esup, m2_bhr, m2_bfdp,
                        m2_fbr, m2_nba, m2_pur, m2_lam, m2_ts, m2_mf, m2_el);
    out.M3 = build_agg(m3_cfdp, m3_ctpr, m3_esup, m3_bhr, m3_bfdp,
                        m3_fbr, m3_nba, m3_pur, m3_lam, m3_ts, m3_mf, m3_el);
    out.M4 = build_agg(m4_cfdp, m4_ctpr, m4_esup, m4_bhr, m4_bfdp,
                        m4_fbr, m4_nba, m4_pur, m4_lam, m4_ts, m4_mf, m4_el);

    std::cout << "  " << label << " — done."
              << "  M1 TPR=" << std::fixed << std::setprecision(3)
              << out.M1.mean_coord_tpr
              << "  M2 TPR=" << out.M2.mean_coord_tpr
              << "  M3 TPR=" << out.M3.mean_coord_tpr
              << "  M4 TPR=" << out.M4.mean_coord_tpr
              << "\n\n" << std::flush;

    return out;
}


// ==============================================================================
// Print — single trial (demo use)
// ==============================================================================

/**
 * @brief Print a formatted single-run result block for all four methods.
 *
 * @param scenario_tag  Label for the scenario (e.g. "INDIVIDUAL").
 * @param dat           DGP metadata.
 * @param res           BlockTrialResult for all four methods.
 * @param tFDR          Target FDR (for display).
 * @param G             Total number of blocks.
 * @param block_size    Variables per block.
 */
inline void print_block_trial_result(
    const std::string&    scenario_tag,
    const GVSDGPResult&   dat,
    const BlockTrialResult& res,
    double                tFDR,
    int                   G,
    int                   block_size,
    double                rho)
{
    auto print_method = [&](const std::string& mname,
                             const BlockMethodResult& m) {
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  " << mname << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Grouping:     purity=" << m.block_purity_rate
                  << "  M=" << m.M_found
                  << "  lambda2=" << m.lambda2_used
                  << "  T_stop=" << m.T_stop << "\n";
        std::cout << "  Coord:        TPR=" << m.coord_tpr
                  << "  FDP=" << m.coord_fdp
                  << "  exact=" << m.exact_support << "\n";
        std::cout << "  Block:        hit_rate=" << m.block_hit_rate
                  << "  block_fdp=" << m.block_fdp
                  << "  full=" << m.full_block_rate
                  << "  null_act=" << m.null_block_activation << "\n";
        std::cout << "  Wall-clock:   " << std::setprecision(2)
                  << m.elapsed_sec << " s\n";
    };

    std::cout << std::string(70, '=') << "\n";
    std::cout << "  Block Bench: " << scenario_tag << "\n";
    std::cout << "  n=" << dat.n << "  p=" << dat.p
              << "  G=" << G << "  block_size=" << block_size
              << "  s=" << dat.s << "\n";
    std::cout << "  rho=" << std::fixed << std::setprecision(2) << rho
              << "  SNR=" << dat.snr
              << "  sigma_y=" << std::setprecision(4) << dat.sigma_y
              << "  tFDR=" << tFDR << "\n";
    std::cout << "  Active blocks: ";
    for (int b : dat.active_blocks) std::cout << b << " ";
    std::cout << "\n";

    print_method("M1  EN  + HAC-clustered", res.M1);
    print_method("M2  EN  + oracle groups", res.M2);
    print_method("M3  IEN + HAC-clustered", res.M3);
    print_method("M4  IEN + oracle groups", res.M4);

    std::cout << std::string(70, '=') << "\n\n";
}


// ==============================================================================
// Print — MC aggregate table (sim use)
// ==============================================================================

/**
 * @brief Print a formatted MC aggregate table for one scenario.
 *
 * Columns: Method | coord_TPR | coord_FDP | block_metric | purity | T_stop | elapsed
 * Rows: M1–M4 for each (rho, snr) cell.
 *
 * @param scenario_tag     Label for the scenario.
 * @param scenario         BlockScenario enum (controls which block metric is shown).
 * @param rho_grid         Rho values swept.
 * @param snr_grid         SNR values swept.
 * @param results          results[i_rho][i_snr] — BlockGridResult.
 * @param cfg              Benchmark configuration.
 */
inline void print_block_grid_table(
    const std::string&                                    scenario_tag,
    BlockScenario                                         scenario,
    const std::vector<double>&                            rho_grid,
    const std::vector<double>&                            snr_grid,
    const std::vector<std::vector<BlockGridResult>>&      results,
    const BlockBenchConfig&                               cfg)
{
    // Choose the scenario-specific block metric name and accessor
    std::string blk_metric_name;
    switch (scenario) {
        case BlockScenario::INDIVIDUAL:    blk_metric_name = "blk_hit"; break;
        case BlockScenario::REPRESENTATIVE: blk_metric_name = "blk_hit"; break;
        case BlockScenario::WHOLE:          blk_metric_name = "full_blk"; break;
    }

    auto get_blk_metric = [&](const BlockMethodAggregate& a) -> double {
        switch (scenario) {
            case BlockScenario::INDIVIDUAL:    return a.mean_block_hit;
            case BlockScenario::REPRESENTATIVE: return a.mean_block_hit;
            case BlockScenario::WHOLE:          return a.mean_full_block;
        }
        return 0.0;
    };

    constexpr int cw = 9;
    const std::string methods[4] = {"M1(EN-C)", "M2(EN-O)", "M3(IE-C)", "M4(IE-O)"};
    auto get_agg = [](const BlockGridResult& g, int m) -> const BlockMethodAggregate& {
        switch (m) {
            case 0: return g.M1;
            case 1: return g.M2;
            case 2: return g.M3;
            default: return g.M4;
        }
    };

    std::cout << "\n" << std::string(90, '=') << "\n"
              << "  Block Bench MC: " << scenario_tag
              << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
              << ", n=" << cfg.n << ", p=" << cfg.p
              << ", G=" << cfg.G << ", blk=" << cfg.block_size << ")\n"
              << std::string(90, '=') << "\n";

    for (std::size_t i_rho = 0; i_rho < rho_grid.size(); ++i_rho) {
        std::cout << "\n  rho = " << std::fixed << std::setprecision(2)
                  << rho_grid[i_rho] << "\n";

        // Header
        std::cout << "  " << std::left << std::setw(12) << "method"
                  << std::right
                  << std::setw(7) << "SNR"
                  << std::setw(cw) << "TPR"
                  << std::setw(cw) << "FDR"
                  << std::setw(cw) << "block_FDR"
                  << std::setw(cw) << blk_metric_name
                  << std::setw(cw) << "purity"
                  << std::setw(cw) << "T_stop"
                  << std::setw(cw) << "M_found"
                  << "\n"
                  << "  " << std::string(12 + 7 + 7 * cw, '-') << "\n";

        for (std::size_t i_snr = 0; i_snr < snr_grid.size(); ++i_snr) {
            for (int m = 0; m < 4; ++m) {
                const auto& a = get_agg(results[i_rho][i_snr], m);
                std::cout << "  " << std::left << std::setw(12) << methods[m]
                          << std::right << std::fixed << std::setprecision(2)
                          << std::setw(7) << snr_grid[i_snr]
                          << std::setprecision(4)
                          << std::setw(cw) << a.mean_coord_tpr
                          << std::setw(cw) << a.mean_coord_fdp
                          << std::setw(cw) << a.mean_block_fdp
                          << std::setw(cw) << get_blk_metric(a)
                          << std::setw(cw) << a.mean_purity
                          << std::setprecision(1)
                          << std::setw(cw) << a.mean_T_stop
                          << std::setprecision(1)
                          << std::setw(cw) << a.mean_M_found
                          << "\n";
            }
            std::cout << "\n";
        }
    }
    std::cout << std::string(90, '=') << "\n\n";
}


// ==============================================================================
} // namespace gvs_demo
// ==============================================================================

#endif // DEMOS_TREX_SELECTOR_METHODS_GVS_BLOCK_BENCH_UTILS_HPP
