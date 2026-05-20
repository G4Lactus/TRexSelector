// ==============================================================================
// da_trex_sim_common.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP
#define DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP
// ==============================================================================
/**
 * @file da_trex_sim_common.hpp
 *
 * @brief Shared infrastructure for DA-TRex MC simulations.
 *
 * Provides:
 *   - BlockSimConfig / SimConfig    — simulation parameter structs
 *   - SolverInfo                    — solver descriptor
 *   - Support policies              — Random, CappedSpread, Clustered, OnePerBlock
 *   - make_support()                — support generator factory
 *   - make_trex_control()           — default TRexControlParameter
 *   - make_da_bt_control()          — default DA binary tree (BT) control
 *   - run_mc_trials()               — inner MC loop
 *   - evaluate_selection()          — single-run FDP/TPP evaluation
 *   - run_snr_sweep()               — full solver × SNR MC sweep
 *   - save_and_print_grid_results() — tabular output to console + file
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP
#include <utils/openmp/utils_openmp.hpp>

// T-Rex includes
#include <trex_selector_methods/trex_da/trex_da.hpp>

//utils includes
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// DGP generators
#include "dgp_generators.hpp"


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace da_sim {

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace rates       = trex::utils::eval::rates;
namespace dummygen    = trex::utils::datageneration::dummygen;
namespace hac         = trex::ml_methods::clustering::hierarchical::agglomerative;

using trex::trex_selector_methods::trex_da::TRexDASelector;
using trex::trex_selector_methods::trex_da::TRexDAControlParameter;
using trex::trex_selector_methods::trex_da::DAMethod;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::utils::solver_dispatch::SolverHyperparameters;


// ==============================================================================
// Configuration
// ==============================================================================

// Base simulation parameters
struct BlockSimConfig {
    int  n             = 150;     // observations
    int  p             = 500;     // predictors
    int  G             = 5;       // active groups (blocks with an active variable)
    int  g             = 5;       // group (block) size
    double rho         = 0.7;     // within-block correlation
    double snr         = 2.0;     // signal-to-noise ratio
    double tFDR        = 0.2;     // target FDR
    std::size_t K      = 20;      // random experiments per selector run
    std::size_t num_MC = 200; // Monte Carlo trials
    int base_seed      = 2026;    // base RNG seed
};

// General simulation parameters
struct SimConfig {
    int  n             = 300;     // observations
    int  p             = 1000;    // predictors
    int  s             = 10;      // number of active predictors
    double amplitude   = 3.0;     // signal coefficient
    double snr         = 2.0;     // signal-to-noise ratio
    double tFDR        = 0.2;     // target FDR
    std::size_t K      = 20;      // random experiments per selector run
    std::size_t num_MC = 200;   // Monte Carlo trials
    int base_seed      = 2026;    // base RNG seed
};

// Solver descriptor
struct SolverInfo {
    SolverTypeForTRex type;
    std::string       name;
    double            rho_afs        = 0.0;
    bool              use_stagnation = true;
};


// ==============================================================================
// Support policies
// ==============================================================================

/**
 * @brief Uniform random support (without replacement).
 *
 * Draws s indices uniformly at random from [0, p), sorted ascending.
 * Support changes per call when the RNG state differs.
 *
 * @param s   Number of active predictors.
 * @param p   Total number of predictors.
 * @param rng Random number generator (modified in-place).
 */
inline std::vector<std::size_t> random_support(int s, int p, std::mt19937& rng) {
    std::vector<std::size_t> pool(static_cast<std::size_t>(p));
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng);
    pool.resize(static_cast<std::size_t>(s));
    std::sort(pool.begin(), pool.end());

    return pool;
}


/**
 * @brief Parameterized cluster support.
 *
 * Distributes s active predictors into num_clusters clusters, each of random
 * size in [1, max_cluster_size]. Cluster anchors are placed at random
 * positions within evenly-spaced zones across [0, p). Produces a natural
 * mixture of singletons and small groups.
 *
 * @param s                Number of active predictors.
 * @param p                Total number of predictors.
 * @param num_clusters     Number of clusters (0 = auto: ceil(s / max_cluster_size)).
 * @param max_cluster_size Maximum elements per cluster (2 or 3 recommended).
 * @param rng              Random number generator (modified in-place).
 */
inline std::vector<std::size_t> clustered_support(int s,
                                                  int p,
                                                  int num_clusters,
                                                  int max_cluster_size,
                                                  std::mt19937& rng)
{
    if (num_clusters <= 0) {
        num_clusters = (s + max_cluster_size - 1) / max_cluster_size;
    }
    num_clusters = std::min(num_clusters, s);

    // Distribute s elements across num_clusters, each size in [1, max_cluster_size]
    std::vector<int> sizes(static_cast<std::size_t>(num_clusters), 1);
    int remaining = s - num_clusters;

    std::vector<int> expandable;
    expandable.reserve(static_cast<std::size_t>(num_clusters));
    for (int c = 0; c < num_clusters; ++c) {
        expandable.push_back(c);
    }

    while (remaining > 0 && !expandable.empty()) {
        std::uniform_int_distribution<int> pick(
            0, static_cast<int>(expandable.size()) - 1);
        int idx = pick(rng);
        int c   = expandable[static_cast<std::size_t>(idx)];
        ++sizes[static_cast<std::size_t>(c)];
        --remaining;
        if (sizes[static_cast<std::size_t>(c)] >= max_cluster_size) {
            expandable.erase(expandable.begin() + idx);
        }
    }

    // Spill-over: if s > num_clusters * max_cluster_size, extend last cluster
    if (remaining > 0) {
        sizes.back() += remaining;
    }

    // Place clusters in evenly-spaced zones
    const int spacing = p / num_clusters;

    std::vector<std::size_t> sup;
    sup.reserve(static_cast<std::size_t>(s));

    for (int c = 0; c < num_clusters; ++c) {
        int zone_start = c * spacing;
        int max_anchor = zone_start + spacing
                       - sizes[static_cast<std::size_t>(c)];
        max_anchor = std::max(max_anchor, zone_start);

        std::uniform_int_distribution<int> anchor_dist(zone_start, max_anchor);
        int anchor = anchor_dist(rng);

        for (Eigen::Index j = 0; j < sizes[static_cast<std::size_t>(c)]; ++j) {
            sup.push_back(static_cast<std::size_t>(anchor + j));
        }
    }

    std::sort(sup.begin(), sup.end());
    return sup;
}


/**
 * @brief Evenly-spaced support with capped gap.
 *
 * Places s active variables at equal intervals, but the gap is capped at
 * max_gap. For s=10, p=1000, max_gap=10: {0, 10, 20, ..., 90}.
 * This gives mild residual correlation under AR(1) (rho^10 ≈ 0.03 at rho=0.7).
 *
 * @param s       Number of active predictors.
 * @param p       Total number of predictors.
 * @param max_gap Maximum gap between consecutive active indices (default 10).
 */
inline std::vector<std::size_t> capped_spread_support(int s,
                                                      int p,
                                                      int max_gap = 10)
{
    const int gap = std::min(p / s, max_gap);
    std::vector<std::size_t> sup(static_cast<std::size_t>(s));
    for (Eigen::Index i = 0; i < s; ++i) {
        sup[i] = i * gap;
    }
    return sup;
}


/**
 * @brief One-per-block support: exactly one active variable per block.
 *
 * Places s active predictors with exactly one active per block of size block_size.
 * Block k starts at index k*block_size. If block_size <= 0, auto-compute as p/s.
 * Total active variables returned: min(s, ceil(p/block_size)).
 *
 * Example: s=5, p=1000, block_size=100 (blocks 0–9) → {0, 100, 200, 300, 400}.
 *
 * @param s          Number of desired active predictors.
 * @param p          Total number of predictors.
 * @param block_size Size of each block (0 = auto: p/s).
 */
inline std::vector<std::size_t> one_per_block_support(int s,
                                                      int p,
                                                      int block_size = 0)
{
    if (s <= 0 || p <= 0) return {};

    int g = (block_size > 0) ? block_size : (p / s);
    g = std::max(1, g);

    const int max_blocks = p / g;
    const int active_blocks = std::min(s, max_blocks);

    std::vector<std::size_t> sup;
    sup.reserve(static_cast<std::size_t>(active_blocks));

    for (int b = 0; b < active_blocks; ++b) {
        sup.push_back(static_cast<std::size_t>(b * g));
    }

    return sup;
}


// ==============================================================================
// Support policy enum + factory
// ==============================================================================

/** @brief Configurable support-placement strategy for MC simulations. */
enum class SupportPolicy { CappedSpread, Random, Clustered, OnePerBlock };

/**
 * @brief Human-readable label for a SupportPolicy.
 *
 * @param pol SupportPolicy value.
 */
inline std::string support_policy_label(SupportPolicy pol) {
    switch (pol) {
        // Deterministic policies
        case SupportPolicy::CappedSpread: return "CappedSpread";
        case SupportPolicy::OnePerBlock:  return "OnePerBlock";
        // Stochastic policies
        case SupportPolicy::Random:       return "Random";
        case SupportPolicy::Clustered:    return "Clustered";
    }
    return "Unknown";
}


/**
 * @brief Build support indices according to the chosen policy.
 *
 * Deterministic policies (ignore seed):
 * - CappedSpread: evenly spaced with capped gap.
 * - OnePerBlock: one active per block of given size.
 *
 * Stochastic policies (use seed + 500000 for RNG isolation from DGP noise seed):
 * - Random: uniform random support indices.
 * - Clustered: parameterized cluster support.
 *
 * @param policy            Support placement strategy.
 * @param s                 Number of active predictors.
 * @param p                 Total number of predictors.
 * @param max_gap           Maximum gap for CappedSpread policy (unused for OnePerBlock).
 * @param seed              Trial seed (used only by stochastic policies).
 * @param num_clusters      Number of clusters for Clustered policy (0 = auto).
 * @param max_cluster_size  Maximum cluster size for Clustered policy.
 * @param block_size        Block size for OnePerBlock policy (0 = auto: p/s).
 */
inline std::vector<std::size_t> make_support(SupportPolicy policy,
                                             int s,
                                             int p,
                                             int max_gap,
                                             unsigned seed,
                                             int num_clusters = 0,
                                             int max_cluster_size = 3,
                                             int block_size = 0) {
    switch (policy) {
        // Deterministic policies
        case SupportPolicy::CappedSpread: {
            return capped_spread_support(s, p, max_gap);
        }

        case SupportPolicy::OnePerBlock: {
            return one_per_block_support(s, p, block_size);
        }

        // Stochastic policies
        case SupportPolicy::Random: {
            std::mt19937 rng(seed + 500000u);
            return random_support(s, p, rng);
        }

        case SupportPolicy::Clustered: {
            std::mt19937 rng(seed + 500000u);
            return clustered_support(s, p, num_clusters,
                                     max_cluster_size, rng);
        }
    }

    return capped_spread_support(s, p, max_gap);
}



// ==============================================================================
// Default control parameter factories
// ==============================================================================

/** @brief T-Rex control parameters. */
inline TRexControlParameter make_trex_control(std::size_t K = 20) {
    TRexControlParameter ctrl;
    ctrl.K                        = K;
    ctrl.max_dummy_multiplier     = 10;
    ctrl.use_max_T_stop           = true;
    ctrl.dummy_distribution       = dummygen::Distribution::Normal();
    ctrl.lloop_strategy           = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop    = true;
    ctrl.tloop_max_stagnant_steps = 5;
    ctrl.use_memory_mapping       = false;
    return ctrl;
}

/** @brief DA binary tree (BT) control parameters. */
inline TRexDAControlParameter make_da_bt_control() {
    TRexDAControlParameter da;
    da.method     = DAMethod::BT;
    da.hc_linkage = hac::LinkageMethod::Average;
    return da;
}


// ==============================================================================
// Standard MC inner loop
// ==============================================================================

/** @brief Monte Carlo trial result. */
struct MCTrialResult {
    double fdp = 0.0;
    double tpp = 0.0;
    double L   = 0.0;
    double T   = 0.0;
};

/** @brief Aggregate (FDR, TPR, avg L, avg T, sd FDR, sd TPR) for each grid point. */
struct GridPointResult {
    double avg_fdr = 0.0;
    double avg_tpr = 0.0;
    double avg_L   = 0.0;
    double avg_T   = 0.0;
    double sd_fdr  = 0.0;
    double sd_tpr  = 0.0;
};

using DGPFactory = std::function<DGPData(unsigned seed)>;

/**
 * @brief Run num_MC parallel trials, printing a start and done line.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in the progress line.
 * @param make_data         Lambda (seed) -> DGPData for generating data.
 * @param tFDR              Target FDR for the selector.
 * @param trex_ctrl         TRexControlParameter for the selector.
 * @param da_ctrl           TRexDAControlParameter for the selector.
 * @param base_seed_offset  Base offset for RNG seeds (each trial uses base_seed_offset + mc).
 *
 * @return GridPointResult with average FDR, TPR, L, T, and sd FDR, TPR across trials.
 */
inline GridPointResult run_mc_trials(
    std::size_t num_MC,
    const std::string& progress_label,
    const DGPFactory& make_data,
    double tFDR,
    const TRexControlParameter& trex_ctrl,
    const TRexDAControlParameter& da_ctrl,
    unsigned base_seed_offset)
{
    const int iMC = static_cast<int>(num_MC);

    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);
    std::vector<double> L_vec(num_MC, 0.0);
    std::vector<double> T_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                             dat.X.rows(),
                                             dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(X_map,
                                y_map,
                                tFDR,
                                da_ctrl,
                                trex_ctrl,
                                -1,
                                false);
        selector.select();

        auto sel_idx = selector.getSelectedIndices();
        fdp_vec[mc] = rates::compute_fdp(sel_idx, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel_idx, dat.true_support);
        L_vec[mc]   = static_cast<double>(selector.getDummyMultiplierL());
        T_vec[mc]   = static_cast<double>(selector.getTStop());
    }

    // Compute mean
    const double dMC = static_cast<double>(num_MC);
    double avg_fdr = 0.0, avg_tpr = 0.0, avg_L = 0.0, avg_T = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_fdr += fdp_vec[mc];
        avg_tpr += tpp_vec[mc];
        avg_L   += L_vec[mc];
        avg_T   += T_vec[mc];
    }
    avg_fdr /= dMC;  avg_tpr /= dMC;  avg_L /= dMC;  avg_T /= dMC;

    // Compute Bessel-corrected sd (sd = 0 if num_MC <= 1)
    double sd_fdr = 0.0, sd_tpr = 0.0;
    if (num_MC > 1) {
        double sum_sq_fdr = 0.0, sum_sq_tpr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            sum_sq_fdr += (fdp_vec[mc] - avg_fdr) * (fdp_vec[mc] - avg_fdr);
            sum_sq_tpr += (tpp_vec[mc] - avg_tpr) * (tpp_vec[mc] - avg_tpr);
        }
        sd_fdr = std::sqrt(sum_sq_fdr / (dMC - 1.0));
        sd_tpr = std::sqrt(sum_sq_tpr / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " \u2014 done. TPP=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDP=" << avg_fdr << "\n\n" << std::flush;

    return {avg_fdr, avg_tpr,
            avg_L, avg_T,
            sd_fdr, sd_tpr};
}


// ==============================================================================
// Base T-Rex MC inner loop (no DA correction)
// ==============================================================================

/**
 * @brief Run num_MC parallel trials using base TRexSelector (no DA).
 *
 * Provides a baseline comparison against DA-TRex.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in the progress line.
 * @param make_data         Lambda (seed) -> DGPData for generating data.
 * @param tFDR              Target FDR for the selector.
 * @param trex_ctrl         TRexControlParameter for the selector.
 * @param base_seed_offset  Base offset for RNG seeds.
 *
 * @return GridPointResult with average FDR, TPR, L, T, and sd FDR, TPR across trials.
 */
inline GridPointResult run_mc_trials_base(
    std::size_t num_MC,
    const std::string& progress_label,
    const DGPFactory& make_data,
    double tFDR,
    const TRexControlParameter& trex_ctrl,
    unsigned base_seed_offset)
{
    const int iMC = static_cast<int>(num_MC);

    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);
    std::vector<double> L_vec(num_MC, 0.0);
    std::vector<double> T_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                             dat.X.rows(),
                                             dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexSelector selector(X_map,
                                y_map,
                                tFDR,
                                trex_ctrl,
                                -1,
                                false);
        selector.select();

        auto sel_idx = selector.getSelectedIndices();
        fdp_vec[mc] = rates::compute_fdp(sel_idx, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel_idx, dat.true_support);
        L_vec[mc]   = static_cast<double>(selector.getDummyMultiplierL());
        T_vec[mc]   = static_cast<double>(selector.getTStop());
    }

    // Compute mean
    const double dMC = static_cast<double>(num_MC);
    double avg_fdr = 0.0, avg_tpr = 0.0, avg_L = 0.0, avg_T = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_fdr += fdp_vec[mc];
        avg_tpr += tpp_vec[mc];
        avg_L   += L_vec[mc];
        avg_T   += T_vec[mc];
    }
    avg_fdr /= dMC;  avg_tpr /= dMC;  avg_L /= dMC;  avg_T /= dMC;

    // Compute Bessel-corrected sd (sd = 0 if num_MC <= 1)
    double sd_fdr = 0.0, sd_tpr = 0.0;
    if (num_MC > 1) {
        double sum_sq_fdr = 0.0, sum_sq_tpr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            sum_sq_fdr += (fdp_vec[mc] - avg_fdr) * (fdp_vec[mc] - avg_fdr);
            sum_sq_tpr += (tpp_vec[mc] - avg_tpr) * (tpp_vec[mc] - avg_tpr);
        }
        sd_fdr = std::sqrt(sum_sq_fdr / (dMC - 1.0));
        sd_tpr = std::sqrt(sum_sq_tpr / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " \u2014 done. TPP=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDP=" << avg_fdr << "\n\n" << std::flush;

    return {avg_fdr, avg_tpr,
            avg_L, avg_T,
            sd_fdr, sd_tpr};
}


// ==============================================================================
// Save / print grid results
// ==============================================================================

/**
 * @brief Save a grid-sweep result table to console and file.
 *
 * @param scenario_tag   Short scenario identifier (used in filename).
 * @param grid_label     Column header for the grid dimension (e.g. "SNR", "Rho").
 * @param grid_values    The swept values.
 * @param solvers        Solver list.
 * @param fdr_map        fdr_map[solver_name](grid_idx).
 * @param tpr_map        tpr_map[solver_name](grid_idx).
 * @param sd_fdr_map     sd_fdr_map[solver_name](grid_idx).
 * @param sd_tpr_map     sd_tpr_map[solver_name](grid_idx).
 * @param avg_L_map      Average L results (optional, may be empty).
 * @param avg_T_map      Average T results (optional, may be empty).
 * @param header_extra   Extra info line printed after the main header.
 * @param output_dir     Output directory (default: "simulations/demos/trex_da/").
 */
inline void save_and_print_grid_results(
    const std::string& scenario_tag,
    const std::string& grid_label,
    const std::vector<double>& grid_values,
    std::size_t num_MC,
    const std::vector<SolverInfo>& solvers,
    const std::map<std::string, Eigen::VectorXd>& fdr_map,
    const std::map<std::string, Eigen::VectorXd>& tpr_map,
    const std::map<std::string, Eigen::VectorXd>& sd_fdr_map,
    const std::map<std::string, Eigen::VectorXd>& sd_tpr_map,
    const std::map<std::string, Eigen::VectorXd>& avg_L_map,
    const std::map<std::string, Eigen::VectorXd>& avg_T_map,
    const std::string& header_extra = "",
    const std::string& output_dir   = "simulations/demos/trex_da/")
{
    const std::string filename = "da_trex_mc_" + scenario_tag + ".txt";
    std::ofstream out_file(output_dir + filename);

    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // Header
    std::stringstream hdr;
    hdr << "\n" << std::string(78, '=') << "\n"
        << "  DA-TRex MC: " << scenario_tag
        << "  (MC=" << num_MC << ")\n";
    if (!header_extra.empty())
        hdr << "  " << header_extra << "\n";
    hdr << std::string(78, '=') << "\n\n";
    print_dual(hdr.str());

    // Table dimensions
    constexpr std::size_t sw = 16, mw = 8, cw = 10;
    const auto ncols = grid_values.size();

    // Column header
    std::stringstream th;
    th << std::left  << std::setw(sw) << "Solver"
       << std::left  << std::setw(mw) << "Metric"
       << std::right << std::setw(5)  << grid_label;
    for (double v : grid_values) {
        if (v == std::floor(v) && std::abs(v) < 1e6)
            th << std::setw(cw) << static_cast<int>(v);
        else
            th << std::fixed << std::setprecision(2) << std::setw(cw) << v;
    }
    th << "\n" << std::string(sw + mw + 5 + cw * ncols, '-') << "\n";
    print_dual(th.str());

    // Row printer
    auto print_row = [&](const std::string& solver, const std::string& metric,
                         const Eigen::VectorXd& data, bool first) {
        std::stringstream row;
        row << std::left  << std::setw(sw) << (first ? solver : "")
            << std::left  << std::setw(mw) << metric
            << std::setw(5) << ""
            << std::right;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(ncols); ++i)
            row << std::fixed << std::setprecision(4) << std::setw(cw) << data(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& sv : solvers) {
        const auto& nm = sv.name;
        print_row(nm, "FDR",    fdr_map.at(nm),    true);
        print_row(nm, "sd_FDR", sd_fdr_map.at(nm), false);
        print_row(nm, "TPR",    tpr_map.at(nm),     false);
        print_row(nm, "sd_TPR", sd_tpr_map.at(nm),  false);
        if (avg_L_map.count(nm)) {
            print_row(nm, "Avg L", avg_L_map.at(nm), false);
        }
        if (avg_T_map.count(nm)) {
            print_row(nm, "Avg T", avg_T_map.at(nm), false);
        }
        print_dual("\n");
    }

    print_dual(std::string(78, '=') + "\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] Saved to: " << output_dir + filename << "\n\n";
        out_file.close();
    }
}


// ==============================================================================
// Single-run evaluation
// ==============================================================================

/** @brief Metrics from a single selector run. */
struct SingleRunMetrics {
    std::string name;
    int  n_selected = 0;
    int  TP         = 0;
    int  FP         = 0;
    double TPP      = 0.0;
    double FDP      = 0.0;
};


/**
 * @brief Evaluate selection against true support.
 *  @param name          Name of the selector.
 *  @param selected      Indices of selected features.
 *  @param true_support  Indices of true support features.
 *
 *  @return SingleRunMetrics containing evaluation results.
 */
inline SingleRunMetrics evaluate_selection(
    const std::string& name,
    const std::vector<std::size_t>& selected,
    const std::vector<std::size_t>& true_support)
{
    double fdp = rates::compute_fdp(selected, true_support);
    double tpp = rates::compute_tpp(selected, true_support);

    int n_sel = static_cast<int>(selected.size());
    int tp    = static_cast<int>(
        std::round(tpp * static_cast<double>(true_support.size())));
    int fp    = n_sel - tp;

    return {name, n_sel, tp, fp, tpp, fdp};
}


/**
 * @brief Print a single-run result to console.
 *
 * @param m     SingleRunMetrics to print.
 * @param tFDR  Target FDR.
 */
inline void print_single_result(const SingleRunMetrics& m, double tFDR) {
    std::cout << "\n" << std::string(60, '-') << "\n"
              << "  " << m.name << "\n"
              << std::string(60, '-') << "\n"
              << std::fixed
              << "  Selection:   " << m.n_selected << " selected  |  TP = " << m.TP
              << "  FP = " << m.FP << "\n"
              << "               TPP = " << std::setprecision(2) << m.TPP
              << "  |  FDP = " << m.FDP
              << "  (target <= " << tFDR << ")\n"
              << std::string(60, '-') << "\n";
}


// ==============================================================================
// Generic parameter sweep
// ==============================================================================

/**
 * @brief Run an MC parameter sweep: for each solver x grid value, run num_MC trials.
 *
 * @param scenario_tag      Label for output file and header.
 * @param param_label       Column header for the swept parameter (e.g. "SNR", "Rho").
 * @param grid_values       Parameter grid to sweep.
 * @param num_MC            Number of MC trials per grid point.
 * @param tFDR              Target FDR.
 * @param base_seed         Base RNG seed.
 * @param solvers           Solver list.
 * @param make_dgp          Callable (param_value, seed) -> DGPData.
 * @param da_ctrl           DA control parameters.
 * @param trex_ctrl_base    Base T-Rex control parameters.
 * @param header_extra      Extra info for the output header.
 * @param include_base_trex If true, also run base T-Rex (no DA) with TLARS as a comparison row.
 */
template <typename MakeDGP>
void run_param_sweep(
    const std::string& scenario_tag,
    const std::string& param_label,
    const std::vector<double>& grid_values,
    std::size_t num_MC,
    double tFDR,
    int base_seed,
    const std::vector<SolverInfo>& solvers,
    MakeDGP make_dgp,
    const TRexDAControlParameter& da_ctrl,
    const TRexControlParameter& trex_ctrl_base,
    const std::string& header_extra = "",
    bool include_base_trex = false)
{
    cdianostics::print_section_header("MC: " + scenario_tag);

    const auto ngrid = static_cast<Eigen::Index>(grid_values.size());
    std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map,
                                           sd_fdr_map, sd_tpr_map,
                                           avg_L_map, avg_T_map;
    for (const auto& sv : solvers) {
        fdr_map[sv.name]    = Eigen::VectorXd::Zero(ngrid);
        tpr_map[sv.name]    = Eigen::VectorXd::Zero(ngrid);
        sd_fdr_map[sv.name] = Eigen::VectorXd::Zero(ngrid);
        sd_tpr_map[sv.name] = Eigen::VectorXd::Zero(ngrid);
        avg_L_map[sv.name]  = Eigen::VectorXd::Zero(ngrid);
        avg_T_map[sv.name]  = Eigen::VectorXd::Zero(ngrid);
    }

    auto trex_ctrl = trex_ctrl_base;

    for (const auto& sv : solvers) {
        std::cout << "\n  Solver: " << sv.name << "\n";
        trex_ctrl.solver_type           = sv.type;
        trex_ctrl.solver_params.rho_afs = sv.rho_afs;
        trex_ctrl.tloop_stagnation_stop = sv.use_stagnation;

        for (std::size_t gi = 0; gi < grid_values.size(); ++gi) {
            const double val = grid_values[gi];

            std::stringstream lbl;
            lbl << param_label << " = "
                << std::fixed << std::setprecision(2) << val;

            auto dgp_factory = [&](unsigned seed) -> DGPData {
                return make_dgp(val, seed);
            };

            unsigned base_offset = static_cast<unsigned>(base_seed)
                                 + static_cast<unsigned>(gi) * 10000u;

            auto res = run_mc_trials(
                num_MC, lbl.str(), dgp_factory,
                tFDR, trex_ctrl, da_ctrl, base_offset);

            auto idx = static_cast<Eigen::Index>(gi);
            fdr_map[sv.name](idx)    = res.avg_fdr;
            tpr_map[sv.name](idx)    = res.avg_tpr;
            sd_fdr_map[sv.name](idx) = res.sd_fdr;
            sd_tpr_map[sv.name](idx) = res.sd_tpr;
            avg_L_map[sv.name](idx)  = res.avg_L;
            avg_T_map[sv.name](idx)  = res.avg_T;
        }
    }

    // --- Optional: run base T-Rex (no DA) for each solver as comparison ---
    std::vector<SolverInfo> all_solvers = solvers;
    if (include_base_trex) {
        for (const auto& sv : solvers) {
            // Derive base name: "TREX-DA: TLARS" -> "TREX: TLARS"
            std::string base_name = sv.name;
            std::string trex_prefix = "TREX: ";
            const std::string da_prefix = "TREX-DA: ";
            if (base_name.rfind(da_prefix, 0) == 0) {
                base_name = trex_prefix.append(base_name.substr(da_prefix.size()));
            } else {
                base_name = trex_prefix.append(base_name);
            }

            all_solvers.push_back({sv.type, base_name, sv.rho_afs,
                                      sv.use_stagnation});

            fdr_map[base_name]    = Eigen::VectorXd::Zero(ngrid);
            tpr_map[base_name]    = Eigen::VectorXd::Zero(ngrid);
            sd_fdr_map[base_name] = Eigen::VectorXd::Zero(ngrid);
            sd_tpr_map[base_name] = Eigen::VectorXd::Zero(ngrid);
            avg_L_map[base_name]  = Eigen::VectorXd::Zero(ngrid);
            avg_T_map[base_name]  = Eigen::VectorXd::Zero(ngrid);

            auto base_ctrl = trex_ctrl_base;
            base_ctrl.solver_type           = sv.type;
            base_ctrl.solver_params.rho_afs = sv.rho_afs;
            base_ctrl.tloop_stagnation_stop = sv.use_stagnation;

            std::cout << "\n  Solver: " << base_name << " (no DA)\n";
            for (std::size_t gi = 0; gi < grid_values.size(); ++gi) {
                const double val = grid_values[gi];
                std::stringstream lbl;
                lbl << param_label << " = "
                    << std::fixed << std::setprecision(2) << val;

                auto dgp_factory = [&](unsigned seed) -> DGPData {
                    return make_dgp(val, seed);
                };

                unsigned base_offset = static_cast<unsigned>(base_seed)
                                     + static_cast<unsigned>(gi) * 10000u;

                auto res = run_mc_trials_base(
                    num_MC, lbl.str(), dgp_factory,
                    tFDR, base_ctrl, base_offset);

                auto idx = static_cast<Eigen::Index>(gi);
                fdr_map[base_name](idx)    = res.avg_fdr;
                tpr_map[base_name](idx)    = res.avg_tpr;
                sd_fdr_map[base_name](idx) = res.sd_fdr;
                sd_tpr_map[base_name](idx) = res.sd_tpr;
                avg_L_map[base_name](idx)  = res.avg_L;
                avg_T_map[base_name](idx)  = res.avg_T;
            }
        }
    }

    save_and_print_grid_results(
        scenario_tag, param_label, grid_values, num_MC,
        all_solvers, fdr_map, tpr_map, sd_fdr_map, sd_tpr_map,
        avg_L_map, avg_T_map, header_extra);
}


// ==============================================================================
// SNR sweeper
// ==============================================================================

/**
 * @brief Run an MC SNR sweep: for each solver x SNR value, run num_MC trials.
 *
 * Convenience wrapper around run_param_sweep with param_label = "SNR".
 *
 * @param scenario_tag      Label for output file and header.
 * @param snr_values        SNR values to sweep.
 * @param num_MC            Number of MC trials per SNR value.
 * @param tFDR              Target FDR.
 * @param base_seed         Base RNG seed.
 * @param solvers           Solver list.
 * @param make_dgp          Callable (snr_value, seed) -> DGPData.
 * @param da_ctrl           DA control parameters.
 * @param trex_ctrl_base    Base T-Rex control parameters.
 * @param header_extra      Extra info for the output header.
 * @param include_base_trex If true, also run base T-Rex (no DA) as a comparison.
 */
template <typename MakeDGP>
void run_snr_sweep(
    const std::string& scenario_tag,
    const std::vector<double>& snr_values,
    std::size_t num_MC,
    double tFDR,
    int base_seed,
    const std::vector<SolverInfo>& solvers,
    MakeDGP make_dgp,
    const TRexDAControlParameter& da_ctrl,
    const TRexControlParameter& trex_ctrl_base,
    const std::string& header_extra = "",
    bool include_base_trex = false)
{
    run_param_sweep(scenario_tag, "SNR", snr_values, num_MC, tFDR,
                    base_seed, solvers, std::move(make_dgp), da_ctrl,
                    trex_ctrl_base, header_extra, include_base_trex);
}


// ==============================================================================
// Default solver set
// ==============================================================================

/** @brief Default solver configurations for T-Rex DA. */
inline std::vector<SolverInfo> default_solvers() {
    return {
        {SolverTypeForTRex::TLARS, "TREX-DA: TLARS", 0.0,
            false},
        {SolverTypeForTRex::TAFS,  "TREX-DA: TAFS",  0.3,
            true},
        {SolverTypeForTRex::TOMP,  "TREX-DA: TOMP",  0.0,
            true},
    };
}

// ==============================================================================
} /* End of namespace da_sim */
// ==============================================================================

#endif /* End of DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP */
