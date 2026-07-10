// ==============================================================================
// trex_da_sim_common.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP
#define DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP
// ==============================================================================
/**
 * @file trex_da_sim_common.hpp
 *
 * @brief Shared infrastructure for DA-TRex MC simulations.
 *
 * Provides:
 *   - SimConfig                     — simulation parameter struct
 *   - SolverInfo                    — solver descriptor
 *   - Support policies              — Random, CappedSpread, Clustered, OnePerBlock
 *   - make_support()                — support generator factory
 *   - make_trex_control()           — default TRexControlParameter
 *   - make_da_bt_control()          — default DA binary tree (BT) control
 *   - run_mc_trials()               — inner MC loop
 *   - run_snr_sweep()               — full solver × SNR MC sweep
 *   - save_and_print_grid_results() — tabular output to console + file
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
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
// Number formatting
// ==============================================================================

/**
 * @brief Format a double for console/title/label strings: fixed-point with at
 *        most @p max_decimals decimals, trailing zeros (and a trailing '.')
 *        trimmed. E.g. fmt_num(0.05) -> "0.05", fmt_num(2.0) -> "2".
 */
inline std::string fmt_num(double v, int max_decimals = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(max_decimals) << v;
    std::string s = oss.str();
    if (s.find('.') != std::string::npos) {
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}


// ==============================================================================
// Configuration
// ==============================================================================

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
    ctrl.lloop_strategy           = LLoopStrategy::STANDARD;
    ctrl.tloop_stagnation_stop    = true;
    ctrl.tloop_max_stagnant_steps = 7;
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
 * @param verbose           If true, enable selector verbose output (default: false).
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
    unsigned base_seed_offset,
    bool verbose = false)
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
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif

        unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                             dat.X.rows(),
                                             dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        // TRexDAControlParameter nests its own trex_ctrl member.
        TRexDAControlParameter trex_da_ctrl = da_ctrl;
        trex_da_ctrl.trex_ctrl = trex_ctrl;

        TRexDASelector selector(X_map,
                                y_map,
                                tFDR,
                                trex_da_ctrl,
                                -1,
                                verbose);
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
 * @param verbose           If true, enable selector verbose output (default: false).
 *
 * @return GridPointResult with average FDR, TPR, L, T, and sd FDR, TPR across trials.
 */
inline GridPointResult run_mc_trials_base(
    std::size_t num_MC,
    const std::string& progress_label,
    const DGPFactory& make_data,
    double tFDR,
    const TRexControlParameter& trex_ctrl,
    unsigned base_seed_offset,
    bool verbose = false)
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
                                verbose);
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
 * @param output_dir     Output directory (default: DEMO_OUTPUT_DIR).
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
    const std::string& output_dir   = DEMO_OUTPUT_DIR)
{
    std::filesystem::create_directories(output_dir);
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
        std::cout << "[Info] TXT saved to: " << output_dir + filename << "\n";
        out_file.close();
    }

    // CSV — tidy long format: solver,metric,grid_label,grid_value,value
    const std::string csv_filename = "da_trex_mc_" + scenario_tag + ".csv";
    std::ofstream csv_file(output_dir + csv_filename);
    if (csv_file.is_open()) {
        // Sanitize grid_label to a valid CSV column name (replace spaces with _)
        std::string csv_col = grid_label;
        for (char& c : csv_col) if (c == ' ') c = '_';

        csv_file << "solver,metric," << csv_col << ",value\n"
                 << std::fixed << std::setprecision(6);

        for (const auto& sv : solvers) {
            const auto& nm = sv.name;
            for (std::size_t i = 0; i < grid_values.size(); ++i) {
                const double gv = grid_values[i];
                csv_file << nm << ",FDR,"    << gv << "," << fdr_map.at(nm)(static_cast<Eigen::Index>(i))    << "\n";
                csv_file << nm << ",sd_FDR," << gv << "," << sd_fdr_map.at(nm)(static_cast<Eigen::Index>(i)) << "\n";
                csv_file << nm << ",TPR,"    << gv << "," << tpr_map.at(nm)(static_cast<Eigen::Index>(i))    << "\n";
                csv_file << nm << ",sd_TPR," << gv << "," << sd_tpr_map.at(nm)(static_cast<Eigen::Index>(i)) << "\n";
                if (avg_L_map.count(nm))
                    csv_file << nm << ",Avg_L," << gv << "," << avg_L_map.at(nm)(static_cast<Eigen::Index>(i)) << "\n";
                if (avg_T_map.count(nm))
                    csv_file << nm << ",Avg_T," << gv << "," << avg_T_map.at(nm)(static_cast<Eigen::Index>(i)) << "\n";
            }
        }
        std::cout << "[Info] CSV saved to: " << output_dir + csv_filename << "\n\n";
    }
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
 * @param verbose           If true, enable selector verbose output (default: false).
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
    bool include_base_trex = false,
    bool verbose = false)
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
                tFDR, trex_ctrl, da_ctrl, base_offset, verbose);

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
                    tFDR, base_ctrl, base_offset, verbose);

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
 * @param verbose           If true, enable selector verbose output (default: false).
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
    bool include_base_trex = false,
    bool verbose = false)
{
    run_param_sweep(scenario_tag, "SNR", snr_values, num_MC, tFDR,
                    base_seed, solvers, std::move(make_dgp), da_ctrl,
                    trex_ctrl_base, header_extra, include_base_trex, verbose);
}


// ==============================================================================
// Default solver set
// ==============================================================================

/** @brief Default solver configurations for T-Rex DA.
 *
 *  The trailing bool is the per-solver stagnation-stop toggle, mapped onto
 *  TRexControlParameter::tloop_stagnation_stop by run_param_sweep(). LARS-based
 *  solvers (TLARS/TLASSO/TENET/TENET_Aug) have monotone solution paths and can
 *  safely run with stagnation control OFF; greedy solvers (TAFS/TOMP) default
 *  it ON. Flip any flag on demand to compare with/without stagnation control.
 */
inline std::vector<SolverInfo> default_solvers() {
    //          type                       name              rho_afs  use_stagnation
    return {
        {SolverTypeForTRex::TLARS, "TREX-DA: TLARS", 0.0,  /*stagnation=*/false},
        {SolverTypeForTRex::TAFS,  "TREX-DA: TAFS",  0.3,  /*stagnation=*/true},
        {SolverTypeForTRex::TOMP,  "TREX-DA: TOMP",  0.0,  /*stagnation=*/true},
    };
}

// ==============================================================================
// 2D parameter sweep with two full support-type matrices (CappedSpread + Random)
// ==============================================================================

/**
 * @brief Save a 2D parameter sweep (CappedSpread and Random as full matrices)
 *        to console, TXT, and CSV.
 *
 * Both support types produce a full [n_row × n_col] result matrix.
 * Writes one block per solver with four sub-tables per support type:
 * mean_TPP, sd_TPP, mean_FDP, sd_FDP.
 *
 * @param scenario_tag       Short identifier used in filenames.
 * @param row_label          Name of the row parameter (e.g. "Kappa").
 * @param col_label          Name of the column parameter (e.g. "Rho").
 * @param row_grid           Row parameter values.
 * @param col_grid           Column parameter values.
 * @param num_MC             Number of Monte Carlo trials per cell.
 * @param solvers            Ordered solver list.
 * @param tpp_cs_map         mean TPP, CappedSpread: solver → MatrixXd [n_row × n_col].
 * @param fdp_cs_map         mean FDP, CappedSpread.
 * @param sd_tpp_cs_map      sd  TPP, CappedSpread.
 * @param sd_fdp_cs_map      sd  FDP, CappedSpread.
 * @param tpp_rand_map       mean TPP, Random.
 * @param fdp_rand_map       mean FDP, Random.
 * @param sd_tpp_rand_map    sd  TPP, Random.
 * @param sd_fdp_rand_map    sd  FDP, Random.
 * @param header_extra       Extra info line printed after the main header.
 * @param output_dir         Output directory (default: DEMO_OUTPUT_DIR).
 */
inline void save_and_print_2d_two_support_results(
    const std::string&                              scenario_tag,
    const std::string&                              row_label,
    const std::string&                              col_label,
    const std::vector<double>&                      row_grid,
    const std::vector<double>&                      col_grid,
    std::size_t                                     num_MC,
    const std::vector<SolverInfo>&                  solvers,
    const std::map<std::string, Eigen::MatrixXd>&   tpp_cs_map,
    const std::map<std::string, Eigen::MatrixXd>&   fdp_cs_map,
    const std::map<std::string, Eigen::MatrixXd>&   sd_tpp_cs_map,
    const std::map<std::string, Eigen::MatrixXd>&   sd_fdp_cs_map,
    const std::map<std::string, Eigen::MatrixXd>&   tpp_rand_map,
    const std::map<std::string, Eigen::MatrixXd>&   fdp_rand_map,
    const std::map<std::string, Eigen::MatrixXd>&   sd_tpp_rand_map,
    const std::map<std::string, Eigen::MatrixXd>&   sd_fdp_rand_map,
    const std::string&                              header_extra = "",
    const std::string&                              output_dir   = DEMO_OUTPUT_DIR)
{
    std::filesystem::create_directories(output_dir);
    const std::string txt_path = output_dir + "da_trex_mc_" + scenario_tag + ".txt";
    const std::string csv_path = output_dir + "da_trex_mc_" + scenario_tag + ".csv";

    std::ofstream out_file(txt_path);

    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    const auto n_row = row_grid.size();
    const auto n_col = col_grid.size();

    constexpr int rlw = 12;  // row-label width
    constexpr int cw  = 9;   // column width

    // Format a double as integer string if it is a whole number
    auto fmt_val = [](double v) -> std::string {
        if (v == std::floor(v) && std::abs(v) < 1e6) {
            std::ostringstream o; o << static_cast<int>(v); return o.str();
        }
        std::ostringstream o;
        o << std::fixed << std::setprecision(2) << v;
        return o.str();
    };

    // Column header row + separator line
    auto make_col_header = [&]() -> std::string {
        std::ostringstream h;
        h << std::string(rlw, ' ');
        for (double cv : col_grid)
            h << std::right << std::setw(cw) << (col_label + "=" + fmt_val(cv));
        h << "\n" << "  " << std::string(rlw + cw * n_col, '-') << "\n";
        return h.str();
    };

    // Print one labelled matrix (rows = row_grid, cols = col_grid)
    auto print_matrix_block = [&](const std::string& label,
                                  const Eigen::MatrixXd& mat) {
        std::ostringstream block;
        block << "  " << label << "\n" << make_col_header();
        for (std::size_t ir = 0; ir < n_row; ++ir) {
            const std::string rl_str = row_label + "=" + fmt_val(row_grid[ir]);
            block << "  " << std::left << std::setw(rlw) << rl_str;
            for (std::size_t ic = 0; ic < n_col; ++ic)
                block << std::right << std::fixed << std::setprecision(4)
                      << std::setw(cw)
                      << mat(static_cast<Eigen::Index>(ir),
                             static_cast<Eigen::Index>(ic));
            block << "\n";
        }
        block << "\n";
        print_dual(block.str());
    };

    // ── Header ───────────────────────────────────────────────────────────────
    {
        std::ostringstream hdr;
        hdr << "\n" << std::string(78, '=') << "\n"
            << "  DA-TRex MC: " << scenario_tag
            << "  (MC=" << num_MC << ")\n";
        if (!header_extra.empty())
            hdr << "  " << header_extra << "\n";
        hdr << std::string(78, '=') << "\n\n";
        print_dual(hdr.str());
    }

    // ── Per-solver blocks ────────────────────────────────────────────────────
    for (const auto& sv : solvers) {
        const auto& nm = sv.name;
        {
            std::ostringstream sep;
            sep << std::string(78, '-') << "\n"
                << "  Solver: " << nm << "\n"
                << std::string(78, '-') << "\n\n";
            print_dual(sep.str());
        }

        print_dual("  [CappedSpread]\n\n");
        print_matrix_block("mean_TPP", tpp_cs_map.at(nm));
        print_matrix_block("sd_TPP",   sd_tpp_cs_map.at(nm));
        print_matrix_block("mean_FDP", fdp_cs_map.at(nm));
        print_matrix_block("sd_FDP",   sd_fdp_cs_map.at(nm));

        print_dual("  [Random]\n\n");
        print_matrix_block("mean_TPP", tpp_rand_map.at(nm));
        print_matrix_block("sd_TPP",   sd_tpp_rand_map.at(nm));
        print_matrix_block("mean_FDP", fdp_rand_map.at(nm));
        print_matrix_block("sd_FDP",   sd_fdp_rand_map.at(nm));
    }

    print_dual(std::string(78, '=') + "\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] TXT saved to: " << txt_path << "\n";
        out_file.close();
    }

    // ── CSV (tidy long format) ────────────────────────────────────────────────
    std::ofstream csv_file(csv_path);
    if (csv_file.is_open()) {
        auto sanitize = [](std::string s) {
            for (char& c : s) if (c == ' ') c = '_';
            return s;
        };
        const std::string csv_row = sanitize(row_label);
        const std::string csv_col = sanitize(col_label);

        csv_file << "solver,support_type,metric,"
                 << csv_row << "," << csv_col << ",value\n"
                 << std::fixed << std::setprecision(6);

        auto write_block = [&](const std::string& nm,
                                const std::string& support_type,
                                const std::map<std::string, Eigen::MatrixXd>& tpp_m,
                                const std::map<std::string, Eigen::MatrixXd>& fdp_m,
                                const std::map<std::string, Eigen::MatrixXd>& sd_tpp_m,
                                const std::map<std::string, Eigen::MatrixXd>& sd_fdp_m) {
            for (std::size_t ir = 0; ir < n_row; ++ir) {
                for (std::size_t ic = 0; ic < n_col; ++ic) {
                    const double     rv = row_grid[ir];
                    const double     cv = col_grid[ic];
                    const Eigen::Index ri = static_cast<Eigen::Index>(ir);
                    const Eigen::Index ci = static_cast<Eigen::Index>(ic);
                    csv_file << nm << "," << support_type << ",mean_TPP," << rv << "," << cv << ","
                             << tpp_m.at(nm)(ri, ci)    << "\n";
                    csv_file << nm << "," << support_type << ",sd_TPP,"   << rv << "," << cv << ","
                             << sd_tpp_m.at(nm)(ri, ci) << "\n";
                    csv_file << nm << "," << support_type << ",mean_FDP," << rv << "," << cv << ","
                             << fdp_m.at(nm)(ri, ci)    << "\n";
                    csv_file << nm << "," << support_type << ",sd_FDP,"   << rv << "," << cv << ","
                             << sd_fdp_m.at(nm)(ri, ci) << "\n";
                }
            }
        };

        for (const auto& sv : solvers) {
            write_block(sv.name, "CappedSpread",
                        tpp_cs_map,   fdp_cs_map,   sd_tpp_cs_map,   sd_fdp_cs_map);
            write_block(sv.name, "Random",
                        tpp_rand_map, fdp_rand_map, sd_tpp_rand_map, sd_fdp_rand_map);
        }
        std::cout << "[Info] CSV saved to: " << csv_path << "\n\n";
    }
}


// ==============================================================================
// 2D Gap × Rho saving utility
// ==============================================================================

/**
 * @brief Save a 2D Gap × Rho MC sweep result to console, TXT, and CSV.
 *
 * Writes one block per solver containing mean_TPP, sd_TPP, mean_FDP, sd_FDP
 * matrices.  Columns are the gap values (CappedSpread section) plus a final
 * "Random" column; rows are rho values.
 *
 * @param scenario_tag      Short identifier used in the output filenames.
 * @param gap_grid          Gap values swept (columns for CappedSpread section).
 * @param rho_grid          Rho values swept (rows).
 * @param kappa_boundary    kappa = ceil(log(0.02)/log(rho)) per rho entry.
 * @param num_MC            Number of Monte Carlo trials per grid cell.
 * @param solvers           Ordered solver list (determines block order).
 * @param tpp_cs_map        mean TPP, CappedSpread: solver → MatrixXd [n_rho × n_gap].
 * @param fdp_cs_map        mean FDP, CappedSpread.
 * @param sd_tpp_cs_map     sd  TPP, CappedSpread.
 * @param sd_fdp_cs_map     sd  FDP, CappedSpread.
 * @param tpp_rand_map      mean TPP, Random: solver → VectorXd [n_rho].
 * @param fdp_rand_map      mean FDP, Random.
 * @param sd_tpp_rand_map   sd  TPP, Random.
 * @param sd_fdp_rand_map   sd  FDP, Random.
 * @param header_extra      Extra info line printed after the main header.
 * @param output_dir        Output directory (default: DEMO_OUTPUT_DIR).
 */
inline void save_and_print_2d_gap_rho_results(
    const std::string&                              scenario_tag,
    const std::vector<int>&                         gap_grid,
    const std::vector<double>&                      rho_grid,
    const std::vector<int>&                         kappa_boundary,
    std::size_t                                     num_MC,
    const std::vector<SolverInfo>&                  solvers,
    const std::map<std::string, Eigen::MatrixXd>&   tpp_cs_map,
    const std::map<std::string, Eigen::MatrixXd>&   fdp_cs_map,
    const std::map<std::string, Eigen::MatrixXd>&   sd_tpp_cs_map,
    const std::map<std::string, Eigen::MatrixXd>&   sd_fdp_cs_map,
    const std::map<std::string, Eigen::VectorXd>&   tpp_rand_map,
    const std::map<std::string, Eigen::VectorXd>&   fdp_rand_map,
    const std::map<std::string, Eigen::VectorXd>&   sd_tpp_rand_map,
    const std::map<std::string, Eigen::VectorXd>&   sd_fdp_rand_map,
    const std::string&                              header_extra = "",
    const std::string&                              output_dir   = DEMO_OUTPUT_DIR)
{
    std::filesystem::create_directories(output_dir);
    const std::string txt_path = output_dir + "da_trex_mc_" + scenario_tag + ".txt";
    const std::string csv_path = output_dir + "da_trex_mc_" + scenario_tag + ".csv";

    std::ofstream out_file(txt_path);

    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    const auto n_rho = rho_grid.size();
    const auto n_gap = gap_grid.size();
    const int  n_cols = static_cast<int>(n_gap) + 1;   // gap cols + Random

    constexpr int rw = 10;  // row-label width
    constexpr int cw = 9;   // column width

    // Build a column-header + separator string for the gap+Random layout
    auto make_col_header = [&]() -> std::string {
        std::ostringstream h;
        h << std::string(rw, ' ');
        for (int g : gap_grid)
            h << std::right << std::setw(cw) << ("gap=" + std::to_string(g));
        h << std::setw(cw) << "Random" << "\n";
        h << "  " << std::string(rw + cw * n_cols, '-') << "\n";
        return h.str();
    };

    // Print one labelled matrix (cs columns + Random column)
    auto print_matrix_block = [&](const std::string& label,
                                  const Eigen::MatrixXd& cs_mat,
                                  const Eigen::VectorXd& rand_vec) {
        std::ostringstream block;
        block << "  " << label << "\n" << make_col_header();
        for (std::size_t ir = 0; ir < n_rho; ++ir) {
            std::ostringstream rl;
            rl << "rho=" << std::fixed << std::setprecision(1) << rho_grid[ir];
            block << "  " << std::left << std::setw(rw) << rl.str();
            for (std::size_t ig = 0; ig < n_gap; ++ig)
                block << std::right << std::fixed << std::setprecision(4)
                      << std::setw(cw)
                      << cs_mat(static_cast<Eigen::Index>(ir),
                                static_cast<Eigen::Index>(ig));
            block << std::right << std::fixed << std::setprecision(4)
                  << std::setw(cw)
                  << rand_vec(static_cast<Eigen::Index>(ir)) << "\n";
        }
        block << "\n";
        print_dual(block.str());
    };

    // ── Header ──────────────────────────────────────────────────────────────
    {
        std::ostringstream hdr;
        hdr << "\n" << std::string(78, '=') << "\n"
            << "  DA-TRex MC: " << scenario_tag
            << "  (MC=" << num_MC << ")\n";
        if (!header_extra.empty())
            hdr << "  " << header_extra << "\n";
        hdr << std::string(78, '=') << "\n\n";
        print_dual(hdr.str());
    }

    // ── Kappa boundary ───────────────────────────────────────────────────────
    {
        std::ostringstream kb;
        kb << "  DA+AR1 correction window: kappa = ceil(log(0.02) / log(rho))\n\n";
        kb << "  " << std::left << std::setw(10) << "rho";
        for (double r : rho_grid)
            kb << std::right << std::fixed << std::setprecision(1) << std::setw(6) << r;
        kb << "\n";
        kb << "  " << std::left << std::setw(10) << "kappa";
        for (int k : kappa_boundary)
            kb << std::right << std::setw(6) << k;
        kb << "\n\n";
        print_dual(kb.str());
    }

    // ── Per-solver blocks ────────────────────────────────────────────────────
    for (const auto& sv : solvers) {
        const auto& nm = sv.name;
        {
            std::ostringstream sep;
            sep << std::string(78, '-') << "\n"
                << "  Solver: " << nm << "\n"
                << std::string(78, '-') << "\n\n";
            print_dual(sep.str());
        }
        print_matrix_block("mean_TPP", tpp_cs_map.at(nm),    tpp_rand_map.at(nm));
        print_matrix_block("sd_TPP",   sd_tpp_cs_map.at(nm), sd_tpp_rand_map.at(nm));
        print_matrix_block("mean_FDP", fdp_cs_map.at(nm),    fdp_rand_map.at(nm));
        print_matrix_block("sd_FDP",   sd_fdp_cs_map.at(nm), sd_fdp_rand_map.at(nm));
    }

    print_dual(std::string(78, '=') + "\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] TXT saved to: " << txt_path << "\n";
        out_file.close();
    }

    // ── CSV (tidy long format) ────────────────────────────────────────────────
    std::ofstream csv_file(csv_path);
    if (csv_file.is_open()) {
        csv_file << "solver,metric,rho,support_type,gap,value\n"
                 << std::fixed << std::setprecision(6);
        for (const auto& sv : solvers) {
            const auto& nm = sv.name;
            for (std::size_t ir = 0; ir < n_rho; ++ir) {
                const double     rv = rho_grid[ir];
                const Eigen::Index ri = static_cast<Eigen::Index>(ir);

                // CappedSpread entries (one per gap value)
                for (std::size_t ig = 0; ig < n_gap; ++ig) {
                    const Eigen::Index gi = static_cast<Eigen::Index>(ig);
                    const int gv = gap_grid[ig];
                    csv_file << nm << ",mean_TPP," << rv << ",CappedSpread," << gv
                             << "," << tpp_cs_map.at(nm)(ri, gi)    << "\n";
                    csv_file << nm << ",sd_TPP,"   << rv << ",CappedSpread," << gv
                             << "," << sd_tpp_cs_map.at(nm)(ri, gi) << "\n";
                    csv_file << nm << ",mean_FDP," << rv << ",CappedSpread," << gv
                             << "," << fdp_cs_map.at(nm)(ri, gi)    << "\n";
                    csv_file << nm << ",sd_FDP,"   << rv << ",CappedSpread," << gv
                             << "," << sd_fdp_cs_map.at(nm)(ri, gi) << "\n";
                }

                // Random entries (gap column left empty)
                csv_file << nm << ",mean_TPP," << rv << ",Random,,"
                         << tpp_rand_map.at(nm)(ri)    << "\n";
                csv_file << nm << ",sd_TPP,"   << rv << ",Random,,"
                         << sd_tpp_rand_map.at(nm)(ri) << "\n";
                csv_file << nm << ",mean_FDP," << rv << ",Random,,"
                         << fdp_rand_map.at(nm)(ri)    << "\n";
                csv_file << nm << ",sd_FDP,"   << rv << ",Random,,"
                         << sd_fdp_rand_map.at(nm)(ri) << "\n";
            }
        }
        std::cout << "[Info] CSV saved to: " << csv_path << "\n\n";
    }
}

// ==============================================================================
} /* End of namespace da_sim */
// ==============================================================================
#endif /* End of DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP */
