// ==============================================================================
// validation_trex_gvs_01_scaling_solver_comparison.cpp
// ==============================================================================
/**
 * @file validation_trex_gvs_01_scaling_solver_comparison.cpp
 *
 * @brief Diagnose the GVS / EN scaling bug: L2 vs z-score x TENET vs TENET_AUG.
 *
 * @details
 * This diagnostic has two parts:
 *
 *   Part 1:
 *     Check whether hierarchical clustering labels are unchanged under
 *     ScalingMode::L2 versus ScalingMode::ZSCORE on one fixed Hastie dataset.
 *
 *   Part 2:
 *     Run a Monte Carlo 2x2 grid over
 *       {L2, ZSCORE} x {TENET, TENET_AUG}
 *     and report mean FDR, mean TPR, lambda2_used, T_stop, num_dummies,
 *     and M_found.
 *
 * The goal is to separate a possible clustering failure from a solver/scaling
 * interaction in the EN branch of T-Rex+GVS.
 */
// ==============================================================================

// project includes
#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// std includes
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ==============================================================================

using namespace gvs_demo;

namespace dn = trex::trex_selector_methods::utils::data_normalizer;
using dn::ScalingMode;

// =============================================================================
// Local result type
// =============================================================================

struct GVSScalingRun {
    std::vector<std::size_t> selected;
    double fdp = 0.0;
    double tpp = 0.0;
    double lambda2_used = 0.0;
    std::size_t T_stop = 0;
    std::size_t num_dummies = 0;
    std::size_t M_found = 0;
    Eigen::VectorXi groups_vec;
};

// =============================================================================
// Preset
// =============================================================================

struct ScalingBugPreset {
    int n = 200;
    int p = 500;
    double sd_x = std::sqrt(0.01);
    double tFDR = 0.10;
    std::size_t K = 20;
    std::size_t num_MC = 100;
    double snr = 2.0;
    double corr_max = 0.98;
    int base_seed = 2026;
};

struct ScalingSolverCell {
    ScalingMode scaling;
    ENSolverType solver;
};

struct ScalingSolverAgg {
    double mean_fdp = 0.0;
    double mean_tpp = 0.0;
    double mean_lambda2 = 0.0;
    double mean_T_stop = 0.0;
    double mean_num_dummies = 0.0;
    double mean_M_found = 0.0;
};

static ScalingBugPreset make_scaling_bug_preset()
{
    return ScalingBugPreset{};
}

// =============================================================================
// Small helpers
// =============================================================================

static std::string scaling_to_string(ScalingMode scaling)
{
    switch (scaling) {
    case ScalingMode::L2:
        return "L2";
    case ScalingMode::ZSCORE:
        return "ZSCORE";
    default:
        return "UNKNOWN";
    }
}

static std::string solver_to_string(ENSolverType solver)
{
    switch (solver) {
    case ENSolverType::TENET:
        return "TENET";
    case ENSolverType::TENET_AUG:
        return "TENET_AUG";
    default:
        return "UNKNOWN";
    }
}

static std::string cell_tag(const ScalingSolverCell& cell)
{
    return scaling_to_string(cell.scaling) + " | " + solver_to_string(cell.solver);
}

static unsigned make_data_seed(const ScalingBugPreset& preset, std::size_t trial)
{
    return static_cast<unsigned>(preset.base_seed + 1000 * static_cast<int>(trial));
}

static int make_selector_seed(unsigned data_seed)
{
    return static_cast<int>(7000 + data_seed);
}

static TRexGVSControlParameter make_en_gvs_ctrl(
    const ScalingBugPreset& preset,
    ENSolverType solver)
{
    TRexGVSControlParameter gvs_ctrl;
    gvs_ctrl.gvs_type = GVSType::EN;
    gvs_ctrl.en_solver = solver;
    gvs_ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
    gvs_ctrl.corr_max = preset.corr_max;
    gvs_ctrl.hc_linkage = hac::LinkageMethod::Single;
    return gvs_ctrl;
}

static GVSDGPResult make_hastie_data(
    const ScalingBugPreset& preset,
    unsigned seed)
{
    return make_hastie_dgp(
        preset.n,
        preset.p,
        preset.snr,
        preset.sd_x,
        seed);
}

// =============================================================================
// Local run helper
// =============================================================================

static GVSScalingRun run_gvs_scaled(
    const GVSDGPResult& dat,
    double tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    TRexControlParameter trex_ctrl,
    ScalingMode scaling,
    int seed)
{
    Eigen::MatrixXd X = dat.X;
    Eigen::VectorXd y = dat.y;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    trex_ctrl.scaling_mode = scaling;

    TRexGVSControlParameter trex_gvs_ctrl = gvs_ctrl;
    trex_gvs_ctrl.trex_ctrl = trex_ctrl;

    TRexGVSSelector selector(
        X_map, y_map, tFDR,
        trex_gvs_ctrl,
        seed, /*verbose=*/false);
    selector.select();

    const auto& gvs_res = selector.getGVSResult();
    const auto& sel_idx = selector.getSelectedIndices();

    GVSScalingRun out;
    out.selected = sel_idx;
    out.fdp = rates::compute_fdp(sel_idx, dat.true_support);
    out.tpp = rates::compute_tpp(sel_idx, dat.true_support);
    out.lambda2_used = gvs_res.lambda2_used;
    out.T_stop = gvs_res.T_stop;
    out.num_dummies = gvs_res.num_dummies;
    out.M_found = gvs_res.max_clusters;
    out.groups_vec = gvs_res.groups_vec;
    return out;
}

// =============================================================================
// Part 1
// =============================================================================

static void run_part1_clustering_integrity(const ScalingBugPreset& preset)
{
    cdiag::print_section_header(
        "Part 1: Clustering integrity (correlation distance is scale-invariant)"
        "\nn=" + std::to_string(preset.n) +
        " p=" + std::to_string(preset.p) +
        " corr_max=" + std::to_string(preset.corr_max).substr(0, 4));

    const GVSDGPResult dat =
        make_hastie_data(preset, static_cast<unsigned>(preset.base_seed));

    const auto gvs_ctrl = make_en_gvs_ctrl(preset, ENSolverType::TENET_AUG);
    const auto trex_ctrl = make_gvs_trex_control(preset.K);
    const int seed = preset.base_seed;

    const auto run_l2 = run_gvs_scaled(
        dat, preset.tFDR, gvs_ctrl, trex_ctrl, ScalingMode::L2, seed);

    const auto run_zs = run_gvs_scaled(
        dat, preset.tFDR, gvs_ctrl, trex_ctrl, ScalingMode::ZSCORE, seed);

    const bool labels_match =
        (run_l2.groups_vec.size() == run_zs.groups_vec.size()) &&
        (run_l2.groups_vec == run_zs.groups_vec);

    std::size_t M = 0;
    std::size_t max_size = 0;

    const auto& g = run_l2.groups_vec;
    const int gmax = g.size() ? g.maxCoeff() : -1;
    std::vector<std::size_t> sizes(static_cast<std::size_t>(gmax + 1), 0);

    for (Eigen::Index i = 0; i < g.size(); ++i) {
        sizes[static_cast<std::size_t>(g[i])]++;
    }

    for (std::size_t s : sizes) {
        if (s > 0) {
            ++M;
            max_size = std::max(max_size, s);
        }
    }

    std::cout << " clusters (L2)      : " << run_l2.M_found << "\n";
    std::cout << " clusters (z-score) : " << run_zs.M_found << "\n";
    std::cout << " distinct labels    : " << M
              << " (largest cluster = " << max_size << " vars)\n";
    std::cout << " groups_vec identical : "
              << (labels_match
                  ? "YES -> clustering NOT corrupted by scaling"
                  : "NO -> clustering DIFFERS (investigate!)")
              << "\n";
    std::cout << " non-degenerate       : "
              << ((M > 1 && M < static_cast<std::size_t>(preset.p)) ? "YES" : "NO")
              << "\n\n";
}

// =============================================================================
// Part 2
// =============================================================================

static ScalingSolverAgg run_part2_cell_mc(
    const ScalingBugPreset& preset,
    const ScalingSolverCell& cell)
{
    const auto gvs_ctrl = make_en_gvs_ctrl(preset, cell.solver);
    const auto trex_ctrl = make_gvs_trex_control(preset.K);

    double sum_fdp = 0.0;
    double sum_tpp = 0.0;
    double sum_lambda2 = 0.0;
    double sum_T = 0.0;
    double sum_dummies = 0.0;
    double sum_M = 0.0;

    for (std::size_t t = 0; t < preset.num_MC; ++t) {
        const unsigned data_seed = make_data_seed(preset, t);
        const GVSDGPResult dat = make_hastie_data(preset, data_seed);

        const int selector_seed = make_selector_seed(data_seed);

        const auto r = run_gvs_scaled(
            dat,
            preset.tFDR,
            gvs_ctrl,
            trex_ctrl,
            cell.scaling,
            selector_seed);

        sum_fdp += r.fdp;
        sum_tpp += r.tpp;
        sum_lambda2 += r.lambda2_used;
        sum_T += static_cast<double>(r.T_stop);
        sum_dummies += static_cast<double>(r.num_dummies);
        sum_M += static_cast<double>(r.M_found);
    }

    const double inv = 1.0 / static_cast<double>(preset.num_MC);

    ScalingSolverAgg out;
    out.mean_fdp = sum_fdp * inv;
    out.mean_tpp = sum_tpp * inv;
    out.mean_lambda2 = sum_lambda2 * inv;
    out.mean_T_stop = sum_T * inv;
    out.mean_num_dummies = sum_dummies * inv;
    out.mean_M_found = sum_M * inv;
    return out;
}

static void run_part2_scaling_solver_grid(const ScalingBugPreset& preset)
{
    cdiag::print_section_header(
        "Part 2: Scaling x Solver FDR grid (MC=" +
        std::to_string(preset.num_MC) +
        ", tFDR=" + std::to_string(preset.tFDR).substr(0, 3) + ")"
        "\nExpect: L2 controlled for both solvers; z-score over-selects.");

    const std::vector<ScalingSolverCell> cells = {
        {ScalingMode::L2,     ENSolverType::TENET},
        {ScalingMode::L2,     ENSolverType::TENET_AUG},
        {ScalingMode::ZSCORE, ENSolverType::TENET},
        {ScalingMode::ZSCORE, ENSolverType::TENET_AUG}
    };

    std::cout << std::left << std::setw(20) << " scaling | solver"
              << std::right
              << std::setw(10) << "mean_FDR"
              << std::setw(10) << "mean_TPR"
              << std::setw(12) << "lambda2"
              << std::setw(9)  << "T_stop"
              << std::setw(9)  << "dummies"
              << std::setw(9)  << "M"
              << "\n";
    std::cout << " " << std::string(75, '-') << "\n";

    for (const auto& cell : cells) {
        const auto agg = run_part2_cell_mc(preset, cell);

        std::cout << std::left << " " << std::setw(18) << cell_tag(cell)
                  << std::right << std::fixed
                  << std::setw(10) << std::setprecision(3) << agg.mean_fdp
                  << std::setw(10) << std::setprecision(3) << agg.mean_tpp
                  << std::setw(12) << std::setprecision(2) << agg.mean_lambda2
                  << std::setw(9)  << std::setprecision(1) << agg.mean_T_stop
                  << std::setw(9)  << std::setprecision(1) << agg.mean_num_dummies
                  << std::setw(9)  << std::setprecision(1) << agg.mean_M_found
                  << "\n";
    }

    std::cout << "\n";
}

// =============================================================================
// Driver
// =============================================================================

static void run_scaling_solver_bug_hunt(const ScalingBugPreset& preset)
{
    cdiag::print_section_header(
        "validation_trex_gvs_01 : GVS/EN scaling bug hunt (L2 vs z-score)"
        "\nR ref (n=200,p=500,rho~0.99,SNR=2): EN mean_FDP=0.1174, TPP=1.000");

    run_part1_clustering_integrity(preset);
    run_part2_scaling_solver_grid(preset);

    std::cout << "Done.\n";
}

// =============================================================================
// main
// =============================================================================

int main()
{
    run_scaling_solver_bug_hunt(make_scaling_bug_preset());
    return 0;
}
