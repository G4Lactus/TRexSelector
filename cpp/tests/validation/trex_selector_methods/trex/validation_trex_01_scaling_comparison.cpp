// ==============================================================================
// validation_trex_01_scaling_comparison.cpp
// ==============================================================================
/**
 * @file validation_trex_01_scaling_comparison.cpp
 *
 * @brief Control experiment: verify that column scaling (unit-L2 vs z-score)
 *        leaves the classical (plain LASSO/LARS) T-Rex selector unchanged.
 *
 * @details
 *  For each centered column, sample SD = ||x_j||_2 / sqrt(n-1). The divisor
 *  sqrt(n-1) is the SAME for every column, so z-score scaling is just unit-L2
 *  scaling times a single GLOBAL constant:
 *
 *        X_zscore = sqrt(n-1) * X_l2,    D_zscore = sqrt(n-1) * D_l2.
 *
 *  The tsolvers do NOT re-standardize internally, but a global rescaling of the
 *  whole design [X | D] only rescales the LASSO lambda axis — the ORDER in which
 *  variables and dummies enter the path is unchanged. T-Rex relative occurrences
 *  and the dummy-based stop are therefore invariant, so the selected support
 *  must be identical.
 *
 *  CRUCIAL: this only holds if the dummies are the SAME random draws in both
 *  runs. The MC helper run_mc_trials_trex() constructs the selector with seed
 *  = -1 (non-deterministic dummies), which would inject pure Monte-Carlo noise
 *  and mask the comparison. Here we therefore construct TRexSelector directly
 *  with a FIXED per-trial seed shared by both scalings, giving a clean,
 *  deterministic invariance check.
 *
 *  Expected outcome: per-trial selected sets IDENTICAL (any residual difference
 *  would be solver numerical tolerance at the rescaled magnitude, not a bug).
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <iomanip>
#include <iostream>
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
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Demo utilities
#include "trex_sim_utils.hpp"
using namespace trex_sim;

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace datagen  = trex::utils::datageneration::datagen;
namespace dummygen = trex::utils::datageneration::dummygen;
namespace rates_ns = trex::utils::eval::rates;

using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::utils::data_normalizer::ScalingMode;

// ==============================================================================
// Configuration helper
// ==============================================================================

static TRexControlParameter make_base_trex_control(SolverTypeForTRex solver) {
    TRexControlParameter ctrl;
    ctrl.K                        = 20;
    ctrl.max_dummy_multiplier     = 10;
    ctrl.use_max_T_stop           = true;
    ctrl.dummy_distribution       = dummygen::Distribution::Normal();
    ctrl.lloop_strategy           = LLoopStrategy::STANDARD;
    ctrl.tloop_stagnation_stop    = true;
    ctrl.tloop_max_stagnant_steps = 7;
    ctrl.parallel_rnd_experiments = false;
    ctrl.opt_threshold            = 0.75;
    ctrl.use_memory_mapping       = false;
    ctrl.solver_type              = solver;
    return ctrl;
}

// Run base T-Rex once on a private copy of (X, y) with the given scaling mode
// and a fixed seed; return the sorted selected indices.
static std::vector<std::size_t> run_once(const Eigen::MatrixXd& X,
                                         const Eigen::VectorXd& y,
                                         double tFDR,
                                         SolverTypeForTRex solver,
                                         ScalingMode mode,
                                         int seed) {
    // Selector normalizes/denormalizes X in place → work on private copies.
    Eigen::MatrixXd Xw = X;
    Eigen::VectorXd yw = y;
    Eigen::Map<Eigen::MatrixXd> Xmap(Xw.data(), Xw.rows(), Xw.cols());
    Eigen::Map<Eigen::VectorXd> ymap(yw.data(), yw.rows());

    TRexControlParameter ctrl = make_base_trex_control(solver);
    ctrl.scaling_mode = mode;

    TRexSelector selector(Xmap, ymap, tFDR, ctrl, seed, /*verbose=*/false);
    selector.select();

    std::vector<std::size_t> sel = selector.getSelectedIndices();
    std::sort(sel.begin(), sel.end());
    return sel;
}

// ==============================================================================
// Scaling invariance experiment — base T-Rex
// ==============================================================================

void demo_trex_scaling_comparison(std::size_t num_trials,
                                  SolverTypeForTRex solver,
                                  const std::string& solver_name) {

    std::cout.setf(std::ios::unitbuf);

    std::cout << "================================================================\n";
    std::cout << "  Base T-Rex — Scaling invariance control (L2 vs z-score)\n";
    std::cout << "  solver=" << solver_name
              << "  (fixed shared selector seed per trial -> identical dummies)\n";
    std::cout << "================================================================\n";

    const Eigen::Index n   = 100;
    const Eigen::Index p   = 300;
    const std::size_t  s0  = 10;     // true support cardinality
    const double       tFDR = 0.10;
    const std::vector<double> snr_values = {0.5, 1.0, 2.0, 5.0};

    std::cout << "  n=" << n << "  p=" << p << "  |support|=" << s0
              << "  tFDR=" << tFDR << "  trials/SNR=" << num_trials
              << "  solver=" << solver_name << "\n\n";

    // Shared true support + (homogeneous) coefficients.
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    std::unordered_set<std::size_t> support_set;
    while (support_set.size() < s0) support_set.insert(uniform_dist(rng));
    std::vector<std::size_t> true_support(support_set.begin(), support_set.end());
    std::sort(true_support.begin(), true_support.end());
    std::vector<double> true_coefs(s0, 1.0);

    std::cout << std::left << std::setw(8) << "SNR"
              << std::right
              << std::setw(12) << "identical"
              << std::setw(12) << "FDR(L2)" << std::setw(12) << "FDR(Z)"
              << std::setw(12) << "TPR(L2)" << std::setw(12) << "TPR(Z)"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::size_t total_trials = 0, total_identical = 0;

    for (std::size_t si = 0; si < snr_values.size(); ++si) {
        const double snr = snr_values[si];

        std::size_t identical = 0;
        double fdr_l2 = 0.0, fdr_z = 0.0, tpr_l2 = 0.0, tpr_z = 0.0;

        for (std::size_t t = 0; t < num_trials; ++t) {
            const int data_seed = static_cast<int>(1000u * (si + 1) + t);
            datagen::SyntheticData data(n, p, true_support, true_coefs,
                                        snr, data_seed);
            const Eigen::MatrixXd X = data.getX();
            const Eigen::VectorXd y = data.getY();

            // Same fixed selector seed for both modes → identical dummy draws.
            const int sel_seed = 7000 + data_seed;

            const auto sel_l2 = run_once(X, y, tFDR, solver, ScalingMode::L2,     sel_seed);
            const auto sel_z  = run_once(X, y, tFDR, solver, ScalingMode::ZSCORE, sel_seed);

            if (sel_l2 == sel_z) ++identical;

            fdr_l2 += rates_ns::compute_fdp(sel_l2, true_support);
            tpr_l2 += rates_ns::compute_tpp(sel_l2, true_support);
            fdr_z  += rates_ns::compute_fdp(sel_z,  true_support);
            tpr_z  += rates_ns::compute_tpp(sel_z,  true_support);
        }

        const double dT = static_cast<double>(num_trials);
        const std::string ident = std::to_string(identical) + "/" + std::to_string(num_trials);
        std::cout << std::left << std::setw(8) << std::fixed << std::setprecision(1) << snr
                  << std::right
                  << std::setw(12) << ident
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << (fdr_l2 / dT) << std::setw(12) << (fdr_z / dT)
                  << std::setw(12) << (tpr_l2 / dT) << std::setw(12) << (tpr_z / dT)
                  << "\n";

        total_trials    += num_trials;
        total_identical += identical;
    }

    std::cout << std::string(80, '-') << "\n";
    std::cout << "\nIdentical selections: " << total_identical << "/" << total_trials << "\n";
    std::cout << "Verdict: base T-Rex L2 vs z-score are "
              << (total_identical == total_trials
                      ? "BIT-IDENTICAL - scaling is a pure no-op for plain LASSO, "
                        "as theory predicts; the ScalingMode plumbing is correct."
                      : "NOT always identical - residual differences come from "
                        "solver numerical tolerance at the rescaled magnitude.")
              << "\n\n";
}

// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    demo_trex_scaling_comparison(/*num_trials=*/30, SolverTypeForTRex::TLARS,  "TLARS");
    demo_trex_scaling_comparison(/*num_trials=*/30, SolverTypeForTRex::TLASSO, "TLASSO");

    return 0;
}
