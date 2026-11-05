// ==============================================================================
// validation_trex_spca_02_solver_comparison.cpp
// ==============================================================================
/**
 * @file validation_trex_spca_02_solver_comparison.cpp
 *
 * @brief Validate that the two EN solver variants of T-Rex SPCA agree.
 *
 * @details
 *  The per-PC GVS(EN) sub-selector can use either of two mathematically
 *  equivalent solvers (for lambda2 > 0), now selectable via
 *  TRexSPCAControlParameter::en_solver:
 *
 *    - ENSolverType::TENET     — Gram-based EN (Cholesky update on
 *                                X_A^T X_A + lambda2*I).
 *    - ENSolverType::TENET_AUG — augmented-LASSO EN (default).
 *
 *  Two checks are performed on identical data / identical seeds:
 *
 *    Part A — Per-trial exact equivalence.
 *      For a handful of seeds at each SNR, the same centered X is fed to both
 *      solvers and the resulting active sets (per PC) are compared element by
 *      element. If the solvers are equivalent the supports must be identical.
 *
 *    Part B — Monte Carlo FDR / TPR side-by-side.
 *      The SNR sweep is run with both solvers; averaged FDR / TPR / PEV are
 *      tabulated next to each other with their differences.
 *
 *  Historical note: an earlier version of this comment suggested switching the
 *  column scaling from L2-norm to z-scaling as a next debugging step for the
 *  -10 dB FDR overshoot. That path is now known to be a dead end — z-scoring
 *  the DEMO pipeline's X destroys the factor amplitude signal and inflates the
 *  FDR to ~0.5 (see validation_trex_spca_06_handrolled_comparison), and the
 *  selector-internal scaling choice (L2 vs z-score) is immaterial on centered
 *  X. The remaining small overshoot (~0.03 vs the CRAN R reference) is probed
 *  by the lambda2 / rdump programs (01, 04, 05).
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
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

// Demo utilities (transitively includes trex_spca.hpp, pca.hpp, utils_openmp.hpp)
#include "trex_spca_sim_utils.hpp"

// ==============================================================================

using namespace spca_sim;

// ==============================================================================
// Helpers
// ==============================================================================

/** @brief Run a single TRexSPCA fit and return the per-PC active sets.
 *
 *  X_centered is taken by value so each solver receives an identical, private
 *  copy (TRexSPCA centers / restores in place).
 */
static std::vector<std::vector<Eigen::Index>>
run_single_spca(Eigen::MatrixXd X_centered,
                Eigen::Index    M,
                double          tFDR,
                unsigned        seed,
                ENSolverType    en_solver)
{
    TRexSPCAControlParameter ctrl;
    ctrl.mode                    = SPCAMode::ActiveSet;
    ctrl.en_solver               = en_solver;
    ctrl.gvs_ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
    ctrl.gvs_ctrl.lambda_2       = -1.0;  // < 0: auto-compute via CV (0 would mean no ridge)

    Eigen::Map<Eigen::MatrixXd> X_map(
        X_centered.data(), X_centered.rows(), X_centered.cols());
    TRexSPCA spca(X_map, M, tFDR, ctrl, static_cast<int>(seed));
    return spca.select().active_sets;
}


/** @brief Compare two per-PC active-set collections for set equality. */
static bool active_sets_equal(
    const std::vector<std::vector<Eigen::Index>>& a,
    const std::vector<std::vector<Eigen::Index>>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t m = 0; m < a.size(); ++m) {
        std::vector<Eigen::Index> sa = a[m];
        std::vector<Eigen::Index> sb = b[m];
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        if (sa != sb) {
            return false;
        }
    }
    return true;
}


/** @brief Format a per-PC active set as "{i,j,k}". */
static std::string format_active_set(const std::vector<Eigen::Index>& s)
{
    std::vector<Eigen::Index> sorted = s;
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream oss;
    oss << "{";
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (i) oss << ",";
        oss << sorted[i];
    }
    oss << "}";
    return oss.str();
}


// ==============================================================================
// Part A — Per-trial exact equivalence
// ==============================================================================

void run_equivalence_check(const SimConfig&           cfg,
                           const std::vector<double>& snr_values,
                           std::size_t                n_seeds)
{
    std::cout << "================================================================\n";
    std::cout << "  Part A — Per-trial TENET vs TENET_AUG exact equivalence\n";
    std::cout << "  (identical centered X and seed; comparing PC active sets)\n";
    std::cout << "================================================================\n\n";

    std::size_t total = 0;
    std::size_t matches = 0;

    for (double snr : snr_values) {
        std::size_t snr_matches = 0;
        std::ostringstream snr_ss;
        snr_ss << std::fixed << std::setprecision(1) << snr;
        std::cout << "  SNR = " << std::setw(6) << snr_ss.str() << " dB : ";

        for (std::size_t s = 0; s < n_seeds; ++s) {
            const unsigned seed =
                static_cast<unsigned>(cfg.base_seed) + static_cast<unsigned>(s) * 7919u;

            auto dat = DataGenerator::generate_sparse_factor_model(
                cfg.n, cfg.p, cfg.p1, cfg.M, snr, cfg.overlap_pool_size, seed);
            dat.X.rowwise() -= dat.X.colwise().mean();   // center once (shared input)

            const auto as_aug   = run_single_spca(dat.X, dat.V.cols(), cfg.tFDR,
                                                  seed, ENSolverType::TENET_AUG);
            const auto as_tenet = run_single_spca(dat.X, dat.V.cols(), cfg.tFDR,
                                                  seed, ENSolverType::TENET);

            ++total;
            if (active_sets_equal(as_aug, as_tenet)) {
                ++matches;
                ++snr_matches;
            } else {
                std::cout << "\n    [MISMATCH] seed=" << seed << "\n";
                for (std::size_t m = 0; m < as_aug.size(); ++m) {
                    std::cout << "      PC" << (m + 1)
                              << "  AUG="   << format_active_set(as_aug[m])
                              << "  TENET=" << format_active_set(as_tenet[m]) << "\n";
                }
                std::cout << "  SNR = " << std::setw(6) << snr_ss.str() << " dB : ";
            }
        }
        std::cout << snr_matches << " / " << n_seeds << " identical\n";
    }

    std::cout << "\n  ----------------------------------------------------------\n";
    std::cout << "  TOTAL: " << matches << " / " << total
              << " trials produced identical active sets\n";
    std::cout << "  ----------------------------------------------------------\n\n";
}


// ==============================================================================
// Part B — Monte Carlo FDR / TPR side-by-side
// ==============================================================================

void run_mc_comparison(const SimConfig&           cfg,
                       const std::vector<double>& snr_values)
{
    std::cout << "================================================================\n";
    std::cout << "  Part B — Monte Carlo FDR / TPR (" << cfg.num_MC
              << " trials, ActiveSet mode)\n";
    std::cout << "================================================================\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  SNR(dB)   FDR_AUG  FDR_TENET   dFDR     TPR_AUG  TPR_TENET   dTPR\n";
    std::cout << "  ---------------------------------------------------------------------\n";

    for (double snr : snr_values) {
        const int      snr_int = static_cast<int>(std::round(snr));
        const unsigned offset  = static_cast<unsigned>(cfg.base_seed + snr_int);

        std::ostringstream lbl_ss;
        lbl_ss << std::fixed << std::setprecision(1) << snr;
        const std::string lbl = "SNR=" + lbl_ss.str();

        const auto make_data = [cfg, snr](unsigned seed) -> FactorModelData {
            return DataGenerator::generate_sparse_factor_model(
                cfg.n, cfg.p, cfg.p1, cfg.M, snr, cfg.overlap_pool_size, seed);
        };

        auto r_aug = run_mc_trials_trex_spca(
            cfg.num_MC, lbl + " [TENET_AUG]", make_data, cfg.tFDR,
            SPCAMode::ActiveSet, offset, cfg.lambda_2, ENSolverType::TENET_AUG);

        auto r_tenet = run_mc_trials_trex_spca(
            cfg.num_MC, lbl + " [TENET]", make_data, cfg.tFDR,
            SPCAMode::ActiveSet, offset, cfg.lambda_2, ENSolverType::TENET);

        std::cout << "  " << std::setw(6) << lbl_ss.str() << "   "
                  << std::setw(7) << r_aug.avg_fdr   << "  "
                  << std::setw(8) << r_tenet.avg_fdr << "  "
                  << std::setw(7) << std::fabs(r_aug.avg_fdr - r_tenet.avg_fdr) << "   "
                  << std::setw(7) << r_aug.avg_tpr   << "  "
                  << std::setw(8) << r_tenet.avg_tpr << "  "
                  << std::setw(7) << std::fabs(r_aug.avg_tpr - r_tenet.avg_tpr) << "\n";
    }
    std::cout << "\n";
}


// ==============================================================================
// Main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    SimConfig cfg;
    cfg.n                 = 50;
    cfg.p                 = 100;
    cfg.p1                = 5;
    cfg.M                 = 3;
    cfg.overlap_pool_size = 30;
    cfg.tFDR              = 0.10;
    cfg.num_MC            = 200;
    cfg.base_seed         = 42;
    cfg.lambda_2          = -1.0; // < 0 triggers k-fold CV for lambda_2 selection (0 = no ridge)

    const std::vector<double> snr_values = {-10.0, -5.0, 0.0, 5.0, 10.0};

    // Part A: a few seeds per SNR for an exact, per-trial equivalence check.
    run_equivalence_check(cfg, snr_values, /*n_seeds=*/8);

    // Part B: full MC FDR/TPR comparison.
    run_mc_comparison(cfg, snr_values);

    return 0;
}
