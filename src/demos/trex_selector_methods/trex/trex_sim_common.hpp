// ==============================================================================
// trex_sim_common.hpp
// ==============================================================================
#ifndef TREX_DEMOS_TRX_SIM_COMMON_HPP
#define TREX_DEMOS_TRX_SIM_COMMON_HPP
// ==============================================================================
/**
 * @file trex_sim_common.hpp
 *
 * @brief Shared infrastructure for classic T-Rex MC simulations.
 *
 * Provides:
 *   - TrexDGPData               — data container (owned X, y, true_support)
 *   - TrexGridPointResult       — MC aggregate (avg/sd FDR, TPR, L, T)
 *   - TrexDGPFactory            — factory callable type: seed → TrexDGPData
 *   - TrexMmapDGPFactory        — mmap factory type: seed, X_path, y_path → TrexDGPData
 *   - run_mc_trials_trex()      — parallel MC loop using in-memory TRexSelector
 *   - run_mc_trials_trex_mmap() — parallel MC loop using memory-mapped X
 *
 * Parallelism pattern (mirrors trex_da_sim_common.hpp):
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

// std includes
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Demo utilities (MmapFileGuard used by callers of run_mc_trials_trex_mmap)
#include "demo_trex_utils.hpp"


// ==============================================================================
// Namespace
// ==============================================================================

namespace trx_sim {

namespace rates = trex::utils::eval::rates;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::trex_core::TRexControlParameter;


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

        TRexSelector selector(X_map, y_map, tFDR, trex_ctrl, -1, false);
        selector.select();

        const auto sel  = selector.getSelectedIndices();
        fdp_vec[mc] = rates::compute_fdp(sel, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel, dat.true_support);
        L_vec[mc]   = static_cast<double>(selector.getDummyMultiplierL());
        T_vec[mc]   = static_cast<double>(selector.getStoppingTimeT());
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
        T_vec[mc]   = static_cast<double>(selector.getStoppingTimeT());
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

} // namespace trx_sim

#endif // TREX_DEMOS_TRX_SIM_COMMON_HPP
// ==============================================================================
