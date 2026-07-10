// ==============================================================================
// trex_spca_sim_utils.hpp
// ==============================================================================
/**
 * @file trex_spca_sim_utils.hpp
 *
 * @brief Shared simulation utilities for T-Rex SPCA demo files.
 *
 * @details Provides all types, data generation, output helpers, and MC loop
 *          infrastructure shared by T-Rex SPCA demo files, all under namespace
 *          spca_sim:
 *
 *   Data generation:
 *    - FactorModelData              — data container (X, Z, V, E, actual_snr_db)
 *    - DataGenerator                — sparse factor model data generator
 *
 *   Configuration:
 *    - SimConfig                    — simulation parameter struct
 *    - SPCADGPFactory               — DGP factory callable type: seed → FactorModelData
 *
 *   Results:
 *    - SPCASingleResult             — per-trial metrics (mean TPR/FDR over M components, PEV)
 *    - SPCAGridPointResult          — MC-aggregated (avg/sd TPR, FDR; avg PEV)
 *
 *   Evaluation:
 *    - evaluate_spca()              — compute SPCASingleResult from V_est, Z_est, V_true
 *
 *   MC loop functions:
 *    - run_mc_trials_trex_spca()    — parallel MC loop for T-Rex SPCA
 *    - run_mc_trials_pca()          — parallel MC loop for ordinary PCA baseline
 *    - run_mc_trials_oracle_pca()   — parallel MC loop for oracle thresholded PCA
 *
 *   Output:
 *    - save_and_print_spca_mc_results() — table + tidy CSV (console + file)
 *
 * Parallelism pattern:
 *   #pragma omp parallel for schedule(dynamic)
 *   for (int mc = 0; mc < iMC; ++mc) { ... }
 *
 * Thread safety:
 *   - Each trial owns its data (returned by-value from the DGP factory).
 *   - DataGenerator::generate_sparse_factor_model() seeds its own RNG from the
 *     trial seed — no shared mutable state.
 *   - PCA and TRexSPCA objects are created per thread (stack-local).
 *   - DGP factory lambdas must capture all parameters by value.
 *
 * @note Demo-internal header — not part of the TRexSelector library interface.
 *       Never include from library code.
 */
// ==============================================================================

#ifndef DEMOS_TREX_SELECTOR_METHODS_TREX_SPCA_SIM_UTILS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_TREX_SPCA_SIM_UTILS_HPP

// std includes
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex SPCA includes
#include <trex_selector_methods/trex_spca/trex_spca.hpp>

// PCA baseline
#include <ml_methods/pca/pca.hpp>


// ==============================================================================
// Namespace
// ==============================================================================

namespace spca_sim {

// Namespace aliases
namespace pca_ns = trex::ml_methods::pca;
using trex::trex_selector_methods::trex_gvs::LambdaSelectionMethod;
using trex::trex_selector_methods::trex_gvs::ENSolverType;
using trex::trex_selector_methods::utils::data_normalizer::ScalingMode;
using trex::trex_selector_methods::trex_spca::TRexSPCA;
using trex::trex_selector_methods::trex_spca::TRexSPCAControlParameter;
using trex::trex_selector_methods::trex_spca::TRexSPCAResult;
using trex::trex_selector_methods::trex_spca::SPCAMode;


// ==============================================================================
// Data container
// ==============================================================================

/** @brief Container for generated synthetic data and ground-truth support. */
struct FactorModelData {
    /** @brief Centered observation matrix (n x p). */
    Eigen::MatrixXd X;

    /** @brief True factor matrix (n x M). */
    Eigen::MatrixXd Z;

    /** @brief True sparse loadings matrix (p x M). */
    Eigen::MatrixXd V;

    /** @brief Scaled noise matrix (n x p). */
    Eigen::MatrixXd E;

    /** @brief Actual signal-to-noise ratio (dB). */
    double actual_snr_db;
};


// ==============================================================================
// Data generator
// ==============================================================================

/**
 * @brief Synthetic data generator for the sparse factor model.
 *
 * @details Generates data from the model X = Z * V^T + E,
 *  where Z (n x M) has column-wise N(0, sigma_m^2) factors with
 *  sigma = {5, 3, 1} (falling back to 1 for M > 3), V (p x M) has sparse
 *  loadings (exactly p1 non-zeros per column drawn without replacement from a
 *  shared overlap pool of size overlap_pool_size), and E (n x p) is i.i.d.
 *  Gaussian noise scaled to match the target SNR in dB.
 */
class DataGenerator {
public:

    /**
     * @brief Generate a single dataset from the sparse factor model.
     *
     * @param n                  Number of samples (default 50).
     * @param p                  Number of features (default 100).
     * @param p1                 Number of true active loadings per factor (default 5).
     * @param M                  Number of latent factors (default 3).
     * @param target_snr         Target signal-to-noise ratio in dB (default 0.0).
     * @param overlap_pool_size  Size of the shared candidate-index pool from which
     *                           each factor's support is drawn (default 30). Controls
     *                           how much factor supports can overlap.
     * @param seed               Random seed for reproducibility (default 42).
     *
     * @return FactorModelData containing the generated data and true support.
     */
    static FactorModelData generate_sparse_factor_model(
        Eigen::Index n                 = 50,
        Eigen::Index p                 = 100,
        Eigen::Index p1                = 5,
        Eigen::Index M                 = 3,
        double       target_snr        = 0.0,
        Eigen::Index overlap_pool_size = 30,
        unsigned int seed              = 42
    ) {
        if (p1 > p)
            throw std::invalid_argument("p1 must be less than or equal to p.");
        if (n <= 0 || p <= 0 || p1 <= 0)
            throw std::invalid_argument("n, p, and p1 must be positive integers.");

        FactorModelData data;
        std::mt19937 rng(seed);


        // 1. True factors Z (n x M) with decreasing signal strength
        data.Z = Eigen::MatrixXd(n, M);
        const std::vector<double> factor_stds = {5.0, 3.0, 1.0};
        for (Eigen::Index m = 0; m < M; ++m) {
            const double std_dev = (m < static_cast<Eigen::Index>(factor_stds.size()))
                                 ? factor_stds[m] : 1.0;
            std::normal_distribution<double> nd(0.0, std_dev);
            data.Z.col(m) = Eigen::VectorXd::NullaryExpr(n, [&]() { return nd(rng); });
        }


        // 2. Sparse loadings V (p x M): each factor draws p1 indices from pool
        data.V = Eigen::MatrixXd::Zero(p, M);

        const Eigen::Index actual_pool = std::min(overlap_pool_size, p);
        if (p1 > actual_pool)
            throw std::invalid_argument("p1 must be less than or equal to the overlap pool size.");

        std::vector<Eigen::Index> candidates(actual_pool);
        std::iota(candidates.begin(), candidates.end(), 0);

        for (Eigen::Index m = 0; m < M; ++m) {
            std::shuffle(candidates.begin(), candidates.end(), rng);
            for (Eigen::Index k = 0; k < p1; ++k)
                data.V(candidates[k], m) = 0.9;
        }


        // 3. Signal matrix S = Z * V^T
        const Eigen::MatrixXd S = data.Z * data.V.transpose();

        const double signal_mean = S.mean();
        const double signal_var  = (S.array() - signal_mean).square().sum()
                                 / static_cast<double>(S.size() - 1);


        // 4. Gaussian noise E scaled to target SNR
        // SNR_dB = 10 * log10(var_S / var_E)  =>  var_E = var_S / 10^(SNR_dB / 10)
        std::normal_distribution<double> std_normal(0.0, 1.0);
        Eigen::MatrixXd E_raw = Eigen::MatrixXd::NullaryExpr(n, p, [&]() {
            return std_normal(rng);
        });

        const double raw_noise_var       = (E_raw.array() - E_raw.mean()).square().sum()
                                         / static_cast<double>(E_raw.size() - 1);
        const double target_noise_var    = signal_var / std::pow(10.0, target_snr / 10.0);
        const double noise_scaling_factor = std::sqrt(target_noise_var / raw_noise_var);

        data.E = E_raw * noise_scaling_factor;


        // 5. Assemble X
        data.X = S + data.E;

        const double final_noise_var = (data.E.array() - data.E.mean()).square().sum()
                                     / static_cast<double>(data.E.size() - 1);
        data.actual_snr_db = 10.0 * std::log10(signal_var / final_noise_var);

        return data;
    }

private:
    DataGenerator() = delete;
};


// ==============================================================================
// Simulation configuration
// ==============================================================================

/** @brief Simulation parameters shared across T-Rex SPCA demo sections. */
struct SimConfig {
    Eigen::Index n                 = 50;    ///< Number of samples
    Eigen::Index p                 = 100;   ///< Number of features
    Eigen::Index p1                = 5;     ///< True active loadings per factor
    Eigen::Index M                 = 3;     ///< Number of latent factors
    Eigen::Index overlap_pool_size = 30;    ///< Shared candidate-index pool size
    double       tFDR              = 0.10;  ///< Target FDR for T-Rex SPCA
    double       lambda_2          = -1.0;  ///< Ridge penalty (LARS units); < 0 = auto-compute via CV, 0 = no ridge, > 0 = fixed
    std::size_t  num_MC            = 100;   ///< Monte Carlo trials
    int          base_seed         = 42;    ///< Base RNG seed
};

/** @brief DGP factory callable type: seed → FactorModelData. */
using SPCADGPFactory = std::function<FactorModelData(unsigned seed)>;


// ==============================================================================
// Result types
// ==============================================================================

/**
 * @brief Single-trial evaluation result for a sparse PCA method.
 *
 * All per-component metrics are averaged over M components within one MC trial.
 */
struct SPCASingleResult {
    double tpr = 0.0;  ///< TPR of PC1's support only (see evaluate_spca for rationale)
    double fdr = 0.0;  ///< FDR of PC1's support only (see evaluate_spca for rationale)
    double pev = 0.0;  ///< Cumulative percentage of explained variance across all M components
};

/** @brief MC-aggregated result for one grid point (method × SNR). */
struct SPCAGridPointResult {
    double avg_tpr = 0.0;  ///< Mean TPR across MC trials
    double avg_fdr = 0.0;  ///< Mean FDR across MC trials
    double avg_pev = 0.0;  ///< Mean cumulative PEV across MC trials
    double sd_tpr  = 0.0;  ///< Bessel-corrected SD of TPR
    double sd_fdr  = 0.0;  ///< Bessel-corrected SD of FDR
    double sd_pev  = 0.0;  ///< Bessel-corrected SD of cumulative PEV
};


// ==============================================================================
// Single-trial evaluator
// ==============================================================================

/**
 * @brief Compute per-trial sparse PCA metrics from estimated loadings and scores.
 *
 * @details
 *  - TPR / FDR are evaluated strictly on the **first** principal component (PC1).
 *    PCs 2 and 3 are omitted because PCA's orthogonality constraint mixes their
 *    loading supports across factors, making per-component support recovery
 *    metrics ambiguous beyond the first PC.
 *  - Cumulative PEV is computed via QR decomposition of Z_est across all M components.
 *
 * @param X       Centered observation matrix (n x p). Must be the same X that was
 *                used to produce Z_est (no additional preprocessing applied here).
 * @param V_est   Estimated sparse loading matrix (p x M).
 * @param Z_est   Estimated score matrix (n x M).
 * @param V_true  True loading matrix (p x M); non-zeros mark the true active set.
 *
 * @return SPCASingleResult: PC1 support TPR and FDR, plus cumulative PEV.
 */
inline SPCASingleResult evaluate_spca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& V_est,
    const Eigen::MatrixXd& Z_est,
    const Eigen::MatrixXd& V_true)
{
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    const Eigen::Index M = V_est.cols();

    // -----------------------------------------------------------------
    // 1. Evaluate TPR and FDR strictly on the FIRST Component (m = 0)
    // -----------------------------------------------------------------
    // Paper methodology: PCs 2 and 3 are mixed due to the orthogonality
    // constraint of ordinary PCA.
    // True support metrics are only meaningful for the first PC.

    std::vector<Eigen::Index> true_supp_pc1, est_supp_pc1;
    for (Eigen::Index i = 0; i < p; ++i) {
        if (V_true(i, 0) != 0.0) {
            true_supp_pc1.push_back(i);
        }
        if (std::abs(V_est(i, 0)) > 1e-12) {
            est_supp_pc1.push_back(i);
        }
    }

    int tp = 0;
    for (auto idx : est_supp_pc1) {
        if (std::find(true_supp_pc1.begin(), true_supp_pc1.end(), idx)
            != true_supp_pc1.end()) {
            ++tp;
        }
    }

    const int false_disc = static_cast<int>(est_supp_pc1.size()) - tp;

    double pc1_tpr = true_supp_pc1.empty() ? 1.0 : static_cast<double>(tp) / static_cast<double>(true_supp_pc1.size());
    double pc1_fdr = est_supp_pc1.empty()  ? 0.0 : static_cast<double>(false_disc) / static_cast<double>(est_supp_pc1.size());

    // ----------------------------------------------------------------------
    // 2. Cumulative PEV across ALL components
    // ----------------------------------------------------------------------
    // PEV is evaluated cumulatively via QR decomposition of Z_est
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Z_est);
    const Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    const double total_var = X.squaredNorm() / static_cast<double>(n - 1);
    double cum_ev = 0.0;

    for (Eigen::Index m = 0; m < M; ++m) {
        const double r_mm = R(m, m);
        cum_ev += (r_mm * r_mm) / static_cast<double>(n - 1);
    }

    double final_pev = (total_var > 0.0) ? (cum_ev / total_var) : 0.0;

    // Return the isolated PC1 recovery metrics and the total cumulative PEV
    return {pc1_tpr, pc1_fdr, final_pev};
}


// ==============================================================================
// Preprocessing
// ==============================================================================

/**
 * @brief Center columns of X in place (mean subtraction only, NO scaling).
 *
 * @details Covariance-PCA footing, shared by every method (matching the legacy
 *          R reference pipeline and TRexSPCA's own internal convention). The
 *          column scales must NOT be normalized: in the sparse factor model
 *          the factor amplitude signal lives in the column variances, and
 *          z-scoring the columns (correlation PCA) destroys it — measured
 *          effect at -10 dB: T-Rex SPCA FDR degrades from ~0.14 to ~0.52 and
 *          OraclePCA TPR from ~1.0 to ~0.87 (see
 *          validation_trex_spca_06_handrolled_comparison).
 *
 * @param X Observation matrix (n x p), modified in place.
 */
inline void center_columns(Eigen::MatrixXd& X)
{
    X.rowwise() -= X.colwise().mean();
}


// ==============================================================================
// MC loop helpers
// ==============================================================================

namespace detail {

/** @brief Compute mean/sd from per-trial vectors and print a done line. */
inline SPCAGridPointResult aggregate_mc(
    std::size_t                 num_MC,
    const std::string&          label,
    const std::vector<double>&  tpr_vec,
    const std::vector<double>&  fdr_vec,
    const std::vector<double>&  pev_vec)
{
    const int    iMC = static_cast<int>(num_MC);
    const double dMC = static_cast<double>(num_MC);

    double avg_tpr = 0.0, avg_fdr = 0.0, avg_pev = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_tpr += tpr_vec[mc];
        avg_fdr += fdr_vec[mc];
        avg_pev += pev_vec[mc];
    }
    avg_tpr /= dMC;  avg_fdr /= dMC;  avg_pev /= dMC;

    double sd_tpr = 0.0, sd_fdr = 0.0;
    if (num_MC > 1) {
        double ss_tpr = 0.0, ss_fdr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            ss_tpr += (tpr_vec[mc] - avg_tpr) * (tpr_vec[mc] - avg_tpr);
            ss_fdr += (fdr_vec[mc] - avg_fdr) * (fdr_vec[mc] - avg_fdr);
        }
        sd_tpr = std::sqrt(ss_tpr / (dMC - 1.0));
        sd_fdr = std::sqrt(ss_fdr / (dMC - 1.0));
    }

    std::cout << "  " << label
              << " \u2014 done. TPR=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDR=" << avg_fdr << "\n\n" << std::flush;

    return {avg_tpr, avg_fdr, avg_pev, sd_tpr, sd_fdr};
}

} // namespace detail


// ==============================================================================
// Parallel MC inner loops
// ==============================================================================

/**
 * @brief Run num_MC parallel T-Rex SPCA trials.
 *
 * @details Data flow per trial:
 *          - make_data() returns raw FactorModelData (X = Z V^T + E, uncentered).
 *          - The pipeline centers X once (covariance-PCA convention); this same
 *            centered X is shared by the selector and by evaluate_spca().
 *          - TRexSPCA re-centers internally (a no-op on already-centered X) and
 *            restores X before returning, so dat.X stays centered for evaluation.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in start/done lines.
 * @param make_data         Factory: seed → FactorModelData (called per trial).
 *                          Must capture all parameters by value for OMP safety.
 * @param tFDR              Target FDR for T-Rex SPCA.
 * @param mode              Loading assembly mode (ActiveSet or Thresholded).
 * @param base_seed_offset  Base seed; trial mc uses base_seed_offset + mc * 1000.
 * @param lambda_2          Ridge penalty (LARS units); < 0 = auto-compute via CV
 *                          (default), 0 = no ridge (pure T-LASSO), > 0 = fixed.
 * @param en_solver         EN solver variant (TENET or TENET_AUG; default TENET_AUG).
 * @param scaling_mode      Column scaling for X/dummies inside the per-PC
 *                          T-Rex selector (L2 or ZSCORE; default L2).
 *
 * @return SPCAGridPointResult with avg/sd TPR, FDR and avg PEV.
 */
inline SPCAGridPointResult run_mc_trials_trex_spca(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    double                tFDR,
    SPCAMode              mode,
    unsigned              base_seed_offset,
    double                lambda_2    = -1.0,
    ENSolverType          en_solver   = ENSolverType::TENET,
    ScalingMode           scaling_mode = ScalingMode::L2
) {

    const int iMC = static_cast<int>(num_MC);
    std::vector<double> tpr_vec(num_MC, 0.0);
    std::vector<double> fdr_vec(num_MC, 0.0);
    std::vector<double> pev_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        // Deterministic per-trial data seed (reproducible from base_seed_offset).
        // Disjoint 1000-wide bands keep the DGP streams of different trials apart.
        const unsigned seed = base_seed_offset + static_cast<unsigned>(mc) * 1000u;

        auto dat = make_data(seed);

        // Pipeline preprocessing: center X once (covariance-PCA footing, legacy
        // convention). All methods and metrics share this same centered X.
        center_columns(dat.X);

        TRexSPCAControlParameter ctrl;
        ctrl.mode                    = mode;
        ctrl.en_solver               = en_solver;
        // TRexSPCAControlParameter has no top-level trex_ctrl of its own --
        // the base control parameters live under gvs_ctrl.trex_ctrl (GVS
        // sub-selector), avoiding a duplicate/out-of-sync copy.
        ctrl.gvs_ctrl.trex_ctrl.scaling_mode = scaling_mode;
        ctrl.gvs_ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
        ctrl.gvs_ctrl.lambda_2       = lambda_2;   // < 0: auto-compute via CV; 0: no ridge; > 0: fixed

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        // Selector seed -1: hardware entropy per trial, so the computer-generated
        // dummies are independent across trials (required for valid MC FDR
        // estimates). The data DGP above is what the deterministic seed controls.
        TRexSPCA       spca(X_map, dat.V.cols(), tFDR, ctrl, -1);
        TRexSPCAResult res = spca.select();

        const auto m   = evaluate_spca(dat.X, res.V, res.Z, dat.V);
        tpr_vec[mc] = m.tpr;
        fdr_vec[mc] = m.fdr;
        pev_vec[mc] = m.pev;
    }

    return detail::aggregate_mc(num_MC, progress_label, tpr_vec, fdr_vec, pev_vec);
}


/**
 * @brief Run num_MC parallel ordinary PCA trials.
 *
 * @details Ordinary PCA selects all p variables per component (no sparsity),
 *          so FDR ≈ (p - p1) / p and TPR = 1 regardless of SNR.
 *          The metric of interest is the cumulative PEV.
 *
 *          Data flow: make_data() returns raw data; the pipeline centers X once
 *          (covariance-PCA convention). PCA is then constructed with center=false
 *          (X is already centered), and evaluate_spca() shares the same centered X.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in start/done lines.
 * @param make_data         Factory: seed → FactorModelData.
 * @param base_seed_offset  Base seed; trial mc uses base_seed_offset + mc * 1000.
 *
 * @return SPCAGridPointResult with avg/sd TPR, FDR and avg PEV.
 */
inline SPCAGridPointResult run_mc_trials_pca(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    unsigned              base_seed_offset)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> tpr_vec(num_MC, 0.0);
    std::vector<double> fdr_vec(num_MC, 0.0);
    std::vector<double> pev_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned seed = base_seed_offset + static_cast<unsigned>(mc) * 1000u;
        auto dat = make_data(seed);

        // Pipeline preprocessing: center X once (covariance PCA), shared by all methods.
        center_columns(dat.X);

        // X is already centered → center=false, normalize=false; covariance PCA.
        pca_ns::PCA       pca(dat.X, dat.V.cols(), /*center=*/false, /*normalize=*/false);
        pca_ns::PCAResult res = pca.fit();

        const auto m = evaluate_spca(dat.X, res.V, res.Z, dat.V);
        tpr_vec[mc] = m.tpr;
        fdr_vec[mc] = m.fdr;
        pev_vec[mc] = m.pev;
    }

    return detail::aggregate_mc(num_MC, progress_label, tpr_vec, fdr_vec, pev_vec);
}


/**
 * @brief Run num_MC parallel oracle-thresholded PCA trials.
 *
 * @details Oracle PCA uses the ordinary PCA loadings but retains exactly the
 *          top p1 entries (by absolute value) per component — i.e., it knows
 *          the true support cardinality. This provides an upper bound on what
 *          any data-driven thresholding method can achieve.
 *
 *          Data flow: make_data() returns raw data; the pipeline centers X once
 *          (covariance-PCA convention). PCA is constructed with center=false
 *          (X is already centered). The oracle score matrix Z_oracle = X * V_oracle
 *          and evaluate_spca() share the same centered X.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in start/done lines.
 * @param make_data         Factory: seed → FactorModelData.
 * @param p1                Oracle threshold: true active loadings per factor.
 * @param base_seed_offset  Base seed; trial mc uses base_seed_offset + mc * 1000.
 *
 * @return SPCAGridPointResult with avg/sd TPR, FDR and avg PEV.
 */
inline SPCAGridPointResult run_mc_trials_oracle_pca(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    Eigen::Index          p1,
    unsigned              base_seed_offset)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> tpr_vec(num_MC, 0.0);
    std::vector<double> fdr_vec(num_MC, 0.0);
    std::vector<double> pev_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned seed = base_seed_offset + static_cast<unsigned>(mc) * 1000u;
        auto dat = make_data(seed);

        // Pipeline preprocessing: center X once (covariance PCA), shared by all methods.
        center_columns(dat.X);

        const Eigen::Index p = dat.X.cols();
        const Eigen::Index M = dat.V.cols();

        // X is already centered → center=false, normalize=false; covariance PCA.
        pca_ns::PCA       pca(dat.X, M, /*center=*/false, /*normalize=*/false);
        pca_ns::PCAResult ord = pca.fit();

        // Threshold: keep top p1 entries per loading vector, then normalize
        Eigen::MatrixXd V_oracle = Eigen::MatrixXd::Zero(p, M);
        for (Eigen::Index comp = 0; comp < M; ++comp) {
            const Eigen::VectorXd v_m = ord.V.col(comp);

            std::vector<double> abs_vals(static_cast<std::size_t>(p));
            for (Eigen::Index i = 0; i < p; ++i)
                abs_vals[static_cast<std::size_t>(i)] = std::abs(v_m(i));

            std::nth_element(abs_vals.begin(),
                             abs_vals.begin() + (p - p1),
                             abs_vals.end());
            const double thresh = abs_vals[static_cast<std::size_t>(p - p1)];

            for (Eigen::Index i = 0; i < p; ++i)
                if (std::abs(v_m(i)) >= thresh) V_oracle(i, comp) = v_m(i);
            V_oracle.col(comp).normalize();
        }
        const Eigen::MatrixXd Z_oracle = dat.X * V_oracle;

        const auto m   = evaluate_spca(dat.X, V_oracle, Z_oracle, dat.V);
        tpr_vec[mc] = m.tpr;
        fdr_vec[mc] = m.fdr;
        pev_vec[mc] = m.pev;
    }

    return detail::aggregate_mc(num_MC, progress_label, tpr_vec, fdr_vec, pev_vec);
}


// ==============================================================================
// Output
// ==============================================================================

/**
 * @brief Print MC-averaged T-Rex SPCA results as an aligned table (console + .txt)
 *        and write a tidy long-format CSV.
 *
 * @param num_MC        Number of Monte Carlo trials.
 * @param file_stem     Output file base name (no folder, no extension).
 * @param snr_values    SNR grid (dB) used during the simulation.
 * @param method_names  Ordered list of method names (keys into result maps).
 * @param fdr_map       Averaged FDR per method, indexed by SNR grid position.
 * @param tpr_map       Averaged TPR per method, indexed by SNR grid position.
 * @param pev_map       Averaged cumulative PEV per method, indexed by SNR grid position.
 */
inline void save_and_print_spca_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     snr_values,
    const std::vector<std::string>&                method_names,
    const std::map<std::string, Eigen::VectorXd>&  fdr_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_map,
    const std::map<std::string, Eigen::VectorXd>&  pev_map)
{
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);

    std::ofstream out_file(folder + file_stem + ".txt");
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // Header
    {
        std::ostringstream ss;
        ss << "\n"
           << "======================================================================\n"
           << "=== T-Rex SPCA Results (averaged over " << num_MC << " MC trials) ===\n"
           << "======================================================================\n\n";
        print_dual(ss.str());
    }

    // Table dimensions
    const std::size_t method_width  = 22;
    const std::size_t metric_width  = 8;
    const std::size_t snr_col_width = 8;
    const std::size_t col_width     = 10;

    // Column header + dashed separator
    {
        std::ostringstream hdr;
        hdr << std::left  << std::setw(method_width)  << "Method"
            << std::left  << std::setw(metric_width)  << "Metric"
            << std::right << std::setw(snr_col_width) << "SNR(dB)";
        for (double snr : snr_values)
            hdr << std::fixed << std::setprecision(1) << std::setw(col_width) << snr;
        hdr << "\n"
            << std::string(method_width + metric_width + snr_col_width +
                           col_width * snr_values.size(), '-') << "\n";
        print_dual(hdr.str());
    }

    // Data rows
    auto print_row = [&](const std::string& mname, const std::string& metric,
                         const Eigen::VectorXd& data, bool first_row) {
        std::ostringstream row;
        row << std::left  << std::setw(method_width)  << (first_row ? mname : "")
            << std::left  << std::setw(metric_width)  << metric
            << std::setw(snr_col_width) << "" << std::right;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i)
            row << std::fixed << std::setprecision(4) << std::setw(col_width) << data(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& name : method_names) {
        print_row(name, "FDR", fdr_map.at(name), true);
        print_row(name, "TPR", tpr_map.at(name), false);
        print_row(name, "PEV", pev_map.at(name), false);
        print_dual("\n");
    }

    // Tidy long-format CSV (method, metric, snr_db, value)
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "method,metric,snr_db,value\n" << std::fixed << std::setprecision(6);
        for (const auto& name : method_names) {
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                std::ostringstream snr_ss;
                snr_ss << std::fixed << std::setprecision(6) << snr_values[i];
                const std::string s = snr_ss.str();
                csv << name << ",FDR," << s << "," << fdr_map.at(name)(ei) << "\n";
                csv << name << ",TPR," << s << "," << tpr_map.at(name)(ei) << "\n";
                csv << name << ",PEV," << s << "," << pev_map.at(name)(ei) << "\n";
            }
        }
        std::cout << "[Info] CSV results saved to:             "
                  << folder + file_stem + ".csv\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: " << folder + file_stem + ".csv\n";
    }

    if (out_file.is_open()) {
        std::cout << "[Info] Results successfully saved to: " << folder + file_stem + ".txt\n\n";
        out_file.close();
    } else {
        std::cout << "[Warning] Could not open output file: " << folder + file_stem + ".txt\n\n";
    }
}
// ==============================================================================
} // namespace spca_sim
// ==============================================================================
#endif /* End of DEMOS_TREX_SELECTOR_METHODS_TREX_SPCA_SIM_UTILS_HPP */
