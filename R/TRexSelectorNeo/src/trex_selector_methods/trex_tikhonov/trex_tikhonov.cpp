// ===================================================================================
// trex_tikhonov.cpp
// ===================================================================================
/**
 * @file trex_tikhonov.cpp
 *
 * @brief Implementation of TRexTikhonovSelector (general-Tikhonov T-Rex).
 */
// ===================================================================================

// Class header
#include <trex_selector_methods/trex_tikhonov/trex_tikhonov.hpp>

// std includes
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// lambda_2 CV tuners (sparse-K overloads)
#include <ml_methods/model_selection/ienet_cv_ccd.hpp>
#include <ml_methods/model_selection/tikhonov_cv_svd.hpp>

// Shared cluster-aware dummy machinery (GVS parity)
#include <trex_selector_methods/trex_utils/trex_cluster_dummies.hpp>
#include <trex_selector_methods/trex_utils/trex_cluster_hac.hpp>

// Seed mixing (CV fold seed derivation, mirrors TRexGVSSelector)
#include <utils/datageneration/utils_dummygen.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_tikhonov {

// Local namespace aliases
namespace cdum = trex::trex_selector_methods::utils::cluster_dummies;
namespace dn   = trex::trex_selector_methods::utils::data_normalizer;
namespace ms   = trex::ml_methods::model_selection;


// ===================================================================================
// Constructor
// ===================================================================================

TRexTikhonovSelector::TRexTikhonovSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    double                        tFDR,
    TRexTikhonovControlParameter  trex_tik_ctrl,
    int                           seed,
    bool                          verbose
) :
    tc::TRexSelector(X, y, tFDR, trex_tik_ctrl.trex_ctrl, seed, verbose),
    trex_tik_ctrl_(std::move(trex_tik_ctrl))
{
    // ---- Penalty matrix ---------------------------------------------------
    const auto p_idx = static_cast<Eigen::Index>(p_);
    if (trex_tik_ctrl_.tikhonov_K.nonZeros() == 0) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: tikhonov_K is empty; supply a p x p PSD "
            "penalty matrix (e.g. via gammaToK()).");
    }
    if (trex_tik_ctrl_.tikhonov_K.rows() != p_idx ||
        trex_tik_ctrl_.tikhonov_K.cols() != p_idx) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: tikhonov_K must be p x p "
            "(p = X.cols(); PSD symmetry is validated by the CV tuner / "
            "consumed as-is by the solver).");
    }
    trex_tik_ctrl_.tikhonov_K.makeCompressed();

    // ---- Solver gate ------------------------------------------------------
    // TCIENET is the only solver with a general sparse-K mode; the shared
    // dispatch would reject anything K-aware for other enumerators anyway,
    // so fail fast with a precise message.
    if (trex_ctrl_.solver_type != sd::SolverTypeForTRex::TCIENET) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: solver_type must be TCIENET (the general "
            "Tikhonov penalty is only implemented by the CCD IEN solver).");
    }

    // ---- Cluster-aware dummy gates ----------------------------------------
    if (trex_tik_ctrl_.cluster_dummies) {
        // All stored, on-demand, and memory-mapped strategies are served by
        // the DummyGenerator cluster mode (deterministic layer streams; the
        // mmap writer regenerates layers bit-exactly). Only the PERMUTATION
        // variants are rejected: K row-permuted copies of ONE base cannot
        // carry the per-cluster MVN convention (same reasoning as
        // TRexGVSSelector).
        switch (trex_ctrl_.lloop_strategy) {
            case tc::LLoopStrategy::PERMUTATION:
            case tc::LLoopStrategy::PERMUTATION_SEEDED:
                throw std::invalid_argument(
                    "TRexTikhonovSelector: the PERMUTATION-family L-loop "
                    "strategies cannot carry cluster-MVN dummies; choose "
                    "STANDARD, HCONCAT, SKIPL, or SEEDED, or set "
                    "cluster_dummies = false.");
            default:
                break;
        }
        cdum::validatePriorGroups(trex_tik_ctrl_.prior_groups, p_,
                                  "TRexTikhonovSelector");
        if (trex_tik_ctrl_.prior_groups.empty() &&
            (trex_tik_ctrl_.corr_max < 0.0 || trex_tik_ctrl_.corr_max > 1.0)) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: corr_max must be in [0, 1]. Got: " +
                std::to_string(trex_tik_ctrl_.corr_max));
        }
    }

    // ---- CV fold seed (mirrors TRexGVSSelector) ---------------------------
    if (trex_tik_ctrl_.cv_seed >= 0) {
        resolved_cv_seed_ = static_cast<unsigned int>(trex_tik_ctrl_.cv_seed);
    } else if (seed_ >= 0) {
        resolved_cv_seed_ = trex::utils::datageneration::dummygen::mix_seed(
            static_cast<std::uint32_t>(seed_), 1u);
    } else {
        resolved_cv_seed_ = std::random_device{}();
    }
}


// ===================================================================================
// gammaToK — K = Gamma^T Gamma
// ===================================================================================

Eigen::SparseMatrix<double> TRexTikhonovSelector::gammaToK(
    const Eigen::SparseMatrix<double>& gamma)
{
    Eigen::SparseMatrix<double> K =
        (Eigen::SparseMatrix<double>(gamma.transpose()) * gamma).pruned();
    K.makeCompressed();
    return K;
}


// ===================================================================================
// onSelectBegin — resolve lambda_2 before the L-loop
// ===================================================================================

void TRexTikhonovSelector::onSelectBegin() {

    // 1. Dummy cluster model (X_ is already centered + scaled by the base
    //    constructor, so both the HAC distance and Sigma_m are valid).
    if (trex_tik_ctrl_.cluster_dummies) {
        if (verbose_) {
            printProgress("Tikhonov: Setting up dummy cluster structure...");
        }
        setupClusterModel();
        if (verbose_) {
            printProgress("Tikhonov: dummy clusters M = " +
                          std::to_string(clusters_list_.size()));
        }
    }

    // 2. lambda_2 (solver scale).
    lambda2_ = computeLambda2();
    if (verbose_) {
        std::ostringstream oss;
        oss << "Tikhonov: lambda_2 (solver scale) = " << lambda2_;
        printProgress(oss.str());
    }
}


// ===================================================================================
// setupClusterModel — dummy clustering + per-cluster Cholesky factors
// ===================================================================================

void TRexTikhonovSelector::setupClusterModel() {

    if (!trex_tik_ctrl_.prior_groups.empty()) {
        clusters_list_ = cdum::clustersFromPriorGroups(
            trex_tik_ctrl_.prior_groups, p_);
    } else {
        const auto merges = cdum::runCorrelationHAC(
            *X_, trex_tik_ctrl_.hc_linkage, verbose_);
        clusters_list_ = cdum::clustersFromHAC(
            merges, p_, trex_tik_ctrl_.corr_max);
    }

    auto cholesky_lower_list = cdum::buildClusterCholeskys(
        *X_, clusters_list_, n_, "TRexTikhonovSelector");

    // Hand the model to the base DummyGenerator: from here on, EVERY
    // generation entry point — stored (STANDARD / HCONCAT / SKIPL),
    // on-demand (one D_k at a time, serial), and the memory-mapped writers
    // (mmap pre-allocated at maximum dimensions, filled per L-iteration) —
    // draws per-cluster colored MVN layers on the shared streams instead of
    // i.i.d. columns. No hook override is needed; the base
    // prepareDummiesForLStep / ExperimentRunner machinery runs unchanged.
    dummy_gen_.setClusterModel(clusters_list_,
                               std::move(cholesky_lower_list), p_);
}


// ===================================================================================
// buildRunnerConfig — base config + Tikhonov injection
// ===================================================================================

er::ExperimentRunnerConfig TRexTikhonovSelector::buildRunnerConfig(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool use_warm_start,
    er::ExperimentStrategy strategy,
    std::size_t seed_factor,
    std::size_t existing_cols_on_disk) const
{
    er::ExperimentRunnerConfig cfg = tc::TRexSelector::buildRunnerConfig(
        num_dummies, T_stop, use_warm_start, strategy, seed_factor,
        existing_cols_on_disk);

    // The K matrix lives in trex_tik_ctrl_ and outlives every run() call;
    // lambda2_ was resolved in onSelectBegin().
    cfg.tikhonov_K            = &trex_tik_ctrl_.tikhonov_K;
    cfg.tikhonov_fold_dummies = trex_tik_ctrl_.fold_dummy_coupling;
    cfg.solver_params.lambda2 = lambda2_;

    return cfg;
}


// ===================================================================================
// lambda_2 — user supplied or auto-computed via the sparse-K tuners
// ===================================================================================

double TRexTikhonovSelector::computeLambda2() const {

    if (trex_tik_ctrl_.lambda_2 >= 0.0) {
        // Fixed user value (>= 0); lambda_2 == 0 -> pure lasso path.
        // Interpreted in the WORKING column scale (trex_ctrl_.scaling_mode).
        return trex_tik_ctrl_.lambda_2;
    }

    // CV is entered with X centered + scaled per trex_ctrl_.scaling_mode and
    // y centered (base-class constructor). The tuners re-normalize per fold
    // to unit L2, so the selected lambda_2 is on the unit-L2 solver scale;
    // the ZSCORE working scale (squared column norm n - 1 instead of 1)
    // needs the (n - 1) factor — same conversion as the IEN-geometry
    // methods in TRexGVSSelector::computeLambda2().
    const double scale_adjust =
        (trex_ctrl_.scaling_mode == dn::ScalingMode::ZSCORE && n_ > 1)
            ? static_cast<double>(n_ - 1)
            : 1.0;

    switch (trex_tik_ctrl_.lambda2_method) {
        case TikLambda2Method::CV_1SE_IEN_CCD:
        case TikLambda2Method::CV_MIN_IEN_CCD: {
            // 2D (lambda_2 x lambda_1) CV with lambda_1 profiled out;
            // well-posed in any p/n regime. Grid sizes are the class
            // defaults (25 x 100) — the scan is a 2D surface, not a 1D
            // analytic grid.
            ms::ienet_cv_ccd cv;
            cv.fit(*X_, y_, trex_tik_ctrl_.tikhonov_K,
                   trex_tik_ctrl_.cv_n_folds,
                   /*n_lambda2=*/25,
                   /*lambda2_ratio=*/1e4,
                   /*n_lambda1=*/100,
                   /*lambda1_min_ratio=*/-1.0,
                   resolved_cv_seed_);
            const double l2 =
                (trex_tik_ctrl_.lambda2_method ==
                 TikLambda2Method::CV_MIN_IEN_CCD)
                    ? cv.cv_min() : cv.cv_1se();
            return l2 * scale_adjust;
        }
        case TikLambda2Method::CV_1SE_TIK_SVD:
        case TikLambda2Method::CV_MIN_TIK_SVD: {
            // Analytic lambda_1 = 0 backbone; refused when the unpenalized
            // null-space block interpolates the training folds (any pick
            // there is floor-driven).
            ms::tikhonov_cv_svd cv;
            cv.fit(*X_, y_, trex_tik_ctrl_.tikhonov_K,
                   trex_tik_ctrl_.cv_n_folds,
                   trex_tik_ctrl_.cv_n_lambda,
                   /*lambda_ratio=*/1e4,
                   resolved_cv_seed_);
            if (cv.contrast_interpolates()) {
                throw std::invalid_argument(
                    "TRexTikhonovSelector::computeLambda2: CV_*_TIK_SVD is "
                    "structurally uninformative on this design "
                    "(null_dim >= n_train: the unpenalized null-space block "
                    "interpolates the training folds). Use CV_*_IEN_CCD "
                    "instead.");
            }
            const double l2 =
                (trex_tik_ctrl_.lambda2_method ==
                 TikLambda2Method::CV_MIN_TIK_SVD)
                    ? cv.cv_min() : cv.cv_1se();
            return l2 * scale_adjust;
        }
    }
    throw std::invalid_argument(
        "TRexTikhonovSelector::computeLambda2: unknown TikLambda2Method.");
}

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_tikhonov */
// ===================================================================================
