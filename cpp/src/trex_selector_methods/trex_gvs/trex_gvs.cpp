// ===================================================================================
// trex_gvs.cpp
// ===================================================================================
/**
 * @file trex_gvs.cpp
 *
 * @brief Implementation of the Grouped Variable Selection T-Rex Selector (GVS T-Rex).
 */
// ===================================================================================

// std includes
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


// GVS header
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

// Ridge lambda-selection (header-only): two retained k-fold CV variants.
//   SVD  -> ridge_cv_svd           (JacobiSVD direct ridge)
//   CCD  -> enet_cv_ccd            (glmnet-faithful coordinate descent, alpha=0)
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/model_selection/enet_cv_ccd.hpp>

// Data normalization helpers
#include <trex_selector_methods/trex_utils/trex_data_normalizer.hpp>

// Solver dispatch
#include <trex_selector_methods/trex_utils/trex_solver_dispatch.hpp>

// Concrete solver types (for direct in-memory warm-start)
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_aug_solver.hpp>
#include <tsolvers/linear_model/lars_based/tienet_aug_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>

// OpenMP (parallel-K loop)
#ifdef _OPENMP
#include <omp.h>
#endif

// ===================================================================================

namespace trex::trex_selector_methods::trex_gvs {

// Local namespace aliases
namespace dn   = trex::trex_selector_methods::utils::data_normalizer;
namespace lars = trex::tsolvers::linear_model::lars_based;
namespace mm   = trex::trex_selector_methods::utils::memmap_manager;

// Static helper: return a copy of `p` with `use_memory_mapping` forced to
// false, so the base ctor does not pre-allocate `memmap_mgr_` (which would
// use the base row count `n_` instead of the GVS effective row count
// `n_eff_ = n_ + M` required for IEN).
// GVS re-allocates the manager in `onSelectBegin()` once `n_eff_` is known.
static tc::TRexControlParameter stripMmapForBase(
    const tc::TRexControlParameter& p)
{
    tc::TRexControlParameter q = p;
    q.use_memory_mapping = false;
    return q;
}


// ===================================================================================
// Constructor
// ===================================================================================

TRexGVSSelector::TRexGVSSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    double                       tFDR,
    TRexGVSControlParameter      trex_gvs_ctrl,
    int                          seed,
    bool                         verbose
) :
    tc::TRexSelector(X, y, tFDR,
                     stripMmapForBase(trex_gvs_ctrl.trex_ctrl),
                     seed, verbose),
    trex_gvs_ctrl_(std::move(trex_gvs_ctrl)),
    gvs_use_mmap_(trex_gvs_ctrl_.trex_ctrl.use_memory_mapping)
{
    // Resolve CV fold-permutation seed (analogous to permutation_base_seed_ in base).
    if (trex_gvs_ctrl_.cv_seed >= 0) {
        resolved_cv_seed_ = static_cast<unsigned int>(trex_gvs_ctrl_.cv_seed);
    } else if (seed_ >= 0) {
        resolved_cv_seed_ = trex::utils::datageneration::dummygen::mix_seed(
            static_cast<std::uint32_t>(seed_), 1u);
    } else {
        resolved_cv_seed_ = std::random_device{}();
    }
    // ---- Solver-type compatibility checks ---------------------------------
    if (trex_gvs_ctrl_.gvs_type == GVSType::EN) {
        // EN picks its concrete solver from en_solver (TENET / TENET_AUG), so
        // the base solver_type must name a member of that TENET family.
        if (trex_ctrl_.solver_type != sd::SolverTypeForTRex::TENET &&
            trex_ctrl_.solver_type != sd::SolverTypeForTRex::TENET_AUG) {
            throw std::invalid_argument(
                "TRexGVSSelector: gvs_type = EN requires solver_type = TENET or "
                "TENET_AUG (matching en_solver)."
            );
        }
    } else { // IEN
        if (trex_ctrl_.solver_type != sd::SolverTypeForTRex::TIENET_AUG) {
            throw std::invalid_argument(
                "TRexGVSSelector: gvs_type = IEN requires solver_type = "
                "TIENET_AUG (the group penalty is absorbed into the "
                "row-augmented system built by TIENETAug_Solver)."
            );
        }
    }

    // ---- Memory mapping --------------------------------------------------
    // The user-requested flag is captured in the member-initializer list as
    // `gvs_use_mmap_` (read from the *original* parameter copy). The flag
    // passed to the base ctor was stripped via stripMmapForBase() so the
    // base does not pre-allocate `memmap_mgr_` at the wrong row count.
    // GVS re-allocates the manager in `onSelectBegin()` once `n_eff_` is
    // known.

    // ---- L-loop strategy gate ---------------------------------------------
    // Supported: STANDARD, HCONCAT, SKIPL, DIRECT.
    //   * STANDARD / SKIPL / DIRECT: redraw all `LL` layers from scratch per
    //     L-iteration. SKIPL collapses to a single iteration with
    //     `LL = max_dummy_multiplier`. DIRECT is functionally identical to
    //     STANDARD inside GVS because the overridden `evaluateStep`
    //     consumes `D_solver_bufs_` rather than streaming from disk.
    //   * HCONCAT: append one fresh layer per L-iteration.
    // Rejected: PERMUTATION, PERMUTATION_ONDEMAND. Row permutations of the
    // base dummy matrix would destroy the per-cluster MVN covariance
    // structure that the GVS dummies are explicitly designed to mirror.
    if (trex_ctrl_.lloop_strategy == tc::LLoopStrategy::PERMUTATION ||
        trex_ctrl_.lloop_strategy == tc::LLoopStrategy::PERMUTATION_ONDEMAND) {
        throw std::invalid_argument(
            "TRexGVSSelector: PERMUTATION and PERMUTATION_ONDEMAND L-loop "
            "strategies are not supported. Row permutation of the base "
            "dummy matrix would destroy the per-cluster MVN covariance "
            "structure that GVS dummies are designed to mirror."
        );
    }

    validateGVSParameters();
}


// ===================================================================================
// Validation
// ===================================================================================

void TRexGVSSelector::validateGVSParameters() const {

    if (trex_gvs_ctrl_.corr_max < 0.0 || trex_gvs_ctrl_.corr_max > 1.0) {
        throw std::invalid_argument(
            "TRexGVSSelector: corr_max must be in [0, 1]. Got: " +
            std::to_string(trex_gvs_ctrl_.corr_max));
    }

    // Reject Ward (non-Euclidean correlation distance).
    if (trex_gvs_ctrl_.hc_linkage != hac::LinkageMethod::Single &&
        trex_gvs_ctrl_.hc_linkage != hac::LinkageMethod::Complete &&
        trex_gvs_ctrl_.hc_linkage != hac::LinkageMethod::Average &&
        trex_gvs_ctrl_.hc_linkage != hac::LinkageMethod::WPGMA) {
        throw std::invalid_argument(
            "TRexGVSSelector: unsupported linkage method. "
            "Supported: Single, Complete, Average, WPGMA.");
    }

    // Note: lambda_2 < 0 is the "auto-compute" sentinel (R's NULL); it is a
    // valid input, so it is not rejected here. lambda_2 == 0 is the degenerate
    // (pure-TLASSO) case; lambda_2 > 0 is a fixed ridge penalty.

    // Validate prior_groups (length p, 0-based, contiguous in [0, M-1]).
    if (!trex_gvs_ctrl_.prior_groups.empty()) {
        if (trex_gvs_ctrl_.prior_groups.size() != p_) {
            throw std::invalid_argument(
                "TRexGVSSelector: prior_groups length must equal p = " +
                std::to_string(p_) +
                ". Got: " + std::to_string(trex_gvs_ctrl_.prior_groups.size()));
        }

        Eigen::Index max_id = -1;
        for (auto id : trex_gvs_ctrl_.prior_groups) {
            if (id < 0) {
                throw std::invalid_argument(
                    "TRexGVSSelector: prior_groups must be 0-based "
                    "non-negative integers.");
            }
            if (id > max_id) max_id = id;
        }

        // check: every id in [0, max_id] must appear at least once.
        std::vector<bool> seen(static_cast<std::size_t>(max_id + 1), false);
        for (auto id : trex_gvs_ctrl_.prior_groups) {
            seen[static_cast<std::size_t>(id)] = true;
        }
        for (std::size_t m = 0; m < seen.size(); ++m) {
            if (!seen[m]) {
                throw std::invalid_argument(
                    "TRexGVSSelector: prior_groups IDs are non-contiguous. "
                    "Cluster ID " + std::to_string(m) + " is missing.");
            }
        }

        // check: every label in group_labels (if provided) they must correspond to a cluster ID.
        if (!trex_gvs_ctrl_.group_labels.empty() &&
            trex_gvs_ctrl_.group_labels.size() !=
                static_cast<std::size_t>(max_id + 1)) {
            throw std::invalid_argument(
                "TRexGVSSelector: group_labels.size() must equal the number "
                "of unique cluster IDs.");
        }
    }
}


// ===================================================================================
// Setup: cluster discovery and per-cluster pre-computation
// ===================================================================================

void TRexGVSSelector::setupGVS() {
    if (!trex_gvs_ctrl_.prior_groups.empty()) {
        setupGVS_PriorGroups();
    } else {
        setupGVS_Cluster();
    }
    finalizeSetup();
}


void TRexGVSSelector::setupGVS_PriorGroups() {

    gvs_setup_.hc_method_used = "prior";

    // Determine M
    Eigen::Index max_id = -1;
    for (auto id : trex_gvs_ctrl_.prior_groups) {
        if (id > max_id) max_id = id;
    }
    const std::size_t M = static_cast<std::size_t>(max_id + 1);
    gvs_setup_.max_clusters = M;

    // Variables by cluster
    gvs_setup_.clusters_list.assign(M, {});
    for (Eigen::Index j = 0;
         j < static_cast<Eigen::Index>(p_); ++j) {
        const Eigen::Index id = trex_gvs_ctrl_.prior_groups[static_cast<std::size_t>(j)];
        gvs_setup_.clusters_list[static_cast<std::size_t>(id)].push_back(j);
    }

    // Groups_vec
    gvs_setup_.groups_vec.resize(static_cast<Eigen::Index>(p_));
    for (Eigen::Index j = 0;
         j < static_cast<Eigen::Index>(p_); ++j) {
        gvs_setup_.groups_vec(j) = static_cast<int>(
            trex_gvs_ctrl_.prior_groups[static_cast<std::size_t>(j)]);
    }
}


std::vector<hac::MergeStep> TRexGVSSelector::runClustering() const {
    using MapType = Eigen::Map<Eigen::MatrixXd>;
    using CorrDist = hac::DistancePolicy<MapType, hac::DistanceMetric::Correlation>;

    switch (trex_gvs_ctrl_.hc_linkage) {
        case hac::LinkageMethod::Single:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Single>(*X_, /*use_mmap=*/false, verbose_);
        case hac::LinkageMethod::Complete:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Complete>(*X_, /*use_mmap=*/false, verbose_);
        case hac::LinkageMethod::Average:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Average>(*X_, /*use_mmap=*/false, verbose_);
        case hac::LinkageMethod::WPGMA:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::WPGMA>(*X_, /*use_mmap=*/false, verbose_);
        default:
            throw std::invalid_argument(
                "TRexGVSSelector: unsupported linkage method.");
    }
}


void TRexGVSSelector::setupGVS_Cluster() {

    // Linkage tag
    switch (trex_gvs_ctrl_.hc_linkage) {
        case hac::LinkageMethod::Single:   gvs_setup_.hc_method_used = "single";   break;
        case hac::LinkageMethod::Complete: gvs_setup_.hc_method_used = "complete"; break;
        case hac::LinkageMethod::Average:  gvs_setup_.hc_method_used = "average";  break;
        case hac::LinkageMethod::WPGMA:    gvs_setup_.hc_method_used = "wpgma";    break;
        default:                           gvs_setup_.hc_method_used = "unknown";  break;
    }

    auto merges = runClustering();

    // Cut at height = 1 - corr_max
    const double height = 1.0 - trex_gvs_ctrl_.corr_max;

    // Determine cluster labels
    auto labels = hac::DendrogramUtils::cut_tree_by_height(
        merges, static_cast<Eigen::Index>(p_), height);

    auto clusters = hac::DendrogramUtils::group_indices_by_label(labels);

    gvs_setup_.clusters_list  = std::move(clusters);
    gvs_setup_.max_clusters   = gvs_setup_.clusters_list.size();

    // Groups vector
    gvs_setup_.groups_vec.resize(static_cast<Eigen::Index>(p_));
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(labels.size()); ++j) {
        gvs_setup_.groups_vec(j) = static_cast<int>(labels[static_cast<std::size_t>(j)]);
    }
}


void TRexGVSSelector::finalizeSetup() {

    const std::size_t M = gvs_setup_.max_clusters;
    const auto p_idx = static_cast<Eigen::Index>(p_);
    const double inv_nm1 = 1.0 / static_cast<double>(static_cast<Eigen::Index>(n_) - 1);

    // Per-cluster sizes
    gvs_setup_.cluster_sizes.resize(M);
    gvs_setup_.cholesky_lower_list.resize(M);

    // Per-cluster Sigma_m and Cholesky factor L_m on X_ (already normalized
    // per trex_ctrl_.scaling_mode; the estimator below adapts to either mode).
    for (std::size_t m = 0; m < M; ++m) {

        const auto& cols = gvs_setup_.clusters_list[m];
        const auto p_m = static_cast<Eigen::Index>(cols.size());
        gvs_setup_.cluster_sizes[m] = static_cast<std::size_t>(p_m);

        // Group covariance computation: Sigma_m = (1 / (n-1)) * X_m^T X_m.
        // This auto-adapts to the column scaling chosen for X_:
        //   - ScalingMode::L2     -> ||x_j|| = 1, so diag(Sigma_m) = 1/(n-1)
        //                            (dummies sampled unit-L2, matching X_).
        //   - ScalingMode::ZSCORE -> ||x_j|| = sqrt(n-1), so diag(Sigma_m) = 1
        //                            (dummies sampled unit-SD, matching X_).
        // In both cases the sampled dummy columns share the scale of X_.
        Eigen::MatrixXd Sigma_m(p_m, p_m);
        for (Eigen::Index i = 0; i < p_m; ++i) {
            for (Eigen::Index j = i; j < p_m; ++j) {
                const double v = inv_nm1 *
                                 X_->col(cols[static_cast<std::size_t>(i)])
                                 .dot(X_->col(cols[static_cast<std::size_t>(j)]));
                Sigma_m(i, j) = v;
                Sigma_m(j, i) = v;
            }
            Sigma_m(i, i) += 1e-10;
        }

        // Cholesky factorization for MVN sampling
        Eigen::LLT<Eigen::MatrixXd> llt(Sigma_m);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error(
                "TRexGVSSelector: Cholesky factorization failed for cluster " +
                std::to_string(m) + ".");
        }
        // store only the lower triangular factor (L_m) since Sigma_m is symmetric
        gvs_setup_.cholesky_lower_list[m] = llt.matrixL();
    }

    // (M x p) IEN binary support matrix (row m = 1_m^T).
    gvs_setup_.IEN_cl_id_vectors = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(M), p_idx);

    for (std::size_t m = 0; m < M; ++m) {
        for (auto j : gvs_setup_.clusters_list[m]) {
            gvs_setup_.IEN_cl_id_vectors(static_cast<Eigen::Index>(m), j) = 1.0;
        }
    }

    // Group labels (carry through if supplied).
    if (trex_gvs_ctrl_.group_labels.size() == M) { // supplied labels match number of clusters
        gvs_setup_.group_labels = trex_gvs_ctrl_.group_labels;
    } else {
        gvs_setup_.group_labels.clear();
    }
}


// ===================================================================================
// lambda_2 (LARS units) -- user supplied or auto-computed via the rule
// selected in TRexGVSControlParameter::lambda2_method.
// ===================================================================================

double TRexGVSSelector::computeLambda2() const {

    if (trex_gvs_ctrl_.lambda_2 >= 0.0) {
        // Fixed user value (>= 0). lambda_2 == 0 -> degenerate TLASSO (no ridge).
        // Interpreted in the WORKING column scale (trex_ctrl_.scaling_mode):
        // under ZSCORE the columns have squared norm (n-1) instead of 1, so a
        // value calibrated for unit-L2 columns must be multiplied by (n-1) by
        // the caller if the same effective shrinkage is desired.
        return trex_gvs_ctrl_.lambda_2;
    }
    // lambda_2 < 0: "not supplied" sentinel -> auto-compute below.

    // Auto-compute lambda_2 and convert to LARS units:
    //   lambda_lars = lambda_cv * p / 2 * scale_adjust
    //
    // CV is entered with X centered and scaled per trex_ctrl_.scaling_mode and
    // y centered, as prepared by the base-class constructor. Each CV method
    // extracts per-fold copies of (X_train, X_test, y_train, y_test) and
    // re-normalises them to UNIT L2 for itself (so no fold leaks global
    // statistics and lambda_cv is always on the unit-L2 scale regardless of
    // the working scaling mode).
    //
    // scale_adjust converts the unit-L2-calibrated lambda_cv to the working
    // column scale: the ridge penalty balances against ||x_j||^2, which is 1
    // under L2 and (n-1) under ZSCORE (sample-SD scaling). With
    // lambda_working = (n-1) * lambda_unit the augmented system
    // [c·X; sqrt(lambda_working)·I]·d2' is a global scalar multiple of the
    // unit-L2 system [X; sqrt(lambda_unit)·I]·d2 (c = sqrt(n-1)), and LARS
    // paths are invariant under global column scaling — so the selection is
    // identical to the L2 run (pinned by the ScalingModeParity GVS tests).
    //
    // Two retained variants (selected by lambda2_method):
    //   * CV_*_SVD (ridge_cv_svd, JacobiSVD): normalises each fold column to unit
    //     L2-norm; the grid is anchored at c_max = max_j|x_j^T y_c|, so lambda_cv is on
    //     the unit-L2 scale.
    //   * CV_*_CCD (enet_cv_ccd, alpha=0, coordinate descent): glmnet-faithful;
    //     standardises X/y internally (overwriting the base-class L2-normalisation),
    //     anchors at glmnet's alpha=0 lambda_max and applies fdev/devmax early-termination.
    //     Returns lambda on glmnet's reported scale (matches cv.glmnet(alpha=0)); the p/2
    //     factor below converts to LARS units exactly as R's lm_dummy does
    //     (lambda_2_lars = lambda * ncol(X) / 2).
    //
    // Both EN and IEN fit on the original (X, y).  Fitting on the IEN augmented reference
    // system [X; G_ref] was investigated and reverted: for high-correlation DGPs (rho ≈ 0.99)
    // it inflates lambda_2 by ~3 orders of magnitude, driving all active-group columns to
    // near-identical normalised structure (bottom row ≈ 0.97), which collapses T-Rex FDR
    // control in the same way as small lambda_2.
    //
    // IEN lambda_2 sensitivity (inherited from the R reference, which uses the
    // SAME CV value and the SAME * p/2 conversion for both EN and IEN even
    // though the IEN appends only M << p ridge rows): the IEN group-indicator
    // rows scale with sqrt(lambda_2), so a large CV-chosen lambda_2 (typical
    // for n > p designs) can make the shared bottom rows dominate the
    // augmented column geometry — each active's group-aligned dummies become
    // nearly collinear with the active itself and the FDP estimate never
    // drops below the target. If the IEN track selects nothing, supply a
    // smaller fixed lambda_2 (see the EndToEnd_IEN test for a worked example).
    namespace ms = trex::ml_methods::model_selection;

    double lambda_cv = 0.0;
    switch (trex_gvs_ctrl_.lambda2_method) {
        // == SVD-based CV (ridge_cv_svd, JacobiSVD direct ridge) =====================
        case LambdaSelectionMethod::CV_MIN_SVD: {
            ms::ridge_cv_svd cv;
            cv.fit(*X_, y_,
                   trex_gvs_ctrl_.cv_n_folds,
                   trex_gvs_ctrl_.cv_n_lambda,
                   /*lambda_ratio=*/1000.0,
                   resolved_cv_seed_);
            lambda_cv = cv.cv_min();
            break;
        }
        case LambdaSelectionMethod::CV_1SE_SVD: {
            // Largest lambda within 1 SE of the CV minimum (SVD path).
            ms::ridge_cv_svd cv;
            cv.fit(*X_, y_,
                   trex_gvs_ctrl_.cv_n_folds,
                   trex_gvs_ctrl_.cv_n_lambda,
                   /*lambda_ratio=*/1000.0,
                   resolved_cv_seed_);
            lambda_cv = cv.cv_1se();
            break;
        }

        // == CCD-based CV (enet_cv_ccd, alpha=0, glmnet-faithful) ===========
        case LambdaSelectionMethod::CV_MIN_CCD: {
            ms::enet_cv_ccd cv;
            cv.fit(*X_, y_,
                   /*alpha=*/0.0,
                   trex_gvs_ctrl_.cv_n_folds,
                   trex_gvs_ctrl_.cv_n_lambda,
                   /*lambda_min_ratio=*/-1.0,   // auto: 1e-4 (n>p) or 1e-2 (n<=p)
                   resolved_cv_seed_);
            lambda_cv = cv.cv_min();
            break;
        }
        case LambdaSelectionMethod::CV_1SE_CCD: {
            // glmnet-faithful 1SE rule via coordinate descent (default). See CV_MIN_CCD.
            ms::enet_cv_ccd cv;
            cv.fit(*X_, y_,
                   /*alpha=*/0.0,
                   trex_gvs_ctrl_.cv_n_folds,
                   trex_gvs_ctrl_.cv_n_lambda,
                   /*lambda_min_ratio=*/-1.0,
                   resolved_cv_seed_);
            lambda_cv = cv.cv_1se();
            break;
        }
        default:
            throw std::invalid_argument(
                "TRexGVSSelector::computeLambda2: unknown LambdaSelectionMethod.");
    }

    // Working-scale conversion (see the scale_adjust note above).
    const double scale_adjust =
        (trex_ctrl_.scaling_mode == dn::ScalingMode::ZSCORE && n_ > 1)
            ? static_cast<double>(n_ - 1)
            : 1.0;

    return lambda_cv * static_cast<double>(p_) / 2.0 * scale_adjust;

}


// ===================================================================================
// Augmentation note
// ===================================================================================
//
// All row augmentation lives inside the solvers:
//   - TENETAug_Solver  (EN, Zou-Hastie ridge blocks; R lm_dummy.R GVS_type "EN"),
//   - TIENETAug_Solver (IEN, group-indicator block;  R lm_dummy.R GVS_type "IEN").
// GVS only assembles the plain (n x L) MVN dummy blocks below.


// ===================================================================================
// Cluster-aware MVN dummy draw
// ===================================================================================

Eigen::MatrixXd TRexGVSSelector::drawClusterDummyLayer(std::mt19937& rng) const {

    const auto n_rows = static_cast<Eigen::Index>(n_);
    const auto p = static_cast<Eigen::Index>(p_);

    Eigen::MatrixXd layer = Eigen::MatrixXd::Zero(n_rows, p);
    std::normal_distribution<double> N01(0.0, 1.0);

    for (std::size_t m = 0; m < gvs_setup_.max_clusters; ++m) {

        const auto& cols = gvs_setup_.clusters_list[m];
        const auto p_m = static_cast<Eigen::Index>(cols.size());

        // skip empty clusters (should not happen)
        if (p_m == 0) {
            continue;
        }

        // drawn univariate Z ~ N(0, I) of shape (n_rows x p_m).
        Eigen::MatrixXd Z(n_rows, p_m);
        for (Eigen::Index j = 0; j < p_m; ++j) {
            for (Eigen::Index i = 0; i < n_rows; ++i) {
                Z(i, j) = N01(rng);
            }
        }

        // coloring transform: block = Z * L_m^T  (rows of block ~ N(0, Sigma_m)).
        Eigen::MatrixXd block = Z * gvs_setup_.cholesky_lower_list[m].transpose();

        // Scatter into output columns.
        for (Eigen::Index j = 0; j < p_m; ++j) {
            layer.col(cols[static_cast<std::size_t>(j)]) = block.col(j);
        }
    }

    return layer;
}


// ===================================================================================
// Assemble one experiment's plain (n x L) dummy block into `dst`
// ===================================================================================
//
// dst is (n x L) = horzcat of `dummy_layers_[k]`. All augmentation (ridge /
// group-indicator rows) is built inside TENETAug_Solver / TIENETAug_Solver.
// Column scaling: caller (EN variants center + rescale; IEN passes raw).

template <typename Derived>
void TRexGVSSelector::assembleDummyBlock(std::size_t k,
                                         Eigen::MatrixBase<Derived>& dst) const {

    const auto n = static_cast<Eigen::Index>(n_);
    const auto p = static_cast<Eigen::Index>(p_);

    const auto& layers = dummy_layers_[k];
    Eigen::Index col = 0;
    for (const auto& layer : layers) {
        dst.block(0, col, n, p) = layer;
        col += p;
    }
}


// ===================================================================================
// extractPhiContribFromPath -- accumulate per-experiment phi_T contribution
// ===================================================================================

Eigen::MatrixXd TRexGVSSelector::extractPhiContribFromPath(
    const tsolvers::SparseBetaPath& beta_path,
    std::size_t            p,
    std::size_t            num_dummies,
    std::size_t            T_stop,
    double                 eps,
    bool                   verbose)
{
    const auto p_idx = static_cast<Eigen::Index>(p);
    const auto T_idx = static_cast<Eigen::Index>(T_stop);
    const std::size_t n_steps = beta_path.steps.size();

    Eigen::MatrixXd contrib = Eigen::MatrixXd::Zero(p_idx, T_idx);

    if (num_dummies == 0 || n_steps == 0) return contrib;

    // Active-dummy count per step from the sparse supports (same eps
    // threshold as the former dense row scan).
    std::vector<std::size_t> dummy_count(n_steps, 0);
    for (std::size_t s = 0; s < n_steps; ++s) {
        const auto& st = beta_path.steps[s];
        for (std::size_t e = 0; e < st.idx.size(); ++e) {
            if (st.idx[e] >= p && std::abs(st.val[e]) > eps) {
                ++dummy_count[s];
            }
        }
    }

    // For each t in [1, T_stop], find the FIRST step where exactly t dummy
    // variables are non-zero.  At that step, every active original feature
    // gets a +1 contribution.
    for (std::size_t t = 1; t <= T_stop; ++t) {

        std::size_t found_col = n_steps;
        for (std::size_t s = 0; s < n_steps; ++s) {
            if (dummy_count[s] == t) {
                found_col = s;
                break;
            }
        }

        if (found_col == n_steps) {
            if (verbose) {
                // Path never reached t active dummies; row stays 0.
            }
            continue;
        }

        // Add indicator of active original features at that step.
        const auto& st = beta_path.steps[found_col];
        for (std::size_t e = 0; e < st.idx.size(); ++e) {
            if (st.idx[e] < p && std::abs(st.val[e]) > eps) {
                contrib(static_cast<Eigen::Index>(st.idx[e]),
                        static_cast<Eigen::Index>(t - 1)) += 1.0;
            }
        }
    }

    return contrib;
}


// ===================================================================================
// Run K experiments (serial), with optional in-memory warm-start across T
// ===================================================================================

TRexGVSSelector::ExpAgg TRexGVSSelector::runKExperiments(
    std::size_t T_stop,
    std::size_t num_dummies,
    bool        use_warm_start)
{
    const std::size_t K                      = trex_ctrl_.K;
    const auto p_idx     = static_cast<Eigen::Index>(p_);
    const auto T_idx     = static_cast<Eigen::Index>(T_stop);
    const bool ien                           = (trex_gvs_ctrl_.gvs_type == GVSType::IEN);
    const bool en_aug    = (trex_gvs_ctrl_.gvs_type == GVSType::EN &&
                            trex_gvs_ctrl_.en_solver == ENSolverType::TENET_AUG);

    // Single scaling convention for the whole pipeline: X (base class),
    // dummies D_k (prepareDummiesForLStep), and the solvers below all follow
    // trex_ctrl_.scaling_mode. The solvers receive already-scaled X/D_k
    // (normalize=false, intercept=false), so this only governs any internal
    // augmented-column handling; propagating it keeps the convention coherent
    // (mirrors the base-class conversion in trex.cpp buildSolverConfig()).
    const tsolvers::ScalingMode solver_scaling =
        (trex_ctrl_.scaling_mode == dn::ScalingMode::ZSCORE)
            ? tsolvers::ScalingMode::ZSCORE
            : tsolvers::ScalingMode::L2;

    // Per-experiment phi_T contributions; summed after the parallel section.
    std::vector<Eigen::MatrixXd> phi_T_per_k(K);

    // If starting fresh (no warm-start), wipe leftover solver instances.
    // D_solver_bufs_ / D_solver_maps_ are populated by prepareDummiesForLStep.
    if (!use_warm_start) {
        solvers_cache_.clear();
        solvers_cache_.resize(K);
    }

    // ---- OpenMP setup (mirror base ExperimentRunner pattern) ------------
    const bool do_parallel =
        trex_ctrl_.parallel_rnd_experiments && (trex_ctrl_.max_outer_threads > 1);
    #ifdef _OPENMP
    int old_max_levels = omp_get_max_active_levels();
    if (do_parallel) omp_set_max_active_levels(2);
    #endif

    #pragma omp parallel if(do_parallel)
    {
        #ifdef _OPENMP
        if (do_parallel) omp_set_num_threads(trex_ctrl_.max_inner_threads);
        #endif
        Eigen::setNbThreads(trex_ctrl_.max_inner_threads);

        #pragma omp for schedule(dynamic)
        for (std::size_t k = 0; k < K; ++k) {

            tsolvers::SparseBetaPath beta_path;

            if (use_warm_start && solvers_cache_[k] != nullptr) {
                // -- Warm-start branch: solver already bound to *D_solver_maps_[k]. --
                solvers_cache_[k]->executeStep(T_stop, /*early_stop=*/true);
                beta_path = solvers_cache_[k]->getBetaPathSparse();

            } else {
                // -- Fresh construction branch:
                //    D_solver_maps_[k] was bound by prepareDummiesForLStep
                //    (in-memory or mmap). X_solver_map_ / y_solver_map_ view
                //    the base-normalized X_ and centered y_ for ALL variants;
                //    the augmented solvers build and own their augmented
                //    systems internally.
                std::unique_ptr<tsolvers::TSolver_Base> solver;
                if (ien) {
                    // IEN: group-informed row augmentation inside
                    // TIENETAug_Solver. D_k is RAW (the solver applies the
                    // R-faithful post-augmentation column scaling itself);
                    // normalize/intercept=true only govern the degenerate
                    // lambda2 == 0 collapse, where the raw dummies must be
                    // normalized by the inner solver.
                    solver = std::make_unique<lars::TIENETAug_Solver>(
                        *X_solver_map_,
                        *D_solver_maps_[k],
                        *y_solver_map_,
                        lambda2_,
                        gvs_setup_.groups_vec,
                        /*normalize=*/true,
                        /*intercept=*/true,
                        /*verbose=*/false,
                        solver_scaling,
                        trex_gvs_ctrl_.tenet_aug_use_lars
                    );
                } else if (en_aug) {
                    // EN-aug: Zou-Hastie augmentation inside TENETAug_Solver,
                    // run AS CONSTRUCTED (inner normalize=false for
                    // lambda2 > 0): with the pre-normalized unit-scale X and
                    // D_k columns the d1/d2 algebra keeps the augmented
                    // columns unit-scale, reproducing the Gram-based TENET
                    // path to machine precision. normalize/intercept=false
                    // also cover the lambda2 == 0 collapse (data already
                    // externally scaled).
                    solver = std::make_unique<lars::TENETAug_Solver>(
                        *X_solver_map_,
                        *D_solver_maps_[k],
                        *y_solver_map_,
                        lambda2_,
                        /*normalize=*/false,
                        /*intercept=*/false,
                        /*verbose=*/false,
                        solver_scaling,
                        trex_gvs_ctrl_.tenet_aug_use_lars
                    );
                }
                // Plain EN (TENET): LARS-EN path algorithm with Gram-matrix
                // ridge modification (augmentation handled inside the solver).
                else {
                    solver = std::make_unique<lars::TENET_Solver>(
                        *X_solver_map_,
                        *D_solver_maps_[k],
                        *y_solver_map_,
                        lambda2_,
                        /*normalize=*/false,
                        /*intercept=*/false,
                        /*verbose=*/false,
                        solver_scaling
                    );
                }

                solver->executeStep(T_stop, /*early_stop=*/true);
                beta_path = solver->getBetaPathSparse();
                solvers_cache_[k] = std::move(solver);
            }

            // Per-experiment phi_T contribution (independent slots).
            phi_T_per_k[k] = extractPhiContribFromPath(
                beta_path, p_, num_dummies, T_stop, eps_, verbose_);
        }
    }

    #ifdef _OPENMP
    if (do_parallel) omp_set_max_active_levels(old_max_levels);
    #endif

    // Aggregate per-K contributions.
    Eigen::MatrixXd phi_T_acc = Eigen::MatrixXd::Zero(p_idx, T_idx);
    for (std::size_t k = 0; k < K; ++k) {
        phi_T_acc += phi_T_per_k[k];
    }

    // Average over K experiments.
    phi_T_acc /= static_cast<double>(K);

    // Combine outputs into a single struct
    ExpAgg out;
    out.phi_T_mat = std::move(phi_T_acc);
    out.Phi       = out.phi_T_mat.col(T_idx - 1);
    return out;
}


// ===================================================================================
// evaluateStep override -- used by inherited runTLoop
// ===================================================================================

TRexGVSSelector::StepView TRexGVSSelector::evaluateStep(
    std::size_t            num_dummies,
    std::size_t            T_stop,
    bool                   use_warm_start,
    tc::er::ExperimentStrategy /*strategy*/,
    std::size_t            /*seed_factor*/,
    std::size_t            /*existing_on_disk*/)
{
    StepView view;

    auto agg = runKExperiments(T_stop, num_dummies, use_warm_start);

    view.exp_results.phi_T_mat = std::move(agg.phi_T_mat);
    view.exp_results.Phi       = view.exp_results.phi_T_mat.col(
        static_cast<Eigen::Index>(T_stop) - 1);

    view.Phi       = view.exp_results.Phi;
    view.Phi_prime = computePhiPrime(view.exp_results.phi_T_mat,
                                     view.exp_results.Phi,
                                     num_dummies);
    view.FDP_hat   = computeFDPHat(voting_grid_, view.exp_results.Phi,
                                   view.Phi_prime);
    return view;
}


// ===================================================================================
// L-loop hook: cluster-aware dummy generation.
// Strategies handled: STANDARD, HCONCAT, SKIPL, DIRECT.
//   - STANDARD / SKIPL / DIRECT : redraw all LL layers from scratch.
//   - HCONCAT                   : append one fresh layer per L-iteration.
// ===================================================================================

void TRexGVSSelector::prepareDummiesForLStep(LStepContext& ctx)
{
    const std::size_t K     = trex_ctrl_.K;
    const std::size_t LL    = ctx.L_iter;

    const bool ien = (trex_gvs_ctrl_.gvs_type == GVSType::IEN);

    // Fresh dummy data invalidates the cached solvers. Clear them BEFORE the
    // dummy buffers they view are reassigned below, so no dangling view
    // exists even transiently (same contract as the base-class L-loop hook).
    solvers_cache_.clear();
    solvers_cache_.resize(K);

    const auto n_idx = static_cast<Eigen::Index>(n_);

    // Three of the four supported strategies follow the redraw-all path.
    // HCONCAT is the only append-only strategy. PERMUTATION variants are
    // rejected in the constructor and trigger a defensive throw here.
    bool redraw_all = false;
    switch (trex_ctrl_.lloop_strategy) {
        case tc::LLoopStrategy::STANDARD:
        case tc::LLoopStrategy::SKIPL:
        case tc::LLoopStrategy::ONDEMAND:
            redraw_all = true;
            break;
        case tc::LLoopStrategy::HCONCAT:
            redraw_all = false;
            break;
        case tc::LLoopStrategy::PERMUTATION:
        case tc::LLoopStrategy::PERMUTATION_ONDEMAND:
        default:
            throw std::logic_error(
                "TRexGVSSelector::prepareDummiesForLStep: unsupported "
                "LLoopStrategy reached the dummy-preparation hook "
                "(should have been rejected by the constructor).");
    }

    // (Re)allocate the per-experiment layer cache on first iteration.
    // SKIPL only ever calls this hook once with LL = max_dummy_multiplier,
    // so the LL == 1 trigger generalises to "first call this select()".
    if (LL == 1 || dummy_layers_.size() != K) {
        dummy_layers_.assign(K, {});
    }

    // Deterministic per-(k, LL) RNG stream seeded from `seed_`.
    const std::uint64_t base_seed = (seed_ < 0)
        ? static_cast<std::uint64_t>(std::random_device{}())
        : static_cast<std::uint64_t>(seed_);

    // ---- Generate / extend per-experiment dummy layers -------------------
    for (std::size_t k = 0; k < K; ++k) {

        // Seed the FULL mt19937 state via std::seed_seq.  Seeding mt19937 from
        // a single 32-bit value (e.g. a scalar mix_seed result) leaves nearby
        // seeds able to emit correlated initial output — fatal for T-Rex,
        // whose FDR calibration assumes the K dummy realisations are mutually
        // independent and independent of X.  seed_seq scrambles the supplied
        // entropy words across the entire 624-word generator state, so every
        // (experiment k, L-iteration LL) draws a decorrelated dummy stream.
        std::seed_seq seq{
            static_cast<std::uint32_t>(base_seed & 0xFFFFFFFFu),
            static_cast<std::uint32_t>(base_seed >> 32),
            static_cast<std::uint32_t>(k),
            static_cast<std::uint32_t>(LL),
            0x9E3779B9u  // golden-ratio constant for extra avalanche mixing
        };
        std::mt19937 rng(seq);

        if (redraw_all) {
            // Redraw all LL layers from scratch.
            dummy_layers_[k].clear();
            dummy_layers_[k].reserve(LL);
            for (std::size_t li = 0; li < LL; ++li) {
                dummy_layers_[k].push_back(drawClusterDummyLayer(rng));
            }
        } else {
            // HCONCAT: append one fresh layer per L-iteration.
            while (dummy_layers_[k].size() < LL) {
                dummy_layers_[k].push_back(drawClusterDummyLayer(rng));
            }
        }
    }

    // ---- (Re)build solver-side D buffers ----------------------------------
    // EN variants (TENET, TENETAug): the dummy columns are centered + rescaled
    // with the same convention as X_ (TENET consumes them as-is; TENETAug's
    // d1/d2 algebra requires unit-scale inputs to keep the augmented columns
    // unit-scale). IEN: the block stays RAW — TIENETAug applies the R-faithful
    // post-augmentation column scaling to [D; B] itself.
    if (gvs_use_mmap_) {
        // Memory-mapped path: write directly into the per-K mmap buffers
        // pre-allocated in onSelectBegin() (n rows each).
        D_solver_bufs_.clear();
        D_solver_maps_.clear();
        D_solver_maps_.resize(K);
        for (std::size_t k = 0; k < K; ++k) {
            auto dmap = memmap_mgr_->getDMap(k, ctx.num_dummies);
            if (dmap.rows() != n_idx) {
                throw std::runtime_error(
                    "TRexGVSSelector::prepareDummiesForLStep: "
                    "mmap buffer row count does not match n.");
            }
            assembleDummyBlock(k, dmap);
            if (!ien) {
            // FIX (2026-07-08): dummies enter the solver AS DRAWN (centered
            // only, NO per-column norm equalization). The cluster-MVN draw
            // already carries the correct population scale (Sigma_m computed
            // on the normalized X_); its random realized column norms are an
            // essential part of the dummy null distribution. Exactly
            // unit-normalizing each dummy column erased that fluctuation and
            // made dummies systematically less competitive in the EN path --
            // measured on the trex_spca rdump10 head-to-head (identical X and
            // lambda_2): PC1 FDR inflated from ~0.10 (CRAN R reference, which
            // keeps the fluctuation) to ~0.13. Center-only matches the R
            // pipeline (dummies are centered by the augmented scale() there).
                dmap.rowwise() -= dmap.colwise().mean();
            }
            // Bind the long-lived Map for runKExperiments. The owning
            // storage is the mmap file managed by memmap_mgr_.
            D_solver_maps_[k] = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
                dmap.data(), dmap.rows(), dmap.cols());
        }
    } else {
        // In-memory path.
        D_solver_bufs_.assign(K, Eigen::MatrixXd(
            n_idx, static_cast<Eigen::Index>(ctx.num_dummies)));
        D_solver_maps_.clear();
        D_solver_maps_.resize(K);
        for (std::size_t k = 0; k < K; ++k) {
            assembleDummyBlock(k, D_solver_bufs_[k]);
            if (!ien) {
            // FIX (2026-07-08): dummies enter the solver AS DRAWN (centered
            // only, NO per-column norm equalization). The cluster-MVN draw
            // already carries the correct population scale (Sigma_m computed
            // on the normalized X_); its random realized column norms are an
            // essential part of the dummy null distribution. Exactly
            // unit-normalizing each dummy column erased that fluctuation and
            // made dummies systematically less competitive in the EN path --
            // measured on the trex_spca rdump10 head-to-head (identical X and
            // lambda_2): PC1 FDR inflated from ~0.10 (CRAN R reference, which
            // keeps the fluctuation) to ~0.13. Center-only matches the R
            // pipeline (dummies are centered by the augmented scale() there).
                D_solver_bufs_[k].rowwise() -=
                    D_solver_bufs_[k].colwise().mean();
            }
            D_solver_maps_[k] = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
                D_solver_bufs_[k].data(),
                D_solver_bufs_[k].rows(),
                D_solver_bufs_[k].cols());
        }
    }

    ctx.existing_on_disk = 0;

    if (verbose_) {
        std::ostringstream oss;
        oss << "GVS L = " << std::setw(2) << LL
            << " | num_dummies = " << ctx.num_dummies;
        printProgress(oss.str());
    }
}


// ===================================================================================
// onSelectBegin -- pre-L-loop variant setup
// ===================================================================================

void TRexGVSSelector::onSelectBegin() {

    // 1. Group structure (Sigma_m, L_m, IEN_cl_id_vectors).
    if (verbose_) { printProgress("GVS: Setting up group structure..."); }
    setupGVS();

    if (verbose_) {
        printProgress("GVS: Method = " + gvs_setup_.hc_method_used +
                      ", M = " + std::to_string(gvs_setup_.max_clusters));
    }

    // 2. lambda_2 (LARS units).
    lambda2_ = computeLambda2();
    if (verbose_) {
        std::ostringstream oss;
        oss << "GVS: lambda_2 (LARS units) = " << lambda2_;
        printProgress(oss.str());
    }

    // 3. Solver-side Maps over X_ and y_ (all variants; the augmented
    //    solvers TENETAug / TIENETAug build and own their augmented systems
    //    internally). Heap-allocated so the Map objects outlive any cached
    //    solver holding pointers to them.
    X_solver_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
        X_->data(), X_->rows(), X_->cols());
    y_solver_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(
        y_.data(), y_.size());

    // 4. Allocate per-K memory-mapped D files at max size if requested.
    //    shared=false: GVS draws per-experiment distinct MVN dummies, so
    //    each k requires its own backing file (also a prerequisite for the
    //    parallel-K loop in runKExperiments). Rows = n for all variants
    //    (the files hold the plain dummy blocks; augmented buffers are
    //    solver-owned).
    if (gvs_use_mmap_) {
        const std::size_t max_dummies = trex_ctrl_.max_dummy_multiplier * p_;
        memmap_mgr_ = std::make_unique<mm::MemmapManager>(
            n_, max_dummies, trex_ctrl_.K, /*shared=*/false, verbose_);
        memmap_mgr_->initialize();
    }
}


// ===================================================================================
// finalizeSelectionResult -- widen into GVSSelectionResult + GVS-side cleanup
// ===================================================================================

TRexGVSSelector::SelectionResult TRexGVSSelector::finalizeSelectionResult(
    SelectionResult&& base_result)
{
    // Copy base fields into the persistent GVS result.
    gvs_result_.selected_var   = base_result.selected_var;
    gvs_result_.v_thresh       = base_result.v_thresh;
    gvs_result_.R_mat          = base_result.R_mat;
    gvs_result_.T_stop         = base_result.T_stop;
    gvs_result_.num_dummies    = base_result.num_dummies;
    gvs_result_.dummy_factor_L = dummy_multiplier_LL_;
    gvs_result_.Phi_prime      = base_result.Phi_prime;
    gvs_result_.voting_grid    = base_result.voting_grid;
    gvs_result_.FDP_hat_mat    = base_result.FDP_hat_mat;
    gvs_result_.Phi_mat        = base_result.Phi_mat;

    // GVS-specific fields.
    gvs_result_.lambda2_used   = lambda2_;
    gvs_result_.gvs_type       = trex_gvs_ctrl_.gvs_type;
    gvs_result_.max_clusters   = gvs_setup_.max_clusters;
    gvs_result_.hc_method_used = gvs_setup_.hc_method_used;
    gvs_result_.groups_vec     = gvs_setup_.groups_vec;
    gvs_result_.group_labels   = gvs_setup_.group_labels;

    if (verbose_) {
        printProgress("GVS: Selection complete. Selected " +
                      std::to_string(getNumSelected()) + " variables.\n");
    }

    // GVS-specific cleanup: solver cache first (solvers view the D buffers
    // and the X/y Maps), then buffers and Maps. Base `select()` will
    // subsequently run warm_start_mgr_/memmap cleanup and denormalize X.
    solvers_cache_.clear();
    D_solver_bufs_.clear();
    D_solver_maps_.clear();
    X_solver_map_.reset();
    y_solver_map_.reset();
    dummy_layers_.clear();

    // Return the (also-updated) base-sliced result.  We propagate the
    // dummy_factor_L the GVS hook has just resolved so any base consumer
    // that inspects the SelectionResult sees the correct value.
    base_result.dummy_factor_L = dummy_multiplier_LL_;
    return base_result;
}


// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_gvs */
// ===================================================================================
