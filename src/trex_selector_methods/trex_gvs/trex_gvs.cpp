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

// Ridge lambda-selection (header-only): GCV-optimal and k-fold CV.
#include <ml_methods/model_selection/ridge_gcv.hpp>
#include <ml_methods/model_selection/ridge_cv.hpp>

// Data normalization helpers
#include <trex_selector_methods/trex_utils/trex_data_normalizer.hpp>

// Solver dispatch
#include <trex_selector_methods/trex_utils/trex_solver_dispatch.hpp>

// Concrete solver types (for direct in-memory warm-start)
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
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
// `n_eff_ = n_ + M` required for IEN). GVS re-allocates the manager in
// `onSelectBegin()` once `n_eff_` is known.
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
    TRexGVSControlParameter      gvs_control,
    tc::TRexControlParameter     trex_control,
    int                          seed,
    bool                         verbose
) :
    tc::TRexSelector(X, y, tFDR, stripMmapForBase(trex_control), seed, verbose),
    gvs_ctrl_(std::move(gvs_control)),
    gvs_use_mmap_(trex_control.use_memory_mapping)
{
    // Resolve CV fold-permutation seed (analogous to permutation_base_seed_ in base).
    if (gvs_ctrl_.cv_seed >= 0) {
        resolved_cv_seed_ = static_cast<unsigned int>(gvs_ctrl_.cv_seed);
    } else if (seed_ >= 0) {
        resolved_cv_seed_ = trex::utils::datageneration::dummygen::mix_seed(
            static_cast<std::uint32_t>(seed_), 1u);
    } else {
        resolved_cv_seed_ = std::random_device{}();
    }
    // ---- Solver-type compatibility checks ---------------------------------
    if (gvs_ctrl_.gvs_type == GVSType::EN) {
        if (trex_ctrl_.solver_type != sd::SolverTypeForTRex::TENET) {
            throw std::invalid_argument(
                "TRexGVSSelector: gvs_type = EN requires solver_type = TENET.");
        }
    } else { // IEN
        if (trex_ctrl_.solver_type != sd::SolverTypeForTRex::TLASSO) {
            throw std::invalid_argument(
                "TRexGVSSelector: gvs_type = IEN requires solver_type = TLASSO "
                "(the L2 penalty is fully absorbed into the augmented system)."
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
    // Rejected: PERMUTATION, PERMUTATION_DIRECT. Row permutations of the
    // base dummy matrix would destroy the per-cluster MVN covariance
    // structure that the GVS dummies are explicitly designed to mirror.
    if (trex_ctrl_.lloop_strategy == tc::LLoopStrategy::PERMUTATION ||
        trex_ctrl_.lloop_strategy == tc::LLoopStrategy::PERMUTATION_DIRECT) {
        throw std::invalid_argument(
            "TRexGVSSelector: PERMUTATION and PERMUTATION_DIRECT L-loop "
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

    if (gvs_ctrl_.corr_max < 0.0 || gvs_ctrl_.corr_max > 1.0) {
        throw std::invalid_argument(
            "TRexGVSSelector: corr_max must be in [0, 1]. Got: " +
            std::to_string(gvs_ctrl_.corr_max));
    }

    // Reject Ward (non-Euclidean correlation distance).
    if (gvs_ctrl_.hc_linkage != hac::LinkageMethod::Single &&
        gvs_ctrl_.hc_linkage != hac::LinkageMethod::Complete &&
        gvs_ctrl_.hc_linkage != hac::LinkageMethod::Average &&
        gvs_ctrl_.hc_linkage != hac::LinkageMethod::WPGMA) {
        throw std::invalid_argument(
            "TRexGVSSelector: unsupported linkage method. "
            "Supported: Single, Complete, Average, WPGMA.");
    }

    if (gvs_ctrl_.lambda_2 < 0.0) {
        throw std::invalid_argument(
            "TRexGVSSelector: lambda_2 must be >= 0. Got: " +
            std::to_string(gvs_ctrl_.lambda_2));
    }

    // Validate prior_groups (length p, 0-based, contiguous in [0, M-1]).
    if (!gvs_ctrl_.prior_groups.empty()) {
        if (gvs_ctrl_.prior_groups.size() != p_) {
            throw std::invalid_argument(
                "TRexGVSSelector: prior_groups length must equal p = " +
                std::to_string(p_) +
                ". Got: " + std::to_string(gvs_ctrl_.prior_groups.size()));
        }

        Eigen::Index max_id = -1;
        for (auto id : gvs_ctrl_.prior_groups) {
            if (id < 0) {
                throw std::invalid_argument(
                    "TRexGVSSelector: prior_groups must be 0-based "
                    "non-negative integers.");
            }
            if (id > max_id) max_id = id;
        }

        // check: every id in [0, max_id] must appear at least once.
        std::vector<bool> seen(static_cast<std::size_t>(max_id + 1), false);
        for (auto id : gvs_ctrl_.prior_groups) {
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
        if (!gvs_ctrl_.group_labels.empty() &&
            gvs_ctrl_.group_labels.size() !=
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
    if (!gvs_ctrl_.prior_groups.empty()) {
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
    for (auto id : gvs_ctrl_.prior_groups) {
        if (id > max_id) max_id = id;
    }
    const std::size_t M = static_cast<std::size_t>(max_id + 1);
    gvs_setup_.max_clusters = M;

    // Variables by cluster
    gvs_setup_.clusters_list.assign(M, {});
    for (Eigen::Index j = 0;
         j < static_cast<Eigen::Index>(p_); ++j) {
        const Eigen::Index id = gvs_ctrl_.prior_groups[static_cast<std::size_t>(j)];
        gvs_setup_.clusters_list[static_cast<std::size_t>(id)].push_back(j);
    }

    // Groups_vec
    gvs_setup_.groups_vec.resize(static_cast<Eigen::Index>(p_));
    for (Eigen::Index j = 0;
         j < static_cast<Eigen::Index>(p_); ++j) {
        gvs_setup_.groups_vec(j) = static_cast<int>(
            gvs_ctrl_.prior_groups[static_cast<std::size_t>(j)]);
    }
}


std::vector<hac::MergeStep> TRexGVSSelector::runClustering() const {
    using MapType = Eigen::Map<Eigen::MatrixXd>;
    using CorrDist = hac::DistancePolicy<MapType, hac::DistanceMetric::Correlation>;

    switch (gvs_ctrl_.hc_linkage) {
        case hac::LinkageMethod::Single:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Single>(*X_);
        case hac::LinkageMethod::Complete:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Complete>(*X_);
        case hac::LinkageMethod::Average:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Average>(*X_);
        case hac::LinkageMethod::WPGMA:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::WPGMA>(*X_);
        default:
            throw std::invalid_argument(
                "TRexGVSSelector: unsupported linkage method.");
    }
}


void TRexGVSSelector::setupGVS_Cluster() {

    // Linkage tag
    switch (gvs_ctrl_.hc_linkage) {
        case hac::LinkageMethod::Single:   gvs_setup_.hc_method_used = "single";   break;
        case hac::LinkageMethod::Complete: gvs_setup_.hc_method_used = "complete"; break;
        case hac::LinkageMethod::Average:  gvs_setup_.hc_method_used = "average";  break;
        case hac::LinkageMethod::WPGMA:    gvs_setup_.hc_method_used = "wpgma";    break;
        default:                           gvs_setup_.hc_method_used = "unknown";  break;
    }

    auto merges = runClustering();

    // Cut at height = 1 - corr_max
    const double height = 1.0 - gvs_ctrl_.corr_max;

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

    // Per-cluster Sigma_m and Cholesky factor L_m on the (already L2-normalized) X_.
    for (std::size_t m = 0; m < M; ++m) {

        const auto& cols = gvs_setup_.clusters_list[m];
        const auto p_m = static_cast<Eigen::Index>(cols.size());
        gvs_setup_.cluster_sizes[m] = static_cast<std::size_t>(p_m);

        // Group covariance computation: Sigma_m = (1 / (n-1)) * X_m^T X_m
        // Note: X_ is L2-normalized but NOT YET divided by sqrt(n - 1): done here.
        // The Diagonal of X_m^T X_m is 1.0 per column.
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
    if (gvs_ctrl_.group_labels.size() == M) { // supplied labels match number of clusters
        gvs_setup_.group_labels = gvs_ctrl_.group_labels;
    } else {
        gvs_setup_.group_labels.clear();
    }
}


// ===================================================================================
// lambda_2 (LARS units) -- user supplied or auto-computed via the rule
// selected in TRexGVSControlParameter::lambda2_method.
// ===================================================================================

double TRexGVSSelector::computeLambda2() const {

    if (gvs_ctrl_.lambda_2 > 0.0) {
        return gvs_ctrl_.lambda_2;
    }

    // Auto-compute lambda_2 and convert from glmnet to LARS units: lambda_lars = lambda_glmnet * p / 2
    // (matches R's trex_GVS(), which uses cv.glmnet(alpha=0)$lambda.1se * p / 2).
    //
    // Method notes:
    //   GCV    — closed-form GCV via SVD; introduced as a large-data shortcut (large n and p make
    //            K-fold CV prohibitively expensive). Not present in the original R reference.
    //   CV_MIN — K-fold CV, lambda minimising mean MSE. Matches R's lambda.min.
    //   CV_1SE — K-fold CV, glmnet-style 1SE rule (largest lambda within 1 SE of the CV minimum).
    //            Intentionally selects a larger lambda_2 than CV_MIN, making EN more ridge-like:
    //            correlated variables couple and enter the LARS path together. On high-correlation
    //            DGPs this produces TPR = 1.0 at controlled FDR — expected behaviour, not a bug.
    //            Matches R's lambda.1se and is the default in the R reference implementation.
    //
    // Both EN and IEN fit on the original (X, y).  Fitting on the IEN augmented reference system
    // [X; G_ref] was investigated and reverted: for high-correlation DGPs (rho ≈ 0.99) it inflates
    // lambda_2 by ~3 orders of magnitude, driving all active-group columns (real and dummy) to near-
    // identical normalised structure (bottom row ≈ 0.97), which collapses T-Rex FDR control in the
    // same way as small lambda_2 — TPP is unchanged at ≈ 0.02.  The root cause of IEN TPP ≈ 0.02
    // on the Hastie all-active DGP is not lambda_2 calibration; see assembleD() / drawClusterDummyLayer().
    namespace ms = trex::ml_methods::model_selection;

    double lambda_glmnet = 0.0;
    switch (gvs_ctrl_.lambda2_method) {
        case LambdaSelectionMethod::GCV: {
            ms::ridge_gcv gcv;
            gcv.fit(*X_, y_);
            lambda_glmnet = gcv.gcv_optimal();
            break;
        }
        case LambdaSelectionMethod::CV_MIN: {
            ms::ridge_cv cv;
            cv.fit(*X_, y_,
                   gvs_ctrl_.cv_n_folds,
                   gvs_ctrl_.cv_n_lambda,
                   /*lambda_ratio=*/1000.0,
                   resolved_cv_seed_);
            lambda_glmnet = cv.cv_min();
            break;
        }
        case LambdaSelectionMethod::CV_1SE: {
            // Selects the largest lambda within 1 SE of the CV minimum — more regularised than
            // CV_MIN. See the block comment above for the expected high-TPR / controlled-FDR
            // behaviour this induces on high-correlation DGPs.
            ms::ridge_cv cv;
            cv.fit(*X_, y_,
                   gvs_ctrl_.cv_n_folds,
                   gvs_ctrl_.cv_n_lambda,
                   /*lambda_ratio=*/1000.0,
                   resolved_cv_seed_);
            lambda_glmnet = cv.cv_1se();
            break;
        }
        default:
            throw std::invalid_argument(
                "TRexGVSSelector::computeLambda2: unknown LambdaSelectionMethod.");
    }

    return lambda_glmnet * static_cast<double>(p_) / 2.0;
}


// ===================================================================================
// IEN: build augmented X_aug, y_aug, and reusable bottom blocks
// ===================================================================================

void TRexGVSSelector::buildIENAugmentation() {

    const auto n     = static_cast<Eigen::Index>(n_);
    const auto p     = static_cast<Eigen::Index>(p_);
    const auto M     = static_cast<Eigen::Index>(gvs_setup_.max_clusters);
    const auto n_aug = n + M;
    const double sqrt_lambda2 = std::sqrt(lambda2_);

    // ---- ien_bottom_block_p_ : (M x p) row m = (sqrt(lambda2) / sqrt(p_m)) * 1_m -----
    ien_bottom_block_p_ = Eigen::MatrixXd::Zero(M, p);
    for (std::size_t m = 0; m < gvs_setup_.max_clusters; ++m) {
        const double scale =
            sqrt_lambda2 / std::sqrt(static_cast<double>(gvs_setup_.cluster_sizes[m]));
        for (auto j : gvs_setup_.clusters_list[m]) {
            ien_bottom_block_p_(static_cast<Eigen::Index>(m), j) = scale;
        }
    }

    // ---- X_aug_ : (n + M) x p ------------------------------------------
    X_aug_.resize(n_aug, p);
    X_aug_.topRows(n) =  *X_;                       // copy normalized X
    X_aug_.bottomRows(M) = ien_bottom_block_p_;  // copy bottom block

    // ---- y_aug_ : (n + M) x 1, [y; 0_M] ---------------------------------
    y_aug_.resize(n_aug);
    y_aug_.head(n) = y_;
    y_aug_.tail(M).setZero();

    // ---- Per-column centering + unit-L2 normalization over (n + M) rows.
    // External counterpart of R's scale() step in .lm_dummy_gvs().
    // This restores within-cluster scale homogeneity (each column's bottom-
    // block contribution sqrt(lambda2 / p_m) would otherwise inflate the
    // column L2 by a p_m-dependent factor).
    // y_aug_ mean is ~0 to floating-point precision; subtract for parity.
    normalizeAugmentedColumns(X_aug_, eps_);
    y_aug_.array() -= y_aug_.mean();
}


// ===================================================================================
// Per-column centering + unit-L2 normalization (in-place)
// ===================================================================================

template <typename Derived>
void TRexGVSSelector::normalizeAugmentedColumns(Eigen::MatrixBase<Derived>& M, double eps) {
    const Eigen::Index n_rows = M.rows();
    if (n_rows == 0) return;
    const double inv_n = 1.0 / static_cast<double>(n_rows);

    for (Eigen::Index j = 0; j < M.cols(); ++j) {
        // Center.
        const double mean_j = M.col(j).sum() * inv_n;
        M.col(j).array() -= mean_j;

        // Unit-L2 normalize.
        const double norm_j = M.col(j).norm();
        if (norm_j < eps) {
            throw std::runtime_error(
                "TRexGVSSelector::normalizeAugmentedColumns: column " +
                std::to_string(j) +
                " has L2 norm below eps after centering (degenerate column).");
        }
        M.col(j) /= norm_j;
    }
}


// ===================================================================================
// Cluster-aware MVN dummy draw
// ===================================================================================

Eigen::MatrixXd TRexGVSSelector::drawClusterDummyLayer(std::mt19937& rng) const {

    const auto n = static_cast<Eigen::Index>(n_);
    const auto p = static_cast<Eigen::Index>(p_);

    Eigen::MatrixXd layer(n, p);

    std::normal_distribution<double> N01(0.0, 1.0);

    for (std::size_t m = 0; m < gvs_setup_.max_clusters; ++m) {

        const auto& cols = gvs_setup_.clusters_list[m];
        const auto p_m = static_cast<Eigen::Index>(cols.size());

        // Z ~ N(0, I) of shape (n x p_m).
        Eigen::MatrixXd Z(n, p_m);
        for (Eigen::Index j = 0; j < p_m; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) {
                Z(i, j) = N01(rng);
            }
        }

        // block = Z * L_m^T  (rows of block ~ N(0, Sigma_m)).
        Eigen::MatrixXd block = Z * gvs_setup_.cholesky_lower_list[m].transpose();

        // Scatter into output columns.
        for (Eigen::Index j = 0; j < p_m; ++j) {
            layer.col(cols[static_cast<std::size_t>(j)]) = block.col(j);
        }
    }

    return layer;
}


// ===================================================================================
// Assemble per-experiment dummy matrix D_k
// ===================================================================================

Eigen::MatrixXd TRexGVSSelector::assembleD(std::size_t k,
                                           std::size_t num_dummies) const {

    const auto n     = static_cast<Eigen::Index>(n_);
    const auto p     = static_cast<Eigen::Index>(p_);
    const auto L     = static_cast<Eigen::Index>(num_dummies);
    const auto M     = static_cast<Eigen::Index>(gvs_setup_.max_clusters);
    const bool ien   = (gvs_ctrl_.gvs_type == GVSType::IEN);
    const auto n_eff = ien ? (n + M) : n;

    Eigen::MatrixXd D(n_eff, L);

    // Top n rows: horzcat of the cached dummy layers for experiment k.
    const auto& layers = dummy_layers_[k];
    Eigen::Index col = 0;
    for (const auto& layer : layers) {
        D.block(0, col, n, p) = layer;
        col += p;
    }

    // Bottom M rows for IEN: replicated bottom block, one per layer.
    if (ien) {
        col = 0;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            D.block(n, col, M, p) = ien_bottom_block_p_;
            col += p;
        }
    }

    return D;
}


// ===================================================================================
// Assemble per-experiment dummy matrix directly into a mmap'd buffer
// ===================================================================================

void TRexGVSSelector::writeAssembledDIntoMap(
    std::size_t k,
    std::size_t num_dummies,
    Eigen::Map<Eigen::MatrixXd>& dst) const
{
    const auto n   = static_cast<Eigen::Index>(n_);
    const auto p   = static_cast<Eigen::Index>(p_);
    const auto M   = static_cast<Eigen::Index>(gvs_setup_.max_clusters);
    const bool ien = (gvs_ctrl_.gvs_type == GVSType::IEN);

    if (dst.cols() != static_cast<Eigen::Index>(num_dummies)) {
        throw std::runtime_error(
            "TRexGVSSelector::writeAssembledDIntoMap: column-count mismatch.");
    }
    if (dst.rows() != (ien ? (n + M) : n)) {
        throw std::runtime_error(
            "TRexGVSSelector::writeAssembledDIntoMap: row-count mismatch.");
    }

    // Top n rows: horzcat of cached dummy layers for experiment k.
    const auto& layers = dummy_layers_[k];
    Eigen::Index col = 0;
    for (const auto& layer : layers) {
        dst.block(0, col, n, p) = layer;
        col += p;
    }

    // Bottom M rows for IEN: replicated bottom block, one per layer.
    if (ien) {
        col = 0;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            dst.block(n, col, M, p) = ien_bottom_block_p_;
            col += p;
        }
    }
}


// ===================================================================================
// extractPhiContribFromPath -- accumulate per-experiment phi_T contribution
// ===================================================================================

Eigen::MatrixXd TRexGVSSelector::extractPhiContribFromPath(
    const Eigen::MatrixXd& beta_path,
    std::size_t            p,
    std::size_t            num_dummies,
    std::size_t            T_stop,
    double                 eps,
    bool                   verbose)
{
    const auto p_idx = static_cast<Eigen::Index>(p);
    const auto T_idx = static_cast<Eigen::Index>(T_stop);
    const auto L     = static_cast<Eigen::Index>(num_dummies);
    const auto n_steps      = beta_path.cols();

    Eigen::MatrixXd contrib = Eigen::MatrixXd::Zero(p_idx, T_idx);

    if (L == 0 || n_steps == 0) return contrib;

    // For each t in [1, T_stop], find the FIRST column c where exactly t
    // dummy variables (rows [p, p+L)) are non-zero.  At that step, every
    // active original-feature row (rows [0, p)) gets a +1 contribution.
    for (std::size_t t = 1; t <= T_stop; ++t) {

        Eigen::Index found_col = -1;
        for (Eigen::Index c = 0; c < n_steps; ++c) {
            const Eigen::Index n_active_dummies = (beta_path.block(
                p_idx, c,
                L, 1).array().abs() > eps).count();

            if (n_active_dummies == static_cast<Eigen::Index>(t)) {
                found_col = c;
                break;
            }
        }

        if (found_col < 0) {
            if (verbose) {
                // Path never reached t active dummies; row stays 0.
            }
            continue;
        }

        // Add indicator of active original features at that column.
        for (Eigen::Index j = 0; j < p_idx; ++j) {
            if (std::abs(beta_path(j, found_col)) > eps) {
                contrib(j, static_cast<Eigen::Index>(t - 1)) += 1.0;
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
    const bool ien                           = (gvs_ctrl_.gvs_type == GVSType::IEN);

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

            Eigen::MatrixXd beta_path;

            if (use_warm_start && solvers_cache_[k] != nullptr) {
                // -- Warm-start branch: solver already bound to *D_solver_maps_[k]. --
                solvers_cache_[k]->executeStep(T_stop, /*early_stop=*/true);
                beta_path = solvers_cache_[k]->getBetaPath();

            } else {
                // -- Fresh construction branch: D_solver_maps_[k] was bound by
                //    prepareDummiesForLStep (in-memory or mmap). --
                std::unique_ptr<tsolvers::TSolver_Base> solver;
                if (ien) {
                    solver = std::make_unique<lars::TLASSO_Solver>(
                        *X_solver_map_,
                        *D_solver_maps_[k],
                        *y_solver_map_,
                        /*normalize=*/false,
                        /*intercept=*/false,
                        /*verbose=*/false);
                } else {
                    solver = std::make_unique<lars::TENET_Solver>(
                        *X_solver_map_,
                        *D_solver_maps_[k],
                        *y_solver_map_,
                        lambda2_,
                        /*normalize=*/false,
                        /*intercept=*/false,
                        /*verbose=*/false);
                }

                solver->executeStep(T_stop, /*early_stop=*/true);
                beta_path = solver->getBetaPath();
                solvers_cache_[k] = std::move(solver);
            }

            // Per-experiment phi_T contribution (independent slot, no race).
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
    const auto        n_eff_idx = static_cast<Eigen::Index>(n_eff_);

    // Three of the four supported strategies follow the redraw-all path.
    // HCONCAT is the only append-only strategy. PERMUTATION variants are
    // rejected in the constructor and trigger a defensive throw here.
    bool redraw_all = false;
    switch (trex_ctrl_.lloop_strategy) {
        case tc::LLoopStrategy::STANDARD:
        case tc::LLoopStrategy::SKIPL:
        case tc::LLoopStrategy::DIRECT:
            redraw_all = true;
            break;
        case tc::LLoopStrategy::HCONCAT:
            redraw_all = false;
            break;
        case tc::LLoopStrategy::PERMUTATION:
        case tc::LLoopStrategy::PERMUTATION_DIRECT:
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

        std::mt19937 rng(trex::utils::datageneration::dummygen::mix_seed(
            trex::utils::datageneration::dummygen::mix_seed(
                static_cast<std::uint32_t>(base_seed), k), LL));

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

    // ---- (Re)build solver-side D buffers + invalidate solver cache -------
    if (gvs_use_mmap_) {
        // Memory-mapped path: write directly into the per-K mmap buffers
        // pre-allocated in onSelectBegin().
        D_solver_bufs_.clear();
        D_solver_maps_.clear();
        D_solver_maps_.resize(K);
        const auto num_d_idx = static_cast<Eigen::Index>(ctx.num_dummies);
        for (std::size_t k = 0; k < K; ++k) {
            // getDMap returns an Eigen::Map view onto the first
            // `num_dummies` columns of file k (n_eff x num_dummies).
            auto dmap = memmap_mgr_->getDMap(k, ctx.num_dummies);
            if (dmap.rows() != n_eff_idx || dmap.cols() != num_d_idx) {
                throw std::runtime_error(
                    "TRexGVSSelector::prepareDummiesForLStep: "
                    "mmap returned unexpected shape.");
            }
            writeAssembledDIntoMap(k, ctx.num_dummies, dmap);
            normalizeAugmentedColumns(dmap, eps_);
            // Bind the long-lived Map for runKExperiments. The owning
            // storage is the mmap file managed by memmap_mgr_.
            D_solver_maps_[k] = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
                dmap.data(), dmap.rows(), dmap.cols());
        }
    } else {
        // In-memory path.
        D_solver_bufs_.assign(K, Eigen::MatrixXd(
            n_eff_idx, static_cast<Eigen::Index>(ctx.num_dummies)));
        D_solver_maps_.clear();
        D_solver_maps_.resize(K);
        for (std::size_t k = 0; k < K; ++k) {
            D_solver_bufs_[k] = assembleD(k, ctx.num_dummies);
            // Per-column centering + unit-L2 normalization over n_eff rows
            // (external counterpart of R's scale() on the augmented system).
            // Re-applied every L-iteration regardless of strategy: solver cache
            // is cleared each L-iteration, so the cost is amortised.
            normalizeAugmentedColumns(D_solver_bufs_[k], eps_);
            D_solver_maps_[k] = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
                D_solver_bufs_[k].data(),
                D_solver_bufs_[k].rows(),
                D_solver_bufs_[k].cols());
        }
    }
    // Fresh dummy data => fresh solvers at T = 1.
    solvers_cache_.clear();
    solvers_cache_.resize(K);

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

    // 3. Build IEN augmentation if needed; set up effective shapes.
    const bool ien = (gvs_ctrl_.gvs_type == GVSType::IEN);
    if (ien) {
        buildIENAugmentation();
        n_eff_ = n_ + gvs_setup_.max_clusters;

        // Solver-side X and y are the augmented buffers (long-lived members).
        X_solver_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
            X_aug_.data(), X_aug_.rows(), X_aug_.cols());

        y_solver_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(
            y_aug_.data(), y_aug_.size());

    } else { // EN: no augmentation, solver sees original X and y.
        n_eff_ = n_;

        X_solver_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
            X_->data(), X_->rows(), X_->cols());

        y_solver_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(
            y_.data(), y_.size());
    }

    // 4. Allocate per-K memory-mapped D files at max size if requested.
    //    shared=false: GVS draws per-experiment distinct MVN dummies, so
    //    each k requires its own backing file (also a prerequisite for the
    //    parallel-K loop in runKExperiments).
    if (gvs_use_mmap_) {
        const std::size_t max_dummies = trex_ctrl_.max_dummy_multiplier * p_;
        memmap_mgr_ = std::make_unique<mm::MemmapManager>(
            n_eff_, max_dummies, trex_ctrl_.K, /*shared=*/false, verbose_);
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
    gvs_result_.gvs_type       = gvs_ctrl_.gvs_type;
    gvs_result_.max_clusters   = gvs_setup_.max_clusters;
    gvs_result_.hc_method_used = gvs_setup_.hc_method_used;
    gvs_result_.groups_vec     = gvs_setup_.groups_vec;
    gvs_result_.group_labels   = gvs_setup_.group_labels;

    if (verbose_) {
        printProgress("GVS: Selection complete. Selected " +
                      std::to_string(getNumSelected()) + " variables.\n");
    }

    // GVS-specific cleanup: solver cache + buffers + augmented system.
    // Base `select()` will subsequently run warm_start_mgr_/memmap cleanup
    // and denormalize X.
    solvers_cache_.clear();
    D_solver_bufs_.clear();
    D_solver_maps_.clear();
    X_solver_map_.reset();
    y_solver_map_.reset();
    X_aug_.resize(0, 0);
    y_aug_.resize(0);
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
