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

// GVS header
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

// Ridge GCV (header-only, lambda_2 auto-tuning)
#include <trex_selector_methods/trex_gvs/ridge_gcv.hpp>

// Data normalization helpers
#include <trex_selector_methods/utils/trex_data_normalizer.hpp>

// Solver dispatch
#include <trex_selector_methods/utils/trex_solver_dispatch.hpp>

// Concrete solver types (for direct in-memory warm-start)
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_gvs {

// Local namespace aliases
namespace dn   = trex::trex_selector_methods::utils::data_normalizer;
namespace lars = trex::tsolvers::linear_model::lars_based;


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
    tc::TRexSelector(X, y, tFDR, trex_control, seed, verbose),
    gvs_ctrl_(std::move(gvs_control))
{
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

    // ---- Reject memory mapping (GVS uses in-memory buffers only) ----------
    if (trex_ctrl_.use_memory_mapping) {
        throw std::invalid_argument(
            "TRexGVSSelector: memory mapping is not supported."
        );
    }

    // ---- Reject unsupported L-loop strategies -----------------------------
    if (trex_ctrl_.lloop_strategy != tc::LLoopStrategy::STANDARD &&
        trex_ctrl_.lloop_strategy != tc::LLoopStrategy::HCONCAT) {
        throw std::invalid_argument(
            "TRexGVSSelector: only LLoopStrategy::STANDARD and HCONCAT are supported."
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

    if (gvs_ctrl_.lambda2_lars < 0.0) {
        throw std::invalid_argument(
            "TRexGVSSelector: lambda2_lars must be >= 0. Got: " +
            std::to_string(gvs_ctrl_.lambda2_lars));
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
// lambda_2 (LARS units) -- user supplied or GCV-optimal
// ===================================================================================

double TRexGVSSelector::computeLambda2() const {

    if (gvs_ctrl_.lambda2_lars > 0.0) {
        return gvs_ctrl_.lambda2_lars;
    }

    // Auto-compute via GCV-optimal ridge:
    // Convert from glmnet to LARS units by multiplying by p / 2 (as in R's trex_GVS).
    Eigen::MatrixXd X_dense = *X_; // RidgeGCV::fit takes const MatrixXd&.
    ridge::RidgeGCV gcv;
    gcv.fit(X_dense, y_);
    const double lambda_glmnet = gcv.gcv_optimal();
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
    X_aug_.topRows(n)       = *X_;               // copy normalized X
    X_aug_.bottomRows(M) = ien_bottom_block_p_;  // copy bottom block

    // ---- y_aug_ : (n + M) x 1, [y; 0_M] ---------------------------------
    y_aug_.resize(n_aug);
    y_aug_.head(n) = y_;
    y_aug_.tail(M).setZero();
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

    Eigen::MatrixXd phi_T_acc = Eigen::MatrixXd::Zero(p_idx, T_idx);

    // If we are starting fresh (no warm-start), wipe any leftover solver
    // instances.
    // D_solver_bufs_ is assumed to already be populated by the L-loop driver for this iteration.
    if (!use_warm_start) {
        solvers_cache_.clear();
        solvers_cache_.resize(K);
        D_solver_maps_.clear();
        D_solver_maps_.resize(K);
    }

    for (std::size_t k = 0; k < K; ++k) {

        // Build the Map for this experiment's D buffer ONLY when constructing
        // a fresh solver.  On warm-start the existing solver holds a pointer
        // to the previously created Map (*D_solver_maps_[k]); overwriting
        // that unique_ptr would invalidate the solver's bound reference.
        Eigen::MatrixXd beta_path;

        if (use_warm_start && solvers_cache_[k] != nullptr) {
            // -- Warm-start branch (Map / buffer untouched). --
            solvers_cache_[k]->executeStep(T_stop, /*early_stop=*/true);
            beta_path = solvers_cache_[k]->getBetaPath();

        } else {
            // -- Fresh construction branch: (re)bind the Map first. --
            D_solver_maps_[k] = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
                D_solver_bufs_[k].data(),
                D_solver_bufs_[k].rows(),
                D_solver_bufs_[k].cols());

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

        // Accumulate per-experiment contribution into phi_T_acc.
        phi_T_acc += extractPhiContribFromPath(
            beta_path, p_, num_dummies, T_stop, eps_, verbose_);
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
// L-loop: cluster-aware dummy generation, STANDARD or HCONCAT
// ===================================================================================

void TRexGVSSelector::runGVSLLoop(Eigen::VectorXd& FDP_hat,
                                  Eigen::VectorXd& Phi_out)
{
    const std::size_t K     = trex_ctrl_.K;
    const std::size_t L_max = trex_ctrl_.max_dummy_multiplier;
    const bool standard     = (trex_ctrl_.lloop_strategy == tc::LLoopStrategy::STANDARD);
    const auto p_idx     = static_cast<Eigen::Index>(p_);
    const auto M_idx     = static_cast<Eigen::Index>(gvs_setup_.max_clusters);
    const bool ien                           = (gvs_ctrl_.gvs_type == GVSType::IEN);
    const auto n_eff_idx = static_cast<Eigen::Index>(n_eff_);

    // Initial state.
    dummy_multiplier_LL_ = 1;
    dummy_layers_.assign(K, {});

    // RNG seeded deterministically from seed_; per-experiment unique streams.
    const std::uint64_t base_seed = (seed_ < 0)
        ? static_cast<std::uint64_t>(std::random_device{}())
        : static_cast<std::uint64_t>(seed_);

    if (verbose_) {
        printProgress("GVS L-loop: starting (" +
                      std::string(standard ? "STANDARD" : "HCONCAT") + ")");
    }

    while (dummy_multiplier_LL_ <= L_max &&
           (FDP_hat.size() == 0 || FDP_hat(static_cast<Eigen::Index>(opt_point_)) > tFDR_)) {

        const std::size_t LL = dummy_multiplier_LL_;
        num_dummies_ = LL * p_;

        // ---- Generate / extend per-experiment dummy layers -------------
        for (std::size_t k = 0; k < K; ++k) {

            std::mt19937 rng(static_cast<std::uint32_t>(
                base_seed +
                static_cast<std::uint64_t>(k) * 1000003ULL +
                static_cast<std::uint64_t>(LL)));

            if (standard) {
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

        // ---- (Re)build solver-side D buffers + invalidate solver cache --
        D_solver_bufs_.assign(K, Eigen::MatrixXd(n_eff_idx,
                                                 static_cast<Eigen::Index>(num_dummies_)));
        for (std::size_t k = 0; k < K; ++k) {
            D_solver_bufs_[k] = assembleD(k, num_dummies_);
        }
        // Invalidate solvers (fresh dummy data => fresh solvers at T = 1).
        solvers_cache_.clear();
        solvers_cache_.resize(K);

        // ---- Run the K experiments at T = 1 (no warm-start) ------------
        T_stop_ = 1;
        auto agg = runKExperiments(/*T_stop=*/1, num_dummies_,
                                   /*use_warm_start=*/false);

        // Compute Phi_prime, FDP_hat from this iteration's K aggregate.
        Phi_prime_ = computePhiPrime(agg.phi_T_mat, agg.Phi, num_dummies_);
        FDP_hat    = computeFDPHat(voting_grid_, agg.Phi, Phi_prime_);
        Phi_out    = agg.Phi;

        if (verbose_) {
            std::ostringstream oss;
            oss << "GVS L = " << std::setw(2) << LL
                << " | num_dummies = " << num_dummies_
                << " | FDP@opt = " << FDP_hat(static_cast<Eigen::Index>(opt_point_));
            printProgress(oss.str());
        }

        // The exp_results we return through the seed StepView for runTLoop:
        // not stored explicitly here; runTLoop will call evaluateStep(2, ...)
        // which will warm-start from solvers_cache_ already populated by the
        // last runKExperiments call above.

        ++dummy_multiplier_LL_;

        // Suppress unused-variable warning when assertions disabled.
        (void)p_idx; (void)M_idx; (void)ien;
    }

    // Roll back: dummy_multiplier_LL_ was incremented past the last applied LL.
    if (dummy_multiplier_LL_ > 1) --dummy_multiplier_LL_;

    if (verbose_) {
        printProgress("GVS L-loop: converged at LL = " +
                      std::to_string(dummy_multiplier_LL_) + "\n");
    }
}


// ===================================================================================
// select() -- main entry point
// ===================================================================================

TRexGVSSelector::SelectionResult TRexGVSSelector::select() {

    // 1. Voting grid + opt point.
    voting_grid_ = createVotingGrid();
    const std::size_t v_len = voting_grid_.size();
    setPercentOptPoint(v_len, trex_ctrl_.opt_threshold);

    // 2. Group structure (Sigma_m, L_m, IEN_cl_id_vectors).
    if (verbose_) { printProgress("GVS: Setting up group structure..."); }
    setupGVS();

    if (verbose_) {
        printProgress("GVS: Method = " + gvs_setup_.hc_method_used +
                      ", M = " + std::to_string(gvs_setup_.max_clusters));
    }

    // 3. lambda_2 (LARS units).
    lambda2_ = computeLambda2();
    if (verbose_) {
        std::ostringstream oss;
        oss << "GVS: lambda_2 (LARS units) = " << lambda2_;
        printProgress(oss.str());
    }

    // 4. Build IEN augmentation if needed; set up effective shapes.
    const bool ien = (gvs_ctrl_.gvs_type == GVSType::IEN);
    if (ien) {
        buildIENAugmentation();
        n_eff_ = n_ + gvs_setup_.max_clusters;

        // Solver-side X and y are the augmented buffers (long-lived members).
        X_solver_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
            X_aug_.data(), X_aug_.rows(), X_aug_.cols());

        y_solver_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(
            y_aug_.data(), y_aug_.size());

    } else { // EN variant: no augmentation, solver sees original X and y.
        n_eff_ = n_;

        // Solver-side X and y view directly into the base buffers.
        X_solver_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(
            X_->data(), X_->rows(), X_->cols());

        y_solver_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(
            y_.data(), y_.size());
    }

    // 5. L-loop (overrides inherited runLLoop).
    Eigen::VectorXd FDP_hat;
    Eigen::VectorXd Phi_seed;
    runGVSLLoop(FDP_hat, Phi_seed);

    // 6. T-loop -- reuse the inherited driver via overridden evaluateStep.
    //    Need to seed exp_results so runTLoop can populate row 0.
    tc::er::ExperimentResults exp_results;
    exp_results.Phi = Phi_seed;
    exp_results.phi_T_mat.resize(static_cast<Eigen::Index>(p_), 1);
    exp_results.phi_T_mat.col(0) = Phi_seed;
    runTLoop(FDP_hat, exp_results);

    // 7. Variable selection (base helper).
    auto base_result =
        selectedVariables(FDP_hat_mat_, Phi_mat_,
                          voting_grid_, T_stop_);

    // 8. Assemble GVS result + cleanup + denormalize.
    return assembleGVSResult(Phi_mat_,
                             base_result.selected_var.cast<double>().eval());

    // (We pass the final phi/Phi via gvs_result_ inside assembleGVSResult below;
    //  parameters here are placeholders -- see implementation.)
}


// ===================================================================================
// Result assembly + cleanup + denormalization
// ===================================================================================

TRexGVSSelector::SelectionResult TRexGVSSelector::assembleGVSResult(
    const Eigen::MatrixXd& /*final_phi_T_mat*/,
    const Eigen::VectorXd& /*final_Phi*/)
{
    // Re-derive base-style selection from the accumulator matrices.
    auto base = selectedVariables(
        FDP_hat_mat_, Phi_mat_, voting_grid_, T_stop_);

    // Populate base fields.
    gvs_result_.selected_var   = base.selected_var;
    gvs_result_.v_thresh       = base.v_thresh;
    gvs_result_.R_mat          = base.R_mat;
    gvs_result_.T_stop         = T_stop_;
    gvs_result_.num_dummies    = num_dummies_;
    gvs_result_.dummy_factor_L = dummy_multiplier_LL_;
    gvs_result_.Phi_prime      = Phi_prime_;
    gvs_result_.voting_grid    = voting_grid_;
    gvs_result_.FDP_hat_mat    = FDP_hat_mat_;
    gvs_result_.Phi_mat        = Phi_mat_;

    // GVS-specific fields.
    gvs_result_.lambda2_used   = lambda2_;
    gvs_result_.gvs_type       = gvs_ctrl_.gvs_type;
    gvs_result_.max_clusters   = gvs_setup_.max_clusters;
    gvs_result_.hc_method_used = gvs_setup_.hc_method_used;
    gvs_result_.groups_vec     = gvs_setup_.groups_vec;
    gvs_result_.group_labels   = gvs_setup_.group_labels;

    // Persist into base accessors.
    storeResults(gvs_result_);

    if (verbose_) {
        printProgress("GVS: Selection complete.  Selected " +
                      std::to_string(getNumSelected()) + " variables.\n");
    }

    // Cleanup: drop solver cache + buffers, then denormalize X.
    solvers_cache_.clear();
    D_solver_bufs_.clear();
    D_solver_maps_.clear();
    X_solver_map_.reset();
    y_solver_map_.reset();
    X_aug_.resize(0, 0);
    y_aug_.resize(0);
    dummy_layers_.clear();

    if (X_is_normalized_) {
        if (verbose_) { printProgress("GVS: Denormalizing X to original scale.\n"); }
        dn::denormalizeX(*X_, norm_params_);
        X_is_normalized_ = false;
    }

    // Return a base-sliced SelectionResult (R API expects it).
    SelectionResult out;
    out.selected_var   = gvs_result_.selected_var;
    out.v_thresh       = gvs_result_.v_thresh;
    out.R_mat          = gvs_result_.R_mat;
    out.T_stop         = gvs_result_.T_stop;
    out.num_dummies    = gvs_result_.num_dummies;
    out.dummy_factor_L = gvs_result_.dummy_factor_L;
    out.Phi_prime      = gvs_result_.Phi_prime;
    out.voting_grid    = gvs_result_.voting_grid;
    out.FDP_hat_mat    = gvs_result_.FDP_hat_mat;
    out.Phi_mat        = gvs_result_.Phi_mat;
    return out;
}


// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_gvs */
// ===================================================================================
