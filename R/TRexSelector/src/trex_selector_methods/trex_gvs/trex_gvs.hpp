// ===================================================================================
// trex_gvs.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_GVS_HPP
#define TREX_SELECTOR_METHODS_TREX_GVS_HPP
// ===================================================================================
/**
 * @file trex_gvs.hpp
 *
 * @brief Grouped Variable Selection T-Rex (T-Rex+GVS) Selector.
 *
 * @details
 *  Extends the base class TRexSelector with cluster-aware multivariate-normal
 *  dummy generation and one of two augmentation policies:
 *
 *    - GVSType::EN    Ordinary Elastic Net (Zou & Hastie, 2005) & (Machkour et al., 2022).
 *                     Cluster-aware MVN dummies are passed to the solver.
 *                     The L2 penalty is handled by the terminating solver itself via its
 *                     `lambda2` hyperparameter.
 *                     The required solver is the T-Elastic-Net (TENET).
 *
 *    - GVSType::IEN   Informed Elastic Net (Machkour et al., CAMSAP 2023).
 *                     NOTE:The IEN requires a dedicated solver algorithm currently not available.
 *                     Thus we form the augmented Lasso-type system explicitly
 *                     and feed it to the T-LASSO solver:
 *                       X_aug = [ X            ]   (n  rows)
 *                               [ B            ]   (M  rows)
 *                                with row m = (sqrt(lambda2) /sqrt(p_m)) * 1_m
 *
 *                       D_aug = [ D            ]   (n  rows)
 *                               [ B repeated   ]   (M  rows, one (sqrt(lambda2) /sqrt(p_m)) * 1_m
 *                                                  segment per dummy layer)
 *                       y_aug = [ y ; 0_M    ]
 *
 *                     The L2 penalty is fully absorbed into the augmented
 *                     system (Theorem 1 in Machkour et al. 2023).
 *
 *  Design notes:
 *  ------------
 *    - GVS does NOT use the inherited:
 *      DummyGenerator or ExperimentRunner.
 *      It owns a private K-experiment loop, a private per-experiment
 *      dummy-layer cache, and (for IEN) a private warm-start
 *      cache of in-memory T-LASSO solvers. The inherited
 *      `MemmapManager` IS reused (re-allocated in `onSelectBegin()` with
 *      the GVS-specific `n_eff` row count, and torn down by the base
 *      class at the end of `select()`); see `gvs_use_mmap_`.
 *
 *    - The L-loop honors LLoopStrategy::STANDARD, HCONCAT, SKIPL, and
 *      DIRECT.
 *      STANDARD and DIRECT are equivalent inside GVS (the overridden
 *      `evaluateStep` consumes `D_solver_bufs_` rather than streaming dummies from disk).
 *      SKIPL collapses to a single L-iter with LL = max_dummy_multiplier.
 *      PERMUTATION and PERMUTATION_DIRECT are rejected because row permutation
 *      would destroy the per-cluster MVN covariance structure that GVS dummies
 *      are designed to mirror.
 *
 *    - Memory mapping is supported uniformly across all four allowed L-loop strategies.
 *      When enabled, K independent (non-shared) D files of size
 *.     `n_eff x (max_dummy_multiplier * p)` are allocated up-front in `onSelectBegin()`
 *      and the per-K columns consumed by the L-loop are filled in place by
 *      `writeAssembledDIntoMap`.
 *
 *    - The lambda_2 ridge parameter is either user-supplied
 *      (TRexGVSControlParameter::lambda_2 > 0) or computed once via the
 *      Generalized Cross-Validation (GCV) optimum on the original (X, y),
 *      scaled by p / 2 to convert from glmnet to LARS units.
 *
 *  Indexing convention:
 *    - All cluster IDs and column indices are 0-based throughout the C++
 *      API. Users coming from R must pass `groups - 1L`.
 *
 *  References:
 *  -----------
 *   Zou & Hastie (2005): "Regularization and variable selection via the
 *      elastic net", JRSSB 67(2), 301-320.
 *
 *   Machkour, Muma, Palomar (2022): "False Discovery Rate Control for Grouped Variable
 *     Selection in High-Dimensional Linear Models Using the T-Knock Filter",
 *     pp.892-896, EUSIPCO 2022.
 *
 *   Machkour, Muma, Palomar (2023): "The Informed Elastic Net for Fast Grouped Variable
 *     Selection and FDR Control in Genomics Research", pp. 466 - 470, CAMSAP 2023.
 *
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex base class
#include <trex_selector_methods/trex_core/trex.hpp>

// Solver dispatch (constants + SolverConfig)
#include <trex_selector_methods/trex_utils/trex_solver_dispatch.hpp>

// Hierarchical clustering (Route 2 -- auto-cluster from |cor(X)|)
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>

// Solver base for warm-start cache
#include <tsolvers/tsolver_base.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_gvs {

// Namespace aliases
namespace tc  = trex::trex_selector_methods::trex_core;
namespace sd  = trex::trex_selector_methods::utils::solver_dispatch;
namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;

// ===================================================================================
// Enums
// ===================================================================================

/**
 * @brief Variable selection policy for T-Rex+GVS.
 */
enum class GVSType {
    /** @brief Ordinary Elastic Net  (no data augmentation + TENET solver). */
    EN,
    /** @brief Informed Elastic Net (row-augmented system + TLASSO solver). */
    IEN,
    /** @brief Augmented EN LASSO: zero-padded X, diagonal identity block in D.
     *  Supports TLASSO (correct) and TLARS (semi-correct, matches R reference).
     *  n_eff is fixed at n + max_dummy_multiplier * p; zero-padded rows are
     *  algebraically transparent to LARS/LASSO inner products. */
    EN_AUG
};


/**
 * @brief Selection rule for the ridge regularisation parameter `lambda_2`
 *        when no user-supplied value is given.
 *
 * @details
 *   - `CV_1SE`  : k-fold cross-validation, glmnet-style one-standard-error
 *                 rule (largest lambda whose CV-MSE is within one SE of
 *                 the CV-min, biased toward stronger regularisation).
 *                 Default: produces higher TPR at controlled FDR on
 *                 high-correlation DGPs.
 *   - `CV_MIN`  : k-fold cross-validation, lambda minimising mean MSE.
 *   - `GCV`     : minimise generalised cross-validation criterion
 *                 (closed-form via SVD; fast but tends toward the Lasso
 *                 solution on low-noise problems).
 */
enum class LambdaSelectionMethod {
    CV_1SE,
    CV_MIN,
    GCV
};


// ===================================================================================
// Control Parameters
// ===================================================================================

/**
 * @brief GVS-specific control parameters.
 */
struct TRexGVSControlParameter {

    /** @brief GVS selection policy algorithm (default: EN). */
    GVSType gvs_type = GVSType::EN;

    /** @brief Maximum allowed pairwise correlation between variables from
     *  different clusters (only when groups are not provided).
     *  Default: 0.5.
     */
    double corr_max = 0.5;

    /** @brief Linkage method for hierarchical clustering.
     *  Supported: Single (default), Complete, Average, WPGMA.
     *  All other linkage methods are rejected bcauase:
     *  Ward is rejected because the d = 1 - |cor| metric is non-Euclidean.
     *  Mean and Median centroids are non-reducible.
     */
    hac::LinkageMethod hc_linkage = hac::LinkageMethod::Single;

    /** @brief User-supplied lambda_2 in LARS units.
     *  Default is 0.0, which triggers an auto-computation according to
     *  `lambda2_method` (see below). The auto-computed glmnet-units lambda
     *  is then scaled by p / 2 to obtain LARS units (matches R's
     *  `trex_GVS()` convention).
     */
    double lambda_2 = 0.0;

    /** @brief Selection rule for auto-computing `lambda_2` when
     *  `lambda_2 == 0`. Default: CV_1SE.
     */
    LambdaSelectionMethod lambda2_method = LambdaSelectionMethod::CV_1SE;

    /** @brief Number of folds for k-fold cross-validation when
     *  `lambda2_method` is `CV_MIN` or `CV_1SE`. Default: 10
     *  (matches R's `cv.glmnet`). Ignored when `lambda2_method == GCV`.
     */
    int cv_n_folds = 10;

    /** @brief Number of points on the geometric lambda grid searched by
     *  the auto-computation. Default: 100. Applies to both GCV and CV
     *  paths.
     */
    Eigen::Index cv_n_lambda = 100;

    /** @brief Seed for the deterministic fold-permutation RNG when
     *  `lambda2_method` is `CV_MIN` or `CV_1SE`.
     *
     *  -  `-1` (default): derived from the T-Rex `seed` parameter —
     *      deterministic when `seed >= 0`, hardware-entropy when `seed < 0`.
     *  - `>= 0`: explicit fold seed; overrides `seed` for CV only.
     */
    int cv_seed = -1;

    /** @brief Optional prior cluster assignment (length p, 0-based, contiguous
     *  IDs in [0, M-1]). Empty (default) triggers Route 2 (auto-cluster).
     */
    std::vector<Eigen::Index> prior_groups;

    /** @brief Optional human-readable cluster names; size must equal the
     *  number of unique cluster IDs (M). Empty (default) is allowed.
     */
    std::vector<std::string> group_labels;
};


// ===================================================================================
// GVS Setup Result (internal, populated once per select())
// ===================================================================================

/**
 * @brief Pre-computed group structure. Contains deterministic functions of X (or of
 *        the user-supplied prior groups).
 */
struct GVSSetupResult {

    /** @brief Per-cluster column indices into X (0-based). Length M. */
    std::vector<std::vector<Eigen::Index>> clusters_list;

    /** @brief Per-cluster size p_m = clusters_list[m].size(). Length M. */
    std::vector<std::size_t> cluster_sizes;

    /** @brief Per-cluster lower-triangular Cholesky factor L_m of Sigma_m.
     *  Pre-computed once so that MVN draws cost only one matrix-multiplication.
     */
    std::vector<Eigen::MatrixXd> cholesky_lower_list;

    /** @brief (M x p) binary support matrix.
     *  Row m is the indicator vector 1_m of cluster m.
     */
    Eigen::MatrixXd IEN_cl_id_vectors;

    /** @brief Number of clusters M. */
    std::size_t max_clusters = 0;

    /** @brief Group labels carried through from the control parameters.
     *  Allowed to be empty.
     */
    std::vector<std::string> group_labels;

    /** @brief Linkage method used; "prior" if Route 1. */
    std::string hc_method_used;

    /** @brief 0-based cluster assignment per variable, length p. */
    Eigen::VectorXi groups_vec;
};


// ===================================================================================
// TRexGVSSelector
// ===================================================================================

/**
 * @brief Grouped Variable Selection T-Rex Selector.
 *
 * @details
 *  Inherits from TRexSelector. The base `select()` orchestration is reused;
 *  GVS plugs in via the variant hooks:
 *  `onSelectBegin()`,
 *  `prepareDummiesForLStep()`, `evaluateStep()`, and
 *  `finalizeSelectionResult()`.
 *  The inherited `DummyGenerator` and
 *  `ExperimentRunner` are bypassed (GVS uses cluster-aware MVN dummy
 *  generation, an in-memory K-experiment loop, and an in-memory solver
 *  warm-start cache). The inherited `MemmapManager` IS reused when the
 *  user requests memory mapping; see `gvs_use_mmap_`.
 */
class TRexGVSSelector : public tc::TRexSelector {

public:

    // ============================================================
    // Result Structure
    // ============================================================

    /**
     * @brief Extended result for T-Rex+GVS, adding group / lambda diagnostics.
     */
    struct GVSSelectionResult : public SelectionResult {

        /** @brief lambda_2 actually used (resolved from user or GCV). */
        double lambda2_used = 0.0;

        /** @brief Augmentation policy that was applied. */
        GVSType gvs_type = GVSType::IEN;

        /** @brief Number of clusters M. */
        std::size_t max_clusters = 0;

        /** @brief Linkage method used; "prior" if Route 1. */
        std::string hc_method_used;

        /** @brief 0-based cluster assignment per variable (length p). */
        Eigen::VectorXi groups_vec;

        /** @brief Group labels (may be empty if not supplied). */
        std::vector<std::string> group_labels;

        /** @brief Virtual destructor. */
        ~GVSSelectionResult() override = default;
    };


    // ============================================================
    // Constructor / Destructor
    // ============================================================

    /**
     * @brief Construct a TRexGVSSelector.
     *
     * @param X            Feature matrix (n x p), Eigen::Map, not copied.
     * @param y            Response vector (n x 1), Eigen::Map, copied internally.
     * @param tFDR         Target FDR level (default: 0.1).
     * @param gvs_control  GVS-specific control parameters.
     * @param trex_control Base T-Rex algorithmic control parameters.
     * @param seed         Random seed (< 0 for random).
     * @param verbose      Enable verbose output (default: true).
     *
     * @throws std::invalid_argument on incompatible solver / strategy choice
     *         or malformed prior_groups.
     */
    TRexGVSSelector(
        Eigen::Map<Eigen::MatrixXd>& X,
        Eigen::Map<Eigen::VectorXd>& y,
        double                       tFDR         = 0.1,
        TRexGVSControlParameter      gvs_control  = TRexGVSControlParameter(),
        tc::TRexControlParameter     trex_control = tc::TRexControlParameter(),
        int                          seed         = -1,
        bool                         verbose      = true
    );

    /** @brief Virtual destructor. */
    ~TRexGVSSelector() override = default;

    /** @brief Deleted copy constructor. */
    TRexGVSSelector(const TRexGVSSelector&) = delete;

    /** @brief Deleted copy assignment operator. */
    TRexGVSSelector& operator=(const TRexGVSSelector&) = delete;

    /** @brief Deleted move constructor. */
    TRexGVSSelector(TRexGVSSelector&&) = delete;

    /** @brief Deleted move assignment operator. */
    TRexGVSSelector& operator=(TRexGVSSelector&&) = delete;


    // ============================================================
    // Main Method
    // ============================================================

    /**
     * @brief Retrieve the full GVS result from the last `select()` call.
     *
     * @details `select()` itself is inherited from `TRexSelector`; this
     * accessor exposes the widened GVS-specific result populated by
     * `finalizeSelectionResult()`.
     */
    const GVSSelectionResult& getGVSResult() const noexcept { return gvs_result_; }


protected:

    // ============================================================
    // GVS Data Members
    // ============================================================

    /** @brief GVS-specific control parameters. */
    TRexGVSControlParameter gvs_ctrl_;

    /** @brief Pre-computed group structure. */
    GVSSetupResult gvs_setup_;

    /** @brief Resolved lambda_2 in LARS units (> 0 after computeLambda2()). */
    double lambda2_{0.0};

    /** @brief Resolved CV fold-permutation seed.
     *
     *  Computed once in the constructor from `gvs_ctrl_.cv_seed` and `seed_`:
     *  - `cv_seed >= 0`: used directly (explicit override).
     *  - `cv_seed < 0 && seed_ >= 0`: `mix_seed(seed_, 1u)` (deterministic).
     *  - `cv_seed < 0 && seed_ < 0`:  `std::random_device{}()` (entropy).
     */
    unsigned int resolved_cv_seed_{0};

    /** @brief Full GVS result of the last select() call. */
    GVSSelectionResult gvs_result_;

    // ----- Per-experiment dummy-layer cache -----

    /** @brief Per-experiment list of MVN-drawn dummy layers.
     *  Layout: dummy_layers_[k][layer_idx] is (n x p).
     *  STANDARD strategy clears and refills;
     *  HCONCAT only appends.
     */
    std::vector<std::vector<Eigen::MatrixXd>> dummy_layers_;

    // ----- IEN-only augmented system (row-augmented) -----

    /** @brief Augmented predictor matrix X_aug ((n + M) x p).
     *  IEN: holds [X ; bottom_block] after L2-column-normalization.
     *  EN : empty.
     */
    Eigen::MatrixXd X_aug_;

    /** @brief Augmented response y_aug (n + M).
     *  IEN only.
     *  EN: empty.
     */
    Eigen::VectorXd y_aug_;

    /** @brief Augmented predictor matrix X_aug_en_ ((n + p) x p).
     *  EN_AUG only. Holds the Zou & Hastie (2005) augmentation
     *  [X ; sqrt(lambda2)*I_p], column-normalized to unit-L2 over n+p rows.
     */
    Eigen::MatrixXd X_aug_en_;

    /** @brief Augmented response y_aug_en_ (n + p).
     *  EN_AUG only. Holds [y ; 0_p], zero-centered.
     */
    Eigen::VectorXd y_aug_en_;

    /** @brief Per-cluster IEN bottom-row contribution (M x p): row m has
     *  value (sqrt(lambda2) / sqrt(p_m)) at columns of cluster m, zero elsewhere.
     *  Built once per select() if gvs_type == IEN.
     */
    Eigen::MatrixXd ien_bottom_block_p_;

    // ----- Effective data shapes used by the solver -----

    /** @brief Number of effective rows seen by the solver:
     *  n for EN, n + M for IEN, n + p for EN_AUG.
     *  Set once per select() in onSelectBegin().
     */
    std::size_t n_eff_{0};

    /** @brief User-requested memory-mapping flag, captured before passing
     *  the base control parameter to the parent constructor. The base
     *  copy is forced to `false` so that the inherited `memmap_mgr_` is
     *  not auto-allocated with the wrong row count; GVS re-allocates it
     *  in `onSelectBegin()` with `n_eff_` once it is known.
     */
    bool gvs_use_mmap_{false};

    // ----- Solver-side buffers + Maps (must outlive any cached solver) -----

    /** @brief Per-experiment dummy block D ((n_eff x num_dummies) per k).
     *  In-memory mode: stable across T-iterations within one L-iter,
     *  rebuilt per L-iter. Memory-mapped mode: empty (storage lives in
     *  the inherited `memmap_mgr_`). */
    std::vector<Eigen::MatrixXd> D_solver_bufs_;

    /** @brief Stable Map over X (EN: *X_, IEN: X_aug_) shared across all k. */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_solver_map_;

    /** @brief Stable Map over y_ (EN) or y_aug_ (IEN). */
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_solver_map_;

    /** @brief Stable Maps over D_solver_bufs_, one per k. */
    std::vector<std::unique_ptr<Eigen::Map<Eigen::MatrixXd>>> D_solver_maps_;

    // ----- In-memory warm-start cache for the T-loop -----

    /** @brief Owned solver instances, one per experiment k.
     *  Cleared at the end of every L-iteration.
     *  Within one L-iteration the same instance is reused across T-iterations.
     */
    std::vector<std::unique_ptr<tsolvers::TSolver_Base>> solvers_cache_;

    // ============================================================
    // Setup helpers (called from select())
    // ============================================================

    /** @brief Validate GVS-specific parameters; throws std::invalid_argument. */
    void validateGVSParameters() const;

    /** @brief Dispatch to setupGVS_PriorGroups() or setupGVS_Cluster(). */
    void setupGVS();

    /** @brief Route 1: build clusters from user-supplied prior_groups. */
    void setupGVS_PriorGroups();

    /** @brief Route 2: hierarchical clustering on the |cor(X)| distance. */
    void setupGVS_Cluster();

    /** @brief Run hierarchical clustering with the configured linkage. */
    std::vector<hac::MergeStep> runClustering() const;

    /** @brief Common post-clustering finalization: per-cluster Sigma_m,
     *  Cholesky factors, IEN_cl_id_vectors, groups_vec.
     */
    void finalizeSetup();

    /** @brief Resolve lambda_2 -- user-supplied or GCV-optimal.
     *  Returns the positive scalar in LARS units.
     */
    double computeLambda2() const;

    /** @brief Build the IEN-augmented X_aug_, y_aug_, and bottom block.
     *  Called once per select() when gvs_type == IEN.
     */
    void buildIENAugmentation();

    /** @brief Build the EN_AUG-augmented X_aug_en_ and y_aug_en_.
     *  Called once per select() when gvs_type == EN_AUG.
     *  Implements the Zou & Hastie (2005) LARS-EN data augmentation:
     *  X_aug = [X ; sqrt(lambda2)*I_p] (column-normalized), y_aug = [y ; 0_p].
     *  n_eff is fixed at n + p.
     */
    void buildENAugmentation();

    /**
     * @brief Center each column to mean 0 and rescale to unit L2-norm,
     *        in place.
     *
     * @details
     *  Templated on `Derived` so it accepts both owning `Eigen::MatrixXd`
     *  buffers and `Eigen::Map<Eigen::MatrixXd>` views over
     *  memory-mapped storage. Operates over all rows of `M` (i.e., the
     *  full effective row count, including IEN bottom-block rows).
     *
     *  Applied externally to TLASSO / TENET (both solvers receive
     *  `normalize=false, intercept=false`) so that the augmented system
     *  handed to LARS conforms to the project-wide unit-L2 + zero-mean
     *  convention.
     *
     * @tparam Derived Any Eigen dense matrix expression.
     *
     * @param[in,out] M    Matrix to normalize in place.
     * @param eps          Numerical zero. A column whose L2 norm after
     *                     centering falls below `eps` triggers a
     *                     `std::runtime_error` (degenerate input).
     */
    template <typename Derived>
    static void normalizeAugmentedColumns(Eigen::MatrixBase<Derived>& M, double eps);

    /** @brief Write the assembled (n_eff x num_dummies) D-matrix for
     *  experiment `k` directly into `dst` (typically an Eigen::Map view
     *  over a memory-mapped buffer obtained from
     *  `memmap_mgr_->getDMap`). Layout matches `assembleD()` exactly
     *  (top n rows = horzcat of `dummy_layers_[k]`, bottom M rows for
     *  IEN = repeated `ien_bottom_block_p_`). No allocation.
     */
    void writeAssembledDIntoMap(std::size_t k,
                                std::size_t num_dummies,
                                Eigen::Map<Eigen::MatrixXd>& dst) const;


    // ============================================================
    // Per-experiment dummy machinery
    // ============================================================

    /**
     * @brief Draw one cluster-aware dummy layer (n x p).
     *
     * @param rng Mersenne-Twister RNG, advanced in-place.
     *
     * @return One layer of (n x p) MVN draws, columns aligned with the
     *         clusters in gvs_setup_.clusters_list.
     */
    Eigen::MatrixXd drawClusterDummyLayer(std::mt19937& rng) const;

    /**
     * @brief Assemble the (n_eff x num_dummies) dummy matrix for experiment k.
     *
     * @details
     *   - EN : horzcat of dummy_layers_[k] (n x num_dummies).
     *   - IEN: horzcat of dummy_layers_[k] vertically stacked on top of
     *          (M x num_dummies) bottom block whose [m, layer*p + j] entry
     *          equals ien_bottom_block_p_(m, j).
     */
    Eigen::MatrixXd assembleD(std::size_t k, std::size_t num_dummies) const;


    // ============================================================
    // K-experiment loop
    // ============================================================

    /**
     * @brief Result aggregated over K experiments at one (T_stop, num_dummies).
     */
    struct ExpAgg {
        /** @brief (p x T_stop), already divided by K. */
        Eigen::MatrixXd phi_T_mat;

        /** @brief (p x 1) = phi_T_mat.col(T_stop - 1). */
        Eigen::VectorXd Phi;
    };

    /**
     * @brief Run K random experiments; warm-start across T-iterations only.
     *
     * @param T_stop          Stopping threshold for this step.
     * @param num_dummies     Total number of dummy columns.
     * @param use_warm_start  If true and solvers_cache_ is populated, reuse
     *                        existing solver instances; else build fresh.
     */
    ExpAgg runKExperiments(std::size_t T_stop,
                           std::size_t num_dummies,
                           bool        use_warm_start);

    /**
     * @brief Extract phi_T contribution for one experiment from a beta path.
     *
     * @param beta_path    ((p + L) x n_steps) full LARS path.
     * @param p            Number of original features.
     * @param num_dummies  L = number of dummy columns.
     * @param T_stop       Stopping threshold.
     * @param eps          Numerical zero.
     * @param verbose      If true, emit a warning when fewer than t dummies
     *                     are ever active for some t in [1, T_stop].
     *
     * @return (p x T_stop) binary contribution matrix (0 or 1, NOT divided by K).
     */
    static Eigen::MatrixXd extractPhiContribFromPath(
        const Eigen::MatrixXd& beta_path,
        std::size_t            p,
        std::size_t            num_dummies,
        std::size_t            T_stop,
        double                 eps,
        bool                   verbose);


    // ============================================================
    // Variant hooks (override base TRexSelector hooks)
    // ============================================================

    /**
     * @brief Lifecycle hook: cluster discovery, lambda_2 resolution,
     *        IEN-augmentation construction, and solver-side Map setup.
     *
     * @details Called once at the start of the inherited `select()`,
     *  immediately after the voting-grid setup. Replaces the
     *  setup phase of the previous full `select()` override.
     */
    void onSelectBegin() override;

    /**
     * @brief Per-L-iteration dummy preparation for GVS.
     *
     * @details
     *  Honors `LLoopStrategy::STANDARD`, `HCONCAT`, `SKIPL`, and `DIRECT`.
     *  STANDARD / SKIPL / DIRECT redraw all `LL` cluster-MVN dummy layers
     *  from scratch per L-iteration; HCONCAT appends one fresh layer per
     *  L-iteration. SKIPL is invoked exactly once per `select()` with
     *  `LL = max_dummy_multiplier`. PERMUTATION and PERMUTATION_DIRECT are
     *  rejected in the constructor; reaching them here triggers a defensive
     *  `std::logic_error`. Also (re)builds `D_solver_bufs_` for the K
     *  experiments and clears `solvers_cache_` so the next `evaluateStep()`
     *  call starts with fresh solvers.
     */
    void prepareDummiesForLStep(LStepContext& ctx) override;

    /**
     * @brief Override the inherited `evaluateStep` so that the inherited
     *        L-loop and T-loop drivers can call into GVS machinery.
     *
     * @details
     *  `strategy`, `seed_factor`, and `existing_on_disk` are ignored: GVS
     *  uses its own per-experiment dummy cache populated by
     *  `prepareDummiesForLStep()`.
     */
    StepView evaluateStep(std::size_t                num_dummies,
                          std::size_t                T_stop,
                          bool                       use_warm_start,
                          tc::er::ExperimentStrategy strategy,
                          std::size_t                seed_factor,
                          std::size_t                existing_on_disk) override;

    /**
     * @brief Result widening + GVS-specific cleanup.
     *
     * @details
     *  Populates `gvs_result_` with both the base fields (copied from
     *  `base_result`) and the GVS-specific diagnostics, then releases the
     *  per-experiment buffers and solver cache. Returns a base-sliced
     *  `SelectionResult` so that the inherited `select()` can continue with
     *  warm-start cleanup and X denormalization.
     */
    SelectionResult finalizeSelectionResult(
        SelectionResult&& base_result) override;

// -----------------------------------------------------------------------
}; /* End of class TRexGVSSelector */

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_gvs */
// ===================================================================================

#endif /* TREX_SELECTOR_METHODS_TREX_GVS_HPP */
