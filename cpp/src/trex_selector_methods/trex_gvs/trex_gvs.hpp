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
 *  dummy generation and one of two selection policies. All augmentation is
 *  ENCAPSULATED IN THE SOLVERS: GVS always hands the solver the plain
 *  (n x p) design X_, the plain (n x L) MVN dummy block D_k, and the centered
 *  response y_.
 *
 *    - GVSType::EN    Elastic-Net penalty for grouped variable selection.
 *                     The concrete solver is chosen by ENSolverType:
 *
 *                     - ENSolverType::TENET (default) — Gram-based EN
 *                       (TENET_Solver): a pathwise LARS-EN solver that absorbs
 *                       the ridge penalty via a Cholesky update on
 *                       X_A^T X_A + lambda2*I (Zou & Hastie, 2005).
 *
 *                     - ENSolverType::TENET_AUG — augmented-LASSO formulation
 *                       (TENETAug_Solver; Machkour et al. 2022, 2023). The
 *                       solver internally builds the Zou-Hastie system
 *                         X_aug = d2 * [ X ; d1*I_p ; 0_L ],
 *                         D_aug = d2 * [ D ; 0_p ; d1*I_L ],
 *                         y_aug = [ y ; 0_{p+L} ],
 *                       with d1 = sqrt(lambda2), d2 = 1/sqrt(1+lambda2), and
 *                       runs an inner T-LASSO (or pure T-LARS when
 *                       `tenet_aug_use_lars` is set) on it AS CONSTRUCTED —
 *                       for pre-normalized unit-scale X and D columns the
 *                       augmented columns are already unit-scale by the d1/d2
 *                       algebra, and this exactly reproduces the Gram-based
 *                       TENET path (verified to machine precision; see
 *                       tenet_aug_solver.hpp). GVS therefore passes the SAME
 *                       pre-normalized D_k as for plain TENET.
 *
 *    - GVSType::IEN   Informed Elastic Net (TIENETAug_Solver; Machkour et
 *                     al., CAMSAP 2023). The solver internally builds the
 *                     group-informed Lasso-type system of Theorem 1
 *                         X_aug = [ X ; B ]  (M extra rows; B(m, .) =
 *                                 (sqrt(lambda2)/sqrt(p_m)) on cluster-m cols),
 *                         D_aug = [ D ; B tiled once per dummy layer ],
 *                         y_aug = [ y ; 0_M ],
 *                       applies the R-faithful post-augmentation preprocessing
 *                       (column centering + rescaling, y centering; R
 *                       lm_dummy.R `scale()`), and runs an inner T-LASSO.
 *                       GVS passes the RAW MVN dummy block (the solver
 *                       normalizes the augmented columns itself) plus the
 *                       0-based cluster assignment from the one-time
 *                       clustering of X.
 *
 *    Reassembly rule (ALL variants): the per-experiment cluster-aware MVN
 *    dummy block D_k is (re)assembled every L-iteration because num_dummies
 *    (L) changes; fresh solvers are then constructed at T = 1.
 *
 *    Column scaling: X_ is centered + rescaled by the base class
 *    (ScalingMode::L2 or ZSCORE). The EN-variant dummy blocks are centered +
 *    rescaled with the same convention before being handed to the solver;
 *    the IEN dummy block is passed raw (see above).
 *
 *  Design notes:
 *  ------------
 *    - GVS does NOT use the inherited `DummyGenerator` or `ExperimentRunner`.
 *      It owns a private K-experiment loop, a private per-experiment
 *      dummy-layer cache, and a private warm-start cache of in-memory solvers
 *      (TENET / TENETAug / TIENETAug). The inherited `MemmapManager` IS
 *      reused (re-allocated in `onSelectBegin()` with n rows, and torn down
 *      by the base class at the end of `select()`); see `gvs_use_mmap_`.
 *
 *    - Support L-looop strategies:
 *.     `LLoopStrategy::STANDARD`, `HCONCAT`, `SKIPL`, and `DIRECT`.
 *      `STANDARD` and `DIRECT` are equivalent inside GVS (the overridden
 *      `evaluateStep` consumes `D_solver_bufs_` rather than streaming dummies from disk).
 *      `SKIPL` collapses to a single L-iter with LL = max_dummy_multiplier.
 *      `PERMUTATION` and `PERMUTATION_DIRECT` are rejected because row permutation
 *      would destroy the per-cluster MVN covariance structure that GVS dummies
 *      are designed to mirror.
 *
 *    - Memory mapping is supported uniformly across all four allowed L-loop strategies.
 *      When enabled, K independent (non-shared) D files are allocated up-front
 *      in `onSelectBegin()` with n rows (the solvers own any augmented
 *      buffers internally); each L-loop step fills the per-K columns in place
 *      via `assembleDummyBlock`.
 *
 *    - The lambda_2 ridge parameter (LARS units) is resolved as follows:
 *        * lambda_2 <  0 (default -1): "not supplied" sentinel (R's NULL
 *          equivalent) -> auto-computed via k-fold CV on the centered +
 *          z-score/L2-normalised (X, y), scaled by p / 2 to LARS units;
 *        * lambda_2 == 0: degenerate case -> pure T-LASSO (no ridge);
 *        * lambda_2 >  0: fixed ridge penalty.
 *
 *  Indexing convention:
 *    - All cluster IDs and column indices are 0-based throughout the C++ API.
 *      Users coming from R must pass `groups - 1L`.
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
    /** @brief Ordinary Elastic Net; solver variant controlled by ENSolverType. */
    EN,
    /** @brief Informed Elastic Net (TIENETAug_Solver: group-informed
     *  row-augmented system built inside the solver). */
    IEN
};


/**
 * @brief EN solver variant for GVSType::EN.
 */
enum class ENSolverType {
    /** @brief Gram-based EN (TENET_Solver): X, D, y are passed unaugmented;
     *  the ridge penalty is absorbed via a Cholesky update on X_A^T X_A + lambda2*I.
     */
    TENET,

    /** @brief Augmented-LASSO EN (TENETAug_Solver): builds augmented matrices internally.
     *  d1 = sqrt(lambda2),  d2 = 1/sqrt(1+lambda2)
     *  X_aug = d2*[X; d1*I_p; 0_L]    (n+p+L rows)
     *  D_aug = d2*[D; 0_p;   d1*I_L]  (n+p+L rows)
     *  y_aug = [y; 0_{p+L}]
     *  Proved equivalent to TENET for lambda2 >= 0.
     */
    TENET_AUG
};


/**
 * @brief Selection rule for the ridge regularisation parameter `lambda_2`
 *        when no user-supplied value is given.
 *
 * @details
 *   Active methods:
 *   - `CV_1SE_SVD` / `CV_MIN_SVD` : same semantics but via `ridge_cv_svd`
 *                 (JacobiSVD); for numerical validation.
 *   - `CV_1SE_CCD` / `CV_MIN_CCD` : glmnet-faithful coordinate-descent ridge CV
 *                 (`enet_cv_ccd`, alpha=0) with glmnet's fdev/devmax path
 *                 early-termination. Reproduces `cv.glmnet(alpha=0)` lambda.min exactly
 *                 and lambda.1se within CV fold noise; returns lambda on glmnet's
 *                 reported scale (like the GLMNET variants), so the p/2 factor converts
 *                 to LARS units as in R's `lm_dummy`.
 */
enum class LambdaSelectionMethod {
    /** @brief JacobiSVD direct ridge CV, 1SE. */
    CV_1SE_SVD,
    /** @brief JacobiSVD direct ridge CV, min. */
    CV_MIN_SVD,
    /** @brief Coordinate-descent ridge CV, glmnet-faithful 1SE. */
    CV_1SE_CCD,
    /** @brief Coordinate-descent ridge CV, glmnet-faithful min. */
    CV_MIN_CCD
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

    /** @brief EN solver variant when gvs_type == GVSType::EN.
     *  Only TENET and TENET_AUG are supported. Default: TENET.
     */
    ENSolverType en_solver = ENSolverType::TENET;

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
     *  - `< 0` (default `-1.0`): sentinel meaning "not supplied" (the C++
     *    equivalent of R's `NULL`); triggers auto-computation via
     *    `lambda2_method`. The auto-computed lambda is scaled by p / 2 to
     *    LARS units.
     *  - `== 0`: fixed degenerate case -> pure TLASSO (no ridge penalty).
     *  - `> 0` : fixed ridge penalty in LARS units.
     *
     *  A negative sentinel (rather than NaN) is used so the value stays
     *  well-defined under aggressive floating-point optimisation.
     */
    double lambda_2 = -1.0;

    /** @brief Selection rule for auto-computing `lambda_2` when `lambda_2 < 0`.
     *         Options are: CV_1SE_SVD, CV_MIN_SVD, CV_1SE_CCD, CV_MIN_CCD.
     *          - CV_1SE_SVD / CV_MIN_SVD : JacobiSVD ridge CV (ridge_cv_svd);
     *            for numerical validation.
     *          - CV_1SE_CCD / CV_MIN_CCD : glmnet-faithful coordinate-descent ridge CV
     *            (enet_cv_ccd, alpha=0) with glmnet's fdev/devmax path early
     *            stopping. Default: CV_1SE_CCD.
     */
    LambdaSelectionMethod lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;

    /** @brief Number of folds for k-fold cross-validation used by the
     *  `lambda2_method` CV paths. Default: 10 (matches R's `cv.glmnet`).
     */
    int cv_n_folds = 10;

    /** @brief Number of points on the geometric lambda grid searched by
     *  the auto-computation. Default: 1000. Applies to CV paths.
     */
    Eigen::Index cv_n_lambda = 1000;

    /** @brief Seed for the deterministic fold-permutation RNG used by the
     *  `lambda2_method` CV paths.
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

    /** @brief Diagnostic / R-parity option for the augmented solvers
     *  (`en_solver == TENET_AUG` and `gvs_type == IEN`). When true, the
     *  augmented system is solved with a pure-LARS inner solver (never drops
     *  variables, matching R's `trex(type="lar")`) instead of the default
     *  LARS-LASSO inner solver. Ignored for plain TENET. Default: false.
     */
    bool tenet_aug_use_lars = false;

    /** @brief Base T-Rex algorithmic control parameters (nested).
     *  GVS restricts `solver_type` to the solvers it actually drives (see
     *  `TRexGVSSelector`'s validation): TENET/TENET_AUG for `GVSType::EN`,
     *  TIENET_AUG for `GVSType::IEN`. The default here overrides the base
     *  class's own default (TLARS) with TENET, matching the default
     *  `gvs_type = EN` / `en_solver = TENET`, so a fully-defaulted
     *  `TRexGVSControlParameter` constructs without throwing.
     */
    tc::TRexControlParameter trex_ctrl = [] {
        tc::TRexControlParameter t;
        t.solver_type = sd::SolverTypeForTRex::TENET;
        return t;
    }();
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
 *  `prepareDummiesForLStep()`,
 *  `evaluateStep()`, and
 *  `finalizeSelectionResult()`.
 *  The inherited `DummyGenerator` and `ExperimentRunner` are bypassed
 *  (GVS uses cluster-aware MVN dummy generation, an in-memory K-experiment
 *  loop, and an in-memory solver warm-start cache).
 *  The inherited `MemmapManager` IS reused when the user requests memory mapping;
 *  see `gvs_use_mmap_`.
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
        GVSType gvs_type = GVSType::EN;

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
     * @param trex_gvs_ctrl  GVS-specific control parameters, nesting the base
     *                       T-Rex algorithmic control parameters as
     *                       `trex_gvs_ctrl.trex_ctrl`.
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
        TRexGVSControlParameter      trex_gvs_ctrl = TRexGVSControlParameter(),
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

    /** @brief GVS-specific control parameters (nests the base trex_ctrl). */
    TRexGVSControlParameter trex_gvs_ctrl_;

    /** @brief Pre-computed group structure. */
    GVSSetupResult gvs_setup_;

    /** @brief Resolved lambda_2 in LARS units (> 0 after computeLambda2()). */
    double lambda2_{0.0};

    /** @brief Resolved CV fold-permutation seed.
     *
     *  Computed once in the constructor from `trex_gvs_ctrl_.cv_seed` and `seed_`:
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

    /** @brief User-requested memory-mapping flag, captured before passing
     *  the base control parameter to the parent constructor. The base
     *  copy is forced to `false` so that the inherited `memmap_mgr_` is
     *  not auto-allocated with the wrong row count; GVS re-allocates it
     *  in `onSelectBegin()` with `n_eff_` once it is known.
     */
    bool gvs_use_mmap_{false};

    // ----- Solver-side buffers + Maps (must outlive any cached solver) -----

    /** @brief Per-experiment dummy block D ((n x num_dummies) per k).
     *  In-memory mode: stable across T-iterations within one L-iter,
     *  rebuilt per L-iter. Memory-mapped mode: empty (storage lives in
     *  the inherited `memmap_mgr_`). EN variants: centered + rescaled;
     *  IEN: raw MVN draws (TIENETAug normalizes the augmented columns). */
    std::vector<Eigen::MatrixXd> D_solver_bufs_;

    /** @brief Stable Map over X_ seen by the solver (all variants; the
     *  augmented solvers own their augmented copies internally). */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_solver_map_;

    /** @brief Stable Map over y_ seen by the solver (all variants). */
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

    /** @brief Resolve lambda_2 -- user-supplied or auto-computed via CV.
     *  Returns a non-negative scalar in LARS units (0 => degenerate TLASSO).
     */
    double computeLambda2() const;

    /** @brief Assemble experiment `k`'s plain (n x L) dummy block into `dst`
     *  (an owning `Eigen::MatrixXd` or an `Eigen::Map` over mmap storage):
     *  the horizontal concatenation of `dummy_layers_[k]`. All augmentation
     *  (ridge rows, group blocks) lives inside the solvers.
     *
     *  Column scaling is the CALLER's choice: the EN variants center +
     *  rescale afterwards (TENET consumes pre-normalized columns; TENETAug's
     *  d1/d2 algebra preserves unit scale exactly); IEN passes the block raw
     *  (TIENETAug normalizes the augmented columns internally, mirroring R's
     *  post-augmentation `scale()`).
     */
    template <typename Derived>
    void assembleDummyBlock(std::size_t k,
                            Eigen::MatrixBase<Derived>& dst) const;


    // ============================================================
    // Per-experiment dummy machinery
    // ============================================================

    /**
     * @brief Draw one cluster-aware dummy layer (n_ rows).
     *
     * @param rng  Mersenne-Twister RNG, advanced in-place.
     *
     * @return One layer of (n_ x p) MVN draws, columns aligned with the
     *         clusters in gvs_setup_.clusters_list.
     */
    Eigen::MatrixXd drawClusterDummyLayer(std::mt19937& rng) const;


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
     * @brief Lifecycle hook: cluster discovery, lambda_2 resolution, and
     *        solver-side Map setup.
     *
     * @details Called once at the start of the inherited `select()`,
     *  immediately after the voting-grid setup.
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

#endif /* End of TREX_SELECTOR_METHODS_TREX_GVS_HPP */
