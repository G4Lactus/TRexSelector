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
 *  dummy generation and one of two selection policies:
 *
 *    - GVSType::EN    Elastic-Net penalty for grouped variable selection.
 *                     The concrete inner solver is chosen by ENSolverType:
 *
 *                     - ENSolverType::TENET (default) — Gram-based EN
 *                       (TENET_Solver): a pathwise LARS-EN solver that absorbs
 *                       the ridge penalty via a Cholesky update on
 *                       X_A^T X_A + lambda2*I (Zou & Hastie, 2005). X, D, y are
 *                       passed UNAUGMENTED (n rows); n_eff = n.
 *
 *                     - ENSolverType::TENET_AUG — augmented-LASSO formulation
 *                       (Machkour et al. 2022, 2023). The ridge penalty is
 *                       lifted OUT of the solver into an explicitly augmented
 *                       system that is fed to a plain T-LASSO solver. Because
 *                       the elastic net penalises every coefficient, the ridge
 *                       blocks are identity matrices I_p (real variables) and
 *                       I_L (dummies), with d1 = sqrt(lambda2) and the global
 *                       Zou-Hastie factor d2 = 1/sqrt(1+lambda2):
 *                         X_aug = d2 * [ X       ]  (n rows)
 *                                      [ d1*I_p  ]  (p rows)
 *                                      [ 0       ]  (L rows)
 *                         D_aug = d2 * [ D       ]  (n rows)
 *                                      [ 0       ]  (p rows)
 *                                      [ d1*I_L  ]  (L rows)
 *                         y_aug = [ y ; 0_{p+L} ]   (NO d2, NO scaling on y)
 *                       => n_eff = n + p + L. Since L (= num_dummies) CHANGES
 *                       across the L-loop, the I_L block grows, so X_aug, D_aug
 *                       and y_aug are REBUILT every L-iteration (see
 *                       prepareDummiesForLStep / buildENXAugmentation /
 *                       buildENyAugmentation / buildENDAugmentation).
 *                       The global d2 = 1/sqrt(1+lambda2) factor is applied
 *                       EXPLICITLY. Although the post-augmentation
 *                       per-column scaling (always center, then divide by
 *                       L2-norm or sample SD) divides every column
 *                       by its own norm and would cancel any global scalar, d2
 *                       is kept for a 1:1 mirror of the reference. Equivalent
 *                       to TENET for lambda2 > 0.
 *
 *    - GVSType::IEN   Informed Elastic Net (Machkour et al., CAMSAP 2023).
 *                     No dedicated IEN solver exists yet, so the L2 penalty is
 *                     absorbed into an explicit augmented Lasso-type system fed
 *                     to a plain T-LASSO solver. Unlike EN (per-coefficient
 *                     identity ridge), IEN penalises GROUP structure, so the
 *                     ridge block is the cluster-membership matrix B with
 *                     M = max_clusters rows. M is fixed by the ONE-TIME
 *                     clustering of X and is independent of L:
 *                         X_aug = [ X ]  (n rows)
 *                                 [ B ]  (M rows), row m =
 *                                        (sqrt(lambda2)/sqrt(p_m)) on cluster-m cols
 *                         D_aug = [ D          ]  (n rows)
 *                                 [ B repeated ]  (M rows, one B segment per
 *                                                  dummy layer)
 *                         y_aug = [ y ; 0_M ]
 *                       => n_eff = n + M (FIXED). X_aug is built ONCE per
 *                       select(); but because L changes across the L-loop, the
 *                       DUMMY matrix D_aug (whose repeated-B bottom block widens
 *                       with the number of dummy layers) is still REASSEMBLED
 *                       every L-iteration.
 *
 *    Reassembly rule (ALL variants): the per-experiment cluster-aware MVN dummy
 *    matrix is (re)assembled every L-iteration because num_dummies (L) changes;
 *    only the fixed real-variable augmentation (IEN's X_aug) is built once.
 *
 *    Column scaling (ALL variants): every column is ALWAYS centered to mean 0
 *    first, then rescaled by either its L2-norm (ScalingMode::L2) or its sample
 *    SD (ScalingMode::ZSCORE). Centering is applied in BOTH cases. The augmented
 *    real, dummy, and response columns share the exact convention applied to X_
 *    by the base class, mirroring R `scale()` in lm_dummy.R / add_dummies_GVS.R.
 *
 *  Design notes:
 *  ------------
 *    - GVS does NOT use the inherited `DummyGenerator` or `ExperimentRunner`.
 *      It owns a private K-experiment loop, a private per-experiment
 *      dummy-layer cache, and a private warm-start cache of in-memory solvers
 *      (T-LASSO for IEN / EN-aug, TENET for plain EN) shared by ALL variants.
 *      The inherited `MemmapManager` IS reused (re-allocated in
 *      `onSelectBegin()` at the maximum augmented row count, and torn down by
 *      the base class at the end of `select()`); see `gvs_use_mmap_`.
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
 *      in `onSelectBegin()`. Their row count is the MAXIMUM augmented height
 *      (n for plain EN, n+M for IEN, n+p+max_dummies for EN-aug); each L-loop
 *      step fills the per-K columns in place via `buildENDAugmentation` /
 *      `buildIENDAugmentation`. For
 *      EN-aug the flat buffer is reinterpreted per step as a contiguous
 *      (n+p+L) x L view, since the augmented height varies with L.
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
    /** @brief Informed Elastic Net (row-augmented system + TLASSO solver). */
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

    /** @brief Diagnostic / R-parity option for `en_solver == TENET_AUG`.
     *  When true, the augmented EN system is solved with a pure-LARS inner
     *  solver (never drops variables, matching R's `trex(type="lar")`) instead
     *  of the default LARS-LASSO inner solver. Ignored for other solvers.
     *  Default: false.
     */
    bool tenet_aug_use_lars = false;
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

    // ----- Augmented system (row-augmented; IEN and EN-aug) -----

    /** @brief Augmented predictor matrix X_aug, column-normalized.
     *  - IEN     : [X ; bottom_block], fixed ((n + M) x p), built once.
     *  - EN-aug  : [X ; sqrt(lambda2)*I_p ; 0_{L x p}], ((n + p + L) x p),
     *              rebuilt every L-step (L = num_dummies varies).
     *  - plain EN: unused (solver sees *X_ directly).
     */
    Eigen::MatrixXd X_aug_;

    /** @brief Augmented response y_aug.
     *  - IEN     : [y ; 0_M]   (n + M).
     *  - EN-aug  : [y ; 0_{p+L}] (n + p + L), rebuilt every L-step.
     *  - plain EN: unused.
     */
    Eigen::VectorXd y_aug_;

    /** @brief Per-cluster IEN bottom-row contribution (M x p): row m has
     *  value (sqrt(lambda2) / sqrt(p_m)) at columns of cluster m, zero elsewhere.
     *  Built once per select() if gvs_type == IEN.
     */
    Eigen::MatrixXd ien_bottom_block_p_;

    // ----- Effective data shapes used by the solver -----

    /** @brief Number of effective rows seen by the solver:
     *  n for plain EN, n + M for IEN, n + p + L for EN-aug (L = num_dummies).
     *  For IEN / plain EN set once in onSelectBegin(); for EN-aug updated each
     *  L-step in prepareDummiesForLStep().
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

    /** @brief Stable Map over X seen by the solver:
     *  plain EN -> *X_; IEN / EN-aug -> X_aug_. Shared across all k.
     *  Rebound per L-step for EN-aug (X_aug_ is rebuilt each L-step). */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_solver_map_;

    /** @brief Stable Map over y seen by the solver:
     *  plain EN -> y_; IEN / EN-aug -> y_aug_. */
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

    /** @brief Build the FIXED IEN real-variable design `X_aug_` and the
     *  reusable cluster-membership bottom block `ien_bottom_block_p_`.
     *  Called ONCE per select() when gvs_type == IEN, because these depend only
     *  on the one-time clustering of X (rows = n + M, M = max_clusters) and are
     *  independent of the dummy count L. No global d2 factor for IEN (the
     *  sqrt(l2)*(1/sqrt(l2)) on the data block cancels in R; only the bottom
     *  group-indicator block carries sqrt(lambda2)/sqrt(p_m)). The matrix is
     *  column-normalized in place (R's `scale()`).
     */
    void buildIENXAugmentation();

    /** @brief Build the FIXED IEN augmented response `y_aug_ = [ y ; 0_M ]`,
     *  then center it (R: `y <- y - mean(y)`; no d2, no SD scaling on y).
     *  Called ONCE per select() when gvs_type == IEN.
     */
    void buildIENyAugmentation();

    /** @brief Build the EN-augmented (Zou-Hastie) design `X_aug_` for the
     *  TENET_AUG solver variant. Unlike IEN (fixed `n + M` rows), the EN
     *  augmentation has `n + p + L` rows where `L = num_dummies` varies per
     *  L-iteration, so this is (re)built every L-step from
     *  `prepareDummiesForLStep`.
     *
     *  X_aug_ = d2 * [ X ; d1*I_p ; 0_{L x p} ]   ((n+p+L) x p)
     *  with d1 = sqrt(lambda2), d2 = 1/sqrt(1+lambda2). The global d2 factor is
     *  applied EXPLICITLY (faithful to R `lm_dummy.R`), then the matrix is
     *  column-normalized by `normalizeAugmentedColumns` (R's `scale()`).
     *
     *  @param num_dummies Current dummy count L (sets the trailing zero block).
     */
    void buildENXAugmentation(std::size_t num_dummies);

    /** @brief Build the EN-augmented response `y_aug_ = [ y ; 0_{p+L} ]` for
     *  the TENET_AUG variant, then center it (R: `y <- y - mean(y)`; no d2, no
     *  SD scaling on y). (Re)built every L-step from `prepareDummiesForLStep`.
     *
     *  @param num_dummies Current dummy count L (sets the trailing zero block).
     */
    void buildENyAugmentation(std::size_t num_dummies);

    /**
     * @brief Center each column to mean 0 and rescale it, in place, using
     *        the same column-scaling convention as the externally applied
     *        `centerAndL2NormalizeX` (L2-norm or z-score / sample SD).
     *
     * @details
     *  Templated on `Derived` so it accepts both owning `Eigen::MatrixXd`
     *  buffers and `Eigen::Map<Eigen::MatrixXd>` views over
     *  memory-mapped storage. Operates over all rows of `M` (i.e., the
     *  full effective row count, including IEN bottom-block rows).
     *
     *  Applied externally to TLASSO / TENET (both solvers receive
     *  `normalize=false, intercept=false`) so that the augmented system
     *  handed to LARS conforms to the chosen column-scaling + zero-mean
     *  convention.
     *
     *  The scale divisor of centered column `x_c` matches
     *  `centerAndL2NormalizeX` exactly:
     *    - L2     (`zscore == false`): `||x_c||`        -> unit-L2 columns.
     *    - ZSCORE (`zscore == true`) : `||x_c|| / sqrt(rows - 1)`
     *                                  (sample SD) -> unit-SD columns,
     *                                  i.e. column L2-norm `sqrt(rows - 1)`.
     *  This MUST track the scaling applied to `X_`: under z-score, the real
     *  variables carry column mass `sqrt(n - 1)`, so the dummies in `M` must
     *  be z-scaled too, otherwise the dummy-based FDR calibration compares
     *  variables and dummies on mismatched scales.
     *
     * @tparam Derived Any Eigen dense matrix expression.
     *
     * @param[in,out] M    Matrix to normalize in place.
     * @param eps          Numerical zero. A column whose scale after
     *                     centering falls below `eps` triggers a
     *                     `std::runtime_error` (degenerate input).
     * @param zscore       If true, divide by sample SD (z-score) instead of
     *                     the plain L2-norm. Default false (unit-L2).
     */
    template <typename Derived>
    static void normalizeAugmentedColumns(Eigen::MatrixBase<Derived>& M, double eps,
                                          bool zscore = false);

    /** @brief Assemble experiment `k`'s EN dummy matrix into `dst`
     *  (an owning `Eigen::MatrixXd` or an `Eigen::Map` over mmap storage).
     *  Handles BOTH EN sub-variants, sized by the caller:
     *    - plain EN (TENET)     : dst is (n x L), D = horzcat of dummy layers.
     *    - EN-aug   (TENET_AUG) : dst is (n+p+L x L),
     *        D_aug = d2 * [ D ; 0_{p x L} ; d1*I_L ], d2=1/sqrt(1+l2), d1=sqrt(l2).
     *  The d2 factor (EN-aug only) is applied EXPLICITLY for fidelity to R.
     *  No column scaling here -- the caller applies `normalizeAugmentedColumns`.
     */
    template <typename Derived>
    void buildENDAugmentation(std::size_t k, std::size_t num_dummies,
                              Eigen::MatrixBase<Derived>& dst) const;

    /** @brief Assemble experiment `k`'s IEN dummy matrix into `dst`
     *  (an owning `Eigen::MatrixXd` or an `Eigen::Map` over mmap storage).
     *  dst is (n + M x L): top n rows = horzcat of `dummy_layers_[k]`; bottom
     *  M rows = `ien_bottom_block_p_` repeated once per dummy layer. No global
     *  d2 factor (see buildIENXAugmentation). No column scaling here -- the
     *  caller applies `normalizeAugmentedColumns`.
     */
    template <typename Derived>
    void buildIENDAugmentation(std::size_t k, std::size_t num_dummies,
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

#endif /* End of TREX_SELECTOR_METHODS_TREX_GVS_HPP */
