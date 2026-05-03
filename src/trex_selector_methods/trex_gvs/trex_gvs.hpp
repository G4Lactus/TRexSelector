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
 *    - GVSType::EN    Ordinary Elastic Net (Zou & Hastie, 2005).
 *                     Cluster-aware MVN dummies are passed to the
 *                     solver. The L2 penalty is handled by the terminating solver
 *                     itself via its `lambda2` hyperparameter.
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
 *                     system (Theorem 1 in Machkour et al.).
 *
 *  Design notes:
 *  ------------
 *    - GVS does NOT use the inherited:
 *      DummyGenerator, ExperimentRunner, or MemmapManager.
 *      It owns a private K-experiment loop, a private per-experiment
 *      dummy-layer cache, and (for IEN) a private warm-start
 *      cache of in-memory T-LASSO solvers.
 *
 *    - The L-loop honors only LLoopStrategy::STANDARD (redraw all layers
 *      per L-iteration) and LLoopStrategy::HCONCAT (append one new layer
 *      per L-iteration).  Other strategies and memory mapping are rejected
 *      in the constructor.
 *
 *    - The lambda_2 ridge parameter is either user-supplied
 *      (TRexGVSControlParameter::lambda2_lars > 0) or computed once via the
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
#include <trex_selector_methods/utils/trex_solver_dispatch.hpp>

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
    IEN
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
     *  different clusters (used only when groups are not provided).
     *  Default: 0.5.
     */
    double corr_max = 0.5;

    /** @brief Linkage method for hierarchical clustering.
     *  Supported: Single (default), Complete, Average, WPGMA.
     *  Ward is rejected because the d = 1 - |cor| metric is non-Euclidean.
     */
    hac::LinkageMethod hc_linkage = hac::LinkageMethod::Single;

    /** @brief User-supplied lambda_2 in LARS units.
     *  0.0 (default) means auto-compute once via GCV-optimal ridge scaled by p / 2.
     */
    double lambda2_lars = 0.0;

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

    /** @brief Group labels carried through from the control parameters (may
     *  be empty).
     */
    std::vector<std::string> group_labels;

    /** @brief Linkage method actually used; "prior" if Route 1. */
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
 *  Inherits from TRexSelector. The select() method is overridden in full --
 *  the base implementation is NOT reused because GVS bypasses the inherited
 *  DummyGenerator/ExperimentRunner/MemmapManager pipeline. The base class
 *  is used only for normalization, voting-grid / FDP utilities, and result
 *  bookkeeping.
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
     * @brief Run T-Rex+GVS selection.
     *
     * @return SelectionResult. Use getGVSResult() for full GVS diagnostics.
     */
    SelectionResult select() override;

    /**
     * @brief Retrieve the full GVS result from the last select() call.
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

    /** @brief Per-cluster IEN bottom-row contribution (M x p): row m has
     *  value (sqrt(lambda2) / sqrt(p_m)) at columns of cluster m, zero elsewhere.
     *  Built once per select() if gvs_type == IEN.
     */
    Eigen::MatrixXd ien_bottom_block_p_;

    // ----- Effective data shapes used by the solver -----

    /** @brief Number of effective rows seen by the solver (n for EN,
     *  n + M for IEN).  Set once per select() inside setupGVS().
     */
    std::size_t n_eff_{0};

    // ----- Solver-side buffers + Maps (must outlive any cached solver) -----

    /** @brief Per-experiment dummy block D ((n_eff x num_dummies) per k).
     *  Stable across T-iterations within one L-iter.  Rebuilt per L-iter. */
    std::vector<Eigen::MatrixXd> D_solver_bufs_;

    /** @brief Stable Map over X (EN: *X_, IEN: X_aug_) shared across all k. */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_solver_map_;

    /** @brief Stable Map over y_ (EN) or y_aug_ (IEN). */
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_solver_map_;

    /** @brief Stable Maps over D_solver_bufs_, one per k. */
    std::vector<std::unique_ptr<Eigen::Map<Eigen::MatrixXd>>> D_solver_maps_;

    // ----- In-memory warm-start cache for the T-loop -----

    /** @brief Owned solver instances, one per experiment k.
     *  Cleared at the end of every L-iteration.  Within one L-iteration the
     *  same instance is reused across T-iterations. */
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
    // Custom L/T loops (replace inherited runLLoop/runTLoop)
    // ============================================================

    /**
     * @brief Override the inherited evaluateStep so that the inherited
     *        runTLoop driver can call into GVS machinery.
     *
     * @details
     *  We do NOT use the inherited runLLoop because it hard-codes the
     *  DummyGenerator pipeline.
     *  Instead we provide a private runGVSLLoop().
     *  However, the inherited runTLoop is generic enough
     *  (uses only evaluateStep, initTAccumulators, growTAccumulators,
     *  appendTRow, trimTAccumulators) that we can reuse it by overriding
     *  evaluateStep().
     *
     *  Strategy and seed_factor arguments are ignored -- GVS uses its own
     *  per-experiment dummy cache.
     */
    StepView evaluateStep(std::size_t                num_dummies,
                          std::size_t                T_stop,
                          bool                       use_warm_start,
                          tc::er::ExperimentStrategy strategy,
                          std::size_t                seed_factor,
                          std::size_t                existing_on_disk) override;

    /**
     * @brief L-loop calibration that honours STANDARD or HCONCAT and uses
     *        cluster-aware MVN dummy generation.
     *
     * @param FDP_hat  Output: FDP estimates after the final L-iteration.
     * @param Phi_out  Output: Phi vector from the final K-experiment pass
     *                 (used to seed the T-loop accumulator's row 0).
     */
    void runGVSLLoop(Eigen::VectorXd& FDP_hat, Eigen::VectorXd& Phi_out);


    // ============================================================
    // Result assembly
    // ============================================================

    /**
     * @brief Populate gvs_result_, run cleanup + denormalization, and
     *        return the base-sliced SelectionResult.
     */
    SelectionResult assembleGVSResult(const Eigen::MatrixXd& final_phi_T_mat,
                                      const Eigen::VectorXd& final_Phi);

// -----------------------------------------------------------------------
}; /* End of class TRexGVSSelector */

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_gvs */
// ===================================================================================

#endif /* TREX_SELECTOR_METHODS_TREX_GVS_HPP */
