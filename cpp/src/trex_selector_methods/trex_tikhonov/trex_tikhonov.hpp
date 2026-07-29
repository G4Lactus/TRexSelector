// ===================================================================================
// trex_tikhonov.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_TIKHONOV_HPP
#define TREX_SELECTOR_METHODS_TREX_TIKHONOV_HPP
// ===================================================================================
/**
 * @file trex_tikhonov.hpp
 *
 * @brief T-Rex selector with a general Tikhonov (anisotropic ridge) penalty.
 *
 * @details
 *
 *  Method:
 *  -------
 *  TRexTikhonovSelector runs the standard T-Rex calibration with the CCD
 *  Informed-Elastic-Net solver (TCIENET_Solver) in its general sparse-K
 *  mode: the quadratic penalty is (lambda_2 / 2) * beta^T K beta for a
 *  user-supplied PSD matrix K = Gamma^T Gamma (p x p). This generalizes:
 *    - K = I                     -> plain elastic net (TCENET),
 *    - K = sum_m 1_m 1_m^T / p_m -> the group-informed IEN penalty that
 *      TRexGVSSelector builds from a clustering (kept separate on purpose:
 *      GVS owns the cluster-aware K-experiment loop; this class runs the
 *      shared base runner),
 *    - banded / Laplacian K      -> smoothness-informed selection.
 *
 *  Dummy coupling (default FOLDED, `fold_dummy_coupling`): dummy copies
 *  join their originating variable's K-coupling — the exact generalization
 *  of GVS's layered group convention — so dummy/null exchangeability holds
 *  for arbitrary K. The INDEPENDENT_RIDGE escape (decoupled kappa_dummy = 1
 *  dummy block) preserves the K = I ≡ TCENET collapse for validation runs;
 *  under it, prefer diag(K) = 1 (see tcienet_solver.hpp).
 *
 *  Dummy generation:
 *  -----------------
 *  Cluster-aware by default (`cluster_dummies = true`), on GVS parity:
 *  variables are partitioned into clusters (user-supplied `prior_groups`,
 *  else correlation-HAC with `corr_max` / `hc_linkage`), and dummies are
 *  drawn as per-cluster colored MVN layers via the shared machinery in
 *  trex_cluster_dummies.hpp — same seed_seq scheme, same center-only /
 *  realized-norms convention as TRexGVSSelector. An informed K encodes
 *  structure among correlated variables; i.i.d. dummies would be
 *  systematically less competitive than correlated null reals there,
 *  deforming the dummy/null exchangeability the FDR calibration relies on.
 *  `cluster_dummies = false` restores the canonical i.i.d. dummy path
 *  (validation / equivalence runs only). The model is installed into the
 *  base DummyGenerator (`setClusterModel`), whose cluster mode serves
 *  EVERY strategy natively: STANDARD / HCONCAT / SKIPL (stored matrices),
 *  SEEDED (one D_k at a time, serial — content bit-identical to
 *  STANDARD at the same L), and the memory-mapped writers (the mmap region
 *  is pre-allocated at maximum dimensions and layers are regenerated into
 *  it deterministically). Only the PERMUTATION variants are rejected: K
 *  row-permuted copies of one base cannot carry the per-cluster MVN
 *  convention.
 *
 *  Architecture:
 *  -------------
 *  Unlike GVS (private K-experiment loop), this class reuses the ENTIRE base
 *  machinery — L/T loops, warm starts, the shared ExperimentRunner —
 *  by injecting the penalty matrix into the shared solver dispatch:
 *  `buildRunnerConfig()` is overridden to set
 *  `ExperimentRunnerConfig::tikhonov_K` (forwarded to
 *  `SolverConfig::tikhonov_K`, where dispatchByType() routes TCIENET through
 *  its sparse-K constructor) and to overwrite `solver_params.lambda2` with
 *  the resolved penalty weight. Cluster-aware dummies enter through the
 *  DummyGenerator cluster mode (`setClusterModel()` in `onSelectBegin()`),
 *  so the base dummy hook, the ExperimentRunner, and the memory-mapped
 *  writers all run unchanged.
 *
 *  lambda_2 selection:
 *  -------------------
 *  Auto-computed in `onSelectBegin()` (X is already centered + scaled, y
 *  centered) via the sparse-K overloads of the IEN-geometry tuners:
 *    - CV_*_IEN_CCD (`ienet_cv_ccd`): lambda_1 path profiled out per
 *      lambda_2 candidate; well-posed in any p/n regime (recommended).
 *    - CV_*_TIK_SVD (`tikhonov_cv_svd`): analytic lambda_1 = 0 backbone;
 *      exact for n > p, refused when the unpenalized null-space block
 *      interpolates the training folds.
 *  The selected lambda_2 is on the unit-L2 SOLVER scale (no p/2
 *  conversion); only the ZSCORE working-scale factor (n - 1) applies.
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Base T-Rex
#include <trex_selector_methods/trex_core/trex.hpp>

// Hierarchical clustering types (LinkageMethod for the dummy clustering)
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::trex_tikhonov
namespace trex::trex_selector_methods::trex_tikhonov {

// Local namespace aliases
namespace tc = trex::trex_selector_methods::trex_core;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;
namespace er = trex::trex_selector_methods::utils::experiment_runner;
namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;


// ===================================================================================
// Enums
// ===================================================================================

/**
 * @brief Selection rule for the Tikhonov penalty weight `lambda_2` when no
 *        user-supplied value is given. All four rules consume the penalty
 *        matrix K and return lambda_2 on the solver scale (no p/2
 *        conversion; see the file header).
 */
enum class TikLambda2Method {
    /** @brief IEN-geometry coordinate-descent CV (lambda1 profiled), 1SE. */
    CV_1SE_IEN_CCD,
    /** @brief IEN-geometry coordinate-descent CV (lambda1 profiled), min. */
    CV_MIN_IEN_CCD,
    /** @brief Generalized-Tikhonov SVD ridge CV (lambda1 = 0 backbone), 1SE. */
    CV_1SE_TIK_SVD,
    /** @brief Generalized-Tikhonov SVD ridge CV (lambda1 = 0 backbone), min. */
    CV_MIN_TIK_SVD
};


// ===================================================================================
// Control Parameters
// ===================================================================================

/**
 * @brief Tikhonov-specific control parameters.
 */
struct TRexTikhonovControlParameter {

    /** @brief General Tikhonov matrix K = Gamma^T Gamma (p x p, sparse,
     *  PSD). REQUIRED: an empty matrix throws at construction. Build from a
     *  regularization operator Gamma (m x p) via
     *  `TRexTikhonovSelector::gammaToK()`. Prefer diag(K) = 1 so the real
     *  columns carry the same ridge as the dummies (kappa_dummy = 1). */
    Eigen::SparseMatrix<double> tikhonov_K;

    /** @brief User-supplied lambda_2 on the solver scale.
     *  - `< 0` (default `-1.0`): "not supplied" sentinel; triggers
     *    auto-computation via `lambda2_method`.
     *  - `== 0`: degenerate case -> pure TCCD lasso path (no penalty).
     *  - `> 0` : fixed penalty weight. */
    double lambda_2 = -1.0;

    /** @brief Selection rule for auto-computing `lambda_2` when
     *  `lambda_2 < 0`. Default: CV_1SE_IEN_CCD (well-posed in any p/n
     *  regime). */
    TikLambda2Method lambda2_method = TikLambda2Method::CV_1SE_IEN_CCD;

    /** @brief Number of folds for the lambda_2 CV. Default: 10. */
    int cv_n_folds = 10;

    /** @brief lambda_2 grid size for the CV_*_TIK_SVD analytic scan.
     *  (CV_*_IEN_CCD scans a 2D surface and uses its class defaults,
     *  25 x 100.) Default: 100. */
    Eigen::Index cv_n_lambda = 100;

    /** @brief Seed for the deterministic CV fold-permutation RNG.
     *  - `-1` (default): derived from the T-Rex `seed` parameter.
     *  - `>= 0`: explicit fold seed; overrides `seed` for CV only. */
    int cv_seed = -1;

    /** @brief Dummy-coupling mode of the sparse-K TCIENET penalty.
     *  Default: true (FOLDED) — dummy copy t joins its originating
     *  variable (t - p) mod p in the K-coupling, the exact generalization
     *  of GVS's layered group convention; dummy/null exchangeability then
     *  holds for arbitrary K (with K = sum_m 1_m 1_m^T / p_m the solver
     *  reproduces GVS-TCIENET bit-exactly). Set to false
     *  (INDEPENDENT_RIDGE: decoupled unit-ridge dummy block) ONLY for
     *  validation / equivalence runs — it preserves the K = I ≡ TCENET
     *  collapse but deforms exchangeability whenever K has off-diagonals
     *  (conservative for positive off-diagonals such as group-mean K,
     *  anti-conservative for negative ones such as Laplacians). */
    bool fold_dummy_coupling = true;

    /** @brief Cluster-aware dummy generation (GVS parity). Default: true.
     *  When true, dummies are drawn as per-cluster colored MVN layers on
     *  the clustering below via the DummyGenerator cluster mode — all
     *  strategies and memory mapping supported; only the PERMUTATION
     *  variants throw. Set to false ONLY for validation / equivalence runs
     *  against the canonical i.i.d. dummy path (e.g. the K = I ≡ TCENET
     *  anchor test): with an informed K, i.i.d. dummies deform the
     *  dummy/null exchangeability the FDR calibration relies on. */
    bool cluster_dummies = true;

    /** @brief User-supplied dummy clustering: 0-based cluster ID per
     *  variable (length p, contiguous IDs). Empty (default): clusters are
     *  discovered by correlation-HAC with `corr_max` / `hc_linkage`.
     *  Ignored when `cluster_dummies == false`. */
    std::vector<Eigen::Index> prior_groups;

    /** @brief Maximum allowed pairwise |corr| between variables from
     *  different clusters when clusters are discovered by HAC (the
     *  dendrogram is cut at height 1 - corr_max). Same semantics and
     *  default as TRexGVSSelector. Ignored when `prior_groups` is set or
     *  `cluster_dummies == false`. */
    double corr_max = 0.5;

    /** @brief Linkage method for the HAC dummy clustering. Ignored when
     *  `prior_groups` is set or `cluster_dummies == false`. */
    hac::LinkageMethod hc_linkage = hac::LinkageMethod::Single;

    /** @brief Base T-Rex algorithmic control parameters (nested).
     *  `solver_type` must remain TCIENET (the only solver with a general-K
     *  mode); the default here overrides the base class's own default
     *  (TLARS) accordingly. All other base knobs (K, L-loop strategy, warm
     *  starts, memory mapping, CD hyperparameters, ...) apply unchanged. */
    tc::TRexControlParameter trex_ctrl = [] {
        tc::TRexControlParameter t;
        t.solver_type = sd::SolverTypeForTRex::TCIENET;
        return t;
    }();
};


// ===================================================================================
// TRexTikhonovSelector
// ===================================================================================

/**
 * @class TRexTikhonovSelector
 *
 * @brief T-Rex variant driving the CCD IEN solver with a user-supplied
 *        general Tikhonov penalty matrix (see the file header).
 */
class TRexTikhonovSelector : public tc::TRexSelector {

public:

    /**
     * @brief Construct a TRexTikhonovSelector.
     *
     * @param X    Design matrix map (n x p); normalized in place, restored
     *             on destruction (base-class contract).
     * @param y    Response vector map (n).
     * @param tFDR Target FDR level in (0, 1].
     * @param trex_tik_ctrl Tikhonov control parameters (must carry a
     *             non-empty p x p `tikhonov_K`).
     * @param seed Seed for reproducible dummies / folds (< 0: entropy).
     * @param verbose Progress output.
     *
     * @throws std::invalid_argument on an empty or mis-dimensioned
     *         `tikhonov_K`, a `solver_type` other than TCIENET, malformed
     *         `prior_groups` / `corr_max`, or — with `cluster_dummies` on —
     *         a PERMUTATION-family L-loop strategy.
     */
    TRexTikhonovSelector(Eigen::Map<Eigen::MatrixXd>& X,
                         Eigen::Map<Eigen::VectorXd>& y,
                         double tFDR = 0.1,
                         TRexTikhonovControlParameter trex_tik_ctrl =
                             TRexTikhonovControlParameter(),
                         int seed = -1,
                         bool verbose = true);

    /** @brief Build K = Gamma^T Gamma from a regularization operator
     *  Gamma (m x p), pruned and compressed. */
    static Eigen::SparseMatrix<double> gammaToK(
        const Eigen::SparseMatrix<double>& gamma);

    /** @brief The lambda_2 actually used by the last select() run
     *  (user-supplied or CV-resolved); -1 before select(). */
    double getLambda2Used() const noexcept { return lambda2_; }

protected:

    /** @brief Builds the dummy cluster model (prior groups or
     *  correlation-HAC on the normalized X), installs it into the base
     *  DummyGenerator (cluster mode), and resolves lambda_2 (CV on the
     *  normalized X / centered y) before the L-loop runs. The base dummy
     *  hook and ExperimentRunner then run unchanged — the generator draws
     *  cluster layers for every strategy and the memory-mapped writers. */
    void onSelectBegin() override;

    /** @brief Base config plus the Tikhonov injection: sets
     *  `tikhonov_K` and overwrites `solver_params.lambda2` with the
     *  resolved penalty weight. */
    er::ExperimentRunnerConfig buildRunnerConfig(
        std::size_t num_dummies,
        std::size_t T_stop,
        bool use_warm_start,
        er::ExperimentStrategy strategy,
        std::size_t seed_factor = 0,
        std::size_t existing_cols_on_disk = 0) const override;

private:

    /** @brief lambda_2 resolution (user value or sparse-K CV tuners). */
    double computeLambda2() const;

    /** @brief Build the dummy cluster model: clusters from `prior_groups`
     *  or correlation-HAC on the normalized X_, then the per-cluster
     *  Cholesky factors (shared machinery, trex_cluster_dummies.hpp /
     *  trex_cluster_hac.hpp); installed into the base DummyGenerator via
     *  setClusterModel(). */
    void setupClusterModel();

    /** @brief Tikhonov control parameters (owns the K matrix the runner
     *  configs point into). */
    TRexTikhonovControlParameter trex_tik_ctrl_;

    /** @brief Resolved CV fold-permutation seed. */
    unsigned int resolved_cv_seed_{0};

    /** @brief Resolved penalty weight for the current select() run. */
    double lambda2_{-1.0};

    // ----- Cluster-aware dummy state (cluster_dummies == true) -----

    /** @brief Per-cluster column indices into X (0-based). Length M.
     *  Kept for diagnostics; the Cholesky factors live inside the
     *  DummyGenerator cluster model. */
    std::vector<std::vector<Eigen::Index>> clusters_list_;
};

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_tikhonov */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_TREX_TIKHONOV_HPP */
