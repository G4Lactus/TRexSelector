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
 *  Tikhonov-Net solver (TCDTIKNET_Solver) in its general sparse-K
 *  mode: the quadratic penalty is (lambda_2 / 2) * beta^T K beta for a
 *  user-supplied PSD matrix K = Gamma^T Gamma (p x p).
 *  This generalizes:
 *    - K = I                     -> plain elastic net (TCDENET),
 *    - K = sum_m 1_m 1_m^T / p_m -> the group-informed IEN penalty that
 *      TRexGVSSelector builds from a clustering (kept separate on purpose:
 *      GVS owns the cluster-aware K-experiment loop; this class runs the
 *      shared base runner),
 *    - banded / Laplacian K      -> smoothness-informed selection.
 *
 *  Dummy coupling (default FOLDED, `fold_dummy_coupling`): dummy copies
 *  join their originating variable's K-coupling — the exact generalization
 *  of GVS's layered group convention — so dummy/null exchangeability holds
 *  for arbitrary K.
 *  The INDEPENDENT_RIDGE escape (decoupled kappa_dummy = 1
 *  dummy block) preserves the K = I ≡ TCDENET collapse for validation runs;
 *  under it, prefer diag(K) = 1 (see tcdtiknet_solver.hpp).
 *
 *  Dummy generation:
 *  -----------------
 *  Cluster-aware by default (`cluster_dummies = true`), on GVS parity:
 *  variables are partitioned into clusters (user-supplied `prior_groups`,
 *  else correlation-HAC with `corr_max` / `hc_linkage`), and dummies are
 *  drawn as per-cluster colored MVN layers via the shared machinery in
 *  trex_cluster_dummies.hpp.
 *  The same seed_seq scheme, same center-only / realized-norms convention
 *  as TRexGVSSelector.
 *  An informed K encodes structure among correlated variables;
 *  i.i.d. dummies would be systematically less competitive than correlated
 *  null reals there, deforming the dummy/null exchangeability the FDR
 *  calibration relies on.
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
 *  `SolverConfig::tikhonov_K`, where dispatchByType() routes TCDTIKNET
 *  through its sparse-K constructor) and to overwrite `solver_params.lambda2` with
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
 *
 *  The pilot rule (PILOT_KNEE, the default) decides lambda_2 from the random
 *  experiments themselves, still in `onSelectBegin()` and before the main
 *  L-loop: for every candidate of `pilot_grid` the shared ExperimentRunner
 *  runs `pilot_K` experiments with L_0 = p dummies to the stopping time
 *  `pilot_T`, on ONE dummy set common to every candidate, and the count of
 *  originals ahead of the T-th dummy in more than half of the experiments
 *  (the T-Rex's own count at voting level 1/2; or the calibrated pilot
 *  selection, `pilot_statistic`) is read off. The pick is the knee of that
 *  profile, the smallest lambda_2 whose count is within
 *  `pilot_knee_tolerance` of the largest — the selector's own principle
 *  applied to lambda_2 — at a fraction of the cost of one full selection
 *  per candidate: about |grid| x pilot_K paths at (L_0, pilot_T) against
 *  |grid| x K x (L-steps + T-steps) paths. The pilot dummies live in a
 *  reserved seed stream, so the main run is a standard T-Rex at a fixed
 *  lambda_2 whose dummies are independent of the pick; the pick itself
 *  depends on y and is outside the fixed-lambda_2 FDR proof — its cost was
 *  measured to be nil under the global null and at most 0.02 of TPR
 *  against the oracle pin under the alternatives in the Tikhonov T-Rex
 *  workspace. The record is in `getLambda2PilotResult()`.
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Base T-Rex
#include <trex_selector_methods/ad/trex_core/trex.hpp>

// Hierarchical clustering types (LinkageMethod for the dummy clustering)
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// lambda_2 tuner health record
#include <ml_methods/model_selection/lambda2_cv_diagnostics.hpp>

// The pick of the power / pilot rules (selector-free)
#include <trex_selector_methods/ad/trex_tikhonov/tik_lambda2_pick.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::ad::trex_tikhonov
namespace trex::trex_selector_methods::ad::trex_tikhonov {

// Local namespace aliases
namespace tc = trex::trex_selector_methods::ad::trex_core;
namespace disp = trex::trex_selector_methods::ad::utils::solver_dispatch;
namespace er = trex::trex_selector_methods::ad::utils::experiment_runner;
namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;
namespace ms = trex::ml_methods::model_selection;


// ===================================================================================
// Enums
// ===================================================================================

/**
 * @brief Selection rule for the Tikhonov penalty weight `lambda_2` when no
 *        user-supplied value is given. The four CV rules consume the penalty
 *        matrix K on the normalized data before the random experiments and
 *        return lambda_2 on the solver scale (no p/2 conversion; see the
 *        file header). The pilot rule reads lambda_2 off a few random
 *        experiments per candidate of `pilot_grid` (see the file header,
 *        "lambda_2 selection").
 */
enum class TikLambda2Method {
    /** @brief IEN-geometry coordinate-descent CV (lambda1 profiled), 1SE. */
    CV_1SE_IEN_CCD,
    /** @brief IEN-geometry coordinate-descent CV (lambda1 profiled), min. */
    CV_MIN_IEN_CCD,
    /** @brief Generalized-Tikhonov SVD ridge CV (lambda1 = 0 backbone), 1SE. */
    CV_1SE_TIK_SVD,
    /** @brief Generalized-Tikhonov SVD ridge CV (lambda1 = 0 backbone), min. */
    CV_MIN_TIK_SVD,
    /** @brief Pilot rule (the default): over `pilot_grid`, the smallest
     *  lambda_2 whose pilot selection count is within
     *  `pilot_knee_tolerance` of the largest. */
    PILOT_KNEE
};

/** @brief The count the pilot rule picks on, per grid point. Both are
 *  always recorded in `Lambda2PilotResult`. */
enum class PilotStatistic {
    /** @brief The number of originals ahead of the T_pilot-th dummy in more
     *  than half of the pilot experiments — the T-Rex's own count at voting
     *  level 1/2, without the FDP gate. Zero under the global null. */
    MAJORITY,
    /** @brief The T-Rex selection rule applied to the pilot data (Phi',
     *  FDP-hat on a grid of step 1/K_pilot, the largest selection subject
     *  to FDP-hat <= tFDR over t <= T_pilot). Coarser at a small K_pilot. */
    CALIBRATED,
    /** @brief The MAJORITY count restricted to the groups of `prior_groups`
     *  in which at least `pilot_group_fraction` of the members are ahead:
     *  a group that enters as a whole contributes all of its members
     *  ahead, a group that enters as a minority contributes nothing. The
     *  false positives a group prior admits under crosstalk are scattered
     *  members of null groups correlated with an active one (a few per
     *  touched group); they inflate MAJORITY as lambda_2 grows and pull a
     *  count-based pick upward in exactly the trials that are false-
     *  positive-prone, whereas an active group crosses the fraction as a
     *  whole. Needs `prior_groups` (refused otherwise). Falls to zero when
     *  no group reaches the fraction (very low SNR): the pick is then the
     *  grid's first point. */
    GROUP_MAJORITY
};

/** @brief What the pilot of the last select() / runPilotOnly() did. */
struct Lambda2PilotResult {

    /** @brief A pilot executed (false: lambda_2 came from the caller or a
     *  CV tuner). */
    bool ran = false;

    /** @brief The grid, ascending and de-duplicated. */
    std::vector<double> grid;

    /** @brief The count the pick was read off (MAJORITY or CALIBRATED),
     *  per grid point. */
    std::vector<std::size_t> counts;

    /** @brief The majority count per grid point and pilot stopping time
     *  t = 1 .. T_pilot (`curve_majority[g][t - 1]`). */
    std::vector<std::vector<std::size_t>> curve_majority;

    /** @brief The mean number of originals ahead of the t-th dummy, per
     *  grid point and t (the column sums of phi_T_mat; diagnostic). */
    std::vector<std::vector<double>> curve_soft;

    /** @brief The calibrated pilot selection count per grid point. */
    std::vector<std::size_t> calibrated;

    /** @brief The group-majority count per grid point (members ahead of the
     *  T_pilot-th dummy in the majority of experiments, summed over the
     *  groups in which at least `pilot_group_fraction` of the members are
     *  ahead), and the number of such groups; zeros without `prior_groups`. */
    std::vector<std::size_t> group_majority;
    std::vector<std::size_t> groups_qualifying;

    /** @brief Wall-clock seconds per grid point. */
    std::vector<double> probe_seconds;

    /** @brief The pick diagnostics on `counts`. */
    Lambda2PickInfo pick;

    /** @brief Index of the pick into `grid`, and the picked lambda_2. */
    std::size_t index   = 0;
    double      lambda2 = 0.0;

    /** @brief The pick sits on the first / last grid point. */
    bool at_lower_end = false;
    bool at_upper_end = false;

    /** @brief The budget actually used. */
    std::size_t K_pilot   = 0;
    std::size_t T_pilot   = 0;
    std::size_t L_0       = 0;
    std::size_t dummy_tag = 0;

    /** @brief Wall-clock seconds of the whole pilot. */
    double seconds = 0.0;
};


/**
 * @brief Structural form of the Tikhonov penalty (solver routing).
 *
 * @details SPARSE_GENERAL passes `tikhonov_K` to the solver's matrix-passing
 *  core (O(nnz(K col)) per coordinate update). GROUP_CENTERING declares
 *  K = blockdiag(I_{p_m} - 1_m 1_m^T / p_m) over `prior_groups` and routes
 *  the solver through the O(1) group-centering hooks instead — the
 *  mathematically identical fast path (FOLDED coupling by construction).
 *  With GROUP_CENTERING, `tikhonov_K` may be left empty: the selector builds
 *  it internally for the lambda_2 CV tuners.
 */
enum class TikPenaltyStructure { SPARSE_GENERAL, GROUP_CENTERING };


// ===================================================================================
// Control Parameters
// ===================================================================================

/**
 * @brief Tikhonov-specific control parameters.
 */
struct TRexTikhonovControlParameter {

    /** @brief Structural form of the penalty (see TikPenaltyStructure).
     *  Default SPARSE_GENERAL (matrix-passing). GROUP_CENTERING requires
     *  non-empty `prior_groups` and routes the solver through the O(1)
     *  group-centering hooks. */
    TikPenaltyStructure penalty_structure = TikPenaltyStructure::SPARSE_GENERAL;

    /** @brief General Tikhonov matrix K = Gamma^T Gamma (p x p, sparse,
     *  PSD). REQUIRED for SPARSE_GENERAL: an empty matrix throws at
     *  construction. Build from a regularization operator Gamma (m x p) via
     *  `TRexTikhonovSelector::gammaToK()`. Prefer diag(K) = 1 so the real
     *  columns carry the same ridge as the dummies (kappa_dummy = 1).
     *  OPTIONAL for GROUP_CENTERING: when empty, the selector builds
     *  K_GC = blockdiag(I - J/p_m) from `prior_groups` internally (the CV
     *  tuners consume it); when supplied it must equal that structure —
     *  the solver runs on the O(1) hooks either way. */
    Eigen::SparseMatrix<double> tikhonov_K;

    /** @brief User-supplied lambda_2 on the solver scale.
     *  - `< 0` (default `-1.0`): "not supplied" sentinel; triggers
     *    auto-computation via `lambda2_method`.
     *  - `== 0`: degenerate case -> pure TCDLASSO lasso path (no penalty).
     *  - `> 0` : fixed penalty weight. */
    double lambda_2 = -1.0;

    /** @brief Selection rule for auto-computing `lambda_2` when
     *  `lambda_2 < 0`. Default: PILOT_KNEE, the knee of the pilot profile;
     *  the CV tuners (well-posed in any p/n regime) are the alternatives. */
    TikLambda2Method lambda2_method = TikLambda2Method::PILOT_KNEE;

    /** @brief Number of folds for the lambda_2 CV. Default: 10. */
    int cv_n_folds = 10;

    /** @brief lambda_2 grid size for the CV_*_TIK_SVD analytic scan.
     *  Default: 100. */
    Eigen::Index cv_n_lambda = 100;

    /** @brief lambda_2 grid size for the CV_*_IEN_CCD 2D tuner. The scan is
     *  warm-started across the lambda_2 grid with glmnet-style path
     *  truncation decided by a full-data reference scan (cv.glmnet
     *  grid-once semantics), so cost grows sublinearly in this value.
     *  Default: 100 — resolution parity with the tuner's internal lambda_1
     *  path and glmnet's nlambda. */
    Eigen::Index cv_n_lambda2 = 100;

    /** @brief Length of the internal lambda_1 path of the CV_*_IEN_CCD 2D
     *  tuner. lambda_1 is a NUISANCE parameter there — profiled out, needed
     *  only to keep the contrast space well-posed — so it does not need
     *  glmnet's nlambda = 100 resolution. Cost in this value is U-shaped
     *  (coarser steps degrade the pathwise warm starts faster than the point
     *  count falls); 40 is the smallest length that still reproduces the
     *  100-point grid's lambda_2 picks exactly. Default: 40. */
    Eigen::Index cv_n_lambda1 = 40;

    /** @brief Support cap of the CV_*_IEN_CCD 2D tuner as a fraction of the
     *  smallest CV fold's training size n_train_min: the full-data
     *  reference scan stops each lambda_1 path at the first point with more
     *  than floor(cv_dfmax_ratio * n_train_min) nonzero coefficients
     *  (glmnet `dfmax`). `<= 0` disables the cap. Larger values allow larger
     *  supports at a higher tuner cost (past nnz ~ n_train the coordinate
     *  descent slows sharply). Must be finite. Default: 0.8. */
    double cv_dfmax_ratio = 0.8;

    /** @brief Seed for the deterministic CV fold-permutation RNG.
     *  - `-1` (default): derived from the T-Rex `seed` parameter.
     *  - `>= 0`: explicit fold seed; overrides `seed` for CV only. */
    int cv_seed = -1;

    // ----- Pilot rule (lambda2_method = PILOT_KNEE) -----

    /** @brief lambda_2 candidates on the WORKING column scale (the scale of
     *  `lambda_2`; the pilot runs the solver at these values verbatim, so no
     *  (n - 1) ZSCORE conversion applies, unlike the CV tuners). Sorted and
     *  de-duplicated internally. Keep 0 on it so that "no penalty" is an
     *  admissible answer: a prior that misfits the truth loses power
     *  monotonically and the rule then switches it off. */
    std::vector<double> pilot_grid = {0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
                                      1000.0, 3000.0};

    /** @brief Pilot experiments per grid point (>= 2; capped at
     *  `trex_ctrl.K`). The majority count has resolution 1/pilot_K. */
    std::size_t pilot_K = 10;

    /** @brief Stopping time of the pilot paths (>= 1; capped at
     *  min(L_0, ceil(n / 2)) as the T-loop caps T). The pick reads the
     *  count at this t; the counts at every t <= pilot_T are recorded. */
    std::size_t pilot_T = 5;

    /** @brief Pilot dummies L_0 = pilot_L_multiplier * p (in
     *  [1, trex_ctrl.max_dummy_multiplier]). */
    std::size_t pilot_L_multiplier = 1;

    /** @brief Selections below the largest pilot count that still count as
     *  "on the plateau" (>= 0); 0 reads the largest count. */
    double pilot_knee_tolerance = 1.0;

    /** @brief The count the pick is read off. */
    PilotStatistic pilot_statistic = PilotStatistic::MAJORITY;

    /** @brief GROUP_MAJORITY: the share of a group's members that must be
     *  ahead for the group to count, in (0, 1]; 1/2 = at least half. */
    double pilot_group_fraction = 0.5;

    /** @brief false (default): the pilot draws its dummies from a reserved
     *  seed stream, independent of the main run's — conditional on the
     *  pick, the main run is a standard T-Rex whose dummies are independent
     *  of the pick, and its selection equals a fixed-lambda_2 run's at that
     *  value with the same seed. true: the pilot reuses the main strategy's
     *  first dummy block (HCONCAT / STANDARD / SKIPL / SEEDED; the
     *  PERMUTATION family is refused). */
    bool pilot_shared_dummies = false;

    /** @brief Dummy-coupling mode of the sparse-K TCDTIKNET penalty.
     *  Default: true (FOLDED) — dummy copy t joins its originating
     *  variable (t - p) mod p in the K-coupling, the exact generalization
     *  of GVS's layered group convention; dummy/null exchangeability then
     *  holds for arbitrary K (with K = sum_m 1_m 1_m^T / p_m the solver
     *  reproduces GVS-TCDIENET bit-exactly). Set to false
     *  (INDEPENDENT_RIDGE: decoupled unit-ridge dummy block) ONLY for
     *  validation / equivalence runs — it preserves the K = I ≡ TCDENET
     *  collapse but deforms exchangeability whenever K has off-diagonals
     *  (conservative for positive off-diagonals such as group-mean K,
     *  anti-conservative for negative ones such as Laplacians). */
    bool fold_dummy_coupling = true;

    /** @brief Cluster-aware dummy generation (GVS parity). Default: true.
     *  When true, dummies are drawn as per-cluster colored MVN layers on
     *  the clustering below via the DummyGenerator cluster mode — all
     *  strategies and memory mapping supported; only the PERMUTATION
     *  variants throw. Set to false ONLY for validation / equivalence runs
     *  against the canonical i.i.d. dummy path (e.g. the K = I ≡ TCDENET
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
     *  `solver_type` must be one of the general-K solvers: TCDTIKNET (CCD,
     *  default) or TIKNET (native LARS homotopy, FOLDED-coupling only).
     *  With penalty_structure = GROUP_CENTERING, the TCDTIKNET dispatch
     *  routes to TCDIENET's O(1) group-centering hooks, so the solver's
     *  warnings (sweep budget, certification fallback) carry the tag
     *  [TCDIENET]. TIKNET consumes the internally built K_GC on its sparse
     *  path (the O(1) fast-path hooks are CCD-only). The default here
     *  overrides the base class's own default (TLARS) accordingly. All other
     *  base knobs (K, L-loop strategy, warm starts, memory mapping, CD
     *  hyperparameters, ...) apply unchanged. */
    tc::TRexControlParameter trex_ctrl = [] {
        tc::TRexControlParameter t;
        t.solver_type = disp::SolverTypeForTRex::TCDTIKNET;
        return t;
    }();
};


// ===================================================================================
// TRexTikhonovSelector
// ===================================================================================

/**
 * @class TRexTikhonovSelector
 *
 * @brief T-Rex variant driving the CCD Tikhonov-Net solver with a
 *        user-supplied general Tikhonov penalty matrix (see the file
 *        header).
 *
 *  References:
 *  -----------
 *   A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization Method," Soviet Math.
 *     Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl. Akad. Nauk SSSR 151 (1963), no. 3,
 *     501-504).
 *   Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov Regularization
 *     Problems in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25, 2018,
 *     doi: 10.1016/j.cam.2018.04.049.
 *   A. J. Smola and R. Kondor, "Kernels and Regularization on Graphs," in Learning Theory and Kernel Machines
 *     (COLT/Kernel 2003), Lecture Notes in Artificial Intelligence vol. 2777, 2003, pp. 144-158.
 *   F. R. K. Chung, Spectral Graph Theory. CBMS Regional Conference Series in Mathematics, no. 92. American
 *     Mathematical Society, 1997.
 *   A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems,"
 *     Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article in the Special 40th
 *     Anniversary Issue).
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
     *         `tikhonov_K`, a `solver_type` other than TCDTIKNET / TIKNET,
     *         malformed `prior_groups` / `corr_max`, or — with
     *         `cluster_dummies` on — a PERMUTATION-family L-loop strategy.
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

    /** @brief Build the within-group centering matrix
     *  K_GC = blockdiag(I_{p_m} - 1_m 1_m^T / p_m) from a 0-based
     *  contiguous group assignment (length p; singleton groups contribute
     *  nothing). The canonical GROUP_CENTERING penalty representation
     *  (used by the CV tuners; the solver runs on the O(1) hooks). */
    static Eigen::SparseMatrix<double> buildGroupCenteringK(
        const std::vector<Eigen::Index>& prior_groups);

    /** @brief The lambda_2 actually used by the last select() run
     *  (user-supplied or CV-resolved); -1 before select(). */
    double getLambda2Used() const noexcept { return lambda2_; }

    /** @brief Health record of the lambda_2 resolution of the last
     *  `select()` call (`ran == false`: the value came from the caller). */
    const ms::lambda2_cv_diagnostics& getLambda2CVDiagnostics() const noexcept {
        return lambda2_cv_diag_;
    }

    /** @brief What the pilot rule of the last `select()` /
     *  `runPilotOnly()` did (`ran == false`: no pilot executed). */
    const Lambda2PilotResult& getLambda2PilotResult() const noexcept {
        return lambda2_pilot_;
    }

    /**
     * @brief Run ONLY the lambda_2 pilot (cluster model + pilot experiments
     *        + pick) and release X, without the main selection.
     *
     * @details For callers that want the pick alone (a study that scores it
     *   against selections it already has). The object is spent afterwards:
     *   call either `runPilotOnly()` or `select()` on a selector, not both.
     *   Requires `lambda_2 < 0` and a PILOT_* `lambda2_method`.
     *
     * @throws std::invalid_argument when the control does not request a
     *         pilot rule.
     */
    const Lambda2PilotResult& runPilotOnly();

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

    /** @brief The control requests a pilot rule (lambda_2 < 0 and a PILOT_*
     *  method). */
    bool pilotRequested() const noexcept;

    /** @brief The lambda_2 pilot: for every grid point, `pilot_K` random
     *  experiments with L_0 dummies to T_pilot on ONE stored dummy set,
     *  the counts of `PilotStatistic`, then the pick. Leaves the main run's
     *  dummy streams untouched (reserved tag, or the strategy's own first
     *  block when `pilot_shared_dummies`). Returns the picked lambda_2 and
     *  fills `lambda2_pilot_`. */
    double runLambda2Pilot();

    /** @brief The seed tag under which the main strategy draws its first
     *  dummy block (the `pilot_shared_dummies` option). */
    std::size_t pilotDummyTag() const;

    /** @brief The reserved seed tag of an independent pilot: the main run
     *  only uses tags 0 (prefix-stable streams; in cluster mode layer li
     *  draws from stream li + 1) and 1 .. max_dummy_multiplier. */
    static constexpr std::size_t kPilotDummyTag = std::size_t{1} << 30U;

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

    /** @brief Health record of the last `computeLambda2()` call. Mutable
     *  because `computeLambda2()` is const — it reports what it did
     *  alongside the value it returns. */
    mutable ms::lambda2_cv_diagnostics lambda2_cv_diag_;

    /** @brief What the last pilot did (`ran == false` when none ran). */
    Lambda2PilotResult lambda2_pilot_;

    /** @brief GROUP_CENTERING: group ids as VectorXi for the solver
     *  dispatch (owned; runner configs point into it). */
    Eigen::VectorXi groups_vec_i_{};

    // ----- Cluster-aware dummy state (cluster_dummies == true) -----

    /** @brief Per-cluster column indices into X (0-based). Length M.
     *  Kept for diagnostics; the Cholesky factors live inside the
     *  DummyGenerator cluster model. */
    std::vector<std::vector<Eigen::Index>> clusters_list_;
};

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::ad::trex_tikhonov */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_TREX_TIKHONOV_HPP */
