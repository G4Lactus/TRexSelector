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
 *      GVS owns the cluster-aware dummy machinery; this class uses the
 *      canonical i.i.d. T-Rex dummies),
 *    - banded / Laplacian K      -> smoothness-informed selection.
 *
 *  Dummies get the neutral unit ridge (kappa_dummy = 1), preserving the
 *  real/dummy exchangeability the FDR calibration relies on; an informed K
 *  should therefore keep diag(K) = 1 (see tcienet_solver.hpp).
 *
 *  Architecture:
 *  -------------
 *  Unlike GVS (private K-experiment loop), this class reuses the ENTIRE base
 *  machinery — dummy generation, L/T loops, warm starts, memory mapping —
 *  by injecting the penalty matrix into the shared solver dispatch:
 *  `buildRunnerConfig()` is overridden to set
 *  `ExperimentRunnerConfig::tikhonov_K` (forwarded to
 *  `SolverConfig::tikhonov_K`, where dispatchByType() routes TCIENET through
 *  its sparse-K constructor) and to overwrite `solver_params.lambda2` with
 *  the resolved penalty weight.
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

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Base T-Rex
#include <trex_selector_methods/trex_core/trex.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::trex_tikhonov
namespace trex::trex_selector_methods::trex_tikhonov {

// Local namespace aliases
namespace tc = trex::trex_selector_methods::trex_core;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;
namespace er = trex::trex_selector_methods::utils::experiment_runner;


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
     *         `tikhonov_K`, or a `solver_type` other than TCIENET.
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

    /** @brief Resolves lambda_2 (CV on the normalized X / centered y)
     *  before the L-loop runs. */
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

    /** @brief Tikhonov control parameters (owns the K matrix the runner
     *  configs point into). */
    TRexTikhonovControlParameter trex_tik_ctrl_;

    /** @brief Resolved CV fold-permutation seed. */
    unsigned int resolved_cv_seed_{0};

    /** @brief Resolved penalty weight for the current select() run. */
    double lambda2_{-1.0};
};

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_tikhonov */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_TREX_TIKHONOV_HPP */
