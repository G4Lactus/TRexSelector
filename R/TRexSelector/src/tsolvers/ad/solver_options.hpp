// ===================================================================================
// solver_options.hpp
// ===================================================================================
/**
 * @file solver_options.hpp
 *
 * @brief SolverOptions — every T-solver input except the data X, D and y.
 *
 * @details One structure carries the preprocessing flags, the numerical
 *  tolerance, the tie-breaking configuration and every family
 *  hyperparameter, so a solver is constructed as
 *  `Solver(X, D, y, options)` (plus the structural inputs that are data,
 *  not options: the group vector of the IEN solvers, the Tikhonov matrix of
 *  the sparse-K solvers). Each family reads the fields it needs and ignores
 *  the rest; the per-field comments say which solvers consume a field. The
 *  defaults are the solver-level defaults (what a direct C++ caller got
 *  before); the T-Rex layer maps its own `SolverHyperparameters` onto this
 *  structure in the dispatch (trex_solver_dispatch.hpp) and keeps its own
 *  defaults there.
 */
// ===================================================================================
#ifndef TREX_TSOLVERS_SOLVER_OPTIONS_HPP
#define TREX_TSOLVERS_SOLVER_OPTIONS_HPP

// std includes
#include <cstddef>
#include <limits>

// tsolvers includes
#include <tsolvers/ad/solver_utils/solver_preprocessing.hpp>

// ===================================================================================

namespace trex::tsolvers::ad {

// ===================================================================================

/** @brief Column-scaling convention used by all solvers (unit-L2 or z-score). */
using ScalingMode = solver_utils::preprocessing::ScalingMode;

/**
 * @brief Variants of the Norm-Corrective GMP solver (TNCGMP).
 */
enum class NCGMPVariant : int {
    LineSearch = 0,       ///< Standard Matching Pursuit (1D projection)
    FullyCorrective = 1   ///< Orthogonal Matching Pursuit (least squares over the active set)
};

/**
 * @brief Structured penalty of the group CCD solver (TCDIENET).
 *
 *  - MEAN      : K = blockdiag(1_m 1_m^T / p_m) — the canonical IEN group-sum
 *                penalty.
 *  - CENTERING : K = blockdiag(I_{p_m} - 1_m 1_m^T / p_m) — within-group
 *                centering (complete-graph Laplacian per group). Null space:
 *                group-constant directions; singleton groups unpenalized.
 *                Mathematically identical to the TCDTIKNET FOLDED path with
 *                K_GC, at O(1) instead of O(p_m) per coordinate update.
 */
enum class GroupPenaltyVariant { MEAN, CENTERING };

// ===================================================================================

/**
 * @brief Every T-solver input except X, D and y.
 *
 * @note Construction-time only. The execution parameters (T_stop,
 *  early_stop) belong to executeStep(); serialization keeps its own record
 *  of the fields it needs.
 */
struct SolverOptions {

    // --- Preprocessing (every solver) -------------------------------------

    /** @brief Scale each column of X (and D) to unit L2 norm (or unit SD,
     *  see scaling_mode). Off when the caller pre-normalized (the T-Rex
     *  selectors). Default true. */
    bool normalize = true;

    /** @brief Center X, D and y (an intercept is fitted). Off when the caller
     *  pre-centered. Default true. */
    bool intercept = true;

    /** @brief The caller centered X, D and y (intercept = false on centered
     *  data, as in the T-Rex selectors). The centered columns span at most
     *  n - 1 dimensions, so the solver counts the intercept as with
     *  intercept = true: the active set is capped at n - 1 and the degrees
     *  of freedom and Cp include the intercept. The data are not touched.
     *  Redundant when intercept is true. Default false. */
    bool precentered = false;

    /** @brief Column-scaling convention applied when normalize is true:
     *  L2 (unit L2 norm, default) or ZSCORE (sample SD). */
    ScalingMode scaling_mode = ScalingMode::L2;

    /** @brief Print status and diagnostics during execution. Default false. */
    bool verbose = false;

    // --- Threads (every solver) -------------------------------------------

    /** @brief Thread budget of the solver's own OpenMP regions, the passes
     *  over the design (correlation refreshes, the step-length search, the
     *  T-OLS downdate, the standardization and its restore). 1 (default):
     *  serial; at least 1. A region runs in parallel only when the budget
     *  is above 1, the work of the pass exceeds
     *  TSolver_Base::kParallelWorkThreshold entries and the solver is not
     *  inside a parallel region of the caller. Not serialized: a loaded
     *  solver starts serial, setNumThreads() gives it a budget. */
    int n_threads = 1;

    // --- Numerics and tie-breaking (every solver) -------------------------

    /** @brief Numerical tolerance of the solver's internal comparisons
     *  (collinearity, step roots, ...). Default: machine epsilon. */
    double tol = std::numeric_limits<double>::epsilon();

    /** @brief Seed of the solver's tie-break engine (pruneTiedDummies
     *  shuffle; the exchangeable-tie draw). Default 0: a fixed seed, so a
     *  solver run is reproducible also where exact ties occur. < 0: seeded
     *  from std::random_device, i.e. nondeterministic. */
    long long tie_seed = 0;

    /** @brief Exchangeable-tie band width for the greedy solvers (TOMP, TAFS
     *  families), in units of the pairwise ranking-noise sd: statistically
     *  indistinguishable top candidates within highly correlated clusters
     *  are picked uniformly at random per step. 0 (default): off. Ignored by
     *  the path solvers. */
    double exch_tie_alpha = 0.0;

    /** @brief Minimum |correlation| for exchangeable-tie candidates in
     *  [0, 1). Default 0.5. Ignored unless exch_tie_alpha > 0. */
    double exch_tie_floor = 0.5;

    // --- Family hyperparameters --------------------------------------------

    /** @brief L2 (ridge / group-ridge / Tikhonov) penalty weight, >= 0.
     *  Read by TENET, TENETAug, TIENET, TIENETAug, TIKNET, TCDENET,
     *  TCDIENET, TCDTIKNET. Default 0 (no quadratic penalty: the elastic
     *  nets reduce to their lasso). */
    double lambda2 = 0.0;

    /** @brief Shrinkage step of TAFS in (0, 1]; 1 (default) is forward
     *  stepwise. Read by TAFS only. */
    double rho_afs = 1.0;

    /** @brief NCGMP variant. Read by TNCGMP only. Default LineSearch. */
    NCGMPVariant ncgmp_variant = NCGMPVariant::LineSearch;

    /** @brief TENETAug / TIENETAug: run the augmented system with a pure-LARS
     *  inner path that never drops variables (true) instead of the default
     *  LASSO inner path (false). */
    bool tenet_aug_use_lars = false;

    /** @brief Ridge weight of the dummy block: TCDENET (kappa * I on the
     *  dummies) and TCDTIKNET under INDEPENDENT_RIDGE coupling (ignored when
     *  fold_dummies is true). Default 1.0. */
    double kappa_dummy = 1.0;

    /** @brief TCDTIKNET dummy coupling: false (default) INDEPENDENT_RIDGE —
     *  the dummies get a decoupled kappa_dummy * I ridge; true FOLDED —
     *  dummy copy t joins variable (t - p) mod p in the K-coupling (GVS
     *  layered convention; requires D.cols() to be a multiple of X.cols()).
     *  TIKNET implements FOLDED exclusively and ignores this. */
    bool fold_dummies = false;

    /** @brief TCDIENET structured penalty. Default MEAN. */
    GroupPenaltyVariant group_penalty = GroupPenaltyVariant::MEAN;

    // --- CD family knobs (TCDLASSO, TCDENET, TCDIENET, TCDTIKNET) ---------

    /** @brief Relative lambda tolerance of the crossing search, in (0, 1).
     *  Default 1e-3. */
    double cd_lambda_rel_tol = 1e-3;

    /** @brief Certification-sweep tolerance (recorded steps), > 0.
     *  Default 1e-10. */
    double cd_tol_certify = 1e-10;

    /** @brief Probe-sweep tolerance (jump / bisection solves), > 0.
     *  Default 1e-8. */
    double cd_tol_probe = 1e-8;

    /** @brief Ever-active Gram slots cached before the solver falls back to
     *  naive residual sweeps. Default 400. */
    std::size_t cd_gram_cap = 400;

    /** @brief Stall guard: maximum coordinate sweeps per fixed-lambda solve.
     *  Default 2000. */
    std::size_t cd_max_sweeps = 2000;

    /** @brief Crossing localization: true (default) CERTIFIED — duality-gap
     *  and coordinate-status certificates, closed-form continuation, records
     *  at the knot; false LEGACY — KKT-guided jumps plus probe bisection
     *  with the support-nesting stop. */
    bool cd_certified_localization = true;
};

// ===================================================================================
} /* End of namespace trex::tsolvers::ad */

#endif /* TREX_TSOLVERS_SOLVER_OPTIONS_HPP */
