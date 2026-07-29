// ===================================================================================
// trex_solver_dispatch.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_UTILS_SOLVER_DISPATCH_HPP
#define TREX_SELECTOR_METHODS_UTILS_SOLVER_DISPATCH_HPP
// ===================================================================================
/**
 * @file trex_solver_dispatch.hpp
 *
 * @brief Type-safe solver dispatch for TRex.
 *
 * @details
 *
 *  Design:
 *  -------
 *    1.  SolverConfig — bundles the parameters every solver needs (data refs,
 *        flags, optionals like lambda2).
 *
 *    2.  dispatchSolver<TSolver>() - a function template that encapsulates the
 *        full create-or-warm-start → execute → extract → optionally-save
 *        lifecycle. A single template handles all solver types uniformly.
 *
 *    3.  dispatchByType() — the one remaining switch/case that maps the
 *        runtime SolverTypeForTRex enum to a compile-time concrete type and
 *        forwards to dispatchSolver<TSolver>.
 *
 *  Adding a new solver:
 *    1.  Write the solver class (inheriting TSolver_Base) with the standard
 *        constructor, default constructor, save(), and static load().
 *    2.  Add one `case` line in dispatchByType().
 */
// ===================================================================================

// std includes
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Solver includes
#include <tsolvers/tsolvers.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::solver_dispatch
namespace trex::trex_selector_methods::utils::solver_dispatch {

// ===================================================================================

/** @brief Sparse beta path handed from the solvers to the T-Rex aggregation
 *  (per-step support + values; O(sum_t |support_t|) memory instead of the
 *  former dense (p + num_dummies) x steps matrix). */
using SparseBetaPath = trex::tsolvers::SparseBetaPath;

// ===================================================================================


// ===================================================================================
// Solver type enum for TRex dispatch
// ===================================================================================

/**
 * @brief Solver types available for the T-Rex Selector.
 */
enum class SolverTypeForTRex {

    // LARS-based solvers
    // -------------------------------
    TLARS,      // Terminating LARS
    TLASSO,     // Terminating LASSO
    TENET,      // Terminating ENET
    TENET_AUG,  // Terminating ENET via augmented LASSO (GVS only)
    TIENET,     // Terminating Informed ENET, native pathwise (GVS only)
    TIENET_AUG, // Terminating Informed ENET via augmented LASSO (GVS only)
    TSTEPWISE,  // Terminating Stepwise
    TSTAGEWISE, // Terminating Stagewise
    // -------------------------------

    // CCD-based solvers (exact penalized-lasso minimizers per crossing)
    // -------------------------------
    TCCD,       // Terminating Cyclic Coordinate Descent (LASSO)
    TCENET,     // Terminating CCD Elastic Net (diagonal ridge)
    TCIENET,    // Terminating CCD Informed Elastic Net (GVS only)
    // -------------------------------

    // OMP-based solvers
    // -------------------------------
    TOMP,       // Terminating OMP
    TGP,        // Terminating Gradient Pursuit
    TACGP,      // Terminating Approximate Conjugate Gradient Pursuit
    TMP,        // Terminating Matching Pursuit
    TNCGMP,     // Terminating Norm-Corrected Generalized Matching Pursuit
    TOOLS,      // Terminating Orthogonal Least Squares
    // -------------------------------

    // AFS-based solvers
    // --------------------------------
    TAFS
    // --------------------------------
};


// ===================================================================================
// Encapsulated Solver Hyperparameters
// ===================================================================================
/**
 * @brief Bundles all specific hyperparameters required by individual solver algorithms.
 *
 * @details Adding a new parameter here automatically makes it available to the
 * dispatcher without touching intermediate routing layers.
 */
struct SolverHyperparameters {
    /** @brief L2 penalty for ENET-type solvers (ignored by non-using solvers). */
    double lambda2 = 0.1;   //
    /** @brief Shrinkage step size for AFS solver in (0, 1] (ignored by non-using solvers). */
    double rho_afs = 0.3;   //
    /** @brief Variant selector for NCGMP solver
     * (0: Line Search/MP, 1: Fully Corrective/OMP) (ignored by non-using solvers). */
    int ncgmp_variant = 0;  //
    /** @brief Numerical tolerance for solver steps. */
    double tol = 1e-6;      //
    /** @brief Exchangeable-tie band width for greedy solvers (TOMP/TAFS), in
     * units of the pairwise ranking-noise sd. When > 0, statistically
     * indistinguishable top candidates within highly correlated clusters are
     * picked uniformly at random per step, restoring the within-experiment
     * occurrence spread that the DA deflation's FDR control relies on.
     * Recommended for greedy solvers under trex+DA: 0.25.
     * 0 = off (exact legacy behavior; ignored by path solvers). */
    double exch_tie_alpha = 0.0;
    /** @brief Minimum |correlation| for exchangeable-tie candidates in
     * (0, 1) (ignored unless exch_tie_alpha > 0). */
    double exch_tie_floor = 0.5;

    // --- CD-family knobs (TCCD / TCENET / TCIENET; ignored by other solvers) ---

    /** @brief Relative lambda tolerance of the CD crossing bisection.
     * The TRex-layer default (1e-5) is deliberately tighter than the
     * solver's own generic default (1e-3): exact crossing localization
     * keeps the recorded supports aligned with the LARS-path variants
     * (a loose lambda window can pull a borderline real into the crossing
     * step). <= 0 keeps the solver's own default. */
    double cd_lambda_rel_tol = 1e-5;

    /** @brief CD certification-sweep tolerance (recorded steps). Applied
     * together with cd_tol_probe; <= 0 (default) keeps the solver's own
     * defaults (1e-10 / 1e-8). */
    double cd_tol_certify = -1.0;

    /** @brief CD probe-sweep tolerance (jump/bisection solves). See
     * cd_tol_certify. */
    double cd_tol_probe = -1.0;

    /** @brief Max ever-active gram slots before the CD solver falls back to
     * naive residual sweeps. 0 (default) keeps the solver's own default
     * (400). */
    std::size_t cd_gram_cap = 0;

    /** @brief Stall guard: max coordinate sweeps per fixed-lambda solve.
     * 0 (default) keeps the solver's own default (2000). */
    std::size_t cd_max_sweeps = 0;
};


// ===================================================================================
// SolverConfig — everything a solver needs to be created or warm-started
// ===================================================================================

/**
 * @brief Bundles all parameters required to construct or warm-start any TSolver.
 *
 * @note References to X, D, y must outlive the SolverConfig (non-owning).
 */
struct SolverConfig {

    // --- Data references (non-owning, must outlive config) ---

    /** @brief Reference to the original design matrix X (n x p). */
    Eigen::Map<Eigen::MatrixXd>& X;

    /** @brief Reference to the dummy design matrix D (n x num_dummies). */
    Eigen::Map<Eigen::MatrixXd>& D;

    /** @brief Reference to the response vector y (n x 1). */
    Eigen::Map<Eigen::VectorXd>& y;

    // --- Solver construction flags ---

    /** @brief Whether the solver should normalize columns internally.
     *  In TRex context always false (TRex normalizes X/D beforehand).
     */
    bool normalize = false;

    /** @brief Whether the solver should fit an intercept internally.
     *  In TRex context always false (TRex centers y beforehand).
     */
    bool intercept = false;

    /** @brief Enable verbose output from the solver. */
    bool verbose   = false;

    /** @brief Internal scaling mode applied by the solver to its own columns.
     *  In TRex context X/D are pre-normalized, so this is usually L2; set to
     *  ZSCORE only when the caller wants the solver to z-score internally
     *  (e.g. augmented EN paths that must keep the ridge block consistent).
     */
    trex::tsolvers::ScalingMode scaling_mode = trex::tsolvers::ScalingMode::L2;

    // --- Execution parameters ---

    /** @brief Early stopping threshold (number of active dummies). */
    std::size_t T_stop = 0;

    /** @brief If true, stop when count_active_dummies >= T_stop. */
    bool early_stop = true;

    // --- Warm-start / serialization ---

    /** @brief If true, attempt to load from solver_file before executing. */
    bool use_warm_start = false;

    /**
     * @brief Path to the serialized solver state file. Empty string means "no serialization".
     */
    std::string solver_file{};

    // --- Solver-specific extras ---
    /** @brief Encapsulated solver hyperparameters for t-solvers. */
    SolverHyperparameters hyperparams{};

    /** @brief Deterministic tie-break seed for the solver's internal RNG
     *  (pruneTiedDummies shuffle). < 0 keeps the solver's default
     *  std::random_device seeding (nondeterministic). Applied only on fresh
     *  construction — warm-started / deserialized solvers keep their RNG
     *  state so the shuffle stream continues instead of restarting. */
    long long tie_seed = -1;

    // --- In-memory warm start ---

    /** @brief Retained solver to continue in-memory (nullptr = none).
     *  When set, dispatchSolver() skips construction entirely and resumes
     *  this solver via executeStep(T_stop, early_stop). The solver holds
     *  non-owning views into X and its dummy matrix from the run that
     *  created it; cfg.X / cfg.D are ignored on this path. */
    tsolvers::TSolver_Base* warm_solver = nullptr;

    /** @brief Retention sink for the freshly constructed solver (nullptr = none).
     *  When set, dispatchSolver() heap-allocates the solver and moves it here
     *  after execution so the caller can retain it for the next T-step.
     *  The caller must keep the X and D buffers alive for as long as the
     *  solver is retained. */
    std::unique_ptr<tsolvers::TSolver_Base>* retain_sink = nullptr;

    // --- General-Tikhonov penalty (TCIENET sparse-K dispatch) ---

    /** @brief Non-owning pointer to a general Tikhonov matrix K = Gamma^T
     *  Gamma (p x p, PSD; nullptr = none). When set, TCIENET becomes
     *  dispatchable through dispatchByType() via its sparse-K constructor
     *  (used by TRexTikhonovSelector). Must outlive the dispatch call; on
     *  serialized warm starts K travels inside the solver state instead. */
    const Eigen::SparseMatrix<double>* tikhonov_K = nullptr;

    /** @brief Dummy-coupling mode of the sparse-K TCIENET dispatch:
     *  false = INDEPENDENT_RIDGE (kappa_dummy * I dummy block; preserves
     *  the K = I ≡ TCENET collapse), true = FOLDED (dummy copies join
     *  their originating variable's K-coupling — GVS layered convention;
     *  restores dummy/null exchangeability for arbitrary K). Ignored
     *  when tikhonov_K is nullptr. */
    bool tikhonov_fold_dummies = false;

};


// Aliases for clean reading in the if constexpr block
namespace lars = trex::tsolvers::linear_model::lars_based;
namespace cd   = trex::tsolvers::linear_model::cd_based;
namespace omp  = trex::tsolvers::linear_model::omp_based;
namespace afs  = trex::tsolvers::linear_model::afs_based;


/**
 * @brief Apply the CD-family hyperparameter knobs to a CCD solver.
 *
 * @details Sentinel values (<= 0 / 0) keep the solver's own defaults. Called
 *          on fresh construction and after deserialization (the knobs are
 *          runtime configuration, not part of the serialized path state).
 *          In-memory warm-started solvers keep their construction-time knobs.
 *          Also used directly by TRexGVSSelector, which constructs its CCD
 *          solvers outside this dispatch.
 */
inline void applyCdKnobs(cd::TCCD_Solver& solver,
                         const SolverHyperparameters& hp) {
    if (hp.cd_lambda_rel_tol > 0.0) {
        solver.setLambdaRelTol(hp.cd_lambda_rel_tol);
    }
    if (hp.cd_tol_certify > 0.0 && hp.cd_tol_probe > 0.0) {
        solver.setCdTolerances(hp.cd_tol_certify, hp.cd_tol_probe);
    }
    if (hp.cd_gram_cap > 0) {
        solver.setGramCap(hp.cd_gram_cap);
    }
    if (hp.cd_max_sweeps > 0) {
        solver.setMaxSweepsPerSolve(hp.cd_max_sweeps);
    }
}


/**
 * @brief Construct a solver on the heap from the config.
 *
 * @details Dispatches on the solver-specific constructor signature via
 *          `if constexpr`:
 *          - TENET:  requires an extra lambda2 hyperparameter.
 *          - TAFS:   requires an extra rho_afs hyperparameter.
 *          - TNCGMP: requires an NCGMPVariant enum.
 *          - TOMP / TLARS: carry an algorithm_type before scaling_mode.
 *          - All other solvers: standard (X, D, y, normalize, intercept,
 *            verbose, scaling_mode) constructor.
 *
 * @tparam TSolver Concrete solver type (must satisfy the TSolver_Base interface).
 *
 * @param cfg Populated SolverConfig with data references and execution flags.
 *
 * @return Owning pointer to the freshly constructed solver.
 */
template <typename TSolver>
std::unique_ptr<TSolver> makeSolverForConfig(const SolverConfig& cfg) {
    if constexpr (std::is_same_v<TSolver, lars::TENET_Solver> ||
                  std::is_same_v<TSolver, cd::TCENET_Solver>) {

        // TENET / TCENET share the (X, D, y, lambda2, ...) signature.
        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, cfg.hyperparams.lambda2, cfg.normalize,
            cfg.intercept, cfg.verbose, cfg.scaling_mode);

    } else if constexpr (std::is_same_v<TSolver, afs::TAFS_Solver>) {

        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, cfg.hyperparams.rho_afs, cfg.normalize,
            cfg.intercept, cfg.verbose, cfg.scaling_mode);

    } else if constexpr (std::is_same_v<TSolver, omp::TNCGMP_Solver>) {

        auto variant =
            static_cast<omp::NCGMPVariant>(cfg.hyperparams.ncgmp_variant);
        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, variant, cfg.normalize, cfg.intercept,
            cfg.verbose, cfg.scaling_mode);

    } else if constexpr (std::is_same_v<TSolver, omp::TOMP_Solver>) {

        // Bare TOMP_Solver's public ctor carries an algorithm_type before
        // scaling_mode; pass it explicitly so scaling_mode binds correctly.
        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, cfg.normalize, cfg.intercept,
            cfg.verbose, omp::SolverTypeOMPBased::TOMP, cfg.scaling_mode);

    } else if constexpr (std::is_same_v<TSolver, lars::TLARS_Solver>) {

        // Bare TLARS_Solver's public ctor carries an algorithm_type before
        // scaling_mode; pass it explicitly so scaling_mode binds correctly.
        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, cfg.normalize, cfg.intercept,
            cfg.verbose, lars::SolverTypeLarsBased::TLARS, cfg.scaling_mode);

    } else if constexpr (std::is_same_v<TSolver, cd::TCIENET_Solver>) {

        // General-Tikhonov CCD: requires the sparse penalty matrix K
        // (group-mode construction stays GVS-internal).
        if (cfg.tikhonov_K == nullptr) {
            throw std::invalid_argument(
                "trex_solver_dispatch: TCIENET requires a Tikhonov matrix "
                "(SolverConfig::tikhonov_K) on the general-K dispatch path.");
        }
        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, cfg.hyperparams.lambda2, *cfg.tikhonov_K,
            cfg.normalize, cfg.intercept, cfg.verbose, cfg.scaling_mode,
            /*kappa_dummy=*/1.0, cfg.tikhonov_fold_dummies);

    } else if constexpr (std::is_same_v<TSolver, cd::TCCD_Solver>) {

        // Bare TCCD_Solver's public ctor carries an algorithm_type before
        // scaling_mode; pass it explicitly so scaling_mode binds correctly.
        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, cfg.normalize, cfg.intercept,
            cfg.verbose, cd::SolverTypeCdBased::TCCD, cfg.scaling_mode);

    } else {

        return std::make_unique<TSolver>(
            cfg.X, cfg.D, cfg.y, cfg.normalize, cfg.intercept, cfg.verbose,
            cfg.scaling_mode);

    }
}


/**
 * @brief Execute a single solver lifecycle: construct (or warm-start), run, extract path,
 *        and optionally serialize.
 *
 * @details Three paths, checked in order:
 *          1. In-memory warm start (cfg.warm_solver != nullptr): the retained
 *             solver is resumed via executeStep(T_stop, early_stop); no
 *             construction, no serialization.
 *          2. Serialized warm start (cfg.use_warm_start && solver_file set):
 *             the solver is deserialized from disk, reconnected to cfg.X /
 *             cfg.D, continued, and re-saved.
 *          3. Fresh construction via makeSolverForConfig(). If a solver_file
 *             is set, the state is saved for later resumption; if
 *             cfg.retain_sink is set, ownership of the solver is handed to
 *             the caller for in-memory retention.
 *
 * @tparam TSolver Concrete solver type (must satisfy the TSolver_Base interface).
 *
 * @param cfg Populated SolverConfig with data references and execution flags.
 *
 * @return Sparse beta path ((p + num_dummies)-space support per step).
 */
template <typename TSolver>
SparseBetaPath dispatchSolver(const SolverConfig& cfg) {

    // 1. In-memory warm start: resume the retained solver. executeStep()
    //    continues the existing path up to the new T_stop; getBetaPathSparse()
    //    returns the full accumulated path.
    if (cfg.warm_solver != nullptr) {
        cfg.warm_solver->setTolerance(cfg.hyperparams.tol);
        cfg.warm_solver->setExchangeableTie(cfg.hyperparams.exch_tie_alpha,
                                            cfg.hyperparams.exch_tie_floor);
        cfg.warm_solver->executeStep(cfg.T_stop, cfg.early_stop);
        return cfg.warm_solver->getBetaPathSparse();
    }

    // 2. Serialized warm start: restore from disk, continue, re-save.
    if (cfg.use_warm_start && !cfg.solver_file.empty()) {
        TSolver solver = TSolver::load(cfg.solver_file, cfg.X, cfg.D);
        solver.setTolerance(cfg.hyperparams.tol);
        solver.setExchangeableTie(cfg.hyperparams.exch_tie_alpha,
                                  cfg.hyperparams.exch_tie_floor);
        if constexpr (std::is_base_of_v<cd::TCCD_Solver, TSolver>) {
            applyCdKnobs(solver, cfg.hyperparams);
        }
        solver.executeStep(cfg.T_stop, cfg.early_stop);
        SparseBetaPath path = solver.getBetaPathSparse();
        solver.save(cfg.solver_file);
        return path;
    }

    // 3. Fresh construction.
    std::unique_ptr<TSolver> solver = makeSolverForConfig<TSolver>(cfg);
    solver->setTolerance(cfg.hyperparams.tol);
    solver->setExchangeableTie(cfg.hyperparams.exch_tie_alpha,
                               cfg.hyperparams.exch_tie_floor);
    if constexpr (std::is_base_of_v<cd::TCCD_Solver, TSolver>) {
        applyCdKnobs(*solver, cfg.hyperparams);
    }
    if (cfg.tie_seed >= 0) {
        // Deterministic tie-break shuffles for reproducible selections
        // (user-seeded runs); < 0 keeps random_device seeding.
        solver->setTieSeed(static_cast<std::uint32_t>(cfg.tie_seed));
    }
    solver->executeStep(cfg.T_stop, cfg.early_stop);
    SparseBetaPath path = solver->getBetaPathSparse();
    if (!cfg.solver_file.empty()) solver->save(cfg.solver_file);
    if (cfg.retain_sink != nullptr) *cfg.retain_sink = std::move(solver);
    return path;
}


/**
 * @brief Map a runtime SolverTypeForTRex enum to a concrete solver type and dispatch.
 *
 * @details The sole switch/case in the dispatch layer. Each case instantiates the
 *          corresponding solver type via dispatchSolver<TSolver>(cfg).
 *          To add a new solver: write the class, add one case here.
 *
 * @param solver_type  Runtime enum value identifying the desired solver.
 * @param cfg          Populated SolverConfig with data references and execution flags.
 *
 * @return Sparse beta path ((p + num_dummies)-space support per step).
 *
 * @throws std::invalid_argument if solver_type is not a recognized enumerator.
 */
inline SparseBetaPath dispatchByType(SolverTypeForTRex solver_type, const SolverConfig& cfg) {
    switch (solver_type) {

        // LARS family
        case SolverTypeForTRex::TLARS:      return dispatchSolver<lars::TLARS_Solver>(cfg);
        case SolverTypeForTRex::TLASSO:     return dispatchSolver<lars::TLASSO_Solver>(cfg);
        case SolverTypeForTRex::TSTEPWISE:  return dispatchSolver<lars::TSTEPWISE_Solver>(cfg);
        case SolverTypeForTRex::TENET:      return dispatchSolver<lars::TENET_Solver>(cfg);
        case SolverTypeForTRex::TENET_AUG:  throw std::invalid_argument(
                "trex_solver_dispatch: TENET_AUG is a GVS-only solver; "
                "use TRexGVSSelector with gvs_type=EN.");
        case SolverTypeForTRex::TIENET:     throw std::invalid_argument(
                "trex_solver_dispatch: TIENET is a GVS-only solver (needs a "
                "group assignment); use TRexGVSSelector with gvs_type=IEN.");
        case SolverTypeForTRex::TIENET_AUG: throw std::invalid_argument(
                "trex_solver_dispatch: TIENET_AUG is a GVS-only solver; "
                "use TRexGVSSelector with gvs_type=IEN.");
        case SolverTypeForTRex::TSTAGEWISE: return dispatchSolver<lars::TSTAGEWISE_Solver>(cfg);

        // CCD family
        case SolverTypeForTRex::TCCD:       return dispatchSolver<cd::TCCD_Solver>(cfg);
        case SolverTypeForTRex::TCENET:     return dispatchSolver<cd::TCENET_Solver>(cfg);
        case SolverTypeForTRex::TCIENET:
            // Dispatchable on the general-K path (TRexTikhonovSelector sets
            // cfg.tikhonov_K); without a penalty matrix it stays a
            // group-informed GVS-only solver.
            if (cfg.tikhonov_K != nullptr) {
                return dispatchSolver<cd::TCIENET_Solver>(cfg);
            }
            throw std::invalid_argument(
                "trex_solver_dispatch: TCIENET needs a group assignment "
                "(TRexGVSSelector with gvs_type=IEN) or a Tikhonov matrix "
                "(TRexTikhonovSelector / SolverConfig::tikhonov_K).");

        // OMP family
        case SolverTypeForTRex::TOMP:       return dispatchSolver<omp::TOMP_Solver>(cfg);
        case SolverTypeForTRex::TGP:        return dispatchSolver<omp::TGP_Solver>(cfg);
        case SolverTypeForTRex::TACGP:      return dispatchSolver<omp::TACGP_Solver>(cfg);
        case SolverTypeForTRex::TMP:        return dispatchSolver<omp::TMP_Solver>(cfg);
        case SolverTypeForTRex::TNCGMP:     return dispatchSolver<omp::TNCGMP_Solver>(cfg);
        case SolverTypeForTRex::TOOLS:      return dispatchSolver<omp::TOOLS_Solver>(cfg);

        // AFS family
        case SolverTypeForTRex::TAFS:       return dispatchSolver<afs::TAFS_Solver>(cfg);

        default:
            throw std::invalid_argument("trex_solver_dispatch: unsupported SolverTypeForTRex");
    }
}

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::utils::solver_dispatch */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_UTILS_SOLVER_DISPATCH_HPP */
