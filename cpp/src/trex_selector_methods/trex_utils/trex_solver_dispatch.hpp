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

// Solver includes
#include <tsolvers/tsolvers.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::solver_dispatch
namespace trex::trex_selector_methods::utils::solver_dispatch {

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
    TSTEPWISE,  // Terminating Stepwise
    TSTAGEWISE, // Terminating Stagewise
    //TIENET,   // Terminating Informed ENET (future) // TODO - add when implemented
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

};


// Aliases for clean reading in the if constexpr block
namespace lars = trex::tsolvers::linear_model::lars_based;
namespace omp  = trex::tsolvers::linear_model::omp_based;
namespace afs  = trex::tsolvers::linear_model::afs_based;


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
    if constexpr (std::is_same_v<TSolver, lars::TENET_Solver>) {

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
 * @return Beta path matrix (n_features + n_dummies) × n_steps).
 */
template <typename TSolver>
Eigen::MatrixXd dispatchSolver(const SolverConfig& cfg) {

    // 1. In-memory warm start: resume the retained solver. executeStep()
    //    continues the existing path up to the new T_stop; getBetaPath()
    //    returns the full accumulated path.
    if (cfg.warm_solver != nullptr) {
        cfg.warm_solver->setTolerance(cfg.hyperparams.tol);
        cfg.warm_solver->executeStep(cfg.T_stop, cfg.early_stop);
        return cfg.warm_solver->getBetaPath();
    }

    // 2. Serialized warm start: restore from disk, continue, re-save.
    if (cfg.use_warm_start && !cfg.solver_file.empty()) {
        TSolver solver = TSolver::load(cfg.solver_file, cfg.X, cfg.D);
        solver.setTolerance(cfg.hyperparams.tol);
        solver.executeStep(cfg.T_stop, cfg.early_stop);
        Eigen::MatrixXd pathMatrix = solver.getBetaPath();
        solver.save(cfg.solver_file);
        return pathMatrix;
    }

    // 3. Fresh construction.
    std::unique_ptr<TSolver> solver = makeSolverForConfig<TSolver>(cfg);
    solver->setTolerance(cfg.hyperparams.tol);
    solver->executeStep(cfg.T_stop, cfg.early_stop);
    Eigen::MatrixXd pathMatrix = solver->getBetaPath();
    if (!cfg.solver_file.empty()) solver->save(cfg.solver_file);
    if (cfg.retain_sink != nullptr) *cfg.retain_sink = std::move(solver);
    return pathMatrix;
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
 * @return Beta path matrix ((p + num_dummies) × n_steps).
 *
 * @throws std::invalid_argument if solver_type is not a recognized enumerator.
 */
inline Eigen::MatrixXd dispatchByType(SolverTypeForTRex solver_type, const SolverConfig& cfg) {
    switch (solver_type) {

        // LARS family
        case SolverTypeForTRex::TLARS:      return dispatchSolver<lars::TLARS_Solver>(cfg);
        case SolverTypeForTRex::TLASSO:     return dispatchSolver<lars::TLASSO_Solver>(cfg);
        case SolverTypeForTRex::TSTEPWISE:  return dispatchSolver<lars::TSTEPWISE_Solver>(cfg);
        case SolverTypeForTRex::TENET:      return dispatchSolver<lars::TENET_Solver>(cfg);
        case SolverTypeForTRex::TENET_AUG:  throw std::invalid_argument(
                "trex_solver_dispatch: TENET_AUG is a GVS-only solver; "
                "use TRexGVSSelector with gvs_type=EN.");
        case SolverTypeForTRex::TSTAGEWISE: return dispatchSolver<lars::TSTAGEWISE_Solver>(cfg);

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
