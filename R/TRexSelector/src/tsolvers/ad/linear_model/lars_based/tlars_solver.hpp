// ===================================================================================
// tlars_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TLARS_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TLARS_SOLVER_HPP
// ===================================================================================
/**
 * @file tlars_solver.hpp
 *
 * @brief Header file for the Terminating Least Angle Regression (T-LARS) solver.
 *
 * @details The TLARS_Solver class implements the T-LARS algorithm for variable
 * selection in linear regression, with support for dummy variable augmentation.
 * It inherits state management, standardization, and diagnostics from the
 * TSolver_Base class.
 *
 * Furthermore, it serves as mother class for:
 * - TLASSO_solver,
 * - TSTEPWISE_solver,
 * - TENET_solver,
 * - TIENET_solver.
 */
// ===================================================================================

// std includes
#include <string>
#include <utility>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Cereal includes
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

// TSolver Base class include
#include <tsolvers/ad/tsolver_base.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::ad::linear_model::lars_based namespace
namespace trex::tsolvers::ad::linear_model::lars_based {

// ===================================================================================

/**
 * @brief Enumerates all supported T-Solver family algorithms with
 * step-wise solution path based on the Least Angle Regression (LARS) framework.
 *
 * Used for algorithm selection, configuration, and logging/reporting.
 */
enum class SolverTypeLarsBased: int {
    TLARS      = 0,       // Terminating Least Angle Regression (T-LARS)
    TLASSO     = 1,       // T-LARS/LASSO with variable removal
    TSTEPWISE  = 2,       // T-Stepwise selection (forward stepwise without variable removal)
    TENET      = 3,       // T-Elastic Net (L1 + L2) regularization
    TIENET     = 4,       // T-Informed Elastic Net
    TSTAGEWISE = 5,       // T-Stagewise regression
    TIKNET     = 6        // T-Tikhonov Net (general sparse-K quadratic penalty)
};


/**
 * @brief Terminating Least Angle Regression (T-LARS) solver.
 * Supports in-memory and memory-mapped data over Eigen::Map.
 *
 *  References:
 *  -----------
 *   B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
 *     no. 2, pp. 407-499, 2004.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TLARS_Solver : public TSolver_Base {
protected:
    // ==========================================================================
    // LARS specific state variables
    // ==========================================================================

    /** @brief Regularization parameter sequence (max. correlation per step). */
    std::vector<double> lambda_;

    /**
    * @brief Kahan summation compensation vector for incremental correlation updates.
     * Used to maintain numerical precision across multiple steps.
     */
    Eigen::VectorXd correlation_compensation_{};

    /** @brief Equiangular vector normalization value for current step. */
    double A_A_{};

    /** @brief Signs for active set variables (+1 or -1). */
    std::vector<int> Sign_;

    /** @brief Kahan summation refresh interval (steps before full recomputation). */
    std::size_t kahan_refresh_interval_{20};

    /** @brief Algorithm variant type. */
    SolverTypeLarsBased algo_type_{SolverTypeLarsBased::TLARS};

    /** @brief Monotonic counter of variable removals. Stays 0 in plain T-LARS;
     *  maintained by cycling descendants (T-LASSO, T-ENET, T-Stagewise). */
    std::size_t num_removals_{0};

    /** @brief Pending drop-recovery step, shared by the
     *  cycling descendants (T-LASSO, T-ENET, T-IENET): set when the previous
     *  step ended in a LASSO-style drop. The next executeStep iteration then
     *  adds NO variable and moves along the REDUCED active set until an
     *  inactive genuinely catches up -- the R-lars / CRAN-tlars
     *  drop-breakpoint semantics. Without it, the just-dropped variable
     *  (still exactly at the equicorrelation level) was re-added immediately
     *  and its coefficient rode a non-lasso branch (sign(beta) != sign(c)).
     *  Stays false in plain T-LARS (no drops). Serialized so warm starts
     *  resume correctly. */
    bool drop_recovery_pending_{false};

    /** @brief Variables dropped by the MOST RECENT processLassoDrops call.
     *  The following recovery step excludes them from candidacy: a dropped
     *  variable cannot catch up at the very breakpoint it left (its
     *  correlation falls strictly below the active level along the reduced
     *  direction -- Efron's lemma); only a scale-DEPENDENT fp-noise root of
     *  the step-size computation says otherwise. Genuine re-entry happens
     *  at a later breakpoint. Serialized for warm-start correctness. */
    std::vector<std::size_t> last_dropped_vars_{};

    /** @brief Protected constructor for inheritance. */
    explicit TLARS_Solver(SolverTypeLarsBased type);

    /**
     * @brief Protected data constructor for inheritance: the public
     *        constructor with the algorithm variant passed through.
     */
    TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y,
                 const SolverOptions& options,
                 SolverTypeLarsBased algorithm_type);

public:

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    /**
     * @brief Construct a TLARS_Solver object using data and options.
     *
     * @param X Original design matrix (n x p).
     * @param D Dummy design matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param options Every other input (SolverOptions): preprocessing flags,
     *        tolerance and tie-breaking. Plain T-LARS reads no family
     *        hyperparameter.
     *
     * @note X and D are not copied nor owned, so changes to X or D alter the underlying
     *       data.
     *
     * @note Has an internal column-dropping logic for unit L2-norm columns
     *       which is active if options.normalize is true. Otherwise ensure
     *       that you exclude collinear columns a priori before passing X to
     *       the algorithm.
     */
    TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y,
                 const SolverOptions& options = SolverOptions{});

    /**
     * @brief Default constructor for serialization or deferred initialization.
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TLARS_Solver();

    /**
     * @brief Virtual destructor (X_ pointer is not owned, so not deleted).
     * @note Present for completeness and to support inheritance.
     */
    virtual ~TLARS_Solver() override = default;

    /** @brief Deleted copy constructor and assignment. */
    TLARS_Solver(const TLARS_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TLARS_Solver& operator=(const TLARS_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TLARS_Solver(TLARS_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TLARS_Solver& operator=(TLARS_Solver&&) = delete;

    // ============================================================================
    // Main Algorithm (Overrides Pure Virtual)
    // ============================================================================

    /**
     * @brief Compute T-LARS solution path with early stopping based on dummy
     * threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold,
     * (0 = full solution path, default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts and
     * serialization.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /**
     * @brief Convert enum to human-readable string for reporting/logging.
     *
     * @param type The SolverTypeLarsBased enum value.
     *
     * @return Static string name.
     */
    std::string solverTypeToString() const override{
        switch(algo_type_) {
            case SolverTypeLarsBased::TENET:       return "TENET";
            case SolverTypeLarsBased::TIENET:      return "TIENET";
            case SolverTypeLarsBased::TLARS:       return "TLARS";
            case SolverTypeLarsBased::TLASSO:      return "TLASSO";
            case SolverTypeLarsBased::TSTEPWISE:   return "TSTEPWISE";
            case SolverTypeLarsBased::TSTAGEWISE:  return "TSTAGEWISE";
            case SolverTypeLarsBased::TIKNET:      return "TIKNET";
            default:                               return "UnknownSolver";
        }
    }

    // ============================================================================
    // Specific Getters and Setters for TLARS
    // ============================================================================

    /** @brief Get regularization parameter sequence (lambda = max. correlation per step).
     *  @note T-ENET uses a different convention: an initial entry followed by
     *  the decreasing L1 penalty per step, i.e. size() == steps + 1. */
    const std::vector<double>& getLambda() const noexcept { return lambda_; }

    /**
     * @brief Set Kahan summation refresh interval (steps between full recomputations).
     *
     * @param interval Number of steps between full refreshes computations.
     *                 Smaller values (10-20) provide higher numerical stability.
     *                 Larger values (50-100) reduce computational cost for stability.
     *
     * @note Must be called before executeStep(); Typical range: 10 - 50.
     */
    void setKahanRefreshInterval(std::size_t interval);

    /**
     * @brief Get current Kahan summation refresh interval.
     */
    std::size_t getKahanRefreshInterval() const noexcept { return kahan_refresh_interval_; }

    /**
     * @brief Get total number of variable removals (negative actions).
     * @return Monotonic count of all drops throughout the solution path
     *         (always 0 for non-cycling solvers).
     */
    std::size_t getNumRemovals() const noexcept { return num_removals_; }

    /**
     * @brief Get cycling ratio (removals / additions).
     * @return Ratio, or 0.0 if no additions yet.
     */
    double getCyclingRatio() const {
        if (num_additions_ == 0) { return 0.0; }
        return static_cast<double>(num_removals_) /
               static_cast<double>(num_additions_);
    }

    /** @brief TSolver_Base::memoryBytes() plus the lambda sequence (reserved for
     *  maxSteps + 1 entries), the Kahan compensation and the sign and drop records. */
    std::size_t memoryBytes() const noexcept override;

    // ============================================================================
    // De-/Serialization
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal model and path state except for input matrices.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        // Enum passes through a local int (reinterpret_cast aliasing is UB);
        // the assignment after archive() applies the loaded value and is a
        // no-op round-trip on save.
        int algo_type = static_cast<int>(algo_type_);
        archive(
            cereal::base_class<TSolver_Base>(this),
            cereal::make_nvp("algo_type", algo_type),
            CEREAL_NVP(lambda_),
            CEREAL_NVP(correlation_compensation_),
            CEREAL_NVP(A_A_),
            CEREAL_NVP(Sign_),
            CEREAL_NVP(kahan_refresh_interval_),
            CEREAL_NVP(num_removals_),
            CEREAL_NVP(drop_recovery_pending_),
            CEREAL_NVP(last_dropped_vars_)
        );
        algo_type_ = static_cast<SolverTypeLarsBased>(algo_type);
    }

    /**
     * @brief Save the current TLARS solver/model state to file (binary serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Path to output file.
     *
     * @note Numerical quantities are saved up to a precision of 1e-12.
     */
    virtual void save(const std::string& filename) const;

    /**
     * @brief Load TLARS solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TLARS_Solver object connected to X and D.
     */
    static TLARS_Solver load(const std::string& filename,
                             Eigen::Map<Eigen::MatrixXd>& X,
                             Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ============================================================================
    // Internal LARS math helper
    // ============================================================================

    /**
     * @brief Update correlations incrementally with periodic refresh.
     *
     * @details Uses Kahan's algorithm for incremental updates c_j = c_j - gamma * a_j,
     *          maintaining numerical stability. Full recomputation from residuals every
     *          20th step prevents long-term error drift accumulation.
     *
     * @param gamma Step size taken along equiangular direction.
     * @param a Projection vector where a[i] = X^T_{inactives[i]} * u from computeStepSize().
     *
     * @note Uses parallelized inner products for speed (OpenMP if available).
     */
    virtual void updateCorrelations(double gamma, const Eigen::Ref<const Eigen::VectorXd>& a);

    /**
     * @brief Attempt to update the active set by adding new variables and performing Cholesky
     * updates.
     *
     * @details For each candidate, tries Cholesky rank-1 update and accepts
     *          on rank increase, or flags column as dropped otherwise.
     *          Updates actives, dropped indices, and sign tracking.
     *          In non-cycling algorithms only additions are performed.
     *          Descendants with cycling may override to implement drops or
     *          cycling logic.
     *
     * @param new_vars Candidate indices for addition.
     *
     * @return Vector of actions for this step: positive index = accepted,
     *         negative = dropped.
     */
    std::vector<int> updateActiveSet(const std::vector<std::size_t>& new_vars);

    /**
     * @brief Update and record the solution path coefficients after each
     *        algorithmic step.
     *
     * @param w_a Current equiangular direction weights (for actives).
     * @param gamma Step size to move along direction.
     */
    void updateBetaPath(const Eigen::Ref<const Eigen::VectorXd>& w_A, double gamma);

    /**
     * @brief Construct sign vector (+1/-1) for current active set correlations.
     *
     * @return Eigen vector of size |active set| with entries +1/-1.
     */
    Eigen::VectorXd signVector() const;

    /**
     * @brief Compute T-LARS/Stagewise equiangular direction for current
     *        actives/signs.
     *
     * @param sign Sign vector over actives (+1/-1).
     *
     * @return Direction vector in active set space.
     */
    Eigen::VectorXd equiangularDirection(const Eigen::Ref<const Eigen::VectorXd>& sign);

    /**
     * @brief Compute the equiangular predictor vector for current actives.
     *
     * @param w_A Direction weights for actives.
     *
     * @return Combined predictor vector u.
     */
    virtual Eigen::VectorXd equiangularVector(const Eigen::Ref<const Eigen::VectorXd>& w_A) const;

    /**
     * @brief Compute maximal feasible step size along equiangular direction.
     *
     * @param Cmax Current maximal correlation (step size scaling).
     * @param u Equiangular predictor combination.
     *
     * @return Pair of {gamma, projection_vector} where projection[i] = X^T_{inactives} u.
     *         Maximum safe increment (gamma), constrained to no over-shooting/ties.
     */
    virtual std::pair<double, Eigen::VectorXd> computeStepSize(
        double Cmax,
        const Eigen::Ref<const Eigen::VectorXd>& u) const;

    // ============================================================================
    // Shared LASSO-style drop machinery (used by cycling descendants:
    // T-LASSO, T-ENET; T-Stagewise reuses the correlation-refresh hook)
    // ============================================================================

    /**
     * @brief Compute the minimum positive step size to a coefficient
     *        zero-crossing (LASSO modification of the LARS step).
     *
     * @param gamhat Step size from the LARS entry logic.
     * @param drops  Output: marks active variables crossing zero at the
     *               returned step size (size = |actives_|).
     * @param w_A    Equiangular direction weights for the active set.
     *
     * @return Smallest positive crossing step size, or gamhat if no
     *         coefficient crosses zero before the next variable entry.
     */
    double computeGammaSignChange(double gamhat,
                                  std::vector<bool>& drops,
                                  const Eigen::Ref<const Eigen::VectorXd>& w_A) const;

    /**
     * @brief Remove zero-crossing variables from the active set.
     *
     * @details Processes drops in reverse index order: downdates the Cholesky
     * factor, zeroes the coefficient at the current step, records a negative
     * action, re-adds the variable to inactives_ (cycling), refreshes its
     * correlation via refreshDroppedCorrelation(), and updates the dummy and
     * removal counters.
     *
     * @param drops Boolean vector marking which active variables to drop.
     */
    void processLassoDrops(std::vector<bool>& drops);

    /**
     * @brief Refresh correlations_ for a variable removed from the active set
     * so it can compete for re-entry (its entry is stale from activation time).
     *
     * @details Default: plain correlation with the current residual plus a
     * Kahan-compensation reset. Penalized descendants (T-ENET) override with
     * their augmented formula.
     */
    /**
     * @brief Cap a recovery-step gamma at the earliest GENUINE catch-up of
     * the excluded (just-dropped) variables. They cannot be ADDED during
     * the recovery step, but they still BOUND it: on near-tie data a
     * dropped variable can genuinely overtake the decaying active level
     * mid-step, and overshooting past that point makes the next entry
     * lambda jump upward (non-monotone path). Roots below
     * floor_rel * (Cmax / A_A) are the scale-invariant fp-noise band of
     * the just-dropped variable's gamma = 0 crossing and are ignored.
     *
     * @param gamma Step size from the (excluded-free) entry computation.
     * @param Cmax Current active correlation level.
     * @param ac Per-excluded-variable (a_v, c_v) pairs in the solver's
     *           modified-correlation convention.
     * @param floor_rel Relative root floor (default 1e-6).
     */
    double capGammaByExcluded(
        double gamma, double Cmax,
        const std::vector<std::pair<double, double>>& ac,
        double floor_rel = 1e-6) const;

    virtual void refreshDroppedCorrelation(std::size_t dropped_var);
};

// ===================================================================================

} /* End of namespace trex::tsolvers::ad::linear_model::lars_based */


#endif /* TSOLVERS_LINEAR_MODEL_LARS_BASED_TLARS_SOLVER_HPP */
