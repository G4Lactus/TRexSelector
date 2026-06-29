// =============================================================================
//  TLASSO_Solver.hpp
// =============================================================================
#ifndef TSOLVERS_TLASSO_SOLVER_HPP
#define TSOLVERS_TLASSO_SOLVER_HPP
// =============================================================================
/**
 * @file tlasso_solver.hpp
 *
 * @brief Header file for the Terminating LASSO (T-LASSO) solver class,
 * extending T-LARS solver.
 */
// =============================================================================

// std includes
#include <string>
#include <vector>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>

// =============================================================================

// Embed into trex::tsolvers::linear_model::lars_based namespace
namespace trex::tsolvers::linear_model::lars_based {

// =============================================================================

/**
 * @brief Terminating LASSO (T-LASSO) solver with dummy variable support for
 *        FDR control.
 *
 * @details
 *  Implements T-LASSO regression by extending T-LARS with variable removal
 *  logic:
 *  - Monitors coefficient sign changes during T-LARS steps
 *  - Removes variables when coefficients cross zero (L1 penalty enforcement)
 *  - Performs Cholesky downdate using Givens rotations for numerical stability
 *  - Supports variable cycling (add → remove → re-add sequences)
 *  - Tracks dummy variables for FDR-controlled variable selection
 *  - Supports early stopping when T_stop dummies enter active set
 *  - Produces exact T-LASSO solution path equivalent to standard LASSO
 *
 *   Algorithm Overview:
 *   1. Perform standard T-LARS step (add variable with max absolute
 *       correlation)
 *   2. Compute two step sizes:
 *      - gamma_lars: Standard T-LARS step size to next variable entry
 *      - gamma_sign: Distance to first coefficient zero-crossing
 *   3. Take minimum gamma = min(gamma_lars, gamma_sign)
 *   4. If gamma = gamma_sign:
 *      - Remove all zero-crossing variables from active set
 *      - Downdate Cholesky factor R using Givens rotations
 *      - Record removal actions (negative indices)
 *   5. Track dummy variable entries for early stopping
 *   6. Repeat until stopping criteria met
 *
 *   Stopping Criteria:
 *   - Active set becomes empty
 *   - Maximum correlation falls below 100·eps
 *   - Early stopping: count_active_dummies >= T_stop
 *   - Maximum steps reached (maxSteps_)
 *
 *   Cycling Behavior:
 *   High cycling ratios (num_removals / num_additions > 0.5) may indicate:
 *   - Strong collinearity in design matrix
 *   - Numerical instability in updates/downdates
 *   - Need for regularization parameter tuning
 */
class TLASSO_Solver: public TLARS_Solver {

protected:

    // ==========================================================================
    // T-LASSO specific state variables
    // ==========================================================================

    /**
     * @brief Monotonic counter of variable removals (never decremented).
     *
     * @details Tracks total number of negative actions (variable drops)
     *          throughout solution path.
     *          Forms basis for cycling ratio = num_removals / num_additions.
     */
    std::size_t num_removals_{0};

    // ==================================================================================
    // Constructor protected for inheritance
    // ==================================================================================
    /**
     * @brief Protected constructor for inheritance by derived classes.
     *
     * @param solver_type Algorithm variant as SolverTypeLarsBased enum
     *                   (e.g., SolverTypeLarsBased::TLASSO).
     */
    explicit TLASSO_Solver(SolverTypeLarsBased solver_type) :
                            TLARS_Solver(solver_type) {}

public:
    // ==========================================================================
    // Constructors/Destructors
    // ==========================================================================

    /**
     * @brief Construct a new T-LASSO solver with data and configuration.
     *
     * @param X Predictor matrix (n x p).
     * @param D Dummy matrix (n x num_dummies).
     * @param y Response vector (size n).
     * @param normalize If true, columns are scaled to unit L2-norm. Default true.
     * @param intercept If true, variables are centered. Default true.
     * @param verbose If true, print detailed status and diagnostics. Default false.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     */
    TLASSO_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  bool normalize = true,
                  bool intercept = true,
                  bool verbose = false,
                  ScalingMode scaling_mode = ScalingMode::L2)
        : TLARS_Solver(X, D, y, normalize, intercept, verbose,
                       SolverTypeLarsBased::TLASSO, scaling_mode) {}

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TLASSO_Solver() : TLARS_Solver(SolverTypeLarsBased::TLASSO) {}

    /**
     * @brief Deleted copy constructor to avoid raw pointer ownership problems.
     */
    TLASSO_Solver(const TLASSO_Solver&) = delete;

    /**
     * @brief Deleted copy assignment operator to avoid raw pointer ownership problems.
     */
    TLASSO_Solver& operator=(const TLASSO_Solver&) = delete;

    /**
     * @brief Move constructor (defaulted).
     */
    TLASSO_Solver(TLASSO_Solver&&) = default;

    /**
     * @brief Deleted move assignment operator to avoid raw pointer ownership problems.
     */
    TLASSO_Solver& operator=(TLASSO_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TLASSO_Solver() override = default;


    // ==========================================================================
    // Core Algorithm Execution
    // ==========================================================================

    /**
     * @brief Execute T-LASSO solution path with zero-crossing detection
     *        and dummy tracking.
     *
     * @param T_stop Number of dummies for early stopping threshold
     *               (0 = full solution path, default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop. Default true.
     *
     * @note Overrides base T-LARS executeStep() with variable removal logic.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;


    // ==========================================================================
    // Output & Accessors
    // ==========================================================================

    /**
     * @brief Get total number of variable removals (negative actions).
     *
     * @return Monotonic count of all variable drops throughout solution path.
     *
     * @note Never decremented; forms numerator of cycling ratio.
     */
    std::size_t getNumRemovals() const noexcept { return num_removals_; }

    /**
     * @brief Get the cycling ratio (removals / additions).
     *
     * @return Ratio of removals to additions, or 0.0 if no additions yet.
     *
     * @details
     *   The cycling ratio = num_removals / num_additions indicates solution
     *   path complexity.
     *
     *   Interpretation:
     *   - ratio < 0.3: Minimal cycling, stable path
     *   - ratio 0.3-0.5: Moderate cycling, typical for correlated predictors
     *   - ratio > 0.5: Heavy cycling, potential issues:
     *                  * Strong collinearity in design matrix
     *                  * Numerical instability in Cholesky updates/downdates
     *                  * Regularization parameter may need tuning
     *
     * @note High cycling ratios (>0.5) may suggest numerical issues or
     *       ill-conditioning.
     * @note Well-conditioned problems should have low cycling ratios (<0.2).
     */
    double getCyclingRatio() const;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-LASSO solver state except for input matrices.
     *
     * @tparam Archive Cereal archive type (e.g., cereal::PortableBinaryOutputArchive).
     *
     * @param archive  Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TLARS_Solver>(this),
            CEREAL_NVP(num_removals_)
        );
    }

    /**
     * @brief Save T-LASSO solver/model state to file (binary serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Output file path for checkpoint.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load T-LASSO solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TLASSO_Solver object connected to X and ready for
     *         warm-start.
     */
    static TLASSO_Solver load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ============================================================================
    // T-LASSO Internals
    // ============================================================================

    /**
     * @brief Compute minimum positive step size (gamma) to coefficient
     *        zero-crossing.
     *
     * @param gamhat Step size from T-LARS logic (first element of pair
     *               returned by computeStepSize(), representing gamma to
     *               next variable entry).
     * @param drops  Output boolean vector marking which active variables
     *               cross zero at gamma_sign (size = |actives_|).
     * @param w_A    Current equiangular direction weights for active set.
     *
     * @return Smallest positive gamma where any coefficient crosses zero,
     *         or gamhat if no crossings occur before next variable entry.
     *
     * @details
     * For each active variable j with index i in active set:
     * - Current coefficient: beta_j = betaPath_(actives_[i], currentStep_ - 1)
     * - Direction: d_j = w_A(i)
     * - Zero-crossing when: beta_j + gamma * d_j = 0
     * - Solve: gamma = -beta_j / d_j (if positive and sign change occurs)
     *
     * Returns minimum over all positive gamma values, and marks corresponding
     * drops[i] = true for variables that cross zero at this gamma.
     *
     * Multiple simultaneous crossings (within eps tolerance) are handled
     * by marking all crossing variables in drops vector.
     *
     * @note Only considers positive step sizes; negative crossings ignored.
     * @note Uses eps_ tolerance for detecting simultaneous crossings.
     */
    double computeGammaSignChange(double gamhat,
                                  std::vector<bool>& drops,
                                  const Eigen::Ref<const Eigen::VectorXd>& w_A) const;

    /**
     * @brief Remove all zero-crossing variables from active set and downdate
     *        Cholesky factor (R).
     *
     * @param drops Boolean vector marking which active variables to drop
     *              (size = |actives_|, true = drop this variable).
     *
     * @details
     * For each marked variable (processed in reverse order for index stability):
     * 1. Downdate Cholesky factor R using Givens rotations (downdateR())
     * 2. Zero out coefficient in betaPath_ at current step
     * 3. Record negative action in actions_.back() (removal indicator)
     * 4. Re-add variable to inactives_ (enables cycling/re-entry)
     * 5. Remove variable from actives_ vector (erase)
     * 6. Remove sign from Sign_ vector (erase)
     * 7. Decrement count_active_dummies_ if dropping a dummy
     * 8. Increment num_removals_ counter
     *
     * @note Processes drops in reverse order (high to low index) to maintain
     *       index stability during vector erasure operations.
     * @note Re-adding to inactives_ is CRITICAL for LASSO cycling behavior.
     * @note Updates Sign_ vector to match reduced active set.
     * @note Thread-safe for single-threaded execution (modifies class state).
     */
    void processLassoDrops(std::vector<bool>& drops);
};

// =============================================================================
} // End of namespace trex::tsolvers::linear_model::lars_based

#endif /* TSOLVERS_TLASSO_SOLVER_HPP */
