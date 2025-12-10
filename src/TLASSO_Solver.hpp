#ifndef TLASSO_SOLVER_HPP
#define TLASSO_SOLVER_HPP

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include <cereal/archives/portable_binary.hpp>

#include "TLARS_Solver.hpp"

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
     *          High values relative to additions indicate strong cycling
     *          behavior.
     */
    std::size_t num_removals_{0};


protected:
    /**
     * @brief Protected constructor for inheritance by derived classes.
     *
     * @param solver_type Algorithm variant as SolverType enum
     *                   (e.g., SolverType::TLASSO).
     *
     * @note Used by further derived classes that extend T-LASSO behavior.
     */
    explicit TLASSO_Solver(SolverType solver_type) :
                            TLARS_Solver(solver_type) {}


public:
    // ==========================================================================
    // Constructors/Destructors
    // ==========================================================================

    /**
     * @brief Construct a new T-LASSO solver with data and configuration.
     *
     * @param X Augmented design matrix [X_original | X_dummies]
     *          (RAM/MMAP, column-major, not owned,
     *          must persist for solver lifetime).
     * @param y Response vector (size n).
     * @param num_dummies Number of dummy variables appended at end of X.
     * @param normalize If true, each column of X is scaled to unit L2-norm
     *                  before fitting.
     * @param intercept If true, X and y are centered (intercept fitted and
     *                  removed).
     * @param verbose If true, print detailed status and diagnostics during
     *                execution.
     *
     * @note X is not copied nor owned by solver - changes to X affect
     *       underlying data.
     * @note Has internal column-dropping logic for low-variance columns
     *       when normalize=true.
     *       Otherwise ensure collinear columns are excluded a priori.
     * @note Initializes base T-LARS state with SolverType::TLASSO.
     */
    TLASSO_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::VectorXd>& y,
                  std::size_t num_dummies,
                  bool normalize = true,
                  bool intercept = true,
                  bool verbose = false)
        : TLARS_Solver(X, y, num_dummies, normalize, intercept, verbose,
                       SolverType::TLASSO) {}

    /**
     * @brief Default constructor for serialization or deferred initialization.
     *
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TLASSO_Solver() : TLARS_Solver(SolverType::TLASSO) {}

    /**
     * @brief Deleted copy constructor to avoid raw pointer ownership problems.
     */
    TLASSO_Solver(const TLASSO_Solver&) = delete;

    /**
     * @brief Deleted copy assignment operator to avoid raw pointer
     *        ownership problems.
     */
    TLASSO_Solver& operator=(const TLASSO_Solver&) = delete;

    /**
     * @brief Move constructor (default implementation).
     */
    TLASSO_Solver(TLASSO_Solver&&) = default;

    /**
     * @brief Virtual destructor (X_ pointer is not owned, so not deleted).
     *
     * @note Present for completeness and to support inheritance.
     */
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
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Overrides base T-LARS executeStep() with variable removal logic.
     *
     * @details
     *   Extends T-LARS with sign-change monitoring:
     *   - At each step, computes gamma_sign (distance to zero-crossing)
     *   - Takes gamma = min(gamma_lars, gamma_sign)
     *   - If gamma = gamma_sign: removes zero-crossing variables
     *   - Downdates Cholesky factor using Givens rotations
     *   - Repeats until no more crossings in current step
     *
     *   Stopping Conditions:
     *   - Active set becomes empty
     *   - Maximum correlation < 100·eps (numerical convergence)
     *   - Early stopping: count_active_dummies_ >= T_stop
     *   - Maximum steps reached
     *
     * @note Internal state is updated in-place; supports warm-starts and
     *       serialization.
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
     *     * Strong collinearity in design matrix
     *     * Numerical instability in Cholesky updates/downdates
     *     * Regularization parameter may need tuning
     *
     * @note High cycling ratios (>0.5) may suggest numerical issues or
     *       ill-conditioning.
     */
    double getCyclingRatio() const;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    /**
     * @brief Serialize all internal T-LASSO solver state except for X_.
     *
     * @tparam Archive Cereal archive type (e.g.,
     *          cereal::PortableBinaryOutputArchive).
     * @param archive Output/input archive object.
     *
     * @details Saves base T-LARS state plus T-LASSO-specific
     *          num_removals_ counter. Call reconnect(X) after loading to
     *          restore design matrix pointer.
     *
     * @note X_ must be reconnected after deserialization.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        // Serialize base class
        TLARS_Solver::serialize(archive);

        // Serialize T-LASSO-specific state
        CEREAL_NVP(num_removals_);
    }

    /**
     * @brief Load T-LASSO solver state from file (binary serialization).
     *
     * @param filename Input file path for checkpoint.
     * @param X Reference to augmented design matrix to restore pointer
     *          connection.
     *
     * @return Deserialized TLASSO_Solver object connected to X and ready for
     *         warm-start.
     *
     * @note Static method usage:
     *      `auto tlasso = TLASSO_Solver::load("checkpoint.bin", X);`
     * @note X dimensions must match original design matrix used during save().
     *
     * @throws std::runtime_error on file I/O errors or dimension mismatch.
     */
    static TLASSO_Solver load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X);

protected:

    // ============================================================================
    // T-LASSO Internals
    // ============================================================================

    /**
     * @brief Compute minimum positive step size (gamma) to coefficient
     *        zero-crossing.
     *
     * @param gamhat Step size proposal from T-LARS logic (gamma to next
     *               variable entry).
     * @param drops Output boolean vector marking which actives cross zero
     *              at gamma_sign.
     * @param w_A Current equiangular direction weights for active set.
     *
     * @return Smallest positive gamma where any coefficient crosses zero,
     *         or gamhat if none.
     *
     * @details
     *   For each active variable j:
     *   - Current coefficient: beta_j
     *   - Direction: d_j (from w_A)
     *   - Zero-crossing when: beta_j + gamma * d_j = 0
     *   - Solve: gamma = -beta_j / d_j (if positive and sign change occurs)
     *
     *   Returns minimum over all positive gamma values, and marks corresponding
     *   drops[j] = true for variables that cross zero at this gamma.
     *
     * @note Only considers positive step sizes; negative crossings are ignored.
     */
    double computeGammaSignChange(double gamhat,
                                  std::vector<bool>& drops,
                                  const Eigen::VectorXd& w_A) const;

    /**
     * @brief Remove all zero-crossing variables from active set and downdate
     *        Cholesky factor.
     *
     * @param drops Boolean vector marking which active variables to drop
     *              (size = |actives_|).
     *
     * @details
     *   For each marked variable:
     *   1. Downdate Cholesky factor R using Givens rotations
     *   2. Remove variable from actives_ vector
     *   3. Add to inactives_ (allows re-entry in cycling)
     *   4. Record negative action in actions_.back()
     *   5. Increment num_removals_ counter
     *   6. Track dummy removals if applicable
     *
     * @note Processes drops in reverse order to maintain index stability
     *       during removal.
     * @note Updates Sign_ vector to match reduced active set.
     */
    void processLassoDrops(std::vector<bool>& drops);


};


#endif /* End of TLASSO_Solver */
