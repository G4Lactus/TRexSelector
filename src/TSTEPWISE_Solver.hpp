#ifndef TSTEPWISE_SOLVER
#define TSTEPWISE_SOLVER

#include <chrono>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "TLARS_Solver.hpp"

/**
 * @brief T-Stepwise Regression Solver (Forward Selection with Dummy Variables)
 *
 * @details Specializes the TLARS_Solver for classical stepwise (greedy)
 *          regression with dummy variable augmentation to support FDR-controlled
 *          variable selection.
 *
 * Algorithm:
 * Given current residual r and active set A, at step k:
 * 1. Find j* = argmax_j |X_j^T r| (over inactive predictors)
 * 2. Add j* to active set: A ← A ∪ {j*}
 * 3. Compute equiangular direction w_A
 * 4. Compute equiangular vector u = X_A w_A
 * 5. Take full orthogonal step: γ = C_max / A_A
 *    where C_max = max|X_j^T r| and A_A = 1/√(s^T G_A^{-1} s)
 * 6. Update: β ← β + γ w_A, r ← r - γ u
 * 7. Result: All active variables have zero correlation with new residual
 *
 * Key Difference from T-LARS:
 *  - Step size: Full orthogonal step (γ = C_max / A_A) vs minimum step
 *  - Correlation: Exactly zero after each step vs small positive
 *  - Complexity: Simple (no step size search, no cycling, no drops)
 *  - Optimality: Greedy selection not LARS path
 *
 * Numerical Stability:
 * - Inherits Cholesky updates from TLARS_Solver (collinearity detection)
 * - Full correlation recomputation each step (no incremental drift)
 * - Orthogonal steps prevent accumulaiton of numerical errors
 *
 * @note Inherits all T-LARS capabilities: MMAP support, serialization,
 *       warm-starts, openMP support.
 * @note Designed for use with augmented design matrix X = [X_original | X_dummies]
 */
class TSTEPWISE_Solver: public TLARS_Solver {

public:

    // ==========================================================================
    // Constructors/Destructor
    // ==========================================================================

    /**
     * @brief Construct a new TSTEPWISE_Solver object
     *
     * @param X Augmented design matrix [X_original | X_dummies]
     *          (RAM/MMAP, column-major, not owned)
     * @param y Response vector
     * @param num_dummies Number of dummy variables appended to X
     * @param normalize If true, each column of X is scaled to unit L2-norm
     * @param intercept If true, X and y are centered (intercept fitted)
     * @param verbose If true, print status and diagnostics during execution
     */
    TSTEPWISE_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                     Eigen::Map<Eigen::VectorXd>& y,
                     std::size_t num_dummies,
                     bool normalize = true,
                     bool intercept = true,
                     bool verbose = false)
                : TLARS_Solver(X, y, num_dummies, normalize, intercept, verbose,
                    SolverType::TStepwise) {}

    /** @brief Default constructor for serialization or deferred initialization.
     */
    TSTEPWISE_Solver()
        : TLARS_Solver(SolverType::TStepwise) {}

    /**
     * @brief Delete copy constructor to prevent raw pointer ownership problems.
     */
    TSTEPWISE_Solver(const TSTEPWISE_Solver&) = delete;

    /**
     * @brief Delete assignment operator to avoid raw pointer ownership problems.
     */
    TSTEPWISE_Solver& operator=(const TSTEPWISE_Solver&) = delete;

    /** @brief Move constructor. */
    TSTEPWISE_Solver(TSTEPWISE_Solver&&) = default;

    /** @brief Destructor. */
    ~TSTEPWISE_Solver() override = default;


    // ==========================================================================
    // Core Execution Step T-Stepwise
    // ==========================================================================

    /**
     * @brief Execute T-STEPWISE solution path (greedy forward selection).
     *
     * @param T_stop Number of dummies to trigger early stopping
     *               (0: full solution path, default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @details
     * Overrides T-LARS path algorithm with full orthogonal steps:
     * - Selects variable with maximum absolute correlation
     * - Computes equiangular direction (same as T-LARS)
     * - Takes full step: γ = C_max / A_A (reduces correlation to zero)
     * - Fully recomputes all correlations from residual (no incremental updates)
     * - Resets sign vector for next iteration
     * - No drops, no cycling, pure forward selection
     *
     * Stopping Conditions:
     * - Active set becomes empty (shouldn't happen in forward selection)
     * - Maximum correlation < 100·eps (numerical convergence)
     * - Early stopping: count_active_dummies_ >= T_stop (after step completes)
     * - Maximum steps reached (maxSteps_)
     *
     * @note Internal state is updated in-place; supports warm-starts and
     *       serialization.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;


    // ==========================================================================
    // Serialization
    // ==========================================================================

    /**
     * @brief Serialize all internal model and path state except for X_.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     *
     * @note X_ must be reconnected after deserialization.
     * @note Inherits TLARS_Solver serialization.
     */
    template<class Archive>
    void serialize(Archive& archive) {
            TLARS_Solver::serialize(archive);
    }

    /**
     * @brief Load TSTEPWISE_Solver from file, automatically restoring all
     *  internal state except X_ (pointer).
     *
     * @param filename File path to load from.
     * @param X Reference to augmented design matrix to restore pointer
     *          connection.
     *
     * @return Deserialized TSTEPWISE_Solver, connected to X.
     *
     * @throws std::runtime_error On IO or deserialization failures.
     */
    static TSTEPWISE_Solver load(const std::string& filename,
                                 Eigen::Map<Eigen::MatrixXd>& X);

protected:

    // ==========================================================================
    // T-Stepwise Internals
    // ==========================================================================

    /**
     * @brief Reset sign vector to enable orthogonal (axis-aligned) steps.
     *
     * @details Sets all elements of Sign_ to zero, to ensure that each newly
     *          entering variable starts with a fresh correlation sign, based on
     *          its current correlation with the residual.
     *
     * Necessary for stepwise regression because:
     * - We take full orthogonal steps (correlation -> 0)
     * - Signs from previous iterations are no longer valid
     * - Next iteration needs fresh sign determination
     *
     * @note Called after each step, before correlation recomputation.
     */
    void zeroAllSigns();


    /**
     * @brief Fully recompute correlations for all inactive variables.
     */
    void recomputeCorrelations();

};

#endif /* End of TSTEPWISE_SOLVER */
