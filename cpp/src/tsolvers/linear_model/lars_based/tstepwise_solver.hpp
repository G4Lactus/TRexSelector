// =============================================================================
// tstepwise_solver.hpp
// =============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTEPWISE_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTEPWISE_SOLVER_HPP
// =============================================================================
/**
 * @file tstepwise_solver.hpp
 *
 * @brief T-STEPWISE Regression Solver (Forward Selection with Dummy Variables)
 */
// =============================================================================

// std includes
#include <string>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>

// =============================================================================

// Embedded in namespace trex::tsolvers::linear_model::lars_based
namespace trex::tsolvers::linear_model::lars_based {

// =============================================================================

/**
 * @brief T-Stepwise Regression Solver (Forward Selection with Dummy Variables)
 *
 * @details Specializes the TLARS_Solver for classical stepwise (greedy)
 * regression with dummy variable augmentation to support FDR-controlled
 * variable selection.
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
 * Numerical Stability:
 * - Inherits Cholesky updates from TLARS_Solver (collinearity detection)
 * - Full correlation recomputation each step (no incremental drift)
 * - Orthogonal steps prevent accumulation of numerical errors
 */
class TSTEPWISE_Solver: public TLARS_Solver {

protected:
    /**
    * @brief Protected constructor for inheritance by derived classes.
    */
    explicit TSTEPWISE_Solver(SolverTypeLarsBased solver_type) : TLARS_Solver(solver_type) {}

public:

    // ==========================================================================
    // Constructors/Destructor
    // ==========================================================================

    /**
     * @brief Construct a new TSTEPWISE_Solver object
     *
     * @param X Predictor matrix (n x p).
     * @param D Dummy matrix (n x num_dummies).
     * @param y Response vector (size n).
     * @param normalize If true, columns are scaled to unit L2-norm.
     * @param intercept If true, variables are centered.
     * @param verbose If true, print detailed status and diagnostics.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     */
    TSTEPWISE_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                     Eigen::Map<Eigen::MatrixXd>& D,
                     Eigen::Map<Eigen::VectorXd>& y,
                     bool normalize = true,
                     bool intercept = true,
                     bool verbose = false,
                     ScalingMode scaling_mode = ScalingMode::L2)
        : TLARS_Solver(X, D, y, normalize, intercept, verbose,
                       SolverTypeLarsBased::TSTEPWISE, scaling_mode) {}

    /** @brief Default constructor for serialization or deferred initialization. */
    TSTEPWISE_Solver() : TLARS_Solver(SolverTypeLarsBased::TSTEPWISE) {}

    /** @brief Deleted copy constructor. */
    TSTEPWISE_Solver(const TSTEPWISE_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TSTEPWISE_Solver& operator=(const TSTEPWISE_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TSTEPWISE_Solver(TSTEPWISE_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TSTEPWISE_Solver& operator=(TSTEPWISE_Solver&&) = delete;

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
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop. Default: true.
     *
     * @details
     * Overrides T-LARS path algorithm with full orthogonal steps:
     * - Selects variable with maximum absolute correlation
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

    friend class cereal::access;

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
        archive(
            cereal::base_class<TLARS_Solver>(this)
        );
    }

    /**
     * @brief Save T-Stepwise solver/model state to file (binary serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Path to output file.
     *
     * @note Numerical quantities are saved up to a precision of 1e-12.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load TSTEPWISE_Solver from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     */
    static TSTEPWISE_Solver load(const std::string& filename,
                                 Eigen::Map<Eigen::MatrixXd>& X,
                                 Eigen::Map<Eigen::MatrixXd>& D);

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

// =============================================================================
} /* End of namespace trex::tsolvers::linear_model::lars_based */
#endif /* TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTEPWISE_SOLVER_HPP */
