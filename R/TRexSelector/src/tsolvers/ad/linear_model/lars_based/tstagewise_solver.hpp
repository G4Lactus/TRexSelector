// ============================================================================
// tstagewise_solver.hpp
// ============================================================================
/**
 * @file tstagewise_solver.hpp
 *
 * @brief Declaration of TSTAGEWISE_Solver, a T-LARS-based stagewise regression
 * solver.
 */
// ============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTAGEWISE_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTAGEWISE_SOLVER_HPP
// ============================================================================

// std includes
#include <string>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>

// tsolvers includes
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::ad::linear_model::lars_based
namespace trex::tsolvers::ad::linear_model::lars_based {

// ============================================================================

/**
 * @brief T-Stagewise solver via LARS-Stagewise algorithm.
 *
 * @details Implements the Incremental Forward Stagewise Regression path.
 * Enforces monotonicity, i.e., coefficients must evolve in the direction of their
 * correlation signs (w_j * Sign_j > 0).
 * If this is violated, a non-negative least squares (NNLS) reduction is applied
 * to the active set.
 *
 * @note A variable the NNLS reduction removes from the active set keeps its
 * coefficient (frozen), so the support of the coefficient path can exceed the
 * active set. The selection is read from the coefficient path, as the T-Rex
 * selector and cdiagnostics::selected_variables() do; getActives() lists the
 * active set only. The dummy count that stops the path at T_stop follows the
 * support as well: a dummy removed with a frozen nonzero coefficient keeps
 * counting, and a frozen dummy that re-enters is not counted twice, so the
 * path stops at the first step whose support holds T_stop dummies, the step
 * the T-Rex runner maps to t = T_stop. A variable removed within a step is no candidate for that
 * step's move and rejoins with an exact correlation after it (the
 * drop-breakpoint rule of the lasso-family solvers); its correlation then lies
 * strictly below the active level, so the move is never a rounding-noise step.
 * At saturation the path still cycles through removals and re-entries up to
 * the step limit.
 *
 *  References:
 *  -----------
 *   B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
 *     no. 2, pp. 407-499, 2004.
 *   T. Hastie, J. Taylor, R. Tibshirani, and G. Walther, "Forward Stagewise Regression and the Monotone
 *     Lasso," Electron. J. Stat., vol. 1, pp. 1-29, 2007, doi: 10.1214/07-EJS004.
 *   R. J. Tibshirani, "A General Framework for Fast Stagewise Algorithms," J. Mach. Learn. Res., vol. 16,
 *     pp. 2543-2588, 2015.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TSTAGEWISE_Solver : public TLARS_Solver {

protected:
    // ========================================================================
    // STAGEWISE specific states
    // ========================================================================

    /** @brief Variables the NNLS reduction removed in the current step
     *  (transient, empty between steps, not serialized). */
    std::vector<std::size_t> nnls_dropped_step_;

public:
    // ========================================================================
    // Constructor/Destructor
    // ========================================================================

    /**
     * @brief Construct a new tstagewise solver object
     *
     * @param X Original design matrix (n x p).
     * @param D Dummy design matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param normalize If true, each column of X is scaled to unit L2-norm.
     * @param intercept If true, X and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     */
    TSTAGEWISE_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                      Eigen::Map<Eigen::MatrixXd>& D,
                      Eigen::Map<Eigen::VectorXd>& y,
                      const SolverOptions& options = SolverOptions{});

    /** @brief Default constructor for serialization or deferred initialization. */
    TSTAGEWISE_Solver() : TLARS_Solver(SolverTypeLarsBased::TSTAGEWISE) {}

    /** @brief Deleted copy constructor and assignment. */
    TSTAGEWISE_Solver(const TSTAGEWISE_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TSTAGEWISE_Solver& operator=(const TSTAGEWISE_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TSTAGEWISE_Solver(TSTAGEWISE_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TSTAGEWISE_Solver& operator=(TSTAGEWISE_Solver&&) = delete;

    /** @brief Virtual destructor override. */
    ~TSTAGEWISE_Solver() override = default;

    // ========================================================================
    // Main Algorithm (Overrides Pure Virtual)
    // ========================================================================

    /**
     * @brief Compute T-Stagewise solution path with early stopping based on dummy
     * threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold,
     * (0 = full solution path, default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts and serialization.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ========================================================================
    // Specific Getters and Setters for T-Stagewise
    // ========================================================================

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal model and path state except for input matrices.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param ar Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TLARS_Solver>(this)
        );
    }

    /**
     * @brief Save the current TSTAGEWISE solver/model state to the given file (binary
     * serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Path to output file.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load TSTAGEWISE state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TSTAGEWISE_Solver object connected to X and D.
     */
    static TSTAGEWISE_Solver load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X,
                                  Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ========================================================================
    // Internal Stagewise math helpers
    // ========================================================================

    /**
     * @brief Computes the equiangular direction enforcing the stagewise monotonicity constraint.
     *
     * @details Iteratively checks if the unscaled LARS direction is currently aligned with
     * the signs of current correlations. If a violation occurs, it triggers processStagewiseNNLS()
     * to drop the violating variable and re-solves until a strictly positive active subset is
     * found.
     *
     * @return Valid equiangular direction vector w_A, or an empty vector if the active set
     * becomes empty.
     */
    Eigen::VectorXd computeStagewiseDirection();


    /**
     * @brief Performs Non-Negative Least Squares reduction on active set.
     *
     * @details Corresponds to the inner feasibility loop of Lawson-Hanson NNLS.
     * Drops the most violating constraint and downdates the Cholesky factor.
     *
     * @param directions Vector of current step directions for active coefficients.
     */
    void processStagewiseNNLS(const Eigen::Ref<const Eigen::VectorXd>& directions);
};

// ============================================================================
} /* End of namespace trex::tsolvers::ad::linear_model::lars_based  */

#endif /* TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTAGEWISE_SOLVER_HPP */
