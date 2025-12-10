#ifndef TACGP_SOLVER_HPP
#define TACGP_SOLVER_HPP

#include "TGP_Solver.hpp"

/**
 * @class TACGP_Solver
 *
 * @brief Terminating Approximate Conjugate Gradient Pursuit (T-ACGP) solver.
 *
 * @details
 * Implements T-ACGP based on Blumensath & Davies (2008): "Gradient Pursuits" with:
 * - Approximate conjugate gradient direction instead of pure gradient
 * - Better approximation of OMP than GP
 * - Computational complexity: O(np + nk) per iteration
 *    - Dominated by O(np) for correlation updates (X^T r)
 *    - Same complexity as GP (no additional overhead from conjugate correction)
 * - All T-OMP infrastructure: dummy variables, early stopping, path tracking
 *
 * @note T-ACGP extends T-GP by computing an approximate conjugate direction:
 *       d^n = g^n_active + beta1 * d^{n-1}
 *       where beta1 = <X^T X d^{n-1}, g^n> / ||c^{n-1}||^2
 *
 * @note This conjugate correction significantly improves convergence over pure GP
 *       while maintaining the same O(np + nk) computational complexity.
 *
 * @note Reference: Blumensath, T. and Davies, M.E. (2008). Gradient Pursuits.
 *       IEEE Transactions on Signal Processing, 56(6), pp.2370-2382.
 */
class TACGP_Solver : public TGP_Solver {

protected:
    // ==================================================================================
    // Data Members
    // ==================================================================================

    /**
     * @brief Previous direction vector d^{n-1} for conjugate gradient computation.
     */
    Eigen::VectorXd direction_prev_{};

    /**
     * @brief Previous projected direction c^{n-1} = X_active * d^{n-1}.
     *        Stored to avoid recomputation in beta1 calculation.
     */
    Eigen::VectorXd direction_projected_prev_{};

    /**
     * @brief Conjugate gradient correction term beta1.
     *        Computed as: beta1 = <X^T X d^{n-1}, g^n> / ||c^{n-1}||^2
     */
    double beta1_{0.0};

    /**
     * @brief Protected constructor for potential derived classes.
     *
     * @param type Solver type enum (to be set by derived class).
     */
    explicit TACGP_Solver(SolverType type);

public:
    // ==================================================================================
    // Constructor/Destructor
    // ==================================================================================

    /**
     * @brief Construct a TACGP_Solver object using data and configuration.
     *
     * @param X Augmented design matrix [X_original | X_dummies] (RAM/MMAP, column-major,
     *          not owned, must persist for solver lifetime).
     * @param y Response vector.
     * @param num_dummies Number of dummy columns at the end of X.
     * @param normalize If true, each column of X is scaled to unit L2-norm.
     * @param intercept If true, X and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     */
    TACGP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::VectorXd>& y,
                 std::size_t num_dummies,
                 bool normalize = true,
                 bool intercept = true,
                 bool verbose = false);

    /**
     * @brief Default constructor for serialization or deferred initialization.
     *
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TACGP_Solver();

    /**
     * @brief Virtual destructor.
     */
    virtual ~TACGP_Solver() = default;

    /** @brief Deleted copy constructor to avoid raw pointer ownership problems. */
    TACGP_Solver(const TACGP_Solver&) = delete;

    /** @brief Deleted copy assignment operator to avoid raw pointer ownership problems. */
    TACGP_Solver& operator=(const TACGP_Solver&) = delete;

    /** @brief Move constructor (default implementation). */
    TACGP_Solver(TACGP_Solver&&) = default;

    // ==================================================================================
    // Main Algorithm (Override)
    // ==================================================================================

    /**
     * @brief Execute T-ACGP solution path with approximate conjugate gradient updates
     *        and early stopping based on dummy threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold (0 = full solution path).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop;
     *                   if false, run full path ignoring T_stop.
     *
     * @note Overrides TGP_Solver::executeStep() to store previous direction
     *       after each iteration for conjugate gradient computation.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ==================================================================================
    // De-/Serialization
    // ==================================================================================

    /**
     * @brief Serialize T-ACGP-specific state plus base class state.
     *
     * @tparam Archive Cereal archive type.
     * @param archive Output/input archive object.
     *
     * @note X_ must be reconnected after deserialization.
     */
    template <class Archive>
    void serialize(Archive& archive) {
        // Serialize base class (TGP_Solver, which serializes TOMP_Solver)
        TGP_Solver::serialize(archive);

        // Serialize TACGP-specific members
        archive(
            CEREAL_NVP(direction_prev_),
            CEREAL_NVP(direction_projected_prev_),
            CEREAL_NVP(beta1_)
        );
    }

    /**
     * @brief Save the current solver/model state to the given file (binary serialization).
     *
     * @param filename Output file path.
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load solver state from file (binary serialization).
     *
     * @param filename Input file path.
     * @param X Reference to augmented design matrix to restore pointer connection.
     *
     * @return Deserialized TACGP_Solver object connected to X.
     */
    static TACGP_Solver load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X);

protected:
    // ==================================================================================
    // T-ACGP-Specific Methods and Overrides
    // ==================================================================================

    /**
     * @brief Compute approximate conjugate gradient direction.
     *
     * @details T-ACGP direction formula:
     *          d^n = g^n_active + beta1 * d^{n-1}
     *
     *          where beta1 = <X^T X d^{n-1}, g^n> / ||c^{n-1}||^2
     *
     *          This provides better approximation to OMP than pure GP
     *          while maintaining O(np + nk) computational complexity.
     *
     * @note On first iteration (no previous direction), falls back to pure gradient.
     * @note Overrides TGP_Solver::computeGradientDirection().
     */
    void computeGradientDirection() override;

    /**
     * @brief Compute conjugate correction term beta1.
     *
     * @details Computes: beta1 = <X^T X d^{n-1}, g^n> / ||c^{n-1}||^2
     *
     *          This is efficiently computed as:
     *          1. Compute X d^{n-1} (already available as direction_projected_prev_)
     *          2. Compute X^T (X d^{n-1}) by restricted dot products with active columns
     *          3. Compute <result, g^n> / ||c^{n-1}||^2
     *
     * @return Conjugate correction coefficient beta1.
     */
    double computeBeta1();

    /**
     * @brief Store current direction and projected direction for next iteration.
     *
     * @note Called at the end of each iteration after projectDirection()
     *       to prepare for next conjugate gradient computation.
     */
    void storePreviousDirection();
};

#endif /* End of TACGP_SOLVER_HPP */
