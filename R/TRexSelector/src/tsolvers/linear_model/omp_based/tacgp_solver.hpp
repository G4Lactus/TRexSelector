// =================================================================================
// tacgp_solver.hpp
// =================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_OMP_BASED_TACGP_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_OMP_BASED_TACGP_SOLVER_HPP
// =================================================================================
/**
 * @file tacgp_solver.hpp
 *
 * @brief Header file for the Terminating Approximate Conjugate Gradient Pursuit
 * (T-ACGP) solver.
 */
// =================================================================================

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tgp_solver.hpp>

// ==================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================

/**
 * @class TACGP_Solver
 *
 * @brief Terminating Approximate Conjugate Gradient Pursuit (T-ACGP) solver.
 *
 * @details
 * Implements T-ACGP based on Blumensath & Davies (2008): "Gradient Pursuits" (Section III-C):
 * - Approximate conjugate gradient direction instead of pure gradient
 * - Better approximation of OMP than GP
 * - Computational complexity: O(np + nk) per iteration
 *   - Dominated by O(np) for correlation updates (X^T r)
 *   - Same complexity as GP (no additional overhead from conjugate correction)
 * - All T-OMP infrastructure: dummy variables, early stopping, path tracking
 *
 * Notation convention used throughout this class (differs from paper):
 *   y        response vector                       (their: x)
 *   X        design matrix (n x p)                 (their: Phi)
 *   X_A      active columns of X                   (their: Phi_{Gamma^n})
 *   beta_A   active coefficients                   (their: y_{Gamma^n})
 *   r        residual = y - X_A beta_A             (their: r^n)
 *   c^n_A    active correlations = X_A^T r         (their: g^n_{Gamma^n})
 *   d^n      update direction (k x 1)              (their: d^n_{Gamma^n})
 *   p^n      projected direction = X_A d^n         (their: c^n)
 *   alpha^n  step size                             (their: a^n)
 *   G        Gram matrix = X_A^T X_A               (their: G_{Gamma^n})
 *   b_1      conjugate correction coefficient      (their: b_1, Eq. 26)
 *
 * @note T-ACGP extends T-GP by computing an approximate conjugate direction
 *       (Blumensath & Davies, Eq. 24, with b_0 = 1):
 *
 *         d^n = c^n_A + b_1 * d^{n-1}
 *
 *       where b_1 carries a built-in negative sign (Eq. 26):
 *
 *         b_1 = - <p^{n-1}, X_A c^n_A> / ||p^{n-1}||^2
 *
 *       p^{n-1} = X_A d^{n-1} is already available from the previous
 *       iteration (stored as projected_direction_prev_), so the numerator
 *       requires one additional matrix-vector product X_A c^n_A of cost O(nk).
 *       On the first iteration d^{n-1} is zero, so b_1 = 0 and the method
 *       falls back to pure GP.
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
     * @brief Previous conjugate direction d^{n-1} for conjugate gradient computation.
     * Size matches actives_.size() at the previous iteration.
     */
    Eigen::VectorXd direction_prev_{};

    /**
     * @brief Previous projected direction p^{n-1} = X_A d^{n-1}.
     * Stored after projectDirection() in each iteration to avoid recomputation of
     * the denominator ||p^{n-1}||^2 in b_1.
     */
    Eigen::VectorXd projected_direction_prev_{};

    /**
     * @brief Conjugate gradient correction coefficient b_1.
     *   Computed as: b_1 = -<p^{n-1}, X_A c^n_A> / ||p^{n-1}||^2.
     *   Zero on the first iteration or after an active set size change.
     */
    double b_1_{0.0};

    /**
     * @brief Protected constructor for potential derived classes.
     *
     * @param solver_type Type of the solver.
     */
    explicit TACGP_Solver(SolverTypeOMPBased solver_type);

public:
    // ==================================================================================
    // Constructor/Destructor
    // ==================================================================================

    /**
     * @brief Construct a TACGP_Solver object using data and configuration.
     *
     * @param X Design matrix (n x p).
     * @param D Dummy matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param normalize If true, each column of X and D is scaled to unit L2-norm.
     * @param intercept If true, X, D, and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     */
    TACGP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y,
                 bool normalize = true,
                 bool intercept = true,
                 bool verbose = false)
        : TGP_Solver(X, D, y, normalize, intercept, verbose) {
            algo_type_ = SolverTypeOMPBased::TACGP;
          }

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TACGP_Solver() : TGP_Solver(SolverTypeOMPBased::TACGP) {}

    /** @brief Deleted copy constructor to avoid raw pointer ownership problems. */
    TACGP_Solver(const TACGP_Solver&) = delete;

    /** @brief Deleted copy assignment operator to avoid raw pointer ownership problems. */
    TACGP_Solver& operator=(const TACGP_Solver&) = delete;

    /** @brief Move constructor (default implementation). */
    TACGP_Solver(TACGP_Solver&&) = default;

    /** @brief Deleted move assignment operator to avoid raw pointer ownership problems. */
    TACGP_Solver& operator=(TACGP_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TACGP_Solver() override = default;


    // ==================================================================================
    // Main Algorithm (Override)
    // ==================================================================================

    /**
     * @brief Execute T-ACGP solution path with approximate conjugate gradient updates
     *        and early stopping based on dummy threshold.
     *
     * @param T_stop Number of dummies for early stopping (0 = full path).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note storePreviousDirection() is placed after projectDirection() and before
     *       computeStepSize(). This guarantees p^n is fully formed before saving,
     *       and d^{n-1} / p^{n-1} are available for the next iteration's
     *       computeGradientDirection() without any dependency hazard.
     *
     * @note Overrides TGP_Solver::executeStep().
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;


    // ==================================================================================
    // De-/Serialization
    // ==================================================================================

    /** @brief Friend class for Cereal serialization. */
    friend class cereal::access;

    /**
     * @brief Serialize T-ACGP-specific state plus base class state.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     *
     * @note X_ and D_ must be reconnected via reconnect() after deserialization.
     */
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TGP_Solver>(this),
            CEREAL_NVP(direction_prev_),
            CEREAL_NVP(projected_direction_prev_),
            CEREAL_NVP(b_1_)
        );
    }

    /**
     * @brief Save the current TACGP solver/model state to file (binary serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Output file path.
     *
     * @note Numerical quantities are saved up to a precision of 1e-12.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load a TACGP solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TACGP_Solver object connected to X and D.
     */
    static TACGP_Solver load(const std::string& filename,
                            Eigen::Map<Eigen::MatrixXd>& X,
                            Eigen::Map<Eigen::MatrixXd>& D);
protected:
    // ==================================================================================
    // T-ACGP-Specific Methods and Overrides
    // ==================================================================================

    /**
     * @brief Compute approximate conjugate gradient direction.
     *
     * @details T-ACGP direction formula (Eq. 24, Blumensath & Davies 2008):
     *   d^n = c^n_A + b_1 * d^{n-1}
     *   where b_1 = -<p^{n-1}, X_A c^n_A> / ||p^{n-1}||^2
     *
     *   Falls back to pure gradient (d^n = c^n_A, b_1 = 0) on the first
     *   iteration or when the active set size changed since the last iteration.
     *
     * @note Overrides TGP_Solver::computeGradientDirection().
     */
    void computeGradientDirection() override;

    /**
     * @brief Compute conjugate correction coefficient b_1 (Eq. 26).
     *
     * @details Computes: b_1 = -<p^{n-1}, X_A c^n_A> / ||p^{n-1}||^2
     *
     *   Efficiently computed as:
     *   1. X_A^T p^{n-1}   (dot products against stored projected_direction_prev_)
     *   2. Numerator       = <c^n_A, X_A^T p^{n-1}>
     *   3. Denominator     = ||p^{n-1}||^2  (squaredNorm of projected_direction_prev_)
     *
     * @return Conjugate correction coefficient b_1. Returns 0.0 if
     *   ||p^{n-1}||^2 < eps_^2 (numerical fallback to pure gradient).
     */
    double computeB_1();

    /**
     * @brief Store current direction and projected direction for next iteration.
     *
     * @details Sets:
     *   direction_prev_           <- direction_            (d^n  -> d^{n-1})
     *   projected_direction_prev_ <- projected_direction_  (p^n  -> p^{n-1})
     *
     * @note Must be called after projectDirection() and before computeStepSize()
     *   so that p^n is fully formed before saving.
     */
    void storePreviousDirection();
};

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */

#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TACGP_SOLVER_HPP */
