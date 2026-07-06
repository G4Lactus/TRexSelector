// ==================================================================================
// tgp_solver.hpp
// ==================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_OMP_BASED_TGP_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_OMP_BASED_TGP_SOLVER_HPP
// ==================================================================================
/**
 * @file tgp_solver.hpp
 *
 * @brief Header file for the Terminating Gradient Pursuit (T-GP) solver.
 *
 * @details The TGP_Solver class implements the T-GP algorithm for variable selection
 * in linear regression, with support for dummy variable driven early stopping, as
 * necesary for the TRex framework and its FDR control.
 * It inherits state management, standardization, and diagnostics from the TOMP_Solver
 * class.
 */
// ===================================================================================

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================
/**
* @class TGP_Solver
*
* @brief Terminating Gradient Pursuit (T-GP) solver with dummy variables for FDR control.
*
* @details
* Implements T-GP based on Blumensath & Davies (2008): "Gradient Pursuits" (Section III-B):
* - Gradient-based directional updates instead of orthogonal projections
* - Computational complexity: O(np + nk) per iteration
*   - Dominated by O(np) for correlation updates (X^T r)
*   - Avoids the O(nk + k^3) Cholesky solve required by OMP
* - All T-OMP infrastructure usable: dummy variables, early stopping, path tracking
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
*
* @note T-GP sets the direction to the active correlations (Blumensath & Davies, Eq. 7):
*
*         d^n = c^n_A
*
*       followed by the optimal step size (Eq. 8):
*
*         alpha^n = <r^{n-1}, p^n> / ||p^n||^2,   p^n = X_A d^n
*
* @note Unlike OMP, GP may select the same atom multiple times if the residual is not
*       sufficiently orthogonal to previously selected atoms. This is by design and
*       allows GP to auto-calibrate how many directional updates it needs per atom.
*
* @note The Cholesky factorization inherited from TOMP_Solver is intentionally unused.
*       Collinearity is handled implicitly: collinear variables produce ||p^n||^2 ≈ 0,
*       which drives alpha^n ≈ 0 and terminates progress on that variable.
*
* @note Reference: Blumensath, T. and Davies, M.E. (2008). Gradient Pursuits.
*       IEEE Transactions on Signal Processing, 56(6), pp.2370-2382.
*/
class TGP_Solver : public TOMP_Solver {

protected:

    // ==================================================================================
    // TGP specific state variables
    // ==================================================================================

    /**
     * @brief Update direction for active variables d^{n} = c^{n}_A.
     * Size matches actives_.size().
     */
    Eigen::VectorXd direction_{};

    /**
     * @brief Projected direction p^{n} = X_A * d^{n}.
     * Used for step size computation and residual update.
     */
    Eigen::VectorXd projected_direction_{};

    /**
     * @brief Step size alpha^{n} for current directional update.
     *   Computed as alpha^{n} = <r^{n-1}, p^{n}> / ||p^{n}||^2.
     */
    double step_size_{0.0};

    /**
     * @brief Current coefficient vector for active variables only.
     * Updated incrementally: beta^{n}_A = beta^{n-1}_A + alpha^{n} * d^{n}.
     */
    Eigen::VectorXd active_coefficients_{};

    // ==================================================================================
    // Constructor protected for inheritance
    // ==================================================================================

    /** @brief Protected constructor for inheritance.
     *
     *  @param solver_type Type of the solver.
     */
    explicit TGP_Solver(SolverTypeOMPBased solver_type);

public:

    // ==================================================================================
    // Constructor/Destructor
    // ==================================================================================

    /**
     * @brief Construct a new TGP_Solver object using data and configuration.
     *
     * @param X Map to the predictor matrix (n x p).
     * @param D Map to the dummy predictor matrix (n x num_dummies).
     * @param y Map to the response vector (n x 1).
     * @param normalize If true, each column of X and D is scaled to unit L2-norm.
     * @param intercept If true, X, D, and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     */
    TGP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
               Eigen::Map<Eigen::MatrixXd>& D,
               Eigen::Map<Eigen::VectorXd>& y,
               bool normalize = true,
               bool intercept = true,
               bool verbose = false,
               ScalingMode scaling_mode = ScalingMode::L2)
        : TOMP_Solver(X, D, y, normalize, intercept, verbose,
            SolverTypeOMPBased::TGP, scaling_mode) {}

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TGP_Solver() : TOMP_Solver(SolverTypeOMPBased::TGP) {}

    /** @brief Deleted copy constructor to avoid raw pointer ownership problems. */
    TGP_Solver(const TGP_Solver&) = delete;

    /** @brief Deleted copy assignment operator to avoid raw pointer ownership problems. */
    TGP_Solver& operator=(const TGP_Solver&) = delete;

    /** @brief Move constructor (default implementation). */
    TGP_Solver(TGP_Solver&&) = default;

    /** @brief Deleted move assignment operator to avoid raw pointer ownership problems. */
    TGP_Solver& operator=(TGP_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TGP_Solver() override = default;


    // ==================================================================================
    // Main algorithm and Overrides
    // ==================================================================================

    /**
     * @brief Compute T-GP solution path with gradient-based directional updates and
     *        early stopping based on dummy threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold (0 = full solution path,
     *               default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Overrides base class T-OMP with T-GP from Blumensath & Davies (2008).
     *       Uses gradient direction d^n = c^n_A instead of orthogonal projections.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ==================================================================================
    // De-/Serialization
    // ==================================================================================

    /** @brief Friend class for Cereal serialization. */
    friend class cereal::access;

    /**
     * @brief Serialize TGP-specific state plus base class state.
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the
     * static type of *this at the call site — not the runtime type.
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
            cereal::base_class<TOMP_Solver>(this),
            CEREAL_NVP(direction_),
            CEREAL_NVP(projected_direction_),
            CEREAL_NVP(step_size_),
            CEREAL_NVP(active_coefficients_)
        );
    }

    /**
     * @brief Save the current TGP solver/model state to the given file (binary serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Output file path.
     *
     * @note Numerical quantities are saved up to a precision of 1e-12.
     */
    virtual void save(const std::string& filename) const override;

    /**
     * @brief Load a TGP solver state from file (binary serialization)
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TGP_Solver object connected to X.
     */
    static TGP_Solver load(const std::string& filename,
                           Eigen::Map<Eigen::MatrixXd>& X,
                           Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ==================================================================================
    // GP-Specific Algorithm Components
    // ==================================================================================

    /**
     * @brief Compute gradient direction for active variables: d^{n} = c^{n}_A.
     *
     * @note Overriden by descendant TACGP_Solver to add conjugate correction.
     */
    virtual void computeGradientDirection();

    /**
     * @brief Project direction into data space: c^{n} = X_A * d^{n}.
     *
     * @note Used for step size computation and residual updates.
     */
    void projectDirection();

    /**
     * @brief Compute step size alpha^{n} for current directional update.
     *
     * @details Here: alpha^{n} = <r^{n-1}, c^{n}> / ||c^{n}||^2.
     * Returns 0 if ||p^{n}||^2 < eps^2 (implicit collinerity check).
     *
     * @return Step size alpha^{n} for directional update.
     */
    double computeStepSize();

    /**
     * @brief Update active coefficients: beta_A^n = beta_A^{n-1} + alpha^n * d^n.
     */
    void updateActiveCoefficients();

    // ==================================================================================
    // Overrides of Base Class Methods
    // ==================================================================================

    /**
     * @brief Find maximum |correlation| among ALL non-dropped variables.
     *
     * @return Pair of (max_correlation, vector of tied indices at maximum).
     *
     * @note GP override: searches all non-dropped variables to allow re-selection.
     *       Base class only searches inactives.
     */
    std::pair<double, std::vector<std::size_t>> findTiedMaxCorrelations() const override;

    /**
     * @brief Update the active set by adding new variables.
     *
     * @details For GP, distinguishes between:
     *          - New atoms: Add directly to active set (no Cholesky check).
     *          - Re-selected atoms: Record action only (coefficients updated via gradient step).
     *
     *          Supports batch additions when multiple atoms are tied at maximum
     *          correlation |c^{n}|.
     *
     * @note No Cholesky factorization is performed. Collinearity is handled
     *       implicitly via ||p^n||^2 ≈ 0 in computeStepSize().
     *
     * @param new_vars Candidate indices for addition (may contain multiple tied atoms).
     *
     * @return Vector of actions for this step: positive index = accepted (no drops in pure GP).
     */
    std::vector<int> updateActiveSet(const std::vector<std::size_t>& new_vars) override;

    /**
     * @brief Update residual using GP directional update:
     *        r^{n} = r^{n-1} - alpha_n * p^{n}.
     *
     * @note GP override: uses incremental directional update instead of full recomputation.
     */
    void updateResiduals() override;

    /**
     * @brief Update betaPath_ by copying active_coefficients_ to appropriate positions.
     *
     * @note GP override: avoids Cholesky solve, directly uses active_coefficients_.
     */
    void updateBetaPath() override;

    /**
     * @brief Update correlations c^{n} = X^{T} r^{n} for ALL non-dropped variables.
     *
     * @note GP override: computes for all variables (not just inactives) to allow re-selection.
     *       Base class TOMP_Solver only updates inactives.
     */
    void updateCorrelations() override;

};

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */

#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TGP_SOLVER_HPP */
