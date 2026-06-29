// ==================================================================================
// tmp_solver.hpp
// ==================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_OMP_BASED_TMP_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_OMP_BASED_TMP_SOLVER_HPP
// ==================================================================================
/**
* @file tmp_solver.hpp
*
* @brief Header file for the Terminating Matching Pursuit (T-MP) solver.
*/
// ==================================================================================

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>

// ==================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================
/**
* @class TMP_Solver
*
* @brief Terminating Matching Pursuit (T-MP) solver with dummy variables for FDR control.
*
* @details
* Implements T-MP based on Blumensath & Davies (2008): "Gradient Pursuits" (Section III-A):
* - Selects a single-atom update per iteration
* - Lowest computational complexity: O(np) per iteration
*   - Dominated by the full correlation update X^T r
*   - No matrix-vector products beyond a single column subtraction
*
* Notation convention used throughout this class (differs from paper):
*   y        response vector                       (their: x)
*   X        design matrix (n x p)                 (their: Phi)
*   beta_A   active coefficients                   (their: y_{Gamma^n})
*   r        residual = y - X_A beta_A             (their: r^n)
*   c^n      correlations = X^T r                  (their: g^n)
*   alpha^n  step size = c^n_{i^n}                 (their: a^n)
*   i^n      selected atom index                   (their: lambda^n)
*
* @note T-MP selects the single most correlated atom and updates only its coefficient
*       (Blumensath & Davies, Eq. 3):
*
*         alpha^n = c^n_{i^n} = <r^{n-1}, x_{i^n}>
*
*       For normalized columns ||x_{i^n}||^2 = 1 this is exact. The residual
*       update follows directly (Eq. 5):
*
*         r^n = r^{n-1} - alpha^n * x_{i^n}
*
* @note Unlike OMP, MP may re-select the same atom multiple times. This is by
*       algorithmic design: each step greedily reduces the projection of the
*       residual onto the most correlated direction.
*
* @note The Cholesky factorization inherited from TOMP_Solver is intentionally
*       unused. MP requires no matrix factorization — collinearity is handled
*       implicitly via |alpha^n| < eps_ in the outer loop.
*
* @note Reference: Blumensath, T. and Davies, M.E. (2008). Gradient Pursuits.
*       IEEE Transactions on Signal Processing, 56(6), pp.2370-2382.
*/
class TMP_Solver : public TOMP_Solver {

protected:

    // ==================================================================================
    // TMP specific state variables
    // ==================================================================================

    /**
    * @brief Step size alpha^n = c^{n}_{i^{n}} = <r^{n-1}, x_{i^{n}}> .
    *   Equals the correlation of the selected atom for normalized columns.
    */
    double step_size_{0.0};

    /**
    * @brief Coefficient vector for active variables only.
    *   Updated incrementally: beta_A^{n}[i^{n}] = beta_A^{n-1}[i^{n}] + alpha^{n}.
    *   Size grows by one when a new atom is selected (re-selected atoms
    *   do not extend the vector).
    */
    Eigen::VectorXd active_coefficients_{};

    /** @brief Algorithm variant type. */
    SolverTypeOMPBased algo_type_{SolverTypeOMPBased::TMP};

    // ==================================================================================
    // Constructor protected for inheritance
    // ==================================================================================

    /**
    * @brief Protected constructor for inheritance.
    *
    * @param solver_type Type of the solver.
    */
    explicit TMP_Solver(SolverTypeOMPBased solver_type);

public:

    // ==================================================================================
    // Constructor/Destructor
    // ==================================================================================

    /**
    * @brief Construct a TMP_Solver object using data and configuration.
    *
    * @param X Map to the predictor matrix (n x p).
    * @param D Map to the dummy predictor matrix (n x num_dummies).
    * @param y Map to the response vector (n x 1).
    * @param normalize If true, each column of X and D is scaled to unit L2-norm.
    * @param intercept If true, X, D, and y are centered (intercept fitted).
    * @param verbose If true, print status and diagnostics during execution.
    */
    TMP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
            Eigen::Map<Eigen::MatrixXd>& D,
            Eigen::Map<Eigen::VectorXd>& y,
            bool normalize = true,
            bool intercept = true,
            bool verbose = false,
            ScalingMode scaling_mode = ScalingMode::L2)
        : TOMP_Solver(X, D, y, normalize, intercept, verbose,
                    SolverTypeOMPBased::TMP, scaling_mode) {}

    /**
    * @brief Default constructor for serialization or deferred initialization.
    */
    TMP_Solver() : TOMP_Solver(SolverTypeOMPBased::TMP) {}

    /** @brief Deleted copy constructor to avoid raw pointer ownership problems. */
    TMP_Solver(const TMP_Solver&) = delete;

    /** @brief Deleted copy assignment operator to avoid raw pointer ownership problems. */
    TMP_Solver& operator=(const TMP_Solver&) = delete;

    /** @brief Move constructor (default implementation). */
    TMP_Solver(TMP_Solver&&) = default;

    /** @brief Deleted move assignment operator to avoid raw pointer ownership problems. */
    TMP_Solver& operator=(TMP_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TMP_Solver() override = default;

    // ==================================================================================
    // Main algorithm
    // ==================================================================================

    /**
    * @brief Compute T-MP solution path with single-atom updates and early stopping
    *   based on dummy threshold.
    *
    * @param T_stop    Number of dummies for early stopping threshold
    *                  (0 = full solution path).
    * @param early_stop  If true, stop when count_active_dummies_ >= T_stop.
    *
    * @note Overrides TOMP_Solver::executeStep() with T-MP from Blumensath & Davies (2008).
    */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ==================================================================================
    // De-/Serialization
    // ==================================================================================

    /** @brief Friend class for Cereal serialization. */
    friend class cereal::access;

    /**
    * @brief Serialize TMP-specific state plus base class state.
    *
    * @tparam Archive  Cereal archive type.
    * @param archive   Output/input archive object.
    *
    * @note X_ and D_ must be reconnected via reconnect() after deserialization.
    */
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TOMP_Solver>(this),
            CEREAL_NVP(step_size_),
            CEREAL_NVP(active_coefficients_)
        );
    }

    /**
    * @brief Save the current T-MP solver/model state to the given file
    *   (binary serialization).
    *
    * @details Must be class-specific because Cereal does not use virtual dispatch.
    *
    * @param filename  Output file path.
    *
    * @note Numerical quantities are saved up to a precision of 1e-12.
    */
    void save(const std::string& filename) const override;

    /**
    * @brief Load a T-MP solver state from file (binary serialization).
    *
    * @param filename  Path to input file.
    * @param X         Map to feature matrix.
    * @param D         Map to dummy matrix.
    *
    * @return Deserialized TMP_Solver object reconnected to X and D.
    */
    static TMP_Solver load(const std::string& filename,
                            Eigen::Map<Eigen::MatrixXd>& X,
                            Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ==================================================================================
    // MP-Specific Algorithm Components
    // ==================================================================================

    /**
    * @brief Find maximum |correlation| among ALL non-dropped variables.
    *
    * @return Pair of (max_correlation, vector of tied indices at maximum).
    *
    * @note MP override: searches all non-dropped variables to allow re-selection.
    */
    std::pair<double, std::vector<std::size_t>> findTiedMaxCorrelations() const override;

    /**
    * @brief Update the active set by adding the selected atom.
    *
    * @details Distinguishes between:
    *   - NEW atom: appended to actives_, coefficient slot initialized to 0.
    *   - RE-SELECTED atom: action recorded only; coefficient incremented via
    *     incrementCoefficient().
    *
    * @param new_vars  Candidate indices (typically one atom; ties possible).
    *
    * @return Vector of actions for this step (positive index = accepted).
    */
    std::vector<int> updateActiveSet(const std::vector<std::size_t>& new_vars) override;

    /**
    * @brief Increment coefficient of selected atom: beta_A^{n}[i^{n}] += alpha^{n}.
    *
    * @param atom_idx  Position of selected atom within actives_.
    */
    void incrementCoefficient(std::size_t atom_idx);

    /**
    * @brief Update residual: r^{n} = r^{n-1} - alpha^{n} * x_{i^{n}}.
    *
    * @param col_idx  Column index i^{n} of the selected atom in X.
    */
    void subtractAtom(std::size_t col_idx);

    /**
    * @brief Update betaPath_ by copying active_coefficients_ to appropriate positions.
    */
    void updateBetaPath() override;

    /**
    * @brief Update correlations c^{n} = X^T r for ALL non-dropped variables.
    *
    * @note MP override: updates all variables (not just inactives) to allow re-selection.
    */
    void updateCorrelations() override;

};

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */

#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TMP_SOLVER_HPP */
