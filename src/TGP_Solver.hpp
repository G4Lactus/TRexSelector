#ifndef TGP_SOLVER_HPP
#define TGP_SOLVER_HPP

#include "TOMP_Solver.hpp"


/**
 * @brief Terminating Gradient Pursuit (T-GP) solver with dummy variables for FDR control.
 *
 * @details
 *  Implements T-GP based on Blumensath & Davies (2008): "Gradient Pursuits" with:
 *  - Gradient-based directional updates instead of orthogonal projections
 *  - Computational complexity: O(np + nk) per iteration (as MP)
 *  - Approximates OMP performance with lower computational cost
 *  - All T-OMP infrastructure usable: dummy variables, early stopping, path tracking
 *
 * @note GP uses gradient direction for updates, avoiding expensive Cholesky updates required
 *       for full orthogonal projection.
 *
 * @note Unlike OMP, GP may select the same atom multiple times if residual is not sufficiently
 *       orthogonal to previously selected atoms. This is by algorithmic design and allows it to
 *       auto-calibrate how many directional updates it needs per atom.
 *
 * @note Reference: Blumensath, T. and Davies, M.E. (2008). Gradient Pursuits.
 *       IEEE Transactions on Signal Processing, 56(6), pp.2370-2382.
 */
class TGP_Solver : public TOMP_Solver {

protected:

    // ==================================================================================
    // Data Members
    // ==================================================================================

    /**
     * @brief Current gradient direction for active variables
     *        (d_n in Blumensath & Davies 2008).
     *        Size matches actives_.size().
     */
    Eigen::VectorXd direction_{};


    /**
     * @brief Direction projected into data space: c_n = X_active * d_n.
     *        Used for step size computation and residual updates.
     */
    Eigen::VectorXd direction_projected_{};

    /**
     * @brief Step size for current directional update (alpha_n in Blumensath & Davies 2008).
     */
    double step_size_{0.0};

    /**
     * @brief Current coefficient vector for active variables only.
     *        Updated incrementally: y_n = y_{n-1} + alpha_n * d_n.
     */
    Eigen::VectorXd active_coefficients_{};

    /**
     * @brief Protected default constructor for derived classes.
     */
    explicit TGP_Solver(SolverType type);

    /**
     * @brief Protected constructor with full parameter for derived classes.
     *
     * @param X Augmented design matrix [X_original | X_dummies] (RAM/MMAP, column-major,
     *          not owned, must persist for solver lifetime).
     * @param y Response vector.
     * @param num_dummies Number of dummy columns at the end of X.
     * @param normalize If true, each column of X is scaled to unit L2-norm.
     * @param intercept If true, X and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     */
    TGP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
               Eigen::Map<Eigen::VectorXd>& y,
               std::size_t num_dummies,
               bool normalize,
               bool intercept,
               bool verbose,
               SolverType type);

public:

    // ==================================================================================
    // Constructor/Destructor
    // ==================================================================================

    /**
     * @brief Construct a TGP_Solver object using data and configuration.
     *
     * @param X Augmented design matrix [X_original | X_dummies] (RAM/MMAP, column-major,
     *          not owned, must persist for solver lifetime).
     * @param y Response vector.
     * @param num_dummies Number of dummy columns at the end of X.
     * @param normalize If true, each column of X is scaled to unit L2-norm.
     * @param intercept If true, X and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     */
    TGP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
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
    TGP_Solver();

    /**
     * @brief Virtual destructor (X_ pointer is not owned, so not deleted).
     *
     * @note Present for completeness and to support inheritance.
     */
    virtual ~TGP_Solver() = default;

    /** @brief Deleted copy constructor to avoid raw pointer ownership problems. */
    TGP_Solver(const TGP_Solver&) = delete;

    /** @brief Deleted copy assignment operator to avoid raw pointer ownership problems. */
    TGP_Solver& operator=(const TGP_Solver&) = delete;

    /** @brief Move constructor (default implementation). */
    TGP_Solver(TGP_Solver&&) = default;

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
     *       Uses gradient direction instead of orthogonal projections.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ==================================================================================
    // De-/Serialization
    // ==================================================================================

    /**
     * @brief Serialize TGP-specific state plus base class state.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     *
     * @note X_ must be reconnected after deserialization.
     */
    template <class Archive>
    void serialize(Archive& archive) {
            // Serialize base class
            TOMP_Solver::serialize(archive);

            // Serialize TGP-specific members
            archive(
                CEREAL_NVP(direction_),
                CEREAL_NVP(direction_projected_),
                CEREAL_NVP(step_size_),
                CEREAL_NVP(active_coefficients_)
            );
    }

    /**
     * @brief Save the current solver/model state to the given file (binary serialization).
     *
     * @param filename Output file path.
     *
     * @note Numerical quantities are saved up to a precision of 1e-12.
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load solver state from file (binary serialization)
     *
     * @param filename Input file path.
     * @param X Reference to augmented design matrix to restore pointer connection.
     *
     * @return Deserialized TGP_Solver object connected to X.
     */
    static TGP_Solver load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X);

protected:

    // ==================================================================================
    // GP-Specific Algorithm Components
    // ==================================================================================

     /**
      * @brief Compute gradient direction for active variables: d^n = g^n_active.
      *
      */
     virtual void computeGradientDirection();

     /**
      * @brief Project direction into data space: c^n = X_active * d^n.
      *
      * @note Used for step size computation and residual updates.
      */
     void projectDirection();

    /**
     * @brief Compute step size alpha_n for current directional update.
     *        Here: alpha^n = <r^T_{n-1} c^n> / ||c^n||^2
     *
     * @return Step size alpha_n for directional update.
     */
    double computeStepSize();

    /**
     * @brief Update active coefficients using directional update:
     *        y^{{n}_active} = y^{{n-1}_active} + alpha^n * d^n.
     */
    void updateActiveCoefficients();

    // ==================================================================================
    // Overrides of Base Class Methods
    // ==================================================================================

    /**
     * @brief Find maximum |correlation| among ALL variables (not just inactives).
     *
     * @return Pair of (max_correlation, vector of tied indices at maximum).
     *
     * @note GP override: searches all non-dropped variables to allow re-selection.
     *       Base class only searches inactives.
     */
    std::pair<double, std::vector<std::size_t>> findTiedMaxCorrelations() const override;

    /**
     * @brief Attempt to update the active set by adding new variables.
     *
     * @details For GP, distinguishes between:
     *          - NEW atoms: Add directly to active set (no Cholesky check)
     *          - RE-SELECTED atoms: Record action only (coefficients updated via gradient step)
     *
     *          Supports batch additions when multiple atoms are tied at maximum correlation.
     *
     *          Unlike OMP, GP does not use Cholesky factorization for collinearity checking.
     *          Instead, collinearity is naturally handled by the step size computation:
     *          collinear variables result in ||c_n||^2 ≈ 0, leading to step_size ≈ 0,
     *          which effectively prevents progress on that variable.
     *
     * @param new_vars Candidate indices for addition (may contain multiple tied atoms).
     *
     * @return Vector of actions for this step: positive index = accepted (no drops in pure GP).
     */
    std::vector<int> updateActiveSet(const std::vector<std::size_t>& new_vars) override;

    /**
     * @brief Update residual using GP directional update:
     *        r^n = r^{n-1} - alpha_n * c^n.
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
     * @brief Compute correlations X^T * r for ALL non-dropped variables.
     *
     * @note GP override: computes for all variables (not just inactives) to allow re-selection.
     *       Base class only updates inactives.
     */
    void computeCorrelations() override;

};

#endif /* End of TGP_SOLVER_HPP */
