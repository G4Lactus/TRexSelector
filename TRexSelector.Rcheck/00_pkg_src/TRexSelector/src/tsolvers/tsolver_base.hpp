// ===================================================================================
// tsolver_base.hpp
// ===================================================================================
#ifndef TSOLVERS_TSOLVER_BASE_HPP
#define TSOLVERS_TSOLVER_BASE_HPP
// ===================================================================================
/**
 * @file tsolver_base.hpp
 *
 * @brief Abstract base class TSolver_Base as template for all terminating variable
 * selection solvers.
 */
// ===================================================================================

// std includes
#include <limits>
#include <sstream>
#include <string>
#include <random>
#include <vector>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>


// Eigen includes
#include <Eigen/Dense>

// Utils includes
#include <utils/serialization/utils_cereal_eigen.hpp>

// ===================================================================================

namespace trex::tsolvers {

// ===================================================================================

/**
 * @brief Abstract base class for all terminating variable selection solvers.
 * Provides unified state management, memory-mapping support, and handles the
 * split design matrix (Original Predictors X + Dummy Predictors D) via a
 * unified index space.
 */
class TSolver_Base {
protected:
    // ==========================================================================
    // Data Members
    // ==========================================================================
    /** @brief Pointer to the original design matrix X (nullptr allows update). */
    Eigen::Map<Eigen::MatrixXd>* X_{nullptr};

    /** @brief Pointer to the dummy design matrix D (nullptr allows update). */
    Eigen::Map<Eigen::MatrixXd>* D_{nullptr};

    /** @brief Response vector y. */
    Eigen::VectorXd y_{};

    /** @brief Residual vector r. */
    Eigen::VectorXd r_{};

    /** @brief Correlations vector. */
    Eigen::VectorXd correlations_{};

    /** @brief Cholesky factor of the active set. */
    Eigen::MatrixXd R_{};

    /** @brief Rank of the last update to R. */
    int last_updateR_rank_{0};

    // ==========================================================================
    // Path and Diagnostics
    // ==========================================================================
    /** @brief Coefficient path for beta coefficients. */
    Eigen::MatrixXd betaPath_{};

    /** @brief Residual sum of squares at each step. */
    std::vector<double> RSS_{};

    /** @brief Coefficient of determination (R^2) at each step. */
    std::vector<double> R2_{};

    /** @brief Actions taken at each step. */
    std::vector<std::vector<int>> actions_{};

    /** @brief Degrees of freedom at each step. */
    std::vector<std::size_t> DoF_{};

    /** @brief Number of variable additions at each step. */
    std::size_t num_additions_{0};

    /** @brief Indices of active variables at each step. */
    std::vector<std::size_t> actives_{};

    /** @brief Indices of inactive variables at each step. */
    std::vector<std::size_t> inactives_{};

    /** @brief Indices of dropped variables at each step. */
    std::vector<std::size_t> dropped_indices_{};

    /** @brief Flag indicating if any variables have been dropped. Default is false. */
    bool any_dropped_{false};

    // ==========================================================================
    // Configuration & Preprocessing
    // ==========================================================================
    /** @brief Flag indicating if the data should be normalized. Default is true. */
    bool normalize_{true};

    /** @brief Flag indicating if an intercept should be included. Default is true. */
    bool intercept_{true};

    /** @brief Flag indicating if verbose output is enabled. Default is false. */
    bool verbose_{false};

    /** @brief Means of the original predictors X. */
    Eigen::VectorXd meansx_{};

    /** @brief Norms of the original predictors X. */
    Eigen::VectorXd normsx_{};

    /** @brief Means of the dummy predictors D. */
    Eigen::VectorXd meansd_{};

    /** @brief Norms of the dummy predictors D. */
    Eigen::VectorXd normsd_{};

    /** @brief Mean of the response variable y. Default is 0.0. */
    double mu_y_{0.0};

    /** @brief Maximum number of steps allowed. Default is 0. */
    std::size_t maxSteps_{0};

    /** @brief Current step in the algorithm. Default is 0. */
    std::size_t currentStep_{0};

    /** @brief Effective number of samples. Default is 0. */
    std::size_t effective_n_{0};

    /** @brief Maximum allowed size of the active set. Default is 0. */
    std::size_t max_actives_{0};

    /**
     * @brief Numerical tolerance for various checks (e.g., collinearity).
     * Default is machine epsilon.
     */
    const double eps_ = std::numeric_limits<double>::epsilon();

    /** @brief Flag indicating if the solver is connected to the data. Default is false. */
    bool is_connected_{false};

    // ==========================================================================
    // Dummy Tracking for FDR
    // ==========================================================================
    /** @brief Number of dummy variables. Default is 0. */
    std::size_t num_dummies_{0};

    /** @brief Number of active dummy variables. Default is 0. */
    std::size_t count_active_dummies_{0};

    /** @brief Number of original predictors. Default is 0. */
    std::size_t p_original_{0};

    /** @brief Start index of dummy variables. Default is 0. */
    std::size_t dummy_start_idx_{0};

    /** @brief Indices of dummy variables at each step. */
    std::vector<std::size_t> dummies_at_step_{};

    // ==========================================================================
    // Constructors
    // ==========================================================================
    /** @brief Default constructor. */
    TSolver_Base() = default;

    /**
     * @brief Constructor of a TSolver_Base object using data and configuration parameters.
     *
     * @param X Design matrix (n x p).
     * @param D Dummy design matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param normalize If true, each column of X (and D if provided) is scaled to unit L2-norm.
     * @param intercept If true, X and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     *
     * @note X and D are not copied nor owned, so changes to them alter the underlying data.
     *
     * @note The class has an internal column-dropping logic for unit L2-norm columns which is
     * active if normalize is true. Otherwise ensure that you exclude collinear columns a priori
     * before passing X or use an external preprocessing step for centering and l2-normalization.
     */
    TSolver_Base(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y,
                 bool normalize,
                 bool intercept,
                 bool verbose);

public:
    /** @brief Virtual default destructor. (X_, D_, not owned, not deleted). */
    virtual ~TSolver_Base() = default;

    /** @brief Deleted copy constructor. */
    TSolver_Base(const TSolver_Base&) = delete;

    /** @brief Deleted copy assignment operator. */
    TSolver_Base& operator=(const TSolver_Base&) = delete;

    /** @brief Defaulted move constructor. */
    TSolver_Base(TSolver_Base&&) = default;

    // ==========================================================================
    // Pure Virtual Interface
    // ==========================================================================
    /**
     * @brief Execute a single step of the solver.
     *
     * @param T_stop The maximum number of dummy variables to include into the active set.
     * @param early_stop If true, the solver may stop early. Otherwise, the full path is computed.
     */
    virtual void executeStep(std::size_t T_stop = 0, bool early_stop = true) = 0;

    /** @brief Return the type of the solver as a string. */
    virtual std::string solverTypeToString() const = 0;

    // ==========================================================================
    // Unified Public API
    // ==========================================================================
    /** @brief Return the estimated regression coefficients at a given step. */
    virtual Eigen::VectorXd getBeta(int step = -1) const;

    /** @brief Return the estimated intercept at a given step. */
    double getIntercept(int step = -1) const;

    /** @brief Return the estimated path matrix of regression coefficients. */
    virtual Eigen::MatrixXd getBetaPath() const;

    /** @brief Return the estimated Mallow's Cp values at each step. */
    virtual std::vector<double> getCp() const;

    /** @brief Return the estimated Mallow's Cp values at each step with provided sigma squared. */
    std::vector<double> getCp(double sigma_hat_sq) const;

    /**
     * @brief Restore the solver's state with new data matrices and response vector.
     *
     * @param X Reference to the new design matrix (n x p).
     * @param D Reference to the new dummy design matrix (n x num_dummies).
     * @param y Reference to the new response vector (n).
     */
    void restore(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y) const;

    /**
     * @brief Reconnect the solver with new data matrices.
     *
     * @param X Reference to the new design matrix (n x p).
     * @param D Reference to the new dummy design matrix (n x num_dummies).
     */
    virtual void reconnect(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::MatrixXd>& D);

    // --- Inline Getters (no explicit inline attribute necessary) ---
    /** @brief Return the residual sum of squares at each step. */
    const std::vector<double>& getRSS() const noexcept { return RSS_; }

    /** @brief Return the R-squared values at each step. */
    const std::vector<double>& getR2() const noexcept { return R2_; }

    /** @brief Return the degrees of freedom at each step. */
    const std::vector<std::size_t>& getDoF() const noexcept { return DoF_; }

    /** @brief Return the residuals at the current step. */
    Eigen::VectorXd getResiduals() const noexcept { return r_; }

    /** @brief Return the actions taken at each step. */
    const std::vector<std::vector<int>>& getActions() const noexcept { return actions_; }

    /** @brief Return the indices of active variables at each step. */
    const std::vector<std::size_t>& getActives() const noexcept { return actives_; }

    /** @brief Return the indices of inactive variables at each step. */
    const std::vector<std::size_t>& getInactives() const noexcept { return inactives_; }

    /** @brief Return the indices of dropped variables at each step. */
    const std::vector<std::size_t>& getDroppedIndices() const noexcept { return dropped_indices_; }

    /** @brief Return the number of active variables at the current step. */
    std::size_t getNumActives() const noexcept { return actives_.size(); }

    /** @brief Return the starting index of dummy variables. */
    std::size_t getDummyStartIndex() const noexcept { return dummy_start_idx_; }

    /** @brief Return the number of steps taken so far. */
    std::size_t getNumSteps() const noexcept { return currentStep_; }

    /** @brief Return whether the solver is currently connected to data. */
    bool isConnected() const noexcept { return is_connected_; }

    /** @brief Return the indices of active predictor variables at the current step. */
    std::vector<std::size_t> getActivePredictorIndices() const;

    /** @brief Return the indices of active dummy variables at the current step. */
    std::vector<std::size_t> getActiveDummyIndices() const;

protected:
    // ==========================================================================
    // Unified Internal Tools
    // ==========================================================================

    /**
     * @brief Unified index management. Returns an Eigen::Ref to the correct
     * column from either X_ or D_ with zero copy overhead.
     *
     * @param j The unified column index (0 to p_original + num_dummies - 1).
     *
     * @return A reference to the j-th column of X_ if j < p_original.
     */
    inline Eigen::Ref<const Eigen::VectorXd> getColumn(std::size_t j) const {
        if (j < p_original_) {
            return X_->col(static_cast<Eigen::Index>(j));
        } else {
            return D_->col(static_cast<Eigen::Index>(j - p_original_));
        }
    }

    /** @brief Preprocess the data as part of class constructor. */
    void preprocess();

    /** @brief Validate that the solver is connected to data. */
    void validateConnected() const;

    /** @brief Update the tracking of dummy variables. */
    void updateDummyTracking();

    /** @brief Initialize inactives_ to all predictor indices not in dropped_indices_. */
    void initializeInactives();

    /** @brief Rebuild inactives_ after drop event, exluding actives_ and dropped_indices_.
     * In no-dropping solvers this a no-op. Derived classes that perform active-set cycling
     * (e.g., T-LASSO, T-ENET) should call this after every removal.
     */
    void updateInactiveSet();

    // ============================================================================
    // Correlation Utilities
    // ============================================================================

    /**
    * @brief Initialize correlations_ as X^T y over all inactive columns.
    * Called once in the constructor after preprocessing.
    */
    void initializeCorrelations();

    /**
    * @brief Find all inactive columns tied at maximum absolute correlation.
    *
    * Returns the maximum value and all indices within eps_ of that maximum.
    * Virtual to allow OMP-family descendants (TGP, TACGP) to override
    * tie-selection behaviour.
    *
    * @return {Cmax, tied_indices}
    */
    virtual std::pair<double, std::vector<std::size_t>>
        findTiedMaxCorrelations() const;

    // ============================================================================
    // Cholesky update/downdate
    // ============================================================================

    /**
     * @brief Cholesky rank-1 update adding xnew to the active set factor R.
     *
     * @details
     * Default implementation covers the standard unpenalized case.
     * Override in penalized variants (e.g., TENET, TIENET) to incorporate
     * the L2 penalty term into the pivot computation.
     *
     * @param xnew  New column to incorporate (from getColumn(j)).
     * @param eps   Numerical tolerance for collinearity detection.
     *
     * @return      Updated (k+1 x k+1) upper-triangular Cholesky factor.
     */
    virtual Eigen::MatrixXd updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps);

    /**
     * @brief Cholesky rank-1 downdate removing the k-th variable from R.
     *
     * @details
     * Restores upper-triangular form after column removal via Givens rotations.
     * No penalty-specific modifications are required in the downdate — the L2
     * penalty effects are already encoded in R from the update phase.
     * Not used by the OMP family.
     *
     * @param R    Current upper-triangular Cholesky factor.
     * @param k    Index (within active set) of the variable to remove.
     * @param eps  Numerical tolerance to prevent zeros on the diagonal.
     *
     * @return     Updated (k-1 x k-1) upper-triangular Cholesky factor,
     *             or an empty matrix if the last variable is removed.
     */
    static Eigen::MatrixXd downdateR(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                     std::size_t k,
                                     double eps);

    // ==========================================================================
    // Triangular Solvers (stateless)
    // ==========================================================================

    /**
     * @brief Solve R x = b for x (upper-triangular, R n x n).
     *
     * @param R Upper-triangular matrix (Cholesky factor).
     * @param b Right-hand side vector.
     *
     * @return Solution vector x.
     *
     * @note Uses Eigen's triangular solve; throws if singular/ill-conditioned.
     */
    static Eigen::VectorXd backsolve(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                     const Eigen::Ref<const Eigen::VectorXd>& b);

    /**
     * @brief Solve R^T x = b for x (upper-triangular, R n x n).
     *
     * @param R Upper-triangular matrix (Cholesky factor).
     * @param b Right-hand side vector.
     *
     * @return Solution vector x.
     *
     * @note Uses Eigen's triangular solve; throws if singular/ill-conditioned.
     */
    static Eigen::VectorXd backsolveT(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                      const Eigen::Ref<const Eigen::VectorXd>& b);


    // ==========================================================================
    // Tie Handling
    // ==========================================================================

    /** @brief Random engine for tie-breaking dummy selection at T_stop boundary. */
    std::mt19937 rng_{std::random_device{}()};

    /**
     * @brief Set the random seed for reproducible tie-breaking.
     *
     * @param seed The seed value to initialize the random engine.
     */
    void setTieSeed(uint32_t seed) { rng_.seed(seed);}

    /** @brief Get the current random seed state (for serialization and reporting). */
    const std::mt19937& getRNG() const noexcept { return rng_; }

    /**
     * @brief Prune tied dummy candidates to respect T_stop budget.
     *
     * @details If new_vars contains more dummy entries than the remaining budget
     * (T_stop - count_active_dummies_), a random subset of size `budget` is retained.
     * Real predictor ties are never pruned.
     * No-op if early_stop is false, T_stop is 0, or new_vars has <= 1 entry.
     *
     * @param new_vars
     * @param T_stop
     * @param early_stop
     */
    void pruneTiedDummies(std::vector<std::size_t>& new_vars,
                          std::size_t T_stop,
                          bool early_stop);

    // ==========================================================================
    // Logging utilities
    // ==========================================================================

    /**
    * @brief Log a message.
    *
    * @param msg The message to log.
    */
    void logMsg(const std::string& msg) const;

    /**
    * @brief Log a warning message.
    *
    * @param msg The warning message to log.
    */
    void logWarning(const std::string& msg) const;

    /**
    *@brief Log an informational message.
    *
    * @param msg The informational message to log.
    */
    void logInfo(const std::string& msg) const;

    /**
     * @brief Concatenate multiple arguments into a single string message.
     *
     * @tparam Args The types of the arguments to concatenate.
     *
     * @param args The arguments to concatenate.
     * @return A string containing the concatenated arguments.
     */
    template<typename... Args>
    std::string concatMsg(Args&&... args) const {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }

    // ==========================================================================
    // Base Serialization
    // ==========================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all necessary data members to reconstruct the solver's state.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     *
     * @note X_ and D_ are not serialized and must be reconnected after deserialization.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            // Step tracking
            CEREAL_NVP(currentStep_), CEREAL_NVP(maxSteps_),

            // Solution path
            CEREAL_NVP(betaPath_), CEREAL_NVP(actions_),
            CEREAL_NVP(RSS_), CEREAL_NVP(R2_), CEREAL_NVP(DoF_),

            // Active set
            CEREAL_NVP(actives_), CEREAL_NVP(inactives_),
            CEREAL_NVP(dropped_indices_), CEREAL_NVP(max_actives_),
            CEREAL_NVP(num_additions_),

            // Cholesky Factor
            CEREAL_NVP(R_), CEREAL_NVP(last_updateR_rank_),

            // Residuals and correlations
            CEREAL_NVP(y_), CEREAL_NVP(r_), CEREAL_NVP(correlations_),

            // Preprocessing
            CEREAL_NVP(normalize_), CEREAL_NVP(intercept_),
            CEREAL_NVP(verbose_),
            CEREAL_NVP(meansx_), CEREAL_NVP(normsx_),
            CEREAL_NVP(meansd_), CEREAL_NVP(normsd_),
            CEREAL_NVP(mu_y_), CEREAL_NVP(effective_n_),

            // T-Solver specific
            CEREAL_NVP(num_dummies_), CEREAL_NVP(count_active_dummies_),
            CEREAL_NVP(p_original_), CEREAL_NVP(dummy_start_idx_),
            CEREAL_NVP(dummies_at_step_)
        );
    }
};

// ============================================================================
} // namespace trex::tsolvers

#endif /* TSOLVERS_TSOLVER_BASE_HPP */
