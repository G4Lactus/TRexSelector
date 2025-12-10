#ifndef TOMP_SOLVER_HPP
#define TOMP_SOLVER_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <cereal/archives/portable_binary.hpp>

#include "utils_cereal_eigen.hpp"
#include "utils_openmp.hpp"

/**
 * @brief Enumerates all supported TOMP-family solver algorithms with step-wise solution path.
 *
 * Used for algorithm selection, configuration, and logging/reporting
 *
 */
enum class SolverType: int {
    TOMP = 0,        // Terminating Orthogonal Matching Pursuit
    TGP = 1,         // Terminating Gradient Pursuit
    TACGP = 2        // Terminating Approximate Conjugate Gradient Pursuit
};


/**
 * @brief Terminating Orthogonal Matching Pursuit (T-OMP) solver with dummy variables for
 *        FDR control.
 *
 * @details
 *  Implements T-OMP with:
 *  - Dummy variable augmentation for FDR-controlled variable selection
 *  - Early stopping when T_stop dummies enter active set
 *  - Efficient O(n) incremental correlation tracking
 *  - Rank-robust, and numerically stable Cholesky updates (rowwise/batch/tie compatible)
 *  - Numerical tie-breaking: multiple atoms within tolerance are added simultaneously
 *  - Pathwise solution tracking and selectable model diagnostics
 *  - Support for memory-mapped (MMAP) and in-memory data
 *  - External serialization and warm-start functionality
 *
 * @note This implementation prioritizes numerical stability over strict textbook OMP.
 *       When multiple atoms have correlations within floating-point tolerance of the maximum,
 *       all tied atoms are added simultaneously (batch-OMP behavior) rather than making
 *       arbitrary selections based on numerical noise.
 */
class TOMP_Solver {

protected:

    // ==========================================================================
    // Data Members
    // ==========================================================================

    /**
     * @brief Pointer to augmented design matrix X = [X_original | X_dummies] (RAM/MMAP),
     *         not owned, nor serialized.
     *         Must be reconnected after serialization. Assumed to be column-major.
     */
    Eigen::Map<Eigen::MatrixXd>* X_{nullptr};

    /** @brief Response (centered if intercept fitted).
     *         Holds working copy of y for all computations.
     */
    Eigen::VectorXd y_{};


    // ==========================================================================
    // Core state of OMP
    // ==========================================================================

    /** @brief Residual vector. */
    Eigen::VectorXd r_{};

    /** @brief Correlation vector X^T r for all inactive variables. */
    Eigen::VectorXd correlations_{};

    /** @brief Cholesky factor of active Gram matrix (X_active^T X_active).
     *         Used for rank-1 updates on active set.
     */
    Eigen::MatrixXd R_{};

    /** @brief Rank tracker for R_ (active set). Supports robust detection and rejection of
     *         nearly collinear inclusions.
     */
    int last_updateR_rank_{};


    // ==========================================================================
    // Path and Diagnostics
    // ==========================================================================

    /** @brief Full path of coefficients for computed solution path ((p + num_dummies) x steps).
     *         Each column is a model along solution path. Includes dummy coefficients.
     */
    Eigen::MatrixXd betaPath_{};

    /** @brief Residual sum of squares for each model on the path. */
    std::vector<double> RSS_;

    /** @brief R-squared statistics for each step (explained variance vs. null) */
    std::vector<double> R2_;

    /**
     * @brief List of variable entry (+j) actions per step.
     *        Encodes exact path traversal and tie handling (multiple variables per step
     *        theoretically possible).
     *
     * @note OMP only adds variables; drops (-j) may occur only for collinearity handling.
     */
    std::vector<std::vector<int>> actions_;

    /** @brief Degrees of freedom (active set size (including dummies) + intercept if fitted) for
     *         each model.
     */
    std::vector<std::size_t> DoF_{};

    /** @brief Monotonic counter of variable additions to actives_ (not decremented). */
    std::size_t num_additions_{0};

    // ==========================================================================
    // Variable tracking
    // ==========================================================================

    /** @brief Indices of variables currently in the active set
     *         (includes both real and dummy variables).
     */
    std::vector<std::size_t> actives_{};

    /** @brief Indices of variables currently available for entry. */
    std::vector<std::size_t> inactives_{};

    /** @brief Indices of variables permanently dropped due to collinearity or zero variance. */
    std::vector<std::size_t> dropped_indices_{};

    /** @brief Flag variable for collinear drops, used in descendant classes */
    bool any_dropped_{false};

    // ==========================================================================
    // Preprocessing & Config
    // ==========================================================================

    /** @brief If true, columns of X are scaled to unit norm before regression. */
    bool normalize_{true};

    /** @brief If true, estimate intercept; X and y are centered before fitting. */
    bool intercept_{true};

    /** @brief If true, prints detailed status/progress information to console. */
    bool verbose_{false};

    /** @brief Means of columns of X (computed if intercept enabled). */
    Eigen::VectorXd meansx_{};

    /** @brief L2 norms of columns of X (computed if normalization enabled). */
    Eigen::VectorXd normsx_{};

    /** @brief Mean of y (computed if intercept enabled). */
    double mu_y_{0.0};

    /** @brief Maximum allowed steps for algorithm (auto-determined from p, n). */
    std::size_t maxSteps_{0};

    /** @brief Counter for current path step during solution sequence. */
    std::size_t currentStep_{0};

    /** @brief Effective sample size after centering/intercept handling. */
    std::size_t effective_n_{0};

    /** @brief Maximum allowed active set size. */
    std::size_t max_actives_{0};

    /** @brief  Numerical tolerance/threshold constant (used for collinearity rejection,
     *          norm checks, etc.).
     */
    const double eps_ = std::numeric_limits<double>::epsilon();

    /**
     * @brief True if X_ pointer is set and connected for computation.
     *        Cleared on serialization, must be reset by reconnect().
     */
    bool is_connected_{false};

    // ==========================================================================
    // T-Algorithm Specific Members
    // ==========================================================================

    /** @brief Total number of dummy variables in augmented X. */
    std::size_t num_dummies_{0};

    /** @brief Number of dummies currently in active set. */
    std::size_t count_active_dummies_{0};

    /** @brief Original predictor count (before augmentation with dummies). */
    std::size_t p_original_{0};

    /** @brief First dummy column index in X (= p_original). */
    std::size_t dummy_start_idx_{0};

    /** @brief Count of dummies in active set at each step (for tracking). */
    std::vector<std::size_t> dummies_at_step_;

    /**
     * @brief Algorithm variant type.
     */
    SolverType algo_type_{SolverType::TOMP};

    /**
     * @brief Protected constructor for inheritance.
     *
     * @param type Algorithm variant as SolverType enum.
     */
    explicit TOMP_Solver(SolverType type);

public:

    // ==========================================================================
    // Constructor/Destructor
    // ==========================================================================

    /**
     * @brief Construct a TOMP_Solver object using data and configuration.
     *
     * @param X Augmented design matrix [X_original | X_dummies] (RAM/MMAP, column-major,
     *          not owned, must persist for solver lifetime).
     * @param y Response vector.
     * @param num_dummies Number of dummy columns at the end of X.
     * @param normalize If true, each column of X is scaled to unit L2-norm.
     * @param intercept If true, X and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     * @param algorithm_type SolverType variant (default: TOMP).
     *
     * @note X is not copied nor owned, so changes to X alter the underlying data.
     */
    TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                Eigen::Map<Eigen::VectorXd>& y,
                std::size_t num_dummies,
                bool normalize = true,
                bool intercept = true,
                bool verbose = false,
                SolverType algorithm_type = SolverType::TOMP
               );

    /**
     * @brief Default constructor for serialization or deferred initialization.
     *
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TOMP_Solver();

    /**
     * @brief Virtual destructor (X_ pointer is not owned, so not deleted).
     *
     * @note Present for completeness and to support inheritance.
     */
    virtual ~TOMP_Solver() = default;

    /** @brief Deleted copy constructor to avoid raw pointer ownership problems. */
    TOMP_Solver(const TOMP_Solver&) = delete;

    /**
     * @brief Deleted copy assignment operator to avoid raw pointer ownership problems.
     */
    TOMP_Solver& operator=(const TOMP_Solver&) = delete;

    /** @brief Move constructor (default implementation). */
    TOMP_Solver(TOMP_Solver&&) = default;


    // ==========================================================================
    // Main algorithm
    // ==========================================================================

    /**
     * @brief Compute T-OMP solution path with early stopping based on dummy threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold (0 = full solution path,
     *        default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts and serialization.
     */
    virtual void executeStep(std::size_t T_stop = 0, bool early_stop = true);


    // ==========================================================================
    // Output & accessors
    // ==========================================================================

    /**
     * @brief Retrieve coefficient vector for a given path step (latest if step == -1).
     *
     * @param step Index along path (-1 for last step).
     *
     * @return Coefficient vector (size p + num_dummies, includes dummies).
     */
    Eigen::VectorXd getBeta(int step = -1) const;

    /**
     * @brief Retrieve the estimated intercept (mean adjustment) for a given path step.
     *
     * @param step Path step index (-1 = last).
     *
     * @return Scalar intercept.
     */
    double getIntercept(int step = -1) const;

    /**
     * @brief Retrieve the full beta solution path ((p + num_dummies) x steps) for all
     *        computed steps.
     *
     * @return Matrix of shape (p + num_dummies, steps), includes dummy coefficients.
     */
    Eigen::MatrixXd getBetaPath() const;

    /**
     * @brief Retrieve vector of residual sum-of-squares for each path model.
     *
     * @return RSS vector.
     */
    inline const std::vector<double>& getRSS() const noexcept { return RSS_; }

    /**
     * @brief Retrieve vector of R-squared statistics for each path model.
     *
     * @return R2 vector (explained variance fraction).
     */
    inline const std::vector<double>& getR2() const noexcept { return R2_; }

    /**
     * @brief Compute Mallows' Cp statistics for all steps using internally estimated
     *        residual variance.
     *
     * @return Cp vector (NaN for DoF >= n (high-dimensionality)).
     *
     * @note Use this for standard model selection; robust to ill-posed steps.
     * @note DoF excludes active dummies from count.
     */
    virtual std::vector<double> getCp() const;

    /**
     * @brief Compute Mallows' Cp statistics using a user-supplied residual variance.
     *
     * @param sigma_hat_sq Reference variance (e.g., from OLS or cross-validation).
     *
     * @return Cp vector (NaN for DoF >= n (high-dimensionality) or nonpositive variance).
     *
     * @note Use for cross-method/model comparison.
     */
    std::vector<double> getCp(double sigma_hat_sq) const;

    /**
     * @brief Retrieve vector of degrees-of-freedom estimates
     *        (number of variables (including dummies) + intercept if used).
     *
     * @return DoF vector (path length).
     */
    inline const std::vector<std::size_t>& getDoF() const noexcept { return DoF_; }

    /**
     * @brief Return current residual vector at last path step.
     *
     * @return Residual vector (size n).
     */
    inline Eigen::VectorXd getResiduals() const noexcept { return r_; }

    // ==========================================================================
    // Quick Diagnostics
    // ==========================================================================

    /**
     * @brief Get actions taken at each step (entering variables as indices).
     *
     * @return Vector of integer vectors, each inner vector holds actions at that step.
     *         Positive index = variable added. Standard OMP does not drop variables.
     */
    const std::vector<std::vector<int>>& getActions() const noexcept { return actions_; }

    /**
     * @brief Get indices of currently active variables (includes both real and dummy variables).
     *
     * @return Vector of active variable indices.
     */
    const std::vector<std::size_t>& getActives() const noexcept { return actives_; }

    /**
     * @brief Get indices of currently available (inactive) variables.
     *
     * @return Vector of indices not in active set.
     */
    const std::vector<std::size_t>& getInactives() const noexcept { return inactives_; }

    /**
     * @brief Get indices of variables permanently dropped (collinear/zero variance).
     *
     * @return Vector of dropped variable indices.
     */
    const std::vector<std::size_t>& getDroppedIndices() const noexcept { return dropped_indices_; }

    /**
     * @brief Get number of currently active variables (includes both real and dummy variables).
     *
     * @return Active set size.
     */
    inline std::size_t getNumActives() const noexcept { return actives_.size(); }

    /**
     * @brief Get number of computed solution path steps.
     *
     * @return Path step count.
     */
    std::size_t getNumSteps() const noexcept { return currentStep_; }

    /**
     * @brief Return sum of squares of y at null fit (step 0).
     *
     * @return Null model residual sum of squares (RSS).
     */
    double getSSY() const noexcept { return !RSS_.empty() ? RSS_[0] : 0.0; }

    /**
     * @brief Report verbose/logging status (true if enabled).
     *
     * @return True if verbose mode active.
     */
    inline bool getVerbose() const noexcept { return verbose_; }

    /**
     * @brief Report if normalization of columns is enabled.
     *
     * @return True if column scaling applied.
     */
    inline bool getNormalize() const noexcept { return normalize_; }

    /**
     * @brief Get mean of y (computed if intercept fitted).
     *
     * @return Mean value of y.
     */
    double getMuY() const noexcept { return mu_y_; }

    /**
     * @brief Get mean vector of columns of X (if intercept computed).
     *
     * @return Vector of column means.
     */
    inline const Eigen::VectorXd& getMeansx() const noexcept { return meansx_; }

    /**
     * @brief Get L2-norm vector for columns of X (if normalization computed).
     *
     * @return Vector of column norms.
     */
    inline const Eigen::VectorXd& getNormsx() const noexcept { return normsx_; }

    // ==========================================================================
    // T-Algorithm Specific Getters
    // ==========================================================================

    /**
     * @brief Get original predictor count (before dummy augmentation).
     *
     * @return Number of real predictors.
     */
    std::size_t getPOriginal() const noexcept { return p_original_; }

    /**
     * @brief Get number of dummy variables in augmented design.
     *
     * @return Total dummy count.
     */
    std::size_t getNumDummies() const noexcept { return num_dummies_; }

    /**
     * @brief Get number of active dummies in current model.
     *
     * @return Active dummy count.
     */
    std::size_t getCountActiveDummies() const noexcept { return count_active_dummies_; }

    /**
     * @brief Get starting index of dummy variables in design matrix.
     *
     * @return Index where dummies start (p_original).
     */
    std::size_t getDummyStartIndex() const noexcept { return dummy_start_idx_; }

    /**
     * @brief Get history of active dummy counts at each step.
     *
     * @return Vector of dummy counts per step.
     */
    const std::vector<std::size_t>& getDummiesAtStep() const noexcept { return dummies_at_step_; }

    /**
     * @brief Get indices of currently active predictor variables (excluding dummies).
     *
     * @return Vector of active predictor indices.
     */
    std::vector<std::size_t> getActivePredictorIndices() const;

    /**
     * @brief Get indices of currently active dummy variables.
     *
     * @return Vector of active dummy indices.
     */
    std::vector<std::size_t> getActiveDummyIndices() const;

    // ==========================================================================
    // Restoration
    // ==========================================================================

    /**
     * @brief Restore original X, y for a given fitted path by applying reverse centering/scaling.
     *
     * @param X Matrix to be restored in place
     * @param y Response vector to be restored in place
     *
     * @note Applies stored means/norms/mu_y and throws on dimension mismatch.
     */
    void restore(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y) const;

    // ==========================================================================
    // Serialization & state management
    // ==========================================================================

    /**
     * @brief Serialize all internal model and path state except for X_.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     *
     * @note X_ must be reconnected after deserialization.
     */
    template <class Archive>
    void serialize(Archive& archive) {
        int& algo_type_int = reinterpret_cast<int&>(algo_type_);
        archive(
            // Algorithm type
            CEREAL_NVP(algo_type_int),

            // Step tracking
            CEREAL_NVP(currentStep_),
            CEREAL_NVP(maxSteps_),

            // Solution path
            CEREAL_NVP(betaPath_),
            CEREAL_NVP(actions_),
            CEREAL_NVP(RSS_),
            CEREAL_NVP(R2_),
            CEREAL_NVP(DoF_),

            // Active set
            CEREAL_NVP(actives_),
            CEREAL_NVP(inactives_),
            CEREAL_NVP(dropped_indices_),
            CEREAL_NVP(max_actives_),
            CEREAL_NVP(num_additions_),

            // Cholesky
            CEREAL_NVP(R_),
            CEREAL_NVP(last_updateR_rank_),

            // Residuals and correlations
            CEREAL_NVP(y_),
            CEREAL_NVP(r_),
            CEREAL_NVP(correlations_),

            // Preprocessing
            CEREAL_NVP(normalize_),
            CEREAL_NVP(intercept_),
            CEREAL_NVP(verbose_),
            CEREAL_NVP(meansx_),
            CEREAL_NVP(normsx_),
            CEREAL_NVP(mu_y_),
            CEREAL_NVP(effective_n_),

            // T-Algorithm specific
            CEREAL_NVP(num_dummies_),
            CEREAL_NVP(count_active_dummies_),
            CEREAL_NVP(p_original_),
            CEREAL_NVP(dummy_start_idx_),
            CEREAL_NVP(dummies_at_step_)
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
     * @return Deserialized TOMP_Solver object connected to X.
     */
    static TOMP_Solver load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X);

    /**
     * @brief Restore pointer connection to previously used X_, required after deserialization.
     *
     * @param X Pointer to user-managed augmented design matrix.
     *
     * @note Does not alter internal model state and must be called after any load/serialize
     *       sequence.
     */
    void reconnect(Eigen::Map<Eigen::MatrixXd>& X);

    /**
     * @brief Check if design matrix pointer is valid and connected.
     *
     * @return True if ready for algorithm execution.
     */
    inline bool isConnected() const noexcept { return is_connected_; }

    /**
     * @brief Convert enum to human-readable string for reporting/logging.
     *
     * @param type The SolverType enum value.
     *
     * @return Static string name.
     */
    static const char* solverTypeToString(SolverType type) {
        switch (type) {
            case SolverType::TOMP:      return "TOMP";
            case SolverType::TGP:       return "TGP";
            case SolverType::TACGP:     return "TACGP";
            default:                    return "Unknown SolverType";
        }
    }

    /**
     * @brief Return static string name for current instance algorithm type.
     *
     * @return Name of subclass algorithm.
     */
    const char* solverTypeToString() const { return solverTypeToString(algo_type_); }


protected:

    // ==========================================================================
    // Preprocessing & core algebra
    // ==========================================================================

    /**
     * @brief Center and/or scale X and y according to current configuration (normalize/intercept).
     *
     * @note Modifies X and y in-place. Columns with low variance may be dropped.
     *
     * @throws std::runtime_error on major dimension mismatch or NaN encountered.
     */
    void preprocess();


    // ==========================================================================
    // Cholesky update
    // ==========================================================================

    /**
     * @brief Incrementally update the Cholesky factor R for active set with a new column.
     *
     * @param xnew New variable column to incorporate (typically from X->col(j)).
     * @param eps Numerical tolerance for collinearity/degeneracy.
     *
     * @return Updated (k+1 x k+1) Cholesky matrix, or previous R if no rank increase.
     */
    Eigen::MatrixXd updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps);


    // ==========================================================================
    // Triangular solvers
    // ==========================================================================

    /**
     * @brief Solve Rx = b for x (upper-triangular, R n x n).
     *
     * @param R Upper-triangular matrix (Cholesky factor).
     * @param b Right-hand side vector.
     *
     * @return Solution vector x.
     *
     * @note Uses Eigen triangular view; throws if singular/ill-conditioned.
     */
    Eigen::VectorXd backsolve(const Eigen::Ref<const Eigen::MatrixXd>& R, const Eigen::Ref<const Eigen::VectorXd>& b) const;

    /**
     * @brief Solve R^T x = b for x (lower-triangular, R n x n).
     *
     * @param R Upper-triangular matrix (original Cholesky factor).
     * @param b Right-hand side vector.
     *
     * @return Solution vector x.
     *
     * @note Uses Eigen triangular view; throws if singular/ill-conditioned.
     */
    Eigen::VectorXd backsolveT(const Eigen::Ref<const Eigen::MatrixXd>& R, const Eigen::Ref<const Eigen::VectorXd>& b) const;


    // ===== Internal OMP helper =====

    /**
     * @brief Efficiently setup correlation vector X^T y for all inactives.
     *
     * @note Uses parallelized inner products for speed (OpenMP if available).
     */
    void initializeCorrelations();

    /**
     * @brief Efficiently update correlation vector X^T r for all inactives.
     *
     * @note Uses parallelized inner products for speed (OpenMP if available).
     */
    virtual void computeCorrelations();

    /**
     * @brief Identify all inactives tied at maximum absolute correlation.
     *
     * @return Maximum absolute correlation and vector of indices for variables tied at max |cor|.
     *
     * @note Implements numerical tie-breaking for stability: atoms with correlations within
     *       ±eps of the maximum are considered tied and returned together. This avoids
     *       arbitrary selection based on floating-point noise and enables batch additions
     *       when atoms are genuinely equivalent. Prioritizes numerical stability over
     *       strict single-atom-per-iteration textbook OMP.
     */
    virtual std::pair<double, std::vector<std::size_t>> findTiedMaxCorrelations() const;

    /**
     * @brief Attempt to update the active set by adding new variables.
     *
     * @details For each candidate, tries Cholesky rank-1 update and accepts on rank increase,
     *          or flags column as dropped otherwise. Updates actives and dropped indices.
     *          Supports batch additions when multiple atoms are tied at maximum correlation
     *          (from numerical tie-breaking). Each candidate is validated independently via
     *          Cholesky update; collinear atoms are rejected even if initially tied.
     *
     * @param new_vars Candidate indices for addition (may contain multiple tied atoms).
     *
     * @return Vector of actions for this step: positive index = accepted, negative = dropped.
     */
    virtual std::vector<int> updateActiveSet(const std::vector<std::size_t>& new_vars);


    /**
     * @brief Initialize the inactive set with all predictors not yet active or dropped.
     *
     * @note Parallel support with OpenMP is available.
     */
    void initializeInactives();


    /**
     * @brief Updates inactives_ to all predictors not in actives_ or dropped_indices_.
     *
     * @details Rebuilds inactives_ as all predictors not currently in actives_ or dropped_indices_.
     *          Should be called by any descendant of OMP_Solver that supports active cycling
     *          in extended variants.
     *
     * @note Parallel support with OpenMP is available.
     */
    void updateInactiveSet();

    /**
     * @brief Update residual vector after active set changes.
     */
    virtual void updateResiduals();

    /**
     * @brief Update and record the solution path coefficients after each algorithmic step.
     */
    virtual void updateBetaPath();

    /**
     * @brief Update count of active dummies and dummies_at_step_ vector.
     *
     * @note Called after each step to track dummy variable selection for FDR control.
     */
    void updateDummyTracking();


    // ===== Validation =====

    /**
     * @brief Ensure design matrix is connected and state is internally coherent.
     *
     * @throws std::runtime_error if not connected or if serious shape mismatch detected.
     */
    void validateConnected() const;

    // ===== Logging =====

    /**
     * @brief Print information string if verbose or trace is enabled.
     *
     * @param msg Log message
     */
    void logMsg(const std::string& msg) const;

    /**
     * @brief Print warning message (prefix [WARNING]) if verbose or trace is enabled.
     *
     * @param msg Warning detail
     */
    void logWarning(const std::string& msg) const;

    /**
     * @brief Print informational message (prefix [Info]) if verbose/trace.
     *
     * @param msg Status detail
     */
    void logInfo(const std::string& msg) const;

    /**
     * @brief Concatenate arbitrary arguments into string (stream-style) for logging.
     *
     * @tparam Args Argument pack.
     *
     * @param args Log message fragments.
     *
     * @return std::string with all args concatenated.
     */
    template<typename... Args>
    std::string concatMsg(Args&&... args) const {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }
};

#endif /* End of TOMP_SOLVER_HPP */
