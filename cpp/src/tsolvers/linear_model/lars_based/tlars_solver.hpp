// ===================================================================================
// tlars_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TLARS_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TLARS_SOLVER_HPP
// ===================================================================================
/**
 * @file tlars_solver.hpp
 *
 * @brief Header file for the Terminating Least Angle Regression (T-LARS) solver.
 *
 * @details The TLARS_Solver class implements the T-LARS algorithm for variable
 * selection in linear regression, with support for dummy variable augmentation.
 * It inherits state management, standardization, and diagnostics from the
 * TSolver_Base class.
 *
 * Furthermore, it serves as mother class for:
 * - TLASSO_solver,
 * - TSTEPWISE_solver,
 * - TENET_solver,
 * - TIENET_solver.
 */
// ===================================================================================

// std includes
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Cereal includes
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

// TSolver Base class include
#include <tsolvers/tsolver_base.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::linear_model::lars_based namespace
namespace trex::tsolvers::linear_model::lars_based {

// ===================================================================================

/**
 * @brief Enumerates all supported T-Solver family algorithms with
 * step-wise solution path based on the Least Angle Regression (LARS) framework.
 *
 * Used for algorithm selection, configuration, and logging/reporting.
 */
enum class SolverTypeLarsBased: int {
    TLARS = 0,      // Terminating Least Angle Regression (T-LARS)
    TLASSO = 1,     // T-LARS/LASSO with variable removal
    TSTEPWISE = 2,  // T-Stepwise selection (forward stepwise without variable removal)
    TENET = 3,      // T-Elastic Net (L1 + L2) regularization
    TIENET = 4,      // T-Informed Elastic Net
    TSTAGEWISE = 5  // T-Stagewise regression
};


/**
 * @brief Terminating Least Angle Regression (T-LARS) solver.
 * Supports in-memory and memory-mapped data over Eigen::Map.
 */
class TLARS_Solver : public TSolver_Base {
protected:
    // ==========================================================================
    // LARS specific state variables
    // ==========================================================================

    /** @brief Regularization parameter sequence (max. correlation per step). */
    std::vector<double> lambda_;

    /**
    * @brief Kahan summation compensation vector for incremental correlation updates.
     * Used to maintain numerical precision across multiple steps.
     */
    Eigen::VectorXd correlation_compensation_{};

    /** @brief Equiangular vector normalization value for current step. */
    double A_A_{};

    /** @brief Signs for active set variables (+1 or -1). */
    std::vector<int> Sign_;

    /** @brief Kahan summation refresh interval (steps before full recomputation). */
    std::size_t kahan_refresh_interval_{20};

    /** @brief Algorithm variant type. */
    SolverTypeLarsBased algo_type_{SolverTypeLarsBased::TLARS};

    /** @brief Protected constructor for inheritance. */
    explicit TLARS_Solver(SolverTypeLarsBased type);

public:

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    /**
     * @brief Construct a TLARS_Solver object using data and configuration.
     *
     * @param X Original design matrix (n x p).
     * @param D Dummy design matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param normalize If true, each column of X and D is scaled to unit L2-norm.
     * @param intercept If true, X, D, and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     * @param algorithm_type SolverTypeLarsBased variant.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     *
     * @note X and D are not copied nor owned, so changes to X or D alter the underlying
     *       data.
     *
     * @note Has an internal column-dropping logic for unit L2-norm columns
     *       which is active if normalize is true. Otherwise ensure that you
     *       exclude collinear columns a priori before passing X to the
     *       algorithm.
     */
    TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y,
                 bool normalize = true,
                 bool intercept = true,
                 bool verbose = false,
                 SolverTypeLarsBased algorithm_type = SolverTypeLarsBased::TLARS,
                 ScalingMode scaling_mode = ScalingMode::L2
                );

    /**
     * @brief Default constructor for serialization or deferred initialization.
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TLARS_Solver();

    /**
     * @brief Virtual destructor (X_ pointer is not owned, so not deleted).
     * @note Present for completeness and to support inheritance.
     */
    virtual ~TLARS_Solver() override = default;

    /** @brief Deleted copy constructor and assignment. */
    TLARS_Solver(const TLARS_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TLARS_Solver& operator=(const TLARS_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TLARS_Solver(TLARS_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TLARS_Solver& operator=(TLARS_Solver&&) = delete;

    // ============================================================================
    // Main Algorithm (Overrides Pure Virtual)
    // ============================================================================

    /**
     * @brief Compute T-LARS solution path with early stopping based on dummy
     * threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold,
     * (0 = full solution path, default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts and
     * serialization.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /**
     * @brief Convert enum to human-readable string for reporting/logging.
     *
     * @param type The SolverTypeLarsBased enum value.
     *
     * @return Static string name.
     */
    std::string solverTypeToString() const override{
        switch(algo_type_) {
            case SolverTypeLarsBased::TENET:       return "TENET";
            case SolverTypeLarsBased::TIENET:      return "TIENET";
            case SolverTypeLarsBased::TLARS:       return "TLARS";
            case SolverTypeLarsBased::TLASSO:      return "TLASSO";
            case SolverTypeLarsBased::TSTEPWISE:   return "TSTEPWISE";
            case SolverTypeLarsBased::TSTAGEWISE:  return "TSTAGEWISE";
            default:                               return "UnknownSolver";
        }
    }

    // ============================================================================
    // Specific Getters and Setters for TLARS
    // ============================================================================

    /** @brief Get regularization parameter sequence (lambda = max. correlation per step) */
    const std::vector<double>& getLambda() const noexcept { return lambda_; }

    /**
     * @brief Set Kahan summation refresh interval (steps between full recomputations).
     *
     * @param interval Number of steps between full refreshes computations.
     *                 Smaller values (10-20) provide higher numerical stability.
     *                 Larger values (50-100) reduce computational cost for stability.
     *
     * @note Must be called before executeStep(); Typical range: 10 - 50.
     */
    void setKahanRefreshInterval(std::size_t interval);

    /**
     * @brief Get current Kahan summation refresh interval.
     */
    std::size_t getKahanRefreshInterval() const noexcept { return kahan_refresh_interval_; }

    // ============================================================================
    // De-/Serialization
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal model and path state except for input matrices.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        int& algo_type_int = reinterpret_cast<int&>(algo_type_);
        archive(
            cereal::base_class<TSolver_Base>(this),
            CEREAL_NVP(algo_type_int),
            CEREAL_NVP(lambda_),
            CEREAL_NVP(correlation_compensation_),
            CEREAL_NVP(A_A_),
            CEREAL_NVP(Sign_),
            CEREAL_NVP(kahan_refresh_interval_)
        );
    }

    /**
     * @brief Save the current TLARS solver/model state to file (binary serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Path to output file.
     *
     * @note Numerical quantities are saved up to a precision of 1e-12.
     */
    virtual void save(const std::string& filename) const;

    /**
     * @brief Load TLARS solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TLARS_Solver object connected to X and D.
     */
    static TLARS_Solver load(const std::string& filename,
                             Eigen::Map<Eigen::MatrixXd>& X,
                             Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ============================================================================
    // Internal LARS math helper
    // ============================================================================

    /**
     * @brief Update correlations incrementally with periodic refresh.
     *
     * @details Uses Kahan's algorithm for incremental updates c_j = c_j - gamma * a_j,
     *          maintaining numerical stability. Full recomputation from residuals every
     *          20th step prevents long-term error drift accumulation.
     *
     * @param gamma Step size taken along equiangular direction.
     * @param a Projection vector where a[i] = X^T_{inactives[i]} * u from computeStepSize().
     *
     * @note Uses parallelized inner products for speed (OpenMP if available).
     */
    virtual void updateCorrelations(double gamma, const Eigen::Ref<const Eigen::VectorXd>& a);

    /**
     * @brief Attempt to update the active set by adding new variables and performing Cholesky
     * updates.
     *
     * @details For each candidate, tries Cholesky rank-1 update and accepts
     *          on rank increase, or flags column as dropped otherwise.
     *          Updates actives, dropped indices, and sign tracking.
     *          In non-cycling algorithms only additions are performed.
     *          Descendants with cycling may override to implement drops or
     *          cycling logic.
     *
     * @param new_vars Candidate indices for addition.
     *
     * @return Vector of actions for this step: positive index = accepted,
     *         negative = dropped.
     */
    std::vector<int> updateActiveSet(const std::vector<std::size_t>& new_vars);

    /**
     * @brief Update and record the solution path coefficients after each
     *        algorithmic step.
     *
     * @param w_a Current equiangular direction weights (for actives).
     * @param gamma Step size to move along direction.
     */
    void updateBetaPath(const Eigen::Ref<const Eigen::VectorXd>& w_A, double gamma);

    /**
     * @brief Construct sign vector (+1/-1) for current active set correlations.
     *
     * @return Eigen vector of size |active set| with entries +1/-1.
     */
    Eigen::VectorXd signVector() const;

    /**
     * @brief Compute T-LARS/Stagewise equiangular direction for current
     *        actives/signs.
     *
     * @param sign Sign vector over actives (+1/-1).
     *
     * @return Direction vector in active set space.
     */
    Eigen::VectorXd equiangularDirection(const Eigen::Ref<const Eigen::VectorXd>& sign);

    /**
     * @brief Compute the equiangular predictor vector for current actives.
     *
     * @param w_A Direction weights for actives.
     *
     * @return Combined predictor vector u.
     */
    virtual Eigen::VectorXd equiangularVector(const Eigen::Ref<const Eigen::VectorXd>& w_A) const;

    /**
     * @brief Compute maximal feasible step size along equiangular direction.
     *
     * @param Cmax Current maximal correlation (step size scaling).
     * @param u Equiangular predictor combination.
     *
     * @return Pair of {gamma, projection_vector} where projection[i] = X^T_{inactives} u.
     *         Maximum safe increment (gamma), constrained to no over-shooting/ties.
     */
    virtual std::pair<double, Eigen::VectorXd> computeStepSize(
        double Cmax,
        const Eigen::Ref<const Eigen::VectorXd>& u) const;

};

// ===================================================================================

} /* End of namespace trex::tsolvers::linear_model::lars_based */


#endif /* TSOLVERS_LINEAR_MODEL_LARS_BASED_TLARS_SOLVER_HPP */
