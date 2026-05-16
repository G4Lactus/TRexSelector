// ===================================================================================
// tomp_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_OMP_BASED_TOMP_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_OMP_BASED_TOMP_SOLVER_HPP
// ===================================================================================
/**
 * @file tomp_solver.hpp
 *
 * @brief Header file for the Terminating Orthogonal Matching Pursuit (T-OMP) solver.
 *
 * @details The TOMP_Solver class implements the T-OMP algorithm for variable selection
 * in linear regression, with support for dummy variable augmentation.
 * It inherits state management, standardization, and diagnostics from the
 * TSolver_Base class.
 *
 * Furthermore, it serves as mother class for:
 * - TGP_solver,
 * - TACGP_solver,
 * - TMP_solver,
 * - TOOLS_solver.
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

// Base class
#include <tsolvers/tsolver_base.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {

// ===================================================================================
/**
 * @brief Enumerates all supported TOMP-family solver algorithms.
 */
enum class SolverTypeOMPBased: int {
    TOMP = 0,        // Terminating Orthogonal Matching Pursuit
    TGP = 1,         // Terminating Gradient Pursuit
    TACGP = 2,       // Terminating Approximate Conjugate Gradient Pursuit
    TMP = 3,         // Terminating Matching Pursuit
    TOOLS = 4        // Terminating Orthogonal Least Squares (OLS)
};

/**
 * @brief Terminating Orthogonal Matching Pursuit (T-OMP) solver.
 * Inherits state management, standardization, and memory mapping from TSolver_Base.
 */
class TOMP_Solver : public TSolver_Base {
protected:
    /** @brief Algorithm variant type. */
    SolverTypeOMPBased algo_type_{SolverTypeOMPBased::TOMP};

    /** @brief Protected constructor for inheritance. */
    explicit TOMP_Solver(SolverTypeOMPBased type);

public:
    // ==========================================================================
    // Constructor/Destructor
    // ==========================================================================

    /**
     * @brief Construct a new TOMP_Solver object.
     *
     * @param X Map to predictor matrix (n x p).
     * @param D Map to the dummy predictor matrix (n x num_dummies).
     * @param y Map to the response vector (n x 1).
     * @param normalize Whether to normalize the predictors.
     * @param intercept Whether to include an intercept term.
     * @param verbose Whether to enable verbose output.
     * @param algorithm_type Specific T-OMP variant to execute
     * candidates are: TOMP, TGP, TACGP, TMP, TOOLS.
     */
    TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                Eigen::Map<Eigen::MatrixXd>& D,
                Eigen::Map<Eigen::VectorXd>& y,
                bool normalize = true,
                bool intercept = true,
                bool verbose = false,
                SolverTypeOMPBased algorithm_type = SolverTypeOMPBased::TOMP);

    /** @brief Default constructor for serialization. */
    TOMP_Solver();

    /** @brief Virtual destructor. */
    ~TOMP_Solver() override = default;

    /** @brief Deleted copy operations. */
    TOMP_Solver(const TOMP_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TOMP_Solver& operator=(const TOMP_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TOMP_Solver(TOMP_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TOMP_Solver& operator=(TOMP_Solver&&) = delete;

    // ==========================================================================
    // Main algorithm
    // ==========================================================================

    /**
     * @brief Execute one step of the T-OMP algorithm with early stopping based on
     *        dummy variable count.
     *
     * @param T_stop Number of dummies for early stopping threshold (0 = full path).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts and
     * serialization.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /**
     * @brief Get the string representation of the solver type based on TOMP class.
     *
     * @return String representation of the solver type.
     */
    std::string solverTypeToString() const override {
        switch (algo_type_) {
            case SolverTypeOMPBased::TOMP:      return "TOMP";
            case SolverTypeOMPBased::TGP:       return "TGP";
            case SolverTypeOMPBased::TACGP:     return "TACGP";
            case SolverTypeOMPBased::TMP:       return "TMP";
            case SolverTypeOMPBased::TOOLS:     return "TOOLS";
            default:                            return "UnknownSolver";
        }
    }

    // ==========================================================================
    // Serialization
    // ==========================================================================

    /** @brief Friend class for Cereal serialization. */
    friend class cereal::access;

    /**
     * @brief Serialize all internal model and path state except for input matrices.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template <class Archive>
    void serialize(Archive& archive) {
        int& algo_type_int = reinterpret_cast<int&>(algo_type_);
        archive(
            cereal::base_class<TSolver_Base>(this),
            CEREAL_NVP(algo_type_int)
        );
    }

    /**
     * @brief Save the current TOMP solver/model state to the given file (binary serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Output file path.
     *
     * @note Numerical quantities are saved up to a precision of 1e-12.
     */
    virtual void save(const std::string& filename) const;

    /**
     * @brief Load a TOMP solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TOMP_Solver object connected to X and D.
     */
    static TOMP_Solver load(const std::string& filename,
                            Eigen::Map<Eigen::MatrixXd>& X,
                            Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ==========================================================================
    // Internal OMP math helper
    // ==========================================================================

    /**
     * @brief Update the correlation vector X^T r for all inactives.
     */
    virtual void updateCorrelations();

    /**
     * @brief Attempt to update the active set by adding new variables and performing Cholesky
     * updates.
     *
     * @param new_vars Candidate indices for addition.
     *
     * @return Vector of actions for this step: positive index = accepted, negative = dropped.
     */
    virtual std::vector<int> updateActiveSet(const std::vector<std::size_t>& new_vars);

    /**
     * @brief Update residuals for OMP algorithm.
     */
    virtual void updateResiduals();

    /**
     * @brief Update the beta path for OMP algorithm.
     */
    virtual void updateBetaPath();
};

// =============================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
// =============================================================================
#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TOMP_SOLVER_HPP */
