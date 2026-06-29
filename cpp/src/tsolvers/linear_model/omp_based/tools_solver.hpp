// ===================================================================================
// tools_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_OMP_BASED_TOOLS_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_OMP_BASED_TOOLS_SOLVER_HPP
// ===================================================================================
/**
 * @file tools_solver.hpp
 *
 * @brief Terminating Orthogonal Least Squares (T-OLS) solver.
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
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>

// ===================================================================================

// Embedding into namespace trex::tsolvers::linear_model::omp_based
namespace trex::tsolvers::linear_model::omp_based {

// ===================================================================================

/**
 * @brief Terminating Orthogonal Least Squares (T-OLS) solver.
 *
 * @details Inherits the execution flow and state management from TOMP_Solver, but
 * overrides the variable selection step to project candidate columns onto
 * the orthogonal complement of the active set prior to correlation calculation.
 */
class TOOLS_Solver : public tsolvers::linear_model::omp_based::TOMP_Solver {
public:
    // ==========================================================================
    // Constructor/Destructor
    // ==========================================================================

    /**
     * @brief Construct a new TOOLS_Solver object.
     *
     * @param X Map to the predictor matrix (n x p).
     * @param D Map to the dummy predictor matrix (n x num_dummies).
     * @param y Map to the response vector (n x 1).
     * @param normalize Whether to normalize the predictors.
     * @param intercept Whether to include an intercept term.
     * @param verbose Whether to enable verbose output.
     */
    TOOLS_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y,
                 bool normalize = true,
                 bool intercept = true,
                 bool verbose = false,
                 ScalingMode scaling_mode = ScalingMode::L2);

    /** @brief Default constructor for serialization. */
    TOOLS_Solver();

    /** @brief Virtual destructor. */
    ~TOOLS_Solver() override = default;

    /** @brief Deleted copy operations. */
    TOOLS_Solver(const TOOLS_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TOOLS_Solver& operator=(const TOOLS_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TOOLS_Solver(TOOLS_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TOOLS_Solver& operator=(TOOLS_Solver&&) = delete;

    /** @brief Get the string representation of the solver type. */
    std::string solverTypeToString() const override {
        return "TOOLS";
    }

    // ==========================================================================
    // Serialization
    // ==========================================================================

    /** @brief Friend class for serialization. */
    friend class cereal::access;

    /**
     * @brief Serialize the TOOLS_Solver object.
     *
     * @tparam Archive The type of the archive.
     * @param archive The archive object.
     */
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<trex::tsolvers::linear_model::omp_based::TOMP_Solver>(this)
        );
    }

    /**
     * @brief Save the current TOOLS_Solver object to a file (binary serialization).
     *
     * @param filename The name of the file to save the object to.
     */
    virtual void save(const std::string& filename) const override;

    /**
     * @brief Load a TOOLS_Solver object from a file (binary deserialization).
     *
     * @param filename The name of the file to load the object from.
     * @param X Map to the predictor matrix (used for reconnecting after loading).
     * @param D Map to the dummy predictor matrix (used for reconnecting after loading).
     * @return A TOOLS_Solver object loaded from the specified file.
     */
    static TOOLS_Solver load(const std::string& filename,
                             Eigen::Map<Eigen::MatrixXd>& X,
                             Eigen::Map<Eigen::MatrixXd>& D);

protected:
    // ==========================================================================
    // Core OLS Selection
    // ==========================================================================

    /**
     * @brief Find inactive variables tied for the maximum OLS selection score.
     * * Calculates the correlation of the residual with the normalized orthogonal
     * projection of each inactive variable using the Cholesky factor R_.
     */
    std::pair<double, std::vector<std::size_t>> findTiedMaxCorrelations() const override;
};

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
// ==================================================================================
#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TOOLS_SOLVER_HPP */
