// ===================================================================================
// tafs_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_AFS_BASED_TAFS_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_AFS_BASED_TAFS_SOLVER_HPP
// ===================================================================================
/**
 * @file tafs_solver.hpp
 *
 * @brief Terminating Adaptive Forward Stepwise (T-AFS) solver.
 */
// ===================================================================================

// std includes
#include <string>

// Eigen includes
#include <Eigen/Dense>

// Cereal includes
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>

// Base class
#include <tsolvers/tsolver_base.hpp>

// ===================================================================================

// Embedding into namespace trex::tsolvers::linear_model::afs_based
namespace trex::tsolvers::linear_model::afs_based {

// ===================================================================================

/**
 * @brief Terminating Adaptive Forward Stepwise (T-AFS) solver.
 * Implements and adapts the AFS algorithm by Zhang & Tibshirani (2026) into the
 * terminating solver framework. It interpolates between Forward Stepwise (rho = 1)
 * and LAR/LASSO (rho -> 0).
 */
class TAFS_Solver : public TSolver_Base {
protected:
    /**
     * @brief Shrinkage step size in (0, 1]. Forward Stepwise (rho = 1) to LAR/LASSO
     * (rho -> 0). Default is 1.0 (no shrinkage, i.e., Forward Stepwise).
     */
    double rho_{1.0};

public:
    // ==========================================================================
    // Constructor/Destructor
    // ==========================================================================

    /**
     * @brief Construct a TAFS_Solver object.
     *
     * @param X Map to the predictor matrix.
     * @param D Map to the dummy predictor matrix.
     * @param y Map to the response vector.
     * @param rho Shrinkage parameter in (0, 1]. rho=1 yields Forward Stepwise.
     * @param normalize Whether to internally standardize predictors.
     * @param intercept Whether to include an intercept in the model.
     * @param verbose Whether to print verbose output during execution.
     */
    TAFS_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                Eigen::Map<Eigen::MatrixXd>& D,
                Eigen::Map<Eigen::VectorXd>& y,
                double rho = 1.0,
                bool normalize = true,
                bool intercept = true,
                bool verbose = false,
                ScalingMode scaling_mode = ScalingMode::L2);

    /** @brief Default constructor for serialization. */
    TAFS_Solver();

    /** @brief Virtual destructor. */
    ~TAFS_Solver() override = default;

    /** @brief Deleted copy constructor. */
    TAFS_Solver(const TAFS_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TAFS_Solver& operator=(const TAFS_Solver&) = delete;

    /** @brief Defaulted move operations. */
    TAFS_Solver(TAFS_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TAFS_Solver& operator=(TAFS_Solver&&) = delete;

    // ==========================================================================
    // Main algorithm
    // ==========================================================================

    /**
     * @brief Execute T-AFS solution path until T_stop or early stopping criterion
     * is met.
     *
     * @param T_stop Maximum number of steps to execute. If 0, no limit is applied.
     * @param early_stop Whether to stop early if the stopping criterion is met.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /** @brief Return a string identifier for the solver type. */
    std::string solverTypeToString() const override {
        return "TAFS";
    }

    // ==========================================================================
    // Serialization
    // ==========================================================================

    /** @brief Friend class for cereal serialization. */
    friend class cereal::access;

    /**
     * @brief Serialize the TAFS solver object.
     *
     * @tparam Archive Cereal archive type for serialization.
     * @param archive Cereal archive object to serialize to.
     */
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TSolver_Base>(this),
            CEREAL_NVP(rho_)
        );
    }

    /**
     * @brief Save the current TAFS solver state to a file (binary serialization).
     *
     * @param filename Path to the output file.
     *
     * @note Numerical quantities are saved with a precision of 1e-12.
     */
    virtual void save(const std::string& filename) const;

    /**
     * @brief Load a TAFS solver state from a file (binary serialization).
     *
     * @param filename Path to the input file.
     * @param X Map to the predictor matrix.
     * @param D Map to the dummy predictor matrix.
     *
     * @return Deserialized TAFS_Solver object connected to X and D.
     */
    static TAFS_Solver load(const std::string& filename,
                            Eigen::Map<Eigen::MatrixXd>& X,
                            Eigen::Map<Eigen::MatrixXd>& D);

protected:
    // ==========================================================================
    // Internal Helpers
    // ==========================================================================

    /** @brief Updates the correlation vector against the current residual. */
    void updateCorrelations();
};

// ===================================================================================
} /* End of namespace trex::tsolvers::linear_model::afs_based */
// ===================================================================================
#endif /* TSOLVERS_LINEAR_MODEL_AFS_BASED_TAFS_SOLVER_HPP */
