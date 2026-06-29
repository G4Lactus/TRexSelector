// ===================================================================================
// tncgmp_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_OMP_BASED_TNCGMP_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_OMP_BASED_TNCGMP_SOLVER_HPP
// ===================================================================================
/**
 * @file tncgmp_solver.hpp
 *
 * @brief Terminating Norm-Corrective Generalized Matching Pursuit (T-NCGMP) solver.
 *
 */
// ===================================================================================

// std includes
#include <string>

// Eigen includes
#include <Eigen/Dense>

// Cereal includes
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>

// tsolvers includes
#include <tsolvers/tsolver_base.hpp>

// ===================================================================================

// Embedding into namespace trex::tsolvers::linear_model::omp_based
namespace trex::tsolvers::linear_model::omp_based {

// ===================================================================================

/**
 * @brief Variants for the Norm-Corrective GMP solver.
 */
enum class NCGMPVariant : int {
    LineSearch = 0,       // Standard Matching Pursuit (1D projection)
    FullyCorrective = 1   // Orthogonal Matching Pursuit (Least Squares over active set)
};

/**
 * @brief Terminating Norm-Corrective Generalized Matching Pursuit (T-NCGMP) solver.
 */
class TNCGMP_Solver : public TSolver_Base {
protected:

    /** @brief Variant of the Norm-Corrective GMP solver. */
    NCGMPVariant variant_{NCGMPVariant::LineSearch};

public:
    // ==========================================================================
    // Constructor/Destructor
    // ==========================================================================

    /**
     * @brief Construct a new tncgmp solver object
     *
     * @param X Map to the predictor matrix (n x p).
     * @param D Map to the dummy predictor matrix (n x num_dummies).
     * @param y Map to the response vector (n x 1).
     * @param variant Variant of the NCGMP algorithm to use (Line Search or Fully Corrective).
     * @param normalize Whether to normalize the predictors.
     * @param intercept Whether to include an intercept term.
     * @param verbose Whether to enable verbose output.
     */
    TNCGMP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  NCGMPVariant variant = NCGMPVariant::LineSearch,
                  bool normalize = true,
                  bool intercept = true,
                  bool verbose = false,
                  ScalingMode scaling_mode = ScalingMode::L2);

    /** @brief Default constructor for serialization. */
    TNCGMP_Solver();

    /** @brief Virtual destructor. */
    ~TNCGMP_Solver() override = default;

    /** @brief Deleted copy constructor. */
    TNCGMP_Solver(const TNCGMP_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TNCGMP_Solver& operator=(const TNCGMP_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TNCGMP_Solver(TNCGMP_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TNCGMP_Solver& operator=(TNCGMP_Solver&&) = delete;

    // ==========================================================================
    // Main algorithm
    // ==========================================================================

    /**
     * @brief Execute a single step of the T-NCGMP algorithm.
     *
     * @param T_stop The maximum number of iterations to perform.
     * @param early_stop Whether to enable early stopping based on a convergence criterion.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /** @brief Get the string representation of the solver type. */
    std::string solverTypeToString() const override {
        return (variant_ == NCGMPVariant::LineSearch) ? "TNCGMP_LineSearch"
                                                      : "TNCGMP_FullyCorrective";
    }

    // ==========================================================================
    // Serialization
    // ==========================================================================

    /** @brief Friend class for serialization. */
    friend class cereal::access;

    /**
     * @brief Serialize the TNCGMP_Solver object.
     *
     * @tparam Archive The type of the archive.
     * @param archive The archive object.
     */
    template <class Archive>
    void serialize(Archive& archive) {
        int& variant_int = reinterpret_cast<int&>(variant_);
        archive(
            cereal::base_class<TSolver_Base>(this),
            CEREAL_NVP(variant_int)
        );
    }

    /**
     * @brief Save the current TNCGMP_Solver object to a file (binary serialization).
     *
     * @param filename The name of the file to save the object to.
     */
    virtual void save(const std::string& filename) const;

    /**
     * @brief Load a TNCGMP_Solver object from a file (binary deserialization).
     *
     * @param filename The name of the file to load the object from.
     * @param X Map to the predictor matrix (required for reconstruction).
     * @param D Map to the dummy predictor matrix (required for reconstruction).
     * @return TNCGMP_Solver The deserialized TNCGMP_Solver object.
     */
    static TNCGMP_Solver load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D);

protected:
    // ==========================================================================
    // Internal Helpers
    // ==========================================================================
    /** @brief Update the correlations between the residual and the predictors. */
    void updateCorrelations();

    /** @brief Update the beta path and residuals. */
    void updateBetaPathAndResiduals();

    /** @brief Rebuild the inactive set for the Line Search variant. */
    void rebuildInactiveSetForLineSearch();
};

// ===================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
// ===================================================================================
#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TNCGMP_SOLVER_HPP */
