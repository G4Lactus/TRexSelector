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
#include <tsolvers/ad/linear_model/omp_based/tomp_solver.hpp>

// ===================================================================================

// Embedding into namespace trex::tsolvers::ad::linear_model::omp_based
namespace trex::tsolvers::ad::linear_model::omp_based {

// ===================================================================================

/**
 * @brief Terminating Orthogonal Least Squares (T-OLS) solver.
 *
 * @details Inherits the execution flow and state management from TOMP_Solver, but
 * overrides the variable selection step to project candidate columns onto
 * the orthogonal complement of the active set prior to correlation calculation.
 *
 * The selection score of candidate j is |x_j^T r| / ||P_perp x_j||, with r the
 * least-squares residual on the active set (r is orthogonal to the active span,
 * so x_j^T r = (P_perp x_j)^T r). The squared projected norms of all candidates
 * are kept and downdated by (q^T x_j)^2 for every new orthonormal direction q of
 * the active span (q = X_A R^{-1} e_k, the k-th column of the thin Q factor),
 * inside the correlation pass: one O(n (p + L)) pass per step, as for T-OMP,
 * instead of re-orthogonalizing every candidate against the whole active set
 * (the modified Gram-Schmidt form of OLS, without an orthogonalized copy of
 * the dictionary).
 *
 *  References:
 *  -----------
 *   S. Chen, S. A. Billings, and W. Luo, "Orthogonal Least Squares Methods and Their Application to
 *     Non-Linear System Identification," Int. J. Control, vol. 50, no. 5, pp. 1873-1896, 1989,
 *     doi: 10.1080/00207178908953472.
 *   T. Blumensath and M. E. Davies, "On the Difference Between Orthogonal Matching Pursuit and Orthogonal
 *     Least Squares," unpublished note, draft of March 29, 2007.
 *   F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control,"
 *     in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
 */
class TOOLS_Solver : public tsolvers::ad::linear_model::omp_based::TOMP_Solver {
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
                 const SolverOptions& options = SolverOptions{});

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

    /** @brief TOMP_Solver::memoryBytes() plus the projected norms of all columns. */
    std::size_t memoryBytes() const noexcept override;

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
            cereal::base_class<trex::tsolvers::ad::linear_model::omp_based::TOMP_Solver>(this),
            CEREAL_NVP(proj_sq_), CEREAL_NVP(proj_rank_)
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
    // OLS State
    // ==========================================================================

    /** @brief Squared norms ||P_perp x_j||^2 of all columns (unified index)
     *  projected onto the orthogonal complement of the active span; current
     *  for the inactive columns. */
    Eigen::VectorXd proj_sq_{};

    /** @brief Number of active positions already downdated into proj_sq_. */
    std::size_t proj_rank_{0};

    // ==========================================================================
    // Core OLS Selection
    // ==========================================================================

    /**
     * @brief Find inactive variables tied for the maximum OLS selection score
     * |x_j^T r| / ||P_perp x_j|| (candidates with a numerically vanishing
     * projection are excluded).
     */
    std::pair<double, std::vector<std::size_t>> findTiedMaxCorrelations() const override;

    /**
     * @brief Downdate proj_sq_ for the active positions added since the last
     * call, fused with the refresh of the correlations X^T r of the inactives.
     */
    void updateCorrelations() override;
};

// ==================================================================================
} /* End of namespace trex::tsolvers::ad::linear_model::omp_based */
// ==================================================================================
#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TOOLS_SOLVER_HPP */
