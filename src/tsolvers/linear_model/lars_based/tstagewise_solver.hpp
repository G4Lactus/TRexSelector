// ============================================================================
// tstagewise_solver.hpp
// ============================================================================
/**
 * @file tstagewise_solver.hpp
 *
 * @brief Declaration of TSTAGEWISE_Solver, a T-LARS-based stagewise regression
 * solver.
 */
// ============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTAGEWISE_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTAGEWISE_SOLVER_HPP
// ============================================================================

// std includes
#include <string>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::linear_model::lars_based
namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================

/**
 * @brief T-Stagewise solver via LARS-Stagewise algorithm.
 *
 * @details Implements the Incremental Forward Stagewise Regression path.
 * Enroces monotonicity, i.e., coefficients must evolve in the direction of their
 * correlation signs (w_j * Sign_j > 0).
 * If this is violated, a non-negative least squares (NNLS) reduction is applied
 * to the active set.
 */
class TSTAGEWISE_Solver : public TLARS_Solver {

protected:
    // ========================================================================
    // STAGEWISE specific states
    // ========================================================================

    /** @brief Number of coefficient removals performed. */
    std::size_t num_removals_{0};

public:
    // ========================================================================
    // Constructor/Destructor
    // ========================================================================

    /**
     * @brief Construct a new tstagewise solver object
     *
     * @param X Original design matrix (n x p).
     * @param D Dummy design matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param normalize If true, each column of X is scaled to unit L2-norm.
     * @param intercept If true, X and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     */
    TSTAGEWISE_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                      Eigen::Map<Eigen::MatrixXd>& D,
                      Eigen::Map<Eigen::VectorXd>& y,
                      bool normalize = true,
                      bool intercept = true,
                      bool verbose = false);

    /** @brief Default constructor for serialization or deferred initialization. */
    TSTAGEWISE_Solver() : TLARS_Solver(SolverTypeLarsBased::TSTAGEWISE) {}

    /** @brief Deleted copy constructor and assignment. */
    TSTAGEWISE_Solver(const TSTAGEWISE_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TSTAGEWISE_Solver& operator=(const TSTAGEWISE_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TSTAGEWISE_Solver(TSTAGEWISE_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TSTAGEWISE_Solver& operator=(TSTAGEWISE_Solver&&) = delete;

    /** @brief Virtual destructor override. */
    ~TSTAGEWISE_Solver() override = default;

    // ========================================================================
    // Main Algorithm (Overrides Pure Virtual)
    // ========================================================================

    /**
     * @brief Compute T-Stagewise solution path with early stopping based on dummy
     * threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold,
     * (0 = full solution path, default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts and serialization.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ========================================================================
    // Specific Getters and Setters for T-Stagewise
    // ========================================================================

    /** @brief Get the number of coefficient removals performed. */
    std::size_t getNumRemovals() const { return num_removals_; }

    /** @brief Get the cycling ratio. */
    double getCyclingRatio() const;

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal model and path state except for input matrices.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param ar Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TLARS_Solver>(this),
            CEREAL_NVP(num_removals_)
        );
    }

    /**
     * @brief Save the current TSTAGEWISE solver/model state to the given file (binary
     * serialization).
     *
     * @details Must be class specific implemented because Cereal does not use virtual dispatch.
     * When you write oarchive(*this), cereal resolves the serialize template based on the static
     * type of *this at the call site — not the runtime type.
     *
     * @param filename Path to output file.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load TSTAGEWISE state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TSTAGEWISE_Solver object connected to X and D.
     */
    static TSTAGEWISE_Solver load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X,
                                  Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ========================================================================
    // Internal Stagewise math helpers
    // ========================================================================

    /**
     * @brief Computes the equiangular direction enforcing the stagewise monotonicity constraint.
     *
     * @details Iteratively checks if the unscaled LARS direction is currently aligned with
     * the signs of current correlations. If a violation occurs, it triggers processStagewiseNNLS()
     * to drop the violating variable and re-solves until a strictly positive active subset is
     * found.
     *
     * @return Valid equiangular direction vector w_A, or an empty vector if the active set
     * becomes empty.
     */
    Eigen::VectorXd computeStagewiseDirection();


    /**
     * @brief Performs Non-Negative Least Squares reduction on active set.
     *
     * @details Corresponds to the inner feasibility loop of Lawson-Hanson NNLS.
     * Drops the most violating constraint and downdates the Cholesky factor.
     *
     * @param directions Vector of current step directions for active coefficients.
     */
    void processStagewiseNNLS(const Eigen::Ref<const Eigen::VectorXd>& directions);
};

// ============================================================================
} /* End of namespace trex::tsolvers::linear_model::lars_based  */

#endif /* TSOLVERS_LINEAR_MODEL_LARS_BASED_TSTAGEWISE_SOLVER_HPP */
