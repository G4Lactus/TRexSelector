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
#include <tsolvers/ad/tsolver_base.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::ad::linear_model::omp_based namespace
namespace trex::tsolvers::ad::linear_model::omp_based {

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
 *
 *  References:
 *  -----------
 *   Y. C. Pati, R. Rezaiifar, and P. S. Krishnaprasad, "Orthogonal Matching Pursuit: Recursive Function
 *     Approximation with Applications to Wavelet Decomposition," in Proc. 27th Asilomar Conf. Signals, Syst.,
 *     Comput., 1993, pp. 40-44.
 *   J. A. Tropp, "Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Inf. Theory,
 *     vol. 50, no. 10, pp. 2231-2242, 2004, doi: 10.1109/TIT.2004.834793.
 *   T. Blumensath and M. E. Davies, "On the Difference Between Orthogonal Matching Pursuit and Orthogonal
 *     Least Squares," unpublished note, draft of March 29, 2007.
 *   F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control,"
 *     in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
 */
class TOMP_Solver : public TSolver_Base {
protected:
    /** @brief Algorithm variant type. */
    SolverTypeOMPBased algo_type_{SolverTypeOMPBased::TOMP};

    /** @brief Cache of X_A^T y in active-set order for the refit in
     *  updateBetaPath(). Not serialized: a loaded solver recomputes it on
     *  its next step with the same dot products. */
    std::vector<double> xty_active_{};

    /** @brief Protected constructor for inheritance. */
    explicit TOMP_Solver(SolverTypeOMPBased type);

    /** @brief Protected data constructor for inheritance: the public
     *  constructor with the algorithm variant passed through. */
    TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                Eigen::Map<Eigen::MatrixXd>& D,
                Eigen::Map<Eigen::VectorXd>& y,
                const SolverOptions& options,
                SolverTypeOMPBased algorithm_type);

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
     * @param options Every other input (SolverOptions): preprocessing flags,
     *        tolerance and tie-breaking (the greedy family reads
     *        exch_tie_alpha / exch_tie_floor).
     */
    TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                Eigen::Map<Eigen::MatrixXd>& D,
                Eigen::Map<Eigen::VectorXd>& y,
                const SolverOptions& options = SolverOptions{});

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

    /** @brief TSolver_Base::memoryBytes() plus the X_A^T y cache of the refit. */
    std::size_t memoryBytes() const noexcept override;

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
        // Enum passes through a local int (reinterpret_cast aliasing is UB);
        // the assignment after archive() applies the loaded value and is a
        // no-op round-trip on save.
        int algo_type = static_cast<int>(algo_type_);
        archive(
            cereal::base_class<TSolver_Base>(this),
            cereal::make_nvp("algo_type", algo_type)
        );
        algo_type_ = static_cast<SolverTypeOMPBased>(algo_type);
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
} /* End of namespace trex::tsolvers::ad::linear_model::omp_based */
// =============================================================================
#endif /* TSOLVERS_LINEAR_MODEL_OMP_BASED_TOMP_SOLVER_HPP */
