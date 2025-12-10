#ifndef TSTEPWISE_SOLVER
#define TSTEPWISE_SOLVER

#include <algorithm>
#include <fstream>
#include <sstream>

#include "TLARS_Solver.hpp"

/**
 * @brief T-Stepwise Regression Solver (Forward Selection with Dummy Variables)
 *
 * @details Specializes the TLARS_Solver for classical stepwise (greedy)
 * regression with dummy variable augmentation for FDR-controlled variable
 * selection.
 * The algorithm takes a full step to the axis-aligned (zero-correlation)
 * solution, without coefficient drops or LASSO L1 path tracking.
 *
 * Early stopping occurs when T_stop dummy variables enter the active set.
 */
class TSTEPWISE_Solver: public TLARS_Solver {

public:

    // ==========================================================================
    // Constructors/Destructor
    // ==========================================================================

    /**
     * @brief Construct a new TSTEPWISE_Solver object
     *
     * @param X Augmented design matrix [X_original | X_dummies]
     *          (RAM/MMAP, column-major, not owned)
     * @param y Response vector
     * @param num_dummies Number of dummy variables appended to X
     * @param normalize If true, each column of X is scaled to unit L2-norm
     * @param intercept If true, X and y are centered (intercept fitted)
     * @param verbose If true, print status and diagnostics during execution
     */
    TSTEPWISE_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                     Eigen::Map<Eigen::VectorXd>& y,
                     std::size_t num_dummies,
                     bool normalize = true,
                     bool intercept = true,
                     bool verbose = false)
                : TLARS_Solver(X, y, num_dummies, normalize, intercept, verbose,
                    SolverType::TStepwise) {}

    /** @brief Default constructor for serialization or deferred initialization.
     */
    TSTEPWISE_Solver()
        : TLARS_Solver(SolverType::TStepwise) {}

    /**
     * @brief Delete copy constructor to prevent raw pointer ownership problems.
     */
    TSTEPWISE_Solver(const TSTEPWISE_Solver&) = delete;

    /**
     * @brief Delete assignment operator to avoid raw pointer ownership problems.
     */
    TSTEPWISE_Solver& operator=(const TSTEPWISE_Solver&) = delete;

    /** @brief Move constructor. */
    TSTEPWISE_Solver(TSTEPWISE_Solver&&) = default;

    /** @brief Destructor. */
    ~TSTEPWISE_Solver() override = default;


    // ==========================================================================
    // Core Execution Step T-Stepwise
    // ==========================================================================

    /**
     * @brief Overrides main T-LARS path algorithm for stepwise forward
     *        selection with FDR control.
     *
     * @details On each iteration, takes the maximum orthogonal step.
     * Stops early when T_stop dummies enter the active set
     * (if early_stop=true).
     *
     * @param T_stop Number of dummies to trigger early stopping
     *               (0: full solution path, default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;


    // ==========================================================================
    // Serialization
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
    template<class Archive>
    void serialize(Archive& archive) {
            TLARS_Solver::serialize(archive);
    }

    /**
     * @brief Load TSTEPWISE_Solver from file, automatically restoring all
     *  internal state except X_ (pointer).
     *
     * @param filename File path to load from.
     * @param X Reference to augmented design matrix to restore pointer
     *          connection.
     *
     * @return Deserialized TSTEPWISE_Solver, connected to X.
     *
     * @throws std::runtime_error On IO or deserialization failures.
     */
    static TSTEPWISE_Solver load(const std::string& filename,
                                 Eigen::Map<Eigen::MatrixXd>& X);

protected:

    // ==========================================================================
    // T-Stepwise Internals
    // ==========================================================================

    /**
     * @brief Reset sign vector to enable orthogonal (axis-aligned) steps.
     *
     * @details Internal helper to zero all signs for stepwise regression.
     */
    void zeroAllSigns();

};

#endif /* End of TSTEPWISE_SOLVER */
