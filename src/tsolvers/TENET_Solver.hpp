/**
 * @file TENET_Solver.hpp
 * @brief T-Elastic Net solver via T-LARS-EN algorithm with L2-modified Cholesky updates and dummy variables
 */

#ifndef TENET_SOLVER_HPP
#define TENET_SOLVER_HPP

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <cereal/archives/binary.hpp>

#include "TLARS_Solver.hpp"

/**
 * @brief T-ElasticNet solver via T-LARS-EN with L1 + L2 (ridge) regularization and dummy variables
 *
 * @details Implements T-ElasticNet by T-LARS-EN path algorithm with:
 *  - Dummy variable augmentation for FDR-controlled variable selection
 *  - Early stopping when T_stop dummies enter active set
 *  - L1 penalty (LASSO): Variable selection via zero-crossing removal
 *  - L2 penalty (Ridge): Gram matrix modification G_A -> G_A + λ₂I
 *  - When λ₂ = 0, reduces to T-Lasso
 *
 * References:
 *  - Zou & Hastie (2005). "Regularization and variable selection via the elastic net"
 */
class TENET_Solver: public TLARS_Solver {
protected:

    // ============================================================================
    // Elastic Net specific state variables
    // ============================================================================

    /** @brief Ridge (L2) penalty parameter (λ₂ ≥ 0) */
    double lambda2_{};

    /** @brief Scaling factor d1 for correlation decorrelation adjustment */
    double d1_{};

    /** @brief Scaling factor d2 to account for L2's decorrelation effect */
    double d2_{};

    /** @brief L1 norm of current coefficient vector */
    double l1norm_{};

    /** @brief Count of variable removals along solution path (for cycling ratio) */
    std::size_t num_removals_{0};

    /** @brief Maximum number of steps allowed in the solution path */
    std::size_t maxSteps_{};

    /** @brief Maximum number of active variables allowed */
    std::size_t max_actives_{};

public:
    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Construct a new T-ElasticNet object
     *
     * @param X Augmented design matrix [X_original | X_dummies] (RAM/MMAP, column-major, not owned)
     * @param y Response vector
     * @param num_dummies Number of dummy variables appended to X
     * @param lambda2 Ridge penalty λ₂ >= 0 (=0 : reduces to T-LASSO)
     * @param normalize If true, each column of X is scaled to unit L2-norm
     * @param intercept If true, X and y are centered (intercept fitted)
     * @param verbose If true, print status and diagnostics during execution
     */
    TENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::VectorXd>& y,
                 std::size_t num_dummies,
                 double lambda2,
                 bool normalize = true,
                 bool intercept = true,
                 bool verbose = false);


    /**
     * @brief Default constructor for serialization or deferred initialization.
     *
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TENET_Solver() : TLARS_Solver(SolverTypeLarsBased::TENET) {}

    /**
     * @brief Deleted copy constructor to avoid raw pointer ownership problems.
     */
    TENET_Solver(const TENET_Solver&) = delete;

    /**
     * @brief Deleted copy assignment operator to avoid raw pointer ownership problems.
     */
    TENET_Solver& operator=(const TENET_Solver&) = delete;

    /**
     * @brief Move constructor (default implementation).
     */
    TENET_Solver(TENET_Solver&&) = default;

    /**
     * @brief Virtual destructor (X_ pointer is not owned, so not deleted).
     *
     * @note Present for completeness and to support inheritance.
     */
    ~TENET_Solver() override = default;


    // ============================================================================
    // Core Execution Step ElasticNet
    // ============================================================================

    /**
     * @brief Execute T-ElasticNet solution path with zero-crossing detection and dummy tracking.
     *
     * @param T_stop Number of dummies for early stopping threshold (0 = full solution path,
     *               default).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Overrides base T-LARS executeStep() with variable removal logic.
     *
     * @details
     *   Extends T-LARS with Elastic Net modifications:
     *   - L2 penalty modifies Gram matrix: G_A → G_A + λ₂·I
     *   - Split equiangular vector: u = u₁ - u₂
     *   - Monitors coefficient sign changes for zero-crossing removal
     *   - Takes gamma = min(gamma_entry, gamma_sign)
     *   - Removes zero-crossing variables and downdates Cholesky
     *   - Tracks dummy entries/removals for early stopping
     *
     *   Stopping Conditions:
     *   - Active set becomes empty
     *   - Maximum correlation < 100·eps (numerical convergence)
     *   - Early stopping: count_active_dummies_ >= T_stop
     *   - Maximum steps reached
     *
     * @note Internal state is updated in-place; supports warm-starts and serialization.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;


    // ============================================================================
    // Getters
    // ============================================================================

    /** @brief Get L2 (Ridge) penalty parameter λ₂ */
    inline double getLambda2() const noexcept { return lambda2_; }

    /**
     * @brief Retrieve EN-correct coefficient vector at given path step.
     *
     * @param step Solution path index (default -1: last step).
     *
     * @return Elastic Net scaled coefficient vector
     *  (with scaling according to Zou & Hastie Lemma 1).
     */
    Eigen::VectorXd getBeta(int step = -1) const override;

    /**
     * @brief Retrieve full solution path with EN-rescaled coefficients.
     *
     * @return Matrix of shape (p, steps) with Elastic Net scaled coefficients.
     */
    Eigen::MatrixXd getBetaPath() const override;


    /** @brief Get effective penalty ratio α = λ₁/(λ₁+λ₂) at current step
     *
     * @return Mixing parameter α ∈ [0,1], where:
     *  - α = 1: Pure LASSO (λ₂ = 0)
     *  - α = 0: Pure Ridge (λ1 = 0)
     *  - 0 < α < 1: ElasticNet mixture
     *
     * @note At step k, λ₁ corresponds to lambda_[currentStep_]
     */
    double getAlpha() const;

    /**
     * @brief Get the cycling ratio (removals / additions).
     *
     * @return Ratio of removals to additions, or 0.0 if no additions yet.
     *
     * @details
     *   The cycling ratio = num_removals / num_additions indicates solution path complexity.
     *
     *   Interpretation:
     *   - ratio < 0.3: Minimal cycling, stable path
     *   - ratio 0.3-0.5: Moderate cycling, typical for correlated predictors
     *   - ratio > 0.5: Heavy cycling, potential issues:
     *     * Strong collinearity in design matrix
     *     * Numerical instability in Cholesky updates/downdates
     *     * Regularization parameters may need tuning
     *
     * @note High cycling ratios (>0.5) may suggest numerical issues or ill-conditioning.
     */
    double getCyclingRatio() const;

    /**
     * @brief Get the number of variable removals along solution path.
     *
     * @return Total count of variables removed (cycling count).
     */
    std::size_t getNumRemovals() const noexcept { return num_removals_; }

    /**
     * @brief Compute Mallows' Cp for Elastic Net using Hastie's formula.
     *
     * @return Cp vector using EN-specific formula: Cp = ((n-m-1)*RSS)/RSS_final - n + 2*df
     *
     * @details Overrides base LARS to use the formula from R's elasticnet package:
     *          Cp <- ((n - m - 1) * RSS) / rev(RSS)[1] - n + 2 * df
     *          where m is DoF and RSS_final is the last RSS value.
     */
    std::vector<double> getCp() const override;


    // ============================================================================
    // Serialization
    // ============================================================================


    /**
     * @brief Serialize Elastic Net solver state.
     *
     * @tparam Archive Cereal archive type.
     * @param archive Output/input archive object.
     *
     * @details Saves base LARS state + λ₂ + scaling factors (d₁, d₂).
     *   Call reconnect(X) after loading.
     *
     * @note X_ must be reconnected after deserialization.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        int& algo_type_int = reinterpret_cast<int&>(algo_type_);
        archive(
            // Step tracking
            CEREAL_NVP(currentStep_),
            CEREAL_NVP(maxSteps_),
            CEREAL_NVP(max_actives_),

            // Solution path
            CEREAL_NVP(betaPath_),
            CEREAL_NVP(lambda_),
            CEREAL_NVP(actions_),
            CEREAL_NVP(RSS_),
            CEREAL_NVP(R2_),
            CEREAL_NVP(DoF_),
            CEREAL_NVP(Sign_),

            // Active set
            CEREAL_NVP(actives_),
            CEREAL_NVP(inactives_),
            CEREAL_NVP(dropped_indices_),
            CEREAL_NVP(num_additions_),

            // Cholesky
            CEREAL_NVP(R_),
            CEREAL_NVP(A_A_),

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
            CEREAL_NVP(dummies_at_step_),

            // Algorithm_type
            CEREAL_NVP(algo_type_int),

            // ElasticNet-specific state
            CEREAL_NVP(lambda2_),
            CEREAL_NVP(d1_),
            CEREAL_NVP(d2_),
            CEREAL_NVP(l1norm_),
            CEREAL_NVP(num_removals_)
        );
    }

    /**
     * @brief Save the current EN solver state to file.
     *
     * @param filename Output file path.
     *
     * @note Overrides base class to ensure derived class serialize() is called.
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load T-ElasticNet solver state from file (binary serialization).
     *
     * @param filename Input file path for checkpoint.
     * @param X Reference to augmented design matrix to restore pointer connection.
     *
     * @return Deserialized TENET_Solver object connected to X and ready for warm-start.
     *
     * @note Static method usage: `auto tenet = TENET_Solver::load("checkpoint.bin", X);`
     * @note X dimensions must match original design matrix used during save().
     *
     * @throws std::runtime_error on file I/O errors or dimension mismatch.
     */
    static TENET_Solver load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X);

    /**
     * @brief Reconnect design matrix after deserialization
     *
     * @param X Design matrix (original n×p, not augmented)
     *
     * @details Overrides base class to handle augmented residual dimensions.
     *          The EN solver uses r_ with (n+p) elements, but X has n rows.
     */
    void reconnect(Eigen::Map<Eigen::MatrixXd>& X) override;

protected:

    // ============================================================================
    // Elastic Net Internals
    // ============================================================================

    /**
     * @brief Initialize correlations X^T r with EN-specific adjustments.
     *
     * @details Sets up initial correlation state accounting for L2 penalty.
     */
    void initializeCorrelations();


    /**
     * @brief Compute correlations X^T r for inactives with EN adjustments.
     *
     * @details EN-specific correlation computation accounting for L2 penalty effects.
     */
    void updateCorrelationsENET();


    /**
     * @brief Update Cholesky factor R with L2-modified Gram matrix.
     *
     * @param xnew New variable column to incorporate.
     * @param eps Numerical tolerance for collinearity/degeneracy.
     *
     * @return Updated Cholesky matrix with modified Gram: G_A + λ₂·I.
     *
     * @details Overrides base LARS to incorporate ridge penalty into active Gram matrix.
     *  The L2 penalty modifies the Gram matrix as: X_A^T X_A → X_A^T X_A + lambda2_·I
     *  requiring adjusted Cholesky updates for numerical stability.
     */
    Eigen::MatrixXd updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps) override;


    /**
     * @brief Compute data-driven component of equiangular vector (u1 = X_A w_A).
     *
     * @param w_A Direction weights for active variables.
     *
     * @return Data component u1 = X_A w_A of the equiangular direction.
     *
     * @details In EN, the equiangular direction splits into u = u1 - u2 where
     *  u1 is the standard LARS data component.
     */
    Eigen::VectorXd equiangularVectorU1(const Eigen::Ref<const Eigen::VectorXd>& w_A) const;


    /**
     * @brief Compute L2 penalty component of equiangular vector.
     *
     * @param w_A Direction weights for active variables.
     *
     * @return Penalty component u₂ accounting for ridge regularization effect.
     *
     * @details The L2 penalty introduces a correction term in the equiangular direction.
     *  The full direction is u = u₁ - u₂, where u₂ represents the shrinkage effect.
     */
    Eigen::VectorXd equiangularVectorU2(const Eigen::Ref<const Eigen::VectorXd>& w_A) const;


    /**
     * @brief Compute maximal feasible step size for EN with split equiangular vectors.
     *
     * @param Cmax Current maximal correlation (step size scaling).
     * @param u1 Data component of equiangular direction (X_A w_A).
     * @param u2 L2 penalty component of equiangular direction.
     *
     * @return Maximum safe step size (gamma) to next variable entry.
     *
     * @details EN-specific computation with split direction components u₁ and u₂.
     */
    double computeStepSizeEN(double Cmax, const Eigen::Ref<const Eigen::VectorXd>& u1, const Eigen::Ref<const Eigen::VectorXd>& u2) const;


    /**
     * @brief Compute minimum positive step (gamma) to zero crossing for any active coefficient.
     *
     * @param gamhat Step size proposal from EN logic.
     * @param drops Candidate boolean vector for variable drops.
     * @param w_A Current equiangular direction weights for actives.
     *
     * @return Smallest positive gamma (step size for zero crossing).
     */
    double computeGammaSignChange(double gamhat, std::vector<bool>& drops, const Eigen::Ref<const Eigen::VectorXd>& w_A);


    /**
     * @brief Process zero-crossing variables: downdate Cholesky and remove from active set.
     *
     * @param drops Boolean vector marking which active variables crossed zero.
     *
     * @details Removes variables with zero-crossing, downdates Cholesky factor,
     *  and records removal actions.
     */
    void processLassoDrops(std::vector<bool>& drops);

};

#endif /* End of TENET_SOLVER_HPP */
