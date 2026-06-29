// ============================================================================
// tenet_solver.hpp
// ============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TENET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TENET_SOLVER_HPP
// ============================================================================
/**
 * @file tenet_solver.hpp
 *
 * @brief T-Elastic Net solver via T-LARS-EN algorithm with L2-modified
 * Cholesky updates and dummy variables
 */
// ============================================================================

// std includes
#include <string>
#include <vector>

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
 * @brief T-ElasticNet solver via T-LARS-EN with L1 + L2 (ridge)
 * regularization and dummy variables.
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
    double lambda2_{0.0}; ///< L2 regularization parameter
    double d1_{0.0};      ///< Derived parameter: sqrt(lambda2)
    double d2_{0.0};      ///< Derived parameter: 1 / sqrt(1 + lambda2)
    double l1norm_{0.0};  ///< L1 norm of active coefficients (scaled by d2)
    std::size_t num_removals_{0}; ///< Monotonic counter of variable removals

    // ============================================================================
    // Protected constructor for inheritance
    // ============================================================================
    /**
     * @brief Protected constructor for derived classes.
     * @param solver_type Algorithm variant as SolverTypeLarsBased enum.
     */
    explicit TENET_Solver(SolverTypeLarsBased solver_type) :
                          TLARS_Solver(solver_type) {}

public:
    // ============================================================================
    // Constructors / Destructors
    // ============================================================================

    /**
     * @brief Construct a new T-ElasticNet solver with data and configuration.
     *
     * @param X         Predictor matrix (n x p).
     * @param D         Dummy matrix (n x num_dummies).
     * @param y         Response vector (size n).
     * @param lambda2   Ridge (L2) regularization parameter (>= 0).
     * @param normalize If true, columns are scaled to unit L2-norm. Default true.
     * @param intercept If true, variables are centered. Default true.
     * @param verbose   If true, print detailed status. Default false.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     */
    TENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::MatrixXd>& D,
                 Eigen::Map<Eigen::VectorXd>& y,
                 double lambda2,
                 bool normalize = true,
                 bool intercept = true,
                 bool verbose = false,
                 ScalingMode scaling_mode = ScalingMode::L2);

    /**
     * @brief Default constructor for Cereal deserialization.
     */
    TENET_Solver() : TLARS_Solver(SolverTypeLarsBased::TENET) {}

    /** @brief Deleted copy constructor. */
    TENET_Solver(const TENET_Solver&) = delete;

    /** @brief Deleted copy assignment. */
    TENET_Solver& operator=(const TENET_Solver&) = delete;

    /** @brief Default move constructor. */
    TENET_Solver(TENET_Solver&&) = default;

    /** @brief Deleted move assignment. */
    TENET_Solver& operator=(TENET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TENET_Solver() override = default;

    // ============================================================================
    // Core Algorithm Execution
    // ============================================================================

    /**
     * @brief Execute T-ElasticNet solution path.
     *
     * @param T_stop    Number of dummies for early stopping threshold (0 = full path).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ============================================================================
    // Output & Accessors
    // ============================================================================

    /**
     * @brief Get total number of variable removals (negative actions).
     * @return Monotonic count of all variable drops throughout solution path.
     */
    std::size_t getNumRemovals() const noexcept { return num_removals_; }

    /**
     * @brief Get cycling ratio (removals / additions).
     * @return Ratio, or 0.0 if no additions yet.
     */
    double getCyclingRatio() const;

    /**
     * @brief Compute Mallows' Cp for each step of the solution path.
     * @return Vector of Cp values (NaN for steps where it is undefined).
     */
    std::vector<double> getCp() const override;

    /**
     * @brief Get EN-corrected coefficient vector at a given path step.
     * @param step Path step index (-1 = last step).
     * @return Coefficient vector with EN scaling removed.
     */
    Eigen::VectorXd getBeta(int step = -1) const override;

    /**
     * @brief Get full EN-corrected coefficient path matrix.
     * @return Matrix of coefficients (p_total x steps) with EN scaling removed.
     */
    Eigen::MatrixXd getBetaPath() const override;

    /**
     * @brief Get elastic net mixing parameter alpha = lambda1 / (lambda1 + lambda2).
     * @return Mixing parameter in [0, 1], or 1.0 if no regularization.
     */
    double getAlpha() const;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-ElasticNet solver state except input matrices.
     * @tparam Archive Cereal archive type.
     * @param archive  Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TLARS_Solver>(this),
            CEREAL_NVP(lambda2_),
            CEREAL_NVP(d1_),
            CEREAL_NVP(d2_),
            CEREAL_NVP(l1norm_),
            CEREAL_NVP(num_removals_)
        );
    }

    /**
     * @brief Save solver state to file (binary serialization).
     * @param filename Output file path.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load solver state from file and reconnect to data matrices.
     * @param filename Path to input file.
     * @param X        Map to feature matrix.
     * @param D        Map to dummy matrix.
     * @return Deserialized TENET_Solver connected to X and D.
     */
    static TENET_Solver load(const std::string& filename,
                             Eigen::Map<Eigen::MatrixXd>& X,
                             Eigen::Map<Eigen::MatrixXd>& D);

    /**
     * @brief Reconnect solver to data matrices after deserialization.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     */
    void reconnect(Eigen::Map<Eigen::MatrixXd>& X,
                   Eigen::Map<Eigen::MatrixXd>& D) override;

protected:
    // ============================================================================
    // EN-specific Cholesky & Correlation Updates
    // ============================================================================

    /**
     * @brief Initialize correlations for the augmented EN residuals.
     */
    void initializeCorrelations();

    /**
     * @brief Update correlations using the EN-specific formula after a step.
     */
    void updateCorrelationsENET();

    /**
     * @brief Update Cholesky factor R with EN Gram matrix scaling.
     * @param xnew New column entering the active set.
     * @param eps  Numerical tolerance for rank detection.
     * @return Updated Cholesky factor.
     */
    Eigen::MatrixXd updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew,
                            double eps) override;

private:
    // ============================================================================
    // EN-specific Internal Helpers
    // ============================================================================

    /**
     * @brief Validate that lambda2 is non-negative.
     * @param x Candidate ridge penalty.
     * @return x if valid.
     * @throws std::invalid_argument if x < 0.
     */
    double validateLambda2(double x);

    /**
     * @brief Compute observation-space equiangular direction for EN.
     * @param w_A Equiangular direction weights for active set.
     * @return u1 vector (size n).
     */
    Eigen::VectorXd equiangularVectorU1(
        const Eigen::Ref<const Eigen::VectorXd>& w_A) const;

    /**
     * @brief Compute penalty-space equiangular direction for EN.
     * @param w_A Equiangular direction weights for active set.
     * @return u2 vector (size p_total).
     */
    Eigen::VectorXd equiangularVectorU2(
        const Eigen::Ref<const Eigen::VectorXd>& w_A) const;

    /**
     * @brief Compute EN step size (joining time for next inactive variable).
     * @param Cmax Maximum absolute correlation.
     * @param u1   Observation-space equiangular vector.
     * @param u2   Penalty-space equiangular vector.
     * @return Minimum positive step size.
     */
    double computeStepSizeEN(double Cmax,
                             const Eigen::Ref<const Eigen::VectorXd>& u1,
                             const Eigen::Ref<const Eigen::VectorXd>& u2) const;

    /**
     * @brief Compute minimum positive step size to coefficient zero-crossing.
     * @param gamhat  Step size from EN joining-time logic.
     * @param drops   Output: marks which active variables cross zero at gamhat.
     * @param w_A     Equiangular direction weights for active set.
     * @return Minimum crossing step size (or gamhat if no crossings).
     */
    double computeGammaSignChange(double gamhat,
                                  std::vector<bool>& drops,
                                  const Eigen::Ref<const Eigen::VectorXd>& w_A);

    /**
     * @brief Remove zero-crossing variables from active set and downdate R.
     * @param drops Boolean vector marking which active variables to drop.
     */
    void processLassoDrops(std::vector<bool>& drops);

// ============================================================================
}; /* End of class TENET_Solver */

// ============================================================================
} /* End of namespace trex::tsolvers::linear_model::lars_based */
// ============================================================================
#endif // TSOLVERS_LINEAR_MODEL_LARS_BASED_TENET_SOLVER_HPP
