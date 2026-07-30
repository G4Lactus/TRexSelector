// ============================================================================
// tienet_solver.hpp
// ============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TIENET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TIENET_SOLVER_HPP
// ============================================================================
/**
 * @file tienet_solver.hpp
 *
 * @brief Terminating Informed Elastic Net solver via the native pathwise
 * LARS-IEN algorithm (no row augmentation), extending the T-LASSO solver.
 *
 * @details
 *  Implements the Informed Elastic Net (IEN) Lagrangian
 *
 *      L(beta) = 1/2 ||y - X beta||^2 + lambda1 ||beta||_1
 *                + (lambda2/2) sum_m (1_m^T beta)^2 / p_m
 *
 *  over M disjoint variable groups G_1, ..., G_M as a terminating LARS-type
 *  path algorithm, absorbing the rank-M group penalty W = sum_m 1_m 1_m^T/p_m
 *  directly into the algorithm state (LARS-IEN):
 *
 *  - Modified correlations  c_j = x_j^T r - (lambda2/p_m) sigma_m with the
 *    group sums sigma_m = sum_{i in G_m} beta_i maintained as O(M)
 *    bookkeeping (no augmented residual, no split state, no d1/d2 scaling;
 *    r stays in the observation space R^n).
 *  - Group Gram matrix G_A = X_A^T X_A + lambda2 W_A maintained through its
 *    Cholesky factor: the rank-1 update adds the within-group coupling
 *    lambda2/p_m to the cross-products of same-group active members and to
 *    the new diagonal entry. The downdate is penalty-agnostic (Givens), so
 *    the shared T-LASSO drop machinery applies unchanged.
 *  - Joining rate of an inactive variable j in G_m along the equiangular
 *    direction:  a_j = x_j^T u + (lambda2/p_m) tau_m,  where
 *    tau_m = sum_{i in G_m ∩ A} w_i is the group direction sum. The coupling
 *    term reflects the drift of sigma_m along the step (equivalently, the
 *    penalty-row component of the augmented-system inner product); it
 *    vanishes only when G_m has no active members.
 *  - Coefficients are in the natural parameterization (no back-transform);
 *    lambda_ records Cmax per step, which equals lambda1 at the breakpoints.
 *
 *  Dummy handling (T-Rex+GVS convention, identical to TIENETAug_Solver):
 *  D consists of w = L/p layers of p columns; dummy column j of every layer
 *  belongs to the same group as original variable j and contributes to the
 *  same group sum; the group size p_m remains the REAL group size.
 *
 *  Degenerate cases: lambda2 == 0 reduces exactly to T-LASSO; all-singleton
 *  groups give the isotropic elastic net penalty (per-variable ridge)
 *  WITHOUT the Zou-Hastie augmented scaling of TENET_Solver.
 *
 *  Equivalence: the path coincides with T-LASSO run on the pure Theorem-1
 *  augmented system [X; B], [y; 0_M] WITHOUT re-normalization of the
 *  augmented columns — exactly what TIENETAug_Solver builds (both solvers
 *  standardize the DATA blocks before augmentation and leave the penalty
 *  rows raw), so the two produce identical solution paths.
 *
 * References:
 *  - Machkour, Muma, Palomar (2023). "The Informed Elastic Net for Fast
 *    Grouped Variable Selection and FDR Control in Genomics Research".
 *  - Efron, Hastie, Johnstone, Tibshirani (2004). "Least Angle Regression".
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
 * @brief Terminating Informed Elastic Net (T-IENET) solver via the native
 * pathwise LARS-IEN algorithm with dummy variables.
 */
class TIENET_Solver: public TLARS_Solver {

protected:
    // ============================================================================
    // IEN specific state variables
    // ============================================================================

    /** @brief Group-ridge parameter lambda2 (>= 0). */
    double lambda2_{0.0};

    /** @brief 0-based group id per original variable (length p). */
    Eigen::VectorXi groups_;

    /** @brief Number of disjoint groups M. */
    std::size_t M_{0};

    /** @brief Unified group id per column of [X D] (dummy layers share the
     *  group of their originating variable). */
    std::vector<int> group_of_unified_{};

    /** @brief REAL group sizes p_m (dummy copies do not enlarge p_m). */
    Eigen::VectorXd group_size_{};

    /** @brief Precomputed lambda2 / p_m per group. */
    Eigen::VectorXd l2_over_pm_{};

    /** @brief Group sums sigma_m = sum_{i in G_m} beta_i (reals + dummy
     *  copies), maintained incrementally along the path. */
    Eigen::VectorXd group_sums_{};

    /** @brief Per-step group direction sums tau_m = sum_{i in G_m ∩ A} w_i. */
    Eigen::VectorXd group_dir_sums_{};

    // ============================================================================
    // Protected constructor for inheritance
    // ============================================================================
    /**
     * @brief Protected constructor for derived classes.
     * @param solver_type Algorithm variant as SolverTypeLarsBased enum.
     */
    explicit TIENET_Solver(SolverTypeLarsBased solver_type) :
                           TLARS_Solver(solver_type) {}

public:
    // ============================================================================
    // Constructors / Destructors
    // ============================================================================

    /**
     * @brief Construct a new T-IENET solver with data, groups, and
     * configuration.
     *
     * @param X         Predictor matrix (n x p).
     * @param D         Dummy matrix (n x L) with L a multiple of p for
     *                  lambda2 > 0; dummy column j of each layer shares
     *                  original variable j's group.
     * @param y         Response vector (size n).
     * @param lambda2   Group-ridge (L2) regularization parameter (>= 0).
     * @param groups    0-based contiguous group ids in [0, M-1], length p.
     * @param normalize If true, columns are scaled to unit L2-norm. Default true.
     * @param intercept If true, variables are centered. Default true.
     * @param verbose   If true, print detailed status. Default false.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     */
    TIENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  double lambda2,
                  const Eigen::VectorXi& groups,
                  bool normalize = true,
                  bool intercept = true,
                  bool verbose = false,
                  ScalingMode scaling_mode = ScalingMode::L2);

    /**
     * @brief Default constructor for Cereal deserialization.
     */
    TIENET_Solver() : TLARS_Solver(SolverTypeLarsBased::TIENET) {}

    /** @brief Deleted copy constructor. */
    TIENET_Solver(const TIENET_Solver&) = delete;

    /** @brief Deleted copy assignment. */
    TIENET_Solver& operator=(const TIENET_Solver&) = delete;

    /** @brief Default move constructor. */
    TIENET_Solver(TIENET_Solver&&) = default;

    /** @brief Deleted move assignment. */
    TIENET_Solver& operator=(TIENET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TIENET_Solver() override = default;

    // ============================================================================
    // Core Algorithm Execution
    // ============================================================================

    /**
     * @brief Execute the T-IENET solution path with zero-crossing removal and
     * dummy tracking (LARS-IEN with the T-LASSO drop machinery).
     *
     * @param T_stop     Number of dummies for early stopping threshold
     *                   (0 = full path).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ============================================================================
    // Output & Accessors
    // ============================================================================

    /** @brief Group-ridge parameter used. */
    double getLambda2() const noexcept { return lambda2_; }

    /** @brief 0-based group id per original variable (length p). */
    const Eigen::VectorXi& getGroups() const noexcept { return groups_; }

    /** @brief Number of disjoint groups M. */
    std::size_t getNumGroups() const noexcept { return M_; }

    /** @brief Current group sums sigma_m (valid after executeStep()). */
    const Eigen::VectorXd& getGroupSums() const noexcept { return group_sums_; }

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-IENET solver state except input
     * matrices.
     *
     * @tparam Archive Cereal archive type.
     * @param archive  Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TLARS_Solver>(this),
            CEREAL_NVP(lambda2_),
            CEREAL_NVP(groups_),
            CEREAL_NVP(M_),
            CEREAL_NVP(group_of_unified_),
            CEREAL_NVP(group_size_),
            CEREAL_NVP(l2_over_pm_),
            CEREAL_NVP(group_sums_),
            CEREAL_NVP(group_dir_sums_)
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
     * @return Deserialized TIENET_Solver connected to X and D.
     */
    static TIENET_Solver load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D);

protected:
    // ============================================================================
    // IEN-specific path machinery
    // ============================================================================

    /**
     * @brief Active set update with the IEN Cholesky rank-1 update: the
     * cross-products of same-group active members and the new diagonal entry
     * receive the within-group coupling lambda2 / p_m. Mirrors
     * TLARS_Solver::updateActiveSet otherwise (collinear rejection via the
     * pivot guard, sign tracking, dummy counting).
     *
     * @param new_vars Candidate indices for addition.
     * @return Vector of actions for this step.
     */
    std::vector<int> updateActiveSetIEN(const std::vector<std::size_t>& new_vars);

    /**
     * @brief IEN Cholesky rank-1 update for the group Gram matrix
     * G_A = X_A^T X_A + lambda2 W_A (needs the entering variable's index for
     * its group, hence not an override of the column-only updateR()).
     *
     * @param j_new Entering unified column index.
     * @param eps   Numerical tolerance for collinearity detection.
     * @return Updated (k+1 x k+1) upper-triangular Cholesky factor.
     */
    Eigen::MatrixXd updateRIEN(std::size_t j_new, double eps);

    /**
     * @brief Compute the group direction sums tau_m = sum_{i in G_m ∩ A} w_i
     * for the current step (O(q_k)).
     */
    void computeGroupDirSums(const Eigen::Ref<const Eigen::VectorXd>& w_A);

    /**
     * @brief Maximal feasible step size with the IEN joining rate
     * a_j = x_j^T u + (lambda2/p_m) tau_m (the group-coupling term reflects
     * the drift of sigma_m along the step).
     *
     * @param Cmax Current maximal modified correlation.
     * @param u    Equiangular predictor combination (observation space).
     * @return Pair of {gamma, projection vector a over inactives}.
     */
    std::pair<double, Eigen::VectorXd> computeStepSizeIEN(
        double Cmax,
        const Eigen::Ref<const Eigen::VectorXd>& u) const;

    /**
     * @brief Update modified correlations incrementally
     * (c_j -= gamma * a_j with Kahan compensation) or refresh them fully as
     * c_j = x_j^T r - (lambda2/p_m) sigma_m every kahan_refresh_interval_
     * steps. Assumes group_sums_ are already advanced to the current step.
     */
    void updateCorrelationsIEN(double gamma,
                               const Eigen::Ref<const Eigen::VectorXd>& a);

    /**
     * @brief Refresh the modified correlation of a variable removed from the
     * active set (overrides the plain-residual default of T-LASSO).
     */
    void refreshDroppedCorrelation(std::size_t dropped_var) override;
};

// ============================================================================
} /* End of namespace trex::tsolvers::linear_model::lars_based */

// ============================================================================
// Cereal polymorphic registration
// ============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::linear_model::lars_based::TIENET_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_LARS_BASED_TIENET_SOLVER_HPP */
