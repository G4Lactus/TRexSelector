// ============================================================================
// tiknet_solver.hpp
// ============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TIKNET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TIKNET_SOLVER_HPP
// ============================================================================
/**
 * @file tiknet_solver.hpp
 *
 * @brief Terminating Tikhonov Net (T-TIKNET) solver: pathwise LARS with a
 * general sparse-K quadratic penalty and FOLDED dummy coupling, extending
 * the T-LASSO drop machinery of the T-LARS solver.
 *
 * @details
 *  Implements the generalized Tikhonov Lagrangian
 *
 *      L(beta) = 1/2 ||y - X beta||^2 + lambda1 ||beta||_1
 *                + (lambda2/2) s^T K s,        s = fold(beta),
 *
 *  for a user-supplied sparse PSD matrix K = Gamma^T Gamma (p x p) as a
 *  terminating LARS-type path algorithm (LARS-TIK). This generalizes the
 *  native LARS-IEN algorithm of TIENET_Solver from the group-mean penalty
 *  W = sum_m 1_m 1_m^T / p_m to arbitrary sparse K, mirroring the
 *  matrix-passing FOLDED mode of the CCD side (TCDTIKNET_Solver):
 *
 *  - FOLDED dummy convention (T-Rex+GVS layered): the penalty acts on the
 *    folded coefficient sums s in R^p with s_b = beta_b + sum of the betas
 *    of b's dummy copies; dummy copy t couples as its base
 *    b(t) = (t - p) mod p. Dummy/null exchangeability holds for arbitrary K.
 *  - Modified correlations  c_j = x_j^T r - lambda2 * (K s)_{b(j)} with the
 *    maintained coupling state v = K s (O(nnz(K col b)) per coefficient
 *    change; the LARS analogue of the CCD folded_w_). r stays in the
 *    observation space R^n.
 *  - Active Gram matrix G_A = X_A^T X_A + lambda2 * K-tilde_A with
 *    K-tilde_{tu} = K_{b(t), b(u)} over unified active columns, maintained
 *    through its Cholesky factor: the rank-1 update adds
 *    lambda2 * K_{b(t), b(u)} to the cross-products and
 *    lambda2 * K_{bb} to the new diagonal entry. The downdate is
 *    penalty-agnostic (Givens), so the shared T-LASSO drop machinery
 *    applies unchanged.
 *  - Joining rate of an inactive variable j along the equiangular
 *    direction:  a_j = x_j^T u + lambda2 * (K tau)_{b(j)},  where
 *    tau in R^p holds the folded direction sums
 *    tau_b = sum_{t in A, b(t) = b} w_t. The coupling term is the drift of
 *    v = K s along the step; K tau is refreshed once per step
 *    (O(nnz over active bases)).
 *  - Kahan-compensated incremental correlation updates with a periodic full
 *    refresh of s, v = K s (kahan_refresh_interval_, as in TIENET).
 *
 *  Degenerate cases: lambda2 == 0 reduces exactly to T-LASSO. With
 *  K = sum_m 1_m 1_m^T / p_m the algorithm coincides with the native
 *  TIENET_Solver (the group solver becomes the special case; accumulation
 *  orders differ, so knife-edge designs can flip borderline selections at
 *  ULP margins — the anchor tests pin wide-margin designs). K = I gives
 *  the TIENET-with-singleton-groups behavior (each variable ridged jointly
 *  with its own dummy copies through the folded sum — NOT the decoupled
 *  TENET ridge). K = K_GC = blockdiag(I - J/p_m) walks the same path
 *  family as the CCD GROUP_CENTERING mode (crossing-support equivalence,
 *  opposite algorithmic sides).
 */
// ============================================================================

// std includes
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Sparse>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>

// tsolvers includes
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::ad::linear_model::lars_based
namespace trex::tsolvers::ad::linear_model::lars_based {

// ============================================================================

/**
 * @brief Terminating Tikhonov Net (T-TIKNET) solver via the native pathwise
 * LARS-TIK algorithm with FOLDED dummy coupling.
 *
 *  References:
 *  -----------
 *   A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization Method," Soviet Math.
 *     Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl. Akad. Nauk SSSR 151 (1963), no. 3,
 *     501-504).
 *   Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov Regularization
 *     Problems in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25, 2018,
 *     doi: 10.1016/j.cam.2018.04.049.
 *   A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems,"
 *     Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article in the Special 40th
 *     Anniversary Issue).
 *   Y. Miche, M. van Heeswijk, P. Bas, O. Simula, and A. Lendasse, "TROP-ELM: A Double-Regularized ELM Using
 *     LARS and Tikhonov Regularization," Neurocomputing, vol. 74, pp. 2413-2421, 2011,
 *     doi: 10.1016/j.neucom.2010.12.042.
 *   B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
 *     no. 2, pp. 407-499, 2004.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TIKNET_Solver: public TLARS_Solver {

protected:
    // ============================================================================
    // TIK specific state variables
    // ============================================================================

    /** @brief Quadratic penalty weight lambda2 (>= 0). */
    double lambda2_{0.0};

    /** @brief Sparse PSD Tikhonov matrix K = Gamma^T Gamma (p x p) on the
     *  standardized scale; per-column nnz drives the update cost. */
    Eigen::SparseMatrix<double> tikhonov_K_{};

    /** @brief Folded coefficient sums s in R^p (s_b = beta_b + dummy-copy
     *  betas of b), maintained incrementally along the path. */
    Eigen::VectorXd folded_s_{};

    /** @brief Maintained coupling state v = K s (UNSCALED by lambda2; the
     *  modified correlation subtracts lambda2 * v_{b(j)}). */
    Eigen::VectorXd ks_{};

    /** @brief Per-step folded direction sums tau in R^p
     *  (tau_b = sum of w_t over active t with base b). */
    Eigen::VectorXd tau_{};

    /** @brief Per-step K tau (drift of v along the equiangular step). */
    Eigen::VectorXd ktau_{};

    /** @brief Dense gather workspace for one K column during the Cholesky
     *  rank-1 update (transient; lazily sized). */
    Eigen::VectorXd kcol_gather_{};

    /** @brief Base variable of unified column t (identity for reals,
     *  (t - p) mod p for dummy copies — FOLDED layered convention). */
    std::size_t foldedBase(std::size_t t) const noexcept {
        return (t < p_original_) ? t : (t - p_original_) % p_original_;
    }

public:
    // ============================================================================
    // Constructors / Destructors
    // ============================================================================

    /**
     * @brief Construct a new T-TIKNET solver with data, penalty matrix, and
     * configuration.
     *
     * @param X         Predictor matrix (n x p).
     * @param D         Dummy matrix (n x L) with L a multiple of p for
     *                  lambda2 > 0 (FOLDED layered convention: dummy column
     *                  j of each layer couples as variable j).
     * @param y         Response vector (size n).
     * @param lambda2   Quadratic penalty weight (>= 0).
     * @param tikhonov_matrix Sparse PSD Tikhonov matrix K = Gamma^T Gamma
     *                  (p x p) applied to the folded sums on the
     *                  standardized scale.
     * @param normalize If true, columns are scaled to unit L2-norm. Default true.
     * @param intercept If true, variables are centered. Default true.
     * @param verbose   If true, print detailed status. Default false.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     */
    TIKNET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  const Eigen::SparseMatrix<double>& tikhonov_matrix,
                  const SolverOptions& options = SolverOptions{});

    /**
     * @brief Default constructor for Cereal deserialization.
     */
    TIKNET_Solver() : TLARS_Solver(SolverTypeLarsBased::TIKNET) {}

    /** @brief Deleted copy constructor. */
    TIKNET_Solver(const TIKNET_Solver&) = delete;

    /** @brief Deleted copy assignment. */
    TIKNET_Solver& operator=(const TIKNET_Solver&) = delete;

    /** @brief Default move constructor. */
    TIKNET_Solver(TIKNET_Solver&&) = default;

    /** @brief Deleted move assignment. */
    TIKNET_Solver& operator=(TIKNET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TIKNET_Solver() override = default;

    // ============================================================================
    // Core Algorithm Execution
    // ============================================================================

    /**
     * @brief Execute the T-TIKNET solution path with zero-crossing removal
     * and dummy tracking (LARS-TIK with the T-LASSO drop machinery).
     *
     * @param T_stop     Number of dummies for early stopping threshold
     *                   (0 = full path).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    // ============================================================================
    // Output & Accessors
    // ============================================================================

    /** @brief Quadratic penalty weight used. */
    double getLambda2() const noexcept { return lambda2_; }

    /** @brief The sparse Tikhonov matrix K. */
    const Eigen::SparseMatrix<double>& getTikhonovK() const noexcept {
        return tikhonov_K_;
    }

    /** @brief Current folded coefficient sums s (valid after executeStep()). */
    const Eigen::VectorXd& getFoldedSums() const noexcept { return folded_s_; }

    /** @brief TLARS_Solver::memoryBytes() plus the copy of K, the folded sums,
     *  the coupling state and the direction and gather workspaces. */
    std::size_t memoryBytes() const noexcept override;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-TIKNET solver state except input
     * matrices. K passes through triplets (portable, no Eigen-sparse
     * adapter); the per-step direction sums are transient.
     *
     * @tparam Archive Cereal archive type.
     * @param archive  Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        // Sparse K passes through triplets (same convention as TCDLASSO).
        std::vector<int> K_rows, K_cols;
        std::vector<double> K_vals;
        for (int j = 0; j < tikhonov_K_.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(tikhonov_K_, j);
                 it; ++it) {
                K_rows.push_back(static_cast<int>(it.row()));
                K_cols.push_back(j);
                K_vals.push_back(it.value());
            }
        }
        std::size_t K_dim = static_cast<std::size_t>(tikhonov_K_.rows());

        archive(
            cereal::base_class<TLARS_Solver>(this),
            CEREAL_NVP(lambda2_),
            cereal::make_nvp("K_dim", K_dim),
            cereal::make_nvp("K_rows", K_rows),
            cereal::make_nvp("K_cols", K_cols),
            cereal::make_nvp("K_vals", K_vals),
            CEREAL_NVP(folded_s_),
            CEREAL_NVP(ks_)
        );

        // Rebuild sparse K from the (de)serialized triplets (no-op on save).
        if (K_dim > 0) {
            std::vector<Eigen::Triplet<double>> trip;
            trip.reserve(K_vals.size());
            for (std::size_t t = 0; t < K_vals.size(); ++t) {
                trip.emplace_back(K_rows[t], K_cols[t], K_vals[t]);
            }
            tikhonov_K_.resize(static_cast<Eigen::Index>(K_dim),
                               static_cast<Eigen::Index>(K_dim));
            tikhonov_K_.setFromTriplets(trip.begin(), trip.end());
            tikhonov_K_.makeCompressed();
        }
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
     * @return Deserialized TIKNET_Solver connected to X and D.
     */
    static TIKNET_Solver load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D);

protected:
    // ============================================================================
    // TIK-specific path machinery
    // ============================================================================

    /**
     * @brief Active set update with the TIK Cholesky rank-1 update: the
     * cross-products receive lambda2 * K_{b(t), b(u)} and the new diagonal
     * entry lambda2 * K_{bb}. Mirrors TLARS_Solver::updateActiveSet
     * otherwise (collinear rejection via the pivot guard, sign tracking,
     * dummy counting).
     *
     * @param new_vars Candidate indices for addition.
     * @return Vector of actions for this step.
     */
    std::vector<int> updateActiveSetTIK(const std::vector<std::size_t>& new_vars);

    /**
     * @brief TIK Cholesky rank-1 update for the penalized Gram matrix
     * G_A = X_A^T X_A + lambda2 * K-tilde_A (needs the entering variable's
     * index for its base, hence not an override of the column-only
     * updateR()).
     *
     * @param j_new Entering unified column index.
     * @param eps   Numerical tolerance for collinearity detection.
     * @return Updated (k+1 x k+1) upper-triangular Cholesky factor.
     */
    Eigen::MatrixXd updateRTIK(std::size_t j_new, double eps);

    /**
     * @brief Compute the folded direction sums tau_b = sum of w_t over
     * active t with base b, and refresh ktau_ = K tau
     * (O(q_k + nnz over active bases)).
     */
    void computeFoldedDirSums(const Eigen::Ref<const Eigen::VectorXd>& w_A);

    /**
     * @brief Maximal feasible step size with the TIK joining rate
     * a_j = x_j^T u + lambda2 * (K tau)_{b(j)} (the coupling term is the
     * drift of v = K s along the step).
     *
     * @param Cmax Current maximal modified correlation.
     * @param u    Equiangular predictor combination (observation space).
     * @return Pair of {gamma, projection vector a over inactives}.
     */
    std::pair<double, Eigen::VectorXd> computeStepSizeTIK(
        double Cmax,
        const Eigen::Ref<const Eigen::VectorXd>& u) const;

    /**
     * @brief Update modified correlations incrementally
     * (c_j -= gamma * a_j with Kahan compensation) or refresh them fully as
     * c_j = x_j^T r - lambda2 * (K s)_{b(j)} every kahan_refresh_interval_
     * steps (s and v = K s are rebuilt from the running coefficients
     * first). Assumes folded_s_ / ks_ are already advanced to the current
     * step.
     */
    void updateCorrelationsTIK(double gamma,
                               const Eigen::Ref<const Eigen::VectorXd>& a);

    /**
     * @brief Refresh the modified correlation of a variable removed from the
     * active set (overrides the plain-residual default of T-LASSO).
     */
    void refreshDroppedCorrelation(std::size_t dropped_var) override;
};

// ============================================================================
} /* End of namespace trex::tsolvers::ad::linear_model::lars_based */

// ============================================================================
// Cereal polymorphic registration
// ============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::ad::linear_model::lars_based::TIKNET_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_LARS_BASED_TIKNET_SOLVER_HPP */
