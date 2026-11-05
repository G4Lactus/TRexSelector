// =============================================================================
//  tcdtiknet_solver.hpp
// =============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_CD_BASED_TCDTIKNET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_CD_BASED_TCDTIKNET_SOLVER_HPP
// =============================================================================
/**
 * @file tcdtiknet_solver.hpp
 *
 * @brief Header file for the Terminating CCD Tikhonov Net (T-CDTIKNET)
 * solver class, extending the T-CDENET solver.
 */
// =============================================================================

// std includes
#include <string>

// Eigen includes
#include <Eigen/Sparse>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdenet_solver.hpp>
#include <tsolvers/ad/solver_utils/tikhonov_validation.hpp>

// =============================================================================

// Embed into trex::tsolvers::ad::linear_model::cd_based namespace
namespace trex::tsolvers::ad::linear_model::cd_based {

// =============================================================================

/**
 * @brief Terminating CCD Tikhonov Net (T-CDTIKNET) solver with dummy
 *        variable support for FDR control.
 *
 * @details
 *  Implements the generalized Tikhonov penalty
 *
 *      F(beta) = 1/2 ||y - X beta||^2 + lambda1 ||beta||_1
 *                + (lambda2/2) s^T K s,        s = fold(beta),
 *
 *  for an arbitrary user-supplied sparse PSD matrix K = Gamma^T Gamma
 *  (p x p) over the real block as a terminating pathwise coordinate
 *  descent — the CCD analog of the native LARS homotopy TIKNET_Solver.
 *  K is passed as data (matrix-passing T-CDLASSO
 *  core): it enters only through the coordinate denominator, the
 *  incrementally maintained lambda2 * K * beta vector (O(nnz(K col)) per
 *  coordinate change), and the penalized KKT conditions. Dummy coupling is
 *  selectable:
 *
 *  - INDEPENDENT_RIDGE (fold_dummies = false): dummies receive a
 *    kappa_dummy * I ridge block, decoupled from K. Preserves the
 *    K = I ≡ TCDENET collapse WITH dummies, but penalizes null reals
 *    differently from dummies whenever K has off-diagonals — the
 *    exchangeability deformation documented below.
 *  - FOLDED (fold_dummies = true): dummy copy t joins its originating
 *    variable base(t) = (t - p) mod p in the K-coupling — the exact
 *    generalization of the TCDIENET GROUP-mode layered dummy convention.
 *    The penalty state is the folded sum s in R^p (s_i = beta_i + sum of
 *    the copies' betas) with the maintained vector w = lambda2 * K * s
 *    (O(nnz(K col)) per coordinate change); gradient of unified column t
 *    is w_base(t), diagonal lambda2 * K_bb (kappa_dummy is ignored).
 *    With K = sum_m 1_m 1_m^T / p_m this reproduces the TCDIENET GROUP
 *    mode mathematically exactly ((K s)_b = sigma_m / p_m, K_bb = 1 / p_m)
 *    — NOT bit-exactly: the two solvers accumulate in different
 *    floating-point orders (FOLDED maintains w += lambda2*delta*K.col(b)
 *    per update, GROUP accumulates sigma_m += delta and multiplies once),
 *    so knife-edge designs can flip borderline selections at ULP margins
 *    (the anchor tests pin wide-margin designs). Dummy/null
 *    exchangeability holds for arbitrary K.
 *
 *  Methodological caveat (INDEPENDENT_RIDGE only): informed penalties with
 *  off-diagonal K couple null reals to their neighbors' coefficients while
 *  dummies stay uncoupled, deforming the dummy/null exchangeability the
 *  T-Rex FDR calibration relies on; diag(K) = 1 with kappa_dummy = 1 only
 *  equalizes the self-ridge. FOLDED coupling removes the deformation (at
 *  the price of the K = I ≡ TCDENET collapse — mirroring the TCDIENET
 *  "singleton groups + dummies != TCDENET" trade-off).
 *
 *  Structured special cases: K = sum_m 1_m 1_m^T / p_m (group-mean IEN)
 *  and K_GC = blockdiag(I - J/p_m) (within-group centering) are served at
 *  O(1) per coordinate update by the specialized TCDIENET_Solver hooks —
 *  prefer TCDIENET whenever the penalty is a group structure.
 *
 *  References:
 *  -----------
 *   A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization Method," Soviet Math.
 *     Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl. Akad. Nauk SSSR 151 (1963), no. 3,
 *     501-504).
 *   Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov Regularization
 *     Problems in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25, 2018,
 *     doi: 10.1016/j.cam.2018.04.049.
 *   J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
 *     Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
 *   J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
 *     Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TCDTIKNET_Solver: public TCDENET_Solver {

protected:

    // ==========================================================================
    // T-CDTIKNET specific state variables (FOLDED dummy coupling)
    // ==========================================================================

    /** @brief Dummy-coupling mode: false = INDEPENDENT_RIDGE
     *  (kappa_dummy * I block, base matrix-passing machinery), true =
     *  FOLDED (dummy copies join their originating variable's K-coupling;
     *  see the file header). */
    bool fold_dummies_{false};

    /** @brief FOLDED coupling: maintained w = lambda2 * K * s over the
     *  REAL index space, where s_i sums the coefficients of variable i and
     *  all its dummy copies (transient: rebuilt in ensureCdState()). */
    Eigen::VectorXd folded_w_{};

public:
    // ==========================================================================
    // Constructors/Destructors
    // ==========================================================================

    /**
     * @brief Construct a new T-CDTIKNET solver with a generalized Tikhonov
     * penalty matrix.
     *
     * @param X Predictor matrix (n x p).
     * @param D Dummy matrix (n x num_dummies).
     * @param y Response vector (size n).
     * @param lambda2 Quadratic penalty weight (>= 0).
     * @param tikhonov_matrix PSD Tikhonov matrix K = Gamma^T Gamma (p x p,
     *        sparse) applied to the real block on the standardized scale.
     *        Dense K should be split (diagonal + low-rank) before passing;
     *        per-column nnz drives the update cost.
     * @param options Every other input (SolverOptions); this solver reads
     *        lambda2, kappa_dummy (ridge weight of the dummy block under
     *        INDEPENDENT_RIDGE coupling; ignored when fold_dummies is true)
     *        and fold_dummies — the dummy-coupling mode (see the file
     *        header): false (default) INDEPENDENT_RIDGE, dummies get a
     *        decoupled kappa_dummy * I ridge (preserves K = I ≡ TCDENET with
     *        dummies); true FOLDED, dummy copy t joins variable (t - p) mod p
     *        in the K-coupling (GVS layered convention; restores dummy/null
     *        exchangeability for arbitrary K; requires D.cols() to be a
     *        multiple of X.cols()) — plus the cd_* knobs.
     */
    TCDTIKNET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                     Eigen::Map<Eigen::MatrixXd>& D,
                     Eigen::Map<Eigen::VectorXd>& y,
                     const Eigen::SparseMatrix<double>& tikhonov_matrix,
                     const SolverOptions& options = SolverOptions{})
        : TCDENET_Solver(X, D, y, options, SolverTypeCdBased::TCDTIKNET) {
        const bool fold_dummies = options.fold_dummies;
        if (static_cast<std::size_t>(tikhonov_matrix.rows()) != p_original_ ||
            static_cast<std::size_t>(tikhonov_matrix.cols()) != p_original_) {
            throw std::invalid_argument(concatMsg(
                "TCDTIKNET_Solver: Tikhonov matrix is ", tikhonov_matrix.rows(),
                "x", tikhonov_matrix.cols(), ", expected ",
                p_original_, "x", p_original_));
        }
        if (fold_dummies && num_dummies_ > 0 &&
            num_dummies_ % p_original_ != 0) {
            throw std::invalid_argument(concatMsg(
                "TCDTIKNET_Solver: FOLDED dummy coupling requires D.cols() "
                "to be a multiple of X.cols() (layered dummy convention); "
                "got L = ", num_dummies_, ", p = ", p_original_));
        }
        // Finiteness, symmetry, PSD within tolerance (shared trio helper).
        solver_utils::tikhonov::validateTikhonovK(tikhonov_matrix,
                                                  "TCDTIKNET_Solver");
        penalty_kind_ = PenaltyKind::SPARSE;
        fold_dummies_ = fold_dummies;
        tikhonov_K_ = tikhonov_matrix;
        tikhonov_K_.makeCompressed();
    }

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TCDTIKNET_Solver() : TCDENET_Solver(SolverTypeCdBased::TCDTIKNET) {}

    /** @brief Deleted copy constructor. */
    TCDTIKNET_Solver(const TCDTIKNET_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TCDTIKNET_Solver& operator=(const TCDTIKNET_Solver&) = delete;

    /** @brief Move constructor (defaulted). */
    TCDTIKNET_Solver(TCDTIKNET_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TCDTIKNET_Solver& operator=(TCDTIKNET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TCDTIKNET_Solver() override = default;

    // ==========================================================================
    // Output & Accessors
    // ==========================================================================

    /** @brief Whether the solver runs FOLDED dummy coupling. */
    bool isFoldedDummyCoupling() const noexcept { return fold_dummies_; }

    /** @brief TCDENET_Solver::memoryBytes() (the copy of K included) plus the
     *  folded coupling state. */
    std::size_t memoryBytes() const noexcept override;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-CDTIKNET solver state except for input
     * matrices. The folded coupling state is transient (rebuilt in
     * ensureCdState()); the Tikhonov matrix travels in the T-CDLASSO base state.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TCDENET_Solver>(this),
            CEREAL_NVP(fold_dummies_)   // folded_w_ is transient (ensureCdState)
        );
    }

    /**
     * @brief Save T-CDTIKNET solver/model state to file (binary serialization).
     *
     * @param filename Output file path for checkpoint.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load T-CDTIKNET solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TCDTIKNET_Solver object connected to X and ready
     *         for warm-start.
     */
    static TCDTIKNET_Solver load(const std::string& filename,
                                 Eigen::Map<Eigen::MatrixXd>& X,
                                 Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ============================================================================
    // Specialized penalty hooks (FOLDED dummy coupling).
    // In the INDEPENDENT_RIDGE mode every hook falls back to the
    // matrix-passing implementation of the T-CDLASSO core.
    // ============================================================================

    /**
     * @brief Rebuild the transient CD caches of the base plus (FOLDED) the
     * folded coupling state w = lambda2 * K * s.
     */
    void ensureCdState() override;

    /** @brief FOLDED: lambda2 * K_bb of the base variable for every
     *  unified column (dummies included, kappa_dummy ignored); otherwise
     *  the base diagonal (Tikhonov diag + dummy ridge). */
    void configurePenaltyDiag() override;

    /** @brief FOLDED: O(nnz(K col base(j))) folded-w update; otherwise
     *  the base lambda2 * K * beta maintenance. */
    void penaltyRankUpdate(std::size_t j, double delta) override;

    /** @brief FOLDED: w_base(j) from the folded K-coupling state;
     *  otherwise the base maintained gradient. */
    double penaltyGradient(std::size_t j) const override {
        if (fold_dummies_) {
            return folded_w_(static_cast<Eigen::Index>(foldedBase(j)));
        }
        return TCDLASSO_Solver::penaltyGradient(j);
    }

    /** @brief FOLDED: lambda2 * K(base(i), base(j)); otherwise the base
     *  Tikhonov pair (real block) / dummy ridge. */
    double penaltyPair(std::size_t i, std::size_t j) const override;

    /** @brief FOLDED: out_j = [lambda2 K s(v)]_base(j) with the folded base
     *  sums s(v) of v over A; otherwise the base mat-vec. */
    void penaltyMatVec(const std::vector<std::size_t>& A,
                       const Eigen::VectorXd& vA,
                       Eigen::VectorXd& out) const override;

    /** @brief Zero the folded coupling state before a rebuild. */
    void zeroPenaltyState() override;
};

// =============================================================================
} // End of namespace trex::tsolvers::ad::linear_model::cd_based

// =============================================================================
// Cereal polymorphic registration
// =============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::ad::linear_model::cd_based::TCDTIKNET_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_CD_BASED_TCDTIKNET_SOLVER_HPP */
