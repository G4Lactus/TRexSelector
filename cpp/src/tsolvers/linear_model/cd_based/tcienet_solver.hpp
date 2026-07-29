// =============================================================================
//  tcienet_solver.hpp
// =============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_CD_BASED_TCIENET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_CD_BASED_TCIENET_SOLVER_HPP
// =============================================================================
/**
 * @file tcienet_solver.hpp
 *
 * @brief Header file for the Terminating CCD Informed Elastic Net (T-CIENET)
 * solver class, extending the T-CENET solver.
 */
// =============================================================================

// std includes
#include <string>
#include <vector>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>

// tsolvers includes
#include <tsolvers/linear_model/cd_based/tcenet_solver.hpp>

// =============================================================================

// Embed into trex::tsolvers::linear_model::cd_based namespace
namespace trex::tsolvers::linear_model::cd_based {

// =============================================================================

/**
 * @brief Terminating CCD Informed Elastic Net (T-CIENET) solver with dummy
 *        variable support for FDR control.
 *
 * @details
 *  Implements the Informed Elastic Net (IEN) penalty over M disjoint groups
 *  G_1, ..., G_M of predictors (Machkour, Muma, Palomar, CAMSAP 2023):
 *
 *      F(beta) = 1/2 ||y - X beta||^2 + lambda1 ||beta||_1
 *                + (lambda2/2) sum_m (1_m^T beta)^2 / p_m
 *
 *  as a terminating pathwise coordinate descent. The rank-M group penalty
 *  W = sum_m 1_m 1_m^T / p_m is NOT materialized: mirroring the LARS-IEN
 *  design, the entire penalty state is the group-sum vector
 *  sigma_m = sum_{i in G_m} beta_i, maintained as O(M) bookkeeping. The
 *  specialized penalty hooks cost O(1) per coordinate update:
 *
 *  - penalty gradient:      [lambda2 W beta]_j = lambda2 * sigma_m / p_m,
 *  - penalty rank update:   sigma_m += delta,
 *  - penalty diagonal:      lambda2 * W_jj = lambda2 / p_m.
 *
 *  The modified correlation of the KKT/entry conditions and of the crossing
 *  predictor is c_j = x_j^T r - lambda2 * sigma_m / p_m, exactly the
 *  LARS-IEN modified correlation of the native TIENET_Solver — the two
 *  solvers compute the same solution path family from opposite algorithmic
 *  sides (exact fixed-lambda1 minimizers here, piecewise-linear homotopy
 *  there).
 *
 *  Dummy handling (T-Rex+GVS convention, identical to TIENET_Solver /
 *  TIENETAug_Solver): the dummy matrix D consists of L/p layers of p
 *  columns; dummy column j of every layer belongs to the group of original
 *  variable j and contributes to its group sum; the group size p_m remains
 *  the REAL group size.
 *
 *  Generalized Tikhonov variant: the second constructor accepts an arbitrary
 *  sparse PSD matrix K = Gamma^T Gamma over the real block instead of a
 *  group partition. K is passed as data (matrix-passing core): it enters
 *  only through the coordinate denominator, the incrementally maintained
 *  lambda2 * K * beta vector (O(nnz(K col)) per coordinate change), and the
 *  penalized KKT conditions. Dummy coupling is selectable:
 *
 *  - INDEPENDENT_RIDGE (fold_dummies = false): dummies receive a
 *    kappa_dummy * I ridge block, decoupled from K. Preserves the
 *    K = I ≡ TCENET collapse WITH dummies, but penalizes null reals
 *    differently from dummies whenever K has off-diagonals — the
 *    exchangeability deformation documented below.
 *  - FOLDED (fold_dummies = true): dummy copy t joins its originating
 *    variable base(t) = (t - p) mod p in the K-coupling — the exact
 *    generalization of the GROUP-mode layered dummy convention. The
 *    penalty state is the folded sum s in R^p (s_i = beta_i + sum of the
 *    copies' betas) with the maintained vector w = lambda2 * K * s
 *    (O(nnz(K col)) per coordinate change); gradient of unified column t
 *    is w_base(t), diagonal lambda2 * K_bb (kappa_dummy is ignored).
 *    With K = sum_m 1_m 1_m^T / p_m this reproduces the GROUP mode
 *    bit-exactly ((K s)_b = sigma_m / p_m, K_bb = 1 / p_m), and dummy/null
 *    exchangeability holds for arbitrary K.
 *
 *  The canonical group constructor is the O(1)-per-update specialization of
 *  this general form (FOLDED coupling by construction).
 *
 *  Degenerate cases: WITHOUT dummies, all-singleton groups reproduce
 *  TCENET_Solver exactly (with dummies they do not: every dummy copy joins
 *  its originating variable's group and couples with it through the group
 *  sum, whereas TCENET gives each dummy its own independent ridge — the
 *  same collapse/coupling trade-off as K = I under FOLDED). A single
 *  global group is statistically degenerate and should be avoided.
 */
class TCIENET_Solver: public TCENET_Solver {

protected:

    // ==========================================================================
    // T-CIENET specific state variables (GROUP kind)
    // ==========================================================================

    /** @brief Group id per UNIFIED column of [X D] (0..M-1; dummy layers
     *  share the group of their originating variable). Empty in the
     *  generalized Tikhonov (SPARSE) mode. */
    std::vector<int> group_of_{};

    /** @brief REAL group sizes p_m (dummy copies do not enlarge p_m). */
    std::vector<std::size_t> group_size_{};

    /** @brief Group-sum bookkeeping sigma_m = sum_{i in G_m} beta_i
     *  (transient: rebuilt from the coefficients in ensureCdState()). */
    std::vector<double> group_sums_{};

    /** @brief Precomputed lambda2 / p_m per group (transient). */
    std::vector<double> lambda2_over_pm_{};

    // ==================================================================================
    // T-CIENET specific state variables (SPARSE kind, FOLDED dummy coupling)
    // ==================================================================================

    /** @brief Dummy-coupling mode of the SPARSE kind: false =
     *  INDEPENDENT_RIDGE (kappa_dummy * I block, base machinery), true =
     *  FOLDED (dummy copies join their originating variable's K-coupling;
     *  see the file header). Always false in GROUP mode (the group hooks
     *  fold by construction). */
    bool fold_dummies_{false};

    /** @brief FOLDED coupling: maintained w = lambda2 * K * s over the
     *  REAL index space, where s_i sums the coefficients of variable i and
     *  all its dummy copies (transient: rebuilt in ensureCdState()). */
    Eigen::VectorXd folded_w_{};

    /** @brief FOLDED coupling: base variable of unified column t
     *  (identity for reals, (t - p) mod p for dummy copies). */
    std::size_t foldedBase(std::size_t t) const noexcept {
        return (t < p_original_) ? t : (t - p_original_) % p_original_;
    }

    // ==================================================================================
    // Constructor protected for inheritance
    // ==================================================================================
    /**
     * @brief Protected constructor for inheritance by derived classes.
     *
     * @param solver_type Algorithm variant as SolverTypeCdBased enum.
     */
    explicit TCIENET_Solver(SolverTypeCdBased solver_type) :
                             TCENET_Solver(solver_type) {}

public:
    // ==========================================================================
    // Constructors/Destructors
    // ==========================================================================

    /**
     * @brief Construct a new T-CIENET solver with the canonical group-mean
     * penalty (T-Rex+GVS layered dummy convention; API parity with
     * TIENET_Solver / TIENETAug_Solver).
     *
     * @param X Predictor matrix (n x p).
     * @param D Dummy matrix (n x L) with L a multiple of p; dummy column j
     *        of each layer shares original variable j's group.
     * @param y Response vector (size n).
     * @param lambda2 Group-mean (L2) penalty weight (>= 0).
     * @param groups 0-based contiguous group ids in [0, M-1], length p.
     * @param normalize If true, columns are scaled to unit L2-norm. Default true.
     * @param intercept If true, variables are centered. Default true.
     * @param verbose If true, print detailed status and diagnostics. Default false.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     */
    TCIENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                   Eigen::Map<Eigen::MatrixXd>& D,
                   Eigen::Map<Eigen::VectorXd>& y,
                   double lambda2,
                   const Eigen::VectorXi& groups,
                   bool normalize = true,
                   bool intercept = true,
                   bool verbose = false,
                   ScalingMode scaling_mode = ScalingMode::L2)
        : TCENET_Solver(X, D, y, lambda2, normalize, intercept, verbose,
                        scaling_mode, 1.0,
                        SolverTypeCdBased::TCIENET) {
        penalty_kind_ = PenaltyKind::GROUP;
        penalty_dvec_.resize(0);            // unused by the GROUP hooks
        assignGroupsLayered(groups);
    }

    /**
     * @brief Construct a new T-CIENET solver with a generalized Tikhonov
     * penalty matrix (matrix-passing variant).
     *
     * @param X Predictor matrix (n x p).
     * @param D Dummy matrix (n x num_dummies).
     * @param y Response vector (size n).
     * @param lambda2 Quadratic penalty weight (>= 0).
     * @param tikhonov_matrix PSD Tikhonov matrix K = Gamma^T Gamma (p x p,
     *        sparse) applied to the real block on the standardized scale.
     *        Dense K should be split (diagonal + low-rank) before passing;
     *        per-column nnz drives the update cost.
     * @param normalize If true, columns are scaled to unit L2-norm. Default true.
     * @param intercept If true, variables are centered. Default true.
     * @param verbose If true, print detailed status and diagnostics. Default false.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     * @param kappa_dummy Ridge weight of the dummy block under
     *        INDEPENDENT_RIDGE coupling. Default 1.0. Ignored when
     *        fold_dummies is true.
     * @param fold_dummies Dummy-coupling mode (see the file header).
     *        false (default): INDEPENDENT_RIDGE — dummies get a decoupled
     *        kappa_dummy * I ridge (preserves K = I ≡ TCENET with dummies).
     *        true: FOLDED — dummy copy t joins variable (t - p) mod p in
     *        the K-coupling (GVS layered convention; restores dummy/null
     *        exchangeability for arbitrary K). Requires D.cols() to be a
     *        multiple of X.cols().
     *
     * @note Methodological caveat (INDEPENDENT_RIDGE only): informed
     * penalties with off-diagonal K couple null reals to their neighbors'
     * coefficients while dummies stay uncoupled, deforming the dummy/null
     * exchangeability the T-Rex FDR calibration relies on; diag(K) = 1 with
     * kappa_dummy = 1 only equalizes the self-ridge. FOLDED coupling
     * removes the deformation (at the price of the K = I ≡ TCENET
     * collapse — mirroring "singleton groups + dummies != TCENET").
     */
    TCIENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                   Eigen::Map<Eigen::MatrixXd>& D,
                   Eigen::Map<Eigen::VectorXd>& y,
                   double lambda2,
                   const Eigen::SparseMatrix<double>& tikhonov_matrix,
                   bool normalize = true,
                   bool intercept = true,
                   bool verbose = false,
                   ScalingMode scaling_mode = ScalingMode::L2,
                   double kappa_dummy = 1.0,
                   bool fold_dummies = false)
        : TCENET_Solver(X, D, y, lambda2, normalize, intercept, verbose,
                        scaling_mode, kappa_dummy,
                        SolverTypeCdBased::TCIENET) {
        if (static_cast<std::size_t>(tikhonov_matrix.rows()) != p_original_ ||
            static_cast<std::size_t>(tikhonov_matrix.cols()) != p_original_) {
            throw std::invalid_argument(concatMsg(
                "TCIENET_Solver: Tikhonov matrix is ", tikhonov_matrix.rows(),
                "x", tikhonov_matrix.cols(), ", expected ",
                p_original_, "x", p_original_));
        }
        if (fold_dummies && num_dummies_ > 0 &&
            num_dummies_ % p_original_ != 0) {
            throw std::invalid_argument(concatMsg(
                "TCIENET_Solver: FOLDED dummy coupling requires D.cols() "
                "to be a multiple of X.cols() (layered dummy convention); "
                "got L = ", num_dummies_, ", p = ", p_original_));
        }
        penalty_kind_ = PenaltyKind::SPARSE;
        fold_dummies_ = fold_dummies;
        tikhonov_K_ = tikhonov_matrix;
        tikhonov_K_.makeCompressed();
    }

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TCIENET_Solver() : TCENET_Solver(SolverTypeCdBased::TCIENET) {}

    /** @brief Deleted copy constructor. */
    TCIENET_Solver(const TCIENET_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TCIENET_Solver& operator=(const TCIENET_Solver&) = delete;

    /** @brief Move constructor (defaulted). */
    TCIENET_Solver(TCIENET_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TCIENET_Solver& operator=(TCIENET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TCIENET_Solver() override = default;

    // ==========================================================================
    // Output & Accessors
    // ==========================================================================

    /** @brief Number of groups M (0 in the generalized Tikhonov mode). */
    std::size_t getNumGroups() const noexcept { return group_size_.size(); }

    /** @brief Group id per unified column (empty in Tikhonov mode). */
    const std::vector<int>& getGroupOf() const noexcept { return group_of_; }

    /** @brief Group sizes p_m. */
    const std::vector<std::size_t>& getGroupSizes() const noexcept {
        return group_size_;
    }

    /** @brief Current group sums sigma_m (valid after executeStep()). */
    const std::vector<double>& getGroupSums() const noexcept {
        return group_sums_;
    }

    /** @brief Whether the SPARSE kind runs FOLDED dummy coupling. */
    bool isFoldedDummyCoupling() const noexcept { return fold_dummies_; }

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-CIENET solver state except for input
     * matrices. Group sums and per-group constants are transient (rebuilt in
     * ensureCdState()); the Tikhonov matrix travels in the T-CCD base state.
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TCENET_Solver>(this),
            CEREAL_NVP(group_of_),
            CEREAL_NVP(group_size_),
            CEREAL_NVP(fold_dummies_)   // folded_w_ is transient (ensureCdState)
        );
    }

    /**
     * @brief Save T-CIENET solver/model state to file (binary serialization).
     *
     * @param filename Output file path for checkpoint.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load T-CIENET solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TCIENET_Solver object connected to X and ready
     *         for warm-start.
     */
    static TCIENET_Solver load(const std::string& filename,
                               Eigen::Map<Eigen::MatrixXd>& X,
                               Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ============================================================================
    // Specialized penalty hooks (group-mean structure, LARS-IEN analog).
    // In the generalized Tikhonov (SPARSE) mode every hook falls back to the
    // matrix-passing implementation of the T-CCD core.
    // ============================================================================

    /**
     * @brief Build group_of_/group_size_ from Aug-style contiguous group ids
     * with the layered dummy convention (dummy column j of each layer joins
     * variable j's group; p_m = real sizes).
     */
    void assignGroupsLayered(const Eigen::VectorXi& groups);

    /**
     * @brief Rebuild the transient CD caches of the base plus (GROUP mode)
     * the group-sum bookkeeping sigma and the per-group constants
     * lambda2 / p_m, or (SPARSE + FOLDED) the folded coupling state
     * w = lambda2 * K * s.
     */
    void ensureCdState() override;

    /** @brief GROUP: lambda2 / p_m for every grouped column;
     *  SPARSE + FOLDED: lambda2 * K_bb of the base variable for every
     *  unified column (dummies included, kappa_dummy ignored); otherwise
     *  the base diagonal (Tikhonov diag + dummy ridge). */
    void configurePenaltyDiag() override;

    /** @brief GROUP: O(1) group-sum update sigma_m += delta;
     *  SPARSE + FOLDED: O(nnz(K col base(j))) folded-w update; otherwise
     *  the base lambda2 * K * beta maintenance. */
    void penaltyRankUpdate(std::size_t j, double delta) override;

    /** @brief GROUP: [lambda2 W beta]_j = lambda2 * sigma_m / p_m;
     *  SPARSE + FOLDED: w_base(j) from the folded K-coupling state;
     *  otherwise the base maintained gradient. */
    double penaltyGradient(std::size_t j) const override {
        if (penalty_kind_ == PenaltyKind::GROUP) {
            const std::size_t m = static_cast<std::size_t>(group_of_[j]);
            return lambda2_over_pm_[m] * group_sums_[m];
        }
        if (penalty_kind_ == PenaltyKind::SPARSE && fold_dummies_) {
            return folded_w_(static_cast<Eigen::Index>(foldedBase(j)));
        }
        return TCCD_Solver::penaltyGradient(j);
    }
};

// =============================================================================
} // End of namespace trex::tsolvers::linear_model::cd_based

// =============================================================================
// Cereal polymorphic registration
// =============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::linear_model::cd_based::TCIENET_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_CD_BASED_TCIENET_SOLVER_HPP */
