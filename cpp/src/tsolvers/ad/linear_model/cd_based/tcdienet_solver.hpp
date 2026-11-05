// =============================================================================
//  tcdienet_solver.hpp
// =============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_CD_BASED_TCDIENET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_CD_BASED_TCDIENET_SOLVER_HPP
// =============================================================================
/**
 * @file tcdienet_solver.hpp
 *
 * @brief Header file for the Terminating CCD Informed Elastic Net (T-CDIENET)
 * solver class, extending the T-CDENET solver.
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
#include <tsolvers/ad/linear_model/cd_based/tcdenet_solver.hpp>

// =============================================================================

// Embed into trex::tsolvers::ad::linear_model::cd_based namespace
namespace trex::tsolvers::ad::linear_model::cd_based {

// =============================================================================

/**
 * @brief Structured group-penalty variant of the TCDIENET group constructor.
 *
 * @details Both variants share the layered dummy convention and O(1) hooks:
 *  - MEAN      : K = sum_m 1_m 1_m^T / p_m — the canonical IEN group-mean
 *                (group-sum) penalty. Null space: sum-balanced directions.
 *  - CENTERING : K = blockdiag(I_{p_m} - 1_m 1_m^T / p_m) — within-group
 *                centering (complete-graph Laplacian per group). Null space:
 *                group-constant directions; singleton groups unpenalized.
 *                Mathematically identical to the TCDTIKNET FOLDED path with
 *                K_GC, at O(1) instead of O(p_m) per coordinate update.
 */
// Defined with the other option enums in tsolvers/solver_options.hpp and
// re-exported here.
using trex::tsolvers::ad::GroupPenaltyVariant;

// =============================================================================

/**
 * @brief Terminating CCD Informed Elastic Net (T-CDIENET) solver with dummy
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
 *  Generalized Tikhonov penalties (arbitrary sparse PSD K = Gamma^T Gamma
 *  with selectable FOLDED / INDEPENDENT_RIDGE dummy coupling) live in the
 *  dedicated TCDTIKNET_Solver (tcdtiknet_solver.hpp) — the CCD analog of
 *  TIKNET. This class owns the structured GROUP-family
 *  specializations, which are the O(1)-per-update fast paths of that
 *  general form (FOLDED coupling by construction).
 *
 *  Degenerate cases: WITHOUT dummies, all-singleton groups reproduce
 *  TCDENET_Solver exactly (with dummies they do not: every dummy copy joins
 *  its originating variable's group and couples with it through the group
 *  sum, whereas TCDENET gives each dummy its own independent ridge — the
 *  same collapse/coupling trade-off as K = I under TCDTIKNET's FOLDED
 *  mode). A single global group is statistically degenerate and should be
 *  avoided.
 *
 *  References:
 *  -----------
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped Variable Selection
 *     and FDR Control in Genomics Research," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt.
 *     Process. (CAMSAP), 2023, pp. 466-470.
 *   H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser.
 *     B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
 *   J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
 *     Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
 *   J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
 *     Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TCDIENET_Solver: public TCDENET_Solver {

protected:

    // ==========================================================================
    // T-CDIENET specific state variables (GROUP kind)
    // ==========================================================================

    /** @brief Group id per UNIFIED column of [X D] (0..M-1; dummy layers
     *  share the group of their originating variable). */
    std::vector<int> group_of_{};

    /** @brief REAL group sizes p_m (dummy copies do not enlarge p_m). */
    std::vector<std::size_t> group_size_{};

    /** @brief Group-sum bookkeeping sigma_m = sum_{i in G_m} beta_i
     *  (unified columns; transient: rebuilt from the coefficients in
     *  ensureCdState()). Shared by GROUP and GROUP_CENTERING. */
    std::vector<double> group_sums_{};

    /** @brief Precomputed lambda2 / p_m per group (transient). */
    std::vector<double> lambda2_over_pm_{};

    /** @brief GROUP_CENTERING only: per-base folded coefficient sums
     *  s_i = beta_i + sum of variable i's dummy-copy betas (length p;
     *  transient: rebuilt in ensureCdState()). The centering gradient of
     *  unified column t with base b and group m is
     *  lambda2 * (s_b - sigma_m / p_m). */
    Eigen::VectorXd folded_s_{};

    // ==================================================================================
    // Constructor protected for inheritance
    // ==================================================================================
    /**
     * @brief Protected constructor for inheritance by derived classes.
     *
     * @param solver_type Algorithm variant as SolverTypeCdBased enum.
     */
    explicit TCDIENET_Solver(SolverTypeCdBased solver_type) :
                             TCDENET_Solver(solver_type) {}

public:
    // ==========================================================================
    // Constructors/Destructors
    // ==========================================================================

    /**
     * @brief Construct a new T-CDIENET solver with the canonical group-mean
     * penalty (T-Rex+GVS layered dummy convention; API parity with
     * TIENET_Solver / TIENETAug_Solver).
     *
     * @param X Predictor matrix (n x p).
     * @param D Dummy matrix (n x L) with L a multiple of p; dummy column j
     *        of each layer shares original variable j's group.
     * @param y Response vector (size n).
     * @param lambda2 Group-mean (L2) penalty weight (>= 0).
     * @param groups 0-based contiguous group ids in [0, M-1], length p.
     * @param options Every other input (SolverOptions); this solver reads
     *        lambda2 and group_penalty — MEAN (default, the canonical IEN
     *        group-sum penalty) or CENTERING, the within-group centering
     *        penalty K = blockdiag(I - J/p_m) with the same O(1) hook cost
     *        (see GroupPenaltyVariant) — plus the cd_* knobs. kappa_dummy
     *        is not read (the group penalty couples the dummies).
     */
    TCDIENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                   Eigen::Map<Eigen::MatrixXd>& D,
                   Eigen::Map<Eigen::VectorXd>& y,
                   const Eigen::VectorXi& groups,
                   const SolverOptions& options = SolverOptions{})
        : TCDENET_Solver(X, D, y, withUnitKappa(options),
                        SolverTypeCdBased::TCDIENET) {
        penalty_kind_ = (options.group_penalty == GroupPenaltyVariant::CENTERING)
                            ? PenaltyKind::GROUP_CENTERING
                            : PenaltyKind::GROUP;
        penalty_dvec_.resize(0);            // unused by the GROUP-family hooks
        assignGroupsLayered(groups);
    }

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TCDIENET_Solver() : TCDENET_Solver(SolverTypeCdBased::TCDIENET) {}

private:
    /** @brief The options with kappa_dummy pinned to 1 (unused by the group
     *  hooks; keeps the TCDENET base construction canonical). */
    static SolverOptions withUnitKappa(SolverOptions options) {
        options.kappa_dummy = 1.0;
        return options;
    }

public:

    /** @brief Deleted copy constructor. */
    TCDIENET_Solver(const TCDIENET_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TCDIENET_Solver& operator=(const TCDIENET_Solver&) = delete;

    /** @brief Move constructor (defaulted). */
    TCDIENET_Solver(TCDIENET_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TCDIENET_Solver& operator=(TCDIENET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TCDIENET_Solver() override = default;

    // ==========================================================================
    // Output & Accessors
    // ==========================================================================

    /** @brief Number of groups M. */
    std::size_t getNumGroups() const noexcept { return group_size_.size(); }

    /** @brief Group id per unified column. */
    const std::vector<int>& getGroupOf() const noexcept { return group_of_; }

    /** @brief Group sizes p_m. */
    const std::vector<std::size_t>& getGroupSizes() const noexcept {
        return group_size_;
    }

    /** @brief Current group sums sigma_m (valid after executeStep()). */
    const std::vector<double>& getGroupSums() const noexcept {
        return group_sums_;
    }

    /** @brief TCDENET_Solver::memoryBytes() plus the group map, the per-group
     *  sizes, sums and penalty weights and the folded sums. */
    std::size_t memoryBytes() const noexcept override;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-CDIENET solver state except for input
     * matrices. Group sums and per-group constants are transient (rebuilt in
     * ensureCdState()).
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TCDENET_Solver>(this),
            CEREAL_NVP(group_of_),
            CEREAL_NVP(group_size_)
        );
    }

    /**
     * @brief Save T-CDIENET solver/model state to file (binary serialization).
     *
     * @param filename Output file path for checkpoint.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load T-CDIENET solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TCDIENET_Solver object connected to X and ready
     *         for warm-start.
     */
    static TCDIENET_Solver load(const std::string& filename,
                               Eigen::Map<Eigen::MatrixXd>& X,
                               Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ============================================================================
    // Specialized penalty hooks (group-mean structure, LARS-IEN analog).
    // ============================================================================

    /**
     * @brief Build group_of_/group_size_ from Aug-style contiguous group ids
     * with the layered dummy convention (dummy column j of each layer joins
     * variable j's group; p_m = real sizes).
     */
    void assignGroupsLayered(const Eigen::VectorXi& groups);

    /**
     * @brief Rebuild the transient CD caches of the base plus the group-sum
     * bookkeeping sigma and the per-group constants lambda2 / p_m
     * (CENTERING additionally the per-base folded sums s).
     */
    void ensureCdState() override;

    /** @brief GROUP: lambda2 / p_m for every grouped column;
     *  GROUP_CENTERING: lambda2 * (1 - 1/p_m). */
    void configurePenaltyDiag() override;

    /** @brief GROUP: O(1) group-sum update sigma_m += delta;
     *  GROUP_CENTERING: O(1) sigma_m += delta and s_base(j) += delta. */
    void penaltyRankUpdate(std::size_t j, double delta) override;

    /** @brief GROUP: [lambda2 W beta]_j = lambda2 * sigma_m / p_m;
     *  GROUP_CENTERING: lambda2 * (s_base(j) - sigma_m / p_m). */
    double penaltyGradient(std::size_t j) const override {
        if (penalty_kind_ == PenaltyKind::GROUP) {
            const std::size_t m = static_cast<std::size_t>(group_of_[j]);
            return lambda2_over_pm_[m] * group_sums_[m];
        }
        if (penalty_kind_ == PenaltyKind::GROUP_CENTERING) {
            const std::size_t m = static_cast<std::size_t>(group_of_[j]);
            return lambda2_ *
                       folded_s_(static_cast<Eigen::Index>(foldedBase(j)))
                   - lambda2_over_pm_[m] * group_sums_[m];
        }
        return TCDLASSO_Solver::penaltyGradient(j);
    }

    /** @brief GROUP: lambda2/p_m for two columns of one group (dummies
     *  included), else 0; GROUP_CENTERING: lambda2 ([base(i)=base(j)] -
     *  [same group]/p_m). Other kinds: base. */
    double penaltyPair(std::size_t i, std::size_t j) const override;

    /** @brief out = lambda2 K_{:,A} v_A for the group kinds (group sums /
     *  folded base sums of v). Other kinds: base. */
    void penaltyMatVec(const std::vector<std::size_t>& A,
                       const Eigen::VectorXd& vA,
                       Eigen::VectorXd& out) const override;

    /** @brief Zero the group sums (and folded sums) before a rebuild. */
    void zeroPenaltyState() override;
};

// =============================================================================
} // End of namespace trex::tsolvers::ad::linear_model::cd_based

// =============================================================================
// Cereal polymorphic registration
// =============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::ad::linear_model::cd_based::TCDIENET_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_CD_BASED_TCDIENET_SOLVER_HPP */
