// ============================================================================
// tienet_aug_solver.hpp
// ============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TIENET_AUG_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TIENET_AUG_SOLVER_HPP
// ============================================================================
/**
 * @file tienet_aug_solver.hpp
 *
 * @brief Terminating Informed Elastic Net via row augmentation
 *        (TIENETAug_Solver): thin wrapper around TLASSO_Solver / TLARS_Solver
 *        on a group-informed row-augmented system.
 *
 * @details
 *  Implements the Informed Elastic Net (IEN) of Machkour, Muma, Palomar,
 *  "The Informed Elastic Net for Fast Grouped Variable Selection and FDR
 *  Control in Genomics Research", CAMSAP 2023, as a terminating forward
 *  selector for the T-Rex framework.
 *
 *  The IEN Lagrangian
 *
 *      L(beta) = ||y - X beta||^2 + lambda1 ||beta||_1
 *                + lambda2 * sum_m (1_m^T beta)^2 / p_m
 *
 *  is cast as a Lasso-type problem (Theorem 1 of the paper) on the
 *  row-augmented system
 *
 *      X_aug = [ X ]   (n rows: data)
 *              [ B ]   (M rows: group-ridge block),
 *      B(m, j) = sqrt(lambda2) / sqrt(p_m)  if variable j is in group m,
 *                0                          otherwise,
 *      y_aug = [ y ; 0_M ],
 *
 *  where M is the number of disjoint variable groups and p_m the size of
 *  group m. Unlike the Zou-Hastie elastic-net augmentation (TENETAug_Solver),
 *  the IEN appends only M << p rows and carries NO global 1/sqrt(1+lambda2)
 *  factor.
 *
 *  Dummy handling (T-Rex convention): the dummy matrix D consists of
 *  w = L / p layers of p columns each, where dummy column j of every layer
 *  belongs to the same group as original variable j (the T-Rex+GVS dummy
 *  generator draws cluster-aware MVN dummies with exactly this layout). The
 *  augmented dummy block therefore tiles B once per layer:
 *
 *      D_aug = [ D               ]   (n rows)
 *              [ B  B  ...  B    ]   (M rows, w copies).
 *
 *  Preprocessing (family convention): the DATA blocks X and D are
 *  standardized BEFORE augmentation (column centering and unit-L2 / unit-SD
 *  scaling per the caller's intercept/normalize flags and ScalingMode; y is
 *  centered), the group-ridge rows are appended afterwards, and the
 *  augmented columns are NOT re-normalized. The inner solver then runs with
 *  `normalize = false, intercept = false`. This makes the wrapper exactly
 *  equivalent to the native pathwise TIENET_Solver. The augmented copies are
 *  owned, so memoryBytes() grows with 8 (n + M)(p + L + 1) bytes (M = 0 at
 *  lambda2 = 0).
 *
 *  NOTE (R parity): the R reference (lm_dummy.R, GVS_type == "IEN") instead
 *  rescales the AUGMENTED system (`X_Dummy <- scale(X_Dummy)`; centered
 *  response over n + M entries). That convention divides data and penalty
 *  rows by sqrt(||x_j||^2 + lambda2/p_m) — a per-column reweighting of
 *  lambda1 that deviates from the IEN Lagrangian for heterogeneous group
 *  sizes, and from raw-scale inputs shrinks the effective lambda2 by the
 *  data-column norms. It also centers the B rows (structural zeros become
 *  -mean, weakly coupling all groups). This wrapper deliberately uses the
 *  exact pre-augmentation convention instead.
 *
 *  Degenerate case lambda2 == 0: the group penalty vanishes and the solver
 *  collapses to a plain TLASSO / TLARS on the ORIGINAL (X, D, y), forwarding
 *  the caller's `normalize` / `intercept` flags (mirror of TENETAug_Solver).
 *
 *  @note This wrapper is distinct from the reserved path-wise TIENET solver
 *        (tienet_solver.hpp), which would absorb the group penalty into the
 *        Gram matrix like TENET_Solver does for the EN.
 */
// ============================================================================

// std includes
#include <memory>
#include <string>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/ad/tsolver_base.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>

// ============================================================================

namespace trex::tsolvers::ad::linear_model::lars_based {

// ============================================================================

/**
 * @brief Terminating Informed Elastic Net (T-IENET) solver via row augmentation, wrapping TLASSO_Solver.
 *
 *  References:
 *  -----------
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped Variable Selection
 *     and FDR Control in Genomics Research," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt.
 *     Process. (CAMSAP), 2023, pp. 466-470.
 *   H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser.
 *     B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
 *   B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
 *     no. 2, pp. 407-499, 2004.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TIENETAug_Solver : public TSolver_Base {

    // ========================================================================
    // Owned augmented data
    // ========================================================================

    /** @brief Group-ridge regularisation parameter (>= 0). */
    double lambda2_{0.0};

    /** @brief 0-based group id per original variable (length p). */
    Eigen::VectorXi groups_;

    /** @brief Number of disjoint groups. */
    std::size_t M_{0};

    /** @brief Augmented, normalised X ((n+M) x p) */
    Eigen::MatrixXd X_aug_owned_;

    /** @brief Augmented, normalised D ((n+M) x L) */
    Eigen::MatrixXd D_aug_owned_;

    /** @brief Augmented, centred y (n+M) */
    Eigen::VectorXd y_aug_owned_;

    // Maps pointing into the above buffers (heap-allocated so that
    // their addresses are stable after the constructor runs)
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_aug_map_;
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> D_aug_map_;
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_aug_map_;

    // ========================================================================
    // Inner forward-selection solver
    // ========================================================================
    /** @brief Inner solver run on the augmented system. Held by base pointer so
     *  it can be either a TLASSO_Solver (LARS-LASSO; can drop variables — the
     *  default) or a plain TLARS_Solver (pure LARS; never drops — matches R's
     *  trex(type="lar")). */
    std::unique_ptr<TLARS_Solver> inner_;

    /** @brief If true, the inner solver is a pure-LARS TLARS_Solver instead of
     *  the default TLASSO_Solver. Diagnostic / R-parity option. */
    bool use_lars_inner_{false};

    // ========================================================================
    // Static helpers
    // ========================================================================

    /**
     * @brief Build the (M x p) group-ridge block B.
     *
     * B(m, j) = sqrt(lambda2) / sqrt(p_m) for groups[j] == m, else 0.
     *
     * @param groups  0-based contiguous group ids, length p.
     * @param M       Number of groups.
     * @param lambda2 Group-ridge parameter.
     * @return B, shape (M x p).
     */
    static Eigen::MatrixXd buildGroupBlock(
        const Eigen::VectorXi& groups,
        std::size_t M,
        double lambda2);

    /**
     * @brief Center and/or rescale each column of `A` in place.
     *
     * Divide the centered column by its L2-norm (ScalingMode::L2) or by its
     * sample SD (ScalingMode::ZSCORE). A degenerate column (scale below eps)
     * keeps scale 1.0, consistent with the shared data-normalizer policy.
     * Applied to the DATA blocks before augmentation (never to the appended
     * group-ridge rows).
     *
     * @param A            Matrix to normalize in place.
     * @param eps          Numerical zero for the degenerate-column guard.
     * @param scaling_mode Column-scaling convention.
     * @param center       Center the columns (intercept convention).
     * @param scale        Rescale the columns (normalize convention).
     */
    static void centerAndScaleColumns(Eigen::MatrixXd& A, double eps,
                                      ScalingMode scaling_mode,
                                      bool center = true, bool scale = true);

    /** @brief Construct Eigen::Maps from the three owned buffers. */
    void buildMaps();

    /**
     * @brief Copy the inner solver's diagnostic state into the inherited
     * TSolver_Base members so all base getters (getRSS, getActives,
     * getNumSteps, ...) report the inner solver's progress.
     *
     * Called after construction, after every executeStep(), and after
     * deserialization.
     */
    void syncDiagnosticsFromInner();

    /**
     * @brief Create the inner solver, forwarding preprocessing flags.
     * @param normalize Passed to the inner solver (L2-normalise columns).
     * @param intercept Passed to the inner solver (centre columns and y).
     * @param precentered Passed to the inner solver (caller-centered data;
     *        false on the augmented system).
     * @param verbose Passed to the inner solver.
     * @param scaling_mode Passed to the inner solver (L2 or z-score scaling).
     * @param tie_seed Seed of the inner tie-break engine (SolverOptions::tie_seed).
     * @param n_threads Thread budget of the inner solver (SolverOptions::n_threads).
     */
    void buildInnerSolver(bool normalize, bool intercept, bool precentered,
                          bool verbose, ScalingMode scaling_mode, long long tie_seed,
                          int n_threads);

public:
    // ========================================================================
    // Constructors / destructor
    // ========================================================================

    /**
     * @brief Construct a TIENETAug_Solver.
     *
     * Standardizes the DATA blocks per the caller's intercept/normalize
     * flags (pre-augmentation family convention), appends the group-ridge
     * rows un-normalized, then creates the inner solver with
     * `normalize = false, intercept = false`.
     *
     * @param X          Predictor matrix (n x p).
     * @param D          Dummy matrix (n x L) with L a multiple of p; dummy
     *                   column j of each layer shares original variable j's
     *                   group.
     * @param y          Response vector (n).
     * @param lambda2    Group-ridge (L2) regularisation parameter (>= 0).
     * @param groups     0-based contiguous group ids in [0, M-1], length p.
     * @param normalize  Rescale the DATA-block columns (unit L2 / unit SD per
     *                   scaling_mode) before augmentation; for lambda2 == 0
     *                   forwarded to the inner solver instead. Default true.
     * @param intercept  Center the DATA-block columns and y before
     *                   augmentation; for lambda2 == 0 forwarded to the
     *                   inner solver instead. Default true.
     * @param verbose    Forwarded to the inner solver. Default false.
     * @param scaling_mode Column-scaling convention for the post-augmentation
     *                   preprocessing (and the lambda2 == 0 inner solver).
     *                   Default L2.
     * @param use_lars_inner If true, the inner forward-selection solver is a
     *                   pure-LARS TLARS_Solver (never drops variables) instead
     *                   of the default TLASSO_Solver.
     *
     * @throws std::invalid_argument on lambda2 < 0, malformed groups, or
     *         D.cols() not a multiple of X.cols() (lambda2 > 0 only).
     */
    TIENETAug_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                     Eigen::Map<Eigen::MatrixXd>& D,
                     Eigen::Map<Eigen::VectorXd>& y,
                     const Eigen::VectorXi& groups,
                     const SolverOptions& options = SolverOptions{});

    /**
     * @brief Default constructor for Cereal deserialization.
     *
     * @note After deserialization the Maps and inner solver are rebuilt
     * automatically by the serialize() template (via buildMaps()).
     */
    TIENETAug_Solver() = default;

    /** @brief Deleted copy constructor. */
    TIENETAug_Solver(const TIENETAug_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TIENETAug_Solver& operator=(const TIENETAug_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TIENETAug_Solver(TIENETAug_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TIENETAug_Solver& operator=(TIENETAug_Solver&&) = delete;

    /** @brief Defaulted destructor. */
    ~TIENETAug_Solver() override = default;

    // ========================================================================
    // Core solver interface (TSolver_Base pure virtuals)
    // ========================================================================

    /**
     * @brief Execute the inner solution path on the augmented system.
     *
     * Delegates to the inner solver; warm-starting is supported (calling
     * again with a larger T_stop continues the path).
     *
     * @param T_stop    Maximum number of dummy entries allowed in the active set.
     * @param early_stop If true, stop as soon as T_stop dummies have entered.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /** @brief Return the solver type as a string. */
    std::string solverTypeToString() const override { return "TIENETAug"; }

    // ========================================================================
    // Solution extraction overrides
    // ========================================================================

    /**
     * @brief Return the sparse coefficient path (inner path, unscaled).
     *
     * Coefficients live on the scale of the internally normalized augmented
     * columns (no global rescaling factor exists for the IEN, unlike the
     * Zou-Hastie d2 of TENETAug). The T-Rex framework consumes only the
     * active/inactive pattern, which is scale-invariant.
     */
    SparseBetaPath getBetaPathSparse() const override;

    /**
     * @brief Return coefficients at a given step (normalized augmented scale).
     * @param step Step index (-1 = last step).
     * @return Vector (p + L).
     */
    Eigen::VectorXd getBeta(int step = -1) const override;

    /**
     * @brief Return the intercept at a given step (forwards to the inner solver).
     * @param step Step index (-1 = last step).
     */
    double getIntercept(int step = -1) const override;

    /** @brief Set the tolerance on the wrapper and on the inner solver. */
    void setTolerance(double tol) override { eps_ = tol; inner_->setTolerance(tol); }

    /** @brief Set the thread budget on the wrapper and on the inner solver
     *  (the inner solver owns every region). */
    void setNumThreads(int n_threads) override {
        TSolver_Base::setNumThreads(n_threads);
        inner_->setNumThreads(n_threads);
    }

    /** @brief Set the tie-breaking seed on the inner solver (the wrapper's own
     *  random engine is unused). */
    void setTieSeed(uint32_t seed) override { inner_->setTieSeed(seed); }

    /** @brief The owned augmented X, D and y, the group vector, the wrapper's
     *  diagnostic mirror, the inner solver object and view objects (heap
     *  allocations of the wrapper) and inner_->memoryBytes(); the inner X and
     *  D view the owned copies, so nothing is counted twice. */
    std::size_t memoryBytes() const noexcept override;

    // ========================================================================
    // IEN-specific getters
    // ========================================================================

    /** @brief Group-ridge parameter used. */
    double getLambda2() const noexcept { return lambda2_; }

    /** @brief 0-based group id per original variable (length p). */
    const Eigen::VectorXi& getGroups() const noexcept { return groups_; }

    /** @brief Number of disjoint groups M. */
    std::size_t getNumGroups() const noexcept { return M_; }

    // ========================================================================
    // Cereal serialization
    // ========================================================================

    friend class cereal::access;

    /**
     * @brief Serialize owned buffers, wrapper metadata, and inner solver state.
     *
     * @note X_ and D_ (non-owned TSolver_Base pointers) are NOT serialized.
     * On load, the Maps are rebuilt from the owned buffers, the inner solver
     * is reconnected to them, and the wrapper's diagnostic mirror is refreshed.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(lambda2_),
                CEREAL_NVP(groups_),
                CEREAL_NVP(M_),
                CEREAL_NVP(eps_),
                CEREAL_NVP(use_lars_inner_),
                CEREAL_NVP(p_original_), CEREAL_NVP(num_dummies_),
                CEREAL_NVP(dummy_start_idx_), CEREAL_NVP(effective_n_),
                CEREAL_NVP(normalize_), CEREAL_NVP(intercept_), CEREAL_NVP(precentered_),
                CEREAL_NVP(verbose_),
                CEREAL_NVP(X_aug_owned_),
                CEREAL_NVP(D_aug_owned_),
                CEREAL_NVP(y_aug_owned_));
        archive(CEREAL_NVP(inner_));

        // Rebuild maps and reconnect the inner solver ONLY on load (rebuilding
        // during save would destroy the Map objects the inner solver points
        // into). The rebuild must be unconditional on load: when deserializing
        // into an already-constructed solver (e.g. the R bindings' load path)
        // stale maps exist but point into buffers cereal may have reallocated,
        // and the freshly deserialized inner solver is always disconnected.
        if constexpr (Archive::is_loading::value) {
            if (X_aug_owned_.size() > 0) {
                buildMaps();
                if (inner_) {
                    inner_->reconnect(*X_aug_map_, *D_aug_map_);
                    syncDiagnosticsFromInner();
                }
            }
        }
    }

    /**
     * @brief Save the current TIENETAug solver state to file (binary serialization).
     * @param filename Output file path.
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load a TIENETAug solver state from file (binary serialization).
     *
     * @details The wrapper owns its (augmented) data buffers, so unlike the
     * other solvers no reconnection to the passed maps happens; X and D are
     * only validated against the serialized dimensions for API parity.
     *
     * @param filename Path to input file.
     * @param X Map to the original feature matrix (dimension check only).
     * @param D Map to the original dummy matrix (dimension check only).
     * @return Deserialized TIENETAug_Solver.
     */
    static TIENETAug_Solver load(const std::string& filename,
                                 Eigen::Map<Eigen::MatrixXd>& X,
                                 Eigen::Map<Eigen::MatrixXd>& D);
};

// ============================================================================
} // namespace trex::tsolvers::ad::linear_model::lars_based

// ============================================================================
// Cereal polymorphic registration
// ============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::ad::linear_model::lars_based::TIENETAug_Solver)
CEREAL_REGISTER_POLYMORPHIC_RELATION(trex::tsolvers::ad::TSolver_Base,
                                     trex::tsolvers::ad::linear_model::lars_based::TIENETAug_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_LARS_BASED_TIENET_AUG_SOLVER_HPP */
