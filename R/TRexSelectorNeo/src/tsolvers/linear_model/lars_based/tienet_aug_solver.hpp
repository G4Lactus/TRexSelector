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
 *  Preprocessing (R reference, lm_dummy.R with GVS_type == "IEN"): after
 *  augmentation the full system is column-centered and rescaled
 *  (R: `X_Dummy <- scale(X_Dummy)`), and the augmented response is centered
 *  (R: `y <- y - mean(y)`). This wrapper performs exactly that on its OWNED
 *  augmented buffers (honouring ScalingMode: unit-L2 or unit-SD columns) and
 *  then runs the inner solver with `normalize = false, intercept = false`.
 *  Consequently the reported coefficients live on the scale of the
 *  normalized augmented columns; the T-Rex framework only consumes the
 *  active/inactive pattern, which is scale-invariant.
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
#include <tsolvers/tsolver_base.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>

// ============================================================================

namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================

class TIENETAug_Solver : public TSolver_Base {

    // ========================================================================
    // Owned augmented data
    // ========================================================================
    double lambda2_{0.0};      ///< Group-ridge parameter (>= 0).

    Eigen::VectorXi groups_;   ///< 0-based group id per original variable (length p).
    std::size_t     M_{0};     ///< Number of disjoint groups.

    Eigen::MatrixXd X_aug_owned_;  ///< Augmented, normalised X  ((n+M) x p)
    Eigen::MatrixXd D_aug_owned_;  ///< Augmented, normalised D  ((n+M) x L)
    Eigen::VectorXd y_aug_owned_;  ///< Augmented, centred y     (n+M)

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
     * @brief Center each column of `A` and rescale it in place.
     *
     * Mirrors R's `scale()` on the augmented system: divide the centered
     * column by its L2-norm (ScalingMode::L2) or by its sample SD
     * (ScalingMode::ZSCORE). A degenerate column (scale below eps) keeps
     * scale 1.0, consistent with the shared data-normalizer policy.
     *
     * @param A            Matrix to normalize in place.
     * @param eps          Numerical zero for the degenerate-column guard.
     * @param scaling_mode Column-scaling convention.
     */
    static void centerAndScaleColumns(Eigen::MatrixXd& A, double eps,
                                      ScalingMode scaling_mode);

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
     * @param normalize  Passed to the inner solver (L2-normalise columns).
     * @param intercept  Passed to the inner solver (centre columns and y).
     * @param verbose    Passed to the inner solver.
     * @param scaling_mode Passed to the inner solver (L2 or z-score scaling).
     */
    void buildInnerSolver(bool normalize, bool intercept, bool verbose,
                          ScalingMode scaling_mode = ScalingMode::L2);

public:
    // ========================================================================
    // Constructors / destructor
    // ========================================================================

    /**
     * @brief Construct a TIENETAug_Solver.
     *
     * Builds the group-informed augmented system, applies the R-faithful
     * post-augmentation preprocessing (column centering + rescaling of
     * X_aug / D_aug, centering of y_aug), then creates the inner solver with
     * `normalize = false, intercept = false`.
     *
     * @param X          Predictor matrix (n x p).
     * @param D          Dummy matrix (n x L) with L a multiple of p; dummy
     *                   column j of each layer shares original variable j's
     *                   group.
     * @param y          Response vector (n).
     * @param lambda2    Group-ridge (L2) regularisation parameter (>= 0).
     * @param groups     0-based contiguous group ids in [0, M-1], length p.
     * @param normalize  Only used for the degenerate lambda2 == 0 path,
     *                   where it is forwarded to the inner solver. Default true.
     * @param intercept  Only used for the degenerate lambda2 == 0 path.
     *                   Default true.
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
                     double lambda2,
                     const Eigen::VectorXi& groups,
                     bool normalize = true,
                     bool intercept = true,
                     bool verbose = false,
                     ScalingMode scaling_mode = ScalingMode::L2,
                     bool use_lars_inner = false);

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

    /** @brief Set the tie-breaking seed on the inner solver (the wrapper's own
     *  random engine is unused). */
    void setTieSeed(uint32_t seed) { inner_->setTieSeed(seed); }

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
                CEREAL_NVP(normalize_), CEREAL_NVP(intercept_),
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
} // namespace trex::tsolvers::linear_model::lars_based

// ============================================================================
// Cereal polymorphic registration
// ============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::linear_model::lars_based::TIENETAug_Solver)
CEREAL_REGISTER_POLYMORPHIC_RELATION(trex::tsolvers::TSolver_Base,
                                     trex::tsolvers::linear_model::lars_based::TIENETAug_Solver)

#endif /* TSOLVERS_LINEAR_MODEL_LARS_BASED_TIENET_AUG_SOLVER_HPP */
