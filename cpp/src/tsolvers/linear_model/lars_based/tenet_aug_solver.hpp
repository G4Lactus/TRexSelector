// ============================================================================
// tenet_aug_solver.hpp
// ============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_LARS_BASED_TENET_AUG_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_LARS_BASED_TENET_AUG_SOLVER_HPP
// ============================================================================
/**
 * @file tenet_aug_solver.hpp
 *
 * @brief T-ElasticNet (Augmented) solver: thin wrapper around TLASSO_Solver
 * running on a data-augmented system that encodes the EN ridge penalty.
 *
 * @details
 *  Builds the augmented system once in the constructor:
 *    d1 = sqrt(lambda2),  d2 = 1/sqrt(1+lambda2)
 *
 *    X_aug = d2 * [ X    ]   ((n+p+L) × p)
 *                 [ d1·Ip]
 *                 [ 0_L  ]
 *
 *    D_aug = d2 * [ D    ]   ((n+p+L) × L)
 *                 [ 0_p  ]
 *                 [ d1·IL]
 *
 *    y_aug = [ y ; 0_{p+L} ]
 *
 *  Special case: when lambda2 = 0 the augmented rows are all zero and the
 *  inner TLASSO would see inflated effective_n = n+p+L instead of n, giving
 *  wrong max_actives.  The constructor therefore skips augmentation and passes
 *  the original X, D, y directly to the inner TLASSO (TENETAug collapses to
 *  TLASSO, consistent with TENET collapsing to TLASSO at lambda2 = 0).
 *
 *  All preprocessing (centering, L2-normalisation) and the LARS path are
 *  delegated to the inner TLASSO_Solver via the normalize / intercept flags.
 *
 *  Coefficient recovery:
 *    getBetaPathSparse() returns the inner path with values / d2
 *    = beta * sqrt(1+lambda2)
 */
// ============================================================================

// std includes
#include <memory>
#include <string>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/tsolver_base.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>

// ============================================================================

namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================

/**
 * @brief T-ElasticNet solver via data augmentation.
 *
 * @details Delegates all LARS logic to an inner TLASSO_Solver running on
 * the augmented (n+p+L)-row system.  The augmentation encodes the ridge
 * penalty in the data itself, yielding the same Gram structure as R's
 * trex_GVS() reference implementation.
 *
 * Public interface: pass X, D, y, lambda2 and call executeStep() /
 * getBetaPathSparse().  In-memory warm-starting is supported via the inner
 * solver.
 */
class TENETAug_Solver : public TSolver_Base {

    // ========================================================================
    // Owned augmented data
    // ========================================================================
    double   lambda2_{0.0};   ///< Ridge parameter
    double   d2_{1.0};        ///< 1 / sqrt(1 + lambda2) — beta-recovery scale

    Eigen::MatrixXd X_aug_owned_;  ///< Augmented, normalised X  ((n+p+L)×p)
    Eigen::MatrixXd D_aug_owned_;  ///< Augmented, normalised D  ((n+p+L)×L)
    Eigen::VectorXd y_aug_owned_;  ///< Augmented, centred y     (n+p+L)

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
     *  default, matching the augmented-Lasso EN recipe) or a plain TLARS_Solver
     *  (pure LARS; never drops — matches R's trex(type="lar")). */
    std::unique_ptr<TLARS_Solver> inner_;

    /** @brief If true, the inner solver is a pure-LARS TLARS_Solver instead of
     *  the default TLASSO_Solver. Diagnostic / R-parity option. */
    bool use_lars_inner_{false};

    // ========================================================================
    // Static helpers
    // ========================================================================

    /**
     * @brief Build the raw augmented X matrix (no normalisation).
     *
     * Constructs d2 · [X ; d1·Ip ; 0_L].  Centering and L2-normalisation
     * are delegated to the inner TLASSO_Solver.
     *
     * @param X   Predictor matrix (n × p).
     * @param p   Number of original predictors.
     * @param L   Number of dummy predictors.
     * @param d1  sqrt(lambda2).
     * @param d2  1 / sqrt(1 + lambda2).
     * @return Raw augmented X, shape (n+p+L) × p.
     */
    static Eigen::MatrixXd buildXAug(
        const Eigen::Map<Eigen::MatrixXd>& X,
        std::size_t p,
        std::size_t L,
        double d1,
        double d2);

    /**
     * @brief Build the raw augmented D matrix (no normalisation).
     *
     * Constructs d2 · [D ; 0_p ; d1·IL].  Centering and L2-normalisation
     * are delegated to the inner TLASSO_Solver.
     *
     * @param D   Dummy matrix (n × L).
     * @param p   Number of original predictors (X-ridge block rows).
     * @param L   Number of dummy predictors.
     * @param d1  sqrt(lambda2).
     * @param d2  1 / sqrt(1 + lambda2).
     * @return Raw augmented D, shape (n+p+L) × L.
     */
    static Eigen::MatrixXd buildDAug(
        const Eigen::Map<Eigen::MatrixXd>& D,
        std::size_t p,
        std::size_t L,
        double d1,
        double d2);

    /**
     * @brief Build the augmented response vector: y_aug = [y ; 0_{p+L}].
     *
     * Centering is delegated to the inner TLASSO_Solver (intercept flag).
     *
     * @param y   Response vector (n).
     * @param pL  Number of augmentation rows (p + L).
     * @return Augmented y, shape (n + pL).
     */
    static Eigen::VectorXd buildYAug(
        const Eigen::Map<Eigen::VectorXd>& y,
        std::size_t pL);

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
     * @brief Create the inner TLASSO_Solver, forwarding preprocessing flags.
     * @param normalize  Passed to TLASSO_Solver (L2-normalise columns).
     * @param intercept  Passed to TLASSO_Solver (centre columns and y).
     * @param verbose    Passed to TLASSO_Solver.
     * @param scaling_mode Passed to TLASSO_Solver (L2 or z-score scaling).
     */
    void buildInnerSolver(bool normalize, bool intercept, bool verbose,
                          ScalingMode scaling_mode = ScalingMode::L2);

public:
    // ========================================================================
    // Constructors / destructor
    // ========================================================================

    /**
     * @brief Construct a TENETAug_Solver.
     *
     * Builds the augmented system (d1/d2 formula), then creates an inner
     * TLASSO_Solver that handles all centering and L2-normalisation.
     *
     * @param X          Predictor matrix (n × p).
     * @param D          Dummy matrix (n × L).
     * @param y          Response vector (n).
     * @param lambda2    Ridge (L2) regularisation parameter (≥ 0).
     * @param normalize  Forwarded to TLASSO_Solver: if true, columns of the
     *                   augmented matrices are L2-normalised to unit norm.
     *                   Default true.
     * @param intercept  Forwarded to TLASSO_Solver: if true, columns and
     *                   response are centred.  Default true.
     * @param verbose    Forwarded to TLASSO_Solver.  Default false.
     * @param scaling_mode Forwarded to the inner solver: L2 (unit-L2) or z-score
     *                   column scaling.  Default L2.
     * @param use_lars_inner If true, the inner forward-selection solver is a
     *                   pure-LARS TLARS_Solver (never drops variables) instead of
     *                   the default TLASSO_Solver. Diagnostic / R-parity option.
     */
    TENETAug_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                    Eigen::Map<Eigen::MatrixXd>& D,
                    Eigen::Map<Eigen::VectorXd>& y,
                    double lambda2,
                    bool normalize = true,
                    bool intercept = true,
                    bool verbose = false,
                    ScalingMode scaling_mode = ScalingMode::L2,
                    bool use_lars_inner = false);

    /**
     * @brief Default constructor for Cereal deserialization.
     *
     * @note After deserialization the Maps and inner solver are rebuilt
     * automatically by the serialize() template (via buildMaps() +
     * buildInnerSolver()).
     */
    TENETAug_Solver() = default;

    /** @brief Deleted copy constructor. */
    TENETAug_Solver(const TENETAug_Solver&) = delete;

    /** @brief Deleted copy assignment. */
    TENETAug_Solver& operator=(const TENETAug_Solver&) = delete;

    /** @brief Default move constructor. */
    TENETAug_Solver(TENETAug_Solver&&) = default;

    /** @brief Deleted move assignment. */
    TENETAug_Solver& operator=(TENETAug_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TENETAug_Solver() override = default;

    // ========================================================================
    // Core solver interface (TSolver_Base pure virtuals)
    // ========================================================================

    /**
     * @brief Execute the T-LASSO solution path on the augmented system.
     *
     * Delegates to the inner TLASSO_Solver; warm-starting is supported
     * (calling again with a larger T_stop continues the path).
     *
     * @param T_stop    Maximum number of dummy entries allowed in the active set.
     * @param early_stop If true, stop as soon as T_stop dummies have entered.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /** @brief Return the solver type as a string. */
    std::string solverTypeToString() const override { return "TENETAug"; }

    // ========================================================================
    // Solution extraction overrides
    // ========================================================================

    /**
     * @brief Return the sparse coefficient path scaled back to original units.
     *
     * The inner LASSO runs in the augmented space where each column is scaled
     * by d₂ = 1/sqrt(1+λ).  Dividing by d₂ (i.e. multiplying by sqrt(1+λ))
     * recovers coefficients in the same scale as the original predictors.
     */
    SparseBetaPath getBetaPathSparse() const override;

    /**
     * @brief Return coefficients at a given step in original units.
     * @param step Step index (-1 = last step).
     * @return Vector (p + L), original-scale coefficients.
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
                CEREAL_NVP(d2_),
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
     * @brief Save the current TENETAug solver state to file (binary serialization).
     * @param filename Output file path.
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load a TENETAug solver state from file (binary serialization).
     *
     * @details The wrapper owns its (augmented) data buffers, so unlike the
     * other solvers no reconnection to the passed maps happens; X and D are
     * only validated against the serialized dimensions for API parity.
     *
     * @param filename Path to input file.
     * @param X Map to the original feature matrix (dimension check only).
     * @param D Map to the original dummy matrix (dimension check only).
     * @return Deserialized TENETAug_Solver.
     */
    static TENETAug_Solver load(const std::string& filename,
                                Eigen::Map<Eigen::MatrixXd>& X,
                                Eigen::Map<Eigen::MatrixXd>& D);
};

// ============================================================================
} // namespace trex::tsolvers::linear_model::lars_based

// ============================================================================
// Cereal polymorphic registration
// ============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::linear_model::lars_based::TENETAug_Solver)
CEREAL_REGISTER_POLYMORPHIC_RELATION(trex::tsolvers::TSolver_Base,
                                     trex::tsolvers::linear_model::lars_based::TENETAug_Solver)

#endif /* TSOLVERS_LINEAR_MODEL_LARS_BASED_TENET_AUG_SOLVER_HPP */
