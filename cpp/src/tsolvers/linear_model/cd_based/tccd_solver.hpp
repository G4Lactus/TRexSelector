// ===================================================================================
// tccd_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_CD_BASED_TCCD_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_CD_BASED_TCCD_SOLVER_HPP
// ===================================================================================
/**
 * @file tccd_solver.hpp
 *
 * @brief Header file for the Terminating Cyclic Coordinate Descent (T-CCD)
 * solver.
 *
 * @details The TCCD_Solver class implements a terminating pathwise coordinate
 * descent for the penalized family
 *
 *     F(beta) = 1/2 ||y - X beta||^2 + lambda1 ||beta||_1
 *               + (lambda2/2) beta^T K beta,
 *
 * with support for dummy variable augmentation. Termination follows the T-Rex
 * contract: the lambda1 path is traversed until T_stop dummies are active,
 * localizing each dummy-count crossing by KKT-guided jumps plus bisection with
 * a support-nesting early stop. One path step is recorded per localized
 * crossing (per lambda1 grid point in full-path mode).
 *
 * The quadratic penalty is passed as data (matrix-passing core): K enters only
 * through the coordinate denominator, an incrementally maintained lambda2*K*beta
 * vector, and the KKT/entry conditions. Descendants configure K:
 * - TCENET_Solver:  K = I (diagonal elastic net),
 * - TCIENET_Solver: K = Gamma^T Gamma sparse (Tikhonov-informed elastic net).
 *
 * It inherits state management, standardization, and diagnostics from the
 * TSolver_Base class. Path recording semantics: solutions are exact penalized
 * lasso minimizers at each recorded lambda1 (T-LASSO-equivalent supports; a
 * variable that entered and left the support is absent, unlike plain T-LARS).
 */
// ===================================================================================

// std includes
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Cereal includes
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

// TSolver Base class include
#include <tsolvers/tsolver_base.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::linear_model::cd_based namespace
namespace trex::tsolvers::linear_model::cd_based {

// ===================================================================================

/**
 * @brief Enumerates all supported T-Solver family algorithms with
 * coordinate-descent solution path based on the pathwise CCD framework.
 *
 * Used for algorithm selection, configuration, and logging/reporting.
 */
enum class SolverTypeCdBased: int {
    TCCD = 0,     // Terminating Cyclic Coordinate Descent (LASSO)
    TCENET = 1,   // T-CCD Elastic Net (diagonal ridge, K = I)
    TCIENET = 2   // T-CCD Informed Elastic Net (group means / Tikhonov K)
};


/**
 * @brief Terminating Cyclic Coordinate Descent (T-CCD) solver.
 * Supports in-memory and memory-mapped data over Eigen::Map.
 */
class TCCD_Solver : public TSolver_Base {
protected:
    // ==========================================================================
    // CCD specific state variables
    // ==========================================================================

    /** @brief Regularization parameter sequence (lambda1 at each recorded step). */
    std::vector<double> lambda_;

    /** @brief lambda1 at the start of the path (max |X^T y|). */
    double lambda_max_{0.0};

    /** @brief Current position on the lambda1 path (persists across warm restarts). */
    double lambda_current_{0.0};

    /** @brief Quadratic penalty kind. NONE/DIAGONAL/SPARSE are served by the
     *  matrix-passing core; GROUP marks structurally specialized descendants
     *  (TCIENET group mode) that override the penalty hooks entirely. */
    enum class PenaltyKind : int { NONE = 0, DIAGONAL = 1, SPARSE = 2, GROUP = 3 };

    /** @brief Configured penalty kind (NONE for plain T-CCD/LASSO). */
    PenaltyKind penalty_kind_{PenaltyKind::NONE};

    /** @brief Quadratic penalty weight lambda2 (>= 0). */
    double lambda2_{0.0};

    /** @brief Ridge weight of the dummy block (dummies always get kappa * I). */
    double kappa_dummy_{1.0};

    /** @brief Diagonal penalty weights over the real block (DIAGONAL kind). */
    Eigen::VectorXd penalty_dvec_{};

    /** @brief Tikhonov matrix K = Gamma^T Gamma over the real block (SPARSE kind). */
    Eigen::SparseMatrix<double> tikhonov_K_{};

    /** @brief Precomputed lambda2 * K_jj over the unified index space. */
    Eigen::VectorXd penalty_diag_{};

    /** @brief Maintained lambda2 * K * beta over the unified index space. */
    Eigen::VectorXd l2kb_{};

    /** @brief Dense CD coefficients over the unified index space
     *  (snapshotted into the sparse base path at each recorded step). */
    Eigen::VectorXd beta_cd_{};

    /** @brief Squared column norms (1 after L2 standardization; general otherwise). */
    Eigen::VectorXd colnorm2_{};

    /** @brief O(1) support membership (actives_ holds the ordered support). */
    std::vector<char> in_active_{};

    /** @brief O(1) lookup for preprocessing-dropped columns. */
    std::vector<char> is_dropped_{};

    /** @brief r_ is stale w.r.t. beta_cd_ (gram-mode sweeps defer the update). */
    bool r_dirty_{false};

    // --- Gram / covariance cache over ever-active slots (transient, rebuilt) ---

    /** @brief Pristine X^T y correlations (rebuilt on reconnect). */
    Eigen::VectorXd correlations_y_{};

    /** @brief Variable -> gram slot (-1: none). */
    std::vector<Eigen::Index> slot_of_{};

    /** @brief Gram slot -> variable. */
    std::vector<std::size_t> slot_var_{};

    /** @brief Gram rows over slots (data part only; row s: x_var(s)^T x_var(t)). */
    std::vector<Eigen::VectorXd> gram_rows_{};

    /** @brief Maintained q(s) = x_var(s)^T (y - X beta) over slots. */
    Eigen::VectorXd gram_q_{};

    /** @brief False once the ever-active set exceeded gram_cap_ (naive sweeps). */
    bool gram_active_{true};

    /** @brief Transient guard: CD caches (colnorm2_, gram, correlations_y_) built. */
    bool cd_ready_{false};

    // --- Crossing predictor state (secant on the dummy-correlation decay) ---

    /** @brief Previous predictor sample: lambda position. */
    double pred_prev_lambda_{-1.0};

    /** @brief Previous predictor sample: max inactive-dummy |penalized corr|. */
    double pred_prev_value_{-1.0};

    // --- Algorithm knobs (setters below) ---

    /** @brief Relative lambda tolerance of the crossing bisection. */
    double lambda_rel_tol_{1e-3};

    /** @brief Overshoot margin applied to the predicted crossing lambda. */
    double jump_margin_{0.995};

    /** @brief Coordinate-sweep convergence tolerance (certification solves). */
    double cd_tol_{1e-10};

    /** @brief Looser sweep tolerance for jump/bisection probe solves. */
    double cd_tol_probe_{1e-8};

    /** @brief Relative KKT violation slack for support entry. */
    double kkt_slack_{1e-7};

    /** @brief Max ever-active gram slots before falling back to naive sweeps. */
    std::size_t gram_cap_{400};

    /** @brief Number of lambda1 grid points in full-path mode (T_stop == 0). */
    std::size_t grid_points_{100};

    /** @brief lambda_min / lambda_max ratio of the full-path grid. */
    double lambda_min_ratio_{1e-3};

    /** @brief Safety cap on visited lambda points per executeStep call. */
    std::size_t max_lambda_points_{100000};

    /** @brief Stall guard: max coordinate sweeps per fixed-lambda solve.
     *  CD on nearly collinear active columns can limit-cycle with deltas
     *  hovering at the tolerance; the budget bounds the tail (the KKT
     *  certificate still reports the honest residual violation). */
    std::size_t max_sweeps_per_solve_{2000};

    /** @brief Diagnostic: number of solves that hit the sweep budget. */
    long sweep_cap_hits_{0};

    // --- Diagnostics counters ---

    /** @brief Monotonic counter of support removals (pruned exact zeros). */
    std::size_t num_removals_{0};

    /** @brief Visited lambda points (jumps + bisection probes + certifications). */
    long lambda_points_{0};

    /** @brief Full O(n * (p + num_dummies)) KKT passes. */
    long full_kkt_passes_{0};

    /** @brief Candidate-restricted (cheap) KKT passes. */
    long restricted_kkt_passes_{0};

    /** @brief Coordinate sweeps over the support. */
    long cd_sweeps_{0};

    /** @brief Algorithm variant type. */
    SolverTypeCdBased algo_type_{SolverTypeCdBased::TCCD};

    /** @brief Protected constructor for inheritance. */
    explicit TCCD_Solver(SolverTypeCdBased type);

public:

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    /**
     * @brief Construct a TCCD_Solver object using data and configuration.
     *
     * @param X Original design matrix (n x p).
     * @param D Dummy design matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param normalize If true, each column of X and D is scaled to unit L2-norm.
     * @param intercept If true, X, D, and y are centered (intercept fitted).
     * @param verbose If true, print status and diagnostics during execution.
     * @param algorithm_type SolverTypeCdBased variant.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     *
     * @note X and D are not copied nor owned, so changes to X or D alter the
     *       underlying data.
     *
     * @note Near-constant columns are dropped by the shared preprocessing when
     *       normalize is true. Unlike the LARS family, exact collinearity among
     *       the remaining columns is tolerated by coordinate descent (the
     *       minimizer may be non-unique; convergence is unaffected).
     */
    TCCD_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                Eigen::Map<Eigen::MatrixXd>& D,
                Eigen::Map<Eigen::VectorXd>& y,
                bool normalize = true,
                bool intercept = true,
                bool verbose = false,
                SolverTypeCdBased algorithm_type = SolverTypeCdBased::TCCD,
                ScalingMode scaling_mode = ScalingMode::L2
               );

    /**
     * @brief Default constructor for serialization or deferred initialization.
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TCCD_Solver();

    /** @brief Virtual destructor (X_ pointer is not owned, so not deleted). */
    virtual ~TCCD_Solver() override = default;

    /** @brief Deleted copy constructor and assignment. */
    TCCD_Solver(const TCCD_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TCCD_Solver& operator=(const TCCD_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TCCD_Solver(TCCD_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TCCD_Solver& operator=(TCCD_Solver&&) = delete;

    // ============================================================================
    // Main Algorithm (Overrides Pure Virtual)
    // ============================================================================

    /**
     * @brief Compute the T-CCD solution path with early stopping based on the
     * dummy threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold
     *               (0 = full solution path over a geometric lambda1 grid).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts (a later
     * call with a larger T_stop continues the path from lambda_current_) and
     * serialization. One path step is recorded per localized dummy-count
     * crossing; simultaneous entries within the lambda tolerance are recorded
     * at a single step (dummies_at_step_ then advances by more than one).
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /**
     * @brief Convert enum to human-readable string for reporting/logging.
     *
     * @return Static string name.
     */
    std::string solverTypeToString() const override {
        switch(algo_type_) {
            case SolverTypeCdBased::TCCD:     return "TCCD";
            case SolverTypeCdBased::TCENET:   return "TCENET";
            case SolverTypeCdBased::TCIENET:  return "TCIENET";
            default:                          return "UnknownSolver";
        }
    }

    // ============================================================================
    // Specific Getters and Setters for TCCD
    // ============================================================================

    /** @brief Get regularization parameter sequence (lambda1 per recorded step). */
    const std::vector<double>& getLambda() const noexcept { return lambda_; }

    /** @brief Get the quadratic penalty weight lambda2. */
    double getLambda2() const noexcept { return lambda2_; }

    /**
     * @brief Get total number of support removals (negative actions).
     * @return Monotonic count of all removals throughout the solution path.
     */
    std::size_t getNumRemovals() const noexcept { return num_removals_; }

    /**
     * @brief Get cycling ratio (removals / additions).
     * @return Ratio, or 0.0 if no additions yet.
     */
    double getCyclingRatio() const {
        if (num_additions_ == 0) { return 0.0; }
        return static_cast<double>(num_removals_) /
               static_cast<double>(num_additions_);
    }

    /**
     * @brief Set the relative lambda tolerance of the crossing bisection.
     * @note Must be called before executeStep(); typical range 1e-4 - 1e-2.
     */
    void setLambdaRelTol(double tol);

    /** @brief Get the relative lambda tolerance of the crossing bisection. */
    double getLambdaRelTol() const noexcept { return lambda_rel_tol_; }

    /**
     * @brief Set coordinate-sweep convergence tolerances.
     *
     * @param cd_tol Certification tolerance (recorded steps). Typical 1e-10.
     * @param cd_tol_probe Probe tolerance (jump/bisection solves). Typical 1e-8.
     */
    void setCdTolerances(double cd_tol, double cd_tol_probe);

    /**
     * @brief Set the maximum number of ever-active gram slots before the
     * solver falls back to naive residual sweeps (gram build is O(n * S^2)).
     */
    void setGramCap(std::size_t cap) { gram_cap_ = cap; }

    /**
     * @brief Configure the full-path mode grid (T_stop == 0).
     *
     * @param grid_points Number of geometric lambda1 grid points.
     * @param lambda_min_ratio lambda_min / lambda_max of the grid.
     */
    void setFullPathGrid(std::size_t grid_points, double lambda_min_ratio);

    /** @brief Get visited lambda points (diagnostics). */
    long getLambdaPoints() const noexcept { return lambda_points_; }

    /** @brief Get full KKT pass count (the O(n*(p+L)) unit; diagnostics). */
    long getFullKktPasses() const noexcept { return full_kkt_passes_; }

    /** @brief Get restricted KKT pass count (diagnostics). */
    long getRestrictedKktPasses() const noexcept { return restricted_kkt_passes_; }

    /** @brief Get coordinate sweep count (diagnostics). */
    long getCdSweeps() const noexcept { return cd_sweeps_; }

    /**
     * @brief Set the stall guard: max coordinate sweeps per fixed-lambda
     * solve. Sweep convergence is tested RELATIVE to the coefficient scale
     * (max(1, max|beta|)); the budget bounds pathological limit cycles on
     * nearly collinear active sets. Typical range 500 - 10000.
     */
    void setMaxSweepsPerSolve(std::size_t cap) {
        if (cap > 0) { max_sweeps_per_solve_ = cap; }
    }

    /** @brief Get number of solves that hit the sweep budget (diagnostics). */
    long getSweepCapHits() const noexcept { return sweep_cap_hits_; }

    /**
     * @brief Max KKT violation of the current state at the given lambda1
     * (0 up to numerical tolerance = certified exact penalized-lasso solution).
     *
     * @param lambda1 L1 penalty to certify against (default: lambda_current_).
     */
    double getKktViolation(double lambda1 = -1.0);

    // ============================================================================
    // De-/Serialization
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal model and path state except for input
     * matrices and the transient CD caches (gram, colnorm2_, correlations_y_),
     * which are rebuilt lazily after reconnect().
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        // Enums pass through local ints (reinterpret_cast aliasing is UB);
        // the assignments after archive() apply the loaded values and are
        // no-op round-trips on save.
        int algo_type = static_cast<int>(algo_type_);
        int penalty_kind = static_cast<int>(penalty_kind_);

        // Sparse K passes through triplets (portable, no Eigen-sparse adapter).
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
            cereal::base_class<TSolver_Base>(this),
            cereal::make_nvp("algo_type", algo_type),
            CEREAL_NVP(lambda_),
            CEREAL_NVP(lambda_max_),
            CEREAL_NVP(lambda_current_),
            cereal::make_nvp("penalty_kind", penalty_kind),
            CEREAL_NVP(lambda2_),
            CEREAL_NVP(kappa_dummy_),
            CEREAL_NVP(penalty_dvec_),
            cereal::make_nvp("K_dim", K_dim),
            cereal::make_nvp("K_rows", K_rows),
            cereal::make_nvp("K_cols", K_cols),
            cereal::make_nvp("K_vals", K_vals),
            CEREAL_NVP(l2kb_),
            CEREAL_NVP(beta_cd_),
            CEREAL_NVP(in_active_),
            CEREAL_NVP(pred_prev_lambda_),
            CEREAL_NVP(pred_prev_value_),
            CEREAL_NVP(lambda_rel_tol_),
            CEREAL_NVP(jump_margin_),
            CEREAL_NVP(cd_tol_),
            CEREAL_NVP(cd_tol_probe_),
            CEREAL_NVP(kkt_slack_),
            CEREAL_NVP(gram_cap_),
            CEREAL_NVP(grid_points_),
            CEREAL_NVP(lambda_min_ratio_),
            CEREAL_NVP(num_removals_),
            CEREAL_NVP(lambda_points_),
            CEREAL_NVP(full_kkt_passes_),
            CEREAL_NVP(restricted_kkt_passes_),
            CEREAL_NVP(cd_sweeps_),
            CEREAL_NVP(max_sweeps_per_solve_),
            CEREAL_NVP(sweep_cap_hits_)
        );

        algo_type_ = static_cast<SolverTypeCdBased>(algo_type);
        penalty_kind_ = static_cast<PenaltyKind>(penalty_kind);

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
        }

        // Transient caches must be rebuilt after load.
        cd_ready_ = false;
    }

    /**
     * @brief Save the current TCCD solver/model state to file (binary
     * serialization).
     *
     * @details Must be class specific implemented because Cereal does not use
     * virtual dispatch: cereal resolves the serialize template based on the
     * static type at the call site.
     *
     * @param filename Path to output file.
     */
    virtual void save(const std::string& filename) const;

    /**
     * @brief Load TCCD solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TCCD_Solver object connected to X and D.
     */
    static TCCD_Solver load(const std::string& filename,
                            Eigen::Map<Eigen::MatrixXd>& X,
                            Eigen::Map<Eigen::MatrixXd>& D);

protected:

    // ============================================================================
    // Internal CCD math helpers
    // ============================================================================

    /**
     * @brief Lazily (re)build the transient CD caches: squared column norms,
     * pristine X^T y correlations, dropped-column lookup, penalty diagonal,
     * and an empty gram cache. Called at the top of executeStep(); required
     * after construction and after reconnect()/load. Virtual so structurally
     * specialized descendants can rebuild their own penalty bookkeeping
     * (e.g. TCIENET group sums) after deserialization.
     */
    virtual void ensureCdState();

    /** @brief Build penalty_diag_ = lambda2 * diag(K) over the unified space.
     *  Virtual: specialized descendants provide their structural diagonal. */
    virtual void configurePenaltyDiag();

    /**
     * @brief Apply lambda2 * K * (delta e_j) to the maintained penalty state.
     * Matrix-passing core: updates l2kb_ (O(1) for NONE/DIAGONAL,
     * O(nnz(K col j)) for SPARSE). Virtual: specialized descendants maintain
     * their own compressed state (TCIENET group mode: one group-sum scalar, O(1)).
     */
    virtual void penaltyRankUpdate(std::size_t j, double delta);

    /**
     * @brief Quadratic penalty gradient component [lambda2 * K * beta]_j of
     * the current coefficients. Matrix-passing core: reads the maintained
     * l2kb_ vector. Virtual: specialized descendants compute it from their
     * compressed state (TCIENET group mode: lambda2 * sigma_m / p_m).
     */
    virtual double penaltyGradient(std::size_t j) const {
        return l2kb_(static_cast<Eigen::Index>(j));
    }

    /**
     * @brief One penalized coordinate update of variable j.
     *
     * @param j Unified column index.
     * @param xr Current x_j^T r (from gram q or a residual dot).
     * @param lambda1 L1 penalty.
     *
     * @return Coefficient change delta (0.0 if unchanged).
     */
    double coordinateUpdate(std::size_t j, double xr, double lambda1);

    /**
     * @brief Warm-started penalized lasso solve at fixed lambda1 with the
     * active-set strategy: cycle the support to convergence, then a KKT pass
     * over the non-support (restricted to candidates when given, else full);
     * add violators and repeat. Exact zeros are pruned from the support on
     * exit (gram slots are kept: ever-active caching).
     *
     * @param lambda1 L1 penalty.
     * @param cd_tol Sweep convergence tolerance (certification vs probe).
     * @param candidates Restricted KKT scan set (nullptr = full scan).
     *
     * @return Number of coordinate sweeps performed.
     */
    long solveAtLambda(double lambda1, double cd_tol,
                       const std::vector<std::size_t>* candidates);

    /** @brief Rebuild r_ = y - X beta from the support, O(n |A|). */
    void refreshResidualCd();

    /**
     * @brief Enter variable j into the gram cache (first entry: O(n * slots)
     * row build; re-entry: free, the slot and its maintained q persist).
     *
     * @return Slot index of j.
     */
    Eigen::Index gramEnter(std::size_t j);

    /**
     * @brief Predict the lambda1 of the next dummy crossing from the KKT
     * correlations (inactive dummies enter when lambda1 falls to their
     * penalized correlation), refined by a secant extrapolation of the
     * dummy-correlation decay along the path.
     */
    double predictNextCrossing();

    /**
     * @brief Record one path step at the localized crossing lambda1: step
     * counter, lambda_, actions_ (support diff vs the previous recorded
     * step), sparse beta snapshot, RSS_/R2_/DoF_, dummy tracking, and the
     * inactive set rebuild.
     */
    void recordCrossingStep(double lambda1);

    /** @brief Count dummies in the current support (nonzero coefficients). */
    std::size_t countActiveDummiesCd() const;

    /**
     * @brief True if the bracket's support growth from hi_support to
     * lo_support consists of entering dummies only (the crossing's real
     * support is then already determined — bisection can stop early).
     */
    bool supportOnlyDummyGrowth(const std::vector<std::size_t>& hi_support,
                                const std::vector<std::size_t>& lo_support) const;

    /**
     * @brief Collect the restricted KKT candidate set for a bisection bracket:
     * the support plus everything within 90% of the bracket bottom.
     */
    void collectCandidates(double lambda_lo,
                           std::vector<std::size_t>& candidates) const;
};

// ===================================================================================

} /* End of namespace trex::tsolvers::linear_model::cd_based */


#endif /* End of TSOLVERS_LINEAR_MODEL_CD_BASED_TCCD_SOLVER_HPP */
