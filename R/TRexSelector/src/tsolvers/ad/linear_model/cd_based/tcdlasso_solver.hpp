// ===================================================================================
// tcdlasso_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_LINEAR_MODEL_CD_BASED_TCDLASSO_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_CD_BASED_TCDLASSO_SOLVER_HPP
// ===================================================================================
/**
 * @file tcdlasso_solver.hpp
 *
 * @brief Header file for the Terminating Cyclic Coordinate Descent (T-CDLASSO)
 * solver.
 *
 * @details The TCDLASSO_Solver class implements a terminating pathwise coordinate
 * descent for the penalized family
 *
 *     F(beta) = 1/2 ||y - X beta||^2 + lambda1 ||beta||_1
 *               + (lambda2/2) beta^T K beta,
 *
 * with support for dummy variable augmentation. Termination follows the T-Rex
 * contract: the lambda1 path is traversed until T_stop dummies are active,
 * one path step being recorded per dummy-count crossing (per lambda1 grid
 * point in full-path mode).
 *
 * Crossing localization (default CERTIFIED): every bracket endpoint is a
 * *certified point* — a fixed-lambda1 solve carrying a duality-gap
 * certificate and coordinate-status certificates (certified OUT / IN with
 * sign) until the signed support is exact — and every stretch of the path
 * between consecutive certified points is covered by a bracket certificate:
 * knot-free (closed-form affine continuation from the Gram cache), dummy-
 * bounded (interval screen), or single-knot (two-sided continuation). A
 * crossing is recorded AT the knot, with the certified signed support of the
 * segment below it; ties within the lambda tolerance form one batch. The
 * LEGACY mode (KKT-guided jumps plus probe bisection with a support-nesting
 * early stop) stays available behind setCertifiedLocalization(false).
 *
 * The quadratic penalty is passed as data (matrix-passing core): K enters only
 * through the coordinate denominator, an incrementally maintained lambda2*K*beta
 * vector, and the KKT/entry conditions. Descendants configure K:
 * - TCDENET_Solver:    K = I (diagonal elastic net),
 * - TCDIENET_Solver:   structured group penalties (group-mean IEN / within-group
 *                     centering) with O(1) bookkeeping hooks,
 * - TCDTIKNET_Solver: K = Gamma^T Gamma sparse (general Tikhonov, optional
 *                     FOLDED dummy coupling).
 *
 * It inherits state management, standardization, and diagnostics from the
 * TSolver_Base class. Path recording semantics: solutions are exact penalized
 * lasso minimizers at each recorded lambda1 (T-LASSO-equivalent supports; a
 * variable that entered and left the support is absent, unlike plain T-LARS).
 */
// ===================================================================================

// std includes
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Cereal includes
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

// TSolver Base class include
#include <tsolvers/ad/tsolver_base.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::ad::linear_model::cd_based namespace
namespace trex::tsolvers::ad::linear_model::cd_based {

// ===================================================================================

/**
 * @brief Enumerates all supported T-Solver family algorithms with
 * coordinate-descent solution path based on the pathwise CCD framework.
 *
 * Used for algorithm selection, configuration, and logging/reporting.
 */
enum class SolverTypeCdBased: int {
    TCDLASSO = 0,      // Terminating Cyclic Coordinate Descent (LASSO)
    TCDENET = 1,    // T-CDLASSO Elastic Net (diagonal ridge, K = I)
    TCDIENET = 2,   // T-CDLASSO Informed Elastic Net (structured group penalties)
    TCDTIKNET = 3  // T-CDLASSO Tikhonov Net (general sparse K = Gamma^T Gamma)
};


/**
 * @brief Terminating Cyclic Coordinate Descent (T-CDLASSO) solver.
 * Supports in-memory and memory-mapped data over Eigen::Map. The Gram rows
 * of the ever-active variables and the continuation cache stay allocated
 * after executeStep(), so memoryBytes() counts them.
 *
 *  References:
 *  -----------
 *   R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," J. R. Stat. Soc. Ser. B Stat.
 *     Methodol., vol. 58, no. 1, pp. 267-288, 1996.
 *   J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
 *     Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
 *   J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
 *     Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
 *   W. J. Fu, "Penalized Regression: The Bridge Versus the Lasso," J. Comput. Graph. Statist., vol. 7,
 *     pp. 397-416, 1998.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TCDLASSO_Solver : public TSolver_Base {
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
     *  matrix-passing core; GROUP (group-mean / IEN) and GROUP_CENTERING
     *  (within-group centering, K = I - sum_m 1_m 1_m^T / p_m) mark
     *  structurally specialized descendants (TCDIENET) that override the
     *  penalty hooks entirely with O(1) group bookkeeping. */
    enum class PenaltyKind : int {
        NONE            = 0,
        DIAGONAL        = 1,
        SPARSE          = 2,
        GROUP           = 3,
        GROUP_CENTERING = 4
    };

    /** @brief Configured penalty kind (NONE for plain T-CDLASSO/LASSO). */
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

    /** @brief Base variable of unified column t under the layered dummy
     *  convention (identity for reals, (t - p) mod p for dummy copies).
     *  Shared by the folding descendants (TCDIENET GROUP_CENTERING,
     *  TCDTIKNET FOLDED coupling). */
    std::size_t foldedBase(std::size_t t) const noexcept {
        return (t < p_original_) ? t : (t - p_original_) % p_original_;
    }

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

    /** @brief Relative lambda tolerance: the tie tolerance of the certified
     *  localization (knots closer than this are one batch; margin of the
     *  bracket certificates), and the width stop of the LEGACY bisection. */
    double lambda_rel_tol_{1e-3};

    /** @brief Overshoot margin applied to the predicted crossing lambda. */
    double jump_margin_{0.995};

    /** @brief Coordinate-sweep convergence tolerance (certification solves).
     *
     *  Kept at 1e-10 deliberately. In
     *  CERTIFIED mode it only seeds the first certification round —
     *  certifyPoint() tightens by x0.1 per round until every coordinate is
     *  decided — so a looser seed saves little (5-9 % on the scenario-B
     *  agreement probe) and starts the rounds from a worse iterate: at 1e-7
     *  the A0-w fallbacks rose tenfold (4 -> 39 over 8 x 20 paths). In
     *  LEGACY mode 1e-7 (the value aligned with kkt_slack_, which decides
     *  membership) is 2.2x faster with identical selections on that one
     *  design; it stays a per-run knob (cd_tol_certify) until the evidence
     *  covers more designs. */
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

    // --- Certified crossing localization (T-CD theory note, Part II) ---

    /** @brief Crossing-localization mode.
     *  LEGACY:    probe bisection with the support-nesting early stop and
     *             the relative-width fallback (pre-2026-08 behaviour).
     *  CERTIFIED: certified points (duality gap + coordinate-status
     *             certificates), closed-form affine continuation from the
     *             Gram cache, bracket certificates (knot-free / dummy-bounded
     *             / single-knot), records located at the knot and carrying
     *             the certified signed support. */
    enum class LocalizationMode : int { LEGACY = 0, CERTIFIED = 1 };

    /** @brief Active localization mode (default CERTIFIED). */
    LocalizationMode localization_{LocalizationMode::CERTIFIED};

    /** @brief A certified point: exact signed support at lambda (Def. 11),
     *  issued by certifyPoint(). `conventional` marks the path start
     *  lambda_max (support empty by Def. 1; it is a knot and cannot anchor a
     *  downward continuation). */
    struct CertifiedPoint {
        double lambda{0.0};
        std::vector<std::size_t> support{};   // sorted unified indices (certified IN)
        std::vector<int> sign{};              // aligned signs (+1/-1)
        std::vector<double> beta{};           // aligned coefficient values
        std::size_t num_dummies{0};
        double gap{0.0};                      // duality gap at certification
        double kkt{0.0};                      // KKT violation at certification
        double rss{0.0};
        bool conventional{false};

        template<class Archive>
        void serialize(Archive& ar) {
            ar(CEREAL_NVP(lambda), CEREAL_NVP(support), CEREAL_NVP(sign),
               CEREAL_NVP(beta), CEREAL_NVP(num_dummies), CEREAL_NVP(gap),
               CEREAL_NVP(kkt), CEREAL_NVP(rss), CEREAL_NVP(conventional));
        }
    };

    /** @brief Closed-form affine continuation from a certified point
     *  (Def. 5 / Prop. 2): anchor beta*_A, direction d_A, penalized
     *  correlation and its slope, and the next-knot time with the
     *  candidates attaining it. */
    struct Continuation {
        bool ok{false};
        bool downward{true};
        double lambda0{0.0};
        bool has_next{false};                  // false: no candidate (no knot in this direction)
        double t_next{std::numeric_limits<double>::max()};   // finite sentinel when !has_next (fast-math safe)
        std::vector<std::size_t> event_vars{}; // candidates within eta of t_next
        std::vector<int> event_kind{};         // +1 entry / -1 exit (aligned)
        std::vector<int> event_sign{};         // entry sign / exit sign (aligned)
        Eigen::VectorXd beta_star{};           // over the point's support
        Eigen::VectorXd dir{};                 // over the point's support
        Eigen::VectorXd c_exact{};             // penalized correlation at lambda0
        Eigen::VectorXd cdot{};                // d c / d(-lambda) along the continuation
    };

    /** @brief Bracket certificate tag (Defs. 12-13, Prop. 3). */
    enum class BracketTag : int { KF = 0, DB = 1, SK = 2, UNRESOLVED = 3 };

    /** @brief Result of certifyBracket(). */
    struct BracketResult {
        BracketTag tag{BracketTag::UNRESOLVED};
        double knot{0.0};          // SK: largest knot of the batch (kappa_down)
        double kappa_down{0.0};    // first knot below the upper point (0 if unknown)
        double kappa_up{0.0};      // first knot above the lower point (0 if unknown)
        bool supports_differ{false};
    };

    /** @brief Ledger top P_cur: lower point of the last record (or the
     *  conventional point lambda_max). Persists across warm restarts. */
    CertifiedPoint frontier_top_{};

    /** @brief Certified points below the ledger top not yet consumed,
     *  sorted by decreasing lambda (spare cuts / overshoot probes). */
    std::vector<CertifiedPoint> frontier_below_{};

    /** @brief Frontier initialized (conventional top installed). */
    bool frontier_init_{false};

    /** @brief Certification became unavailable along the path (A0-w boundary,
     *  e.g. lasso saturation |A| -> n): the solver fell back to the LEGACY
     *  localization for the remaining path; those records carry gap/kkt = -1
     *  (uncertified). Persistent across warm restarts. */
    bool certified_degraded_{false};

    /** @brief Transient cache of continuations keyed by (lambda, direction):
     *  a certified point is re-tested against several brackets after cuts;
     *  its continuation is exact and can be reused. Not serialized. */
    std::vector<Continuation> cont_cache_{};

    /** @brief Cache capacity (entries); oldest evicted first. */
    std::size_t cont_cache_cap_{32};

    /** @brief Certification rounds per point before STALL (lambda at a knot). */
    std::size_t max_certify_rounds_{8};

    /** @brief Nudges of a stalled certification point before giving up. */
    std::size_t max_nudges_{8};

    /** @brief Max size of the unscreened set for the restricted radius
     *  lambda_min(H_SS) (O(|S|^3) eigen-solve); larger sets keep tightening. */
    std::size_t mu_s_cap_{400};

    // --- Certified-mode diagnostics ---
    long certified_solves_{0};
    long certify_stalls_{0};
    long brackets_kf_{0};
    long brackets_db_{0};
    long brackets_sk_{0};
    long brackets_cut_{0};
    long continuations_{0};

    /** @brief Duality gap / KKT violation of every recorded step (certified
     *  mode; 0 for the initial step). */
    std::vector<double> record_gap_{};
    std::vector<double> record_kkt_{};

    /** @brief Algorithm variant type. */
    SolverTypeCdBased algo_type_{SolverTypeCdBased::TCDLASSO};

    /** @brief Protected constructor for inheritance. */
    explicit TCDLASSO_Solver(SolverTypeCdBased type);

    /** @brief Protected data constructor for inheritance: the public
     *  constructor with the algorithm variant passed through. Applies the
     *  CD knobs of the options (lambda tolerance, sweep tolerances, Gram
     *  cap, sweep cap, localization mode). */
    TCDLASSO_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                    Eigen::Map<Eigen::MatrixXd>& D,
                    Eigen::Map<Eigen::VectorXd>& y,
                    const SolverOptions& options,
                    SolverTypeCdBased algorithm_type);

public:

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    /**
     * @brief Construct a TCDLASSO_Solver object using data and configuration.
     *
     * @param X Original design matrix (n x p).
     * @param D Dummy design matrix (n x num_dummies).
     * @param y Response vector (n).
     * @param options Every other input (SolverOptions): preprocessing flags,
     *        tolerance, tie-breaking and the cd_* knobs.
     *
     * @note X and D are not copied nor owned, so changes to X or D alter the
     *       underlying data.
     *
     * @note Near-constant columns are dropped by the shared preprocessing when
     *       normalize is true. Unlike the LARS family, exact collinearity among
     *       the remaining columns is tolerated by coordinate descent (the
     *       minimizer may be non-unique; convergence is unaffected).
     */
    TCDLASSO_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                Eigen::Map<Eigen::MatrixXd>& D,
                Eigen::Map<Eigen::VectorXd>& y,
                const SolverOptions& options = SolverOptions{});

    /**
     * @brief Default constructor for serialization or deferred initialization.
     * @note X_ must be set via reconnect() before running any algorithm.
     */
    TCDLASSO_Solver();

    /** @brief Virtual destructor (X_ pointer is not owned, so not deleted). */
    virtual ~TCDLASSO_Solver() override = default;

    /** @brief Deleted copy constructor and assignment. */
    TCDLASSO_Solver(const TCDLASSO_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TCDLASSO_Solver& operator=(const TCDLASSO_Solver&) = delete;

    /** @brief Defaulted move constructor. */
    TCDLASSO_Solver(TCDLASSO_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TCDLASSO_Solver& operator=(TCDLASSO_Solver&&) = delete;

    // ============================================================================
    // Main Algorithm (Overrides Pure Virtual)
    // ============================================================================

    /**
     * @brief Compute the T-CDLASSO solution path with early stopping based on the
     * dummy threshold.
     *
     * @param T_stop Number of dummies for early stopping threshold
     *               (0 = full solution path over a geometric lambda1 grid).
     * @param early_stop If true, stop when count_active_dummies_ >= T_stop.
     *
     * @note Internal state is updated in-place; supports warm-starts (a later
     * call with a larger T_stop continues the path from the ledger top /
     * lambda_current_) and serialization. One path step is recorded per
     * dummy-count crossing, at the knot, carrying the certified signed
     * support of the segment below it (CERTIFIED mode); simultaneous entries
     * within the lambda tolerance are recorded at a single step
     * (dummies_at_step_ then advances by more than one).
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    /**
     * @brief Convert enum to human-readable string for reporting/logging.
     *
     * @return Static string name.
     */
    std::string solverTypeToString() const override {
        switch(algo_type_) {
            case SolverTypeCdBased::TCDLASSO:      return "TCDLASSO";
            case SolverTypeCdBased::TCDENET:    return "TCDENET";
            case SolverTypeCdBased::TCDIENET:   return "TCDIENET";
            case SolverTypeCdBased::TCDTIKNET: return "TCDTIKNET";
            default:                           return "UnknownSolver";
        }
    }

    // ============================================================================
    // Specific Getters and Setters for TCDLASSO
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
     * @brief Set the relative lambda tolerance: tie tolerance of the certified
     * localization (knots closer than tol * lambda are recorded as one batch)
     * and width stop of the LEGACY bisection.
     * @note Must be called before executeStep(); typical range 1e-6 - 1e-2.
     */
    void setLambdaRelTol(double tol);

    /** @brief Get the relative lambda tolerance (tie tolerance). */
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

    /**
     * @brief Select the crossing-localization mode (default: certified).
     * @param certified true = CERTIFIED (certificates, closed-form
     *        continuation, records at the knot), false = LEGACY (probe
     *        bisection with support-nesting stop). Must be set before the
     *        first executeStep(); switching mid-path re-certifies the current
     *        state at the next call.
     */
    void setCertifiedLocalization(bool certified) {
        localization_ = certified ? LocalizationMode::CERTIFIED
                                  : LocalizationMode::LEGACY;
    }

    /** @brief True if the solver localizes crossings with certificates. */
    bool isCertifiedLocalization() const noexcept {
        return localization_ == LocalizationMode::CERTIFIED;
    }

    /** @brief True once certification became unavailable along the path
     *  (A0-w boundary) and the solver fell back to LEGACY localization;
     *  records from the fallback carry gap/kkt = -1 in getRecordGaps() /
     *  getRecordKktViolations(). */
    bool isCertifiedDegraded() const noexcept { return certified_degraded_; }

    /** @brief Certified-mode diagnostics (zeros in LEGACY mode). */
    struct CertifiedDiagnostics {
        long certified_solves{0};   // certifyPoint() calls
        long certify_stalls{0};     // points that stalled (lambda at a knot) and were nudged
        long brackets_kf{0};        // knot-free brackets
        long brackets_db{0};        // dummy-bounded brackets
        long brackets_sk{0};        // single-knot brackets
        long brackets_cut{0};       // unresolved brackets that were cut
        long continuations{0};      // closed-form continuations computed
    };

    /** @brief Get the certified-mode diagnostics. */
    CertifiedDiagnostics getCertifiedDiagnostics() const noexcept {
        return CertifiedDiagnostics{certified_solves_, certify_stalls_,
                                    brackets_kf_, brackets_db_, brackets_sk_,
                                    brackets_cut_, continuations_};
    }

    /** @brief Duality gap (Def. 9) of every recorded step (certified mode). */
    const std::vector<double>& getRecordGaps() const noexcept { return record_gap_; }

    /** @brief KKT violation (Def. 8) of every recorded step (certified mode). */
    const std::vector<double>& getRecordKktViolations() const noexcept { return record_kkt_; }

    /**
     * @brief Certified signed support of the ledger top (the lower point of
     * the last recorded crossing): unified indices and aligned signs.
     * Empty before the first record or in LEGACY mode.
     */
    std::pair<std::vector<std::size_t>, std::vector<int>> getCertifiedSupport() const {
        return {frontier_top_.support, frontier_top_.sign};
    }

    /** @brief TSolver_Base::memoryBytes() plus the lambda sequence, penalty and
     *  CD state, the Gram rows, the certified frontier, the continuation cache
     *  (both kept after executeStep(), not serialized) and the step records. */
    std::size_t memoryBytes() const noexcept override;

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
        int localization = static_cast<int>(localization_);

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
            CEREAL_NVP(sweep_cap_hits_),
            // Certified localization state (resumption carries the ledger
            // top and the spare certified points below it; Prop. 4).
            cereal::make_nvp("localization", localization),
            CEREAL_NVP(frontier_top_),
            CEREAL_NVP(frontier_below_),
            CEREAL_NVP(frontier_init_),
            CEREAL_NVP(certified_degraded_),
            CEREAL_NVP(certified_solves_),
            CEREAL_NVP(certify_stalls_),
            CEREAL_NVP(brackets_kf_),
            CEREAL_NVP(brackets_db_),
            CEREAL_NVP(brackets_sk_),
            CEREAL_NVP(brackets_cut_),
            CEREAL_NVP(continuations_),
            CEREAL_NVP(record_gap_),
            CEREAL_NVP(record_kkt_)
        );

        algo_type_ = static_cast<SolverTypeCdBased>(algo_type);
        penalty_kind_ = static_cast<PenaltyKind>(penalty_kind);
        localization_ = static_cast<LocalizationMode>(localization);

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
     * @brief Save the current TCDLASSO solver/model state to file (binary
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
     * @brief Load TCDLASSO solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TCDLASSO_Solver object connected to X and D.
     */
    static TCDLASSO_Solver load(const std::string& filename,
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
     * (e.g. TCDIENET group sums) after deserialization.
     */
    virtual void ensureCdState();

    /** @brief Build penalty_diag_ = lambda2 * diag(K) over the unified space.
     *  Virtual: specialized descendants provide their structural diagonal. */
    virtual void configurePenaltyDiag();

    /**
     * @brief Apply lambda2 * K * (delta e_j) to the maintained penalty state.
     * Matrix-passing core: updates l2kb_ (O(1) for NONE/DIAGONAL,
     * O(nnz(K col j)) for SPARSE). Virtual: specialized descendants maintain
     * their own compressed state (TCDIENET group mode: one group-sum scalar, O(1)).
     */
    virtual void penaltyRankUpdate(std::size_t j, double delta);

    /**
     * @brief Quadratic penalty gradient component [lambda2 * K * beta]_j of
     * the current coefficients. Matrix-passing core: reads the maintained
     * l2kb_ vector. Virtual: specialized descendants compute it from their
     * compressed state (TCDIENET group mode: lambda2 * sigma_m / p_m).
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

    // ============================================================================
    // Certified localization (T-CD theory note, Part II; §6 pseudocode)
    // ============================================================================

    /** @brief The LEGACY crossing loop (probe bisection, support-nesting stop). */
    void executeStepLegacy(std::size_t T_stop, long point_budget,
                           std::size_t active_cap);

    /** @brief The CERTIFIED crossing loop (Algorithm 4: frontier / ledger). */
    void executeStepCertified(std::size_t T_stop, long point_budget,
                              std::size_t active_cap);

    /**
     * @brief lambda2 * K_ij over the unified index space (penalty pair hook
     * for the active-set Hessian H_AA = X_A^T X_A + lambda2 K_AA). Virtual:
     * structurally specialized descendants (group / folded kinds) override.
     */
    virtual double penaltyPair(std::size_t i, std::size_t j) const;

    /**
     * @brief out = lambda2 * K_{:,A} v_A over the unified index space
     * (penalty part of the continuation's data pass). Virtual as penaltyPair.
     */
    virtual void penaltyMatVec(const std::vector<std::size_t>& A,
                               const Eigen::VectorXd& vA,
                               Eigen::VectorXd& out) const;

    /** @brief Zero the maintained penalty state (l2kb_ and descendants'
     *  compressed state) before a rebuild from beta_cd_. */
    virtual void zeroPenaltyState();

    /** @brief Rebuild the maintained penalty state from beta_cd_ over the
     *  current support (zeroPenaltyState + penaltyRankUpdate per active). */
    void rebuildPenaltyState();

    /**
     * @brief Lower bound mu on lambda_min(H) when known (A0-s): the diagonal
     * penalty floor for DIAGONAL kinds; 0 = unknown (restricted radius used).
     */
    virtual double strongConvexityMu() const;

    /** @brief x_i^T x_j from the Gram cache when both are slotted, else a dot. */
    double dataGram(std::size_t i, std::size_t j) const;

    /** @brief Coefficient of sqrt(2G) in the OUT / interval-screen tests:
     *  sqrt(H_jj) in general, ||x_j|| for diagonal penalty kinds (sharper,
     *  see the implementation note). */
    double outTestCoefficient(std::size_t j) const;

    /**
     * @brief Duality gap of the live iterate at lambda1 (Def. 9, augmented
     * form): requires fresh r_ and correlations_ = X^T r over ALL columns
     * (a full-pass solveAtLambda leaves them so). Also returns ||c||_inf of
     * the penalized correlation.
     */
    double currentGap(double lambda1, double& c_inf);

    /**
     * @brief Certification solve at lambda1 (Algorithm 1, CERTIFY): CD to
     * tolerance, full pass, gap, OUT/IN statuses (Def. 10) with the global
     * (A0-s) or restricted (Lemma 2(d)) radius; tightens until no
     * coordinate is undecided (Def. 11) or the round budget is hit (STALL:
     * lambda1 is within the tie tolerance of a knot, Lemma 3/4).
     */
    bool certifyPoint(double lambda1, CertifiedPoint& P);

    /**
     * @brief certifyPoint with the nudge rule: on STALL move lambda1 down by
     * twice the tie tolerance (staying above lambda_floor) and retry.
     * @return true on success (lambda1 holds the certified level).
     */
    bool certifyPointNudged(double& lambda1, double lambda_floor,
                            CertifiedPoint& P);

    /**
     * @brief Closed-form continuation from a certified point (Algorithm 2):
     * H_AA from the Gram cache + penalty pairs, one Cholesky, two solves,
     * one data pass; candidate event times; t_next and the attaining
     * candidates. Not defined for the conventional point.
     */
    bool continuation(const CertifiedPoint& P, bool downward,
                      Continuation& C);

    /** @brief continuation() through the transient cache (keyed by lambda
     *  and direction; support sizes must match). */
    bool continuationCached(const CertifiedPoint& P, bool downward,
                            Continuation& C);

    /**
     * @brief Bracket certificate (Algorithm 3): KF (Def. 12, from above or
     * below), DB (Prop. 3), SK (Def. 13 item 3), else UNRESOLVED with the
     * two first knots for the cut. `C_down` is the downward continuation of
     * P_hi if already computed (nullptr: computed here).
     */
    BracketResult certifyBracket(const CertifiedPoint& P_hi,
                                 const CertifiedPoint& P_lo,
                                 const Continuation* C_down);

    /**
     * @brief Install a certified point as the live CD iterate: beta_cd_,
     * actives_, penalty state, Gram q, lambda_current_ (count_active_dummies_
     * is NOT touched: it tracks the last recorded step).
     */
    void loadIterate(const CertifiedPoint& P);

    /**
     * @brief Install the continuation's solution at decrease t (warm start
     * for the next solve below the point): beta = beta* + t d on the support.
     */
    void loadContinuation(const CertifiedPoint& P, const Continuation& C,
                          double t);

    /**
     * @brief Tie rule (§4 of the note): extend the record at `knot` down over
     * every further knot within the tie tolerance, merging supports; P_lo is
     * replaced by the new lower point. Points of frontier_below_ at or above
     * the new P_lo are discarded (covered).
     */
    void closeBatch(CertifiedPoint& P_lo, double knot);

    /** @brief Jump predictor from exact penalized correlations (R5). */
    double predictNextCrossingFrom(const Eigen::VectorXd& c_pen, double lambda_cur);

    /** @brief Insert a certified point into frontier_below_ (sorted desc). */
    void frontierInsert(CertifiedPoint P);

    /** @brief Switch to the LEGACY localization for the remaining path
     *  (certification unavailable: A0-w boundary) and finish the call there;
     *  further records are flagged uncertified. */
    void fallBackToLegacy(std::size_t T_stop, long point_budget,
                          std::size_t active_cap);

    /** @brief Align record_gap_/record_kkt_ with the step count using -1
     *  sentinels for uncertified (LEGACY-fallback) records. */
    void padUncertifiedRecords();
};

// ===================================================================================

} /* End of namespace trex::tsolvers::ad::linear_model::cd_based */


#endif /* End of TSOLVERS_LINEAR_MODEL_CD_BASED_TCDLASSO_SOLVER_HPP */
