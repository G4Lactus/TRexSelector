// ===================================================================================
// trex.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_CORE_HPP
#define TREX_SELECTOR_METHODS_TREX_CORE_HPP
// ===================================================================================
/**
 * @file trex.hpp
 *
 * @brief T-Rex Selector for FDR-controlled variable selection.
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// TRex includes
#include <trex_selector_methods/trex_utils/trex_data_normalizer.hpp>
#include <trex_selector_methods/trex_utils/trex_solver_dispatch.hpp>
#include <trex_selector_methods/trex_utils/trex_warm_start_manager.hpp>
#include <trex_selector_methods/trex_utils/trex_memmap_manager.hpp>
#include <trex_selector_methods/trex_utils/trex_dummy_generator.hpp>
#include <trex_selector_methods/trex_utils/trex_experiment_runner.hpp>

// Dummy generation distribution type
#include <utils/datageneration/utils_dummygen.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::trex_core
namespace trex::trex_selector_methods::trex_core {

// Namespace alias
namespace dn = trex::trex_selector_methods::utils::data_normalizer;
namespace dg = trex::trex_selector_methods::utils::dummy_generator;
namespace wsm = trex::trex_selector_methods::utils::warm_start_manager;
namespace mm = trex::trex_selector_methods::utils::memmap_manager;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;
namespace er = trex::trex_selector_methods::utils::experiment_runner;
namespace dummygen = trex::utils::datageneration::dummygen;

// ===================================================================================
// Constants
// ===================================================================================

/**
 * @brief Sentinel value for "auto-estimate this correlation coefficient from X".
 *
 * @details Any value outside [−1, 1] works. We use −2.0 because it is safe under
 *          -ffast-math (unlike quiet_NaN, which breaks std::isnan under
 *          -ffinite-math-only).
 */
inline constexpr double AUTO_ESTIMATE_CORRELATION = -2.0;

// ===================================================================================
// Enums & Control Structures
// ===================================================================================

/**
 * @brief L-loop calibration strategies.
 */
enum class LLoopStrategy {
    SKIPL,                 // Skip L-loop: fixed dummies = max_dummy_multiplier * p.
    STANDARD,              // Fresh dummies at each L-loop iteration (conservative).
    HCONCAT,               // Horizontally expand dummy matrices (faster, same result).
    PERMUTATION,           // Deterministic permutations of base dummy matrix.
    PERMUTATION_DIRECT,    // Seed-based permutations without base dummy matrix.
    DIRECT                 // Seed-based dummy generation without base matrix.
};


/**
 * @brief Algorithmic control parameters for T-Rex Selector.
 */
struct TRexControlParameter {

    // ====================================
    // Experiment Configuration
    // ====================================

    /** @brief Number of random experiments (default: 20). */
    std::size_t K = 20;

    /** @brief Maximum number of dummy variables as a multiple of p (default: 10). */
    std::size_t max_dummy_multiplier = 10;

    /** @brief If true, limit T_stop to ceiling(n/2) (default: true). */
    bool use_max_T_stop = true;

    /** @brief Optimization threshold (default: 0.75). */
    double opt_threshold = 0.75;

    // ===================================
    // Dummy Generation
    // ===================================

    /** @brief Distribution for generating dummies (default: Normal). */
    dummygen::Distribution dummy_distribution = dummygen::Distribution::Normal();

    // ===================================
    // L-Loop Strategy
    // ===================================

    /** @brief Strategy for L-loop calibration (default: HCONCAT). */
    LLoopStrategy lloop_strategy = LLoopStrategy::HCONCAT;

    // ==================================
    // T-Loop Early Stopping
    // ==================================

    /** @brief If true, perform early stopping if support set stagnates (default: true). */
    bool tloop_stagnation_stop = true;

    /** @brief Number of stagnant steps to trigger early T-loop stopping.
     *  Recommendation: Pick from {3, 5, 7} (default: 5).
     */
    std::size_t tloop_max_stagnant_steps = 5;

    // =================================
    // Parallel random experiments
    // =================================

    /** @brief If true, allows parallel execution of random experiments (default: false). */
    bool parallel_rnd_experiments = false;

    /** @brief Computed number of threads for the outer loop (experiments) (default: 1). */
    int max_outer_threads = 1;

    /** @brief Computed number of threads for the inner loop (solvers/Eigen) (default: 1). */
    int max_inner_threads = 1;

    /** @brief Automatically configure thread resources based on K and hardware. */
    void autoConfigResources() {
#ifdef _OPENMP
        if (parallel_rnd_experiments) {
            int total_cores = omp_get_max_threads();
            int k_exp = static_cast<int>(K);
            if (k_exp > 1) {
                max_outer_threads = std::min(total_cores, k_exp);
                max_inner_threads = std::max(1, total_cores / max_outer_threads);
            } else {
                max_outer_threads = 1;
                max_inner_threads = total_cores;
            }
        } else {
            max_outer_threads = 1;
            max_inner_threads = omp_get_max_threads();
        }
#else
        max_outer_threads = 1;
        max_inner_threads = 1;
#endif
    }

    // ===================================
    // Memory-Mapping
    // ===================================

    /** @brief Enable memory-mapped D storage (default: false). */
    bool use_memory_mapping = false;

    // ===================================
    // Solver Configuration
    // ===================================

    /** @brief Solver type (default: TLARS). */
    sd::SolverTypeForTRex solver_type = sd::SolverTypeForTRex::TLARS;

    /** @brief Bundled hyperparameters for all solver algorithms. */
    sd::SolverHyperparameters solver_params{};

};


// ===================================================================================
// TRexSelector
// ===================================================================================

/**
 * @brief T-Rex Selector for FDR-controlled variable selection.
 *
 * @details
 *  Base class implementing the T-Rex algorithm.
 *  Supports in-memory and memory-mapped data via Eigen::Map.
 *  Serves as base for derived variants
 *  - DA-TRex
 *  - Screen-TRex
 *  - GVS-TRex
 *  - Sparse-PCA-TRex
 *
 *  Key design:
 *    - Does not own X (p >> n, too large to copy).
 *    - Normalizes X in-place, centers y.
 *    - Delegates dummy generation, solver dispatch, warm-start management,
 *      memory mapping, and experiment coordination to helper modules.
 *    - select() returns SelectionResult with all key outputs.
 */
class TRexSelector {
protected:

    // ============================================================
    // Data Members — Input Data
    // ============================================================

    /** @brief Non-owning pointer to design matrix X (n x p). */
    Eigen::Map<Eigen::MatrixXd>* X_;

    /** @brief Owned response vector y (n x 1). */
    Eigen::VectorXd y_{};

    /** @brief Number of observations. */
    const std::size_t n_;

    /** @brief Number of features. */
    const std::size_t p_;

    // ============================================================
    // Data Members — Configuration
    // ============================================================

    /** @brief Target False Discovery Rate. */
    const double tFDR_;

    /** @brief Algorithmic control parameters. */
    const TRexControlParameter trex_ctrl_;

    /** @brief Random seed. */
    const int seed_;

    /** @brief Verbose output flag. */
    const bool verbose_;

    /** @brief Numerical tolerance.
     *  @note  Initialized to 0 here; the constructor always overwrites this
     *         with `solver_params.tol` (default 1e-6) or, when tol <= 0,
     *         with `std::numeric_limits<double>::epsilon()`. */
    double eps_{0.0};

    // ============================================================
    // Data Members — Normalization (via trex_data_normalizer)
    // ============================================================

    /** @brief Normalization parameters for X and y. */
    dn::NormalizationParams norm_params_;

    /** @brief Tracks whether X is currently in normalized state.
     *  It guards against double denormalization (select() + destructor).
     */
    bool X_is_normalized_{false};

    // ============================================================
    // Data Members — Results
    // ============================================================

    /** @brief Binary support vector (0/1). */
    Eigen::VectorXi selected_var_;

    /** @brief Indices of selected variables. */
    std::vector<std::size_t> selected_indices_;

    /** @brief Stopping time. */
    std::size_t T_stop_{0};

    /** @brief Number of dummy variables used. */
    std::size_t num_dummies_{0};

    /** @brief Voting grid used. */
    Eigen::VectorXd voting_grid_;

    /** @brief Voting threshold. */
    double voting_threshold_{0.0};

    /** @brief Estimated FDP matrix. */
    Eigen::MatrixXd FDP_hat_mat_;

    /** @brief Relative occurrence matrix. */
    Eigen::MatrixXd Phi_mat_;

    /** @brief Selection frequency matrix. */
    Eigen::MatrixXd R_mat_;

    /** @brief Adjusted relative occurrences (p x 1). */
    Eigen::VectorXd Phi_prime_;

    /** @brief Optimization point for dummy calibration. */
    std::size_t opt_point_;

    /** @brief Current dummy multiplier in L-loop. */
    std::size_t dummy_multiplier_LL_{0};

    /** @brief Cached base seed for the PERMUTATION L-loop strategy.
     *
     *  Resolved once in the constructor so that `prepareDummiesForLStep`
     *  observes the same value across every L-iteration regardless of
     *  whether `seed_ >= 0` (deterministic) or `seed_ < 0` (random_device).
     */
    unsigned int permutation_base_seed_{0};

    // ============================================================
    // Helper Module Members
    // ============================================================

    /** @brief Dummy generation (all strategies). */
    dg::DummyGenerator dummy_gen_;

    /** @brief Warm-start management (in-memory or serialized). */
    wsm::WarmStartManager warm_start_mgr_;

    /** @brief Memory-mapped D file management (nullptr if not used). */
    std::unique_ptr<mm::MemmapManager> memmap_mgr_;

    /** @brief Experiment coordination (K-experiment loop). */
    std::unique_ptr<er::ExperimentRunner> experiment_runner_;


public:

    // ============================================================
    // Result Structure
    // ============================================================

    /** @brief Result from variable selection. */
    struct SelectionResult {

        /** @brief Binary support vector (0/1, length p). */
        Eigen::VectorXi selected_var;

        /** @brief Voting threshold applied to Phi to obtain the selection. */
        double v_thresh;

        /** @brief Selection frequency matrix (T_stop × v_len). */
        Eigen::MatrixXd R_mat;

        /** @brief Calibrated stopping time T_stop. */
        std::size_t T_stop;

        /** @brief Total number of dummy variables used. */
        std::size_t num_dummies;

        /** @brief Dummy multiplier L determined by the L-loop. */
        std::size_t dummy_factor_L;

        /** @brief Estimated FDP matrix (T_stop × v_len). */
        Eigen::MatrixXd FDP_hat_mat;

        /** @brief Relative occurrence matrix Phi_T (T_stop × p). */
        Eigen::MatrixXd Phi_mat;

        /** @brief Adjusted relative occurrences Phi' (p × 1). */
        Eigen::VectorXd Phi_prime;

        /** @brief Voting grid V = [0.5, 0.5+1/K, ..., 1-eps]. */
        Eigen::VectorXd voting_grid;

        virtual ~SelectionResult() = default;
    };


    // ============================================================
    // Constructor / Destructor
    // ============================================================

    /**
     * @brief Construct a TRexSelector object.
     *
     * @param X Design matrix (n x p) — Eigen::Map, not copied.
     * @param y Response vector (n x 1) — Eigen::Map, copied internally.
     * @param tFDR Target False Discovery Rate (default: 0.1).
     * @param trex_control Algorithmic control parameters.
     * @param seed Random seed (< 0 for non-deterministic).
     * @param verbose Enable verbose output (default: true).
     */
    TRexSelector(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::VectorXd>& y,
                 double tFDR = 0.1,
                 TRexControlParameter trex_control = TRexControlParameter(),
                 int seed = -1,
                 bool verbose = true);

    /** @brief Virtual destructor. */
    virtual ~TRexSelector();

    /** @brief Delete copy constructor */
    TRexSelector(const TRexSelector&) = delete;

    /** @brief Delete copy assignment operator. */
    TRexSelector& operator=(const TRexSelector&) = delete;

    /** @brief Delete move constructor. */
    TRexSelector(TRexSelector&&) = delete;

    /** @brief Delete move assignment operator. */
    TRexSelector& operator=(TRexSelector&&) = delete;


    // ============================================================
    // Main Method
    // ============================================================

    /**
     * @brief Run T-Rex selection algorithm.
     *
     * @return SelectionResult containing selected variables and
     *         diagnostics.
     */
    virtual SelectionResult select();


    // ============================================================
    // Public Getters — Configuration
    // ============================================================

    /** @brief Get target FDR level. */
    double getTFDR() const noexcept { return tFDR_; }

    /** @brief Get number of random experiments K. */
    std::size_t getK() const noexcept { return trex_ctrl_.K; }

    /** @brief Get maximum dummy multiplier used for model calibrations. */
    std::size_t getMaxNumDummies() const noexcept { return trex_ctrl_.max_dummy_multiplier; }

    /** @brief Check if maximum stopping time is used. */
    bool getMaxTStop() const noexcept { return trex_ctrl_.use_max_T_stop; }

    /** @brief Check if stagnation check is enabled. */
    bool getStagnationCheck() const noexcept { return trex_ctrl_.tloop_stagnation_stop; }

    // ============================================================
    // Public Getters — Normalization
    // ============================================================

    /** @brief Get means of the design matrix columns. */
    const Eigen::VectorXd& getXMeans() const noexcept { return norm_params_.X_means; }

    /** @brief Get L2 norms of the design matrix columns. */
    const Eigen::VectorXd& getXL2Norms() const noexcept { return norm_params_.X_l2norms; }

    /** @brief Get mean of the response vector. */
    double getYMean() const noexcept { return norm_params_.y_mean; }

    // ============================================================
    // Public Getters — Results
    // ============================================================

    /** @brief Get selected variables as a binary vector. */
    const Eigen::VectorXi& getSelectedVar() const noexcept { return selected_var_; }

    /** @brief Get indices of selected variables. */
    const std::vector<std::size_t>& getSelectedIndices() const noexcept {
        return selected_indices_; }

    /** @brief Get number of selected variables. */
    std::size_t getNumSelected() const noexcept { return selected_indices_.size(); }

    /** @brief Get stopping time T_stop. */
    std::size_t getTStop() const noexcept { return T_stop_; }

    /** @brief Get voting grid vector V = [0.5, 0.5+1/K, ..., 1-eps]. */
    const Eigen::VectorXd& getVotingGrid() const noexcept { return voting_grid_; }

    /** @brief Get voting threshold. */
    double getVotingThreshold() const noexcept { return voting_threshold_; }

    /** @brief Get FDP hat matrix. */
    const Eigen::MatrixXd& getFDPHatMat() const noexcept { return FDP_hat_mat_; }

    /** @brief Get Phi matrix. */
    const Eigen::MatrixXd& getPhiMat() const noexcept { return Phi_mat_; }

    /** @brief Get selection frequency matrix R. */
    const Eigen::MatrixXd& getRMat() const noexcept { return R_mat_; }

    /** @brief Get Phi prime vector. */
    const Eigen::VectorXd& getPhiPrime() const noexcept { return Phi_prime_; }

    /** @brief Get dummy multiplier L. */
    std::size_t getDummyMultiplierL() const noexcept { return dummy_multiplier_LL_; }

protected:

    // ============================================================
    // Core T-Rex Computation — Voting & FDP
    // ============================================================

    /** @brief Generate voting grid V = [0.5, 0.5 + 1/K, ..., 1-eps]. */
    Eigen::VectorXd createVotingGrid() const;

    /**
     * @brief Set optimization point.
     *
     * @param v_len Length of the voting grid.
     * @param threshold Voting threshold to determine optimization point (default: 0.75).
     */
    void setPercentOptPoint(std::size_t v_len, double threshold = 0.75);

    /**
     * @brief Compute Phi_prime (calibration parameter).
     *
     * @param phi_T_mat T x p matrix of relative occurrences.
     * @param Phi p x 1 vector of relative occurrences at T_stop.
     * @param num_dummies Number of dummy variables used (for scaling).
     *
     * @return Phi-prime vector (p x 1) for use in FDP estimation.
     */
    Eigen::VectorXd computePhiPrime(
        const Eigen::MatrixXd& phi_T_mat,
        const Eigen::VectorXd& Phi,
        std::size_t num_dummies) const;

    /**
     * @brief Compute FDP estimates for voting grid.
     *
     * @param voting_grid Voting grid vector.
     * @param Phi p x 1 vector of relative occurrences at T_stop.
     * @param Phi_prime p x 1 vector of calibrated occurrences.
     * @return FDP estimates for each voting threshold.
     */
    Eigen::VectorXd computeFDPHat(
        const Eigen::VectorXd& voting_grid,
        const Eigen::VectorXd& Phi,
        const Eigen::VectorXd& Phi_prime) const;

    /**
     * @brief Select variables with FDR control.
     *
     * @param FDP_hat_mat Matrix of FDP estimates.
     * @param Phi_mat Matrix of relative occurrences.
     * @param voting_grid Voting grid vector.
     * @param T_stop Stopping time.
     *
     * @return Selection result containing selected variables and related information.
     */
    SelectionResult selectedVariables(
        const Eigen::MatrixXd& FDP_hat_mat,
        const Eigen::MatrixXd& Phi_mat,
        const Eigen::VectorXd& voting_grid,
        std::size_t T_stop) const;


    // ============================================================
    // Per-Iteration Step + Accumulator Hooks (overridable)
    // ============================================================

    /**
     * @brief Per-L-iteration dummy-preparation context.
     *
     * @details
     *  Passed to `prepareDummiesForLStep()` once per L-loop iteration. The
     *  hook is expected to populate / refresh whatever per-experiment dummy
     *  state the subsequent `evaluateStep()` call will consume, and to write
     *  back `existing_on_disk` for HCONCAT memmap accounting.
     */
    struct LStepContext {
        /** Current dummy multiplier (1-based; equals max_dummy_multiplier
         *  for the SKIPL one-shot path). */
        std::size_t L_iter = 1;

        /** Total number of dummy columns this iteration (`L_iter * p_`). */
        std::size_t num_dummies = 0;

        /** Active L-loop strategy. */
        LLoopStrategy strategy = LLoopStrategy::STANDARD;

        /** Output: number of dummy columns already persisted on disk.
         *  Used by the HCONCAT memmap path to skip rewriting prefix
         *  columns; set to 0 by all other strategies. */
        std::size_t existing_on_disk = 0;
    };

    /**
     * @brief Lifecycle hook fired once at the start of `select()`, after
     *        voting-grid setup and before the L-loop.
     *
     * @details
     *  Default base implementation: no-op. Subclasses may override to perform
     *  one-shot setup (e.g. precomputing cluster structure) that the L-loop
     *  / T-loop drivers depend on.
     */
    virtual void onSelectBegin();

    /**
     * @brief Per-L-iteration dummy-preparation hook.
     *
     * @details
     *  Default base implementation: dispatches on `ctx.strategy` and performs
     *  the canonical T-Rex dummy work via `dummy_gen_` (and invalidates
     *  `warm_start_mgr_` when fresh dummy data has been produced). Subclasses
     *  that bypass `dummy_gen_` (e.g. cluster-aware MVN draws in GVS) may
     *  override this hook to populate their own per-experiment dummy state
     *  while still inheriting the L-loop control flow.
     *
     *  Subclasses MUST set `ctx.existing_on_disk` to the appropriate value
     *  (typically 0 unless they explicitly implement an HCONCAT memmap
     *  variant).
     *
     * @param ctx Per-iteration context (input + output).
     */
    virtual void prepareDummiesForLStep(LStepContext& ctx);

    /**
     * @brief Final result post-processing hook.
     *
     * @details
     *  Called from `select()` after `selectedVariables()` and `storeResults()`,
     *  but before warm-start / memmap cleanup and before X is denormalized.
     *  Subclasses may override to widen the returned object (e.g. into a
     *  derived result struct) and to perform variant-specific cleanup of
     *  any extra resources allocated in `onSelectBegin()` or in the L-loop.
     *
     *  Default base implementation: returns `base_result` unchanged.
     *
     * @param base_result The base SelectionResult produced by the standard pipeline.
     * @return The (possibly widened) SelectionResult to return from `select()`.
     */
    virtual SelectionResult finalizeSelectionResult(SelectionResult&& base_result);

    /**
     * @brief Bundle of per-step outputs shared by the L-loop and T-loop drivers.
     *
     * @details
     *  Subclasses (e.g. dependency-aware variants) override `evaluateStep` to
     *  inject extra processing — typically a deflation step — between the
     *  experiment runner and the FDP computation.
     *  They populate `FDP_hat` with the canonical 1D channel that the base
     *  drivers rely on for the L-loop reference-point guard, the T-loop
     *  edge guard, and stagnation tracking. Side data (e.g. per-rho 2D matrices)
     *  is stashed in subclass-private member state.
     */
    struct StepView {
        /** @brief Raw experiment-runner output (Phi, phi_T_mat, etc.). */
        er::ExperimentResults exp_results;

        /** @brief Canonical FDP estimates for the voting grid (length v_len). */
        Eigen::VectorXd FDP_hat;

        /** @brief Canonical Phi vector (length p). For DA-BT variants this is
         *  the deflated Phi at the calibration rho-level. */
        Eigen::VectorXd Phi;

        /** @brief Canonical Phi_prime vector (length p). */
        Eigen::VectorXd Phi_prime;
    };

    /**
     * @brief Run one experiment step and produce the canonical FDP/Phi view.
     *
     * @details
     *  Default base implementation: runs the experiment runner, computes
     *  `Phi_prime = computePhiPrime(...)` and `FDP_hat = computeFDPHat(...)`.
     *  Subclasses may override to insert a deflation step.
     *
     *  Subclasses MUST populate `view.FDP_hat`, `view.Phi`, and
     *  `view.Phi_prime` with vectors that the L-loop reference-point guard,
     *  the T-loop edge guard, and the base accumulator append logic can use.
     *
     * @param num_dummies        Number of dummies for this step.
     * @param T_stop             Stopping time for this step.
     * @param use_warm_start     Pass-through to the experiment runner.
     * @param strategy           Experiment strategy.
     * @param seed_factor        Seed factor for the runner config.
     * @param existing_on_disk   `existing_cols_on_disk` for HCONCAT memmap.
     * @return Populated StepView.
     */
    virtual StepView evaluateStep(
        std::size_t num_dummies,
        std::size_t T_stop,
        bool use_warm_start,
        er::ExperimentStrategy strategy,
        std::size_t seed_factor,
        std::size_t existing_on_disk);

    /**
     * @brief Initialise T-loop accumulators with the given capacity.
     *
     * @details
     *  Default base: zero-initialises `Phi_mat_` (capacity x p) and
     *  `FDP_hat_mat_` (capacity x v_len). Subclasses may extend to allocate
     *  their own per-rho accumulators.
     */
    virtual void initTAccumulators(std::size_t capacity, std::size_t v_len);

    /**
     * @brief Grow T-loop accumulators to a larger capacity (capacity-doubling).
     *
     * @details
     *  Default base: `conservativeResize` on `Phi_mat_` / `FDP_hat_mat_` and
     *  zero the newly added bottom rows.
     */
    virtual void growTAccumulators(std::size_t old_capacity,
                                   std::size_t new_capacity);

    /**
     * @brief Append the current step's row at the given index.
     *
     * @details
     *  Default base: writes `view.Phi` and `view.FDP_hat` into row `row_idx`
     *  of `Phi_mat_` and `FDP_hat_mat_`.
     */
    virtual void appendTRow(Eigen::Index row_idx, const StepView& view);

    /**
     * @brief Trim T-loop accumulators to the actually populated rows.
     *
     * @details Default base: `conservativeResize` to `n_rows` rows.
     */
    virtual void trimTAccumulators(Eigen::Index n_rows);


    // ============================================================
    // L-Loop Calibration Strategies
    // ============================================================

    /**
     * @brief Dispatch L-loop strategy.
     *
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results Current experiment results (updated in-place).
     */
    void runLLoop(Eigen::VectorXd& FDP_hat, er::ExperimentResults& exp_results);

    /**
     * @brief Per-iteration policy context for the unified L-loop driver.
     *
     * Bundles the strategy enum, a flag telling the driver whether to forward
     * the current `dummy_multiplier_LL_` as `seed_factor` (vs. fixed 0), and
     * the strategy-specific dummy-generation callable.
     */
    struct LLoopPolicyContext {
        /** Experiment strategy to forward to ExperimentRunnerConfig. */
        er::ExperimentStrategy strategy;
        /** If true, pass `dummy_multiplier_LL_` as `seed_factor`; else 0. */
        bool seed_factor_uses_LL;
        /**
         * @brief Strategy-specific per-iteration dummy work.
         * @param LL Current value of `dummy_multiplier_LL_` (1-based).
         * @param num_dummies `LL * p_` for this iteration.
         * @param existing_on_disk_out Output: columns already on memmap disk
         *        (used by HCONCAT for the runner config); set to 0 if N/A.
         *
         * Contract: the callable must perform any required calls on
         * `dummy_gen_` (e.g. `generateAndStore`, `expandStored`,
         * `storeBaseDummies`) and invoke `warm_start_mgr_.invalidate()`
         * whenever it mutates dummy data that previously cached solvers
         * depend on.
         */
        std::function<void(std::size_t LL,
                           std::size_t num_dummies,
                           std::size_t& existing_on_disk_out)> policy;
    };

    /**
     * @brief Unified L-loop calibration driver shared by the four looping
     *        strategies (STANDARD, HCONCAT, PERMUTATION, DIRECT).
     *
     * Implements the common while-loop skeleton:
     *   - guard `dummy_multiplier_LL_ <= max_dummy_multiplier &&
     *           (FDP_hat empty || FDP_hat(opt_point_) > tFDR_)`,
     *   - compute `num_dummies_ = dummy_multiplier_LL_ * p_`,
     *   - delegate strategy-specific dummy work to `ctx.policy`,
     *   - build runner config + run experiments,
     *   - update `Phi_prime` and `FDP_hat` via `computePhiPrime` /
     *     `computeFDPHat`,
     *   - increment `dummy_multiplier_LL_`.
     *
     * @note SKIPL is intentionally NOT routed through this driver: it has no
     *       loop, performs a single shot with `num_dummies_ =
     *       max_dummy_multiplier * p_`, and never increments
     *       `dummy_multiplier_LL_`.
     *
     * @param ctx Strategy policy context.
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results Current experiment results (updated in-place).
     */
    void runLLoopCalibration(const LLoopPolicyContext& ctx,
                             Eigen::VectorXd& FDP_hat,
                             er::ExperimentResults& exp_results);

    /**
     * @brief Run L-loop calibration using the SKIPL strategy.
     *
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results Current experiment results (updated in-place).
     */
    void runLLoopCalibration_SKIPL(Eigen::VectorXd& FDP_hat, er::ExperimentResults& exp_results);

    /**
     * @brief Run L-loop calibration using the STANDARD strategy.
     *
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results Current experiment results (updated in-place).
     */
    void runLLoopCalibration_Standard(Eigen::VectorXd& FDP_hat, er::ExperimentResults& exp_results);

    /**
     * @brief Run L-loop calibration using the HCONCAT strategy.
     *
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results  Current experiment results (updated in-place).
     */
    void runLLoopCalibration_HCONCAT(Eigen::VectorXd& FDP_hat, er::ExperimentResults& exp_results);

    /**
     * @brief Run L-loop calibration using the PERMUTATION strategy.
     *
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results Current experiment results (updated in-place).
     */
    void runLLoopCalibration_Permutation(Eigen::VectorXd& FDP_hat,
                                         er::ExperimentResults& exp_results);

    /**
     * @brief Run L-loop calibration using the PERMUTATION_DIRECT strategy.
     *
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results Current experiment results (updated in-place).
     */
    void runLLoopCalibration_Direct(Eigen::VectorXd& FDP_hat, er::ExperimentResults& exp_results);


    // ============================================================
    // T-Loop Calibration
    // ============================================================

    /** @brief Run T-loop to determine stopping threshold T_stop.
     *
     * @param FDP_hat Current FDP estimates (updated in-place).
     * @param exp_results Current experiment results (updated in-place).
     */
    void runTLoop(Eigen::VectorXd& FDP_hat, er::ExperimentResults& exp_results);


    // ============================================================
    // Utilities
    // ============================================================

    /** @brief Validate input data dimensions. */
    void validateTRexParameters() const;

    /**
    * @brief Print progress message (if verbose).
    *
    * @param message Message to print.
    */
    void printProgress(const std::string& message) const;

    /**  @brief Update selected_indices_ from selected_var_. */
    void updateSelectedIndices();

    /**
     * @brief Store results after selection.
     *
     * @param result Selection result to store.
     */
    void storeResults(const SelectionResult& result);

    /**
     * @brief Build an ExperimentRunnerConfig from current TRex state.
     *
     * @param num_dummies Number of dummies for this iteration.
     * @param T_stop Stopping threshold.
     * @param use_warm_start Whether to use warm start.
     * @param strategy Experiment strategy.
     * @param seed_factor L-loop iteration multiplier for seed mixing (default: 0).
     * @param existing_cols_on_disk Columns already on disk for HCONCAT memmap path (default: 0).
     *
     * @return Populated ExperimentRunnerConfig.
     */
    er::ExperimentRunnerConfig buildRunnerConfig(
        std::size_t num_dummies,
        std::size_t T_stop,
        bool use_warm_start,
        er::ExperimentStrategy strategy,
        std::size_t seed_factor = 0,
        std::size_t existing_cols_on_disk = 0) const;

// -----------------------------------------------------------------------
}; /* End of class TRexSelector */

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_core */
// ===================================================================================
#endif /* TREX_SELECTOR_METHODS_TREX_CORE_HPP */
