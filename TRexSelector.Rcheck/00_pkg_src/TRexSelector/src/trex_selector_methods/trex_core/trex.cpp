// ===================================================================================
// trex.cpp
// ===================================================================================
/**
 * @file trex.cpp
 *
 * @brief Implementation of TRexSelector variable selection algorithm.
 */
// ===================================================================================

// std includes
#include <utils/logging/logger.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

// T-Rex header
#include <trex_selector_methods/trex_core/trex.hpp>

// utils includes
#include <utils/fp_classify/fp_classify.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::trex_core
namespace trex::trex_selector_methods::trex_core {

// Namespace aliases
namespace dn = trex::trex_selector_methods::utils::data_normalizer;
namespace dg = trex::trex_selector_methods::utils::dummy_generator;
namespace fpc = trex::utils::fp_classify;
namespace wsm = trex::trex_selector_methods::utils::warm_start_manager;
namespace mm = trex::trex_selector_methods::utils::memmap_manager;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;
namespace er = trex::trex_selector_methods::utils::experiment_runner;

// ===================================================================================

// Debug macro (NDEBUG builds compile this to a no-op)
#ifndef NDEBUG
    #define TREX_DEBUG_PRINT(msg) \
      do { \
        if (verbose_) { \
            TREX_INFO( "T-Rex Selector: [DEBUG]" << (msg) ); \
        } \
      } while (0)
#else
    #define TREX_DEBUG_PRINT(msg) ((void)0)
#endif


// ===================================================================================
// Constructor
// ===================================================================================

TRexSelector::TRexSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    double tFDR,
    TRexControlParameter trex_control,
    int seed,
    bool verbose
) :
    X_(&X),
    y_(y),
    n_(static_cast<std::size_t>(X_->rows())),
    p_(static_cast<std::size_t>(X_->cols())),
    tFDR_(tFDR),
    trex_ctrl_([&]() {
        auto ctrl = trex_control;
        ctrl.autoConfigResources();
        return ctrl;
    }()),
    seed_(seed),
    verbose_(verbose),

    // --- Helper modules ---
    dummy_gen_(n_,
               trex_ctrl_.dummy_distribution,
               seed_,
               verbose_),
    warm_start_mgr_(trex_ctrl_.K,
                    trex_ctrl_.use_memory_mapping
                        ? wsm::WarmStartMode::SERIALIZED
                        : wsm::WarmStartMode::IN_MEMORY,
                    verbose_)
{
    // 1. Validate
    // -----------------------------------------------------
    validateTRexParameters();

    // 2. Data Preprocessing: Normalize X, center y
    // -----------------------------------------------------
    dn::centerY(y_, norm_params_);
    dn::centerAndL2NormalizeX(*X_, norm_params_, eps_, verbose_);
    X_is_normalized_ = true;

    // 3. Initialize memory-mapped D manager (if requested)
    // -----------------------------------------------------
    if (trex_ctrl_.use_memory_mapping) {
        const std::size_t max_dummies = trex_ctrl_.max_dummy_multiplier * p_;
        const bool shared = (trex_ctrl_.lloop_strategy == LLoopStrategy::PERMUTATION
                          || trex_ctrl_.lloop_strategy == LLoopStrategy::PERMUTATION_DIRECT
                          || trex_ctrl_.lloop_strategy == LLoopStrategy::DIRECT);
        memmap_mgr_ = std::make_unique<mm::MemmapManager>(
            n_, max_dummies, trex_ctrl_.K, shared, verbose_);
        memmap_mgr_->initialize();
    }

    // 4. Create experiment runner
    // -----------------------------------------------------
    experiment_runner_ = std::make_unique<er::ExperimentRunner>(
        *X_, y_, dummy_gen_, warm_start_mgr_,
        memmap_mgr_.get()  // nullptr if not using memmap
    );

    // 5. Initialize containers
    // -----------------------------------------------------
    selected_var_.resize(0);
    selected_indices_.clear();
    voting_grid_.resize(0);
    FDP_hat_mat_.resize(0, 0);
    Phi_mat_.resize(0, 0);
    R_mat_.resize(0, 0);
    Phi_prime_.resize(0);

    // 6. Resolve cached permutation base seed (used only by the PERMUTATION
    //    L-loop strategy, but resolved unconditionally for simplicity).
    // -----------------------------------------------------
    permutation_base_seed_ = (seed_ >= 0)
        ? static_cast<unsigned int>(seed_)
        : std::random_device{}();
}


// ===================================================================================
// Destructor
// ===================================================================================

TRexSelector::~TRexSelector() {
    // Helper modules clean up via their own destructors.

    // Restore X to original scale (necessary since we use a Map).
    if (X_is_normalized_) {
        dn::denormalizeX(*X_, norm_params_);
        X_is_normalized_ = false;
    }
}


 // ===================================================================================
 // Validation
 // ===================================================================================

void TRexSelector::validateTRexParameters() const {

    if (n_ == 0 || p_ == 0) {
        throw std::invalid_argument("Feature matrix X must have non-zero dimensions.");
    }

    if (static_cast<std::size_t>(y_.size()) != n_) {
        throw std::invalid_argument(
            "Response vector y must have length equal to the number of rows in X. "
            "Got y.size() = " + std::to_string(y_.size()) +
            ", X.rows() = " + std::to_string(n_));
    }

    if (!fpc::allFinite(y_)) {
        throw std::invalid_argument("Response vector y contains NaN or Inf values.");
    }

    if (tFDR_ < 0.0 || tFDR_ > 1.0) {
        throw std::invalid_argument(
            "Target FDR tFDR must be in [0, 1]. Got: " + std::to_string(tFDR_));
    }

    if (trex_ctrl_.K < 2) {
        throw std::invalid_argument(
            "Number of random experiments K must be >= 2. Got: "
            + std::to_string(trex_ctrl_.K));
    }

    if (trex_ctrl_.max_dummy_multiplier == 0) {
        throw std::invalid_argument(
            "max_dummy_multiplier must be >= 1. Got: "
            + std::to_string(trex_ctrl_.max_dummy_multiplier));
    }

    // Warn when stagnation detection is disabled for greedy solvers.
    if (!trex_ctrl_.tloop_stagnation_stop) {
        const sd::SolverTypeForTRex st = trex_ctrl_.solver_type;
        const bool is_greedy =
            st == sd::SolverTypeForTRex::TSTEPWISE  ||
            st == sd::SolverTypeForTRex::TOMP       ||
            st == sd::SolverTypeForTRex::TGP        ||
            st == sd::SolverTypeForTRex::TACGP      ||
            st == sd::SolverTypeForTRex::TMP        ||
            st == sd::SolverTypeForTRex::TNCGMP     ||
            st == sd::SolverTypeForTRex::TOOLS      ||
            st == sd::SolverTypeForTRex::TAFS;
        if (is_greedy) {
            TREX_WARN( "[TRexSelector Warning] tloop_stagnation_stop is disabled "
                         "for a greedy solver. Greedy solvers tend to run into a noise trap "
                         "and select until all allowed dummies are selected. "
                         "Convergence without stagnation detection is not guaranteed. Consider "
                         "re-enabling it." );
        }
    }
}


// ===================================================================================
// Main T-Rex algorithm: select() method
// ===================================================================================

TRexSelector::SelectionResult TRexSelector::select() {

    // 1. Voting grid and optimization point
    // -----------------------------------------------------
    voting_grid_ = createVotingGrid();
    const std::size_t v_len = voting_grid_.size();
    setPercentOptPoint(v_len, trex_ctrl_.opt_threshold);

    // 1a. Variant-specific lifecycle hook (default: no-op).
    // -----------------------------------------------------
    onSelectBegin();

    // 2. L-Loop: calibrate num_dummies_ with T_stop = 1
    // -----------------------------------------------------
    if (verbose_) { printProgress("L-loop: Calibrating number of dummies..."); }

    Eigen::VectorXd FDP_hat;
    er::ExperimentResults exp_results;
    runLLoop(FDP_hat, exp_results);

    if (verbose_) {
        printProgress("L-loop: converged with " +
                      std::to_string(num_dummies_) + " dummies.\n");
    }

    // 3. T-Loop: calibrate T_stop >= 1
    // -----------------------------------------------------
    runTLoop(FDP_hat, exp_results);

    // 4. Final phi_T_mat and Phi_prime
    // -----------------------------------------------------
    Phi_prime_ = computePhiPrime(exp_results.phi_T_mat,
                                 exp_results.Phi,
                                 num_dummies_);

    // 5. Variable selection & Store results
    // -----------------------------------------------------
    SelectionResult result = selectedVariables(
        FDP_hat_mat_,
        Phi_mat_,
        voting_grid_,
        T_stop_
    );

    result.T_stop = T_stop_;
    result.num_dummies = num_dummies_;
    result.FDP_hat_mat = FDP_hat_mat_;
    result.Phi_mat = Phi_mat_;
    result.Phi_prime = Phi_prime_;
    result.voting_grid = voting_grid_;

    storeResults(result);
    if (verbose_) {
        printProgress("Selection complete. Selected "
                      + std::to_string(getNumSelected()) + " variables.");
    }

    // 5a. Variant-specific result widening / cleanup hook (default: returns
    //     `result` unchanged). Fires while X is still normalized so subclasses
    //     can rely on `*X_` reflecting the operating data.
    // -----------------------------------------------------
    result = finalizeSelectionResult(std::move(result));

    // 6. Cleanup helpers
    // -----------------------------------------------------
    warm_start_mgr_.cleanup();
    if (memmap_mgr_) { memmap_mgr_->cleanup(); }

    // 7. Restore X to original scale
    // -----------------------------------------------------
    if (verbose_) { printProgress("Denormalizing X to original scale.\n"); }
    if (X_is_normalized_) {
        dn::denormalizeX(*X_, norm_params_);
        X_is_normalized_ = false;
    }

    return result;
}


// ===================================================================================
// buildRunnerConfig — populates ExperimentRunnerConfig
// ===================================================================================

er::ExperimentRunnerConfig TRexSelector::buildRunnerConfig(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool use_warm_start,
    er::ExperimentStrategy strategy,
    std::size_t seed_factor,
    std::size_t existing_cols_on_disk) const
{
    er::ExperimentRunnerConfig cfg;
    cfg.K                     = trex_ctrl_.K;
    cfg.num_dummies           = num_dummies;
    cfg.T_stop                = T_stop;
    cfg.p                     = p_;
    cfg.strategy              = strategy;
    cfg.seed_factor           = seed_factor;
    cfg.existing_cols_on_disk = existing_cols_on_disk;
    cfg.solver_type           = trex_ctrl_.solver_type;
    cfg.solver_params         = trex_ctrl_.solver_params;
    cfg.use_warm_start        = use_warm_start;
    cfg.use_memory_mapping    = trex_ctrl_.use_memory_mapping;
    cfg.max_outer_threads     = trex_ctrl_.max_outer_threads;
    cfg.max_inner_threads     = trex_ctrl_.max_inner_threads;
    cfg.verbose               = verbose_;
    cfg.eps                   = eps_;

    return cfg;
}


// ===================================================================================
// Voting grid helpers
// ===================================================================================

Eigen::VectorXd TRexSelector::createVotingGrid() const {
    const double v_start = 0.5;
    const double v_end = 1.0 - eps_;
    const double v_step = 1.0 / static_cast<double>(trex_ctrl_.K);

    const std::size_t v_len = static_cast<std::size_t>(
        std::floor((v_end - v_start) / v_step)) + 1;

    Eigen::VectorXd grid(v_len);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(v_len); ++i) {
        grid(i) = v_start + static_cast<double>(i) * v_step;
    }
    return grid;
}


void TRexSelector::setPercentOptPoint(std::size_t v_len, double threshold) {
    opt_point_ = 0;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(v_len); ++i) {
        if (voting_grid_(i) <= threshold) { opt_point_ = i; }
        else { break; }
    }
}


// ===================================================================================
// FDP Computation
// ===================================================================================

Eigen::VectorXd TRexSelector::computeFDPHat(
    const Eigen::VectorXd& voting_grid,
    const Eigen::VectorXd& Phi,
    const Eigen::VectorXd& Phi_prime) const
{
    const std::size_t v_len = voting_grid.size();
    const std::size_t p = static_cast<std::size_t>(Phi.size());

    Eigen::VectorXd fdp_hat = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(v_len));

    const Eigen::VectorXd one_minus_phi_prime =
        Eigen::VectorXd::Ones(static_cast<Eigen::Index>(p)) - Phi_prime;

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(v_len); ++i) {
        const double threshold = voting_grid(i);

        const Eigen::Array<bool, Eigen::Dynamic, 1> selected_mask =
            (Phi.array() > threshold);

        const std::size_t num_sel_var = selected_mask.count();

        if (num_sel_var == 0) {
            fdp_hat(i) = 0.0;
            continue;
        }

        const double sum_deflated =
            selected_mask.select(one_minus_phi_prime.array(), 0.0).sum();

        fdp_hat(i) = std::max(
            0.0,
            std::min(1.0, sum_deflated / static_cast<double>(num_sel_var))
        );

    }

    return fdp_hat;
}


// ===================================================================================
// Phi Prime Computation
// ===================================================================================

Eigen::VectorXd TRexSelector::computePhiPrime(
    const Eigen::MatrixXd& phi_T_mat,
    const Eigen::VectorXd& Phi,
    std::size_t num_dummies) const
{
    std::size_t current_T = phi_T_mat.cols();

    Eigen::VectorXd av_num_var_sel = phi_T_mat.colwise().sum();

    // Filter: keep only rows where Phi > 0.5
    std::vector<std::size_t> phi_geg_fifty_indices;
    phi_geg_fifty_indices.reserve(p_);
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p_); ++j) {
        if (Phi(j) > 0.5) {
            phi_geg_fifty_indices.push_back(j);
        }
    }

    Eigen::VectorXd delta_av_num_var_sel = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(current_T));
    for (const auto idx : phi_geg_fifty_indices) {
        delta_av_num_var_sel += phi_T_mat.row(static_cast<Eigen::Index>(idx)).transpose();
    }

    Eigen::MatrixXd phi_T_mat_mod = phi_T_mat;
    if (current_T > 1) {
        for (Eigen::Index t = static_cast<Eigen::Index>(current_T) - 1; t > 0; --t) {
            delta_av_num_var_sel(t) -= delta_av_num_var_sel(t - 1);
            phi_T_mat_mod.col(t) -= phi_T_mat_mod.col(t - 1);
        }
    }

    Eigen::VectorXd phi_scale = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(current_T));
    for (Eigen::Index t = 0; t < static_cast<Eigen::Index>(current_T); ++t) {
        if (delta_av_num_var_sel(t) > eps_) {
            double num_remaining_orig = static_cast<double>(p_) - av_num_var_sel(t);
            double num_remaining_dummies = static_cast<double>(num_dummies)
                                           - static_cast<double>(t);
            double ratio_remaining = num_remaining_orig / num_remaining_dummies;
            phi_scale(t) = 1.0 - (ratio_remaining / delta_av_num_var_sel(t));
        }
    }

    return (phi_T_mat_mod * phi_scale).cwiseMax(0.0).cwiseMin(1.0);
}


// ===================================================================================
// Variable Selection
// ===================================================================================

TRexSelector::SelectionResult TRexSelector::selectedVariables(
    const Eigen::MatrixXd& FDP_hat_mat,
    const Eigen::MatrixXd& Phi_mat,
    const Eigen::VectorXd& voting_grid,
    std::size_t T_stop) const
{
    if (FDP_hat_mat.rows() == 0 || Phi_mat.rows() == 0) {
        if (verbose_) {
            printProgress(
                "Warning: No valid T-steps recorded. Returning empty selection."
            );
        }
        SelectionResult result;
        result.selected_var = Eigen::VectorXi::Zero(static_cast<Eigen::Index>(p_));
        result.v_thresh = 1.0;
        result.R_mat = Eigen::MatrixXd::Zero(0, 0);
        return result;
    }

    std::size_t safe_T = std::min(T_stop, static_cast<std::size_t>(FDP_hat_mat.rows()));
    if (safe_T == 0) {
        SelectionResult result;
        result.selected_var = Eigen::VectorXi::Zero(static_cast<Eigen::Index>(p_));
        result.v_thresh = 1.0;
        result.R_mat = Eigen::MatrixXd::Zero(0, 0);
        return result;
    }

    // Drop the last T-row (the stopping-criterion boundary step) when T > 1.
    // Matches R reference: FDP_hat_mat[-T_stop, , drop = FALSE]
    if (safe_T > 1) { --safe_T; }

    Eigen::MatrixXd FDP_use = FDP_hat_mat.topRows(safe_T);
    Eigen::MatrixXd Phi_use = Phi_mat.topRows(safe_T);
    const std::size_t n_rows = FDP_use.rows();
    const std::size_t n_cols = FDP_use.cols();

    // Build R_mat (consensus size matrix)
    Eigen::MatrixXd R_mat = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_rows), static_cast<Eigen::Index>(n_cols));
    for (Eigen::Index VV = 0; VV < static_cast<Eigen::Index>(n_cols); ++VV) {
        const double threshold = voting_grid(VV);
        for (Eigen::Index TT = 0; TT < static_cast<Eigen::Index>(n_rows); ++TT) {
            R_mat(TT, VV) = static_cast<double>(
                (Phi_use.row(TT).array() > threshold).count()
            );
        }
    }

    // Find max in R_mat where FDP <= tFDR
    double val_max = -1.0;
    Eigen::Index ind_max_row = 0;
    Eigen::Index ind_max_col = 0;

    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n_cols); ++j) {
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_rows); ++i) {
            if (FDP_use(i, j) <= tFDR_ && R_mat(i, j) >= val_max) {
                val_max = R_mat(i, j);
                ind_max_row = i;
                ind_max_col = j;
            }
        }
    }

    if (fpc::isinf(val_max) || val_max < 0) {
        SelectionResult result;
        result.selected_var = Eigen::VectorXi::Zero(static_cast<Eigen::Index>(p_));
        result.v_thresh = 1.0;
        result.R_mat = R_mat;
        return result;
    }

    double best_threshold = voting_grid(ind_max_col);
    Eigen::VectorXi selected_var = Eigen::VectorXi::Zero(static_cast<Eigen::Index>(p_));
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p_); ++j) {
        if (Phi_use(ind_max_row, j) > best_threshold) {
            selected_var(j) = 1;
        }
    }

    SelectionResult result;
    result.selected_var = selected_var;
    result.v_thresh = best_threshold;
    result.R_mat = R_mat;
    return result;
}


// ===================================================================================
// Per-Iteration Step + Accumulator Hooks (overridable)
// ===================================================================================

TRexSelector::StepView TRexSelector::evaluateStep(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool use_warm_start,
    er::ExperimentStrategy strategy,
    std::size_t seed_factor,
    std::size_t existing_on_disk)
{
    auto cfg = buildRunnerConfig(
        num_dummies, T_stop, use_warm_start, strategy,
        seed_factor, existing_on_disk);

    StepView view;
    view.exp_results = experiment_runner_->run(cfg);
    view.Phi         = view.exp_results.Phi;
    view.Phi_prime   = computePhiPrime(view.exp_results.phi_T_mat,
                                       view.exp_results.Phi,
                                       num_dummies);
    view.FDP_hat     = computeFDPHat(voting_grid_, view.Phi, view.Phi_prime);
    return view;
}


void TRexSelector::initTAccumulators(std::size_t capacity, std::size_t v_len) {
    Phi_mat_     = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(capacity),
                                         static_cast<Eigen::Index>(p_));
    FDP_hat_mat_ = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(capacity),
                                         static_cast<Eigen::Index>(v_len));
}


void TRexSelector::growTAccumulators(std::size_t old_capacity,
                                     std::size_t new_capacity)
{
    const auto old_cap = static_cast<Eigen::Index>(old_capacity);
    const auto new_cap = static_cast<Eigen::Index>(new_capacity);
    Phi_mat_.conservativeResize(new_cap, Eigen::NoChange);
    FDP_hat_mat_.conservativeResize(new_cap, Eigen::NoChange);
    Phi_mat_.bottomRows(new_cap - old_cap).setZero();
    FDP_hat_mat_.bottomRows(new_cap - old_cap).setZero();
}


void TRexSelector::appendTRow(Eigen::Index row_idx, const StepView& view) {
    Phi_mat_.row(row_idx)     = view.Phi;
    FDP_hat_mat_.row(row_idx) = view.FDP_hat;
}


void TRexSelector::trimTAccumulators(Eigen::Index n_rows) {
    Phi_mat_.conservativeResize(n_rows, Eigen::NoChange);
    FDP_hat_mat_.conservativeResize(n_rows, Eigen::NoChange);
}


// ===================================================================================
// Variant Hooks (default implementations)
// ===================================================================================

void TRexSelector::onSelectBegin() {
    // Default: no-op. Subclasses (e.g. GVS) may override to perform one-shot
    // setup before the L-loop runs.
}


void TRexSelector::prepareDummiesForLStep(LStepContext& ctx) {
    // Default: dispatch on strategy and perform the canonical T-Rex dummy
    // work via `dummy_gen_`. Whenever fresh dummy data is produced we
    // invalidate `warm_start_mgr_` so the experiment runner builds new
    // solver state for this iteration.
    ctx.existing_on_disk = 0;

    switch (ctx.strategy) {
        case LLoopStrategy::SKIPL: {
            // One-shot: generate K dummy matrices sized L_max * p.
            dummy_gen_.generateAndStore(trex_ctrl_.K,
                                        ctx.num_dummies,
                                        /*seed_factor=*/0);
            break;
        }
        case LLoopStrategy::STANDARD: {
            // Fresh dummies each iteration → invalidate warm-start.
            dummy_gen_.generateAndStore(trex_ctrl_.K,
                                        ctx.num_dummies,
                                        ctx.L_iter);
            warm_start_mgr_.invalidate();
            break;
        }
        case LLoopStrategy::HCONCAT: {
            if (ctx.L_iter == 1) {
                // Initial dummies: K matrices of size n x p.
                dummy_gen_.generateAndStore(trex_ctrl_.K,
                                            p_,
                                            /*seed_factor=*/0);
                ctx.existing_on_disk = 0;
            } else {
                // Append p new columns to each stored matrix:
                // [D_k_old | D_k_new].
                dummy_gen_.expandStored(p_, ctx.L_iter);
                warm_start_mgr_.invalidate();
                ctx.existing_on_disk = (ctx.L_iter - 1) * p_;
            }
            break;
        }
        case LLoopStrategy::PERMUTATION: {
            if (ctx.L_iter == 1) {
                Eigen::MatrixXd base = dummy_gen_.generate(p_, permutation_base_seed_);
                dummy_gen_.storeBaseDummies(std::move(base), permutation_base_seed_);
            } else {
                // Expand: generate p new columns and append to existing base.
                const unsigned int expansion_seed =
                    dummygen::mix_seed(permutation_base_seed_, ctx.L_iter);
                Eigen::MatrixXd new_cols = dummy_gen_.generate(p_, expansion_seed);

                // getPermuted(0, ...) returns the base as-is.
                Eigen::MatrixXd expanded =
                    dummy_gen_.getPermuted(0, ctx.num_dummies - p_);
                expanded.conservativeResize(
                    Eigen::NoChange,
                    static_cast<Eigen::Index>(ctx.num_dummies));
                expanded.rightCols(p_) = new_cols;
                dummy_gen_.storeBaseDummies(std::move(expanded),
                                            permutation_base_seed_);
            }
            warm_start_mgr_.invalidate();
            break;
        }
        case LLoopStrategy::DIRECT:
        case LLoopStrategy::PERMUTATION_DIRECT: {
            // Seed-based, on-the-fly: nothing to pre-generate.
            warm_start_mgr_.invalidate();
            break;
        }
        default: {
            throw std::invalid_argument(
                "TRexSelector::prepareDummiesForLStep: unsupported L-loop strategy.");
        }
    }
}


TRexSelector::SelectionResult TRexSelector::finalizeSelectionResult(
    SelectionResult&& base_result)
{
    // Default: pass-through. Subclasses (e.g. GVS) may override to widen
    // into a derived result struct and to release variant-specific
    // resources allocated in `onSelectBegin()` / the L-loop.
    return base_result;
}


// ===================================================================================
// L-Loop Strategy Dispatcher
// ===================================================================================

void TRexSelector::runLLoop(
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results)
{
    switch (trex_ctrl_.lloop_strategy) {
        case LLoopStrategy::SKIPL: {
            runLLoopCalibration_SKIPL(FDP_hat, exp_results);
            break;
        }
        case LLoopStrategy::STANDARD: {
            runLLoopCalibration_Standard(FDP_hat, exp_results);
            break;
        }
        case LLoopStrategy::HCONCAT: {
            runLLoopCalibration_HCONCAT(FDP_hat, exp_results);
            break;
        }
        case LLoopStrategy::PERMUTATION: {
            runLLoopCalibration_Permutation(FDP_hat, exp_results);
            break;
        }
        case LLoopStrategy::DIRECT: // Same as PERMUTATION_DIRECT but without base dummy matrix
        case LLoopStrategy::PERMUTATION_DIRECT: {
            runLLoopCalibration_Direct(FDP_hat, exp_results);
            break;
        }
        default: {
            throw std::invalid_argument("Invalid L-loop strategy.");
        }
    }
}


// ===================================================================================
// L-Loop: SKIPL
// ===================================================================================

void TRexSelector::runLLoopCalibration_SKIPL(
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results)
{
    num_dummies_ = trex_ctrl_.max_dummy_multiplier * p_;
    T_stop_ = 1;

    if (verbose_) {
        printProgress("L-Loop: SKIPL strategy. Fixed dummies = "
                      + std::to_string(num_dummies_));
    }

    // Per-step dummy work via virtual hook (subclasses may override; default
    // generates K dummy matrices sized num_dummies via dummy_gen_).
    LStepContext lctx;
    lctx.L_iter      = trex_ctrl_.max_dummy_multiplier;
    lctx.num_dummies = num_dummies_;
    lctx.strategy    = LLoopStrategy::SKIPL;
    prepareDummiesForLStep(lctx);

    // Per-step body via virtual hook (subclasses may inject deflation).
    StepView view = evaluateStep(num_dummies_, T_stop_, /*use_warm_start=*/false,
                                 er::ExperimentStrategy::Standard,
                                 /*seed_factor=*/0,
                                 /*existing_on_disk=*/0);
    exp_results = std::move(view.exp_results);
    FDP_hat     = std::move(view.FDP_hat);
}


// ===================================================================================
// L-Loop: Unified driver
// ===================================================================================

void TRexSelector::runLLoopCalibration(
    const LLoopPolicyContext& ctx,
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results)
{
    while (dummy_multiplier_LL_ <= trex_ctrl_.max_dummy_multiplier &&
          (FDP_hat.size() == 0 ||
           FDP_hat(static_cast<Eigen::Index>(opt_point_)) > tFDR_)) {

        num_dummies_ = dummy_multiplier_LL_ * p_;

        std::size_t existing_on_disk = 0;
        ctx.policy(dummy_multiplier_LL_, num_dummies_, existing_on_disk);

        const std::size_t seed_factor =
            ctx.seed_factor_uses_LL ? dummy_multiplier_LL_ : 0;

        // Per-step body via virtual hook (subclasses may inject deflation).
        StepView view = evaluateStep(num_dummies_, T_stop_, /*use_warm_start=*/false,
                                     ctx.strategy, seed_factor, existing_on_disk);
        exp_results = std::move(view.exp_results);
        FDP_hat     = std::move(view.FDP_hat);

        if (verbose_) {
            printProgress("Appended dummies: " + std::to_string(num_dummies_));
        }

        ++dummy_multiplier_LL_;
    }

    dummy_multiplier_LL_ =
        (dummy_multiplier_LL_ > 1) ? (dummy_multiplier_LL_ - 1) : 1;
}


// ===================================================================================
// L-Loop: STANDARD
// ===================================================================================

void TRexSelector::runLLoopCalibration_Standard(
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results
    )
{
    T_stop_ = 1;
    dummy_multiplier_LL_ = 1;
    FDP_hat.resize(0);

    LLoopPolicyContext ctx;
    ctx.strategy            = er::ExperimentStrategy::Standard;
    ctx.seed_factor_uses_LL = true;
    ctx.policy = [this](std::size_t LL,
                        std::size_t num_dummies,
                        std::size_t& existing_on_disk_out) {
        // Per-iteration dummy work via virtual hook (subclasses may override).
        LStepContext lctx;
        lctx.L_iter      = LL;
        lctx.num_dummies = num_dummies;
        lctx.strategy    = LLoopStrategy::STANDARD;
        prepareDummiesForLStep(lctx);
        existing_on_disk_out = lctx.existing_on_disk;
    };

    runLLoopCalibration(ctx, FDP_hat, exp_results);
}


// ===================================================================================
// L-Loop: HCONCAT
// ===================================================================================

void TRexSelector::runLLoopCalibration_HCONCAT(
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results
    )
{
    T_stop_ = 1;
    dummy_multiplier_LL_ = 1;
    FDP_hat.resize(0);

    LLoopPolicyContext ctx;
    ctx.strategy            = er::ExperimentStrategy::Standard;
    ctx.seed_factor_uses_LL = true;
    ctx.policy = [this](std::size_t LL,
                        std::size_t num_dummies,
                        std::size_t& existing_on_disk_out) {
        // Per-iteration dummy work via virtual hook (subclasses may override).
        // The default impl performs the L=1 initial generation and the
        // L>1 horizontal expansion internally.
        LStepContext lctx;
        lctx.L_iter      = LL;
        lctx.num_dummies = num_dummies;
        lctx.strategy    = LLoopStrategy::HCONCAT;
        prepareDummiesForLStep(lctx);
        existing_on_disk_out = lctx.existing_on_disk;
    };

    runLLoopCalibration(ctx, FDP_hat, exp_results);
}


// ===================================================================================
// L-Loop: PERMUTATION
// ===================================================================================

void TRexSelector::runLLoopCalibration_Permutation(
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results
    )
{
    T_stop_ = 1;
    dummy_multiplier_LL_ = 1;
    FDP_hat.resize(0);

    LLoopPolicyContext ctx;
    ctx.strategy            = er::ExperimentStrategy::Permutation;
    ctx.seed_factor_uses_LL = false;
    ctx.policy = [this](std::size_t LL,
                        std::size_t num_dummies,
                        std::size_t& existing_on_disk_out) {
        // Per-iteration dummy work via virtual hook (subclasses may override).
        // The default impl uses `permutation_base_seed_` (resolved in the
        // constructor) to generate or expand the base dummy matrix.
        LStepContext lctx;
        lctx.L_iter      = LL;
        lctx.num_dummies = num_dummies;
        lctx.strategy    = LLoopStrategy::PERMUTATION;
        prepareDummiesForLStep(lctx);
        existing_on_disk_out = lctx.existing_on_disk;
    };

    runLLoopCalibration(ctx, FDP_hat, exp_results);

    if (verbose_) {
        printProgress("L-loop converged: " + std::to_string(num_dummies_)
                      + " dummies (base stored, K permutations on-demand).");
    }
}


// ===================================================================================
// L-Loop: DIRECT
// ===================================================================================

void TRexSelector::runLLoopCalibration_Direct(
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results)
{
    T_stop_ = 1;
    dummy_multiplier_LL_ = 1;
    FDP_hat.resize(0);

    // The L-loop dispatcher routes both PERMUTATION_DIRECT and DIRECT to this
    // method; mirror that disambiguation when forwarding to the hook.
    const LLoopStrategy hook_strategy =
        (trex_ctrl_.lloop_strategy == LLoopStrategy::DIRECT)
            ? LLoopStrategy::DIRECT
            : LLoopStrategy::PERMUTATION_DIRECT;

    LLoopPolicyContext ctx;
    ctx.strategy            = er::ExperimentStrategy::Direct;
    ctx.seed_factor_uses_LL = false;
    ctx.policy = [this, hook_strategy](std::size_t LL,
                                       std::size_t num_dummies,
                                       std::size_t& existing_on_disk_out) {
        // Per-iteration dummy work via virtual hook (subclasses may override).
        LStepContext lctx;
        lctx.L_iter      = LL;
        lctx.num_dummies = num_dummies;
        lctx.strategy    = hook_strategy;
        prepareDummiesForLStep(lctx);
        existing_on_disk_out = lctx.existing_on_disk;
    };

    runLLoopCalibration(ctx, FDP_hat, exp_results);

    if (verbose_) {
        printProgress("L-loop converged: " + std::to_string(num_dummies_)
                      + " dummies (Direct generation from seeds).");
    }
}


// ===================================================================================
// T-Loop Calibration
// ===================================================================================

void TRexSelector::runTLoop(
    Eigen::VectorXd& FDP_hat,
    er::ExperimentResults& exp_results
    )
{
    if (verbose_) {
        printProgress("T-loop: Calibrating stopping threshold (T_stop)...");
    }

    const Eigen::Index v_len = static_cast<Eigen::Index>(voting_grid_.size());

    std::size_t max_T = trex_ctrl_.use_max_T_stop
        ? std::min(
            num_dummies_,
            static_cast<std::size_t>(std::ceil(static_cast<double>(n_) / 2.0))
          )
        : num_dummies_;

    // Capacity-doubling accumulators (delegated to virtual hooks so subclasses
    // can grow side accumulators in lock-step).
    std::size_t capacity = std::min<std::size_t>(20, max_T);
    initTAccumulators(capacity, static_cast<std::size_t>(v_len));

    std::size_t prev_num_selected = 0;
    std::size_t stagnant_iterations = 0;

    // Determine L-loop strategy
    er::ExperimentStrategy exp_strategy;
    switch (trex_ctrl_.lloop_strategy) {
        case LLoopStrategy::PERMUTATION:
        case LLoopStrategy::PERMUTATION_DIRECT:
            exp_strategy = er::ExperimentStrategy::Permutation;
            break;
        case LLoopStrategy::DIRECT:
            exp_strategy = er::ExperimentStrategy::Direct;
            break;
        default:
            exp_strategy = er::ExperimentStrategy::Standard;
            break;
    }

    // Seed row 0 with the L-loop result (T = 1) to mirror R's
    // `acc <- list(FDP = matrix(FDP_hat, nrow=1), Phi = matrix(step$Phi, nrow=1))`
    // before the T-loop. This avoids re-running the (expensive) T = 1
    // experiment as the first T-loop iteration and guarantees a valid row
    // exists even when the while-guard rejects the first check.
    StepView seed;
    seed.exp_results = exp_results;
    seed.Phi         = exp_results.Phi;
    seed.FDP_hat     = FDP_hat;
    appendTRow(0, seed);
    Eigen::Index row_idx = 1;

    // Initialise stagnation baseline from the seed row so the first body
    // iteration's comparison matches the pre-refactor behaviour.
    {
        Eigen::Index v_star_idx = 0;
        for (Eigen::Index i = 0; i < v_len; ++i) {
            if (FDP_hat(i) <= tFDR_) v_star_idx = i;
        }
        const double v_star = (v_star_idx < v_len) ? voting_grid_(v_star_idx) : 0.0;
        if (v_star > eps_) {
            prev_num_selected =
                static_cast<std::size_t>((exp_results.Phi.array() > v_star).count());
        }
    }

    while (FDP_hat(v_len - 1) <= tFDR_ && T_stop_ < max_T) {

        ++T_stop_;

        // Per-step body via virtual hook (subclasses may inject deflation).
        StepView view = evaluateStep(num_dummies_, T_stop_, /*use_warm_start=*/true,
                                     exp_strategy,
                                     /*seed_factor=*/dummy_multiplier_LL_,
                                     /*existing_on_disk=*/0);
        exp_results = view.exp_results;
        FDP_hat     = view.FDP_hat;

        // Find optimal v* at current T
        Eigen::Index v_star_idx = 0;
        for (Eigen::Index i = 0; i < v_len; ++i) {
            if (FDP_hat(i) <= tFDR_) v_star_idx = i;
        }
        double v_star = (v_star_idx < v_len) ? voting_grid_(v_star_idx) : 0.0;

        std::size_t num_selected = 0;
        if (v_star > eps_) {
            num_selected = (view.Phi.array() > v_star).count();
        }

        // Stagnation check
        if (trex_ctrl_.tloop_stagnation_stop && T_stop_ > 1) {
            if (num_selected <= prev_num_selected) {
                ++stagnant_iterations;
                if (stagnant_iterations >= trex_ctrl_.tloop_max_stagnant_steps) {
                    if (verbose_) {
                        printProgress("T-loop EARLY STOP: Stagnation detected");
                    }
                    break;
                }
            } else {
                stagnant_iterations = 0;
            }
        }
        prev_num_selected = num_selected;

        // Diagnostic output
        if (verbose_) {
            std::ostringstream oss;
            oss << "T = " << std::setw(3) << T_stop_
                << " | v* = " << v_star
                << ", R = " << num_selected
                << ", FDP in [" << FDP_hat.minCoeff() << ", " << FDP_hat.maxCoeff() << "]";
            if (stagnant_iterations > 0) {
                oss << " (stagnant: " << stagnant_iterations << ")";
            }
            printProgress(oss.str());
        }

        // Dynamic capacity growth
        if (static_cast<std::size_t>(row_idx) >= capacity) {
            const std::size_t new_capacity =
                std::min(capacity * 2, max_T);
            growTAccumulators(capacity, new_capacity);
            capacity = new_capacity;
        }

        appendTRow(row_idx, view);
        ++row_idx;
    }

    // Trim
    trimTAccumulators(row_idx);

    if (verbose_) {
        std::string reason;
        if (stagnant_iterations >= trex_ctrl_.tloop_max_stagnant_steps) {
            reason = " (early stop: stagnation)";
        }
        else if (T_stop_ >= max_T) { reason = " (reached max_T)"; }
        else { reason = " (FDP > target FDR)"; }
        printProgress(
            "T-loop converged at T_stop = " +
            std::to_string(T_stop_) + reason + "\n"
        );
    }
}


// ===================================================================================
// Utility methods
// ===================================================================================

void TRexSelector::storeResults(const SelectionResult& result) {
    selected_var_ = result.selected_var;
    voting_threshold_ = result.v_thresh;
    R_mat_ = result.R_mat;
    updateSelectedIndices();
}


void TRexSelector::updateSelectedIndices() {
    selected_indices_.clear();
    for (Eigen::Index j = 0; j < selected_var_.size(); ++j) {
        if (selected_var_(j) == 1) {
            selected_indices_.push_back(j);
        }
    }
}


void TRexSelector::printProgress(const std::string& message) const {
    if (verbose_) {
        TREX_INFO( "[T-Rex Selector] " << message << "\n");
    }
}


}
