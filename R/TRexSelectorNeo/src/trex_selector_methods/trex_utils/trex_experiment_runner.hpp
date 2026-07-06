// ===================================================================================
// trex_experiment_runner.hpp
// ===================================================================================
#ifndef TREX_TREX_SELECTOR_METHODS_UTILS_EXPERIMENT_RUNNER_HPP
#define TREX_TREX_SELECTOR_METHODS_UTILS_EXPERIMENT_RUNNER_HPP
// ===================================================================================
/**
 * @file trex_experiment_runner.hpp
 *
 * @brief Administration of the K random experiments:
 * - generate dummies
 * - dispatch solvers
 * - aggregate beta paths into phi_T_mat
 *
 * @details
 * A single run() method accepts an ExperimentRunnerConfig and dispatches
 * to the appropriate internal path. The runner holds non-owning references
 * to:
 *      - DummyGenerator&    (generates / retrieves D matrices)
 *      - wsm::WarmStartManager&  (manages solver warm-start state)
 *      - mm::MemmapManager*     (optional, for memory-mapped D storage)
 *      - sd::SolverConfig template fields (from TRexControlParameter)
 *
 *   Ressources are owned by the TRexSelector only.
 *
 *  Split X / D:
 *    - X is passed as Eigen::Map<Eigen::MatrixXd>& (non-owning view).
 *    - D is obtained from DummyGenerator (in-memory) or MemmapManager (memory-mapped).
 *    - Both are passed to dispatchByType() via sd::SolverConfig.
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// utils inclues: OpenMP
#include <utils/openmp/utils_openmp.hpp>

// TRex helper modules
#include <trex_selector_methods/trex_utils/trex_solver_dispatch.hpp>
#include <trex_selector_methods/trex_utils/trex_warm_start_manager.hpp>
#include <trex_selector_methods/trex_utils/trex_memmap_manager.hpp>
#include <trex_selector_methods/trex_utils/trex_dummy_generator.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::experiment_runner
namespace trex::trex_selector_methods::utils::experiment_runner {

// Alias namespaces
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;
namespace dg = trex::trex_selector_methods::utils::dummy_generator;
namespace wsm = trex::trex_selector_methods::utils::warm_start_manager;
namespace mm = trex::trex_selector_methods::utils::memmap_manager;

// ===================================================================================
// ExperimentStrategy: handles how dummies are provided per experiment
// ===================================================================================

enum class ExperimentStrategy {
    Standard,      // Pre-generated K dummy matrices (stored in DummyGenerator).
    Permutation,   // Deterministic row permutations of a base dummy matrix.
    Direct         // On-the-fly seed-based dummy generation per experiment.
};


// ===================================================================================
// ExperimentRunnerConfig: everything the runner needs per invocation
// ===================================================================================

/**
 * @brief Configuration for a single run() invocation.
 */
struct ExperimentRunnerConfig {

    // --- Experiment dimensions ---
    std::size_t K = 0;              // Number of random experiments.
    std::size_t num_dummies = 0;    // Number of dummy columns for this iteration.
    std::size_t T_stop = 0;         // Early stopping threshold.
    std::size_t p = 0;              // Number of original predictors (for phi_T_mat).

    // --- Strategy ---
    ExperimentStrategy strategy = ExperimentStrategy::Standard;

    // --- L-loop seed factor (for deterministic dummy regeneration) ---
    std::size_t seed_factor = 0;    ///< L-loop iteration multiplier for seed mixing.

    // --- HCONCAT: number of previously written columns already on disk ---
    std::size_t existing_cols_on_disk = 0;  ///< For HCONCAT memmap: columns from prior L.

    // --- Solver ---
    sd::SolverTypeForTRex solver_type = sd::SolverTypeForTRex::TLARS;
    sd::SolverHyperparameters solver_params{};

    // --- Scaling ---
    /** @brief Internal column-scaling mode applied by each solver.
     *  L2 (default) preserves prior behaviour; ZSCORE makes solvers z-score
     *  their own columns (needed for augmented-EN consistency). */
    trex::tsolvers::ScalingMode scaling_mode = trex::tsolvers::ScalingMode::L2;

    // --- Warm start ---
    bool use_warm_start = false;

    // --- Memory mapping ---
    bool use_memory_mapping = false;

    // --- Parallelism ---
    int max_outer_threads = 1;
    int max_inner_threads = 1;

    // --- Diagnostics ---
    bool verbose = false;
    double eps = std::numeric_limits<double>::epsilon();

    // --- Screen-TRex beta collection ---
    /** @brief If true, also accumulate per-experiment beta sums and non-zero dummy betas.
     *  Used by ScreenTRexSelector for bootstrap and mean-coefficient computation.
     *  Only meaningful when T_stop == 1. */
    bool collect_screen_betas = false;
};


// ===================================================================================
// ExperimentResults: output of a run() invocation
// ===================================================================================

/** @brief Aggregated results from K random experiments. */
struct ExperimentResults {

    /** @brief Relative occurrence matrix (p × T_stop). */
    Eigen::MatrixXd phi_T_mat;

    /** @brief Phi vector = phi_T_mat.col(T_stop - 1). */
    Eigen::VectorXd Phi;

    /** @brief Sum of original-feature betas over K experiments (p × 1).
     *  Non-empty only when ExperimentRunnerConfig::collect_screen_betas is true. */
    Eigen::VectorXd beta_sums;

    /** @brief All non-zero dummy betas across K experiments.
     *  Non-empty only when ExperimentRunnerConfig::collect_screen_betas is true. */
    std::vector<double> dummy_betas;
};


// ===================================================================================
// ExperimentRunner
// ===================================================================================

class ExperimentRunner {
public:

    // ==========================================================================
    // Construction
    // ==========================================================================

    /**
     * @brief Construct an ExperimentRunner.
     *
     * @param X Non-owning reference to normalized design matrix.
     * @param y Non-owning reference to centered response vector.
     * @param dummy_gen DummyGenerator for creating / retrieving D matrices.
     * @param warm_start_mgr WarmStartManager for solver state persistence.
     * @param memmap_mgr Optional MemmapManager for memory-mapped D storage
     *                   (nullptr if not using memory mapping).
     */
    ExperimentRunner(Eigen::Map<Eigen::MatrixXd>& X,
                     Eigen::VectorXd& y,
                     dg::DummyGenerator& dummy_gen,
                     wsm::WarmStartManager& warm_start_mgr,
                     mm::MemmapManager* memmap_mgr = nullptr)
        : X_(X),
          y_(y),
          dummy_gen_(dummy_gen),
          warm_start_mgr_(warm_start_mgr),
          memmap_mgr_(memmap_mgr)
    {}


    // ==========================================================================
    // Main entry point
    // ==========================================================================

    /**
     * @brief Run K random experiments and aggregate results.
     *
     * @param cfg Configuration for this run.
     *
     * @return ExperimentResults containing phi_T_mat and Phi vector.
     */
    ExperimentResults run(const ExperimentRunnerConfig& cfg) {

        // Initialize the checkpoint directory if serialized warm-starts are
        // requested (IN_MEMORY retention needs no disk space).
        if (cfg.use_warm_start
            && warm_start_mgr_.mode() == wsm::WarmStartMode::SERIALIZED) {
            warm_start_mgr_.initializeTempDirectory();
        }

        // Dispatch to strategy-specific runner
        std::vector<Eigen::MatrixXd> beta_paths;

        if (cfg.use_memory_mapping && memmap_mgr_ != nullptr) {
            // Memory-mapped path
            beta_paths = runMemoryMapped(cfg);
        } else {
            // In-memory path
            switch (cfg.strategy) {
                case ExperimentStrategy::Standard:
                    beta_paths = runStandard(cfg);
                    break;
                case ExperimentStrategy::Permutation:
                    beta_paths = runPermutation(cfg);
                    break;
                case ExperimentStrategy::Direct:
                    beta_paths = runDirect(cfg);
                    break;
            }
        }

        // Aggregate results
        return aggregateResults(beta_paths, cfg);
    }


private:

    // ==========================================================================
    // Data reference members
    // ==========================================================================

    /** @brief Non-owning reference to normalized design matrix. */
    Eigen::Map<Eigen::MatrixXd>& X_;

    /** @brief Non-owning reference to centered response vector. */
    Eigen::VectorXd& y_;

    /** @brief DummyGenerator for creating / retrieving D matrices. */
    dg::DummyGenerator& dummy_gen_;

    /** @brief WarmStartManager for solver state persistence. */
    wsm::WarmStartManager& warm_start_mgr_;

    /** @brief Optional MemmapManager for memory-mapped D storage. */
    mm::MemmapManager* memmap_mgr_;


    // ==========================================================================
    // In-memory strategies
    // ==========================================================================

    /**
     * @brief STANDARD/HCONCAT/SKIP: K pre-stored dummy matrices.
     *        Supports serial and parallel execution over K experiments.
     *
     * @param cfg ExperimentRunnerConfig with strategy = STANDARD/HCONCAT/SKIP.
     *
     * @return Vector of K beta path matrices (one per experiment).
     */
    std::vector<Eigen::MatrixXd> runStandard(const ExperimentRunnerConfig& cfg) {

        // Validate that K dummies are stored
        if (!dummy_gen_.hasStored(cfg.K)) {
            throw std::runtime_error(
                "ExperimentRunner::runStandard: DummyGenerator does not have "
                + std::to_string(cfg.K) + " stored dummy matrices"
            );
        }

        std::vector<Eigen::MatrixXd> beta_paths(cfg.K);

        const bool do_parallel = (cfg.max_outer_threads > 1);

        // --- OpenMP setup ---
        #ifdef _OPENMP
        int old_max_levels = omp_get_max_active_levels();
        omp_set_max_active_levels(2);
        #endif

        #pragma omp parallel if(do_parallel)
        {
            #ifdef _OPENMP
            omp_set_num_threads(cfg.max_inner_threads);
            #endif
            Eigen::setNbThreads(cfg.max_inner_threads);

            #pragma omp for schedule(dynamic)
            for (std::size_t k = 0; k < cfg.K; ++k) {
                beta_paths[k] = runSingleExperiment(
                    k, dummy_gen_.getStored(k), cfg);
            }
        }

        #ifdef _OPENMP
        omp_set_max_active_levels(old_max_levels);
        #endif

        return beta_paths;
    }


    /**
     * @brief PERMUTATION: Sequential, base-permuted dummies.
     *
     * @param cfg ExperimentRunnerConfig with strategy = PERMUTATION.
     *
     * @return Vector of K beta path matrices (one per experiment).
     */
    std::vector<Eigen::MatrixXd> runPermutation(const ExperimentRunnerConfig& cfg) {

        std::vector<Eigen::MatrixXd> beta_paths(cfg.K);

        // Sequential — permutation reuses one buffer
        #ifdef _OPENMP
        Eigen::setNbThreads(omp_get_max_threads());
        #endif

        for (std::size_t k = 0; k < cfg.K; ++k) {
            beta_paths[k] = runWithTemporaryDummies(
                k, cfg,
                [&]() { return dummy_gen_.getPermuted(k, cfg.num_dummies); });
        }

        return beta_paths;
    }


    /**
     * @brief DIRECT: Sequential, on-the-fly seed-based dummies.
     *
     * @param cfg ExperimentRunnerConfig with strategy = DIRECT.
     *
     * @return Vector of K beta path matrices (one per experiment).
     */
    std::vector<Eigen::MatrixXd> runDirect(const ExperimentRunnerConfig& cfg) {

        std::vector<Eigen::MatrixXd> beta_paths(cfg.K);

        #ifdef _OPENMP
        Eigen::setNbThreads(omp_get_max_threads());
        #endif

        for (std::size_t k = 0; k < cfg.K; ++k) {
            beta_paths[k] = runWithTemporaryDummies(
                k, cfg,
                [&]() { return dummy_gen_.generateDirect(cfg.num_dummies, k); });
        }

        return beta_paths;
    }


    /**
     * @brief Run experiment k for strategies whose D_k is a per-call temporary
     *        (PERMUTATION, DIRECT), handling in-memory warm-start retention.
     *
     * @details When a retained solver exists for slot k, the (expensive)
     *          dummy generation is skipped entirely: the solver already holds
     *          a view of its D_k, whose buffer the WarmStartManager co-retains.
     *          Otherwise the dummies are produced via `make_dummies`, the
     *          experiment runs, and — if dispatchForExperiment stored a fresh
     *          solver for this slot — the D_k buffer is moved into the manager
     *          so the solver's non-owning view stays valid across T-steps
     *          (moving an Eigen::MatrixXd transfers the heap buffer without
     *          changing its address).
     *
     * @param k            Experiment index.
     * @param cfg          Runner configuration.
     * @param make_dummies Callable producing the dummy matrix for slot k.
     *
     * @return Beta path matrix from the solver.
     */
    template <typename MakeDummiesFn>
    Eigen::MatrixXd runWithTemporaryDummies(std::size_t k,
                                            const ExperimentRunnerConfig& cfg,
                                            MakeDummiesFn&& make_dummies) {

        const bool in_memory =
            warm_start_mgr_.mode() == wsm::WarmStartMode::IN_MEMORY;

        if (cfg.use_warm_start && in_memory && warm_start_mgr_.hasSolver(k)) {
            // Warm continuation — cfg.D is unused on this path.
            Eigen::MatrixXd D_unused(0, 0);
            return runSingleExperiment(k, D_unused, cfg);
        }

        Eigen::MatrixXd D_k = make_dummies();
        Eigen::MatrixXd path = runSingleExperiment(k, D_k, cfg);

        // dispatchForExperiment retained a fresh solver iff hasSolver(k) now
        // holds; co-retain D_k's buffer so the solver's view stays valid.
        if (cfg.use_warm_start && in_memory && warm_start_mgr_.hasSolver(k)) {
            warm_start_mgr_.retainDummies(k, std::move(D_k));
        }

        return path;
    }


    // ==========================================================================
    // Memory-mapped path
    // ==========================================================================

    /**
     * @brief Run experiments using memory-mapped D files.
     *
     * @details Strategy determines whether D files are shared (sequential)
     *          or per-experiment (parallel).
     *
     * @param cfg ExperimentRunnerConfig with use_memory_mapping = true.
     *
     * @return Vector of K beta path matrices (one per experiment).
     */
    std::vector<Eigen::MatrixXd> runMemoryMapped(const ExperimentRunnerConfig& cfg) {

        if (!memmap_mgr_->isInitialized()) {
            throw std::runtime_error(
                "ExperimentRunner: MemmapManager not initialized");
        }

        std::vector<Eigen::MatrixXd> beta_paths(cfg.K);

        // Determine if we can parallelize (only for non-shared files)
        const bool shared = (memmap_mgr_->numFiles() == 1);
        const bool do_parallel = !shared && (cfg.max_outer_threads > 1);

        #ifdef _OPENMP
        int old_max_levels = omp_get_max_active_levels();
        if (do_parallel) omp_set_max_active_levels(2);
        #endif

        #pragma omp parallel if(do_parallel)
        {
            #ifdef _OPENMP
            omp_set_num_threads(cfg.max_inner_threads);
            #endif
            Eigen::setNbThreads(cfg.max_inner_threads);

            #pragma omp for schedule(dynamic)
            for (std::size_t k = 0; k < cfg.K; ++k) {
                // Get memory-mapped D region for experiment k
                auto D_map = memmap_mgr_->getDMap(k, cfg.num_dummies);

                // Write dummies into the memory-mapped region
                writeDummiesToMap(k, cfg, D_map);

                // Create y map
                Eigen::Map<Eigen::VectorXd> y_map(y_.data(), y_.size());

                // Build SolverConfig and dispatch
                beta_paths[k] = dispatchForExperiment(k, X_, D_map, y_map, cfg);
            }
        }

        #ifdef _OPENMP
        if (do_parallel) omp_set_max_active_levels(old_max_levels);
        #endif

        return beta_paths;
    }


    // ==========================================================================
    // Single experiment dispatch
    // ==========================================================================

    /**
     * @brief Run a single in-memory experiment: build SolverConfig and dispatch.
     *
     * @param k Experiment index.
     * @param D_k Dummy matrix for this experiment (n × num_dummies).
     * @param cfg Runner configuration.
     *
     * @return Beta path matrix.
     */
    Eigen::MatrixXd runSingleExperiment(
        std::size_t k,
        const Eigen::MatrixXd& D_k,
        const ExperimentRunnerConfig& cfg) {

        // Create Eigen::Maps for the solver
        Eigen::Map<Eigen::MatrixXd> D_map(
                const_cast<double*>(D_k.data()),
                D_k.rows(), D_k.cols());

        Eigen::Map<Eigen::VectorXd> y_map(y_.data(), y_.size());

        return dispatchForExperiment(k, X_, D_map, y_map, cfg);
    }


    /**
     * @brief Build SolverConfig and call dispatchByType for experiment k.
     *
     * @details Warm-start wiring (active only when cfg.use_warm_start):
     *          - IN_MEMORY mode: if a solver is retained for slot k, it is
     *            resumed via SolverConfig::warm_solver; otherwise the freshly
     *            constructed solver is captured through
     *            SolverConfig::retain_sink and stored for the next T-step.
     *          - SERIALIZED mode: the deterministic checkpoint path is passed
     *            as SolverConfig::solver_file (dispatchSolver loads from it
     *            when a saved state is registered, and always re-saves); after
     *            the run the path is registered so hasSolver(k) reports a
     *            resumable state.
     *
     * @param k Experiment index.
     * @param X Map of the original design matrix (n × p).
     * @param D Map of the dummy matrix for this experiment (n × num_dummies).
     * @param y Map of the response vector (n).
     * @param cfg Runner configuration.
     *
     * @return Beta path matrix from the solver.
     */
    Eigen::MatrixXd dispatchForExperiment(
        std::size_t k,
        Eigen::Map<Eigen::MatrixXd>& X,
        Eigen::Map<Eigen::MatrixXd>& D,
        Eigen::Map<Eigen::VectorXd>& y,
        const ExperimentRunnerConfig& cfg) {

        const bool warm_available =
            cfg.use_warm_start && warm_start_mgr_.hasSolver(k);

        sd::SolverConfig solver_cfg{
            .X = X,
            .D = D,
            .y = y,
            .normalize = false,
            .intercept = false,
            .verbose = false,
            .scaling_mode = cfg.scaling_mode,
            .T_stop = cfg.T_stop,
            .early_stop = true,
            .use_warm_start = warm_available,
            .solver_file = {},
            .hyperparams = cfg.solver_params
        };

        std::unique_ptr<trex::tsolvers::TSolver_Base> retained;
        if (cfg.use_warm_start) {
            if (warm_start_mgr_.mode() == wsm::WarmStartMode::IN_MEMORY) {
                if (warm_available) {
                    solver_cfg.warm_solver = warm_start_mgr_.getSolver(k);
                } else {
                    solver_cfg.retain_sink = &retained;
                }
            } else {  // SERIALIZED
                solver_cfg.solver_file = warm_start_mgr_.plannedSolverFile(k);
            }
        }

        Eigen::MatrixXd path = dispatchByType(cfg.solver_type, solver_cfg);

        // Register the produced warm-start state (per-slot writes; safe for
        // distinct k inside the OpenMP loops — vectors are pre-allocated).
        if (retained) {
            warm_start_mgr_.storeSolver(k, std::move(retained));
        }
        if (cfg.use_warm_start
            && warm_start_mgr_.mode() == wsm::WarmStartMode::SERIALIZED
            && !solver_cfg.solver_file.empty()) {
            warm_start_mgr_.registerSolverFile(k);
        }

        return path;
    }


    // ==========================================================================
    // Memory-mapped dummy writing
    // ==========================================================================

    /**
     * @brief Generate dummies directly into a memory-mapped D region.
     *
     * @details All strategies write directly into D_map without intermediate
     *          RAM allocation.
     *
     *  Standard (incl. SKIP):
     *    Generates all num_dummies columns from scratch into D_map.
     *    Seed derivation mirrors generateAndStore() via seed_factor.
     *
     *  HCONCAT (Standard with existing_cols_on_disk > 0):
     *    The first `existing_cols_on_disk` columns are already on disk
     *    from the prior L-loop iteration.  Only the new rightmost columns
     *    are generated via expandInto().
     *
     *  Permutation:
     *    Row-permutes the base dummy matrix directly into D_map.
     *    Requires one base matrix in RAM.
     *
     *  Direct:
     *    Generates from seed directly into D_map.
     *
     * @param k     Experiment index.
     * @param cfg   Runner configuration.
     * @param D_map Map over the memory-mapped D region for experiment k.
     */
    void writeDummiesToMap(std::size_t k,
                           const ExperimentRunnerConfig& cfg,
                           Eigen::Map<Eigen::MatrixXd>& D_map
    ) {

        switch (cfg.strategy) {
            case ExperimentStrategy::Standard: {
                if (cfg.existing_cols_on_disk > 0) {
                    // HCONCAT: expand — only write new rightmost columns
                    dummy_gen_.expandInto(
                        D_map,
                        cfg.existing_cols_on_disk,
                        k,
                        cfg.seed_factor
                    );
                } else {
                    // Full generation (Standard / SKIP / first HCONCAT iteration)
                    dummy_gen_.generateInto(D_map,
                                            k,
                                            cfg.seed_factor);
                }
                break;
            }

            case ExperimentStrategy::Permutation: {
                dummy_gen_.permuteInto(D_map, k, cfg.num_dummies);
                break;
            }

            case ExperimentStrategy::Direct: {
                dummy_gen_.generateDirectInto(D_map, k);
                break;
            }

        }
    }


    // ==========================================================================
    // Result aggregation
    // ==========================================================================

    /**
     * @brief Compute phi_T_mat from K beta paths.
     *
     * @details For each experiment k and each T from 1...T_stop:
     *          Find the first step where exactly T dummies are active,
     *          then count which original predictors (j < p) have nonzero
     *          coefficients at that step.
     *          Accumulate as relative frequency (divide by K).
     *
     * @param beta_paths Vector of K beta path matrices from the solvers.
     * @param cfg Runner configuration (for p, T_stop, eps).
     *
     * @return ExperimentResults with phi_T_mat (p × T_stop) and Phi.
     */
    ExperimentResults aggregateResults(
        const std::vector<Eigen::MatrixXd>& beta_paths,
        const ExperimentRunnerConfig& cfg
    ) const {

        const std::size_t p = cfg.p;
        const std::size_t T_stop = cfg.T_stop;

        Eigen::MatrixXd phi_T_mat = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(p),
            static_cast<Eigen::Index>(T_stop)
        );

        for (std::size_t k = 0; k < cfg.K; ++k) {
            const auto& bp = beta_paths[k];
            const auto num_steps = static_cast<std::size_t>(bp.cols());
            if (num_steps == 0) { continue; }

            // Count dummies included at each step
            Eigen::VectorXi dummy_count = Eigen::VectorXi::Zero(
                static_cast<Eigen::Index>(num_steps)
            );

            for (std::size_t step = 0; step < num_steps; ++step) {
                int count = 0;
                for (std::size_t d = p; d < p + cfg.num_dummies; ++d) {
                    if (std::abs(bp(static_cast<Eigen::Index>(d),
                                       static_cast<Eigen::Index>(step))) > cfg.eps) {
                        ++count;
                    }
                }
                dummy_count(static_cast<Eigen::Index>(step)) = count;
            }

            // For each T from 1 to T_stop
            for (std::size_t t = 1; t <= T_stop; ++t) {
                // Note: R uses strict equality (== t), we use >= t for
                // robustness (greedy solvers can admit several dummies in one
                // step, skipping the exact count).
                //
                // Deliberate difference to R when NO step reaches t dummies
                // (solver terminated early): R falls back to the LAST path
                // column with a warning, inflating Phi with whatever was
                // active at termination; we skip the experiment for this t
                // (zero contribution), which biases Phi downward and is the
                // conservative choice.
                std::size_t step_idx = num_steps - 1;
                bool found = false;
                for (std::size_t s = 0; s < num_steps; ++s) {
                    if (static_cast<std::size_t>(dummy_count(
                        static_cast<Eigen::Index>(s))) >= t) {
                        step_idx = s;
                        found = true;
                        break;
                    }
                }

                if (!found &&
                    dummy_count(static_cast<Eigen::Index>(num_steps - 1)) <
                    static_cast<int>(t)) {
                    continue;
                }

                // Count original predictors with nonzero coefficients
                for (std::size_t j = 0; j < p; ++j) {
                    if (std::abs(bp(static_cast<Eigen::Index>(j),
                                    static_cast<Eigen::Index>(step_idx))) > cfg.eps) {
                        phi_T_mat(static_cast<Eigen::Index>(j),
                                  static_cast<Eigen::Index>(t - 1)) +=
                                  (1.0 / static_cast<double>(cfg.K));
                    }
                }
            }
        }

        ExperimentResults results;
        results.phi_T_mat = std::move(phi_T_mat);
        results.Phi = results.phi_T_mat.col(static_cast<Eigen::Index>(T_stop - 1));

        // --- Optional Screen-TRex beta collection ---
        if (cfg.collect_screen_betas) {
            Eigen::VectorXd b_sums =
                Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));
            std::vector<double> d_betas;

            for (std::size_t k = 0; k < cfg.K; ++k) {
                const auto& bp = beta_paths[k];
                if (bp.cols() == 0) { continue; }
                const Eigen::Index last =
                    static_cast<Eigen::Index>(bp.cols()) - 1;

                for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j) {
                    b_sums(j) += bp(j, last);
                }
                for (std::size_t d = p; d < p + cfg.num_dummies; ++d) {
                    const double v = bp(static_cast<Eigen::Index>(d), last);
                    if (std::abs(v) > cfg.eps) {
                        d_betas.push_back(v);
                    }
                }
            }

            results.beta_sums   = std::move(b_sums);
            results.dummy_betas = std::move(d_betas);
        }

        return results;
    }
// -------------------------------------------------------------------------
}; /* End of class ExperimentRunner */

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::utils::experiment_runner */
// ===================================================================================
#endif /* End of TREX_TREX_SELECTOR_METHODS_UTILS_EXPERIMENT_RUNNER_HPP */
