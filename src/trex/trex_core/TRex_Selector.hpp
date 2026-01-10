// ===================================================================================
// TRex_Selector.hpp
// ===================================================================================
/**
 * @file TRex_Selector.hpp
 *
 * @brief Implementation of the T-Rex Selector for FDR-controlled variable selection.
 *
 */
// ===================================================================================

#ifndef TREX_SELECTOR_HPP
#define TREX_SELECTOR_HPP

// ===================================================================================

// std includes
#include <cstddef>
#include <limits>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>

// Eigen includes
#include <Eigen/Dense>

// T-Rex Selector includes
#include <tsolvers/TLARS_Solver.hpp>
#include <tsolvers/TLASSO_Solver.hpp>
#include <tsolvers/TSTEPWISE_Solver.hpp>
#include <tsolvers/TENET_Solver.hpp>
#include <tsolvers/TOMP_Solver.hpp>
#include <tsolvers/TGP_Solver.hpp>
#include <tsolvers/TACGP_Solver.hpp>
#include <utils/datagen/utils_dummygen.hpp>
#include <utils/memmap/MemoryMappedMatrix.hpp>

// ===================================================================================
// Namespace aliases
// ===========================================================
namespace dummygen = trex::utils::dummygen;
namespace memmap = trex::utils::memmap;

// ===================================================================================


// ============================================================
// Enums & Control Structures for TRex and Solvers
// ============================================================

/**
 * @brief Solver types for T-Rex Selector
 */
enum class SolverTypeForTRex {
    TLARS,      // Terminating LARS
    TLASSO,     // Terminating LASSO
    TSTEPWISE,  // Terminating Stepwise
    TENET,      // Terminating ENET
    TOMP,       // Terminating OMP
    TGP,        // Terminating Gradient Pursuit
    TACGP       // Terminating Approximate Conjugate Gradient Pursuit
};


/**
 * @brief L-loop calibration strategies
 */
enum class LLoopStrategy {
    SKIP,           // Skip L-loop calibration: use fixed number of dummies
    STANDARD,       // Generate fresh dummies in each L-loop iteration (conservative)
    ADAPTIVE        // Horizontally expand dummy matrices (faster - same result)
};


/**
 * @brief Algorithmic control parameters for T-Rex Selector.
 *
 * @details Groups all T-Rex algorithm control parameters:
 *          - Experiment configuration
 *          - Dummy Generation
 *          - Loop Strategies
 *          - Early stopping Strategies
 *          - Calibration
 *          - Memory Management
 *          - Solver Configuration
 */
struct TRexControlParameter {

    // ====================================
    // Experiment Configuration
    // ====================================
    /** @brief Number of random experiments (default: 20). */
    std::size_t K = 20;

    /**
     * @brief Maximum number of dummy variables as a multiple of p.
     *
     * @details For STANDARD/ADAPTIVE: L = p, 2p, .., max_dummy_multiplier * p (Default: 10).
     *          For SKIP: Set fixed number of dummies = max_dummy_multiplier * p.
     */
    std::size_t max_dummy_multiplier = 10;

    /** @brief If true, limit T_stop to ceiling(n/2) (default: true). */
    bool max_T_stop = true;

    // ===================================
    // Dummy Generation
    // ===================================

    /** @brief Distribution for generating dummies (default: Normal). */
    dummygen::Distribution dummy_distribution = dummygen::Distribution::Normal();

    // ===================================
    // L-Loop Strategy
    // ===================================
    /**
     * @brief Strategy for L-loop calibration (default: ADAPTIVE).
     *
     * @details
     *    - SKIP:     Skip L-loop calibration, use fixed number of
     *                dummies = max_dummy_multiplier * p.
     *    - STANDARD: Generate fresh dummies in each L-loop iteration (conservative).
     *    - ADAPTIVE: Horizontally expand dummy matrices (faster - same result).
     */
    LLoopStrategy lloop_strategy = LLoopStrategy::ADAPTIVE;

    // ==================================
    // T-Loop Early Strategy
    // ==================================

    /** @brief If true, perform early stopping if support set stagnates (default: true). */
    bool tloop_stagnation_stop = true;

    /** @brief Number of stagnant steps required to trigger early T-loop stopping (default: 3). */
    std::size_t max_stagnant_steps = 3;

    // =================================
    // Parallel random experiments
    // =================================

    /** @brief If true, allows parallel execution of random experiments. */
    bool parallel_rnd_experiments = false;

    /** @brief Computed number of threads for the outer loop (experiments). */
    int max_outer_threads = 1;

    /** @brief Computed number of threads for the inner loop (solvers/Eigen). */
    int max_inner_threads = 1;

    /**
     * @brief Automatically configures thread resources based on K and hardware.
     */
    void autoConfigResources() {
#ifdef _OPENMP
        // Only auto-config if parallel is requested
        if (parallel_rnd_experiments) {
            int total_cores = omp_get_max_threads();
            int k_experiments = static_cast<int>(K);

            if (k_experiments > 1) {
                // Hybrid: Limit outer threads to K
                max_outer_threads = std::min(total_cores, k_experiments);
                // Give remaining capacity to the solver (floor division)
                max_inner_threads = std::max(1, total_cores / max_outer_threads);
            } else {
                // Single experiment -> Solver gets everything
                max_outer_threads = 1;
                max_inner_threads = total_cores;
            }
        } else {
             // Serial experiments -> Solver gets everything
             max_outer_threads = 1;
             max_inner_threads = omp_get_max_threads();
        }
#else
        // Fallback for non-OpenMP builds
        max_outer_threads = 1;
        max_inner_threads = 1;
#endif
    }

    // ===================================
    // Memory-Mapping Management
    // ===================================

    /**
     * @brief Enable memory-mapped storage for augmented matrices.
     *
     * @details When true:
     *   MemMap Strategy 1: Creates K memory-mapped files [X | D_k] on disk instead of storing
     *                      dummy matrices in RAM. Essential for ultra-high-dimensional data,
     *                      where K * n * num_dummies exceeds available RAM.
     *
     *                      Memory-mapped files are sparse and pre-allocated to max size:
     *                      n x (p + max_dummy_multiplier * p). Default: false.
     */
    bool use_memory_mapping = false;

    // ===================================
    // Solver Configuration
    // ===================================
    /** @brief Configuration for solvers (TLARS, TLASSO, TENET, TSTEPWISE, TOMP, TGP, TACGP) */

    /** @brief Solver type for T-Rex Selector (default: TLARS). */
    SolverTypeForTRex solver_type = SolverTypeForTRex::TLARS;

    /** @brief L2 penalty for TENET solver (default: 0.1). */
    double tenet_lambda2 = 0.1;
};


// ============================================================
// T-Rex Selector Class
// ============================================================

/**
 * @brief T-Rex Selector for FDR-controlled variable selection.
 *
 * @details
 * Base class implementing the T-Rex selector for FDR-controlled variable selection.
 * Serves also as base for inheritance.
 * Supports in-memory and memory-mapped data via Eigen::Map.
 *
 * Key design features:
 *  - Does not own X (too large for p>>n scenarios), y (n) is owned
 *  - Normalizes X in-place
 *  - Stores normalization parameters for later use and restoration (means, stds, l2norms)
 *  - select() returns SelectionResult with all key outputs
 *
 * Usage:
 *   1. Construct with X and y (non-const, will be normalized in-place)
 *   2. Call select() for variable selection
 *   3. Access results via getters
 *
 * Derived classes implement specific T-Rex variants, s.a.:
 *  - Dependency Aware (DA) T-Rex
 *  - Screen-TRex
 *  - Dummy Permutation (DP) T-Rex
 *  - GVS TRex
 *  - Sparse PCA TRex
 *  - etc.
 */
class TRexSelector {
protected:

    // ============================================================
    // Data Members - Input Data
    // ============================================================

    /**
     * @brief Pointer to Feature matrix (n x p) (view).
     *        The user must ensure that the data lifetime exceeds.
     *        That of the TRexSelector instance.
     */
    Eigen::Map<Eigen::MatrixXd>* X_;

    /** @brief Response vector (owned) (n x 1). */
    Eigen::VectorXd y_{};

    /** @brief Number of observations. */
    std::size_t n_;

    /** @brief Number of features. */
    std::size_t p_;


    // ============================================================
    // Data Members - Configuration
    // ============================================================

    /** @brief Target False Discovery Rate. */
    const double tFDR_;

    /** @brief Algorithmic control parameters. */
    const TRexControlParameter trex_ctrl_;

    /** @brief Random seed. */
    const int seed_;

    /** @brief Verbose output flag. */
    const bool verbose_;

    /** @brief Numerical tolerance. */
    const double eps_{std::numeric_limits<double>::epsilon()};


    // ============================================================
    // Data Members - Normalization parameters
    // ============================================================

    /** @brief Feature column means of original X (size p). */
    Eigen::VectorXd X_means_;

    /** @brief Feature column L2 norms of original X (size p). */
    Eigen::VectorXd X_l2norms_;

    /** @brief Response mean. */
    double y_mean_;


    // ============================================================
    // Protected Data Members for Results
    // ============================================================

    /** @brief Binary support vector (0/1). */
    Eigen::VectorXi selected_var_;

    /** @brief Indices of selected variables. */
    std::vector<std::size_t> selected_indices_;

    /** @brief Stopping time. */
    std::size_t T_stop_;

    /** @brief Number of dummy variables used. */
    std::size_t num_dummies_;

    /** @brief Voting grid used. */
    Eigen::VectorXd voting_grid_;

    /** @brief Voting threshold. */
    double voting_threshold_;

    /** @brief Estimated FDP matrix. */
    Eigen::MatrixXd FDP_hat_mat_;

    /** @brief Relative occurrence matrix. */
    Eigen::MatrixXd Phi_mat_;

    /** @brief Selection frequency matrix. */
    Eigen::MatrixXd R_mat_;

    /** @brief Adjusted relative occurrences (p x 1). */
    Eigen::VectorXd Phi_prime_;

    /** @brief Tracks dummy inclusion frequencies across experiments.
     *         Dimensions: (K x num_dummies_) or aggregated form.
     */
    Eigen::MatrixXd phi_T_mat_;

    /** @brief Optimization point for dummy calibration (typically at v = 0.75). */
    std::size_t opt_point_;

    /** @brief Current dummy multiplier in L-loop. */
    std::size_t dummy_multiplier_LL_;

    // ============================================================
    // Members for Solver Serialization (T-loop warm start)
    // ============================================================

    /** @brief Storage for dummy matrices (for T-loop reuse) */
    std::vector<Eigen::MatrixXd> stored_dummies_;

    /**
     * @brief Directory for temporary solver state files.
     *
     * @details Created during T-Loop, deleted after selection.
     */
    std::string temp_dir_;

    /**
     * @brief Paths to serialized solver files (K files).
     *
     * @details Only populated during T-Loop for warm starts.
     */
    std::vector<std::string> solver_files_;

    /**
     * @brief Flag indicating if serialized solvers are available.
     */
    bool has_serialized_solvers_;

    // ============================================================
    // Memory-Mapping Matrix Storage
    // ============================================================

    /**
     * @brief Setup memory-mapped augmented matrices [X | D_k] for K experiments.
     *
     * @details Only used if trex_ctrl_.use_memory_mapping is true.
     *          Each matrix is pre-allocated to size n x (p + max_dummy_multiplier * p)
     *          as a sparse file. Active columns [0, p + num_dummies) are filled
     *          during the L-loop. The remaining columns stay sparse (zero disk usage).
     *          Files created in temp_dir_ as "XD_aug_{0}.bin", ..., "XD_aug_{K-1}.bin".
     *          The memory is released when the TRexSelector instance is destroyed.
     */
    std::vector<std::unique_ptr<memmap::MemoryMappedMatrix<double>>> XD_memmaps_;

    /**
     * @brief Maximum number of columns allocated for memory-mapped matrices.
     *
     * @details max_cols = p + trex_ctrl_.max_dummy_multiplier * p.
     *          Set during initialization if memory-mapping is enabled.
     */
    std::size_t max_cols_memmap_;


public:
    // ===========================================================
    // Internal Result Structures
    // ===========================================================

    /** @brief Result from variable selection step. */
    struct SelectionResult {
        Eigen::VectorXi selected_var;   /** Binary support vector (0/1) */
        double v_thresh;                /** Voting threshold */
        Eigen::MatrixXd R_mat;          /** Selection frequency matrix */
        std::size_t T_stop;             /** Number of dummies used */
        std::size_t num_dummies;        /** Total number of dummies used */
        Eigen::MatrixXd FDP_hat_mat;    /** Estimated FDP matrix */
        Eigen::MatrixXd Phi_mat;        /** Relative occurrence matrix */
        Eigen::VectorXd Phi_prime;      /** Adjusted relative occurrences (p x 1) */
        Eigen::VectorXd voting_grid;    /** Voting grid used */
    };


    // ============================================================
    // Constructor
    // ============================================================

    /**
     * @brief Constructor for TRexSelector.
     *
     * @param X Feature matrix (n x p) - accepts Eigen::Map.
     * @param y Response vector (n x 1) - accepts Eigen::Map.
     * @param tFDR Target False Discovery Rate (default: 0.1).
     * @param trex_control Algorithmic control parameters.
     * @param seed Random seed (< 0 for random seed).
     * @param verbose Enable verbose output (default: true).
     */
    TRexSelector(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::VectorXd>& y,
                 double tFDR = 0.1,
                 TRexControlParameter trex_control = TRexControlParameter(),
                 int seed = -1,
                 bool verbose = true
    );

    /** @brief Virtual destructor for proper polymorphic behavior. */
    virtual ~TRexSelector() = default;

    // Delete copy semantics
    /** @brief Delete copy constructor. */
    TRexSelector(const TRexSelector&) = delete;

    /** @brief Delete copy assignment operator. */
    TRexSelector& operator=(const TRexSelector&) = delete;

    // Delete Move semantics
    /** @brief Delete move constructor. */
    TRexSelector(TRexSelector&&) = delete;

    /** @brief Delete move assignment operator. */
    TRexSelector& operator=(TRexSelector&&) = delete;


    // ============================================================
    // Main method
    // ============================================================

    /**
     * @brief Run T-Rex selection algorithm.
     *
     * @details
     * Main entry point for implementing T-Rex variable selection:
     * - L-loop: calibrate number of dummies
     * - T-loop: calibrate stopping threshold
     * - Variable selection with FDR control
     *
     * Results are stored internally and can be accessed via getters.
     *
     * @return SelectionResult containing selected variables and
     *         diagnostics.
     */
    virtual SelectionResult select();


    // ============================================================
    // Public Getters - Configuration Parameters
    // ============================================================

    /** @brief Get target False Discovery Rate. */
    inline double getTFDR() const noexcept { return tFDR_; }

    /** @brief Get number of random experiments. */
    inline std::size_t getK() const noexcept { return trex_ctrl_.K; }

    /** @brief Get dummy multiplier. */
    inline std::size_t getMaxNumDummies() const noexcept {
        return trex_ctrl_.max_dummy_multiplier;
    }

    /** @brief Get maximum T_stop flag. */
    inline bool getMaxTStop() const noexcept { return trex_ctrl_.max_T_stop; }

    /** @brief Get stagnation check flag. */
    inline bool getStagnationCheck() const noexcept {
        return trex_ctrl_.tloop_stagnation_stop;
    }

    // ============================================================
    // Public Getters - Normalization Parameters
    // ============================================================

    /** @brief Get means of features. */
    inline const Eigen::VectorXd& getXMeans() const noexcept { return X_means_; }

    /** @brief Get L2 norms of features. */
    inline const Eigen::VectorXd& getXL2Norms() const noexcept { return X_l2norms_; }

    /** @brief Get mean of response. */
    inline double getYMean() const noexcept { return y_mean_; }


    // ============================================================
    // Public Getters - Output Result Members
    // ============================================================

    /** @brief Get binary support vector. */
    inline const Eigen::VectorXi& getSelectedVar() const noexcept { return selected_var_; }

    /** @brief Get indices of selected variables. */
    inline const std::vector<std::size_t>& getSelectedIndices() const noexcept {
        return selected_indices_;
    }

    /** @brief Get number of selected variables. */
    inline std::size_t getNumSelected() const noexcept {
        return selected_indices_.size();
    }

    /** @brief Get T_stop value (number of dummies before stopping). */
    inline std::size_t getTStop() const noexcept { return T_stop_; }

    /** @brief Get voting grid. */
    inline const Eigen::VectorXd& getVotingGrid() const noexcept { return voting_grid_; }

    /** @brief Get voting threshold. */
    inline double getVotingThreshold() const noexcept { return voting_threshold_; }

    /** @brief Get FDP estimates matrix. */
    inline const Eigen::MatrixXd& getFDPHatMat() const noexcept { return FDP_hat_mat_; }

    /** @brief Get relative occurrence matrix Phi. */
    inline const Eigen::MatrixXd& getPhiMat() const noexcept { return Phi_mat_; }

    /** @brief Get selection frequency matrix R. */
    inline const Eigen::MatrixXd& getRMat() const noexcept { return R_mat_; }

    /** @brief Get adjusted relative occurrences Phi'. */
    inline const Eigen::VectorXd& getPhiPrime() const noexcept { return Phi_prime_; }

protected:
    // ============================================================
    // Protected Methods
    // ============================================================

    // ===========================================================
    // Data normalization methods
    // ===========================================================

    /** @brief Center y in-place. Stores y_mean internally. */
    void centerY();


    /** @brief Denormalize y to original scale. */
    void decenterY();


    /**
     * @brief Center and L2 normalize X in-place.
     * @details Centers each column, then scales to unit L2 norm.
     *          Stores X_means_ and X_l2norms_.
     *          Result: Each column has mean = 0, L2 norm = 1.
     */
    void centerAndL2NormalizeX();


    /**
     * @brief Center and L2 normalize matrix in-place.
     * @details Centers each column, then scales to unit L2 norm.
     *          Does NOT store normalization parameters.
     * @param X Mapped matrix to normalize (modified in-place).
     */
    void centerAndL2NormalizeMatrix(Eigen::Map<Eigen::MatrixXd>& M) const;


    /**
     * @brief Restore X to original scale.
     *
     * @details Reverses l2-score normalization: X = X_l2normalized * X_l2norms + X_means.
     */
    void denormalizeX();


    // ===========================================================
    // Core T-Rex computation methods part 1
    // ===========================================================

    /** @brief Generate voting grid V = [0.5, 0.5 + 1/K, ..., 1-eps]. */
    Eigen::VectorXd createVotingGrid() const;


    /** @brief Set 75% optimization point for determining number of dummies required.
     *
     * @param v_len Length of voting grid.
     */
    void set75PercentOptPoint(const std::size_t v_len);


    /**
     * @brief Run K random experiments and compute phi_T_mat.
     *
     * @param num_dummies Number of dummies to append.
     * @param T_stop Early stopping threshold.
     * @param generate_new_dummies If true, generate new dummies;
     *                             if false, reuse stored dummies.
     *
     * @return phi_T_mat (p x T_stop) and Phi vector.
     */
    struct ExperimentResults {
        Eigen::MatrixXd phi_T_mat;   /** Dummy inclusion tracking matrix (p x T_stop) */
        Eigen::VectorXd Phi;         /** Relative occurrence vector (p x 1) */
    };


    /**
     * @brief Run K random experiments.
     *
     * @param num_dummies Number of dummies to append.
     * @param T_stop Early stopping threshold.
     * @param generate_new_dummies If true, generate new dummies;
     * @param use_warm_start If true, load and continue from saved state.
     * @param seed_factor Seed factor for dummy generation.
     *
     * @return Experiment results (phi_T_mat and Phi).
     */
    ExperimentResults runRandomExperiments(
        std::size_t num_dummies,
        std::size_t T_stop,
        bool generate_new_dummies,
        bool use_warm_start,
        std::size_t seed_factor
    );


    // ===========================================================
    // L-loop strategies
    // ===========================================================

    /**
     * @brief Run L-loop with NONE strategy (fixed number of dummies).
     *
     * @details Skips L-loop calibration and uses fixed number_of_dummies = max_dummy_multiplier * p.
     *          Runs a single experiment to compute initial FDP estimate.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void runLLoopCalibration_SKIP(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


    /**
     * @brief Run L-loop calibration strategy to determine number of dummies.
     *
     * @details Generates fresh dummies in each L-loop iteration.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void runLLoopCalibration(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


    /**
     * @brief Run L-loop calibration with ADAPTIVE strategy.
     *
     * @details Horizontally expands dummy matrices across iterations.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void runLLoopCalibration_Adaptive(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


    /**
     * @brief Apply L-loop strategy as configured.
     *
     * @details Dispatches appropriate L-loop strategy according to trex_ctrl_.lloop_strategy.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void applyLLoopStrategy(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


    // ===========================================================
    // T-loop method
    // ===========================================================

    /**
     * @brief Run T-loop calibration to determine stopping threshold T_stop.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void runTLoopCalibration(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


    // ===========================================================
    // Core T-Rex computation methods part 2
    // ===========================================================

    /**
     * @brief Compute phi_T_mat from collection of beta paths.
     *
     * @param beta_paths Vector of K beta paths, each (p + d) x steps.
     * @param num_dummies Number of dummies appended.
     * @param T_stop Stopping threshold.
     *
     * @return phi_T_mat (p x T_stop) - relative occurrence matrix.
     */
    Eigen::MatrixXd computePhiTMat(
        const std::vector<Eigen::MatrixXd>& beta_paths,
        std::size_t num_dummies,
        std::size_t T_stop
    ) const;


    /**
     * @brief Compute Phi_prime (calibration parameter).
     *
     * @param phi_T_mat Matrix tracking dummy inclusion frequencies.
     * @param Phi Current Phi vector.
     * @param num_dummies Number of dummies used.
     *
     * @return Adjusted relative occurrence Phi'.
     */
    Eigen::VectorXd computePhiPrime(
        const Eigen::MatrixXd& phi_T_mat,
        const Eigen::VectorXd& Phi,
        std::size_t num_dummies
    ) const;


    /** @brief Compute FDP estimates for voting grid.
     *
     * @param voting_grid Voting grid.
     * @param Phi Relative occurrence matrix.
     * @param Phi_prime Adjusted relative occurrence.
     *
     * @return FDP estimates vector for voting grid.
     */
    Eigen::VectorXd computeFDPHat(
        const Eigen::VectorXd& voting_grid,
        const Eigen::VectorXd& Phi,
        const Eigen::VectorXd& Phi_prime
    ) const;


    /**
     * @brief Select variables with FDR control.
     *
     * Implements the T-Rex variable selection criterion.
     *
     * @param FDP_hat_mat Matrix of FDP estimates (T_stop x |V|).
     * @param Phi_mat Matrix of relative occurrences (T_stop x p).
     * @param V Voting grid.
     * @param T_stop Number of dummy inclusion steps.
     *
     * @return Selection result (support vector, threshold, R_mat).
     */
    SelectionResult selectedVariables(
        const Eigen::MatrixXd& FDP_hat_mat,
        const Eigen::MatrixXd& Phi_mat,
        const Eigen::VectorXd& voting_grid,
        std::size_t T_stop
    ) const;


    /**
     * @brief Generate dummy matrix using configured distribution.
     *
     * @details Delegates to utils::dummygen::generate_dummies with distribution specified
     *          in trex_ctrl_.dummy_distribution.
     *          Default: Normal distribution.
     *
     * @param num_dummies Number of dummy columns.
     * @param experiment_id Experiment index (for seed offset).
     *
     * @return Dummy matrix with (n x num_dummies), L2 normalized columns.
     */
    Eigen::MatrixXd generateDummies(
        std::size_t num_dummies,
        std::size_t experiment_id
    ) const;


    /**
     * @brief Run a single experiment with a specific solver type.
     *
     * @details Handles solver creation, warm start loading, execution, and saving.
     *
     * @param XD_map Augmented design matrix.
     * @param y_map Response vector.
     * @param T_stop Early stopping threshold.
     * @param use_warm_start If true, load from file; else create fresh.
     * @param solver_file Path to solver file for load/save.
     *
     * @return Beta path matrix.
     */
    Eigen::MatrixXd runSingleExperimentWithSolver(
        Eigen::Map<Eigen::MatrixXd>& XD_map,
        Eigen::Map<Eigen::VectorXd>& y_map,
        std::size_t T_stop,
        bool use_warm_start,
        const std::string& solver_file
    );


    /**
     * @brief Store results after selection.
     *
     * Called after selection completes to update internal state.
     */
    void storeResults(
        const SelectionResult& result
    );

    // ===========================================================
    // Utilities
    // ===========================================================

    /** @brief Validate input data dimensions */
    void validateTRexParameters() const;

    /** @brief Print progress message (if verbose == true) */
    void printProgress(const std::string& message) const;

    /** @brief Update selected_indices_ from selected_var_ */
    void updateSelectedIndices();

    // ============================================================
    // Methods for Solver State Serialization (T-loop warm start)
    // ============================================================

    /**
     * @brief Create temporary directory for solver state files.
     *
     * @return Path to temporary directory.
     */
    std::string createTempDirectory();

    /**
     * @brief Clean up temporary directory and all solver files.
     */
    void cleanupTempDirectory();

    // ============================================================
    // Memory-Mapping Matrix Management
    // ============================================================

    /**
     * @brief Initialize K memory-mapped augmented matrices [X | D_k].
     *
     * @details Creates K sparse files in temp_dir_, each pre-allocated to
     *          size n x (p + max_dummy_multiplier * p). Files are initialized
     *          as sparse matrices (zero disk usage) and filled incrementally
     *          during L-loop iterations.
     *          Only used if trex_ctrl_.use_memory_mapping is true.
     */
    void initializeMemoryMappedMatrices();

    /**
     * @brief Write X data to all K memory-mapped matrices.
     *
     * @details Copies X into the first p columns of each of the K augmented matrices at the
     *          beginning of the L-loop to avoid redundant writes.
     *          Only used if trex_ctrl_.use_memory_mapping is true.
     */
    void writeXToMemoryMappedMatrices();

    /**
     * @brief Clean up memory-mapped matrices and associated files.
     *
     * @details Closes and deletes all memory-mapped and temporary files.
     *          Called at end of select() or during destructor.
     */
    void cleanupMemoryMappedMatrices();


    /**
     * @brief Run K random experiments using memory-mapped matrices.
     *
     * @details Memory-mapped version that operates on pre-allocated XD_mammaps_ instead
     *          of in-memory dummy matrices.
     *          Used when trex_ctrl_.use_memory_mapping is true.
     *
     * @param num_dummies Number of dummies to append.
     * @param T_stop Early stopping threshold.
     * @param use_warm_start If true, load from file; else create fresh.
     *
     * @return Experiment results (phi_T_mat and Phi).
     */
    ExperimentResults runRandomExperiments_MemMap(
        std::size_t num_dummies,
        std::size_t T_stop,
        bool use_warm_start
    );


}; /* End of TRexSelector class */


#endif /* End of TREX_SELECTOR_HPP */
