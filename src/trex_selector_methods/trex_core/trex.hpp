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
#include <limits>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// TRex includes
#include <trex_selector_methods/utils/trex_data_normalizer.hpp>
#include <trex_selector_methods/utils/trex_solver_dispatch.hpp>
#include <trex_selector_methods/utils/trex_warm_start_manager.hpp>
#include <trex_selector_methods/utils/trex_memmap_manager.hpp>
#include <trex_selector_methods/utils/trex_dummy_generator.hpp>
#include <trex_selector_methods/utils/trex_experiment_runner.hpp>

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

    /** @brief Numerical tolerance. */
    const double eps_{std::numeric_limits<double>::epsilon()};

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
     * @return SelectionResult containing selected variables and diagnostics.
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

    /** @brief Get stopping time T. */
    std::size_t getStoppingTimeT() const noexcept { return T_stop_; }


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
