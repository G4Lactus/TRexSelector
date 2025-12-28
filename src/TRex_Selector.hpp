#ifndef TREX_SELECTOR_HPP
#define TREX_SELECTOR_HPP

#include <cstddef>
#include <limits>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>

#include <Eigen/Dense>

#include "TLARS_Solver.hpp"
#include "TLASSO_Solver.hpp"
#include "TSTEPWISE_Solver.hpp"
#include "TENET_Solver.hpp"
#include "TOMP_Solver.hpp"
#include "TGP_Solver.hpp"
#include "TACGP_Solver.hpp"


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
    STANDARD,       // Generate fresh dummies each iteration (conservative)
    ADAPTIVE        // Horizontally expand dummy matrices (faster)
};


/**
 * @brief Base class for solver-specific configuration
 */
struct SolverConfig {
    virtual ~SolverConfig() = default;
};

/**
 * @brief TENET solver configuration
 */
struct TENETConfig : public SolverConfig {
    double lambda2;

    explicit TENETConfig(double lambda2_val = 1.0)
        : lambda2(lambda2_val) {}
};

/**
 * @brief Configuration for solvers without extra parameters
 * (TLARS, TLASSO, TSTEPWISE, TOMP, TGP, TACGP)
 */
struct DefaultConfig : public SolverConfig {
    DefaultConfig() = default;
};

// Future solver configs can be added here as needed
// Example:
// struct TGPConfig : public SolverConfig {
//     double step_size;
//     int max_iterations;
//
//     TGPConfig(double mu = 0.1, int mi = 1000)
//         : step_size(mu), max_iterations(mi) {}
// };


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
 * Derived classes implement specfic T-Rex variants, s.a.:
 *  - Dependendy Aware (DA) T-Rex
 *  - Screen-TRex
 *  - Dummy Permutation (DP) T-Rex
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

    /**
     * @brief Response vector (owned) (n x 1).
     */
    Eigen::VectorXd y_{};

    /**
     * @brief Number of observations.
     */
    std::size_t n_;

    /**
     * @brief Number of features.
     */
    std::size_t p_;


    // ============================================================
    // Data Members - Configuration
    // ============================================================

    /** @brief Solver type. */
    const SolverTypeForTRex solver_;

    /** @brief Target False Discovery Rate. */
    const double tFDR_;

    /** @brief Number of random experiments. */
    const std::size_t K_;

    /** @brief Maximum number of dummy variables (a multiple of p;
     *          L = p, 2p, 3p .. 10p usually). */
    const std::size_t max_num_dummies_;

    /** @brief Use ceiling(n/2) as max T_stop. */
    const bool max_T_stop_;

    // ============================================================
    // Member Methods - Normalization
    // ============================================================

    /** @brief L-loop calibration strategy */
    const LLoopStrategy lloop_strategy_;


    // ============================================================
    // Data Members - Configuration
    // ============================================================

    /** @brief Random seed. */
    const int seed_;

    /** @brief Numerical tolerance. */
    const double eps_{std::numeric_limits<double>::epsilon()};

    /** @brief Verbose output flag. */
    const bool verbose_;

    // ============================================================
    // Solver specific configuration
    // ============================================================

    /** @brief Solver-specific configuration (e.g., TENET lambda2). */
    std::unique_ptr<SolverConfig> solver_config_;


    // ============================================================
    // Data Members - Normalization parameters
    // ============================================================

    /** @brief Feature column means of original X (size p). */
    Eigen::VectorXd X_means_;

    /** @brief Feature column standard deviations of original X (size p).
     *         Normalization with Bessel factor (n-1).
     */
    Eigen::VectorXd X_stds_;

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
     *         Dimensions: (K x num_dummies_) or aggreated form.
     */
    Eigen::MatrixXd phi_T_mat_;

    /** @brief Optimization point for dummy calibration (typically at v = 0.75). */
    std::size_t opt_point_;

    /** @brief Current dummy multiplier in L-loop. */
    std::size_t dummy_multiplier_LL_;

    /** @brief Storage for dummy matrices (for T-loop reuse) */
    std::vector<Eigen::MatrixXd> stored_dummies_;

    // ============================================================
    // Members for Solver Serialization (T-loop warm start)
    // ============================================================

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
        Eigen::MatrixXd Phi_mat;        /** Relative occurence matrix */
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
     * @param K Number of random experiments (default: 20).
     * @param solver Solver type (default: TLARS).
     * @param max_num_dummies Maximum Dummy multiplier (default: 10).
     * @param max_T_stop Use ceiling(n/2) as max T_stop (default: true).
     * @param lloop_strategy L-loop calibration strategy (default: STANDARD).
     * @param seed Random seed (< 0 for random seed).
     * @param verbose Enable verbose output (default: true).
     * @param solver_config Solver-specific configuration (e.g., TENETConfig for TENET).
     *
     * @return TRexSelector instance.
     */
    TRexSelector(Eigen::Map<Eigen::MatrixXd>& X,
                 Eigen::Map<Eigen::VectorXd>& y,
                 double tFDR = 0.1,
                 std::size_t K = 20,
                 SolverTypeForTRex solver = SolverTypeForTRex::TLARS,
                 std::size_t max_num_dummies = 10,
                 bool max_T_stop = true,
                 LLoopStrategy lloop_strategy = LLoopStrategy::STANDARD,
                 int seed = -1,
                 bool verbose = true,
                 std::unique_ptr<SolverConfig> solver_config = nullptr
    );

    /** @brief Virtual destructor for proper polymorphic behavior. */
    virtual ~TRexSelector() = default;

    // Delete copy semantics
    /** @brief Delete copy constructor. */
    TRexSelector(const TRexSelector&) = delete;

    /** @brief Delete copy assignment operator. */
    TRexSelector& operator=(const TRexSelector&) = delete;

    // Enable Move semantics
    /** @brief Default move constructor. */
    TRexSelector(TRexSelector&&) = delete;

    /** @brief Default move assignment operator. */
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
    inline std::size_t getK() const noexcept { return K_; }

    /** @brief Get dummy multiplier. */
    inline std::size_t getMaxNumDummies() const noexcept { return max_num_dummies_; }

    /** @brief Get maximum T_stop flag. */
    inline bool getMaxTStop() const noexcept { return max_T_stop_; }


    // ============================================================
    // Public Getters - Normalization Parameters
    // ============================================================

    /** @brief Get means of features. */
    inline const Eigen::VectorXd& getXMeans() const noexcept { return X_means_; }

    /** @brief Get standard deviations of features. */
    inline const Eigen::VectorXd& getXStds() const noexcept { return X_stds_; }

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

    /** @brief Get relative occurence matrix Phi. */
    inline const Eigen::MatrixXd& getPhiMat() const noexcept { return Phi_mat_; }

    /** @brief Get selection frequency matrix R. */
    inline const Eigen::MatrixXd& getRMat() const noexcept { return R_mat_; }

    /** @brief Get adjusted relative occurrences Phi'. */
    inline const Eigen::VectorXd& getPhiPrime() const noexcept { return Phi_prime_; }

protected:
    // ============================================================
    // Protected Methods
    // ============================================================

    /** @brief Center y in-place. Stores y_mean internally. */
    void centerY();


    /** @brief Denormalize y to original scale. */
    void decenterY();


    // Legacy - kept for potential future use ----------------------------------------------
    /**
     * @brief Center and z-score normalize X in-place.
     * @details Centers each column (mean = 0) and scales to unit variance (std = 1)
     *          using Bessel correction (n-1). Stores X_means_ and X_stds_.
     *          Result: Each column has mean=0, std=1, L2 norm=sqrt(n-1).
     */
    void centerAndZscoreNormalizeX();


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
    void centerAndL2NormalizeMatrix(Eigen::Map<Eigen::MatrixXd>& X) const;

    // -------------------------------------------------------------------------------------


    /**
     * @brief Convert z-score normalized matrix to L2 normalized.
     * @details Divides each column by sqrt(n-1) to convert from z-score
     *          (L2 norm = sqrt(n-1)) to unit L2 norm (L2 norm = 1).
     * @param Z Matrix in z-score normalization (modified in-place).
     */
    void convertZscoreToL2(Eigen::Map<Eigen::MatrixXd>& Z) const;


    /**
     * @brief Convert L2 normalized matrix to z-score normalized.
     * @details Multiplies each column by sqrt(n-1) to convert from unit L2 norm
     *          (L2 norm = 1) to z-score (L2 norm = sqrt(n-1)).
     * @param Z Matrix in L2 normalization (modified in-place).
     */
    void convertL2ToZscore(Eigen::Map<Eigen::MatrixXd>& Z) const;


    /**
     * @brief Restore X to original scale.
     * @details Reverses z-score normalization: X = X_zscore * X_stds + X_means.
     */
    void denormalizeX();


    // ===========================================================
    // Core T-Rex algorithm methods
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
     *
     * @return Experiment results (phi_T_mat and Phi).
     */
    ExperimentResults runRandomExperiments(
        std::size_t num_dummies,
        std::size_t T_stop,
        bool generate_new_dummies,
        bool use_warm_start
    );


    /**
     * @brief Run L-loop calibration to determine number of dummies.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void runLLoopCalibration(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


    /**
     * @brief Run L-loop calibration (ADAPTIVE strategy).
     *
     * @details Horizontally expands dummy matrices across iterations.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void runLLoopCalibration_Adaptive(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


    /**
     * @brief Run T-loop calibration to determine stopping threshold T_stop.
     *
     * @param FDP_hat FDP estimates.
     * @param exp_results Experiment results structure to fill.
     */
    void runTLoopCalibration(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results);


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
     * @brief Generate dummy matrix.
     *
     * @details Generate dummy matrix with N(0,1) entries,
     *          z-score normalized (n x num_dummies).
     *
     * @param num_dummies Number of dummy columns.
     * @param experiment_id Experiment index (for seed offset).
     * @return Dummy matrix with N(0,1) entries, z-score + L2 normalized.
     */
    Eigen::MatrixXd generateDummies(
        std::size_t num_dummies,
        std::size_t experiment_id
    ) const;


    /**
     * @brief Create augmented matrix XD = [X | D].
     *
     * @param D Dummy matrix (n x num_dummies).
     *
     * @return Augmented matrix (n x (p + num_dummies)).
     */
    Eigen::MatrixXd createAugmentedMatrix(const Eigen::MatrixXd& D) const;


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

    /** @brief Print diagnostics of current TRexSelector state */
    void printDiagnostics() const;

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

}; /* End of TRexSelector class */


#endif /* End of TREX_SELECTOR_HPP */
