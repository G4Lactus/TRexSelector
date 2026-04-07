// ===================================================================================
// trex_screening.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_SCREENING_TREX_SCREENING_HPP
#define TREX_SELECTOR_METHODS_TREX_SCREENING_TREX_SCREENING_HPP
// ===================================================================================
/**
 * @file trex_screening.hpp
 *
 * @brief Screen-TRex Selector — fast FDR-controlled screening at T=1, L=p.
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex base
#include <trex_selector_methods/trex_core/trex.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_screening {

// Namespace aliases
namespace tc = trex::trex_selector_methods::trex_core;
namespace er = trex::trex_selector_methods::utils::experiment_runner;

// ===================================================================================
// Enums & Control Structures for Screen-TRex
// ===================================================================================

/**
 * @brief Screen-TRex variants available.
 */
enum class ScreenTRexMethod {
    /** @brief Standard T-Rex */
    TREX,

    /** @brief Dependency-Aware T-Rex (Auto-Regressive 1) */
    TREX_DA_AR1,

    /** @brief Dependency-Aware T-Rex (Equi-Correlated) */
    TREX_DA_EQUI
};


/** @brief Configuration for Screen-TRex specific parameters. */
struct ScreenTRexControlParameter {
    /** @brief If true, perform the Confidence-Based Screen-TRex (bootstrapping). */
    bool use_bootstrap_CI = false;

    /** @brief Number of bootstrap resamples (default: 1000). */
    std::size_t R_boot = 1000;

    /** @brief Grid step size for confidence level search in [0, 1] (default: 0.001). */
    double ci_grid_step = 0.001;

    /** @brief Screen-TRex variant to use (default: TREX). */
    ScreenTRexMethod trex_method = ScreenTRexMethod::TREX;

    /** @brief Correlation threshold for DA variants (default: 0.02). */
    double rho_thr_DA = 0.02;

    /** @brief Manually specified correlation coefficient.
     *  NaN triggers automatic estimation.
     */
    double cor_coef = std::numeric_limits<double>::quiet_NaN();
};


/** @brief Result structure for Screen-TRex, extending SelectionResult.
 *
 * @details
 *  Adds Screen-TRex specific diagnostics on top of the standard result.
 */
struct ScreenTRexSelectionResult : public tc::TRexSelector::SelectionResult {
    /** @brief Estimated FDR = 1 / max(1, R). */
    double estimated_FDR = 0.0;

    /** @brief Whether bootstrap CI was applied. */
    bool used_bootstrap = false;

    /** @brief Confidence level selected by bootstrap grid search. */
    double confidence_level = 0.0;

    /** @brief Averaged beta coefficients for original features (p × 1). */
    Eigen::VectorXd beta_mat;

    /** @brief Non-zero dummy coefficients collected for bootstrap. */
    Eigen::VectorXd dummy_betas;

    /** @brief Estimated / used correlation coefficient (for DA variants). */
    double estimated_correlation = 0.0;
};


// ===================================================================================
// ScreenTRexSelector
// ===================================================================================

/**
 * @brief Screen-TRex Selector.
 *
 * @details
 *  Overrides TRexSelector::select() to run K experiments at T=1, L=p with no
 *  T-loop or L-loop calibration.  Supports ordinary thresholding (Phi > 0.5),
 *  a bootstrap-CI variant, and Dependency-Aware (DA) Phi adjustment for AR(1)
 *  and equi-correlated designs.
 *
 *  Supported dummy strategies: STANDARD and PERMUTATION only.
 *  Memory-mapped D storage is supported through the base class MemmapManager
 *  when TRexControlParameter::use_memory_mapping is set.
 *
 *  Variants:
 *  1. Ordinary Screen-TRex: selects variables where Phi > 0.5.
 *  2. Confidence-Based Screen-TRex: bootstrap CI on dummy betas.
 *  3. DA-AR1: adjusts Phi using a local AR(1) neighborhood window.
 *  4. DA-EQUI: adjusts Phi using a global equi-correlation comparison.
 */
class ScreenTRexSelector : public tc::TRexSelector {

protected:
    // ==========================================================================
    // Screen-TRex Data Members
    // ==========================================================================

    /** @brief Screen-TRex control parameters. */
    ScreenTRexControlParameter screen_ctrl_;

    /** @brief Sum of betas for original features across K experiments (p × 1). */
    Eigen::MatrixXd accumulated_beta_mat_;

    /** @brief Non-zero dummy betas collected across K experiments. */
    std::vector<double> collected_dummy_betas_;

    /** @brief Full Screen-TRex result from the last select() call. */
    ScreenTRexSelectionResult screen_result_;

public:
    // ==========================================================================
    // Constructor / Destructor
    // ==========================================================================

    /**
     * @brief Construct a ScreenTRexSelector.
     *
     * @param X             Feature matrix (n × p) — Eigen::Map, not copied.
     * @param y             Response vector (n × 1) — Eigen::Map, copied internally.
     * @param screen_control Screen-TRex specific parameters.
     * @param trex_control  Algorithmic control (solver, dummy strategy, memmap).
     *                      Only STANDARD and PERMUTATION lloop_strategy are accepted.
     * @param seed          Random seed (< 0 for non-deterministic).
     * @param verbose       Enable verbose output.
     *
     * @throws std::invalid_argument if lloop_strategy is not STANDARD or PERMUTATION.
     */
    ScreenTRexSelector(
        Eigen::Map<Eigen::MatrixXd>& X,
        Eigen::Map<Eigen::VectorXd>& y,
        ScreenTRexControlParameter   screen_control = ScreenTRexControlParameter(),
        tc::TRexControlParameter     trex_control   = tc::TRexControlParameter(),
        int  seed    = -1,
        bool verbose = true
    );

    /** @brief Virtual destructor. */
    virtual ~ScreenTRexSelector() = default;

    /** @brief Deleted copy constructor. */
    ScreenTRexSelector(const ScreenTRexSelector&) = delete;

    /** @brief Deleted copy assignment. */
    ScreenTRexSelector& operator=(const ScreenTRexSelector&) = delete;

    /** @brief Deleted move constructor. */
    ScreenTRexSelector(ScreenTRexSelector&&) = delete;

    /** @brief Deleted move assignment. */
    ScreenTRexSelector& operator=(ScreenTRexSelector&&) = delete;

    // ==========================================================================
    // Main method
    // ==========================================================================

    /**
     * @brief Run Screen-TRex selection (T=1, L=p).
     *
     * @details Overrides the base class select().  Bypasses the T-loop and L-loop
     *          to run exactly one fixed-budget experiment set, then selects variables
     *          via ordinary thresholding or bootstrap CI on dummy betas.
     *
     * @return SelectionResult (sliced to base type). Use getScreenResult() for
     *         full Screen-TRex diagnostics.
     */
    virtual SelectionResult select() override;

    // ==========================================================================
    // Public Getter
    // ==========================================================================

    /**
     * @brief Retrieve the full Screen-TRex result from the last select() call.
     *
     * @return Const reference to ScreenTRexSelectionResult.
     */
    const ScreenTRexSelectionResult& getScreenResult() const;


protected:
    // =========================================================================
    // Core Screen-TRex Operations
    // =========================================================================

    /**
     * @brief Run K experiments at T=1, L=p and collect beta statistics.
     *
     * @details  Uses ExperimentRunner with collect_screen_betas=true.
     *           Populates accumulated_beta_mat_ and collected_dummy_betas_.
     *
     * @return ExperimentResults containing phi_T_mat and Phi.
     */
    er::ExperimentResults runScreenExperiments();


    /**
     * @brief Ordinary Screen-TRex: select variables where Phi > 0.5.
     *
     * @param Phi        Relative occurrences (p × 1).
     * @param result_out Result object to populate.
     */
    void performOrdinarySelection(
        const Eigen::VectorXd&    Phi,
        ScreenTRexSelectionResult& result_out
    ) const;


    /**
     * @brief Confidence-Based Screen-TRex: bootstrap CI on dummy betas.
     *
     * @details Grid-searches gamma ∈ [0, 1) for the smallest confidence level
     *          such that the number of selections ≤ ordinary Screen-TRex count.
     *
     * @param Phi        Relative occurrences (p × 1).
     * @param beta_means Averaged beta coefficients (p × 1).
     * @param result     Result object to populate with bootstrap diagnostics.
     */
    void performBootstrapSelection(
        const Eigen::VectorXd&    Phi,
        const Eigen::VectorXd&    beta_means,
        ScreenTRexSelectionResult& result
    );


    /**
     * @brief Compute estimated FDR = 1 / max(1, R).
     *
     * @param result Reference to result object (read selected_var, write estimated_FDR).
     */
    void computeEstimatedFDR(ScreenTRexSelectionResult& result) const;


    /**
     * @brief Calculate a normal bootstrap confidence interval.
     *
     * @param means_dist       Bootstrap distribution of means.
     * @param confidence_level Confidence level in (0, 1).
     *
     * @return Pair {lower_bound, upper_bound}.
     */
    std::pair<double, double> calculateBootstrapCI(
        const std::vector<double>& means_dist,
        double confidence_level
    ) const;


    /**
     * @brief Inverse normal CDF (probit).
     *
     * @details Uses Boost Math erf_inv: z = √2 · erfinv(2p − 1).
     *
     * @param p Probability in (0, 1).
     * @return Corresponding z-score.
     */
    double probit(double p) const;


    // =========================================================================
    // Dependency-Aware (DA) Helpers
    // =========================================================================

    /**
     * @brief Apply Dependency-Aware Phi adjustment (AR1 or EQUI variant).
     *
     * @param Phi Relative occurrences (modified in-place).
     */
    void applyDependencyAwareAdjustment(Eigen::VectorXd& Phi);

    /**
     * @brief Estimate AR(1) correlation coefficient from X.
     *
     * @details Row-wise Yule-Walker estimator, averaged over n rows.
     *
     * @return Absolute average AR(1) autocorrelation.
     */
    double estimateAR1Correlation() const;

    /**
     * @brief Estimate equi-correlation coefficient from X.
     *
     * @details Uses variance of row sums: ρ̄ = (‖row_sums‖² − p) / (p(p−1)).
     *
     * @return Average off-diagonal correlation.
     */
    double estimateEquiCorrelation() const;

    /**
     * @brief Compute DA adjustment deltas for AR(1) strategy.
     *
     * @details Window size κ = ⌈log(ρ_thr) / log(ρ)⌉.  For each j, finds the
     *          minimum |Phi[j] − Phi[k]| in [j−κ, j+κ].
     *
     * @param Phi Relative occurrences (p × 1).
     * @param rho AR(1) correlation estimate.
     *
     * @return DA_delta adjustment vector (p × 1).
     */
    Eigen::VectorXd computeDA_AR1_Delta(
        const Eigen::VectorXd& Phi, double rho) const;

    /**
     * @brief Compute DA adjustment deltas for equi-correlation strategy.
     *
     * @details Exact O(p²) global minimum |Phi[j] − Phi[k]|.
     *          Skipped entirely when |ρ| ≤ ρ_thr_DA.
     *
     * @param Phi Relative occurrences (p × 1).
     * @param rho Equi-correlation estimate.
     *
     * @return DA_delta adjustment vector (p × 1).
     */
    Eigen::VectorXd computeDA_EQUI_Delta(
        const Eigen::VectorXd& Phi, double rho) const;


    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate that lloop_strategy is STANDARD or PERMUTATION.
     *
     * @throws std::invalid_argument for unsupported strategies.
     */
    void validateScreenTRexStrategy() const;

};

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_screening */
// ===================================================================================
#endif /* TREX_SELECTOR_METHODS_TREX_SCREENING_TREX_SCREENING_HPP */
