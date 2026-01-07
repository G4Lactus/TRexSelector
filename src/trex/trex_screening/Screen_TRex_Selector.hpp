#ifndef SCREEN_TREX_SELECTOR_HPP
#define SCREEN_TREX_SELECTOR_HPP

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <utility>

#include <Eigen/Dense>

#include "TRex_Selector.hpp"


// =============================================================
// Enums & Control Structures for Screen-TRex
// =============================================================

/**
 * @brief Screen-TRex Methods available.
 */
enum class ScreenTRexMethod {
    TREX,           /** Standard T-Rex */
    TREX_DA_AR1,    /** Dependency-Aware T-Rex (Auto-Regressive 1) */
    TREX_DA_EQUI    /** Dependency-Aware T-Rex (Equi-Correlated) */
    /* FUTURE: TREX_DA_EQUI_FAST  // O(p log p) approximation */
};


/** @brief Configuration for Screen-TRex specific parameters */
struct ScreenTRexControlParameter {
    /** @brief If true, perform the Confidence-Based Screen-TRex (bootstrapping) */
    bool use_bootstrap_CI = false;

    /** @brief Number of bootstrap resamples (default: 1000) */
    std::size_t R_boot = 1000;

    /** @brief Grid step size for confidence level search (0 to 1) */
    double ci_grid_step = 0.001;

    /** @brief Screen-TRex Method to use */
    ScreenTRexMethod trex_method = ScreenTRexMethod::TREX;

    /** @brief Correlation threshold for DA methods (default: 0.02) */
    double rho_thr_DA = 0.02;

    /** @brief Manually specified correlation coefficient. */
    double cor_coef = std::numeric_limits<double>::quiet_NaN();
};


/** @brief Result structure for Screen-TRex.
 *
 *  @details
 *  Extends the standard result with Screen-TRex specific diagnostics.
 */
struct ScreenTRexSelectionResult : public TRexSelector::SelectionResult {
    double estimated_FDR = 0.0;             /** Estimated FDR (1 / max(1, selected)) */
    bool used_bootstrap = false;            /** Whether bootstrap was applied */
    double confidence_level = 0.0;          /** Selected confidence (if bootstrap used) */
    Eigen::MatrixXd beta_mat;               /** Averaged beta coefficients */
    Eigen::VectorXd dummy_betas;            /** Collected non-zero dummy coefficients used
                                                for bootstrap */
    double estimated_correlation = 0.0;     /** Estimated/used correlation coefficient
                                                (for DA methods) */
};


/**
 * @brief Screen-TRex Selector.
 *
 * @details
 * Implements the Screen-TRex selector for fast screening of high-dimensional data.
 * Supports ordinary TRex and dependency-aware variants for AR(1) and equi-correlated data.
 *
 * Variants:
 * 1. Ordinary Screen-TRex: Selects with relative occurrence Phi > 0.5 at T = 1.
 * 2. Confidence-Based Screen-TRex: Uses bootstrapped dummy coefficients.
 * 3. DA-AR1: Adjusts Phi based on local neighborhood dependencies.
 * 4. DA-EQUI: Adjusts Phi based on global dependencies.
 */
class ScreenTRexSelector : public TRexSelector {

protected:
    // ==========================================================================
    // Screen-TRex Members
    // ==========================================================================

    /** @brief Screen-TRex control parameters */
    ScreenTRexControlParameter screen_ctrl_;

    //Internal storage for betas from experiments
    /** @brief Sum of betas for original vars (p x 1) */
    Eigen::MatrixXd accumulated_beta_mat_;

    /** @brief Non-zero dummy coefficients */
    std::vector<double> collected_dummy_betas_;

    /** @brief Storage for the full specific result */
    ScreenTRexSelectionResult screen_result_;

public:
    // ==========================================================================
    // Constructor
    // ==========================================================================

    /**
     * @brief Construct a new Screen_TRex_Selector object
     *
     * @param X Feature matrix (n x p) - accepts Eigen::Map.
     * @param y Response vector (n x 1 ) - accepts Eigen::Map.
     * @param screen_control Screen-TRex control parameters.
     * @param trex_control TRex control parameters.
     * @param solver_control Solver type and specific math config.
     * @param seed Random seed (< 0 for random seed).
     * @param verbose Enable verbose output (default: true).
     */
    ScreenTRexSelector(Eigen::Map<Eigen::MatrixXd>& X,
                       Eigen::Map<Eigen::VectorXd>& y,
                       ScreenTRexControlParameter screen_control = ScreenTRexControlParameter(),
                       TRexControlParameter trex_control = TRexControlParameter(),
                       SolverControl solver_control = SolverControl(),
                       int seed = -1,
                       bool verbose = true
    );

    /** @brief Virtual destructor for proper polymorphic behavior */
    virtual ~ScreenTRexSelector() = default;

    // Delete copy semantics
    /** @brief Delete copy constructor */
    ScreenTRexSelector(const ScreenTRexSelector&) = delete;

    /** @brief Delete copy assignment operator */
    ScreenTRexSelector& operator=(const ScreenTRexSelector&) = delete;

    // Delete Move semantics
    /** @brief Delete move constructor */
    ScreenTRexSelector(ScreenTRexSelector&&) = delete;

    /** @brief Delete move assignment operator */
    ScreenTRexSelector& operator=(ScreenTRexSelector&&) = delete;

    // ==========================================================================
    // Main method
    // ==========================================================================

    /**
     * @brief Run Screen-TRex selection (Fast Logic).
     *
     * @details This overrides the base class virtual method. It forces the execution
     * of the fast Screen-TRex logic (T = 1, L = p) instead of the T-Loop.
     *
     * @return SelectionResult (sliced to base type). Use getScreenResult() to access
     *         full Screen-TRex specific fields.
     */
    virtual SelectionResult select() override;

    // ==========================================================================
    // Screen-TRex Public Getters
    // ==========================================================================

    /**
     * @brief Retrieve the detailed Screen-TRex result from the last run.
     *
     * @return Const reference to the specific result structure containing beta_mat,
     *         est_FDR, etc.
     */
    const ScreenTRexSelectionResult& getScreenResult() const;


protected:
    // =========================================================================
    // Screen-TRex Internal Methods
    // =========================================================================

    /** @brief Run K experiments with T = 1 and capture beta coefficients. */
    void runScreenExperiments();


    /**
     * @brief Performs boostrap filtering on the selectionn.
     *
     * @param Phi Vector of relative occurrences (p x 1).
     * @param beta_means Vector of averaged coefficients.
     * @param result_out Reference to result object to populate with bootstrap stats.
     */
    void performBootstrapSelection(
        const Eigen::VectorXd& Phi,
        const Eigen::VectorXd& beta_means,
        ScreenTRexSelectionResult& result_out
    );


    /**
     * @brief Performs standard thresholding (Ordinary Screen-TRex).
     *
     * @details Selects variables where Phi > 0.5.
     *
     * @param Phi Vector of relative occurrences (p x 1).
     * @param result_out Reference to result object to populate with selection.
     */
    void performOrdinarySelection(
        const Eigen::VectorXd& Phi,
        ScreenTRexSelectionResult& result_out
    ) const;


    /**
     * @brief Computes the estimated FDR based on selection count.
     *
     * @details Estimated FDR = 1 / max(1, R), where R is the number of selected variables.
     *
     * @param result Reference to result object to populate with estimated FDR.
     */
    void computeEstimatedFDR(
        ScreenTRexSelectionResult& result
    ) const;


    /**
     * @brief Calculates Confidence Interval for a given level.
     *
     * @param means_dist Distribution of boostrap means.
     * @param confidence_level  The confidence level (0,1).
     *
     * @return Pair {lower_bound, upper_bound} of the CI.
     */
    std::pair<double, double> calculateBoostrapCI(
        const std::vector<double>& means_dist,
        double confidence_level
    ) const;


    /**
     * @brief Inverse CDF to get z-scores from probabilities.
     *
     * @details Uses Boost Math library's normal distribution quantile function.
     *          Calculates the Z-score such that P(Z < z) = p.
     *          Relies on: z = sqrt(2) * erfinv(2p - 1)
     *
     * @param p Probability value (0 < p < 1).
     *
     * @return Corresponding z-score.
     */
    double probit(double p) const;


    // =========================================================================
    // Dependency Aware (DA) Helpers
    // =========================================================================

    /**
     * @brief Estimates the AR(1) correlation coefficient (rho) from X.
     *
     * @details Computes lag-1 autocorrelation per sample (row), and
     *          and averaged over n.
     */
    double estimateAR1Correlation() const;

    /**
     * @brief Estimates the Equi-correlation coefficient from X.
     *
     * @details Uses variance of row sums to compute average off-diagonal
     *          correlation in O(np) rather O(p^2) for full matrix.
     */
    double estimateEquiCorrelation() const;

    /**
     * @brief Applies Dependency-Aware adjustment to Phi.
     *
     * @details Modifies phi_T_mat and Phi in place based on
     *          neighbor similarity.
     *
     * @param Phi Vector of relative occurrences.
     */
    void applyDependencyAwareAdjustment(Eigen::VectorXd& Phi);

    /**
     * @brief Computes DA adjustment factors for AR(1) strategy.
     *
     * @details Uses local window of size kap = ceil(log(rho_thr) / log(rho))
     *          to compute minimum difference between Phi[j] and its neighbors.
     *
     * @param Phi Vector of relative occurrences.
     * @param rho Estimated AR(1) correlation coefficient.
     *
     * @return Vector of DA_delta adjustment factors (p x 1).
     */
    Eigen::VectorXd computeDA_AR1_Delta(const Eigen::VectorXd&, double rho) const;

    /**
     * @brief Computes DA adjustment factors for Equi-correlation strategy.
     *
     * @details Compares each variable against all others (exact O(p^2))
     *          to find minimum difference in Phi. Uses vectorization.
     *
     * @param Phi Vector of relative occurrences.
     * @param rho Estimated equi-correlation coefficient.
     *
     * @return Vector of DA_delta adjustment factors (p x 1).
     */
    Eigen::VectorXd computeDA_EQUI_Delta(const Eigen::VectorXd&, double rho) const;

};

#endif /* End of SCREEN_TREX_SELECTOR_HPP */
