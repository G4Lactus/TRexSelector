// ===================================================================================
// trex_da.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_DA_HPP
#define TREX_SELECTOR_METHODS_TREX_DA_HPP
// ===================================================================================
/**
 * @file trex_da.hpp
 *
 * @brief Dependency-Aware T-Rex (DA-TRex) Selector.
 *
 * @details
 *  Extends the base class TRexSelector with dependency-aware deflation that accounts
 *  for the correlation structure among predictors.
 *
 *  Five methods are supported:
 *  - AR1:          AR(1) windowed deflation (ordered predictors).
 *  - EQUI:         Global equicorrelation deflation.
 *  - BT:           Binary-tree dendrogram from hierarchical clustering.
 *  - NN:           Nearest-neighbour threshold sweep on the correlation matrix.
 *  - PRIOR_GROUPS: User-supplied group structure at multiple hierarchy levels.
 *
 *  BT, NN, and PRIOR_GROUPS ("BT-style") perform 3D calibration over
 *  (T_stop, V, rho_grid);
 *  AR1 and EQUI stay on the 2D (T_stop, V) grid of the base class.
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <limits>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex base class
#include <trex_selector_methods/trex_core/trex.hpp>

// Hierarchical clustering
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_da {

// Namespace aliases
namespace tc  = trex::trex_selector_methods::trex_core;
namespace er  = trex::trex_selector_methods::utils::experiment_runner;
namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;

// ===================================================================================
// Enums
// ===================================================================================

/**
 * @brief Dependency-aware correction methods.
 */
enum class DAMethod {
    AR1,           ///< AR(1) windowed correction for ordered predictors.
    EQUI,          ///< Equicorrelation correction (all-vs-all).
    BT,            ///< Binary-tree dendrogram-based clustering.
    NN,            ///< Nearest-neighbour correlation threshold sweep.
    PRIOR_GROUPS   ///< User-supplied group labels at multiple hierarchy levels.
};

// ===================================================================================
// Control Parameters
// ===================================================================================

/**
 * @brief DA-specific control parameters.
 */
struct TRexDAControlParameter {

    /** @brief Dependency-aware method (default: BT). Ignored when prior_groups is non-empty. */
    DAMethod method = DAMethod::BT;

    /** @brief Correlation coefficient for AR1/EQUI. NaN = auto-estimate from X. */
    double cor_coef = std::numeric_limits<double>::quiet_NaN();

    /** @brief Threshold below which equi-correlation correction is skipped (default: 0.02). */
    double rho_thr_DA = 0.02;

    /** @brief Linkage method for BT clustering (default: Single).
     *  Supported: Single, Complete, Average (UPGMA), WPGMA (McQuitty).
     */
    hac::LinkageMethod hc_linkage = hac::LinkageMethod::Single;

    /** @brief Number of dendrogram / NN grid points (default: 0 = min(20, p)). */
    std::size_t hc_grid_length = 0;

    /** @brief Prior group labels: L vectors of length p, one per hierarchy level (fine to coarse).
     *  When non-empty, the method field is ignored and full 3D calibration is used.
     */
    std::vector<std::vector<Eigen::Index>> prior_groups;

    /** @brief Optional semantic labels for each prior-group level (e.g. correlation thresholds).
     *  Must have the same length as prior_groups. Defaults to 0, 1, 2, ... */
    std::vector<double> rho_grid_labels;
};

// ===================================================================================
// DA Setup Result (internal)
// ===================================================================================

/**
 * @brief Internal structure holding the precomputed neighbourhood information.
 */
struct DASetupResult {
    /** @brief True for BT, NN, or PRIOR_GROUPS (3D calibration paths). */
    bool use_BT_style = false;

    /** @brief Group-mate indices: gr_j_list[j][rho_level] = vector of neighbour
     *  indices for variable j.
     */
    std::vector<std::vector<std::vector<Eigen::Index>>> gr_j_list;

    /** @brief Correlation-level grid (length rho_grid_len). Null-like (size 0) for AR1/EQUI. */
    Eigen::VectorXd rho_grid;

    /** @brief Length of the rho_grid (0 for AR1/EQUI). */
    std::size_t rho_grid_len = 0;

    /** @brief Reference index into rho_grid for the 75 % operating point. */
    std::size_t opt_point_BT = 0;

    /** @brief Estimated or user-supplied correlation coefficient. */
    double cor_coef = std::numeric_limits<double>::quiet_NaN();

    /** @brief AR(1) window half-width (0 unless AR1). */
    std::size_t kap = 0;
};

// ===================================================================================
// DA Correction Result (internal)
// ===================================================================================

/**
 * @brief Output of the DA deflation step.
 */
struct DACorrectionResult {
    /** @brief Deflated phi_T_mat (p x T_stop). Modified in-place for AR1/EQUI.
     *  Copy for BT-style.
     */
    Eigen::MatrixXd phi_T_mat;

    /** @brief Deflated Phi (p x 1). */
    Eigen::VectorXd Phi;

    /** @brief BT-style only: deflated Phi per rho level (p x rho_grid_len). */
    Eigen::MatrixXd Phi_BT;

    /** @brief BT-style only: deflated phi_T_mat per rho level.
     *  phi_T_array_BT[rho] is a (p x T_stop) matrix.
     */
    std::vector<Eigen::MatrixXd> phi_T_array_BT;
};


// ===================================================================================
// TRexDASelector
// ===================================================================================

/**
 * @brief Dependency-Aware T-Rex Selector.
 *
 * @details
 *  Inherits from TRexSelector. The select() method is overridden to insert a
 *  dependency-aware deflation step between the random-experiment aggregation and
 *  the FDP estimation.
 *. For BT-style methods the calibration is performed over a 3D grid (T_stop x V x rho_grid);
 *  for AR1/EQUI the standard 2D grid is used with modified phi_T / Phi values.
 */
class TRexDASelector : public tc::TRexSelector {

public:

    // ============================================================
    // Result Structure
    // ============================================================

    /**
     * @brief Extended result for DA-TRex, adding rho_grid diagnostics.
     */
    struct DASelectionResult : public SelectionResult {
        /** @brief Selected rho threshold (NaN for AR1/EQUI). */
        double rho_thresh = std::numeric_limits<double>::quiet_NaN();

        /** @brief Correlation-level grid used (empty for AR1/EQUI). */
        Eigen::VectorXd rho_grid;

        /** @brief DA method that was applied. */
        DAMethod method = DAMethod::BT;

        /** @brief Estimated or supplied correlation coefficient. */
        double cor_coef = std::numeric_limits<double>::quiet_NaN();

        /** @brief BT-style: FDP estimates per rho level.
         * FDP_hat_array_BT[rho] = (T_stop x V_len).
         */
        std::vector<Eigen::MatrixXd> FDP_hat_array_BT;

        /** @brief BT-style: relative occurrences per rho level.
         * Phi_array_BT[rho] = (T_stop x p).
         */
        std::vector<Eigen::MatrixXd> Phi_array_BT;

        /** @brief BT-style: selection-count array. R_array[rho] = (T_stop x V_len). */
        std::vector<Eigen::MatrixXd> R_array_BT;

        /** @brief Virtual destructor. */
        ~DASelectionResult() override = default;
    };


    // ============================================================
    // Constructor / Destructor
    // ============================================================

    /**
     * @brief Construct a TRexDASelector.
     *
     * @param X             Feature matrix (n x p) — Eigen::Map, not copied.
     * @param y             Response vector (n x 1) — Eigen::Map, copied internally.
     * @param tFDR          Target FDR level (default: 0.1).
     * @param da_control    DA-specific control parameters.
     * @param trex_control  Base T-Rex algorithmic control parameters.
     * @param seed          Random seed (< 0 for non-deterministic).
     * @param verbose       Enable verbose output (default: true).
     */
    TRexDASelector(
        Eigen::Map<Eigen::MatrixXd>& X,
        Eigen::Map<Eigen::VectorXd>& y,
        double tFDR = 0.1,
        TRexDAControlParameter da_control  = TRexDAControlParameter(),
        tc::TRexControlParameter trex_control = tc::TRexControlParameter(),
        int  seed    = -1,
        bool verbose = true
    );

    /** @brief Virtual destructor. */
    ~TRexDASelector() override = default;

    /** @brief Deleted copy constructor. */
    TRexDASelector(const TRexDASelector&) = delete;

    /** @brief Deleted copy assignment operator. */
    TRexDASelector& operator=(const TRexDASelector&) = delete;

    /** @brief Deleted move constructor. */
    TRexDASelector(TRexDASelector&&) = delete;

    /** @brief Deleted move assignment operator. */
    TRexDASelector& operator=(TRexDASelector&&) = delete;


    // ============================================================
    // Main Method
    // ============================================================

    /**
     * @brief Run DA-TRex selection.
     *
     * @return SelectionResult (sliced to base type).
     *         Use getDAResult() for the full DA diagnostics.
     */
    SelectionResult select() override;

    /**
     * @brief Retrieve the full DA result from the last select() call.
     *
     * @return Const reference to DASelectionResult.
     */
    const DASelectionResult& getDAResult() const noexcept { return da_result_; }


protected:

    // ============================================================
    // DA Data Members
    // ============================================================

    /** @brief DA-specific control parameters. */
    TRexDAControlParameter da_ctrl_;

    /** @brief Precomputed neighbourhood / correlation structure. */
    DASetupResult da_setup_;

    /** @brief Full DA result from the last select() call. */
    DASelectionResult da_result_;


    // ============================================================
    // DA Setup — Neighbourhood / Correlation Dispatch
    // ============================================================

    /**
     * @brief Build the dependency structure from X (once, before the loops).
     *
     * Dispatches to:
     *  - setupDA_PriorGroups() if prior_groups is non-empty.
     *  - setupDA_BT()          for DAMethod::BT.
     *  - setupDA_NN()          for DAMethod::NN.
     *  - setupDA_AR1()         for DAMethod::AR1.
     *  - setupDA_EQUI()        for DAMethod::EQUI.
     */
    void setupDA();

    /** @brief Route 1: prior group labels supplied by the user. */
    void setupDA_PriorGroups();

    /** @brief Route 2: dendrogram-based clustering on |cor(X)|. */
    void setupDA_BT();

    /** @brief Route 3: nearest-neighbour correlation threshold sweep. */
    void setupDA_NN();

    /** @brief Route 4: AR(1) model — estimate ρ and compute window κ. */
    void setupDA_AR1();

    /** @brief Route 5: equicorrelation — estimate ρ̄. */
    void setupDA_EQUI();


    // ============================================================
    // DA Correction
    // ============================================================

    /**
     * @brief Apply the dependency-aware deflation to experiment results.
     *
     * @param phi_T_mat Relative occurrence matrix (p x T_stop).
     * @param Phi       Relative occurrence vector at T_stop (p x 1).
     * @param T_stop    Current stopping time.
     *
     * @return DACorrectionResult with deflated values.
     */
    DACorrectionResult daCorrect(
        const Eigen::MatrixXd& phi_T_mat,
        const Eigen::VectorXd& Phi,
        std::size_t T_stop) const;


    // ============================================================
    // FDP Computation for BT-style
    // ============================================================

    /**
     * @brief Compute Phi_prime and FDP_hat across the rho_grid for BT-style methods.
     *
     * @param da_corr   DA correction result (contains phi_T_array_BT, Phi_BT).
     * @param num_dummies Number of dummy variables used.
     *
     * @return Pair of (Phi_prime_BT, FDP_hat_BT) where each is a matrix (p x rho_grid_len)
     *         and (V_len x rho_grid_len), respectively.
     */
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> computeFDP_BT(
        const DACorrectionResult& da_corr,
        std::size_t num_dummies) const;


    // ============================================================
    // Variable Selection for BT-style
    // ============================================================

    /**
     * @brief Select variables over the 3D grid (T, V, rho_grid) for BT-style methods.
     *
     * @param FDP_hat_array_BT  FDP estimates: one (T_stop x V_len) matrix per rho level.
     * @param Phi_array_BT      Deflated Phi: one (T_stop x p) matrix per rho level.
     * @param voting_grid       Voting grid V.
     * @param rho_grid          Correlation-level grid.
     * @param T_stop            Stopping time.
     *
     * @return DASelectionResult with selected variables, thresholds, and R_array.
     */
    DASelectionResult selectVariables_BT(
        const std::vector<Eigen::MatrixXd>& FDP_hat_array_BT,
        const std::vector<Eigen::MatrixXd>& Phi_array_BT,
        const Eigen::VectorXd& voting_grid,
        const Eigen::VectorXd& rho_grid,
        std::size_t T_stop) const;


    // ============================================================
    // Correlation Estimation Helpers
    // ============================================================

    /** @brief Estimate AR(1) coefficient from X via row-wise Yule-Walker. */
    double estimateAR1Correlation() const;

    /** @brief Estimate mean pairwise correlation from X. */
    double estimateEquiCorrelation() const;


    // ============================================================
    // Validation
    // ============================================================

    /** @brief Validate DA-specific parameters. */
    void validateDAParameters() const;


    // ============================================================
    // Clustering Dispatch Helper
    // ============================================================

    /**
     * @brief Run hierarchical clustering with the configured linkage method.
     *
     * @return Sorted merge matrix (p-1 merge steps).
     */
    std::vector<hac::MergeStep> runClustering() const;

// -----------------------------------------------------------------------
}; /* End of class TRexDASelector */

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_da */
// ===================================================================================

#endif /* TREX_SELECTOR_METHODS_TREX_DA_HPP */
