// =============================================================================
// trex_spca.hpp
// ==============================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_SPCA_HPP
#define TREX_SELECTOR_METHODS_TREX_SPCA_HPP
// ==============================================================================
/**
 * @file trex_spca.hpp
 *
 * @brief T-Rex Sparse PCA selector class.
 *
 * @details
 *  Implements from the T-Rex SPCA paper:
 *    5. T-Rex+GVS(EN) + ActiveSet   loading assembly
 *    6. T-Rex+GVS(EN) + Thresholded loading assembly
 *
 *  Implementations are in trex_spca.cpp.
 */
// ==============================================================================

// std includes
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Centering (stored as member for exception-safe restoration)
#include <ml_methods/standardization/lp_norm_scaler.hpp>

// TRex core (TRexControlParameter)
#include <trex_selector_methods/trex_core/trex.hpp>

// TRex GVS (TRexGVSSelector, TRexGVSControlParameter, GVSType)
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

// ==============================================================================

namespace trex::trex_selector_methods::trex_spca {

// Namespace aliases
namespace tc         = trex::trex_selector_methods::trex_core;
namespace tg         = trex::trex_selector_methods::trex_gvs;
namespace std_scaler = trex::ml_methods::standardization;

// ==============================================================================
// Enums
// ==============================================================================

/**
 * @brief Sparse loading assembly strategy.
 */
enum class SPCAMode {
    /** @brief Ridge regression of z_m on X_{A_m}, normalized (methods 5, 7). */
    ActiveSet,

    /** @brief Top |A_m| ordinary PCA loadings, normalized (methods 6, 8). */
    Thresholded
};


// ==============================================================================
// Result Structure
// ==============================================================================

/**
 * @brief Container for final sparse PCA components.
 */
struct TRexSPCAResult {
    /** @brief Sparse PC score matrix (n x M). */
    Eigen::MatrixXd Z;

    /** @brief Sparse loading matrix (p x M). */
    Eigen::MatrixXd V;

    /** @brief Support indices per PC. */
    std::vector<std::vector<Eigen::Index>> active_sets;

    /** @brief Marginal adjusted explained variance per component (M x 1). */
    Eigen::VectorXd adjusted_ev;

    /** @brief Cumulative percentage of explained variance (M x 1). */
    Eigen::VectorXd cumulative_ev;
};


// ==============================================================================
// Control Parameters
// ==============================================================================

/**
 * @brief Algorithmic control parameters for the T-Rex Sparse PCA selector.
 */
struct TRexSPCAControlParameter {

    // ====================================
    // Loading Assembly
    // ====================================

    /** @brief Sparse loading assembly strategy (default: ActiveSet). */
    SPCAMode mode = SPCAMode::ActiveSet;

    /** @brief Ridge regularization penalty for the ActiveSet loading assembly step.
     *  Used in the Ridge regression `(X_A^T X_A + lambda2_ridge_loadings * I)^{-1} X_A^T z_m`
     *  to obtain sparse loading vectors on the selected active set.
     *  (default: 1e-6).
     */
    double lambda2_ridge_loadings = 1e-6;


    // ====================================
    // GVS Sub-Selector
    // ====================================

    /** @brief GVS-specific control parameters forwarded to each per-PC TRexGVSSelector run.
     *
     *  EN + TENET is always used: gvs_ctrl.gvs_type is overridden to EN
     *  and trex_ctrl.solver_type is overridden to TENET internally.
     *  (IEN is not used by TRexSPCA.)
     *
     *  The GVS ridge penalty is controlled via:
     *    - gvs_ctrl.lambda_2       : set > 0.0 to supply a fixed value; 0.0 (default)
     *                                triggers auto-determination.
     *    - gvs_ctrl.lambda2_method : selects the auto-determination strategy
     *                                (GCV, CV_MIN, CV_1SE; default: CV_1SE).
     */
    tg::TRexGVSControlParameter gvs_ctrl;

    // ====================================
    // Base T-Rex Sub-Selector
    // ====================================

    /** @brief Base T-Rex algorithmic control parameters forwarded to each per-PC
     *  TRexGVSSelector run.
     */
    tc::TRexControlParameter trex_ctrl;
};


// ==============================================================================
// TRexSPCA
// ==============================================================================

/**
 * @class TRexSPCA
 *
 * @brief T-Rex Sparse PCA selector.
 *
 * @details
 *  For each principal component m = 1, ..., M:
 *    1. Compute ordinary PCA to obtain the PC response vector z_m.
 *    2. Run TRexGVSSelector (EN) with target FDR to obtain a sparse
 *       active set A_m.
 *    3. Assemble the sparse loading vector v_m:
 *       - SPCAMode::ActiveSet:   Ridge regression of z_m on X_{A_m}, normalized.
 *       - SPCAMode::Thresholded: Top |A_m| ordinary loadings, normalized.
 *    4. Compute sparse PC scores: Z_m = X_{A_m} * v_m.
 *
 *  After all M components are assembled, explained variance is adjusted for
 *  non-orthogonality via a QR decomposition of the full score matrix Z.
 *
 *  Data contract:
 *    - X is centered in-place at the start of select() and restored on return
 *      (normal path) or by the destructor (exception path).
 *    - X is not owned; the Map must remain valid for the lifetime of the object.
 */
class TRexSPCA {
public:

    // ============================================================
    // Constructor / Destructor
    // ============================================================

    /**
     * @brief Construct a TRexSPCA selector.
     *
     * @param X          Design matrix (n x p) — Eigen::Map, not copied.
     *                   X is centered in-place during select() and restored on return.
     * @param M          Number of sparse principal components to extract.
     * @param tFDR       Target false discovery rate.
     * @param spca_ctrl  Control parameters (loading mode, GVS type, lambda2, T-Rex params).
     * @param seed       Random seed for reproducibility.
     *                   seed >= 0 : per-PC deterministic runs via mix_seed(seed, m).
     *                   seed < 0  : non-deterministic (default: -1).
     * @param verbose    Enable verbose output (default: false).
     *
     * @throws std::invalid_argument if M > min(n, p).
     */
    TRexSPCA(Eigen::Map<Eigen::MatrixXd>& X,
             Eigen::Index M,
             double tFDR,
             TRexSPCAControlParameter spca_ctrl = TRexSPCAControlParameter(),
             int seed = -1,
             bool verbose = false);

    /** @brief Destructor. Restores X if select() was interrupted by an exception. */
    ~TRexSPCA();

    /** @brief Deleted copy constructor. */
    TRexSPCA(const TRexSPCA&) = delete;

    /** @brief Deleted copy assignment operator. */
    TRexSPCA& operator=(const TRexSPCA&) = delete;

    /** @brief Deleted move constructor. */
    TRexSPCA(TRexSPCA&&) = delete;

    /** @brief Deleted move assignment operator. */
    TRexSPCA& operator=(TRexSPCA&&) = delete;


    // ============================================================
    // Main Method
    // ============================================================

    /**
     * @brief Run T-Rex Sparse PCA selection.
     *
     * @return TRexSPCAResult containing sparse PCs, loadings, active sets, and
     *         explained variance.
     */
    TRexSPCAResult select();

    /**
     * @brief Retrieve the result from the last select() call.
     *
     * @return Const reference to TRexSPCAResult.
     */
    const TRexSPCAResult& getResult() const noexcept { return result_; }


private:

    // ============================================================
    // Data Members
    // ============================================================

    /** @brief Non-owning pointer to design matrix X (n x p). */
    Eigen::Map<Eigen::MatrixXd>* X_;

    /** @brief Number of sparse PCs to extract. */
    const Eigen::Index M_;

    /** @brief Target false discovery rate. */
    const double tFDR_;

    /** @brief Algorithmic control parameters. */
    TRexSPCAControlParameter spca_ctrl_;

    /** @brief Random seed (< 0 for non-deterministic). */
    const int seed_;

    /** @brief Verbose output flag. */
    const bool verbose_;

    /** @brief Result from the last select() call. */
    TRexSPCAResult result_;

    /** @brief Centering scaler; stored as member so the destructor can call
     *  inverse_transform_inplace if select() was interrupted by an exception. */
    std_scaler::LpNormScaler scaler_;

    /** @brief True while X is in centered state (between transform and inverse_transform
     *  inside select()). Guards against leaving X permanently mutated on exceptions. */
    bool X_is_centered_{false};


    // ============================================================
    // Private Helpers
    // ============================================================

    /**
     * @brief Assemble loadings via Ridge regression on the active set.
     *
     * @param X_ref      Centered design matrix (n x p).
     * @param z_m        Ordinary PC response vector (n x 1).
     * @param active_set Selected variable indices.
     * @param v_out      Output loading vector (p x 1), written in-place.
     */
    void assembleActiveSetLoadings(const Eigen::Ref<const Eigen::MatrixXd>& X_ref,
                                   const Eigen::Ref<const Eigen::VectorXd>& z_m,
                                   const std::vector<Eigen::Index>& active_set,
                                   Eigen::Ref<Eigen::VectorXd> v_out) const;

    /**
     * @brief Assemble thresholded loadings: top |A| ordinary loadings, normalized.
     *
     * @param ord_v       Ordinary PCA loading vector (p x 1).
     * @param active_size Cardinality of the active set |A|.
     * @param v_out       Output loading vector (p x 1), written in-place.
     */
    void assembleThresholdedLoadings(const Eigen::Ref<const Eigen::VectorXd>& ord_v,
                                     Eigen::Index active_size,
                                     Eigen::Ref<Eigen::VectorXd> v_out) const;

    /**
     * @brief Compute adjusted explained variance via QR decomposition of Z.
     *
     * @param result  TRexSPCAResult to update (Z must already be assembled).
     * @param X_ref   Centered design matrix (n x p) for total variance computation.
     */
    void adjustExplainedVariance(TRexSPCAResult& result,
                                 const Eigen::Ref<const Eigen::MatrixXd>& X_ref) const;
};

// ==============================================================================
} /* End of namespace trex::trex_selector_methods::trex_spca */
// ==============================================================================
#endif /* End of TREX_SELECTOR_METHODS_TREX_SPCA_HPP */
