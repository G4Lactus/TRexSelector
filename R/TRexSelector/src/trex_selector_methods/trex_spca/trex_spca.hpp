// =============================================================================
// trex_spca.hpp
// ==============================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_SPCA_HPP
#define TREX_SELECTOR_METHODS_TREX_SPCA_HPP
// ==============================================================================
/**
 * @file trex_spca.hpp
 *
 * @brief T-Rex Sparse PCA and Thresholded PCA selector class.
 *
 * @details Implementations are in trex_spca.cpp.
 */
// ==============================================================================

// std includes
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex base
#include <trex_selector_methods/trex_core/trex.hpp>

// ==============================================================================

// Namespace integration
namespace trex {
namespace trex_selector_methods {
namespace trex_spca {

namespace tc = trex::trex_selector_methods::trex_core;

// ==============================================================================

/**
 * @brief Selects which sparse loading strategy to use.
 */
enum class SPCAMode {
    /** @brief Ridge regression on the active set to compute loadings. */
    ActiveSet,

    /** @brief Keep and normalize the top |A| ordinary PCA loadings. */
    Thresholded
};


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


/**
 * @brief Algorithmic control parameters for the T-Rex Sparse PCA selector.
 */
struct TRexSPCAControlParameter {

    // ====================================
    // Loading Assembly
    // ====================================

    /** @brief Sparse loading strategy (default: ActiveSet). */
    SPCAMode mode = SPCAMode::ActiveSet;

    /** @brief Ridge regularization penalty for ActiveSet loading assembly (default: 1e-6). */
    double lambda2 = 1e-6;

    // ====================================
    // T-Rex Sub-Selection
    // ====================================

    /** @brief Random seed forwarded to each per-PC T-Rex run.
     *  Use seed < 0 for non-deterministic behavior (default: -1). */
    int seed = -1;

    /** @brief Algorithmic control parameters forwarded to each per-PC T-Rex run. */
    tc::TRexControlParameter trex_ctrl;
};


/**
 * @class TRexSPCA
 *
 * @brief T-Rex Sparse PCA orchestrator.
 *
 * @details
 *  For each principal component 1,...,M:
 *    1. Compute ordinary PCA to obtain the PC response vector z_m.
 *    2. Run T-Rex selector with target FDR to obtain a sparse active set A_m.
 *    3. Assemble the sparse loading vector v_m:
 *       - ActiveSet:   Ridge regression of z_m on X_{A_m}, normalized.
 *       - Thresholded: Top |A_m| ordinary loadings, normalized.
 *    4. Compute sparse scores Z_m = X_{A_m} v_m.
 *
 *  After all M components are assembled, explained variance is adjusted for
 *  non-orthogonality via a QR decomposition of the full score matrix Z.
 *
 *  All entry points are static; no instance state is required.
 */
class TRexSPCA {
public:

    /**
     * @brief Run T-Rex Sparse PCA selection.
     *
     * @param X         Design matrix (n x p), assumed to be centered.
     * @param M         Number of sparse principal components to extract.
     * @param tFDR      Target false discovery rate.
     * @param spca_ctrl Control parameters (loading mode, lambda2, seed, T-Rex params).
     *
     * @return TRexSPCAResult containing sparse PCs, loadings, active sets, and
     *         explained variance.
     */
    static TRexSPCAResult select(Eigen::Map<Eigen::MatrixXd>& X,
                                 Eigen::Index M,
                                 double tFDR,
                                 TRexSPCAControlParameter spca_ctrl = TRexSPCAControlParameter());

private:

    /**
     * @brief Computes rigorous loadings via Ridge regression on the selected active set.
     *
     * @param X          Design matrix (n x p).
     * @param z_m        Ordinary PC response vector (n x 1).
     * @param active_set Indices of the active set variables.
     * @param lambda2    Ridge penalty parameter.
     * @param v_out      Output loading vector (p x 1), written in-place.
     */
    static void assemble_active_set_loadings(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                             const Eigen::Ref<const Eigen::VectorXd>& z_m,
                                             const std::vector<Eigen::Index>& active_set,
                                             double lambda2,
                                             Eigen::Ref<Eigen::VectorXd> v_out);

    /**
     * @brief Computes thresholded loadings by keeping only the top |A| ordinary loadings.
     *
     * @param ord_v       Ordinary PCA loading vector (p x 1).
     * @param active_size Cardinality of the active set |A|.
     * @param v_out       Output loading vector (p x 1), written in-place.
     */
    static void assemble_thresholded_loadings(const Eigen::Ref<const Eigen::VectorXd>& ord_v,
                                              Eigen::Index active_size,
                                              Eigen::Ref<Eigen::VectorXd> v_out);
};


// ==============================================================================
} /* namespace trex_spca */
} /* namespace trex_selector_methods */
} /* namespace trex */
// ==============================================================================
#endif /* TREX_SELECTOR_METHODS_TREX_SPCA_HPP */
