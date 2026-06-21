// ===================================================================================
// pca.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_PCA_HPP
#define TREX_ML_METHODS_PCA_HPP
// ===================================================================================
/**
 * @file pca.hpp
 * @brief Principal Component Analysis (PCA) implementation using SVD solver.
 *        Supports optional in-place centering with automatic restoration via RAII.
 *        The aim is to prevent data corruption in workflows where PCA is a subroutine.
 */
// ===================================================================================

// std includes
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/svd/svd.hpp>


// ===================================================================================

// Namespace declarations
namespace trex {
namespace ml_methods {
namespace pca {

// ===================================================================================

/** @brief Struct to hold PCA results */
struct PCAResult {
    /** @brief Transformed data (principal components scores) */
    Eigen::MatrixXd Z;

    /** @brief Principal component loadings (right singular vectors, p x M) */
    Eigen::MatrixXd V;

    /** @brief Explained variance per component (eigenvalues of sample covariance) */
    Eigen::VectorXd explained_variance;
};


/**
 * @class PCA
 *
 * @brief Stateful Principal Component Analysis with optional RAII centering.
 *
 * @details Fits a truncated PCA model via the dimensionality-aware SVDSolver.
 *          Centering is controlled at construction time:
 *          - PCA{}        — standalone use: centers X in-place during fit(),
 *                           stores column means, and restores X on destruction
 *                           (or via an explicit restore() call).
 *          - PCA{false}   — orchestrator use: assumes X is already column-centered;
 *                           no centering or restoration is performed.
 *
 * Typical standalone workflow:
 * @code
 *   PCA pca;
 *   PCAResult result = pca.fit(X, M);   // X is centered in-place
 *   Eigen::MatrixXd Z_new = pca.transform(X_new);
 *   // pca goes out of scope → X is restored automatically
 * @endcode
 *
 * Typical orchestrator workflow (centering handled externally):
 * @code
 *   PCA pca(false);
 *   PCAResult result = pca.fit(X_centered, M);
 * @endcode
 */
class PCA {

public:

    /**
     * @brief Construct a PCA solver.
     *
     * @param center If true (default), column-means are computed and subtracted
     *               in-place inside fit(), and the original values are restored
     *               on destruction (or via restore()). If false, X is assumed
     *               pre-centered and no modification or restoration is performed.
     */
    explicit PCA(bool center = true)
        : center_(center), data_ptr_(nullptr), n_(0), p_(0) {}


    /** @brief Destructor — restores X if centering was applied during fit(). */
    ~PCA() { restore(); }


    // PCA owns a restoration obligation: non-copyable, move-only.
    PCA(const PCA&)            = delete;
    PCA& operator=(const PCA&) = delete;
    PCA(PCA&&)                 = default;
    PCA& operator=(PCA&&)      = default;


    /**
     * @brief Fit PCA to the data matrix X, reducing to M principal components.
     *
     * @details When center=true, X is modified in-place (column means subtracted).
     *          Call restore() or let the PCA object go out of scope to undo this.
     *
     * @param X Input data matrix (n x p). Modified in-place when center=true.
     * @param M Number of principal components to retain.
     *
     * @return PCAResult containing the principal component scores (Z),
     *         loadings (V), and explained variance.
     */
    PCAResult fit(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Index M) {

        Eigen::Index n = X.rows();
        Eigen::Index p = X.cols();

        if (M > std::min(n, p)) {
            throw std::invalid_argument("M must not exceed min(n, p)");
        }
        if (n <= 1) {
            throw std::invalid_argument("PCA requires at least 2 samples to compute variance.");
        }

        if (center_) {
            mean_ = X.colwise().mean();
            X.rowwise() -= mean_;
            // Store for restoration
            data_ptr_ = X.data();
            n_ = n;
            p_ = p;
        } else {
            mean_ = Eigen::RowVectorXd::Zero(p);
        }

        svd::SVDResult svd_res = svd::SVDSolver{}.compute(X, M);

        loadings_           = svd_res.V;
        explained_variance_ = svd_res.S.array().square() / static_cast<double>(n - 1);

        PCAResult result;
        result.V                  = loadings_;
        result.Z                  = svd_res.U * svd_res.S.asDiagonal();
        result.explained_variance = explained_variance_;
        return result;
    }


    /**
     * @brief Restore X to its original (uncentered) state by adding back the
     *        column means computed during fit().
     *
     * @details This is called automatically by the destructor. Calling it
     *          explicitly allows early restoration before the PCA object goes
     *          out of scope. Safe to call multiple times (idempotent).
     *          Has no effect when center=false.
     */
    void restore() {
        if (data_ptr_) {
            Eigen::Map<Eigen::MatrixXd>(data_ptr_, n_, p_).rowwise() += mean_;
            data_ptr_ = nullptr;
        }
    }


    /**
     * @brief Project new observations onto the fitted principal components.
     *
     * @details Subtracts the mean stored during fit() (zero when center=false),
     *          then multiplies by the loadings matrix.
     *
     * @param X New data matrix (n_new x p). Must have the same number of columns
     *          as the matrix passed to fit().
     *
     * @return Score matrix of shape (n_new x M).
     */
    Eigen::MatrixXd transform(const Eigen::Ref<const Eigen::MatrixXd>& X) const {
        if (loadings_.cols() == 0) {
            throw std::runtime_error("PCA::transform called before fit().");
        }
        return (X.rowwise() - mean_) * loadings_;
    }


    // ===================================================================================
    // Getters
    // ===================================================================================

    /** @brief Column means computed during fit() (zero-vector when center=false). */
    const Eigen::RowVectorXd& getMean() const { return mean_; }

    /** @brief Principal component loadings (p x M right singular vectors). */
    const Eigen::MatrixXd& getLoadings() const { return loadings_; }

    /** @brief Explained variance per component (eigenvalues of sample covariance). */
    const Eigen::VectorXd& getExplainedVariance() const { return explained_variance_; }


private:

    /** @brief Whether to center the data during fit(). */
    bool         center_;

    /** @brief Raw pointer into the data buffer of X passed to fit();
               null when center=false or after restore() has been called.
    */
    double*      data_ptr_;

    /** @brief Number of samples (rows) in the data matrix. */
    Eigen::Index n_;

    /** @brief Number of features (columns) in the data matrix. */
    Eigen::Index p_;

    /** @brief Column means computed during fit() (zero-vector when center=false). */
    Eigen::RowVectorXd mean_;

    /** @brief Principal component loadings (p x M right singular vectors). */
    Eigen::MatrixXd    loadings_;

    /** @brief Explained variance per component (eigenvalues of sample covariance). */
    Eigen::VectorXd    explained_variance_;
};

// ===================================================================================
} /* namespace pca */
} /* namespace ml_methods */
} /* namespace trex */
// ===================================================================================
#endif /* End of TREX_ML_METHODS_PCA_HPP */
