// ===================================================================================
// pca.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_PCA_HPP
#define TREX_ML_METHODS_PCA_HPP
// ===================================================================================
/**
 * @file pca.hpp
 *
 * @brief Principal Component Analysis (PCA) implementation using SVD solver.
 *        Supports independent in-place centering and L2-norm column normalisation,
 *        with explicit caller-controlled restoration via restore().
 */
// ===================================================================================

// std includes
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/svd/svd.hpp>


// ===================================================================================

// Namespace declarations and embedding: trex::ml_methods::pca
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
 * @brief Stateful Principal Component Analysis with optional in-place centering
 *        and L2-norm column normalisation.
 *
 * @details Fits a truncated PCA model via the dimensionality-aware SVDSolver.
 *          Preprocessing (centering, normalisation) is applied in-place to X
 *          immediately in the constructor via preprocess_(). fit() then runs a
 *          truncated SVD on the already-preprocessed data. The caller is
 *          responsible for restoring X via restore(X) when done.
 *
 *          - center=true    — column means are subtracted from X in-place.
 *          - normalize=true — columns are divided by their L2 norm after centering,
 *                             yielding unit-norm columns. Near-zero norms (< 1e-10)
 *                             are clamped to 1 to avoid division by zero.
 *          - Both false     — X is assumed to be pre-processed by the caller.
 *
 *          State flags is_centered_ / is_normalized_ track what has actually been
 *          applied to X, making restore() safe to call multiple times (idempotent).
 *
 * Typical standalone workflow:
 * @code
 *   Eigen::MatrixXd X = ...;
 *   PCA pca(X, M);               // X is centered + normalised in-place here
 *   PCAResult result = pca.fit(); // runs truncated SVD
 *   Eigen::MatrixXd Z_new = pca.transform(X_new);
 *   pca.restore(X);              // undo in-place modifications
 * @endcode
 *
 * Typical orchestrator workflow (preprocessing handled externally):
 * @code
 *   PCA pca(X_preprocessed, M, false, false);
 *   PCAResult result = pca.fit();
 * @endcode
 */
class PCA {

public:

    /**
     * @brief Construct a PCA solver and preprocess X in-place immediately.
     *
     * @details Validates dimensions, then calls preprocess_() to apply centering
     *          and/or L2-norm normalisation to X in-place. Stores means_ and norms_
     *          (always valid after construction). Also stores a raw pointer to X's
     *          data buffer for use by fit(). X must remain valid until fit() returns.
     *
     * @param X         Input data matrix (n x p). Preprocessed in-place at
     *                  construction when center or normalize are true.
     *                  Must outlive the call to fit().
     * @param M         Number of principal components to retain. Must satisfy
     *                  M <= min(n, p).
     * @param center    If true (default), column means are subtracted in-place.
     *                  Stored in means_; reversed by restore().
     * @param normalize If true (default), columns are divided by their L2 norm after
     *                  centering. Stored in norms_; reversed by restore().
     *
     * @throws std::invalid_argument if M > min(n, p) or n <= 1.
     */
    explicit PCA(Eigen::Ref<Eigen::MatrixXd> X,
                 Eigen::Index M,
                 bool center    = true,
                 bool normalize = true)
        : data_ptr_(X.data()),
          outer_stride_(X.outerStride()),
          n_(X.rows()),
          p_(X.cols()),
          M_(M),
          center_(center),
          normalize_(normalize),
          is_centered_(false),
          is_normalized_(false),
          fitted_(false)
    {
        if (M_ > std::min(n_, p_)) {
            throw std::invalid_argument("PCA: M must not exceed min(n, p).");
        }
        if (n_ <= 1) {
            throw std::invalid_argument("PCA: requires at least 2 samples to compute variance.");
        }
        preprocess_(X);
    }

    /** @brief Destructor — no RAII restoration. Call restore(X) explicitly. */
    ~PCA() = default;

    // Non-copyable, move-only.
    /** @brief Copy constructor (deleted) */
    PCA(const PCA&)            = delete;

    /** @brief Copy assignment operator (deleted) */
    PCA& operator=(const PCA&) = delete;

    /** @brief Move constructor (defaulted) */
    PCA(PCA&&)                 = default;

    /** @brief Move assignment operator (defaulted) */
    PCA& operator=(PCA&&)      = default;


    /**
     * @brief Compute the truncated SVD on the already-preprocessed X.
     *
     * @details X was preprocessed in-place by the constructor. fit() runs the
     *          truncated SVD and stores loadings_ and explained_variance_.
     *          The raw pointer to X is released after fit() returns.
     *          Call restore(X) to undo the in-place preprocessing.
     *
     * @return PCAResult containing the principal component scores (Z),
     *         loadings (V), and explained variance per component.
     *
     * @throws std::runtime_error if called on an already-fitted instance.
     */
    PCAResult fit() {

        if (fitted_) {
            throw std::runtime_error(
                "PCA::fit() called on an already-fitted instance.");
        }

        // Rebuild the caller's view with its ORIGINAL outer stride: Eigen::Ref
        // accepts non-contiguous column blocks (e.g. X.topRows(k)), so a plain
        // stride-less Map would silently misread such inputs.
        Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>> X_map(
            data_ptr_, n_, p_, Eigen::OuterStride<>(outer_stride_));

        svd::SVDResult svd_res = svd::SVDSolver{}.compute(X_map, M_);

        loadings_           = svd_res.V;
        explained_variance_ = svd_res.S.array().square() / static_cast<double>(n_ - 1);
        fitted_             = true;
        data_ptr_           = nullptr;  // release stale pointer

        PCAResult result;
        result.V                  = loadings_;
        result.Z                  = svd_res.U * svd_res.S.asDiagonal();
        result.explained_variance = explained_variance_;
        return result;
    }


    /**
     * @brief Restore X to its original state by reversing the in-place preprocessing.
     *
     * @details Undoes normalisation then centering (reverse of application order):
     *            X.col(j) *= norms_(j)  [if is_normalized_]
     *            X.rowwise() += means_  [if is_centered_]
     *          Uses the is_normalized_ / is_centered_ state flags, so calling it
     *          multiple times is safe: each flag is cleared after its step is applied,
     *          making the second call a no-op.
     *
     * @param X Matrix to restore. Must have the same dimensions (n x p) as the
     *          matrix passed to the constructor.
     *
     * @throws std::runtime_error    if called before fit().
     * @throws std::invalid_argument if X dimensions do not match.
     */
    void restore(Eigen::Ref<Eigen::MatrixXd> X) {
        if (!fitted_) {
            throw std::runtime_error("PCA::restore() called before fit().");
        }
        if (X.rows() != n_ || X.cols() != p_) {
            throw std::invalid_argument(
                "PCA::restore(): X dimensions do not match the matrix used in fit().");
        }
        if (!is_centered_ && !is_normalized_) { return; }

        // Reverse order: undo normalisation first, then centering.
        // Each flag is cleared immediately after its step so the call is idempotent.
        if (is_normalized_) {
            X.array().rowwise() *= norms_.array();
            is_normalized_ = false;
        }
        if (is_centered_) {
            X.rowwise() += means_;
            is_centered_ = false;
        }
    }


    /**
     * @brief Project new observations onto the fitted principal components.
     *
     * @details Applies the same preprocessing as fit() (centering then normalisation)
     *          before projecting: Z_new = ((X_new - means_) ./ norms_) * V.
     *          norms_ is a ones vector when normalize=false, so this formula holds
     *          unconditionally.
     *
     * @param X New data matrix (n_new x p). Must have p columns.
     *
     * @return Score matrix of shape (n_new x M).
     *
     * @throws std::runtime_error if called before fit().
     */
    Eigen::MatrixXd transform(const Eigen::Ref<const Eigen::MatrixXd>& X) const {
        if (!fitted_) {
            throw std::runtime_error("PCA::transform() called before fit().");
        }
        Eigen::MatrixXd Xc = (X.rowwise() - means_).eval();
        Xc.array().rowwise() /= norms_.array();
        return Xc * loadings_;
    }


    // ===================================================================================
    // Getters
    // ===================================================================================

    /**
     * @brief Column means computed at construction (zero-vector when center=false).
     *        Available immediately after construction, before fit().
     */
    const Eigen::RowVectorXd& getMeans() const { return means_; }

    /**
     * @brief Column L2 norms computed at construction (ones-vector when normalize=false).
     *        Available immediately after construction, before fit().
     */
    const Eigen::RowVectorXd& getNorms() const { return norms_; }

    /**
     * @brief Principal component loadings (p x M right singular vectors).
     * @throws std::runtime_error if called before fit().
     */
    const Eigen::MatrixXd& getLoadings() const {
        if (!fitted_) throw std::runtime_error("PCA::getLoadings() called before fit().");
        return loadings_;
    }

    /**
     * @brief Explained variance per component (eigenvalues of sample covariance).
     * @throws std::runtime_error if called before fit().
     */
    const Eigen::VectorXd& getExplainedVariance() const {
        if (!fitted_) throw std::runtime_error("PCA::getExplainedVariance() called before fit().");
        return explained_variance_;
    }


private:

    /** @brief Raw pointer into X's data buffer; valid until fit() completes, then null. */
    double* data_ptr_;

    /** @brief Outer stride of X's buffer (== n_ for contiguous storage, larger for views). */
    Eigen::Index outer_stride_;

    /** @brief Number of rows in X. */
    Eigen::Index n_;

    /** @brief Number of columns in X. */
    Eigen::Index p_;

    /** @brief Number of principal components to compute. */
    Eigen::Index M_;

    /** @brief Intent flag: whether to subtract column means (set at construction). */
    bool center_;

    /** @brief Intent flag: whether to divide columns by their L2 norm (set at construction). */
    bool normalize_;

    /** @brief State flag: true while centering has been applied and not yet restored. */
    bool is_centered_;

    /** @brief State flag: true while normalisation has been applied and not yet restored. */
    bool is_normalized_;

    /** @brief True after fit() has been called successfully. */
    bool fitted_;

    /** @brief Column means computed at construction (zero-vector when center=false). */
    Eigen::RowVectorXd means_;

    /** @brief Column L2 norms computed at construction (ones-vector when normalize=false). */
    Eigen::RowVectorXd norms_;


    // ===================================================================================
    // Private helpers
    // ===================================================================================

    /**
     * @brief Apply centering and L2-norm normalisation to X in-place.
     *
     * @details Called from the constructor. Updates means_, norms_, is_centered_,
     *          and is_normalized_ according to the center_ and normalize_ intent flags.
     *
     * @param X Data matrix to preprocess (same object passed to the constructor).
     */
    void preprocess_(Eigen::Ref<Eigen::MatrixXd> X) {
        if (center_) {
            means_ = X.colwise().mean();
            X.rowwise() -= means_;
            is_centered_ = true;
        } else {
            means_ = Eigen::RowVectorXd::Zero(p_);
        }

        if (normalize_) {
            norms_.resize(p_);
            constexpr double eps = 1e-10;
            for (Eigen::Index j = 0; j < p_; ++j) {
                const double col_norm = X.col(j).norm();
                norms_(j) = (col_norm < eps) ? 1.0 : col_norm;
                X.col(j) /= norms_(j);
            }
            is_normalized_ = true;
        } else {
            norms_ = Eigen::RowVectorXd::Ones(p_);
        }
    }

    /** @brief Principal component loadings (p x M right singular vectors). */
    Eigen::MatrixXd loadings_;

    /** @brief Explained variance per component (eigenvalues of sample covariance). */
    Eigen::VectorXd explained_variance_;
};

// ===================================================================================
} /* End of namespace pca */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================
#endif /* End of TREX_ML_METHODS_PCA_HPP */
