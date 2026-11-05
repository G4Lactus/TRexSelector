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
#include <algorithm>
#include <stdexcept>
#include <utility>

// Eigen includes
#include <Eigen/Dense>

// utils includes
#include <utils/openmp/utils_eigen_threads.hpp>

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
 *
 *  References:
 *  -----------
 *   G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.
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
                 bool normalize = true,
                 int n_threads  = ::trex::utils::openmp::get_num_threads())
        : data_ptr_(X.data()),
          writable_ptr_(X.data()),
          outer_stride_(X.outerStride()),
          n_(X.rows()),
          p_(X.cols()),
          M_(M),
          center_(center),
          normalize_(normalize),
          is_centered_(false),
          is_normalized_(false),
          fitted_(false),
          n_threads_(n_threads)
    {
        if (n_threads < 1) {
            throw std::invalid_argument("PCA: n_threads must be >= 1.");
        }
        validate_();
        preprocess_(X);
    }

    /**
     * @brief Read-only PCA: no centering, no normalisation, X is never
     *        written.
     *
     * @details Accepts a const matrix or a read-only memory-mapped file;
     *          fit() reads it through a const map. restore() is a no-op and
     *          the destructor writes nothing back. Use it when the data is
     *          already centered / scaled (a scaler in front keeps the
     *          statistics), or for the raw SVD-based decomposition. A named
     *          factory rather than a constructor overload, so that
     *          `PCA(X, M)` keeps its meaning (center and normalize in place).
     *
     * @param X Input data matrix (n x p), read only; must outlive fit().
     * @param M Number of principal components to retain, M <= min(n, p).
     */
    static PCA readOnly(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Index M) {
        return PCA(ReadOnlyTag{}, X, M);
    }

    /**
     * @brief Destructor: writes X back if the in-place preprocessing is still
     *        in effect (the object was never restore()d), then releases.
     *
     * @details The same rule as the T-solvers and the T-Rex selector: the
     *          caller's buffer never stays preprocessed
     *          without an owner. The buffer must still be alive; the bindings
     *          pin it (R: the XPtr protection slot; Python: keep_alive).
     *          A restore()d or read-only object writes nothing.
     */
    ~PCA() { write_back_(); }

    // Non-copyable, move-only. The write-back duty travels with a move: the
    // moved-from object must not restore X when it is destroyed.
    /** @brief Copy constructor (deleted) */
    PCA(const PCA&)            = delete;

    /** @brief Copy assignment operator (deleted) */
    PCA& operator=(const PCA&) = delete;

    /** @brief Move constructor: takes over the buffer and the write-back duty. */
    PCA(PCA&& other) noexcept
        : data_ptr_(other.data_ptr_),
          writable_ptr_(other.writable_ptr_),
          outer_stride_(other.outer_stride_),
          n_(other.n_), p_(other.p_), M_(other.M_),
          center_(other.center_), normalize_(other.normalize_),
          is_centered_(other.is_centered_), is_normalized_(other.is_normalized_),
          fitted_(other.fitted_),
          means_(std::move(other.means_)), norms_(std::move(other.norms_)),
          loadings_(std::move(other.loadings_)),
          explained_variance_(std::move(other.explained_variance_))
    {
        other.disarm_();
    }

    /** @brief Move assignment: writes this object's X back first, then takes over. */
    PCA& operator=(PCA&& other) noexcept {
        if (this != &other) {
            write_back_();
            data_ptr_ = other.data_ptr_; writable_ptr_ = other.writable_ptr_;
            outer_stride_ = other.outer_stride_;
            n_ = other.n_; p_ = other.p_; M_ = other.M_;
            center_ = other.center_; normalize_ = other.normalize_;
            is_centered_ = other.is_centered_; is_normalized_ = other.is_normalized_;
            fitted_ = other.fitted_;
            means_ = std::move(other.means_); norms_ = std::move(other.norms_);
            loadings_ = std::move(other.loadings_);
            explained_variance_ = std::move(other.explained_variance_);
            other.disarm_();
        }
        return *this;
    }


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
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);

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
     * @param X The matrix passed to the constructor (the same buffer; the
     *          preprocessing was applied there and is undone there). May be
     *          called before fit(): the preprocessing happened at
     *          construction, so a caller who does not fit still gets X back.
     *
     * @throws std::invalid_argument if X is not the constructor's matrix.
     */
    void restore(Eigen::Ref<Eigen::MatrixXd> X) {
        if (!is_centered_ && !is_normalized_) { return; }   // nothing to undo
        if (X.rows() != n_ || X.cols() != p_ || X.data() != writable_ptr_) {
            throw std::invalid_argument(
                "PCA::restore(): X is not the matrix this PCA was constructed on.");
        }
        write_back_();
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
        // Eigen's products run on this object's budget for the scope of the call
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
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

    /** @brief Read pointer into X's data buffer for fit(); null after fit(). */
    const double* data_ptr_;

    /** @brief Write pointer into X's buffer for the write-back (restore() and
     *  the destructor); null for the read-only constructor and after the
     *  write-back duty has been handed over by a move. Stays valid after
     *  fit(): the caller's buffer outlives the object (pinned by the
     *  bindings). */
    double* writable_ptr_;

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

    /** @brief Thread budget of the Eigen products of fit() and transform() (>= 1). */
    int n_threads_;

    /** @brief Column means computed at construction (zero-vector when center=false). */
    Eigen::RowVectorXd means_;

    /** @brief Column L2 norms computed at construction (ones-vector when normalize=false). */
    Eigen::RowVectorXd norms_;


    // ===================================================================================
    // Private helpers
    // ===================================================================================

    /** @brief Tag of the read-only constructor (see readOnly()). */
    struct ReadOnlyTag {};

    /** @brief Read-only construction: no preprocessing, no write pointer. */
    PCA(ReadOnlyTag, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Index M,
        int n_threads = ::trex::utils::openmp::get_num_threads())
        : data_ptr_(X.data()),
          writable_ptr_(nullptr),
          outer_stride_(X.outerStride()),
          n_(X.rows()),
          p_(X.cols()),
          M_(M),
          center_(false),
          normalize_(false),
          is_centered_(false),
          is_normalized_(false),
          fitted_(false),
          n_threads_(n_threads)
    {
        if (n_threads < 1) {
            throw std::invalid_argument("PCA: n_threads must be >= 1.");
        }
        validate_();
        means_ = Eigen::RowVectorXd::Zero(p_);
        norms_ = Eigen::RowVectorXd::Ones(p_);
    }

    /** @brief Shape checks shared by both constructors. */
    void validate_() const {
        if (M_ > std::min(n_, p_)) {
            throw std::invalid_argument("PCA: M must not exceed min(n, p).");
        }
        if (n_ <= 1) {
            throw std::invalid_argument("PCA: requires at least 2 samples to compute variance.");
        }
    }

    /**
     * @brief Undo the in-place preprocessing through the stored write pointer
     *        (reverse order: normalisation first, then centering); idempotent,
     *        a no-op for a read-only, restored or disarmed object. The core of
     *        restore() and of the destructor.
     */
    void write_back_() noexcept {
        if (writable_ptr_ == nullptr || (!is_centered_ && !is_normalized_)) { return; }
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> X(
            writable_ptr_, n_, p_, Eigen::OuterStride<>(outer_stride_));
        if (is_normalized_) {
            X.array().rowwise() *= norms_.array();
            is_normalized_ = false;
        }
        if (is_centered_) {
            X.rowwise() += means_;
            is_centered_ = false;
        }
    }

    /** @brief Drop the buffer and the write-back duty (moved-from state). */
    void disarm_() noexcept {
        data_ptr_ = nullptr;
        writable_ptr_ = nullptr;
        is_centered_ = false;
        is_normalized_ = false;
    }

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
