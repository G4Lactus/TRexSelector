// ===================================================================================
// ridge_cv_svd.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_RIDGE_CV_SVD_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_RIDGE_CV_SVD_HPP
// ===================================================================================
/**
 * @file ridge_cv_svd.hpp
 *
 * @brief K-fold cross-validated ridge regression with `lambda.min` and `lambda.1se`
 *        selection rules.  Uses `Eigen::JacobiSVD` directly per fold with inline
 *        preprocessing; no dependency on an external GCV class.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SVD>

// ===================================================================================

// Namespace embedding: trex::ml_methods::model_selection
namespace trex {
namespace ml_methods {
namespace model_selection {

// ===================================================================================

/**
 * @brief K-fold cross-validated ridge regression with `lambda.min` and `lambda.1se`
 *        selection rules, implemented via `Eigen::JacobiSVD` per fold.
 *
 * @details
 *   Uses `Eigen::JacobiSVD` directly on each fold's normalised training matrix
 *   with inline centering and column L2-normalisation.  Fold splitting, MSE
 *   aggregation, and 1-SE selection rule follow standard glmnet conventions.
 *
 *   **Why JacobiSVD?**  `BDCSVD` can silently produce wrong results under aggressive
 *   compiler optimisation (`-O3 -march=native`) with some Eigen versions.
 *   `JacobiSVD` is slower for large matrices but safe at all optimisation levels.
 *
 *   **Lambda grid**: built inline from the full-data centred + L2-normalised
 *   `max_j |(X_{s,c}^T y_c)_j|` anchor (`c_max`).  The secondary SVD-based floor
 *   is omitted — it is only active when `y` is (nearly) orthogonal to `X`, in
 *   which case this class throws.
 *
 *   **Public interface**:
 *   `fit()`, `cv_min()`, `cv_1se()`, `lambdas()`, `cv_mse()`, `cv_sem()`,
 *   `index_min()`, `index_1se()`.
 */
class ridge_cv_svd {

public:

    /** @brief Default constructor */
    ridge_cv_svd() = default;

    /**
     * @brief Fit K-fold cross-validated ridge over a default lambda grid.
     *
     * @param X            Predictor matrix (n x p).
     * @param y            Response vector (n).
     * @param n_folds      Number of CV folds (default 10).
     * @param n_lambda     Number of lambdas on the geometric grid (default 1000).
     * @param lambda_ratio Grid range factor: grid spans
     *                     `[lambda_max / lambda_ratio, lambda_max]` (default 1000.0).
     * @param seed         RNG seed for the fold-index permutation
     *                     (default 0; identical seed → identical folds).
     *
     * @throws std::invalid_argument if `n_folds < 2`, `n_folds > n`,
     *                               `n_lambda <= 0`, or `lambda_ratio <= 0`.
     * @throws std::runtime_error    if `y` is (nearly) orthogonal to `X`, or if
     *                               a training fold has numerically zero rank.
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             int n_folds           = 10,
             Eigen::Index n_lambda = 1000,
             double lambda_ratio   = 1000.0,
             unsigned int seed     = 0) {

        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "ridge_cv_svd::fit: X.rows() != y.size()");
        }
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        if (n_folds < 2) {
            throw std::invalid_argument(
                "ridge_cv_svd::fit: n_folds must be >= 2");
        }
        if (static_cast<Eigen::Index>(n_folds) > n) {
            throw std::invalid_argument(
                "ridge_cv_svd::fit: n_folds must be <= n_samples");
        }
        if (n_lambda <= 0) {
            throw std::invalid_argument(
                "ridge_cv_svd::fit: n_lambda must be positive");
        }
        if (lambda_ratio <= 0.0) {
            throw std::invalid_argument(
                "ridge_cv_svd::fit: lambda_ratio must be positive");
        }


        // ---- Build lambda grid from full-data preprocessing (no full SVD) ------
        //
        // Inline replication of ridge_gcv's centering + L2-normalisation + grid
        // anchor (primary branch only; the SVD-eigenvalue floor is omitted since
        // it requires a full-data SVD and is only active when y ⊥ X).
        //
        //   x_scale_j = ||X_c[:,j]||_2 = sqrt( ||X[:,j]||^2 - n * x_mean_j^2 )
        //   c_max     = max_j |(X_{s,c}^T y_c)_j|   (normalised correlations)
        //   lambda_max = c_max                        (natural anchor for L2-normalised data)

        const Eigen::VectorXd x_mean_full = X.colwise().mean();
        const double          y_mean_full = y.mean();

        Eigen::VectorXd x_scale_full =
            (X.colwise().squaredNorm().transpose()
             - static_cast<double>(n) * x_mean_full.cwiseAbs2()
            ).cwiseMax(0.0).cwiseSqrt();

        // Guard constant columns (zero variance → scale = 1 to avoid division by zero).
        // If ALL columns are constant, maxCoeff() == 0 and a relative tolerance would
        // be 0 too — set every scale to 1; the c_max check below then throws cleanly
        // instead of propagating 0/0 NaNs.
        {
            const double max_scale = x_scale_full.maxCoeff();
            if (max_scale <= 0.0) {
                x_scale_full.setOnes();
            } else {
                const double scale_tol = 1e-10 * max_scale;
                for (Eigen::Index j = 0; j < p; ++j) {
                    if (x_scale_full(j) < scale_tol) x_scale_full(j) = 1.0;
                }
            }
        }

        // X_{s,c}^T y_c  =  D^{-1} X_c^T y_c  =  (X^T y - n*y_mean*x_mean) / x_scale
        Eigen::VectorXd Xsty_c = X.transpose() * y;
        Xsty_c -= static_cast<double>(n) * y_mean_full * x_mean_full;
        Xsty_c = Xsty_c.cwiseQuotient(x_scale_full);
        const double c_max = Xsty_c.cwiseAbs().maxCoeff();

        if (c_max <= 1e-12) {
            throw std::runtime_error(
                "ridge_cv_svd::fit: y is (nearly) orthogonal to X; "
                "lambda grid anchor c_max ≈ 0.");
        }

        const double lambda_max = c_max;
        const double log_hi     = std::log10(lambda_max);
        const double log_lo     = std::log10(lambda_max / lambda_ratio);

        lambdas_.resize(n_lambda);
        if (n_lambda == 1) {
            lambdas_(0) = std::pow(10.0, 0.5 * (log_lo + log_hi));
        } else {
            for (Eigen::Index i = 0; i < n_lambda; ++i) {
                lambdas_(i) = std::pow(
                    10.0,
                    log_lo + (log_hi - log_lo) *
                    static_cast<double>(i) / static_cast<double>(n_lambda - 1));
            }
        }

        const Eigen::Index K = lambdas_.size();
        cv_mse_.setZero(K);
        cv_sem_.setZero(K);


        // ---- Build deterministic folds (identical to ridge_cv) ----------------
        std::vector<Eigen::Index> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), Eigen::Index{0});
        std::mt19937 rng(seed);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<std::vector<Eigen::Index>> fold_idx(n_folds);
        for (Eigen::Index i = 0; i < n; ++i) {
            const int f = static_cast<int>(i % n_folds);
            fold_idx[static_cast<std::size_t>(f)].push_back(
                perm[static_cast<std::size_t>(i)]);
        }


        // ---- Per-fold: preprocess → SVD → predict_path → MSE -----------------
        // fold_mse(k, f) = MSE of fold f at lambda k
        Eigen::MatrixXd fold_mse(K, n_folds);
        fold_mse.setZero();

        for (int f = 0; f < n_folds; ++f) {

            const auto& test_idx   = fold_idx[static_cast<std::size_t>(f)];
            const Eigen::Index n_test  = static_cast<Eigen::Index>(test_idx.size());
            const Eigen::Index n_train = n - n_test;

            // Build logical test mask
            std::vector<bool> in_test(static_cast<std::size_t>(n), false);
            for (Eigen::Index t : test_idx) {
                in_test[static_cast<std::size_t>(t)] = true;
            }

            // Split X, y into training and test halves
            Eigen::MatrixXd X_train(n_train, p);
            Eigen::VectorXd y_train(n_train);
            Eigen::MatrixXd X_test(n_test, p);
            Eigen::VectorXd y_test(n_test);

            {
                Eigen::Index it_tr = 0, it_te = 0;
                for (Eigen::Index i = 0; i < n; ++i) {
                    if (in_test[static_cast<std::size_t>(i)]) {
                        X_test.row(it_te) = X.row(i);
                        y_test(it_te)     = y(i);
                        ++it_te;
                    } else {
                        X_train.row(it_tr) = X.row(i);
                        y_train(it_tr)     = y(i);
                        ++it_tr;
                    }
                }
            }

            // Per-fold centering + L2 normalisation (training fold only) ---------
            //
            // x_scale_tr_j = ||X_c_tr[:,j]||_2
            // X_s_tr       = (X_train - x_mean_tr) / x_scale_tr  (n_train x p)
            // y_c_tr       = y_train - y_mean_tr

            const Eigen::VectorXd x_mean_tr = X_train.colwise().mean();
            const double          y_mean_tr = y_train.mean();

            Eigen::VectorXd x_scale_tr =
                (X_train.colwise().squaredNorm().transpose()
                 - static_cast<double>(n_train) * x_mean_tr.cwiseAbs2()
                ).cwiseMax(0.0).cwiseSqrt();

            {
                // Same all-constant guard as the full-data block: with scales of 1
                // the fold matrix becomes all-zero and the rank check below throws.
                const double max_scale = x_scale_tr.maxCoeff();
                if (max_scale <= 0.0) {
                    x_scale_tr.setOnes();
                } else {
                    const double scale_tol = 1e-10 * max_scale;
                    for (Eigen::Index j = 0; j < p; ++j) {
                        if (x_scale_tr(j) < scale_tol) x_scale_tr(j) = 1.0;
                    }
                }
            }

            const Eigen::MatrixXd X_s_tr =
                (X_train.rowwise() - x_mean_tr.transpose()).array()
                .rowwise() / x_scale_tr.transpose().array();

            const Eigen::VectorXd y_c_tr = y_train.array() - y_mean_tr;

            // Thin JacobiSVD of X_s_tr.
            // JacobiSVD is safe under -O3/-march=native (unlike BDCSVD).
            // Singular values are returned in descending order.
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                X_s_tr, Eigen::ComputeThinU | Eigen::ComputeThinV);

            const Eigen::VectorXd& sv = svd.singularValues();

            // Rank truncation: same tolerance formula as ridge_gcv
            const double s_max_fold = (sv.size() > 0) ? sv(0) : 0.0;
            const double rank_tol   = std::numeric_limits<double>::epsilon()
                                    * s_max_fold
                                    * std::sqrt(static_cast<double>(n_train));
            Eigen::Index r = 0;
            for (Eigen::Index i = 0; i < sv.size(); ++i) {
                if (sv(i) > rank_tol) ++r;
                else break;
            }
            if (r == 0) {
                throw std::runtime_error(
                    "ridge_cv_svd::fit: SVD rank is zero in training fold " +
                    std::to_string(f));
            }

            // Truncated SVD components
            const auto U_r = svd.matrixU().leftCols(r);
            const auto V_r = svd.matrixV().leftCols(r);
            const auto s_r = sv.head(r);

            // U_r^T y_c_tr  (r-vector)
            const Eigen::VectorXd Uty = U_r.transpose() * y_c_tr;

            // Filter matrix D (r x K): D(i,k) = s_i / (s_i^2 + lambda_k)
            Eigen::MatrixXd D(r, K);
            for (Eigen::Index k = 0; k < K; ++k) {
                const double lam = lambdas_(k);
                D.col(k) = s_r.array() / (s_r.array().square() + lam);
            }

            // Coefficient matrix in normalised space B_s (p x K):
            //   B_s[:,k] = V_r * diag(s_i/(s_i^2+lambda_k)) * Uty
            //            = V_r * diag(Uty) * D[:,k]
            // → one DGEMM: (p x r) * (r x K) = (p x K)
            const Eigen::MatrixXd B_s = V_r * (Uty.asDiagonal() * D);

            // Back-scale to original space: Beta(j,k) = B_s(j,k) / x_scale_tr(j)
            const Eigen::MatrixXd Beta =
                B_s.array().colwise() / x_scale_tr.array();

            // Intercepts b0_k = y_mean_tr - x_mean_tr . Beta[:,k]  (1 x K row vector)
            const Eigen::RowVectorXd b0 =
                Eigen::RowVectorXd::Constant(K, y_mean_tr) -
                (x_mean_tr.transpose() * Beta);

            // Predictions (n_test x K): one DGEMM + intercept broadcast
            // Note: X_test is in the original (unprocessed) space — correct, because
            // Beta and b0 are already in the original-space parameterisation.
            const Eigen::MatrixXd preds = (X_test * Beta).rowwise() + b0;

            // Per-lambda MSE for this fold
            fold_mse.col(f) =
                (preds.colwise() - y_test).colwise().squaredNorm().transpose() /
                static_cast<double>(n_test);
        }


        // ---- Aggregate mean + SEM across folds (identical to ridge_cv) --------
        const double inv_F      = 1.0 / static_cast<double>(n_folds);
        const double inv_Fminus = 1.0 / static_cast<double>(n_folds - 1);
        for (Eigen::Index k = 0; k < K; ++k) {
            const double mean = fold_mse.row(k).mean();
            cv_mse_(k) = mean;
            double var = 0.0;
            for (int f = 0; f < n_folds; ++f) {
                const double d = fold_mse(k, f) - mean;
                var += d * d;
            }
            var *= inv_Fminus;
            cv_sem_(k) = std::sqrt(var * inv_F);
        }


        // ---- Selection: lambda.min and lambda.1se (identical to ridge_cv) ------
        Eigen::Index k_min = 0;
        double best = std::numeric_limits<double>::max();
        for (Eigen::Index k = 0; k < K; ++k) {
            if (cv_mse_(k) < best) {
                best = cv_mse_(k);
                k_min = k;
            }
        }
        idx_min_ = k_min;
        const double threshold = cv_mse_(k_min) + cv_sem_(k_min);

        // glmnet semantics: largest lambda whose mean MSE <= threshold.
        // Grid is ascending (low → high lambda), matching default_lambda_grid.
        Eigen::Index k_1se = k_min;
        for (Eigen::Index k = K - 1; k >= 0; --k) {
            if (cv_mse_(k) <= threshold) {
                k_1se = k;
                break;
            }
            if (k == 0) break;
        }
        idx_1se_ = k_1se;

        fitted_ = true;
    }

    // public member accessors
    // ------------------------------------------------------------

    /** @brief Lambda grid that was searched. */
    const Eigen::VectorXd& lambdas() const noexcept { return lambdas_; }

    /** @brief Per-lambda mean CV-MSE across folds. */
    const Eigen::VectorXd& cv_mse() const noexcept { return cv_mse_; }

    /** @brief Per-lambda standard error of the CV-MSE mean across folds. */
    const Eigen::VectorXd& cv_sem() const noexcept { return cv_sem_; }

    /** @brief Lambda minimising the mean CV-MSE (glmnet's `lambda.min`). */
    double cv_min() const {
        check_fitted();
        return lambdas_(idx_min_);
    }

    /** @brief Largest lambda whose mean CV-MSE is within one SE of the CV-min
     *         (glmnet's `lambda.1se`).
     */
    double cv_1se() const {
        check_fitted();
        return lambdas_(idx_1se_);
    }

    /** @brief Index of the lambda minimising mean CV-MSE. */
    Eigen::Index index_min() const noexcept { return idx_min_; }

    /** @brief Index of the largest lambda within one SE of the minimum. */
    Eigen::Index index_1se() const noexcept { return idx_1se_; }


private:

    // private member functions
    // ------------------------------------------------------------

    /** @brief Check if the model has been fitted. */
    void check_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "ridge_cv_svd: call fit() before using this method");
        }
    }

    // private member variables
    // ------------------------------------------------------------

    /** @brief Flag indicating whether the model has been fitted. */
    bool fitted_ = false;

    /** @brief Lambda grid that was searched. */
    Eigen::VectorXd lambdas_;

    /** @brief Per-lambda mean CV-MSE across folds. */
    Eigen::VectorXd cv_mse_;

    /** @brief Per-lambda standard error of the CV-MSE mean across folds. */
    Eigen::VectorXd cv_sem_;

    /** @brief Index of the lambda minimising mean CV-MSE. */
    Eigen::Index idx_min_ = 0;

    /** @brief Index of the largest lambda within one SE of the minimum. */
    Eigen::Index idx_1se_ = 0;
};

// ===================================================================================
} /* End of namespace model_selection */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_RIDGE_CV_SVD_HPP */
