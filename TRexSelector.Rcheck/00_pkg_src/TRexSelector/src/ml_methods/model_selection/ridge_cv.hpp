// ===================================================================================
// ridge_cv.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_RIDGE_CV_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_RIDGE_CV_HPP
// ===================================================================================
/**
 * @file ridge_cv.hpp
 *
 * @brief K-fold cross-validated ridge regression with `lambda.min` and `lambda.1se`
 *        selection rules.
 *
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

// Project includes
#include <ml_methods/model_selection/ridge_gcv.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace model_selection {

// ===================================================================================

/**
 * @brief K-fold cross-validated ridge *regression* with `lambda.min` and
 *        `lambda.1se` selection rules.
 *
 * @details
 *   Re-uses `ridge_gcv` per fold to avoid duplicating the centred-SVD
 *   machinery. For each lambda on a shared geometric grid, K fold-fits
 *   are computed; per-fold prediction MSE is averaged and its standard
 *   error of the mean is recorded. The grid is built once from a global
 *   `ridge_gcv` fit on the full (X, y) so that calls to `ridge_gcv` and
 *   `ridge_cv` search identical lambda scales.
 *
 *   Two selection rules are exposed:
 *     - `cv_min()`  : lambda minimising mean CV-MSE.
 *     - `cv_1se()`  : largest lambda whose CV-MSE is within one
 *                     standard error of the CV-MSE at `cv_min()` (the
 *                     glmnet-style `lambda.1se` rule).
 *   This follows the conventions of `glmnet::cv.glmnet` in R, which is a
 *   widely used reference.
 *
 *   Folds are deterministic: a `std::mt19937` permutation of the index
 *   vector seeded by the user-supplied `seed`, then partitioned into
 *   roughly equal contiguous blocks.
 */
class ridge_cv {

public:

    /** @brief Default constructor */
    ridge_cv() = default;

    /**
     * @brief Fit K-fold cross-validated ridge over a default lambda grid.
     *
     * @param X            Predictor matrix (n x p).
     * @param y            Response vector (n).
     * @param n_folds      Number of CV folds (default 10).
     * @param n_lambda     Number of lambdas on the geometric grid (default 100).
     * @param lambda_ratio Grid range factor passed to `default_lambda_grid`
     *                     (default 1000.0).
     * @param seed         RNG seed for the fold-index permutation
     *                     (default 0; identical seed -> identical folds).
     *
     * @throws std::invalid_argument if `n_folds < 2` or `n_folds > n`.
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             int n_folds           = 10,
             Eigen::Index n_lambda = 100,
             double lambda_ratio   = 1000.0,
             unsigned int seed     = 0) {

        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "ridge_cv::fit: X.rows() != y.size()");
        }
        const Eigen::Index n = X.rows();
        if (n_folds < 2) {
            throw std::invalid_argument(
                "ridge_cv::fit: n_folds must be >= 2");
        }
        if (static_cast<Eigen::Index>(n_folds) > n) {
            throw std::invalid_argument(
                "ridge_cv::fit: n_folds must be <= n_samples");
        }


        // ---- Build shared lambda grid from a global ridge_gcv fit ----------
        ridge_gcv full_fit;
        full_fit.fit(X, y);
        lambdas_ = full_fit.default_lambda_grid(n_lambda, lambda_ratio);

        const Eigen::Index K = lambdas_.size();
        cv_mse_.setZero(K);
        cv_sem_.setZero(K);


        // ---- Build deterministic folds --------------------------------------
        std::vector<Eigen::Index> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), Eigen::Index{0});
        std::mt19937 rng(seed);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<std::vector<Eigen::Index>> fold_idx(n_folds);
        for (Eigen::Index i = 0; i < n; ++i) {
            const int f = static_cast<int>(i % n_folds);
            fold_idx[static_cast<std::size_t>(f)].push_back(perm[static_cast<std::size_t>(i)]);
        }


        // ---- Per-fold training, accumulate per-lambda squared errors --------
        // fold_mse(k, f) = MSE of fold f at lambda k
        Eigen::MatrixXd fold_mse(K, n_folds);
        fold_mse.setZero();

        for (int f = 0; f < n_folds; ++f) {
            const auto& test_idx = fold_idx[static_cast<std::size_t>(f)];
            const Eigen::Index n_test  = static_cast<Eigen::Index>(test_idx.size());
            const Eigen::Index n_train = n - n_test;

            // Build masks
            std::vector<bool> in_test(static_cast<std::size_t>(n), false);
            for (Eigen::Index t : test_idx) {
                in_test[static_cast<std::size_t>(t)] = true;
            }

            Eigen::MatrixXd X_train(n_train, X.cols());
            Eigen::VectorXd y_train(n_train);
            Eigen::MatrixXd X_test(n_test, X.cols());
            Eigen::VectorXd y_test(n_test);

            Eigen::Index it_tr = 0, it_te = 0;
            for (Eigen::Index i = 0; i < n; ++i) {
                if (in_test[static_cast<std::size_t>(i)]) {
                    X_test.row(it_te) = X.row(i);
                    y_test(it_te) = y(i);
                    ++it_te;
                } else {
                    X_train.row(it_tr) = X.row(i);
                    y_train(it_tr) = y(i);
                    ++it_tr;
                }
            }

            ridge_gcv fold_fit;
            fold_fit.fit(X_train, y_train);

            // predict_path: one dgemm (n_test x p) * (p x K) instead of K dgemv calls
            const Eigen::MatrixXd preds = fold_fit.predict_path(X_test, lambdas_);
            fold_mse.col(f) =
                (preds.colwise() - y_test).colwise().squaredNorm().transpose() /
                static_cast<double>(n_test);
        }


        // ---- Aggregate mean + SEM across folds ------------------------------
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
            cv_sem_(k) = std::sqrt(var * inv_F); // SE of the mean
        }


        // ---- Selection: lambda.min and lambda.1se ---------------------------
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
        // Our grid is geometric in ascending order (low -> high lambda),
        // matching `default_lambda_grid`.
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

    /** @brief Largest lambda whose mean CV-MSE is within one SE of the
     *         CV-min (glmnet's `lambda.1se`).
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

    /** @brief Check if the model has been fitted. */
    void check_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "ridge_cv: call fit() before using this method");
        }
    }

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
} /* namespace model_selection */
} /* namespace ml_methods */
} /* namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_RIDGE_CV_HPP */
