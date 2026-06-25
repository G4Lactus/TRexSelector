// ===================================================================================
// ridge_gcv.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_RIDGE_GCV_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_RIDGE_GCV_HPP
// ===================================================================================
/**
 * @file ridge_gcv.hpp
 *
 * @brief Ridge regression with Generalized Cross-Validation (GCV) for model
 *        selection. "SVD-based multi-λ solver. For single-λ use without amortization,
 *        see ridge_regression/ridge.hpp."
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>


// ===================================================================================

namespace trex {
namespace ml_methods {
namespace model_selection {

// ===================================================================================

/** @brief Summary structure for ridge regression path results */
struct ridge_path {
    /** @brief Vector of lambda values used in the ridge path */
    Eigen::VectorXd lambdas;

    /** @brief Matrix of coefficients corresponding to each lambda */
    Eigen::MatrixXd coefficients;

    /** @brief Vector of GCV scores corresponding to each lambda */
    Eigen::VectorXd gcv_scores;

    /** @brief Vector of effective degrees of freedom corresponding to each lambda */
    Eigen::VectorXd df_effective;

    /** @brief Index of the best lambda based on GCV score */
    Eigen::Index best_index = 0;
};


/**
 * @brief Ridge generalized cross-validation (GCV) class.
 *
 */
class ridge_gcv {

public:

    /** @brief Default constructor */
    ridge_gcv() = default;


    /** @brief Fit the ridge regression model using GCV.
     *
     * @param X Design matrix (n x p)
     * @param y Response vector (n)
     */
    void fit(
        Eigen::Ref<const Eigen::MatrixXd> X, // NOLINT
        Eigen::Ref<const Eigen::VectorXd> y  // NOLINT
    ) {

        if (X.rows() != y.size()) {
            throw std::invalid_argument("ridge_gcv::fit: X.rows() != y.size()");
        }

        n_ = X.rows();
        p_ = X.cols();

        x_mean_ = X.colwise().mean();
        y_mean_ = y.mean();

        // ||y_c||^2 = ||y||^2 - n * y_mean^2  (no copy of y needed)
        yc_sq_norm_ = y.squaredNorm() - static_cast<double>(n_) * y_mean_ * y_mean_;

        Eigen::VectorXd evals_desc;

        if (n_ >= p_) {

            // -------------------------------------------------------------
            // n >= p: Eigen-decompose the p x p covariance matrix X^T X
            // -------------------------------------------------------------
            // X_c^T X_c = X^T X - n * x_mean * x_mean^T  (rank-1 correction)
            Eigen::MatrixXd K = X.transpose() * X;
            K -= static_cast<double>(n_) * x_mean_ * x_mean_.transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(K);

            if (eig.info() != Eigen::Success) {
                throw std::runtime_error("ridge_gcv::fit: Eigen decomposition failed");
            }

            // Eigen returns ascending order; reverse to descending for SVD parity
            evals_desc = eig.eigenvalues().reverse();
            Eigen::MatrixXd V_desc = eig.eigenvectors().rowwise().reverse();

            double s_max = std::sqrt(std::max(0.0, evals_desc(0)));
            double tol = std::numeric_limits<double>::epsilon() *
                         s_max *
                         std::sqrt(static_cast<double>(n_));

            r_ = 0;
            for (Eigen::Index i = 0; i < p_; ++i) {
                if (std::sqrt(std::max(0.0, evals_desc(i))) > tol) ++r_;
                else break;
            }

            if (r_ == 0) throw std::runtime_error("ridge_gcv::fit: numerically zero");

            // sigma_i = sqrt(lambda_i)
            sigma_ = evals_desc.head(r_).cwiseMax(0.0).cwiseSqrt();
            V_ = V_desc.leftCols(r_);

            // Instead of forming U (which is huge), compute U^T y algebraically:
            // X^T y = V Sigma U^T y  ->  U^T y = Sigma^{-1} V^T X^T y
            // X_c^T y_c = X^T y - n * y_mean * x_mean  (no copy)
            Eigen::VectorXd Xty = X.transpose() * y;
            Xty -= static_cast<double>(n_) * y_mean_ * x_mean_;
            Eigen::VectorXd Z = V_.transpose() * Xty;
            Uty_ = Z.cwiseQuotient(sigma_);

        } else {

            // -------------------------------------------------------------
            // n < p: Eigen-decompose the n x n Gram matrix X X^T
            // -------------------------------------------------------------
            // X_c X_c^T = X X^T - mu*1^T - 1*mu^T + ||x_mean||^2 * 1*1^T
            // where mu = X * x_mean  (n-vector, no full copy of X)
            Eigen::VectorXd mu = X * x_mean_;
            Eigen::MatrixXd K = X * X.transpose();
            K.colwise() -= mu;
            K.rowwise() -= mu.transpose();
            K.array() += x_mean_.squaredNorm();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(K);

            evals_desc = eig.eigenvalues().reverse();
            Eigen::MatrixXd U_desc = eig.eigenvectors().rowwise().reverse();

            double s_max = std::sqrt(std::max(0.0, evals_desc(0)));
            double tol = std::numeric_limits<double>::epsilon()
                       * s_max
                       * std::sqrt(static_cast<double>(p_));

            r_ = 0;
            for (Eigen::Index i = 0; i < n_; ++i) {
                if (std::sqrt(std::max(0.0, evals_desc(i))) > tol) ++r_;
                else break;
            }

            if (r_ == 0) throw std::runtime_error("ridge_gcv::fit: numerically zero");

            sigma_ = evals_desc.head(r_).cwiseMax(0.0).cwiseSqrt();
            U_ = U_desc.leftCols(r_);

            // U^T y_c = U^T y - y_mean * U^T 1
            Uty_ = U_.transpose() * y;
            Uty_ -= y_mean_ * U_.colwise().sum().transpose();

            // Recover V_ algebraically: V = X_c^T U Sigma^{-1}
            // X_c^T U = X^T U - x_mean * (1^T U) = X^T U - x_mean * U.colwise().sum()
            Eigen::MatrixXd XtU = X.transpose() * U_;
            XtU -= x_mean_ * U_.colwise().sum();
            V_ = XtU * sigma_.cwiseInverse().asDiagonal();
        }


        Uty_sq_norm_ = Uty_.squaredNorm();

        // Guard against tiny negative values from cancellation,
        // Guard against potential NaNs before std::max.
        if (std::isnan(yc_sq_norm_) || std::isnan(Uty_sq_norm_)) {
             y_perp_sq_ = 0.0; // Fallback to avoid propagating NaN via max()
        } else {
             y_perp_sq_ = std::max(0.0, yc_sq_norm_ - Uty_sq_norm_);
        }

        fitted_ = true;
    }


    /**
     * @brief Solve the ridge regression problem for a given regularization parameter.
     *
     * @param lambda Regularization parameter.
     *
     * @return Coefficients of the ridge regression model.
     */
    Eigen::VectorXd solve(double lambda) const {
        check_fitted();
        check_lambda(lambda, "solve");

        Eigen::VectorXd d(r_);
        for (Eigen::Index i = 0; i < r_; ++i) {
            const double s = sigma_(i);
            d(i) = s / (s * s + lambda);
        }

        return V_ * (d.array() * Uty_.array()).matrix();
    }


    /**
     * @brief Compute the intercept of the ridge regression model for a given
     *        regularization parameter.
     *
     * @param lambda Regularization parameter.
     *
     * @return Intercept of the ridge regression model.
     */
    double intercept(double lambda) const {
        const Eigen::VectorXd beta = solve(lambda);
        return y_mean_ - x_mean_.dot(beta);
    }


    /**
     * @brief Compute the Generalized Cross-Validation (GCV) score for a given
     *        regularization parameter.
     *
     * @param lambda Regularization parameter.
     *
     * @return GCV score.
     */
    double gcv(double lambda) const {
        check_fitted();
        check_lambda(lambda, "gcv");

        double rss = y_perp_sq_;
        double df = 0.0;

        for (Eigen::Index i = 0; i < r_; ++i) {
            const double s2 = sigma_(i) * sigma_(i);
            const double denom = s2 + lambda;
            const double shrink = lambda / denom;
            const double filter = s2 / denom;

            rss += shrink * shrink * Uty_(i) * Uty_(i);
            df += filter;
        }

        const double denom = 1.0 - df / static_cast<double>(n_);
        // Return +inf whenever the hat-matrix trace is within 1% of n.
        // In the underdetermined regime (r_ close to n) the denominator collapses
        // toward zero, making GCV numerically unreliable.  Callers such as
        // gcv_optimal() rely on this to stay out of the degenerate region.
        if (std::abs(denom) < 1e-2) {
            return std::numeric_limits<double>::max();
        }

        return (rss / static_cast<double>(n_)) / (denom * denom);
    }


    /**
     * @brief Determine effective degrees of freedom for a given regularization parameter.
     *
     * @param lambda Regularization parameter.
     *
     * @return Effective degrees of freedom.
     */
    double effective_df(double lambda) const {
        check_fitted();
        check_lambda(lambda, "effective_df");

        double df = 0.0;
        for (Eigen::Index i = 0; i < r_; ++i) {
            const double s2 = sigma_(i) * sigma_(i);
            df += s2 / (s2 + lambda);
        }

        return df;
    }


    /**
     * @brief Predict the response for new data using the ridge regression model.
     *
     * @param X_new  New data matrix.
     * @param lambda Regularization parameter.
     *
     * @return Predicted response vector.
     */
    Eigen::VectorXd predict(
        Eigen::Ref<const Eigen::MatrixXd> X_new, // NOLINT
        double lambda
    ) const {
        check_fitted();
        check_lambda(lambda, "predict");

        if (X_new.cols() != p_) {
            throw std::invalid_argument(
                "ridge_gcv::predict: X_new.cols() != p");
        }

        const Eigen::VectorXd beta = solve(lambda);
        const double b0 = y_mean_ - x_mean_.dot(beta);
        return (X_new * beta).array() + b0;
    }


    /**
     * @brief Predict responses for new data across a path of regularization parameters.
     *
     * @details One dgemm replaces K independent dgemv calls.
     *          Intercepts are computed as a cheap 1xK row-vector subtraction.
     *
     * @param X_new   New data matrix (n_new x p).
     * @param lambdas Vector of regularization parameters (K).
     *
     * @return Prediction matrix (n_new x K); column k corresponds to lambdas(k).
     */
    Eigen::MatrixXd predict_path(
        Eigen::Ref<const Eigen::MatrixXd> X_new, // NOLINT
        Eigen::Ref<const Eigen::VectorXd> lambdas
    ) const {
        check_fitted();

        if (X_new.cols() != p_) {
            throw std::invalid_argument(
                "ridge_gcv::predict_path: X_new.cols() != p");
        }

        // validates all lambdas
        const ridge_path path = solve_path(std::move(lambdas));

        // Intercepts: b0_k = y_mean - x_mean^T beta_k  (1 x K)
        Eigen::RowVectorXd b0 =
            Eigen::RowVectorXd::Constant(path.coefficients.cols(), y_mean_) -
            (x_mean_.transpose() * path.coefficients);

        // One dgemm: (n_new x p) * (p x K) = (n_new x K), then broadcast intercepts
        return (X_new * path.coefficients).rowwise() + b0;
    }


    /**
     * @brief Solve the ridge regression problem for a path of regularization parameters.
     *
     * @param lambdas Vector of regularization parameters.
     *
     * @return Ridge path containing coefficients, GCV scores, and effective degrees of freedom.
     */
    ridge_path solve_path(
        Eigen::Ref<const Eigen::VectorXd> lambdas // NOLINT
    ) const {
        check_fitted();

        const Eigen::Index K = lambdas.size();

        ridge_path path;
        path.lambdas = Eigen::VectorXd(lambdas);
        path.coefficients.resize(p_, K);
        path.gcv_scores.resize(K);
        path.df_effective.resize(K);
        path.best_index = 0;

        double best_gcv = std::numeric_limits<double>::max();

        for (Eigen::Index k = 0; k < K; ++k) {
            check_lambda(lambdas(k), "solve_path");
            path.coefficients.col(k) = solve(lambdas(k));
            path.gcv_scores(k) = gcv(lambdas(k));
            path.df_effective(k) = effective_df(lambdas(k));

            if (path.gcv_scores(k) < best_gcv) {
                best_gcv = path.gcv_scores(k);
                path.best_index = k;
            }
        }

        return path;
    }


    /**
     * @brief Generate a default grid of lambda values for ridge regression path solving.
     *
     * @details The grid is anchored to the maximum absolute initial LARS correlation
     *          c_max = max_j |x_j^T y_c|, mirroring R's cv.glmnet grid for alpha = 0:
     *
     *            lambda_max = c_max / sqrt(n - 1)
     *            grid       = [ lambda_max / ratio, lambda_max ]  (geometrically spaced)
     *
     *          This makes the grid scale with the signal strength in y so that ridge_cv
     *          selects lambda values that translate correctly to TENET's lambda_2 across
     *          different DGP configurations (e.g., varying numbers of active variables).
     *          Falls back to an SVD-eigenvalue-based grid when c_max is numerically zero
     *          (y orthogonal to X).
     *
     * @param num_lambda Number of lambda values to generate.
     * @param ratio      Ratio of grid maximum to grid minimum (default 1000, matching
     *                   glmnet's lambda.min.ratio = 0.001 for p > n problems).
     *
     * @return Vector of lambda values in ascending order.
     */
    Eigen::VectorXd default_lambda_grid(Eigen::Index num_lambda = 100,
                                        double ratio = 1000.0) const {
        check_fitted();

        if (num_lambda <= 0) {
            throw std::invalid_argument(
                "ridge_gcv::default_lambda_grid: num_lambda must be positive");
        }

        if (ratio <= 0.0) {
            throw std::invalid_argument(
                "ridge_gcv::default_lambda_grid: ratio must be positive");
        }

        // Compute c_max = max_j |x_j^T y_c| from stored SVD components.
        // In both code paths (n >= p and n < p):
        //   V_ * diag(sigma_) * Uty_ = X_c^T y_c  (exact).
        const Eigen::VectorXd Xty_c = V_ * sigma_.cwiseProduct(Uty_);
        const double c_max = Xty_c.cwiseAbs().maxCoeff();

        double log_lo, log_hi;

        if (c_max > 1e-12) {
            // Hybrid anchor:
            // 1.  y-dependent grid anchored to c_max / sqrt(n - 1), mirroring glmnet's
            //     lambda_max = max_j|X_R_j^T y| / n = sqrt(n-1) * c_max / n.
            // 2. SVD-eigenvalue-based grid anchored to sigma_max / sqrt(n - 1).
            const double lambda_max_y = c_max / std::sqrt(static_cast<double>(n_ - 1));
            const double lambda_max_svd = sigma_(0) / std::sqrt(static_cast<double>(n_ - 1));
            const double lambda_max     = std::max(lambda_max_y, lambda_max_svd);
            log_hi = std::log10(lambda_max);
            log_lo = std::log10(lambda_max / ratio);
        } else {
            // Fallback: y is (nearly) orthogonal to X; use SVD-eigenvalue-based range.
            double s_max = sigma_(0);
            double s_min = sigma_(r_ - 1);
            if (s_min < 1e-15 * s_max) {
                s_min = s_max * 1e-10;
            }
            log_lo = std::log10(s_min * s_min / ratio);
            log_hi = std::log10(s_max * s_max * ratio);
        }

        Eigen::VectorXd grid(num_lambda);

        if (num_lambda == 1) {
            grid(0) = std::pow(10.0, 0.5 * (log_lo + log_hi));
            return grid;
        }

        for (Eigen::Index i = 0; i < num_lambda; ++i) {
            grid(i) = std::pow(
                10.0,
                log_lo + (log_hi - log_lo) *
                static_cast<double>(i) / static_cast<double>(num_lambda - 1));
        }

        return grid;
    }


    /**
     * @brief Determine the GCV-optimal lambda using a coarse grid search followed by
     *        golden-section refinement.
     *
     * @param num_grid Number of grid points for the initial coarse search (must be at least 3).
     * @param tol      Tolerance for the golden-section search convergence (must be positive).
     *
     * @return GCV-optimal lambda value.
     */
    double gcv_optimal(int num_grid = 200, double tol = 1e-8) const {
        check_fitted();

        if (num_grid < 3) {
            throw std::invalid_argument(
                "ridge_gcv::gcv_optimal: num_grid must be at least 3");
        }
        if (tol <= 0.0) {
            throw std::invalid_argument(
                "ridge_gcv::gcv_optimal: tol must be positive");
        }

        double s_max_sq = sigma_(0) * sigma_(0);
        double s_min_sq = sigma_(r_ - 1) * sigma_(r_ - 1);

        if (s_min_sq < 1e-20 * s_max_sq) {
            s_min_sq = s_max_sq * 1e-10;
        }

        double log_lo = std::log10(s_min_sq * 1e-4);
        const double log_hi = std::log10(s_max_sq * 1e4);

        // When the rank is close to n the GCV denominator (1 - df/n)^2 collapses toward
        // zero.
        // gcv() already returns +inf whenever |1 - df/n| < 0.01, so the grid and
        // golden-section searches will naturally avoid that region.
        // As a belt-and-suspenders measure, also pre-shift log_lo to the lambda that
        // reduces df_eff to 98% of n (denom ≈ 0.02, safely above the gcv() guard of 0.01).
        double lambda_floor = 0.0;  // 0 == inactive for well-determined systems
        if (r_ >= n_ - 1) {
            lambda_floor = compute_df_floor((1.0 - 1e-2) * static_cast<double>(n_));
            log_lo = std::max(log_lo, std::log10(lambda_floor));
        }

        double best_lambda =
            std::exp(0.5 * (std::log(s_min_sq) + std::log(s_max_sq)));
        double best_gcv = std::numeric_limits<double>::max();

        for (int i = 0; i < num_grid; ++i) {
            const double lambda = std::pow(
                10.0,
                log_lo + (log_hi - log_lo) *
                static_cast<double>(i) / static_cast<double>(num_grid - 1));
            const double score = gcv(lambda);

            if (score < best_gcv) {
                best_gcv = score;
                best_lambda = lambda;
            }
        }

        // Refine locally on log-scale around the best grid point.
        const double log_step =
            (log_hi - log_lo) / static_cast<double>(num_grid - 1);

        double a = std::pow(10.0, std::log10(best_lambda) - 2.0 * log_step);
        double b = std::pow(10.0, std::log10(best_lambda) + 2.0 * log_step);
        a = std::max(a, std::pow(10.0, log_lo));
        if (lambda_floor > 0.0) a = std::max(a, lambda_floor);
        b = std::min(b, std::pow(10.0, log_hi));

        // Golden-section search in log-lambda space
        constexpr double phi = 0.6180339887498949;

        double la = std::log(a);
        double lb = std::log(b);
        double lx1 = lb - phi * (lb - la);
        double lx2 = la + phi * (lb - la);
        double f1 = gcv(std::exp(lx1));
        double f2 = gcv(std::exp(lx2));

        for (int iter = 0; iter < 200; ++iter) {
            if ((lb - la) < tol) {
                break;
            }

            if (f1 < f2) {
                lb = lx2;
                lx2 = lx1;
                f2 = f1;
                lx1 = lb - phi * (lb - la);
                f1 = gcv(std::exp(lx1));
            } else {
                la = lx1;
                lx1 = lx2;
                f1 = f2;
                lx2 = la + phi * (lb - la);
                f2 = gcv(std::exp(lx2));
            }
        }

        return std::exp(0.5 * (la + lb));
    }

    /** @brief Number of samples in the dataset. */
    Eigen::Index n_samples() const noexcept { return n_; }

    /** @brief Number of features in the dataset. */
    Eigen::Index n_features() const noexcept { return p_; }

    /** @brief Rank of the matrix. */
    Eigen::Index rank() const noexcept { return r_; }

    /** @brief Singular values of the matrix. */
    const Eigen::VectorXd& singular_values() const noexcept { return sigma_; }

    /** @brief Left singular vectors of the matrix. */
    const Eigen::MatrixXd& matrix_u() const noexcept { return U_; }

    /** @brief Right singular vectors of the matrix. */
    const Eigen::MatrixXd& matrix_v() const noexcept { return V_; }

    /** @brief Rotated response vector. */
    const Eigen::VectorXd& rotated_response() const noexcept { return Uty_; }

    /** @brief Mean of the response vector. */
    double y_mean() const noexcept { return y_mean_; }

    /** @brief Mean of the feature matrix. */
    const Eigen::VectorXd& x_mean() const noexcept { return x_mean_; }


private:

    /** @brief Check if the model has been fitted. */
    void check_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "ridge_gcv: call fit() before using this method");
        }
    }

    /** @brief Check if the lambda value is valid. */
    static void check_lambda(double lambda, const char* where) {
        if (lambda < 0.0) {
            throw std::invalid_argument(
                std::string("ridge_gcv::") + where +
                ": lambda must be >= 0");
        }
    }

    /**
     * @brief Bisect in log-lambda space for the smallest lambda where
     *        effective_df(lambda) <= target_df.
     *
     * @details Used by gcv_optimal() to compute a pre-filter floor when rank is
     *          close to n (underdetermined system).  Typical call: target_df = 0.98*n,
     *          which places the floor at a lambda where (1-df/n) >= 0.02 — safely
     *          above the gcv() stability guard of 0.01.
     *          At lambda -> 0, df_eff = r_; at lambda = sigma_max^2, df_eff < r_/2.
     *          Both bounds bracket the root for r_ >= 2.
     *
     * @param target_df  Target effective degrees of freedom (e.g., 0.98 * n).
     * @return Smallest lambda such that effective_df(lambda) <= target_df.
     */
    double compute_df_floor(double target_df) const {
        double log_lo = std::log(sigma_(r_ - 1) * sigma_(r_ - 1) * 1e-10);
        double log_hi = std::log(sigma_(0)      * sigma_(0));

        for (int iter = 0; iter < 100; ++iter) {
            if ((log_hi - log_lo) < 1e-10) break;
            const double log_mid = 0.5 * (log_lo + log_hi);
            // effective_df is strictly decreasing in lambda:
            //   df too high -> need more regularisation -> raise lower bound
            if (effective_df(std::exp(log_mid)) > target_df)
                log_lo = log_mid;
            else
                log_hi = log_mid;
        }
        return std::exp(log_hi);  // conservative: df(result) <= target_df
    }

    /** @brief Indicates whether the model has been fitted. */
    bool fitted_ = false;

    /** @brief Number of samples in the dataset. */
    Eigen::Index n_ = 0;

    /** @brief Number of features in the dataset. */
    Eigen::Index p_ = 0;

    /** @brief Rank of the matrix. */
    Eigen::Index r_ = 0;

    /** @brief Left singular vectors of the matrix. */
    Eigen::MatrixXd U_;

    /** @brief Singular values of the matrix. */
    Eigen::VectorXd sigma_;

    /** @brief Right singular vectors of the matrix. */
    Eigen::MatrixXd V_;

    /** @brief Rotated response vector. */
    Eigen::VectorXd Uty_;

    /** @brief Squared norm of the centered response vector. */
    double yc_sq_norm_ = 0.0;

    /** @brief Squared norm of the rotated response vector. */
    double Uty_sq_norm_ = 0.0;

    /** @brief Squared norm of the perpendicular component of the response vector. */
    double y_perp_sq_ = 0.0;

    /** @brief Mean of the response vector. */
    double y_mean_ = 0.0;

    /** @brief Mean of the feature matrix. */
    Eigen::VectorXd x_mean_;
};

// ===================================================================================
} /* End of namespace model_selection */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_RIDGE_GCV_HPP */
