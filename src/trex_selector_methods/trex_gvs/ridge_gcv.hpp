// ridge_gcv.hpp
// Ridge Regression with Generalized Cross-Validation (GCV)
// Optimized for the high-dimensional regime p >> n.
//
// Mathematical foundation: Golub, Van Loan, "Matrix Computations", 4th ed., §6.1.4.
//
// Ridge problem (Eq. 6.1.11):
//   min_x  ||Ax - b||_2^2  +  lambda ||x||_2^2
//
// Normal equations (Eq. 6.1.12):
//   (A^T A + lambda I) x = A^T b
//
// SVD-based solution (Eq. 6.1.14):
//   A = U Sigma V^T   =>   x(lambda) = sum_i  sigma_i u_i^T b / (sigma_i^2 + lambda) v_i
//
// GCV criterion (Eq. 6.1.19), unit weights w_k = 1:
//   C(lambda) = (1/m) sum_{k=1}^{m}
//       [ b_tilde_k - sum_{j=1}^{r} u_{kj} (sigma_j^2 / (sigma_j^2 + lambda)) b_tilde_j ]^2
//       / [ 1 - sum_{j=1}^{r} u_{kj}^2 (sigma_j^2 / (sigma_j^2 + lambda)) ]^2
//
// where b_tilde = U^T b.
//
// Efficient form for GCV (Golub, Heath, Wahba 1979):
//   G(lambda) = || (I - A(lambda)) b ||_2^2 / [ trace(I - A(lambda)) ]^2 * m
//
// where A(lambda) = U diag(sigma_j^2 / (sigma_j^2 + lambda)) U^T  is the hat/influence matrix.
//
// Key design decision for p >> n:
//   When p >> n, A is m-by-p with m = n_samples, p = n_features, and m << p.
//   The economy SVD of A yields at most r = min(m, p) = m non-zero singular values.
//   We compute the thin SVD: A = U_r Sigma_r V_r^T where U_r is m-by-r, V_r is p-by-r.
//   All operations (solve, GCV, predict) run in O(m^2 p) for SVD, O(m) per lambda eval.
//   Coefficient recovery: x(lambda) = V_r * diag(sigma_i / (sigma_i^2 + lambda)) * U_r^T b
//   which is O(rp) = O(mp) per solve — unavoidable since x has p components.
//
// The class stores the thin SVD factors after fit() and supports:
//   - solve(lambda)          : compute x(lambda) for a given lambda
//   - gcv(lambda)            : evaluate GCV score at lambda
//   - gcv_optimal(lambdas)   : find best lambda from a grid
//   - gcv_optimal_golden()   : find best lambda via golden-section search
//   - predict(X_new, lambda) : predict for new data
//   - effective_df(lambda)   : effective degrees of freedom = sum sigma_i^2/(sigma_i^2+lambda)

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

namespace ridge {

/// Result structure for the full regularization path.
struct RidgePath {
    Eigen::VectorXd lambdas;     ///< Evaluated lambda values
    Eigen::MatrixXd coefficients; ///< p-by-K coefficient matrix, each column = x(lambda_k)
    Eigen::VectorXd gcv_scores;  ///< GCV score at each lambda
    Eigen::VectorXd df_effective; ///< Effective degrees of freedom at each lambda
    Eigen::Index best_index;     ///< Index of lambda with smallest GCV
};

/// Ridge Regression with GCV-based lambda selection.
///
/// Designed for the p >> n regime. Internally computes the thin SVD of the
/// (centered) design matrix. All subsequent operations (solve, GCV evaluation)
/// reuse the stored SVD factors at negligible cost per lambda.
///
/// Centering: the class centers both X and y internally. The intercept is
/// recovered as  intercept = y_mean - X_mean^T beta.  The intercept is NOT
/// penalized (standard convention, see GVL4 §6.1.4 and discussion in memory).
class RidgeGCV {
public:
    RidgeGCV() = default;

    /// Compute and store the thin SVD of the centered design matrix.
    ///
    /// @param X   Design matrix, m-by-p  (m samples, p features).
    /// @param y   Response vector, length m.
    ///
    /// Complexity: O(m^2 p) for thin SVD when p >> m.
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        if (X.rows() != y.size())
            throw std::invalid_argument("RidgeGCV::fit: X.rows() != y.size()");

        m_ = X.rows();
        p_ = X.cols();

        // ---- Centering ----
        x_mean_ = X.colwise().mean();                   // 1-by-p
        y_mean_ = y.mean();
        Eigen::MatrixXd Xc = X.rowwise() - x_mean_.transpose();
        Eigen::VectorXd yc = y.array() - y_mean_;

        // ---- Thin SVD of Xc ----
        // For p >> m, Eigen's BDCSVD with ComputeThinU | ComputeThinV is optimal.
        // U: m-by-r,  Sigma: r,  V: p-by-r   where r = min(m, p).
        Eigen::BDCSVD<Eigen::MatrixXd> svd(
            Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::VectorXd sigma_full = svd.singularValues();  // min(m,p)
        Eigen::Index r_full = sigma_full.size();

        // Determine numerical rank: discard singular values below
        // machine-epsilon * sigma_max * sqrt(max(m,p))  (cf. LAPACK convention).
        double tol = std::numeric_limits<double>::epsilon()
                     * sigma_full(0)
                     * std::sqrt(static_cast<double>(std::max(m_, p_)));
        r_ = 0;
        for (Eigen::Index i = 0; i < r_full; ++i) {
            if (sigma_full(i) > tol) ++r_;
            else break;  // sorted descending
        }
        if (r_ == 0)
            throw std::runtime_error("RidgeGCV::fit: design matrix is numerically zero");

        sigma_ = sigma_full.head(r_);
        U_ = svd.matrixU().leftCols(r_);                // m-by-r
        V_ = svd.matrixV().leftCols(r_);                // p-by-r

        // ---- Precompute U^T y (the "rotated response") ----
        // This is b_tilde in GVL4 notation.  Length r.
        Uty_ = U_.transpose() * yc;                     // r-by-1

        // ---- Precompute residual component orthogonal to range(X) ----
        // ||yc||^2 - ||U^T yc||^2  =  || (I - U U^T) yc ||^2
        yc_sq_norm_ = yc.squaredNorm();
        Uty_sq_norm_ = Uty_.squaredNorm();

        // Guard: ||y_perp||^2 = ||yc||^2 - ||U^T yc||^2 can be tiny-negative
        // due to floating-point cancellation when rank(X) = m.
        y_perp_sq_ = std::max(0.0, yc_sq_norm_ - Uty_sq_norm_);

        fitted_ = true;
    }

    // ------------------------------------------------------------------
    //  Core computational routines — all O(r) or O(rp)
    // ------------------------------------------------------------------

    /// Compute the ridge coefficient vector x(lambda).
    /// Implements GVL4 Eq. (6.1.14) via the stored thin SVD.
    ///
    /// x(lambda) = V diag( sigma_i / (sigma_i^2 + lambda) ) U^T y
    ///
    /// @return  Coefficient vector of length p (does NOT include intercept).
    Eigen::VectorXd solve(double lambda) const {
        check_fitted();
        if (lambda < 0.0)
            throw std::invalid_argument("RidgeGCV::solve: lambda must be >= 0");

        // d_i = sigma_i / (sigma_i^2 + lambda)
        Eigen::VectorXd d(r_);
        for (Eigen::Index i = 0; i < r_; ++i)
            d(i) = sigma_(i) / (sigma_(i) * sigma_(i) + lambda);

        // x = V * (d .* (U^T y))
        return V_ * (d.array() * Uty_.array()).matrix();
    }

    /// Recover the intercept for a given lambda.
    /// intercept = y_mean - x_mean^T beta(lambda).
    double intercept(double lambda) const {
        Eigen::VectorXd beta = solve(lambda);
        return y_mean_ - x_mean_.dot(beta);
    }

    /// Evaluate the Generalized Cross-Validation score at lambda.
    ///
    /// Uses the efficient "trace form" from Golub, Heath, Wahba (1979):
    ///
    ///   G(lambda) = (1/m) || r(lambda) ||^2  /  [ 1 - df(lambda)/m ]^2
    ///
    /// where
    ///   || r(lambda) ||^2 = sum_i [ lambda / (sigma_i^2 + lambda) ]^2 (u_i^T y)^2
    ///                       + || (I - UU^T) y ||^2
    ///
    ///   df(lambda)  = sum_i  sigma_i^2 / (sigma_i^2 + lambda)
    ///
    /// This is the standard simplification of Eq. (6.1.19) with unit weights,
    /// replacing the per-observation formula with the equivalent global form.
    /// See GVL4 p.308 and Golub, Heath, Wahba (1979), Technometrics 21.
    ///
    /// Complexity: O(r) per evaluation.
    double gcv(double lambda) const {
        check_fitted();
        if (lambda < 0.0)
            throw std::invalid_argument("RidgeGCV::gcv: lambda must be >= 0");

        // Residual squared norm in rotated coordinates.
        // || r(lambda) ||^2 = sum_i [lambda / (sigma_i^2 + lambda)]^2 * (u_i^T y)^2
        //                   + || y_perp ||^2
        //
        // where y_perp = (I - U U^T) y  has squared norm = yc_sq - ||U^T y||^2.
        double rss = 0.0;
        double df  = 0.0;
        for (Eigen::Index i = 0; i < r_; ++i) {
            double s2 = sigma_(i) * sigma_(i);
            double denom = s2 + lambda;
            double filter = s2 / denom;         // sigma_i^2 / (sigma_i^2 + lambda)
            double shrink = lambda / denom;      // lambda / (sigma_i^2 + lambda)

            rss += shrink * shrink * Uty_(i) * Uty_(i);
            df  += filter;
        }
        // Add the component of y orthogonal to the column space of X.
        rss += y_perp_sq_;

        // Denominator: [ 1 - df/m ]^2
        double denom = 1.0 - df / static_cast<double>(m_);
        if (std::abs(denom) < 1e-15)
            return std::numeric_limits<double>::infinity();

        return (rss / static_cast<double>(m_)) / (denom * denom);
    }

    /// Effective degrees of freedom at lambda.
    ///   df(lambda) = sum_i sigma_i^2 / (sigma_i^2 + lambda)
    double effective_df(double lambda) const {
        check_fitted();
        double df = 0.0;
        for (Eigen::Index i = 0; i < r_; ++i) {
            double s2 = sigma_(i) * sigma_(i);
            df += s2 / (s2 + lambda);
        }
        return df;
    }

    /// Predict for new data: y_hat = X_new * beta(lambda) + intercept(lambda).
    Eigen::VectorXd predict(const Eigen::MatrixXd& X_new, double lambda) const {
        check_fitted();
        if (X_new.cols() != p_)
            throw std::invalid_argument("RidgeGCV::predict: X_new.cols() != p");

        Eigen::VectorXd beta = solve(lambda);
        double b0 = y_mean_ - x_mean_.dot(beta);
        return (X_new * beta).array() + b0;
    }

    // ------------------------------------------------------------------
    //  Lambda selection
    // ------------------------------------------------------------------

    /// Evaluate GCV on a grid of lambda values and return the full path.
    ///
    /// @param lambdas  Vector of candidate lambda values (must be >= 0).
    /// @return         RidgePath with coefficients, GCV scores, and best index.
    RidgePath solve_path(const Eigen::VectorXd& lambdas) const {
        check_fitted();
        Eigen::Index K = lambdas.size();

        RidgePath path;
        path.lambdas = lambdas;
        path.coefficients.resize(p_, K);
        path.gcv_scores.resize(K);
        path.df_effective.resize(K);

        double best_gcv = std::numeric_limits<double>::infinity();
        path.best_index = 0;

        for (Eigen::Index k = 0; k < K; ++k) {
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

    /// Generate a logarithmically spaced grid of lambda values.
    ///
    /// Range: [sigma_min^2 / ratio,  sigma_max^2 * ratio]
    /// where sigma_min, sigma_max are the smallest/largest singular values.
    /// This covers the "interesting" range where GCV transitions.
    ///
    /// @param n_lambda   Number of grid points.
    /// @param ratio      Expansion factor beyond the singular value range.
    /// @return           Log-spaced lambda grid, sorted ascending.
    Eigen::VectorXd default_lambda_grid(
        Eigen::Index n_lambda = 100,
        double ratio = 1000.0) const
    {
        check_fitted();
        double s_max = sigma_(0);
        double s_min = sigma_(r_ - 1);

        // Guard against zero singular values.
        if (s_min < 1e-15 * s_max) s_min = s_max * 1e-10;

        double log_lo = std::log10(s_min * s_min / ratio);
        double log_hi = std::log10(s_max * s_max * ratio);

        Eigen::VectorXd grid(n_lambda);
        for (Eigen::Index i = 0; i < n_lambda; ++i)
            grid(i) = std::pow(10.0,
                log_lo + (log_hi - log_lo) * i / (n_lambda - 1));
        return grid;
    }

    /// Find the GCV-optimal lambda by grid search on a log-spaced grid,
    /// followed by golden-section refinement around the best grid point.
    ///
    /// GCV is not guaranteed to be unimodal (especially near the
    /// interpolation boundary when rank ~ m), so a pure golden-section
    /// search can converge to a degenerate local minimum at lambda ~ 0.
    /// The two-phase approach is robust against this.
    ///
    /// @param n_grid   Number of initial grid points (log-spaced).
    /// @param tol      Relative tolerance for refinement.
    /// @return         Approximately optimal lambda.
    double gcv_optimal(
        int n_grid = 200,
        double tol = 1e-8) const
    {
        check_fitted();

        // ---- Phase 1: coarse grid search ----
        double s_max_sq = sigma_(0) * sigma_(0);
        double s_min_sq = sigma_(r_ - 1) * sigma_(r_ - 1);
        if (s_min_sq < 1e-20 * s_max_sq) s_min_sq = s_max_sq * 1e-10;

        // Search range: [s_min^2 * 1e-4,  s_max^2 * 1e4]
        // This spans from well below the smallest squared singular value
        // to well above the largest, covering the entire transition region.
        double log_lo = std::log10(s_min_sq * 1e-4);
        double log_hi = std::log10(s_max_sq * 1e4);

        double best_lam = std::exp(0.5 * (std::log(s_min_sq) + std::log(s_max_sq)));
        double best_gcv = std::numeric_limits<double>::infinity();

        for (int i = 0; i < n_grid; ++i) {
            double lam = std::pow(10.0,
                log_lo + (log_hi - log_lo) * i / (n_grid - 1));
            double g = gcv(lam);
            if (g < best_gcv) {
                best_gcv = g;
                best_lam = lam;
            }
        }

        // ---- Phase 2: golden-section refinement ----
        // Bracket: one grid step on each side.
        double log_step = (log_hi - log_lo) / (n_grid - 1);
        double a = std::pow(10.0, std::log10(best_lam) - 2.0 * log_step);
        double b = std::pow(10.0, std::log10(best_lam) + 2.0 * log_step);
        a = std::max(a, std::pow(10.0, log_lo));
        b = std::min(b, std::pow(10.0, log_hi));

        // Golden section on log-scale for better resolution.
        constexpr double phi = 0.6180339887498949;
        double la = std::log(a), lb = std::log(b);
        double lx1 = lb - phi * (lb - la);
        double lx2 = la + phi * (lb - la);
        double f1 = gcv(std::exp(lx1));
        double f2 = gcv(std::exp(lx2));

        for (int iter = 0; iter < 200; ++iter) {
            if ((lb - la) < tol) break;

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

    // ------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------

    Eigen::Index n_samples()  const { return m_; }
    Eigen::Index n_features() const { return p_; }
    Eigen::Index rank()       const { return r_; }

    /// Singular values of the centered design matrix (descending order).
    const Eigen::VectorXd& singular_values() const { return sigma_; }

    /// Left singular vectors U (m-by-r).
    const Eigen::MatrixXd& matrixU() const { return U_; }

    /// Right singular vectors V (p-by-r).
    const Eigen::MatrixXd& matrixV() const { return V_; }

    /// Rotated response: U^T (y - y_mean).
    const Eigen::VectorXd& rotated_response() const { return Uty_; }

    double y_mean() const { return y_mean_; }
    const Eigen::VectorXd& x_mean() const { return x_mean_; }

private:
    void check_fitted() const {
        if (!fitted_)
            throw std::runtime_error("RidgeGCV: call fit() before using this method");
    }

    bool fitted_ = false;

    Eigen::Index m_ = 0;          ///< Number of samples
    Eigen::Index p_ = 0;          ///< Number of features
    Eigen::Index r_ = 0;          ///< Rank (number of non-zero singular values stored)

    Eigen::MatrixXd U_;           ///< Left singular vectors,  m-by-r
    Eigen::VectorXd sigma_;       ///< Singular values,        r-by-1
    Eigen::MatrixXd V_;           ///< Right singular vectors, p-by-r

    Eigen::VectorXd Uty_;         ///< U^T y_centered,         r-by-1
    double yc_sq_norm_ = 0.0;     ///< ||y_centered||^2
    double Uty_sq_norm_ = 0.0;    ///< ||U^T y_centered||^2
    double y_perp_sq_ = 0.0;      ///< max(0, ||yc||^2 - ||U^Tyc||^2)

    double y_mean_ = 0.0;
    Eigen::VectorXd x_mean_;      ///< Column means of X,      p-by-1
};

} // namespace ridge
