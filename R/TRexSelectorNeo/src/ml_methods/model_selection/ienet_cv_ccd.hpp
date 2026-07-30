// ===================================================================================
// ienet_cv_ccd.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_IENET_CV_CCD_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_IENET_CV_CCD_HPP
// ===================================================================================
/**
 * @file ienet_cv_ccd.hpp
 *
 * @brief Cyclic coordinate-descent Informed Elastic Net solver in the
 *        T-solver objective scaling, plus a K-fold cross-validation wrapper
 *        that selects lambda2 over a DIRECT grid by profiling the lambda1
 *        path.  Companion of `enet_cv_ccd` for the IEN geometry: the selected
 *        lambda2 is consumable AS-IS by `TIENET_Solver` / `TCIENET_Solver`
 *        (no p/2 conversion).
 *
 * @details
 *   **Objective (solver scale).**  `ien_gaussian` computes exact minimizers of
 *
 *       F(beta) = 1/2 ||y_c - X_s beta||^2 + lambda1 ||beta||_1
 *                 + (lambda2/2) beta^T K beta
 *
 *   on the standardized (centered, column-L2-normalized) scale used by the
 *   LARS/CD T-solvers — NOT glmnet's 1/(2n) parameterization.  Two penalty
 *   specializations mirror `TCIENET_Solver`:
 *
 *   - Group mode: the IEN group-mean penalty K = W = sum_m 1_m 1_m^T / p_m,
 *     maintained through the O(M) group-sum bookkeeping
 *     sigma_m = sum_{i in G_m} beta_i (O(1) per coordinate update);
 *   - General mode: an arbitrary sparse PSD Tikhonov matrix K, maintained
 *     through the incrementally updated vector lambda2 * K * beta
 *     (O(nnz(K col)) per coordinate change).
 *
 *   The coordinate update is the penalized soft-threshold step
 *
 *       beta_j <- S( x_j^T r + d_j beta_j - lambda2 [(K beta)_j - K_jj beta_j],
 *                    lambda1 ) / (d_j + lambda2 K_jj) ,
 *
 *   whose fixed points satisfy exactly the IEN KKT conditions with the
 *   modified correlation c_j = x_j^T r - lambda2 (K beta)_j of the T-CIENET /
 *   LARS-IEN solvers.
 *
 *   **Well-posedness.**  Unlike the pure Tikhonov backbone of
 *   `tikhonov_cv_svd` — which for the rank-M group penalty is singular
 *   whenever p > n_train + M and whose prediction-CV curve is constant in
 *   lambda2 once p - M >= n_train — every point of the lambda1 path here has
 *   lambda1 > 0, so the estimator is well-defined in any p/n regime.  This is
 *   the recommended lambda2 prediction-CV tuner for the T-Rex p > n setting.
 *
 *   **CV wrapper.**  `ienet_cv_ccd` fits, for every lambda2 candidate on an
 *   ascending geometric grid, the full descending lambda1 path per training
 *   fold (warm-started), scores held-out MSE at every (lambda2, lambda1),
 *   and PROFILES over lambda1:
 *
 *       cv_mse(lambda2) = min_{lambda1}  mean-fold MSE(lambda2, lambda1) .
 *
 *   `lambda.min` / `lambda.1se` are then applied to the profiled curve
 *   (1se: largest lambda2 within one SE of the minimum, using the SEM at the
 *   profiled lambda1).  The lambda1 grid is anchored once on the full data
 *   (lambda1_max = max_j |x_{s,j}^T y_c|, shared across folds and lambda2 —
 *   cv.glmnet semantics); the lambda2 grid is anchored at the same c_max so
 *   both parameters live on the solver correlation scale.
 *
 *   **Caveat.**  Prediction-CV — even in the correct penalty geometry — is
 *   biased toward smaller lambda2 than the selection-optimal value for
 *   terminating selectors: it validates shrinkage, not support recovery.
 *   Empirically `cv_min()` can pin to the LOWER grid edge; prefer the 1SE
 *   rule (`cv_1se()`), which counteracts the bias.
 *
 *   **Public interface** (parity with `enet_cv_ccd` / `ridge_cv_svd`):
 *   `fit()`, `cv_min()`, `cv_1se()`, `lambdas()`, `cv_mse()`, `cv_sem()`,
 *   `index_min()`, `index_1se()`; additionally `lambda1s()`,
 *   `lambda1_profile()`, `lambda1_at_min()`, `lambda1_at_1se()`,
 *   `cv_mse_full()`.
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
#include <Eigen/SparseCore>

// ===================================================================================

// Namespace embedding: trex::ml_methods::model_selection
namespace trex {
namespace ml_methods {
namespace model_selection {

// ===================================================================================

/**
 * @brief Exact pathwise CCD solver for the Informed Elastic Net in the
 *        T-solver objective scaling (fixed lambda2, descending lambda1 grid,
 *        warm starts), with group-mean and general sparse-K penalties.
 */
class ien_gaussian {

public:

    /** @brief Default constructor. */
    ien_gaussian() = default;

    /**
     * @brief Fit the IEN path with the group-mean penalty (TIENET geometry).
     *
     * @param X            Predictor matrix (n x p).
     * @param y            Response vector (n).
     * @param lambda2      Group-mean (L2) penalty weight (>= 0).
     * @param groups       0-based contiguous group ids in [0, M-1], length p.
     * @param lambda1_grid Explicit lambda1 values (> 0, any order; sorted
     *                     descending internally for warm starts).
     * @param max_iter     Max CD sweeps per lambda1 (default 100000).
     * @param tol          Convergence tolerance on the max coordinate change
     *                     in standardized units of the unit-SD response
     *                     (default 1e-7).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             double lambda2,
             const Eigen::VectorXi& groups,
             Eigen::Ref<const Eigen::VectorXd> lambda1_grid,
             int    max_iter = 100000,
             double tol      = 1e-7)
    {
        validate(X, y, lambda2, lambda1_grid);
        setup_group_penalty(groups, X.cols());
        preprocess(X, y);
        solve(lambda2, lambda1_grid, max_iter, tol);
    }

    /**
     * @brief Fit the IEN path with a general sparse PSD Tikhonov matrix K
     *        (matrix-passing variant; K acts on the standardized scale).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             double lambda2,
             const Eigen::SparseMatrix<double>& K,
             Eigen::Ref<const Eigen::VectorXd> lambda1_grid,
             int    max_iter = 100000,
             double tol      = 1e-7)
    {
        validate(X, y, lambda2, lambda1_grid);
        setup_sparse_penalty(K, X.cols());
        preprocess(X, y);
        solve(lambda2, lambda1_grid, max_iter, tol);
    }

    // public member accessors ------------------------------------------------

    /** @brief lambda1 grid actually used (descending, solver scale). */
    const Eigen::VectorXd& lambda1s() const noexcept { return lambda1s_; }

    /** @brief Coefficient path in ORIGINAL predictor scale (p x n_lambda1). */
    const Eigen::MatrixXd& coef() const noexcept { return beta_; }

    /** @brief Coefficient path on the STANDARDIZED (solver) scale
     *         (p x n_lambda1) — the scale on which K acts. */
    const Eigen::MatrixXd& coef_std() const noexcept { return beta_std_; }

    /** @brief Intercepts per lambda1 (n_lambda1). */
    const Eigen::VectorXd& intercepts() const noexcept { return a0_; }

    /** @brief True if every lambda1 reached the convergence tolerance. */
    bool converged() const noexcept { return n_nonconverged_ == 0; }

    /** @brief Number of lambda1 points that hit `max_iter`. */
    int n_nonconverged() const noexcept { return n_nonconverged_; }

    /** @brief Predictions for new data: (X_new * beta).colwise() + a0. */
    Eigen::MatrixXd predict(Eigen::Ref<const Eigen::MatrixXd> X_new) const {
        return (X_new * beta_).rowwise() + a0_.transpose();
    }

private:

    // ---- penalty configuration ----------------------------------------------

    void setup_group_penalty(const Eigen::VectorXi& groups, Eigen::Index p)
    {
        if (groups.size() != p)
            throw std::invalid_argument(
                "ien_gaussian: groups.size() != X.cols()");
        const int M = groups.size() > 0 ? groups.maxCoeff() + 1 : 0;
        if (M <= 0)
            throw std::invalid_argument("ien_gaussian: empty grouping");
        std::vector<double> psz(static_cast<std::size_t>(M), 0.0);
        for (Eigen::Index j = 0; j < p; ++j) {
            const int m = groups(j);
            if (m < 0 || m >= M)
                throw std::invalid_argument(
                    "ien_gaussian: group ids must be 0-based contiguous");
            psz[static_cast<std::size_t>(m)] += 1.0;
        }
        for (double s : psz)
            if (s <= 0.0)
                throw std::invalid_argument(
                    "ien_gaussian: group ids must be 0-based contiguous "
                    "(empty group id encountered)");
        use_group_ = true;
        group_of_.assign(static_cast<std::size_t>(p), 0);
        for (Eigen::Index j = 0; j < p; ++j)
            group_of_[static_cast<std::size_t>(j)] = groups(j);
        inv_pm_.resize(static_cast<std::size_t>(M));
        for (int m = 0; m < M; ++m)
            inv_pm_[static_cast<std::size_t>(m)] =
                1.0 / psz[static_cast<std::size_t>(m)];
    }

    void setup_sparse_penalty(const Eigen::SparseMatrix<double>& K,
                              Eigen::Index p)
    {
        if (K.rows() != p || K.cols() != p)
            throw std::invalid_argument(
                "ien_gaussian: K must be p x p with p = X.cols()");
        use_group_ = false;
        K_ = K;
        K_.makeCompressed();
        Kdiag_.resize(p);
        for (Eigen::Index j = 0; j < p; ++j)
            Kdiag_(j) = K_.coeff(j, j);
        if (Kdiag_.minCoeff() < -1e-12)
            throw std::invalid_argument(
                "ien_gaussian: K has negative diagonal (not PSD)");
    }

    // ---- preprocessing: center + column-L2-normalize (T-solver scale) ------
    //
    // The response is additionally scaled to unit population SD INTERNALLY:
    // with beta~ = beta_s / y_scale and lambda1~ = lambda1 / y_scale the
    // problem is an exact reparameterization with the SAME lambda2 (the
    // quadratic terms scale together), so the reported lambda2 remains on the
    // unscaled-y solver scale while the CD tolerance acts in unit-SD units.
    void preprocess(Eigen::Ref<const Eigen::MatrixXd> X,
                    Eigen::Ref<const Eigen::VectorXd> y)
    {
        n_ = X.rows();
        p_ = X.cols();

        x_mean_ = X.colwise().mean();
        y_mean_ = y.mean();

        Xs_ = X.rowwise() - x_mean_.transpose();
        x_scale_.resize(p_);
        for (Eigen::Index j = 0; j < p_; ++j) {
            const double nrm = Xs_.col(j).norm();
            x_scale_(j) = (nrm > 1e-10) ? nrm : 1.0;
            Xs_.col(j) /= x_scale_(j);
        }

        yc_ = y.array() - y_mean_;
        y_scale_ = std::sqrt(yc_.squaredNorm() / static_cast<double>(n_));
        if (y_scale_ > 1e-12) { yc_ /= y_scale_; } else { y_scale_ = 1.0; }
    }

    // ---- pathwise CD core ---------------------------------------------------

    void solve(double lambda2, Eigen::Ref<const Eigen::VectorXd> grid,
               int max_iter, double tol)
    {
        lambda1s_ = grid;
        std::sort(lambda1s_.data(), lambda1s_.data() + lambda1s_.size(),
                  std::greater<double>());
        const Eigen::Index K1 = lambda1s_.size();

        // d_j = ||x_{s,j}||^2 (== 1 after normalization; kept general).
        Eigen::VectorXd d(p_);
        for (Eigen::Index j = 0; j < p_; ++j)
            d(j) = Xs_.col(j).squaredNorm();

        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p_);  // internal scale
        Eigen::VectorXd r    = yc_;
        beta_std_.setZero(p_, K1);
        n_nonconverged_ = 0;

        // penalty state
        std::vector<double> sigma;                 // group sums (group mode)
        Eigen::VectorXd     Kbeta;                 // K * beta (sparse mode)
        if (use_group_) sigma.assign(inv_pm_.size(), 0.0);
        else            Kbeta = Eigen::VectorXd::Zero(p_);

        const auto pen_grad_excl = [&](Eigen::Index j) -> double {
            // lambda2 * [(K beta)_j - K_jj beta_j]
            if (use_group_) {
                const std::size_t m =
                    static_cast<std::size_t>(group_of_[static_cast<std::size_t>(j)]);
                return lambda2 * (sigma[m] - beta(j)) * inv_pm_[m];
            }
            return lambda2 * (Kbeta(j) - Kdiag_(j) * beta(j));
        };
        const auto pen_diag = [&](Eigen::Index j) -> double {
            if (use_group_) {
                const std::size_t m =
                    static_cast<std::size_t>(group_of_[static_cast<std::size_t>(j)]);
                return lambda2 * inv_pm_[m];
            }
            return lambda2 * Kdiag_(j);
        };
        const auto pen_update = [&](Eigen::Index j, double delta) {
            if (use_group_) {
                sigma[static_cast<std::size_t>(
                    group_of_[static_cast<std::size_t>(j)])] += delta;
            } else {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K_, j); it;
                     ++it)
                    Kbeta(it.row()) += it.value() * delta;
            }
        };

        // one coordinate update; returns the standardized |change|.
        const auto update_coord = [&](Eigen::Index j, double l1) -> double {
            const double old_b = beta(j);
            const double z     = Xs_.col(j).dot(r) + d(j) * old_b
                               - pen_grad_excl(j);
            const double new_b = soft_threshold(z, l1) / (d(j) + pen_diag(j));
            const double diff  = new_b - old_b;
            if (diff != 0.0) {
                r.noalias() -= Xs_.col(j) * diff;
                beta(j) = new_b;
                pen_update(j, diff);
                return std::abs(diff) * std::sqrt(d(j));
            }
            return 0.0;
        };

        const double inv_yscale = 1.0 / y_scale_;
        for (Eigen::Index l = 0; l < K1; ++l) {
            const double l1 = lambda1s_(l) * inv_yscale;  // internal scale

            // Full sweeps with an inner active-set loop (glmnet's covariance-
            // free scheme without the strong rule): a full sweep both makes
            // progress and certifies the KKT conditions at the fixed point.
            bool converged = false;
            int  it = 0;
            while (it < max_iter) {
                double max_change = 0.0;
                for (Eigen::Index j = 0; j < p_; ++j)
                    max_change = std::max(max_change, update_coord(j, l1));
                ++it;
                if (max_change < tol) { converged = true; break; }

                // inner loop on the current support
                std::vector<Eigen::Index> active;
                active.reserve(static_cast<std::size_t>(p_));
                for (Eigen::Index j = 0; j < p_; ++j)
                    if (beta(j) != 0.0) active.push_back(j);
                while (it < max_iter) {
                    double mc = 0.0;
                    for (Eigen::Index j : active)
                        mc = std::max(mc, update_coord(j, l1));
                    ++it;
                    if (mc < tol) break;
                }
            }
            if (!converged && it >= max_iter) ++n_nonconverged_;

            beta_std_.col(l) = beta * y_scale_;   // solver (unscaled-y) scale
        }

        // back-scale to original predictor space + intercepts
        beta_.resize(p_, K1);
        for (Eigen::Index j = 0; j < p_; ++j)
            beta_.row(j) = beta_std_.row(j) / x_scale_(j);
        a0_.resize(K1);
        for (Eigen::Index l = 0; l < K1; ++l)
            a0_(l) = y_mean_ - x_mean_.dot(beta_.col(l));
    }

    static void validate(Eigen::Ref<const Eigen::MatrixXd> X,
                         Eigen::Ref<const Eigen::VectorXd> y,
                         double lambda2,
                         Eigen::Ref<const Eigen::VectorXd> grid)
    {
        if (X.rows() != y.size())
            throw std::invalid_argument("ien_gaussian: X.rows() != y.size()");
        if (X.rows() < 2)
            throw std::invalid_argument("ien_gaussian: need n >= 2");
        if (lambda2 < 0.0)
            throw std::invalid_argument("ien_gaussian: lambda2 must be >= 0");
        if (grid.size() == 0)
            throw std::invalid_argument("ien_gaussian: lambda1_grid is empty");
        if (grid.minCoeff() <= 0.0)
            throw std::invalid_argument(
                "ien_gaussian: lambda1 values must be > 0 (the lambda1 = 0 "
                "backbone is ill-posed for p > n; see tikhonov_cv_svd)");
    }

    static inline double soft_threshold(double z, double gamma) noexcept {
        if (z >  gamma) return z - gamma;
        if (z < -gamma) return z + gamma;
        return 0.0;
    }

    // private member variables ----------------------------------------------
    Eigen::Index    n_ = 0;
    Eigen::Index    p_ = 0;
    Eigen::MatrixXd Xs_;
    Eigen::VectorXd yc_;
    Eigen::VectorXd x_mean_;
    Eigen::VectorXd x_scale_;
    double          y_mean_  = 0.0;
    double          y_scale_ = 1.0;

    bool                        use_group_ = true;
    std::vector<int>            group_of_;
    std::vector<double>         inv_pm_;
    Eigen::SparseMatrix<double> K_;
    Eigen::VectorXd             Kdiag_;

    Eigen::VectorXd lambda1s_;
    Eigen::MatrixXd beta_;       ///< original scale (p x K1)
    Eigen::MatrixXd beta_std_;   ///< standardized/solver scale (p x K1)
    Eigen::VectorXd a0_;
    int             n_nonconverged_ = 0;
};

// ===================================================================================

/**
 * @brief K-fold cross-validated lambda2 selection for the Informed Elastic
 *        Net via the CCD `ien_gaussian` engine: per lambda2 candidate the
 *        full lambda1 path is fitted per fold and profiled out.
 */
class ienet_cv_ccd {

public:

    /** @brief Default constructor. */
    ienet_cv_ccd() = default;

    /**
     * @brief Fit K-fold CV with the group-mean penalty (TIENET geometry).
     *
     * @param X                 Predictor matrix (n x p).
     * @param y                 Response vector (n).
     * @param groups            0-based contiguous group ids, length p.
     * @param n_folds           Number of folds (default 10).
     * @param n_lambda2         lambda2 grid size (default 25).
     * @param lambda2_ratio     Grid spans `[l2_max / ratio, l2_max]` with
     *                          l2_max = c_max = max_j |x_{s,j}^T y_c|
     *                          (default 1e4).
     * @param n_lambda1         lambda1 path length (default 100).
     * @param lambda1_min_ratio lambda1_min / lambda1_max.  Negative (default)
     *                          auto-selects glmnet's rule
     *                          (1e-2 if n < p else 1e-4).
     * @param seed              Fold-permutation seed (default 0).
     * @param max_iter          CD sweeps per lambda1 (default 100000).
     * @param tol               CD tolerance (default 1e-7).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             const Eigen::VectorXi& groups,
             int          n_folds           = 10,
             Eigen::Index n_lambda2         = 25,
             double       lambda2_ratio     = 1e4,
             Eigen::Index n_lambda1         = 100,
             double       lambda1_min_ratio = -1.0,
             unsigned int seed              = 0,
             int          max_iter          = 100000,
             double       tol               = 1e-7)
    {
        groups_ = groups;
        use_group_ = true;
        run_cv(X, y, n_folds, n_lambda2, lambda2_ratio, n_lambda1,
               lambda1_min_ratio, seed, max_iter, tol);
    }

    /**
     * @brief Fit K-fold CV with a general sparse PSD Tikhonov matrix K.
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             const Eigen::SparseMatrix<double>& K,
             int          n_folds           = 10,
             Eigen::Index n_lambda2         = 25,
             double       lambda2_ratio     = 1e4,
             Eigen::Index n_lambda1         = 100,
             double       lambda1_min_ratio = -1.0,
             unsigned int seed              = 0,
             int          max_iter          = 100000,
             double       tol               = 1e-7)
    {
        K_ = K;
        use_group_ = false;
        run_cv(X, y, n_folds, n_lambda2, lambda2_ratio, n_lambda1,
               lambda1_min_ratio, seed, max_iter, tol);
    }

    // public member accessors ------------------------------------------------

    /** @brief lambda2 grid that was searched (ascending, solver scale). */
    const Eigen::VectorXd& lambdas() const noexcept { return lambdas_; }

    /** @brief Profiled mean CV-MSE per lambda2 (min over the lambda1 path). */
    const Eigen::VectorXd& cv_mse() const noexcept { return cv_mse_; }

    /** @brief SEM of the CV-MSE mean at the profiled lambda1, per lambda2. */
    const Eigen::VectorXd& cv_sem() const noexcept { return cv_sem_; }

    /** @brief lambda2 minimising the profiled CV-MSE (`lambda.min`),
     *         directly consumable by TIENET_Solver / TCIENET_Solver. */
    double cv_min() const { check_fitted(); return lambdas_(idx_min_); }

    /** @brief Largest lambda2 within one SE of the CV-min (`lambda.1se`). */
    double cv_1se() const { check_fitted(); return lambdas_(idx_1se_); }

    /** @brief Index of the lambda2 minimising the profiled CV-MSE. */
    Eigen::Index index_min() const noexcept { return idx_min_; }

    /** @brief Index of the largest lambda2 within one SE of the minimum. */
    Eigen::Index index_1se() const noexcept { return idx_1se_; }

    /** @brief Shared lambda1 path (descending, solver scale). */
    const Eigen::VectorXd& lambda1s() const noexcept { return lambda1s_; }

    /** @brief Profiled lambda1 (argmin of the path CV-MSE) per lambda2. */
    const Eigen::VectorXd& lambda1_profile() const noexcept {
        return lambda1_profile_;
    }

    /** @brief Profiled lambda1 at `lambda.min` / `lambda.1se`. */
    double lambda1_at_min() const { check_fitted();
        return lambda1_profile_(idx_min_); }
    double lambda1_at_1se() const { check_fitted();
        return lambda1_profile_(idx_1se_); }

    /** @brief Full mean CV-MSE surface (n_lambda2 x n_lambda1). */
    const Eigen::MatrixXd& cv_mse_full() const noexcept { return cv_mse_full_; }

private:

    void run_cv(Eigen::Ref<const Eigen::MatrixXd> X,
                Eigen::Ref<const Eigen::VectorXd> y,
                int n_folds, Eigen::Index n_lambda2, double lambda2_ratio,
                Eigen::Index n_lambda1, double lambda1_min_ratio,
                unsigned int seed, int max_iter, double tol)
    {
        if (X.rows() != y.size())
            throw std::invalid_argument(
                "ienet_cv_ccd::fit: X.rows() != y.size()");
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        if (n_folds < 2)
            throw std::invalid_argument("ienet_cv_ccd::fit: n_folds >= 2");
        if (static_cast<Eigen::Index>(n_folds) > n)
            throw std::invalid_argument("ienet_cv_ccd::fit: n_folds <= n");
        if (n_lambda2 <= 0 || n_lambda1 <= 0)
            throw std::invalid_argument(
                "ienet_cv_ccd::fit: grid sizes must be positive");
        if (lambda2_ratio <= 0.0)
            throw std::invalid_argument(
                "ienet_cv_ccd::fit: lambda2_ratio must be positive");

        // ---- full-data anchor c_max = max |x_s^T y_c| (solver scale) -------
        double c_max = 0.0;
        {
            const Eigen::VectorXd x_mean = X.colwise().mean();
            const double y_mean = y.mean();
            Eigen::VectorXd x_scale =
                (X.colwise().squaredNorm().transpose()
                 - static_cast<double>(n) * x_mean.cwiseAbs2()
                ).cwiseMax(0.0).cwiseSqrt();
            for (Eigen::Index j = 0; j < p; ++j)
                if (x_scale(j) < 1e-10) x_scale(j) = 1.0;
            Eigen::VectorXd Xty = X.transpose() * y;
            Xty -= static_cast<double>(n) * y_mean * x_mean;
            c_max = Xty.cwiseQuotient(x_scale).cwiseAbs().maxCoeff();
        }
        if (!(c_max > 1e-12) || !std::isfinite(c_max))
            throw std::runtime_error(
                "ienet_cv_ccd::fit: y is (nearly) orthogonal to X; "
                "grid anchor c_max ~ 0.");

        // ---- lambda2 grid (ascending) and shared lambda1 path (descending) -
        lambdas_.resize(n_lambda2);
        {
            const double log_hi = std::log10(c_max);
            const double log_lo = std::log10(c_max / lambda2_ratio);
            if (n_lambda2 == 1) {
                lambdas_(0) = std::pow(10.0, 0.5 * (log_lo + log_hi));
            } else {
                for (Eigen::Index i = 0; i < n_lambda2; ++i)
                    lambdas_(i) = std::pow(
                        10.0, log_lo + (log_hi - log_lo)
                              * static_cast<double>(i)
                              / static_cast<double>(n_lambda2 - 1));
            }
        }
        const double l1mr = (lambda1_min_ratio > 0.0)
                          ? lambda1_min_ratio
                          : (n < p ? 1e-2 : 1e-4);
        lambda1s_.resize(n_lambda1);
        {
            const double log_hi = std::log(c_max);
            const double log_lo = std::log(c_max * l1mr);
            if (n_lambda1 == 1) {
                lambda1s_(0) = c_max;
            } else {
                for (Eigen::Index i = 0; i < n_lambda1; ++i)
                    lambda1s_(i) = std::exp(
                        log_hi - (log_hi - log_lo)
                                 * static_cast<double>(i)
                                 / static_cast<double>(n_lambda1 - 1));
            }
        }
        const Eigen::Index K2 = n_lambda2, K1 = n_lambda1;

        // ---- deterministic folds (identical scheme to ridge_cv_svd) --------
        std::vector<Eigen::Index> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), Eigen::Index{0});
        std::mt19937 rng(seed);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<std::vector<Eigen::Index>> fold_idx(
            static_cast<std::size_t>(n_folds));
        for (Eigen::Index i = 0; i < n; ++i)
            fold_idx[static_cast<std::size_t>(i % n_folds)].push_back(
                perm[static_cast<std::size_t>(i)]);

        // fold-wise MSE surfaces
        std::vector<Eigen::MatrixXd> fold_mse(
            static_cast<std::size_t>(n_folds));

        for (int f = 0; f < n_folds; ++f) {
            const auto& test_idx      = fold_idx[static_cast<std::size_t>(f)];
            const Eigen::Index n_test = static_cast<Eigen::Index>(test_idx.size());
            const Eigen::Index n_tr   = n - n_test;

            std::vector<bool> in_test(static_cast<std::size_t>(n), false);
            for (Eigen::Index t : test_idx)
                in_test[static_cast<std::size_t>(t)] = true;

            Eigen::MatrixXd X_tr(n_tr, p), X_te(n_test, p);
            Eigen::VectorXd y_tr(n_tr),    y_te(n_test);
            for (Eigen::Index i = 0, itr = 0, ite = 0; i < n; ++i) {
                if (in_test[static_cast<std::size_t>(i)]) {
                    X_te.row(ite) = X.row(i); y_te(ite) = y(i); ++ite;
                } else {
                    X_tr.row(itr) = X.row(i); y_tr(itr) = y(i); ++itr;
                }
            }

            Eigen::MatrixXd mse(K2, K1);
            for (Eigen::Index k2 = 0; k2 < K2; ++k2) {
                ien_gaussian en;
                if (use_group_)
                    en.fit(X_tr, y_tr, lambdas_(k2), groups_, lambda1s_,
                           max_iter, tol);
                else
                    en.fit(X_tr, y_tr, lambdas_(k2), K_, lambda1s_,
                           max_iter, tol);
                const Eigen::MatrixXd preds = en.predict(X_te); // n_test x K1
                mse.row(k2) =
                    (preds.colwise() - y_te).colwise().squaredNorm()
                    / static_cast<double>(n_test);
            }
            fold_mse[static_cast<std::size_t>(f)] = std::move(mse);
        }

        // ---- aggregate mean + SEM over folds, per (lambda2, lambda1) -------
        cv_mse_full_.setZero(K2, K1);
        for (int f = 0; f < n_folds; ++f)
            cv_mse_full_ += fold_mse[static_cast<std::size_t>(f)];
        cv_mse_full_ /= static_cast<double>(n_folds);

        Eigen::MatrixXd sem_full(K2, K1);
        {
            const double inv_F      = 1.0 / static_cast<double>(n_folds);
            const double inv_Fminus = 1.0 / static_cast<double>(n_folds - 1);
            for (Eigen::Index k2 = 0; k2 < K2; ++k2) {
                for (Eigen::Index k1 = 0; k1 < K1; ++k1) {
                    double var = 0.0;
                    for (int f = 0; f < n_folds; ++f) {
                        const double d =
                            fold_mse[static_cast<std::size_t>(f)](k2, k1)
                            - cv_mse_full_(k2, k1);
                        var += d * d;
                    }
                    sem_full(k2, k1) = std::sqrt(var * inv_Fminus * inv_F);
                }
            }
        }

        // ---- profile over lambda1 per lambda2 ------------------------------
        cv_mse_.resize(K2);
        cv_sem_.resize(K2);
        lambda1_profile_.resize(K2);
        for (Eigen::Index k2 = 0; k2 < K2; ++k2) {
            Eigen::Index k1_best = 0;
            double best = std::numeric_limits<double>::max();
            for (Eigen::Index k1 = 0; k1 < K1; ++k1)
                if (cv_mse_full_(k2, k1) < best) {
                    best = cv_mse_full_(k2, k1);
                    k1_best = k1;
                }
            cv_mse_(k2)          = best;
            cv_sem_(k2)          = sem_full(k2, k1_best);
            lambda1_profile_(k2) = lambda1s_(k1_best);
        }

        // ---- lambda.min / lambda.1se on the profiled curve -----------------
        Eigen::Index k_min = 0;
        double best = std::numeric_limits<double>::max();
        for (Eigen::Index k = 0; k < K2; ++k)
            if (cv_mse_(k) < best) { best = cv_mse_(k); k_min = k; }
        idx_min_ = k_min;

        const double threshold = cv_mse_(k_min) + cv_sem_(k_min);
        Eigen::Index k_1se = k_min;
        for (Eigen::Index k = K2 - 1; k >= 0; --k) {
            if (cv_mse_(k) <= threshold) { k_1se = k; break; }
            if (k == 0) break;
        }
        idx_1se_ = k_1se;

        fitted_ = true;
    }

    void check_fitted() const {
        if (!fitted_)
            throw std::runtime_error(
                "ienet_cv_ccd: call fit() before using this method");
    }

    // private member variables ----------------------------------------------
    bool                        fitted_    = false;
    bool                        use_group_ = true;
    Eigen::VectorXi             groups_;
    Eigen::SparseMatrix<double> K_;

    Eigen::VectorXd lambdas_;          ///< lambda2 grid (ascending)
    Eigen::VectorXd lambda1s_;         ///< shared lambda1 path (descending)
    Eigen::VectorXd cv_mse_;           ///< profiled curve
    Eigen::VectorXd cv_sem_;
    Eigen::VectorXd lambda1_profile_;
    Eigen::MatrixXd cv_mse_full_;      ///< K2 x K1 surface
    Eigen::Index    idx_min_ = 0;
    Eigen::Index    idx_1se_ = 0;
};

// ===================================================================================
} /* End of namespace model_selection */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_IENET_CV_CCD_HPP */
