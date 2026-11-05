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
 *        lambda2 is consumable AS-IS by `TIENET_Solver` / `TCDIENET_Solver`
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
 *   specializations mirror `TCDIENET_Solver`:
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
 *   modified correlation c_j = x_j^T r - lambda2 (K beta)_j of the T-CDIENET /
 *   LARS-IEN solvers.
 *
 *   **Well-posedness.**  Unlike the pure Tikhonov backbone of
 *   `tikhonov_cv_svd` — which for the rank-M group penalty is singular
 *   whenever p > n_train + M and whose prediction-CV curve is constant in
 *   lambda2 once p - M >= n_train — every point of the lambda1 path here has
 *   lambda1 > 0, so the estimator is well-defined in any p/n regime.  This is
 *   the recommended lambda2 prediction-CV tuner for the T-Rex p > n setting.
 *
 *   **CV wrapper.**  `ienet_cv_ccd` transplants the speed mechanisms of
 *   `enet_cv_ccd` / cv.glmnet onto the 2D (lambda2 x lambda1) scan:
 *
 *   1. GRID-ONCE: a warm-started full-data reference scan over the ascending
 *      lambda2 grid truncates each descending lambda1 path with glmnet's
 *      fdev/devmax/dfmax rules and fixes the informative lambda1 prefix per
 *      lambda2; every fold then evaluates exactly that (ragged) grid.
 *   2. WARM STARTS across lambda2: each (lambda2, lambda1_l) fit starts from
 *      the previous lambda2's solution at the same lambda1 — the outer-grid
 *      analog of glmnet's pathwise warm starts (the CD fixed point is
 *      unchanged; only the iteration count shrinks).
 *   3. TRUNCATION: folds never fit the deviance-saturated lambda1 tail; the
 *      held-out MSE surface is constant-extended there (duplicating the last
 *      computed value, which cannot create a spurious minimum).
 *   4. SUPPORT CAP (glmnet `dfmax`): the path additionally stops once the
 *      support exceeds `dfmax_ratio * n_train`.  This is the decisive
 *      mechanism for the IEN geometry and has no counterpart in the
 *      ridge-only EN tuner.  Past nnz ~ n_train the active-set system is
 *      rank-deficient; large lambda2 blocks the group-mean explanation and
 *      forces a dense within-group CONTRAST fit, which lives in the
 *      near-null space of X_A^T X_A + lambda2 K_AA.  Cyclic CD then zigzags
 *      through flat valleys for tens of thousands of (legitimately
 *      converging, `max_iter` never hit) sweeps per path point.  Measured on
 *      a design with n = 200, p = 500, three rho = 0.7 blocks of 50 + 350
 *      white singletons, one cold lambda1 path at
 *      lambda2 = c_max costs 2.38 M sweeps / 64 s, of which 98.2 % fall past
 *      the cap frontier — while the held-out MSE minimum sits nine grid
 *      points BEFORE it and rises monotonically thereafter.  The capped tail
 *      is overfit territory the CV profile cannot select;
 *      `n_profile_at_edge()` reports any lambda2 whose profiled optimum did
 *      land on a truncation edge.
 *
 *   Held-out MSE is scored at every (lambda2, lambda1) of the fixed grid and
 *   PROFILED over lambda1:
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

// utils includes
#include <utils/openmp/utils_eigen_threads.hpp>
#include <Eigen/SparseCore>

// Model-selection includes
#include <ml_methods/model_selection/lambda2_cv_diagnostics.hpp>

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

    /** @brief Set the thread budget of the Eigen products of fit() (>= 1);
     *  the library's default budget unless set. */
    void set_num_threads(int n_threads) {
        if (n_threads < 1) {
            throw std::invalid_argument("ien_gaussian::set_num_threads: n_threads must be >= 1.");
        }
        n_threads_ = n_threads;
    }

    /** @brief The thread budget of the Eigen products of fit(). */
    int num_threads() const noexcept { return n_threads_; }

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
     * @param fdev         glmnet-style path truncation: stop once the
     *                     fractional deviance gain between consecutive
     *                     lambda1 points drops below `fdev` (0 = disabled,
     *                     the default; glmnet's factory value is 1e-5).
     * @param devmax       Companion truncation rule: stop once the fraction
     *                     of null deviance explained exceeds `devmax`
     *                     (1 = disabled, the default; glmnet uses 0.999).
     *                     Truncated path columns are constant-extended with
     *                     the last computed solution; `n_computed()` reports
     *                     the number of genuinely fitted points.
     * @param dfmax        glmnet-style SUPPORT cap: stop once the number of
     *                     nonzero coefficients exceeds `dfmax` (negative =
     *                     disabled, the default).  The trigger point itself
     *                     is retained, so the fitted prefix ends at the
     *                     first path point with nnz > dfmax.  Same
     *                     constant-extension semantics as fdev/devmax.
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             double lambda2,
             const Eigen::VectorXi& groups,
             Eigen::Ref<const Eigen::VectorXd> lambda1_grid,
             int          max_iter = 100000,
             double       tol      = 1e-7,
             double       fdev     = 0.0,
             double       devmax   = 1.0,
             Eigen::Index dfmax    = -1)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        validate(X, y, lambda2, lambda1_grid);
        setup_group_penalty(groups, X.cols());
        preprocess(X, y);
        solve(lambda2, lambda1_grid, max_iter, tol, fdev, devmax, dfmax);
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
             int          max_iter = 100000,
             double       tol      = 1e-7,
             double       fdev     = 0.0,
             double       devmax   = 1.0,
             Eigen::Index dfmax    = -1)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        validate(X, y, lambda2, lambda1_grid);
        setup_sparse_penalty(K, X.cols());
        preprocess(X, y);
        solve(lambda2, lambda1_grid, max_iter, tol, fdev, devmax, dfmax);
    }

    /**
     * @brief Provide a warm-start path for the NEXT fit() call (consumed
     *        once, then cleared).
     *
     * @details `warm` must be on the STANDARDIZED/solver scale (a previous
     *  fit's `coef_std()` on the same data, e.g. at a neighboring lambda2)
     *  with p rows; path point l is initialized from column min(l, cols-1),
     *  so a shorter (truncated) warm path extends with its last column.
     *  The matrix must stay alive through the fit() call. The CD fixed
     *  point is unchanged — warm starts only shorten the iteration.
     */
    void set_warm_start(const Eigen::MatrixXd* warm) noexcept { warm_ = warm; }

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

    /** @brief Fraction of null deviance explained per lambda1 point (full
     *         grid length; constant past a truncation). */
    const Eigen::VectorXd& dev_ratios() const noexcept { return dev_ratio_; }

    /** @brief CD sweeps spent per lambda1 point (outer + inner active-set;
     *         diagnostic, zero past a truncation). */
    const Eigen::VectorXi& sweeps_per_point() const noexcept {
        return sweeps_per_point_;
    }

    /** @brief Total CD sweeps of the last fit. */
    long n_sweeps() const noexcept { return n_sweeps_total_; }

    /** @brief Number of lambda1 points genuinely fitted (== grid size unless
     *         the fdev/devmax/dfmax rules truncated the path). */
    Eigen::Index n_computed() const noexcept { return n_computed_; }

    /** @brief True if the fitted prefix ended on the `dfmax` support cap
     *         rather than on a deviance rule or the end of the grid. */
    bool stopped_on_dfmax() const noexcept { return stopped_on_dfmax_; }

    /** @brief Predictions for new data: (X_new * beta).colwise() + a0. */
    Eigen::MatrixXd predict(Eigen::Ref<const Eigen::MatrixXd> X_new) const {
        return (X_new * beta_).rowwise() + a0_.transpose();
    }

private:

    /** @brief Thread budget of the Eigen products of fit() (>= 1). */
    int n_threads_{::trex::utils::openmp::get_num_threads()};

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
               int max_iter, double tol, double fdev, double devmax,
               Eigen::Index dfmax)
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
        dev_ratio_.setZero(K1);
        sweeps_per_point_.setZero(K1);
        n_sweeps_total_ = 0;
        n_computed_ = K1;
        stopped_on_dfmax_ = false;
        const bool allow_early_stop = (fdev > 0.0) || (devmax < 1.0);
        const bool allow_dfmax_stop = (dfmax >= 0);
        double rsq_prev = 0.0;
        if (warm_ != nullptr && (warm_->rows() != p_ || warm_->cols() == 0))
            throw std::invalid_argument(
                "ien_gaussian: warm-start matrix must be p x k (k >= 1) on "
                "the standardized/solver scale");

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

        // NOTE (negative result): an exact line search along each
        // group's 1_m direction after every sweep — intended to remove the
        // stiff group-mean mode of the block Hessian d*I + (lambda2/p_m)*11^T
        // — was implemented and measured to fire ZERO times on the
        // scenario-B design at any lambda2.  Cyclic CD does not choke on
        // STIFF directions here, it chokes on FLAT ones (the within-group
        // contrast space in the near-null space of X_A^T X_A + lambda2 K_AA
        // once nnz > n_train); a singleton-group control probe is fast even
        // at lambda2 = c_max.  The acceleration was removed again; the
        // `dfmax` support cap addresses the actual mechanism.  Archived under
        // TRex_Research/ienet_cv_ccd_speedup_2026-08-05/.

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

        // Inner-loop support buffer, hoisted out of the path loop: the
        // active set is rebuilt every outer sweep, but the allocation is not.
        std::vector<Eigen::Index> active;
        active.reserve(static_cast<std::size_t>(p_));

        const double inv_yscale = 1.0 / y_scale_;
        for (Eigen::Index l = 0; l < K1; ++l) {
            const double l1 = lambda1s_(l) * inv_yscale;  // internal scale

            // Warm start: initialize this path point from the caller-provided
            // neighboring-lambda2 solution at the SAME lambda1 (column
            // min(l, cols-1)); residual and penalty state are rebuilt to
            // match. The fixed point is unchanged — only the iteration count
            // shrinks.
            if (warm_ != nullptr) {
                const Eigen::Index wl = std::min(l, warm_->cols() - 1);
                beta = warm_->col(wl) * inv_yscale;
                // Residual rebuilt via level-1 column ops, NOT a GEMV: with
                // EIGEN_USE_BLAS a threaded Accelerate gemv is not
                // run-to-run deterministic at the ULP level, and this feeds
                // the CD iteration (breaks the same-seed => identical-curves
                // contract of the CV wrapper).
                r = yc_;
                for (Eigen::Index j = 0; j < p_; ++j)
                    if (beta(j) != 0.0) r.noalias() -= Xs_.col(j) * beta(j);
                if (use_group_) {
                    std::fill(sigma.begin(), sigma.end(), 0.0);
                    for (Eigen::Index j = 0; j < p_; ++j)
                        if (beta(j) != 0.0)
                            sigma[static_cast<std::size_t>(group_of_[
                                static_cast<std::size_t>(j)])] += beta(j);
                } else {
                    Kbeta.noalias() = K_ * beta;
                }
            }

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
                active.clear();
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
            sweeps_per_point_(l) = it;
            n_sweeps_total_     += it;

            beta_std_.col(l) = beta * y_scale_;   // solver (unscaled-y) scale

            // glmnet path truncation (only when fdev/devmax/dfmax are armed).
            // Null deviance == n (yc_ has unit population SD). Truncated
            // columns are constant-extended with this solution.
            const double rsq =
                1.0 - r.squaredNorm() / static_cast<double>(n_);
            dev_ratio_(l) = rsq;
            const bool dev_stop = allow_early_stop && l > 0 &&
                (rsq > devmax || (rsq - rsq_prev) < fdev * rsq);
            const bool df_stop  = allow_dfmax_stop &&
                (beta.array() != 0.0).count() > dfmax;
            if (dev_stop || df_stop) {
                n_computed_       = l + 1;
                stopped_on_dfmax_ = df_stop && !dev_stop;
                for (Eigen::Index ll = l + 1; ll < K1; ++ll) {
                    beta_std_.col(ll) = beta_std_.col(l);
                    dev_ratio_(ll)    = rsq;
                }
                break;
            }
            rsq_prev = rsq;
        }

        warm_ = nullptr;   // consumed by this fit

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
    Eigen::VectorXd dev_ratio_;  ///< deviance ratio per lambda1 point
    Eigen::VectorXi sweeps_per_point_;  ///< CD sweeps per lambda1 point
    Eigen::Index    n_computed_ = 0;  ///< genuinely fitted path length
    long            n_sweeps_total_   = 0;
    int             n_nonconverged_   = 0;
    bool            stopped_on_dfmax_ = false;

    /// Optional warm-start path (standardized scale) for the next fit();
    /// non-owning, consumed once. See set_warm_start().
    const Eigen::MatrixXd* warm_ = nullptr;
};

// ===================================================================================

/**
 * @brief K-fold cross-validated lambda2 selection for the Informed Elastic
 *        Net via the CCD `ien_gaussian` engine: per lambda2 candidate the
 *        full lambda1 path is fitted per fold and profiled out.
 *
 *  References:
 *  -----------
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped Variable Selection
 *     and FDR Control in Genomics Research," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt.
 *     Process. (CAMSAP), 2023, pp. 466-470.
 *   J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
 *     Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
 *   R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor, and R. J. Tibshirani, "Strong Rules
 *     for Discarding Predictors in Lasso-Type Problems," J. R. Stat. Soc. Ser. B Stat. Methodol., vol. 74,
 *     no. 2, pp. 245-266, 2012, doi: 10.1111/j.1467-9868.2011.01004.x.
 *   L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression Trees.
 *     Wadsworth, Belmont, CA, 1984.
 */
class ienet_cv_ccd {

public:

    /** @brief Default constructor. */
    ienet_cv_ccd() = default;

    /** @brief Set the thread budget of the Eigen products of fit() (>= 1);
     *  the library's default budget unless set. The fold fits inherit it. */
    void set_num_threads(int n_threads) {
        if (n_threads < 1) {
            throw std::invalid_argument("ienet_cv_ccd::set_num_threads: n_threads must be >= 1.");
        }
        n_threads_ = n_threads;
    }

    /** @brief The thread budget of the Eigen products of fit(). */
    int num_threads() const noexcept { return n_threads_; }

    /**
     * @brief Fit K-fold CV with the group-mean penalty (TIENET geometry).
     *
     * @param X                 Predictor matrix (n x p).
     * @param y                 Response vector (n).
     * @param groups            0-based contiguous group ids, length p.
     * @param n_folds           Number of folds (default 10).
     * @param n_lambda2         lambda2 grid size (default 100 — resolution
     *                          parity with the lambda1 path and glmnet's
     *                          nlambda; the scan is warm-started across
     *                          lambda2, so cost grows sublinearly).
     * @param lambda2_ratio     Grid spans `[l2_max / ratio, l2_max]` with
     *                          l2_max = c_max = max_j |x_{s,j}^T y_c|
     *                          (default 1e4).
     * @param n_lambda1         lambda1 path length (default 40); per
     *                          lambda2 the fitted prefix is decided by the
     *                          full-data reference scan (fdev/devmax/dfmax).
     *                          lambda1 is a NUISANCE parameter here — it is
     *                          profiled out and only has to keep the
     *                          contrast space well-posed — so it does not
     *                          need glmnet's nlambda = 100 resolution.  Cost
     *                          in this parameter is U-SHAPED, not monotone:
     *                          a coarser grid means fewer path points but
     *                          larger lambda1 steps, which degrade the
     *                          pathwise warm starts faster than the point
     *                          count falls.  Measured at B scale
     *                          (n_lambda2 = 25): 100 -> 18.8 s, 60 -> 13.9 s,
     *                          40 -> 10.1 s, 25 -> 7.2 s, 15 -> 8.3 s,
     *                          10 -> 9.9 s.  40 is the smallest value that
     *                          still reproduces the lambda2 picks of the
     *                          100-point grid exactly; below it the profiled
     *                          cv_min starts shifting by a grid point.
     * @param lambda1_min_ratio lambda1_min / lambda1_max.  Negative (default)
     *                          auto-selects 1e-2 in BOTH dimensional
     *                          regimes — deliberately not glmnet's
     *                          (1e-2 if n < p else 1e-4) rule, because
     *                          lambda1 is profiled out here rather than
     *                          reported, so the deep tail only spends grid
     *                          points where the CV profile never looks. In
     *                          the low-dimensional regime
     *                          (`is_low_dimensional()`) that matters most:
     *                          the support cap is structurally inert there,
     *                          so nothing else truncates the tail.  See the
     *                          measurement table at the rule's use site.
     * @param seed              Fold-permutation seed (default 0).
     * @param max_iter          CD sweeps per lambda1 (default 100000).
     * @param tol               CD tolerance of the tuner's solves (default
     *                          1e-4; the pathwise fit() keeps 1e-7). The
     *                          wrapper consumes only predict(), and the
     *                          criterion sqrt(d_j)*|delta_j| already
     *                          measures the change in FITTED VALUES, so the
     *                          lambda2 pick needs far less precision than a
     *                          reported path. Measured at the shipped grid
     *                          (n_lambda2 = 100, n_lambda1 = 40, dfmax 0.8)
     *                          on two designs (n=150/p=100 and n=200/p=500
     *                          Hastie blocks): both picks bit-identical from
     *                          1e-7 down to 1e-4, the CV curve within 1e-4
     *                          relative, 5-6x faster (26.7 s -> 5.3 s and
     *                          34.8 s -> 5.6 s per call). In this file's
     *                          unit-L2 column scaling glmnet's own criterion
     *                          is |delta| < sqrt(n)*3.16e-4 (~4.5e-3 at
     *                          n = 200): 1e-4 is still ~45x stricter.
     * @param dfmax_ratio       Support cap as a fraction of the (smallest)
     *                          fold training size: the reference scan stops
     *                          each lambda1 path at the first point with
     *                          nnz > dfmax_ratio * n_train (default 0.8;
     *                          <= 0 disables the cap).  See mechanism 4 in
     *                          the file header — this is the dominant speed
     *                          mechanism of the tuner.
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             const Eigen::VectorXi& groups,
             int          n_folds           = 10,
             Eigen::Index n_lambda2         = 100,
             double       lambda2_ratio     = 1e4,
             Eigen::Index n_lambda1         = 40,
             double       lambda1_min_ratio = -1.0,
             unsigned int seed              = 0,
             int          max_iter          = 100000,
             double       tol               = 1e-4,
             double       dfmax_ratio       = 0.8)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        groups_ = groups;
        use_group_ = true;
        run_cv(X, y, n_folds, n_lambda2, lambda2_ratio, n_lambda1,
               lambda1_min_ratio, seed, max_iter, tol, dfmax_ratio);
    }

    /**
     * @brief Fit K-fold CV with a general sparse PSD Tikhonov matrix K.
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             const Eigen::SparseMatrix<double>& K,
             int          n_folds           = 10,
             Eigen::Index n_lambda2         = 100,
             double       lambda2_ratio     = 1e4,
             Eigen::Index n_lambda1         = 40,
             double       lambda1_min_ratio = -1.0,
             unsigned int seed              = 0,
             int          max_iter          = 100000,
             double       tol               = 1e-4,
             double       dfmax_ratio       = 0.8)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        K_ = K;
        use_group_ = false;
        run_cv(X, y, n_folds, n_lambda2, lambda2_ratio, n_lambda1,
               lambda1_min_ratio, seed, max_iter, tol, dfmax_ratio);
    }

    // public member accessors ------------------------------------------------

    /** @brief lambda2 grid that was searched (ascending, solver scale). */
    const Eigen::VectorXd& lambdas() const noexcept { return lambdas_; }

    /** @brief Profiled mean CV-MSE per lambda2 (min over the lambda1 path). */
    const Eigen::VectorXd& cv_mse() const noexcept { return cv_mse_; }

    /** @brief SEM of the CV-MSE mean at the profiled lambda1, per lambda2. */
    const Eigen::VectorXd& cv_sem() const noexcept { return cv_sem_; }

    /** @brief lambda2 minimising the profiled CV-MSE (`lambda.min`),
     *         directly consumable by TIENET_Solver / TCDIENET_Solver. */
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

    /** @brief Genuinely fitted lambda1 prefix length per lambda2, as decided
     *         once by the full-data reference scan (fdev/devmax/dfmax).
     *         Entries below `lambda1s().size()` mark a truncated path whose
     *         MSE row is constant-extended. */
    const Eigen::VectorXi& path_lengths() const noexcept {
        return path_lengths_;
    }

    /** @brief True when the run was LOW-DIMENSIONAL, i.e. the support cap
     *         could not possibly bind because p <= dfmax (or the cap was
     *         disabled outright).
     *
     * @details The companion of `tikhonov_cv_svd::contrast_interpolates()`,
     *  which flags the opposite boundary. In this regime nnz <= p < the cap
     *  at every path point, so mechanism 4 (the support cap) contributes
     *  nothing and the deviance rules carry the truncation alone. Not an
     *  error — the CV curve is fully valid — but it explains why the tuner
     *  costs what it costs on small-p problems, and it is the reason the
     *  auto lambda1 range does not follow glmnet's n-vs-p rule. */
    bool is_low_dimensional() const noexcept { return low_dimensional_; }

    /** @brief The support cap actually applied (glmnet dfmax), or -1 when
     *         disabled. Compare against `lambda1s().size()` and p to see
     *         whether it could bind; see `is_low_dimensional()`. */
    Eigen::Index support_cap() const noexcept { return support_cap_; }

    /** @brief Number of lambda2 values whose profiled lambda1 optimum landed
     *         on the last computed path point.
     *
     * @details Zero is the healthy case: every profiled optimum was interior,
     *  so the truncation rules only removed lambda1 values the CV would not
     *  have selected anyway. A nonzero count means the support cap (or a
     *  deviance rule, or the end of the lambda1 grid) was binding for that
     *  many lambda2 values — the tuner's answer there is a boundary solution.
     *  Raise `dfmax_ratio` or lower `lambda1_min_ratio` if it is not zero. */
    Eigen::Index n_profile_at_edge() const noexcept {
        return n_profile_at_edge_;
    }

    /** @brief True when the profiled lambda1 optimum of lambda2 grid point
     *         @p k2 landed on the last computed path point (one of the
     *         `n_profile_at_edge()` cases). */
    bool profile_at_edge(Eigen::Index k2) const {
        check_fitted();
        return profile_at_edge_[static_cast<std::size_t>(k2)] != 0;
    }

    /** @brief True when `cv_min()` or `cv_1se()` landed on the FIRST or LAST
     *         lambda2 grid point — the pick is the grid boundary, not an
     *         optimum (parity with `enet_cv_ccd` / `ridge_cv_svd` /
     *         `tikhonov_cv_svd`). A flat profiled curve (low SNR) lands
     *         here with `n_profile_at_edge() == 0`, which is why the guard
     *         (`diagnostics()`) reads both. */
    bool selection_at_grid_edge() const noexcept {
        const Eigen::Index K = lambdas_.size();
        if (K == 0) { return false; }
        return idx_min_ == 0 || idx_min_ == K - 1 ||
               idx_1se_ == 0 || idx_1se_ == K - 1;
    }

    /** @brief The structural half of the tuner health record for the pick
     *         the caller uses; `seconds` is the caller's to fill (this
     *         class does not time itself).
     *
     * @param one_se True when the caller returns `cv_1se()`, false for
     *        `cv_min()`: the grid-pick fields (`grid_index`, `at_grid_edge`,
     *        `profile_at_edge`, `profile_at_support_cap`, `boundary_pick`)
     *        describe THAT pick. */
    lambda2_cv_diagnostics diagnostics(bool one_se) const {
        check_fitted();
        lambda2_cv_diagnostics d;
        d.ran               = true;
        d.n_profile_at_edge = n_profile_at_edge_;
        d.support_cap       = support_cap_;
        d.dfmax_ratio       = dfmax_ratio_;
        d.low_dimensional   = low_dimensional_;
        const Eigen::Index idx = one_se ? idx_1se_ : idx_min_;
        d.profile_at_edge   = profile_at_edge(idx);
        d.profile_at_support_cap =
            d.profile_at_edge &&
            stopped_on_cap_[static_cast<std::size_t>(idx)] != 0;
        record_grid_pick(d, idx, lambdas_.size());
        return d;
    }

private:

    /** @brief Thread budget of the Eigen products of fit() and the fold fits (>= 1). */
    int n_threads_{::trex::utils::openmp::get_num_threads()};

    void run_cv(Eigen::Ref<const Eigen::MatrixXd> X,
                Eigen::Ref<const Eigen::VectorXd> y,
                int n_folds, Eigen::Index n_lambda2, double lambda2_ratio,
                Eigen::Index n_lambda1, double lambda1_min_ratio,
                unsigned int seed, int max_iter, double tol,
                double dfmax_ratio)
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
        // ---- support cap (glmnet dfmax) + dimensional regime ---------------
        // The cap is sized on the SMALLEST fold training set, not on the full
        // n: the reference scan only decides path lengths, but those lengths
        // are consumed by the folds, whose active-set systems become
        // rank-deficient at their own n_train. Round-robin assignment gives a
        // largest test fold of ceil(n / n_folds).
        const Eigen::Index n_test_max  = (n + n_folds - 1) / n_folds;
        const Eigen::Index n_train_min = n - n_test_max;
        const Eigen::Index dfmax =
            (dfmax_ratio > 0.0)
                ? std::max<Eigen::Index>(
                      1, static_cast<Eigen::Index>(
                             dfmax_ratio * static_cast<double>(n_train_min)))
                : Eigen::Index{-1};

        // LOW-DIMENSIONAL regime: the support can never exceed p, so if
        // p <= dfmax the cap is structurally INERT and the path has no
        // adaptive truncation from it at all (only the deviance rules).
        // Mirrors tikhonov_cv_svd::contrast_interpolates(), which reports the
        // opposite boundary; reported here by is_low_dimensional().
        low_dimensional_ = (dfmax < 0) || (p <= dfmax);
        support_cap_     = dfmax;
        dfmax_ratio_     = dfmax_ratio;

        // lambda1 range: DELIBERATELY NOT glmnet's rule (n < p ? 1e-2 : 1e-4).
        // glmnet descends to 1e-4 when n > p because there the nearly
        // unpenalized end is a meaningful fit worth reporting. This class does
        // not report the lambda1 path — it PROFILES lambda1 out to select
        // lambda2 — so the only thing the deep tail can do is consume grid
        // points in a region the CV profile never selects. In the low-
        // dimensional regime that is doubly wasteful because the support cap
        // is inert there, so nothing truncates the tail either.
        //
        // Measured (scenario-A shape n=150, p=100, rho=0.75, K2=100, K1=40;
        // reference = the same run at K1=100 with the deep range, lambda2_1se
        // = 1.0992):
        //     1e-4 (glmnet)  21.3 s   lambda2_1se 1.2064  <- MISSES the ref
        //     1e-3           16.9 s   lambda2_1se 1.0992
        //     1e-2            9.5 s   lambda2_1se 1.0992  <- shipped
        //     3e-2            5.5 s   lambda2_1se 1.0992
        // The deep range is not merely slower, it is LESS accurate at fixed
        // grid size: the 40 points land where the profile is flat instead of
        // where it turns. n_profile_at_edge() stayed 0 throughout, i.e. the
        // truncation never bound on the answer — that accessor is the guard
        // if a future design does push the range too far.
        const double l1mr = (lambda1_min_ratio > 0.0)
                          ? lambda1_min_ratio
                          : kLambda1MinRatio;

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

        // ---- full-data reference scan (cv.glmnet grid-once semantics) ------
        // One warm-started pass over the ASCENDING lambda2 grid on the full
        // data, truncating each lambda1 path with glmnet's fdev/devmax/dfmax
        // rules: this decides, once, the informative lambda1 prefix per
        // lambda2. The folds then evaluate exactly this fixed (ragged) grid
        // — no per-fold stopping decisions, full fold comparability.
        path_lengths_.resize(K2);
        std::vector<Eigen::Index> k1_len(static_cast<std::size_t>(K2), K1);
        stopped_on_cap_.assign(static_cast<std::size_t>(K2), 0);
        {
            Eigen::MatrixXd warm_ref;
            for (Eigen::Index k2 = 0; k2 < K2; ++k2) {
                ien_gaussian ref;
                ref.set_num_threads(n_threads_);
                if (k2 > 0) ref.set_warm_start(&warm_ref);
                if (use_group_)
                    ref.fit(X, y, lambdas_(k2), groups_, lambda1s_,
                            max_iter, tol, kPathFdev, kPathDevMax, dfmax);
                else
                    ref.fit(X, y, lambdas_(k2), K_, lambda1s_,
                            max_iter, tol, kPathFdev, kPathDevMax, dfmax);
                k1_len[static_cast<std::size_t>(k2)] = ref.n_computed();
                path_lengths_(k2) = static_cast<int>(ref.n_computed());
                stopped_on_cap_[static_cast<std::size_t>(k2)] =
                    ref.stopped_on_dfmax() ? 1 : 0;
                warm_ref = ref.coef_std();
            }
        }

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

            // Warm-started ascending-lambda2 scan over the reference-fixed
            // ragged grid. Beyond a lambda2's truncated lambda1 prefix the
            // MSE row is constant-extended (duplicates the last computed
            // value — cannot create a spurious row minimum), keeping the
            // rectangular surface contract intact.
            Eigen::MatrixXd mse(K2, K1);
            Eigen::MatrixXd warm_fold;
            for (Eigen::Index k2 = 0; k2 < K2; ++k2) {
                const Eigen::Index len =
                    k1_len[static_cast<std::size_t>(k2)];
                ien_gaussian en;
                en.set_num_threads(n_threads_);
                if (k2 > 0) en.set_warm_start(&warm_fold);
                if (use_group_)
                    en.fit(X_tr, y_tr, lambdas_(k2), groups_,
                           lambda1s_.head(len), max_iter, tol);
                else
                    en.fit(X_tr, y_tr, lambdas_(k2), K_,
                           lambda1s_.head(len), max_iter, tol);
                warm_fold = en.coef_std();
                const Eigen::MatrixXd preds = en.predict(X_te); // n_test x len
                const Eigen::RowVectorXd row =
                    (preds.colwise() - y_te).colwise().squaredNorm()
                    / static_cast<double>(n_test);
                mse.row(k2).head(len) = row;
                if (len < K1)
                    mse.row(k2).tail(K1 - len).setConstant(row(len - 1));
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
        // The row minimum is taken over the FULL grid, but past a truncation
        // the row is constant, so the argmin can only land on the last
        // COMPUTED point (index len - 1) — never deeper. A profiled optimum
        // sitting exactly there means the truncation was binding for that
        // lambda2 and the tuner would have preferred a smaller lambda1;
        // n_profile_at_edge() surfaces that instead of hiding it.
        cv_mse_.resize(K2);
        cv_sem_.resize(K2);
        lambda1_profile_.resize(K2);
        profile_at_edge_.assign(static_cast<std::size_t>(K2), 0);
        n_profile_at_edge_ = 0;
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
            if (k1_best == k1_len[static_cast<std::size_t>(k2)] - 1) {
                profile_at_edge_[static_cast<std::size_t>(k2)] = 1;
                ++n_profile_at_edge_;
            }
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

    // glmnet factory truncation constants for the full-data reference scan
    // (glmnet.control: fdev = 1e-5, devmax = 0.999). Folds never truncate —
    // they inherit the reference scan's per-lambda2 path lengths.
    static constexpr double kPathFdev   = 1.0e-5;
    static constexpr double kPathDevMax = 0.999;

    /// Auto lambda1_min_ratio for BOTH dimensional regimes — see the
    /// derivation at its use site in run_cv(). Not glmnet's n-vs-p rule:
    /// lambda1 is profiled out here, so the deep tail only wastes grid
    /// points (and in the low-dimensional regime nothing truncates it,
    /// because the support cap is inert once p <= dfmax).
    static constexpr double kLambda1MinRatio = 1.0e-2;

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
    Eigen::VectorXi path_lengths_;     ///< fitted lambda1 prefix per lambda2
    Eigen::Index    idx_min_ = 0;
    Eigen::Index    idx_1se_ = 0;
    Eigen::Index    n_profile_at_edge_ = 0;
    std::vector<char> profile_at_edge_; ///< per lambda2: profiled optimum on the truncation edge
    std::vector<char> stopped_on_cap_;  ///< per lambda2: reference path stopped on the support cap
    Eigen::Index    support_cap_       = -1;
    double          dfmax_ratio_       = 0.0;   ///< ratio passed to fit()
    bool            low_dimensional_   = false;
};

// ===================================================================================
} /* End of namespace model_selection */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_IENET_CV_CCD_HPP */
