// ===================================================================================
// enet_cv_ccd.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_ENET_CV_CCD_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_ENET_CV_CCD_HPP
// ===================================================================================
/**
 * @file enet_cv_ccd.hpp
 *
 * @brief Cyclic coordinate-descent (CCD) elastic-net solver for the Gaussian
 *        family, replicating R `glmnet`'s algorithm and conventions, plus a
 *        K-fold cross-validation wrapper that reproduces `glmnet::cv.glmnet`.
 *
 * @details
 *   Unlike `ridge_cv_svd` — which solves the ridge problem analytically via the
 *   SVD — this file implements the *same* pathwise cyclic coordinate-descent
 *   algorithm that `glmnet` uses internally.
 *
 *   **Performance.**  Tibshirani's sequential strong rule prunes the active set;
 *   the inner CD sweeps only active coordinates and a KKT pass re-admits any
 *   violated predictor.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <utility>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Core>

// utils includes
#include <utils/openmp/utils_eigen_threads.hpp>

// ===================================================================================

// Namespace embedding: trex::ml_methods::model_selection
namespace trex {
namespace ml_methods {
namespace model_selection {

// ===================================================================================

/**
 * @brief Cyclic coordinate-descent elastic-net path solver (Gaussian family).
 *
 * @details Replicates `glmnet`'s pathwise coordinate descent.  Use `fit()` to
 *  solve over an automatically generated glmnet-style @f$\lambda@f$ grid, or the
 *  explicit-grid `fit()` overload to solve at a caller-supplied sequence (e.g.
 *  `glmnet$lambda`) for a coefficient-by-coefficient comparison.
 */
class enet_gaussian {

public:

    /** @brief Default constructor: the copying engine. */
    enet_gaussian() = default;

    /**
     * @brief Choose the engine.
     *
     * @param low_memory false (default): the coordinate descent runs on a
     *        centred and standardised working copy of X (one n x p matrix
     *        while fitting; bit-identical to every earlier release). true:
     *        the descent runs on the raw X through the caller's view — a
     *        memory-mapped file included — with the centring carried as a
     *        scalar residual offset and the standardisation as a per-column
     *        scale; no allocation of the size of X. Same estimator, same
     *        grid; the two engines agree to rounding (about 1e-12 in the
     *        coefficients), so a CV pick can differ only on a flat curve.
     */
    explicit enet_gaussian(bool low_memory) : low_memory_(low_memory) {}

    /** @brief Set the thread budget of the Eigen products of fit() (>= 1);
     *  the library's default budget unless set. */
    void set_num_threads(int n_threads) {
        if (n_threads < 1) {
            throw std::invalid_argument("enet_gaussian::set_num_threads: n_threads must be >= 1.");
        }
        n_threads_ = n_threads;
    }

    /** @brief The thread budget of the Eigen products of fit(). */
    int num_threads() const noexcept { return n_threads_; }

    /** @brief Switch the engine (see the constructor); takes effect at the next fit(). */
    void set_low_memory(bool low_memory) noexcept { low_memory_ = low_memory; }

    /** @brief Whether the raw-X (no working copy) engine is selected. */
    bool low_memory() const noexcept { return low_memory_; }

    /**
     * @brief Fit the elastic-net path over an auto-generated glmnet-style grid.
     *
     * @param X                Predictor matrix (n x p).
     * @param y                Response vector (n).
     * @param alpha            Elastic-net mixing in [0,1]; 1 = lasso, 0 = ridge.
     * @param n_lambda         Number of grid points (default 100).
     * @param lambda_min_ratio @f$\lambda_{\min}/\lambda_{\max}@f$.  Pass a
     *                         negative value (default) to auto-select glmnet's
     *                         convention (1e-2 if n<p else 1e-4).
     * @param standardize      Standardise columns to unit population SD
     *                         (glmnet `standardize=TRUE`; default true).
     * @param intercept        Fit an unpenalised intercept by centering X and y
     *                         (glmnet `intercept=TRUE`; default true).
     * @param use_strong_rule  Enable the sequential strong rule (default false;
     *                         glmnet does not use it).
     * @param max_iter         Max CD sweeps per @f$\lambda@f$ (default 100000).
     * @param tol              Convergence tolerance on the max coordinate change
     *                         in standardised units (default 1e-7).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             double       alpha            = 1.0,
             Eigen::Index n_lambda         = 100,
             double       lambda_min_ratio = -1.0,
             bool         standardize      = true,
             bool         intercept        = true,
             bool         use_strong_rule  = false,
             int          max_iter         = 100000,
             double       tol              = 1e-7)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        validate_inputs(X, y, alpha);
        if (low_memory_) {
            fit_raw_(X, y, /*mask=*/nullptr, /*lambda_grid=*/nullptr, alpha, n_lambda,
                     lambda_min_ratio, standardize, intercept, use_strong_rule,
                     max_iter, tol, /*allow_early_stop=*/true);
            return;
        }
        preprocess(X, y, standardize, intercept);

        const double alpha_eff = (alpha > 1e-3) ? alpha : 1e-3;
        // yc_ is standardised (unit population SD); report lambda on glmnet's
        // original scale, i.e. internal lambda * y_scale_.
        const double lambda_max =
            (Xs_.transpose() * yc_).cwiseAbs().maxCoeff()
            / (static_cast<double>(n_) * alpha_eff) * y_scale_;

        // Guard the grid anchor: a constant y (or y orthogonal to X) gives
        // lambda_max == 0, and log(0) would poison the whole grid with NaNs.
        if (!(lambda_max > 1e-12) || !std::isfinite(lambda_max)) {
            throw std::runtime_error(
                "enet_gaussian::fit: lambda grid anchor is degenerate "
                "(y is constant or (nearly) orthogonal to X).");
        }

        const double lmr = (lambda_min_ratio > 0.0)
                         ? lambda_min_ratio
                         : (n_ < p_ ? 1e-2 : 1e-4);

        lambdas_ = make_log_grid(lambda_max, lambda_max * lmr, n_lambda);
        solve(alpha, use_strong_rule, max_iter, tol, /*allow_early_stop=*/true);
        release_working_set();
    }

    /**
     * @brief Fit the elastic-net path at a caller-supplied @f$\lambda@f$ grid.
     *
     * @details The grid is sorted descending internally so warm starts remain
     *  valid.  Use this overload to evaluate the C++ path at exactly
     *  `glmnet$lambda` for a direct coefficient comparison.
     *
     * @param X               Predictor matrix (n x p).
     * @param y               Response vector (n).
     * @param lambda_grid     Explicit @f$\lambda@f$ values (any order, > 0).
     * @param alpha           Elastic-net mixing in [0,1]; 1 = lasso, 0 = ridge.
     * @param standardize     Standardise columns to unit population SD.
     * @param intercept       Fit an unpenalised intercept by centering X and y.
     * @param use_strong_rule Enable the sequential strong rule.
     * @param max_iter        Max CD sweeps per @f$\lambda@f$.
     * @param tol             Convergence tolerance (standardised units).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd>  X,
             Eigen::Ref<const Eigen::VectorXd>  y,
             Eigen::Ref<const Eigen::VectorXd>  lambda_grid,
             double alpha           = 1.0,
             bool   standardize     = true,
             bool   intercept       = true,
             bool   use_strong_rule = false,
             int    max_iter        = 100000,
             double tol             = 1e-7)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        validate_inputs(X, y, alpha);
        if (lambda_grid.size() == 0) {
            throw std::invalid_argument(
                "enet_gaussian::fit: lambda_grid is empty");
        }
        if (low_memory_) {
            const Eigen::VectorXd grid = lambda_grid;
            fit_raw_(X, y, /*mask=*/nullptr, &grid, alpha, 0, -1.0,
                     standardize, intercept, use_strong_rule, max_iter, tol,
                     /*allow_early_stop=*/false);
            return;
        }
        preprocess(X, y, standardize, intercept);

        lambdas_ = lambda_grid;
        std::sort(lambdas_.data(), lambdas_.data() + lambdas_.size(),
                  std::greater<double>());
        solve(alpha, use_strong_rule, max_iter, tol, /*allow_early_stop=*/false);
        release_working_set();
    }

    /**
     * @brief Fit at a caller-supplied grid on an OWNED predictor matrix,
     *        which becomes the working copy Xs_ (centred and standardised in
     *        place): one n x p matrix instead of two. Used by the CV folds.
     *        Results are bit-identical to fit() on the same matrix.
     */
    void fit_owned(Eigen::MatrixXd&& X_owned,
                   Eigen::Ref<const Eigen::VectorXd> y,
                   Eigen::Ref<const Eigen::VectorXd> lambda_grid,
                   double alpha           = 1.0,
                   bool   standardize     = true,
                   bool   intercept       = true,
                   bool   use_strong_rule = false,
                   int    max_iter        = 100000,
                   double tol             = 1e-7)
    {
        validate_inputs(X_owned, y, alpha);
        if (lambda_grid.size() == 0) {
            throw std::invalid_argument(
                "enet_gaussian::fit_owned: lambda_grid is empty");
        }
        preprocess_owned(std::move(X_owned), y, standardize, intercept);
        lambdas_ = lambda_grid;
        std::sort(lambdas_.data(), lambdas_.data() + lambdas_.size(),
                  std::greater<double>());
        solve(alpha, use_strong_rule, max_iter, tol, /*allow_early_stop=*/false);
        release_working_set();
    }

    /**
     * @brief Fit at a caller-supplied grid on the rows selected by a 0/1 mask,
     *        with the raw-X engine: the training fold is never gathered
     *        (used by enet_cv_ccd in low-memory mode).
     *
     * @param mask  Length n, 1 for a training row, 0 for a held-out row. The
     *              column statistics, the response scaling and every dot
     *              product are taken over the training rows only.
     */
    void fit_masked(Eigen::Ref<const Eigen::MatrixXd>  X,
                    Eigen::Ref<const Eigen::VectorXd>  y,
                    const Eigen::VectorXd&             mask,
                    Eigen::Ref<const Eigen::VectorXd>  lambda_grid,
                    double alpha           = 1.0,
                    bool   standardize     = true,
                    bool   intercept       = true,
                    bool   use_strong_rule = false,
                    int    max_iter        = 100000,
                    double tol             = 1e-7)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        validate_inputs(X, y, alpha);
        if (lambda_grid.size() == 0) {
            throw std::invalid_argument(
                "enet_gaussian::fit_masked: lambda_grid is empty");
        }
        const Eigen::VectorXd grid = lambda_grid;
        fit_raw_(X, y, &mask, &grid, alpha, 0, -1.0, standardize, intercept,
                 use_strong_rule, max_iter, tol, /*allow_early_stop=*/false);
    }

    /**
     * @brief Predictions per lambda for a subset of rows of X, read row by
     *        row (no gathered copy of the rows); companion of fit_masked().
     */
    Eigen::MatrixXd predict_rows(Eigen::Ref<const Eigen::MatrixXd> X,
                                 const std::vector<Eigen::Index>& rows) const
    {
        const Eigen::Index K = beta_.cols();
        Eigen::MatrixXd out(static_cast<Eigen::Index>(rows.size()), K);
        for (std::size_t t = 0; t < rows.size(); ++t) {
            out.row(static_cast<Eigen::Index>(t)) =
                (X.row(rows[t]) * beta_) + a0_.transpose();
        }
        return out;
    }

    // public member accessors ------------------------------------------------

    /** @brief @f$\lambda@f$ grid actually used (descending). */
    const Eigen::VectorXd& lambdas() const noexcept { return lambdas_; }

    /** @brief Coefficient path in ORIGINAL predictor scale (p x n_lambda). */
    const Eigen::MatrixXd& coef() const noexcept { return beta_; }

    /** @brief Intercepts per @f$\lambda@f$ (n_lambda); zero if intercept=false. */
    const Eigen::VectorXd& intercepts() const noexcept { return a0_; }

    /** @brief Fraction of null deviance explained at each fitted @f$\lambda@f$
     *         (glmnet `%Dev`); length equals the possibly-truncated grid. */
    const Eigen::VectorXd& dev_ratio() const noexcept { return dev_ratio_; }

    /** @brief True if every @f$\lambda@f$ reached the convergence tolerance. */
    bool converged() const noexcept { return n_nonconverged_ == 0; }

    /** @brief Number of @f$\lambda@f$ points that hit `max_iter` without converging. */
    int n_nonconverged() const noexcept { return n_nonconverged_; }

    /** @brief Predictions for new data: (X_new * beta).colwise() + a0. */
    Eigen::MatrixXd predict(Eigen::Ref<const Eigen::MatrixXd> X_new) const {
        return (X_new * beta_).rowwise() + a0_.transpose();
    }

private:

    /** @brief Thread budget of the Eigen products of fit() (>= 1). */
    int n_threads_{::trex::utils::openmp::get_num_threads()};

    // ---- shared standardised-space coordinate-descent core -----------------
    //
    // Solves the full path on the already-preprocessed (Xs_, yc_) over lambdas_
    // and stores back-scaled coefficients in beta_ / a0_.
    //
    // When `allow_early_stop` is true (auto-generated grid only, never a
    // user-supplied grid) the path is truncated with glmnet's deviance rules:
    //   * devmax: stop once the fraction of null deviance explained exceeds
    //             `kDevMax` (0.999), and
    //   * fdev  : stop once the fractional increase in deviance-explained
    //             between consecutive lambdas drops below `kFdev` (1e-5).
    // These are glmnet's factory defaults (see `glmnet.control`).  cv.glmnet
    // builds its lambda sequence from exactly such an early-stopped full-data
    // fit, then evaluates every fold at that fixed sequence.
    void solve(double alpha, bool use_strong_rule, int max_iter, double tol,
               bool allow_early_stop = false)
    {
        constexpr double kFdev   = 1.0e-5;   // glmnet fdev   (fractional dev change)
        constexpr double kDevMax = 0.999;    // glmnet devmax (max dev.ratio)

        const Eigen::Index K = lambdas_.size();
        const double invN    = 1.0 / static_cast<double>(n_);
        const bool   is_ridge = (alpha == 0.0);

        // d_j = ||x_{s,j}||^2 / N  (== 1 for standardised columns; variance else)
        Eigen::VectorXd d(p_);
        for (Eigen::Index j = 0; j < p_; ++j) {
            d(j) = Xs_.col(j).squaredNorm() * invN;
        }

        Eigen::MatrixXd Bs = Eigen::MatrixXd::Zero(p_, K);  // standardised-space path
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p_);   // warm-started across lambda
        Eigen::VectorXd r    = yc_;                         // working residual (unit-SD y)
        std::vector<char> active(static_cast<std::size_t>(p_), 0);
        n_nonconverged_ = 0;

        // deviance ratio per fitted lambda; null deviance == n_ (yc_ unit-SD).
        dev_ratio_ = Eigen::VectorXd::Zero(K);
        double   rsq_prev = 0.0;
        Eigen::Index n_fit = K;

        // lambdas_ are on glmnet's reported scale; the standardised-y problem
        // uses the internal value lambda / y_scale_.
        const double inv_yscale = 1.0 / y_scale_;
        for (Eigen::Index l = 0; l < K; ++l) {
            const double lambda = lambdas_(l) * inv_yscale;
            const double l1     = lambda * alpha;
            const double l2     = lambda * (1.0 - alpha);
            const double lambda_prev =
                (l == 0) ? lambda : lambdas_(l - 1) * inv_yscale;

            // --- strong-rule active set (lasso / EN only) -------------------
            std::fill(active.begin(), active.end(), 0);
            if (use_strong_rule && !is_ridge) {
                const double thr = alpha * (2.0 * lambda - lambda_prev);
                for (Eigen::Index j = 0; j < p_; ++j) {
                    if (d(j) <= 1e-12) continue;
                    if (std::abs(beta(j)) > 0.0) { active[static_cast<std::size_t>(j)] = 1; continue; }
                    if (std::abs(Xs_.col(j).dot(r) * invN) >= thr)
                        active[static_cast<std::size_t>(j)] = 1;
                }
            } else {
                for (Eigen::Index j = 0; j < p_; ++j)
                    if (d(j) > 1e-12) active[static_cast<std::size_t>(j)] = 1;
            }

            // --- coordinate descent with KKT re-admission -------------------
            bool kkt_ok = false;
            while (!kkt_ok) {
                bool converged = false;
                for (int iter = 0; iter < max_iter; ++iter) {
                    double max_change = 0.0;
                    for (Eigen::Index j = 0; j < p_; ++j) {
                        if (!active[static_cast<std::size_t>(j)]) continue;
                        const double old_b = beta(j);
                        const double z_j   = Xs_.col(j).dot(r) * invN + d(j) * old_b;
                        const double new_b = is_ridge
                            ? z_j / (d(j) + l2)
                            : soft_threshold(z_j, l1) / (d(j) + l2);
                        const double diff  = new_b - old_b;
                        if (diff != 0.0) {
                            r.noalias() -= Xs_.col(j) * diff;
                            beta(j)      = new_b;
                            max_change   = std::max(max_change, std::abs(diff) * std::sqrt(d(j)));
                        }
                    }
                    if (max_change < tol) { converged = true; break; }
                }
                if (!converged) ++n_nonconverged_;

                if (!use_strong_rule || is_ridge) { kkt_ok = true; break; }

                kkt_ok = true;
                for (Eigen::Index j = 0; j < p_; ++j) {
                    if (active[static_cast<std::size_t>(j)] || d(j) <= 1e-12) continue;
                    if (std::abs(Xs_.col(j).dot(r) * invN) > l1 + 1e-9) {
                        active[static_cast<std::size_t>(j)] = 1;
                        kkt_ok = false;
                    }
                }
            }

            Bs.col(l) = beta;

            // --- glmnet deviance-based path early-termination ---------------
            const double rsq = 1.0 - r.squaredNorm() * invN;   // 1 - RSS/nulldev
            dev_ratio_(l) = rsq;
            if (allow_early_stop && l > 0) {
                if (rsq > kDevMax || (rsq - rsq_prev) < kFdev * rsq) {
                    n_fit = l + 1;
                    break;
                }
            }
            rsq_prev = rsq;
        }

        finish_path_(Bs, n_fit);
    }

    /** @brief Truncate the path to the lambdas actually fitted and back-scale
     *  the standardised coefficients to the original predictor space with
     *  the intercepts (shared by both engines). */
    void finish_path_(const Eigen::MatrixXd& Bs, Eigen::Index n_fit)
    {
        const Eigen::Index K = lambdas_.size();
        if (n_fit != K) {
            lambdas_.conservativeResize(n_fit);
            dev_ratio_.conservativeResize(n_fit);
        }
        const Eigen::Index Kfit = n_fit;

        // Bs is in (standardised-x, standardised-y) units; restore both scales.
        beta_.resize(p_, Kfit);
        for (Eigen::Index j = 0; j < p_; ++j) {
            beta_.row(j) = Bs.row(j).head(Kfit) * (y_scale_ / x_scale_(j));
        }
        a0_.resize(Kfit);
        for (Eigen::Index l = 0; l < Kfit; ++l) {
            a0_(l) = y_mean_ - x_mean_.dot(beta_.col(l));
        }
    }

    // ---- low-memory engine: coordinate descent on the raw X ------------------
    //
    // The copying engine works on Xs = (X - m) / s. Here the descent reads
    // X itself:
    //   * the column means m_j, the centred sums of squares and the scales s_j
    //     are computed in two read-only passes over the (masked) rows;
    //   * the residual r_true = r + c is kept as a vector r plus a scalar
    //     offset c, because subtracting a standardised column changes every
    //     row by the same amount m_j * diff / s_j; the vector update touches
    //     X's column only;
    //   * a correlation with a standardised column is
    //       ((w o x_j) . r + c * sum_w(x_j) - m_j * sum_w(r_true)) / s_j,
    //     where w is the 0/1 training mask (dropped when there is none);
    //   * the offset is folded into r once per sweep, which keeps the
    //     training residual mean-zero and the dot products free of the
    //     cancellation a large column mean would otherwise cause.
    // Same estimator and grid as the copying engine; they agree to rounding.

    void fit_raw_(Eigen::Ref<const Eigen::MatrixXd> X,
                  Eigen::Ref<const Eigen::VectorXd> y,
                  const Eigen::VectorXd* mask,
                  const Eigen::VectorXd* lambda_grid,
                  double alpha, Eigen::Index n_lambda, double lambda_min_ratio,
                  bool standardize, bool intercept, bool use_strong_rule,
                  int max_iter, double tol, bool allow_early_stop)
    {
        n_ = X.rows();
        p_ = X.cols();
        const bool masked = (mask != nullptr);
        if (masked && mask->size() != n_) {
            throw std::invalid_argument("enet_gaussian::fit_masked: mask length != n");
        }
        const double n_w = masked ? mask->sum() : static_cast<double>(n_);
        if (n_w < 2.0) {
            throw std::invalid_argument("enet_gaussian: fewer than 2 training rows");
        }
        const double inv_nw = 1.0 / n_w;

        // column statistics over the training rows: two read-only passes
        x_mean_.resize(p_);
        col_sum_.resize(p_);
        d_.resize(p_);
        x_scale_ = Eigen::VectorXd::Ones(p_);
        for (Eigen::Index j = 0; j < p_; ++j) {
            const auto   xj  = X.col(j);
            const double sum = masked ? mask->dot(xj) : xj.sum();
            col_sum_(j) = sum;
            const double m = intercept ? sum * inv_nw : 0.0;
            x_mean_(j) = m;
            const double css = masked
                ? (mask->array() * (xj.array() - m).square()).sum()
                : (xj.array() - m).square().sum();
            if (standardize) {
                const double sd = std::sqrt(css * inv_nw);
                if (sd > 1e-12) { x_scale_(j) = sd; }
            }
            d_(j) = css * inv_nw / (x_scale_(j) * x_scale_(j));
        }

        // response: centred over the training rows, held-out rows carry zero
        y_mean_ = intercept ? (masked ? mask->dot(y) * inv_nw : y.mean()) : 0.0;
        yc_ = y.array() - y_mean_;
        if (masked) { yc_.array() *= mask->array(); }
        y_scale_ = std::sqrt(yc_.squaredNorm() * inv_nw);
        if (y_scale_ > 1e-12) { yc_ /= y_scale_; } else { y_scale_ = 1.0; }

        // lambda grid
        if (lambda_grid == nullptr) {
            const double alpha_eff = (alpha > 1e-3) ? alpha : 1e-3;
            const double sum_yc = yc_.sum();
            double cmax = 0.0;
            for (Eigen::Index j = 0; j < p_; ++j) {
                const double xr = masked ? mask->cwiseProduct(X.col(j)).dot(yc_)
                                         : X.col(j).dot(yc_);
                const double corr = (xr - x_mean_(j) * sum_yc) / x_scale_(j);
                cmax = std::max(cmax, std::abs(corr));
            }
            const double lambda_max = cmax / (n_w * alpha_eff) * y_scale_;
            if (!(lambda_max > 1e-12) || !std::isfinite(lambda_max)) {
                throw std::runtime_error(
                    "enet_gaussian::fit: lambda grid anchor is degenerate "
                    "(y is constant or (nearly) orthogonal to X).");
            }
            const double lmr = (lambda_min_ratio > 0.0)
                             ? lambda_min_ratio
                             : (n_ < p_ ? 1e-2 : 1e-4);
            lambdas_ = make_log_grid(lambda_max, lambda_max * lmr, n_lambda);
        } else {
            lambdas_ = *lambda_grid;
            std::sort(lambdas_.data(), lambdas_.data() + lambdas_.size(),
                      std::greater<double>());
        }

        solve_raw_(X, mask, n_w, alpha, use_strong_rule, max_iter, tol, allow_early_stop);
        release_working_set();
    }

    void solve_raw_(Eigen::Ref<const Eigen::MatrixXd> X,
                    const Eigen::VectorXd* mask, double n_w,
                    double alpha, bool use_strong_rule, int max_iter, double tol,
                    bool allow_early_stop)
    {
        constexpr double kFdev   = 1.0e-5;
        constexpr double kDevMax = 0.999;

        const Eigen::Index K = lambdas_.size();
        const double inv_nw   = 1.0 / n_w;
        const bool   is_ridge = (alpha == 0.0);
        const bool   masked   = (mask != nullptr);

        Eigen::MatrixXd Bs   = Eigen::MatrixXd::Zero(p_, K);
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p_);
        Eigen::VectorXd r    = yc_;     // r_true = r + c
        double c     = 0.0;             // scalar residual offset
        double sum_r = masked ? mask->dot(r) : r.sum();   // sum of r over the training rows
        std::vector<char> active(static_cast<std::size_t>(p_), 0);
        n_nonconverged_ = 0;

        dev_ratio_ = Eigen::VectorXd::Zero(K);
        double   rsq_prev = 0.0;
        Eigen::Index n_fit = K;
        const double inv_yscale = 1.0 / y_scale_;

        // fold the offset into r: keeps the training residual mean-zero and
        // recomputes its sum exactly (the incremental sum only bridges a sweep)
        auto fold_offset = [&]() {
            if (c != 0.0) { r.array() += c; c = 0.0; }
            sum_r = masked ? mask->dot(r) : r.sum();
        };
        // correlation of standardised column j with r_true over the training rows
        auto corr = [&](Eigen::Index j) -> double {
            const double xr = masked ? mask->cwiseProduct(X.col(j)).dot(r)
                                     : X.col(j).dot(r);
            return (xr + c * col_sum_(j) - x_mean_(j) * (sum_r + c * n_w)) / x_scale_(j);
        };

        for (Eigen::Index l = 0; l < K; ++l) {
            const double lambda = lambdas_(l) * inv_yscale;
            const double l1     = lambda * alpha;
            const double l2     = lambda * (1.0 - alpha);
            const double lambda_prev =
                (l == 0) ? lambda : lambdas_(l - 1) * inv_yscale;

            std::fill(active.begin(), active.end(), 0);
            if (use_strong_rule && !is_ridge) {
                fold_offset();
                const double thr = alpha * (2.0 * lambda - lambda_prev);
                for (Eigen::Index j = 0; j < p_; ++j) {
                    if (d_(j) <= 1e-12) continue;
                    if (std::abs(beta(j)) > 0.0) { active[static_cast<std::size_t>(j)] = 1; continue; }
                    if (std::abs(corr(j) * inv_nw) >= thr)
                        active[static_cast<std::size_t>(j)] = 1;
                }
            } else {
                for (Eigen::Index j = 0; j < p_; ++j)
                    if (d_(j) > 1e-12) active[static_cast<std::size_t>(j)] = 1;
            }

            bool kkt_ok = false;
            while (!kkt_ok) {
                bool converged = false;
                for (int iter = 0; iter < max_iter; ++iter) {
                    double max_change = 0.0;
                    for (Eigen::Index j = 0; j < p_; ++j) {
                        if (!active[static_cast<std::size_t>(j)]) continue;
                        const double old_b = beta(j);
                        const double z_j   = corr(j) * inv_nw + d_(j) * old_b;
                        const double new_b = is_ridge
                            ? z_j / (d_(j) + l2)
                            : soft_threshold(z_j, l1) / (d_(j) + l2);
                        const double diff  = new_b - old_b;
                        if (diff != 0.0) {
                            const double step = diff / x_scale_(j);
                            r.noalias() -= X.col(j) * step;     // held-out rows are never read
                            c     += x_mean_(j) * step;
                            sum_r -= step * col_sum_(j);
                            beta(j)     = new_b;
                            max_change  = std::max(max_change, std::abs(diff) * std::sqrt(d_(j)));
                        }
                    }
                    fold_offset();
                    if (max_change < tol) { converged = true; break; }
                }
                if (!converged) ++n_nonconverged_;

                if (!use_strong_rule || is_ridge) { kkt_ok = true; break; }

                kkt_ok = true;
                for (Eigen::Index j = 0; j < p_; ++j) {
                    if (active[static_cast<std::size_t>(j)] || d_(j) <= 1e-12) continue;
                    if (std::abs(corr(j) * inv_nw) > l1 + 1e-9) {
                        active[static_cast<std::size_t>(j)] = 1;
                        kkt_ok = false;
                    }
                }
            }

            Bs.col(l) = beta;

            fold_offset();
            const double rss = masked ? mask->cwiseProduct(r).dot(r) : r.squaredNorm();
            const double rsq = 1.0 - rss * inv_nw;
            dev_ratio_(l) = rsq;
            if (allow_early_stop && l > 0) {
                if (rsq > kDevMax || (rsq - rsq_prev) < kFdev * rsq) {
                    n_fit = l + 1;
                    break;
                }
            }
            rsq_prev = rsq;
        }

        finish_path_(Bs, n_fit);
    }

    // ---- preprocessing: center (if intercept) + standardise (if requested) --
    void preprocess(Eigen::Ref<const Eigen::MatrixXd> X,
                    Eigen::Ref<const Eigen::VectorXd> y,
                    bool standardize, bool intercept)
    {
        n_ = X.rows();
        p_ = X.cols();

        x_mean_ = intercept ? X.colwise().mean().transpose().eval()
                            : Eigen::VectorXd::Zero(p_);

        Xs_ = X.rowwise() - x_mean_.transpose();
        finish_preprocess(y, standardize, intercept);
    }

    /**
     * @brief Preprocess an OWNED predictor matrix: it becomes Xs_ and is
     *        centred in place, so no second n x p copy is made (the CV folds
     *        use it). Bit-identical to preprocess(): the same column means
     *        over the same contiguous columns, the same per-element
     *        subtraction.
     */
    void preprocess_owned(Eigen::MatrixXd&& X_owned,
                          Eigen::Ref<const Eigen::VectorXd> y,
                          bool standardize, bool intercept)
    {
        n_ = X_owned.rows();
        p_ = X_owned.cols();

        x_mean_ = intercept ? X_owned.colwise().mean().transpose().eval()
                            : Eigen::VectorXd::Zero(p_);

        Xs_ = std::move(X_owned);
        Xs_.rowwise() -= x_mean_.transpose();
        finish_preprocess(y, standardize, intercept);
    }

    /** @brief The response scaling and the column standardisation of Xs_,
     *  shared by preprocess() and preprocess_owned(). */
    void finish_preprocess(Eigen::Ref<const Eigen::VectorXd> y,
                           bool standardize, bool intercept)
    {
        y_mean_ = intercept ? y.mean() : 0.0;
        yc_ = y.array() - y_mean_;

        // glmnet standardises the gaussian response to unit population SD
        // internally (independent of the predictor `standardize` flag) and
        // reports lambda on the original scale.  Mirror that here.
        y_scale_ = std::sqrt(yc_.squaredNorm() / static_cast<double>(n_));
        if (y_scale_ > 1e-12) { yc_ /= y_scale_; } else { y_scale_ = 1.0; }

        x_scale_ = Eigen::VectorXd::Ones(p_);
        if (standardize) {
            const double invN = 1.0 / static_cast<double>(n_);
            for (Eigen::Index j = 0; j < p_; ++j) {
                const double sd = std::sqrt(Xs_.col(j).squaredNorm() * invN);
                if (sd > 1e-12) { x_scale_(j) = sd; Xs_.col(j) /= sd; }
            }
        }
    }

    /** @brief Free the centred working copy of X (and the centred y) once the
     *  path is solved: predict() and every accessor read beta_, a0_ and the
     *  scaling vectors only, so a fitted object holds p x K, not n x p. */
    void release_working_set() noexcept
    {
        Eigen::MatrixXd().swap(Xs_);
        Eigen::VectorXd().swap(yc_);
    }

    static void validate_inputs(Eigen::Ref<const Eigen::MatrixXd> X,
                                Eigen::Ref<const Eigen::VectorXd> y,
                                double alpha)
    {
        if (X.rows() != y.size())
            throw std::invalid_argument("enet_gaussian: X.rows() != y.size()");
        if (X.rows() < 2)
            throw std::invalid_argument("enet_gaussian: need n >= 2");
        if (alpha < 0.0 || alpha > 1.0)
            throw std::invalid_argument("enet_gaussian: alpha must be in [0,1]");
    }

    static Eigen::VectorXd make_log_grid(double hi, double lo, Eigen::Index n_lambda)
    {
        if (n_lambda <= 0)
            throw std::invalid_argument("enet_gaussian: n_lambda must be positive");
        Eigen::VectorXd g(n_lambda);
        if (n_lambda == 1) { g(0) = hi; return g; }
        const double log_hi = std::log(hi);
        const double log_lo = std::log(lo);
        for (Eigen::Index i = 0; i < n_lambda; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(n_lambda - 1);
            g(i) = std::exp(log_hi - frac * (log_hi - log_lo));  // descending
        }
        return g;
    }

    static inline double soft_threshold(double z, double gamma) noexcept {
        if (z >  gamma) return z - gamma;
        if (z < -gamma) return z + gamma;
        return 0.0;
    }

    // private member variables ----------------------------------------------
    Eigen::Index    n_ = 0;
    Eigen::Index    p_ = 0;
    Eigen::MatrixXd Xs_;        ///< centred (+ standardised) predictors
    Eigen::VectorXd yc_;        ///< centred response
    Eigen::VectorXd x_mean_;    ///< column means (zero if intercept=false)
    Eigen::VectorXd x_scale_;   ///< column SDs   (ones if standardize=false)
    double          y_mean_  = 0.0;
    double          y_scale_ = 1.0;  ///< response population SD (glmnet scaling)

    Eigen::VectorXd lambdas_;   ///< grid (descending)
    Eigen::MatrixXd beta_;      ///< original-scale coefficients (p x n_lambda)
    Eigen::VectorXd a0_;        ///< intercepts (n_lambda)
    Eigen::VectorXd dev_ratio_; ///< fraction of null deviance explained per lambda
    int             n_nonconverged_ = 0;

    // raw-X engine
    bool            low_memory_ = false;
    Eigen::VectorXd col_sum_;   ///< masked column sums of X
    Eigen::VectorXd d_;         ///< ||x_{s,j}||^2 / n over the training rows
};

// ===================================================================================

/**
 * @brief K-fold cross-validated elastic-net replicating `glmnet::cv.glmnet`.
 *
 * @details Mirrors the public contract of `ridge_cv_svd`
 *  (`fit / cv_min / cv_1se / lambdas / cv_mse / cv_sem / index_min / index_1se`)
 *  but solves each fold with the coordinate-descent `enet_gaussian`
 *  engine instead of the SVD.  The @f$\lambda@f$ grid is built once on the full
 *  data (glmnet semantics) and reused unchanged across folds.
 *
 *  For ridge (@f$\alpha=0@f$) the returned lambdas are on glmnet's scale and the
 *  T-Rex conversion `lambda_2_lars = lambda.1se * p / 2` applies, exactly as for
 *  `ridge_cv_svd`.
 *
 *  References:
 *  -----------
 *   H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser.
 *     B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
 *   J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
 *     Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
 *   R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor, and R. J. Tibshirani, "Strong Rules
 *     for Discarding Predictors in Lasso-Type Problems," J. R. Stat. Soc. Ser. B Stat. Methodol., vol. 74,
 *     no. 2, pp. 245-266, 2012, doi: 10.1111/j.1467-9868.2011.01004.x.
 *   L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression Trees.
 *     Wadsworth, Belmont, CA, 1984.
 */
class enet_cv_ccd {

public:

    /** @brief Default constructor. */
    enet_cv_ccd() = default;

    /** @brief Choose the engine of the full-data and the fold fits (see
     *  enet_gaussian(bool)): true runs every fit on the raw X with a 0/1
     *  training mask, so the CV makes no allocation of the size of X. */
    explicit enet_cv_ccd(bool low_memory) : low_memory_(low_memory) {}

    /** @brief Set the thread budget of the Eigen products of fit() (>= 1);
     *  the library's default budget unless set. The fold fits inherit it. */
    void set_num_threads(int n_threads) {
        if (n_threads < 1) {
            throw std::invalid_argument("enet_cv_ccd::set_num_threads: n_threads must be >= 1.");
        }
        n_threads_ = n_threads;
    }

    /** @brief The thread budget of the Eigen products of fit(). */
    int num_threads() const noexcept { return n_threads_; }

    /** @brief Switch the engine; takes effect at the next fit(). */
    void set_low_memory(bool low_memory) noexcept { low_memory_ = low_memory; }

    /** @brief Whether the raw-X engine is selected. */
    bool low_memory() const noexcept { return low_memory_; }

    /**
     * @brief Fit K-fold CV over a glmnet-style grid using CD per fold.
     *
     * @param X                Predictor matrix (n x p).
     * @param y                Response vector (n).
     * @param alpha            Elastic-net mixing in [0,1]; 0 = ridge (default).
     * @param n_folds          Number of folds (default 10).
     * @param n_lambda         Grid size (default 100).
     * @param lambda_min_ratio Negative (default) auto-selects glmnet's rule.
     * @param seed             Fold-permutation RNG seed (default 0).
     * @param standardize      glmnet `standardize` (default true).
     * @param intercept        glmnet `intercept` (default true).
     * @param max_iter         CD sweeps per @f$\lambda@f$ (default 100000).
     * @param tol              CD tolerance (default 1e-7).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             double       alpha            = 0.0,
             int          n_folds          = 10,
             Eigen::Index n_lambda         = 100,
             double       lambda_min_ratio = -1.0,
             unsigned int seed             = 0,
             bool         standardize      = true,
             bool         intercept        = true,
             int          max_iter         = 100000,
             double       tol              = 1e-7)
    {
        // Eigen's products run on this object's budget for the scope of the fit
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);
        if (X.rows() != y.size())
            throw std::invalid_argument("enet_cv_ccd::fit: X.rows() != y.size()");
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        if (n_folds < 2)            throw std::invalid_argument("enet_cv_ccd::fit: n_folds >= 2");
        if (static_cast<Eigen::Index>(n_folds) > n)
            throw std::invalid_argument("enet_cv_ccd::fit: n_folds <= n");
        if (n_lambda <= 0)          throw std::invalid_argument("enet_cv_ccd::fit: n_lambda > 0");

        // ---- full-data grid (descending), reused across folds --------------
        n_lambda_requested_ = n_lambda;
        {
            enet_gaussian full(low_memory_);
            full.set_num_threads(n_threads_);
            full.fit(X, y, alpha, n_lambda, lambda_min_ratio,
                     standardize, intercept, /*use_strong_rule=*/false, max_iter, tol);
            lambdas_ = full.lambdas();
        }
        const Eigen::Index K = lambdas_.size();

        // ---- deterministic folds (seeded shuffle, round-robin assignment;
        //      same scheme as R's cv.glmnet with a fixed fold id vector) ----
        std::vector<Eigen::Index> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), Eigen::Index{0});
        std::mt19937 rng(seed);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<std::vector<Eigen::Index>> fold_idx(static_cast<std::size_t>(n_folds));
        for (Eigen::Index i = 0; i < n; ++i)
            fold_idx[static_cast<std::size_t>(i % n_folds)].push_back(perm[static_cast<std::size_t>(i)]);

        Eigen::MatrixXd fold_mse(K, n_folds);
        fold_mse.setZero();

        for (int f = 0; f < n_folds; ++f) {
            const auto& test_idx      = fold_idx[static_cast<std::size_t>(f)];
            const Eigen::Index n_test = static_cast<Eigen::Index>(test_idx.size());
            const Eigen::Index n_tr   = n - n_test;

            if (low_memory_) {
                // Raw-X engine: the training fold is a 0/1 mask, the held-out
                // rows are predicted row by row; nothing of the size of X.
                Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
                Eigen::VectorXd y_te(n_test);
                for (Eigen::Index t = 0; t < n_test; ++t) {
                    w(test_idx[static_cast<std::size_t>(t)]) = 0.0;
                    y_te(t) = y(test_idx[static_cast<std::size_t>(t)]);
                }
                enet_gaussian en(true);
                en.set_num_threads(n_threads_);
                en.fit_masked(X, y, w, lambdas_, alpha,
                              standardize, intercept, /*use_strong_rule=*/false, max_iter, tol);
                const Eigen::MatrixXd preds = en.predict_rows(X, test_idx);
                fold_mse.col(f) =
                    (preds.colwise() - y_te).colwise().squaredNorm().transpose()
                    / static_cast<double>(n_test);
                continue;
            }
            std::vector<bool> in_test(static_cast<std::size_t>(n), false);
            for (Eigen::Index t : test_idx) in_test[static_cast<std::size_t>(t)] = true;

            Eigen::MatrixXd X_tr(n_tr, p),  X_te(n_test, p);
            Eigen::VectorXd y_tr(n_tr),     y_te(n_test);
            for (Eigen::Index i = 0, itr = 0, ite = 0; i < n; ++i) {
                if (in_test[static_cast<std::size_t>(i)]) { X_te.row(ite) = X.row(i); y_te(ite) = y(i); ++ite; }
                else                                      { X_tr.row(itr) = X.row(i); y_tr(itr) = y(i); ++itr; }
            }

            // Fit CD path on the training fold at the SHARED full-data grid.
            // The gathered fold becomes the fit's working copy (centred and
            // standardised in place): one n_tr x p matrix per fold, not two.
            enet_gaussian en;
            en.set_num_threads(n_threads_);
            en.fit_owned(std::move(X_tr), y_tr, lambdas_, alpha,
                         standardize, intercept, /*use_strong_rule=*/false, max_iter, tol);

            // Predict held-out fold; per-lambda MSE.
            const Eigen::MatrixXd preds = en.predict(X_te);  // n_test x K
            fold_mse.col(f) =
                (preds.colwise() - y_te).colwise().squaredNorm().transpose()
                / static_cast<double>(n_test);
        }

        // ---- aggregate mean + SEM across folds (identical to ridge_cv*) ----
        cv_mse_.setZero(K);
        cv_sem_.setZero(K);
        const double inv_F      = 1.0 / static_cast<double>(n_folds);
        const double inv_Fminus = 1.0 / static_cast<double>(n_folds - 1);
        for (Eigen::Index k = 0; k < K; ++k) {
            const double mean = fold_mse.row(k).mean();
            cv_mse_(k) = mean;
            double var = 0.0;
            for (int f = 0; f < n_folds; ++f) { const double d = fold_mse(k, f) - mean; var += d * d; }
            cv_sem_(k) = std::sqrt(var * inv_Fminus * inv_F);
        }

        // ---- lambda.min and lambda.1se (glmnet semantics) ------------------
        // Grid is DESCENDING: index 0 = largest lambda.
        Eigen::Index k_min = 0;
        double best = std::numeric_limits<double>::max();
        for (Eigen::Index k = 0; k < K; ++k)
            if (cv_mse_(k) < best) { best = cv_mse_(k); k_min = k; }
        idx_min_ = k_min;

        const double threshold = cv_mse_(k_min) + cv_sem_(k_min);
        // largest lambda whose mean MSE <= threshold == smallest index here.
        Eigen::Index k_1se = k_min;
        for (Eigen::Index k = 0; k < K; ++k)
            if (cv_mse_(k) <= threshold) { k_1se = k; break; }
        idx_1se_ = k_1se;

        fitted_ = true;
    }

    // public member accessors ------------------------------------------------
    const Eigen::VectorXd& lambdas()  const noexcept { return lambdas_; }
    const Eigen::VectorXd& cv_mse()   const noexcept { return cv_mse_; }
    const Eigen::VectorXd& cv_sem()   const noexcept { return cv_sem_; }

    double cv_min() const { check_fitted(); return lambdas_(idx_min_); }
    double cv_1se() const { check_fitted(); return lambdas_(idx_1se_); }

    Eigen::Index index_min() const noexcept { return idx_min_; }
    Eigen::Index index_1se() const noexcept { return idx_1se_; }

    /** @brief True when `cv_min()` or `cv_1se()` landed on the FIRST or LAST
     *         grid point — the pick is the grid boundary, not an optimum.
     *
     * @details Deliberately the ONLY dimensional diagnostic on this class.
     *  The support cap and the regime-dependent lambda range that
     *  `ienet_cv_ccd` needs do NOT apply here, for two structural reasons:
     *
     *   1. The T-Rex pipeline calls this class only at alpha = 0 (pure
     *      ridge). There is then no support at all — every coefficient is
     *      nonzero at every lambda — so no support saturation, no
     *      rank-deficient active set, and none of the flat-valley cost that
     *      motivated the cap. Measured: this tuner costs 0.357 s on the same
     *      design where the IEN tuner cost 3108 s before the cap.
     *   2. Its lambda is the SELECTION TARGET of a 1-D CV, not a nuisance
     *      profiled out. The deep end of the grid is therefore a region the
     *      answer may legitimately occupy, so glmnet's
     *      (n < p ? 1e-2 : 1e-4) rule is right here and must NOT be replaced
     *      by the flat 1e-2 that `ienet_cv_ccd` uses.
     *
     *  Verified on both regimes (n=150,p=100 and n=200,p=500):
     *  picks interior at idx_min 61/100 and 88/100, idx_1se 50/100 and
     *  53/100 — no edge, no retuning warranted. */
    bool selection_at_grid_edge() const noexcept {
        const Eigen::Index K = lambdas_.size();
        if (K == 0) { return false; }
        return idx_min_ == 0 || idx_min_ == K - 1 ||
               idx_1se_ == 0 || idx_1se_ == K - 1;
    }

    /** @brief True when glmnet's fdev/devmax rules truncated the full-data
     *         grid, so `lambdas()` is shorter than the requested `n_lambda`.
     *
     * @details Read this together with `selection_at_grid_edge()`, because on
     *  this class the two causes of a last-index pick are different things:
     *
     *   - at-edge AND truncated  -> BENIGN. The deviance rules stopped the
     *     path because decreasing lambda further stopped improving the fit;
     *     the CV is simply indifferent over the tail it never saw.
     *   - at-edge AND NOT truncated -> the requested lambda RANGE ran out
     *     with the CV still improving; widen `lambda_min_ratio`.
     *   - idx_min_ == 0 (largest lambda) is always the range ceiling,
     *     truncation cannot cause it. */
    bool grid_truncated() const noexcept {
        return lambdas_.size() < n_lambda_requested_;
    }

private:

    /** @brief Thread budget of the Eigen products of fit() and the fold fits (>= 1). */
    int n_threads_{::trex::utils::openmp::get_num_threads()};

    void check_fitted() const {
        if (!fitted_)
            throw std::runtime_error("enet_cv_ccd: call fit() first");
    }

    bool            fitted_  = false;
    bool            low_memory_ = false;
    Eigen::VectorXd lambdas_;
    Eigen::VectorXd cv_mse_;
    Eigen::VectorXd cv_sem_;
    Eigen::Index    idx_min_ = 0;
    Eigen::Index    idx_1se_ = 0;
    Eigen::Index    n_lambda_requested_ = 0;
};

// ===================================================================================
} /* End of namespace model_selection */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_ENET_CV_CCD_HPP */
