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
 *   algorithm that `glmnet` uses internally.  Because `glmnet` IS coordinate
 *   descent, the coefficient path produced here matches `glmnet`'s to solver
 *   tolerance (not merely within CV noise), which makes it the most faithful
 *   reference for the lambda_2 (ridge / elastic-net penalty) determination.
 *
 *   **Objective (glmnet Gaussian parametrisation).**  For standardised columns
 *   the solver minimises, over @f$\beta@f$,
 *   @f[
 *     \frac{1}{2N}\,\lVert y_c - X_s\beta \rVert_2^2
 *       \;+\; \lambda\Big(\alpha\lVert\beta\rVert_1
 *       + \tfrac{1-\alpha}{2}\lVert\beta\rVert_2^2\Big),
 *   @f]
 *   with per-coordinate update
 *   @f$ \beta_j \leftarrow
 *       S\!\big(z_j,\ \lambda\alpha\big)\big/\big(d_j + \lambda(1-\alpha)\big) @f$,
 *   where @f$ z_j = \tfrac1N x_{s,j}^\top r + d_j\beta_j @f$,
 *   @f$ d_j = \tfrac1N\lVert x_{s,j}\rVert_2^2 @f$ (= 1 when standardised),
 *   @f$ S @f$ is the soft-threshold operator, and @f$ r @f$ is the working
 *   residual.
 *
 *   **glmnet conventions reproduced.**
 *    - Standardisation by the *population* SD (divisor @f$N@f$), so
 *      @f$\lVert x_{s,j}\rVert_2^2 = N@f$ and @f$d_j = 1@f$.
 *    - @f$\lambda_{\max} = \max_j |x_{s,j}^\top y_c| / (N\,\alpha_{\text{eff}})@f$
 *      with @f$\alpha_{\text{eff}}=\max(\alpha,10^{-3})@f$ (the glmnet ridge
 *      fallback, so @f$\alpha=0@f$ yields a finite anchor).
 *    - Log-spaced grid down to @f$\lambda_{\max}\cdot\varepsilon@f$,
 *      @f$\varepsilon=10^{-2}@f$ when @f$N<p@f$ else @f$10^{-4}@f$.
 *    - Coefficients returned in the ORIGINAL predictor scale together with the
 *      intercept @f$ \beta_0 = \bar y - \bar x^\top\beta @f$.
 *
 *   **Performance.**  Tibshirani's sequential strong rule prunes the active set;
 *   the inner CD sweeps only active coordinates and a KKT pass re-admits any
 *   violated predictor.  Warm starts propagate @f$\beta@f$ down the (descending)
 *   @f$\lambda@f$ grid.  Pure ridge (@f$\alpha=0@f$) skips the strong rule (no
 *   exact zeros) and uses the closed per-coordinate ridge update.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Core>

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

    /** @brief Default constructor. */
    enet_gaussian() = default;

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
        validate_inputs(X, y, alpha);
        preprocess(X, y, standardize, intercept);

        const double alpha_eff = (alpha > 1e-3) ? alpha : 1e-3;
        // yc_ is standardised (unit population SD); report lambda on glmnet's
        // original scale, i.e. internal lambda * y_scale_.
        const double lambda_max =
            (Xs_.transpose() * yc_).cwiseAbs().maxCoeff()
            / (static_cast<double>(n_) * alpha_eff) * y_scale_;

        const double lmr = (lambda_min_ratio > 0.0)
                         ? lambda_min_ratio
                         : (n_ < p_ ? 1e-2 : 1e-4);

        lambdas_ = make_log_grid(lambda_max, lambda_max * lmr, n_lambda);
        solve(alpha, use_strong_rule, max_iter, tol, /*allow_early_stop=*/true);
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
        validate_inputs(X, y, alpha);
        if (lambda_grid.size() == 0) {
            throw std::invalid_argument(
                "enet_gaussian::fit: lambda_grid is empty");
        }
        preprocess(X, y, standardize, intercept);

        lambdas_ = lambda_grid;
        std::sort(lambdas_.data(), lambdas_.data() + lambdas_.size(),
                  std::greater<double>());
        solve(alpha, use_strong_rule, max_iter, tol, /*allow_early_stop=*/false);
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

        // truncate the path to the lambdas actually fitted (early-stop case).
        if (n_fit != K) {
            lambdas_.conservativeResize(n_fit);
            dev_ratio_.conservativeResize(n_fit);
        }
        const Eigen::Index Kfit = n_fit;

        // --- back-scale to original predictor space + intercepts ------------
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

    // ---- preprocessing: center (if intercept) + standardise (if requested) --
    void preprocess(Eigen::Ref<const Eigen::MatrixXd> X,
                    Eigen::Ref<const Eigen::VectorXd> y,
                    bool standardize, bool intercept)
    {
        n_ = X.rows();
        p_ = X.cols();

        x_mean_ = intercept ? X.colwise().mean().transpose().eval()
                            : Eigen::VectorXd::Zero(p_);
        y_mean_ = intercept ? y.mean() : 0.0;

        Xs_ = X.rowwise() - x_mean_.transpose();
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
 */
class enet_cv_ccd {

public:

    /** @brief Default constructor. */
    enet_cv_ccd() = default;

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
        if (X.rows() != y.size())
            throw std::invalid_argument("enet_cv_ccd::fit: X.rows() != y.size()");
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        if (n_folds < 2)            throw std::invalid_argument("enet_cv_ccd::fit: n_folds >= 2");
        if (static_cast<Eigen::Index>(n_folds) > n)
            throw std::invalid_argument("enet_cv_ccd::fit: n_folds <= n");
        if (n_lambda <= 0)          throw std::invalid_argument("enet_cv_ccd::fit: n_lambda > 0");

        // ---- full-data grid (descending), reused across folds --------------
        {
            enet_gaussian full;
            full.fit(X, y, alpha, n_lambda, lambda_min_ratio,
                     standardize, intercept, /*use_strong_rule=*/false, max_iter, tol);
            lambdas_ = full.lambdas();
        }
        const Eigen::Index K = lambdas_.size();

        // ---- deterministic folds (same scheme as ridge_cv_glmnet) ----------
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

            std::vector<bool> in_test(static_cast<std::size_t>(n), false);
            for (Eigen::Index t : test_idx) in_test[static_cast<std::size_t>(t)] = true;

            Eigen::MatrixXd X_tr(n_tr, p),  X_te(n_test, p);
            Eigen::VectorXd y_tr(n_tr),     y_te(n_test);
            for (Eigen::Index i = 0, itr = 0, ite = 0; i < n; ++i) {
                if (in_test[static_cast<std::size_t>(i)]) { X_te.row(ite) = X.row(i); y_te(ite) = y(i); ++ite; }
                else                                      { X_tr.row(itr) = X.row(i); y_tr(itr) = y(i); ++itr; }
            }

            // Fit CD path on the training fold at the SHARED full-data grid.
            enet_gaussian en;
            en.fit(X_tr, y_tr, lambdas_, alpha,
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

private:

    void check_fitted() const {
        if (!fitted_)
            throw std::runtime_error("enet_cv_ccd: call fit() first");
    }

    bool            fitted_  = false;
    Eigen::VectorXd lambdas_;
    Eigen::VectorXd cv_mse_;
    Eigen::VectorXd cv_sem_;
    Eigen::Index    idx_min_ = 0;
    Eigen::Index    idx_1se_ = 0;
};

// ===================================================================================
} /* End of namespace model_selection */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_ENET_CV_CCD_HPP */
