// ===================================================================================
// tikhonov_cv_svd.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_TIKHONOV_CV_SVD_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_TIKHONOV_CV_SVD_HPP
// ===================================================================================
/**
 * @file tikhonov_cv_svd.hpp
 *
 * @brief K-fold cross-validated generalized-Tikhonov (anisotropic) ridge with
 *        `lambda.min` / `lambda.1se` selection over a DIRECT lambda2 grid,
 *        implemented analytically via per-fold SVDs.  Companion of
 *        `ridge_cv_svd` for the Informed Elastic Net geometry: the selected
 *        lambda2 is on the solver scale and is consumable AS-IS by the
 *        `TIENET_Solver` / `TCIENET_Solver` constructors (no p/2 conversion).
 *
 * @details
 *   **Estimator.**  For a PSD penalty matrix K (p x p) the per-fold model is
 *
 *       beta_hat(l2) = argmin  1/2 ||y_c - X_s beta||^2
 *                             + (l2/2)  beta^T K beta
 *                             + (eps/2) ||P0 beta||^2 ,
 *
 *   where X_s is the per-fold centered + column-L2-normalized training matrix
 *   (the selector scale of the T-solvers), y_c the centered response, and
 *   P0 the orthogonal projector onto null(K).  The eps-term is the
 *   *null-space floor*: it regularizes ONLY the directions the Tikhonov
 *   penalty leaves free, leaving the K-penalized directions untouched.
 *
 *   **Specializations.**
 *   - Group mode (TIENET / T-CIENET): `fit(X, y, groups, ...)` uses the IEN
 *     group-mean penalty W = sum_m 1_m 1_m^T / p_m.  W is the orthogonal
 *     projector onto span{ 1_m / sqrt(p_m) }, so its eigenstructure is known
 *     in closed form: the penalized block is spanned by the M normalized
 *     group-mean features and null(W) by the within-group Helmert contrasts.
 *     No p x p eigendecomposition is performed.
 *   - General mode: `fit(X, y, K, ...)` accepts an arbitrary sparse PSD
 *     Tikhonov matrix K = Gamma^T Gamma; one dense symmetric
 *     eigendecomposition of K (O(p^3), fold-independent) splits the
 *     penalized range from the null space.  All-singleton groups (K = I)
 *     reproduce `ridge_cv_svd` exactly.
 *
 *   **Well-posedness (p > n regime) — read this before trusting the curve.**
 *   The pure Tikhonov backbone (eps = 0) is well-posed iff
 *   null(X_s) ∩ null(K) = {0}.  With the rank-M group penalty this fails
 *   whenever p > n_train + M: the fit then has a flat of minimizers and the
 *   class resolves it by the minimum-norm convention (pseudoinverse in the
 *   contrast block, the eps -> 0 limit).  Worse, if p - M >= n_train the
 *   unpenalized contrast block generically INTERPOLATES the training
 *   response, the profiled group-block problem degenerates (G = 0), and the
 *   CV curve is CONSTANT in lambda2 — prediction-CV carries no information
 *   about lambda2 at all.  A positive eps breaks the interpolation, but then
 *   the curve (and the selected lambda2) is shaped by the arbitrary floor.
 *   `fit()` therefore reports the regime through `contrast_interpolates()`;
 *   in that regime prefer the path-based `ienet_cv_ccd` tuner (lambda1 > 0
 *   restores well-posedness).
 *
 *   **Computation per fold** (all lambda2 values analytic, two SVDs):
 *   transform to the K-eigenbasis, a = Lambda_+^{1/2} V_+^T beta (penalized,
 *   dim r) and c = V_0^T beta (free, dim p - r); profile out c through the
 *   ridge-hat identity  min_c 1/2||z - XC c||^2 + (eps/2)||c||^2
 *   = 1/2 z^T (I - H_eps) z ; then the profiled a-problem is an isotropic
 *   ridge in dimension r solved by one small SVD for the whole grid.
 *   `Eigen::JacobiSVD` is used throughout (BDCSVD is unsafe under -O3
 *   -march=native with some Eigen versions; see ridge_cv_svd).
 *
 *   **Public interface** (parity with `ridge_cv_svd` / `enet_cv_ccd`):
 *   `fit()`, `cv_min()`, `cv_1se()`, `lambdas()`, `cv_mse()`, `cv_sem()`,
 *   `index_min()`, `index_1se()`.
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
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

// ===================================================================================

// Namespace embedding: trex::ml_methods::model_selection
namespace trex {
namespace ml_methods {
namespace model_selection {

// ===================================================================================

/**
 * @brief K-fold cross-validated generalized-Tikhonov ridge over a direct
 *        lambda2 grid (SVD-analytic per fold), with group-mean (IEN) and
 *        general sparse-K specializations.
 */
class tikhonov_cv_svd {

public:

    /** @brief Default constructor. */
    tikhonov_cv_svd() = default;

    /**
     * @brief Fit K-fold CV for the IEN group-mean penalty (TIENET geometry).
     *
     * @param X            Predictor matrix (n x p).
     * @param y            Response vector (n).
     * @param groups       0-based contiguous group ids in [0, M-1], length p.
     * @param n_folds      Number of CV folds (default 10).
     * @param n_lambda     Number of lambda2 values on the geometric grid
     *                     (default 100).
     * @param lambda_ratio Grid range factor: grid spans
     *                     `[lambda2_max / lambda_ratio, lambda2_max]`
     *                     (default 1e4).
     * @param seed         RNG seed for the fold permutation (default 0).
     * @param epsilon      Null-space floor eps (>= 0).  Negative (default)
     *                     auto-selects: 0 (exact min-norm) when the contrast
     *                     block cannot interpolate (p - M < smallest training
     *                     fold), else 1e-3 with a well-posedness flag raised
     *                     (see `contrast_interpolates()`).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             const Eigen::VectorXi& groups,
             int          n_folds      = 10,
             Eigen::Index n_lambda     = 100,
             double       lambda_ratio = 1e4,
             unsigned int seed         = 0,
             double       epsilon      = -1.0)
    {
        validate_common(X, y, n_folds, n_lambda, lambda_ratio);
        build_group_basis(groups, X.cols());
        run_cv(X, y, n_folds, n_lambda, lambda_ratio, seed, epsilon);
    }

    /**
     * @brief Fit K-fold CV using a matrix-passing variant
     *        for a general sparse PSD Tikhonov matrix
     *        K = Gamma^T Gamma.
     *
     * @param X            Predictor matrix (n x p).
     * @param y            Response vector (n).
     * @param K            PSD Tikhonov matrix (p x p, sparse), applied to
     *                     beta on the standardized (selector) scale.  One
     *                     dense O(p^3) eigendecomposition is performed.
     * @param n_folds      Number of CV folds (default 10).
     * @param n_lambda     Grid size (default 100).
     * @param lambda_ratio Grid range factor (default 1e4).
     * @param seed         Fold-permutation seed (default 0).
     * @param epsilon      Null-space floor (see group overload).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> X,
             Eigen::Ref<const Eigen::VectorXd> y,
             const Eigen::SparseMatrix<double>& K,
             int          n_folds      = 10,
             Eigen::Index n_lambda     = 100,
             double       lambda_ratio = 1e4,
             unsigned int seed         = 0,
             double       epsilon      = -1.0)
    {
        validate_common(X, y, n_folds, n_lambda, lambda_ratio);
        build_general_basis(K, X.cols());
        run_cv(X, y, n_folds, n_lambda, lambda_ratio, seed, epsilon);
    }

    // public member accessors ------------------------------------------------

    /** @brief Lambda2 grid that was searched (ascending, solver scale). */
    const Eigen::VectorXd& lambdas() const noexcept { return lambdas_; }

    /** @brief Per-lambda2 mean CV-MSE across folds. */
    const Eigen::VectorXd& cv_mse() const noexcept { return cv_mse_; }

    /** @brief Per-lambda2 standard error of the CV-MSE mean across folds. */
    const Eigen::VectorXd& cv_sem() const noexcept { return cv_sem_; }

    /** @brief Lambda2 minimising the mean CV-MSE (`lambda.min`), directly
     *         consumable by TIENET_Solver / TCIENET_Solver. */
    double cv_min() const { check_fitted(); return lambdas_(idx_min_); }

    /** @brief Largest lambda2 within one SE of the CV-min (`lambda.1se`). */
    double cv_1se() const { check_fitted(); return lambdas_(idx_1se_); }

    /** @brief Index of the lambda2 minimising mean CV-MSE. */
    Eigen::Index index_min() const noexcept { return idx_min_; }

    /** @brief Index of the largest lambda2 within one SE of the minimum. */
    Eigen::Index index_1se() const noexcept { return idx_1se_; }

    /** @brief Null-space floor actually used (after auto-selection). */
    double epsilon() const noexcept { return epsilon_used_; }

    /** @brief Dimension of null(K) (p - M in group mode). */
    Eigen::Index null_dim() const noexcept { return p_ - r_; }

    /** @brief True if the unpenalized contrast block can interpolate the
     *         smallest training fold (null_dim >= n_train_min): in this
     *         regime the eps = 0 CV curve is constant in lambda2 and any
     *         eps > 0 pick is floor-driven — the Tikhonov prediction-CV is
     *         structurally uninformative; prefer `ienet_cv_ccd`. */
    bool contrast_interpolates() const noexcept { return interpolating_; }

private:

    // ---- basis construction -------------------------------------------------
    //
    // The computational core only needs the two linear maps
    //     XA = X_s * (V_+ Lambda_+^{-1/2})    (penalized block, r columns)
    //     XC = X_s * V_0                      (free block, p - r columns)
    // realized by the dense p x r and p x (p-r) matrices TA_ / TC_.
    // Group mode fills them in closed form (Q_G and per-group Helmert
    // contrasts); general mode eigendecomposes K once.

    void build_group_basis(const Eigen::VectorXi& groups,
                           Eigen::Index p)
    {
        if (groups.size() != p)
            throw std::invalid_argument(
                "tikhonov_cv_svd: groups.size() != X.cols()");
        const int M = groups.size() > 0 ? groups.maxCoeff() + 1 : 0;
        if (M <= 0)
            throw std::invalid_argument("tikhonov_cv_svd: empty grouping");
        std::vector<std::vector<Eigen::Index>> members(
            static_cast<std::size_t>(M));
        for (Eigen::Index j = 0; j < p; ++j) {
            const int m = groups(j);
            if (m < 0 || m >= M)
                throw std::invalid_argument(
                    "tikhonov_cv_svd: group ids must be 0-based contiguous");
            members[static_cast<std::size_t>(m)].push_back(j);
        }
        for (const auto& g : members)
            if (g.empty())
                throw std::invalid_argument(
                    "tikhonov_cv_svd: group ids must be 0-based contiguous "
                    "(empty group id encountered)");

        p_ = p;
        r_ = M;
        TA_.setZero(p, M);
        TC_.setZero(p, p - M);

        // Q_G: column m = 1_m / sqrt(p_m); Lambda_+ = I (W is a projector).
        Eigen::Index c = 0;
        for (int m = 0; m < M; ++m) {
            const auto& g  = members[static_cast<std::size_t>(m)];
            const auto  pm = static_cast<double>(g.size());
            const double q = 1.0 / std::sqrt(pm);
            for (Eigen::Index j : g) TA_(j, m) = q;
            // Helmert contrasts h_k = (1,...,1,-k,0,...)/sqrt(k(k+1)),
            // k = 1..p_m-1: orthonormal basis of the within-group contrasts.
            for (std::size_t k = 1; k < g.size(); ++k, ++c) {
                const double s =
                    1.0 / std::sqrt(static_cast<double>(k) *
                                    static_cast<double>(k + 1));
                for (std::size_t i = 0; i < k; ++i) TC_(g[i], c) = s;
                TC_(g[k], c) = -static_cast<double>(k) * s;
            }
        }
    }


    /**
     * @brief Build a general basis from a sparse PSD Tikhonov matrix K = Gamma^T Gammma.
     *
     * @param K Sparse PSD Tikhonov matrix (p x p).
     * @param p Number of predictors (X.cols()).
     */
    void build_general_basis(const Eigen::SparseMatrix<double>& K,
                             Eigen::Index p)
    {
        if (K.rows() != p || K.cols() != p)
            throw std::invalid_argument(
                "tikhonov_cv_svd: K must be p x p with p = X.cols()");
        const Eigen::MatrixXd Kd = Eigen::MatrixXd(K);

        if ((Kd - Kd.transpose()).cwiseAbs().maxCoeff() >
            1e-10 * (1.0 + Kd.cwiseAbs().maxCoeff()))
            throw std::invalid_argument("tikhonov_cv_svd: K is not symmetric");

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Kd);
        if (eig.info() != Eigen::Success)
            throw std::runtime_error(
                "tikhonov_cv_svd: eigendecomposition of K failed");
        const Eigen::VectorXd& ev = eig.eigenvalues();   // ascending
        const double ev_max = ev(p - 1);
        if (ev_max <= 0.0)
            throw std::invalid_argument("tikhonov_cv_svd: K is zero/negative");
        if (ev(0) < -1e-10 * ev_max)
            throw std::invalid_argument("tikhonov_cv_svd: K is not PSD");

        const double tol = 1e-10 * ev_max;
        Eigen::Index r = 0;
        for (Eigen::Index i = 0; i < p; ++i)
            if (ev(i) > tol) ++r;

        p_ = p;
        r_ = r;
        TC_ = eig.eigenvectors().leftCols(p - r);
        TA_.resize(p, r);
        for (Eigen::Index i = 0; i < r; ++i) {
            const Eigen::Index k = p - r + i;   // ascending order
            TA_.col(i) = eig.eigenvectors().col(k) / std::sqrt(ev(k));
        }
    }

    // ---- shared CV core -----------------------------------------------------

    /** @brief Run K-fold cross-validation.
     *
     * @param X Design matrix (n x p).
     * @param y Response vector (n).
     * @param n_folds Number of folds.
     * @param n_lambda Number of lambda2 values.
     * @param lambda_ratio Ratio between max and min lambda2.
     * @param seed Random seed for fold permutation.
     * @param epsilon Null-space floor.
     */
    void run_cv(Eigen::Ref<const Eigen::MatrixXd> X,
                Eigen::Ref<const Eigen::VectorXd> y,
                int n_folds,
                Eigen::Index n_lambda,
                double lambda_ratio,
                unsigned int seed,
                double epsilon)
    {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        // Smallest training-fold size (round-robin folds over n rows).
        const Eigen::Index n_test_max =
            (n + static_cast<Eigen::Index>(n_folds) - 1) /
            static_cast<Eigen::Index>(n_folds);
        const Eigen::Index n_train_min = n - n_test_max;

        interpolating_ = (p_ - r_) >= n_train_min;
        epsilon_used_  = (epsilon >= 0.0)
                       ? epsilon
                       : (interpolating_ ? 1e-3 : 0.0);

        // ---- full-data pass: standardization + grid anchor ------------------
        Eigen::MatrixXd XA_f, XC_f;
        Eigen::VectorXd yc_f;
        double y_mean_f = 0.0;
        standardize_and_project(X, y, XA_f, XC_f, yc_f, y_mean_f);

        // Anchor c_max on the PROFILED group-block problem, i.e. after the
        // eps-hat projection of the contrast block (mirrors ridge_cv_svd's
        // c_max = max |X_s^T y_c| which is the singleton special case).
        {
            Eigen::MatrixXd G;
            Eigen::VectorXd yt;
            Eigen::MatrixXd UC, VC;
            Eigen::VectorXd sC;
            profile_contrasts(XA_f, XC_f, yc_f, epsilon_used_,
                              G, yt, UC, VC, sC);
            const double c_max = (G.cols() > 0 && G.size() > 0)
                ? (G.transpose() * yt).cwiseAbs().maxCoeff() : 0.0;
            if (!(c_max > 1e-12) || !std::isfinite(c_max)) {
                throw std::runtime_error(
                    "tikhonov_cv_svd::fit: lambda2 grid anchor is degenerate "
                    "(profiled penalized block carries no correlation with y; "
                    "for p - M >= n_train with eps = 0 the contrast block "
                    "interpolates and prediction-CV cannot identify lambda2 — "
                    "see contrast_interpolates()).");
            }
            make_grid(c_max, lambda_ratio, n_lambda);
        }
        const Eigen::Index K = lambdas_.size();

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

        Eigen::MatrixXd fold_mse(K, n_folds);
        fold_mse.setZero();

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

            // Per-fold standardization (training stats) + projection; the
            // validation block is standardized with the SAME training stats.
            Eigen::MatrixXd XA_tr, XC_tr, XA_te, XC_te;
            Eigen::VectorXd yc_tr;
            double y_mean_tr = 0.0;
            standardize_and_project(X_tr, y_tr, XA_tr, XC_tr, yc_tr, y_mean_tr,
                                    &X_te, &XA_te, &XC_te);

            // Profile out the contrast block, then solve the r-dimensional
            // isotropic ridge for the WHOLE lambda2 grid analytically.
            Eigen::MatrixXd G;
            Eigen::VectorXd yt;
            Eigen::MatrixXd UC, VC;
            Eigen::VectorXd sC;
            profile_contrasts(XA_tr, XC_tr, yc_tr, epsilon_used_,
                              G, yt, UC, VC, sC);

            Eigen::JacobiSVD<Eigen::MatrixXd> svdG(
                G, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Eigen::VectorXd& sg = svdG.singularValues();
            const double rank_tol_g = std::numeric_limits<double>::epsilon()
                * (sg.size() > 0 ? sg(0) : 0.0)
                * std::sqrt(static_cast<double>(n_tr));
            Eigen::Index rg = 0;
            for (Eigen::Index i = 0; i < sg.size(); ++i)
                if (sg(i) > rank_tol_g) ++rg; else break;

            // A_hat (r x K) via the ridge filter s/(s^2 + lambda2).
            Eigen::MatrixXd A_hat = Eigen::MatrixXd::Zero(r_, K);
            if (rg > 0) {
                const auto Ug = svdG.matrixU().leftCols(rg);
                const auto Vg = svdG.matrixV().leftCols(rg);
                const auto sr = sg.head(rg);
                const Eigen::VectorXd Uty = Ug.transpose() * yt;
                Eigen::MatrixXd Df(rg, K);
                for (Eigen::Index k = 0; k < K; ++k)
                    Df.col(k) = sr.array() /
                                (sr.array().square() + lambdas_(k));
                A_hat = Vg * (Uty.asDiagonal() * Df);
            }

            // Contrast coefficients (reduced): C_red (rC x K) with
            // c = V_C diag(s/(s^2+eps)) U_C^T (y - XA a).
            const Eigen::Index rC = sC.size();
            Eigen::MatrixXd preds =
                XA_te * A_hat;                       // n_test x K
            if (rC > 0) {
                const Eigen::MatrixXd R0 =
                    (-XA_tr * A_hat).colwise() + yc_tr;   // y - XA a
                Eigen::VectorXd fc(rC);
                for (Eigen::Index i = 0; i < rC; ++i)
                    fc(i) = sC(i) / (sC(i) * sC(i) + epsilon_used_);
                const Eigen::MatrixXd C_red =
                    fc.asDiagonal() * (UC.transpose() * R0); // rC x K
                preds.noalias() += (XC_te * VC) * C_red;
            }
            preds.array() += y_mean_tr;

            fold_mse.col(f) =
                (preds.colwise() - y_te).colwise().squaredNorm().transpose()
                / static_cast<double>(n_test);
        }

        aggregate_and_select(fold_mse, n_folds);
        fitted_ = true;
    }


    /**
     * @brief Standardize and project operation for the penalized and free blocks.
     *        Center + column L2-normalization, ridge_cv_svd conventionns.
     *        Map into the penalized/free blocks: XA = X_s TA_, XC = X_s TC_.
     *        If X_te is given, the SAME training stats are applied to it.
     *
     * @param X Input design matrix (n x p).
     * @param y Input response vector (n).
     * @param XA Output penalized block (n x r).
     * @param XC Output free block (n x (p - r)).
     * @param yc Output centered response (n).
     * @param y_mean Output mean of the response.
     * @param X_te Optional test design matrix (n_te x p).
     * @param XA_te Optional output penalized block for test data (n_te x r).
     * @param XC_te Optional output free block for test data (n_te x (p - r)).
     */
    void standardize_and_project(Eigen::Ref<const Eigen::MatrixXd> X,
                                 Eigen::Ref<const Eigen::VectorXd> y,
                                 Eigen::MatrixXd& XA,
                                 Eigen::MatrixXd& XC,
                                 Eigen::VectorXd& yc,
                                 double& y_mean,
                                 const Eigen::MatrixXd* X_te = nullptr,
                                 Eigen::MatrixXd* XA_te = nullptr,
                                 Eigen::MatrixXd* XC_te = nullptr) const
    {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        const Eigen::VectorXd x_mean = X.colwise().mean();
        y_mean = y.mean();

        Eigen::VectorXd x_scale =
            (X.colwise().squaredNorm().transpose()
             - static_cast<double>(n) * x_mean.cwiseAbs2()
            ).cwiseMax(0.0).cwiseSqrt();
        {
            const double max_scale = x_scale.maxCoeff();
            if (max_scale <= 0.0) {
                x_scale.setOnes();
            } else {
                const double scale_tol = 1e-10 * max_scale;
                for (Eigen::Index j = 0; j < p; ++j)
                    if (x_scale(j) < scale_tol) x_scale(j) = 1.0;
            }
        }

        const Eigen::MatrixXd X_s =
            (X.rowwise() - x_mean.transpose()).array().rowwise()
            / x_scale.transpose().array();
        XA.noalias() = X_s * TA_;
        XC.noalias() = X_s * TC_;
        yc = y.array() - y_mean;

        if (X_te != nullptr) {
            const Eigen::MatrixXd Xte_s =
                (X_te->rowwise() - x_mean.transpose()).array().rowwise()
                / x_scale.transpose().array();
            XA_te->noalias() = Xte_s * TA_;
            XC_te->noalias() = Xte_s * TC_;
        }
    }


    /**
     * @brief Profiles the free block of the partial objective:
     *        1/2||y - XA a - XC c||^2 + (eps/2)||c||^2:
     *        min_c ... = 1/2 || M^{1/2} (y - XA a) ||^2 ,
     *        M^{1/2} z = z - U_C diag(1 - sqrt(eps/(s^2+eps))) U_C^T z
     *
     *        via one (eps-independent) SVD of XC.
     *        Returns the profiled design:
     *        G = M^{1/2} XA, response yt = M^{1/2} y,
     *        and the kept SVD factors of XC for the later recovery of c.
     *
     * @param XA Penalized block (n x r).
     * @param XC Free block (n x (p - r)).
     * @param yc Centered response (n).
     * @param eps Null-space floor.
     * @param G Output profiled penalized design (n x r).
     * @param yt Output profiled response (n).
     * @param UC Output left singular vectors of XC (n x rC).
     * @param VC Output right singular vectors of XC ((p - r) x rC).
     * @param sC Output singular values of XC (rC).
     */
    void profile_contrasts(const Eigen::MatrixXd& XA,
                           const Eigen::MatrixXd& XC,
                           const Eigen::VectorXd& yc,
                           double eps,
                           Eigen::MatrixXd& G,
                           Eigen::VectorXd& yt,
                           Eigen::MatrixXd& UC,
                           Eigen::MatrixXd& VC,
                           Eigen::VectorXd& sC) const
    {
        if (XC.cols() == 0) {
            G  = XA;
            yt = yc;
            UC.resize(yc.size(), 0);
            VC.resize(0, 0);
            sC.resize(0);
            return;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svdC(
            XC, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd& s = svdC.singularValues();
        const double rank_tol = std::numeric_limits<double>::epsilon()
            * (s.size() > 0 ? s(0) : 0.0)
            * std::sqrt(static_cast<double>(XC.rows()));
        Eigen::Index rC = 0;
        for (Eigen::Index i = 0; i < s.size(); ++i)
            if (s(i) > rank_tol) ++rC; else break;

        UC = svdC.matrixU().leftCols(rC);
        VC = svdC.matrixV().leftCols(rC);
        sC = s.head(rC);

        Eigen::VectorXd shrink(rC);   // 1 - sqrt(eps/(s^2+eps))
        for (Eigen::Index i = 0; i < rC; ++i)
            shrink(i) = 1.0 - std::sqrt(eps / (sC(i) * sC(i) + eps));

        G  = XA - UC * (shrink.asDiagonal() * (UC.transpose() * XA));
        yt = yc - UC * (shrink.asDiagonal() * (UC.transpose() * yc));
    }


    /**
     * @brief Make grid of possible lambda2 values on an ascending geometric scale.
     *
     * @param lambda_max Maximum lambda2 value (grid anchor).
     * @param lambda_ratio Ratio between max and min lambda2 (grid range).
     * @param n_lambda Number of lambda2 values to generate.
     */
    void make_grid(double lambda_max, double lambda_ratio,
                   Eigen::Index n_lambda)
    {
        const double log_hi = std::log10(lambda_max);
        const double log_lo = std::log10(lambda_max / lambda_ratio);
        lambdas_.resize(n_lambda);
        if (n_lambda == 1) {
            lambdas_(0) = std::pow(10.0, 0.5 * (log_lo + log_hi));
            return;
        }
        for (Eigen::Index i = 0; i < n_lambda; ++i)
            lambdas_(i) = std::pow(
                10.0, log_lo + (log_hi - log_lo) * static_cast<double>(i)
                                / static_cast<double>(n_lambda - 1));
    }


    /**
     * @brief Aggregate the fold MSEs and select the best lambda2 indices (min and 1se).
     *
     * @param fold_mse Matrix of fold MSEs (K x n_folds).
     * @param n_folds Number of cross-validation folds.
     */
    void aggregate_and_select(const Eigen::MatrixXd& fold_mse, int n_folds)
    {
        const Eigen::Index K = fold_mse.rows();
        cv_mse_.setZero(K);
        cv_sem_.setZero(K);
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
            cv_sem_(k) = std::sqrt(var * inv_Fminus * inv_F);
        }

        Eigen::Index k_min = 0;
        double best = std::numeric_limits<double>::max();
        for (Eigen::Index k = 0; k < K; ++k)
            if (cv_mse_(k) < best) { best = cv_mse_(k); k_min = k; }
        idx_min_ = k_min;

        // Ascending grid: largest lambda2 within one SE = scan from the top.
        const double threshold = cv_mse_(k_min) + cv_sem_(k_min);
        Eigen::Index k_1se = k_min;
        for (Eigen::Index k = K - 1; k >= 0; --k) {
            if (cv_mse_(k) <= threshold) { k_1se = k; break; }
            if (k == 0) break;
        }
        idx_1se_ = k_1se;
    }


    /**
     * @brief Validate common parameters for the fit methods.
     *
     * @param X Input design matrix (n x p).
     * @param y Input response vector (n).
     * @param n_folds Number of cross-validation folds.
     * @param n_lambda Number of lambda2 values to generate.
     * @param lambda_ratio Ratio between max and min lambda2 (grid range).
     */
    static void validate_common(Eigen::Ref<const Eigen::MatrixXd> X,
                                Eigen::Ref<const Eigen::VectorXd> y,
                                int n_folds,
                                Eigen::Index n_lambda,
                                double lambda_ratio)
    {
        if (X.rows() != y.size())
            throw std::invalid_argument(
                "tikhonov_cv_svd::fit: X.rows() != y.size()");
        if (n_folds < 2)
            throw std::invalid_argument(
                "tikhonov_cv_svd::fit: n_folds must be >= 2");
        if (static_cast<Eigen::Index>(n_folds) > X.rows())
            throw std::invalid_argument(
                "tikhonov_cv_svd::fit: n_folds must be <= n_samples");
        if (n_lambda <= 0)
            throw std::invalid_argument(
                "tikhonov_cv_svd::fit: n_lambda must be positive");
        if (lambda_ratio <= 0.0)
            throw std::invalid_argument(
                "tikhonov_cv_svd::fit: lambda_ratio must be positive");
    }


    /**
     * @brief Check if the model has been fitted. If not, throw a runtime error.
     */
    void check_fitted() const {
        if (!fitted_)
            throw std::runtime_error(
                "tikhonov_cv_svd: call fit() before using this method");
    }


    // private member variables ----------------------------------------------
    /** @brief Flag indicating whether the model has been fitted. */
    bool fitted_ = false;

    /** @brief Number of predictors. */
    Eigen::Index p_ = 0;

    /** @brief Rank of K (M in group mode). */
    Eigen::Index r_ = 0;

    /** @brief Map for the penalized block (p x r): V_+ Lambda_+^{-1/2}. */
    Eigen::MatrixXd TA_;

    /** @brief Map for the free block (p x (p-r)): V_0. */
    Eigen::MatrixXd TC_;

    /** @brief Epsilon value used in the model. */
    double epsilon_used_  = 0.0;

    /** @brief Flag indicating whether the model is interpolating. */
    bool interpolating_ = false;

    /** @brief lambda2 grid (ascending) */
    Eigen::VectorXd lambdas_;

    /** @brief Cross-validated mean squared error for each lambda2. */
    Eigen::VectorXd cv_mse_;

    /** @brief Standard error of the cross-validated mean squared error. */
    Eigen::VectorXd cv_sem_;

    /** @brief Index of the lambda2 with the minimum cross-validated MSE. */
    Eigen::Index idx_min_ = 0;

    /** @brief Index of the largest lambda2 within one standard error of the minimum. */
    Eigen::Index idx_1se_ = 0;
};

// ===================================================================================
} /* End of namespace model_selection */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================

#endif /* End of TREX_ML_METHODS_MODEL_SELECTION_TIKHONOV_CV_SVD_HPP */
