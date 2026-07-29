// ==================================================================================
/**
 * @file test_ml_methods_model_selection.cpp
 *
 * @brief Unit tests for ML methods model selection: the coordinate-descent
 *        elastic-net engine (enet_gaussian), its CV wrapper (enet_cv_ccd), and
 *        the SVD-based ridge CV (ridge_cv_svd).
 */
// ==================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <cmath>
#include <random>
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// project ml_methods includes
#include <ml_methods/model_selection/enet_cv_ccd.hpp>
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/model_selection/ienet_cv_ccd.hpp>
#include <ml_methods/model_selection/tikhonov_cv_svd.hpp>

// Eigen sparse (Tikhonov K construction)
#include <Eigen/SparseCore>

// ==================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::model_selection {

using namespace trex::ml_methods::model_selection;

// ==================================================================================
// Helpers: deterministic data generation independent of global RNG state
// ==================================================================================

namespace {

Eigen::MatrixXd random_matrix(Eigen::Index n, Eigen::Index p, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    return Eigen::MatrixXd::NullaryExpr(
        n, p, [&](Eigen::Index, Eigen::Index) { return dist(gen); });
}

Eigen::VectorXd random_vector(Eigen::Index n, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    return Eigen::VectorXd::NullaryExpr(
        n, [&](Eigen::Index) { return dist(gen); });
}

/** @brief glmnet-style preprocessing replicated for reference computations. */
struct GlmnetScaling {
    Eigen::VectorXd x_mean;   // column means
    Eigen::VectorXd x_scale;  // column population SDs
    Eigen::MatrixXd Xs;       // centered + standardised predictors
    double          y_mean;
    double          y_scale;  // population SD of y
    Eigen::VectorXd ycs;      // centered response / y_scale
};

GlmnetScaling glmnet_preprocess(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    const Eigen::Index n = X.rows(), p = X.cols();
    GlmnetScaling s;
    s.x_mean = X.colwise().mean();
    Eigen::MatrixXd Xc = X.rowwise() - s.x_mean.transpose();
    s.x_scale.resize(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        s.x_scale(j) = std::sqrt(Xc.col(j).squaredNorm() / static_cast<double>(n));
    }
    s.Xs = Xc * s.x_scale.cwiseInverse().asDiagonal();
    s.y_mean = y.mean();
    Eigen::VectorXd yc = y.array() - s.y_mean;
    s.y_scale = std::sqrt(yc.squaredNorm() / static_cast<double>(n));
    s.ycs = yc / s.y_scale;
    return s;
}

} // namespace


// ==================================================================================
// enet_gaussian: coordinate-descent engine
// ==================================================================================

/**
 * @brief Ridge (alpha = 0) at a single lambda must match the closed-form solution
 *        of glmnet's standardised-space problem:
 *          beta_s = (Xs^T Xs / n + (lambda/y_scale) I)^{-1} Xs^T ycs / n
 *        back-scaled to the original parameterisation.
 */
TEST(EnetGaussianTest, RidgeClosedFormSingleLambda) {
    const Eigen::Index n = 20, p = 5;
    Eigen::MatrixXd X = random_matrix(n, p, 7);
    Eigen::VectorXd w(p);
    w << 2.0, -1.0, 0.5, 0.0, 1.5;
    Eigen::VectorXd y = X * w + 0.1 * random_vector(n, 8);

    const double lambda = 0.5;
    Eigen::VectorXd grid(1);
    grid << lambda;

    enet_gaussian en;
    en.fit(X, y, grid, /*alpha=*/0.0, /*standardize=*/true, /*intercept=*/true,
           /*use_strong_rule=*/false, /*max_iter=*/100000, /*tol=*/1e-12);
    ASSERT_TRUE(en.converged());

    // Closed-form reference
    const GlmnetScaling s = glmnet_preprocess(X, y);
    const double lam_int = lambda / s.y_scale;
    const Eigen::MatrixXd A =
        s.Xs.transpose() * s.Xs / static_cast<double>(n)
        + lam_int * Eigen::MatrixXd::Identity(p, p);
    const Eigen::VectorXd b_s =
        A.ldlt().solve(s.Xs.transpose() * s.ycs / static_cast<double>(n));
    const Eigen::VectorXd beta_ref =
        (b_s.array() * s.y_scale / s.x_scale.array()).matrix();
    const double a0_ref = s.y_mean - s.x_mean.dot(beta_ref);

    ASSERT_EQ(en.coef().cols(), 1);
    EXPECT_TRUE(en.coef().col(0).isApprox(beta_ref, 1e-6))
        << "CD ridge deviates from the closed-form solution.";
    EXPECT_NEAR(en.intercepts()(0), a0_ref, 1e-6);
}


/**
 * @brief Lasso path invariants: all coefficients are zero at lambda_max (by
 *        definition of the grid anchor), the model is non-trivial at the smallest
 *        lambda, and the deviance ratio is bounded and non-decreasing along the path.
 */
TEST(EnetGaussianTest, LassoPathStartsEmptyAndGrows) {
    const Eigen::Index n = 30, p = 8;
    Eigen::MatrixXd X = random_matrix(n, p, 11);
    Eigen::VectorXd y = 2.0 * X.col(0) - X.col(3) + 0.3 * random_vector(n, 12);

    enet_gaussian en;
    en.fit(X, y, /*alpha=*/1.0, /*n_lambda=*/50);

    const Eigen::Index K = en.lambdas().size();
    ASSERT_GE(K, 2);
    ASSERT_EQ(en.coef().cols(), K);

    // At the anchor lambda_max every coefficient is exactly soft-thresholded to 0.
    EXPECT_LT(en.coef().col(0).cwiseAbs().maxCoeff(), 1e-10);

    // At the smallest fitted lambda the true signal variables are active.
    EXPECT_GT(std::abs(en.coef()(0, K - 1)), 0.1);
    EXPECT_GT(std::abs(en.coef()(3, K - 1)), 0.1);

    // Deviance ratio: within [0, 1], non-decreasing as lambda decreases.
    const Eigen::VectorXd& dr = en.dev_ratio();
    ASSERT_EQ(dr.size(), K);
    for (Eigen::Index l = 0; l < K; ++l) {
        EXPECT_GE(dr(l), -1e-12);
        EXPECT_LE(dr(l), 1.0 + 1e-12);
        if (l > 0) EXPECT_GE(dr(l), dr(l - 1) - 1e-8);
    }
}


/**
 * @brief The lasso solution must satisfy the KKT optimality conditions in the
 *        standardised space: |x_j^T r / n| == lambda_int for active coordinates
 *        (with matching sign) and <= lambda_int for inactive ones.
 */
TEST(EnetGaussianTest, LassoKKTConditionsHold) {
    const Eigen::Index n = 30, p = 8;
    Eigen::MatrixXd X = random_matrix(n, p, 21);
    Eigen::VectorXd y = 2.0 * X.col(0) - X.col(3) + 0.3 * random_vector(n, 22);

    const GlmnetScaling s = glmnet_preprocess(X, y);

    // A mid-path lambda: 30% of the anchor
    const double lambda_max =
        (s.Xs.transpose() * s.ycs).cwiseAbs().maxCoeff()
        / static_cast<double>(n) * s.y_scale;
    const double lambda = 0.3 * lambda_max;

    Eigen::VectorXd grid(1);
    grid << lambda;
    enet_gaussian en;
    en.fit(X, y, grid, /*alpha=*/1.0, /*standardize=*/true, /*intercept=*/true,
           /*use_strong_rule=*/false, /*max_iter=*/100000, /*tol=*/1e-12);
    ASSERT_TRUE(en.converged());

    // Map the reported coefficients back to standardised units
    const Eigen::VectorXd b_s =
        (en.coef().col(0).array() * s.x_scale.array() / s.y_scale).matrix();
    const Eigen::VectorXd r = s.ycs - s.Xs * b_s;
    const Eigen::VectorXd g = s.Xs.transpose() * r / static_cast<double>(n);
    const double lam_int = lambda / s.y_scale;

    for (Eigen::Index j = 0; j < p; ++j) {
        if (std::abs(b_s(j)) > 1e-12) {
            EXPECT_NEAR(g(j), lam_int * (b_s(j) > 0 ? 1.0 : -1.0), 1e-6)
                << "Active coordinate " << j << " violates stationarity.";
        } else {
            EXPECT_LE(std::abs(g(j)), lam_int + 1e-6)
                << "Inactive coordinate " << j << " violates the KKT bound.";
        }
    }
}


/** @brief Input validation and the degenerate-response guard. */
TEST(EnetGaussianTest, ValidationAndDegenerateThrows) {
    Eigen::MatrixXd X = random_matrix(10, 3, 31);
    Eigen::VectorXd y = random_vector(10, 32);

    enet_gaussian en;
    // Dimension mismatch / bad alpha / empty explicit grid
    EXPECT_THROW(en.fit(X, random_vector(9, 33)), std::invalid_argument);
    EXPECT_THROW(en.fit(X, y, /*alpha=*/-0.1), std::invalid_argument);
    EXPECT_THROW(en.fit(X, y, /*alpha=*/1.1), std::invalid_argument);
    Eigen::VectorXd empty_grid(0);
    EXPECT_THROW(en.fit(X, y, empty_grid), std::invalid_argument);

    // Constant y: lambda grid anchor degenerates -> must throw, not emit NaNs.
    Eigen::VectorXd y_const = Eigen::VectorXd::Ones(10);
    EXPECT_THROW(en.fit(X, y_const), std::runtime_error);
}


// ==================================================================================
// ridge_cv_svd: SVD-based ridge cross-validation
// ==================================================================================

/**
 * @brief With a noiseless linear response and n >> p, regularisation only hurts:
 *        the CV-minimal lambda must be the smallest grid point (index 0, grid is
 *        ascending) and its CV-MSE must be orders of magnitude below the largest
 *        lambda's.
 */
TEST(RidgeCvSvdTest, NoiselessSignalSelectsSmallestLambda) {
    const Eigen::Index n = 40, p = 5;
    Eigen::MatrixXd X = random_matrix(n, p, 41);
    Eigen::VectorXd w(p);
    w << 1.0, -2.0, 0.5, 3.0, -1.5;
    Eigen::VectorXd y = X * w;  // exact linear signal, no noise

    ridge_cv_svd cv;
    cv.fit(X, y, /*n_folds=*/5, /*n_lambda=*/50, /*lambda_ratio=*/1e4, /*seed=*/1);

    const Eigen::Index K = cv.lambdas().size();
    ASSERT_EQ(K, 50);
    EXPECT_EQ(cv.index_min(), 0);
    EXPECT_LT(cv.cv_mse()(cv.index_min()), 1e-4 * cv.cv_mse()(K - 1));

    // lambda.1se is the largest lambda within one SE — never below lambda.min.
    EXPECT_GE(cv.cv_1se(), cv.cv_min());
    EXPECT_LE(cv.cv_mse()(cv.index_1se()),
              cv.cv_mse()(cv.index_min()) + cv.cv_sem()(cv.index_min()) + 1e-12);
}


/** @brief Grid structure: ascending, spans exactly lambda_ratio, SEM non-negative. */
TEST(RidgeCvSvdTest, GridPropertiesAndAccessors) {
    const Eigen::Index n = 30, p = 4;
    Eigen::MatrixXd X = random_matrix(n, p, 51);
    Eigen::VectorXd y = X * random_vector(p, 52) + 0.2 * random_vector(n, 53);

    const double ratio = 1000.0;
    ridge_cv_svd cv;
    cv.fit(X, y, /*n_folds=*/5, /*n_lambda=*/25, ratio, /*seed=*/0);

    const Eigen::VectorXd& lam = cv.lambdas();
    const Eigen::Index K = lam.size();
    ASSERT_EQ(K, 25);
    for (Eigen::Index k = 1; k < K; ++k) {
        EXPECT_GT(lam(k), lam(k - 1)) << "Grid is not strictly ascending.";
    }
    EXPECT_NEAR(lam(K - 1) / lam(0), ratio, ratio * 1e-9);

    ASSERT_EQ(cv.cv_mse().size(), K);
    ASSERT_EQ(cv.cv_sem().size(), K);
    for (Eigen::Index k = 0; k < K; ++k) {
        EXPECT_GE(cv.cv_mse()(k), 0.0);
        EXPECT_GE(cv.cv_sem()(k), 0.0);
    }
}


/** @brief Identical seeds must give identical folds and therefore identical results. */
TEST(RidgeCvSvdTest, DeterministicWithSeed) {
    const Eigen::Index n = 30, p = 4;
    Eigen::MatrixXd X = random_matrix(n, p, 61);
    Eigen::VectorXd y = X * random_vector(p, 62) + 0.3 * random_vector(n, 63);

    ridge_cv_svd cv1, cv2;
    cv1.fit(X, y, 5, 30, 1000.0, /*seed=*/7);
    cv2.fit(X, y, 5, 30, 1000.0, /*seed=*/7);

    EXPECT_TRUE(cv1.cv_mse().isApprox(cv2.cv_mse(), 0.0));
    EXPECT_EQ(cv1.index_min(), cv2.index_min());
    EXPECT_EQ(cv1.index_1se(), cv2.index_1se());
}


/** @brief Input validation, unfitted access, and the orthogonal-response guard. */
TEST(RidgeCvSvdTest, ValidationAndDegenerateThrows) {
    Eigen::MatrixXd X = random_matrix(10, 3, 71);
    Eigen::VectorXd y = random_vector(10, 72);

    ridge_cv_svd cv;
    EXPECT_THROW(cv.fit(X, random_vector(9, 73)), std::invalid_argument);
    EXPECT_THROW(cv.fit(X, y, /*n_folds=*/1), std::invalid_argument);
    EXPECT_THROW(cv.fit(X, y, /*n_folds=*/11), std::invalid_argument);
    EXPECT_THROW(cv.fit(X, y, 5, /*n_lambda=*/0), std::invalid_argument);
    EXPECT_THROW(cv.fit(X, y, 5, 10, /*lambda_ratio=*/0.0), std::invalid_argument);

    // Accessors before a successful fit
    EXPECT_THROW(cv.cv_min(), std::runtime_error);
    EXPECT_THROW(cv.cv_1se(), std::runtime_error);

    // y exactly orthogonal to every centered column -> c_max == 0 -> throw.
    Eigen::MatrixXd X_orth(4, 2);
    X_orth.col(0) << 1.0, 1.0, -1.0, -1.0;
    X_orth.col(1) << 1.0, -1.0, -1.0, 1.0;
    Eigen::VectorXd y_orth(4);
    y_orth << 1.0, -1.0, 1.0, -1.0;
    EXPECT_THROW(cv.fit(X_orth, y_orth, /*n_folds=*/2), std::runtime_error);
}


// ==================================================================================
// enet_cv_ccd: glmnet-faithful CV wrapper
// ==================================================================================

/**
 * @brief Noiseless ridge CV: the CV curve must reach a near-zero minimum, the grid
 *        is descending, and the 1-SE selection obeys glmnet semantics
 *        (largest lambda within one SE, hence lambda.1se >= lambda.min).
 */
TEST(EnetCvCcdTest, NoiselessRidgeStructure) {
    const Eigen::Index n = 40, p = 5;
    Eigen::MatrixXd X = random_matrix(n, p, 81);
    Eigen::VectorXd w(p);
    w << 1.0, -2.0, 0.5, 3.0, -1.5;
    Eigen::VectorXd y = X * w;  // exact linear signal

    enet_cv_ccd cv;
    cv.fit(X, y, /*alpha=*/0.0, /*n_folds=*/5, /*n_lambda=*/60);

    const Eigen::VectorXd& lam = cv.lambdas();
    const Eigen::Index K = lam.size();
    ASSERT_GE(K, 2);
    for (Eigen::Index k = 1; k < K; ++k) {
        EXPECT_LT(lam(k), lam(k - 1)) << "Grid is not strictly descending.";
    }

    // glmnet's ridge convention bounds how low the path descends: the grid anchor
    // uses alpha_eff = 1e-3 (very high lambda_max) and the devmax rule truncates
    // the path once dev.ratio exceeds 0.999 — so the minimum CV-MSE saturates
    // around 1e-3 * Var(y) even for a noiseless response.
    EXPECT_LT(cv.cv_mse().minCoeff(), 1e-2 * cv.cv_mse().maxCoeff());
    EXPECT_GE(cv.cv_1se(), cv.cv_min());
    EXPECT_LE(cv.cv_mse()(cv.index_1se()),
              cv.cv_mse()(cv.index_min()) + cv.cv_sem()(cv.index_min()) + 1e-12);
}


/** @brief Identical seeds must give identical folds and therefore identical results. */
TEST(EnetCvCcdTest, DeterministicWithSeed) {
    const Eigen::Index n = 30, p = 4;
    Eigen::MatrixXd X = random_matrix(n, p, 91);
    Eigen::VectorXd y = X * random_vector(p, 92) + 0.3 * random_vector(n, 93);

    enet_cv_ccd cv1, cv2;
    cv1.fit(X, y, 0.0, 5, 40, -1.0, /*seed=*/3);
    cv2.fit(X, y, 0.0, 5, 40, -1.0, /*seed=*/3);

    EXPECT_TRUE(cv1.cv_mse().isApprox(cv2.cv_mse(), 0.0));
    EXPECT_EQ(cv1.index_min(), cv2.index_min());
    EXPECT_EQ(cv1.index_1se(), cv2.index_1se());
}


/** @brief Input validation and unfitted access. */
TEST(EnetCvCcdTest, ValidationThrows) {
    Eigen::MatrixXd X = random_matrix(10, 3, 95);
    Eigen::VectorXd y = random_vector(10, 96);

    enet_cv_ccd cv;
    EXPECT_THROW(cv.fit(X, random_vector(9, 97)), std::invalid_argument);
    EXPECT_THROW(cv.fit(X, y, 0.0, /*n_folds=*/1), std::invalid_argument);
    EXPECT_THROW(cv.fit(X, y, 0.0, /*n_folds=*/11), std::invalid_argument);
    EXPECT_THROW(cv.fit(X, y, 0.0, 5, /*n_lambda=*/0), std::invalid_argument);
    EXPECT_THROW(cv.cv_min(), std::runtime_error);
    EXPECT_THROW(cv.cv_1se(), std::runtime_error);
}

// ==================================================================================
// tikhonov_cv_svd (generalized-Tikhonov ridge CV, IEN geometry)
// ==================================================================================

/** @brief All-singleton groups (K = I) must reproduce ridge_cv_svd exactly:
 *  identical grid anchor, folds (same seed), and CV curves — for BOTH the
 *  closed-form group basis and the general eigendecomposition path. */
TEST(TikhonovCvSvdTest, SingletonGroupsMatchRidgeCvSvd) {
    const Eigen::Index n = 60, p = 15;
    Eigen::MatrixXd X = random_matrix(n, p, 11);
    Eigen::VectorXd y = X.col(0) - 2.0 * X.col(3) + 0.5 * random_vector(n, 12);

    ridge_cv_svd ref;
    ref.fit(X, y, 5, 50, 1e4, 3);

    // Closed-form group basis (singleton groups).
    Eigen::VectorXi singleton = Eigen::VectorXi::LinSpaced(
        p, 0, static_cast<int>(p) - 1);
    tikhonov_cv_svd grp;
    grp.fit(X, y, singleton, 5, 50, 1e4, 3);

    // General eigendecomposition path (sparse identity K).
    Eigen::SparseMatrix<double> I(p, p);
    I.setIdentity();
    tikhonov_cv_svd gen;
    gen.fit(X, y, I, 5, 50, 1e4, 3);

    ASSERT_EQ(ref.lambdas().size(), grp.lambdas().size());
    for (Eigen::Index k = 0; k < ref.lambdas().size(); ++k) {
        EXPECT_NEAR(grp.lambdas()(k), ref.lambdas()(k),
                    1e-9 * ref.lambdas()(k));
        EXPECT_NEAR(grp.cv_mse()(k), ref.cv_mse()(k),
                    1e-7 * (1.0 + ref.cv_mse()(k)));
        EXPECT_NEAR(gen.cv_mse()(k), ref.cv_mse()(k),
                    1e-7 * (1.0 + ref.cv_mse()(k)));
    }
    EXPECT_EQ(grp.index_min(), ref.index_min());
    EXPECT_EQ(grp.index_1se(), ref.index_1se());
    EXPECT_EQ(gen.index_min(), ref.index_min());
}


/** @brief The closed-form group basis and the general sparse-K
 *  eigendecomposition path must coincide for a NON-trivial K: the group-mean
 *  matrix W = sum_m 1_m 1_m^T / p_m (rank M, null_dim p - M). Also pins the
 *  tall-regime diagnostics: no interpolation, zero null-space floor. */
TEST(TikhonovCvSvdTest, GroupSparseKMatchesGroupMode) {
    const Eigen::Index n = 60, p = 20;
    const int M = 5;                       // 5 groups of 4
    Eigen::MatrixXd X = random_matrix(n, p, 61);
    Eigen::VectorXd y = 2.0 * X.col(2) - X.col(9) + 0.5 * random_vector(n, 62);
    Eigen::VectorXi groups(p);
    for (Eigen::Index j = 0; j < p; ++j) groups(j) = static_cast<int>(j / 4);

    Eigen::SparseMatrix<double> W(p, p);
    {
        std::vector<Eigen::Triplet<double>> trip;
        for (Eigen::Index i = 0; i < p; ++i)
            for (Eigen::Index j = 0; j < p; ++j)
                if (groups(i) == groups(j))
                    trip.emplace_back(i, j, 0.25);   // 1 / p_m, p_m = 4
        W.setFromTriplets(trip.begin(), trip.end());
    }

    tikhonov_cv_svd grp, gen;
    grp.fit(X, y, groups, 5, 40, 1e4, 3);
    gen.fit(X, y, W, 5, 40, 1e4, 3);

    ASSERT_EQ(grp.lambdas().size(), gen.lambdas().size());
    for (Eigen::Index k = 0; k < grp.lambdas().size(); ++k) {
        EXPECT_NEAR(gen.lambdas()(k), grp.lambdas()(k),
                    1e-9 * grp.lambdas()(k));
        EXPECT_NEAR(gen.cv_mse()(k), grp.cv_mse()(k),
                    1e-7 * (1.0 + grp.cv_mse()(k)));
    }
    EXPECT_EQ(gen.index_min(), grp.index_min());
    EXPECT_EQ(gen.index_1se(), grp.index_1se());

    // Tall regime (n_train = 48 > null_dim = 15): informative curve.
    EXPECT_FALSE(grp.contrast_interpolates());
    EXPECT_EQ(grp.epsilon(), 0.0);
    EXPECT_EQ(grp.null_dim(), p - M);
    EXPECT_EQ(gen.null_dim(), p - M);
}


/** @brief Malformed inputs and pre-fit accessors must throw. */
TEST(TikhonovCvSvdTest, ValidationAndDegenerateThrows) {
    const Eigen::Index n = 30, p = 8;
    Eigen::MatrixXd X = random_matrix(n, p, 71);
    Eigen::VectorXd y = random_vector(n, 72);

    // Pre-fit accessors.
    tikhonov_cv_svd unfitted;
    EXPECT_THROW(unfitted.cv_min(), std::runtime_error);
    EXPECT_THROW(unfitted.cv_1se(), std::runtime_error);

    // Non-contiguous group ids.
    Eigen::VectorXi bad_groups = Eigen::VectorXi::Zero(p);
    bad_groups(p - 1) = 2;                 // ids {0, 2}: 1 is missing
    tikhonov_cv_svd cv;
    EXPECT_THROW(cv.fit(X, y, bad_groups, 5, 20, 1e4, 3),
                 std::invalid_argument);

    // Dimension-mismatched K.
    Eigen::SparseMatrix<double> K_bad(p + 1, p + 1);
    K_bad.setIdentity();
    EXPECT_THROW(cv.fit(X, y, K_bad, 5, 20, 1e4, 3), std::invalid_argument);

    // Non-PSD K (negative identity).
    Eigen::SparseMatrix<double> K_neg(p, p);
    K_neg.setIdentity();
    K_neg *= -1.0;
    EXPECT_THROW(cv.fit(X, y, K_neg, 5, 20, 1e4, 3), std::invalid_argument);
}


/** @brief In the wide regime p - M >= n_train the unpenalized contrast block
 *  interpolates the training folds: the class must flag it (and auto-select
 *  a positive null-space floor). */
TEST(TikhonovCvSvdTest, InterpolationFlagInWideRegime) {
    const Eigen::Index n = 30, p = 60;
    Eigen::MatrixXd X = random_matrix(n, p, 21);
    Eigen::VectorXd y = X.col(0) + 0.5 * random_vector(n, 22);

    Eigen::VectorXi groups(p);           // 6 groups of 10 -> null_dim = 54
    for (Eigen::Index j = 0; j < p; ++j)
        groups(j) = static_cast<int>(j / 10);

    tikhonov_cv_svd cv;
    cv.fit(X, y, groups, 5, 30, 1e4, 3);
    EXPECT_TRUE(cv.contrast_interpolates());
    EXPECT_GT(cv.epsilon(), 0.0);
    EXPECT_EQ(cv.null_dim(), p - 6);
}

// ==================================================================================
// ien_gaussian (CCD IEN path engine)
// ==================================================================================

/** @brief Solutions must satisfy the IEN KKT conditions on the standardized
 *  solver scale: |x_j^T r - lambda2 sigma_m / p_m| <= lambda1 on the
 *  non-support, == lambda1 * sign(beta_j) on the support. */
TEST(IenGaussianTest, KktConditionsHoldGroupMode) {
    const Eigen::Index n = 50, p = 20;
    Eigen::MatrixXd X = random_matrix(n, p, 31);
    Eigen::VectorXd y = 3.0 * X.col(0) - 2.0 * X.col(5)
                        + 0.5 * random_vector(n, 32);
    Eigen::VectorXi groups(p);           // 5 groups of 4
    for (Eigen::Index j = 0; j < p; ++j) groups(j) = static_cast<int>(j / 4);
    const double lambda2 = 0.7;

    Eigen::VectorXd grid(3);
    grid << 2.0, 1.0, 0.5;

    ien_gaussian en;
    en.fit(X, y, lambda2, groups, grid);
    ASSERT_TRUE(en.converged());

    // Standardized replica (center + column-L2-normalize X, center y).
    Eigen::MatrixXd Xs = X.rowwise() - X.colwise().mean();
    for (Eigen::Index j = 0; j < p; ++j) Xs.col(j) /= Xs.col(j).norm();
    Eigen::VectorXd yc = y.array() - y.mean();

    for (Eigen::Index l = 0; l < grid.size(); ++l) {
        const double l1 = en.lambda1s()(l);
        const Eigen::VectorXd beta = en.coef_std().col(l);
        const Eigen::VectorXd r = yc - Xs * beta;

        Eigen::VectorXd sigma = Eigen::VectorXd::Zero(5);
        Eigen::VectorXd pm = Eigen::VectorXd::Zero(5);
        for (Eigen::Index j = 0; j < p; ++j) {
            sigma(groups(j)) += beta(j);
            pm(groups(j)) += 1.0;
        }

        const double tol = 1e-4 * std::max(1.0, l1);
        for (Eigen::Index j = 0; j < p; ++j) {
            const double c_j = Xs.col(j).dot(r)
                - lambda2 * sigma(groups(j)) / pm(groups(j));
            if (beta(j) == 0.0) {
                EXPECT_LE(std::abs(c_j), l1 + tol)
                    << "KKT violated on non-support at (l=" << l
                    << ", j=" << j << ")";
            } else {
                EXPECT_NEAR(c_j, l1 * (beta(j) > 0.0 ? 1.0 : -1.0), tol)
                    << "KKT violated on support at (l=" << l
                    << ", j=" << j << ")";
            }
        }
    }
}


/** @brief The O(1) group-sum specialization and the matrix-passing sparse-K
 *  path must produce identical coefficient paths for K = W. */
TEST(IenGaussianTest, GroupMatchesSparseW) {
    const Eigen::Index n = 50, p = 20;
    Eigen::MatrixXd X = random_matrix(n, p, 41);
    Eigen::VectorXd y = 2.0 * X.col(1) - X.col(7) + 0.5 * random_vector(n, 42);
    Eigen::VectorXi groups(p);
    for (Eigen::Index j = 0; j < p; ++j) groups(j) = static_cast<int>(j / 4);

    // W = sum_m 1_m 1_m^T / p_m as an explicit sparse matrix.
    Eigen::SparseMatrix<double> W(p, p);
    {
        std::vector<Eigen::Triplet<double>> trip;
        for (Eigen::Index i = 0; i < p; ++i)
            for (Eigen::Index j = 0; j < p; ++j)
                if (groups(i) == groups(j))
                    trip.emplace_back(i, j, 0.25);   // 1 / p_m, p_m = 4
        W.setFromTriplets(trip.begin(), trip.end());
    }

    Eigen::VectorXd grid(4);
    grid << 3.0, 1.5, 0.8, 0.4;

    ien_gaussian grp, spk;
    grp.fit(X, y, 0.9, groups, grid);
    spk.fit(X, y, 0.9, W, grid);

    EXPECT_TRUE(grp.coef().isApprox(spk.coef(), 1e-8));
    EXPECT_TRUE(grp.intercepts().isApprox(spk.intercepts(), 1e-8));
}

// ==================================================================================
// ienet_cv_ccd (2D CV with lambda1 profiled out)
// ==================================================================================

/** @brief Structural coherence and seed determinism of the profiled CV. */
TEST(IenetCvCcdTest, StructureAndDeterminism) {
    const Eigen::Index n = 60, p = 24;
    Eigen::MatrixXd X = random_matrix(n, p, 51);
    Eigen::VectorXd y = 2.0 * X.col(0) + X.col(1) - X.col(6)
                        + 0.5 * random_vector(n, 52);
    Eigen::VectorXi groups(p);
    for (Eigen::Index j = 0; j < p; ++j) groups(j) = static_cast<int>(j / 3);

    ienet_cv_ccd cv;
    cv.fit(X, y, groups, 5, 10, 1e3, 30, -1.0, 9);

    // Grids: lambda2 ascending, lambda1 descending; surface dimensions.
    ASSERT_EQ(cv.lambdas().size(), 10);
    ASSERT_EQ(cv.lambda1s().size(), 30);
    for (Eigen::Index k = 1; k < cv.lambdas().size(); ++k)
        EXPECT_GT(cv.lambdas()(k), cv.lambdas()(k - 1));
    for (Eigen::Index k = 1; k < cv.lambda1s().size(); ++k)
        EXPECT_LT(cv.lambda1s()(k), cv.lambda1s()(k - 1));
    EXPECT_EQ(cv.cv_mse_full().rows(), 10);
    EXPECT_EQ(cv.cv_mse_full().cols(), 30);

    // Profiled curve is the row-min of the surface; picks are on-grid.
    for (Eigen::Index k = 0; k < 10; ++k)
        EXPECT_NEAR(cv.cv_mse()(k), cv.cv_mse_full().row(k).minCoeff(),
                    1e-12);
    EXPECT_GT(cv.cv_min(), 0.0);
    EXPECT_GE(cv.cv_1se(), cv.cv_min());
    EXPECT_GE(cv.lambda1_at_min(), cv.lambda1s()(cv.lambda1s().size() - 1));
    EXPECT_LE(cv.lambda1_at_min(), cv.lambda1s()(0));

    // Same seed => identical curves.
    ienet_cv_ccd cv2;
    cv2.fit(X, y, groups, 5, 10, 1e3, 30, -1.0, 9);
    EXPECT_TRUE(cv.cv_mse().isApprox(cv2.cv_mse(), 0.0));
    EXPECT_EQ(cv.index_min(), cv2.index_min());
    EXPECT_EQ(cv.index_1se(), cv2.index_1se());
}


/** @brief The sparse-K overload with the group-mean matrix W must reproduce
 *  the O(1) group-sum specialization: identical grids, CV surface, profiled
 *  curve, and picks (same folds via the same seed). Also pins the lambda1
 *  profile accessors to the surface argmin. */
TEST(IenetCvCcdTest, SparseKOverloadMatchesGroupMode) {
    const Eigen::Index n = 60, p = 20;
    Eigen::MatrixXd X = random_matrix(n, p, 81);
    Eigen::VectorXd y = 2.0 * X.col(0) - X.col(5) + 0.5 * random_vector(n, 82);
    Eigen::VectorXi groups(p);
    for (Eigen::Index j = 0; j < p; ++j) groups(j) = static_cast<int>(j / 4);

    Eigen::SparseMatrix<double> W(p, p);
    {
        std::vector<Eigen::Triplet<double>> trip;
        for (Eigen::Index i = 0; i < p; ++i)
            for (Eigen::Index j = 0; j < p; ++j)
                if (groups(i) == groups(j))
                    trip.emplace_back(i, j, 0.25);   // 1 / p_m, p_m = 4
        W.setFromTriplets(trip.begin(), trip.end());
    }

    ienet_cv_ccd grp, spk;
    grp.fit(X, y, groups, 4, 8, 1e3, 25, -1.0, 9);
    spk.fit(X, y, W, 4, 8, 1e3, 25, -1.0, 9);

    EXPECT_TRUE(grp.lambdas().isApprox(spk.lambdas(), 1e-12));
    EXPECT_TRUE(grp.lambda1s().isApprox(spk.lambda1s(), 1e-12));
    EXPECT_TRUE(grp.cv_mse_full().isApprox(spk.cv_mse_full(), 1e-6));
    EXPECT_TRUE(grp.cv_mse().isApprox(spk.cv_mse(), 1e-6));
    EXPECT_EQ(grp.index_min(), spk.index_min());
    EXPECT_EQ(grp.index_1se(), spk.index_1se());

    // lambda1 profile: one on-grid lambda1 per lambda2, equal to the row
    // argmin of the CV surface; the at-min / at-1se accessors read it.
    ASSERT_EQ(grp.lambda1_profile().size(), grp.lambdas().size());
    for (Eigen::Index k = 0; k < grp.lambdas().size(); ++k) {
        Eigen::Index amin = 0;
        grp.cv_mse_full().row(k).minCoeff(&amin);
        EXPECT_DOUBLE_EQ(grp.lambda1_profile()(k), grp.lambda1s()(amin));
    }
    EXPECT_DOUBLE_EQ(grp.lambda1_at_min(),
                     grp.lambda1_profile()(grp.index_min()));
    EXPECT_DOUBLE_EQ(grp.lambda1_at_1se(),
                     grp.lambda1_profile()(grp.index_1se()));
}


/** @brief Malformed inputs and pre-fit accessors must throw; the pathwise
 *  engine rejects non-positive lambda1 grid entries (lambda1 = 0 is the
 *  tikhonov_cv_svd backbone, not an IEN path point). */
TEST(IenetCvCcdTest, ValidationThrows) {
    const Eigen::Index n = 30, p = 8;
    Eigen::MatrixXd X = random_matrix(n, p, 91);
    Eigen::VectorXd y = random_vector(n, 92);
    Eigen::VectorXi groups(p);
    for (Eigen::Index j = 0; j < p; ++j) groups(j) = static_cast<int>(j / 2);

    // Pre-fit accessors.
    ienet_cv_ccd unfitted;
    EXPECT_THROW(unfitted.cv_min(), std::runtime_error);
    EXPECT_THROW(unfitted.cv_1se(), std::runtime_error);

    // Non-contiguous group ids.
    Eigen::VectorXi bad_groups = Eigen::VectorXi::Zero(p);
    bad_groups(p - 1) = 2;                 // ids {0, 2}: 1 is missing
    ienet_cv_ccd cv;
    EXPECT_THROW(cv.fit(X, y, bad_groups, 4, 5, 1e3, 10, -1.0, 9),
                 std::invalid_argument);

    // Degenerate fold counts.
    EXPECT_THROW(cv.fit(X, y, groups, 1, 5, 1e3, 10, -1.0, 9),
                 std::invalid_argument);

    // Path engine: a grid containing lambda1 <= 0 is rejected.
    ien_gaussian en;
    Eigen::VectorXd bad_grid(2);
    bad_grid << 1.0, 0.0;
    EXPECT_THROW(en.fit(X, y, 0.5, groups, bad_grid), std::invalid_argument);
}

// ==================================================================================
} /* End of namespace trex::test::ml_methods::model_selection */
// ==================================================================================
