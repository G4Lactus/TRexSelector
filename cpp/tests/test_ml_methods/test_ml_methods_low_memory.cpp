// ==================================================================================
/**
 * @file test_ml_methods_low_memory.cpp
 *
 * @brief The raw-X (low-memory) elastic-net engine against the copying engine.
 *
 * @details Same estimator, same grid, two arithmetic routes: the copying
 *          engine centres and standardises X into a working copy, the raw
 *          engine reads X and carries the centring as a scalar residual
 *          offset. They agree to rounding; the tolerances below are far
 *          above rounding and far below anything a model would notice.
 */
// ==================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <cmath>
#include <random>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// project ml_methods includes
#include <ml_methods/model_selection/enet_cv_ccd.hpp>

// ==================================================================================

namespace trex::test::ml_methods::low_memory {

using trex::ml_methods::model_selection::enet_cv_ccd;
using trex::ml_methods::model_selection::enet_gaussian;

namespace {

Eigen::MatrixXd random_matrix(Eigen::Index n, Eigen::Index p, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    return Eigen::MatrixXd::NullaryExpr(
        n, p, [&](Eigen::Index, Eigen::Index) { return dist(gen); });
}

struct Shape { const char* name; Eigen::Index n, p; };
const Shape kShapes[] = {{"tall", 300, 25}, {"wide", 50, 400}, {"square", 100, 100}};

void make_data(const Shape& s, Eigen::MatrixXd& X, Eigen::VectorXd& y, bool big_means = false) {
    X = random_matrix(s.n, s.p, 100u + static_cast<unsigned>(s.p));
    if (big_means) {
        // columns with means far larger than their spread: the case where a
        // naive raw-X dot product would cancel catastrophically
        for (Eigen::Index j = 0; j < s.p; ++j) X.col(j).array() += 1000.0 * (1.0 + static_cast<double>(j % 7));
    }
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(s.p);
    beta(0) = 2.0; beta(3) = -1.5; beta(std::min<Eigen::Index>(7, s.p - 1)) = 1.0;
    std::mt19937 gen(9u);
    std::normal_distribution<double> dist(0.0, 1.0);
    y = X * beta;
    for (Eigen::Index i = 0; i < s.n; ++i) y(i) += 0.5 * dist(gen);
}

double rel_max_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    const double scale = std::max(1e-300, std::max(a.cwiseAbs().maxCoeff(), b.cwiseAbs().maxCoeff()));
    return (a - b).cwiseAbs().maxCoeff() / scale;
}

} // namespace

class LowMemoryEngine : public ::testing::TestWithParam<Shape> {};

/** @brief Auto-grid path: same grid, same coefficients to 1e-8 relative,
 *         same intercepts, same deviance ratios. */
TEST_P(LowMemoryEngine, PathAgreesWithCopyEngine) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    for (const double alpha : {1.0, 0.5, 0.0}) {
        for (const bool standardize : {true, false}) {
            for (const bool intercept : {true, false}) {
                enet_gaussian copy;      copy.fit(X, y, alpha, 30, -1.0, standardize, intercept, false, 100000, 1e-9);
                enet_gaussian raw(true); raw.fit(X, y, alpha, 30, -1.0, standardize, intercept, false, 100000, 1e-9);
                ASSERT_EQ(copy.lambdas().size(), raw.lambdas().size()) << GetParam().name;
                EXPECT_LT(rel_max_diff(copy.lambdas(), raw.lambdas()), 1e-12);
                EXPECT_LT(rel_max_diff(copy.coef(), raw.coef()), 1e-7)
                    << GetParam().name << " alpha " << alpha << " std " << standardize << " int " << intercept;
                EXPECT_LT(rel_max_diff(copy.intercepts(), raw.intercepts()), 1e-7);
                EXPECT_LT((copy.dev_ratio() - raw.dev_ratio()).cwiseAbs().maxCoeff(), 1e-8);
            }
        }
    }
}

/** @brief Explicit grid with the strong rule: same active sets. */
TEST_P(LowMemoryEngine, StrongRuleGridAgrees) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    enet_gaussian ref; ref.fit(X, y, 0.8, 25);
    const Eigen::VectorXd grid = ref.lambdas();
    enet_gaussian copy;      copy.fit(X, y, grid, 0.8, true, true, /*strong=*/true, 100000, 1e-9);
    enet_gaussian raw(true); raw.fit(X, y, grid, 0.8, true, true, /*strong=*/true, 100000, 1e-9);
    EXPECT_LT(rel_max_diff(copy.coef(), raw.coef()), 1e-7);
    for (Eigen::Index l = 0; l < grid.size(); ++l) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            const bool a = std::abs(copy.coef()(j, l)) > 1e-9;
            const bool b = std::abs(raw.coef()(j, l))  > 1e-9;
            EXPECT_EQ(a, b) << "active set differs at lambda " << l << " variable " << j;
        }
    }
}

/** @brief Columns with large means: the folded offset keeps the raw engine
 *         accurate (the naive raw dot product would lose the signal). */
TEST_P(LowMemoryEngine, LargeColumnMeansStayAccurate) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y, /*big_means=*/true);
    enet_gaussian copy;      copy.fit(X, y, 1.0, 30, -1.0, true, true, false, 100000, 1e-9);
    enet_gaussian raw(true); raw.fit(X, y, 1.0, 30, -1.0, true, true, false, 100000, 1e-9);
    ASSERT_EQ(copy.lambdas().size(), raw.lambdas().size());
    EXPECT_LT(rel_max_diff(copy.coef(), raw.coef()), 1e-6);
    EXPECT_LT(rel_max_diff(copy.intercepts(), raw.intercepts()), 1e-6);
}

/** @brief Masked fit equals the copy engine on the gathered rows;
 *         predict_rows equals predict on the gathered held-out rows. */
TEST_P(LowMemoryEngine, MaskedFitEqualsGatheredFit) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    const Eigen::Index n = X.rows();
    std::vector<Eigen::Index> test_idx, train_idx;
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (i % 5 == 2) { test_idx.push_back(i); w(i) = 0.0; } else { train_idx.push_back(i); }
    }
    Eigen::MatrixXd X_tr(static_cast<Eigen::Index>(train_idx.size()), X.cols());
    Eigen::VectorXd y_tr(static_cast<Eigen::Index>(train_idx.size()));
    for (std::size_t t = 0; t < train_idx.size(); ++t) {
        X_tr.row(static_cast<Eigen::Index>(t)) = X.row(train_idx[t]);
        y_tr(static_cast<Eigen::Index>(t)) = y(train_idx[t]);
    }
    Eigen::MatrixXd X_te(static_cast<Eigen::Index>(test_idx.size()), X.cols());
    for (std::size_t t = 0; t < test_idx.size(); ++t) X_te.row(static_cast<Eigen::Index>(t)) = X.row(test_idx[t]);

    enet_gaussian ref; ref.fit(X, y, 0.5, 20);
    const Eigen::VectorXd grid = ref.lambdas();
    enet_gaussian copy; copy.fit(X_tr, y_tr, grid, 0.5, true, true, false, 100000, 1e-9);
    enet_gaussian raw(true); raw.fit_masked(X, y, w, grid, 0.5, true, true, false, 100000, 1e-9);
    EXPECT_LT(rel_max_diff(copy.coef(), raw.coef()), 1e-7);
    EXPECT_LT(rel_max_diff(copy.intercepts(), raw.intercepts()), 1e-7);
    EXPECT_LT(rel_max_diff(copy.predict(X_te), raw.predict_rows(X, test_idx)), 1e-7);
    // predict_rows is the same arithmetic as predict on the gathered rows
    EXPECT_LT(rel_max_diff(raw.predict(X_te), raw.predict_rows(X, test_idx)), 1e-12);
}

/** @brief CV: same curve to 1e-7 relative, same lambda picks. */
TEST_P(LowMemoryEngine, CVAgreesWithCopyEngine) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    enet_cv_ccd copy;      copy.fit(X, y, 0.5, 5, 30, -1.0, 3u, true, true, 100000, 1e-9);
    enet_cv_ccd raw(true); raw.fit(X, y, 0.5, 5, 30, -1.0, 3u, true, true, 100000, 1e-9);
    ASSERT_EQ(copy.lambdas().size(), raw.lambdas().size());
    EXPECT_LT(rel_max_diff(copy.cv_mse(), raw.cv_mse()), 1e-7);
    EXPECT_LT(rel_max_diff(copy.cv_sem(), raw.cv_sem()), 1e-6);
    EXPECT_EQ(copy.index_min(), raw.index_min());
    EXPECT_EQ(copy.index_1se(), raw.index_1se());
}

/** @brief The raw engine reads a const matrix (a read-only view). */
TEST(LowMemoryEngineConst, FitsAConstMatrix) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(kShapes[0], X, y);
    const Eigen::MatrixXd Xc = X;
    const Eigen::Map<const Eigen::MatrixXd> Xm(Xc.data(), Xc.rows(), Xc.cols());
    enet_gaussian raw(true);
    EXPECT_NO_THROW(raw.fit(Xm, y, 1.0, 20));
    EXPECT_EQ((X - Xc).cwiseAbs().maxCoeff(), 0.0);
}

INSTANTIATE_TEST_SUITE_P(Shapes, LowMemoryEngine, ::testing::ValuesIn(kShapes),
                         [](const ::testing::TestParamInfo<Shape>& info) {
                             return std::string(info.param.name);
                         });

} // namespace trex::test::ml_methods::low_memory


// ==================================================================================
// Ridge CV: the kernel (dual) fold engine against the SVD route
// ==================================================================================

#include <ml_methods/model_selection/ridge_cv_svd.hpp>

namespace trex::test::ml_methods::low_memory {

using trex::ml_methods::model_selection::ridge_cv_svd;

class LowMemoryRidge : public ::testing::TestWithParam<Shape> {};

/** @brief Same grid, CV curve to 1e-7 relative, same picks, same
 *         interpolation flag. */
TEST_P(LowMemoryRidge, CVAgreesWithSVDRoute) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    ridge_cv_svd svd;          svd.fit(X, y, 5, 60, 1000.0, 11u);
    ridge_cv_svd kernel(true); kernel.fit(X, y, 5, 60, 1000.0, 11u);
    ASSERT_EQ(svd.lambdas().size(), kernel.lambdas().size());
    EXPECT_LT(rel_max_diff(svd.lambdas(), kernel.lambdas()), 1e-14);   // built the same way
    EXPECT_LT(rel_max_diff(svd.cv_mse(), kernel.cv_mse()), 1e-7) << GetParam().name;
    EXPECT_LT(rel_max_diff(svd.cv_sem(), kernel.cv_sem()), 1e-6) << GetParam().name;
    EXPECT_EQ(svd.index_min(), kernel.index_min()) << GetParam().name;
    EXPECT_EQ(svd.index_1se(), kernel.index_1se()) << GetParam().name;
    EXPECT_EQ(svd.interpolates(), kernel.interpolates());
}

/** @brief Large column means: the two-pass fold statistics stay accurate. */
TEST_P(LowMemoryRidge, LargeColumnMeansAgree) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y, /*big_means=*/true);
    ridge_cv_svd svd;          svd.fit(X, y, 5, 40, 1000.0, 5u);
    ridge_cv_svd kernel(true); kernel.fit(X, y, 5, 40, 1000.0, 5u);
    EXPECT_LT(rel_max_diff(svd.cv_mse(), kernel.cv_mse()), 1e-6) << GetParam().name;
    EXPECT_EQ(svd.index_min(), kernel.index_min());
}

/** @brief The kernel engine reads a const matrix. */
TEST(LowMemoryRidgeConst, FitsAConstMatrix) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(kShapes[1], X, y);
    const Eigen::MatrixXd Xc = X;
    const Eigen::Map<const Eigen::MatrixXd> Xm(Xc.data(), Xc.rows(), Xc.cols());
    ridge_cv_svd kernel(true);
    EXPECT_NO_THROW(kernel.fit(Xm, y, 5, 30, 1000.0, 2u));
    EXPECT_EQ((X - Xc).cwiseAbs().maxCoeff(), 0.0);
}

INSTANTIATE_TEST_SUITE_P(Shapes, LowMemoryRidge, ::testing::ValuesIn(kShapes),
                         [](const ::testing::TestParamInfo<Shape>& info) {
                             return std::string(info.param.name);
                         });

} // namespace trex::test::ml_methods::low_memory
