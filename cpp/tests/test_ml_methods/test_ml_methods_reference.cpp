// ==================================================================================
/**
 * @file test_ml_methods_reference.cpp
 *
 * @brief Bit-identity harness for the zero-copy work on the ML methods.
 *
 * @details Every ML core is run on fixed synthetic data of three shapes
 *          (tall, wide, square) and every output is compared byte for byte
 *          against a recorded reference. The reference lives OUTSIDE the
 *          repository (it is platform- and BLAS-specific):
 *
 *            TREX_ML_REF_DIR=<dir>  — compare against <dir>; a case whose
 *                                     file does not exist yet is recorded
 *                                     (and reported), so the first run on a
 *                                     tree records, every later run compares.
 *            unset                  — the tests are skipped.
 *
 *          Changes at the binding boundary and copy fusions must keep
 *          every case identical; an algebraic rewrite re-records after
 *          its tolerance has been checked and reported.
 */
// ==================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// project ml_methods includes
#include <ml_methods/model_selection/enet_cv_ccd.hpp>
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/svd/svd.hpp>
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// ==================================================================================

namespace trex::test::ml_methods::reference {

namespace fs = std::filesystem;
using trex::ml_methods::model_selection::enet_cv_ccd;
using trex::ml_methods::model_selection::enet_gaussian;
using trex::ml_methods::model_selection::ridge_cv_svd;
using trex::ml_methods::pca::PCA;
using trex::ml_methods::pca::PCAResult;
using trex::ml_methods::ridge::RidgeSolver;
using trex::ml_methods::scaler_methods::LpNormScaler;
using trex::ml_methods::scaler_methods::ZScoreScaler;
using trex::ml_methods::svd::SVDResult;
using trex::ml_methods::svd::SVDSolver;

// ==================================================================================
// Fixed data
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
    return Eigen::VectorXd::NullaryExpr(n, [&](Eigen::Index) { return dist(gen); });
}

struct Shape {
    const char* name;
    Eigen::Index n, p;
};

const Shape kShapes[] = {
    {"tall",   400, 30},
    {"wide",    60, 900},
    {"square", 120, 120},
};

/** @brief y = X beta + noise with a sparse beta, deterministic per shape. */
void make_data(const Shape& s, Eigen::MatrixXd& X, Eigen::VectorXd& y) {
    X = random_matrix(s.n, s.p, 1000u + static_cast<unsigned>(s.p));
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(s.p);
    beta(0) = 2.0; beta(3) = -1.5; beta(std::min<Eigen::Index>(7, s.p - 1)) = 1.0;
    y = X * beta + 0.5 * random_vector(s.n, 77u + static_cast<unsigned>(s.n));
}

// ==================================================================================
// The recorded byte stream of one case
// ==================================================================================

class Stream {
public:
    /** @brief Any dense matrix, vector or expression, column-major order. */
    template <class Derived>
    void add(const char* tag, const Eigen::MatrixBase<Derived>& expr) {
        const Eigen::MatrixXd m = expr;
        put_tag(tag);
        put_i64(m.rows()); put_i64(m.cols());
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
            for (Eigen::Index i = 0; i < m.rows(); ++i) { put_f64(m(i, j)); }
        }
    }
    void add(const char* tag, double x) { put_tag(tag); put_i64(1); put_i64(1); put_f64(x); }
    void add(const char* tag, std::int64_t x) { put_tag(tag); put_i64(x); }
    void add(const char* tag, bool x) { put_tag(tag); put_i64(x ? 1 : 0); }
    void add(const char* tag, const std::vector<std::size_t>& v) {
        put_tag(tag); put_i64(static_cast<std::int64_t>(v.size()));
        for (auto x : v) { put_i64(static_cast<std::int64_t>(x)); }
    }
    const std::vector<unsigned char>& bytes() const { return bytes_; }

private:
    void put_tag(const char* tag) {
        const auto len = std::strlen(tag);
        bytes_.insert(bytes_.end(), tag, tag + len);
        bytes_.push_back(0);
    }
    void put_i64(std::int64_t x) {
        unsigned char b[8]; std::memcpy(b, &x, 8); bytes_.insert(bytes_.end(), b, b + 8);
    }
    void put_f64(double x) {
        unsigned char b[8]; std::memcpy(b, &x, 8); bytes_.insert(bytes_.end(), b, b + 8);
    }
    std::vector<unsigned char> bytes_;
};

/** @brief Compare against <dir>/<case>.bin, or record it if absent. */
void compare_or_record(const std::string& case_name, const Stream& s) {
    const char* dir = std::getenv("TREX_ML_REF_DIR");
    if (dir == nullptr || *dir == '\0') {
        GTEST_SKIP() << "TREX_ML_REF_DIR not set: bit-identity reference not checked";
    }
    fs::create_directories(dir);
    const fs::path file = fs::path(dir) / (case_name + ".bin");
    if (!fs::exists(file)) {
        std::ofstream out(file, std::ios::binary);
        out.write(reinterpret_cast<const char*>(s.bytes().data()),
                  static_cast<std::streamsize>(s.bytes().size()));
        std::cout << "[ RECORDED ] " << file << " (" << s.bytes().size() << " bytes)\n";
        return;
    }
    std::ifstream in(file, std::ios::binary);
    std::vector<unsigned char> ref((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    ASSERT_EQ(ref.size(), s.bytes().size()) << case_name << ": byte length differs from " << file;
    // First differing offset, for the report
    std::size_t first_diff = ref.size();
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != s.bytes()[i]) { first_diff = i; break; }
    }
    EXPECT_EQ(first_diff, ref.size())
        << case_name << ": output differs from the recorded reference " << file
        << " at byte " << first_diff << " (binding-boundary and copy-fusion "
        << "changes must be bit-identical; an algebraic rewrite re-records "
        << "after its tolerance report)";
}

} // namespace

// ==================================================================================
// Cases
// ==================================================================================

class MLReference : public ::testing::TestWithParam<Shape> {};

TEST_P(MLReference, RidgeCV) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    ridge_cv_svd cv;
    cv.fit(X, y, /*n_folds=*/5, /*n_lambda=*/60, /*lambda_ratio=*/1000.0, /*seed=*/11u);
    Stream s;
    s.add("lambdas", cv.lambdas()); s.add("cv_mse", cv.cv_mse()); s.add("cv_sem", cv.cv_sem());
    s.add("cv_min", cv.cv_min()); s.add("cv_1se", cv.cv_1se());
    s.add("index_min", static_cast<std::int64_t>(cv.index_min()));
    s.add("index_1se", static_cast<std::int64_t>(cv.index_1se()));
    s.add("interpolates", cv.interpolates());
    s.add("at_edge", cv.selection_at_grid_edge());
    compare_or_record(std::string("ridge_cv_") + GetParam().name, s);
}

TEST_P(MLReference, ElasticNetPathAndGrid) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    enet_gaussian en;
    en.fit(X, y, /*alpha=*/0.7, /*n_lambda=*/40, /*lambda_min_ratio=*/-1.0,
           /*standardize=*/true, /*intercept=*/true, /*use_strong_rule=*/false,
           /*max_iter=*/100000, /*tol=*/1e-7);
    Stream s;
    s.add("lambdas", en.lambdas()); s.add("coef", en.coef()); s.add("intercepts", en.intercepts());
    s.add("dev_ratio", en.dev_ratio()); s.add("converged", en.converged());
    s.add("n_nonconverged", static_cast<std::int64_t>(en.n_nonconverged()));
    const Eigen::MatrixXd X_new = X.topRows(std::min<Eigen::Index>(7, X.rows()));
    s.add("predict", en.predict(X_new));

    // Explicit grid at a subset of the auto grid, no standardization, strong rule
    const Eigen::VectorXd grid = en.lambdas().segment(5, 10);
    enet_gaussian en2;
    en2.fit(X, y, grid, /*alpha=*/0.7, /*standardize=*/false, /*intercept=*/true,
            /*use_strong_rule=*/true, /*max_iter=*/100000, /*tol=*/1e-7);
    s.add("grid_lambdas", en2.lambdas()); s.add("grid_coef", en2.coef());
    s.add("grid_intercepts", en2.intercepts());

    // No intercept
    enet_gaussian en3;
    en3.fit(X, y, /*alpha=*/1.0, /*n_lambda=*/20, -1.0, true, /*intercept=*/false, false, 100000, 1e-7);
    s.add("noint_coef", en3.coef()); s.add("noint_intercepts", en3.intercepts());
    compare_or_record(std::string("enet_") + GetParam().name, s);
}

TEST_P(MLReference, ElasticNetCV) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    enet_cv_ccd cv;
    cv.fit(X, y, /*alpha=*/0.5, /*n_folds=*/5, /*n_lambda=*/40, -1.0, /*seed=*/3u,
           true, true, 100000, 1e-7);
    Stream s;
    s.add("lambdas", cv.lambdas()); s.add("cv_mse", cv.cv_mse()); s.add("cv_sem", cv.cv_sem());
    s.add("cv_min", cv.cv_min()); s.add("cv_1se", cv.cv_1se());
    s.add("index_min", static_cast<std::int64_t>(cv.index_min()));
    s.add("index_1se", static_cast<std::int64_t>(cv.index_1se()));
    s.add("at_edge", cv.selection_at_grid_edge()); s.add("truncated", cv.grid_truncated());
    compare_or_record(std::string("enet_cv_") + GetParam().name, s);
}

TEST_P(MLReference, SVD) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    const Eigen::Index M = 5;
    Stream s;
    const SVDResult r = SVDSolver{}.compute(X, M);          // default dispatch
    s.add("U", r.U); s.add("S", r.S); s.add("V", r.V);
    // Force the randomized path (small threshold) on the same data
    const SVDResult rr = SVDSolver{/*min_dim_thresh=*/16, /*block_size=*/64}.compute(X, M);
    s.add("rand_U", rr.U); s.add("rand_S", rr.S); s.add("rand_V", rr.V);
    compare_or_record(std::string("svd_") + GetParam().name, s);
}

TEST_P(MLReference, PCA) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    const Eigen::MatrixXd X0 = X;
    const Eigen::Index M = 4;
    Stream s;
    {
        PCA pca(X, M, /*center=*/true, /*normalize=*/true);
        s.add("pre_X", X);                                     // the in-place preprocessing
        s.add("means", pca.getMeans().transpose()); s.add("norms", pca.getNorms().transpose());
        const PCAResult r = pca.fit();
        s.add("Z", r.Z); s.add("V", r.V); s.add("ev", r.explained_variance);
        s.add("transform", pca.transform(X0.topRows(6)));
        pca.restore(X);
        s.add("restored_X", X);
    }
    {
        PCA pca(X, M, /*center=*/true, /*normalize=*/false);
        const PCAResult r = pca.fit();
        s.add("c_Z", r.Z); s.add("c_V", r.V); s.add("c_ev", r.explained_variance);
        pca.restore(X);
    }
    compare_or_record(std::string("pca_") + GetParam().name, s);
}

TEST_P(MLReference, Ridge) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    Stream s;
    s.add("beta_1e-2", RidgeSolver::solve(X, y, 1e-2));
    s.add("beta_1", RidgeSolver::solve(X, y, 1.0));
    s.add("beta_1e2", RidgeSolver::solve(X, y, 1e2));
    compare_or_record(std::string("ridge_") + GetParam().name, s);
}

TEST_P(MLReference, Scalers) {
    Eigen::MatrixXd X; Eigen::VectorXd y; make_data(GetParam(), X, y);
    X.col(1).setConstant(3.0);                                 // a constant column (dropped)
    Stream s;
    {
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        ZScoreScaler z(true, true);
        z.fit_transform_inplace(Xm);
        s.add("z_X", X); s.add("z_centers", z.get_centers()); s.add("z_scales", z.get_scales());
        s.add("z_dropped", z.get_dropped_indices());
        z.inverse_transform_inplace(Xm);
        s.add("z_back", X);
    }
    {
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        LpNormScaler l(LpNormScaler::NormType::L1, false, true);
        l.fit_transform_inplace(Xm);
        s.add("l1_X", X); s.add("l1_scales", l.get_scales());
        l.inverse_transform_inplace(Xm);
        s.add("l1_back", X);
    }
    compare_or_record(std::string("scalers_") + GetParam().name, s);
}

INSTANTIATE_TEST_SUITE_P(Shapes, MLReference, ::testing::ValuesIn(kShapes),
                         [](const ::testing::TestParamInfo<Shape>& info) {
                             return std::string(info.param.name);
                         });

} // namespace trex::test::ml_methods::reference
