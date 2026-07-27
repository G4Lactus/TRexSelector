// ========================================================================================
// test_trex_tikhonov.cpp
// ========================================================================================
/**
 * @file test_trex_tikhonov.cpp
 *
 * @brief Unit tests for TRexTikhonovSelector (general-Tikhonov T-Rex):
 * construction validation, the K = I collapse onto the TCENET base run,
 * planted-group recovery with an informed K, and the CV lambda_2 path.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// std includes
#include <random>
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_tikhonov/trex_tikhonov.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_tikhonov {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_tikhonov;
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::utils::datageneration::datagen;

// ========================================================================================

namespace {

/** @brief Identity K (p x p). */
Eigen::SparseMatrix<double> identityK(Eigen::Index p) {
    Eigen::SparseMatrix<double> K(p, p);
    K.setIdentity();
    return K;
}

/** @brief Group-mean IEN matrix W = sum_m 1_m 1_m^T / p_m for contiguous
 *  groups of size `gsize`. */
Eigen::SparseMatrix<double> groupMeanK(Eigen::Index p, Eigen::Index gsize) {
    Eigen::SparseMatrix<double> W(p, p);
    std::vector<Eigen::Triplet<double>> trip;
    for (Eigen::Index i = 0; i < p; ++i)
        for (Eigen::Index j = 0; j < p; ++j)
            if (i / gsize == j / gsize)
                trip.emplace_back(i, j, 1.0 / static_cast<double>(gsize));
    W.setFromTriplets(trip.begin(), trip.end());
    return W;
}

/** @brief Correlated grouped design with the planted actives in the first
 *  two groups (mirrors the GVS test generator: within-group AR structure,
 *  additive signal on cols 0-5). */
struct GroupedData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    GroupedData(Eigen::Index n, Eigen::Index p, double rho, double sigma,
                unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> N01(0.0, 1.0);

        X.resize(n, p);
        for (Eigen::Index g = 0; g < p; g += 5) {
            Eigen::VectorXd z(n);
            for (Eigen::Index i = 0; i < n; ++i) z(i) = N01(rng);
            for (Eigen::Index j = g; j < std::min(g + 5, p); ++j) {
                for (Eigen::Index i = 0; i < n; ++i) {
                    X(i, j) = rho * z(i) + std::sqrt(1.0 - rho * rho) * N01(rng);
                }
            }
        }

        y.resize(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            double signal = 0.0;
            for (Eigen::Index j = 0; j < 3; ++j) { signal += X(i, j); }
            for (Eigen::Index j = 3; j < 6; ++j) { signal -= X(i, j); }
            y(i) = signal + sigma * N01(rng);
        }
    }
};

} // namespace

// ========================================================================================

class TRexTikhonovTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::Map<Eigen::MatrixXd> X_map;
    Eigen::Map<Eigen::VectorXd> y_map;

    TRexTikhonovTest()
        : X(Eigen::MatrixXd::Random(30, 10)),
          y(Eigen::VectorXd::Random(30)),
          X_map(X.data(), X.rows(), X.cols()),
          y_map(y.data(), y.size()) {}
};


// ========================================================================================
// Construction validation
// ========================================================================================

/** @brief An empty or mis-dimensioned K, or a non-TCIENET solver_type,
 *  must throw at construction. */
TEST_F(TRexTikhonovTest, Validation_ThrowsOnBadKOrSolver) {

    // Default control: K is empty.
    TRexTikhonovControlParameter ctrl;
    EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                 std::invalid_argument);

    // Wrong dimensions (p = 10 here).
    ctrl.tikhonov_K = identityK(11);
    EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                 std::invalid_argument);

    // Correct K, wrong solver.
    ctrl.tikhonov_K = identityK(10);
    ctrl.trex_ctrl.solver_type = sd::SolverTypeForTRex::TLARS;
    EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                 std::invalid_argument);

    // Correct K, default solver (TCIENET) constructs.
    ctrl.trex_ctrl.solver_type = sd::SolverTypeForTRex::TCIENET;
    EXPECT_NO_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl));
}


/** @brief gammaToK builds K = Gamma^T Gamma (first-difference operator ->
 *  tridiagonal graph Laplacian). */
TEST_F(TRexTikhonovTest, GammaToK_FirstDifferenceGivesLaplacian) {
    const Eigen::Index p = 6;
    Eigen::SparseMatrix<double> gamma(p - 1, p);
    std::vector<Eigen::Triplet<double>> trip;
    for (Eigen::Index i = 0; i < p - 1; ++i) {
        trip.emplace_back(i, i, -1.0);
        trip.emplace_back(i, i + 1, 1.0);
    }
    gamma.setFromTriplets(trip.begin(), trip.end());

    const Eigen::MatrixXd K =
        Eigen::MatrixXd(TRexTikhonovSelector::gammaToK(gamma));
    const Eigen::MatrixXd K_ref =
        Eigen::MatrixXd(gamma.transpose()) * Eigen::MatrixXd(gamma);
    EXPECT_TRUE(K.isApprox(K_ref, 1e-14));
    EXPECT_DOUBLE_EQ(K(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(K(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(K(0, 1), -1.0);
}


// ========================================================================================
// K = I collapse: must reproduce the base TCENET run exactly
// ========================================================================================

/** @brief With K = I the TCIENET sparse-K penalty equals the diagonal
 *  elastic net, and TRexTikhonovSelector reuses the base dummy machinery —
 *  so a run with identical seed and lambda_2 must reproduce the plain
 *  TRexSelector + TCENET selection exactly. */
TEST_F(TRexTikhonovTest, Equivalence_IdentityKMatchesTcenetBaseRun) {
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    const double lambda2 = 0.5;

    auto run_tik = [&]() {
        SyntheticData data(150, 40, support, coefs, 1.0, 42);
        auto Xd = data.getX();
        auto yd = data.getY();
        Eigen::Map<Eigen::MatrixXd> Xm(Xd.data(), Xd.rows(), Xd.cols());
        Eigen::Map<Eigen::VectorXd> ym(yd.data(), yd.size());

        TRexTikhonovControlParameter ctrl;
        ctrl.tikhonov_K = identityK(40);
        ctrl.lambda_2   = lambda2;
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    auto run_cenet = [&]() {
        SyntheticData data(150, 40, support, coefs, 1.0, 42);
        auto Xd = data.getX();
        auto yd = data.getY();
        Eigen::Map<Eigen::MatrixXd> Xm(Xd.data(), Xd.rows(), Xd.cols());
        Eigen::Map<Eigen::VectorXd> ym(yd.data(), yd.size());

        TRexControlParameter ctrl;
        ctrl.solver_type = sd::SolverTypeForTRex::TCENET;
        ctrl.solver_params.lambda2 = lambda2;
        ctrl.K = 5;
        ctrl.max_dummy_multiplier = 2;
        TRexSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_tik   = run_tik();
    const Eigen::VectorXi sel_cenet = run_cenet();

    ASSERT_EQ(sel_tik.size(), sel_cenet.size());
    EXPECT_TRUE((sel_tik.array() == sel_cenet.array()).all())
        << "K = I Tikhonov run diverged from the TCENET base run.\n"
        << "  Tikhonov n_selected = " << sel_tik.sum() << "\n"
        << "  TCENET   n_selected = " << sel_cenet.sum();
}


// ========================================================================================
// End-to-end recovery with an informed K
// ========================================================================================

/** @brief Group-mean K on a correlated grouped design: the planted actives
 *  (cols 0-5) are recovered with few false selections, and the fixed
 *  lambda_2 is reported back verbatim. */
TEST_F(TRexTikhonovTest, EndToEnd_GroupMeanKRecoversPlanted) {
    const Eigen::Index n = 150, p = 40;
    GroupedData d(n, p, 0.75, 1.0, 7);
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    TRexTikhonovControlParameter ctrl;
    ctrl.tikhonov_K = groupMeanK(p, 5);
    ctrl.lambda_2   = 1.0;
    ctrl.trex_ctrl.K = 5;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;

    TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
    auto result = trex.select();

    ASSERT_EQ(result.selected_var.size(), p);
    EXPECT_DOUBLE_EQ(trex.getLambda2Used(), 1.0);
    for (Eigen::Index j = 0; j < result.Phi_prime.size(); ++j) {
        EXPECT_GE(result.Phi_prime(j), 0.0);
        EXPECT_LE(result.Phi_prime(j), 1.0);
    }

    int hits = 0, false_sel = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
        if (result.selected_var(j) == 1) {
            if (j < 6) { ++hits; } else { ++false_sel; }
        }
    }
    EXPECT_GE(hits, 4)
        << "Group-mean K failed to recover the planted actives.";
    EXPECT_LE(false_sel, 4)
        << "Group-mean K selected too many null variables.";
}


// ========================================================================================
// Auto lambda_2 (sparse-K CV tuners)
// ========================================================================================

/** @brief Analytic Tikhonov CV backbone on an n > p design: resolves a
 *  positive lambda_2 and completes selection. */
TEST_F(TRexTikhonovTest, AutoLambda2_TikSvdResolvesAndRuns) {
    const Eigen::Index n = 150, p = 40;
    GroupedData d(n, p, 0.75, 1.0, 7);
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    TRexTikhonovControlParameter ctrl;
    ctrl.tikhonov_K     = groupMeanK(p, 5);
    ctrl.lambda_2       = -1.0;   // sentinel -> CV
    ctrl.lambda2_method = TikLambda2Method::CV_1SE_TIK_SVD;
    ctrl.cv_n_folds     = 5;
    ctrl.cv_n_lambda    = 50;
    ctrl.trex_ctrl.K = 3;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;

    TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
    auto result = trex.select();

    ASSERT_EQ(result.selected_var.size(), p);
    EXPECT_GT(trex.getLambda2Used(), 0.0);
}


/** @brief The profiled IEN-CCD tuner (recommended default) resolves a
 *  positive lambda_2 and completes selection. */
TEST_F(TRexTikhonovTest, AutoLambda2_IenCcdResolvesAndRuns) {
    const Eigen::Index n = 150, p = 40;
    GroupedData d(n, p, 0.75, 1.0, 7);
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    TRexTikhonovControlParameter ctrl;
    ctrl.tikhonov_K     = groupMeanK(p, 5);
    ctrl.lambda_2       = -1.0;
    ctrl.lambda2_method = TikLambda2Method::CV_1SE_IEN_CCD;
    ctrl.cv_n_folds     = 5;
    ctrl.trex_ctrl.K = 3;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;

    TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
    auto result = trex.select();

    ASSERT_EQ(result.selected_var.size(), p);
    EXPECT_GT(trex.getLambda2Used(), 0.0);
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::trex_tikhonov */
// ========================================================================================
