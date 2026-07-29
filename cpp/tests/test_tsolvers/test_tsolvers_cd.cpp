// ========================================================================================
// test_tsolvers_cd.cpp
// ========================================================================================
/**
 * @file test_tsolvers_cd.cpp
 *
 * @brief Unit tests for the CCD-based terminating solvers (TCCD, TCENET,
 * TCIENET): KKT certification, crossing-support equivalence against the
 * LARS-based family, degenerate collapses, warm starts, and full-path mode.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <filesystem>
#include <set>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// tsolvers includes
#include <tsolvers/linear_model/cd_based/tccd_solver.hpp>
#include <tsolvers/linear_model/cd_based/tcenet_solver.hpp>
#include <tsolvers/linear_model/cd_based/tcienet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tienet_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::linear_model {

using namespace trex::tsolvers::linear_model::cd_based;
using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;
using trex::tsolvers::SparseBetaPath;

// ========================================================================================

namespace {

/** @brief Real-variable support at the FIRST recorded step with exactly t
 *  active dummies, for t = 1..T (the quantity the T-Rex phi calibration
 *  consumes). result[t-1] is empty when no recorded step hits count t. */
std::vector<std::set<std::size_t>> supportsAtDummyCounts(
    const SparseBetaPath& path, std::size_t p, std::size_t T,
    double eps = 1e-12) {

    std::vector<std::set<std::size_t>> result(T);
    std::vector<bool> found(T, false);

    for (const auto& st : path.steps) {
        std::size_t dummies = 0;
        for (std::size_t e = 0; e < st.idx.size(); ++e) {
            if (st.idx[e] >= p && std::abs(st.val[e]) > eps) { ++dummies; }
        }
        if (dummies >= 1 && dummies <= T && !found[dummies - 1]) {
            found[dummies - 1] = true;
            for (std::size_t e = 0; e < st.idx.size(); ++e) {
                if (st.idx[e] < p && std::abs(st.val[e]) > eps) {
                    result[dummies - 1].insert(st.idx[e]);
                }
            }
        }
    }
    return result;
}

} // namespace

// ========================================================================================

class TSolverCdTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;
    Eigen::VectorXi groups;   // 10 contiguous groups of 5 over p = 50

    void SetUp() override {
        SyntheticData data(
            200, 50, 50,
            {0, 10, 20},
            {5.0, -3.0, 2.0},
            1.0, 42, -1, -1,
            predictor_policy::Normal(),
            dummygen::Distribution::Normal(),
            noisegen::noise_policy::Normal()
        );
        X = data.getX();
        D = data.getD();
        y = data.getY();

        groups.resize(X.cols());
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            groups(j) = static_cast<int>(j / 5);
        }
    }

    // Fresh maps over private copies (the solvers normalize in place).
    struct Mapped {
        Eigen::MatrixXd Xc, Dc;
        Eigen::VectorXd yc;
        Eigen::Map<Eigen::MatrixXd> Xm, Dm;
        Eigen::Map<Eigen::VectorXd> ym;
        Mapped(const Eigen::MatrixXd& X, const Eigen::MatrixXd& D,
               const Eigen::VectorXd& y)
            : Xc(X), Dc(D), yc(y),
              Xm(Xc.data(), Xc.rows(), Xc.cols()),
              Dm(Dc.data(), Dc.rows(), Dc.cols()),
              ym(yc.data(), yc.size()) {}
    };

    /** @brief Serialize -> load -> continue must replicate the single-shot
     *  run: partial execution to T_partial, checkpoint to disk, reload
     *  (reconnecting to the in-place-normalized buffers), continue to
     *  T_final, and compare actions + final beta against an uninterrupted
     *  reference. The KKT certificate on the resumed solver exercises the
     *  gram re-entry in ensureCdState() (slots are empty after load while
     *  actives_ is not). */
    template <class SolverT, class MakeFn>
    void expectRoundTripMatchesContinuous(MakeFn make, const std::string& tag,
                                          std::size_t T_partial = 3,
                                          std::size_t T_final = 6) {
        Mapped mr(X, D, y);
        SolverT ref = make(mr);
        ref.executeStep(T_final, true);

        Mapped mp(X, D, y);
        const std::string file =
            ::testing::TempDir() + "cd_ckpt_" + tag + ".bin";
        {
            SolverT part = make(mp);
            part.executeStep(T_partial, true);
            part.save(file);
        }
        ASSERT_TRUE(std::filesystem::exists(file));

        SolverT res = SolverT::load(file, mp.Xm, mp.Dm);
        res.executeStep(T_final, true);

        EXPECT_EQ(ref.getActions(), res.getActions())
            << tag << ": resumed path diverged from the continuous run.";
        EXPECT_TRUE(ref.getBeta(-1).isApprox(res.getBeta(-1), 1e-9))
            << tag << ": resumed final beta diverged from the continuous run.";
        EXPECT_LT(res.getKktViolation(), 1e-6)
            << tag << ": resumed state is not a certified minimizer.";

        std::filesystem::remove(file);
    }
};

// ========================================================================================
// KKT certification
// ========================================================================================

/** @brief Every recorded state is a certified penalized-lasso minimizer:
 *  the KKT residual at the final lambda must vanish up to tolerance. */
TEST_F(TSolverCdTest, KktCertificateAtTermination) {
    {
        Mapped m(X, D, y);
        TCCD_Solver s(m.Xm, m.Dm, m.ym, true, true, false);
        s.executeStep(5, true);
        EXPECT_GT(s.getNumSteps(), 0u);
        EXPECT_LT(s.getKktViolation(), 1e-6);
    }
    {
        Mapped m(X, D, y);
        TCENET_Solver s(m.Xm, m.Dm, m.ym, 0.5, true, true, false);
        s.executeStep(5, true);
        EXPECT_LT(s.getKktViolation(), 1e-6);
    }
    {
        Mapped m(X, D, y);
        TCIENET_Solver s(m.Xm, m.Dm, m.ym, 1.0, groups, true, true, false);
        s.executeStep(5, true);
        EXPECT_LT(s.getKktViolation(), 1e-6);
    }
}

// ========================================================================================
// Crossing-support equivalence against the LARS-based family
// ========================================================================================

/** @brief T-CCD and T-LASSO solve the same lasso path; the real supports at
 *  each dummy-count crossing (what T-Rex consumes) must coincide. The CD
 *  bisection runs at a tight lambda tolerance to exclude window artifacts. */
TEST_F(TSolverCdTest, CrossingSupportsMatchTLASSO) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());

    Mapped mc(X, D, y);
    TCCD_Solver ccd(mc.Xm, mc.Dm, mc.ym, true, true, false);
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);

    Mapped ml(X, D, y);
    TLASSO_Solver lasso(ml.Xm, ml.Dm, ml.ym, true, true, false);
    lasso.executeStep(T, true);

    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);
    const auto sup_lasso = supportsAtDummyCounts(lasso.getBetaPathSparse(), p, T);

    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_lasso[t - 1])
            << "TCCD vs TLASSO real support diverged at dummy count t = " << t;
    }
}


/** @brief T-CENET and T-ENET solve the same elastic-net path family (the
 *  Zou-Hastie rescaling changes values, never supports). */
TEST_F(TSolverCdTest, CrossingSupportsMatchTENET) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());
    const double lambda2 = 0.5;

    Mapped mc(X, D, y);
    TCENET_Solver ccd(mc.Xm, mc.Dm, mc.ym, lambda2, true, true, false);
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);

    Mapped ml(X, D, y);
    TENET_Solver enet(ml.Xm, ml.Dm, ml.ym, lambda2, true, true, false);
    enet.executeStep(T, true);

    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);
    const auto sup_en = supportsAtDummyCounts(enet.getBetaPathSparse(), p, T);

    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_en[t - 1])
            << "TCENET vs TENET real support diverged at dummy count t = " << t;
    }
}


/** @brief T-CIENET (group mode) and the native pathwise T-IENET solve the
 *  identical IEN problem (same modified correlations, same group sums). */
TEST_F(TSolverCdTest, CrossingSupportsMatchTIENET) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());
    const double lambda2 = 1.0;

    Mapped mc(X, D, y);
    TCIENET_Solver ccd(mc.Xm, mc.Dm, mc.ym, lambda2, groups, true, true, false);
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);

    Mapped ml(X, D, y);
    TIENET_Solver ien(ml.Xm, ml.Dm, ml.ym, lambda2, groups, true, true, false);
    ien.executeStep(T, true);

    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);
    const auto sup_ien = supportsAtDummyCounts(ien.getBetaPathSparse(), p, T);

    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_ien[t - 1])
            << "TCIENET vs TIENET real support diverged at dummy count t = " << t;
    }
}

// ========================================================================================
// Degenerate collapses
// ========================================================================================

/** @brief lambda2 == 0 collapses T-CENET exactly onto plain T-CCD. */
TEST_F(TSolverCdTest, TcenetLambda2ZeroReducesToTccd) {
    Mapped me(X, D, y);
    TCENET_Solver enet(me.Xm, me.Dm, me.ym, 0.0, true, true, false);
    enet.executeStep(5, true);

    Mapped mc(X, D, y);
    TCCD_Solver ccd(mc.Xm, mc.Dm, mc.ym, true, true, false);
    ccd.executeStep(5, true);

    EXPECT_EQ(enet.getActions(), ccd.getActions());
    EXPECT_TRUE(enet.getBeta(-1).isApprox(ccd.getBeta(-1), 1e-9));
}


/** @brief The generalized Tikhonov constructor with K = I must reproduce the
 *  diagonal elastic net exactly (exercises the SPARSE matrix-passing hooks
 *  against the DIAGONAL specialization). */
TEST_F(TSolverCdTest, TikhonovIdentityMatchesTcenet) {
    const double lambda2 = 0.5;
    const Eigen::Index p = X.cols();

    Eigen::SparseMatrix<double> K(p, p);
    K.setIdentity();

    Mapped mt(X, D, y);
    TCIENET_Solver tik(mt.Xm, mt.Dm, mt.ym, lambda2, K, true, true, false);
    tik.executeStep(5, true);

    Mapped me(X, D, y);
    TCENET_Solver enet(me.Xm, me.Dm, me.ym, lambda2, true, true, false);
    enet.executeStep(5, true);

    EXPECT_EQ(tik.getActions(), enet.getActions());
    EXPECT_TRUE(tik.getBeta(-1).isApprox(enet.getBeta(-1), 1e-9));
}


/** @brief WITHOUT dummies, all-singleton groups reduce the group-mean IEN
 *  penalty to the isotropic elastic net (Corollary 1): T-CIENET == T-CENET.
 *  (With dummies the corollary does NOT hold: dummy copies join their
 *  originating variable's group and couple with it through the group sum.)
 *  Runs the full-path grid since termination needs no dummies here. */
TEST_F(TSolverCdTest, SingletonGroupsNoDummiesMatchTcenet) {
    const double lambda2 = 0.5;
    Eigen::VectorXi singleton = Eigen::VectorXi::LinSpaced(
        X.cols(), 0, static_cast<int>(X.cols()) - 1);
    Eigen::MatrixXd D_empty(X.rows(), 0);

    Eigen::MatrixXd Xi = X;
    Eigen::VectorXd yi = y;
    Eigen::Map<Eigen::MatrixXd> Xi_m(Xi.data(), Xi.rows(), Xi.cols());
    Eigen::Map<Eigen::MatrixXd> Di_m(D_empty.data(), D_empty.rows(), 0);
    Eigen::Map<Eigen::VectorXd> yi_m(yi.data(), yi.size());
    TCIENET_Solver ien(Xi_m, Di_m, yi_m, lambda2, singleton,
                       true, true, false);
    ien.setFullPathGrid(20, 1e-2);
    ien.executeStep(0, true);

    Eigen::MatrixXd Xe = X;
    Eigen::VectorXd ye = y;
    Eigen::Map<Eigen::MatrixXd> Xe_m(Xe.data(), Xe.rows(), Xe.cols());
    Eigen::Map<Eigen::MatrixXd> De_m(D_empty.data(), D_empty.rows(), 0);
    Eigen::Map<Eigen::VectorXd> ye_m(ye.data(), ye.size());
    TCENET_Solver enet(Xe_m, De_m, ye_m, lambda2, true, true, false);
    enet.setFullPathGrid(20, 1e-2);
    enet.executeStep(0, true);

    EXPECT_EQ(ien.getActions(), enet.getActions());
    EXPECT_TRUE(ien.getBeta(-1).isApprox(enet.getBeta(-1), 1e-9));
}

// ========================================================================================
// Warm start and full-path mode
// ========================================================================================

/** @brief In-memory warm start: run to T = 3, continue to T = 6; the recorded
 *  path must equal the single-shot T = 6 run. */
TEST_F(TSolverCdTest, WarmStartMatchesContinuous) {
    Mapped mw(X, D, y);
    TCCD_Solver warm(mw.Xm, mw.Dm, mw.ym, true, true, false);
    warm.executeStep(3, true);
    warm.executeStep(6, true);

    Mapped mc(X, D, y);
    TCCD_Solver cont(mc.Xm, mc.Dm, mc.ym, true, true, false);
    cont.executeStep(6, true);

    EXPECT_EQ(warm.getActions(), cont.getActions());
    EXPECT_TRUE(warm.getBeta(-1).isApprox(cont.getBeta(-1), 1e-9));
}


/** @brief Full-path mode (T_stop == 0): records the geometric lambda1 grid
 *  with strictly decreasing lambda and a growing support. */
TEST_F(TSolverCdTest, FullPathModeRecordsGrid) {
    Mapped m(X, D, y);
    TCCD_Solver s(m.Xm, m.Dm, m.ym, true, true, false);
    s.setFullPathGrid(25, 1e-2);
    s.executeStep(0, true);

    const auto& lambda = s.getLambda();
    ASSERT_GE(lambda.size(), 2u);
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LT(lambda[k], lambda[k - 1]) << "lambda grid not decreasing";
    }
    EXPECT_GT(s.getNumSteps(), 0u);
    EXPECT_GT(s.getRSS().front(), s.getRSS().back());
}

// ========================================================================================
// Serialization: save -> load -> continue
// ========================================================================================

/** @brief T-CCD round trip (lasso, no penalty config). */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCCD) {
    expectRoundTripMatchesContinuous<TCCD_Solver>(
        [](Mapped& m) {
            return TCCD_Solver(m.Xm, m.Dm, m.ym, true, true, false);
        },
        "tccd");
}


/** @brief T-CENET round trip (diagonal ridge travels in the base state). */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCENET) {
    expectRoundTripMatchesContinuous<TCENET_Solver>(
        [](Mapped& m) {
            return TCENET_Solver(m.Xm, m.Dm, m.ym, 0.5, true, true, false);
        },
        "tcenet");
}


/** @brief T-CIENET group-mode round trip: group sums and lambda2/p_m are
 *  transient and must be rebuilt by ensureCdState() after load. */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCIENET) {
    expectRoundTripMatchesContinuous<TCIENET_Solver>(
        [this](Mapped& m) {
            return TCIENET_Solver(m.Xm, m.Dm, m.ym, 1.0, groups,
                                  true, true, false);
        },
        "tcienet");
}


/** @brief T-CIENET sparse-K round trip: the Tikhonov matrix travels as
 *  triplets and must be rebuilt (compressed) on deserialization. Uses a
 *  discrete-Laplacian K so the off-diagonal structure is actually exercised. */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCIENET_SparseK) {
    const Eigen::Index p = X.cols();
    Eigen::SparseMatrix<double> K(p, p);
    std::vector<Eigen::Triplet<double>> trips;
    for (Eigen::Index j = 0; j < p; ++j) {
        trips.emplace_back(j, j, 2.0);
        if (j + 1 < p) {
            trips.emplace_back(j, j + 1, -1.0);
            trips.emplace_back(j + 1, j, -1.0);
        }
    }
    K.setFromTriplets(trips.begin(), trips.end());

    expectRoundTripMatchesContinuous<TCIENET_Solver>(
        [&K](Mapped& m) {
            return TCIENET_Solver(m.Xm, m.Dm, m.ym, 0.5, K,
                                  true, true, false);
        },
        "tcienet_sparse_k");
}


/** @brief The gram-cache fallback (ever-active set beyond gram_cap_ ->
 *  naive residual sweeps) must survive a round trip: a resumed solver
 *  forced onto the naive path must still replicate the default-cap
 *  continuous run (gram caching is an implementation detail, never a
 *  result-changing one). */
TEST_F(TSolverCdTest, SerializeLoadGramCapFallbackMatchesDefault) {
    Mapped mr(X, D, y);
    TCCD_Solver ref(mr.Xm, mr.Dm, mr.ym, true, true, false);
    ref.executeStep(6, true);

    Mapped mp(X, D, y);
    const std::string file =
        ::testing::TempDir() + "cd_ckpt_gramcap.bin";
    {
        TCCD_Solver part(mp.Xm, mp.Dm, mp.ym, true, true, false);
        part.setGramCap(2);   // forces naive sweeps almost immediately
        part.executeStep(3, true);
        part.save(file);
    }

    TCCD_Solver res = TCCD_Solver::load(file, mp.Xm, mp.Dm);
    res.setGramCap(2);        // runtime knob: re-applied after load
    res.executeStep(6, true);

    EXPECT_EQ(ref.getActions(), res.getActions());
    EXPECT_TRUE(ref.getBeta(-1).isApprox(res.getBeta(-1), 1e-9));
    EXPECT_LT(res.getKktViolation(), 1e-6);

    std::filesystem::remove(file);
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::linear_model */
