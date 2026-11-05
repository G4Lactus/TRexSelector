// ========================================================================================
// test_tsolvers_cd.cpp
// ========================================================================================
/**
 * @file test_tsolvers_cd.cpp
 *
 * @brief Unit tests for the CCD-based terminating solvers (TCDLASSO, TCDENET,
 * TCDIENET, TCDTIKNET): KKT certification, crossing-support equivalence
 * against the LARS-based family, degenerate collapses, warm starts, and
 * full-path mode.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <algorithm>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdlasso_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdenet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdienet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdtiknet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::ad::linear_model {

using namespace trex::tsolvers::ad::linear_model::cd_based;
using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;
using trex::tsolvers::ad::SparseBetaPath;

// ========================================================================================

namespace {

/** @brief Real-variable support at the FIRST recorded step with at least t
 *  active dummies (first-reach, the T-Rex phi extraction semantics), for
 *  t = 1..T. result[t-1] is empty when no recorded step reaches count t. */
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
        for (std::size_t t = 1; t <= std::min(dummies, T); ++t) {
            if (found[t - 1]) { continue; }
            found[t - 1] = true;
            for (std::size_t e = 0; e < st.idx.size(); ++e) {
                if (st.idx[e] < p && std::abs(st.val[e]) > eps) {
                    result[t - 1].insert(st.idx[e]);
                }
            }
        }
    }
    return result;
}

/** @brief Index of the FIRST recorded step with at least t active dummies
 *  (path.steps indexing; 0 = initial state), or steps.size() if none. */
std::size_t firstStepReaching(const SparseBetaPath& path, std::size_t p,
                              std::size_t t, double eps = 1e-12) {
    for (std::size_t s = 0; s < path.steps.size(); ++s) {
        std::size_t dummies = 0;
        for (std::size_t e = 0; e < path.steps[s].idx.size(); ++e) {
            if (path.steps[s].idx[e] >= p &&
                std::abs(path.steps[s].val[e]) > eps) { ++dummies; }
        }
        if (dummies >= t) { return s; }
    }
    return path.steps.size();
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
        // The writer `part` stays alive across the resumed run: the resumed
        // reader expects the buffers in the saved (preprocessed) state, and
        // a writer writes X and D back when it is destroyed.
        SolverT part = make(mp);
        part.executeStep(T_partial, true);
        part.save(file);
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
        TCDLASSO_Solver s(m.Xm, m.Dm, m.ym);
        s.executeStep(5, true);
        EXPECT_GT(s.getNumSteps(), 0u);
        EXPECT_LT(s.getKktViolation(), 1e-6);
    }
    {
        Mapped m(X, D, y);
        TCDENET_Solver s(m.Xm, m.Dm, m.ym, trex::tsolvers::ad::SolverOptions{.lambda2 = 0.5});
        s.executeStep(5, true);
        EXPECT_LT(s.getKktViolation(), 1e-6);
    }
    {
        Mapped m(X, D, y);
        TCDIENET_Solver s(m.Xm, m.Dm, m.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0});
        s.executeStep(5, true);
        EXPECT_LT(s.getKktViolation(), 1e-6);
    }
}

// ========================================================================================
// Crossing-support equivalence against the LARS-based family
// ========================================================================================

/** @brief T-CDLASSO and T-LASSO solve the same lasso path; the real supports at
 *  each dummy-count crossing (what T-Rex consumes) must coincide. The CD
 *  bisection runs at a tight lambda tolerance to exclude window artifacts. */
TEST_F(TSolverCdTest, CrossingSupportsMatchTLASSO) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());

    Mapped mc(X, D, y);
    TCDLASSO_Solver ccd(mc.Xm, mc.Dm, mc.ym);
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);

    Mapped ml(X, D, y);
    TLASSO_Solver lasso(ml.Xm, ml.Dm, ml.ym);
    lasso.executeStep(T, true);

    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);
    const auto sup_lasso = supportsAtDummyCounts(lasso.getBetaPathSparse(), p, T);

    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_lasso[t - 1])
            << "TCDLASSO vs TLASSO real support diverged at dummy count t = " << t;
    }
}


/** @brief T-CDENET and T-ENET solve the same elastic-net path family (the
 *  Zou-Hastie rescaling changes values, never supports). */
TEST_F(TSolverCdTest, CrossingSupportsMatchTENET) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());
    const double lambda2 = 0.5;

    Mapped mc(X, D, y);
    TCDENET_Solver ccd(mc.Xm, mc.Dm, mc.ym, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);

    Mapped ml(X, D, y);
    TENET_Solver enet(ml.Xm, ml.Dm, ml.ym, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    enet.executeStep(T, true);

    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);
    const auto sup_en = supportsAtDummyCounts(enet.getBetaPathSparse(), p, T);

    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_en[t - 1])
            << "TCDENET vs TENET real support diverged at dummy count t = " << t;
    }
}


/** @brief T-CDIENET (group mode) and the native pathwise T-IENET solve the
 *  identical IEN problem (same modified correlations, same group sums). */
TEST_F(TSolverCdTest, CrossingSupportsMatchTIENET) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());
    const double lambda2 = 1.0;

    Mapped mc(X, D, y);
    TCDIENET_Solver ccd(mc.Xm, mc.Dm, mc.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);

    Mapped ml(X, D, y);
    TIENET_Solver ien(ml.Xm, ml.Dm, ml.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    ien.executeStep(T, true);

    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);
    const auto sup_ien = supportsAtDummyCounts(ien.getBetaPathSparse(), p, T);

    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_ien[t - 1])
            << "TCDIENET vs TIENET real support diverged at dummy count t = " << t;
    }
}

// ========================================================================================
// Degenerate collapses
// ========================================================================================

/** @brief lambda2 == 0 collapses T-CDENET exactly onto plain T-CDLASSO. */
TEST_F(TSolverCdTest, TcdenetLambda2ZeroReducesToTcdlasso) {
    Mapped me(X, D, y);
    TCDENET_Solver enet(me.Xm, me.Dm, me.ym);
    enet.executeStep(5, true);

    Mapped mc(X, D, y);
    TCDLASSO_Solver ccd(mc.Xm, mc.Dm, mc.ym);
    ccd.executeStep(5, true);

    EXPECT_EQ(enet.getActions(), ccd.getActions());
    EXPECT_TRUE(enet.getBeta(-1).isApprox(ccd.getBeta(-1), 1e-9));
}


/** @brief The generalized Tikhonov solver with K = I must reproduce the
 *  diagonal elastic net exactly (exercises the SPARSE matrix-passing hooks
 *  against the DIAGONAL specialization). */
TEST_F(TSolverCdTest, TikhonovIdentityMatchesTcdenet) {
    const double lambda2 = 0.5;
    const Eigen::Index p = X.cols();

    Eigen::SparseMatrix<double> K(p, p);
    K.setIdentity();

    Mapped mt(X, D, y);
    TCDTIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    tik.executeStep(5, true);

    Mapped me(X, D, y);
    TCDENET_Solver enet(me.Xm, me.Dm, me.ym, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    enet.executeStep(5, true);

    EXPECT_EQ(tik.getActions(), enet.getActions());
    EXPECT_TRUE(tik.getBeta(-1).isApprox(enet.getBeta(-1), 1e-9));
}


/** @brief WITHOUT dummies, all-singleton groups reduce the group-mean IEN
 *  penalty to the isotropic elastic net (Corollary 1): T-CDIENET == T-CDENET.
 *  (With dummies the corollary does NOT hold: dummy copies join their
 *  originating variable's group and couple with it through the group sum.)
 *  Runs the full-path grid since termination needs no dummies here. */
TEST_F(TSolverCdTest, SingletonGroupsNoDummiesMatchTcdenet) {
    const double lambda2 = 0.5;
    Eigen::VectorXi singleton = Eigen::VectorXi::LinSpaced(
        X.cols(), 0, static_cast<int>(X.cols()) - 1);
    Eigen::MatrixXd D_empty(X.rows(), 0);

    Eigen::MatrixXd Xi = X;
    Eigen::VectorXd yi = y;
    Eigen::Map<Eigen::MatrixXd> Xi_m(Xi.data(), Xi.rows(), Xi.cols());
    Eigen::Map<Eigen::MatrixXd> Di_m(D_empty.data(), D_empty.rows(), 0);
    Eigen::Map<Eigen::VectorXd> yi_m(yi.data(), yi.size());
    TCDIENET_Solver ien(Xi_m, Di_m, yi_m, singleton, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    ien.setFullPathGrid(20, 1e-2);
    ien.executeStep(0, true);

    Eigen::MatrixXd Xe = X;
    Eigen::VectorXd ye = y;
    Eigen::Map<Eigen::MatrixXd> Xe_m(Xe.data(), Xe.rows(), Xe.cols());
    Eigen::Map<Eigen::MatrixXd> De_m(D_empty.data(), D_empty.rows(), 0);
    Eigen::Map<Eigen::VectorXd> ye_m(ye.data(), ye.size());
    TCDENET_Solver enet(Xe_m, De_m, ye_m, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    enet.setFullPathGrid(20, 1e-2);
    enet.executeStep(0, true);

    EXPECT_EQ(ien.getActions(), enet.getActions());
    EXPECT_TRUE(ien.getBeta(-1).isApprox(enet.getBeta(-1), 1e-9));
}


/** @brief The O(1) GROUP_CENTERING hooks must reproduce the matrix-passing
 *  TCDTIKNET FOLDED path with K_GC = blockdiag(I - J/p_m) (mathematically
 *  identical penalties; FP accumulation orders differ, so beta is compared
 *  with a tolerance while the action sequence is pinned exactly on this
 *  wide-margin design — same convention as the GROUP vs FOLDED anchor). */
TEST_F(TSolverCdTest, GroupCenteringMatchesSparseFoldedKgc) {
    const double lambda2 = 2.0;
    const Eigen::Index p = X.cols();

    // K_GC from the fixture's 10 contiguous groups of 5.
    std::vector<Eigen::Triplet<double>> trip;
    for (int g = 0; g < 10; ++g) {
        for (int a = 0; a < 5; ++a) {
            for (int b = 0; b < 5; ++b) {
                const double delta = (a == b) ? 1.0 : 0.0;
                trip.emplace_back(g * 5 + a, g * 5 + b, delta - 0.2);
            }
        }
    }
    Eigen::SparseMatrix<double> K_gc(p, p);
    K_gc.setFromTriplets(trip.begin(), trip.end());

    Mapped ms(X, D, y);
    TCDTIKNET_Solver sparse_path(ms.Xm, ms.Dm, ms.ym, K_gc, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2,
        .fold_dummies = true});
    sparse_path.setLambdaRelTol(1e-6);
    sparse_path.executeStep(5, true);

    Mapped mg(X, D, y);
    TCDIENET_Solver fast_path(mg.Xm, mg.Dm, mg.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2,
        .group_penalty = GroupPenaltyVariant::CENTERING});
    fast_path.setLambdaRelTol(1e-6);
    fast_path.executeStep(5, true);

    EXPECT_EQ(fast_path.getActions(), sparse_path.getActions());
    EXPECT_TRUE(fast_path.getBeta(-1).isApprox(sparse_path.getBeta(-1), 1e-8));
}


/** @brief All-singleton groups make the centering penalty vanish
 *  (K_GC = 0): GROUP_CENTERING must reduce to plain T-CDLASSO exactly, dummies
 *  included (nothing couples through a singleton). */
TEST_F(TSolverCdTest, GroupCenteringSingletonGroupsMatchTcdlasso) {
    Eigen::VectorXi singleton = Eigen::VectorXi::LinSpaced(
        X.cols(), 0, static_cast<int>(X.cols()) - 1);

    Mapped mg(X, D, y);
    TCDIENET_Solver cen(mg.Xm, mg.Dm, mg.ym, singleton, trex::tsolvers::ad::SolverOptions{.lambda2 = 3.0,
        .group_penalty = GroupPenaltyVariant::CENTERING});
    cen.executeStep(5, true);

    Mapped mc(X, D, y);
    TCDLASSO_Solver ccd(mc.Xm, mc.Dm, mc.ym);
    ccd.executeStep(5, true);

    EXPECT_EQ(cen.getActions(), ccd.getActions());
    EXPECT_TRUE(cen.getBeta(-1).isApprox(ccd.getBeta(-1), 1e-9));
}

// ========================================================================================
// Warm start and full-path mode
// ========================================================================================

/** @brief In-memory warm start: run to T = 3, continue to T = 6; the recorded
 *  path must equal the single-shot T = 6 run. */
TEST_F(TSolverCdTest, WarmStartMatchesContinuous) {
    Mapped mw(X, D, y);
    TCDLASSO_Solver warm(mw.Xm, mw.Dm, mw.ym);
    warm.executeStep(3, true);
    warm.executeStep(6, true);

    Mapped mc(X, D, y);
    TCDLASSO_Solver cont(mc.Xm, mc.Dm, mc.ym);
    cont.executeStep(6, true);

    EXPECT_EQ(warm.getActions(), cont.getActions());
    EXPECT_TRUE(warm.getBeta(-1).isApprox(cont.getBeta(-1), 1e-9));
}


/** @brief Full-path mode (T_stop == 0): records the geometric lambda1 grid
 *  with strictly decreasing lambda and a growing support. */
TEST_F(TSolverCdTest, FullPathModeRecordsGrid) {
    Mapped m(X, D, y);
    TCDLASSO_Solver s(m.Xm, m.Dm, m.ym);
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

/** @brief T-CDLASSO round trip (lasso, no penalty config). */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCDLASSO) {
    expectRoundTripMatchesContinuous<TCDLASSO_Solver>(
        [](Mapped& m) {
            return TCDLASSO_Solver(m.Xm, m.Dm, m.ym);
        },
        "tcdlasso");
}


/** @brief T-CDENET round trip (diagonal ridge travels in the base state). */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCDENET) {
    expectRoundTripMatchesContinuous<TCDENET_Solver>(
        [](Mapped& m) {
            return TCDENET_Solver(m.Xm, m.Dm, m.ym, trex::tsolvers::ad::SolverOptions{.lambda2 = 0.5});
        },
        "tcdenet");
}


/** @brief T-CDIENET group-mode round trip: group sums and lambda2/p_m are
 *  transient and must be rebuilt by ensureCdState() after load. */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCDIENET) {
    expectRoundTripMatchesContinuous<TCDIENET_Solver>(
        [this](Mapped& m) {
            return TCDIENET_Solver(m.Xm, m.Dm, m.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0});
        },
        "tcdienet");
}


/** @brief T-CDTIKNET sparse-K round trip: the Tikhonov matrix travels as
 *  triplets and must be rebuilt (compressed) on deserialization. Uses a
 *  discrete-Laplacian K so the off-diagonal structure is actually exercised. */
TEST_F(TSolverCdTest, SerializeLoadContinueMatchesContinuous_TCDTIKNET) {
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

    expectRoundTripMatchesContinuous<TCDTIKNET_Solver>(
        [&K](Mapped& m) {
            return TCDTIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 0.5});
        },
        "tcdtiknet_sparse_k");
}


/** @brief The gram-cache fallback (ever-active set beyond gram_cap_ ->
 *  naive residual sweeps) must survive a round trip: a resumed solver
 *  forced onto the naive path must still replicate the default-cap
 *  continuous run (gram caching is an implementation detail, never a
 *  result-changing one). */
TEST_F(TSolverCdTest, SerializeLoadGramCapFallbackMatchesDefault) {
    Mapped mr(X, D, y);
    TCDLASSO_Solver ref(mr.Xm, mr.Dm, mr.ym);
    ref.executeStep(6, true);

    Mapped mp(X, D, y);
    const std::string file =
        ::testing::TempDir() + "cd_ckpt_gramcap.bin";
    // `part` stays alive: the resumed reader expects the saved buffer state
    TCDLASSO_Solver part(mp.Xm, mp.Dm, mp.ym);
    part.setGramCap(2);   // forces naive sweeps almost immediately
    part.executeStep(3, true);
    part.save(file);

    TCDLASSO_Solver res = TCDLASSO_Solver::load(file, mp.Xm, mp.Dm);
    res.setGramCap(2);        // runtime knob: re-applied after load
    res.executeStep(6, true);

    EXPECT_EQ(ref.getActions(), res.getActions());
    EXPECT_TRUE(ref.getBeta(-1).isApprox(res.getBeta(-1), 1e-9));
    EXPECT_LT(res.getKktViolation(), 1e-6);

    std::filesystem::remove(file);
}

// ========================================================================================
// Certified localization (T-CD theory note, Part II)
// ========================================================================================

/** @brief Every record of a certified run carries a duality-gap and a KKT
 *  certificate at the numerical floor, for all four variants; the ledger
 *  top's certified signed support equals the last record's snapshot. */
TEST_F(TSolverCdTest, CertifiedRecordsCarryCertificates) {
    auto check = [&](TCDLASSO_Solver& s, const std::string& tag) {
        ASSERT_TRUE(s.isCertifiedLocalization()) << tag;
        s.executeStep(5, true);
        const auto& gaps = s.getRecordGaps();
        const auto& kkts = s.getRecordKktViolations();
        ASSERT_GT(s.getNumSteps(), 0u) << tag;
        ASSERT_EQ(gaps.size(), s.getNumSteps()) << tag;
        ASSERT_EQ(kkts.size(), s.getNumSteps()) << tag;
        for (std::size_t k = 0; k < gaps.size(); ++k) {
            EXPECT_LT(gaps[k], 1e-6) << tag << ": gap at record " << k;
            EXPECT_LT(kkts[k], 1e-7) << tag << ": KKT at record " << k;
        }
        EXPECT_LT(s.getKktViolation(), 1e-7) << tag;
        const auto diag = s.getCertifiedDiagnostics();
        EXPECT_GT(diag.certified_solves, 0) << tag;
        EXPECT_GT(diag.brackets_sk, 0) << tag;

        // Ledger top == last record (0b: the snapshot IS the certified support).
        const auto [sup, sgn] = s.getCertifiedSupport();
        const Eigen::VectorXd b = s.getBeta(-1);
        std::set<std::size_t> from_beta;
        for (Eigen::Index j = 0; j < b.size(); ++j) {
            if (b(j) != 0.0) { from_beta.insert(static_cast<std::size_t>(j)); }
        }
        EXPECT_EQ(std::set<std::size_t>(sup.begin(), sup.end()), from_beta) << tag;
        for (std::size_t k = 0; k < sup.size(); ++k) {
            EXPECT_EQ(sgn[k], (b(static_cast<Eigen::Index>(sup[k])) > 0.0) ? 1 : -1) << tag;
        }
    };
    {
        Mapped m(X, D, y);
        TCDLASSO_Solver s(m.Xm, m.Dm, m.ym);
        check(s, "TCDLASSO");
    }
    {
        Mapped m(X, D, y);
        TCDENET_Solver s(m.Xm, m.Dm, m.ym, trex::tsolvers::ad::SolverOptions{.lambda2 = 0.5});
        check(s, "TCDENET");
    }
    {
        Mapped m(X, D, y);
        TCDIENET_Solver s(m.Xm, m.Dm, m.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0});
        check(s, "TCDIENET");
    }
    {
        const Eigen::Index p = X.cols();
        std::vector<Eigen::Triplet<double>> trip;
        for (int g = 0; g < 10; ++g) {
            for (int a = 0; a < 5; ++a) {
                for (int b = 0; b < 5; ++b) {
                    trip.emplace_back(g * 5 + a, g * 5 + b, ((a == b) ? 1.0 : 0.0) - 0.2);
                }
            }
        }
        Eigen::SparseMatrix<double> K_gc(p, p);
        K_gc.setFromTriplets(trip.begin(), trip.end());
        Mapped m(X, D, y);
        TCDTIKNET_Solver s(m.Xm, m.Dm, m.ym, K_gc, trex::tsolvers::ad::SolverOptions{.lambda2 = 2.0,
            .fold_dummies = true});
        check(s, "TCDTIKNET(folded)");
    }
}


/** @brief The certified solver records each crossing AT the knot (Def. 13
 *  item 4): its lambda must coincide with the T-LASSO breakpoint at which the
 *  same dummy count is first reached, to the accuracy of two independent
 *  computations of the same exact path (Cholesky continuation vs LARS Cmax). */
TEST_F(TSolverCdTest, CertifiedKnotsMatchTLASSOBreakpoints) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());

    Mapped mc(X, D, y);
    TCDLASSO_Solver ccd(mc.Xm, mc.Dm, mc.ym);
    ccd.executeStep(T, true);

    Mapped ml(X, D, y);
    TLASSO_Solver lasso(ml.Xm, ml.Dm, ml.ym);
    lasso.executeStep(T, true);

    const auto path_cd = ccd.getBetaPathSparse();
    const auto path_la = lasso.getBetaPathSparse();
    const auto& lam_cd = ccd.getLambda();      // lam_cd[s-1] = lambda of recorded step s
    const auto& lam_la = lasso.getLambda();    // lam_la[s-1] = Cmax at LARS step s

    for (std::size_t t = 1; t <= T; ++t) {
        const std::size_t s_cd = firstStepReaching(path_cd, p, t);
        const std::size_t s_la = firstStepReaching(path_la, p, t);
        ASSERT_LT(s_cd, path_cd.steps.size()) << "CD never reached t = " << t;
        ASSERT_LT(s_la, path_la.steps.size()) << "TLASSO never reached t = " << t;
        ASSERT_GE(s_cd, 1u);
        ASSERT_GE(s_la, 1u);
        const double knot_cd = lam_cd[s_cd - 1];
        const double knot_la = lam_la[s_la - 1];
        EXPECT_NEAR(knot_cd, knot_la, 1e-6 * knot_la)
            << "knot of crossing t = " << t << " differs from the T-LASSO breakpoint";
    }

    // The supports at the crossings coincide without any lambda tolerance.
    const auto sup_ccd = supportsAtDummyCounts(path_cd, p, T);
    const auto sup_lasso = supportsAtDummyCounts(path_la, p, T);
    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_lasso[t - 1])
            << "certified TCDLASSO vs TLASSO real support diverged at t = " << t;
    }
}


/** @brief LEGACY localization (probe bisection, support-nesting stop) stays
 *  available behind the flag and still reproduces the T-LASSO supports at a
 *  tight lambda tolerance (the A/B baseline of the re-validation protocol). */
TEST_F(TSolverCdTest, LegacyLocalizationMatchesTLASSO) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());

    Mapped mc(X, D, y);
    TCDLASSO_Solver ccd(mc.Xm, mc.Dm, mc.ym);
    ccd.setCertifiedLocalization(false);
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);
    EXPECT_FALSE(ccd.isCertifiedLocalization());
    EXPECT_EQ(ccd.getCertifiedDiagnostics().certified_solves, 0);

    Mapped ml(X, D, y);
    TLASSO_Solver lasso(ml.Xm, ml.Dm, ml.ym);
    lasso.executeStep(T, true);

    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);
    const auto sup_lasso = supportsAtDummyCounts(lasso.getBetaPathSparse(), p, T);
    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_ccd[t - 1], sup_lasso[t - 1])
            << "legacy TCDLASSO vs TLASSO real support diverged at t = " << t;
    }
}


/** @brief Certified resumption: pausing after T = 3 and continuing to T = 6
 *  reproduces the uninterrupted records including the knot locations
 *  (Prop. 4), through the serialized frontier. */
TEST_F(TSolverCdTest, CertifiedResumeReproducesKnots) {
    Mapped mr(X, D, y);
    TCDLASSO_Solver ref(mr.Xm, mr.Dm, mr.ym);
    ref.executeStep(6, true);

    Mapped mp(X, D, y);
    const std::string file = ::testing::TempDir() + "cd_ckpt_certified.bin";
    // `part` stays alive: the resumed reader expects the saved buffer state
    TCDLASSO_Solver part(mp.Xm, mp.Dm, mp.ym);
    part.executeStep(3, true);
    part.save(file);
    TCDLASSO_Solver res = TCDLASSO_Solver::load(file, mp.Xm, mp.Dm);
    res.executeStep(6, true);

    ASSERT_EQ(ref.getLambda().size(), res.getLambda().size());
    for (std::size_t k = 0; k < ref.getLambda().size(); ++k) {
        EXPECT_NEAR(ref.getLambda()[k], res.getLambda()[k], 1e-9 * ref.getLambda()[k]);
    }
    EXPECT_EQ(ref.getActions(), res.getActions());
    std::filesystem::remove(file);
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::ad::linear_model */

