// ========================================================================================
// test_tsolvers_tik.cpp
// ========================================================================================
/**
 * @file test_tsolvers_tik.cpp
 *
 * @brief Unit tests for the native sparse-K Tikhonov solver (TIKNET):
 * degenerate anchors (lambda2 = 0 collapses, group-mean K == native
 * TIENET, K = I == singleton-group TIENET), crossing-support equivalence
 * against the CCD GROUP_CENTERING reference,
 * save/load/continue round trips, drop-recovery semantics (parity, re-entry,
 * knot monotonicity, mid-recovery checkpoints), rank-bound continuation,
 * and penalty-matrix content validation across the sparse-K trio.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <algorithm>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdienet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdtiknet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tiknet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tomp_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::ad::linear_model {

using namespace trex::tsolvers::ad::linear_model::cd_based;
using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::tsolvers::ad::linear_model::omp_based;
using namespace trex::utils::datageneration::datagen;
using trex::tsolvers::ad::SparseBetaPath;

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


/** @brief Group-mean IEN matrix W = sum_m 1_m 1_m^T / p_m for contiguous
 *  groups of size gsize (entries 1/gsize within each group block). */
Eigen::SparseMatrix<double> groupMeanK(Eigen::Index p, Eigen::Index gsize) {
    Eigen::SparseMatrix<double> W(p, p);
    std::vector<Eigen::Triplet<double>> trip;
    for (Eigen::Index i = 0; i < p; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            if (i / gsize == j / gsize) {
                trip.emplace_back(i, j, 1.0 / static_cast<double>(gsize));
            }
        }
    }
    W.setFromTriplets(trip.begin(), trip.end());
    return W;
}


/** @brief Within-group centering matrix K_GC = blockdiag(I - J/p_m) for
 *  contiguous groups of size gsize (idempotent projector). */
Eigen::SparseMatrix<double> groupCenteringK(Eigen::Index p,
                                            Eigen::Index gsize) {
    Eigen::SparseMatrix<double> K(p, p);
    std::vector<Eigen::Triplet<double>> trip;
    const double w = 1.0 / static_cast<double>(gsize);
    for (Eigen::Index i = 0; i < p; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            if (i / gsize == j / gsize) {
                const double delta = (i == j) ? 1.0 : 0.0;
                trip.emplace_back(i, j, delta - w);
            }
        }
    }
    K.setFromTriplets(trip.begin(), trip.end());
    return K;
}


/** @brief Identity K (p x p). */
Eigen::SparseMatrix<double> identityK(Eigen::Index p) {
    Eigen::SparseMatrix<double> K(p, p);
    K.setIdentity();
    return K;
}


/** @brief Center every column (and y) and scale the columns to unit
 *  L2-norm in place — the solvers' working scale, applied externally for
 *  the augmented cross-check (penalty rows must stay raw). */
void standardizeInPlace(Eigen::MatrixXd& X, Eigen::MatrixXd& D,
                        Eigen::VectorXd& y) {
    auto scale_cols = [](Eigen::MatrixXd& M) {
        for (Eigen::Index j = 0; j < M.cols(); ++j) {
            M.col(j).array() -= M.col(j).mean();
            const double nrm = M.col(j).norm();
            if (nrm > 0.0) { M.col(j) /= nrm; }
        }
    };
    scale_cols(X);
    scale_cols(D);
    y.array() -= y.mean();
}


/** @brief Drop-forcing design (the proven recipe from
 *  test_tsolvers_execution.cpp): cols 1 and 2 near-collinear with col 0 and
 *  opposing signal signs provoke coefficient sign changes, hence LASSO
 *  drops (and recovery steps) on the path. */
struct DropDesign {
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;
    DropDesign(Eigen::Index p, unsigned seed,
               double c1 = 0.95, double c2 = 0.90, double noise = 0.1) {
        std::srand(seed);
        X = Eigen::MatrixXd::Random(200, p);
        X.col(1) = c1 * X.col(0) + (1.0 - c1) * X.col(1);
        X.col(2) = c2 * X.col(0) + (1.0 - c2) * X.col(2);
        D = Eigen::MatrixXd::Random(200, p);
        y = 5.0 * X.col(0) - 4.0 * X.col(1) + 2.0 * X.col(2) +
            noise * Eigen::VectorXd::Random(200);
    }
};


/** @brief Portable twin of DropDesign: the same construction drawn from
 *  mt19937 + uniform_real_distribution, whose generate_canonical the
 *  standard specifies exactly, instead of std::rand behind Eigen::Random,
 *  which differs between glibc and the macOS libc. The design is therefore
 *  bit-identical on every platform, and a seed that produces a given path
 *  event (here: a checkpoint mid-recovery) on one platform produces it on
 *  all. Filled with explicit loops so the draw order is fixed. */
struct PortableDropDesign {
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;
    PortableDropDesign(Eigen::Index p, unsigned seed,
                       double c1 = 0.95, double c2 = 0.90, double noise = 0.1) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        auto draw = [&](Eigen::Index rows, Eigen::Index cols) {
            Eigen::MatrixXd M(rows, cols);
            for (Eigen::Index c = 0; c < cols; ++c)
                for (Eigen::Index r = 0; r < rows; ++r)
                    M(r, c) = u(gen);
            return M;
        };
        X = draw(200, p);
        X.col(1) = c1 * X.col(0) + (1.0 - c1) * X.col(1);
        X.col(2) = c2 * X.col(0) + (1.0 - c2) * X.col(2);
        D = draw(200, p);
        y = 5.0 * X.col(0) - 4.0 * X.col(1) + 2.0 * X.col(2) +
            noise * draw(200, 1).col(0);
    }
};


/** @brief Small saturating design: n = 12 << p + L = 30 columns, kept
 *  UNCENTERED with unit-norm columns so that (normalize = false,
 *  intercept = false) leaves the data untouched, effective_n = n, and the
 *  rank bound max_actives = min(p + L, n + rank(K)) = 22 is exactly
 *  reachable for K = I (p = 10). The trio's collinearity is milder than
 *  DropDesign's — at n = 12 the 0.95 mixing triggers Cholesky collinearity
 *  rejections that shrink the reachable rank below the bound. */
struct RankBoundDesign {
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;
    RankBoundDesign(unsigned seed, bool collinear_trio) {
        std::srand(seed);
        X = Eigen::MatrixXd::Random(12, 10);
        D = Eigen::MatrixXd::Random(12, 20);
        if (collinear_trio) {
            X.col(1) = 0.8 * X.col(0) + 0.2 * X.col(1);
            X.col(2) = 0.7 * X.col(0) + 0.3 * X.col(2);
            y = 5.0 * X.col(0) - 4.0 * X.col(1) + 2.0 * X.col(2) +
                0.1 * Eigen::VectorXd::Random(12);
        } else {
            y = 3.0 * X.col(0) - 2.0 * X.col(1) + X.col(2) +
                0.1 * Eigen::VectorXd::Random(12);
        }
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            X.col(j) /= X.col(j).norm();
        }
        for (Eigen::Index j = 0; j < D.cols(); ++j) {
            D.col(j) /= D.col(j).norm();
        }
    }
};


/** @brief Max TIK-modified KKT residual over the active set for K = I on
 *  the working scale: max_a |x_a^T (y - [X D] beta) - lambda2 * s_{b(a)}|
 *  with s(b) the folded coefficient stack of base b. At the end of a
 *  complete homotopy (terminal LS step taken) this vanishes; a premature
 *  bound exit leaves it at the last knot level. */
double activeKktResidualIdentityK(const Eigen::MatrixXd& X,
                                  const Eigen::MatrixXd& D,
                                  const Eigen::VectorXd& y,
                                  const Eigen::VectorXd& beta,
                                  const std::vector<std::size_t>& actives,
                                  double lambda2) {
    const Eigen::Index p = X.cols();
    Eigen::VectorXd r = y - X * beta.head(p) - D * beta.tail(D.cols());
    Eigen::VectorXd s = beta.head(p);
    for (Eigen::Index t = 0; t < D.cols(); ++t) {
        s(t % p) += beta(p + t);
    }
    double kkt = 0.0;
    for (std::size_t a : actives) {
        const auto ai = static_cast<Eigen::Index>(a);
        const Eigen::VectorXd col = ai < p ? X.col(ai) : D.col(ai - p);
        const double c = col.dot(r) - lambda2 * s(ai < p ? ai : (ai - p) % p);
        kkt = std::max(kkt, std::abs(c));
    }
    return kkt;
}

} // namespace

// ========================================================================================

class TSolverTikTest : public ::testing::Test {
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
     *  run (K travels as triplets; the coupling state is restored or
     *  rebuilt). */
    template <class SolverT, class MakeFn>
    void expectRoundTripMatchesContinuous(MakeFn make, const std::string& tag,
                                          std::size_t T_partial = 3,
                                          std::size_t T_final = 6) {
        Mapped mr(X, D, y);
        SolverT ref = make(mr);
        ref.executeStep(T_final, true);

        Mapped mp(X, D, y);
        const std::string file =
            ::testing::TempDir() + "tik_ckpt_" + tag + ".bin";
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

        std::filesystem::remove(file);
    }
};

// ========================================================================================
// TIKNET degenerate anchors
// ========================================================================================

/** @brief lambda2 == 0 collapses T-TIKNET exactly onto T-LASSO (every
 *  coupling term multiplies to literal zero; K is carried but inert). */
TEST_F(TSolverTikTest, TiknetLambda2ZeroReducesToTlasso) {
    Mapped mt(X, D, y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, groupCenteringK(X.cols(), 5));
    tik.executeStep(5, true);

    Mapped ml(X, D, y);
    TLASSO_Solver lasso(ml.Xm, ml.Dm, ml.ym);
    lasso.executeStep(5, true);

    EXPECT_EQ(tik.getActions(), lasso.getActions());
    EXPECT_TRUE(tik.getBeta(-1).isApprox(lasso.getBeta(-1), 1e-9));
}


/** @brief With K = sum_m 1_m 1_m^T / p_m the LARS-TIK recursion computes
 *  the identical modified correlations, Gram couplings, and joining rates
 *  as the native LARS-IEN algorithm — TIENET becomes the special case.
 *  (Accumulation orders differ in the joining rate, so knife-edge designs
 *  could flip at ULP margins; this fixture is wide-margin and the action
 *  sequence is pinned exactly.) */
TEST_F(TSolverTikTest, TiknetGroupMeanKMatchesTienet) {
    const double lambda2 = 1.0;

    Mapped mt(X, D, y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, groupMeanK(X.cols(), 5),
        trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    tik.executeStep(5, true);

    Mapped mi(X, D, y);
    TIENET_Solver ien(mi.Xm, mi.Dm, mi.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    ien.executeStep(5, true);

    EXPECT_EQ(tik.getActions(), ien.getActions());
    EXPECT_TRUE(tik.getBeta(-1).isApprox(ien.getBeta(-1), 1e-8));
}


/** @brief K = I equals TIENET with all-singleton groups: each variable is
 *  ridged jointly with its own dummy copies through the folded sum (NOT
 *  the decoupled TENET ridge — the same collapse/coupling trade-off as on
 *  the CCD side). */
TEST_F(TSolverTikTest, TiknetIdentityKMatchesTienetSingletonGroups) {
    const double lambda2 = 0.5;

    Mapped mt(X, D, y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, identityK(X.cols()), trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    tik.executeStep(5, true);

    Eigen::VectorXi singleton = Eigen::VectorXi::LinSpaced(
        X.cols(), 0, static_cast<int>(X.cols()) - 1);
    Mapped mi(X, D, y);
    TIENET_Solver ien(mi.Xm, mi.Dm, mi.ym, singleton, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    ien.executeStep(5, true);

    EXPECT_EQ(tik.getActions(), ien.getActions());
    EXPECT_TRUE(tik.getBeta(-1).isApprox(ien.getBeta(-1), 1e-9));
}


// ========================================================================================
// TIKNET vs the CCD reference: crossing-support equivalence
// ========================================================================================

/** @brief T-TIKNET with K_GC and the CCD GROUP_CENTERING mode solve the
 *  same penalized-lasso path family from opposite algorithmic sides
 *  (piecewise-linear homotopy vs exact fixed-lambda1 minimizers); the real
 *  supports at each dummy-count crossing (what T-Rex consumes) must
 *  coincide. */
TEST_F(TSolverTikTest, TiknetKgcCrossingSupportsMatchCcdGroupCentering) {
    const std::size_t T = 5;
    const std::size_t p = static_cast<std::size_t>(X.cols());
    const double lambda2 = 2.0;

    Mapped mt(X, D, y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, groupCenteringK(X.cols(), 5),
        trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    tik.executeStep(T, true);

    Mapped mc(X, D, y);
    TCDIENET_Solver ccd(mc.Xm, mc.Dm, mc.ym, groups, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2,
        .group_penalty = GroupPenaltyVariant::CENTERING});
    ccd.setLambdaRelTol(1e-6);
    ccd.executeStep(T, true);

    const auto sup_tik = supportsAtDummyCounts(tik.getBetaPathSparse(), p, T);
    const auto sup_ccd = supportsAtDummyCounts(ccd.getBetaPathSparse(), p, T);

    for (std::size_t t = 1; t <= T; ++t) {
        EXPECT_EQ(sup_tik[t - 1], sup_ccd[t - 1])
            << "TIKNET vs CCD GROUP_CENTERING real support diverged at "
               "dummy count t = " << t;
    }
}


// ========================================================================================
// Serialization: save -> load -> continue
// ========================================================================================

/** @brief T-TIKNET round trip: K travels as triplets; the folded coupling
 *  state (s, K s) is part of the archive. */
TEST_F(TSolverTikTest, SerializeLoadContinueMatchesContinuous_TIKNET) {
    const Eigen::Index p = X.cols();
    expectRoundTripMatchesContinuous<TIKNET_Solver>(
        [p](Mapped& m) {
            return TIKNET_Solver(m.Xm, m.Dm, m.ym, groupCenteringK(p, 5),
                trex::tsolvers::ad::SolverOptions{.lambda2 = 2.0});
        },
        "tiknet");
}


// ========================================================================================
// Drop recovery (R-lars drop-breakpoint semantics)
// ========================================================================================

/** @brief TIKNET's drop recovery against the CRAN-validated TIENET recovery
 *  on the same problem (K = group-mean W, collinear trio inside group 0): a
 *  path with drops must agree in the full signed action stream — including
 *  the empty no-ADD recovery entries and any cascading drops — and in the
 *  final coefficients. (Accumulation orders differ, so the fixture must
 *  stay wide-margin.) */
TEST_F(TSolverTikTest, TiknetGroupMeanKMatchesTienetOnDropPath) {
    DropDesign dd(20, 3);
    const double lambda2 = 1.0;

    Mapped mt(dd.X, dd.D, dd.y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, groupMeanK(20, 5), trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    tik.executeStep(0, false);
    ASSERT_GT(tik.getNumRemovals(), 0u)
        << "fixture no longer provokes drops; retune the seed";

    Eigen::VectorXi g20(20);
    for (Eigen::Index j = 0; j < 20; ++j) {
        g20(j) = static_cast<int>(j / 5);
    }
    Mapped mi(dd.X, dd.D, dd.y);
    TIENET_Solver ien(mi.Xm, mi.Dm, mi.ym, g20, trex::tsolvers::ad::SolverOptions{.lambda2 = lambda2});
    ien.executeStep(0, false);

    EXPECT_EQ(tik.getActions(), ien.getActions());
    EXPECT_TRUE(tik.getBeta(-1).isApprox(ien.getBeta(-1), 1e-8));
}


/** @brief A dropped path must show the R-lars recovery observables: knots
 *  stay monotone, the step after a drop is a no-ADD recovery step (the only
 *  source of EMPTY action entries on a full-path run), the dropped
 *  coefficient is exactly zero at the drop step, and a dropped variable
 *  genuinely re-enters later. */
TEST_F(TSolverTikTest, TiknetDropRecoveryReentersAndKeepsLambdaMonotone) {
    DropDesign dd(20, 3);
    Mapped mt(dd.X, dd.D, dd.y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, groupMeanK(20, 5), trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0});
    tik.executeStep(0, false);
    ASSERT_GT(tik.getNumRemovals(), 0u)
        << "fixture no longer provokes drops; retune the seed";

    const auto& lambda = tik.getLambda();
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LE(lambda[k], lambda[k - 1] + 1e-8)
            << "lambda jumped upward at step " << k;
    }

    const auto& actions = tik.getActions();
    EXPECT_EQ(actions.size(), tik.getRSS().size() - 1);
    EXPECT_EQ(tik.getRSS().size(), tik.getDoF().size());
    EXPECT_EQ(actions.size(), lambda.size());

    bool any_empty = false;
    for (std::size_t k = 0; k < actions.size(); ++k) {
        if (actions[k].empty()) { any_empty = true; }
        bool has_drop = false;
        for (int act : actions[k]) {
            if (act < 0) { has_drop = true; }
        }
        if (has_drop && k + 1 < actions.size()) {
            for (int act : actions[k + 1]) {
                EXPECT_LE(act, 0)
                    << "ADD in the step right after a drop (recovery "
                       "semantics violated) at step " << k + 1;
            }
        }
    }
    EXPECT_TRUE(any_empty) << "no recovery step recorded";

    Eigen::MatrixXd path = tik.getBetaPathSparse().dense();
    bool reentry = false;
    for (std::size_t k = 0; k < actions.size(); ++k) {
        for (int act : actions[k]) {
            if (act >= 0) { continue; }
            const int v = -act - 1;
            EXPECT_EQ(path(v, static_cast<Eigen::Index>(k) + 1), 0.0)
                << "dropped variable " << v << " has nonzero beta at step "
                << k + 1;
            for (std::size_t k2 = k + 1; k2 < actions.size() && !reentry;
                 ++k2) {
                for (int act2 : actions[k2]) {
                    if (act2 == v + 1) { reentry = true; }
                }
            }
        }
    }
    EXPECT_TRUE(reentry)
        << "no dropped variable re-entered; retune the seed";
}


/** @brief Two collinear trios double the drop pressure; the path must keep
 *  its unconditional invariants regardless of how the drops interleave.
 *  Cascading/simultaneous drops are covered opportunistically here (the
 *  deterministic guarantee is the TIENET parity above): whenever a step
 *  drops two or more variables, the following step must not ADD. */
TEST_F(TSolverTikTest, TiknetMultiDropPathStaysAligned) {
    DropDesign dd(20, 7);
    dd.X.col(11) = 0.95 * dd.X.col(10) + 0.05 * dd.X.col(11);
    dd.X.col(12) = 0.90 * dd.X.col(10) + 0.10 * dd.X.col(12);
    dd.y += 5.0 * dd.X.col(10) - 4.0 * dd.X.col(11) + 2.0 * dd.X.col(12);

    Mapped mt(dd.X, dd.D, dd.y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, groupMeanK(20, 5), trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0});
    tik.executeStep(0, false);
    ASSERT_GT(tik.getNumRemovals(), 0u)
        << "fixture no longer provokes drops; retune the seed";

    const auto& actions = tik.getActions();
    const auto& lambda = tik.getLambda();
    EXPECT_EQ(actions.size(), tik.getRSS().size() - 1);
    EXPECT_EQ(tik.getRSS().size(), tik.getDoF().size());
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LE(lambda[k], lambda[k - 1] + 1e-8)
            << "lambda jumped upward at step " << k;
    }

    Eigen::MatrixXd path = tik.getBetaPathSparse().dense();
    for (std::size_t k = 0; k < actions.size(); ++k) {
        std::size_t num_drops = 0;
        for (int act : actions[k]) {
            if (act < 0) {
                ++num_drops;
                EXPECT_EQ(path(-act - 1, static_cast<Eigen::Index>(k) + 1),
                          0.0)
                    << "dropped variable " << (-act - 1)
                    << " has nonzero beta at step " << k + 1;
            }
        }
        if (num_drops >= 2 && k + 1 < actions.size()) {
            for (int act : actions[k + 1]) {
                EXPECT_LE(act, 0)
                    << "ADD right after a simultaneous drop at step "
                    << k + 1;
            }
        }
    }
}


/** @brief Save/load/continue with checkpoints swept across a drop-rich
 *  horizon: drop_recovery_pending_ and last_dropped_vars_ travel in the
 *  archive, so a checkpoint taken exactly at a drop step (pending recovery)
 *  must resume onto the identical path. The probe run identifies which
 *  checkpoints land mid-recovery (its final action holds a drop) and the
 *  sweep must hit at least one. */
TEST_F(TSolverTikTest, SerializeLoadContinueAcrossDropRecovery_TIKNET) {
    // Whether a probe stopped by its dummy count ends on a drop step depends
    // on the drawn data, so the design is the PortableDropDesign (identical
    // on every platform; DropDesign's std::rand data differ between glibc
    // and the macOS libc, and its seed 12 hit only on the latter). The
    // configuration below was found by a search over seeds, signal columns
    // and amplitudes: the T = 2 probe ends on a drop step.
    PortableDropDesign dd(20, 75);
    dd.y += 5.0 * dd.D.col(5);  // dummy signal pulls a dummy admission
                                // onto a drop step (pending checkpoint)
    X = dd.X;   // the round-trip helper reads the fixture members
    D = dd.D;
    y = dd.y;

    bool any_pending_checkpoint = false;
    for (std::size_t T_partial = 1; T_partial <= 5; ++T_partial) {
        {
            Mapped mp(X, D, y);
            TIKNET_Solver probe(mp.Xm, mp.Dm, mp.ym, groupMeanK(20, 5),
                trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0});
            probe.executeStep(T_partial, true);
            if (!probe.getActions().empty()) {
                for (int act : probe.getActions().back()) {
                    if (act < 0) { any_pending_checkpoint = true; }
                }
            }
        }
        expectRoundTripMatchesContinuous<TIKNET_Solver>(
            [](Mapped& m) {
                return TIKNET_Solver(m.Xm, m.Dm, m.ym, groupMeanK(20, 5),
                    trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0});
            },
            "tiknet_droprec_T" + std::to_string(T_partial), T_partial, 6);
    }
    EXPECT_TRUE(any_pending_checkpoint)
        << "no checkpoint landed with a pending recovery; retune the sweep";
}


/** @brief Near-tie recovery roots: with amplified collinearity the
 *  just-dropped variable sits within fp noise of the equicorrelation level
 *  it left. The capGammaByExcluded floor must discard exactly that noise
 *  root — knots stay monotone and the dropped variable is never re-added in
 *  the recovery step immediately following its drop (Efron's lemma). */
TEST_F(TSolverTikTest, TiknetExcludedCatchUpKeepsKnotsMonotone) {
    DropDesign dd(20, 11, 0.99, 0.98, 0.05);
    Mapped mt(dd.X, dd.D, dd.y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, groupMeanK(20, 5), trex::tsolvers::ad::SolverOptions{.lambda2 = 0.5});
    tik.executeStep(0, false);
    ASSERT_GT(tik.getNumRemovals(), 0u)
        << "fixture no longer provokes drops; retune the seed";

    const auto& lambda = tik.getLambda();
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LE(lambda[k], lambda[k - 1] + 1e-8)
            << "lambda jumped upward at step " << k;
    }

    const auto& actions = tik.getActions();
    for (std::size_t k = 0; k + 1 < actions.size(); ++k) {
        for (int act : actions[k]) {
            if (act < 0) {
                for (int act2 : actions[k + 1]) {
                    EXPECT_NE(act2, -act)
                        << "immediate fp-noise re-add of dropped variable "
                        << (-act - 1) << " at step " << k + 1;
                }
            }
        }
    }
}


// ========================================================================================
// Rank-bound continuation (terminal LS segment with joining disabled)
// ========================================================================================

/** @brief Once the active set saturates min(p + L, n + rank(K)), the path
 *  must still take the terminal LS segment of the saturated system instead
 *  of exiting at a phantom joining breakpoint: the TIK-modified active
 *  correlations vanish at the end. */
TEST_F(TSolverTikTest, TiknetRankBoundTakesTerminalStep) {
    RankBoundDesign rb(2, false);
    const double lambda2 = 0.5;

    Mapped m(rb.X, rb.D, rb.y);
    TIKNET_Solver tik(m.Xm, m.Dm, m.ym, identityK(10), trex::tsolvers::ad::SolverOptions{.normalize = false,
        .intercept = false, .lambda2 = lambda2});
    tik.executeStep(0, false);

    const auto& dof = tik.getDoF();
    const std::size_t max_dof =
        *std::max_element(dof.begin(), dof.end());
    ASSERT_EQ(max_dof, 22u) << "rank bound not reached; retune the design";

    const auto& lambda = tik.getLambda();
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LE(lambda[k], lambda[k - 1] + 1e-8)
            << "lambda jumped upward at step " << k;
    }

    EXPECT_LT(activeKktResidualIdentityK(m.Xc, m.Dc, m.yc, tik.getBeta(-1),
                                         tik.getActives(), lambda2),
              1e-6)
        << "path ended before the terminal LS segment of the saturated "
           "active system";
}


/** @brief A complete homotopy at the bound: after saturation, either a zero
 *  crossing truncates the terminal segment (drop action at/after the
 *  saturating step, path continues) or the untruncated terminal step ends
 *  the path on the saturated LS solution. One of the two must happen. */
TEST_F(TSolverTikTest, TiknetRankBoundDropContinuesPath) {
    RankBoundDesign rb(14, true);
    const double lambda2 = 0.5;

    Mapped m(rb.X, rb.D, rb.y);
    TIKNET_Solver tik(m.Xm, m.Dm, m.ym, identityK(10), trex::tsolvers::ad::SolverOptions{.normalize = false,
        .intercept = false, .lambda2 = lambda2});
    tik.executeStep(0, false);

    const auto& dof = tik.getDoF();
    const std::size_t max_dof =
        *std::max_element(dof.begin(), dof.end());
    ASSERT_EQ(max_dof, 22u) << "rank bound not reached; retune the design";
    EXPECT_GT(tik.getNumRemovals(), 0u)
        << "no drop on the path; retune the seed";

    const auto& lambda = tik.getLambda();
    for (std::size_t k = 1; k < lambda.size(); ++k) {
        EXPECT_LE(lambda[k], lambda[k - 1] + 1e-8)
            << "lambda jumped upward at step " << k;
    }

    const auto& actions = tik.getActions();
    std::size_t k_sat = actions.size();
    for (std::size_t k = 0; k + 1 < dof.size(); ++k) {
        if (dof[k + 1] == 22u) {
            k_sat = k;
            break;
        }
    }
    ASSERT_LT(k_sat, actions.size());
    bool drop_at_or_after_sat = false;
    for (std::size_t k = k_sat; k < actions.size(); ++k) {
        for (int act : actions[k]) {
            if (act < 0) { drop_at_or_after_sat = true; }
        }
    }
    const double kkt = activeKktResidualIdentityK(
        m.Xc, m.Dc, m.yc, tik.getBeta(-1), tik.getActives(), lambda2);
    EXPECT_TRUE(drop_at_or_after_sat || kkt < 1e-6)
        << "neither a drop nor the terminal LS step occurred at the bound "
           "(kkt = " << kkt << ")";
}


/** @brief The K = I == singleton-group TIENET collapse must survive
 *  saturation: both solvers share the bound n + p here by construction
 *  (TIKNET counts nonzero K columns = p, TIENET counts groups = p), so the
 *  saturated terminal/drop semantics must agree action for action. */
TEST_F(TSolverTikTest, TiknetIdentityKMatchesTienetSingletonGroupsAtRankBound) {
    RankBoundDesign rb(14, true);
    const double lambda2 = 0.5;

    Mapped mt(rb.X, rb.D, rb.y);
    TIKNET_Solver tik(mt.Xm, mt.Dm, mt.ym, identityK(10), trex::tsolvers::ad::SolverOptions{.normalize = false,
        .intercept = false, .lambda2 = lambda2});
    tik.executeStep(0, false);

    Eigen::VectorXi singleton = Eigen::VectorXi::LinSpaced(10, 0, 9);
    Mapped mi(rb.X, rb.D, rb.y);
    TIENET_Solver ien(mi.Xm, mi.Dm, mi.ym, singleton, trex::tsolvers::ad::SolverOptions{.normalize = false,
        .intercept = false, .lambda2 = lambda2});
    ien.executeStep(0, false);

    EXPECT_EQ(tik.getActions(), ien.getActions());
    EXPECT_TRUE(tik.getBeta(-1).isApprox(ien.getBeta(-1), 1e-8));
}


// ========================================================================================
// Penalty-matrix content validation (shared sparse-K trio helper)
// ========================================================================================

/** @brief TIKNET rejects non-finite, asymmetric, and non-PSD K at
 *  construction — before any Cholesky pivot can evaluate sqrt() on an
 *  invalid negative value. */
TEST_F(TSolverTikTest, TiknetRejectsInvalidTikhonovK) {
    const Eigen::Index p = X.cols();
    {
        Eigen::SparseMatrix<double> K = identityK(p);
        K.coeffRef(3, 7) = std::numeric_limits<double>::quiet_NaN();
        K.coeffRef(7, 3) = std::numeric_limits<double>::quiet_NaN();
        Mapped m(X, D, y);
        EXPECT_THROW(
            TIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}),
            std::invalid_argument);
    }
    {
        Eigen::SparseMatrix<double> K = identityK(p);
        K.coeffRef(0, 1) = 0.5;
        Mapped m(X, D, y);
        EXPECT_THROW(
            TIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}),
            std::invalid_argument);
    }
    {
        Eigen::SparseMatrix<double> K = identityK(p);
        K.coeffRef(p - 1, p - 1) = -1.0;
        Mapped m(X, D, y);
        EXPECT_THROW(
            TIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}),
            std::invalid_argument);
    }
    {
        Eigen::SparseMatrix<double> K = -1.0 * identityK(p);
        Mapped m(X, D, y);
        EXPECT_THROW(
            TIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}),
            std::invalid_argument);
    }
}


/** @brief The three standard penalties pass validation — including the
 *  group-centering projector, whose zero eigenvalues sit exactly on the
 *  PSD boundary (the tol-shift must keep it clear of the LDLT zero-pivot
 *  edge). */
TEST_F(TSolverTikTest, TiknetAcceptsValidTikhonovK) {
    const Eigen::Index p = X.cols();
    {
        Mapped m(X, D, y);
        EXPECT_NO_THROW(TIKNET_Solver(m.Xm, m.Dm, m.ym, identityK(p),
            trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}));
    }
    {
        Mapped m(X, D, y);
        EXPECT_NO_THROW(TIKNET_Solver(m.Xm, m.Dm, m.ym, groupMeanK(p, 5),
            trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}));
    }
    {
        Mapped m(X, D, y);
        EXPECT_NO_THROW(TIKNET_Solver(m.Xm, m.Dm, m.ym, groupCenteringK(p, 5),
            trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}));
    }
}


/** @brief TCDTIKNET shares the trio validation contract. */
TEST_F(TSolverTikTest, TcdtiknetRejectsInvalidTikhonovK) {
    const Eigen::Index p = X.cols();
    {
        Eigen::SparseMatrix<double> K = identityK(p);
        K.coeffRef(3, 7) = std::numeric_limits<double>::quiet_NaN();
        K.coeffRef(7, 3) = std::numeric_limits<double>::quiet_NaN();
        Mapped m(X, D, y);
        EXPECT_THROW(
            TCDTIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}),
            std::invalid_argument);
    }
    {
        Eigen::SparseMatrix<double> K = identityK(p);
        K.coeffRef(0, 1) = 0.5;
        Mapped m(X, D, y);
        EXPECT_THROW(
            TCDTIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}),
            std::invalid_argument);
    }
    {
        Eigen::SparseMatrix<double> K = -1.0 * identityK(p);
        Mapped m(X, D, y);
        EXPECT_THROW(
            TCDTIKNET_Solver(m.Xm, m.Dm, m.ym, K, trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}),
            std::invalid_argument);
    }
    {
        Mapped m(X, D, y);
        EXPECT_NO_THROW(TCDTIKNET_Solver(m.Xm, m.Dm, m.ym, groupCenteringK(p, 5),
            trex::tsolvers::ad::SolverOptions{.lambda2 = 1.0}));
    }
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::ad::linear_model */
