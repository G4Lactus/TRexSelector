// ========================================================================================
// test_trex_tikhonov.cpp
// ========================================================================================
/**
 * @file test_trex_tikhonov.cpp
 *
 * @brief Unit tests for TRexTikhonovSelector (general-Tikhonov T-Rex):
 * construction validation, the K = I collapse onto the TCENET base run
 * (i.i.d.-dummy escape hatch), the cluster-aware dummy path (gates, layer
 * correlation, prior-groups end-to-end), planted-group recovery with an
 * informed K, and the CV lambda_2 path.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// std includes
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>
#include <trex_selector_methods/trex_tikhonov/trex_tikhonov.hpp>
#include <trex_selector_methods/trex_utils/trex_cluster_dummies.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_tikhonov {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_tikhonov;
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace cdum = trex::trex_selector_methods::utils::cluster_dummies;
namespace gvs  = trex::trex_selector_methods::trex_gvs;

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
 *  elastic net, and with the two validation escape hatches —
 *  `cluster_dummies = false` (i.i.d. dummy draws) and
 *  `fold_dummy_coupling = false` (INDEPENDENT_RIDGE dummy block) —
 *  TRexTikhonovSelector reuses the base machinery exactly, so a run with
 *  identical seed and lambda_2 must reproduce the plain TRexSelector +
 *  TCENET selection. This certifies the sparse-K solver dispatch and
 *  penalty math; the DEFAULT path (cluster dummies + FOLDED coupling) is
 *  covered by the ClusterDummies_* tests and the GVS-TCIENET anchor. */
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
        ctrl.cluster_dummies     = false;  // i.i.d. dummies: base parity
        ctrl.fold_dummy_coupling = false;  // independent dummy ridge: TCENET
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


// ========================================================================================
// Cluster-aware dummies (default path, GVS parity)
// ========================================================================================

/** @brief Configurations the cluster-aware dummy path cannot serve must be
 *  rejected at construction: the PERMUTATION-family strategies (K
 *  row-permuted copies of one base cannot carry the per-cluster MVN
 *  convention), malformed prior_groups, and an out-of-range corr_max.
 *  Everything else — including SEEDED and memory mapping — is served
 *  natively by the DummyGenerator cluster mode. */
TEST_F(TRexTikhonovTest, ClusterDummies_GateThrowsOnUnsupportedConfig) {

    TRexTikhonovControlParameter base;
    base.tikhonov_K = identityK(10);

    {   // PERMUTATION cannot carry cluster-MVN dummies.
        auto ctrl = base;
        ctrl.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION;
        EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                     std::invalid_argument);
    }
    {   // PERMUTATION_SEEDED likewise.
        auto ctrl = base;
        ctrl.trex_ctrl.lloop_strategy = LLoopStrategy::PERMUTATION_SEEDED;
        EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                     std::invalid_argument);
    }
    {   // SEEDED and memory mapping are supported.
        auto ctrl = base;
        ctrl.trex_ctrl.lloop_strategy = LLoopStrategy::SEEDED;
        ctrl.trex_ctrl.use_memory_mapping = true;
        EXPECT_NO_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl));
    }
    {   // prior_groups length mismatch (p = 10 here).
        auto ctrl = base;
        ctrl.prior_groups.assign(9, 0);
        EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                     std::invalid_argument);
    }
    {   // Non-contiguous cluster IDs (ID 1 missing).
        auto ctrl = base;
        ctrl.prior_groups.assign(10, 0);
        ctrl.prior_groups[5] = 2;
        EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                     std::invalid_argument);
    }
    {   // corr_max out of range (HAC route).
        auto ctrl = base;
        ctrl.corr_max = 1.5;
        EXPECT_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl),
                     std::invalid_argument);
    }
    {   // With cluster_dummies off, the gated knobs are inert.
        auto ctrl = base;
        ctrl.cluster_dummies = false;
        ctrl.trex_ctrl.lloop_strategy = LLoopStrategy::SEEDED;
        EXPECT_NO_THROW(TRexTikhonovSelector(X_map, y_map, 0.1, ctrl));
    }
}


/** @brief The shared layer draw reproduces the per-cluster correlation of
 *  the working-scale X: strong within-cluster, near-zero across clusters. */
TEST_F(TRexTikhonovTest, ClusterDummies_LayerCarriesClusterCorrelation) {
    const Eigen::Index n = 200, p = 20;
    GroupedData d(n, p, 0.9, 1.0, 3);   // within-group corr = 0.81

    // Selector working scale: centered + unit-L2 columns.
    Eigen::MatrixXd Xs = d.X;
    Xs.rowwise() -= Xs.colwise().mean();
    for (Eigen::Index j = 0; j < p; ++j) { Xs.col(j) /= Xs.col(j).norm(); }
    Eigen::Map<Eigen::MatrixXd> Xm(Xs.data(), n, p);

    std::vector<std::vector<Eigen::Index>> clusters;
    for (Eigen::Index g = 0; g < p; g += 5) {
        std::vector<Eigen::Index> c;
        for (Eigen::Index j = g; j < g + 5; ++j) { c.push_back(j); }
        clusters.push_back(std::move(c));
    }

    const auto chol = cdum::buildClusterCholeskys(Xm, clusters, n, "test");
    std::mt19937 rng(123);
    Eigen::MatrixXd L = cdum::drawClusterDummyLayer(
        static_cast<std::size_t>(n), static_cast<std::size_t>(p),
        clusters, chol, rng);
    L.rowwise() -= L.colwise().mean();

    auto corr = [&L](Eigen::Index a, Eigen::Index b) {
        return L.col(a).dot(L.col(b)) / (L.col(a).norm() * L.col(b).norm());
    };

    double min_within = 1.0;
    double max_cross  = 0.0;
    for (Eigen::Index a = 0; a < p; ++a) {
        for (Eigen::Index b = a + 1; b < p; ++b) {
            const double c = std::abs(corr(a, b));
            if (a / 5 == b / 5) {
                min_within = std::min(min_within, c);
            } else {
                max_cross = std::max(max_cross, c);
            }
        }
    }
    EXPECT_GT(min_within, 0.6)
        << "Within-cluster dummy correlation too weak (design corr 0.81).";
    EXPECT_LT(max_cross, 0.35)
        << "Cross-cluster dummy correlation too strong (expected ~0).";
}


/** @brief Under cluster dummies the layer draws depend only on
 *  (seed, k, L): SEEDED (one D_k at a time, serial) produces content
 *  bit-identical to STANDARD at the same L, so both runs must select
 *  identically. */
TEST_F(TRexTikhonovTest, ClusterDummies_SeededMatchesStandard) {
    const Eigen::Index n = 150, p = 40;

    auto run_with = [&](LLoopStrategy strategy) {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

        TRexTikhonovControlParameter ctrl;
        ctrl.tikhonov_K = groupMeanK(p, 5);
        ctrl.lambda_2   = 1.0;
        ctrl.prior_groups.resize(static_cast<std::size_t>(p));
        for (Eigen::Index j = 0; j < p; ++j) {
            ctrl.prior_groups[static_cast<std::size_t>(j)] = j / 5;
        }
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        ctrl.trex_ctrl.lloop_strategy = strategy;

        TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_std = run_with(LLoopStrategy::STANDARD);
    const Eigen::VectorXi sel_od  = run_with(LLoopStrategy::SEEDED);

    ASSERT_EQ(sel_std.size(), sel_od.size());
    EXPECT_TRUE((sel_std.array() == sel_od.array()).all())
        << "SEEDED diverged from STANDARD under cluster dummies.\n"
        << "  STANDARD n_selected = " << sel_std.sum() << "\n"
        << "  SEEDED n_selected = " << sel_od.sum();
}


/** @brief The memory-mapped writers regenerate the cluster layers
 *  bit-exactly from the deterministic streams, so for every supported
 *  strategy the memory-mapped run must select identically to the
 *  in-memory run. */
TEST_F(TRexTikhonovTest, ClusterDummies_MemmapMatchesInMemory) {
    const Eigen::Index n = 150, p = 40;

    auto run_with = [&](LLoopStrategy strategy, bool use_mmap) {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

        TRexTikhonovControlParameter ctrl;
        ctrl.tikhonov_K = groupMeanK(p, 5);
        ctrl.lambda_2   = 1.0;
        ctrl.prior_groups.resize(static_cast<std::size_t>(p));
        for (Eigen::Index j = 0; j < p; ++j) {
            ctrl.prior_groups[static_cast<std::size_t>(j)] = j / 5;
        }
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        ctrl.trex_ctrl.lloop_strategy = strategy;
        ctrl.trex_ctrl.use_memory_mapping = use_mmap;

        TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const struct { LLoopStrategy s; const char* name; } strategies[] = {
        {LLoopStrategy::STANDARD, "STANDARD"},
        {LLoopStrategy::HCONCAT,  "HCONCAT"},
        {LLoopStrategy::SEEDED, "SEEDED"},
    };
    for (const auto& st : strategies) {
        const Eigen::VectorXi sel_mem  = run_with(st.s, false);
        const Eigen::VectorXi sel_mmap = run_with(st.s, true);
        ASSERT_EQ(sel_mem.size(), sel_mmap.size()) << st.name;
        EXPECT_TRUE((sel_mem.array() == sel_mmap.array()).all())
            << st.name << ": memory-mapped selection differs from the "
            << "in-memory selection.\n"
            << "  in-memory n_selected = " << sel_mem.sum() << "\n"
            << "  memmap    n_selected = " << sel_mmap.sum();
    }
}


/** @brief FOLDED dummy coupling with the group-mean K must reproduce
 *  TRexGVSSelector(IEN, TCIENET) exactly: identical prior groups, lambda_2,
 *  and seed give bit-identical cluster-dummy draws (shared makeLayerStream)
 *  and — with the folded K-coupling being the exact generalization of the
 *  GVS layered group convention — the identical solver problem, so the
 *  selected sets must coincide. This is the FOLDED-mode anchor
 *  complementing the K = I ≡ TCENET anchor of the INDEPENDENT_RIDGE mode. */
TEST_F(TRexTikhonovTest, Equivalence_FoldedGroupKMatchesGvsTcienet) {
    const Eigen::Index n = 150, p = 40;

    std::vector<Eigen::Index> prior_groups(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
        prior_groups[static_cast<std::size_t>(j)] = j / 5;
    }

    auto run_tik = [&]() {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

        TRexTikhonovControlParameter ctrl;
        ctrl.tikhonov_K   = groupMeanK(p, 5);
        ctrl.lambda_2     = 1.0;
        ctrl.prior_groups = prior_groups;
        // Defaults under test: cluster_dummies = true, fold_dummy_coupling
        // = true (the FOLDED anchor is only meaningful on the default path).
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    auto run_gvs = [&]() {
        GroupedData d(n, p, 0.75, 1.0, 7);
        Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

        gvs::TRexGVSControlParameter ctrl;
        ctrl.gvs_type     = gvs::GVSType::IEN;
        ctrl.prior_groups = prior_groups;
        ctrl.lambda_2     = 1.0;
        ctrl.trex_ctrl.solver_type = sd::SolverTypeForTRex::TCIENET;
        ctrl.trex_ctrl.K = 5;
        ctrl.trex_ctrl.max_dummy_multiplier = 2;
        gvs::TRexGVSSelector trex(Xm, ym, 0.2, ctrl, 42, false);
        return trex.select().selected_var;
    };

    const Eigen::VectorXi sel_tik = run_tik();
    const Eigen::VectorXi sel_gvs = run_gvs();

    ASSERT_EQ(sel_tik.size(), sel_gvs.size());
    EXPECT_TRUE((sel_tik.array() == sel_gvs.array()).all())
        << "FOLDED group-mean-K Tikhonov run diverged from GVS-TCIENET.\n"
        << "  Tikhonov n_selected = " << sel_tik.sum() << "\n"
        << "  GVS      n_selected = " << sel_gvs.sum();
}


/** @brief End-to-end with user-supplied prior groups: the cluster-aware
 *  default path recovers the planted actives with few false selections. */
TEST_F(TRexTikhonovTest, ClusterDummies_PriorGroupsEndToEndRecovers) {
    const Eigen::Index n = 150, p = 40;
    GroupedData d(n, p, 0.75, 1.0, 7);
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    TRexTikhonovControlParameter ctrl;
    ctrl.tikhonov_K = groupMeanK(p, 5);
    ctrl.lambda_2   = 1.0;
    ctrl.prior_groups.resize(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
        ctrl.prior_groups[static_cast<std::size_t>(j)] = j / 5;
    }
    ctrl.trex_ctrl.K = 5;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;

    TRexTikhonovSelector trex(Xm, ym, 0.2, ctrl, 42, false);
    auto result = trex.select();

    ASSERT_EQ(result.selected_var.size(), p);

    int hits = 0, false_sel = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
        if (result.selected_var(j) == 1) {
            if (j < 6) { ++hits; } else { ++false_sel; }
        }
    }
    EXPECT_GE(hits, 4)
        << "Cluster-aware dummies failed to recover the planted actives.";
    EXPECT_LE(false_sel, 4)
        << "Cluster-aware dummies selected too many null variables.";
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
