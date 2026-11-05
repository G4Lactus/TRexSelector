// ========================================================================================
/**
 * @file test_trex_lloop_seeded.cpp
 *
 * @brief Tests for the SEEDED L-loop strategies (2026-07 rename/rework).
 *
 * @details PERMUTATION_SEEDED is the stateless twin of PERMUTATION: the base
 * dummy matrix is re-derived from the seed instead of stored, and the same
 * per-experiment row-permutation engines are used (keyed on the identical
 * base id). For the same user seed the two strategies must therefore produce
 * BIT-IDENTICAL experiments and selections. SEEDED (former DIRECT) provides
 * independent per-experiment dummies without storage.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <vector>

// project includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <trex_selector_methods/ad/trex_da/trex_da.hpp>
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>
#include <trex_selector_methods/ad/trex_utils/trex_dummy_generator.hpp>
#include <utils/datageneration/utils_datagen.hpp>

// Eigen includes
#include <Eigen/Dense>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::lloop_seeded {

namespace datagen = trex::utils::datageneration::datagen;
namespace dummygen = trex::utils::datageneration::dummygen;
using trex::trex_selector_methods::ad::trex_core::TRexSelector;
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::LLoopStrategy;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::ad::utils::dummy_generator::DummyGenerator;

namespace {

/** @brief Run a small seeded T-Rex selection and return the selected set. */
std::vector<std::size_t> runSelection(LLoopStrategy strategy, int seed) {
    const Eigen::Index n = 60, p = 100;
    std::vector<std::size_t> sup = {3, 20, 45, 70, 90};
    std::vector<double> coefs(sup.size(), 2.0);

    datagen::SyntheticData data(n, p, sup, coefs, /*snr=*/3.0, /*seed=*/12345,
                                datagen::predictor_policy::Normal(),
                                datagen::noisegen::noise_policy::Normal());
    Eigen::MatrixXd X = data.getX();
    Eigen::VectorXd y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = 10;
    ctrl.max_dummy_multiplier = 3;
    ctrl.solver_type = SolverTypeForTRex::TLARS;
    ctrl.lloop_strategy = strategy;

    TRexSelector sel(X_map, y_map, /*tFDR=*/0.1, ctrl, seed, /*verbose=*/false);
    sel.select();
    return sel.getSelectedIndices();
}

} // namespace

// ========================================================================================

/** @brief PERMUTATION_SEEDED must reproduce PERMUTATION bit-exactly for
 *         the same seed (same base id, same permutation engines). */
TEST(LLoopSeeded, PermutationSeededMatchesStoredPermutation) {
    for (int seed : {7, 42, 2026}) {
        const auto stored   = runSelection(LLoopStrategy::PERMUTATION, seed);
        const auto seeded = runSelection(LLoopStrategy::PERMUTATION_SEEDED, seed);
        EXPECT_EQ(stored, seeded)
            << "PERMUTATION vs PERMUTATION_SEEDED diverged at seed " << seed;
    }
}


/** @brief PERMUTATION_SEEDED must actually PERMUTE — its per-experiment
 *         dummies must differ from SEEDED's independent draws (this is the
 *         regression test for the old bug where PERMUTATION_DIRECT silently
 *         aliased DIRECT). Checked at the DummyGenerator level. */
TEST(LLoopSeeded, SeededPermutationsAreRowPermutationsNotFreshDraws) {
    const std::size_t n = 30, p = 20;
    DummyGenerator gen(n, dummygen::Distribution::Normal(), /*seed=*/55);

    const std::uint64_t base_id = 1234u;
    const Eigen::MatrixXd base = gen.generate(p, static_cast<std::size_t>(base_id));

    for (std::size_t k = 1; k <= 4; ++k) {
        const Eigen::MatrixXd Dk = gen.permuteCopy(base, k, base_id);

        // A row permutation preserves each column's sorted entries exactly;
        // an independent fresh draw (old buggy behavior) cannot.
        for (Eigen::Index j = 0; j < Dk.cols(); ++j) {
            std::vector<double> a(Dk.col(j).data(), Dk.col(j).data() + n);
            std::vector<double> b(base.col(j).data(), base.col(j).data() + n);
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            ASSERT_EQ(a, b) << "k=" << k << ", column " << j
                            << " is not a row permutation of the base";
        }
        // ... and it must not be the identity permutation.
        EXPECT_FALSE(Dk == base) << "k=" << k << " returned the unpermuted base";
    }

    // Stateless engines must match the stored path's engines.
    gen.storeBaseDummies(base, base_id);
    for (std::size_t k = 0; k < 5; ++k) {
        EXPECT_TRUE(gen.permuteCopy(base, k, base_id) == gen.getPermuted(k, p))
            << "stateless and stored permutation engines diverge at k=" << k;
    }
}


/** @brief Deterministic reproducibility: the same seed must give the same
 *         selection twice for both SEEDED strategies (nothing stored means
 *         everything must be re-derivable). */
TEST(LLoopSeeded, SeededStrategiesAreSeedReproducible) {
    for (auto strategy : {LLoopStrategy::SEEDED,
                          LLoopStrategy::PERMUTATION_SEEDED}) {
        const auto first  = runSelection(strategy, 11);
        const auto second = runSelection(strategy, 11);
        EXPECT_EQ(first, second) << "strategy not seed-reproducible";
    }
}

namespace {

/** @brief Run a seeded core selection and return the whole result. */
TRexSelector::SelectionResult runResult(LLoopStrategy strategy, int seed, bool mmap = false) {
    const Eigen::Index n = 60, p = 100;
    std::vector<std::size_t> sup = {3, 20, 45, 70, 90};
    std::vector<double> coefs(sup.size(), 2.0);
    datagen::SyntheticData data(n, p, sup, coefs, /*snr=*/3.0, /*seed=*/12345,
                                datagen::predictor_policy::Normal(),
                                datagen::noisegen::noise_policy::Normal());
    Eigen::MatrixXd X = data.getX();
    Eigen::VectorXd y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = 10;
    ctrl.max_dummy_multiplier = 3;
    ctrl.solver_type = SolverTypeForTRex::TLARS;
    ctrl.lloop_strategy = strategy;
    ctrl.use_memory_mapping = mmap;

    TRexSelector sel(X_map, y_map, /*tFDR=*/0.1, ctrl, seed, /*verbose=*/false);
    return sel.select();
}

void expectSameResult(const TRexSelector::SelectionResult& a,
                      const TRexSelector::SelectionResult& b, const char* what) {
    EXPECT_TRUE((a.selected_var.array() == b.selected_var.array()).all()) << what;
    EXPECT_EQ(a.T_stop, b.T_stop) << what;
    EXPECT_EQ(a.dummy_factor_L, b.dummy_factor_L) << what;
    EXPECT_EQ(a.v_thresh, b.v_thresh) << what;
    ASSERT_EQ(a.Phi_mat.rows(), b.Phi_mat.rows()) << what;
    EXPECT_TRUE((a.Phi_mat.array() == b.Phi_mat.array()).all()) << what;
    ASSERT_EQ(a.FDP_hat_mat.rows(), b.FDP_hat_mat.rows()) << what;
    EXPECT_TRUE((a.FDP_hat_mat.array() == b.FDP_hat_mat.array()).all()) << what;
}

} // namespace


/** @brief SEEDED is the storage-free twin of STANDARD: the same seed gives
 *         bit-identical selections, stopping times, calibrated L and the
 *         Phi and FDP-hat matrices, for the core selector. */
TEST(LLoopSeeded, SeededMatchesStandard) {
    for (int seed : {7, 42, 2026}) {
        SCOPED_TRACE(seed);
        expectSameResult(runResult(LLoopStrategy::STANDARD, seed),
                         runResult(LLoopStrategy::SEEDED, seed),
                         "SEEDED diverged from STANDARD");
    }
}


/** @brief At the generator level: the SEEDED draw of experiment k at L step
 *         LL equals the stored STANDARD matrix of the same step, and differs
 *         from the HCONCAT (tag 0) draw it used to equal. */
TEST(LLoopSeeded, SeededDrawsAreStandardsDraws) {
    const std::size_t n = 30, p = 20, K = 4;
    DummyGenerator gen(n, dummygen::Distribution::Normal(), /*seed=*/55);
    for (std::size_t LL = 1; LL <= 3; ++LL) {
        gen.generateAndStore(K, LL * p, /*l_tag=*/LL);
        for (std::size_t k = 0; k < K; ++k) {
            const Eigen::MatrixXd seeded = gen.generateSeeded(LL * p, k, LL);
            EXPECT_TRUE(seeded == gen.getStored(k)) << "LL=" << LL << " k=" << k;
        }
    }
    gen.generateAndStore(K, p, /*l_tag=*/0);   // the HCONCAT base
    EXPECT_FALSE(gen.generateSeeded(p, 0, 1) == gen.getStored(0))
        << "SEEDED at L step 1 must not equal the prefix-stable tag-0 base";
}


/** @brief The memory-mapped and the in-memory runs of the storage-free
 *         strategies give the same result. */
TEST(LLoopSeeded, SeededStrategiesAgreeInMemoryAndMapped) {
    for (auto strategy : {LLoopStrategy::SEEDED, LLoopStrategy::PERMUTATION_SEEDED}) {
        SCOPED_TRACE(static_cast<int>(strategy));
        expectSameResult(runResult(strategy, 42, /*mmap=*/false),
                         runResult(strategy, 42, /*mmap=*/true),
                         "memory-mapped run diverged from the in-memory run");
    }
}


/** @brief SEEDED equals STANDARD in the dependency-aware selector too. */
TEST(LLoopSeeded, SeededMatchesStandardInDA) {
    namespace da = trex::trex_selector_methods::ad::trex_da;
    auto run = [](LLoopStrategy strategy) {
        const Eigen::Index n = 60, p = 100;
        std::vector<std::size_t> sup = {3, 20, 45, 70, 90};
        std::vector<double> coefs(sup.size(), 2.0);
        datagen::SyntheticData data(n, p, sup, coefs, 3.0, 12345,
                                    datagen::predictor_policy::Normal(),
                                    datagen::noisegen::noise_policy::Normal());
        Eigen::MatrixXd X = data.getX();
        Eigen::VectorXd y = data.getY();
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());
        da::TRexDAControlParameter ctrl;
        ctrl.method = da::DAMethod::AR1;
        ctrl.trex_ctrl.K = 10;
        ctrl.trex_ctrl.max_dummy_multiplier = 3;
        ctrl.trex_ctrl.lloop_strategy = strategy;
        da::TRexDASelector sel(X_map, y_map, 0.1, ctrl, 42, false);
        return sel.select();
    };
    expectSameResult(run(LLoopStrategy::STANDARD), run(LLoopStrategy::SEEDED),
                     "DA: SEEDED diverged from STANDARD");
}


/** @brief SEEDED equals STANDARD in the grouped selector (cluster dummies). */
TEST(LLoopSeeded, SeededMatchesStandardInGVS) {
    namespace gvs = trex::trex_selector_methods::ad::trex_gvs;
    auto run = [](LLoopStrategy strategy) {
        const Eigen::Index n = 60, p = 100;
        std::vector<std::size_t> sup = {3, 20, 45, 70, 90};
        std::vector<double> coefs(sup.size(), 2.0);
        datagen::SyntheticData data(n, p, sup, coefs, 3.0, 12345,
                                    datagen::predictor_policy::Normal(),
                                    datagen::noisegen::noise_policy::Normal());
        Eigen::MatrixXd X = data.getX();
        Eigen::VectorXd y = data.getY();
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());
        gvs::TRexGVSControlParameter ctrl;
        ctrl.gvs_type = gvs::GVSType::EN;
        ctrl.lambda_2 = 1.0;
        ctrl.trex_ctrl.K = 10;
        ctrl.trex_ctrl.max_dummy_multiplier = 3;
        ctrl.trex_ctrl.lloop_strategy = strategy;
        gvs::TRexGVSSelector sel(X_map, y_map, 0.1, ctrl, 42, false);
        return sel.select();
    };
    expectSameResult(run(LLoopStrategy::STANDARD), run(LLoopStrategy::SEEDED),
                     "GVS: SEEDED diverged from STANDARD");
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::lloop_seeded */
