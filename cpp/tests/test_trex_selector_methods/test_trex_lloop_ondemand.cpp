// ========================================================================================
/**
 * @file test_trex_lloop_ondemand.cpp
 *
 * @brief Tests for the ONDEMAND L-loop strategies (2026-07 rename/rework).
 *
 * @details PERMUTATION_ONDEMAND is the stateless twin of PERMUTATION: the base
 * dummy matrix is re-derived from the seed instead of stored, and the same
 * per-experiment row-permutation engines are used (keyed on the identical
 * base id). For the same user seed the two strategies must therefore produce
 * BIT-IDENTICAL experiments and selections. ONDEMAND (former DIRECT) provides
 * independent per-experiment dummies without storage.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <vector>

// project includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_utils/trex_dummy_generator.hpp>
#include <utils/datageneration/utils_datagen.hpp>

// Eigen includes
#include <Eigen/Dense>

// ========================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::lloop_ondemand {

namespace datagen = trex::utils::datageneration::datagen;
namespace dummygen = trex::utils::datageneration::dummygen;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::utils::dummy_generator::DummyGenerator;

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

/** @brief PERMUTATION_ONDEMAND must reproduce PERMUTATION bit-exactly for
 *         the same seed (same base id, same permutation engines). */
TEST(LLoopOnDemand, PermutationOnDemandMatchesStoredPermutation) {
    for (int seed : {7, 42, 2026}) {
        const auto stored   = runSelection(LLoopStrategy::PERMUTATION, seed);
        const auto ondemand = runSelection(LLoopStrategy::PERMUTATION_ONDEMAND, seed);
        EXPECT_EQ(stored, ondemand)
            << "PERMUTATION vs PERMUTATION_ONDEMAND diverged at seed " << seed;
    }
}


/** @brief PERMUTATION_ONDEMAND must actually PERMUTE — its per-experiment
 *         dummies must differ from ONDEMAND's independent draws (this is the
 *         regression test for the old bug where PERMUTATION_DIRECT silently
 *         aliased DIRECT). Checked at the DummyGenerator level. */
TEST(LLoopOnDemand, OnDemandPermutationsAreRowPermutationsNotFreshDraws) {
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
 *         selection twice for both ONDEMAND strategies (nothing stored means
 *         everything must be re-derivable). */
TEST(LLoopOnDemand, OnDemandStrategiesAreSeedReproducible) {
    for (auto strategy : {LLoopStrategy::ONDEMAND,
                          LLoopStrategy::PERMUTATION_ONDEMAND}) {
        const auto first  = runSelection(strategy, 11);
        const auto second = runSelection(strategy, 11);
        EXPECT_EQ(first, second) << "strategy not seed-reproducible";
    }
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::lloop_ondemand */
