// ========================================================================================
/**
 * @file test_utils_dummy_seeding.cpp
 *
 * @brief Unit tests for the 64-bit collision-free dummy seeding pipeline.
 *
 * @details Guards against the class of bugs where two dummy columns in one
 * design matrix are bit-identical because their per-column PRNG seeds
 * collided in a 32-bit space (observed in practice as spurious
 * "Variable N collinear; dropped" solver warnings under HCONCAT).
 *
 * Covers:
 *  - mix_seed64 injectivity in the index for a fixed base (block-level and
 *    column-level uniqueness by construction).
 *  - HCONCAT expandStored: no duplicate columns across expansion blocks;
 *    prefix stability; equivalence with a one-shot full generation
 *    (memory-mapped path contract).
 *  - PERMUTATION expandBaseDummies: same guarantees for the shared base.
 *  - Distinct dummies across the K experiments.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

// project includes
#include <utils/datageneration/utils_dummygen.hpp>
#include <trex_selector_methods/trex_utils/trex_dummy_generator.hpp>

// Eigen includes
#include <Eigen/Dense>

// ========================================================================================

// Embed into test namespace
namespace trex::test::utils::dummyseeding {

namespace dummygen = trex::utils::datageneration::dummygen;
using trex::trex_selector_methods::utils::dummy_generator::DummyGenerator;

namespace {

/** @brief Count pairs of bit-identical columns in a matrix (exact equality). */
int countDuplicateColumnPairs(const Eigen::MatrixXd& M) {
    int dups = 0;
    for (Eigen::Index a = 0; a < M.cols(); ++a) {
        for (Eigen::Index b = a + 1; b < M.cols(); ++b) {
            if (M.col(a) == M.col(b)) { ++dups; }
        }
    }
    return dups;
}

} // namespace

// ========================================================================================

/** @brief mix_seed64 must be injective in the index for a fixed base. */
TEST(DummySeeding, MixSeed64InjectiveInIndexForFixedBase) {
    // Exhaustive over a contiguous index range crossing bit boundaries plus
    // packed (l_tag << 32 | k) style indices — no output may repeat.
    const std::uint64_t base = 0x0123456789ABCDEFULL;
    std::unordered_set<std::uint64_t> seen;

    for (std::uint64_t j = 0; j < 200000; ++j) {
        ASSERT_TRUE(seen.insert(dummygen::mix_seed64(base, j)).second)
            << "collision at index " << j;
    }
    // Packed (k, L) block ids: k < 64, l_tag < 32
    for (std::uint64_t l = 0; l < 32; ++l) {
        for (std::uint64_t k = 0; k < 64; ++k) {
            ASSERT_TRUE(seen.insert(dummygen::mix_seed64(base, (l << 32) | k)).second
                        || l == 0)  // l==0 ids coincide with plain j above by design
                << "collision at packed (k=" << k << ", l=" << l << ")";
        }
    }
}


/** @brief HCONCAT: grown design must contain no bit-identical columns and be
 *         prefix-stable across expansions. This is the regression test for
 *         the 32-bit cross-block seed collisions that produced duplicate
 *         dummy columns (spurious collinear-drop warnings). */
TEST(DummySeeding, HconcatExpandStoredNoDuplicatesAndPrefixStable) {
    const std::size_t n = 30, p = 50, K = 4, L_max = 6;

    // Many independent generator instances (fresh random_device base each) to
    // exercise different base values; exactness of the guarantee does not
    // depend on the base, this just varies the inputs.
    for (int rep = 0; rep < 5; ++rep) {
        DummyGenerator gen(n);
        gen.generateAndStore(K, p, /*l_tag=*/0);

        std::vector<Eigen::MatrixXd> after_first_block;
        after_first_block.reserve(K);
        for (std::size_t k = 0; k < K; ++k) {
            after_first_block.push_back(gen.getStored(k));
        }

        for (std::size_t L = 2; L <= L_max; ++L) { gen.expandStored(p); }

        for (std::size_t k = 0; k < K; ++k) {
            const Eigen::MatrixXd& D = gen.getStored(k);
            ASSERT_EQ(D.cols(), static_cast<Eigen::Index>(L_max * p));

            // 1. No duplicate columns anywhere in the grown design
            EXPECT_EQ(countDuplicateColumnPairs(D), 0)
                << "duplicate dummy columns in experiment " << k;

            // 2. Prefix stability: first block unchanged by expansions
            EXPECT_TRUE(D.leftCols(p) == after_first_block[k])
                << "expansion mutated the first block of experiment " << k;
        }
    }
}


/** @brief HCONCAT/memory-map contract: a one-shot full generation with
 *         l_tag = 0 must reproduce the incrementally expanded design
 *         bit-exactly (T-loop mmap regeneration equivalence). */
TEST(DummySeeding, HconcatFullGenerationMatchesIncrementalBuild) {
    const std::size_t n = 25, p = 40, K = 3, L_max = 4;

    DummyGenerator gen(n, dummygen::Distribution::Normal(), /*seed=*/1234);

    gen.generateAndStore(K, p, /*l_tag=*/0);
    for (std::size_t L = 2; L <= L_max; ++L) { gen.expandStored(p); }

    for (std::size_t k = 0; k < K; ++k) {
        Eigen::MatrixXd full(n, L_max * p);
        gen.generateInto(full, k, /*l_tag=*/0);
        EXPECT_TRUE(full == gen.getStored(k))
            << "one-shot generation diverges from incremental build, k=" << k;
    }
}


/** @brief PERMUTATION: expanded base must contain no duplicate columns and
 *         keep the original base as an exact prefix. */
TEST(DummySeeding, PermutationExpandBaseNoDuplicatesAndPrefixStable) {
    const std::size_t n = 30, p = 50, L_max = 5;

    DummyGenerator gen(n, dummygen::Distribution::Normal(), /*seed=*/77);

    const std::size_t base_id = 424242u;
    Eigen::MatrixXd base = gen.generate(p, base_id);
    gen.storeBaseDummies(base, base_id);

    for (std::size_t L = 2; L <= L_max; ++L) { gen.expandBaseDummies(p); }

    const Eigen::MatrixXd grown = gen.getPermuted(0, L_max * p);
    EXPECT_EQ(countDuplicateColumnPairs(grown), 0)
        << "duplicate columns in expanded PERMUTATION base";
    EXPECT_TRUE(grown.leftCols(p) == base)
        << "expansion mutated the original PERMUTATION base";
}


/** @brief The K experiments must receive pairwise distinct dummy matrices
 *         (cross-experiment independence at the bit level). */
TEST(DummySeeding, ExperimentsReceiveDistinctDummies) {
    const std::size_t n = 20, p = 30, K = 8;

    DummyGenerator gen(n, dummygen::Distribution::Normal(), /*seed=*/5);
    gen.generateAndStore(K, p, /*l_tag=*/0);

    for (std::size_t a = 0; a < K; ++a) {
        for (std::size_t b = a + 1; b < K; ++b) {
            EXPECT_FALSE(gen.getStored(a) == gen.getStored(b))
                << "experiments " << a << " and " << b
                << " received identical dummy matrices";
        }
    }
}


/** @brief STANDARD vs HCONCAT tags: the same experiment under different
 *         l_tags must produce different blocks (fresh-per-iteration
 *         semantics), while the same tag reproduces identically. */
TEST(DummySeeding, LTagSeparatesStandardIterationsAndReproduces) {
    const std::size_t n = 20, p = 30;

    DummyGenerator gen(n, dummygen::Distribution::Normal(), /*seed=*/99);

    Eigen::MatrixXd d_tag1(n, p), d_tag2(n, p), d_tag1_again(n, p);
    gen.generateInto(d_tag1, /*experiment_k=*/0, /*l_tag=*/1);
    gen.generateInto(d_tag2, /*experiment_k=*/0, /*l_tag=*/2);
    gen.generateInto(d_tag1_again, /*experiment_k=*/0, /*l_tag=*/1);

    EXPECT_FALSE(d_tag1 == d_tag2) << "distinct l_tags produced identical dummies";
    EXPECT_TRUE(d_tag1 == d_tag1_again) << "same l_tag failed to reproduce";
}

// ========================================================================================
} /* End of namespace trex::test::utils::dummyseeding */
