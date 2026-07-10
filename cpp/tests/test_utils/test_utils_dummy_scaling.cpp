// ========================================================================================
/**
 * @file test_utils_dummy_scaling.cpp
 *
 * @brief Unit tests for dummy column scaling correctness (L2 and ZSCORE modes).
 *
 * @details The T-Rex FDR machinery requires every dummy column to live on the
 * SAME scale as the (identically normalized) predictor columns:
 *   - L2:     centered, unit L2 norm            (||d_c|| = 1)
 *   - ZSCORE: centered, unit sample SD          (||d_c|| = sqrt(n-1))
 *
 * A dummy column on the wrong scale competes unfairly in the max-correlation
 * selection and silently biases the FDP estimate. These tests pin the scaling
 * invariant on every generation path (initial blocks, HCONCAT expansions,
 * PERMUTATION base + expansions) and verify that row permutations preserve
 * both the scaling and the exact column content.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <algorithm>
#include <cmath>
#include <vector>

// project includes
#include <utils/datageneration/utils_dummygen.hpp>
#include <trex_selector_methods/trex_utils/trex_data_normalizer.hpp>
#include <trex_selector_methods/trex_utils/trex_dummy_generator.hpp>

// Eigen includes
#include <Eigen/Dense>

// ========================================================================================

// Embed into test namespace
namespace trex::test::utils::dummyscaling {

namespace dummygen = trex::utils::datageneration::dummygen;
namespace dn = trex::trex_selector_methods::utils::data_normalizer;
using trex::trex_selector_methods::utils::dummy_generator::DummyGenerator;

namespace {

/** @brief Expected column norm for a scaling mode on n rows. */
double expectedNorm(dn::ScalingMode mode, Eigen::Index n) {
    return (mode == dn::ScalingMode::ZSCORE)
        ? std::sqrt(static_cast<double>(n - 1))
        : 1.0;
}

/** @brief Assert every column of M is centered and on the mode's scale. */
void expectCenteredAndScaled(const Eigen::MatrixXd& M,
                             dn::ScalingMode mode,
                             const std::string& label,
                             double tol = 1e-10) {
    const double target = expectedNorm(mode, M.rows());
    for (Eigen::Index j = 0; j < M.cols(); ++j) {
        EXPECT_NEAR(M.col(j).mean(), 0.0, tol)
            << label << ": column " << j << " not centered";
        EXPECT_NEAR(M.col(j).norm(), target, tol * target)
            << label << ": column " << j << " off-scale";
    }
}

/** @brief True iff column a is an exact element permutation of column b. */
bool isElementPermutation(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    std::vector<double> sa(a.data(), a.data() + a.size());
    std::vector<double> sb(b.data(), b.data() + b.size());
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    return sa == sb;  // bitwise: permutation only reorders, never recomputes
}

struct ModeCase {
    dn::ScalingMode mode;
    const char* name;
};

const ModeCase kModes[] = {
    {dn::ScalingMode::L2,     "L2"},
    {dn::ScalingMode::ZSCORE, "ZSCORE"},
};

} // namespace

// ========================================================================================

/** @brief Initial generation: every column centered + on-scale, for both
 *         modes and several distributions. */
TEST(DummyScaling, GenerateAndStoreColumnsCenteredAndScaled) {
    const std::size_t n = 40, p = 25, K = 3;

    const dummygen::Distribution dists[] = {
        dummygen::Distribution::Normal(),
        dummygen::Distribution::Uniform(),
        dummygen::Distribution::Rademacher(),
    };

    for (const auto& mc : kModes) {
        for (const auto& dist : dists) {
            DummyGenerator gen(n, dist, /*seed=*/321, /*verbose=*/false, mc.mode);
            gen.generateAndStore(K, p, /*l_tag=*/0);
            for (std::size_t k = 0; k < K; ++k) {
                expectCenteredAndScaled(gen.getStored(k), mc.mode,
                                        std::string(mc.name) + "/initial k="
                                            + std::to_string(k));
            }
        }
    }
}


/** @brief HCONCAT expansions: appended blocks must satisfy the same scaling
 *         invariant as the initial block (expandInto normalizes only the new
 *         block — this test catches any miss). */
TEST(DummyScaling, ExpandStoredKeepsAllBlocksOnScale) {
    const std::size_t n = 35, p = 20, K = 2, L_max = 5;

    for (const auto& mc : kModes) {
        DummyGenerator gen(n, dummygen::Distribution::Normal(),
                           /*seed=*/11, /*verbose=*/false, mc.mode);
        gen.generateAndStore(K, p, /*l_tag=*/0);
        for (std::size_t L = 2; L <= L_max; ++L) { gen.expandStored(p); }

        for (std::size_t k = 0; k < K; ++k) {
            ASSERT_EQ(gen.getStored(k).cols(),
                      static_cast<Eigen::Index>(L_max * p));
            expectCenteredAndScaled(gen.getStored(k), mc.mode,
                                    std::string(mc.name) + "/expanded k="
                                        + std::to_string(k));
        }
    }
}


/** @brief PERMUTATION: row-permuted variants must preserve centering, the
 *         mode's scale, and the exact multiset of column entries (a
 *         permutation reorders — it must never rescale or recompute). */
TEST(DummyScaling, PermutationPreservesScaleAndContentExactly) {
    const std::size_t n = 30, p = 15, K = 5;

    for (const auto& mc : kModes) {
        DummyGenerator gen(n, dummygen::Distribution::Normal(),
                           /*seed=*/7, /*verbose=*/false, mc.mode);

        const std::size_t base_id = 99u;
        Eigen::MatrixXd base = gen.generate(p, base_id);
        gen.storeBaseDummies(base, base_id);

        for (std::size_t k = 0; k < K; ++k) {
            const Eigen::MatrixXd Dk = gen.getPermuted(k, p);

            // Scaling preserved (fp-exact up to summation order)
            expectCenteredAndScaled(Dk, mc.mode,
                                    std::string(mc.name) + "/permuted k="
                                        + std::to_string(k), 1e-12);

            // Content preserved: each column is an exact element permutation
            for (Eigen::Index j = 0; j < Dk.cols(); ++j) {
                EXPECT_TRUE(isElementPermutation(Dk.col(j), base.col(j)))
                    << mc.name << ": permuted column " << j
                    << " is not an element permutation of the base (k="
                    << k << ")";
            }
        }

        // k = 0 must be the base verbatim
        EXPECT_TRUE(gen.getPermuted(0, p) == base);
    }
}


/** @brief PERMUTATION base expansion: the grown base must be fully on-scale
 *         in both modes, and permuted variants of the grown base likewise. */
TEST(DummyScaling, ExpandBaseDummiesGrownBaseOnScale) {
    const std::size_t n = 30, p = 15, L_max = 4;

    for (const auto& mc : kModes) {
        DummyGenerator gen(n, dummygen::Distribution::Normal(),
                           /*seed=*/13, /*verbose=*/false, mc.mode);

        const std::size_t base_id = 5u;
        gen.storeBaseDummies(gen.generate(p, base_id), base_id);
        for (std::size_t L = 2; L <= L_max; ++L) { gen.expandBaseDummies(p); }

        const Eigen::MatrixXd grown = gen.getPermuted(0, L_max * p);
        expectCenteredAndScaled(grown, mc.mode,
                                std::string(mc.name) + "/grown base");

        const Eigen::MatrixXd permuted = gen.getPermuted(2, L_max * p);
        expectCenteredAndScaled(permuted, mc.mode,
                                std::string(mc.name) + "/permuted grown base",
                                1e-12);
    }
}


/** @brief mmap parity: permuteInto must produce the identical matrix as
 *         getPermuted for every experiment (scaling rides along). */
TEST(DummyScaling, PermuteIntoMatchesGetPermuted) {
    const std::size_t n = 25, p = 12, K = 4;

    for (const auto& mc : kModes) {
        DummyGenerator gen(n, dummygen::Distribution::Normal(),
                           /*seed=*/3, /*verbose=*/false, mc.mode);
        const std::size_t base_id = 17u;
        gen.storeBaseDummies(gen.generate(p, base_id), base_id);

        for (std::size_t k = 0; k < K; ++k) {
            Eigen::MatrixXd target(n, p);
            gen.permuteInto(target, k, p);
            EXPECT_TRUE(target == gen.getPermuted(k, p))
                << mc.name << ": permuteInto != getPermuted at k=" << k;
        }
    }
}


/** @brief X and dummies must share one convention: the dummy normalizer and
 *         the X normalizer must place columns on the identical scale for the
 *         same mode (guards against convention drift, e.g. sqrt(n) vs
 *         sqrt(n-1) in ZSCORE). */
TEST(DummyScaling, XAndDummyNormalizersAgreeOnScale) {
    const Eigen::Index n = 50, p = 8;

    for (const auto& mc : kModes) {
        // Same raw matrix through both normalizers
        Eigen::MatrixXd raw = Eigen::MatrixXd::Random(n, p) * 3.7;
        raw.array() += 1.5;  // nonzero means so centering matters

        Eigen::MatrixXd as_X = raw;
        dn::NormalizationParams params;
        dn::centerAndL2NormalizeX(as_X, params,
                                  std::numeric_limits<double>::epsilon(),
                                  false, mc.mode);

        Eigen::MatrixXd as_D = raw;
        dn::centerAndL2NormalizeMatrix(as_D,
                                       std::numeric_limits<double>::epsilon(),
                                       false, mc.mode);

        EXPECT_TRUE(as_X.isApprox(as_D, 1e-14))
            << mc.name << ": X-normalizer and dummy-normalizer diverge";
    }
}

// ========================================================================================
} /* End of namespace trex::test::utils::dummyscaling */
