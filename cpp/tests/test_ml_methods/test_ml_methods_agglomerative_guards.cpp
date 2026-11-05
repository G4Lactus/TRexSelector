// ===================================================================================
/**
 * @file test_ml_methods_agglomerative_guards.cpp
 *
 * @brief Input guards of the agglomerative clustering:
 *  non-finite data is rejected at the dispatcher before any engine runs, a
 *  zero-norm column keeps its documented "maximally dissimilar" convention,
 *  a RAM distance matrix beyond the physical memory is refused before its
 *  zero fill, and cut_tree / cut_tree_by_height refuse malformed merge
 *  matrices instead of indexing out of range.
 */
// ===================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// project includes
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/block_tiled_matrix_policy.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp"
#include "utils/physical_memory.hpp"

// ===================================================================================

namespace trex::test::ml_methods::clustering::hierarchical::agglomerative {

using namespace trex::ml_methods::clustering::hierarchical::agglomerative;

namespace {

/** @brief The five-point fixture of the dispatcher tests: x = {0, 2, 5, 10, 16}. */
Eigen::MatrixXd fivePoints() {
    Eigen::MatrixXd data(2, 5);
    data.row(0) << 0, 2, 5, 10, 16;
    data.row(1) << 0, 0, 0, 0, 0;
    return data;
}

template <DistanceMetric Metric, LinkageMethod Method>
std::vector<MergeStep> run(const Eigen::MatrixXd& data) {
    return AgglomerativeClustering::cluster<
        Eigen::MatrixXd, DistancePolicy<Eigen::MatrixXd, Metric>, Method>(
            data, /*use_mmap=*/false, /*verbose=*/false);
}

}  // namespace

// ===================================================================================

/** @brief NaN and Inf are rejected at the entry, for every engine family
 *  (SLINK, NN chain, generic linkage) and both exact metrics. */
TEST(AgglomerativeGuardsTest, NonFiniteInputIsRejectedAtEveryEngine) {
    for (const double bad : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
        Eigen::MatrixXd data = fivePoints();
        data(1, 2) = bad;
        EXPECT_THROW((run<DistanceMetric::Euclidean, LinkageMethod::Single>(data)),
                     std::invalid_argument);
        EXPECT_THROW((run<DistanceMetric::Correlation, LinkageMethod::Average>(data)),
                     std::invalid_argument);
        EXPECT_THROW((run<DistanceMetric::Euclidean, LinkageMethod::Ward>(data)),
                     std::invalid_argument);
        EXPECT_THROW((run<DistanceMetric::Euclidean, LinkageMethod::Centroid>(data)),
                     std::invalid_argument);
    }
    // The message names the entry.
    Eigen::MatrixXd data = fivePoints();
    data(1, 2) = std::numeric_limits<double>::quiet_NaN();
    try {
        (void)run<DistanceMetric::Euclidean, LinkageMethod::Single>(data);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("row 1, column 2"), std::string::npos) << e.what();
    }
}


/** @brief A zero-norm column is legitimate under the correlation distance:
 *  it is maximally dissimilar (distance 1) to everything and joins last. */
TEST(AgglomerativeGuardsTest, ZeroNormColumnIsMaximallyDissimilar) {
    Eigen::MatrixXd data(4, 4);
    data << 1.0, 1.0, -1.0, 0.0,
            2.0, 2.1, -2.0, 0.0,
            3.0, 2.9, -3.0, 0.0,
            4.0, 4.2, -4.0, 0.0;
    std::vector<MergeStep> merges;
    EXPECT_NO_THROW(merges = (run<DistanceMetric::Correlation, LinkageMethod::Average>(data)));
    ASSERT_EQ(merges.size(), 3u);
    EXPECT_DOUBLE_EQ(merges.back().distance, 1.0);
    const auto labels = DendrogramUtils::cut_tree(merges, 4, 2);
    // The three correlated columns form one cluster, the zero column the other.
    EXPECT_EQ(labels[0], labels[1]);
    EXPECT_EQ(labels[0], labels[2]);
    EXPECT_NE(labels[0], labels[3]);
}


/** @brief Every malformed shape of a merge matrix throws before indexing. */
TEST(AgglomerativeGuardsTest, CutTreeRejectsMalformedMerges) {
    const auto merges = run<DistanceMetric::Euclidean, LinkageMethod::Single>(fivePoints());
    ASSERT_EQ(merges.size(), 4u);
    // Sanity: the well-formed matrix cuts.
    EXPECT_NO_THROW(DendrogramUtils::cut_tree(merges, 5, 2));

    // Too few rows for the requested cut (K = 1 needs 4 merges, 2 given) —
    // while K = 3 (2 merges) still works on the same prefix.
    std::vector<MergeStep> two(merges.begin(), merges.begin() + 2);
    EXPECT_THROW(DendrogramUtils::cut_tree(two, 5, 1), std::invalid_argument);
    EXPECT_NO_THROW(DendrogramUtils::cut_tree(two, 5, 3));

    // A cluster id that does not exist yet at that step (5 = n at merge 0).
    auto bad = merges;
    bad[0].cluster2 = 5;
    EXPECT_THROW(DendrogramUtils::cut_tree(bad, 5, 2), std::invalid_argument);
    bad = merges;
    bad[0].cluster1 = -1;
    EXPECT_THROW(DendrogramUtils::cut_tree(bad, 5, 2), std::invalid_argument);

    // A self-merge.
    bad = merges;
    bad[0].cluster2 = bad[0].cluster1;
    EXPECT_THROW(DendrogramUtils::cut_tree(bad, 5, 2), std::invalid_argument);

    // An id consumed twice (object 0 was merged in step 0).
    bad = merges;
    bad[1].cluster1 = 0;
    EXPECT_THROW(DendrogramUtils::cut_tree(bad, 5, 2), std::invalid_argument);

    // A non-finite distance.
    bad = merges;
    bad[1].distance = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(DendrogramUtils::cut_tree(bad, 5, 2), std::invalid_argument);

    // Height cut: unsorted distances are refused instead of silently truncated.
    bad = merges;
    std::swap(bad[0].distance, bad[3].distance);
    EXPECT_THROW(DendrogramUtils::cut_tree_by_height(bad, 5, 10.0), std::invalid_argument);
    EXPECT_NO_THROW(DendrogramUtils::cut_tree_by_height(merges, 5, 10.0));
}

// ===================================================================================

/** @brief The RAM backend refuses a distance matrix beyond the physical memory
 *  before its zero fill and names the memory-mapped backend as the remedy. A
 *  fitting buffer and an unknown physical size (0) pass. */
TEST(AgglomerativeGuardsTest, RamBufferBeyondPhysicalMemoryIsRefused) {
    constexpr std::size_t GiB = std::size_t{1} << 30;
    EXPECT_NO_THROW(detail::checkRamBudget(8 * GiB, 16 * GiB));
    EXPECT_NO_THROW(detail::checkRamBudget(16 * GiB, 16 * GiB));
    EXPECT_NO_THROW(detail::checkRamBudget(64 * GiB, 0));
    try {
        detail::checkRamBudget(64 * GiB, 16 * GiB);
        FAIL() << "a buffer beyond the physical memory must be refused";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("64.00 GiB"), std::string::npos) << msg;
        EXPECT_NE(msg.find("16.00 GiB"), std::string::npos) << msg;
        EXPECT_NE(msg.find("use_mmap = true"), std::string::npos) << msg;
    }

    // End to end: 1,000,000 objects need a tiled matrix of about 7 TiB. The
    // check runs before any buffer is touched, so the constructor throws at once.
    const std::size_t physical = trex::utils::physical_memory::physicalMemoryBytes();
    if (physical == 0) GTEST_SKIP() << "physical memory unknown on this platform";
    EXPECT_GT(physical, 256 * (std::size_t{1} << 20));
    const Eigen::MatrixXd data = Eigen::MatrixXd::Zero(1, 1'000'000);
    using Policy = BlockTiledMatrixPolicy<
        Eigen::MatrixXd, DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>,
        LinkageMethod::Average>;
    EXPECT_THROW([&] { Policy policy(data, /*use_mmap=*/false, /*verbose=*/false); }(),
                 std::runtime_error);
}

// ===================================================================================
} /* End of namespace trex::test::ml_methods::clustering::hierarchical::agglomerative */
