// ====================================================================================
/**
 * @file test_ml_methods_agglomerative_cores.cpp
 *
 * @brief Unit tests for agglomerative clustering cores.
 */
// ====================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <cmath>
#include <limits>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// project ml_methods includes
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/nnchain_core.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/slink_core.hpp"

// ===================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::clustering::hierarchical::agglomerative {

using namespace trex::ml_methods::clustering::hierarchical::agglomerative;

// ===================================================================================
// MOCK DISTANCE POLICY FOR NNCHAIN
// ===================================================================================

/**
 * Define a mock distance policy that overrides distance computations allowing exact
 * logic checks isolated from Lance-Williams block matrices.
 */
struct MockDistancePolicy {
    Eigen::MatrixXd dist_matrix;

    // In this mock, we initialize with a precomputed symmetric distance matrix.
    MockDistancePolicy(const Eigen::MatrixXd& data) : dist_matrix(data) {}

    // find_nearest_neighbor logic explicitly returns the minimum available edge.
    Eigen::Index find_nearest_neighbor(
        Eigen::Index c,
        const std::vector<bool>& active,
        Eigen::Index /*current_cluster_id*/,
        double& min_dist
    ) {
        Eigen::Index best_idx = -1;
        min_dist = std::numeric_limits<double>::max();

        for (Eigen::Index i = 0; i < dist_matrix.cols(); ++i) {
            if (i != c && active[i]) {
                if (dist_matrix(c, i) < min_dist) {
                    min_dist = dist_matrix(c, i);
                    best_idx = i;
                }
            }
        }
        return best_idx;
    }

    // merge_clusters applies Single Linkage explicitly for the mock test.
    void merge_clusters(MergeClustersArgs args, const std::vector<Eigen::Index>& /*sizes*/) {
        // Expand the distance matrix to make room for combination (just appending a row/col).
        Eigen::Index new_size = dist_matrix.rows() + 1;
        Eigen::MatrixXd new_dist = Eigen::MatrixXd::Constant(
            new_size, new_size, std::numeric_limits<double>::max());
        new_dist.topLeftCorner(dist_matrix.rows(), dist_matrix.cols()) = dist_matrix;

        // Populate the new cluster's distances using single linkage
        for (Eigen::Index i = 0; i < dist_matrix.rows(); ++i) {
            if (i != args.c_id && i != args.d_id) {
                double d = std::min(dist_matrix(args.c_id, i),
                                    dist_matrix(args.d_id, i));
                new_dist(args.new_id, i) = d;
                new_dist(i, args.new_id) = d;
            }
        }
        new_dist(args.new_id, args.new_id) = 0.0;
        dist_matrix = new_dist;
    }
};


// ===================================================================================
// TESTS
// ===================================================================================

/** @brief Test UnionFind behavior. */
TEST(DendrogramUtilsTest, UnionFindBehavior) {
    UnionFind uf(5);

    // Initial state
    EXPECT_EQ(uf.find(0), 0);
    EXPECT_EQ(uf.get_cluster_size(0), 1);

    // Unite 0 and 1 -> new id 5
    uf.unite({0, 1, 5});
    EXPECT_EQ(uf.find(0), uf.find(1));
    EXPECT_EQ(uf.get_cluster_id(0), 5);
    EXPECT_EQ(uf.get_cluster_size(0), 2);

    // Unite 2 and 3 -> new id 6
    uf.unite({2, 3, 6});

    // Unite (0,1) and (2,3) -> new id 7
    uf.unite({0, 2, 7});

    EXPECT_EQ(uf.find(1), uf.find(3));
    EXPECT_EQ(uf.get_cluster_size(1), 4);
    EXPECT_EQ(uf.get_cluster_id(1), 7);
}


/** @brief Test conversion from pointer representation to merge matrix. */
TEST(DendrogramUtilsTest, PointerToMergeMatrix) {
    // Simulated SLINK result for 4 elements, in Sibson's convention as produced by
    // SLINK::cluster(): pi[i] points FORWARD (pi[i] > i), the root is the last
    // element with pi[i] == i and lambda[i] == infinity.
    std::vector<Eigen::Index> pi = {1, 2, 3, 3};
    std::vector<double> lambda = {1.0, 2.0, 3.0,
        std::numeric_limits<double>::max()};

    auto merges = DendrogramUtils::pointer_to_merge_matrix(pi, lambda);

    // Check sizes
    ASSERT_EQ(merges.size(), 3);

    // Expected merges (ordered by lambda):
    // 1st: {0, 1} at dist=1.0 size=2  (Cluster 4)
    // 2nd: {2, 4} at dist=2.0 size=3  (Cluster 5)
    // 3rd: {3, 5} at dist=3.0 size=4

    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_EQ(merges[0].distance, 1.0);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 4); // 4 was the newly assigned id
    EXPECT_EQ(merges[1].distance, 2.0);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 5);
    EXPECT_EQ(merges[2].distance, 3.0);
    EXPECT_EQ(merges[2].new_cluster_size, 4);
}


/** @brief Test formatting of NNChain merges. */
TEST(DendrogramUtilsTest, FormatNNChainMerges) {
    // NNChain merges aren't necessarily sorted by distance, format_nnchain_merges fixes this.
    std::vector<MergeStep> raw_merges = {
        {2, 3, 5.0, 2}, // Creates cluster 4
        {0, 1, 2.0, 2}, // Creates cluster 5
        {4, 5, 8.0, 4}  // Creates cluster 6
    };

    auto sorted = DendrogramUtils::format_nnchain_merges(
        raw_merges, 4);

    ASSERT_EQ(sorted.size(), 3);

    // First should be dist 2.0 {0, 1} -> 4
    EXPECT_EQ(sorted[0].distance, 2.0);
    EXPECT_EQ(sorted[0].cluster1, 0);
    EXPECT_EQ(sorted[0].cluster2, 1);

    // Second should be dist 5.0 {2, 3} -> 5
    EXPECT_EQ(sorted[1].distance, 5.0);
    EXPECT_EQ(sorted[1].cluster1, 2);
    EXPECT_EQ(sorted[1].cluster2, 3);

    // Third should be dist 8.0 {4, 5}
    EXPECT_EQ(sorted[2].distance, 8.0);
}


/**
 * @brief Regression test: cluster sizes must be recomputed from the Union-Find state,
 * not copied from the chronological record, when the input merges are non-monotone.
 *
 * @details GenericLinkage (Centroid/Median) can emit inversions: a later merge with a
 * SMALLER distance than an earlier one. After the global re-sort, an edge may join
 * different components than at discovery time, so the chronologically recorded
 * new_cluster_size is stale.
 */
TEST(DendrogramUtilsTest, FormatNNChainMergesInversionSizes) {
    // Chronological (inverted) sequence on 3 objects:
    //  - {0, 1} merged first at distance 5.0        -> creates cluster 3 (size 2)
    //  - {2, cluster 3} merged at distance 3.0 (!)  -> inversion (size 3)
    std::vector<MergeStep> raw_merges = {
        {0, 1, 5.0, 2},
        {2, 3, 3.0, 3}
    };

    auto sorted = DendrogramUtils::format_nnchain_merges(raw_merges, 3);
    ASSERT_EQ(sorted.size(), 2);

    // Monotone rebuild: the 3.0 edge is processed first and now joins two SINGLETONS
    // (leaf representatives 2 and 0), so its size is 2 — not the recorded 3.
    EXPECT_EQ(sorted[0].cluster1, 0);
    EXPECT_EQ(sorted[0].cluster2, 2);
    EXPECT_DOUBLE_EQ(sorted[0].distance, 3.0);
    EXPECT_EQ(sorted[0].new_cluster_size, 2);

    // The 5.0 edge then joins {0,2} (id 3) with singleton 1: size 3 — not the recorded 2.
    EXPECT_EQ(sorted[1].cluster1, 1);
    EXPECT_EQ(sorted[1].cluster2, 3);
    EXPECT_DOUBLE_EQ(sorted[1].distance, 5.0);
    EXPECT_EQ(sorted[1].new_cluster_size, 3);
}


/** @brief Test cutting a dendrogram into clusters. */
TEST(DendrogramUtilsTest, CutTree) {
    // Using the sorted merge from previous
    std::vector<MergeStep> merges = {
        {0, 1, 2.0, 2}, // creates 4
        {2, 3, 5.0, 2}, // creates 5
        {4, 5, 8.0, 4}  // creates 6
    };

    auto labels = DendrogramUtils::cut_tree(
        merges, 4, 2);
    EXPECT_EQ(labels.size(), 4);

    // 0 and 1 should have the same label, 2 and 3 should have the same label
    EXPECT_EQ(labels[0], labels[1]);
    EXPECT_EQ(labels[2], labels[3]);
    EXPECT_NE(labels[0], labels[2]); // Two distinct clusters
}


/** @brief Test height-based dendrogram cutting and label grouping. */
TEST(DendrogramUtilsTest, CutTreeByHeightAndGrouping) {
    std::vector<MergeStep> merges = {
        {0, 1, 2.0, 2}, // creates 4
        {2, 3, 5.0, 2}, // creates 5
        {4, 5, 8.0, 4}  // creates 6
    };

    // Height below every merge: all singletons (K == N early-return path)
    auto labels_all = DendrogramUtils::cut_tree_by_height(merges, 4, 1.0);
    EXPECT_EQ(labels_all, (std::vector<Eigen::Index>{0, 1, 2, 3}));

    // Height 5.0 (threshold is inclusive): first two merges applied -> {0,1}, {2,3}
    auto labels_k2 = DendrogramUtils::cut_tree_by_height(merges, 4, 5.0);
    EXPECT_EQ(labels_k2[0], labels_k2[1]);
    EXPECT_EQ(labels_k2[2], labels_k2[3]);
    EXPECT_NE(labels_k2[0], labels_k2[2]);

    // Height above the root: one single cluster (K == 1)
    auto labels_k1 = DendrogramUtils::cut_tree_by_height(merges, 4, 10.0);
    EXPECT_EQ(labels_k1, (std::vector<Eigen::Index>{0, 0, 0, 0}));

    // Grouping buckets the original column indices by label
    auto groups = DendrogramUtils::group_indices_by_label(labels_k2);
    ASSERT_EQ(groups.size(), 2);
    EXPECT_EQ(groups[0], (std::vector<Eigen::Index>{0, 1}));
    EXPECT_EQ(groups[1], (std::vector<Eigen::Index>{2, 3}));
}


/** @brief Test execution of the NNChain core engine. */
TEST(NNChainTest, CoreEngineExecution) {
    // distance matrix for 4 points
    Eigen::MatrixXd dists(4, 4);
    dists << 0, 1, 4, 5,
             1, 0, 6, 7,
             4, 6, 0, 2,
             5, 7, 2, 0;

    NNChain<Eigen::MatrixXd, MockDistancePolicy> model(dists);
    // Needs to trigger exact structure
    EXPECT_NO_THROW(model.cluster());

    const auto& merges = model.get_merges();
    ASSERT_EQ(merges.size(), 3);

    // Since our Mock Policy acts greedily with single linkage, it must:
    // merge {0,1} at dist 1 -> cluster 4
    // merge {2,3} at dist 2 -> cluster 5
    // merge {4,5} at dist min(4,6,5,7) = 4 -> cluster 6
    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_DOUBLE_EQ(merges[0].distance, 1.0);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 3);
    EXPECT_DOUBLE_EQ(merges[1].distance, 2.0);
    EXPECT_EQ(merges[1].new_cluster_size, 2);

    EXPECT_EQ(merges[2].cluster1, 4);
    EXPECT_EQ(merges[2].cluster2, 5);
    EXPECT_DOUBLE_EQ(merges[2].distance, 4.0);
    EXPECT_EQ(merges[2].new_cluster_size, 4);

    // Active states should show only root (id 6) is active
    const auto& active = model.get_active_clusters();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FALSE(active[i]);
    }
    EXPECT_TRUE(active[6]);
}


// ===================================================================================
// SLINK CORE TESTS
// ===================================================================================

/** @brief Convenience alias: exact squared-Euclidean distance policy. */
using EuclideanPolicy = DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>;

/** @brief SLINK pointer representation for a 1D chain with distinct distances. */
TEST(SlinkTest, PointerRepresentationChain) {
    // 5 points on a line; squared adjacent gaps: 4, 9, 25, 36 (all distinct)
    Eigen::MatrixXd data(1, 5);
    data << 0, 2, 5, 10, 16;

    SLINK<Eigen::MatrixXd, EuclideanPolicy> slink(data);
    slink.cluster();

    // Sibson's pointer representation: pi[i] > i, the root (last element) points
    // to itself with lambda = infinity.
    const std::vector<Eigen::Index> expected_pi = {1, 2, 3, 4, 4};
    EXPECT_EQ(slink.get_pi(), expected_pi);

    const auto lambda = slink.get_lambda();
    ASSERT_EQ(lambda.size(), 5);
    EXPECT_DOUBLE_EQ(lambda[0], 4.0);
    EXPECT_DOUBLE_EQ(lambda[1], 9.0);
    EXPECT_DOUBLE_EQ(lambda[2], 25.0);
    EXPECT_DOUBLE_EQ(lambda[3], 36.0);
    EXPECT_EQ(lambda[4], std::numeric_limits<double>::max());
}


/**
 * @brief SLINK with fully tied distances must produce a deterministic result via the
 * tie-aware lexicographic ordering (Müllner Section 5 fix) and the stable Kruskal sort.
 */
TEST(SlinkTest, TieAwareDeterministicMerges) {
    // 4 equally spaced points: all three adjacent squared gaps tie at 1.0
    Eigen::MatrixXd data(1, 4);
    data << 0, 1, 2, 3;

    SLINK<Eigen::MatrixXd, EuclideanPolicy> slink(data);
    slink.cluster();

    // Lexicographic tie-breaking resolves the chain strictly left-to-right.
    const std::vector<Eigen::Index> expected_pi = {1, 2, 3, 3};
    EXPECT_EQ(slink.get_pi(), expected_pi);

    const auto lambda = slink.get_lambda();
    ASSERT_EQ(lambda.size(), 4);
    EXPECT_DOUBLE_EQ(lambda[0], 1.0);
    EXPECT_DOUBLE_EQ(lambda[1], 1.0);
    EXPECT_DOUBLE_EQ(lambda[2], 1.0);
    EXPECT_EQ(lambda[3], std::numeric_limits<double>::max());

    // End-to-end conversion: all merges at height 1.0, deterministic order
    // {0,1} -> 4, {2,4} -> 5, {3,5} -> 6.
    auto merges = DendrogramUtils::pointer_to_merge_matrix(slink.get_pi(),
                                                           slink.get_lambda());
    ASSERT_EQ(merges.size(), 3);

    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_DOUBLE_EQ(merges[0].distance, 1.0);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 4);
    EXPECT_DOUBLE_EQ(merges[1].distance, 1.0);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 5);
    EXPECT_DOUBLE_EQ(merges[2].distance, 1.0);
    EXPECT_EQ(merges[2].new_cluster_size, 4);
}


// ===================================================================================
// DISTANCE POLICY TESTS
// ===================================================================================

namespace {

/** @brief Fills two orthogonal +-1 sign patterns of length 64 (dot product zero). */
void make_orthogonal_patterns(Eigen::VectorXd& u, Eigen::VectorXd& w) {
    const Eigen::Index n = 64;
    u.resize(n);
    w.resize(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        u(i) = (i % 2 == 0) ? 1.0 : -1.0; // + - + - ...
        w(i) = (i % 4 < 2) ? 1.0 : -1.0;  // + + - - ...
    }
}

} // namespace


/** @brief Exact metrics: Manhattan distance and Correlation edge cases. */
TEST(DistancePolicyTest, ManhattanAndCorrelationEdgeCases) {
    Eigen::MatrixXd data(2, 3);
    data.col(0) << 0.0, 0.0;
    data.col(1) << 1.0, -2.0;
    data.col(2) << 1.0, -1.0;

    DistancePolicy<Eigen::MatrixXd, DistanceMetric::Manhattan> manhattan(data);
    EXPECT_DOUBLE_EQ(manhattan.get_distance(0, 1), 3.0); // |1| + |-2|
    EXPECT_DOUBLE_EQ(manhattan.get_distance(1, 2), 1.0); // |0| + |-1|
    EXPECT_DOUBLE_EQ(manhattan.get_distance(1, 1), 0.0);

    DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation> corr(data);

    // A zero-norm (constant) column is treated as maximally dissimilar.
    EXPECT_DOUBLE_EQ(corr.get_distance(0, 1), 1.0);

    // 1 - |corr|: corr(col1, col2) = (1 + 2) / (sqrt(5) * sqrt(2)) = 3 / sqrt(10)
    EXPECT_NEAR(corr.get_distance(1, 2), 1.0 - 3.0 / std::sqrt(10.0), 1e-12);
}


/** @brief SimHash approximation: exact behavior for parallel / anti-parallel columns,
 *  large distance for orthogonal columns. */
TEST(DistancePolicyTest, LSHApproxAngularDistances) {
    Eigen::VectorXd u, w;
    make_orthogonal_patterns(u, w);

    Eigen::MatrixXd data(64, 4);
    data.col(0) = u;
    data.col(1) = u;  // duplicate      -> identical signature, Hamming 0
    data.col(2) = -u; // anti-parallel  -> all 64 bits flip, |cos(pi)| = 1
    data.col(3) = w;  // orthogonal     -> Hamming ~ 32, distance ~ 1

    DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation_LSH_Approx> lsh(data);

    EXPECT_DOUBLE_EQ(lsh.get_distance(0, 0), 0.0); // i == j short-circuit
    EXPECT_NEAR(lsh.get_distance(0, 1), 0.0, 1e-12);
    // 1 - |cos(pi)| = 0: anti-correlated variables cluster together by design
    EXPECT_NEAR(lsh.get_distance(0, 2), 0.0, 1e-12);
    EXPECT_GT(lsh.get_distance(0, 3), 0.5);
}


/** @brief LSH gatekeeper: correlated pairs pass the Hamming gate and get the exact
 *  distance; pairs beyond the gate are clamped to exactly 1.0. */
TEST(DistancePolicyTest, LSHFilterGatekeeper) {
    Eigen::VectorXd u, w;
    make_orthogonal_patterns(u, w);

    const double pi = 3.14159265358979323846;
    const double theta = 5.0 * pi / 180.0;

    Eigen::MatrixXd data(64, 3);
    data.col(0) = u;
    // corr(col0, col1) = cos(5 deg); the scale factor must cancel in the exact recheck
    data.col(1) = 1.5 * (std::cos(theta) * u + std::sin(theta) * w);
    data.col(2) = w;

    DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation_LSH_Filter> lsh(data);

    // Hamming ~ 64 * (5/180) < 14: gate passes -> exact 1 - |corr|
    EXPECT_NEAR(lsh.get_distance(0, 1), 1.0 - std::cos(theta), 1e-9);

    // Hamming ~ 32 > 14: gate rejects -> exactly 1.0
    EXPECT_DOUBLE_EQ(lsh.get_distance(0, 2), 1.0);
}

// ========================================================================================

} /* End of namespace trex::test::ml_methods::clustering::hierarchical::agglomerative */
