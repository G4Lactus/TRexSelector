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
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// project ml_methods includes
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/nnchain_core.hpp"

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
    // Simulated SLINK result for 4 elements
    std::vector<Eigen::Index> pi = {0, 0, 1, 2};
    std::vector<double> lambda = {std::numeric_limits<double>::max(),
        1.0, 2.0, 3.0};

    auto merges = DendrogramUtils::pointer_to_merge_matrix(pi, lambda);

    // Check sizes
    ASSERT_EQ(merges.size(), 3);

    // Expected merges (ordered by lambda):
    // 1st: {0, 1} at dist=1.0 size=2  (Cluster 4)
    // 2nd: {4, 2} at dist=2.0 size=3  (Cluster 5)
    // 3rd: {5, 3} at dist=3.0 size=4

    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_EQ(merges[0].distance, 1.0);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 4); // 4 was the newly assigned id
    EXPECT_EQ(merges[1].distance, 2.0);
    EXPECT_EQ(merges[1].new_cluster_size, 3);
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
    EXPECT_EQ(merges.size(), 3);

    // Since our Mock Policy acts greedily with single linkage, it should:
    // merge 0,1 (dist 1)
    // merge 2,3 (dist 2)
    // merge the rest

    // The sizes verify correct counting over multiple RNN iterations.
    EXPECT_EQ(merges.back().new_cluster_size, 4);

    // Active states should show only root (id 6) is active
    const auto& active = model.get_active_clusters();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FALSE(active[i]);
    }
    EXPECT_TRUE(active[6]);
}

// ========================================================================================

} /* End of namespace trex::test::ml_methods::clustering::hierarchical::agglomerative */
