// ===================================================================================
/**
 * @file test_ml_methods_agglomerative_dispatcher.cpp
 *
 * @brief Unit tests for the Agglomerative Clustering Dispatcher.
 */
// ===================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <vector>

// project ml_methods includes
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp"

// ===================================================================================

// Namespace import for convenience
using namespace trex::ml_methods::clustering::hierarchical::agglomerative;

// ===================================================================================
// End-To-End Integration Tests validating Dispatcher Routing and SciPy Geometry
// ===================================================================================
// Note on SciPy Equivalence:
// 1. Distance Metrics: The C++ engine computes squared distances natively for performance.
//    `DistanceMetric::Euclidean` maps directly to SciPy's `metric='sqeuclidean'`.
//
// 2. Inversions: For non-reducible linkage methods (Centroid, Median), SciPy preserves
//    chronological inversions (later merges with smaller distances). This framework enforces
//    distance monotonicity by globally sorting merges (via `format_nnchain_merges`).
//    Thus, the tree geometry matches, but the chronological row order might differ if
//    inversions occur.
//
// 3. Ward's Method: SciPy's Ward implementation internally takes the square root of the
//    variance delta. Our framework propagates the squared ESS natively.
//
// 4. Tie-breaking: Minor differences may occur depending on internal sort stability and
//    array traversal loop orientation.
// ===================================================================================

/** @brief Test case for Single Linkage Agglomerative Clustering matching SciPy results */
TEST(AgglomerativeDispatcherTest, SingleLinkageSciPyMatch) {

    // 5 points in 1D space, mapped to 2D
    Eigen::MatrixXd data(2, 5);
    // P0=0, P1=2, P2=5, P3=10, P4=16
    data.row(0) << 0, 2, 5, 10, 16;
    data.row(1) << 0, 0, 0, 0, 0;

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>,
        LinkageMethod::Single>(data, false);

    ASSERT_EQ(merges.size(), 4);

    // Sequence (Squared Euclidean Distance):
    // 1: {0, 1}, dist=4, size=2   -> ID 5
    // 2: {2, 5}, dist=9, size=3   -> ID 6
    // 3: {3, 6}, dist=25, size=4  -> ID 7
    // 4: {4, 7}, dist=36, size=5  -> ID 8

    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 4.0, 1e-6);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 5);
    EXPECT_NEAR(merges[1].distance, 9.0, 1e-6);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 6);
    EXPECT_NEAR(merges[2].distance, 25.0, 1e-6);
    EXPECT_EQ(merges[2].new_cluster_size, 4);

    EXPECT_EQ(merges[3].cluster1, 4);
    EXPECT_EQ(merges[3].cluster2, 7);
    EXPECT_NEAR(merges[3].distance, 36.0, 1e-6);
    EXPECT_EQ(merges[3].new_cluster_size, 5);
}


/** @brief Test case for Average Linkage Agglomerative Clustering matching SciPy results */
TEST(AgglomerativeDispatcherTest, AverageLinkageSciPyMatch) {

    Eigen::MatrixXd data(2, 5);
    data.row(0) << 0, 2, 5, 10, 16;
    data.row(1) << 0, 0, 0, 0, 0;

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>,
        LinkageMethod::Average>(data, false);

    ASSERT_EQ(merges.size(), 4);

    // Sequentially building centers and averaging out the vectors.
    // 1: {0, 1}, dist=4, size=2   -> ID 5
    // 2: {2, 5}, dist=17, size=3  -> ID 6
    // 3: {3, 4}, dist=36, size=2  -> ID 7
    // 4: {6, 7}, dist=127, size=5 -> ID 8

    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 4.0, 1e-6);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 5);
    EXPECT_NEAR(merges[1].distance, 17.0, 1e-6);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 4);
    EXPECT_NEAR(merges[2].distance, 36.0, 1e-6);
    EXPECT_EQ(merges[2].new_cluster_size, 2);

    EXPECT_EQ(merges[3].cluster1, 6);
    EXPECT_EQ(merges[3].cluster2, 7);
    EXPECT_NEAR(merges[3].distance, 127.0, 1e-6);
    EXPECT_EQ(merges[3].new_cluster_size, 5);
}


/** @brief Test case for Centroid Linkage Agglomerative Clustering matching SciPy results */
TEST(AgglomerativeDispatcherTest, CentroidLinkageSciPyMatch) {

    Eigen::MatrixXd data(2, 5);
    data.row(0) << 0, 2, 5, 10, 16;
    data.row(1) << 0, 0, 0, 0, 0;

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>,
        LinkageMethod::Centroid>(data, false);

    ASSERT_EQ(merges.size(), 4);

    // Sequence utilizing the GenericLinkage loop internally over non-reducible Lance Williams
    // 1: {0, 1}, dist=4, size=2   -> ID 5
    // 2: {2, 5}, dist=16, size=3  -> ID 6
    // 3: {3, 4}, dist=36, size=2  -> ID 7
    // 4: {6, 7}, dist=113.77777777777777, size=5 -> ID 8

    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 4.0, 1e-6);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 5);
    EXPECT_NEAR(merges[1].distance, 16.0, 1e-6);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 4);
    EXPECT_NEAR(merges[2].distance, 36.0, 1e-6);
    EXPECT_EQ(merges[2].new_cluster_size, 2);

    EXPECT_EQ(merges[3].cluster1, 6);
    EXPECT_EQ(merges[3].cluster2, 7);
    EXPECT_NEAR(merges[3].distance, 113.77777777777777, 1e-5);
    EXPECT_EQ(merges[3].new_cluster_size, 5);
}
