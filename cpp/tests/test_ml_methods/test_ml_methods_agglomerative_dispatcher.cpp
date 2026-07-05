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
#include <cmath>
#include <vector>

// project ml_methods includes
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp"

// ===================================================================================

// Embed into test namespace
namespace trex::test::ml_methods::clustering::hierarchical::agglomerative {

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
//    When inversions occur, this is more than a row reordering: the monotone rebuild can
//    produce a tree with a DIFFERENT topology than SciPy's chronological dendrogram.
//    The test data below is chosen inversion-free, so the trees coincide.
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


// ===================================================================================
// Correlation-metric tests mirroring the production configuration
// (trex_gvs / trex_da dispatch with DistanceMetric::Correlation and
// Single/Complete/Average/WPGMA linkage).
// ===================================================================================

namespace {

constexpr double kPi = 3.14159265358979323846;

/**
 * @brief Builds a 4-sample x 4-variable matrix of centered columns whose pairwise
 * Pearson correlations are cos(delta_theta) by construction.
 *
 * @details Columns are built as x_j = s_j * (cos(theta_j) e1 + sin(theta_j) e2) with
 * the centered orthogonal basis e1 = (1,-1,0,0), e2 = (0,0,1,-1) and angles
 * theta = {0, 15, 40, 90} degrees. The correlation distance is therefore exactly
 * 1 - |cos(theta_i - theta_j)|. Column scales s_j differ on purpose: the Correlation
 * policy normalizes by the column norms, so scaling must not change the result.
 */
Eigen::MatrixXd make_correlation_test_data() {
    const Eigen::Vector4d e1(1.0, -1.0, 0.0, 0.0);
    const Eigen::Vector4d e2(0.0, 0.0, 1.0, -1.0);
    const double thetas_deg[4] = {0.0, 15.0, 40.0, 90.0};
    const double scales[4] = {1.0, 2.0, 0.5, 3.0};

    Eigen::MatrixXd data(4, 4);
    for (int j = 0; j < 4; ++j) {
        const double t = thetas_deg[j] * kPi / 180.0;
        data.col(j) = scales[j] * (std::cos(t) * e1 + std::sin(t) * e2);
    }
    return data;
}

// Pairwise angular gaps used by the expected heights below:
// d(0,1) = 1-cos15, d(0,2) = 1-cos40, d(0,3) = 1-cos90 = 1,
// d(1,2) = 1-cos25, d(1,3) = 1-cos75, d(2,3) = 1-cos50.
inline double cos_deg(double deg) { return std::cos(deg * kPi / 180.0); }

} // namespace


/** @brief Complete linkage with the Correlation metric (production configuration). */
TEST(AgglomerativeDispatcherTest, CompleteLinkageCorrelationMetric) {
    Eigen::MatrixXd data = make_correlation_test_data();

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation>,
        LinkageMethod::Complete>(data, false, false);

    ASSERT_EQ(merges.size(), 3);

    // 1: {0, 1} at 1-cos15                              -> ID 4
    // 2: {2, 4} at max(1-cos40, 1-cos25) = 1-cos40      -> ID 5
    // 3: {3, 5} at max(1-cos50, 1-cos90) = 1            -> ID 6
    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 1.0 - cos_deg(15.0), 1e-9);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 4);
    EXPECT_NEAR(merges[1].distance, 1.0 - cos_deg(40.0), 1e-9);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 5);
    EXPECT_NEAR(merges[2].distance, 1.0, 1e-9);
    EXPECT_EQ(merges[2].new_cluster_size, 4);
}


/** @brief Average (UPGMA) linkage with the Correlation metric (production configuration). */
TEST(AgglomerativeDispatcherTest, AverageLinkageCorrelationMetric) {
    Eigen::MatrixXd data = make_correlation_test_data();

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation>,
        LinkageMethod::Average>(data, false, false);

    ASSERT_EQ(merges.size(), 3);

    // 1: {0, 1} at 1-cos15                                        -> ID 4
    // 2: {2, 4} at mean of {1-cos40, 1-cos25}                     -> ID 5
    // 3: {3, 5} at mean of {1-cos90, 1-cos75, 1-cos50} over pairs -> ID 6
    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 1.0 - cos_deg(15.0), 1e-9);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 4);
    EXPECT_NEAR(merges[1].distance,
                1.0 - (cos_deg(40.0) + cos_deg(25.0)) / 2.0, 1e-9);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 5);
    EXPECT_NEAR(merges[2].distance,
                1.0 - (cos_deg(90.0) + cos_deg(75.0) + cos_deg(50.0)) / 3.0, 1e-9);
    EXPECT_EQ(merges[2].new_cluster_size, 4);
}


/** @brief WPGMA (McQuitty) linkage with the Correlation metric (production configuration). */
TEST(AgglomerativeDispatcherTest, WPGMALinkageCorrelationMetric) {
    Eigen::MatrixXd data = make_correlation_test_data();

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation>,
        LinkageMethod::WPGMA>(data, false, false);

    ASSERT_EQ(merges.size(), 3);

    // 1: {0, 1} at 1-cos15                                            -> ID 4
    // 2: {2, 4} at (d(2,0)+d(2,1))/2 (same as UPGMA for singletons)   -> ID 5
    // 3: {3, 5} at (d(3,4)+d(3,2))/2 = 1 - cos75/4 - cos50/2          -> ID 6
    //    where d(3,4) = (d(3,0)+d(3,1))/2 = 1 - cos75/2
    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 1.0 - cos_deg(15.0), 1e-9);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 4);
    EXPECT_NEAR(merges[1].distance,
                1.0 - (cos_deg(40.0) + cos_deg(25.0)) / 2.0, 1e-9);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 5);
    EXPECT_NEAR(merges[2].distance,
                1.0 - cos_deg(75.0) / 4.0 - cos_deg(50.0) / 2.0, 1e-9);
    EXPECT_EQ(merges[2].new_cluster_size, 4);
}


/** @brief Test case for Ward Linkage matching SciPy geometry (squared-ESS convention). */
TEST(AgglomerativeDispatcherTest, WardLinkageSciPyMatch) {
    Eigen::MatrixXd data(2, 5);
    data.row(0) << 0, 2, 5, 10, 16;
    data.row(1) << 0, 0, 0, 0, 0;

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>,
        LinkageMethod::Ward>(data, false, false);

    ASSERT_EQ(merges.size(), 4);

    // Lance-Williams Ward on squared distances. Heights equal 2x the ESS increase;
    // SciPy's ward heights are the square roots of these values.
    // 1: {0, 1} at 4       -> ID 5
    // 2: {2, 5} at 64/3    -> ID 6  [(2*25 + 2*9 - 4) / 3]
    // 3: {3, 4} at 36      -> ID 7
    // 4: {6, 7} at 4096/15 -> ID 8  [(4*(529/6) + 4*(1681/6) - 3*36) / 5]
    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 4.0, 1e-9);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 5);
    EXPECT_NEAR(merges[1].distance, 64.0 / 3.0, 1e-9);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 4);
    EXPECT_NEAR(merges[2].distance, 36.0, 1e-9);
    EXPECT_EQ(merges[2].new_cluster_size, 2);

    EXPECT_EQ(merges[3].cluster1, 6);
    EXPECT_EQ(merges[3].cluster2, 7);
    EXPECT_NEAR(merges[3].distance, 4096.0 / 15.0, 1e-9);
    EXPECT_EQ(merges[3].new_cluster_size, 5);
}


/** @brief Test case for Median (WPGMC) Linkage matching SciPy geometry. */
TEST(AgglomerativeDispatcherTest, MedianLinkageSciPyMatch) {
    Eigen::MatrixXd data(2, 5);
    data.row(0) << 0, 2, 5, 10, 16;
    data.row(1) << 0, 0, 0, 0, 0;

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>,
        LinkageMethod::Median>(data, false, false);

    ASSERT_EQ(merges.size(), 4);

    // Squared distances between cluster midpoints:
    // 1: {0, 1} at 4    -> ID 5 (midpoint 1)
    // 2: {2, 5} at 16   -> ID 6 ((5-1)^2, midpoint 3)
    // 3: {3, 4} at 36   -> ID 7 (midpoint 13)
    // 4: {6, 7} at 100  -> ID 8 ((13-3)^2)
    EXPECT_EQ(merges[0].cluster1, 0);
    EXPECT_EQ(merges[0].cluster2, 1);
    EXPECT_NEAR(merges[0].distance, 4.0, 1e-9);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    EXPECT_EQ(merges[1].cluster1, 2);
    EXPECT_EQ(merges[1].cluster2, 5);
    EXPECT_NEAR(merges[1].distance, 16.0, 1e-9);
    EXPECT_EQ(merges[1].new_cluster_size, 3);

    EXPECT_EQ(merges[2].cluster1, 3);
    EXPECT_EQ(merges[2].cluster2, 4);
    EXPECT_NEAR(merges[2].distance, 36.0, 1e-9);
    EXPECT_EQ(merges[2].new_cluster_size, 2);

    EXPECT_EQ(merges[3].cluster1, 6);
    EXPECT_EQ(merges[3].cluster2, 7);
    EXPECT_NEAR(merges[3].distance, 100.0, 1e-9);
    EXPECT_EQ(merges[3].new_cluster_size, 5);
}


/**
 * @brief Centroid linkage with a genuine inversion: the monotone rebuild must emit
 * sorted heights and recompute sizes from the actual rebuilt components.
 *
 * @details Near-equilateral triangle: {0,1} merge chronologically first at 1.0, but
 * their centroid is CLOSER to point 2 (0.81 < 1.0) — an inversion in GenericLinkage's
 * chronological output. The monotone rebuild processes the 0.81 edge first, producing
 * a different (but sorted) topology, as documented in the dispatcher.
 */
TEST(AgglomerativeDispatcherTest, CentroidInversionMonotoneRebuild) {
    Eigen::MatrixXd data(2, 3);
    data.col(0) << 0.0, 0.0;
    data.col(1) << 1.0, 0.0;
    data.col(2) << 0.5, 0.9;

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Euclidean>,
        LinkageMethod::Centroid>(data, false, false);

    ASSERT_EQ(merges.size(), 2);

    // The 0.81 edge joins two singletons (leaves 1 and 2) in the rebuild -> size 2.
    EXPECT_EQ(merges[0].cluster1, 1);
    EXPECT_EQ(merges[0].cluster2, 2);
    EXPECT_NEAR(merges[0].distance, 0.81, 1e-9);
    EXPECT_EQ(merges[0].new_cluster_size, 2);

    // The 1.0 edge then joins leaf 0 with the rebuilt cluster 3 -> size 3.
    EXPECT_EQ(merges[1].cluster1, 0);
    EXPECT_EQ(merges[1].cluster2, 3);
    EXPECT_NEAR(merges[1].distance, 1.0, 1e-9);
    EXPECT_EQ(merges[1].new_cluster_size, 3);
}


/**
 * @brief Ward + Correlation_LSH_Approx routes to the projected geometric engine
 * (64-bit SimHash) and must recover well-separated correlation bundles.
 */
TEST(AgglomerativeDispatcherTest, WardLSHApproxSeparatesClusters) {
    // Two tight bundles of variables: angles {0, 2, 4} and {90, 92, 94} degrees in
    // the plane spanned by two orthogonal sign patterns (48 samples).
    const Eigen::Index n = 48;
    Eigen::VectorXd u(n), w(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        u(i) = (i % 2 == 0) ? 1.0 : -1.0;
        w(i) = (i % 4 < 2) ? 1.0 : -1.0;
    }

    const double thetas_deg[6] = {0.0, 2.0, 4.0, 90.0, 92.0, 94.0};
    Eigen::MatrixXd data(n, 6);
    for (int j = 0; j < 6; ++j) {
        const double t = thetas_deg[j] * kPi / 180.0;
        data.col(j) = std::cos(t) * u + std::sin(t) * w;
    }

    auto merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd,
        DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation_LSH_Approx>,
        LinkageMethod::Ward>(data, false, false);

    ASSERT_EQ(merges.size(), 5);

    // Output must be monotone after the global re-sort and cover all 6 variables.
    for (std::size_t i = 1; i < merges.size(); ++i) {
        EXPECT_GE(merges[i].distance, merges[i - 1].distance);
    }
    EXPECT_EQ(merges.back().new_cluster_size, 6);

    // Cutting at K = 2 must recover the two bundles exactly (intra-bundle Hamming
    // distances ~ 0-2 bits vs ~ 32 bits across bundles).
    auto labels = DendrogramUtils::cut_tree(merges, 6, 2);
    EXPECT_EQ(labels[0], labels[1]);
    EXPECT_EQ(labels[1], labels[2]);
    EXPECT_EQ(labels[3], labels[4]);
    EXPECT_EQ(labels[4], labels[5]);
    EXPECT_NE(labels[0], labels[3]);
}


/** @brief The mmap distance-matrix backend must produce results identical to RAM. */
TEST(AgglomerativeDispatcherTest, MmapBackendMatchesRam) {
    Eigen::MatrixXd data = make_correlation_test_data();

    using CorrPolicy = DistancePolicy<Eigen::MatrixXd, DistanceMetric::Correlation>;

    auto ram_merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd, CorrPolicy, LinkageMethod::Average>(
            data, /*use_mmap=*/false, /*verbose=*/false);
    auto mmap_merges = AgglomerativeClustering::cluster<
        Eigen::MatrixXd, CorrPolicy, LinkageMethod::Average>(
            data, /*use_mmap=*/true, /*verbose=*/false);

    ASSERT_EQ(ram_merges.size(), mmap_merges.size());
    for (std::size_t i = 0; i < ram_merges.size(); ++i) {
        EXPECT_EQ(ram_merges[i].cluster1, mmap_merges[i].cluster1);
        EXPECT_EQ(ram_merges[i].cluster2, mmap_merges[i].cluster2);
        EXPECT_DOUBLE_EQ(ram_merges[i].distance, mmap_merges[i].distance);
        EXPECT_EQ(ram_merges[i].new_cluster_size, mmap_merges[i].new_cluster_size);
    }
}

// ===================================================================================
} /* namespace trex::test::ml_methods::clustering::hierarchical::agglomerative */
