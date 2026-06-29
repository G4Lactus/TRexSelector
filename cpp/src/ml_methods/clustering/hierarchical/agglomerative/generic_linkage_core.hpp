// ===================================================================================
// generic_linkage_core.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_GENERIC_LINKAGE_CORE_HPP
#define TREX_ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_GENERIC_LINKAGE_CORE_HPP
// ===================================================================================
/**
 * @brief Class implementation for Müllner's generic Generic Linkage algorithm.
 *
 * @details This algorithm operates using a lazy-evaluated priority queue bounded
 * strictly to N elements. It efficiently computes cluster merges for methods that
 * do not satisfy the reducibility property (such as Centroid and Median linkage),
 * avoiding the O(N^3) time penalty of naive AGNES implementations.
 *
 * References:
 * -----------
 * Müllner, D., "Modern hierarchical, agglomerative clustering algorithms",
 *   arXiv preprint arXiv:1109.2378v1, pp.1-29, 2011.
 *
 * Organization:
 * -------------
 * |- src/ml_methods/clustering/hierarchical/agglomerative/
 * |- Generic_Linkage_core.hpp
 */
// ===================================================================================

// std includes
#include <queue>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

/**
 * @brief GenericLinkage: Core engine for Müllner's generic agglomerative clustering algorithm.
 *
 * @details
 * To handle massive datasets the class supports `Eigen::Map` (e.g.,
 * `Eigen::Map<const Eigen::MatrixXf>`). The underlying `Scalar` type seamlessly
 * supports both real numbers  (e.g., `float`, `double`) and complex numbers
 * (e.g., `std::complex<float>`).
 *
 * @tparam MatrixType The type of input data matrix. To handle massive datasets
 * efficiently without triggering deep copies, use Eigen::Map.
 * @tparam DistanceUpdater The type of the distance updater functor.
 */
template <typename MatrixType, typename DistanceUpdater>
class GenericLinkage {
private:
    /** @brief Number of objects to cluster */
    Eigen::Index num_objects_;

    /** @brief Distance updater functor */
    DistanceUpdater dist_updater_;

    /** @brief Vector of merge steps */
    std::vector<MergeStep> merges_;

    /** @brief Struct to represent elements in the priority queue */
    struct QueueElement {
        /** @brief Index of the first cluster */
        Eigen::Index u;

        /** @brief Index of the second cluster */
        Eigen::Index v;

        /** @brief Distance between the two clusters */
        double dist;

        /** @brief Comparison operator for priority queue */
        bool operator>(const QueueElement& other) const { return dist > other.dist; }
    };

public:

    /** @brief Constructor for GenericLinkage class.
     *
     * @param data Eigen::matrix or map containing the variables to cluster.
     */
    template <typename... PolicyArgs>
    GenericLinkage(const MatrixType& data, PolicyArgs&&... args)
        : num_objects_(data.cols()), dist_updater_(data, std::forward<PolicyArgs>(args)...)
    {
        if (num_objects_ < 2) {
            throw std::invalid_argument("Need at least 2 columns.");
        }
        merges_.reserve(num_objects_ - 1);
    }

    /** @brief Perform hierarchical clustering using Müllner's generic algorithm. */
    void cluster() {
        std::vector<Eigen::Index> S;    // Active cluster labels
        S.reserve(num_objects_);
        for (Eigen::Index i = 0; i < num_objects_; ++i) { S.push_back(i); }

        std::vector<Eigen::Index> cluster_size(num_objects_, 1);
        std::vector<Eigen::Index> nearest_neighbor(num_objects_, -1);
        std::vector<double> min_dist(num_objects_, std::numeric_limits<double>::max());
        std::vector<bool> active(num_objects_, true); // Safety tracker for Ghost nodes

        std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> Q;

        // 1. Initialize nearest neighbors (checking y > x)
        for (Eigen::Index x = 0; x < num_objects_ - 1; ++x) {
            double min_d = std::numeric_limits<double>::max();
            Eigen::Index best_y = -1;
            for (Eigen::Index y = x + 1; y < num_objects_; ++y) {
                double d = dist_updater_.get_logical_distance(x, y);
                if (d < min_d) {
                    min_d = d;
                    best_y = y;
                }
            }
            nearest_neighbor[x] = best_y;
            min_dist[x] = min_d;
            if (best_y != -1) { Q.push({x, best_y, min_d}); }
        }

        // 2. Main loop
        for (Eigen::Index step = 0; step < num_objects_ - 1; ++step) {
            Eigen::Index a = -1, b = -1;
            double delta = 0.0;

            while (!Q.empty()) {
                QueueElement top = Q.top();
                Q.pop();

                a = top.u;
                // Guard: Ignore ghost nodes
                if (!active[a]) { continue; }

                // Guard: Ignore stale lower bounds
                if (top.dist > min_dist[a]) { continue; }

                b = nearest_neighbor[a];

                // Müllner's Lazy Evaluation: Is 'b' still valid and distance accurate?
                if (b != -1 && active[b] && top.dist == dist_updater_.get_logical_distance(a, b)) {
                    delta = top.dist;
                    break;
                }

                // If invalid, recalculate nearest neighbor for 'a' among active clusters
                double min_d = std::numeric_limits<double>::max();
                Eigen::Index best_y = -1;
                for (Eigen::Index y : S) {
                    if (y > a) {
                        double d = dist_updater_.get_logical_distance(a, y);
                        if (d < min_d) { min_d = d; best_y = y; }
                    }
                }
                nearest_neighbor[a] = best_y;
                min_dist[a] = min_d;
                if (best_y != -1) Q.push({a, best_y, min_d});
            }

            // Mark 'a' as a ghost node. Re-use 'b' for the new cluster.
            active[a] = false;

            // 3. Record the merge
            merges_.push_back({a, b, delta, cluster_size[a] + cluster_size[b]});
            S.erase(std::remove(S.begin(), S.end(), a), S.end());

            // 4. Lance-Williams Distance Update
            dist_updater_.merge_clusters(
                MergeClustersArgs{a, b, b},
                cluster_size
            );

            // 5. Update states (utilize index 'b' for the new cluster size)
            cluster_size[b] += cluster_size[a];

            // 6. Update nearest neighbor candidates
            for (Eigen::Index x : S) {
                if (x < a) {
                    if (nearest_neighbor[x] == a) {
                        nearest_neighbor[x] = b;
                    }
                }
            }

            for (Eigen::Index x : S) {
                if (x < b) {
                    double d_xb = dist_updater_.get_logical_distance(x, b);
                    if (d_xb < min_dist[x]) {
                        nearest_neighbor[x] = b;
                        min_dist[x] = d_xb;
                        Q.push({x, b, d_xb});   // preserve the lower bound
                    }
                }
            }

            // Find the nearest neighbor for the newly merged cluster 'b'
            double min_d = std::numeric_limits<double>::max();
            Eigen::Index best_y = -1;
            for (Eigen::Index y : S) {
                if (y > b) {
                    double d = dist_updater_.get_logical_distance(b, y);
                    if (d < min_d) { min_d = d; best_y = y; }
                }
            }
            nearest_neighbor[b] = best_y;
            min_dist[b] = min_d;
            if (best_y != -1) Q.push({b, best_y, min_d});
        }
    }

    /** @brief Get the merge steps resulting from the clustering.
     *  @return Const reference to the vector of MergeStep objects in merge order.
     */
    const std::vector<MergeStep>& get_merges() const { return merges_; }
};

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */

#endif /* End of TREX_ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_GENERIC_LINKAGE_CORE_HPP */
