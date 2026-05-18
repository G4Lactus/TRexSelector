// =======================================================
// agnes_core.hpp
// =======================================================

#ifndef TREX_ML_METHODS_CLUSTERING_DENDROGRAMS_AGNES_CORE_HPP
#define TREX_ML_METHODS_CLUSTERING_DENDROGRAMS_AGNES_CORE_HPP

#include <queue>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

namespace trex {
namespace ml_methods {
namespace clustering {
namespace dendrograms {

#ifndef MERGE_STEP_DEFINED
#define MERGE_STEP_DEFINED
struct MergeStep {
    Eigen::Index cluster1;
    Eigen::Index cluster2;
    double distance;
    Eigen::Index new_size;
};
#endif

template <typename MatrixType, typename DistancePolicy>
class AGNES {
private:
    Eigen::Index num_objects_;
    DistancePolicy dist_updater_;
    std::vector<MergeStep> merges_;
    std::vector<Eigen::Index> cluster_sizes_;
    std::vector<bool> active_clusters_;

    // The Heap Structure
    struct Edge {
        Eigen::Index u;
        Eigen::Index v;
        double weight;
        bool operator>(const Edge& other) const { return weight > other.weight; }
    };

public:
    AGNES(const MatrixType& data)
        : num_objects_(data.cols()), dist_updater_(data)
    {
        if (num_objects_ < 2) {
            throw std::invalid_argument("Need at least 2 objects to cluster.");
        }

        merges_.reserve(num_objects_ - 1);
        cluster_sizes_.assign(num_objects_ * 2 - 1, 1);
        active_clusters_.assign(num_objects_ * 2 - 1, false);

        for (Eigen::Index i = 0; i < num_objects_; ++i) {
            active_clusters_[i] = true;
        }
    }

    void cluster() {
        // Initialize the Min-Heap
        std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> min_heap;

        // Populate initial P(P-1)/2 edges
        for (Eigen::Index i = 0; i < num_objects_; ++i) {
            for (Eigen::Index j = 0; j < i; ++j) {
                double dist = dist_updater_.get_distance(i, j);
                min_heap.push({i, j, dist});
            }
        }

        Eigen::Index current_cluster_id = num_objects_;

        // Exceute P-1 merges
        for (Eigen::Index step = 0; step < num_objects_ - 1; ++step) {
            Edge top_edge;

            // Pop the heap until we find an edge connecting two active clusters
            while (!min_heap.empty()) {
                top_edge = min_heap.top();
                min_heap.pop();
                if (active_clusters_[top_edge.u] && active_clusters_[top_edge.v]) {
                    break;
                }
            }

            Eigen::Index c = top_edge.u;
            Eigen::Index d = top_edge.v;

            // Record the merge
            merges_.push_back({std::min(c, d), std::max(c, d),
            top_edge.weight, cluster_sizes_[c] + cluster_sizes_[d]});

            // Update cluster states
            active_clusters_[c] = false;
            active_clusters_[d] = false;
            active_clusters_[current_cluster_id] = true;
            cluster_sizes_[current_cluster_id] = cluster_sizes_[c] + cluster_sizes_[d];

            // Lance Williams Update
            dist_updater_.merge_clusters(
                typename DistancePolicy::MergeClustersArgs{c, d, current_cluster_id},
                cluster_sizes_
            );

            // Push the new edges into the heap
            for (Eigen::Index k = 0; k < current_cluster_id; ++k) {
                if (active_clusters_[k]) {
                    double new_dist = dist_updater_.get_distance(current_cluster_id, k);
                    min_heap.push({current_cluster_id, k, new_dist});
                }
            }
            current_cluster_id++;
        }
    }

    const std::vector<MergeStep>& get_merges() const { return merges_; }
};

}   /* End of namespace dendrograms */
}   /* End of namespace clustering */
}   /* End of namespace ml_methods */
}   /* End of namespace trex */

#endif /* End of TREX_ML_METHODS_CLUSTERING_DENDROGRAMS_AGNES_CORE_HPP */
