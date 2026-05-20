// ===================================================================================
// nnchain_core.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_NNCHAIN_CORE_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_NNCHAIN_CORE_HPP
// ===================================================================================
/**
 * @brief Class implementation for the core NN-CHAIN algorithm to compute the agglomerative
 * hierarchical clustering merge matrix.
 *
 * @details The Nearest-Neighbor Chain (NN-Chain) algorithm supports agglomerative
 *  hierarchical clustering metrics that satisfy the reducibility property
 * (e.g., Complete Linkage, Ward's Minimum Variance, Average Linkage, etc.)
 * It builds the hierarchy by maintaining a stack of reciprocal nearest neighbors.
 * The time complexity of the algorithm scales as O(N^2) and avoids computing the
 * full distance matrix if updating centroids iteratively. The space complexity is O(N).
 *
 * References:
 * -----------
 * - Murtagh, F., "A Survey of Recent Advances in Hierarchical Clustering Algorithms".
 *   The Computer Journal, 26(4), pp.354-359, 1983.
 * - Murtagh, F., "Algorithms for hierarchical clustering I: an overview".
 *   John Wiley & Sons, Vol. 00, pp.1-4, 2011.
 * - Murtagh, F. "Algorithms for hierarchical clustering II: an overview".
 *   John Wiley & Sons, Vol. 07, pp.1-16, 2017.
 * - Benzécri, J. P., "Construction d'une classification ascendante hiérarchique par
 *   la recherche en chaîne des voisins", Les Cahiers de l'Analyse des Données, 1982.
 *
 * Organization:
 * -------------
 * |- src/ml_methods/clustering/hierarchical/agglomerative/
 * |- NNCHAIN_core.hpp
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

/**
 * @brief NNChain: Core engine for Nearest-Neighbor Chain hierarchical clustering using the
 * Nearest-Neighbor Chain (NN-Chain) algorithm.
 *
 * @details
 * To handle massive datasets the class supports `Eigen::Map` (e.g.,
 * `Eigen::Map<const Eigen::MatrixXf>`). The underlying `Scalar` type seamlessly
 * supports both real numbers  (e.g., `float`, `double`) and complex numbers
 * (e.g., `std::complex<float>`).
 *
 * @tparam MatrixType The type of input data matrix. To handle massive datasets
 * efficiently without triggering deep copies, use Eigen::Map.
 * @tparam DistancePolicy The type of the distance updater functor.
 */
template <typename MatrixType, typename DistancePolicy>
class NNChain {
private:
    /** @brief Number of objects to cluster */
    Eigen::Index num_objects_;

    /** @brief Distance updater functor */
    DistancePolicy distance_updater_;

    /** @brief Vector of merge steps */
    std::vector<MergeStep> merges_;

    /** @brief Vector of cluster sizes */
    std::vector<Eigen::Index> cluster_sizes_;

    /** @brief Vector of active clusters */
    std::vector<bool> active_clusters_;

public:

    /**
     * @brief Constructor
     *
     * @param data Eigen matrix or map containing the variables to cluster.
     *
     * @note Clustering is done over columns (variables). Expected data shape is
     * (n_samples x n_features).
     */
    template <typename... PolicyArgs>
    NNChain(const MatrixType& data, PolicyArgs&&... args)
        : num_objects_(data.cols()),
          distance_updater_(data, std::forward<PolicyArgs>(args)...)
    {
        if (num_objects_ < 2) {
            throw std::invalid_argument("Need at least 2 columns.");
        }

        merges_.reserve(num_objects_ - 1);
        cluster_sizes_.assign(num_objects_ * 2 - 1, 1); // max possible nodes: 2N - 1
        active_clusters_.assign(num_objects_ * 2 - 1, false);

        for (Eigen::Index i = 0; i < num_objects_; ++i) {
            active_clusters_[i] = true;
        }
    }

    /**
     * @brief Perform hierarchical clustering using the NN-CHAIN algorithm.
     *
     * @note The clustering process updates the internal state of the object,
     * including the merge steps and cluster sizes.
     */
    void cluster() {
        std::vector<Eigen::Index> chain; // Stack of active clusters
        chain.reserve(num_objects_);
        // New clusters get IDs: N, N + 1, ..., 2N - 2
        Eigen::Index current_cluster_id = num_objects_;

        for (Eigen::Index step = 0; step < num_objects_ - 1; ++step) {

            // Step 1: If chain is empty, push an arbitrary active cluster
            if (chain.empty()) {
                for (Eigen::Index i = 0; i < current_cluster_id; ++i) {
                    if (active_clusters_[i]) {
                        chain.push_back(i);
                        break;
                    }
                }
            }

            while (true) {
                Eigen::Index c = chain.back();

                // Step 2 & 3: Find the nearest active neighbor 'd' of 'c'
                double min_dist;
                Eigen::Index d = distance_updater_.find_nearest_neighbor(
                    c, active_clusters_, current_cluster_id, min_dist
                );

                // Step 4: Reciprocal Nearest Neighbor (RNN) check
                if (chain.size() > 1 && d == chain[chain.size() - 2]) {
                    // RNN found, merge 'c' and 'd'
                    chain.pop_back(); // Remove 'c' from chain
                    chain.pop_back(); // Remove 'd' from chain

                    // Record the merge
                    merges_.push_back({std::min(c, d), std::max(c, d), min_dist,
                                        cluster_sizes_[c] + cluster_sizes_[d]});

                    // Update state
                    active_clusters_[c] = false;
                    active_clusters_[d] = false;
                    active_clusters_[current_cluster_id] = true;
                    cluster_sizes_[current_cluster_id] = cluster_sizes_[c] + cluster_sizes_[d];

                    // Instruct the policy to update distances for the new merged cluster
                    distance_updater_.merge_clusters(
                        MergeClustersArgs{
                            .c_id = c,
                            .d_id = d,
                            .new_id = current_cluster_id,
                        },
                        cluster_sizes_
                    );
                    // Move to next cluster ID for future merges
                    ++current_cluster_id;
                    // Move to next iteration of the main loop
                    break;

                } else {
                    // Step 5: Not an RNN, push 'd' onto the chain and continue
                    chain.push_back(d);
                }
            }
        }
    }

    // Getter
    // -----------------------
    /** @brief Get the merge steps */
    const std::vector<MergeStep>& get_merges() const {return merges_; }

    /** @brief Get the cluster sizes */
    const std::vector<Eigen::Index>& get_cluster_sizes() const {return cluster_sizes_; }

    /** @brief Get the active clusters */
    const std::vector<bool>& get_active_clusters() const {return active_clusters_; }
};

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */

#endif /* End of ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_NNCHAIN_CORE_HPP */
