// ===================================================================================
// dendrogram_utils.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DENDROGRAM_UTILS_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DENDROGRAM_UTILS_HPP
// ===================================================================================
/**
 * @brief Dendrogram_Utils.hpp
 * Utility functions for processing and cutting hierarchical clustering dendrograms.
 *
 * @details
 * Provides O(N log(N)) algorithms using Union-Find to convert the pointer
 * representation from SLINK/CLINK into standard merge matrices.
 * Also, extracts flat cluster labels by cutting the dendrogram tree at K clusters.
 *
 * |- src/ml_methods/clustering/hierarchical/agglomerative
 * |- Dendrogram_Utils.hpp
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// ===================================================================================

// Embedded into namespace trex::ml_methods::clustering::hierarchical::agglomerative
namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

/**
 * @class UnionFind
 * @brief An optimized Union-Find (Disjoint Set Union) data structure with path
 * compression and union by rank.
 */
class UnionFind {
private:
    /* Parent pointers */
    std::vector<Eigen::Index> parent_;

    /* Size of each set */
    std::vector<Eigen::Index> size_;

    /* Active cluster IDs */
    std::vector<Eigen::Index> active_cluster_id_;

public:
    /** @brief Named arguments for unite() to prevent accidental argument swaps. */
    struct UniteArgs {
        Eigen::Index i_idx;
        Eigen::Index j_idx;
        Eigen::Index new_cluster_id;
    };

    /**
     * @brief Construct a new UnionFind object
     *
     * @param n The number of elements (original variables/objects). Initially, each element is in
     *  its own cluster.
     */
    UnionFind(Eigen::Index n) {
        parent_.resize(n);
        size_.assign(n, 1);
        active_cluster_id_.resize(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            parent_[i] = i;
            // Initially, each element is its own cluster
            active_cluster_id_[i] = i;
        }
    }

    /**
     * @brief Find the root of the set containing element `i` with path compression.
     *
     * @param i The element index.
     *
     * @return The root of the set containing element `i`.
     */
    Eigen::Index find(Eigen::Index i) {
        if (parent_[i] == i) { return i; }
        // Path compression
        parent_[i] = find(parent_[i]);
        return parent_[i];
    }

    /**
     * @brief Unite two sets containing elements `i` and `j` with a new cluster ID.
     *
     * @param args Named arguments specifying the two element indices and the new cluster ID.
     */
    void unite(const UniteArgs& args) {
        Eigen::Index root_i = find(args.i_idx);
        Eigen::Index root_j = find(args.j_idx);

        if (root_i != root_j) {
            // Union by size
            if (size_[root_i] < size_[root_j]) {
                std::swap(root_i, root_j);
            }
            parent_[root_j] = root_i;
            size_[root_i] += size_[root_j];

            // The surviving root takes the new ID of the merged cluster
            active_cluster_id_[root_i] = args.new_cluster_id;
        }
    }

    /**
     * @brief Get the cluster ID of the set containing element `i_idx`.
     *
     * @param i_idx The element index.
     * @return The cluster ID of the set containing element `i_idx`.
     */
    Eigen::Index get_cluster_id(Eigen::Index i_idx) {
        return active_cluster_id_[find(i_idx)];
    }

    /**
     * @brief Get the size of the cluster containing element `i_idx`.
     *
     * @param i_idx The element index.
     * @return The size of the cluster containing element `i_idx`.
     */
    Eigen::Index get_cluster_size(Eigen::Index i_idx) {
        return size_[find(i_idx)];
    }
};

// ---------------------------------------------------------

/**
 * @class DendrogramUtils
 * @brief Static utility methods for dendrogram manipulation.
 */
class DendrogramUtils {
public:

    /**
     * @brief Converts the Pointer Representation (pi, lambda) from SLINK/CLINK into a
     * standard Hierarchical Clustering Merge Matrix.
     *
     * @param pi The parent pointer array.
     * @param lambda The distance array.
     *
     * @return std::vector<MergeStep> The resulting (N-1) merge matrix.
     */
    template<typename RealScalar>
    static std::vector<MergeStep> pointer_to_merge_matrix(
        const std::vector<Eigen::Index>& pi,
        const std::vector<RealScalar>& lambda)
    {
        Eigen::Index n = static_cast<Eigen::Index>(pi.size());
        std::vector<MergeStep> merges;
        merges.reserve(n - 1);

        // Step 1. Extract all valid edges from the pointer representation
        struct Edge {
            Eigen::Index u;
            Eigen::Index v;
            double weight;

            bool operator<(const Edge& other) const {
                return weight < other.weight;
            }
        };

        std::vector<Edge> edges;
        edges.reserve(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            if (pi[i] != i && lambda[i] != std::numeric_limits<double>::max()) {
                edges.push_back({i, pi[i], static_cast<double>(lambda[i])});
            }
        }

        // Step 2. Sort edges by weight (distance) - Kruskal's algorithm - stable sort required
        std::stable_sort(edges.begin(), edges.end());

        // Step 3. Rebuild the merge sequence using Union-Find
        UnionFind uf(n);
        Eigen::Index current_new_id = n;

        for (const auto& edge: edges) {
            Eigen::Index root_u = uf.find(edge.u);
            Eigen::Index root_v = uf.find(edge.v);

            if (root_u != root_v) {
                Eigen::Index id_u = uf.get_cluster_id(root_u);
                Eigen::Index id_v = uf.get_cluster_id(root_v);
                Eigen::Index new_size_u = uf.get_cluster_size(root_u) +
                                          uf.get_cluster_size(root_v);

                // Standardize: smaller ID first
                merges.push_back({
                    std::min(id_u, id_v),
                    std::max(id_u, id_v),
                    edge.weight,
                    new_size_u
                });

                uf.unite(UnionFind::UniteArgs{
                    .i_idx = root_u,
                    .j_idx = root_v,
                    .new_cluster_id = current_new_id,
                });
                current_new_id++;
            }
        }

        return merges;
    }

    // ---------------------------------------------------------

    /**
     * @brief Formats an unsorted NN-Chain merge array into a standard,
     * distance-sorted hierarchical merge matrix using Kruskal's algorithm.
     */
    static std::vector<MergeStep> format_nnchain_merges(
        const std::vector<MergeStep>& nn_merges,
        Eigen::Index num_orig_objs)
    {
        // 1. Find a representative leaf for every node in the tree
        std::vector<Eigen::Index> leaf_rep(num_orig_objs * 2 - 1);
        for (Eigen::Index i = 0; i < num_orig_objs; ++i) {
            leaf_rep[i] = i;
        }

        Eigen::Index current_id = num_orig_objs;
        for (const auto& m : nn_merges) {
            leaf_rep[current_id] = leaf_rep[m.cluster1];
            current_id++;
        }

        // 2. Create a list of edges using the leaf representatives
        struct Edge {
            Eigen::Index u;
            Eigen::Index v;
            double weight;
            Eigen::Index size;
            bool operator<(const Edge& other) const { return weight < other.weight; }
        };

        std::vector<Edge> edges;
        edges.reserve(nn_merges.size());
        for (const auto& m : nn_merges) {
            edges.push_back({leaf_rep[m.cluster1], leaf_rep[m.cluster2],
                             m.distance, m.new_cluster_size});
        }

        // 3. Sort edges globally by distance (Resolves NN-Chain disjoint jumps)
        std::sort(edges.begin(), edges.end());

        // 4. Rebuild the standard chronological merge sequence
        std::vector<MergeStep> standard_merges;
        standard_merges.reserve(nn_merges.size());
        UnionFind uf(num_orig_objs);
        Eigen::Index new_cluster_id = num_orig_objs;

        for (const auto& edge : edges) {
            Eigen::Index root_u = uf.find(edge.u);
            Eigen::Index root_v = uf.find(edge.v);

            if (root_u != root_v) {
                Eigen::Index id_u = uf.get_cluster_id(root_u);
                Eigen::Index id_v = uf.get_cluster_id(root_v);

                standard_merges.push_back({
                    std::min(id_u, id_v),
                    std::max(id_u, id_v),
                    edge.weight,
                    edge.size
                });

                uf.unite(UnionFind::UniteArgs{
                    .i_idx = root_u,
                    .j_idx = root_v,
                    .new_cluster_id = new_cluster_id
                });
                new_cluster_id++;
            }
        }
        return standard_merges;
    }

    /**
     * @brief Extracts flat cluster labels by cutting the dendrogram to form K clusters.
     *
     * @param merges The standard (N-1) merge matrix (from NNChain or pointer_to_merge_matrix).
     * @param num_orig_objs The number of original variables/objects (N).
     * @param num_clusters The desired number of clusters (K).
     *
     * @return std::vector<Eigen::Index> An array of size N where index `i` contains the
     * assigned cluster label (0 to K-1) for the original variable/object `i`.
     */
    static std::vector<Eigen::Index> cut_tree(
        const std::vector<MergeStep>& merges,
        Eigen::Index num_orig_objs,
        Eigen::Index num_clusters)
    {
        if (num_clusters < 1 || num_clusters > num_orig_objs) {
            throw std::invalid_argument(
                "Requested number of clusters must be between 1 and the number of original objects."
            );
        }

        Eigen::Index n = num_orig_objs;

        // If K == N, every object is its own cluster
        if (num_clusters == n) {
            std::vector<Eigen::Index> labels(n);
            std::iota(labels.begin(), labels.end(), 0);
            return labels;
        }

        // Merge number for exactly K active trees
        Eigen::Index merges_to_process = n - num_clusters;

        // Use Union-Find to merge
        UnionFind uf(n);

        // Map from any cluster ID (0 to 2N - 2) to an original item inside it
        std::vector<Eigen::Index> cluster_to_item(n * 2 - 1);
        for (Eigen::Index i = 0; i < n; ++i) {
            cluster_to_item[i] = i;
        }

        for (Eigen::Index i = 0; i < merges_to_process; ++i) {
            Eigen::Index c1 = merges[i].cluster1;
            Eigen::Index c2 = merges[i].cluster2;
            Eigen::Index new_id = n + i;

            Eigen::Index item1 = cluster_to_item[c1];
            Eigen::Index item2 = cluster_to_item[c2];

            // Unite the items using our optimized class
            uf.unite(UnionFind::UniteArgs{
                .i_idx = item1,
                .j_idx = item2,
                .new_cluster_id = new_id
            });

            // The new cluster maps to the root of the newly merged set
            cluster_to_item[new_id] = uf.find(item1);
        }

        // Assign labels (0 to K-1)
        std::vector<Eigen::Index> final_labels(n);
        Eigen::Index current_label = 0;
        std::vector<Eigen::Index> root_to_label(n, -1);

        for (Eigen::Index i = 0; i < n; ++i) {
            // Path compression
            Eigen::Index root = uf.find(i);

            if (root_to_label[root] == -1) {
                root_to_label[root] = current_label++;
            }

            final_labels[i] = root_to_label[root];
        }

        return final_labels;
    }

    /**
     * @brief Extracts flat cluster labels by cutting the dendrogram at a specific height (distance).
     *
     * @param merges The standard (N-1) sorted merge matrix.
     * @param num_orig_objs The number of original variables/objects.
     * @param height The maximum distance threshold. Merges above this distance are severed.
     *
     * @return std::vector<Eigen::Index> An array of size N where index `i` contains the cluster
     *  labels.
     */
    static std::vector<Eigen::Index> cut_tree_by_height(
        const std::vector<MergeStep>& merges,
        Eigen::Index num_orig_objs,
        double height
    ) {
        Eigen::Index merges_to_process = 0;

        for (const auto& m: merges) {
            if (m.distance <= height) {
                merges_to_process++;
            } else {
                break;
            }
        }

        // If we processed M merges on N objects, we are left with exactly N - M clusters
        Eigen::Index resulting_k = num_orig_objs - merges_to_process;

        // Reuse K-cluster cutting
        return cut_tree(merges, num_orig_objs, resulting_k);
    }

    /**
     * @brief Groups the original variabl indices into their respective clusters.
     *
     * @param labels The label array returned by cut_tree or cut_tree_by_height.
     *
     * @return std::vector<std::vector<Eigen::Index>> A list of buckets, where each
     *  bucket contains the column indices of the variables in that cluster.
     */
    static std::vector<std::vector<Eigen::Index>> group_indices_by_label(
        const std::vector<Eigen::Index>& labels)
    {
        // Find the maximum label to know how many buckets are needed
        Eigen::Index k = 0;
        for (auto l : labels) { if (l > k) k = l; }
        k++; // Since labels are 0-indexed

        std::vector<std::vector<Eigen::Index>> clusters(k);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(labels.size()); ++i) {
            clusters[labels[i]].push_back(i);
        }

        return clusters;
    }

}; /* End of class DendrogramUtils */

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */
// ===================================================================================
#endif /* End of ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DENDROGRAM_UTILS_HPP */
