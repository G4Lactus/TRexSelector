// ===================================================================================
// geometric_update_policy.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_GEOMETRIC_UPDATE_POLICY_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_GEOMETRIC_UPDATE_POLICY_HPP
// ===================================================================================
/**
 * @file geometric_update_policy.hpp
 *
 * @brief Geometric update policy for hierarchical agglomerative clustering.
 *
 */
 // ===================================================================================

// std include
#include <vector>
#include <limits>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

/**
 * @brief Geometric centroid update policy for Ward, Average, Centroid, and Median linkage.
 *
 * @details Maintains a dense centroid matrix of strictly O(N × P) memory by recycling
 * physical column slots after each merge. Centroid arithmetic is performed in-place
 * to avoid memory blowouts on large datasets.
 *
 * @tparam MatrixType        Type of the input data matrix (e.g., Eigen::MatrixXd or Eigen::Map).
 * @tparam DistancePolicyType Distance metric policy type; must provide a
 *                           `get_distance(i, j)` method and `static constexpr DistanceMetric Metric`.
 * @tparam SelectedMethod    Compile-time linkage method (Ward, Average, Centroid, or Median).
 */
template <typename MatrixType, typename DistancePolicyType, LinkageMethod SelectedMethod>
class GeometricUpdatePolicy {
public:
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using CentroidMatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // Rebind the distance policy to natively accept our dense matrix
    using InternalDistancePolicy = DistancePolicy<CentroidMatrixType, DistancePolicyType::Metric>;

    static_assert(
        SelectedMethod == LinkageMethod::Ward ||
        SelectedMethod == LinkageMethod::Average ||
        SelectedMethod == LinkageMethod::Centroid ||
        SelectedMethod == LinkageMethod::Median,
        "GeometricUpdatePolicy only supports Ward, Average, Centroid, and Median linkages.");

private:
    /** @brief Matrix holding the active cluster centroids. Size is strictly N x P. */
    CentroidMatrixType centroids_;

    /** @brief Distance policy initialized with the centroids_ matrix. */
    InternalDistancePolicy dist_policy_;

    /** @brief Maps logical cluster ID (0 to 2P-2) to physical matrix column index (0 to P-1). */
    std::vector<Eigen::Index> id_to_idx_;

    /** @brief Internal cache of cluster sizes required for Ward's weighting. */
    std::vector<Eigen::Index> sizes_;

    /** @brief Number of original objects (columns) in the input data matrix. */
    Eigen::Index num_original_;

public:
    /**
     * @brief Constructor. Allocates strictly O(N * P) memory and recycles slots.
     */
    GeometricUpdatePolicy(const MatrixType& data)
        : num_original_(data.cols()),
          // Allocate strictly P columns, halving the memory footprint!
          centroids_(data.rows(), data.cols()),
          // dist_policy_ now matches centroids_ type perfectly.
          dist_policy_(centroids_)
    {
        // 1. Copy raw data. Total active footprint equals original NNChain + Condensed1D.
        centroids_ = data;

        // 2. Initialize tracking arrays (Logical IDs go up to 2P-2)
        id_to_idx_.assign(num_original_ * 2 - 1, -1);
        sizes_.assign(num_original_ * 2 - 1, 1);

        for (Eigen::Index i = 0; i < num_original_; ++i) {
            id_to_idx_[i] = i; // Initial mapping is 1:1
        }
    }

    /**
     * @brief Compute the logical distance between two clusters.
     *
     * @param c_id Logical cluster ID of the first cluster.
     * @param d_id Logical cluster ID of the second cluster.
     * @return Distance between the two clusters; returns `numeric_limits::max()` for dead clusters.
     */
    double get_logical_distance(Eigen::Index c_id, Eigen::Index d_id) const {
        const Eigen::Index idx_c = id_to_idx_[c_id];
        const Eigen::Index idx_d = id_to_idx_[d_id];

        if (idx_c == -1 || idx_d == -1) {
            return std::numeric_limits<double>::max();
        }

        double raw_dist = dist_policy_.get_distance(idx_c, idx_d);

        if constexpr (SelectedMethod == LinkageMethod::Ward) {
            double n_c = static_cast<double>(sizes_[c_id]);
            double n_d = static_cast<double>(sizes_[d_id]);
            return raw_dist * ((n_c * n_d) / (n_c + n_d));
        }

        return raw_dist;
    }

    /**
     * @brief Find the nearest active neighbor of a given cluster.
     *
     * @param c_id              Logical cluster ID to search from.
     * @param active_clusters   Boolean flags indicating which cluster IDs are still active.
     * @param current_cluster_id Upper bound (exclusive) of cluster IDs to consider.
     * @param min_dist          Output: distance to the nearest neighbor found.
     * @return Logical cluster ID of the nearest neighbor, or -1 if none found.
     */
    Eigen::Index find_nearest_neighbor(
        Eigen::Index c_id,
        const std::vector<bool>& active_clusters,
        Eigen::Index current_cluster_id,
        double& min_dist
    ) const {
        min_dist = std::numeric_limits<double>::max();
        Eigen::Index best_d = -1;

        for (Eigen::Index i = 0; i < current_cluster_id; ++i) {
            if (!active_clusters[i] || i == c_id) { continue; }

            double d = get_logical_distance(c_id, i);

            if (d < min_dist) {
                min_dist = d;
                best_d = i;
            }
        }
        return best_d;
    }

    /**
     * @brief Synthesizes the new geometric centroid in-place, preventing memory blowouts.
     *
     * @param args          Named-argument struct specifying c_id, d_id, and new_id.
     * @param cluster_sizes Current cluster sizes indexed by logical cluster ID.
     */
    void merge_clusters(
        const MergeClustersArgs& args,
        const std::vector<Eigen::Index>& cluster_sizes)
    {
        // Physical slots in the strictly N x P matrix
        const Eigen::Index phys_c = id_to_idx_[args.c_id];
        const Eigen::Index phys_d = id_to_idx_[args.d_id];
        const Eigen::Index new_logical = args.new_id;

        double n_c = static_cast<double>(cluster_sizes[args.c_id]);
        double n_d = static_cast<double>(cluster_sizes[args.d_id]);

        // Synthesize the new centroid and OVERWRITE phys_c to save memory
        if constexpr (SelectedMethod == LinkageMethod::Median) {
            centroids_.col(phys_c) = (centroids_.col(phys_c) + centroids_.col(phys_d)) / 2.0;
        } else {
            centroids_.col(phys_c) = (n_c * centroids_.col(phys_c) +
                                      n_d * centroids_.col(phys_d)) / (n_c + n_d);
        }

        // Trigger the distance cache update for the recycled physical slot
        dist_policy_.update_node(phys_c);

        // Map the new logical cluster ID to the recycled physical slot
        id_to_idx_[new_logical] = phys_c;
        sizes_[new_logical] = cluster_sizes[args.c_id] + cluster_sizes[args.d_id];

        // Mark old logical clusters as dead (safeguarded against GenericLinkage recycling IDs)
        if (args.c_id != new_logical) { id_to_idx_[args.c_id] = -1; }
        if (args.d_id != new_logical) { id_to_idx_[args.d_id] = -1; }
    }
};

// ===================================================================================
} /* End of trex::ml_methods::clustering::hierarchical::agglomerative  */

#endif /* ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_GEOMETRIC_UPDATE_POLICY_HPP */
