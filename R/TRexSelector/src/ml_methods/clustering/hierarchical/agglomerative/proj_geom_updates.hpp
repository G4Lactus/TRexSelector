// ===================================================================================
// projected_geometric_update_policy.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_PROJECTED_GEOMETRIC_POLICY_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_PROJECTED_GEOMETRIC_POLICY_HPP
// ===================================================================================
/**
 * @file projected_geometric_update_policy.hpp
 *
 * @brief SimHash-projected geometric update policy for approximate hierarchical clustering.
 *
 * @details Reads the input data once and projects it to a 64-dimensional shadow via random
 * hyperplanes. All subsequent distance computations use O(1) hardware-accelerated
 * bitwise Hamming distance on 64-bit signatures, giving a ~250x speedup over exact
 * methods for dense latent-factor data.
 *
 */
 // ===================================================================================

// std includes
#include <utils/logging/logger.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdint>
#include <bit> // C++20 std::popcount
#include <random>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

/**
 * @brief SimHash-projected geometric update policy for approximate hierarchical clustering.
 *
 * @details Projects input data to 64 random hyperplanes once at construction, then answers all
 * pairwise distance queries in O(1) via hardware-accelerated Hamming distance on
 * 64-bit binary signatures. Centroid updates are performed directly in 64D space.
 *
 * @note Suitable for dense latent-factor data; accuracy degrades for spatially correlated
 *       (e.g., AR(1)) data where the random hyperplane quantization shatters clusters.
 *
 * @tparam MatrixType        Type of the input data matrix (e.g., Eigen::MatrixXd or Eigen::Map).
 * @tparam DistancePolicyType Distance metric policy type (currently unused; kept for interface
 *                           consistency with other update policy types).
 * @tparam SelectedMethod    Compile-time linkage method (Ward, Average, Centroid, or Median).
 */
template <typename MatrixType, typename DistancePolicyType, LinkageMethod SelectedMethod>
class ProjectedGeometricUpdatePolicy {
public:
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    // The compressed 64-dimensional shadow matrix
    using ProjMatrixType = Eigen::Matrix<RealScalar, 64, Eigen::Dynamic>;

    static_assert(
        SelectedMethod == LinkageMethod::Ward ||
        SelectedMethod == LinkageMethod::Average ||
        SelectedMethod == LinkageMethod::Centroid ||
        SelectedMethod == LinkageMethod::Median,
        "ProjectedGeometricUpdatePolicy only supports geometric linkages.");

private:
    ProjMatrixType V_;
    std::vector<uint64_t> signatures_;

    std::vector<Eigen::Index> id_to_idx_;
    std::vector<Eigen::Index> sizes_;
    Eigen::Index num_original_;

    inline void update_signature(Eigen::Index phys_idx) {
        uint64_t sig = 0;
        for(Eigen::Index b = 0; b < 64; ++b) {
            if(V_(b, phys_idx) > 0.0) {
                sig |= (1ULL << b);
            }
        }
        signatures_[phys_idx] = sig;
    }

public:
    /**
     * @brief Constructor. Reads the massive dataset once, projects it to 64D, and stores in RAM.
     */
    ProjectedGeometricUpdatePolicy(const MatrixType& data, bool /*use_mmap*/ = false)
        : num_original_(data.cols())
    {
        Eigen::Index N = data.rows();

        V_.resize(64, num_original_);
        signatures_.resize(num_original_);

        id_to_idx_.assign(num_original_ * 2 - 1, -1);
        sizes_.assign(num_original_ * 2 - 1, 1);
        for (Eigen::Index i = 0; i < num_original_; ++i) id_to_idx_[i] = i;

        // 1. Generate the 64 random hyperplanes
        Eigen::Matrix<RealScalar, Eigen::Dynamic, 64> H(N, 64);
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 1.0);
        for(Eigen::Index i = 0; i < N; ++i) {
            for(Eigen::Index b = 0; b < 64; ++b) {
                H(i, b) = dist(rng);
            }
        }

        TREX_INFO( "  -> [Backend] Projecting data to 64D ("
                  << (64 * num_original_ * sizeof(RealScalar) / (1024*1024)) << " MB RAM)..." );

        // 2. Compress the dataset in cache-friendly chunks (avoids RAM spikes)
        const Eigen::Index chunk_size = 5000;
        #pragma omp parallel for schedule(dynamic)
        for (Eigen::Index j = 0; j < num_original_; j += chunk_size) {
            Eigen::Index cols = std::min(chunk_size, num_original_ - j);
            // Matrix multiplication applies the projection
            V_.block(0, j, 64, cols) = H.transpose() * data.block(0, j, N, cols);
        }

        // 3. Initialize 64-bit signatures
        #pragma omp parallel for schedule(static)
        for(Eigen::Index j = 0; j < num_original_; ++j) {
            update_signature(j);
        }
    }

    /**
     * @brief Compute the approximate angular distance between two clusters using SimHash.
     *
     * @param c_id Logical cluster ID of the first cluster.
     * @param d_id Logical cluster ID of the second cluster.
     * @return Approximate correlation distance in [0, 2]; returns `numeric_limits::max()` for dead clusters.
     */
    double get_logical_distance(Eigen::Index c_id, Eigen::Index d_id) const {
        const Eigen::Index idx_c = id_to_idx_[c_id];
        const Eigen::Index idx_d = id_to_idx_[d_id];
        if (idx_c == -1 || idx_d == -1) return std::numeric_limits<double>::max();

        // O(1) Hardware-accelerated bitwise Hamming distance
        uint64_t xor_val = signatures_[idx_c] ^ signatures_[idx_d];
        int hamming = std::popcount(xor_val);

        // Convert to correlation approx distance
        double angle = (M_PI * hamming) / 64.0;
        double raw_dist = std::max(0.0, 1.0 - std::cos(angle));

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
     * @param min_dist          Output: approximate distance to the nearest neighbor found.
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
            if (!active_clusters[i] || i == c_id) continue;
            double d = get_logical_distance(c_id, i);
            if (d < min_dist) { min_dist = d; best_d = i; }
        }
        return best_d;
    }

    /**
     * @brief Merge two clusters by updating the 64D projected centroid and binary signature.
     *
     * @param args          Named-argument struct specifying c_id, d_id, and new_id.
     * @param cluster_sizes Current cluster sizes indexed by logical cluster ID.
     */
    void merge_clusters(const MergeClustersArgs& args, const std::vector<Eigen::Index>& cluster_sizes) {
        const Eigen::Index phys_c = id_to_idx_[args.c_id];
        const Eigen::Index phys_d = id_to_idx_[args.d_id];
        const Eigen::Index new_logical = args.new_id;

        double n_c = static_cast<double>(cluster_sizes[args.c_id]);
        double n_d = static_cast<double>(cluster_sizes[args.d_id]);

        // Synthesize new centroid directly in the 64D space! O(64) cost.
        if constexpr (SelectedMethod == LinkageMethod::Median) {
            V_.col(phys_c) = (V_.col(phys_c) + V_.col(phys_d)) / 2.0;
        } else {
            V_.col(phys_c) = (n_c * V_.col(phys_c) + n_d * V_.col(phys_d)) / (n_c + n_d);
        }

        update_signature(phys_c);

        id_to_idx_[new_logical] = phys_c;
        sizes_[new_logical] = cluster_sizes[args.c_id] + cluster_sizes[args.d_id];

        if (args.c_id != new_logical) { id_to_idx_[args.c_id] = -1; }
        if (args.d_id != new_logical) { id_to_idx_[args.d_id] = -1; }
    }
};

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */

#endif /* ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_PROJECTED_GEOMETRIC_POLICY_HPP */
