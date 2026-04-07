// ===================================================================================
// cluster_update_policy.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_CLUSTER_UPDATE_POLICY_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_CLUSTER_UPDATE_POLICY_HPP
// ===================================================================================
/**
 * @brief Cluster_Update_Policy.hpp
 *  Unified dynamic distance policies for the NN-Chain algorithm.
 *
 * @details Uses the Lance-Williams update for merging clusters
 * in O(1) time. Offers three storage backends selectable via StorageMode:
 *  - Full2D:      Full N×N Eigen matrix. Use for tiny P or non-reducible methods.
 *  - Condensed1D: 1D packed lower-triangular vector. 50% memory saving vs Full2D.
 *  - LazyHashMap: On-demand hash map. Only allocates entries actually accessed,
 *                 making it the only viable option when O(P²) allocation is impossible
 *                 (e.g. P = 400k, where even the condensed form would require ~640 GB).
 *
 * Organization:
 * -------------
 * |- src/ml_methods/clustering/hierarchical/agglomerative
 * |- Cluster_Update_Policy.hpp
 */
// ===================================================================================

// std includes
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================
// Common Types & Enums
// ===================================================================================

/**
 * @brief Available linkage methods for dynamical cluster updating.
 */
enum class LinkageMethod {
    Ward,       // Ward's Minimum Variance
    Average,    // UPGMA / Average Linkage
    Complete,   // Maximum Linkage
    Single,     // Minimum Linkage
    WPGMA,      // Weighted Average Linkage / McQuitty
    Median,     // Median Linkage (WPGMC) - not supported by NN-CHAIN
    Centroid    // Centroid Linkage (UPGMC) - not supported by NN-CHAIN
};


/**
 * @brief Storage backend for the pairwise distance matrix.
 *
 * | Mode        | Memory       | Init cost | Best for                               |
 * |-------------|--------------|-----------|----------------------------------------|
 * | Full2D      | O(P²)        | O(P²)     | tiny P; Centroid/Median (non-reducible)|
 * | Condensed1D | O(P²/2)      | O(P²)     | moderate P, fits in RAM                |
 * | LazyHashMap | O(accessed)  | O(P)      | large P where O(P²) allocation fails   |
 *
 */
enum class StorageMode {
    Full2D,       /* Full N×N Eigen matrix. */
    Condensed1D,  /* Lower-triangular 1D packed vector. */
    LazyHashMap   /* On-demand hash map; allocates only accessed pairs. */
};


// ===================================================================================
// Internal mathematical utilities
// ===================================================================================
namespace detail {

    /**
     * @brief Container for Lance-Williams coefficients
     */
    struct LanceWilliamsCoefficients {
        double alpha_c = 0.0;
        double alpha_d = 0.0;
        double beta = 0.0;
        double gamma = 0.0;
    };

    /**
     * @brief Pure function to compute Lance-Williams coefficients at
     * compile time.
     */
    template <LinkageMethod SelectedMethod>
    inline LanceWilliamsCoefficients compute_LW_coeffs(double n_c, double n_d, double n_L) {
        LanceWilliamsCoefficients coeffs;

        // Lance Williams coefficients for all supported methods
        // ------------------------------------------------------------
        // Ward's Minimum Variance
        if constexpr (SelectedMethod == LinkageMethod::Ward) {
            double total_n = n_c + n_d + n_L;
            coeffs.alpha_c = (n_c + n_L) / total_n;
            coeffs.alpha_d = (n_d + n_L) / total_n;
            coeffs.beta = -n_L / total_n;
        }
        // UPGMA / Average Linkage
        else if constexpr (SelectedMethod == LinkageMethod::Average) {
            double total_n_cd = n_c + n_d;
            coeffs.alpha_c = n_c / total_n_cd;
            coeffs.alpha_d = n_d / total_n_cd;
        }
        // Complete Linkage
        else if constexpr (SelectedMethod == LinkageMethod::Complete) {
            coeffs.alpha_c = 0.5;
            coeffs.alpha_d = 0.5;
            coeffs.gamma = 0.5;
        }
        // Single Linkage
        else if constexpr (SelectedMethod == LinkageMethod::Single) {
            coeffs.alpha_c = 0.5;
            coeffs.alpha_d = 0.5;
            coeffs.gamma = -0.5;
        }
        // Weighted Average Linkage (WPGMA) / McQuitty
        else if constexpr (SelectedMethod == LinkageMethod::WPGMA) {
            coeffs.alpha_c = 0.5;
            coeffs.alpha_d = 0.5;
        }
        // Centroid Linkage (UPGMC)
        else if constexpr (SelectedMethod == LinkageMethod::Centroid) {
            double total_n_cd = n_c + n_d;
            coeffs.alpha_c = n_c / total_n_cd;
            coeffs.alpha_d = n_d / total_n_cd;
            coeffs.beta = - (n_c * n_d) / (total_n_cd * total_n_cd);
        }
        // Median Linkage (WPGMC)
        else if constexpr (SelectedMethod == LinkageMethod::Median) {
            coeffs.alpha_c = 0.5;
            coeffs.alpha_d = 0.5;
            coeffs.beta = -0.25;
        }
        // Unsupported Linkage Method
        else {
            static_assert(sizeof(SelectedMethod) == 0,
                          "Unsupported Linkage Method.");
        }

        return coeffs;
    }
} /* End of namespace detail */


// ===================================================================================
// Unified Cluster Update Policy
// ===================================================================================

/**
 * @class ClusterUpdatePolicy
 *
 * @brief Manages a Lance-Williams distance matrix for agglomerative clustering.
 * The storage backend is selected at compile time via the StorageMode template
 * parameter.
 *
 * @tparam MatrixType          The type of the input data matrix.
 * @tparam DistancePolicyType  The unified distance policy (e.g., Euclidean, Correlation).
 * @tparam SelectedMethod      The Lance-Williams linkage method to use.
 * @tparam Mode                Storage backend. Default: StorageMode::Full2D.
 */
template <typename MatrixType, typename DistancePolicyType,
          LinkageMethod SelectedMethod, StorageMode Mode = StorageMode::Full2D>
class ClusterUpdatePolicy {
public:
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    // Guard: Centroid and Median require the Full2D matrix (non-reducible methods).
    static_assert(
        Mode == StorageMode::Full2D ||
        (SelectedMethod != LinkageMethod::Centroid &&
         SelectedMethod != LinkageMethod::Median),
        "Condensed1D and LazyHashMap storage modes do not support Centroid or Median linkage.");

private:
    /** @brief Compile-time storage selection. */
    using StorageType = std::conditional_t<
        Mode == StorageMode::Full2D,
        Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>,
        std::conditional_t<
            Mode == StorageMode::Condensed1D,
            std::vector<RealScalar>,
            std::unordered_map<uint64_t, double>
        >
    >;

    /** @brief The distance store. */
    StorageType dist_matrix_;

    /**
     * @brief Secondary index for LazyHashMap: maps each physical slot to the set of
     * other physical slots it has a live entry with in dist_matrix_. Used for O(M)
     * dead-slot pruning in merge_clusters. Empty and unused for other storage modes.
     */
    std::vector<std::unordered_set<Eigen::Index>> slot_neighbors_;

    /** @brief Maps logical cluster ID to physical matrix index. */
    std::vector<Eigen::Index> id_to_idx_;

    /** @brief Maps physical matrix index to logical cluster ID (-1 = dead slot). */
    std::vector<Eigen::Index> idx_to_id_;

    /** @brief Number of original columns. */
    Eigen::Index num_original_;

    /** @brief The chosen distance policy. */
    DistancePolicyType dist_policy_;

    // -------------------------------------------------------------------------------
    // Storage Abstraction Interface
    // -------------------------------------------------------------------------------

    /**
     * @brief Encodes a physical index pair into a single 64-bit key (LazyHashMap only).
     * Both indices must be < 2^32.
     */
    static inline uint64_t make_key(Eigen::Index r, Eigen::Index c) noexcept {
        if (r > c) { std::swap(r, c); }
        return (static_cast<uint64_t>(r) << 32) | static_cast<uint64_t>(c);
    }

    /**
     * @brief Retrieves the distance between two physical slots.
     *
     * For LazyHashMap: on a cache miss the distance is recomputed via dist_policy_
     * and returned WITHOUT caching. Only Lance-Williams post-merge values (written
     * by set_physical_distance) live in the map; original-pair distances are always
     * recomputed on demand since dist_policy_ computes them in O(1) (LSH).
     */
    inline double get_physical_distance(Eigen::Index r, Eigen::Index c) const {
        if constexpr (Mode == StorageMode::Condensed1D) {
            if (r > c) { std::swap(r, c); }
            Eigen::Index idx = (num_original_ * r) - (r * (r + 1) / 2) + (c - r - 1);
            return dist_matrix_[idx];
        } else if constexpr (Mode == StorageMode::Full2D) {
            return dist_matrix_(r, c);
        } else {
            // LazyHashMap: hit → cached LW value; miss → recompute, do NOT cache.
            // Invariant: a cache miss implies both slots are original columns
            // (synthetic-cluster distances are always explicitly written by merge_clusters).
            const uint64_t key = make_key(r, c);
            const auto it = dist_matrix_.find(key);
            if (it != dist_matrix_.end()) { return it->second; }
            return dist_policy_.get_distance(idx_to_id_[r], idx_to_id_[c]);
        }
    }

    /**
     * @brief Sets the distance between two physical slots.
     */
    inline void set_physical_distance(Eigen::Index r, Eigen::Index c, double value) {
        if constexpr (Mode == StorageMode::Condensed1D) {
            if (r > c) { std::swap(r, c); }
            Eigen::Index idx = (num_original_ * r) - (r * (r + 1) / 2) + (c - r - 1);
            dist_matrix_[idx] = value;
        } else if constexpr (Mode == StorageMode::Full2D) {
            dist_matrix_(r, c) = value;
            dist_matrix_(c, r) = value;
        } else {
            // LazyHashMap: write LW distance and maintain slot_neighbors_ index.
            const uint64_t key = make_key(r, c);
            auto [it, inserted] = dist_matrix_.emplace(key, value);
            if (inserted) {
                // New entry: register both endpoints in the secondary index.
                slot_neighbors_[r].insert(c);
                slot_neighbors_[c].insert(r);
            } else {
                // Overwrite existing entry (slot reuse after a merge).
                it->second = value;
            }
        }
    }

public:
    /**
     * @brief Constructor. Initialises index maps and, for non-lazy modes, pre-computes
     * the full pairwise distance matrix.
     *
     * @param data The data matrix (columns are the objects to cluster).
     */
    ClusterUpdatePolicy(const MatrixType& data) : dist_policy_(data) {
        num_original_ = data.cols();
        id_to_idx_.assign(num_original_ * 2 - 1, -1);
        idx_to_id_.assign(num_original_, -1);

        if constexpr (Mode == StorageMode::Condensed1D) {
            const Eigen::Index total_elements = (num_original_ * (num_original_ - 1)) / 2;
            dist_matrix_.assign(total_elements, 0.0);

            std::vector<RealScalar> temp_col(num_original_);
            for (Eigen::Index i = 0; i < num_original_; ++i) {
                id_to_idx_[i] = i;
                idx_to_id_[i] = i;

                if (i > 0) {
                    Eigen::Map<Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>> col_map(
                        temp_col.data(), i);
                    dist_policy_.compute_distances(i, col_map);
                    for (Eigen::Index j = 0; j < i; ++j) {
                        set_physical_distance(i, j, temp_col[j]);
                    }
                }
            }

        } else if constexpr (Mode == StorageMode::Full2D) {
            dist_matrix_.resize(num_original_, num_original_);
            for (Eigen::Index i = 0; i < num_original_; ++i) {
                id_to_idx_[i] = i;
                idx_to_id_[i] = i;
                dist_matrix_(i, i) = 0.0;

                if (i > 0) {
                    Eigen::Map<Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>> col_map(
                        &dist_matrix_(0, i), i);
                    dist_policy_.compute_distances(i, col_map);

                    for (Eigen::Index j = 0; j < i; ++j) {
                        dist_matrix_(i, j) = dist_matrix_(j, i);
                    }
                }
            }

        } else {
            // LazyHashMap: O(P) init only. No distances are computed or stored upfront.
            slot_neighbors_.resize(num_original_);
            for (Eigen::Index i = 0; i < num_original_; ++i) {
                id_to_idx_[i] = i;
                idx_to_id_[i] = i;
            }
        }
    }

    /**
     * @brief Safely retrieves the current distance between two logical clusters.
     *
     * @param c_id Logical cluster ID of the first cluster.
     * @param d_id Logical cluster ID of the second cluster.
     *
     * @return The distance between the two clusters.
     */
    double get_logical_distance(Eigen::Index c_id, Eigen::Index d_id) const {
        const Eigen::Index idx_c = id_to_idx_[c_id];
        const Eigen::Index idx_d = id_to_idx_[d_id];
        if (idx_c == -1 || idx_d == -1) {
            return std::numeric_limits<double>::max();
        }
        return get_physical_distance(idx_c, idx_d);
    }


    /**
     * @brief Scans the distance matrix for the nearest active neighbor.
     *
     * @param c_id The logical cluster ID for which the nearest neighbor is being searched.
     * @param active_clusters Boolean vector of currently active logical IDs.
     * @param current_cluster_id The highest logical ID currently issued.
     * @param min_dist Output parameter for the found minimum distance.
     *
     * @return The logical ID of the nearest neighbor.
     */
    Eigen::Index find_nearest_neighbor(
        Eigen::Index c_id,
        const std::vector<bool>& active_clusters,
        Eigen::Index current_cluster_id,
        double& min_dist
        ) const
    {
        min_dist = std::numeric_limits<double>::max();
        Eigen::Index best_d = -1;
        const Eigen::Index idx_c = id_to_idx_[c_id];

        for (Eigen::Index i = 0; i < current_cluster_id; ++i) {
            if (!active_clusters[i] || i == c_id) { continue; }
            const double d = get_physical_distance(idx_c, id_to_idx_[i]);

            if (d < min_dist) {
                min_dist = d;
                best_d = i;
            }
        }
        return best_d;
    }


    /**
     * @brief Applies Lance-Williams calculus to update distances after a merge.
     *
     * The operation is split into three phases to guarantee correctness across all
     * storage modes, including LazyHashMap:
     *
     *   Phase 1 — Read: Collect dist(C, L) and dist(D, L) for all active L while
     *             idx_to_id_ still maps physical slots to their pre-merge column IDs.
     *             This is critical for LazyHashMap: the lazy fallback in
     *             get_physical_distance calls dist_policy_.get_distance(idx_to_id_[r], ...)
     *             and requires valid original-column IDs at that point.
     *
     *   Phase 2 — Remap: Reassign physical slot idx_c to the new cluster and mark
     *             idx_d as a dead slot. Safe now because phase 1 has cached all
     *             needed values.
     *
     *   Phase 3 — Write: Store the Lance-Williams distances for the new cluster.
     *             After this phase, all active-cluster distances to the new cluster
     *             are explicitly in the store, so future reads are always cache hits.
     *
     * @param args The structured arguments containing cluster IDs for the merge.
     * @param cluster_sizes Vector of current cluster sizes.
     */
    void merge_clusters(
        const MergeClustersArgs& args,
        const std::vector<Eigen::Index>& cluster_sizes)
    {
        const Eigen::Index idx_c = id_to_idx_[args.c_id];
        const Eigen::Index idx_d = id_to_idx_[args.d_id];

        const double n_c = static_cast<double>(cluster_sizes[args.c_id]);
        const double n_d = static_cast<double>(cluster_sizes[args.d_id]);
        const double dist_cd = get_physical_distance(idx_c, idx_d);

        // Phase 1: Read all required distances before remapping slot ownership.
        std::vector<Eigen::Index> active_k;
        std::vector<double> dist_cL, dist_dL;
        active_k.reserve(num_original_);

        for (Eigen::Index k = 0; k < num_original_; ++k) {
            if (k == idx_c || k == idx_d || idx_to_id_[k] == -1) { continue; }
            active_k.push_back(k);
            dist_cL.push_back(get_physical_distance(idx_c, k));
            dist_dL.push_back(get_physical_distance(idx_d, k));
        }

        // Phase 2: Remap — slot idx_c now belongs to the new cluster; idx_d is freed.
        id_to_idx_[args.new_id] = idx_c;
        idx_to_id_[idx_c] = args.new_id;
        idx_to_id_[idx_d] = -1;

        // Phase 2b (LazyHashMap only): Prune all dead entries for the freed slot idx_d.
        // Each live entry (idx_d, s) in dist_matrix_ is erased and removed from both
        // endpoints' slot_neighbors_ lists. This keeps the map bounded to O(active pairs)
        // rather than accumulating O(P²) dead entries over the full merge sequence.
        if constexpr (Mode == StorageMode::LazyHashMap) {
            for (const Eigen::Index s : slot_neighbors_[idx_d]) {
                dist_matrix_.erase(make_key(idx_d, s));
                slot_neighbors_[s].erase(idx_d);
            }
            slot_neighbors_[idx_d].clear();
        }

        // Phase 3: Write Lance-Williams updated distances for the new cluster.
        for (std::size_t i = 0; i < active_k.size(); ++i) {
            const Eigen::Index k = active_k[i];
            const Eigen::Index L = idx_to_id_[k];
            const double n_L = static_cast<double>(cluster_sizes[L]);

            const auto coeffs = detail::compute_LW_coeffs<SelectedMethod>(n_c, n_d, n_L);

            double new_dist = coeffs.alpha_c * dist_cL[i] +
                              coeffs.alpha_d * dist_dL[i] +
                              coeffs.beta * dist_cd +
                              coeffs.gamma * std::abs(dist_cL[i] - dist_dL[i]);

            if (new_dist < 0.0) { new_dist = 0.0; }

            set_physical_distance(idx_c, k, new_dist);
        }
    }
};

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */

#endif /* ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_CLUSTER_UPDATE_POLICY_HPP */
