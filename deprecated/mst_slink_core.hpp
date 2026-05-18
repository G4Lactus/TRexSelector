// ===================================================================================
// mst_slink_core.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_MST_SLINK_CORE_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_MST_SLINK_CORE_HPP
// ===================================================================================
/**
 * @brief Class implementation for Rohlf's MST-based algorithm for single-linkage
 * hierarchical clustering.
 *
 * @details This is based on Gower and Ross's observation that Single Linkage
 * clustering can be derived from a Minimum Spanning Tree (MST). This uses Rohlf's
 * algorithm (a simplified Prim's algorithm).
 * * Note: While theoretically O(N^2) time and O(N) space, modern benchmarks show
 * this approach suffers from branch prediction penalties and L3 cache misses compared
 * to Sibson's strictly linear SLINK algorithm. It is preserved here as a reference
 * implementation.
 *
 * References:
 * ------------
 * Rohlf, F. J., Classification Pattern Recognition and Reduction of Dimensionality,
 *  vol. 2, Handbook of Statistics, pp. 267-284, Elsevier, 1982.
 * Müllner, D., "Modern hierarchical, agglomerative clustering algorithms",
 *   arXiv preprint arXiv:1109.2378v1, pp.1-29, 2011.
 *
 * Organization:
 * -------------
 * |- src/ml_methods/clustering/hierarchical/agglomerative/
 * |- MST_SLINK_core.hpp
 */
// ===================================================================================

#include <vector>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace clustering {
namespace hierarchical {
namespace agglomerative {

// ===================================================================================

/**
 * @brief Class implementation for Rohlf's MST-based algorithm for single-linkage.
 *
 * @tparam MatrixType Type of input data matrix. To handle massive datasets efficiently without
 * triggering deep copies, use Eigen::Map.
 * @tparam DistancePolicy The policy class determining how distances are computed
 * (e.g., Correlation, Euclidean). The requirement is that the method is implemented.
 */
template <typename MatrixType, typename DistancePolicy>
class MST_SLINK {
private:
    /** @brief Original scalar type of the input matrix (real or complex). */
    using Scalar = typename MatrixType::Scalar;
    /** @brief Real scalar type for distances. */
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

private:

    /** @brief Distance policy functor. */
    DistancePolicy dist_policy_;

    /** @brief Number of objects to cluster. */
    Eigen::Index num_objects_;

    /** @brief Vector to store merge steps. */
    std::vector<MergeStep> merges_;

    // State representation and O(N) space tracking arrays
    /** @brief Distance array for the MST construction. */
    std::vector<double> dist_array_;

    /** @brief Active cluster tracking for the MST construction. */
    std::vector<bool> active_;

public:
    /** @brief Constructor for MST_SLINK class.
     *
     * @param data Eigen::matrix or map containing the variables to cluster.
     */
    MST_SLINK(const MatrixType& data)
        : dist_policy_(data), num_objects_(data.cols())
    {
        if (num_objects_ < 2) {
            throw std::invalid_argument(
                "Matrix must have at least 2 columns to perform clustering."
            );
        }

        merges_.reserve(num_objects_ - 1);
        dist_array_.assign(num_objects_, std::numeric_limits<double>::max());
        active_.assign(num_objects_, true);
    }


    /**
     * @brief Execute the MST-based SLINK algorithm to build the merge steps for single-linkage.
     */
    void cluster() {
        // Start from arbitrary node 0
        Eigen::Index c = 0;
        active_[c] = false;

        // Main loop
        // ---------
        for (Eigen::Index i = 1; i < num_objects_; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            Eigen::Index n = -1;

            // Update distances and find the nearest neighbor to the growing MST
            // Optimized for branch prediction stability: iterate over all nodes and check actives
            for (Eigen::Index s = 0; s < num_objects_; ++s) {
                if (active_[s]) {
                    double dist = dist_policy_.get_distance(c, s);
                    // dist_array_[s] holds the shortest distance from 's' to any node in the MST
                    if (dist < dist_array_[s]) {
                        dist_array_[s] = dist;
                    }
                    if (dist_array_[s] < min_dist) {
                        min_dist = dist_array_[s];
                        n = s;
                    }
                }
            }

            // Append the step
            // ----------------
            // Size is 0 because UnionFind will calculate true sizes later
            merges_.push_back({c, n, min_dist, 0});

            // The nearest node 'n' becomes the new current node 'c'
            c = n;
            active_[c] = false;
        }
    }

    /** @brief Getter for the merge steps after clustering. */
    const std::vector<MergeStep>& get_merges() const { return merges_; }
};

} /* End of namespace agglomerative */
} /* End of namespace hierarchical */
} /* End of namespace clustering */
} /* End of namespace ml_methods */
} /* End of namespace trex */

#endif /* ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_MST_SLINK_CORE_HPP */
