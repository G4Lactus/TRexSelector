// ===================================================================================
// clink_core.hpp
// ===================================================================================
/**
 * @brief CLINK_core.hpp
 * Class implementation for the core CLINK algorithm to compute
 * the pointer representation for complete-linkage hierarchical clustering.
 *
 * @details The CLINK algorithm is an efficient method for computing the complete-linkage
 * hierarchical clustering dendrogram.
 * The algorithm achieves the theoretical order-of-magnitude bounds for both compactness
 * of storage O(N) and speed of operation O(N^2).
 *
 * Reference: Defays, D., "CLINK: An efficient algorithm for a complete link method",
 * The Computer Journal, 20(4), pp.364-366, 1977.
 *
 * |- src/ml_methods/clustering/dendrograms
 *    |- CLINK_core.hpp
 */
// ===================================================================================

#ifndef TREX_ML_METHODS_CLUSTERING_DENDROGRAMS_CLINK_CORE_HPP
#define TREX_ML_METHODS_CLUSTERING_DENDROGRAMS_CLINK_CORE_HPP

// ===================================================================================

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace clustering {
namespace dendrograms {

// ===================================================================================

/**
 * @brief CLINK:
 * Core engine for Complete-Linkage hierarchical clustering using the CLINK algorithm.
 *
 * @details
 * To handle massive datasets (e.g., p >> n scenarios) efficiently, it is recommended to use
 * `Eigen::Map` (e.g., `Eigen::Map<const Eigen::MatrixXf>`).
 * The underlying `Scalar` type seamlessly supports both real numbers  (e.g., `float`, `double`)
 * and complex numbers (e.g., `std::complex<float>`, `std::complex<double>`).
 *
 * @tparam MatrixType The type of input data matrix. To handle massive datasets
 * efficiently without triggering deep copies, use Eigen::Map.
 * @tparam DistancePolicy The policy class determining how distances are computed
     * (e.g., Correlation, Euclidean). The policy must expose a `compute_distances` method.
 */
template <typename MatrixType, typename DistancePolicy>
class CLINK {
public:
    /** @brief Original scalar type of the input matrix (real or complex) */
    using Scalar = typename MatrixType::Scalar;

    /** @brief Distances: real numbers */
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

private:
    /** @brief Input data matrix (reference to avoid copies of Eigen objects) */
    const MatrixType& data_;

    /** @brief Distance policy functor */
    DistancePolicy distance_policy_;

    /** @brief Number of objects (columns) to cluster */
    Eigen::Index num_objects_;

    // CLINK state representation
    // --------------------------------------
    /** @brief Pointer to cluster representative */
    std::vector<Eigen::Index> pi_;

    /** @brief Distance to cluster representative */
    std::vector<RealScalar> lambda_;

    /** @brief Temporary buffer for distances */
    std::vector<RealScalar> M_;

public:

    /**
     * @brief Constructor
     *
     * @param data Eigen matrix or map containing the variables to cluster.
     *
     * @note Clustering is done over columns (variables). Expected data shape is
     * (n_samples x n_features).
     */
    CLINK(const MatrixType& data)
        : data_(data),
        distance_policy_(data),
        num_objects_(data.cols())
    {
        if (num_objects_ < 2) {
            throw std::invalid_argument(
                "Matrix must have at least 2 columns to perform clustering."
            );
        }

        pi_.resize(num_objects_);
        lambda_.resize(num_objects_);
        M_.resize(num_objects_);
    }

    /**
     * @brief Execute the CLINK algorithm to build the pointer representation for
     * complete-linkage hierarchical clustering.
     */
    void cluster() {
        const RealScalar INF = std::numeric_limits<RealScalar>::max();

        // Initialize the first object (0-based indexing)
        pi_[0] = 0;
        lambda_[0] = INF;

        // Main CLINK loop
        // -------------------------------------
        // `i` represents the new object being added to the dendrogram
        for (Eigen::Index i = 1; i < num_objects_; ++i) {

            // Step 1: Initialize new node
            pi_[i] = i;
            lambda_[i] = INF;

            // Step 2: Compute distances M(j) = d(j, i) for j = 0, ..., i-1
            Eigen::Map<Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>> M_map(M_.data(), i);
            distance_policy_.compute_distances(i, M_map);

            // Step 3: Upward propagation of maxima
            for (Eigen::Index j = 0; j < i; ++j) {
                if (lambda_[j] < M_[j]) {
                    Eigen::Index parent = pi_[j];
                    M_[parent] = std::max(M_[parent], M_[j]);
                    M_[j] = INF;
                }
            }

            // Steps 4 & 5: Find insertion point 'a' by scanning backwards
            Eigen::Index a = i - 1;
            for (Eigen::Index j = i - 1; j >= 0; --j) {
                Eigen::Index parent = pi_[j];
                if (lambda_[j] >= M_[parent]) {
                    if (M_[j] < M_[a]) {
                        a = j;
                    }
                } else {
                    M_[j] = INF;
                }
            }

            // Step 6: Begin chain reconstruction
            Eigen::Index b = pi_[a];
            RealScalar c = lambda_[a];
            pi_[a] = i;
            lambda_[a] = M_[a];

            // Step 7: Rewire the branch to maintain the ultrametric inequality
            if (a < i - 1) {
                while (b < i - 1) {
                    Eigen::Index d = pi_[b];
                    RealScalar e = lambda_[b];

                    pi_[b] = i;
                    lambda_[b] = c;

                    b = d;
                    c = e;
                }
                // Handle the b == (i - 1) case (root of existing subtree)
                if (b == i - 1) {
                    pi_[b] = i;
                    lambda_[b] = c;
                }
            }

            // Step 8: Final pointer cleanup
            for (Eigen::Index j = 0; j < i; ++j) {
                Eigen::Index parent = pi_[j];
                // Check if the parent merges directly into the new node
                if (pi_[parent] == i) {
                    if (lambda_[j] >= lambda_[parent]) {
                        pi_[j] = i;
                    }
                }
            }
        }
    }

    // Accessors for final pointer representation
    const std::vector<Eigen::Index>& get_pi() const { return pi_; }
    const std::vector<RealScalar>& get_lambda() const { return lambda_; }
};

// ===================================================================================

} /* End of namespace dendrograms */
} /* End of namespace clustering */
} /* End of namespace ml_methods */
} /* End of namespace trex */

#endif /* End of CLINK_CORE_HPP */
