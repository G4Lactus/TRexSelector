// ===================================================================================
// slink_core.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_SLINK_CORE_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_SLINK_CORE_HPP
// ===================================================================================
/**
 * @brief Class implementation for the core SLINK algorithm to compute the pointer
 * representation for single-linkage hierarchical clustering.
 *
 * @details The SLINK algorithm is an efficient method for computing the single-linkage
 * hierarchical clustering dendrogram.
 *
 * The algorithm achieves the theoretical order-of-magnitude bounds for both
 * compactness of storage O(N) and speed of operation O(N^2).
 *
 * References:
 * -----------
 * Sibson, R., "SLINK: An optimally efficient algorithm for the
 *   single-link cluster method", The Computer Journal, 16(1), pp.30-34, 1973.
 * Müllner, D., "Modern hierarchical, agglomerative clustering algorithms",
 *   arXiv preprint arXiv:1109.2378v1, pp.1-29, 2011.
 *
 * Organization:
 * -------------
 * |- src/ml_methods/clustering/hierarchical/agglomerative/
 * |- SLINK_core.hpp
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// ===================================================================================

// Embedded into namespace trex::ml_methods::clustering::hierarchical::agglomerative
namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

/**
 * @brief A tie-breaking distance wrapper as proposed by Daniel Müllner in
 *  "Modern hierarchical, agglomerative clustering algorithms" (2011).
 *  If two distances are mathematically identical, it breaks the tie using lexicographic
 *  order of their node indices.
 *  We include his proposed fix according to Section 5 to make the SLINK algorithm
 *  fully tie-aware.
 */
struct TieAwareDistance {
    /** @brief Distance between two nodes u and v */
    double distance;
    /** @brief Index of the first node */
    Eigen::Index u;
    /** @brief Index of the second node */
    Eigen::Index v;

    // Operator overloads for tie-aware comparisons
    /** @brief Lexicographic tie-breaking operator overloading */
    bool operator<(const TieAwareDistance& other) const {
        if (distance != other.distance) { return distance < other.distance; }

        // Tie-breaking by lexicographic ordering of node indices
        Eigen::Index this_min = std::min(u, v);
        Eigen::Index other_min = std::min(other.u, other.v);
        if (this_min != other_min) {return this_min < other_min; }

        return std::max(u, v) < std::max(other.u, other.v);
    }

    /** @brief Lexicographic tie-breaking operator overloading */
    bool operator>=(const TieAwareDistance& other) const {
        return !(*this < other);
    }

    /** @brief Lexicographic tie-breaking operator overloading */
    bool operator<=(const TieAwareDistance& other) const {
        return !(other < *this);
    }
};


/**
 * @brief SLINK:
 * Core engine for single-linkage hierarchical clustering using the SLINK algorithm.
 *
 * @details
 * To handle massive datasets (e.g., p >> n scenarios) efficiently, it is recommended to use
 * `Eigen::Map` (e.g., `Eigen::Map<const Eigen::MatrixXf>`).
 * The underlying `Scalar` type seamlessly supports both real numbers (e.g., `float`, `double`)
 * and complex numbers (e.g., `std::complex<float>`).
 *
 * @tparam MatrixType The type of input data matrix. To handle massive datasets
 * efficiently without triggering deep copies, use Eigen::Map.
 * @tparam DistancePolicy The policy class determining how distances are computed
 * (e.g., Correlation, Euclidean). The requirement is that the method is implemented.
 */
template <typename MatrixType, typename DistancePolicy>
class SLINK {
public:
    /** @brief Original scalar type of the input matrix (real or complex). */
    using Scalar = typename MatrixType::Scalar;

    /** @brief Real scalar type for distances. */
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

private:

    /** @brief Distance policy functor. */
    DistancePolicy distance_policy_;

    /** @brief Number of objects to cluster. */
    Eigen::Index num_objects_;

    // SLINK state representation
    // --------------------------------------
    /** @brief Pointer to cluster representative. */
    std::vector<Eigen::Index> pi_;

    /** @brief Distance to cluster representative must be tie-aware. */
    std::vector<TieAwareDistance> lambda_;

    /** @brief Temporary buffer for distances to maintain Eigen's SIMD vectorization. */
    std::vector<RealScalar> M_;

public:

    /**
     * @brief Constructor for SLINK class.
     *
     * @param data Eigen matrix or map containing the variables to cluster.
     *
     * @note Clustering is done over columns (variables). Expected data shape is
     * (n_samples x n_features).
     */
    SLINK(const MatrixType& data)
        : distance_policy_(data), num_objects_(data.cols())
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
     * @brief Execute the SLINK algorithm to build the pointer representation for
     * single-linkage hierarchical clustering.
     */
    void cluster() {
        // Initialize the first object (0-based indexing)
        // Step 1: Pi(1) = 1, Lambda(1) = infinity
        // The sentinel is double max (lambda distances are stored as double) so that
        // downstream consumers (pointer_to_merge_matrix) can test against
        // numeric_limits<double>::max() regardless of the input RealScalar type.
        const double INF = std::numeric_limits<double>::max();
        pi_[0] = 0;
        lambda_[0] = {INF, 0, 0};

        // Temporary array for the tie-aware structs during the loop
        std::vector<TieAwareDistance> M_tie(num_objects_);

        // Main SLINK loop
        // -------------------------------------
        // Index `i` represents the new object being added to the dendrogram
        for (Eigen::Index i = 1; i < num_objects_; ++i) {

            // Step 1: Initialize new node
            pi_[i] = i;
            lambda_[i] = {INF, i, i};

            // Step 2: Compute distances M(j) = d(j, i) for j = 0, ..., i-1
            // Map the first `i` elements of std::vector M_ to an Eigen vector.
            // This allows the policy to vectorize the computation directly into the buffer.
            Eigen::Map<Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>> M_map(M_.data(), i);
            distance_policy_.compute_distances(i, M_map);

            // Tie-aware wrapping
            for (Eigen::Index j = 0; j < i; ++j) {
                M_tie[j] = {static_cast<double>(M_[j]), j, i};
            }

            // Step 3: Sibson's pointer update with tie awareness
            for (Eigen::Index j = 0; j < i; ++j) {
                if (lambda_[j] >= M_tie[j]) {
                    // M_tie[pi_[j]] = min(M_tie[pi_[j]], lambda_[j])
                    if (lambda_[j] < M_tie[pi_[j]]) { M_tie[pi_[j]] = lambda_[j]; }
                    lambda_[j] = M_tie[j];
                    pi_[j] = i;
                } else {
                    // M_tie[pi_[j]] = min(M_tie[pi_[j]], M_tie[j])
                    if (M_tie[j] < M_tie[pi_[j]]) { M_tie[pi_[j]] = M_tie[j]; }
                }
            }

            // Step 4: Finalize pointers for this step
            for (Eigen::Index j = 0; j < i; ++j) {
                if (lambda_[j] >= lambda_[pi_[j]]) {
                    pi_[j] = i;
                }
            }
        }
    }

    // Accessors for final pointer representation
    // -------------------------------------------
    /** @brief Pointer to cluster representative for each object. */
    const std::vector<Eigen::Index>& get_pi() const { return pi_; }

    /** @brief Distance to cluster representative for each object. */
    std::vector<double> get_lambda() const {
        std::vector<double> raw_lambda(num_objects_);
        for (Eigen::Index i = 0; i < num_objects_; ++i) {
            raw_lambda[i] = lambda_[i].distance;
        }
        return raw_lambda;
    }
};

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative  */

#endif /* End of ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_SLINK_CORE_HPP */
