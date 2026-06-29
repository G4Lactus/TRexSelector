// ===================================================================================
// distance_policy.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DISTANCE_POLICY_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DISTANCE_POLICY_HPP
// ===================================================================================
/**
 * @file distance_policy.hpp
 * @brief Setup of unified distance policies for hierarchical clustering algorithms.
 *
 * @details Uses compile-time evaluation (if constexpr) to select the metric, enabling
 * zero-overhead abstraction for distance computations inside deep parallel loops.
 *
 * |- src/ml_methods/clustering/hierarchical/agglomerative
 * |- Distance_Policy.hpp
 *
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <random>

// Eigen includes
#include <Eigen/Dense>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

/**
 * @brief Available distance metrics for clustering.
 *
 * @details
 * Local Sensitivity Hashing (LSH):
 *  - Latent Factor / Dense Block Data: If your variables move together based on
 * shared latent signals (forming "dense balls" in high-dimensional space),
 * `Correlation_LSH_Approx` provides massive O(1) speedups (up to 250x) while
 * maintaining near-perfect accuracy (ARI ~ 1.0) for SLINK, UPGMA, and Complete Linkage.
 *
 * - AR(1) / Chain-like Data: If your variables represent spatial or temporal sequences
 * where correlation decays over distance (forming continuous "strings"),
 * DO NOT USE LSH. The random hyperplanes will slice the continuous gradient,
 * creating quantization noise that completely shatters the clusters (ARI ~ 0.0).
 * Instead use exact `Correlation` combined with Single Linkage (SLINK).
 *
 * Ward's Method: LSH metrics here are based on Cosine/SimHash. They cannot be used
 * with Ward's Minimum Variance, which strictly requires Euclidean space.
 */
enum class DistanceMetric {
    Euclidean,       /* Exact: ||x - y||^2, squared Euclidean distance. Required for Ward,
                        Centroid, Median. (Equivalent to SciPy's metric='sqeuclidean') */
    Correlation,     /* Exact: 1 - |corr(x_i, x_j)|, with corr normalized by column norms.
                        Requires centered data; scale-invariant (works for unit-L2 OR z-scored
                        columns). Safe for all topologies.*/
    Manhattan,       /* Exact L1 norm ||x - y||_1. Manhattan distance is sum of absolute
                        differences. */
    Correlation_LSH_Filter, /* Locality-Sensitive Hashing: approximation for correlation distance,
                               as Gatekeeper (exact dot product) very fast for high-dimensional
                               data. */
    Correlation_LSH_Approx, /* SimHash: O(1) Angular Approximation for correlation distance. */

    // more metrics...
};


/**
 * @class DistancePolicy
 * @brief Computes vectorized distances between columns of data matrix.
 *
 * @details For Euclidean distances, it precomputes 64-bit SimHash signatures via random hyperplanes
 *  in O(N * P) time, reducing pairwise query time from O(N) to O(1).
 *
 * @tparam MatrixType The type of the input data matrix (e.g., Eigen::Map)
 * @tparam SelectedMetric Distance measure to compute.
 */
template<typename MatrixType, DistanceMetric SelectedMetric>
class DistancePolicy {
public:
     /** @brief Scalar type of the input data, support for real and complex numbers. */
    using Scalar = typename MatrixType::Scalar;

    /** @brief Real scalar type for distances. */
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    /** @brief Selected distance metric exported for rebinding. */
    static constexpr DistanceMetric Metric = SelectedMetric;

private:
    /** @brief Reference to the input data matrix */
    const MatrixType& data_;

    /** @brief Cache for Euclidean squared norms to prevent RAM trashing.
    * Stores a single 1D vector of the squared norms.
    */
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> sq_norms_;

    /** @brief Cache for LSH 64-bit signatures */
    std::vector<uint64_t> lsh_signatures_;

    /** @brief Elevated to class member to allow dynamic LSH updates */
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 64> random_planes_;

public:

    /**
     * @brief Constructor with dynamic precomputation limit.
     *
     * @param data Reference to the data matrix.
     * @param precompute_limit Instructs the policy to only precalculate the first N columns.
     *        If -1, it precalculates all columns (used by SLINK).
     */
    DistancePolicy(const MatrixType& data, Eigen::Index precompute_limit = -1) : data_(data) {
        Eigen::Index n = data_.rows();
        Eigen::Index p = data_.cols();
        Eigen::Index limit = (precompute_limit == -1) ? p : precompute_limit;

        // Precompute squared column norms for metrics that need them.
        // Correlation normalizes the dot product by these norms (true Pearson),
        // making the distance scale-invariant; Correlation_LSH_Filter reuses them
        // for its exact recheck after the Hamming gate.
        if constexpr (SelectedMetric == DistanceMetric::Euclidean ||
                      SelectedMetric == DistanceMetric::Correlation ||
                      SelectedMetric == DistanceMetric::Correlation_LSH_Filter) {
            sq_norms_.resize(p);
            #pragma omp parallel for schedule(static)
            for (Eigen::Index j = 0; j < limit; ++j) {
                sq_norms_(j) = data_.col(j).squaredNorm();
            }
        }

        if constexpr (SelectedMetric == DistanceMetric::Correlation_LSH_Filter ||
                      SelectedMetric == DistanceMetric::Correlation_LSH_Approx) {
            lsh_signatures_.resize(p, 0);
            random_planes_.resize(n, 64);
            std::mt19937 rng(42);
            std::normal_distribution<double> dist(0.0, 1.0);
            for (Eigen::Index i = 0; i < n; ++i) {
                for (Eigen::Index b = 0; b < 64; ++b) {
                    random_planes_(i, b) = dist(rng);
                }
            }
            #pragma omp parallel for schedule(static)
            for (Eigen::Index j = 0; j < limit; ++j) {
                update_node(j);
            }
        }
    }

    /**
     * @brief Dynamically updates the precomputed caches for a specific column.
     * Essential for O(N*D) geometric clustering when a new centroid is synthesized.
     *
     * @note Real valued only supported, otherwise complex projections needed.
     *
     * @param i The index of the column to update.
     */
    void update_node(Eigen::Index i) {
        if constexpr (SelectedMetric == DistanceMetric::Euclidean ||
                      SelectedMetric == DistanceMetric::Correlation ||
                      SelectedMetric == DistanceMetric::Correlation_LSH_Filter) {
            sq_norms_(i) = data_.col(i).squaredNorm();
        }

        if constexpr (SelectedMetric == DistanceMetric::Correlation_LSH_Filter ||
                      SelectedMetric == DistanceMetric::Correlation_LSH_Approx) {
            uint64_t signature = 0;
            Eigen::Matrix<RealScalar, 1, 64> projections =
                data_.col(i).transpose() * random_planes_;

            for (int b = 0; b < 64; ++b) {
                if (projections(b) > 0.0) {
                    signature |= (1ULL << b);
                }
            }
            lsh_signatures_[i] = signature;
        }
    }

    /**
     * @brief Computes distances from a given column to all previous columns.
     *
     * @tparam OutputVectorType The type of the output vector.
     *
     * @param i The index of the column for which distances are computed.
     * @param out_distances The output vector to store the computed distances.
     */
    template <typename OutputVectorType>
    void compute_distances(Eigen::Index i, OutputVectorType& out_distances) const {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < i; ++j) {
            out_distances(j) = get_distance(i, j);
        }
    }


    /**
     * @brief Get the distance object.
     *
     * @param i The index of the first column.
     * @param j The index of the second column.
     *
     * @return The computed distance between the two columns.
     */
    double get_distance(Eigen::Index i, Eigen::Index j) const {
        if (i == j) { return 0.0; }

        // Squared Euclidean distance using the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
        if constexpr (SelectedMetric == DistanceMetric::Euclidean) {
            RealScalar dot_val = data_.col(i).dot(data_.col(j));
            RealScalar dist = sq_norms_(i) + sq_norms_(j) - RealScalar(2.0) * dot_val;
            return dist > 0.0 ? dist : 0.0;
        }

        // Exact Correlation distance (1 - |corr|). Normalizes the dot product by the
        // column norms => true Pearson correlation on centered data, independent of
        // the column scale (unit-L2 or z-scored). A zero-norm (constant) column is
        // treated as maximally dissimilar.
        else if constexpr (SelectedMetric == DistanceMetric::Correlation) {
            const RealScalar denom = std::sqrt(sq_norms_(i) * sq_norms_(j));
            if (denom <= RealScalar(0.0)) { return RealScalar(1.0); }
            const Scalar dot_val = data_.col(i).dot(data_.col(j));
            const RealScalar corr =
                std::min(std::abs(dot_val) / denom, RealScalar(1.0));
            return RealScalar(1.0) - corr;
        }

        // Correlation LSH Gatekeeper Filter with fixed Hamming threshold
        else if constexpr (SelectedMetric == DistanceMetric::Correlation_LSH_Filter) {
            const int HAMMING_THRESHOLD = 14;
            int bit_diff = __builtin_popcountll(lsh_signatures_[i] ^ lsh_signatures_[j]);
            if (bit_diff <= HAMMING_THRESHOLD) {
                const RealScalar denom = std::sqrt(sq_norms_(i) * sq_norms_(j));
                if (denom <= RealScalar(0.0)) { return RealScalar(1.0); }
                const Scalar dot_val = data_.col(i).dot(data_.col(j));
                const RealScalar corr =
                    std::min(std::abs(dot_val) / denom, RealScalar(1.0));
                return RealScalar(1.0) - corr;
            } else {
                return 1.0;
            }
        }

        // Correlation LSH Approximation
        else if constexpr (SelectedMetric == DistanceMetric::Correlation_LSH_Approx) {
            int bit_diff = __builtin_popcountll(lsh_signatures_[i] ^ lsh_signatures_[j]);
            double theta = M_PI * (static_cast<double>(bit_diff) / 64.0);
            double approx_corr = std::cos(theta);
            return 1.0 - std::abs(approx_corr);
        }

        // Manhattan distance
        else if constexpr (SelectedMetric == DistanceMetric::Manhattan) {
            return (data_.col(i) - data_.col(j)).cwiseAbs().sum();
        }

        // Distance not supported
        else {
            static_assert(sizeof(MatrixType) == 0, "Distance metric not implemented.");
        }
    }

}; /* End of class DistancePolicy */

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */

#endif /* End of ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DISTANCE_POLICY_HPP */
