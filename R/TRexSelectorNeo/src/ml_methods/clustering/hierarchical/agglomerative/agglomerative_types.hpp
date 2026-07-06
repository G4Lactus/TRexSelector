// ===================================================================================
// agglomerative_types.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_TYPES_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_TYPES_HPP
// ===================================================================================
/**
 * @file agglomerative_types.hpp
 *
 * @brief Type definitions and structures for agglomerative hierarchical clustering.
 *
 * @details Contains currently:
 * - LinkageMethod enum
 * - MergeStep struct
 * - MergeClustersArgs struct
 *
 * Organization:
 * -------------
 * |- src/ml_methods/clustering/hierarchical/agglomerative
 * |- Agglomerative_Types.hpp
 */
// ===================================================================================

// Eigen includes
#include <Eigen/Dense>

// ===================================================================================

// Embedded into namespace trex::ml_methods::clustering::hierarchical::agglomerative
namespace trex::ml_methods::clustering::hierarchical::agglomerative {

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
 * @brief Struct to record each merge step in the dendrogram construction.
 *
 * @details This standard stepwise output is utilized by NN-Chain, Generic Linkage,
 * and MST-based linkage methods.
 */
struct MergeStep {
    /** @brief Cluster 1 id. */
    Eigen::Index cluster1;

    /** @brief Cluster 2 id. */
    Eigen::Index cluster2;

    /** @brief Distance at which the merge occurs. */
    double distance;

    /** @brief Size of the new cluster formed by merging cluster1 and cluster2. */
    Eigen::Index new_cluster_size;
};


/** @brief Arguments for merge_clusters to prevent wrong usage. */
struct MergeClustersArgs {
    /** @brief Logical cluster ID of the first cluster to merge. */
    Eigen::Index c_id;

    /** @brief Logical cluster ID of the second cluster to merge. */
    Eigen::Index d_id;

    /** @brief Logical cluster ID of the newly formed cluster. */
    Eigen::Index new_id;
};

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */

#endif /* End of ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_TYPES_HPP */
