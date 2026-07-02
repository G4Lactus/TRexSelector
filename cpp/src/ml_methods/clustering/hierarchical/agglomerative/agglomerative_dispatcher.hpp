// ===================================================================================
// agglomerative_dispatcher.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DISPATCHER_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DISPATCHER_HPP
// ===================================================================================
/**
 * @file agglomerative_dispatcher.hpp
 *
 * @brief The unified routing engine for hierarchical clustering.
 *
 * @details This dispatcher preserves a clean, unified API for the user while
 * secretly routing the execution to mathematically specialized algorithms and
 * memory policies under the hood. It prevents users from accidentally triggering
 * O(N^3) meltdowns or memory limits by enforcing the correct algorithmic pairings.
 *
 * Organization:
 * -------------
 * |- src/ml_methods/clustering/hierarchical/agglomerative
 * |- Agglomerative_Dispatcher.hpp
 */
// ===================================================================================

// std includes
#include <vector>
#include <stdexcept>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/slink_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/nnchain_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/generic_linkage_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/geometric_update_policy.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/block_tiled_matrix_policy.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/proj_geom_updates.hpp>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

class AgglomerativeClustering {
public:
    /**
     * @brief Executes hierarchical clustering, automatically dispatching to the
     * mathematically optimal engine.
     *
     * @note SciPy Equivalence:
     * - Metrics: `DistanceMetric::Euclidean` calculates squared distances (equivalent
     *   to SciPy's `metric='sqeuclidean'`).
     * - Inversions: Standard SciPy preserves non-monotonic inversions for methods like
     *   Centroid and Median. This framework currently enforces strict distance
     *   monotonicity globally, which may reorder rows compared to SciPy's chronological
     *   output for those specific non-reducible methods.
     *
     * @tparam MatrixType          The type of the input data matrix.
     * @tparam DistancePolicyType  The distance metric policy.
     * @tparam Method              The linkage method requested by the user.
     *
     * @param data The data matrix (columns are objects to cluster).
     * @return std::vector<MergeStep> The standard, chronologically sorted merge matrix.
     */
    template <typename MatrixType, typename DistancePolicyType, LinkageMethod Method>
    static std::vector<MergeStep> cluster(const MatrixType& data, bool use_mmap = false,
                                          bool verbose = true) {

        Eigen::Index num_objs = data.cols();

        // -----------------------------------------------------------------------
        // PATH 1: SINGLE LINKAGE (SLINK Engine, O(N) Space)
        // -----------------------------------------------------------------------
        if constexpr (Method == LinkageMethod::Single) {
            SLINK<MatrixType, DistancePolicyType> slink(data);
            slink.cluster();

            // Note: SLINK returns a pointer representation, needs Kruskal's conversion
            return DendrogramUtils::pointer_to_merge_matrix(slink.get_pi(),
                                                            slink.get_lambda());
        }

        // -----------------------------------------------------------------------
        // PATH 2: UPGMA & WARD
        // -----------------------------------------------------------------------
        else if constexpr (Method == LinkageMethod::Average || Method == LinkageMethod::Ward) {

            if constexpr (DistancePolicyType::Metric == DistanceMetric::Correlation_LSH_Approx
                          && Method == LinkageMethod::Ward) {
                // Ward + LSH_Approx ONLY: Ward distance is a function of centroids, so the
                // geometric centroid update in 64D SimHash space is mathematically valid.
                // Do NOT add Average here — the geometric centroid update does NOT implement
                // the Lance-Williams UPGMA formula and violates reducibility, causing NNChain
                // cycles for non-perfectly-separated data.
                using Policy = ProjectedGeometricUpdatePolicy<MatrixType, DistancePolicyType, Method>;
                NNChain<MatrixType, Policy> nnchain(data, use_mmap, verbose);
                nnchain.cluster();
                return DendrogramUtils::format_nnchain_merges(nnchain.get_merges(), num_objs);
            }
            else {
                // Average (all metrics), or Ward with non-LSH_Approx metrics:
                // use the exact Lance-Williams Matrix Engine.
                using Policy = BlockTiledMatrixPolicy<MatrixType, DistancePolicyType, Method>;
                NNChain<MatrixType, Policy> nnchain(data, use_mmap, verbose);
                nnchain.cluster();
                return DendrogramUtils::format_nnchain_merges(nnchain.get_merges(), num_objs);
            }
        }

        // -----------------------------------------------------------------------
        // PATH 3: LANCE-WILLIAMS MATRIX METHODS (Disk-Bound NVMe, O(N^2) Space)
        // -----------------------------------------------------------------------
        else if constexpr (Method == LinkageMethod::Complete ||
                           Method == LinkageMethod::WPGMA ||
                           Method == LinkageMethod::Average ||
                           Method == LinkageMethod::Ward) {
            using Policy = BlockTiledMatrixPolicy<MatrixType, DistancePolicyType, Method>;

            // Pass the use_mmap flag down through NNChain into the BlockTiled constructor
            NNChain<MatrixType, Policy> nnchain(data, use_mmap, verbose);
            nnchain.cluster();
            return DendrogramUtils::format_nnchain_merges(nnchain.get_merges(),
                                                            num_objs);
        }

        else if constexpr (Method == LinkageMethod::Centroid || Method == LinkageMethod::Median) {
            // Non-reducible methods MUST use the O(1) Matrix Policy to survive GenericLinkage
            using Policy = BlockTiledMatrixPolicy<MatrixType, DistancePolicyType, Method>;
            GenericLinkage<MatrixType, Policy> generic_linkage(data, use_mmap, verbose);
            generic_linkage.cluster();
            return DendrogramUtils::format_nnchain_merges(generic_linkage.get_merges(),
                                                          num_objs);
        }

        // Safety Fallback
        else {
            throw std::invalid_argument("Unsupported Linkage Method requested.");
        }
    }
};

// ===================================================================================
} /* End of namespace trex::ml_methods::clustering::hierarchical::agglomerative */

#endif /* End of ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_DISPATCHER_HPP */
