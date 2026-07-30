// ===================================================================================
// trex_cluster_hac.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_UTILS_TREX_CLUSTER_HAC_HPP
#define TREX_SELECTOR_METHODS_UTILS_TREX_CLUSTER_HAC_HPP
// ===================================================================================
/**
 * @file trex_cluster_hac.hpp
 *
 * @brief Shared correlation-HAC cluster discovery for the T-Rex selectors.
 *
 * @details
 *  The HAC-dependent half of the shared cluster-dummy machinery (see
 *  trex_cluster_dummies.hpp for the Eigen-only draw/Cholesky/stream part):
 *  agglomerative clustering on the 1 - |corr| distance with the configured
 *  linkage, and the dendrogram cut into disjoint variable clusters. Kept in
 *  its own header so the DummyGenerator can consume the draw machinery
 *  without dragging the agglomerative backend into the base T-Rex
 *  translation units.
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Hierarchical clustering
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::cluster_dummies
// (same namespace as trex_cluster_dummies.hpp — one logical component).
namespace trex::trex_selector_methods::utils::cluster_dummies {

// Local namespace alias
namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;


/**
 * @brief Run agglomerative clustering on the 1 - |corr| distance.
 *
 * @param X       Working design matrix (columns = variables; centered so the
 *                correlation distance is valid).
 * @param linkage Linkage method (Single / Complete / Average / WPGMA).
 * @param verbose Progress output of the clustering backend.
 *
 * @return Full merge sequence (dendrogram).
 *
 * @throws std::invalid_argument on an unsupported linkage method.
 */
inline std::vector<hac::MergeStep> runCorrelationHAC(
    Eigen::Map<Eigen::MatrixXd>& X,
    hac::LinkageMethod linkage,
    bool verbose)
{
    using MapType = Eigen::Map<Eigen::MatrixXd>;
    using CorrDist = hac::DistancePolicy<MapType, hac::DistanceMetric::Correlation>;

    switch (linkage) {
        case hac::LinkageMethod::Single:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Single>(X, /*use_mmap=*/false, verbose);
        case hac::LinkageMethod::Complete:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Complete>(X, /*use_mmap=*/false, verbose);
        case hac::LinkageMethod::Average:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Average>(X, /*use_mmap=*/false, verbose);
        case hac::LinkageMethod::WPGMA:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::WPGMA>(X, /*use_mmap=*/false, verbose);
        default:
            throw std::invalid_argument(
                "cluster_dummies::runCorrelationHAC: unsupported linkage method.");
    }
}


/**
 * @brief Cut the dendrogram at height 1 - corr_max and group the variables.
 *
 * @param merges   Merge sequence from `runCorrelationHAC()`.
 * @param p        Number of variables.
 * @param corr_max Maximum allowed cross-cluster |corr|; cut height 1 - corr_max.
 *
 * @return clusters_list (per-cluster 0-based column indices).
 */
inline std::vector<std::vector<Eigen::Index>> clustersFromHAC(
    const std::vector<hac::MergeStep>& merges,
    std::size_t p,
    double corr_max)
{
    const double height = 1.0 - corr_max;
    auto labels = hac::DendrogramUtils::cut_tree_by_height(
        merges, static_cast<Eigen::Index>(p), height);
    return hac::DendrogramUtils::group_indices_by_label(labels);
}

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::utils::cluster_dummies */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_UTILS_TREX_CLUSTER_HAC_HPP */
