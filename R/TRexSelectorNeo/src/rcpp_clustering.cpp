// ========================================================================================
// rcpp_clustering.cpp - Rcpp bindings for hierarchical agglomerative clustering
// ========================================================================================
/*
 * Note on Roxygen Documentation:
 * All functions exported via Rcpp in this file are documented using standard Roxygen tags
 * but strictly include the `@noRd` tag. These are internal C++ bindings that are wrapped
 * by user-friendly R6 classes and R functions in the package namespace. Generating `.Rd`
 * manuals for these endpoints would clutter the public API.
 */
 //

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <stdexcept>

// Pull in the agglomerative clustering types and dispatcher
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp"

// Memory-mapped matrix support
#include <utils/memmap/memory_mapped_matrix.hpp>

// ========================================================================================

using namespace Rcpp;
using namespace trex::ml_methods::clustering::hierarchical::agglomerative;

// ========================================================================================
// Helper: Convert vector of MergeStep clusters to a SciPy/R compatible linkage matrix
// ========================================================================================

/**
 * @brief Merges a vector of MergeStep objects into a single Eigen::MatrixXd representing the
 *        linkage matrix.
 *
 * @param merges A vector of MergeStep objects
 *
 * @return An Eigen::MatrixXd representing the linkage matrix
 */
Eigen::MatrixXd rcpp_merge_steps_to_linkage(const std::vector<MergeStep>& merges) {
    if (merges.empty()) {
        return Eigen::MatrixXd(0, 4);
    }
    Eigen::MatrixXd out(merges.size(), 4);
    for (Eigen::Index i = 0; i < merges.size(); ++i) {
        out(i, 0) = static_cast<double>(merges[i].cluster1);
        out(i, 1) = static_cast<double>(merges[i].cluster2);
        out(i, 2) = merges[i].distance;
        out(i, 3) = static_cast<double>(merges[i].new_cluster_size);
    }
    return out;
}


/**
 * @brief Converts a linkage matrix to a vector of MergeStep objects
 *
 * @param linkage An Eigen::Map<Eigen::MatrixXd> representing the linkage matrix
 *
 * @return A vector of MergeStep objects
 */
std::vector<MergeStep> rcpp_linkage_to_merge_steps(const Eigen::Map<Eigen::MatrixXd>& linkage) {
    std::vector<MergeStep> merges;
    merges.reserve(linkage.rows());
    for (Eigen::Index i = 0; i < linkage.rows(); ++i) {
        merges.push_back({
            static_cast<Eigen::Index>(linkage(i, 0)),
            static_cast<Eigen::Index>(linkage(i, 1)),
            linkage(i, 2),
            static_cast<Eigen::Index>(linkage(i, 3))
        });
    }
    return merges;
}


// ========================================================================================
// Helper: Dispatch specific Linkage Method
// ========================================================================================

/**
 * @brief Internal helper to dispatch the correct template instantiation of the clustering engine
 *
 * @tparam TransposedType The type of the transposed input data
 *                        (e.g., Eigen::MatrixXd, Eigen::Map<Eigen::MatrixXd>, etc.)
 * @tparam Metric         The distance metric to use
 *
 * @param transposed      The transposed input data
 * @param method          The linkage method to use
 * @param use_mmap        Boolean flag for memory mapping
 *
 * @return A vector of MergeStep representing the clustering result
 */
template <typename TransposedType, DistanceMetric Metric>
std::vector<MergeStep> internal_dispatch_linkage(
    const TransposedType& transposed,
    LinkageMethod method,
    bool use_mmap) {

    using DP = DistancePolicy<TransposedType, Metric>;

    switch (method) {
        case LinkageMethod::Ward:
            // Ward's Lance-Williams recurrence is only valid for (squared)
            // Euclidean geometry; other combinations are rejected at compile
            // time by the dispatcher, so only instantiate the valid one.
            if constexpr (Metric == DistanceMetric::Euclidean) {
                return AgglomerativeClustering::cluster<TransposedType, DP,
                                                LinkageMethod::Ward>(transposed, use_mmap);
            } else {
                throw std::invalid_argument(
                    "Ward linkage strictly requires Euclidean distance.");
            }
        case LinkageMethod::Average:
            return AgglomerativeClustering::cluster<TransposedType, DP,
                                                LinkageMethod::Average>(transposed, use_mmap);
        case LinkageMethod::Complete:
            return AgglomerativeClustering::cluster<TransposedType, DP,
                                                LinkageMethod::Complete>(transposed, use_mmap);
        case LinkageMethod::Single:
            return AgglomerativeClustering::cluster<TransposedType, DP,
                                                LinkageMethod::Single>(transposed, use_mmap);
        case LinkageMethod::WPGMA:
            return AgglomerativeClustering::cluster<TransposedType, DP,
                                                LinkageMethod::WPGMA>(transposed, use_mmap);
        case LinkageMethod::Median:
            // Median/Centroid recurrences likewise require Euclidean geometry
            if constexpr (Metric == DistanceMetric::Euclidean) {
                return AgglomerativeClustering::cluster<TransposedType, DP,
                                                LinkageMethod::Median>(transposed, use_mmap);
            } else {
                throw std::invalid_argument(
                    "Median linkage strictly requires Euclidean distance.");
            }
        case LinkageMethod::Centroid:
            if constexpr (Metric == DistanceMetric::Euclidean) {
                return AgglomerativeClustering::cluster<TransposedType, DP,
                                                LinkageMethod::Centroid>(transposed, use_mmap);
            } else {
                throw std::invalid_argument(
                    "Centroid linkage strictly requires Euclidean distance.");
            }
        default:
            throw std::invalid_argument("Unknown LinkageMethod");
    }
}


// ========================================================================================
// Exported Functions
// ========================================================================================

//' @title Rcpp Agglomerative Clustering
//'
//' @description Perform agglomerative clustering
//'
//' @param data Matrix of shape (N, Features).
//' @param method_idx Linkage method index (0=Ward, 1=Average, 2=Complete, 3=Single, 4=WPGMA,
//'                      5=Median, 6=Centroid)
//' @param metric_idx Distance metric index (0=Euclid, 1=Corr, 2=Manhattan, 3=CorrLSHF, 4=CorrLSHA)
//' @param use_mmap Boolean flag for memory mapping
//'
//' @return A linkage matrix
//'
//' @noRd
// [[Rcpp::export]]
NumericMatrix rcpp_agglomerative_cluster(
    const Eigen::Map<Eigen::MatrixXd>& data,
    int method_idx,
    int metric_idx,
    bool use_mmap = false) {

    // Cast integers back to C++ enums for dispatch
    LinkageMethod method = static_cast<LinkageMethod>(method_idx);
    DistanceMetric metric = static_cast<DistanceMetric>(metric_idx);

    // Engine expects columns to be objects, so transpose.
    // Force evaluation into a concrete MatrixXd — passing a lazy
    // Eigen::Transpose<const Eigen::Map<>> as the MatrixType template parameter
    // causes UB for Average+LSH_Approx (ProjectedGeometricUpdatePolicy) while
    // Ward's different merge ordering happens to avoid the faulty codepath.
    // Using a concrete type mirrors what the C++ demo does with Eigen::Map.
    Eigen::MatrixXd transposed = data.transpose();
    using TransposedType = Eigen::MatrixXd;

    std::vector<MergeStep> merges;

    switch (metric) {
        case DistanceMetric::Euclidean:
            merges = internal_dispatch_linkage<TransposedType,
                                DistanceMetric::Euclidean>(transposed, method, use_mmap);
            break;
        case DistanceMetric::Correlation:
            merges = internal_dispatch_linkage<TransposedType,
                                DistanceMetric::Correlation>(transposed, method, use_mmap);
            break;
        case DistanceMetric::Manhattan:
            merges = internal_dispatch_linkage<TransposedType,
                                DistanceMetric::Manhattan>(transposed, method, use_mmap);
            break;
        case DistanceMetric::Correlation_LSH_Filter:
            merges = internal_dispatch_linkage<TransposedType,
                            DistanceMetric::Correlation_LSH_Filter>(transposed, method, use_mmap);
            break;
        case DistanceMetric::Correlation_LSH_Approx:
            merges = internal_dispatch_linkage<TransposedType,
                            DistanceMetric::Correlation_LSH_Approx>(transposed, method, use_mmap);
            break;
        default:
            stop("Unknown DistanceMetric specified.");
    }

    Eigen::MatrixXd res = rcpp_merge_steps_to_linkage(merges);
    return wrap(res);
}


//' @title Rcpp Agglomerative Clustering (memory-mapped input)
//'
//' @description Perform agglomerative clustering on a memory-mapped input matrix.
//'              The intermediate distance storage is also disk-backed (use_mmap = true).
//'
//' @param data_ptr XPtr to a MemoryMappedMatrix<double> of shape (N, Features).
//' @param method_idx Linkage method index (0=Ward, 1=Average, 2=Complete, 3=Single, 4=WPGMA,
//'                      5=Median, 6=Centroid)
//' @param metric_idx Distance metric index (0=Euclid, 1=Corr, 2=Manhattan, 3=CorrLSHF, 4=CorrLSHA)
//'
//' @return A linkage matrix
//'
//' @noRd
// [[Rcpp::export]]
NumericMatrix rcpp_agglomerative_cluster_mmap(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> data_ptr,
    int method_idx,
    int metric_idx) {

    const Eigen::Map<Eigen::MatrixXd>& data = data_ptr->getMap();

    // Cast integers back to C++ enums for dispatch
    LinkageMethod method = static_cast<LinkageMethod>(method_idx);
    DistanceMetric metric = static_cast<DistanceMetric>(metric_idx);

    // Engine expects columns to be objects, so transpose
    auto transposed = data.transpose();
    using TransposedType = decltype(transposed);

    std::vector<MergeStep> merges;

    switch (metric) {
        case DistanceMetric::Euclidean:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Euclidean>(
                            transposed, method, /*use_mmap=*/true
                        );
            break;
        case DistanceMetric::Correlation:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Correlation>(
                            transposed, method, /*use_mmap=*/true
                        );
            break;
        case DistanceMetric::Manhattan:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Manhattan>(
                            transposed, method, /*use_mmap=*/true
                        );
            break;
        case DistanceMetric::Correlation_LSH_Filter:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Correlation_LSH_Filter>(
                            transposed, method, /*use_mmap=*/true
                        );
            break;
        case DistanceMetric::Correlation_LSH_Approx:
            merges = internal_dispatch_linkage<TransposedType,
                            DistanceMetric::Correlation_LSH_Approx>(
                                transposed, method, /*use_mmap=*/true
                            );
            break;
        default:
            stop("Unknown DistanceMetric specified.");
    }

    Eigen::MatrixXd res = rcpp_merge_steps_to_linkage(merges);
    return wrap(res);
}


//' @title Rcpp Cut Tree
//'
//' @description Cut a linkage matrix into flat clusters
//'
//' @param linkage Linkage matrix from agglomerative_cluster
//' @param num_orig_objs Original number of rows in dataset
//' @param num_clusters Desired number of flat clusters
//'
//' @return Integer vector of cluster indices
//'
//' @noRd
// [[Rcpp::export]]
IntegerVector rcpp_cut_tree(
    const Eigen::Map<Eigen::MatrixXd>& linkage,
    int num_orig_objs,
    int num_clusters) {

    auto merges = rcpp_linkage_to_merge_steps(linkage);

    auto clusters = DendrogramUtils::cut_tree(merges,
                                                    num_orig_objs, num_clusters);

    IntegerVector out(clusters.size());
    for (int i = 0; i < clusters.size(); ++i) {
        out[i] = static_cast<int>(clusters[i]);
    }

    return out;
}
