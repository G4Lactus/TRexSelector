// ========================================================================================
// clustering_bindings.hpp -- pybind11 bindings for hierarchical agglomerative clustering
// ========================================================================================
#ifndef TREX_PYTHON_CLUSTERING_BINDINGS_HPP
#define TREX_PYTHON_CLUSTERING_BINDINGS_HPP
// ========================================================================================
/**
 * @file clustering_bindings.hpp
 *
 * @brief Pybind11 bindings for Cpp hierarchical agglomerative clustering methods used in
 *        the T-Rex Selector package.
 *
 */
// ========================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// ml_methods includes for clustering includes
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp"
#include "ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp"

// ========================================================================================

namespace py = pybind11;

namespace trex {
namespace python_bindings {

using namespace trex::ml_methods::clustering::hierarchical::agglomerative;

// ========================================================================================


/**
 * @brief Convert a vector of MergeStep structures into a SciPy-compatible linkage matrix.
 *
 * @details Converts `std::vector<MergeStep>` into an `[N-1, 4]`
 *          Eigen::MatrixXd, where each row represents a merge step in the clustering process.
 *          The columns correspond to:
 *          0: index of the first cluster
 *          1: index of the second cluster
 *          2: distance between the merged clusters
 *          3: size of the new cluster
 *
 * @param merges The vector of MergeStep structures.
 *
 * @return An Eigen::MatrixXd representing the linkage matrix.
 */
inline Eigen::MatrixXd merge_steps_to_linkage(const std::vector<MergeStep>& merges) {
    Eigen::MatrixXd linkage(merges.size(), 4);
    for (Eigen::Index i = 0; i < merges.size(); ++i) {
        linkage(i, 0) = static_cast<double>(merges[i].cluster1);
        linkage(i, 1) = static_cast<double>(merges[i].cluster2);
        linkage(i, 2) = merges[i].distance;
        linkage(i, 3) = static_cast<double>(merges[i].new_cluster_size);
    }
    return linkage;
}


/**
 * @brief Convert a SciPy-compatible linkage matrix into a vector of MergeStep structures.
 *
 * @details Converts an `[N-1, 4]` Eigen::MatrixXd into `std::vector<MergeStep>`, where each
 *          row represents a merge step in the clustering process.
 *          The columns correspond to:
 *          0: index of the first cluster
 *          1: index of the second cluster
 *          2: distance between the merged clusters
 *          3: size of the new cluster
 *
 * @param linkage The Eigen::MatrixXd representing the linkage matrix.
 *
 * @return A vector of MergeStep structures.
 */
inline std::vector<MergeStep> linkage_to_merge_steps(const Eigen::MatrixXd& linkage) {
    if (linkage.cols() != 4) {
        throw std::invalid_argument("Linkage matrix must have 4 columns.");
    }
    std::vector<MergeStep> merges(linkage.rows());
    for (Eigen::Index i = 0; i < linkage.rows(); ++i) {
        merges[i].cluster1 = static_cast<int>(linkage(i, 0));
        merges[i].cluster2 = static_cast<int>(linkage(i, 1));
        merges[i].distance = linkage(i, 2);
        merges[i].new_cluster_size = static_cast<int>(linkage(i, 3));
    }
    return merges;
}


/**
 * @brief Inner dispatch helper to resolve LinkageMethod at runtime for a fixed DistanceMetric.
 *
 * @tparam TransposedType The data type of the transposed matrix (e.g., Eigen::Map).
 * @tparam MetricEnum The locally resolved distance metric evaluated at compile time.
 *
 * @param transposed_data The transposed input matrix where columns represent observations.
 * @param method The runtime-selected linkage method specifying the cluster merging criterion.
 * @param use_mmap Boolean flag indicating whether to use memory-mapping for very large datasets.
 * @param verbose Print the backend's allocation line.
 *
 * @return A sequence of merge steps representing the hierarchical clustering.
 */
template <typename TransposedType, DistanceMetric MetricEnum>
std::vector<MergeStep> internal_dispatch_linkage(
    const TransposedType& transposed_data,
    LinkageMethod method,
    bool use_mmap,
    bool verbose,
    int n_threads) {

    using DistPol = DistancePolicy<TransposedType, MetricEnum>;

    switch (method) {
        case LinkageMethod::Ward:
            // Ward strictly requires Euclidean spatial geometry.
            // Correlation_LSH_Approx is the one supported exception: the core
            // dispatcher routes it to ProjectedGeometricUpdatePolicy (Ward in
            // 64-D SimHash-projected Euclidean space).
            if constexpr (MetricEnum == DistanceMetric::Euclidean ||
                          MetricEnum == DistanceMetric::Correlation_LSH_Approx) {
                return AgglomerativeClustering::cluster<TransposedType, DistPol,
                        LinkageMethod::Ward>(transposed_data, use_mmap, verbose, n_threads);
            } else {
                throw std::invalid_argument(
                    "Ward linkage strictly requires Euclidean distance "
                    "(exception: Correlation_LSH_Approx, which runs Ward in a "
                    "SimHash-projected Euclidean space).");
            }
        case LinkageMethod::Average:
            return AgglomerativeClustering::cluster<TransposedType, DistPol,
                    LinkageMethod::Average>(transposed_data, use_mmap, verbose, n_threads);
        case LinkageMethod::Complete:
            return AgglomerativeClustering::cluster<TransposedType, DistPol,
                    LinkageMethod::Complete>(transposed_data, use_mmap, verbose, n_threads);
        case LinkageMethod::Single:
            return AgglomerativeClustering::cluster<TransposedType, DistPol,
                     LinkageMethod::Single>(transposed_data, use_mmap, verbose, n_threads);
        case LinkageMethod::WPGMA:
            return AgglomerativeClustering::cluster<TransposedType, DistPol,
                     LinkageMethod::WPGMA>(transposed_data, use_mmap, verbose, n_threads);
        case LinkageMethod::Median:
            // Median strictly requires (squared) Euclidean geometry
            if constexpr (MetricEnum == DistanceMetric::Euclidean) {
                return AgglomerativeClustering::cluster<TransposedType, DistPol,
                         LinkageMethod::Median>(transposed_data, use_mmap, verbose, n_threads);
            } else {
                throw std::invalid_argument("Median linkage strictly requires Euclidean distance.");
            }
        case LinkageMethod::Centroid:
            // Centroid strictly requires (squared) Euclidean geometry
            if constexpr (MetricEnum == DistanceMetric::Euclidean) {
                return AgglomerativeClustering::cluster<TransposedType, DistPol,
                         LinkageMethod::Centroid>(transposed_data, use_mmap, verbose, n_threads);
            } else {
                throw std::invalid_argument("Centroid linkage strictly requires Euclidean distance.");
            }
        default:
            throw std::invalid_argument("Unknown LinkageMethod");
    }
}


/**
 * @brief Performs hierarchical agglomerative clustering on the provided data.
 *
 * @details This function provides the primary entry point for agglomerative clustering
 *          from Python. It supports multiple linkage rules and exact/approximate distances.
 *          It constructs a SciPy-compatible linkage matrix as its result.
 *
 * @param data The input data matrix where rows are objects and columns are features.
 * @param method The runtime-selected linkage method specifying the cluster merging criterion.
 * @param metric The requested distance metric to compute pairwise distances.
 * @param use_mmap Boolean flag indicating whether to use memory-mapping for very large datasets.
 *
 * @return An Eigen::MatrixXd of shape [N-1, 4] representing the hierarchical linkage.
 *         Columns correspond to [cluster1, cluster2, distance, new_cluster_size].
 */
inline Eigen::MatrixXd py_agglomerative_cluster(
    Eigen::Ref<const Eigen::MatrixXd> data,
    LinkageMethod method,
    DistanceMetric metric,
    bool use_mmap = false,
    bool verbose = false,
    std::optional<int> n_threads_opt = std::nullopt) {

    // The thread budget of the clustering: the library's default unless given
    const int n_threads = n_threads_opt.value_or(trex::utils::openmp::get_num_threads());

    // Python has row objects, the C++ engine expects columns to be objects.
    // `data` is a read-only view of the caller's Fortran-ordered float64 array
    // (a C-ordered array is converted once by pybind11); a
    // `const Eigen::MatrixXd&` parameter would copy it in full on every call.
    auto transposed_data = data.transpose();
    using TransposedType = decltype(transposed_data);

    std::vector<MergeStep> merges;

    switch (metric) {
        case DistanceMetric::Euclidean:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Euclidean>(transposed_data, method, use_mmap, verbose, n_threads);
            break;
        case DistanceMetric::Correlation:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Correlation>(transposed_data, method, use_mmap, verbose, n_threads);
            break;
        case DistanceMetric::Manhattan:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Manhattan>(transposed_data, method, use_mmap, verbose, n_threads);
            break;
        case DistanceMetric::Correlation_LSH_Filter:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Correlation_LSH_Filter>(transposed_data, method, use_mmap, verbose, n_threads);
            break;
        case DistanceMetric::Correlation_LSH_Approx:
            merges = internal_dispatch_linkage<TransposedType,
                        DistanceMetric::Correlation_LSH_Approx>(transposed_data, method, use_mmap, verbose, n_threads);
            break;
        default:
            throw std::invalid_argument("Unknown DistanceMetric");
    }

    return merge_steps_to_linkage(merges);
}


/**
 * @brief Cuts a hierarchical clustering linkage matrix to form a specified number of flat clusters.
 *
 * @details Takes a SciPy-compatible linkage matrix and recursively flattens the
 *          dendrogram to extract exactly `num_clusters` discrete groups.
 *
 * @param linkage The linkage matrix generated by `py_agglomerative_cluster`.
 * @param num_orig_objs The original number of objects (rows) in the dataset.
 * @param num_clusters The desired number of flat clusters to extract.
 *
 * @return A vector of integer cluster labels corresponding to each original object.
 */
inline std::vector<int> py_cut_tree(
    const Eigen::MatrixXd& linkage, int num_orig_objs, int num_clusters) {

    auto merges = linkage_to_merge_steps(linkage);

    // DendrogramUtils::cut_tree returns std::vector<Eigen::Index>
    auto clusters = DendrogramUtils::cut_tree(
        merges, num_orig_objs, num_clusters);

    // Cast Eigen::Index to int for standard numpy int32 array output
    std::vector<int> out(clusters.size());
    for (size_t i = 0; i < clusters.size(); ++i) {
        out[i] = static_cast<int>(clusters[i]);
    }

    return out;
}


/**
 * @brief Bind clustering-related functions and enums to a Python module.
 *
 * @param m The Python module to bind to.
 */
inline void bind_clustering(py::module_& m) {

    py::enum_<LinkageMethod>(m, "LinkageMethod",
        "Linkage criterion of the agglomerative clustering, applied through the Lance-Williams\n"
        "update when two clusters merge.\n"
        "\n"
        "References\n"
        "----------\n"
        "G. N. Lance and W. T. Williams, \"A General Theory of Classificatory Sorting Strategies:\n"
        "  I. Hierarchical Systems,\" Comput. J., vol. 9, pp. 373-380, 1967.\n"
        "D. Müllner, \"Modern Hierarchical, Agglomerative Clustering Algorithms,\" arXiv:1109.2378,\n"
        "  2011.\n")
        .value("Ward", LinkageMethod::Ward)
        .value("Average", LinkageMethod::Average)
        .value("Complete", LinkageMethod::Complete)
        .value("Single", LinkageMethod::Single)
        .value("WPGMA", LinkageMethod::WPGMA)
        .value("Median", LinkageMethod::Median)
        .value("Centroid", LinkageMethod::Centroid)
        .export_values();

    py::enum_<DistanceMetric>(m, "DistanceMetric",
        "Pairwise distance between data columns for the agglomerative clustering: exact metrics and\n"
        "two locality-sensitive hashing (SimHash) approximations of the correlation distance.\n"
        "\n"
        "References\n"
        "----------\n"
        "M. S. Charikar, \"Similarity Estimation Techniques from Rounding Algorithms,\" in Proc.\n"
        "  ACM Symp. Theory Comput. (STOC), 2002.\n"
        "P. Indyk and R. Motwani, \"Approximate Nearest Neighbors: Towards Removing the Curse of\n"
        "  Dimensionality,\" in Proc. 30th ACM Symp. Theory Comput. (STOC), 1998, pp. 604-613.\n"
        "P. Li, T. J. Hastie, and K. W. Church, \"Very Sparse Random Projections,\" in Proc. ACM\n"
        "  SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD), 2006.\n")
        .value("Euclidean", DistanceMetric::Euclidean)
        .value("Correlation", DistanceMetric::Correlation)
        .value("Manhattan", DistanceMetric::Manhattan)
        .value("Correlation_LSH_Filter", DistanceMetric::Correlation_LSH_Filter)
        .value("Correlation_LSH_Approx", DistanceMetric::Correlation_LSH_Approx)
        .export_values();

    m.def("agglomerative_cluster", &py_agglomerative_cluster,
          py::arg("data"),
          py::arg("method"),
          py::arg("metric") = DistanceMetric::Euclidean,
          py::arg("use_mmap") = false,
          py::arg("verbose") = false,
          py::arg("n_threads") = py::none(),
          "Perform hierarchical agglomerative clustering, returning a SciPy-style [N-1, 4] "
          "linkage matrix. verbose=True prints the backend's allocation line (the size of the "
          "distance matrix in RAM or in the memory-mapped file). n_threads is the thread budget "
          "of the distance loops (None: the library's default budget, utils.get_num_threads(), "
          "initially 1).\n"
          "\n"
          "References\n"
          "----------\n"
          "R. Sibson, \"SLINK: An Optimally Efficient Algorithm for the Single-Link Cluster Method,\"\n"
          "  Comput. J., vol. 16, no. 1, pp. 30-34, 1973.\n"
          "J.-P. Benzécri, \"Construction d'une classification ascendante hiérarchique par la\n"
          "  recherche en chaîne des voisins réciproques,\" Les Cahiers de l'Analyse des Données,\n"
          "  vol. 7, no. 2, pp. 209-218, 1982.\n"
          "J. Juan, \"Programme de classification hiérarchique par l'algorithme de la recherche en\n"
          "  chaîne des voisins réciproques,\" Les Cahiers de l'Analyse des Données, vol. 7, no. 2,\n"
          "  pp. 219-225, 1982.\n"
          "G. N. Lance and W. T. Williams, \"A General Theory of Classificatory Sorting Strategies:\n"
          "  I. Hierarchical Systems,\" Comput. J., vol. 9, pp. 373-380, 1967.\n"
          "D. Müllner, \"Modern Hierarchical, Agglomerative Clustering Algorithms,\" arXiv:1109.2378,\n"
          "  2011.\n"
          "F. Murtagh and P. Contreras, \"Algorithms for Hierarchical Clustering: An Overview,\"\n"
          "  Wiley Interdiscip. Rev. Data Min. Knowl. Discov., vol. 2, no. 1, pp. 86-97, 2012,\n"
          "  doi: 10.1002/widm.53.\n"
          "F. Murtagh and P. Contreras, \"Algorithms for Hierarchical Clustering: An Overview, II,\"\n"
          "  Wiley Interdiscip. Rev. Data Min. Knowl. Discov., vol. 7, no. 6, Art. no. e1219, 2017,\n"
          "  doi: 10.1002/widm.1219.\n");

    m.def("cut_tree", &py_cut_tree,
          py::arg("linkage"), py::arg("num_orig_objs"), py::arg("num_clusters"),
          "Cut a hierarchical linkage matrix to form exactly `num_clusters` flat clusters.");
}

// ========================================================================================
} /* End of namespace python_bindings */
} /* End of namespace trex */
// ========================================================================================
#endif /* End of TREX_PYTHON_CLUSTERING_BINDINGS_HPP */
