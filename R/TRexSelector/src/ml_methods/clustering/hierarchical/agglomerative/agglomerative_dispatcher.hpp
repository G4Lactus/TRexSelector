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
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/slink_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/nnchain_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/generic_linkage_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/block_tiled_matrix_policy.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/proj_geom_updates.hpp>

// bit-based finiteness (consumers may compile this header with -ffast-math)
#include <utils/fp_classify/fp_classify.hpp>

// std includes
#include <atomic>
#include <complex>
#include <string>
#include <utils/openmp/utils_openmp.hpp>

// ===================================================================================

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

// ===================================================================================

namespace detail {

/**
 * @brief Reject non-finite input before any engine sees it.
 *
 * @details A NaN or Inf entry makes every distance involving that column
 *  NaN; the nearest-neighbour searches then compare false everywhere and
 *  return their sentinel index into container indexing (the CTRexSelector
 *  hclust crash, misdiagnosed for a day as an OpenMP stack overflow). The
 *  check is O(np) against the O(p^2 n) clustering and scans the columns in
 *  parallel on the process-wide thread count (a one-shot kernel). The
 *  smallest offending column wins, so the message names the entry a serial
 *  scan names. Bit-based (fp_classify): std::isfinite folds away under
 *  -ffast-math.
 *
 * @throws std::invalid_argument naming the first offending entry.
 */
template <typename MatrixType>
void reject_non_finite(const MatrixType& data, int n_threads) {
    namespace fpc = trex::utils::fp_classify;
    using Scalar = typename MatrixType::Scalar;
    const Eigen::Index n_rows = data.rows();
    const Eigen::Index n_cols = data.cols();

    auto is_finite = [](const Scalar& v) -> bool {
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
            return fpc::isfinite(static_cast<double>(std::real(v))) &&
                   fpc::isfinite(static_cast<double>(std::imag(v)));
        } else {
            return fpc::isfinite(static_cast<double>(v));
        }
    };
    // First non-finite row of column j, -1 for a clean column
    auto first_bad_row = [&](Eigen::Index j) -> Eigen::Index {
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            if (!is_finite(data(i, j))) return i;
        }
        return -1;
    };

    std::atomic<Eigen::Index> bad_col{n_cols};
    [[maybe_unused]] const int nt = ::trex::utils::openmp::threads_for(
        n_threads, static_cast<double>(n_rows) * static_cast<double>(n_cols));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (Eigen::Index j = 0; j < n_cols; ++j) {
        ::trex::utils::openmp::team_probe::record();
        // Columns past an offending one need no scan
        if (j >= bad_col.load(std::memory_order_relaxed)) continue;
        if (first_bad_row(j) < 0) continue;
        Eigen::Index seen = bad_col.load(std::memory_order_relaxed);
        while (j < seen &&
               !bad_col.compare_exchange_weak(seen, j, std::memory_order_relaxed)) {}
    }

    const Eigen::Index j = bad_col.load();
    if (j == n_cols) return;
    throw std::invalid_argument(
        "AgglomerativeClustering: non-finite entry at row " +
        std::to_string(first_bad_row(j)) + ", column " + std::to_string(j) +
        " — every distance of that column would be NaN and the "
        "nearest-neighbour search undefined; clean the data first.");
}

}  // namespace detail

// ===================================================================================

/**
 * @brief Agglomerative hierarchical clustering, dispatching to the SLINK, NN-chain or generic linkage engine.
 *
 *  References:
 *  -----------
 *   R. Sibson, "SLINK: An Optimally Efficient Algorithm for the Single-Link Cluster Method," Comput. J.,
 *     vol. 16, no. 1, pp. 30-34, 1973.
 *   J.-P. Benzécri, "Construction d'une classification ascendante hiérarchique par la recherche en chaîne des
 *     voisins réciproques," Les Cahiers de l'Analyse des Données, vol. 7, no. 2, pp. 209-218, 1982.
 *   J. Juan, "Programme de classification hiérarchique par l'algorithme de la recherche en chaîne des voisins
 *     réciproques," Les Cahiers de l'Analyse des Données, vol. 7, no. 2, pp. 219-225, 1982.
 *   G. N. Lance and W. T. Williams, "A General Theory of Classificatory Sorting Strategies: I. Hierarchical
 *     Systems," Comput. J., vol. 9, pp. 373-380, 1967.
 *   D. Müllner, "Modern Hierarchical, Agglomerative Clustering Algorithms," arXiv:1109.2378, 2011.
 *   F. Murtagh and P. Contreras, "Algorithms for Hierarchical Clustering: An Overview," Wiley Interdiscip.
 *     Rev. Data Min. Knowl. Discov., vol. 2, no. 1, pp. 86-97, 2012, doi: 10.1002/widm.53.
 *   F. Murtagh and P. Contreras, "Algorithms for Hierarchical Clustering: An Overview, II," Wiley
 *     Interdiscip. Rev. Data Min. Knowl. Discov., vol. 7, no. 6, Art. no. e1219, 2017,
 *     doi: 10.1002/widm.1219.
 */
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
     *   monotonicity globally by re-sorting all merges by distance and rebuilding the
     *   tree via Kruskal's algorithm. For the non-reducible methods (Centroid, Median)
     *   this is more than a row reordering: when inversions occur, the rebuilt tree can
     *   have a DIFFERENT topology than SciPy's chronological dendrogram, so cut_tree
     *   results may differ around inverted merges.
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
                                          bool verbose = true,
                                          int n_threads = ::trex::utils::openmp::get_num_threads()) {

        Eigen::Index num_objs = data.cols();
        if (n_threads < 1) {
            throw std::invalid_argument("AgglomerativeClustering: n_threads must be >= 1.");
        }

        // Every engine below assumes finite distances.
        detail::reject_non_finite(data, n_threads);

        // -----------------------------------------------------------------------
        // PATH 1: SINGLE LINKAGE (SLINK Engine, O(N) Space)
        // -----------------------------------------------------------------------
        if constexpr (Method == LinkageMethod::Single) {
            SLINK<MatrixType, DistancePolicyType> slink(data, n_threads);
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
                NNChain<MatrixType, Policy> nnchain(data, use_mmap, verbose, n_threads);
                nnchain.cluster();
                return DendrogramUtils::format_nnchain_merges(nnchain.get_merges(), num_objs);
            }
            else {
                // Average (all metrics), or Ward with non-LSH_Approx metrics:
                // use the exact Lance-Williams Matrix Engine.
                // Ward's LW recurrence is only valid on squared Euclidean distances.
                static_assert(Method != LinkageMethod::Ward ||
                              DistancePolicyType::Metric == DistanceMetric::Euclidean,
                              "Ward linkage requires DistanceMetric::Euclidean "
                              "(squared Euclidean); other exact metrics are not valid "
                              "in the Lance-Williams recurrence for Ward.");
                using Policy = BlockTiledMatrixPolicy<MatrixType, DistancePolicyType, Method>;
                NNChain<MatrixType, Policy> nnchain(data, use_mmap, verbose, n_threads);
                nnchain.cluster();
                return DendrogramUtils::format_nnchain_merges(nnchain.get_merges(), num_objs);
            }
        }

        // -----------------------------------------------------------------------
        // PATH 3: LANCE-WILLIAMS MATRIX METHODS (Disk-Bound NVMe, O(N^2) Space)
        // -----------------------------------------------------------------------
        else if constexpr (Method == LinkageMethod::Complete ||
                           Method == LinkageMethod::WPGMA) {
            using Policy = BlockTiledMatrixPolicy<MatrixType, DistancePolicyType, Method>;

            // Pass the use_mmap flag down through NNChain into the BlockTiled constructor
            NNChain<MatrixType, Policy> nnchain(data, use_mmap, verbose, n_threads);
            nnchain.cluster();
            return DendrogramUtils::format_nnchain_merges(nnchain.get_merges(),
                                                            num_objs);
        }

        else if constexpr (Method == LinkageMethod::Centroid || Method == LinkageMethod::Median) {
            // Non-reducible methods MUST use the O(1) Matrix Policy to survive GenericLinkage.
            // Their LW recurrences are only valid on squared Euclidean distances.
            static_assert(DistancePolicyType::Metric == DistanceMetric::Euclidean,
                          "Centroid/Median linkage requires DistanceMetric::Euclidean "
                          "(squared Euclidean); the Lance-Williams centroid/median "
                          "recurrences are not valid for other metrics.");
            using Policy = BlockTiledMatrixPolicy<MatrixType, DistancePolicyType, Method>;
            GenericLinkage<MatrixType, Policy> generic_linkage(data, use_mmap, verbose, n_threads);
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
