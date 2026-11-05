# =============================================================================
# clustering.R - Hierarchical Clustering and Tree Cutting
#
# Functions are based on the implementations within the TRexSelector package,
# written in Cpp with Rcpp interfaces.
# =============================================================================


#' @importFrom Rcpp evalCpp
NULL


#' @title Linkage Methods
#'
#' @description Enumeration of supported hierarchical clustering linkage methods.
#'              Supported methods include:
#'              - `Ward`: Ward's minimum variance method: codes as 0.
#'              - `Average` (UPGMA): Average linkage: codes as 1.
#'              - `Complete`: Complete linkage: codes as 2.
#'              - `Single`: Single linkage: codes as 3.
#'              - `WPGMA`: Weighted pair group method with arithmetic mean: codes as 4.
#'              - `Median`: Median linkage: codes as 5.
#'              - `Centroid`: Centroid linkage: codes as 6.
#'
#' @references
#' G. N. Lance and W. T. Williams, "A General Theory of Classificatory Sorting Strategies: I. Hierarchical
#'   Systems," Comput. J., vol. 9, pp. 373-380, 1967.
#'
#' D. Müllner, "Modern Hierarchical, Agglomerative Clustering Algorithms," arXiv:1109.2378, 2011.
#'
#' @examples
#' ml_linkage_method$Ward
#' ml_linkage_method$Average
#' @family ML methods
#' @export
ml_linkage_method <- list(
  Ward = 0,
  Average = 1,
  Complete = 2,
  Single = 3,
  WPGMA = 4,
  Median = 5,
  Centroid = 6
)


#' @title Distance Metrics
#'
#' @description Enumeration of supported distance metrics for hierarchical clustering.
#'             Supported metrics include:
#'            - `Euclidean`: Standard Euclidean distance: codes as 0.
#'            - `Correlation`: Correlation distance (1 - correlation): codes as 1.
#'            - `Manhattan`: Manhattan (L1) distance: codes as 2.
#'            - `Correlation_LSH_Filter`: Correlation distance with LSH-based filtering: codes as 3.
#'            - `Correlation_LSH_Approx`: Approximate correlation distance using LSH: codes as 4.
#'
#' @references
#' M. S. Charikar, "Similarity Estimation Techniques from Rounding Algorithms," in Proc. ACM Symp. Theory
#'   Comput. (STOC), 2002.
#'
#' P. Indyk and R. Motwani, "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality," in
#'   Proc. 30th ACM Symp. Theory Comput. (STOC), 1998, pp. 604-613.
#'
#' P. Li, T. J. Hastie, and K. W. Church, "Very Sparse Random Projections," in Proc. ACM SIGKDD Int. Conf.
#'   Knowl. Discovery Data Mining (KDD), 2006.
#'
#' @examples
#' ml_distance_metric$Euclidean
#' ml_distance_metric$Correlation
#' @family ML methods
#' @export
ml_distance_metric <- list(
  Euclidean = 0,
  Correlation = 1,
  Manhattan = 2,
  Correlation_LSH_Filter = 3,
  Correlation_LSH_Approx = 4
)


#' @title Agglomerative Clustering
#'
#' @description Performs hierarchical agglomerative clustering on a numeric matrix.
#'
#' @param data   A numeric matrix, or an \code{mmap_matrix} created by
#'               \code{\link{mem_matrix_from}}.
#' @param method A value from `ml_linkage_method` (e.g. `ml_linkage_method$Ward`).
#' @param metric A value from `ml_distance_metric` (e.g. `ml_distance_metric$Euclidean`).
#'               Default is Euclidean.
#' @param use_mmap Whether to use memory mapping files for intermediate storage. Default is FALSE.
#'                 Ignored when \code{data} is already an \code{mmap_matrix} (always \code{TRUE}).
#' @param verbose Whether the backend prints its allocation line (the size of the distance matrix in
#'                RAM or in the memory-mapped file). Default is FALSE.
#' @param n_threads Integer, the thread budget of the distance loops (\code{NULL}, the default: the
#'                  library's default budget, \code{\link{trex_get_num_threads}()}, initially 1).
#'
#' @return A numeric matrix representing the computed hierarchical linkage.
#'
#' @references
#' R. Sibson, "SLINK: An Optimally Efficient Algorithm for the Single-Link Cluster Method," Comput. J.,
#'   vol. 16, no. 1, pp. 30-34, 1973.
#'
#' J.-P. Benzécri, "Construction d'une classification ascendante hiérarchique par la recherche en chaîne des
#'   voisins réciproques," Les Cahiers de l'Analyse des Données, vol. 7, no. 2, pp. 209-218, 1982.
#'
#' J. Juan, "Programme de classification hiérarchique par l'algorithme de la recherche en chaîne des voisins
#'   réciproques," Les Cahiers de l'Analyse des Données, vol. 7, no. 2, pp. 219-225, 1982.
#'
#' G. N. Lance and W. T. Williams, "A General Theory of Classificatory Sorting Strategies: I. Hierarchical
#'   Systems," Comput. J., vol. 9, pp. 373-380, 1967.
#'
#' D. Müllner, "Modern Hierarchical, Agglomerative Clustering Algorithms," arXiv:1109.2378, 2011.
#'
#' F. Murtagh and P. Contreras, "Algorithms for Hierarchical Clustering: An Overview," Wiley Interdiscip. Rev.
#'   Data Min. Knowl. Discov., vol. 2, no. 1, pp. 86-97, 2012, doi: 10.1002/widm.53.
#'
#' F. Murtagh and P. Contreras, "Algorithms for Hierarchical Clustering: An Overview, II," Wiley Interdiscip.
#'   Rev. Data Min. Knowl. Discov., vol. 7, no. 6, Art. no. e1219, 2017, doi: 10.1002/widm.1219.
#'
#' @examples
#' set.seed(1)
#' data <- matrix(rnorm(30), nrow = 10)
#' ml_agglomerative_cluster(data, method = ml_linkage_method$Ward)
#' @family ML methods
#' @export
ml_agglomerative_cluster <- function(data,
                                  method = ml_linkage_method$Ward,
                                  metric = ml_distance_metric$Euclidean,
                                  use_mmap = FALSE,
                                  verbose = FALSE,
                                  n_threads = NULL) {
  n_threads <- resolve_n_threads(n_threads)
  is_mmap <- inherits(data, "mmap_matrix")
  if (!is_mmap && (!is.matrix(data) || !is.numeric(data))) {
    stop("Input 'data' must be a numeric matrix or mmap_matrix.")
  }

  # Euclidean-geometry methods require Euclidean distances. Ward, Median, and
  # Centroid are all centroid-based and their Lance-Williams update formulas
  # are derived from the law of cosines in Euclidean space. Applying them to
  # correlation or Manhattan distances produces a mathematically invalid
  # dendrogram without any runtime error (consistent with SciPy's behaviour,
  # but disallowed here following scikit-learn's stricter convention).
  #
  # Exception: Ward + Correlation_LSH_Approx is a defined approximate operation
  # — the dispatcher routes it to a 64-dimensional SimHash-projected Euclidean
  # space (ProjectedGeometricUpdatePolicy), so it is permitted.
  euclidean_only_methods <- c(ml_linkage_method$Ward, ml_linkage_method$Median, ml_linkage_method$Centroid)
  ward_lsh_approx_exception <- (method == ml_linkage_method$Ward &&
                                  metric == ml_distance_metric$Correlation_LSH_Approx)

  if (method %in% euclidean_only_methods &&
        metric  != ml_distance_metric$Euclidean  &&
        !ward_lsh_approx_exception) {

    method_name <- switch(as.character(method),
      "0" = "Ward",
      "5" = "Median",
      "6" = "Centroid",
      paste0("ml_linkage_method index ", method)
    )
    stop(sprintf(
      paste0("'%s' linkage requires Euclidean distances because its merge criterion\n",
             "is defined in terms of Euclidean centroids (law of cosines).\n",
             "Use ml_linkage_method$Average, $Complete, $Single, or $WPGMA for non-Euclidean metrics.\n",
             "Exception: Ward + ml_distance_metric$Correlation_LSH_Approx is permitted\n",
             "(Ward is performed in a 64-dimensional SimHash-projected Euclidean space)."),
      method_name
    ))
  }

  # Delegate to Rcpp layer
  if (is_mmap) {
    result <- rcpp_agglomerative_cluster_mmap(data, method, metric, as.logical(verbose), n_threads)
  } else {
    result <- rcpp_agglomerative_cluster(data, method, metric, use_mmap, as.logical(verbose),
                                         n_threads)
  }
  # convert 0-based Cpp cluster IDs to 1-based R convention
  result[, 1:2] <- result[, 1:2] + 1L
  result
}


#' @title Cut Hierarchical Linkage Matrix
#'
#' @description Flatten the dendrogram clustering results based on the number of requested groups.
#'
#' @param linkage       A numeric matrix representing linkage, generated by `ml_agglomerative_cluster`.
#' @param num_orig_objs Original number of observations data rows.
#' @param num_clusters  Target flat group capacity.
#'
#' @return An integer vector mapping observation indexes to cluster arrays.
#'
#' @examples
#' set.seed(1)
#' data <- matrix(rnorm(30), nrow = 10)
#' linkage <- ml_agglomerative_cluster(data, method = ml_linkage_method$Ward)
#' ml_cut_tree(linkage, num_orig_objs = 10, num_clusters = 3)
#' @family ML methods
#' @export
ml_cut_tree <- function(linkage, num_orig_objs, num_clusters) {
  if (!is.matrix(linkage) || !is.numeric(linkage)) {
    stop("Input 'linkage' must be a numeric matrix generated by ml_agglomerative_cluster().")
  }
  if (num_clusters < 1 || num_clusters > num_orig_objs) {
    stop("num_clusters must be > 0 and <= num_orig_objs")
  }

  # Convert 1-based R linkage cluster IDs back to 0-based Cpp before calling Rcpp;
  linkage_0 <- linkage
  linkage_0[, 1:2] <- linkage_0[, 1:2] - 1L
  # then shift the returned 0-based labels to 1-based R convention
  rcpp_cut_tree(linkage_0, as.integer(num_orig_objs), as.integer(num_clusters)) + 1L
}
