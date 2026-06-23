# =============================================================================
# trex_spca_selector.R - TRexSPCASelector class and control parameter constructor
# =============================================================================

#' @name trex_spca_control
#'
#' @title T-Rex SPCA Control Parameters
#'
#' @description Build a control list for the T-Rex Sparse PCA selector.
#'
#' @param mode Character, sparse loading strategy: \code{"ActiveSet"} (default)
#'   assembles loadings via Ridge regression on the T-Rex active set;
#'   \code{"Thresholded"} keeps and normalizes the top-\code{|A|} ordinary
#'   PCA loadings.
#' @param lambda2_ridge_loadings Numeric, Ridge penalty for ActiveSet loading
#'   assembly (default: \code{1e-6}).
#' @param gvs_type Character, GVS variant used internally for per-PC variable
#'   selection: \code{"EN"} (default, Elastic Net via TENET solver) or
#'   \code{"IEN"} (Informed Elastic Net via TLASSO solver).
#' @param lambda_2 Numeric, L2/Elastic Net penalty forwarded to the internal
#'   TRexGVSSelector. \code{0} (default) triggers automatic determination
#'   (method selected by \code{lambda2_method}). Supply a positive value to
#'   bypass auto-determination.
#' @param lambda2_method Character, auto-determination strategy for
#'   \code{lambda_2} when \code{lambda_2 = 0}:
#'   \code{"GCV"} (default), \code{"COND_NUM"}, \code{"CV_MIN"}, \code{"CV_1SE"}.
#' @param seed Integer random seed forwarded to each per-PC T-Rex run.
#'   Use \code{-1L} (default) for non-deterministic (hardware entropy) behavior.
#' @param K Integer, number of T-Rex random experiments per PC (default: 20).
#' @param max_dummy_multiplier Integer, maximum dummy feature multiplier for
#'   T-Rex sub-selection (default: 10).
#'
#' @return A named list for passing to \code{TRexSPCASelector$new()}.
#'
#' @examples
#' trex_spca_control()
#' trex_spca_control(mode = "Thresholded", K = 10)
#' trex_spca_control(gvs_type = "IEN", lambda_2 = 0.01)
#'
#' @export
trex_spca_control <- function(mode                   = "ActiveSet",
                              lambda2_ridge_loadings  = 1e-6,
                              gvs_type               = "EN",
                              lambda_2               = 0.0,
                              lambda2_method          = "GCV",
                              seed                   = -1L,
                              K                      = 20L,
                              max_dummy_multiplier   = 10L) {
  mode          <- match.arg(mode,          c("ActiveSet", "Thresholded"))
  gvs_type      <- match.arg(gvs_type,      c("EN", "IEN"))
  lambda2_method <- match.arg(lambda2_method, c("GCV", "COND_NUM", "CV_MIN", "CV_1SE"))
  stopifnot(
    is.numeric(lambda2_ridge_loadings), lambda2_ridge_loadings >= 0,
    is.numeric(lambda_2),               lambda_2 >= 0,
    is.numeric(seed),
    is.numeric(K),                      K >= 1L,
    is.numeric(max_dummy_multiplier),   max_dummy_multiplier >= 1L
  )
  list(
    mode                  = mode,
    lambda2_ridge_loadings = as.numeric(lambda2_ridge_loadings),
    gvs_type              = gvs_type,
    lambda_2              = as.numeric(lambda_2),
    lambda2_method        = lambda2_method,
    seed                  = as.integer(seed),
    K                     = as.integer(K),
    max_dummy_multiplier  = as.integer(max_dummy_multiplier)
  )
}


#' @name TRexSPCASelector
#'
#' @title T-Rex Sparse PCA Selector
#'
#' @description Extracts sparse principal components under FDR control via
#'   T-Rex variable selection. For each of the \code{M} components, ordinary
#'   PCA identifies a PC-response direction, T-Rex selects a sparse active set
#'   at the target FDR, and a sparse loading vector is assembled (ActiveSet or
#'   Thresholded mode). Explained variance is adjusted for non-orthogonality
#'   via QR decomposition.
#'
#' @importFrom R6 R6Class
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' n <- 60; p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' sel <- TRexSPCASelector$new(X, control = trex_spca_control(K = 5))
#' sel$select(M = 2, tFDR = 0.2)
#' sel$adjusted_ev
#' sel$active_sets
#' }
#'
#' @export
TRexSPCASelector <- R6::R6Class("TRexSPCASelector",
  private = list(
    X_ref   = NULL,
    ctrl    = NULL,
    result_ = NULL
  ),
  public = list(

    #' @description Create a new TRexSPCASelector.
    #'
    #' @param X Numeric matrix (n x p). Stored by reference; centered in-place
    #'   during \code{select()} and restored on return.
    #' @param control A control list from \code{\link{trex_spca_control}()}
    #'   (default: \code{trex_spca_control()}).
    initialize = function(X, control = trex_spca_control()) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(X), is.numeric(control$lambda2_ridge_loadings))
      private$X_ref <- X
      private$ctrl  <- control
    },

    #' @description Run T-Rex Sparse PCA selection.
    #'
    #' @param M Integer, number of sparse principal components to extract.
    #' @param tFDR Numeric, target false discovery rate in \code{(0, 1)}.
    #'
    #' @return Invisible \code{self} for method chaining. Results accessible
    #'   via active fields \code{Z}, \code{V}, \code{active_sets},
    #'   \code{adjusted_ev}, \code{cumulative_ev}.
    select = function(M, tFDR) {
      stopifnot(is.numeric(M), M >= 1L,
                is.numeric(tFDR), tFDR > 0, tFDR < 1)
      seed <- if (!is.null(private$ctrl$seed)) as.integer(private$ctrl$seed) else -1L
      private$result_ <- trex_spca_select(
        private$X_ref,
        as.integer(M),
        as.numeric(tFDR),
        private$ctrl,
        seed
      )
      invisible(self)
    }
  ),
  active = list(

    #' @field Z Score matrix (n x M) from last \code{select()} call.
    Z = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      private$result_$Z
    },

    #' @field V Loading matrix (p x M) from last \code{select()} call.
    V = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      private$result_$V
    },

    #' @field active_sets List of M integer vectors (1-based) from last \code{select()} call.
    active_sets = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      private$result_$active_sets
    },

    #' @field adjusted_ev Marginal adjusted explained variance per component (M-vector).
    adjusted_ev = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      private$result_$adjusted_ev
    },

    #' @field cumulative_ev Cumulative explained variance percentage (M-vector).
    cumulative_ev = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      private$result_$cumulative_ev
    }
  )
)
