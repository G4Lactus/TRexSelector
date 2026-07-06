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
#'   \code{"IEN"} (Informed Elastic Net via TIENET_AUG solver).
#' @param lambda_2 Numeric, L2/Elastic Net penalty forwarded to the internal
#'   TRexGVSSelector, in LARS units. A negative value (default \code{-1})
#'   triggers automatic determination (method selected by
#'   \code{lambda2_method}); \code{0} is the degenerate pure T-LASSO case;
#'   a positive value is used as-is.
#' @param lambda2_method Character, auto-determination strategy for
#'   \code{lambda_2} when \code{lambda_2 < 0}:
#'   \code{"CV_1SE_CCD"} (default), \code{"CV_MIN_CCD"},
#'   \code{"CV_1SE_SVD"}, \code{"CV_MIN_SVD"}.
#' @param en_solver Character, Elastic Net solver variant used internally:
#'   \code{"TENET_AUG"} (default, row-augmented LASSO) or \code{"TENET"}
#'   (Gram-based).
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
                              lambda_2               = -1.0,
                              lambda2_method          = "CV_1SE_CCD",
                              en_solver              = "TENET_AUG",
                              K                      = 20L,
                              max_dummy_multiplier   = 10L) {
  mode          <- match.arg(mode,          c("ActiveSet", "Thresholded"))
  gvs_type      <- match.arg(gvs_type,      c("EN", "IEN"))
  lambda2_method <- match.arg(lambda2_method,
                              c("CV_1SE_CCD", "CV_MIN_CCD",
                                "CV_1SE_SVD", "CV_MIN_SVD"))
  en_solver     <- match.arg(en_solver, c("TENET_AUG", "TENET"))
  stopifnot(
    is.numeric(lambda2_ridge_loadings), lambda2_ridge_loadings >= 0,
    is.numeric(lambda_2),
    is.numeric(K),                      K >= 1L,
    is.numeric(max_dummy_multiplier),   max_dummy_multiplier >= 1L
  )
  list(
    mode                  = mode,
    lambda2_ridge_loadings = as.numeric(lambda2_ridge_loadings),
    gvs_type              = gvs_type,
    lambda_2              = as.numeric(lambda_2),
    lambda2_method        = lambda2_method,
    en_solver             = en_solver,
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
#' set.seed(1)
#' n <- 100; p <- 30
#' X <- matrix(rnorm(n * p), n, p)
#' sel <- TRexSPCASelector$new(X, M = 2, tFDR = 0.2,
#'                              control = trex_spca_control(K = 5))
#' sel$select()
#' sel$adjusted_ev
#' sel$active_sets
#' }
#'
#' @export
TRexSPCASelector <- R6::R6Class("TRexSPCASelector",
  private = list(
    X_ref   = NULL,
    M_      = NULL,
    tFDR_   = NULL,
    seed_   = NULL,
    verbose_= NULL,
    ctrl    = NULL,
    result_ = NULL
  ),
  public = list(

    #' @description Create a new TRexSPCASelector.
    #'
    #' @param X Numeric matrix (n x p). Stored by reference; centered in-place
    #'   during \code{select()} and restored on return.
    #' @param M Integer, number of sparse principal components to extract.
    #' @param tFDR Numeric, target false discovery rate in \code{(0, 1)}.
    #' @param control A control list from \code{\link{trex_spca_control}()}
    #'   (default: \code{trex_spca_control()}).
    #' @param seed Integer random seed. Use \code{-1L} (default) for
    #'   non-deterministic (hardware entropy) behaviour.
    #' @param verbose Logical, whether to print progress (default: \code{FALSE}).
    initialize = function(X, M, tFDR,
                          control = trex_spca_control(),
                          seed    = -1L,
                          verbose = FALSE) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(
        is.numeric(X),
        is.numeric(M),    M >= 1L,
        is.numeric(tFDR), tFDR > 0, tFDR < 1,
        is.numeric(control$lambda2_ridge_loadings),
        is.logical(verbose)
      )
      # lambda_2 == 0 is the degenerate no-ridge case, NOT automatic CV
      # (the sentinel for that is lambda_2 < 0). Warn so a user who passed 0
      # expecting CV is not silently handed pure T-LASSO.
      if (isTRUE(control$lambda_2 == 0)) {
        warning("control$lambda_2 = 0 is the degenerate no-ridge case ",
                "(pure T-LASSO), not automatic cross-validation. ",
                "Use lambda_2 < 0 (e.g. -1) to auto-determine lambda_2 via CV.")
      }
      # Double storage so the zero-copy select() view aliases this buffer
      # instead of a silent integer->double copy (see .as_double_matrix).
      private$X_ref    <- .as_double_matrix(X)
      private$M_       <- as.integer(M)
      private$tFDR_    <- as.numeric(tFDR)
      private$seed_    <- as.integer(seed)
      private$verbose_ <- as.logical(verbose)
      private$ctrl     <- control
    },

    #' @description Run T-Rex Sparse PCA selection.
    #'
    #' @return Invisible \code{self} for method chaining. Results accessible
    #'   via active fields \code{Z}, \code{V}, \code{active_sets},
    #'   \code{adjusted_ev}, \code{cumulative_ev}.
    select = function() {
      private$result_ <- trex_spca_select(
        private$X_ref,
        private$M_,
        private$tFDR_,
        private$ctrl,
        private$seed_,
        private$verbose_
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
