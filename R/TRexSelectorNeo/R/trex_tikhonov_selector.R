#' @name TRexTikhonovSelector
#'
#' @title T-Rex Selector with a general Tikhonov penalty
#'
#' @description FDR-controlled variable selection with a user-supplied
#'   Tikhonov penalty matrix \code{K = t(Gamma) \%*\% Gamma} driving the
#'   informed elastic net (TCIENET sparse-K mode). \code{K = I} collapses to
#'   the plain elastic net; a group-mean K reproduces the GVS-IEN penalty;
#'   banded / Laplacian K encode smoothness-informed selection. Dummies are
#'   cluster-aware and FOLDED-coupled by default (see
#'   \code{\link{trex_tikhonov_control}}).
#'
#' @importFrom R6 R6Class
#'
#' @examples
#' \donttest{
#' data(Gauss_data)
#' X <- Gauss_data$X
#' y <- Gauss_data$y
#' ctrl <- trex_tikhonov_control(tikhonov_K = diag(ncol(X)), lambda_2 = 0.5,
#'                               cluster_dummies = FALSE,
#'                               fold_dummy_coupling = FALSE)
#' sel <- TRexTikhonovSelector$new(X, y, tFDR = 0.1, verbose = FALSE,
#'                                 tik_control = ctrl)
#' sel$select()
#' sel$selected_indices
#' }
#' @export
TRexTikhonovSelector <- R6::R6Class("TRexTikhonovSelector",
  inherit = TRexSelector,
  public = list(
    #' @description Create a new TRexTikhonovSelector object.
    #'
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param tik_control A Tikhonov control list from
    #'   \code{\link{trex_tikhonov_control}()} (required — it carries the
    #'   penalty matrix K).
    #' @param control A control list from \code{\link{trex_control}()}
    #'   (default: \code{trex_control()}). The \code{solver} field is forced
    #'   to \code{"TCIENET"} — the only solver with a general-K mode; a
    #'   warning is issued if it was explicitly set to anything else.
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          tik_control,
                          control = trex_control()) {

      if (missing(tik_control)) {
        stop("tik_control is required (build it with trex_tikhonov_control(); ",
             "it carries the penalty matrix K).")
      }

      # TCIENET is the only solver with a general-K mode; the C++ glue
      # forces it. Warn on an explicit non-matching choice ("TLARS" is the
      # trex_control() default and cannot be distinguished from an explicit
      # choice, so it never warns).
      if (!(control$solver %in% c("TLARS", "TCIENET"))) {
        warning("control$solver = \"", control$solver,
                "\" is ignored by TRexTikhonovSelector: the general-K ",
                "penalty requires \"TCIENET\".")
      }
      control$solver <- "TLARS"  # neutral; the C++ glue sets TCIENET

      if (inherits(X, "MemoryMappedMatrix")) {
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_tikhonov_mmap_create(
          get_ptr(X), y, tFDR, tik_control, control, seed, verbose
        )
      } else {
        X <- .as_double_matrix(X)
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_tikhonov_create(
          X, y, tFDR, tik_control, control, seed, verbose
        )
      }
    },

    #' @description Run the Tikhonov T-Rex algorithm.
    select = function() {
      trex_tikhonov_select(private$ptr)
      invisible(self)
    }
  ),
  active = list(
    #' @field lambda2_used The lambda_2 actually used by the last
    #'   \code{select()} run (user-supplied or CV-resolved; solver scale).
    #'   \code{-1} before \code{select()}.
    lambda2_used = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_tikhonov_get_lambda2_used(private$ptr)
    }
  )
)


#' @name gamma_to_k
#'
#' @title Build a Tikhonov penalty matrix K from an operator Gamma
#'
#' @description Computes \code{K = t(Gamma) \%*\% Gamma} (pruned and
#'   compressed) for use as \code{tikhonov_K} in
#'   \code{\link{trex_tikhonov_control}}. For example, a first-difference
#'   Gamma yields the smoothness-encoding graph Laplacian of a chain.
#'
#' @param gamma Regularization operator (m x p): a base \code{matrix} or a
#'   sparse \code{Matrix::dgCMatrix}.
#'
#' @return A sparse \code{Matrix::dgCMatrix} (p x p). Requires the Matrix
#'   package (Suggests) for the sparse return class.
#'
#' @examples
#' # First-difference operator over 5 variables -> chain Laplacian
#' p <- 5
#' gamma <- diff(diag(p))
#' K <- gamma_to_k(gamma)
#'
#' @export
gamma_to_k <- function(gamma) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("gamma_to_k() returns a sparse Matrix::dgCMatrix and needs the ",
         "Matrix package installed.")
  }
  trex_tikhonov_gamma_to_k(gamma)
}
