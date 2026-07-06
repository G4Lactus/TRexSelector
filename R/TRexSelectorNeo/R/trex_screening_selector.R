# =========================================================================
# trex_screening_selector.R - TRexScreeningSelector class definition
# =========================================================================

#' @name TRexScreeningSelector
#'
#' @title Screen T-Rex Selector
#'
#' @description Fast variable screening algorithm based on TRex architecture.
#'
#' @importFrom R6 R6Class
#'
#' @examples
#' \donttest{
#' data(Gauss_data)
#' X <- Gauss_data$X
#' y <- Gauss_data$y
#' sel <- TRexScreeningSelector$new(X, y, verbose = FALSE)
#' sel$select()
#' sel$selected_indices
#' }
#'
#' @export
TRexScreeningSelector <- R6::R6Class("TRexScreeningSelector",
  inherit = TRexSelector,
  public = list(
    #' @description Create a new TRexScreeningSelector object.
    #'
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param screen_control A screen control list from \code{\link{trex_screen_control}()}
    #'   (default: \code{trex_screen_control()}).
    #' @param control A control list from \code{\link{trex_control}()} (default:
    #'   \code{trex_control()}). The \code{lloop_strategy} field is forced to "STANDARD".
    initialize = function(X, y,
                          seed = -1,
                          verbose = TRUE,
                          screen_control = trex_screen_control(),
                          control = trex_control()) {

      # Screening always uses STANDARD L-loop strategy
      control$lloop_strategy <- "STANDARD"

      if (inherits(X, "MemoryMappedMatrix")) {
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_screening_mmap_create(
          get_ptr(X), y, screen_control, control, seed, verbose
        )
      } else {
        X <- .as_double_matrix(X)
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_screening_create(
          X, y, screen_control, control, seed, verbose
        )
      }
    },

    #' @description Run the Screening algorithm.
    select = function() {
      trex_screening_select(private$ptr)
      invisible(self)
    }
  ),
  active = list(
    #' @field estimated_FDR Empirical FDR expectation bounds.
    estimated_FDR = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_screening_get_estimated_fdr(private$ptr)
    },

    #' @field used_bootstrap Employs bootstrapping.
    used_bootstrap = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_screening_get_used_bootstrap(private$ptr)
    },

    #' @field confidence_level Bounds the bootstrap estimates.
    confidence_level = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_screening_get_confidence_level(private$ptr)
    },

    #' @field estimated_correlation Covariance max value.
    estimated_correlation = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_screening_get_estimated_correlation(private$ptr)
    },

    #' @field beta_mat Complete variables parameter space.
    beta_mat = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_screening_get_beta_mat(private$ptr)
    },

    #' @field dummy_betas Dummy variables parameter space.
    dummy_betas = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_screening_get_dummy_betas(private$ptr)
    }
  )
)
