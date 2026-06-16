#' @name TRexGVSSelector
#'
#' @title T-Rex Selector with Group Variable Selection (TRex GVS)
#'
#' @description FDR-controlled variable selection algorithm utilizing group-based structural
#'              augmentation.
#'
#' @importFrom R6 R6Class
#'
#' @examples
#' \donttest{
#' data(Gauss_data)
#' X <- Gauss_data$X
#' y <- Gauss_data$y
#' sel <- TRexGVSSelector$new(X, y, tFDR = 0.1, verbose = FALSE,
#'                            gvs_control = trex_gvs_control(gvs_type = "EN"))
#' sel$select()
#' sel$selected_indices
#' }
#' @export
TRexGVSSelector <- R6::R6Class("TRexGVSSelector",
  inherit = TRexSelector,
  public = list(
    #' @description Create a new TRexGVSSelector object.
    #'
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param gvs_control A GVS control list from \code{\link{trex_gvs_control}()} (default:
    #'   \code{trex_gvs_control()}). The solver is derived from \code{gvs_type}:
    #'   "EN" uses TENET, "IEN" uses TLASSO. The \code{solver} field in \code{control}
    #'   is ignored.
    #' @param control A control list from \code{\link{trex_control}()} (default:
    #'   \code{trex_control()}). The \code{solver} field is overridden by \code{gvs_type}.
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          gvs_control = trex_gvs_control(),
                          control = trex_control()) {

      private$refs <- list(X = X, y = y)

      # Derive solver from gvs_type; overrides any user-specified control$solver
      control$solver <- if (gvs_control$gvs_type == "IEN") "TLASSO" else "TENET"

      if (inherits(X, "MemoryMappedMatrix")) {
        private$ptr <- trex_gvs_mmap_create(
          get_ptr(X), y, tFDR, gvs_control, control, seed, verbose
        )
      } else {
        private$ptr <- trex_gvs_create(
          X, y, tFDR, gvs_control, control, seed, verbose
        )
      }
    },

    #' @description Run the GVS algorithm.
    select = function() {
      trex_gvs_select(private$ptr)
      invisible(self)
    }
  ),
  active = list(
    #' @field lambda2_used Lambda2 regularization term selected/applied.
    lambda2_used = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_gvs_get_lambda2_used(private$ptr)
    },

    #' @field gvs_type The augmentation policy name used (e.g., "EN", "IEN").
    gvs_type = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_gvs_get_gvs_type(private$ptr)
    },

    #' @field max_clusters The number of unique clusters dynamically formulated.
    max_clusters = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_gvs_get_max_clusters(private$ptr)
    }
  )
)
