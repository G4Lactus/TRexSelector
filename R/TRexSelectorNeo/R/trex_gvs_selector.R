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
    #'   "EN" uses TENET (or TENET_AUG, per \code{en_solver}), "IEN" uses
    #'   TIENET_AUG. The \code{solver} field in \code{control} is ignored;
    #'   a warning is issued if it is set to a non-matching solver.
    #' @param control A control list from \code{\link{trex_control}()} (default:
    #'   \code{trex_control()}). The \code{solver} field is overridden by \code{gvs_type}.
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          gvs_control = trex_gvs_control(),
                          control = trex_control()) {

      # The effective solver is derived inside the C++ glue from gvs_type
      # (EN -> TENET/TENET_AUG via en_solver, IEN -> TIENET_AUG); the
      # control$solver field is ignored. Warn when the user explicitly set a
      # non-matching solver ("TLARS" is the trex_control() default and cannot
      # be distinguished from an explicit choice, so it never warns).
      derived_solver <- if (gvs_control$gvs_type == "IEN") {
        "TIENET_AUG"
      } else if (identical(gvs_control$en_solver, "TENET_AUG")) {
        "TENET_AUG"
      } else {
        "TENET"
      }
      if (!(control$solver %in% c("TLARS", derived_solver))) {
        warning("control$solver = \"", control$solver,
                "\" is ignored by TRexGVSSelector: gvs_type = \"",
                gvs_control$gvs_type, "\" derives solver \"",
                derived_solver, "\".")
      }
      # Reset to a value the shared control parser accepts (the AUG solver
      # names are not part of the base solver vocabulary); the C++ glue
      # overrides solver_type regardless.
      control$solver <- "TLARS"

      # lambda_2 == 0 is the degenerate no-ridge case, NOT automatic CV
      # (the sentinel for that is lambda_2 < 0). A user passing exactly 0
      # in the belief it triggers CV silently gets pure T-LASSO.
      if (isTRUE(gvs_control$lambda_2 == 0)) {
        warning("gvs_control$lambda_2 = 0 is the degenerate no-ridge case ",
                "(pure T-LASSO), not automatic cross-validation. ",
                "Use lambda_2 < 0 (e.g. -1) to auto-determine lambda_2 via CV.")
      }

      if (inherits(X, "MemoryMappedMatrix")) {
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_gvs_mmap_create(
          get_ptr(X), y, tFDR, gvs_control, control, seed, verbose
        )
      } else {
        X <- .as_double_matrix(X)
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
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
