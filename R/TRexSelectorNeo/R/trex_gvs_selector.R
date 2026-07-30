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
    #'   \code{trex_gvs_control()}). For \code{gvs_type = "EN"} the solver is
    #'   derived from \code{en_solver} (TENET, TENET_AUG, or TCENET) and the
    #'   \code{solver} field in \code{control} is ignored (a warning is issued
    #'   if it is set to a non-matching solver). For \code{gvs_type = "IEN"}
    #'   the \code{solver} field selects the IEN solver: "TIENET"
    #'   (default, native pathwise), "TIENET_AUG", or "TCIENET".
    #' @param control A control list from \code{\link{trex_control}()} (default:
    #'   \code{trex_control()}). For \code{gvs_type = "EN"} the \code{solver}
    #'   field is overridden; for "IEN" it selects the IEN-family solver.
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          gvs_control = trex_gvs_control(),
                          control = trex_control()) {

      # Solver selection mirrors the C++ design:
      #   EN  -> derived from en_solver (TENET / TENET_AUG / TCENET);
      #          control$solver is ignored (warn when it was explicitly set
      #          to a non-matching value; "TLARS" is the trex_control()
      #          default and cannot be distinguished from an explicit
      #          choice, so it never warns).
      #   IEN -> control$solver IS the IEN-solver choice: "TIENET" (native
      #          pathwise, default), "TIENET_AUG" (row-augmented), or
      #          "TCIENET" (CCD). The default "TLARS" maps to "TIENET";
      #          any other value warns and falls back to "TIENET".
      if (gvs_control$gvs_type == "IEN") {
        ien_family <- c("TIENET", "TIENET_AUG", "TCIENET")
        if (!(control$solver %in% c("TLARS", ien_family))) {
          warning("control$solver = \"", control$solver,
                  "\" is not an IEN-family solver; gvs_type = \"IEN\" ",
                  "accepts \"TIENET\", \"TIENET_AUG\", or \"TCIENET\". ",
                  "Using \"TIENET\".")
          control$solver <- "TLARS"  # glue defaults it to TIENET
        }
      } else {
        derived_solver <- if (identical(gvs_control$en_solver, "TENET_AUG")) {
          "TENET_AUG"
        } else if (identical(gvs_control$en_solver, "TCENET")) {
          "TCENET"
        } else {
          "TENET"
        }
        if (!(control$solver %in% c("TLARS", derived_solver))) {
          warning("control$solver = \"", control$solver,
                  "\" is ignored by TRexGVSSelector: gvs_type = \"EN\" ",
                  "derives solver \"", derived_solver,
                  "\" from en_solver.")
        }
        # The C++ glue derives solver_type from en_solver regardless.
        control$solver <- "TLARS"
      }

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
    },

    #' @field groups Integer vector of length \code{p} giving the 1-based cluster
    #'   label assigned to each variable (prior groups for Route 1, or the
    #'   hierarchical-clustering assignment for Route 2). Enables computing
    #'   cluster diagnostics (e.g. purity) on the R side.
    groups = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_gvs_get_groups(private$ptr)
    },

    #' @field group_labels Optional human-readable cluster names (length
    #'   \code{max_clusters}), or an empty character vector when none were
    #'   supplied.
    group_labels = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_gvs_get_group_labels(private$ptr)
    }
  )
)
