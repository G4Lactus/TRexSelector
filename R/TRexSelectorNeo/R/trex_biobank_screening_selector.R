# =============================================================================
# TRexBiobankScreeningSelector.R - Biobank Screening T-Rex Selector
# =============================================================================

#' @name TRexBiobankScreeningSelector
#'
#' @title Biobank Screen T-Rex Selector
#'
#' @description Genomic biobank phenotype screening algorithm based on TRex architecture.
#'
#' @importFrom R6 R6Class
#'
#' @examples
#' \donttest{
#' data(Gauss_data)
#' X <- Gauss_data$X
#' y <- Gauss_data$y
#' sel <- TRexBiobankScreeningSelector$new(X, y, tFDR = 0.1, verbose = FALSE)
#' res <- sel$select()
#' res$selected_indices
#' }
#'
#' @export
TRexBiobankScreeningSelector <- R6::R6Class("TRexBiobankScreeningSelector",
  public = list(
    #' @description Create a new TRexBiobankScreeningSelector object.
    #' @param X Feature matrix.
    #' @param Y Response vector (1D) or matrix (2D).
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param biobank_control A biobank control list from \code{\link{trex_biobank_control}()}
    #'   (default: \code{trex_biobank_control()}).
    #' @param screen_control A screen control list from \code{\link{trex_screen_control}()}
    #'   (default: \code{trex_screen_control()}).
    #' @param control A control list from \code{\link{trex_control}()} (default:
    #'   \code{trex_control()}). The \code{lloop_strategy} field is forced to "STANDARD".
    initialize = function(X, Y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          biobank_control = trex_biobank_control(),
                          screen_control = trex_screen_control(),
                          control = trex_control()) {

      biobank_control_list <- c(list(target_FDR_trex = tFDR), biobank_control)

      # Biobank/screening always uses STANDARD L-loop strategy
      screen_control$lloop_strategy <- "STANDARD"
      control$lloop_strategy <- "STANDARD"

      if (!is.numeric(Y)) {
        stop("Response evaluation Y must be either a numeric vector or a numeric matrix.")
      }

      X <- .as_double_matrix(X)

      if (is.vector(Y) || (is.array(Y) && length(dim(Y)) == 1)) {
        if (nrow(X) != length(Y)) stop("Dimension mismatch: nrow(X) != length(Y)")
        Y <- .as_double_vector(Y)
        private$refs <- list(X = X, Y = Y)
        private$is_multi <- FALSE
        private$ptr <- trex_biobank_screening_1d_create(
          X, Y, biobank_control_list, screen_control, control, seed, verbose
        )
      } else if (is.matrix(Y)) {
        if (nrow(X) != nrow(Y)) stop("Dimension mismatch: nrow(X) != nrow(Y)")
        Y <- .as_double_matrix(Y)
        private$refs <- list(X = X, Y = Y)
        private$is_multi <- TRUE
        private$ptr <- trex_biobank_screening_2d_create(
          X, Y, biobank_control_list, screen_control, control, seed, verbose
        )
      } else {
        stop("Response evaluation Y must be either a numeric vector or a numeric matrix.")
      }
    },

    #' @description Run the Biobank Screening algorithm.
    #'
    #' @return A list with results. For vector 'Y', it gives a single phenotype result.
    #' For matrix 'Y', it gives a multi-phenotype result containing a `$statistics` data.frame
    #' and `$selected_indices` arrays.
    select = function() {
      if (private$is_multi) {
        return(trex_biobank_screening_screen_phenotypes(private$ptr))
      } else {
        return(trex_biobank_screening_screen_phenotype(private$ptr))
      }
    }
  ),
  private = list(
    ptr = NULL,
    refs = NULL,
    is_multi = FALSE
  )
)
