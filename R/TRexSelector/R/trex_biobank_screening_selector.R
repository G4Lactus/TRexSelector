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
#' @references
#' J. Machkour, M. Muma, and D. P. Palomar, "False Discovery Rate Control for Fast Screening of Large-Scale
#'   Genomics Biobanks," in Proc. 22nd IEEE Statist. Signal Process. Workshop (SSP), 2023, pp. 666-670.
#'
#' F. Scheidt, J. Machkour, and M. Muma, "Solving FDR-Controlled Sparse Regression Problems with Five Million
#'   Variables on a Laptop," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt. Process.
#'   (CAMSAP), 2023, pp. 116-120.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
#' @family T-Rex selectors and controls
#' @export
TRexBiobankScreeningSelector <- R6::R6Class("TRexBiobankScreeningSelector",
  public = list(
    #' @description Create a new TRexBiobankScreeningSelector object.
    #' @param X Feature matrix (n x p), or an \code{mmap_matrix} opened
    #'   \code{"readwrite"} (mapped directly, no copy; the screening's inner
    #'   selectors normalize it in place per call and restore it).
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

      # A memory-mapped X is mapped directly ("readwrite": the screening's
      # inner selectors normalize it in place per call and restore it); an R
      # matrix is bound zero-copy in double storage.
      is_mmap <- inherits(X, "mmap_matrix")
      if (!is_mmap) X <- .as_double_matrix(X)
      n_x <- if (is_mmap) dim(X)[1] else nrow(X)

      if (is.vector(Y) || (is.array(Y) && length(dim(Y)) == 1)) {
        if (n_x != length(Y)) stop("Dimension mismatch: nrow(X) != length(Y)")
        Y <- .as_double_vector(Y)
        private$refs <- list(X = X, Y = Y)
        private$is_multi <- FALSE
        private$ptr <- if (is_mmap) {
          trex_biobank_screening_1d_mmap_create(
            X, Y, biobank_control_list, screen_control, control, seed, verbose)
        } else {
          trex_biobank_screening_1d_create(
            X, Y, biobank_control_list, screen_control, control, seed, verbose)
        }
      } else if (is.matrix(Y)) {
        if (n_x != nrow(Y)) stop("Dimension mismatch: nrow(X) != nrow(Y)")
        Y <- .as_double_matrix(Y)
        private$refs <- list(X = X, Y = Y)
        private$is_multi <- TRUE
        private$ptr <- if (is_mmap) {
          trex_biobank_screening_2d_mmap_create(
            X, Y, biobank_control_list, screen_control, control, seed, verbose)
        } else {
          trex_biobank_screening_2d_create(
            X, Y, biobank_control_list, screen_control, control, seed, verbose)
        }
      } else {
        stop("Response evaluation Y must be either a numeric vector or a numeric matrix.")
      }
    },

    #' @description Run the Biobank Screening algorithm.
    #'
    #' @return For a vector \code{Y}, a single per-phenotype record: a named list
    #'   with \code{phenotype_index}, \code{selected_indices}, \code{estimated_FDR},
    #'   \code{method_used}, \code{estimated_FDR_screen_ordinary},
    #'   \code{estimated_FDR_screen_bootstrap}, \code{selected_indices_screen_ordinary},
    #'   \code{selected_indices_screen_bootstrap}, and \code{used_fallback_trex}
    #'   (indices are 1-based). For a matrix \code{Y}, a plain list of such records,
    #'   one per column, so \code{res[[i]]} has the same shape as the vector case.
    #'   This mirrors the Python binding, which returns one result object (vector)
    #'   or a list of them (matrix). A tidy summary table is one line away, e.g.
    #'   \code{do.call(rbind, lapply(res, function(r)}
    #'   \code{as.data.frame(r[c("phenotype_index", "estimated_FDR", "method_used", "used_fallback_trex")])))}.
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
