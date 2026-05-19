#' @name TRexBiobankScreeningSelector
#' @title Biobank Screen T-Rex Selector
#' @description Genomic biobank phenotype screening algorithm based on TRex architecture.
#' @importFrom R6 R6Class
#' @export
TRexBiobankScreeningSelector <- R6::R6Class("TRexBiobankScreeningSelector",
  public = list(
    #' @description Create a new TRexBiobankScreeningSelector object.
    #' @param X Feature matrix.
    #' @param Y Response vector (1D) or matrix (2D).
    #' @param target_FDR_trex Target FDR for fallback T-Rex.
    #' @param lower_bound_FDR Lower bound for acceptable estimated FDR.
    #' @param upper_bound_FDR Upper bound for acceptable estimated FDR.
    #' @param use_bootstrap_CI Whether to use bootstrap CI screening.
    #' @param R_boot Number of bootstrap iterations.
    #' @param ci_grid_step Grid step sizes for interval calculations.
    #' @param trex_method Screening method: "TREX", "TREX_DA_AR1", "TREX_DA_EQUI", "TREX_DA_BLOCK_EQUI"
    #' @param method The solver method, e.g., "TLARS", "TACGP", "TENET" (default: "TLARS").
    #' @param use_omp Whether to use OpenMP parallelization (default: TRUE).
    #' @param use_memory_mapping Use out-of-core memory mapping (default: FALSE).
    #' @param max_outer_threads Thread count for K repetitions (default: 1).
    #' @param max_inner_threads Thread count for inner solvers (default: 1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    initialize = function(X, Y,
                          target_FDR_trex = 0.1,
                          lower_bound_FDR = 0.05,
                          upper_bound_FDR = 0.15,
                          use_bootstrap_CI = FALSE,
                          R_boot = 1000,
                          ci_grid_step = 0.001,
                          trex_method = "TREX",
                          method = "TLARS",
                          use_omp = TRUE,
                          use_memory_mapping = FALSE,
                          max_outer_threads = 1,
                          max_inner_threads = 1,
                          seed = -1,
                          verbose = TRUE) {

      private$refs <- list(X = X, Y = Y)

      biobank_control_list <- list(
        target_FDR_trex = target_FDR_trex,
        lower_bound_FDR = lower_bound_FDR,
        upper_bound_FDR = upper_bound_FDR
      )

      screen_control_list <- list(
        use_bootstrap_CI = use_bootstrap_CI,
        R_boot = R_boot,
        ci_grid_step = ci_grid_step,
        trex_method = trex_method,
        lloop_strategy = "STANDARD"
      )

      trex_control_list <- list(
        method = method,
        parallel_rnd_experiments = use_omp,
        use_memory_mapping = use_memory_mapping,
        max_outer_threads = max_outer_threads,
        max_inner_threads = max_inner_threads,
        lloop_strategy = "STANDARD"
      )

      if (!is.numeric(Y)) {
        stop("Response evaluation Y must be either a numeric vector or a numeric matrix.")
      }

      if (is.vector(Y) || (is.array(Y) && length(dim(Y)) == 1)) {
        if (nrow(X) != length(Y)) stop("Dimension mismatch: nrow(X) != length(Y)")
        private$is_multi <- FALSE
        private$ptr <- trex_biobank_screening_1d_create(
          X, Y, biobank_control_list, screen_control_list, trex_control_list, seed, verbose
        )
      } else if (is.matrix(Y)) {
        if (nrow(X) != nrow(Y)) stop("Dimension mismatch: nrow(X) != nrow(Y)")
        private$is_multi <- TRUE
        private$ptr <- trex_biobank_screening_2d_create(
          X, Y, biobank_control_list, screen_control_list, trex_control_list, seed, verbose
        )
      } else {
        stop("Response evaluation Y must be either a numeric vector or a numeric matrix.")
      }
    },

    #' @description Run the Biobank Screening algorithm.
    #' @return A list with results. For vector 'Y', it gives a single phenotype result. For matrix 'Y', it gives a multi-phenotype result containing a `$statistics` data.frame and `$selected_indices` arrays.
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
