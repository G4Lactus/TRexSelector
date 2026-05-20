#' @name TRexScreeningSelector
#' @title Screen T-Rex Selector
#' @description Fast variable screening algorithm based on TRex architecture.
#' @importFrom R6 R6Class
#' @export
TRexScreeningSelector <- R6::R6Class("TRexScreeningSelector",
  public = list(
    #' @description Create a new TRexScreeningSelector object.
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param use_bootstrap_CI Whether to use bootstrap CI screening.
    #' @param R_boot Number of bootstrap iterations.
    #' @param ci_grid_step the grid step sizes for interval calculations
    #' @param trex_method Screening method: "TREX", "TREX_DA_AR1", "TREX_DA_EQUI", "TREX_DA_BLOCK_EQUI"
    #' @param method The solver method, e.g., "LARS", "TACGP", "TENET" (default: "LARS").
    #' @param use_omp Whether to use OpenMP parallelization (default: TRUE).
    #' @param use_memory_mapping Use out-of-core memory mapping (default: FALSE).
    #' @param max_outer_threads Thread count for K repetitions (default: 1).
    #' @param max_inner_threads Thread count for inner solvers (default: 1).
    initialize = function(X, y,
                          seed = -1,
                          verbose = TRUE,
                          use_bootstrap_CI = FALSE,
                          R_boot = 1000,
                          ci_grid_step = 0.001,
                          trex_method = "TREX",
                          method = "LARS",
                          use_omp = TRUE,
                          use_memory_mapping = FALSE,
                          max_outer_threads = 1,
                          max_inner_threads = 1) {
                          
      private$refs <- list(X = X, y = y)
      
      screen_control_list <- list(
        use_bootstrap_CI = use_bootstrap_CI,
        R_boot = R_boot,
        ci_grid_step = ci_grid_step,
        trex_method = trex_method
      )

      trex_control_list <- list(
        method = method,
        parallel_rnd_experiments = use_omp,
        use_memory_mapping = use_memory_mapping,
        max_outer_threads = max_outer_threads,
        max_inner_threads = max_inner_threads
      )
      
      if (inherits(X, "MemoryMappedMatrix")) {
        private$ptr <- trex_screening_mmap_create(
          X$get_ptr(), y, screen_control_list, trex_control_list, seed, verbose
        )
      } else {
        private$ptr <- trex_screening_create(
          X, y, screen_control_list, trex_control_list, seed, verbose
        )
      }
    },
    
    #' @description Run the Screening algorithm.
    select = function() {
      trex_screening_select(private$ptr)
    },
    
    #' @description Get the integer indices (1-based) of the selected variables.
    #' @return Numeric vector of selected indices.
    get_selected_indices = function() {
      trex_screening_get_selected_indices(private$ptr)
    },
    
    #' @description Get generic solver results.
    #' @return A list containing results.
    get_results = function() {
      trex_screening_get_results(private$ptr)
    },
    
    #' @description Get solver matrices.
    #' @return A list containing matrices.
    get_matrices = function() {
      trex_screening_get_matrices(private$ptr)
    }
  ),
  private = list(
    ptr = NULL,
    refs = NULL 
  )
)
