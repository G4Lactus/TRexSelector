#' @name TRexDASelector
#' @title Dependency-Aware T-Rex Selector (TRexDA)
#' @description FDR-controlled variable selection algorithm that handles strong correlations.
#' @importFrom R6 R6Class
#' @export
TRexDASelector <- R6::R6Class("TRexDASelector",
  public = list(
    #' @description Create a new TRexDASelector object.
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param method The solver method, e.g., "LARS", "TACGP", "TENET" (default: "LARS").
    #' @param da_method Dependency method: "BT", "AR1", "EQUI" (default: "BT").
    #' @param cor_coef Correlation coefficient for AR1/EQUI.
    #' @param rho_thr_DA Threshold below which equi-correlation is skipped (default: 0.02).
    #' @param hc_linkage Hierarchical clustering linkage: "Single", "Complete", "Average", "WPGMA", "Ward" (default: "Single").
    #' @param hc_grid_length Number of grid points for BT (default: 0).
    #' @param K Number of random experiments (default: 20).
    #' @param max_dummy_multiplier Max dummy features multiplier (default: 10).
    #' @param use_max_T_stop Whether to bound max T_stop (default: TRUE).
    #' @param opt_threshold Optimization threshold (default: 0.75).
    #' @param lloop_strategy Strategy for L-loop (default: "HCONCAT").
    #' @param tloop_stagnation_stop Whether to stop early if stagnation (default: TRUE).
    #' @param tloop_max_stagnant_steps Steps before stagnation trigger (default: 5).
    #' @param use_omp Whether to use OpenMP parallelization (default: TRUE).
    #' @param use_memory_mapping Use out-of-core memory mapping (default: FALSE).
    #' @param max_outer_threads Thread count for K repetitions (default: 1).
    #' @param max_inner_threads Thread count for inner solvers (default: 1).
    #' @param tol Tolerance for solver algorithms (default: 1e-4).
    #' @param lambda2 ENET hyperparameter (default: 0.1).
    #' @param rho_afs AFS hyperparameter (default: 0.3).
    #' @param ncgmp_variant NCGMP strategy (default: 0).
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          method = "LARS",
                          da_method = "BT",
                          cor_coef = -2.0, # AUTO_ESTIMATE_CORRELATION
                          rho_thr_DA = 0.02,
                          hc_linkage = "Single",
                          hc_grid_length = 0,
                          K = 20,
                          max_dummy_multiplier = 10,
                          use_max_T_stop = TRUE,
                          opt_threshold = 0.75,
                          lloop_strategy = "HCONCAT",
                          tloop_stagnation_stop = TRUE,
                          tloop_max_stagnant_steps = 5,
                          use_omp = TRUE,
                          use_memory_mapping = FALSE,
                          max_outer_threads = 1,
                          max_inner_threads = 1,
                          tol = 1e-4,
                          lambda2 = 0.1,
                          rho_afs = 0.3,
                          ncgmp_variant = 0) {
                          
      private$refs <- list(X = X, y = y)
      
      da_control_list <- list(
        da_method = da_method,
        cor_coef = cor_coef,
        rho_thr_DA = rho_thr_DA,
        hc_linkage = hc_linkage,
        hc_grid_length = hc_grid_length
      )

      trex_control_list <- list(
        method = method,
        K = K,
        max_dummy_multiplier = max_dummy_multiplier,
        use_max_T_stop = use_max_T_stop,
        opt_threshold = opt_threshold,
        lloop_strategy = lloop_strategy,
        tloop_stagnation_stop = tloop_stagnation_stop,
        tloop_max_stagnant_steps = tloop_max_stagnant_steps,
        parallel_rnd_experiments = use_omp,
        use_memory_mapping = use_memory_mapping,
        max_outer_threads = max_outer_threads,
        max_inner_threads = max_inner_threads,
        tol = tol,
        lambda2 = lambda2,
        rho_afs = rho_afs,
        ncgmp_variant = ncgmp_variant
      )
      
      if (inherits(X, "MemoryMappedMatrix")) {
        private$ptr <- trex_da_mmap_create(
          X$get_ptr(), y, tFDR, da_control_list, trex_control_list, seed, verbose
        )
      } else {
        private$ptr <- trex_da_create(
          X, y, tFDR, da_control_list, trex_control_list, seed, verbose
        )
      }
    },
    
    #' @description Run the Dependency-Aware TRexSelector algorithm.
    select = function() {
      trex_da_select(private$ptr)
    },
    
    #' @description Get the integer indices (1-based) of the selected variables.
    #' @return Numeric vector of selected indices.
    get_selected_indices = function() {
      trex_da_get_selected_indices(private$ptr)
    },
    
    #' @description Get generic solver results.
    #' @return A list containing results.
    get_results = function() {
      trex_da_get_results(private$ptr)
    },
    
    #' @description Get DA diagnostic matrices.
    #' @return A list containing structured matrices.
    get_matrices = function() {
      trex_da_get_matrices(private$ptr)
    }
  ),
  private = list(
    ptr = NULL,
    refs = NULL 
  )
)
