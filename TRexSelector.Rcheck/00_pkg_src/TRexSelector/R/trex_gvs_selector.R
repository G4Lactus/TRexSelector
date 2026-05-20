#' @name TRexGVSSelector
#' @title T-Rex Selector with Group Variable Selection (TRex GVS)
#' @description FDR-controlled variable selection algorithm utilizing group-based structural augmentation.
#' @importFrom R6 R6Class
#' @export
TRexGVSSelector <- R6::R6Class("TRexGVSSelector",
  public = list(
    #' @description Create a new TRexGVSSelector object.
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param method The solver method (default: "LARS").
    #' @param gvs_type Augmentation policy: "EN", "IEN", "GL", "IGL", "SGL", "ISGL" (default: "EN").
    #' @param corr_max Maximum pairwise correlation for clustering (default: 0.5).
    #' @param hc_linkage Hierarchical clustering linkage method: "Single", "Complete", "Average", "WPGMA" (default: "Single").
    #' @param lambda_2 Lambda2 for grouping.
    #' @param groups User-specified groups (default: NULL).
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
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          method = "LARS",
                          gvs_type = "EN",
                          corr_max = 0.5,
                          hc_linkage = "Single",
                          lambda_2 = 0.0,
                          groups = NULL,
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
                          max_inner_threads = 1) {
                          
      private$refs <- list(X = X, y = y)
      
      gvs_control_list <- list(
        gvs_type = gvs_type,
        corr_max = corr_max,
        hc_linkage = hc_linkage,
        lambda_2 = lambda_2,
        groups = groups
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
        max_inner_threads = max_inner_threads
      )
      
      if (inherits(X, "MemoryMappedMatrix")) {
        private$ptr <- trex_gvs_mmap_create(
          X$get_ptr(), y, tFDR, gvs_control_list, trex_control_list, seed, verbose
        )
      } else {
        private$ptr <- trex_gvs_create(
          X, y, tFDR, gvs_control_list, trex_control_list, seed, verbose
        )
      }
    },
    
    #' @description Run the GVS algorithm.
    select = function() {
      trex_gvs_select(private$ptr)
    },
    
    #' @description Get the integer indices (1-based) of the selected variables.
    #' @return Numeric vector of selected indices.
    get_selected_indices = function() {
      trex_gvs_get_selected_indices(private$ptr)
    },
    
    #' @description Get generic solver results.
    #' @return A list containing results.
    get_results = function() {
      trex_gvs_get_results(private$ptr)
    }
  ),
  private = list(
    ptr = NULL,
    refs = NULL 
  )
)
