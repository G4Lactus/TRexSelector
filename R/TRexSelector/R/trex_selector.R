#' @name TRexSelector
#'
#' @title T-Rex Selector
#'
#' @description FDR-controlled variable selection algorithm.
#'
#' @importFrom R6 R6Class
#'
#' @export
TRexSelector <- R6::R6Class("TRexSelector",
  public = list(
    #' @description Create a new TRexSelector object.
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param method The solver method, e.g., "TLARS", "TACGP", "TENET" (default: "TLARS").
    #' @param K Number of random experiments (default: 20).
    #' @param max_dummy_multiplier Max dummy features multiplier (default: 10).
    #' @param use_max_T_stop Whether to bound max T_stop (default: TRUE).
    #' @param opt_threshold Optimization threshold (default: 0.75).
    #' @param lloop_strategy Strategy for L-loop (default: "HCONCAT").
    #' @param tloop_stagnation_stop Whether to stop early if stagnation (default: FALSE).
    #' @param tloop_max_stagnant_steps Steps before stagnation trigger (default: 5).
    #' @param use_omp Whether to use OpenMP parallelization (default: FALSE).
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
                          method = "TLARS",
                          K = 20,
                          max_dummy_multiplier = 10,
                          use_max_T_stop = TRUE,
                          opt_threshold = 0.75,
                          lloop_strategy = "HCONCAT",
                          tloop_stagnation_stop = FALSE,
                          tloop_max_stagnant_steps = 5,
                          use_omp = FALSE,
                          use_memory_mapping = FALSE,
                          max_outer_threads = 1,
                          max_inner_threads = 1,
                          tol = 1e-4,
                          lambda2 = 0.1,
                          rho_afs = 0.3,
                          ncgmp_variant = 0) {

      if (max_dummy_multiplier < 1) {
        stop("max_dummy_multiplier must be >= 1. Got: ", max_dummy_multiplier)
      }
      if (tloop_max_stagnant_steps < 0) {
        stop("tloop_max_stagnant_steps must be >= 0. Got: ", tloop_max_stagnant_steps)
      }
      if (max_outer_threads < 1) {
        stop("max_outer_threads must be >= 1. Got: ", max_outer_threads)
      }
      if (max_inner_threads < 1) {
        stop("max_inner_threads must be >= 1. Got: ", max_inner_threads)
      }

      private$refs <- list(X = X, y = y)

      control_list <- list(
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
        private$ptr <- trex_selector_mmap_create(
          X$get_ptr(), y, tFDR, control_list, seed, verbose
        )
      } else {
        private$ptr <- trex_selector_create(
          X, y, tFDR, control_list, seed, verbose
        )
      }
    },

    #' @description Run the TRexSelector algorithm.
    #' @return A list containing results including selected_var, v_thresh, and selected_indices.
    select = function() {
      trex_selector_select(private$ptr)
      self$get_results()
    },

    #' @description Get a concatenated list of outputs (legacy format).
    #' @return List of selection results
    get_results = function() {
      list(
        selected_var   = self$selected_var,
        v_thresh       = self$v_thresh,
        T_stop         = self$T_stop,
        num_dummies    = self$L * ncol(self$.ptr_X),
        dummy_factor_L = self$L,
        voting_grid    = self$voting_grid,
        selected_indices = self$selected_indices
      )
    }
  ),
  active = list(
    #' @field .ptr Get internal pointer.
    .ptr = function(value) {
      if(!missing(value)) stop(".ptr is read-only")
      private$ptr
    },
    
    #' @field .ptr_X internal proxy for data size
    .ptr_X = function(value) {
       if(!missing(value)) stop(".ptr_X read-only")
       private$refs$X
    },

    #' @field selected_var Logical vector indicating which variables were selected.
    selected_var = function(value) {
      if (!missing(value)) stop("selected_var is read-only.")
      trex_selector_get_selected_var(private$ptr)
    },

    #' @field selected_indices Integer vector of selected 1-based indices.
    selected_indices = function(value) {
      if (!missing(value)) stop("selected_indices is read-only.")
      trex_selector_get_selected_indices(private$ptr)
    },

    #' @field v_thresh The chosen voting threshold.
    v_thresh = function(value) {
      if (!missing(value)) stop("v_thresh is read-only.")
      trex_selector_get_voting_threshold(private$ptr)
    },

    #' @field T_stop The length of the path / maximum number of included variables.
    T_stop = function(value) {
      if (!missing(value)) stop("T_stop is read-only.")
      trex_selector_get_t_stop(private$ptr)
    },

    #' @field L The number of dummy replications used.
    L = function(value) {
      if (!missing(value)) stop("L is read-only.")
      trex_selector_get_dummy_multiplier_l(private$ptr)
    },

    #' @field voting_grid The grid of voters over trials and variables.
    voting_grid = function(value) {
      if (!missing(value)) stop("voting_grid is read-only.")
      trex_selector_get_voting_grid(private$ptr)
    },

    #' @field fdp_hat_mat estimated false discovery proportions
    fdp_hat_mat = function(value) {
        if(!missing(value)) stop("fdp_hat_mat is read-only")
        trex_selector_get_fdp_hat_mat(private$ptr)
    },

    #' @field phi_mat probability mass function evaluations
    phi_mat = function(value) {
        if(!missing(value)) stop("phi_mat is read-only")
        trex_selector_get_phi_mat(private$ptr)
    },

    #' @field r_mat empirical CDF computations
    r_mat = function(value) {
        if(!missing(value)) stop("r_mat is read-only")
        trex_selector_get_r_mat(private$ptr)
    },

    #' @field phi_prime smoothed probability masses
    phi_prime = function(value) {
        if(!missing(value)) stop("phi_prime is read-only")
        trex_selector_get_phi_prime(private$ptr)
    },

    #' @field x_means computed means of the predictor variables
    x_means = function(value) {
        if(!missing(value)) stop("x_means is read-only")
        trex_selector_get_x_means(private$ptr)
    },

    #' @field x_l2_norms computed L2 norms of the scaled predictors
    x_l2_norms = function(value) {
        if(!missing(value)) stop("x_l2_norms is read-only")
        trex_selector_get_x_l2_norms(private$ptr)
    },

    #' @field y_mean computed mean of the response sequence
    y_mean = function(value) {
        if(!missing(value)) stop("y_mean is read-only")
        trex_selector_get_y_mean(private$ptr)
    }
  ),
  private = list(
    ptr = NULL,
    refs = NULL
  )
)
