# ==============================================================================
# random_experiments.R
#
# Executes K independent T-LARS forward-selection runs and aggregates the
# matrix of relative occurrences phi_T_mat and the terminal vector Phi.
#
# Public interface:
#   random_experiments()   -- main entry point
#
# Internal helpers (not exported):
#   .check_rand_exp()      -- input validation
#   .combine_foreach()     -- list combiner for foreach's .combine
#
# External dependencies (must be on the search path):
#   lm_dummy(), tlars::tlars_model(), tlars::tlars_cpp
# ==============================================================================


# ==============================================================================
# .check_rand_exp
# ==============================================================================

.check_rand_exp <- function(X, y, K, T_stop, num_dummies,
                             parallel_process, parallel_max_cores,
                             lars_state_list) {

  if (!is.matrix(X))              stop("'X' must be a matrix.")
  if (!is.numeric(X))             stop("'X' only allows numerical values.")
  if (anyNA(X))                   stop("'X' contains NAs.")
  if (!is.vector(drop(y)))        stop("'y' must be a vector.")
  if (!is.numeric(y))             stop("'y' only allows numerical values.")
  if (anyNA(y))                   stop("'y' contains NAs.")
  if (nrow(X) != length(drop(y))) stop("nrow(X) != length(y).")

  if (length(K) != 1 || K < 2 || K %% 1 != 0)
    stop("'K' must be an integer >= 2.")
  if (length(num_dummies) != 1 || num_dummies %% 1 != 0 || num_dummies < 1)
    stop("'num_dummies' must be a positive integer.")
  if (length(T_stop) != 1 || !(T_stop %in% seq(1, num_dummies)))
    stop(paste0("'T_stop' must be an integer in [1, ", num_dummies, "]."))

  if (parallel_process) {
    if (length(parallel_max_cores) != 1 ||
        parallel_max_cores %% 1 != 0 ||
        parallel_max_cores < 2)
      stop("'parallel_max_cores' must be an integer >= 2 for parallel processing.")
  }

  if (!missing(lars_state_list) && !is.null(lars_state_list)) {
    if (length(lars_state_list) != K)
      stop("Length of 'lars_state_list' must equal K.")
  }

  invisible(NULL)
}


# ==============================================================================
# .combine_foreach
# ==============================================================================

# List combiner for the foreach .combine argument.
# Merges a growing list-of-lists element-wise across K iterations.
.combine_foreach <- function(x, ...) {
  lapply(seq_along(x), function(i) {
    c(x[[i]], lapply(list(...), function(y) y[[i]]))
  })
}


# ==============================================================================
# random_experiments (main entry point)
# ==============================================================================

#' Run K T-Rex random experiments
#'
#' Executes K independent T-LARS forward selections on X augmented with
#' N(0,1) dummies, stopping each run after T_stop dummies have entered.
#' Returns the matrix of relative occurrences phi_T_mat[j, t] and the
#' terminal occurrence vector Phi[j].
#'
#' @param X               Real valued predictor matrix (n x p).
#' @param y               Response vector (length n).
#' @param K               Number of random experiments (integer >= 2).
#' @param T_stop          Number of dummies included before stopping.
#' @param num_dummies     Number of dummies appended to X.
#' @param type            'lar' (LARS, default) or 'lasso'.
#' @param early_stop      Logical. Stop after T_stop dummies. Default TRUE.
#' @param lars_state_list List of tlars_cpp warm-start states (length K).
#' @param verbose         Logical. Show progress. Default TRUE.
#' @param intercept       Logical. Include intercept. Default FALSE.
#' @param standardize     Logical. Standardize X / centre y. Default TRUE.
#' @param dummy_coef      Logical. Return dummy coefficient matrix. Default FALSE.
#' @param parallel_process  Logical. Run in parallel. Default FALSE.
#' @param parallel_max_cores Max cores.
#' @param seed            RNG seed (parallel mode only).
#' @param eps             Numerical zero. Default .Machine$double.eps.
#'
#' @return Named list: phi_T_mat, Phi, rand_exp_last_betas_mat,
#'   dummy_rand_exp_last_betas_mat, lars_state_list, and metadata.
#'
#' @importFrom parallel detectCores makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach getDoParWorkers registerDoSEQ `%do%` `%dopar%` foreach
#' @importFrom doRNG `%dorng%`
#' @importFrom methods is
#'
#' @export
random_experiments <- function(X,
                               y,
                               K                  = 20,
                               T_stop             = 1,
                               num_dummies        = ncol(X),
                               type               = "lar",
                               early_stop         = TRUE,
                               lars_state_list,
                               verbose            = TRUE,
                               intercept          = FALSE,
                               standardize        = TRUE,
                               dummy_coef         = FALSE,
                               parallel_process   = FALSE,
                               parallel_max_cores = min(K, max(1L, parallel::detectCores(logical = FALSE))),
                               seed               = NULL,
                               eps                = .Machine$double.eps) {

  # ── Validate and coerce inputs ─────────────────────────────────────────────

  type <- match.arg(type, c("lar", "lasso"))
  .check_rand_exp(X, y, K, T_stop, num_dummies,
                  parallel_process, parallel_max_cores,
                  if (missing(lars_state_list)) NULL else lars_state_list)

  # Silently clamp parallel_max_cores
  if (parallel_process) {
    hard_limit <- min(K, max(1L, parallel::detectCores(logical = FALSE)))
    if (parallel_max_cores > hard_limit) {
      message("Setting parallel_max_cores = ", hard_limit, " (physical cores).\n")
      parallel_max_cores <- hard_limit
    }
  }

  p <- ncol(X)

  # Initialise warm-start list if absent
  if (missing(lars_state_list) || is.null(lars_state_list))
    lars_state_list <- vector(mode = "list", length = K)

  # ── Parallel cluster ───────────────────────────────────────────────────────

  if (parallel_process && foreach::getDoParWorkers() == 1) {
    cl <- parallel::makeCluster(parallel_max_cores)
    doParallel::registerDoParallel(cl)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    on.exit(foreach::registerDoSEQ(),  add = TRUE)
  }

  `%par_exe%` <- ifelse(parallel_process, doRNG::`%dorng%`, foreach::`%do%`)

  # ── K random experiments ───────────────────────────────────────────────────

  h <- NULL
  res <- foreach::foreach(
    h             = seq(K),
    .combine      = .combine_foreach,
    .multicombine = TRUE,
    .init         = list(list(), list(), list(), list()),
    .options.RNG  = seed
  ) %par_exe% {

    # Recreate tlars_cpp object after parallel serialisation if necessary
    lars_state <- if (parallel_process &&
                      !is.null(lars_state_list[[h]]) &&
                      !methods::is(lars_state_list[[h]], "tlars_cpp")) {
      tlars::tlars_model(lars_state = lars_state_list[[h]])
    } else {
      lars_state_list[[h]]
    }

    # Run one experiment
    lars_state <- lm_dummy(
      X           = X,
      y           = y,
      model_tlars = lars_state,
      T_stop      = T_stop,
      num_dummies = num_dummies,
      type        = type,
      early_stop  = early_stop,
      verbose     = verbose,
      intercept   = intercept,
      standardize = standardize
    )

    # Solution path: rows = (original vars | dummies), cols = LARS steps
    lars_path      <- do.call(cbind, lars_state$get_beta_path())
    dummy_num_path <- colSums(
      matrix(abs(lars_path[(p + 1):(p + num_dummies), ]) > eps, nrow = num_dummies)
    )

    # Relative occurrence matrix phi_T_mat[j, t] = I(x_j active | T=t) / K
    phi_T_mat <- matrix(0, nrow = p, ncol = T_stop)
    for (c in seq(T_stop)) {
      if (!any(dummy_num_path == c)) {
        ind_sol_path <- length(dummy_num_path)
        warning(paste0("T_stop = ", c, ": LARS exhausted before selecting ", c, " dummies."))
      } else {
        ind_sol_path <- which(as.numeric(dummy_num_path) == c)[1]
      }
      phi_T_mat[, c] <- (1 / K) * (abs(lars_path[1:p, ind_sol_path]) > eps)
    }

    last_betas       <- lars_path[1:p, ncol(lars_path)]
    last_betas_dummy <- if (dummy_coef) lars_path[seq(p + 1, p + num_dummies), ncol(lars_path)] else NULL

    # Serialise for parallel transport
    if (parallel_process) lars_state <- lars_state$get_all()

    list(phi_T_mat, last_betas, lars_state, last_betas_dummy)
  }

  # ── Aggregate across K experiments ────────────────────────────────────────

  lars_state_list         <- res[[3]]
  names(lars_state_list)  <- paste0("lars_state (K=", seq(K), ")")

  phi_T_mat               <- Reduce("+", res[[1]])
  rand_exp_last_betas_mat <- unname(Reduce(rbind, res[[2]]))
  Phi                     <- apply(abs(rand_exp_last_betas_mat) > eps, 2, sum) / K

  dummy_rand_exp_last_betas_mat <- if (dummy_coef) unname(Reduce(rbind, res[[4]])) else NULL

  # ── Return ────────────────────────────────────────────────────────────────

  list(
    phi_T_mat                     = phi_T_mat,
    Phi                           = Phi,
    rand_exp_last_betas_mat       = rand_exp_last_betas_mat,
    dummy_rand_exp_last_betas_mat = dummy_rand_exp_last_betas_mat,
    lars_state_list               = lars_state_list,
    K                             = K,
    T_stop                        = T_stop,
    num_dummies                   = num_dummies,
    type                          = type,
    seed                          = seed,
    eps                           = eps
  )
}
