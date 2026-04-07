# ==============================================================================
# trex.R
#
# T-Rex Selector -- FDR-controlled forward variable selection.
#
# Public interface:
#   trex()            -- main entry point
#
# Internal helpers (not exported):
#   .check_trex()     -- input validation
#   .compute_fdp()    -- Phi_prime + FDP_hat pipeline
#
# External dependencies (must be on the search path):
#   random_experiments(), Phi_prime_fun(), fdp_hat(), select_var_fun()
# ==============================================================================


# ==============================================================================
# .check_trex
# ==============================================================================

.check_trex <- function(X, y, tFDR, K, max_num_dummies,
                        parallel_process, parallel_max_cores) {

  if (!is.matrix(X))              stop("'X' must be a matrix.")
  if (!is.numeric(X))             stop("'X' only allows numerical values.")
  if (anyNA(X))                   stop("'X' contains NAs.")
  if (!is.vector(drop(y)))        stop("'y' must be a vector.")
  if (!is.numeric(y))             stop("'y' only allows numerical values.")
  if (anyNA(y))                   stop("'y' contains NAs.")
  if (nrow(X) != length(drop(y))) stop("nrow(X) != length(y).")

  if (length(tFDR) != 1 || tFDR < 0 || tFDR > 1)
    stop("'tFDR' must be a number in [0, 1].")
  if (length(K) != 1 || K < 2 || K %% 1 != 0)
    stop("'K' must be an integer >= 2.")
  if (length(max_num_dummies) != 1 ||
      max_num_dummies < 1 ||
      max_num_dummies %% 1 != 0)
    stop("'max_num_dummies' must be a positive integer.")

  if (parallel_process) {
    if (length(parallel_max_cores) != 1 ||
        parallel_max_cores %% 1 != 0 ||
        parallel_max_cores < 2)
      stop("'parallel_max_cores' must be an integer >= 2 for parallel processing.")
  }

  invisible(NULL)
}


# ==============================================================================
# .compute_fdp
# ==============================================================================

#' Compute deflated relative occurrences (Phi_prime) and the conservative
#' FDP estimate vector over the full voting grid V.
#'
#' @return Named list: Phi_prime (length-p vector), FDP_hat (length-V vector).
.compute_fdp <- function(p, T_stop, num_dummies, phi_T_mat, Phi, V, eps) {

  Phi_prime <- Phi_prime_fun(
    p           = p,
    T_stop      = T_stop,
    num_dummies = num_dummies,
    phi_T_mat   = phi_T_mat,
    Phi         = Phi,
    eps         = eps
  )

  FDP_hat <- fdp_hat(V = V, Phi = Phi, Phi_prime = Phi_prime)

  list(Phi_prime = Phi_prime, FDP_hat = FDP_hat)
}


# ==============================================================================
# trex (main entry point)
# ==============================================================================

#' Run the T-Rex selector
#'
#' Performs fast FDR-controlled variable selection in high-dimensional
#' linear models via K terminating random experiments (T-LARS paths on
#' X augmented with N(0,1) dummies), followed by a two-loop calibration
#' that jointly optimises the stopping threshold T* and voting level v*.
#'
#' @param X               Real valued predictor matrix (n x p).
#' @param y               Response vector (length n).
#' @param tFDR            Target FDR level in [0, 1]. Default 0.2.
#' @param K               Number of random experiments (integer >= 2). Default 20.
#' @param max_num_dummies Maximum dummies as a multiple of p. Default 10.
#' @param max_T_stop      If TRUE, cap T* at ceiling(n / 2). Default TRUE.
#' @param type            'lar' (LARS, default) or 'lasso'.
#' @param parallel_process  Logical. Run experiments in parallel. Default FALSE.
#' @param parallel_max_cores Max cores for parallel execution.
#' @param seed            RNG seed (parallel mode only).
#' @param eps             Numerical zero. Default .Machine$double.eps.
#' @param verbose         Print progress. Default TRUE.
#'
#' @return Named list:
#'   \item{selected_var}{0/1 support vector of length p.}
#'   \item{tFDR}{Target FDR.}
#'   \item{T_stop}{Optimal T*.}
#'   \item{num_dummies}{Number of dummies used.}
#'   \item{V}{Voting level grid.}
#'   \item{v_thresh}{Optimal voting threshold v*.}
#'   \item{Phi_prime}{Deflated relative occurrences at the final step.}
#'   \item{type}{Forward selection type used.}
#'   \item{FDP_hat_mat}{FDP estimates matrix (T x |V|).}
#'   \item{Phi_mat}{Relative occurrences matrix (T x p).}
#'   \item{R_mat}{Number of selected variables matrix (T x |V|).}
#'   \item{phi_T_mat}{Raw phi_T matrix from the final random experiments.}
#'
#' @importFrom parallel detectCores makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach getDoParWorkers registerDoSEQ
#'
#' @export
#'
#' @examples
#' data("Gauss_data")
#' X <- Gauss_data$X
#' y <- c(Gauss_data$y)
#' set.seed(1234)
#' res          <- trex(X = X, y = y)
#' selected_var <- res$selected_var
trex <- function(X,
                 y,
                 tFDR               = 0.2,
                 K                  = 20,
                 max_num_dummies    = 10,
                 max_T_stop         = TRUE,
                 type               = "lar",
                 parallel_process   = FALSE,
                 parallel_max_cores = min(K, max(1L, parallel::detectCores(logical = FALSE))),
                 seed               = NULL,
                 eps                = .Machine$double.eps,
                 verbose            = TRUE) {

  # ── Validate and coerce inputs ─────────────────────────────────────────────

  type <- match.arg(type, c("lar", "lasso"))
  .check_trex(X, y, tFDR, K, max_num_dummies, parallel_process, parallel_max_cores)

  # Silently clamp parallel_max_cores to available physical cores
  if (parallel_process) {
    hard_limit <- min(K, max(1L, parallel::detectCores(logical = FALSE)))
    if (parallel_max_cores > hard_limit) {
      message("Setting parallel_max_cores = ", hard_limit, " (physical cores).\n")
      parallel_max_cores <- hard_limit
    }
  }

  # ── Dimensions and preprocessing ──────────────────────────────────────────

  n <- nrow(X)
  p <- ncol(X)

  if (p > 1e4) warning("Feature space > 10,000 variables. RAM issues may occur.")

  X <- scale(X)
  y <- y - mean(y)

  # ── Voting grid and calibration reference accessors ────────────────────────

  V     <- seq(0.5, 1 - eps, by = 1 / K)
  V_len <- length(V)

  opt_point <- which(abs(V - 0.75) < eps)
  if (length(opt_point) == 0) opt_point <- length(V[V < 0.75])

  # FDP at the 75% calibration reference (L-loop exit condition)
  fdp_ref  <- function(FDP_hat) FDP_hat[opt_point]
  # FDP at the most conservative voting level (T-loop exit condition)
  fdp_edge <- function(FDP_hat) FDP_hat[V_len]

  # ── Parallel cluster ───────────────────────────────────────────────────────

  if (parallel_process && foreach::getDoParWorkers() == 1) {
    cl <- parallel::makeCluster(parallel_max_cores)
    doParallel::registerDoParallel(cl)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    on.exit(foreach::registerDoSEQ(),  add = TRUE)
  }

  # ── Closure: one full calibration step (K experiments + FDP pipeline) ─────

  .run_step <- function(T_stop, num_dummies, lars_state_list = NULL) {
    suppressWarnings(
      rand_exp <- random_experiments(
        X                  = X,
        y                  = y,
        K                  = K,
        T_stop             = T_stop,
        num_dummies        = num_dummies,
        type               = type,
        early_stop         = TRUE,
        lars_state_list    = lars_state_list,
        verbose            = verbose,
        intercept          = FALSE,
        standardize        = TRUE,
        parallel_process   = parallel_process,
        parallel_max_cores = parallel_max_cores,
        seed               = seed,
        eps                = eps
      )
    )

    fdp <- .compute_fdp(p, T_stop, num_dummies,
                        rand_exp$phi_T_mat, rand_exp$Phi, V, eps)

    list(
      rand_exp  = rand_exp,
      phi_T_mat = rand_exp$phi_T_mat,
      Phi       = rand_exp$Phi,
      Phi_prime = fdp$Phi_prime,
      FDP_hat   = fdp$FDP_hat
    )
  }

  # ── L-loop: find sufficient num_dummies ────────────────────────────────────

  LL      <- 1L
  T_stop  <- 1L
  FDP_hat <- rep(NA_real_, V_len)

  while ((LL <= max_num_dummies && fdp_ref(FDP_hat) > tFDR) ||
         sum(!is.na(FDP_hat)) == 0) {

    num_dummies <- LL * p
    LL          <- LL + 1L
    step        <- .run_step(T_stop, num_dummies)
    FDP_hat     <- step$FDP_hat

    if (verbose) cat("\n Appended dummies:", num_dummies, "\n")
  }

  # ── T-loop: extend T_stop while FDP is at or below target ─────────────────

  acc   <- list(FDP = matrix(FDP_hat,  nrow = 1),
                Phi = matrix(step$Phi, nrow = 1))
  max_T <- if (max_T_stop) min(num_dummies, ceiling(n / 2)) else num_dummies

  if (!is.null(seed)) seed <- seed + 12345L

  while (fdp_edge(FDP_hat) <= tFDR && T_stop < max_T) {

    T_stop  <- T_stop + 1L
    step    <- .run_step(T_stop, num_dummies, step$rand_exp$lars_state_list)
    FDP_hat <- step$FDP_hat
    acc     <- list(FDP = rbind(acc$FDP, FDP_hat),
                    Phi = rbind(acc$Phi, step$Phi))
  }

  if (verbose) cat("\n Included dummies before stopping:", T_stop, "\n")

  # ── Variable selection: maximise |Â| subject to FDP_hat <= tFDR ───────────

  sel <- select_var_fun(
    p           = p,
    tFDR        = tFDR,
    T_stop      = T_stop,
    FDP_hat_mat = acc$FDP,
    Phi_mat     = acc$Phi,
    V           = V
  )

  # ── Assemble and return ────────────────────────────────────────────────────

  c(
    list(
      selected_var = sel$selected_var,
      tFDR         = tFDR,
      T_stop       = T_stop,
      num_dummies  = num_dummies,
      V            = V,
      v_thresh     = sel$v_thresh,
      Phi_prime    = step$Phi_prime,
      type         = type
    ),
    list(
      FDP_hat_mat = acc$FDP,
      Phi_mat     = acc$Phi,
      R_mat       = sel$R_mat,
      phi_T_mat   = step$phi_T_mat
    )
  )
}
