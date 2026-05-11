# =============================================================================
# trex_screening.R — Screen-T-Rex Selector
#
# This file consolidates all functions required by the Screen-T-Rex selector
# (doi:10.1109/SSP53291.2023.10207957) from the T-Rex reference codebase.
#
# Dependency order (bottom-up):
#   1. add_dummies             – append random dummy predictors to X
#   2. lm_dummy                – run one T-LARS random experiment
#   3. random_experiments      – run K random experiments, collect phi_T_mat
#
#   Private helpers (screen_trex only):
#   .estimate_cor_coef         – data-driven AR1 / equi correlation estimate
#   .ar1_neighbors             – sliding-window neighbor indices for DA+AR1
#   .da_delta_mat              – dependency-awareness delta matrix
#   .da_adjust                 – apply DA correction to phi_T_mat and Phi
#   .screen_majority_vote      – majority-vote variable screening
#   .screen_bootstrap          – bootstrap variable screening
#
#   4. screen_trex             – top-level Screen-T-Rex selector
#
# Functions NOT included (belong to the full T-Rex calibration only):
#   fdp_hat, Phi_prime_fun, select_var_fun, FDP, TPP
# =============================================================================


# =============================================================================
# 1. add_dummies
# =============================================================================

#' Add dummy predictors to the original predictor matrix
#'
#' Sample num_dummies dummy predictors from the univariate standard normal
#' distribution and append them to the predictor matrix X.
#'
#' @param X          Real valued predictor matrix.
#' @param num_dummies Number of dummies that are appended to the predictor matrix.
#'
#' @return Enlarged predictor matrix [X | D] of dimension n x (p + num_dummies).
#'
#' @importFrom stats rnorm
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' n <- 50
#' p <- 100
#' X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
#' add_dummies(X = X, num_dummies = p)
add_dummies <- function(X,
                        num_dummies) {
  # Error control
  if (!is.matrix(X)) {
    stop("'X' must be a matrix.")
  }
  if (!is.numeric(X)) {
    stop("'X' only allows numerical values.")
  }
  if (anyNA(X)) {
    stop("'X' contains NAs. Please remove or impute them before proceeding.")
  }
  if (length(num_dummies) != 1 ||
      num_dummies %% 1 != 0 ||
      num_dummies < 1) {
    stop("'num_dummies' must be an integer larger or equal to 1.")
  }

  # Number of rows of X
  n <- nrow(X)

  # Create matrix of dummy predictors
  dummies <- matrix(
    stats::rnorm(n * num_dummies),
    nrow = n,
    ncol = num_dummies,
    byrow = FALSE
  )

  # Append dummies to X
  X_Dummy <- cbind(X, dummies)

  return(X_Dummy)
}


# =============================================================================
# 2. lm_dummy
# =============================================================================

#' Perform one random experiment
#'
#' Run one random experiment of the T-Rex selector
#' (doi:10.48550/arXiv.2110.06048): generates dummies, appends them to the
#' predictor matrix, and runs the forward selection algorithm until it is
#' terminated after T_stop dummies have been selected.
#'
#' @param X            Real valued predictor matrix.
#' @param y            Response vector.
#' @param model_tlars  Object of the class tlars_cpp. Contains all state
#'                     variables of the previous T-LARS step (warm-start).
#' @param T_stop       Number of included dummies after which the experiment
#'                     is stopped.
#' @param num_dummies  Number of dummies appended to the predictor matrix.
#' @param method       'trex'          T-Rex selector (doi:10.48550/arXiv.2110.06048)
#'                     'trex+GVS'      T-Rex+GVS selector (doi:10.23919/EUSIPCO55093.2022.9909883)
#'                     'trex+DA+AR1'   T-Rex+DA+AR1 selector
#'                     'trex+DA+equi'  T-Rex+DA+equi selector
#'                     'trex+DA+BT'    T-Rex+DA+BT selector (doi:10.48550/arXiv.2401.15796)
#'                     'trex+DA+NN'    T-Rex+DA+NN selector (doi:10.48550/arXiv.2401.15139)
#' @param GVS_type     'IEN' Informed Elastic Net (doi:10.1109/CAMSAP58249.2023.10403489)
#'                     'EN'  ordinary Elastic Net (doi:10.1111/j.1467-9868.2005.00503.x)
#' @param type         'lar' for LARS, 'lasso' for Lasso.
#' @param corr_max     Maximum allowed correlation between predictors from
#'                     different clusters (method = 'trex+GVS').
#' @param lambda_2_lars lambda_2-value for LARS-based Elastic Net.
#' @param early_stop   Logical. If TRUE stop after T_stop dummies; otherwise
#'                     compute the entire solution path.
#' @param verbose      Logical. If TRUE show progress.
#' @param intercept    Logical. If TRUE include an intercept.
#' @param standardize  Logical. If TRUE standardize predictors, center response.
#'
#' @return Object of the class tlars_cpp.
#'
#' @importFrom tlars tlars_model tlars tlars_cpp
#' @importFrom glmnet cv.glmnet
#' @importFrom stats rnorm
#' @importFrom methods is
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' eps <- .Machine$double.eps
#' n <- 75
#' p <- 100
#' X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
#' beta <- c(rep(3, times = 3), rep(0, times = 97))
#' y <- X %*% beta + rnorm(n)
#' res <- lm_dummy(X = X, y = y, T_stop = 1, num_dummies = 5 * p)
#' beta_hat <- res$get_beta()[seq(p)]
#' support <- abs(beta_hat) > eps
#' support
lm_dummy <- function(X,
                     y,
                     model_tlars,
                     T_stop       = 1,
                     num_dummies  = ncol(X),
                     method       = "trex",
                     GVS_type     = "IEN",
                     type         = "lar",
                     corr_max     = 0.5,
                     lambda_2_lars = NULL,
                     early_stop   = TRUE,
                     verbose      = TRUE,
                     intercept    = FALSE,
                     standardize  = TRUE) {

  # Numerical zero
  eps <- .Machine$double.eps

  # Error control
  method   <- match.arg(method,   c("trex", "trex+GVS", "trex+DA+AR1",
                                     "trex+DA+equi", "trex+DA+BT", "trex+DA+NN"))
  type     <- match.arg(type,     c("lar", "lasso"))
  GVS_type <- match.arg(GVS_type, c("IEN", "EN"))

  if (!is.matrix(X))          stop("'X' must be a matrix.")
  if (!is.numeric(X))         stop("'X' only allows numerical values.")
  if (anyNA(X))               stop("'X' contains NAs. Please remove or impute them before proceeding.")
  if (!is.vector(drop(y)))    stop("'y' must be a vector.")
  if (!is.numeric(y))         stop("'y' only allows numerical values.")
  if (anyNA(y))               stop("'y' contains NAs. Please remove or impute them before proceeding.")
  if (nrow(X) != length(drop(y))) stop("Number of rows in X does not match length of y.")

  if (!(missing(model_tlars) || is.null(model_tlars))) {
    if (!methods::is(object = model_tlars, class2 = tlars::tlars_cpp)) {
      stop("'model_tlars' must be an object of class tlars_cpp.")
    }
  }

  if (method %in% c("trex", "trex+DA+AR1", "trex+DA+equi", "trex+DA+BT", "trex+DA+NN")) {
    if (length(num_dummies) != 1 || num_dummies %% 1 != 0 || num_dummies < 1) {
      stop("'num_dummies' must be an integer larger or equal to 1.")
    }
  }

  if (method == "trex+GVS") {
    if (length(num_dummies) != 1 ||
        num_dummies %% ncol(X) != 0 ||
        num_dummies < 1) {
      stop("'num_dummies' must be a positive integer multiple of the total number of original predictors in X.")
    }
  }

  if (length(T_stop) != 1 || !(T_stop %in% seq(1, num_dummies))) {
    stop(paste0("Value of 'T_stop' not valid. 'T_stop' must be an integer from 1 to ", num_dummies, "."))
  }

  if (method == "trex+GVS") {
    if (length(corr_max) != 1 || corr_max < 0 || corr_max > 1) {
      stop("'corr_max' must have a value between zero and one.")
    }
  }

  if (!is.null(lambda_2_lars)) {
    if (length(lambda_2_lars) != 1 || lambda_2_lars < eps) {
      stop("'lambda_2_lars' must be a number larger than zero.")
    }
  }

  # -------------------------------------------------------------------------
  # Create T-LARS model if not supplied (i.e., no warm-start)
  # -------------------------------------------------------------------------
  if (T_stop == 1 ||
      missing(model_tlars) ||
      is.null(model_tlars)) {

    if (method %in% c("trex", "trex+DA+AR1", "trex+DA+equi", "trex+DA+BT", "trex+DA+NN")) {
      X_Dummy <- add_dummies(X = X, num_dummies = num_dummies)
    } else {
      # trex+GVS
      GVS_dummies <- add_dummies_GVS(X = X, num_dummies = num_dummies, corr_max = corr_max)
      X_Dummy     <- GVS_dummies$X_Dummy

      # Ridge regression to determine lambda_2 for elastic net
      if (is.null(lambda_2_lars)) {
        n_col <- ncol(X)
        alpha <- 0
        cvfit <- glmnet::cv.glmnet(
          x            = X,
          y            = y,
          intercept    = intercept,
          standardize  = standardize,
          alpha        = alpha,
          type.measure = "mse",
          family       = "gaussian",
          nfolds       = 10
        )
        lambda_2_glmnet <- cvfit$lambda.1se
        lambda_2_lars   <- lambda_2_glmnet * n_col * (1 - alpha) / 2
      }

      # Data modification for Elastic Net (EN)
      if (GVS_type == "EN") {
        p_dummy <- ncol(X_Dummy)
        X_Dummy <- (1 / sqrt(1 + lambda_2_lars)) *
          rbind(X_Dummy,
                diag(rep(sqrt(lambda_2_lars), times = p_dummy)))
        y <- append(y, rep(0, times = p_dummy))
      }

      # Data modification for Informed Elastic Net (IEN)
      if (GVS_type == "IEN") {
        p_orig           <- ncol(X)
        p_dummy          <- ncol(X_Dummy)
        max_clusters     <- GVS_dummies$max_clusters
        cluster_sizes    <- GVS_dummies$cluster_sizes
        IEN_cl_id_vectors <- GVS_dummies$IEN_cl_id_vectors
        X_Dummy <- sqrt(lambda_2_lars) *
          rbind(
            (1 / sqrt(lambda_2_lars)) * X_Dummy,
            (1 / sqrt(cluster_sizes)) *
              matrix(rep(IEN_cl_id_vectors, times = p_dummy / p_orig),
                     ncol = ncol(IEN_cl_id_vectors) * p_dummy / p_orig)
          )
        y <- append(y, rep(0, times = max_clusters))
      }
    }

    # Scale data
    X_Dummy <- scale(X_Dummy)
    y       <- y - mean(y)

    # Create new T-LARS model
    model_tlars <- tlars::tlars_model(
      X           = X_Dummy,
      y           = y,
      num_dummies = num_dummies,
      verbose     = FALSE,
      info        = FALSE,
      intercept   = intercept,
      standardize = standardize,
      type        = type
    )

  }

  # -------------------------------------------------------------------------
  # Execute T-LARS step (new model or warm-start)
  # -------------------------------------------------------------------------
  tlars::tlars(
    model      = model_tlars,
    T_stop     = T_stop,
    early_stop = early_stop,
    info       = FALSE
  )

  return(model_tlars)
}


# =============================================================================
# 3. random_experiments
# =============================================================================

#' Run K random experiments
#'
#' Run K early-terminated T-Rex (doi:10.48550/arXiv.2110.06048) random
#' experiments and compute the matrix of relative occurrences for all
#' variables and all numbers of included dummies before stopping.
#'
#' @param X                   Real valued predictor matrix.
#' @param y                   Response vector.
#' @param K                   Number of random experiments.
#' @param T_stop              Number of included dummies after which each
#'                            experiment is stopped.
#' @param num_dummies         Number of dummies appended to the predictor matrix.
#' @param method              See lm_dummy for details.
#' @param GVS_type            'IEN' or 'EN' (see lm_dummy).
#' @param type                'lar' or 'lasso'.
#' @param corr_max            Maximum allowed correlation between predictors
#'                            from different clusters (method = 'trex+GVS').
#' @param lambda_2_lars       lambda_2-value for LARS-based Elastic Net.
#' @param early_stop          Logical. If TRUE stop after T_stop dummies.
#' @param lars_state_list     List of T-LARS state objects for warm-starts.
#' @param verbose             Logical. If TRUE show progress.
#' @param intercept           Logical. If TRUE include an intercept.
#' @param standardize         Logical. If TRUE standardize predictors.
#' @param dummy_coef          Logical. If TRUE also return terminal dummy
#'                            coefficient matrix.
#' @param parallel_process    Logical. If TRUE run experiments in parallel.
#' @param parallel_max_cores  Maximum number of cores for parallel processing.
#' @param seed                RNG seed (only used when parallel_process = TRUE).
#' @param eps                 Numerical zero.
#'
#' @return List containing the results of the K random experiments:
#'   phi_T_mat, rand_exp_last_betas_mat, dummy_rand_exp_last_betas_mat,
#'   Phi, lars_state_list, K, T_stop, num_dummies, method, GVS_type, type,
#'   corr_max, lambda_2_lars, seed, eps.
#'
#' @importFrom parallel detectCores makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach getDoParWorkers registerDoSEQ `%do%` `%dopar%` foreach
#' @importFrom doRNG `%dorng%`
#' @importFrom methods is
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' data("Gauss_data")
#' X <- Gauss_data$X
#' y <- c(Gauss_data$y)
#' res <- random_experiments(X = X, y = y)
#' relative_occurrences_matrix <- res$phi_T_mat
#' relative_occurrences_matrix
random_experiments <- function(X,
                               y,
                               K                  = 20,
                               T_stop             = 1,
                               num_dummies        = ncol(X),
                               method             = "trex",
                               GVS_type           = "EN",
                               type               = "lar",
                               corr_max           = 0.5,
                               lambda_2_lars      = NULL,
                               early_stop         = TRUE,
                               lars_state_list,
                               verbose            = TRUE,
                               intercept          = FALSE,
                               standardize        = TRUE,
                               dummy_coef         = FALSE,
                               parallel_process   = FALSE,
                               parallel_max_cores = min(K, max(1, parallel::detectCores(logical = FALSE))),
                               seed               = NULL,
                               eps                = .Machine$double.eps) {

  # Error control
  method   <- match.arg(method,   c("trex", "trex+GVS", "trex+DA+AR1",
                                     "trex+DA+equi", "trex+DA+BT", "trex+DA+NN"))
  type     <- match.arg(type,     c("lar", "lasso"))
  GVS_type <- match.arg(GVS_type, c("IEN", "EN"))

  if (!is.matrix(X))          stop("'X' must be a matrix.")
  if (!is.numeric(X))         stop("'X' only allows numerical values.")
  if (anyNA(X))               stop("'X' contains NAs. Please remove or impute them before proceeding.")
  if (!is.vector(drop(y)))    stop("'y' must be a vector.")
  if (!is.numeric(y))         stop("'y' only allows numerical values.")
  if (anyNA(y))               stop("'y' contains NAs. Please remove or impute them before proceeding.")
  if (nrow(X) != length(drop(y))) stop("Number of rows in X does not match length of y.")
  if (length(K) != 1 || K < 2 || K %% 1 != 0) {
    stop("The number of random experiments 'K' must be an integer larger or equal to 2.")
  }

  # Number of variables in X
  p <- ncol(X)

  if (method %in% c("trex", "trex+DA+AR1", "trex+DA+equi", "trex+DA+BT", "trex+DA+NN")) {
    if (length(num_dummies) != 1 || num_dummies %% 1 != 0 || num_dummies < 1) {
      stop("'num_dummies' must be an integer larger or equal to 1.")
    }
  }

  if (method == "trex+GVS") {
    if (length(num_dummies) != 1 || num_dummies %% p != 0 || num_dummies < 1) {
      stop("'num_dummies' must be a positive integer multiple of the total number of original predictors in X.")
    }
  }

  if (length(T_stop) != 1 || !(T_stop %in% seq(1, num_dummies))) {
    stop(paste0("Value of 'T_stop' not valid. 'T_stop' must be an integer from 1 to ", num_dummies, "."))
  }

  if (method == "trex+GVS") {
    if (length(corr_max) != 1 || corr_max < 0 || corr_max > 1) {
      stop("'corr_max' must have a value between zero and one.")
    }
  }

  if (!is.null(lambda_2_lars)) {
    if (length(lambda_2_lars) != 1 || lambda_2_lars < eps) {
      stop("'lambda_2_lars' must be a number larger than zero.")
    }
  }

  if (parallel_process &&
      (length(parallel_max_cores) != 1 ||
       parallel_max_cores %% 1 != 0 ||
       parallel_max_cores < 2)) {
    stop("For parallel processing at least two workers have to be registered:
'parallel_max_cores' must be an integer larger or equal to 2.")
  }

  if (parallel_process &&
      parallel_max_cores > min(K, max(1, parallel::detectCores(logical = FALSE)))) {
    parallel_max_cores <- min(K, max(1, parallel::detectCores(logical = FALSE)))
    message(paste0(
      "For computing ", K, " random experiments, it is not useful/possible to register ",
      parallel_max_cores, " workers. Setting parallel_max_cores = ",
      min(K, max(1, parallel::detectCores(logical = FALSE))),
      " (# physical cores) ...
"
    ))
  }

  if (parallel_process && T_stop == 1 && num_dummies <= p) {
    message(
      "Computing random experiments in parallel...
",
      "Note that this is only advantageous if you have at least a few thousand predictors
",
      "and/or data points in 'X'. Otherwise, the overhead will slow down the computations
",
      "in parallel. Thus, for small data sizes it is better to set parallel_process = FALSE.

",
      "Be careful!"
    )
  }

  if (!(missing(lars_state_list) || is.null(lars_state_list))) {
    if (length(lars_state_list) != K) {
      stop("Length of 'lars_state_list' must be equal to number of random experiments 'K'.")
    }
  }

  # Create empty lars_state_list if missing or NULL
  if (missing(lars_state_list) || is.null(lars_state_list)) {
    lars_state_list <- vector(mode = "list", length = K)
  }

  # Helper: combines output lists of parallel foreach loop
  comb_fun <- function(x, ...) {
    lapply(
      seq_along(x),
      function(i) {
        c(x[[i]], lapply(list(...), function(y) y[[i]]))
      }
    )
  }

  # Setup cluster for parallel processing
  if (parallel_process && foreach::getDoParWorkers() == 1) {
    cl <- parallel::makeCluster(parallel_max_cores)
    doParallel::registerDoParallel(cl)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    on.exit(foreach::registerDoSEQ(),  add = TRUE)
  }

  `%par_exe%` <- ifelse(parallel_process, doRNG::`%dorng%`, foreach::`%do%`)

  h <- NULL
  res <- foreach::foreach(
    h             = seq(K),
    .combine      = comb_fun,
    .multicombine = TRUE,
    .init         = list(list(), list(), list(), list()),
    .options.RNG  = seed
  ) %par_exe% {

    # Recreate tlars_cpp object from state list if running in parallel
    if (parallel_process &&
        !is.null(lars_state_list[[h]]) &&
        !methods::is(object = lars_state_list[[h]], class2 = tlars::tlars_cpp)) {
      lars_state <- tlars::tlars_model(lars_state = lars_state_list[[h]])
    } else {
      lars_state <- lars_state_list[[h]]
    }

    # Run one random experiment
    lars_state <- lm_dummy(
      X             = X,
      y             = y,
      model_tlars   = lars_state,
      T_stop        = T_stop,
      num_dummies   = num_dummies,
      method        = method,
      GVS_type      = GVS_type,
      type          = type,
      corr_max      = corr_max,
      lambda_2_lars = lambda_2_lars,
      early_stop    = early_stop,
      verbose       = verbose,
      intercept     = intercept,
      standardize   = standardize
    )

    # Extract T-LARS solution path (columns = steps, rows = predictors + dummies)
    lars_path <- do.call(cbind, lars_state$get_beta_path())

    # Serialize state object for parallel transport
    if (parallel_process) {
      lars_state <- lars_state$get_all()
    }

    # Number of included dummies along solution path
    dummy_num_path <- colSums(matrix(
      abs(lars_path[(p + 1):(p + num_dummies), ]) > eps,
      nrow = num_dummies,
      ncol = ncol(lars_path)
    ))

    # Relative occurrences: phi_T_mat[j, t] = P(variable j active | T = t dummies included)
    phi_T_mat <- matrix(0, nrow = p, ncol = T_stop)
    for (c in seq(T_stop)) {
      if (!any(dummy_num_path == c)) {
        ind_sol_path <- length(dummy_num_path)
        warning(paste0(
          "For T_stop = ", c,
          " LARS is running until k = min(n, p) and stops there before selecting ", c, " dummies."
        ))
      } else {
        ind_sol_path <- which(as.numeric(dummy_num_path) == c)[1]
      }
      phi_T_mat[, c] <- (1 / K) * (abs(lars_path[1:p, ind_sol_path]) > eps)
    }

    # Terminal coefficient vectors of original variables and dummies
    rand_exp_last_betas_mat <- lars_path[1:p, ncol(lars_path)]

    if (dummy_coef) {
      dummy_rand_exp_last_betas_mat <- lars_path[seq(p + 1, p + num_dummies), ncol(lars_path)]
    } else {
      dummy_rand_exp_last_betas_mat <- NULL
    }

    list(phi_T_mat, rand_exp_last_betas_mat, lars_state, dummy_rand_exp_last_betas_mat)
  }

  # -------------------------------------------------------------------------
  # Merge results across all K experiments
  # -------------------------------------------------------------------------
  lars_state_list <- res[[3]]
  names(lars_state_list) <- paste0("lars_state (K = ", seq(K), ")")

  phi_T_mat               <- Reduce("+", res[[1]])
  rand_exp_last_betas_mat <- unname(Reduce(rbind, res[[2]]))

  # Relative occurrences at T = T_stop (fraction of experiments in which
  # each variable was active when the T_stop-th dummy entered the model)
  Phi <- apply(abs(rand_exp_last_betas_mat) > eps, 2, sum) / K

  if (dummy_coef) {
    dummy_rand_exp_last_betas_mat <- unname(Reduce(rbind, res[[4]]))
  } else {
    dummy_rand_exp_last_betas_mat <- NULL
  }

  rand_exp_res <- list(
    phi_T_mat                    = phi_T_mat,
    rand_exp_last_betas_mat      = rand_exp_last_betas_mat,
    dummy_rand_exp_last_betas_mat = dummy_rand_exp_last_betas_mat,
    Phi                          = Phi,
    lars_state_list              = lars_state_list,
    K                            = K,
    T_stop                       = T_stop,
    num_dummies                  = num_dummies,
    method                       = method,
    GVS_type                     = GVS_type,
    type                         = type,
    corr_max                     = corr_max,
    lambda_2_lars                = lambda_2_lars,
    seed                         = seed,
    eps                          = eps
  )

  return(rand_exp_res)
}

# =============================================================================
# Helpers for screen_trex (package-private, prefixed with a dot)
# =============================================================================

# -----------------------------------------------------------------------------
# .estimate_cor_coef
#
# Estimates the correlation coefficient required by the T-Rex+DA variants:
#   - trex+DA+AR1  → AR(1) autocorrelation of each row of X (averaged over rows)
#   - trex+DA+equi → mean pairwise correlation of the columns of X
#
# @param X      Scaled predictor matrix (n x p).
# @param method One of 'trex+DA+AR1' or 'trex+DA+equi'.
#
# @return Scalar correlation coefficient.
# -----------------------------------------------------------------------------
.estimate_cor_coef <- function(X, method) {
  if (method == "trex+DA+AR1") {
    abs(mean(apply(X, 1L, function(row) {
      stats::coef(
        stats::arima(row, order = c(1L, 0L, 0L),
                     include.mean = FALSE, method = "ML")
      )
    })))
  } else {
    # trex+DA+equi
    cor_mat  <- stats::cor(X)
    mean(cor_mat[lower.tri(cor_mat, diag = FALSE)])
  }
}


# -----------------------------------------------------------------------------
# .ar1_neighbors
#
# Returns the indices of the sliding-window neighbors of variable j under an
# AR(1) structure with bandwidth kap = ceil(log(rho_thr) / log(rho)).
#
# Avoids the self-reference that arises from naive max/min clipping.
#
# @param j    Variable index (1-based).
# @param p    Total number of variables.
# @param kap  Window half-width.
#
# @return Integer vector of neighbor indices (never contains j).
# -----------------------------------------------------------------------------
.ar1_neighbors <- function(j, p, kap) {
  left  <- if (j > 1L) seq.int(max(1L, j - kap), j - 1L) else integer(0L)
  right <- if (j < p)  seq.int(j + 1L, min(p, j + kap))  else integer(0L)
  c(left, right)
}


# -----------------------------------------------------------------------------
# .da_delta_mat
#
# Computes the (p x T_stop) dependency-awareness delta matrix used to down-
# weight relative occurrences of variables that co-occur with their neighbors.
#
# The correction is based on:
#   delta[j, t] = 2 - min_{ k in neighbors(j) }  |phi[j,t] - phi[k,t]|
#
# Values close to 2 (neighbors have very different phi) → little correction.
# Values close to 1 (one neighbor has nearly the same phi) → strong correction.
#
# @param phi_T_mat   p x T_stop matrix of relative occurrences.
# @param p           Number of candidate variables.
# @param T_stop      Stopping index (number of dummy columns).
# @param method      'trex+DA+AR1' or 'trex+DA+equi'.
# @param cor_coef    Estimated correlation coefficient.
# @param rho_thr_DA  Correlation threshold below which no correction is applied
#                    (equi only).
#
# @return p x T_stop numeric matrix, or NULL when no correction is needed.
# -----------------------------------------------------------------------------
.da_delta_mat <- function(phi_T_mat, p, T_stop, method, cor_coef, rho_thr_DA) {

  if (method == "trex+DA+equi" && abs(cor_coef) <= rho_thr_DA) {
    return(NULL)   # correlation below threshold → no adjustment
  }

  delta <- matrix(NA_real_, nrow = p, ncol = T_stop)

  if (method == "trex+DA+AR1") {
    kap <- ceiling(log(rho_thr_DA) / log(cor_coef))

    for (t in seq_len(T_stop)) {
      for (j in seq_len(p)) {
        nb         <- .ar1_neighbors(j, p, kap)
        delta[j, t] <- 2 - min(abs(phi_T_mat[j, t] - phi_T_mat[nb, t]))
      }
    }

  } else {
    # trex+DA+equi with |cor_coef| > rho_thr_DA: all other variables as neighbors
    for (t in seq_len(T_stop)) {
      for (j in seq_len(p)) {
        nb         <- seq_len(p)[-j]
        delta[j, t] <- 2 - min(abs(phi_T_mat[j, t] - phi_T_mat[nb, t]))
      }
    }
  }

  delta
}


# -----------------------------------------------------------------------------
# .da_adjust
#
# Applies the dependency-awareness correction to phi_T_mat and the marginal
# relative-occurrence vector Phi.
#
# @param phi_T_mat   p x T_stop matrix of relative occurrences.
# @param Phi         Length-p vector of marginal relative occurrences at T_stop.
# @param p           Number of candidate variables.
# @param T_stop      Stopping index.
# @param method      'trex+DA+AR1' or 'trex+DA+equi'.
# @param cor_coef    Estimated correlation coefficient.
# @param rho_thr_DA  Correlation threshold.
#
# @return Named list: phi_T_mat (adjusted), Phi (adjusted).
# -----------------------------------------------------------------------------
.da_adjust <- function(phi_T_mat, Phi, p, T_stop, method, cor_coef, rho_thr_DA) {

  delta <- .da_delta_mat(phi_T_mat, p, T_stop, method, cor_coef, rho_thr_DA)

  if (is.null(delta)) {
    return(list(phi_T_mat = phi_T_mat, Phi = Phi))
  }

  list(
    phi_T_mat = phi_T_mat / delta,
    Phi       = Phi       / delta[, T_stop]
  )
}


# -----------------------------------------------------------------------------
# .screen_majority_vote
#
# Screens variables via a majority-vote rule: a variable is selected if its
# relative occurrence exceeds 0.5, i.e., it was active in more than half of
# the K random experiments when the first dummy entered the model.
#
# @param Phi  Length-p vector of relative occurrences.
#
# @return Named list: set_zero (logical, TRUE = not selected), conf_level (NA).
# -----------------------------------------------------------------------------
.screen_majority_vote <- function(Phi) {
  list(
    set_zero   = Phi <= 0.5,
    conf_level = NA_real_
  )
}


# -----------------------------------------------------------------------------
# .screen_bootstrap
#
# Screens variables via a non-parametric bootstrap on the terminal dummy
# coefficients.  A variable is removed from the selected set if its mean
# terminal coefficient falls inside the bootstrap confidence interval of the
# mean dummy coefficient (i.e., it is indistinguishable from noise).
#
# The confidence level is chosen as the tightest CI that does not select more
# variables than the plain majority-vote rule.
#
# @param beta_mat        K x p matrix of terminal candidate coefficients.
# @param dummy_beta_mat  K x num_dummies matrix of terminal dummy coefficients.
# @param Phi             Length-p vector of relative occurrences.
# @param R               Number of bootstrap resamples.
# @param conf_level_grid Numeric vector of confidence levels in [0, 1].
# @param eps             Numerical zero.
#
# @return Named list: set_zero (logical, TRUE = not selected), conf_level.
# -----------------------------------------------------------------------------
.screen_bootstrap <- function(beta_mat,
                               dummy_beta_mat,
                               Phi,
                               R,
                               conf_level_grid,
                               eps) {

  # Pool of active dummy coefficients (those that entered the LARS path)
  active_dummy_coefs <- c(dummy_beta_mat[abs(dummy_beta_mat) > eps])

  # Non-parametric bootstrap of the mean active dummy coefficient
  boot_stat <- function(data, idx) mean(data[idx])
  boot_out  <- boot::boot(data = active_dummy_coefs, statistic = boot_stat, R = R)

  # Normal CIs at every level of the confidence grid (single vectorised call)
  ci_mat <- boot::boot.ci(
    boot.out = boot_out,
    conf     = conf_level_grid,
    type     = "norm"
  )$normal[, c(2L, 3L), drop = FALSE]

  # Majority-vote baseline: number of variables with Phi > 0.5
  R_majority_vote <- sum(Phi > 0.5)
  beta_means      <- colMeans(beta_mat)

  # For each CI level: count how many candidates have their mean beta OUTSIDE
  # the CI (i.e., distinguishable from dummy noise → selected)
  n_selected_per_level <- vapply(
    seq_len(nrow(ci_mat)),
    function(b) sum(beta_means < ci_mat[b, 1L] | beta_means > ci_mat[b, 2L]),
    FUN.VALUE = integer(1L)
  )

  # Choose the tightest CI that does not exceed the majority-vote count
  conf_level_index <- which.max(n_selected_per_level <= R_majority_vote)
  conf_level       <- conf_level_grid[conf_level_index]

  # Variables whose mean beta lies inside the chosen CI are treated as noise
  set_zero <- beta_means >= ci_mat[conf_level_index, 1L] &
              beta_means <= ci_mat[conf_level_index, 2L]

  list(set_zero = set_zero, conf_level = conf_level)
}


# =============================================================================
# 4. screen_trex
# =============================================================================

#' Run the Screen-T-Rex selector (\doi{10.1109/SSP53291.2023.10207957})
#'
#' The Screen-T-Rex selector performs very fast variable selection in
#' high-dimensional settings while reporting the automatically determined
#' false discovery rate (FDR).
#'
#' \strong{Algorithm overview:}
#' \enumerate{
#'   \item Fix \eqn{T_{\rm stop} = 1} and run \eqn{K} T-LARS random
#'         experiments, collecting the relative-occurrence vector
#'         \eqn{\hat{\Phi}} and the terminal coefficient matrices.
#'   \item (Optional) Apply a dependency-aware correction to \eqn{\hat{\Phi}}
#'         for the \code{trex+DA+AR1} or \code{trex+DA+equi} variants.
#'   \item Screen: select variable \eqn{j} iff \eqn{\hat{\Phi}_j > 0.5}
#'         (majority vote), or use the stricter bootstrap criterion when
#'         \code{bootstrap = TRUE}.
#'   \item Report the closed-form FDR estimate
#'         \eqn{\widehat{\rm FDR} = T_{\rm stop} / |\hat{S}|}.
#' }
#'
#' @param X                   Real valued predictor matrix (\eqn{n \times p}).
#' @param y                   Response vector of length \eqn{n}.
#' @param K                   Number of random experiments (default: 20).
#' @param R                   Number of bootstrap resamples (default: 1000).
#' @param method              Selector variant:
#'   \describe{
#'     \item{\code{'trex'}}{T-Rex selector (\doi{10.48550/arXiv.2110.06048})}
#'     \item{\code{'trex+GVS'}}{T-Rex+GVS (\doi{10.23919/EUSIPCO55093.2022.9909883})}
#'     \item{\code{'trex+DA+AR1'}}{T-Rex+DA with AR(1) prior}
#'     \item{\code{'trex+DA+equi'}}{T-Rex+DA with equicorrelation prior}
#'   }
#' @param bootstrap           Logical. If \code{TRUE} use bootstrap screening
#'                            instead of the majority-vote rule (default: \code{FALSE}).
#' @param conf_level_grid     Confidence level grid for bootstrap CIs
#'                            (default: \code{seq(0, 1, by = 0.001)}).
#' @param cor_coef            Correlation coefficient for DA variants.
#'                            Estimated from \code{X} when \code{NA} (default).
#' @param type                \code{'lar'} for LARS (default), \code{'lasso'} for Lasso.
#' @param corr_max            Maximum allowed inter-cluster correlation
#'                            (\code{method = 'trex+GVS'} only; default: 0.5).
#' @param lambda_2_lars       \eqn{\lambda_2} for the LARS-based Elastic Net.
#'                            Determined by cross-validation when \code{NULL}.
#' @param rho_thr_DA          Correlation threshold below which the DA correction
#'                            is skipped for \code{'trex+DA+equi'} (default: 0.02).
#' @param parallel_process    Logical. If \code{TRUE} run experiments in parallel.
#' @param parallel_max_cores  Maximum number of cores (default: physical core count).
#' @param seed                Integer RNG seed for reproducible parallel runs.
#' @param eps                 Numerical zero (default: \code{.Machine$double.eps}).
#' @param verbose             Logical. If \code{TRUE} show progress (default: \code{TRUE}).
#'
#' @return A named list with components:
#'   \describe{
#'     \item{\code{selected_var}}{Integer vector of length \eqn{p}
#'           (1 = selected, 0 = not selected).}
#'     \item{\code{FDR_estimate}}{Automatically selected FDR level
#'           \eqn{= T_{\rm stop} / |\hat{S}|}.}
#'     \item{\code{dummy_beta_mat}}{K x \code{num_dummies} matrix of terminal
#'           dummy coefficients.}
#'     \item{\code{beta_mat}}{K x p matrix of terminal candidate coefficients.}
#'     \item{\code{Phi}}{Length-p vector of (possibly DA-adjusted) relative
#'           occurrences.}
#'     \item{\code{K, R, method, bootstrap, conf_level, cor_coef, type,
#'           corr_max, lambda_2_lars, rho_thr_DA}}{Input parameters, echoed.}
#'   }
#'
#' @importFrom parallel detectCores
#' @importFrom stats coef arima cor
#' @importFrom boot boot boot.ci
#'
#' @export
#'
#' @examples
#' data("Gauss_data")
#' X <- Gauss_data$X
#' y <- c(Gauss_data$y)
#' set.seed(123)
#' res <- screen_trex(X = X, y = y)
#' res$selected_var
screen_trex <- function(X,
                        y,
                        K                  = 20,
                        R                  = 1000,
                        method             = "trex",
                        bootstrap          = FALSE,
                        conf_level_grid    = seq(0, 1, by = 0.001),
                        cor_coef           = NA,
                        type               = "lar",
                        corr_max           = 0.5,
                        lambda_2_lars      = NULL,
                        rho_thr_DA         = 0.02,
                        parallel_process   = FALSE,
                        parallel_max_cores = min(K, max(1L, parallel::detectCores(logical = FALSE))),
                        seed               = NULL,
                        eps                = .Machine$double.eps,
                        verbose            = TRUE) {

  # ---------------------------------------------------------------------------
  # Input validation
  # ---------------------------------------------------------------------------
  method <- match.arg(method, c("trex", "trex+GVS", "trex+DA+AR1", "trex+DA+equi"))
  type   <- match.arg(type,   c("lar", "lasso"))

  if (!is.matrix(X))               stop("'X' must be a matrix.")
  if (!is.numeric(X))              stop("'X' only allows numerical values.")
  if (anyNA(X))                    stop("'X' contains NAs. Please remove or impute them before proceeding.")
  if (!is.vector(drop(y)))         stop("'y' must be a vector.")
  if (!is.numeric(y))              stop("'y' only allows numerical values.")
  if (anyNA(y))                    stop("'y' contains NAs. Please remove or impute them before proceeding.")
  if (nrow(X) != length(drop(y)))  stop("Number of rows in X does not match length of y.")
  if (length(K) != 1L || K < 2L || K %% 1L != 0L) {
    stop("'K' must be an integer >= 2.")
  }
  if (method == "trex+GVS" && (length(corr_max) != 1L || corr_max < 0 || corr_max > 1)) {
    stop("'corr_max' must be a scalar in [0, 1].")
  }
  if (!is.null(lambda_2_lars) && (length(lambda_2_lars) != 1L || lambda_2_lars < eps)) {
    stop("'lambda_2_lars' must be a positive scalar.")
  }
  if (bootstrap && (min(conf_level_grid) < 0 || max(conf_level_grid) > 1)) {
    stop("'conf_level_grid' must contain values in [0, 1].")
  }

  # ---------------------------------------------------------------------------
  # Preprocessing
  # ---------------------------------------------------------------------------
  X <- scale(X)
  y <- y - mean(y)

  p      <- ncol(X)
  T_stop <- 1L          # Screen-T-Rex is defined with T_stop = 1

  # ---------------------------------------------------------------------------
  # Step 1 — Run K random experiments
  # ---------------------------------------------------------------------------
  rand_exp <- random_experiments(
    X                  = X,
    y                  = y,
    K                  = K,
    T_stop             = T_stop,
    num_dummies        = p,
    method             = method,
    type               = type,
    corr_max           = corr_max,
    lambda_2_lars      = lambda_2_lars,
    early_stop         = TRUE,
    verbose            = verbose,
    intercept          = FALSE,
    standardize        = TRUE,
    dummy_coef         = TRUE,
    parallel_process   = parallel_process,
    parallel_max_cores = parallel_max_cores,
    seed               = seed,
    eps                = eps
  )

  beta_mat       <- rand_exp$rand_exp_last_betas_mat       # K x p
  dummy_beta_mat <- rand_exp$dummy_rand_exp_last_betas_mat # K x num_dummies
  phi_T_mat      <- rand_exp$phi_T_mat                     # p x T_stop
  Phi            <- rand_exp$Phi                           # length p

  # ---------------------------------------------------------------------------
  # Step 2 — Dependency-aware correction (DA variants only)
  # ---------------------------------------------------------------------------
  da_methods <- c("trex+DA+AR1", "trex+DA+equi")

  if (method %in% da_methods) {
    if (is.na(cor_coef)) {
      cor_coef <- .estimate_cor_coef(X, method)
    }

    da_res    <- .da_adjust(phi_T_mat, Phi, p, T_stop, method, cor_coef, rho_thr_DA)
    phi_T_mat <- da_res$phi_T_mat
    Phi       <- da_res$Phi
  }

  # ---------------------------------------------------------------------------
  # Step 3 — Variable screening
  # ---------------------------------------------------------------------------
  screen_res <- if (bootstrap) {
    .screen_bootstrap(beta_mat, dummy_beta_mat, Phi, R, conf_level_grid, eps)
  } else {
    .screen_majority_vote(Phi)
  }

  set_zero   <- screen_res$set_zero
  conf_level <- screen_res$conf_level

  # ---------------------------------------------------------------------------
  # Step 4 — Estimated support and automatic FDR
  # ---------------------------------------------------------------------------
  selected_var            <- integer(p)
  selected_var[!set_zero] <- 1L

  # Closed-form FDR estimate: hat{FDR} = T_stop / |S|  (= 1/|S| since T_stop=1)
  FDR_estimate <- if (any(!set_zero)) T_stop / sum(!set_zero) else 0

  # ---------------------------------------------------------------------------
  # Return
  # ---------------------------------------------------------------------------
  list(
    selected_var   = selected_var,
    FDR_estimate   = FDR_estimate,
    dummy_beta_mat = dummy_beta_mat,
    beta_mat       = beta_mat,
    Phi            = Phi,
    K              = K,
    R              = R,
    method         = method,
    bootstrap      = bootstrap,
    conf_level     = conf_level,
    cor_coef       = cor_coef,
    type           = type,
    corr_max       = corr_max,
    lambda_2_lars  = lambda_2_lars,
    rho_thr_DA     = rho_thr_DA
  )
}
