# =============================================================================
# tsolvers.R - T-Solver R6 class definitions
# =============================================================================


#' @importFrom Rcpp evalCpp
#' @import R6
NULL

# ==============================================================================
# TSolver_Base
# ==============================================================================

#' T-Solver Options
#'
#' @description
#' Every T-solver input except the data \code{X}, \code{D} and \code{y},
#' as one list (the C++ \code{SolverOptions}). Pass it to a solver's
#' \code{$new(X, D, y, options = ...)}. Each solver reads the fields it
#' needs and ignores the rest; the defaults are the solver-level defaults.
#'
#' @param normalize Logical; scale each column of X and D to unit L2 norm (or
#'   unit SD, see \code{scaling_mode}). \code{FALSE} when the caller
#'   pre-scaled (see the in-place note below). Default \code{TRUE}.
#' @param intercept Logical; center X, D and y (an intercept is fitted).
#'   \code{FALSE} when the caller pre-centered. Default \code{TRUE}.
#' @param precentered Logical; the caller centered X, D and y
#'   (\code{intercept = FALSE} on centered data, as in the T-Rex
#'   selectors). The centered columns span at most n - 1 dimensions, so the
#'   solver counts the intercept as with \code{intercept = TRUE}: the active
#'   set is capped at n - 1 and the degrees of freedom and Cp include the
#'   intercept. The data are not touched. Redundant when \code{intercept}
#'   is \code{TRUE}. Default \code{FALSE}.
#'
#' @section In-place preprocessing (the writer/reader contract):
#' The T-solvers work on the caller's matrices without copying, in R as in
#' C++ and Python. With \code{normalize = TRUE} or \code{intercept = TRUE} a
#' solver is a \emph{writer}: it centers / scales the columns of X and D in
#' place (visible on the caller's matrix or memory-mapped file), keeps the
#' statistics itself, and holds a claim on both until \code{$restore()} or
#' garbage collection, both of which write X and D back; a second writer
#' (T-solver or T-Rex selector) on a claimed matrix is an error, because a
#' second in-place transformation moves the data by rounding. With
#' \code{normalize = FALSE} and \code{intercept = FALSE} a solver is a
#' \emph{reader} that runs as constructed and never writes; a pipeline with
#' several solvers on one matrix pre-scales once in front ([MLLpNormScaler])
#' and constructs readers. y is never written. See [TSolver_Base].
#' @param scaling_mode \code{"L2"} (default) or \code{"ZSCORE"}: the column
#'   scaling applied when \code{normalize} is \code{TRUE}.
#' @param verbose Logical; print status and diagnostics. Default \code{FALSE}.
#' @param tol Numerical tolerance of the solver's internal comparisons.
#'   Default: machine epsilon.
#' @param tie_seed Seed of the solver's tie-break engine. Default 0: a fixed
#'   seed, so a solver run is reproducible also where exact ties occur; a
#'   negative value seeds from the system entropy source (nondeterministic).
#' @param n_threads Thread budget of the solver's own passes over the design
#'   (the correlation refreshes, the step-length search, the standardization).
#'   Default 1 (serial); a whole number of at least 1. A pass splits only
#'   above about a million entries of the design, and never inside a parallel
#'   region of the caller. The T-Rex selectors set it from
#'   \code{trex_control(max_inner_threads = )}.
#' @param exch_tie_alpha Exchangeable-tie band width for the greedy solvers
#'   (TOMP and TAFS families), in units of the pairwise ranking-noise sd;
#'   0 (default) turns it off. Ignored by the path solvers.
#' @param exch_tie_floor Minimum absolute correlation for exchangeable-tie
#'   candidates, in [0, 1). Default 0.5.
#' @param lambda2 L2 (ridge / group-ridge) penalty weight, >= 0. Read by
#'   TENET, TENETAug and TIENETAug. Default 0 (no quadratic penalty: the
#'   elastic nets reduce to their lasso).
#' @param rho_afs TAFS shrinkage step in (0, 1]; 1 (default) is forward
#'   stepwise.
#' @param ncgmp_variant TNCGMP variant: \code{"line_search"} (default) or
#'   \code{"fully_corrective"}.
#' @param tenet_aug_use_lars Logical; TENETAug and TIENETAug run the
#'   augmented system with a pure-LARS inner path that never drops variables
#'   (default \code{FALSE}: LASSO inner path).
#' @param kappa_dummy Ridge weight of the dummy block of the CCD elastic nets
#'   (not bound to R yet). Default 1.
#' @param fold_dummies Logical; TCDTIKNET dummy coupling (not bound to R
#'   yet). Default \code{FALSE}.
#' @param group_penalty TCDIENET structured penalty, \code{"MEAN"} (default)
#'   or \code{"CENTERING"} (not bound to R yet).
#' @param cd_lambda_rel_tol,cd_tol_certify,cd_tol_probe,cd_gram_cap,cd_max_sweeps,cd_certified_localization
#'   Knobs of the CD solver family (not bound to R yet); the defaults are the
#'   solver defaults (1e-3, 1e-10, 1e-8, 400, 2000, \code{TRUE}).
#'
#' @return A named list suitable for the \code{options} argument of every
#'   T-solver constructor.
#'
#' @examples
#' tsolver_options()
#' tsolver_options(lambda2 = 0.5, normalize = FALSE)
#'
#' @family T-solvers
#' @export
tsolver_options <- function(normalize = TRUE,
                            intercept = TRUE,
                            precentered = FALSE,
                            scaling_mode = c("L2", "ZSCORE"),
                            verbose = FALSE,
                            tol = .Machine$double.eps,
                            tie_seed = 0,
                            n_threads = 1,
                            exch_tie_alpha = 0,
                            exch_tie_floor = 0.5,
                            lambda2 = 0,
                            rho_afs = 1,
                            ncgmp_variant = c("line_search", "fully_corrective"),
                            tenet_aug_use_lars = FALSE,
                            kappa_dummy = 1,
                            fold_dummies = FALSE,
                            group_penalty = c("MEAN", "CENTERING"),
                            cd_lambda_rel_tol = 1e-3,
                            cd_tol_certify = 1e-10,
                            cd_tol_probe = 1e-8,
                            cd_gram_cap = 400,
                            cd_max_sweeps = 2000,
                            cd_certified_localization = TRUE) {
  scaling_mode  <- match.arg(scaling_mode)
  ncgmp_variant <- match.arg(ncgmp_variant)
  group_penalty <- match.arg(group_penalty)
  list(
    normalize                 = as.logical(normalize),
    intercept                 = as.logical(intercept),
    precentered               = as.logical(precentered),
    scaling_mode              = scaling_mode,
    verbose                   = as.logical(verbose),
    tol                       = as.numeric(tol),
    tie_seed                  = as.numeric(tie_seed),
    n_threads                 = as.integer(n_threads),
    exch_tie_alpha            = as.numeric(exch_tie_alpha),
    exch_tie_floor            = as.numeric(exch_tie_floor),
    lambda2                   = as.numeric(lambda2),
    rho_afs                   = as.numeric(rho_afs),
    ncgmp_variant             = ncgmp_variant,
    tenet_aug_use_lars        = as.logical(tenet_aug_use_lars),
    kappa_dummy               = as.numeric(kappa_dummy),
    fold_dummies              = as.logical(fold_dummies),
    group_penalty             = group_penalty,
    cd_lambda_rel_tol         = as.numeric(cd_lambda_rel_tol),
    cd_tol_certify            = as.numeric(cd_tol_certify),
    cd_tol_probe              = as.numeric(cd_tol_probe),
    cd_gram_cap               = as.numeric(cd_gram_cap),
    cd_max_sweeps             = as.numeric(cd_max_sweeps),
    cd_certified_localization = as.logical(cd_certified_localization)
  )
}


#' Base T-Solver Class
#'
#' @description
#' Abstract base class for all T-Solver R6 wrappers. Provides the common solver
#' API backed by a Cpp \code{TSolver_Base} instance. Users should instantiate one
#' of the concrete subclasses (\code{TLARS_Solver}, \code{TLASSO_Solver}, etc.)
#' rather than this class directly.
#'
#' @section In-place preprocessing (the writer/reader contract):
#' The solver works on the caller's memory without copying: the R matrices
#' \code{X} and \code{D} (double storage; an integer matrix is coerced first,
#' which is a copy) or the memory-mapped file behind an \code{mmap_matrix}.
#' With \code{normalize = TRUE} or \code{intercept = TRUE} in
#' \code{\link{tsolver_options}} the solver is a \emph{writer}: it centers /
#' scales the columns of \code{X} and \code{D} in place (the change is
#' visible on the caller's matrix), keeps the statistics itself, and holds a
#' claim on both until \code{$restore()} or garbage collection, both of which
#' write \code{X} and \code{D} back (\code{$restore()} is idempotent and the
#' fitted path stays available); constructing a second writer (T-solver or
#' T-Rex selector) on a claimed matrix is an error, because a second
#' in-place transformation moves the data by rounding. With both
#' \code{FALSE} the solver is a \emph{reader}: it runs as constructed and
#' never writes. A pipeline with several solvers on one matrix pre-scales
#' once in front ([MLLpNormScaler]) and constructs readers. \code{$load()}
#' expects \code{X} and \code{D} in the state they were saved in: load into
#' a reader on pre-scaled data, or into a writer constructed with the same
#' options on the same data (it keeps the duty to write back). \code{y} is
#' never written (the solver centers an internal copy).
#'
#' @field X Numeric matrix of predictors, or an \code{mmap_matrix}.
#' @field D Numeric matrix of dummies.
#' @field y Numeric response vector.
#' @field ptr External pointer to the underlying Cpp \code{TSolver_Base} object.
#' @examples
#' # TSolver_Base is abstract; use a concrete subclass instead, e.g.:
#' # solver <- TLARS_Solver$new(X, D, y)
#' @keywords internal
TSolver_Base <- R6::R6Class(
  "TSolver_Base",
  public = list(
    X   = NULL,
    D   = NULL,
    y   = NULL,
    ptr = NULL,

    #' @description
    #' Validate and store data fields. Called by every subclass \code{initialize}.
    #' Not intended for direct instantiation. The stored \code{X} and \code{D}
    #' are the matrices the C++ solver maps (no copy for double storage).
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    initialize = function(X, D, y) {
      is_mmap <- inherits(X, "mmap_matrix")
      if (!is.matrix(X) && !is_mmap) {
        stop("Input X must be a numeric matrix or mmap_matrix.")
      }
      if (!is.matrix(D) || !is.numeric(y)) {
        stop("Inputs D must be a matrix, and y must be numeric.")
      }
      self$X <- if (is_mmap) X else .as_double_matrix(X)
      self$D <- .as_double_matrix(D, arg = "D")
      self$y <- .as_double_vector(as.vector(y))
    },

    #' @description Execute solver step.
    #' @param T_stop Integer stopping criterion; \code{-1} runs until the
    #'   early-stop condition is met.
    #' @param early_stop Logical; stop early when enough dummies are active.
    #' @return Invisibly returns \code{self} for method chaining.
    execute_step = function(T_stop = -1, early_stop = TRUE) {
      tsolver_execute_step(self$ptr, T_stop, early_stop)
      invisible(self)
    },

    #' @description Get the number of steps executed so far.
    #' @return Integer number of completed steps.
    get_num_steps = function() {
      tsolver_get_num_steps(self$ptr)
    },

    #' @description Heap bytes the solver holds: its copy of \code{y}, the
    #'   residual, correlations, Cholesky factor, coefficient path,
    #'   diagnostics and reserved capacity (the C++ \code{memoryBytes()}).
    #'   \code{X} and \code{D} are borrowed and not counted, nor are the
    #'   temporaries inside a step, so the peak of a step is higher.
    #' @return Numeric (double) byte count.
    get_memory_bytes = function() {
      tsolver_get_memory_bytes(self$ptr)
    },

    #' @description Get the action path (variable additions and removals per
    #'   step). The C++ solver already records the R \code{lars} convention:
    #'   1-based signed codes, \code{+j} for the addition and \code{-j} for the
    #'   removal of variable \code{j} (dummies continue after the predictors).
    #' @param r_index Logical; if \code{TRUE} (default) the 1-based signed
    #'   codes are returned as recorded. If \code{FALSE} the 0-based indices
    #'   of the variables acted on are returned (\code{abs(a) - 1}); the sign
    #'   is dropped there because a 0-based signed code cannot distinguish
    #'   the addition from the removal of the first variable.
    #' @return A list of integer vectors, one per step.
    get_actions = function(r_index = TRUE) {
      acts <- tsolver_get_actions(self$ptr)
      if (r_index) {
        acts
      } else {
        lapply(acts, function(a) abs(a) - 1L)
      }
    },

    #' @description Get coefficient vector at a given step.
    #' @param step Integer step index; \code{-1} returns the current/final step.
    #' @return Numeric vector of coefficients.
    get_beta = function(step = -1) {
      b <- tsolver_get_beta(self$ptr, step)
      b[seq_len(ncol(self$X))]
    },

    #' @description Get Mallows Cp scores along the solution path.
    #' @return Numeric vector of Cp scores, one per step.
    get_cp = function() {
      tsolver_get_cp(self$ptr)[-1]
    },

    #' @description Get residual sum of squares along the solution path.
    #' @return Numeric vector of RSS values, one per step.
    get_rss = function() {
      tsolver_get_rss(self$ptr)[-1]
    },

    #' @description Get R-squared values along the solution path.
    #' @return Numeric vector of R-squared values, one per step.
    get_r2 = function() {
      tsolver_get_r2(self$ptr)[-1]
    },

    #' @description Get degrees of freedom along the solution path.
    #' @return Numeric vector of degrees of freedom, one per step.
    get_dof = function() {
      tsolver_get_dof(self$ptr)[-1]
    },

    #' @description Get currently active variable indices.
    #' @param r_index Logical; if \code{TRUE} (default) indices are 1-based
    #'   (R convention); if \code{FALSE} they are 0-based (Cpp convention).
    #' @return Integer vector of active variable indices.
    get_actives = function(r_index = TRUE) {
      p <- ncol(self$X)
      res <- tsolver_get_actives(self$ptr)
      res <- res[res < p]
      if (r_index) res <- res + 1L
      res
    },

    #' @description Get currently inactive variable indices.
    #' @param r_index Logical; if \code{TRUE} (default) indices are 1-based
    #'   (R convention); if \code{FALSE} they are 0-based (Cpp convention).
    #' @return Integer vector of inactive variable indices.
    get_inactives = function(r_index = TRUE) {
      p <- ncol(self$X)
      res <- tsolver_get_inactives(self$ptr)
      res <- res[res < p]
      if (r_index) res <- res + 1L
      res
    },

    #' @description Get the fitted intercept at a given step.
    #' @param step Integer step index; \code{-1} returns the current/final step.
    #' @return Scalar numeric intercept.
    get_intercept = function(step = -1) {
      tsolver_get_intercept(self$ptr, step)
    },

    #' @description Serialize the solver state to a binary file.
    #' @param filename Character; path to the output file.
    #' @return Invisibly returns \code{self}.
    save = function(filename) {
      tsolver_save(self$ptr, filename)
      invisible(self)
    },

    #' @description Restore the solver state from a binary file created by
    #'   \code{$save()}. The loaded solver expects \code{X} and \code{D} in
    #'   the state they were saved in and is a reader (it never writes).
    #' @param filename Character; path to the input file.
    #' @return Invisibly returns \code{self}.
    load = function(filename) {
      tsolver_load(self$ptr, filename)
      invisible(self)
    },

    #' @description Residual vector at the current step.
    #' @return Numeric vector of length n.
    get_residuals = function() {
      tsolver_get_residuals(self$ptr)
    },

    #' @description Coefficient path: one column per step including the
    #'   all-zero start (column 1), predictors first, dummies from
    #'   \code{get_dummy_start_index()} on; the same layout as Python's
    #'   \code{getBetaPath()}.
    #' @return Numeric matrix ((p + num_dummies) x (steps + 1)).
    get_beta_path = function() {
      tsolver_get_beta_path(self$ptr)
    },

    #' @description Mallows' Cp along the path with a supplied noise variance.
    #' @param sigma_hat_sq Positive numeric, the noise variance estimate.
    #' @return Numeric vector, one per step.
    get_cp_with_sigma = function(sigma_hat_sq) {
      tsolver_get_cp_with_sigma(self$ptr, as.numeric(sigma_hat_sq))[-1]
    },

    #' @description Name of the underlying solver type (e.g. \code{"TLARS"}).
    #' @return Character scalar.
    solver_type_to_string = function() {
      tsolver_solver_type_to_string(self$ptr)
    },

    #' @description Variables dropped during the path (1-based, predictors and
    #'   dummies alike).
    #' @return Integer vector.
    get_dropped_indices = function() {
      as.integer(tsolver_get_dropped_indices(self$ptr)) + 1L
    },

    #' @description Active predictor indices (1-based, dummies excluded).
    #' @return Integer vector.
    get_active_predictor_indices = function() {
      as.integer(tsolver_get_active_predictor_indices(self$ptr)) + 1L
    },

    #' @description Active dummy indices (1-based within the dummy block).
    #' @return Integer vector.
    get_active_dummy_indices = function() {
      as.integer(tsolver_get_active_dummy_indices(self$ptr)) - self$get_dummy_start_index() + 2L
    },

    #' @description Number of active variables (predictors and dummies).
    #' @return Integer.
    get_num_actives = function() {
      tsolver_get_num_actives(self$ptr)
    },

    #' @description 1-based column at which the dummies start in the merged
    #'   (predictors, dummies) index space, i.e. \code{p + 1}.
    #' @return Integer.
    get_dummy_start_index = function() {
      tsolver_get_dummy_start_index(self$ptr) + 1L
    },

    #' @description Whether the solver is connected to its data views.
    #' @return Logical.
    is_connected = function() {
      tsolver_is_connected(self$ptr)
    },

    #' @description Set the numerical tolerance of the internal comparisons.
    #' @param tol Positive numeric.
    #' @return Invisibly returns \code{self}.
    set_tolerance = function(tol) {
      tsolver_set_tolerance(self$ptr, as.numeric(tol))
      invisible(self)
    },

    #' @description Seed the tie-break engine.
    #' @param seed Non-negative integer.
    #' @return Invisibly returns \code{self}.
    set_tie_seed = function(seed) {
      tsolver_set_tie_seed(self$ptr, as.numeric(seed))
      invisible(self)
    },

    #' @description Set the thread budget of the solver's own passes over the
    #'   design (\code{tsolver_options(n_threads = )}): 1 serial, at least 1.
    #' @param n_threads Non-negative integer.
    #' @return Invisibly returns \code{self}.
    set_num_threads = function(n_threads) {
      tsolver_set_num_threads(self$ptr, as.integer(n_threads))
      invisible(self)
    },

    #' @description The resolved thread budget of the solver's own passes.
    #' @return Integer, at least 1.
    get_num_threads = function() {
      tsolver_get_num_threads(self$ptr)
    },

    #' @description Exchangeable-tie band for the greedy solvers (TOMP / TAFS
    #'   families); \code{alpha <= 0} turns it off.
    #' @param alpha Band width in units of the ranking-noise sd.
    #' @param floor Minimum |correlation| of a tie candidate (default 0.5).
    #' @return Invisibly returns \code{self}.
    set_exchangeable_tie = function(alpha, floor = 0.5) {
      tsolver_set_exchangeable_tie(self$ptr, as.numeric(alpha), as.numeric(floor))
      invisible(self)
    },

    #' @description Warnings recorded by the solver (dropped columns, the
    #'   already-preprocessed input, collinear rejections, ...).
    #' @return Character vector.
    get_warnings = function() {
      tsolver_get_warnings(self$ptr)
    },

    #' @description Clear the recorded warnings.
    #' @return Invisibly returns \code{self}.
    clear_warnings = function() {
      tsolver_clear_warnings(self$ptr)
      invisible(self)
    },

    #' @description Undo the in-place centering / scaling a writer solver
    #'   applied to \code{X} and \code{D} at construction and release its
    #'   claim on them, so that another writer may take the matrices. The
    #'   solver's fitted path stays available. \code{y} is not modified. No-op
    #'   for a reader (\code{normalize = FALSE, intercept = FALSE}).
    #' @return Invisibly returns \code{self}.
    restore = function() {
      tsolver_restore(self$ptr)
      invisible(self)
    }
  )
)


# ==============================================================================
# LARS-family subclasses
# ==============================================================================

#' T-LARS Solver
#'
#' @description
#' Terminating Least Angle Regression (T-LARS) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TLARS_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#' solver$get_actives()
#'
#' @family T-solvers
#' @export
TLARS_Solver <- R6::R6Class(
  "TLARS_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TLARS_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_lars_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_lars_create(self$X, self$D, self$y, options)
      }
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    },

    #' @description Kahan refresh interval of the LARS-family solver (steps
    #'   between exact refreshes of the compensated sums).
    #' @return Integer.
    get_kahan_refresh_interval = function() {
      tsolver_lars_get_kahan_refresh_interval(self$ptr)
    },

    #' @description Set the Kahan refresh interval.
    #' @param interval Positive integer.
    #' @return Invisibly returns \code{self}.
    set_kahan_refresh_interval = function(interval) {
      tsolver_lars_set_kahan_refresh_interval(self$ptr, as.integer(interval))
      invisible(self)
    }
  )
)


#' T-LASSO Solver
#'
#' @description
#' Terminating LASSO solver (T-LARS with variable removal).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," J. R. Stat. Soc. Ser. B Stat. Methodol.,
#'   vol. 58, no. 1, pp. 267-288, 1996.
#'
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' M. R. Osborne, B. Presnell, and B. A. Turlach, "On the LASSO and Its Dual," J. Comput. Graph. Statist.,
#'   vol. 9, no. 2, pp. 319-337, 2000.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TLASSO_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TLASSO_Solver <- R6::R6Class(
  "TLASSO_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TLASSO_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tlasso_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tlasso_create(self$X, self$D, self$y, options)
      }
    },

    #' @description Get the number of coefficient removals.
    #' @return Integer count of removals.
    get_num_removals = function() {
      tsolver_get_num_removals(self$ptr)
    },

    #' @description Get the cycling ratio (removals / total steps).
    #' @return Scalar numeric cycling ratio.
    get_cycling_ratio = function() {
      tsolver_get_cycling_ratio(self$ptr)
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


#' T-Stepwise Solver
#'
#' @description
#' Terminating Forward Stepwise Regression solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TSTEPWISE_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TSTEPWISE_Solver <- R6::R6Class(
  "TSTEPWISE_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TSTEPWISE_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tstepwise_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tstepwise_create(self$X, self$D, self$y, options)
      }
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


#' T-ENET Solver
#'
#' @description
#' Terminating Elastic Net solver (L1 + L2 regularization).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser. B
#'   Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
#'
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TENET_Solver$new(X, D, y, options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TENET_Solver <- R6::R6Class(
  "TENET_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TENET_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2}, the L2 (ridge) penalty (>= 0; the
    #'   default 0 is the plain lasso path).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tenet_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tenet_create(self$X, self$D, self$y, options)
      }
    },

    #' @description Get the number of coefficient removals.
    #'
    #' @return Integer count of removals.
    get_num_removals = function() {
      tsolver_get_num_removals(self$ptr)
    },

    #' @description Get the cycling ratio (removals / total steps).
    #'
    #' @return Scalar numeric cycling ratio.
    get_cycling_ratio = function() {
      tsolver_get_cycling_ratio(self$ptr)
    },

    #' @description Get the regularization (lambda) path.
    #'
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    },

    #' @description The elastic-net mixing alpha implied by lambda2.
    #' @return Numeric scalar.
    get_alpha = function() {
      tsolver_tenet_get_alpha(self$ptr)
    }
  )
)


#' T-ENET-Aug Solver
#'
#' @description
#' Terminating Elastic Net solver via row augmentation (TENETAug). The ridge
#' penalty is encoded in an augmented data system solved by an inner T-LASSO
#' (or pure T-LARS) path, matching the R reference implementation of
#' \code{trex_GVS()}. Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser. B
#'   Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
#'
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "False Discovery Rate Control for Grouped Variable Selection in
#'   High-Dimensional Linear Models Using the T-Knock Filter," in Proc. 30th Eur. Signal Process. Conf.
#'   (EUSIPCO), 2022, pp. 892-896.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TENETAug_Solver$new(X, D, y, options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TENETAug_Solver <- R6::R6Class(
  "TENETAug_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TENETAug_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2} (the ridge penalty, >= 0) and
    #'   \code{tenet_aug_use_lars} (run the inner path with a pure-LARS
    #'   solver that never drops variables).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tenet_aug_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tenet_aug_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-IENET-Aug Solver
#'
#' @description
#' Terminating Informed Elastic Net solver via row augmentation (TIENETAug).
#' The group-ridge penalty (one augmented row per group, scaled by
#' \code{sqrt(lambda2)}) is encoded in an augmented data system solved by an
#' inner T-LASSO (or pure T-LARS) path. This is the solver used internally by
#' \code{TRexGVSSelector} for \code{gvs_type = "IEN"}.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped Variable Selection and
#'   FDR Control in Genomics Research," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt.
#'   Process. (CAMSAP), 2023, pp. 466-470.
#'
#' H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser. B
#'   Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
#'
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' groups <- c(1, 1, 2, 2)  # 1-based group id per predictor
#' solver <- TIENETAug_Solver$new(X, D, y, groups = groups,
#'                                options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TIENETAug_Solver <- R6::R6Class(
  "TIENETAug_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TIENETAug_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param groups Integer vector of 1-based group ids, one per predictor
    #'   column of \code{X} (contiguous ids starting at 1).
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2} (the group-ridge penalty, >= 0) and
    #'   \code{tenet_aug_use_lars} (run the inner path with a pure-LARS
    #'   solver that never drops variables).
    initialize = function(X, D, y, groups, options = tsolver_options()) {
      super$initialize(X, D, y)
      groups <- as.integer(groups)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tienet_aug_mmap_create(self$X, self$D, self$y,
                                                   groups, options)
      } else {
        self$ptr <- tsolver_tienet_aug_create(self$X, self$D, self$y,
                                              groups, options)
      }
    },

    #' @description Get the group-ridge penalty lambda2 used.
    #' @return Scalar numeric lambda2.
    get_lambda2 = function() {
      tsolver_tienet_aug_get_lambda2(self$ptr)
    },

    #' @description Get the group ids (1-based, one per predictor).
    #' @return Integer vector of group ids.
    get_groups = function() {
      tsolver_tienet_aug_get_groups(self$ptr)
    },

    #' @description Get the number of disjoint groups.
    #' @return Integer group count.
    get_num_groups = function() {
      tsolver_tienet_aug_get_num_groups(self$ptr)
    }
  )
)


#' T-Stagewise Solver
#'
#' @description
#' Terminating Stagewise Regression solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' T. Hastie, J. Taylor, R. Tibshirani, and G. Walther, "Forward Stagewise Regression and the Monotone Lasso,"
#'   Electron. J. Stat., vol. 1, pp. 1-29, 2007, doi: 10.1214/07-EJS004.
#'
#' R. J. Tibshirani, "A General Framework for Fast Stagewise Algorithms," J. Mach. Learn. Res., vol. 16,
#'   pp. 2543-2588, 2015.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TSTAGEWISE_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TSTAGEWISE_Solver <- R6::R6Class(
  "TSTAGEWISE_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TSTAGEWISE_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tstagewise_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tstagewise_create(self$X, self$D, self$y, options)
      }
    },

    #' @description Get the number of coefficient removals.
    #' @return Integer count of removals.
    get_num_removals = function() {
      tsolver_get_num_removals(self$ptr)
    },

    #' @description Get the cycling ratio (removals / total steps).
    #' @return Scalar numeric cycling ratio.
    get_cycling_ratio = function() {
      tsolver_get_cycling_ratio(self$ptr)
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


# ==============================================================================
# OMP-family subclasses
# ==============================================================================

#' T-OMP Solver
#'
#' @description
#' Terminating Orthogonal Matching Pursuit (T-OMP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' Y. C. Pati, R. Rezaiifar, and P. S. Krishnaprasad, "Orthogonal Matching Pursuit: Recursive Function
#'   Approximation with Applications to Wavelet Decomposition," in Proc. 27th Asilomar Conf. Signals, Syst.,
#'   Comput., 1993, pp. 40-44.
#'
#' J. A. Tropp, "Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Inf. Theory,
#'   vol. 50, no. 10, pp. 2231-2242, 2004, doi: 10.1109/TIT.2004.834793.
#'
#' T. Blumensath and M. E. Davies, "On the Difference Between Orthogonal Matching Pursuit and Orthogonal Least
#'   Squares," unpublished note, draft of March 29, 2007.
#'
#' F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control," in
#'   Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TOMP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TOMP_Solver <- R6::R6Class(
  "TOMP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TOMP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_omp_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_omp_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-GP Solver
#'
#' @description
#' Terminating Gradient Pursuit (T-GP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE Trans. Signal Process., vol. 56, no. 6,
#'   pp. 2370-2382, 2008.
#'
#' F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control," in
#'   Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TGP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TGP_Solver <- R6::R6Class(
  "TGP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TGP_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tgp_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tgp_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-ACGP Solver
#'
#' @description
#' Terminating Approximate Conjugate Gradient Pursuit (T-ACGP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE Trans. Signal Process., vol. 56, no. 6,
#'   pp. 2370-2382, 2008.
#'
#' F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control," in
#'   Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TACGP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TACGP_Solver <- R6::R6Class(
  "TACGP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TACGP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tacgp_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tacgp_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-MP Solver
#'
#' @description
#' Terminating Matching Pursuit (T-MP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' S. G. Mallat and Z. Zhang, "Matching Pursuits with Time-Frequency Dictionaries," IEEE Trans. Signal
#'   Process., vol. 41, no. 12, pp. 3397-3415, 1993, doi: 10.1109/78.258082.
#'
#' F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control," in
#'   Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TMP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TMP_Solver <- R6::R6Class(
  "TMP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TMP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tmp_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tmp_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-OLS Solver
#'
#' @description
#' Terminating Orthogonal Least Squares (T-OLS) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' S. Chen, S. A. Billings, and W. Luo, "Orthogonal Least Squares Methods and Their Application to Non-Linear
#'   System Identification," Int. J. Control, vol. 50, no. 5, pp. 1873-1896, 1989,
#'   doi: 10.1080/00207178908953472.
#'
#' T. Blumensath and M. E. Davies, "On the Difference Between Orthogonal Matching Pursuit and Orthogonal Least
#'   Squares," unpublished note, draft of March 29, 2007.
#'
#' F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control," in
#'   Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TOOLS_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TOOLS_Solver <- R6::R6Class(
  "TOOLS_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TOOLS_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}} (default:
    #'   \code{tsolver_options()}).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tools_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tools_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-NCGMP Solver
#'
#' @description
#' Terminating Norm-Corrected Gradient Matching Pursuit (T-NCGMP) solver.
#' Two variants are available: \code{"line_search"} (standard, 1-D projection)
#' and \code{"fully_corrective"} (OMP-style least squares over the active set).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' F. Locatello, R. Khanna, M. Tschannen, and M. Jaggi, "A Unified Optimization View on Generalized Matching
#'   Pursuit and Frank-Wolfe," in Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS), JMLR: W&CP vol. 54,
#'   2017.
#'
#' F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control," in
#'   Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TNCGMP_Solver$new(X, D, y,
#'                             options = tsolver_options(ncgmp_variant = "line_search"))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TNCGMP_Solver <- R6::R6Class(
  "TNCGMP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TNCGMP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{ncgmp_variant}: \code{"line_search"} (default)
    #'   or \code{"fully_corrective"}.
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tncgmp_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tncgmp_create(self$X, self$D, self$y, options)
      }
    }
  )
)


# ==============================================================================
# AFS-family subclasses
# ==============================================================================

#' T-AFS Solver
#'
#' @description
#' Terminating Adaptive Forward Stepwise (T-AFS) solver.
#' The shrinkage parameter \code{rho} interpolates between Forward Stepwise
#' (\code{rho = 1}) and LAR/LASSO (\code{rho -> 0}).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' I. Zhang and R. Tibshirani, "Adaptive Forward Stepwise: A Method for High Sparsity Regression," J. Mach.
#'   Learn. Res., vol. 27, pp. 1-24, 2026.
#'
#' F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False Discovery Rate Control," in
#'   Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026, pp. 2511-2515.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TAFS_Solver$new(X, D, y, options = tsolver_options(rho_afs = 0.5))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TAFS_Solver <- R6::R6Class(
  "TAFS_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TAFS_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{rho_afs}, the shrinkage parameter in (0, 1]
    #'   (default 1: forward stepwise).
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tafs_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tafs_create(self$X, self$D, self$y, options)
      }
    }
  )
)


# ==============================================================================
# Native T-IENET, the sparse-K solvers and the CD family. All take
# $new(X, D, y, [groups | tikhonov_K,] options).
# ==============================================================================

#' T-IENET Solver (native pathwise)
#'
#' @description
#' Terminating Informed Elastic Net solver, native LARS homotopy on the
#' group-ridge penalty (no augmented copies). The solver used internally by
#' \code{TRexGVSSelector} for \code{gvs_type = "IEN"} with
#' \code{solver = "TIENET"}. Inherits all common methods from
#' \code{\link{TSolver_Base}}.
#'
#' @references
#' J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped Variable Selection and
#'   FDR Control in Genomics Research," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt.
#'   Process. (CAMSAP), 2023, pp. 466-470.
#'
#' H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser. B
#'   Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
#'
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' groups <- c(1, 1, 2, 2)  # 1-based group id per predictor
#' solver <- TIENET_Solver$new(X, D, y, groups = groups,
#'                             options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TIENET_Solver <- R6::R6Class(
  "TIENET_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TIENET_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies (a multiple of \code{ncol(X)}
    #'   columns when \code{lambda2 > 0}: one group-aligned layer per p).
    #' @param y Numeric response vector.
    #' @param groups Integer vector of 1-based group ids, one per predictor
    #'   column of \code{X} (contiguous ids starting at 1).
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2} (the group-ridge penalty, >= 0).
    initialize = function(X, D, y, groups, options = tsolver_options()) {
      super$initialize(X, D, y)
      groups <- as.integer(groups)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tienet_mmap_create(self$X, self$D, self$y, groups, options)
      } else {
        self$ptr <- tsolver_tienet_create(self$X, self$D, self$y, groups, options)
      }
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() tsolver_get_lambda(self$ptr),

    #' @description Get the group-ridge penalty lambda2 used.
    #' @return Scalar numeric lambda2.
    get_lambda2 = function() tsolver_tienet_get_lambda2(self$ptr),

    #' @description Get the group ids (1-based, one per predictor).
    #' @return Integer vector of group ids.
    get_groups = function() tsolver_tienet_get_groups(self$ptr),

    #' @description Get the number of disjoint groups.
    #' @return Integer count of groups.
    get_num_groups = function() tsolver_tienet_get_num_groups(self$ptr)
  )
)


#' T-TIKNET Solver
#'
#' @description
#' Terminating Tikhonov Net solver: native LARS homotopy with a general
#' quadratic penalty \code{lambda2 * t(beta) K beta} on the folded sums (a
#' sparse PSD \code{K = t(Gamma) Gamma}, see \code{\link{trex_gamma_to_k}}).
#' FOLDED dummy coupling only. The solver used internally by
#' \code{TRexTikhonovSelector} for \code{solver = "TIKNET"}. Inherits all
#' common methods from \code{\link{TSolver_Base}}.
#'
#' @references
#' A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization Method," Soviet Math.
#'   Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl. Akad. Nauk SSSR 151 (1963), no. 3,
#'   501-504).
#'
#' Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov Regularization Problems
#'   in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25, 2018, doi: 10.1016/j.cam.2018.04.049.
#'
#' A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems,"
#'   Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article in the Special 40th
#'   Anniversary Issue).
#'
#' Y. Miche, M. van Heeswijk, P. Bas, O. Simula, and A. Lendasse, "TROP-ELM: A Double-Regularized ELM Using
#'   LARS and Tikhonov Regularization," Neurocomputing, vol. 74, pp. 2413-2421, 2011,
#'   doi: 10.1016/j.neucom.2010.12.042.
#'
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' K <- diag(p)
#' solver <- TIKNET_Solver$new(X, D, y, tikhonov_K = K,
#'                             options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TIKNET_Solver <- R6::R6Class(
  "TIKNET_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TIKNET_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies (a multiple of \code{ncol(X)}
    #'   columns when \code{lambda2 > 0}: FOLDED layered convention).
    #' @param y Numeric response vector.
    #' @param tikhonov_K p x p PSD Tikhonov matrix: a
    #'   \code{Matrix::dgCMatrix} or a dense matrix.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2} (the quadratic penalty weight, >= 0).
    initialize = function(X, D, y, tikhonov_K, options = tsolver_options()) {
      super$initialize(X, D, y)
      tikhonov_K <- .as_tikhonov_k(tikhonov_K)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tiknet_mmap_create(self$X, self$D, self$y, tikhonov_K, options)
      } else {
        self$ptr <- tsolver_tiknet_create(self$X, self$D, self$y, tikhonov_K, options)
      }
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() tsolver_get_lambda(self$ptr),

    #' @description Get the quadratic penalty weight lambda2 used.
    #' @return Scalar numeric lambda2.
    get_lambda2 = function() tsolver_tik_get_lambda2(self$ptr),

    #' @description The Tikhonov matrix K used (sparse \code{dgCMatrix}).
    #' @return A \code{Matrix::dgCMatrix} (p x p).
    get_tikhonov_k = function() {
      tsolver_tik_get_tikhonov_k(self$ptr)
    }
  )
)


#' CD-family base class
#'
#' @description
#' Shared getters of the coordinate-descent solver family
#' (\code{TCDLASSO_Solver}, \code{TCDENET_Solver}, \code{TCDIENET_Solver},
#' \code{TCDTIKNET_Solver}): the recorded lambda_1 path, the quadratic penalty
#' weight, and the certified-localization diagnostics. Not intended for direct
#' instantiation.
#'
#' @keywords internal
TCDSolver_Base <- R6::R6Class(
  "TCDSolver_Base",
  inherit = TSolver_Base,
  public = list(
    #' @description Get lambda_1 at every recorded step.
    #' @return Numeric vector, one per step.
    get_lambda = function() tsolver_cd_get_lambda(self$ptr),

    #' @description Get the quadratic penalty weight lambda2 (0 for T-CDLASSO).
    #' @return Scalar numeric lambda2.
    get_lambda2 = function() tsolver_cd_get_lambda2(self$ptr),

    #' @description Is the certified crossing localization active?
    #' @return Logical.
    is_certified_localization = function() tsolver_cd_is_certified_localization(self$ptr),

    #' @description Did certification become unavailable along the path
    #'   (A0-w boundary), so the solver fell back to LEGACY localization?
    #' @return Logical.
    is_certified_degraded = function() tsolver_cd_is_certified_degraded(self$ptr),

    #' @description Get the duality gap at every recorded step.
    #' @return Numeric vector, one per step.
    get_record_gaps = function() tsolver_cd_get_record_gaps(self$ptr),

    #' @description Get the KKT violation at every recorded step.
    #' @return Numeric vector, one per step.
    get_record_kkt_violations = function() tsolver_cd_get_record_kkt_violations(self$ptr),

    #' @description KKT violation of the current state (a certificate of the
    #'   coordinate-descent minimizer).
    #' @param lambda1 The lambda_1 to certify at; negative (default) uses the
    #'   solver's current one.
    #' @return Numeric scalar.
    get_kkt_violation = function(lambda1 = -1) {
      tsolver_cd_get_kkt_violation(self$ptr, as.numeric(lambda1))
    }
  )
)


#' Coerce a Tikhonov matrix argument for the sparse-K solvers.
#'
#' A dense matrix passes through (the C++ side converts it with sparseView());
#' a Matrix object becomes a \code{dgCMatrix} (symmetric \code{dsCMatrix}
#' storage, which \code{Matrix::Matrix(..., sparse = TRUE)} produces for a
#' symmetric K, is not accepted by RcppEigen).
#' @noRd
.as_tikhonov_k <- function(K) {
  if (isS4(K) && !methods::is(K, "dgCMatrix")) {
    K <- methods::as(methods::as(K, "generalMatrix"), "CsparseMatrix")
  }
  K
}


#' T-CDLASSO Solver
#'
#' @description
#' Terminating coordinate-descent LASSO: exact penalized-lasso minimizers at
#' every dummy crossing, localized with certificates (duality gap and
#' coordinate statuses; \code{cd_certified_localization = FALSE} in the
#' options selects the LEGACY probe bisection). The base of the CD family
#' (\code{TCDENET_Solver}, \code{TCDIENET_Solver}, \code{TCDTIKNET_Solver}
#' inherit its getters). Inherits all common methods from
#' \code{\link{TSolver_Base}}.
#'
#' @references
#' R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," J. R. Stat. Soc. Ser. B Stat. Methodol.,
#'   vol. 58, no. 1, pp. 267-288, 1996.
#'
#' J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
#'   Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
#'
#' J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
#'   Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
#'
#' W. J. Fu, "Penalized Regression: The Bridge Versus the Lasso," J. Comput. Graph. Statist., vol. 7,
#'   pp. 397-416, 1998.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TCDLASSO_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#' solver$is_certified_localization()
#'
#' @family T-solvers
#' @export
TCDLASSO_Solver <- R6::R6Class(
  "TCDLASSO_Solver",
  inherit = TCDSolver_Base,
  public = list(
    #' @description Create a new \code{TCDLASSO_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads the \code{cd_*} knobs.
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tcdlasso_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tcdlasso_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-CDENET Solver
#'
#' @description
#' Terminating coordinate-descent Elastic Net (diagonal ridge \code{lambda2},
#' the dummies get \code{kappa_dummy * lambda2}). The solver used internally
#' by \code{TRexGVSSelector} for \code{en_solver = "TCDENET"}. Inherits the
#' CD getters from \code{\link{TCDSolver_Base}} and all common methods from
#' \code{\link{TSolver_Base}}.
#'
#' @references
#' H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser. B
#'   Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
#'
#' J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
#'   Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
#'
#' J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
#'   Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TCDENET_Solver$new(X, D, y, options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TCDENET_Solver <- R6::R6Class(
  "TCDENET_Solver",
  inherit = TCDSolver_Base,
  public = list(
    #' @description Create a new \code{TCDENET_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2}, \code{kappa_dummy} and the \code{cd_*}
    #'   knobs.
    initialize = function(X, D, y, options = tsolver_options()) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tcdenet_mmap_create(self$X, self$D, self$y, options)
      } else {
        self$ptr <- tsolver_tcdenet_create(self$X, self$D, self$y, options)
      }
    }
  )
)


#' T-CDIENET Solver
#'
#' @description
#' Terminating coordinate-descent Informed Elastic Net: exact fixed-lambda1
#' IEN minimizers per dummy crossing with O(1) group-penalty hooks
#' (\code{group_penalty = "MEAN"}, the canonical group-sum penalty, or
#' \code{"CENTERING"}). The solver used internally by \code{TRexGVSSelector}
#' for \code{solver = "TCDIENET"}. Inherits the CD getters from
#' \code{\link{TCDSolver_Base}} and all common methods from
#' \code{\link{TSolver_Base}}.
#'
#' @references
#' J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped Variable Selection and
#'   FDR Control in Genomics Research," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt.
#'   Process. (CAMSAP), 2023, pp. 466-470.
#'
#' H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser. B
#'   Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
#'
#' J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
#'   Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
#'
#' J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
#'   Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TCDIENET_Solver$new(X, D, y, groups = c(1, 1, 2, 2),
#'                               options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TCDIENET_Solver <- R6::R6Class(
  "TCDIENET_Solver",
  inherit = TCDSolver_Base,
  public = list(
    #' @description Create a new \code{TCDIENET_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies (a multiple of \code{ncol(X)}
    #'   columns: one group-aligned layer per p).
    #' @param y Numeric response vector.
    #' @param groups Integer vector of 1-based group ids, one per predictor
    #'   column of \code{X} (contiguous ids starting at 1).
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2}, \code{group_penalty} and the
    #'   \code{cd_*} knobs.
    initialize = function(X, D, y, groups, options = tsolver_options()) {
      super$initialize(X, D, y)
      groups <- as.integer(groups)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tcdienet_mmap_create(self$X, self$D, self$y, groups, options)
      } else {
        self$ptr <- tsolver_tcdienet_create(self$X, self$D, self$y, groups, options)
      }
    },

    #' @description Get the group ids (1-based, one per predictor).
    #' @return Integer vector of group ids.
    get_groups = function() tsolver_tcdienet_get_groups(self$ptr),

    #' @description Get the number of disjoint groups.
    #' @return Integer count of groups.
    get_num_groups = function() tsolver_tcdienet_get_num_groups(self$ptr)
  )
)


#' T-CDTIKNET Solver
#'
#' @description
#' Terminating coordinate-descent Tikhonov Net: exact fixed-lambda1
#' minimizers per dummy crossing under the general quadratic penalty
#' \code{lambda2 * t(beta) K beta} with a sparse PSD \code{K}. Dummy coupling
#' per \code{fold_dummies} in the options (\code{FALSE}: independent
#' \code{kappa_dummy} ridge; \code{TRUE}: FOLDED layered convention). The
#' solver used internally by \code{TRexTikhonovSelector} for
#' \code{solver = "TCDTIKNET"}. Inherits the CD getters from
#' \code{\link{TCDSolver_Base}} and all common methods from
#' \code{\link{TSolver_Base}}.
#'
#' @references
#' A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization Method," Soviet Math.
#'   Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl. Akad. Nauk SSSR 151 (1963), no. 3,
#'   501-504).
#'
#' Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov Regularization Problems
#'   in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25, 2018, doi: 10.1016/j.cam.2018.04.049.
#'
#' J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
#'   Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
#'
#' J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
#'   Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TCDTIKNET_Solver$new(X, D, y, tikhonov_K = diag(p),
#'                                options = tsolver_options(lambda2 = 0.1))
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @family T-solvers
#' @export
TCDTIKNET_Solver <- R6::R6Class(
  "TCDTIKNET_Solver",
  inherit = TCDSolver_Base,
  public = list(
    #' @description Create a new \code{TCDTIKNET_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies (a multiple of \code{ncol(X)}
    #'   columns when \code{fold_dummies = TRUE}).
    #' @param y Numeric response vector.
    #' @param tikhonov_K p x p PSD Tikhonov matrix: a
    #'   \code{Matrix::dgCMatrix} or a dense matrix.
    #' @param options Solver options from \code{\link{tsolver_options}}; this
    #'   solver reads \code{lambda2}, \code{kappa_dummy}, \code{fold_dummies}
    #'   and the \code{cd_*} knobs.
    initialize = function(X, D, y, tikhonov_K, options = tsolver_options()) {
      super$initialize(X, D, y)
      tikhonov_K <- .as_tikhonov_k(tikhonov_K)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tcdtiknet_mmap_create(self$X, self$D, self$y, tikhonov_K, options)
      } else {
        self$ptr <- tsolver_tcdtiknet_create(self$X, self$D, self$y, tikhonov_K, options)
      }
    },

    #' @description Is the FOLDED dummy coupling active?
    #' @return Logical.
    is_folded_dummy_coupling = function() tsolver_tcdtiknet_is_folded(self$ptr)
  )
)
