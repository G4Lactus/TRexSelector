#' @name TRexTikhonovSelector
#'
#' @title T-Rex Selector with a general Tikhonov penalty
#'
#' @description FDR-controlled variable selection with a Tikhonov penalty
#'   matrix \code{K = t(Gamma) \%*\% Gamma} driving the Tikhonov-net solvers
#'   (TCDTIKNET by default, or TIKNET). \code{K = I} collapses to
#'   the plain elastic net; a group-mean K reproduces the GVS-IEN penalty;
#'   banded / Laplacian K encode smoothness-informed selection;
#'   \code{penalty_structure = "GROUP_CENTERING"} builds the centering
#'   penalty from the dummy clustering. lambda_2 comes from the caller, a
#'   CV tuner, or the pilot rule (see \code{\link{trex_tikhonov_control}}).
#'   Dummies are cluster-aware and FOLDED-coupled by default.
#'
#' @importFrom R6 R6Class
#'
#' @references
#' A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization Method," Soviet Math.
#'   Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl. Akad. Nauk SSSR 151 (1963), no. 3,
#'   501-504).
#'
#' Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov Regularization Problems
#'   in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25, 2018, doi: 10.1016/j.cam.2018.04.049.
#'
#' A. J. Smola and R. Kondor, "Kernels and Regularization on Graphs," in Learning Theory and Kernel Machines
#'   (COLT/Kernel 2003), Lecture Notes in Artificial Intelligence vol. 2777, 2003, pp. 144-158.
#'
#' F. R. K. Chung, Spectral Graph Theory. CBMS Regional Conference Series in Mathematics, no. 92. American
#'   Mathematical Society, 1997.
#'
#' A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems,"
#'   Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article in the Special 40th
#'   Anniversary Issue).
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
#' ctrl <- trex_tikhonov_control(tikhonov_K = diag(ncol(X)), lambda_2 = 0.5,
#'                               cluster_dummies = FALSE,
#'                               fold_dummy_coupling = FALSE)
#' sel <- TRexTikhonovSelector$new(X, y, tFDR = 0.1, verbose = FALSE,
#'                                 tik_control = ctrl)
#' sel$select()
#' sel$selected_indices
#' }
#' @family T-Rex selectors and controls
#' @export
TRexTikhonovSelector <- R6::R6Class("TRexTikhonovSelector",
  inherit = TRexSelector,
  public = list(
    #' @description Create a new TRexTikhonovSelector object.
    #'
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param tik_control A Tikhonov control list from
    #'   \code{\link{trex_tikhonov_control}()} (required — it carries the
    #'   penalty matrix K).
    #' @param control A control list from \code{\link{trex_control}()}
    #'   (default: \code{trex_control()}). The \code{solver} field selects
    #'   the Tikhonov-family solver: \code{"TCDTIKNET"} (default; the
    #'   neutral \code{trex_control()} default \code{"TLARS"} maps to it),
    #'   or \code{"TIKNET"} (native pathwise); any other value warns and
    #'   falls back to \code{"TCDTIKNET"}.
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          tik_control,
                          control = trex_control()) {

      if (missing(tik_control)) {
        stop("tik_control is required (build it with trex_tikhonov_control(); ",
             "it carries the penalty matrix K or the penalty structure).")
      }

      # Only the Tikhonov family has a general-K mode. "TLARS" is the
      # trex_control() default and cannot be distinguished from an explicit
      # choice, so it never warns; the C++ glue maps it to TCDTIKNET.
      tik_family <- c("TCDTIKNET", "TIKNET")
      if (!(control$solver %in% c("TLARS", tik_family))) {
        warning("control$solver = \"", control$solver,
                "\" is not a Tikhonov-family solver; TRexTikhonovSelector ",
                "accepts \"TCDTIKNET\" or \"TIKNET\". ",
                "Using \"TCDTIKNET\".")
        control$solver <- "TLARS"  # the C++ glue defaults it to TCDTIKNET
      }

      if (inherits(X, "MemoryMappedMatrix")) {
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_tikhonov_mmap_create(
          get_ptr(X), y, tFDR, tik_control, control, seed, verbose
        )
      } else {
        X <- .as_double_matrix(X)
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_tikhonov_create(
          X, y, tFDR, tik_control, control, seed, verbose
        )
      }
    },

    #' @description Run the Tikhonov T-Rex algorithm.
    select = function() {
      trex_tikhonov_select(private$ptr)
      invisible(self)
    }
  ),
  active = list(
    #' @field lambda2_used The lambda_2 actually used by the last
    #'   \code{select()} run (user-supplied or CV-resolved; solver scale).
    #'   \code{-1} before \code{select()}.
    lambda2_used = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_tikhonov_get_lambda2_used(private$ptr)
    },

    #' @field lambda2_cv_diagnostics Health record of the lambda_2 tuner of
    #'   the last \code{select()}: a list with \code{ran} (a tuner or the
    #'   pilot ran), \code{seconds}, \code{n_profile_at_edge} (lambda_2
    #'   values whose profiled lambda_1 optimum sat on a truncation edge; 0
    #'   is healthy), \code{support_cap} (-1 = disabled),
    #'   \code{dfmax_ratio} (IEN tuner: the \code{cv_dfmax_ratio} it was
    #'   called with; 0 otherwise), \code{low_dimensional}, and the
    #'   boundary-pick guard fields \code{grid_index} / \code{grid_size},
    #'   \code{at_grid_edge}, \code{profile_at_edge},
    #'   \code{profile_at_support_cap} and \code{boundary_pick} (see
    #'   \code{TRexGVSSelector}; the CV methods warn once on a boundary pick
    #'   and return it unchanged; the pilot rule leaves them at their
    #'   defaults).
    lambda2_cv_diagnostics = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_tikhonov_get_lambda2_cv_diagnostics(private$ptr)
    },

    #' @field lambda2_pilot Record of the lambda_2 pilot rule of the last
    #'   \code{select()} (\code{ran == FALSE} when no pilot ran): the
    #'   \code{grid}, the \code{counts} the pick was read off, the
    #'   per-grid-point curves over the pilot stopping times
    #'   (\code{curve_majority}, \code{curve_soft}; G x T matrices), the
    #'   \code{calibrated} and \code{group_majority} counts, the 1-based
    #'   \code{index} of the pick, the picked \code{lambda2},
    #'   \code{max_selected}, \code{n_ties}, the boundary flags
    #'   \code{at_lower_end} / \code{at_upper_end} (widen the grid before
    #'   trusting a boundary pick), the budget actually used (\code{K_pilot},
    #'   \code{T_pilot}, \code{L_0}) and the timings.
    lambda2_pilot = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_tikhonov_get_lambda2_pilot_result(private$ptr)
    }
  )
)


#' @name trex_tikhonov_lambda2_power_rule
#'
#' @title The knee of the full lambda_2 profile of the Tikhonov T-Rex
#'
#' @description Runs one FDR-controlled Tikhonov T-Rex selection per value of
#'   \code{grid} with the same seed, so the probes share their dummies and
#'   differ by the penalty weight alone, and picks the smallest lambda_2 whose
#'   selection is within \code{knee_tolerance} selections of the largest (the
#'   knee of the profile). This is the reference rule behind the default
#'   \code{lambda2_method = "PILOT_KNEE"} of
#'   \code{\link{trex_tikhonov_control}}, which reads the same knee off a few
#'   random experiments per grid value at a fraction of the cost; the full
#'   profile costs one selection per grid value and returns every probe's
#'   selection. The pick depends on y and lies outside the fixed-lambda_2 FDR
#'   guarantee that each probe carries.
#'
#' @param X Feature matrix (n x p). Every probe normalizes it in place and
#'   hands it back, so \code{X} is unchanged on return and never copied.
#' @param y Response vector (length n).
#' @param tFDR Target FDR level (default: 0.1).
#' @param tik_control A Tikhonov control list from
#'   \code{\link{trex_tikhonov_control}()}; its \code{lambda_2} is
#'   overwritten per probe, everything else is shared by every probe.
#' @param control A control list from \code{\link{trex_control}()} (default:
#'   \code{trex_control()}); the solver maps as in
#'   \code{\link{TRexTikhonovSelector}}.
#' @param seed Selector seed, the same for every probe (default: 0). A
#'   negative seed would draw fresh dummies per probe and is refused.
#' @param grid lambda_2 candidates on the solver scale (default
#'   \code{c(0, 1, 3, 10, 30, 100, 300, 1000, 3000)}), sorted and
#'   de-duplicated. Keep 0 on it so that "no penalty" is an admissible
#'   answer.
#' @param knee_tolerance Selections below the largest count that still
#'   count as "on the plateau" (default: 1); 0 reads the largest selection,
#'   ties toward the smallest lambda_2.
#' @param verbose Whether the probes print their progress (default:
#'   \code{FALSE}).
#'
#' @return A list: \code{lambda2} (the pick), \code{index} (1-based, into
#'   \code{grid}), \code{selected} (the 1-based selection at the pick),
#'   \code{grid} (ascending, de-duplicated), \code{counts} (selection sizes
#'   per grid value), \code{probes} (one list per grid value with
#'   \code{lambda2}, \code{selected}, \code{T_stop} and \code{seconds}),
#'   \code{max_selected}, \code{n_ties} (grid values within the tolerance
#'   of the maximum), the boundary flags \code{at_lower_end} /
#'   \code{at_upper_end} (widen the grid before trusting a boundary pick)
#'   and \code{seconds}.
#'
#' @examples
#' \donttest{
#' data(Gauss_data)
#' X <- Gauss_data$X
#' y <- Gauss_data$y
#' ctrl <- trex_tikhonov_control(tikhonov_K = diag(ncol(X)))
#' prof <- trex_tikhonov_lambda2_power_rule(X, y, tFDR = 0.1, tik_control = ctrl,
#'                                          grid = c(0, 1, 10), seed = 1)
#' prof$lambda2
#' prof$counts
#' }
#' @family T-Rex selectors and controls
#' @export
trex_tikhonov_lambda2_power_rule <- function(X, y,
                                             tFDR = 0.1,
                                             tik_control,
                                             control = trex_control(),
                                             seed = 0,
                                             grid = c(0, 1, 3, 10, 30, 100,
                                                      300, 1000, 3000),
                                             knee_tolerance = 1,
                                             verbose = FALSE) {
  if (missing(tik_control)) {
    stop("tik_control is required (build it with trex_tikhonov_control(); ",
         "it carries the penalty matrix K or the penalty structure).")
  }
  if (inherits(X, "MemoryMappedMatrix")) {
    stop("trex_tikhonov_lambda2_power_rule() takes an in-memory matrix X.")
  }
  if (!is.numeric(seed) || length(seed) != 1 || !is.finite(seed) || seed < 0) {
    stop("seed must be a single non-negative integer: the probes share ",
         "their dummies. Got: ", format(seed))
  }
  if (!is.numeric(grid) || length(grid) == 0 || any(!is.finite(grid)) ||
      any(grid < 0)) {
    stop("grid must be a non-empty vector of finite values >= 0.")
  }
  if (!is.numeric(knee_tolerance) || length(knee_tolerance) != 1 ||
      !is.finite(knee_tolerance) || knee_tolerance < 0) {
    stop("knee_tolerance must be a single number >= 0. Got: ",
         format(knee_tolerance))
  }
  tik_family <- c("TCDTIKNET", "TIKNET")
  if (!(control$solver %in% c("TLARS", tik_family))) {
    warning("control$solver = \"", control$solver,
            "\" is not a Tikhonov-family solver; the Tikhonov T-Rex ",
            "accepts \"TCDTIKNET\" or \"TIKNET\". Using \"TCDTIKNET\".")
    control$solver <- "TLARS"  # the C++ glue defaults it to TCDTIKNET
  }
  X <- .as_double_matrix(X)
  y <- .as_double_vector(y)
  trex_tikhonov_lambda2_power_rule_impl(X, y, tFDR, tik_control, control,
                                        as.integer(seed), as.numeric(grid),
                                        as.numeric(knee_tolerance), verbose)
}


#' @name trex_gamma_to_k
#'
#' @title Build a Tikhonov penalty matrix K from an operator Gamma
#'
#' @description Computes \code{K = t(Gamma) \%*\% Gamma} (pruned and
#'   compressed) for use as \code{tikhonov_K} in
#'   \code{\link{trex_tikhonov_control}}. For example, a first-difference
#'   Gamma yields the smoothness-encoding graph Laplacian of a chain.
#'
#' @param gamma Regularization operator (m x p): a base \code{matrix} or a
#'   sparse \code{Matrix::dgCMatrix}.
#'
#' @return A sparse \code{Matrix::dgCMatrix} (p x p). Requires the Matrix
#'   package (Suggests) for the sparse return class.
#'
#' @examples
#' # First-difference operator over 5 variables -> chain Laplacian
#' p <- 5
#' gamma <- diff(diag(p))
#' K <- trex_gamma_to_k(gamma)
#'
#' @family T-Rex selectors and controls
#' @export
trex_gamma_to_k <- function(gamma) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("trex_gamma_to_k() returns a sparse Matrix::dgCMatrix and needs the ",
         "Matrix package installed.")
  }
  trex_tikhonov_gamma_to_k(gamma)
}
