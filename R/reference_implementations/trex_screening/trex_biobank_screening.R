# =============================================================================
# trex_biobank_screening.R — Biobank Screening Workflow (Algorithm 1)
# Machkour, Muma, Palomar (SSP 2023, doi:10.1109/SSP53291.2023.10207957)
#
# This file implements the outer biobank-screening workflow from Algorithm 1
# of the paper. It is intentionally separate from trex_screening.R,
# which implements a single-phenotype run (Step 2.1 below).
#
# Algorithm structure
# -------------------
# Input : alpha   – target FDR for the fallback T-Rex selector
#         alpha_l – lower bound on the accepted estimated FDR
#         alpha_u – upper bound on the accepted estimated FDR
#
# For each phenotype (X_i, y_i) in the biobank:
#
#   Step 2.1  Run Screen-T-Rex (both ordinary and confidence-based versions)
#             and obtain:
#               alpha_hat   = 1 / (R_{1,p}(0.5) v 1)          [ordinary]
#               alpha_hat_C = 1 / (R^(C)_{1,p}(gamma) v 1)    [conf.-based]
#
#   Step 2.2  Select the final active set A_hat according to:
#
#             Case 1 (conf.-based preferred):
#               alpha_l <= alpha_hat_C <= alpha_u  AND
#               max{ alpha_hat_C, alpha_hat * I(alpha_hat <= alpha_u) }
#                 == alpha_hat_C
#               => A_hat = A_hat^(C)_p(gamma, 1)
#
#             Case 2 (ordinary preferred):
#               alpha_l <= alpha_hat <= alpha_u  AND
#               max{ alpha_hat_C * I(alpha_hat_C <= alpha_u), alpha_hat }
#                 == alpha_hat
#               => A_hat = A_hat_p(0.5, 1)
#
#             Tie-break convention (alpha_hat == alpha_hat_C, both cases met):
#               => A_hat = A_hat^(C)_p(gamma, 1)   [conf.-based wins]
#
#             Otherwise:
#               => A_hat = empty set
#
#   Step 2.3  If A_hat == empty set, fall back to the full T-Rex selector
#             with target FDR alpha and obtain A_hat = A_hat_L(v*, T*).
#
# Output: list of selected active sets, one per phenotype.
# =============================================================================


# =============================================================================
# Dependencies (must be sourced first):
#   source("trex_screening.R")          # provides screen_trex()
#   library(TRexSelector)               # provides trex()  [Step 2.3 fallback]
# =============================================================================


# -----------------------------------------------------------------------------
# .select_active_set
#
# Implements the case logic of Step 2.2 of Algorithm 1.
# Returns a named list with the chosen active set and the rule that was applied.
#
# @param A_hat_ord   Binary vector (length p): ordinary Screen-T-Rex result.
# @param A_hat_conf  Binary vector (length p): confidence-based result.
# @param alpha_hat   Scalar: ordinary FDR estimate.
# @param alpha_hat_C Scalar: confidence-based FDR estimate.
# @param alpha_l     Scalar: lower FDR bound.
# @param alpha_u     Scalar: upper FDR bound.
#
# @return Named list:
#   $A_hat   – selected binary support vector (or NULL for empty set)
#   $rule    – character: "confidence", "ordinary", or "empty"
# -----------------------------------------------------------------------------
.select_active_set <- function(A_hat_ord,
                                A_hat_conf,
                                alpha_hat,
                                alpha_hat_C,
                                alpha_l,
                                alpha_u) {

  # Indicator functions from the paper
  I_ord_in_range  <- as.numeric(alpha_hat   <= alpha_u)
  I_conf_in_range <- as.numeric(alpha_hat_C <= alpha_u)

  # Case 1: confidence-based result is preferred
  case1_fdr_ok    <- alpha_l <= alpha_hat_C && alpha_hat_C <= alpha_u
  case1_dominates <- max(alpha_hat_C,
                         alpha_hat * I_ord_in_range) == alpha_hat_C

  # Case 2: ordinary result is preferred
  case2_fdr_ok    <- alpha_l <= alpha_hat && alpha_hat <= alpha_u
  case2_dominates <- max(alpha_hat_C * I_conf_in_range,
                         alpha_hat) == alpha_hat

  # Tie-break: if both cases fire and alpha_hat == alpha_hat_C,
  # the paper's convention gives the confidence-based result (Section III.C).
  if (case1_fdr_ok && case1_dominates) {
    return(list(A_hat = A_hat_conf, rule = "confidence"))
  }

  if (case2_fdr_ok && case2_dominates) {
    return(list(A_hat = A_hat_ord, rule = "ordinary"))
  }

  list(A_hat = NULL, rule = "empty")
}


# -----------------------------------------------------------------------------
# screen_trex_biobank
#
# Screens a set of phenotypes (responses) against a common predictor matrix
# following Algorithm 1 of the Screen-T-Rex paper.
#
# Both Screen-T-Rex versions share the same K random experiments per phenotype
# (the ordinary and confidence-based results are derived in a single call via
# an internal two-pass approach, saving K experiment runs).
#
# @param X           Real valued predictor matrix (n x p), shared across all
#                    phenotypes.
# @param Y           Response matrix (n x num_phenotypes), one column per
#                    phenotype. May also be a plain vector for a single phenotype.
# @param alpha       Target FDR level for the fallback T-Rex selector.
# @param alpha_l     Lower bound on the accepted estimated FDR (default: 0).
# @param alpha_u     Upper bound on the accepted estimated FDR (default: 1).
# @param K           Number of random experiments per phenotype (default: 20).
# @param R           Number of bootstrap resamples for the confidence-based
#                    version (default: 1000).
# @param method      Selector variant passed to screen_trex() (default: 'trex').
# @param conf_level_grid Confidence level grid for bootstrap CIs.
# @param verbose     Logical. If TRUE print per-phenotype progress.
# @param ...         Additional arguments forwarded to screen_trex().
#
# @return A named list with one entry per phenotype, each containing:
#   $A_hat         – integer support vector (1 = selected, 0 = not selected),
#                    or NULL if the T-Rex fallback was triggered.
#   $rule          – character: "confidence", "ordinary", "empty", or "trex"
#   $alpha_hat     – ordinary Screen-T-Rex FDR estimate
#   $alpha_hat_C   – confidence-based Screen-T-Rex FDR estimate
#   $trex_result   – full T-Rex result (only when rule == "trex", else NULL)
#
# @importFrom TRexSelector trex
#
# @export
#
# @examples
# data("Gauss_data")
# X <- Gauss_data$X
# y <- c(Gauss_data$y)
# set.seed(123)
# res <- screen_trex_biobank(X = X, Y = y, alpha = 0.1, alpha_l = 0.05,
#                            alpha_u = 0.20)
# res[[1]]$rule
# res[[1]]$A_hat
screen_trex_biobank <- function(X,
                                 Y,
                                 alpha          = 0.1,
                                 alpha_l        = 0,
                                 alpha_u        = 1,
                                 K              = 20,
                                 R              = 1000,
                                 method         = "trex",
                                 conf_level_grid = seq(0, 1, by = 0.001),
                                 verbose        = TRUE,
                                 ...) {

  # ---------------------------------------------------------------------------
  # Coerce Y to a matrix so we can iterate over columns uniformly
  # ---------------------------------------------------------------------------
  if (is.vector(Y) || (is.matrix(Y) && ncol(Y) == 1L)) {
    Y            <- matrix(drop(Y), ncol = 1L)
    colnames(Y)  <- colnames(Y) %||% "phenotype_1"
  }
  stopifnot(is.matrix(Y), nrow(Y) == nrow(X))

  num_phenotypes <- ncol(Y)
  results        <- vector("list", length = num_phenotypes)
  names(results) <- colnames(Y) %||% paste0("phenotype_", seq_len(num_phenotypes))

  # ---------------------------------------------------------------------------
  # Loop over phenotypes — Algorithm 1, Step 2
  # ---------------------------------------------------------------------------
  for (i in seq_len(num_phenotypes)) {

    y_i <- Y[, i]

    if (verbose) {
      message(sprintf("[%d/%d] Phenotype: %s",
                      i, num_phenotypes, names(results)[i]))
    }

    # -------------------------------------------------------------------------
    # Step 2.1 — Run Screen-T-Rex (ordinary + confidence-based)
    #
    # Both variants share the same random experiments (K LARS runs).
    # We first run the ordinary version and then re-apply the bootstrap
    # screening criterion to the same phi_T_mat / beta matrices.
    # -------------------------------------------------------------------------
    res_ord <- screen_trex(
      X               = X,
      y               = y_i,
      K               = K,
      R               = R,
      method          = method,
      bootstrap       = FALSE,
      conf_level_grid = conf_level_grid,
      verbose         = FALSE,
      ...
    )

    # Reuse the relative occurrences and coefficient matrices from the
    # ordinary run for the confidence-based screening step (no re-running
    # of the random experiments).
    screen_conf <- .screen_bootstrap(
      beta_mat        = res_ord$beta_mat,
      dummy_beta_mat  = res_ord$dummy_beta_mat,
      Phi             = res_ord$Phi,
      R               = R,
      conf_level_grid = conf_level_grid,
      eps             = .Machine$double.eps
    )

    p              <- length(res_ord$selected_var)
    A_hat_conf     <- integer(p)
    A_hat_conf[!screen_conf$set_zero] <- 1L

    alpha_hat   <- res_ord$FDR_estimate
    alpha_hat_C <- if (any(!screen_conf$set_zero)) {
      1L / sum(!screen_conf$set_zero)   # T_stop = 1
    } else {
      0
    }

    if (verbose) {
      message(sprintf("  alpha_hat = %.4f  |  alpha_hat_C = %.4f",
                      alpha_hat, alpha_hat_C))
    }

    # -------------------------------------------------------------------------
    # Step 2.2 — Select the final active set
    # -------------------------------------------------------------------------
    sel <- .select_active_set(
      A_hat_ord   = res_ord$selected_var,
      A_hat_conf  = A_hat_conf,
      alpha_hat   = alpha_hat,
      alpha_hat_C = alpha_hat_C,
      alpha_l     = alpha_l,
      alpha_u     = alpha_u
    )

    trex_result <- NULL

    # -------------------------------------------------------------------------
    # Step 2.3 — Fallback to full T-Rex selector if A_hat == empty set
    # -------------------------------------------------------------------------
    if (sel$rule == "empty") {
      if (verbose) {
        message("  Both Screen-T-Rex FDR estimates outside [alpha_l, alpha_u].",
                " Falling back to T-Rex (target FDR = ", alpha, ").")
      }

      trex_result <- TRexSelector::trex(
        X     = X,
        y     = y_i,
        tFDR  = alpha,
        ...
      )

      sel <- list(
        A_hat = trex_result$selected_var,
        rule  = "trex"
      )
    }

    results[[i]] <- list(
      A_hat       = sel$A_hat,
      rule        = sel$rule,
      alpha_hat   = alpha_hat,
      alpha_hat_C = alpha_hat_C,
      trex_result = trex_result
    )
  }

  results
}


# -----------------------------------------------------------------------------
# Utility: %||%  (NULL-coalescing operator, in case base R < 4.4)
# -----------------------------------------------------------------------------
`%||%` <- function(x, y) if (!is.null(x)) x else y
