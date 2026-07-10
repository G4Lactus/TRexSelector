# ==============================================================================
# ts_demo_utils.R
# ==============================================================================
#
# Shared helpers for the standalone terminating-solver (tsolvers) demos.
#
# Mirrors the C++ utilities used by cpp/tsolvers/demo_ts_*:
#   - datagen::SyntheticData with predictor_policy::Normal(),
#     dummygen::Distribution::Normal(), noisegen::noise_policy::Normal()
#   - cdiagnostics::print_section_header / print_talgo_demo_config /
#     print_selection / print_selection_quality
#
# Public functions:
#   gen_synthetic_data(n, p, num_dummies, support, coefs, snr, seed)
#   print_section_header(title)
#   print_demo_config(n, p, num_dummies, T_stop, support, coefs, snr)
#   print_selection(solver, true_support, p)
#   print_selection_quality(solver, true_support, p)
#
# ==============================================================================

# Synthetic regression data with explicit dummy matrix, mirroring
# C++ datagen::SyntheticData:
#
#   X[i, j] iid N(0, 1)   (n x p signal predictors)
#   D[i, j] iid N(0, 1)   (n x num_dummies dummies)
#   y = X[, support] %*% coefs + eps,  eps ~ N(0, noise_sigma^2 I_n)
#
# noise_sigma = sqrt(Var(signal) / snr), with the same small-sample rule as
# C++ noisegen::calculate_noise_params(): Bessel-corrected variance (n - 1)
# for n <= 100, population variance (n) otherwise.
#
# `support` is 1-based (R convention); order pairs positionally with `coefs`.
gen_synthetic_data <- function(n, p, num_dummies, support, coefs,
                               snr = 1.0, seed = NULL) {
  stopifnot(length(support) == length(coefs),
            all(support >= 1L), all(support <= p))
  if (!is.null(seed)) set.seed(seed)

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  D <- matrix(rnorm(n * num_dummies), nrow = n, ncol = num_dummies)

  signal <- as.numeric(X[, support, drop = FALSE] %*% coefs)
  denom <- if (n <= 100) max(n, 2) - 1 else n
  var_sig <- sum((signal - mean(signal))^2) / denom
  noise_sigma <- if (snr > 0 && var_sig > 0) sqrt(var_sig / snr) else 0
  y <- signal + rnorm(n) * noise_sigma

  list(X = X, D = D, y = y)
}


# Mirrors cdiagnostics::print_section_header.
print_section_header <- function(title) {
  bar <- strrep("=", 78)
  cat("\n", bar, "\n", title, "\n", bar, "\n", sep = "")
}


# Mirrors cdiagnostics::print_talgo_demo_config.
print_demo_config <- function(n, p, num_dummies, T_stop, support, coefs, snr) {
  cat("Configuration:\n")
  cat(sprintf("  n            = %d\n", n))
  cat(sprintf("  p            = %d\n", p))
  cat(sprintf("  num_dummies  = %d\n", num_dummies))
  cat(sprintf("  T_stop       = %d\n", T_stop))
  cat("  true_support =", paste(support, collapse = ", "), "  (1-based)\n")
  cat("  true_coefs   =", paste(coefs, collapse = ", "), "\n")
  cat(sprintf("  snr          = %g\n", snr))
}


# Selected predictors and active-dummy count. The R binding's get_actives()
# returns active predictors only (dummies are filtered out), so the number of
# active dummies is reconstructed from the signed action path (+idx = add,
# -idx = drop; 1-based combined indexing, dummies at p+1..p+L).
.split_actives <- function(solver, p) {
  predictors <- sort(solver$get_actives(r_index = TRUE))
  acts <- unlist(solver$get_actions())
  dummy_acts <- acts[abs(acts) > p]
  n_dummies <- sum(dummy_acts > 0) - sum(dummy_acts < 0)
  list(predictors = predictors, n_dummies = n_dummies)
}


# Mirrors cdiagnostics::print_selection: selected predictors (1-based),
# active dummy count, and number of path steps.
print_selection <- function(solver, true_support, p) {
  sp <- .split_actives(solver, p)
  hits <- sort(intersect(sp$predictors, true_support))
  cat("\nSelection:\n")
  cat(sprintf("  Steps taken        : %d\n", solver$get_num_steps()))
  cat("  Selected predictors:", paste(sp$predictors, collapse = ", "), "\n")
  cat("  True support hits  :", paste(hits, collapse = ", "), "\n")
  cat(sprintf("  Active dummies     : %d\n", sp$n_dummies))
}


# Mirrors cdiagnostics::print_selection_quality: TPP/FDP, precision, recall,
# and F1 of the selected predictor set vs. the true support.
print_selection_quality <- function(solver, true_support, p) {
  sp <- .split_actives(solver, p)
  selected <- sp$predictors
  tpp <- TRexSelectorNeo::compute_tpp(selected, true_support)
  fdp <- TRexSelectorNeo::compute_fdp(selected, true_support)
  prec <- TRexSelectorNeo::compute_precision(selected, true_support)
  rec <- TRexSelectorNeo::compute_recall(selected, true_support)
  f1 <- if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
  cat("Selection quality:\n")
  cat(sprintf("  TPP (recall) = %.4f\n", tpp))
  cat(sprintf("  FDP          = %.4f\n", fdp))
  cat(sprintf("  Precision    = %.4f\n", prec))
  cat(sprintf("  Recall       = %.4f\n", rec))
  cat(sprintf("  F1 score     = %.4f\n", f1))
}
