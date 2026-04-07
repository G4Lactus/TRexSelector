# ==============================================================================
# demo_trex_da.R
#
# Demonstration of DA-TREX on sparse high-dimensional data.
#
# One scenario is constructed for each dependency model:
#
#   1. AR(1)         -- ordered predictors, Toeplitz covariance
#   2. Equi          -- single latent factor, compound symmetry
#   3. NN            -- banded MA(kappa) covariance
#   4. BT            -- two-level hierarchical block structure
#   5. Prior groups  -- multi-level latent factors, known structure
#
# Each scenario uses n = 150, p = 500, s = 5 (p >> n, strongly sparse).
#
# Usage:
#   source("dgp_functions.R")
#   source("trex_da.R")
#   source("demo_trex_da.R")
# ==============================================================================

source("dgp_functions.R")
source("trex_da.R")

# ── Global parameters ─────────────────────────────────────────────────────────

PARAMS <- list(
  n         = 150,   # observations
  p         = 500,   # predictors  (p/n ratio = 3.3 -- strongly overdetermined)
  s         = 5,     # active predictors
  amplitude = 3,     # signal amplitude for each active beta_j
  sigma     = 1,     # noise standard deviation
  K         = 20,    # random experiments per selector run
  tFDR      = 0.2,   # target FDR level
  seed      = 2026   # global RNG seed
)


# ── Evaluation helper ─────────────────────────────────────────────────────────

#' Evaluate a trex_da result against the ground-truth coefficient vector.
#'
#' @param result   Return value of trex_da().
#' @param beta     True coefficient vector (length p).
#'
#' @return Named list: n_selected, TP, FP, TPP, FDP,
#'                     T_stop, num_dummies, v_thresh, rho_thresh.
evaluate <- function(result, beta) {
  true_set <- which(abs(beta) > .Machine$double.eps)
  sel_set  <- which(result$selected_var == 1L)

  n_sel <- length(sel_set)
  n_tp  <- length(intersect(sel_set, true_set))
  n_fp  <- n_sel - n_tp

  list(
    n_selected = n_sel,
    TP         = n_tp,
    FP         = n_fp,
    TPP        = n_tp / length(true_set),
    FDP        = n_fp / max(1L, n_sel),
    T_stop     = result$T_stop,
    num_dummies= result$num_dummies,
    v_thresh   = result$v_thresh,
    rho_thresh = result$rho_thresh
  )
}


#' Print a formatted single-scenario result block.
.print_result <- function(scenario_name, dat, result) {
  ev <- evaluate(result, dat$beta)
  cat("\n")
  cat(strrep("─", 60), "\n")
  cat(sprintf("  %s\n", scenario_name))
  cat(strrep("─", 60), "\n")
  cat(sprintf("  Data:       n = %d,  p = %d,  s = %d,  SNR = %.2f\n",
              dat$n, dat$p, dat$s, dat$snr))
  cat(sprintf("  Calibration: T_stop = %d,  dummies = %d,  v* = %.3f\n",
              ev$T_stop, ev$num_dummies, ev$v_thresh))
  if (!is.na(ev$rho_thresh))
    cat(sprintf("               rho* = %.4f\n", ev$rho_thresh))
  cat(sprintf("  Selection:   %d selected  |  TP = %d  FP = %d\n",
              ev$n_selected, ev$TP, ev$FP))
  cat(sprintf("               TPP = %.2f  |  FDP = %.2f  (target <= %.2f)\n",
              ev$TPP, ev$FDP, PARAMS$tFDR))
  cat(strrep("─", 60), "\n")
  invisible(ev)
}


# ── Scenario 1: AR(1) ─────────────────────────────────────────────────────────
#
# Predictor correlation decays exponentially with variable index distance.
# The window half-width kappa is estimated automatically from the data as
# kappa = ceil(log(rho_thr) / log(rho_hat)).

cat("\n======================================================================\n")
cat("  DA-TREX Demo  |  n =", PARAMS$n, " p =", PARAMS$p, " s =", PARAMS$s, "\n")
cat("======================================================================\n")

cat("\n[1/5] Generating AR(1) data  (rho = 0.7) ...\n")
dat_ar1 <- dgp_ar1(
  n         = PARAMS$n,
  p         = PARAMS$p,
  rho       = 0.7,
  s         = PARAMS$s,
  amplitude = PARAMS$amplitude,
  sigma     = PARAMS$sigma,
  seed      = PARAMS$seed
)

cat("[1/5] Running trex+DA+AR1 ...\n")
res_ar1 <- trex_da(
  X       = dat_ar1$X,
  y       = dat_ar1$y,
  method  = "trex+DA+AR1",
  K       = PARAMS$K,
  tFDR    = PARAMS$tFDR,
  verbose = FALSE,
  seed    = PARAMS$seed
)
ev_ar1 <- .print_result("Scenario 1 -- AR(1)  [method = trex+DA+AR1]", dat_ar1, res_ar1)


# ── Scenario 2: Equicorrelation ───────────────────────────────────────────────
#
# All predictors share one common latent factor. The equicorrelation
# coefficient is estimated as the mean of the lower triangle of R_hat.

cat("\n[2/5] Generating equicorrelation data  (rho = 0.5) ...\n")
dat_equi <- dgp_equi(
  n         = PARAMS$n,
  p         = PARAMS$p,
  rho       = 0.5,
  s         = PARAMS$s,
  amplitude = PARAMS$amplitude,
  sigma     = PARAMS$sigma,
  seed      = PARAMS$seed
)

cat("[2/5] Running trex+DA+equi ...\n")
res_equi <- trex_da(
  X       = dat_equi$X,
  y       = dat_equi$y,
  method  = "trex+DA+equi",
  K       = PARAMS$K,
  tFDR    = PARAMS$tFDR,
  verbose = FALSE,
  seed    = PARAMS$seed
)
ev_equi <- .print_result("Scenario 2 -- Equicorrelation  [method = trex+DA+equi]",
                          dat_equi, res_equi)


# ── Scenario 3: Nearest Neighbours ───────────────────────────────────────────
#
# Correlation is non-zero only within a bandwidth of kappa = 3 lags.
# The NN route sweeps over correlation thresholds to discover the bandwidth.

cat("\n[3/5] Generating NN data  (kappa = 3,  rho = 0.7) ...\n")
dat_nn <- dgp_nn(
  n         = PARAMS$n,
  p         = PARAMS$p,
  kappa     = 3,
  rho       = 0.7,
  s         = PARAMS$s,
  amplitude = PARAMS$amplitude,
  sigma     = PARAMS$sigma,
  seed      = PARAMS$seed
)

cat("[3/5] Running trex+DA+NN ...\n")
res_nn <- trex_da(
  X       = dat_nn$X,
  y       = dat_nn$y,
  method  = "trex+DA+NN",
  K       = PARAMS$K,
  tFDR    = PARAMS$tFDR,
  verbose = FALSE,
  seed    = PARAMS$seed
)
ev_nn <- .print_result("Scenario 3 -- NN  [method = trex+DA+NN]", dat_nn, res_nn)


# ── Scenario 4: Binary Tree (Hierarchical Clustering) ────────────────────────
#
# Two-level block structure: 10 blocks of 50 predictors each.
# Within-block correlation = 0.8, between-block correlation = 0.1.
# The BT route builds a dendrogram from the sample correlation matrix and
# sweeps over its cut heights.

cat("\n[4/5] Generating BT data  (10 blocks of 50,  rho_w = 0.8,  rho_b = 0.1) ...\n")
dat_bt <- dgp_bt(
  n            = PARAMS$n,
  p            = PARAMS$p,
  n_blocks     = 10,
  rho_within   = 0.8,
  rho_between  = 0.1,
  s            = PARAMS$s,
  amplitude    = PARAMS$amplitude,
  sigma        = PARAMS$sigma,
  seed         = PARAMS$seed
)

cat("[4/5] Running trex+DA+BT ...\n")
res_bt <- trex_da(
  X       = dat_bt$X,
  y       = dat_bt$y,
  method  = "trex+DA+BT",
  hc_dist = "average",
  K       = PARAMS$K,
  tFDR    = PARAMS$tFDR,
  verbose = FALSE,
  seed    = PARAMS$seed
)
ev_bt <- .print_result("Scenario 4 -- BT  [method = trex+DA+BT,  hc_dist = average]",
                        dat_bt, res_bt)


# ── Scenario 5: Prior Groups ──────────────────────────────────────────────────
#
# Three-level hierarchy for p = 500 predictors:
#
#   Level 1 (fine):   50 groups of 10   -- rho = 0.80
#   Level 2 (medium): 10 groups of 50   -- rho = 0.50
#   Level 3 (coarse):  2 groups of 250  -- rho = 0.20
#
# The group structure is passed directly to both the DGP (dgp_groups) and
# the selector (trex_da), bypassing all dendrogram computation.

cat("\n[5/5] Generating prior-groups data  (3 levels: 50/10/2 groups) ...\n")

groups_demo <- list(
  rep(seq(1,  50), each = 10),   # level 1: 50 groups of 10
  rep(seq(1,  10), each = 50),   # level 2: 10 groups of 50
  rep(seq(1,   2), each = 250)   # level 3:  2 groups of 250
)
rho_levels_demo <- c(0.80, 0.50, 0.20)

dat_grp <- dgp_groups(
  n          = PARAMS$n,
  p          = PARAMS$p,
  groups     = groups_demo,
  rho_levels = rho_levels_demo,
  s          = PARAMS$s,
  amplitude  = PARAMS$amplitude,
  sigma      = PARAMS$sigma,
  seed       = PARAMS$seed
)

cat("[5/5] Running trex+DA with prior groups ...\n")
res_grp <- trex_da(
  X               = dat_grp$X,
  y               = dat_grp$y,
  groups          = groups_demo,
  rho_grid_labels = rho_levels_demo,
  K               = PARAMS$K,
  tFDR            = PARAMS$tFDR,
  verbose         = FALSE,
  seed            = PARAMS$seed
)
ev_grp <- .print_result("Scenario 5 -- Prior groups  [groups supplied directly]",
                         dat_grp, res_grp)


# ── Summary table ─────────────────────────────────────────────────────────────

cat("\n")
cat(strrep("=", 78), "\n")
cat(sprintf("  %-28s  %4s  %3s  %3s  %6s  %6s  %6s\n",
            "Scenario", "Sel.", "TP", "FP", "TPP", "FDP", "T_stop"))
cat(strrep("─", 78), "\n")

scenarios <- list(
  list(name = "AR(1)        trex+DA+AR1",   ev = ev_ar1),
  list(name = "Equi         trex+DA+equi",  ev = ev_equi),
  list(name = "NN           trex+DA+NN",    ev = ev_nn),
  list(name = "BT           trex+DA+BT",    ev = ev_bt),
  list(name = "Prior groups groups=list()", ev = ev_grp)
)

for (sc in scenarios) {
  ev <- sc$ev
  cat(sprintf("  %-28s  %4d  %3d  %3d  %6.3f  %6.3f  %6d\n",
              sc$name,
              ev$n_selected, ev$TP, ev$FP,
              ev$TPP, ev$FDP, ev$T_stop))
}
cat(strrep("=", 78), "\n")
cat(sprintf("  Target FDR: %.2f   |   True active set: first %d variables\n",
            PARAMS$tFDR, PARAMS$s))
cat(strrep("=", 78), "\n\n")
