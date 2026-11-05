# =======================================================================================
# TRexTikhonovSelector - Unit tests (general-Tikhonov T-Rex through R)
# =======================================================================================

# Grouped design shared by the anchor tests: blocks of 5 correlated
# columns, signal on the first three variables of block 1.
#
# The dimensions mirror the C++ FOLDED anchor (n = 150, p = 40,
# rho = 0.75) DELIBERATELY: the FOLDED sparse-K penalty and GVS's GROUP
# hooks are mathematically identical but accumulate in different
# floating-point orders (w += lambda2*delta*K.col vs sigma_m += delta,
# multiplied once), so selections can differ at ULP-level margins. On a
# knife-edge design (n = 80, rho = 0.85) the GVS anchor flipped 2
# borderline variables; keep the margins wide here.
.make_tik_data <- function(n = 150, p = 40, rho = 0.75, data_seed = 42) {
  set.seed(data_seed)
  n_groups <- p / 5
  z <- matrix(rnorm(n * n_groups), n, n_groups)
  X <- matrix(0, n, p)
  for (g in 1:n_groups) {
    for (j in 1:5) {
      X[, (g - 1) * 5 + j] <- rho * z[, g] + sqrt(1 - rho^2) * rnorm(n)
    }
  }
  y <- as.vector(rowSums(X[, 1:3]) * 3 + rnorm(n))
  groups <- rep(1:n_groups, each = 5)
  list(X = X, y = y, groups = groups, p = p)
}

# Group-mean K = sum_m 1_m 1_m^T / p_m for equally-sized groups of 5.
.group_mean_k <- function(p, group_size = 5) {
  K <- matrix(0, p, p)
  for (g in seq_len(p / group_size)) {
    idx <- ((g - 1) * group_size + 1):(g * group_size)
    K[idx, idx] <- 1 / group_size
  }
  K
}


test_that("trex_tikhonov_control validates its arguments", {
  expect_error(trex_tikhonov_control(), "tikhonov_K is required")
  expect_error(
    trex_tikhonov_control(tikhonov_K = diag(4), lambda2_method = "CV_1SE_CCD"),
    "arg"  # match.arg rejects the EN-shaped name: only IEN-geometry tuners
  )
  ctrl <- trex_tikhonov_control(tikhonov_K = diag(4))
  expect_identical(ctrl$lambda2_method, "PILOT_KNEE")
  expect_error(
    trex_tikhonov_control(tikhonov_K = diag(4), lambda2_method = "PILOT_MAX"),
    "arg"  # the argmax variant is gone; the knee with tolerance 0 reads it
  )
  expect_identical(ctrl$penalty_structure, "SPARSE_GENERAL")
  expect_true(ctrl$fold_dummy_coupling)
  expect_true(ctrl$cluster_dummies)
  expect_identical(ctrl$cv_n_lambda1, 40L)
  expect_identical(ctrl$cv_n_lambda2, 100L)
  expect_identical(ctrl$cv_dfmax_ratio, 0.8)
  wide <- trex_tikhonov_control(tikhonov_K = diag(4), cv_dfmax_ratio = 1)
  expect_identical(wide$cv_dfmax_ratio, 1)
  expect_error(
    trex_tikhonov_control(tikhonov_K = diag(4), cv_dfmax_ratio = NaN),
    "cv_dfmax_ratio"
  )

  # GROUP_CENTERING builds K itself: no tikhonov_K needed.
  gc <- trex_tikhonov_control(penalty_structure = "GROUP_CENTERING",
                              lambda2_method = "PILOT_KNEE")
  expect_null(gc$tikhonov_K)
  expect_identical(gc$lambda2_method, "PILOT_KNEE")
  expect_identical(gc$pilot_statistic, "MAJORITY")

  # Pilot gates.
  expect_error(trex_tikhonov_control(tikhonov_K = diag(4), pilot_K = 1),
               "pilot_K must be >= 2")
  expect_error(trex_tikhonov_control(tikhonov_K = diag(4), pilot_T = 0),
               "pilot_T must be >= 1")
  expect_error(trex_tikhonov_control(tikhonov_K = diag(4), pilot_grid = c(1, -1)),
               "pilot_grid")
  expect_error(trex_tikhonov_control(tikhonov_K = diag(4),
                                     pilot_group_fraction = 1.5),
               "pilot_group_fraction")
})


test_that("construction gates translate as R errors", {
  d <- .make_tik_data()

  # Mis-dimensioned K.
  expect_error(
    TRexTikhonovSelector$new(
      d$X, d$y, verbose = FALSE,
      tik_control = trex_tikhonov_control(tikhonov_K = diag(d$p - 1))),
    "[Kk]"
  )

  # PERMUTATION-family strategies cannot carry cluster-MVN dummies.
  expect_error(
    TRexTikhonovSelector$new(
      d$X, d$y, verbose = FALSE,
      tik_control = trex_tikhonov_control(tikhonov_K = diag(d$p)),
      control = trex_control(lloop_strategy = "PERMUTATION")),
    "PERMUTATION"
  )

  # Malformed prior_groups (length mismatch).
  expect_error(
    TRexTikhonovSelector$new(
      d$X, d$y, verbose = FALSE,
      tik_control = trex_tikhonov_control(tikhonov_K = diag(d$p),
                                          prior_groups = rep(1, d$p - 1))),
    "prior_groups"
  )

  # A non-Tikhonov-family solver choice warns (and falls back to TCDTIKNET).
  expect_warning(
    TRexTikhonovSelector$new(
      d$X, d$y, verbose = FALSE,
      tik_control = trex_tikhonov_control(tikhonov_K = diag(d$p),
                                          lambda_2 = 0.5,
                                          cluster_dummies = FALSE),
      control = trex_control(solver = "TOMP")),
    "not a Tikhonov-family solver"
  )

  # fold_dummy_coupling = FALSE exists only on the TCDTIKNET path. (The
  # warning case above constructed a live selector on d$X; a fresh copy
  # keeps the shared-buffer guard out of this check.)
  expect_error(
    TRexTikhonovSelector$new(
      d$X + 0, d$y, verbose = FALSE,
      tik_control = trex_tikhonov_control(tikhonov_K = diag(d$p),
                                          lambda_2 = 0.5,
                                          fold_dummy_coupling = FALSE),
      control = trex_control(solver = "TIKNET")),
    "TCDTIKNET"
  )
})


test_that("TIKNET and TCDTIKNET agree on the zero-K path (all-singleton GROUP_CENTERING)", {
  d <- .make_tik_data()

  # A HAC cut above the within-group correlation yields singletons: K_GC
  # vanishes and the path is the plain lasso on per-variable cluster
  # dummies. Both solvers must run and select identically.
  run <- function(solver) {
    sel <- TRexTikhonovSelector$new(
      d$X + 0, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
      tik_control = trex_tikhonov_control(
        penalty_structure = "GROUP_CENTERING",
        prior_groups = seq_len(d$p), lambda_2 = 10),
      control = trex_control(solver = solver, K = 5,
                             max_dummy_multiplier = 2))
    sel$select()
    sel$selected_var
  }

  sel_cd   <- run("TCDTIKNET")
  sel_lars <- run("TIKNET")
  expect_gt(sum(sel_cd), 0)
  expect_identical(sel_cd, sel_lars)
})


test_that("the pilot rule picks lambda_2 from its grid and records the profile", {
  d <- .make_tik_data()

  sel <- TRexTikhonovSelector$new(
    d$X, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
    tik_control = trex_tikhonov_control(
      tikhonov_K = .group_mean_k(d$p), prior_groups = d$groups,
      lambda_2 = -1, lambda2_method = "PILOT_KNEE",
      pilot_grid = c(10, 0, 1), pilot_K = 5, pilot_T = 3),
    control = trex_control(K = 5, max_dummy_multiplier = 2))
  sel$select()

  pilot <- sel$lambda2_pilot
  expect_true(pilot$ran)
  expect_identical(pilot$grid, c(0, 1, 10))        # sorted, de-duplicated
  expect_length(pilot$counts, 3)
  expect_equal(dim(pilot$curve_majority), c(3, 3))  # G x T_pilot
  expect_equal(dim(pilot$curve_soft), c(3, 3))
  expect_identical(pilot$K_pilot, 5)
  expect_identical(pilot$T_pilot, 3)
  expect_identical(pilot$L_0, d$p)
  expect_true(pilot$index %in% 1:3)
  expect_identical(pilot$grid[pilot$index], pilot$lambda2)
  expect_identical(sel$lambda2_used, pilot$lambda2)
  expect_identical(pilot$counts[pilot$index], pilot$max_selected)
  # The pilot is timed as the lambda_2 resolution step.
  expect_true(sel$lambda2_cv_diagnostics$ran)
})


test_that("K = I with both escape hatches collapses onto the TCDENET run", {
  d <- .make_tik_data()

  run_tik <- function() {
    sel <- TRexTikhonovSelector$new(
      d$X + 0, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
      tik_control = trex_tikhonov_control(
        tikhonov_K = diag(d$p), lambda_2 = 0.5,
        cluster_dummies = FALSE,       # i.i.d. dummies: base parity
        fold_dummy_coupling = FALSE),  # independent dummy ridge: TCDENET
      control = trex_control(K = 5, max_dummy_multiplier = 2))
    sel$select()
    sel$selected_var
  }
  run_tcenet <- function() {
    sel <- TRexSelector$new(
      d$X + 0, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
      control = trex_control(solver = "TCDENET", lambda2 = 0.5, K = 5,
                             max_dummy_multiplier = 2))
    sel$select()
    sel$selected_var
  }

  expect_identical(run_tik(), run_tcenet())
})


test_that("group-mean K with prior groups matches GVS-IEN TCDIENET", {
  d <- .make_tik_data()

  run_tik <- function() {
    sel <- TRexTikhonovSelector$new(
      d$X + 0, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
      tik_control = trex_tikhonov_control(
        tikhonov_K = .group_mean_k(d$p), lambda_2 = 1.0,
        prior_groups = d$groups),
      control = trex_control(K = 5, max_dummy_multiplier = 2))
    sel$select()
    sel$selected_var
  }
  run_gvs <- function() {
    sel <- TRexGVSSelector$new(
      d$X + 0, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
      gvs_control = trex_gvs_control(gvs_type = "IEN", lambda_2 = 1.0,
                                     groups = d$groups),
      control = trex_control(solver = "TCDIENET", K = 5,
                             max_dummy_multiplier = 2))
    sel$select()
    sel$selected_var
  }

  expect_identical(run_tik(), run_gvs())
})


test_that("prior-groups end-to-end recovers the planted signal", {
  d <- .make_tik_data()

  sel <- TRexTikhonovSelector$new(
    d$X, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
    tik_control = trex_tikhonov_control(
      tikhonov_K = .group_mean_k(d$p), lambda_2 = 1.0,
      prior_groups = d$groups),
    control = trex_control(K = 5, max_dummy_multiplier = 2))
  sel$select()

  idx <- sel$selected_indices
  expect_true(length(intersect(idx, 1:3)) >= 2)
  expect_identical(sel$lambda2_used, 1.0)
  # A caller-supplied lambda_2: neither a tuner nor a pilot ran.
  expect_false(sel$lambda2_cv_diagnostics$ran)
  expect_false(sel$lambda2_pilot$ran)
})


test_that("lambda_2 auto-resolves via the IEN-geometry tuner", {
  d <- .make_tik_data()

  sel <- TRexTikhonovSelector$new(
    d$X, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
    tik_control = trex_tikhonov_control(
      tikhonov_K = .group_mean_k(d$p), lambda_2 = -1,
      prior_groups = d$groups, lambda2_method = "CV_1SE_IEN_CCD",
      cv_n_lambda1 = 20, cv_n_lambda2 = 10, cv_dfmax_ratio = 0.9),
    control = trex_control(K = 5, max_dummy_multiplier = 2))
  sel$select()

  expect_gt(sel$lambda2_used, 0)
  diag <- sel$lambda2_cv_diagnostics
  expect_true(diag$ran)
  expect_gte(diag$seconds, 0)
  expect_gte(diag$n_profile_at_edge, 0)
  expect_false(sel$lambda2_pilot$ran)
  # The support-cap ratio reached the tuner and is recorded as passed.
  expect_identical(diag$dfmax_ratio, 0.9)
  expect_true(diag$profile_at_edge || !diag$profile_at_support_cap)

  # Boundary-pick guard fields.
  expect_identical(diag$grid_size, 10L)
  expect_true(diag$grid_index >= 0L && diag$grid_index < 10L)
  expect_identical(diag$boundary_pick, diag$at_grid_edge || diag$profile_at_edge)
})


test_that("trex_gamma_to_k builds the chain Laplacian from a first-difference Gamma", {
  skip_if_not_installed("Matrix")
  p <- 5
  gamma <- diff(diag(p))
  K <- trex_gamma_to_k(gamma)
  Kd <- as.matrix(K)
  expect_equal(dim(Kd), c(p, p))
  expect_equal(Kd, t(Kd))
  expect_equal(diag(Kd), c(1, 2, 2, 2, 1))
  expect_equal(Kd[1, 2], -1)
  # A sparse dgCMatrix Gamma is accepted too.
  K2 <- trex_gamma_to_k(Matrix::Matrix(gamma, sparse = TRUE))
  expect_equal(as.matrix(K2), Kd)
})


test_that("the full-profile knee runs one selection per grid value and keeps the contract", {
  d <- .make_tik_data()
  X_before <- d$X + 0  # a private copy: the probes hand X back unchanged
  tik <- trex_tikhonov_control(tikhonov_K = .group_mean_k(d$p),
                               prior_groups = d$groups)
  ctrl <- trex_control(K = 5, max_dummy_multiplier = 2)

  prof <- trex_tikhonov_lambda2_power_rule(d$X, d$y, tFDR = 0.2, tik_control = tik,
                                           control = ctrl, seed = 42,
                                           grid = c(10, 0, 1, 10))
  expect_identical(prof$grid, c(0, 1, 10))          # sorted, de-duplicated
  expect_length(prof$probes, 3)
  expect_identical(prof$counts, vapply(prof$probes, function(p) length(p$selected), 1))
  expect_true(prof$index %in% 1:3)
  expect_identical(prof$lambda2, prof$grid[prof$index])
  expect_identical(prof$selected, prof$probes[[prof$index]]$selected)
  expect_identical(prof$max_selected, max(prof$counts))
  expect_true(prof$seconds > 0)
  expect_equal(d$X, X_before)

  # The knee: the smallest grid value within one selection of the largest.
  knee <- which(prof$counts >= max(prof$counts) - 1)[1]
  expect_identical(prof$index, knee)

  # Tolerance 0 reads the largest selection, ties toward the smallest lambda_2.
  argmax <- trex_tikhonov_lambda2_power_rule(d$X, d$y, tFDR = 0.2, tik_control = tik,
                                             control = ctrl, seed = 42,
                                             grid = c(0, 1, 10), knee_tolerance = 0)
  expect_identical(argmax$index, which.max(prof$counts))

  # The contract: a fixed-lambda_2 run at the pick with the same seed selects
  # the same variables.
  tik_fixed <- tik
  tik_fixed$lambda_2 <- prof$lambda2
  sel <- TRexTikhonovSelector$new(d$X, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
                                  tik_control = tik_fixed, control = ctrl)
  sel$select()
  expect_identical(sort(sel$selected_indices), prof$selected)

  # Gates.
  expect_error(trex_tikhonov_lambda2_power_rule(d$X, d$y, tik_control = tik, seed = -1),
               "seed")
  expect_error(trex_tikhonov_lambda2_power_rule(d$X, d$y, tik_control = tik, grid = c(-1, 1)),
               "grid")
  expect_error(trex_tikhonov_lambda2_power_rule(d$X, d$y, tik_control = tik,
                                                knee_tolerance = -1),
               "knee_tolerance")
  expect_error(trex_tikhonov_lambda2_power_rule(d$X, d$y), "tik_control")
})
