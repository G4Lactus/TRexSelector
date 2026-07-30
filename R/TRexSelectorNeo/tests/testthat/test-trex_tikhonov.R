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
  expect_identical(ctrl$lambda2_method, "CV_1SE_IEN_CCD")
  expect_true(ctrl$fold_dummy_coupling)
  expect_true(ctrl$cluster_dummies)
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

  # Non-TCIENET solver choice warns (and is overridden).
  expect_warning(
    TRexTikhonovSelector$new(
      d$X, d$y, verbose = FALSE,
      tik_control = trex_tikhonov_control(tikhonov_K = diag(d$p),
                                          lambda_2 = 0.5,
                                          cluster_dummies = FALSE),
      control = trex_control(solver = "TOMP")),
    "requires \"TCIENET\""
  )
})


test_that("K = I with both escape hatches collapses onto the TCENET run", {
  d <- .make_tik_data()

  run_tik <- function() {
    sel <- TRexTikhonovSelector$new(
      d$X + 0, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
      tik_control = trex_tikhonov_control(
        tikhonov_K = diag(d$p), lambda_2 = 0.5,
        cluster_dummies = FALSE,       # i.i.d. dummies: base parity
        fold_dummy_coupling = FALSE),  # independent dummy ridge: TCENET
      control = trex_control(K = 5, max_dummy_multiplier = 2))
    sel$select()
    sel$selected_var
  }
  run_tcenet <- function() {
    sel <- TRexSelector$new(
      d$X + 0, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
      control = trex_control(solver = "TCENET", lambda2 = 0.5, K = 5,
                             max_dummy_multiplier = 2))
    sel$select()
    sel$selected_var
  }

  expect_identical(run_tik(), run_tcenet())
})


test_that("group-mean K with prior groups matches GVS-IEN TCIENET", {
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
      control = trex_control(solver = "TCIENET", K = 5,
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
})


test_that("lambda_2 auto-resolves via the IEN-geometry tuner", {
  d <- .make_tik_data()

  sel <- TRexTikhonovSelector$new(
    d$X, d$y, tFDR = 0.2, seed = 42, verbose = FALSE,
    tik_control = trex_tikhonov_control(
      tikhonov_K = .group_mean_k(d$p), lambda_2 = -1,
      prior_groups = d$groups),
    control = trex_control(K = 5, max_dummy_multiplier = 2))
  sel$select()

  expect_gt(sel$lambda2_used, 0)
})


test_that("gamma_to_k builds the chain Laplacian from a first-difference Gamma", {
  skip_if_not_installed("Matrix")
  p <- 5
  gamma <- diff(diag(p))
  K <- gamma_to_k(gamma)
  Kd <- as.matrix(K)
  expect_equal(dim(Kd), c(p, p))
  expect_equal(Kd, t(Kd))
  expect_equal(diag(Kd), c(1, 2, 2, 2, 1))
  expect_equal(Kd[1, 2], -1)
  # A sparse dgCMatrix Gamma is accepted too.
  K2 <- gamma_to_k(Matrix::Matrix(gamma, sparse = TRUE))
  expect_equal(as.matrix(K2), Kd)
})
