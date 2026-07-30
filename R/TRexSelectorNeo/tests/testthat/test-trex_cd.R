# =======================================================================================
# CCD solver family (TCCD / TCENET / TCIENET) through the R selector surface
# =======================================================================================

# Shared planted design: strong signal on variables 1:5.
.make_cd_data <- function(n = 100, p = 20, data_seed = 42) {
  set.seed(data_seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- c(rep(5, 5), rep(0, p - 5))
  y <- as.vector(X %*% beta + rnorm(n))
  list(X = X, y = y, support = 1:5, p = p)
}


test_that("trex_control validates the CD knobs", {

  # certify/probe must be set together (the C++ side applies the pair only
  # when both are > 0 — a half-set pair must fail loudly, not silently).
  expect_error(
    trex_control(cd_tol_certify = 1e-6),
    "cd_tol_certify and cd_tol_probe must be set together"
  )
  expect_error(
    trex_control(cd_tol_probe = 1e-6),
    "cd_tol_certify and cd_tol_probe must be set together"
  )
  expect_silent(trex_control(cd_tol_certify = 1e-6, cd_tol_probe = 1e-7))

  expect_error(trex_control(cd_gram_cap = -1), "cd_gram_cap must be >= 0")
  expect_error(trex_control(cd_max_sweeps = -1), "cd_max_sweeps must be >= 0")

  # The CCD family records exact minimizers per crossing: non-greedy, so the
  # stagnation-stop auto-detection must default to FALSE (mirrors the C++
  # isGreedySolver classification).
  for (s in c("TCCD", "TCENET", "TCIENET", "TENET_AUG")) {
    expect_false(trex_control(solver = s)$tloop_stagnation_stop)
  }
  expect_true(trex_control(solver = "TOMP")$tloop_stagnation_stop)
})


test_that("TCCD and TCENET run end-to-end and recover the planted support", {
  d <- .make_cd_data()

  for (s in c("TCCD", "TCENET")) {
    selector <- TRexSelector$new(d$X, d$y, tFDR = 0.2, seed = 42,
                                 verbose = FALSE,
                                 control = trex_control(solver = s, K = 5))
    selector$select()

    expect_length(selector$selected_var, d$p)
    idx <- selector$selected_indices
    expect_true(all(d$support %in% idx),
                info = paste0(s, ": planted support not recovered. Selected: ",
                              paste(idx, collapse = ", ")))
    expect_lte(length(setdiff(idx, d$support)), 3)
  }
})


test_that("CD selection is seed-reproducible", {
  d <- .make_cd_data()

  run_once <- function() {
    selector <- TRexSelector$new(d$X, d$y, tFDR = 0.2, seed = 7,
                                 verbose = FALSE,
                                 control = trex_control(solver = "TCENET",
                                                        K = 5))
    selector$select()
    selector$selected_var
  }
  expect_identical(run_once(), run_once())
})


test_that("CD knobs pass through the control parser", {
  d <- .make_cd_data()

  ctrl <- trex_control(solver = "TCENET", K = 3,
                       cd_lambda_rel_tol = 1e-4,
                       cd_tol_certify = 1e-6, cd_tol_probe = 1e-7,
                       cd_gram_cap = 100, cd_max_sweeps = 500)
  selector <- TRexSelector$new(d$X, d$y, tFDR = 0.2, seed = 42,
                               verbose = FALSE, control = ctrl)
  selector$select()
  expect_true(all(d$support %in% selector$selected_indices))
})


test_that("GVS-only solvers are rejected by the plain selector with guidance", {
  d <- .make_cd_data(n = 50, p = 10)

  run_with <- function(s) {
    selector <- TRexSelector$new(d$X, d$y, tFDR = 0.2, seed = 42,
                                 verbose = FALSE,
                                 control = trex_control(solver = s, K = 3))
    selector$select()
  }
  expect_error(run_with("TCIENET"), "needs a group assignment")
  expect_error(run_with("TIENET"), "GVS-only solver")
  expect_error(run_with("TIENET_AUG"), "GVS-only solver")
})


test_that("TENET_AUG runs in the plain selector and matches TENET", {
  d <- .make_cd_data()

  run_with <- function(s) {
    selector <- TRexSelector$new(d$X, d$y, tFDR = 0.2, seed = 42,
                                 verbose = FALSE,
                                 control = trex_control(solver = s, K = 5))
    selector$select()
    selector$selected_var
  }

  sel_tenet <- run_with("TENET")
  sel_aug   <- run_with("TENET_AUG")

  # TENET_AUG is the row-augmented formulation of the same elastic-net
  # problem (proved equivalent for lambda2 >= 0): identical data and seed
  # must select identically.
  expect_identical(sel_aug, sel_tenet)
  expect_true(all(d$support %in% which(sel_aug)))
})


test_that("GVS runs the CCD family: en_solver = TCENET and IEN solver choices", {
  set.seed(42)
  n <- 80
  p <- 20
  # Grouped design: 4 blocks of 5 correlated columns, signal on block 1.
  z <- matrix(rnorm(n * 4), n, 4)
  X <- matrix(0, n, p)
  for (g in 1:4) {
    for (j in 1:5) {
      X[, (g - 1) * 5 + j] <- 0.85 * z[, g] + sqrt(1 - 0.85^2) * rnorm(n)
    }
  }
  y <- as.vector(rowSums(X[, 1:3]) * 3 + rnorm(n))
  # A live selector claims and normalizes its X buffer in place
  # (shared-buffer claim guard), so every constructor gets its own copy.
  fresh <- function() X + 0

  # EN track, CCD elastic net.
  sel_en <- TRexGVSSelector$new(
    fresh(), y, tFDR = 0.2, seed = 42, verbose = FALSE,
    gvs_control = trex_gvs_control(gvs_type = "EN", en_solver = "TCENET",
                                   lambda_2 = 1.0),
    control = trex_control(K = 5)
  )
  sel_en$select()
  expect_length(sel_en$selected_var, p)

  # IEN track, CCD informed elastic net via control$solver.
  sel_ien <- TRexGVSSelector$new(
    fresh(), y, tFDR = 0.2, seed = 42, verbose = FALSE,
    gvs_control = trex_gvs_control(gvs_type = "IEN", lambda_2 = 1.0),
    control = trex_control(solver = "TCIENET", K = 5)
  )
  sel_ien$select()
  expect_length(sel_ien$selected_var, p)

  # EN track warns when control$solver conflicts with the en_solver axis.
  expect_warning(
    TRexGVSSelector$new(
      fresh(), y, tFDR = 0.2, seed = 42, verbose = FALSE,
      gvs_control = trex_gvs_control(gvs_type = "EN", lambda_2 = 1.0),
      control = trex_control(solver = "TCENET", K = 5)
    ),
    "derives solver"
  )

  # IEN track warns on a non-IEN-family control$solver and falls back.
  expect_warning(
    TRexGVSSelector$new(
      fresh(), y, tFDR = 0.2, seed = 42, verbose = FALSE,
      gvs_control = trex_gvs_control(gvs_type = "IEN", lambda_2 = 1.0),
      control = trex_control(solver = "TOMP", K = 5)
    ),
    "not an IEN-family solver"
  )
})
