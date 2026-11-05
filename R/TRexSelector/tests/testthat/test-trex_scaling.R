# =======================================================================================
# scaling_mode = "NONE" (user-managed design) and input rejection
# =======================================================================================

planted_design <- function(seed = 5, n = 300, p = 60) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  y <- 5 * X[, 2] - 3 * X[, 3] + 2 * X[, 4] + rnorm(n)
  list(X = X, y = y)
}

unit_l2_columns <- function(X) {
  sweep(X, 2, sqrt(colSums(X^2)), "/")
}


test_that("trex_control exposes scaling_mode with the L2 default", {
  expect_equal(trex_control()$scaling_mode, "L2")
  expect_equal(trex_control(scaling_mode = "ZSCORE")$scaling_mode, "ZSCORE")
  expect_equal(trex_control(scaling_mode = "NONE")$scaling_mode, "NONE")
  expect_error(trex_control(scaling_mode = "MAX"), "scaling_mode")
})


test_that("trex_normalize_design reproduces the selector preprocessing", {
  d <- planted_design(n = 40, p = 6)
  X0 <- d$X + 0
  y0 <- d$y + 0

  l2 <- trex_normalize_design(d$X, d$y)
  expect_identical(d$X, X0)   # inputs untouched
  expect_identical(d$y, y0)
  expect_equal(colMeans(l2$X), rep(0, 6), tolerance = 1e-12)
  expect_equal(sqrt(colSums(l2$X^2)), rep(1, 6), tolerance = 1e-12)
  expect_equal(l2$X_means, colMeans(d$X))
  expect_equal(l2$X_scales, sqrt(colSums(sweep(d$X, 2, colMeans(d$X))^2)))
  expect_equal(l2$y_mean, mean(d$y))
  expect_equal(l2$y, d$y - mean(d$y))

  z <- trex_normalize_design(d$X, d$y, scaling_mode = "ZSCORE")
  expect_equal(sqrt(colSums(z$X^2)), rep(sqrt(39), 6), tolerance = 1e-12)
  expect_equal(z$X_scales, apply(d$X, 2, sd))

  none <- trex_normalize_design(d$X, d$y, scaling_mode = "NONE")
  expect_identical(none$X, X0)
  expect_identical(none$y, y0)
  expect_true(all(none$X_means == 0))
  expect_true(all(none$X_scales == 1))
  expect_identical(none$y_mean, 0)

  expect_error(trex_normalize_design(d$X, d$y, scaling_mode = "MAX"))
})


test_that("NONE on a pre-normalized design equals L2 on the raw design", {
  d <- planted_design()

  sel_l2 <- TRexSelector$new(d$X + 0, d$y, tFDR = 0.2, seed = 11, verbose = FALSE,
                             control = trex_control(K = 20))
  sel_l2$select()

  pre <- trex_normalize_design(d$X, d$y)
  X_pre <- pre$X + 0
  sel_none <- TRexSelector$new(pre$X, pre$y, tFDR = 0.2, seed = 11, verbose = FALSE,
                               control = trex_control(K = 20, scaling_mode = "NONE"))
  sel_none$select()

  expect_identical(pre$X, X_pre)   # NONE never writes into the caller's matrix
  expect_equal(sel_none$T_stop, sel_l2$T_stop)
  expect_identical(sel_none$selected_var, sel_l2$selected_var)
  expect_identical(sel_none$phi_prime, sel_l2$phi_prime)   # bitwise
  expect_true(any(sel_l2$selected_var))
  expect_true(all(sel_l2$selected_indices %in% 2:4))

  # Identity parameters under NONE.
  expect_true(all(sel_none$x_means == 0))
  expect_true(all(sel_none$x_l2_norms == 1))
  expect_identical(sel_none$y_mean, 0)
})


test_that("non-finite entries and dead columns are rejected in every mode", {
  d <- planted_design(n = 40, p = 8)
  X <- unit_l2_columns(d$X)

  for (mode in c("L2", "ZSCORE", "NONE")) {
    ctrl <- trex_control(K = 3, scaling_mode = mode)
    Xn <- X + 0
    Xn[3, 4] <- NA_real_
    expect_error(TRexSelector$new(Xn, d$y, verbose = FALSE, control = ctrl),
                 "X\\(2, 3\\) is not finite")
    expect_error(trex_normalize_design(Xn, d$y, scaling_mode = mode),
                 "X\\(2, 3\\) is not finite")
    yi <- d$y + 0
    yi[2] <- Inf
    # The selector's parameter validation catches y first (its own message);
    # the normalizer names the entry when called directly.
    expect_error(TRexSelector$new(X + 0, yi, verbose = FALSE, control = ctrl),
                 "NaN or Inf")
    expect_error(trex_normalize_design(X, yi, scaling_mode = mode),
                 "y\\(1\\) is not finite")
  }

  # Constant column: rejected under L2 / ZSCORE with the column named.
  Xc <- d$X + 0
  Xc[, 5] <- 3.5
  expect_error(TRexSelector$new(Xc, d$y, verbose = FALSE, control = trex_control(K = 3)),
               "column 4 .*constant")
  expect_error(trex_normalize_design(Xc, d$y, scaling_mode = "ZSCORE"),
               "column 4 .*constant")

  # Intercept-free dictionary: a unit-L2 constant (DC) atom is legitimate
  # under NONE; an all-zero column is not.
  Xd <- X + 0
  Xd[, 5] <- 1 / sqrt(40)
  sel <- TRexSelector$new(Xd, d$y, verbose = FALSE,
                          control = trex_control(K = 3, scaling_mode = "NONE"))
  expect_silent(sel$select())
  Xz <- Xd + 0
  Xz[, 5] <- 0
  expect_error(TRexSelector$new(Xz, d$y, verbose = FALSE,
                                control = trex_control(K = 3, scaling_mode = "NONE")),
               "column 4 .*NONE")
})
