# =======================================================================================
# TRexGVSSelector - Unit tests
# =======================================================================================

test_that("TRexGVSSelector instantiation and dimensional validation work", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # 1. Valid instantiation
  expect_silent(TRexGVSSelector$new(X, y, verbose = FALSE, gvs_control = trex_gvs_control(gvs_type = "EN")))

  # 2. Dimension mismatch (y length != X rows)
  y_bad <- rnorm(n + 10)
  expect_error(
    TRexGVSSelector$new(X, y_bad, verbose = FALSE, gvs_control = trex_gvs_control(gvs_type = "EN")),
    "length equal to the number of rows"
  )

  # 3. Degenerate feature matrix dimensions (n=0 or p=0)
  X_empty <- matrix(numeric(0), nrow = 0, ncol = p)
  y_empty <- numeric(0)
  expect_error(
    TRexGVSSelector$new(X_empty, y_empty, verbose = FALSE, gvs_control = trex_gvs_control(gvs_type = "EN")),
    "non-zero dimensions"
  )

  # 4. Non-finite values in the response vector (NaN or Inf)
  y_na <- y
  y_na[1] <- NA
  expect_error(
    TRexGVSSelector$new(X, y_na, verbose = FALSE, gvs_control = trex_gvs_control(gvs_type = "EN")),
    "NaN or Inf values"
  )

})



test_that("TRexGVSSelector executes correctly with valid data", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Inject some strong signals in the first 3 variables
  y <- X[, 1] * 5 + X[, 2] * -4 + X[, 3] * 3 + rnorm(n)

  # Instantiate TRex selector
  selector <- TRexGVSSelector$new(X, y, tFDR = 0.1, verbose = FALSE,
                                  gvs_control = trex_gvs_control(gvs_type = "IEN"),
                                  control = trex_control(K = 5))

  # Execute variable selection
  selector$select()

  # Assert result structure and logic
  res_indices <- selector$selected_indices
  res_T_stop <- selector$T_stop

  expect_type(res_indices, "integer")

  expect_type(res_T_stop, "integer")

  # Bounds check on output selected indices (1 <= index <= p)
  if (length(res_indices) > 0) {
    expect_true(all(res_indices >= 1 & res_indices <= p))
  }
})



test_that("TRexGVSSelector handles concurrent execution properly (OpenMP)", {

  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Ensure the object can be created and bound appropriately with concurrency settings
  selector_openmp <- expect_no_error(
    TRexGVSSelector$new(X, y,
                        gvs_control = trex_gvs_control(gvs_type = "IEN"),
                        control = trex_control(K = 5, use_openmp = TRUE, max_inner_threads = 2,
                                              max_outer_threads = 2),
                        verbose = FALSE)
  )

  # Run the algorithm targeting out threaded branches; fall back if openMP is disabled
  selector_openmp$select()

  res_openmp_indices <- selector_openmp$selected_indices
  res_openmp_T_stop <- selector_openmp$T_stop


  expect_type(res_openmp_indices, "integer")

  expect_type(res_openmp_T_stop, "integer")
})



test_that("TRexGVSSelector correctly guards against invalid parameters", {
  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Unsupported L-Loop Strategy
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN"),
                        control = trex_control(lloop_strategy = "PERMUTATION"), verbose = FALSE),
    "are not supported"
  )

  # Invalid Hierarchical Control Parameters
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN", corr_max = -0.1), verbose = FALSE),
    "corr_max"
  )
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN", corr_max = 1.1), verbose = FALSE),
    "corr_max"
  )

  # Invalid linkage triggers Rcpp stop()
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN", hc_linkage = "UNKNOWN_LINKAGE"), verbose = FALSE),
    "Unsupported linkage method for GVS"
  )

  # Negative Ridge lambda_2 penalty
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN", lambda_2 = -1.0), verbose = FALSE),
    "lambda_2 must be >= 0"
  )
})



test_that("TRexGVSSelector handles manual grouping vectors correctly", {
  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Wrong length for groups
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN", groups = c(1, 2)), verbose = FALSE),
    "length must equal p"
  )

  # Missing clusters (non-contiguous IDs)
  # Trying to assign clusters 1 and 3 (R 1-based indexing, skips 2)
  bad_groups <- rep(c(1, 3), length.out = p)
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN", groups = bad_groups), verbose = FALSE),
    "non-contiguous"
  )

  # Negative or zero group IDs from R
  # (R uses 1-based indexing; sending 0 becomes -1 in C++)
  zero_groups <- rep(c(0, 1), length.out = p)
  expect_error(
    TRexGVSSelector$new(X, y, gvs_control = trex_gvs_control(gvs_type = "EN", groups = zero_groups), verbose = FALSE),
    "non-negative"
  )
})



test_that("TRexGVSSelector behaves memory safe with memory mapping enabled", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Use use_memory_mapping = TRUE to confirm tempdir() doesn't crash
  selector <- TRexGVSSelector$new(X, y, verbose = FALSE,
                                  gvs_control = trex_gvs_control(gvs_type = "IEN"),
                                  control = trex_control(K = 3, use_memory_mapping = TRUE))
  selector$select()

  res_indices <- selector$selected_indices
  res_T_stop <- selector$T_stop

  expect_type(res_indices, "integer")

})


test_that("TRexGVSSelector selected_var and selected_indices are consistent (1-based)", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + X[, 3] * 3 + rnorm(n)

  for (gvs in c("EN", "IEN")) {
    sel <- TRexGVSSelector$new(X, y, tFDR = 0.1, verbose = FALSE,
                               gvs_control = trex_gvs_control(gvs_type = gvs),
                               control = trex_control(K = 3))
    sel$select()

    idx <- sel$selected_indices
    sv  <- sel$selected_var

    expect_type(idx, "integer")
    expect_length(sv, p)

    if (length(idx) > 0) {
      expect_true(all(idx >= 1L & idx <= p), label = paste(gvs, "1-based in [1,p]"))
    }
    expect_equal(sort(idx), which(sv), label = paste(gvs, "consistency"))
  }
})


test_that("TRexGVSSelector phi_prime is in [0, 1] and phi_mat has correct dimensions", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  selector <- TRexGVSSelector$new(X, y, tFDR = 0.1, verbose = FALSE,
                                  gvs_control = trex_gvs_control(gvs_type = "IEN"),
                                  control = trex_control(K = 3))
  selector$select()

  phi_p <- selector$phi_prime
  expect_length(phi_p, p)
  expect_true(all(phi_p >= 0.0 & phi_p <= 1.0))

  ts <- selector$T_stop
  pm <- selector$phi_mat
  if (ts > 0L) {
    expect_gte(nrow(pm), 1L)
    expect_equal(ncol(pm), p)
  }
})


test_that("TRexGVSSelector GVS-specific fields have valid values", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  for (gvs in c("EN", "IEN")) {
    sel <- TRexGVSSelector$new(X, y, verbose = FALSE,
                               gvs_control = trex_gvs_control(gvs_type = gvs),
                               control = trex_control(K = 3))
    sel$select()

    expect_equal(sel$gvs_type, gvs, label = paste(gvs, "gvs_type roundtrip"))

    mc <- sel$max_clusters
    expect_type(mc, "integer")
    expect_true(mc >= 1L & mc <= p, label = paste(gvs, "max_clusters in [1,p]"))

    expect_true(sel$lambda2_used >= 0.0, label = paste(gvs, "lambda2_used >= 0"))
  }
})


test_that("TRexScreeningSelector correctly executes confidence-based bootstrapping", {
  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)

  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  # Instantiate with CI Bootstrapping active
  selector_boot <- expect_no_error(
    TRexScreeningSelector$new(
      X, y,
      screen_control = trex_screen_control(use_bootstrap_CI = TRUE, R_boot = 100, ci_grid_step = 0.01),
      verbose = FALSE
    )
  )

  selector_boot$select()

  # Assert result structure includes Bootstrap diagnostics
  res_boot_used <- selector_boot$used_bootstrap
  res_boot_conf <- selector_boot$confidence_level
  res_boot_fdr <- selector_boot$estimated_FDR
  expect_true(res_boot_used)
  expect_type(res_boot_conf, "double")

  # Ensure FDR is bounded properly
  expect_true(res_boot_fdr >= 0.0 && res_boot_fdr <= 1.0)
})
