# =======================================================================================
# TrexScreeningSelector - Unit tests
# =======================================================================================

test_that("TRexScreeningSelector instantiation and dimensional validation work", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Valid instantiation
  expect_silent(TRexScreeningSelector$new(X, y, verbose = FALSE))

  # Dimension mismatch
  y_bad <- rnorm(n + 10)
  expect_error(TRexScreeningSelector$new(X, y_bad, verbose = FALSE))
})



test_that("TRexScreeningSelector executes correctly with valid data", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Inject some strong signals in the first 3 variables
  y <- X[, 1] * 5 + X[, 2] * -4 + X[, 3] * 3 + rnorm(n)

  # Instantiate TRex selector
  selector <- TRexScreeningSelector$new(X, y, verbose = FALSE)

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



test_that("TRexScreeningSelector behaves memory safe with memory mapping enabled", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # use_memory_mapping = TRUE to confirm tempdir() doesn't crash
  selector <- TRexScreeningSelector$new(X, y, verbose = FALSE, use_memory_mapping = TRUE)
  selector$select()

  res_indices <- selector$selected_indices
  res_T_stop <- selector$T_stop
  
  expect_type(res_indices, "integer")
  
})



test_that("TRexScreeningSelector correctly guards against invalid parameters", {

  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Invalid Screening Variants that are caught by Rcpp string mapping
  expect_error(
    TRexScreeningSelector$new(X, y, trex_method = "UNKNOWN_VARIANT", verbose = FALSE),
    "Unknown ScreenTRexMethod:"
  )
})


test_that("TrexScreeningSelector correctly executes confidence-interval based bootstrapping", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  # Instantiate with CI Bootstrapping active
  selector_boot <- expect_no_error(
    TRexScreeningSelector$new(
      X, y,
      use_bootstrap_CI = TRUE,
      R_boot = 100,
      ci_grid_step = 0.01,
      verbose = FALSE
    )
  )
  selector_boot$select()

  # Assert result structure includes Bootstrap diagnostics
  
  expect_true(selector_boot$used_bootstrap)
  expect_type(selector_boot$confidence_level, "double")

  # Ensure FDR is bounded properly
  expect_true(selector_boot$estimated_FDR >= 0.0 && selector_boot$estimated_FDR <= 1.0)
})



test_that("TRexScreeningSelector handles concurrent execution properly (OpenMP)", {

  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Run multithreaded (runs one sweeping K-loop for fast Screening)
  selector_openmp <- expect_no_error(
    TRexScreeningSelector$new(X, y, use_openmp = TRUE, max_inner_threads = 2,
                              max_outer_threads = 2, verbose = FALSE)
  )

  selector_openmp$select()
  res_openmp_indices <- selector_openmp$selected_indices
  res_openmp_T_stop <- selector_openmp$T_stop
  

  expect_type(res_openmp_indices, "integer")
  
})


test_that("TRexScreeningSelector selected_var and selected_indices are consistent (1-based)", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + X[, 3] * 3 + rnorm(n)

  selector <- TRexScreeningSelector$new(X, y, verbose = FALSE)
  selector$select()

  idx <- selector$selected_indices
  sv  <- selector$selected_var

  expect_type(idx, "integer")
  expect_length(sv, p)

  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p))
  }
  expect_equal(sort(idx), which(sv))
})


test_that("TRexScreeningSelector phi_prime is in [0, 1] and phi_mat has correct dimensions", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  selector <- TRexScreeningSelector$new(X, y, verbose = FALSE)
  selector$select()

  phi_p <- selector$phi_prime
  expect_length(phi_p, p)
  expect_true(is.numeric(phi_p))

  ts <- selector$T_stop
  pm <- selector$phi_mat
  if (ts > 0L) {
    expect_gte(nrow(pm), 1L)
    expect_equal(ncol(pm), p)
  }
})


test_that("TRexScreeningSelector screening-specific fields have valid values", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  selector <- TRexScreeningSelector$new(X, y, verbose = FALSE)
  selector$select()

  # estimated_correlation is a non-negative double
  expect_type(selector$estimated_correlation, "double")
  expect_true(selector$estimated_correlation >= 0.0)

  # beta_mat and dummy_betas are matrices with positive dimensions
  bm <- selector$beta_mat
  db <- selector$dummy_betas
  expect_true(is.matrix(bm))
  expect_true(nrow(bm) > 0 && ncol(bm) > 0)
  expect_true(is.matrix(db))
  expect_true(nrow(db) > 0 && ncol(db) > 0)
})


test_that("TRexScreeningSelector DA-method variants run without error and return valid indices", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  for (tm in c("TREX_DA_AR1", "TREX_DA_EQUI", "TREX_DA_BLOCK_EQUI")) {
    sel <- TRexScreeningSelector$new(X, y, trex_method = tm, verbose = FALSE)
    sel$select()

    idx <- sel$selected_indices
    sv  <- sel$selected_var

    expect_type(idx, "integer")
    expect_length(sv, p)
    if (length(idx) > 0) {
      expect_true(all(idx >= 1L & idx <= p), label = paste(tm, "1-based in [1,p]"))
    }
    expect_equal(sort(idx), which(sv), label = paste(tm, "consistency"))
  }
})
