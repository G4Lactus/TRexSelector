# =======================================================================================
# TRexBiobankScreeningSelector - Unit tests
# =======================================================================================

test_that("TRexBiobankScreeningSelector instantiation and data types work", {

  set.seed(42)
  n <- 50
  p <- 15
  X <- matrix(rnorm(n * p), n, p)

  # Valid 1D
  y_vec <- rnorm(n)
  expect_silent(TRexBiobankScreeningSelector$new(X, y_vec, verbose = FALSE))

  # Valid 2D
  Y_mat <- matrix(rnorm(n * 3), n, 3)
  expect_silent(TRexBiobankScreeningSelector$new(X, Y_mat, verbose = FALSE))

  # Invalid type for Y
  expect_error(
    TRexBiobankScreeningSelector$new(X, "invalid", verbose = FALSE),
    "Response evaluation Y must be either a numeric vector or a numeric matrix"
  )

  # Dimension mismatch 1D
  y_bad <- rnorm(n + 10)
  expect_error(TRexBiobankScreeningSelector$new(X, y_bad, verbose = FALSE))


  # Dimension mismatch 2D
  Y_bad <- matrix(rnorm((n + 1) * 3), n + 1, 3)
  expect_error(TRexBiobankScreeningSelector$new(X, Y_bad, verbose = FALSE))
})



test_that("TRexBiobankScreeningSelector screens single phenotype correctly (1D)", {

  set.seed(42)
  n <- 50
  p <- 15
  X <- matrix(rnorm(n * p), n, p)

  # Inject specific signal
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  selector <- TRexBiobankScreeningSelector$new(X, y, verbose = FALSE)
  res <- selector$select()

  # Check object structure
  expect_type(res, "list")
  expect_true("phenotype_index" %in% names(res))
  expect_true("selected_indices" %in% names(res))
  expect_true("estimated_FDR" %in% names(res))
  expect_true("method_used" %in% names(res))
  expect_true("used_fallback_trex" %in% names(res))

  # R indexing base check
  expect_equal(res$phenotype_index, 1)
})



test_that("TRexBiobankScreeningSelector screens multiple phenotypes correctly (2D)", {

  set.seed(42)
  n <- 50
  p <- 15
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(rnorm(n * 3), n, 3)

  # Inject signals for some phenotypes
  Y[, 1] <- X[, 1] * 5 + rnorm(n)
  Y[, 2] <- X[, 2] * 4 + X[, 3] * -3 + rnorm(n)

  selector <- TRexBiobankScreeningSelector$new(X, Y, verbose = FALSE)
  res <- selector$select()

  # Check overarching object structure
  expect_type(res, "list")
  expect_true("statistics" %in% names(res))
  expect_true("selected_indices" %in% names(res))

  # Validate DataFrame properties
  stats <- res$statistics
  expect_s3_class(stats, "data.frame")
  expect_equal(nrow(stats), 3)

  required_cols <- c("phenotype_index", "estimated_FDR", "method_used", "used_fallback_trex")
  expect_true(all(required_cols %in% colnames(stats)))

  # Ensure indices list matches phenotypes
  expect_equal(length(res$selected_indices), 3)
})



test_that("TRexBiobankScreeningSelector correctly triggers fallback T-Rex decision logic", {

  set.seed(42)
  n <- 50
  p <- 15
  X <- matrix(rnorm(n * p), n, p)

  # Strong noisy signal to trigger an FDR outside constraints
  y <- X[, 1] * 5 + rnorm(n)

  # Force fallback by setting an impossible FDR acceptance range for Screen-TRex
  selector <- TRexBiobankScreeningSelector$new(
    X, y,
    biobank_control = trex_biobank_control(lower_bound_FDR = 0.0001, upper_bound_FDR = 0.0002),
    verbose = FALSE
  )
  res <- selector$select()

  # With such extreme bounds, it should reject Screen-TRex evaluation and use the full T-Rex
  expect_true(res$used_fallback_trex)
})



test_that("TRexBiobankScreeningSelector protects against invalid framework strings", {

  set.seed(42)
  n <- 50
  p <- 15
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Invalid solver fallback
  expect_error(
    TRexBiobankScreeningSelector$new(X, y, control = trex_control(method = "UNKNOWN_SOLVER"), verbose = FALSE),
    "Unknown TRex selector method"
  )

  # Invalid core screening method parameter mapping
  expect_error(
    TRexBiobankScreeningSelector$new(X, y, screen_control = trex_screen_control(trex_method = "UNKNOWN_VARIANT"), verbose = FALSE),
    "Unknown ScreenTRexMethod"
  )
})


test_that("TRexBiobankScreeningSelector 1D result has 1-based indices in [1, p]", {
  set.seed(42)
  n <- 50
  p <- 15
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + rnorm(n)

  selector <- TRexBiobankScreeningSelector$new(X, y, verbose = FALSE)
  res <- selector$select()

  # phenotype_index is 1 (single phenotype, 1-based)
  expect_equal(res$phenotype_index, 1L)

  # selected_indices are 1-based integers in [1, p]
  idx <- res$selected_indices
  expect_type(idx, "integer")
  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p))
  }

  # sub-method index fields are present and also 1-based in [1, p]
  expect_true("selected_indices_screen_ordinary" %in% names(res))
  expect_true("selected_indices_screen_bootstrap" %in% names(res))

  for (field in c("selected_indices_screen_ordinary", "selected_indices_screen_bootstrap")) {
    sub_idx <- res[[field]]
    expect_type(sub_idx, "integer")
    if (length(sub_idx) > 0) {
      expect_true(all(sub_idx >= 1L & sub_idx <= p), label = paste(field, "1-based in [1,p]"))
    }
  }

  # estimated_FDR is a non-negative double
  expect_type(res$estimated_FDR, "double")
  expect_true(res$estimated_FDR >= 0.0)
})


test_that("TRexBiobankScreeningSelector 2D result has 1-based phenotype_index and per-phenotype indices", {
  set.seed(42)
  n <- 50
  p <- 15
  q <- 3
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(rnorm(n * q), n, q)
  Y[, 1] <- X[, 1] * 5 + rnorm(n)
  Y[, 2] <- X[, 2] * 4 + X[, 3] * -3 + rnorm(n)

  selector <- TRexBiobankScreeningSelector$new(X, Y, verbose = FALSE)
  res <- selector$select()

  stats <- res$statistics

  # phenotype_index column is 1-based: 1, 2, ..., q
  expect_equal(stats$phenotype_index, seq_len(q))

  # selected_indices list: each element is 1-based in [1, p]
  sel_list <- res$selected_indices
  expect_equal(length(sel_list), q)
  for (i in seq_len(q)) {
    idx_i <- sel_list[[i]]
    expect_type(idx_i, "integer")
    if (length(idx_i) > 0) {
      expect_true(all(idx_i >= 1L & idx_i <= p),
                  label = paste("phenotype", i, "1-based in [1,p]"))
    }
  }

  # sub-method index lists are present and each element is 1-based in [1, p]
  expect_true("selected_indices_screen_ordinary" %in% names(res))
  expect_true("selected_indices_screen_bootstrap" %in% names(res))

  for (field in c("selected_indices_screen_ordinary", "selected_indices_screen_bootstrap")) {
    sub_list <- res[[field]]
    expect_equal(length(sub_list), q, label = paste(field, "length"))
    for (i in seq_len(q)) {
      sub_idx <- sub_list[[i]]
      expect_type(sub_idx, "integer")
      if (length(sub_idx) > 0) {
        expect_true(all(sub_idx >= 1L & sub_idx <= p),
                    label = paste(field, "phenotype", i, "1-based in [1,p]"))
      }
    }
  }
})
