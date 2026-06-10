# =============================================================================
# Unit tests for ml_methods (ml_methods.R)
#
# Test groups:
#   1.  ZScoreScaler — basic fit, transform, inverse_transform, getters
#   2.  ZScoreScaler — no centering (with_mean = FALSE)
#   3.  ZScoreScaler — constant column (dropped_indices is 1-based)
#   4.  ZScoreScaler — save/load round-trip
#   5.  LpNormScaler — L2 norm fit, transform, inverse_transform, getters
#   6.  LpNormScaler — L1 norm fit, transform, getters
#   7.  LpNormScaler — no centering (with_mean = FALSE)
#   8.  LpNormScaler — constant column (dropped_indices is 1-based)
#   9.  LpNormScaler — save/load round-trip
#   10. RidgeGCV — tall matrix (n > p): getters, grid, solve, predict, gcv_optimal
#   11. RidgeGCV — wide matrix (n < p): rank, predict_path
#   12. RidgeGCV — edge cases (dimension mismatch, negative lambda)
#   13. RidgeCV  — cross-validation rules (index ordering, 1SE threshold)
#   14. RidgeCV  — edge cases (dimension mismatch, invalid folds)
#
# Index convention (R convention throughout):
#   get_dropped_indices() returns 1-based column indices.
#   index_min() and index_1se() in RidgeCV are 1-based.
#   solve_path()$best_index in RidgeGCV is 1-based.
# =============================================================================

library(TRexSelector)

# =============================================================================
# Group 1: ZScoreScaler — basic fit, transform, inverse_transform
# Mirrors C++ test ZScoreScalerBasic
# Data: 3x2 matrix, col1 = {1,2,3}, col2 = {4,6,8}
#   means:  {2, 6};  scales (sample std): {1, 2}
# =============================================================================

test_that("ZScoreScaler basic fit: correct means and scales", {
  data <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  scaler <- ZScoreScaler$new(with_mean = TRUE, with_std = TRUE)
  scaler$fit(data)

  expect_true(scaler$is_fitted())
  expect_true(scaler$get_with_mean())
  expect_true(scaler$get_with_std())
  expect_equal(scaler$get_means()[1], 2.0)
  expect_equal(scaler$get_means()[2], 6.0)
  expect_equal(scaler$get_scales()[1], 1.0)
  expect_equal(scaler$get_scales()[2], 2.0)
})

test_that("ZScoreScaler basic transform produces standardized columns", {
  data <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  scaler <- ZScoreScaler$new()
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  # col 1: (x - 2) / 1 = -1, 0, 1
  expect_equal(result[1, 1], -1.0)
  expect_equal(result[2, 1],  0.0)
  expect_equal(result[3, 1],  1.0)
  # col 2: (x - 6) / 2 = -1, 0, 1
  expect_equal(result[1, 2], -1.0)
  expect_equal(result[2, 2],  0.0)
  expect_equal(result[3, 2],  1.0)
})

test_that("ZScoreScaler inverse_transform restores original values", {
  data <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  scaler <- ZScoreScaler$new()
  scaler$fit(data)
  transformed <- scaler$transform_inplace(data)
  restored    <- scaler$inverse_transform_inplace(transformed)

  expect_equal(restored[1, 1], 1.0, tolerance = 1e-10)
  expect_equal(restored[3, 2], 8.0, tolerance = 1e-10)
})

# =============================================================================
# Group 2: ZScoreScaler — no centering
# Mirrors C++ test ZScoreScalerNoCentering
# Data: {1, 3, 5};  with_mean=FALSE → raw-sum variance
#   var = (1^2 + 3^2 + 5^2) / (n-1) = 35 / 2 = 17.5
#   std = sqrt(17.5)
# =============================================================================

test_that("ZScoreScaler no centering: mean stored as 0, scale uses raw std", {
  data   <- matrix(c(1, 3, 5), nrow = 3, ncol = 1)
  scaler <- ZScoreScaler$new(with_mean = FALSE, with_std = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_means()[1], 0.0)
  expect_equal(scaler$get_scales()[1], sqrt(17.5), tolerance = 1e-12)
})

test_that("ZScoreScaler no centering: transform divides by raw std", {
  data   <- matrix(c(1, 3, 5), nrow = 3, ncol = 1)
  scaler <- ZScoreScaler$new(with_mean = FALSE, with_std = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 1.0 / sqrt(17.5), tolerance = 1e-12)
  expect_equal(result[2, 1], 3.0 / sqrt(17.5), tolerance = 1e-12)
  expect_equal(result[3, 1], 5.0 / sqrt(17.5), tolerance = 1e-12)
})

# =============================================================================
# Group 3: ZScoreScaler — constant column
# Mirrors C++ test ZScoreScalerConstantColumn
# Constant column → std = 0 → scale set to 1.0, column index added to
# dropped_indices.  dropped_indices must be 1-based (R convention).
# =============================================================================

test_that("ZScoreScaler constant column: scale is 1, dropped_indices is 1-based", {
  data   <- matrix(rep(42.0, 3), nrow = 3, ncol = 1)
  scaler <- ZScoreScaler$new()
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 1.0)
  dropped <- scaler$get_dropped_indices()
  expect_length(dropped, 1L)
  expect_equal(dropped[1], 1L)   # 1-based: column 1 (C++ column 0)
})

test_that("ZScoreScaler constant column: transform leaves values unchanged", {
  data   <- matrix(rep(42.0, 3), nrow = 3, ncol = 1)
  scaler <- ZScoreScaler$new()
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 42.0)
  expect_equal(result[2, 1], 42.0)
  expect_equal(result[3, 1], 42.0)
})

# =============================================================================
# Group 4: ZScoreScaler — save/load round-trip
# Mirrors C++ test ZScoreScalerSerialization
# =============================================================================

test_that("ZScoreScaler save/load preserves means, scales, dropped_indices", {
  data     <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  original <- ZScoreScaler$new()
  original$fit(data)

  tf <- tempfile()
  on.exit(unlink(tf))
  original$save(tf)

  restored <- ZScoreScaler$new()
  restored$load(tf)

  expect_true(restored$is_fitted())
  expect_equal(restored$get_means(),           original$get_means())
  expect_equal(restored$get_scales(),          original$get_scales())
  expect_equal(restored$get_dropped_indices(), original$get_dropped_indices())
})

# =============================================================================
# Group 5: LpNormScaler — L2 norm
# Mirrors C++ test LpNormScalerL2
# Data: {1, 2, 3};  with_mean=TRUE → centered: {-1, 0, 1}, L2 norm = sqrt(2)
# =============================================================================

test_that("LpNormScaler L2: correct means and scales", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, with_mean = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_means()[1],  2.0)
  expect_equal(scaler$get_scales()[1], sqrt(2.0))
})

test_that("LpNormScaler L2: transform normalises to unit L2 norm of centered data", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, with_mean = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], -1.0 / sqrt(2.0))
  expect_equal(result[2, 1],  0.0)
  expect_equal(result[3, 1],  1.0 / sqrt(2.0))
})

test_that("LpNormScaler L2: inverse_transform restores original values", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, with_mean = TRUE)
  scaler$fit(data)
  transformed <- scaler$transform_inplace(data)
  restored    <- scaler$inverse_transform_inplace(transformed)

  expect_equal(restored[1, 1], 1.0, tolerance = 1e-10)
  expect_equal(restored[3, 1], 3.0, tolerance = 1e-10)
})

# =============================================================================
# Group 6: LpNormScaler — L1 norm
# Mirrors C++ test LpNormScalerL1
# Data: {1, 2, 3};  with_mean=TRUE → centered: {-1, 0, 1}, L1 norm = 2.0
# =============================================================================

test_that("LpNormScaler L1: correct scale (L1 norm of centered data)", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 1, with_mean = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 2.0)
})

test_that("LpNormScaler L1: transform normalises to unit L1 norm", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 1, with_mean = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], -0.5)
  expect_equal(result[2, 1],  0.0)
  expect_equal(result[3, 1],  0.5)
})

# =============================================================================
# Group 7: LpNormScaler — no centering, L2
# Mirrors C++ test LpNormScalerNoCenteringL2
# Data: {3, 4, 0};  with_mean=FALSE → L2 norm of raw data = sqrt(9+16+0) = 5
# =============================================================================

test_that("LpNormScaler no centering L2: mean is 0, scale is L2 norm of raw data", {
  data   <- matrix(c(3, 4, 0), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, with_mean = FALSE)
  scaler$fit(data)

  expect_equal(scaler$get_means()[1],  0.0)
  expect_equal(scaler$get_scales()[1], 5.0)
})

test_that("LpNormScaler no centering L2: transform divides by L2 norm", {
  data   <- matrix(c(3, 4, 0), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, with_mean = FALSE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 0.6)
  expect_equal(result[2, 1], 0.8)
  expect_equal(result[3, 1], 0.0)
})

# =============================================================================
# Group 8: LpNormScaler — constant column
# Mirrors C++ test LpNormScalerConstantColumn
# Constant column → centered values all 0 → norm = 0 → scale set to 1.0,
# column added to dropped_indices.  dropped_indices must be 1-based.
# =============================================================================

test_that("LpNormScaler constant column: scale is 1, dropped_indices is 1-based", {
  data   <- matrix(rep(5.0, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, with_mean = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 1.0)
  dropped <- scaler$get_dropped_indices()
  expect_length(dropped, 1L)
  expect_equal(dropped[1], 1L)   # 1-based: column 1 (C++ column 0)
})

test_that("LpNormScaler constant column: transform leaves values unchanged", {
  data   <- matrix(rep(5.0, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, with_mean = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 5.0)
  expect_equal(result[2, 1], 5.0)
  expect_equal(result[3, 1], 5.0)
})

# =============================================================================
# Group 9: LpNormScaler — save/load round-trip
# Mirrors C++ test LpNormScalerSerialization
# =============================================================================

test_that("LpNormScaler save/load preserves means, scales, dropped_indices", {
  data     <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  original <- LpNormScaler$new(norm_type = 2, with_mean = TRUE)
  original$fit(data)

  tf <- tempfile()
  on.exit(unlink(tf))
  original$save(tf)

  restored <- LpNormScaler$new(norm_type = 2, with_mean = TRUE)
  restored$load(tf)

  expect_true(restored$is_fitted())
  expect_equal(restored$get_means(),           original$get_means())
  expect_equal(restored$get_scales(),          original$get_scales())
  expect_equal(restored$get_dropped_indices(), original$get_dropped_indices())
})

# =============================================================================
# Group 10: RidgeGCV — tall matrix (n > p)
# Mirrors C++ test RidgeGCVTest::FitTallMatrix
# =============================================================================

test_that("RidgeGCV tall matrix: n_samples, n_features, rank", {
  set.seed(42)
  n <- 100L
  p <- 10L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 0.1

  m <- RidgeGCV$new()
  expect_no_error(m$fit(X, y))
  expect_equal(m$n_samples(),  n)
  expect_equal(m$n_features(), p)
  expect_lte(m$rank(), p)
})

test_that("RidgeGCV tall matrix: default_lambda_grid is monotone increasing", {
  set.seed(42)
  n <- 100L
  p <- 10L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 0.1

  m <- RidgeGCV$new()
  m$fit(X, y)
  grid <- m$default_lambda_grid(50L, 1000.0)

  expect_equal(length(grid), 50L)
  expect_true(all(diff(grid) > 0))
})

test_that("RidgeGCV tall matrix: solve and predict return correct shapes", {
  set.seed(42)
  n <- 100L
  p <- 10L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 0.1

  m <- RidgeGCV$new()
  m$fit(X, y)
  lambda <- m$default_lambda_grid(50L, 1000.0)[11]

  expect_equal(length(m$solve(lambda)),      p)
  expect_equal(length(m$predict(X, lambda)), n)
})

test_that("RidgeGCV tall matrix: gcv_optimal returns positive lambda", {
  set.seed(42)
  n <- 100L
  p <- 10L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 0.1

  m <- RidgeGCV$new()
  m$fit(X, y)

  expect_gt(m$gcv_optimal(50L, 1e-4), 0.0)
})

# =============================================================================
# Group 11: RidgeGCV — wide matrix (n < p)
# Mirrors C++ test RidgeGCVTest::FitWideMatrix
# =============================================================================

test_that("RidgeGCV wide matrix: rank <= n, predict_path is n x k", {
  set.seed(42)
  n <- 20L
  p <- 50L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 0.1

  m <- RidgeGCV$new()
  expect_no_error(m$fit(X, y))
  expect_equal(m$n_samples(),  n)
  expect_equal(m$n_features(), p)
  expect_lte(m$rank(), n)

  grid       <- m$default_lambda_grid(10L)
  path_preds <- m$predict_path(X, grid)
  expect_equal(nrow(path_preds), n)
  expect_equal(ncol(path_preds), 10L)

  expect_gt(m$gcv_optimal(100L, 1e-4), 0.0)
})

# =============================================================================
# Group 12: RidgeGCV — edge cases
# Mirrors C++ test RidgeGCVTest::EdgeCases
# =============================================================================

test_that("RidgeGCV rejects dimension mismatch between X and y", {
  X     <- matrix(rnorm(50), 10, 5)
  y_bad <- rnorm(11)
  m     <- RidgeGCV$new()
  expect_error(m$fit(X, y_bad))
})

test_that("RidgeGCV rejects negative lambda in solve, gcv, predict", {
  set.seed(42)
  X <- matrix(rnorm(50), 10, 5)
  y <- rnorm(10)
  m <- RidgeGCV$new()
  m$fit(X, y)

  expect_error(m$solve(-1.0))
  expect_error(m$gcv(-1.0))
  expect_error(m$predict(X, -1.0))
})

# =============================================================================
# Group 13: RidgeCV — cross-validation rules
# Mirrors C++ test RidgeCVTest::CrossValidationRules
# All indices (index_min, index_1se) are 1-based (R convention).
# =============================================================================

test_that("RidgeCV CV: cv_errors and cv_std have length n_lambda", {
  set.seed(42)
  n <- 150L
  p <- 20L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 2.0

  m <- RidgeCV$new()
  m$fit(X, y, num_folds = 5L, n_lambda = 100L, lambda_ratio = 1000.0, seed = 42L)

  expect_equal(length(m$get_cv_errors()), 100L)
  expect_equal(length(m$get_cv_std()),    100L)
})

test_that("RidgeCV CV: index_1se >= index_min and cv_1se >= cv_min", {
  set.seed(42)
  n <- 150L
  p <- 20L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 2.0

  m <- RidgeCV$new()
  m$fit(X, y, num_folds = 5L, n_lambda = 100L, lambda_ratio = 1000.0, seed = 42L)

  idx_min <- m$index_min()   # 1-based
  idx_1se <- m$index_1se()   # 1-based

  # Lambda grid is increasing; 1SE rule picks a more regularised (larger)
  # lambda → higher index; its CV error must be >= the minimum CV error
  expect_gte(idx_1se, idx_min)
  expect_gte(m$cv_1se(), m$cv_min())

  # Indices must fall inside [1, n_lambda]
  expect_gte(idx_min, 1L)
  expect_lte(idx_1se, 100L)
})

test_that("RidgeCV CV: mse at index_1se <= mse_min + sem_min (1SE rule)", {
  set.seed(42)
  n <- 150L
  p <- 20L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 2.0

  m <- RidgeCV$new()
  m$fit(X, y, num_folds = 5L, n_lambda = 100L, lambda_ratio = 1000.0, seed = 42L)

  mse     <- m$get_cv_errors()
  sem     <- m$get_cv_std()
  idx_min <- m$index_min()   # 1-based
  idx_1se <- m$index_1se()   # 1-based

  threshold <- mse[idx_min] + sem[idx_min]
  expect_lte(mse[idx_1se], threshold)
})

# =============================================================================
# Group 14: RidgeCV — edge cases
# Mirrors C++ test RidgeCVTest::EdgeCases
# =============================================================================

test_that("RidgeCV rejects dimension mismatch between X and y", {
  X     <- matrix(rnorm(100), 20, 5)
  y_bad <- rnorm(21)
  m     <- RidgeCV$new()
  expect_error(m$fit(X, y_bad))
})

test_that("RidgeCV rejects num_folds < 2", {
  X <- matrix(rnorm(100), 20, 5)
  y <- rnorm(20)
  m <- RidgeCV$new()
  expect_error(m$fit(X, y, num_folds = 1L))
})

test_that("RidgeCV rejects num_folds > n", {
  X <- matrix(rnorm(100), 20, 5)
  y <- rnorm(20)
  m <- RidgeCV$new()
  expect_error(m$fit(X, y, num_folds = 25L))
})
