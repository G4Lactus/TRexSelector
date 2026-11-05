# =============================================================================
# Unit tests for ml_methods (ml_methods.R)
#
# Test groups:
#   1.  MLZScoreScaler — basic fit, transform, inverse_transform, getters
#   2.  MLZScoreScaler — no centering (center = FALSE)
#   3.  MLZScoreScaler — constant column (dropped_indices is 1-based)
#   4.  MLZScoreScaler — save/load round-trip
#   5.  MLLpNormScaler — L2 norm fit, transform, inverse_transform, getters
#   6.  MLLpNormScaler — L1 norm fit, transform, getters
#   7.  MLLpNormScaler — no centering (center = FALSE)
#   8.  MLLpNormScaler — constant column (dropped_indices is 1-based)
#   9.  MLLpNormScaler — save/load round-trip
#   13. MLRidgeCV  — cross-validation rules (index ordering, 1SE threshold)
#   14. MLRidgeCV  — edge cases (dimension mismatch, invalid folds)
#
# Index convention (R convention throughout):
#   get_dropped_indices() returns 1-based column indices.
#   index_min() and index_1se() in MLRidgeCV are 1-based.
#   solve_path()$best_index in RidgeGCV is 1-based.
# =============================================================================

library(TRexSelector)

# =============================================================================
# Group 1: MLZScoreScaler — basic fit, transform, inverse_transform
# Mirrors C++ test ZScoreScalerBasic
# Data: 3x2 matrix, col1 = {1,2,3}, col2 = {4,6,8}
#   means:  {2, 6};  scales (sample std): {1, 2}
# =============================================================================

test_that("MLZScoreScaler basic fit: correct means and scales", {
  data <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  scaler <- MLZScoreScaler$new(center = TRUE, scale = TRUE)
  scaler$fit(data)

  expect_true(scaler$is_fitted())
  expect_true(scaler$get_center())
  expect_true(scaler$get_scale())
  expect_equal(scaler$get_centers()[1], 2.0)
  expect_equal(scaler$get_centers()[2], 6.0)
  expect_equal(scaler$get_scales()[1], 1.0)
  expect_equal(scaler$get_scales()[2], 2.0)
})

test_that("MLZScoreScaler basic transform produces standardized columns", {
  data <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  scaler <- MLZScoreScaler$new()
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

test_that("MLZScoreScaler inverse_transform restores original values", {
  data <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  scaler <- MLZScoreScaler$new()
  scaler$fit(data)
  transformed <- scaler$transform_inplace(data)
  restored    <- scaler$inverse_transform_inplace(transformed)

  expect_equal(restored[1, 1], 1.0, tolerance = 1e-10)
  expect_equal(restored[3, 2], 8.0, tolerance = 1e-10)
})

# =============================================================================
# Group 2: MLZScoreScaler — no centering
# Mirrors C++ test ZScoreScalerNoCentering
# Data: {1, 3, 5};  center=FALSE → RMS around 0 (R's scale() behavior)
#   var = (1^2 + 3^2 + 5^2) / (n-1) = 35 / 2 = 17.5
#   std = sqrt(17.5)
# =============================================================================

test_that("MLZScoreScaler no centering: center stored as 0, scale uses RMS around 0", {
  data   <- matrix(c(1, 3, 5), nrow = 3, ncol = 1)
  scaler <- MLZScoreScaler$new(center = FALSE, scale = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_centers()[1], 0.0)
  expect_equal(scaler$get_scales()[1], sqrt(17.5), tolerance = 1e-12)
})

test_that("MLZScoreScaler no centering: transform divides by RMS around 0", {
  data   <- matrix(c(1, 3, 5), nrow = 3, ncol = 1)
  scaler <- MLZScoreScaler$new(center = FALSE, scale = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 1.0 / sqrt(17.5), tolerance = 1e-12)
  expect_equal(result[2, 1], 3.0 / sqrt(17.5), tolerance = 1e-12)
  expect_equal(result[3, 1], 5.0 / sqrt(17.5), tolerance = 1e-12)
})

# =============================================================================
# Group 3: MLZScoreScaler — constant column
# Mirrors C++ test ZScoreScalerConstantColumn
# Constant column → std = 0 → scale set to 1.0, column index added to
# dropped_indices (1-based in R).  sklearn semantics: the column is still
# centered (constant becomes 0); only the division is suppressed.
# =============================================================================

test_that("MLZScoreScaler constant column: scale is 1, dropped_indices is 1-based", {
  data   <- matrix(rep(42.0, 3), nrow = 3, ncol = 1)
  scaler <- MLZScoreScaler$new()
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 1.0)
  dropped <- scaler$get_dropped_indices()
  expect_length(dropped, 1L)
  expect_equal(dropped[1], 1L)   # 1-based: column 1 (C++ column 0)
})

test_that("MLZScoreScaler constant column: still centered to 0, inverse restores", {
  data   <- matrix(rep(42.0, 3), nrow = 3, ncol = 1)
  scaler <- MLZScoreScaler$new()
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 0.0)
  expect_equal(result[2, 1], 0.0)
  expect_equal(result[3, 1], 0.0)

  restored <- scaler$inverse_transform_inplace(result)
  expect_equal(restored[1, 1], 42.0)
  expect_equal(restored[3, 1], 42.0)
})

# =============================================================================
# Group 4: MLZScoreScaler — save/load round-trip
# Mirrors C++ test ZScoreScalerSerialization
# =============================================================================

test_that("MLZScoreScaler save/load preserves means, scales, dropped_indices", {
  data     <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  original <- MLZScoreScaler$new()
  original$fit(data)

  tf <- tempfile()
  on.exit(unlink(tf))
  original$save(tf)

  restored <- MLZScoreScaler$new()
  restored$load(tf)

  expect_true(restored$is_fitted())
  expect_equal(restored$get_centers(),           original$get_centers())
  expect_equal(restored$get_scales(),          original$get_scales())
  expect_equal(restored$get_dropped_indices(), original$get_dropped_indices())
})

# =============================================================================
# Group 5: MLLpNormScaler — L2 norm
# Mirrors C++ test LpNormScalerL2
# Data: {1, 2, 3};  center=TRUE → centered: {-1, 0, 1}, L2 norm = sqrt(2)
# =============================================================================

test_that("MLLpNormScaler L2: correct means and scales", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_centers()[1],  2.0)
  expect_equal(scaler$get_scales()[1], sqrt(2.0))
})

test_that("MLLpNormScaler L2: transform normalises to unit L2 norm of centered data", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], -1.0 / sqrt(2.0))
  expect_equal(result[2, 1],  0.0)
  expect_equal(result[3, 1],  1.0 / sqrt(2.0))
})

test_that("MLLpNormScaler L2: inverse_transform restores original values", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)
  transformed <- scaler$transform_inplace(data)
  restored    <- scaler$inverse_transform_inplace(transformed)

  expect_equal(restored[1, 1], 1.0, tolerance = 1e-10)
  expect_equal(restored[3, 1], 3.0, tolerance = 1e-10)
})

# =============================================================================
# Group 6: MLLpNormScaler — L1 norm
# Mirrors C++ test LpNormScalerL1
# Data: {1, 2, 3};  center=TRUE → centered: {-1, 0, 1}, L1 norm = 2.0
# =============================================================================

test_that("MLLpNormScaler L1: correct scale (L1 norm of centered data)", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 1, center = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 2.0)
})

test_that("MLLpNormScaler L1: transform normalises to unit L1 norm", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 1, center = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], -0.5)
  expect_equal(result[2, 1],  0.0)
  expect_equal(result[3, 1],  0.5)
})

# =============================================================================
# Group 7: MLLpNormScaler — no centering, L2
# Mirrors C++ test LpNormScalerNoCenteringL2
# Data: {3, 4, 0};  center=FALSE → L2 norm of raw data = sqrt(9+16+0) = 5
# =============================================================================

test_that("MLLpNormScaler no centering L2: mean is 0, scale is L2 norm of raw data", {
  data   <- matrix(c(3, 4, 0), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 2, center = FALSE)
  scaler$fit(data)

  expect_equal(scaler$get_centers()[1],  0.0)
  expect_equal(scaler$get_scales()[1], 5.0)
})

test_that("MLLpNormScaler no centering L2: transform divides by L2 norm", {
  data   <- matrix(c(3, 4, 0), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 2, center = FALSE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 0.6)
  expect_equal(result[2, 1], 0.8)
  expect_equal(result[3, 1], 0.0)
})

# =============================================================================
# Group 8: MLLpNormScaler — constant column
# Mirrors C++ test LpNormScalerConstantColumn
# Constant column → centered values all 0 → norm = 0 → scale set to 1.0,
# column added to dropped_indices (1-based).  sklearn semantics: still
# centered (constant becomes 0); only the division is suppressed.
# =============================================================================

test_that("MLLpNormScaler constant column: scale is 1, dropped_indices is 1-based", {
  data   <- matrix(rep(5.0, 3), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 1.0)
  dropped <- scaler$get_dropped_indices()
  expect_length(dropped, 1L)
  expect_equal(dropped[1], 1L)   # 1-based: column 1 (C++ column 0)
})

test_that("MLLpNormScaler constant column: still centered to 0, inverse restores", {
  data   <- matrix(rep(5.0, 3), nrow = 3, ncol = 1)
  scaler <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], 0.0)
  expect_equal(result[2, 1], 0.0)
  expect_equal(result[3, 1], 0.0)

  restored <- scaler$inverse_transform_inplace(result)
  expect_equal(restored[1, 1], 5.0)
  expect_equal(restored[3, 1], 5.0)
})

# =============================================================================
# Group 9: MLLpNormScaler — save/load round-trip
# Mirrors C++ test LpNormScalerSerialization
# =============================================================================

test_that("MLLpNormScaler save/load preserves means, scales, dropped_indices", {
  data     <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  original <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  original$fit(data)

  tf <- tempfile()
  on.exit(unlink(tf))
  original$save(tf)

  restored <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  restored$load(tf)

  expect_true(restored$is_fitted())
  expect_equal(restored$get_centers(),           original$get_centers())
  expect_equal(restored$get_scales(),          original$get_scales())
  expect_equal(restored$get_dropped_indices(), original$get_dropped_indices())
})


# =============================================================================
# Group 10: fit_transform_inplace — R's scale() in one call
# =============================================================================

test_that("MLZScoreScaler fit_transform_inplace matches base::scale()", {
  set.seed(7)
  data     <- matrix(rnorm(30 * 4), 30, 4)
  expected <- scale(data, center = TRUE, scale = TRUE)

  scaler <- MLZScoreScaler$new()
  result <- scaler$fit_transform_inplace(data)

  expected_mat <- expected
  attr(expected_mat, "scaled:center") <- NULL
  attr(expected_mat, "scaled:scale") <- NULL

  expect_true(scaler$is_fitted())
  expect_equal(result, unclass(expected_mat), tolerance = 1e-12)
  expect_equal(scaler$get_centers(), as.numeric(attr(expected, "scaled:center")),
               tolerance = 1e-12)
  expect_equal(scaler$get_scales(), as.numeric(attr(expected, "scaled:scale")),
               tolerance = 1e-12)
})

test_that("MLZScoreScaler center=FALSE matches base::scale(center = FALSE)", {
  set.seed(8)
  data     <- matrix(rnorm(20 * 3) + 5.0, 20, 3)
  expected <- scale(data, center = FALSE, scale = TRUE)

  scaler <- MLZScoreScaler$new(center = FALSE, scale = TRUE)
  result <- scaler$fit_transform_inplace(data)

  expected_mat <- expected
  attr(expected_mat, "scaled:scale") <- NULL

  expect_equal(result, unclass(expected_mat), tolerance = 1e-12)
})

test_that("MLLpNormScaler fit_transform_inplace equals fit + transform", {
  set.seed(9)
  data  <- matrix(rnorm(25 * 3), 25, 3)
  X_one <- data + 0.0
  X_two <- data + 0.0

  one_call <- MLLpNormScaler$new(norm_type = 2)
  r1 <- one_call$fit_transform_inplace(X_one)

  two_calls <- MLLpNormScaler$new(norm_type = 2)
  two_calls$fit(X_two)
  r2 <- two_calls$transform_inplace(X_two)

  expect_equal(r1, r2, tolerance = 0)
})


# =============================================================================
# Group 13: MLRidgeCV — cross-validation rules
# Mirrors C++ test RidgeCVTest::CrossValidationRules
# All indices (index_min, index_1se) are 1-based (R convention).
# =============================================================================

test_that("MLRidgeCV CV: cv_errors and cv_std have length n_lambda", {
  set.seed(42)
  n <- 150L
  p <- 20L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 2.0

  m <- MLRidgeCV$new()
  m$fit(X, y, n_folds = 5L, n_lambda = 100L, lambda_ratio = 1000.0, seed = 42L)

  expect_equal(length(m$get_cv_errors()), 100L)
  expect_equal(length(m$get_cv_std()),    100L)
})

test_that("MLRidgeCV CV: index_1se >= index_min and cv_1se >= cv_min", {
  set.seed(42)
  n <- 150L
  p <- 20L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 2.0

  m <- MLRidgeCV$new()
  m$fit(X, y, n_folds = 5L, n_lambda = 100L, lambda_ratio = 1000.0, seed = 42L)

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

test_that("MLRidgeCV CV: mse at index_1se <= mse_min + sem_min (1SE rule)", {
  set.seed(42)
  n <- 150L
  p <- 20L
  X <- matrix(rnorm(n * p), n, p)
  y <- X %*% rnorm(p) + rnorm(n) * 2.0

  m <- MLRidgeCV$new()
  m$fit(X, y, n_folds = 5L, n_lambda = 100L, lambda_ratio = 1000.0, seed = 42L)

  mse     <- m$get_cv_errors()
  sem     <- m$get_cv_std()
  idx_min <- m$index_min()   # 1-based
  idx_1se <- m$index_1se()   # 1-based

  threshold <- mse[idx_min] + sem[idx_min]
  expect_lte(mse[idx_1se], threshold)
})

# =============================================================================
# Group 14: MLRidgeCV — edge cases
# Mirrors C++ test RidgeCVTest::EdgeCases
# =============================================================================

test_that("MLRidgeCV rejects dimension mismatch between X and y", {
  X     <- matrix(rnorm(100), 20, 5)
  y_bad <- rnorm(21)
  m     <- MLRidgeCV$new()
  expect_error(m$fit(X, y_bad))
})

test_that("MLRidgeCV rejects n_folds < 2", {
  X <- matrix(rnorm(100), 20, 5)
  y <- rnorm(20)
  m <- MLRidgeCV$new()
  expect_error(m$fit(X, y, n_folds = 1L))
})

test_that("MLRidgeCV rejects n_folds > n", {
  X <- matrix(rnorm(100), 20, 5)
  y <- rnorm(20)
  m <- MLRidgeCV$new()
  expect_error(m$fit(X, y, n_folds = 25L))
})

test_that("MLRidgeCV reports the interpolation regime and grid-edge picks", {
  set.seed(7)
  # p < n_train in every fold: no interpolation, default grid of 100 points
  X <- matrix(rnorm(150 * 20), 150, 20)
  y <- X %*% rnorm(20) + rnorm(150)
  m <- MLRidgeCV$new()
  m$fit(X, y, n_folds = 5L, seed = 1L)
  expect_length(m$get_lambdas(), 100L)
  expect_false(m$interpolates())
  expect_type(m$selection_at_grid_edge(), "logical")
  # p >= n_train in every fold: the lambda -> 0 end interpolates
  Xw <- matrix(rnorm(30 * 60), 30, 60)
  yw <- Xw[, 1:3] %*% c(2, -1, 1) + rnorm(30)
  mw <- MLRidgeCV$new()
  mw$fit(Xw, yw, n_folds = 5L, seed = 1L)
  expect_true(mw$interpolates())
  # a two-point grid puts every pick on an edge
  me <- MLRidgeCV$new()
  me$fit(X, y, n_folds = 5L, n_lambda = 2L, seed = 1L)
  expect_true(me$selection_at_grid_edge())
})


# =============================================================================
# Group 15: MLSVD — compute top-M SVD
# =============================================================================

test_that("MLSVD compute returns correct dimensions (tall matrix)", {
  set.seed(42)
  X <- matrix(rnorm(40 * 8), 40, 8)
  M <- 3L
  res <- MLSVD$new()$compute(X, M)

  expect_named(res, c("U", "S", "V"))
  expect_equal(dim(res$U), c(40L, M))
  expect_equal(length(res$S), M)
  expect_equal(dim(res$V), c(8L, M))
})

test_that("MLSVD compute returns correct dimensions (wide matrix)", {
  set.seed(42)
  X <- matrix(rnorm(10 * 30), 10, 30)
  M <- 4L
  res <- MLSVD$new()$compute(X, M)

  expect_equal(dim(res$U), c(10L, M))
  expect_equal(length(res$S), M)
  expect_equal(dim(res$V), c(30L, M))
})

test_that("MLSVD singular values are non-negative and non-increasing", {
  set.seed(7)
  X <- matrix(rnorm(30 * 10), 30, 10)
  res <- MLSVD$new()$compute(X, M = 5L)

  expect_true(all(res$S >= 0))
  expect_true(all(diff(res$S) <= 0))
})

test_that("MLSVD U columns are orthonormal", {
  set.seed(3)
  X  <- matrix(rnorm(40 * 8), 40, 8)
  U  <- MLSVD$new()$compute(X, M = 4L)$U
  UtU <- t(U) %*% U
  expect_equal(UtU, diag(4L), tolerance = 1e-10)
})

test_that("MLSVD V columns are orthonormal", {
  set.seed(5)
  X  <- matrix(rnorm(40 * 8), 40, 8)
  V  <- MLSVD$new()$compute(X, M = 4L)$V
  VtV <- t(V) %*% V
  expect_equal(VtV, diag(4L), tolerance = 1e-10)
})

test_that("MLSVD low-rank reconstruction is accurate (full rank)", {
  # For full-rank M, X = U diag(S) t(V) should be exact.
  set.seed(9)
  n <- 8; p <- 6; M <- 6L
  X   <- matrix(rnorm(n * p), n, p)
  res <- MLSVD$new()$compute(X, M)
  X_rec <- res$U %*% diag(res$S) %*% t(res$V)
  expect_equal(X_rec, X, tolerance = 1e-8)
})

test_that("MLSVD rejects M = 0", {
  expect_error(MLSVD$new()$compute(matrix(rnorm(20), 5, 4), M = 0L))
})

test_that("MLSVD rejects M > min(n, p)", {
  expect_error(MLSVD$new()$compute(matrix(rnorm(20), 5, 4), M = 5L))
})


# =============================================================================
# Group 16: PCA — fit, transform, getters, RAII restore
# =============================================================================

test_that("PCA fit returns correct score and loading dimensions", {
  set.seed(42)
  X <- matrix(rnorm(30 * 8), 30, 8)
  M <- 3L
  pca <- MLPCA$new(X, M, center = TRUE)
  pca$fit()

  expect_equal(dim(pca$get_scores()),   c(30L, M))
  expect_equal(dim(pca$get_loadings()), c(8L, M))
  expect_equal(length(pca$get_explained_variance()), M)
})

test_that("PCA loadings are orthonormal", {
  set.seed(1)
  X <- matrix(rnorm(40 * 6), 40, 6)
  pca <- MLPCA$new(X, M = 4L)
  pca$fit()
  V   <- pca$get_loadings()
  VtV <- t(V) %*% V
  expect_equal(VtV, diag(4L), tolerance = 1e-10)
})

test_that("PCA explained variance is non-negative and non-increasing", {
  set.seed(2)
  X <- matrix(rnorm(50 * 10), 50, 10)
  pca <- MLPCA$new(X, M = 6L)
  pca$fit()
  ev <- pca$get_explained_variance()

  expect_true(all(ev >= 0))
  expect_true(all(diff(ev) <= 1e-10))
})

test_that("PCA get_means stores column means when center = TRUE", {
  set.seed(3)
  X <- matrix(rnorm(20 * 4), 20, 4)
  # Capture column means before construction preprocesses X in-place.
  col_means <- colMeans(X)
  pca <- MLPCA$new(X, M = 2L, center = TRUE, normalize = FALSE)
  pca$fit()
  expect_equal(pca$get_means(), col_means, tolerance = 1e-12)
})

test_that("PCA get_means is zero when center = FALSE", {
  set.seed(4)
  X <- matrix(rnorm(20 * 4), 20, 4)
  pca <- MLPCA$new(X, M = 2L, center = FALSE, normalize = FALSE)
  pca$fit()
  expect_equal(pca$get_means(), rep(0.0, 4L), tolerance = 1e-12)
})

test_that("PCA get_norms is one when normalize = FALSE", {
  set.seed(8)
  X <- matrix(rnorm(20 * 4), 20, 4)
  pca <- MLPCA$new(X, M = 2L, normalize = FALSE)
  pca$fit()
  expect_equal(pca$get_norms(), rep(1.0, 4L), tolerance = 1e-12)
})

test_that("PCA transform scores match fit scores for training data", {
  set.seed(5)
  n <- 30; p <- 6; M <- 3L
  X <- matrix(rnorm(n * p), n, p)
  pca <- MLPCA$new(X, M, center = TRUE)
  pca$fit()
  # X was preprocessed in-place at construction; restore brings it back to
  # the original values. transform() on the restored original X must
  # reproduce the fit scores.
  pca$restore()
  Z_transform <- pca$transform(X)
  expect_equal(Z_transform, pca$get_scores(), tolerance = 1e-10)
})

test_that("PCA transform held-out data has correct shape", {
  set.seed(6)
  X     <- matrix(rnorm(30 * 6), 30, 6)
  X_new <- matrix(rnorm(10 * 6), 10, 6)
  pca <- MLPCA$new(X, M = 3L)
  pca$fit()
  Z_new <- pca$transform(X_new)
  expect_equal(dim(Z_new), c(10L, 3L))
})

test_that("PCA restore un-centers and un-scales X", {
  set.seed(7)
  X <- matrix(rnorm(20 * 4), 20, 4)
  # `X * 1.0` forces a deep copy; R's COW does not protect against direct
  # Rcpp/Eigen writes to the underlying SEXP memory.
  X_orig <- X * 1.0
  pca <- MLPCA$new(X, M = 2L, center = TRUE, normalize = TRUE)
  pca$fit()
  # X was preprocessed in-place at construction; restore should undo that
  pca$restore()
  expect_equal(X, X_orig, tolerance = 1e-12)
})

test_that("PCA rejects M > min(n, p)", {
  expect_error(MLPCA$new(matrix(rnorm(20), 5, 4), M = 5L))
})


# =============================================================================
# Group 17: MLRidge — primal/dual dispatch and coefficient recovery
# =============================================================================

test_that("MLRidge solve returns correct length (tall: n > p)", {
  set.seed(10)
  X <- matrix(rnorm(40 * 5), 40, 5)
  y <- rnorm(40)
  beta <- MLRidge$new()$solve(X, y, lambda = 0.1)
  expect_equal(length(beta), 5L)
})

test_that("MLRidge solve returns correct length (wide: n < p)", {
  set.seed(11)
  X <- matrix(rnorm(10 * 20), 10, 20)
  y <- rnorm(10)
  beta <- MLRidge$new()$solve(X, y, lambda = 1.0)
  expect_equal(length(beta), 20L)
})

test_that("MLRidge satisfies KKT condition (normal equations)", {
  # KKT: (X'X + lambda I) beta == X' y
  set.seed(12)
  n <- 20; p <- 5
  X <- matrix(sin(outer(seq_len(n), seq_len(p))), n, p)
  y <- X %*% c(1, -2, 0, 0.5, -1) + rnorm(n, sd = 0.01)
  lam <- 0.5
  beta <- MLRidge$new()$solve(X, y, lambda = lam)
  lhs <- t(X) %*% X %*% beta + lam * beta
  rhs <- t(X) %*% y
  expect_equal(as.numeric(lhs), as.numeric(rhs), tolerance = 1e-8)
})

test_that("MLRidge dual path satisfies KKT (n < p)", {
  set.seed(13)
  n <- 8; p <- 15
  X   <- matrix(cos(outer(seq_len(n), seq_len(p))), n, p)
  y   <- rnorm(n)
  lam <- 1.0
  beta <- MLRidge$new()$solve(X, y, lambda = lam)
  lhs  <- t(X) %*% X %*% beta + lam * beta
  rhs  <- t(X) %*% y
  expect_equal(as.numeric(lhs), as.numeric(rhs), tolerance = 1e-8)
})

test_that("MLRidge shrinks coefficients toward zero as lambda increases", {
  set.seed(14)
  X <- matrix(rnorm(30 * 5), 30, 5)
  y <- X %*% rep(1, 5) + rnorm(30, sd = 0.1)
  beta_small <- MLRidge$new()$solve(X, y, lambda = 1e-4)
  beta_large <- MLRidge$new()$solve(X, y, lambda = 1e4)
  expect_gt(sqrt(sum(beta_small^2)), sqrt(sum(beta_large^2)))
})

test_that("MLRidge is deterministic", {
  set.seed(99)
  X <- matrix(rnorm(20 * 4), 20, 4)
  y <- rnorm(20)
  b1 <- MLRidge$new()$solve(X, y, lambda = 0.1)
  b2 <- MLRidge$new()$solve(X, y, lambda = 0.1)
  expect_equal(b1, b2)
})

test_that("MLRidge rejects negative lambda", {
  expect_error(MLRidge$new()$solve(matrix(rnorm(20), 5, 4), rnorm(5),
                                        lambda = -1))
})


# =============================================================================
# Group 15: MLElasticNet — coordinate-descent path (enet_gaussian)
# Index convention: lambda grid is DESCENDING (index 1 = largest lambda).
# =============================================================================

test_that("MLElasticNet path: coef is p x n_lambda, converges, dev_ratio rises", {
  set.seed(1)
  n <- 60; p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] * 2 + X[, 3] - X[, 6] + rnorm(n))
  en <- MLElasticNet$new()
  en$fit(X, y, alpha = 1.0, n_lambda = 60)
  cf <- en$get_coef()
  lam <- en$get_lambdas()
  expect_equal(nrow(cf), p)
  expect_equal(ncol(cf), length(lam))
  expect_true(all(diff(lam) <= 0), label = "lambda grid descending")
  expect_true(en$converged())
  dev <- en$get_dev_ratio()
  expect_true(all(diff(dev) >= -1e-8), label = "dev_ratio non-decreasing")
  # true signal active at the least-regularised end (last column)
  expect_true(any(abs(cf[c(1, 3, 6), ncol(cf)]) > 1e-3))
})

test_that("MLElasticNet predict has shape nrow(X_new) x n_lambda", {
  set.seed(2)
  n <- 40; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] - X[, 2] + rnorm(n))
  en <- MLElasticNet$new()$fit(X, y, alpha = 0.5, n_lambda = 30)
  pr <- en$predict(X[1:5, ])
  expect_equal(nrow(pr), 5L)
  expect_equal(ncol(pr), length(en$get_lambdas()))
})

test_that("MLElasticNet fit_grid matches auto path at the same lambdas", {
  set.seed(3)
  n <- 50; p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] * 1.5 + rnorm(n))
  auto <- MLElasticNet$new()$fit(X, y, alpha = 0.5, n_lambda = 40)
  grid <- MLElasticNet$new()$fit_grid(X, y, auto$get_lambdas(), alpha = 0.5)
  expect_equal(grid$get_coef(), auto$get_coef(), tolerance = 1e-9)
})

test_that("MLElasticNet ridge limit (alpha=0) leaves all coefficients non-zero", {
  set.seed(4)
  n <- 40; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X %*% rep(1, p) + rnorm(n))
  en <- MLElasticNet$new()$fit(X, y, alpha = 0.0, n_lambda = 30)
  last <- en$get_coef()[, ncol(en$get_coef())]
  expect_true(all(abs(last) > 0))
})


# =============================================================================
# Group 16: MLElasticNetCV — K-fold CV lambda selection (enet_cv_ccd)
# Index convention: grid DESCENDING, indices 1-based; index_1se <= index_min.
# =============================================================================

test_that("MLElasticNetCV: cv_1se >= cv_min and index_1se <= index_min (1-based)", {
  set.seed(5)
  n <- 80; p <- 12
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] * 2 - X[, 4] + rnorm(n))
  cv <- MLElasticNetCV$new()
  cv$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 100, seed = 0)
  lam <- cv$get_lambdas()
  expect_true(all(diff(lam) <= 0), label = "lambda grid descending")
  expect_gte(cv$cv_1se(), cv$cv_min())
  idx_min <- cv$index_min()
  idx_1se <- cv$index_1se()
  expect_true(idx_min >= 1L && idx_min <= length(lam))
  expect_true(idx_1se >= 1L && idx_1se <= idx_min)
  expect_length(cv$get_cv_errors(), length(lam))
  expect_length(cv$get_cv_std(), length(lam))
})

test_that("MLElasticNetCV is deterministic under a fixed seed", {
  set.seed(6)
  n <- 60; p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] - X[, 3] + rnorm(n))
  a <- MLElasticNetCV$new()$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 50, seed = 9)
  b <- MLElasticNetCV$new()$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 50, seed = 9)
  expect_equal(a$cv_min(), b$cv_min())
  expect_equal(a$get_cv_errors(), b$get_cv_errors())
})

test_that("MLElasticNetCV rejects dimension mismatch and invalid folds", {
  X <- matrix(rnorm(50 * 5), 50, 5)
  expect_error(MLElasticNetCV$new()$fit(X, rnorm(49)))
  expect_error(MLElasticNetCV$new()$fit(X, rnorm(50), n_folds = 1))
})


# =============================================================================
# Memory-mapped inputs (zero-copy dispatch on mmap_matrix handles)
# =============================================================================

test_that("MLZScoreScaler operates zero-copy on an mmap_matrix handle", {
  set.seed(31)
  n <- 40; p <- 6
  X <- matrix(rnorm(n * p, mean = 3, sd = 2), n, p)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mm <- mem_matrix_from(X, f)

  sc <- MLZScoreScaler$new()
  sc$fit(mm)
  sc$transform_inplace(mm)

  # Must match the in-memory path bit-for-bit (same Eigen code path).
  X_mem <- X + 0
  sc_mem <- MLZScoreScaler$new()
  sc_mem$fit(X_mem)
  X_mem <- sc_mem$transform_inplace(X_mem)
  expect_identical(as.matrix(mm), X_mem)

  # Inverse round-trip restores the file contents.
  sc$inverse_transform_inplace(mm)
  expect_equal(as.matrix(mm), X, tolerance = 1e-12)
})

test_that("MLLpNormScaler fit_transform_inplace works on an mmap_matrix handle", {
  set.seed(32)
  n <- 40; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mm <- mem_matrix_from(X, f)

  l2 <- MLLpNormScaler$new(norm_type = 2L, center = TRUE)
  l2$fit_transform_inplace(mm)
  expect_equal(max(abs(sqrt(colSums(as.matrix(mm)^2)) - 1)), 0, tolerance = 1e-12)

  l2$inverse_transform_inplace(mm)
  expect_equal(as.matrix(mm), X, tolerance = 1e-12)
})

test_that("scaler fit() reads a readonly mmap_matrix; the in-place transforms need readwrite", {
  set.seed(33)
  X <- matrix(rnorm(20 * 4), 20, 4)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mem_matrix_from(X, f)
  ro <- mem_matrix(f, rows = 20, cols = 4, mode = "readonly")

  # fit() only reads: it binds a const map, so a "readonly"
  # handle is accepted and the statistics equal the in-memory fit exactly.
  sc <- MLZScoreScaler$new()
  expect_no_error(sc$fit(ro))
  sc_mem <- MLZScoreScaler$new(); sc_mem$fit(X)
  expect_identical(sc$get_centers(), sc_mem$get_centers())
  expect_identical(sc$get_scales(), sc_mem$get_scales())
  # The in-place transforms write the file and still need a "readwrite" handle.
  expect_error(sc$transform_inplace(ro), "ReadOnly")
})

test_that("MLRidgeCV fits zero-copy from a readonly mmap_matrix handle", {
  set.seed(34)
  n <- 60; p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] * 2 + rnorm(n))
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mem_matrix_from(X, f)
  ro <- mem_matrix(f, rows = n, cols = p, mode = "readonly")

  cv_mm <- MLRidgeCV$new()
  cv_mm$fit(ro, y, n_folds = 5, n_lambda = 20, seed = 7)
  cv_mem <- MLRidgeCV$new()
  cv_mem$fit(X, y, n_folds = 5, n_lambda = 20, seed = 7)

  expect_equal(cv_mm$cv_min(), cv_mem$cv_min())
  expect_equal(cv_mm$get_cv_errors(), cv_mem$get_cv_errors())
  expect_error(MLRidgeCV$new()$fit(ro, rnorm(n - 1)))  # dimension mismatch
})


# =============================================================================
# Memory-mapped inputs of the regression / SVD / PCA classes: the mmap path
# calls the same core through an equivalent view, so it must match the
# in-memory path bit-for-bit; the read-only methods accept a "readonly"
# mapping.
# =============================================================================

local({
  set.seed(33)
  n <- 50; p <- 12
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 2 - X[, 4] + rnorm(n)
  f <- tempfile(fileext = ".bin")
  invisible(mem_matrix_from(X, f))                 # the file, written once
  ro <- mem_matrix(f, n, p, mode = "readonly")     # read-only handle
  on.exit(unlink(f), add = TRUE)

  test_that("MLRidgeCV on a readonly mmap_matrix equals the in-memory fit", {
    a <- MLRidgeCV$new(); a$fit(X, y, n_folds = 5, n_lambda = 30, seed = 1)
    b <- MLRidgeCV$new(); b$fit(ro, y, n_folds = 5, n_lambda = 30, seed = 1)
    expect_identical(b$get_cv_errors(), a$get_cv_errors())
    expect_identical(b$cv_1se(), a$cv_1se())
  })

  test_that("MLElasticNet fit / fit_grid / predict on a readonly mmap_matrix equal the in-memory path", {
    a <- MLElasticNet$new(); a$fit(X, y, alpha = 0.6, n_lambda = 25)
    b <- MLElasticNet$new(); b$fit(ro, y, alpha = 0.6, n_lambda = 25)
    expect_identical(b$get_coef(), a$get_coef())
    expect_identical(b$predict(ro), a$predict(X))
    g <- a$get_lambdas()[3:8]
    a2 <- MLElasticNet$new(); a2$fit_grid(X, y, g, alpha = 0.6)
    b2 <- MLElasticNet$new(); b2$fit_grid(ro, y, g, alpha = 0.6)
    expect_identical(b2$get_coef(), a2$get_coef())
  })

  test_that("MLElasticNetCV on a readonly mmap_matrix equals the in-memory fit", {
    a <- MLElasticNetCV$new(); a$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 25, seed = 2)
    b <- MLElasticNetCV$new(); b$fit(ro, y, alpha = 0.5, n_folds = 5, n_lambda = 25, seed = 2)
    expect_identical(b$get_cv_errors(), a$get_cv_errors())
    expect_identical(b$index_min(), a$index_min())
  })

  test_that("MLSVD and MLRidge on a readonly mmap_matrix equal the in-memory path", {
    expect_identical(MLSVD$new()$compute(ro, M = 3), MLSVD$new()$compute(X, M = 3))
    expect_identical(MLRidge$new()$solve(ro, y, lambda = 0.5),
                     MLRidge$new()$solve(X, y, lambda = 0.5))
  })

  test_that("MLPCA on a readwrite mmap_matrix centers the file in place and restores it", {
    rw <- mem_matrix(f, n, p, mode = "readwrite")
    X0 <- as.matrix(rw)
    Xm <- X0 + 0
    a <- MLPCA$new(Xm, M = 3); a$fit()
    b <- MLPCA$new(rw, M = 3)
    expect_identical(as.matrix(rw), Xm)              # the same in-place preprocessing
    b$fit()
    expect_identical(b$get_loadings(), a$get_loadings())
    expect_identical(b$get_explained_variance(), a$get_explained_variance())
    expect_identical(b$transform(ro), a$transform(ro))
    a$restore(); b$restore()
    expect_equal(as.matrix(rw), X0, tolerance = 1e-12)
    expect_equal(Xm, X0, tolerance = 1e-12)
  })

  test_that("an integer matrix is coerced once and the R matrix is not read twice", {
    Xi <- matrix(1:(n * p), n, p)
    expect_no_error(MLRidge$new()$solve(Xi, y, lambda = 1))
    expect_no_error(MLSVD$new()$compute(Xi, M = 2))
  })
})


# =============================================================================
# low_memory engine: the raw-X coordinate descent agrees with the copying
# engine to rounding and reads a readonly mmap_matrix in place.
# =============================================================================

test_that("MLElasticNet(low_memory = TRUE) agrees with the copying engine", {
  set.seed(81)
  n <- 80; p <- 30
  X <- matrix(rnorm(n * p), n, p); X[, 4] <- X[, 4] + 500   # a large-mean column
  y <- X[, 1] * 2 - X[, 4] * 0.01 + rnorm(n)
  a <- MLElasticNet$new(); a$fit(X, y, alpha = 0.7, n_lambda = 30, tol = 1e-9)
  b <- MLElasticNet$new(low_memory = TRUE); b$fit(X, y, alpha = 0.7, n_lambda = 30, tol = 1e-9)
  expect_equal(b$get_lambdas(), a$get_lambdas(), tolerance = 1e-12)
  expect_equal(b$get_coef(), a$get_coef(), tolerance = 1e-7)
  expect_equal(b$get_intercepts(), a$get_intercepts(), tolerance = 1e-7)
  expect_equal(b$predict(X[1:5, ]), a$predict(X[1:5, ]), tolerance = 1e-7)

  f <- tempfile(fileext = ".bin"); on.exit(unlink(f), add = TRUE)
  invisible(mem_matrix_from(X, f))
  ro <- mem_matrix(f, n, p, mode = "readonly")
  c <- MLElasticNet$new(low_memory = TRUE); c$fit(ro, y, alpha = 0.7, n_lambda = 30, tol = 1e-9)
  expect_identical(c$get_coef(), b$get_coef())   # the same engine on the same numbers
})

test_that("MLElasticNetCV(low_memory = TRUE) agrees with the copying engine", {
  set.seed(82)
  n <- 80; p <- 30
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 2] * 2 + rnorm(n)
  a <- MLElasticNetCV$new(); a$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 30, seed = 3, tol = 1e-9)
  b <- MLElasticNetCV$new(low_memory = TRUE); b$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 30, seed = 3, tol = 1e-9)
  expect_equal(b$get_cv_errors(), a$get_cv_errors(), tolerance = 1e-7)
  expect_identical(b$index_min(), a$index_min())
  expect_identical(b$index_1se(), a$index_1se())
})

test_that("MLRidgeCV(low_memory = TRUE) agrees with the SVD route and reads a readonly mmap", {
  set.seed(84)
  n <- 60; p <- 300
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 2 + rnorm(n)
  a <- MLRidgeCV$new(); a$fit(X, y, n_folds = 5, n_lambda = 40, seed = 4)
  b <- MLRidgeCV$new(low_memory = TRUE); b$fit(X, y, n_folds = 5, n_lambda = 40, seed = 4)
  expect_equal(b$get_lambdas(), a$get_lambdas(), tolerance = 1e-14)
  expect_equal(b$get_cv_errors(), a$get_cv_errors(), tolerance = 1e-7)
  expect_identical(b$index_min(), a$index_min())
  f <- tempfile(fileext = ".bin"); on.exit(unlink(f), add = TRUE)
  invisible(mem_matrix_from(X, f))
  ro <- mem_matrix(f, n, p, mode = "readonly")
  c <- MLRidgeCV$new(low_memory = TRUE); c$fit(ro, y, n_folds = 5, n_lambda = 40, seed = 4)
  expect_identical(c$get_cv_errors(), b$get_cv_errors())
})

test_that("MLSVD thresholds select the randomized (no-copy) path and give a valid decomposition", {
  set.seed(85)
  n <- 120; p <- 80
  # A decaying spectrum (rank-4 signal above noise): the regime the randomized
  # route is built for. On a flat Gaussian spectrum it is only approximate.
  X <- matrix(rnorm(n * 4), n, 4) %*% diag(c(40, 30, 20, 10)) %*% matrix(rnorm(4 * p), 4, p) +
       0.1 * matrix(rnorm(n * p), n, p)
  direct <- MLSVD$new()$compute(X, M = 4)
  rnd    <- MLSVD$new(min_dim_thresh = 16, block_size = 32)$compute(X, M = 4)
  expect_equal(dim(rnd$U), c(n, 4)); expect_equal(dim(rnd$V), c(p, 4))
  expect_equal(crossprod(rnd$U), diag(4), tolerance = 1e-10)
  expect_equal(crossprod(rnd$V), diag(4), tolerance = 1e-10)
  expect_equal(rnd$S, direct$S, tolerance = 1e-4)   # the randomized route's accuracy
})

test_that("MLElasticNet$n_nonconverged() and the thread controls are exposed", {
  set.seed(86)
  X <- matrix(rnorm(60 * 8), 60, 8); y <- X[, 1] + rnorm(60)
  en <- MLElasticNet$new(); en$fit(X, y, n_lambda = 10)
  expect_identical(en$n_nonconverged(), 0L)
  expect_true(en$converged())
  # The library's default budget starts at 1; the setter returns the previous budget
  expect_true(trex_get_max_threads() >= 1)
  expect_identical(trex_get_num_threads(), 1L)
  expect_identical(trex_set_num_threads(2), 1L)
  expect_identical(trex_get_num_threads(), 2L)
  expect_identical(trex_set_num_threads(1), 2L)
  expect_identical(trex_get_num_threads(), 1L)
  expect_error(trex_set_num_threads(0))
  # n_threads of the kernels: NULL is the default budget, a budget changes no result
  expect_error(MLZScoreScaler$new(n_threads = 0))
  sc1 <- MLZScoreScaler$new(); sc1$fit(X)
  sc2 <- MLZScoreScaler$new(n_threads = 2); sc2$fit(X)
  expect_equal(sc2$get_scales(), sc1$get_scales())
  expect_error(ml_agglomerative_cluster(X, n_threads = 0))
  l1 <- ml_agglomerative_cluster(X, method = ml_linkage_method$Average)
  l2 <- ml_agglomerative_cluster(X, method = ml_linkage_method$Average, n_threads = 2)
  expect_equal(l2, l1)
  d1 <- trex_normalize_design(X, y)
  d2 <- trex_normalize_design(X, y, n_threads = 2)
  expect_equal(d2$X, d1$X)
  expect_equal(MLRidge$new()$solve(X, y, lambda = 0.5, n_threads = 2),
               MLRidge$new()$solve(X, y, lambda = 0.5))
  # mem_matrix_from coerces an integer matrix explicitly
  f <- tempfile(fileext = ".bin"); on.exit(unlink(f), add = TRUE)
  mm <- mem_matrix_from(matrix(1:12, 3, 4), f)
  expect_equal(as.matrix(mm), matrix(as.double(1:12), 3, 4))
})
