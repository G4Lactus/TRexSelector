# =============================================================================
# Unit tests for ml_methods (ml_methods.R)
#
# Test groups:
#   1.  ZScoreScaler — basic fit, transform, inverse_transform, getters
#   2.  ZScoreScaler — no centering (center = FALSE)
#   3.  ZScoreScaler — constant column (dropped_indices is 1-based)
#   4.  ZScoreScaler — save/load round-trip
#   5.  LpNormScaler — L2 norm fit, transform, inverse_transform, getters
#   6.  LpNormScaler — L1 norm fit, transform, getters
#   7.  LpNormScaler — no centering (center = FALSE)
#   8.  LpNormScaler — constant column (dropped_indices is 1-based)
#   9.  LpNormScaler — save/load round-trip
#   13. RidgeCV  — cross-validation rules (index ordering, 1SE threshold)
#   14. RidgeCV  — edge cases (dimension mismatch, invalid folds)
#
# Index convention (R convention throughout):
#   get_dropped_indices() returns 1-based column indices.
#   index_min() and index_1se() in RidgeCV are 1-based.
#   solve_path()$best_index in RidgeGCV is 1-based.
# =============================================================================

library(TRexSelectorNeo)

# =============================================================================
# Group 1: ZScoreScaler — basic fit, transform, inverse_transform
# Mirrors C++ test ZScoreScalerBasic
# Data: 3x2 matrix, col1 = {1,2,3}, col2 = {4,6,8}
#   means:  {2, 6};  scales (sample std): {1, 2}
# =============================================================================

test_that("ZScoreScaler basic fit: correct means and scales", {
  data <- matrix(c(1, 2, 3, 4, 6, 8), nrow = 3, ncol = 2)
  scaler <- ZScoreScaler$new(center = TRUE, scale = TRUE)
  scaler$fit(data)

  expect_true(scaler$is_fitted())
  expect_true(scaler$get_center())
  expect_true(scaler$get_scale())
  expect_equal(scaler$get_centers()[1], 2.0)
  expect_equal(scaler$get_centers()[2], 6.0)
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
# Data: {1, 3, 5};  center=FALSE → RMS around 0 (R's scale() behavior)
#   var = (1^2 + 3^2 + 5^2) / (n-1) = 35 / 2 = 17.5
#   std = sqrt(17.5)
# =============================================================================

test_that("ZScoreScaler no centering: center stored as 0, scale uses RMS around 0", {
  data   <- matrix(c(1, 3, 5), nrow = 3, ncol = 1)
  scaler <- ZScoreScaler$new(center = FALSE, scale = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_centers()[1], 0.0)
  expect_equal(scaler$get_scales()[1], sqrt(17.5), tolerance = 1e-12)
})

test_that("ZScoreScaler no centering: transform divides by RMS around 0", {
  data   <- matrix(c(1, 3, 5), nrow = 3, ncol = 1)
  scaler <- ZScoreScaler$new(center = FALSE, scale = TRUE)
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
# dropped_indices (1-based in R).  sklearn semantics: the column is still
# centered (constant becomes 0); only the division is suppressed.
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

test_that("ZScoreScaler constant column: still centered to 0, inverse restores", {
  data   <- matrix(rep(42.0, 3), nrow = 3, ncol = 1)
  scaler <- ZScoreScaler$new()
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
  expect_equal(restored$get_centers(),           original$get_centers())
  expect_equal(restored$get_scales(),          original$get_scales())
  expect_equal(restored$get_dropped_indices(), original$get_dropped_indices())
})

# =============================================================================
# Group 5: LpNormScaler — L2 norm
# Mirrors C++ test LpNormScalerL2
# Data: {1, 2, 3};  center=TRUE → centered: {-1, 0, 1}, L2 norm = sqrt(2)
# =============================================================================

test_that("LpNormScaler L2: correct means and scales", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_centers()[1],  2.0)
  expect_equal(scaler$get_scales()[1], sqrt(2.0))
})

test_that("LpNormScaler L2: transform normalises to unit L2 norm of centered data", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], -1.0 / sqrt(2.0))
  expect_equal(result[2, 1],  0.0)
  expect_equal(result[3, 1],  1.0 / sqrt(2.0))
})

test_that("LpNormScaler L2: inverse_transform restores original values", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)
  transformed <- scaler$transform_inplace(data)
  restored    <- scaler$inverse_transform_inplace(transformed)

  expect_equal(restored[1, 1], 1.0, tolerance = 1e-10)
  expect_equal(restored[3, 1], 3.0, tolerance = 1e-10)
})

# =============================================================================
# Group 6: LpNormScaler — L1 norm
# Mirrors C++ test LpNormScalerL1
# Data: {1, 2, 3};  center=TRUE → centered: {-1, 0, 1}, L1 norm = 2.0
# =============================================================================

test_that("LpNormScaler L1: correct scale (L1 norm of centered data)", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 1, center = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 2.0)
})

test_that("LpNormScaler L1: transform normalises to unit L1 norm", {
  data   <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 1, center = TRUE)
  scaler$fit(data)
  result <- scaler$transform_inplace(data)

  expect_equal(result[1, 1], -0.5)
  expect_equal(result[2, 1],  0.0)
  expect_equal(result[3, 1],  0.5)
})

# =============================================================================
# Group 7: LpNormScaler — no centering, L2
# Mirrors C++ test LpNormScalerNoCenteringL2
# Data: {3, 4, 0};  center=FALSE → L2 norm of raw data = sqrt(9+16+0) = 5
# =============================================================================

test_that("LpNormScaler no centering L2: mean is 0, scale is L2 norm of raw data", {
  data   <- matrix(c(3, 4, 0), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, center = FALSE)
  scaler$fit(data)

  expect_equal(scaler$get_centers()[1],  0.0)
  expect_equal(scaler$get_scales()[1], 5.0)
})

test_that("LpNormScaler no centering L2: transform divides by L2 norm", {
  data   <- matrix(c(3, 4, 0), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, center = FALSE)
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
# column added to dropped_indices (1-based).  sklearn semantics: still
# centered (constant becomes 0); only the division is suppressed.
# =============================================================================

test_that("LpNormScaler constant column: scale is 1, dropped_indices is 1-based", {
  data   <- matrix(rep(5.0, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, center = TRUE)
  scaler$fit(data)

  expect_equal(scaler$get_scales()[1], 1.0)
  dropped <- scaler$get_dropped_indices()
  expect_length(dropped, 1L)
  expect_equal(dropped[1], 1L)   # 1-based: column 1 (C++ column 0)
})

test_that("LpNormScaler constant column: still centered to 0, inverse restores", {
  data   <- matrix(rep(5.0, 3), nrow = 3, ncol = 1)
  scaler <- LpNormScaler$new(norm_type = 2, center = TRUE)
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
# Group 9: LpNormScaler — save/load round-trip
# Mirrors C++ test LpNormScalerSerialization
# =============================================================================

test_that("LpNormScaler save/load preserves means, scales, dropped_indices", {
  data     <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  original <- LpNormScaler$new(norm_type = 2, center = TRUE)
  original$fit(data)

  tf <- tempfile()
  on.exit(unlink(tf))
  original$save(tf)

  restored <- LpNormScaler$new(norm_type = 2, center = TRUE)
  restored$load(tf)

  expect_true(restored$is_fitted())
  expect_equal(restored$get_centers(),           original$get_centers())
  expect_equal(restored$get_scales(),          original$get_scales())
  expect_equal(restored$get_dropped_indices(), original$get_dropped_indices())
})


# =============================================================================
# Group 10: fit_transform_inplace — R's scale() in one call
# =============================================================================

test_that("ZScoreScaler fit_transform_inplace matches base::scale()", {
  set.seed(7)
  data     <- matrix(rnorm(30 * 4), 30, 4)
  expected <- scale(data, center = TRUE, scale = TRUE)

  scaler <- ZScoreScaler$new()
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

test_that("ZScoreScaler center=FALSE matches base::scale(center = FALSE)", {
  set.seed(8)
  data     <- matrix(rnorm(20 * 3) + 5.0, 20, 3)
  expected <- scale(data, center = FALSE, scale = TRUE)

  scaler <- ZScoreScaler$new(center = FALSE, scale = TRUE)
  result <- scaler$fit_transform_inplace(data)

  expected_mat <- expected
  attr(expected_mat, "scaled:scale") <- NULL

  expect_equal(result, unclass(expected_mat), tolerance = 1e-12)
})

test_that("LpNormScaler fit_transform_inplace equals fit + transform", {
  set.seed(9)
  data  <- matrix(rnorm(25 * 3), 25, 3)
  X_one <- data + 0.0
  X_two <- data + 0.0

  one_call <- LpNormScaler$new(norm_type = 2)
  r1 <- one_call$fit_transform_inplace(X_one)

  two_calls <- LpNormScaler$new(norm_type = 2)
  two_calls$fit(X_two)
  r2 <- two_calls$transform_inplace(X_two)

  expect_equal(r1, r2, tolerance = 0)
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


# =============================================================================
# Group 15: SVDSolver — compute top-M SVD
# =============================================================================

test_that("SVDSolver compute returns correct dimensions (tall matrix)", {
  set.seed(42)
  X <- matrix(rnorm(40 * 8), 40, 8)
  M <- 3L
  res <- SVDSolver$new()$compute(X, M)

  expect_named(res, c("U", "S", "V"))
  expect_equal(dim(res$U), c(40L, M))
  expect_equal(length(res$S), M)
  expect_equal(dim(res$V), c(8L, M))
})

test_that("SVDSolver compute returns correct dimensions (wide matrix)", {
  set.seed(42)
  X <- matrix(rnorm(10 * 30), 10, 30)
  M <- 4L
  res <- SVDSolver$new()$compute(X, M)

  expect_equal(dim(res$U), c(10L, M))
  expect_equal(length(res$S), M)
  expect_equal(dim(res$V), c(30L, M))
})

test_that("SVDSolver singular values are non-negative and non-increasing", {
  set.seed(7)
  X <- matrix(rnorm(30 * 10), 30, 10)
  res <- SVDSolver$new()$compute(X, M = 5L)

  expect_true(all(res$S >= 0))
  expect_true(all(diff(res$S) <= 0))
})

test_that("SVDSolver U columns are orthonormal", {
  set.seed(3)
  X  <- matrix(rnorm(40 * 8), 40, 8)
  U  <- SVDSolver$new()$compute(X, M = 4L)$U
  UtU <- t(U) %*% U
  expect_equal(UtU, diag(4L), tolerance = 1e-10)
})

test_that("SVDSolver V columns are orthonormal", {
  set.seed(5)
  X  <- matrix(rnorm(40 * 8), 40, 8)
  V  <- SVDSolver$new()$compute(X, M = 4L)$V
  VtV <- t(V) %*% V
  expect_equal(VtV, diag(4L), tolerance = 1e-10)
})

test_that("SVDSolver low-rank reconstruction is accurate (full rank)", {
  # For full-rank M, X = U diag(S) t(V) should be exact.
  set.seed(9)
  n <- 8; p <- 6; M <- 6L
  X   <- matrix(rnorm(n * p), n, p)
  res <- SVDSolver$new()$compute(X, M)
  X_rec <- res$U %*% diag(res$S) %*% t(res$V)
  expect_equal(X_rec, X, tolerance = 1e-8)
})

test_that("SVDSolver rejects M = 0", {
  expect_error(SVDSolver$new()$compute(matrix(rnorm(20), 5, 4), M = 0L))
})

test_that("SVDSolver rejects M > min(n, p)", {
  expect_error(SVDSolver$new()$compute(matrix(rnorm(20), 5, 4), M = 5L))
})


# =============================================================================
# Group 16: PCA — fit, transform, getters, RAII restore
# =============================================================================

test_that("PCA fit returns correct score and loading dimensions", {
  set.seed(42)
  X <- matrix(rnorm(30 * 8), 30, 8)
  M <- 3L
  pca <- PCA$new(X, M, center = TRUE)
  pca$fit()

  expect_equal(dim(pca$get_scores()),   c(30L, M))
  expect_equal(dim(pca$get_loadings()), c(8L, M))
  expect_equal(length(pca$get_explained_variance()), M)
})

test_that("PCA loadings are orthonormal", {
  set.seed(1)
  X <- matrix(rnorm(40 * 6), 40, 6)
  pca <- PCA$new(X, M = 4L)
  pca$fit()
  V   <- pca$get_loadings()
  VtV <- t(V) %*% V
  expect_equal(VtV, diag(4L), tolerance = 1e-10)
})

test_that("PCA explained variance is non-negative and non-increasing", {
  set.seed(2)
  X <- matrix(rnorm(50 * 10), 50, 10)
  pca <- PCA$new(X, M = 6L)
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
  pca <- PCA$new(X, M = 2L, center = TRUE, normalize = FALSE)
  pca$fit()
  expect_equal(pca$get_means(), col_means, tolerance = 1e-12)
})

test_that("PCA get_means is zero when center = FALSE", {
  set.seed(4)
  X <- matrix(rnorm(20 * 4), 20, 4)
  pca <- PCA$new(X, M = 2L, center = FALSE, normalize = FALSE)
  pca$fit()
  expect_equal(pca$get_means(), rep(0.0, 4L), tolerance = 1e-12)
})

test_that("PCA get_norms is one when normalize = FALSE", {
  set.seed(8)
  X <- matrix(rnorm(20 * 4), 20, 4)
  pca <- PCA$new(X, M = 2L, normalize = FALSE)
  pca$fit()
  expect_equal(pca$get_norms(), rep(1.0, 4L), tolerance = 1e-12)
})

test_that("PCA transform scores match fit scores for training data", {
  set.seed(5)
  n <- 30; p <- 6; M <- 3L
  X <- matrix(rnorm(n * p), n, p)
  pca <- PCA$new(X, M, center = TRUE)
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
  pca <- PCA$new(X, M = 3L)
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
  pca <- PCA$new(X, M = 2L, center = TRUE, normalize = TRUE)
  pca$fit()
  # X was preprocessed in-place at construction; restore should undo that
  pca$restore()
  expect_equal(X, X_orig, tolerance = 1e-12)
})

test_that("PCA rejects M > min(n, p)", {
  expect_error(PCA$new(matrix(rnorm(20), 5, 4), M = 5L))
})


# =============================================================================
# Group 17: RidgeSolver — primal/dual dispatch and coefficient recovery
# =============================================================================

test_that("RidgeSolver solve returns correct length (tall: n > p)", {
  set.seed(10)
  X <- matrix(rnorm(40 * 5), 40, 5)
  y <- rnorm(40)
  beta <- RidgeSolver$new()$solve(X, y, lambda = 0.1)
  expect_equal(length(beta), 5L)
})

test_that("RidgeSolver solve returns correct length (wide: n < p)", {
  set.seed(11)
  X <- matrix(rnorm(10 * 20), 10, 20)
  y <- rnorm(10)
  beta <- RidgeSolver$new()$solve(X, y, lambda = 1.0)
  expect_equal(length(beta), 20L)
})

test_that("RidgeSolver satisfies KKT condition (normal equations)", {
  # KKT: (X'X + lambda I) beta == X' y
  set.seed(12)
  n <- 20; p <- 5
  X <- matrix(sin(outer(seq_len(n), seq_len(p))), n, p)
  y <- X %*% c(1, -2, 0, 0.5, -1) + rnorm(n, sd = 0.01)
  lam <- 0.5
  beta <- RidgeSolver$new()$solve(X, y, lambda = lam)
  lhs <- t(X) %*% X %*% beta + lam * beta
  rhs <- t(X) %*% y
  expect_equal(as.numeric(lhs), as.numeric(rhs), tolerance = 1e-8)
})

test_that("RidgeSolver dual path satisfies KKT (n < p)", {
  set.seed(13)
  n <- 8; p <- 15
  X   <- matrix(cos(outer(seq_len(n), seq_len(p))), n, p)
  y   <- rnorm(n)
  lam <- 1.0
  beta <- RidgeSolver$new()$solve(X, y, lambda = lam)
  lhs  <- t(X) %*% X %*% beta + lam * beta
  rhs  <- t(X) %*% y
  expect_equal(as.numeric(lhs), as.numeric(rhs), tolerance = 1e-8)
})

test_that("RidgeSolver shrinks coefficients toward zero as lambda increases", {
  set.seed(14)
  X <- matrix(rnorm(30 * 5), 30, 5)
  y <- X %*% rep(1, 5) + rnorm(30, sd = 0.1)
  beta_small <- RidgeSolver$new()$solve(X, y, lambda = 1e-4)
  beta_large <- RidgeSolver$new()$solve(X, y, lambda = 1e4)
  expect_gt(sqrt(sum(beta_small^2)), sqrt(sum(beta_large^2)))
})

test_that("RidgeSolver is deterministic", {
  set.seed(99)
  X <- matrix(rnorm(20 * 4), 20, 4)
  y <- rnorm(20)
  b1 <- RidgeSolver$new()$solve(X, y, lambda = 0.1)
  b2 <- RidgeSolver$new()$solve(X, y, lambda = 0.1)
  expect_equal(b1, b2)
})

test_that("RidgeSolver rejects negative lambda", {
  expect_error(RidgeSolver$new()$solve(matrix(rnorm(20), 5, 4), rnorm(5),
                                        lambda = -1))
})


# =============================================================================
# Group 15: ElasticNet — coordinate-descent path (enet_gaussian)
# Index convention: lambda grid is DESCENDING (index 1 = largest lambda).
# =============================================================================

test_that("ElasticNet path: coef is p x n_lambda, converges, dev_ratio rises", {
  set.seed(1)
  n <- 60; p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] * 2 + X[, 3] - X[, 6] + rnorm(n))
  en <- ElasticNet$new()
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

test_that("ElasticNet predict has shape nrow(X_new) x n_lambda", {
  set.seed(2)
  n <- 40; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] - X[, 2] + rnorm(n))
  en <- ElasticNet$new()$fit(X, y, alpha = 0.5, n_lambda = 30)
  pr <- en$predict(X[1:5, ])
  expect_equal(nrow(pr), 5L)
  expect_equal(ncol(pr), length(en$get_lambdas()))
})

test_that("ElasticNet fit_grid matches auto path at the same lambdas", {
  set.seed(3)
  n <- 50; p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] * 1.5 + rnorm(n))
  auto <- ElasticNet$new()$fit(X, y, alpha = 0.5, n_lambda = 40)
  grid <- ElasticNet$new()$fit_grid(X, y, auto$get_lambdas(), alpha = 0.5)
  expect_equal(grid$get_coef(), auto$get_coef(), tolerance = 1e-9)
})

test_that("ElasticNet ridge limit (alpha=0) leaves all coefficients non-zero", {
  set.seed(4)
  n <- 40; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X %*% rep(1, p) + rnorm(n))
  en <- ElasticNet$new()$fit(X, y, alpha = 0.0, n_lambda = 30)
  last <- en$get_coef()[, ncol(en$get_coef())]
  expect_true(all(abs(last) > 0))
})


# =============================================================================
# Group 16: ElasticNetCV — K-fold CV lambda selection (enet_cv_ccd)
# Index convention: grid DESCENDING, indices 1-based; index_1se <= index_min.
# =============================================================================

test_that("ElasticNetCV: cv_1se >= cv_min and index_1se <= index_min (1-based)", {
  set.seed(5)
  n <- 80; p <- 12
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] * 2 - X[, 4] + rnorm(n))
  cv <- ElasticNetCV$new()
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

test_that("ElasticNetCV is deterministic under a fixed seed", {
  set.seed(6)
  n <- 60; p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- as.numeric(X[, 1] - X[, 3] + rnorm(n))
  a <- ElasticNetCV$new()$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 50, seed = 9)
  b <- ElasticNetCV$new()$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 50, seed = 9)
  expect_equal(a$cv_min(), b$cv_min())
  expect_equal(a$get_cv_errors(), b$get_cv_errors())
})

test_that("ElasticNetCV rejects dimension mismatch and invalid folds", {
  X <- matrix(rnorm(50 * 5), 50, 5)
  expect_error(ElasticNetCV$new()$fit(X, rnorm(49)))
  expect_error(ElasticNetCV$new()$fit(X, rnorm(50), n_folds = 1))
})
