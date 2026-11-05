#!/usr/bin/env Rscript
# =============================================================================
# zero_copy_reference.R — bit-identity harness of the R ML surface.
#
#   Rscript tools/zero_copy_reference.R record  <dir>   # on the current tree
#   Rscript tools/zero_copy_reference.R compare <dir>   # after a change
#
# Every ML class is run on fixed synthetic data of three shapes and every
# output is saved with saveRDS; compare uses identical() on the whole list,
# so a change at the binding boundary or a copy fusion must
# reproduce the recording exactly. Run with R_LIBS pointing at the library
# under test. The reference is platform-specific; keep it out of git.
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L || !args[1] %in% c("record", "compare")) {
  stop("usage: zero_copy_reference.R record|compare <dir>")
}
mode <- args[1]; dir <- args[2]
suppressPackageStartupMessages(library(TRexSelector))

shapes <- list(tall = c(400L, 30L), wide = c(60L, 900L), square = c(120L, 120L))

make_data <- function(n, p) {
  set.seed(1000L + p)
  X <- matrix(rnorm(n * p), n, p)
  beta <- numeric(p); beta[1] <- 2; beta[4] <- -1.5; beta[min(8L, p)] <- 1
  set.seed(77L + n)
  y <- as.vector(X %*% beta + 0.5 * rnorm(n))
  list(X = X, y = y)
}

run_case <- function(name, n, p) {
  d <- make_data(n, p); X <- d$X; y <- d$y; X0 <- X + 0
  out <- list()

  cv <- MLRidgeCV$new()
  cv$fit(X, y, num_folds = 5, n_lambda = 60, lambda_ratio = 1000, seed = 11)
  out$ridge_cv <- list(lambdas = cv$get_lambdas(), mse = cv$get_cv_errors(),
                       sem = cv$get_cv_std(), min = cv$cv_min(), se = cv$cv_1se(),
                       imin = cv$index_min(), i1se = cv$index_1se())

  en <- MLElasticNet$new()
  en$fit(X, y, alpha = 0.7, n_lambda = 40)
  out$enet <- list(lambdas = en$get_lambdas(), coef = en$get_coef(),
                   a0 = en$get_intercepts(), dev = en$get_dev_ratio(),
                   conv = en$converged(), pred = en$predict(X[1:7, , drop = FALSE]))
  grid <- en$get_lambdas()[6:15]
  en2 <- MLElasticNet$new()
  en2$fit_grid(X, y, grid, alpha = 0.7, standardize = FALSE, use_strong_rule = TRUE)
  out$enet_grid <- list(lambdas = en2$get_lambdas(), coef = en2$get_coef(),
                        a0 = en2$get_intercepts())

  ecv <- MLElasticNetCV$new()
  ecv$fit(X, y, alpha = 0.5, n_folds = 5, n_lambda = 40, seed = 3)
  out$enet_cv <- list(lambdas = ecv$get_lambdas(), mse = ecv$get_cv_errors(),
                      sem = ecv$get_cv_std(), min = ecv$cv_min(), se = ecv$cv_1se(),
                      imin = ecv$index_min(), i1se = ecv$index_1se())

  out$svd <- MLSVD$new()$compute(X, M = 5)

  pca <- MLPCA$new(X, M = 4, center = TRUE, normalize = TRUE)
  out$pca_pre_X <- X + 0
  pca$fit()
  out$pca <- list(Z = pca$get_scores(), V = pca$get_loadings(),
                  ev = pca$get_explained_variance(), means = pca$get_means(),
                  norms = pca$get_norms(), tr = pca$transform(X0[1:6, , drop = FALSE]))
  pca$restore()
  out$pca_restored_X <- X + 0

  out$ridge <- lapply(c(1e-2, 1, 1e2), function(l) MLRidge$new()$solve(X, y, lambda = l))

  Xs <- X + 0; Xs[, 2] <- 3
  z <- MLZScoreScaler$new(center = TRUE, scale = TRUE)
  z$fit_transform_inplace(Xs)
  out$zscore <- list(X = Xs + 0, centers = z$get_centers(), scales = z$get_scales(),
                     dropped = z$get_dropped_indices())
  z$inverse_transform_inplace(Xs)
  out$zscore_back <- Xs + 0
  l1 <- MLLpNormScaler$new(norm_type = 1, center = FALSE, scale = TRUE)
  l1$fit_transform_inplace(Xs)
  out$lp <- list(X = Xs + 0, scales = l1$get_scales())
  l1$inverse_transform_inplace(Xs)
  out$lp_back <- Xs + 0

  # Clustering of the rows (the R surface clusters observations)
  Xc <- X[1:min(n, 60L), , drop = FALSE]
  out$hac_average <- ml_agglomerative_cluster(Xc, method = ml_linkage_method$Average,
                                              metric = ml_distance_metric$Euclidean)
  out$hac_ward <- ml_agglomerative_cluster(Xc, method = ml_linkage_method$Ward,
                                           metric = ml_distance_metric$Euclidean)
  out$hac_cut <- ml_cut_tree(out$hac_ward, nrow(Xc), 3L)

  # The input matrix must be back to its original values at the end
  out$X_final <- X + 0
  out$X_untouched <- identical(X, X0)
  out
}

results <- lapply(names(shapes), function(nm) run_case(nm, shapes[[nm]][1], shapes[[nm]][2]))
names(results) <- names(shapes)

dir.create(dir, showWarnings = FALSE, recursive = TRUE)
file <- file.path(dir, "r_ml_reference.rds")
if (mode == "record") {
  saveRDS(results, file)
  cat("recorded", file, "\n")
} else {
  ref <- readRDS(file)
  ok <- TRUE
  for (nm in names(results)) {
    for (k in names(results[[nm]])) {
      if (!identical(results[[nm]][[k]], ref[[nm]][[k]])) {
        ok <- FALSE
        cat("DIFFERS:", nm, k, "\n")
      }
    }
  }
  if (!ok) { cat("R ML reference: DIFFERENCES\n"); quit(status = 1L) }
  cat("R ML reference: identical\n")
}
