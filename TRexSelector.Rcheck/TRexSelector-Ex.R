pkgname <- "TRexSelector"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('TRexSelector')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("DistanceMetric")
### * DistanceMetric

flush(stderr()); flush(stdout())

### Name: DistanceMetric
### Title: Distance Metrics
### Aliases: DistanceMetric
### Keywords: datasets

### ** Examples

DistanceMetric$Euclidean
DistanceMetric$Correlation



cleanEx()
nameEx("Gauss_data")
### * Gauss_data

flush(stderr()); flush(stdout())

### Name: Gauss_data
### Title: Toy data generated from a Gaussian linear model
### Aliases: Gauss_data
### Keywords: datasets

### ** Examples

# Generated as follows:
set.seed(789)
n <- 50
p <- 100
X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
beta <- c(rep(5, times = 3), rep(0, times = 97))
support <- beta > 0
y <- X %*% beta + stats::rnorm(n)
Gauss_data <- list(
  X = X,
  y = y,
  beta = beta,
  support = support
)



cleanEx()
nameEx("LinkageMethod")
### * LinkageMethod

flush(stderr()); flush(stdout())

### Name: LinkageMethod
### Title: Linkage Methods
### Aliases: LinkageMethod
### Keywords: datasets

### ** Examples

LinkageMethod$Ward
LinkageMethod$Average



cleanEx()
nameEx("LpNormScaler")
### * LpNormScaler

flush(stderr()); flush(stdout())

### Name: LpNormScaler
### Title: Lp-Norm Scaler
### Aliases: LpNormScaler

### ** Examples

set.seed(1)
X <- matrix(rnorm(50 * 5), 50, 5)
scaler <- LpNormScaler$new(norm_type = 2)
scaler$fit(X)
X_norm <- scaler$transform_inplace(X)




cleanEx()
nameEx("RidgeCV")
### * RidgeCV

flush(stderr()); flush(stdout())

### Name: RidgeCV
### Title: Ridge Regression with Cross-Validation
### Aliases: RidgeCV

### ** Examples

set.seed(1)
X <- matrix(rnorm(50 * 5), 50, 5)
y <- X[, 1] * 2 + rnorm(50)
cv <- RidgeCV$new()
cv$fit(X, y, num_folds = 5)
cv$cv_min()




cleanEx()
nameEx("RidgeGCV")
### * RidgeGCV

flush(stderr()); flush(stdout())

### Name: RidgeGCV
### Title: Ridge Regression with GCV
### Aliases: RidgeGCV

### ** Examples

set.seed(1)
X <- matrix(rnorm(50 * 5), 50, 5)
y <- X[, 1] * 2 + rnorm(50)
gcv <- RidgeGCV$new()
gcv$fit(X, y)
lam_opt <- gcv$gcv_optimal()
gcv$solve(lam_opt)




cleanEx()
nameEx("TACGPSolver")
### * TACGPSolver

flush(stderr()); flush(stdout())

### Name: TACGPSolver
### Title: T-ACGP Solver
### Aliases: TACGPSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TACGPSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TAFSSolver")
### * TAFSSolver

flush(stderr()); flush(stdout())

### Name: TAFSSolver
### Title: T-AFS Solver
### Aliases: TAFSSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TAFSSolver$new(X, D, y, rho = 0.5)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TENETSolver")
### * TENETSolver

flush(stderr()); flush(stdout())

### Name: TENETSolver
### Title: T-ENET Solver
### Aliases: TENETSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TENETSolver$new(X, D, y, lambda2 = 0.1)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TGPSolver")
### * TGPSolver

flush(stderr()); flush(stdout())

### Name: TGPSolver
### Title: T-GP Solver
### Aliases: TGPSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TGPSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TLARSSolver")
### * TLARSSolver

flush(stderr()); flush(stdout())

### Name: TLARSSolver
### Title: T-LARS Solver
### Aliases: TLARSSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TLARSSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()
solver$get_actives()




cleanEx()
nameEx("TLASSOSolver")
### * TLASSOSolver

flush(stderr()); flush(stdout())

### Name: TLASSOSolver
### Title: T-LASSO Solver
### Aliases: TLASSOSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TLASSOSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TMPSolver")
### * TMPSolver

flush(stderr()); flush(stdout())

### Name: TMPSolver
### Title: T-MP Solver
### Aliases: TMPSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TMPSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TNCGMPSolver")
### * TNCGMPSolver

flush(stderr()); flush(stdout())

### Name: TNCGMPSolver
### Title: T-NCGMP Solver
### Aliases: TNCGMPSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TNCGMPSolver$new(X, D, y, variant = "line_search")
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TOMPSolver")
### * TOMPSolver

flush(stderr()); flush(stdout())

### Name: TOMPSolver
### Title: T-OMP Solver
### Aliases: TOMPSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TOMPSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TOOLSSolver")
### * TOOLSSolver

flush(stderr()); flush(stdout())

### Name: TOOLSSolver
### Title: T-OLS Solver
### Aliases: TOOLSSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TOOLSSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TRexBiobankScreeningSelector")
### * TRexBiobankScreeningSelector

flush(stderr()); flush(stdout())

### Name: TRexBiobankScreeningSelector
### Title: Biobank Screen T-Rex Selector
### Aliases: TRexBiobankScreeningSelector

### ** Examples





cleanEx()
nameEx("TRexDASelector")
### * TRexDASelector

flush(stderr()); flush(stdout())

### Name: TRexDASelector
### Title: Dependency-Aware T-Rex Selector (TRexDA)
### Aliases: TRexDASelector

### ** Examples





cleanEx()
nameEx("TRexGVSSelector")
### * TRexGVSSelector

flush(stderr()); flush(stdout())

### Name: TRexGVSSelector
### Title: T-Rex Selector with Group Variable Selection (TRex GVS)
### Aliases: TRexGVSSelector

### ** Examples




cleanEx()
nameEx("TRexScreeningSelector")
### * TRexScreeningSelector

flush(stderr()); flush(stdout())

### Name: TRexScreeningSelector
### Title: Screen T-Rex Selector
### Aliases: TRexScreeningSelector

### ** Examples





cleanEx()
nameEx("TRexSelector")
### * TRexSelector

flush(stderr()); flush(stdout())

### Name: TRexSelector
### Title: T-Rex Selector
### Aliases: TRexSelector

### ** Examples




cleanEx()
nameEx("TSTEPWISESolver")
### * TSTEPWISESolver

flush(stderr()); flush(stdout())

### Name: TSTEPWISESolver
### Title: T-Stepwise Solver
### Aliases: TSTEPWISESolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TSTEPWISESolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("TSolverBase")
### * TSolverBase

flush(stderr()); flush(stdout())

### Name: TSolverBase
### Title: Base T-Solver Class
### Aliases: TSolverBase

### ** Examples

# TSolverBase is abstract; use a concrete subclass instead, e.g.:
# solver <- TLARSSolver$new(X, D, y)



cleanEx()
nameEx("TStagewiseSolver")
### * TStagewiseSolver

flush(stderr()); flush(stdout())

### Name: TStagewiseSolver
### Title: T-Stagewise Solver
### Aliases: TStagewiseSolver

### ** Examples

set.seed(1)
n <- 20; p <- 5
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * 2 + rnorm(n)
solver <- TStagewiseSolver$new(X, D, y)
solver$execute_step(T_stop = 3)
solver$get_beta()




cleanEx()
nameEx("ZScoreScaler")
### * ZScoreScaler

flush(stderr()); flush(stdout())

### Name: ZScoreScaler
### Title: Z-Score Scaler
### Aliases: ZScoreScaler

### ** Examples

set.seed(1)
X <- matrix(rnorm(50 * 5), 50, 5)
scaler <- ZScoreScaler$new()
scaler$fit(X)
X_std <- scaler$transform_inplace(X)
scaler$get_means()




cleanEx()
nameEx("agglomerative_cluster")
### * agglomerative_cluster

flush(stderr()); flush(stdout())

### Name: agglomerative_cluster
### Title: Agglomerative Clustering
### Aliases: agglomerative_cluster

### ** Examples

set.seed(1)
data <- matrix(rnorm(30), nrow = 10)
agglomerative_cluster(data, method = LinkageMethod$Ward)



cleanEx()
nameEx("as.matrix.mmap_matrix")
### * as.matrix.mmap_matrix

flush(stderr()); flush(stdout())

### Name: as.matrix.mmap_matrix
### Title: Get Zero-Copy View as Numeric Vector (Block)
### Aliases: as.matrix.mmap_matrix

### ** Examples





cleanEx()
nameEx("compute_fdp")
### * compute_fdp

flush(stderr()); flush(stdout())

### Name: compute_fdp
### Title: Compute False Discovery Proportion (FDP) using index sets
### Aliases: compute_fdp

### ** Examples

compute_fdp(c(1L, 3L, 5L), c(1L, 2L, 3L))




cleanEx()
nameEx("compute_fdp_dense")
### * compute_fdp_dense

flush(stderr()); flush(stdout())

### Name: compute_fdp_dense
### Title: Compute False Discovery Proportion (FDP) using dense vectors
### Aliases: compute_fdp_dense

### ** Examples

compute_fdp_dense(c(1.5, 0, 0.8, 0, 0), c(2.0, 0, 1.0, 0, 0))



cleanEx()
nameEx("compute_precision")
### * compute_precision

flush(stderr()); flush(stdout())

### Name: compute_precision
### Title: Compute Precision using index sets
### Aliases: compute_precision

### ** Examples

compute_precision(c(1L, 3L, 5L), c(1L, 2L, 3L))




cleanEx()
nameEx("compute_recall")
### * compute_recall

flush(stderr()); flush(stdout())

### Name: compute_recall
### Title: Compute Recall using index sets
### Aliases: compute_recall

### ** Examples

compute_recall(c(1L, 3L, 5L), c(1L, 2L, 3L))




cleanEx()
nameEx("compute_tpp")
### * compute_tpp

flush(stderr()); flush(stdout())

### Name: compute_tpp
### Title: Compute True Positive Proportion (TPP) using index sets
### Aliases: compute_tpp

### ** Examples

compute_tpp(c(1L, 3L, 5L), c(1L, 2L, 3L))




cleanEx()
nameEx("compute_tpp_dense")
### * compute_tpp_dense

flush(stderr()); flush(stdout())

### Name: compute_tpp_dense
### Title: Compute True Positive Proportion (TPP) using dense vectors
### Aliases: compute_tpp_dense

### ** Examples

compute_tpp_dense(c(1.5, 0, 0.8, 0, 0), c(2.0, 0, 1.0, 0, 0))




cleanEx()
nameEx("convert_to_memory_mapped")
### * convert_to_memory_mapped

flush(stderr()); flush(stdout())

### Name: convert_to_memory_mapped
### Title: Convert R Matrix to System Memory Mapped Matrix
### Aliases: convert_to_memory_mapped

### ** Examples





cleanEx()
nameEx("cut_tree")
### * cut_tree

flush(stderr()); flush(stdout())

### Name: cut_tree
### Title: Cut Hierarchical Linkage Matrix
### Aliases: cut_tree

### ** Examples

set.seed(1)
data <- matrix(rnorm(30), nrow = 10)
linkage <- agglomerative_cluster(data, method = LinkageMethod$Ward)
cut_tree(linkage, num_orig_objs = 10, num_clusters = 3)



cleanEx()
nameEx("dim.mmap_matrix")
### * dim.mmap_matrix

flush(stderr()); flush(stdout())

### Name: dim.mmap_matrix
### Title: Get mmap_matrix dimensions
### Aliases: dim.mmap_matrix

### ** Examples




cleanEx()
nameEx("get_ptr")
### * get_ptr

flush(stderr()); flush(stdout())

### Name: get_ptr
### Title: Get the External Pointer of a MemoryMappedMatrix
### Aliases: get_ptr

### ** Examples





cleanEx()
nameEx("mmap_matrix")
### * mmap_matrix

flush(stderr()); flush(stdout())

### Name: mmap_matrix
### Title: Memory-Mapped Matrix Object
### Aliases: mmap_matrix

### ** Examples





cleanEx()
nameEx("print.mmap_matrix")
### * print.mmap_matrix

flush(stderr()); flush(stdout())

### Name: print.mmap_matrix
### Title: Print mmap_matrix
### Aliases: print.mmap_matrix

### ** Examples





cleanEx()
nameEx("sub-.mmap_matrix")
### * sub-.mmap_matrix

flush(stderr()); flush(stdout())

### Name: [.mmap_matrix
### Title: Extract block from mmap_matrix
### Aliases: [.mmap_matrix

### ** Examples




cleanEx()
nameEx("trex_biobank_control")
### * trex_biobank_control

flush(stderr()); flush(stdout())

### Name: trex_biobank_control
### Title: TRex Biobank Control Parameters
### Aliases: trex_biobank_control

### ** Examples

trex_biobank_control()
trex_biobank_control(lower_bound_FDR = 0.02, upper_bound_FDR = 0.10)




cleanEx()
nameEx("trex_control")
### * trex_control

flush(stderr()); flush(stdout())

### Name: trex_control
### Title: TRex Control Parameters
### Aliases: trex_control

### ** Examples

trex_control()
trex_control(method = "TLASSO", K = 10)




cleanEx()
nameEx("trex_da_control")
### * trex_da_control

flush(stderr()); flush(stdout())

### Name: trex_da_control
### Title: TRex DA Control Parameters
### Aliases: trex_da_control

### ** Examples

trex_da_control()
trex_da_control(da_method = "BT", hc_linkage = "Average")




cleanEx()
nameEx("trex_dummy_distribution")
### * trex_dummy_distribution

flush(stderr()); flush(stdout())

### Name: trex_dummy_distribution
### Title: TRex Dummy Variable Distribution
### Aliases: trex_dummy_distribution

### ** Examples

# Standard Normal (default)
trex_dummy_distribution()

# Student-t with 3 degrees of freedom
trex_dummy_distribution("StudentT", df = 3)




cleanEx()
nameEx("trex_gvs_control")
### * trex_gvs_control

flush(stderr()); flush(stdout())

### Name: trex_gvs_control
### Title: TRex GVS Control Parameters
### Aliases: trex_gvs_control

### ** Examples

trex_gvs_control()
trex_gvs_control(gvs_type = "IEN", corr_max = 0.3)




cleanEx()
nameEx("trex_screen_control")
### * trex_screen_control

flush(stderr()); flush(stdout())

### Name: trex_screen_control
### Title: TRex Screen Control Parameters
### Aliases: trex_screen_control

### ** Examples

trex_screen_control()
trex_screen_control(use_bootstrap_CI = TRUE, R_boot = 500)




### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
