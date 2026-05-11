setwd("./R/references/trex_screening")

source("dgp_correlated.R")
source("trex_screening.R")
library(plotly)

plot_cormat <- function(cor_matrix) {
  plot_ly(
    x = colnames(cor_matrix),
    y = rownames(cor_matrix),
    z = cor_matrix,
    type = "heatmap",
    colorscale = "RdBu", # Good for correlations (-1 to 1)
    zmin = -1,
    zmax = 1
  )
}

# AR(1) test
d_AR1   <- dgp_ar1(n = 300, p = 1000, p1 = 10, rho = 0.5, snr = 5, seed = 1)
dgp_diagnostics(d_AR1, type = "ar1")          # verify correlation structure
cor_mat_d_AR1 <- cor(d_AR1$X)
plot_cormat(cor_mat_d_AR1)
res <- screen_trex(X = d_AR1$X, y = d_AR1$y, method = "trex+DA+AR1")

# Equi-Correlated test
d_equi   <- dgp_equi(n = 300, p = 1000, p1 = 10, rho = 0.6, snr = 5, seed = 1)
dgp_diagnostics(d_equi, type = "equi")
cor_mat_d_equi <- cor(d_equi$X)
plot_cormat(cor_mat_d_equi)
res <- screen_trex(X = d_equi$X, y = d_equi$y, method = "trex+DA+equi")

# Block Equi-Correlated test
d_block_equi   <- dgp_block_equi(n = 300, p = 1000, p1 = 10, rho = 0.7,
                                 n_blocks = 5, snr = 5, betaval = 3, seed = 1)
dgp_diagnostics(d_block_equi, type = "block_equi")
cor_mat_d_block_equi <- cor(d_block_equi$X)
plot_cormat(cor_mat_d_block_equi)
