<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# At next we create a file that acts as data generating process for the AR1 process and the equi-correlated data type. I want to test if the algorithmic modifications can handle it.

`dgp_correlated.R`  — 336 lines, three exported functions.

## What is in the file

### Private helpers (shared by both DGPs)

| Helper | Purpose |
| :-- | :-- |
| `.make_beta(p, p1, beta_val)` | Sparse $\beta$ with `p1` active variables at evenly spaced indices — spacing avoids all actives clustering inside one correlated neighbourhood |
| `.make_y(X, beta, snr)` | Generates $y = X\beta + \varepsilon$ with $\sigma = \sqrt{\lVert X\beta\rVert_2^2 / (n \cdot \text{SNR})}$ matching the paper's SNR definition |

### `dgp_ar1(n, p, p1, rho, snr, beta_val, seed)`

Uses the column recursion matching the AR(1) model from `screen_trex_math.R`:

$$
x_1 \sim \mathcal{N}(0, I_n), \quad x_j = \rho\, x_{j-1} + \sqrt{1-\rho^2}\, w_j
$$

This exactly reproduces $\mathrm{Cor}(x_j, x_k) = \rho^{|j-k|}$ without forming the full $p \times p$ Toeplitz matrix, which is important for large $p$.

### `dgp_equi(n, p, p1, rho, snr, beta_val, seed)`

Uses the one-factor model from `screen_trex_math.R`:

$$
x_j = \sqrt{\rho}\, z + \sqrt{1-\rho}\, w_j, \quad z,\,w_j \overset{\text{iid}}{\sim} \mathcal{N}(0,I_n)
$$

giving exactly $\mathrm{Cor}(x_j, x_k) = \rho$ for all $j \neq k$.

### `dgp_diagnostics(dgp, type, max_col)`

Sanity-check utility — computes empirical lag-1/lag-2 correlations (AR1) and mean off-diagonal correlation (equi) against their theoretical values, so you can verify a generated data set before handing it to `screen_trex()`.

### Intended test workflow

```r
source("dgp_correlated.R")
source("trex_screening.R")

# AR(1) test
d   <- dgp_ar1(n = 300, p = 1000, p1 = 10, rho = 0.7, snr = 5, seed = 1)
dgp_diagnostics(d, type = "ar1")          # verify correlation structure
res <- screen_trex(X = d$X, y = d$y, method = "trex+DA+AR1")

# Equi test
d   <- dgp_equi(n = 300, p = 1000, p1 = 10, rho = 0.7, snr = 5, seed = 1)
dgp_diagnostics(d, type = "equi")
res <- screen_trex(X = d$X, y = d$y, method = "trex+DA+equi")
```

