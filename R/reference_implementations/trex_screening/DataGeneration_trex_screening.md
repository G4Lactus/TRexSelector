# Data Generating Processes to be tested within the Screen-TRex framework

The setup assumes $n$ observations and $p$ features, so the design matrix
$\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_p]$
$\in \mathbb{R}^{n \times p}$, where each column $\boldsymbol{x}_j$ is a random vector in
$\mathbb{R}^n$.
To generate design matrices with specific correlation structures efficiently, we can define the
data generating process (DGP) directly via independent standard normal vectors.
By using major column notation—where the design matrix $X \in \mathbb{R}^{n \times p}$ is
partitioned into its $p$ column vectors $X = [x_1, x_2, \dots, x_p]$, the computational cost of
$\mathcal{O}(p^3)$ of a full covariance matrix factorization is avoided.

---

## Linear Model in the Orthogonal/Uncorrelated Design

The classical linear model generates
$$
\boldsymbol{y} = \boldsymbol{X} \, \boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$
with $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^{2} \boldsymbol{I})$ and
$\sigma = \sqrt{\lVert \boldsymbol{X} \boldsymbol{\beta} \rVert_2^2 / (n \cdot \text{SNR})}$.

---

## Auto-Regressive Process of Order 1 - AR(1) Design

The generative model of the AR(1) process in column recursion matching allows us to formulate the
data generating process in major column format:

With $\boldsymbol{x}_1 \sim \mathcal{N}(0, \boldsymbol{I}_n)$ and $\boldsymbol{x}_1 =
\boldsymbol{z}_1$
$$
\boldsymbol{x}_j = \rho \, \boldsymbol{x}_{j-1} + \sqrt{1 - \rho^2} \, \boldsymbol{z}_j
$$
for $j = 2, \dots, p$, without forming the full $p \times p$ Toeplitz matrix, which is important
for large $p$.
The underlying latent variables $\boldsymbol{z}_1, \boldsymbol{z}_2, \ldots, \boldsymbol{z}_p$ are
all mutually independent standard normal vectors, thus
$\boldsymbol{z}_k \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}_n)$,
$\boldsymbol{z}_{k} \in \mathbb{R}^{n}$.

**Expectation:**
Because all $\boldsymbol{z}_k$ have an expectation of $\mathbf{0}$, the expectation of any column
$\boldsymbol{x}_j$ evaluates to a zero vector:
$$
\mathbb{E}[\boldsymbol{x}_j] = \mathbf{0}.
$$

**Variance / Covariance Matrix:**
The variance of a single observation within any column $j$ is strictly 1.
We can see this through induction: if $\text{Var}(\boldsymbol{x}_{j-1}) = \boldsymbol{I}_n$, then
$\text{Var}(\boldsymbol{x}_j) = \rho^2 \text{Var}(\boldsymbol{x}_{j-1}) +
(1 - \rho^2) \text{Var}(\boldsymbol{z}_j) = \rho^2 \boldsymbol{I}_n +
(1 - \rho^2) \boldsymbol{I}_n = \boldsymbol{I}_n$.

To find the cross-covariance between two different columns $j$ and $k$ (assuming $k > j$), we
recursively expand $x_k$ backwards to $\boldsymbol{x}_j$:

$$
\text{Cov}(\boldsymbol{x}_j, \boldsymbol{x}_k) =
\mathbb{E}[\boldsymbol{x}_j \boldsymbol{x}_k^\top] =
\mathbb{E}\left[\boldsymbol{x}_j \left(\rho^{k-j} \boldsymbol{x}_j +
 \sum_{m=j + 1}^k \rho^{k - m} \sqrt{1 - \rho^2} \boldsymbol{z}_m\right)^\top\right]
$$

Because $\boldsymbol{x}_j$ is entirely independent of future noise vectors $\boldsymbol{z}_m$,
all cross-terms vanish to zero.
The covariance evaluates simply to the lagged term:

$$
\text{Cov}(\boldsymbol{x}_j, \boldsymbol{x}_k) =
\rho^{|j-k|} \mathbb{E}[\boldsymbol{x}_j \boldsymbol{x}_j^\top] =
\rho^{|j-k|} \boldsymbol{I}_n
$$

Overall, if we look at the $p \times p$ covariance matrix $\boldsymbol{\Sigma}$ for a single row
observation (a $1 \times p$ vector), its entries are defined as
$\boldsymbol{\Sigma}_{j,k} = \rho^{|j-k|}$.

---

## Equi-Correlated Design (Compound Symmetry)

To generate an equi-correlated design process, we use again column recursion matching, which
allows again to formulate the data generating process in major column format:

$$
\boldsymbol{x}_j = \sqrt{\rho} \, \boldsymbol{z}_0 + \sqrt{1 - \rho} \, \boldsymbol{z}_j,
$$
for $j = 1, \dots, p$ with  $\boldsymbol{z}_j \overset{\text{iid}}{\sim} \mathcal{N}(0,I_n)$,
which gives exactly $\mathrm{Cor}(\boldsymbol{x}_j, \boldsymbol{x}_k) = \rho$ for all $j \neq k$.

**Expectation:**
The expectation is obtained by taking advantage of the linearity property of the expectation value
$$
\mathbb{E}[\boldsymbol{x}_j] =
\sqrt{\rho} \, \mathbb{E}[\boldsymbol{z}_0] + \sqrt{1 - \rho} \, \mathbb{E}[\boldsymbol{z}_j] =
\mathbf{0}.
$$

**Variance / Covariance Matrix:**
The variance of any column $\boldsymbol{x}_j$ calculates perfectly to $\boldsymbol{I}_n$ because
the shared factor $\boldsymbol{z}_0$ and the unique factor $\boldsymbol{z}_j$ are independent:
$$
\text{Var}(\boldsymbol{x}_j) =
\text{Cov}(\boldsymbol{x}_{j}, \boldsymbol{x}_{j}) =
\mathbb{E}[\boldsymbol{x}_j \boldsymbol{x}_j^\top] =
\rho \, \mathbb{E}[\boldsymbol{z}_0 \boldsymbol{z}_{0}^\top] +
(1 - \rho) \, \mathbb{E}[\boldsymbol{z}_j \boldsymbol{z}_{j}^\top] =
\rho \, \boldsymbol{I}_n + (1 - \rho) \boldsymbol{I}_n =
\boldsymbol{I}_n.
$$

For the cross-covariance between any two distinct columns $x_j$ and $x_k$ (where $j \neq k$),
the unique noise vectors $z_j$ and $z_k$ are completely independent, so their cross-products vanish.
The covariance relies entirely on the shared factor $z_0$:

$$
\text{Cov}(\boldsymbol{x}_j, \boldsymbol{x}_k) =
\mathbb{E}[\boldsymbol{x}_j \boldsymbol{x}_j^\top] =
\mathbb{E}[(\sqrt{\rho} \, \boldsymbol{z}_0 + \sqrt{1 - \rho} \, \boldsymbol{z}_j)
(\sqrt{\rho} \, \boldsymbol{z}_0 + \sqrt{1 - \rho} \, \boldsymbol{z}_k)^\top ] =
 \rho \, \mathbb{E}[\boldsymbol{z}_0 \boldsymbol{z}_0^\top] =
\rho \, \boldsymbol{I}_n.
$$

The resulting covariance matrix $\Sigma \in \mathbb{R}^{p \times p}$ is exactly the compound
symmetry matrix:
$$
\Sigma = (1-\rho)I_p + \rho J_p
$$
where $J_p$ is a matrix of ones.

In high-dimensional settings, an equi-correlated design (also known as compound symmetry) with a
large $\rho$ is structurally extreme and generally unrealistic for purely observational data.
It acts less as a model of natural phenomena and more as a mathematical boundary condition.

The severe multicollinearity is a direct consequence of the covariance matrix's eigenspectrum.
This matrix has exactly two distinct eigenvalues:

1. A single dominant eigenvalue: $\lambda_1 = 1 + (p - 1) \, \rho$.
2. A repeating eigenvalue for the remaining dimensions: $\lambda_2 = \dots = \lambda_p = 1 - \rho$.

The condition number of the design matrix scales as
$$
\kappa = \frac{1 + (p-1)\rho}{1 - \rho}.
$$

As $\rho$ grows or $p$ increases, the entire feature space rapidly collapses into a single
one-dimensional latent factor (the shared $z_0$ from the data generating process), causing
standard matrix inversions to fail and variance inflation factors (VIFs) to skyrocket.

While unrealistic for flat, high-dimensional feature matrices, an equi-correlated structure
naturally arises in specific hierarchical or grouped data settings:

- **Mixed-Effects and Panel Data:** In longitudinal studies, this design models "compound symmetry".
It perfectly represents a shared, time-invariant random intercept. For example, if you measure the
same subject multiple times, a shared underlying genetic or behavioral trait affects all
measurements equally, regardless of the time lag between them.
- **Clustered Macro-Shocks:** In econometrics or sensor networks, if multiple units (e.g., sensors
in a star-topology, or students in a single classroom) are exposed to the exact same environmental
factor or identical noise source, they will exhibit uniform correlation.

---

## Block Equi-Correlated Data Generating Process (Block Compound Symmetry)

While a fully equi-correlated design is mathematically convenient for deriving theoretical bounds
and stress-testing algorithms, it is an extreme artifact.
In reality, data naturally clumps into clusters. Examples are found in genetics in the same
regulatory pathway, or in sensor networks at the same physical location, or assets in the same
market sector.
This DGP is exactly what algorithms like the Group Lasso, Elastic Net, and T-Rex selector are meant
to solve.

Assuming the $p$ variables of the design matrix $\boldsymbol{X}$ are partitioned into $K$ disjoint
groups (or blocks) $G_1, G_2, \dots, G_K$.
Let $p_k = |G_k|$ be the number of variables in the $k$-th block, such that $\sum_{k=1}^K p_k = p$.

Instead of a single global latent factor $\boldsymbol{z}_0$, there are $K$ block-specific latent
factors $\boldsymbol{z}_{01}, \boldsymbol{z}_{02}, \dots, \boldsymbol{z}_{0K}$, where each
$\boldsymbol{z}_{0k} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}_n)$.
Feature-specific mutually independent noise vectors are
$\boldsymbol{z}_1, \dots, \boldsymbol{z}_p \sim  \mathcal{N}(\mathbf{0}, \boldsymbol{I}_n)$.
For each block $k$, a block-specific intra-class correlation parameter $\rho_k \in [0, 1)$ is set.
For a variable index $j$ that belongs to block $G_k$, the $j$-th column of the design matrix is
generated as:
$$
\boldsymbol{x}_j = \sqrt{\rho_k} \, \boldsymbol{z}_{0k} +
\sqrt{1 - \rho_k} \, \boldsymbol{z}_j
$$
for $j \in G_k$.

Because the shared latent factor $z_{0k}$ is restricted to its specific block,
the expectation remains $\mathbb{E}[x_j] = \mathbf{0}$,
and the variance remains $\text{Var}(x_j) = I_n$.

The covariance structure $\Sigma$ now becomes a block-diagonal matrix:

1. **Within-Block Covariance:** If $x_j$ and $x_m$ belong to the *same* block $G_k$ ($j \neq m$),
   they share $z_{0k}$, leading to an equi-correlated relationship:

$$
\text{Cov}(\boldsymbol{x}_j, \boldsymbol{x}_m) = \rho_k \boldsymbol{I}_n
$$

1. **Between-Block Covariance:** If $x_j \in G_k$ and $x_m \in G_q$ belong to *different* blocks
($k \neq q$), they depend on disjoint latent factors ($z_{0k}$ and $z_{0q}$), which are independent:

$$
\text{Cov}(\boldsymbol{x}_j, \boldsymbol{x}_m) = \boldsymbol{0}
$$

*(Note: To model a weaker "global" background correlation between the blocks, you can easily extend
 this by adding a global latent factor $z_{global}$ scaled by an off-diagonal correlation
 $\rho_{off}$, a structure formally known as general block compound symmetry.)*

---

## R Functions

The following functions support the demo of the data generating processes.

| Function | Description |
| -------- | ----------- |
| `dgp_ar1(n, p, p1, rho, snr, beta_val, seed)` | Generates a predictor matrix with AR(1) covariance design. |
| `dgp_equi(n, p, p1, rho, snr, beta_val, seed)` | Generates a predictor matrix with Equi-Correlated covariance design. |
| `.make_beta(p, p1, beta_val)` | Sparse $\beta$ with `p1` active variables at evenly spaced indices — spacing avoids all actives clustering inside one correlated neighbourhood. |
| `.make_y(X, beta, snr)` | Generates $y = X\beta + \varepsilon$ with $\sigma = \sqrt{\lVert X\beta\rVert_2^2 / (n\cdot \text{SNR})}$ according to the SNR definition. |
| `dgp_diagnostics(dgp, type, max_col)` | Sanity-check utility — computes empirical lag-1/lag-2 correlations (AR1) and mean off-diagonal correlation (equi) against their theoretical values, so you can verify a generated data set before handing it to `screen_trex()`. |

---

### Intended Workflow

```R
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
dgp_diagnostics(d_AR1, type = "ar1")
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

```
