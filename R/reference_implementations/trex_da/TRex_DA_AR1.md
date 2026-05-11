# T-Rex DA

## AR(1) Covariance Structure

For a stationary AR(1) process with auto-correlation factor $0 < \rho < 1$ with postive spatial
correlation the correlation between variables $j$ and $j'$ is purely a function of lag:

$$
\rho_{j,j'} = \rho^{|j - j'|}, \quad 0 < \rho < 1.
$$

The full covariance is a Toeplitz matrix with no block separation.
**Every pair is correlated**, with correlation decaying geometrically with distance.

## Closed-Form Group Definition and Verification

Since the correlation structure is known analytically, the dendrogram step becomes unnecessary.
Variable $j'$ belongs to the group of variable $j$ if and only if:

$$
\rho^{|j - j'|} \geq \rho_{\text{thr}} \iff |j - j'| \leq r(\rho_{\text{thr}}),
\quad \text{where }
r(\rho_{\text{thr}}) = \left\lfloor \frac{\log \rho_{\text{thr}}}{\log \rho} \right\rfloor.
$$
When $\rho_{\text{thr}} = 1$ then $\log{\rho_{\text{thr}}} = 0$, so $r = 0$ and
$\text{Gr}=\emptyset$ recovers the ordinary T-Rex. When $\rho_{\text{thr}} = \rho^{r}$, i.e., the
threshold sits exactly on a lag boundary, the floor includes that lag.

So the group is an explicit **sliding window of radius** $r$ centered at $j$:

$$
\text{Gr}(j,\, \rho_{\text{thr}}) =
\bigl\{j' \in \{1,\ldots,p\}\setminus\{j\} : |j - j'| \leq r(\rho_{\text{thr}})\bigr\}.
$$

### Verifying Theorem 3 Still Holds

The group design principle requires
$\rho_2 > \rho_1 \Rightarrow \text{Gr}(j,\rho_2) \subseteq \text{Gr}(j,\rho_1)$.
Since $\log\rho < 0$ for $0 < \rho < 1$:

$$
\rho_2 > \rho_1 \;\Rightarrow\; \frac{\log\rho_2}{\log\rho} <
\frac{\log\rho_1}{\log\rho} \;\Rightarrow\; r(\rho_2) \leq r(\rho_1)
$$

A smaller window is always a subset of a larger one, so **the subset condition holds analytically**
— no clustering computation needed. ✓

### Estimation Goals

From the data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$, we need:

1. The AR(1) parameter $\hat{\rho}$ — to know the decay rate
2. The effective lag radius $\hat{r}$ — treated as the calibration parameter $\rho_{\text{thr}}$
   swept in Algorithm 1.

These are not independent: once $\hat{\rho}$ is fixed, $r$ and $\rho_{\text{thr}}$ are
interchangeable via $\rho_{\text{thr}} = \hat{\rho}^r$.

#### Estimation of $\hat{\rho}$

The sample correlation matrix $\hat{\mathbf{R}}$ has entries
$\hat{\rho}_{j,j'} = \hat{\mathbf{x}}_j^\top \hat{\mathbf{x}}_{j'}$,
where the columns of $\mathbf{X}$ are standardized to zero mean and unit variance
as required by the T-Rex framework.
Under AR(1), each off-diagonal band $d$ gives $(p-d)$ estimates of $\rho^{d}$, so the banded-average
log-correlation is

$$
\overline{\log|\hat{\rho}_d|} = \frac{1}{p - d} \sum_{j=1}^{p-d} \log|\hat{\rho}_{j,j+d}|,
\quad d = 1, \ldots, D,
$$

where $D \ll p$ is the maximum lag beyond which estimates are too noisy.
Under the model $\overline{\log|\hat{\rho}_d|} \approx d \cdot \log\rho$,
i.e., a regression through the origin with predictor $d$,
the WLS estimator with weights $w_d = (p - d)$ — reflecting that shorter lags
average over more pairs and are therefore more reliable — is:

$$
\widehat{\log\rho} =
\frac{\displaystyle\sum_{d=1}^{D} (p-d)\, d\; \overline{\log|\hat{\rho}_d|}}
     {\displaystyle\sum_{d=1}^{D} (p-d)\, d^2},
\qquad
\hat{\rho} = \exp\!\bigl(\widehat{\log\rho}\bigr).
$$

A practical upper bound for $D$ is $D = \lfloor\sqrt{n}\rfloor$,
beyond which the variance of $\overline{\log|\hat{\rho}_d|}$ grows
too large to contribute reliably.

To avoid numerical instability, pairs with $|\hat{\rho}_{j,j+d}| < \epsilon$
(e.g., $\epsilon = 10^{-6}$) are excluded from the band average.

#### Treating $r$ as the Calibration Parameter

Since Algorithm 1 already sweeps $\rho_{\text{thr}}$ over all discrete dendrogram levels,
the AR(1) structure allows replacing this sweep with a sweep over integer radii
$r = 0, 1, 2, \ldots, r_{\max}$, setting $\rho_{\text{thr}} = \hat{\rho}^r$ analytically
at each step — no clustering computation is needed.

The estimation error in $\hat{\rho}$ propagates into the group radius $r$ but
**does not affect the FDR control guarantee**: the martingale argument in
Theorem 2 holds for any fixed group assignment, regardless of how the groups
were constructed. The choice of $r$ affects TPR only, not FDR validity.

A natural upper bound is $r_{\max} = \lfloor \log(0.1) / \log\hat{\rho} \rfloor$,
the lag at which the estimated correlation drops below 10%.

## Penalty Analysis for AR(1)

Definition 1 postulates the dependency-aware relative occurrence of variable $j \in \{1, \ldots, p\}$
is defined by
$$
\Phi^{DA}_{T,L}(j, \rho_{\text{thr}}) := \Psi_{T,L}(j, \rho_{\text{thr}}) \Phi_{T,L}(j),
$$
where
$$
\Psi_{T,L}(j, \rho_{\text{thr}}) \;=\;
\begin{cases}
\dfrac{1}
{2 - \displaystyle \min_{j' \in \text{Gr}(j,\rho_{\text{thr}})} |\Phi_{T,L}(j) - \Phi_{T,L}(j')|},
& \text{Gr}(j, \rho_{\text{thr}}) \neq \emptyset \\
\dfrac{1}{2}, & \text{Gr}(j, \rho_{\text{thr}}) = \emptyset
\end{cases}
$$
with $\Psi_{T,L}(j, \rho_{\text{thr}}) \in [0.5, 1]$, being the deflating penalty factor,
and
$\text{Gr}(j, \rho_{\text{thr}}) \subseteq \{1, \ldots, p\} \setminus \{j\}$,
is the generic definition of the group of variables that are associated with varibale $j$, and
$\rho_{\text{thr}} \in [0, 1]$ is a parameter that determines the size of the variable groups.

The structure is a **reciprocal**, not a subtraction.
Defining $\delta := \min_{j'}|\Phi(j) - \Phi(j')| \in [0,1]$, the range of $\Psi \in [1/2, 1]$ is
guaranteed since $\Phi_{T,L}(j) \in [0,1]$ for all $j$, and $2 - \delta \in [1, 2]$.

| $\delta$ | $\Psi = \frac{1}{2-\delta}$ | Interpretation |
| :-- | :-- | :-- |
| 0 (identical $\Phi$) | $1/2$ | Maximum penalty — DA halves the relative occurrence. |
| 0.5 | $2/3$ | Moderate competition. |
| $\to 1$ (maximally different) | $\to 1$ | No penalty — variable dominates all its neighbors. |

**Remark (empty-group penalty).**
The case $\text{Gr}(j, \rho_{\text{thr}}) = \emptyset$ arises at the finest
resolution where all variables are singletons.
The paper assigns $\Psi = 1/2$, the maximum penalty, so
$\Phi^{DA}(j) = \frac{1}{2}\Phi(j)$.
Since $v \in [0.5, 1)$, selection requires $\Phi^{DA}(j) > v \geq 0.5$,
i.e., $\Phi(j) > 2v \geq 1$, which is never satisfied.
At the singleton level T-Rex+DA therefore selects nothing —
it does **not** reduce to ordinary T-Rex as one might expect.
