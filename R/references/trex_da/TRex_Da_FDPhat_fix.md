# Mathematical Analysis

## Notation

Let $p$ be the number of predictors, $K$ the number of random experiments, $T$ the current T-loop
step, and $L$ the number of dummies. Define the raw vote matrix:

$$\phi_{j,t} \in [0,1], \quad j = 1,\ldots,p,\quad t = 1,\ldots,T$$

where $\phi_{j,t}$ is the fraction of $K$ experiments in which predictor $j$ is selected at step
$t$.
By construction, $\phi_{j,\cdot}$ is **non-decreasing** in $t$, and the final vote fraction is
$\Phi_j = \phi_{j,T}$.

---

### Step 1 — AR1 Deflation in `daCorrect`

For each pair $(j, t)$, let $\mathcal{N}_\kappa(j) = \{k : |k-j| \leq \kappa,\, k \neq j\}$ be the
AR(1) correction window of half-width $\kappa = \lceil \log(0.02) / \log(\hat\rho) \rceil$. Define
the minimum absolute neighbor difference:

$$d_{j,t} = \min_{k \in \mathcal{N}_\kappa(j)} |\phi_{j,t} - \phi_{k,t}|$$

The deflation factor $\delta_{j,t} = 2 - d_{j,t} \in [1, 2]$, and the deflated vote matrix is:

$$\tilde\phi_{j,t} = \frac{\phi_{j,t}}{\delta_{j,t}}$$

**Critical observation.** Because $\delta_{j,t}$ depends on the full neighbourhood at each
individual step $t$, it is a different scalar for every $(j,t)$. Even though $\phi_{j,\cdot}$
is non-decreasing in $t$, dividing by a $t$-varying scalar destroys this property:

$$\tilde\phi_{j,t} \not\geq \tilde\phi_{j,t-1} \quad \text{in general}$$

The incremental (backward-differenced) deflated votes
$\Delta\tilde\phi_{j,t} = \tilde\phi_{j,t} - \tilde\phi_{j,t-1}$ can therefore be **negative**.

---

### Step 2 — `computePhiPrime` and the Blow-Up

Let $S_{>1/2} = \{j : \Phi_j > 1/2\}$ index the strongly-voting predictors.
Define column-sum accumulators:

$$a_t = \sum_{j=1}^p \tilde\phi_{j,t}, \qquad b_t = \sum_{j \in S_{>1/2}} \tilde\phi_{j,t}$$

After backward differencing: $\Delta b_t = b_t - b_{t-1}$ (with $\Delta b_1 = b_1$).
The per-step scale factor is (when $\Delta b_t > \varepsilon$):

$$s_t = 1 - \frac{p - a_t}{(L - t)\,\Delta b_t}$$

The corrected vote probability is then:

$$\Phi'_j = \sum_{t=1}^{T} \Delta\tilde\phi_{j,t} \cdot s_t$$

**The blow-up mechanism.** When the support is densely packed (small `max_gap`) and $\kappa$ is
large, every true predictor falls inside every other true predictor's correction window.
This causes $\delta_{j,t}$ to vary strongly across $t$, making $\Delta\tilde\phi_{j,t} \ll 0$ for
some $(j,t)$.
Simultaneously, $\Delta b_t \approx 0^+$ (the incremental strongly-voting mass is near-zero), so:

$$s_t = 1 - \frac{p - a_t}{(L-t)\,\Delta b_t} \ll -1$$

The product of two negatives gives:

$$\Delta\tilde\phi_{j,t} \cdot s_t > 0 \quad \text{and} \gg 1$$

Accumulating over $t$:

$$\Phi'_j = \sum_t \Delta\tilde\phi_{j,t} \cdot s_t \gg 1 \quad \text{for some } j \in S_{>1/2}$$

---

### Step 3 — Negative FDP in `computeFDPHat`

For threshold $v$, let $\hat{S}(v) = \{j : \Phi_j > v\}$ with $R(v) = |\hat{S}(v)|$.
The FDP estimator is:

$$\widehat{\text{FDP}}(v) = \frac{1}{R(v)} \sum_{j \in \hat{S}(v)} (1 - \Phi'_j)$$

When $\Phi'_j > 1$ for predictors $j \in \hat{S}(v)$, the term $(1 - \Phi'_j) < 0$ contributes a
negative mass to the sum.
If enough such predictors are selected, the sum becomes negative and $\widehat{\text{FDP}}(v) < 0$,
violating the basic constraint $\widehat{\text{FDP}}(v) \in [0,1]$.

---

### The Two Fixes

**Fix 1 — Clamp $\Phi'_j$**  
Since $\Phi'_j$ is a corrected vote probability, it must satisfy $\Phi'_j \in [0,1]$ by definition.
Enforce:

$$
\Phi'_j \leftarrow \max\!\bigl(0,\, \min(1, \Phi'_j)\bigr) \quad \forall j
$$

This ensures $(1 - \Phi'_j) \in [0,1]$ for every predictor, making it impossible for any term in
the FDP sum to be negative.

**Fix 2 — Clamp $\widehat{\text{FDP}}(v)$.**  
Regardless of what $\Phi'_j$ computes, the FDP estimator is a ratio bounded in $[0,1]$ by
definition. Enforce:

$$
\widehat{\text{FDP}}(v) \leftarrow \max\!\left(0,\, \min\!\left(1,\, \frac{1}{R(v)} \sum_{j \in \hat{S}(v)} (1 - \Phi'_j)\right)\right)
$$

* Fix 1 addresses the corrupted intermediate quantity $\Phi'$.
* Fix 2 is a defensive semantic boundary on the final estimator.
* Together they guarantee $\widehat{\text{FDP}}(v) \in [0,1]$ under all configurations, including
degenerate AR1 regimes with gap $<$ $\kappa$.
