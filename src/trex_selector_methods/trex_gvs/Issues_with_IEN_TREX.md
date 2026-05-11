# Known Issue: IEN T-Rex Collapses at Near-Perfect Intra-Group Correlation (ρ ≈ 0.99)

**Status:** Confirmed, not a bug, not fixable without a theoretical extension.  
**Affected component:** `TRexGVSSelector` with `GVSType::IEN`.  
**Reference data:** `R/simulations/test_trex_gvs/simulation_results/demo_trex_gvs_01_hastie_models.md`

---

## Observed behaviour

On the Hastie DGP (n=200, p=500, 3 equicorrelated blocks of size 50 with ρ ≈ 0.99, all active,
SNR=2):

| Method | λ₂    | T_stop | Dummies | TPP   | FDP   |
|--------|-------|--------|---------|-------|-------|
| EN     | 1.010 | 27     | 500     | 0.933 | 0.091 |
| IEN    | 1.010 | 6      | 2000    | 0.020 | 0.000 |

IEN TPP ≈ 1/p_m = 1/50 = 0.020 — exactly one variable selected per active group.

**This is not a C++ implementation bug.** The CRAN `TRexSelector` package (R reference) produces
the same result. Monte Carlo simulations confirm the behaviour across all SNR values:

| SNR  | IEN mean_TPP | IEN mean_FDP |
|------|-------------|-------------|
| 0.1  | 0.007       | 0.000       |
| 0.5  | 0.018       | 0.000       |
| 2.0  | 0.020       | 0.000       |
| 5.0  | 0.020       | 0.000       |

SNR-invariance is the signature of a structural trap, not a signal-strength failure.
IEN works correctly up to ρ ≈ 0.95 (TPP = 1.0 at SNR ≥ 1) and fails sharply only at ρ ≈ 0.99.

---

## Root cause: equicorrelation of real and dummy columns in the IEN augmented system

### IEN objective (LASSO on augmented system)

The IEN LASSO solves:

$$\min_\beta \;\frac{1}{2}\|y - X\beta\|^2 \;+\; \frac{\lambda_2}{2}\sum_{m=1}^{M} \frac{1}{p_m}\!\left(\sum_{j\in\mathcal{C}_m}\beta_j\right)^{\!2} \;+\; \lambda_1\|\beta\|_1$$

by reformulating it as LASSO on the augmented system
$[X_\text{aug};\, y_\text{aug}]$ where $X_\text{aug} = [X;\, B(\lambda_2)]$,
$B(\lambda_2)_{m,j} = \frac{\sqrt{\lambda_2}}{\sqrt{p_m}}\,\mathbb{1}[j\in\mathcal{C}_m]$.

After `normalizeAugmentedColumns`, column $j$ in cluster $m(j)$ is rescaled by
$1/\sqrt{1 + \lambda_2/p_{m(j)}}$. The cross-correlation between two variables
$j, j'$ in the same cluster (with intra-group data correlation ρ) becomes:

$$G_\text{IEN}(j', j) = \frac{\rho + \lambda_2/p_m}{1 + \lambda_2/p_m}$$

For ρ=0.99, λ₂=1, p_m=50: $G_\text{IEN} = (0.99 + 0.02)/(1 + 0.02) \approx 0.990$.

Contrast with TENET (EN), where $d_2 = 1/\sqrt{1+\lambda_2}$ scales all top rows:

$$G_\text{EN}(j', j) = \frac{\rho}{1 + \lambda_2} \approx \frac{0.99}{2.01} \approx 0.493$$

EN breaks the equicorrelation; IEN preserves it.

### The LARS equicorrelation condition

LARS adds the variable with the **highest absolute correlation with the current residual**.
Under the equicorrelation condition — all $p_m$ active-group members (real *and* dummy) have
identical correlation with the residual — all $50 \times (1 + L)$ columns enter the path
*simultaneously* in the first step (or nearly so for LASSO with exclusion).

T-Rex interprets simultaneous dummy entries as false discoveries. With $L=4$ dummy layers
and 50-column equicorrelated groups, the estimated FDP after the very first steps already
exceeds tFDR=0.10, so T-Rex stops at T_stop ≈ 6.

### Why the CRAN TRexSelector makes it worse

The CRAN package uses `lar` (plain LARS) as default solver. LARS has no variable
exclusion — once a variable joins the active set it never leaves. Under near-perfect
equicorrelation the equicorrelation condition causes all 50 group members (real and dummy)
to enter the path in a single LARS step, immediately saturating the FDP estimate.

The C++ `TLASSO_Solver` has exclusion, but since real and dummy columns are statistically
indistinguishable (dummies drawn from MVN(0, Σ_m), augmented with the **same** bottom block
as real variables), T-Rex still cannot tell them apart.

---

## Why λ₂ calibration does not fix this

Both values λ₂ ≈ 1 (original-X GCV) and λ₂ ≈ 929 (augmented-ref GCV, investigated and
reverted) give TPP ≈ 0.02.

- **λ₂ ≈ 1 (small):** bottom rows ≈ 0, IEN reduces to plain LASSO. $G_\text{IEN}(j',j) \approx 0.990$.
  All group members remain equicorrelated → simultaneous LARS entry → T-Rex stops early.

- **λ₂ ≈ 929 (large, λ₂/p_m ≈ 19):** bottom row amplitude ≈ 0.97, top rows suppressed to ≈
  0.24. All 50 group columns (real and dummy) become nearly identical unit vectors →
  $G_\text{IEN}(j',j) \approx 0.9995$. The equicorrelation is even stronger → same outcome.

- **Any intermediate λ₂:** $G_\text{IEN}(j',j) = (\rho + \lambda_2/p_m)/(1 + \lambda_2/p_m)$.
  Since ρ = 0.99 ≈ 1, this is always ≥ ρ (monotone increasing in λ₂, and equal to ρ only at
  λ₂ = 0). There is no value of λ₂ that brings $G_\text{IEN}(j',j)$ down to the EN level of 0.49.

The root cause is structural: the IEN augmented system preserves equicorrelation across
the active group, while EN's d₂ factor breaks it.

---

## What is *not* a fix (approaches investigated and discarded)

1. **Augmented-ref GCV for λ₂:** Investigated and reverted. See `computeLambda2()` comment
   in `trex_gvs.cpp`. Makes things worse at ρ ≈ 0.99 (λ₂/p_m rises from 0.02 to 19).

2. **Post-selection group expansion:** Returns all group members when one is selected.
   Raises individual-variable TPP from 0.02 to 1.0, but this is an evaluation trick —
   it does not change what T-Rex selected or whether FDR is controlled at group level.
   Not proposed in the original GVS theory.

3. **Reducing dummy multiplier L:** Delays the structural collapse but does not prevent it;
   at ρ ≈ 0.99 the first LARS step already enters the full equicorrelated group.

---

## Summary

IEN T-Rex with ρ ≈ 0.99 all-active groups hits the LARS equicorrelation trap. The IEN
augmented system cannot break the near-perfect within-group correlation in the way that
EN's d₂ factor does. This is an inherent limitation of combining IEN (group-mean LASSO
penalty) with the T-Rex knockoff framework when all group members are simultaneously active
and near-perfectly correlated. EN (TENET solver) does not have this limitation because
d₂ = 1/√(1+λ₂) geometrically separates active-group columns from their knockoff dummies.


The IEN LASSO on the augmented system with bottom block $B(\lambda_2)$ is:

$$\min_\beta \;\frac{1}{2}\|y - X\beta\|^2 \;+\; \underbrace{\frac{\lambda_2}{2}\sum_{m=1}^{M} \frac{1}{p_m}\!\left(\sum_{j\in\mathcal{C}_m}\beta_j\right)^{\!2}}_{\|B(\lambda_2)\,\beta\|^2/2,\; \text{group-mean penalty}} \;+\; \lambda_1\|\beta\|_1$$

with $B(\lambda_2)_{m,j} = \frac{\sqrt{\lambda_2}}{\sqrt{p_m}}\,\mathbb{1}[j\in\mathcal{C}_m]$.

**After** `normalizeAugmentedColumns`, column $j$ (belonging to cluster $m(j)$) has pre-centering L2 norm $\sqrt{1 + \lambda_2/p_{m(j)}}$. The normalized top rows are therefore scaled down by this factor:

$$\tilde{x}_j \;\approx\; \frac{x_j}{\sqrt{1 + \lambda_2/p_{m(j)}}}, \qquad \text{(bottom row } m(j) \approx \frac{\sqrt{\lambda_2/p_{m(j)}}}{\sqrt{1+\lambda_2/p_{m(j)}}}\text{)}$$

Two limiting regimes:
- $\lambda_2 \ll p_m$: bottom rows $\to 0$, top rows unchanged → **plain LASSO** on $(X,y)$, no group structure
- $\lambda_2 \gg p_m$: top rows $\to 0$, bottom rows $\to 1$ → **group LASSO** on group-mean indicators

The ratio $\lambda_2/p_m$ controls the trade-off. IEN is designed to sit between these extremes.

---

## Why original-X ridge CV fails for IEN

The current `computeLambda2()` solves:

$$\lambda_2^* = \arg\min_\lambda \operatorname{GCV}(X, y, \lambda) = \arg\min_\lambda \frac{\|y - \hat{y}(\lambda)\|^2/n}{\left(1 - \operatorname{df}(\lambda)/n\right)^2}$$

This is the optimal isotropic ridge penalty for **prediction from $X$**. For high-correlation data the spectrum of $X^TX$ is extremely spread: $\sigma_1 \gg \sigma_p$. The GCV/CV selects $\lambda_2^* \approx \sigma_{\min}^2$ — very small. Concretely, if $\sigma_{\min} \approx 0.01$ then $\lambda_2^* \approx 10^{-4}$, so $\lambda_2^*/p_m \approx 10^{-5}$ for $p_m=10$ → bottom rows vanish → plain LASSO.

Note: **both** GCV and CV_1SE on original $X$ give small $\lambda_2$ in this regime. CV_1SE goes 2–5× larger than the CV minimum, but if $\lambda_2^{\min} \approx 10^{-4}$, then $\lambda_2^{1SE} \approx 5\times10^{-4}$ — still negligible for IEN. This is why both methods give TPR ≈ 0.02.

---

## Proposed fix: GCV/CV on the reference augmented system

Build:

$$X_{\text{aug,ref}} = \begin{pmatrix} X \\ G_{\text{ref}} \end{pmatrix} \in \mathbb{R}^{(n+M)\times p}, \qquad G_{\text{ref},m,j} = \frac{1}{\sqrt{p_m}}\,\mathbb{1}[j\in\mathcal{C}_m], \qquad y_{\text{aug,ref}} = \begin{pmatrix} y \\ 0_M \end{pmatrix}$$

$G_{\text{ref}} = B(\lambda_2 = 1)$ — the bottom block at unit amplitude. Normalize columns to unit L2 norm, then run the selected GCV/CV method.

**What the GCV selects here.** The augmented Gram matrix is:

$$X_{\text{aug,ref}}^T X_{\text{aug,ref}} = X^TX \;+\; \underbrace{G_{\text{ref}}^T G_{\text{ref}}}_{\displaystyle=:\;Q}$$

where $Q$ is block-diagonal with block $m$ equal to $\frac{1}{p_m}\mathbf{1}_{p_m}\mathbf{1}_{p_m}^T$ — the **within-cluster projection**. $Q$ adds an eigenvalue of 1 per cluster to the Gram matrix. The GCV now finds:

$$\lambda_{\text{iso}}^* = \arg\min_\lambda \operatorname{GCV}(X_{\text{aug,ref}}, y_{\text{aug,ref}}, \lambda)$$

which corresponds to the optimal isotropic regularization for the model $(X^TX + Q + \lambda_{\text{iso}} I)^{-1} X^T y$. Because $Q$ adds $M$ eigenvalues of order 1 to a Gram matrix that may have many near-zero eigenvalues, $\lambda_{\text{iso}}^*$ will be substantially larger than the original-X result.

**Limitation — it is a heuristic.** We use $\lambda_{\text{iso}}^* \cdot p/2$ as the IEN group-mean amplitude $\lambda_2$, but the GCV selects the isotropic ridge $\lambda_{\text{iso}}$ on the augmented system, not the group-mean amplitude directly. The fully calibrated approach would cross-validate $\lambda_2$ as an outer loop over the IEN LASSO — expensive, and not done in the R reference either. The reference simply uses $\lambda_{1\text{SE}}^{\text{ridge}}(X, y) \cdot p/2$, which is also a heuristic. What we gain: the GCV sees the group projection $Q$ and selects a $\lambda_2$ that ensures the group rows are non-negligible in the actual IEN augmentation.

---

**Summary:** the fix is mathematically sound as a heuristic. It is no less principled than the R reference — it just uses a different (group-aware) calibration matrix. Shall I proceed with implementation?




The augmented-ref GCV for IEN inflated λ₂ from ≈1 to ≈929. This does NOT help: at λ₂/p_m ≈ 19, the IEN augmented system makes all 50 active-group members (real and dummy) nearly identical in normalised column space (bottom row ≈ 0.97, cross-correlation G(j′,j) ≈ 0.9995). They all enter the LARS path in the same step, so T-Rex still stops at T_stop = 6 with TPP = 0.02.

The actual root cause

IEN T-Rex TPP ≈ 0.02 on the Hastie all-active DGP (ρ ≈ 0.99) is not a bug in λ₂ calibration. The structural issue is:

EN (TENET_Solver)	IEN (TLASSO on augmented)
G(j′,j) for active group	0.99/(1+λ₂) ≈ 0.49	(0.99 + λ₂/p_m)/(1 + λ₂/p_m) ≈ 0.990
Active-group dummies ∼ real active?	No — d2 factor suppresses	Yes — identical bottom block
T_stop	27	6
TPP	0.933	0.020
IEN dummies are drawn from MVN(0, Σ_m) and augmented with the same IEN bottom block as real variables → they are statistically indistinguishable from real active-group variables → T-Rex cannot tell them apart → stops extremely early.

EN CV_1SE TPP = 1.0 is correct (documented in the Phase 1 comment — the 1SE rule's larger λ₂ couples correlated variables, which is intentional).

What would actually fix IEN performance (not implemented — needs a design decision from you):

Group-level T-Rex: count one "dummy group entry" per cluster m rather than counting each of the p_m dummy columns separately. This requires changes to the dummy assembly and the T-Rex Φ accumulation.
Post-selection group expansion: after T-Rex selects one representative per active group, expand the selection to all members of those clusters. The R reference provides both selected_var (individual) and groups for exactly this purpose.
Accept IEN as group-representative selector: IEN's individual-variable TPP ≈ 0.02 correctly reflects that it selects 1 representative per active group (group-level TPP = 3/3 = 1.0). Evaluate at the group level.
