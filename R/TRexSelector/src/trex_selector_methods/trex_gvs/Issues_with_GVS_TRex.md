# Critical Assessment: T-Knock+GVS (Machkour et al., EUSIPCO 2022)

## 1. Theoretical Claims vs. Actual Coverage

- Theorem $1$ proves the grouped variable selection property **only for perfectly
  correlated variables** ($| \rho | = 1$), a degenerate case with less relevance to real data.
  It is a direct algebraic consequence of Lemma 2(a) in Zou & Hastie (2005).

- The proof of Theorem $1$ applies to the **full Elastic Net solution**, not to the
  T-LARS stopping point. The theorem is proved for a regime (dense path end) that the
  algorithm never reaches by construction.

- Elastic Net does **not** perform true group selection. Its penalty
  $\lambda_{2} \| \boldsymbol{\beta} \|_{2}^{2} + \lambda_{1} \| \boldsymbol{\beta} \|_{1}$ is
  fully separable over individual coefficients. It merely encourages correlated variables
  toward similar magnitudes — a soft, continuous tendency, not a hard all-in/all-out group
  constraint as in Group Lasso (Yuan & Lin, 2006).

- There is **no theorem** guaranteeing that if a group is truly active, at least one
  member will be selected. The TPR improvement is empirically observed but theoretically ungrounded.

## 2. FDR Definition vs. FDR Evaluation Mismatch

- The FDR is formally defined at the **individual variable level** (Eq. 1), but all
  performance metrics in the GWAS simulation are evaluated at the **group level** (a group
  is active if it contains a disease SNP). These are categorically different quantities.

- Empirical FDR control at group level does not imply, and is not implied by,
  individual-level FDR control.

- The benchmark methods (model-X, T-Knock) achieve FDPs of $3.84%$ and $4.40%$ against
  a $20\%$ target — strongly conservative behavior that inflates the apparent TPR advantage
  of T-Knock+GVS.

## 3. The Algorithm Does Not Run Elastic Net

- T-LARS stops the LARS path as soon as the T-th dummy enters the active
  set. This forces early termination deep in the ultra-sparse regime (large $\lambda_{1}$),
  precisely where the grouping effect of the Elastic Net is **inactive**.
  The grouping property only emerges at the dense end of the path (small $\lambda_{1}$).

- What is actually being executed is **Ridge-penalized LARS**: LARS equiangular steps
  computed in a Gram matrix shifted by $\lambda_{2} \mathbf{I}$, stopped early. This is not
  equivalent to solving the Elastic Net problem and cannot inherit its grouping property.

- The Ridge term $\lambda_{2}$ is determined by $10$-fold cross-validated Ridge regression.
  This $\lambda_{2}$ optimizes prediction, not group selection behavior, and its relationship to the
  stopping threshold T is never analyzed.

## 4. Invalid Knockoff Construction

- Knockoffs are sampled as x̊'_{z,w,i} ~ N(0, Σ̂_z) — independent of the observed
  data. This satisfies only marginal cluster correlation mimicry, not the conditional
  exchangeability (swapping property) required for valid model-X knockoffs.

- The structured, cluster-correlated dummies break the exchangeability assumption
  that T-Rex's FDR control proof is built on. T-Rex was designed for white-noise
  N(0,1) dummies; the two dummy structures are never jointly analyzed.

- Within each cluster, dummies are mutually correlated via Σ̂_z. In the LARS
  equiangular step, correlated dummies do not compete independently against original
  variables — the joint Gram matrix geometry is non-trivial and unanalyzed.

- The Ridge term $\lambda$₂I applied uniformly to the enlarged Gram matrix treats original
  variables and correlated dummies symmetrically in the diagonal, while introducing
  asymmetric off-diagonal structure. The effect on the competition fairness (and hence
  FDR control) is never addressed.

- Estimating a full $|J_z| \times |J_z|$ covariance matrix from $n = 700$ observations for
  potentially large SNP clusters is statistically wasteful and numerically fragile
  (near-singular $\widehat{\Sigma}_{z}$ for large clusters).
  A factor model (low-rank + diagonal) would be both statistically and scientifically more 
  appropriate for LD block structure.

## 5. Computational Nonsense

- The augmented form (X augmented with √$\lambda$₂ · I_p rows, y augmented with zeros) is
  used literally in the implementation. Zou & Hastie (2005) explicitly state in Section 3.2
  that this is a **conceptual device only**, and that practical computation should use the
  modified Gram matrix $X^{\top} X + \lambda_{2} \mathbf{I}$ directly.

- The unnecessary augmentation inflates the effective system from ($n \times p$) to
  ((n+p) × p). For $n=700$, $p=1000$, this is a $\sim 2.4\times$ row inflation applied to every LARS
  solve. Combined with K experiments and L/p sub-knockoff repetitions, the cost explodes.

- The resulting 24× computation overhead vs. base T-Knock ($00:05:55$ vs. $00:00:14$)
  is a direct consequence of this avoidable implementation choice, not an inherent cost
  of the method. The paper attributes it vaguely to "increased row-dimension."

- The base T-Knock/T-Rex paper claims $\mathcal{O}(p)$ computational complexity. This property
  is effectively destroyed in the GVS version, scaling as $\mathcal{O}(K · (n + p) · p)$ per run —
  approximately $O(Kp²)$ in the high-dimensional regime $p \gg n$.

- The paper required a **high-performance computing cluster** (Lichtenberg HPC,
  TU Darmstadt) to run $100$ Monte Carlo replications at $p=1000$. This is an implicit
  admission that the method is computationally impractical at the scales it targets.

## 6. Algorithm Design Failures

- Single-linkage hierarchical clustering is hardwired inside the algorithm with a
  fixed threshold $\rho_\mathrm{thr} = 1/3$. Single-linkage suffers from the chaining
  effect and is poorly suited to identifying compact LD blocks.

- The group structure is a property of $X$, not of the dummy generation procedure.
  Clustering should be performed **outside** the algorithm, with group labels passed as
  input — enabling domain knowledge, alternative clustering methods, and reproducibility.
  This is standard practice in all Group Lasso implementations.

- The dendrogram is computed once per knockoff matrix (Algorithm $1$ is called $K$
  times), yielding $K$ identical clusterings. This is computationally redundant; the
  clustering result is deterministic given $X$ and $\rho_{\mathrm{thr}}$.

- Fixing $\rho_{\mathrm{thr}} = 1/3$ "empirically" from GWAS domain knowledge hardcodes
  a domain-specific assumption into a general-purpose algorithm without justification.

## 7. Simulation Study Validity

- A **linear regression model** with Gaussian noise is fitted to a **binary
  case/control phenotype** ($500$ cases, $200$ controls). This is a linear probability
  model: residuals are heteroscedastic by construction, predicted probabilities leave
  $[0,1]$, and the Gaussian noise assumption is violated. Logistic regression is standard
  for case-control GWAS.

- With only $100$ Monte Carlo replications and a single genomic region (first $1000$
  SNPs on Chromosome $15$), the simulation is too narrow to support general claims about
  grouped variable selection performance.

- The "group-level FDR" evaluation metric is defined ad hoc for the simulation and
  is not connected to the formal FDR definition used in the theoretical sections.
