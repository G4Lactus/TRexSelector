# Hierarchical Clustering Investigation — Session Summary

## 1. Starting Point: CLINK (Defays 1977)

- CLINK is the complete-linkage analogue of SLINK, running in O(n²) time and O(n) space.
- **It is a heuristic, not exact.** Results are order-dependent and can diverge from
  true complete-linkage due to inability to guarantee globally optimal merges.
- Recent theoretical work (Röglin et al. 2017, Wargalla HHU 2024) focuses on
  approximation ratio analysis of complete-linkage as a clustering objective,
  not on fixing CLINK itself.
- **Decision: drop CLINK; replace with NN-Chain for complete linkage.**

---

## 2. NN-Chain Algorithm

- Exact $\mathcal{O}(n^{2})$ algorithm for any **reducible** linkage (complete, UPGMA, WPGMA, Ward).
- Core idea: grow a nearest-neighbour chain on a stack; merge on Reciprocal
  Nearest Neighbour (RNN) pairs — guaranteed safe by the reducibility property:
  $$
    d(Cᵢ∪Cⱼ, Cₖ) ≥ min(d(Cᵢ,Cₖ), d(Cⱼ,Cₖ))
  $$
- Standard implementation (fastcluster, scipy) maintains $\mathcal{O}(n^{2})$ condensed distance
  matrix updated via Lance-Williams recurrence after each merge.
- **ParChain (VLDB 2021):** parallel extension — grows multiple chains simultaneously,
  lock-free, exact for complete/Ward/average linkage.

---

## 3. SLINK (Sibson 1973)

- Contrary to CLINK, SLINK is exact
- Time-optimal for single-linkage: achieves the $\Omega(n^{2})$ lower bound exactly.
- Pointer representation $(\pi, \lambda)$: two arrays of length n — true $\mathcal{O}(n)$ extra space.
- More cache-friendly and lower constant than NN-Chain for single linkage.
- **Tie-breaking caveat:** SLINK's pointer representation is ambiguous when integer
  or quantized distances produce ties. Add a tie-breaking guard if needed.
- **Decision: keep SLINK for single linkage; it has no equal for general metrics.**

---

## 4. Minimal Spanning Tree (MST)

- Single-linkage ≡ MST of the complete distance graph (Gower & Ross 1969).
- For dense general metrics: same $\mathcal{O}(n^{2})$ cost as SLINK, with heavier bookkeeping
  (priority queue, union-find) → SLINK wins on constants and cache behaviour.
- For low-dimensional Euclidean data: k-NN graph + Kruskal is subquadratic,
  but requires connectivity guarantee (k ≥ ⌈log₂ p⌉) and degrades in high dim.
- MST wins only in HDBSCAN (density-aware, tolerates approximation) and in
  very large low-dimensional Euclidean problems.
- **For general / correlation metrics: SLINK remains the rational default.**

---

## 5. Problem Setting: $N \times p$ Design Matrix, $p \gg N$, $d_{ij} = 1 − |ρ(xᵢ,xⱼ)|$

- Data matrix $X$ is $N \times p$; we cluster the **p columns** (variables).
- After column-centering and $\ell_{2}$-normalisation, all $p$ variables lie on $S^{N-1}$ unit 
  sphere $\rightarrow$ intrinsic geometry is $N$-dimensional, not $p$-dimensional.
- Distance: $d_{ij}=1-|\tilde{x}^{\top}_{i} \tilde{x}_{j}|$ (absolute angular / projective distance).
- The absolute value $|\rho|$ groups positively and negatively correlated variables —
  desirable for variable selection / factor structure discovery.
- With $p \sim 10⁶–10⁷$ and $N \sim 10³–10⁵$, key scales:
  - True cost per distance evaluation: $\mathcal{O}(N)$ FLOPs (one dot product)
  - Total SLINK cost:    $\mathcal{O}(p^{2} N)$  FLOPs
  - Full corr. matrix:  $p^{2}/2 × 4$ bytes (float32) $\rightarrow$ infeasible for $p > \sim10⁵$.

---

## 6. Statistical Warning: p ≫ N and Noisy Correlations

- Sample correlation matrix $\widehat{C} = X^{\top} X$ has rank $\leq N \ll p$.
- Marchenko-Pastur law: O(N/p) fraction of variance in Ĉ is pure noise.
- **UPGMA (average linkage) recommended** over complete/single for noisy
  correlation estimates — averages out noise across cluster pairs.
- Ward linkage with 1−|ρ|: Lance-Williams formula is algebraically valid
  (produces non-inverting dendrogram) but the WSS minimisation objective
  is undefined for non-Euclidean distances. Use with caution.

---

## 7. Absolute Value Breaks Centroid Representation

- For signed distance $d=1−\rho$:  UPGMA = centroid distance → $\mathcal{O}(p)$ extra space.
- For $d = 1−|\rho|$:  $\mathbb{E} |\rho| \neq |\mathbb{E}(\rho)|$ → centroid trick fails.
- **Consequence:** UPGMA, WPGMA, Complete, Ward all require the stored O(p²)
  condensed distance matrix for exact Lance-Williams updates.
- SLINK remains the only algorithm with true O(p) extra space.

---

## 8. Space-Time Tradeoff for Complete Linkage

| Space budget | Time | Feasible at $p=10^{6}$? |
| ---------------------- | ------------------- | ------------------------- |
| $\mathcal{O}(p^{2})$ stored matrix | $\mathcal{O}(p^{2} N) + \mathcal{O}(p^{2})$ | ❌ ~4 TB |
| $\mathcal{O}(p N)$ input only | $O(p^{3} N)$ worst case | ❌ catastrophic |
| Block tradeoff (2025) | $\mathcal{O}(p^{2} t)$, $O(p^{2}/t)$ | Partial, tunable |

---

## 9. Practical Feasibility by Scale

| p range | Storage for $\widehat{C}$ (float32) | Recommended strategy |
| ------------ | -------------------------- | ----------------------------------------------- |
| p ≲ $10^{4}$ | ≤ 200 MB | DSYRK full Ĉ + NN-Chain (any linkage) |
| p ~ $10^{5}$ | ~10 GB | NN-Chain if RAM permits; else SLINK only |
| p ≳ $10^{6}$ | Infeasible | SLINK exact; NN-Chain needs sparse Graph HAC |

---

## 12. Algorithms Deferred to Future Sessions

- **Graph HAC (Dhulipala et al., ICML 2021):** exact, $\mathcal{O}(m)$ time on sparse
  similarity graphs — the path forward for $p > 10^{5}$.
- **k-NN graph + Kruskal MST:** subquadratic single-linkage for low-dim Euclidean.
- **FAISS / HNSW approximate k-NN:** for $p \sim 10^{6}$+ with approximate dendrogram.
- **Sparse Graph HAC:** combining sparse correlation structure with exact linkage.
