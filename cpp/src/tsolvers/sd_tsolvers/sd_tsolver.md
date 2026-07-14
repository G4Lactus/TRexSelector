# SD T-Solvers — Sparse-Dummy Terminating Solvers

Design notes and validated findings. Status: 2026-07-12.

> **Consolidated method document** (mathematical description, selector
> layer, open problems): `trex_selector_methods/trex_sd/trex_sd.md`.
> This file keeps the solver-level engineering notes and detailed run logs.

The SD family runs the T-Rex early-stopping race against **sparse balanced
Rademacher dummies** instead of an explicit dummy matrix `D`. A dummy is
represented only by its index sets (k positions `+1`, k positions `−1`), so
every inner product against the residual collapses to index-set sums —
additions instead of dot products — and no dummy column exists in memory
until it wins a race (in SD2, not even then).

---

## 1. The dummy law and the fair-race contract

**Law.** A sparsity-k dummy has exactly `2k ≈ rho_d · n` non-zeros, balanced
(`k` positive, `k` negative) at uniformly random positions — identical to the
package's `ConstrainedSparseRademacher(s = rho_d)` in
`utils/datageneration/utils_dummygen.hpp` (`2k = 2·floor(s·n/2)`, clamped to
`[2, n]`, even). Consequences: exact column mean zero and **deterministic**
L2 norm `sqrt(2k)`.

**Scaling (mandatory).** The selection race compares `|x_j^T r|` against
`|d^T r|`, so fairness requires equal column norms. The solvers scale every
dummy statistic by

```text
dummy_scale = ||x|| / sqrt(2k)
```

where `||x||` is the common column norm of X, measured at construction. This
keeps the arithmetic multiplication-free (index-set sums, one scalar
multiply) and makes both preprocessing contracts work automatically:

| contract | column norm | resulting dummy scale |
| --- | --- | --- |
| unit-L2 (canonical; matches the VD family) | 1 | `1/sqrt(2k)` |
| z-score | `sqrt(n-1)` | `sqrt((n-1)/2k)` |

The two differ by the *global* constant `sqrt(n-1)`, so the selection path
is identical — **verified**: identical paths entry-for-entry, lambda ratio
exactly `sqrt(n-1) = 17.29` at n = 300 (`demo_sd_tlars`). Without the
scaling the race is broken outright: unscaled dummies carry norm
`sqrt(rho_d·n)` (≈ 12× at n=300, rho_d=0.5) and win every step — 20 dummies,
0 reals selected.

**Caveat.** Fairness relies on X having *equal* column norms; the solver
averages the norms without complaint, so raw unstandardized input silently
biases the race.

## 2. Statistical structure

- **Second-moment isotropy, every k.** For any balanced k, `E[d] = 0` and
  `E[d dᵀ] = (I − 𝟏𝟏ᵀ/n)/(n−1)` — exactly the second moments of the VD
  family's spherical law on the centered subspace. (Symmetry gives
  `αI + βJ`; exact mean zero forces zero row sums; unit norm fixes the
  trace.) The sparsity k changes only the **higher moments** — the tails of
  the dummy correlation — which is precisely where the race lives. The
  marginal null variance `Var(d^T r) ≈ ||r||²/n` matches a unit-norm random
  column for every k.
- **k = 1 (2-sparse, `d = s(e_i − e_j)`)**: the null is the *pairwise
  difference distribution of the actual residuals* — the Gini
  mean-difference kernel. A nonparametric, permutation-flavored null that
  inherits heavy tails/outliers of the true noise (dense dummies CLT them
  away) — attractive for robust statistics: the race calibrates to the
  empirical noise, not an assumed Gaussian.
- **The price of k=1**: bounded, discrete support. The universe has only
  `C(n,2)` pairs; `max |d^T r| = s·(max r − min r)`; the top pairs are rare
  atoms (~1/44,850 per draw at n=300).
- **Ceiling for general k**: `(Σ top-k r − Σ bottom-k r)/sqrt(2k)` — exact
  from the sorted residual, and non-monotone in k (at n=300 it peaks near
  k = n/4: 15.6σ at k=75 vs 13.9σ at k=150).
- **Relation to VD**: VD keeps a fixed L virtual by integrating the
  spherical law (rotational invariance). Sparse Rademacher is not
  rotationally invariant; SD instead gets **combinatorial integrability**
  at k=1 — the null is a finite, fully enumerable set:
  - best possible dummy in closed form: the pair (argmax r, argmin r);
  - exact beating fraction `π(c) = #{pairs: |r_i − r_j| > c/s} / C(n,2)`
    by one sort + two-pointer pass;
  - the generate-until-beat count is Geometric(π), and the winner is a
    uniform draw from the beating set (rank inversion);
  - even a fixed-L race max is samplable by inverse transform from
    `F(c)^L` without instantiating any of the L dummies.

## 3. Architecture: two arithmetics, two files

| | `sd_tlars_solver` (general k) | `sd2_tlars_solver` (k = 1) |
| --- | --- | --- |
| generation | O(n) partial Fisher–Yates | O(1), two uniform draws |
| correlation `⟨d,r⟩` | 2k adds | one subtraction `s(r_i − r_j)` |
| step size `⟨d,u⟩` | 2k adds | `s(u_i − u_j)` |
| Cholesky cross products | O(n) dots (materialized winner) | O(1) per active column |
| equiangular update of u | full-column axpy | two entries |
| winner storage | materialized into n×T cache | the (i, j) pair — nothing else |
| dummy memory | 2k ints + 2 heap vectors | 24 B POD (flat pool, SIMD-friendly) |

`SD2` replicates the general solver's RNG draw pattern at k=1, so **for the
same seed both solvers race the identical dummy stream** — verified:
identical selection paths (39 and 44 entries in the demo scenarios). This is
the standing cross-validation harness for future changes.

**Incremental pool correlations (both solvers).** `c_j = d^T r` and
`r ← r − γu` imply `c_j ← c_j − γ⟨d, u⟩` exactly; since the step-size search
already computes `⟨d, u⟩` for every pool dummy, the former O(Q·2k) per-step
pool recomputation is a free O(Q) update. Verified: identical selection
paths, and the calibrated p = 10⁶ sweep dropped from 8.3 s to 5.6 s.

**Generation policies (SD2).** `SD2GenPolicy::OnDemand` mirrors the general
solver: explicit pool, generate-until-a-dummy-beats-the-best-real.
`SD2GenPolicy::Geometric` uses the exact pair null (Section 2): Geometric(π)
draw for the generation count, winner sampled from the beating set;
failures are never instantiated — only the virtual pool size F is tracked,
and the standing pool is re-raced each step via `Bernoulli(1 − (1−π)^F)`.
The **exchangeability approximation** (frozen failures treated as fresh
draws against the current residual) makes the policy measurably **more
conservative**: FDP 0.091 vs 0.474 at T=20 (scenario A), 13–15 selected
reals vs 32–41 at T=40 — at identical TPR. LARS pool coherence is the
engineering caveat behind the approximation: the γ-step minimizes over
persistent dummies, which the virtual pool does not represent.

**Sparse beta path.** `actives_` is append-only, so step t stores only its
actives' coefficients (`betaPathCompact_`, O(steps²)) instead of the dense
`(p+L_max) × (8·min(n,p)+1)` matrix. At p = 10⁵ this cut peak memory from
~5.8 GB to 636 MB, and it *structurally* removes the global-index
out-of-bounds bug class: a dummy's global index is now a label, never a row
offset into a fixed-size buffer.

## 4. Measured performance

Same 2-sparse law everywhere; n=2000, p=500, L=10⁴, T=30 (demo 2, scenario B):

| solver | dummy arithmetic | solve time | dummy storage |
| --- | --- | --- | --- |
| TLARS (explicit D) | L·n dots/step | ≈ 150 ms | 153 MiB + build |
| SD_TLARS (k=1) | index-set sums | ≈ 20–28 ms | n×T cache |
| SD2 (pair, on-demand) | subtractions | ≈ 11–16 ms | 24 B/dummy |
| SD2 (geometric) | sorted-residual draws | ≈ 16–20 ms | none (count only) |

≈ 11× vs the classic explicit-D solver, decomposing into ~7× for
virtualizing the dummies and another ~1.7–2.3× for erasing the residual
dots. What remains is the irreducible real-feature work (`X^T u`), shared by
every T-Rex variant — so **large L is nearly free**: L = 2·10⁶ costs 48 MB
as pairs; the explicit-D equivalent is 4.8 GB, and at the upper end of the
T-Rex L-loop (L up to 10p) with p = 10⁶, 24 TB (physically impossible). The
sparse representation is what makes the large-p regime reachable at all.

## 5. Choosing k: the tail trade-off

Signal regime (n=300, p=1000, SNR 3): k ∈ {1, 7, 75} behave essentially
identically — perfect selection at T ≤ 5, first dummy at step 11.

Near-null regime (n=300, p=5000, snr≈0): false reals admitted before the
10th dummy — the k=1 null is tail-deficient and under-stops:

| seed | k=1 | k=7 | k=75 |
| --- | --- | --- | --- |
| 7 | 24 | 18 | 10 |
| 21 | 7 | 8 | 6 |
| 99 | 26 | 14 | 15 |
| **mean** | **19.0** | **13.3** | **10.3** |

Rule of thumb: k=1 for speed and residual-adaptive robustness at moderate
p/n (p ≈ 10³–10⁴ at n=300); grow k roughly with `ln p`; k ≈ 3–7 suffices at
p = 10⁶, n = 300 — still only 6–14 additions per inner product.

## 6. The large-p regime: two failure barriers (analyzed at n=300, p=10⁶)

The race is sequential max-vs-max: as variables enter, the bar a dummy must
clear drops through the **order statistics of the null correlations**
(measured post-signal: 5.0σ at m=1, 3.45σ at m=500, 3.1σ at m=2000). Define
the *entry position* `m_k(L)` = number of null reals that must enter before
the first dummy can win.

**Barrier 1 — budget (dominant, k-independent).** With L = 2000 vs p = 10⁶,
`m(L=2000) ≈ 470–870 for every k` — the top of 2000 draws sits near
3.3–3.5σ regardless of tails; perfect Gaussian dummies die equally. Rule:
**L must scale with p.** T-Rex does not fix L a priori: the L-loop raises
L_max = p, 2p, …, typically up to 10p, and stops the first time the FDP
estimate at dummy count T = 1 exceeds the target. Inside a solver run the
bound is **L_max itself, fixed for the whole run** — every step races the
full budget (generation still stops at the first beater, so the on-demand
economy is preserved). The earlier milestone ladder (`bound = m·p`, m
incrementing while the reals dominate) under-supplied dummies at early
steps and was removed as anti-conservative — FDP-hat underestimation
whenever the L-loop accepted L > p; see the historical note in
trex_sd.md §3/§8. L_max should be given the L-loop's headroom
(≥ p, growing toward ~10p), not a constant.

**Barrier 2 — the k=1 tail ceiling.** The pair null is bounded by
`range(r)/√2 ≈ 2σ̂√(ln n)` (3.8σ here), below the extreme bar
`σ̂√(2 ln p)` (5.0σ). Dead at any budget — confirmed: k=1 with L = 2·10⁶
still admits 51 false reals before dummy #1 (FDP 0.85) and only ever places
3 dummies. k=3 already clears the bar at the same budget (`m ≈ 1`); the
revival run k=7, L = 2·10⁶ stops healthily: **T=1 → 7 steps, 6 TP, 0 FP**.

**Reading note.** TPR < 1 at small T is *correct* here: with SNR 3 spread
over 10 coefficients at p = 10⁶, the weakest signals sit below the ~5σ noise
bar; a calibrated race must refuse them. The k=1 run's higher TPR came
bundled with FDP 0.85.

**On the power criticism (single-path TPR < 1).** The loss decomposes into
three parts. (a) *Intrinsic*: the path positions show signals 8–10 ranked
below dozens of null correlations (k=1 exhibit: greens at positions 12, 15,
~190 in a wall of red) — no FDR-controlling single race can take them
without first admitting the nulls that outrank them; the 7 TP / 0 FP row is
the cleanly separable prefix, i.e. the single-path oracle at FDP 0.
(b) *The selector layer*: single-path demos run ONE random experiment. The
T-Rex selector votes over K experiments with independent dummy draws
(relative occurrence Φ_j ≥ v*), which is exactly the mechanism that recovers
boundary signals — they outrank dummies in most experiments while
exchangeable nulls cannot. Implemented as **TRexSD**
(`trex_selector_methods/trex_sd`, statistical machinery in parity with
TRexVD; inner solver switchable General/Pair/PairGeometric; calibration
modes FixedTL / CalibrateT / CalibrateL / CalibrateBoth — the latter two
run the T-Rex L-loop, L = p, 2p, … until FDP̂(T=1, v=0.75) ≤ tFDR).
Measured at
n=300, p=10⁵: single path T=1 gives TPR 0.7 (FDP 0.125); TRexSD with K=20
gives 7 TP / 1 FP at tFDR=0.1 and 9 TP / 1 FP at tFDR=0.2 with realized
FDP exactly 0.100 — the voting layer converts the target FDR into power.

**Monte Carlo validation** (`sim_trex_sd_mc`, 100 independent trials each —
fresh design, support, noise, and selector seeds; n=300, p=10⁴, K=20,
L=2p). Signal scenario (|S|=10, snr=2, detection boundary):

| method | FDR (target) | TPR |
| --- | --- | --- |
| single path, T=1 | 0.047 ± 0.008 | 0.797 ± 0.018 |
| TRexSD, tFDR=0.1, k=7 | **0.034 ± 0.006** (0.1) | 0.743 ± 0.023 |
| TRexSD, tFDR=0.2, k=7 | **0.149 ± 0.011** (0.2) | **0.949 ± 0.008** |
| TRexSD, tFDR=0.1, k=75 | 0.032 ± 0.006 | 0.747 ± 0.023 |
| TRexSD, tFDR=0.2, k=75 | 0.144 ± 0.010 | 0.954 ± 0.007 |

Near-null scenario (snr=0.01): single path T=1 errs in 52% of trials
(FDR 0.520 ± 0.050); **TRexSD selects the empty set in all 100 trials at
both targets (FDR 0.000)**. Conclusions: (i) FDR is controlled with the
usual T-Rex conservatism margin at every configuration; (ii) the
k=7-vs-k=75 tail concern does not materialize at this scale — results are
statistically identical, so the sparse-dummy null is a valid calibrator at
both sparsities; (iii) the tFDR=0.2 budget buys TPR 0.95 vs 0.80 for the
uncalibrated single path.

**SNR × policy study** (`sim_trex_sd_mc snr`, 100 paired trials per SNR —
same data and seeds for all policies; n=300, p=10³, K=20, L=2p, snr grid
{0.1, 0.2, 0.5, 0.6, 1, 2, 5}). At tFDR=0.1 all three policies control FDR
at every SNR (max 0.079). At tFDR=0.2 the policies separate:

| snr | General (k=7) FDR / TPR | Pair (k=1) FDR / TPR | PairGeometric FDR / TPR |
| --- | --- | --- | --- |
| 0.5 | 0.103 / 0.354 | 0.181 / 0.454 | 0.092 / 0.286 |
| 0.6 | 0.095 / 0.518 | 0.182 / 0.609 | 0.077 / 0.446 |
| 1.0 | 0.153 / 0.896 | **0.221** / 0.906 | 0.093 / 0.832 |
| 2.0 | 0.148 / 1.000 | **0.213** / 1.000 | 0.097 / 0.997 |
| 5.0 | 0.150 / 1.000 | **0.212** / 1.000 | 0.099 / 1.000 |

Findings: (i) **General is the best-calibrated policy** — realized FDR ≈
3/4 of budget, strong power. (ii) **Pair (on-demand, k=1) sits at 0.21–0.22
for snr ≥ 1 — consistently at/above the 0.2 target** (each point ~1–1.4 SE,
but the same direction at three independent SNRs): the predicted k=1 tail
deficit manifests as mild anti-conservatism even at p=10³ where the pair
race is nominally alive — its extra power over General is bought with
budget overdraft. Use k=1 with caution at aggressive tFDR; auto-k is the
fix. (iii) **PairGeometric consumes only ~half the budget** (FDR ≈ 0.09 at
target 0.2) — the exchangeability approximation quantified in FDR terms.
Per unit of *realized* FDR it is frontier-competitive (at snr=1 its
(0.093, 0.832) beats General's interpolated frontier point), so the right
way to run it is at an inflated tFDR. 4,200 selector runs in 46 s. Full 500-run study: under 3 minutes.
(c) *Tunable*: dummy-tail strength sits on the power/FDR frontier — at
T = 10, k=7 gave 9 TP / 8 FP where the calibrated k=75 gave 7 TP / 5 FP;
`m_tol = 2` picks k=3. The auto-calibration default is deliberately at the
conservative end.

## 7. Pre-run calibration: `sd_calibration.hpp`

`sd_calibration::calibrate(X, y, opts)` chooses `(k, rho_d, L)` from the
data before the race:

1. pilot-screen y against its `pilot_screen` most-correlated columns (OLS
   projection — a screening proxy for the post-signal residual);
2. measure the empirical null bars `|x_j^T r|` (screened columns excluded);
3. per candidate k: exact ceiling + MC quantile at `1 − 1/L` (capped at the
   ceiling) → entry position `m_k(L)`;
4. recommend the smallest k with `m_k(L) ≤ m_tol` (default 1) and
   `L = L_factor · p` (default 2).

Cost: two `X^T v` passes, one sort, `mc_draws · |k_grid|` MC samples —
seconds at p = 10⁶. Demo 2's scenario C runs it live; with `m_tol = 1` and
the (slightly signal-contaminated) pilot residual it picks k=75; `m_tol = 2`
would pick k=3. The MC quantile at depth `1 − 1/L` is max-order-statistic
noisy near the feasibility boundary — the ceiling cap and the m-tolerance
absorb this, but treat the boundary k as approximate.

Folded into the solvers: `SD_TLARS_Solver` with `rho_d = 0` auto-calibrates
in the constructor (result readable via `getAutoCalibration()`), and
`L_max = 0` selects the auto budget — the calibration's L when rho_d is
auto, else 2p (SD2 likewise defaults `L_max = 0` to 2p; its k is 1 by
construction).

## 8. Bugs found and fixed along the way

1. **Missing dummy scaling** (blocker): raw ±1 sums made dummies win every
   race under the unit-norm contract → the `||x||/sqrt(2k)` contract of
   Section 1.
2. **`betaPath_` out-of-bounds** (fired under Eigen asserts in the *default*
   demo config; silent heap corruption in Release): dummy generation could
   exceed `L_max` → generation hard-capped at `L_max` (which doubles as the
   fixed dummy-budget semantics); later removed structurally by the sparse
   beta path.
3. **Duplicate-pair collinear stall**: with only `C(n,2)` distinct pairs, a
   duplicate of an active dummy eventually wins, was flagged collinear, and
   the path `break`-ed permanently → rollback + discard-and-continue (each
   discard permanently removes one candidate; no livelock). The classic
   TLARS run hits the same birthday effect (`Variable 10208 collinear;
   dropped`).
4. **Clang OpenMP** rejects capturing structured bindings → named locals in
   the parallel correlation update.
5. **Non-seedable RNG** (`random_device` only) → optional `seed` parameter
   (0 keeps the old behavior); the demos and the SD/SD2 cross-validation
   depend on it.

Verification habit that caught (2): **assert-enabled scratch builds** (no
`-DNDEBUG`) next to the Release demos — Release silently survives
corruption that debug Eigen catches.

## 9. Demos

- `demos/cpp/demo_sd_tlars.cpp` — Demo 1: early stopping + contract
  equivalence (unit-L2 vs z-score; identical paths, exact λ ratio).
- `demos/cpp/demo_sd2_tlars.cpp` — Demo 2:
  - A (n=300, p=10³): SD vs SD2 path identity + policy comparison,
  - B (n=2000, p=500): timing vs classic explicit-D TLARS,
  - C (n=300, p=10⁶): live calibration, calibrated race (T=1: 7 TP, 0 FP),
    and the k=1 boundary exhibit at the same L = 2·10⁶.

- `demos/cpp/demo_trex_sd.cpp` — Demo 3: the K-experiment TRexSD selector at
  the detection boundary (n=300, p=10⁵), single-path baseline vs calibrated
  (v*, T*) selection at tFDR ∈ {0.1, 0.2}; ≈ 11 s per target.
- `demos/cpp/sim_trex_sd_mc.cpp` — Monte Carlo study
  (`sim_trex_sd_mc [signal|null] [trials] [rho_d]`): FDR/power over
  independent trials; results in Section 6.

  Colored paths: green = true positive, red = false positive, blue = dummy;
  demo 2 prints one path snapshot per T stop.

## 10. Open questions and next steps

- **FDP estimator and the adaptive pool — mostly resolved.** Adaptive L is
  already part of validated T-Rex practice (the L-loop, above), and the
  on-demand pool is step-locally *equivalent* to a pre-filled fixed-bound
  race: a real wins a step iff the pool reached its full bound with no
  member beating the best real — exactly the fixed-L admission condition —
  and early generation stop only skips draws that could not flip that
  step's real-vs-dummy outcome. Since the T-counter and the FDP estimator
  consume only this admission sequence, no refinement appears necessary for
  the OnDemand policy. The one residual wedge is **winner identity**: given
  a dummy win, on-demand admits the *first beater* in stream order, a
  pre-filled race admits the *block max*; the admitted column joins the
  active set and reshapes downstream geometry, so the equivalence is
  per-step, not path-global. Conjectured second-order; check it inside the
  K-experiment study below. The Geometric policy's frozen-failure re-racing
  remains a genuine approximation (measured: conservative).
- **Policy calibration study — done** (SNR × policy MC, Section 6):
  Geometric's conservatism is quantified (~half the FDR budget), General is
  well calibrated, and Pair (k=1) shows the predicted mild anti-conservatism
  at snr ≥ 1 / tFDR = 0.2. Remaining refinement: confirm the Pair overdraft
  with more trials and map the frontier at matched *realized* FDR.
- **O(k) generation and distinct sampling (Floyd)** — two related upgrades:
  (i) within a dummy, Floyd's subset sampling draws 2k distinct indices in
  exactly 2k RNG draws (no O(n) iota/alloc as in the current partial
  Fisher–Yates), plus an O(k) shuffle for the uniform P/M split;
  (ii) across dummies at k = 1, enumerate pairs by triangular code
  m ∈ {0..C(n,2)−1} and Floyd-sample distinct codes — L distinct pairs in
  O(L), no birthday duplicates, hence no dummy-vs-dummy collinear discards
  and an honest exhaustion boundary at L = C(n,2). Statistically this swaps
  iid dummies for a simple random sample without replacement
  (exchangeability retained; the Geometric policy's draw count becomes
  hypergeometric). For k ≥ 3 the universe is astronomically large, so (ii)
  is a k = 1 concern only.
- **Family growth**: SD-OMP / SD-AFS analogues of the VD family.
