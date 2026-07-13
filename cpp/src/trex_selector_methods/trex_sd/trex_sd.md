# T-Rex-SD: The Sparse-Dummy T-Rex Selector

Consolidated method notes — solver family (`tsolvers/sd_tsolvers/`) and
selector (`trex_selector_methods/trex_sd/`). **Living document**: the method
is not finished; Section 9 lists the open analysis and improvement
directions. Status: 2026-07-12. Earlier engineering notes:
`tsolvers/sd_tsolvers/sd_tsolver.md`.

---

## 1. Setting and notation

Linear model $y = X\beta + \varepsilon$ with $X \in \mathbb{R}^{n \times p}$,
sparse support $\mathcal{S} = \{j : \beta_j \neq 0\}$, $|\mathcal{S}| \ll p$,
typically $p \gg n$. Goal: select $\hat{\mathcal{S}}$ with
$\mathrm{FDR} = \mathbb{E}[\mathrm{FDP}] = \mathbb{E}\big[\tfrac{|\hat{\mathcal{S}} \setminus \mathcal{S}|}{\max(1, |\hat{\mathcal{S}}|)}\big] \le \alpha$
while maximizing $\mathrm{TPR} = \mathbb{E}\big[\tfrac{|\hat{\mathcal{S}} \cap \mathcal{S}|}{|\mathcal{S}|}\big]$.

**Preprocessing contract**: columns of $X$ centered with a common L2 norm
$\|x\|$ (unit-L2 canonical, matching the VD family; z-score allowed), $y$
centered. Consequence: every residual $r = y - X\hat\beta$ along the path is
exactly centered, $\mathbf{1}^\top r = 0$.

The T-Rex principle (Machkour, Muma, Palomar): augment $X$ with $L$ computer-
generated null ("dummy") variables, run a forward selector on $[X, D]$, stop
when $T$ dummies have entered, and calibrate $(v^*, T^*)$ from $K$ random
experiments so that an FDP estimate stays below $\alpha$. SD replaces the
explicit $D$ with **sparse balanced Rademacher dummies that never exist as
columns**.

## 2. The sparse dummy law

**Definition (SD$_k$ dummy).** Draw disjoint index sets
$P, M \subset \{1,\dots,n\}$ with $|P| = |M| = k$ uniformly at random and set

$$d \;=\; s\Big(\sum_{i \in P} e_i \;-\; \sum_{i \in M} e_i\Big),
\qquad s \;=\; \frac{\|x\|}{\sqrt{2k}} .$$

This is the package's `ConstrainedSparseRademacher(s = \rho_d)` with
$2k = 2\lfloor \rho_d n / 2 \rfloor$ clamped to $[2, n]$, plus the mandatory
fair-race scale $s$. A dummy is represented by $(P, M)$ only; at $k = 1$ by a
single integer pair.

**Exact structural facts.**

1. $\mathbf{1}^\top d = 0$ and $\|d\|_2 = \|x\|$ — *exactly*, not in
   expectation. Every dummy is automatically a legal competitor under the
   contract.
2. $\mathbb{E}[d] = 0$ and, for **every** $k$,
   $$\mathbb{E}[d d^\top] \;=\; \frac{\|x\|^2}{n-1}\Big(I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top\Big).$$
   (Symmetry forces $\alpha I + \beta \mathbf{1}\mathbf{1}^\top$; the exact
   zero column sum forces the row sums to zero; the exact norm fixes the
   trace.) These are precisely the first two moments of the VD family's
   spherical law on the centered subspace: **the SD ensemble is
   second-moment-identical to VD for every sparsity**.
3. **Variance identity.** For any fixed centered $r$,
   $$\operatorname{Var}(d^\top r) \;=\; r^\top \mathbb{E}[dd^\top]\, r
   \;=\; \frac{\|x\|^2 \|r\|^2}{n-1}, \qquad \text{for every } k,$$
   exactly the second moment of $\tilde{x}^\top r$ for a uniformly random
   direction $\tilde{x}$ of norm $\|x\|$ in the centered subspace. The
   sparsity $k$ therefore changes **only the higher moments** — the tails —
   which is exactly where the selection race lives.
4. **Support bound (ceiling).** With $r_{(1)} \le \dots \le r_{(n)}$ the
   order statistics of the residual,
   $$\max_{d \in \mathrm{SD}_k} |d^\top r|
   \;=\; \frac{\|x\|}{\sqrt{2k}} \Big(\sum_{i=1}^{k} r_{(n-i+1)} - \sum_{i=1}^{k} r_{(i)}\Big).$$
   Non-monotone in $k$ (at $n = 300$ it peaks near $k \approx n/4$). At
   $k=1$: $\|x\|\,\mathrm{range}(r)/\sqrt{2} \approx 2\hat\sigma\|x\|\sqrt{\ln n}$.
5. **Tail character.** $k = 1$: the null of $d^\top r$ is the pairwise
   difference distribution of the actual residuals (the Gini mean-difference
   kernel) — nonparametric, permutation-flavored, inherits heavy tails of
   the true noise; bounded, discrete support over $\binom{n}{2}$ atoms.
   Growing $k$: CLT pushes the null toward Gaussian; the reach grows.

**Scale equivariance of the contract.** Unit-L2 and z-scored preprocessing
differ by the global constant $\sqrt{n-1}$ applied to reals and dummies
alike, so the selection path is identical (verified bit-for-bit; $\lambda$
ratio exactly $\sqrt{n-1}$).

## 3. SD-TLARS: the terminating race with implicit dummies

Standard LARS over the augmented design, with active set $\mathcal{A}$,
correlations $c_j = z_j^\top r$ ($z_j$ a real column or dummy), Cholesky
factor $R$ of $G_\mathcal{A} = Z_\mathcal{A}^\top Z_\mathcal{A}$, sign vector
$\mathbf{s}$, and

$$w_\mathcal{A} = A_\mathcal{A} G_\mathcal{A}^{-1} \mathbf{s}, \quad
A_\mathcal{A} = \big(\mathbf{s}^\top G_\mathcal{A}^{-1} \mathbf{s}\big)^{-1/2}, \quad
u = Z_\mathcal{A} w_\mathcal{A},$$

$$\gamma \;=\; {\min_{j \notin \mathcal{A}}}^{+}
\left\{ \frac{C - c_j}{A_\mathcal{A} - a_j},\; \frac{C + c_j}{A_\mathcal{A} + a_j} \right\},
\qquad a_j = z_j^\top u,$$

then $r \leftarrow r - \gamma u$ and the **exact incremental updates**
$c_j \leftarrow c_j - \gamma a_j$ for all inactive candidates (reals and
pool dummies alike — the $a_j$ are already computed by the $\gamma$ search).
The path stops when $T$ dummies have been admitted.

**Dummy arithmetic is dot-free.** For any vector $v$:
$d^\top v = s(\sum_{P} v_i - \sum_{M} v_i)$ — $2k$ additions. At $k = 1$
everything collapses to $O(1)$:
$d^\top v = s(v_i - v_j)$, cross-products against a real active column
$d^\top x_a = s(x_a[i] - x_a[j])$, and against another pair dummy
$d_1^\top d_2 = s^2(\delta_{i_1 i_2} + \delta_{j_1 j_2} - \delta_{i_1 j_2} - \delta_{j_1 i_2})$.
No dummy column is ever materialized in the pair solver; an active dummy is
its $(i, j)$.

**Two implementations, one algorithm** (`sd_tlars_solver` general $k$;
`sd2_tlars_solver` pair-specialized): for the same seed both consume the
identical RNG stream and produce identical selection paths — the standing
cross-validation harness.

**On-demand pool and the $L$ semantics (fixed-L race).** At each step the
pool of candidate dummies is expanded one draw at a time while
$\max_{\text{pool}} |c| \le \max_{\text{candidate reals}} |c|$; the **only**
cap is the total budget $L_{\max}$. Generated dummies stand in the pool with
exactly maintained correlations, so every step races against (effectively)
all $L_{\max}$ dummies — the same semantics as the classic pre-filled
$n \times L$ dummy matrix.

**Proposition (step-local equivalence).** *A real variable is admitted at
step $t$ iff the full budget contains no member beating the best real —
which is exactly the admission condition of the pre-filled fixed-$L$ race.
Early generation stop skips only draws that could not change the step's
real-vs-dummy admission indicator.* Since the $T$-counter and the FDP
estimator consume only this indicator sequence, on-demand generation is not
an approximation at the level the selector sees. The residual gap is
**winner identity**: given a dummy win, on-demand admits the *first beater*
in stream order (stochastically smaller than the pre-filled race's *block
max*); the admitted column reshapes the downstream LARS geometry, so the
equivalence is per-step, not path-global. Conjectured second-order; open
item 9.2.

**Historical note (the milestone-ladder bug).** The original transition
additionally bounded the pool by $B_t = \min(m_t\,p, L_{\max})$ with a
milestone $m_t$ that grew only as the pool lost races — intended as an
"in-solver analogue of the T-Rex L-loop." That conflated two levels: with
the *real* L-loop running in TRexSD, the ladder made early steps race
against far fewer dummies than the $L$ that both the L-loop's acceptance
decision and the $\Phi'$ scaling assume, which is **anti-conservative
exactly when the calibrated $L > p$**. The empirical signature matched
perfectly (near-null FDR 0.12–0.15 at target 0.1 wherever $\bar L > p$;
controlled wherever $\bar L = p$), and removing the ladder reconciled SD
with the classic explicit-$D$ reference (Section 8).

**The greedy family: SD-TOMP and SD-TAFS.** The same race — same dummy law,
same on-demand pool with bound $B_t$ and budget $L_{\max}$ — admits greedy
selection rules in place of the equiangular path (mirroring the VD family's
`vd_omp` / `vd_afs`). **OMP**: the argmax of $|z_j^\top r|$ over inactive
reals and the pool enters $\mathcal{A}$, and the coefficients are the full
OLS refit $\beta_\mathcal{A} = G_\mathcal{A}^{-1} Z_\mathcal{A}^\top y$
(Cholesky append + two triangular solves), $r = y - Z_\mathcal{A}
\beta_\mathcal{A}$. **AFS** with blend $\rho \in (0, 1]$: candidates include
*already-active* features (re-selection blends, never appends, and never
re-counts a dummy toward $T$), and with $\nu = G_\mathcal{A}^{-1}
Z_\mathcal{A}^\top y$ the fit moves fractionally,

$$\mu \leftarrow (1 - \rho)\,\mu + \rho\, Z_\mathcal{A} \nu, \qquad
\beta_\mathcal{A} \leftarrow (1 - \rho)\,\beta_\mathcal{A} + \rho\,\nu .$$

$\rho = 1$ reduces AFS to OMP *exactly* (after a full refit
$r \perp \mathrm{span}(Z_\mathcal{A})$, so re-selection never fires;
validated bitwise, general and pair, both policies); $\rho \to 0$ approaches
LARS-like behavior. Three structural simplifications relative to LARS:
(i) the refit re-orthogonalizes $r$, so pool correlations are *recomputed*
from $r$ each step — $O(k)$ index sums per dummy, no incremental
$c \mathrel{-}= \gamma a$ bookkeeping and no drift; (ii) duplicate/collinear
pool dummies are structurally harmless in OMP ($r \perp$ active span
$\Rightarrow$ their correlation is exactly zero — the LARS rollback
machinery is a dead guard); (iii) there is no $\gamma$-search, so the
per-step cost is one GEMV $X^\top r$ plus the pool scan. The pair twins
(`sd2_tomp_solver`, `sd2_tafs_solver`) replicate the general solvers' RNG
stream at $k = 1$ (bit-identical paths, validated) and support both
`SD2GenPolicy` values; under Geometric the beating set is rebuilt per step
against the refit residual.

## 4. The exact pair null and the Geometric policy ($k = 1$)

The $k=1$ universe is finite: $\binom{n}{2}$ pairs. Sorting $r$ once
($O(n \log n)$) yields, by a two-pointer pass, the exact **beating
fraction**

$$\pi(c) \;=\; \frac{\#\{(i,j) : |r_i - r_j| > c / s\}}{\binom{n}{2}},$$

so the on-demand generation count until the first success is
$\mathrm{Geometric}(\pi)$ — sampled directly instead of looped — and the
winning pair is a uniform draw from the beating set (prefix-rank inversion,
$O(\log n)$). Even a fixed-$L$ race max is samplable by inverse transform
from $F^L$ without instantiating any dummy.

**`SD2GenPolicy::Geometric`** exploits this: failures are never
instantiated — only the virtual pool size $F$ is tracked. Per step: the
standing pool wins with probability $1 - (1-\pi)^F$; otherwise a fresh
Geometric draw runs against the remaining budget. Two approximations,
deliberately accepted and measured: (i) frozen failures are re-raced as
*fresh* draws against the current residual (their historical conditioning
$|c_\tau(d)| \le$ threshold$_\tau$ is ignored — exchangeability
approximation); (ii) virtual dummies contribute no constraint to the
$\gamma$ step. Net effect, quantified in Section 8: the policy consumes
about **half** the FDR budget.

## 5. Choosing $k$ and $L$: the two failure barriers

The race is sequential max-vs-max: as variables enter, the bar a dummy must
clear drops through the **order statistics of the null correlations**,
empirically $b_m \approx \hat\sigma \sqrt{2 \ln(p/m)}$ with
$\hat\sigma = \|r\|/\sqrt{n}$. Define the dummy **reach** at budget $L$ as
$q_k(L) = \min\big(F_{D_k}^{-1}(1 - 1/L),\ \text{ceiling}_k\big)$ and the
**entry position**

$$m_k(L) \;=\; \#\{m : b_m > q_k(L)\}$$

— the number of null reals that must enter before the first dummy can win.

- **Budget barrier ($k$-independent).** For same-law dummies the first dummy
  enters at $m \approx p / L$. $L$ must scale with $p$; $L \ll p$ starves
  any dummy law (measured at $p = 10^6$, $L = 2000$: $m \approx 500$ for
  every $k$).
- **Tail barrier ($k = 1$).** The pair ceiling
  $\approx 2\hat\sigma\sqrt{\ln n}$ falls below the extreme bar
  $\hat\sigma\sqrt{2 \ln p}$ once $\ln p \gtrsim 2 \ln n$; then pairs are
  dead at *any* budget (measured at $p = 10^6$: 51 false reals before the
  first dummy even with $L = 2\cdot10^6$). Rule of thumb: $k$ grows
  logarithmically with $p$; $k \approx 3$–$7$ suffices at $p = 10^6$,
  $n = 300$.

**Pre-run calibration** (`sd_calibration::calibrate`): pilot-screen $y$
against its strongest columns (OLS projection ≈ post-signal residual),
measure the empirical bars $b_m$, compute per candidate $k$ the exact
ceiling + MC quantile $q_k(L)$, and recommend the smallest $k$ with
$m_k(L) \le m_{\mathrm{tol}}$. Folded into the solver constructors
(`rho_d = 0`, `L_max = 0` → auto). The **L-loop** (selector-level,
`CalibrateL` / `CalibrateBoth`): $L = p, 2p, \dots$ until
$\widehat{\mathrm{FDP}}(T{=}1, v{=}0.75) \le \alpha$.

## 6. TRexSD: the K-experiment selector

$K$ independent experiments (fresh dummy streams), each a terminating SD
race. For dummy count $T$, the **relative occurrence** of variable $j$ is
$\Phi_j(T) = \frac{1}{K} \sum_{\kappa=1}^{K} \mathbb{1}\big[j \in \hat{\mathcal{A}}_\kappa(T)\big]$.
With $\phi_{j,t} = \Phi_j(t)$, $a_t = \sum_j \phi_{j,t}$,
$\delta_t = \sum_{j: \Phi_j > 1/2} \phi_{j,t}$, and the step differences
$\tilde\delta_t$, $\tilde\phi_{j,t}$, the estimator (parity with TRexVD /
the T-Rex reference, incl. the $[0,1]$ clamp) is

$$\Phi_j' \;=\; \mathrm{clip}_{[0,1]}\Big( \sum_{t \le T} \tilde\phi_{j,t}\,
\Big[ 1 - \frac{(p - a_t) / (L - t + 1)}{\tilde\delta_t} \Big]_{\tilde\delta_t > \epsilon} \Big),$$

$$\widehat{\mathrm{FDP}}(v, T) \;=\; \min\Big(1,\;
\frac{\sum_{j : \Phi_j(T) > v} \big(1 - \Phi_j'\big)}{\#\{j : \Phi_j(T) > v\}}\Big).$$

Selection maximizes $|\hat{\mathcal{S}}|$ over the grid
$(T, v) \in \{1..T_{\mathrm{stop}}\} \times V$,
$V = \{0.5, 0.5 + \tfrac{1}{K}, \dots, 1 - \epsilon\}$, subject to
$\widehat{\mathrm{FDP}}(v, T) \le \alpha$; then
$\hat{\mathcal{S}} = \{j : \Phi_j(T^*) > v^*\}$. Execution modes: fixed-T,
posthoc, and strided early-stop (stop when
$\widehat{\mathrm{FDP}}(v_{\max}) > \alpha$, with a dummy-burn guard).
Two orthogonal inner-solver switches: the dummy law (`General` / `Pair` /
`PairGeometric`) and the selection algorithm (`LARS` / `OMP` / `AFS` with
blend `rho`, default 0.3) — nine combinations through one factory; every
execution mode and the L-loop inherit them.

## 7. Computational profile

| | classic TLARS (explicit $D$) | SD general $k$ | SD2 pair |
| --- | --- | --- | --- |
| dummy inner product | $O(n)$ dot | $2k$ adds | 1 subtraction |
| generation | build $n \times L$ matrix | $O(n)$/dummy | $O(1)$/dummy |
| dummy storage | $8nL$ bytes | $\approx 8k$ B/dummy | 24 B/dummy |
| measured (n=2000, p=500, L=10⁴, T=30) | ≈ 150 ms / 153 MiB | ≈ 20 ms | ≈ 11–16 ms |

At the T-Rex convention $L \propto p$ with $p = 10^6$, explicit $D$ is
physically impossible (24 TB at $L = 10p$); SD2 holds the same race in tens
of MB. The beta path is stored sparsely ($O(\text{steps}^2)$, actives-only),
and pool correlations are maintained incrementally — the per-step dummy cost
beyond the $\gamma$-search sums is an $O(|\text{pool}|)$ scan. The remaining
per-step cost is the irreducible real-feature work $X^\top u$, shared by
every T-Rex variant.

## 8. Empirical validation (Monte Carlo, `sim_trex_sd_mc`)

All studies: 100 independent trials (fresh design, support, noise, selector
seeds), $K = 20$, $L = 2p$. Full details and additional single-run studies
in `sd_tsolver.md`.

**Signal scenario** ($n=300$, $p=10^4$, $|\mathcal{S}|=10$, snr=2):

| method | FDR (target) | TPR |
| --- | --- | --- |
| single path, T=1 | 0.047 ± 0.008 | 0.797 ± 0.018 |
| TRexSD, α=0.1, k=7 | 0.034 ± 0.006 | 0.743 ± 0.023 |
| TRexSD, α=0.2, k=7 | 0.149 ± 0.011 | 0.949 ± 0.008 |
| TRexSD, α=0.1, k=75 | 0.032 ± 0.006 | 0.747 ± 0.023 |
| TRexSD, α=0.2, k=75 | 0.144 ± 0.010 | 0.954 ± 0.007 |

**Near-null** (snr=0.01): single path errs in 52% of trials; TRexSD selects
the empty set in 100/100 trials at both targets.

**SNR × policy** ($p = 10^3$, paired trials, grid {0.1, 0.2, 0.5, 0.6, 1,
2, 5}): at α=0.1 all policies control FDR everywhere (max 0.079). At α=0.2:

| snr | General (k=7) FDR / TPR | Pair (k=1) FDR / TPR | PairGeometric FDR / TPR |
| --- | --- | --- | --- |
| 0.5 | 0.103 / 0.354 | 0.181 / 0.454 | 0.092 / 0.286 |
| 0.6 | 0.095 / 0.518 | 0.182 / 0.609 | 0.077 / 0.446 |
| 1.0 | 0.153 / 0.896 | **0.221** / 0.906 | 0.093 / 0.832 |
| 2.0 | 0.148 / 1.000 | **0.213** / 1.000 | 0.097 / 0.997 |
| 5.0 | 0.150 / 1.000 | **0.212** / 1.000 | 0.099 / 1.000 |

Readings: **General is best-calibrated** (≈ 3/4 of budget). **Pair (k=1)
consistently overdraws at α=0.2, snr ≥ 1** — each point only ~1–1.4 SE
above target, but the same direction at three independent SNRs: the
predicted tail deficit as mild anti-conservatism. **PairGeometric consumes
≈ half the budget** yet is frontier-competitive per unit of *realized* FDR
(at snr=1 its (0.093, 0.832) beats General's interpolated frontier) — run
it at an inflated α.

**The same study at $p = 10^4$ settles the question decisively.** What was
a ~1 SE overdraft at $p = 10^3$ becomes a catastrophic FDR violation one
decade up (α = 0.2 shown; α = 0.1 violates proportionally):

| snr | General (k=7) FDR / TPR | Pair (k=1) FDR / TPR | PairGeometric FDR / TPR |
| --- | --- | --- | --- |
| 0.5 | 0.015 / 0.042 | **0.296** / 0.263 | **0.222** / 0.194 |
| 1.0 | 0.130 / 0.523 | **0.605** / 0.807 | **0.443** / 0.726 |
| 2.0 | 0.151 / 0.952 | **0.709** / 0.993 | **0.543** / 0.991 |
| 5.0 | 0.154 / 1.000 | **0.725** / 1.000 | **0.530** / 1.000 |

General (k=7) remains controlled at every SNR and both targets (max 0.154);
both pair policies run at 2–3.5× the target — their apparent power is
FDR overdraft, and Geometric's conservatism only dampens, not repairs, the
failure. The mechanism (Section 9.1, now confirmed): at $p = 10^4$,
$n = 300$ the pair null still *reaches* above the extreme bar (the ceiling
$4.8\hat\sigma$ exceeds $b_1 \approx 4.3\hat\sigma$, so the race nominally
runs) but its top quantiles are rare atoms — dummies are severely
under-represented among the extreme order statistics, so dummy admissions
lag null-real admissions and $\widehat{\mathrm{FDP}}$ underestimates
grossly. **FDR validity therefore requires tail *matching* throughout the
selection region — a strictly stronger condition than the race-alive /
reach criterion $m_k(L) \le m_{tol}$ that `sd_calibration` tests.** Note
the auto-$k$ machinery already rejects $k=1$ at this scale (it picks
$k = 7$); the pair policies in this study bypassed calibration by
construction. Practical rule: the pair solver is a *computational* device
and a valid *statistical* calibrator only at moderate $p/n$ — beyond that,
auto-$k$ is mandatory, not optional.

**SNR × algorithm** ($p = 10^4$, paired trials, same grid and seeds;
General $k = 7$, AFS $\rho = 0.3$). This comparison is `sim_trex_sd_mc`'s
*default* study (no mode argument; the recorded run additionally carried
paired Pair × {LARS, OMP} columns for the one-off rescue question below).
The greedy solvers were added on the standing observation that OMP/AFS
handle variable selection and FDR control better than the LARS machinery —
the MC confirms it as strict dominance. At $\alpha = 0.2$ (FDR / TPR):

| snr | Gen-LARS | Gen-OMP | Gen-AFS | Pair-LARS | Pair-OMP |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.015 / 0.042 | 0.006 / 0.048 | 0.014 / 0.056 | **0.296** / 0.263 | 0.186 / 0.180 |
| 0.6 | 0.045 / 0.116 | 0.023 / 0.104 | 0.035 / 0.111 | **0.346** / 0.367 | 0.176 / 0.254 |
| 1.0 | 0.130 / 0.523 | 0.045 / 0.649 | 0.072 / 0.679 | **0.605** / 0.807 | **0.204** / 0.766 |
| 2.0 | 0.151 / 0.952 | 0.061 / 0.994 | 0.097 / 0.997 | **0.709** / 0.993 | **0.225** / 0.996 |
| 5.0 | 0.154 / 1.000 | 0.064 / 1.000 | 0.102 / 1.000 | **0.725** / 1.000 | **0.260** / 1.000 |

At $\alpha = 0.1$ the ordering is identical (Gen: LARS ≤ 0.045, OMP ≤ 0.028,
AFS ≤ 0.032, all controlled; Pair-OMP violates mildly, up to 0.142).
Readings:

1. **Under the General law, OMP and AFS strictly dominate LARS**: lower
   FDR *and* higher TPR simultaneously, at every SNR where they differ.
   The largest power gaps sit at the detection boundary — snr = 1,
   $\alpha = 0.1$: TPR 0.31 (OMP) vs 0.12 (LARS), a factor 2.5; snr = 2:
   0.98 vs 0.78. AFS($\rho = 0.3$) tracks OMP closely (slightly more power
   at snr $\ge$ 1, slightly more budget).
2. **Mechanism**: the full refit removes LARS's shrinkage, so entered
   signal is fully subtracted and the voting profiles sharpen — true
   features reach $\Phi \approx 1$ within a few $T$, nulls stay low, and
   the $(T, v)$ calibration stops at mean $T^* \approx 5$ (OMP/AFS) vs
   8+ (Gen-LARS) and 39 (Pair-LARS, dummy burn). Sharper $\Phi$ separation
   buys both effects at once, which is why there is no power/validity
   trade-off.
3. **The pair deficit is algorithm-independent** (the rescue question,
   answered): OMP compresses the pair violation about 3× (0.725 → 0.260 at
   $\alpha = 0.2$, snr = 5) by shrinking the selection region the tail
   deficit acts on, but a systematic residual violation remains and grows
   with SNR (many SE at snr ≥ 1). No selection algorithm substitutes for
   tail matching of the dummy law — Section 9.1 stands as the top item.
4. Practical default going forward: **General law + OMP** (or AFS at a
   modest $\rho$ when path smoothness is wanted); LARS remains the
   reference/parity implementation.

**⚠ Parametrization correction (supersedes the readings above).** All
studies above ran with a *non-classic parametrization*: fixed $L = 2p$
(no L-loop), the dummy-burn stagnation guard applied to **all** algorithms
with window 3, and an L-loop ceiling of 50p. Classic T-Rex
(`trex_core::TRexControlParameter`) instead runs the adaptive L-loop
($L = p, 2p, \dots \le 10p$, smallest valid $L$ = power-optimal), disables
the stagnation guard for LARS-path solvers (R-reference/paper parity;
enabled only for greedy solvers, window 5), and caps at $10p$. TRexSD now
defaults to classic parity (`CalibrateBoth`, tri-state `stagnation_stop`,
`max_stale_strides = 5`, `max_L_factor = 10`). A 50-trial re-validation at
$p = 10^4$, $\alpha = 0.1$ (TPR, old → corrected):

| snr | LARS | OMP | AFS(0.3) |
| --- | --- | --- | --- |
| 0.5 | 0.003 → **0.148** | 0.000 → **0.100** | 0.000 → **0.128** |
| 0.6 | 0.003 → **0.180** | 0.008 → **0.158** | 0.005 → **0.176** |
| 1.0 | 0.113 → **0.438** | 0.308 → **0.474** | 0.284 → **0.520** |
| 2.0 | 0.805 → 0.822 | 0.983 → 0.998 | 0.986 → 0.994 |

FDR stays controlled at both targets for snr ≥ 0.5 (OMP/AFS with roughly
half of LARS's realized FDR); a mild near-null overdraft at snr ≤ 0.2,
$\alpha = 0.2$ (0.18–0.29, ~1–1.5 SE at 50 trials, tiny selections) needs
the 200-trial run.

**Reconciliation with the classic reference (the milestone-ladder fix,
Section 3).** A 200-trial run of the variable-support MC
(`demo_trex_sd_mc_variable_support`, $n = 300$, $p = 1000$,
$|\mathcal{S}| = 10$, $\alpha = 0.1$, random support per trial) exposed a
near-null FDR overdraft (0.12–0.15) that the classic TLARS with an
*explicit* `ConstrainedSparseRademacher(0.1)` dummy matrix
(demo_trex_05) does not have (0.01–0.02) — same dummy law family, so the
transition, not the law, was at fault. The culprit was the milestone
ladder (Section 3); after its removal SD-LARS reproduces the classic
explicit-$D$ selector almost number for number:

| snr | SD-LARS FDR / TPR (fixed) | classic SparseRad FDR / TPR | SD-LARS $\bar L/p$ / classic |
| --- | --- | --- | --- |
| 0.1 | 0.000 / 0.003 | 0.010 / 0.002 | 2.62 / 2.66 |
| 0.2 | 0.008 / 0.022 | 0.020 / 0.023 | 4.71 / 4.68 |
| 0.5 | 0.041 / 0.274 | 0.039 / 0.268 | 6.87 / 6.68 |
| 0.6 | 0.047 / 0.358 | 0.044 / 0.369 | 6.15 / 5.92 |
| 1.0 | 0.045 / 0.752 | 0.049 / 0.753 | 3.71 / 3.67 |
| 2.0 | 0.035 / 0.979 | 0.041 / 0.979 | 1.91 / 1.89 |
| 5.0 | 0.024 / 1.000 | 0.030 / 1.000 | 1.13 / 1.22 |

This is an end-to-end validation of the SD virtualization against the
reference implementation: FDR, TPR, and even the L-loop's accepted budgets
coincide. OMP/AFS on the same run remain uniformly controlled
(FDR ≤ 0.036) and add their power edge at snr ≥ 1 (0.88 vs 0.75 at
snr = 1) while conceding a little at snr 0.5–0.6.

**SD vs VD at scale (the "alternative" question, answered).** The same MC
at $p = 10^4$ (200 trials, `demo_trex_sd_mc_variable_support`, paired data,
$\alpha = 0.1$; TRexVD with its canonical spherical-Gaussian defaults)
shows **statistical parity of the sparse law (auto-$k$ regime, $k = 7$)
with the Gaussian virtual-dummy reference, per algorithm** — all
differences within MC noise (SE ≈ 0.02–0.03), SD nominally ahead at the
boundary:

| snr = 1 | FDR | TPR |  | snr = 2 | TPR |
| --- | --- | --- | --- | --- | --- |
| SD-LARS / VD-LARS | 0.055 / 0.039 | 0.387 / 0.369 | | | 0.829 / 0.835 |
| SD-OMP / VD-OMP | 0.019 / 0.024 | 0.409 / 0.408 | | | 0.992 / 0.996 |
| SD-AFS / VD-AFS | 0.027 / 0.026 | 0.483 / 0.470 | | | 0.995 / 0.996 |

Even the L-loop's accepted budgets track each other across the grid
(e.g. $\bar L/p$ at snr = 0.6: 6.50 vs 6.49). Three consequences:
(i) the power drop from $p = 10^3$ to $10^4$ at fixed $n$ is
**regime-intrinsic** (the $\sqrt{2\log p}$ bar), not an SD deficiency;
(ii) the conservatism margin (realized FDR 0.02–0.05 against the 0.1
budget) is **shared by the Gaussian law** — it belongs to the T-Rex
$\Phi'$/FDP machinery in this regime, not to the sparse tails, which
demotes tail-matching as a *power* lever (it remains the *validity* fix
for the pair law); (iii) AFS($\rho = 0.3$) is the boundary-power leader in
*both* families. Together with the explicit-$D$ reconciliation above, SD
is now validated against both reference implementations — parity in
statistics, decisive advantage in compute and memory. Revised readings: (a) the earlier "devastating power
loss of the General law" was chiefly a **parametrization artifact**, not a
property of the sparse dummy law — the L-loop adapts $L/p$ from 1 (high
snr) to ~4.6 (boundary) and recovers boundary power 50×; (b) "OMP strictly
dominates LARS" softens to: **unguarded LARS leads slightly at
snr ≤ 0.6** (deep-$T$ exploration, $T^* \approx 28$ vs 6.5), **OMP/AFS
lead at snr ≥ 1** with lower FDR and $T^* \approx 7$; (c) the pair-law
tables above were likewise produced under fixed $L = 2p$ — the tail-deficit
mechanism stands (it is a law property), but the violation magnitudes need
re-validation under the L-loop before being quoted.

## 9. Open problems and improvement directions

The method is not finished. Ordered by expected value:

1. **Pair anti-conservatism — CONFIRMED at $p = 10^4$** (Section 8): FDR
   2–3.5× target for both pair policies while General (k=7) stays
   controlled; the effect is many SE, no further trials needed. Remaining
   tasks: (a) formalize the tail-matching condition ⇒ conservative
   $\widehat{\mathrm{FDP}}$ (the reach criterion $m_k(L) \le m_{tol}$ is
   provably insufficient — quantile matching over the whole selection
   region is needed); (b) fold that stronger criterion into
   `sd_calibration` (match the dummy quantile curve against the empirical
   bars $b_m$ for all relevant $m$, not just the entry position); (c) test
   whether the L-loop (`CalibrateBoth`) can partially repair pairs by
   inflating $L$ — likely not, since the ceiling, not the budget, binds.
   The SNR × algorithm study (Section 8) adds an algorithm-independence
   confirmation: switching the selection rule to OMP compresses the
   violation ~3× but cannot remove it — the deficit lives in the dummy
   law's tail, not in the path geometry.
2. **Winner identity.** Quantify the first-beater vs block-max gap with
   paired runs on identical streams; expected second-order.
3. **Exact Geometric policy.** For $k=1$ the frozen pool's exact
   conditional law is *trackable*: the universe has only $\binom{n}{2}$
   atoms, and each past failure constraint
   $|r_\tau(i) - r_\tau(j)| \le c_\tau / s$ partitions it. Maintaining
   per-atom alive/multiplicity state is $O(n^2)$ memory ($\approx$ 45k
   atoms at $n=300$) — an exact, approximation-free virtual pool that
   would remove the exchangeability approximation entirely. This would be
   the true SD analogue of VD's exact sufficient statistics.
4. **Moment-matched dummies.** $k$ (or a mixture over $k$) chosen so the
   dummy null matches the *empirical* null-correlation distribution beyond
   the (already exact) second moment — quantile matching against the
   measured bars $b_m$ inside `sd_calibration`, rather than the single
   $m_{\mathrm{tol}}$ criterion.
5. **FDR theory for SD dummies.** The T-Rex guarantee is proven for
   Gaussian dummies; what the proof needs is an exchangeability/dominance
   property between dummies and nulls in the race. SD offers exact
   conditional inclusion probabilities ($\pi$, Section 4) — a route to a
   martingale/optional-stopping argument covering the adaptive pool
   (Section 3's proposition is the first step).
6. **Generation upgrades.** Floyd subset sampling: $O(k)$ per dummy, and
   distinct-pair sampling over the coded universe (no birthday duplicates,
   honest exhaustion at $\binom{n}{2}$; Geometric draw becomes
   hypergeometric).
7. **Family growth.** SD-OMP / SD-AFS: **done** (all six solvers +
   TRexSD `SDAlgo` dispatch, Sections 3, 6, 8; General + OMP is the new
   recommended default). Remaining: a logistic AFS analogue
   (`vd_afs_logistic` counterpart), integration with screening
   (Screen-TRex) for the very-large-$p$ regime, and GVS-style grouping.
   Open greedy-specific question: an AFS $\rho$ sweep (only
   $\rho \in \{0.3, 1\}$ is characterized; the $\rho \to 0$ limit should
   interpolate back to LARS behavior).

## 10. File map

| file | role |
| --- | --- |
| `tsolvers/sd_tsolvers/sd_tsolver_base.hpp` | shared state, dummy law, fair-race scale, sparse beta path |
| `tsolvers/sd_tsolvers/sd_tlars_solver.{hpp,cpp}` | general-$k$ SD-TLARS (index-set arithmetic, auto-$k$ ctor) |
| `tsolvers/sd_tsolvers/sd2_tlars_solver.{hpp,cpp}` | pair solver ($k{=}1$, dot-free; OnDemand / Geometric policies) |
| `tsolvers/sd_tsolvers/sd_tomp_solver.{hpp,cpp}` | general-$k$ SD-TOMP (greedy argmax + full OLS refit) |
| `tsolvers/sd_tsolvers/sd2_tomp_solver.{hpp,cpp}` | pair SD2-TOMP (dot-free, both policies) |
| `tsolvers/sd_tsolvers/sd_tafs_solver.{hpp,cpp}` | general-$k$ SD-TAFS (blend $\rho$, re-selection; $\rho{=}1$ ≡ OMP) |
| `tsolvers/sd_tsolvers/sd2_tafs_solver.{hpp,cpp}` | pair SD2-TAFS (dot-free, both policies) |
| `tsolvers/sd_tsolvers/sd_calibration.hpp` | pre-run auto-$k$ / auto-$L$ (bars, ceilings, $m_k(L)$) |
| `trex_selector_methods/trex_sd/trex_sd.{hpp,cpp}` | TRexSD selector ($K$ experiments, $\Phi/\Phi'$, FDP, $(T,v)$, L-loop; law × algo factory) |
| `demos/cpp/demo_sd_tlars.cpp` | contract equivalence (unit-L2 vs z-score) |
| `demos/cpp/demo_sd2_tlars.cpp` | arithmetic + policies + calibrated $p{=}10^6$ scenario |
| `demos/cpp/demo_sd_omp_afs.cpp` | greedy solvers: algorithm comparison, pair twins, auto-calibrated large $p$ |
| `demos/cpp/demo_trex_sd.cpp` | selector at the detection boundary + L-loop |
| `demos/cpp/demo_trex_sd_mc_variable_support.cpp` | **the** clean MC simulation (random support per trial, classic demo_trex_03 pattern, directly comparable with non-SD tables) |
| `demos/cpp/sim_trex_sd_mc.cpp` | legacy grid studies (default: LARS vs OMP vs AFS; named modes: signal / null / snr = law × algorithm cross product) |
