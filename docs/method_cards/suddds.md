# Method card — SuDDDS (Subset Discovery of Difference-in-Differences)

**Source:** thesis ch. 6 (never published outside the thesis; this is the first public
implementation). **Governing math:** [docs/math_audit_final.md](../math_audit_final.md) — the
audit wins every conflict with the printed chapter. **Modules:** `natex.did.*` (panel,
background, statistics, mdss, suddds, controls, effects, metrics), `natex.validate.panel`,
`natex.data.synthetic_did`. **Benchmark:** `tests/backtests/test_prop99.py` (run of record in
[docs/status/phase-3.md](../status/phase-3.md)).

## What it does

Given panel records `(x_i, t_i, theta_i, y_i)` with categorical covariates `x`, time `t`,
treatment `theta` and outcome `y`, SuDDDS

1. **discovers** a treated subset `s_tau` (a conjunction over dimensions of unions over
   values) and an intervention time `T0` where `theta` jumps — reading only `(x, t, theta)`,
   never `y`;
2. **validates** the discovery (fitted-null LLR calibration, composition/anticipation checks,
   per-dimension composition placebos);
3. **identifies a control** (standard DD, synthetic control, or GESS) and **estimates** the
   outcome effect `tau_hat` with a two-sided studentized placebo randomization test.

## Corrected math

### Window convention

All scan kernels restrict to `[T0 - W, T0)` (pre, `g0`) / `[T0, T0 + W)` (post, `g1`).
Records outside the window carry `c = b = 0`, so subset sums restrict automatically.

### Double-beta model (Eqs 6.6–6.12)

Residuals `r_i = theta_i - f(unit_i, t_i)` from the background fit; `H1` gives the window
sides separate means `beta_g0, beta_g1`. Sufficient statistics use **sigma-squared** weights —
`c_i = r_i / sigma_i^2`, `b_i = 1 / sigma_i^2`. The thesis page prints `sigma_i`; the printed
Eq 6.9 only follows from Eq 6.7 with the squared weights (audit §1, adjudicated). Eq 6.9:

```
LLR(s, g) = C1^2/(2 B1) + C2^2/(2 B2) − (C1 + C2)^2 / (2 (B1 + B2))
```

with `C1, B1` the post-side and `C2, B2` the pre-side sums over `s`. Eq 6.10's **prose** swaps
`q1/q2 ↔ beta_g0/beta_g1`; the equations are right (audit item 17) — in this implementation
`q1 = C1/B1` is the **post**-side mean. The `q1 − q2` priority suffers a Simpson's-paradox
problem; remedies are the greedy ordering and weighted convex combinations (`rho·q1 +
(1−rho)·(−q2)`, `n_rho` draws), both heuristic, **not exact** (audit item 16).

### Corrected single-Delta model (Eqs 6.13–6.16, audit item 15)

The printed statistic treats per-profile means `mu_i` as known and sums over all records. The
repair **profiles `mu_i` out under both hypotheses** and **restricts to the window**: with
side signs `d_j = +1` (post) / `−1` (pre), per profile `p` the precision-weighted correction
gives the corrected sufficient statistics

```
C-tilde_p = sum_j d_j c_j  −  delta-bar_p * sum_j d_j b_j,
B-tilde_p = sum_j b_j (d_j − delta-bar_p)^2,   delta-bar_p = (sum_j d_j b_j) / (sum_j b_j)
```

and the profile GLR `LLR = C^2 / (2 B)` with `C = sum_p C-tilde_p`, `B = sum_p B-tilde_p`,
MLE `Delta_hat = C / B`. The statistic is even in `Delta`, so **both signs of the shift are
scanned** (sign matters only in the priority ordering). This is the fast exact-priority path
(`method="single_delta"`); it requires the normal model.

### Bernoulli variant (audit item 19)

The model class must match `theta`'s type. For binary `theta` the window contrast is scored
with **exact Bernoulli log-odds offsets** on `eta_i = logit(p_hat_i)` (reusing the phase-2
masked-offset machinery), never a Normal approximation; `model="auto"` selects it iff `theta`
is binary in {0, 1}, and `model="normal"` stays forceable (the thesis-parity path — the thesis
ran Normal-residual models on Prop 99's binary treatment). Working residuals
`(theta − p_hat) / sqrt(p_hat (1 − p_hat))` are used **only to order MDSS priorities**; every
LLR evaluation is exact-Bernoulli. Null replicas draw `theta* ~ Bernoulli(p_hat)` directly
(audit item 2 — the released generator's clipped-Normal indicator has the wrong success
probability). A **Poisson variant for count treatments** (traffic-stop replication) is
documented future work.

## Algorithms 6–9 with the audit repairs

| Alg | Role | Repairs |
|-----|------|---------|
| 6 | Alternating `(s_tau, T0)` optimization over a `W` grid with restarts | **Item 11**: a single global incumbent `(s*, T0*, W*, llr*)` is kept across ALL windows and restarts; every converged local optimum is recorded, deduped on `(mask, t0)`, ranked by LLR — `discoveries[0]` is the global incumbent, never the last window's best. |
| 7 | Exhaustive `T0` over in-subset times | **Item 12**: a cutoff candidate qualifies only with `>= min_side` in-subset records on EACH side inside the window; no qualifying candidate → the caller keeps its incumbent (an empty side is never scored). |
| 8 | MDSS subset optimization per dimension | **Item 13**: the printed algorithm is deletion-only; repair = relax dimension `j`, slice over ALL its values, retain the incumbent explicitly. **Item 16**: when a dimension has `k_j <= exhaustive_max_values` values, all `2^k − 1` value subsets are enumerated — exact for the printed statistic; the `rho`-priority scan is a documented heuristic beyond that. |
| 9 | GESS control expansion | **Item 14**: line 6 is **argmin** over candidate control MSEs (the printed argmax is a typo); the incumbent MSE initializes at `+inf` via the empty control of `s_sup = s_tau`, so the first nonempty expansion always wins. Single-value and whole-dimension (`full_dimension=True`, Eq 6.21 intent) expansion modes. |

Background model (Eq 6.4 lineage): unit one-hot effects + polynomial in standardized time,
solved by pinv/lstsq — the printed global-intercept-plus-full-dummies design is rank-deficient
(audit typo list). Per-profile residual variances are shrunk toward the global variance with a
count-based weight and a **data-scaled** floor `1e-12 · s2_global` — no absolute variance
floor (audit item 24).

### Printed-equation typos absorbed (audit typo list)

* **Eq 6.18 / 6.20** — denominators must count the records **actually summed** (the printed
  `1/|D \ s_tau|` and `1/(T0 |·|)` assume a balanced panel); `T0` doubling as "number of pre
  periods" is replaced by explicit pre-record counts. A time with no control records gives a
  NaN counterfactual, never 0. (Also `s → s_tau` subscript.)
* **Eq 6.19** — synthetic-control weights live on **units**, not records.
* **Eq 6.21** — ill-typed as printed; implemented as the whole-dimension relaxation mode of
  GESS.
* **Eqs 6.24–6.25** (synthetic DGP) — dimensionally invalid as printed; the corrected DGP is
  in `natex.data.synthetic_did` (docstrings give the exact form).
* **Eq 6.27** — creates heteroskedasticity (time-scaled noise), not correlation; implemented
  as printed-intent with the docstring stating what it actually does.
* **Eq 6.28** — set notation corrected (`s_g = s_tau ∪ s_c`, `s_c ⊆ D \ s_tau`).
* **Thesis xref 6.26 → 5.22** — the thesis's chapter-5 results text ("the data
  generating process of y in Eq. (6.26)") cites this chapter's outcome DGP where
  Eq. (5.22) is meant; the KDD parallel passage correctly cites Eq. (22).

## Validation battery (audit items 1, 2, 5, 18)

* **`panel_randomization_test`** — a **parametric bootstrap against the fitted null, NOT an
  exact randomization test** (audit item 1): +1-rank p-values, every replica refits its own
  background and reruns the full scan. Null kinds: `ar1_unit` (unit random effects + per-unit
  stationary AR(1) — dependence-preserving, audit item 18), `iid` (documented
  dependence-breaking comparison), `bernoulli` (direct `Bernoulli(p_hat)`, audit item 2).
  **Unbalanced-panel limit:** the `ar1_unit` fit pools lag-1 products over consecutive
  time-sorted records per unit; with irregular gaps the single `phi` misstates dependence at
  long gaps. **Deterministic-treatment limit (Prop 99 finding):** when all but one unit have
  identically-zero residuals, the pooled AR(1) absorbs the treated unit's step as
  autocorrelated noise (`phi ≈ 0.94`) and hands that noise to every replica unit — the null
  max-LLR dwarfs the observed and p pins at 1.0. The audit-19-consistent calibration for a
  binary policy dummy is the Bernoulli scan + Bernoulli null (Prop 99: p = 0.01, Q = 99).
* **Composition + anticipation checks replace McCrary-in-time** (audit item 18): McCrary on
  calendar time is information-free when record times are design-determined. Composition:
  chi-square independence of unit/profile counts pre vs post inside the window. Anticipation:
  placebo jumps at `T0 − shift·step` restricted to `t < T0`, Holm across shifts.
* **`tau_randomization_test`** (audit item 5): two-sided **studentized** statistic
  `|tau_p / se_p|` (`se` = sd of per-post-period mean gaps / `sqrt(h)`, a documented simple
  choice), +1-rank p, placebo subsets **matched in shape** — same number of full covariate
  profiles, same `T0`, drawn from profiles with no `s_tau` records, `s_tau` records removed
  from the placebo panel, and (gess seeding) the **same constrained-dimension pattern** as the
  observed discovery. Exact zero movement (`tau = 0` and `se = 0`, e.g. composition shares of
  time-invariant covariates) maps to statistic 0.0 — provably no movement is the least extreme
  outcome, not a failure; any other undefined case stays NaN.
* **Per-dimension composition placebos**: for each dimension not defining `s_tau`, the modal
  one-hot share replaces the outcome, the tested dimension is removed from the profile
  definition (matched shapes again), Holm across dimensions.

### The precise conditional statement (replacing "independence of the two tests")

The thesis claims the LLR discovery test and the `tau_hat` placebo test are independent
because the scan never touches `y`. The correct statement: *the scan statistic is a function
of `(x, t, theta)` only, so **conditional on the discovered `(s_tau, T0)`** the `tau_hat` test
uses `y` information that selection never used; placebo profiles are treated as exchangeable
with `s_tau` under `H0` — an assumption, not a theorem.* No unconditional independence is
claimed anywhere in natex.

## Hyperparameters and defaults

The thesis reports **none** of these (spec §10 unreported-hyperparameter risk: number of
randomization draws, W grid, restart count, rho draw count are all absent from the chapter);
every default below is a natex decision, documented here and in docstrings.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `windows` (W grid) | `default_windows(t)`: `(span/8, span/4, span/2)` snapped up to multiples of the median time step, each `>= 2` steps | Prop 99 backtest pins `(5, 8, 10)` years (plan-fixed). |
| `restarts` | 8 | Restart 0 initializes `s = D` (thesis practice); later restarts draw per-dimension Bernoulli(1/2) value masks. |
| `bins` | 4 | Quantile bins per numeric covariate; 4 matches the thesis's Prop 99 setup. |
| `min_side` | 3 | Audit-12 minimum two-sided in-window support per cutoff candidate. |
| `n_rho` | 10 | WCC rho draws, U(0,1). |
| `exhaustive_max_values` | 12 | Per-dimension exact enumeration threshold (audit 16). |
| `degree` | 1 | Background polynomial order in standardized time. **For a binary one-shot policy dummy use 0**: theta's only time variation IS the candidate jump, so any time polynomial partially absorbs the signal and redistributes it as a spurious trend across every control unit (Prop 99: degree=1 manufactures a whole-panel `t0=1980` optimum, llr 8.53, that traps greedy/wcc; degree=0 recovers California exactly with all methods, llr 13.8). |
| `shrink` | count-based `10/(10+n_prof)` | Variance shrinkage toward global; data-scaled floor (audit 24). |
| `Q` (LLR calibration) | caller-set (99 in backtests) | +1-rank p; parametric bootstrap, never "exact". |
| `Q` (tau placebos) | `"auto"`: enumerate when the profile pool yields <= 200 placebos, else sample 199 | Enumerate mode on single-profile `s_tau` = Abadie's placebo-in-space. |

## Deviations (documented, never silent)

1. **Prop 99 covariate aggregation**: the thesis quantizes "the covariates" into 4 bins but
   subsets are covariate profiles across ALL time points; natex derives **state-level,
   time-invariant summaries** (means of lnincome/retprice/age15to24/beer over available years
   + lagged cigsale 1975/1980/1988) so profiles select whole state trajectories. The thesis
   does not state its aggregation. With bins=4 all 39 states have distinct profiles.
2. **Composition/anticipation replace McCrary-in-time** (audit 18, above).
3. **Synthetic control is outcome-only**: unit-level simplex weights fit on pre-period
   outcome trajectories (SLSQP, scale-normalized objective — the raw-scale SSE stalled SLSQP
   at its uniform start); Abadie et al.'s covariate V-weight nesting is not implemented. On
   Prop 99 the fit still concentrates on ADH's published donors (Utah .39, Montana .23,
   Nevada .21, Connecticut .11 vs ADH's Utah .33, Nevada .23, Montana .20, Colorado .16,
   Connecticut .07).
4. **Poisson count-treatment variant deferred** (audit 19 notes it; Bernoulli covers the
   binary case in scope).
5. **`ar1_unit` null limits**: unbalanced panels (pooled single-phi) and deterministic
   policy dummies (structural conservatism) — see validation battery above.
6. **Estimation uses the full pre/post split** at `T0` (`t < T0` / `t >= T0`), not the scan
   window — Eq 6.17/6.18 as printed. See the Prop 99 magnitude reconciliation below.
7. **Staggered adoption**: `DiDEstimatorBackend` protocol only (spec non-goal);
   Callaway–Sant'Anna is a future backend.

## Prop 99 backtest — investigated resolutions (2026-07-11)

Run of record: `tests/backtests/test_prop99.py` (windows (5, 8, 10), bins 4, restarts 8,
degree 0, seed 0; 39 states × 31 years).

* **Recovery**: greedy, wcc, single_delta (normal model, thesis parity) and the Bernoulli/wcc
  variant all return `(California, 1989)` as the top discovery with precision = recall = 1.0
  (LLR 13.82 normal / 13.86 Bernoulli). Thesis §6.4.3 parity: perfect rediscovery.
* **Effect magnitudes vs Table 6.1** (printed: DD −10.94, SC −8.96, GESS −6.67): natex gets
  **dd −27.35, synthetic −19.51, gess −26.65** (gess control = Montana, pre-MSE 18.4;
  synthetic pre-MSE 2.7 « dd's 51.2). Same sign, same ordering of pre-fit quality; magnitudes
  are ~2–2.5×. Investigated cause: the effect accumulates (~−5 packs in 1989 to ~−30 by
  2000) and natex averages the **full** post period 1989–2000 (Eq 6.18 as printed), while the
  thesis's numbers match a ~5-year effective post window (natex dd gap restricted to
  1989–1993: −18.8; symmetric-W=5 2×2 DD: −12.2 vs printed −10.94; synthetic 1989–1993:
  −12.3 vs printed −8.96). The thesis does not report its estimation window (also note its
  §6.4.3 text says "30 US states" where the ADH panel has 39). The synthetic −19.5 matches
  the canonical ADH average gap through 2000 (~−19).
* **Significance**: the corrected two-sided studentized placebo test (38 enumerated placebo
  states) gives **dd rank 13/39 (p = 0.33), synthetic rank 7/39 (p = 0.18), gess rank 9/39
  (p = 0.23)** — the thesis's "all significant at 5%" does **not** survive the audit-5
  statistic: p ≤ 0.05 requires California to be the single most extreme state, and genuinely
  non-parallel placebos (Missouri, West Virginia: flat level-shifted post gaps with tiny se)
  out-t California's trending gap. This is a finding about the estimators on this panel, not
  a scan failure: the **discovery** calibration is significant (Bernoulli scan vs
  Bernoulli(p_hat) nulls: p = 0.01 at Q = 99), while the normal/ar1_unit null is structurally
  conservative here (p = 1.0, pinned as a regression).
* **Validation battery**: composition chi-square p = 1 (balanced panel); anticipation
  placebo jumps at shifts 1–3 exactly 0 (constant pre-period residuals); per-dimension
  composition placebos Holm p = 1.0 on all four free dimensions (time-invariant covariates —
  the zero-movement studentization case).
* **y-blindness**: the scan with `outcome=None` equals the outcome-loaded scan bitwise
  (llr, t0, mask) across all discoveries.
