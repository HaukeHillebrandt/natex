# Method card — IV/SC discovery (instrument search + synthetic-control donor selection)

**Sources:** Belloni, Chen, Chernozhukov & Hansen (2012, Econometrica) for Lasso instrument
selection ("BCCH"); Abadie, Diamond & Hainmueller (2010, JASA) for synthetic-control donor
selection and in-space placebo inference ("ADH"). The Chen et al. (2026) Springer-roadmap
chapter that motivates this phase is **conceptual only** (its own notes warn it "cannot
adjudicate implementation details") — see [Out of scope](#out-of-scope-springer-roadmap-and-deviations).
**Governing math:** [docs/math_audit_final.md](../math_audit_final.md) — the audit wins every
conflict. **Modules:** `natex.iv.*` (search, pipeline, donors), `natex.estimate.iv2sls`,
`natex.estimate.simplex` (shared with `did/controls`), `natex.data.synthetic_iv`,
`natex.data.synthetic_sc`. **Benchmark:** `benchmarks/run_iv_selection.py` (CI-small slices in
`tests/test_iv_benchmarks_small.py`). **Backtest:** `tests/backtests/test_prop99_donors.py`
(run of record in [docs/status/phase-5.md](../status/phase-5.md)).

## What it does

1. **Instrument search** (`iv.search.select_instruments`): Belloni-style iterated plug-in
   Lasso first-stage selection over a candidate instrument pool, post-Lasso OLS first stage,
   HC1 first-stage F / partial R² / `weak` flag. Selection reads only `(T, pool, controls)` —
   the signature cannot even receive an outcome (the phase's discovery-honesty analog).
2. **General k-instrument 2SLS** (`estimate.iv2sls.iv_2sls`): HC1-sandwich 2SLS, Hansen J
   overidentification diagnostic, and closed-form Anderson–Rubin/Fieller confidence sets —
   also wired additively into `estimate.local2sls.local_2sls` (`ar_ci`/`ar_kind` on
   `EffectEstimate`) at discovered RD cutoffs.
3. **Honest pipeline** (`iv.pipeline.discover_instruments`): select on a discovery half,
   estimate + J + AR on the disjoint estimation half.
4. **SC donor selection** (`iv.donors`): unit×time outcome matrix, per-donor pre-trend
   scores, top-k pool, simplex weights, counterfactual + post-period ATT, and ADH in-space
   RMSPE-ratio placebo inference with +1-rank p-values.

CLI: `natex instruments` (1 + 3) and `natex donors` (4).

## Instrument selection — pinned BCCH plug-in formulas

With `d` = treatment and the pool `Z` both residualized on `[1, controls]` by OLS
(Frisch–Waugh), the weighted Lasso solves

```
min_pi (1/n) ||d − Z pi||²  +  (lam/n) Σ_j psi_j |pi_j|
```

* **Penalty (pinned):** `lam = 2 c √n Φ⁻¹(1 − gamma/(2p))` with `c = 1.1` and
  `gamma = 0.1 / log(max(n, p))` (documented natex choice of BCCH's admissible range;
  `p` is the pool size *after* the zero-variance drop below, and `n` the finite-row count).
* **Loadings (pinned):** heteroskedasticity-robust `psi_j = sqrt((1/n) Σ_i z_ij² eps_i²)`.
* **Iteration (pinned):** `eps⁰` = residual of `d` on the 5 pool columns most correlated
  with it (BCCH-style init; plain partialled `d` when p < 5); after each Lasso fit `eps` is
  refreshed from the **post-Lasso** residual on the current support; stop when the support
  repeats or after `max_iter = 15`. If a perfect post-Lasso fit drives any loading to 0 the
  last valid fit is kept (`extras["loading_degenerate"]`).
* **Exact sklearn mapping (pinned):** with `u_j = z_j / psi_j` and `b_j = psi_j pi_j` the
  objective equals `2 [(1/2n)||d − U b||² + (lam/2n)||b||₁]`, i.e.
  `sklearn.linear_model.Lasso(alpha = lam/(2n), fit_intercept=False)` on the partialled
  (hence centered) data; recover `pi_j = b_j / psi_j`. Verified against an analytic
  soft-threshold pin in `tests/test_iv_search.py`.
* **Zero-variance columns** (including columns exactly collinear with the controls) get
  `psi = ∞` semantics: structural up-front exclusion with a diagnostic, never a
  divide-by-zero. The threshold is **scale-relative** — `col_ss ≤ raw_ss (n·eps_machine)²`
  (audit item 24: no absolute floors; partialling a constant leaves an ~1e-32 relative
  remnant).
* **`lam="cv"` caveat (documented):** `LassoCV` (rng-seeded 5-fold shuffling) chooses only
  the lambda — the iterated-loading fit is then identical to an explicit float — but a CV
  lambda **voids the plug-in sparsity guarantee**; the plug-in is the default. "plugin" and
  float lambdas are RNG-free and bitwise deterministic.

**Post-Lasso rationale.** The Lasso point estimate is shrinkage-biased; BCCH's IV estimator
refits **OLS on the selected support** (post-Lasso) and uses those instruments in 2SLS. natex
reports both (`pi_lasso`, `pi_post`) and computes all first-stage diagnostics from the
post-Lasso selected block. An **empty selection is reported honestly** — `selected = []`,
`first_stage_F = NaN`, `partial_r2 = NaN`, `weak = True` — never a fabricated 0.0. On the
BCCH DGP at μ² = 8 the plug-in penalty *refuses* selection in 17/20 seeds: refusal is the
designed behavior in a weak first stage, not a failure.

## First-stage strength — the F < 10 heuristic (audit item 10)

Selection (or a significant scan LLR) never implies relevance: the HC1 Wald F of the selected
block in `T ~ [1, controls, instruments]` (χ²/k form) and the instruments' partial R² after
controls are **always computed**, and `weak = F < 10 or F = NaN`. The 10 is documented as the
**Staiger–Stock rule-of-thumb convention, not a Stock–Yogo critical value** (those are
homoskedastic, k-dependent, and calibrated to 2SLS bias/size targets) and not the
Montiel Olea–Pflueger effective F (out of scope, below). The honest pipeline re-checks F on
the estimation half — discovery-half selection never implies estimation-half relevance. Weak
does not gate estimation; it gates *trust*, and the AR set is the weak-robust answer.

## 2SLS (audit item 4)

2SLS is the **only** estimator family — the papers' printed group-instrument form
(Eq 5.14, `W = T − μ`) is inconsistent and is never implemented. `iv_2sls` is HC1-sandwich
2SLS with intercept + controls in both stages, `n/(n − p)` small-sample scaling, rank checks
(an unidentified projected design returns NaN, flagged `rank_deficient`), and row-wise
finite filtering counted in `extras["n_dropped"]`. Every degenerate path returns **NaN,
never 0.0**.

## Hansen J — scope statement (audit item 3 lineage)

**Exclusion is untestable.** The Hansen J statistic (df = k − 1 with one endogenous
regressor) tests **only the overidentifying restrictions, given at least one valid
instrument**: it can detect mutually inconsistent instruments, and it can never certify
exclusion itself — a full set of instruments sharing the same violation passes J. natex never
claims more. When the model is just-identified (k = 1), `j_stat`/`j_p` are `None` — never a
fabricated 1.0 — and `j_df = 0`.

## Anderson–Rubin / Fieller confidence sets (audit §3 adopted)

With intercept + controls partialled out of `(y, T, Z)` and `P` the projection onto the
partialled instruments, the level-α set is

```
{tau : r' A r ≤ 0},   r = y~ − tau T~,   A = P − c_k (I − P),
c_k = k F_crit(k, n − q − k; 1 − alpha) / (n − q − k)
```

a **quadratic in tau** `a·tau² + b·tau + c ≤ 0` with `a = T~'A T~`, `b = −2 y~'A T~`,
`c = y~'A y~`, each moment expanded as `(1 + c_k) m'Pm − c_k m'm` (closed form, no grid).
`k = 1` is exactly the Fieller construction.

**The four set kinds, reported as found — never coerced to a finite interval:**

| kind | quadratic geometry | meaning |
|------|--------------------|---------|
| `interval` | a > 0, disc > 0 | bounded CI `[r1, r2]` |
| `disjoint` | a < 0, disc > 0 | two rays `(−∞, r1] ∪ [r2, ∞)` — weak identification |
| `unbounded` | a < 0, disc ≤ 0 (or degenerate a ≈ 0) | whole line / half-line — no identification at level α |
| `empty` | a > 0, disc ≤ 0 | model rejected at every tau; reachable only when k ≥ 2 |

**Boundedness ⇔ first-stage-F identity:** `a > 0` iff the **homoskedastic** first-stage F,
`(T~'P T~ / k) / ((T~'T~ − T~'P T~) / (n − q − k))`, exceeds `F_crit` — the AR set is bounded
exactly when the classical first-stage F clears the AR critical value (regression-tested in
`tests/test_ar_ci.py::test_boundedness_iff_first_stage_f_exceeds_crit` across a strength
sweep straddling the threshold). Note the identity is in the *homoskedastic* F; the reported
diagnostic F is HC1, so the two can disagree near the boundary.

**Why it exists (the honesty gap):** the Wald CI `tau ± z·se` is finite even when the first
stage carries no information — it never admits weakness. Calibration (n = 250, μ² ≈ 4,
endogeneity 0.95, 200 reps; `tests/test_ar_ci.py`): Wald covers 170–185/200 vs AR 187–195/200
(nominal 190/200); Wald undercoverage needs *high* endogeneity — at 0.6 it covers 189/200. AR
set kinds at μ² ≈ 4 split ≈ 50% interval / 30% disjoint / 20% unbounded (implementation-time
calibration); at zero first-stage strength every calibrated seed lands unbounded/disjoint.
Full-sample AR is bounded whenever the homoskedastic F > F_crit ≈ 2 (n = 500, k ≥ 5), so on
the benchmark DGP the AR-vs-Wald gap shows on the **honest estimation half** (F halves there):
at explicit lam = 60 on μ² = 8, `honest_ar_kind` is unbounded/disjoint in 17/20 seeds while
every Wald CI (full and half) stays finite. `iv_2sls` reports `ar_ci` (interval kind) and
`ar_kind` always; `local_2sls` carries the same fields at discovered RD cutoffs.

## Honest discovery/estimation split (audit item 1 lineage)

`discover_instruments` splits rows with `validate.honest.honest_split`:
`select_instruments` sees **only** the discovery half; `iv_2sls` (2SLS + J + AR on the
selected columns) sees **only** the estimation half. Because the halves are disjoint, the
selection event is a function of the discovery half alone and is independent of the
estimation noise — the estimation-half J/AR/Wald p-values retain their nominal distributions
with **no post-selection correction**. `honest=False` selects and estimates on the full
sample and sets an explicit caveat string in `extras` (and the CLI payload): the same noise
that picked the instruments enters the p-values, which are optimistic to an uncontrolled
degree. `outcome=None` runs selection only — discovery never reads the outcome.

## SC donor selection protocol (ADH 2010)

On a unit×time outcome matrix (`unit_time_matrix`: duplicate cells mean-aggregate, empty
cells are NaN — never 0):

1. **Balanced-donor rule (phase-3 rule):** candidates are units with finite outcomes at
   **every** pre-`t0` time where the treated unit is observed; dropped candidates are counted
   in `extras["n_dropped_incomplete"]`.
2. **Pre-only scoring:** every complete candidate is scored against the treated
   pre-trajectory over the common pre times — RMSE (default) or Pearson correlation
   (NaN when < 3 common points, never fabricated). Scoring, ranking, and weight fitting read
   **only pre-`t0` columns**; the post period is the estimation target. This inherent SC use
   of pre-period outcomes is a documented method property, enforced by mutation tests
   (multiplying every post outcome by 10 leaves scores/donors/weights bitwise unchanged).
3. **Simplex fit:** the top `n_donors` (default: all complete candidates) get weights from
   `estimate.simplex.fit_simplex_weights` — the deterministic SLSQP fitter extracted verbatim
   from `did/controls.py`, keeping the phase-3 **scale-normalized SSE** objective (audit
   item 24: SLSQP's absolute accuracy threshold stalls on raw-scale outcomes) and the
   `MISSING_W_TOL = 0.1` counterfactual convention (missing-donor weight mass ≤ 0.1 →
   renormalize; beyond → NaN, never a silently zeroed time).
4. **Effect:** counterfactual over all times, gap `y_treated − y0_hat`, pre/post RMSPE and
   post-period mean ATT. Failure paths: no complete candidate / treated unobserved pre-`t0`
   → `att_post = NaN`, `pre_rmspe = +inf` (never 0), reason in `extras["failure"]`; no
   defined post period → NaN ATT, `post_rmspe = +inf`.
5. **In-space placebo (`sc_placebo_test`):** every complete donor candidate is refit as
   pseudo-treated under the **identical selection rule** (same `n_donors`, same scoring —
   audit item 5 matched shapes), with the treated unit's row deleted from every placebo panel
   so its real post-period effect never contaminates a placebo counterfactual. The statistic
   is the post/pre RMSPE **ratio** — sign-agnostic, hence two-sided by construction — and

   ```
   p = (1 + #{placebo ratio ≥ treated ratio}) / (n_used + 1)
   ```

   the +1-rank Monte Carlo form (audit item 1 lineage). Undefined ratios are **skipped,
   never a fake 0.0**: a failed fit is NaN; a perfect pre fit (`pre_rmspe == 0`, post > 0)
   is +inf; `post/inf` would fabricate 0.0 so failed fits never enter. `exclude_poor_fit = m`
   drops placebos with pre-RMSPE > m× the treated unit's (ADH's poor-pre-fit exclusion;
   default `None` — on small panels extreme-but-honest donors trip it too, so exclusion is
   opt-in and every dropped unit is listed in `extras["poor_fit_units"]`). Fewer than 5
   usable placebos — or an undefined treated ratio — gives `p = NaN`, never a fake 1.0.
   The whole donor path is deterministic: no rng anywhere.

## Hyperparameters and defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `c` (plug-in) | 1.1 | BCCH's slack constant. |
| `gamma` (plug-in) | `0.1 / log(max(n, p))` | Documented choice; `p` after the zero-variance drop. |
| `max_iter` (loadings) | 15 | Stops earlier on a stable support. |
| `lam` | `"plugin"` | `"cv"` (rng-seeded; voids the sparsity guarantee) or explicit float. |
| weak-F threshold | 10.0 | Heuristic convention, not Stock–Yogo (see above). |
| `alpha` (2SLS/AR) | 0.05 | One α for Wald CI, J, and the AR set. |
| `frac_discovery` | 0.5 | Honest split fraction. |
| `n_donors` | `None` (all complete) | Top-k by pre-trend score otherwise. |
| `scoring` | `"rmse"` | `"corr"` available; NaN keys rank last. |
| `exclude_poor_fit` | `None` | Opt-in ADH poor-pre-fit exclusion multiplier. |
| min usable placebos | 5 | Below it `p = NaN` (phase-3 policy). |
| `MISSING_W_TOL` | 0.1 | Counterfactual missing-mass renormalization tolerance. |

## Out of scope (Springer roadmap) and deviations

Documented, never silently dropped:

1. **Deep IV / GAN (DeepMatch-style) / text-based instrument matching** — the Springer
   chapter's forward-looking material: heavy optional deps (torch), no benchmark row, and the
   chapter is conceptual only. Candidates for a later phase behind optional extras.
2. **Sup-score weak-identification-robust *selection*** (BCCH §4) not implemented; the
   weak-IV-robust piece natex ships is AR/Fieller **inference** on the estimation half, plus
   the plug-in penalty's honest refusal to select at weak concentration.
3. **Montiel Olea–Pflueger effective F** not implemented — the HC1 Wald F with the F < 10
   convention is documented as a heuristic instead.
4. **No covariate V-weights in SC** (same deviation as phase 3): outcome-only pre fit. On
   Prop 99 the fit still concentrates on ADH's published donors (below), but the placebo
   ranking differs from ADH's figure (below).
5. **Matrix-completion / elastic-net SC variants and staggered adoption** — out of scope
   (spec non-goals this phase).
6. **`lam="cv"`** is provided but caveated (voids the plug-in sparsity guarantee).

## Prop 99 backtest — reconciliation notes (2026-07-11)

Run of record: `tests/backtests/test_prop99_donors.py` (39 states × 31 years, treated =
California, t0 = 1989, scoring = "rmse", deterministic).

* **Pre-trend ranks vs the simplex fit:** the top-8 pre-RMSE ranks are Montana(1), Idaho(2),
  West Virginia(3), Iowa(4), Colorado(5), Nebraska(6), Connecticut(7), Wisconsin(8) — exactly
  3 of ADH's five {Colorado, Connecticut, Montana, Nevada, Utah}. Nevada (~190 packs) and
  Utah (~65) match California's *shape* but not its *level*, so raw pre-RMSE screens them
  out; the **simplex fit, which can mix levels, recovers them**: full-pool nonzero weights
  Utah .394, Montana .232, Nevada .205, Connecticut .109, New Hampshire .045, Colorado .015 —
  summed weight on the ADH five = 0.955, and the same four heavyweights as the phase-3 SuDDDS
  synthetic control. Screening and weighting answer different questions; use `n_donors=None`
  when the pool is small enough to fit whole.
* **Effect:** `att_post = −19.51` (full pool; −22.65 at n_donors = 8), matching the canonical
  ADH average gap through 2000 (~−19) and the phase-3 finding. The thesis Table 6.1 value
  −8.96 reflects a shorter effective post window — the same reconciliation as
  [the SuDDDS card](suddds.md).
* **Placebo:** all 38 placebos usable, 0 skipped; California's RMSPE ratio ranks 3/39 in both
  variants → p = 3/39 ≈ 0.077, vs ADH's 1/39. The gap is our protocol's `exclude_poor_fit =
  None` plus the outcome-only fit: Missouri (ratio 23.9) and Virginia (19.8) out-rank CA
  here, while ADH's figure discards poor-pre-fit placebos before ranking. Pinned as a
  regression, not "fixed" — the exclusion stays opt-in.
* **Egger stretch finding (documented, xfail-pinned):** on the Bavarian council-size data,
  in-sample `rcsize` is *exactly* the statutory step function of population (zero residual),
  so the plug-in loadings `psi_j = sqrt(mean(z_j² eps²))` collapse toward 0 on iteration, the
  penalty vanishes, and the final Lasso support balloons to all 10 dummies — the strict
  "decoy-free support" claim fails in this zero-noise degenerate first stage. What holds (and
  is pinned as a passing regression): plug-in coefficients put ~4 orders of magnitude more
  mass on statutory dummies than decoys, and post-Lasso OLS recovers the statutory jumps
  (4, 2, 2, 4, 4) with decoy coefficients ≈ 0. Lesson: the plug-in penalty assumes a noisy
  first stage; deterministic assignment rules are LoRD3/RDD territory, not BCCH territory.
