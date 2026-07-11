# Method card — DEE (Debiased Effect Estimation)

**Source:** Jakubowski, Chattopadhyay et al., "Exploiting Discovered Regression Discontinuities
to Debias Conditioned-on-observable Estimators" (JMLR 2023) and its released R/Python repo.
**Governing math:** [docs/math_audit_final.md](../math_audit_final.md) — the audit wins every
conflict with the printed paper. **Modules:** `natex.dee.*` (vknn, gp, noise, bma,
observational, debias; optional `forest`, `gp_torch`), `natex.data.synthetic_dee`.
**Benchmark:** `benchmarks/run_dee_sim.py` (scaled simulation-1 replication; run of record in
[docs/status/phase-4.md](../status/phase-4.md)); CI-small slice in
`tests/test_dee_benchmarks_small.py`.

## What it does

Given a dataset with covariates, a treatment and an outcome, plus a possibly-confounded
conditioned-on-observables CATE estimator, DEE

1. **discovers** local treatment-assignment discontinuities (the phase-1 LoRD3 scan — never
   reading `y`),
2. **repairs** the overlapping discoveries into disjoint quasi-experiments (Voronoi–KNN,
   Algorithm 1) and estimates a local effect `tau_hat_u` at each experiment (frozen-side 2SLS,
   HC1),
3. **debiases**: cross-fits the observational estimator, observes
   `bias_u = cate_obs(center_u) − tau_hat_u` at each experiment, fits a heteroskedastic-noise
   GP bias surface plus a direct CATE-extrapolation GP, and mixes the two models with
   stacked weights — returning `cate_debiased(x) = cate_raw(x) − bias_gp(x)`, the direct GP,
   and the mixture posterior everywhere.

## Pipeline

```
lord3_scan (phase 1; y-blind)                       Dataset (Z_std geometry everywhere)
      │  top-M′ candidates (fixed --m-prime, or select_m_prime
      │  from the phase-1 fitted-null Monte Carlo when --q-null > 0)
      ▼
voronoi_knn_repair ──► balance_filter (phase-1 placebo battery, Holm)      [y-blind]
      │  disjoint QuasiExperiments (k′-NN ball ∩ Voronoi cell)
      ▼
experiment_effects: frozen-side 2SLS, HC1, first-stage diagnostics ──► tau_hat_u, SE_u
      │
experiment_crossfit_cate (audit 9): leave-experiment-out obs-CATE at projected centers
      │  bias_u = cate_obs_u − tau_hat_u          (pinned sign convention)
      ▼
smooth_noise (stage 1): chi-square measurement model on SE² ──► noise variances
      │
HeteroskedasticGP × 2: bias surface (model A) + direct CATE surface (model B)
      │
model weights: buffered predictive stacking (default) | LOO | MLL softmax
      ▼
query grid: cate_raw − bias_gp = cate_debiased;  direct GP;  audit-8 mixture posterior
```

One caller-supplied `numpy.random.Generator` drives every stochastic step (cross-fit folds,
estimator seeds, GP restarts, stacking folds); identical seed ⇒ identical output. Failures
propagate NaN, never 0.0.

## Corrected Algorithm 1 (Voronoi–KNN repair)

Forward-stepwise over LLR-ranked candidates: tentatively add a candidate, reassign every
accepted center's members to its k′-NN ball ∩ Voronoi cell (over accepted centers only,
distances in Z_std), accept iff **every** accepted center keeps ≥ `t_side` members on both
sides of its frozen hyperplane. Two real code fixes vs the paper's repo (audit §3; repo
risks 1–3):

- **the first candidate is thresholded like every other** — the repo installed it unchecked;
- **all inputs are explicit** (`dataset, discoveries, m_prime, k_prime, t_side`) — the repo's
  `get_voroni_knn_...` ignored its `train_data` argument and read a global `df`.

Also: the projected centroid is the **direct orthogonal projection**
`centroid − ((centroid − Z_std[center])·n̂)n̂` — deterministic, replacing the repo's unseeded
random Gram–Schmidt basis (the rotation was mathematically basis-independent but consumed
unseeded RNG). Ties: signed distance ≥ 0 ⇒ group 1 (audit 23, same as the scan); Voronoi
ownership ties go to the earlier-accepted center. The audit's "recompute index sets after
recentering" suggestion was **rejected** — the projection leaves the separating hyperplane
unchanged, so index sets are invariant. Index sets are computed **once** and passed by object
to every later stage (repo risk 4: the repo's step 4 re-ran VKNN and silently truncated on row
mismatch).

## Pinned sign convention (Codex #30)

The paper's appendices drift on the bias sign. natex pins, with a regression test
(`test_sign_convention_regression`):

```
bias_u          = cate_obs(center_u) − tau_hat_u
cate_debiased(x) = cate_raw(x) − bias_gp_mean(x)
```

A +3 confound therefore yields bias observations near +3 and a debiased surface near the true
CATE; a flipped convention fails loudly.

## Audit corrections implemented

### Item 7 — Theorem 1 / Lemmas 1–2 (identification-error bound)

The printed Lemma 1 misses cross terms and Lemma 2's step `Z₁ ≤ E[q]` is false. The repaired
route (sphere geometry) gives `E[q²] ≤ 2ρ²`, so the implication carries a **√2 factor** and
holds only along **fixed-shape scaling** of the experiment. natex therefore **never computes
the paper's printed bound**: `experiment_radius` reports each experiment's Z_std radius ρ as a
diagnostic (`DEEResult.diagnostics["radii"]`), and any bound a user forms from it must carry
the √2 factor and the fixed-shape caveat.

### Item 8 — mixture posterior

The rural-roads code drew an independent model indicator per prediction point, destroying
between-model covariance (aggregate CIs far too narrow). natex (`dee/bma.py`):

- analytic mixture covariance `w Σ_a + (1−w) Σ_b + w(1−w)(μ_a−μ_b)(μ_a−μ_b)ᵀ`;
- sampling draws **one Bernoulli(w) model label per posterior draw**, then the whole draw from
  that component.

Model A's posterior is the bias posterior shifted by `cate_raw` — the observational
estimator's own uncertainty is not modeled (paper behavior, documented).

### Item 9 — same-data correlation and estimated noise

Fitting the observational estimator on the rows that produced the local-IV effects correlates
the two stages and contaminates the bias observations. Repairs:

- **experiment-level cross-fitting** (`experiment_crossfit_cate`): experiments are rng-assigned
  to folds; each experiment's centroid is predicted by a model whose training rows exclude
  every fold-mate experiment's members — in particular its own (poisoned-member leak test).
- **SE² are estimated, never known**: the 2SLS SEs are HC1 (the repo used classical iid vcov —
  repo risk 7) and enter the GPs only through the stage-1 **chi-square measurement model**
  (`smooth_noise`: `log SE²` debiased by `ψ(df/2) − log(df/2)` with known variance `ψ₁(df/2)`,
  smoothed by the exact GP, back-transformed with a data-scaled floor). Audit §3 adopted this
  hierarchical form — a Normal likelihood plus a measurement model on SE², not both a t
  likelihood and a latent variance.

### Item 10 — first-stage relevance after repair

Every experiment carries `first_stage_t` / `weak_instrument` diagnostics. Weak-instrument
experiments are **kept, not dropped**: their large SE² automatically downweights them in the
heteroskedastic GP.

## Model weighting: buffered stacking replaces softmax-PLP

The paper combines models by softmax of posterior-log-prob scores, including an **unseeded
1-MC "random dist" buffered-LOOCV strategy** (repo risk 8: `np.random.choice` with no seed —
the published 1-MC numbers are not reproducible). natex:

- **default = buffered predictive stacking** (audit §3 adopted): both GPs' hyperparameters are
  refit within spatially buffered folds (train centers within `buffer` of any held-out center
  are excluded; buffer defaults to the median NN distance), and the mixture weight maximizes
  the summed held-out log mixture density on a deterministic 101-point grid over [0, 1] — a
  proper 1-D optimization, not softmax-PLP;
- MLL and closed-form-LOO softmax weights are kept as comparison strategies
  (`weighting="mll" | "loo"`);
- the 1-MC strategy is **not reimplemented** (superseded; a seeded version would still be a
  1-sample estimate of the stacking objective);
- a NaN stacking weight (fewer than 3 usable centers, or no jointly-scored fold) falls back to
  `loo_weights`, with the stacking detail preserved under `detail["stacking_fallback"]`.

## Conventions

- **Features = Z_std** everywhere: the scan, the VKNN geometry, the GP inputs, and the
  observational estimator (causal-forest features included) all live in the dataset's
  standardized-covariate space; query points arrive in raw forcing units and are standardized
  once via `Dataset.standardize` (bitwise consistent with `Z_std`).
- Aggregation across experiments assumes **disjointness by construction** (VKNN). The audit's
  alternative — influence-function-based aggregation that tolerates overlap — was considered
  and **deferred**: disjointness is the simple default and the repair already enforces it.
- Discovery (scan + VKNN + balance filter) never reads `y`; a permuted outcome leaves the
  repaired experiments bitwise identical (`test_discovery_stages_are_y_blind`).
- Degenerate pipelines (< 3 usable experiments) return NaN GP-derived outputs with
  `diagnostics["reason"]` — never 0.0.

## Observational estimator layer

`ObservationalEstimator` is a 2-method protocol (`fit(X, T, y)`, `predict_cate(Xq)`):

| Estimator | Deps | Treatments | Notes |
|-----------|------|-----------|-------|
| `TLearner` (default) | core (sklearn) | binary only | two GradientBoostingRegressors, CATE = μ̂₁ − μ̂₀; estimates the conditional contrast |
| `CausalForestEstimator` (`natex.dee.forest`) | `natex[ml]` (econml) | binary + continuous | `CausalForestDML`; serial by default for bitwise determinism (`n_jobs=-1` opt-in); `n_estimators` must be divisible by econml's `subforest_size` (4) |

**Estimand caveat** (documented deviation, calibrated in `tests/test_dee_optional.py`): the
DML forest estimates an orthogonalized residual-on-residual functional, not the conditional
contrast. On the synthetic constant-bias scenario its debiased error plateaus ≈ 1.25 (vs the
T-learner's 0.49) because the DML functional varies with the region propensities and a smooth
bias GP cannot fully absorb it; the optional-extra test gates on ≥ 40% error reduction rather
than the T-learner's absolute band.

## GP backends

The default GP is the exact numpy/scipy `HeteroskedasticGP` (`dee/gp.py`) — DEE surfaces live
on tens of centers, so O(|U|³) is trivial (spec §10 small-N defaults). `natex[gp]` adds
`TorchHeteroskedasticGP` (`dee/gp_torch.py`) with the identical
`fit/posterior/log_marginal_likelihood` surface for scale: FixedNoiseGaussianLikelihood + RBF,
fit via **`fit_gpytorch_mll`** — the maintained API, not the removed `fit_gpytorch_model` that
broke the paper's repo on modern stacks (repo risk 9). `fit_gpytorch_mll` is a **botorch** API,
so the `gp` extra includes botorch. Posterior parity with the numpy backend at fixed shared
hyperparameters is analytic (tested at atol 1e-3; observed ~1e-8, float64).

## Deviations from the paper / repo (investigated, not silently dropped)

1. **Scale**: the simulation-1 replication runs at n = 4000, k = 100, M′ = 40, k′ = 400,
   t_side = 25, 20 seeds (the paper: n = 20,000, k = 200 NN, 200 replications) — CI-viable
   while reproducing the qualitative claim; the harness exposes every knob.
2. **Exact conditional-bias DGP**: the paper's complier-shift calibration is repo code, not
   printed math; `make_dee_synthetic` instead constructs the confound `c` so that the
   conditional-on-observables contrast is exactly `tau(X) + beta(X)` (closed form from the
   complier-type probabilities), making the sampled bias surface the observational bias by
   construction. GP surfaces are one exact anchor-lattice draw kriged to the data points
   (documented in the module).
3. **type_probs default (0.1, 0.4, 0.4, 0.1)** in the benchmark harness: the paper-uniform
   (0.25,)⁴ leaves the local 2SLS too weak at the scaled-down n (calibration table in
   `tests/test_dee_debias.py`); the knob accepts the uniform mix.
4. **No Cattaneo/Kallus R benchmarks**: the paper's comparison arms need rdrobust/rdmulti/grf —
   treated as paper artifacts, not natex features.
5. **Rural-roads application deferred**: no local data (PMGSY microdata is access-controlled);
   candidate future backtest per the phase-3 follow-up style.
6. **M vs M′ notation**: the paper uses both for the candidate count; natex uses `m_prime`
   everywhere, selected either fixed or by `select_m_prime` (count of per-center max-LLRs
   strictly above the fitted-null 95% quantile — the paper's rural-roads rule).
7. **Weighted-BLOOCV mixture weights**: the paper's weighted-BLOOCV scoring drops the mixture
   weights from the predictive density; natex's stacking objective scores the full two-model
   mixture density directly.
