# Method card — LoRD3 (Local Regression Discontinuity Design Discovery)

**Source:** Herlands, McFowland III, Wilson & Neill, "Automated Local Regression
Discontinuity Design Discovery" (KDD 2018) + thesis ch. 5 (Herlands 2019 — the extended
version with the full LLR derivations and the effect-estimation section).
**Governing math:** [docs/math_audit_final.md](../math_audit_final.md) — the audit wins
every conflict with the printed papers and the released code. **Modules:** `natex.rdd.*`
(lord3, metrics), `natex.scan.*` (neighborhoods, geometry, statistics, coarse),
`natex.validate.*` (randomization, placebo, density, honest),
`natex.estimate.local2sls`. **Backtests:** `tests/backtests/` (test scores, academic
probation, ED visits; run of record in [docs/status/phase-2.md](../status/phase-2.md)).

## What it does

Given `(x, T, y)` with real-valued forcing columns `z ⊆ x`, LoRD3

1. **fits** one smooth global treatment model `T_i = f(x_i) + eps_i` (polynomial
   background; heteroskedastic Normal residual model for real-valued `T`, Bernoulli
   log-odds model for binary `T`) — reading only `(x, T)`, never `y`;
2. **scans** the k-nearest neighborhood (distances in standardized `z`) of every data
   point, bisecting each neighborhood with the k−1 hyperplanes through its center and
   scoring each bisection by the LLR of a two-group mean/odds-shift alternative
   against a single-shift null; discoveries are ranked by max LLR, and the
   responsible forcing columns are read off the normalized hyperplane normal (the
   forcing-variable influence statistic);
3. **validates**: fitted-null randomization calibration of the max LLR, local
   intercept-continuity placebo tests on the remaining covariates (Holm-corrected),
   and a density falsification test on the signed distance along the frozen
   discovered normal;
4. **estimates** the local effect at the discovered boundary with frozen
   side-indicator 2SLS (HC1 errors) plus a non-parametric Wald ratio, with
   first-stage diagnostics (jump, t, weak-instrument flag) always on.

## Corrected math (audit items in parentheses)

* **+1-rank Monte Carlo p-values, never "exact"** (item 1): the randomization test is
  a parametric bootstrap against the fitted null — each replica redraws `T*` from the
  fitted model and reruns the full scan. natex states this honestly and offers an
  honest discovery/estimation split (`natex.validate.honest`) as the primary
  post-selection guarantee.
* **Bernoulli nulls are drawn as Bernoulli(p̂) directly** (item 2): the released
  generator thresholds a Gaussian at a uniform draw, whose success probability is not
  p̂ (≈ .176 instead of .1 at p̂ = .1, σ̂ = .3).
* **Placebo tests use local intercept-continuity contrasts** (item 3): the printed
  side-mean contrasts are nonzero for any smooth covariate even under a valid design,
  so they mechanically reject; natex fits side-specific trends and tests intercept
  equality at the cutoff, Holm-corrected across covariates.
* **Frozen side-indicator 2SLS replaces the printed group instrument** (item 4): Eq
  5.14's `W = T − μ` strips the exogenous jump and keeps the endogenous residual —
  inconsistent under the paper's own DGP. First-stage relevance is never inferred
  from a high residual LLR (item 10): the first-stage jump, its t, and a
  weak-instrument flag accompany every estimate.
* **Density test on the frozen geometry only** (item 6): oblique projection can hide
  or manufacture density jumps, and search-selected geometry invalidates fixed-cutoff
  critical values; natex runs the test on the signed distance along the *frozen*
  discovered normal and documents it as a falsification test only.
* **Legacy outputs are not parity ground truth** (implementation landmines, items
  20–24): the released code indexes *reverse* neighbors when estimating per-point
  residual spread (item 20 — variable-size, possibly singleton sets that collapse to
  the floor), drops sharp pure-group splits (natex scores them with the boundary
  likelihood supremum, item 21), and applies an absolute 1e-6 floor to the residual
  spread (natex uses a data-scaled floor, item 24). Do not treat legacy scan output
  as ground truth in parity tests.
* **Printed-equation typos absorbed** (adjudicated typo list): the `γ_r x^R`
  polynomial exponent is `x^r` (the typo repeats in the KDD paper and both thesis
  chapters — confirmed by 576-dpi rendering); Eq 5.18's union is an intersection.

## Hyperparameters and defaults

| Parameter | Default | Notes |
|---|---|---|
| `k` | 50 | neighborhood size; the scan scores every point's kNN in standardized `z`. |
| `q` | 99 | fitted-null replicas; p = (1 + #{replica max LLR ≥ observed}) / (q + 1). |
| `degree` | 1 | polynomial order of the background model `f`. |
| `coarse` | off | two-stage coarse-to-fine scan for large n (subsample `n_coarse` = 2000). |

## Conventions and failure paths (documented, never silent)

1. **Hyperplane tie convention** (item 23): the neighborhood center lies on every
   candidate hyperplane; membership is explicit — signed distance ≥ 0 ⇒ group 1.
2. **Failure paths**: a neighborhood that cannot be scored is skipped and reported;
   missing statistics serialize to JSON null, never 0.0 (house rule).
3. **One rng**: a single `numpy.random.Generator` flows through every stochastic call
   (scan, null replicas, estimation), so runs replay bitwise given the seed.
