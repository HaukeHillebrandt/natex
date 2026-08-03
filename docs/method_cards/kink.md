# Method card — known-cutoff regression kink and difference-in-kinks designs

## What it does

`natex.kink` estimates a marginal causal response at a **known cutoff** from a change in
the slope of a policy schedule. It supports sharp and fuzzy regression kink designs (RKD)
and sharp and fuzzy difference-in-kinks designs (DiK). The DiK estimand follows Böckerman,
Jysmä, and Kanninen, *Difference-in-Kinks Design*, IZA DP 18313 (2025):
[paper](https://docs.iza.org/dp18313.pdf).

This is candidate evaluation, not unknown-kink discovery. The analyst must supply the
policy cutoff, a bandwidth, and either a known policy-slope contrast (sharp) or the observed
policy variable (fuzzy). Automatically searching cutoffs or reform dates would require a
search-calibrated selective-inference procedure beyond the paper.

## Estimands and orientation

Let `x = running - cutoff`. For variable `Z`, period `t`, and side `s`, let
`m[Z,t,s]` denote the one-sided derivative of `E[Z | x]` at zero. natex always defines a
kink as **right-minus-left**:

```text
kappa[Z,t] = m[Z,t,right] - m[Z,t,left]
```

The paper's `+`/`-` superscripts are inconsistent across sections, so the implementation
uses side names everywhere. Reversing both numerator and denominator would not change the
ratio, but it would reverse the reported reduced-form and first-stage signs.

| Design | natex estimand |
|---|---|
| Sharp RKD | `kappa[Y] / known_policy_kink` |
| Fuzzy RKD | `kappa[Y] / kappa[policy]` |
| Sharp DiK | `(kappa[Y,post] - kappa[Y,pre]) / known_policy_kink_change` |
| Fuzzy DiK | `(kappa[Y,post] - kappa[Y,pre]) / (kappa[policy,post] - kappa[policy,pre])` |
| Group DiK | as DiK with `group1`/`group0` (a binary group) in place of `post`/`pre` |

**Group DiK** (`group=` / `--group`) is the treated-vs-control variant: the same stacked
estimator contrasts the 1-coded group's kink against the 0-coded group's at one time,
with neutral cell labels `group0_left ... group1_right` and
`extras["dik_contrast"] = "group1_minus_group0"`. It is numerically identical to aliasing
the group indicator as the DiK time variable — the labels, recorded contrast, and caveat
text are what change: a group DiK is a **cross-group contrast**, so identification needs
parallel non-policy slope kinks *across groups* and a *group-stable* marginal response,
not a time-stable one.

The RKD ratio is the marginal average causal response at the cutoff. Under the fuzzy DiK
conditions below, the ratio is a kink-change-weighted marginal response: latent policy types
with larger individual kink changes receive more weight.

## Local-polynomial fit

Within `|x| <= bandwidth`, each side (and each pre/post side for DiK) gets its own
polynomial intercept and coefficients. The weighted objective is the paper's Equation 8:

```text
sum_i K(x_i / bandwidth) * (Z_i - polynomial_cell(x_i))^2
```

The same degree, bandwidth, kernel, donut, rows, and adjustment covariates are used for the
outcome and fuzzy first stage. Internal powers use `x / bandwidth` for conditioning; all
reported slopes are transformed back to the original running-variable units.

- Default degree: local linear (`degree=1`); higher degrees are explicit sensitivity choices.
- Default kernel: triangular; `uniform` and `epanechnikov` are also available.
- Optional `donut` excludes observations closest to the cutoff.
- Numeric covariates enter additively after scaling; constant covariates are dropped and
  counted. Side/period polynomial coefficients remain fully saturated.
- Optional per-observation weights (`weights=` / `--weights`) are multiplied into the
  kernel weights for every fit (outcome, first stage, combined influence). Use
  `1/sigma_i^2` when the outcome carries published standard errors or CIs
  (precision-weighted kink). Weights are fixed known constants: zero-weight rows are
  excluded and counted like zero kernel weights, non-finite weights drop the row, negative
  weights are rejected, and the HC1 `n − k` degrees-of-freedom correction keeps counting
  rows, not effective sample size — rescaling all weights by a constant changes nothing.
- Non-finite rows are dropped only when the requested fit uses that variable. Counts are
  returned in `n_used`, `n_by_cell`, and `extras`.

There is **no automatic DiK bandwidth selector** in the paper, so `bandwidth` is required.
Report estimates over a defensible bandwidth grid, donut sizes, and shifted placebo cutoffs.

## Inference and weak first stages

The paper gives point estimators but does not specify its covariance or weak-denominator
procedure. The following are natex choices:

- HC1 sandwich covariance by default.
- CR1 cluster-robust covariance when `clusters=` / `--cluster` is supplied, with
  `t(G-1)` critical values. Every local side/period cell must contain at least two clusters.
- HAC (Newey-West/Driscoll-Kraay) covariance when `hac_lags=` / `--hac-lags` is supplied,
  for serially correlated series where the running variable is time: scores are pooled by
  distinct running value (ascending), the meat adds Bartlett-weighted lag cross-products
  over that sequence (a lag is one step in the sorted distinct values, so gaps count as
  adjacent), and critical values are `t(n_time_points - 1)`. `hac_lags=0` reduces exactly
  to CR1 clustered by distinct running value. Mutually exclusive with `clusters=`.
- Every estimate reports `extras["residual_lag1_autocorr"]` (per-cell lag-1
  autocorrelation of per-distinct-running-value mean residuals) and, when it reaches 0.25
  without `hac_lags`, an `extras["autocorrelation_warning"]`.
- Joint outcome-policy sandwich covariance in the fuzzy delta-method standard error,
  evaluated through the combined outcome-minus-ratio-times-policy influence score to avoid
  numerical cancellation.
- First-stage slope contrast, standard error, Wald F, and `weak_first_stage` (`F < 10`, a
  heuristic rather than a design-specific critical value).
- A Fieller confidence set for fuzzy ratios. Its honest shape can be `interval`, `disjoint`,
  `unbounded`, or `empty`; it is never coerced to a finite interval.
- Zero or unidentified denominators produce `NaN`/JSON `null`, never a fabricated `0.0`.
- Cells without residual degrees of freedom fail explicitly instead of returning a zero
  standard error.

The ordinary Wald interval is conventional local-polynomial inference and may retain
**smoothing bias**. Robust bias correction is not implemented; polynomial and bandwidth
sensitivity is part of the required analysis.

**HAC is a mitigation, not a certificate.** On short, strongly persistent windows even a
correctly specified HAC covariance stays oversized, because the side-specific local fits
absorb the low-frequency noise before the residual scores ever see it. Null calibration on
AR(1) noise with `rho = 0.75` over 64 time points measured empirical size at nominal 0.05
of ~0.48 for HC1 and still ~0.34 for Newey-West across lags 4-20 (VAR(1) prewhitening:
~0.19). On such series the honest headline inference is the placebo-cutoff calibration
below (`placebo_calibrated_p`), with `hac_lags` as the better-behaved nominal covariance.

### SE convention

natex applies the HC1 degrees-of-freedom correction **jointly across all cells**: `n − k`
counts every observation and coefficient in the stacked side/period regression, not each
side separately. Cross-checks that fit each side on its own and apply per-side `n − k` will
report SEs roughly 5% larger at n ≈ 50 while matching the point estimate exactly. In the
METR time-horizon cross-validation (uniform kernel, bw = 720), natex and an independent
per-side OLS fit agreed on the kink to 10 decimals (both 0.0067291763) while the SEs were
0.0023236 (joint-cell dof) vs 0.0024334 (per-side dof). This is a convention difference,
not an error; the gap shrinks as cell sizes grow.

## Identification assumptions

### Sharp RKD

1. The non-policy outcome derivative has no kink at the known cutoff.
2. The marginal response to the policy is continuous at the cutoff.
3. The known policy schedule has a nonzero slope kink.

The first condition covers both direct running-variable effects and selection/composition.

### Sharp DiK

1. The post-minus-pre change in the non-policy slope kink is zero (parallel kink trends).
2. The marginal response is continuous across the cutoff and **time-stable**.
3. The known policy kink changes over time. Both policy slopes may change; DiK does not
   require one clean, fixed-slope side.

A stable nuisance kink or level jump is allowed and is the central advantage over a
cross-sectional RKD. Multiple pre-period estimates are a pseudo-test of parallel trends,
not proof of the assumption.

### Fuzzy designs

Fuzzy designs require the analogous conditions within latent policy-schedule types and a
same-sign/monotonicity restriction on individual kink changes. For fuzzy DiK, natex states
an additional **composition-stability** condition: the latent schedule-type distribution at
the cutoff must be stable across periods, or observations must be validly reweighted to a
common distribution.

This extra condition closes a gap in the paper's Proposition 2 proof. The setup permits a
period-specific latent distribution, but the proof takes one expectation of a within-type
post-minus-pre contrast. Interchanging derivatives and expectations separately by period
does not make those two measures equal. Without composition stability, the claimed
same-sign positive weights need not result. The software records this caveat but cannot test
the assumption.

### Time as the running variable

Calendar-time RKD — "did the trend bend at this dated event?" — is a legitimate use of the
estimator but **not** a manipulable-running-variable design: no unit sorts itself across a
date, so density/manipulation arguments give no protection. Identification reduces to a
single untestable assumption: *no co-located slope-changing event* — nothing else may bend
the outcome's expected slope at that exact date. Two practices are therefore mandatory,
not optional:

- **Always run the placebo-kink grid** (`placebo_kinks`), and turn it into the headline
  p with `placebo_calibrated_p(estimate, grid)` — the add-one share of placebo cutoffs at
  least as extreme as the candidate on the `|z|` (or `|estimate|`) scale. Aggregate series
  are serially correlated, HC1/CR1 p-values are oversized there (see the calibration
  numbers above), and the estimator warns via `extras["autocorrelation_warning"]`; pass
  the same `hac_lags` to the grid as to the headline so both use one covariance recipe.
  Read the grid as separating **bend existence** from **date attribution**. A significant
  kink at the true cutoff plus
  significant kinks at shifted cutoffs means the series bends over an era, not at the
  event. In the Epoch field pass, the METR time-horizon kink was positive in 8/8
  bandwidth-donut cells, yet pre-side placebos at −270/−180/−90 days also rejected
  (empirical size 3/6 at bw = 720): an era bend, honestly reported as "reasoning-era slope
  doubling", not "o1-preview caused it". GPQA-diamond, in contrast, passed with empirical
  size 0/7 — a bend that does localize to the date.
- **Run a sibling-series falsification**: an aggregate or related series where the same
  test demonstrably has power but reads null at the candidate date. The Epoch Capabilities
  Index played this role for the METR/GPQA claims: its placebo grid rejected at 4/7 shifted
  cutoffs (so the test has power in that data) while every specification at the o1 date was
  null (t between −0.60 and −1.54) — evidence that the per-benchmark bends are not a global
  measurement shift.

With a dummy unit denominator (`--policy-kink 1.0`), tau is a descriptive slope change in
outcome units per day, not a marginal causal response to a measured policy variable; label
it as such. A worked example of all of the above is the Epoch case study:
[docs/case_studies/epoch-kinks.md](../case_studies/epoch-kinks.md).

## API

```python
from natex import difference_in_kinks, regression_kink

rkd = regression_kink(
    y,
    running,
    policy_kink=-0.4,       # omit and pass treatment=policy for fuzzy RKD
    cutoff=0.0,
    bandwidth=1500.0,
)

dik = difference_in_kinks(
    y,
    running,
    post=time >= 2011,
    treatment=policy,      # or policy_kink_change=-1.0 for sharp DiK
    cutoff=0.0,
    bandwidth=1500.0,
    clusters=person_id,
)
print(dik.tau, dik.ci, dik.fieller_kind, dik.first_stage_F)
```

CLI equivalents write NaN-clean `out/kink.json`; an undefined core estimate is still
written for diagnosis and then returns a nonzero process status:

```bash
natex kink data.csv --design rkd --outcome y --running score \
  --policy-kink -0.4 --cutoff 0 --bandwidth 1500 --out out/

natex kink panel.csv --design dik --outcome y --running score \
  --treatment policy --time year --t0 2011 --bandwidth 1500 \
  --cluster person_id --out out/

natex kink groups.csv --design dik --outcome capex --running quarter \
  --group big4 --policy-kink-change 1.0 --bandwidth 8 --hac-lags 4 --out out/
```

## Diagnostics the estimator does and does not provide

Every successful result reports the cell-specific outcome slopes, fuzzy policy slopes,
pre/post kinks, effective cell counts, design rank, row loss, covariance choice, first-stage
strength, and ratio confidence sets. These are computation diagnostics, not assumption
certification.

The paper's validation battery (Figures A2-A4, Table A3) is callable from `natex.kink`:

```python
from natex.kink import (
    covariate_kinks,       # predetermined covariates as placebo outcomes (Fig. A4 B-D)
    density_kink_difference,  # binned pre/post density-difference kink test (Fig. A4 A)
    event_study_kinks,     # per-period kinks relative to a base period (Fig. A2 D)
    placebo_calibrated_p,  # add-one placebo-calibrated p for the headline contrast
    placebo_kinks,         # shifted placebo cutoffs with empirical size (Fig. A3 C)
    sensitivity_grid,      # bandwidth-by-donut re-estimation grid (Fig. A3 A-B)
)
```

All five reuse the estimator's right-minus-left reduced-form contrast and NaN-never-0.0
row handling, and all accept the same `hac_lags` passthrough as the estimator. The density test defaults to a degree-2 bin regression because the paper's
degree-13 specification over-rejects under the null in calibration; the Table A3 spec is
available via `degree=13`. These grids are falsification evidence — passing them does not
certify the identifying assumptions, and a joint pretrend test, CLI flags, and report/paper
bundle integration remain follow-ups.

Under the `plot` extra, `natex.report.figures.kink_fit_plot` renders the estimation-window
scatter, the two-sided kernel-weighted local-linear fits, the dashed pre-trend continuation
past the cutoff, and an optional tau/se annotation from a `KinkEstimate`.
