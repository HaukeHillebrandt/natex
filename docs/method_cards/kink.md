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
```

## Diagnostics the estimator does and does not provide

Every successful result reports the cell-specific outcome slopes, fuzzy policy slopes,
pre/post kinks, effective cell counts, design rank, row loss, covariance choice, first-stage
strength, and ratio confidence sets. These are computation diagnostics, not assumption
certification.

Before a causal claim, also inspect binned plots, multiple bandwidths and donuts, shifted
placebo cutoffs, density/sorting behavior, predetermined covariates as placebo outcomes,
and period-specific pretrend kink contrasts. The current API leaves those analysis grids
explicit rather than silently choosing a specification or mechanically declaring a design
valid.
