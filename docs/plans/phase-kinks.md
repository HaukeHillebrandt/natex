# Kink-design implementation plan — RKD and difference-in-kinks

## Objective

Add first-class estimation support for known-cutoff regression kink designs (RKD) and
difference-in-kinks designs (DiK), covering sharp and fuzzy policy schedules. The public
surface consists of Python estimators, reproducible synthetic DGPs, and a `natex kink` CLI
command. This phase estimates a user-specified policy kink; it does not scan arbitrary
cutoffs for kinks because identification requires a known policy schedule and cutoff.

The DiK estimand follows Böckerman, Jysmä, and Kanninen (2025), equations 7, 9, and 10:
the post-minus-pre change in the right-minus-left outcome slope, divided by the analogous
known policy-slope change (sharp) or estimated treatment-slope change (fuzzy). RKD is the
same ratio without the post-minus-pre contrast.

## Statistical contract

- Fit separate local-polynomial intercepts and slopes on each side (and each pre/post cell
  for DiK), with a user-supplied bandwidth and triangular, uniform, or Epanechnikov kernel.
- Express every slope in the original running-variable units even though the internal design
  scales distance by the bandwidth for numerical stability.
- Permit additive numeric covariates and optional cluster-robust inference. Use HC1 by
  default and CR1 when clusters are supplied.
- For fuzzy designs, compute the reduced-form/first-stage covariance and use the full delta
  method for the ratio. Always report the first-stage slope contrast, Wald F, and weak-design
  flag, plus a Fieller confidence set that may honestly be interval, disjoint, or unbounded.
- Require a nonzero user-supplied policy kink for sharp designs. Degenerate fitted designs
  return NaN estimates with a reason, never a fabricated zero.
- Drop only rows non-finite in variables used by the requested fit, and report effective cell
  counts and total row loss.

## Identification guardrails

- RKD requires continuity of the non-policy component's derivative and of the treatment
  response at the known cutoff, plus a nonzero policy kink.
- DiK replaces derivative continuity with parallel changes in the non-policy slopes and also
  requires a time-stable treatment response at the cutoff.
- The paper's fuzzy DiK proof integrates period-specific latent policy heterogeneity as though
  its cutoff distribution were stable across periods. The implementation and method card
  therefore state an additional composition-stability (or valid reweighting) assumption and
  the same-sign individual kink-change condition; software cannot test either condition.
- There is no automatic DiK bandwidth selector in the paper. The CLI therefore requires an
  explicit bandwidth and tells users to report bandwidth, donut, and placebo-cutoff
  sensitivity rather than presenting a guessed optimum.

## Tasks

1. Add failing unit tests for exact sharp RKD/DiK contrasts, noisy sharp and fuzzy recovery,
   weak first stages, Fieller sets, HC1/CR1 inference, missing rows, and input validation.
2. Implement `natex.kink` local-polynomial estimation and export it from the package root.
3. Add seeded sharp/fuzzy RKD and DiK DGPs, including a stable confounding kink that biases
   a post-period RKD but cancels in DiK.
4. Add failing CLI tests, then implement `natex kink` with strict sharp/fuzzy argument
   validation and NaN-clean `kink.json` output.
5. Add a method card, README/API/CLI examples, agent-facing command documentation, and a
   phase status record. Keep the known-cutoff boundary and fuzzy-DiK caveat prominent.
6. Run targeted tests after each green cycle, then ruff and the full non-backtest suite; review
   the final diff without folding unrelated review findings into the feature.

