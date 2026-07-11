# Phase 3 status — SuDDDS (did/) + Prop 99 backtest

Date: 2026-07-11. Plan: [docs/plans/phase-3.md](../plans/phase-3.md). Spec gate (§9 phase 3):
**SuDDDS with every audit repair; Prop 99 recovers (California, 1989) with Table 6.1-consistent
signs; ch. 6 synthetic benchmarks** — met, with the investigated deviations logged below and in
[docs/method_cards/suddds.md](../method_cards/suddds.md).

## What shipped

- **Panel data layer** (`did/panel.py`): categorical coding, quantile bins, cached profile
  ids, unit column with profile fallback.
- **Treatment background** (`did/background.py`): unit effects + standardized-time polynomial
  via pinv/lstsq (Eq 6.4 rank fix), per-profile variance shrinkage with a data-scaled floor
  (audit 24), ridge-logistic Bernoulli path.
- **Scan statistics** (`did/statistics.py`): vectorized double-beta (Eq 6.9, sigma^2-weight
  adjudication), corrected single-Delta profile GLR (audit 15: profiled mu, window-restricted,
  C-tilde/B-tilde, both signs), exact Bernoulli window LLR (audit 19) reusing the phase-2
  masked-offset kernels.
- **MDSS** (`did/mdss.py`): relax-dim repair (audit 13), explicit incumbent, exact
  per-dimension enumeration at small cardinality (audit 16), greedy / wcc / single-delta
  priorities.
- **SuDDDS driver** (`did/suddds.py`): Algorithms 6–7 with the global incumbent (audit 11)
  and min-two-sided-support cutoffs (audit 12); model="auto" follows theta's type (audit 19).
- **Controls** (`did/controls.py`): count-corrected DD (Eq 6.18 typos), unit-level
  scale-invariant synthetic control, GESS argmin (+inf init, audit 14).
- **Effects** (`did/effects.py`): mean-gap estimator with dose normalization (audit 19),
  `DiDEstimatorBackend` protocol (staggered adoption = future backend, spec non-goal),
  two-sided studentized tau randomization test (audit 5: +1-rank, matched shapes, stated
  conditional claim), per-dimension composition placebos with Holm.
- **Validation** (`validate/panel.py`): fitted-null LLR calibration (audit 1: parametric
  bootstrap, +1-rank, per-replica background refits) with ar1_unit / iid / bernoulli nulls
  (audits 18 + 2); composition and anticipation checks replacing McCrary-in-time (audit 18).
- **Synthetic DGP + benchmarks** (`data/synthetic_did.py`, `benchmarks/run_did_curves.py`,
  CI-small slices in `tests/test_did_benchmarks_small.py`): corrected Eqs 6.22–6.28 DGP,
  Fig 6.1/6.3/6.5 analogs (wcc/single-delta > greedy recall; GESS advantage under the
  heterogeneous DGP).
- **Data + CLI**: `prop99` registry entry with derived state-level DiD columns and
  `natex fetch-data`; `natex discover --design did` with the full validation/effects payload.
- **Prop 99 backtest** (`tests/backtests/test_prop99.py`, 9 tests) and the SuDDDS method card.

## Final gate record (2026-07-11, Apple Silicon macOS arm64, Python 3.13.14)

1. `uv run ruff check src tests` — `All checks passed!`
2. `uv run pytest -q` — `270 passed, 24 deselected in 84.70s` (unit/property/CI-small,
   no backtests; 32 test modules).
3. `NATEX_DATA=".../RDD/data" uv run pytest -q -m backtest` — **`24 passed in 126.49s`**:
   all 15 phase-2 rows still green (test_score, academic_probation, ed_inpatient,
   egger_koethenbuerger) plus the 9 new prop99 tests. No regressions.

Prop 99 module wall-clock ≈ 84 s of the 126 s (target < 5 min): the Q=99 Bernoulli-null LLR
calibration is 76 s (each replica reruns the full 3-window × 8-restart scan and refits its
own background); everything else is sub-second.

## Prop 99 results (run of record; details in the method card)

| Check | Result |
|-------|--------|
| Recovery, normal model (greedy / wcc / single_delta) | (California, 1989.0) exact; precision = recall = 1.0; LLR 13.82 |
| Recovery, Bernoulli model (wcc) | (California, 1989.0) exact; precision = 1.0; LLR 13.86 |
| LLR significance (Bernoulli scan, Bernoulli(p_hat) null, Q=99) | p = 0.010 (null max 7.95 vs observed 13.86) |
| ar1_unit null on the same panel | p = 1.0 — structurally conservative on a deterministic policy dummy (pinned as a regression) |
| dd effect | tau = −27.35 (band −30..−20), pre-MSE 51.2 |
| synthetic effect | tau = −19.51 (band −25..−14), pre-MSE 2.74 < dd's; donors ≈ ADH's (Utah/Montana/Nevada/Connecticut) |
| gess effect | tau = −26.65 (band −32..−20), control = Montana, pre-MSE 18.4 |
| tau placebo ranks (38 enumerated states, two-sided studentized) | dd 13/39 (p 0.33), synthetic 7/39 (p 0.18), gess 9/39 (p 0.23) — all pinned |
| Composition / anticipation / dimension placebos | pass (p = 1 each; balanced panel, constant pre residuals, time-invariant covariates) |
| y-blindness | outcome-None scan bitwise equal (llr, t0, mask) |

## Deviations log (investigated, not silently widened)

1. **`degree=0` background for Prop 99** (protocol, documented in the backtest docstring and
   method card): theta is a pure policy dummy, so any global time polynomial partially
   absorbs the jump and leaks it into all control units' residuals — degree=1 manufactures a
   spurious whole-panel t0=1980 optimum (LLR 8.53) that traps greedy/wcc. The thesis never
   reports its background (spec §10 risk).
2. **Effect magnitudes ~2–2.5× Table 6.1**: natex estimates on the full 1989–2000 post
   period (Eq 6.18 as printed, count-corrected; dd equals the independently verified 2×2 DD
   −27.35); the thesis's printed −10.94/−8.96/−6.67 match a ~5-year effective post window
   (natex 1989–1993 restrictions: dd −18.8, synthetic −12.3; symmetric-W=5 2×2: −12.2). The
   thesis reports neither its estimation window nor its exact panel ("30 US states" vs ADH's
   39). Bands were re-centered on the investigated values; signs and pre-fit ordering agree.
3. **"All significant at 5%" not reproduced**: under the audit-5 corrected two-sided
   studentized placebo test with 38 enumerated placebos, p ≤ 0.05 requires rank 1/39;
   genuinely non-parallel placebo states (Missouri, West Virginia) out-t California. Ranks
   pinned as regressions; the surviving 5% claim is the discovery calibration (Bernoulli
   null, p = 0.01).
4. **ar1_unit null structurally conservative on deterministic treatments** (p = 1.0) — the
   pooled AR(1) absorbs the step as noise; documented in `validate/panel.py` limits and
   pinned by `test_ar1_null_documented_conservative`.
5. **Synthetic control outcome-only** (no ADH covariate V-weights) — phase-3 scope; the fit
   still lands on ADH's donor states.
6. **Poisson count-treatment model deferred**; **staggered adoption = interface only**
   (`DiDEstimatorBackend`); both spec non-goals.

## Repairs made during the backtest (each TDD'd as a unit regression first)

- `synthetic_control`: scale-normalized SLSQP objective (raw-scale SSE stalled the optimizer
  at its uniform start — pre-MSE 257 vs 2.74 after the fix).
- `tau_randomization_test`: placebo discoveries constrain the same dims as the observed
  discovery (audit-5 matched shapes; full-profile seeding starved 38/38 prop99 gess
  placebos).
- `_studentized`: exact zero movement (tau = 0, se = 0) scores 0.0 instead of NaN — provably
  absent composition movement (time-invariant covariates) is the least extreme outcome, not
  a failure.

## Follow-ups for phase 4+

- Thesis Fig 6.7 rho-noise robustness curve on Prop 99 (`benchmarks/run_did_curves.py
  --prop99-noise`) — not run this phase; the rho-injection machinery exists in the synthetic
  benchmarks.
- Stanford Open Policing replication (thesis §6.4.4) — no local data; candidate backtest.
- Poisson observation model for count treatments; Callaway–Sant'Anna backend behind
  `DiDEstimatorBackend`.
- `natex.discover(design="auto")` unified RDD + DiD ranking (phase 6 analyst pass).
- Consider an ADH-style post/pre-MSPE-ratio variant of the tau placebo statistic as an
  additional documented option (investigated here: it also does not reach rank 1/39 on
  prop99 with the outcome-only synthetic fit — near-perfectly pre-fitted placebos like
  Missouri dominate the ratio).
