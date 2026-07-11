# Phase 5 status — IV/SC discovery (iv/)

Date: 2026-07-11. Plan: [docs/plans/phase-5.md](../plans/phase-5.md). Spec gate (§9 phase 5,
§5 items 4 & 10, adopted AR/Fieller improvement): **Belloni-style instrument search +
strength/exclusion diagnostics, weak-IV-robust AR/Fieller sets at discovered cutoffs, honest
discovery/estimation pipeline, and SC donor selection with pre-trend scoring, gated by the
Prop 99 donor backtest** — met (ADH donor pool recovered with summed weight 0.955 on ADH's
five donors, ATT −19.5, placebo rank 3/39), with the deviations logged here and in
[docs/method_cards/iv_sc.md](../method_cards/iv_sc.md). This entire phase is core-deps
(sklearn Lasso, scipy optimize/stats); no new dependencies, no new extras.

## What shipped

- **General k-instrument 2SLS** (`estimate/iv2sls.py`): HC1 sandwich (audit item 4 — the
  printed group-instrument form is never implemented), Hansen J with the audit-item-3 scope
  statement (`j_p = None` when just-identified, never a fabricated value), always-on
  first-stage HC1 F / partial R² / weak flag (audit item 10).
- **Closed-form Anderson–Rubin/Fieller sets** (`ar_confidence_set`, audit §3 adopted): the
  quadratic set inversion with all four kinds reported honestly (interval / empty / disjoint /
  unbounded); k = 1 is exactly Fieller; wired additively into `IVEstimate` and — at discovered
  RD cutoffs — `EffectEstimate` via `local_2sls` (`ar_ci`/`ar_kind`).
- **Sparse-first-stage IV DGP** (`data/synthetic_iv.py`): BCCH exponential design with a
  targeted population concentration μ², Toeplitz-correlated pool, tunable endogeneity, and
  plantable exclusion violators (last `n_invalid` columns, exclusion-only).
- **Belloni plug-in Lasso selection** (`iv/search.py`): pinned λ/γ/c/loadings/iteration and
  the exact sklearn mapping (see the method card), post-Lasso OLS, scale-relative
  zero-variance drop (audit 24), honest empty selection (NaN diagnostics, `weak=True`).
- **Honest pipeline** (`iv/pipeline.py`): select on the discovery half, estimate + J + AR on
  the estimation half (audit item 1); `honest=False` carries an explicit caveat string.
- **Shared simplex fitter** (`estimate/simplex.py`): extracted verbatim from `did/controls.py`
  with the phase-3 scale-normalization and `MISSING_W_TOL` conventions regression-tested at
  both levels.
- **SC donor selection** (`iv/donors.py`): unit×time matrix, balanced-donor rule, pre-only
  RMSE/corr scoring (mutation-tested post-y blindness), top-k pool, simplex weights,
  counterfactual + post-period ATT, and the deterministic in-space RMSPE-ratio placebo test
  with +1-rank p (audit items 1/5 lineage; `exclude_poor_fit` opt-in).
- **Factor-model SC DGP** (`data/synthetic_sc.py`): random-walk factors, treated unit an
  exact convex combination of an edge-clustered true donor set (identifiability design
  documented in the module docstring).
- **CLI**: `natex instruments` (selection + honest estimation JSON) and `natex donors`
  (donor/ATT/placebo JSON), NaN serialized as null.
- **Benchmarks** (`benchmarks/run_iv_selection.py` + CI-small slices in
  `tests/test_iv_benchmarks_small.py`): μ² ∈ {8, 30, 80, 180, 400} × 20 seeds IV block;
  noise ∈ {0.25, 0.5, 1.0} × 20 seeds SC block.
- **Prop 99 donor backtest** (`tests/backtests/test_prop99_donors.py`) + the optional Egger
  IV stretch (one xfail finding, one passing coefficient-mass regression).
- **Docs**: [IV/SC method card](../method_cards/iv_sc.md), README quickstart + roadmap tick.

## Final gate record (2026-07-11, Apple Silicon macOS arm64, Python 3.13.14)

1. `uv run ruff check src tests` — `All checks passed!`
2. `uv run pytest -q` — **`523 passed, 32 deselected in 124.00s`** (unit/property/CI-small,
   no backtests; 125 tests added this phase). No regressions in phases 1–4.
3. `NATEX_DATA=".../RDD/data" uv run pytest -q -m backtest` —
   **`31 passed, 523 deselected, 1 xfailed in 119.81s`**: all 15 phase-2 RDD rows, the 9
   phase-3 Prop 99 SuDDDS tests, and the 8 new phase-5 tests
   (6 Prop 99 donor tests incl. the two placebo variants, 1 Egger coefficient-mass
   regression, 1 non-blocking Egger xfail).
4. Benchmark run of record: `uv run python benchmarks/run_iv_selection.py` (2026-07-11,
   `benchmarks/out/iv_selection.csv` + `sc_recovery.csv`), summarized below.

### Benchmark run of record (20 seeds per cell, n=500, p=50, s=5, endog=0.6)

IV block (`iv_selection.csv`; precision/recall vs `true_support`. An empty selection has
recall 0 but NaN precision/F/bias, so precision / median-F / bias columns average over the
non-empty seeds only):

| μ² | nonempty sel. | precision | recall top-3 | median F | weak rate | median 2SLS bias | median OLS bias | honest AR kind |
|----|---------------|-----------|--------------|----------|-----------|------------------|-----------------|----------------|
| 8 | 3/20 | 1.0 | 0.08 | 14.1 | 0.85 | 0.43 | 0.59 | empty selection 20/20 (plug-in refuses) |
| 30 | 18/20 | 1.0 | 0.48 | 17.9 | 0.10 | 0.20 | 0.57 | 15/20 empty-sel; 5/20 interval |
| 80 | 20/20 | 1.0 | 0.78 | 29.4 | 0.00 | 0.10 | 0.51 | 19/20 interval |
| 180 | 20/20 | 1.0 | 0.98 | 47.9 | 0.00 | 0.06 | 0.44 | 20/20 interval |
| 400 | 20/20 | 1.0 | 1.00 | 89.9 | 0.00 | 0.04 | 0.33 | 20/20 interval |

Honest-half coverage of the true τ where estimable: AR 3/5, 17/19, 19/20, 20/20 vs Wald
4/5, 16/19, 16/20, 16/20 at μ² = 30/80/180/400 — AR ≥ Wald at every strong-identification
cell (nominal 19/20). Precision is 1.0 everywhere the plug-in selects (never a false
positive); at weak concentration the plug-in **refuses** rather than selecting junk — the
weak-regime honesty gap therefore lives at an explicit sub-plug-in λ (CI-small slice below).

SC block (`sc_recovery.csv`; defaults n_units=20, n_pre=15, n_post=10, k_true=3, effect=10,
n_donors=8):

| noise | donor recovery | weight on true | mean abs(ATT err) | max abs(ATT err) |
|-------|----------------|----------------|--------------------|-------------------|
| 0.25 | 0.983 | 0.948 | 0.26 | 0.86 |
| 0.50 | 0.983 | 0.919 | 0.48 | 1.70 |
| 1.00 | 0.983 | 0.874 | 0.85 | 2.71 |

## Calibration evidence for every pinned threshold

Repo statistical-test policy (phase 4): every stochastic assertion calibrated across ≥ 5
seeds during implementation, one seed pinned with margin, observed ranges recorded next to
the assertion. Where each pinned number's evidence lives:

| Pinned threshold / constant | Evidence (observed range, seeds) | Where recorded |
|-----------------------------|----------------------------------|----------------|
| Plug-in λ = 2c√n Φ⁻¹(1−γ/2p), c=1.1, γ=0.1/log(max(n,p)) | analytic soft-threshold pin (max_iter=1, z ∈ {−1,+1}, ψ=1 exactly); BCCH DGP (n=500, p=50, s=5, μ²=180): selection ⊆ true_support at seeds 0–7 (0 false positives, F 38.5–65.3), exact top-3 6/8 seeds (Toeplitz ρ=0.5 absorbs z3 at seeds 0, 3; seed 2 pinned) | `tests/test_iv_search.py` |
| Zero-variance drop `col_ss ≤ raw_ss(n·eps)²` | partialled constants leave ~1e-28 relative SS; audit-24 scale-relative | `tests/test_iv_search.py` |
| weak ⇔ F < 10 (heuristic, not Stock–Yogo) | μ²=8 plug-in nonempty seeds have F 12.2–17.9 (not weak); explicit λ=60 gives F 2.8–5.6, weak=1 in 20/20 | `tests/test_iv_benchmarks_small.py` |
| AR α=0.05 coverage vs Wald | n=250, μ²≈4 (π=√(4/n)), endog=0.95, 200 reps, 5 seed bases: Wald 170–185/200 vs AR 187–195/200 (nominal 190/200; pinned base 20000: AR 188, Wald 173); undercoverage needs HIGH endogeneity — at endog=0.6 Wald covers 189/200 | `tests/test_ar_ci.py` |
| Four AR set kinds handled | μ²≈4 kinds split ≈ 50% interval / 30% disjoint / 20% unbounded (implementation-time calibration); zero-strength seeds 7,17,27,37,47 all unbounded/disjoint (pinned 7); "empty" pinned at seed 11 (k=2, planted violator); coverage helpers handle all four | `tests/test_ar_ci.py`, `benchmarks/run_iv_selection.py` |
| Boundedness ⇔ homoskedastic F > F_crit | strength sweep straddling the threshold, both sides observed | `tests/test_ar_ci.py::test_boundedness_iff_first_stage_f_exceeds_crit` |
| Realized concentration within 25% of target μ² | rel. err ≈ [0.02, 0.10] at n=500 (scale-free in μ²) | `tests/test_synthetic_iv.py` |
| Honest pipeline estimation-half accuracy | n=1000/μ²=180: est.-half abs(τ−1) 0.003–0.179 over seeds 0–7 (0.15 gate pinned at a passing seed); valid-run j_p null-uniform (seed 2 of 0–4 lands 0.034 — avoided) | `tests/test_iv_pipeline.py` |
| Weak-selection honesty gap (explicit λ=60, μ²=8) | full-sample AR "interval" 19/20 (homosk. F > F_crit ≈ 2.2), honest-half AR unbounded/disjoint 17/20, Wald CI finite 40/40 runs; pinned seeds (0, 1, 2) all "unbounded" | `tests/test_iv_benchmarks_small.py` |
| Strong-regime CI-small pins (seeds 0, 2, 4) | recall_top3 = 1.0 (19/20 seeds; miss = seed 1), precision = 1.0 & weak = 0 in 20/20, bias_2sls < bias_ols in 20/20 with margin 0.27–0.43 | `tests/test_iv_benchmarks_small.py` |
| SC DGP identifiable donor cluster (μ=12, λ=1.5, shrink 0.4) | w_true 0.80–1.00 in 10/10 seeds (0.0–0.95 before the edge-cluster fix); treated-vs-combo pre gap pure noise, sd = noise·√(1+w·w) (seed 0 a legit 2.9σ draw — fidelity pins seeds 1–5) | `tests/test_synthetic_sc.py`, `tests/test_donors.py` |
| SC slice gate mean abs(ATT err) < 1.5 (noise 0.5, n_donors 8) | per-seed 0.010–0.921 over seeds 0–9 (mean 0.33); pinned seeds (0, 1, 2) errors (0.921, 0.010, 0.322) | `tests/test_iv_benchmarks_small.py` |
| Placebo p arithmetic + min 5 usable placebos | signal DGP (effect=10, noise=0.5, n=20): p = 1/20 exactly in 10/10 seeds (treated ratio 15.7–29.7 vs max placebo 6.57); null mean p 0.725; `exclude_poor_fit=2` on n_units=12 drops 2–5 units (extreme honest donors trip it too) → exclusion opt-in, `p=NaN` below 5 | `tests/test_donors.py` |
| `MISSING_W_TOL = 0.1`, scale-normalized SSE | inherited phase-3 regressions, re-tested at the helper level after extraction | `tests/test_simplex.py`, `tests/test_did_controls.py` |
| CLI happy-path pins | instruments: DGP (n=600, p=20, s=3, μ²=150, seed 2) + CLI seed 0 → selected ['z2'], F ≈ 45.3, τ ≈ 1.031, ar_kind "interval", j_p null; donors: SC defaults seed 0 → att ≈ 10.92, p = 1/20 | `tests/test_cli_iv.py` |
| Prop 99 pinned bands | att full-pool −19.514 (band [−25, −14]), top-8 −22.648 (band [−30, −15]); ADH-five weight 0.955 (band [0.90, 1.0]); placebo rank 3/39 both variants (gate ≤ 5, p ≤ 4/39); outrankers ⊆ {Missouri, Virginia, Georgia}; bitwise-deterministic rerun | `tests/backtests/test_prop99_donors.py` |

## Deviations log (investigated, not silently widened)

1. **Plug-in selection refuses at weak concentration** (μ²=8: 17/20 empty full-sample, 20/20
   empty on the honest half): designed behavior, not a defect — the weak-IV honesty-gap
   benchmark row therefore holds selection open with an explicit λ=60 and documents it.
2. **Egger stretch (spec §10 stress case)**: strict decoy-free support FAILS
   (`xfail(strict=False)`) — the in-sample council-size first stage is *deterministic*, so
   iterated loadings collapse and the penalty vanishes; the passing regression pins the
   ~4-orders-of-magnitude coefficient-mass separation and exact statutory post-Lasso jumps.
   Finding recorded in the method card: the BCCH plug-in assumes a noisy first stage.
3. **Prop 99 placebo rank 3/39 vs ADH's 1/39**: our protocol fixes `exclude_poor_fit=None`
   and fits outcomes only (no V-weights); Missouri and Virginia out-rank California. Pinned
   as a regression with the reconciliation in the method card — the exclusion stays opt-in.
4. **Pre-RMSE screening ≠ simplex weighting on Prop 99**: Nevada/Utah are excluded from the
   top-8 by level-RMSE yet dominate the full-pool weights (method card note); the backtest
   asserts both behaviors.
5. **Out of scope, documented in the method card** (plan header list): Deep IV / GAN /
   text-matching (Springer roadmap is conceptual only), sup-score robust selection,
   Montiel Olea–Pflueger effective F (HC1 F + the F<10 heuristic instead), SC covariate
   V-weights (phase-3 deviation carried forward), matrix-completion/elastic-net SC,
   staggered adoption.

## Open questions for phase 6

- Should the LLM analyst pass (phase 6) consume `ar_kind` as a first-class caution signal
  when narrating discovered-cutoff effects (an "unbounded" AR set is a strong "do not trust
  the Wald CI" cue)?
- Wire `natex instruments` into `natex discover` output (instrument pools built from
  discovered threshold dummies, Egger-style) — the stretch finding suggests this needs a
  noise-aware guard before the plug-in Lasso is applied to deterministic assignment rules.
- SC donor selection currently exposes no time-placebo (in-time) test; ADH's in-time placebo
  would be a natural phase-6/7 validation-battery addition alongside the reporting pipeline.
- Benchmark follow-up: an `exclude_poor_fit` sensitivity row for the SC block (the Prop 99
  gap's driver) and a μ² grid point between 8 and 30 to trace the plug-in refusal boundary.
