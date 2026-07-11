# Phase 2 status — RDD backtests, synthetic benchmarks, scaling engineering

Date: 2026-07-11. Plan: [docs/plans/phase-2.md](../plans/phase-2.md). Spec gate (§9 phase 2):
**all RDD backtest rows of spec §8 pass** — met (Egger stretch row passes strict, see below).

## What passed

- **Dataset registry and loaders** (`natex.data.registry`): all five benchmark datasets keyed
  on `NATEX_DATA`, with row-count verification, glob fallback for the Egger download suffix,
  and fetch instructions surfaced by `natex datasets` and by every skip/error message.
- **NaN-outcome-tolerant estimators**: `local_2sls` / `wald_estimate` drop missing `y`
  rowwise (never zero-fill), report `n_used`, and return NaN — never 0.0 — when underpowered.
- **Scaling engineering**: vectorized Bernoulli LLR across splits (with boundary-suprema and
  bracketed-Newton parity to the scalar kernel), scan-geometry cache with Kmax-NN prefix
  reuse shared across randomization replicas, homogeneous-treatment fast path, and the
  seeded coarse-to-fine scan with the §6b coverage contract (`frac_centers_scanned` + full
  params always reported).
- **Synthetic DGP fidelity + benchmark harness** (KDD-2018 ch.5, Eqs 17–22 with the audit's
  binary log-odds correction): NIG/power/τ̂ curves and the label-noise protocol
  P(T_ρ = T) = ρ; CI-small seeded slices run by default, full curves via
  `benchmarks/run_nig_curve.py`.
- **Discovery clustering** (`cluster_discoveries`) for multi-cutoff assertions.
- **CLI**: `natex datasets` (registry found/missing table, exit 0 always) and
  `natex discover --degree/--coarse/--n-coarse` with the `coarse` coverage block in
  `results.json`.
- **All backtest rows** (details below).

## Backtest results and wall-clock times

Run of record (final gate, 2026-07-11, Apple Silicon macOS arm64, Python 3.13.14):

```
NATEX_DATA=".../RDD/data" uv run pytest tests/backtests -m backtest -q --durations=0
15 passed in 52.61s
```

(Times below are pytest `--durations` wall-clock; a single-core synthetic-benchmark job ran
concurrently, so they are conservative upper bounds.)

| Backtest | Result | Wall-clock |
|----------|--------|-----------|
| `test_test_score.py` (MDRC sharp RDD, 2,767 rows) | pass — pretest-215 cutoff rediscovered, τ ≈ 10 bracketed | 7.9 s (7.62 s of it the Q = 19 randomization test) |
| `test_academic_probation.py` (LSO 2010, 44,362 rows) | pass — `dist_from_cut` ranked #1 of 4; cutoff at 0 recovered; 2SLS sign matches Lindo +0.233 | 38.6 s total: full 44k coarse-to-fine scan 4.70 s, 8k-subsample Q = 19 randomization 31.76 s, k = 150 effect rescan 2.18 s |
| `test_ed_inpatient.py` (ADG 2012, 161 + 73 cells) | pass — ages 19 and 23 (and the paper's 16y10m) among top clusters; density negative control holds | 1.1 s (both files, incl. two Q = 99 randomization tests) |
| `test_egger_koethenbuerger.py` (42,005 rows after protocol cap) | pass strict (stretch) — see below | 4.9 s (one shared coarse-to-fine scan, k = 100) |

The 44,362-row LSO scan at 4.7 s is the headline of the phase-2 scaling work (vectorized
Bernoulli LLR + geometry cache + coarse-to-fine): the plan's budget for the whole LSO file
was ~10 minutes.

## Egger–Köthenbürger outcome (stretch goal, spec §10)

**Passes strict — no xfail, zero tuning.** With the protocol restriction `wpop < 20000`
(fixed before scanning; 42,005 of 43,175 rows), `coarse_to_fine_scan(k=100, n_coarse=3000,
top_m=30, seed 0)` on `log_pop` under the normal model finds **4 of the 5 statutory
population thresholds** — {2001, 3001, 5001, 10001} — as top-15 cluster representatives
within 15% (top cluster populations 4994 / 9992 / 1999 / 2995; max LLRs ≈ 70456 / 6085 /
5716 / 138). They are the *only* four clusters formed by the top-50 discoveries. The fifth
threshold (1001) is not lost: its best discovery is overall rank 54 (pop 1000, LLR ≈ 116),
i.e. cluster #5 when clustering the full discovery list — all five statutory thresholds are
the top five clusters — it merely falls outside the plan's top-50 cut. Full coarse-to-fine
scan wall-clock: ≈ 4 s.

## Benchmark curve summary (endpoints)

Seeded endpoint runs of record (n = 1000, k = 50, τ = 5, degree 1, Q = 49 fitted-null
replicas, 15 experiments per cell; label noise n = 2000, 10 experiments per ρ):

```
uv run python benchmarks/run_nig_curve.py --kind real --zetas 0 2.5 --degrees 1 \
    --n-experiments 15 --Q 49 --seed 0
uv run python benchmarks/run_nig_curve.py --kind binary --zetas 0 5 --degrees 1 \
    --n-experiments 15 --Q 49 --seed 0 --label-noise --rhos 0.5 1.0 --noise-experiments 10
```

p-values are +1-rank Monte Carlo (parametric bootstrap, never "exact"); power = share of
experiments with p ≤ 0.05.

| Curve | Endpoint | NIG (mean ± SE) | Power | Notes |
|-------|----------|-----------------|-------|-------|
| real T, normal model | ζ = 0 | 0.086 ± 0.039 | 0.07 | ≈ α, as it must be under the null; mean p = 0.50 |
| real T, normal model | ζ = 2.5 | 0.866 ± 0.032 | 1.00 | τ̂₂ₛₗₛ median 4.95 (true 5) |
| binary T, Bernoulli model | ζ = 0 | 0.038 ± 0.020 | 0.00 | mean p = 0.66; normal model: NIG 0.058 ± 0.034, power 0.07 |
| binary T, Bernoulli model | ζ = 5 | 0.873 ± 0.032 | 1.00 | normal model 0.871 ± 0.023 (Fig-7 direction Bernoulli ≥ Normal holds, margin small at this saturated endpoint); τ̂₂ₛₗₛ median 5.04 (true 5) |
| label noise | ρ = 0.5 | 0.093 ± 0.073 | — | no signal by construction |
| label noise | ρ = 1.0 | 0.926 ± 0.033 | — | near-sharp recovery |

Full-run wall-clock at these settings: real endpoints 43 s; binary endpoints + label noise
11 min 19 s (CSV outputs kept under `benchmarks/out/`, gitignored). The full paper grid
(six ζ values, degrees 1/2/4, 50 experiments, Q = 99) is a
`benchmarks/run_nig_curve.py --kind both --label-noise` run, sized for an unattended
session rather than CI; CI runs the seeded small slices in
`tests/test_benchmarks_small.py`.

## Deviations and caveats (documented, not silent)

- **LSO randomization on a subsample**: fitted-null significance for the 44k-row scan is
  asserted on a seeded 8,000-row subsample with Q = 19 (full-sample Q-replica scan is out of
  compute budget). The claim is conditionally valid for that subsample's scan distribution,
  not the full-sample statistic.
- **LSO estimation bandwidth**: at detection bandwidth (k = 75) the single-neighborhood 2SLS
  is noise-dominated (HC1 SE ≈ 0.36 vs true ≈ 0.23), so the effect test rescans at the
  discovered center with k_est = 150 (detection pins location, estimation widens the local
  sample) — standard RD practice, stable in k_est.
- **ED estimand caveat**: on the unweighted 161 aggregate month cells the age-19 cutoff
  legitimately dominates (−6.2 pp vs −0.8 pp jump) and the 23rd-birthday cluster ranks 4th;
  the paper's "strongest RDD at 23y3m" comes from 2.2M individual-level rows whose implicit
  weighting does not transfer. Only the sign of the local effect is asserted. The
  trend-adjusted 2SLS first stage trips the weak-instrument rule of thumb on power grounds;
  instrument relevance is asserted through the trend-free Wald first stage (t ≈ 8).
- **Inpatient small-n**: with 73 cells the randomization test is power-limited; the
  significance threshold is 0.10 by design.
- **Egger protocol restriction**: rows with `wpop ≥ 20000` are excluded before scanning
  (heavy right tail drowns the low statutory thresholds); fixed in the plan before the scan
  ran, and the 1001 threshold ranks outside the top-50 discovery cut (see above).
- **Coarse `discover` validation**: with `--coarse`, the randomization test's null replicas
  rescan all centers while the observed max-LLR comes from the fine-stage subset — a
  conservative (never anti-conservative) comparison.

## Final gate record (2026-07-11)

1. `uv run ruff check src tests` — `All checks passed!`
2. `uv run pytest -q` — `92 passed, 15 deselected in 70.12s` (no backtests; includes the
   CI-small benchmark slices and the new CLI tests; comfortably under the ~2 min added-CI
   budget)
3. `NATEX_DATA="/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data" uv run pytest tests/backtests -m backtest -q` —
   `15 passed in 52.61s` (all spec §8 RDD rows, Egger passing strict under its documented
   stretch contract)
