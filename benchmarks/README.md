# Synthetic benchmarks (KDD-2018 ch.5 evaluation curves)

`run_nig_curve.py` reproduces the LoRD3 paper's synthetic evaluation protocol
on natex's implementation. Outputs land in `benchmarks/out/` (gitignored).

```sh
uv run python benchmarks/run_nig_curve.py --kind both --label-noise
```

CSV outputs: `nig_curve_real.csv`, `nig_curve_binary.csv`, `label_noise.csv`
(one row per sweep cell; see `natex.benchmarks.CURVE_COLUMNS`). If matplotlib
is installed (`uv sync --extra plot`), PNG line charts are also written;
without it, plotting is silently skipped — matplotlib stays an optional extra.

## Paper protocol

- 50 experiments per zeta, each on a fresh synthetic dataset with n = 1000,
  k = 50, tau = 5, unobserved confounder u ~ U(0, 1), heteroskedastic noise,
  random axis-aligned corner region (Eqs 17–22 with the audit's log-odds
  correction for binary T).
- zeta swept over [0, 2.5] for real-valued T and up to ~5 for binary T;
  background polynomial orders 1, 2, 4; for binary T both the Normal and
  Bernoulli observation models are scanned (Fig 7 comparison).
- p-values are +1-rank fitted-null Monte Carlo (Q = 99 here; a parametric
  bootstrap, never "exact"); power = fraction of experiments with p <= alpha
  at alpha = 0.05.
- Label-noise protocol: sharp-ish binary synthetic (T = D via large zeta),
  T replaced by T_rho with P(T_rho = T) = rho in [0.5, 1]; 25 experiments per
  rho on n = 2000 points; top-1 NIG reported.

## Expected qualitative shapes (read_kdd2018.md)

- NIG and power both increase with zeta (Fig 3).
- Polynomial orders 2 and 4 are mildly worse than order 1 (overfitting) but
  comparable — robustness to background misspecification.
- On binary T, the Bernoulli model dominates the Normal model on NIG for all
  specifications (Fig 7); the Normal model produces spurious high-LLR regions.
- tau-hat is overestimated at low zeta (no real discontinuity to exploit) and
  converges to the true tau = 5 as zeta grows (Figs 4, 8).
- Label-noise NIG rises with rho: ~no signal at rho = 0.5, near-sharp
  recovery well before rho = 1 (Figs 12–13).

Small seeded slices of these curves run in default CI:
`tests/test_benchmarks_small.py`.

The changepoint-baseline comparison (KDD Fig 9) is out of scope for phase 2.
