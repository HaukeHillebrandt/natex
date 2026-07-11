# natex

**Automated natural-experiment discovery**: find, validate, and estimate regression
discontinuities (and, in later phases, difference-in-differences designs) in any tabular
dataset. natex is a modern reimplementation of the LoRD3 lineage — Herlands, Moraffah,
McFowland & Neill (KDD 2018), Herlands (PhD thesis, CMU 2019), and Jakubowski et al.
(JMLR 2023) — that searches local neighborhoods of a dataset for treatment-assignment
discontinuities using a log-likelihood-ratio scan, then subjects each candidate to a
validation battery (fitted-null Monte Carlo, placebo, and density tests) before
estimating a local treatment effect by 2SLS.

natex is **not** a line-for-line port. A full mathematical audit of the source papers and
their released code ([docs/math_audit_final.md](docs/math_audit_final.md)) found design-level
errors — an inconsistent instrument, placebo tests that mechanically reject valid designs,
mis-specified null replicas, and a variance-indexing bug in the legacy scan — and natex
implements the corrected versions throughout (see
[Corrections vs the papers](#corrections-vs-the-papers)). Design rules: discovery never
touches the outcome `y` (it enters only in estimation and outcome placebos), every
stochastic step consumes one explicit `numpy.random.Generator` (same seed ⇒ same result),
and failed computations return `NaN`, never `0.0`.

## Install

From GitHub:

```bash
uv add git+https://github.com/HaukeHillebrandt/natex
```

From source:

```bash
git clone https://github.com/HaukeHillebrandt/natex
cd natex
uv sync --extra dev
```

Requires Python ≥ 3.11. Core dependencies: numpy, scipy, pandas, scikit-learn, typer,
pydantic. The import name is `natex` (the distribution is named `natex-discovery`; not yet
on PyPI).

## Quickstart

### CLI

Point `natex discover` at a CSV with a treatment column (binary or continuous) and,
optionally, an outcome column:

```bash
uv run natex discover data.csv \
  --treatment T --outcome y \
  --forcing score,age \
  --k 50 --q 99 --seed 0 --out out/
```

- `--forcing` — comma-separated candidate forcing (running) variables; defaults to all
  numeric non-treatment/outcome columns.
- `--k` — neighborhood size for the local scan; `--q` — number of null replicas for the
  randomization test; `--seed` — RNG seed.
- `--degree` — background polynomial degree of the treatment model (default 1).
- `--coarse` / `--no-coarse` (default off) with `--n-coarse` (default 2000) — coarse-to-fine
  scan for large datasets: a seeded center subsample is scanned first, then the regions
  around the best coarse candidates are rescanned at full resolution. Search coverage is
  never silently truncated: `results.json` gains a `coarse` block reporting
  `frac_centers_scanned` and every coarse parameter.

It prints a short summary and writes `out/results.json` with the top-20 discoveries
(center values, LLR, hyperplane normal, per-variable forcing influence), the scan
p-value, placebo/density validation p-values, effect estimates (2SLS + Wald with
first-stage diagnostics), and the parameters/seed used.

`natex datasets [--root PATH]` (default root: env `NATEX_DATA`) prints one line per
registered benchmark dataset — found/missing, row count, row-count check — and, for
missing ones, the instructions for obtaining the file (see
[Backtests on real data](#backtests-on-real-data)). It always exits 0.

### Python API

```python
import numpy as np

from natex import Dataset, lord3_scan
from natex.estimate.local2sls import local_2sls
from natex.validate.randomization import randomization_test

ds = Dataset.from_csv("data.csv", treatment="T", outcome="y", forcing=["score", "age"])
rng = np.random.default_rng(0)

res = lord3_scan(ds, k=50, rng=rng)                 # discovery: uses only (x, z, T)
rep = randomization_test(ds, res, Q=99, rng=rng,    # fitted-null Monte Carlo, +1-rank p
                         scan_kwargs={"k": 50})
print(rep.p_value)

top = res.discoveries[0]                            # best local discontinuity
est = local_2sls(ds, top)                           # frozen-side 2SLS, HC1 errors
print(est.tau, est.ci, est.first_stage_t, est.weak_instrument)
```

Also available: `natex.validate.placebo.placebo_tests` (intercept-continuity placebo
battery), `natex.validate.density.density_test` (signed-distance McCrary-style density
check), `natex.validate.honest.honest_split` (discovery/estimation split), and
`natex.estimate.local2sls.wald_estimate`.

## Backtests on real data

Unit and synthetic tests run by default (`uv run pytest`). Real-data backtests are marked
`backtest`, deselected by default, and need the `NATEX_DATA` environment variable pointing
at a local data directory (datasets are never committed):

```bash
export NATEX_DATA="/path/to/RDD/data"
uv run natex datasets                   # verify what is present / how to fetch the rest
uv run pytest tests/backtests -m backtest -q
```

Expected layout under `NATEX_DATA` (the registry in `natex.data.registry` is the source
of truth):

```
NATEX_DATA/
├── test_score_2012/RDD_Guide_Dataset_0.csv
├── AcademicProbation_LSO_2010/data_orig.csv
├── ED_visits/P03_ED_Analysis_File.csv
├── Inpatient_visits/P10_Inpatient_CSV_File.csv
└── EggerKoethenbuerger_AEJ_Data.csv        # "... (1).csv" download suffix also accepted
```

The five datasets and the known truths each backtest asserts (natex is never told the
cutoffs — it must rediscover them):

| Dataset | Rows | Design truth asserted |
|---------|------|-----------------------|
| `test_score_2012` — MDRC RDD practice dataset (Jacob, Zhu, Somers & Bloom 2012) | 2,767 | Sharp RDD at pretest = 215 (treatment goes to low scorers); pretest dominates the discovered normal; 2SLS brackets the known τ ≈ 10 with a strong first stage. |
| `academic_probation` — Lindo, Sanders & Oreopoulos (2010), AEJ:Applied | 44,362 | Fuzzy RDD at `dist_from_cut = 0`; `dist_from_cut` ranked #1 of 4 candidate forcing variables; 2SLS sign matches Lindo's +0.233. Runs through the coarse-to-fine scan (the phase-2 scale gate). |
| `ed_visits` — Anderson, Dobkin & Gross (2012), AEJ:Economic Policy | 161 cells | Fuzzy insurance-loss RDDs at ages 19 and 23 (`months_23` = −48 and 0) among the top clusters, plus the paper's age-16y10m discovery; uniform month grid is a density-test negative control. |
| `inpatient_visits` — ADG (2012) companion file | 73 cells | Age-23 cutoff recovered on only 73 aggregated cells (small-n robustness). |
| `egger_koethenbuerger` — Egger & Köthenbürger (2010), AEJ:Applied | 43,175 | ≥ 2 statutory population thresholds of the Bavarian council-size schedule discovered on `log_pop` (stretch goal; observed: 4 of 5 as the top clusters). |

Fetch instructions (also printed by `natex datasets` for anything missing):

- **test_score_2012** — download the practice CSV from
  <https://www.mdrc.org/publication/practical-guide-regression-discontinuity> and place it
  at `test_score_2012/RDD_Guide_Dataset_0.csv`.
- **academic_probation** — AEJ:Applied data archive on openICPSR (login-gated); search the
  archives at <https://www.openicpsr.org/> for Lindo, Sanders & Oreopoulos (2010) and place
  `data_orig.csv` at `AcademicProbation_LSO_2010/data_orig.csv`.
- **ed_visits** / **inpatient_visits** — Anderson, Dobkin & Gross (2012) data archive on
  openICPSR (login-gated); place `P03_ED_Analysis_File.csv` under `ED_visits/` and
  `P10_Inpatient_CSV_File.csv` under `Inpatient_visits/`.
- **egger_koethenbuerger** — Egger & Köthenbürger (2010) data archive on openICPSR
  (login-gated); place the CSV at `EggerKoethenbuerger_AEJ_Data.csv` (a downloaded
  `EggerKoethenbuerger_AEJ_Data (1).csv` filename also works).

Backtest outcomes, wall-clock times, and documented deviations are recorded in
[docs/status/phase-2.md](docs/status/phase-2.md).

## Benchmarks

Synthetic benchmarks reproduce the KDD-2018 ch.5 evaluation protocol (NIG and power vs
discontinuity strength ζ across polynomial orders, τ̂ convergence, Bernoulli-vs-Normal on
binary treatment, and the label-noise protocol P(T_ρ = T) = ρ):

```bash
uv run python benchmarks/run_nig_curve.py --kind real
uv run python benchmarks/run_nig_curve.py --kind both --label-noise   # full protocol
```

CSVs (and PNG line charts, if the optional `plot` extra is installed) land in
`benchmarks/out/` (gitignored). Small seeded slices of the same curves run in default CI
(`tests/test_benchmarks_small.py`); see [benchmarks/README.md](benchmarks/README.md) for
the protocol and the expected qualitative shapes.

## Corrections vs the papers

natex deviates from the published LoRD3 papers and their released code wherever the math
audit found errors. The audit file [docs/math_audit_final.md](docs/math_audit_final.md) is
the governing document; headline corrections implemented from phase 1 onward:

- **Randomization test stated honestly** — it is a fitted-null parametric bootstrap, not an
  exact test; natex reports +1-rank Monte Carlo p-values and offers an honest
  discovery/estimation split.
- **Bernoulli null replicas** are drawn as `Bernoulli(p̂)` — the legacy generator's
  thresholded-Gaussian draw has the wrong success probability.
- **Placebo tests** use local intercept-continuity contrasts (side-specific trends, Holm
  correction) — the papers' side-mean contrasts mechanically reject valid RDDs.
- **Effect estimation** uses frozen side-indicator 2SLS with HC1 errors — the papers'
  group instrument (Eq. 5.14) is inconsistent as printed. First-stage diagnostics
  (jump, t, weak-instrument flag) are always on.
- **Variance estimation** uses each point's *own* k-neighborhood residuals with a
  data-scaled floor (`1e-3 · Var(r)`) — the legacy code indexed reverse neighbors and
  used an absolute `1e-6` floor.
- **Sharp (pure-group) splits** are scored via boundary likelihood suprema instead of
  being dropped as `NA`.
- **Hyperplane tie convention** is explicit: signed distance ≥ 0 ⇒ group 1 (the center
  belongs to group 1).
- **Density falsification** runs on the signed distance along the *frozen* discovered
  normal and is documented as a falsification test only.

Legacy scan outputs are therefore **not** treated as ground truth in parity tests.

## Roadmap

Phase 1 delivered the scaffold, the corrected scan core, LoRD3 discovery, the validation
battery, 2SLS estimation, the intake profiler, the CLI, and the first real-data backtest.
Phase 2 (this repo state) delivered the remaining RDD backtests, the synthetic benchmark
suite, and the scaling engineering (vectorized kernels, geometry caching, coarse-to-fine
scan) — see [docs/status/phase-2.md](docs/status/phase-2.md). Next:

| Phase | Scope |
|-------|-------|
| 2 | **Done** — remaining RDD backtests + synthetic benchmarks (NIG/power curves); gate met: all RDD backtest rows pass ([status](docs/status/phase-2.md)) |
| 3 | SuDDDS difference-in-differences discovery (`did/`) + Prop 99 (California tobacco) backtest |
| 4 | DEE debiased-effect-estimation layer (`dee/`) + scaled simulation benchmark |
| 5 | IV / synthetic-control discovery (`iv/`) from the Springer roadmap |
| 6 | LLM analyst pass + scan guidance (Null/Agent/API backends) with a guided-vs-unguided evaluation on the backtest suite |
| 7 | Reporting & paper pipeline (`report/`) |
| 8 | Agent skills + docs; optional PyPI release |

## Development

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest -q            # excludes backtests by default
```

CI (GitHub Actions) runs lint + the non-backtest suite on Python 3.11–3.14.

## License

MIT — see [LICENSE](LICENSE).
