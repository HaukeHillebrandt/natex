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

It prints a 5-line summary and writes `out/results.json` with the top-20 discoveries
(center values, LLR, hyperplane normal, per-variable forcing influence), the scan
p-value, placebo/density validation p-values, effect estimates (2SLS + Wald with
first-stage diagnostics), and the parameters/seed used.

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
at a local data directory (datasets are not committed):

```bash
export NATEX_DATA="/path/to/RDD/data"   # contains test_score_2012/RDD_Guide_Dataset_0.csv
uv run pytest tests/backtests -m backtest -q
```

The first backtest (MDRC RDD practice dataset, `test_score_2012`) checks that natex
rediscovers the pretest-215 cutoff without being told it exists, that pretest dominates
the discovered hyperplane normal, that the scan is significant under the randomization
test, and that the 2SLS effect brackets the known τ ≈ 10 with a strong first stage.

## Corrections vs the papers

natex deviates from the published LoRD3 papers and their released code wherever the math
audit found errors. The audit file [docs/math_audit_final.md](docs/math_audit_final.md) is
the governing document; headline corrections implemented in this phase:

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

Phase 1 (this repo state) delivered the scaffold, the corrected scan core, LoRD3
discovery, the validation battery, 2SLS estimation, the intake profiler, the CLI, and the
first real-data backtest. Next:

| Phase | Scope |
|-------|-------|
| 2 | Remaining RDD backtests + synthetic benchmarks (NIG/precision-recall curves); gate: all RDD backtest rows pass |
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
