# natex

**Automated natural-experiment discovery**: find, validate, and estimate regression
discontinuities, difference-in-differences designs, instrumental variables, and
synthetic-control donor pools in any tabular dataset. natex is a modern reimplementation of the LoRD3 lineage — Herlands, Moraffah,
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
on PyPI). Optional extras: `natex-discovery[plot]` (figures and benchmark charts),
`natex-discovery[report]` (jinja2 paper templates for `natex paper`),
`natex-discovery[paperbanana]` (methodology diagrams; needs its own provider key),
`natex-discovery[ml]` (econml causal forest for the DEE observational layer),
`natex-discovery[gp]` (GPyTorch/botorch GP backend for scale),
`natex-discovery[llm]` (Anthropic/Gemini API guidance backends for the analyst pass) —
everything runs on core deps without them.

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

Don't know the treatment column, or facing a messy CSV? Run the Stage-0 analyst pass
first and let it plan the scan ([method card](docs/method_cards/llm_analyst.md)):

```bash
uv run natex study data.csv --context "county-level school funding, 2004-2012" \
  --backend null --seed 0 --out out/
uv run natex discover --plan out/intake_report.json --seed 0 --out out/
```

`natex study` profiles the data, infers column roles and dataset shape, applies a
declarative prep plan (drops, filters, optional seeded subsample — user-editable at
`out/prep_plan.json`), and ranks candidate designs into `out/intake_report.json`.
`natex discover --plan` then scans the ranked candidates first and the exhaustive
remainder after, within budget — the report always records what was and wasn't searched
(`scanned` / `skipped_budget` / `failed` / `invalid` per configuration; budget cuts are
listed, never silently dropped).

The default `--backend null` is deterministic, offline heuristics — no network, no API
key. `--backend agent` writes each question as a JSON file under `out/guidance/requests/`
and waits for a matching response file (zero-cost guidance from a calling coding agent);
`--backend anthropic|gemini` use the respective APIs (`pip install
'natex-discovery[llm]'`). Guidance is advisory only: it orders the search and annotates
the results (`advisory` blocks, veto flags), but the statistics are bitwise identical
with and without it. Every request+response lands in `out/guidance_log.jsonl`.

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

### DEE: debias an observational CATE estimator

Turn the discovered discontinuities into disjoint quasi-experiments and use their local
2SLS effects to debias a conditioned-on-observables estimator
([method card](docs/method_cards/dee.md)):

```python
from natex import dee_debias

res = dee_debias(ds, query=ds.Z[:200], discoveries=res, m_prime=25,
                 rng=np.random.default_rng(1))
print(res.cate_debiased)        # cate_raw - bias-GP posterior mean at the query points
print(res.weights.w_debias)     # stacked weight on the debias model vs the direct GP
```

The same pipeline runs from the CLI: `natex debias data.csv --treatment T --outcome y
--m-prime 25 --out out/` writes `out/dee_result.json` (weights, per-experiment effects
table, raw/debiased/direct/mixture grid predictions, diagnostics).

### IV: instrument search with honest post-selection inference

Belloni-style plug-in Lasso selection over a candidate instrument pool
([method card](docs/method_cards/iv_sc.md)) — selection reads only the treatment, pool,
and controls, never the outcome:

```bash
uv run natex instruments data.csv --treatment T --outcome y \
  --pool z1,z2,z3,z4,z5 --controls x1,x2 --seed 0 --out out/
```

By default (`--honest`) instruments are selected on a discovery half and the effect is
estimated on the other half — 2SLS with HC1 errors, the closed-form Anderson–Rubin/Fieller
confidence set (reported honestly as `interval`/`disjoint`/`unbounded`/`empty`, never
coerced to a finite interval), and the Hansen J overidentification diagnostic (`null` when
just-identified; exclusion itself is untestable). `out/instruments.json` carries the
selection block (names, λ, loadings, first-stage F, weak flag), the split sizes, and the
estimation block. The same API is available as
`natex.iv.pipeline.discover_instruments` / `natex.iv.search.select_instruments` /
`natex.estimate.iv2sls.iv_2sls`.

### SC: synthetic-control donor selection and placebo inference

Abadie–Diamond–Hainmueller donor selection on a long panel — pre-trend scoring, simplex
weights, post-period ATT, and the in-space RMSPE-ratio placebo test with +1-rank p-values
(deterministic; no rng anywhere in the donor path):

```bash
uv run natex donors smoking.csv --outcome cigsale --unit state --time year \
  --treated-unit California --t0 1989 --out out/
```

`out/donors.json` records ranked donor scores, weights, the counterfactual gap by time,
pre/post RMSPE, `att_post`, and the placebo block (`--exclude-poor-fit MULT` opts into
ADH's poor-pre-fit exclusion). Python API: `natex.iv.donors.select_donors` /
`sc_placebo_test`.

## From discovery to paper

The reporting layer turns a finished run into a results bundle, publication figures, and
an AI-drafted paper skeleton. Three commands end to end:

```bash
natex study data.csv --context "where the data came from" --out out
natex discover data.csv --plan out/intake_report.json --out out
natex paper --bundle out --format md     # or --format latex (compiles when tectonic is installed)
```

`natex paper` accepts any of: a saved bundle directory (`results.json` written by
`ResultsBundle.save()`), a `natex discover --out` directory (`discover_report.json`), or a
single-scan `results.json` from the plain `natex discover` path. `--format md` writes
`paper/paper.md` (always works); `--format latex` writes `paper/paper.tex` and compiles it
to `paper.pdf` when [tectonic](https://tectonic-typesetting.github.io) is on `PATH` — a
missing or failing compiler prints a message and leaves the `.tex`, never an error.
Rendering needs the `report` extra, figures the `plot` extra, and the optional methodology
diagram the `paperbanana` extra (which needs its own image-model provider key):

```bash
uv add 'natex-discovery[report]'         # jinja2 — natex paper / render_paper
uv add 'natex-discovery[plot]'           # matplotlib — figures
uv add 'natex-discovery[paperbanana]'    # optional method diagram (own provider key)
```

### Python API

```python
from natex.report import ResultsBundle, render_paper, research_brief
from natex.report.figures import rdd_figures  # or did_figures

bundle = ResultsBundle.from_discover(report, "out/", dataset=ds, intake=intake, seed=0)
bundle.save()                                # out/results.json + figures/ + paper/
figs = rdd_figures(bundle, ds, res)          # PNG+PDF per figure, manifest in results.json
draft = render_paper(bundle, format="md")    # out/paper/paper.md
brief = research_brief(bundle, "out/")       # out/research-brief.md
```

`results.json` is JSON-native (NaN → null) and records the natex version, seed,
parameters, full search coverage (`scanned` / `skipped_budget` / `failed` / `invalid` per
configuration), the guidance-log path, and the figure manifest — every number the paper
renders comes from this one file. `rdd_figures` (discovery scatter, signed-distance
density histogram, effect forest) and `did_figures` (pre-trend plot, effect forest) each
save PNG (150 dpi) + PDF and register themselves in the bundle. Missing numbers render as
"—", never `nan`.

### Deep-research handoff

`research_brief` writes `research-brief.md`: a self-contained brief (data context,
discovered designs, effects, validation status, and numbered literature questions) meant
to be pasted verbatim into a deep-research agent — e.g. a Gemini Deep Research query — to
retrieve related work for the draft's related-work section. natex performs no research
calls itself; the handoff is a text file, and what comes back is for you to vet and merge.

### Getting the draft into Google Docs

natex does **not** integrate the Google Docs API. The manual route: render markdown
(`natex paper --bundle out --format md`) and paste `paper/paper.md` into a Google Doc, or
upload the `.md` file to Google Drive and choose "Open with → Google Docs" to convert it.

### Human in the loop

Every rendered draft — markdown and LaTeX — opens with the banner
"AI-generated draft — verify all claims before circulation", and the CLI repeats the
warning on every run. The draft is a starting point: check every number against
`results.json`, read the validation section skeptically, and review all claims before the
draft is shared or submitted anywhere.

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
Phase 2 delivered the remaining RDD backtests, the synthetic benchmark suite, and the
scaling engineering ([status](docs/status/phase-2.md)); phase 3 the SuDDDS DiD scan and
the Prop 99 backtest ([status](docs/status/phase-3.md)); phase 4 the DEE debiasing layer
([status](docs/status/phase-4.md)); phase 5 the IV/SC discovery layer
([status](docs/status/phase-5.md)); phase 6 the LLM analyst pass and guidance backends
([status](docs/status/phase-llm-analyst.md)); phase 7 (this repo state) the reporting and
paper pipeline ([status](docs/status/phase-report-paper.md)). Next:

| Phase | Scope |
|-------|-------|
| 2 | **Done** — remaining RDD backtests + synthetic benchmarks (NIG/power curves); gate met: all RDD backtest rows pass ([status](docs/status/phase-2.md)) |
| 3 | **Done** — SuDDDS difference-in-differences discovery (`did/`) + Prop 99 backtest; gate met: (California, 1989) recovered with Table 6.1-consistent signs ([status](docs/status/phase-3.md)) |
| 4 | **Done** — DEE debiased-effect-estimation layer (`dee/`) + scaled simulation-1 benchmark; gate met: mixture beats the raw causal-forest MSE in every config's median ([status](docs/status/phase-4.md)) |
| 5 | **Done** — IV/SC discovery (`iv/`): BCCH plug-in Lasso instrument search, honest 2SLS/J/AR estimation, SC donor selection + in-space placebo; gate met: Prop 99 donor backtest recovers the ADH pool (weight 0.955 on ADH's five donors, ATT −19.5) ([status](docs/status/phase-5.md)) |
| 6 | **Done** — LLM analyst pass (`natex study` → `natex discover --plan`) + scan guidance (Null/Agent/Anthropic/Gemini backends, `[llm]` extra) with the blind-vs-informed eval scaffold; gate met: guidance provably never alters a statistic, coverage always reported ([status](docs/status/phase-llm-analyst.md)) |
| 7 | **Done** — reporting & paper pipeline (`report/`): results bundle, standard figures, jinja2 md/LaTeX drafts with the AI-draft banner, `natex paper`, deep-research brief, paperbanana adapter; gate met: bundle → figures → paper renders end to end for rdd and did, markdown always, LaTeX compiling under tectonic ([status](docs/status/phase-report-paper.md)) |
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
