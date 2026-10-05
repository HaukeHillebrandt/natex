# natex

natex finds, validates and estimates natural experiments in tabular data: regression
discontinuities, difference-in-differences designs, known-cutoff regression kinks and
difference-in-kinks, instrumental variables, and synthetic-control donor pools. It
reimplements the LoRD3 lineage (Herlands, Moraffah, McFowland and Neill, KDD 2018;
Herlands, PhD thesis, CMU 2019; Jakubowski et al., JMLR 2023): a log-likelihood-ratio scan
over local neighbourhoods for treatment-assignment discontinuities, a validation battery
for every candidate, and a local 2SLS effect at the boundary.

It is not a port. A mathematical audit of the source papers and their released code
([docs/math_audit_final.md](docs/math_audit_final.md)) found design-level errors, and natex
implements the corrected versions throughout (see
[Corrections vs the papers](#corrections-vs-the-papers)). Three rules hold everywhere:
discovery never reads the outcome, every stochastic step consumes one explicit
`numpy.random.Generator`, and a failed computation returns `NaN`, never `0.0`.

## What changed in v0.3.0

A review of the July 2026 paper collection found three structural problems in the
automated pipeline. All three are fixed in 0.3.0
([release notes](docs/release_notes/v0.3.0.md)).

- **The search is exhaustive.** The analyst plan (heuristics or an LLM) used to decide
  which configurations were scanned. Now it only orders them: `discover` and `survey`
  scan every binary treatment column against every single non-time forcing column, record
  what they excluded and why, and estimate every candidate outcome at each discovered
  boundary.
- **`inconclusive` is a verdict.** A refused placebo test, a configuration that could not
  be scanned, a mechanical rediscovery of a constructed time step, or a degenerate fit
  used to be reported as `null`. They are now `inconclusive`, and `null` means the design
  was tested.
- **Calendar-time kinks are gated on the placebo-calibrated p.** The same estimator runs at
  up to 49 shifted placebo dates and the declared date is ranked among them. Nominal
  robust p-values, which reject at 14 to 88% of placebo dates on real series, are reported
  for reference only. Fewer than 19 placebo positions cannot reach 5% and give
  `inconclusive`.

## Papers

Collection index (HTML and PDF): <https://haukehillebrandt.github.io/natex/>

**Flagship: Placebo-Calibrated Kink Tests of Four Claimed Trend Breaks in Epoch AI Data**
([HTML](https://haukehillebrandt.github.io/natex/paper/),
[PDF](https://haukehillebrandt.github.io/natex/main.pdf), source in [`paper/`](paper/),
numbers of record in [docs/case_studies/epoch-kinks.md](docs/case_studies/epoch-kinks.md)).
Rewritten in October 2026 under the 0.3.0 inference rule. The point estimates are the
July ones; two verdicts changed.

| Series | Cutoff | Verdict under placebo calibration |
|---|---|---|
| GPQA-Diamond logit score | o1-preview, Sept 2024 | dated bend: most extreme of 49 placebo dates, p 0.02 |
| METR 50% time horizon | o1-preview | era bend, not dated: p 0.28 |
| Epoch Capabilities Index | o1-preview | null: p 0.78 |
| China legal chip stock vs US hyperscalers | export controls, Oct 2023 | sign-robust but inconclusive: quarterly data hold only 7 placebo positions |

The sixteen companion notes and the capstone survey under [`papers/`](papers/) were
written in July 2026 on v0.2.0 with seven-placebo grids and nominal inference. They have
not been re-run. Read their "credible" verdicts against the flagship's table above: a
calendar-time result that rests on a seven-placebo grid has a calibrated floor of 0.125
and is not a 5% finding. Layout and build: [papers/README.md](papers/README.md). The
HTML is rendered by [LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/) on every push that
touches `paper/**` or `papers/**`.

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

Python 3.11 or later. Core dependencies: numpy, scipy, pandas, scikit-learn, typer,
pydantic. The import name is `natex`; the distribution is `natex-discovery` (not on PyPI
yet). Optional extras, all of which degrade gracefully when missing:
`natex-discovery[plot]` (figures), `natex-discovery[report]` (jinja2 paper templates),
`natex-discovery[paperbanana]` (methodology diagrams, own provider key),
`natex-discovery[ml]` (econml causal forest for DEE), `natex-discovery[gp]`
(GPyTorch backend), `natex-discovery[llm]` (Anthropic and Gemini guidance backends).

## Quickstart

### One-command survey

`natex survey` runs one dataset against all seven method families (rdd, did, kink, iv,
sc, bunching, dee) and writes one report with a verdict per family, including reasoned
skips:

```bash
uv run natex survey mydata.csv --seed 0
uv run natex survey mydata.csv --seed 0 --context "state-year panel of cigarette sales" \
  --cutoff score=215 --instrument rainfall --threshold income=85000 --out out/survey
open out/survey/report.html    # report.md is always written alongside
```

The report opens with the banner "AI-generated — verify before citing", then a verdict
table (credible / null / inconclusive / skipped / needs_input / failed, one row per family
with a one-sentence reason), then a section per family with key numbers, the per-outcome
effects table for rdd, figures and caveats. `null` means the design was tested and nothing
credible surfaced. `inconclusive` means it could not be validly tested and is never a
negative finding. Families whose inputs you did not declare (kink cutoffs, instruments,
bunching thresholds) are reported as `needs_input`. The rdd and did families scan the full
role space and record what they excluded. Details:
[docs/method_cards/survey.md](docs/method_cards/survey.md) and the
[natex-survey skill](skills/natex-survey/SKILL.md).

### CLI

Point `natex discover` at a CSV with a treatment column and, optionally, an outcome:

```bash
uv run natex discover data.csv \
  --treatment T --outcome y \
  --forcing score,age \
  --k 50 --q 99 --seed 0 --out out/
```

- `--forcing`: candidate running variables, comma-separated; default all numeric
  non-treatment, non-outcome columns.
- `--covariates` (alias `--dims`): columns admitted to the scan space; default all
  non-reserved columns, so exclude stray labels or NaN-heavy metrics here.
- `--k`: neighbourhood size; `--q`: null replicas for the randomization test; `--seed`.
- `--degree`: polynomial degree of the background treatment model (default 1).
- `--coarse` with `--n-coarse` (default 2000): coarse-to-fine scan for large data. Coverage
  is reported in `results.json` (`frac_centers_scanned`), never silently cut.

The command writes `out/results.json`: the top-20 discoveries (centres, LLR, hyperplane
normal, per-variable forcing influence), the scan p-value, placebo and density p-values,
2SLS and Wald effects with first-stage diagnostics, and the seed.

If you do not know the treatment column, run the analyst pass first
([method card](docs/method_cards/llm_analyst.md)):

```bash
uv run natex study data.csv --context "county-level school funding, 2004-2012" \
  --backend null --seed 0 --out out/
uv run natex discover --plan out/intake_report.json --seed 0 --out out/
```

`natex study` profiles the data, infers column roles, applies a declarative prep plan
(editable at `out/prep_plan.json`, passed back with `--prep-plan`) and ranks candidate
designs. `natex discover --plan` scans the ranked candidates first and then the full role
space from the profile, within budget. The report records every configuration as
`scanned`, `skipped_budget`, `failed` or `invalid`, the enumerated roles in `role_space`,
and the excluded columns with reasons. Plan mode also writes the results bundle that
`natex paper` and `natex brief` read.

`--backend null` is deterministic and offline. `--backend agent` writes each question as
a JSON file under `out/guidance/requests/` and waits for the answer file, so a coding agent
can serve as the analyst at no API cost. `--backend anthropic|gemini` use the APIs. Guidance
orders the search and annotates results; the statistics are identical with and without it.

`natex datasets [--root PATH]` (default `NATEX_DATA`) lists the registered benchmark
datasets with found/missing status and fetch instructions.

### A 30-second seeded demo

```bash
mkdir -p /tmp/natex-demo
uv run python - <<'EOF'
import numpy as np
from natex.data.synthetic import make_synthetic

ds, _ = make_synthetic(n=500, zeta=6.0, kind="binary", rng=np.random.default_rng(0))
ds.df.to_csv("/tmp/natex-demo/synth.csv", index=False)
EOF
uv run natex discover /tmp/natex-demo/synth.csv --treatment T --outcome y \
  --k 40 --q 49 --seed 0 --out /tmp/natex-demo/out
```

Output of a real run on v0.3.0 (results path shortened):

```console
model=bernoulli  max LLR=21.16  scan p=0.020
top center (raw z): [0.757951   0.51275872]
placebo passed: — (no non-forcing covariate was testable; battery vacuous)   density p: 0.702
2SLS tau=2.047 CI=(1.024,3.070) weak_iv=False
results: …/natex-demo/out/results.json
```

The planted effect is 2. The scan finds the discontinuity, the density test passes, and
the 2SLS interval covers the truth. The covariate placebo battery is vacuous here because
the synthetic file has no covariate besides the two forcing variables.

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

top = res.discoveries[0]
side1 = top.members[top.group1]                     # global row indices, distance >= 0 side
side0 = top.members[~top.group1]                    # group1 is a mask over members
est = local_2sls(ds, top)                           # frozen-side 2SLS, HC1 errors
print(est.tau, est.ci, est.first_stage_t, est.weak_instrument)
```

Also available: `natex.validate.placebo.placebo_tests`,
`natex.validate.density.density_test`, `natex.validate.honest.honest_split`,
`natex.estimate.local2sls.wald_estimate`, and `natex.discover` with `profile=` for the
exhaustive role-space scan and per-outcome effects.

### Known-cutoff kink designs

For a policy with a known cutoff and a slope kink, use the RKD and DiK evaluator
([docs/method_cards/kink.md](docs/method_cards/kink.md)). A sharp design supplies the
policy-slope contrast; a fuzzy design supplies the observed policy variable. A bandwidth is
always required:

```bash
uv run natex kink data.csv --design rkd --outcome y --running score \
  --policy-kink -0.4 --cutoff 0 --bandwidth 1500 --out out/

uv run natex kink panel.csv --design dik --outcome y --running score \
  --treatment policy --time year --t0 2011 --bandwidth 1500 \
  --cluster person_id --out out/
```

For sharp DiK replace `--treatment policy` with `--policy-kink-change VALUE`; for a
treated-vs-control contrast at one date use `--group COL`. Kinks are right minus left;
DiK is post minus pre or group 1 minus group 0. `out/kink.json` carries cell slopes, HC1,
CR1 or HAC inference, first-stage strength and a Fieller set for fuzzy ratios.

```python
from natex import difference_in_kinks, regression_kink
from natex.kink import placebo_kinks, placebo_calibrated_p

rkd = regression_kink(y, running, policy_kink=-0.4, bandwidth=1500)
dik = difference_in_kinks(y, running, post=year >= 2011, treatment=policy,
                          bandwidth=1500, clusters=person_id)
grid = placebo_kinks(y, running, shifted_cutoffs, bandwidth=1500)
print(rkd.tau, dik.tau, placebo_calibrated_p(rkd, grid), grid.min_attainable_p)
```

When the running variable is calendar time, the placebo-calibrated p is the inference of
record and needs at least 19 placebo positions; the survey applies this rule automatically.
A worked application is the [Epoch kinks case study](docs/case_studies/epoch-kinks.md).

### DEE, IV and synthetic control

Debias an observational CATE estimator with the discovered discontinuities
([docs/method_cards/dee.md](docs/method_cards/dee.md)):

```bash
uv run natex debias data.csv --treatment T --outcome y --m-prime 25 --out out/
```

Belloni-style instrument search with an honest discovery/estimation split and
Anderson-Rubin inference ([docs/method_cards/iv_sc.md](docs/method_cards/iv_sc.md)):

```bash
uv run natex instruments data.csv --treatment T --outcome y \
  --pool z1,z2,z3,z4,z5 --controls x1,x2 --seed 0 --out out/
```

Synthetic-control donor selection with in-space placebo inference:

```bash
uv run natex donors smoking.csv --outcome cigsale --unit state --time year \
  --treated-unit California --t0 1989 --out out/
```

Python equivalents: `natex.dee_debias`, `natex.iv.pipeline.discover_instruments`,
`natex.iv.donors.select_donors` and `sc_placebo_test`.

## From discovery to paper

The reporting layer turns a finished run into a results bundle, figures and an AI-drafted
paper skeleton:

```bash
natex study data.csv --context "where the data came from" --out out
natex discover data.csv --plan out/intake_report.json --out out
natex paper --bundle out --format md     # or --format latex (compiles when tectonic is installed)
natex brief --bundle out                 # deep-research handoff (out/research-brief.md)
```

`natex paper --bundle` accepts a saved bundle directory, a `natex discover --out`
directory, or a single-scan `results.json`. `--format md` always works; `--format latex`
writes `paper/paper.tex` and compiles it when [tectonic](https://tectonic-typesetting.github.io)
is on `PATH`, leaving the `.tex` in place otherwise. Rendering needs the `report` extra,
figures the `plot` extra, and the optional method diagram the `paperbanana` extra:

```bash
uv add 'natex-discovery[report]'
uv add 'natex-discovery[plot]'
uv add 'natex-discovery[paperbanana]'
```

Python API:

```python
from natex.report import ResultsBundle, render_paper, research_brief
from natex.report.figures import rdd_figures  # or did_figures

bundle = ResultsBundle.from_discover(report, "out/", dataset=ds, intake=intake, seed=0)
bundle.save()                                # out/results.json + figures/ + paper/
figs = rdd_figures(bundle, ds, res)          # PNG and PDF per figure
draft = render_paper(bundle, format="md")    # out/paper/paper.md
brief = research_brief(bundle, "out/")       # out/research-brief.md
```

`results.json` records the natex version, seed, parameters, full search coverage, the
guidance-log path and the figure manifest; every number in the draft comes from it.
Missing numbers render as "—", never as `nan`.

`natex brief` writes `research-brief.md`, a self-contained brief (data context, designs,
effects, validation status, numbered literature questions) to paste into a deep-research
agent. natex performs no research calls itself; what comes back is for you to vet.

natex does **not** integrate the Google Docs API. Render markdown and paste
`paper/paper.md` into a Google Doc, or upload it to Google Drive and open it with Google
Docs.

Every rendered draft opens with the banner "AI-generated draft — verify all claims before
circulation". Check every number against `results.json`, and apply the writing rules in the
[natex-write-paper skill](skills/natex-write-paper/SKILL.md): a calendar-time design is a
finding only if its placebo-calibrated p clears 5%, `inconclusive` verdicts are neither
findings nor nulls, caveats are stated once, and titles describe rather than sell.

## Agent skills

Three Claude Code skills ship in [skills/](skills/):

- [discover-natural-experiments](skills/discover-natural-experiments/SKILL.md): find and
  validate natural experiments in a CSV, serving natex's file-based guidance protocol as the
  analyst.
- [natex-write-paper](skills/natex-write-paper/SKILL.md): render the draft, verify every
  number, and keep to the style contract.
- [natex-lit-review](skills/natex-lit-review/SKILL.md): generate the research brief, hand it
  to deep-research tooling, vet the citations.

Install into Claude Code by symlinking (or copying) the skill directories into
`~/.claude/skills`:

```bash
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/"*/ ~/.claude/skills/
```

[AGENTS.md](AGENTS.md) documents the same surface for other agents.

## Backtests on real data

Unit and synthetic tests run by default. Real-data backtests are marked `backtest`,
deselected by default, and need `NATEX_DATA` pointing at a local data directory:

```bash
export NATEX_DATA="/path/to/RDD/data"
uv run natex datasets
uv run pytest tests/backtests -m backtest -q
```

With all six archives in place the check prints one line per registered dataset (paths
shortened):

```console
$ uv run natex datasets
test_score_2012  found  rows=2767  ok=True  path=…/data/test_score_2012/RDD_Guide_Dataset_0.csv
academic_probation  found  rows=44362  ok=True  path=…/data/AcademicProbation_LSO_2010/data_orig.csv
ed_visits  found  rows=161  ok=True  path=…/data/ED_visits/P03_ED_Analysis_File.csv
inpatient_visits  found  rows=73  ok=True  path=…/data/Inpatient_visits/P10_Inpatient_CSV_File.csv
egger_koethenbuerger  found  rows=43175  ok=True  path=…/data/EggerKoethenbuerger_AEJ_Data (1).csv
prop99  found  rows=1209  ok=True  path=…/data/prop99/smoking_data.csv
```

For anything missing it prints the expected layout and fetch instructions; the registry in
`natex.data.registry` is the source of truth. The MDRC file is a public download; the
others are login-gated openICPSR archives. What each backtest asserts is tabulated under
[Project status](#project-status); outcomes and timings are in
[docs/status/phase-2.md](docs/status/phase-2.md).

## Benchmarks

Synthetic benchmarks reproduce the KDD-2018 evaluation protocol (power against
discontinuity strength across polynomial orders, estimator convergence, Bernoulli against
Normal on binary treatment, label noise):

```bash
uv run python benchmarks/run_nig_curve.py --kind real
uv run python benchmarks/run_nig_curve.py --kind both --label-noise
```

Outputs land in `benchmarks/out/` (gitignored); small seeded slices run in CI. Protocol
and expected shapes: [benchmarks/README.md](benchmarks/README.md).

## Corrections vs the papers

natex deviates from the published LoRD3 papers and their released code wherever the audit
found errors. [docs/math_audit_final.md](docs/math_audit_final.md) governs every conflict.

- **Randomization test stated as what it is**: a fitted-null parametric bootstrap with
  add-one Monte Carlo p-values, plus an honest discovery/estimation split.
- **Bernoulli null replicas** are drawn as `Bernoulli(p̂)`; the legacy thresholded-Gaussian
  draw has the wrong success probability.
- **Placebo tests** are local intercept-continuity contrasts with Holm correction; the
  papers' side-mean contrasts reject valid designs mechanically.
- **Effect estimation** is frozen side-indicator 2SLS with HC1 errors; the papers' group
  instrument is inconsistent as printed. First-stage diagnostics are always on.
- **Variance estimation** uses each point's own neighbourhood residuals with a data-scaled
  floor; the legacy code indexed reverse neighbours with an absolute floor.
- **Sharp splits** are scored by boundary likelihood suprema instead of dropped.
- **Hyperplane ties**: signed distance at or above zero is group 1, the centre included.
- **Density falsification** runs on the signed distance along the frozen discovered normal
  and is documented as a falsification test only.

Legacy scan outputs are not ground truth in parity tests.

## Project status

Latest release: v0.3.0 (October 2026). Run of record: `uv run pytest -q` collects 1269
offline tests across Python 3.11 to 3.14 in CI, `uv run pytest -m backtest` collects 32
real-data backtests, and `uv run ruff check src tests` is clean.

| Phase | Scope |
|---|---|
| 1 | **Done**: corrected LoRD3 scan, validation battery, frozen-side 2SLS, profiler, CLI, first backtest |
| 2 | **Done**: remaining RDD backtests and synthetic benchmarks ([status](docs/status/phase-2.md)) |
| 3 | **Done**: SuDDDS difference-in-differences discovery and the Prop 99 backtest ([status](docs/status/phase-3.md)) |
| 4 | **Done**: DEE debiasing layer ([status](docs/status/phase-4.md)) |
| 5 | **Done**: instrument search, honest 2SLS, synthetic-control donors ([status](docs/status/phase-5.md)) |
| 6 | **Done**: analyst pass and guidance backends ([status](docs/status/phase-llm-analyst.md)) |
| 7 | **Done**: results bundle, figures, paper and brief rendering ([status](docs/status/phase-report-paper.md)) |
| 8 | **Done**: agent skills, AGENTS.md, v0.1.0 ([status](docs/status/phase-skills-docs.md)) |
| Kinks | **Done**: sharp and fuzzy RKD and DiK, HC1, CR1, HAC and Fieller inference, `natex kink` ([status](docs/status/phase-kinks.md)) |
| Survey | **Done**: `natex survey` over seven families with one report ([method card](docs/method_cards/survey.md)) |
| v0.3.0 | **Done**: exhaustive role-space scan, `inconclusive` verdicts, placebo-calibrated calendar-time kinks ([status](docs/status/phase-exhaustive-scan.md)) |

What each real-data backtest recovers. natex is never told the cutoff, the treated unit or
the timing; details are in the linked status files.

| Dataset | Design | Result |
|---|---|---|
| `test_score_2012` (MDRC RDD practice set, 2,767 rows) | sharp RDD | pretest-215 cutoff recovered; the known effect of about 10 lies inside the 2SLS interval |
| `academic_probation` (Lindo, Sanders and Oreopoulos 2010; 44,362 rows) | fuzzy RDD | `dist_from_cut = 0` ranked first of four forcing candidates through the coarse-to-fine scan |
| `ed_visits` (Anderson, Dobkin and Gross 2012; 161 cells) | fuzzy RDD | insurance-loss cutoffs at ages 19 and 23 among the top clusters |
| `inpatient_visits` (ADG 2012 companion; 73 cells) | fuzzy RDD | age-23 cutoff recovered on 73 aggregated cells |
| `egger_koethenbuerger` (Egger and Koethenbuerger 2010; 43,175 rows) | multi-cutoff RDD | at least two statutory council-size thresholds on `log_pop` (four of five observed) |
| `prop99` (Abadie, Diamond and Hainmueller; 1,209 rows) | DiD and synthetic control | SuDDDS recovers (California, 1989); donors put weight 0.955 on the ADH five, ATT about -19.5 |

## Development

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest -q            # excludes backtests by default
```

## License

MIT; see [LICENSE](LICENSE).
