---
name: natex-survey
description: Run the natex one-command survey to systematically test a tabular dataset against all seven quasi-experimental method families (rdd, did, kink, iv, sc, bunching, dee) and produce one visual report with a reasoned applicability verdict per family. Use when the user says things like "survey this dataset for natural experiments", "run natex against this dataset", "which quasi-experimental designs apply to this data", or "one-command natex report". Covers the natex survey CLI, serving the method_applicability guidance task as the analyst backend, declared-input flags for offline runs, and honest presentation of report.html.
---

# Survey a dataset with natex — one command, one report

## 1. The one-command flow

```bash
uv run natex survey data.csv --context "<where the data came from, one sentence>" --seed 0 --out out/survey
```

This runs the dataset systematically against ALL seven natex method families, in a
fixed order, and writes one visual report with an applicability verdict per family —
including reasoned SKIPs. The seven families:

- **rdd** — regression discontinuity: a rule assigns treatment at a cutoff in a
  numeric running variable; the LoRD3 scan searches for the discontinuity.
- **did** — difference-in-differences: treated units' outcomes break from untreated
  units' after adoption in a panel; the SuDDDS scan searches subsets and windows.
- **kink** — regression kink: a policy's *slope* (not level) changes at a cutoff you
  declare; natex evaluates declared cutoffs, it never searches for them.
- **iv** — instrumental variables: candidate instruments you declare are screened
  with an honest Lasso split (discovery half selects, estimation half estimates).
- **sc** — synthetic control: a weighted donor pool tracks one treated unit's
  pre-period outcome; inference by in-space placebos over the donors.
- **bunching** — excess mass piling up at a policy threshold you declare, tested
  with a binned-Poisson density break; thresholds are tested, never searched.
- **dee** — discovered-experiment ensemble debiasing of an observational estimate;
  needs the gp extra installed and only runs after a credible rdd result.

What lands under `--out out/survey`:

- `survey.json` — machine-readable verdicts: per-family status (`credible`, `null`,
  `skipped`, `needs_input`, `failed`), reason, key numbers, coverage counts.
- `report.html` (with the `[report]` extra) and `report.md` (always) — the report.
- `families/<name>.json` — full per-family detail.
- `figures/` — per-family PNGs (with the `[plot]` extra); a missing figure is
  recorded as `no figure: <reason>`, never silently absent.
- `intake/` — the stage-0 analyst artifacts (profile, understanding, prep plan,
  search plan) plus `intake/guidance_log.jsonl`, the reproducibility log of every
  guidance request and response.

A failed family is a recorded outcome, not a crash: the command exits 0 whenever
`survey.json` was written.

## 2. Serve the guidance protocol as the analyst (`--backend agent`)

Add `--backend agent` and the run BLOCKS on guidance questions written as files.
Watch `out/survey/guidance/requests/` (poll it; files are named
`{seq:04d}_{task}.json`). A survey asks the three study tasks first —
`0000_understand.json`, `0001_prepare.json`, `0002_search_plan.json`, served exactly
as in the discover-natural-experiments skill — then ONE
`0003_method_applicability.json`. Answer each by writing a single JSON object
matching the request's `schema_hint` to `out/survey/guidance/responses/<same
filename>` (the request's `respond_to` path) in one write.

The `method_applicability` payload carries the column profile, your `--context`, the
declared inputs, per-family requirement checklists with met/unmet flags, and the
heuristic verdicts. You decide per family: `run` true/false, a reason, and optional
`config_hints`. Rules:

- You may override the heuristics BOTH ways; every override is recorded in
  `survey.json` (`heuristic_said` vs `analyst_said` plus your reason). A family you
  turn off shows as `skipped` with your reason — never silently absent.
- Hints feed **config only — never statistics**: cutoffs, instruments, thresholds,
  treated unit and t0 become run configuration; no statistic is computed, changed,
  or dropped because of your answer. Guidance is advisory.
- Never invent columns not in the payload and never fabricate numbers. Hints naming
  unknown or non-numeric columns are dropped and the drop is recorded.
- Explicit user declarations (the CLI flags in section 4) always win over hints.
- Families your reply omits keep their heuristic decision.

Worked example: the profile shows a firm-year panel and the context says wage
subsidies taper above a statutory 30-employee threshold. The heuristics left kink
and bunching at `needs_input` (no declared cutoff or threshold) — the statutory
threshold is exactly the input they need, so hint it and turn them on, and confirm
did on the panel. Response written to
`out/survey/guidance/responses/0003_method_applicability.json` (three decisions
shown; omitted families keep their heuristic verdicts):

```json
{
 "families": [
  {"family": "kink", "run": true,
   "reason": "context names a statutory 30-employee threshold where the subsidy schedule changes slope",
   "config_hints": {"cutoffs": [{"column": "n_employees", "value": 30}],
                    "instruments": [], "thresholds": [],
                    "treated_unit": null, "t0": null}},
  {"family": "bunching", "run": true,
   "reason": "firms have an incentive to stay just below the statutory threshold",
   "config_hints": {"cutoffs": [], "instruments": [],
                    "thresholds": [{"column": "n_employees", "value": 30}],
                    "treated_unit": null, "t0": null}},
  {"family": "did", "run": true,
   "reason": "firm-year panel with staggered subsidy adoption fits the SuDDDS scan"}
 ]
}
```

Backend or parse failures fall back to the heuristic verdicts per family with the
error recorded — never a crash. The default `--backend null` runs fully offline
with deterministic heuristics: no files to serve.

## 3. Present report.html to the user — with the caveats

- Keep the banner: **AI-generated — verify before citing**. Never drop it when
  summarizing, quoting, or converting the report.
- Always quote the per-family caveat line shown in each family's section whenever
  you summarize that family (e.g. rdd/did scan p-values are fitted-null Monte Carlo
  p-values, not exact; sc placebo p-values have granularity 1/(n_used+1); iv
  exclusion is untestable from data).
- `credible` means the design survived that family's validation battery (placebo,
  density, composition checks) — a strong scan score alone is never a finding.
- Skipped families are reasoned decisions listed in the verdict table (heuristic or
  analyst reason, with any override, recorded) — present the reason, keep the row.
- NEVER call a `null` or `needs_input` family a negative finding: `null` means no
  credible design surfaced under this configuration (often underpowered), and
  `needs_input` means a declared input is missing. A `null`/"—" in any field means
  the computation failed or was underpowered — report it as such and never
  substitute a number.

## 4. Declared-input flags for non-LLM runs

With the default `--backend null` there is no analyst to propose config, so declare
inputs yourself to unlock the declared-input families:

```bash
uv run natex survey data.csv --seed 0 --out out/survey \
  --time year --unit firm_id \
  --cutoff n_employees=30 \
  --threshold n_employees=30 \
  --instrument distance_to_office
```

- `--time` / `--unit` name the panel columns (did and sc).
- `--cutoff COL=VALUE` (repeatable) declares a kink cutoff.
- `--instrument COL` (repeatable) declares candidate instruments for the iv screen.
- `--threshold COL=VALUE` (repeatable) declares a bunching threshold.

Budget knobs mirror discover: `--k`, `--q`, `--coarse`, `--n-coarse`,
`--max-configs` — anything cut by the budget is listed as `skipped_budget`,
never dropped.
