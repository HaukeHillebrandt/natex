---
name: discover-natural-experiments
description: Run natex to find and validate natural experiments (RDDs, DiDs, IVs, synthetic-control donors) in a tabular dataset, serving natex's file-based guidance protocol yourself as the LLM backend. Use when the user says things like "find natural experiments in this dataset", "discover RDDs", "is there a discontinuity in this data", "run natex on this CSV", or "what quasi-experiments are hiding in here". Covers install, the natex study analyst pass, answering guidance request files, natex discover --plan, and honest interpretation of results.json.
---

# Discover natural experiments with natex

## 1. What natex is

natex is a Python toolkit that scans a tabular dataset for natural experiments —
regression discontinuities (RDD), difference-in-differences (DiD) events, instruments,
and synthetic-control donor pools — and puts every candidate through a validation
battery before it may be called a finding. Three hard guarantees hold everywhere:

- **Discovery never reads the outcome variable** — search runs on treatment and
  covariates only, so discovered designs are not fit to the outcome.
- **Guidance is advisory only.** Whatever you (or any LLM backend) answer can propose
  and veto but never fabricate statistics; the statistical output is bitwise identical
  with and without guidance, and a veto is recorded as a flag only.
- **`null`/`NaN` means failed or underpowered, never zero.** natex never replaces a
  failed computation with a number, and neither may you.

## 2. Install

Requires Python >= 3.11. Into the user's existing uv project:

```bash
uv add git+https://github.com/HaukeHillebrandt/natex
```

Or work from a clone:

```bash
git clone https://github.com/HaukeHillebrandt/natex
cd natex
uv sync
```

Verify the install prints the command table:

```bash
uv run natex --help
```

## 3. Run the analyst pass

```bash
uv run natex study data.csv --context "<where the data came from, one sentence>" --backend agent --seed 0 --out out/
```

This profiles the CSV (column types, candidate treatments, forcing variables, panel
structures) and then asks guidance questions **by writing files** — with
`--backend agent` the command BLOCKS on each question (default 600 s per question)
until an answer file appears. Run it in the background or a second terminal, then
serve the protocol below while it waits. Pass `--context` whenever the user told you
anything about the data's origin; it is forwarded verbatim into every question.

## 4. Serve the AgentBackend protocol (you ARE the LLM backend)

While `natex study` runs, watch `out/guidance/requests/` for new `*.json` files
(poll the directory; files are named `{seq:04d}_{task}.json`, e.g.
`0000_understand.json`). Answer each one by writing a JSON file with the **same
filename** into `out/guidance/responses/`.

Every request file contains exactly five keys:

- `task` — one of `understand`, `prepare`, `search_plan`, `interpret_discovery`,
  `audit_assumptions`, `review_control_group`.
- `payload` — the data profile or discovery summary you must reason over.
- `schema_hint` — the JSON schema your answer must satisfy.
- `instructions` — what to do for this task.
- `respond_to` — the exact absolute path to write your answer to.

**Compose the answer JSON yourself** — you are the intelligence here. Read the
payload, think, and produce a single JSON object matching `schema_hint`. Never invent
columns that are not in the payload; never fabricate statistics or numbers; you may
flag concerns or veto (e.g. in `audit_assumptions`), and the veto is recorded as an
advisory flag only — no statistic is dropped or changed because of it.

Write the full, valid JSON object in **one write** to the `respond_to` path (which
equals `out/guidance/responses/<same filename>`). A partial file is simply re-polled;
a complete non-object (e.g. a bare list) is an error. Either the bare content object
or a wrapper `{"content": {...}}` is accepted.

Worked example — request file `out/guidance/requests/0000_understand.json`
(`schema_hint` abbreviated here; the real file carries the full JSON schema):

```json
{
 "task": "understand",
 "payload": {
  "profile": {
   "n_rows": 4632,
   "columns": [
    {"name": "test_score", "dtype": "float64", "n_unique": 3187, "missing_frac": 0.0,
     "is_numeric": true, "is_binary": false, "is_time_like": false},
    {"name": "scholarship", "dtype": "int64", "n_unique": 2, "missing_frac": 0.0,
     "is_numeric": true, "is_binary": true, "is_time_like": false}
   ],
   "panel_candidates": [],
   "forcing_candidates": ["test_score"],
   "treatment_candidates": ["scholarship"]
  },
  "context": "2012 college entrance exam administrative extract"
 },
 "schema_hint": {
  "title": "Understanding",
  "type": "object",
  "properties": {
   "shape": {"enum": ["cross-section", "time-series", "panel", "aggregated-cells"]},
   "unit_of_observation": {"type": "string"},
   "treatments": {"type": "array"},
   "outcomes": {"type": "array"},
   "forcing": {"type": "array"},
   "did_structures": {"type": "array"},
   "quirks": {"type": "array"},
   "notes": {"type": "string"}
  },
  "required": ["shape"]
 },
 "instructions": "You are given a column-level profile of a tabular dataset plus any user-supplied context. Describe what each column most likely measures and which columns could serve as treatment, outcome, forcing (running) variable, time, or unit identifiers. Answer as JSON matching the provided schema; do not invent columns.",
 "respond_to": "/abs/path/to/out/guidance/responses/0000_understand.json"
}
```

Your answer, written to `out/guidance/responses/0000_understand.json`:

```json
{
 "shape": "cross-section",
 "unit_of_observation": "student",
 "treatments": [
  {"column": "scholarship",
   "reason": "binary award indicator; plausibly assigned at a test_score cutoff"}
 ],
 "outcomes": [],
 "forcing": [
  {"column": "test_score",
   "reason": "continuous entrance-exam score; classic running variable"}
 ],
 "did_structures": [],
 "quirks": [],
 "notes": "Two-column extract with no outcome column; downstream effect estimates will need an outcome merged in."
}
```

Fallback: if you miss the timeout, the run aborts with a TimeoutError that names both
the request and response paths — write the response and re-run. Passing
`--backend null` instead re-runs fully offline with deterministic heuristics
(no files to serve).

When every question is answered, `natex study` finishes and writes
`out/intake_report.json` (profile, understanding, prep plan, ranked search plan) and
`out/guidance_log.jsonl` (every request and response, for reproducibility).

## 5. Run discovery

```bash
uv run natex discover --plan out/intake_report.json --seed 0 --out out/
```

Add `--backend agent` here too if you want to answer the discovery-time guidance
questions (`interpret_discovery`, `audit_assumptions`, `review_control_group`) — the
same file-serving protocol applies. Ranked plan candidates are scanned first, then
the exhaustive remainder within budget; coverage counts (scanned / skipped_budget /
failed / invalid) are always recorded, never silently dropped.

## 6. Interpret the results for the user

Plan mode writes `out/discover_report.json`; plain `natex discover` without `--plan`
writes `out/results.json`. Report to the user:

- a discoveries table: design, location/center values, forcing-variable influence,
  and the local likelihood ratio (LLR);
- the scan p-value;
- the validation battery: randomization test, placebo battery (intercept-continuity
  tests with Holm correction, not side-means), density test;
- effects, both `2sls` and `wald`: `tau`, `se`, `ci`, `first_stage_t`,
  `weak_instrument`.

Honest-inference caveats — state these whenever you summarize results:

- p-values come from a **fitted-null Monte Carlo** (parametric bootstrap) with the
  **+1-rank** correction — they are **not exact** and must not be described as exact.
- Prefer the **honest split** — separate discovery and estimation halves — before
  making any headline effect claim; estimates from the discovery half are
  post-selection and optimistic.
- Always surface `weak_instrument` and `first_stage_t`. Report `ci` exactly as given:
  AR/Fieller confidence sets may be disjoint, unbounded, or empty, and natex never
  coerces them into a tidy interval.
- The density (manipulation) test is a **falsification check only**, computed on the
  signed distance along the *frozen* discovered normal — passing it does not prove
  the design.
- `null`/`NaN`/"—" in any field means the computation **failed or was
  underpowered** — report it as such and NEVER substitute a number.
- Guidance (including everything you wrote in step 4) is advisory only: the
  statistics are identical with and without it.

## 7. Warnings

- **Never fabricate** statistics, p-values, or columns — every number you report must
  come from natex's output files or from a payload you were given.
- Every discovery needs the full **validation battery** (randomization, placebo,
  density) before you call it a finding; a high LLR alone is not a finding.
- Anything drafted downstream of these results is **AI-generated** and must be
  verified by a human before circulation.
