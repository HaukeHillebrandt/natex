# AGENTS.md — using natex from any coding agent

natex is a Python toolkit for automated discovery of natural experiments in
tabular data: a LoRD3 scan finds regression-discontinuity candidates, SuDDDS finds
difference-in-differences events, DEE debiases observational predictions with
the discovered quasi-experiments, and a Belloni-style search selects
instruments and synthetic-control donors. Every candidate passes a validation
battery (randomization, placebo, density tests) before it may be called a
finding. The implementation follows the corrections in
`docs/math_audit_final.md`, and three guarantees hold everywhere: discovery
never reads the outcome variable; every stochastic step is driven by one
seeded `numpy.random.Generator` (identical seed, identical output); and a
failed or underpowered computation is reported as `null`/`NaN`, never as 0.

This file is the contract for **any** coding agent (Claude Code, Codex,
Cursor, aider, …) that drives natex. Claude Code users can also install the
step-by-step skills in `skills/` (see "Where things live").

## Install

Requires Python >= 3.11. Into an existing uv project:

```bash
uv add git+https://github.com/HaukeHillebrandt/natex
```

Or work from a clone (dev extra brings pytest + ruff):

```bash
git clone https://github.com/HaukeHillebrandt/natex
cd natex
uv sync --extra dev
```

Core dependencies are exactly numpy/scipy/pandas/scikit-learn/typer/pydantic;
everything else is an optional extra: `plot` (matplotlib figures), `report`
(jinja2 LaTeX paper), `paperbanana` (diagram generation), `ml` (econml
T-learner), `gp` (torch/gpytorch/botorch surfaces for DEE), `llm` (anthropic +
google-genai API backends). All features degrade gracefully when an extra is
missing.

## CLI surface

Run any command as `uv run natex <command> --help` for the full option list.

| Command | Purpose | Key inputs | Output files |
| --- | --- | --- | --- |
| `datasets` | Registry status: one line per benchmark dataset — found/missing, rows, ok, fetch source. Informational only, always exits 0. | `--root` (default: env `NATEX_DATA`) | stdout only |
| `fetch-data` | Download a dataset with a public direct URL into the data root (atomic rename + verify). Login-gated datasets print instructions and exit 1. | `NAME`, `--root`, `--force` | dataset file under the data root |
| `study` | Stage-0 analyst pass: profile -> understand -> prep plan -> search plan. Fully offline with the default `--backend null`. | `CSV`, `--context`, `--backend null\|agent\|anthropic\|gemini`, `--workdir` (default `OUT/guidance`), `--seed`, `--out` | `out/intake_report.json`, `out/prep_plan.json`, `out/guidance_log.jsonl` |
| `discover` | LoRD3 RDD scan / SuDDDS DiD discovery with the validation battery and honest effect estimates; `--plan` consumes an intake report. | `CSV` + `--treatment` (or `--plan intake_report.json`), `--design`, `--k`, `--q`, `--seed`, `--out` | `out/results.json` (plan mode also: `out/discover_report.json`) |
| `debias` | DEE debiasing: scan -> VKNN repair -> local 2SLS -> GP debiasing of the observational T-learner. | `CSV`, `--treatment`, `--outcome`, `--forcing`, `--m-prime` | `out/dee_result.json` |
| `instruments` | Belloni-style instrument selection plus honest 2SLS/J/AR estimation block. | `CSV`, `--treatment`, `--pool`, `--controls`, `--outcome` | `out/instruments.json` |
| `donors` | Synthetic-control donor selection, ATT and in-space placebo inference. | `CSV`, `--outcome`, `--unit`, `--time`, `--treated-unit`, `--t0` | `out/donors.json` |
| `paper` | Render the AI-draft paper from a results bundle (markdown always works; latex compiles when tectonic is on PATH). | `--bundle`, `--format md\|latex`, `--out` | `BUNDLE/paper/paper.md` or `.tex`/`.pdf` |
| `brief` | Write the deep-research handoff brief from a results bundle. | `--bundle`, `--out` | `research-brief.md` |

## The AgentBackend file protocol (normative)

With `--backend agent`, natex asks its guidance questions **by writing files**
and blocking until you answer. You — the coding agent — are the LLM backend.

- **Request path.** natex writes `WORKDIR/requests/{seq:04d}_{task}.json`
  (e.g. `0000_understand.json`). `WORKDIR` is `--workdir`, defaulting to
  `OUT/guidance` — so with `--out out/` you watch `out/guidance/requests/`.
- **Request shape.** Every request file is a JSON object with exactly five
  keys: `task` — one of `understand`, `prepare`, `search_plan`,
  `interpret_discovery`, `audit_assumptions`, `review_control_group`;
  `payload` — the profile or discovery summary to reason over; `schema_hint`
  — the JSON schema the answer must satisfy; `instructions` — what to do for
  this task; `respond_to` — the absolute path to write the answer to.
- **Response path.** Write a single JSON object to `respond_to`, which always
  equals `WORKDIR/responses/<same filename>` (i.e. `out/guidance/responses/…`).
  Either the bare content object or a wrapper `{"content": {...}}` is
  accepted; a complete non-object (e.g. a bare list) is an error. Write the
  full object in one write — a partial file is simply re-polled.
- **Polling and timeout.** natex polls every 0.5 s and raises a `TimeoutError`
  naming both paths after 600 s (default) — write the response and re-run, or
  fall back to `--backend null` (deterministic offline heuristics).
- **Restart-safe sequencing.** The sequence number continues after any request
  files already on disk, so a re-run never overwrites earlier questions.
- **Logging.** Every request and response is appended to
  `OUT/guidance_log.jsonl` for reproducibility.
- **Advisory-only contract.** Your answers may propose and veto but never
  fabricate statistics: a veto is recorded as a flag only, and the statistical
  output is bitwise identical with and without guidance. Never invent columns
  that are not in the payload; never substitute a number for a `null`.

Worked example — natex writes `out/guidance/requests/0000_understand.json`
(`schema_hint` abbreviated; the real file carries the full JSON schema):

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

You answer by writing `out/guidance/responses/0000_understand.json`:

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

A typical guided session:

```bash
uv run natex study data.csv --context "..." --backend agent --seed 0 --out out/
# (answer the request files while study blocks)
uv run natex discover --plan out/intake_report.json --seed 0 --out out/
```

## Where things live

- `docs/method_cards/lord3.md` — LoRD3 RDD scan: LLR score, validation battery, honest split.
- `docs/method_cards/suddds.md` — SuDDDS DiD discovery: GESS control-group expansion, event effects.
- `docs/method_cards/dee.md` — DEE: VKNN repair, local 2SLS, GP debiasing, BMA mixture.
- `docs/method_cards/iv_sc.md` — instrument selection (Belloni/AR) and synthetic-control donors.
- `docs/method_cards/llm_analyst.md` — the analyst pass and guidance backends (null/agent/API).
- `docs/math_audit_final.md` — the governing math audit; **it wins any conflict** with other docs or comments.
- `skills/` — Claude Code agent skills (`discover-natural-experiments`, `natex-write-paper`, `natex-lit-review`), symlink-installable into `.claude/skills/`.
- `docs/status/` — run-of-record numbers per phase (test counts, backtest results).
- `docs/plans/` — implementation history (the executed phase plans).

## Testing conventions

- `uv run pytest -q` runs the full offline suite; backtests are deselected via
  `addopts = -m 'not backtest'` in `pyproject.toml`.
- `uv run pytest tests/backtests -m backtest -q` runs the real-data backtests
  and requires the `NATEX_DATA` environment variable to point at the local
  benchmark data root (see `natex datasets` for what is present).
- Datasets are **never** committed to the repository — `natex fetch-data`
  reconstructs the data root; never commit anything under it or under `out/`.
- `uv run ruff check src tests` must be clean (line-length 100).
- CI runs Python 3.11–3.14; optional-extra tests skip gracefully when the
  extra is not installed.
