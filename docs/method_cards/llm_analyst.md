# Method card — LLM analyst pass (Stage 0) + scan guidance backends

**Source:** design spec §6 (6a Stage-0 analyst pass, 6b targeted-first-exhaustive-still,
6c in-scan guidance hooks + backends) and §4 (`natex.study` / `natex.discover` API contract);
spec §10 risk "LLM guidance could bias discovery" governs every guarantee below.
**Governing math:** [docs/math_audit_final.md](../math_audit_final.md) — this layer adds NO
new inference code; all statistics are reused verbatim from the scan/validate/estimate
modules (audit item 1 lineage: +1-rank Monte-Carlo p-values, honest framing).
**Modules:** `natex.llm.*` (backends, agent, api, log), `natex.intake.*` (profiler, prep,
plans, analyst), `natex.discover`, `natex.guidance_eval`.
**Eval scaffold:** `benchmarks/guidance_eval.py` (blind-vs-informed rank of the true design;
CI slice in `tests/test_guidance_eval.py`). **Run of record:**
[docs/status/phase-llm-analyst.md](../status/phase-llm-analyst.md).

## What it does

An optional LLM sits BESIDE the pipeline as an analyst, never inside the statistics:

1. **Stage 0 — `natex.study(csv_or_df, context, guidance, rng, out)`**: profile the raw
   dataframe (`natex.intake.profiler`), ask the backend to **understand** it (column roles,
   dataset shape, quirks), propose a declarative **PrepPlan** (drops, filters, encodings,
   discretization, optional seeded subsample — executed ONLY by natex code; the LLM never
   emits code), and rank design candidates into a **SearchPlan** (RDD forcing variables, DiD
   unit x time panels, budget hints). Returns a serializable `IntakeReport`;
   `report.prepare()` yields a `Dataset` ready for discovery.
2. **`natex.discover(data, design, guidance, search_plan, rng, budget, out)`**: scans
   ranked plan candidates first at full resolution, then the exhaustive remainder derived
   from the bound dataset spec, within budget — and ALWAYS reports what was and wasn't
   searched (spec 6b).
3. **In-scan hooks (spec 6c)**: after each scanned config's validation block the backend may
   `interpret_discovery` and `audit_assumptions`; on the DiD GESS effects path it may
   `review_control_group`. All ADVISORY — responses land in `ConfigRecord.advisory` and the
   guidance log, never in a statistic.

Everything works with no LLM at all: the default `NullBackend` degrades to deterministic
profile-only heuristics, so `natex study` / `natex discover` run offline with no API key.

## Task vocabulary

Six task literals in `natex.llm.backends.TASKS`: `understand`, `prepare`, `search_plan`
(Stage 0, in this fixed order — the MockBackend contract), `interpret_discovery`,
`audit_assumptions`, `review_control_group` (in-scan, in this order per scanned config).
The spec's mid-run `propose_forcing_variables` / `propose_candidate_events` hooks (6c rows
1–2) are satisfied by the Stage-0 SearchPlan in this phase; the two extra task literals are
NOT in `TASKS` until a phase needs mid-run refinement.

## NullBackend heuristics (the no-LLM degradation path, spec 6a)

Deterministic, derived from the request payload alone — NO rng, NO data access, NO network.
Identical payload ⇒ identical content and bitwise-identical `raw_text`
(`json.dumps(content, sort_keys=True)`). Constants are module-level in
`natex/llm/backends.py`. NullBackend NEVER vetoes.

| task | payload contract | heuristic content |
|---|---|---|
| `understand` | `{"profile": <IntakeProfile dict>, "context": str\|None}` | `shape`: `"panel"` if `panel_candidates` else `"aggregated-cells"` if `n_rows < _AGG_MAX_ROWS (5000)` and some column name matches `^(n\|count\|n_obs\|weight\|wt\|pop\|population\|cells?)$` (case-insensitive) else `"time-series"` if some time-like column has `n_unique == n_rows` else `"cross-section"`. `unit_of_observation`: first panel candidate's unit column else `"row"`. `treatments` = profile `treatment_candidates` (reason "binary 0/1 column"). `outcomes` = numeric, non-binary, non-time-like columns (reason "numeric, non-binary"). `forcing` = profile `forcing_candidates` minus time-like columns (reason "numeric with >= 20 distinct values"). `did_structures` from `panel_candidates` (reason "unit x time grid covers >= 95% of rows"). `quirks`: `"<col>: constant column"` for `n_unique <= 1`; `"<col>: {pct}% missing"` for `missing_frac > _QUIRK_MISSING (0.2)`. `notes` names the backend: `"NullBackend heuristics (no LLM)"`. |
| `prepare` | `{"profile": ..., "understanding": ..., "seed": int, "context": ...}` | PrepPlan dict: `drop_cols` = constant columns + columns with `missing_frac > _DROP_MISSING (0.5)`; `subsample = {"n": _SUB_N (20000), "seed": payload["seed"]}` iff `n_rows > _SUB_MAX (50000)`; everything else empty (profile-only degradation, spec 6a). |
| `search_plan` | `{"profile": <post-prep profile dict>, "understanding": ..., "context": ...}` | For each treatment guess `t` (profile order): one `rdd` candidate — outcome = first outcome guess ≠ t (else None), forcing = all forcing guesses ∉ {t, outcome}; skipped if forcing empty. Then for each (t × did_structure): one `did` candidate (unit/time from the structure, outcome as above). `priority` = running index. `rationale` states the rule that fired. `budget` = `{"k": 50, "q": 99, "coarse": n_rows > _COARSE_MIN (10000), "n_coarse": 2000}`. |
| `interpret_discovery` | `{"candidate": ..., "summary": <summary block WITHOUT effects>, "context": None}` | `{"summary": "<deterministic sentence naming design, dominant forcing column (max abs influence) and location>", "matched_policies": [], "confounded_risk": "unknown", "note": "NullBackend: no domain knowledge applied"}` |
| `audit_assumptions` | `{"candidate": ..., "validation": {p_value, placebo/composition fields}}` | `{"excludability": "unreviewed", "monotonicity": "unreviewed", "sutva": "unreviewed", "veto": false, "caveats": ["NullBackend: assumption audit requires a human or LLM reviewer"]}` |
| `review_control_group` | GESS expansion dict (profile, expansions, mse_trace, subset_values, n_control, n_tau) | `{"face_valid": null, "veto": false, "reason": "NullBackend performs no substantive review; n_expansions=<len(expansions)>"}` — NullBackend NEVER vetoes. |

The `seed` in the `prepare` payload is drawn from the pipeline's single
`numpy.random.Generator` by `study()` and stored IN the plan, so a serialized plan replays
bitwise without the original rng.

## Backend matrix

All backends satisfy the `GuidanceBackend` protocol (`name: str`,
`complete(GuidanceRequest) -> GuidanceResponse`) and are importable from `natex.llm`.

| backend | `name` | transport | determinism | install | CI testing |
|---|---|---|---|---|---|
| `NullBackend` | `null` | none (pure function of the payload) | bitwise | core | full unit tests (`tests/test_llm_null.py`) |
| `MockBackend` | `mock` | none (canned response list, records every request) | as canned | core (tests) | it IS the CI backend for study/discover/hook/eval tests |
| `AgentBackend` | `agent` | file-based request/response under a workdir (subscription mode, zero API cost) | as the answering agent | core | thread-answered round trips (`tests/test_llm_agent.py`); manual smoke in the status doc |
| `AnthropicBackend` | `anthropic` | Anthropic Messages API, structured `output_config` json_schema output | non-deterministic | `[llm]` extra | fake injected client (`_client=`) pins kwargs; real-SDK constructor smoke skips without the extra; NEVER a live call in CI |
| `GeminiBackend` | `gemini` | google-genai `generate_content`, `response_mime_type` + `response_json_schema` (one-shot TypeError fallback to schema-free JSON on older SDKs) | non-deterministic | `[llm]` extra | same fake-client policy |

Install line for the API backends (SDKs imported lazily; a missing SDK raises ImportError
with exactly this remedy):

```bash
pip install 'natex-discovery[llm]'      # anthropic>=0.40, google-genai>=1.0
```

Both API backends send one user message (`TASK_INSTRUCTIONS[task]` + sorted-JSON payload +
"Respond with a single JSON object.") and, when the request carries a `schema_hint`, a
strictified schema (`_strict_schema`: every object node gets `additionalProperties: false`
and `required = all properties`; numeric/length/pattern constraint keys are stripped —
which is why pydantic models destined for schema hints validate ranges in
`field_validator`s, never `gt=`/`le=` field constraints).

### AgentBackend file protocol

`AgentBackend(workdir)` creates `workdir/requests/` and `workdir/responses/`. Each
`complete()` writes `requests/{seq:04d}_{task}.json`, echoes one instruction line
(`natex guidance request (<task>): answer by writing JSON matching schema_hint to
<response_path>`), and polls `responses/{seq:04d}_{task}.json` (default every 0.5 s,
timeout 600 s). `seq` continues after existing request files, so restarts are safe and
monotone. Empty or partially-written response files are tolerated (the poll just
continues); a parsed non-dict is a `ValueError`; the timeout error names the exact response
path to write. The answering agent may reply with either the bare content object or a
`{"content": {...}, ...}` envelope.

Example request (`out/agent/requests/0000_understand.json`, abridged):

```json
{
 "task": "understand",
 "payload": {
  "profile": {
   "n_rows": 300,
   "columns": [
    {"name": "x0", "dtype": "float64", "n_unique": 300, "missing_frac": 0.0,
     "is_numeric": true, "is_binary": false, "is_time_like": false},
    {"name": "T", "dtype": "float64", "n_unique": 2, "missing_frac": 0.0,
     "is_numeric": true, "is_binary": true, "is_time_like": false}
   ],
   "panel_candidates": [],
   "forcing_candidates": ["x0", "x1", "x2", "y"],
   "treatment_candidates": ["T"]
  },
  "context": null
 },
 "schema_hint": { "...": "Understanding.model_json_schema()" },
 "instructions": "You are given a column-level profile of a tabular dataset ...",
 "respond_to": ".../out/agent/responses/0000_understand.json"
}
```

Example hand-written response (`out/agent/responses/0000_understand.json`):

```json
{
 "shape": "cross-section",
 "unit_of_observation": "row",
 "treatments": [{"column": "T", "reason": "binary 0/1 indicator"}],
 "outcomes": [{"column": "y", "reason": "continuous response variable"}],
 "forcing": [{"column": "x0", "reason": "plausible running variable"},
             {"column": "x1", "reason": "plausible running variable"}],
 "did_structures": [],
 "quirks": [],
 "notes": "answered by the calling coding agent"
}
```

## Guarantees (spec 6b/6c — how guidance cannot bias discovery)

- **Coverage is always reported (6b).** A SearchPlan ORDERS the scan, it never truncates
  it: every enumerated configuration becomes a `ConfigRecord` with `status` in
  `scanned / skipped_budget / failed / invalid`, and `DiscoverReport.searched` carries
  `n_total / n_scanned / n_skipped_budget / n_failed / n_invalid`, the effective budget
  dict, and the plan-vs-exhaustive candidate split. Budget cuts are listed as
  `skipped_budget`, never dropped; invalid plan candidates are recorded with the reason,
  never silently discarded.
- **Guidance never gates statistics (6c).** Hooks fire only when a backend was passed, are
  advisory-only, and are mutation-tested: the same data + seed with `guidance=None` and
  with an always-vetoing MockBackend produce bitwise-identical statistical output once the
  `advisory` / `advisory_veto` / `vetoed_by_guidance` keys are stripped
  (`tests/test_discover.py`).
- **A veto is only ever a flag.** `audit_assumptions` veto sets `advisory["vetoed"]` and
  `summary["advisory_veto"]`; `review_control_group` veto sets
  `effects["gess"]["vetoed_by_guidance"]`. τ̂/se/p are computed and reported regardless.
- **Discovery never reads the outcome.** Hook payloads carry no raw data arrays and no
  outcome values (asserted in tests); Stage-0 prep filters that touch a surviving
  candidate's outcome column are flagged in `guidance_errors` as a data-snooping warning
  (audit item 1 lineage) — warning only, never a hard failure.
- **Failures never fabricate numbers.** A failed config gets `status="failed"` with
  `llr`/`p_value` = `None` (NaN serializes to null, never 0.0); a failed hook becomes
  `advisory[<hook>] = {"error": ...}` with the config's statistics untouched; a failed
  Stage-0 guidance step falls back to NullBackend heuristics, recorded by name in
  `guidance_errors` (or re-raised under `study(strict=True)`).
- **One rng.** `study()` and `discover()` require an explicit `numpy.random.Generator`;
  hooks consume no randomness; identical seed ⇒ identical report JSON.

## Reproducibility: guidance log + MockBackend replay

Every request+response — for EVERY backend, including Null — is appended to an append-only
JSONL guidance log (`guidance_log.jsonl` next to the outputs whenever `out=` is given):
one line per call with `seq`, `ts`, `task`, `backend`, the full `request`
(task/payload/schema_hint) and `response` (content/raw_text/backend). API backends are
inherently non-deterministic; a run is replayed by feeding the logged response contents, in
logged order, back through `MockBackend([...])` — the Stage-0 task order
(understand → prepare → search_plan) and the per-config hook order
(interpret_discovery → audit_assumptions → review_control_group on the GESS path) are
documented contracts, so the canned list lines up.

## Evaluation scaffold

`natex.guidance_eval.run_guidance_eval(make_backend, ...)` buries a known true design among
decoys per `EvalCase` and reports the rank of the truth in the resulting SearchPlan, blind
(`guidance=None` ⇒ Null heuristics) vs informed (the provided backend). The gate question:
does the informed plan hit the true config at a lower rank? CI proves the discrimination
with MockBackend only (informed rank 0 vs blind rank ≥ 1 on the decoy-first rdd cases);
API arms are manual-only via `benchmarks/guidance_eval.py`.

## Boundaries (deliberate v1 scope)

- The spec's sandboxed-pandas escape hatch (6a.3) is OUT of scope — prep is declarative
  plans only, validated against the real dataframe and executed by natex code.
- `propose_forcing_variables` / `propose_candidate_events` are not separate mid-run tasks
  (see Task vocabulary above).
- API-backend live calls are never tested in CI; SDK call shape is pinned by fake-client
  kwarg assertions and the manual smoke recorded in the status doc.
- `natex study` always applies at least NullBackend heuristics; CLI `--backend null` passes
  `guidance=None` to `discover`, keeping the null case hook-free.
