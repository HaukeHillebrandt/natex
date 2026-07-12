# Phase llm-analyst implementation plan — LLM analyst pass (Stage 0) + guidance backends

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` **section 6** (6a Stage-0
analyst pass, 6b targeted-first-exhaustive-still, 6c in-scan guidance hooks + backends), section 4
(`natex.study` / `natex.discover` API contract), section 10 risk "LLM guidance could bias discovery".
**First commit of task 1 is this plan file itself** (`docs: phase llm-analyst implementation plan`
— already done at `04902c3`; on resume, the first action is the recovery protocol's step-1 commit
of this updated plan instead).

## Current state & recovery (verified 2026-07-12 — read FIRST, resume idempotently)

A previous execution attempt completed tasks 1–3 of this plan; tasks 1–2 are committed, task 3
is code-complete but UNCOMMITTED in the working tree. Verified on 2026-07-12:
`uv run ruff check src tests` → "All checks passed!"; `uv run pytest -q tests/test_llm_backends.py
tests/test_llm_null.py tests/test_intake_plans.py tests/test_prep_plan.py tests/test_profiler.py`
→ 61 passed.

| Plan task | State | Evidence |
|---|---|---|
| Plan file committed | DONE | commit `04902c3` |
| Task 1 (llm core: models, protocol, MockBackend, GuidanceLog/LoggedBackend) | DONE, committed | commit `b51c0c3`; `src/natex/llm/{__init__,backends,log}.py`, `tests/test_llm_backends.py` |
| Task 2 (PrepPlan + executor) | DONE, committed | commit `e817b0f`; `src/natex/intake/prep.py`, `tests/test_prep_plan.py` |
| Task 3 (plans models + NullBackend) | code DONE, **uncommitted** | modified `src/natex/llm/backends.py` (+`NullBackend`) and `src/natex/llm/__init__.py` (export); untracked `src/natex/intake/plans.py`, `tests/test_intake_plans.py`, `tests/test_llm_null.py` — all green |
| Tasks 4–11 | NOT STARTED | — |

**Recovery protocol (replaces task 3's implementation work; everything else unchanged):**
1. First action: commit this updated plan file —
   `git add docs/plans/phase-llm-analyst.md && git commit -m "docs: update phase llm-analyst plan with recovery state"`.
2. Re-run the two verification commands above; both must be green (they were on 2026-07-12).
   If anything fails, fix per task 3's spec below before committing.
3. Commit the recovered task-3 work exactly as task 3 specifies:
   `git add src/natex/llm/backends.py src/natex/llm/__init__.py src/natex/intake/plans.py tests/test_intake_plans.py tests/test_llm_null.py`
   then `git commit -m "feat(llm): NullBackend profile-only heuristics; Understanding/DesignCandidate/SearchPlan models"`.
4. Run the FULL suite once (`uv run pytest -q`) to confirm no cross-module breakage, then
   proceed to task 4. Do not re-implement tasks 1–3; if a later task requires changing their
   files, that change belongs to the later task's commit.

## Phase objective

The LLM-as-analyst layer, and nothing else (no DEE, no IV, no reporting work):

1. `natex/llm/` — `GuidanceBackend` protocol + `GuidanceRequest`/`GuidanceResponse` pydantic
   models; implementations `NullBackend` (deterministic profile-only heuristics), `MockBackend`
   (canned responses for tests), `AgentBackend` (file-based request/response, subscription mode),
   `AnthropicBackend`/`GeminiBackend` (API mode, behind a new `[llm]` extra). Every
   request+response appended to a JSONL guidance log.
2. `natex/intake/` Stage 0 — `natex.study(csv_or_df, context, guidance, rng) -> IntakeReport`:
   profile → understand → declarative `PrepPlan` (executed ONLY by natex code; the LLM never
   emits code) → `SearchPlan` (ranked `DesignCandidate`s + budget hints). `IntakeReport`
   serializable; `intake.prepare()` returns a `Dataset` ready for discovery.
3. `natex/discover.py` — `natex.discover(data, design, guidance, search_plan, rng, budget)`:
   ranked candidates scanned first, remaining configurations after, within budget; the report
   ALWAYS records searched vs not-searched configurations (spec 6b).
4. In-scan hooks (spec 6c): `interpret_discovery` + `audit_assumptions` after validation,
   `review_control_group` on the DiD GESS path — all ADVISORY, never gating statistics; a veto is
   only ever a flag in the output.
5. CLI: `natex study CSV --context ... --backend null|agent|anthropic|gemini --out DIR` and
   `natex discover --plan intake_report.json`.
6. Blind-vs-informed eval scaffold: `benchmarks/guidance_eval.py` measuring the rank of the true
   design in the search plan, Null vs a provided backend; CI-tested with MockBackend only.

**Hard constraints:** everything works WITHOUT any LLM (NullBackend degrades to profile-only
heuristics); CI never needs network or API keys (all LLM tests use MockBackend / fake injected
clients / import-guard monkeypatching); `anthropic` and `google-genai` live ONLY under the new
`llm` extra; core deps stay numpy/scipy/pandas/scikit-learn/typer/pydantic.

## Audit / spec corrections that bind this phase

| # | Rule | Where implemented |
|---|---|---|
| spec 6b | Search plan orders the scan, never silently truncates it: ranked candidates first at full resolution, exhaustive remainder within budget, and the results bundle always reports what was and wasn't searched | `discover.py`: every enumerated configuration becomes a `ConfigRecord` with `status` in `scanned/skipped_budget/failed/invalid`; `searched` summary block; budget cuts are recorded, never dropped |
| spec 6c | Guidance proposes and vetoes but never fabricates statistics; every hook response is logged into the results bundle; hooks are optional | advisory-only `ConfigRecord.advisory`; mutation test: identical statistical output with and without a (veto-ing) backend |
| spec 6c | `review_control_group` veto respected ONLY as a flag | `effects["gess"]["vetoed_by_guidance"]` boolean; τ̂/se/p never dropped or changed |
| spec §4 | Discovery never reads the outcome | scan paths untouched; guidance payloads carry NO raw outcome values (test asserts absence); Stage-0 prep filters that touch a candidate outcome column are flagged in `guidance_errors` (data-snooping warning, audit item 1 lineage) |
| audit 1 lineage | +1-rank Monte-Carlo p-values, honest framing | `discover.py` reuses `randomization_test`/`panel_randomization_test` verbatim; no new inference code |
| audit 8 / house | NaN never 0.0 on failure; no bare except | failed configs get `status="failed"`, `llr/p_value=None` in JSON (via the shared `jsonable` helper, NaN→null); backends catch ONLY `(TimeoutError, ValueError, RuntimeError, json.JSONDecodeError)` where stated |
| house | One `numpy.random.Generator` through every stochastic call; identical seed ⇒ identical output | `study`/`discover` require an explicit Generator (raise ValueError if None, repo convention); `PrepPlan.subsample.seed` is drawn from that Generator by `study()` and stored IN the plan so the serialized plan replays bitwise without the original rng |
| spec 6c reproducibility | Guidance logged into the results bundle | `GuidanceLog` JSONL: one line per request+response, every backend, including Null/Mock |
| spec §10 risk | Guided vs unguided must be benchmarkable | `benchmarks/guidance_eval.py` (task 10) |

## House rules (bind every task)

Python ≥3.11, CI 3.11–3.14 green (`.github/workflows/ci.yml` unchanged: `uv sync --extra dev`,
`uv run ruff check src tests`, `uv run pytest -q`). `uv run pytest -q` excludes backtests; this
phase adds NO backtest-marked tests. Never commit datasets. Conventional commit after every green
cycle. NaN never 0.0. No bare except. Subagent cwd resets between Bash calls — absolute quoted
paths, prefix commands with `cd /Users/haukehillebrandt/dev/natex &&`.

**TDD discipline for every task:** write the failing test(s) first, run them to confirm failure,
implement, run `uv run pytest -q` and `uv run ruff check src tests`, then commit.

**Polling-test policy (binding):** any test that exercises AgentBackend polling must use
`poll_interval <= 0.05 s` and `timeout <= 5 s` (task 4 uses 0.02 s / 0.15 s) so nothing ever
blocks the suite; the 600 s production default is asserted on the constructor signature only,
never waited on.

**Statistical-test policy** (napkin): every stochastic assertion is seeded; calibrate thresholds
across ≥5 seeds during implementation, pin one seed with margin, record observed ranges in the
test-file docstring. The scans in this phase's tests are small (n≤500, k=25, Q=9) so suite time
stays bounded.

## Current interfaces built upon (do not break; all changes additive)

- `natex.intake.profiler.profile(df) -> IntakeProfile` — `n_rows`, `columns:
  list[ColumnProfile(name, dtype, n_unique, missing_frac, is_numeric, is_binary, is_time_like)]`,
  `panel_candidates: list[(unit, time)]`, `forcing_candidates`, `treatment_candidates`,
  `.to_json()`. The Null heuristics consume EXACTLY this.
- `natex.data.spec.Dataset / DatasetSpec` — `from_csv` covariate default = all columns minus
  {treatment, outcome}; forcing must be ⊆ covariates; `spec.time`/`spec.unit` optional; the
  DiD path takes a `Dataset` with `time` set (covariates may include time/unit — `build_panel`
  excludes them from dims).
- `natex.rdd.lord3.lord3_scan(dataset, k, model, degree, rng, geometry, centers) -> LoRD3Result`
  (`discoveries: list[Discovery(center_index, k, llr, normal, members, group1)]`);
  `natex.scan.coarse.coarse_to_fine_scan(...) -> CoarseToFineResult` (`.result`,
  `.frac_centers_scanned`, `.params`).
- `natex.validate.randomization.randomization_test(ds, res, Q, rng, scan_kwargs) ->
  RandomizationReport(p_value, observed_max_llr, ...)`;
  `natex.validate.placebo.placebo_tests(ds, d) -> PlaceboReport(p_holm, passed)`;
  `natex.validate.density.density_test(ds, d) -> DensityReport(p_value)`.
- `natex.estimate.local2sls.local_2sls / wald_estimate -> EffectEstimate(tau, se, ci, method,
  first_stage_t, weak_instrument, ...)`.
- DiD: `build_panel(ds, bins)`, `suddds_scan(ds, windows, restarts, model, method, bins, degree,
  rng, panel) -> SuDDDSResult`, `panel_randomization_test`, `composition_test`,
  `fit_did_background`, `anticipation_test`, `did_effect(panel, top, control=str|ControlResult)`,
  `tau_randomization_test`, `gess_control(panel, discovery) -> ControlResult` with
  `extras{"profile","expansions","mse_trace","n_control","n_tau"}`. `did_effect` already accepts
  a precomputed `ControlResult` — the GESS hook wiring needs NO change to `did/`.
- CLI `cli.py::_clean` (NaN→null JSON scrubber) — extracted to a shared module in task 6.
- Synthetic DGPs for tests: `natex.data.synthetic.make_synthetic(n, px, pz, zeta, tau, kind,
  rng, ...) -> (Dataset, D)` (columns `x0..x{px-1}, T, y`; forcing = `x0..x{pz-1}`);
  `natex.data.synthetic_did.make_did_synthetic`.
- Existing fake-registry test helper pattern: `tests/test_cli.py::_write_fake_test_score` (columns
  of the MDRC test-score CSV, tiny n, no network) — reuse for the null-backend study test.

**SDK facts pinned for task 5** (verified against the claude-api reference, 2026-07): Anthropic
structured output = `client.messages.create(..., output_config={"format": {"type": "json_schema",
"schema": <schema>}})`; response JSON is in the first `text` content block; schemas require
`additionalProperties: false` + `required` on every object node and REJECT numeric/string
constraints (`minimum`, `maximum`, `exclusiveMinimum/Maximum`, `multipleOf`, `minLength`,
`maxLength`); model id `claude-sonnet-5` is valid. google-genai: `client.models.generate_content
(model=..., contents=prompt, config={...})` with `response_mime_type="application/json"` and the
JSON schema passed via `response_json_schema` (fall back to `response_schema` on older SDKs);
parsed text at `resp.text`. Neither SDK is imported at module top level anywhere in `natex`.

---

## Task 1 — plan commit + `natex/llm` core: request/response models, protocol, MockBackend, guidance log

**First action: commit this plan file** — `git add docs/plans/phase-llm-analyst.md && git commit -m "docs: phase llm-analyst implementation plan"`.

**Create** `src/natex/llm/__init__.py`, `src/natex/llm/backends.py`, `src/natex/llm/log.py`,
`tests/test_llm_backends.py`.

`backends.py` (module docstring: spec 6c contract — guidance proposes/vetoes, never fabricates
statistics; everything optional; everything logged):

```python
TASKS = ("understand", "prepare", "search_plan",
         "interpret_discovery", "audit_assumptions", "review_control_group")
Task = Literal["understand", "prepare", "search_plan",
               "interpret_discovery", "audit_assumptions", "review_control_group"]

class GuidanceRequest(BaseModel):
    task: Task
    payload: dict
    schema_hint: dict = Field(default_factory=dict)  # JSON schema the content should satisfy

class GuidanceResponse(BaseModel):
    content: dict
    raw_text: str = ""
    backend: str

@runtime_checkable
class GuidanceBackend(Protocol):
    name: str
    def complete(self, request: GuidanceRequest) -> GuidanceResponse: ...

TASK_INSTRUCTIONS: dict[str, str]   # one short instruction paragraph per task, shared by
                                    # AgentBackend request files and both API backends' prompts

class MockBackend:
    """Canned responses for tests. Pops `responses` in order; records every request."""
    name = "mock"
    def __init__(self, responses: list[dict | GuidanceResponse]): ...
    requests: list[GuidanceRequest]                      # appended on every complete()
    def complete(self, request) -> GuidanceResponse      # dict -> wrapped as GuidanceResponse(
                                                         #   content=d, raw_text=json.dumps(d),
                                                         #   backend="mock")
    # exhausted -> RuntimeError(f"MockBackend exhausted after {n} responses (task={task!r})")
```

`log.py`:

```python
class GuidanceLog:
    def __init__(self, path: str | Path): ...      # parent dirs created; file NOT truncated
    path: Path
    n_entries: int
    def append(self, request: GuidanceRequest, response: GuidanceResponse) -> None
    # one JSON line: {"seq": n, "ts": <UTC ISO-8601>, "task": ..., "backend": response.backend,
    #                 "request": request.model_dump(), "response": response.model_dump()}

class LoggedBackend:
    """Decorator: delegates to `inner`, appends every request+response to `log`."""
    def __init__(self, inner: GuidanceBackend, log: GuidanceLog): ...
    name: str  # == inner.name (the log records the REAL backend)
    def complete(self, request) -> GuidanceResponse
```

`__init__.py` re-exports: `TASKS, GuidanceRequest, GuidanceResponse, GuidanceBackend,
MockBackend, GuidanceLog, LoggedBackend` (Null/Agent/API backends added in tasks 3–5).

**Tests** (`tests/test_llm_backends.py`):
- `GuidanceRequest(task="bogus", ...)` raises pydantic `ValidationError`; all six TASKS validate.
- MockBackend returns responses in order; dict responses wrapped with `backend == "mock"` and
  `json.loads(raw_text) == content`; `requests` records tasks in call order; exhaustion raises
  `RuntimeError` mentioning the task name.
- `isinstance(MockBackend([]), GuidanceBackend)` (runtime-checkable protocol).
- GuidanceLog: two appends → file has exactly 2 lines, each `json.loads`-able with keys
  `{seq, ts, task, backend, request, response}`, `seq` = 0 then 1; constructing a second
  `GuidanceLog` on the same path appends (does not truncate).
- LoggedBackend: wraps a MockBackend; response passed through unchanged (same object fields);
  log gains one entry per call; `name == "mock"`.
- `TASK_INSTRUCTIONS` has exactly the six TASKS keys, all non-empty strings.

**Commit:** `feat(llm): guidance request/response models, backend protocol, MockBackend, JSONL guidance log`

---

## Task 2 — declarative PrepPlan + executor (the LLM never emits code)

**Create** `src/natex/intake/prep.py`, `tests/test_prep_plan.py`.

```python
ROLES = ("treatment", "outcome", "forcing", "covariate", "time", "unit", "ignore")
OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "notna")

class PrepFilter(BaseModel):
    col: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "notna"]
    value: object | None = None          # list required for "in"; ignored for "notna"

class Subsample(BaseModel):
    n: int                               # validated > 0 in a field_validator (NOT gt=0 —
    seed: int = 0                        # constraint keys break strict LLM schemas, task 5)

class PrepPlan(BaseModel):
    version: int = 1
    column_roles: dict[str, str] = Field(default_factory=dict)   # col -> ROLES member
    encodings: dict[str, Literal["onehot", "ordinal"]] = Field(default_factory=dict)
    discretize: dict[str, int] = Field(default_factory=dict)     # col -> n_bins (>= 2)
    drop_cols: list[str] = Field(default_factory=list)
    subsample: Subsample | None = None
    filters: list[PrepFilter] = Field(default_factory=list)

    def validate_against(self, df: pd.DataFrame) -> None
    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]
```

`validate_against` raises `ValueError` with a message NAMING the offending entries for: unknown
columns anywhere (roles / encodings / discretize / drop_cols / filters), role value not in ROLES,
more than one column per role for treatment/outcome/time/unit, `discretize` bins < 2 or on a
non-numeric column, `encodings` on a column also in `discretize`, a column both dropped and
role-assigned/encoded/discretized, an `"in"` filter whose value is not a list/tuple, a
non-"notna" filter with `value=None`, `subsample.n < 1`.

`apply` (deterministic; same plan + same df ⇒ bitwise-identical output; never mutates the input
df): order = `validate_against` → filters (row masks; comparison ops via pandas, NaN comparisons
are False, never coerced) → drop_cols → encodings (`onehot`: `pd.get_dummies(df, columns=[c],
dtype=float)`; `ordinal`: `pd.factorize(df[c], sort=True)` codes as float, NaN stays NaN via
code −1 → np.nan) → discretize (reuse `natex.did.panel.quantile_bins`; column replaced by
float codes) → subsample (`np.random.default_rng(seed)` + `rng.choice(len(df), size=min(n, len),
replace=False)`, then `np.sort(idx)` so row order is stable; documented: the plan carries its
OWN seed so a serialized plan replays without the caller's Generator — `study()` draws that seed
from the pipeline Generator, task 6). Returns `(df2, log)` where `log` is human-readable lines
like `"filter pretest >= 100: 30 -> 24 rows"`, `"drop columns: ['ID']"`, `"onehot mtongue ->
['mtongue_en', ...]"`, `"subsample n=20000 seed=7: 60000 -> 20000 rows"`. Empty result after
filters is allowed but logged `"WARNING: 0 rows remain"`.

**Tests** (`tests/test_prep_plan.py`):
- Golden apply: hand-built 8-row df; plan with one of each op class (`>=`, `in`, `notna`), one
  drop, one ordinal, one onehot, discretize n_bins=2, subsample n=3 seed=0 → assert EXACT
  resulting shape, columns, and (for the ordinal/discretize columns) exact code values; assert
  `log` contains a line per step; run `apply` twice → `pd.testing.assert_frame_equal` bitwise.
- Input df unmutated (copy compared before/after).
- Rejections, each `pytest.raises(ValueError, match=...)` naming the offender: unknown column in
  each of the five places; bad role; two outcomes; bins=1; discretize on a string column; "in"
  with scalar; `==` with value None; drop+role conflict.
- Round trip: `PrepPlan.model_validate_json(plan.model_dump_json())` applies identically.
- NaN policy: filter `> 5` on a column with NaN drops the NaN rows (comparison False), and the
  ordinal encoding of a NaN cell is NaN — never 0.0.

**Commit:** `feat(intake): declarative PrepPlan with validated, deterministic executor`

---

## Task 3 — Understanding/DesignCandidate/SearchPlan models + NullBackend heuristics

**Create** `src/natex/intake/plans.py`, `tests/test_intake_plans.py`; **modify**
`src/natex/llm/backends.py` (+ exports), **create** `tests/test_llm_null.py`.

`plans.py`:

```python
class ColumnGuess(BaseModel):
    column: str
    reason: str = ""

class DiDStructure(BaseModel):
    unit: str
    time: str
    reason: str = ""

class Understanding(BaseModel):
    shape: Literal["cross-section", "time-series", "panel", "aggregated-cells"]
    unit_of_observation: str = "row"
    treatments: list[ColumnGuess] = Field(default_factory=list)
    outcomes: list[ColumnGuess] = Field(default_factory=list)
    forcing: list[ColumnGuess] = Field(default_factory=list)
    did_structures: list[DiDStructure] = Field(default_factory=list)
    quirks: list[str] = Field(default_factory=list)
    notes: str = ""

class DesignCandidate(BaseModel):
    design: Literal["rdd", "did"]
    treatment: str
    outcome: str | None = None
    forcing: list[str] = Field(default_factory=list)   # rdd: must be nonempty
    unit: str | None = None                            # did only
    time: str | None = None                            # did: must be set
    rationale: str = ""
    priority: int = 0                                  # 0 = scan first
    # model_validator: rdd -> forcing nonempty; did -> time is not None
    def key(self) -> tuple    # ("rdd", treatment, tuple(sorted(forcing))) |
                              # ("did", treatment, unit, time) — dedup key used by discover()

class SearchPlan(BaseModel):
    candidates: list[DesignCandidate] = Field(default_factory=list)
    budget: dict = Field(default_factory=dict)         # hints: k, q, coarse, n_coarse, max_configs
    def ranked(self) -> list[DesignCandidate]          # stable sort by priority
```

`NullBackend` in `backends.py` — deterministic, derived from the request payload alone, NO rng,
NO data access. Per-task heuristics (documented verbatim in the class docstring; constants
module-level and named):

| task | payload contract | heuristic content |
|---|---|---|
| `understand` | `{"profile": <IntakeProfile dict>, "context": str\|None}` | `shape`: `"panel"` if `panel_candidates` else `"aggregated-cells"` if `n_rows < _AGG_MAX_ROWS (5000)` and some column name matches `^(n\|count\|n_obs\|weight\|wt\|pop\|population\|cells?)$` (case-insensitive) else `"time-series"` if some time-like column has `n_unique == n_rows` else `"cross-section"`. `unit_of_observation`: first panel candidate's unit column else `"row"`. `treatments` = profile `treatment_candidates` (reason `"binary 0/1 column"`). `outcomes` = numeric, non-binary, non-time-like columns (reason `"numeric, non-binary"`). `forcing` = profile `forcing_candidates` minus time-like columns (reason `"numeric with >= 20 distinct values"`). `did_structures` from `panel_candidates` (reason `"unit x time grid covers >= 95% of rows"`). `quirks`: `"<col>: constant column"` for n_unique <= 1; `"<col>: {pct}% missing"` for missing_frac > `_QUIRK_MISSING (0.2)`. `notes` names the backend: `"NullBackend heuristics (no LLM)"`. |
| `prepare` | `{"profile": ..., "understanding": ..., "seed": int, "context": ...}` | PrepPlan dict: `drop_cols` = constant columns + columns with missing_frac > `_DROP_MISSING (0.5)`; `subsample = {"n": _SUB_N (20000), "seed": payload["seed"]}` iff `n_rows > _SUB_MAX (50000)`; everything else empty (profile-only degradation, spec 6a). |
| `search_plan` | `{"profile": <post-prep profile dict>, "understanding": ..., "context": ...}` | For each treatment guess `t` (profile order): one `rdd` candidate — outcome = first outcome guess ≠ t (else None), forcing = all forcing guesses ∉ {t, outcome}; skipped if forcing empty. Then for each (t × did_structure): one `did` candidate (unit/time from the structure, outcome as above). `priority` = running index. `rationale` states the rule that fired. `budget` = `{"k": 50, "q": 99, "coarse": n_rows > _COARSE_MIN (10000), "n_coarse": 2000}`. |
| `interpret_discovery` | discovery summary dict (task 8 defines it) | `{"summary": "<deterministic sentence naming design, dominant forcing column (max abs influence) and location>", "matched_policies": [], "confounded_risk": "unknown", "note": "NullBackend: no domain knowledge applied"}` |
| `audit_assumptions` | candidate + validation dict | `{"excludability": "unreviewed", "monotonicity": "unreviewed", "sutva": "unreviewed", "veto": false, "caveats": ["NullBackend: assumption audit requires a human or LLM reviewer"]}` |
| `review_control_group` | GESS expansion dict (task 8) | `{"face_valid": null, "veto": false, "reason": "NullBackend performs no substantive review; n_expansions=<len(expansions)>"}` — NullBackend NEVER vetoes. |

`NullBackend.complete` returns `GuidanceResponse(content=..., raw_text=json.dumps(content,
sort_keys=True), backend="null")`. Unknown task (can't happen through `GuidanceRequest`) is a
`ValueError`, not silent.

**Tests:**
- `tests/test_intake_plans.py` — model validation: rdd candidate without forcing raises; did
  without time raises; `ranked()` stable-sorts by priority; `key()` dedup semantics (forcing
  order-insensitive); JSON round trip.
- `tests/test_llm_null.py` — determinism: same payload twice ⇒ `content` equal AND `raw_text`
  bitwise equal. Shape rules: profile of a panel df (reuse `test_profiler` construction) ⇒
  `"panel"` + did_structures nonempty; time-series rule; aggregated-cells rule (df with `count`
  column, 100 rows); cross-section fallback. Every Null `understand`/`prepare`/`search_plan`
  content VALIDATES through `Understanding`/`PrepPlan`/`SearchPlan` (`model_validate` — this is
  the contract `study()` relies on for fallback). `prepare` on a 100-row profile has no
  subsample; on a fabricated 60000-row profile has `{"n": 20000, "seed": payload seed}`.
  `search_plan` on the fake test-score profile ranks a candidate with `treatment == "treat"` and
  nonempty forcing; priorities are 0..m-1. `review_control_group` veto is always False;
  `audit_assumptions` veto False.

**Commit:** `feat(llm): NullBackend profile-only heuristics; Understanding/DesignCandidate/SearchPlan models`

---

## Task 4 — AgentBackend (file-based subscription-mode protocol)

**Create** `src/natex/llm/agent.py`, `tests/test_llm_agent.py`; export from `natex.llm`.

```python
class AgentBackend:
    """File-based request/response guidance for a calling coding agent (spec 6c, zero API cost).

    complete() writes `workdir/requests/{seq:04d}_{task}.json`, prints one instruction line via
    `echo`, then polls `workdir/responses/{seq:04d}_{task}.json` until it parses as JSON or
    `timeout` elapses.
    """
    name = "agent"
    def __init__(self, workdir: str | Path, poll_interval: float = 0.5,
                 timeout: float = 600.0, echo: Callable[[str], None] = print): ...
    def complete(self, request: GuidanceRequest) -> GuidanceResponse
```

Details: `requests/` and `responses/` created on init. `seq` starts at the number of existing
files in `requests/` (restart-safe, monotone). Request file content =
`request.model_dump() | {"instructions": TASK_INSTRUCTIONS[task], "respond_to":
str(response_path)}`, indent=1. Echo line: `f"natex guidance request ({task}): answer by writing
JSON matching schema_hint to {response_path}"`. Poll loop: deadline via `time.monotonic()`;
each tick, if the response file exists, read text; empty text or `json.JSONDecodeError` ⇒ keep
polling (tolerates partial writes — the ONLY exception caught); a parsed non-dict (e.g. a JSON
list) ⇒ `ValueError`. Accepted shapes: a full `{"content": {...}, ...}` envelope (dict `content`
key) → content = that value; any other dict → the whole object is the content. Returns
`GuidanceResponse(content=..., raw_text=<file text>, backend="agent")`. On deadline:
`TimeoutError(f"no guidance response at {response_path} after {timeout:.0f}s; write JSON matching
the schema_hint in {request_path} to that path and re-run, or use --backend null")`.

**Tests** (`tests/test_llm_agent.py`, tmp_path, no sleeps beyond ~0.5 s total):
- Round trip via thread: `poll_interval=0.02`; `threading.Thread` sleeps 0.1 s then writes
  `{"shape": "panel"}` → `complete` returns that content, `backend == "agent"`; request file
  exists, parses, has `task`, `payload`, `schema_hint`, `instructions`, `respond_to`; echo
  captured via a list-appending callable and contains the response path.
- Envelope shape: thread writes `{"content": {"a": 1}, "note": "x"}` → content == `{"a": 1}`.
- Partial-write tolerance: pre-create the response file containing `"{invalid"`, thread
  overwrites with valid JSON after 0.1 s → succeeds.
- Timeout: `timeout=0.15` and nobody writes → `pytest.raises(TimeoutError)` with the response
  path in the message; elapsed < 1 s.
- Sequencing: two `complete` calls (thread answers both) produce `0000_understand.json` and
  `0001_prepare.json`; a NEW AgentBackend on the same workdir starts at seq 2.

**Commit:** `feat(llm): AgentBackend file-based request/response protocol with polling and timeout`

---

## Task 5 — AnthropicBackend + GeminiBackend behind the `[llm]` extra

**Create** `src/natex/llm/api.py`, `tests/test_llm_api.py`; **modify** `pyproject.toml`
(`llm = ["anthropic>=0.40", "google-genai>=1.0"]` under `[project.optional-dependencies]`);
export both backends from `natex.llm`.

```python
_INSTALL_MSG = "requires the {pkg!r} package: pip install 'natex-discovery[llm]'"

def _strict_schema(schema: dict) -> dict
    # deep-copied JSON schema made structured-output-safe: every object node gets
    # additionalProperties: false and required = list(properties); strips unsupported
    # constraint keys {minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf,
    # minLength, maxLength, minItems, maxItems, pattern}; recurses into properties/items/
    # anyOf/allOf/$defs. Pure function, no SDK dependency (unit-testable everywhere).

def _prompt(request: GuidanceRequest) -> str
    # TASK_INSTRUCTIONS[task] + "\n\nInput payload (JSON):\n" + json.dumps(payload, indent=1,
    # sort_keys=True, default=str) + "\n\nRespond with a single JSON object."

class AnthropicBackend:
    name = "anthropic"
    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None,
                 max_tokens: int = 4096, _client=None):
        # _client injection for tests; otherwise:
        #   try: import anthropic
        #   except ImportError: raise ImportError(f"AnthropicBackend {_INSTALL_MSG.format(pkg='anthropic')}") from None
        #   self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    def complete(self, request) -> GuidanceResponse
        # kwargs: model, max_tokens, messages=[{"role": "user", "content": _prompt(request)}];
        # plus output_config={"format": {"type": "json_schema",
        #                                "schema": _strict_schema(request.schema_hint)}}
        # iff request.schema_hint is truthy. Text = first block with .type == "text".
        # json.loads(text) must be a dict -> content; else ValueError with a <=200-char snippet.

class GeminiBackend:
    name = "gemini"
    def __init__(self, model: str = "gemini-3.1-pro", api_key: str | None = None, _client=None):
        # try: from google import genai
        # except ImportError: raise ImportError(f"GeminiBackend {_INSTALL_MSG.format(pkg='google-genai')}") from None
        # self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
    def complete(self, request) -> GuidanceResponse
        # config = {"response_mime_type": "application/json"}
        # + {"response_json_schema": _strict_schema(schema_hint)} iff schema_hint truthy
        #   (documented: on SDKs without response_json_schema the implementer verifies manually
        #    and may pass "response_schema" — a TypeError from generate_content falls back once
        #    to the schema-free config; recorded in the docstring, tested via fake client)
        # resp = self._client.models.generate_content(model=self.model, contents=prompt,
        #                                             config=config)
        # json.loads(resp.text) -> content (dict required, same ValueError policy).
```

Non-determinism note (docstring + method card): API backends are inherently non-deterministic;
reproducibility comes from the guidance log — a run is replayable by feeding the logged responses
back through `MockBackend`.

**Tests** (`tests/test_llm_api.py` — NO network, NO API keys, run on CI regardless of installs):
- `_strict_schema`: on `PrepPlan.model_json_schema()` — every object node (walk recursively,
  incl. `$defs`) has `additionalProperties is False` and `required` == its property names; no
  stripped constraint key remains anywhere (the `Subsample.n` validator from task 2 keeps `gt`
  OUT of the schema — regression-guarded here); input dict unmutated.
- Import guards: `monkeypatch.setitem(sys.modules, "anthropic", None)` ⇒
  `pytest.raises(ImportError, match="natex-discovery\\[llm\\]")` on `AnthropicBackend()`;
  same with `"google"`/`"google.genai"` for `GeminiBackend` (set both keys to None).
- Fake-client tests (always run): a `FakeAnthropicClient` recording kwargs and returning an
  object with `.content = [SimpleNamespace(type="text", text='{"veto": false}')]` ⇒ content
  parsed, `backend == "anthropic"`, recorded kwargs contain `model == "claude-sonnet-5"` and
  `output_config["format"]["schema"]["additionalProperties"] is False` when a schema_hint is
  given, and NO `output_config` key when schema_hint is `{}`. Fake returning non-JSON text ⇒
  `ValueError` with snippet. Same trio for `FakeGeminiClient` (`.models.generate_content`
  recorded; `resp.text` payload; config keys asserted; model `gemini-3.1-pro`).
- `skipif(importlib.util.find_spec("anthropic") is None)` smoke test: constructing
  `AnthropicBackend(api_key="test-key")` succeeds (real SDK import path, still no network);
  mirrored for google-genai. These SKIP gracefully on CI (extra not installed) — CI stays green.
- `uv run pip install -e '.[llm]'` is NOT part of CI; add a comment in pyproject pinning why.

**Commit:** `feat(llm): Anthropic and Gemini structured-output backends behind the llm extra`

---

## Task 6 — Stage-0 pipeline: `natex.study()` + `IntakeReport`

**Create** `src/natex/intake/analyst.py`, `src/natex/jsonutil.py`, `tests/test_study.py`;
**modify** `src/natex/cli.py` (replace `_clean` body with `from natex.jsonutil import jsonable`
— behavior-preserving, existing CLI tests guard it), `src/natex/__init__.py` (export `study`,
`IntakeReport`, `PrepPlan`, `SearchPlan`, `DesignCandidate`).

`jsonutil.py`: `def jsonable(obj)` — exact current `cli._clean` semantics (ndarray→list,
np scalars→python, non-finite floats→None, recurse dict/list/tuple).

`analyst.py`:

```python
@dataclass
class IntakeReport:
    profile: IntakeProfile              # RAW dataframe profile (pre-prep)
    understanding: Understanding
    prep_plan: PrepPlan
    search_plan: SearchPlan
    guidance_log_path: str | None
    context: str | None
    source: str                         # csv path, or "<dataframe>"
    guidance_errors: list[str]          # fallbacks, dropped candidates, snooping warnings
    prep_log: list[str]                 # PrepPlan.apply log of the study() run
    _df: pd.DataFrame | None = field(default=None, repr=False, compare=False)  # not serialized

    def to_json(self) -> str            # jsonable() over all serializable fields
    def save(self, out: str | Path) -> Path
        # writes out/intake_report.json (full report) and out/prep_plan.json (the plan alone,
        # editable by the user); returns the report path
    @classmethod
    def load(cls, path: str | Path) -> "IntakeReport"   # _df=None; profile rebuilt via
        # IntakeProfile(**...) with ColumnProfile(**...) items
    def prepare(self, df: pd.DataFrame | None = None, candidate: int = 0) -> Dataset

def study(csv_or_df, context: str | None = None, guidance: GuidanceBackend | None = None,
          rng: np.random.Generator | None = None, out: str | Path | None = None,
          strict: bool = False) -> IntakeReport
```

`study()` steps (docstring documents the FIXED task order — understand, prepare, search_plan —
which is the MockBackend contract):
1. `rng` required: `raise ValueError("pass an explicit numpy Generator (reproducibility
   contract)")` (repo convention).
2. `df = pd.read_csv(csv_or_df)` if str/Path else the DataFrame; `source` accordingly.
3. Backend = `guidance if guidance is not None else NullBackend()`; if `out` is given, wrap in
   `LoggedBackend(backend, GuidanceLog(Path(out) / "guidance_log.jsonl"))` and record
   `guidance_log_path`; else `guidance_log_path=None`.
4. `prof = profile(df)`; `prof_dict = json.loads(prof.to_json())`.
5. **understand**: `GuidanceRequest(task="understand", payload={"profile": prof_dict,
   "context": context}, schema_hint=Understanding.model_json_schema())`. Validate
   `resp.content` via `Understanding.model_validate`. Backend/parse failure policy (uniform for
   all three steps): catch `(ValidationError, ValueError, TimeoutError, RuntimeError)`; if
   `strict` re-raise as `ValueError` naming the step; else append
   `f"{task}: {exc} -- fell back to NullBackend heuristics"` to `guidance_errors` and take
   `NullBackend().complete(request)` (valid by task-3 test contract). The failed attempt is
   still in the log; the Null fallback response is logged too.
6. **prepare**: payload `{"profile": prof_dict, "understanding": understanding.model_dump(),
   "seed": int(rng.integers(2**31 - 1)), "context": context}`,
   `schema_hint=PrepPlan.model_json_schema()`. After model validation also run
   `prep_plan.validate_against(df)` inside the same fallback policy (an LLM plan naming unknown
   columns falls back rather than crashing, with the rejection message recorded — spec 6a
   "validate the plan against the actual dataframe").
7. `df2, prep_log = prep_plan.apply(df)`; `prof2 = profile(df2)`.
8. **search_plan**: payload `{"profile": json.loads(prof2.to_json()), "understanding": ...,
   "context": context}`, `schema_hint=SearchPlan.model_json_schema()`. Post-validate each
   candidate against `df2.columns`: candidates referencing unknown columns are DROPPED with a
   `guidance_errors` entry naming candidate and columns; if none survive → Null fallback.
9. Snooping guard (audit 1 lineage): for every prep filter / drop col that equals any surviving
   candidate's `outcome`, append a warning to `guidance_errors`
   (`"prep filter touches candidate outcome 'y' -- possible outcome snooping"`). Warning only,
   never a hard failure.
10. Return `IntakeReport(..., _df=df)`.

`prepare(df=None, candidate=0)`: resolve the frame (`df` arg → `self._df` → `pd.read_csv(source)`
if source is an existing path → `ValueError("no dataframe available; pass df=")`); re-apply
`prep_plan`; pick `search_plan.ranked()[candidate]` (IndexError → ValueError naming the range);
build the spec mirroring `Dataset.from_csv` defaults — covariates = all df2 columns minus
{treatment, outcome}; rdd: `forcing=c.forcing`; did: `forcing=[]`, `time=c.time`, `unit=c.unit`;
return `Dataset(df2, spec)` (its constructor enforces numeric forcing etc. — errors propagate,
they are real spec bugs).

**Tests** (`tests/test_study.py`):
- rng required: `pytest.raises(ValueError, match="Generator")`.
- Null end-to-end on the fake test-score CSV (copy the `_write_fake_test_score` helper into this
  file or a small `tests/helpers.py`; no network): `study(csv, context="MDRC test score demo",
  rng=default_rng(0), out=tmp)` ⇒ files `intake_report.json`, `prep_plan.json`,
  `guidance_log.jsonl` exist; log has exactly 3 lines with tasks
  `["understand", "prepare", "search_plan"]` and `backend == "null"`;
  `understanding.shape == "cross-section"`; some candidate has `treatment == "treat"`;
  `IntakeReport.load(path)` round-trips (`search_plan`/`prep_plan`/`understanding` equal);
  `report.prepare()` returns a `Dataset` with `spec.treatment == "treat"` and n>0.
- Determinism: two `study(...)` runs with `default_rng(0)` on the same csv ⇒ identical
  `to_json()` strings.
- Mock end-to-end with fallback: MockBackend with (a) valid understand content, (b) prepare
  content naming an unknown column → assert `guidance_errors` has one entry matching
  `"prepare:.*unknown"`, the applied plan is the Null one, and the run STILL completes; (c)
  valid search_plan whose 2nd candidate references a bogus column → that candidate dropped +
  recorded, 1st candidate survives. Also `strict=True` with (b) ⇒
  `pytest.raises(ValueError, match="prepare")`.
- Mock ranks truth first (the core spec-6a assertion): synthetic CSV from
  `make_synthetic(n=400, px=3, pz=2, zeta=6.0, kind="binary", rng=default_rng(0))` with a decoy
  binary column `holiday` inserted BEFORE `T` (so Null ranks the decoy first); mock search_plan
  puts `("rdd", treatment="T", forcing=["x0", "x1"], priority=0)` first ⇒
  `report.search_plan.ranked()[0].treatment == "T"`; the Null run on the same csv ranks a
  `holiday` candidate at position 0 (guards the eval scaffold's discrimination, task 10).
- Snooping warning: mock prepare plan filtering on `y` + search plan with outcome `y` ⇒
  `guidance_errors` contains `"outcome"` warning; statistics unaffected (plan still applied).
- CLI `_clean` refactor guarded by the existing `tests/test_cli.py` (rerun, unchanged).

**Commit:** `feat(intake): natex.study Stage-0 analyst pipeline with IntakeReport and guidance fallbacks`

---

## Task 7 — `natex.discover`: targeted first, exhaustive still, budget-aware, coverage always reported

**Create** `src/natex/discover.py`, `tests/test_discover.py`; **modify** `src/natex/__init__.py`
(export `discover`, `DiscoverReport`, `ConfigRecord`).

```python
_BUDGET_DEFAULTS = {"max_configs": None, "k": 50, "q": 99, "degree": 1, "coarse": False,
                    "n_coarse": 2000, "bins": 4, "restarts": 8, "method": "single_delta",
                    "model": "auto", "windows": None}

@dataclass
class ConfigRecord:
    candidate: DesignCandidate
    source: str          # "plan" | "exhaustive"
    status: str          # "scanned" | "skipped_budget" | "failed" | "invalid"
    llr: float | None
    p_value: float | None
    n_discoveries: int
    summary: dict        # design-specific result block (below); {} unless scanned
    advisory: dict       # guidance hook responses / errors (task 8); ALWAYS advisory
    error: str | None

@dataclass
class DiscoverReport:
    configs: list[ConfigRecord]          # execution order: plan-ranked first, then exhaustive
    searched: dict                       # spec 6b: {"n_total", "n_scanned", "n_skipped_budget",
                                         #  "n_failed", "n_invalid", "budget": <effective dict>,
                                         #  "plan_candidates": int, "exhaustive_candidates": int}
    best_index: int | None               # argmax llr over scanned configs; None if none scanned
    guidance_log_path: str | None
    def best(self) -> ConfigRecord | None
    def to_json(self) -> str             # via natex.jsonutil.jsonable (NaN -> null, never 0)
    def save(self, out) -> Path          # out/discover_report.json

def enumerate_configs(data: Dataset, design: str = "auto") -> list[DesignCandidate]
def discover(data: Dataset, design: str = "auto",
             guidance: GuidanceBackend | None = None,
             search_plan: SearchPlan | None = None,
             rng: np.random.Generator | None = None,
             budget: dict | None = None,
             out: str | Path | None = None) -> DiscoverReport
```

Semantics:
- `rng` required (ValueError, repo convention). `design` ∈ {"auto", "rdd", "did"} else ValueError.
- **Effective budget** = `_BUDGET_DEFAULTS` ← `search_plan.budget` (hints) ← `budget` arg
  (explicit wins); unknown keys → ValueError naming them. Recorded verbatim in `searched`.
- **Config list** (spec 6b — plan orders, never truncates):
  1. plan candidates (`search_plan.ranked()` if given), filtered by `design`; a candidate
     referencing a column not in `data.df.columns`, or a non-numeric forcing column, becomes a
     `ConfigRecord(status="invalid", error=<reason>)` — recorded, not silently dropped;
  2. exhaustive remainder from `enumerate_configs(data, design)`: an rdd candidate from the
     bound spec (treatment/outcome/forcing of `data.spec`; requires nonempty forcing) and a did
     candidate iff `data.spec.time is not None`; dedup against plan candidates on
     `DesignCandidate.key()`.
- **Execution**: sequential over valid configs; once `max_configs` scans have happened, the rest
  get `status="skipped_budget"` (still listed, spec 6b). Per config: if the candidate equals the
  bound spec, use `data`; else build `Dataset(data.df, DatasetSpec(...))` with the from_csv-style
  covariate default. Any `(ValueError, RuntimeError, np.linalg.LinAlgError)` inside a config ⇒
  `status="failed"`, `error=str(exc)`, sweep continues (a failed config never kills the report,
  and never fabricates numbers — llr/p stay None).
- **rdd config**: `coarse_to_fine_scan(ds, k, n_coarse, degree=degree, rng=rng)` when
  `budget["coarse"]` else `lord3_scan(ds, k=k, degree=degree, rng=rng)`; empty discoveries ⇒
  `status="failed"`, error `"no scoreable neighborhood"`. Otherwise `randomization_test(ds, res,
  Q=q, rng=rng, scan_kwargs={"k": k, "degree": degree})`, `placebo_tests`, `density_test` on the
  top discovery; effects (`local_2sls`, `wald_estimate`) iff `ds.y` is not None. `summary` =
  `{"design": "rdd", "center_z", "normal", "forcing_influence": {col: |normal_j|}, "llr",
  "p_value", "placebo_passed", "placebo_holm", "density_p", "effects": {...},
  "coarse": <coverage block or None>}`.
- **did config**: `build_panel(ds, bins=bins)`, `suddds_scan(ds, windows=windows,
  restarts=restarts, model=model, method=method, bins=bins, degree=degree, rng=rng,
  panel=panel)`; empty ⇒ failed `"no qualifying discovery"`. Then `panel_randomization_test`,
  `composition_test`, `fit_did_background` + `anticipation_test`; effects per control
  (`dd`, `synthetic`, `gess`) via `did_effect` + `tau_randomization_test` iff `ds.y` is not
  None (gess wiring is task 8's hook point — build it as `ctrl = gess_control(panel, top)` then
  `did_effect(panel, top, control=ctrl)` NOW so task 8 only adds the hook call). `summary` =
  `{"design": "did", "subset_values", "t0", "window", "llr", "p_value", "null_kind",
  "composition_passed", "anticipation_passed", "effects", "searched_windows": list(res.windows),
  "restarts"}`.
- `ConfigRecord.llr` = observed max LLR, `p_value` from the randomization report,
  `n_discoveries` = len(result.discoveries).
- `guidance`/`out` accepted now; hooks land in task 8 (`advisory={}` until then). If `out` given,
  wrap guidance in `LoggedBackend` exactly as `study()` does.

**Tests** (`tests/test_discover.py`; all with small n and q=9, seeded):
- rng required; bad design/budget key raise with names in the message.
- rdd auto path: `make_synthetic(n=300, zeta=6, kind="binary", rng=default_rng(0))` ⇒
  `discover(ds, rng=default_rng(1), budget={"k": 25, "q": 9})`: 1 config, scanned, `llr > 0`,
  `0 < p <= 1`, `best()` is it; JSON round trip: `json.loads(rep.to_json())` has the `searched`
  block with `n_total == 1`.
- Plan-first ordering + full coverage: same df with a decoy binary column; search_plan with
  priority-0 candidate (treatment "T") and a priority-1 decoy candidate (treatment "holiday",
  same forcing) ⇒ `configs[0].source == "plan"` and treatment "T"; the decoy AND the exhaustive
  spec config all appear (dedup: the exhaustive (T, x0, x1) config is absorbed by the identical
  plan candidate — `n_total == 2`); with `budget={"max_configs": 1, ...}` the decoy is
  `skipped_budget` and STILL listed (`searched["n_skipped_budget"] == 1`) — the spec-6b core
  assertion.
- Invalid plan candidate (unknown treatment column) ⇒ `status == "invalid"`, error names the
  column, other configs still scanned.
- Failure isolation: plan candidate whose forcing column is constant/degenerate (or monkeypatch
  `lord3_scan` to raise ValueError for the first config) ⇒ first record `failed` with the
  message, second scanned; NaN policy — failed record's llr/p are `None` in `to_json()`.
- did path: reuse the `test_cli.py::make_did_synthetic` construction (bins=2, restarts=2, q=9)
  ⇒ scanned did config with `summary["design"] == "did"`, effects for all three controls
  present, `searched_windows` nonempty.
- Determinism: identical calls with `default_rng(7)` twice ⇒ `to_json()` bitwise equal.

**Commit:** `feat(discover): budget-aware ranked-first exhaustive-still discovery orchestrator with full search coverage reporting`

---

## Task 8 — in-scan guidance hooks: interpret_discovery, audit_assumptions, review_control_group

**Modify** `src/natex/discover.py`; **extend** `tests/test_discover.py`.

Hook wiring inside `discover()` (all spec 6c; each hook fires only when a guidance backend was
passed; hook ORDER per scanned config is part of the docstring contract for MockBackend users:
`interpret_discovery` → `audit_assumptions` → (did only, during gess effects)
`review_control_group`):

- After a config's validation block (both designs): payload =
  `{"candidate": candidate.model_dump(), "summary": <the summary block WITHOUT the effects key>,
  "context": None}` → `interpret_discovery`; response content stored at
  `advisory["interpret_discovery"]`.
- Before the effects block (both designs; fires even when outcome is None — the audit is about
  the design): payload = `{"candidate": ..., "validation": {p_value, placebo/composition
  fields}}` → `audit_assumptions`; stored at `advisory["audit_assumptions"]`; if
  `content.get("veto")` is truthy → `advisory["vetoed"] = True` AND
  `summary["advisory_veto"] = True`. Effects are STILL computed (never gate statistics).
- GESS path (did effects): after `ctrl = gess_control(panel, top)` and BEFORE reporting, payload
  = `{"profile": ctrl.extras["profile"], "expansions": ctrl.extras["expansions"], "mse_trace":
  ctrl.extras["mse_trace"], "subset_values": top.subset_values, "n_control":
  ctrl.extras["n_control"], "n_tau": ctrl.extras["n_tau"]}` → `review_control_group`; stored at
  `advisory["control_review"]`; `effects["gess"]["vetoed_by_guidance"] =
  bool(content.get("veto", False))`. τ̂/se/p for gess are computed and reported REGARDLESS.
- Hook failure isolation: each hook call wrapped in
  `try/except (TimeoutError, ValueError, RuntimeError) as exc:` →
  `advisory[<hook>] = {"error": str(exc)}`; the config's statistics are untouched.
- Payload hygiene (audit "discovery never reads y" lineage): hook payloads carry NO raw data
  arrays and NO outcome values — only summary statistics already destined for the results
  bundle. Enforced by test.

**Tests** (extend `tests/test_discover.py`):
- Mock hooks recorded: rdd run with `MockBackend([{...interpret...}, {"veto": True, ...}])` ⇒
  `advisory["interpret_discovery"]` equals the canned dict; `advisory["vetoed"] is True`;
  `summary["advisory_veto"] is True`; effects block STILL present with finite tau (never gated).
- Never-gates mutation test: same data + same seed, run A `guidance=None`, run B mock with
  veto=True everywhere ⇒ strip `advisory`/`advisory_veto`/`vetoed_by_guidance` keys from both
  reports' JSON and assert the remainders are EQUAL (hooks consume no rng, statistics bitwise
  identical).
- did gess review: mock third response `{"face_valid": false, "veto": true, "reason": "expanded
  into implausible profile"}` ⇒ `effects["gess"]["vetoed_by_guidance"] is True`,
  `effects["gess"]["tau"]` unchanged vs the guidance-None run; `advisory["control_review"]`
  recorded; hook payload (captured from `mock.requests`) has keys exactly
  `{profile, expansions, mse_trace, subset_values, n_control, n_tau}`.
- Hook error isolation: backend whose `complete` raises `TimeoutError("agent silent")` ⇒ config
  still `scanned`, `advisory["interpret_discovery"]["error"]` contains "agent silent".
- Payload hygiene: for every request recorded by the mock, `json.dumps(request.payload)` does
  NOT contain any value of the outcome column (assert none of the first 5 y-values' repr appears)
  and contains no key named "y" / the outcome column name mapping to a list.
- Guidance log: `discover(..., guidance=mock, out=tmp)` writes `guidance_log.jsonl` with one
  line per hook call, and `DiscoverReport.guidance_log_path` points at it.

**Commit:** `feat(discover): advisory guidance hooks (interpret/audit/control-review) logged into the results bundle, never gating statistics`

---

## Task 9 — CLI: `natex study` and `natex discover --plan`

**Modify** `src/natex/cli.py`; **create** `tests/test_cli_study.py`.

- Backend factory `_make_backend(backend: str, model: str | None, workdir: Path | None)
  -> GuidanceBackend | None`: `"null"→None` (study/discover substitute NullBackend internally;
  passing None keeps `discover` hook-free for the null case — document that `natex study`
  ALWAYS uses at least NullBackend), `"agent"→AgentBackend(workdir or out/"agent")`,
  `"anthropic"→AnthropicBackend(model=model or default)`, `"gemini"→GeminiBackend(...)`; unknown
  ⇒ echo + `typer.Exit(2)`. ImportError from missing extras ⇒ echo the install message +
  `typer.Exit(2)` (no traceback).
- `natex study CSV --context TEXT --backend null --model TEXT --workdir PATH --seed 0 --out out`:
  runs `study(csv, context=..., guidance=..., rng=default_rng(seed), out=out)`, calls
  `report.save(out)`, echoes: shape + unit of observation, number of candidates + the top
  candidate line (`design/treatment/forcing-or-time`), any `guidance_errors` (prefixed
  `"warning: "`), and the three output paths. Exit 0.
- `natex discover`: make `csv` optional (`typer.Argument(None)`) and `treatment` optional; add
  `--plan PATH`, `--backend`, `--model`, `--workdir`, `--max-configs INT`. Dispatch:
  - `--plan` given: `report = IntakeReport.load(plan)`; frame from `csv` arg if provided else
    `report.source`; `ds = report.prepare(df=...)`; effective budget from the plan's
    `search_plan.budget` merged with CLI `k/q/coarse/n_coarse/max_configs`; run
    `natex.discover(ds, design=..., guidance=..., search_plan=report.search_plan,
    rng=default_rng(seed), budget=..., out=out)`; `rep.save(out)`; echo
    `"scanned X/Y configs (Z skipped by budget)"`, the best config line, and the report path.
    Exit 1 if no config scanned successfully.
  - no `--plan`: EXACTLY the existing rdd/did behavior (requires `--treatment`; error message
    + exit 2 when both plan and treatment missing). Existing `tests/test_cli.py` must pass
    unchanged.
- Help text for `--backend` states the no-network default and that anthropic/gemini need
  `pip install 'natex-discovery[llm]'`.

**Tests** (`tests/test_cli_study.py`, CliRunner, tmp_path):
- `natex study` on the fake test-score CSV with `--backend null` ⇒ exit 0; `intake_report.json`,
  `prep_plan.json`, `guidance_log.jsonl` exist; output mentions `cross-section` and `treat`.
- Unknown backend ⇒ exit 2 with the name in output. Missing-extra path: monkeypatch
  `sys.modules["anthropic"] = None` ⇒ `--backend anthropic` exits 2 and prints
  `natex-discovery[llm]`.
- study→discover round trip on a synthetic CSV (n=300, q=9, k=25): run `study`, then
  `natex discover --plan <out>/intake_report.json <csv> --q 9 --k 25 --seed 0 --out out2` ⇒
  exit 0; `discover_report.json` parses; `searched["n_total"] >= 1`; every config record has a
  `status`; output contains `"scanned"`.
- `--plan` with `--max-configs 1` on a plan with ≥2 candidates ⇒ report shows
  `n_skipped_budget >= 1` (spec 6b through the CLI).
- No plan and no treatment ⇒ exit 2, message names both options.
- Existing `tests/test_cli.py` unchanged and green.

**Commit:** `feat(cli): natex study command and plan-driven natex discover`

---

## Task 10 — blind-vs-informed guidance eval scaffold

**Create** `src/natex/guidance_eval.py` (importable logic, mirrors the `natex.benchmarks` /
`benchmarks/run_*.py` split), `benchmarks/guidance_eval.py` (manual runner), 
`tests/test_guidance_eval.py`; **modify** `benchmarks/README.md` (one section).

`src/natex/guidance_eval.py`:

```python
@dataclass
class EvalCase:
    name: str
    df: pd.DataFrame          # true design buried among decoys
    context: str              # free-text hint an informed backend can exploit
    design: str               # "rdd" | "did"
    treatment: str
    forcing: tuple[str, ...]  # rdd truth ((), for did)
    time: str | None          # did truth

def make_eval_cases(n_rdd: int = 4, include_did: bool = True,
                    rng: np.random.Generator | None = None) -> list[EvalCase]
    # rdd cases: make_synthetic(n=400, px=3, pz=2, zeta=6.0, kind="binary", rng=rng) with a
    # decoy binary column inserted BEFORE T (name from a fixed list: holiday, eligible_flag, ...)
    # and one pure-noise numeric decoy appended; truth = ("rdd", "T", ("x0", "x1")).
    # did case: make_did_synthetic-based panel with unit/time columns; truth matched on
    # (design, treatment, time) only. rng required (repo convention).

def rank_of_truth(plan: SearchPlan, case: EvalCase) -> int | None
    # index in plan.ranked() of the first candidate with matching design + treatment and
    # (rdd) forcing a superset-or-equal of the truth / (did) matching time; None if absent.

def run_guidance_eval(make_backend: Callable[[EvalCase], GuidanceBackend | None],
                      n_rdd: int = 4, include_did: bool = True, seed: int = 0) -> pd.DataFrame
    # per case: study(df, context, guidance=None, rng=default_rng(seed)) -> rank_null;
    #           study(df, context, guidance=make_backend(case), rng=default_rng(seed)) -> rank;
    # columns: [case, design, n_candidates_null, n_candidates_backend, rank_null, rank_backend]
    # (ranks may be None -> pandas NA). No file IO here.
```

`benchmarks/guidance_eval.py`: argparse `--backend null|agent|anthropic|gemini`, `--model`,
`--workdir`, `--n-rdd`, `--seed`, `--out benchmarks/out/guidance_eval.csv`; builds
`make_backend` closures (null ⇒ `lambda case: None`, i.e. the blind arm twice — a smoke mode);
prints the mean rank per arm and writes the CSV. Docstring: the gate question is "does the
informed plan hit the true config earlier (lower rank) than the blind Null plan"; API arms are
manual-only, never CI.

**Tests** (`tests/test_guidance_eval.py`, MockBackend only, no network):
- `make_eval_cases(rng=default_rng(0))` deterministic (two calls ⇒ equal dfs) and each rdd case
  contains its decoy before `T`.
- `rank_of_truth`: hand-built plans → 0, 2, and None.
- End-to-end with an informed mock: `make_backend` returns a fresh
  `MockBackend([<null-style understand>, <null-style prepare>, <search plan with the TRUE config
  at priority 0>])` per case (build the first two by calling `NullBackend` on the case's
  profile — keeps the mock aligned with the payload contract) ⇒ resulting frame has
  `rank_backend == 0` for every case AND `rank_null >= 1` for every rdd case (the decoy-first
  construction from task 6's test guarantees it; calibrate across seeds 0–4 and pin per the
  statistical-test policy) — i.e. the scaffold measurably separates blind from informed.
- Frame shape/columns exact; runtime guard: n_rdd=2, include_did=False for CI speed.

**Commit:** `feat(benchmarks): blind-vs-informed guidance eval scaffold (rank of true design)`

---

## Task 11 — docs, method card, status, final gates

**Create** `docs/method_cards/llm_analyst.md` (Stage-0 pipeline; per-task Null heuristic table
copied from task 3; backend matrix incl. AgentBackend file protocol with an example
request/response pair; spec 6b/6c guarantees — coverage always reported, guidance never gates,
veto is a flag; reproducibility = guidance log + MockBackend replay; `[llm]` extra install
line), `docs/status/phase-llm-analyst.md` (what shipped, test counts, deviations); **modify**
`README.md` (quickstart gains the `natex study` → `natex discover --plan` flow + roadmap tick),
`.claude/napkin.md` (new gotchas learned during the phase, per napkin curation rules).

**Final gates (run and paste outputs into the status doc):**
1. `cd /Users/haukehillebrandt/dev/natex && uv run ruff check src tests`
2. `cd /Users/haukehillebrandt/dev/natex && uv run pytest -q` — full suite green, no network,
   no API keys, `anthropic`/`google-genai` NOT installed (skips visible for the two smoke tests).
3. `uv sync --all-extras && uv run pytest -q tests/test_llm_api.py` — green WITH the extras
   installed (both smoke tests now run; still no network).
4. Backtest regression: `NATEX_DATA="/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data" uv run pytest -q -m backtest`
   — phases 2–5 rows still pass (this phase must not disturb them).
5. Manual smoke (document, don't automate): `uv run natex study <synthetic csv> --backend agent
   --out /tmp/natex-agent-demo` in one shell, answer the request file by hand, confirm the
   pipeline resumes.

**Commit:** `docs: llm-analyst method card, README study/discover flow, phase status`

---

## Deviations & boundaries (read before coding)

- **backends.py file split:** the brief nominates `natex/llm/backends.py` for all backends; to
  keep files reviewable, `AgentBackend` lives in `natex/llm/agent.py` and the API backends in
  `natex/llm/api.py`, ALL re-exported from `natex.llm` (and importable as documented). The
  protocol + models + Null + Mock stay in `backends.py`.
- **`prepare()` return type:** always a `Dataset` (with `spec.time`/`spec.unit` set for did
  candidates) — the repo's DiD entry points consume `Dataset`, and `CategoricalPanel` is built
  inside `discover()`; this satisfies "Dataset or PanelDataset ready for discovery".
- **The spec's sandboxed-pandas escape hatch (6a.3) is OUT of scope** — declarative plans only,
  documented in the method card as a deliberate v1 boundary.
- **`propose_forcing_variables` / `propose_candidate_events` mid-run hooks (6c rows 1–2) are
  satisfied by the Stage-0 SearchPlan** for this phase; the two extra task literals are NOT
  added to `TASKS` until a phase needs mid-run refinement. Documented in the method card.
- **API-backend live calls are never tested in CI**; correctness of the SDK call shape is pinned
  by the fake-client kwarg assertions (task 5) and a manual smoke run recorded in the status doc.
