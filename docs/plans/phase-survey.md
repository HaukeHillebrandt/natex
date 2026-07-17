# Phase survey implementation plan — one-command systematic design survey with visual report

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` sections 4 (API
contract), 6a–6c (analyst pass, targeted-first-exhaustive-still, guidance is advisory), 7
(reporting), 10 (non-engineer user).
**First action of task 1 is committing this plan file itself**
(`docs: phase survey implementation plan`).

## Phase objective

The top-layer UX: ONE command runs a dataset systematically against ALL natex method families and
outputs a single visual report, with an applicability verdict per family — including reasoned
SKIPs. Seven families, fixed order everywhere: **rdd, did, kink, iv, sc, bunching, dee**.

1. `src/natex/survey/registry.py` — MethodFamily registry: name, one-paragraph plain-language
   description (reused in the report), honest-inference caveat line, and REQUIREMENTS as
   declarative predicates over the intake profile / DatasetSpec / declared inputs ONLY — never
   over dataset content.
2. `src/natex/survey/applicability.py` — (a) `heuristic_applicability(profile, spec, declared)`
   from registry predicates alone; (b) new guidance task `method_applicability` (Literal +
   TASK_INSTRUCTIONS + NullBackend echo + schema-safe response models): the LLM can override
   heuristics BOTH ways, every override recorded (`heuristic_said` vs `analyst_said` + reason);
   LLM proposals feed config (cutoffs, instruments, thresholds, treated unit), never statistics.
3. `src/natex/survey/runner.py` — `survey(...) -> SurveyResult`: profile → study-style
   understanding → applicability → per-family execution REUSING existing modules; per-family
   try/except isolation; one rng, per-family sub-seeds via `rng.spawn`; `out_dir/survey.json`.
4. Figures for EVERY executed family via `natex.report.figures` (+ a new `bunching_hist`);
   families with no figure get an explicit `no figure: <reason>` entry.
5. `src/natex/report/survey_html.py` — self-contained `report.html` (base64 PNGs, badges,
   AI-generated banner, per-family sections) + always-working `report.md` text fallback.
6. CLI `natex survey CSV [--context] [--backend] [--out] [--seed] [--time] [--unit]
   [--cutoff COL=VALUE ...] [--instrument COL ...] [--threshold COL=VALUE ...] [--k --q
   --coarse --n-coarse --max-configs]` — prints the verdict table + report path.
7. `skills/natex-survey/SKILL.md`.
8. GENERALIZABILITY GUARD test: no dataset-specific tokens in survey code; applicability is
   provably content-blind (attribute-access-recording stub).
9. Tests: applicability units, three synthetic-shape e2e runs, failure isolation, HTML report,
   CLI determinism.
10. Docs: `docs/method_cards/survey.md` + README "One-command survey" near the top of quickstart.

## Global constraints (binding, from the phase-1 plan)

- Python >= 3.11; core deps stay exactly numpy/scipy/pandas/scikit-learn/typer/pydantic.
  jinja2 stays under `[report]`, matplotlib under `[plot]`. The survey degrades gracefully:
  no matplotlib → per-family `no figure: matplotlib not installed (pip install
  "natex-discovery[plot]")`; no jinja2 → `report.md` only plus a clear echoed message.
  CI (3.11–3.14, `--extra dev --extra plot --extra report`) must stay green; a core-only
  install must stay green too (extras tests `pytest.importorskip`).
- One `numpy.random.Generator` through every stochastic call. `survey()` raises ValueError
  without an explicit rng (mirror `study()`/`discover()`); per-family sub-generators come from
  ONE upfront `rng.spawn(7)` in registry order so a skipped family never shifts another
  family's stream.
- Discovery never reads the outcome (delegated to the reused modules; the survey layer adds NO
  new inference code — the only statistical change is a pure refactor exposing the existing
  density GLM, task 4).
- NaN never 0.0 on failure; JSON through `natex.jsonutil.jsonable`; report renders missing
  numbers as "—", never "nan"/"None" (napkin: paper-context string hygiene).
- No bare `except`. The ONE documented isolation boundary (task 5) is `except Exception` at the
  per-family runner call — named, commented, records the verbatim error and never catches
  BaseException (KeyboardInterrupt/SystemExit propagate).
- Guidance is advisory for statistics. `method_applicability` decides SCOPE (which families
  run) — that is config, like the search plan's ordering/budget — and every decision is
  recorded with the heuristic verdict beside it; a family the analyst turns off is
  `skipped` with the analyst's reason AND the recorded override, never silently absent.
- Schema-hint models must NOT use pydantic field constraints (napkin: `_strict_schema` strips
  them) and must NOT use dict-typed fields (`llm/api._make_strict` sets
  `additionalProperties: false` on every object node, which would destroy a
  `dict[str, Model]` field) — use lists of keyed submodels; range checks in `field_validator`s.
- Never commit datasets. Conventional commit after every green cycle. `uv run pytest -q`
  excludes backtests; this phase adds no backtests. Stochastic assertions: calibrate across
  >= 5 seeds during implementation, pin one with margin, record ranges in the test docstring.
- Verify each cycle with `cd /Users/haukehillebrandt/dev/natex && uv run ruff check src tests
  && uv run pytest -q` (subagent cwd resets between Bash calls; `uv run python`, never bare
  `python`).

## Current repo state (interfaces this phase builds on — verified 2026-07-17, HEAD 8ba547a)

- `natex.intake.profiler.profile(df) -> IntakeProfile` — `n_rows`, `columns:
  list[ColumnProfile(name, dtype, n_unique, missing_frac, is_numeric, is_binary, is_time_like,
  first_valid_index, prefix_missing_frac, is_monotone)]`, `panel_candidates: list[(unit,
  time)]`, `forcing_candidates`, `treatment_candidates`, `boundary_values`. `to_json()`.
- `natex.intake.analyst.study(csv_or_df, context, guidance, rng, out, strict) -> IntakeReport`
  (`profile`, `understanding: Understanding(shape, ...)`, `prep_plan`, `search_plan`,
  `guidance_errors`, `prepare(df=None, candidate=0) -> Dataset`); fixed task order
  understand → prepare → search_plan; NullBackend fallback recorded in `guidance_errors`;
  `out` given ⇒ `out/intake_report.json` + `out/guidance_log.jsonl` (`GuidanceLog` APPENDS).
- `natex.discover.discover(data, design="auto"|"rdd"|"did", guidance, search_plan, rng,
  budget, out) -> DiscoverReport(configs: list[ConfigRecord], searched, best_index,
  guidance_log_path)`. Budget keys: `max_configs, k, q, degree, coarse, n_coarse, bins,
  restarts, method, model, windows`; unknown explicit keys raise. rdd summary:
  `center_z, normal, forcing_influence, llr, p_value, placebo_passed, placebo_holm,
  density_p, coarse, effects{"2sls"|"wald": {tau, se, ci, first_stage_t, weak_instrument}}`;
  did summary: `subset_values, t0, window, llr, p_value, null_kind, composition_passed,
  anticipation_passed, searched_windows, restarts, effects{"dd"|"synthetic"|"gess":
  {tau, se, p, pre_mse, dose}}`. `report.save(out)` → `out/discover_report.json`.
- `natex.data.spec.Dataset(df, spec)` / `DatasetSpec(treatment, outcome, forcing, covariates,
  time, unit)`; forcing ⊆ covariates enforced; listwise deletion + row bookkeeping.
- `natex.llm.backends` — `TASKS` tuple + `Task` Literal + `TASK_INSTRUCTIONS` (6 tasks today),
  `GuidanceRequest(task, payload, schema_hint)`, `GuidanceResponse`, `MockBackend(responses)`
  (pops in order, records requests), `NullBackend` (payload-only deterministic handlers, one
  per task, NEVER vetoes), `LoggedBackend`, `GuidanceLog`. `llm/api._strict_schema` strips
  constraint keys and forces `additionalProperties: false` (see Global constraints).
- `natex.kink.regression_kink(y, running, *, treatment=None, policy_kink=None, cutoff=0.0,
  bandwidth, degree=1, kernel="triangular", donut=0.0, covariates=None, clusters=None,
  alpha=0.05) -> KinkEstimate(tau, se, ci, method, reduced_form, reduced_form_se,
  first_stage, first_stage_se, first_stage_F, weak_first_stage, n_used, n_by_cell,
  fieller_ci, fieller_kind, extras)`; `policy_kink=1.0` makes `tau` the reduced-form outcome
  kink. `natex.kink.sensitivity_grid` for bandwidth sweeps. Slope convention right-minus-left.
- `natex.iv.pipeline.discover_instruments(df, treatment, pool, outcome=None, controls=None,
  honest=True, frac_discovery=0.5, lam="plugin", rng) -> InstrumentDiscovery(search, estimate,
  honest, n_discovery, n_estimation, extras)`; empty selection ⇒ NaN estimate with
  `extras["reason"]`, never 0.
- `natex.iv.donors.unit_time_matrix(df, unit, time, outcome) -> (Y, units, times)`;
  `select_donors(Y, units, times, treated_unit, t0, n_donors=None, scoring="rmse") ->
  DonorSelectionResult(donors, scores, weights, y0_hat, times, pre_rmspe, post_rmspe,
  att_post, effect_by_time, extras)`; `sc_placebo_test(...) -> SCPlaceboReport(p_value
  (+1-rank, two-sided by ratio construction, NaN when n_used < 5), ratio_treated, ratios,
  placebo_units, n_skipped, extras)`. rng-free/deterministic.
- `natex.validate.density.density_test(dataset, d, n_bins=20) -> DensityReport(p_value,
  theta)` — binned-Poisson IRLS GLM on `signed_distance(dataset, d)`; audit-6 frozen-geometry
  caveat.
- `natex.dee.debias.dee_debias(dataset, query, discoveries, *, m_prime, k_prime=200,
  t_side=30, ..., rng) -> DEEResult` — runs on core deps (numpy GP); `natex[gp]` is the
  torch/gpytorch scale backend. The brief nevertheless gates the survey's dee family on the gp
  extra being installed; implement the gate as
  `importlib.util.find_spec("torch") and importlib.util.find_spec("gpytorch")` and document in
  the method card that the fitted GP itself is the core numpy one.
- `natex.report.figures` (plot extra, lazy `_mpl()`): `discovery_scatter(Z, llr, *,
  top_centers, top_normals, names, out_dir, stem)`, `density_hist(s, *, out_dir, cutoff=0.0,
  n_bins=20, p_value, stem)`, `pretrend_plot(times, gaps, t0, *, n, out_dir, stem)`,
  `effect_forest(labels, tau, lo, hi, *, pooled, out_dir, stem)`, `kink_fit_plot(running,
  outcome, cutoff, bandwidth, out_stem, *, estimate, ...)`. Each writes `<stem>.png` (150 dpi)
  + `.pdf`, returns `FigurePaths(png, pdf)`. Missing numbers render as em dash.
- `natex.report.paper._env()` — jinja2 lazily imported, `FileSystemLoader(report/templates)`,
  StrictUndefined; `_REPORT_EXTRA_MSG` names `natex-discovery[report]`.
- `natex.report.research_brief` — pure-Python markdown writer (the no-jinja2 pattern to copy).
- CLI (`src/natex/cli.py`, typer): `_make_backend(backend, model, workdir)` (null → None;
  missing `[llm]` exits 2), `_candidate_line`, `_clean = jsonable`; `study`/`discover`
  commands show the budget-flag conventions (`--k --q --coarse/--no-coarse --n-coarse
  --max-configs`, seed → `np.random.default_rng(seed)` once).
- `natex.did.effects.period_gaps(panel, discovery, control) -> PeriodGaps(times, gap, t0, n)`;
  `natex.did.panel.build_panel`; `natex.did.suddds.suddds_scan`, `resolve_default_model`;
  `natex.rdd.lord3.lord3_scan`; `natex.validate.placebo.signed_distance`.
- Tests to imitate: `tests/test_cli_study.py` (CliRunner + synthetic CSV helpers),
  `tests/test_report_figures.py` (importorskip matplotlib), `tests/test_skills.py`
  (frontmatter contract; append new dir to `SKILL_DIRS`), `tests/test_docs.py`
  (whitespace-normalized README asserts), `tests/test_llm_null.py` (NullBackend determinism).

## Audit corrections that bind this phase

| Audit item | Obligation here |
|---|---|
| 1 (+1-rank fitted-null MC, never exact) | every scan p in the report carries the caveat "fitted-null Monte Carlo p-value, not exact"; wording in registry caveat lines |
| 3 (placebo redefinition) | rdd family verdict `null` when placebo fails uses the "descriptive only — placebo battery failed" phrasing |
| 5 (two-sided studentized τ̂ / SC ratio test) | sc key numbers report the +1-rank RMSPE-ratio p verbatim from `sc_placebo_test`; never recomputed |
| 6 (density test valid only for frozen geometry) | density caveat kept for rdd; bunching at DECLARED thresholds states "declared threshold — not searched, so no selection correction is needed; binned-Poisson approximation" |
| 10 (first-stage relevance not implied) | iv family surfaces `weak` flags from both halves; weak ⇒ verdict `null` with reason |
| 18 (calendar-time McCrary information-free; composition/anticipation instead) | bunching on a time-like column attaches the audit-18 caveat; kink on a time-like running variable attaches the calendar-time RKD caveat (docs/method_cards/kink.md, e077462) |
| 19 (dose normalization; model matching) | did effects rows show `dose`; no new estimand math |
| NaN policy (item 8 / spec 5.8) | failed families: `status="failed"`, key_numbers absent — never fabricated zeros |

Statuses are fixed: per-family `credible | null | skipped | needs_input | failed`; heuristic
applicability statuses `applicable | inapplicable | needs_input`.

---

## Task 1 — plan file + survey package + method-family registry

**First action: commit this plan file** (`docs: phase survey implementation plan`).

**Files:** `src/natex/survey/__init__.py` (new), `src/natex/survey/registry.py` (new),
`tests/test_survey_registry.py` (new).

**Interfaces (exact):**

```python
# src/natex/survey/registry.py
from dataclasses import dataclass, field
from typing import Callable
from natex.data.spec import DatasetSpec
from natex.intake.profiler import IntakeProfile

@dataclass(frozen=True)
class DeclaredInputs:
    """User/analyst-declared survey inputs; the ONLY non-profile evidence predicates may read."""
    time: str | None = None
    unit: str | None = None
    cutoffs: dict[str, float] = field(default_factory=dict)      # col -> cutoff value
    instruments: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)   # col -> threshold value
    treated_unit: str | None = None                              # sc hint (guidance/config only)
    t0: float | None = None                                      # sc hint

@dataclass(frozen=True)
class Requirement:
    key: str          # e.g. "needs_numeric_forcing"
    description: str  # human sentence, used verbatim in reasons
    user_suppliable: bool  # unmet+True -> needs_input; unmet+False -> inapplicable
    check: Callable[[IntakeProfile, DatasetSpec | None, DeclaredInputs], bool]

@dataclass(frozen=True)
class MethodFamily:
    name: str    # "rdd"|"did"|"kink"|"iv"|"sc"|"bunching"|"dee"
    title: str   # e.g. "Regression discontinuity (LoRD3 scan)"
    description: str  # ONE plain-language paragraph, reused verbatim in the report
    caveat: str       # honest-inference caveat line (phrasing pulled from the method cards)
    requirements: tuple[Requirement, ...]

FAMILY_ORDER: tuple[str, ...] = ("rdd", "did", "kink", "iv", "sc", "bunching", "dee")
FAMILIES: dict[str, MethodFamily]  # insertion order == FAMILY_ORDER
```

Named predicate helpers (module level, so the guard test can exercise them one by one):
`_has_treatment(p, s, d)` (profile.treatment_candidates nonempty OR spec is not None),
`_has_numeric_forcing` (profile.forcing_candidates nonempty OR (spec and spec.forcing)),
`_has_panel` (profile.panel_candidates nonempty OR (d.unit and d.time) OR (spec and spec.time
and spec.unit)), `_has_time` (any ColumnProfile.is_time_like OR d.time), `_has_outcome`
(any column is_numeric and not is_binary and not is_time_like, OR spec.outcome),
`_has_declared_cutoff` (d.cutoffs), `_has_declared_instruments` (d.instruments),
`_has_declared_threshold` (d.thresholds), `_min_rows(n)` (closure; profile.n_rows >= n),
`_gp_extra_installed(p, s, d)` (importlib.util.find_spec for torch AND gpytorch — an
environment predicate, still content-blind). Every predicate reads ONLY
`IntakeProfile`/`DatasetSpec`/`DeclaredInputs` attributes — no DataFrame anywhere in this
module's signatures.

Requirements per family (min_rows values are conservative floors, documented in the card):
- **rdd**: min_rows(100), needs_binary_treatment, needs_numeric_forcing.
- **did**: min_rows(60), needs_binary_treatment, needs_panel(unit,time).
- **kink**: min_rows(60), needs_numeric_forcing, needs_declared_cutoff (user_suppliable=True;
  canonical reason "no pre-declared cutoff (kink is candidate evaluation, not discovery)").
- **iv**: min_rows(80), needs_binary_or_continuous_treatment (= _has_treatment),
  needs_candidate_instruments (user_suppliable=True).
- **sc**: min_rows(40), needs_panel(unit,time), needs_outcome.
- **bunching**: min_rows(60), needs_declared_threshold (user_suppliable=True).
- **dee**: min_rows(200), needs_binary_treatment, needs_numeric_forcing, needs_outcome,
  needs_gp_extra (user_suppliable=False; reason names `pip install "natex-discovery[gp]"`).

Descriptions: one honest plain-language paragraph each (what the design is, what the family
does in a survey run, what "credible" means for it). Caveats (verbatim sources):
rdd → audit-1 fitted-null MC + audit-6 frozen-geometry density; did → composition/anticipation
+ descriptive-when-placebo-fails; kink → conventional local-polynomial inference may retain
smoothing bias + calendar-time warning (kink card); iv → exclusion is untestable, honest split
is the guarantee (iv_sc card); sc → +1-rank in-space placebo granularity 1/(n_used+1);
bunching → declared thresholds only, binned-Poisson approximation; dee → debiasing quality
depends on discovered-experiment coverage, not a hypothesis test.

**Tests (`tests/test_survey_registry.py`, failing first):**
- `test_family_order_and_keys`: `list(FAMILIES) == list(FAMILY_ORDER)`; exactly 7 entries;
  every family has nonempty title/description/caveat; descriptions >= 200 chars (a real
  paragraph); no description contains "TODO".
- `test_requirement_keys_are_declarative`: every requirement key matches
  `^needs_[a-z0-9_]+$|^min_rows$`; every `check` accepts `(profile, None, DeclaredInputs())`
  built from a 3-column dummy profile without raising.
- `test_predicates_profile_only`: call each family's checks with a
  `_RecordingProfile` stub (implemented in this test file; `__getattr__` records the attribute
  name, returns realistic values: n_rows=500, columns=[...], panel_candidates=[], etc.) and
  assert the union of accessed attributes ⊆
  `{"n_rows", "columns", "panel_candidates", "forcing_candidates", "treatment_candidates"}` —
  the first half of the generalizability guard, in place from day one.
- `test_gp_predicate_env_only(monkeypatch)`: monkeypatch
  `importlib.util.find_spec` → None ⇒ dee's needs_gp_extra check returns False; → truthy ⇒
  True.
- `test_no_bunching_hyphenation_traps`: descriptions/caveats never contain the substrings
  "None" or case-insensitive standalone "nan" (napkin string-hygiene rule; regex
  `\bnan\b|\bNone\b` on the concatenated registry text).

**Cycle:** ruff + full pytest. **Commit:** `feat(survey): method-family registry with
declarative applicability requirements` (plan file committed first, separately).

## Task 2 — heuristic applicability

**Files:** `src/natex/survey/applicability.py` (new), `tests/test_survey_applicability.py`
(new).

**Interfaces (exact):**

```python
@dataclass
class FamilyVerdict:
    family: str
    status: str          # "applicable" | "inapplicable" | "needs_input"
    reason: str          # met: "all requirements met"; unmet: joined requirement descriptions
    unmet: list[str]     # unmet requirement keys, registry order

def heuristic_applicability(
    profile: IntakeProfile,
    spec: DatasetSpec | None,
    declared: DeclaredInputs,
) -> dict[str, FamilyVerdict]:   # ALWAYS all 7 keys, FAMILY_ORDER order
```

Status rule: all requirements met → `applicable`. Otherwise: if EVERY unmet requirement is
`user_suppliable` → `needs_input`, else `inapplicable`. Reason = "; ".join of unmet
requirement descriptions (deterministic, registry order); the kink no-cutoff and iv
no-instruments canonical strings come through the requirement descriptions verbatim.
Pure function: no rng, no I/O, no DataFrame parameter — CONTENT-BLIND by construction.

**Tests (failing first; use real `profile(df)` on small constructed frames):**
- `test_cross_section_skips_did_and_kink_in_time`: 300-row cross-section (binary `T`, numeric
  `z`, numeric `y`, no time-like or id column, no declared inputs) ⇒ rdd `applicable`; did
  `inapplicable` with reason mentioning panel/unit-time; kink `needs_input` with the exact
  substring "no pre-declared cutoff (kink is candidate evaluation, not discovery)"; sc
  `inapplicable`; bunching `needs_input`; iv `needs_input`.
- `test_panel_enables_did_and_sc`: units×years grid (string `state` col, int `year` col
  1990–2005, binary `T`, numeric `y`) ⇒ did and sc lose the panel objection (did
  `applicable`; sc `applicable`).
- `test_declared_cutoff_enables_kink`: same cross-section +
  `DeclaredInputs(cutoffs={"z": 1.5})` ⇒ kink `applicable`.
- `test_declared_instruments_enable_iv`; `test_thresholds_enable_bunching`.
- `test_min_rows_gate`: 20-row frame ⇒ every family `inapplicable` and each reason names the
  row floor.
- `test_dee_needs_gp_and_outcome(monkeypatch)`: find_spec→None ⇒ dee `inapplicable` with
  reason containing `natex-discovery[gp]`.
- `test_all_seven_always_present`: any input ⇒ exactly `FAMILY_ORDER` keys in order.

**Commit:** `feat(survey): heuristic per-family applicability from registry predicates`

## Task 3 — `method_applicability` guidance task with recorded overrides

**Files:** `src/natex/llm/backends.py` (modify: TASKS, `Task` Literal, TASK_INSTRUCTIONS,
NullBackend handler), `src/natex/survey/applicability.py` (extend),
`tests/test_survey_guidance.py` (new), `tests/test_llm_null.py` (extend).

**Interfaces (exact):**

```python
# schema-safe response models (NO dict fields, NO field constraints — see Global constraints)
class ConfigValueHint(BaseModel):   # cutoffs and thresholds share this shape
    column: str
    value: float

class ConfigHints(BaseModel):
    cutoffs: list[ConfigValueHint] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    thresholds: list[ConfigValueHint] = Field(default_factory=list)
    treated_unit: str | None = None
    t0: float | None = None

class FamilyDecision(BaseModel):
    family: str
    run: bool
    reason: str = ""
    config_hints: ConfigHints = Field(default_factory=ConfigHints)

class ApplicabilityResponse(BaseModel):
    families: list[FamilyDecision]

@dataclass
class FamilyPlan:
    family: str
    run: bool
    reason: str
    heuristic: FamilyVerdict
    config_hints: ConfigHints
    override: dict | None    # {"heuristic_said", "analyst_said", "reason"} when run differs
    guidance_error: str | None  # backend/parse failure -> fell back to heuristics

def resolve_applicability(
    profile: IntakeProfile,
    spec: DatasetSpec | None,
    declared: DeclaredInputs,
    guidance: GuidanceBackend | None,
    *,
    context: str | None = None,
) -> tuple[dict[str, FamilyPlan], DeclaredInputs]:
    """Heuristics -> optional one-shot method_applicability request -> merged declared inputs.

    guidance=None: no request fires; run = (status == "applicable"); overrides all None.
    Backend given: ONE GuidanceRequest(task="method_applicability",
      payload={"profile": <profile dict>, "context": context,
               "declared": <declared as jsonable dict>,
               "families": [{name, title, description,
                             requirements: [{key, description, met}]} ...],
               "heuristics": {name: {"status", "reason", "unmet"}}},
      schema_hint=ApplicabilityResponse.model_json_schema()).
    Parse failures / _FALLBACK_EXCEPTIONS (ValidationError, ValueError, TimeoutError,
    RuntimeError — same tuple as intake.analyst) fall back to heuristics per family with
    guidance_error recorded. Unknown family names in the reply are dropped (recorded);
    families the reply omits keep their heuristic decision.
    Hint hygiene: cutoff/threshold/instrument columns must exist in the profile and be
    numeric; treated_unit must be a string; invalid hints are DROPPED and recorded in the
    family's guidance_error — never a crash, and hints never overwrite an explicitly
    declared value (CLI/user declarations win). Returns the merged DeclaredInputs used by
    the runner. Hints feed config ONLY — no statistic is touched here.
    """
```

`llm/backends.py` changes: append `"method_applicability"` to `TASKS` and the `Task` Literal;
`TASK_INSTRUCTIONS["method_applicability"]` = short paragraph: "You are shown a dataset
profile, user context, declared inputs, per-family method descriptors and natex's heuristic
applicability verdicts. For each family decide run true/false with a reason a non-statistician
can follow, and optionally propose config hints (kink cutoffs, candidate instrument columns,
bunching thresholds, a synthetic-control treated unit and t0) grounded in the context. You may
override the heuristics in either direction; overrides are recorded, and your hints feed
configuration only — never statistics." NullBackend handler `_method_applicability(payload)`:
pure echo of `payload["heuristics"]` — `{"families": [{"family": name, "run": h["status"] ==
"applicable", "reason": h["reason"], "config_hints": {...empty...}} for name, h in ...]}`,
deterministic, payload-only, empty hints (NullBackend proposes nothing).

**Tests:**
- `tests/test_llm_null.py::test_method_applicability_echoes_heuristics`: NullBackend on a
  canned payload returns run flags equal to `status=="applicable"`, bitwise-stable
  `raw_text`, empty hints.
- `test_survey_guidance.py::test_null_guidance_matches_no_guidance`: `resolve_applicability`
  with `guidance=None` and with `NullBackend()` produce identical `{name: (run, reason)}`
  maps and all-None overrides.
- `test_mock_override_both_ways`: MockBackend reply flips kink (heuristic needs_input →
  run=True with `config_hints.cutoffs=[{"column": "z", "value": 1.5}]`) and rdd (heuristic
  applicable → run=False, reason "context says treatment was assigned alphabetically") ⇒
  both `FamilyPlan.override` blocks populated with exact heuristic_said/analyst_said/reason;
  merged DeclaredInputs gains `cutoffs["z"]==1.5`.
- `test_hint_hygiene`: hints naming a nonexistent column and a non-numeric column are
  dropped + recorded; an explicit declared cutoff for the same column is NOT overwritten.
- `test_backend_failure_falls_back`: MockBackend raising RuntimeError ⇒ heuristic decisions,
  `guidance_error` recorded on every family.
- `test_schema_hint_is_strict_safe`: `_strict_schema(ApplicabilityResponse.model_json_schema())`
  contains no `_STRIP_KEYS` members and every `"type": "object"` node has
  `additionalProperties: false` without losing the families item schema (regression against
  the dict-field trap).
- `test_task_registered`: `"method_applicability" in TASKS`, in TASK_INSTRUCTIONS, and
  `GuidanceRequest(task="method_applicability", payload={})` validates.

**Commit:** `feat(survey): method_applicability guidance task with recorded heuristic
overrides`

## Task 4 — bunching statistic (pure refactor) + `bunching_hist` figure

**Files:** `src/natex/validate/density.py` (modify), `src/natex/report/figures.py` (modify),
`tests/test_validation.py` (extend), `tests/test_report_figures.py` (extend).

**Interfaces:**

```python
# validate/density.py — extract the existing IRLS Poisson GLM verbatim; NO new math
def binned_poisson_jump(s: np.ndarray, n_bins: int = 20) -> DensityReport:
    """Binned-Poisson intercept-jump test on signed distances s (cutoff at 0).

    Extracted from density_test so declared-threshold bunching reuses the identical
    statistic; density_test now delegates: density_test(ds, d, n_bins) ==
    binned_poisson_jump(signed_distance(ds, d), n_bins). Non-finite s dropped; s with
    < 2 distinct finite values -> DensityReport(nan, nan) (NaN, never 0).
    """

# report/figures.py
def bunching_hist(values, threshold: float, *, out_dir, n_bins: int = 20, p_value=None,
                  name: str | None = None, stem: str = "bunching_hist") -> FigurePaths:
    """Raw-value histogram split at a DECLARED threshold (no audit-6 search caveat;
    annotation says 'declared threshold — not searched'). Mirrors density_hist styling."""
```

**Tests:**
- `test_validation.py::test_binned_poisson_jump_matches_density_test`: on a seeded Dataset +
  discovery, `density_test(ds, d).p_value == binned_poisson_jump(signed_distance(ds, d)).p_value`
  bitwise (pure-refactor proof) and same theta.
- `test_binned_poisson_jump_detects_gap`: rng(0) sample, 2000 draws uniform on [-1,1] with the
  mass on [0, 0.1] tripled ⇒ p < 0.01; symmetric null sample ⇒ p > 0.05 (calibrate across
  seeds 0–4, pin one, record ranges in the docstring).
- `test_binned_poisson_jump_degenerate`: constant input ⇒ NaN p, NaN theta.
- `test_report_figures.py::test_bunching_hist` (importorskip matplotlib): writes png+pdf,
  png nonempty; title contains the threshold; no "nan" in axis text when p_value=None.

**Commit:** `refactor(validate): expose binned_poisson_jump for declared-threshold bunching +
bunching_hist figure`

## Task 5 — runner skeleton: SurveyResult, rdd/did families via discover, failure isolation

**Files:** `src/natex/survey/runner.py` (new), `src/natex/survey/__init__.py` (export
`survey`, `SurveyResult`, `FamilyResult`), `src/natex/__init__.py` (add `survey` to imports +
`__all__`, mirroring the `discover` module/function precedent), `tests/test_survey_runner.py`
(new).

**Interfaces (exact — later tasks and the report depend on these fields):**

```python
ALPHA = 0.05            # verdict gate for scan/kink/bunching p-values
SC_ALPHA = 0.10         # sc placebo gate (coarse +1-rank granularity, documented)

@dataclass
class FamilyResult:
    family: str
    status: str                  # credible|null|skipped|needs_input|failed
    reason: str                  # one sentence, always set
    applicability: dict          # FamilyPlan serialized: heuristic + analyst + override
    key_numbers: dict            # flat name->number (jsonable at save; NaN -> null)
    diagnostics: dict            # per-family extras incl. caveat strings actually attached
    figures: dict[str, str]      # figure name -> path RELATIVE to out_dir (posix)
    no_figure_reason: str | None # set whenever figures == {}
    details_path: str | None     # families/<name>.json relative to out_dir
    error: str | None            # verbatim str(exc) when status == "failed"

@dataclass
class SurveyResult:
    out_dir: Path
    families: dict[str, FamilyResult]   # ALWAYS all seven, FAMILY_ORDER order
    coverage: dict     # {"ran": [...], "not_run": {name: reason},
                       #  "rdd": <discover.searched or None>, "did": <...>}
    dataset: dict      # {"source", "n_rows", "n_cols", "columns_truncated": <=20 names,
                       #  "time_column", "time_range": [min, max] | None}
    natex_version: str
    seed: int | None
    created: str       # UTC isoformat — the ONLY nondeterministic survey.json field
    context: str | None
    guidance_log_path: str | None
    report_html: str | None      # set by task 8
    report_md: str | None
    def to_json(self) -> str                     # json.dumps(jsonable(...), indent=1)
    def save(self) -> Path                       # out_dir/survey.json
    @classmethod
    def load(cls, path) -> "SurveyResult"

def survey(
    csv_or_df, *,
    context: str | None = None,
    guidance: GuidanceBackend | None = None,
    rng: np.random.Generator,          # ValueError when None (house contract)
    out_dir: str | Path,
    budget: dict | None = None,        # validated against discover's budget keys
    time: str | None = None,
    unit: str | None = None,
    cutoffs: dict[str, float] | None = None,
    instruments: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    seed: int | None = None,           # metadata only; rng governs randomness
) -> SurveyResult
```

Flow (document verbatim in the module docstring):
1. Load df (csv path or DataFrame), `declared = DeclaredInputs(...)` from the kwargs.
2. `fam_rngs = dict(zip(FAMILY_ORDER, rng.spawn(7)))` FIRST (stream stability), then
   `intake = study(csv_or_df, context=context, guidance=guidance, rng=rng,
   out=out_dir/"intake")` (study-style understanding; NullBackend heuristics when
   guidance=None; its guidance_log.jsonl is the survey's log).
3. `plans, declared = resolve_applicability(intake.profile, None, declared, wrapped_guidance,
   context=context)` where wrapped_guidance appends to the SAME
   `out_dir/"intake"/guidance_log.jsonl` via `LoggedBackend` (GuidanceLog appends);
   `guidance_log_path` = that file when guidance is not None else None.
4. For each family in FAMILY_ORDER: not run ⇒ FamilyResult(status = "skipped" if analyst said
   no or heuristic inapplicable, "needs_input" if heuristic needs_input and no override,
   reason from the plan). Run ⇒ call `_run_<family>(...)` inside the isolation boundary:

```python
try:
    result = runner_fn(df, intake, declared, plans[name], eff_budget,
                       fam_rngs[name], fam_dir)
except Exception as exc:  # noqa: BLE001 — DOCUMENTED isolation boundary (plan task 5):
    # a family failure must never abort the survey; BaseException still propagates.
    result = FamilyResult(family=name, status="failed", reason="family raised",
                          error=str(exc), ...)  # traceback.format_exc() -> diagnostics
```

5. Write per-family `out_dir/families/<name>.json` (jsonable full detail), figures under
   `out_dir/figures/` (task 7), `survey.json`, then reports (task 8).

**rdd family (`_run_rdd`)** — REUSE `natex.discover`: build the Dataset via
`intake.prepare(candidate=i)` for the first ranked rdd candidate; when the search plan has no
rdd candidate, construct `DatasetSpec(treatment=profile.treatment_candidates[0],
outcome=<first Understanding outcome guess != treatment, else None>,
forcing=profile.forcing_candidates minus {treatment, outcome}, covariates=all minus
outcome-role columns)` directly (mirrors `Dataset.from_csv` defaults). Then
`rep = discover(ds, design="rdd", search_plan=intake.search_plan, rng=fam_rng,
budget=eff_budget, out=fam_dir)`. Verdict from `rep.best()`: no scanned config ⇒ status
`failed` (first config error verbatim) unless every config was invalid ⇒ `null` with reason;
best scanned: `p_value <= ALPHA and placebo_passed and density_p > ALPHA` ⇒ `credible`; else
`null` with the failing gate named ("scan p=0.21 above 0.05" / "descriptive only — placebo
battery failed" / "density test rejects (p=...) — manipulation risk"). key_numbers: llr,
p_value, density_p, placebo_holm, n_configs_scanned, and 2sls tau/se/ci when present.
coverage["rdd"] = rep.searched. details = fam_dir/"discover_report.json".

**did family (`_run_did`)** — same via `design="did"`; dataset from the first ranked did
candidate, else constructed spec with time from `declared.time or profile.panel_candidates[0][1]`,
unit likewise (unit column may be non-numeric: keep it out of forcing; DatasetSpec allows
that). Verdict: `p<=ALPHA and composition_passed and anticipation_passed` ⇒ credible; else
null naming the gate. key_numbers: llr, p_value, t0, window, dd/synthetic/gess tau+se+p+dose.

Budget: `_effective_budget = {"k":50,"q":99,...}` — reuse discover's validation by simply
passing the user dict through `discover(budget=...)`; survey-level default overrides only
`q` when the caller passed none? NO — keep defaults identical to discover's; tests pass small
budgets explicitly (`{"q": 9, "k": 25}`) for speed.

**Tests (failing first; all seeded, budget q<=9):**
- `test_survey_requires_rng`: ValueError without rng.
- `test_rdd_shape_end_to_end(tmp_path)`: `make_synthetic`-style CSV (reuse the
  `tests/test_cli_study.py::_write_synthetic_csv` recipe, n=300) ⇒ SurveyResult has ALL
  SEVEN families; rdd status in {"credible","null"} with key_numbers["p_value"] finite; did
  skipped/needs_input-or-inapplicable with a nonempty reason; kink needs_input with the
  canonical no-cutoff string; `out/survey.json` exists and round-trips through
  `SurveyResult.load` with identical families dict.
- `test_plain_cross_section_lists_all_seven(tmp_path)`: 200 rows of pure rng normals
  (x0..x3), no binary column ⇒ every family is skipped/needs_input/inapplicable-mapped —
  none "failed" — each with a nonempty reason; survey.json still written; coverage["not_run"]
  has 7 entries.
- `test_failure_isolation(monkeypatch)`: monkeypatch `natex.survey.runner._run_rdd` to raise
  `RuntimeError("boom-xyzzy")` ⇒ rdd status "failed", `error == "boom-xyzzy"` verbatim,
  survey completes, other families untouched, survey.json written.
- `test_spawn_stability(tmp_path)`: two runs same seed, one with bunching thresholds declared
  and one without ⇒ rdd key_numbers identical (skipping a family never shifts another's
  stream).
- `test_status_vocabulary`: every family status ∈ the 5-value set.

**Commit:** `feat(survey): survey runner skeleton with rdd/did families and per-family
failure isolation`

## Task 6 — kink, iv, sc, bunching, dee family runners

**Files:** `src/natex/survey/runner.py` (extend), `tests/test_survey_runner.py` (extend),
`tests/test_survey_families.py` (new).

**kink (`_run_kink`)** — for each `(col, c)` in merged `declared.cutoffs` (skip cols missing/
non-numeric with a recorded per-cutoff reason): running = df[col] floats; outcome = first
Understanding outcome guess not in {col} (none ⇒ family `needs_input`, reason "no numeric
outcome column identified"). Default bandwidth (documented, NO optimality claim):
`bw = float(np.nanquantile(np.abs(r - c), 0.5))` (median absolute distance — about half the
sample in-window); ValueError if not finite/positive ⇒ that cutoff recorded failed.
`est = regression_kink(y, r, policy_kink=1.0, cutoff=c, bandwidth=bw)` — with policy_kink=1
`tau` IS the reduced-form outcome slope kink (right-minus-left, kink-card convention).
Per-cutoff p = `2*stats.norm.sf(abs(tau/se))` (NaN-safe). Family verdict: Holm over the m
cutoff p-values (reuse the same Holm convention as placebo_tests — no new math, plain
step-down on the vector), min adjusted p <= ALPHA ⇒ credible, else null; all NaN ⇒ null
"kink fits degenerate". diagnostics: per-cutoff `sensitivity_grid` over bandwidths
{0.5, 1, 2}×bw (tau/se rows only), calendar-time caveat attached when the running column
`is_time_like` in the profile. key_numbers: per first cutoff tau/se/ci/bandwidth/n_used +
`min_holm_p`.

**iv (`_run_iv`)** — pool = merged `declared.instruments` (existing+numeric enforced by
resolve_applicability hygiene); treatment = spec/understanding treatment candidate; outcome
= first outcome guess (may be None ⇒ selection only). `res = discover_instruments(df,
treatment, pool, outcome=outcome, honest=True, rng=fam_rng)`. Verdict: selection empty ⇒
null "no instrument selected from the declared pool"; `search.weak or (est and
est.weak_instrument)` ⇒ null naming the weak first stage (audit 10); else credible.
key_numbers: n_selected, first_stage_F, partial_r2, tau/se/ci, ar_ci, ar_kind, j_p,
n_discovery/n_estimation. diagnostics carries the honest-split statement + exclusion caveat.

**sc (`_run_sc`)** — unit/time from declared else first profile panel candidate; outcome as
above (required). `Y, units, times = unit_time_matrix(df, unit, time, outcome)`. Treated
unit: `declared.treated_unit`/`t0` (CLI has no flag — these arrive via guidance config_hints)
else derive from the binary treatment column: per-unit ever-treated set; exactly one
ever-treated unit ⇒ treated, `t0 = min(time where T==1)`; zero or >1 ⇒ family `needs_input`
with reason "could not identify a single treated unit from '<T>' (found k); provide
treated_unit/t0 via guidance". Then `sel = select_donors(Y, units, times, treated, t0)`;
`rep = sc_placebo_test(Y, units, times, treated, t0)`. Verdict: `sel.extras["failure"]` ⇒
failed(reason verbatim); `rep.p_value` NaN ⇒ null "too few usable placebos (<5) for the
ratio test"; `p <= SC_ALPHA` ⇒ credible else null. key_numbers: att_post, pre_rmspe,
post_rmspe, ratio_treated, p_value, n_donors, n_placebos.

**bunching (`_run_bunching`)** — for each `(col, thr)` in merged thresholds:
`s = df[col].to_numpy(float) - thr` (finite only); `DensityReport =
binned_poisson_jump(s)`. Holm across thresholds; min adjusted p <= ALPHA ⇒ credible
("density discontinuity at a declared threshold — bunching/manipulation signal"), else null.
Column time-like ⇒ attach the audit-18 calendar-time caveat in diagnostics. key_numbers per
first threshold: p, theta, n_finite + `min_holm_p`.

**dee (`_run_dee`)** — runtime gates ON TOP of applicability: requires the rdd FamilyResult
status == "credible" (else skipped, reason "no validated rdd discovery to debias") — the gp
extra requirement already lives in applicability. Rebuild the rdd family's Dataset, run
`scan = lord3_scan(ds, k=eff k, degree=1, rng=fam_rng)`; query lattice: `grid=8` points per
forcing dim over observed ranges (CLI debias recipe, smaller default);
`res = dee_debias(ds, query, scan, m_prime=min(10, len(scan.discoveries)), rng=fam_rng)`.
Verdict: `"reason" in res.diagnostics` (degenerate) ⇒ null with that reason; else credible
with reason "debiased CATE surface fitted over N experiments" (documented status-semantics
stretch: dee is a surface fit, not a test — the card explains). key_numbers: w_debias,
n_experiments, n_experiments_used, mean raw/debiased/direct CATE (NaN-safe nanmean).

**Tests (`tests/test_survey_families.py`; every stochastic gate calibrated over >= 5 seeds
during implementation, one pinned, ranges recorded in docstrings):**
- `test_panel_shape_end_to_end(tmp_path)`: constructed panel — 12 string units × 16 int
  years, binary T switching on for exactly ONE unit at year 10 with an effect on y (reuse
  the tests/test_donors.py DGP idea, noise 0.5, effect 10) ⇒ did ran (status credible|null,
  finite p), sc ran and key_numbers["p_value"] == 1/(n_used+1)-granular finite value,
  treated unit auto-derived; rdd inapplicable-or-run per profile; all seven present.
- `test_kink_declared_cutoff(tmp_path)`: piecewise-linear DGP y = 1 + 0.5·min(z,2) +
  2.5·max(z−2,0) + rng.normal(0,0.3), n=800, `cutoffs={"z": 2.0}` ⇒ kink credible;
  key_numbers tau within [1.0, 3.0] of the true slope change 2.0 at the pinned seed;
  a no-kink DGP ⇒ null. Sensitivity block present with 3 bandwidth rows.
- `test_iv_declared_instruments(tmp_path)`: `make_iv_synthetic`-based frame (mu2=180, seed
  from the calibrated strong-seed base — napkin item: base 100) with pool columns declared ⇒
  iv credible, first_stage_F > 10; a pi=0 pure-noise pool (mu2=0) ⇒ null with "no instrument
  selected" reason.
- `test_bunching_declared_threshold(tmp_path)`: tripled-mass DGP from task 4 as a column ⇒
  credible; uniform column ⇒ null; threshold on a `year` column ⇒ audit-18 caveat string in
  diagnostics.
- `test_sc_needs_input_when_ambiguous`: two ever-treated units ⇒ sc needs_input, reason
  contains "found 2".
- `test_dee_gate(monkeypatch, tmp_path)`: find_spec→None ⇒ dee never runs (applicability
  inapplicable, reason names the extra); with find_spec real and torch actually absent the
  same skip holds. A `@pytest.mark.skipif(find_spec("torch") is None)` variant on the rdd
  synthetic asserts dee reaches "credible|null" when rdd is credible; plus a cheap
  always-running unit: rdd forced to "null" (high noise) ⇒ dee skipped with "no validated
  rdd discovery to debias".
- `test_guidance_hint_flows_to_config(tmp_path)`: MockBackend proposing a kink cutoff via
  config_hints ⇒ kink runs at that cutoff and the applicability block shows the override.

**Commit:** `feat(survey): kink, iv, sc, bunching and dee family runners`

## Task 7 — figures for every executed family

**Files:** `src/natex/survey/figures.py` (new; glue only — rendering stays in
`natex.report.figures`), `src/natex/survey/runner.py` (wire in), `tests/test_survey_figures.py`
(new).

**Interface:** `render_family_figures(name, artifacts, out_dir) -> tuple[dict[str, str],
str | None]` — returns (figures rel-path map, no_figure_reason). `artifacts` is a small
per-family payload of ARRAYS/objects the runner collected while it had live objects (never
re-reads the CSV): matplotlib importability probed ONCE per survey via
`importlib.util.find_spec("matplotlib")`; absent ⇒ every executed family gets
`no figure: matplotlib not installed (pip install "natex-discovery[plot]")`.

Per family (stems prefixed `<family>_` under `out_dir/figures/`):
- **rdd**: after `discover`, ONE presentational re-scan of the best config's dataset
  (`lord3_scan(ds, k, degree, rng=fam_rng)` — the randomization test is NOT re-run; cost ~
  1/(q+1) of the family; documented) feeding `discovery_scatter` (Z at scored centers, llr,
  top 5 centers/normals, forcing names) + `density_hist(signed_distance(ds, top),
  p_value=summary["density_p"])` + `effect_forest` (2sls/wald rows from the summary, ivw
  pooled row as in `rdd_figures`).
- **did**: rebuild `build_panel` + re-run `suddds_scan` on the best config (same budget
  params, fam_rng) → `period_gaps` → `pretrend_plot`; `effect_forest` from dd/synthetic/gess
  (ci = tau ± 1.96·se; NO pooled row — same rationale as `did_figures`).
- **kink**: `kink_fit_plot` per cutoff (max 3), estimate annotation from the family's
  KinkEstimate.
- **iv**: `effect_forest` with the 2SLS row and, when `ar_ci` is finite, an "AR (weak-IV
  robust)" row; selection-only runs (no outcome) ⇒ `no figure: selection-only run (no
  outcome column)`.
- **sc**: `pretrend_plot(times, effect_by_time, t0)` labeled "treated − synthetic gap".
- **bunching**: `bunching_hist` per threshold (max 3), p annotated.
- **dee**: `effect_forest` of per-experiment local-2SLS taus (labels `exp 0..`, ci = tau ±
  1.96·se, NaN rows point-only) — capped at 15 rows.
Skipped/needs_input/failed families: `no figure: family did not run (<reason>)`.

**Tests** (importorskip matplotlib for the rendering half):
- `test_executed_families_have_figures(tmp_path)`: rdd synthetic run ⇒ rdd figures dict
  nonempty, each path exists under out_dir/figures and ends .png; skipped families have
  `no_figure_reason` starting "no figure:".
- `test_no_matplotlib_reasons(monkeypatch)`: monkeypatch find_spec("matplotlib")→None ⇒
  executed family gets the exact extra-install reason and survey still completes.
- `test_kink_and_bunching_figures(tmp_path)`: declared-cutoff run produces
  `kink_fit_plot`/`bunching_hist` files.
- `test_figure_failure_is_isolated(monkeypatch)`: a figure function raising ValueError ⇒
  family keeps its statistical status; `no_figure_reason` records "figure rendering failed:
  ..." (figures are presentation — never change a verdict).

**Commit:** `feat(survey): per-family figures with graceful no-matplotlib degradation`

## Task 8 — report.md (always) + report.html (report extra, self-contained)

**Files:** `src/natex/report/survey_html.py` (new),
`src/natex/report/templates/survey.html.j2` (new), `src/natex/survey/runner.py` (wire in),
`tests/test_survey_report.py` (new).

**Interfaces:**

```python
# both consume the JSON-NATIVE dict (survey.json content), so `natex survey` and any
# re-render from a saved survey.json share one path
def render_survey_md(result: dict, out_dir: str | Path) -> Path    # pure Python, no jinja2
def render_survey_html(result: dict, out_dir: str | Path) -> Path  # jinja2 lazy, ImportError
                                                                   # names natex-discovery[report]
```

`report.md`: header (dataset name/shape/date-range, natex version, seed, created, the
banner line "AI-generated — verify before citing"), verdict table (family | status | reason),
one section per family: description, applicability (incl. override lines "heuristic said X;
analyst said Y — reason"), key-numbers table, figure PATHS referenced (not embedded),
diagnostics bullets, the registry caveat line. Numbers through a local `_fmt` (em dash for
None/non-finite — never "nan"/"None").

`report.html` (jinja2, StrictUndefined, template in report/templates like paper.py): single
self-contained file — inline CSS, every figure PNG read from `out_dir` and embedded as
`data:image/png;base64,...`; no external requests. Layout: header block (dataset
name/shape/date-range from `dataset`, natex version, seed, timestamp, prominent banner
"AI-generated — verify before citing"); verdict summary table of all seven with status badges
(distinct colors + icons: credible `#009E73` ✓, null `#999` ○, skipped `#0072B2` –,
needs-input `#E69F00` ⚠, failed `#D55E00` ✗ — Okabe–Ito, matching figures) + reasons; then
one `<section id="<family>">` per family: plain-language description (registry), the
applicability verdict incl. any heuristic-vs-analyst override, key-numbers table (estimates,
SEs, CIs, p-values), embedded figures (or the no-figure reason), diagnostics list, and the
per-method honest-inference caveat line. Missing PNG file ⇒ the no-figure reason is shown
instead (never a broken img).

Runner wiring: after `save()`, always `render_survey_md`; then try `render_survey_html`
except ImportError ⇒ record `report_html=None` and append the install message to
`coverage["notes"]`; re-save survey.json with report paths.

**Tests:**
- `test_report_md_always(tmp_path)`: cross-section run WITHOUT the report extra assumption —
  monkeypatch jinja2 import away (`monkeypatch.setitem(sys.modules, "jinja2", None)` pattern
  from the paper tests) ⇒ report.md exists, contains all seven family titles, the banner, the
  canonical kink skip reason; report_html is None and the message mentions
  `natex-discovery[report]`.
- `test_report_html_contract(tmp_path)` (importorskip jinja2 + matplotlib): rdd synthetic run
  ⇒ report.html exists; contains all seven `<section` blocks; >= 1 `data:image/png;base64,`;
  the banner; all five badge classes present across a doctored result dict (render the
  template directly with one family per status); NO substring `nan` (case-insensitive word)
  or `None` in rendered text (napkin trap words respected: assert with `\bnan\b|\bNone\b`).
- `test_html_self_contained`: rendered html contains no `http://`/`https://` src/href except
  inside free-text reasons (assert no `<img src="http`, no `<link`, no `<script src`).
- `test_override_rendered`: doctored result with a kink override ⇒ "heuristic said" and
  "analyst said" both appear.
- `test_md_html_render_from_loaded_json(tmp_path)`: both renderers run off
  `json.loads(survey.json.read_text())` — proves the JSON-native contract.

**Commit:** `feat(report): self-contained survey report.html with md fallback`

## Task 9 — CLI `natex survey` + seeded determinism

**Files:** `src/natex/cli.py` (add command), `tests/test_cli_survey.py` (new).

**Interface:**

```
natex survey CSV
  --context TEXT           free-text dataset context passed to the analyst
  --backend null|agent|anthropic|gemini   (default null; reuse _make_backend + _BACKEND_HELP)
  --model NAME             LLM model (--backend anthropic|gemini)
  --workdir DIR            agent-backend request/response dir; default OUT/guidance
  --out DIR                default out/survey
  --seed N                 default 0; converted ONCE to the run's single numpy Generator
  --time COL --unit COL
  --cutoff COL=VALUE       repeatable (typer list[str]); malformed -> exit 2 naming the arg
  --instrument COL         repeatable
  --threshold COL=VALUE    repeatable
  --k --q --coarse/--no-coarse --n-coarse --max-configs   (same names/defaults as discover;
                           forwarded into budget only when explicitly passed — reuse the
                           ctx.get_parameter_source(...).name == "COMMANDLINE" idiom)
```

Behavior: parse COL=VALUE pairs (`col, _, raw = item.partition("=")`; float(raw); exit 2 with
the offending item on failure); unreadable CSV exit 2; call
`survey(csv, ..., rng=np.random.default_rng(seed), seed=seed, out_dir=out)`; print the
verdict summary table to stdout (fixed-width: FAMILY 9 / STATUS 12 / REASON truncated to
terminal-safe 70 chars) followed by `report: <report.html or report.md>` and
`survey: <out>/survey.json`. Exit 0 whenever survey.json was written (failed families are a
recorded outcome, not a CLI failure).

**Tests:**
- `test_cli_survey_smoke(tmp_path)`: rdd synthetic CSV, `--seed 0 --q 9 --k 25` ⇒ exit 0;
  stdout has 7 family lines + the report path; files exist (survey.json, report.md,
  families/, intake/).
- `test_cli_survey_declared_flags(tmp_path)`: `--cutoff z=2.0 --threshold z=2.0
  --instrument x1` reach the result's applicability inputs (assert from survey.json).
- `test_cli_survey_bad_cutoff`: `--cutoff z:2` ⇒ exit 2, message names `z:2`.
- `test_cli_survey_deterministic(tmp_path)`: TWO runs, same CSV/seed/flags, different out
  dirs ⇒ `survey.json` payloads identical after (a) dropping every `created` key and (b)
  rewriting both out-dir prefixes in path strings to `<out>`; a THIRD run with a different
  seed differs in at least one key_numbers value (guards against accidentally
  seed-independent statistics).
- `test_cli_survey_unknown_backend`: exit 2 via `_make_backend` (reuse existing pattern).

**Commit:** `feat(cli): natex survey — one-command systematic design survey`

## Task 10 — generalizability guard

**Files:** `tests/test_survey_generalizability.py` (new).

- `test_no_dataset_specific_tokens`: read every file under `src/natex/survey/` plus
  `src/natex/report/survey_html.py` and `src/natex/report/templates/survey.html.j2`;
  case-insensitive assert NONE of the tokens appear:
  `fitbit, epoch, prop99, smoking, chinchilla, gpqa, metr, lmarena, takeout`
  (build the token list at runtime from parts, e.g. `"fit" + "bit"`, so the test file itself
  cannot trip the grep).
- `test_heuristic_applicability_is_content_blind`: the `_RecordingProfile` stub from task 1,
  now driven through the full `heuristic_applicability(stub, None, DeclaredInputs(...))` —
  assert accessed attribute names ⊆ the allowed metadata set and that NO attribute access
  returns/receives a pandas object (stub returns only ints/lists/dataclass-like columns);
  also assert by signature inspection that neither `heuristic_applicability` nor any registry
  `check` accepts a DataFrame parameter (`inspect.signature` annotation scan).
- `test_no_column_name_literals_in_verdict_logic`: registry + applicability sources contain
  no string equality against specific data column names — enforced by asserting the modules
  contain none of the benchmark role names as string literals (`"pretest"`, `"posttest"`,
  `"months_23"`, `"cigsale"`, `"dist_from_cut"`).

**Commit:** `test(survey): generalizability guard — token grep + content-blind applicability
proof`

## Task 11 — skills/natex-survey/SKILL.md

**Files:** `skills/natex-survey/SKILL.md` (new), `tests/test_skills.py` (modify: append
`"natex-survey"` to `SKILL_DIRS` — that is the failing-test-first step, as prior phases did).

Frontmatter (single-line description, house rule): name `natex-survey`; description with
trigger phrases: "survey this dataset for natural experiments", "run natex against this
dataset", "which quasi-experimental designs apply to this data", "one-command natex report".
Body teaches:
1. The one-command flow (`uv run natex survey data.csv --context "..." --seed 0 --out
   out/survey`), what lands where (survey.json, report.html/report.md, families/, figures/,
   intake/guidance_log.jsonl), and the seven families with one-line meanings.
2. Serving the guidance protocol as the analyst (`--backend agent`): watch
   `OUT/guidance/requests/`, answer each `NNNN_method_applicability.json` (and the study
   tasks) by writing the response file; how to compose sensible per-family run/skip verdicts
   + config hints from the dataset context (concrete worked example: a panel with a statutory
   threshold ⇒ kink cutoff hint + did run=true), and that every override is recorded and
   hints feed config only — never statistics.
3. Presenting report.html to the user WITH the caveats: always quote the per-family caveat
   line, never drop the "AI-generated — verify before citing" banner, state that
   skipped families are reasoned decisions listed in the verdict table, and never call a
   `null`/`needs_input` family a negative finding.
4. Declared-input flags for non-LLM runs (`--cutoff/--instrument/--threshold/--time/--unit`).

Test contracts (existing harness): kebab name matches dir; one-line description mentioning
"survey"; every `natex ...` command taught is a registered CLI command (`commands_taught ⊆
registered_commands` — `survey` exists after task 9); JSON blocks parse; body mentions
`method_applicability` and `guidance` (add explicit asserts in a new
`test_survey_skill_contract`).

**Commit:** `docs(skills): natex-survey skill — one-command survey with agent-served
applicability guidance`

## Task 12 — method card + README + degradation notes

**Files:** `docs/method_cards/survey.md` (new), `README.md` (modify), `tests/test_docs.py`
(extend).

`docs/method_cards/survey.md`: how the orchestrator works (flow diagram in text: profile →
study → applicability → per-family pipelines → figures → report); how applicability verdicts
are made (registry predicates over profile/declared only) and overridden (method_applicability
task, override recording, hint hygiene, "config not statistics"); status semantics table
(incl. the documented dee stretch and sc's SC_ALPHA=0.10 granularity rationale); report
anatomy (header, badges, per-family sections); default choices that carry NO optimality claim
(kink median-absolute-distance bandwidth, dee grid=8, ALPHA constants); the gp-extra gate on
dee (and that the underlying GP is the core numpy one); extras degradation matrix
(none/plot/report); audit-item table from this plan verbatim. No "None"/"nan" substrings in
prose (card feeds future report contexts; napkin hygiene rule — beware "Nonetheless").

README: new "One-command survey" subsection at the TOP of the quickstart (before the
study/discover flow): 5-line copy-paste (`uv run natex survey mydata.csv --seed 0`), what the
report shows, the banner sentence, pointer to the method card and skill. Keep hard-wrap;
tests must whitespace-normalize (napkin: `" ".join(text.split())`).

`tests/test_docs.py` additions: README flat-text contains "one-command survey" and
"natex survey"; `docs/method_cards/survey.md` exists, names all seven families, contains
"never statistics" and "verify before citing"; the card's family list matches
`FAMILY_ORDER` programmatically.

Final gate for the phase: `uv run ruff check src tests` + `uv run pytest -q` green on the
dev env; then re-run with `uv sync --extra dev --extra plot --extra report` (CI config) and
confirm skip counts changed only by the intended new extras-gated tests.

**Commit:** `docs(survey): method card + README one-command survey section`

---

## Task order recap (each = failing test first → implement → ruff + full pytest → commit)

1. Commit this plan; survey package + MethodFamily registry.
2. `heuristic_applicability`.
3. `method_applicability` guidance task + recorded overrides + NullBackend echo.
4. `binned_poisson_jump` refactor + `bunching_hist`.
5. Runner skeleton: SurveyResult/FamilyResult, rdd/did via discover, isolation, coverage.
6. kink/iv/sc/bunching/dee family runners.
7. Per-family figures + no-matplotlib degradation.
8. report.md + report.html.
9. CLI `natex survey` + determinism.
10. Generalizability guard tests.
11. natex-survey skill.
12. Method card + README.
