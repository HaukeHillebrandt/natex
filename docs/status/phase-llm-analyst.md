# Phase llm-analyst status — LLM analyst pass (Stage 0) + guidance backends

Date: 2026-07-12. Plan: [docs/plans/phase-llm-analyst.md](../plans/phase-llm-analyst.md).
Spec gate (design spec §6, §4 API contract, §10 "LLM guidance could bias discovery" risk):
**Stage-0 analyst pipeline (`natex.study`) + plan-driven budget-aware discovery
(`natex.discover`) + advisory in-scan hooks + Null/Mock/Agent/Anthropic/Gemini guidance
backends, with everything runnable offline and guidance provably unable to alter a
statistic** — met, with the deviations logged here and in
[docs/method_cards/llm_analyst.md](../method_cards/llm_analyst.md). Core deps unchanged;
`anthropic`/`google-genai` live ONLY under the new `[llm]` extra and are never needed by CI.

## What shipped

- **Guidance core** (`llm/backends.py`, `llm/log.py`): `GuidanceRequest`/`GuidanceResponse`
  pydantic models, the `GuidanceBackend` protocol, the six-task vocabulary +
  `TASK_INSTRUCTIONS`, `MockBackend` (canned responses, records every request), and the
  append-only JSONL `GuidanceLog` + `LoggedBackend` decorator — every request+response for
  EVERY backend is logged (spec 6c reproducibility).
- **NullBackend** (`llm/backends.py`): deterministic profile-only heuristics for all six
  tasks (bitwise-stable `raw_text`, never vetoes) — the no-LLM degradation path every
  pipeline entry point falls back to.
- **Declarative prep** (`intake/prep.py`): `PrepPlan` (filters → drops → encodings →
  discretize → seeded subsample) with `validate_against(df)` and a deterministic,
  logging `apply()`; the LLM proposes plans, ONLY natex code executes them.
- **Plan models** (`intake/plans.py`): `Understanding`, `DesignCandidate` (design-specific
  validation + order-insensitive dedup `key()`), `SearchPlan` (stable priority ranking,
  budget hints).
- **AgentBackend** (`llm/agent.py`): file-based request/response subscription protocol for
  a calling coding agent (zero API cost) — restart-safe sequence numbers, partial-write
  tolerance, envelope-or-bare response shapes, actionable timeout message.
- **API backends** (`llm/api.py`, `[llm]` extra): `AnthropicBackend` (structured
  `output_config` json_schema output) and `GeminiBackend` (`response_json_schema` with a
  one-shot TypeError fallback for older SDKs), lazy SDK imports with the install-line
  ImportError, shared `_strict_schema`/`_prompt` helpers.
- **Stage 0** (`intake/analyst.py`): `natex.study()` — profile → understand → validated
  PrepPlan → SearchPlan with uniform fall-back-to-Null policy (`strict=True` re-raises),
  candidate column post-validation, outcome-snooping warnings, serializable `IntakeReport`
  (`save`/`load`/`prepare`), plus the shared `natex.jsonutil.jsonable` (extracted from the
  CLI's `_clean`, NaN → null).
- **Discovery orchestrator** (`discover.py`): plan-ranked-first, exhaustive-still,
  budget-aware sweep where every enumerated config becomes a `ConfigRecord`
  (`scanned/skipped_budget/failed/invalid`) and `DiscoverReport.searched` always reports
  coverage (spec 6b); statistics reused verbatim from scan/validate/estimate (audit 1).
- **In-scan hooks** (`discover.py`): `interpret_discovery` → `audit_assumptions` →
  (did GESS) `review_control_group`, advisory-only with error isolation; vetoes are flags
  (`advisory_veto`, `effects["gess"]["vetoed_by_guidance"]`), mutation-tested to leave
  statistical output bitwise unchanged (spec 6c).
- **CLI**: `natex study CSV --context ... --backend null|agent|anthropic|gemini --out DIR`
  and `natex discover --plan intake_report.json` (budget = plan hints ← explicit CLI flags,
  detected via `ctx.get_parameter_source`); missing `[llm]` extra exits 2 with the install
  message, no traceback.
- **Eval scaffold** (`guidance_eval.py` + `benchmarks/guidance_eval.py` +
  benchmarks/README section): blind-vs-informed rank of the true design; CI proves the
  discrimination with MockBackend (informed rank 0, blind rank ≥ 1 on decoy-first cases).
- **Docs**: [LLM-analyst method card](../method_cards/llm_analyst.md), README quickstart
  `study` → `discover --plan` flow + roadmap tick, this status doc.

## Test counts

134 tests added this phase (`test_llm_backends.py`, `test_llm_null.py`,
`test_intake_plans.py`, `test_prep_plan.py`, `test_llm_agent.py`, `test_llm_api.py`,
`test_study.py`, `test_discover.py`, `test_cli_study.py`, `test_guidance_eval.py`):
523 (phase 5) → 657. All LLM tests use MockBackend / fake injected clients /
import-guard monkeypatching — no network, no API keys, anywhere.

## Final gate record (2026-07-12, Apple Silicon macOS arm64, Python 3.13.14)

1. `uv run ruff check src tests` — `All checks passed!`
2. NO extras installed (`uv sync --extra dev`; note dev is itself an extra):
   `uv run pytest -q` — **`643 passed, 14 skipped, 32 deselected in 114.47s`**. The two
   `[llm]` SDK smoke tests skip visibly:
   `SKIPPED [1] tests/test_llm_api.py:268: anthropic SDK not installed`,
   `SKIPPED [1] tests/test_llm_api.py:276: google-genai SDK not installed`
   (remaining skips = the pre-existing ml/gp/plot optional-extra tests).
3. `uv sync --all-extras && uv run pytest -q tests/test_llm_api.py` — **`20 passed`**
   (both SDK smokes now run; still no network). Full suite with all extras:
   **`657 passed, 32 deselected in 171.85s`**.
4. Backtest regression (`NATEX_DATA=".../RDD/data" uv run pytest -q -m backtest`) —
   **`31 passed, 657 deselected, 1 xfailed in 123.09s`**: identical outcome set to the
   phase-5 record (all 15 phase-2 RDD rows, 9 phase-3 Prop 99 SuDDDS tests, 8 phase-5
   donor/IV tests incl. the non-blocking Egger xfail). This phase disturbed nothing.

### Manual AgentBackend smoke (gate 5, documented not automated)

2026-07-12, synthetic CSV from `make_synthetic(n=300, px=3, pz=2, zeta=6.0, kind="binary",
rng=default_rng(0))` (columns x0,x1,x2,T,y):

1. `uv run natex study synthetic.csv --backend agent --seed 0 --out out/` blocked polling
   after echoing
   `natex guidance request (understand): answer by writing JSON matching schema_hint to .../out/guidance/responses/0000_understand.json` [dir renamed OUT/agent → OUT/guidance in phase skills-docs; path updated].
2. The three request files (`0000_understand.json`, `0001_prepare.json`,
   `0002_search_plan.json`) were answered by hand — the `understand` reply as a bare
   content object, the `prepare` reply deliberately in the `{"content": {...}}` envelope
   form. After each response file was written the pipeline resumed within one poll
   interval and issued the next request.
3. Exit 0 with `shape: cross-section  unit of observation: row`,
   `candidates: 1  top: rdd treatment=T forcing=x0,x1`; `intake_report.json`,
   `prep_plan.json` and `guidance_log.jsonl` written; the log's three lines all record
   `"backend": "agent"`.
4. Round trip: `uv run natex discover --plan out/intake_report.json synthetic.csv --q 9
   --k 25 --seed 0 --out out2/` → exit 0,
   `scanned 1/1 configs (0 skipped by budget)`,
   `best: rdd treatment=T forcing=x0,x1  llr=11.51  p=0.100` (the +1-rank floor at Q=9),
   `discover_report.json` written.

## Deviations log (all deliberate, none silent)

1. **Recovery start**: a previous execution attempt had committed plan tasks 1–2 and left
   task 3 code-complete but uncommitted; this run committed the recovered task-3 work
   verbatim per the plan's recovery protocol before proceeding to task 4.
2. **File split vs the brief**: `AgentBackend` lives in `llm/agent.py` and the API backends
   in `llm/api.py` (not all in `backends.py`), all re-exported from `natex.llm` — kept for
   reviewability, per the plan's deviations section.
3. **`IntakeReport.prepare()` returns `Dataset` always** (never a separate PanelDataset);
   DiD entry points consume `Dataset` and `CategoricalPanel` is built inside `discover()`.
4. **Sandboxed-pandas escape hatch (spec 6a.3) out of scope** — declarative plans only
   (method card "Boundaries").
5. **`propose_forcing_variables` / `propose_candidate_events` are not separate task
   literals** — satisfied by the Stage-0 SearchPlan this phase (method card).
6. **API-backend live calls never run in CI** — SDK call shape pinned by fake-client kwarg
   assertions; the `[llm]` extra is deliberately not installed in CI (pyproject comment),
   and the constructor smokes skip gracefully without it.
7. **Schema-hint models avoid pydantic field constraints** (`gt=`/`le=` etc.) in favor of
   `field_validator`s: `_strict_schema` strips constraint keys for strict structured-output
   modes, so constraints must live where stripping cannot lose them
   (regression-guarded in `tests/test_llm_api.py`).

## Open questions for the next phases

- Reporting pipeline (phase 7) should render `ConfigRecord.advisory` (interpretation,
  assumption audit, control review, veto flags) alongside the statistics — the hooks
  already log everything it needs.
- Blind-vs-informed eval on the real backtest suite (spec's guided-vs-unguided evaluation)
  with an API arm remains a manual run (`benchmarks/guidance_eval.py --backend anthropic`);
  worth a run of record once an API key budget is allocated.
- Phase-5 open question carried forward: should `interpret_discovery` payloads include
  `ar_kind` as a first-class caution signal when narrating discovered-cutoff effects?
