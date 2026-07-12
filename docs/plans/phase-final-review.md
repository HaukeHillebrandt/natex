# Phase final-review implementation plan — multi-agent review, fixes, re-release

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/superpowers/specs/2026-07-10-natex-design.md"`.
**Method extraction notes:** `"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes"`.
**Local benchmark data root:** `NATEX_DATA="/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data"`.
**First action of task 1 is committing this plan file itself**
(`docs: phase final-review implementation plan`).

## Phase objective

This is a REVIEW-AND-FIX phase, not a feature phase. Four review streams, then fixes, then a gate:

- **(a) Math-audit conformance:** walk `docs/math_audit_final.md` §2 (consolidated errata,
  items 1–24 plus the pure-typos block) and §3 (improvement decisions) ITEM BY ITEM; for each,
  verify the natex implementation honors it and record `file:line` evidence in
  `docs/status/final-review.md`. Any item not honored is a finding.
- **(b) API consistency sweep:** every stochastic entry point takes an explicit
  `numpy.random.Generator` (no global seeding); NaN-never-0.0 on failure paths; result
  dataclasses follow the established conventions (`tau`/`se`/`ci` naming, extras dicts); CLI
  flags consistent across the 9 commands; no bare except; no silent caps.
- **(c) Docs accuracy:** EXECUTE the commands documented in `README.md`, `AGENTS.md`, and the
  three `skills/*/SKILL.md` files (scratch directory, packaged synthetic, small seeded runs)
  and confirm documented outputs/paths match reality — including serving one AgentBackend
  request/response round-trip by hand the way the skill instructs.
- **(d) Completeness critic vs the design spec:** section by section, every promise is either
  implemented (cite where) or explicitly recorded in `docs/status/future_work.md` with a
  one-line rationale.

Findings from (a)–(d) are triaged into **confirmed-must-fix** (correctness / doc-wrong) vs
**noted** (future work); fixes are implemented test-first. Two already-confirmed nits from
phase skills-docs are fixed regardless: stale `out/agent` example paths in
`docs/method_cards/llm_analyst.md` + `docs/status/phase-llm-analyst.md`, and the missing Typer
help text on the `discover` command.

**Deliverables:** `docs/status/final-review.md` (audit-item evidence table + findings + fixes),
`docs/status/future_work.md`, all fixes committed and pushed, CI green, tag `v0.1.0` moved to
the reviewed state and the GitHub release edited to say so.

## Global constraints (binding, from the phase-1 plan)

- Python >= 3.11. Core deps stay exactly numpy/scipy/pandas/scikit-learn/typer/pydantic;
  everything else is an optional extra whose tests skip gracefully when missing (CI on
  3.11–3.14 must stay green). **No new dependency anywhere in this phase.**
- One `numpy.random.Generator` through every stochastic call; discovery never reads the
  outcome; NaN never 0.0 on failure; no bare except; no silent caps.
- Never commit datasets or anything under `out/` / scratch dirs.
- Conventional commits after every green cycle.
- `uv run pytest -q` excludes backtests (`addopts = -m 'not backtest'`); backtests run as
  `NATEX_DATA="…/RDD/data" uv run pytest tests/backtests -m backtest -q`.
- TDD discipline for every FIX: failing test first, then implementation, full suite, commit.
  Review tasks (2–6) produce docs + findings, committed as `docs:` after each task.

## Verified current state (2026-07-12, planner)

- HEAD `62c2aa6` on `main`, clean tree, in sync with `origin/main`. Tag `v0.1.0` points AT
  HEAD; GitHub release `v0.1.0` exists ("natex v0.1.0 — automated natural-experiment
  discovery"). Any commit this phase lands ⇒ the tag must be moved at the gate (task 9).
- Run of record (phase skills-docs): ruff clean; `uv run pytest -q` → 796 passed,
  32 deselected; backtests → 31 passed, 1 xfailed.
- **Pinned counts:** `tests/test_readme_release.py:22-23` pins `N_NONBACKTEST = 796`,
  `N_BACKTEST = 32` against README (line ~430). Every test added this phase must bump BOTH
  the constant and the README sentence in the same commit, or the suite goes red.
- The 9 CLI commands (`src/natex/cli.py`): `datasets`, `fetch-data`, `study`, `discover`,
  `debias`, `instruments`, `donors`, `paper`, `brief`.
- Known nit 1 confirmed: `docs/method_cards/llm_analyst.md` lines 106, 128, 132 and
  `docs/status/phase-llm-analyst.md` line 90 still say `out/agent/...`; the default was
  renamed to `OUT/guidance` in phase skills-docs. (`docs/plans/phase-skills-docs.md` mentions
  `OUT/agent` only historically — that is fine and must NOT be edited.)
- Known nit 2 confirmed: `def discover(...)` (`src/natex/cli.py:280`) has NO docstring, so
  `natex --help` shows an empty description for `discover`; every other command has one.
- CI (`.github/workflows/ci.yml`): matrix 3.11–3.14, `uv sync --extra dev --extra plot
  --extra report`, ruff + pytest.
- Doc-test infrastructure already exists and must be extended, not duplicated:
  `tests/doc_helpers.py` (`flat`, `fenced_blocks`, `json_blocks`, `registered_commands`,
  `commands_taught`), `tests/test_skills.py`, `tests/test_agent_docs.py`, `tests/test_docs.py`,
  `tests/test_readme_release.py`, `tests/test_release_notes.py`.

## Findings register format (used by tasks 2–6, consumed by tasks 7–8)

All findings live in `docs/status/final-review.md` §"Findings register", one table:

| ID | Source | Severity | Where | Defect | Resolution |
|---|---|---|---|---|---|

- **ID**: `F-A1…` (task 2, audit items 1–10), `F-B1…` (task 3, items 11–24 + typos + §3),
  `F-C1…` (task 4, API sweep), `F-D1…` (task 5, docs execution), `F-E1…` (task 6, spec
  completeness). Prefixes are disjoint so tasks 2–6 can, if the executor chooses, run as
  parallel subagents appending to their own sections without ID collisions; the plan is
  written to work sequentially too.
- **Severity**: `must-fix` (implementation contradicts the audit/spec, or a doc states
  something false about behavior) or `noted` (real but deliberately out of scope → one line
  in `future_work.md`).
- **Resolution**: filled by tasks 7–8: `fixed <commit-sha>` / `noted → future_work.md` /
  `no-change (justification)`.

Severity rubric: anything where the CODE disagrees with `math_audit_final.md` §2/§3 is
must-fix. Anything where a DOC misdescribes actual behavior is must-fix (fix the doc, or the
code if the doc states the contract we want). Spec promises never implemented are `noted`
unless the README/AGENTS/skills claim they exist.

---

## Task 1 — Commit the plan; scaffold the two status documents

**Files:** commit `docs/plans/phase-final-review.md` (this file) as the FIRST action;
create `docs/status/final-review.md` and `docs/status/future_work.md` skeletons.

`docs/status/final-review.md` skeleton (exact section headings — later tasks append under
them, and the task-8/9 tests key on them):

```markdown
# Final review — audit conformance, API sweep, docs execution, spec completeness

Date started: 2026-07-12. Plan: [docs/plans/phase-final-review.md](../plans/phase-final-review.md).

## A. Math-audit conformance (docs/math_audit_final.md §2 + §3)

| Audit item | Status | Evidence |
|---|---|---|

Status vocabulary: HONORED / HONORED-WITH-CAVEAT / NOT-HONORED / DOC-ONLY (typo items:
documented in a method card, no code impact).

## B. API consistency sweep

## C. Docs accuracy execution log

| Doc | Command executed | Expected (as documented) | Observed | Verdict |
|---|---|---|---|---|

## D. Spec completeness matrix

| Spec section | Promise | Disposition | Evidence / future_work entry |
|---|---|---|---|

## Findings register

| ID | Source | Severity | Where | Defect | Resolution |
|---|---|---|---|---|---|

## Fixes applied

## Run of record
```

`docs/status/future_work.md` skeleton:

```markdown
# Future work — deliberate scope boundaries as of v0.1.0 (final review)

Each row is a design-spec or audit promise intentionally not in v0.1.0, with the one-line
rationale recorded by the final review (docs/status/final-review.md §D).

| Item | Spec/audit ref | Rationale |
|---|---|---|
```

**Test requirement (failing-first):** add `tests/test_final_review_docs.py` with
`test_status_docs_exist_with_required_sections` — asserts both files exist and
`final-review.md` contains the exact headings `## A. Math-audit conformance`, `## B. API
consistency sweep`, `## C. Docs accuracy execution log`, `## D. Spec completeness matrix`,
`## Findings register`, `## Fixes applied`, `## Run of record`; `future_work.md` contains its
table header row. Write the test, watch it fail, create the skeletons, watch it pass.
Bump `N_NONBACKTEST` 796 → 797 and the README count sentence in the same commit.

**Commits:** `docs: phase final-review implementation plan` (plan file alone, first), then
`docs(review): final-review + future_work status skeletons with pinned sections`.

## Task 2 — Review stream (a) part 1: audit §2 items 1–10 (statistical validity)

**Files modified:** `docs/status/final-review.md` (§A table + Findings register). Read-only
with respect to `src/`.

For EACH of items 1–10, read the implementation, run the pinning test(s) if in doubt, and add
one table row `item → status → evidence` where evidence is `path:line` (function or the exact
expression) PLUS the test file that pins the behavior. The pointers below are the planner's
starting hints — the reviewer must verify, not trust them:

1. Fitted-null Monte Carlo (not exact), +1-rank p-values, honest split —
   `src/natex/validate/randomization.py`, `src/natex/validate/honest.py`; check the p-value
   formula is `(1 + #{null ≥ obs}) / (Q + 1)` and no docstring/output string claims exactness.
2. Bernoulli nulls drawn `T* ~ Bernoulli(p̂)` directly (never `1{p̂+σ̂Z>U}`) —
   `src/natex/validate/randomization.py` + `src/natex/scan/statistics.py` (Bernoulli model);
   confirm the replica generator calls `rng.binomial`/`rng.random() < p̂` on the fitted mean.
3. Placebo = local intercept-continuity with side-specific trends, joint/multiplicity
   (Holm/Bonferroni), categorical covariates via distributional tests — `src/natex/validate/placebo.py`.
4. Group-instrument repaired: frozen side-indicator 2SLS primary; the printed `W = T − μ`
   form absent or explicitly quarantined — `src/natex/estimate/iv2sls.py`,
   `src/natex/estimate/local2sls.py`.
5. τ̂ placebo (ch.6): two-sided studentized statistic, +1 rank, matched subset shapes —
   `src/natex/did/statistics.py` / `src/natex/validate/panel.py`.
6. Density test on signed distance along the FROZEN normal, honest split, documented as
   falsification-only — `src/natex/validate/density.py`, `src/natex/rdd/metrics.py`.
7. DEE Theorem 1 √2 factor / corrected variance usage — `src/natex/dee/debias.py`,
   `docs/method_cards/dee.md` (this one may legitimately be DOC-ONLY plus a conservative
   implementation choice; record which).
8. Mixture intervals: ONE model label per posterior draw; mixture covariance
   `wΣ_β+(1−w)Σ_τ+w(1−w)(μ_β−μ_τ)(μ_β−μ_τ)ᵀ` — `src/natex/dee/bma.py`.
9. Cross-fitting between observational estimator and discovery/IV stage; classical 2SLS SE²
   never treated as known independent GP noise — `src/natex/dee/observational.py`,
   `src/natex/dee/noise.py`, `src/natex/dee/gp.py`.
10. First-stage relevance checked AFTER validation/repair; weak-IV-robust (AR/Fieller)
    intervals surfaced — `src/natex/estimate/iv2sls.py` (`first_stage_t`, `weak_instrument`,
    AR CI), `tests/test_ar_ci.py`.

Every NOT-HONORED or HONORED-WITH-CAVEAT row also gets a `F-A#` findings-register entry with
proposed severity. **Acceptance:** 10 rows, none blank; register updated.

**Commit:** `docs(review): audit conformance evidence, items 1-10`.

## Task 3 — Review stream (a) part 2: audit §2 items 11–24, typos block, §3 decisions

**Files modified:** `docs/status/final-review.md` (§A table + register).

Items 11–24 (same row format; hints):

11. Alg 6 global incumbent across windows/restarts — `src/natex/did/suddds.py`.
12. Alg 7 minimum precision mass both sides of every cutoff candidate — `src/natex/did/suddds.py`.
13. Alg 8 relax-dimension subset updates retaining the incumbent — `src/natex/did/mdss.py`.
14. Alg 9 `argmin` + control-set MSE initialized `+∞` — `src/natex/did/controls.py` (GESS).
15. Single-Δ profile GLR: C̃ᵢ/B̃ᵢ with precision-weighted δ̄ correction, window restriction,
    both signs of Δ — `src/natex/scan/statistics.py` / `src/natex/did/statistics.py`
    (audit's numeric check: printed LLR 4.74 vs correct 5.33 — cite the regression test that
    pins the corrected value).
16. WCC heuristic labeled as such; exhaustive per-dimension enumeration (2^V−1) the exact
    default at small cardinality — `src/natex/did/mdss.py`.
17. Eq 6.10 prose swap — DOC-ONLY: `docs/method_cards/suddds.md` must note it.
18. Panel replica nulls preserve unit/time dependence; NO DiD-in-time McCrary — composition/
    anticipation checks instead — `src/natex/validate/panel.py`, `src/natex/did/background.py`.
19. Continuous-treatment estimand (ζ·τ dose normalization or IV form) + model class matches
    T's type (Bernoulli/Poisson for binary/count) — `src/natex/did/effects.py`,
    `cli.py` `--model auto` plumbing.
20. Legacy reverse-neighbor variance bug: natex uses point i's own kNN variance; legacy
    outputs NOT used as σ̂² ground truth in any parity test — `src/natex/scan/statistics.py`
    + audit of `tests/backtests/` and `tests/test_statistics_normal.py` for illegitimate
    parity assertions.
21. Sharp-RDD pure-group bisections scored via boundary likelihood supremum (never NA-dropped)
    — `src/natex/scan/statistics.py`.
22. Bracketed Newton/bisection in θ=log β; no bracket-doubling runaway; complement-dedup fixed
    — `src/natex/scan/statistics.py`, `src/natex/scan/neighborhoods.py`.
23. Hyperplane tie convention explicit (signed distance ≥ 0 → group 1, documented) —
    `src/natex/scan/geometry.py` + method card.
24. Data-scaled variance shrinkage, no absolute 1e-6 floor — `src/natex/scan/statistics.py`.

**Typos block (§2 "Pure typos"):** one row `typos-block → status → evidence`; every listed
typo must appear in the relevant method card (`lord3.md`, `suddds.md`, `dee.md`) as a
documented correction. Spot-check each card; missing mentions → one aggregated finding.

**§3 improvement decisions:** one row per bullet (9 adopted, 4 rejected/downgraded). Adopted
⇒ cite implementation `path:line`; rejected ⇒ cite absence AND that no doc claims it (e.g.
within-neighborhood permutation must not exist as a validation path; ANN opt-in only).

**Acceptance:** 14 item rows + 1 typos row + 13 §3 rows, none blank; register updated.

**Commit:** `docs(review): audit conformance evidence, items 11-24 + typos + section-3 decisions`.

## Task 4 — Review stream (b): API consistency sweep

**Files modified:** `docs/status/final-review.md` (§B + register). Read-only on `src/`.

Concrete checks (record command + result for each in §B):

1. **Generator discipline.** `grep -rn "np\.random\.\(seed\|rand\|randn\|choice\|normal\|uniform\|default_rng()\)" src/natex` must return NOTHING (module-level `default_rng` without an argument or any legacy `np.random.*` call is a finding). Enumerate every public function/dataclass with stochastic behavior (grep `rng` and `seed` params across `src/natex`); each must accept `rng: np.random.Generator` (or a `seed` that is converted ONCE at the CLI boundary). List them in §B.
2. **NaN-never-0.0.** For each failure path in estimators/validators (grep `float("nan")`,
   `np.nan`, `return 0.0`, `except`): verify no failure path yields 0.0/empty-success. Every
   `return 0.0` hit must be a genuine mathematical zero, not a fallback.
3. **Result dataclass conventions.** Table over the 10 result dataclasses
   (`LoRD3Result`, `SuDDDSResult`, `CoarseToFineResult`, `ControlResult`, `VKNNResult`,
   `DEEResult`, `InstrumentSearchResult`, `DonorSelectionResult`, `PaperResult`,
   `ResultsBundle`): effect fields named `tau`/`se`/`ci` (not ad-hoc synonyms), open-ended
   metadata in an `extras`-style dict, JSON-serializable via `natex.jsonutil.jsonable`
   (no pickled namespaces).
4. **CLI flag consistency.** For the 9 commands: `--seed` default 0 wherever stochastic;
   `--out` default `Path("out")` wherever files are written; shared concepts share spellings
   (`--treatment`, `--outcome`, `--forcing`, `--workdir`, `--backend`); every option has
   `help=`; every command has a docstring (nit 2 will fail here — record as `F-C#`
   cross-referenced to task 7).
5. **No bare except.** `grep -rn "except:" src tests` empty AND
   `uv run ruff check --select E722,BLE001 src` (advisory: report BLE001 hits; only E722 is
   auto-must-fix).
6. **No silent caps.** Audit every `min(`/`max(`/`np.clip` applied to user-supplied sizes/
   budgets (`k`, `q`, `m_prime`, `max_configs`, `n_coarse`, window widths, restarts): each
   must either be reported in the result (`skipped_budget`-style accounting) or documented in
   the option help. Undocumented truncation = finding.

**Acceptance:** §B records each check with the exact command run and its output summary;
findings `F-C#` filed. **Commit:** `docs(review): API consistency sweep evidence`.

## Task 5 — Review stream (c): execute the documented commands

**Files modified:** `docs/status/final-review.md` (§C log + register). Nothing under the repo
gets written by the runs themselves — ALL runs happen in a scratch dir OUTSIDE the repo (use
the session scratchpad or `mktemp -d`; never `out/` inside the repo).

Protocol — for each documented flow, run it verbatim (substituting the scratch CSV), then diff
observed outputs/paths/filenames against what the doc states; each row in the §C table cites
the doc line:

1. **Synthetic input.** Generate the demo CSV exactly as README's quickstart heredoc does
   (README ~line 126: `natex.data.synthetic` / `make_synthetic`, seeded) into
   `SCRATCH/synth.csv`.
2. **README flows:** `uv run natex --help`; the discover quickstart (~line 133, seeded) —
   confirm `out/results.json` and the printed table match the pasted-output block; the study →
   discover `--plan` flow (~lines 82–84) with `--backend null`; `instruments` (~line 203) and
   `donors` (~line 224) on small synthetic panels (build with
   `natex.data.synthetic_iv` / `synthetic_sc` generators, seeded); the agent-flow block
   (~lines 239–242) including `natex paper --bundle` and `natex brief --bundle` — confirm
   `paper/paper.md` and `research-brief.md` land where documented; `natex datasets`
   (with `NATEX_DATA` set) matches the pasted block's format; benchmark smoke:
   `uv run python benchmarks/run_nig_curve.py --help` (existence + flags only, no full run).
3. **AGENTS.md:** verify the CLI-surface table (all 9 rows: key inputs and output filenames
   truthful — e.g. `discover` plan mode writes `discover_report.json`, plain mode
   `results.json`); run the "typical guided session" (lines ~153–157).
4. **AgentBackend round-trip BY HAND** (the core of this task, per
   `skills/discover-natural-experiments/SKILL.md` §3–4): launch
   `uv run natex study SCRATCH/synth.csv --backend agent --seed 0 --out SCRATCH/out/` in the
   background; poll `SCRATCH/out/guidance/requests/`; confirm `0000_understand.json` appears,
   has EXACTLY the five documented keys (`task`, `payload`, `schema_hint`, `instructions`,
   `respond_to`), and `respond_to` == `WORKDIR/responses/0000_understand.json`; hand-write a
   schema-conforming answer (one write); confirm the run unblocks within ~one poll interval,
   answer subsequent requests (use the `{"content": {...}}` envelope for at least one, since
   AGENTS.md documents both forms), and confirm `intake_report.json`, `prep_plan.json`,
   `guidance_log.jsonl` all appear as documented and the log contains every request+response.
5. **Remaining skills:** `natex-write-paper` — run its documented `natex paper` invocation on
   the results bundle from step 2 and follow its verify-numbers instruction once
   (spot-check ≥3 numbers in `paper.md` against `results.json`); `natex-lit-review` — run its
   `natex brief` invocation, confirm the handoff file and that the skill's described sections
   exist in it.
6. Every mismatch (wrong path, wrong filename, stale flag, output shape differing from a doc's
   pasted block) → finding `F-D#`, severity must-fix (doc-wrong). Expect at least the two
   `out/agent` stale-path hits from the method card / status doc if not already registered.

**Acceptance:** §C table has one row per executed command (≥15 rows), each with a verdict;
scratch dir path recorded; nothing new under the repo tree (`git status` clean except status
doc). **Commit:** `docs(review): docs-accuracy execution log (README, AGENTS, skills)`.

## Task 6 — Review stream (d): completeness critic vs the design spec

**Files modified:** `docs/status/final-review.md` (§D matrix + register),
`docs/status/future_work.md` (rows).

Walk the spec (quote the path — it has spaces) section by section (§1 goal bullets 1–6 and
non-goals, §3 repo-layout tree, §4 core API contract + design rules, §5 corrections 1–8,
§6/6a–6d LLM roles, §7 reporting 1–5, §8 backtest table, §9 phases, §10 risks). One matrix
row per promise: `Implemented (path:line / test)` or `Future work (rationale)` or
`Deviation (finding)`.

Planner's pre-identified candidates the critic MUST adjudicate explicitly (not exhaustive):

- Spec §3 CLI is `discover|validate|estimate|report|paper|fetch-data`; actual CLI is
  `datasets|fetch-data|study|discover|debias|instruments|donors|paper|brief` with
  validate/estimate/report folded into `discover`/bundle outputs. Deviation-by-design →
  future_work/no-change row with rationale (README/AGENTS document the real surface; verify
  no doc still promises `natex validate` etc.).
- Spec §3 `legacy/` directory ("annotated copy of original Herlands src/") — not present in
  the repo. Either cite where the legacy bug cross-reference lives (method cards / audit) and
  record the directory as intentionally dropped (never commit third-party code), or file a
  finding.
- Spec §3 core deps list includes statsmodels + matplotlib; house rules made matplotlib an
  extra and statsmodels absent — record the resolution (house rules win; note it).
- rdrobust/rddensity bridges (§10): own implementations only → future work row.
- Google Docs export (§7.2): manual route documented in README (~line 292) → future work row
  ("intentionally manual").
- PyPI publish (§2, §9 phase 8): pending naming decision (`natex-discovery` reserved) →
  future work row.
- Staggered-adoption DiD beyond pluggable backend (§1 non-goals, audit §3 "group-time ATT
  backend") — record what exists vs deferred.
- Multi-k calibration scope, ANN opt-in scale path (§6d), mypy in CI (spec layout comment says
  "typecheck (mypy, loose)"; actual CI has no mypy step), Egger–Köthenbürger stretch-goal
  status (§10: failure is a documented finding — cite the backtest xfail as the
  documentation), Deep-Research integration boundary (skill + brief, no API call from natex).
- §4 API contract: `natex.study(...)`, `DatasetSpec.from_csv(...)`, `natex.discover(...)`,
  `result.discoveries[0].effect`, `result.to_report("out/")` — run each in a scratch Python
  session; any signature drift between spec and reality must be a future_work/no-change row
  ONLY if README/AGENTS document the real signature; if docs echo the spec's stale form, it
  is a must-fix doc finding.

**Test requirement (failing-first, this task's only code):** extend
`tests/test_final_review_docs.py` with `test_future_work_rows_have_rationales` — parses the
`future_work.md` table, asserts ≥ 6 rows and every row has non-empty Item, Spec/audit ref and
Rationale cells. Bump pinned counts. **Acceptance:** every spec section appears in §D; no
promise left unclassified.

**Commit:** `docs(review): spec completeness matrix + future_work register`.

## Task 7 — Fix the two known nits (test-first)

**Files:** `tests/test_agent_docs.py` (or `tests/test_docs.py` — wherever method-card checks
live; extend, don't duplicate), `tests/test_cli.py`, `docs/method_cards/llm_analyst.md`,
`docs/status/phase-llm-analyst.md`, `src/natex/cli.py`.

**Nit 1 — stale `out/agent` paths.**
- Failing tests first: `test_llm_analyst_card_uses_guidance_dir` — read
  `docs/method_cards/llm_analyst.md`, assert `"out/agent"` NOT in text and
  `"out/guidance/requests/0000_understand.json"` present; `test_phase_llm_analyst_status_notes_rename`
  — read `docs/status/phase-llm-analyst.md`, assert `"out/agent"` not present.
- Fix: in the method card, update lines 106/128/132 (`out/agent/…` → `out/guidance/…`). In the
  status doc, update the quoted echo path at line 90 to `out/guidance/...` and append a
  bracketed editorial note on that line: `[dir renamed OUT/agent → OUT/guidance in phase
  skills-docs; path updated]` — run-of-record docs may not silently rewrite history.
  Do NOT touch `docs/plans/phase-skills-docs.md` (it documents the rename itself).

**Nit 2 — `discover` missing help text.**
- Failing test first: in `tests/test_cli.py`, `test_every_command_has_help_text` — via
  `CliRunner().invoke(app, ["--help"])`, parse the Commands table; assert EVERY command row
  has a non-empty description (generic — this pins the convention, not just `discover`); plus
  assert `"LoRD3"` appears in `invoke(app, ["discover", "--help"]).output`.
- Fix: add a docstring to `discover` in `src/natex/cli.py` (first line becomes the help row):
  `"""Scan for natural experiments: LoRD3 RDD scan or SuDDDS DiD scan, with the validation battery and honest effect estimates."""`
  plus 2–3 lines naming outputs (`out/results.json`; plan mode `out/discover_report.json`)
  mirroring the AGENTS.md table row. Verify `tests/test_skills.py`/`test_agent_docs.py` help
  assertions still pass (they parse the Commands table via `doc_helpers.registered_commands`).

Bump pinned counts (`N_NONBACKTEST` + README sentence) once for the tests added.
Update the two findings' Resolution cells (`fixed <sha>`) and add a "Fixes applied" entry.

**Commits:** `fix(docs): llm_analyst card + status use OUT/guidance paths` and
`fix(cli): discover command help text` (separate green cycles).

## Task 8 — Triage and fix all remaining confirmed findings (test-first, one green cycle each)

**Files:** whatever the register demands; `docs/status/final-review.md` (triage decisions +
"Fixes applied"); `docs/status/future_work.md` (noted items).

1. **Triage pass.** Walk the full register (F-A/B/C/D/E). For each finding set Severity
   definitively per the rubric and record a one-line decision. Must-fix = code contradicts
   audit/spec-corrections, or doc-wrong. Noted = add the `future_work.md` row and mark
   `noted → future_work.md`.
2. **Fix loop, per must-fix finding:** (i) failing test that reproduces the defect — for math
   items: a seeded numeric regression test with a meaningful threshold (e.g. corrected vs
   printed statistic values from the audit, `rtol` explicit; statistical assertions use fixed
   `default_rng(seed)` and thresholds robust to ±2 SE); for doc items: a text-contract test in
   the existing doc-test files; (ii) minimal fix; (iii) `uv run ruff check src tests` +
   `uv run pytest -q` green (bump pinned counts when tests were added); (iv) conventional
   commit citing the finding ID, e.g. `fix(dee): one model label per posterior draw (F-A8)`;
   (v) update the register row (`fixed <sha>`) and, if it was an §A NOT-HONORED row, flip the
   evidence-table status and cite the new test.
3. **Escalation rule:** if a must-fix touches numerical behavior pinned by backtests, run the
   affected backtest file immediately (`NATEX_DATA="…/RDD/data" uv run pytest
   tests/backtests/test_<x>.py -m backtest -q`) before committing.
4. **Exit criteria:** register has ZERO unresolved must-fix rows; every noted row exists in
   `future_work.md`; §A table contains no NOT-HONORED row without a `fixed` cross-reference.

**Test requirement (failing-first, meta):** extend `tests/test_final_review_docs.py` with
`test_findings_register_fully_resolved` — parse the register table in
`docs/status/final-review.md`; assert every row's Resolution cell is non-empty and none says
`open`/`TBD`. (Written at the START of this task — it fails while findings are open and gates
the task's completion mechanically.)

**Commits:** one per fix as above, plus `docs(review): triage decisions + resolutions`.

## Task 9 — Final gates, push, CI, tag move, release edit

No new code. Order matters:

1. Finalize `docs/status/final-review.md`: fill "## Run of record" with REAL pasted output of
   the three gate commands (below), summary counts (findings by severity, fixes landed), date.
2. Gates, in order, all green:
   - `uv run ruff check src tests`
   - `uv run pytest -q` (confirm collected count == pinned `N_NONBACKTEST`)
   - `NATEX_DATA="/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data" uv run pytest tests/backtests -m backtest -q`
     (expect ≥ 31 passed, 1 xfailed — any new failure blocks the phase)
3. Commit `docs(review): final-review run of record` ; `git push`.
4. CI: `gh run watch` (or poll `gh run list --limit 1`) until the matrix (3.11–3.14) is green.
   Red CI ⇒ fix forward (TDD) and repeat from step 2.
5. Tag move (commits HAVE landed after `v0.1.0`, which pointed at `62c2aa6`):
   `git tag -f v0.1.0 && git push -f origin v0.1.0`.
6. Release note: fetch current body (`gh release view v0.1.0 --json body`), append a short
   section `## Post-release review (2026-07-…)` — one paragraph: full math-audit conformance
   review + API sweep + docs execution + spec completeness completed; N findings, M fixed;
   links to `docs/status/final-review.md` and `docs/status/future_work.md` at `blob/v0.1.0/`;
   then `gh release edit v0.1.0 --notes-file <scratch file>`. Check whether
   `tests/test_release_notes.py` pins `docs/release_notes/v0.1.0.md`; if the release-notes FILE
   is edited too, keep that test green (append-only section mirrors the GitHub edit).
7. Verify: `git status` clean; `git describe --tags` == `v0.1.0` at HEAD; release body shows
   the appended section.

**Acceptance:** all three gate commands pasted with green output; CI run URL recorded in the
status doc; tag on HEAD; release edited.

---

## Task summary (ordered)

1. Commit plan; scaffold `final-review.md` + `future_work.md` (pinning test first).
2. Audit conformance evidence, §2 items 1–10.
3. Audit conformance evidence, §2 items 11–24 + typos block + §3 decisions.
4. API consistency sweep (Generator/NaN/dataclasses/CLI/except/caps).
5. Docs-accuracy execution (README, AGENTS.md, 3 skills, AgentBackend round-trip by hand).
6. Spec completeness matrix + `future_work.md` rows.
7. Fix known nits: `out/agent` stale paths; `discover` help text (test-first).
8. Triage register; fix every confirmed must-fix test-first; resolve all rows.
9. Gates (ruff, pytest, backtests), push, CI green, `git tag -f v0.1.0`, release edit.
