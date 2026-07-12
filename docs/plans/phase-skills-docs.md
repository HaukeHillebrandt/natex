# Phase skills-docs implementation plan — Agent skills, docs, v0.1.0 release

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` — repo-layout
`skills/` block ("agent skills, symlink-installable to `.claude/skills/`: discover-natural-experiments,
natex-write-paper, natex-lit-review; plus AGENTS.md for non-Claude agents"), §7.4–7.5 (lit-review
handoff, human-in-the-loop), §9 phase 8, §10 risk "non-engineer user".
**First action of task 1 is committing this plan file itself**
(`docs: phase skills-docs implementation plan`).

## Phase objective

Agent skills, agent docs, README finalization, and the v0.1.0 GitHub release ONLY — **no new
statistical code**:

1. `skills/` — three Claude Code agent skills, each a directory with a `SKILL.md`
   (YAML frontmatter `name:` kebab-case + one-paragraph `description:` with concrete trigger
   phrases; self-contained step-by-step body an agent with zero repo context can follow):
   `discover-natural-experiments`, `natex-write-paper`, `natex-lit-review`.
2. `AGENTS.md` at repo root — what natex is, install, CLI surface table, the AgentBackend
   file-protocol spec with a worked JSON example, where method cards and the math audit live,
   testing conventions.
3. `CLAUDE.md` at repo root — conventions for agents working ON natex.
4. README finalization — coherence pass, "Agent skills" section, project-status table
   (phases / test counts / backtest table), REAL pasted output from `uv run natex datasets`
   and a small seeded `uv run natex discover` on a synthetic CSV.
5. Release — ruff+pytest green, version bump to 0.1.0, `git tag v0.1.0`, push,
   `gh release create v0.1.0` with the mandated notes. **Do NOT publish to PyPI.**

Two tiny non-statistical code changes are in scope because the skills' documented contract
requires them (task 1): the agent-backend default request/response dir is renamed
`OUT/agent` → `OUT/guidance`, and a thin `natex brief` CLI wrapper around the existing
`research_brief()` API is added. Nothing else in `src/` changes.

## Global constraints (binding, from the phase-1 plan)

- Python >= 3.11. Core deps stay exactly numpy/scipy/pandas/scikit-learn/typer/pydantic.
  **No new dependency anywhere in this phase** — the doc tests parse YAML frontmatter with a
  10-line splitter, NOT pyyaml; `tomllib` (stdlib since 3.11) reads `pyproject.toml`.
  Optional-extra tests keep skipping gracefully; CI on 3.11–3.14 must stay green.
- One `numpy.random.Generator` through every stochastic call (this phase adds no stochastic
  code; the README demo run is seeded `--seed 0` and the bundle fixtures reuse
  `tests/report_helpers.py`).
- Discovery never reads the outcome; NaN never 0.0 on failure; no bare except — the docs and
  skills must *state* these guarantees, and the text-contract tests assert the statements.
- Never commit datasets: the README demo CSV is generated into a temp/scratch dir and only its
  *stdout* is pasted; `.gitignore` already excludes `out/` artifacts — verify nothing lands in git.
- `uv run pytest -q` excludes backtests (`addopts = -m 'not backtest'`); backtests run with
  `-m backtest` and `NATEX_DATA` set. This phase adds no backtests.
- TDD discipline per task: failing test first, implement, `uv run ruff check src tests` +
  `uv run pytest -q` (full suite), conventional commit.

## Current repo state (verified 2026-07-12; interfaces this phase builds on)

- CLI (`src/natex/cli.py`, typer app `natex.cli:app`) — registered commands: `datasets`,
  `fetch-data`, `study`, `discover`, `debias`, `instruments`, `donors`, `paper`. `study`
  (line ~151) and the plan branch of `discover` (`_discover_plan`, line ~198) build the
  guidance backend via `_make_backend(backend, model, workdir if workdir is not None else
  out / "agent")`; help strings say "default OUT/agent" (lines ~157–159, ~313–314).
  `paper` (line ~887) is the pattern for `brief`: `ResultsBundle.load` errors
  `(FileNotFoundError, ValueError, KeyError)` → echo + exit 2, no traceback.
- `natex.llm.agent.AgentBackend(workdir, poll_interval=0.5, timeout=600.0, echo=print)` —
  `complete()` writes `workdir/requests/{seq:04d}_{task}.json` containing the
  `GuidanceRequest` fields **plus** `instructions` (from `TASK_INSTRUCTIONS[task]`) and
  `respond_to` (absolute response path), echoes one instruction line, then polls
  `workdir/responses/{same filename}` until it parses as a JSON object (bare content dict,
  or `{"content": {...}}`) or `timeout` elapses (raises `TimeoutError` naming the paths).
  Restart-safe seq = count of files already in `requests/`.
- `natex.llm.backends` — `TASKS = ("understand", "prepare", "search_plan",
  "interpret_discovery", "audit_assumptions", "review_control_group")`;
  `GuidanceRequest(task, payload, schema_hint)`; `GuidanceResponse(content, raw_text, backend)`;
  `TASK_INSTRUCTIONS` dict; `NullBackend` (deterministic, never vetoes); guidance is advisory
  only — statistics are bitwise identical with and without it; every request+response appended
  to `out/guidance_log.jsonl`.
- `natex.report` — `ResultsBundle.load(dir)` / `.save()`; `render_paper(bundle, format="md",
  out_dir=None)`; `research_brief(bundle, out) -> Path` writes `research-brief.md` (if `out`
  ends in `.md` it writes exactly there, else `out/research-brief.md`); `BRIEF_FILENAME =
  "research-brief.md"` in `natex/report/research_brief.py`;
  `natex.report.paper.BANNER = "AI-generated draft — verify all claims before circulation"`.
- `natex.discover.DiscoverReport.save(out)` → `out/discover_report.json`; plain
  `natex discover` (no `--plan`) writes `out/results.json` with `discoveries` (center values,
  llr, normal, forcing influence), scan `p_value`, placebo/density validation, `effects`
  (`2sls`/`wald` with `tau`, `se`, `ci`, `first_stage_t`, `weak_instrument`), params/seed.
- Data registry (`natex.data.registry.REGISTRY`) keys: `test_score_2012`,
  `academic_probation`, `ed_visits`, `inpatient_visits`, `egger_koethenbuerger`, `prop99`.
  `natex datasets` prints `{name}  found  rows=N  ok=BOOL  path=...` or
  `{name}  missing  rows=?  ok=False  fetch: ...`, always exit 0.
- Docs: method cards `docs/method_cards/{lord3,suddds,dee,iv_sc,llm_analyst}.md`; math audit
  `docs/math_audit_final.md`; status files `docs/status/phase-{2,3,4,5,llm-analyst,report-paper}.md`
  (run-of-record numbers for the README/release backtest table live there).
- Tests: 735 collected non-backtest / 32 backtest (2026-07-12); `tests/report_helpers.py`
  provides seeded `make_rdd_bundle(tmp_path)` / `make_did_bundle(tmp_path)` (core deps only);
  `tests/test_docs.py` is the house pattern for README text-contract tests (flat-whitespace
  matching, section slicing); `tests/conftest.py` already disables typer terminal styling.
- Versioning: `pyproject.toml` `version = "0.1.0.dev0"` AND `src/natex/__init__.py`
  `__version__ = "0.1.0.dev0"` (duplicated — both must be bumped, task 8 pins them equal).
- Git: remote `git@github.com:HaukeHillebrandt/natex.git`, branch master, no tags yet.
  CI `.github/workflows/ci.yml`: ruff + pytest on 3.11/3.12/3.13/3.14 with
  `--extra dev --extra plot --extra report`.

## Audit corrections that bind this phase (they are TEXT requirements here)

No new math is implemented, but every skill/doc that describes results must state the audited
semantics honestly, and the text-contract tests assert the exact phrases:

- **Randomization test** is a *fitted-null Monte Carlo* (parametric bootstrap), NOT exact;
  p-values are +1-rank. Skills must say "fitted-null Monte Carlo" and must not call it exact.
- **Honest split** — discovery-vs-estimation splits exist (`honest_split`) and drafts/effects
  must be caveated accordingly.
- **Weak-IV** — `weak_instrument` flag and `first_stage_t` are always on; skills must tell the
  agent to surface them, and to report `ci` honestly (AR/Fieller sets may be
  `disjoint`/`unbounded`/`empty` in the IV pipeline — never coerced).
- **Density test** is a falsification test only, on the signed distance along the *frozen*
  discovered normal.
- **Placebo battery** is intercept-continuity with Holm correction (not side-means).
- **NaN never 0.0** — a NaN/`null`/"—" in results means *failed/underpowered*, and skills must
  forbid the agent from replacing it with a number.
- Guidance (all backends incl. the agent protocol) **proposes and vetoes but never fabricates
  statistics**; a veto is a flag only. Skills serving the protocol must repeat this contract.

---

## Task 1 — Commit the plan; align the CLI with the documented protocol (`OUT/guidance` + `natex brief`)

**First action: commit this plan file** — `git add docs/plans/phase-skills-docs.md &&
git commit -m "docs: phase skills-docs implementation plan"`.

The skills (tasks 2–4) document two contracts that must be true in code first:
the guidance request/response dir is `OUT/guidance/…`, and the research brief has a CLI.
Both are non-statistical plumbing; nothing statistical changes.

### Tests first

**Create `tests/test_cli_guidance_dir.py`** (new):

- Fixture `FakeAgentBackend` — records `workdir` at construction, `name = "agent"`, and
  `complete(request)` delegates to `natex.llm.backends.NullBackend().complete(request)` (valid
  schema content, deterministic, no polling). Monkeypatch `natex.cli.AgentBackend` to it.
- `test_study_agent_default_workdir_is_out_guidance(tmp_path, monkeypatch)` — write a tiny
  csv (20 rows, cols `T` binary, `y`, `x` numeric via pandas, no rng needed), run
  `CliRunner().invoke(app, ["study", str(csv), "--backend", "agent", "--out", str(out)])`;
  assert `exit_code == 0` and the recorded `workdir == out / "guidance"`.
- `test_study_agent_workdir_flag_overrides(tmp_path, monkeypatch)` — same with
  `--workdir custom`; recorded `workdir == custom`, and `out / "guidance"` was never created.
- `test_discover_plan_agent_default_workdir(tmp_path, monkeypatch)` — run `study` with
  `--backend null` to get `out/intake_report.json`, then
  `invoke(app, ["discover", str(csv), "--plan", str(out/"intake_report.json"), "--backend",
  "agent", "--out", str(out2)])` with the fake; recorded `workdir == out2 / "guidance"`.
- `test_help_text_names_guidance_dir()` — `invoke(app, ["study", "--help"])` output contains
  `OUT/guidance` and not `OUT/agent` (same for `discover --help`).

**Create `tests/test_cli_brief.py`** (new; uses `tests/report_helpers.make_rdd_bundle`):

- `test_brief_writes_default_path(tmp_path)` — `make_rdd_bundle(tmp_path)`, then
  `invoke(app, ["brief", "--bundle", str(tmp_path)])`; assert exit 0, stdout contains
  `research-brief.md`, and `(tmp_path / "research-brief.md").exists()` with content containing
  `"## Literature questions for deep research"` (real section emitted by `_brief_text`).
- `test_brief_out_md_writes_exactly_there(tmp_path)` — `--out tmp_path/"sub"/"my-brief.md"`;
  file exists at exactly that path.
- `test_brief_reruns_byte_identical(tmp_path)` — run twice, read bytes, assert equal
  (pins the documented determinism the lit-review skill relies on).
- `test_brief_missing_bundle_exits_2(tmp_path)` — `--bundle tmp_path/"nope"`; exit code 2,
  no traceback in output (`"Traceback" not in result.output`).

Run: the new tests fail (`brief` unknown command; workdir default is `out/agent`).

### Implement

**Modify `src/natex/cli.py`:**

1. In `study` and `_discover_plan`: `out / "agent"` → `out / "guidance"` (two call sites,
   lines ~172 and ~250). Help strings: `"agent-backend request/response dir; default
   OUT/agent"` → `... default OUT/guidance` (both `workdir` options, `study` and `discover`).
2. New command, appended after `paper` (mirror its error handling exactly):

```python
@app.command()
def brief(
    bundle: Path = typer.Option(..., "--bundle",
        help="results bundle dir (ResultsBundle.save, or a discover --out dir)"),
    out: Path = typer.Option(None,
        help="output dir or .md path; default BUNDLE/research-brief.md"),
):
    """Write the deep-research handoff brief (research-brief.md) from a results bundle."""
    try:
        loaded = ResultsBundle.load(bundle)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    path = research_brief(loaded, out if out is not None else bundle)
    typer.echo(f"brief: {path}")
    typer.echo("hand this file to your deep-research tooling; verify everything it returns")
```

   Import `research_brief` alongside the existing `ResultsBundle`/`render_paper` imports
   (`from natex.report import research_brief` — it is core-dep pure text, no extra needed).

3. **Modify `README.md`** minimally (full README pass is task 7 — this keeps the repo truthful
   at this commit): the one sentence "`--backend agent` writes each question as a JSON file
   under `out/agent/requests/`" → `out/guidance/requests/`.

### Verify + commit

`uv run ruff check src tests` && `uv run pytest -q` (full suite green; the report/figure tests
and `test_cli_study.py` don't reference the old default — verified by grep, but the full run is
the gate). Commit:
`feat(cli): guidance-dir default OUT/guidance and natex brief command` .

---

## Task 2 — Skills test harness + `skills/discover-natural-experiments/SKILL.md`

### Tests first

**Create `tests/test_skills.py`** with a dependency-free frontmatter helper used by tasks 2–4:

```python
ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"

def frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Parse leading '---' YAML block: single-line 'key: value' pairs only.
    Returns (meta, body). Raises AssertionError on malformed frontmatter."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path}: missing frontmatter"
    end = text.index("\n---", 4)
    meta = {}
    for line in text[4:end].splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, text[end + 4:]
```

(House rule for authoring: frontmatter is exactly two single-line keys, `name` and
`description` — long description stays on ONE line so the parser stays trivial.)

Parametrized contract tests over all three skills (they will fail per-skill until tasks 2–4
land; use `pytest.mark.parametrize` with the three dir names and let tasks 3–4 un-fail their
rows — OR structure as three modules; **choose**: one module, and tasks 3–4 each un-skip their
parameter by creating the file. Simplest honest TDD: write the full parametrized module now;
task 2 makes the `discover-natural-experiments` rows pass; tasks 3–4 make the rest pass; the
suite is only fully green after task 4 — NOT acceptable under "full suite green per task".
**Therefore**: parametrize via a module-level list `SKILL_DIRS` that starts with only
`"discover-natural-experiments"`; tasks 3 and 4 each append their dir name as their
failing-test-first step.)

Shared assertions, for every dir in `SKILL_DIRS`:

- `test_skill_file_exists` — `SKILLS / d / "SKILL.md"` exists.
- `test_frontmatter_name_matches_dir` — `meta["name"] == d` and matches `^[a-z0-9]+(-[a-z0-9]+)*$`.
- `test_description_is_one_paragraph_with_triggers` — `meta["description"]` non-empty, no `\n`,
  length > 100 chars.
- `test_safety_warnings_present` — body (whitespace-flattened, case-insensitive) contains all
  of: `"never fabricate"`, `"validation battery"`, `"ai-generated"`, and `"verify"`.
- `test_only_real_cli_commands_are_taught` — regex-extract every `natex <word>` token from
  fenced code blocks; assert each `<word>` is in the set of registered commands obtained from
  `CliRunner().invoke(app, ["--help"]).output` (i.e. datasets, fetch-data, study, discover,
  debias, instruments, donors, paper, brief). Catches typo'd/imagined commands forever.

Skill-specific tests (this task):

- `test_discover_skill_triggers` — description contains the concrete phrases
  `"find natural experiments in this dataset"` and `"discover RDDs"`.
- `test_discover_skill_install_paths` — body contains both
  `uv add git+https://github.com/HaukeHillebrandt/natex` and `uv sync`.
- `test_discover_skill_serves_protocol` — flattened body contains, in order of the workflow:
  `--backend agent`, `guidance/requests`, `guidance/responses`, `schema_hint`, `respond_to`,
  `intake_report.json`, `natex discover --plan`, `results.json`. Also contains every task name
  in `natex.llm.backends.TASKS` (import it — docs can then never drift from the protocol).
- `test_discover_skill_example_json_parses` — extract fenced ```json blocks; assert >= 2
  (one request, one response); `json.loads` each; the request example has keys
  `{"task", "payload", "schema_hint", "instructions", "respond_to"}`.
- `test_discover_skill_states_honest_inference` — flattened body contains
  `"fitted-null Monte Carlo"`, `"not exact"` (or `"parametric bootstrap"` — pick and pin
  `"fitted-null Monte Carlo"` + `"+1-rank"`), `"weak_instrument"`, `"discovery"` +
  `"estimation"` split mention (`"honest split"`), and `"advisory"` (guidance never alters
  statistics).

### Implement

**Create `skills/discover-natural-experiments/SKILL.md`.** Frontmatter:

```markdown
---
name: discover-natural-experiments
description: Run natex to find and validate natural experiments (RDDs, DiDs, IVs, synthetic-control donors) in a tabular dataset, serving natex's file-based guidance protocol yourself as the LLM backend. Use when the user says things like "find natural experiments in this dataset", "discover RDDs", "is there a discontinuity in this data", "run natex on this CSV", or "what quasi-experiments are hiding in here". Covers install, the natex study analyst pass, answering guidance request files, natex discover --plan, and honest interpretation of results.json.
---
```

Body sections (self-contained; an agent with no repo context can follow it):

1. **What natex is** (two sentences + the three hard guarantees: discovery never reads the
   outcome; guidance is advisory only — statistics identical with/without it; NaN/null means
   failure, never zero).
2. **Install** — `uv add git+https://github.com/HaukeHillebrandt/natex` into the user's
   project, or clone + `uv sync` from a checkout; Python >= 3.11; verify with
   `uv run natex --help`.
3. **Run the analyst pass** —
   `uv run natex study data.csv --context "<where the data came from>" --backend agent --seed 0 --out out/`.
   Explain: it profiles the CSV and asks guidance questions **by writing files** — the command
   BLOCKS (default 600 s per question) until each is answered.
4. **Serve the AgentBackend protocol (you ARE the LLM backend)** — the core of the skill:
   - Run `natex study` in the background (or a second terminal). Watch
     `out/guidance/requests/*.json` (poll; files are named `{seq:04d}_{task}.json`, e.g.
     `0000_understand.json`).
   - For each request file: read it; it contains `task` (one of `understand`, `prepare`,
     `search_plan`, `interpret_discovery`, `audit_assumptions`, `review_control_group`),
     `payload` (the data profile / discovery summary to reason over), `schema_hint`
     (JSON schema your answer must satisfy), `instructions` (what to do), and `respond_to`
     (the exact absolute path to answer at).
   - **Compose the answer JSON yourself** — you are the intelligence here; reason over the
     payload and produce a JSON object matching `schema_hint`. Never invent columns that
     aren't in the payload; never fabricate statistics; you may flag/veto, which is recorded
     as advisory only.
   - Write it to `out/guidance/responses/<same filename>` (equal to `respond_to`). Write
     atomically-ish: full valid JSON in one write (a partial file is re-polled, a non-object
     is an error).
   - Include the worked example: one real-shape request (fenced ```json with all five keys,
     an `understand` task with a two-column profile payload) and its response
     (```json object matching the schema hint). Note the fallback: on timeout, natex tells
     you the paths; `--backend null` re-runs fully offline with heuristics.
5. **Run discovery** —
   `uv run natex discover --plan out/intake_report.json --seed 0 --out out/`
   (same protocol serving applies if `--backend agent` is passed here too; ranked candidates
   scan first, the exhaustive remainder within budget; coverage — scanned / skipped_budget /
   failed / invalid — is always recorded, never silently dropped).
6. **Interpret the results for the user** — read `out/discover_report.json` (plan mode) or
   `out/results.json` (plain mode). Report a discoveries table (design, location/center,
   forcing influence, LLR), the scan p-value, the validation battery (randomization, placebo
   with Holm correction, density), and effects (2SLS + Wald: `tau`, `se`, `ci`,
   `first_stage_t`, `weak_instrument`). **Honest-inference caveats verbatim in the skill**:
   p-values are fitted-null Monte Carlo with +1-rank correction — not exact; prefer the
   honest split (discovery vs estimation halves) before headline claims; always surface
   `weak_instrument`; the density test is a falsification check on the frozen discovered
   normal only; `null`/`NaN`/"—" means the computation failed or was underpowered — report
   it as such, NEVER substitute a number.
7. **Warnings** (bulleted, mirrored by the tests): never fabricate statistics or columns;
   every discovery needs the validation battery before being called a finding; anything
   drafted downstream is AI-generated and must be verified by a human.

### Verify + commit

Full suite green. Commit:
`feat(skills): discover-natural-experiments agent skill + skills text-contract tests`.

---

## Task 3 — `skills/natex-write-paper/SKILL.md`

### Tests first (extend `tests/test_skills.py`)

- Append `"natex-write-paper"` to `SKILL_DIRS` (shared contract rows now fail).
- `test_paper_skill_triggers` — description contains `"write up the discovery as a paper"`.
- `test_paper_skill_commands` — body contains `natex paper --bundle`, `--format md`,
  `--format latex`, and `tectonic`.
- `test_paper_skill_quotes_real_banner` — flattened body contains
  `natex.report.paper.BANNER` verbatim (imported, like `tests/test_docs.py` does — the skill
  can never drift from the code).
- `test_paper_skill_google_docs_route_is_manual` — flattened body contains `"Google Docs"`,
  `"Google Drive"`, and (case-insensitive) `"does not integrate"`.
- `test_paper_skill_review_requirements` — flattened body contains `"results.json"` and
  `"every number"` (the check-numbers-against-the-bundle instruction).

### Implement

**Create `skills/natex-write-paper/SKILL.md`.** Frontmatter `name: natex-write-paper`;
description one line with triggers: "write up the discovery as a paper", "draft a paper from
the natex results", "turn this discovery into a manuscript", "render the natex paper".

Body:

1. **Prerequisite** — a finished bundle dir (`natex discover --out OUT`, ideally via the
   discover-natural-experiments skill); install `uv add 'natex-discovery[report]'` (jinja2)
   and optionally `[plot]` for figures.
2. **Render** — `uv run natex paper --bundle OUT --format md` (always works) →
   `OUT/paper/paper.md`; `--format latex` → `paper.tex`, compiled to `paper.pdf` only when
   `tectonic` is on PATH (install note + link; a missing compiler leaves the `.tex` with a
   message, never an error).
3. **The AI-draft banner is non-negotiable** — every draft opens with
   "AI-generated draft — verify all claims before circulation" (quote verbatim). Walk the
   user through it: check every number in the draft against `OUT/results.json` (the single
   source every rendered number comes from), read the validation section skeptically, missing
   values render as "—" and must stay that way — never fill one in.
4. **Google Docs (manual route)** — natex does not integrate the Google Docs API: render md,
   then either paste `paper.md` into a new Doc or upload the `.md` to Google Drive and
   "Open with → Google Docs".
5. **Warnings** — never fabricate statistics; the draft covers only validated discoveries
   (validation battery); AI-generated: the user must verify before sharing/submission.

### Verify + commit

Full suite green. Commit: `feat(skills): natex-write-paper agent skill`.

---

## Task 4 — `skills/natex-lit-review/SKILL.md`

### Tests first (extend `tests/test_skills.py`)

- Append `"natex-lit-review"` to `SKILL_DIRS`.
- `test_lit_skill_triggers` — description contains `"literature review for my discovery"`.
- `test_lit_skill_brief_generation` — body contains `natex brief --bundle` AND the Python
  fallback `research_brief` (both routes documented); contains `research-brief.md`.
- `test_lit_skill_handoff_and_merge` — flattened body (case-insensitive) contains
  `"deep research"` (or `deep-research`), `"related work"`, and `"citation"`.
- `test_lit_skill_vetting_warning` — flattened body contains `"verify"` and a phrase pinning
  citation hygiene: `"never fabricate"` (shared) plus `"every citation"`.

### Implement

**Create `skills/natex-lit-review/SKILL.md`.** Frontmatter `name: natex-lit-review`;
description triggers: "literature review for my discovery", "find related work for this
natural experiment", "what papers relate to this RDD", "deep research on my natex results".

Body:

1. **Generate the brief** — `uv run natex brief --bundle OUT` → `OUT/research-brief.md`
   (deterministic, byte-identical on rerun; pure text from `results.json`). Python API
   alternative: `from natex.report import ResultsBundle, research_brief;
   research_brief(ResultsBundle.load("OUT"), "OUT")`.
2. **What the brief contains** — data context, discovered designs, effect estimates,
   validation status, and numbered literature questions, formatted to be pasted verbatim
   into a deep-research agent.
3. **Hand off** — give `research-brief.md` to the user's deep-research tooling (e.g. a Gemini
   deep-research skill / Deep Research query). natex performs no research calls itself; the
   handoff is a text file.
4. **Merge results back** — take the returned review: verify every citation actually exists
   (resolve DOI/title; drop anything unverifiable — never fabricate a reference), select the
   genuinely related work, then edit the related-work section of `OUT/paper/paper.md`
   (or `.tex`) to weave the vetted citations in. Keep the AI-draft banner; re-check that no
   statistic in the draft changed.
5. **Warnings** — deep-research output is itself AI-generated: verify before merging; the
   validation battery, not the literature, decides whether a discovery stands.

### Verify + commit

Full suite green. Commit: `feat(skills): natex-lit-review agent skill`.

---

## Task 5 — `AGENTS.md` (repo root)

### Tests first

**Create `tests/test_agent_docs.py`** (covers AGENTS.md here; CLAUDE.md rows in task 6):

- `test_agents_md_exists_and_opens_with_what_natex_is` — file exists; first non-title
  paragraph mentions (flattened) `"natural experiment"` and `"discovery"`.
- `test_cli_table_covers_every_command` — for every command name parsed from
  `CliRunner().invoke(app, ["--help"]).output` (datasets, fetch-data, study, discover,
  debias, instruments, donors, paper, brief): assert `f"`{name}`" in AGENTS` (each command
  appears in backticks in the table). Reverse drift guard: every `natex <word>` in fenced
  blocks is a registered command (reuse the helper from `tests/test_skills.py` — move that
  helper into `tests/doc_helpers.py` in THIS task and import it from both test modules;
  plain helper module, not a conftest).
- `test_protocol_spec_with_parsing_example` — flattened text contains `guidance/requests`,
  `guidance/responses`, `schema_hint`, `respond_to`, `{seq:04d}_{task}.json` (or the literal
  `0000_understand.json`), and every task name in `natex.llm.backends.TASKS`; extract fenced
  ```json blocks, assert >= 2, `json.loads` each, request example has the five keys.
- `test_pointers_exist_on_disk` — `docs/method_cards` and `docs/math_audit_final.md` are
  referenced in the text AND exist; every `docs/method_cards/*.md` filename referenced exists.
- `test_testing_conventions` — flattened text contains `uv run pytest -q`, `-m backtest`,
  `NATEX_DATA`, and `never commit` (datasets rule).

### Implement

**Create `AGENTS.md`** with sections:

1. **What natex is** — one paragraph (automated natural-experiment discovery: LoRD3 scan for
   RDDs, SuDDDS for DiDs, DEE debiasing, IV search, SC donors; corrected per the math audit;
   discovery never reads the outcome; seeded determinism; NaN never 0.0).
2. **Install** — `uv add git+https://github.com/HaukeHillebrandt/natex` / clone + `uv sync
   --extra dev`; extras table one-liner (plot/report/paperbanana/ml/gp/llm).
3. **CLI surface** — markdown table, one row per command × (purpose, key inputs, output
   files): `datasets`, `fetch-data`, `study`, `discover`, `debias`, `instruments`, `donors`,
   `paper`, `brief`.
4. **The AgentBackend file protocol** — normative spec: request path
   `WORKDIR/requests/{seq:04d}_{task}.json` (WORKDIR defaults to `OUT/guidance`), request
   JSON shape (`task` ∈ the six tasks, `payload`, `schema_hint`, `instructions`,
   `respond_to`), response file at `WORKDIR/responses/<same filename>` = a JSON object
   (bare content, or wrapped `{"content": {...}}`), 0.5 s poll / 600 s default timeout,
   restart-safe sequencing, everything logged to `OUT/guidance_log.jsonl`; the advisory-only
   contract (propose/veto, never fabricate; statistics bitwise identical with and without
   guidance). One worked request+response ```json example pair.
5. **Where things live** — `docs/method_cards/lord3.md`, `suddds.md`, `dee.md`, `iv_sc.md`,
   `llm_analyst.md` (one line each); `docs/math_audit_final.md` **governs any math conflict**;
   `skills/` (the three skills); `docs/status/` (run-of-record numbers);
   `docs/plans/` (implementation history).
6. **Testing conventions** — `uv run pytest -q` (backtests deselected via `addopts`);
   `uv run pytest tests/backtests -m backtest -q` with `NATEX_DATA` pointing at the local
   data root; datasets are never committed; `uv run ruff check src tests` (line-length 100);
   CI = 3.11–3.14.

### Verify + commit

Full suite green. Commit: `docs: AGENTS.md — CLI surface, guidance file protocol, conventions`.

---

## Task 6 — `CLAUDE.md` (repo root)

### Tests first (extend `tests/test_agent_docs.py`)

`test_claude_md_conventions` — CLAUDE.md exists; flattened text contains ALL of:
`"uv "` (uv-first tooling), `"line-length 100"` (or `line length 100` — pin the exact
pyproject spelling `line-length 100`), `"uv run pytest -q"`, `"-m backtest"`, `"NATEX_DATA"`,
`"never commit"` (datasets), `"numpy.random.Generator"`, `"NaN"` + `"0.0"` (never-zero rule),
`"math_audit_final.md"` + `"governs"`, and `"conventional commit"`.

`test_claude_md_is_short` — under 60 lines (it's a conventions card, not a manual; keeps it
load-bearing for agents).

### Implement

**Create `CLAUDE.md`** — terse bullets for agents working ON natex:

- Tooling: uv only (`uv sync --extra dev`, `uv run …`); ruff `line-length 100`
  (`uv run ruff check src tests`).
- Tests: `uv run pytest -q` excludes backtests by default; real-data backtests:
  `NATEX_DATA=<data root> uv run pytest tests/backtests -m backtest -q`. Never commit
  datasets (or any file derived from them).
- Statistics house rules: one `numpy.random.Generator` threaded through every stochastic
  call (no fresh generators mid-pipeline, no global seeding); discovery never reads the
  outcome; failed computations return `NaN`, never `0.0`; no bare `except`.
- `docs/math_audit_final.md` governs whenever code, papers, or docs disagree about math.
- Conventional commits after every green ruff+pytest cycle.
- Pointers: `AGENTS.md` (CLI + protocol), `docs/plans/` (how each phase was built).

### Verify + commit

Full suite green. Commit: `docs: CLAUDE.md conventions for agents working on natex`.

---

## Task 7 — README finalization (coherence pass, Agent skills section, status table, REAL output)

### Generate the real output FIRST (not committed as data — text only)

1. `cd /Users/haukehillebrandt/dev/natex && NATEX_DATA="/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data" uv run natex datasets`
   — capture stdout verbatim (6 lines). If any dataset is missing locally that's fine — the
   output is genuine either way.
2. Demo discover, in a scratch dir (never inside the repo; nothing gets committed):

```bash
uv run python - <<'EOF'
import numpy as np
from natex.data.synthetic import make_synthetic
ds, _ = make_synthetic(n=500, zeta=6.0, kind="binary", rng=np.random.default_rng(0))
ds.df.to_csv("/tmp-scratch-path/synth.csv", index=False)
EOF
uv run natex discover "/tmp-scratch-path/synth.csv" --treatment T --outcome y \
  --k 40 --q 49 --seed 0 --out "/tmp-scratch-path/out"
```

   Capture stdout verbatim; truncate long lines with a trailing `…` marker if needed, keeping
   the `results:` line and the discovery/p-value lines intact. Paste BOTH the generating
   snippet and the genuine output block into the README (labelled "output, truncated" where
   truncated). If the exact `make_synthetic` signature differs, adapt the snippet to the real
   one — the README snippet must be copy-paste-runnable.

### Tests first

**Create `tests/test_readme_release.py`** (keep `tests/test_docs.py` untouched — it still
guards the phase-7 section):

- `test_agent_skills_section` — README has `## Agent skills`; it names all three skill dirs
  and links `skills/<name>/SKILL.md` for each; each linked file exists; section contains a
  one-line description per skill (assert each skill name is followed within its bullet by
  `—` or `:` and non-empty text).
- `test_project_status_table` — README has a `## Project status` section containing: a row
  (regex) per phase 1–8 each marked `Done`; the literal non-backtest/backtest collected test
  counts (implementer inserts the true numbers from a fresh `uv run pytest --collect-only -q`
  run at authoring time; the test pins the SAME numbers via a module constant so drift is a
  conscious two-line edit); and a backtest table whose header matches
  `| Dataset | Design | Result |` with one row per registry key — assert all six keys from
  `natex.data.registry.REGISTRY` appear in the section (imported, never hardcoded).
- `test_real_output_pasted` — README contains a fenced block with `uv run natex datasets`
  followed by a block containing at least one `rows=` and one of `found`/`missing`; and a
  fenced block with `natex discover` demo output containing `results:` and (`llr` or
  `p=`) — text drawn from the actual CLI echo format.
- `test_quickstart_coherence` — `guidance/requests` present; `out/agent` absent everywhere;
  `natex brief` appears in the "From discovery to paper" section; roadmap row 8 matches
  `\|\s*8\s*\|\s*\*\*Done\*\*`; the word "Next:" no longer introduces the roadmap table.
- `test_skills_install_line` — the Agent skills section shows how to install the skills into
  Claude Code (a `.claude/skills` symlink or copy line, per the design spec).

### Implement (README edits)

1. **Coherence pass** — read top-to-bottom; fix anything stale: the roadmap intro sentence,
   phase-8 row → `**Done** — agent skills (skills/), AGENTS.md + CLAUDE.md, v0.1.0 release
   ([status](docs/status/phase-skills-docs.md))` (status file lands in task 8 — write the
   link now, the file exists before the README ships in the release; test for its existence
   lives in task 8), `natex brief` added to the paper-flow command list, guidance-dir wording
   already fixed in task 1.
2. **New `## Agent skills` section** (after "From discovery to paper"): three bullets —
   `[discover-natural-experiments](skills/discover-natural-experiments/SKILL.md)` — find and
   validate natural experiments in a CSV, serving the guidance protocol;
   `[natex-write-paper](skills/natex-write-paper/SKILL.md)` — render and human-verify the
   AI-draft paper; `[natex-lit-review](skills/natex-lit-review/SKILL.md)` — deep-research
   handoff and citation merge. Install line:
   `ln -s "$(pwd)/skills/"*/ ~/.claude/skills/` (or copy the directories), plus one sentence
   that AGENTS.md documents the same surface for non-Claude agents.
3. **New `## Project status` section** (before Development): phases table 1–8 all Done with
   one-line scope + status links; a sentence with the real test counts ("`uv run pytest -q`:
   N passed; `-m backtest`: M backtests over six datasets", numbers from the fresh run); the
   backtest table `| Dataset | Design | Result |` — six rows sourced from
   `docs/status/phase-2.md` (five RDD rows: cutoffs recovered, τ in CI, forcing ranked #1,
   ages 19/23, age-23 small-n, ≥2 statutory thresholds) and phase-3/phase-5 (prop99 DiD:
   (California, 1989) recovered, Table 6.1-consistent signs; prop99 SC donors: ADH pool,
   weight 0.955 on ADH's five donors, ATT −19.5).
4. **Paste the real output blocks** into Quickstart (datasets block where `natex datasets`
   is introduced; the demo discover block at the end of the CLI quickstart, with its
   generating snippet).

### Verify + commit

Full suite green (includes `tests/test_docs.py` still passing — the phase-7 section must
survive the pass). Commit: `docs: README agent-skills + project-status sections with real CLI output`.

---

## Task 8 — Version 0.1.0, release notes, phase status doc

### Tests first

**Create `tests/test_version.py`:**

- `test_version_synced` — `tomllib.loads((ROOT / "pyproject.toml").read_text())` →
  `project.version` equals `natex.__version__`.
- `test_version_is_release` — `re.fullmatch(r"\d+\.\d+\.\d+", natex.__version__)` (no dev
  suffix; stays valid for future releases).

**Create `tests/test_release_notes.py`** — `docs/release_notes/v0.1.0.md`:

- exists; contains a summary paragraph before the first `##`;
- methods table containing all of `LoRD3`, `SuDDDS`, `DEE`, `IV`, `SC` (or
  `synthetic control`) AND a corrections column (assert header row contains `Correction`);
- backtest table: all six `REGISTRY` keys appear (import, don't hardcode);
- guidance summary: contains `null`, `agent`, `anthropic`, `gemini`, and `advisory`;
- links: contains `docs/method_cards`, `docs/math_audit_final.md`, and all three
  `skills/<name>/SKILL.md` paths, each as an ABSOLUTE URL prefixed
  `https://github.com/HaukeHillebrandt/natex/blob/v0.1.0/` (release notes render outside the
  repo — relative links would be dead);
- install: contains `uv add git+https://github.com/HaukeHillebrandt/natex`;
- PyPI note: contains `not yet on PyPI` and `natex-discovery` and `pending`.

Also extend `tests/test_readme_release.py` (or here): `docs/status/phase-skills-docs.md`
exists and contains `uv run ruff check src tests` and `uv run pytest -q` (run-of-record, same
contract `tests/test_docs.py` enforces for phase 7).

### Implement

1. `pyproject.toml`: `version = "0.1.0"`. `src/natex/__init__.py`: `__version__ = "0.1.0"`.
2. **Create `docs/release_notes/v0.1.0.md`** (this exact file is passed to
   `gh release create --notes-file`):
   - Title line + summary paragraph: what natex v0.1.0 does end to end (CSV in → analyst
     pass → LoRD3/SuDDDS discovery → validation battery → honest effect estimation →
     figures + AI-draft paper), corrected against the math audit, deterministic under one
     seeded Generator.
   - `## Methods` table: rows LoRD3 (RDD scan) / SuDDDS (DiD scan) / DEE (debiased CATE) /
     IV search (BCCH Lasso + honest 2SLS/AR/J) / SC donors (ADH + in-space placebo), columns
     `Method | What it does | Correction vs the papers` (corrections condensed from the
     README "Corrections vs the papers" list: honest fitted-null MC p-values, Bernoulli(p̂)
     replicas, intercept-continuity placebos, frozen side-indicator 2SLS + HC1, own-k
     variance floor, boundary suprema, tie convention, frozen-normal density test).
   - `## Backtests` — the same six-row Dataset/Design/Result table as the README.
   - `## LLM analyst + guidance` — one paragraph: `natex study` Stage-0 pass; four backends
     (null deterministic default / agent file protocol / anthropic / gemini via `[llm]`);
     advisory-only guarantee; `guidance_log.jsonl`.
   - `## Docs & skills` — absolute `blob/v0.1.0/` links: three skills, method cards dir,
     math audit.
   - `## Install` — the uv lines + extras one-liner + Python >= 3.11.
   - `## PyPI` — "not yet on PyPI; the dist name `natex-discovery` is reserved — publish
     decision pending. Install from GitHub (above)."
3. **Create `docs/status/phase-skills-docs.md`** — what shipped (skills, AGENTS.md,
   CLAUDE.md, README sections, guidance-dir rename, `natex brief`, version bump), the
   run-of-record command outputs (`uv run ruff check src tests`, `uv run pytest -q` with the
   final counts, and — if run locally — the `-m backtest` line), deviations/decisions log
   (explicitly record: default workdir renamed OUT/agent → OUT/guidance and why; `natex
   brief` added as the only new CLI surface; PyPI deferred).

### Verify + commit

Full suite green. Commit: `chore(release): v0.1.0 version bump, release notes, phase status`.

---

## Task 9 — Green gate, tag, GitHub release (NO PyPI)

No new tests — this task is executed verification. Every step's output must be observed
before the next (verification-before-completion):

1. Fresh gate at HEAD: `uv run ruff check src tests` → "All checks passed!";
   `uv run pytest -q` → all green (record the final counts; update
   `docs/status/phase-skills-docs.md` + the README status sentence if the numbers moved,
   amend-free: a follow-up `docs:` commit).
   Optionally (local only, not gating): `NATEX_DATA=".../RDD/data" uv run pytest
   tests/backtests -m backtest -q`.
2. Push master: `git push origin master`. Watch CI green on ALL of 3.11/3.12/3.13/3.14:
   `gh run watch` (or `gh run list --limit 1` until `completed success`). Do not tag before
   CI is green.
3. Tag + push tag: `git tag v0.1.0 && git push origin v0.1.0`.
4. Release: `gh release create v0.1.0 --title 'natex v0.1.0 — automated natural-experiment
   discovery' --notes-file docs/release_notes/v0.1.0.md`.
5. Verify: `gh release view v0.1.0 --web`-free check via `gh release view v0.1.0` — title,
   notes body renders the tables, tag points at the gated commit
   (`git rev-parse v0.1.0 == git rev-parse origin/master`).
6. **Do NOT run any PyPI/`uv publish`/twine step.** The release notes' PyPI-pending sentence
   is the only PyPI artifact of this phase.

Done criteria for the phase: three skills exist and pass their text contracts; AGENTS.md and
CLAUDE.md pass theirs; README carries the Agent-skills + Project-status sections with genuine
pasted output; `v0.1.0` tag and GitHub release exist with the mandated notes; suite green on
3.11–3.14; zero datasets committed.

---

## Task order & commit messages (summary)

| # | Commit | Files created/modified |
|---|--------|------------------------|
| 1 | `docs: phase skills-docs implementation plan` then `feat(cli): guidance-dir default OUT/guidance and natex brief command` | `docs/plans/phase-skills-docs.md`; `src/natex/cli.py`; `README.md` (one line); `tests/test_cli_guidance_dir.py`; `tests/test_cli_brief.py` |
| 2 | `feat(skills): discover-natural-experiments agent skill + skills text-contract tests` | `skills/discover-natural-experiments/SKILL.md`; `tests/test_skills.py` |
| 3 | `feat(skills): natex-write-paper agent skill` | `skills/natex-write-paper/SKILL.md`; `tests/test_skills.py` |
| 4 | `feat(skills): natex-lit-review agent skill` | `skills/natex-lit-review/SKILL.md`; `tests/test_skills.py` |
| 5 | `docs: AGENTS.md — CLI surface, guidance file protocol, conventions` | `AGENTS.md`; `tests/test_agent_docs.py`; `tests/doc_helpers.py` |
| 6 | `docs: CLAUDE.md conventions for agents working on natex` | `CLAUDE.md`; `tests/test_agent_docs.py` |
| 7 | `docs: README agent-skills + project-status sections with real CLI output` | `README.md`; `tests/test_readme_release.py` |
| 8 | `chore(release): v0.1.0 version bump, release notes, phase status` | `pyproject.toml`; `src/natex/__init__.py`; `docs/release_notes/v0.1.0.md`; `docs/status/phase-skills-docs.md`; `tests/test_version.py`; `tests/test_release_notes.py` |
| 9 | (no code commit; tag `v0.1.0` + GitHub release; follow-up `docs:` commit only if final counts moved) | git tag, GitHub release |
