# Phase skills-docs status — agent skills, docs, v0.1.0 release

Date: 2026-07-12. Plan: [docs/plans/phase-skills-docs.md](../plans/phase-skills-docs.md).
Spec gate (design spec repo-layout `skills/` block, §7.4–7.5 lit-review handoff +
human-in-the-loop, §9 phase 8, §10 "non-engineer user" risk): **three symlink-installable
Claude Code agent skills, AGENTS.md for non-Claude agents, CLAUDE.md for agents working
ON natex, README finalization with real pasted CLI output, version 0.1.0 + release
notes — no new statistical code** — met. Core deps unchanged; no new dependency anywhere
in the phase (frontmatter parsed with a regex splitter, `pyproject.toml` with stdlib
`tomllib`).

## What shipped

- **`skills/`** — three agent skills, each a directory whose `SKILL.md` (YAML
  frontmatter `name:` + trigger-phrase `description:`; self-contained body) an agent
  with zero repo context can follow:
  - `discover-natural-experiments` — CSV → `natex study` → `natex discover`, serving the
    file-based guidance protocol (`OUT/guidance/requests/` → `responses/`) as the LLM
    backend yourself.
  - `natex-write-paper` — results bundle → `natex paper`, then verify every number in
    the draft against `results.json` (AI-draft banner rule quoted verbatim).
  - `natex-lit-review` — `natex brief` → deep-research handoff → vet every returned
    citation before merging.
- **`AGENTS.md`** (repo root) — what natex is, install, CLI surface table, the
  AgentBackend file-protocol spec with a worked JSON request/response example, method
  cards + math audit locations, testing conventions.
- **`CLAUDE.md`** (repo root) — conventions for agents working on natex itself (TDD,
  determinism, NaN-never-0.0, no bare except, never commit datasets, backtest marker).
- **README finalization** — coherence pass; "Agent skills" section with install line;
  "Project status" section (phases 1–8 Done, pinned test counts, six-row
  Dataset/Design/Result backtest table); REAL pasted output from `uv run natex datasets`
  and a seeded `uv run natex discover` demo (CSV generated in a scratch dir, only stdout
  pasted — nothing landed in git).
- **CLI alignment (task 1, the only `src/` changes)** — agent-backend default
  request/response dir renamed `OUT/agent` → `OUT/guidance` (two call sites + help
  strings), and `natex brief` added as a thin wrapper around the existing
  `research_brief()` API (mirrors `natex paper` error handling: bad bundle → message +
  exit 2, no traceback).
- **Release artifacts (task 8)** — version bumped `0.1.0.dev0` → `0.1.0` in BOTH
  `pyproject.toml` and `natex.__version__` (pinned equal by `tests/test_version.py`);
  [docs/release_notes/v0.1.0.md](../release_notes/v0.1.0.md) (summary, Methods table
  with a Correction column, six-dataset backtest table, guidance summary, absolute
  `blob/v0.1.0/` links, install lines, PyPI-pending note) pinned by
  `tests/test_release_notes.py`; this status doc.

## Test counts

61 tests added this phase (`test_cli_guidance_dir.py` 4, `test_cli_brief.py` 4,
`test_skills.py` 30, `test_agent_docs.py` 7, `test_readme_release.py` 5,
`test_version.py` 2, `test_release_notes.py` 9), bringing the default suite to 796
collected non-backtest tests; backtests unchanged at 32.

## Run of record

```
$ uv run ruff check src tests
All checks passed!

$ uv run pytest -q
796 passed, 32 deselected in 208.22s (0:03:28)

$ NATEX_DATA=".../RDD/data" uv run pytest tests/backtests -m backtest -q
31 passed, 1 xfailed in 149.89s (0:02:29)
```

## Deviations / decisions log

- **Guidance dir renamed `OUT/agent` → `OUT/guidance`** (task 1). The skills document
  the file protocol for end users, and the old name collided with the `--backend agent`
  flag while saying nothing about what the directory holds. The new name matches the
  rest of the guidance surface (`GuidanceRequest`/`GuidanceResponse`,
  `guidance_log.jsonl`), so the documented contract is self-describing. `--workdir`
  still overrides; pinned by `tests/test_cli_guidance_dir.py` (help text must say
  `OUT/guidance` and must not say `OUT/agent`).
- **`natex brief` is the only new CLI surface.** The lit-review skill must be followable
  CLI-only by a non-engineer; `research_brief()` existed but had no command. The wrapper
  adds zero new logic (core-dep pure text, byte-identical on rerun — pinned by
  `tests/test_cli_brief.py`). Nothing else in `src/` changed this phase.
- **PyPI deferred.** v0.1.0 is a GitHub-only release (`git tag v0.1.0` +
  `gh release create`); the dist name `natex-discovery` is reserved and the publish
  decision is pending. No `uv publish`/twine step was run; the release notes carry the
  install-from-GitHub lines and the PyPI-pending sentence.
