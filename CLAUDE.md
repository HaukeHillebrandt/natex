# CLAUDE.md — conventions for agents working ON natex

Terse card. What natex is, the CLI surface table, and the guidance file protocol live in
[AGENTS.md](AGENTS.md). How each phase was built (plans, task breakdowns) lives in `docs/plans/`.

## Tooling

- uv only: `uv sync --extra dev` once, then `uv run <cmd>` for everything.
- Lint: ruff, line-length 100 — `uv run ruff check src tests`.
- Python >= 3.11. Core deps stay exactly numpy/scipy/pandas/scikit-learn/typer/pydantic;
  everything else is an optional extra whose tests skip gracefully when it is missing.

## Tests

- `uv run pytest -q` runs the full suite and excludes backtests by default
  (`addopts = -m 'not backtest'` in `pyproject.toml`).
- Real-data backtests: `NATEX_DATA=<data root> uv run pytest tests/backtests -m backtest -q`.
- Datasets: never commit them, nor any file derived from them (`out/` artifacts, fetched CSVs).

## Statistics house rules

- One `numpy.random.Generator` threaded through every stochastic call — no fresh
  generators mid-pipeline, no global seeding.
- Discovery never reads the outcome.
- Failed computations return NaN, never 0.0.
- No bare `except`.

## Math

- `docs/math_audit_final.md` governs whenever code, papers, or docs disagree about math.
- Per-method summaries with equations: `docs/method_cards/`.

## Workflow

- TDD: failing test first, implement, then `uv run ruff check src tests` + `uv run pytest -q`.
- Make a conventional commit (`feat:`, `fix:`, `docs:`, `test:`) after every green cycle.
