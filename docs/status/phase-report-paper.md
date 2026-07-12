# Phase report-paper status — reporting & paper pipeline

Date: 2026-07-12. Plan: [docs/plans/phase-report-paper.md](../plans/phase-report-paper.md).
Spec gate (design spec §7, §4 `to_report` bundle contract + extras policy, §6b coverage
always reported, §10 "non-engineer user" risk): **results bundle + standard figures +
Jinja2 markdown/LaTeX paper drafts with the AI-draft banner + `natex paper` + paperbanana
diagram adapter + deep-research handoff brief, everything rendering numbers already
computed (no new inference code)** — met, with the limitations logged below. Core deps
unchanged; jinja2 lives ONLY under the new `[report]` extra, paperbanana under the new
`[paperbanana]` extra (never installed in CI), matplotlib stays under `[plot]`.

## What shipped

- **ResultsBundle** (`report/bundle.py`): `from_discover` / `from_scan_payload` / `load`
  / `save` / `add_figure`; `results.json` is JSON-native (`jsonable` runs exactly once on
  save, NaN/inf → null) and records the `natex_bundle` schema marker, natex version,
  created timestamp, seed, params, search coverage verbatim (spec 6b), all
  `ConfigRecord`s, `best_index`, guidance-log path, data/intake provenance blocks, and
  the figure manifest. `load` resolves a saved bundle, a `discover --out` directory
  (`discover_report.json`), or a single-scan `results.json`, and names every path it
  looked for on failure. `ivw_pooled` is the ONLY presentational combiner (documented as
  indicative — its inputs share a neighborhood).
- **Figures** (`report/figures.py`, `[plot]` extra, lazy matplotlib import):
  `discovery_scatter`, `density_hist`, `pretrend_plot`, `effect_forest`, plus the
  `rdd_figures` / `did_figures` bundle helpers — each figure saves PNG (150 dpi) + PDF
  and registers itself in the bundle manifest. The did forest deliberately has NO pooled
  row (dd/synthetic/gess share treated cells).
- **`period_gaps`** (`did/effects.py`): public per-period treated-minus-control gap
  helper (fitted-contrast reuse, audit 19) backing `pretrend_plot`.
- **Paper renderer** (`report/paper.py` + `templates/paper.md.j2` / `paper.tex.j2`, NEW
  `[report]` extra, jinja2>=3.1): `render_paper(bundle, format="md"|"latex")`; every
  draft opens with the banner "AI-generated draft — verify all claims before
  circulation"; method cards inlined from `docs/method_cards` with an installed-wheel
  placeholder fallback; `texesc` + bounded `_md_to_tex` for LaTeX safety; tectonic
  compile when on PATH, graceful message (never an exception) otherwise; missing numbers
  render as "—", never `nan`/`None`.
- **CLI**: `natex paper --bundle DIR --format md|latex [--out DIR]`; missing `[report]`
  extra or a bad bundle exits 2 with an actionable message, no traceback; the
  review-before-sharing warning prints on every run.
- **paperbanana adapter** (`report/paperbanana.py`, NEW `[paperbanana]` extra):
  `generate_method_diagram(bundle, out)` with the single documented call contract
  `paperbanana.generate_diagram(description=..., output_path=...)`; the pipeline
  description is pure text from `bundle.results`.
- **Deep-research handoff** (`report/research_brief.py`, core deps only):
  `research_brief(bundle, out)` writes `research-brief.md` — data context, discovered
  designs, effects, validation status, numbered literature questions — byte-identical on
  rerun; natex performs no research calls.
- **CI**: `.github/workflows/ci.yml` now syncs `--extra dev --extra plot --extra report`
  so figure and render tests actually run on 3.11–3.14; core-only skip paths stay
  covered by the no-extras gate below.
- **Docs**: README "From discovery to paper" section (three-command flow, Python API,
  deep-research handoff, manual Google Docs route — natex does NOT integrate the Google
  Docs API — extras install lines, banner rule) + roadmap tick, this status doc, and
  `tests/test_docs.py` documentation contracts.

## Test counts

78 tests added this phase (`test_report_bundle.py` 18, `test_report_figures.py` 8,
`test_report_paper.py` 18, `test_cli_paper.py` 6, `test_paperbanana.py` 7,
`test_research_brief.py` 8, 4 `period_gaps` tests in `test_did_effects.py`, 9 doc
contracts in `test_docs.py`): 657 (phase llm-analyst) → 735. All paperbanana tests use a
fake `sys.modules` module; LaTeX content tests monkeypatch `shutil.which` to None; one
skipif-gated test runs a real tectonic compile when the binary is present — no network in
CI anywhere.

## Final gate record (2026-07-12, Apple Silicon macOS arm64, Python 3.13.14)

1. `uv run ruff check src tests` — `All checks passed!`
2. NO extras installed (`uv sync --extra dev`; dev is itself an extra):
   `uv run pytest -q` — **`702 passed, 26 skipped, 32 deselected`**. The 12 report-layer
   skips are visible and graceful: the `test_report_figures.py` module skip (matplotlib),
   8 render tests in `test_report_paper.py` (jinja2), 3 CLI render tests in
   `test_cli_paper.py` (jinja2); the remaining 14 are the pre-existing ml/gp/llm
   optional-extra skips (no pre-existing test skips on matplotlib alone).
3. CI configuration (`uv sync --extra dev --extra plot --extra report`):
   spot-run `uv run pytest -q tests/test_report_bundle.py tests/test_report_figures.py
   tests/test_report_paper.py tests/test_cli_paper.py tests/test_paperbanana.py
   tests/test_research_brief.py` — **`65 passed`**, zero skips (the skipif-gated real
   tectonic compile runs locally because tectonic is installed). Full suite:
   **`721 passed, 14 skipped, 32 deselected`** (the 14 = pre-existing ml/gp/llm
   optional-extra skips).
4. ALL extras (`uv sync --all-extras`, now including paperbanana 0.3.0):
   `uv run pytest -q` — **`735 passed, 32 deselected`** — zero skips; the paperbanana
   tests still exercise only the fake module (no key, no network).
5. Backtest regression (`NATEX_DATA=".../RDD/data" uv run pytest -q -m backtest`) —
   **`31 passed, 735 deselected, 1 xfailed`**: identical outcome set to the phase-5 and
   phase-6 records (all 15 phase-2 RDD rows, 9 phase-3 Prop 99 SuDDDS tests, 8 phase-5
   donor/IV tests incl. the non-blocking Egger xfail). This phase added no backtests and
   disturbed none.

## Known limitations (all deliberate, none silent)

1. **`_md_to_tex` is lossy on tables**: markdown tables inside method cards are replaced
   in the LaTeX draft by an omission marker pointing at the markdown card; numbered
   lists, nesting, and block quotes degrade to escaped plain text. The markdown draft is
   always faithful.
2. **paperbanana contract is tested against the fake module only** — the documented call
   `paperbanana.generate_diagram(description=..., output_path=...)` is pinned by
   monkeypatched recorder tests; drift in the real library's API surfaces only at
   runtime with a real provider key.
3. **Installed-wheel method-card fallback**: outside a repo checkout (no
   `docs/method_cards` next to the package or in cwd), drafts inline a placeholder
   pointing at the repository instead of the method-card text.
4. **`texesc` is single-application** (not idempotent) and normalizes em/en dashes to
   `--`; non-Latin glyphs in method cards (⊆, →) drop under tectonic's default font with
   warnings only — the compile still succeeds.
5. **`ivw_pooled` is indicative, not meta-analytic**: the 2SLS and Wald rows it pools
   share the same discovered neighborhood, so the pooled row is labeled indicative in
   the forest plot and templates.
6. **Google Docs route is manual by design**: export markdown and paste into a Doc, or
   upload the `.md` via Google Drive — natex does NOT integrate the Google Docs API.

## Run of record

```bash
uv run ruff check src tests
uv run pytest -q                                   # excludes backtests (addopts)
uv sync --extra dev && uv run pytest -q            # core-only skip-path proof
uv sync --extra dev --extra plot --extra report    # the CI configuration
uv run pytest -q tests/test_report_bundle.py tests/test_report_figures.py \
  tests/test_report_paper.py tests/test_cli_paper.py tests/test_paperbanana.py \
  tests/test_research_brief.py
uv sync --all-extras && uv run pytest -q           # full local run
NATEX_DATA="/path/to/RDD/data" uv run pytest -q -m backtest
```

## Open questions for phase 8

- The design spec's `natex report` CLI subcommand (bundle+figures in one shot) is folded
  into the Python API this phase; a thin CLI wrapper is a natural phase-8 addition once
  the agent-skill workflow settles.
- The deep-research merge-back (spec §7.4: fold the returned review into the
  related-work section) stays manual; an agent skill could own the round trip.
- Rendering `ConfigRecord.advisory` blocks (guidance interpretations, assumption audits,
  veto flags) in the paper's robustness section — the bundle already carries them.
