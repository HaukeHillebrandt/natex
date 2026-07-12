# Phase report-paper implementation plan — Report + paper pipeline

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` **section 7**
(reporting & paper pipeline), section 4 (`result.to_report("out/")` bundle contract, extras policy),
section 6b (coverage always reported), section 10 risk "non-engineer user".
**First action of task 1 is committing this plan file itself**
(`docs: phase report-paper implementation plan`).

## Phase objective

The reporting/paper layer ONLY — nothing else:

1. `natex/report/bundle.py` — `ResultsBundle`: DiscoverReport (or single scan results) +
   validation + effects + intake/guidance artifacts → `results.json` (JSON-native, NaN→null,
   version/seed/params/coverage/guidance-log-path) with `figures/` and `paper/` subdirs.
   API: `ResultsBundle.from_discover(report, out_dir)`, `.save()`, `.load(dir)`.
2. `natex/report/figures.py` (existing `plot` extra, import-guarded): `discovery_scatter`,
   `density_hist`, `pretrend_plot`, `effect_forest` — each saves PNG (150 dpi) + PDF, returns paths.
3. `natex/report/paper.py` (NEW `report` extra, jinja2>=3.1): `render_paper(bundle, format, out_dir)`
   from Jinja2 templates (`paper.tex.j2`, `paper.md.j2`); tectonic compile when present, always-working
   markdown; prominent AI-draft banner.
4. CLI `natex paper --bundle DIR --format latex|md --out DIR`.
5. `natex/report/paperbanana.py` — optional diagram adapter behind a `paperbanana` extra; CI test
   uses a monkeypatched fake module, never a real API call.
6. `natex/report/research_brief.py` — deep-research handoff `research-brief.md`; pure text.
7. README section "From discovery to paper" (flow, deep-research handoff, manual Google Docs route —
   natex does NOT integrate the Docs API).

## Global constraints (binding, from the phase-1 plan)

- Python >= 3.11. Core deps stay exactly numpy/scipy/pandas/scikit-learn/typer/pydantic.
  matplotlib stays under `[plot]`; jinja2 goes under a NEW `[report]` extra; paperbanana under a NEW
  `[paperbanana]` extra. Every optional-dep test skips gracefully when the dep is missing
  (`pytest.importorskip`), so a core-only install stays green on 3.11–3.14.
- One `numpy.random.Generator` through every stochastic call (this phase adds NO stochastic code;
  tests that build bundles seed `np.random.default_rng(0)` end to end).
- Discovery never reads the outcome — reporting only *renders* numbers already computed; no new
  inference code anywhere in `natex/report/` except the presentational `ivw_pooled` combiner
  (documented as indicative, see task 1).
- NaN never 0.0 on failure; JSON via `natex.jsonutil.jsonable` (NaN/inf → null); templates render
  missing numbers as "—", never "nan"/"None".
- No bare except. Never commit datasets. Conventional commit after every green cycle.
- `uv run pytest -q` excludes backtests (`addopts = -m 'not backtest'`); this phase adds no backtests.
- TDD discipline per task: write the failing test first, implement, `uv run ruff check src tests`
  + `uv run pytest -q` (full suite), commit.

## Current repo state (interfaces this phase builds on — verified 2026-07-12)

- `natex.discover.DiscoverReport` — `configs: list[ConfigRecord]`, `searched: dict` (coverage:
  n_total/n_scanned/n_skipped_budget/n_failed/n_invalid/budget/plan_candidates/exhaustive_candidates),
  `best_index: int | None`, `guidance_log_path: str | None`; `.best()`, `.to_json()`,
  `.save(out)` → `out/discover_report.json`. `ConfigRecord.to_dict()` gives
  candidate/source/status/llr/p_value/n_discoveries/summary/advisory/error. The rdd summary carries
  `center_z`, `normal`, `forcing_influence`, `llr`, `p_value`, `placebo_passed`, `placebo_holm`,
  `density_p`, `coarse`, `effects` (`{"2sls"|"wald": {tau, se, ci, first_stage_t, weak_instrument}}`);
  the did summary carries `subset_values`, `t0`, `window`, `llr`, `p_value`, `null_kind`,
  `composition_passed`, `anticipation_passed`, `searched_windows`, `restarts`, `effects`
  (`{"dd"|"synthetic"|"gess": {tau, se, p, pre_mse, dose}}`).
- `natex.intake.analyst.IntakeReport` — `profile`, `understanding: Understanding`
  (`shape`, `unit_of_observation`, `quirks`, `notes`, …), `prep_plan`, `search_plan`,
  `guidance_log_path`, `context`, `source`, `guidance_errors`, `prep_log`; `.load(path)`.
- `natex.rdd.lord3` — `Discovery(center_index, k, llr, normal, members, group1, …)`,
  `LoRD3Result(discoveries, model, k, centers)`.
- `natex.validate.placebo.signed_distance(dataset, d) -> np.ndarray` (standardized-Z signed distance
  over `d.members`); `natex.validate.density.density_test` (McCrary-style; audit item 6 caveat:
  valid only for the FROZEN discovered geometry).
- `natex.did.panel.CategoricalPanel` (`t`, `theta`, `y`, `unit`, `dim_names`, `subset_mask`);
  `natex.did.suddds.DiDDiscovery` (`subset_values`, `mask`, `t0`, `window`, `llr`);
  `natex.did.effects` — `did_effect`, `_resolve_control`, `_apply_control_to` (fitted-contrast reuse,
  audit 19), `_mean_gap` (per-period means computed internally but not exposed → task 2).
- `natex.jsonutil.jsonable` — the ONLY JSON coercion path (NaN/inf → None).
- Synthetic data for tests: `natex.data.synthetic.make_synthetic(n, zeta, kind, rng, …) ->
  (Dataset, D)` (rdd; treatment "T", outcome "y", forcing ["x0","x1"]);
  `natex.data.synthetic_did.make_did_synthetic(n, d, V, zeta, tau, rng, …) -> (Dataset, DiDTruth)`
  (time "t", treatment "theta", outcome "y").
- CLI patterns: `_clean = jsonable`; optional-dep failures exit code 2 with an install message
  (`_make_backend`); `tests/conftest.py` already disables typer terminal styling for CI-safe asserts.
- CI: `.github/workflows/ci.yml` runs `uv sync --extra dev` then ruff + pytest on 3.11–3.14.
- Method cards: `docs/method_cards/{suddds,dee,iv_sc,llm_analyst}.md`. **There is no LoRD3/rdd card**
  → task 4 writes `docs/method_cards/lord3.md` (the paper's Methods section needs it).
- `docs/status/` convention: short phase status file at the end (task 9).

## Audit corrections that bind THIS phase (rendering obligations)

The report layer must never weaken the audit's inference-honesty language:

- **Audit 1 (inference honesty):** every p-value is rendered as a "+1-rank Monte Carlo p-value";
  templates and the brief NEVER say "exact". The corrections note in the paper's Introduction
  references `docs/math_audit_final.md` explicitly.
- **Audit 2 (placebo redefinition):** placebo results are labeled "local intercept-continuity
  placebo (Holm-adjusted)" — not "covariate balance".
- **Audit 3 (repaired estimator):** effects tables always carry the `weak_instrument` flag next to
  2SLS rows; the wald row is labeled as the auxiliary estimator.
- **Audit 6 (frozen-geometry falsification):** the density figure caption and the Robustness text
  state the test is valid for the frozen discovered geometry only and does not account for the
  search having selected normal and cutoff.
- **Spec 6b (coverage):** the Robustness section always states what was and wasn't searched,
  from `searched` verbatim — including skipped_budget/failed/invalid configs by name.
- **House rule (NaN never 0.0):** null/NaN render as "—"; `ivw_pooled` returns NaN (not 0) when no
  usable input; templates must never print "nan", "None", or fabricate a number.
- **DiD SE caveat:** the did effects table footnote carries the documented simple choice — "SE = sd
  of per-post-period mean gaps / √h" (from `DiDEffect.se`).
- **Human-in-the-loop (spec 7.5):** every rendered draft (md, tex, pdf) carries the banner
  "AI-generated draft — verify all claims before circulation" prominently at the top.

---

## Task 1 — ResultsBundle (`natex/report/bundle.py`)

**First action: commit this plan file** —
`git add docs/plans/phase-report-paper.md && git commit -m "docs: phase report-paper implementation plan"`.

**Create:** `src/natex/report/__init__.py`, `src/natex/report/bundle.py`, `tests/test_report_bundle.py`,
`tests/report_helpers.py` (plain helper module for all report tests).
**Modify:** `src/natex/__init__.py` (export `ResultsBundle`).

### Interfaces (exact — later tasks depend on them)

```python
# src/natex/report/bundle.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from natex.data.spec import Dataset
from natex.discover import DiscoverReport
from natex.intake.analyst import IntakeReport
from natex.jsonutil import jsonable

BUNDLE_SCHEMA = 1  # top-level "natex_bundle" marker distinguishes bundle vs raw scan payloads


@dataclass(frozen=True)
class PooledEffect:
    tau: float          # NaN when no usable input — never 0.0
    se: float           # NaN when no usable input
    ci: tuple[float, float]
    n_used: int


def ivw_pooled(tau, se) -> PooledEffect:
    """Fixed-effect inverse-variance pooling; PRESENTATIONAL ONLY.

    Drops entries where tau or se is non-finite or se <= 0. weights = 1/se^2;
    pooled se = sqrt(1/sum w); ci = tau +/- 1.96 se. All-NaN input -> PooledEffect
    with NaN tau/se/ci and n_used=0. Callers must label the pooled row as
    indicative (the per-estimator inputs are NOT independent).
    """


class ResultsBundle:
    """A results directory: results.json + figures/ + paper/ (spec section 7)."""

    def __init__(self, out_dir: str | Path, results: dict): ...
    # results is ALWAYS JSON-native (already passed through jsonable)

    out_dir: Path
    results: dict

    @property
    def results_path(self) -> Path: ...   # out_dir / "results.json"
    @property
    def figures_dir(self) -> Path: ...    # out_dir / "figures"
    @property
    def paper_dir(self) -> Path: ...      # out_dir / "paper"

    @classmethod
    def from_discover(cls, report: DiscoverReport, out_dir: str | Path, *,
                      dataset: Dataset | None = None,
                      intake: IntakeReport | None = None,
                      seed: int | None = None,
                      params: dict | None = None) -> ResultsBundle: ...

    @classmethod
    def from_scan_payload(cls, payload: dict, out_dir: str | Path) -> ResultsBundle:
        """Wrap a single-scan results.json payload (the non-plan `natex discover`
        schema: params/scan/discoveries/validation/effects) under {"scan": payload},
        lifting seed from payload["params"]["seed"] when present."""

    @classmethod
    def load(cls, dir: str | Path) -> ResultsBundle:
        """Load a bundle directory. Resolution order:
        1. dir/results.json with "natex_bundle" key -> bundle as saved;
        2. dir/results.json WITHOUT the marker -> from_scan_payload(payload, dir);
        3. dir/discover_report.json -> adapt the DiscoverReport JSON (configs/
           searched/best_index/guidance_log_path verbatim; version/seed/params null);
        4. else FileNotFoundError naming all three paths it looked for."""

    def add_figure(self, name: str, png: Path, pdf: Path) -> None:
        """Append/replace (by name) a manifest entry in results["figures"];
        paths stored POSIX-relative to out_dir."""

    def save(self) -> Path:
        """mkdir out_dir, figures/, paper/; write results.json =
        json.dumps(jsonable(results), indent=1); return results_path."""
```

### `results.json` schema (from_discover)

```json
{
 "natex_bundle": 1,
 "natex_version": "<natex.__version__>",
 "created": "<ISO-8601 UTC or null>",
 "seed": 0,
 "params": { "...budget/caller params; defaults to searched.budget..." },
 "searched": { "...report.searched verbatim (spec 6b coverage)..." },
 "configs": [ { "...ConfigRecord.to_dict() verbatim..." } ],
 "best_index": 0,
 "guidance_log_path": null,
 "data": {"n_rows": 300, "treatment": "T", "outcome": "y", "forcing": ["x0","x1"],
          "time": null, "unit": null, "covariates": ["x0","x1"], "source": null},
 "intake": {"source": "...", "context": "...", "understanding": {"...model_dump..."},
            "guidance_errors": [], "prep_log": []},
 "figures": []
}
```

Rules: `data` from `dataset.spec` when given (n_rows = len(dataset.df)), else derived from the
best (or first) config's candidate with `n_rows: null`; `intake` null when not given;
`guidance_log_path` = `report.guidance_log_path` else `intake.guidance_log_path` else null;
`created` is written but NEVER asserted in tests; everything passes through `jsonable` exactly once
(in `save()`), so NaN → null.

### Tests (`tests/report_helpers.py` + `tests/test_report_bundle.py`)

`tests/report_helpers.py` (imported by all report test modules; NOT a conftest):

```python
SMALL = {"k": 25, "q": 9}
def make_rdd_bundle(tmp_path, *, with_outcome=True, seed=0) -> tuple[ResultsBundle, DiscoverReport, Dataset]
    # make_synthetic(n=300, zeta=6.0, kind="binary", rng=default_rng(seed));
    # drop the outcome column + spec.outcome=None when with_outcome=False;
    # discover(ds, rng=default_rng(seed), budget=SMALL); ResultsBundle.from_discover(..., dataset=ds, seed=seed).save()
def make_did_bundle(tmp_path, *, seed=0) -> tuple[ResultsBundle, DiscoverReport, Dataset]
    # make_did_synthetic(n=400, d=2, V=3, zeta=8.0, rng=default_rng(seed)); same wrap
```

Assertions (all seeded, deterministic):
1. `from_discover(...).save()` writes `results.json` and creates `figures/`, `paper/` dirs.
2. `results["natex_version"] == natex.__version__`; `results["seed"] == 0`;
   `results["searched"] == json.loads(report.to_json())["searched"]` (coverage verbatim);
   `results["configs"][best_index]["summary"]["effects"]["2sls"]["tau"]` is a finite float.
3. Round-trip: `ResultsBundle.load(dir).results == saved.results` (exact dict equality — JSON-native).
4. NaN→null: mutate one summary value to `float("nan")` before save; `"NaN" not in results_path.read_text()`
   and reloaded value is `None`.
5. Adaptation: (a) write a minimal single-scan payload as `results.json` (keys params/scan) → `load`
   wraps under `"scan"` and lifts `seed`; (b) `report.save(dir)` only (discover_report.json) → `load`
   adapts with `natex_version is None` and configs intact; (c) empty dir → `FileNotFoundError`
   whose message names `results.json` and `discover_report.json`.
6. `ivw_pooled([1.0, 3.0], [1.0, 1.0])` → tau == 2.0 exactly, se == pytest.approx(1/np.sqrt(2)),
   ci == pytest.approx((2 - 1.96/np.sqrt(2), 2 + 1.96/np.sqrt(2))), n_used == 2;
   `ivw_pooled([1.0, np.nan], [1.0, 1.0])` → n_used == 1, tau == 1.0;
   `ivw_pooled([np.nan], [np.nan])` → all NaN, n_used == 0 (never 0.0 tau).
7. `intake=` wiring: build an `IntakeReport` via `natex.study(csv, rng=..., out=...)` on the synthetic
   csv (NullBackend path — no LLM), pass it in → `results["intake"]["understanding"]["shape"]` present
   and `guidance_log_path` non-null.
8. No-outcome bundle (`with_outcome=False`): effects block `{}`, bundle still saves/loads.

**Commit:** `feat(report): ResultsBundle results.json with version/seed/coverage/intake metadata`

---

## Task 2 — `period_gaps` public helper (`natex/did/effects.py`)

The pretrend figure needs per-period treated-minus-control gaps; `_mean_gap` computes period means
internally but discards them. Add a PUBLIC descriptive helper that reuses the FITTED control
contrast (audit 19: same contrast, same control set — NO new inference).

**Modify:** `src/natex/did/effects.py`, `src/natex/did/__init__.py` (export).
**Modify (tests):** `tests/test_did_effects.py` (append a `period_gaps` test class).

```python
@dataclass(frozen=True)
class PeriodGaps:
    times: np.ndarray   # sorted unique usable s_tau periods (pre AND post)
    gap: np.ndarray     # per-period mean of y - y0_hat over usable s_tau records
    n: np.ndarray       # usable record count per period (int)
    t0: float
    control: str        # "dd" | "synthetic" | "gess"


def period_gaps(panel: CategoricalPanel, discovery: DiDDiscovery,
                control: str | ControlResult = "dd") -> PeriodGaps:
    """Per-period mean treated-minus-control gap via _resolve_control +
    _apply_control_to(panel, discovery, control, panel.y). Periods with zero
    usable (finite y and counterfactual) records are OMITTED, never zero-filled.
    Raises ValueError when panel.y is None (reporting never fabricates outcomes)."""
```

### Tests (seeded, robust)

- `make_did_synthetic(n=1500, d=2, V=3, zeta=10.0, tau=10.0, rng=default_rng(0))`, `build_panel(ds, bins=3)`,
  `suddds_scan(..., rng=default_rng(0))` → top discovery; `g = period_gaps(panel, top, "dd")`:
  `np.all(np.diff(g.times) > 0)`; `g.t0 == top.t0`; pre-mask `g.times < g.t0` and post-mask both
  non-empty; `abs(mean(pre gaps)) < 2.0` and `mean(post gaps) > 5.0` (tau=10 with zeta=10 —
  comfortably separated; keep thresholds loose against seed drift but assert post > pre + 4.0 too).
- Consistency: `np.average(post gaps, weights=post n)` == pytest.approx(`did_effect(panel, top, "dd").tau`
  when `dose` normalization is None, rel=1e-6) — same records, same contrast. If the discovery
  normalizes by dose, compare the unnormalized gap path instead (read `DiDEffect.dose` to branch).
- `panel.y = None` path (rebuild panel from a no-outcome Dataset) → `pytest.raises(ValueError)`.

**Commit:** `feat(did): public period_gaps per-period treated-minus-control helper for reporting`

---

## Task 3 — Figures (`natex/report/figures.py`, plot extra)

**Create:** `src/natex/report/figures.py`, `tests/test_report_figures.py`.
**Modify:** `.github/workflows/ci.yml` — `uv sync --extra dev --extra plot` (so figure tests actually
run in CI; they still `importorskip` for core-only installs).

### Interfaces

```python
@dataclass(frozen=True)
class FigurePaths:
    png: Path
    pdf: Path


def _mpl():
    """Lazy import; ImportError('figures require the plot extra: pip install
    "natex-discovery[plot]"') when matplotlib is missing. Module import itself
    never touches matplotlib (import-guarded)."""


def discovery_scatter(Z, llr, *, top_centers=None, top_normals=None, names=None,
                      out_dir, stem="discovery_scatter") -> FigurePaths
    # Z (n, d>=1) RAW forcing coords of scored centers, llr (n,).
    # d == 1: x = Z[:,0], y = llr, top centers marked with vertical cutoff lines.
    # d >= 2: scatter of first two dims colored by LLR (continuous colormap),
    # top discoveries starred + normal drawn as an arrow (first two components,
    # scaled to ~10% of the axis span). Non-finite llr entries are dropped, not zeroed.

def density_hist(s, *, out_dir, cutoff=0.0, n_bins=20, p_value=None,
                 stem="density_hist") -> FigurePaths
    # Signed-distance histogram (from validate.placebo.signed_distance), side-split
    # colors, vertical cutoff line; title carries "McCrary-style, frozen geometry"
    # and the p-value when given (audit 6 caveat in the axes text).

def pretrend_plot(times, gaps, t0, *, n=None, out_dir, stem="pretrend") -> FigurePaths
    # Per-period treated-minus-control gaps (task 2 output), zero reference line,
    # vertical T0 line, pre/post shading; marker size ~ n when given.

def effect_forest(labels, tau, lo, hi, *, pooled=None, out_dir,
                  stem="effect_forest") -> FigurePaths
    # Horizontal forest: one row per (neighborhood/estimator) label with CI
    # whiskers; rows with non-finite lo/hi plot the point WITHOUT whiskers
    # (never a fabricated CI); pooled = (label, tau, lo, hi) rendered last,
    # visually distinct, with zero reference line.
```

Bundle glue (also in figures.py; needs live objects, so it runs at bundle-build time):

```python
def rdd_figures(bundle: ResultsBundle, dataset: Dataset, result: LoRD3Result,
                *, top_m: int = 5) -> dict[str, FigurePaths]
    # discovery_scatter over dataset.Z[d.center_index] / d.llr for all discoveries;
    # density_hist over signed_distance(dataset, result.discoveries[0]) with the
    # bundle's density_p; effect_forest over the best config's effects (2sls +
    # wald rows, pooled = ivw_pooled labeled "IVW pooled (indicative)") when the
    # effects block is non-empty. Registers each via bundle.add_figure and
    # calls bundle.save() once at the end.

def did_figures(bundle: ResultsBundle, panel: CategoricalPanel,
                discovery: DiDDiscovery, *, control: str = "dd") -> dict[str, FigurePaths]
    # pretrend_plot from period_gaps(panel, discovery, control);
    # effect_forest over dd/synthetic/gess rows (ci = tau +/- 1.96*se; NaN se ->
    # point without whiskers), no pooled row (controls share the treated cells).
    # Same add_figure + save() contract.
```

Style (single private `_RC` dict applied via `plt.rc_context` in every function): Okabe–Ito
colorblind-safe cycle (`#0072B2 #E69F00 #009E73 #D55E00 #CC79A7 #56B4E9 #F0E442 #000000`),
`font.family: DejaVu Sans` (tabular digits), spines top/right off, `savefig.bbox: tight`;
PNG at `dpi=150`, PDF vector; no seaborn anywhere.

### Tests (`tests/test_report_figures.py`)

Top of module: `matplotlib = pytest.importorskip("matplotlib"); matplotlib.use("Agg")` BEFORE any
pyplot import. All inputs seeded (`default_rng(0)`).

1. Each of the four functions: returns `FigurePaths`; both files exist; `png.stat().st_size > 2000`
   and `pdf.stat().st_size > 500`; calling twice overwrites idempotently (no figure-handle leak:
   `len(plt.get_fignums()) == 0` after each call — every function closes its figure).
2. `discovery_scatter` with 1-D Z and with 2-D Z; llr containing NaN entries does not raise and the
   files are still written.
3. `effect_forest` with one NaN-CI row and a pooled tuple → files written.
4. `rdd_figures` end to end on `make_rdd_bundle` + a fresh seeded `lord3_scan(k=25)`: returns dict with
   keys `{"discovery_scatter", "density_hist", "effect_forest"}`; `bundle.results["figures"]` has 3
   entries whose relative paths exist under `bundle.figures_dir`; reloading the bundle preserves the
   manifest.
5. `did_figures` end to end on `make_did_bundle` + seeded `build_panel`/`suddds_scan`: keys
   `{"pretrend", "effect_forest"}`, manifest updated.
6. Import guard: `monkeypatch.setitem(sys.modules, "matplotlib", None)` →
   `discovery_scatter(...)` raises `ImportError` matching `natex-discovery\[plot\]`
   (and `import natex.report.figures` itself succeeded — module import is guard-free).

**Commit:** `feat(report): discovery/density/pretrend/forest figures under the plot extra`

---

## Task 4 — Markdown paper renderer + templates + LoRD3 method card

**Create:** `src/natex/report/paper.py`, `src/natex/report/templates/paper.md.j2`,
`docs/method_cards/lord3.md`, `tests/test_report_paper.py`.
**Modify:** `pyproject.toml` (add `report = ["jinja2>=3.1"]` extra),
`.github/workflows/ci.yml` (`uv sync --extra dev --extra plot --extra report`),
`src/natex/report/__init__.py` (export `render_paper` — paper.py's module level must NOT import
jinja2; the import happens inside `render_paper`/`_env()` so `import natex.report` stays core-clean).

### Interfaces

```python
@dataclass(frozen=True)
class PaperResult:
    markdown: Path | None
    tex: Path | None
    pdf: Path | None
    compiled: bool
    message: str


BANNER = "AI-generated draft — verify all claims before circulation"
_CARDS = {"rdd": "lord3.md", "did": "suddds.md"}   # design -> method card file


def _cards_dir(explicit: str | Path | None) -> Path | None:
    """explicit arg, else repo-relative Path(__file__).parents[3]/'docs/method_cards'
    (src layout; works for editable installs), else cwd()/'docs/method_cards',
    else None (installed-wheel fallback -> placeholder text in the Methods section)."""


def _paper_context(bundle: ResultsBundle, fmt: str,
                   cards_dir: str | Path | None = None) -> dict:
    """Pure dict builder — fully testable WITHOUT jinja2. Keys (exact):
    banner, title, version, seed, created, data, intake, designs,
    method_cards [{design, title, body}], discovery_rows, best, validation,
    effects_rows, coverage (searched + not_searched label list), figures
    [{name, relpath}] (relative to paper/, i.e. '../figures/x.png'),
    references (static lineage strings), corrections_note.
    Number formatting through the `_fmt(x, nd=3)` helper: finite -> fixed 3
    significant digits, None/NaN -> '—' (never 'nan'/'None')."""


def render_paper(bundle: ResultsBundle, format: str = "md",
                 out_dir: str | Path | None = None, *,
                 cards_dir: str | Path | None = None) -> PaperResult:
    """format in {'md','latex'} else ValueError naming the value. out_dir
    defaults to bundle.paper_dir. Markdown ALWAYS works (this task); latex in
    task 5. jinja2 imported lazily; missing -> ImportError('render_paper
    requires the report extra: pip install "natex-discovery[report]"')."""
```

Jinja2 env: `Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)` with filters
`fmt` (the number formatter) and (task 5) `texesc`. StrictUndefined so a template/context drift
fails loudly instead of silently rendering blanks.

### `paper.md.j2` section contract (assertions target these)

1. Title + `> **{{ banner }}**` blockquote as the FIRST body line.
2. **Introduction** — method lineage citing Herlands, McFowland III, Somanchi & Neill (2018, KDD),
   Herlands (2019, CMU thesis), Jakubowski, Somanchi, McFowland III & Neill (2023); one paragraph
   corrections note: natex deviates from the printed formulas where the two-model math audit found
   errors — see `docs/math_audit_final.md` and README "Corrections vs the papers"; all p-values are
   +1-rank Monte Carlo, never exact (audit 1).
3. **Data** — from `data` block (+ intake understanding shape/unit/quirks when present); explicit
   sentence "Discovery never reads the outcome column."
4. **Methods** — for each design among scanned configs, the method card body verbatim (or the
   placeholder "(method card `lord3.md` not available in this installation — see the natex
   repository's docs/method_cards/)" when cards_dir is None/missing).
5. **Results** — discovery table (design | source | status | max LLR | MC p-value | #discoveries);
   best-config detail (rdd: cutoff center in raw forcing coords, forcing influence, density p;
   did: subset, T0, window); effects table (estimator | tau | SE | 95% CI | flags) with
   weak-IV flag column (audit 3) and the did SE footnote; figures embedded
   (`![name](../figures/x.png)`) when the manifest is non-empty.
6. **Robustness** — validation battery narrative (randomization calibration, placebo
   intercept-continuity Holm block, density frozen-geometry caveat / composition + anticipation for
   did) + coverage statement: "Of {n_total} enumerated configurations, {n_scanned} were scanned,
   {n_skipped_budget} skipped by budget, {n_failed} failed, {n_invalid} invalid." followed by the
   not-searched list by candidate label (spec 6b).
7. **References** — the three lineage entries + the natex software citation (name, version).

### `docs/method_cards/lord3.md`

Written in the established card style (see `suddds.md` header): Source = Herlands et al. 2018 KDD +
thesis ch. 5; governing math = audit; Modules = `natex.rdd.*`, `natex.scan.*`, `natex.validate.*`,
`natex.estimate.local2sls`; What it does (kNN neighborhoods → bisection LLR scan over treatment
model residuals, never y → fitted-null randomization calibration → placebo/density falsification →
frozen side-indicator local 2SLS); Corrected math bullets condensed from audit items 1, 2, 3, 6
(+ legacy-parity caveat 7). ~60–100 lines; content sourced from `docs/math_audit_final.md` and the
README corrections section — no new claims.

### Tests (`tests/test_report_paper.py`)

Context-builder tests (NO jinja2 needed — run everywhere):
1. `_paper_context(bundle, "md")` on `make_rdd_bundle`: `designs == ["rdd"]`; discovery_rows length
   == n configs; effects_rows contain a `2sls` row with finite tau string and a weak-IV flag field;
   coverage numbers equal the searched block; `"nan" not in json.dumps(ctx)` (case-insensitive) and
   `"None" not in` any rendered-string field.
2. Missing pieces degrade: no-outcome bundle → effects_rows == [] and the context still builds;
   `cards_dir=tmp_path` (empty) → method_cards carry the placeholder body.

Render tests (`pytest.importorskip("jinja2")`):
3. `render_paper(bundle, "md")` → `PaperResult.markdown` exists under `bundle.paper_dir`,
   `.tex/.pdf` are None, `compiled is False`, message mentions markdown.
4. Content: text contains `BANNER`; "Herlands" at least twice and "Jakubowski" once;
   "math_audit_final.md"; "Monte Carlo"; the coverage sentence with the exact n_total/n_scanned
   integers; the lord3 card's title line; `"—"` for at least one field when a NaN is injected into
   an effects se before save; NEVER contains "nan," / "None" as rendered values (regex
   `\bnan\b|\bNone\b` absent).
5. Figures: after `rdd_figures(...)` (skip this test if matplotlib missing), rendered md contains
   `../figures/discovery_scatter.png`.
6. did bundle renders (suddds card pulled, subset/T0 present); bad format →
   `pytest.raises(ValueError, match="banana")`.
7. Import guard: `monkeypatch.setitem(sys.modules, "jinja2", None)` → `render_paper` raises
   ImportError matching `natex-discovery\[report\]`; `import natex.report.paper` succeeds without
   jinja2 (delete from sys.modules first).

**Commit:** `feat(report): jinja2 markdown paper renderer with method cards and AI-draft banner`

---

## Task 5 — LaTeX template + tectonic compile

**Create:** `src/natex/report/templates/paper.tex.j2`.
**Modify:** `src/natex/report/paper.py` (texesc filter, `_md_to_tex`, `_compile_tex`, latex branch),
`tests/test_report_paper.py` (append).

### Interfaces

```python
def texesc(s: str) -> str:
    """Escape LaTeX specials in ORDER (backslash first): \\ { } & % $ # _ ~ ^ ."""

def _md_to_tex(md: str) -> str:
    """Minimal, bounded markdown->LaTeX for method-card bodies ONLY:
    #/##/###/#### -> \\section*/\\subsection*/\\subsubsection*/\\paragraph;
    `code` -> \\texttt{texesc(...)}; **b** -> \\textbf; *i* -> \\emph;
    - lists -> itemize; [t](u) -> t\\footnote{\\texttt{u}}; markdown tables are
    replaced by '(table omitted in LaTeX rendering — see the markdown card)';
    every other line texesc'd verbatim. Documented as lossy by design."""

def _compile_tex(tex: Path, timeout: int = 300) -> tuple[Path | None, bool, str]:
    """shutil.which('tectonic') is None -> (None, False, 'tectonic not found —
    wrote paper.tex; install tectonic (https://tectonic-typesetting.github.io)
    to compile, or use --format md'). Else subprocess.run([exe, tex.name],
    cwd=tex.parent, capture_output=True, text=True, timeout=timeout);
    returncode != 0 -> (None, False, last 2000 chars of stdout+stderr);
    TimeoutExpired caught -> (None, False, message). Success -> (pdf, True,
    'compiled'). NEVER raises for a missing/failed tectonic."""
```

`render_paper(format="latex")`: renders `paper.tex` (markdown NOT written on this branch),
runs `_compile_tex`, returns `PaperResult(markdown=None, tex=..., pdf=maybe, compiled=..., message=...)`.

### `paper.tex.j2` contract

`\documentclass{article}`; packages: `graphicx`, `booktabs`, `hyperref` only (tectonic fetches on
demand); after `\maketitle` a framed prominent banner
(`\noindent\fbox{\parbox{\linewidth}{\centering\textbf{ {{ banner }} }}}`); same seven sections as
md; tables via booktabs with `fmt`-formatted cells (en dash for missing); figures via
`\includegraphics[width=.8\linewidth]{ {{ f.relpath }} }` (PDF variant preferred when present);
References as embedded `thebibliography` (no BibTeX pass needed); ALL data-derived strings pass
through `texesc` (column names with `_`/`%`/`&` must not break compilation); method-card bodies
through `_md_to_tex`.

### Tests (append to `tests/test_report_paper.py`)

1. `texesc` unit: `texesc("50% & _x_ #$ ~^ \\")` produces each escaped form; idempotence NOT
   required (documented), but no character is dropped.
2. `_md_to_tex` unit on a snippet with heading/bold/backtick-code/list/table → asserts
   `\section*`, `\textbf`, `\texttt`, `itemize`, the table-omitted marker; no unescaped `_`
   outside `\texttt{...}`/commands (regex spot-check on a `foo_bar` input).
3. `render_paper(bundle, "latex")` (importorskip jinja2): `paper.tex` exists; contains
   `\documentclass{article}`, the banner text, all `\section{` names, `Herlands`; a dataset with a
   column renamed to `x_0%` (build a variant bundle) renders with `x\_0\%` and never the raw form.
4. tectonic missing (`monkeypatch.setattr(shutil, "which", lambda _: None)` inside natex.report.paper's
   namespace): `compiled is False`, `pdf is None`, message matches `tectonic`, NO exception,
   `paper.tex` still written.
5. `@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")`:
   full compile → `pdf.exists()`, `compiled is True`. (Skips in CI; runs locally when present.)
6. Figures included: with a figures manifest, tex contains `\includegraphics` and the relpath.

**Commit:** `feat(report): LaTeX paper template with graceful tectonic compile`

---

## Task 6 — CLI `natex paper`

**Modify:** `src/natex/cli.py` (new command; module-level `from natex.report.bundle import
ResultsBundle` and `from natex.report.paper import render_paper` are safe — neither imports an
optional dep at module level).
**Create:** `tests/test_cli_paper.py`.

```python
@app.command()
def paper(
    bundle: Path = typer.Option(..., "--bundle",
        help="results bundle dir (ResultsBundle.save, or a discover --out dir)"),
    format: str = typer.Option("md", help="md|latex; latex also compiles when tectonic is on PATH"),
    out: Path = typer.Option(None, help="output dir; default BUNDLE/paper"),
):
    """Render the AI-draft paper from a results bundle (markdown always works)."""
```

Behavior: format not in {"md","latex"} → echo naming it, exit 2. `ResultsBundle.load` failure
(`FileNotFoundError`/`ValueError`/`KeyError`/`json.JSONDecodeError`) → echo, exit 2 (no traceback).
`render_paper` ImportError (jinja2 missing) → echo the install message, exit 2 (matches the
`_make_backend` convention). Success: echo one line per artifact (`paper: <md-or-tex path>`, then
`pdf: <path>` when compiled, else the compile message verbatim), echo the banner reminder line
("review before sharing — AI-generated draft"), exit 0 EVEN when tectonic is absent (tex written).

### Tests (`tests/test_cli_paper.py`, CliRunner)

1. Happy path (importorskip jinja2): build `make_rdd_bundle(tmp_path)`, run
   `["paper", "--bundle", str(dir)]` → exit 0; output contains `paper.md` path; file exists;
   output contains "AI-generated".
2. `--format banana` → exit 2, output names "banana". Nonexistent bundle dir → exit 2, message
   names `results.json`.
3. `--format latex` with `shutil.which` monkeypatched to None (patch inside `natex.report.paper`)
   → exit 0, output mentions tectonic, `paper.tex` exists.
4. jinja2 missing (`monkeypatch.setitem(sys.modules, "jinja2", None)`) → exit 2, output matches
   `natex-discovery\[report\]`.
5. A discover-only dir (only `discover_report.json`) works end to end — proves the load adaptation.

**Commit:** `feat(cli): natex paper renders md/latex drafts from a results bundle`

---

## Task 7 — paperbanana adapter (`natex/report/paperbanana.py`)

**Create:** `src/natex/report/paperbanana.py`, `tests/test_paperbanana.py`.
**Modify:** `pyproject.toml` — add
`paperbanana = ["paperbanana>=0.1"]  # diagram generation; needs its own provider key; CI never installs this`.

```python
def _pipeline_description(bundle: ResultsBundle) -> str:
    """Pure text: the natex pipeline as run — designs scanned (LoRD3 / SuDDDS by
    name), coverage counts, validation battery stages, estimators used, figure
    list. Deterministic; fully unit-testable without paperbanana."""

def generate_method_diagram(bundle: ResultsBundle, out: str | Path) -> Path:
    """Lazy `import paperbanana`; ImportError -> ImportError('method diagrams
    require the paperbanana extra: pip install "natex-discovery[paperbanana]"').
    Documented single call contract: paperbanana.generate_diagram(
    description=<str>, output_path=str(out)); returns Path(result or out).
    The adapter is the ONLY place to update if the real library's API differs
    (tested contract = the fake module below)."""
```

### Tests (no network, ever)

1. `_pipeline_description(make_rdd_bundle(...))` contains "LoRD3", the n_scanned/n_total counts,
   "randomization", "2SLS"; did bundle → contains "SuDDDS". Deterministic: two calls equal.
2. Fake module: `monkeypatch.setitem(sys.modules, "paperbanana",
   types.SimpleNamespace(generate_diagram=recorder))` where recorder captures kwargs and writes a
   1-byte file → `generate_method_diagram(bundle, out)` returns the path, file exists, recorded
   `description` contains "LoRD3" and `output_path == str(out)`.
3. Missing module (`monkeypatch.setitem(sys.modules, "paperbanana", None)`) → ImportError matching
   `natex-discovery\[paperbanana\]`.

**Commit:** `feat(report): paperbanana method-diagram adapter behind the paperbanana extra`

---

## Task 8 — Deep-research handoff (`natex/report/research_brief.py`)

**Create:** `src/natex/report/research_brief.py`, `tests/test_research_brief.py`.
**Modify:** `src/natex/report/__init__.py` (export `research_brief`).

```python
def research_brief(bundle: ResultsBundle, out: str | Path) -> Path:
    """Write research-brief.md (out is a directory unless it endswith '.md').
    Pure string generation from bundle.results — no network, no rng, no LLM.
    Numbers through the same _fmt helper (None/NaN -> '—')."""
```

Fixed section skeleton (exact headings — tests assert them):
`# Research brief: <source or 'dataset'>` · italic AI-generated note line ·
`## Discovery context` (data block + one-line coverage) · `## Discovered designs` (per SCANNED
config: rdd → cutoff center in raw forcing coordinates + top forcing influence + MC p; did → subset
values, T0, window, MC p) · `## Effect estimates` (per estimator with CI and weak-IV flag) ·
`## Validation status` · `## Open questions` (fixed prompts: confounded cutoffs, competing policies
at the same threshold, external validity, selection into the forcing variable) ·
`## Literature questions for deep research` — ≥ 3 numbered, discovery-specific questions built by
template, e.g. "What statutory or administrative rules set a threshold near {cutoff} on
{forcing var}?", "What quasi-experimental studies exploit {forcing var} thresholds in comparable
settings?", did: "What other policies changed at {T0} that could affect {subset}?" ·
`## How to use` — paste this brief verbatim as the query to your deep-research skill
(Gemini Interactions API); natex does not call any research API itself.

### Tests

1. On `make_rdd_bundle`: file written at `out/research-brief.md`; all eight headings present;
   ≥ 3 lines matching `^\d+\.` under the literature section, each ending in `?`; the cutoff
   center coordinates (formatted) appear; "weak" flag text present when the 2sls row has it.
2. On `make_did_bundle`: T0 value and a subset value string appear.
3. Determinism: two calls → byte-identical files. `\bnan\b|\bNone\b` absent (case-sensitive).
4. `out` endswith ".md" → written exactly there. No-outcome bundle → effects section renders
   "no effect estimates (no outcome column was provided)" and nothing crashes.

**Commit:** `feat(report): deep-research handoff brief generator`

---

## Task 9 — README, phase status, final verification

**Modify:** `README.md` — new section `## From discovery to paper` (between Quickstart and
Backtests), covering: the three-command flow

```bash
natex study data.csv --context "where the data came from" --out out
natex discover data.csv --plan out/intake_report.json --out out
natex paper --bundle out --format md     # or --format latex (compiles when tectonic is installed)
```

plus the Python-API variant (`ResultsBundle.from_discover(...)`, `rdd_figures`/`did_figures`,
`render_paper`, `research_brief`); the deep-research handoff paragraph (research-brief.md is a
ready-to-paste query for a Gemini deep-research agent; natex performs no research calls); the
manual Google Docs route — export markdown and paste it into a Google Doc, or upload the `.md` via
Google Drive and open with Docs; **natex does NOT integrate the Google Docs API**; the human-in-the-
loop rule (every draft carries the AI-generated banner; verify all claims before circulation);
install lines for the `report`, `plot` and `paperbanana` extras. Update the Roadmap section's
reporting line to done.

**Create:** `docs/status/phase-report-paper.md` — short status in the established format: what
shipped (modules, CLI, templates, extras), test counts, CI matrix state, known limitations
(`_md_to_tex` lossy on tables; paperbanana contract tested against the fake module only; installed-
wheel method-card fallback), and the run-of-record commands.

**Final verification (before the last commit):**
`uv run ruff check src tests` and `uv run pytest -q` green on the local interpreter; spot-run
`uv run pytest -q tests/test_report_bundle.py tests/test_report_figures.py tests/test_report_paper.py
tests/test_cli_paper.py tests/test_paperbanana.py tests/test_research_brief.py` with and without
the extras installed locally when feasible (`uv sync --extra dev` vs `--extra dev --extra plot
--extra report`) to prove the skip paths; backtests untouched (none added).

**Commit:** `docs: README 'From discovery to paper' section and phase-report-paper status`

---

## Task ordering & dependencies

1 (bundle) → 2 (period_gaps) → 3 (figures; needs 1+2) → 4 (md paper; needs 1) → 5 (latex; needs 4)
→ 6 (CLI; needs 4+5) → 7 (paperbanana; needs 1) → 8 (research brief; needs 1) → 9 (docs; needs all).
Tasks 7 and 8 are independent of 3–6 and may be reordered if convenient, but keep one commit each.
