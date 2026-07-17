# Method card — one-command survey (`natex survey`)

**Modules:** `natex.survey.registry` / `natex.survey.applicability` / `natex.survey.runner` /
`natex.survey.figures`; report renderers in `natex.report.survey_html`.
**CLI:** `natex survey CSV [--context ...] [--backend ...] [--seed N] [--cutoff COL=VALUE ...]
[--instrument COL ...] [--threshold COL=VALUE ...] [--time COL --unit COL] [--out DIR]`.
**Skill:** [skills/natex-survey](../../skills/natex-survey/SKILL.md).
**Governing math:** [docs/math_audit_final.md](../math_audit_final.md) — the audit wins every
conflict. **Plan of record:** [docs/plans/phase-survey.md](../plans/phase-survey.md).

Fixed family order: `rdd, did, kink, iv, sc, bunching, dee`

## What it does

ONE command runs a dataset systematically against ALL seven natex method families and writes
one visual report with an applicability verdict per family — including reasoned skips. The
survey layer adds NO new inference code: every statistic comes from the existing per-family
modules (`natex.discover` for rdd and did; `regression_kink` + `sensitivity_grid`;
`discover_instruments`; `select_donors` + `sc_placebo_test`; `binned_poisson_jump`;
`lord3_scan` + `dee_debias`).

## Flow

```
CSV (or DataFrame)
   │
   ▼
profile + study-style understanding    natex.intake.analyst.study — column roles, prep plan,
   │                                   ranked candidates; NullBackend heuristics offline, or
   │                                   an agent/LLM guidance backend
   ▼
applicability                          registry predicates over the profile and declared
   │                                   inputs only; optional analyst override, every
   │                                   override recorded
   ▼
per-family pipelines                   FAMILY_ORDER: rdd → did → kink → iv → sc → bunching
   │                                   → dee; each family isolated — a failure is recorded
   │                                   and the survey continues
   ▼
figures                                presentation-only glue over natex.report.figures;
   │                                   needs the plot extra
   ▼
report                                 survey.json → report.md (always) + report.html
                                       (report extra; self-contained, base64 figures)
```

Determinism: `survey()` raises ValueError without an explicit `numpy.random.Generator`; the
first stochastic act is ONE upfront `rng.spawn(7)` in registry order, so a skipped family
never shifts another family's random stream. Same seed ⇒ same `survey.json` (the `created`
timestamp is the single nondeterministic field).

## Applicability verdicts

Heuristic verdicts come from declarative requirements registered per family
(`natex.survey.registry`): predicates over the intake profile, an optional declared
`DatasetSpec`, and user-declared inputs ONLY — never over dataset content. No predicate
accepts a DataFrame, and a recording-stub test proves the profile attributes they may touch.
All requirements met ⇒ `applicable`; every unmet requirement user-suppliable (kink cutoffs,
iv instruments, bunching thresholds) ⇒ `needs_input`; otherwise ⇒ `inapplicable`.

The optional `method_applicability` guidance task lets the analyst override heuristics BOTH
ways. Every override is recorded (`heuristic_said` vs `analyst_said` plus the analyst's
reason) — a report reader can always see who decided what. Hint hygiene: proposed cutoff,
instrument, and threshold columns must exist in the profile and be numeric; invalid hints
are dropped and recorded, and explicit user declarations always win. Analyst proposals feed
config (cutoffs, instruments, thresholds, treated unit, t0) — never statistics: the
statistics are bitwise identical with and without guidance.

## Status semantics

| status | meaning |
|---|---|
| `credible` | the family ran and its design-specific gate passed (see below) |
| `null` | the family ran; no credible design at its gate — a real, reported outcome |
| `skipped` | the family did not run: heuristics said inapplicable, the analyst said skip, or a runtime gate held it back (dee without a credible rdd) |
| `needs_input` | requirements unmet that only the user can supply — declare them and re-run |
| `failed` | the family raised; the verbatim error is recorded and other families are unaffected |

Design-specific `credible` gates (ALPHA = 0.05 unless stated):

- **rdd** — scan p ≤ ALPHA, placebo battery passed, density p > ALPHA. A failed placebo
  battery demotes with the audit-3 phrasing "descriptive only — placebo battery failed".
- **did** — scan p ≤ ALPHA plus the composition and anticipation checks passed.
- **kink** — min Holm-adjusted kink p across the declared cutoffs ≤ ALPHA.
- **iv** — instruments selected AND a strong first stage on BOTH halves of the honest
  split; a weak half demotes to `null` (audit 10).
- **sc** — in-space +1-rank RMSPE-ratio placebo p ≤ SC_ALPHA = 0.10. The coarser gate is a
  granularity rationale, not a lower bar: the +1-rank p has granularity 1/(n_used+1), so
  with few usable placebo donors 0.05 is often unattainable by construction.
- **bunching** — min Holm-adjusted binned-Poisson p at the declared thresholds ≤ ALPHA.
- **dee** — a documented status-semantics stretch: dee is a surface fit, not a hypothesis
  test. `credible` means the debiased CATE surface was fitted over a usable ensemble of
  discovered experiments; a degenerate ensemble reports `null` with the diagnostic reason.

## Defaults that carry NO optimality claim

- **kink bandwidth** — the median absolute distance |running − cutoff| (about half the
  sample in-window), with a 0.5× / 1× / 2× sensitivity grid recorded per cutoff. A survey
  convenience; the dedicated `natex kink` command requires an explicit bandwidth for a
  reason.
- **dee query lattice** — 8 points per forcing dimension (the `natex debias` default is
  15): breadth over surface resolution for a survey pass.
- **ALPHA = 0.05 and SC_ALPHA = 0.10** — conventional reporting gates for the verdict
  table, not tuned decision thresholds; the underlying p-values are always reported
  alongside.

## dee gates

dee carries two gates on top of the shared row/column requirements:

- **gp-extra applicability gate** — dee is `inapplicable` unless torch + gpytorch are
  installed (`pip install "natex-discovery[gp]"`). This is an environment predicate (still
  content-blind) marking an explicit opt-in to the heaviest family. The GP the survey run
  actually fits is the CORE numpy `HeteroskedasticGP` — the gp extra's GPyTorch backend
  exists only for scale and is not exercised by the survey.
- **runtime gate** — dee runs only after a `credible` rdd family result: it debiases a
  VALIDATED discovery, and without one it is `skipped` with the reason "no validated rdd
  discovery to debias".

## Report anatomy

- **Header** — title with the dataset source; the banner "**AI-generated — verify before
  citing**"; dataset shape; time column and range; natex version, seed, and created
  timestamp; the free-text context when given.
- **Verdict table** — one row per family: name, status badge, one-sentence reason. Badges
  use the Okabe–Ito palette (✓ credible, ○ null, – skipped, ⚠ needs input, ✗ failed).
- **Per-family sections** — the registry's plain-language description verbatim; the verdict
  and reason; the applicability line (heuristic status and whether the family ran); any
  recorded analyst override; a key-numbers table; embedded figures (base64 in the HTML,
  relative paths in the markdown) or the recorded no-figure reason; diagnostics; the
  verbatim error for failed families; and the family's honest-inference caveat lines.
- Missing or non-finite numbers render as the em dash "—" at every depth — a literal
  missing-value token never reaches a rendered page.

## Extras degradation matrix

The survey degrades gracefully: `survey.json`, per-family `families/<name>.json`, and
`report.md` are ALWAYS written on a core-only install, and a degraded artifact is a
recorded outcome, never a crash (`natex survey` exits 0 whenever `survey.json` was
written).

| extras installed | figures | report.html | report.md + survey.json |
|---|---|---|---|
| core only | per-family reason: `no figure: matplotlib not installed (pip install "natex-discovery[plot]")` | absent; install note recorded under coverage notes and echoed by the CLI | always |
| `plot` | rendered (PNG + PDF under `figures/`) | absent; install note recorded and echoed | always |
| `report` | per-family no-figure reason as above | rendered; figure slots show the no-figure reason | always |
| `plot` + `report` | rendered | rendered, self-contained with base64-embedded PNGs | always |

The `gp` extra gates dee applicability (above); the `llm` extra enables the
anthropic/gemini guidance backends (`--backend agent` and the default `--backend null` work
without it).

## Audit corrections that bind the survey

Verbatim from the phase plan (audit numbers refer to `docs/math_audit_final.md`):

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

## Isolation and failure policy

Each family runs inside a documented isolation boundary: an exception is caught and
recorded (`status="failed"`, verbatim error, traceback in the family's JSON) and the survey
moves on — KeyboardInterrupt and SystemExit still propagate. Figure rendering has its own
narrower isolation (a named exception list): a rendering failure records `figure rendering
failed: ...` and never changes a family's statistical verdict. Discovery never reads the
outcome column (delegated to the reused modules), and failed statistics propagate as
missing values — never fabricated zeros.
