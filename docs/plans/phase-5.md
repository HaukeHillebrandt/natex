# Phase 5 implementation plan — IV/SC discovery (iv/)

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` §3 (iv/: "instrument
search: Lasso first-stage selection over candidate pools (Belloni-style) + strength/exclusion
diagnostics; synthetic-control donor selection with pre-trend scoring"), §5 items 4 & 10, §5/§3
adopted improvement "Weak-IV-robust (AR/Fieller) local inference at discovered cutoffs",
§9 phase 5 ("IV/SC discovery (iv/) — smallest, spec'd from the Springer roadmap").
**Method notes:** `".../RDD/docs/notes/read_springer-chapter.md"` — the Chen et al. (2026) roadmap
is CONCEPTUAL ONLY (its own notes warn it "cannot adjudicate implementation details"); the primary
sources natex implements are **Belloni, Chen, Chernozhukov & Hansen (2012, Econometrica)** for
instrument selection and **Abadie, Diamond & Hainmueller (2010, JASA)** for synthetic-control
donor selection and in-space placebo inference. Deep IV / GAN / text-matching material from the
chapter is OUT OF SCOPE for v1 (heavy deps, no benchmark row) — documented in the method card,
not silently dropped.
**Real data this phase:** only `prop99` (already registered) for the donor-selection backtest;
spec §8 has no IV row, so the IV gate is synthetic. Phases 2–4 backtests must stay green.

## Phase objective (spec §9 phase 5)

1. **Instrument search** (`src/natex/iv/search.py`): Belloni-style Lasso first-stage selection
   over a candidate instrument pool — iterated plug-in penalty with heteroskedasticity-robust
   loadings, post-Lasso OLS first stage, strength diagnostics (HC1 first-stage F, partial R²,
   weak flag). Selection reads only `(T, pool, controls)` — never the outcome (the phase's
   discovery-honesty analog).
2. **General k-instrument 2SLS + weak-IV-robust inference** (`src/natex/estimate/iv2sls.py`):
   HC1-sandwich 2SLS (audit item 4 policy: never the printed group-instrument form), Hansen-J
   overidentification diagnostic (the only testable slice of "exclusion"), and closed-form
   Anderson–Rubin/Fieller confidence sets (audit §3 adopted, still unimplemented after phase 4) —
   wired additively into the existing `local_2sls` at discovered cutoffs.
3. **Honest instrument-discovery pipeline** (`src/natex/iv/pipeline.py`): select on a discovery
   half, estimate + J-test on the estimation half (`validate.honest.honest_split`).
4. **Synthetic-control donor selection with pre-trend scoring** (`src/natex/iv/donors.py`):
   unit×time outcome matrix, per-donor pre-trend scores, top-k donor pool, simplex weights
   (shared helper extracted from `did/controls.py`), counterfactual + post-period ATT, and
   Abadie in-space RMSPE-ratio placebo inference with +1-rank p-values.
5. **Synthetic DGPs + benchmarks**: `data/synthetic_iv.py` (sparse first stage, tunable
   concentration parameter, plantable exclusion violators), `data/synthetic_sc.py` (factor-model
   panel with known donor support), `benchmarks/run_iv_selection.py` + CI-small slices.
6. **Prop 99 donor backtest** (`-m backtest`), CLI `natex instruments` / `natex donors`,
   `docs/method_cards/iv_sc.md`, `docs/status/phase-5.md`, README roadmap tick.

Out of scope (documented in the method card, not silently dropped): Deep IV (torch, no benchmark),
sup-score weak-identification-robust selection, Montiel Olea–Pflueger effective F (HC1 Wald F +
the F<10 convention documented as a heuristic instead), covariate V-weights in SC (same phase-3
deviation), matrix-completion/elastic-net SC variants, staggered adoption.

## Audit corrections that bind this phase (docs/math_audit_final.md)

| # | Correction | Where implemented |
|---|---|---|
| 4 | Group-instrument (Eq 5.14) inconsistent — 2SLS is the only estimator family; W = T − μ is never implemented | `estimate/iv2sls.py::iv_2sls` (general k-instrument HC1 2SLS, same policy as `local2sls`) |
| 10 | First-stage relevance is not implied by selection/LLR — always compute the actual first-stage F after selection; flag weak | `iv/search.py` (post-Lasso HC1 F, `weak` flag), `iv/pipeline.py` (re-checked on the estimation half) |
| §3 adopted | **Weak-IV-robust (AR/Fieller) intervals** | `estimate/iv2sls.py::ar_confidence_set` (closed-form quadratic; k=1 is exactly Fieller); additive `ar_ci`/`ar_kind` on `IVEstimate` and `EffectEstimate` (computed in `local_2sls`) |
| 1 lineage | +1-rank Monte Carlo p-values everywhere a rank test appears | `iv/donors.py::sc_placebo_test` (p = (1 + #{placebo ratio ≥ treated}) / (n_placebos + 1)) |
| 1 lineage | Honest discovery/estimation splits as the post-selection guarantee | `iv/pipeline.py::discover_instruments` (select on split A, estimate + J on split B; full-sample mode prints a documented caveat) |
| 3 lineage | Exclusion is untestable; the J statistic tests only overidentifying restrictions given ≥1 valid instrument — never claim more | `estimate/iv2sls.py` docstring + method card; `j_p=None` when just-identified, never a fabricated 1.0 |
| 5 lineage | Two-sided placebo inference, matched shapes | `sc_placebo_test` uses two-sided RMSPE ratios; placebos get the SAME selection rule (`n_donors`, scoring) as the treated run |
| 24 lineage | No absolute variance/tolerance floors — scale-normalized SSE in the simplex fit (already regression-tested in phase 3) survives the refactor verbatim | `estimate/simplex.py` (extraction keeps the `scale` normalization and its test) |
| 8 (spec §5) | NaN never 0.0 on failure | empty Lasso selection → `selected=[]`, F/tau NaN, `weak=True`; no donors / no defined post period → NaN ATT, `pre_rmspe=+inf` never 0 |

Also inherited: discovery never reads `y` — instrument SELECTION touches only `(T, pool,
controls)`; donor SCORING/WEIGHTING touches only PRE-period outcomes (post-period `y` is the
estimation target; this inherent SC use of pre-outcomes is a documented method property, tested
by mutation tests). One `numpy.random.Generator` through every stochastic call; no bare except.

## House rules (bind every task)

Python ≥3.11; core deps only numpy/scipy/pandas/scikit-learn/typer/pydantic — **this entire phase
is core-deps** (sklearn `Lasso`, scipy `optimize`/`stats`); no new dependencies, no new extras.
CI (3.11–3.14) must stay green. One `numpy.random.Generator` through every stochastic call
(mirror the repo's `raise ValueError("pass an explicit numpy Generator")` convention); identical
seed ⇒ identical output. NaN never 0.0. Never commit datasets. Conventional commit after every
green cycle. `uv run pytest -q` excludes backtests; backtests run with `-m backtest` and
`NATEX_DATA` set (final gate, task 12).

**TDD discipline for every task:** write the failing test(s) first, run to confirm failure,
implement, run `uv run pytest -q` and `uv run ruff check src tests`, then commit.

**Statistical-test policy:** every stochastic assertion is seeded; calibrate thresholds across ≥5
seeds during implementation, then pin one seed with a margin. Thresholds below are starting
points — change only with a code comment stating the calibration evidence.

## Current interfaces built upon (do not break; all changes additive)

- `natex.data.spec.Dataset / DatasetSpec`: `n, T, y, Z, Z_std, X, treatment_is_binary,
  standardize`; `spec.unit` / `spec.time` optional columns; constructor drops NaN scan rows but
  NEVER rows with NaN outcome.
- `natex.estimate.local2sls.local_2sls / wald_estimate -> EffectEstimate(tau, se, ci, method,
  first_stage_jump, first_stage_t, weak_instrument, n_used)` — new fields MUST default so
  phase-1–4 constructions still typecheck (`ar_ci: tuple[float,float] | None = None`,
  `ar_kind: str | None = None`).
- `natex.validate.placebo.hc1_ols(Xmat, yvec) -> (beta, se)` — reuse for every OLS with HC1.
- `natex.validate.honest.honest_split(n, frac_discovery, rng) -> (idx_a, idx_b)`.
- `natex.did.controls.synthetic_control` internals to extract: the SLSQP simplex fit with
  scale-normalized SSE (regression test `test_synthetic_control_scale_invariant_optimization`)
  and `_weighted_counterfactual_by_time` + `_MISSING_W_TOL = 0.1`.
- `natex.data.registry.load_dataset("prop99")` → Dataset with `unit="state"`, `time="year"`,
  binary treatment (California×post-1988), outcome = per-capita cigarette packs.
- `tests/backtests/conftest.py::load_or_skip`; CLI typer `app` in `natex/cli.py` with `_clean`
  JSON helper; CLI tests use `typer.testing.CliRunner` (see `tests/test_cli.py`).
- DGP style templates: `data/synthetic.py::make_synthetic`, `data/synthetic_did.py`,
  `data/synthetic_dee.py` (dataclass result carrying `df` + ground truth + documented repairs).

## Design conventions fixed for the whole phase

- **Belloni plug-in penalty (pinned formulas).** Partial out `[1, controls]` from the endogenous
  `d = T` and every pool column by OLS residuals (Frisch–Waugh); then solve
  `min_π (1/n)‖d − Zπ‖² + (λ/n) Σ_j ψ_j |π_j|` with
  `λ = 2c √n Φ⁻¹(1 − γ/(2p))`, `c = 1.1`, `γ = 0.1 / log(max(n, p))` (documented choice), and
  heteroskedastic loadings `ψ_j = sqrt((1/n) Σ_i z_ij² ε̂_i²)`, iterated: ε̂⁰ = residual of `d`
  on the 5 pool columns most correlated with it (BCCH-style init; plain centered `d` if p < 5),
  refresh ψ from post-Lasso residuals, stop when the support is stable or after `max_iter`.
  **sklearn mapping (exact):** rescale `u_j = z_j / ψ_j`, `b_j = ψ_j π_j`; the objective equals
  `2·[(1/2n)‖d − Ub‖² + (λ/2n)‖b‖₁]`, i.e. `sklearn.linear_model.Lasso(alpha = λ/(2n),
  fit_intercept=False)` on centered data; recover `π_j = b_j / ψ_j`. Zero-variance pool columns
  get ψ_j = ∞ semantics (excluded up front with a diagnostic, never a divide-by-zero).
- **Anderson–Rubin closed form (pinned).** Partial `[1, controls]` out of `y, T, Z` → `ỹ, T̃, Z̃`
  (q columns partialled). With `P = Z̃(Z̃'Z̃)⁻¹Z̃'`, `r(τ) = ỹ − τT̃`,
  `AR(τ) = [r'Pr/k] / [r'(I−P)r/(n−q−k)]`; the level-α set is `{τ : g(τ) ≤ 0}` with
  `g(τ) = r'Ar`, `A = P − c_k(I−P)`, `c_k = k·F_crit(k, n−q−k; 1−α)/(n−q−k)` — a quadratic
  `aτ² + bτ + c` with `a = T̃'AT̃`, `b = −2ỹ'AT̃`, `c = ỹ'Aỹ`. Cases (report `ar_kind`):
  `a>0, disc>0` → `"interval"` [r₁, r₂]; `a>0, disc≤0` → `"empty"` (possible only when k ≥ 2 —
  model rejected at every τ); `a<0, disc>0` → `"disjoint"` (−∞, r₁]∪[r₂, ∞) with `ar_ci=None`
  and both rays in extras; `a<0, disc≤0` → `"unbounded"` (whole line, `ar_ci=None`);
  `a≈0` → linear fallback. Classical identity used as a test: the set is bounded **iff** the
  homoskedastic first-stage F exceeds `F_crit`. k=1 reduces exactly to Fieller.
- **Hansen J (pinned).** With 2SLS residuals `e` and partialled instruments `Z̃`:
  `J = (Z̃'e)' [Σ_i z̃_i z̃_i' e_i²]⁻¹ (Z̃'e)`, df = k − 1 (one endogenous regressor),
  p from χ²_{k−1}; `j_stat = j_p = None` when k = 1 (never a fabricated value).
- **2SLS sandwich (pinned, generalizes `local_2sls`).** `X = [1, controls, T]`,
  `Zfull = [1, controls, instruments]`, `X̂ = P_{Zfull} X`, `β = (X̂'X)⁻¹X̂'y`, `e = y − Xβ`,
  `cov = (X̂'X)⁻¹ X̂'diag(e²)X̂ (X'X̂)⁻¹ · n/(n−p)` (HC1). `pinv` throughout; rank deficiency →
  diagnostic in extras, NaN estimate if τ's column is in the null space.
- **Weak-instrument convention:** `weak = first_stage_F < 10.0` (same threshold family as the
  existing `f_t² < 10` convention; documented as a heuristic, Stock–Yogo caveat in method card).
- **Donor-selection honesty:** pre-trend scores, donor ranking, and simplex weights are computed
  from PRE-t0 outcomes only. Mutation tests enforce it (changing any post-period outcome changes
  neither scores nor weights).
- **SC placebo protocol:** each complete candidate donor becomes pseudo-treated at the SAME t0
  with the SAME `n_donors`/scoring rule; its donor pool excludes the actually-treated unit;
  ratio = post-RMSPE/pre-RMSPE; skip placebos with `pre_rmspe == 0` (diagnostic count); optional
  `exclude_poor_fit: float | None` drops placebos with pre-RMSPE > multiplier × treated's
  (Abadie's 2×/5×/20× convention, default None = keep all).
- **Determinism:** `select_instruments` is RNG-free for `lam="plugin"` and `lam=float` (assert
  two calls bitwise-equal); rng is required only for `lam="cv"` fold shuffling. Donor selection
  and SC placebos are RNG-free (SLSQP from the uniform start, as in phase 3).
- **NaN policy:** empty selection → `selected=[]`, `first_stage_F=NaN`, `weak=True`; `iv_2sls`
  with < p + 3 finite-y rows or empty instrument list → NaN estimate (never 0.0); no complete
  donors / no defined post period → NaN ATT with a diagnostic reason.

---

## Task 1 — Commit this plan; general k-instrument 2SLS + Hansen J

**First action of this task:
`git add docs/plans/phase-5.md && git commit -m "docs: phase 5 implementation plan"` — the plan
file is committed before any code.**

**Create:** `src/natex/estimate/iv2sls.py`, `tests/test_iv2sls.py`.

```python
# estimate/iv2sls.py
@dataclass
class IVEstimate:
    tau: float
    se: float
    ci: tuple[float, float]
    method: str                       # "2sls"
    first_stage_F: float              # HC1 Wald F of the instrument block in the first stage
    partial_r2: float                 # first-stage partial R^2 of instruments after controls
    weak_instrument: bool             # first_stage_F < 10.0 (NaN F -> True)
    j_stat: float | None              # Hansen J; None when just-identified (k == 1)
    j_p: float | None
    j_df: int                         # k - 1 (0 when k == 1)
    n_used: int                       # rows with finite (y, T, instruments, controls)
    ar_ci: tuple[float, float] | None = None   # filled by task 2
    ar_kind: str | None = None                 # filled by task 2
    extras: dict = field(default_factory=dict)

def iv_2sls(
    y: np.ndarray,
    T: np.ndarray,
    instruments: np.ndarray,          # (n, k), k >= 1
    controls: np.ndarray | None = None,   # (n, q) exogenous controls (intercept added inside)
    alpha: float = 0.05,
) -> IVEstimate: ...
```

Implementation per the pinned sandwich/J formulas above. Non-finite rows in ANY input column are
dropped (recorded as `extras["n_dropped"]`); underdetermined → NaN estimate via a `_nan_estimate`
twin of the `local2sls` helper. First-stage F: HC1 Wald test that the k instrument coefficients
are jointly zero in `T ~ [1, controls, instruments]` (chi²/k form; `scipy.stats`).

**Tests (`tests/test_iv2sls.py`), written first:**

- *Analytic just-identified case:* k=1, no controls — `iv_2sls` tau equals the Wald/IV ratio
  `cov(z̃,y)/cov(z̃,T)` (atol 1e-10); `j_stat is None`, `j_df == 0`.
- *Consistency:* seeded DGP (n=4000, one strong instrument, endogeneity 0.6, τ=1): 2SLS τ̂ within
  0.1 of 1.0 while OLS of y on T is biased by > 0.2 (calibrate, pin seed).
- *Overidentified:* k=3 valid instruments ⇒ `j_p` roughly uniform (assert j_p > 0.01 at pinned
  seed); planting one instrument directly in y (φ=0.8) drives `j_p < 0.01`.
- *HC1 SEs:* under heteroskedastic errors (σ_i ∝ |z_i|), 95% CI covers τ in ≥ 90 of 100 seeded
  replications (loose calibration bound; comment the observed rate).
- *Controls:* adding a control that shifts both T and y leaves τ̂ consistent; omitting it breaks
  consistency (sanity check of the partialling path).
- *NaN policy:* y with all-NaN ⇒ NaN tau/se/ci, `weak_instrument is True`, `n_used == 0`; a
  constant instrument column ⇒ NaN estimate + `extras["rank_deficient"]`, never 0.0.
- *First-stage F:* strong design F > 100; pure-noise instrument F < 5 and `weak_instrument`.

Commit: `feat(estimate): general k-instrument HC1 2SLS with Hansen J diagnostics`.

## Task 2 — Anderson–Rubin/Fieller confidence sets; wire into `local_2sls`

**Create:** `tests/test_ar_ci.py`.
**Modify:** `src/natex/estimate/iv2sls.py`, `src/natex/estimate/local2sls.py`.

```python
# estimate/iv2sls.py (additions)
@dataclass
class ARSet:
    kind: str                          # "interval" | "empty" | "disjoint" | "unbounded"
    interval: tuple[float, float] | None   # for "interval"; None otherwise
    rays: tuple[tuple[float, float], tuple[float, float]] | None  # for "disjoint": (-inf,r1],[r2,inf)
    ar_at_2sls: float                  # AR statistic evaluated at the 2SLS point estimate
    f_crit: float

def ar_confidence_set(
    y: np.ndarray, T: np.ndarray, instruments: np.ndarray,
    controls: np.ndarray | None = None, alpha: float = 0.05,
) -> ARSet: ...
```

Closed-form quadratic per the pinned convention. `iv_2sls` calls it and fills
`ar_ci`/`ar_kind` (+ `extras["ar_rays"]` when disjoint). `EffectEstimate` gains
`ar_ci: tuple[float, float] | None = None` and `ar_kind: str | None = None` (additive, defaulted);
`local_2sls` computes the k=1 AR set with the frozen oriented side indicator as the instrument and
`[1, s, s·g]` as controls, on the same finite-y rows. `wald_estimate` and all `_nan_estimate`
paths leave the defaults (None).

**Tests:**

- *Fieller equivalence (k=1):* hand-computable 6-point dataset — the quadratic roots equal the
  hand-solved Fieller bounds (atol 1e-8).
- *Strong-instrument agreement:* seeded strong design ⇒ `kind == "interval"` and AR bounds within
  15% relative of the Wald CI half-width.
- *Weak-instrument honesty:* near-zero first stage ⇒ `kind in {"unbounded", "disjoint"}` and
  `ar_ci is None` — never a finite fabricated interval; boundedness ⇔ homoskedastic first-stage
  F > F_crit (assert the iff on both sides of a strength sweep).
- *Empty set:* k=2 with grossly invalid second instrument ⇒ `kind == "empty"` reachable; assert
  `kind != "empty"` whenever k == 1 (structural impossibility).
- *Coverage:* 200 seeded weak-instrument replications (concentration μ² ≈ 4): AR set covers the
  true τ in ≥ 92% (target 95%; Wald covers materially less — record both rates in a comment).
- *local_2sls wiring:* sharp phase-1 synthetic (make_synthetic) top discovery ⇒ finite `ar_ci`
  overlapping the 2SLS CI; all pre-existing `tests/test_estimate*.py` pass unmodified.

Commit: `feat(estimate): closed-form Anderson-Rubin/Fieller sets; ar_ci on local 2SLS`.

## Task 3 — Sparse-first-stage IV synthetic DGP

**Create:** `src/natex/data/synthetic_iv.py`, `tests/test_synthetic_iv.py`.

```python
# data/synthetic_iv.py
@dataclass
class IVSyntheticData:
    df: pd.DataFrame                  # columns z1..zp, T, y
    pool_names: list[str]             # ["z1", ..., "zp"]
    true_support: list[str]           # the s relevant instruments
    invalid_names: list[str]          # exclusion violators (subset of non-support columns)
    tau: float
    pi: np.ndarray                    # (p,) true first-stage coefficients
    concentration: float              # realized n * pi' Sigma pi / sigma_v^2

def make_iv_synthetic(
    n: int = 500,
    p: int = 50,
    s: int = 5,
    mu2: float = 180.0,               # target concentration parameter
    rho_z: float = 0.5,               # Toeplitz corr rho_z**|j-k| among instruments
    endog: float = 0.6,               # corr(first-stage error v, structural error e)
    tau: float = 1.0,
    n_invalid: int = 0,               # candidates entering y directly with coef phi
    phi: float = 0.5,
    rng: np.random.Generator | None = None,   # required (repo convention)
) -> IVSyntheticData: ...
```

BCCH "exponential design": `pi_j ∝ 0.7**(j-1)` on the first s columns, rescaled so the
population concentration `n·π'Σπ/σ_v²` equals `mu2`; `(v, e)` bivariate normal with unit
variances and correlation `endog`; invalid instruments drawn from the LAST `n_invalid` pool
columns (never overlapping `true_support`; assert s + n_invalid ≤ p).

**Tests (seeded):**

- *Shape/determinism:* same seed bitwise-identical df; different seeds differ.
- *Concentration targeting:* realized `concentration` within 25% of `mu2` for mu2 ∈ {30, 180}.
- *Endogeneity:* OLS τ̂ − τ > 0.15 (median over 5 seeds, n=2000) while 2SLS on `true_support`
  via task-1 `iv_2sls` has |bias| < 0.05.
- *Invalid instruments:* with n_invalid=2 in the instrument set, Hansen `j_p < 0.01`; with valid
  overidentification `j_p > 0.05` (pinned seeds).
- *rng required:* omitting rng raises ValueError (repo convention).

Commit: `feat(data): sparse-first-stage IV DGP with targeted concentration and plantable exclusion violators`.

## Task 4 — Belloni plug-in Lasso instrument selection

**Create:** `src/natex/iv/__init__.py`, `src/natex/iv/search.py`, `tests/test_iv_search.py`.

```python
# iv/search.py
@dataclass
class InstrumentSearchResult:
    selected: list[str]               # post-Lasso support, pool_names order
    pi_lasso: np.ndarray              # (p,) weighted-Lasso coefs on the ORIGINAL column scale
    pi_post: np.ndarray               # (p,) post-Lasso OLS coefs (0.0 off-support)
    lam: float                        # penalty actually used
    loadings: np.ndarray              # (p,) final penalty loadings psi_j
    first_stage_F: float              # HC1 Wald F of selected block (NaN when selected == [])
    partial_r2: float                 # NaN when selected == []
    weak: bool                        # F < 10.0 or NaN
    n_iter: int                       # plug-in iterations performed
    extras: dict = field(default_factory=dict)   # dropped zero-variance cols, support trace, ...

def select_instruments(
    T: np.ndarray,
    pool: np.ndarray,                 # (n, p) candidate instruments
    controls: np.ndarray | None = None,
    pool_names: list[str] | None = None,       # default ["z1", ...]
    lam: float | str = "plugin",      # "plugin" | "cv" | explicit lambda
    c: float = 1.1,
    gamma: float | None = None,       # default 0.1 / log(max(n, p))
    max_iter: int = 15,
    rng: np.random.Generator | None = None,    # required iff lam == "cv"
) -> InstrumentSearchResult: ...
```

Per the pinned plug-in/sklearn-mapping convention. `lam="cv"` uses `LassoCV` with rng-seeded
5-fold shuffling (documented caveat: CV λ voids the sparsity guarantee; plug-in is the default).
**Never reads any outcome** — the function signature cannot even receive y.

**Tests, written first:**

- *Analytic soft-threshold:* single standardized instrument, ψ=1 —
  `pi_lasso = soft(z'd, λ/2)/(z'z)` (atol 1e-8) — pins the sklearn alpha mapping.
- *Recovery:* `make_iv_synthetic(n=500, p=50, s=5, mu2=180)` ⇒ `selected` contains the top-3
  strongest true instruments and no more than 2 false positives (calibrate over ≥5 seeds, pin
  one); `first_stage_F > 10`, `weak is False`.
- *Null pool:* pure-noise pool (π=0) ⇒ `selected == []` in ≥ 4/5 pinned seeds, F is NaN,
  `weak is True` — the honest "no instrument found" result.
- *Controls partialling:* a control that drives both T and a pool column: with `controls` passed
  the spurious column is not selected; without, it is (demonstrates FWL path works).
- *Determinism:* `lam="plugin"` twice ⇒ bitwise-identical result without any rng; `lam="cv"`
  without rng raises ValueError.
- *Zero-variance column:* excluded up front, reported in extras, no NaN/inf in loadings.
- *Explicit lam:* tiny λ selects (almost) everything; huge λ selects nothing — monotone support
  size across a 3-point λ grid.

Commit: `feat(iv): Belloni plug-in Lasso instrument selection with post-Lasso strength diagnostics`.

## Task 5 — Honest instrument-discovery pipeline + exclusion diagnostics

**Create:** `src/natex/iv/pipeline.py`, `tests/test_iv_pipeline.py`.

```python
# iv/pipeline.py
@dataclass
class InstrumentDiscovery:
    search: InstrumentSearchResult    # fitted on the discovery half (or full sample)
    estimate: IVEstimate | None       # 2SLS + J + AR on the estimation half; None if no outcome
    honest: bool
    n_discovery: int
    n_estimation: int
    extras: dict = field(default_factory=dict)

def discover_instruments(
    df: pd.DataFrame,
    treatment: str,
    pool: list[str],
    outcome: str | None = None,
    controls: list[str] | None = None,
    honest: bool = True,
    frac_discovery: float = 0.5,
    lam: float | str = "plugin",
    rng: np.random.Generator | None = None,   # required when honest=True (split) or lam="cv"
) -> InstrumentDiscovery: ...
```

`honest=True`: `honest_split` the rows; `select_instruments` sees ONLY the discovery half;
`iv_2sls` (with the selected columns) sees ONLY the estimation half — the selection event is
independent of the estimation noise, so J/AR/Wald p-values on the estimation half need no
selection correction (docstring states this precisely). `honest=False` runs both on the full
sample with `extras["caveat"]` set (post-selection inference not corrected). NaN handling
delegated to the components; empty selection ⇒ `estimate` is a NaN IVEstimate with
`extras["reason"] = "empty selection"`.

**Tests:**

- *End-to-end recovery:* `make_iv_synthetic(n=1000, p=40, s=4, mu2=180)`: honest pipeline τ̂
  within 0.15 of 1.0, `ar_kind == "interval"`, `weak is False` (pinned seed).
- *Honesty mechanics:* the selection result is bitwise-identical to calling
  `select_instruments` on the discovery-half rows directly; estimation rows ∩ discovery rows = ∅.
- *Exclusion flag:* n_invalid=2 planted violators selected into the instrument set ⇒ estimation-
  half `j_p < 0.05` (calibrate; pin seed) while the all-valid run has `j_p > 0.05`.
- *Weak path:* mu2=2 ⇒ `weak is True` and `ar_kind in {"unbounded", "disjoint"}` — the pipeline
  reports honestly rather than a tight fake CI.
- *No outcome:* `outcome=None` ⇒ `estimate is None`, search still runs (discovery reads no y).
- *rng contract:* honest=True without rng raises ValueError; same seed ⇒ identical split and
  results.

Commit: `feat(iv): honest instrument-discovery pipeline with estimation-half J and AR inference`.

## Task 6 — Extract the shared simplex-weight fitter (refactor, tests as guard)

**Create:** `src/natex/estimate/simplex.py`, `tests/test_simplex.py`.
**Modify:** `src/natex/did/controls.py` (imports the helper; behavior unchanged).

```python
# estimate/simplex.py
MISSING_W_TOL = 0.1   # moved from did/controls._MISSING_W_TOL (re-exported there)

@dataclass
class SimplexFit:
    weights: np.ndarray               # (n_donors,) w >= 0, sum w = 1
    sse: float                        # de-normalized to target units
    converged: bool

def fit_simplex_weights(y_target: np.ndarray, Y_donors: np.ndarray) -> SimplexFit:
    """(n_common,) target vs (n_common, n_donors): SLSQP from the uniform start,
    scale-normalized SSE (the phase-3 scale-invariance fix), deterministic."""

def weighted_counterfactual(contrib: np.ndarray, w: np.ndarray,
                            missing_tol: float = MISSING_W_TOL) -> np.ndarray:
    """(n_donors, n_t) -> (n_t,); moved verbatim from did/controls."""
```

`did/controls.synthetic_control` delegates to both helpers; its public behavior, extras keys, and
every phase-3 test stay bitwise-identical (the guard: run the full suite before AND after,
including `test_did_controls.py::test_synthetic_control_scale_invariant_optimization`).

**Tests (`tests/test_simplex.py`):**

- *Exact recovery:* y_target an exact convex combination of 3 donor rows ⇒ recovered weights
  within 1e-3 (L∞) of truth, sse < 1e-10.
- *Scale invariance:* multiplying target and donors by 1e4 leaves weights within 1e-6 (the
  phase-3 regression, now tested at the helper level).
- *Simplex constraints:* weights ≥ −1e-12, sum within 1e-8 of 1 on a random seeded problem.
- *weighted_counterfactual:* NaN donor cell under tolerance ⇒ renormalized value (hand-checked);
  above tolerance ⇒ NaN, never 0.

Commit: `refactor(estimate): shared simplex-weight fitter extracted from did controls`.

## Task 7 — SC synthetic DGP + donor scoring/selection/ATT

**Create:** `src/natex/data/synthetic_sc.py`, `src/natex/iv/donors.py`,
`tests/test_synthetic_sc.py`, `tests/test_donors.py`.

```python
# data/synthetic_sc.py
@dataclass
class SCSyntheticData:
    df: pd.DataFrame                  # long form: unit, time, y (treated unit named "treated")
    units: list[str]
    treated_unit: str
    t0: float
    true_donors: list[str]            # donors carrying the true convex weights
    true_weights: np.ndarray          # aligned with true_donors
    effect: float                     # constant additive post-period effect on the treated unit

def make_sc_synthetic(
    n_units: int = 20, n_pre: int = 15, n_post: int = 10, n_factors: int = 2,
    k_true: int = 3, effect: float = 10.0, noise: float = 0.5,
    rng: np.random.Generator | None = None,
) -> SCSyntheticData: ...
```

Factor model: `y_ut = mu_u + lambda_u @ f_t + noise*eps`; the treated unit's `(mu, lambda)` is an
exact convex combination (random simplex draw) of `k_true` donors' parameters, so the noiseless
treated trajectory is spanned by the donor pool and the SC estimand is well posed.

```python
# iv/donors.py
@dataclass
class DonorScore:
    unit: object
    pre_rmse: float                   # vs treated pre-trajectory (common finite pre times)
    pre_corr: float                   # Pearson over the same times; NaN if < 3 points
    rank: int                         # 1 = best by the active scoring method

@dataclass
class DonorSelectionResult:
    treated_unit: object
    t0: float
    donors: list[object]              # selected pool, score order
    scores: list[DonorScore]          # ALL complete candidates, ranked
    weights: np.ndarray               # simplex weights over `donors`
    y0_hat: np.ndarray                # (n_t,) counterfactual for the treated unit, NaN where undefined
    times: np.ndarray                 # (n_t,) sorted unique times
    pre_rmspe: float                  # +inf if no defined pre time (never 0 on failure)
    post_rmspe: float
    att_post: float                   # mean post-period (y_treated - y0_hat); NaN on failure
    effect_by_time: np.ndarray        # (n_t,) gap, NaN where undefined
    extras: dict = field(default_factory=dict)  # n_candidates, n_dropped_incomplete, converged, ...

def unit_time_matrix(df: pd.DataFrame, unit: str, time: str, outcome: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Y, units, times): unit-by-time mean-aggregated outcome matrix, NaN for empty cells."""

def select_donors(
    Y: np.ndarray, units: np.ndarray, times: np.ndarray,
    treated_unit: object, t0: float,
    n_donors: int | None = None,      # None -> all complete candidates
    scoring: str = "rmse",            # "rmse" | "corr"
) -> DonorSelectionResult: ...

def select_donors_from_dataset(dataset: Dataset, treated_unit: object, t0: float,
                               n_donors: int | None = None, scoring: str = "rmse",
                               ) -> DonorSelectionResult:
    """Adapter: requires spec.unit, spec.time, spec.outcome; delegates to select_donors."""
```

Candidates = units ≠ treated with finite outcomes at EVERY pre-t0 time where the treated unit is
observed (phase-3 balanced-donor rule; dropped count in extras). Scoring, ranking, and
`fit_simplex_weights` all use pre-period columns only. `y0_hat = weighted_counterfactual` over
all times. Failure paths: no complete candidate → NaN everywhere + reason; no post period →
`att_post = NaN`.

**Tests:**

- *DGP fidelity (`test_synthetic_sc.py`):* determinism; treated pre-trajectory within noise of
  the true convex combination (RMSE < 3·noise/√n_pre); rng required.
- *Donor recovery:* `make_sc_synthetic` (pinned seed), `n_donors=8` ⇒ all `true_donors`
  selected; Σ weights on true_donors ≥ 0.7 (calibrate ≥5 seeds).
- *ATT accuracy:* |att_post − effect| ≤ 1.5 (effect=10, noise=0.5; calibrate).
- *Pre-only honesty (mutation tests):* multiplying every post-t0 outcome of donors AND treated
  by 10 leaves `scores`, `donors`, `weights` bitwise unchanged (only y0_hat post, rmspe_post,
  att change).
- *n_donors=None* uses all complete candidates; `n_donors=3` returns exactly 3, the top-ranked.
- *Failure paths:* single-unit panel ⇒ NaN att + reason; treated unobserved pre-t0 ⇒
  `pre_rmspe == inf`, never 0; unknown treated_unit raises ValueError.
- *unit_time_matrix:* duplicate (unit, time) rows mean-aggregate; missing cell is NaN.

Commit: `feat(iv): synthetic-control donor selection with pre-trend scoring and factor-model DGP`.

## Task 8 — In-space RMSPE-ratio placebo inference

**Create:** `tests/test_sc_placebo.py`.
**Modify:** `src/natex/iv/donors.py`.

```python
# iv/donors.py (additions)
@dataclass
class SCPlaceboReport:
    p_value: float                    # (1 + #{placebo ratio >= treated ratio}) / (n_used + 1); NaN if n_used < 5
    ratio_treated: float              # post_rmspe / pre_rmspe
    ratios: np.ndarray                # (n_used,) placebo ratios, sorted desc
    placebo_units: list[object]       # aligned with ratios
    n_skipped: int                    # zero-pre-rmspe or failed placebo fits
    extras: dict = field(default_factory=dict)

def sc_placebo_test(
    Y: np.ndarray, units: np.ndarray, times: np.ndarray,
    treated_unit: object, t0: float,
    n_donors: int | None = None, scoring: str = "rmse",
    exclude_poor_fit: float | None = None,    # drop placebos with pre_rmspe > mult * treated's
) -> SCPlaceboReport: ...
```

Protocol per the pinned convention (same selection rule for placebos, treated unit excluded from
placebo donor pools, +1-rank p — audit item-1/5 lineage). Fewer than 5 usable placebos ⇒
`p_value = NaN` (mirrors the phase-3 `_MIN_USABLE` policy, never a fake 1.0). Deterministic:
no rng anywhere.

**Tests:**

- *Signal:* `make_sc_synthetic(effect=10, noise=0.5, n_units=20)` ⇒ treated has the largest
  ratio, `p_value == 1/20` (exact +1-rank arithmetic with 19 placebos).
- *Null calibration:* effect=0 across 10 seeds ⇒ p_value ≥ 0.05 in ≥ 8 (placebo exchangeability
  sanity; calibrate and pin).
- *+1-rank arithmetic:* hand-built 4-placebo case with known ratios ⇒ exact p (and NaN when a
  5th is unavailable: n_used=4 < 5 ⇒ NaN).
- *exclude_poor_fit:* a deliberately noisy placebo unit is dropped at mult=2 and counted in
  `n_skipped`; p recomputed over the survivors.
- *Treated exclusion:* the treated unit never appears in any placebo's donor pool (assert via
  extras trace on a small panel).

Commit: `feat(iv): Abadie in-space RMSPE-ratio placebo test with +1-rank p-values`.

## Task 9 — CLI: `natex instruments` and `natex donors`

**Create:** `tests/test_cli_iv.py`.
**Modify:** `src/natex/cli.py`.

```
natex instruments CSV --treatment T --pool "z1,z2,..."           # default pool: all numeric
    [--controls "c1,c2"] [--outcome y] [--honest/--no-honest]    # honest default: True
    [--lam plugin|cv|<float>] [--seed 0] [--out out/]
natex donors CSV --outcome y --unit state --time year --treated-unit California --t0 1989
    [--n-donors K] [--scoring rmse|corr] [--placebo/--no-placebo]  # placebo default: True
    [--exclude-poor-fit MULT] [--out out/]
```

Both write `_clean`-ed JSON payloads (`out/instruments.json`, `out/donors.json`) and echo a
short human summary, mirroring the existing `discover`/`debias` style. `instruments` payload:
selection (names, λ, loadings summary, F, partial R², weak), honest split sizes, and when an
outcome is given the estimation block (tau/se/ci, ar_ci/ar_kind, j_stat/j_p, first_stage_F) plus
the honesty caveat string when `--no-honest`. `donors` payload: ranked scores, selected donors,
weights, pre/post RMSPE, att_post, effect_by_time, and the placebo block (p, treated ratio, top
placebo ratios). Exit code 2 on bad option combos (missing --outcome for donors, unknown unit,
`--lam cv` without deterministic seed is fine — seed defaults to 0).

**Tests (`tests/test_cli_iv.py`, CliRunner on tmp_path CSVs):**

- `instruments` on a small `make_iv_synthetic` CSV: exit 0, JSON has `selection.selected`
  non-empty, `estimate.tau` finite, `estimate.ar_kind == "interval"`; `--no-honest` sets the
  caveat field; omitting `--outcome` yields `estimate: null`.
- NaN-cleanliness: the JSON round-trips through `json.loads` (`_clean` handled NaN → None).
- `donors` on a `make_sc_synthetic` CSV: exit 0, `att_post` within the task-7 tolerance,
  `placebo.p_value == 1/n_placebos+1` arithmetic consistent, `--no-placebo` omits the block.
- Bad inputs: unknown treated unit → exit 2 with a message, not a traceback.

Commit: `feat(cli): natex instruments and natex donors commands`.

## Task 10 — Benchmark harness + CI-small slices

**Create:** `benchmarks/run_iv_selection.py`, `tests/test_iv_benchmarks_small.py`.

`run_iv_selection.py` (style: `run_dee_sim.py`): sweeps concentration μ² ∈ {8, 30, 80, 180, 400}
× 20 seeds at (n=500, p=50, s=5); per cell records selection precision/recall (vs `true_support`),
post-Lasso 2SLS |τ̂ − τ|, OLS |τ̂ − τ|, AR-set boundedness rate, honest-pipeline coverage of τ by
ar_ci. Writes `benchmarks/out/iv_selection.csv` (+ optional matplotlib plot behind the `plot`
extra, importorskip pattern). Also an SC block: donor-recovery rate and |ATT − effect| across
noise ∈ {0.25, 0.5, 1.0} × 20 seeds.

**Tests (`tests/test_iv_benchmarks_small.py`, CI-small — seeded 3-seed slices, each < 10 s):**

- Strong regime (μ²=180): mean recall of the top-3 true instruments = 1.0; mean 2SLS |bias| <
  mean OLS |bias| in 3/3 seeds.
- Weak regime (μ²=8): AR-set unbounded/disjoint in ≥ 1/3 seeds while Wald always reports a
  finite CI — the honesty gap the benchmark exists to show (calibrate; pin seeds).
- SC slice (noise=0.5): donor-recovery (all true donors selected at n_donors=8) in 3/3 seeds;
  mean |ATT − effect| < 1.5.
- Harness function importable and returns a DataFrame with the documented columns (no file I/O
  in the test; pass a 2-cell mini-grid).

Commit: `feat(benchmarks): IV selection/strength curves and SC recovery harness with CI-small gates`.

## Task 11 — Prop 99 donor-selection backtest (+ optional Egger IV stretch)

**Create:** `tests/backtests/test_prop99_donors.py` (`pytestmark = pytest.mark.backtest`).

Protocol (fixed before calibration): `load_or_skip("prop99")`, `unit_time_matrix` on
(state, year, packs outcome), treated_unit = California, t0 = 1989, `n_donors=None` (full
complete pool) and `n_donors=8` variants, scoring="rmse", placebo with `exclude_poor_fit=None`.

**Assertions (starting thresholds; calibrate once against the deterministic run, then pin —
same convention as `test_prop99.py`):**

- ADH donor set: with `n_donors=8`, ≥ 3 of {Colorado, Connecticut, Montana, Nevada, Utah} are
  selected by pre-trend rank; with the full pool, the summed simplex weight on that five-state
  set ≥ 0.5 (phase-3 SuDDDS synthetic control found Utah/Montana/Nevada/Connecticut as top
  donors — reconciliation note in the method card if the sets differ).
- Effect: `att_post` negative and within [−35, −5] packs (phase-3 full-post-period finding was
  ≈ −19.5; the thesis Table 6.1 value −8.96 reflects a shorter effective window — same
  reconciliation as phase 3).
- Placebo: California's RMSPE ratio ranks in the top 5 of the usable placebos and
  `p_value ≤ 0.2` (ADH report CA as most extreme, p = 1/39; our variant may differ — pin the
  observed rank as a regression, document any gap).
- Determinism: two runs bitwise-identical (no rng in the donor path).

**Optional stretch (non-blocking, Egger-style):** on `egger_koethenbuerger`, build threshold-
crossing dummies 1[pop ≥ c] for the statutory cutoffs plus decoy dummies at non-statutory
population points; `select_instruments` for council size (controls: smooth population polynomial)
should select statutory dummies ahead of decoys. If it fails, record as a documented finding in
the method card (spec §10 treats Egger as a stress case), not a test failure — implement as
`@pytest.mark.xfail(strict=False)` or a reported-only assertion.

Commit: `test(backtest): Prop 99 donor selection recovers ADH donor pool, negative ATT, placebo rank`.

## Task 12 — Method card, README, status, full gate

**Create:** `docs/method_cards/iv_sc.md`, `docs/status/phase-5.md`.
**Modify:** `README.md` (roadmap tick + quickstart snippets for `natex instruments`/`donors`).

Method card must cover: BCCH plug-in formulas as pinned (λ, γ, c, loadings, iteration, sklearn
mapping); post-Lasso rationale; the F<10 heuristic + Stock–Yogo caveat; AR/Fieller closed form
with the four set kinds and the boundedness⇔first-stage-F identity; Hansen J scope statement
("tests overidentifying restrictions only — exclusion itself is untestable", audit item-3
lineage); honest-split rationale (audit item 1); SC donor protocol (balanced-donor rule,
pre-only scoring, simplex fit, RMSPE-ratio placebo with +1-rank p, exclude_poor_fit convention);
documented deviations (no V-weights; Springer-chapter items out of scope: Deep IV, GANs, text
matching); Prop 99 reconciliation notes from task 11.

Final gate (all must pass before the status doc is written):

1. `uv run pytest -q` — full unit suite green (including phases 1–4).
2. `uv run ruff check src tests` clean.
3. `NATEX_DATA="/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data" uv run pytest -q -m backtest` —
   ALL backtests green (phases 2–3 rows + the new Prop 99 donor test).
4. `docs/status/phase-5.md`: gate record (test counts, backtest runtimes, calibration evidence
   for every pinned threshold), deviations, open questions for phase 6.

Commit: `docs: IV/SC method card, README, phase-5 status`.
