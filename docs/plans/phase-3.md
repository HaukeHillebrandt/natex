# Phase 3 implementation plan — SuDDDS (did/) + Prop 99 backtest

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` §3 (did/, validate/, estimate/),
§4 (core API), §5 items 4–5, §8 (Prop 99 row), §9 (phase 3), §10 (SuDDDS hyperparameter risk).
**Method notes:** `".../RDD/docs/notes/read_thesis-ch6-did.md"` (SuDDDS Eqs 6.1–6.28, Algorithms 6–9,
experiments, Table 6.1), `".../RDD/docs/notes/review_data-inventory.md"`.
**Local data root:** env `NATEX_DATA` (currently
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data"`).

## Phase objective (spec §9 phase 3)

1. **SuDDDS** — first public implementation of thesis ch. 6, with every audit repair:
   heterogeneous RDiT search (Algorithms 6–8) over categorical covariate profiles with the
   double-β, corrected single-Δ, and Bernoulli observation models; control identification
   (standard DD, synthetic control, GESS Algorithm 9); validation battery (dependence-preserving
   panel randomization test, composition/anticipation checks replacing McCrary-in-time,
   per-dimension placebo); effect estimation with dose normalization and a two-sided studentized
   τ̂ randomization test.
2. **Prop 99 backtest** (spec §8 row 6): SuDDDS recovers (California, 1989); τ̂ sign/magnitude in
   line with Table 6.1 (DD −10.94, Synthetic Control −8.96, GESS −6.67, all significant at 5%).
3. **Ch. 6 synthetic benchmarks**: the corrected DGP (Eqs 6.22–6.28), precision/recall/F vs
   discontinuity magnitude for greedy / weighted-convex / single-Δ (Fig 6.1 analog), the control
   benchmark including the heterogeneous-DGP GESS advantage (Figs 6.3/6.5 analog). CI-small
   seeded slices in tests; full curves via `benchmarks/run_did_curves.py`.

## Audit corrections that bind this phase (docs/math_audit_final.md)

| # | Correction | Where implemented |
|---|---|---|
| 1 | Fitted-null Monte Carlo is a parametric bootstrap, not exact; +1-rank p-values; never claim exactness | `validate/panel.py` docstrings + p-value rule |
| 5 | τ̂ placebo test: **two-sided studentized** statistic, +1-rank p, matched subset shapes, stated assumptions; "independence of the two tests" replaced by the precise conditional statement | `did/effects.py::tau_randomization_test` |
| 11 | Algorithm 6 keeps a **global incumbent (s\*, T₀\*, W\*)** across windows/restarts | `did/suddds.py` |
| 12 | Algorithm 7: require **minimum two-sided support** inside the window for every cutoff candidate | `did/suddds.py::optimize_t0` |
| 13 | Algorithm 8 is deletion-only as printed: repair = **relax dimension j, slice over all values, retain the incumbent explicitly** | `did/mdss.py` |
| 14 | Algorithm 9 line 6 is **argmin** (not argmax); initialize control-set MSE = +∞ | `did/controls.py::gess_control` |
| 15 | Single-Δ statistic must **profile μᵢ under H₁** and **restrict to windows**; use C̃ᵢ, B̃ᵢ with the precision-weighted δ̄ correction; **scan both signs of Δ** | `did/statistics.py::single_delta_*` |
| 16 | WCC ρ-priority scan is heuristic, not exact; default at small cardinality = **exhaustive per-dimension enumeration** (2^V−1 subsets), exact for the printed statistic | `did/mdss.py` (`exhaustive_max_values`) |
| 17 | Eq 6.10 prose swaps q₁/q₂ ↔ β_g0/β_g1 (typo; equations right) | method card note only |
| 18 | **Panel replica nulls must preserve unit/time dependence**; McCrary on calendar time is information-free → replace with **composition/anticipation checks** | `validate/panel.py` |
| 19 | Continuous-treatment DD contrast estimates ζ·τ, not τ → **dose normalization**; **model class must match T's type** (Bernoulli variant for binary treatment) | `did/effects.py`, `did/statistics.py::bernoulli_window_llr*` |
| 24 | No absolute variance floor: **data-scaled shrinkage** for σ̂ᵢ² | `did/background.py` |
| typos | Eq 6.18/6.20 denominators count summed records (+ s→s_τ subscript); Eq 6.19 **unit-level** weights; Eq 6.21 ill-typed; Eqs 6.24–6.25 dimensionally invalid as printed; Eq 6.27 creates heteroskedasticity, not correlation; Eq 6.28 set notation; Eq 6.4 rank deficiency → reference/pinv coding | `did/controls.py`, `data/synthetic_did.py`, method card |
| §3 adopted | corrected single-Δ profile GLR as the fast path; exhaustive per-dimension enumeration as the exact MDSS baseline; staggered adoption via a pluggable group-time backend **interface only** (spec non-goal to implement) | `did/mdss.py`, `did/effects.py::DiDEstimatorBackend` |

Also inherited from the audit's general items: sufficient stats use σ² weights (cᵢ = rᵢ/σᵢ²,
bᵢ = 1/σᵢ² — the thesis page prints σ², adjudicated in §1 of the audit); Bernoulli null draws are
direct Bernoulli(p̂) (item 2); NaN never 0.0 on failure; discovery never reads `y`.

## House rules (bind every task)

Python ≥3.11; core deps only numpy/scipy/pandas/scikit-learn/typer/pydantic — everything else
optional extras with tests that skip gracefully; CI (3.11–3.14) must stay green; **no new
dependencies are needed for this phase** (synthetic-control weights use `scipy.optimize`).
One `numpy.random.Generator` through every stochastic call (same seed ⇒ same result); discovery
never reads the outcome `y`; NaN never 0.0 on failure; no bare `except`; never commit datasets;
conventional commit after every green cycle. `uv run pytest -q` excludes backtests; run backtests
with `-m backtest` and `NATEX_DATA` set.

**TDD discipline for every task:** write the failing test(s) first, run them to confirm failure,
implement, run the full suite (`uv run pytest -q` and `uv run ruff check src tests`), then commit.

**Statistical-test policy:** every stochastic assertion is seeded; calibrate thresholds across ≥5
seeds during implementation, then pin one seed with a margin. Initial thresholds below are
starting points — tighten or loosen only with a code comment stating the calibration evidence.

## Current interfaces built upon (do not break)

- `natex.data.spec.DatasetSpec(treatment, outcome=None, forcing, covariates, time=None)`;
  `Dataset(df, spec)` (drops NaN rows in scan columns only, never `y`), `Dataset.from_csv(...)`,
  properties `n, T, y, Z, Z_std, X, treatment_is_binary`.
- `natex.data.registry`: `DatasetInfo`, `REGISTRY`, `data_root`, `locate`, `verify`,
  `load_dataset(name, root=None, outcome="default")`, `_prepare(name, df)` hook.
- `natex.scan.statistics`: `normal_llr_all_splits(r, w, G)`, `fit_log_odds_offset(t, eta)`,
  `offset_log_lik(theta, t, eta)`, `fit_log_odds_offsets(t, eta, M)`,
  `masked_offset_log_lik(theta, t, eta, M)` — the masked Bernoulli machinery is reused verbatim
  for the DiD Bernoulli model.
- `natex.validate.placebo.hc1_ols(Xmat, yvec) -> (beta, se)`.
- `natex.validate.randomization.RandomizationReport` (naming/shape convention to mirror).
- `natex.rdd.lord3.fit_treatment_model` (ridge-logistic separation-guard rationale to mirror).
- `tests/backtests/conftest.py::load_or_skip` fixture.
- CLI `natex datasets`, `natex discover` (typer app in `natex/cli.py`).

All signature changes are **additive** (new optional fields/kwargs with defaults) so phase-1/2
tests keep passing unmodified.

## Design conventions fixed for the whole phase

- **Window convention:** for a cutoff T₀ and half-width W, the pre-window is
  `T0 - W <= t < T0` (g₀) and the post-window is `T0 <= t < T0 + W` (g₁); records outside
  contribute nothing to any DiD scan statistic (audit item 15's window restriction). `t == T0`
  is post — treatment starts at T₀. Documented in every kernel docstring.
- **Subsets** are conjunctions over dimensions of unions over values:
  `s = {i : code[i, j] ∈ V_j for every dim j}`, represented as per-dimension boolean masks over
  value codes. The all-True state is `s = D`.
- **Determinism:** the caller-supplied Generator drives (in order) restart initialization, MDSS
  dimension shuffles, WCC ρ draws, and validation replicas. Identical seed ⇒ identical result.
- **Estimation vs scan windows:** the RDiT scan is window-restricted; effect estimation
  (controls, τ̂) uses the full pre/post periods split at T₀ (Eqs 6.17–6.20 use all pre records).
  Documented in `did/effects.py`.

---

## Task 1 — Commit this plan; panel data layer

**First action of this task:
`git add docs/plans/phase-3.md && git commit -m "docs: phase 3 implementation plan"` — the plan
file is committed before any code.**

**Create:** `src/natex/did/__init__.py`, `src/natex/did/panel.py`, `tests/test_panel.py`.
**Modify:** `src/natex/data/spec.py`.

`DatasetSpec` gains one additive field, and `Dataset.__init__` gains validation:

```python
class DatasetSpec(BaseModel):
    treatment: str
    outcome: str | None = None
    forcing: list[str]              # may be [] (DiD-only datasets have no forcing variable)
    covariates: list[str]
    time: str | None = None
    unit: str | None = None         # NEW: cross-sectional unit id column (e.g. "state")
```

`Dataset.__init__`: if `spec.time` is set, the column must exist and be numeric; if `spec.unit`
is set, the column must exist (any dtype). Both are appended to the scan-column NaN-drop list
(never the outcome). `Dataset.from_csv` gains `unit: str | None = None` passthrough. Verify (and
test) that `forcing=[]` works end to end: `Z` has shape `(n, 0)`, construction does not raise.

```python
# src/natex/did/panel.py
@dataclass
class CategoricalPanel:
    codes: np.ndarray            # (n, m) int64 value codes, 0..k_j-1 per dimension
    dim_names: list[str]         # length m
    dim_values: list[np.ndarray] # per-dim decoded original values (index = code)
    t: np.ndarray                # (n,) float time
    theta: np.ndarray            # (n,) float treatment
    y: np.ndarray | None         # (n,) outcome or None; the SCAN NEVER READS IT
    unit: np.ndarray             # (n,) int unit codes
    unit_values: np.ndarray      # decoded unit labels (index = code)
    @property
    def n(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def dim_sizes(self) -> tuple[int, ...]: ...   # k_j per dimension
    @property
    def profile_id(self) -> np.ndarray: ...       # (n,) int, lexicographic code over dims (cached)
    def subset_mask(self, included: list[np.ndarray]) -> np.ndarray:
        """(n,) bool from per-dim value masks (conjunction of unions)."""

def quantile_bins(x: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Codes 0..b-1 from quantile edges; returns (codes, edges). Ties collapse
    duplicate edges (fewer effective bins is fine); x must be finite (Dataset
    already dropped NaN scan rows)."""

def build_panel(dataset: Dataset, dims: list[str] | None = None, bins: int = 4) -> CategoricalPanel:
    """Requires dataset.spec.time. dims default: all covariates (minus the time
    and unit columns). Non-numeric dims and numeric dims with <= bins distinct
    values are coded by their sorted unique values; other numeric dims get
    quantile_bins(bins). unit defaults to spec.unit, else profile_id (documented:
    dependence-preserving nulls then treat each profile as one unit)."""
```

**Tests (`tests/test_panel.py`):**
- `quantile_bins`: hand-built 8-point array → known codes; constant column → single bin;
  monotone in x (codes nondecreasing when x sorted).
- `build_panel` on a toy DataFrame (2 categorical dims, 1 continuous dim, unit, time):
  `codes` shape/dtype, `dim_sizes` correct, binary 0/1 column NOT quantile-split,
  `profile_id` equal for equal rows and distinct otherwise, `subset_mask` matches a pandas
  reimplementation on 3 hand-written subsets.
- `DatasetSpec(unit=...)` validation: missing unit column raises `ValueError`; NaN unit rows
  dropped; `forcing=[]` Dataset constructs and `Z.shape == (n, 0)`.
- Missing `time` → `build_panel` raises `ValueError` naming the field.

Commit: `feat(did): panel data layer — categorical panel, quantile binning, unit column`.

---

## Task 2 — DiD background model (treatment on covariates + time)

**Create:** `src/natex/did/background.py`, `tests/test_did_background.py`.

```python
@dataclass
class DiDBackground:
    kind: str                    # "normal" | "bernoulli"
    fitted: np.ndarray           # f(x, t) (normal) or p_hat (bernoulli), (n,)
    r: np.ndarray | None         # theta - fitted (normal only)
    sigma2: np.ndarray | None    # per-record variance (normal only)
    eta: np.ndarray | None       # logit(clip(p_hat)) (bernoulli only)

def fit_did_background(
    panel: CategoricalPanel,
    model: str = "auto",         # "auto" -> "bernoulli" iff theta is binary {0,1}, else "normal"
    degree: int = 1,             # polynomial order R in t
    unit_effects: bool = True,   # per-unit intercepts (thesis 6.4.4 uses state FE)
    shrink: float | None = None, # None -> count-based shrinkage weight (see below)
) -> DiDBackground
```

Design matrix: per-unit one-hot intercepts (no global intercept — Eq 6.4 rank-deficiency typo:
use pinv/lstsq, document) plus `t, t², …, t^degree` columns on time standardized to zero
mean/unit sd (conditioning). Normal path: lstsq fit, residuals `r`; per-**profile** residual
variance with **data-scaled shrinkage** (audit item 24 — no absolute floor):
`sigma2_prof = (1 - lam) * s2_prof + lam * s2_global`, `lam = n0 / (n0 + n_prof)` with
`n0 = 10` by default (`shrink` overrides `lam` directly when given); final floor
`max(sigma2, 1e-12 * s2_global)` — *scaled*, never absolute. Bernoulli path: ridge-penalized
logistic (`C=1.0`, standardized features) mirroring `fit_treatment_model`'s separation-guard
rationale; `eta = logit(clip(p_hat, 1e-6, 1 - 1e-6))`.

**Tests:**
- Seeded panel with planted per-unit intercepts + linear trend, no jump: `r` has per-unit means
  ≈ 0 (|mean| < 0.05) and no time trend (|OLS slope of r on t| < 0.05).
- Shrinkage: a profile with 2 records does NOT get its raw 2-point variance (assert
  `sigma2` strictly between raw profile variance and global variance); zero-variance profile →
  sigma2 ≥ scaled floor > 0.
- `model="auto"` dispatch: binary θ → "bernoulli" with `eta` finite; continuous θ → "normal".
- Determinism: two calls, same inputs → identical arrays.

Commit: `feat(did): treatment background model with unit effects and data-scaled variance shrinkage`.

---

## Task 3 — DiD scan statistics: double-β, corrected single-Δ, Bernoulli window LLR

**Create:** `src/natex/did/statistics.py`, `tests/test_did_statistics.py`.

All kernels are **window-restricted** (records outside `[T0-W, T0+W)` contribute nothing) and
evaluate **many candidate subsets at once** (masks matrix `M`, shape `(n, S)` boolean), mirroring
the vectorized phase-2 kernels.

```python
@dataclass
class WindowStats:
    """Per-record sufficient statistics for a fixed (T0, W). Records outside the
    window have c = b = 0 and g1 = False, so subset sums automatically restrict."""
    in_window: np.ndarray   # (n,) bool
    g1: np.ndarray          # (n,) bool — post side, t in [T0, T0+W)
    c: np.ndarray           # (n,) w * r, zeroed outside window
    b: np.ndarray           # (n,) w = 1/sigma2, zeroed outside window

def window_stats(t, r, sigma2, T0: float, W: float) -> WindowStats

def double_beta_llr_masks(ws: WindowStats, M: np.ndarray) -> np.ndarray:
    """Eq 6.9 per subset column: C1²/2B1 + C0²/2B0 − (C1+C0)²/(2(B1+B0)), with
    C1 = M^T (c·g1) etc. Degenerate columns (zero precision mass on either side)
    score exactly 0.0, detected from in-window COUNTS per side (not float residue
    — same guard as normal_llr_all_splits). Always >= 0."""

def double_beta_q(ws: WindowStats, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(q1, q2) per column: post/pre precision-weighted residual means (Eq 6.10).
    NaN (never 0) for empty sides."""

def single_delta_stats(ws: WindowStats, profile_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Audit item 15 corrected sufficient stats per PROFILE (arrays indexed by
    profile id): with d_j = +1 (post) / -1 (pre) over in-window records,
       delta_bar_i = sum(w d)/sum(w),
       B_tilde_i = sum(w) - (sum(w d))² / sum(w)      [= sum w (d - delta_bar)²],
       C_tilde_i = sum(w (d - delta_bar_i) r).
    mu_i is profiled out under BOTH hypotheses from in-window records. Profiles
    with < 2 in-window records or with one empty side have B_tilde = C_tilde = 0
    (they cannot identify Delta)."""

def single_delta_llr(C_sel: float | np.ndarray, B_sel: float | np.ndarray) -> float | np.ndarray:
    """Profile GLR: C²/(2B), Delta_hat = C/B; 0.0 when B <= 0. Sign-agnostic in
    Delta (both signs enter through the priority ordering, audit item 15)."""

def bernoulli_window_llr_masks(theta, eta, ws: WindowStats, M: np.ndarray) -> np.ndarray:
    """Audit item 19 model-matching variant for binary theta. Per subset column s:
    H0 one common log-odds offset over s ∩ window; H1 separate pre/post offsets.
    Reuses natex.scan.statistics.fit_log_odds_offsets / masked_offset_log_lik with
    the three mask stacks (s∧window, s∧g0, s∧g1); pure sides hit the boundary
    suprema (0.0), degenerate columns score exactly 0.0. Always >= 0."""

def working_residuals(theta, p_hat) -> tuple[np.ndarray, np.ndarray]:
    """(r, sigma2) = (theta - p_hat, p_hat(1-p_hat)) — used ONLY to order
    priorities for the Bernoulli model; LLR evaluation stays exact-Bernoulli."""
```

**Tests:**
- **Hand-computed double-β**: 6-record window, unequal σ², one subset — assert Eq 6.9 value to
  1e-12 against a scalar hand calculation; identity check vs the algebraically equal harmonic
  form `HM(B1,B0)·((q1−q0)/2)²`.
- **Window restriction property**: adding records outside the window (huge residuals) changes
  neither double-β nor single-Δ nor Bernoulli values (exact equality).
- **Single-Δ correctness vs brute force** (the audit-15 regression test): on a seeded 3-profile,
  12-record window, maximize the exact Gaussian log-likelihood ratio numerically over
  (Δ, μ₁, μ₂, μ₃) with `scipy.optimize.minimize`; assert `single_delta_llr` matches to 1e-6.
  Assert the UNprofiled (thesis-printed) statistic — μᵢ frozen at the H₀ MLE — is strictly
  smaller on a case with unbalanced pre/post precision (this is the audit's 4.74-vs-5.33 class
  of counterexample, constructed locally).
- **Both signs**: negate all residuals → identical single-Δ LLR, `Delta_hat` negated.
- **Bernoulli parity**: on ≤ 12-record windows, `bernoulli_window_llr_masks` matches a scalar
  reference built from `fit_log_odds_offset`/`offset_log_lik` per subset (atol 1e-8), including
  a pure-post-side column (boundary supremum, finite, ≥ 0) and a degenerate column (exactly 0.0).
- **Properties (all three kernels):** LLR ≥ 0 everywhere; permutation invariance (shuffling
  record order with the masks permuted alike leaves values unchanged); empty-subset columns 0.0.

Commit: `feat(did): window-restricted LLR kernels — double-beta, profiled single-delta, Bernoulli`.

---

## Task 4 — MDSS: repaired Algorithm 8

**Create:** `src/natex/did/mdss.py`, `tests/test_mdss.py`.

```python
@dataclass
class SubsetState:
    included: list[np.ndarray]     # per-dim bool masks over value codes; all-True = D
    def mask(self, panel: CategoricalPanel) -> np.ndarray
    def values(self, panel: CategoricalPanel) -> dict[str, list]   # decoded, for reports

Evaluator = Callable[[np.ndarray], np.ndarray]   # (n, S) bool masks -> (S,) LLR

def mdss_optimize(
    panel: CategoricalPanel,
    evaluator: Evaluator,          # closure over WindowStats for a fixed (T0, W) + model
    priority: str,                 # "greedy" | "wcc" | "single_delta"
    priority_stats: object,        # WindowStats (+ profile stats for single_delta)
    rng: np.random.Generator,
    init: SubsetState | None = None,       # None -> s = D
    n_rho: int = 10,
    exhaustive_max_values: int = 12,
    max_sweeps: int = 25,
    tol: float = 1e-12,
) -> tuple[SubsetState, float]
```

Each sweep shuffles dimension order via `rng` (Alg 8 line 3). Per-dimension step — **audit item
13 repair**: build the *relaxed* state (dimension j unconstrained, all other dims as-is), then:

- **Exact branch** (`k_j <= exhaustive_max_values`, the audit-16 default): enumerate all
  `2^k_j - 1` nonempty value subsets of dimension j, evaluate their LLRs in one vectorized
  `evaluator` call, take the best.
- **Priority branch** (large `k_j`): per-value priorities on `relaxed ∧ (x_j = v_k)`:
  - `"greedy"`: start from the best single value; iteratively add the value maximizing the
    combined (q₁ − q₂) of the union (thesis's greedy remedy); evaluate the LLR at every prefix;
    candidate = argmax-LLR prefix.
  - `"wcc"`: draw `n_rho` ρ ~ U(0,1) from `rng`; for each, order values by
    ρ·q₁^(k) + (1−ρ)·(−q₂^(k)); evaluate all prefixes for every ρ; best overall.
  - `"single_delta"`: order values by γ_k = C̃_k/B̃_k **descending and ascending** (both signs of
    Δ, audit item 15); evaluate prefixes of both orders; best overall.
- **Retain the incumbent explicitly**: accept the dimension-j candidate only if its LLR exceeds
  the current state's by > `tol`; otherwise keep the current constraint for dimension j.

Converged when a full sweep over all dims yields no accepted update. LLR is weakly increasing by
construction; return `(state, llr)`.

**Tests:**
- **Monotone trace**: instrumented run records LLR after every dimension step — nondecreasing.
- **Exact-branch parity**: tiny panel (m=2 dims, V=3, ~40 records, planted jump), enumerate ALL
  `(2³−1)²` conjunctive subsets by brute force; `mdss_optimize` from `init=D` with the exact
  branch reaches the global optimum LLR (equality to 1e-10) for both the double-β and single-Δ
  evaluators.
- **Planted recovery**: seeded panel (m=3, V=4, 10 periods, n≈1200), jump ζ=8 on 2 values of 1
  dim: recovered mask F-score vs truth ≥ 0.9 for `"single_delta"` and `"wcc"` (calibrate seeds).
- **Incumbent retention regression**: construct a state where every single-dimension move
  lowers the LLR; one sweep leaves the state unchanged (the printed deletion-only Alg 8 would
  have moved).
- **Determinism**: same seed → identical `(state, llr)`; different dim-shuffle seeds may differ
  (no assertion beyond both ≥ init LLR).

Commit: `feat(did): repaired multidimensional subset scan (relax-dim, incumbent, exact small-cardinality)`.

---

## Task 5 — SuDDDS driver: repaired Algorithms 6 and 7

**Create:** `src/natex/did/suddds.py`, `tests/test_suddds.py`.
**Modify:** `src/natex/did/__init__.py` (re-exports).

```python
@dataclass
class DiDDiscovery:
    subset_values: dict[str, list]    # dim -> included decoded values (omit unconstrained dims)
    mask: np.ndarray                  # (n,) record membership of s_tau
    t0: float
    window: float
    llr: float
    model: str                        # "normal" | "bernoulli"
    method: str                       # "greedy" | "wcc" | "single_delta"
    p_value: float | None = None
    extras: dict = field(default_factory=dict)   # e.g. delta_hat, q1, q2, restart index

@dataclass
class SuDDDSResult:
    discoveries: list[DiDDiscovery]   # deduped, sorted by llr desc; [0] is the global incumbent
    model: str
    method: str
    windows: tuple[float, ...]
    restarts: int
    def top(self, m: int) -> list[DiDDiscovery]

def default_windows(t: np.ndarray) -> tuple[float, ...]:
    """Data-driven W grid (thesis never reports one — spec §10 risk): with span =
    t.max()-t.min() and step = median diff of unique times, the grid is
    (span/8, span/4, span/2) snapped up to multiples of step, deduped, each >= 2*step."""

def optimize_t0(
    make_evaluator,                   # (T0) -> (Evaluator, WindowStats) for the current model
    t: np.ndarray, mask: np.ndarray, W: float, min_side: int = 3,
) -> tuple[float, float] | None:
    """Algorithm 7 + audit item 12: candidates = unique t within the current
    subset; REQUIRE >= min_side in-subset records on EACH side inside the window;
    return (T0, llr) argmax or None when no candidate qualifies (caller keeps
    the incumbent — never a silent empty-side score)."""

def suddds_scan(
    dataset: Dataset,
    windows: tuple[float, ...] | None = None,   # None -> default_windows
    restarts: int = 8,
    model: str = "auto",              # audit item 19: "bernoulli" iff binary theta; "normal" forceable (thesis parity)
    method: str = "single_delta",     # "greedy" | "wcc" | "single_delta"
    bins: int = 4,
    degree: int = 1,
    dims: list[str] | None = None,
    rng: np.random.Generator | None = None,     # required, raises without
    min_side: int = 3,
    n_rho: int = 10,
    exhaustive_max_values: int = 12,
    panel: CategoricalPanel | None = None,      # precomputed (validation reuses)
    background: DiDBackground | None = None,    # precomputed (replicas refit their own)
) -> SuDDDSResult
```

Algorithm 6 with the audit repairs: outer loop over `windows`; per window, `restarts` restarts —
restart 0 initializes `s = D` (thesis practice), later restarts draw each dimension's value mask
i.i.d. Bernoulli(1/2) from `rng`, redrawing until the record mask is nonempty. Each restart
alternates `optimize_t0` (Alg 7) and `mdss_optimize` (Alg 8) until the LLR stops improving
(> 1e-12) or `max_alternations = 20`. A **global incumbent (s\*, T₀\*, W\*, llr\*) is kept across
all windows and restarts** (audit item 11), and every restart's converged local optimum is
recorded; `discoveries` dedups identical `(frozen mask bytes, t0)` pairs keeping max LLR.
`method="single_delta"` requires `model="normal"` (raise `ValueError` otherwise — the profile
GLR is a Gaussian statistic); Bernoulli model orders priorities via `working_residuals`
(documented heuristic; evaluation stays exact-Bernoulli).

**The scan never touches `y`**: `suddds_scan` must not read `panel.y` anywhere.

**Tests:**
- **y-blindness**: identical dataset with (a) true outcome, (b) outcome column of garbage
  (e.g. all 1e12), (c) `outcome=None` → bitwise-identical `SuDDDSResult` (compare llr, t0,
  masks).
- **Planted synthetic recovery** (m=3, V=4, T=10 periods, n≈1500, ζ=8): top discovery's
  `t0` equals the true T₀ exactly and mask F-score ≥ 0.9, for each method in
  ("greedy", "wcc", "single_delta") (seeds calibrated).
- **Global incumbent regression** (audit 11): scan with `windows=(2.0, W_true)` where the small
  window produces a weaker optimum — `discoveries[0].llr` equals the max over ALL recorded local
  optima (an implementation that returns the last window's best would fail).
- **Min-side support** (audit 12): panel whose subset has all records on one side of every
  candidate cutoff within W → `optimize_t0` returns None and the scan still terminates with the
  incumbent (no exception, no empty-side score).
- **Determinism**: same seed → identical result; `rng=None` raises `ValueError`.
- **Auto model dispatch**: binary θ → `model == "bernoulli"`; continuous → "normal";
  `model="normal"` on binary θ allowed (thesis parity path).
  `method="single_delta"` + `model="bernoulli"` raises.

Commit: `feat(did): SuDDDS alternating scan with global incumbent and min-support cutoffs`.

---

## Task 6 — Ch. 6 synthetic DGP, subset metrics, discovery benchmarks

**Create:** `src/natex/data/synthetic_did.py`, `src/natex/did/metrics.py`,
`tests/test_synthetic_did.py`, `tests/test_did_benchmarks_small.py` (discovery half),
`benchmarks/run_did_curves.py`.

```python
# src/natex/data/synthetic_did.py
@dataclass
class DiDTruth:
    included: list[np.ndarray]   # per-dim value masks of s_I
    record_mask: np.ndarray      # (n,) treated-subset membership
    t0: float
    zeta: float
    tau: float

def make_did_synthetic(
    n: int = 2000,
    d: int = 4, V: int = 8, periods: int = 10,
    zeta: float = 10.0, tau: float = 10.0,
    s_dims: int = 2, s_values: int = 2,        # intervention complexity (thesis base: 2x2)
    theta_kind: str = "real",                  # "real" (Eq 6.25) | "binary" (thresholded latent, documented addition)
    hetero_group: bool = False,                # Eqs 6.27-6.28 variant for the GESS experiment
    rng: np.random.Generator | None = None,    # required
) -> tuple[Dataset, DiDTruth]
```

Implements Eqs 6.22–6.26 with the audit's dimensional repairs (documented in the module
docstring): x_{ij}, u_i ~ DiscreteUniform(1..V); t_i ~ DiscreteUniform over `periods` integer
times; ε ~ N(0, mean_j x_{ij}) (Eq 6.23, the second parameter is a variance);
θ_i = Σ_j γ_{θ,j,x_ij} + ζ·1[x_i ∈ s_I]·1[t_i ≥ T₀] + ε_θ + u  (per-value coefficients — the
printed x·γ is dimensionally invalid, audit typo #23); y_i = Σ_j γ_{y,j,x_ij} + τ·θ_i + ε_y + u.
`hetero_group=True` applies the Eq 6.27 time-scaled noise on s_g = s_I ∪ (random untreated
subset), with the audit note that this is heteroskedasticity, not correlation — it still creates
the misspecification GESS handles. T₀ is drawn from the middle half of the time range; s_I picks
`s_values` random values in each of `s_dims` random dimensions. Returned `Dataset` has
`time="t"`, `unit=None`, `covariates=[x0..x{d-1}]`, `forcing=[]`.

```python
# src/natex/did/metrics.py
def subset_precision_recall(pred_mask, true_mask) -> tuple[float, float, float]  # (P, R, F); NaN (never 0/0 -> 0) when pred empty
```

**Tests (`tests/test_synthetic_did.py`):** truth mask matches a pandas recomputation from
`included`; θ jump: mean θ of treated records post − pre ≈ ζ + (time-trend-free DGP ⇒ tolerance
±1.5 at ζ=10, n=4000, seeded); determinism; `rng=None` raises; binary variant θ ∈ {0,1}.

**CI-small discovery benchmark (`tests/test_did_benchmarks_small.py`):** config d=3, V=4,
periods=10, n=1500, s = 2 values × 2 dims, averaged over 3 pinned seeds, runtime target < 20 s
total:
- At ζ=8: mean F ≥ 0.6 for `"single_delta"` and `"wcc"`; `"greedy"` mean recall ≤ `"wcc"` mean
  recall + 0.05 **with `exhaustive_max_values=0`** (forcing the heuristic priority paths — with
  the exact branch active at V=4 all double-β methods coincide, which is itself asserted:
  exact-branch greedy LLR == exact-branch wcc LLR on one seed).
- At ζ=0 (same pipeline): top-discovery LLR strictly below every ζ=8 top LLR from the same seeds.
- Fig 6.2 spot-check: complexity s_dims=3 at ζ=10 still reaches mean F ≥ 0.5 (single_delta).

**`benchmarks/run_did_curves.py`:** full Fig 6.1/6.2 analogs (ζ grid 0..20, complexity 1..5,
3 methods, ≥ 20 seeds), CSV + optional matplotlib PNG (plot extra; script degrades to CSV-only
when matplotlib is missing) into `benchmarks/out/`; `--small` flag for a quick pass. Follows
`run_nig_curve.py` conventions.

Commit: `feat(did): ch.6 synthetic DGP (corrected), subset metrics, discovery benchmarks`.

---

## Task 7 — Validation battery: panel randomization, composition, anticipation

**Create:** `src/natex/validate/panel.py`, `tests/test_did_validation.py`.

```python
@dataclass
class PanelRandomizationReport:
    p_value: float
    observed_max_llr: float
    null_max_llrs: np.ndarray
    q: int
    null_kind: str          # "ar1_unit" | "iid" | "bernoulli"

def panel_randomization_test(
    dataset: Dataset,
    scan_result: SuDDDSResult,
    Q: int = 99,
    rng: np.random.Generator | None = None,     # required
    scan_kwargs: dict | None = None,            # forwarded to suddds_scan (windows, method, ...)
    null: str = "auto",                         # "auto": "bernoulli" for bernoulli model, else "ar1_unit"
) -> PanelRandomizationReport
```

Fitted-null Monte Carlo, +1-rank p (audit item 1 framing in the docstring: parametric bootstrap,
NOT exact). Null replica draws (audit item 18 — preserve unit/time dependence):

- `"ar1_unit"` (normal model default): fit the background once on the observed data; estimate
  from its residuals (i) the between-unit variance of unit-mean residuals σ̂²_α, (ii) the pooled
  lag-1 autocorrelation φ̂ of within-unit demeaned residuals (per-unit time-sorted; clip φ̂ to
  [0, 0.95]), (iii) the innovation variance matching the pooled residual variance. Replica:
  r\*_ut = α_u + AR(1)_φ̂ innovations (unit effect α_u ~ N(0, σ̂²_α)); θ\* = fitted + r\*.
  Smooth by construction — no jump under H₀ — while preserving within-unit serial dependence
  and between-unit level dispersion.
- `"iid"`: θ\* = fitted + √σ̂² · N(0,1) (phase-2-style; offered for comparison, documented as
  dependence-breaking).
- `"bernoulli"`: θ\* ~ Bernoulli(p̂) i.i.d. (audit item 2 — direct draws). Documented caveat in
  the docstring + method card: serial dependence of a binary panel treatment is NOT preserved;
  the normal-model ar1_unit test on the same data is the dependence-preserving cross-check.

Every replica **refits its own background** (audit item 1 note) and reruns `suddds_scan` with
the same `windows/method/restarts` and the shared `rng`. Unit-level draws require ≥ 2 units.

```python
@dataclass
class CompositionReport:
    p_value: float
    statistic: float
    table: np.ndarray       # (profiles-or-units x 2) pre/post in-window counts
    passed: bool

def composition_test(panel, discovery, alpha: float = 0.05, by: str = "unit") -> CompositionReport
    # chi-square independence test (scipy.stats.chi2_contingency) of record composition
    # pre vs post inside the discovery window — the audit-18 replacement for the
    # information-free McCrary on calendar time. Rows with all-zero counts dropped;
    # < 2 usable rows or a degenerate table -> p_value = NaN, passed = False (never silently ok).

@dataclass
class AnticipationReport:
    shifts: tuple[int, ...]
    estimates: np.ndarray   # placebo jump estimates at T0 - shift*step (pre-data only)
    p_values: np.ndarray    # two-sided z from the model's analytic variance
    p_holm: np.ndarray
    passed: bool

def anticipation_test(panel, background, discovery, shifts=(1, 2, 3), alpha=0.05) -> AnticipationReport
    # For each shift: placebo cutoff T0' = T0 - shift*step (step = median unique-time diff),
    # RESTRICTED to records with t < T0 (never contaminated by the real jump).
    # Normal/single-delta: Delta_hat with Var = 1/B_tilde; double-beta: q1-q2 with
    # Var = 1/B1 + 1/B0; two-sided normal p, Holm across shifts. Shift with
    # insufficient two-sided support -> NaN p, excluded from Holm, noted in report.
```

**Tests:**
- **Power**: planted ζ=8 synthetic (task 6 config), Q=19, pinned seed → p ≤ 0.10; ζ=0 → p ≥ 0.2
  (seeds calibrated across ≥5 candidates; document the calibration in a comment).
- **Dependence regression**: an AR(1) φ=0.8 null panel with NO jump — `"ar1_unit"` p ≥ 0.1 while
  `"iid"` yields a smaller p on the same seed (demonstrates why audit 18 matters; assert
  `p_ar1 >= p_iid` rather than a fixed gap).
- **+1-rank rule**: with Q=4 and observed strictly above all nulls, p == 1/5 exactly.
- **Determinism**: identical seed ⇒ identical `null_max_llrs`.
- **Composition**: balanced panel (every unit at every time) → p ≈ 1, passed; panel where half
  the units stop reporting post-T₀ → p ≤ 0.01, failed.
- **Anticipation**: smooth pre-period → all Holm p > 0.05, passed; planted pre-jump at T₀−2 →
  failed.
- `rng=None` raises; `Q < 1` raises.

Commit: `feat(validate): dependence-preserving panel randomization, composition and anticipation checks`.

---

## Task 8 — Control identification: standard DD, synthetic control, GESS

**Create:** `src/natex/did/controls.py`, `tests/test_did_controls.py`.

Estimation uses full pre/post split at `discovery.t0` (not the scan window); all methods share:

```python
@dataclass
class ControlResult:
    method: str                     # "dd" | "synthetic" | "gess"
    y0_hat: np.ndarray              # counterfactual y_hat(0) for every s_tau record (n_tau,), NaN where undefined
    pre_mse: float                  # Eq 6.17 over defined pre-period records; +inf if none defined; NEVER 0 on failure
    control_mask: np.ndarray | None # (n,) records forming the control set (dd, gess)
    weights: np.ndarray | None      # per-UNIT simplex weights (synthetic; audit typo: Eq 6.19 weights are unit-level)
    alpha: float | None             # fixed-effect offset (dd, gess)
    extras: dict

def dd_control(panel: CategoricalPanel, discovery: DiDDiscovery) -> ControlResult
def synthetic_control(panel: CategoricalPanel, discovery: DiDDiscovery) -> ControlResult
def gess_control(panel, discovery, full_dimension: bool = False) -> ControlResult
```

- **`dd_control`** (Eq 6.18 with the audit typo repairs): the per-time counterfactual is the
  mean of control records **actually present at that time** (denominator = count of summed
  records, not |D\s_τ|), plus α = (mean of s_τ pre records) − (mean of control pre records),
  each over actual record counts (works on unbalanced panels). Times with zero control records
  → NaN counterfactual (never 0), reported via `extras["n_undefined_times"]`.
- **`synthetic_control`**: unit-level weights w ≥ 0, Σw = 1 minimizing
  ‖ȳ_pre,τ(t) − Σ_u w_u ȳ_pre,u(t)‖² over pre-period times common to all candidate control
  units, solved with `scipy.optimize.minimize(method="SLSQP")` from the uniform start
  (deterministic; document that Abadie covariate V-weights are NOT implemented — outcome-only
  pre-fit, a documented deviation). Control units = units with no s_τ records. `y0_hat` = Σ w·ȳ_u(t)
  (no α offset — weights absorb levels).
- **`gess_control`** (Algorithm 9, audit item 14): start `s_sup = s_τ`, incumbent
  `mse = +∞` replaced by `mse(s_τ)`'s control … concretely: initialize best_mse = +∞; first
  evaluation is of s_sup = s_τ (whose control is empty → mse = +∞, so any expansion that yields
  a nonempty control wins); each step evaluates every candidate expansion — one covariate VALUE
  (default) or one whole DIMENSION (`full_dimension=True`, Eq 6.21) added to the current
  profile — computes `mse(s_sup ∪ candidate)` per Eqs 6.20/6.17 (count-corrected denominators, α
  offset as in dd_control), and takes the **argmin** (audit item 14 — the printed argmax is the
  regression-tested bug); stop when no candidate strictly lowers the MSE. Returns
  s_c = s_sup \ s_τ. Expansion candidates come only from dimensions constrained in s_τ's profile
  plus (for values) unconstrained value additions along constrained dims — i.e., monotone
  profile expansions, per the thesis.

**Tests:**
- `dd_control` hand-check: 3-unit, 4-period toy panel with known means → exact `alpha`,
  `y0_hat`, `pre_mse` values (1e-12).
- Unbalanced panel: drop one control record; denominators follow actual counts (hand-check);
  a time with no controls → NaN counterfactual, `pre_mse` finite over the rest.
- `synthetic_control`: treated unit constructed as an exact convex combination (0.3/0.7) of two
  control units in the pre-period → recovered weights within 0.02, `pre_mse < 1e-6`; weights
  sum to 1 within 1e-6, all ≥ −1e-9.
- **GESS argmin regression** (audit 14): panel where candidate A lowers MSE and candidate B
  raises it → GESS picks A; an argmax implementation would pick B (assert the chosen expansion's
  values). GESS terminates (bounded by total value count) and final `pre_mse` ≤ dd_control's
  `pre_mse` is NOT asserted in general (not guaranteed) — instead assert monotone nonincreasing
  MSE across accepted GESS steps (from `extras["mse_trace"]`).
- Full-dimension variant expands whole dims only (masks all-True on the added dim).
- Empty-control edge: s_τ = D → dd/gess return `pre_mse = +inf`, `y0_hat` all-NaN, no exception
  (NaN-never-0 policy).

Commit: `feat(did): control identification — corrected DD control, simplex synthetic control, GESS argmin`.

---

## Task 9 — Effects: τ̂, dose normalization, backend protocol, τ̂ randomization, per-dimension placebo

**Create:** `src/natex/did/effects.py`, `tests/test_did_effects.py`; extend
`tests/test_did_benchmarks_small.py` (control half).

```python
@dataclass
class DiDEffect:
    tau: float                  # NaN on failure, never 0.0
    se: float                   # time-clustered: sd of per-post-period mean gaps / sqrt(h); documented simple choice
    method: str                 # control method name
    pre_mse: float
    n_treated_post: int
    dose: float | None          # theta DD-contrast used for normalization; None when not applied
    extras: dict

def did_effect(
    panel: CategoricalPanel,
    discovery: DiDDiscovery,
    control: str | ControlResult = "dd",
    dose_normalize: str | bool = "auto",   # auto: True iff theta is not binary {0,1} (audit item 19)
) -> DiDEffect
```

τ̂ = mean over s_τ post-period records of (y − ŷ(0)), skipping NaN counterfactual cells
(count reported); all-NaN → τ̂ = NaN. Dose normalization (audit 19): δ̂ = DD contrast of θ
(same contrast, same control set applied to θ); τ̂ ← τ̂_rf / δ̂; |δ̂| < 1e-10 → NaN.
`panel.y is None` → raise `ValueError` (estimation requires an outcome; discovery upstream
never needed it).

```python
class DiDEstimatorBackend(Protocol):        # spec non-goal boundary: pluggable interface ONLY
    name: str
    def estimate(self, panel, discovery, control: ControlResult) -> DiDEffect: ...

ESTIMATOR_BACKENDS: dict[str, DiDEstimatorBackend]   # {"mean_gap": default backend}
# docstring: staggered-adoption group-time ATT (Callaway–Sant'Anna) is a future backend; the
# protocol is the phase-3 deliverable per audit §3 / spec non-goals.

@dataclass
class TauRandomizationReport:
    p_value: float
    observed: float             # studentized tau_hat / se
    null_stats: np.ndarray      # studentized placebo statistics
    q: int
    mode: str                   # "enumerate" | "sample"

def tau_randomization_test(
    panel, discovery, control: str = "dd",
    Q: int | str = "auto",                    # "auto": enumerate when the untreated-profile pool
    rng: np.random.Generator | None = None,   #   yields <= 200 placebos, else sample Q=199 (rng required)
) -> TauRandomizationReport

def placebo_dimension_tests(
    panel, discovery, control: str = "dd",
    rng: np.random.Generator | None = None, Q: int | str = "auto", alpha: float = 0.05,
) -> PlaceboDimensionReport      # dataclass: p_values: dict[str, float], p_holm, passed
```

`tau_randomization_test` — audit item 5 in full: placebo subsets are **matched in shape** (same
number of covariate profiles as s_τ, drawn from profiles with no treated records, same t0),
statistic is **two-sided studentized** (|τ̂_p/se_p| vs |τ̂/se|), p uses the **+1-rank rule**;
`mode="enumerate"` (all single-profile placebos when s_τ is one profile — the Prop 99 case,
Abadie's placebo-in-space) is deterministic. The docstring states the exchangeability caveat and
the precise conditional claim replacing the thesis's "independence of the two tests": *the scan
statistic is a function of (x, t, θ) only, so conditional on the discovered (s_τ, T₀) the τ̂
test uses y information not used in selection; placebo profiles are treated as exchangeable
with s_τ under H₀, an assumption, not a theorem.* Placebo draws with NaN τ̂ or se are dropped
and counted (`extras["n_failed"]`); fewer than 5 usable placebos → p = NaN (never a fake 1.0).

`placebo_dimension_tests` — thesis §6.3.1(3): for each covariate dimension NOT defining s_τ,
re-estimate τ̂ with that dimension's per-record one-hot share as the outcome and run the same
studentized placebo test; Holm across dimensions; `passed = all holm p > alpha` (vacuously True
when every dim defines s_τ, with a report note).

**Tests:**
- Hand-check τ̂ on the task-8 toy panel: known gap → exact value.
- **Dose normalization** (audit 19): synthetic real-θ DGP (ζ=10, τ=2, n=4000, seeded, true
  RDiT given): raw contrast ≈ ζ·τ (assert within 25%), dose-normalized τ̂ within |τ̂ − 2| ≤ 0.6;
  binary θ auto-skips normalization (`dose is None`).
- **Two-sided regression** (audit 5): planted NEGATIVE effect → p ≤ 0.05 where a one-sided
  95th-percentile rule would give p ≈ 1 (assert the one-sided rank is > 0.5 to prove the case
  discriminates).
- `+1-rank`: enumerate mode with 9 placebos all below observed → p == 1/10.
- Determinism (sampled mode, seeded); `rng=None` with sampling raises; NaN-placebo handling.
- **CI-small control benchmark** (`tests/test_did_benchmarks_small.py`, seeded, true RDiT
  given, 3 seeds averaged): homogeneous DGP (τ=10, ζ=10, binary θ variant, n=2000, d=3, V=4):
  each of dd/synthetic/gess has |τ̂ − 10| ≤ 3 (Fig 6.3 analog). Heterogeneous DGP
  (`hetero_group=True`, same τ): |τ̂_gess − 10| < |τ̂_dd − 10| AND < |τ̂_sc − 10| (Fig 6.5 GESS
  advantage; calibrate seeds — thesis shows order-of-magnitude gaps, so the strict inequality is
  robust).

Commit: `feat(did): effect estimation with dose normalization, studentized tau randomization, dimension placebos`.

---

## Task 10 — Prop 99 registry entry and fetch-data CLI

**Create:** `tests/test_fetch_data.py` (or extend `tests/test_registry.py` + `tests/test_cli.py`).
**Modify:** `src/natex/data/registry.py`, `src/natex/cli.py`.

`DatasetInfo` gains additive fields (defaults keep the 5 existing entries byte-compatible):

```python
@dataclass(frozen=True)
class DatasetInfo:
    ...existing fields...
    time: str | None = None
    unit: str | None = None
    fetch_url: str | None = None    # direct public download when one exists; None = login-gated
```

`load_dataset` passes `time=info.time, unit=info.unit` into `DatasetSpec`.

New entry:

```python
"prop99": DatasetInfo(
    name="prop99",
    relpath="prop99/smoking_data.csv",
    glob_fallback=None,
    treatment="treated",
    outcome="cigsale",
    forcing=(),                       # DiD dataset: no forcing variable
    covariates=("mean_lnincome", "mean_retprice", "mean_age15to24", "mean_beer",
                "cigsale_1975", "cigsale_1980", "cigsale_1988"),
    time="year",
    unit="state",
    n_rows=1209,                      # 39 states x 31 years (1970-2000), header excluded
    fetch_url=("https://raw.githubusercontent.com/OscarEngelbrektson/"
               "SyntheticControlMethods/master/examples/datasets/smoking_data.csv"),
    source=("Abadie, Diamond & Hainmueller (2010) California Prop 99 smoking panel "
            "(state, year, cigsale, lnincome, beer, age15to24, retprice). Fetch with "
            "`natex fetch-data prop99` or download the CSV from the fetch URL and place "
            "it at prop99/smoking_data.csv under NATEX_DATA."),
    notes=("DiD benchmark (thesis ch.6 §6.4.3): treated = (state == 'California') & "
           "(year >= 1989). Covariates are derived at load time as STATE-LEVEL, "
           "time-invariant summaries (means over available years; lnincome starts 1972, "
           "beer covers 1984-1997) plus lagged cigsale in 1975/1980/1988 — the thesis "
           "quantizes each into 4 bins; profile-based subsets then select whole state "
           "trajectories, which is what 'covariate profiles across ALL time points' "
           "requires. Documented deviation: the thesis does not state its aggregation."),
),
```

`_prepare("prop99", df)`: assert expected raw columns; build
`treated = ((state == "California") & (year >= 1989)).astype(float)`; merge per-state
`mean_lnincome/mean_retprice/mean_age15to24/mean_beer` (NaN-skipping means) and
`cigsale_1975/1980/1988` (year pulls) onto every row. Result has no NaN in any spec column.
**Implementation-time check (document result in the method card):** verify California's
4-bin quantized 7-dim profile is unique among the 39 states (`build_panel` + profile_id); if it
is not, raise the bin count for the lagged-cigsale dims to 5 and document.

CLI:

```python
@app.command("fetch-data")
def fetch_data(
    name: str,
    root: Path = typer.Option(None, help="data root; default env NATEX_DATA"),
    force: bool = typer.Option(False, help="re-download over an existing file"),
):
    """Download a dataset that has a public direct URL into the data root.
    Login-gated datasets print their fetch instructions and exit(1)."""
```

Uses `urllib.request.urlopen` (stdlib, no new dependency), streams to a temp file in the target
directory then atomically renames; refuses to overwrite without `--force`; after download runs
`verify(name)` and reports. Never called from tests against the network.

**Tests:**
- Registry: `"prop99" in REGISTRY`; `locate`/`verify` behave with a **synthetic fixture CSV**
  (built in-test: 3 states × 6 years with the 7 raw columns) written under a tmp root —
  `load_dataset("prop99", root=tmp)` (with `n_rows` check monkeypatched or the fixture padded to
  match — simplest: `verify` not called by `load_dataset`, so only column prep is exercised);
  `_prepare` output: `treated` is 1.0 exactly for CA-1989+ rows; derived columns constant per
  state; no NaN in spec columns; spec has `time="year"`, `unit="state"`, `forcing==[]`.
- `natex datasets` (CliRunner) lists prop99 as missing with fetch instructions when root lacks it.
- `fetch-data`: monkeypatch `urllib.request.urlopen` to serve fixture bytes → file lands at
  `prop99/smoking_data.csv`, second call without `--force` exits nonzero with a message, with
  `--force` overwrites; a login-gated name (`academic_probation`) exits(1) printing `source`.

Commit: `feat(data): Prop 99 registry entry with derived DiD columns and fetch-data CLI`.

---

## Task 11 — CLI `--design did` and package exports

**Modify:** `src/natex/cli.py`, `src/natex/__init__.py`, `tests/test_cli.py`.

`natex discover` gains additive options (defaults preserve the phase-2 RDD behavior exactly):

```
--design rdd|did          (default "rdd")
--time COLUMN             (required for did; error message names it)
--unit COLUMN             (optional)
--bins INT                (default 4)
--windows "8,10"          (comma floats; default: data-driven default_windows)
--restarts INT            (default 8)
--method single_delta|wcc|greedy   (default single_delta)
--model auto|normal|bernoulli      (default auto; audit 19 model matching)
```

The did branch: build `Dataset` (with time/unit, `forcing=[]` when none passed), `suddds_scan`,
`panel_randomization_test` (Q=`--q`), `composition_test`, `anticipation_test`, and — when an
outcome column is given — `did_effect` + `tau_randomization_test` for each of dd/synthetic/gess.
Writes the same-shape `results.json` with a `"did"` section: top discoveries (subset_values, t0,
window, llr), scan p-value, validation block, effects block (tau, se, p, pre_mse, dose per
control). Echo lines mirror the RDD branch. The results bundle always reports what was searched:
windows grid, restarts, method, model, dims and bin counts (spec §6b obligation).

Exports in `natex/__init__.py` (append to `__all__`): `suddds_scan`, `SuDDDSResult`,
`DiDDiscovery`, `build_panel`, `make_did_synthetic`, `did_effect`.

**Tests (`tests/test_cli.py` additions, CliRunner):**
- did smoke test on a tiny synthetic DiD CSV (written by `make_did_synthetic`, n=400, d=2, V=3,
  ζ=8, seeded; Q=9, restarts=2, one window): exit 0, `results.json` parses, has `did` section
  with `t0` a float, `p_value` in (0,1], effects present for the three controls, and the
  searched-configuration block present.
- `--design did` without `--time` → exit ≠ 0, message names `--time`.
- `--design rdd` path unchanged (existing smoke test still green, no new options required).
- Import surface: `from natex import suddds_scan, DiDDiscovery` works; `python -c "import natex"`
  stays matplotlib-free (no plotting import at module scope).

Commit: `feat(cli): DiD discovery via --design did with full validation and effects payload`.

---

## Task 12 — Prop 99 backtest, method card, phase status

**Create:** `tests/backtests/test_prop99.py`, `docs/method_cards/suddds.md`,
`docs/status/phase-3.md`.

**Backtest (`@pytest.mark.backtest`, uses the `load_or_skip` fixture; module-scoped fixtures for
the panel and one scan per method; total runtime target < 5 min):**

Fixture: `ds = load_or_skip("prop99")`; `windows=(5.0, 8.0, 10.0)`; `bins=4`; `restarts=8`;
`seed=0` (rng per test from `np.random.default_rng`).

1. `test_scan_recovers_california_1989` — for each method in ("greedy", "wcc", "single_delta")
   with `model="normal"` (thesis parity: the thesis runs Normal-residual models on the binary
   treatment): the TOP discovery has `t0 == 1989.0` exactly and `mask` equal to exactly
   California's 31 records (precision == recall == 1.0). Thesis §6.4.3: all three methods
   recover it perfectly.
2. `test_bernoulli_model_recovers` — `model="bernoulli"` (audit 19 corrected default for binary
   θ), method "wcc": top discovery `t0 == 1989.0` and California ⊆ mask with precision ≥ 0.9.
3. `test_llr_significant` — `panel_randomization_test` (normal model, `null="ar1_unit"`, Q=99,
   seeded): p ≤ 0.05.
4. `test_effects_in_line_with_table_6_1` — conditioning on the recovered discovery:
   - dd: τ̂ ∈ [−17, −5] (printed −10.94), τ̂ < 0;
   - synthetic: τ̂ ∈ [−15, −3] (printed −8.96), `pre_mse` < dd's `pre_mse`;
   - gess: τ̂ ∈ [−13, −1.5] (printed −6.67);
   - `tau_randomization_test` (enumerate mode — 38 placebo states): p ≤ 0.05 for dd AND
     synthetic (thesis: all three significant; assert gess p ≤ 0.11, i.e. rank ≤ 4/39, since our
     corrected two-sided statistic may not reproduce the thesis's one-sided 5% for the weakest
     estimate — if it does come in ≤ 0.05, tighten and note it).
   These bands are "sign/magnitude in line with Table 6.1" (spec §8) around corrected — not
   identical — estimators. If any value lands outside its band, **investigate before widening**
   (systematic-debugging rule) and document the resolution in the method card.
5. `test_validation_battery` — composition test passes (balanced panel → p ≈ 1);
   anticipation test passes (no pre-1989 jump for California at shifts 1–3);
   `placebo_dimension_tests` passes with Holm α = 0.05.
6. `test_scan_never_reads_outcome` — scan on prop99 with `outcome=None` equals the
   outcome-loaded scan bitwise (llr, t0, mask).

**Method card `docs/method_cards/suddds.md`:** the corrected math (window convention, Eq 6.9,
profiled single-Δ derivation with C̃/B̃, Bernoulli variant), Algorithms 6–9 with each audit
repair called out (items 5, 11–19, 24 + the Eq 6.18/6.19/6.20/6.21/6.24–6.28 typos), all
hyperparameters and defaults (W grid, restarts, bins, n_rho, exhaustive_max_values, min_side)
with the spec-§10 note that the thesis reports none of them, deviations (state-level covariate
aggregation for Prop 99; composition/anticipation replacing McCrary-in-time; outcome-only
synthetic control; Poisson count-treatment variant deferred; unbalanced-panel limits of the
ar1_unit null), and the precise conditional statement replacing the "independent tests" claim.

**Status `docs/status/phase-3.md`:** gate record — unit/property/CI-small counts, backtest
results (`-m backtest` output for prop99 + the phase-2 rows still green), ruff clean, runtime
numbers, deviations log, follow-ups for phase 4+.

Final gate (run and record all of):
```
uv run pytest -q                      # full unit suite, no backtests
uv run ruff check src tests
NATEX_DATA="..." uv run pytest -q -m backtest   # ALL backtests incl. phase-2 rows (no regressions)
```

Commit: `test(backtest): Prop 99 SuDDDS recovery, Table 6.1 effects, validation battery + phase-3 docs`.

---

## Explicitly deferred (documented, not silently dropped)

- Poisson observation model for count treatments (traffic-stop replication) — audit 19 notes it;
  method card marks it future work; the Bernoulli variant (the phase's binary case) is in scope.
- Staggered-adoption estimators (Callaway–Sant'Anna) — `DiDEstimatorBackend` protocol ships,
  implementation is a spec non-goal for v1.
- Stanford Open Policing replication (thesis §6.4.4) — no local data; candidate future backtest.
- `natex.discover(design="auto")` unified ranking of RDD + DiD discoveries — phase 6 (analyst
  pass) territory; phase 3 delivers the `--design did` CLI path and library API.
- Thesis Fig 6.7 ρ-noise robustness curve on Prop 99 — provided by `benchmarks/run_did_curves.py
  --prop99-noise` only if time permits; otherwise listed as a follow-up in the status doc.
