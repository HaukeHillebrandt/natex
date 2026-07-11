# Phase 4 implementation plan — DEE layer (dee/) + scaled sim-1 benchmark

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`;
DEE detail in the companion `math_audit.md` and `codex_second_opinion.md` in that notes dir).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` §3 (dee/), §4 (extras
`natex[gp]`, `natex[ml]`), §5 item 5 (DEE repairs), §8 ("a scaled-down DEE simulation-1 replication
(MSE of debiased vs raw causal forest)"), §9 phase 4, §10 ("DEE GP stack is heavy → optional extra;
small-N defaults").
**Method notes:** `".../RDD/docs/notes/review_dee-repo.md"` (pipeline architecture, Algorithm 1
landmines, correctness risks 1–11, refactor recommendations 1–7).
**No real data needed this phase** — spec §8 has no real-data DEE row; the gate is the synthetic
sim-1 replication. Backtests from phases 2–3 must stay green (`-m backtest` with `NATEX_DATA`).

## Phase objective (spec §9 phase 4)

1. **DEE** (Jakubowski et al., JMLR 2023) as `src/natex/dee/`, with every audit repair:
   - **Voronoi–KNN repair** (paper Alg 1): LLR-ranked candidate centers → disjoint
     quasi-experiments (k′-NN ball ∩ Voronoi cell, per-side support threshold **including the
     first center**, `train_data` explicit — never a global), **direct orthogonal projection**
     centroids (no random Gram–Schmidt basis, no RNG), M′ selection from the phase-1
     randomization-test null.
   - **Local effects at the repaired experiments** reusing the phase-1 frozen-side 2SLS
     (HC1, first-stage diagnostics) plus balance filtering reusing the phase-1 placebo battery.
   - **Conditioned-on-observables estimator layer**: `ObservationalEstimator` protocol, a
     core-deps sklearn T-learner default, an econml `CausalForestDML` adapter behind the new
     `natex[ml]` extra, and **experiment-level cross-fitting** (audit 9) so the bias observation
     at each center never comes from a model trained on that center's members.
   - **Two-stage heteroskedastic GP debiasing**: exact numpy/scipy heteroskedastic-noise RBF GP
     (core deps — the surfaces are fit on tens of centers, O(|U|³) is trivial), hierarchical
     stage-1 noise via a chi-square measurement model on SE² (audit §3 adopted), bias-surface GP
     and direct CATE-extrapolation GP. Optional GPyTorch backend behind `natex[gp]` for scale.
   - **Model weighting**: MLL and closed-form-LOO softmax (paper baselines, kept for comparison)
     plus **buffered predictive stacking** as the default (audit §3: refit hyperparameters within
     spatially buffered folds, optimize the mixture weight — not softmax-PLP); mixture posterior
     with the correct covariance and **one model label per posterior draw** (audit 8).
2. **Scaled sim-1 benchmark** (spec §8): corrected latent-index DGP with GP-sampled CATE and bias
   surfaces (`data/synthetic_dee.py`), harness `benchmarks/run_dee_sim.py`, CI-small seeded slice
   asserting **MSE(debiased) < MSE(raw CF)** on the truth grid.
3. `natex debias` CLI, `docs/method_cards/dee.md`, `docs/status/phase-4.md`, README roadmap tick.

Out of scope (documented, not silently dropped): rural-roads application (no local data; the
Cattaneo/Kallus R benchmarks need rdrobust/rdmulti — spec treats them as paper artifacts, not
natex features); overlap-aware influence-function aggregation (audit §3 keeps **disjointness via
VKNN as the simple default** — method-card note only).

## Audit corrections that bind this phase (docs/math_audit_final.md)

| # | Correction | Where implemented |
|---|---|---|
| 7 | Theorem 1 / Lemmas 1–2: Lemma 1 misses cross terms; Lemma 2 repaired via sphere geometry E[q²] ≤ 2ρ²; implication needs the **√2 factor** and holds only along fixed-shape scaling | `dee/vknn.py::experiment_radius` diagnostic + method-card statement of the corrected bound; DEEResult.diagnostics carries per-experiment radii; natex never uses the printed (uncorrected) bound |
| 8 | Mixture intervals: draw the model label **once per posterior draw**; mixture covariance `wΣ_a + (1−w)Σ_b + w(1−w)(μ_a−μ_b)(μ_a−μ_b)ᵀ` | `dee/bma.py::MixturePosterior` (analytic cov + label-per-draw sampling) |
| 9 | Same-data forest + local-IV correlation: **cross-fit** the observational estimator vs the discovery/IV stage; never treat classical SE² as known independent GP noise | `dee/observational.py::experiment_crossfit_cate` (leave-experiment-out folds); SE² are HC1 (not classical, fixing DEE repo risk 7) and enter through the stage-1 chi-square measurement model (`dee/noise.py`), documented as estimated, not known |
| 10 | First-stage relevance checked after repair | reuse `EffectEstimate.first_stage_t / weak_instrument`; weak experiments are kept but their large SE² automatically downweights them in the heteroskedastic GP (documented) |
| §3 adopted | **Buffered predictive stacking** (refit hyperparameters within folds) instead of softmax-PLP; softmax MLL/LOO kept as comparison strategies | `dee/bma.py::buffered_stacking_weights` (default), `mll_weights`, `loo_weights` |
| §3 adopted | **Hierarchical stage-1 noise**: Normal likelihood + chi-square measurement model on SE² (not both t and latent variance) | `dee/noise.py::smooth_noise` |
| §3 adopted | Disjointness (VKNN) as the default aggregation; influence-function overlap aggregation deferred | method card |
| §3 rejected | "Recompute index sets after recentering" is unnecessary (projection leaves the separating hyperplane unchanged); keep the **two real Alg-1 code fixes**: threshold the first candidate, explicit `train_data` | `dee/vknn.py` (both fixes + regression tests) |
| repo risks 1–3 | global-`df` bug → all inputs explicit; first center installed before threshold check → checked like every other; random rnorm Gram–Schmidt basis → deterministic direct projection `c − ((c−x₀)·n̂)n̂` | `dee/vknn.py` |
| repo risk 4 | step-4 silent truncation on row mismatch → VKNN index sets computed **once** and passed by object, never recomputed downstream | `dee/debias.py` consumes `VKNNResult` verbatim |
| repo risk 8 | unseeded 1-MC BLOOCV / grf / rnorm draws → one `numpy.random.Generator` through every stochastic call; the paper's "random dist" 1-MC strategy is **not** reimplemented (superseded by deterministic buffered stacking; method-card note) | all of `dee/` |
| typos | bias **sign-convention drift** across paper appendices (Codex #30): natex pins `bias_u = cate_obs_u − τ̂_u`, `debiased(x) = cate_obs(x) − bias_gp(x)`, with a regression test; M vs M′, weighted-BLOOCV missing mixture weights — method card | `dee/debias.py`, method card |

Also inherited: +1-rank Monte Carlo p-values where p-values appear; NaN never 0.0 on failure;
discovery (scan + VKNN + balance filter) never reads `y`; no bare except.

## House rules (bind every task)

Python ≥3.11; core deps only numpy/scipy/pandas/scikit-learn/typer/pydantic — **the entire DEE
default path runs on core deps** (exact GP in numpy/scipy; T-learner on sklearn). econml
(`natex[ml]`) and torch/gpytorch (`natex[gp]`) are optional extras whose tests
`pytest.importorskip` gracefully; CI (3.11–3.14) installs neither and must stay green.
One `numpy.random.Generator` through every stochastic call (mirror the repo's
`raise ValueError("pass an explicit numpy Generator")` convention); identical seed ⇒ identical
output. NaN never 0.0. Never commit datasets. Conventional commit after every green cycle.
`uv run pytest -q` excludes backtests; the phase-2/3 backtests are re-run once at the end
(task 10 gate) with `-m backtest` and `NATEX_DATA` set.

**TDD discipline for every task:** write the failing test(s) first, run to confirm failure,
implement, run `uv run pytest -q` and `uv run ruff check src tests`, then commit.

**Statistical-test policy:** every stochastic assertion is seeded; calibrate thresholds across ≥5
seeds during implementation, then pin one seed with a margin. Thresholds below are starting
points — change only with a code comment stating the calibration evidence.

## Current interfaces built upon (do not break; all changes additive)

- `natex.data.spec.Dataset` / `DatasetSpec`: properties `n, T, y, Z, Z_std, X,
  treatment_is_binary`; `Z_std` = per-column (Z − mean)/sd with ddof=0 and zero-sd→1.
- `natex.rdd.lord3.Discovery(center_index, k, llr, normal, members, group1, p_value, extras)` —
  `normal` is a unit vector in Z_std space; `lord3_scan(dataset, k, model, degree, rng, geometry,
  centers) -> LoRD3Result` (discoveries LLR-sorted desc).
- `natex.estimate.local2sls.local_2sls(dataset, d) -> EffectEstimate(tau, se, ci, method,
  first_stage_jump, first_stage_t, weak_instrument, n_used)` — duck-typed: uses only
  `d.members, d.group1, d.center_index, d.normal`. **`QuasiExperiment` deliberately carries these
  four attributes with identical names/types so `local_2sls`, `placebo_tests`, `density_test`,
  and `signed_distance` work on it unchanged.**
- `natex.validate.placebo.hc1_ols(Xmat, yvec)`, `signed_distance(dataset, d)`,
  `placebo_tests(dataset, d, alpha) -> PlaceboReport(p_values, p_holm, passed)`.
- `natex.validate.randomization.randomization_test(...) -> RandomizationReport(p_value,
  observed_max_llr, null_max_llrs, q)` — `null_max_llrs` feeds M′ selection.
- `natex.scan.geometry.ScanGeometry / build_geometry(Z_std, k)`; `natex.scan.neighborhoods`
  tie convention: signed distance ≥ 0 ⇒ group 1 (audit 23) — VKNN sides use the same convention.
- `natex.data.synthetic.make_synthetic` (style template for `make_dee_synthetic`).
- CLI typer app in `natex/cli.py` (`datasets`, `fetch-data`, `discover`); `_clean` JSON helper.
- Benchmarks style: `benchmarks/run_did_curves.py` + CI-small slices in
  `tests/test_did_benchmarks_small.py`.

## Design conventions fixed for the whole phase

- **All DEE geometry lives in `Z_std` space** (same space as the scan): k′-NN balls, Voronoi
  cells, projected centroids, GP inputs, buffer distances, and the observational estimator's
  features. Rationale: the paper's surfaces live in the scan space (sim: X1,X2; roads:
  lon/lat/pop); covariates outside Z enter only the balance checks. Documented in the method
  card. Query points are supplied in **raw Z units** and standardized internally via the new
  `Dataset.standardize` (task 2) so they are bitwise-consistent with `Z_std`.
- **Sign convention (pinned):** `bias_u = cate_obs(center_u) − τ̂_u`; `cate_debiased(x) =
  cate_obs(x) − bias_gp_mean(x)`. Regression test: an observational estimator that overshoots by
  +3 yields bias_obs ≈ +3 and debiased ≈ truth.
- **Voronoi tie-break:** a point equidistant to two accepted centers belongs to the earlier-
  accepted (higher-LLR-rank) center — deterministic, documented.
- **Model A / model B naming:** A = "debias" (obs − bias-GP), B = "direct" (CATE-extrapolation
  GP). `ModelWeights.w_debias` is the weight on A.
- **Determinism:** the caller-supplied Generator drives (in order) GP fit restarts, stacking fold
  assignment, estimator seeds (via `int(rng.integers(2**32))`), and mixture sampling. VKNN and
  the projection are RNG-free by construction.
- **NaN policy:** experiments whose 2SLS is underdetermined (NaN tau/se) are excluded from both
  GPs and listed in `DEEResult.diagnostics["dropped"]`; if fewer than 3 usable experiments
  remain, all GP-derived outputs are NaN arrays (never 0.0) with a diagnostic reason.

---

## Task 1 — Commit this plan; Voronoi–KNN repair + M′ selection

**First action of this task:
`git add docs/plans/phase-4.md && git commit -m "docs: phase 4 implementation plan"` — the plan
file is committed before any code.**

**Create:** `src/natex/dee/__init__.py`, `src/natex/dee/vknn.py`, `tests/test_vknn.py`.

```python
# dee/vknn.py
@dataclass
class QuasiExperiment:
    center_index: int        # dataset row index of the accepted center
    members: np.ndarray      # int row indices: k'-NN ball ∩ Voronoi cell (center included)
    group1: np.ndarray       # bool over members; signed distance >= 0 side (audit 23 convention)
    normal: np.ndarray       # unit normal in Z_std space, frozen from the Discovery
    llr: float               # the discovery LLR that ranked this center
    centroid: np.ndarray     # Z_std[members].mean(axis=0)
    projected_center: np.ndarray  # centroid - ((centroid - Z_std[center_index]) @ normal) * normal

@dataclass
class VKNNResult:
    experiments: list[QuasiExperiment]   # in acceptance order (LLR rank order)
    accepted: np.ndarray                 # int ranks into the candidate list that were accepted
    rejected: np.ndarray                 # int ranks rejected by the support test
    k_prime: int
    t_side: int

def voronoi_knn_repair(
    dataset: Dataset,
    discoveries: Sequence[Discovery],
    m_prime: int,
    k_prime: int = 200,
    t_side: int = 30,
) -> VKNNResult: ...

def select_m_prime(
    scan_result: LoRD3Result, null_max_llrs: np.ndarray, level: float = 0.95
) -> int: ...

def experiment_radius(dataset: Dataset, e: QuasiExperiment) -> float: ...
```

Algorithm (paper Alg 1 with both audit-mandated code fixes):

1. Candidates = `discoveries[:m_prime]` (input is LLR-sorted; assert non-increasing llr, else
   sort defensively). All inputs explicit — no module/global state (repo risk 1).
2. k′-NN lists via one `cKDTree(Z_std)` built once; per-candidate `tree.query(Z_std[c],
   k=min(k_prime, n))`, cached per center.
3. Forward stepwise: tentatively add candidate c to the accepted set; recompute every accepted
   center's index set = its k′-NN ball ∩ {rows whose nearest accepted center is that center}
   (Voronoi over accepted centers only, distances in Z_std, tie → earlier-accepted center);
   count members on each side of that center's frozen hyperplane (`(Z_std[members] −
   Z_std[center]) @ normal >= 0`). Accept c iff **every** accepted center (including c, and
   including the very first candidate — repo risk 2 fix) retains ≥ t_side members on **both**
   sides; otherwise reject c and restore the previous index sets.
4. `projected_center` by direct orthogonal projection (repo risk 3 fix — no random basis, no
   RNG): `centroid − ((centroid − Z_std[center]) @ n̂) n̂`.
5. `select_m_prime` = `int(np.sum(llrs > np.quantile(null_max_llrs, level)))` over the
   scan result's per-center max LLRs (strictly greater; the paper's rural-roads M′ rule).
6. `experiment_radius` = `max ||Z_std[members] − projected_center||₂` (Theorem-1 diagnostic;
   the corrected bound E[q²] ≤ 2ρ² with the √2 factor is *stated in the method card*, task 10).

**Tests (`tests/test_vknn.py`), written first:**

- *Hand-built disjointness:* 60 points on a 2-D grid, two synthetic `Discovery` objects with
  known normals; after repair, `set(e1.members) ∩ set(e2.members) == ∅`, every member's nearest
  accepted center is its owner, and each side count ≥ t_side (t_side=5).
- *First-center thresholding (regression for repo risk 2):* a single candidate whose k′-ball has
  only 2 points on one side with t_side=5 ⇒ `experiments == []`, `rejected == [0]` (the legacy
  code would have kept it).
- *Rejection restores state:* candidate B whose acceptance would starve accepted candidate A's
  side counts is rejected and A's members are bitwise unchanged (compare copies before/after).
- *Projection correctness:* projected_center lies on the hyperplane
  (`|(p − Z_std[center]) @ n̂| < 1e-12`), equals the hand formula, and is idempotent
  (projecting again is a no-op); `voronoi_knn_repair` takes no rng and two consecutive calls are
  bitwise identical (RNG-free determinism).
- *Tie-break determinism:* symmetric configuration where a point is exactly equidistant to two
  centers ⇒ owned by the earlier-accepted one.
- *select_m_prime:* llrs [9, 7, 5, 3, 1], null max draws with 95th pct = 4.0 ⇒ 3; all-null-above
  ⇒ 0 (empty candidate list is legal and yields `VKNNResult(experiments=[]...)`).
- *k_prime > n* is clamped, not an error; `m_prime = 0` ⇒ empty result.
- *y-blindness:* repair on a dataset copy with `outcome=None` gives bitwise-identical members /
  group1 / projected centers (VKNN never touches y).

Commit: `feat(dee): Voronoi-KNN repair with thresholded first center, direct projection, M' selection`.

## Task 2 — Effects at experiments, balance filter, `Dataset.standardize`

**Create:** `tests/test_dee_effects.py`.
**Modify:** `src/natex/dee/vknn.py`, `src/natex/data/spec.py`, `tests/test_spec.py`.

```python
# data/spec.py (additive)
class Dataset:
    def standardize(self, z: np.ndarray) -> np.ndarray:
        """Map raw forcing-space points to Z_std coordinates (same mean/sd/ddof=0, zero-sd->1)."""

# dee/vknn.py (additions)
def experiment_effects(dataset: Dataset, result: VKNNResult, method: str = "2sls") -> list[EffectEstimate]:
    ...  # method in {"2sls", "wald"}; delegates to estimate.local2sls on each QuasiExperiment

def balance_filter(dataset: Dataset, result: VKNNResult, alpha: float = 0.05) -> VKNNResult:
    ...  # keep experiments whose placebo_tests(...).passed; returns a NEW VKNNResult
```

`experiment_effects` relies on the duck-typing contract (QuasiExperiment has
members/group1/center_index/normal); no re-implementation. `balance_filter` reuses
`validate.placebo.placebo_tests` per experiment (Holm within experiment — stricter than the
paper's Bonferroni-across-covariates; documented in the method card). Filtering only reads
covariates, never `y`.

**Tests:**

- `Dataset.standardize(dataset.Z)` is bitwise equal to `dataset.Z_std`; a zero-variance column
  passes through unscaled; shape errors raise ValueError.
- Sharp 1-D synthetic (n=1200, T = 1{x ≥ 0}, y = 1 + 2·T + 0.2ε, seeded): build one
  QuasiExperiment straddling the cutoff via `voronoi_knn_repair` on a real `lord3_scan` top
  discovery; `experiment_effects` returns tau within [1.6, 2.4] and `weak_instrument is False`.
- NaN policy: set y[members] = NaN for one experiment ⇒ its EffectEstimate has NaN tau/se
  (never 0.0), n_used == 0-or-small, and other experiments are unaffected.
- Balance filter: inject a covariate that jumps exactly at one experiment's boundary
  (c = 5·group1 + noise); that experiment is dropped, a clean one kept; `alpha=1e-9` keeps both;
  the returned object is new (input VKNNResult unmodified).
- `experiment_radius` on a hand-built 3-point experiment equals the hand-computed max distance.

Commit: `feat(dee): local 2SLS effects and balance filtering on repaired experiments`.

## Task 3 — Exact heteroskedastic GP (core deps) + prior sampling

**Create:** `src/natex/dee/gp.py`, `tests/test_dee_gp.py`.

```python
# dee/gp.py
def rbf_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float, outputscale: float) -> np.ndarray
def sample_gp_prior(X: np.ndarray, lengthscale: float, outputscale: float,
                    rng: np.random.Generator, size: int = 1, jitter: float = 1e-8) -> np.ndarray
    # (size, n) draws via Cholesky with jitter escalation (1e-8 -> 1e-4, then LinAlgError)

@dataclass
class GPPosterior:
    mean: np.ndarray                      # (m,)
    cov: np.ndarray                       # (m, m)
    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray  # (size, m)

@dataclass
class HeteroskedasticGP:
    lengthscale: float
    outputscale: float
    mean_const: float
    X: np.ndarray                         # training inputs actually used (finite rows)
    y: np.ndarray
    noise_var: np.ndarray                 # per-point KNOWN noise variances
    # cached: chol of K + diag(noise), alpha = K^-1 (y - mean_const)

    @classmethod
    def fit(cls, X, y, noise_var, rng: np.random.Generator,
            n_restarts: int = 4,
            lengthscale_bounds: tuple[float, float] = (1e-2, 1e2),
            outputscale_bounds: tuple[float, float] = (1e-4, 1e4)) -> "HeteroskedasticGP"
    def log_marginal_likelihood(self) -> float
    def posterior(self, Xq: np.ndarray) -> GPPosterior
    def loo_log_predictive(self) -> float   # closed form, no refits
```

Implementation notes: zero-mean-plus-constant GP; `fit` drops non-finite (y, noise) rows
(recording how many); maximizes the exact MLL over (log ℓ, log s², m) with
`scipy.optimize.minimize(method="L-BFGS-B")`, restarts drawn from `rng` (log-uniform in bounds),
best MLL wins; deterministic given rng. If < 2 finite rows remain, `posterior` returns NaN
mean/cov (never 0.0) and `log_marginal_likelihood` returns NaN. Closed-form LOO with fixed
hyperparameters via the standard identities `μ₋ᵢ = yᵢ − [K̃⁻¹r]ᵢ/[K̃⁻¹]ᵢᵢ`,
`σ²₋ᵢ = 1/[K̃⁻¹]ᵢᵢ` on K̃ = K + diag(noise) (predictive of the noisy yᵢ). Distances computed on
inputs as given (callers pass Z_std-space points).

**Tests (all seeded):**

- *Analytic 2-point case:* X=[[0],[1]], y=[0,1], noise=1e-12, ℓ=1, s²=1 fixed (bypass fit via
  direct construction): posterior mean at X equals y (atol 1e-6), var < 1e-8; posterior at
  x=0.5 equals the hand-computed kriging mean (atol 1e-8).
- *Heteroskedastic pull:* duplicate x with y=(0, 10), noise=(1e-4, 1e4): posterior mean at x
  within 0.05 of 0.
- *Fit improves MLL:* fitted MLL ≥ MLL of every restart's initial parameters (assert with the
  optimizer's bookkeeping exposed via a `fit_report` dict or by re-evaluating).
- *LOO closed form == brute force:* n=12 random 2-D points; compare `loo_log_predictive`
  against explicitly refit-free leave-one-out posteriors built from the other 11 points with the
  same hyperparameters (rtol 1e-6).
- *Prior draws:* n=400 1-D grid; empirical variance of 200 draws within ±25% of outputscale;
  mean squared successive difference strictly larger for ℓ=0.1 than ℓ=1.0.
- *Determinism:* same rng seed ⇒ bitwise-identical fit and samples; different seeds ⇒ same
  hyperparameters within rtol 1e-3 on an easy problem (restart robustness).
- *NaN policy:* one NaN y row is dropped (fit on the rest); all-NaN ⇒ NaN posterior, never 0.0.

Commit: `feat(dee): exact heteroskedastic RBF GP with closed-form LOO and seeded prior sampling`.

## Task 4 — Stage-1 hierarchical noise smoothing

**Create:** `src/natex/dee/noise.py`, `tests/test_dee_noise.py`.

```python
# dee/noise.py
def smooth_noise(
    X: np.ndarray,               # (u, d) experiment centers (Z_std space)
    se2: np.ndarray,             # (u,) observed squared standard errors
    df: np.ndarray | None,       # per-experiment residual dof (n_used - 4 for 2sls); None -> pooled fallback
    rng: np.random.Generator,
    floor_frac: float = 1e-3,
) -> np.ndarray                  # (u,) smoothed noise variances
```

Audit §3 adopted ("Normal likelihood + chi-square measurement model on SE², not both t and
latent-variance"), moment-matched on the log scale: with SE²·df/σ² ~ χ²_df,
`log SE²` has known bias `ψ(df/2) − log(df/2)` and known variance `ψ₁(df/2)`
(`scipy.special.digamma/polygamma`). Debias `v = log se2 − bias`, fit `HeteroskedasticGP` on
(X, v, noise_var=ψ₁(df/2)) (rng-driven restarts), return
`exp(posterior_mean + 0.5·posterior_var)` (log-normal mean back-transform), floored at
`floor_frac · median(finite se2)` — data-scaled, never absolute (audit 24 lineage).
`df=None` fallback: measurement variance = pooled sample variance of finite `v` (documented
heuristic). Non-finite se2 entries propagate NaN out (caller drops those experiments — NaN
never 0.0). df is clipped to ≥ 1 with a diagnostic.

**Tests (seeded, calibrate across ≥5 seeds then pin one):**

- *Recovery:* constant true σ²=4 over 40 centers, se2 ~ (4/df)·χ²(df=10) draws: mean squared
  log-error of smoothed vs truth < 0.5 × that of raw se2 (starting threshold).
- *Outlier shrinkage:* one center at 50× the ambient se2 among 30 smooth neighbors ⇒ its
  smoothed value < 10× the local median.
- *Chi-square constants:* for df=10, the debiasing constant equals ψ(5) − log(5) and the
  measurement variance equals ψ₁(5) (direct asserts against scipy, guards regressions).
- *Fallback path:* df=None returns finite positive values on clean input.
- *Floor & NaN:* zero se2 in ⇒ output ≥ floor > 0; NaN se2 in ⇒ NaN out at that index only.

Commit: `feat(dee): chi-square measurement-model smoothing of local-effect noise variances`.

## Task 5 — Model weights (MLL / LOO / buffered stacking) + mixture posterior

**Create:** `src/natex/dee/bma.py`, `tests/test_dee_bma.py`.

```python
# dee/bma.py
@dataclass
class ModelWeights:
    w_debias: float              # weight on model A = (obs-CATE − bias GP); w_direct = 1 - w_debias
    strategy: str                # "mll" | "loo" | "stacking"
    detail: dict                 # per-fold log scores, buffer used, fold sizes, ...

def mll_weights(gp_bias: HeteroskedasticGP, gp_direct: HeteroskedasticGP) -> ModelWeights
def loo_weights(gp_bias: HeteroskedasticGP, gp_direct: HeteroskedasticGP) -> ModelWeights

def buffered_folds(Xc: np.ndarray, n_folds: int, buffer: float,
                   rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]
    # rng-shuffled fold assignment; train indices additionally exclude any center
    # within `buffer` (Z_std distance) of ANY held-out center in that fold

def buffered_stacking_weights(
    centers: np.ndarray,                  # (u, d) experiment centers
    tau_hat: np.ndarray,                  # (u,) local effects (the held-out targets)
    obs_at_centers: np.ndarray,           # (u,) cross-fitted observational CATE at centers
    noise_var: np.ndarray,                # (u,) smoothed noise variances
    rng: np.random.Generator,
    n_folds: int = 5,
    buffer: float | str = "auto",         # "auto" -> median inter-center NN distance
    n_restarts: int = 2,
) -> ModelWeights

@dataclass
class MixturePosterior:
    mean: np.ndarray             # w μ_a + (1-w) μ_b
    cov: np.ndarray              # w Σ_a + (1-w) Σ_b + w(1-w)(μ_a-μ_b)(μ_a-μ_b)^T  (audit 8)
    w: float
    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray
        # ONE Bernoulli(w) model label per draw, then a draw from that component (audit 8)

def mixture_posterior(post_a: GPPosterior, post_b: GPPosterior, w: float) -> MixturePosterior
```

`mll_weights`/`loo_weights`: softmax of the two log scores (paper baselines; documented as
kept-for-comparison, superseded by stacking as default). `buffered_stacking_weights`: within
each fold, **refit both GPs' hyperparameters** on the buffered training subset (rng restarts),
model A's held-out predictive = Normal(obs_at_centers[i] − bias_gp₋fold(center_i),
bias_var + noise_var[i]) for target tau_hat[i]; model B's = Normal(direct_gp₋fold mean,
var + noise_var[i]); pick w on a 101-point grid over [0, 1] maximizing the summed held-out
log mixture density (deterministic 1-D optimization; no softmax-PLP — audit §3). Folds with an
empty buffered training set contribute nothing and are recorded in `detail`. If u < n_folds,
n_folds is reduced to max(2, u) with a diagnostic; u < 3 ⇒ ModelWeights(w=NaN) and the caller
falls back to `loo_weights` (documented).

**Tests (seeded):**

- *Softmax correctness:* two GPs with hand-set MLLs (monkeypatch `log_marginal_likelihood`) ⇒
  w = exp(m₁)/(exp(m₁)+exp(m₂)) (rtol 1e-12); equal scores ⇒ 0.5.
- *Mixture covariance formula:* 2-point posteriors with hand-set (μ, Σ) ⇒ cov matches the
  audit-8 formula element-wise (rtol 1e-12); mean is the convex combination.
- *Label-per-draw regression (audit 8):* with μ_a ≠ μ_b, the Monte-Carlo variance (40 000
  seeded draws) of the **average over query points** under `MixturePosterior.sample` exceeds the
  same statistic computed with independent per-point labels (implemented inline in the test) by
  ≥ 1.5× — the exact defect the audit flags (too-narrow aggregate CIs) must be visible.
- *MC agreement:* empirical mean/cov of 40 000 draws within rtol 0.1 of the analytic mixture
  mean/cov.
- *Buffered folds:* no training index within `buffer` of any held-out index (assert over all
  folds); every center held out exactly once; deterministic given seed.
- *Stacking picks the right model:* simulate 24 centers where tau_hat = obs − true smooth bias +
  small noise (model A correct) and the direct surface is flat-wrong ⇒ w_debias > 0.8; mirror
  case ⇒ w_debias < 0.2 (calibrate margins across ≥5 seeds, pin one).
- *Degenerate:* u=2 ⇒ NaN weights, no exception.

Commit: `feat(dee): buffered predictive stacking, softmax baselines, correct mixture posterior`.

## Task 6 — Observational CATE layer + experiment-level cross-fitting

**Create:** `src/natex/dee/observational.py`, `tests/test_dee_observational.py`.

```python
# dee/observational.py
@runtime_checkable
class ObservationalEstimator(Protocol):
    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> "ObservationalEstimator": ...
    def predict_cate(self, Xq: np.ndarray) -> np.ndarray: ...

@dataclass
class TLearner:                       # core-deps default (sklearn GradientBoostingRegressor)
    seed: int                         # callers derive: int(rng.integers(2**32))
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    min_treated: int = 20             # per-arm minimum; below -> predict_cate returns NaN

def default_factory(rng: np.random.Generator) -> Callable[[], ObservationalEstimator]
    # returns a zero-arg factory producing independently seeded TLearners from rng

def experiment_crossfit_cate(
    dataset: Dataset,
    result: VKNNResult,
    factory: Callable[[], ObservationalEstimator],
    rng: np.random.Generator,
    n_folds: int = 5,
) -> np.ndarray                       # (u,) obs-CATE at each experiment's projected_center,
                                      # predicted by a model whose training rows exclude that
                                      # experiment's members (audit 9)
```

Features are `dataset.Z_std` (phase convention above); binary T required for TLearner (fit two
regressors on T==1 / T==0 rows; CATE = μ̂₁ − μ̂₀); non-binary T with the default factory raises
ValueError directing the user to supply an econml factory (task 10). Cross-fitting: experiments
are rng-assigned to `min(n_folds, u)` folds; for fold f, one model is fit on all rows minus the
union of fold-f experiments' members; each fold-f experiment's centroid is predicted from that
model. NaN from an underdetermined arm propagates (never 0.0).

**Tests (seeded):**

- *Protocol conformance:* `isinstance(TLearner(seed=0), ObservationalEstimator)`.
- *Recovery:* n=2000, X~U[0,1]², T~Bern(0.5), y = sin(3x₀) + 2T + 0.3ε ⇒
  |mean(predict_cate on a grid) − 2| < 0.3.
- *Confounded DGP shows bias (sanity for the whole phase):* y = 2T + 3u where u confounds T ⇒
  T-learner grid CATE mean > 3 (it must be biased for DEE to have something to fix).
- *Leak regression (audit 9):* poison one experiment's members with y += 1000; the cross-fitted
  prediction at that experiment's centroid changes by < 5 versus the unpoisoned run, while a
  full-fit prediction changes by > 50 (demonstrates exclusion actually happened).
- *Determinism:* same rng ⇒ bitwise-identical output; folds cover every experiment exactly once.
- *NaN:* an experiment inside a region with < min_treated treated rows in its training folds
  still yields a finite prediction (global model), but an all-control DATASET yields NaN CATE
  (never 0.0).

Commit: `feat(dee): observational CATE protocol, sklearn T-learner, leave-experiment-out cross-fitting`.

## Task 7 — Scaled simulation-1 DGP with GP-sampled surfaces

**Create:** `src/natex/data/synthetic_dee.py`, `tests/test_synthetic_dee.py`.

```python
# data/synthetic_dee.py
@dataclass
class DEETruth:
    cate_train: np.ndarray        # (n,) tau(X_i)
    bias_train: np.ndarray        # (n,) beta(X_i)
    cate_query: np.ndarray        # (m,) tau at grid points
    bias_query: np.ndarray        # (m,)
    query: np.ndarray             # (m, 2) raw-unit grid points
    complier_type: np.ndarray     # (n,) int: 0=never, 1=complier-Z1, 2=complier-Z2, 3=always
    thresholds: tuple[float, float]

def make_dee_synthetic(
    n: int,
    *,
    cate_lengthscale: float = 0.5,
    bias_lengthscale: float = 0.5,
    outputscale: float = 1.0,
    thresholds: tuple[float, float] = (1/3, 2/3),      # nested corner cutoffs b1 < b2
    type_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    grid: int = 25,                                     # query grid resolution (m = grid**2)
    noise_sd: float = 0.5,
    constant_surfaces: tuple[float, float] | None = None,  # (tau0, beta0) overrides GP draws
    rng: np.random.Generator,                           # required (ValueError if None)
) -> tuple[Dataset, DEETruth]
```

Corrected reconstruction of the paper's simulation 1 (deviations documented in the docstring and
method card — the paper's exact complier-shift calibration is repo code, not printed math):

- X ~ U[0,1]² ; nested corner instruments `Z_j = 1{x₀ ≥ b_j and x₁ ≥ b_j}` (two square-corner
  RDs, as in the repo's step 1).
- Complier type G ~ Categorical(type_probs) i.i.d.; observed treatment
  `D = 1[G=3] + 1[G=1]·Z1 + 1[G=2]·Z2` (always/never-takers give overlap everywhere).
- τ(·), β(·): one joint `sample_gp_prior` draw each (reusing `dee/gp.py`) over the stacked
  train+grid points (RBF, given lengthscales/outputscale), or constants when
  `constant_surfaces` is set.
- Confounding term with an **exact conditional-bias identity**: the region r(X) ∈ {00, 10, 11}
  (Z1Z2 patterns; nesting makes 01 impossible) has closed-form
  `q1_r = P(G=3 | D=1, r)` and `q0_r = P(G=0 | D=0, r)` from type_probs; set
  `a_r = 1 / (q1_r + q0_r)` and `c_i = β(X_i)·(a_r·1[G_i=3] − a_r·1[G_i=0])`. Then
  `E[c | D=1, X] − E[c | D=0, X] = β(X)` exactly, so the conditional-on-observables contrast is
  `τ(X) + β(X)` by construction — the sampled bias surface IS the observational bias (the
  paper's design goal, achieved in closed form instead of the repo's calibration code).
- `y = X @ g + τ(X)·D + c + noise_sd·ε`, g ~ N(0, I₂) from rng.
- Dataset: treatment="D", outcome="y", forcing=["x0","x1"], covariates=["x0","x1"].

**Tests (seeded):**

- *Bias identity (the load-bearing test):* `constant_surfaces=(2.0, 3.0)`, n=200 000, one seed:
  within each region, `mean(y|D=1) − mean(y|D=0) − 2.0` is within 0.05 of 3.0 (three asserts,
  one per region).
- *GP-surface identity (weaker, binned):* smooth surfaces (ℓ=0.5), n=120 000; 6×6 interior
  cells with ≥ 200 obs per arm: mean over cells of |empirical contrast − (τ̄+β̄)(cell)| < 0.15
  (calibrate).
- *Overlap:* n=5000 ⇒ every region contains both D=0 and D=1 rows.
- *Discoverability:* n=3000, `lord3_scan(k=50, model="bernoulli")`: at least one top-10
  discovery has a center within 0.1 (raw units) of a corner boundary and |normal · axis| > 0.7
  for one axis (calibrate margins; this guards that the DGP feeds the phase-1 scan as intended).
- *Determinism & shapes:* same seed ⇒ bitwise-equal Dataset.df and DEETruth arrays;
  cate_query.shape == (grid²,); type frequencies within ±3σ of type_probs (n=20 000).
- *Contract:* rng=None raises ValueError; thresholds must satisfy 0 < b1 < b2 < 1.

Commit: `feat(data): scaled DEE simulation-1 DGP with exact conditional-bias construction`.

## Task 8 — DEE orchestrator (`dee_debias`) + package exports

**Create:** `src/natex/dee/debias.py`, `tests/test_dee_debias.py`.
**Modify:** `src/natex/dee/__init__.py`, `src/natex/__init__.py` (export `dee_debias`,
`DEEResult`, `voronoi_knn_repair`, `make_dee_synthetic` — additive).

```python
# dee/debias.py
@dataclass
class DEEResult:
    vknn: VKNNResult
    effects: list[EffectEstimate]        # aligned with vknn.experiments
    used: np.ndarray                     # bool over experiments: finite tau & se, post-filter
    obs_at_centers: np.ndarray           # cross-fitted obs-CATE at projected centers (audit 9)
    bias_obs: np.ndarray                 # obs_at_centers - tau_hat  (pinned sign convention)
    noise_var: np.ndarray                # smoothed (stage-1) noise variances actually used
    gp_bias: HeteroskedasticGP | None
    gp_direct: HeteroskedasticGP | None
    weights: ModelWeights
    query: np.ndarray                    # raw-unit query points as given
    cate_raw: np.ndarray                 # full-fit obs estimator at query
    cate_debiased: np.ndarray            # cate_raw - gp_bias.posterior(query_std).mean
    cate_direct: np.ndarray              # gp_direct posterior mean at query
    mixture: MixturePosterior | None     # over query points
    diagnostics: dict                    # dropped experiments + reasons, radii (experiment_radius),
                                         # m_prime, buffer, n_experiments_used, fold sizes

def dee_debias(
    dataset: Dataset,
    query: np.ndarray,                                   # (m, d) RAW forcing-space points
    discoveries: LoRD3Result | Sequence[Discovery],
    *,
    m_prime: int,
    k_prime: int = 200,
    t_side: int = 30,
    factory: Callable[[], ObservationalEstimator] | None = None,   # None -> default_factory(rng)
    weighting: str = "stacking",                          # "stacking" | "loo" | "mll"
    smooth_noise_stage1: bool = True,
    balance_alpha: float | None = 0.05,                   # None disables the balance filter
    n_folds: int = 5,
    rng: np.random.Generator | None = None,               # required
) -> DEEResult
```

Pipeline (each stage consumes the previous object — index sets are never recomputed, repo
risk 4): `voronoi_knn_repair` → optional `balance_filter` → `experiment_effects` (2sls) → drop
non-finite tau/se into diagnostics → `experiment_crossfit_cate` → `bias_obs` → noise: se²
(+`smooth_noise` with df = n_used − 4 when stage-1 on) → `HeteroskedasticGP.fit` for bias and
direct surfaces on projected centers → weights by strategy (stacking gets the cross-fitted
obs_at_centers) → query: standardize via `Dataset.standardize`; full-fit estimator for
`cate_raw`; model A posterior = shifted bias posterior (mean = cate_raw − bias mean, cov = bias
cov; the observational estimator's own uncertainty is not modeled — paper behavior, documented);
model B = direct posterior; `mixture_posterior(A, B, w_debias)`. Fewer than 3 usable
experiments ⇒ all GP-derived fields None/NaN with `diagnostics["reason"]` (never 0.0). Model
"A/B" naming and the sign convention follow the phase conventions block.

**Tests (seeded; uses `make_dee_synthetic`):**

- *End-to-end constant-bias recovery (the phase's core promise):*
  `make_dee_synthetic(n=3000, constant_surfaces=(2.0, 3.0), seed pinned)`; scan
  (`lord3_scan`, k=50, bernoulli), `m_prime=25`, `k_prime=250`, `t_side=15`, grid 15×15:
  `mean|cate_raw − 2| > 1.5` (raw is bias-dominated) AND `mean|cate_debiased − 2| < 0.75`
  AND `mean|mixture.mean − 2| < 0.75` (starting thresholds; calibrate ≥5 seeds, pin one).
- *Sign-convention regression:* in the same run, `mean(bias_obs[used])` within ±1.0 of +3.0
  (bias = obs − tau, positive when the observational estimator overshoots).
- *Discovery y-blindness:* rerun with y column permuted BUT identical discoveries/vknn inputs ⇒
  vknn/balance stages bitwise identical (members, group1, projected centers); only
  effects/GP outputs change.
- *Determinism:* same seed ⇒ identical `cate_debiased`, `weights.w_debias`, `mixture.mean`
  bitwise.
- *NaN policy:* poison two experiments' member outcomes to NaN ⇒ they appear in
  `diagnostics["dropped"]`, `used` excludes them, outputs remain finite; reduce to 2 usable
  experiments ⇒ GP fields None, arrays NaN, no exception.
- *Strategy switch:* `weighting="mll"` and `"loo"` run and give w ∈ [0,1]; unknown strategy
  raises ValueError.
- *Query contract:* query with wrong dimensionality raises ValueError; standardization
  consistency — passing `dataset.Z` as query hits identical predictions to passing the
  already-standardized points through a monkeypatched identity standardizer.

Commit: `feat(dee): end-to-end debiasing orchestrator with cross-fit bias observations and stacked mixture`.

## Task 9 — Benchmark harness + CI-small MSE gate

**Create:** `benchmarks/run_dee_sim.py`, `tests/test_dee_benchmarks_small.py`.
**Modify:** `benchmarks/README.md`.

```python
# benchmarks/run_dee_sim.py  (mirror run_did_curves.py conventions: argparse/typer, CSV out,
#                             optional matplotlib figure behind the plot extra)
def run_dee_replication(
    seed: int, n: int, cate_ls: float, bias_ls: float, *,
    k: int = 100, m_prime: int | None = 40, q_null: int = 0,   # q_null > 0 -> select_m_prime path
    k_prime: int = 400, t_side: int = 25, grid: int = 25, weighting: str = "stacking",
) -> dict
    # returns {seed, cate_ls, bias_ls, n_experiments, w_debias,
    #          mse_raw, mse_debiased, mse_direct, mse_mixture}  (grid MSE vs DEETruth.cate_query)

def run_dee_grid(seeds: Sequence[int], lengthscales: Sequence[float], n: int, **kw) -> pd.DataFrame
```

Full run (paper analog, scaled): 20 seeds × ℓ ∈ {0.2, 0.5}² at n=4000 — reproduces the
qualitative sim-1 claim (debiased/mixture beats raw CF in median grid MSE). The `q_null` option
exercises `select_m_prime` end-to-end (randomization test with the shared-geometry path from
phase 2). CSV columns exactly the dict keys above; figure = per-config MSE box plot.

**Tests (`tests/test_dee_benchmarks_small.py`, CI-small, no backtest marker, target < 90 s):**

- *MSE gate (the spec §8 phase gate, scaled):* 2 pinned seeds, n=1500, k=50, m_prime=25,
  k_prime=250, t_side=15, grid=15, ℓ_cate=ℓ_bias=0.5:
  `median(mse_mixture) < median(mse_raw)` across the two seeds AND
  `mse_debiased < mse_raw` for both seeds individually (calibrate on ≥5 seeds first; pin the
  two most-typical, comment the calibration table).
- *Sanity of magnitudes:* `mse_direct` finite and `n_experiments >= 3` for both seeds;
  `w_debias ∈ [0, 1]`.
- *select_m_prime path:* one seed with `q_null=9`: returned m_prime > 0 on this
  strong-signal DGP and the replication completes with finite MSEs.
- *Determinism:* rerunning one replication yields the identical row (bitwise on floats).

Commit: `feat(benchmarks): scaled DEE simulation-1 harness; CI-small debiased-vs-raw MSE gate`.

## Task 10 — Optional extras (econml, gpytorch), CLI, method card, status, final gate

**Create:** `src/natex/dee/forest.py`, `src/natex/dee/gp_torch.py`, `tests/test_dee_optional.py`,
`docs/method_cards/dee.md`, `docs/status/phase-4.md`.
**Modify:** `pyproject.toml`, `src/natex/cli.py`, `tests/test_cli.py`, `README.md`.

1. **pyproject extras** (additive): `ml = ["econml>=0.15"]`, `gp = ["torch>=2.2",
   "gpytorch>=1.13"]`. CI workflow unchanged (extras not installed ⇒ optional tests skip).
2. **`dee/forest.py`**: `CausalForestEstimator(seed, n_estimators=1000, **cf_kwargs)`
   implementing `ObservationalEstimator` via `econml.dml.CausalForestDML` (import inside
   `fit`; `ImportError` message names `pip install "natex-discovery[ml]"`). Works for binary and
   continuous T (fills the TLearner gap).
3. **`dee/gp_torch.py`**: `TorchHeteroskedasticGP` with the same `fit/posterior/
   log_marginal_likelihood` surface (FixedNoiseGaussianLikelihood + RBF, fit via
   `fit_gpytorch_mll` — NOT the removed `fit_gpytorch_model`, repo risk 9), seeds torch from the
   numpy rng. Documented as the scale backend; numpy remains the default everywhere.
4. **`tests/test_dee_optional.py`**: `pytest.importorskip("econml")` /
   `pytest.importorskip("gpytorch")` at the top of each test class.
   - econml: protocol conformance + recovery test from task 6 with threshold |·−2| < 0.4;
     plugging the factory into `dee_debias` on the task-8 constant-bias scenario keeps
     `mean|cate_debiased − 2| < 0.75`.
   - gpytorch: posterior-mean parity with `HeteroskedasticGP` at FIXED shared hyperparameters on
     a 10-point 1-D problem (atol 1e-3 — both are exact GPs, so agreement is analytic, not
     statistical); determinism given seed.
5. **CLI `natex debias`** (new typer command): `natex debias CSV --treatment T [--outcome y]
   [--forcing x0,x1] --k 50 --degree 1 --m-prime 25 [--q-null 0] --k-prime 250 --t-side 15
   --grid 15 --weighting stacking --seed 0 --out out/` → runs scan → `dee_debias` → writes
   `dee_result.json` via the existing `_clean` helper (weights, per-experiment effects table,
   grid predictions raw/debiased/direct/mixture mean+sd, diagnostics). `--q-null > 0` routes
   m_prime through `select_m_prime`. Test in `tests/test_cli.py`: smoke on a tiny generated CSV
   (n=500 constant-surfaces DGP written to tmp_path): exit code 0, JSON parses, keys present,
   grid arrays finite, `weights.w_debias ∈ [0,1]`.
6. **`docs/method_cards/dee.md`**: pipeline diagram; corrected Alg 1 (both code fixes); pinned
   sign convention; audit items 7 (√2-corrected E[q²] ≤ 2ρ² bound + fixed-shape caveat — and
   that natex only reports radii, never the printed bound), 8 (mixture covariance + label per
   draw), 9 (cross-fitting; SE² treated as estimated via the chi-square stage-1 model, HC1 not
   classical); stacking-replaces-softmax-PLP decision and the dropped unseeded 1-MC strategy;
   CF-features = Z_std convention; deviations from the paper's repo (scaled N/k/M′, closed-form
   bias construction in the DGP, no Cattaneo/Kallus R benchmarks, rural roads deferred —
   candidate future backtest per phase-3 follow-ups).
7. **README**: roadmap tick for phase 4 + a 5-line `dee_debias` quickstart snippet.
8. **Final gate (record in `docs/status/phase-4.md`):**
   - `uv run ruff check src tests` clean;
   - `uv run pytest -q` all green (unit + CI-small, no backtests);
   - `uv sync --extra ml --extra gp && uv run pytest -q tests/test_dee_optional.py` green
     locally (documented as local-only if a 3.14 wheel gap forces it);
   - `NATEX_DATA=".../RDD/data" uv run pytest -q -m backtest` — all phase-2/3 rows still green
     (no regressions);
   - one full `benchmarks/run_dee_sim.py` run (20 seeds × ℓ grid, n=4000) with the median-MSE
     table pasted into the status doc (run of record for the spec §8 gate);
   - deviations log (investigated, not silently widened) in the phase-3 status style.

Commit(s): `feat(dee): econml causal-forest adapter and gpytorch backend as optional extras`,
`feat(cli): natex debias command`, `docs: DEE method card, README, phase-4 status`.

---

## Task ordering and dependencies

1 → 2 (effects need QuasiExperiment) → 3 (GP standalone) → 4 (noise needs GP) → 5 (bma needs
GP) → 6 (observational standalone, needs VKNN types) → 7 (DGP needs `sample_gp_prior` from 3)
→ 8 (orchestrator needs 1–7) → 9 (benchmark needs 8) → 10 (extras/CLI/docs need everything).
Tasks 3–7 are mutually independent apart from the noted imports and can be reordered if
convenient, but the listed order keeps every commit green.
